# test/MDocRAG.py
import os
import json
import torch
import numpy as np
import logging
from tqdm import tqdm
from io import BytesIO
from pdf2image import convert_from_path
import base64
from difflib import SequenceMatcher
import PyPDF2
import pytesseract
import bitsandbytes     # Added
from transformers import BitsAndBytesConfig
import traceback
import pickle
from typing import List, Dict, Any

# Optional imports based on selected models
try:
    import google.generativeai as genai
except ImportError:
    pass

# For embeddings and retrieval
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# poppler_path = r'C:\Users\Daryn Bang\PycharmProjects\poppler-24.08.0\Library\bin'

import sys
sys.stdout.reconfigure(encoding='utf-8')

# === Decorator for method entry/exit logging ===
def log_entry_exit(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("MDocRAG")
        cls_name = args[0].__class__.__name__ if args else ''
        logger.info(f"[ENTRY] {cls_name}.{func.__name__}()")
        try:
            result = func(*args, **kwargs)
            logger.info(f"[EXIT] {cls_name}.{func.__name__}() completed successfully")
            return result
        except Exception as e:
            logger.exception(f"[ERROR] {cls_name}.{func.__name__}() raised an exception: {e}")
            raise
    return wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("visdomrag.log"), logging.StreamHandler()]
)

logger = logging.getLogger("MDocRAG")
torch.set_default_dtype(torch.bfloat16)

class MDocRAG:
    """
    Multi-modal Document Retrieval-Augmented Generation (MDocRAG) system.
    Supports both text and image (visual) retrieval from PDF documents using:
    - Sentence Transformers for text embeddings (e.g. MiniLM, MPNet)
    - ColPali / ColQwen for visual (page image) embeddings
    """
    def __init__(self, config):
        """
        Initialize the MDocRAG pipeline for Indexing and Retrieval.
        Load configs and initialize visual/textual retriever models.
        Prepare empty cache/index containers for later use.

        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.data_dir = config["data_dir"]
        self.vision_retriever = config["vision_retriever"]
        self.text_retriever = config["text_retriever"]
        self.top_k = config.get("top_k", 3)
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 300)
        self.force_reindex = config.get("force_reindex", False)

        # Initialize document cache
        self.document_cache = {}

        # Visual pipeline
        self.page_embeddings = None
        self.document_page_map = None

        # Textual pipeline
        self.text_index_obj = None
        self.text_chunks = None
        self.text_chunk_mapping = None

        # Initialize retrieval resources
        self._initialize_retrieval_resources()

    @log_entry_exit
    def _initialize_retrieval_resources(self):
        """
        Initialize resources needed for retrieval.
        - Load the correct embedding model for visual (ColQwen / ColPali) and text (MiniLM / MPNet / BGE).
        - Sets up quantization (8-bit) using bitsandbytes for memory-efficient loading.
        """
        if self.vision_retriever in ["colpali", "colqwen"]:
            # --- old code work well ---
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     # It's generally good to set compute_dtype to bfloat16 if your GPU supports it,
            #     # as ColPali was initially loaded with torch_dtype=torch.bfloat16.
            #     bnb_8bit_compute_dtype=torch.bfloat16
            # )
            
            # --- new code ---
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            if self.vision_retriever == "colpali":
                try:
                    from colpali_engine.models import ColPali, ColPaliProcessor
                    logger.info("Loading ColPali model for visual indexing")
                    self.vision_model = ColPali.from_pretrained(
                        "vidore/colpali-v1.2",
                        quantization_config=bnb_config, # comment this if would like to use full
                        torch_dtype=torch.bfloat16,
                        device_map="cuda"
                    ).eval()
                    self.vision_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2", use_fast=True)
                except ImportError:
                    raise ImportError("ColPali models not found. Please install colpali_engine.")
            elif self.vision_retriever == "colqwen":
                try:
                    from colpali_engine.models import ColQwen2, ColQwen2Processor
                    logger.info("Loading ColQwen model for visual indexing")
                    self.vision_model = ColQwen2.from_pretrained(
                        "vidore/colqwen2-v0.1",
                        torch_dtype=torch.bfloat16, 
                        quantization_config=bnb_config, # comment this if would like to use full
                        device_map="cuda"
                    ).eval()
                    self.vision_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1", use_fast=True)
                except ImportError:
                    raise ImportError("ColPali/ColQwen models not found. Please install colpali_engine.")
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

        if self.text_retriever in ["minilm", "mpnet", "bge"]:
            # Map text_retriever to actual model names
            model_map = {
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "bge": "BAAI/bge-small-en-v1.5"
            }

            # Load sentence transformer model
            self.text_model_name = model_map[self.text_retriever]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.text_model_name, device=self.device
            )

            print(f'{self.st_embedding_function} - {self.st_embedding_function.device}')
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")

    @log_entry_exit
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using OCR if needed.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            list: List of text from each page
        """
        try:
            # First try regular PDF extraction
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)
                pages = [page.extract_text() for page in reader.pages]

            # If any page has no text, use OCR
            if any(not page.strip() for page in pages):
                logger.info(f"Using OCR for {pdf_path} as some pages have no text")
                pages = []
                pdf_images = convert_from_path(pdf_path)
                for page_num, page_img in enumerate(pdf_images):
                    text = pytesseract.image_to_string(page_img)
                    pages.append(f"--- Page {page_num + 1} ---\n{text}\n")

            return pages
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            traceback.print_exc()
            return []

    @log_entry_exit
    def split_text(self, text):
        """
        Split long text into manageable chunks with overlap, useful for embedding/indexing.

        Args:
            text (str): Text to split

        Returns:
            list: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_text(text)

    @log_entry_exit
    def cache_documents(self):
        """
        Cache all PDF texts in the dataset for later reference and mapping text chunks to documents/pages.

        Returns:
            dict: Dictionary mapping document IDs to text content
        """
        logger.info("Caching document content")

        try:
            # Extract unique document IDs from the dataset
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    docs = eval(row['documents']) if 'documents' in row else []
                    unique_docs.update(docs)
                except:
                    # Handle case where 'documents' field is not a valid list
                    traceback.print_exc()
                    pass

                if 'doc_path' in row:
                    doc_path = row['doc_path']
                    if isinstance(doc_path, str) and doc_path.strip():
                        unique_docs.add(os.path.basename(doc_path).split('.')[0])

            # Cache content for each document
            cache = {}
            pdf_dir = os.path.join(self.data_dir, "docs")

            for doc_id in tqdm(unique_docs, desc="Caching documents"):
                # Try different possible filename formats
                logger.info(f"Caching: {doc_id}")
                
                possible_paths = [
                    os.path.join(pdf_dir, doc_id),
                    os.path.join(pdf_dir, f"{doc_id}.pdf"),
                    os.path.join(pdf_dir, f"{doc_id.ljust(10, '0')}.pdf"),
                    os.path.join(pdf_dir, f"{doc_id.split('_')[0]}.pdf")
                ]

                for pdf_path in possible_paths:
                    if os.path.exists(pdf_path):
                        cache[doc_id] = self.extract_text_from_pdf(pdf_path)
                        break
                else:
                    logger.warning(f"No PDF file found for document {doc_id}")

            self.document_cache = cache
            logger.info(f"Cached content for {len(cache)} documents")
            return cache

        except Exception as e:
            logger.error(f"Error caching documents: {str(e)}")
            traceback.print_exc()
            return {}

    @log_entry_exit
    def identify_document_and_page(self, chunk):
        """
        Try to match a chunk of text back to its original document and page using fuzzy matching.

        Args:
            chunk (str): Text chunk

        Returns:
            tuple: (arxiv_id, page_num)
        """
        max_ratio = 0
        best_match = (None, None)

        for arxiv_id, pages in self.document_cache.items():
            for page_num, page_text in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page_text).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = (arxiv_id, page_num)

        return best_match

    @log_entry_exit
    def build_visual_index(self, document_paths, output_subdir):
        """
        Build a FAISS-backed visual embedding index for inference.
        - Convert each PDF page to an image.
        - Run visual encoder (ColQwen or ColPali) to extract image embeddings.
        - Save the embeddings and map them to document/page metadata.

        Args:
            document_paths (List[str]): List of PDF paths (absolute or relative to data_dir/docs).
            output_subdir (str): Subdirectory under self.data_dir to store the FAISS index & mapping.

        Returns:
            page_embeddings: { page_id: torch.Tensor } (for backwards compatibility)
            document_page_map: { page_id: {"doc_id": str, "page_idx": int} }
        """
        logger.info(f'Building visual index for inference using {self.vision_retriever}')
        
        # --- old code --- 
        base_pdf_dir = os.path.join(self.data_dir, "docs")
        
        # --- new code ---
        # base_pdf_dir = self.data_dir
        
        # --- old code ---
        # output_dir = os.path.join(self.data_dir, output_subdir)
        
        # --- new code ---
        # output_dir = os.path.normpath(os.path.join(self.data_dir, output_subdir))
        # Remove accidental base folder duplication
        if output_subdir.startswith(self.data_dir):
            output_subdir = os.path.relpath(output_subdir, start=self.data_dir)

        output_dir = os.path.join(self.data_dir, output_subdir)

        emb_dir = os.path.join(output_dir, "embeddings")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)

        print(self.vision_model.device)

        page_embeddings = {}
        document_page_map = {}
        
        # Track N for each page
        # - each vector has [1, N, D]
        # - check the N of all the images and make sure only keeps vectors with the same N
        embedding_n_map = {}  

        for pdf_file in tqdm(document_paths, desc="Processing PDFs for visual index"):
            fname = pdf_file if pdf_file.lower().endswith(".pdf") else f"{pdf_file}.pdf"
            
            # --- old code ---
            # pdf_path = fname if os.path.isabs(fname) else os.path.join(base_pdf_dir, fname)
            
            # --- new code ---
            pdf_path = fname 
            doc_id = os.path.splitext(os.path.basename(fname))[0]
            print(f'{pdf_path} - {doc_id}')

            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue

            try:
                pages = convert_from_path(pdf_path)
            except Exception as e:
                logger.error(f"Error converting PDF {pdf_file} to images: {e}")
                traceback.print_exc()
                continue
            
            logger.info(f"Processing {pdf_path} with {len(pages)} pages")
            
            for page_idx, page_img in enumerate(tqdm(pages, desc=f"{doc_id} pages")):
                page_id = f"{doc_id}_{page_idx}"
                document_page_map[page_id] = {"doc_id": doc_id, "page_idx": page_idx}
                
                logger.info(f"Embedding visual page {page_id}")
                
                pt_path = os.path.join(emb_dir, f"{page_id}.pt")
                if os.path.exists(pt_path) and not self.config.get("force_reindex", False):
                    logger.info(f"Skipping {page_id}, already embedded")
                    embedding = torch.load(pt_path)  # still load it
                    page_embeddings[page_id] = embedding
                    embedding_n_map[page_id] = embedding.shape[1] if embedding.ndim == 3 else 1
                    continue

                # Process and embed the image
                try:
                    if self.vision_retriever in ["colpali", "colqwen"]:
                        # Process the image into model-compatible input
                        processed_image = self.vision_processor.process_images([page_img])
                        processed_image = {k: v.to(self.vision_model.device) for k, v in processed_image.items()}

                        # Generate embedding
                        with torch.no_grad():
                            embedding = self.vision_model(**processed_image).to(torch.bfloat16)
                            embedding = embedding.cpu()
                            
                        # --- new code ---
                        if embedding.ndim == 3:
                            n_val = embedding.shape[1]
                        else:
                            n_val = 1  
                        # --- end new code ---

                        # store raw tensor
                        pt_path = os.path.join(emb_dir, f"{page_id}.pt")
                        torch.save(embedding, pt_path)
                        page_embeddings[page_id] = embedding
                        
                        # --- new code ---
                        embedding_n_map[page_id] = n_val 
                        # --- end new code ---

                        print(f"Finished processing {page_id} for {pdf_path}")
                        print(embedding.dtype, embedding.shape)

                except Exception as e:
                    logger.error(f"Error embedding page {page_id} of {pdf_file}: {e}")
                    traceback.print_exc()
                    continue

        # persist document_page_map
        docmap_path = os.path.join(output_dir, "document_page_map.json")
        with open(docmap_path, "w", encoding="utf-8") as f:
            json.dump(document_page_map, f, indent=2)

        logger.info(f"Saved document_page_map.json ({len(document_page_map)} pages)")
        return page_embeddings, document_page_map

    @log_entry_exit
    def build_text_index(self, document_paths, output_subdir):
        """ 
        Extract text from PDF, split into chunks.
        Embed chunks using SentenceTransformer and store them in a ChromaDB collection.
        """
        logger.info(f"Building text index for inference")
        
        # --- old code ---
        # output_dir = os.path.join(self.data_dir, output_subdir)
        
        # --- new code ---
        # output_dir = os.path.normpath(os.path.join(self.data_dir, output_subdir))
        # Remove accidental base folder duplication
        if output_subdir.startswith(self.data_dir):
            output_subdir = os.path.relpath(output_subdir, start=self.data_dir)

        output_dir = os.path.join(self.data_dir, output_subdir)

        os.makedirs(output_dir, exist_ok=True)

        # Extract and split text
        all_chunks: List[str] = []
        chunk_to_doc_mapping: List[Dict[str, Any]] = []

        # --- old code --- 
        base_pdf_dir = os.path.join(self.data_dir, "docs")
        
        # --- new code ---
        # base_pdf_dir = self.data_dir
        
        for pdf_file in tqdm(document_paths, desc="Reading & Splitting documents"):
            # Normalize file names
            file_name = pdf_file if pdf_file.lower().endswith(".pdf") else f"{pdf_file}.pdf"
            
            # --- old code ---
            # pdf_path = file_name if os.path.isabs(file_name) else os.path.join(base_pdf_dir, file_name)
            
            # --- new code ---
            pdf_path = file_name 
            
            logger.info(f"Processing {pdf_path}")
            
            doc_id = os.path.splitext(os.path.basename(file_name))[0]

            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found, skipping: {pdf_path}")
                continue

            try:
                pages: List[str] = self.extract_text_from_pdf(pdf_path)
            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_path}: {e}")
                traceback.print_exc()
                continue

            # Join pages then split into chunks
            full_text = "\n".join(pages)
            chunks = self.split_text(full_text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                arxiv_id, page_num = self.identify_document_and_page(chunk)
                chunk_to_doc_mapping.append({
                    "chunk_pdf_name": arxiv_id or doc_id,
                    "pdf_page_number": page_num if page_num is not None else i
                })

        # Persist raw chunks + mapping as JSON
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8", errors="replace") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "chunk_mapping.json"), "w", encoding="utf-8", errors="replace") as f:
            json.dump(chunk_to_doc_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(all_chunks)} chunks + mapping to {output_dir}")

        # Build index
        if self.text_retriever in {"minilm", "mpnet", "bge"}:
            # Chroma embedding index
            chroma_client = chromadb.PersistentClient(path="./chroma_db")  # persisted client
            # col_name = f"text_col_{uuid.uuid4().hex[:8]}"
            col_name = f"mdocagent_text_index"
            
            # --- old code --- 
            # collection = chroma_client.create_collection(
            #     name=col_name,
            #     embedding_function=self.st_embedding_function,
            #     metadata={"hnsw:space": "cosine"}
            # )
            # --- end old code ---
            
            # --- new code ---
            if self.config.get("force_reindex", False):
                # === Force reindex: delete then create ===
                try:
                    chroma_client.delete_collection(name=col_name)
                    logger.info(f"Deleted existing Chroma collection '{col_name}' due to force_reindex=True")
                except chromadb.errors.NotFoundError:
                    logger.info(f"Collection '{col_name}' not found; nothing to delete.")

                collection = chroma_client.create_collection(
                    name=col_name,
                    embedding_function=self.st_embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new Chroma collection '{col_name}'")
            else:
                # === Normal mode: try to load existing ===
                try:
                    collection = chroma_client.get_collection(
                        name=col_name,
                        embedding_function=self.st_embedding_function
                    )
                    logger.info(f"Loaded existing Chroma collection '{col_name}'")
                except chromadb.errors.NotFoundError:
                    # If not found, create it
                    collection = chroma_client.create_collection(
                        name=col_name,
                        embedding_function=self.st_embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new Chroma collection '{col_name}' (force_reindex=False but collection was missing)")
            # --- end new code ---

            ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            collection.add(documents=all_chunks, ids=ids)
            logger.info(f"Created Chroma collection '{col_name}' with {len(all_chunks)} chunks")
            return collection, all_chunks, chunk_to_doc_mapping

        else:
            raise ValueError(f"Unknown text_retriever: {self.text_retriever}")

    @log_entry_exit
    def retrieve_visual_contexts(self, query, document_page_map, top_k, index_subdir):
        """
        Retrieve the top-K visual contexts by querying the on disk FAISS index.
        - Embed the query and compare it to visual page embeddings using cosine similarity.
        - Return top-K matching page images and metadata.

        Args:
            query (str): The natural-language query.
            document_page_map (dict): { page_id: {"doc_id": str, "page_idx": int} }.
            top_k (int, optional): How many pages to return (defaults to self.top_k).
            index_subdir (str): Subdirectory where visual index & page_id_map.pkl live.

        Returns:
            List[Dict]: Each dict has:
            - 'image': PIL.Image of the page
            - 'document_id': page_id (e.g. "doc123_0")
            - 'page_number': int
            - 'score': float similarity score
        """
        top_k = top_k or self.top_k
        emb_dir = os.path.join(index_subdir, "embeddings")
        pdf_dir = self.data_dir

        contexts: List[Dict[str, Any]] = []

        try:
            # Load or cache all page embeddings
            if not hasattr(self, "page_embeddings") or not self.page_embeddings:
                self.page_embeddings = {}
                for fname in tqdm(os.listdir(emb_dir), desc="Loading visual embeddings"):
                    if not fname.endswith(".pt"):
                        continue
                    page_id = fname[:-3]
                    try:
                        emb = torch.load(os.path.join(emb_dir, fname))  # shape [K, d] or [1, K, d]
                        self.page_embeddings[page_id] = emb
                    except Exception as e:
                        logger.error(f"Failed to load embedding for {page_id}: {e}")

            # Load or override document_page_map
            doc_map_path = os.path.join(index_subdir, "document_page_map.json")
            if document_page_map is None:
                if not os.path.exists(doc_map_path):
                    logger.error("Missing document_page_map.json; run ingestion first.")
                    return []
                with open(doc_map_path, "r", encoding="utf-8") as f:
                    document_page_map = json.load(f)
                    print("Loaded Document Page Map")

            # Embed the query
            processed_query = self.vision_processor.process_queries([query])
            processed_query = {k: v.to(self.vision_model.device) for k, v in processed_query.items()}

            with torch.no_grad():
                query_embedding = self.vision_model(**processed_query).to(torch.bfloat16).cpu()

            # --- old code ---
            # # Prepare document embeddings tensor
            # page_ids = list(self.page_embeddings.keys())
            # doc_embeddings = torch.cat([self.page_embeddings[pid] for pid in page_ids], dim=0)  # [N, K, d] or [N, d]

            # # Score all pages with multi-vector scorer
            # with torch.no_grad():
            #     scores = self.vision_processor.score_multi_vector(query_embedding, doc_embeddings)
            #     scores = scores.flatten().numpy()  # [N]
                
            # # Select top-K
            # top_idxs = np.argsort(-scores)[:top_k]
            
            # --- end old code ---
            # --- new code ---
            scores_dict = {}  # page_id → score

            # Group by N
            from collections import defaultdict
            grouped = defaultdict(list)
            for pid, emb in self.page_embeddings.items():
                n_val = emb.shape[1] if emb.ndim == 3 else 1
                grouped[n_val].append((pid, emb))

            # Score for each group
            with torch.no_grad():
                for n_val, entries in grouped.items():
                    group_pids = [pid for pid, _ in entries]
                    group_embs = torch.cat([emb for _, emb in entries], dim=0)  # shape [N, n_val, D]
                    score_tensor = self.vision_processor.score_multi_vector(query_embedding, group_embs)
                    for pid, score in zip(group_pids, score_tensor.flatten().tolist()):
                        scores_dict[pid] = score

            # Sort and select top-K
            top_items = sorted(scores_dict.items(), key=lambda x: -x[1])[:top_k]

        except Exception as e:
            logger.error(f"Error in retrieve_visual_contexts_inference: {e}")
            traceback.print_exc()
            return []

        # --- old code ---
        # Load corresponding PDF pages
        # for idx in top_idxs:
        #     pid = page_ids[idx]
        #     score = float(scores[idx])
        #     info = document_page_map.get(pid, {})
        #     doc, pg = info.get("doc_id"), info.get("page_idx")
        #     pdf_path = os.path.join(pdf_dir, f"{doc}.pdf")
        #     if not os.path.exists(pdf_path):
        #         continue
        #     try:
        #         pages = convert_from_path(pdf_path)
        #         if 0 <= pg < len(pages):
        #             contexts.append({
        #                 "image": pages[pg],
        #                 "document_id": pid,
        #                 "page_number": pg,
        #                 "score": score
        #             })
        #     except Exception as e:
        #         logger.error(f"Error loading page {pg} of {doc}: {e}")
        #         continue
        # --- end old code ---
        # --- new code ---
        for pid, score in top_items:
            info = document_page_map.get(pid, {})
            doc, pg = info.get("doc_id"), info.get("page_idx")
            pdf_path = os.path.join(pdf_dir, f"{doc}.pdf")
            if not os.path.exists(pdf_path):
                continue
            try:
                pages = convert_from_path(pdf_path)
                if 0 <= pg < len(pages):
                    contexts.append({
                        "image": pages[pg],
                        "document_id": pid,
                        "page_number": pg,
                        "score": float(score)
                    })
            except Exception as e:
                logger.error(f"Error loading page {pg} of {doc}: {e}")
                continue
        # --- end new code ---
            
        print(f"Visual Contexts: {contexts}")

        logger.info(f"Retrieved {len(contexts)} visual contexts for query.")
        return contexts

    @log_entry_exit
    def retrieve_textual_contexts(self, query, index_obj, all_chunks, chunk_to_doc_mapping, top_k, index_subdir):
        """
        Retrieve the top-K textual contexts for a single query.
        - Embed the query and compare it to text chunks via ChromaDB.
        - Return top-K matching text chunks and their original document/page info.

        Args:
            query (str): Natural-language query.
            index_obj: Either a BM25Okapi instance or a Chroma Collection.
            all_chunks (List[str]): List of text chunks indexed.
            chunk_to_doc_mapping (List[Dict]): Metadata per chunk.
            top_k (int, optional): Number of contexts to return (defaults to self.top_k).
            index_subdir(str)

        Returns:
            List[Dict]: Each dict contains:
              - 'chunk': the text chunk
              - 'chunk_pdf_name': originating document ID
              - 'pdf_page_number': originating page index
              - 'rank': 1-based rank
              - 'score': similarity score (1-distance)
        """
        logger.info(f"Querying Chroma index with: {query}")
        
        top_k = top_k or self.top_k
        contexts: List[Dict[str, Any]] = []

        # Load raw data
        if all_chunks is None or chunk_to_doc_mapping is None:
            with open(os.path.join(index_subdir, "chunks.json"), "r", encoding="utf-8") as f:
                all_chunks = json.load(f)

            with open(os.path.join(index_subdir, "chunk_mapping.json"), "r", encoding="utf-8") as f:
                chunk_to_doc_mapping = json.load(f)
            logger.info(f"Loaded {len(all_chunks)} chunks + mapping from {index_subdir}")

        # Load index object
        if index_obj is None:
            col_name = "mdocagent_text_index"
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            # Load the existing collection
            index_obj = chroma_client.get_collection(
                name="mdocagent_text_index",
                embedding_function=self.st_embedding_function
            )
            logger.info(f"Loaded Chroma collection '{col_name}'")

        try:
            if self.text_retriever in {"minilm", "mpnet", "bge"}:
                # Query Chroma for nearest neighbors
                result = index_obj.query(query_texts=[query], n_results=top_k)
                ids = result["ids"][0]
                distances = result["distances"][0]  # Cosine distance

                for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
                    logger.info(f"[Text Rank {rank}] Chunk ID: {cid} | Similarity: {1 - dist:.4f}")
                    
                    idx = int(cid.split("_")[1])
                    
                    contexts.append({
                        "chunk": all_chunks[idx],
                        "chunk_pdf_name": chunk_to_doc_mapping[idx]["chunk_pdf_name"],
                        "pdf_page_number": chunk_to_doc_mapping[idx]["pdf_page_number"],
                        "rank": rank,
                        # convert distance → similarity
                        "score": float(1.0 - dist)
                    })

            else:
                raise ValueError(f"Unknown text_retriever: {self.text_retriever}")

            print(f"Textual Contexts: {contexts}")

            logger.info(f"Retrieved {len(contexts)} textual contexts for query.")
            return contexts

        except Exception as e:
            logger.error(f"Error in retrieve_textual_contexts_inference: {e}")
            traceback.print_exc()
            return []

    @log_entry_exit
    def ingestion(self, index_dir: str, pdf_dir: str = None):
        """
        Ingest all PDFs in `pdf_dir`, building both visual and textual indexes
        for later inference.
        Main function to run:
        - Visual index builder (page images → visual embeddings)
        - Text index builder (text chunks → text embeddings)

        Args:
            pdf_dir (str): Path to a directory containing .pdf files.
            index_dir (str):
        """
        # Prepare output subdirectories
        if pdf_dir is None:
            pdf_dir = self.data_dir

        vis_subdir = os.path.join(index_dir, "visual")
        txt_subdir = os.path.join(index_dir, "textual")
        os.makedirs(vis_subdir, exist_ok=True)
        os.makedirs(txt_subdir, exist_ok=True)

        # Gather all PDF paths
        pdf_paths = [
            os.path.join(pdf_dir, fn)
            for fn in os.listdir(pdf_dir)
            if fn.lower().endswith(".pdf")
        ]

        logger.info(f"Found {len(pdf_paths)} PDF files for ingestion:")
        for path in pdf_paths:
            logger.info(f"Found PDF: {path}")

        if not pdf_paths:
            logger.warning(f"No PDFs found in {pdf_dir}; nothing to ingest.")
            return

        # Build (or reload) visual index
        #    - Stores: page_embeddings, page_id_map.pkl, document_page_map.json
        # logger.info(f"Ingesting visual index into {vis_subdir}")
        self.page_embeddings, self.document_page_map = self.build_visual_index(
            document_paths=pdf_paths,
            output_subdir=vis_subdir
        )

        # Build (or reload) text index
        #    - Stores under txt_subdir:
        #       • chunks.json
        #       • chunk_mapping.json
        #       • Chroma collection (if embedding‐based)
        logger.info(f"Ingesting text index into {txt_subdir}")
        (
            self.text_index_obj,
            self.text_chunks,
            self.text_chunk_mapping
        ) = self.build_text_index(
            document_paths=pdf_paths,
            output_subdir=txt_subdir
        )

        logger.info(
            f"Ingestion complete:  "
            f"{len(self.page_embeddings)} pages indexed visually,  "
            f"{len(self.text_chunks)} text chunks indexed."
        )

    @log_entry_exit
    def encode_image(self, pil_image):
        """
        Convert a PIL image to base64 format (e.g., for API or frontend display).
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

# Configuration
config = {
    "data_dir": "data/pdf/",
    "vision_retriever": "colpali",       # or 'colpali' or 'colqwen
    "text_retriever": "minilm",          # or 'mpnet', 'bge'
    "top_k": 10,
    "chunk_size": 3000,
    "chunk_overlap": 300,
    # "force_reindex": False
}

# Initialize the class
mdoc = MDocRAG(config)

# Example usage
# Run full ingestion: build both visual and text indexes
index_dir = "data/pdf/index"
pdf_dir = "data/pdf/"   # folder containing PDF files

mdoc.ingestion(index_dir=index_dir, pdf_dir=pdf_dir)

print("\n🔍 Multi-modal Document Chat is ready!")
print("Type your query below (or type 'exit' to quit):\n")

while True:
    try:
        query = input("❓ Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Session ended.")
            break

        # === Retrieve Textual Contexts ===
        print("\n📄 Top Textual Chunks:")
        contexts = mdoc.retrieve_textual_contexts(
            query=query,
            index_obj=mdoc.text_index_obj,
            all_chunks=mdoc.text_chunks,
            chunk_to_doc_mapping=mdoc.text_chunk_mapping,
            top_k=mdoc.config["top_k"],
            index_subdir=os.path.join("data/pdf/index", "textual")
        )

        for ctx in contexts:
            print(f"📘 Doc: {ctx['chunk_pdf_name']} | Page: {ctx['pdf_page_number']} | Score: {ctx['score']:.2f}")
            print(ctx['chunk'].strip(), "\n---\n")

        # === Retrieve Visual Contexts ===
        print("🖼️ Top Visual Pages:")
        visual_contexts = mdoc.retrieve_visual_contexts(
            query=query,
            document_page_map=mdoc.document_page_map,
            top_k=mdoc.config["top_k"],
            index_subdir=os.path.join("data/pdf/index", "visual")
        )

        for vis in visual_contexts:
            print(f"🧾 Document: {vis['document_id']} | Page: {vis['page_number']} | Score: {vis['score']:.2f}")
            # vis['image'].show() # <- this line of code would pop up the image in a window

    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Bye!")
        break
    except Exception as e:
        print(f"❌ Error during query: {e}")
    
    
    
    