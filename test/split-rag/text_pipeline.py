"""
TextRAGPipeline: Text-based RAG (Retrieval-Augmented Generation) pipeline for PDF documents.

This module supports:
- Text extraction from PDFs (including OCR fallback for scanned documents).
- Text chunking with overlap for better embedding performance.
- Embedding and indexing via Sentence Transformers and ChromaDB.
- Query-time retrieval of top-K similar chunks for downstream tasks (e.g., RAG).

Author: PDZ, Bang
"""
# text_pipeline.py
import os, json, traceback
from tqdm import tqdm
from typing import List, Dict, Any
import logging
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from difflib import SequenceMatcher
import torch
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import chromadb.utils.embedding_functions as embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TextRAG")

class TextRAGPipeline:
    """
    A pipeline for building and querying a text-based retrieval index using ChromaDB.

    Attributes:
        config (dict): Configuration dictionary containing parameters like chunk size, model, paths, etc.
        chunk_size (int): Maximum length of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks to maintain context.
        top_k (int): Number of top retrieved chunks to return per query.
        data_dir (str): Directory where PDF files and outputs are stored.
        device (str): Computation device ('cuda' or 'cpu').
        force_reindex (bool): Whether to rebuild the entire index even if it exists.
        text_retriever (str): Key for selecting sentence embedding model.
        embedding_function: Callable embedding model used by ChromaDB.
    """
    def __init__(self, config):
        self.config = config
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.top_k = config["top_k"]
        self.data_dir = config["data_dir"]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.force_reindex = config.get("force_reindex", False)

        # Map text_retriever to actual model names
        self.text_retriever = config["text_retriever"]
        
        self.model_map = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "bge": "BAAI/bge-small-en-v1.5"
        }

        if self.text_retriever not in self.model_map:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")

        # Load sentence transformer model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_map[self.text_retriever],
            device=self.device
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extracts text from a PDF file using PyPDF2, falling back to OCR via pytesseract if needed.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[str]: A list of strings, one per page of the PDF.
        """
        try:
            # First try regular PDF extraction
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)
                pages = [page.extract_text() or "" for page in reader.pages]

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
            logger.error(f"Text extraction failed: {pdf_path} → {e}")
            traceback.print_exc()
            return []

    def split_text(self, text: str) -> List[str]:
        """
        Splits large text into smaller overlapping chunks using RecursiveCharacterTextSplitter.

        Args:
            text (str): The raw input text to be split.

        Returns:
            List[str]: List of text chunks with specified size and overlap.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_text(text)

    def identify_chunk_source(self, chunk: str, doc_cache: Dict[str, List[str]]) -> tuple:
        """
        Attempts to identify the source document and page number of a given text chunk using fuzzy matching.

        Args:
            chunk (str): Text chunk to identify.
            doc_cache (Dict[str, List[str]]): Mapping from document ID to list of pages.

        Returns:
            tuple: (document ID, page number) where the chunk most likely originated.
        """
        best_match, max_ratio = (None, None), 0
        
        for doc_id, pages in doc_cache.items():
            for idx, page in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page).ratio()
                if ratio > max_ratio:
                    best_match = (doc_id, idx)
                    max_ratio = ratio
                    
        return best_match

    def build_text_index(self, document_paths: List[str], output_subdir: str):
        """
        Builds a vector index from the given list of PDF documents.

        Steps:
        - Extracts text from PDF (with OCR fallback).
        - Splits text into chunks with overlap.
        - Embeds each chunk using Sentence Transformers.
        - Stores the embeddings in a persistent ChromaDB collection.
        - Saves chunks and chunk-to-source mapping as JSON for later retrieval.

        Args:
            document_paths (List[str]): List of paths to PDF files.
            output_subdir (str): Subdirectory inside `data_dir` to save outputs.

        Returns:
            tuple: (Chroma collection object, list of text chunks, list of metadata mappings)
        """
        logger.info(f"Building text index for inference")
        
        # Remove accidental base folder duplication
        if output_subdir.startswith(self.data_dir):
            output_subdir = os.path.relpath(output_subdir, start=self.data_dir)

        output_dir = os.path.join(self.data_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        all_chunks, mapping, doc_cache = [], [], {}

        for pdf_file in tqdm(document_paths, desc="Text indexing"):
            # Normalize file names
            file_name = pdf_file if pdf_file.lower().endswith(".pdf") else f"{pdf_file}.pdf"
            
            logger.info(f"Processing {file_name}")
            
            doc_id = os.path.splitext(os.path.basename(file_name))[0]
            
            pdf_path = file_name 
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found, skipping: {pdf_path}")
                continue
            
            try:
                pages: List[str] = self.extract_text_from_pdf(pdf_path)
            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_path}: {e}")
                traceback.print_exc()
                continue
            
            doc_cache[doc_id] = pages
            full_text = "\n".join(pages)
            split_chunks = self.split_text(full_text)

            for i, chunk in enumerate(split_chunks):
                all_chunks.append(chunk)
                
                # (id, page_num)
                source = self.identify_chunk_source(chunk, doc_cache)
                mapping.append({
                    "chunk_pdf_name": source[0] or doc_id,
                    "pdf_page_number": source[1] if source[1] is not None else i
                })
        
        # Persist raw chunks + mapping as JSON
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8", errors="replace") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
        with open(os.path.join(output_dir, "chunk_mapping.json"), "w", encoding="utf-8", errors="replace") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(all_chunks)} chunks + mapping to {output_dir}")

        # Chroma embedding index
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        col_name = "mdocagent_text_index"
        
        if self.force_reindex:
            # === Force reindex: delete then create ===
            try:
                chroma_client.delete_collection(name=col_name)
                logger.info(f"Deleted existing Chroma collection '{col_name}' due to force_reindex=True")
            except chromadb.errors.NotFoundError:
                logger.info(f"Collection '{col_name}' not found; nothing to delete.")
        
        # Create or get existing collection
        collection = chroma_client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Chroma collection '{col_name}'")
        
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        collection.add(documents=all_chunks, ids=ids)
        logger.info(f"Created Chroma collection '{col_name}' with {len(all_chunks)} chunks")

        return collection, all_chunks, mapping

    def retrieve(self, query: str, index_obj, all_chunks, chunk_to_doc_mapping, output_subdir: str) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K most relevant text chunks for a query using vector similarity in ChromaDB.

        Args:
            query (str): Natural language query.
            index_obj: Optional ChromaDB collection object. If None, it is loaded from disk.
            all_chunks (List[str]): All embedded text chunks.
            chunk_to_doc_mapping (List[Dict]): Mapping of each chunk to its PDF source and page.
            output_subdir (str): Directory where chunk files are stored.

        Returns:
            List[Dict[str, Any]]: Top-K retrieved chunks with:
                - 'chunk': text content
                - 'chunk_pdf_name': originating PDF name
                - 'pdf_page_number': page number in source PDF
                - 'rank': rank of the result
                - 'score': cosine similarity score (1 - distance)
        """
        logger.info(f"Text Retriever Querying Chroma index with: {query}")
        
        # Load raw data
        if not all_chunks or not chunk_to_doc_mapping:
            with open(os.path.join(output_subdir, "chunks.json"), "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
                
            with open(os.path.join(output_subdir, "chunk_mapping.json"), "r", encoding="utf-8") as f:
                chunk_to_doc_mapping = json.load(f)
            
            logger.info(f"Loaded {len(all_chunks)} chunks + mapping from {output_subdir}")

        # Load index object
        if index_obj is None:
            col_name = "mdocagent_text_index"
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Load the existing collection
            index_obj = chroma_client.get_collection(
                name=col_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded Chroma collection '{col_name}'")

        # Query Chroma for nearest neighbors
        result = index_obj.query(query_texts=[query], n_results=self.top_k)
        ids = result["ids"][0]
        distances = result["distances"][0]  # Cosine distance
        
        contexts: List[Dict[str, Any]] = []
        
        for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
            logger.info(f"[Text Rank {rank}] Chunk ID: {cid} | Similarity: {1 - dist:.4f}")
            
            idx = int(cid.split("_")[1])
            
            contexts.append({
                "chunk": all_chunks[idx],
                "chunk_pdf_name": chunk_to_doc_mapping[idx]["chunk_pdf_name"],
                "pdf_page_number": chunk_to_doc_mapping[idx]["pdf_page_number"],
                "rank": rank,
                "score": float(1.0 - dist) # convert distance → similarity
            })

        logger.info(f"Textual Contexts: {contexts}")
        logger.info(f"Retrieved {len(contexts)} textual contexts for query.")
        return contexts



