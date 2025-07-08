"""
VisualRAGPipeline: A visual document indexing and retrieval pipeline for multimodal RAG.

This module:
- Converts PDF pages into images.
- Encodes images using ColPali or ColQwen visual transformers.
- Builds a visual index using FAISS-compatible embedding structures.
- Retrieves relevant pages given a visual/natural-language query using vector similarity.

Author: PDZ, Bang
"""
# visual_pipeline.py
import os, json, torch, traceback
from tqdm import tqdm
from typing import List, Dict, Any
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO
import logging
import base64
from transformers import BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisualRAG")

class VisualRAGPipeline:
    """
    Visual retrieval pipeline for processing PDFs into image embeddings and querying them.

    Attributes:
        config (dict): Configuration dictionary.
        data_dir (str): Base directory for storing data and indexes.
        top_k (int): Number of top visual results to return for a query.
        vision_retriever (str): Visual embedding model name ("colpali" or "colqwen").
        vision_model: Loaded visual transformer model.
        vision_processor: Preprocessor/tokenizer associated with vision model.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = config["data_dir"]
        self.top_k = config["top_k"]
        self.vision_retriever = config["vision_retriever"]
        
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

    def build_visual_index(self, document_paths: List[str], output_dir: str):
        """
        Build the visual index from a list of PDFs.

        Each PDF page is converted to an image and passed through a visual encoder
        (ColPali or ColQwen). The resulting embeddings are saved for future retrieval.

        Args:
            document_paths (List[str]): List of PDF file paths.
            output_dir (str): Directory where embeddings and metadata will be stored.

        Returns:
            Tuple:
                page_embeddings (Dict[str, torch.Tensor]): Mapping from page_id to embedding tensor.
                document_page_map (Dict[str, Dict]): Maps page_id to its document ID and page index.
        """
        logger.info(f'Building visual index for inference using {self.vision_retriever}')
        
        # Remove accidental base folder duplication
        if output_dir.startswith(self.data_dir):
            output_dir = os.path.relpath(output_dir, start=self.data_dir)
        output_dir = os.path.join(self.data_dir, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        emb_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        page_embeddings, document_page_map = {}, {}
        
        # Track N for each page
        # - each vector has [1, N, D]
        # - check the N of all the images and make sure only keeps vectors with the same N
        embedding_n_map = {}  

        for pdf_file in tqdm(document_paths, desc="Processing PDFs for visual index"):
            fname = pdf_file if pdf_file.lower().endswith(".pdf") else f"{pdf_file}.pdf"
            
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

    def retrieve(self, query: str, document_page_map: Dict, index_subdir: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most relevant visual contexts for a given natural-language query.

        The function embeds the query using the vision encoder and compares it with stored
        image embeddings using cosine similarity (via score_multi_vector).

        Args:
            query (str): Natural language query string.
            document_page_map (Dict): Maps page IDs to {doc_id, page_idx}.
            index_subdir (str): Path where visual embeddings and metadata are stored.

        Returns:
            List[Dict]: Top K relevant pages, each with:
                - 'image': PIL.Image of the PDF page.
                - 'document_id': page ID string (e.g., "doc123_2").
                - 'page_number': int, page index.
                - 'score': float similarity score.
        """
        top_k = self.top_k
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


