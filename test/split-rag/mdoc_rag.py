"""
mdoc_rag.py

Multimodal Document Retrieval-Augmented Generation (MDocRAG) pipeline.

This module coordinates the ingestion and querying of both textual and visual content
from PDFs, leveraging two pipelines:
- TextRAGPipeline: extracts and indexes text chunks from PDFs.
- VisualRAGPipeline: converts PDF pages into images and indexes visual embeddings.

At runtime, both indexes are queried to return top-K relevant text chunks and images.

Author: [Your Name]
"""
# mdoc_rag.py
# At the very top of mdoc_rag.py, BEFORE ANY OTHER IMPORTS
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visdomrag.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
import os
from text_pipeline import TextRAGPipeline
from visual_pipeline import VisualRAGPipeline

logger = logging.getLogger("MDocRAG")
# torch.set_default_dtype(torch.bfloat16)

class MDocRAG:
    """
    Multimodal RAG pipeline that integrates visual and textual retrieval from PDFs.

    Attributes:
        config (dict): Configuration for model selection, chunking, paths, etc.
        text_rag (TextRAGPipeline): Instance of the text retrieval pipeline.
        visual_rag (VisualRAGPipeline): Instance of the visual retrieval pipeline.
        text_index_obj: The ChromaDB text index object.
        text_chunks (List[str]): List of text chunks extracted from PDFs.
        text_mapping (List[dict]): Mapping from chunk to document and page.
        page_embeddings (Dict[str, Tensor]): Visual embedding tensors keyed by page ID.
        page_map (Dict[str, dict]): Metadata mapping page IDs to source PDF and page number.
    """
    def __init__(self, config):
        self.config = config
        self.text_rag = TextRAGPipeline(config)
        self.visual_rag = VisualRAGPipeline(config)
        
        # Textual pipeline
        self.text_index_obj, self.text_chunks, self.text_mapping = None, None, None
        
        # Visual pipeline
        self.page_embeddings, self.page_map = None, None

    def ingest(self, pdf_dir: str, index_dir: str):
        """
        Ingest all PDF files from a directory and build both text and visual indexes.

        This includes:
        - Text extraction, chunking, embedding, and ChromaDB indexing.
        - PDF-to-image conversion, visual embedding, and persistence to disk.

        Args:
            pdf_dir (str): Path to a directory containing `.pdf` files.
            index_dir (str): Directory where output indexes (visual/textual) will be stored.
        """
        # Prepare output subdirectories
        if pdf_dir is None:
            pdf_dir = self.data_dir

        vis_subdir = os.path.join(index_dir, "visual")
        txt_subdir = os.path.join(index_dir, "textual")
        os.makedirs(vis_subdir, exist_ok=True)
        os.makedirs(txt_subdir, exist_ok=True)
        
        # Gather all PDF paths
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ]
        
        logger.info(f"Found {len(pdf_files)} PDF files for ingestion:")
        for path in pdf_files:
            logger.info(f"Found PDF: {path}")

        if not pdf_files:
            logger.warning(f"No PDFs found in {pdf_dir}; nothing to ingest.")
            return

        # Build (or reload) text index
        #    - Stores under txt_subdir:
        #       • chunks.json
        #       • chunk_mapping.json
        #       • Chroma collection (if embedding‐based)
        self.text_index_obj, self.text_chunks, self.text_mapping = self.text_rag.build_text_index(
            document_paths=pdf_files, 
            output_subdir=txt_subdir
        )

        # Build (or reload) visual index
        #    - Stores: page_embeddings, page_id_map.pkl, document_page_map.json
        # logger.info(f"Ingesting visual index into {vis_subdir}")
        self.page_embeddings, self.page_map = self.visual_rag.build_visual_index(
            document_paths=pdf_files, 
            output_dir=vis_subdir
        )
        
        logger.info(
            f"Ingestion complete:  "
            f"{len(self.page_embeddings)} pages indexed visually,  "
            f"{len(self.text_chunks)} text chunks indexed."
        )

    def query(self, query_str: str):
        """
        Run a multimodal query across both visual and textual indexes.

        Steps:
        - Use `text_rag.retrieve()` to return top-K text chunks and metadata.
        - Use `visual_rag.retrieve()` to return top-K page images and metadata.

        Args:
            query_str (str): User-provided natural language query.
        """
        logger.info("\nTop Text Chunks:")
        results = self.text_rag.retrieve(
            query_str, self.text_index_obj, self.text_chunks, self.text_mapping,
            output_subdir=os.path.join(self.config["data_dir"], "index", "textual")
        )
        for r in results:
            logger.info(f"{r['chunk_pdf_name']} | Page {r['pdf_page_number']} | Score {r['score']:.2f}\n---")
            logger.info(f"{r['chunk'].strip()}\n---")

        logger.info("\nTop Visual Pages:")
        vis = self.visual_rag.retrieve(
            query_str,
            document_page_map=self.page_map,
            index_subdir=os.path.join(self.config["data_dir"], "index", "visual")
        )
        for v in vis:
            logger.info(f"{v['document_id']} | Page {v['page_number']} | Score {v['score']:.2f}")

if __name__ == "__main__":
    config = {
        "data_dir": "../data/pdf/",
        "vision_retriever": "colpali",       # or 'colqwen'
        "text_retriever": "minilm",          # or 'mpnet', 'bge'
        "top_k": 5,
        "chunk_size": 3000,
        "chunk_overlap": 300,
        "force_reindex": False
    }

    rag = MDocRAG(config)

    index_dir = "../data/pdf/index"
    pdf_dir = config["data_dir"]

    # Step 1: Ingest
    rag.ingest(pdf_dir=pdf_dir, index_dir=index_dir)

    # Step 2: Interactive query loop
    logger.info("\nMulti-modal Document Chat is ready!")
    logger.info("Type your query below (or type 'exit' to quit):\n")

    while True:
        try:
            query = input("❓ Ask: ").strip()
            if query.lower() in {"exit", "quit"}:
                logger.info("Session ended.")
                break

            rag.query(query)

        except KeyboardInterrupt:
            logger.info("\nInterrupted.")
            break
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            
            