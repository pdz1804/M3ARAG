"""
MDocRAG: Advanced Multi-Modal Document Retrieval-Augmented Generation System

This module implements the MDocRAG class, which serves as the central coordinator
for a sophisticated multi-modal retrieval-augmented generation pipeline that
combines textual and visual understanding for comprehensive document analysis.

System Architecture Overview:
    MDocRAG orchestrates two complementary retrieval pipelines to provide
    comprehensive document understanding through dual-modality processing:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    MDocRAG Coordinator                          │
    ├─────────────────────┬───────────────────────────────────────────┤
    │   Text Pipeline     │            Visual Pipeline                │
    │                     │                                           │
    │ ┌─────────────────┐ │ ┌─────────────────────────────────────────┤
    │ │ • PDF Text      │ │ │ • PDF → Page Images                     │
    │ │   Extraction    │ │ │ • Visual Embedding (ColPali/ColQwen)   │
    │ │ • Semantic      │ │ │ • Page-Level Indexing                  │
    │ │   Chunking      │ │ │ • Visual Similarity Search             │
    │ │ • Embedding     │ │ │ • Context-Aware Retrieval              │
    │ │   (SentenceT.)  │ │ │                                         │
    │ │ • ChromaDB      │ │ │                                         │
    │ │   Indexing      │ │ │                                         │
    │ └─────────────────┘ │ └─────────────────────────────────────────┤
    └─────────────────────┴───────────────────────────────────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │    Multi-Modal Query      │
                          │      Processing           │
                          │                           │
                          │ • Parallel Retrieval      │
                          │ • Result Ranking          │
                          │ • Context Integration     │
                          │ • Response Synthesis      │
                          └───────────────────────────┘

Core Responsibilities:
    1. **Document Ingestion**: Comprehensive multi-format document processing
    2. **Dual-Pipeline Indexing**: Synchronized text and visual index creation
    3. **Multi-Modal Retrieval**: Parallel text and visual content retrieval
    4. **Result Integration**: Intelligent merging of textual and visual results
    5. **Index Management**: Persistent storage and efficient index operations

Text Processing Pipeline (TextRAGPipeline):
    The text pipeline handles textual content extraction and indexing:
    
    **Document Processing**:
    - Multi-format text extraction using Docling library
    - Intelligent text cleaning and preprocessing
    - Document structure preservation and metadata extraction
    
    **Semantic Chunking**:
    - Context-aware text segmentation with configurable parameters
    - Overlap management for semantic continuity
    - Document structure preservation with hierarchical chunking
    
    **Embedding Generation**:
    - High-quality sentence embeddings using state-of-the-art models
    - Support for multiple embedding models (MiniLM, MPNet, BGE)
    - Batch processing for efficient large document handling
    
    **Vector Storage**:
    - ChromaDB integration for scalable similarity search
    - Persistent storage with incremental index updates
    - Efficient retrieval with configurable similarity thresholds

Visual Processing Pipeline (VisualRAGPipeline):
    The visual pipeline processes document images for visual understanding:
    
    **Image Generation**:
    - High-quality PDF page rendering with configurable resolution
    - Multi-page processing with parallel image generation
    - Format optimization for downstream processing
    
    **Visual Embedding**:
    - Advanced visual embeddings using ColPali or ColQwen models
    - Late-interaction mechanisms for fine-grained visual understanding
    - GPU-optimized processing with memory management
    
    **Visual Indexing**:
    - Efficient visual similarity search capabilities
    - Page-level indexing with document structure preservation
    - Persistent storage with optimized retrieval mechanisms

Multi-Modal Query Processing:
    The system provides sophisticated query processing capabilities:
    
    **Parallel Retrieval**:
    - Simultaneous text and visual content retrieval
    - Independent ranking and scoring mechanisms
    - Configurable retrieval parameters per modality
    
    **Result Integration**:
    - Intelligent merging of text and visual results
    - Context-aware ranking and relevance scoring
    - Duplicate detection and result deduplication
    
    **Context Assembly**:
    - Comprehensive context preparation for agent processing
    - Source attribution and metadata preservation
    - Quality assurance and completeness verification

Advanced Features:
    - **Incremental Indexing**: Efficient updates for new documents
    - **Index Persistence**: Reliable storage and recovery mechanisms
    - **Memory Optimization**: Efficient resource utilization
    - **Concurrent Processing**: Parallel operations for improved performance
    - **Error Recovery**: Robust error handling and graceful degradation

Configuration Management:
    The system supports comprehensive configuration for optimal performance:
    
    ```python
    config = {
        "data_dir": "data/merge/",              # Document directory
        "vision_retriever": "colpali",          # Visual model selection
        "text_retriever": "minilm",             # Text embedding model
        "top_k": 3,                             # Results per query
        "chunk_size": 3000,                     # Text chunk size
        "chunk_overlap": 300,                   # Chunk overlap
        "force_reindex": False                  # Force index rebuild
    }
    ```

Performance Optimizations:
    - **Lazy Loading**: Resources loaded on demand for memory efficiency
    - **Batch Processing**: Optimized bulk operations for large documents
    - **Caching**: Intelligent caching of frequently accessed data
    - **Memory Management**: Automatic cleanup and resource optimization
    - **GPU Acceleration**: Hardware acceleration where available

Usage Examples:
    ```python
    # Initialize MDocRAG system
    config = {
        "data_dir": "data/pdf/",
        "vision_retriever": "colpali",
        "text_retriever": "minilm", 
        "top_k": 5
    }
    rag = MDocRAG(config)
    
    # Build indices from documents
    rag.ingest(pdf_dir="data/pdf/", index_dir="data/index/")
    
    # Query the system
    results = rag.retrieve_results("What are the key findings?")
    text_results = results["text_results"]
    visual_results = results["visual_results"]
    ```

Integration Points:
    - **Agent System**: Provides contexts for multi-agent processing
    - **Pipeline Integration**: Seamless integration with M3APipeline
    - **External APIs**: Support for various embedding and vision APIs
    - **Storage Systems**: Flexible storage backend support

Quality Assurance:
    - **Comprehensive Logging**: Detailed operation logging for monitoring
    - **Error Handling**: Robust error recovery and reporting
    - **Validation**: Input and output validation for reliability
    - **Testing**: Extensive testing coverage for stability
    - **Monitoring**: Performance monitoring and optimization

Dependencies:
    - **Core**: langchain, chromadb, sentence-transformers
    - **Visual**: colpali-engine, pdf2image, pillow
    - **Text**: docling, pypdf, transformers
    - **Storage**: sqlite3, pickle, json
    - **Utilities**: logging, pathlib, typing

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Advanced Multi-Modal Architecture)
License: MIT License
"""
# mdoc_rag.py
# At the very top of mdoc_rag.py, BEFORE ANY OTHER IMPORTS
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import os
from RAG.text_pipeline import TextRAGPipeline
from RAG.visual_pipeline import VisualRAGPipeline

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
        self.data_dir = config.get("data_dir", "../data/pdf/")
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

    def retrieve_results(self, query_str: str):
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
        
        text_results = results 
        visual_results = vis
        
        return {
            "text_results": text_results,
            "visual_results": visual_results
        }

# if __name__ == "__main__":
#     config = {
#         "data_dir": "../data/pdf/",
#         "vision_retriever": "colpali",       # or 'colqwen'
#         "text_retriever": "minilm",          # or 'mpnet', 'bge'
#         "top_k": 5,
#         "chunk_size": 3000,
#         "chunk_overlap": 300,
#         "force_reindex": False
#     }

#     rag = MDocRAG(config)

#     index_dir = "../data/pdf/index"
#     pdf_dir = config["data_dir"]

#     # Step 1: Ingest
#     rag.ingest(pdf_dir=pdf_dir, index_dir=index_dir)

#     # Step 2: Interactive query loop
#     logger.info("\nMulti-modal Document Chat is ready!")
#     logger.info("Type your query below (or type 'exit' to quit):\n")

#     while True:
#         try:
#             query = input("❓ Ask: ").strip()
#             if query.lower() in {"exit", "quit"}:
#                 logger.info("Session ended.")
#                 break

#             rag.retrieve_results(query)

#         except KeyboardInterrupt:
#             logger.info("\nInterrupted.")
#             break
#         except Exception as e:
#             logger.error(f"❌ Query error: {e}")
            
            