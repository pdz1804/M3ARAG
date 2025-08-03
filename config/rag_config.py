# rag_config.py

rag_config = {
    "data_dir": "data/merge/",
    "vision_retriever": "colpali",       # or 'colqwen'
    "text_retriever": "minilm",          # or 'mpnet', 'bge'
    "top_k": 7,                         # Increased from 3 to retrieve more relevant chunks
    "chunk_size": 4000,                 # Increased from 3000 for larger context chunks  
    "chunk_overlap": 500,               # Increased from 300 for better continuity
    "force_reindex": False,
    
    # New parameters for enhanced retrieval
    "diversity_threshold": 0.7,         # Ensure diverse content retrieval
    "relevance_threshold": 0.3,         # Lower threshold to include more potentially relevant content
}
