# rag_config.py

rag_config = {
    "data_dir": "data/merge/",
    "vision_retriever": "colpali",       # or 'colqwen'
    "text_retriever": "minilm",          # or 'mpnet', 'bge'
    "top_k": 3,
    "chunk_size": 3000,
    "chunk_overlap": 300,
    "force_reindex": False
}
