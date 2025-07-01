# rag_text/vectorstore.py
"""Handles embedding and persisting Chroma vector store."""

import os
import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def get_retriever(
    splits: List[Document], persist_path: str = "vectorstores/text_db", top_k: int = 5
) -> VectorStoreRetriever:
    """Create or load Chroma retriever from split docs."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(persist_path):
            logger.info("🔁 Loading existing vectorstore...")
            vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=embedding_model,
            )
        else:
            logger.info("🧠 Building new vectorstore...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=persist_path
            )
            vectorstore.persist()
            logger.info("✅ Chroma DB saved.")

        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    except Exception as e:
        logger.error(f"❌ Failed to initialize vectorstore: {e}")
        raise
