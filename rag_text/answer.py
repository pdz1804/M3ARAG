"""Main API interface for text-based RAG: get_text_answer(question: str)"""

import logging
from rag_text.loader import load_documents_from_folder
from rag_text.vectorstore import get_retriever
from rag_text.chain import build_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lazy init
_chain = None

def get_document_ready() -> bool:
    pass

def get_text_answer(question: str) -> str:
    """Answer question using text-based document RAG."""
    global _chain

    try:
        if _chain is None:
            logger.info("🔄 Initializing RAG chain for first use...")
            docs = load_documents_from_folder()
            
            retriever = get_retriever(docs, persist_path="vectorstores/text_db")
            _chain = build_rag_chain(retriever)

        return _chain.invoke(question)

    except Exception as e:
        logger.error(f"❌ RAG answering failed: {e}")
        return "Sorry, I couldn't retrieve an answer at this time."
