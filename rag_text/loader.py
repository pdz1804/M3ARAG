# # rag_text/loader.py
"""Load and chunk PDF documents from both extract/pdf and store using PyPDFLoader."""

import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_folder(
    extract_path: str = "data/extract/pdf",
    store_path: str = "data/store"
) -> List[Document]:
    """
    Load and split all PDF files from extract and store folders.

    Returns:
        List[Document]: List of split document chunks.
    """
    try:
        all_docs = []

        for folder_path in [extract_path, store_path]:
            folder = Path(folder_path)
            if not folder.exists():
                logger.warning(f"⚠️ Skipped missing folder: {folder}")
                continue

            for pdf_file in folder.glob("*.pdf"):
                logger.info(f"📄 Loading: {pdf_file}")
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = f"{folder.name}/{pdf_file.name}"
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"❌ Failed to load {pdf_file}: {e}")

        if not all_docs:
            logger.warning("⚠️ No PDFs found in extract or store folders.")
            return []

        # === Split into chunks ===
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        return splitter.split_documents(all_docs)

    except Exception as e:
        logger.error(f"❌ Unexpected failure during document loading: {e}")
        return []

