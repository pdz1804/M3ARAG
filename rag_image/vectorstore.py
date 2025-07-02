# rag_image/vectorstore.py
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import os

def get_image_retriever(
    docs: List[Document],
    persist_path: str = "vectorstores/image_db",
    top_k: int = 5
) -> VectorStoreRetriever:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_path):
        print("🔁 Loading existing image vectorstore...")
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding_model,
        )
    else:
        print("🧠 Building new image vectorstore...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_path
        )
        vectorstore.persist()
        print("✅ Image DB saved.")

    return vectorstore.as_retriever(search_kwargs={"k": top_k})
