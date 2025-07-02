# agents/image_agent.py
from agents.base import BaseAgent
from rag_image.loader import caption_images_to_documents
from rag_image.vectorstore import get_image_retriever
from rag_image.chain import build_rag_chain

class ImageRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("ImageRAGAgent")

        docs = caption_images_to_documents()
        if not docs:
            raise RuntimeError("⚠️ No images found.")

        retriever = get_image_retriever(docs, persist_path="vectorstores/image_db")
        self.rag_chain = build_rag_chain(retriever)

    def run(self, input_data: dict) -> str:
        question = input_data.get("question", "")
        return self.rag_chain.invoke(question)


