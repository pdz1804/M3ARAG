# agents/text_agent.py
from agents.base import BaseAgent
from rag_text.loader import load_documents_from_folder
from rag_text.vectorstore import get_retriever
from rag_text.chain import build_rag_chain

class TextRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("TextRAGAgent")

        docs = load_documents_from_folder()
        if not docs:
            raise RuntimeError("⚠️ No text documents found.")

        retriever = get_retriever(docs, persist_path="vectorstores/text_db")
        self.rag_chain = build_rag_chain(retriever)

    def run(self, input_data: dict) -> str:
        question = input_data.get("question", "")
        return self.rag_chain.invoke(question)


