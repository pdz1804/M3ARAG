# agents/text_agent.py
from langchain.agents import tool
from rag_text.loader import load_documents_from_folder
from rag_text.vectorstore import get_retriever
from rag_text.chain import build_rag_chain
from agents.base import BaseAgent

class TextRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("TextRAGAgent")
        self.rag_chain = None

    def run(self, input_data: dict) -> str:
        question = input_data.get("question", "")
        if self.rag_chain is None:
            docs = load_documents_from_folder()
            if not docs:
                return "⚠️ No documents found to answer your question."
            retriever = get_retriever(docs, persist_path="vectorstores/text_db")
            self.rag_chain = build_rag_chain(retriever)
        return self.rag_chain.invoke(question)


