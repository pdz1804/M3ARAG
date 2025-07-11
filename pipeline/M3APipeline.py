"""
pipeline/M3APipeline.py

This module defines the M3APipeline class, which orchestrates an end-to-end multimodal
retrieval-augmented generation (RAG) pipeline combining PDF document ingestion,
multimodal retrieval (text and image), and multi-agent question answering (M3A: Multimodal Multi-Agent).

Key Responsibilities:
- Ingests a directory of PDFs and builds RAG indices (textual and visual).
- Retrieves top-k relevant chunks (text + images) in response to a user query.
- Dispatches multimodal context to a configurable multi-agent system with specialized roles:
    • TextAgent         Answers based on retrieved text.
    • ImageAgent        Answers based on image regions.
    • GeneralizeAgent   Merges or re-evaluates responses.
    • FinalizeAgent     Generates final user-facing answer.

Usage:
    pipeline = M3APipeline(pdf_dir, index_dir, rag_config, agent_config)
    pipeline.ingest_cfg()            # Index PDF documents
    pipeline.process_query(question)  # Ask a multimodal question
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
from RAG.mdoc_rag import MDocRAG as RAGSystem
import os
import warnings

# --- new code --- 
warnings.filterwarnings("ignore")

# import torch
# torch.set_default_dtype(torch.bfloat16)

class M3APipeline:
    """
    End-to-end pipeline that:
      Ingests one or more PDF documents via a RAG indexer
      Answers queries by retrieving multimodal context and dispatching it through the M3ARAG agents
    """

    def __init__(self, pdf_dir, index_dir, rag_config, agent_config, ingest_only=False):
        # Initialize RAG indexer and multi-agent QA system
        os.makedirs(index_dir, exist_ok=True)
        self.multi_agent = None
        self.rag = RAGSystem(rag_config)
        self.index_dir = index_dir
        self.pdf_dir = pdf_dir

        if ingest_only is False:
            self.use_text = agent_config.get("use_text", True)
            self.use_image = agent_config.get("use_image", True)

            # one global qa_model for all agents (you could customize per‐agent too)
            qa_text = agent_config.get("qa_text", "openai")
            qa_image = agent_config.get("qa_image", "gemini")
            qa_generalize = agent_config.get("qa_generalize", "openai")
            qa_planning = agent_config.get("qa_planning", "openai")
            qa_merge = agent_config.get("qa_merge", "openai")
            qa_verifier = agent_config.get("qa_verifier", "openai")

            # --- old code ---
            # self.multi_agent = MultiAgentRunner()
            # --- end old code ---
            
            # --- new code ---
            self.multi_agent = MultiAgentRunner(self.rag, agent_config)
            # --- end new code ---

            if self.use_text:
                self.multi_agent.register_agent("TextAgent", qa_model=qa_text)

            if self.use_image:
                self.multi_agent.register_agent("ImageAgent", qa_model=qa_image)

            self.multi_agent.register_agent("GeneralizeAgent", qa_model=qa_generalize)
            
            # --- new code ---
            self.multi_agent.register_agent("PlanningAgent", qa_model=qa_planning)
            self.multi_agent.register_agent("MergeAgent", qa_model=qa_merge)
            self.multi_agent.register_agent("VerifierAgent", qa_model=qa_verifier)
            # --- end new code --- 

    def ingest_cfg(self) -> None:
        """
        Build the RAG indices over the provided PDF files.
        After calling this, the pipeline is ready to answer queries.
        """
        self.rag.ingest(pdf_dir=self.pdf_dir, index_dir=self.index_dir)

    def process_query(self, question: str):
        """
        Retrieve the top-k text segments and image pages for the question,
        then run the Multi Agent Runner for question answering

        Args:
            question: the user's natural-language question
            top_k: how many images/text chunks to receive

        """
        # --- new code ---
        return self.multi_agent.run(question)
        # --- end new code ---


# # Example usage of the M3APipeline class
# def main():
#     # Initialize pipeline with configs
#     # Correct PDF directory
#     pdf_dir = "data/extract/pdf"
    
#     # Consistent index folder under PDF dir
#     index_dir = "data/extract/pdf/index"
    
#     pipeline = M3APipeline(
#         pdf_dir=pdf_dir, 
#         index_dir=index_dir, 
#         agent_config=agent_config, 
#         rag_config=rag_config, 
#         ingest_only=False
#     )
    
#     # Run ingestion (text + visual indexing)
#     pipeline.ingest_cfg()

#     print("\n🤖 M3A Pipeline Chat Ready! Ask me anything about the documents.")
#     print("🔁 Type 'exit', 'quit', or press Ctrl+C to stop.\n")

#     try:
#         while True:
#             question = input("❓ Ask: ").strip()
#             if question.lower() in {"exit", "quit"}:
#                 print("👋 Exiting chat. Goodbye!")
#                 break

#             # Process and respond to the question
#             pipeline.process_query(question)

#     except KeyboardInterrupt:
#         print("\n👋 Chat interrupted. Goodbye!")

# if __name__ == '__main__':
#     main()
