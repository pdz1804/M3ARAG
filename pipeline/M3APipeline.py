"""
pipeline/M3APipeline.py

This module defines the M3APipeline class, which orchestrates an end-to-end
multimodal retrieval-augmented generation (RAG) pipeline for understanding documents.

It combines PDF ingestion, multimodal (text + image) retrieval, and a multi-agent
LLM system to answer complex queries with iterative refinement.

Key Responsibilities:
- Ingest and index PDF documents into text and image vector stores.
- Retrieve top-k relevant content chunks in response to user questions.
- Orchestrate multi-agent reasoning using a configurable agent setup.

Supported Agents:
- TextAgent:         Answers using retrieved textual chunks.
- ImageAgent:        Answers using image regions or page-level visual context.
- GeneralizeAgent:   Synthesizes answers from multiple agents into a single response.
- PlanningAgent:     Decomposes complex questions into structured sub-questions.
- MergeAgent:        Fuses sub-agent responses into a coherent final answer.
- VerifierAgent:     Evaluates merged answer, determines quality, and suggests refinement.

Usage Example:
    pipeline = M3APipeline(pdf_dir, index_dir, rag_config, agent_config)
    pipeline.ingest_cfg()                     # Index documents
    final_answer = pipeline.process_query("What are the company's core products?")
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
    Multimodal Multi-Agent Pipeline (M3APipeline) that manages end-to-end RAG-based
    document understanding from ingestion to multimodal query answering.

    This pipeline:
    - Indexes documents (PDFs) into text and visual vector stores.
    - Configures a multi-agent system with different LLM-based agents for specialized tasks.
    - Orchestrates retrieval and multi-agent reasoning to answer user queries.

    Agents can include:
    - TextAgent: answers using retrieved text content.
    - ImageAgent: answers using visual information.
    - GeneralizeAgent: merges and generalizes responses from multiple agents.
    - PlanningAgent: decomposes queries into sub-questions.
    - MergeAgent: fuses multiple answers.
    - VerifierAgent: evaluates merged answers for quality control.
    """
    def __init__(self, pdf_dir, index_dir, rag_config, agent_config, ingest_only=False):
        """
        Initialize the M3APipeline.

        Args:
            pdf_dir (str): Directory containing the PDF documents to ingest.
            index_dir (str): Directory where the text and image indices will be stored.
            rag_config (dict): Configuration parameters for the RAG system (e.g., retrievers, top-k).
            agent_config (dict): Configuration for which agents to enable and their corresponding models.
            ingest_only (bool): If True, skip initializing agents and only build the index.
        """
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
        Ingest all PDF documents and build text and image-based vector indices.

        This method must be called before querying, unless precomputed indices already exist.
        """
        self.rag.ingest(pdf_dir=self.pdf_dir, index_dir=self.index_dir)

    def process_query(self, question: str):
        """
        Process a user query by retrieving relevant text and images, and dispatching them
        through a configured multi-agent reasoning system.

        Args:
            question (str): Natural language question provided by the user.

        Returns:
            str: Final answer generated by the agent system.
        """
        return self.multi_agent.run(question)

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
