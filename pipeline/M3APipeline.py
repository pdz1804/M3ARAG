"""
M3APipeline: Multi-Modal Multi-Agent RAG Pipeline Orchestrator

This module implements the core M3APipeline class that coordinates an end-to-end
multimodal retrieval-augmented generation system for comprehensive document 
understanding through specialized AI agents.

System Architecture:
    The M3APipeline orchestrates a sophisticated workflow combining document
    ingestion, multi-modal indexing, and agent-based reasoning to answer
    complex queries with iterative quality improvement.

    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Document      │───▶│  Multi-Modal     │───▶│  Agent-Based    │
    │   Ingestion     │    │  Indexing        │    │  Query Answer   │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
                                     │
                           ┌─────────┴─────────┐
                           │                   │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │ Text Index  │    │Visual Index │
                    │ (ChromaDB)  │    │ (ColPali)   │
                    └─────────────┘    └─────────────┘

Key Responsibilities:
    - Document ingestion and preprocessing coordination
    - Dual-pipeline indexing management (textual + visual)
    - Multi-agent system configuration and orchestration
    - Query processing and response synthesis
    - Quality assurance through iterative refinement

Multi-Agent Architecture:
    The system employs six specialized agents working in coordination:

    1. TextAgent: Analyzes textual content from document chunks
       - Processes retrieved text contexts using configurable LLMs
       - Provides detailed, source-attributed textual insights
       - Supports OpenAI, Gemini, and Qwen models

    2. ImageAgent: Processes visual content from document pages
       - Analyzes charts, diagrams, tables, and visual elements
       - Extracts information from page-level visual contexts
       - Integrates with vision-language models for understanding

    3. GeneralizeAgent: Synthesizes multi-modal responses
       - Combines text and image agent outputs intelligently
       - Resolves conflicts and eliminates redundancy
       - Creates coherent, unified responses with source attribution

    4. PlanningAgent: Decomposes complex queries
       - Analyzes user questions for complexity and scope
       - Generates focused sub-questions for targeted retrieval
       - Ensures comprehensive coverage of user intent

    5. MergeAgent: Consolidates multiple sub-responses
       - Combines answers from different sub-questions
       - Maintains memory across iterations for context
       - Creates natural, flowing narrative responses

    6. VerifierAgent: Quality assessment and improvement
       - Evaluates response quality on multiple criteria
       - Generates follow-up questions for improvement
       - Drives iterative refinement until quality threshold met

Processing Workflow:
    1. Query Input → Planning Agent (decomposition)
    2. Sub-queries → RAG System (text + visual retrieval)
    3. Retrieved contexts → Text/Image Agents (analysis)
    4. Agent responses → Generalize Agent (synthesis)
    5. Sub-answers → Merge Agent (consolidation)
    6. Final answer → Verifier Agent (quality check)
    7. If quality insufficient → Generate follow-ups → Repeat
    8. Return high-quality answer to user

Technical Features:
    - Configurable agent models and parameters
    - Persistent index management with incremental updates
    - Memory-efficient processing with GPU optimization
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    - Scalable architecture for large document collections

Usage Example:
    ```python
    # Initialize pipeline with configuration
    pipeline = M3APipeline(
        pdf_dir="data/merge",
        index_dir="data/merge/index", 
        agent_config=agent_config,
        rag_config=rag_config,
        ingest_only=False
    )
    
    # Build indices from documents
    pipeline.ingest_cfg()
    
    # Process user queries
    answer = pipeline.process_query("What are the key findings?")
    ```

Configuration:
    Agent configuration controls model selection and behavior:
    - Model choices: OpenAI GPT-4o, Gemini 2.0, Qwen2.5-VL
    - Quality thresholds and iteration limits
    - Agent activation flags for customization

Performance Optimizations:
    - Lazy loading of models and indices
    - Batch processing for multiple queries
    - Memory management and cleanup
    - Efficient vector storage and retrieval
    - GPU acceleration where available

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Multi-Agent Architecture)
License: MIT License
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
