"""
agents/text_agent.py

Defines the `TextAgent`, a concrete subclass of `BaseAgent`, responsible for answering
natural language questions using retrieved **textual contexts** from the RAG system.

Functionality:
- Supports multiple LLM backends for question answering:
    • OpenAI GPT-4o-mini
    • Google Gemini
    • Qwen2.5-VL (default)
- Converts top-k retrieved text chunks into a string prompt and passes them to the selected model.
- Uses `RunnableLambda` from LangChain to wrap backend inference functions.
- Manages GPU memory cleanup for long sessions or batch inference.

Usage:
    agent = TextAgent(name="TextAgent", qa_model="qwen")
    answer = agent.run({"question": "What is Mamba?"}, text_chunks)
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from rag_text.base_runner import get_text_captioning_runner
import torch
import gc

class TextAgent(BaseAgent):
    def __init__(self, name: str = "TextAgent", qa_model = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing TextAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.caption_with_llm = get_text_captioning_runner(qa_model)

    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")
        
        if not question or not contexts:
            logger.warning("Missing question or contexts. Skipping generation.")
            return "No answer found."

        # Extract raw text chunks and prepare prompt
        contexts = [ctx['chunk'] for ctx in contexts]
        contexts_str = "\n- ".join(contexts)

        # Clean up memory before inference
        del contexts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate answer using the selected model
        return self.caption_with_llm.invoke({"query": question, "texts": contexts_str})


