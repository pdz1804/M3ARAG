"""
agents/finalize_agent.py

This module defines the FinalizeAgent class, responsible for generating the final answer to the
user's question based on the consolidated response from previous agents (e.g., GeneralizeAgent).
It supports both OpenAI GPT-4o and HuggingFace-based Qwen2.5-VL models.

Usage:
- The agent formats input using a finalization prompt.
- It processes the query and generalized answer to produce a polished final response.
"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config.prompt import FINALIZED_PROMPT
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline
from langchain import HuggingFacePipeline

class FinalizeAgent(BaseAgent):
    def __init__(self, name: str = "FinalizeAgent", qa_model: str = "qwen"):
        super().__init__(name)
        logger.info(f"Initializing FinalizeAgent with backend model: {qa_model}")
        
        prompt = PromptTemplate.from_template(FINALIZED_PROMPT)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()

            qwen_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                device_map="auto",
                max_length=2048,
                truncation=True,
            )
            qwen_llm = HuggingFacePipeline(pipeline=qwen_pipeline)

            self.chain = prompt | qwen_llm | StrOutputParser()
            
    def run(self, input_data: dict) -> str:
        question = input_data.get("question", "")
        general_answer = input_data.get("GeneralizeAgent", "")
        
        if not question.strip():
            logger.warning("Missing question in input.")
            return "No question provided."

        if not general_answer.strip():
            logger.warning("Missing generalized answer to finalize.")
            return "No generalized answer available to finalize."

        try:
            result = self.chain.invoke({
                "question": question,
                "general_answer": general_answer
            })
            logger.info("Finalization complete.")
            return result
        except Exception as e:
            logger.error(f"FinalizeAgent failed to generate output: {e}")
            return "Failed to finalize the answer."


