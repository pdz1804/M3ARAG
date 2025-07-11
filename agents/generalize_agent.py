"""
agents/generalize_agent.py

This module defines the GeneralizeAgent class, which is responsible for consolidating or merging
responses from multiple specialized agents (e.g., TextAgent and ImageAgent) into a single coherent
answer. It supports multiple backend models for generalization logic, including OpenAI GPT-4o-mini
and Qwen2.5.

The agent works by injecting the sub-agent answers into a predefined generalization prompt and
passing the composed input through a language model pipeline (OpenAI or HuggingFace).
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config.prompt import GENERALIZED_PROMPT
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline
from langchain import HuggingFacePipeline

class GeneralizeAgent(BaseAgent):
    def __init__(self, name: str = "GeneralizeAgent", qa_model: str = "qwen"):
        super().__init__(name)
        logger.info(f"Initializing GeneralizeAgent with backend model: {qa_model}")
        
        prompt = PromptTemplate.from_template(GENERALIZED_PROMPT)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()

            qwen_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                device_map="auto",
                max_length=1024,
                truncation=True,
            )
            qwen_llm = HuggingFacePipeline(pipeline=qwen_pipeline)

            self.chain = prompt | qwen_llm | StrOutputParser()
        
        elif "gemini" in qa_model:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) | StrOutputParser()
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                raise e

        else:
            raise ValueError(f"Unknown QA model: {qa_model}")

    def run(self, input_data: dict) -> str:
        text_resp = input_data.get("TextAgent", "")
        image_resp = input_data.get("ImageAgent", "")
        
        if not text_resp and not image_resp:
            logger.warning("⚠️ Both TextAgent and ImageAgent responses are missing or empty.")
            return "No content to generalize."
        
        try:
            result = self.chain.invoke({
                "text_answer": text_resp,
                "image_answer": image_resp
            })
            logger.info("Generalization complete.")
            return result
        except Exception as e:
            logger.error(f"Generalization failed: {e}")
            return "Failed to generalize responses."


