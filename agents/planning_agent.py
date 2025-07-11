import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
from agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config.prompt import PLANNING_PROMPT
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline
from langchain import HuggingFacePipeline

class PlanningAgent(BaseAgent):
    def __init__(self, name="PlanningAgent", qa_model="openai"):
        super().__init__(name)
        logger.info(f"Initializing PlanningAgent with model: {qa_model}")

        prompt = PromptTemplate.from_template(PLANNING_PROMPT)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            llm = pipeline("text2text-generation", model=model, tokenizer=processor.tokenizer, device_map="auto")
            self.chain = prompt | HuggingFacePipeline(pipeline=llm) | StrOutputParser()

    def run(self, input_data: dict) -> list:
        question = input_data.get("question", "")
        logger.info(f"PlanningAgent generating tasks for: {question}")

        try:
            response = self.chain.invoke({"question": question})
            parsed = json.loads(response)
            tasks = parsed.get("tasks", [])
            logger.info(f"Generated tasks: {tasks}")
            return tasks
        except Exception as e:
            logger.error(f"PlanningAgent failed: {e}")
            return []


