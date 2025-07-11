"""
agents/planning_agent.py

This module implements the `PlanningAgent`, responsible for decomposing complex user queries
into a set of smaller, focused sub-questions (tasks) to be answered independently.

Key Responsibilities:
- Loads a language model (OpenAI GPT-4o or Qwen via Hugging Face).
- Applies a prompt template (`PLANNING_PROMPT`) to guide the decomposition process.
- Parses and returns a list of task strings in JSON format.

Typical Use:
    agent = PlanningAgent(qa_model="openai")
    tasks = agent.run({"question": "What are the challenges and opportunities in AI adoption?"})
"""

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
        """
        Generates a list of subtasks based on a user query.

        Args:
            input_data (dict): A dictionary containing the user query under key "question".

        Returns:
            list: A list of subtask strings. Returns an empty list if parsing fails or no tasks are produced.
        """
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


