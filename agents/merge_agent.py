import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from config.prompt import MERGE_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFacePipeline
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline

class MergeAgent(BaseAgent):
    def __init__(self, name="MergeAgent", qa_model="openai"):
        super().__init__(name)
        logger.info(f"Initializing MergeAgent with model: {qa_model}")

        prompt = PromptTemplate.from_template(MERGE_PROMPT)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            llm = pipeline("text2text-generation", model=model, tokenizer=processor.tokenizer, device_map="auto")
            self.chain = prompt | HuggingFacePipeline(pipeline=llm) | StrOutputParser()

        self.memory = []

    def run(self, input_data: dict) -> str:
        new_answers = input_data.get("generalized_answers", [])
        self.memory.extend(new_answers)
        logger.info("Merging answers using LLM + memory.")

        try:
            # Since each entry is a plain string (already generalized), just join them
            prompt_input = "\n".join([entry.strip() for entry in self.memory if isinstance(entry, str)])
            return self.chain.invoke({"answers": prompt_input})
            # --- end new code ---
        except Exception as e:
            logger.error(f"MergeAgent failed: {e}")
            return "Failed to merge answers."


