# agents/generalize_agent.py
from agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import GENERALIZED_PROMPT

class GeneralizeAgent(BaseAgent):
    def __init__(self):
        super().__init__("GeneralizeAgent")
        prompt = PromptTemplate.from_template(GENERALIZED_PROMPT)
        self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

    def run(self, input_data: dict) -> str:
        return self.chain.invoke({
            "text_answer": input_data.get("text_answer", ""),
            "image_answer": input_data.get("image_answer", "")
        })
