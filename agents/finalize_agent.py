# agents/finalize_agent.py
from agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import FINALIZED_PROMPT

class FinalizeAgent(BaseAgent):
    def __init__(self):
        super().__init__("FinalizeAgent")
        prompt = PromptTemplate.from_template(FINALIZED_PROMPT)
        self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

    def run(self, input_data: dict) -> str:
        return self.chain.invoke({
            "question": input_data.get("question", ""),
            "general_answer": input_data.get("general_answer", "")
        })
