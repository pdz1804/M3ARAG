import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from config.prompt import VERIFICATION_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFacePipeline
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline

class VerifierAgent(BaseAgent):
    def __init__(self, name="VerifierAgent", qa_model="openai", threshold=7):
        super().__init__(name)
        self.qa_model = qa_model
        self.threshold = threshold
        logger.info(f"Initializing VerifierAgent with model: {qa_model}, threshold: {threshold}")

        prompt = PromptTemplate.from_template(VERIFICATION_PROMPT)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            llm = pipeline("text2text-generation", model=model, tokenizer=processor.tokenizer, device_map="auto")
            self.chain = prompt | HuggingFacePipeline(pipeline=llm) | StrOutputParser()

    def run(self, input_data: dict) -> dict:
        question = input_data.get("question", "")
        answer = input_data.get("merged_answer", "")
        logger.info("VerifierAgent scoring final merged answer.")

        try:
            response = self.chain.invoke({"question": question, "answer": answer})

            # Parse the response: split into evaluation and score
            lines = response.strip().splitlines()
            eval_lines = [line for line in lines if not line.strip().lower().startswith("score")]
            score_lines = [line for line in lines if line.strip().lower().startswith("score")]

            evaluation = "\n".join(eval_lines).strip()
            score = int("".join(filter(str.isdigit, score_lines[0]))) if score_lines else 0
            
            # --- new code ---
            # After parsing score and evaluation
            followup_lines = [line for line in lines if line.strip().lower().startswith("follow")]
            follow_up_questions = []
            if followup_lines:
                for i in range(1, len(lines)):
                    if lines[i].strip().startswith("- "):
                        follow_up_questions.append(lines[i].strip("- ").strip())

            logger.info(f"VerifierAgent score: {score}/10")
            logger.info(f"VerifierAgent evaluation: {evaluation}")
            logger.info(f"VerifierAgent follow-up questions: {follow_up_questions}")

            return {
                "score": score,
                "needs_retry": score < self.threshold,
                "evaluation": evaluation,
                "merged_answer": answer,
                "follow_up_questions": follow_up_questions,
            }

        except Exception as e:
            logger.error(f"VerifierAgent failed: {e}")
            return {
                "score": 0,
                "needs_retry": True,
                "evaluation": "Verification failed due to error.",
                "merged_answer": answer
            }

