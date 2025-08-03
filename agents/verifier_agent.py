"""
agents/verifier_agent.py

This module defines the `VerifierAgent`, responsible for evaluating the final merged answer
in a multi-agent RAG system. It assesses answer quality based on relevance, completeness,
correctness, and clarity, then returns a score and feedback.

Core Features:
- Uses a configurable LLM (OpenAI or HuggingFace-based) to evaluate generated answers.
- Parses response to extract evaluation text, score (1-10), and follow-up questions.
- Determines whether the answer needs improvement based on a score threshold.

Typical Usage:
    agent = VerifierAgent(qa_model="openai", threshold=7)
    feedback = agent.run({
        "question": "What is Tesla's expansion plan?",
        "merged_answer": "Tesla will open a factory in Singapore..."
    })
"""

import logging
import json
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from config.prompt_manager import prompt_manager
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

        # Use mode-aware prompt selection
        verification_prompt_template = prompt_manager.get_verification_prompt()
        prompt = PromptTemplate.from_template(verification_prompt_template)

        if qa_model == "openai":
            self.chain = prompt | ChatOpenAI(
                model_name="gpt-4o-mini", 
                temperature=0.3,            # Slightly increased for more nuanced verification
                max_tokens=2048,            # Increased for detailed evaluation
                top_p=0.95
            ) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            llm = pipeline("text2text-generation", model=model, tokenizer=processor.tokenizer, device_map="auto")
            self.chain = prompt | HuggingFacePipeline(pipeline=llm) | StrOutputParser()
        
        elif "gemini" in qa_model:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) | StrOutputParser()
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                raise e

        else:
            raise ValueError(f"Unknown QA model: {qa_model}")

    def run(self, input_data: dict) -> dict:
        """
        Evaluates the merged answer against the original user question using a scoring rubric.

        Args:
            input_data (dict): Dictionary with the following keys:
                - 'question' (str): The original user question.
                - 'merged_answer' (str): The answer to evaluate.

        Returns:
            dict: A dictionary containing:
                - 'score' (int): Score between 1 and 10.
                - 'needs_retry' (bool): True if score < threshold.
                - 'evaluation' (str): Explanation of the score.
                - 'merged_answer' (str): The original answer evaluated.
                - 'follow_up_questions' (list[str], optional): Suggestions for improvement if needed.
        """
        question = input_data.get("question", "")
        answer = input_data.get("merged_answer", "")
        logger.info("VerifierAgent scoring final merged answer.")

        try:
            response = self.chain.invoke({"question": question, "answer": answer})

            # Parse JSON response
            # Clean the response - remove any markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.split('```json')[1]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response.rsplit('```')[0]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_response)
                
                # Extract values with defaults
                score = result.get("score", 0)
                evaluation = result.get("evaluation", "No evaluation provided")
                follow_up_questions = result.get("follow_up_questions", [])
                
                # Ensure score is an integer
                if isinstance(score, str):
                    score = int(''.join(filter(str.isdigit, score))) if score else 0
                elif not isinstance(score, int):
                    score = int(score) if score else 0
                    
                # Ensure follow_up_questions is a list
                if not isinstance(follow_up_questions, list):
                    follow_up_questions = []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response}")
                
                # Fallback to old parsing method
                lines = response.strip().splitlines()
                
                score = 0
                evaluation = response.strip()
                
                # Look for score patterns
                for line in lines:
                    score_match = re.search(r'score[:\s]*(\d+)', line.lower())
                    if score_match:
                        score = int(score_match.group(1))
                        break
                
                # Extract follow-up questions
                follow_up_questions = []
                followup_section = False
                for line in lines:
                    if 'follow' in line.lower() and ('question' in line.lower() or 'up' in line.lower()):
                        followup_section = True
                        continue
                    if followup_section and line.strip().startswith('- '):
                        follow_up_questions.append(line.strip('- ').strip())

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

