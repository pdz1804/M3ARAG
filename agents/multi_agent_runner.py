"""
agents/multi_agent_runner.py

This module defines the `MultiAgentRunner` class, which coordinates the execution of
multiple specialized agents in a sequence to perform multimodal retrieval-augmented generation (RAG).

Responsibilities:
- Registers and manages multiple agents (e.g., TextAgent, ImageAgent, FinalizeAgent).
- Maintains a shared memory dictionary (`shared_memory`) used to pass intermediate outputs between agents.
- Dispatches visual and textual context to the appropriate agents.
- Ensures ordered execution and final response aggregation.

Key Features:
- Agents are dynamically registered from the `AGENTS` registry.
- Each agent contributes to shared reasoning via `shared_memory`.
- Includes fallback handling when both Text and Image agents fail to generate useful output.

Usage:
    runner = MultiAgentRunner()
    runner.register_agent("TextAgent", qa_model="gpt")
    runner.register_agent("ImageAgent", qa_model="gemini")
    final_output = runner.run({"question": "What is X?"}, visual_ctx, text_ctx)
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Optional
from agents.base import BaseAgent
from agents.registry import AGENTS

class MultiAgentRunner:
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.shared_memory: Dict[str, str] = {}

    def register_agent(self, agent: str, qa_model: str = "qwen"):
        cls = AGENTS[agent]
        agent = cls(name=agent, qa_model=qa_model)
        self.agents.append(agent)
        logger.info(f"Registered agent: {agent.name} with model: {qa_model}")

    def run(self, initial_input: Dict[str, str],
            visual_contexts: Optional[List[dict]] = None,
            textual_contexts: Optional[List[dict]] = None) -> str:
        
        # a place to call this method is in pipeline/M3APipeline.py
        # initial_input = {"question": "What is the capital of France?"} ==> this would be shared
        # When update is called:
        # 1. If the key already exists, it updates the value.
        # 2. If the key does not exist, it adds the key-value pair.
        logger.info(f"Running MultiAgentPipeline for question: {initial_input.get('question', 'N/A')}")
        self.shared_memory.update(initial_input)

        for agent in self.agents:
            logger.info(f"* Running {agent.name}...")

            if agent.name == "ImageAgent" and visual_contexts is not None:
                logger.info(f"Visual Contexts:")
                for ctx in visual_contexts:
                    img = ctx['image']
                    logger.info(f"{ctx['document_id']} | Page {ctx['page_number']} | "
                                f"Score {ctx['score']:.2f} | Size {img.size[0]}x{img.size[1]}")
                output = agent.run(self.shared_memory, visual_contexts)
                logger.info(f"Output from {agent.name}:\n{output[:1000]}\n")

            elif agent.name == "TextAgent" and textual_contexts is not None:
                logger.info("Textual Contexts:")
                for ctx in textual_contexts:
                    preview = ctx['chunk'][:200].replace('\n', ' ').strip()
                    logger.info(f"{ctx['chunk_pdf_name']} | Page {ctx['pdf_page_number']} | "
                                f"Score {ctx['score']:.2f} | Preview: {preview}...")
                output = agent.run(self.shared_memory, textual_contexts)
                logger.info(f"Output from {agent.name}:\n{output}\n")

            else:
                output = agent.run(self.shared_memory)
                logger.info(f"Output from {agent.name}:\n{output}\n")

            # - Agents to share intermediate outputs (e.g., GeneralizeAgent might read from both "TextAgent" and "ImageAgent" keys).
            # - A centralized shared_memory store for agent-to-agent communication.
            # - You to debug easily by printing the full memory.
            # - Only contains the latest output from each agent.
            # self.shared_memory = {
            #     "question": "What is Mamba in deep learning?",
            #     "TextAgent": "Mamba is a state space model proposed by Meta AI...",
            #     "ImageAgent": "No relevant diagrams found.",
            #     "GeneralizeAgent": "Mamba is a sequence modeling approach combining...",
            #     "FinalizeAgent": "Mamba is a state-of-the-art model architecture introduced by Meta AI in 2024 for efficient sequence modeling tasks."
            # }
            self.shared_memory[agent.name] = output
            
            # Special logic after Text and Image RAG agents
            if agent.name == "ImageAgent":
                self.image_answer = output.strip()

            if agent.name == "TextAgent":
                self.text_answer = output.strip()

        # === Fallback: if both text and image failed ===
        text_empty = not getattr(self, "text_answer", "").strip() or self.text_answer.lower() == "No answer found."
        image_empty = not getattr(self, "image_answer", "").strip() or self.image_answer.lower() == "No answer found."

        if text_empty and image_empty:
            logger.warning("Both TextRAG and ImageRAG returned empty or useless responses.")
            return "No answer found."

        # Return output from final agent
        logger.info(f"Final output returned from last agent: {self.agents[-1].name}")
        return output


