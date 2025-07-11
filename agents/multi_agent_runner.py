"""
agents/multi_agent_runner.py

This module defines the `MultiAgentRunner` class, which coordinates the execution of
multiple specialized agents in a loop to perform multimodal retrieval-augmented generation (RAG)
with iterative refinement based on feedback from a VerifierAgent.

Responsibilities:
- Registers and manages multiple agents (TextAgent, ImageAgent, GeneralizeAgent, PlanningAgent, MergeAgent, VerifierAgent).
- Executes a Planning → Sub-Agent → Merge → Verify loop with score tracking.
- Performs multimodal context retrieval for each sub-query.
- Tracks merged answers across iterations for memory-aware cumulative merging.

Key Features:
- Agents dynamically registered from the AGENTS registry.
- Uses VerifierAgent feedback to decide if further iterations are needed.
- Reuses previous merged context in subsequent loops for richer generation.
- Plots score history across iterations for inspection.

Typical Usage:
    runner = MultiAgentRunner(rag_system, agent_config)
    runner.register_agent("TextAgent", qa_model="qwen")
    runner.register_agent("ImageAgent", qa_model="gemini")
    ...
    answer = runner.run("What are the strengths of Company X?")
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Optional
from agents.base import BaseAgent
from agents.registry import AGENTS
import matplotlib.pyplot as plt

class MultiAgentRunner:
    """
    Coordinates the execution of a multi-agent RAG pipeline with iterative improvement.

    Attributes:
        rag: The RAG system used for document/text/image retrieval.
        config: Dictionary of configuration parameters (e.g., max_loop, max_tasks).
        agents: Dictionary of registered agents.
        shared_memory: Common memory space for passing input context.
        memory_log: Stores merged answers from each loop iteration.
        score_history: List of verification scores for each loop iteration.
        all_sub_answers: List of all generalized answers from each sub-query across iterations.
        subquery_to_answer: Maps each sub-query to its generalized output.
    """
    def __init__(self, rag, config):
        self.score_history = []
        
        # --- new code ---
        self.rag = rag              # ← inject RAG system
        self.config = config        # ← agent_config
        self.agents = {}
        self.shared_memory = {}
        self.memory_log = []        # For storing the merged context across iterations
        
        self.max_loop = config.get("max_loop", 3)
        self.max_tasks = config.get("max_tasks", 5)
        # --- end new code ---
        
        self.all_sub_answers = []  # NEW

    def register_agent(self, agent_name: str, qa_model: str = "qwen"):
        """
        Registers an agent from the AGENTS registry.

        Args:
            agent_name (str): The name of the agent to register.
            qa_model (str): The underlying model the agent should use.
        """
        cls = AGENTS[agent_name]
        
        if agent_name == "VerifierAgent":
            agent_instance = cls(name=agent_name, qa_model=qa_model, threshold=self.config.get("threshold", 5))
        else:
            agent_instance = cls(name=agent_name, qa_model=qa_model)
            
        self.agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name} with model: {qa_model}")

    def run(self, question: str) -> str:
        """
        Executes the iterative agent workflow to answer a user question.

        Workflow:
            1. Plan sub-questions using the PlanningAgent.
            2. Retrieve and process context for each sub-question using Text/Image agents.
            3. Generate sub-answers using GeneralizeAgent.
            4. Merge all answers (including prior loop results) with MergeAgent.
            5. Verify merged answer using VerifierAgent.
            6. Loop until answer is satisfactory or max_loop is reached.

        Args:
            question (str): The user's original question.

        Returns:
            str: The final merged answer or a failure message.
        """
        self.score_history = []
        
        logger.info(f"💬 User Question: {question}")
        self.shared_memory["question"] = question
        self.memory_log = []  # Reset memory for this session

        # Initialize agents (if not done in __init__)
        planning_agent = self.agents["PlanningAgent"]
        merge_agent = self.agents["MergeAgent"]
        verifier_agent = self.agents["VerifierAgent"]

        loop_count = 0
        max_loop = self.max_loop
        max_tasks = self.max_tasks
        
        self.all_sub_answers = []
        self.subquery_to_answer = {}

        while loop_count < max_loop:
            logger.info(f"🔁 Iteration {loop_count+1}")
            # --- new code ---
            if loop_count == 0:
                sub_queries = planning_agent.run({"question": question})[:max_tasks]
            else:
                sub_queries = verification.get("follow_up_questions", [])[:max_tasks]
            # --- end new code ---
            
            # Remove duplicate subqueries
            sub_queries = [q for q in sub_queries if q not in self.subquery_to_answer]
            
            for sub_query in sub_queries:
                results = self.rag.retrieve_results(sub_query)
                text_results = results["text_results"]
                visual_results = results["visual_results"]

                out = self._run_sub_agents(sub_query, text_results, visual_results)
                
                self.subquery_to_answer[sub_query] = out
                self.all_sub_answers.append(out)

            # Merge and Verify: Combine current and past merged context
            if self.memory_log:
                prev_merged = self.memory_log[-1]
                self.all_sub_answers.append(prev_merged)
                
            merged_answer = merge_agent.run({"generalized_answers": self.all_sub_answers})
            self.memory_log.append(merged_answer)

            verification = verifier_agent.run({
                "question": question,
                "merged_answer": merged_answer
            })
            
            self.score_history.append(verification.get("score", 0))

            if not verification["needs_retry"]:
                logger.info("🎯 Final Answer:\n" + verification["merged_answer"])
                logger.info("🧠 Verifier's Reasoning:\n" + verification["evaluation"])
                print("\n✅ Final Answer:\n", verification["merged_answer"])
                print("\n🧠 Verifier's Evaluation:\n", verification["evaluation"])
                
                self._plot_score_history()
                
                return merged_answer
            else:
                loop_count += 1

        return "⚠️ Could not generate a satisfactory answer after multiple attempts."

    def _run_sub_agents(self, question, text_ctx, visual_ctx):
        local_memory = {"question": question}
        
        for name, agent in self.agents.items():
            logger.info(f"🧠 Running {name}")
            if name == "TextAgent":
                output = agent.run(local_memory, text_ctx)
            elif name == "ImageAgent":
                output = agent.run(local_memory, visual_ctx)
            elif name == "GeneralizeAgent":
                output = agent.run(local_memory)
            else:
                continue  # Skip other agents 
            
            local_memory[name] = output
            logger.info(f"✔ Output from {name}: {output[:500]}")
            
        return local_memory.get("GeneralizeAgent", "No answer.")

    def _plot_score_history(self):
        """
        Plots the verification score history over iterations and saves it to 'score_history.png'.
        """
        if not self.score_history:
            return
        plt.plot(range(1, len(self.score_history) + 1), self.score_history, marker='o')
        plt.title("Verifier Score Across Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("score_history.png")
        plt.close()
        logger.info("📈 Score plot saved to score_history.png")


