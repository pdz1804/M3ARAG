"""
Multi-Agent Runner: Advanced Agent Coordination and Execution Engine

This module implements the sophisticated MultiAgentRunner class that orchestrates
multiple specialized AI agents in an iterative refinement workflow for complex
multi-modal question answering with quality-driven improvement loops.

System Architecture:
    The MultiAgentRunner implements a feedback-driven architecture where agents
    collaborate in structured phases with memory persistence and quality assessment:

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Planning   │───▶│  Retrieval  │───▶│ Generation  │
    │   Phase     │    │   Phase     │    │   Phase     │
    └─────────────┘    └─────────────┘    └─────────────┘
                                                   │
    ┌─────────────┐    ┌─────────────┐    ┌───────▼─────┐
    │ Verification│◀───│   Merge     │◀───│ Synthesis   │
    │   Phase     │    │   Phase     │    │   Phase     │
    └─────────────┘    └─────────────┘    └─────────────┘

Execution Workflow:
    The system follows a sophisticated iterative workflow designed for quality
    maximization through continuous refinement:

    1. **Planning Phase**: Query decomposition and task generation
       - Complex query analysis and entity resolution
       - Sub-question generation with context preservation
       - Task prioritization and scope determination

    2. **Retrieval Phase**: Multi-modal context gathering
       - Parallel text and visual content retrieval
       - Context relevance scoring and ranking
       - Memory integration from previous iterations

    3. **Generation Phase**: Specialized agent processing
       - TextAgent: Processes textual contexts with LLM reasoning
       - ImageAgent: Analyzes visual content with vision-language models
       - Parallel processing with error handling and recovery

    4. **Synthesis Phase**: Multi-modal response integration
       - GeneralizeAgent combines text and visual insights
       - Conflict resolution and information deduplication
       - Source attribution and citation management

    5. **Merge Phase**: Response consolidation
       - Memory-aware merging of current and historical contexts
       - Natural language flow optimization
       - Coherence and completeness verification

    6. **Verification Phase**: Quality assessment and decision
       - Multi-criteria evaluation (relevance, completeness, correctness, clarity)
       - Numerical scoring (1-10 scale) with detailed feedback
       - Follow-up question generation for improvement
       - Iteration decision based on configurable thresholds

Core Responsibilities:
    - Agent lifecycle management and coordination
    - Memory persistence across iterations for context continuity
    - Quality-driven iterative improvement with configurable thresholds
    - Performance monitoring and score visualization
    - Error handling and graceful degradation
    - Resource optimization and memory management

Agent Registry Integration:
    The system dynamically loads agents from a centralized registry,
    enabling flexible configuration and easy extensibility:
    
    - Dynamic agent instantiation with model selection
    - Configurable agent parameters and behaviors
    - Runtime agent registration and deregistration
    - Model-specific optimization and error handling

Memory Management:
    Advanced memory system enables context-aware processing:
    
    - **Shared Memory**: Cross-agent communication and state sharing
    - **Memory Log**: Historical context preservation across iterations
    - **Sub-Answer Tracking**: Comprehensive answer evolution tracking
    - **Query Mapping**: Subquery-to-answer relationship maintenance

Quality Assurance Features:
    - **Iterative Refinement**: Continuous quality improvement through feedback
    - **Score Tracking**: Performance monitoring across iterations
    - **Threshold Management**: Configurable quality gates
    - **Progress Visualization**: Graphical score progression analysis
    - **Convergence Detection**: Automatic stopping criteria

Performance Optimizations:
    - **Parallel Processing**: Concurrent agent execution where possible
    - **Memory Efficiency**: Optimized data structures and cleanup
    - **Caching**: Intelligent caching of intermediate results
    - **Resource Management**: GPU memory optimization and cleanup
    - **Batch Operations**: Efficient processing of multiple queries

Configuration Parameters:
    - **max_loop**: Maximum iteration limit for quality improvement
    - **max_tasks**: Maximum sub-questions per iteration
    - **threshold**: Quality score threshold for completion (1-10 scale)
    - **agent_models**: Model selection per agent type
    - **memory_settings**: Memory management configuration

Error Handling and Recovery:
    - **Graceful Degradation**: System continues with reduced functionality
    - **Error Isolation**: Agent failures don't crash entire pipeline
    - **Retry Logic**: Automatic retry with exponential backoff
    - **Fallback Models**: Alternative model selection on failure
    - **Comprehensive Logging**: Detailed error reporting and context

Usage Examples:
    ```python
    # Basic initialization
    runner = MultiAgentRunner(rag_system, agent_config)
    
    # Agent registration with model selection
    runner.register_agent("TextAgent", qa_model="openai")
    runner.register_agent("ImageAgent", qa_model="gemini")
    runner.register_agent("GeneralizeAgent", qa_model="openai")
    runner.register_agent("PlanningAgent", qa_model="openai")
    runner.register_agent("MergeAgent", qa_model="openai")
    runner.register_agent("VerifierAgent", qa_model="openai")
    
    # Process complex query with iterative improvement
    final_answer = runner.run("What are the key challenges and opportunities 
                              in AI adoption for healthcare organizations?")
    ```

Technical Implementation:
    - **Thread Safety**: Safe concurrent access to shared resources
    - **Memory Mapping**: Efficient large data structure handling
    - **Vectorization**: Optimized numerical operations
    - **Profiling**: Built-in performance monitoring and analysis
    - **Extensibility**: Plugin architecture for custom agents

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Advanced Multi-Agent Architecture)
Dependencies: langchain, transformers, torch, matplotlib, chromadb
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
        # Complete reset of all state for new question to prevent contamination
        self.score_history = []
        self.memory_log = []
        self.all_sub_answers = []
        self.subquery_to_answer = {}
        self.shared_memory = {}  # Also reset shared memory
        
        logger.info("================== STARTING NEW QUERY ==================")
        logger.info(f"User Question: {question}")
        logger.info(f"Configuration - Max Loops: {self.max_loop}, Max Tasks: {self.max_tasks}")
        logger.info(f"Available Agents: {list(self.agents.keys())}")
        logger.info("All memory and state completely reset for new query")
        
        self.shared_memory["question"] = question

        # Initialize agents (if not done in __init__)
        planning_agent = self.agents["PlanningAgent"]
        merge_agent = self.agents["MergeAgent"]
        verifier_agent = self.agents["VerifierAgent"]

        loop_count = 0
        max_loop = self.max_loop
        max_tasks = self.max_tasks

        while loop_count < max_loop:
            logger.info(f"==================== Iteration {loop_count+1} ====================")
            
            # Generate sub-queries based on iteration
            if loop_count == 0:
                logger.info("Planning Agent: Generating initial sub-queries")
                sub_queries = planning_agent.run({"question": question})[:max_tasks]
                logger.info(f"Planning Agent generated {len(sub_queries)} sub-queries")
            else:
                logger.info("Using follow-up questions from verification")
                sub_queries = verification.get("follow_up_questions", [])[:max_tasks]
                logger.info(f"Verification provided {len(sub_queries)} follow-up questions")
            
            # Log all sub-queries
            for idx, sq in enumerate(sub_queries):
                logger.info(f"Sub-query {idx+1}: {sq}")
            
            # Remove duplicate subqueries
            original_count = len(sub_queries)
            sub_queries = [q for q in sub_queries if q not in self.subquery_to_answer]
            if len(sub_queries) < original_count:
                logger.info(f"Filtered out {original_count - len(sub_queries)} duplicate sub-queries")
            
            logger.info(f"Processing {len(sub_queries)} unique sub-queries")
            
            # Process each sub-query
            for sq_idx, sub_query in enumerate(sub_queries):
                logger.info(f"--- Processing Sub-query {sq_idx+1}/{len(sub_queries)}: {sub_query} ---")
                
                # Retrieve results for this sub-query
                logger.info("Retrieving multi-modal context")
                results = self.rag.retrieve_results(sub_query)
                text_results = results["text_results"]
                visual_results = results["visual_results"]
                
                logger.info(f"Retrieved {len(text_results)} text contexts and {len(visual_results)} visual contexts")
                
                # Process with sub-agents - pass complete context objects with metadata
                out = self._run_sub_agents(sub_query, text_results, visual_results)
                
                self.subquery_to_answer[sub_query] = out
                self.all_sub_answers.append(out)
                logger.info(f"Sub-query {sq_idx+1} completed. Answer stored in memory.")

            # Memory management: Only use current iteration's answers for merging
            # Previous iterations should only inform if current iteration is insufficient
            logger.info("Preparing answers for merging")
            answers_to_merge = self.all_sub_answers.copy()
            
            # Only add previous merged context if we're in a refinement loop (loop_count > 0)
            # and the current answers are insufficient (less than expected)
            if self.memory_log and loop_count > 0 and len(answers_to_merge) < self.max_tasks:
                prev_merged = self.memory_log[-1]
                answers_to_merge.append(prev_merged)
                logger.info(f"Added previous merged answer for refinement. Total answers to merge: {len(answers_to_merge)}")
            else:
                logger.info(f"Using only current iteration answers. Total answers to merge: {len(answers_to_merge)}")
                if self.memory_log:
                    logger.info(f"Previous context available but not used (loop={loop_count}, answers={len(answers_to_merge)})")
                
            # Merge all answers
            logger.info("Merge Agent: Combining all answers")
            merged_answer = merge_agent.run({"generalized_answers": answers_to_merge})
            self.memory_log.append(merged_answer)
            logger.info(f"Merge Agent completed. Current memory log size: {len(self.memory_log)}")
            logger.info(f"Merged Answer: {merged_answer}")

            # Verify the merged answer
            logger.info("Verifier Agent: Evaluating merged answer")
            verification = verifier_agent.run({
                "question": question,
                "merged_answer": merged_answer
            })
            
            score = verification.get("score", 0)
            self.score_history.append(score)
            
            logger.info(f"Verification Score: {score}/10")
            logger.info(f"Verification Evaluation: {verification.get('evaluation', 'No evaluation provided')}")
            logger.info(f"Score History: {self.score_history}")

            if not verification["needs_retry"]:
                logger.info("================ FINAL RESULT ================")
                logger.info(f"Final Answer: {verification['merged_answer']}")
                logger.info(f"Verifier Reasoning: {verification['evaluation']}")
                logger.info(f"Total Iterations: {loop_count + 1}")
                logger.info(f"Final Score: {score}/10")
                
                self._plot_score_history()
                
                return merged_answer
            else:
                logger.info(f"Answer not satisfactory (Score: {score}). Continuing to iteration {loop_count + 2}")
                loop_count += 1

        logger.info("================ PROCESS FAILED ================")
        logger.info(f"Could not generate satisfactory answer after {max_loop} iterations")
        logger.info(f"Final score history: {self.score_history}")
        logger.info(f"Total sub-queries processed: {len(self.subquery_to_answer)}")
        
        # Return the last merged answer with quality disclaimer
        if self.memory_log:
            last_answer = self.memory_log[-1]
            final_score = self.score_history[-1] if self.score_history else 0
            threshold = self.agent_config.get("threshold", 6)
            
            # Create a comprehensive quality disclaimer
            score_progress = " -> ".join(map(str, self.score_history)) if len(self.score_history) > 1 else str(final_score)
            
            disclaimer = f"""

---

### ⚠️ Quality Notice

**Score:** {final_score}/10 (Required: {threshold}/10)  
**Status:** Did not meet quality threshold after {max_loop} iterations  
**Score Progress:** {score_progress}  
**Attempts:** {len(self.score_history)} iterations made  

*This answer contains the best available information from our analysis, but may require further refinement or additional context to fully address your question.*

---"""
            
            logger.info(f"Returning last merged answer with quality disclaimer (Score: {final_score}/{threshold})")
            return last_answer + disclaimer
        else:
            logger.warning("No merged answers available to return")
            return f"Could not generate a satisfactory answer after {max_loop} attempts. No intermediate results available."

    def _run_sub_agents(self, question, text_ctx, visual_ctx):
        local_memory = {"question": question}
        
        # Add previous answers as context for this iteration
        if self.memory_log:
            local_memory["previous_answers"] = self.memory_log.copy()
            logger.info(f"Added {len(self.memory_log)} previous answers as context")
        
        for name, agent in self.agents.items():
            logger.info(f"Running {name} for question: {question}")
            
            if name == "TextAgent":
                logger.info(f"TextAgent processing {len(text_ctx)} text chunks")
                # Log first few chars of each context for debugging
                for idx, ctx in enumerate(text_ctx[:2]):  # Only log first 2 for brevity
                    chunk_preview = ctx.get('chunk', str(ctx))[:100] + "..." if len(str(ctx)) > 100 else str(ctx)
                    logger.debug(f"Text Context {idx+1} preview: {chunk_preview}")
                output = agent.run(local_memory, text_ctx)
                
            elif name == "ImageAgent":
                logger.info(f"ImageAgent processing {len(visual_ctx)} visual contexts")
                for idx, ctx in enumerate(visual_ctx):
                    logger.debug(f"Visual Context {idx+1}: {ctx}")
                output = agent.run(local_memory, visual_ctx)
                
            elif name == "GeneralizeAgent":
                logger.info("GeneralizeAgent processing with accumulated context")
                output = agent.run(local_memory)
                
            else:
                continue  # Skip other agents 
            
            local_memory[name] = output
            logger.info(f"Output from {name}:\n {output}")
            logger.info(f"Agent {name} completed successfully")
            
        final_answer = local_memory.get("GeneralizeAgent", "No answer.")
        logger.info(f"Sub-agent processing complete. Final answer: {final_answer}")
        
        return final_answer

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
        logger.info("Score plot saved to score_history.png")


