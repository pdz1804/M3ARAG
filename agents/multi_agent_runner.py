# agents/multi_agent_runner.py

from typing import List, Dict
from agents.base import BaseAgent

class MultiAgentRunner:
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.shared_memory: Dict[str, str] = {}

    def register_agent(self, agent: BaseAgent):
        self.agents.append(agent)

    def run(self, initial_input: Dict[str, str]) -> str:
        self.shared_memory.update(initial_input)

        for agent in self.agents:
            print(f"🤖 Running {agent.name}...")
            output = agent.run(self.shared_memory)
            print(f"🧠 Output from {agent.name}:\n{output[:500]}\n{'-'*80}")

            self.shared_memory[agent.name] = output

            # Special logic after Text and Image RAG agents
            if agent.name == "ImageRAGAgent":
                self.image_answer = output.strip()
            if agent.name == "TextRAGAgent":
                self.text_answer = output.strip()

        # === Fallback: if both text and image failed ===
        text_empty = not getattr(self, "text_answer", "").strip() or self.text_answer.lower() == "no answer found."
        image_empty = not getattr(self, "image_answer", "").strip() or self.image_answer.lower() == "no answer found."

        if text_empty and image_empty:
            print("⚠️ Both TextRAG and ImageRAG returned empty or useless responses.")
            return "No answer found."

        # Return output from final agent
        return output


