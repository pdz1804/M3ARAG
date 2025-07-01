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

            # Store output in memory with agent's name as key
            self.shared_memory[agent.name] = output

        # Return the final output from the last agent
        return output


