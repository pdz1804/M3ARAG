# agents/base.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, input_data: dict) -> str:
        """Run the agent with input data and return output string."""
        pass
