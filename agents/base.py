"""
agents/base.py

Defines the abstract base class for all agents in the multi-agent RAG system.

The `BaseAgent` class serves as a template for specialized agents (e.g., TextAgent, ImageAgent, etc.)
that participate in multimodal reasoning and generation tasks. Each concrete agent must implement
the `run()` method with its own logic based on the input data and selected QA model.

Attributes:
    - name (str): Identifier for the agent.
    - qa_model (str): Name of the underlying model used for answering queries (default: "qwen").

Method:
    - run(input_data: dict) -> str: Abstract method that must be overridden by all subclasses.
"""
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, qa_model: str = "qwen"):
        self.qa_model = qa_model
        self.name = name

    @abstractmethod
    def run(self, input_data: dict) -> str:
        """Run the agent with input data and return output string."""
        pass
