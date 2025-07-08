"""
agents/registry.py

This module serves as a centralized agent registry for the multi-agent RAG system.

It defines and exposes a dictionary `AGENTS` that maps agent names (as strings)
to their corresponding agent class implementations. This allows dynamic and
configurable agent instantiation in the `MultiAgentRunner`.

Registered Agents:
- "TextAgent": Handles textual context responses.
- "ImageAgent": Handles visual context responses (e.g., image-based reasoning).
- "GeneralizeAgent": Merges or synthesizes outputs from multiple agents.
- "FinalizeAgent": Produces the final answer for the user based on intermediate outputs.

Usage:
    from agents.registry import AGENTS
    agent_class = AGENTS["TextAgent"]
    agent_instance = agent_class(name="TextAgent", qa_model="gpt")
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.text_agent import TextAgent
from agents.image_agent import ImageAgent
from agents.generalize_agent import GeneralizeAgent
from agents.finalize_agent import FinalizeAgent

# Build agents once and reuse
# This later being called and initialized in the MultiAgentRunner
AGENTS = {
    "TextAgent": TextAgent,
    "ImageAgent": ImageAgent,
    "GeneralizeAgent": GeneralizeAgent,
    "FinalizeAgent": FinalizeAgent,
}



