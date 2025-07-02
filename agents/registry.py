# agents/registry.py
from agents.text_agent import TextRAGAgent
from agents.image_agent import ImageRAGAgent
from agents.generalize_agent import GeneralizeAgent
from agents.finalize_agent import FinalizeAgent

# Build agents once and reuse
AGENTS = {
    "TextRAGAgent": TextRAGAgent(),
    "ImageRAGAgent": ImageRAGAgent(),
    "GeneralizeAgent": GeneralizeAgent(),
    "FinalizeAgent": FinalizeAgent(),
}



