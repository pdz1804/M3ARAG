"""
agents/image_agent.py

This module defines the ImageAgent class, responsible for handling visual question answering (VQA)
using input images and a natural language query. It supports multiple backends such as Qwen2.5-VL,
Gemini 2.0 Flash, and OpenAI GPT-4o-mini. 

The agent prepares and resizes input images, offloads GPU memory if needed, and dispatches the
images to the appropriate backend-specific captioning runner. Each backend is abstracted through
a unified interface, enabling clean modular usage.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from utils.image_utils import prepare_images, release_memory
from rag_image.base_runner import get_image_captioning_runner

import gc
import torch
from PIL import Image
import tempfile

class ImageAgent(BaseAgent):
    def __init__(self, name: str = "ImageAgent", qa_model: str = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing ImageAgent with backend model: {qa_model}")

        # Dynamically load the correct runner
        self.caption_with_llm = get_image_captioning_runner(qa_model)

    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")
        
        if not contexts:
            logger.warning("No contexts passed to ImageAgent.")
            return "No image context provided."
        
        images = [ctx["image"] for ctx in contexts if "image" in ctx]
        
        image_paths = prepare_images(images)
        release_memory()
        
        return self.caption_with_llm.invoke({
            "query": question,
            "images": image_paths   # ← Now Gemini expects file paths, not PIL.Image
        })



