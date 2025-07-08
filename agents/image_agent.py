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

        # --- old code ---
        # del contexts
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # target_size = (512, 1024)
        # resized_images = []
        # for img in images:
        #     # Create a copy to avoid modifying the original PIL image object directly
        #     img_copy = img.copy()

        #     # Use thumbnail to resize while maintaining aspect ratio,
        #     # making sure the largest dimension fits within target_size.
        #     # Image.Resampling.LANCZOS is a high-quality downsampling filter.
        #     img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
        #     resized_images.append(img_copy)
            
        # # === Save to temp files for Gemini ===
        # image_paths = []
        # for i, img in enumerate(resized_images):
        #     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        #         img.save(tmp.name, format="JPEG")
        #         image_paths.append(tmp.name)
        
        # --- end old code ---
        
        image_paths = prepare_images(images)
        release_memory()
        
        return self.caption_with_llm.invoke({
            "query": question,
            "images": image_paths   # ← Now Gemini expects file paths, not PIL.Image
        })



