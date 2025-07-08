# utils/image_utils.py
import torch
import gc
from PIL import Image
import tempfile
from typing import List
import logging

logger = logging.getLogger(__name__)

def prepare_images(images: List[Image.Image], target_size=(512, 1024)) -> List[str]:
    """
    Resize images and save to temporary files.

    Returns:
        List[str]: Paths to resized temporary JPEG image files.
    """
    if not images:
        logger.warning("No images provided for processing.")
        return []

    resized_paths = []

    for img in images:
        try:
            img_copy = img.copy()
            img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img_copy.save(tmp.name, format="JPEG")
                resized_paths.append(tmp.name)
        except Exception as e:
            logger.error(f"Failed to process image: {e}")

    return resized_paths

def release_memory():
    """Clean up GPU and Python memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
