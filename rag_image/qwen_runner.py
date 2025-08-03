# rag_image/image_captioning/qwen_runner.py
import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import logging
from config.prompt_manager import prompt_manager
from utils.model_utils import get_qwen_vl_model_and_processor

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Offload directory setup ===
# os.makedirs("offload", exist_ok=True)

# === Global model states ===
# torch.set_default_dtype(torch.bfloat16)


def generate_caption_with_qwen(input_data: dict) -> str:

    query = input_data["query"]
    images = input_data["images"]
    metadata = input_data.get("metadata", [])

    logger.info(f"Image agent Received query: {query}")
    logger.info(f"Retrieved {len(images)} image(s):")

    # for p in image_paths:
    #     logger.info(f"  - {p}")

    # === Load Qwen2.5-VL (only once) ===
    vision_model, vision_processor = get_qwen_vl_model_and_processor()

    # === Begin captioning ===
    # answers = []

    # Use mode-aware prompt selection
    prompt_template = prompt_manager.get_image_prompt()
    
    # Build context information from metadata for proper citations
    context_info = ""
    if metadata:
        context_info = "\n\nSource Context for Citations:\n"
        for i, meta in enumerate(metadata):
            if isinstance(meta, dict):
                doc_name = meta.get("source", "Unknown Document")
                page_num = meta.get("page", "Unknown Page")
                context_info += f"Image {i+1}: [Page {page_num}, Doc: {doc_name}]\n"
            else:
                context_info += f"Image {i+1}: {str(meta)}\n"
        context_info += "\nWhen referencing specific images, use the format [Page X, Doc: DocumentName] for proper citation."
    
    prompt = prompt_template.format(query=query) + context_info

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
        }
    ]

    text = vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = vision_processor(text=[text], images=images, padding=True, return_tensors="pt")
    inputs = inputs.to(vision_model.device)

    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    generated_ids = vision_model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                             zip(inputs.input_ids, generated_ids)]
    output_text = vision_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    inputs.to("cpu")

    return output_text[0]

