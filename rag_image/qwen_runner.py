# rag_image/image_captioning/qwen_runner.py
import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import logging
from config.prompt import IMAGE_PROMPT_TEMPLATE
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

    logger.info(f"Image agent Received query: {query}")
    logger.info(f"Retrieved {len(images)} image(s):")

    # for p in image_paths:
    #     logger.info(f"  - {p}")

    # === Load Qwen2.5-VL (only once) ===
    vision_model, vision_processor = get_qwen_vl_model_and_processor()

    # === Begin captioning ===
    # answers = []

    prompt = IMAGE_PROMPT_TEMPLATE.format(query=query)

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

