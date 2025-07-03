import torch
from pathlib import Path
from PIL import Image
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os
import logging

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Offload directory setup ===
os.makedirs("offload", exist_ok=True)

# === Global model states ===
model = None 
processor = None
fallback_model = None
fallback_processor = None

def generate_caption_batch(input_data: dict) -> str:
    global model, processor, fallback_model, fallback_processor

    query = input_data["query"]
    image_paths = input_data["image_paths"]

    logger.info(f"📥 Received query: {query}")
    logger.info(f"🔍 Retrieved {len(image_paths)} image(s):")
    
    for p in image_paths:
        logger.info(f"  - {p}")

    # === Load Qwen2.5-VL (only once) ===
    if model is None:
        logger.info("📦 Loading Qwen2.5-VL-3B model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            quantization_config=bnb_config,
            device_map={"": "cpu"},  # ✅ safer default
        ).eval()
        logger.info("✅ Qwen2.5-VL model loaded")

    if processor is None:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        logger.info("✅ Qwen2.5 processor loaded")

    # === Begin captioning ===
    answers = []

    for img_path in image_paths:
        try:
            logger.info(f"🖼️ Processing image: {img_path}")
            image = Image.open(img_path).convert("RGB")

            prompt = f"You are given a research figure. Based on the query: {query}\nDescribe the image."
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                trimmed_ids = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
                caption = processor.batch_decode(
                    trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

            logger.info(f"✅ Caption generated: {caption}")
            answers.append(f"[{Path(img_path).name}]: {caption}")

        except Exception as e:
            logger.warning(f"❌ Qwen2.5-VL failed for {img_path}: {e}")

            # === Fallback: BLIP ===
            try:
                if fallback_model is None or fallback_processor is None:
                    logger.info("🔁 Falling back to BLIP-large...")
                    fallback_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                    fallback_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
                    logger.info("✅ BLIP model loaded")

                inputs = fallback_processor(images=image, return_tensors="pt").to("cpu")
                with torch.no_grad():
                    out = fallback_model.generate(**inputs, max_new_tokens=50)
                    caption = fallback_processor.decode(out[0], skip_special_tokens=True)

                logger.info(f"✅ Fallback caption: {caption}")
                answers.append(f"[{Path(img_path).name} - Fallback]: {caption}")

            except Exception as fb_err:
                logger.error(f"❌ Fallback BLIP model also failed for {img_path}: {fb_err}")
                answers.append(f"[{Path(img_path).name}]: <Failed to caption>")

    final_output = "\n\n".join(answers)
    logger.info("📝 Final combined output:\n" + final_output)
    return final_output


