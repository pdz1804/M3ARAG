"""
rag_text/text_captioning/qwen_runner.py

This module defines the text captioning function for Qwen2.5-VL, used within the RAG system
to answer user queries based on retrieved text chunks. It formats the inputs into a prompt,
runs inference using a locally loaded Qwen2.5-VL model, and returns the generated answer.

Function:
- generate_caption_with_qwen(input_data: dict) -> str:
    Accepts a user question and associated text context, formats them using a shared prompt template,
    then runs the Qwen model to generate a concise answer.

Key Features:
- Loads the Qwen2.5-VL model and processor only once (singleton via `get_qwen_vl_model_and_processor()`).
- Formats prompts using `TEXT_PROMPT_TEMPLATE` from `config.prompt`.
- Uses Hugging Face-style `.generate()` and `.batch_decode()` to obtain the final output.
- Logs input details and supports GPU execution with memory offloading.

Usage:
    from rag_text.text_captioning.qwen_runner import generate_caption_with_qwen
    answer = generate_caption_with_qwen({"query": "What is Mamba?", "texts": "..."})
"""

import torch
import os
import logging
from config.prompt import TEXT_PROMPT_TEMPLATE
from utils.model_utils import get_qwen_vl_model_and_processor

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Offload directory setup ===
# os.makedirs("offload", exist_ok=True)
# torch.set_default_dtype(torch.bfloat16)

def generate_caption_with_qwen(input_data: dict) -> str:
    query = input_data["query"]
    context = input_data["texts"]
    logger.info(f"[Qwen] Received query: {query}")

    # for p in image_paths:
    #     logger.info(f"  - {p}")

    # === Load Qwen2.5-VL (only once) ===
    textual_model, textual_processor = get_qwen_vl_model_and_processor()

    # === Begin captioning ===
    # answers = []

    prompt = TEXT_PROMPT_TEMPLATE.format(query=query, context=context)

    messages = [
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    text = textual_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = textual_processor(text=[text], padding=True, return_tensors="pt")
    inputs = inputs.to(textual_model.device)

    print("\n--- Inputs to Textual Model.generate() ---")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"{k}: type={type(v)}, value={v}")
    print("----------------------------------\n")

    generated_ids = textual_model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = textual_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    inputs.to("cpu")
    return output_text[0]

