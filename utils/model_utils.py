"""
utils/qwen_loader.py

This module provides a singleton loader for the Qwen2.5-VL-3B-Instruct model and its processor,
optimized with 8-bit quantization using BitsAndBytes for efficient memory usage.

Functionality:
- Lazily loads the Qwen2.5 Visual Language model and processor only once.
- Reuses the same model and processor instances across multiple calls.
- Uses AutoProcessor and Qwen2_5_VLForConditionalGeneration from Hugging Face Transformers.
- Configured for low-memory environments with bfloat16 + 8-bit inference using `BitsAndBytesConfig`.

Returns:
    Tuple of (`Qwen2_5_VLForConditionalGeneration`, `AutoProcessor`)
"""


from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
import torch


_model: Qwen2_5_VLForConditionalGeneration = None
_processor: AutoProcessor = None

def get_qwen_vl_model_and_processor():
    """
    Lazily load and return a singleton Qwen2.5-VL model + processor.
    Subsequent calls reuse the same objects.
    """
    global _model, _processor
    if _model is None or _processor is None:
        # — load model
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        #
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_name = r"unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
        print(f"📦 Loading {model_name} model & processor…")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()

        _processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=640 * 28 * 28
        )
        print("✅ Loaded shared Qwen2.5-VL model + processor")
    return _model, _processor

