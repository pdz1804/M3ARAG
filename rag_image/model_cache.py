# rag_image/model_cache.py

from functools import lru_cache
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import BitsAndBytesConfig
from transformers.utils.import_utils import is_flash_attn_2_available

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_copali_model_and_processor(model_name="vidore/colqwen2-v1.0"):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     llm_int8_enable_fp32_cpu_offload=True
    # )
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,                         # Switch from 4-bit to 8-bit quantization
    #     llm_int8_enable_fp32_cpu_offload=True      # Optional: offload large layers to CPU
    # )
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,                               # Enable 8-bit quantization
        llm_int8_threshold=6.0,                          # Threshold for deciding whether a layer is quantized (default: 6.0)
        llm_int8_has_fp16_weight=False,                  # Whether model checkpoint has fp16 weights (set to False unless you’re sure)
        llm_int8_enable_fp32_cpu_offload=True,           # Offload large modules to CPU in FP32 to save VRAM
        llm_int8_skip_modules=None,                      # Optional: list of module names to skip quantization (e.g., ['lm_head'])
        llm_int8_verbose=True                            # Log detailed quantization info per layer
    )

    model = ColQwen2.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor


