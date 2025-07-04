from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# === Step 1: Load Model in INT4 ===
print("📦 Loading model in INT4 mode...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,     # Use double quantization for better performance
    bnb_4bit_quant_type="nf4",          # Use the NF4 quantization type
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"✅ Model loaded on device: {model.device}")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# === Step 2: Load Processor ===
print("🔧 Loading processor...")

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print("✅ Processor loaded.")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# === Step 1: Load local images ===
print("🖼️ Loading local images...")
img_paths = [
    "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-1.png",
    "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-2.png",
    "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-3.png",
    "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-4.png",
    "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-5.png",
]
images = [Image.open(path).convert("RGB") for path in img_paths]

# === Step 2: Create message ===
print("📝 Building messages...")

messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": img} for img in images] + [
            {"type": "text", "text": "Summarize the key visual patterns across all these figures."}
        ],
    }
]

# # === Step 3: Define Input Messages === 
# print("🖼️ Preparing message with image + prompt...")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# === Step 4: Tokenize Prompt & Prepare Vision Inputs ===
print("🧠 Applying chat template...")

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("🖼️ Processing visual information...")
image_inputs, video_inputs = process_vision_info(messages)

print("📥 Converting to model inputs...")
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

print("✅ Inputs ready on device.")

# Inference: Generation of the output
# === Step 5: Generate Response ===
print("🌀 Generating output...")
generated_ids = model.generate(**inputs, max_new_tokens=128)
print("✅ Generation complete.")

# === Step 6: Decode Output ===
print("📝 Decoding output...")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# === Step 7: Display Result ===
print("🎉 Final Output:")
print(output_text[0])
