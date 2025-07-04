import os
import torch
from PIL import Image
from pathlib import Path
from transformers import BitsAndBytesConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from huggingface_hub import model_info

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.models import ColPali, ColPaliProcessor

# === Step 1: Load ColQwen2 in INT4 mode using bitsandbytes ===
# model_name = "vidore/colqwen2-v1.0"
model_name="vidore/colpali-v1.3"

print("📦 Loading ColPali model in INT8 mode with BitsAndBytes...")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     llm_int8_skip_modules=None,
#     bnb_4bit_compute_dtype=torch.bfloat16  # ✅ Match input dtype
# )

# model = ColQwen2.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",  # auto chooses available GPU
#     attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
# ).eval()

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                               # Enable 8-bit quantization
    llm_int8_threshold=6.0,                          # Threshold for deciding whether a layer is quantized (default: 6.0)
    llm_int8_has_fp16_weight=False,                  # Whether model checkpoint has fp16 weights (set to False unless you’re sure)
    # llm_int8_enable_fp32_cpu_offload=True,           # Offload large modules to CPU in FP32 to save VRAM
    llm_int8_verbose=True,                           # Log detailed quantization info per layer
)

model = ColPali.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",  # no meta
    trust_remote_code=True,  # if required for custom models
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    low_cpu_mem_usage=False,  # prevent meta device
).eval()

print(f"✅ Model loaded on device: {model.device}")

# === Step 2: Load Processor ===
print("🔧 Loading processor...")

# processor = ColQwen2Processor.from_pretrained(model_name)
processor = ColPaliProcessor.from_pretrained(model_name)

print("✅ Processor ready.")

# === Step 2.1: Check Memory Usage ===

from transformers import AutoModel
from huggingface_hub import model_info

info = model_info("vidore/colqwen2-v1.0")
print(f"Model name: {info.modelId}")
print(f"Model size on disk: {model.num_parameters() / 1e6:.2f}M parameters")

print(f"🧠 Allocated VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"🧠 Max VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# === Step 3: Load Your Real Images ===
# print("🖼️ Loading paper figure images...")
# img_paths = [
#     "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-1.png",
#     "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-2.png",
#     "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-3.png",
#     "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-4.png",
#     "../data/extract/imgs/arxiv_org_pdf_1706_03762_pdf-pic-5.png",
# ]
# images = [Image.open(path).convert("RGB") for path in img_paths]
# print(f"✅ Loaded {len(images)} images.")

print("🖼️ Loading paper figure images from folder...")
img_dir = "../data/extract/imgs"
img_paths = sorted([
    os.path.join(img_dir, f)
    for f in os.listdir(img_dir)
    if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
])
images = [Image.open(path).convert("RGB") for path in img_paths]
print(f"✅ Loaded {len(images)} images from: {img_dir}")

# === Step 4: Define Queries about Transformer Architecture ===
# print("📝 Building semantic queries...")
# queries = [
#     "What does the architecture of a Transformer model look like?",
#     "Where is the self-attention mechanism shown in the diagram?",
#     "Can you identify the encoder-decoder structure in these figures?",
#     "What layers are repeated in the Transformer architecture?",
#     "Where is positional encoding represented in these images?",
# ]
# print(f"✅ {len(queries)} queries prepared.")

# === Define queries: Transformer + MDocAgent + Multi-agent Collab ===
queries = [
    # --- Transformer-related ---
    "What does the architecture of the Dual Attention looks like?"
]

print(f"🧠 Prepared {len(queries)} semantic queries.")

# # === Step 5: Process Inputs ===
# print("🧪 Processing image batch...")
# batch_images = processor.process_images(images).to(model.device)
# print("📐 Image batch tensor shapes:", {k: v.shape for k, v in batch_images.items()})

# print("🧪 Processing query batch...")
# batch_queries = processor.process_queries(queries).to(model.device)
# print("📐 Query batch tensor shapes:", {k: v.shape for k, v in batch_queries.items()})

# # === Step 6: Forward Pass ===
# print("🚀 Running model forward pass...")
# with torch.no_grad():
#     print("🔄 Extracting image embeddings...")
#     image_embeddings = model(**batch_images)
#     print("✅ Image embeddings shape:", image_embeddings.shape)

#     print("🔄 Extracting query embeddings...")
#     query_embeddings = model(**batch_queries)
#     print("✅ Query embeddings shape:", query_embeddings.shape)

# # === Step 7: Score Similarity ===
# print("📊 Scoring image-query relevance...")
# scores = processor.score_multi_vector(query_embeddings, image_embeddings)
# print("✅ Score matrix computed.")
# print("📈 Similarity Scores Matrix:")
# print(scores)

# # === Step 8: Optional: Display Top Matches
# print("\n🔍 Top match per query:")
# for i, query in enumerate(queries):
#     best_img_idx = torch.argmax(scores[i]).item()
#     best_score = scores[i][best_img_idx].item()
#     print(f"Q{i+1}: '{query}' → Best Match: Image {best_img_idx + 1} with score {best_score:.4f}")




