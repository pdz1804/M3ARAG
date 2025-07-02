# rag_image/loader.py

import re
from PyPDF2 import PdfReader
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
import torch

# === Setup logging ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_pdf_context_from_image_name(
    image_path: Path,
    pdf_dirs: list = [Path("data/extract/pdf"), Path("data/store")],
    num_lines: int = 5,
) -> str:
    """Extract top N lines of text from a PDF file corresponding to an image name."""
    # Extract the PDF stem directly before "-pic-"
    image_stem = image_path.stem
    if "-pic-" not in image_stem:
        print(f"❌ Cannot extract PDF name from image: {image_path.name}")
        return ""

    pdf_stem = image_stem.split("-pic-")[0]
    pdf_filename = f"{pdf_stem}.pdf"

    for pdf_dir in pdf_dirs:
        pdf_path = pdf_dir / pdf_filename
        if pdf_path.exists():
            try:
                reader = PdfReader(str(pdf_path))
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"
                    if len(full_text.splitlines()) >= num_lines * 2:
                        break

                # Clean and select lines
                lines = [line.strip() for line in full_text.splitlines() if line.strip()]
                context = "\n".join(lines[:num_lines])
                print(f"✅ Context extracted from: {pdf_path}")
                return context

            except Exception as e:
                print(f"❌ Failed to read {pdf_path.name}: {e}")
                return ""

    print(f"⚠️ PDF not found for: {pdf_filename}")
    return ""

def caption_images_to_documents(
    image_dir: str = "data/extract/imgs",
    max_new_tokens: int = 128,
    context_lines: int = 10
) -> List[Document]:
    logger.info(f"🖼️ Loading and captioning images from: {image_dir}")

    # === Load Qwen2.5-VL-3B in INT4 ===
    logger.info("📦 Loading Qwen2.5-VL-3B in INT4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # <-- fix here
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.eval()
    logger.info(f"✅ Model loaded on device: {model.device}")

    # === Load image paths ===
    img_dir = Path(image_dir)
    if not img_dir.exists():
        logger.warning(f"⚠️ Image directory does not exist: {img_dir}")
        return []

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        logger.warning(f"⚠️ No .png images found in: {img_dir}")
        return []

    docs = []

    logger.info(f"🔍 Found {len(img_paths)} image(s). Beginning captioning...")

    for path in tqdm(img_paths, desc="📝 Captioning images", ncols=80):
        try:
            # image = Image.open(path).convert("RGB")

            # messages = [{
            #     "role": "user",
            #     "content": [
            #         {"type": "image", "image": image},
            #         {"type": "text", "text": "Describe this image in detail. Focus on key elements, context, and any relevant information."}
            #     ],
            # }]
            
            image = Image.open(path).convert("RGB")
            context = get_pdf_context_from_image_name(path, num_lines=context_lines)

            base_prompt = "Describe this image in detail. Focus on key elements, context, and relevant insights."
            if context:
                logger.info(f"📚 Context found for {path.name} — injecting into prompt.")
                prompt = (
                    f"This image is from a research paper. Based on the following abstract/context:\n\n"
                    f"{context}\n\n{base_prompt}"
                )
            else:
                logger.warning(f"⚠️ No context found for {path.name} — using base prompt only.")
                prompt = base_prompt

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
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                trimmed_ids = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
                caption = processor.batch_decode(
                    trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

            logger.info(f"🖼️ [{path.name}] Caption: {caption[:60]}...")
            docs.append(Document(page_content=caption, metadata={"source": str(path)}))

        except Exception as e:
            logger.error(f"❌ Failed to process {path.name}: {e}")

    logger.info(f"✅ Captioning complete. {len(docs)} document(s) generated.")
    return docs
