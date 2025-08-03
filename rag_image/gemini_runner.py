# rag_image/image_captioning/gemini_runner.py
import logging
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from io import BytesIO
from config.prompt_manager import prompt_manager

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Gemini Client (auto uses GOOGLE_API_KEY) ===
try:
    client = genai.Client()
    logger.info("Gemini client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    client = None

def pil_to_part(pil_img: Image.Image):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return types.Part.from_bytes(buffer.getvalue(), mime_type="image/jpeg")

def generate_caption_with_gemini(input_data: dict) -> str:
    """
    Uses Gemini 2.0 Flash to caption images based on the input query and image_paths.
    """
    query = input_data.get("query", "")
    image_paths = input_data.get("images", [])
    metadata = input_data.get("metadata", [])

    logger.info(f"📥 Received query: {query}")
    logger.info(f"🔍 Retrieved {len(image_paths)} image(s):")
    
    for path in image_paths:
        logger.info(f"  - {path}")

    if not image_paths:
        logger.warning("No images to caption.")
        return "No images to caption."

    if client is None:
        return "Gemini client not initialized."

    try:
        # === Build contents for Gemini ===
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
        contents = [prompt]

        for i, path in enumerate(image_paths):
            try:
                if i == 0:
                    # Upload the first image as a reusable file
                    uploaded_file = client.files.upload(file=path)
                    contents.append(uploaded_file)
                    logger.info(f"Uploaded first image: {Path(path).name}")
                else:
                    # Embed other images as inline byte parts
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                    part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    contents.append(part)
                    logger.info(f"Added inline image: {Path(path).name}")
            except Exception as e:
                logger.warning(f"Failed to process image {path}: {e}")

        logger.info("Sending multimodal request to Gemini...")
        # --- old code --- 
        # response = client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=contents
        # )
        
        # --- new code ---
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config={
                "max_output_tokens": 512,  # or your preferred value
                "temperature": 0.6,         # Increased for more detailed responses  
                "top_p": 0.95,             # High nucleus sampling for comprehensive responses
                "top_k": 40,               # Moderate top-k for balance
                "max_output_tokens": 1024,  # Explicit token limit for detailed image analysis
            }
        )
        
        # --- end new code ---
        
        # --- old code ---
        # contents = [query]

        # for i, path in enumerate(image_paths):
        #     try:
        #         if i == 0:
        #             # Upload the first image as a file reference
        #             uploaded_file = client.files.upload(file=path)
        #             contents.append(uploaded_file)
        #             logger.info(f"📎 Uploaded image: {Path(path).name}")
        #         else:
        #             # Add subsequent images inline
        #             with open(path, "rb") as f:
        #                 img_bytes = f.read()
        #             contents.append(
        #                 types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        #             )
        #             logger.info(f"📎 Added inline image: {Path(path).name}")
        #     except Exception as e:
        #         logger.warning(f"Failed to process image {path}: {e}")

        # # === Generate response ===
        # logger.info("📡 Calling Gemini 2.0 Flash API...")
        # response = client.models.generate_content(
        #     model="gemini-2.0-flash",  # <-- Use latest version
        #     contents=contents,
        # )
        # --- end old code ---
        
        logger.info("Gemini caption generated.")
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return "Failed to generate captions with Gemini."
