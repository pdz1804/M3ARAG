import logging
from pathlib import Path
import os
from google import genai
from google.genai import types

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Gemini Client (auto uses GOOGLE_API_KEY) ===
try:
    client = genai.Client()
    logger.info("✅ Gemini client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini client: {e}")
    client = None


def generate_caption_with_gemini(input_data: dict) -> str:
    """
    Uses Gemini 2.0 Flash to caption images based on the input query and image_paths.
    """
    query = input_data.get("query", "")
    image_paths = input_data.get("image_paths", [])

    logger.info(f"📥 Received query: {query}")
    logger.info(f"🔍 Retrieved {len(image_paths)} image(s):")
    for p in image_paths:
        logger.info(f"  - {p}")

    if not image_paths:
        logger.warning("⚠️ No images to caption.")
        return "❌ No images to caption."

    if client is None:
        return "❌ Gemini client not initialized."

    try:
        # === Prepare content parts ===
        contents = [query]

        for i, path in enumerate(image_paths):
            try:
                if i == 0:
                    # Upload the first image as a file reference
                    uploaded_file = client.files.upload(file=path)
                    contents.append(uploaded_file)
                    logger.info(f"📎 Uploaded image: {Path(path).name}")
                else:
                    # Add subsequent images inline
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                    contents.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                    )
                    logger.info(f"📎 Added inline image: {Path(path).name}")
            except Exception as e:
                logger.warning(f"❌ Failed to process image {path}: {e}")

        # === Generate response ===
        logger.info("📡 Calling Gemini 2.0 Flash API...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # <-- Use latest version
            contents=contents,
        )
        logger.info("✅ Gemini caption generated.")
        return response.text.strip()

    except Exception as e:
        logger.error(f"❌ Gemini API call failed: {e}")
        return "❌ Failed to generate captions with Gemini."
