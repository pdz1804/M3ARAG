import base64
import logging
from openai import OpenAI
from pathlib import Path

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Setup OpenAI Client ===
client = OpenAI()  # Automatically uses OPENAI_API_KEY from env

def generate_caption_with_openai(input_data: dict) -> str:
    """
    Uses GPT-4o-mini (OpenAI v1.x) to caption images based on the input query and image_paths.
    """
    query = input_data["query"]
    image_paths = input_data["image_paths"]

    logger.info(f"📥 Received query: {query}")
    logger.info(f"🔍 Retrieved {len(image_paths)} image(s):")
    
    for p in image_paths:
        logger.info(f"  - {p}")

    if not image_paths:
        logger.warning("⚠️ No images to caption.")
        return "❌ No images to caption."

    # === Prepare messages ===
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""You are given several research figures. Based on this query: {query}, describe the relevant information from each image. 
                            Remeber the cites the source of the information. If no relevant information is found, say 'No relevant information found.'"""
            }
        ]
    }]

    for path in image_paths:
        try:
            with open(path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "auto"
                }
            })
            logger.info(f"📎 Added image to prompt: {Path(path).name}")
        except Exception as e:
            logger.warning(f"❌ Failed to read image {path}: {e}")

    # === Call OpenAI API ===
    try:
        logger.info("📡 Calling OpenAI GPT-4o-mini API with image(s)...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        
        logger.info(f"📥 OpenAI response received: {response.choices[0].message.content[:100]}...")
        
        caption = response.choices[0].message.content.strip()
        logger.info(f"✅ OpenAI caption generated successfully.")
        return caption

    except Exception as e:
        logger.error(f"❌ OpenAI GPT-4o API call failed: {e}")
        return "❌ Failed to generate captions with OpenAI."
