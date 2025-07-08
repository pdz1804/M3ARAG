"""
rag_text/text_captioning/openai_runner.py

This module defines a text captioning function using the OpenAI GPT-4o-mini model,
integrated into the RAG system for answering questions based on retrieved textual context.

Function:
- generate_caption_with_openai(input_data: dict) -> str:
    Formats a prompt using a shared template (`TEXT_PROMPT_TEMPLATE`) and sends it
    to OpenAI's GPT-4o-mini model for generation. Returns the model's response as a string.

Key Features:
- Uses `langchain_core.RunnableLambda` (via base_runner.py) for compatibility with LangChain.
- Supports dynamic question and context inputs.
- Logs all queries and handles API errors gracefully.
- Requires a valid `OPENAI_API_KEY` in the environment or `.env` file.

Usage:
    from rag_text.text_captioning.openai_runner import generate_caption_with_openai
    answer = generate_caption_with_openai({"query": "What is Mamba?", "texts": "..."})
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from config.prompt import TEXT_PROMPT_TEMPLATE

# Setup logging and client
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment or .env file.")

# === Create OpenAI Client ===
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_caption_with_openai(input_data: dict) -> str:
    query = input_data.get("query", "")
    context = input_data.get("texts", "")
    logger.info(f"[OpenAI] Received query: {query}")

    prompt = TEXT_PROMPT_TEMPLATE.format(query=query, context=context)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant for document understanding and question answering."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return "OpenAI failed to generate a response."



