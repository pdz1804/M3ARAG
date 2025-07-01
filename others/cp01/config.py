import os

# === Paths ===
PDF_PATH = os.path.join("data", "pdfs")
IMAGE_DIR = os.path.join("data", "imgs")

TEXT_DB_PATH = os.path.join("vectorstores", "text_db")
IMAGE_DB_PATH = os.path.join("vectorstores", "image_db")

# === Models ===
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4.1-nano-2025-04-14" # "gpt-4o-mini-2024-07-18"

# === Chunking ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
TOP_K = 3

# === Prompts ===
system_prompt = """
You are an intelligent AI assistant with expert-level understanding of Machine Learning and Deep Learning.
Your task is to answer questions by reasoning over both text and image evidence.
You can use tools to retrieve relevant text or image descriptions and then combine them to form your final answer.
Please always cite the document name or image name in your final answer for clarity.

When invoking tools, always pass the entire user question as the `query` argument to the tool, include the full sentence or question for accurate retrieval.
Your goal is to maximize retrieval relevance by using the full user input as context for the tool call.
"""


