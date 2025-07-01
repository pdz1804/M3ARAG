# main.py
"""Main pipeline: load file list + check/store files + extract content using Docling + run text-based RAG on extracted documents.."""
import os
import argparse
from dotenv import load_dotenv

import logging
from pathlib import Path
from utils.process_documents import process_documents
from app.chat import start_rag_chat

# # Uncomment to check if CUDA is available
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

logging.basicConfig(level=logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment.")
    exit(1)

# === Input list: local paths or URLs ===
input_items = [
    "https://arxiv.org/pdf/1706.03762.pdf",
    "https://arxiv.org/pdf/2503.13964.pdf",
    "https://arxiv.org/pdf/2501.06322.pdf",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)",
    "https://www.apple.com/",
    "https://www.ibm.com/us-en",
    "https://www.hp.com/us-en/home.html",
    "https://us.msi.com/",
    "https://www.dell.com/en-vn",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Enable multi-agent answering mode")
    return parser.parse_args()

def main():
    print("=== AgenticRAG Pipeline ===")
    # python main.py --agent
    args = parse_args() 
    
    # === Paths ===
    STORE_DIR = Path("data/store")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    process_documents(input_items, STORE_DIR)
    
    # === Set to True to use agent-based answering (TextAgent → Generalize → Finalize)
    # === Agent mode toggle via CLI
    start_rag_chat(use_agent=args.agent)

if __name__ == "__main__":
    main()
