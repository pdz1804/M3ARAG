# main.py
"""Main pipeline: load file list + check/store files + extract content using Docling + run text-based RAG on extracted documents.."""
import os
import argparse
from dotenv import load_dotenv

import logging
from pathlib import Path
from utils.process_documents import process_documents


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
    # "https://arxiv.org/pdf/1706.03762.pdf",
    # "https://arxiv.org/pdf/2503.13964.pdf",
    # "https://arxiv.org/pdf/2501.06322.pdf",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    # "https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)",
    # "https://www.apple.com/",
    # "https://www.ibm.com/us-en",
    # "https://www.hp.com/us-en/home.html",
    # "https://www.dell.com/en-vn",
    # "https://en.wikipedia.org/wiki/Mixture_of_experts",
    # "http://westwaterresources.net/investors/industry-terms/",
    # "local/[VN-Team] 2025-07-03_Report.pdf",
    # "local/2022_MT_KHMT.pdf",
    # "local/2501.02189v6_Survey of SOTA LVLM.pdf",
    "local/company_profiles_external_final.csv",
    # "local/INTRO_PHASE2_2025.pptx",
    # "local/main_fig.jpg",
    # "local/Text mining by using Python2025.docx",
    # "local/2025-07-03.docx"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true", help="Enable text-based retrieval")
    parser.add_argument("--image", action="store_true", help="Enable image-based retrieval")
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
    
    # from app.chat import start_rag_chat
    # start_rag_chat(use_text=args.text, use_image=args.image)

if __name__ == "__main__":
    main()
