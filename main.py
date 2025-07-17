"""
main.py - Entry point for AgenticRAG Pipeline

This script serves as the main execution pipeline for the AgenticRAG system,
handling document ingestion, extraction, normalization, PDF merging, indexing, 
and RAG-based chat interaction using LangChain agents.

Features:
- Accepts both URLs and local file paths (PDF, DOCX, PPTX, HTML, CSV, MD).
- Uses Docling to extract structured content from documents.
- Normalizes all inputs to PDF format for unified downstream processing.
- Creates mappings for traceability between input and processed outputs.
- Merges PDFs into a single directory for indexing and retrieval.
- Supports ingestion-only mode or full interactive chat using agents.

Usage:
    python main.py --ingest      # Only ingest and index the documents
    python main.py --chat        # Ingest, index, and enter chat mode

Environment Variables:
    - Requires OPENAI_API_KEY in .env file

Logs:
    Logs are written to `agenticrag.log` and stdout each run, starting fresh.
"""
import logging
import sys

# === Create a fresh log file every time ===
LOG_FILE_PATH = "agenticrag.log"

# Remove any existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up new logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),  # Overwrite each run
        logging.StreamHandler(sys.stdout),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

import os
import json
import argparse
from dotenv import load_dotenv
from pathlib import Path
from utils.document_processor import DocumentProcessor, copy_pdfs_to_merge_dir
from pipeline.M3APipeline import M3APipeline
from pipeline.chat import run_chat
from config.agent_config import agent_config
from config.rag_config import rag_config

# # Uncomment to check if CUDA is available
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment.")
    exit(1)

# === Input list: local paths or URLs ===
input_items = [
    # "https://arxiv.org/pdf/1706.03762.pdf",
    "https://arxiv.org/pdf/2503.13964.pdf",
    "https://arxiv.org/pdf/2501.06322.pdf",
    # "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
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
    # "local/company_profiles_external_final.csv",
    "local/INTRO_PHASE2_2025.pptx",
    # "local/main_fig.jpg",
    "local/Text mining by using Python2025.docx"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Only ingest documents")
    parser.add_argument("--download", action="store_true", help="Only for download documents")
    parser.add_argument("--chat", action="store_true", help="Enable chat mode")
    parser.add_argument("--app", action="store_true", help="Run Streamlit app instead of CLI")
    return parser.parse_args()

def main():
    print("=== AgenticRAG Pipeline ===")
    args = parse_args() 
    
    # === Run Streamlit app ===
    if args.app:
        import subprocess
        subprocess.run(["streamlit", "run", "chat_streamlit.py"])
        return
    
    # === Downloading documents ===
    if args.download:
        logger.info("📥 Downloading documents...")
        STORE_DIR = Path("data/store")
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        processor = DocumentProcessor(store_dir=STORE_DIR, extract_dir="data/extract")
        input_to_downloaded, input_to_normalized = processor.process_all(input_items)

        with open("data/input_to_output_mapping.json", "w", encoding="utf-8") as f:
            json.dump(input_to_downloaded, f, indent=2, ensure_ascii=False)

        with open("data/input_to_normalized_mapping.json", "w", encoding="utf-8") as f:
            json.dump(input_to_normalized, f, indent=2, ensure_ascii=False)

        print("📘 Saved:")
        print(" - input_to_output_mapping.json (downloaded files)")
        print(" - input_to_normalized_mapping.json (final normalized PDFs)")
        
        # === Merge all PDFs into a single directory
        extract_pdf_dir = Path("data/extract/pdf")
        store_pdf_dir = Path("data/store")
        local_pdf_dir = Path("local")
        merge_pdf_dir = Path("data/merge")
        
        path_lst = [extract_pdf_dir, store_pdf_dir, local_pdf_dir]

        copy_pdfs_to_merge_dir(path_lst, merge_pdf_dir)

    # === Use merged folder for indexing and chat
    pdf_dir = "data/merge"
    index_dir = "data/merge/index"

    pipeline = M3APipeline(
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        agent_config=agent_config,
        rag_config=rag_config,
        ingest_only=not args.chat
    )   

    if args.ingest:
        pipeline.ingest_cfg()

    # how many agents are there in mdocagent architecture and what are they
    if args.chat:
        run_chat(pipeline)

if __name__ == "__main__":
    main()
