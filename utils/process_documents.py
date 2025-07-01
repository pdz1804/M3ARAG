# utils/process_documents.py

import logging
from pathlib import Path
from download.downloader import download_if_needed
from extract.extractor import extract_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_documents(input_items, store_dir="data/store"):
    """Step 1: Download + extract all items."""
    
    # ✅ Ensure required folders exist
    # Path(store_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Processing {len(input_items)} items...")
    
    for item in input_items:
        print(f"🔍 Checking item: {item}")
        try:
            # Check file existence or download it
            local_path, was_downloaded = download_if_needed(item, store_dir)
            
            print(f"📥 Processing {local_path.name}... (Downloaded: {was_downloaded})")
            
            # ✅ Only extract if it was just downloaded
            if was_downloaded:
                extract_file(local_path)
            else:
                print(f"⏩ Skipped extract for already existing file: {local_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to process {item}: {e}")
            
