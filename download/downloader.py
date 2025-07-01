# downloader.py
"""Handle file download and caching with robust naming based on URL content and structure."""

import requests
import magic
from pathlib import Path
from urllib.parse import urlparse
import re

MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "text/markdown": ".md",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}

def guess_extension_from_bytes(byte_data: bytes) -> str:
    """Guess MIME type and return proper extension from file content."""
    mime = magic.from_buffer(byte_data[:2048], mime=True)
    return MIME_TO_EXT.get(mime, "")

def sanitize_filename(s: str) -> str:
    """Convert URL path into a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

def get_filename_from_url(url: str, content_bytes: bytes) -> str:
    """Generate a safe and unique filename from a URL and content type."""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path.strip("/")
    ext = guess_extension_from_bytes(content_bytes)

    name_part = sanitize_filename(f"{domain}_{path}") or "downloaded_file"
    if not name_part.endswith(ext):
        name_part += ext or ".bin"
    return name_part

def download_if_needed(path_or_url: str, store_dir: Path) -> tuple[Path, bool]:
    """
    Download the file if it doesn't exist. Return:
        - local_path (Path)
        - was_downloaded (bool): True if just downloaded, False if already exists
    """
    store_dir.mkdir(parents=True, exist_ok=True)

    # === Remote URL ===
    if path_or_url.startswith("http"):
        try:
            response = requests.get(path_or_url, timeout=30)
            response.raise_for_status()
            content = response.content

            # Generate clean, unique filename
            filename = get_filename_from_url(path_or_url, content)
            local_path = store_dir / filename

            if not local_path.exists():
                print(f"🌐 Downloading: {path_or_url} → {filename}")
                local_path.write_bytes(content)
                print(f"📥 Saved to: {local_path}")
                return local_path, True  # just downloaded
            else:
                print(f"📂 File already exists: {local_path}")
                return local_path, False  # already exists

        except Exception as e:
            raise RuntimeError(f"Download failed for {path_or_url}: {e}")

    # === Local file ===
    local_path = Path(path_or_url)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file does not exist: {local_path}")
    
    return local_path.resolve(), False

