# downloader.py
"""Handle file download and caching with robust naming based on URL content and structure."""

import requests
import magic
from pathlib import Path
from urllib.parse import urlparse
import re
import pdfkit
from tempfile import NamedTemporaryFile
import time
from playwright.sync_api import sync_playwright

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
    "text/csv": ".csv",
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
            content_type = response.headers.get("Content-Type", "")

            # --- old code ---
            # Generate clean, unique filename
            # filename = get_filename_from_url(path_or_url, content)
            # local_path = store_dir / filename
            
            # --- New code ---
            # HTML page — use pdfkit to render and save PDF
            # if "text/html" in content_type:
            #     filename = sanitize_filename(path_or_url) + ".pdf"
            #     local_path = store_dir / filename
            #     if not local_path.exists():
            #         print(f"🌐 Rendering HTML to PDF: {path_or_url} → {filename}")
            #         pdfkit.from_url(path_or_url, str(local_path))
            #         return local_path, True
            #     else:
            #         return local_path, False

            # Other (like PDF, DOCX, etc.)
            content = response.content
            filename = get_filename_from_url(path_or_url, content)
            local_path = store_dir / filename
            
            # --- end new code ---

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

def render_html_with_playwright(
    html_path: str,
    file_path: Path,
    output_dir: Path,
    zoom_factor: float = 0.9
):
    """
    Load the HTML file in a headless browser, apply zoom, emit a PDF with selectable text
    and a full-page screenshot.

    Args:
        html_path: Path to the downloaded .html file
        output_dir: Directory to write .pdf and .png into
        zoom_factor: e.g. 0.5 for 50% zoom, 0.75 for 75%, 1.0 for default
    Returns:
        (pdf_path, png_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(html_path).stem if not html_path.startswith("http") else Path(html_path.split("://", 1)[1]).stem
    pdf_path = output_dir / f"{stem}.pdf"
    # png_path = output_dir / f"{html_path}.png"


    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )

        page = context.new_page()

        time.sleep(0.5)

        # Optionally adjust viewport size if desired
        # page.set_viewport_size({"width": 1280, "height": 800})

        # load either the URL or the local file
        if html_path.startswith("http"):
            page.goto(html_path)
        else:
            page.goto(f"file://{Path(file_path).resolve()}")

        page.wait_for_load_state("load")
        time.sleep(1)

        if zoom_factor != 1.0:
            page.evaluate(f"document.body.style.zoom = '{zoom_factor}'")

        # export a text‑layer PDF
        page.pdf(path=str(pdf_path), format="A4", print_background=True)

    print(f"🖨️ Rendered HTML → PDF + screenshot: {pdf_path.name}")
    return pdf_path


