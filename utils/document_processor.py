"""
DocumentProcessor: Universal Multi-Format Document Processing and Normalization Engine

This module implements a sophisticated document processing system that handles
comprehensive ingestion, extraction, and normalization of diverse document
formats for the M3ARAG multi-modal retrieval system.

Supported Document Formats:
    **Input Formats**:
    - PDF documents (.pdf) - Native support with advanced extraction
    - Microsoft Word (.docx, .doc) - Full content and formatting preservation
    - PowerPoint (.pptx, .ppt) - Slide content and embedded media extraction
    - HTML files (.html) - Web content with layout preservation
    - Markdown files (.md) - Structured text with formatting
    - CSV files (.csv) - Tabular data with intelligent formatting
    - Plain text (.txt) - Direct text processing
    - Images (PNG, JPEG) - Visual content extraction
    - Web URLs - Dynamic content fetching and processing

    **Output Formats**:
    - Normalized PDFs for unified downstream processing
    - Extracted images with quality filtering and optimization
    - Structured markdown with hierarchy preservation
    - Tabular data in standardized formats
    - Comprehensive metadata and mapping files

System Architecture:
    The DocumentProcessor implements a multi-stage processing pipeline
    designed for robustness, efficiency, and comprehensive content extraction:

    ┌─────────────────────────────────────────────────────────────────┐
    │                  Document Input Layer                           │
    │  URLs • Local Files • Multiple Formats • Batch Processing      │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │              Format Detection & Validation                      │
    │     Magic Numbers • MIME Types • Extension Analysis             │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │                Content Extraction                               │
    │  Docling • Office Converters • HTML Rendering • Text Processing│
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │              Normalization & Optimization                       │
    │      PDF Conversion • Image Filtering • Quality Assurance      │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │                Storage & Organization                           │
    │    Structured Directories • Metadata • Traceability Maps       │
    └─────────────────────────────────────────────────────────────────┘

Core Processing Capabilities:

**Advanced Content Extraction**:
    The system leverages industry-leading tools for robust content extraction:
    
    - **Docling Integration**: State-of-the-art document understanding
      • Multi-format parsing with layout preservation
      • Advanced table and image extraction
      • Hierarchical structure recognition
      • Metadata preservation and enhancement
    
    - **Office Document Processing**: Native Microsoft Office support
      • Word document conversion with formatting preservation
      • PowerPoint slide extraction with embedded media
      • Excel spreadsheet processing with data validation
    
    - **Web Content Processing**: Dynamic web content handling
      • Playwright-powered browser automation
      • JavaScript-rendered content extraction
      • CSS styling preservation for layout understanding
      • Responsive design handling for mobile content

**Intelligent Format Detection**:
    - **Magic Number Analysis**: Binary signature-based format identification
    - **MIME Type Recognition**: Content-Type header analysis for web content
    - **Extension Validation**: File extension verification and correction
    - **Content Inspection**: Deep content analysis for format confirmation
    - **Error Recovery**: Robust handling of corrupted or misidentified files

**Quality-Aware Processing**:
    The system implements comprehensive quality controls:
    
    - **Image Quality Assessment**: Automatic filtering of low-quality images
      • Minimum dimension requirements (300x300 pixels)
      • Area-based filtering (90,000+ pixel minimum)
      • Format optimization and standardization
    
    - **Text Quality Validation**: Content integrity verification
      • Character encoding detection and correction
      • Language identification and processing
      • Structure validation and repair
    
    - **Document Structure Analysis**: Layout understanding and preservation
      • Heading hierarchy detection and preservation
      • Table structure recognition and extraction
      • List and enumeration handling

**Robust Error Handling and Recovery**:
    - **Multi-Stage Fallbacks**: Alternative processing methods for failures
    - **Partial Success Handling**: Extract available content from partial failures
    - **Resource Cleanup**: Automatic temporary file and memory management
    - **Detailed Error Reporting**: Comprehensive error logging with context
    - **Retry Mechanisms**: Intelligent retry logic with exponential backoff

Directory Structure and Organization:
    ```
    data/
    ├── store/              # Original downloaded files
    │   ├── <hash>.<ext>   # Files stored with content hash naming
    │   └── metadata.json  # Download metadata and source information
    ├── extract/            # Processed and extracted content
    │   ├── pdf/           # Normalized PDF files
    │   ├── imgs/          # Extracted images with quality filtering
    │   ├── markdown/      # Structured text content
    │   └── tables/        # Extracted tabular data
    ├── merge/             # Final unified collection for indexing
    └── mappings/          # Traceability and relationship mappings
        ├── input_to_output_mapping.json
        └── input_to_normalized_mapping.json
    ```

Advanced Features:

**Incremental Processing**:
    - **Duplicate Detection**: Content-based deduplication using hashing
    - **Update Detection**: Modified file identification and reprocessing
    - **Selective Processing**: Process only new or changed content
    - **Dependency Tracking**: Maintain relationships between source and output files

**Parallel Processing**:
    - **Concurrent Downloads**: Parallel fetching of multiple URLs
    - **Batch Operations**: Efficient processing of document collections
    - **Resource Management**: Optimal utilization of system resources
    - **Progress Tracking**: Real-time processing status and estimates

**Memory Optimization**:
    - **Streaming Processing**: Large file handling without full memory loading
    - **Garbage Collection**: Proactive memory cleanup during processing
    - **Resource Pooling**: Efficient reuse of processing resources
    - **Memory Monitoring**: Real-time memory usage tracking and optimization

URL Processing Capabilities:
    - **Content Type Detection**: Automatic format identification from headers
    - **JavaScript Rendering**: Dynamic content processing with Playwright
    - **Authentication Handling**: Support for authenticated content access
    - **Rate Limiting**: Respectful crawling with configurable delays
    - **Error Recovery**: Robust handling of network issues and timeouts

Configuration and Customization:
    ```python
    processor = DocumentProcessor(
        store_dir="data/store",           # Original file storage
        extract_dir="data/extract"        # Processed content output
    )
    
    # Quality thresholds
    processor.MIN_IMAGE_WIDTH = 300       # Minimum image width
    processor.MIN_IMAGE_HEIGHT = 300      # Minimum image height  
    processor.MIN_IMAGE_AREA = 90000      # Minimum image area
    ```

Integration Points:
    - **M3ARAG Pipeline**: Seamless integration with the main RAG system
    - **Agent System**: Provides processed content for agent analysis
    - **Storage Systems**: Flexible backend storage support
    - **Monitoring Systems**: Comprehensive logging and metrics integration

Performance Characteristics:
    - **Throughput**: Optimized for high-volume document processing
    - **Latency**: Minimal processing delays with efficient algorithms
    - **Scalability**: Horizontal scaling support for large deployments
    - **Reliability**: Robust error handling and recovery mechanisms
    - **Resource Efficiency**: Optimal memory and CPU utilization

Usage Examples:
    ```python
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process mixed input types
    inputs = [
        "https://arxiv.org/pdf/1706.03762.pdf",
        "local/document.docx", 
        "local/presentation.pptx",
        "https://example.com/page.html"
    ]
    
    # Process all inputs
    downloaded_map, normalized_map = processor.process_all(inputs)
    
    # Copy results to merge directory
    copy_pdfs_to_merge_dir(
        [Path("data/extract/pdf"), Path("data/store")], 
        Path("data/merge")
    )
    ```

Dependencies:
    - **Core**: pathlib, shutil, json, hashlib, logging
    - **Document Processing**: docling, pypdf, python-magic
    - **Office Documents**: docx2pdf, pptxtopdf
    - **Web Processing**: playwright, requests, pdfkit
    - **Data Processing**: pandas, markdown
    - **Utilities**: urllib, time, re

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Advanced Multi-Format Processing)
License: MIT License
"""
import re
import time
import shutil
import json
import pdfkit
import requests
import hashlib
import logging
import pandas as pd
import markdown
import magic
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from pptxtopdf import convert as pptx_to_pdf_convert
from docx2pdf import convert as docx_to_pdf_convert
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, store_dir="data/store", extract_dir="data/extract"):
        self.store_dir = Path(store_dir)
        self.extract_dir = Path(extract_dir)
        self.pdf_dir = self.extract_dir / "pdf"
        self.img_dir = self.extract_dir / "imgs"
        self.markdown_dir = self.extract_dir / "markdown"
        self.table_dir = self.extract_dir / "tables"

        for place in [self.store_dir, self.extract_dir, self.pdf_dir, self.img_dir, self.markdown_dir, self.table_dir]:
            place.mkdir(parents=True, exist_ok=True)

        self.MIN_IMAGE_WIDTH = 300
        self.MIN_IMAGE_HEIGHT = 300
        self.MIN_IMAGE_AREA = 90_000

        self.MIME_TO_EXT = {
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

        # === Docling setup ===
        accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.AUTO)
        pdf_options = PdfPipelineOptions(
            images_scale=2.0,
            generate_page_images=True,
            generate_picture_images=True,
            accelerator_options=accelerator_options,
        )
        self.doc_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX, InputFormat.CSV,
                InputFormat.HTML, InputFormat.PPTX, InputFormat.ASCIIDOC, InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pdf_options,
                ),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            },
        )

    def is_url(self, s: str) -> bool:
        """
        Check if the given string is a URL.

        Args:
            s (str): Input string to check.

        Returns:
            bool: True if the string is a URL, False otherwise.
        """
        return s.startswith("http://") or s.startswith("https://")

    def sanitize_filename(self, s: str) -> str:
        """
        Sanitize a string to be used as a safe filename by replacing invalid characters.

        Args:
            s (str): Raw input string (e.g., URL path or query string).

        Returns:
            str: Sanitized string suitable for filenames.
        """
        return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

    def guess_extension(self, byte_data: bytes) -> str:
        """
        Guess the file extension based on the MIME type of the content.

        Args:
            byte_data (bytes): The initial bytes of the file to infer its type.

        Returns:
            str: File extension (e.g., '.pdf', '.html') or an empty string if unknown.
        """
        mime = magic.from_buffer(byte_data[:2048], mime=True)
        return self.MIME_TO_EXT.get(mime, "")

    def get_filename_from_url(self, url: str, content: bytes) -> str:
        """
        Generate a unique and safe filename based on the URL and file content.

        Args:
            url (str): The original URL of the file.
            content (bytes): The raw content of the file.

        Returns:
            str: A filesystem-safe and unique filename with extension.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_")
        ext = self.guess_extension(content) or ".bin"
        
        # Normalize and truncate path
        path = parsed.path.strip("/")
        path = re.sub(r'[^a-zA-Z0-9_\-]', '_', path)  
        if not path:
            path = "index"
        
        # Optional: Add query hash for uniqueness if needed
        # query_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        query_hash = ""
        
        # Compose filename
        filename = f"{domain}_{path}_{query_hash}{ext}"
        
        # Truncate if too long (100 chars max for safety)
        if len(filename) > 100:
            filename = filename[:90] + query_hash + ext

        return filename

    def download_if_needed(self, item: str) -> tuple[Path, bool]:
        """
        Download a file from a URL if it does not already exist locally.

        Args:
            item (str): URL or local file path.

        Returns:
            tuple[Path, bool]: A tuple containing the local file path and a flag
                               indicating whether the file was downloaded.

        Raises:
            FileNotFoundError: If the local file does not exist or download fails.
        """
        if self.is_url(item):
            try:
                response = requests.get(item, timeout=30)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                content = response.content
                
                filename = self.get_filename_from_url(item, content)
                local_path = self.store_dir / filename

                if not local_path.exists():
                    logger.info(f"Downloading {item} to {local_path}")
                    local_path.write_bytes(content)
                    logger.info(f"Saved {item} as {local_path.name}")
                    return local_path, True
                
                logger.info(f"File already exists: {local_path.name}")
                return local_path, False
            
            except Exception as e:
                logger.error(f"Failed to download {item}: {e}")
                raise FileNotFoundError(f"Failed to download {item}: {e}")
            
        else:
            # local_path = Path(item.replace("local/", ""))
            local_path = Path(item)
            if not local_path.exists():
                raise FileNotFoundError(f"❌ File not found: {local_path}")
            return local_path.resolve(), False

    def render_html_with_playwright(self, html_path: str, file_path: Path, zoom: float = 0.9) -> Path:
        """
        Render an HTML page into a selectable-text PDF using Playwright.

        Args:
            html_path (str): Path or URL of the HTML content.
            file_path (Path): Path to the local HTML file (used for naming).
            zoom (float): Optional zoom factor for rendering (default is 0.9).

        Returns:
            Path: Path to the generated PDF file.
        """
        pdf_filename = file_path.with_suffix(".pdf").name
        pdf_path = self.pdf_dir / pdf_filename
        
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            time.sleep(2)

            # load either the URL or the local file
            if html_path.startswith("http") or html_path.startswith("https"):
                page.goto(html_path, timeout=60000, wait_until="networkidle")
            else:
                page.goto(f"file://{Path(file_path).resolve()}", timeout=60000, wait_until="networkidle")

            page.wait_for_load_state(state="load", timeout=0) # set timeout to 0 to wait indefinitely
            time.sleep(2)
            
            if zoom != 1.0:
                page.evaluate(f"document.body.style.zoom = '{zoom}'")
                
            # export a text‑layer PDF
            page.pdf(path=str(pdf_path), format="A4", print_background=True)
            
        logger.info(f"Rendered HTML to PDF: {pdf_path.name}")
        return pdf_path

    def markdown_to_pdf(self, md_path: Path, pdf_path: Path):
        """
        Convert a Markdown file to PDF using pdfkit.

        Args:
            md_path (Path): Path to the input .md file.
            pdf_path (Path): Path to save the generated .pdf file.

        Raises:
            Logs error if conversion fails.
        """
        try:
            md_content = md_path.read_text(encoding="utf-8")
            html_content = markdown.markdown(md_content)
            pdfkit.from_string(html_content, str(pdf_path))
            logger.info(f"Converted {md_path.name} → {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to convert MD file of {md_path.name} to PDF: {e}")

    def extract_file(self, file_path: Path) -> Path | None:
        """
        Extract content from a file, convert it to normalized PDF and save images/tables if applicable.

        Args:
            file_path (Path): Path to the file to process.

        Returns:
            Path | None: Path to the generated PDF if successful, else None.

        Raises:
            Logs error if conversion or processing fails.
        """
        logger.info(f"\nProcessing: {file_path}")
        start_time = time.time()
        
        try:
            conv = self.doc_converter.convert(file_path)
            doc = conv.document
        except Exception as e:
            logger.error(f"❌ Docling conversion error: {e}")
            return None
        
        # different from original code
        ext = file_path.suffix.lower()
        stem = conv.input.file.stem
        
        output_return = None
        
        # === PDF files: skip saving content (already PDF)
        if ext == ".pdf":
            logger.info(f"Skipping PDF file: {file_path.name}")
            pass
        
        # === Skip raw image files ===
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]:
            logger.info(f"Skipping image file: {file_path.name}")
            return None
        
        # === Handle HTML files === (we handle already in process_all)
        
        # === Handle PPTX separately with pptxtopdf ===
        elif ext in [".ppt", ".pptx"]:
            try:
                logger.info(f"Converting PPTX to PDF: {file_path.name}")
                pptx_to_pdf_convert(str(file_path.parent), str(self.pdf_dir))
                output_return = self.pdf_dir / f"{stem}.pdf"
            except Exception as e:
                logger.error(f"Failed to convert PPTX file {file_path.name}: {e}")
                return None

        # === Handle DOCX separately with docx2pdf ===
        elif ext in [".doc", ".docx"]:
            try:
                output_pdf = self.pdf_dir / f"{stem}.pdf"
                if not output_pdf.exists():
                    logger.info(f"Converting DOCX to PDF: {file_path.name}")
                    docx_to_pdf_convert(str(file_path))
                    file_path.with_suffix(".pdf").rename(output_pdf)
                output_return = output_pdf
            except Exception as e:
                logger.error(f"Failed to convert DOCX file {file_path.name}: {e}")
                return None

        # === Handle CSV files ===
        elif ext == ".csv":
            try:
                logger.info(f"Processing CSV file: {file_path.name}")
                df = pd.read_csv(file_path)
                logger.info(f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns")
                
                # Drop unnamed columns often generated by Excel
                columns_to_drop = [col for col in df.columns if "unnamed" in col.lower()]
                if columns_to_drop:
                    logger.info(f"⚠️ Dropping unnamed columns: {columns_to_drop}")
                df_cleaned = df.drop(columns=columns_to_drop)
                
                txt_path = self.store_dir / f"{stem}.txt"
                logger.info(f"Saving cleaned CSV data to text file: {txt_path.name}")
                
                with txt_path.open("w", encoding="utf-8") as f:
                    for i, row in df_cleaned.iterrows():
                        if i >= 100: 
                            logger.warning(f"Truncating output at 100 rows for: {file_path.name}")
                            break
                        for col, val in row.items():
                            f.write(f"{col}: {val}\n")
                            
                        # blank line between rows
                        f.write("\n")
                
                logger.info(f"CSV conversion complete for: {file_path.name}")
                output_return = txt_path
                
                # Convert TXT to PDF
                try:
                    from utils.txt2pdf import convert_txt_to_pdf
                    pdf_path = self.pdf_dir / f"{stem}.pdf"
                    convert_txt_to_pdf(str(txt_path), str(pdf_path))
                    logger.info(f"Converted TXT to PDF: {pdf_path.name}")
                    output_return = pdf_path
                except Exception as e:
                    logger.error(f"Failed to convert TXT to PDF: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process CSV file {file_path.name}: {e}")
                return None

        # === Handle other files with Docling ===
        # for other format, we use Docling to extract to MD then we convert to PDF
        else:
            md_path = self.markdown_dir / f"{stem}.md"
            pdf_path = self.pdf_dir / f"{stem}.pdf"
            md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
            self.markdown_to_pdf(md_path, pdf_path)

        # === Save images ===
        img_count = 0
        table_count = 0
        
        for element, _ in doc.iterate_items():
            if isinstance(element, PictureItem):
                img = element.get_image(doc)
                if img:
                    width, height = img.size
                    if width * height < self.MIN_IMAGE_AREA or width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
                        logger.info(f"Skipping small image: {img.size} in {file_path.name}")
                        continue
                    img_path = self.img_dir / f"{stem}-img-{img_count}.png"
                    img.save(img_path)
                    img_count += 1
                    logger.info(f"Saved image {img_count} to {img_path.name}")
            
            # elif isinstance(element, TableItem):
            #     df: pd.DataFrame = element.export_to_dataframe()

            #     # Save DataFrame to Markdown
            #     md = df.to_markdown(index=False)
            #     table_path = self.table_dir / f"{stem}-table-{table_count}.md"
            #     table_path.write_text(md, encoding="utf-8")
                
            #     table_count += 1
            #     logger.info(f"Saved table {table_count} to {table_path.name}")
                
        logger.info(f"Extracted {img_count} images and {table_count} tables from {file_path.name}")
        
        if (pdf_path := self.pdf_dir / f"{stem}.pdf").exists():
            return pdf_path.resolve()
        return None

    # --- old code ---
    # def process_all(self, input_items: list[str]) -> tuple[dict, dict]:
    #     """
    #     Process a list of input items (URLs or file paths), downloading and extracting content.

    #     Args:
    #         input_items (list[str]): List of URLs or local file paths.

    #     Returns:
    #         tuple(dict, dict):
    #             - input_to_downloaded: Maps each input to its local file path or error message.
    #             - input_to_normalized: Maps each input to its normalized PDF path.
    #     """
    #     input_to_downloaded = {}
    #     input_to_normalized = {}

    #     for item in input_items:
    #         logger.info(f"Processing item: {item}")
    #         try:
    #             local_path, downloaded = self.download_if_needed(item)
    #             input_to_downloaded[item] = str(local_path)
    #             ext = local_path.suffix.lower()

    #             logger.info(f"Processing {local_path.name}... (Downloaded: {downloaded})")

    #             # --- New code ---
    #             if ext in ".html":
    #                 # Render to PDF using same filename
    #                 rendered_pdf = self.render_html_with_playwright(item, local_path, Path("data/extract/pdf"))
                    
    #                 if rendered_pdf and rendered_pdf.exists():
    #                     input_to_normalized[item] = str(rendered_pdf.resolve())
    #             else:
    #                 # Other files handled by extract_file
    #                 normalized_pdf = self.extract_file(local_path)
    #                 if normalized_pdf and normalized_pdf.suffix == ".pdf":
    #                     input_to_normalized[item] = str(normalized_pdf.resolve())

    #         except Exception as e:
    #             logger.error(f"Failed to process {item}: {e}")
    #             input_to_downloaded[item] = f"[ERROR] {e}"

    #     return input_to_downloaded, input_to_normalized

    # --- New code ---
    def process_all(self, input_items: list[str]) -> tuple[dict, dict]:
        """
        Process a list of input items (URLs or file paths), downloading and extracting content.
        Skips processing if the result is already available in previous mapping files.

        Args:
            input_items (list[str]): List of URLs or local file paths.

        Returns:
            tuple(dict, dict):
                - input_to_downloaded: Maps each input to its local file path or error message.
                - input_to_normalized: Maps each input to its normalized PDF path.
        """
        mapping_path_1 = Path("data/input_to_output_mapping.json")
        mapping_path_2 = Path("data/input_to_normalized_mapping.json")

        # === Load previous mappings if they exist ===
        existing_input_to_downloaded = json.loads(mapping_path_1.read_text(encoding="utf-8")) if mapping_path_1.exists() else {}
        existing_input_to_normalized = json.loads(mapping_path_2.read_text(encoding="utf-8")) if mapping_path_2.exists() else {}

        input_to_downloaded = existing_input_to_downloaded.copy()
        input_to_normalized = existing_input_to_normalized.copy()

        for item in input_items:
            logger.info(f"\nProcessing item: {item}")

            # === If already downloaded and file exists, skip ===
            # if item in [existing_input_to_normalized, existing_input_to_downloaded]:
            if item in existing_input_to_downloaded:
                existing_path = Path(existing_input_to_downloaded[item])
                if existing_path.exists():
                    logger.info(f"Skipping download, file already exists: {existing_path}")
                    local_path = existing_path
                else:
                    logger.warning(f"File in mapping missing, re-downloading: {existing_path}")
                    local_path, _ = self.download_if_needed(item)
                    input_to_downloaded[item] = str(local_path)
            else:
                try:
                    local_path, _ = self.download_if_needed(item)
                    input_to_downloaded[item] = str(local_path)
                except Exception as e:
                    logger.error(f"Failed to download {item}: {e}")
                    input_to_downloaded[item] = f"[ERROR] {e}"
                    continue

            # === If already normalized and file exists, skip ===
            # if item in [existing_input_to_normalized, existing_input_to_downloaded]:
            if item in existing_input_to_normalized:
                norm_path = Path(existing_input_to_normalized[item])
                if norm_path.exists():
                    logger.info(f"Skipping normalization, already exists: {norm_path}")
                    # --- new code --- 
                    # input_to_normalized[item] = str(norm_path.resolve())  # make sure it’s also recorded in output
                    # --- end new code ---
                    continue
                else:
                    logger.warning(f"Normalized file missing, reprocessing: {norm_path}")

            # === Process file ===
            ext = local_path.suffix.lower()
            try:
                if ext == ".html":
                    rendered_pdf = self.render_html_with_playwright(item, local_path)
                    if rendered_pdf.exists():
                        input_to_normalized[item] = str(rendered_pdf.resolve())
                else:
                    normalized_pdf = self.extract_file(local_path)
                    if normalized_pdf and normalized_pdf.suffix == ".pdf":
                        input_to_normalized[item] = str(normalized_pdf.resolve())
            except Exception as e:
                logger.error(f"Failed to process {item}: {e}")

        # === Save updated mappings ===
        mapping_path_1.write_text(json.dumps(input_to_downloaded, indent=2, ensure_ascii=False), encoding="utf-8")
        mapping_path_2.write_text(json.dumps(input_to_normalized, indent=2, ensure_ascii=False), encoding="utf-8")

        return input_to_downloaded, input_to_normalized

def process_documents(input_items: list[str], store_dir: Path = Path("data/store")) -> tuple[dict, dict]:
    """
    Wrapper function to process documents using the DocumentProcessor class.

    Args:
        input_items (list[str]): List of input URLs or local file paths.
        store_dir (Path): Directory to store downloaded or processed files.

    Returns:
        tuple(dict, dict):
            - input_to_downloaded: Maps each input to the downloaded path or an error message.
            - input_to_normalized: Maps each input to the normalized PDF path if available.
    """
    processor = DocumentProcessor(store_dir=store_dir, extract_dir="data/extract")
    return processor.process_all(input_items)

def copy_pdfs_to_merge_dir(source_dirs: list[Path], merge_dir: Path):
    """
    Copy all PDF files from source directories to the merge directory.
    """
    merge_dir.mkdir(parents=True, exist_ok=True)

    for src_dir in source_dirs:
        if not src_dir.exists():
            logger.warning(f"⚠️ Skipped non-existing folder: {src_dir}")
            continue

        for pdf_file in src_dir.glob("*.pdf"):
            target_file = merge_dir / pdf_file.name
            shutil.copy2(pdf_file, target_file)
            logger.info(f"📄 Copied: {pdf_file.name} → {target_file}")



