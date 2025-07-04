# extractor.py
"""Extract content from multiple file formats using Docling and convert Markdown to PDF."""

import time
import pandas as pd
from pathlib import Path
import markdown
import pdfkit

from pptxtopdf import convert as pptx_to_pdf_convert
from docx2pdf import convert as docx_to_pdf_convert

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from PIL import Image

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_IMAGE_WIDTH = 300  # pixels
MIN_IMAGE_HEIGHT = 300
MIN_IMAGE_AREA = 90_000  # width * height

# === Output paths ===
EXTRACT_DIR = Path("data/extract")
IMG_DIR = EXTRACT_DIR / "imgs"
TABLE_DIR = EXTRACT_DIR / "tables"
MARKDOWN_DIR = EXTRACT_DIR / "markdown"
PDF_DIR = EXTRACT_DIR / "pdf"

for d in [IMG_DIR, TABLE_DIR, MARKDOWN_DIR, PDF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Supported formats ===
PDF_FORMATS = {".pdf"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
WORD_EXTS = {".doc", ".docx"}
PPT_EXTS = {".ppt", ".pptx"}
HTML_EXTS = {".html", ".htm"}
CSV_EXTS = {".csv"}

# === Docling setup ===
accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.AUTO)

pdf_options = PdfPipelineOptions(
    images_scale=2.0,
    generate_page_images=True,
    generate_picture_images=True,
    accelerator_options=accelerator_options,
)

# class InputFormat(str, Enum):
#     """A document format supported by document backend parsers."""

#     DOCX = "docx"
#     PPTX = "pptx"
#     HTML = "html"
#     IMAGE = "image"
#     PDF = "pdf"
#     ASCIIDOC = "asciidoc"
#     MD = "md"
#     CSV = "csv"
#     XLSX = "xlsx"
#     XML_USPTO = "xml_uspto"
#     XML_JATS = "xml_jats"
#     JSON_DOCLING = "json_docling"
#     AUDIO = "audio"

doc_converter = DocumentConverter(
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

def markdown_to_pdf(md_path: Path, output_pdf_path: Path):
    """Convert a Markdown file to PDF using pdfkit."""
    try:
        md_content = md_path.read_text(encoding="utf-8")
        html_content = markdown.markdown(md_content)
        pdfkit.from_string(html_content, str(output_pdf_path))
        print(f"📄 Converted {md_path.name} → {output_pdf_path.name}")
    except Exception as e:
        print(f"❌ Failed to convert {md_path.name} to PDF: {e}")

def html_text_to_pdf(html_text: str, output_pdf_path: Path):
    """Convert raw HTML string to PDF using pdfkit."""
    try:
        pdfkit.from_string(html_text, str(output_pdf_path))
        print(f"🌐 HTML → {output_pdf_path.name}")
    except Exception as e:
        print(f"❌ Failed to convert HTML to PDF: {e}")

def extract_file(file_path: Path):
    print(f"\n🚀 Processing: {file_path}")
    start_time = time.time()

    try:
        conv_res = doc_converter.convert(file_path)
    except Exception as e:
        raise RuntimeError(f"Conversion failed for {file_path}: {e}")

    doc = conv_res.document
    stem = conv_res.input.file.stem
    suffix = file_path.suffix.lower()

    # === PDF files: skip saving content (already PDF)
    if suffix == ".pdf":
        pass
    
    # === Skip raw image files ===
    elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]:
        print(f"⏩ Skipped image file: {file_path.name}")
        return
    
    # === Handle PPTX separately with pptxtopdf ===
    elif suffix == ".pptx":
        output_pdf = PDF_DIR / f"{stem}.pdf"
        if output_pdf.exists():
            print(f"⚠️ Skipped (already exists): {output_pdf.name}")
        else:
            try:
                print(f"📊 Converting PPTX → PDF: {file_path.name}")
                pptx_to_pdf_convert(str(file_path.parent), str(PDF_DIR))
                print(f"✅ PPTX converted to: {output_pdf.name}")
            except Exception as e:
                print(f"❌ PPTX conversion failed: {file_path.name} — {e}")
        return  # Done after pptxtopdf conversion

    # === HTML → directly convert to PDF
    # === TTB fix this already on process_documents.py and downloader.py ===
    # elif suffix == ".html":
    #     pdf_path = PDF_DIR / f"{stem}.pdf"
    #     html_text_to_pdf(doc.export_to_html(), pdf_path)
    
    elif suffix in [".csv"]:
        try:
            logger.info(f"📊 Processing CSV file: {file_path.name}")
            df = pd.read_csv(file_path)
            logger.info(f"🔍 Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")

            # Drop unnamed columns often generated by Excel
            columns_to_drop = [col for col in df.columns if "unnamed" in col.lower()]
            if columns_to_drop:
                logger.info(f"⚠️ Dropping unnamed columns: {columns_to_drop}")
            df_cleaned = df.drop(columns=columns_to_drop)

            output_dir = Path("data/store")
            output_dir.mkdir(parents=True, exist_ok=True)
            txt_path = output_dir / f"{file_path.stem}.txt"

            logger.info(f"📄 Writing cleaned CSV to TXT: {txt_path}")

            with txt_path.open("w", encoding="utf-8") as f:
                for idx, row in df_cleaned.iterrows():
                    if idx >= 100:
                        logger.warning(f"⚠️ Truncating output at 100 rows for: {file_path.name}")
                        break
                    for col, val in row.items():
                        f.write(f"{col}: {val}\n")
                    f.write("\n")  # blank line between rows

            logger.info(f"✅ CSV conversion complete for: {file_path.name}")
            return txt_path
        except Exception as e:
            logger.error(f"❌ Error processing CSV file {file_path.name}: {e}")
            return None
        
    # === Handle DOCX separately with docx2pdf ===
    elif suffix in [".doc", ".docx"]:
        output_pdf = PDF_DIR / f"{stem}.pdf"
        if output_pdf.exists():
            print(f"⚠️ Skipped (already exists): {output_pdf.name}")
        else:
            try:
                print(f"📄 Converting DOCX → PDF: {file_path.name}")
                docx_to_pdf_convert(str(file_path))
                # Move converted PDF to target directory
                original_pdf = file_path.with_suffix(".pdf")
                if original_pdf.exists():
                    original_pdf.rename(output_pdf)
                    print(f"✅ DOCX converted to: {output_pdf.name}")
                else:
                    print(f"⚠️ DOCX converted PDF not found where expected: {original_pdf}")
            except Exception as e:
                print(f"❌ DOCX conversion failed: {file_path.name} — {e}")
        return  # Done after docx2pdf conversion

    # === Other (DOCX, MD, etc.) → save Markdown → convert to PDF
    else:
        md_path = MARKDOWN_DIR / f"{stem}.md"
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

        pdf_path = PDF_DIR / f"{stem}.pdf"
        markdown_to_pdf(md_path, pdf_path)

    # === Save images ===
    img_count = 0
    table_count = 0
    for element, _ in doc.iterate_items():
        if isinstance(element, PictureItem):
            img = element.get_image(doc)
            if img:
                width, height = img.size
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT or (width * height) < MIN_IMAGE_AREA:
                    print(f"⚠️ Skipping small image ({width}x{height}) from {file_path.name}")
                    continue  # Skip tiny or irrelevant images
                
                img_count += 1
                img_path = IMG_DIR / f"{stem}-pic-{img_count}.png"
                img.save(img_path)
        elif isinstance(element, TableItem):
            table_count += 1
            df: pd.DataFrame = element.export_to_dataframe()

            # Save table as CSV, HTML, and MD
            # # Save DataFrame to CSV
            # df.to_csv(TABLE_DIR / f"{stem}-table-{table_count}.csv", index=False)
            
            # # Save DataFrame to HTML
            # html = element.export_to_html(doc=doc)
            # (TABLE_DIR / f"{stem}-table-{table_count}.html").write_text(html, encoding="utf-8")
            
            # Save DataFrame to Markdown
            md = df.to_markdown(index=False)
            (TABLE_DIR / f"{stem}-table-{table_count}.md").write_text(md, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"✅ Done in {elapsed:.2f}s — Saved {img_count} images and {table_count} tables")


