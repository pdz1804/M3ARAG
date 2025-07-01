# extractor.py
"""Extract content from multiple file formats using Docling and convert Markdown to PDF."""

import time
import pandas as pd
from pathlib import Path
import markdown
import pdfkit

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

# === Output paths ===
EXTRACT_DIR = Path("data/extract")
IMG_DIR = EXTRACT_DIR / "imgs"
TABLE_DIR = EXTRACT_DIR / "tables"
MARKDOWN_DIR = EXTRACT_DIR / "markdown"
PDF_DIR = EXTRACT_DIR / "pdf"

for d in [IMG_DIR, TABLE_DIR, MARKDOWN_DIR, PDF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Docling setup ===
accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.AUTO)
pdf_options = PdfPipelineOptions(
    images_scale=2.0,
    generate_page_images=True,
    generate_picture_images=True,
    accelerator_options=accelerator_options,
)

doc_converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX,
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

    # === HTML → directly convert to PDF
    elif suffix == ".html":
        pdf_path = PDF_DIR / f"{stem}.pdf"
        html_text_to_pdf(doc.export_to_html(), pdf_path)

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


