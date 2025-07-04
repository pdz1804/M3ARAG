import logging
import time
import json
import yaml
import requests
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# === Configuration ===
IMAGE_RESOLUTION_SCALE = 2.0
OUTPUT_DIR = Path("results")
DATA_DIR = OUTPUT_DIR / "data"
IMG_DIR = OUTPUT_DIR / "imgs"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [DATA_DIR, IMG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Helper: download file from URL if needed ===
def resolve_input(path_or_url: str) -> Path:
    if path_or_url.startswith("http"):
        print(f"🌐 Downloading file from URL: {path_or_url}")
        response = requests.get(path_or_url)
        parsed = urlparse(path_or_url)
        filename = Path(parsed.path).name or "downloaded_file"
        local_path = DATA_DIR / filename
        local_path.write_bytes(response.content)
        print(f"📥 Saved URL content to: {local_path}")
        return local_path
    return Path(path_or_url).resolve()

def main():
    # === Input files/URLs ===
    input_items = [
        # "data/md/wiki.md",
        # "data/html/wiki_duck.html",
        # "data/docx/word_sample.docx",
        # "data/docx/lorem_ipsum.docx",
        # "data/pptx/powerpoint_sample.pptx",
        # "data/2305.03393v1-pg9-img.png",
        # "data/pdf/2206.01062.pdf",
        # "data/asciidoc/test_01.asciidoc",
        "https://arxiv.org/pdf/1706.03762.pdf",  
        "https://arxiv.org/pdf/2503.13964",
        "https://arxiv.org/pdf/2501.06322"
    ]
    input_paths = [resolve_input(p) for p in input_items]

    # === Docling Setup ===
    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.AUTO
    )
    
    pdf_options = PdfPipelineOptions(
        images_scale=IMAGE_RESOLUTION_SCALE,
        generate_page_images=True,
        generate_picture_images=True,
        # do_picture_description=True,
        accelerator_options=accelerator_options,
    )

    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
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

    print(f"🔍 Found {len(input_paths)} files to convert.")

    for input_path in input_paths:
        print(f"\n🚀 Processing: {input_path}")
        start_time = time.time()
        
        try:
            conv_res = doc_converter.convert(input_path)
        except Exception as e:
            print(f"❌ Failed to convert {input_path}: {e}")
            continue
        
        doc = conv_res.document
        stem = conv_res.input.file.stem

        # === Save Markdown, JSON, YAML ===
        (DATA_DIR / f"{stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
        # (DATA_DIR / f"{stem}.json").write_text(json.dumps(doc.export_to_dict(), indent=2), encoding="utf-8")
        # (DATA_DIR / f"{stem}.yaml").write_text(yaml.safe_dump(doc.export_to_dict()), encoding="utf-8")

        # === Save page images ===
        # for page in doc.pages.values():
        #     out_img = IMG_DIR / f"{stem}-page-{page.page_no}.png"
            
        #     # page.image.pil_image.save(out_img, format="PNG")
            
        #     if page.image and page.image.pil_image:
        #         page.image.pil_image.save(out_img, format="PNG")
        #     else:
        #         print(f"⚠️ Skipped page {page.page_no} in {stem} — page image not available.")

        # === Save picture elements ===
        pic_count = 0
        table_count = 0
        for element, _ in doc.iterate_items():
            if isinstance(element, PictureItem):
                pic_count += 1
                img_path = IMG_DIR / f"{stem}-picture-{pic_count}.png"
                img = element.get_image(doc)
                if img is not None:
                    img.save(img_path, format="PNG")
                else:
                    print(f"⚠️ Skipped image {pic_count} in {stem} — get_image() returned None.")
                    
            elif isinstance(element, TableItem):
                table_count += 1
                df: pd.DataFrame = element.export_to_dataframe()
                csv_path = TABLE_DIR / f"{stem}-table-{table_count}.csv"
                html_path = TABLE_DIR / f"{stem}-table-{table_count}.html"
                md_path = TABLE_DIR / f"{stem}-table-{table_count}.md"  # ✅ New line

                # Save as CSV
                df.to_csv(csv_path, index=False)

                # Save as HTML
                html_content = element.export_to_html(doc=doc)
                html_path.write_text(html_content, encoding="utf-8")

                # ✅ Save as Markdown
                md_content = df.to_markdown(index=False)
                md_path.write_text(md_content, encoding="utf-8")

        elapsed = time.time() - start_time
        print(f"✅ Done in {elapsed:.2f}s — Saved {pic_count} figures and {table_count} tables")

if __name__ == "__main__":
    main()
