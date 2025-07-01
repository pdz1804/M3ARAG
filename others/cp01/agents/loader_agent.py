import os
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# New version make use of unstructured library
import os
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Table, Image

def load_and_split_text():
    if not os.path.isdir(PDF_PATH):
        raise NotADirectoryError(f"Expected a directory at: {PDF_PATH}")

    all_pages = []
    
    # Loop through all PDF files in the directory
    for filename in os.listdir(PDF_PATH):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_PATH, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No PDF documents found or all failed to load.")
    
    # Initialize the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    
    # Split all loaded documents
    return splitter.split_documents(all_pages)

# This turn each page into an image and saves it as a JPEG.
def get_images_base64(chunks):
    images_b64 = []
    for i, chunk in enumerate(chunks):
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for j, el in enumerate(chunk_els):
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
                    print(f"   📷 Found image in chunk {i}, element {j}")
    return images_b64

def extract_images_and_tables(pdf_dir: str, image_dir: str):
    print(f"📁 Scanning PDF directory: {pdf_dir}")
    os.makedirs(image_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs found in directory: {pdf_dir}")
        return

    total_images = 0
    total_tables = 0
    total_chunks = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n🔍 Processing: {pdf_file}")
        
        pdf_basename = os.path.splitext(pdf_file)[0]

        chunks = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )
        print(f"   📑 Extracted {len(chunks)} chunks from {pdf_file}")
        total_chunks += len(chunks)

        tables = []
        texts = []
        images_b64 = get_images_base64(chunks)

        for idx, img64 in enumerate(images_b64):
            image_filename = f"{pdf_basename}_image_{idx:03}.png"
            image_path = os.path.join(image_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(base64.b64decode(img64))
            print(f"   💾 Saved image: {image_filename}")

        image_count = len(images_b64)
        total_images += image_count

        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                for el in chunk.metadata.orig_elements:
                    if "Table" in str(type(el)):
                        tables.append(el)
                        print(f"   📊 Found table element")
                texts.append(chunk)

        total_tables += len(tables)
        print(f"✅ {pdf_file} → {image_count} images, {len(tables)} tables, {len(texts)} text chunks")

    print("\n📊 SUMMARY:")
    print(f"   🧾 Total PDFs processed: {len(pdf_files)}")
    print(f"   🧩 Total text chunks: {total_chunks}")
    print(f"   🖼️  Total images extracted: {total_images}")
    print(f"   📈 Total tables extracted: {total_tables}")

    return texts, tables, images_b64


