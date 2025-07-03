# rag_image/loader.py

import re
from PyPDF2 import PdfReader
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
import torch

from transformers import BitsAndBytesConfig
from colpali_engine.models import ColQwen2, ColQwen2Processor

from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.embeddings import Embeddings as EmbeddingsProtocol

from pdf2image import convert_from_path

from rag_image.model_cache import get_copali_model_and_processor

# === Setup logging ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CoPaliImageEmbedder(EmbeddingsProtocol):
    """Wrap CoPali to act as a LangChain Embedding interface."""
    def __init__(self):
        self.model, self.processor = get_copali_model_and_processor()

    def embed_documents(self, image_paths: List[str], batch_size: int = 4) -> List[List[float]]:
        import gc  # ← Import once here
        
        # self._lazy_load_model()  # Only load here
        
        embeddings = []
        print(f"🧠 Embedding {len(image_paths)} images in batches of {batch_size}...")

        for i in tqdm(range(0, len(image_paths), batch_size), desc="🖼️ CoPali embedding", ncols=80):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            batch = self.processor.process_images(images).to(self.model.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch)

                # If shape is [B, T, D], reduce to [B, D]
                if isinstance(batch_embeddings, tuple):  # unwrap (model might return a tuple)
                    batch_embeddings = batch_embeddings[0]
                if batch_embeddings.ndim == 3:
                    batch_embeddings = batch_embeddings.mean(dim=1)

                emb_list = batch_embeddings.cpu().tolist()
                embeddings.extend(emb_list)
            
            # 🔥 Free memory after each batch
            del batch_embeddings, batch, images
            torch.cuda.empty_cache()
            gc.collect()

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        # self._lazy_load_model()  # Only load here
        
        logger.info(f"🧠 Embedding query: {query[:60]}...")
        
        import gc  # ← Import once here
        
        batch = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            output = self.model(**batch)
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim == 3:
                output = output.mean(dim=1)
            query_embedding = output[0].cpu().tolist()  # shape: [D]
            
        # 🔥 Free memory after each batch
        del batch
        torch.cuda.empty_cache()
        gc.collect()
        
        return query_embedding

# === Centralized embedder instance ===
embedder = CoPaliImageEmbedder()

# === Update embed_images_with_copali ===
def embed_images_with_copali(
    image_paths: List[Path],
    persist_dir: str = "vectorstores/image_db"
) -> Chroma:
    logger.info("🖼️ Embedding images with CoPali...")

    if not image_paths:
        logger.warning("⚠️ No image paths provided.")
        return None

    # embedder = CoPaliImageEmbedder()
    logger.info(f"✅ CoPali loaded on device: {embedder.model.device}")

    docs = [
        Document(page_content=str(path), metadata={"source": str(path)}) 
        for path in image_paths
    ]

    # vectorstore = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embedder,
    #     persist_directory=persist_dir,
    # )
    
    # vectorstore.persist()
    
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=persist_dir
    )

    if len(docs) == 0:
        logger.warning("⚠️ No documents to add to vectorstore from function embed_images_with_copali.")
        return vectorstore
    
    vectorstore.add_documents(docs)
    
    logger.info(f"✅ Added {len(image_paths)} images to: {persist_dir} from function embed_images_with_copali.")
    
    return vectorstore

# === Update batched_add_to_chroma ===
def batched_add_to_chroma(docs: List[Document], persist_dir: str, batch_size: int = 4):
    vectorstore = Chroma(
        embedding_function=embedder, 
        persist_directory=persist_dir
    )
    
    print(f"📦 Adding {len(docs)} documents to Chroma in batches of {batch_size}...")

    for i in tqdm(range(0, len(docs), batch_size), desc="📡 Embedding to Chroma", ncols=80):
        batch_docs = docs[i:i + batch_size]
        vectorstore.add_documents(batch_docs)

    return vectorstore

def embed_pdfs_as_images_with_copali(pdf_paths: List[Path], persist_dir: str = "vectorstores/image_db") -> Chroma:
    logger.info("📄 Embedding each PDF page as image using CoPali...")

    if not pdf_paths:
        logger.warning("⚠️ No PDF files provided.")
        return None

    logger.info(f"✅ CoPali loaded on device: {embedder.model.device}")

    page_image_paths = []
    temp_dir = Path("tmp/pdf_pages")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        try:
            logger.info(f"🖼️ Converting {pdf_path.name} to images...")
            images = convert_from_path(str(pdf_path), dpi=200)
            for i, image in enumerate(images):
                image_filename = f"{pdf_path.stem}-page-{i+1}.png"
                image_path = temp_dir / image_filename
                image.save(image_path)
                page_image_paths.append((image_path, f"{pdf_path.name}#page={i+1}"))
        except Exception as e:
            logger.error(f"❌ Failed to render {pdf_path.name}: {e}")

    if not page_image_paths:
        raise RuntimeError("❌ No pages converted to images from PDFs.")

    docs = [
        Document(page_content=str(img_path), metadata={"source": pdf_id}) 
        for (img_path, pdf_id) in page_image_paths
    ]

    image_files = [str(p[0]) for p in page_image_paths]

    logger.info(f"🧠 Embedding {len(image_files)} PDF page images into Chroma DB...")
    
    # vectorstore = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embedder,
    #     persist_directory=persist_dir,
    # )
    # vectorstore.persist()
    
    vectorstore = batched_add_to_chroma(docs, persist_dir, batch_size=2)

    logger.info(f"✅ Embedded {len(image_files)} PDF pages as images to: {persist_dir}")
    return vectorstore

# === Make embedder available for ImageRAGAgent or PDF embedding ===
def get_global_embedder():
    return embedder

