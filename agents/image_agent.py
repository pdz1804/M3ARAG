# # agents/image_agent.py
from agents.base import BaseAgent
from pathlib import Path
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from rag_image.loader import embed_images_with_copali, embed_pdfs_as_images_with_copali

from config import IMAGE_CAPTIONING
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("ImageRAGAgent")

        persist_path = "vectorstores/image_db"
        pdf_dirs = [Path("data/store"), Path("data/extract/pdf")]
        image_root_dir = Path("data/store")

        # === Step 1: Check if vectorstore exists ===
        if not Path(persist_path).exists() or not any(Path(persist_path).glob("*")):
            print("📦 No existing vectorstore found. Embedding PDFs and images with CoPali...")

            # --- 1.1 Gather all PDF paths ---
            pdf_paths = []
            for folder in pdf_dirs:
                if not folder.exists():
                    print(f"⚠️ Skipping missing PDF folder: {folder}")
                    continue
                pdf_paths.extend(folder.glob("*.pdf"))

            pdf_paths = sorted(set(pdf_paths))
            print(f"📄 Found {len(pdf_paths)} PDFs")

            if not pdf_paths:
                raise RuntimeError("❌ No PDF files found to embed.")

            # --- 1.2 Embed each page of each PDF ---
            embed_pdfs_as_images_with_copali(pdf_paths, persist_path)

            # --- 1.3 Embed all images under data/store/** ---
            image_paths = sorted(set(image_root_dir.rglob("*.png")))
            print(f"🖼️ Found {len(image_paths)} image(s)")
            if image_paths:
                embed_images_with_copali(image_paths, persist_path)
            else:
                print("⚠️ No images found to embed.")

        # === Step 2: Initialize retriever ===
        from rag_image.loader import embedder
        
        self.embedder = embedder
        
        self.vectorstore = Chroma(
            embedding_function=self.embedder,
            persist_directory=persist_path,
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'lambda_mult': 0.5}
        )

        # === Step 3: Load Qwen runner ===
        if IMAGE_CAPTIONING == "openai":
            logger.info("🔁 Using OpenAI GPT-4o-mini for image captioning.")
            from rag_image.caption.openai_runner import generate_caption_with_openai
            self.caption_with_llm = RunnableLambda(generate_caption_with_openai)
        elif IMAGE_CAPTIONING == "gemini":
            logger.info("🔁 Using Gemini for image captioning.")
            from rag_image.caption.gemini_runner import generate_caption_with_gemini
            self.caption_with_llm = RunnableLambda(generate_caption_with_gemini)
        else:
            logger.info("🔁 Using Qwen2.5-VL for image captioning.")
            from rag_image.caption.qwen_runner import generate_caption_batch
            self.caption_with_llm = RunnableLambda(generate_caption_batch)
            
        # self.caption_with_qwen = RunnableLambda(generate_caption_batch)

    def run(self, input_data: dict) -> str:
        question = input_data.get("question", "")
        top_docs = self.retriever.invoke(question)
        
        logger.info(f"🔍 Retrieved {len(top_docs)} relevant documents for question: {question}")
        for i, doc in enumerate(top_docs):
            logger.info(f"[Doc {i}] source={doc.metadata.get('source', '')} | content preview={doc.page_content[:80]}")

        # We only caption the image docs
        # img_paths = [doc.metadata["source"] for doc in top_docs if doc.metadata["source"].endswith(".png")]
        img_paths = [
            doc.page_content for doc in top_docs
            if doc.page_content.endswith(".png") and Path(doc.page_content).exists()
        ]
       
        return self.caption_with_llm.invoke({"query": question, "image_paths": img_paths})



