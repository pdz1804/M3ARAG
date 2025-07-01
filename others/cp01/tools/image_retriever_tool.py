from langchain_core.tools import tool
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os, base64
from tqdm import tqdm
from config import IMAGE_DIR, IMAGE_DB_PATH, TEXT_EMBEDDING_MODEL, LLM_MODEL_NAME

def caption_and_index_images():
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
    embedding = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)
    
    prompt_template = f"""Describe the image {img_name} in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""

    docs = []
    for img_name in tqdm(sorted(os.listdir(IMAGE_DIR))):
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            
        prompt_template = f"""Describe the image {img_name} in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt_template}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]
        }
        try:
            caption = llm.invoke([message]).content.strip()
            doc = Document(page_content=caption, metadata={"image": img_name})
            docs.append(doc)
        except Exception as e:
            print(f"❌ Failed to caption {img_name}: {e}")

    os.makedirs(IMAGE_DB_PATH, exist_ok=True)
    vectorstore = Chroma.from_documents(docs, embedding=embedding, persist_directory=IMAGE_DB_PATH)
    print(f"✅ Indexed {len(docs)} image captions.")

@tool
def image_retriever_tool(query: str) -> str:
    """
    Retrieve the most relevant image captions for a given query using vector similarity.

    Args:
        query (str): A user query related to visual content.

    Returns:
        str: Top-k image captions with corresponding filenames.
    """
    embedding = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)
    vectorstore = Chroma(embedding_function=embedding, persist_directory=IMAGE_DB_PATH)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant image captions found."

    results = []
    for i, doc in enumerate(docs):
        img = doc.metadata.get("image", "unknown")
        results.append(f"[Image {i+1}] (File: {img})\n{doc.page_content}")

    return "\n\n".join(results)


