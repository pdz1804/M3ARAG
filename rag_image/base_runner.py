# rag_image/image_captioning/base_runner.py
import logging
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_image_captioning_runner(backend: str = "qwen") -> RunnableLambda:
    """
    Factory function to return the appropriate text captioning runner as a LangChain Runnable.
    """
    if backend == "openai":
        from .openai_runner import generate_caption_with_openai
        logger.info("Image QA model: OpenAI GPT-4o-mini")
        return RunnableLambda(generate_caption_with_openai)

    elif backend == "gemini":
        from .gemini_runner import generate_caption_with_gemini
        logger.info("Image QA model: Gemini 2.0 Flash")
        return RunnableLambda(generate_caption_with_gemini)

    elif backend == "qwen":
        from .qwen_runner import generate_caption_with_qwen
        logger.info("Image QA model: Qwen2.5-VL")
        return RunnableLambda(generate_caption_with_qwen)

    else:
        raise ValueError(f"Image QA model: Unsupported backend: {backend}")


