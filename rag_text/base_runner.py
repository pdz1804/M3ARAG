"""
rag_text/text_captioning/base_runner.py

Provides a unified factory function for instantiating text captioning runners across multiple LLM backends
(OpenAI GPT-4o-mini, Gemini 2.0 Flash, Qwen2.5-VL). This abstraction allows agents such as `TextAgent` to
select and use different QA models at runtime without modifying their logic.

Key Function:
- get_text_captioning_runner(backend: str) -> RunnableLambda:
    Dynamically imports and wraps the appropriate model-specific captioning function in a LangChain `RunnableLambda`.

Supported Backends:
- "openai"  → Uses GPT-4o-mini via OpenAI API.
- "gemini"  → Uses Gemini 2.0 Flash via Google Generative AI.
- "qwen"    → Uses Qwen2.5-VL (local or Hugging Face checkpoint).

Usage:
    from rag_text.text_captioning.base_runner import get_text_captioning_runner
    runner = get_text_captioning_runner("gemini")
    output = runner.invoke({"query": "...", "texts": "..."})
"""

import logging
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_text_captioning_runner(backend: str = "qwen") -> RunnableLambda:
    """
    Factory function to return the appropriate text captioning runner as a LangChain Runnable.
    """
    if backend == "openai":
        from .openai_runner import generate_caption_with_openai
        logger.info("Text QA model: OpenAI GPT-4o-mini")
        return RunnableLambda(generate_caption_with_openai)

    elif backend == "gemini":
        from .gemini_runner import generate_caption_with_gemini
        logger.info("Text QA model: Gemini 2.0 Flash")
        return RunnableLambda(generate_caption_with_gemini)

    elif backend == "qwen":
        from .qwen_runner import generate_caption_with_qwen
        logger.info("Text QA model: Qwen2.5-VL")
        return RunnableLambda(generate_caption_with_qwen)

    else:
        raise ValueError(f"Text QA model: Unsupported backend: {backend}")


