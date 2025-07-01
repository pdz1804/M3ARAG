# rag_text/chain.py
"""RAG chain builder using OpenAI and LangChain Expression Language."""

import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

TEMPLATE = """Use the following context to answer the question.
If unsure, say you don't know. Always cite the source.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(TEMPLATE)

def _format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
    )
    
def _debug_print_context(formatted: str) -> str:
    # print("\n🧠 Selected RAG Context:\n" + "-" * 80)
    # print(formatted)
    # print("-" * 80)
    return formatted

def build_rag_chain(
    retriever: VectorStoreRetriever,
    model_name: str = "gpt-4o-mini",
) -> Runnable:
    """Create the RAG chain with prompt + LLM."""
    try:
        llm: BaseChatModel = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            max_tokens=512
        )

        chain = (
            {
                "context": retriever | _format_docs | RunnableLambda(_debug_print_context),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    except Exception as e:
        logger.error(f"❌ Failed to build chain: {e}")
        raise
