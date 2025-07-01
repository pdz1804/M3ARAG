from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from config import TEXT_DB_PATH, TEXT_EMBEDDING_MODEL, TOP_K

@tool
def text_retriever_tool(query: str) -> str:
    """
    Retrieve and return the most relevant document chunks for a given query using vector similarity.

    Args:
        query (str): A user query string to search for relevant documents.

    Returns:
        str: Concatenated top-k relevant document texts with their sources.
    """
    embedding = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)
    vectorstore = Chroma(embedding_function=embedding, persist_directory=TEXT_DB_PATH)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unknown')
        results.append(f"[Doc {i+1}] (Source: {source})\n{doc.page_content}")

    return "\n\n".join(results)


