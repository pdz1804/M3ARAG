from langchain_core.messages import AIMessage

def finalize_answer(state):
    """
    Combine retrieved text and image-based content to produce final answer.
    """
    messages = state["messages"]

    text_sources = []
    image_sources = []
    text_content = []
    image_content = []

    for msg in messages:
        if hasattr(msg, "tool_call_id") and "retriever_tool" in msg.name:
            text_content.append(msg.content)
            text_sources.append("PDF")
        elif hasattr(msg, "tool_call_id") and "image_retriever_tool" in msg.name:
            image_content.append(msg.content)
            image_sources.append("Images")

    summary = "### Final Answer\n"
    if text_content:
        summary += "\n**Text Insights (from PDF):**\n" + "\n".join(text_content)
    if image_content:
        summary += "\n\n**Image Insights (from images):**\n" + "\n".join(image_content)

    summary += "\n\nReferences: " + ", ".join(set(text_sources + image_sources))
    return {"messages": [AIMessage(content=summary)]}
