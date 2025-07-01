GENERALIZED_PROMPT = """
You are a generalizing agent that combines answers from multiple sources (text-based, image-based).
If they contradict, resolve it by citing both and merging based on credibility.
If one is more complete, prefer it. Cite all sources. If no sources, say "No answer found."

Text-based Answer:
{text_answer}

Image-based Answer:
{image_answer}

Final Combined Answer:
"""

FINALIZED_PROMPT = """
You are the final answer agent. Given the user's question and a generalized answer,
write a clear and concise response. Avoid repetition. Ensure readability. If the generalized answer is incomplete, say "No answer found."

Question:
{question}

Generalized Answer:
{general_answer}

Final Answer:
"""

