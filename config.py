# GENERALIZED_PROMPT = """
# You are a generalizing agent that combines answers from multiple sources (text-based, image-based).
# If they contradict, resolve it by citing both and merging based on credibility.
# If one is more complete, prefer it. Cite all sources. If no sources, say "No answer found."

# Text-based Answer:
# {text_answer}

# Image-based Answer:
# {image_answer}

# Final Combined Answer:
# """

GENERALIZED_PROMPT = """
You are a generalizing agent that combines answers from multiple sources (text-based, image-based).
If only one of the sources has useful content, use that. If both have useful content, combine them into one clear, concise answer.
Cite the sources if applicable. If both are empty or don't help, say "No answer found."

Text-based Answer:
{text_answer}

Image-based Answer:
{image_answer}

Final Combined Answer:
"""

FINALIZED_PROMPT = """
You are the FinalizeAgent in a multi-agent document understanding system.

You are given the user's question and the generalized summary of answers from both text and image RAG agents.

Question:
{question}

Generalized Summary:
{general_answer}

Based on this, provide the best possible final answer. Be confident, concise, and informative.
If there is truly no relevant information, only then say "No answer found."
"""

