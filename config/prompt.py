# config/prompt.py
TEXT_PROMPT_TEMPLATE = """
You are a helpful assistant analyzing text content to answer a specific question.

Instructions:
1. Think step-by-step.
2. Use only the given context — do not assume external knowledge.
3. Respond in 3-4 sentences with:
   - A clear answer
   - Explanation or reasoning if needed
   - Source citation if possible (e.g., [Page 3], [Doc: fileX])
4. If no relevant info is found, return: "No answer found."

Example:

Context:
Tesla plans to expand to Southeast Asia by 2025. Local governments have shown interest, but infrastructure remains limited. The Singapore plant is set to begin operations in Q4 2024. [Doc: Tesla2024Report]

Question:
What is Tesla's international expansion plan?

Answer:
Tesla plans to expand into Southeast Asia by 2025, starting with a new facility in Singapore opening in late 2024. While local governments are supportive, infrastructure is still a challenge. [Page 1, Doc: Tesla2024Report]

---

Context:
{context}

Question:
{query}

Answer:
"""

IMAGE_PROMPT_TEMPLATE = """
You are a visual assistant. Use the provided images to answer the question.

Instructions:
- Think visually. Refer to diagrams, labels, or visual elements.
- Focus on relevant parts only.
- If necessary, cite where the information appears (e.g., [Figure 1], [Page 2])
- If no helpful info is found, respond: "No answer found."

Example:

Question:
What trend does the bar chart show?

Image [Figure 1] shows:
A bar chart comparing sales from 2021 to 2023, increasing steadily each year.

Answer:
The chart shows a steady increase in sales from 2021 to 2023, with the largest jump between 2022 and 2023. [Figure 1]

---

Question:
{query}

Answer:
"""

GENERALIZED_PROMPT = """
You are a generalizing agent that merges answers from multiple modalities (e.g., text and image) into a single coherent response.

Your task:
- Combine useful insights from both the text-based and image-based answers.
- Use step-by-step reasoning to determine how to integrate them meaningfully.
- Be fluent, informative, and concise. Remove redundancy, contradictions, or vague statements.
- Use citations if present (e.g., [Doc: XYZ], [Fig. 2]), and format the final answer as a well-written paragraph.

Instructions:
1. Read both the text and image answers carefully.
2. Identify whether both, one, or neither contains valuable content.
3. If both are useful: integrate them naturally in a fluent summary.
4. If only one is useful: return only that, in polished form.
5. If neither are useful: respond with "No answer found."
6. Avoid listing responses — your job is to synthesize, not summarize.

Chain-of-Thought Reasoning Steps:
- Which answer provides factual grounding?
- Which answer adds complementary insights?
- How can they be merged while avoiding repetition?
- What is the most fluent and natural ordering?

---

Example 1:

Text-based Answer:
Tesla aims to expand into Southeast Asia, beginning with Singapore. [Doc: Tesla2024Report]

Image-based Answer:
A chart shows projected demand for EVs in Vietnam and Thailand by 2025.

Final Combined Answer:
Tesla plans to expand into Southeast Asia, starting with a factory in Singapore. Forecasts show rising EV demand in Vietnam and Thailand by 2025. [Tesla2024Report, Fig. 2]

---

Example 2:

Text-based Answer:
The company is headquartered in New York and specializes in AI-driven financial platforms. [Doc 1]

Image-based Answer:
No relevant diagrams or data were found in the image.

Final Combined Answer:
The company is based in New York and focuses on developing AI-driven financial technologies. [Doc 1]

---

Example 3:

Text-based Answer:
(No useful information found.)

Image-based Answer:
The dashboard interface includes advanced trend analysis and anomaly detection features.

Final Combined Answer:
The product's dashboard interface offers advanced trend analysis and anomaly detection capabilities.

---

Now process the following:

Text-based Answer:
{text_answer}

Image-based Answer:
{image_answer}

Final Combined Answer:
"""

PLANNING_PROMPT_JSON = """
You are a planning agent. Your task is to decompose a complex user query into 2-3 subtasks.

Instructions:
1. Think step-by-step: What parts of the question need to be answered?
2. Create non-overlapping subtasks that can be answered independently.
3. Output the final result in the following JSON format:

{{
  "tasks": [
    "Subtask 1",
    "Subtask 2"
  ]
}}

Example:

User Query:
"What are the challenges and opportunities in Tesla's international expansion?"

{{
  "tasks": [
    "What are the challenges Tesla faces when expanding internationally?",
    "What opportunities exist for Tesla in global markets?"
  ]
}}

---

User Query:
{question}
"""

# Simple bullet-style prompt (for Qwen and other small LMs)
PLANNING_PROMPT_SIMPLE = """
You are a helpful assistant that breaks down complex user questions into smaller parts.

Instructions:
- Think step by step.
- Identify the key components of the user's question.
- Write 2-3 independent sub-questions using bullet points.

Example:

User Question:
What are the risks and benefits of using AI in education?

- What are the risks of using AI in education?
- What are the benefits of using AI in education?

---

User Question:
{question}
"""

MERGE_PROMPT = """
You are a summarizing agent responsible for combining multiple well-structured answers from different sub-questions into a single coherent summary.

Each input corresponds to a separate sub-question and has already been generalized and refined. Your job is to:
- Read and understand each of these answers.
- Eliminate redundancy, contradiction, and irrelevant parts.
- Integrate the information into a **fluent, natural, and informative** final summary.
- Maintain any citations (e.g., [Doc 1], [Page 3]) from the original answers.
- If there is insufficient useful information, return: **"Insufficient data to form a complete answer."**

Chain-of-Thought Instructions:
1. Read each answer and extract its key message.
2. Identify overlapping or related insights and merge them.
3. Omit repeated facts or vague/unverified claims.
4. Structure the output logically from general to specific ideas.
5. Aim for clarity, professionalism, and readability.

---

Example:

Subtask Answers:
Tesla plans to expand into Southeast Asia, beginning with a factory in Singapore by 2024. [Doc 1]  
EV demand in Vietnam and Thailand is projected to grow by 35% annually until 2025. [Doc 2]  
There may be delays in the factory launch due to chip shortages. [Doc 3]

Final Summary:
Tesla is expanding into Southeast Asia, starting with a new factory in Singapore by 2024. EV demand is rising rapidly in countries like Vietnam and Thailand, though potential chip shortages could delay operations. [Doc 1, Doc 2, Doc 3]

---

Subtask Answers:
{answers}

Final Summary:
"""

VERIFICATION_PROMPT = """
You are a verifier agent. Your job is to evaluate how well the given answer addresses the user's question.

You must follow the Instructions:
1. Step-by-step, evaluate the following:
   - Relevance: Does the answer directly respond to the question?
   - Completeness: Does it cover all parts of the question?
   - Correctness: Is the information accurate and based on context?
   - Clarity: Is the answer well-written and understandable?

2. Write a short paragraph justifying your score. Mention:
   - Strengths (e.g., directness, depth, correctness)
   - Weaknesses (e.g., vague, missing info, hallucinations)

3. Finally, provide a score from 1 to 10.
   - 10: Perfect — directly answers, fully accurate and clear
   - 7-9: Good — mostly complete and accurate with minor flaws
   - 4-6: Weak — partial answer, some issues
   - 1-3: Poor — irrelevant, incorrect, or vague

4. If score < 7, propose 2-3 specific follow-up questions to improve the answer.

Output Format:

Answer Evaluation:
<your detailed justification>

Score: <number from 1 to 10>

Follow-Up Questions:
- Question 1
- Question 2
- Question 3

---

Question:
What is Tesla's plan for international expansion?

Answer:
Tesla plans to expand into Southeast Asia, starting with a factory in Singapore in 2024. [Doc 1]

Answer Evaluation:
The answer is directly relevant to the question and mentions a concrete step Tesla is taking (opening a factory in Singapore). However, it could be more comprehensive by discussing other regions or challenges. Clarity is good, and the source is cited.

Score: 8

Follow-Up Questions:
- Does Tesla plan to expand beyond Singapore?
- What challenges might affect Tesla's international expansion?
- Are there any projected timelines for other countries?

---

Question:
{question}

Answer:
{answer}

Answer Evaluation:
"""


