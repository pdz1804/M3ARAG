# 01/07/2025

- Make use of Docling library
- Docling receives many types of format: **md html docx pptx png pdf asciidoc **
- Then let's docstring perform parsing --> markdown format + extract images as PNG + extract tables as MD format
- Reason for choosing Docling instead of Unstructured is that Unstructured provide much "complicated code" + image extraction is not as good as docling (You can see this when you perform parsing the document pdf of MDocAgent)
- Try to implement a Text-based RAG first based on this kind of implementation.

---

# 02/07/2025

- Currently for the ImageRAGAgent, we use the Qwen-2.5-VL-Instruct to perform describing and analyzing the image to provide some captions. Then put that captions into the vectorstore. Then just process it like the TextRAGAgent.



