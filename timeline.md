# 01/07/2025

- Make use of `Docling` library
- `Docling` receives many types of format: **md html docx pptx png pdf asciidoc**
- Then let's docstring perform parsing --> markdown format + extract images as PNG + extract tables as MD format
- Reason for choosing `Docling` instead of Unstructured is that Unstructured provide much "complicated code" + image extraction is not as good as docling (You can see this when you perform parsing the document pdf of MDocAgent)
- Try to implement a Text-based RAG first based on this kind of implementation.

---

# 02/07/2025

- Currently for the `ImageRAGAgent`, we use the `Qwen-2.5-VL-Instruct` to perform describing and analyzing the image to provide some captions. Then put that captions into the vectorstore. Then just process it like the TextRAGAgent.

---

# 03/07/2025 

- Now for Image-RAG pipeline, we use Colpali `vidore/colqwen2-v1.0`. I have tried to use the minimalist config of bitsandbytes of `int4` but the performance for querying on those embeddings is not quite good. BUT i am limited by my VRAM (4 gb vram and 16 gb shared memory). 
- Currently not just need to depend on `Qwen-2.5-VL-Instruct` for generating caption for retrieved images. We can switch to use OpenAI API or Gemini API by changing the value in the `config.py` of `IMAGE_CAPTIONING = "gemini" # "openai" or "qwen" (Qwen2.5-VL)`.

---

# 08/07/2025 

- For RAG System:
  - Integrate the **RAG system of Bang** to replace the old RAG
  - Bang is motivated from **VisDom RAG and MDocAgent RAG** and try to make use of their code, replace the complicated logic, dependencies and organization of their code with a simpler modular robust class of RAG system with text and visual pipeline. 
  - Currently, now we have a **big class of RAG, being decomposed into 2 small classes of TextRAGPipeline and VisualRAGPipeline**. 
  - Later, we would add the logic inside this class to have a way to choose either retrieve by text or by image or by both. 

- For processing documents:
  - We now be able to handle **Words, Powerpoint, Text, Markdown, HTML, PDF, CSV**. They could be URLs or local files is ok also.
  - We make a class to handle this with several different functions, which is more concise and easier for us to later enhance, reduce or reuse our code. 

- For Agents, currently our MultiAgents framework runs sequentially:
  - Starting from retrieving top-k chunks from text and from image RAG Pipeline. We then have a TextAgent for handling text-chunks and ImageAgent for handling image-chunks. These 2 agents would provide concise, detailed and relevant information about the retrieve contents from RAG. 
  - Then a GeneralizeAgent would combine, reduce conflicts with irrelavant information and generalize the response.
  - Currently the Flow could be done for using **openai** for answering **text-input-output** and **gemini** for **image-input text-output**.
  - Lastly, FinalizeAgent would receive the response from GeneralizeAgent and double-check the response with the query to see if it is relevant or not. Then provide the final response which would later be shown for the user.

**Next things:**

- Small bug not fix is that for TextRAG the chunk for PDF start the page at 1 but ImageRAG start at 0.
- We have not rechecked the code for using Qwen for response in place of openai and gemini. 
- Let the Agents to decide whether it needs to retrieve information from Text or Image or both types of RAGs.
- Create the Agent for normalizing the query 
- Create the loop for MultiAgents to continue to retrieve the information after the response come from FinalizeAgent is produced. This would help us to retrieve much more information if the current response is not enough or not satisfy with the query. We should need a `limit` for retrieving recursion.
- We have only handled CSV -> TXT and not yet TXT -> PDF yet. 
- We could make use of docling to handle JSON, Audio (.mp3, .wav) also because they can provide us JSON & Audio -> TXT / MD. We could later transform this to PDFs.
- Should design a minimalist interface for testing uploading multi-format documents and see the response. 
- Should come up with the design for our system

--- 

# 11/07/2025

- The **multiagent system** now consists of:
  - **Text Agent + Image Agent**: For generating insights from the retrieved contexts of the RAG system 
  - **Generalize Agent** would combine and generalize the answers from TextAgent and ImageAgent for each question
  - **Planning Agent** would receive the query from the user and then separates it into several "tasks" or questions for retrieving many information from the RAG system. 
  - **Merge Agent** would combine all the responses from the Generalize Agent and merge them into a response which would answer the initial query from the user.
  - **Verfier Agent** would score the combined answer of the Merge Agent and then telling if we need to query for more information by generating following up questions for continually retrieve information. 

- Currently we add some new information in the `agent_config.py` for `max_loop` for reasking the following up questions from **Verifier Agent**, `max_tasks` for the maximum number of decomposed tasks / questions that we would handle in a single iteration and `threshold` for the score for which the answer is good enough for returning to the user.
- We have fixed `chat_streamlit.py` for updating the new code into it and let the user chat with their uploading documents. 
- We have make the simplified design for our system using Drawio. 

**Next things:**

- Ask for suggestions and comments on our current implementation.
- Small bug not fix is that for TextRAG the chunk for PDF start the page at 1 but ImageRAG start at 0.
- We have not rechecked the code for using Qwen for response in place of openai and gemini. 
- We have only handled CSV -> TXT and not yet TXT -> PDF yet. 
- We could make use of docling to handle JSON, Audio (.mp3, .wav) also because they can provide us JSON & Audio -> TXT / MD. We could later transform this to PDFs.
- Should come up with the design for our system

--- 

# 17/07/2025

- Have done the design for this system.
- We currently could run all the Agent with OpenAI or Gemini models. For Qwen (local), we have only test it for Text-generated only.
- Just make a clean for the repo and fix the README
- Ensure the `requirements.txt` is correct if run from start 
- Mentor suggests this pipeline should be applied in some specific domain as this is currently **abstract** and when we want to apply to some problems we need to customize it. So this is just the good practice for performing all the things that we learn so far about Document Understanding + RAG + Multiagent System. 

**Next things:** 

- (New) Apply this pipeline and customize it to satisfy with the scope of Prof Motohashi. Also, have to show and suggest Prof about the potential of this approach when being compared with his ML approach.
- We would **try to do the (old) things below here to finalize this repository.**
  - Small bug not fix is that for TextRAG the chunk for PDF start the page at 1 but ImageRAG start at 0.
  - We have not rechecked the code for using Qwen for response in place of openai and gemini. 
  - We have only handled CSV -> TXT and not yet TXT -> PDF yet. 
  - We could make use of docling to handle JSON, Audio (.mp3, .wav) also because they can provide us JSON & Audio -> TXT / MD. We could later transform this to PDFs.



