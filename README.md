
# 🤖 M3ARAG: Modular Multi-Agent RAG Pipeline

This project is a fully modular **Multi-Modal Multi-Agent Retrieval-Augmented Generation (RAG)** system capable of processing **PDFs, HTMLs, images, and tables** for answering questions using a pipeline of specialized agents:

- 🔍 `TextAgent` – performs text-based retrieval over documents
- 🔍 `VisualAgent` – performs image-based retrieval over documents (NOT DONE)
- 🧠 `GeneralizeAgent` – merges multimodal answers and resolves conflicts
- 📝 `FinalizeAgent` – generates clean, concise final responses

It supports local document extraction via [Docling](https://github.com/ds4sd/docling), embedding with SentenceTransformers, and multi-agent orchestration.

---

## 📁 Project Structure

```
AgenticRAG_PDZ/
├── agents/                 # Modular agent logic (Text, Generalize, Finalize)
├── app/                    # Chat launcher interface (agent mode entrypoint)
├── data/
│   ├── store/              # Raw downloaded files (PDF, HTML, etc.)
│   └── extract/            # Converted PDFs, extracted images/tables
├── download/               # Download logic (e.g., HTML/PDF fetchers)
├── extract/                # File processing pipelines (Docling wrappers)
├── rag_text/               # RAG pipeline: loader, embedding, vectorstore
├── utils/                  # Helper utilities (e.g., process_documents)
├── vectorstores/           # Persisted Chroma vectorstore
├── main.py                 # Main entrypoint
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/pdz1804/M3ARAG.git
cd M3ARAG
```

### 2. Create Python environment (optional but recommended)

```bash
python -m venv myenv
# Or: conda create -n m3arag python=3.10

# Activate
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If `requirements.txt` is missing, install manually like below:

```bash
pip install sentence-transformers langchain openai chromadb docling python-dotenv
```

### 4. Set up your environment variables

Copy `.env.example` and rename to `.env`, then fill in your OpenAI key:

```env
OPENAI_API_KEY=pdz-...
```

### 5. Run the full pipeline

If you want to run RAG-flow individually without Agents or with Agents:

```bash
# Use both text and image agents
python main.py --text --image

# Use text only
python main.py --text

# Use image only
python main.py --image

# Use standard text-only fallback (no multi-agent)
python main.py

```

This will:
- Download and store files from hardcoded URLs.
- Extract content using Docling.
- Index text via SentenceTransformers + Chroma.
- Start interactive agent-based chat loop.

---

## 🧠 Agents Used

| Agent            | Description |
|------------------|-------------|
| `TextRAGAgent`   | Answers questions by retrieving from embedded text chunks |
| `GeneralizeAgent`| Combines answers from multiple modalities (text, image) |
| `FinalizeAgent`  | Generates clean and concise answers for delivery |

---

## 📥 Input Types Supported

- ✅ PDF documents (`.pdf`)
- ✅ HTML pages (converted to PDF)
- ✅ Extracted images (captioning + indexing coming soon)
- 🧪 Support for `.docx`, `.pptx`, `.md` being tested

---

## 🛠 Tools & Libraries

- 🧱 [LangChain](https://www.langchain.com/)
- 📚 [SentenceTransformers](https://www.sbert.net/)
- 📦 [ChromaDB](https://www.trychroma.com/)
- 🔍 [Docling](https://github.com/ds4sd/docling)
- 🤖 OpenAI GPT (GPT-4o-mini used for generation)

---

## 📌 Optional Enhancements

| Feature                   | Status      |
|---------------------------|-------------|
| ImageRAGAgent             | 🔜 In Progress |
| GUI via Streamlit         | 🔜 Planned |
| Agent state memory        | 🔜 Planned |
| Upload-your-own-doc       | ✅ Supported (manual) |
| Beam-search for retrieval | 🔜 Planned |

---

## 🧪 Testing

You can test document processing independently via:

```bash
python -m utils.process_documents
```

Or test individual agent logic in `agents/`.

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## 🧠 Author

Built by [Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804) and Tieu Tri Bang 

---

## 🙋‍♂️ Contact

Reach out or open an [issue](https://github.com/pdz1804/M3ARAG/issues) for support or ideas.


