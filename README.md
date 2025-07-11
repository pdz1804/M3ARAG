
# 🤖 M3ARAG: Modular Multi-Agent RAG Pipeline

This project is a fully modular **Multi-Modal Multi-Agent Retrieval-Augmented Generation (RAG)** system capable of processing **PDFs, HTMLs, images, and tables** for answering questions using a pipeline of specialized agents:

- `Text Agent` + `Image Agent`: For generating insights from the retrieved contexts of the RAG system 

- `Generalize Agent` would combine and generalize the answers from TextAgent and ImageAgent for each question

- `Planning Agent` would receive the query from the user and then separates it into several "tasks" or questions for retrieving many information from the RAG system. 

- `Merge Agent` would combine all the responses from the Generalize Agent and merge them into a response which would answer the initial query from the user.

- `Verfier Agent` would score the combined answer of the Merge Agent and then telling if we need to query for more information by generating following up questions for continually retrieve information. 

It supports local document extraction via [Docling](https://github.com/ds4sd/docling), embedding with SentenceTransformers, and multi-agent orchestration.

---

## 📁 Project Structure

```
M3ARAG/
├── agents/                 # Modular agent logic.
├── pipeline/               # Pipeline and Chat launcher interface (agent mode entrypoint)
├── data/                   # Storing the downloaded files
│   ├── store/              # Raw downloaded files (PDF, HTML, etc.)
│   ├── merge/              # Single processing location for indexing of RAG.
│   └── extract/            # Converted PDFs, extracted images/tables
├── RAG/                    # RAG system
├── config/                 # Config files for RAG, Agents and Prompt file
│   ├── agent_config.py     # Config for using Agents
│   ├── rag_config.py       # Config for using RAG
│   └── prompt.py           # Prompts Storage.
├── rag_text/               # RAG text captioning
├── rag_image/              # RAG image captioning
├── utils/                  # Helper utilities (e.g., process_documents)
├── test/                   # Testing places
├── main.py                 # Main entry point
├── chat_streamlit.py       # Main function for chatting via streamlit
├── README.md               # Main information about the repository
├── timeline.md             # Tasks and next tasks that we have done
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

### 4. Install Poppler

#### Windows

1. Download Poppler for Windows:
   - Visit: https://github.com/oschwartz10612/poppler-windows/releases/
   - Download the latest `.zip` file under **Assets** (e.g., `poppler-xx_xx_xx.zip`).

2. Extract the zip to a location like `C:\poppler`.

3. Add Poppler to PATH:
   - Open *Start > Environment Variables*.
   - Under **System Variables**, find and select `Path`, click **Edit**.
   - Click **New** and add:
     ```
     C:\poppler\Library\bin
     ```
   - Click OK and restart your terminal.

4. Verify installation:
   ```bash
   where pdfinfo
   ```
   You should see:
   ```
   C:\poppler\Library\bin\pdfinfo.exe
   ```

#### macOS (Homebrew)

```bash
brew install poppler
```

To verify:
```bash
which pdfinfo
```

#### Ubuntu/Linux

```bash
sudo apt update
sudo apt install poppler-utils
```

To verify:
```bash
which pdfinfo
```

### 5. Set up your environment variables

Copy `.env.example` and rename to `.env`, then fill in your keys:

```env
OPENAI_API_KEY=pdz-...
GOOGLE_API_KEY=pdz-...
```

### 6. Run the full pipeline

If you want to run RAG-flow individually without Agents or with Agents:

```bash
# Ingest data only
python main.py --ingest

# Use ingest and chat 
python main.py --ingest --chat

# or
python main.py --chat

# Run it on streamlit: by uploading docs or inputing urls
python main.py --app
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
| `TextAgent`      | Answers questions by retrieving from embedded text chunks |
| `ImageAgent`      | Answers questions by retrieving from embedded images of pages |
| `GeneralizeAgent`| Combines answers from multiple modalities (text, image) |
| `PlanningAgent` |     Decomposes complex questions into structured sub-questions. |
| `MergeAgent` |        Fuses sub-agent responses into a coherent final answer. |
| `VerifierAgent` |    Evaluates merged answer, determines quality, and suggests refinement. |

---

## 📥 Input Types Supported

- ✅ PDF documents (`.pdf`)
- ✅ HTML, MD, PPTX, CSV, DOCX, TXT (converted to PDF)
- ✅ Extracted images (captioning + indexing coming soon)
- 🧪 Support for `audio`, `.json`, `.xml` being tested for later release

---

## 🛠 Tools & Libraries

- 🧱 [LangChain](https://www.langchain.com/)
- 📚 [SentenceTransformers](https://www.sbert.net/)
- 📦 [ChromaDB](https://www.trychroma.com/)
- 🔍 [Docling](https://github.com/ds4sd/docling)
- 🤖 OpenAI GPT (GPT-4o-mini used for generation), Gemini, Qwen is supported

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## 🧠 Author

Built by [Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804) and Tieu Tri Bang 

---

## 🙋‍♂️ Contact

Reach out or open an [issue](https://github.com/pdz1804/M3ARAG/issues) for support or ideas.


