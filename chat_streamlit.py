# chat_streamlit.py
import streamlit as st
from pathlib import Path
import json

from utils.document_processor import DocumentProcessor, copy_pdfs_to_merge_dir
from pipeline.M3APipeline import M3APipeline
from config.agent_config import agent_config
from config.rag_config import rag_config

# === Constants ===
STORE_DIR = Path("data/store")
EXTRACT_DIR = Path("data/extract")
MERGE_DIR = Path("data/merge")
LOCAL_DIR = Path("local")

def run_agenticrag_streamlit():
    st.title("📄 M3ARAG Document Understanding")

    processor = DocumentProcessor(store_dir=STORE_DIR, extract_dir=EXTRACT_DIR)

    if not st.session_state.get("chat_mode", False):
        mode = st.radio("Choose Input Method", ["📤 Upload Documents", "🌐 Enter URLs"])

        input_items = []

        if mode == "📤 Upload Documents":
            uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "pptx", "html", "csv", "md"], accept_multiple_files=True)
            if uploaded_files:
                LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                for file in uploaded_files:
                    path = LOCAL_DIR / file.name
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    input_items.append(str(path))

        elif mode == "🌐 Enter URLs":
            url_input = st.text_area("Enter URLs (comma or newline separated)", key="url_text_input")
            if st.button("Add URLs"):
                raw_urls = url_input.replace(",", "\n").splitlines()
                clean_urls = [url.strip() for url in raw_urls if url.strip()]
                st.session_state["url_inputs"] = clean_urls
                st.rerun()

            if "url_inputs" in st.session_state:
                input_items.extend(st.session_state["url_inputs"])
                st.markdown("✅ Added URLs:")
                for url in st.session_state["url_inputs"]:
                    st.code(url)

        if input_items and st.button("🚀 Process Documents"):
            with st.spinner("Processing..."):
                input_to_downloaded, input_to_normalized = processor.process_all(input_items)

                # Save mappings
                Path("data").mkdir(parents=True, exist_ok=True)
                with open("data/input_to_output_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(input_to_downloaded, f, indent=2, ensure_ascii=False)
                with open("data/input_to_normalized_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(input_to_normalized, f, indent=2, ensure_ascii=False)

                # Merge PDFs
                copy_pdfs_to_merge_dir([Path("data/extract/pdf"), STORE_DIR, LOCAL_DIR], MERGE_DIR)

                # Ingest pipeline
                pipeline = M3APipeline(
                    pdf_dir="data/merge",
                    index_dir="data/merge/index",
                    agent_config=agent_config,
                    rag_config=rag_config,
                    ingest_only=False
                )
                pipeline.ingest_cfg()

                st.session_state["chat_pipeline"] = pipeline
                st.session_state["chat_history"] = []
                st.session_state["chat_mode"] = True  # 🚀 switch to pure chat view

            st.success("✅ Documents processed and indexed!")
            st.rerun()

    # === Chat Interface ===
    if "chat_pipeline" in st.session_state:
        st.subheader("💬 Chat with Your Documents")

        # Display chat history
        for q, a in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

        user_input = st.chat_input("Ask a question...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                answer = st.session_state["chat_pipeline"].process_query(user_input)

            with st.chat_message("assistant"):
                st.markdown(answer if answer else "⚠️ No answer generated.")

            st.session_state["chat_history"].append((user_input, answer if answer else "No answer."))

        if st.button("🔄 Reset Chat"):
            st.session_state.pop("chat_pipeline", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("url_inputs", None)
            st.session_state["chat_mode"] = False  # 👈 go back to upload screen
            st.rerun()

if __name__ == "__main__":
    run_agenticrag_streamlit()
    
    

