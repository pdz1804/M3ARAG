
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st
from pathlib import Path
import json
from utils.document_processor import DocumentProcessor, copy_pdfs_to_merge_dir
from pipeline.M3APipeline import M3APipeline
from config.mode_manager import ModeManager

# === Constants ===
STORE_DIR = Path("data/store")
EXTRACT_DIR = Path("data/extract")
MERGE_DIR = Path("data/merge")
LOCAL_DIR = Path("local")

def run_agenticrag_streamlit():
    """
    Enhanced Streamlit application with mode selection and improved UI.

    Features:
    - Mode selection (Standard vs Enhanced) at startup
    - Enhanced UI with better layout and visual feedback
    - Mode-aware configuration management
    - Uploading and processing documents from user input (file or URL)
    - Normalizing and indexing documents into the RAG system
    - Creating and running the M3ARAG pipeline with selected mode
    - Interactive chat interface for querying document knowledge
    - Maintaining session state and chat history
    """
    
    # Page configuration
    st.set_page_config(
        page_title="M3ARAG - Multi-Modal Multi-Agent RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-indicator {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .standard-mode {
        background-color: #e8f4f8;
        color: #1f77b4;
        border: 2px solid #1f77b4;
    }
    .enhanced-mode {
        background-color: #fff2e8;
        color: #ff7f0e;
        border: 2px solid #ff7f0e;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    /* Custom styling for better visual hierarchy */
    .stExpander > div > div > div > div {
        background-color: #f8f9fa;
    }
    /* Better metric styling */
    .metric-container {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize mode manager
    if "mode_manager" not in st.session_state:
        st.session_state["mode_manager"] = ModeManager()
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 M3ARAG Document Understanding</h1>', unsafe_allow_html=True)
    
    # Sidebar for mode selection and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        if "selected_mode" not in st.session_state:
            st.session_state["selected_mode"] = "standard"
        
        mode_choice = st.radio(
            "Select Response Mode:",
            options=["standard", "enhanced"],
            index=0 if st.session_state["selected_mode"] == "standard" else 1,
            help="Choose between standard concise responses or enhanced detailed responses"
        )
        
        if mode_choice != st.session_state["selected_mode"]:
            st.session_state["selected_mode"] = mode_choice
            st.session_state["mode_manager"].set_mode(mode_choice)
            logger.info(f"Mode changed to: {mode_choice}")
            
        # Mode Information Display
        current_config = st.session_state["mode_manager"].get_current_config()
        mode_class = "enhanced-mode" if mode_choice == "enhanced" else "standard-mode"
        
        st.markdown(f'''
        <div class="mode-indicator {mode_class}">
            📊 Current Mode: {mode_choice.title()}
        </div>
        ''', unsafe_allow_html=True)
        
        # Display mode-specific settings
        if mode_choice == "enhanced":
            st.markdown("### 🚀 Enhanced Features:")
            st.markdown(f"- **Max Iterations:** {current_config['agent_config']['max_loop']}")
            st.markdown(f"- **Max Tasks:** {current_config['agent_config']['max_tasks']}")
            st.markdown(f"- **Quality Threshold:** {current_config['agent_config']['threshold']}")
            st.markdown(f"- **Top-K Retrieval:** {current_config['rag_config']['top_k']}")
            st.markdown(f"- **Chunk Size:** {current_config['rag_config']['chunk_size']}")
            st.markdown("- **Comprehensive Analysis:** ✅")
        else:
            st.markdown("### 📝 Standard Features:")
            st.markdown(f"- **Max Iterations:** {current_config['agent_config']['max_loop']}")
            st.markdown(f"- **Max Tasks:** {current_config['agent_config']['max_tasks']}")
            st.markdown(f"- **Quality Threshold:** {current_config['agent_config']['threshold']}")
            st.markdown(f"- **Top-K Retrieval:** {current_config['rag_config']['top_k']}")
            st.markdown(f"- **Chunk Size:** {current_config['rag_config']['chunk_size']}")
            st.markdown("- **Quick Responses:** ✅")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 System Status")
        if "chat_pipeline" in st.session_state:
            st.markdown("🟢 **Ready for Chat**")
            st.markdown(f"📁 **Documents:** Processed")
            st.markdown(f"💬 **Chat History:** {len(st.session_state.get('chat_history', []))} messages")
        else:
            st.markdown("🟡 **Waiting for Documents**")
    
    processor = DocumentProcessor(store_dir=STORE_DIR, extract_dir=EXTRACT_DIR)

    # Main content area
    if not st.session_state.get("chat_mode", False):
        # Document input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📄 Document Input")
            
            # Input method selection with enhanced UI
            input_method = st.selectbox(
                "Choose how to add your documents:",
                options=["📤 Upload Documents", "🌐 Enter URLs"],
                help="Select whether to upload files or provide URLs to documents"
            )

        with col2:
            st.markdown("## 🎯 Quick Guide")
            
            # Create a more professional supported formats display
            st.markdown("### 📁 Supported Formats")
            
            # Use columns for better layout
            format_col1, format_col2 = st.columns(2)
            
            with format_col1:
                st.markdown('''
                **Documents:**
                - 📄 PDF
                - 📝 DOCX
                - 📊 PPTX
                ''')
            
            with format_col2:
                st.markdown('''
                **Web & Data:**
                - 🌐 HTML
                - 📋 CSV
                - 📝 Markdown
                ''')
            
            # Add usage tips
            st.markdown("### 💡 Tips")
            st.info("💡 Upload multiple files at once for batch processing")
            st.info("🔗 URLs will be automatically downloaded and processed")

        input_items = []

        if input_method == "📤 Upload Documents":
            st.markdown("### 📤 Upload Your Documents")
            uploaded_files = st.file_uploader(
                "Drag and drop files here or click to browse",
                type=["pdf", "docx", "pptx", "html", "csv", "md"], 
                accept_multiple_files=True,
                help="You can upload multiple files at once. All supported formats will be processed automatically."
            )
            
            if uploaded_files:
                LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                
                # Display uploaded files info with better formatting
                st.markdown("### 📋 Uploaded Files")
                
                # Create a container for better file display
                with st.container():
                    for idx, file in enumerate(uploaded_files):
                        # Create expandable file info
                        with st.expander(f"📄 {file.name}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Size", f"{file.size / 1024:.1f} KB")
                            with col2:
                                st.metric("Type", file.type.split('/')[-1].upper() if file.type else "Unknown")
                            with col3:
                                st.metric("Status", "✅ Ready")
                        
                        path = LOCAL_DIR / file.name
                        with open(path, "wb") as f:
                            f.write(file.getbuffer())
                        input_items.append(str(path))
                
                # Summary info
                total_size = sum(file.size for file in uploaded_files) / 1024
                st.success(f"📦 {len(uploaded_files)} files ready ({total_size:.1f} KB total)")
                logger.info(f"Uploaded {len(uploaded_files)} files: {[f.name for f in uploaded_files]}")

        elif input_method == "🌐 Enter URLs":
            st.markdown("### 🌐 Add Document URLs")
            url_input = st.text_area(
                "Enter URLs (one per line or comma-separated)",
                height=100,
                placeholder="https://example.com/document.pdf\nhttps://example.com/paper.html",
                help="Enter URLs to documents you want to process. Supports PDF, HTML, and other web content."
            )
            
            if st.button("➕ Add URLs", type="secondary"):
                if url_input.strip():
                    raw_urls = url_input.replace(",", "\n").splitlines()
                    clean_urls = [url.strip() for url in raw_urls if url.strip()]
                    st.session_state["url_inputs"] = clean_urls
                    st.rerun()
                else:
                    st.warning("Please enter at least one URL")

            if "url_inputs" in st.session_state and st.session_state["url_inputs"]:
                st.markdown("### ✅ Added URLs")
                
                # Display URLs in a more organized way
                with st.container():
                    for i, url in enumerate(st.session_state["url_inputs"], 1):
                        with st.expander(f"🔗 URL {i}", expanded=False):
                            st.code(url)
                            # Show URL preview info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Status", "✅ Added")
                            with col2:
                                domain = url.split('/')[2] if url.startswith('http') else "Unknown"
                                st.metric("Domain", domain)
                
                st.success(f"🌐 {len(st.session_state['url_inputs'])} URLs ready for processing")
                input_items.extend(st.session_state["url_inputs"])

        # Processing section
        if input_items:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                process_button = st.button(
                    f"🚀 Process {len(input_items)} Document(s) in {mode_choice.title()} Mode",
                    type="primary",
                    use_container_width=True
                )
                
            if process_button:
                with st.spinner("🔄 Processing documents... This may take a few minutes."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Document processing
                        status_text.text("📄 Processing documents...")
                        progress_bar.progress(20)
                        logger.info("🔧 Starting document processing...")
                        
                        input_to_downloaded, input_to_normalized = processor.process_all(input_items)
                        progress_bar.progress(40)
                        logger.info("✅ Document processing completed.")

                        # Step 2: Save mappings
                        status_text.text("💾 Saving mappings...")
                        Path("data").mkdir(parents=True, exist_ok=True)
                        with open("data/input_to_output_mapping.json", "w", encoding="utf-8") as f:
                            json.dump(input_to_downloaded, f, indent=2, ensure_ascii=False)
                        with open("data/input_to_normalized_mapping.json", "w", encoding="utf-8") as f:
                            json.dump(input_to_normalized, f, indent=2, ensure_ascii=False)
                        progress_bar.progress(60)
                        logger.info("📁 Saved input-output mappings.")

                        # Step 3: Merge PDFs
                        status_text.text("📎 Merging documents...")
                        copy_pdfs_to_merge_dir([Path("data/extract/pdf"), STORE_DIR, LOCAL_DIR], MERGE_DIR)
                        progress_bar.progress(80)
                        logger.info("📎 PDFs merged into 'data/merge' directory.")
                        
                        # Step 4: Build RAG pipeline with selected mode configuration
                        status_text.text("🔍 Building search index...")
                        current_config = st.session_state["mode_manager"].get_current_config()
                        
                        pipeline = M3APipeline(
                            pdf_dir="data/merge",
                            index_dir="data/merge/index",
                            agent_config=current_config["agent_config"],
                            rag_config=current_config["rag_config"],
                            ingest_only=False
                        )
                        pipeline.ingest_cfg()
                        progress_bar.progress(100)
                        logger.info("🔍 Indexing complete.")

                        # Store to session
                        st.session_state["chat_pipeline"] = pipeline
                        st.session_state["chat_history"] = []
                        st.session_state["chat_mode"] = True
                        st.session_state["processed_docs_count"] = len(input_items)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        logger.error(f"Processing error: {e}")
                        return

                st.success(f"✅ Successfully processed {len(input_items)} documents in {mode_choice.title()} mode!")
                logger.info("🚀 Pipeline is ready. Switching to chat mode.")
                st.rerun()

    # === Enhanced Chat Interface ===
    if "chat_pipeline" in st.session_state:
        st.markdown("## 💬 Chat with Your Documents")
        
        # Chat statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", st.session_state.get("processed_docs_count", 0))
        with col2:
            st.metric("💬 Messages", len(st.session_state.get("chat_history", [])))
        with col3:
            st.metric("🎯 Mode", st.session_state["selected_mode"].title())
        with col4:
            if st.button("🔄 Reset Chat", help="Clear chat history and start over"):
                st.session_state["chat_history"] = []
                logger.info("🔄 Chat history cleared.")
                st.rerun()

        st.markdown("---")

        # Chat container with custom styling
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, (q, a) in enumerate(st.session_state.get("chat_history", [])):
                # User message
                with st.chat_message("user"):
                    st.markdown(q)
                
                # Assistant message with enhanced formatting
                with st.chat_message("assistant"):
                    if a and a != "No answer.":
                        st.markdown(a)
                    else:
                        st.warning("⚠️ No answer generated for this question.")

        # Chat input
        user_input = st.chat_input(
            placeholder=f"Ask a question about your documents... ({st.session_state['selected_mode'].title()} mode)",
            key="chat_input"
        )
        
        if user_input:
            logger.info(f"🧠 Received question: {user_input}")
            
            # Add user message to chat
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response with mode-aware processing
            with st.chat_message("assistant"):
                with st.spinner(f"🤔 Generating {st.session_state['selected_mode']} response..."):
                    try:
                        answer = st.session_state["chat_pipeline"].process_query(user_input)
                        logger.info(f"✅ Answer generated in {st.session_state['selected_mode']} mode.")
                        
                        if answer and answer.strip():
                            st.markdown(answer)
                        else:
                            st.warning("⚠️ No answer generated. Please try rephrasing your question.")
                            answer = "No answer generated."
                            
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Chat error: {e}")
                        answer = error_msg

            # Update chat history
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append((user_input, answer))
            
            # Auto-rerun to show the new message
            st.rerun()

        # Additional options at the bottom
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Process New Documents", help="Go back to document input"):
                st.session_state["chat_mode"] = False
                logger.info("🔄 Returning to document input mode.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Complete Reset", help="Clear everything and start over"):
                st.session_state.clear()
                logger.info("🔄 Complete system reset.")
                st.rerun()
                
        with col3:
            if st.button("📥 Export Chat", help="Download chat history"):
                if st.session_state.get("chat_history"):
                    chat_export = {
                        "mode": st.session_state["selected_mode"],
                        "timestamp": st.session_state.get("chat_timestamp", "unknown"),
                        "documents_processed": st.session_state.get("processed_docs_count", 0),
                        "chat_history": st.session_state["chat_history"]
                    }
                    st.download_button(
                        "📄 Download JSON",
                        data=json.dumps(chat_export, indent=2),
                        file_name=f"m3arag_chat_{st.session_state['selected_mode']}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No chat history to export")
        
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🤖 M3ARAG - Multi-Modal Multi-Agent RAG System | 
        Built by PDZ & Bang | 
        <a href="https://github.com/pdz1804/M3ARAG" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run_agenticrag_streamlit()
    
    

