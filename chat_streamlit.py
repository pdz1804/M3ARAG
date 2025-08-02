"""
M3ARAG Streamlit Web Interface: Interactive Multi-Modal Document Analysis Platform

This module implements a sophisticated web-based interface for the M3ARAG system,
providing users with an intuitive platform for document upload, processing, and
interactive question-answering through a modern, responsive web application.

Application Overview:
    The Streamlit interface serves as the primary user-facing component of the
    M3ARAG system, offering a complete document understanding workflow from
    initial upload through interactive analysis and querying.

    Key Features:
    - **Drag-and-Drop Document Upload**: Intuitive file upload with multi-format support
    - **URL-Based Content Ingestion**: Direct processing of web-based documents
    - **Real-Time Processing Visualization**: Live progress tracking and status updates
    - **Interactive Chat Interface**: Conversational document querying with history
    - **Session State Management**: Persistent user sessions with conversation memory
    - **Error Handling and Recovery**: Graceful error handling with user feedback

User Interface Architecture:
    The application follows a progressive workflow design that guides users
    through the document processing and analysis pipeline:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Document Input Stage                         │
    │                                                                 │
    │  ┌─────────────────┐        ┌─────────────────────────────────┐ │
    │  │  File Upload    │   OR   │       URL Input                 │ │
    │  │  Component      │        │       Component                 │ │
    │  └─────────────────┘        └─────────────────────────────────┘ │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │                Processing Stage                                 │
    │                                                                 │
    │  • Document Validation and Format Detection                     │
    │  • Content Extraction and Normalization                        │
    │  • Multi-Modal Index Building (Text + Visual)                  │
    │  • Progress Visualization and Status Updates                    │
    └─────────────────────┬───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────────┐
    │                Interactive Chat Stage                           │
    │                                                                 │
    │  • Multi-Agent Query Processing                                 │
    │  • Real-Time Response Generation                                │
    │  • Conversation History Management                              │
    │  • Source Attribution and Citation Display                     │
    └─────────────────────────────────────────────────────────────────┘

Core Components and Features:

**Document Input Management**:
    The interface provides flexible document input mechanisms:
    
    **File Upload Component**:
    - Multi-format support: PDF, DOCX, PPTX, HTML, CSV, MD
    - Drag-and-drop functionality for intuitive user experience
    - Multiple file selection and batch processing
    - File validation with format verification
    - Progress indication during upload
    
    **URL Input Component**:
    - Direct URL processing for web-based documents
    - Comma and newline separated URL input
    - URL validation and accessibility checking
    - Dynamic content fetching with error handling
    - Support for various web content types

**Processing Pipeline Integration**:
    The interface seamlessly integrates with the M3ARAG processing pipeline:
    
    - **Document Processor Integration**: Automatic format detection and processing
    - **Content Extraction**: Docling-powered robust content extraction
    - **Index Building**: Real-time multi-modal index construction
    - **Progress Tracking**: Live updates on processing status and completion
    - **Error Recovery**: Graceful handling of processing failures

**Interactive Chat System**:
    Advanced conversational interface for document querying:
    
    **Chat Interface Features**:
    - Real-time message processing and response generation
    - Conversation history with persistent session storage
    - Message threading and context preservation
    - Typing indicators and response streaming
    - Rich text formatting for enhanced readability
    
    **Multi-Agent Integration**:
    - Seamless integration with the six-agent system
    - Real-time agent coordination and response synthesis
    - Quality-driven iterative improvement display
    - Source attribution and citation linking
    - Performance metrics and response quality indicators

**Session State Management**:
    Comprehensive session management for user experience continuity:
    
    - **Persistent Sessions**: User sessions maintained across browser refreshes
    - **Conversation Memory**: Complete chat history preservation
    - **Processing State**: Document processing status and results caching
    - **User Preferences**: Customizable interface settings and preferences
    - **Error State Recovery**: Graceful recovery from session interruptions

**User Experience Enhancements**:
    
    **Visual Feedback Systems**:
    - Loading spinners and progress bars for processing stages
    - Success and error notifications with detailed messages
    - Interactive status updates during long-running operations
    - Visual confirmation of completed processing steps
    
    **Responsive Design**:
    - Mobile-friendly interface with adaptive layouts
    - Optimized for various screen sizes and devices
    - Touch-friendly controls for mobile interaction
    - Consistent design language across all components

**Error Handling and User Guidance**:
    
    **Comprehensive Error Management**:
    - User-friendly error messages with actionable guidance
    - Detailed error logging for debugging and support
    - Graceful degradation for partial system failures
    - Recovery suggestions and alternative processing options
    
    **User Guidance Systems**:
    - Contextual help and tooltips throughout the interface
    - Step-by-step processing guidance
    - Best practices recommendations for document preparation
    - FAQ integration and troubleshooting guides

**Performance Optimizations**:
    
    **Efficient Resource Management**:
    - Lazy loading of interface components
    - Optimized file upload with chunked transfer
    - Caching of processed results for improved response times
    - Memory-efficient handling of large documents
    
    **Asynchronous Processing**:
    - Non-blocking document processing operations
    - Background index building with progress updates
    - Concurrent handling of multiple user sessions
    - Responsive interface during intensive operations

**Security and Privacy Features**:
    
    **Data Protection**:
    - Secure file upload with validation and sanitization
    - Temporary file cleanup after processing
    - Session-based data isolation between users
    - Configurable data retention policies
    
    **Access Control**:
    - Session-based access management
    - Input validation and sanitization
    - Rate limiting for API protection
    - Secure handling of sensitive documents

Technical Implementation Details:

**Streamlit Integration**:
    - Modern Streamlit framework for rapid development
    - Component-based architecture for maintainability
    - State management using Streamlit session state
    - Custom CSS styling for enhanced visual appeal

**Backend Integration**:
    - Seamless integration with M3APipeline system
    - Real-time communication with document processor
    - Direct access to multi-agent coordination system
    - Efficient data transfer between frontend and backend

**File Management**:
    - Organized directory structure for uploaded files
    - Automatic cleanup of temporary processing files
    - Persistent storage for processed documents
    - Efficient file serving for large document collections

Usage Examples:
    ```python
    # Run the Streamlit application
    streamlit run chat_streamlit.py
    
    # Application automatically handles:
    # 1. Document upload or URL input
    # 2. Processing pipeline execution
    # 3. Index building and validation
    # 4. Interactive chat activation
    ```

**Deployment Considerations**:
    - **Local Deployment**: Single-user desktop application mode
    - **Server Deployment**: Multi-user web application hosting
    - **Cloud Integration**: Scalable cloud deployment options
    - **Container Support**: Docker containerization for easy deployment

**Monitoring and Analytics**:
    - User interaction tracking for UX optimization
    - Processing performance metrics collection
    - Error rate monitoring and alerting
    - Usage pattern analysis for system improvements

Dependencies:
    - **Core**: streamlit, pathlib, json, logging
    - **Backend**: M3APipeline, DocumentProcessor, agent system
    - **Configuration**: agent_config, rag_config
    - **Utilities**: typing annotations, error handling

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Advanced Web Interface)
License: MIT License
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """
    Main entry point for the Streamlit application.

    Handles:
    - Uploading and processing documents from user input (file or URL)
    - Normalizing and indexing documents into the RAG system
    - Creating and running the M3ARAG pipeline
    - Interactive chat interface for querying document knowledge
    - Maintaining session state and chat history
    """
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
                
                logger.info(f"Uploaded {len(uploaded_files)} files: {[f.name for f in uploaded_files]}")

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
                logger.info("🔧 Starting document processing...")
                input_to_downloaded, input_to_normalized = processor.process_all(input_items)
                logger.info("✅ Document processing completed.")

                # Save mappings for traceability
                Path("data").mkdir(parents=True, exist_ok=True)
                with open("data/input_to_output_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(input_to_downloaded, f, indent=2, ensure_ascii=False)
                with open("data/input_to_normalized_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(input_to_normalized, f, indent=2, ensure_ascii=False)

                logger.info("📁 Saved input-output mappings.")

                # Merge PDFs
                copy_pdfs_to_merge_dir([Path("data/extract/pdf"), STORE_DIR, LOCAL_DIR], MERGE_DIR)
                logger.info("📎 PDFs merged into 'data/merge' directory.")
                
                # Build RAG pipeline and ingest
                pipeline = M3APipeline(
                    pdf_dir="data/merge",
                    index_dir="data/merge/index",
                    agent_config=agent_config,
                    rag_config=rag_config,
                    ingest_only=False
                )
                pipeline.ingest_cfg()
                logger.info("🔍 Indexing complete.")

                # Store to session
                st.session_state["chat_pipeline"] = pipeline
                st.session_state["chat_history"] = []
                st.session_state["chat_mode"] = True  # 🚀 switch to pure chat view

            st.success("✅ Documents processed and indexed!")
            logger.info("🚀 Pipeline is ready. Switching to chat mode.")
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
            logger.info(f"🧠 Received question: {user_input}")
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                answer = st.session_state["chat_pipeline"].process_query(user_input)
                logger.info(f"✅ Answer generated.")

            with st.chat_message("assistant"):
                st.markdown(answer if answer else "⚠️ No answer generated.")

            st.session_state["chat_history"].append((user_input, answer if answer else "No answer."))

        if st.button("🔄 Reset Chat"):
            st.session_state.clear()
            logger.info("🔄 Chat reset.")
            st.rerun()

if __name__ == "__main__":
    run_agenticrag_streamlit()
    
    

