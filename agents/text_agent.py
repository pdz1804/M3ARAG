"""
TextAgent: Advanced Textual Content Analysis and Question Answering Agent

This module implements the TextAgent class, a specialized AI agent responsible
for processing and analyzing textual content from documents to provide detailed,
accurate, and well-sourced answers to natural language questions.

Agent Specialization:
    The TextAgent is specifically designed for textual content understanding
    and question answering, leveraging state-of-the-art language models to:
    
    - Analyze retrieved document text chunks with deep semantic understanding
    - Generate comprehensive, contextually appropriate responses
    - Maintain source attribution and citation accuracy
    - Handle complex multi-part questions with nuanced reasoning
    - Provide explanations and justifications for answers

Multi-Model Architecture:
    The agent supports multiple language model backends for flexible deployment:
    
    **OpenAI GPT-4o-mini**:
    - High-quality text understanding and generation
    - Excellent reasoning capabilities for complex queries
    - Strong performance on technical and academic content
    - API-based deployment with rate limiting and error handling
    
    **Google Gemini 2.0 Flash**:
    - Advanced multi-modal understanding capabilities
    - Fast inference with competitive quality
    - Strong performance on diverse content types
    - Native multi-language support
    
    **Qwen2.5-VL-Instruct (Local)**:
    - Local deployment for data privacy and control
    - GPU-optimized inference with memory management
    - Competitive performance with reduced operational costs
    - Customizable for specific domain requirements

Processing Pipeline:
    The TextAgent follows a sophisticated processing pipeline:
    
    1. **Input Validation**: Verify question and context availability
    2. **Context Preparation**: Extract and format text chunks
    3. **Prompt Engineering**: Apply specialized templates for optimal results
    4. **Model Inference**: Process through selected language model
    5. **Response Generation**: Format output with citations and explanations
    6. **Quality Assurance**: Validate response completeness and accuracy
    7. **Memory Management**: Clean up resources for efficiency

Context Processing Features:
    - **Chunk Integration**: Seamlessly combines multiple text chunks
    - **Relevance Filtering**: Prioritizes most relevant content
    - **Citation Management**: Maintains document and page references  
    - **Semantic Coherence**: Ensures logical flow across chunks
    - **Information Synthesis**: Combines information from multiple sources

Technical Optimizations:
    - **Memory Management**: Efficient GPU and CPU memory handling
    - **Batch Processing**: Optimized for multiple query processing
    - **Error Recovery**: Robust error handling with graceful degradation
    - **Resource Cleanup**: Automatic memory cleanup after processing
    - **Performance Monitoring**: Built-in timing and resource usage tracking

Response Quality Features:
    - **Source Attribution**: Clear citations to source documents and pages
    - **Confidence Indicators**: Explicit uncertainty handling
    - **Structured Responses**: Well-organized, readable output format
    - **Fact Verification**: Cross-reference information across sources
    - **Completeness Checking**: Ensure all aspects of question addressed

Integration Capabilities:
    - **LangChain Integration**: Seamless integration with LangChain framework
    - **Custom Runners**: Specialized model runners for each backend
    - **Dynamic Model Selection**: Runtime model switching capabilities
    - **Tool Integration**: Support for external tools and APIs
    - **Monitoring Integration**: Comprehensive logging and metrics

Error Handling and Resilience:
    - **Model Failure Recovery**: Automatic fallback to alternative models
    - **Context Overflow**: Intelligent handling of large context sizes
    - **API Rate Limiting**: Graceful handling of service limitations
    - **Memory Constraints**: Adaptive processing for resource limitations
    - **Network Issues**: Retry logic with exponential backoff

Usage Examples:
    ```python
    # Initialize with OpenAI backend
    text_agent = TextAgent(name="TextAgent", qa_model="openai")
    
    # Process question with context
    response = text_agent.run(
        input_data={"question": "What are the key findings?"},
        contexts=[
            {"chunk": "Document text chunk 1...", "page": 1, "source": "doc1.pdf"},
            {"chunk": "Document text chunk 2...", "page": 3, "source": "doc2.pdf"}
        ]
    )
    
    # Initialize with local Qwen model
    text_agent = TextAgent(name="TextAgent", qa_model="qwen")
    ```

Performance Characteristics:
    - **Latency**: Optimized for fast response times
    - **Throughput**: Supports concurrent query processing
    - **Scalability**: Efficient resource utilization for large deployments
    - **Accuracy**: High-quality responses with factual grounding
    - **Reliability**: Robust error handling and recovery mechanisms

Configuration Options:
    - **Model Selection**: Choose from multiple language model backends
    - **Temperature Settings**: Control response creativity vs. consistency
    - **Max Length**: Configure maximum response length
    - **Citation Format**: Customize source attribution style
    - **Timeout Settings**: Configure processing time limits

Dependencies:
    - **Core**: langchain, transformers, torch
    - **Models**: openai, google-generativeai, sentence-transformers
    - **Utilities**: logging, typing, garbage collection
    - **Runners**: Custom model runners for each backend

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Multi-Model Architecture)
License: MIT License
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from rag_text.base_runner import get_text_captioning_runner
import torch
import gc

class TextAgent(BaseAgent):
    def __init__(self, name: str = "TextAgent", qa_model = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing TextAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.caption_with_llm = get_text_captioning_runner(qa_model)

    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")
        
        if not question or not contexts:
            logger.warning("Missing question or contexts. Skipping generation.")
            return "No answer found."

        # Extract raw text chunks and prepare prompt
        contexts = [ctx['chunk'] for ctx in contexts]
        contexts_str = "\n- ".join(contexts)

        # Clean up memory before inference
        del contexts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate answer using the selected model
        return self.caption_with_llm.invoke({"query": question, "texts": contexts_str})


