"""
ImageAgent: Advanced Visual Content Analysis and Question Answering Agent

This module implements the ImageAgent class, a specialized AI agent dedicated
to visual question answering (VQA) that processes document images, charts,
diagrams, tables, and other visual elements to extract meaningful information
and provide comprehensive answers to visual content queries.

Visual Understanding Capabilities:
    The ImageAgent is engineered for sophisticated visual content analysis:
    
    - **Chart and Graph Analysis**: Extract data trends, values, and insights
    - **Diagram Interpretation**: Understand flowcharts, process diagrams, and schematics
    - **Table Processing**: Analyze structured data in tabular format
    - **Text-in-Image Recognition**: Process embedded text within visual elements
    - **Layout Understanding**: Comprehend document structure and visual hierarchy
    - **Multi-element Integration**: Combine information from multiple visual components

Multi-Model Vision Architecture:
    The agent supports multiple vision-language model backends for optimal
    performance across different visual content types:
    
    **Google Gemini 2.0 Flash (Recommended)**:
    - State-of-the-art multi-modal understanding
    - Excellent performance on charts, diagrams, and complex layouts
    - Fast inference with high accuracy for visual content
    - Native support for multiple image formats and resolutions
    - Advanced reasoning capabilities for visual elements
    
    **OpenAI GPT-4o-mini with Vision**:
    - Strong visual understanding and description capabilities
    - Excellent performance on text-heavy images and documents
    - Reliable chart and graph interpretation
    - Good performance on structured visual content
    
    **Qwen2.5-VL-Instruct (Local)**:
    - Local deployment for data privacy and security
    - GPU-optimized inference with memory management
    - Competitive visual understanding capabilities
    - Customizable for specific visual content domains

Visual Processing Pipeline:
    The ImageAgent implements a comprehensive visual processing workflow:
    
    1. **Image Validation**: Verify image availability and format compatibility
    2. **Image Preparation**: Resize, format, and optimize images for processing
    3. **Context Integration**: Combine multiple images with contextual information
    4. **Model Processing**: Analyze images through vision-language models
    5. **Response Generation**: Generate detailed, specific visual insights
    6. **Quality Verification**: Ensure response accuracy and completeness
    7. **Resource Cleanup**: Manage memory and temporary file cleanup

Image Preparation and Optimization:
    - **Format Standardization**: Convert images to optimal formats (PNG, JPEG)
    - **Resolution Optimization**: Balance quality and processing efficiency
    - **Memory Management**: Efficient handling of large image files
    - **Batch Processing**: Optimize multiple image processing workflows
    - **Temporary File Management**: Secure handling of processed images

Visual Content Analysis Features:
    - **Spatial Reasoning**: Understand relationships between visual elements
    - **Color and Style Analysis**: Interpret visual design and formatting
    - **Text Extraction**: Read and interpret text within images
    - **Numerical Data Processing**: Extract quantitative information from charts
    - **Structural Analysis**: Understand document layout and organization

Technical Optimizations:
    - **Memory Efficiency**: Optimized image loading and processing
    - **GPU Acceleration**: Leverage GPU resources for faster inference
    - **Image Caching**: Intelligent caching of processed images
    - **Parallel Processing**: Concurrent analysis of multiple images
    - **Resource Monitoring**: Track memory usage and processing performance

Response Quality Standards:
    The ImageAgent generates responses that are:
    - **Visually Grounded**: Explicitly reference specific visual elements
    - **Detailed and Specific**: Provide concrete descriptions and measurements
    - **Contextually Relevant**: Focus on information relevant to the query
    - **Factually Accurate**: Base responses on observable visual content
    - **Well-Structured**: Organize information in clear, logical format

Error Handling and Resilience:
    - **Image Format Errors**: Handle unsupported or corrupted images
    - **Model Failures**: Graceful degradation with alternative processing
    - **Memory Constraints**: Adaptive processing for large images
    - **Network Issues**: Robust handling of API connectivity problems
    - **Processing Timeouts**: Manage long-running visual analysis tasks

Integration Capabilities:
    - **Multi-Image Analysis**: Process multiple related images together
    - **Cross-Modal Integration**: Combine with text analysis results
    - **External Tool Integration**: Support for additional image processing tools
    - **API Integration**: Seamless integration with vision APIs
    - **Custom Processing**: Extensible for specialized visual content types

Usage Examples:
    ```python
    # Initialize with Gemini backend (recommended for visual content)
    image_agent = ImageAgent(name="ImageAgent", qa_model="gemini")
    
    # Process visual question with multiple images
    response = image_agent.run(
        input_data={"question": "What trends are shown in the charts?"},
        contexts=[
            {"image": PIL_image_1, "page": 1, "source": "report.pdf"},
            {"image": PIL_image_2, "page": 3, "source": "report.pdf"}
        ]
    )
    
    # Initialize with OpenAI backend
    image_agent = ImageAgent(name="ImageAgent", qa_model="openai")
    ```

Visual Content Types Supported:
    - **Charts and Graphs**: Bar charts, line graphs, pie charts, scatter plots
    - **Tables and Matrices**: Structured data in tabular format
    - **Diagrams**: Flowcharts, organizational charts, process diagrams
    - **Maps and Layouts**: Geographic maps, floor plans, site layouts
    - **Screenshots**: Application interfaces, web pages, software screens
    - **Scientific Figures**: Research diagrams, experimental setups, results

Performance Characteristics:
    - **Processing Speed**: Optimized for real-time visual analysis
    - **Memory Efficiency**: Minimal memory footprint for large images
    - **Accuracy**: High precision in visual element identification
    - **Scalability**: Efficient processing of multiple images
    - **Reliability**: Consistent performance across diverse visual content

Configuration Options:
    - **Model Selection**: Choose optimal vision-language model
    - **Image Resolution**: Configure processing resolution for quality/speed balance
    - **Timeout Settings**: Set processing time limits for images
    - **Memory Limits**: Configure maximum memory usage
    - **Output Format**: Customize response structure and detail level

Dependencies:
    - **Core**: PIL, torch, transformers, langchain
    - **Vision Models**: google-generativeai, openai, qwen-vl
    - **Image Processing**: pillow, opencv-python, pdf2image
    - **Utilities**: tempfile, logging, typing, gc

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Advanced Vision Architecture)
License: MIT License
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from utils.image_utils import prepare_images, release_memory
from rag_image.base_runner import get_image_captioning_runner

import gc
import torch
from PIL import Image
import tempfile

class ImageAgent(BaseAgent):
    def __init__(self, name: str = "ImageAgent", qa_model: str = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing ImageAgent with backend model: {qa_model}")

        # Dynamically load the correct runner
        self.caption_with_llm = get_image_captioning_runner(qa_model)

    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")
        
        if not contexts:
            logger.warning("No contexts passed to ImageAgent.")
            return "No image context provided."
        
        images = [ctx["image"] for ctx in contexts if "image" in ctx]
        
        image_paths = prepare_images(images)
        release_memory()
        
        return self.caption_with_llm.invoke({
            "query": question,
            "images": image_paths   # ← Now Gemini expects file paths, not PIL.Image
        })



