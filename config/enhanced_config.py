# config/enhanced_config.py
"""
Enhanced Configuration for Longer and More Detailed Answers

This configuration module provides enhanced parameters to enable the system 
to generate longer, more comprehensive, and detailed responses.

Key Enhancements:
1. Increased token limits for all model backends
2. Enhanced retrieval parameters for more context
3. Modified thresholds for higher quality requirements
4. Extended processing parameters for comprehensive analysis
"""

# Enhanced Agent Configuration
enhanced_agent_config = {
    "use_text": True,
    "use_image": True,
    
    # Model configurations
    "qa_text": "openai",
    "qa_generalize": "openai",         
    "qa_merge": "openai",      
    "qa_planning": "openai",
    "qa_verifier": "openai",
    "qa_image": "gemini",
    
    # Enhanced iteration parameters for detailed answers
    "max_loop": 4,              # Increased from 4 to allow more refinement iterations
    "max_tasks": 3,             # Increased from 3 to allow more detailed task decomposition
    "threshold": 7,           # Increased from 6 to require higher quality responses
    
    # New parameters for enhanced responses
    "detailed_mode": True,      # Enable detailed answer generation mode
    "min_response_length": 300, # Minimum expected response length (words)
    "comprehensive_analysis": True,  # Enable comprehensive analysis mode
}

# Enhanced RAG Configuration
enhanced_rag_config = {
    "data_dir": "data/merge/",
    "vision_retriever": "colpali",
    "text_retriever": "minilm",
    
    # Enhanced retrieval parameters
    "top_k": 5,                # Increased from 3 to retrieve more relevant chunks
    "chunk_size": 4000,        # Increased from 3000 for larger context chunks
    "chunk_overlap": 500,      # Increased from 300 for better continuity
    "force_reindex": False,
    
    # New parameters for enhanced retrieval
    "diversity_threshold": 0.7, # Ensure diverse content retrieval
    "relevance_threshold": 0.3, # Lower threshold to include more potentially relevant content
}

# Enhanced Model Parameters
enhanced_model_config = {
    # OpenAI Enhanced Parameters
    "openai": {
        "text_max_tokens": 2048,    # Increased from 1024
        "image_max_tokens": 1024,   # Increased from 512
        "temperature": 0.7,         # Balanced creativity and consistency
        "top_p": 0.95,             # High nucleus sampling for comprehensive responses
        "presence_penalty": 0.1,    # Slight penalty to avoid repetition
        "frequency_penalty": 0.1,   # Slight penalty for word repetition
    },
    
    # Gemini Enhanced Parameters
    "gemini": {
        "temperature": 0.6,         # Slightly increased for more detailed responses
        "top_p": 0.95,             # High nucleus sampling
        "top_k": 40,               # Moderate top-k for balance
        "max_output_tokens": 2048,  # Explicit token limit for Gemini
    },
    
    # Qwen Enhanced Parameters (for local deployment)
    "qwen": {
        "max_new_tokens": 2048,     # Increased token limit
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.1,
    }
}

# Quality Thresholds for Enhanced Mode
enhanced_quality_thresholds = {
    "minimum_score": 7.5,           # Higher minimum quality score
    "excellent_score": 9.0,         # Score for excellent responses
    "improvement_threshold": 1.0,   # Minimum improvement per iteration
    "max_iterations": 3,            # Maximum refinement iterations
    "convergence_threshold": 0.2,   # Score improvement convergence threshold
}

# Response Enhancement Configuration
response_enhancement_config = {
    "enable_detailed_explanations": True,
    "include_reasoning_steps": True,
    "provide_multiple_perspectives": True,
    "include_examples_when_relevant": True,
    "comprehensive_source_citations": True,
    "structured_response_format": True,
}
