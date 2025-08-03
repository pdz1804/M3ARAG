# config/agent_config.py

agent_config = {
    "use_text": True,               # Whether to activate the TextAgent
    "use_image": True,              # Whether to activate the ImageAgent
    
    # Shared QA model name for all agents - text (currently support: "qwen", "gemini", "openai")
    "qa_text": "openai",
    "qa_generalize": "openai",         
    "qa_merge": "openai",      
    "qa_planning": "openai",
    "qa_verifier": "openai",  
    
    # Shared QA model for image-based agents (currently support: "qwen", "gemini", "openai")             
    "qa_image": "gemini",            
    
    # Enhanced parameters for detailed answers
    "max_loop": 3,                  # Increased from 4 to allow more refinement iterations
    "max_tasks": 3,                 # Increased from 3 to allow more detailed task decomposition
    "threshold": 7,               # Increased from 6 to require higher quality responses
    
    # New parameters for enhanced mode
    "detailed_mode": True,          # Enable detailed answer generation mode
    "min_response_length": 200,     # Minimum expected response length (words)
    "comprehensive_analysis": True, # Enable comprehensive analysis mode
        
}


