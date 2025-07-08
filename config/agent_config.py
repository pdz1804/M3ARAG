# config/agent_config.py

agent_config = {
    "use_text": True,               # Whether to activate the TextAgent
    "use_image": True,              # Whether to activate the ImageAgent
    
    # Shared QA model name for all agents (currently support: "qwen", "gemini", "openai")
    "qa_text": "openai",
    
    # Shared QA model for image-based agents (currently support: "qwen", "gemini", "openai")             
    "qa_image": "gemini",            
    "qa_generalize": "openai",         
    "qa_finalize": "openai",          
}


