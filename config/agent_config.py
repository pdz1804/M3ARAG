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
    
    # --- new code --- 
    "max_loop": 6,
    "max_tasks": 4,
    "threshold": 6,
    # --- end new code --- 
        
}


