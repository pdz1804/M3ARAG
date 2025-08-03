# config/prompt_manager.py
"""
Dynamic Prompt Manager for Mode-Aware Response Generation

This module provides intelligent prompt selection based on the current
operating mode (standard vs enhanced) to generate appropriate response
styles and lengths.
"""

import logging
from config.prompt import (
    TEXT_PROMPT_TEMPLATE, 
    IMAGE_PROMPT_TEMPLATE,
    GENERALIZED_PROMPT,
    PLANNING_PROMPT_JSON,
    MERGE_PROMPT,
    VERIFICATION_PROMPT
)

from config.enhanced_prompts import (
    ENHANCED_TEXT_PROMPT_TEMPLATE,
    ENHANCED_IMAGE_PROMPT_TEMPLATE, 
    ENHANCED_GENERALIZED_PROMPT,
    ENHANCED_PLANNING_PROMPT,
    ENHANCED_MERGE_PROMPT,
    ENHANCED_VERIFICATION_PROMPT
)

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompt templates based on current system mode.
    
    Automatically selects appropriate prompts for:
    - Standard mode: Original concise prompts
    - Enhanced mode: Detailed, comprehensive prompts
    """
    
    def __init__(self):
        self.current_mode = "standard"
        
    def set_mode(self, mode: str):
        """Set the current prompt mode."""
        if mode in ["standard", "enhanced"]:
            self.current_mode = mode
        else:
            logger.warning(f"Unknown prompt mode '{mode}', keeping current mode: {self.current_mode}")
    
    def get_text_prompt(self) -> str:
        """Get text analysis prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_TEXT_PROMPT_TEMPLATE
        return TEXT_PROMPT_TEMPLATE
    
    def get_image_prompt(self) -> str:
        """Get image analysis prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_IMAGE_PROMPT_TEMPLATE
        return IMAGE_PROMPT_TEMPLATE
    
    def get_generalize_prompt(self) -> str:
        """Get generalization prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_GENERALIZED_PROMPT
        return GENERALIZED_PROMPT
    
    def get_planning_prompt(self) -> str:
        """Get planning prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_PLANNING_PROMPT
        return PLANNING_PROMPT_JSON
    
    def get_merge_prompt(self) -> str:
        """Get merge prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_MERGE_PROMPT
        return MERGE_PROMPT
    
    def get_verification_prompt(self) -> str:
        """Get verification prompt based on current mode."""
        if self.current_mode == "enhanced":
            return ENHANCED_VERIFICATION_PROMPT
        return VERIFICATION_PROMPT
    
    def is_enhanced_mode(self) -> bool:
        """Check if currently using enhanced prompts."""
        return self.current_mode == "enhanced"
    
    def get_mode_info(self) -> dict:
        """Get information about current mode and expected response characteristics."""
        if self.current_mode == "enhanced":
            return {
                "mode": "enhanced",
                "response_style": "comprehensive and detailed",
                "expected_length": "200-400+ words per response",
                "analysis_depth": "multi-layered with examples and reasoning",
                "citation_style": "comprehensive source attribution",
                "structure": "well-organized with multiple paragraphs"
            }
        else:
            return {
                "mode": "standard", 
                "response_style": "concise and direct",
                "expected_length": "50-150 words per response",
                "analysis_depth": "focused on key points",
                "citation_style": "basic source references",
                "structure": "brief paragraphs"
            }

# Global prompt manager instance
prompt_manager = PromptManager()
