# config/mode_manager.py
"""
Mode Manager for Enhanced Response Configuration

This module provides functionality to toggle between standard and enhanced modes
for generating longer, more detailed responses across the entire system.
"""

import os
import json
import logging
from typing import Dict, Any
from config.enhanced_config import (
    enhanced_agent_config,
    enhanced_rag_config, 
    enhanced_model_config,
    enhanced_quality_thresholds,
    response_enhancement_config
)
from config.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)

class ModeManager:
    """
    Manages configuration modes for the AgenticRAG system.
    
    Supports:
    - Standard mode: Original concise responses
    - Enhanced mode: Detailed, comprehensive responses
    - Custom mode: User-defined configuration blends
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.current_mode = "standard"
        self.mode_configs = {
            "standard": self._get_standard_config(),
            "enhanced": self._get_enhanced_config(),
        }
        
    def _get_standard_config(self) -> Dict[str, Any]:
        """Return standard (original) configuration settings."""
        return {
            "agent_config": {
                "max_loop": 2,
                "max_tasks": 2,
                "threshold": 6,
                "detailed_mode": False,
                "min_response_length": 50,
                "comprehensive_analysis": False,
            },
            "rag_config": {
                "data_dir": "data/merge/",
                "vision_retriever": "colpali",
                "text_retriever": "minilm",
                "top_k": 3,
                "chunk_size": 3000,
                "chunk_overlap": 300,
                "force_reindex": False,
                "diversity_threshold": 0.7,
                "relevance_threshold": 0.5,
            },
            "model_config": {
                "openai": {
                    "text_max_tokens": 1024,
                    "image_max_tokens": 512,
                    "temperature": 0.7,
                },
                "gemini": {
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                }
            },
            "prompt_mode": "standard"
        }
    
    def _get_enhanced_config(self) -> Dict[str, Any]:
        """Return enhanced configuration settings."""
        return {
            "agent_config": enhanced_agent_config,
            "rag_config": enhanced_rag_config,
            "model_config": enhanced_model_config,
            "quality_thresholds": enhanced_quality_thresholds,
            "response_enhancement": response_enhancement_config,
            "prompt_mode": "enhanced"
        }
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the current operating mode.
        
        Args:
            mode: Either 'standard' or 'enhanced'
            
        Returns:
            True if mode was set successfully, False otherwise
        """
        if mode not in self.mode_configs:
            logger.warning(f"Unknown mode '{mode}'. Available modes: {list(self.mode_configs.keys())}")
            return False
            
        self.current_mode = mode
        
        # Synchronize prompt manager with mode manager
        prompt_manager.set_mode(mode)
        
        logger.info(f"Mode set to: {mode}")
        return True
    
    def get_current_config(self) -> Dict[str, Any]:
        """Return the configuration for the current mode."""
        return self.mode_configs[self.current_mode]
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Return agent configuration for current mode."""
        return self.get_current_config().get("agent_config", {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Return RAG configuration for current mode."""
        return self.get_current_config().get("rag_config", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration for current mode.""" 
        return self.get_current_config().get("model_config", {})
    
    def get_prompt_mode(self) -> str:
        """Return the prompt mode for current configuration."""
        return self.get_current_config().get("prompt_mode", "standard")
    
    def is_enhanced_mode(self) -> bool:
        """Check if currently in enhanced mode."""
        return self.current_mode == "enhanced"
    
    def save_mode_config(self, filename: str = "current_mode.json") -> bool:
        """Save current mode configuration to file."""
        try:
            config_path = os.path.join(self.config_dir, filename)
            with open(config_path, 'w') as f:
                json.dump({
                    "current_mode": self.current_mode,
                    "config": self.get_current_config()
                }, f, indent=2)
            logger.info(f"Mode configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving mode configuration: {e}")
            return False
    
    def load_mode_config(self, filename: str = "current_mode.json") -> bool:
        """Load mode configuration from file."""
        try:
            config_path = os.path.join(self.config_dir, filename)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                self.current_mode = data.get("current_mode", "standard")
                logger.info(f"Mode configuration loaded: {self.current_mode}")
                return True
            else:
                logger.info(f"Configuration file {config_path} not found, using default mode")
                return False
        except Exception as e:
            logger.error(f"Error loading mode configuration: {e}")
            return False
    
    def print_current_settings(self):
        """Print current mode and key settings."""
        config = self.get_current_config()
        agent_config = config.get("agent_config", {})
        rag_config = config.get("rag_config", {})
        
        logger.info(f"Current Mode: {self.current_mode.upper()}")
        logger.info(f"Max Loops: {agent_config.get('max_loop', 'N/A')}")
        logger.info(f"Max Tasks: {agent_config.get('max_tasks', 'N/A')}")
        logger.info(f"Quality Threshold: {agent_config.get('threshold', 'N/A')}")
        logger.info(f"Top-K Retrieval: {rag_config.get('top_k', 'N/A')}")
        logger.info(f"Chunk Size: {rag_config.get('chunk_size', 'N/A')}")
        logger.info(f"Detailed Mode: {agent_config.get('detailed_mode', False)}")
        logger.info(f"Comprehensive Analysis: {agent_config.get('comprehensive_analysis', False)}")

# Global mode manager instance
mode_manager = ModeManager()
