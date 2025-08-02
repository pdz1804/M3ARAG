"""
BaseAgent: Abstract Foundation for Multi-Modal AI Agents

This module defines the abstract base class that serves as the foundational
interface for all specialized agents in the M3ARAG multi-agent system.
It ensures consistent behavior, standardized interfaces, and extensible
architecture across the entire agent ecosystem.

Design Philosophy:
    The BaseAgent class implements the Template Method pattern to provide
    a consistent interface while allowing specialized implementations for
    different agent types. This design ensures:
    
    - **Interface Consistency**: All agents follow the same contract
    - **Extensibility**: Easy addition of new agent types
    - **Maintainability**: Centralized common functionality
    - **Testability**: Standardized testing patterns
    - **Configuration**: Uniform configuration management

Abstract Interface Contract:
    All concrete agent implementations must adhere to this interface:
    
    - **run(input_data: dict) -> str**: Primary execution method
      - Processes input data according to agent specialization
      - Returns string response with consistent formatting
      - Handles errors gracefully with informative messages
    
    - **register_tools(tools: dict)**: Optional tool registration
      - Enables external tool integration for enhanced capabilities
      - Supports dependency injection for testing and flexibility
      - Allows runtime tool configuration and updates

Core Attributes:
    - **name (str)**: Unique identifier for the agent instance
      - Used for logging, debugging, and registry management
      - Enables traceability in multi-agent workflows
      - Supports dynamic agent discovery and configuration
    
    - **qa_model (str)**: Language model identifier and configuration
      - Supports multiple model backends (OpenAI, Gemini, Qwen)
      - Enables model-specific optimization and error handling
      - Allows runtime model switching and A/B testing
    
    - **tool_executor (dict)**: External tool integration interface
      - Optional tools for enhanced agent capabilities
      - Supports function calling and external API integration
      - Enables modular functionality extension

Agent Lifecycle Management:
    1. **Initialization**: Configure agent with model and parameters
    2. **Registration**: Register with MultiAgentRunner system
    3. **Tool Setup**: Optional external tool configuration
    4. **Execution**: Process queries through run() method
    5. **Cleanup**: Resource cleanup and memory management

Concrete Agent Implementations:
    The system includes six specialized agents inheriting from BaseAgent:
    
    - **TextAgent**: Textual content analysis and question answering
    - **ImageAgent**: Visual content processing and analysis
    - **GeneralizeAgent**: Multi-modal response synthesis
    - **PlanningAgent**: Complex query decomposition
    - **MergeAgent**: Response consolidation and integration
    - **VerifierAgent**: Quality assessment and iterative improvement

Error Handling Standards:
    All agents must implement robust error handling:
    - Graceful degradation on model failures
    - Informative error messages with context
    - Resource cleanup on exceptions
    - Logging for debugging and monitoring
    - Fallback behavior for service unavailability

Implementation Guidelines:
    When creating new agent types, follow these best practices:
    
    1. **Single Responsibility**: Each agent should have one clear purpose
    2. **Stateless Design**: Avoid persistent state between calls
    3. **Error Resilience**: Handle all failure modes gracefully
    4. **Resource Management**: Clean up resources properly
    5. **Logging**: Provide detailed operational logs
    6. **Documentation**: Include comprehensive docstrings
    7. **Testing**: Implement unit tests for all functionality

Integration Patterns:
    - **Dependency Injection**: Tools and models injected at runtime
    - **Strategy Pattern**: Different models for same agent type
    - **Observer Pattern**: Logging and monitoring integration
    - **Chain of Responsibility**: Agent coordination in workflows

Performance Considerations:
    - **Memory Efficiency**: Minimize memory footprint
    - **GPU Optimization**: Efficient GPU memory usage
    - **Caching**: Cache expensive operations when appropriate
    - **Batch Processing**: Support batch operations for efficiency
    - **Lazy Loading**: Load resources only when needed

Example Implementation:
    ```python
    class CustomAgent(BaseAgent):
        def __init__(self, name: str, qa_model: str = "openai"):
            super().__init__(name, qa_model)
            self.specialized_config = self._load_config()
        
        def run(self, input_data: dict) -> str:
            try:
                # Validate input
                if not self._validate_input(input_data):
                    return "Invalid input provided"
                
                # Process with model
                result = self._process_with_model(input_data)
                
                # Format response
                return self._format_response(result)
                
            except Exception as e:
                logger.error(f"{self.name} failed: {e}")
                return f"Error in {self.name}: {str(e)}"
        
        def _validate_input(self, data: dict) -> bool:
            # Custom validation logic
            pass
        
        def _process_with_model(self, data: dict) -> str:
            # Model-specific processing
            pass
        
        def _format_response(self, result: str) -> str:
            # Response formatting
            pass
    ```

Author: PDZ (Nguyen Quang Phu), Bang (Tieu Tri Bang)
Version: 2.0 (Multi-Agent Architecture)
License: MIT License
"""
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, qa_model: str = "qwen"):
        self.qa_model = qa_model
        self.name = name
        self.tool_executor = None  # default

    @abstractmethod
    def run(self, input_data: dict) -> str:
        """Run the agent with input data and return output string."""
        pass
    
    def register_tools(self, tools: dict):
        self.tool_executor = tools
