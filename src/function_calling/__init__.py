"""
Function Calling package for M4 delivery chatbot.

This package provides autonomous tool selection and execution capabilities
using OpenAI's Function Calling API integrated with the LangChain framework.

AI Assistant Notes:
- Hybrid approach using OpenAI native function calling with LangChain wrappers
- Autonomous tool selection based on conversation context
- Error handling and fallback mechanisms
- Integration with database repositories
- Tool result processing and response generation
"""

from .tools import FunctionTools
from .executor import FunctionExecutor
from .router import FunctionRouter

__all__ = [
    "FunctionTools",
    "FunctionExecutor",
    "FunctionRouter",
]