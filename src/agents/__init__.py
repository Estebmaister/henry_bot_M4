"""
Agent modules for specialized department handling.
"""

from .base import BaseAgent, AgentResponse
from .rag_agent import RAGAgent

__all__ = ["BaseAgent", "AgentResponse", "RAGAgent"]