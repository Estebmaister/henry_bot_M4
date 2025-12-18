"""
Conversation management package for M4 delivery chatbot.

This package provides conversation memory, context management, and order information
extraction from multi-turn conversations.

AI Assistant Notes:
- Maintains conversation history with configurable memory size
- Extracts order details from conversation context
- Provides session management for user interactions
- Handles context persistence and retrieval
"""

from .memory import ConversationMemory
from .extractor import OrderInformationExtractor
from .context import ConversationContext

__all__ = [
    "ConversationMemory",
    "OrderInformationExtractor",
    "ConversationContext",
]