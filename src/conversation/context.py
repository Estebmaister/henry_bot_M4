"""
Conversation context management for M4 delivery chatbot.

AI Assistant Notes:
- High-level context management interface
- Integrates memory, extraction, and routing
- Provides unified conversation state
- Manages session lifecycle and transitions
"""

from src.utils import observe
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .memory import ConversationMemory
from .extractor import OrderInformationExtractor
from ..function_calling.router import FunctionRouter
from ..database.models import MessageType
from ..utils import langfuse_client

logger = logging.getLogger(__name__)


class ConversationContext:
    """
    High-level conversation context manager.
    Integrates memory, extraction, and routing capabilities.
    """

    def __init__(self, memory_size: int = 20, session_timeout_minutes: int = 60):
        """
        Initialize conversation context manager.

        Args:
            memory_size: Maximum messages to keep in memory
            session_timeout_minutes: Session timeout duration
        """
        self.memory = ConversationMemory(
            memory_size=memory_size,
            session_timeout_minutes=session_timeout_minutes
        )
        self.extractor = OrderInformationExtractor()
        self.router = FunctionRouter()

    def get_or_create_session(self, user_id: str = None) -> str:
        """
        Get existing session or create new one.

        Args:
            user_id: Optional user identifier

        Returns:
            Session ID
        """
        return self.memory.get_or_create_session(user_id)

    @observe(name="conversation_context_processing", as_type="chain")
    def process_user_message(
        self,
        session_id: str,
        message: str,
        user_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and update conversation context.

        Args:
            session_id: Session identifier
            message: User message content
            user_id: Optional user identifier
            metadata: Additional message metadata

        Returns:
            Updated conversation context
        """

        try:
            # Add user message to memory
            self.memory.add_message(
                session_id=session_id,
                content=message,
                message_type=MessageType.USER,
                user_id=user_id,
                metadata=metadata
            )

            # Get conversation history
            history = self.memory.get_conversation_history(session_id)

            # Extract order information
            extracted_info = self.extractor.extract_from_conversation(
                history, message)

            # Analyze function calling opportunities
            function_suggestions = self.router.get_tool_suggestions(
                message, history)

            # Get session statistics
            session_stats = self.memory.get_session_stats(session_id)

            # Get recent context
            recent_context = self.memory.get_recent_context(session_id)

            return {
                'session_id': session_id,
                'user_message': message,
                'conversation_history': history,
                'recent_context': recent_context,
                'extracted_order_info': extracted_info,
                'function_suggestions': function_suggestions,
                'session_stats': session_stats,
                'processed_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return {
                'session_id': session_id,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }

    def add_assistant_response(
        self,
        session_id: str,
        response: str,
        user_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add assistant response to conversation memory.

        Args:
            session_id: Session identifier
            response: Assistant response content
            user_id: User identifier
            metadata: Response metadata
        """
        try:
            self.memory.add_message(
                session_id=session_id,
                content=response,
                message_type=MessageType.ASSISTANT,
                user_id=user_id,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error adding assistant response: {e}")

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive conversation summary.

        Args:
            session_id: Session identifier

        Returns:
            Conversation summary
        """
        try:
            history = self.memory.get_conversation_history(session_id)
            stats = self.memory.get_session_stats(session_id)

            # Extract order information from entire conversation
            extracted_info = self.extractor.extract_from_conversation(history)

            return {
                'session_id': session_id,
                'stats': stats,
                'extracted_order_info': extracted_info,
                'message_count': len(history),
                'last_message_time': history[-1].get('timestamp') if history else None,
                'summary_generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return {
                'session_id': session_id,
                'error': str(e)
            }

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        return self.memory.cleanup_expired_sessions()

    def export_conversation(
        self,
        session_id: str,
        format_type: str = 'json'
    ) -> Dict[str, Any]:
        """
        Export conversation in specified format.

        Args:
            session_id: Session identifier
            format_type: Export format

        Returns:
            Exported conversation data
        """
        return self.memory.export_conversation(session_id, format_type)

    def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results

        Returns:
            Search results
        """
        return self.memory.search_conversations(user_id, query, limit)
