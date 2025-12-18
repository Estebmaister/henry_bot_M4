"""
Conversation memory management for M4 delivery chatbot.

AI Assistant Notes:
- Maintains conversation history with sliding window approach
- Provides session management for user interactions
- Handles conversation persistence to database
- Manages context size for optimal LLM performance
- Supports multiple concurrent sessions
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import uuid

from ..database import ConversationRepository, DatabaseConnection
from ..database.models import ConversationModel, MessageType

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and session context.
    Provides sliding window memory with database persistence.
    """

    def __init__(
        self,
        memory_size: int = 20,
        session_timeout_minutes: int = 60,
        db_connection: DatabaseConnection = None
    ):
        """
        Initialize conversation memory.

        Args:
            memory_size: Maximum number of messages to keep in memory
            session_timeout_minutes: Session timeout in minutes
            db_connection: Database connection for persistence
        """
        self.memory_size = memory_size
        self.session_timeout_minutes = session_timeout_minutes
        self.db = db_connection or DatabaseConnection()
        self.conversation_repo = ConversationRepository(self.db)

        # In-memory session storage for active sessions
        self.active_sessions = {}

    def get_or_create_session(self, user_id: str = None) -> str:
        """
        Get existing session or create new one.

        Args:
            user_id: Optional user identifier

        Returns:
            Session ID
        """
        # Check for existing active session
        if user_id and user_id in self.active_sessions:
            session_data = self.active_sessions[user_id]
            if self._is_session_valid(session_data):
                return session_data['session_id']
            else:
                # Remove expired session
                del self.active_sessions[user_id]

        # Create new session
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0,
            'memory': []
        }

        if user_id:
            self.active_sessions[user_id] = session_data

        logger.info(f"Created new conversation session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        content: str,
        message_type: MessageType = MessageType.USER,
        user_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a message to conversation memory.

        Args:
            session_id: Session identifier
            content: Message content
            message_type: Type of message (user/assistant/system)
            user_id: Optional user identifier
            metadata: Additional message metadata
        """
        try:
            # Validate content
            if not content or not content.strip():
                logger.warning(f"Skipping empty message for session {session_id}")
                return

            # Get message sequence number
            sequence_number = self.conversation_repo.get_last_sequence_number(session_id) + 1

            # Create conversation model
            conversation = ConversationModel(
                session_id=session_id,
                user_id=user_id,
                message_sequence=sequence_number,
                message_type=message_type,
                content=content.strip(),
                metadata=metadata or {},
                timestamp=datetime.now()
            )

            # Save to database
            self.conversation_repo.save_message(conversation)

            # Update in-memory session if active
            self._update_in_memory_session(session_id, conversation)

            # Handle both enum and string message types
            message_type_value = message_type.value if hasattr(message_type, 'value') else str(message_type)
            logger.debug(f"Added message to session {session_id}: {message_type_value}")

        except Exception as e:
            logger.error(f"Error adding message to conversation: {e}")

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of conversation messages in OpenAI format
        """
        try:
            limit = limit or self.memory_size
            conversations = self.conversation_repo.get_conversation_history(session_id, limit)

            # Convert to OpenAI chat format
            history = []
            for conv in conversations:
                role = "assistant" if conv.message_type == MessageType.ASSISTANT else "user"
                history.append({
                    "role": role,
                    "content": conv.content,
                    "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                    "metadata": conv.metadata or {}
                })

            return history

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def get_recent_context(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation context within token limit.

        Args:
            session_id: Session identifier
            max_tokens: Maximum tokens for context

        Returns:
            List of messages within token limit
        """
        try:
            # Get full conversation history
            full_history = self.get_conversation_history(session_id)

            # Filter messages to stay within token limit
            context = []
            current_tokens = 0

            # Simple token estimation (4 chars per token on average)
            for message in reversed(full_history):  # Start from most recent
                estimated_tokens = len(message['content']) // 4

                if current_tokens + estimated_tokens > max_tokens and context:
                    break

                context.insert(0, message)  # Add to beginning
                current_tokens += estimated_tokens

            return context

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return []

    def _update_in_memory_session(self, session_id: str, conversation: ConversationModel) -> None:
        """Update in-memory session data."""
        # Find session in active sessions
        for user_id, session_data in self.active_sessions.items():
            if session_data['session_id'] == session_id:
                session_data['last_activity'] = datetime.now()
                session_data['message_count'] += 1

                # Add to in-memory buffer (limited size)
                session_data['memory'].append({
                    'type': conversation.message_type.value if hasattr(conversation.message_type, 'value') else str(conversation.message_type),
                    'content': conversation.content,
                    'timestamp': conversation.timestamp,
                    'metadata': conversation.metadata
                })

                # Keep only recent messages in memory
                if len(session_data['memory']) > self.memory_size:
                    session_data['memory'] = session_data['memory'][-self.memory_size:]

                break

    def _is_session_valid(self, session_data: Dict[str, Any]) -> bool:
        """Check if session is still valid (not expired)."""
        last_activity = session_data.get('last_activity')
        if not last_activity:
            return False

        expiry_time = last_activity + timedelta(minutes=self.session_timeout_minutes)
        return datetime.now() < expiry_time

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions and old conversation data.

        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0

        # Clean up in-memory sessions
        expired_sessions = []
        for user_id, session_data in self.active_sessions.items():
            if not self._is_session_valid(session_data):
                expired_sessions.append(user_id)

        for user_id in expired_sessions:
            del self.active_sessions[user_id]
            cleaned_count += 1

        # Clean up old conversation records from database
        deleted_conversations = self.conversation_repo.delete_old_conversations(days=30)

        # Only log if there were actual cleanups
        if cleaned_count > 0 or deleted_conversations > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions and {deleted_conversations} old conversation records")
        return cleaned_count

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        try:
            history = self.get_conversation_history(session_id, limit=1000)  # Get full history

            user_messages = [msg for msg in history if msg.get('role') == 'user']
            assistant_messages = [msg for msg in history if msg.get('role') == 'assistant']

            total_chars = sum(len(msg.get('content', '')) for msg in history)
            avg_message_length = total_chars / len(history) if history else 0

            return {
                'session_id': session_id,
                'total_messages': len(history),
                'user_messages': len(user_messages),
                'assistant_messages': len(assistant_messages),
                'average_message_length': round(avg_message_length, 1),
                'total_characters': total_chars,
                'estimated_tokens': total_chars // 4  # Rough estimate
            }

        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {'session_id': session_id, 'error': str(e)}

    def search_conversations(
        self,
        user_id: str,
        query_text: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history for specific text.

        Args:
            user_id: User identifier
            query_text: Text to search for
            limit: Maximum results to return

        Returns:
            List of matching conversation messages
        """
        try:
            # Get all user sessions
            user_sessions = []
            for uid, session_data in self.active_sessions.items():
                if uid == user_id:
                    user_sessions.append(session_data['session_id'])

            # If no active sessions, search database
            if not user_sessions:
                # This would require adding search capability to conversation repository
                # For now, return empty results
                return []

            # Search through active session memories
            results = []
            for session_id in user_sessions:
                history = self.get_conversation_history(session_id, limit=100)
                for msg in history:
                    if query_text.lower() in msg.get('content', '').lower():
                        results.append({
                            'session_id': session_id,
                            'message': msg,
                            'snippet': self._extract_snippet(msg.get('content', ''), query_text)
                        })

                        if len(results) >= limit:
                            break

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    def _extract_snippet(self, content: str, query: str, context_chars: int = 100) -> str:
        """Extract snippet around search query."""
        query_lower = query.lower()
        content_lower = content.lower()
        match_pos = content_lower.find(query_lower)

        if match_pos == -1:
            return content[:context_chars] + "..." if len(content) > context_chars else content

        start = max(0, match_pos - context_chars)
        end = min(len(content), match_pos + len(query) + context_chars)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def export_conversation(
        self,
        session_id: str,
        format_type: str = 'json'
    ) -> Dict[str, Any]:
        """
        Export conversation in specified format.

        Args:
            session_id: Session identifier
            format_type: Export format ('json', 'txt', 'csv')

        Returns:
            Dictionary with exported conversation data
        """
        try:
            history = self.get_conversation_history(session_id)

            if format_type == 'json':
                return {
                    'session_id': session_id,
                    'exported_at': datetime.now().isoformat(),
                    'messages': history,
                    'total_messages': len(history)
                }

            elif format_type == 'txt':
                transcript = []
                transcript.append(f"Conversation Transcript - Session: {session_id}")
                transcript.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                transcript.append("=" * 50)

                for msg in history:
                    role = msg.get('role', 'unknown').title()
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')
                    transcript.append(f"\n[{role}] {timestamp}")
                    transcript.append(content)

                return {
                    'session_id': session_id,
                    'format': 'txt',
                    'content': '\n'.join(transcript)
                }

            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error(f"Error exporting conversation: {e}")
            return {'error': str(e), 'session_id': session_id}