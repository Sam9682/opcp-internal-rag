"""Conversation Memory Service for managing conversation history and context.

This module provides the ConversationMemoryService class that manages conversation
sessions, stores messages, retrieves history with pagination, and handles conversation
expiration and summarization.

Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import uuid4

from sqlalchemy import and_, or_, desc
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .orm_models import Conversation, Message
from .database import DatabaseManager
from .config import get_settings

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """Service for managing conversation history and context.
    
    Provides functionality to:
    - Create new conversation sessions
    - Store messages with timestamps and metadata
    - Retrieve conversation history with pagination
    - Handle conversation expiration
    - Summarize long conversations for context compression
    
    Validates Requirements:
    - 9.1: Create new conversation sessions
    - 9.2: Store messages with timestamps in chronological order
    - 9.3: Return messages in chronological order with pagination
    - 9.4: Expire and delete conversations after retention period
    - 9.5: Summarize older messages to compress context
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize conversation memory service.
        
        Args:
            db_manager: Database manager for session handling
        """
        self.db_manager = db_manager
        self.settings = get_settings()
        logger.info("ConversationMemoryService initialized")
    
    def create_conversation(self, user_id: str = 'anonymous') -> str:
        """Create a new conversation session.
        
        Creates a new conversation with a unique ID and sets expiration
        based on configured retention period.
        
        Validates Requirement 9.1: Create new conversation session when
        query submitted without conversation ID
        
        Args:
            user_id: User identifier (defaults to 'anonymous')
            
        Returns:
            Conversation ID (UUID string)
            
        Raises:
            SQLAlchemyError: If database operation fails
            
        Example:
            conversation_id = service.create_conversation(user_id="user123")
        """
        try:
            with self.db_manager.session_scope() as session:
                # Create new conversation with expiration
                conversation = Conversation(
                    id=uuid4(),
                    user_id=user_id,
                    conversation_metadata={
                        'created_by': 'conversation_memory_service',
                        'version': '1.0'
                    }
                )
                
                session.add(conversation)
                session.flush()  # Ensure ID is generated
                
                conversation_id = str(conversation.id)
                logger.info(
                    f"Created conversation {conversation_id} for user {user_id}, "
                    f"expires at {conversation.expires_at}"
                )
                
                return conversation_id
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a message to a conversation.
        
        Stores a message with timestamp, role, content, and optional metadata.
        Messages are stored in chronological order with monotonically increasing
        timestamps.
        
        Validates Requirement 9.2: Store messages with timestamps in chronological order
        
        Args:
            conversation_id: UUID of the conversation
            role: Message role ('user' or 'assistant')
            content: Message content text
            metadata: Optional metadata dictionary
            sources: Optional list of source citations (for assistant messages)
            
        Raises:
            ValueError: If conversation not found or role is invalid
            SQLAlchemyError: If database operation fails
            
        Example:
            service.add_message(
                conversation_id="123e4567-e89b-12d3-a456-426614174000",
                role="user",
                content="What is RAG?",
                metadata={"ip_address": "192.168.1.1"}
            )
        """
        if role not in {'user', 'assistant'}:
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")
        
        try:
            with self.db_manager.session_scope() as session:
                # Verify conversation exists
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Check if conversation has expired
                if conversation.expires_at and datetime.now() > conversation.expires_at:
                    logger.warning(
                        f"Attempted to add message to expired conversation {conversation_id}"
                    )
                    raise ValueError(f"Conversation {conversation_id} has expired")
                
                # Create message
                message = Message(
                    id=uuid4(),
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    sources=sources or [],
                    message_metadata=metadata or {}
                )
                
                session.add(message)
                
                # Update conversation's updated_at timestamp
                conversation.updated_at = datetime.now()
                
                logger.debug(
                    f"Added {role} message to conversation {conversation_id} "
                    f"at {message.timestamp}"
                )
                
        except ValueError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            raise
    
    def get_history(
        self,
        conversation_id: str,
        limit: int = 10,
        offset: int = 0,
        include_sources: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history with pagination.
        
        Returns messages in chronological order (oldest first) with support
        for pagination through limit and offset parameters.
        
        Validates Requirement 9.3: Return messages in chronological order
        with pagination support
        
        Args:
            conversation_id: UUID of the conversation
            limit: Maximum number of messages to return (default: 10)
            offset: Number of messages to skip (default: 0)
            include_sources: Whether to include source citations (default: True)
            
        Returns:
            List of message dictionaries with keys:
            - id: Message UUID
            - role: 'user' or 'assistant'
            - content: Message text
            - timestamp: ISO format timestamp
            - sources: List of source citations (if include_sources=True)
            - metadata: Message metadata
            
        Raises:
            ValueError: If conversation not found
            SQLAlchemyError: If database operation fails
            
        Example:
            # Get most recent 10 messages
            history = service.get_history(conversation_id, limit=10)
            
            # Get next 10 messages (pagination)
            history = service.get_history(conversation_id, limit=10, offset=10)
        """
        try:
            with self.db_manager.session_scope() as session:
                # Verify conversation exists
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Query messages in chronological order with pagination
                messages = session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(
                    Message.timestamp.asc()  # Chronological order (oldest first)
                ).limit(limit).offset(offset).all()
                
                # Convert to dictionaries
                history = []
                for msg in messages:
                    msg_dict = {
                        'id': str(msg.id),
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'metadata': msg.message_metadata
                    }
                    
                    if include_sources and msg.sources:
                        msg_dict['sources'] = msg.sources
                    
                    history.append(msg_dict)
                
                logger.debug(
                    f"Retrieved {len(history)} messages from conversation {conversation_id} "
                    f"(limit={limit}, offset={offset})"
                )
                
                return history
                
        except ValueError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve history for conversation {conversation_id}: {e}")
            raise
    
    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the most recent messages from a conversation.
        
        Convenience method that returns the most recent messages in
        reverse chronological order (newest first).
        
        Args:
            conversation_id: UUID of the conversation
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries (newest first)
            
        Example:
            # Get last 5 messages for context
            recent = service.get_recent_messages(conversation_id, limit=5)
        """
        try:
            with self.db_manager.session_scope() as session:
                # Query most recent messages
                messages = session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(
                    Message.timestamp.desc()  # Newest first
                ).limit(limit).all()
                
                # Convert to dictionaries
                history = []
                for msg in messages:
                    history.append({
                        'id': str(msg.id),
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'sources': msg.sources if msg.sources else [],
                        'metadata': msg.message_metadata
                    })
                
                return history
                
        except SQLAlchemyError as e:
            logger.error(
                f"Failed to retrieve recent messages for conversation {conversation_id}: {e}"
            )
            raise
    
    def delete_expired_conversations(self) -> int:
        """Delete conversations that have exceeded retention period.
        
        Removes conversations where expires_at is in the past. This should
        be called periodically (e.g., via cron job or scheduled task).
        
        Validates Requirement 9.4: Expire and delete conversations after
        configured retention period
        
        Returns:
            Number of conversations deleted
            
        Raises:
            SQLAlchemyError: If database operation fails
            
        Example:
            # Run periodically to clean up expired conversations
            deleted_count = service.delete_expired_conversations()
            logger.info(f"Deleted {deleted_count} expired conversations")
        """
        try:
            with self.db_manager.session_scope() as session:
                # Find expired conversations
                now = datetime.now()
                expired = session.query(Conversation).filter(
                    and_(
                        Conversation.expires_at.isnot(None),
                        Conversation.expires_at < now
                    )
                ).all()
                
                count = len(expired)
                
                if count > 0:
                    # Delete expired conversations (cascade will delete messages)
                    for conversation in expired:
                        session.delete(conversation)
                    
                    logger.info(f"Deleted {count} expired conversations")
                else:
                    logger.debug("No expired conversations to delete")
                
                return count
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete expired conversations: {e}")
            raise
    
    def summarize_conversation(
        self,
        conversation_id: str,
        max_messages: int = 20
    ) -> str:
        """Generate a summary of conversation for context compression.
        
        Creates a text summary of older messages in a conversation to reduce
        token count while preserving context. This is useful when conversations
        become too long to fit in the prompt.
        
        Validates Requirement 9.5: Summarize older messages to compress context
        when conversation becomes too long
        
        Args:
            conversation_id: UUID of the conversation
            max_messages: Maximum number of messages to include in summary
            
        Returns:
            Summary text of the conversation
            
        Raises:
            ValueError: If conversation not found
            SQLAlchemyError: If database operation fails
            
        Note:
            This is a simple implementation that concatenates messages.
            In production, you would use an LLM to generate a proper summary.
            
        Example:
            summary = service.summarize_conversation(conversation_id, max_messages=20)
            # Use summary in prompt instead of full history
        """
        try:
            with self.db_manager.session_scope() as session:
                # Verify conversation exists
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Get older messages (excluding most recent ones)
                messages = session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(
                    Message.timestamp.asc()
                ).limit(max_messages).all()
                
                if not messages:
                    return "No conversation history to summarize."
                
                # Simple summarization: concatenate messages with role labels
                # In production, use an LLM to generate a proper summary
                summary_parts = [
                    f"Conversation started at {conversation.created_at.isoformat()}"
                ]
                
                for msg in messages:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    # Truncate long messages
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    summary_parts.append(f"{role_label}: {content}")
                
                summary = "\n".join(summary_parts)
                
                logger.debug(
                    f"Generated summary for conversation {conversation_id} "
                    f"({len(messages)} messages)"
                )
                
                return summary
                
        except ValueError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Failed to summarize conversation {conversation_id}: {e}")
            raise
    
    def get_conversation_metadata(self, conversation_id: str) -> Dict[str, Any]:
        """Get metadata about a conversation.
        
        Returns information about the conversation including creation time,
        last update, expiration, and message count.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Dictionary with conversation metadata
            
        Raises:
            ValueError: If conversation not found
            SQLAlchemyError: If database operation fails
        """
        try:
            with self.db_manager.session_scope() as session:
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Count messages
                message_count = session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).count()
                
                return {
                    'id': str(conversation.id),
                    'user_id': conversation.user_id,
                    'created_at': conversation.created_at.isoformat(),
                    'updated_at': conversation.updated_at.isoformat(),
                    'expires_at': conversation.expires_at.isoformat() if conversation.expires_at else None,
                    'message_count': message_count,
                    'metadata': conversation.conversation_metadata
                }
                
        except ValueError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Failed to get metadata for conversation {conversation_id}: {e}")
            raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get basic conversation information including user_id.
        
        Validates Requirement 15.3: Access control for conversations
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Dictionary with conversation info or None if not found
        """
        try:
            with self.db_manager.session_scope() as session:
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    return None
                
                return {
                    'id': str(conversation.id),
                    'user_id': conversation.user_id,
                    'created_at': conversation.created_at,
                    'updated_at': conversation.updated_at,
                    'expires_at': conversation.expires_at
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists and is not expired.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            True if conversation exists and is not expired, False otherwise
        """
        try:
            with self.db_manager.session_scope() as session:
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    return False
                
                # Check if expired
                if conversation.expires_at and datetime.now() > conversation.expires_at:
                    return False
                
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to check conversation existence: {e}")
            return False
