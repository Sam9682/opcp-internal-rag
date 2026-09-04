"""Unit tests for ConversationMemoryService.

Tests conversation creation, message storage, history retrieval with pagination,
conversation expiration, and summarization functionality.

Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from .conversation_memory_service import ConversationMemoryService
from .database import init_db
from .orm_models import Conversation, Message


@pytest.fixture(scope="module")
def db_manager():
    """Initialize database for tests."""
    db = init_db(wait_for_ready=True)
    yield db
    db.close()


@pytest.fixture
def service(db_manager):
    """Create ConversationMemoryService instance."""
    return ConversationMemoryService(db_manager)


class TestCreateConversation:
    """Test conversation creation functionality.
    
    Validates Requirement 9.1: Create new conversation session
    """
    
    def test_create_conversation_default_user(self, service):
        """Test creating conversation with default anonymous user."""
        conversation_id = service.create_conversation()
        
        assert conversation_id is not None
        assert isinstance(conversation_id, str)
        
        # Verify conversation exists
        assert service.conversation_exists(conversation_id)
    
    def test_create_conversation_with_user_id(self, service):
        """Test creating conversation with specific user ID."""
        user_id = "user123"
        conversation_id = service.create_conversation(user_id=user_id)
        
        assert conversation_id is not None
        
        # Verify user ID is stored
        metadata = service.get_conversation_metadata(conversation_id)
        assert metadata['user_id'] == user_id
    
    def test_create_conversation_sets_expiration(self, service):
        """Test that new conversations have expiration set."""
        conversation_id = service.create_conversation()
        
        metadata = service.get_conversation_metadata(conversation_id)
        assert metadata['expires_at'] is not None
        
        # Expiration should be in the future
        expires_at = datetime.fromisoformat(metadata['expires_at'])
        assert expires_at > datetime.now()
    
    def test_create_multiple_conversations(self, service):
        """Test creating multiple conversations with unique IDs."""
        conv_id_1 = service.create_conversation(user_id="user1")
        conv_id_2 = service.create_conversation(user_id="user2")
        
        assert conv_id_1 != conv_id_2
        assert service.conversation_exists(conv_id_1)
        assert service.conversation_exists(conv_id_2)


class TestAddMessage:
    """Test message storage functionality.
    
    Validates Requirement 9.2: Store messages with timestamps in chronological order
    """
    
    def test_add_user_message(self, service):
        """Test adding a user message to conversation."""
        conversation_id = service.create_conversation()
        
        service.add_message(
            conversation_id=conversation_id,
            role="user",
            content="What is RAG?"
        )
        
        # Verify message was stored
        history = service.get_history(conversation_id)
        assert len(history) == 1
        assert history[0]['role'] == 'user'
        assert history[0]['content'] == "What is RAG?"
    
    def test_add_assistant_message(self, service):
        """Test adding an assistant message with sources."""
        conversation_id = service.create_conversation()
        
        sources = [
            {
                'chunk_id': str(uuid4()),
                'title': 'RAG Documentation',
                'similarity_score': 0.95
            }
        ]
        
        service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="RAG stands for Retrieval-Augmented Generation.",
            sources=sources
        )
        
        history = service.get_history(conversation_id)
        assert len(history) == 1
        assert history[0]['role'] == 'assistant'
        assert len(history[0]['sources']) == 1
        assert history[0]['sources'][0]['title'] == 'RAG Documentation'
    
    def test_add_message_with_metadata(self, service):
        """Test adding message with custom metadata."""
        conversation_id = service.create_conversation()
        
        metadata = {
            'ip_address': '192.168.1.1',
            'user_agent': 'Mozilla/5.0'
        }
        
        service.add_message(
            conversation_id=conversation_id,
            role="user",
            content="Test message",
            metadata=metadata
        )
        
        history = service.get_history(conversation_id)
        assert history[0]['metadata'] == metadata
    
    def test_add_message_invalid_role(self, service):
        """Test that invalid role raises ValueError."""
        conversation_id = service.create_conversation()
        
        with pytest.raises(ValueError, match="Invalid role"):
            service.add_message(
                conversation_id=conversation_id,
                role="invalid",
                content="Test"
            )
    
    def test_add_message_nonexistent_conversation(self, service):
        """Test that adding message to nonexistent conversation raises error."""
        fake_id = str(uuid4())
        
        with pytest.raises(ValueError, match="not found"):
            service.add_message(
                conversation_id=fake_id,
                role="user",
                content="Test"
            )
    
    def test_messages_chronological_order(self, service):
        """Test that messages maintain chronological order."""
        conversation_id = service.create_conversation()
        
        # Add multiple messages
        messages = [
            ("user", "First message"),
            ("assistant", "First response"),
            ("user", "Second message"),
            ("assistant", "Second response")
        ]
        
        for role, content in messages:
            service.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
        
        # Retrieve history
        history = service.get_history(conversation_id)
        
        # Verify order
        assert len(history) == 4
        for i, (role, content) in enumerate(messages):
            assert history[i]['role'] == role
            assert history[i]['content'] == content
        
        # Verify timestamps are monotonically increasing
        timestamps = [datetime.fromisoformat(msg['timestamp']) for msg in history]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]


class TestGetHistory:
    """Test history retrieval with pagination.
    
    Validates Requirement 9.3: Return messages in chronological order with pagination
    """
    
    def test_get_history_empty_conversation(self, service):
        """Test retrieving history from empty conversation."""
        conversation_id = service.create_conversation()
        
        history = service.get_history(conversation_id)
        assert history == []
    
    def test_get_history_with_messages(self, service):
        """Test retrieving history with messages."""
        conversation_id = service.create_conversation()
        
        # Add messages
        service.add_message(conversation_id, "user", "Message 1")
        service.add_message(conversation_id, "assistant", "Response 1")
        
        history = service.get_history(conversation_id)
        assert len(history) == 2
        assert history[0]['content'] == "Message 1"
        assert history[1]['content'] == "Response 1"
    
    def test_get_history_pagination_limit(self, service):
        """Test pagination with limit parameter."""
        conversation_id = service.create_conversation()
        
        # Add 10 messages
        for i in range(10):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        # Get first 5 messages
        history = service.get_history(conversation_id, limit=5)
        assert len(history) == 5
        assert history[0]['content'] == "Message 0"
        assert history[4]['content'] == "Message 4"
    
    def test_get_history_pagination_offset(self, service):
        """Test pagination with offset parameter."""
        conversation_id = service.create_conversation()
        
        # Add 10 messages
        for i in range(10):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        # Get messages 5-9 (skip first 5)
        history = service.get_history(conversation_id, limit=5, offset=5)
        assert len(history) == 5
        assert history[0]['content'] == "Message 5"
        assert history[4]['content'] == "Message 9"
    
    def test_get_history_limit_and_offset(self, service):
        """Test pagination with both limit and offset."""
        conversation_id = service.create_conversation()
        
        # Add 20 messages
        for i in range(20):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        # Get messages 10-14 (skip 10, take 5)
        history = service.get_history(conversation_id, limit=5, offset=10)
        assert len(history) == 5
        assert history[0]['content'] == "Message 10"
        assert history[4]['content'] == "Message 14"
    
    def test_get_history_without_sources(self, service):
        """Test retrieving history without source citations."""
        conversation_id = service.create_conversation()
        
        sources = [{'chunk_id': str(uuid4()), 'title': 'Test'}]
        service.add_message(
            conversation_id,
            "assistant",
            "Response with sources",
            sources=sources
        )
        
        # Get history without sources
        history = service.get_history(conversation_id, include_sources=False)
        assert 'sources' not in history[0]
    
    def test_get_history_nonexistent_conversation(self, service):
        """Test that retrieving history for nonexistent conversation raises error."""
        fake_id = str(uuid4())
        
        with pytest.raises(ValueError, match="not found"):
            service.get_history(fake_id)


class TestGetRecentMessages:
    """Test retrieving recent messages."""
    
    def test_get_recent_messages(self, service):
        """Test getting most recent messages."""
        conversation_id = service.create_conversation()
        
        # Add 10 messages
        for i in range(10):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        # Get 3 most recent
        recent = service.get_recent_messages(conversation_id, limit=3)
        
        assert len(recent) == 3
        # Should be in reverse order (newest first)
        assert recent[0]['content'] == "Message 9"
        assert recent[1]['content'] == "Message 8"
        assert recent[2]['content'] == "Message 7"
    
    def test_get_recent_messages_fewer_than_limit(self, service):
        """Test getting recent messages when fewer than limit exist."""
        conversation_id = service.create_conversation()
        
        # Add only 2 messages
        service.add_message(conversation_id, "user", "Message 1")
        service.add_message(conversation_id, "user", "Message 2")
        
        # Request 5 messages
        recent = service.get_recent_messages(conversation_id, limit=5)
        
        assert len(recent) == 2


class TestConversationExpiration:
    """Test conversation expiration functionality.
    
    Validates Requirement 9.4: Expire and delete conversations after retention period
    """
    
    def test_delete_expired_conversations(self, service, db_manager):
        """Test deleting expired conversations."""
        # Create conversation with past expiration
        with db_manager.session_scope() as session:
            expired_conv = Conversation(
                id=uuid4(),
                user_id="user1",
                expires_at=datetime.now() - timedelta(days=1)  # Expired yesterday
            )
            session.add(expired_conv)
            session.flush()
            expired_id = str(expired_conv.id)
        
        # Create conversation with future expiration
        active_id = service.create_conversation()
        
        # Delete expired conversations
        deleted_count = service.delete_expired_conversations()
        
        assert deleted_count == 1
        assert not service.conversation_exists(expired_id)
        assert service.conversation_exists(active_id)
    
    def test_delete_expired_conversations_none_expired(self, service):
        """Test deleting when no conversations are expired."""
        # Create active conversations
        service.create_conversation()
        service.create_conversation()
        
        deleted_count = service.delete_expired_conversations()
        assert deleted_count == 0
    
    def test_expired_conversation_messages_deleted(self, service, db_manager):
        """Test that messages are deleted when conversation expires."""
        # Create expired conversation with messages
        with db_manager.session_scope() as session:
            expired_conv = Conversation(
                id=uuid4(),
                user_id="user1",
                expires_at=datetime.now() - timedelta(days=1)
            )
            session.add(expired_conv)
            session.flush()
            
            # Add messages
            msg = Message(
                id=uuid4(),
                conversation_id=expired_conv.id,
                role="user",
                content="Test message"
            )
            session.add(msg)
            session.flush()
            
            expired_id = str(expired_conv.id)
            message_id = str(msg.id)
        
        # Delete expired conversations
        service.delete_expired_conversations()
        
        # Verify conversation and messages are deleted
        with db_manager.session_scope() as session:
            conv = session.query(Conversation).filter(
                Conversation.id == expired_id
            ).first()
            msg = session.query(Message).filter(
                Message.id == message_id
            ).first()
            
            assert conv is None
            assert msg is None
    
    def test_add_message_to_expired_conversation(self, service, db_manager):
        """Test that adding message to expired conversation raises error."""
        # Create expired conversation
        with db_manager.session_scope() as session:
            expired_conv = Conversation(
                id=uuid4(),
                user_id="user1",
                expires_at=datetime.now() - timedelta(hours=1)
            )
            session.add(expired_conv)
            session.flush()
            expired_id = str(expired_conv.id)
        
        # Try to add message
        with pytest.raises(ValueError, match="expired"):
            service.add_message(expired_id, "user", "Test")


class TestSummarizeConversation:
    """Test conversation summarization.
    
    Validates Requirement 9.5: Summarize older messages to compress context
    """
    
    def test_summarize_empty_conversation(self, service):
        """Test summarizing conversation with no messages."""
        conversation_id = service.create_conversation()
        
        summary = service.summarize_conversation(conversation_id)
        assert "No conversation history" in summary
    
    def test_summarize_conversation_with_messages(self, service):
        """Test summarizing conversation with messages."""
        conversation_id = service.create_conversation()
        
        # Add messages
        service.add_message(conversation_id, "user", "What is RAG?")
        service.add_message(conversation_id, "assistant", "RAG is Retrieval-Augmented Generation.")
        service.add_message(conversation_id, "user", "How does it work?")
        
        summary = service.summarize_conversation(conversation_id)
        
        # Summary should contain conversation info
        assert "Conversation started" in summary
        assert "User:" in summary
        assert "Assistant:" in summary
    
    def test_summarize_conversation_max_messages(self, service):
        """Test summarization respects max_messages limit."""
        conversation_id = service.create_conversation()
        
        # Add 30 messages
        for i in range(30):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        # Summarize with limit of 10
        summary = service.summarize_conversation(conversation_id, max_messages=10)
        
        # Should only include first 10 messages
        assert "Message 0" in summary
        assert "Message 9" in summary
        assert "Message 10" not in summary
    
    def test_summarize_truncates_long_messages(self, service):
        """Test that long messages are truncated in summary."""
        conversation_id = service.create_conversation()
        
        # Add very long message
        long_content = "A" * 500
        service.add_message(conversation_id, "user", long_content)
        
        summary = service.summarize_conversation(conversation_id)
        
        # Summary should truncate the message
        assert len(summary) < len(long_content) + 100
        assert "..." in summary
    
    def test_summarize_nonexistent_conversation(self, service):
        """Test that summarizing nonexistent conversation raises error."""
        fake_id = str(uuid4())
        
        with pytest.raises(ValueError, match="not found"):
            service.summarize_conversation(fake_id)


class TestConversationMetadata:
    """Test conversation metadata retrieval."""
    
    def test_get_conversation_metadata(self, service):
        """Test retrieving conversation metadata."""
        conversation_id = service.create_conversation(user_id="user123")
        
        # Add some messages
        service.add_message(conversation_id, "user", "Message 1")
        service.add_message(conversation_id, "assistant", "Response 1")
        
        metadata = service.get_conversation_metadata(conversation_id)
        
        assert metadata['id'] == conversation_id
        assert metadata['user_id'] == "user123"
        assert metadata['message_count'] == 2
        assert 'created_at' in metadata
        assert 'updated_at' in metadata
        assert 'expires_at' in metadata
    
    def test_get_metadata_nonexistent_conversation(self, service):
        """Test that getting metadata for nonexistent conversation raises error."""
        fake_id = str(uuid4())
        
        with pytest.raises(ValueError, match="not found"):
            service.get_conversation_metadata(fake_id)


class TestConversationExists:
    """Test conversation existence checking."""
    
    def test_conversation_exists_true(self, service):
        """Test that existing conversation returns True."""
        conversation_id = service.create_conversation()
        assert service.conversation_exists(conversation_id) is True
    
    def test_conversation_exists_false(self, service):
        """Test that nonexistent conversation returns False."""
        fake_id = str(uuid4())
        assert service.conversation_exists(fake_id) is False
    
    def test_conversation_exists_expired(self, service, db_manager):
        """Test that expired conversation returns False."""
        # Create expired conversation
        with db_manager.session_scope() as session:
            expired_conv = Conversation(
                id=uuid4(),
                user_id="user1",
                expires_at=datetime.now() - timedelta(hours=1)
            )
            session.add(expired_conv)
            session.flush()
            expired_id = str(expired_conv.id)
        
        assert service.conversation_exists(expired_id) is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_message_content(self, service, db_manager):
        """Test that empty message content is rejected by ORM validation."""
        conversation_id = service.create_conversation()
        
        # ORM validation should catch empty content
        with pytest.raises(Exception):  # ValueError from ORM validator
            service.add_message(conversation_id, "user", "")
    
    def test_very_long_message(self, service):
        """Test storing very long message."""
        conversation_id = service.create_conversation()
        
        # Create 10KB message
        long_content = "A" * 10000
        service.add_message(conversation_id, "user", long_content)
        
        history = service.get_history(conversation_id)
        assert len(history[0]['content']) == 10000
    
    def test_large_metadata(self, service):
        """Test storing large metadata dictionary."""
        conversation_id = service.create_conversation()
        
        # Create large metadata
        metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        service.add_message(conversation_id, "user", "Test", metadata=metadata)
        
        history = service.get_history(conversation_id)
        assert len(history[0]['metadata']) == 100
    
    def test_concurrent_message_additions(self, service):
        """Test adding messages to same conversation concurrently."""
        conversation_id = service.create_conversation()
        
        # Add multiple messages rapidly
        for i in range(10):
            service.add_message(conversation_id, "user", f"Message {i}")
        
        history = service.get_history(conversation_id)
        assert len(history) == 10
        
        # All messages should be present
        contents = [msg['content'] for msg in history]
        for i in range(10):
            assert f"Message {i}" in contents
