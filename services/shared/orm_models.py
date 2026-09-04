"""SQLAlchemy ORM models for RAG application.

This module defines the database models using SQLAlchemy ORM with pgvector support.
All models use UUID primary keys and include proper relationships and constraints.

Validates Requirements: 18.1, 18.2, 18.3, 18.4, 18.5
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, validates
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from .config import get_settings

Base = declarative_base()


class Document(Base):
    """Document model representing a markdown file.
    
    Validates Requirements:
    - 18.1: Atomic transactions for document ingestion
    - 18.4: No duplicate chunks for same document
    - 18.5: Maintain referential integrity
    """
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_path = Column(Text, nullable=False, unique=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    doc_metadata = Column('metadata', JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    ingestion_status = Column(
        String(20),
        nullable=False,
        default='pending'
    )
    
    # Relationships
    chunks = relationship("TextChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "ingestion_status IN ('pending', 'processing', 'completed', 'failed')",
            name="check_ingestion_status"
        ),
        Index("idx_documents_file_path", "file_path"),
        Index("idx_documents_status", "ingestion_status"),
    )
    
    @validates('content')
    def validate_content(self, key, value):
        """Validate that content is not empty."""
        if not value or not value.strip():
            raise ValueError("Document content cannot be empty")
        return value
    
    @validates('ingestion_status')
    def validate_status(self, key, value):
        """Validate ingestion status."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if value not in valid_statuses:
            raise ValueError(f"Invalid ingestion_status: {value}. Must be one of {valid_statuses}")
        return value
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_path='{self.file_path}', status='{self.ingestion_status}')>"


class TextChunk(Base):
    """Text chunk with embedding vector.
    
    Validates Requirements:
    - 18.2: Embedding dimension must match configured model dimension
    - 18.4: No duplicate chunks for same document (via unique constraint)
    - 18.5: Maintain referential integrity (via foreign key)
    """
    
    __tablename__ = "text_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)  # BGE-base produces 768-dimensional vectors
    chunk_metadata = Column('metadata', JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        Index("idx_text_chunks_document_id", "document_id"),
        Index("idx_text_chunks_embedding", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"}),
    )
    
    @validates('chunk_index')
    def validate_chunk_index(self, key, value):
        """Validate that chunk_index is non-negative."""
        if value < 0:
            raise ValueError("chunk_index must be non-negative")
        return value
    
    @validates('text')
    def validate_text(self, key, value):
        """Validate that text is not empty."""
        if not value or not value.strip():
            raise ValueError("Text cannot be empty")
        return value
    
    @validates('embedding')
    def validate_embedding(self, key, value):
        """Validate embedding dimension matches configured model dimension.
        
        Validates Requirement 18.2: Embedding dimension must match configured model dimension
        """
        if value is not None:
            settings = get_settings()
            expected_dim = settings.embedding_dimension
            
            # Handle different input types (list, numpy array, etc.)
            if hasattr(value, '__len__'):
                actual_dim = len(value)
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Embedding dimension must be {expected_dim}, got {actual_dim}"
                    )
        return value
    
    def __repr__(self):
        return f"<TextChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class Conversation(Base):
    """Conversation session model.
    
    Validates Requirements:
    - 18.3: Messages maintain chronological order with monotonically increasing timestamps
    - 18.5: Maintain referential integrity
    """
    
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(Text, nullable=False, default='anonymous')
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    conversation_metadata = Column('metadata', JSONB, nullable=False, default=dict)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.timestamp")
    
    # Constraints
    __table_args__ = (
        Index("idx_conversations_user_id", "user_id"),
        Index("idx_conversations_expires_at", "expires_at"),
    )
    
    def __init__(self, **kwargs):
        """Initialize conversation with default expiration."""
        super().__init__(**kwargs)
        if self.expires_at is None:
            settings = get_settings()
            self.expires_at = datetime.now() + timedelta(days=settings.conversation_retention_days)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', created_at={self.created_at})>"


class Message(Base):
    """Message in a conversation.
    
    Validates Requirements:
    - 18.3: Messages maintain chronological order with monotonically increasing timestamps
    - 18.5: Maintain referential integrity (via foreign key)
    """
    
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSONB, nullable=False, default=list)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    message_metadata = Column('metadata', JSONB, nullable=False, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="check_message_role"
        ),
        Index("idx_messages_conversation_id", "conversation_id"),
        Index("idx_messages_timestamp", "timestamp"),
    )
    
    @validates('role')
    def validate_role(self, key, value):
        """Validate message role."""
        if value not in {'user', 'assistant'}:
            raise ValueError(f"Invalid role: {value}. Must be 'user' or 'assistant'")
        return value
    
    @validates('content')
    def validate_content(self, key, value):
        """Validate that content is not empty."""
        if not value or not value.strip():
            raise ValueError("Message content cannot be empty")
        return value
    
    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role='{self.role}')>"


class IngestionJob(Base):
    """Ingestion job tracking model.
    
    Validates Requirements:
    - 18.1: Atomic transactions for document ingestion
    - 1.5: Retry with exponential backoff when ingestion fails
    - 13.3: Queue jobs for retry with exponential backoff
    """
    
    __tablename__ = "ingestion_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_path = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default='queued')
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    chunks_created = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="check_job_status"
        ),
        CheckConstraint(
            "chunks_created >= 0",
            name="check_chunks_created_non_negative"
        ),
        CheckConstraint(
            "retry_count >= 0",
            name="check_retry_count_non_negative"
        ),
        CheckConstraint(
            "max_retries >= 0",
            name="check_max_retries_non_negative"
        ),
        Index("idx_ingestion_jobs_status", "status"),
        Index("idx_ingestion_jobs_next_retry", "next_retry_at"),
    )
    
    @validates('status')
    def validate_status(self, key, value):
        """Validate job status."""
        valid_statuses = {'queued', 'processing', 'completed', 'failed'}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status: {value}. Must be one of {valid_statuses}")
        return value
    
    @validates('chunks_created')
    def validate_chunks_created(self, key, value):
        """Validate that chunks_created is non-negative."""
        if value < 0:
            raise ValueError("chunks_created must be non-negative")
        return value
    
    @validates('completed_at')
    def validate_completed_at(self, key, value):
        """Validate that completed_at is after started_at."""
        if value is not None and self.started_at is not None:
            if value < self.started_at:
                raise ValueError("completed_at must be after started_at")
        return value
    
    def __repr__(self):
        return f"<IngestionJob(id={self.id}, file_path='{self.file_path}', status='{self.status}')>"
