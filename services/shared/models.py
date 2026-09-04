"""Data models for RAG application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
import numpy as np


@dataclass
class Document:
    """Document model representing a markdown file."""
    
    id: str
    file_path: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    ingestion_status: str  # 'pending', 'processing', 'completed', 'failed'
    
    def __post_init__(self):
        """Validate document fields."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if self.ingestion_status not in valid_statuses:
            raise ValueError(f"Invalid ingestion_status: {self.ingestion_status}")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class TextChunk:
    """Text chunk with embedding vector."""
    
    id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        """Validate text chunk fields."""
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if not self.text:
            raise ValueError("Text cannot be empty")
        if self.embedding.shape[0] != 1024:
            raise ValueError(f"Embedding dimension must be 1024, got {self.embedding.shape[0]}")


@dataclass
class Conversation:
    """Conversation session model."""
    
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: List['Message'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None


@dataclass
class Message:
    """Message in a conversation."""
    
    id: str
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[List['Source']]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message fields."""
        if self.role not in {'user', 'assistant'}:
            raise ValueError(f"Invalid role: {self.role}")
        if not self.content:
            raise ValueError("Message content cannot be empty")


@dataclass
class Source:
    """Source citation for a response."""
    
    chunk_id: str
    document_id: str
    title: str
    excerpt: str
    file_path: str
    similarity_score: float
    
    def __post_init__(self):
        """Validate source fields."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}")
        # Truncate excerpt to reasonable length
        if len(self.excerpt) > 200:
            self.excerpt = self.excerpt[:197] + "..."


@dataclass
class IngestionJob:
    """Ingestion job tracking model."""
    
    id: str
    file_path: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    chunks_created: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate ingestion job fields."""
        valid_statuses = {'queued', 'processing', 'completed', 'failed'}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}")
        if self.chunks_created < 0:
            raise ValueError("chunks_created must be non-negative")
        if self.completed_at and self.started_at:
            if self.completed_at < self.started_at:
                raise ValueError("completed_at must be after started_at")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
