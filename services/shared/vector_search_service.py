"""Vector Search Service for similarity search using pgvector.

This module provides the VectorSearchService class that performs vector similarity
search in PostgreSQL using the pgvector extension. It supports storing embeddings
with text chunks and searching for similar vectors using cosine similarity.

Validates Requirements:
- 4.1: Store text chunks with embeddings in PostgreSQL
- 4.2: Maintain HNSW index for fast similarity search
- 4.3: Replace old chunks when re-ingesting documents
- 4.5: Associate metadata with chunks
- 5.1: Return top-k most similar chunks sorted by similarity
- 5.2: Filter by similarity threshold
- 5.3: Use cosine similarity for distance calculation
- 5.5: Include similarity scores with results
"""

import logging
from typing import List, Dict, Optional, Any
from uuid import UUID
import time

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from .orm_models import TextChunk, Document
from .database import DatabaseManager, get_db_manager
from .logging_config import get_logger
from .metrics import record_vector_search, track_time, vector_search_duration_seconds
from .sentry_config import capture_exception

logger = get_logger(__name__)


class VectorSearchService:
    """Service for vector similarity search using pgvector.
    
    This service provides methods to store text chunks with embeddings
    and perform similarity searches using cosine distance in PostgreSQL
    with pgvector extension.
    
    Validates Requirements:
    - 4.1, 4.2, 4.3, 4.5: Vector storage and indexing
    - 5.1, 5.2, 5.3, 5.5: Vector similarity search
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize Vector Search Service.
        
        Args:
            db_manager: Database manager instance. If None, uses global instance.
        """
        self.db_manager = db_manager or get_db_manager()
        logger.info("VectorSearchService initialized")
    
    def store_embedding(
        self,
        text: str,
        vector: np.ndarray,
        document_id: UUID,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store text chunk with embedding and metadata.
        
        Stores a text chunk with its embedding vector in PostgreSQL.
        If a chunk with the same document_id and chunk_index exists,
        it will be replaced (upsert behavior for re-ingestion).
        
        Validates Requirements:
        - 4.1: Store text chunks with embeddings in PostgreSQL
        - 4.3: Replace old chunks when re-ingesting documents
        - 4.5: Associate metadata with chunks
        
        Args:
            text: The text content of the chunk
            vector: Embedding vector as numpy array
            document_id: UUID of the parent document
            chunk_index: Index of this chunk within the document
            metadata: Optional metadata dictionary
            
        Returns:
            UUID string of the stored chunk
            
        Raises:
            ValueError: If text is empty or vector has invalid shape
            SQLAlchemyError: If database operation fails
            
        Example:
            chunk_id = service.store_embedding(
                text="This is a text chunk",
                vector=np.array([0.1, 0.2, ...]),
                document_id=doc_uuid,
                chunk_index=0,
                metadata={"document_title": "My Doc"}
            )
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if vector is None or len(vector) == 0:
            raise ValueError("Vector cannot be empty")
        
        # Convert numpy array to list for pgvector
        vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        
        with self.db_manager.session_scope() as session:
            # Check if chunk already exists (for re-ingestion)
            existing_chunk = session.query(TextChunk).filter(
                TextChunk.document_id == document_id,
                TextChunk.chunk_index == chunk_index
            ).first()
            
            if existing_chunk:
                # Update existing chunk (re-ingestion case)
                logger.debug(
                    f"Updating existing chunk: document_id={document_id}, "
                    f"chunk_index={chunk_index}"
                )
                existing_chunk.text = text
                existing_chunk.embedding = vector_list
                existing_chunk.chunk_metadata = metadata or {}
                chunk_id = str(existing_chunk.id)
            else:
                # Create new chunk
                chunk = TextChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                    embedding=vector_list,
                    chunk_metadata=metadata or {}
                )
                session.add(chunk)
                session.flush()  # Get the ID before commit
                chunk_id = str(chunk.id)
                
                logger.debug(
                    f"Created new chunk: id={chunk_id}, document_id={document_id}, "
                    f"chunk_index={chunk_index}"
                )
        
        return chunk_id
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find most similar text chunks using cosine similarity.

        Performs vector similarity search using pgvector's cosine distance
        operator. Returns top-k most similar chunks that meet the similarity
        threshold, sorted by similarity score in descending order.

        Validates Requirements:
        - 5.1: Return top-k most similar chunks sorted by similarity
        - 5.2: Filter by similarity threshold
        - 5.3: Use cosine similarity for distance calculation
        - 5.5: Include similarity scores with results

        Args:
            query_vector: Query embedding vector as numpy array
            top_k: Maximum number of results to return (default: 5)
            threshold: Minimum similarity score (0.0 to 1.0, default: 0.7)

        Returns:
            List of dictionaries containing:
                - id: Chunk UUID
                - document_id: Parent document UUID
                - text: Chunk text content
                - metadata: Chunk metadata dictionary
                - similarity_score: Cosine similarity (0.0 to 1.0)
            Results are sorted by similarity_score in descending order.

        Raises:
            ValueError: If query_vector is invalid or parameters out of range

        Example:
            results = service.search_similar(
                query_vector=np.array([0.1, 0.2, ...]),
                top_k=5,
                threshold=0.7
            )
            for result in results:
                print(f"Similarity: {result['similarity_score']:.3f}")
                print(f"Text: {result['text'][:100]}...")
        """
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("Query vector cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        start_time = time.time()

        try:
            # Convert numpy array to list for SQL parameter
            vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            # Build SQL query using pgvector cosine distance operator (<=>)
            # Cosine distance: 1 - cosine_similarity
            # So similarity = 1 - distance
            sql = text("""
                SELECT 
                    id,
                    document_id,
                    text,
                    chunk_metadata,
                    1 - (embedding <=> :query_vector::vector) as similarity_score
                FROM text_chunks
                WHERE embedding IS NOT NULL
                    AND 1 - (embedding <=> :query_vector::vector) >= :threshold
                ORDER BY embedding <=> :query_vector::vector
                LIMIT :top_k
            """)

            with track_time(vector_search_duration_seconds, {'top_k_bucket': 'small' if top_k <= 5 else 'medium' if top_k <= 20 else 'large'}):
                with self.db_manager.session_scope() as session:
                    result = session.execute(
                        sql,
                        {
                            "query_vector": vector_list,
                            "threshold": threshold,
                            "top_k": top_k
                        }
                    )

                    rows = result.fetchall()

            # Format results
            results = []
            for row in rows:
                results.append({
                    "id": str(row.id),
                    "document_id": str(row.document_id),
                    "text": row.text,
                    "metadata": row.chunk_metadata or {},
                    "similarity_score": float(row.similarity_score)
                })

            # Record metrics
            duration = time.time() - start_time
            record_vector_search(duration, top_k, len(results))

            logger.info(
                "Vector search completed",
                results_count=len(results),
                top_k=top_k,
                threshold=threshold,
                duration_ms=duration * 1000
            )

            return results

        except Exception as e:
            logger.error("Vector search failed", error=str(e), exc_info=True)
            capture_exception(e, level="error", tags={"component": "vector_search"})
            raise
