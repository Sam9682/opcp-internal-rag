"""
Embedding Service for RAG Application

This module provides embedding generation functionality using the BGE-M3 model
for converting text into dense vector representations suitable for semantic search.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 14.4, 20.1, 20.2, 20.3
"""

import numpy as np
import os
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import logging
import time
import hashlib

from .logging_config import get_logger
from .metrics import record_embedding_generation, track_time, embedding_generation_duration_seconds
from .sentry_config import capture_exception
from .cache_client import CacheClient

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding service for generating vector embeddings using BGE-M3 model.
    
    This class handles:
    - Loading and caching the BGE-M3 model
    - GPU/CPU device selection
    - Single text embedding generation
    - Batch embedding generation for efficiency
    - Vector normalization for cosine similarity
    - Validation of embedding outputs
    - Redis caching for embeddings (Requirement 14.4)
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 14.4, 20.1, 20.2, 20.3
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        use_fp16: bool = True,
        cache_client: Optional[CacheClient] = None,
        cache_ttl: int = 86400  # 24 hours default
    ):
        """
        Initialize the EmbeddingService with BGE-M3 model.
        
        Loads the embedding model into memory and caches it for subsequent
        requests. Automatically selects GPU if available, otherwise uses CPU.
        Optionally uses Redis cache for embedding results.
        
        Preconditions:
        - model_name is valid Hugging Face model identifier
        - device is None, 'cpu', 'cuda', or 'cuda:N'
        
        Postconditions:
        - Model is loaded and cached in memory
        - Device is set (GPU if available, else CPU)
        - Model is ready for embedding generation
        - Cache client is initialized if provided
        
        Args:
            model_name: Hugging Face model identifier (default: BAAI/bge-m3)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            use_fp16: Whether to use FP16 precision on GPU (default: True)
            cache_client: Optional CacheClient for caching embeddings
            cache_ttl: Time-to-live for cached embeddings in seconds (default: 86400)
            
        Requirements: 14.4, 20.1, 20.2
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl
        
        # Step 1: Determine device (GPU if available, else CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        if self.cache_client and self.cache_client.is_available():
            logger.info(f"Embedding cache enabled with TTL: {self.cache_ttl}s")
        else:
            logger.info("Embedding cache disabled")
        
        # Step 2: Load BGE-M3 model
        try:
            cache_dir = os.getenv("MODEL_CACHE_DIR")
            # CPU doesn't support fp16 for LayerNorm; use fp32 (bge-base fits in ~430 MB)
            # GPU can use fp16 to save VRAM
            dtype = torch.float32 if self.device == "cpu" else torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("BGE-M3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
        # Step 3: Get embedding dimension
        self._embedding_dimension = self.model.config.hidden_size
        
        logger.info(f"Embedding dimension: {self._embedding_dimension}")
    
    def get_embedding_dimension(self) -> int:
        """
        Return the embedding vector dimension.
        
        Postconditions:
        - Returns positive integer (1024 for BGE-M3)
        
        Returns:
            Embedding dimension (1024 for BGE-M3)
            
        Requirements: 3.1, 18.2
        """
        return self._embedding_dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for single text.

        Converts text into a dense vector representation using the BGE-M3 model.
        The output is normalized for cosine similarity computation and validated
        to ensure no NaN or Inf values are present. Uses Redis cache if available.

        Preconditions:
        - text is non-empty string
        - text length <= MAX_INPUT_LENGTH (8192 tokens)
        - Embedding model is loaded and initialized

        Postconditions:
        - Returns numpy array of shape (EMBEDDING_DIMENSION,)
        - Array dtype is float32
        - Vector is normalized (L2 norm ≈ 1.0)
        - No NaN or Inf values in output
        - Deterministic output for same input text

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as numpy array

        Raises:
            ValueError: If text is empty or contains invalid characters
            RuntimeError: If embedding generation fails

        Requirements: 3.1, 3.2, 3.3, 3.5, 14.4
        """
        # Validate preconditions
        if not text or not text.strip():
            raise ValueError("Text must be non-empty")

        # Step 1: Check cache first (Requirement 14.4)
        if self.cache_client and self.cache_client.is_available():
            cached_embedding = self.cache_client.get_embedding(text)
            if cached_embedding is not None:
                logger.debug("Cache hit for embedding")
                # Convert from list back to numpy array
                embedding = np.array(cached_embedding, dtype=np.float32)
                return embedding

        start_time = time.time()

        try:
            # Step 2: Generate embedding using BGE-M3 (CLS token)
            with track_time(embedding_generation_duration_seconds, {'batch_size_bucket': 'single'}):
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, max_length=512, return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

            # Step 3: Ensure float32 numpy
            embedding = embedding.float().cpu().numpy()

            # Step 4: Validate embedding
            assert embedding.shape[0] == self._embedding_dimension, \
                f"Expected dimension {self._embedding_dimension}, got {embedding.shape[0]}"

            # Step 5: Check for NaN or Inf values
            if np.isnan(embedding).any():
                raise RuntimeError("Embedding contains NaN values")
            if np.isinf(embedding).any():
                raise RuntimeError("Embedding contains Inf values")

            # Step 6: Normalize vector for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                raise RuntimeError("Embedding has zero norm")

            # Postcondition: verify normalization
            assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5, \
                "Embedding not properly normalized"

            # Step 7: Store in cache (Requirement 14.4)
            if self.cache_client and self.cache_client.is_available():
                # Convert to list for JSON serialization
                self.cache_client.set_embedding(
                    text,
                    embedding.tolist(),
                    ttl=self.cache_ttl
                )
                logger.debug("Cached embedding")

            # Record metrics
            duration = time.time() - start_time
            record_embedding_generation(duration, batch_size=1)

            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e), exc_info=True)
            capture_exception(e, level="error", tags={"component": "embedding_service"})
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Processes multiple texts in a single forward pass through the model
        for improved efficiency. Batch size is automatically managed based on
        memory constraints.
        
        Preconditions:
        - texts is non-empty list of strings
        - All texts are non-empty
        - Embedding model is loaded and initialized
        
        Postconditions:
        - Returns list of numpy arrays, one per input text
        - Each array has shape (EMBEDDING_DIMENSION,)
        - All arrays are normalized
        - No NaN or Inf values in any output
        - Order of outputs matches order of inputs
        - Results are identical to calling embed_text() individually
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            ValueError: If texts is empty or contains empty strings
            RuntimeError: If batch embedding generation fails
            
        Requirements: 3.4, 20.3
        """
        # Validate preconditions
        if not texts:
            raise ValueError("texts list must be non-empty")
        
        if any(not text or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty")
        
        try:
            # Step 1: Determine optimal batch size
            # Use smaller batches for CPU, larger for GPU
            batch_size = 32 if self.device == "cuda" else 8
            
            # Step 2: Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for batch (CLS token)
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
                
                # Step 3: Validate and normalize each embedding
                for j, embedding in enumerate(batch_embeddings):
                    # Validate dimension
                    assert embedding.shape[0] == self._embedding_dimension, \
                        f"Expected dimension {self._embedding_dimension}, got {embedding.shape[0]}"
                    
                    # Check for NaN or Inf values
                    if np.isnan(embedding).any():
                        raise RuntimeError(f"Embedding {i+j} contains NaN values")
                    if np.isinf(embedding).any():
                        raise RuntimeError(f"Embedding {i+j} contains Inf values")
                    
                    # Normalize vector
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    else:
                        raise RuntimeError(f"Embedding {i+j} has zero norm")
                    
                    # Verify normalization
                    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5, \
                        f"Embedding {i+j} not properly normalized"
                    
                    all_embeddings.append(embedding)
            
            # Postcondition: verify output count matches input count
            assert len(all_embeddings) == len(texts), \
                f"Expected {len(texts)} embeddings, got {len(all_embeddings)}"
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")
