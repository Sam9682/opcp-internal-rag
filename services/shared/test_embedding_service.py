"""
Unit tests for EmbeddingService

Tests embedding generation, batch processing, and validation.
Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import pytest
import numpy as np
from embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""
    
    @pytest.fixture(scope="class")
    def embedding_service(self):
        """Create EmbeddingService instance for testing."""
        # Use CPU for testing to avoid GPU requirements
        return EmbeddingService(device="cpu", use_fp16=False)
    
    def test_initialization(self, embedding_service):
        """Test that EmbeddingService initializes correctly."""
        assert embedding_service is not None
        assert embedding_service.get_embedding_dimension() == 1024
    
    def test_embed_text_basic(self, embedding_service):
        """Test basic text embedding generation."""
        text = "This is a test sentence for embedding."
        embedding = embedding_service.embed_text(text)
        
        # Check shape
        assert embedding.shape == (1024,)
        
        # Check dtype
        assert embedding.dtype == np.float32
        
        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
        
        # Check no NaN or Inf values
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    def test_embed_text_deterministic(self, embedding_service):
        """Test that same text produces identical embeddings (deterministic)."""
        text = "Deterministic test sentence."
        
        embedding1 = embedding_service.embed_text(text)
        embedding2 = embedding_service.embed_text(text)
        
        # Embeddings should be identical
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
    
    def test_embed_text_empty_raises_error(self, embedding_service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text must be non-empty"):
            embedding_service.embed_text("")
        
        with pytest.raises(ValueError, match="Text must be non-empty"):
            embedding_service.embed_text("   ")
    
    def test_embed_batch_basic(self, embedding_service):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        embeddings = embedding_service.embed_batch(texts)
        
        # Check count
        assert len(embeddings) == 3
        
        # Check each embedding
        for embedding in embeddings:
            assert embedding.shape == (1024,)
            assert embedding.dtype == np.float32
            
            # Check normalization
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-5
            
            # Check no NaN or Inf values
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()
    
    def test_embed_batch_consistency(self, embedding_service):
        """Test that batch embedding produces same results as individual embedding."""
        texts = [
            "Consistency test sentence one.",
            "Consistency test sentence two."
        ]
        
        # Generate embeddings individually
        individual_embeddings = [embedding_service.embed_text(text) for text in texts]
        
        # Generate embeddings in batch
        batch_embeddings = embedding_service.embed_batch(texts)
        
        # Compare results
        assert len(individual_embeddings) == len(batch_embeddings)
        for ind_emb, batch_emb in zip(individual_embeddings, batch_embeddings):
            # Should be very close (allowing for minor floating point differences)
            np.testing.assert_array_almost_equal(ind_emb, batch_emb, decimal=5)
    
    def test_embed_batch_empty_list_raises_error(self, embedding_service):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="texts list must be non-empty"):
            embedding_service.embed_batch([])
    
    def test_embed_batch_with_empty_text_raises_error(self, embedding_service):
        """Test that list with empty text raises ValueError."""
        texts = ["Valid text", "", "Another valid text"]
        
        with pytest.raises(ValueError, match="All texts must be non-empty"):
            embedding_service.embed_batch(texts)
    
    def test_embed_text_different_texts_produce_different_embeddings(self, embedding_service):
        """Test that different texts produce different embeddings."""
        text1 = "This is about cats."
        text2 = "This is about dogs."
        
        embedding1 = embedding_service.embed_text(text1)
        embedding2 = embedding_service.embed_text(text2)
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2)
        
        # But both should be valid
        assert embedding1.shape == (1024,)
        assert embedding2.shape == (1024,)
    
    def test_embed_text_long_text(self, embedding_service):
        """Test embedding generation with longer text."""
        # Create a longer text (but within limits)
        text = " ".join(["This is a test sentence."] * 50)
        
        embedding = embedding_service.embed_text(text)
        
        # Should still produce valid embedding
        assert embedding.shape == (1024,)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
