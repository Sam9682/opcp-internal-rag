"""
Unit tests for CacheClient.

Tests the Redis cache client functionality including connection handling,
get/set operations, embedding caching, and response caching.

Requirements: 14.4
"""

import pytest
import time
from cache_client import CacheClient


class TestCacheClient:
    """Test suite for CacheClient."""
    
    def test_cache_client_initialization(self):
        """Test cache client can be initialized."""
        cache = CacheClient()
        # Should not raise exception even if Redis is unavailable
        assert cache is not None
    
    def test_cache_availability_check(self):
        """Test cache availability check."""
        cache = CacheClient()
        # is_available() should return bool
        assert isinstance(cache.is_available(), bool)
    
    def test_basic_get_set_when_available(self):
        """Test basic get/set operations when Redis is available."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        # Test set and get
        key = "test_key"
        value = "test_value"
        
        assert cache.set(key, value, ttl=60)
        result = cache.get(key)
        assert result == value
        
        # Clean up
        cache.delete(key)
    
    def test_json_serialization(self):
        """Test JSON serialization for complex objects."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        # Test with dict
        key = "test_dict"
        value = {"name": "test", "count": 42, "items": [1, 2, 3]}
        
        assert cache.set(key, value, ttl=60)
        result = cache.get(key)
        assert result == value
        
        # Clean up
        cache.delete(key)
    
    def test_embedding_cache(self):
        """Test embedding caching functionality."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        text = "This is a test document"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Cache embedding
        assert cache.set_embedding(text, embedding, ttl=60)
        
        # Retrieve embedding
        result = cache.get_embedding(text)
        assert result == embedding
        
        # Test cache miss
        result = cache.get_embedding("non-existent text")
        assert result is None
    
    def test_response_cache(self):
        """Test response caching functionality."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        query = "What is RAG?"
        context_hash = "abc123def456"
        response = "RAG stands for Retrieval-Augmented Generation..."
        
        # Cache response
        assert cache.set_response(query, context_hash, response, ttl=60)
        
        # Retrieve response
        result = cache.get_response(query, context_hash)
        assert result == response
        
        # Test cache miss with different context
        result = cache.get_response(query, "different_hash")
        assert result is None
    
    def test_ttl_expiration(self):
        """Test TTL expiration (quick test with 1 second TTL)."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        key = "test_ttl"
        value = "expires_soon"
        
        # Set with 1 second TTL
        assert cache.set(key, value, ttl=1)
        
        # Should be available immediately
        result = cache.get(key)
        assert result == value
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        result = cache.get(key)
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics retrieval."""
        cache = CacheClient()
        
        if not cache.is_available():
            pytest.skip("Redis not available")
        
        stats = cache.get_stats()
        
        # Should return dict with expected keys
        assert stats is not None
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'keys' in stats
    
    def test_graceful_degradation_when_unavailable(self):
        """Test that cache operations fail gracefully when Redis is unavailable."""
        # Create cache with invalid host to simulate unavailability
        cache = CacheClient(host='invalid-host-12345', port=9999)
        
        # Should not be available
        assert not cache.is_available()
        
        # Operations should return None/False without raising exceptions
        assert cache.get("any_key") is None
        assert cache.set("any_key", "any_value") is False
        assert cache.get_embedding("any_text") is None
        assert cache.set_embedding("any_text", [1, 2, 3]) is False
        assert cache.get_response("query", "hash") is None
        assert cache.set_response("query", "hash", "response") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
