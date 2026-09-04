#!/usr/bin/env python3
"""
Basic smoke test for cache implementation.
Tests that Redis connection works and basic operations function correctly.
"""

import redis
import json
import hashlib

def test_redis_connection():
    """Test basic Redis connection."""
    try:
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        client.ping()
        print("✓ Redis connection successful")
        return client
    except (redis.ConnectionError, redis.TimeoutError) as e:
        print(f"✗ Redis unavailable: {e}")
        return None

def test_basic_operations(client):
    """Test basic get/set operations."""
    if not client:
        print("✗ Skipping basic operations (Redis unavailable)")
        return
    
    # Test set and get
    key = "test:basic"
    value = "test_value"
    
    client.setex(key, 60, value)
    result = client.get(key)
    
    assert result == value, f"Expected {value}, got {result}"
    print("✓ Basic get/set operations work")
    
    # Clean up
    client.delete(key)

def test_json_operations(client):
    """Test JSON serialization."""
    if not client:
        print("✗ Skipping JSON operations (Redis unavailable)")
        return
    
    key = "test:json"
    value = {"name": "test", "count": 42}
    
    client.setex(key, 60, json.dumps(value))
    result = json.loads(client.get(key))
    
    assert result == value, f"Expected {value}, got {result}"
    print("✓ JSON serialization works")
    
    # Clean up
    client.delete(key)

def test_embedding_cache_pattern(client):
    """Test embedding cache pattern."""
    if not client:
        print("✗ Skipping embedding cache (Redis unavailable)")
        return
    
    text = "This is a test document"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Generate key with hash
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    key = f"embedding:{hash_obj.hexdigest()}"
    
    # Store embedding
    client.setex(key, 86400, json.dumps(embedding))
    
    # Retrieve embedding
    result = json.loads(client.get(key))
    
    assert result == embedding, f"Expected {embedding}, got {result}"
    print("✓ Embedding cache pattern works")
    
    # Clean up
    client.delete(key)

def test_response_cache_pattern(client):
    """Test response cache pattern."""
    if not client:
        print("✗ Skipping response cache (Redis unavailable)")
        return
    
    query = "What is RAG?"
    context_hash = "abc123"
    response = "RAG stands for Retrieval-Augmented Generation"
    
    # Generate key
    cache_data = f"{query}|{context_hash}"
    hash_obj = hashlib.sha256(cache_data.encode('utf-8'))
    key = f"response:{hash_obj.hexdigest()}"
    
    # Store response
    client.setex(key, 3600, response)
    
    # Retrieve response
    result = client.get(key)
    
    assert result == response, f"Expected {response}, got {result}"
    print("✓ Response cache pattern works")
    
    # Clean up
    client.delete(key)

def main():
    """Run all tests."""
    print("Testing cache implementation...")
    print()
    
    client = test_redis_connection()
    print()
    
    test_basic_operations(client)
    test_json_operations(client)
    test_embedding_cache_pattern(client)
    test_response_cache_pattern(client)
    
    print()
    if client:
        print("✓ All cache tests passed!")
    else:
        print("⚠ Redis unavailable - cache will be disabled but app will work")

if __name__ == "__main__":
    main()
