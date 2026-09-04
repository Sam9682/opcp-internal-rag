"""
Smoke test for API Gateway.

This is a minimal test to verify the API can start and respond to basic requests.
Full unit tests are in task 14.7 (optional).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_health_check():
    """Test that health check endpoint works."""
    # Mock all the services to avoid initialization
    with patch('services.api.main.init_db'), \
         patch('services.api.main.EmbeddingService'), \
         patch('services.api.main.VectorSearchService'), \
         patch('services.api.main.LLMService'), \
         patch('services.api.main.LLMGuardService'), \
         patch('services.api.main.ConversationMemoryService'), \
         patch('services.api.main.RAGQueryEngine'), \
         patch('services.api.main.DocumentIngestionService'), \
         patch('services.api.main.TextPreprocessor'):
        
        # Import after patching
        from main import app
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-gateway"


def test_query_endpoint_structure():
    """Test that query endpoint exists and has correct structure."""
    with patch('services.api.main.init_db'), \
         patch('services.api.main.EmbeddingService'), \
         patch('services.api.main.VectorSearchService'), \
         patch('services.api.main.LLMService'), \
         patch('services.api.main.LLMGuardService'), \
         patch('services.api.main.ConversationMemoryService'), \
         patch('services.api.main.RAGQueryEngine'), \
         patch('services.api.main.DocumentIngestionService'), \
         patch('services.api.main.TextPreprocessor'):
        
        from main import app
        
        client = TestClient(app)
        
        # Test query endpoint exists (will fail without RAG engine initialized)
        response = client.post("/api/query", json={"query": "test"})
        # Should return 503 because RAG engine is not initialized in test
        assert response.status_code == 503


def test_ingest_endpoint_structure():
    """Test that ingest endpoint exists."""
    with patch('services.api.main.init_db'), \
         patch('services.api.main.EmbeddingService'), \
         patch('services.api.main.VectorSearchService'), \
         patch('services.api.main.LLMService'), \
         patch('services.api.main.LLMGuardService'), \
         patch('services.api.main.ConversationMemoryService'), \
         patch('services.api.main.RAGQueryEngine'), \
         patch('services.api.main.DocumentIngestionService'), \
         patch('services.api.main.TextPreprocessor'):
        
        from main import app
        
        client = TestClient(app)
        
        # Test ingest endpoint exists
        response = client.post("/api/ingest")
        # Should return 503 because ingestion service is not initialized in test
        assert response.status_code == 503


def test_conversation_endpoint_structure():
    """Test that conversation endpoint exists."""
    with patch('services.api.main.init_db'), \
         patch('services.api.main.EmbeddingService'), \
         patch('services.api.main.VectorSearchService'), \
         patch('services.api.main.LLMService'), \
         patch('services.api.main.LLMGuardService'), \
         patch('services.api.main.ConversationMemoryService'), \
         patch('services.api.main.RAGQueryEngine'), \
         patch('services.api.main.DocumentIngestionService'), \
         patch('services.api.main.TextPreprocessor'):
        
        from main import app
        
        client = TestClient(app)
        
        # Test conversation endpoint exists
        response = client.get("/api/conversations/test-id")
        # Should return 503 because RAG engine is not initialized in test
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
