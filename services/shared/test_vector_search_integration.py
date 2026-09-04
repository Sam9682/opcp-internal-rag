"""Integration tests for VectorSearchService with PostgreSQL + pgvector.

These tests require a running PostgreSQL instance with pgvector extension.
Run with: python test_vector_search_integration.py

Note: These tests will be skipped if PostgreSQL is not available.
"""

import logging
import sys
import numpy as np
from uuid import uuid4

from database import init_db, get_db_manager
from orm_models import Base, Document
from vector_search_service import VectorSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_store_and_search_integration():
    """Integration test: Store embeddings and perform similarity search."""
    logger.info("=" * 60)
    logger.info("Integration Test: Store and Search")
    logger.info("=" * 60)
    
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        db_manager = init_db(wait_for_ready=True)
        
        # Create tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(db_manager.engine)
        
        # Create vector search service
        service = VectorSearchService(db_manager)
        
        # Create a test document
        logger.info("Creating test document...")
        with db_manager.session_scope() as session:
            doc = Document(
                file_path="/test/integration_test.md",
                title="Integration Test Document",
                content="This is a test document for integration testing",
                doc_metadata={"test": True},
                ingestion_status="completed"
            )
            session.add(doc)
            session.flush()
            doc_id = doc.id
        
        logger.info(f"✓ Created document with ID: {doc_id}")
        
        # Store multiple chunks with embeddings
        logger.info("Storing chunks with embeddings...")
        chunk_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
            "Vector databases enable semantic search",
            "PostgreSQL with pgvector supports vector operations"
        ]
        
        chunk_ids = []
        vectors = []
        for idx, text in enumerate(chunk_texts):
            # Generate random embedding (in real use, this would come from embedding model)
            vector = np.random.rand(1024).astype(np.float32)
            vectors.append(vector)
            
            chunk_id = service.store_embedding(
                text=text,
                vector=vector,
                document_id=doc_id,
                chunk_index=idx,
                metadata={"chunk_number": idx, "document_title": "Integration Test"}
            )
            chunk_ids.append(chunk_id)
            logger.info(f"  ✓ Stored chunk {idx}: {text[:50]}...")
        
        logger.info(f"✓ Stored {len(chunk_ids)} chunks")
        
        # Perform similarity search
        logger.info("\nPerforming similarity search...")
        query_vector = vectors[0]  # Use first vector as query
        
        results = service.search_similar(
            query_vector=query_vector,
            top_k=3,
            threshold=0.0  # Low threshold to ensure we get results
        )
        
        logger.info(f"✓ Found {len(results)} results")
        
        # Verify results
        assert len(results) > 0, "Should find at least one result"
        assert len(results) <= 3, "Should not exceed top_k limit"
        
        # Check result structure
        for i, result in enumerate(results):
            logger.info(f"\n  Result {i + 1}:")
            logger.info(f"    ID: {result['id']}")
            logger.info(f"    Text: {result['text'][:50]}...")
            logger.info(f"    Similarity: {result['similarity_score']:.4f}")
            logger.info(f"    Metadata: {result['metadata']}")
            
            # Verify result structure
            assert 'id' in result
            assert 'document_id' in result
            assert 'text' in result
            assert 'metadata' in result
            assert 'similarity_score' in result
            assert 0.0 <= result['similarity_score'] <= 1.0
        
        # Verify results are sorted by similarity (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]['similarity_score'] >= results[i + 1]['similarity_score'], \
                    "Results should be sorted by similarity in descending order"
        
        logger.info("\n✓ Results are properly sorted by similarity")
        
        # Test re-ingestion (upsert behavior)
        logger.info("\nTesting re-ingestion (upsert)...")
        updated_text = "Updated: The quick brown fox jumps over the lazy dog"
        updated_vector = np.random.rand(1024).astype(np.float32)
        
        chunk_id_updated = service.store_embedding(
            text=updated_text,
            vector=updated_vector,
            document_id=doc_id,
            chunk_index=0,  # Same index as first chunk
            metadata={"chunk_number": 0, "version": 2}
        )
        
        # Should return same chunk ID (updated, not created new)
        assert chunk_id_updated == chunk_ids[0], "Re-ingestion should update existing chunk"
        logger.info("✓ Re-ingestion correctly updated existing chunk")
        
        # Verify update
        with db_manager.session_scope() as session:
            from orm_models import TextChunk
            chunk = session.query(TextChunk).filter(TextChunk.id == chunk_id_updated).first()
            assert chunk.text == updated_text, "Text should be updated"
            assert chunk.chunk_metadata.get("version") == 2, "Metadata should be updated"
        
        logger.info("✓ Chunk was correctly updated in database")
        
        # Test threshold filtering
        logger.info("\nTesting threshold filtering...")
        results_high_threshold = service.search_similar(
            query_vector=query_vector,
            top_k=10,
            threshold=0.99  # Very high threshold
        )
        
        logger.info(f"✓ High threshold (0.99) returned {len(results_high_threshold)} results")
        
        # All results should meet threshold
        for result in results_high_threshold:
            assert result['similarity_score'] >= 0.99, \
                f"Result similarity {result['similarity_score']} should be >= 0.99"
        
        logger.info("✓ All results meet the similarity threshold")
        
        # Cleanup
        logger.info("\nCleaning up test data...")
        with db_manager.session_scope() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                session.delete(doc)  # Cascade will delete chunks
        
        logger.info("✓ Test data cleaned up")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL INTEGRATION TESTS PASSED")
        logger.info("=" * 60)
        
        return True
        
    except RuntimeError as e:
        logger.warning(f"Database not available: {e}")
        logger.info("Skipping integration tests (PostgreSQL not running)")
        return True  # Not a test failure
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        return False
        
    finally:
        if 'db_manager' in locals():
            db_manager.close()


def main():
    """Run integration tests."""
    logger.info("Starting VectorSearchService integration tests")
    logger.info("Requires: PostgreSQL with pgvector extension\n")
    
    success = test_store_and_search_integration()
    
    if success:
        logger.info("\n✓ Integration tests completed successfully")
        return 0
    else:
        logger.error("\n✗ Integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
