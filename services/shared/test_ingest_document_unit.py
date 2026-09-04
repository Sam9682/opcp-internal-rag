"""
Unit tests for ingest_document() without database dependency.

Tests the ingest_document() method logic using mocks to avoid
database requirements.

Requirements: 1.4, 1.5, 18.1
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import numpy as np

from .document_ingestion_service import DocumentIngestionService
from .models import IngestionJob


class TestIngestDocumentUnit:
    """Unit tests for ingest_document() method."""
    
    def test_ingest_document_validates_file_exists(self):
        """Test that ingest_document validates file exists."""
        service = DocumentIngestionService()
        
        with pytest.raises(ValueError, match="File does not exist"):
            service.ingest_document(Path("/nonexistent/file.md"))
    
    def test_ingest_document_validates_is_file(self):
        """Test that ingest_document validates path is a file."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="Path is not a file"):
                service.ingest_document(Path(tmp_dir))
    
    def test_ingest_document_validates_markdown_extension(self):
        """Test that ingest_document validates markdown extension."""
        service = DocumentIngestionService()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="not a markdown file"):
                service.ingest_document(temp_path)
        finally:
            temp_path.unlink()
    
    @patch('services.shared.document_ingestion_service.get_db_manager')
    def test_ingest_document_reads_file_content(self, mock_get_db):
        """Test that ingest_document reads file content."""
        # Setup mocks
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        mock_get_db.return_value = mock_db_manager
        
        # Mock query results
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Create service with mocked components
        mock_preprocessor = Mock()
        mock_preprocessor.extract_metadata.return_value = {'title': 'Test'}
        mock_preprocessor.clean_markdown.return_value = "Cleaned text"
        mock_preprocessor.chunk_text.return_value = ["chunk1", "chunk2"]
        
        mock_embedding = Mock()
        mock_embedding.embed_batch.return_value = [
            np.random.rand(1024),
            np.random.rand(1024)
        ]
        
        mock_vector_search = Mock()
        mock_vector_search.store_embedding.return_value = "chunk-id"
        
        service = DocumentIngestionService(
            text_preprocessor=mock_preprocessor,
            embedding_service=mock_embedding,
            vector_search_service=mock_vector_search,
            db_manager=mock_db_manager
        )
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nTest content")
            temp_path = Path(f.name)
        
        try:
            job = service.ingest_document(temp_path)
            
            # Verify file was read
            assert mock_preprocessor.extract_metadata.called
            assert mock_preprocessor.clean_markdown.called
            assert mock_preprocessor.chunk_text.called
        
        finally:
            temp_path.unlink()
    
    @patch('services.shared.document_ingestion_service.get_db_manager')
    def test_ingest_document_handles_empty_file(self, mock_get_db):
        """Test that ingest_document handles empty files."""
        # Setup mocks
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        mock_get_db.return_value = mock_db_manager
        
        service = DocumentIngestionService(db_manager=mock_db_manager)
        
        # Create empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            job = service.ingest_document(temp_path)
            
            # Should fail with empty file
            assert job.status == 'failed'
            assert job.error_message is not None
            assert 'empty' in job.error_message.lower()
        
        finally:
            temp_path.unlink()
    
    @patch('services.shared.document_ingestion_service.get_db_manager')
    def test_ingest_document_creates_job_tracking(self, mock_get_db):
        """Test that ingest_document creates IngestionJob for tracking."""
        # Setup mocks
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        mock_get_db.return_value = mock_db_manager
        
        # Mock query results
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Create service with mocked components
        mock_preprocessor = Mock()
        mock_preprocessor.extract_metadata.return_value = {'title': 'Test'}
        mock_preprocessor.clean_markdown.return_value = "Cleaned text"
        mock_preprocessor.chunk_text.return_value = ["chunk1"]
        
        mock_embedding = Mock()
        mock_embedding.embed_batch.return_value = [np.random.rand(1024)]
        
        mock_vector_search = Mock()
        mock_vector_search.store_embedding.return_value = "chunk-id"
        
        service = DocumentIngestionService(
            text_preprocessor=mock_preprocessor,
            embedding_service=mock_embedding,
            vector_search_service=mock_vector_search,
            db_manager=mock_db_manager
        )
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test\n\nContent")
            temp_path = Path(f.name)
        
        try:
            job = service.ingest_document(temp_path)
            
            # Verify job tracking
            assert isinstance(job, IngestionJob)
            assert job.file_path == str(temp_path)
            assert job.status in ['completed', 'failed']
            assert job.started_at is not None
            assert job.completed_at is not None
        
        finally:
            temp_path.unlink()
    
    @patch('services.shared.document_ingestion_service.get_db_manager')
    def test_ingest_document_coordinates_components(self, mock_get_db):
        """Test that ingest_document coordinates all pipeline components."""
        # Setup mocks
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        mock_get_db.return_value = mock_db_manager
        
        # Mock document object
        mock_doc = MagicMock()
        mock_doc.id = "test-doc-id"
        
        # Mock query results - first call returns None (new doc), second returns the doc
        mock_session.query.return_value.filter.return_value.first.side_effect = [None, mock_doc]
        
        # Create service with mocked components
        mock_preprocessor = Mock()
        mock_preprocessor.extract_metadata.return_value = {'title': 'Test'}
        mock_preprocessor.clean_markdown.return_value = "Cleaned text"
        mock_preprocessor.chunk_text.return_value = ["chunk1", "chunk2"]
        
        mock_embedding = Mock()
        mock_embedding.embed_batch.return_value = [
            np.random.rand(1024),
            np.random.rand(1024)
        ]
        
        mock_vector_search = Mock()
        mock_vector_search.store_embedding.return_value = "chunk-id"
        
        service = DocumentIngestionService(
            text_preprocessor=mock_preprocessor,
            embedding_service=mock_embedding,
            vector_search_service=mock_vector_search,
            db_manager=mock_db_manager
        )
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nTest content")
            temp_path = Path(f.name)
        
        try:
            job = service.ingest_document(temp_path)
            
            # Verify all components were called
            assert mock_preprocessor.extract_metadata.called
            assert mock_preprocessor.clean_markdown.called
            assert mock_preprocessor.chunk_text.called
            assert mock_embedding.embed_batch.called
            assert mock_vector_search.store_embedding.called
            
            # Verify correct order and parameters
            assert mock_preprocessor.chunk_text.call_args[1]['chunk_size'] == 512
            assert mock_preprocessor.chunk_text.call_args[1]['overlap'] == 50
        
        finally:
            temp_path.unlink()
    
    @patch('services.shared.document_ingestion_service.get_db_manager')
    def test_ingest_document_handles_errors_gracefully(self, mock_get_db):
        """Test that ingest_document handles errors and updates job status."""
        # Setup mocks
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        mock_get_db.return_value = mock_db_manager
        
        # Mock query results
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Create service with mocked components that raise error
        mock_preprocessor = Mock()
        mock_preprocessor.extract_metadata.side_effect = Exception("Test error")
        
        service = DocumentIngestionService(
            text_preprocessor=mock_preprocessor,
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db_manager
        )
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test\n\nContent")
            temp_path = Path(f.name)
        
        try:
            job = service.ingest_document(temp_path)
            
            # Verify error handling
            assert job.status == 'failed'
            assert job.error_message is not None
            assert 'Test error' in job.error_message
            assert job.completed_at is not None
        
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
