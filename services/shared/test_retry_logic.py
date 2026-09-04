"""
Unit tests for retry logic with exponential backoff in DocumentIngestionService.

Tests the retry mechanism for failed ingestion jobs, including:
- Exponential backoff calculation
- Retry queue management
- Retry processing
- Max retry limit enforcement

Requirements: 1.5, 13.3
"""

import pytest
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from .document_ingestion_service import DocumentIngestionService
from .models import IngestionJob
from .orm_models import IngestionJob as ORMIngestionJob


class TestRetryLogic:
    """Test suite for retry logic with exponential backoff."""
    
    def test_calculate_retry_delay_exponential_backoff(self):
        """
        Test that retry delay follows exponential backoff pattern.
        
        Validates Requirements:
        - 1.5: Retry with exponential backoff
        - 13.3: Queue jobs for retry with exponential backoff
        """
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=Mock(),
            base_retry_delay=2.0
        )
        
        # Test exponential backoff: delay = base * (2 ^ retry_count)
        assert service.calculate_retry_delay(0) == 2.0   # 2 * 2^0 = 2
        assert service.calculate_retry_delay(1) == 4.0   # 2 * 2^1 = 4
        assert service.calculate_retry_delay(2) == 8.0   # 2 * 2^2 = 8
        assert service.calculate_retry_delay(3) == 16.0  # 2 * 2^3 = 16
        assert service.calculate_retry_delay(4) == 32.0  # 2 * 2^4 = 32
    
    def test_calculate_retry_delay_custom_base(self):
        """Test retry delay calculation with custom base delay."""
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=Mock(),
            base_retry_delay=5.0
        )
        
        assert service.calculate_retry_delay(0) == 5.0   # 5 * 2^0 = 5
        assert service.calculate_retry_delay(1) == 10.0  # 5 * 2^1 = 10
        assert service.calculate_retry_delay(2) == 20.0  # 5 * 2^2 = 20
    
    def test_queue_retry_updates_job_fields(self):
        """
        Test that queue_retry updates job fields correctly.
        
        Validates Requirements:
        - 1.5: Log error and retry with exponential backoff
        - 13.3: Queue jobs for retry with exponential backoff
        """
        # Mock database manager
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        
        # Mock query to return None (no existing job)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db,
            max_retries=3,
            base_retry_delay=2.0
        )
        
        # Create a failed job
        job = IngestionJob(
            id=str(uuid4()),
            file_path="/test/doc.md",
            status='failed',
            retry_count=0,
            max_retries=3
        )
        
        # Queue for retry
        before_time = datetime.now()
        service.queue_retry(job, "Test error")
        after_time = datetime.now()
        
        # Verify job fields updated
        assert job.status == 'queued'
        assert job.retry_count == 1
        assert job.error_message == "Test error"
        assert job.next_retry_at is not None
        assert job.completed_at is None
        
        # Verify next_retry_at is in the future (2 seconds for first retry)
        expected_delay = timedelta(seconds=2.0)
        assert job.next_retry_at >= before_time + expected_delay
        assert job.next_retry_at <= after_time + expected_delay + timedelta(seconds=1)
    
    def test_queue_retry_increments_retry_count(self):
        """Test that retry_count is incremented on each retry."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db,
            max_retries=3
        )
        
        job = IngestionJob(
            id=str(uuid4()),
            file_path="/test/doc.md",
            status='failed',
            retry_count=0,
            max_retries=3
        )
        
        # First retry
        service.queue_retry(job, "Error 1")
        assert job.retry_count == 1
        
        # Second retry
        job.status = 'failed'
        service.queue_retry(job, "Error 2")
        assert job.retry_count == 2
        
        # Third retry
        job.status = 'failed'
        service.queue_retry(job, "Error 3")
        assert job.retry_count == 3
    
    def test_queue_retry_respects_max_retries(self):
        """
        Test that jobs exceeding max_retries are not queued for retry.
        
        Validates Requirements:
        - 1.5: Retry with exponential backoff (with limit)
        - 13.3: Queue jobs for retry with exponential backoff (with limit)
        """
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db,
            max_retries=3
        )
        
        # Job that has already been retried max times
        job = IngestionJob(
            id=str(uuid4()),
            file_path="/test/doc.md",
            status='failed',
            retry_count=3,  # Already at max
            max_retries=3
        )
        
        # Try to queue for retry
        service.queue_retry(job, "Final error")
        
        # Should remain failed, not queued
        assert job.status == 'failed'
        assert job.retry_count == 3
        assert job.completed_at is not None
        assert job.next_retry_at is None
    
    def test_queue_retry_exponential_delay_increases(self):
        """Test that retry delay increases exponentially with each attempt."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db,
            max_retries=3,
            base_retry_delay=2.0
        )
        
        job = IngestionJob(
            id=str(uuid4()),
            file_path="/test/doc.md",
            status='failed',
            retry_count=0,
            max_retries=3
        )
        
        # First retry (delay = 2 seconds)
        now1 = datetime.now()
        service.queue_retry(job, "Error 1")
        delay1 = (job.next_retry_at - now1).total_seconds()
        assert 1.9 <= delay1 <= 2.1  # Allow small timing variance
        
        # Second retry (delay = 4 seconds)
        job.status = 'failed'
        now2 = datetime.now()
        service.queue_retry(job, "Error 2")
        delay2 = (job.next_retry_at - now2).total_seconds()
        assert 3.9 <= delay2 <= 4.1
        
        # Third retry (delay = 8 seconds)
        job.status = 'failed'
        now3 = datetime.now()
        service.queue_retry(job, "Error 3")
        delay3 = (job.next_retry_at - now3).total_seconds()
        assert 7.9 <= delay3 <= 8.1
        
        # Verify exponential increase
        assert delay2 > delay1
        assert delay3 > delay2
    
    def test_ingest_document_queues_retry_on_failure(self):
        """
        Test that failed ingestion automatically queues for retry.
        
        Validates Requirements:
        - 1.5: Log error and retry with exponential backoff when ingestion fails
        """
        # Mock components
        mock_preprocessor = Mock()
        mock_preprocessor.extract_metadata.side_effect = Exception("Test failure")
        
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        
        service = DocumentIngestionService(
            text_preprocessor=mock_preprocessor,
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db,
            max_retries=3
        )
        
        # Create a test file
        test_file = Path("/tmp/test_doc.md")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Test Document\n\nTest content")
        
        try:
            # Attempt ingestion (will fail)
            job = service.ingest_document(test_file)
            
            # Verify job failed
            assert job.status == 'failed'
            assert job.error_message is not None
            
            # Verify retry was queued
            assert job.retry_count == 1
            assert job.next_retry_at is not None
            
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
    
    def test_start_retry_processor_starts_thread(self):
        """Test that start_retry_processor starts the retry thread."""
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=Mock()
        )
        
        # Start retry processor
        service.start_retry_processor()
        
        # Verify thread is running
        assert service.retry_thread is not None
        assert service.retry_thread.is_alive()
        
        # Cleanup
        service.stop_event.set()
        service.retry_thread.join(timeout=2.0)
    
    def test_start_retry_processor_raises_if_already_running(self):
        """Test that starting retry processor twice raises an error."""
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=Mock()
        )
        
        # Start retry processor
        service.start_retry_processor()
        
        try:
            # Try to start again
            with pytest.raises(RuntimeError, match="already running"):
                service.start_retry_processor()
        finally:
            # Cleanup
            service.stop_event.set()
            service.retry_thread.join(timeout=2.0)
    
    def test_stop_watching_stops_retry_thread(self):
        """Test that stop_watching stops the retry processor thread."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        service = DocumentIngestionService(
            text_preprocessor=Mock(),
            embedding_service=Mock(),
            vector_search_service=Mock(),
            db_manager=mock_db
        )
        
        # Start retry processor
        service.start_retry_processor()
        assert service.retry_thread.is_alive()
        
        # Stop watching (should stop retry thread)
        service.stop_watching()
        
        # Verify thread stopped
        assert service.retry_thread is None or not service.retry_thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
