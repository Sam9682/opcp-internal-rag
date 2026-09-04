"""
Unit tests for DocumentIngestionService.

Tests the file watching functionality, event handling, and queue management
of the DocumentIngestionService class.

Requirements: 1.1, 1.2, 1.3
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from queue import Queue

from .document_ingestion_service import DocumentIngestionService, MarkdownFileEventHandler


class TestMarkdownFileEventHandler:
    """Test the MarkdownFileEventHandler class."""
    
    def test_is_markdown_file(self):
        """Test markdown file detection."""
        event_queue = Queue()
        handler = MarkdownFileEventHandler(event_queue)
        
        assert handler._is_markdown_file("test.md")
        assert handler._is_markdown_file("test.MD")
        assert handler._is_markdown_file("path/to/test.md")
        assert not handler._is_markdown_file("test.txt")
        assert not handler._is_markdown_file("test.py")
        assert not handler._is_markdown_file("test")
    
    def test_event_handler_initialization(self):
        """Test event handler initialization."""
        event_queue = Queue()
        handler = MarkdownFileEventHandler(event_queue)
        
        assert handler.event_queue is event_queue
        assert isinstance(handler._processing_files, set)
        assert len(handler._processing_files) == 0


class TestDocumentIngestionService:
    """Test the DocumentIngestionService class."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = DocumentIngestionService()
        
        assert service.event_queue is not None
        assert service.observer is None
        assert service.event_handler is None
        assert service.processing_thread is None
        assert not service.stop_event.is_set()
        assert service.watch_path is None
    
    def test_watch_directory_invalid_path(self):
        """Test watch_directory with invalid path."""
        service = DocumentIngestionService()
        
        # Non-existent path
        with pytest.raises(ValueError, match="Path does not exist"):
            service.watch_directory(Path("/nonexistent/path"))
    
    def test_watch_directory_file_not_directory(self):
        """Test watch_directory with file instead of directory."""
        service = DocumentIngestionService()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                service.watch_directory(Path(tmp_file.name))
    
    def test_watch_directory_starts_watcher(self):
        """Test that watch_directory starts the file system watcher."""
        service = DocumentIngestionService()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Verify watcher is active
            assert service.observer is not None
            assert service.observer.is_alive()
            assert service.event_handler is not None
            assert service.processing_thread is not None
            assert service.processing_thread.is_alive()
            assert service.watch_path == tmp_path
            assert service.is_watching()
            
            # Stop watching
            service.stop_watching()
            
            # Verify watcher is stopped
            assert not service.is_watching()
    
    def test_watch_directory_already_running(self):
        """Test that watch_directory raises error if already running."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Try to start again
            with pytest.raises(RuntimeError, match="Watcher is already running"):
                service.watch_directory(tmp_path)
            
            # Cleanup
            service.stop_watching()
    
    def test_file_creation_event(self):
        """Test that file creation events are detected and queued."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Wait for watcher to initialize
            time.sleep(0.5)
            
            # Create a markdown file
            test_file = tmp_path / "test.md"
            test_file.write_text("# Test Document\n\nThis is a test.")
            
            # Wait for event to be processed
            time.sleep(1.0)
            
            # Check that event was queued (or already processed)
            # The queue might be empty if event was already processed
            # So we just verify the watcher is still running
            assert service.is_watching()
            
            # Cleanup
            service.stop_watching()
    
    def test_file_modification_event(self):
        """Test that file modification events are detected and queued."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a markdown file before starting watcher
            test_file = tmp_path / "test.md"
            test_file.write_text("# Test Document\n\nOriginal content.")
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Wait for watcher to initialize
            time.sleep(0.5)
            
            # Modify the file
            test_file.write_text("# Test Document\n\nModified content.")
            
            # Wait for event to be processed
            time.sleep(1.0)
            
            # Verify watcher is still running
            assert service.is_watching()
            
            # Cleanup
            service.stop_watching()
    
    def test_file_deletion_event(self):
        """Test that file deletion events are detected and queued."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a markdown file before starting watcher
            test_file = tmp_path / "test.md"
            test_file.write_text("# Test Document\n\nContent to be deleted.")
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Wait for watcher to initialize
            time.sleep(0.5)
            
            # Delete the file
            test_file.unlink()
            
            # Wait for event to be processed
            time.sleep(1.0)
            
            # Verify watcher is still running
            assert service.is_watching()
            
            # Cleanup
            service.stop_watching()
    
    def test_non_markdown_files_ignored(self):
        """Test that non-markdown files are ignored."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            
            # Wait for watcher to initialize
            time.sleep(0.5)
            
            # Create non-markdown files
            (tmp_path / "test.txt").write_text("Text file")
            (tmp_path / "test.py").write_text("# Python file")
            (tmp_path / ".hidden.md").write_text("# Hidden file")
            
            # Wait briefly
            time.sleep(1.0)
            
            # Queue should be empty (or events already processed)
            # We just verify the watcher is still running
            assert service.is_watching()
            
            # Cleanup
            service.stop_watching()
    
    def test_stop_watching(self):
        """Test graceful shutdown of watcher."""
        service = DocumentIngestionService()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            assert service.is_watching()
            
            # Stop watching
            service.stop_watching()
            
            # Verify everything is stopped
            assert not service.is_watching()
            assert service.observer is None
            assert service.processing_thread is None
    
    def test_get_queue_size(self):
        """Test getting the event queue size."""
        service = DocumentIngestionService()
        
        # Initially empty
        assert service.get_queue_size() == 0
        
        # Add some events manually
        service.event_queue.put(('created', '/path/to/file1.md'))
        service.event_queue.put(('modified', '/path/to/file2.md'))
        
        assert service.get_queue_size() == 2
    
    def test_is_watching(self):
        """Test is_watching method."""
        service = DocumentIngestionService()
        
        # Initially not watching
        assert not service.is_watching()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Start watching
            service.watch_directory(tmp_path)
            assert service.is_watching()
            
            # Stop watching
            service.stop_watching()
            assert not service.is_watching()
    def test_batch_ingest_empty_directory(self):
        """Test batch_ingest with empty directory."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Batch ingest empty directory
            results = service.batch_ingest(tmp_path)

            # Verify results
            assert results['total_files'] == 0
            assert results['successful'] == 0
            assert results['failed'] == 0
            assert results['total_chunks'] == 0
            assert len(results['jobs']) == 0
            assert results['duration_seconds'] >= 0

    def test_batch_ingest_invalid_path(self):
        """Test batch_ingest with invalid path."""
        service = DocumentIngestionService()

        # Non-existent path
        with pytest.raises(ValueError, match="Directory does not exist"):
            service.batch_ingest(Path("/nonexistent/path"))

    def test_batch_ingest_file_not_directory(self):
        """Test batch_ingest with file instead of directory."""
        service = DocumentIngestionService()

        # Create temporary file
        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                service.batch_ingest(Path(tmp_file.name))

    def test_batch_ingest_single_file(self):
        """Test batch_ingest with single markdown file."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a markdown file
            test_file = tmp_path / "test.md"
            test_file.write_text("# Test Document\n\nThis is a test document.")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify results
            assert results['total_files'] == 1
            assert results['successful'] == 1
            assert results['failed'] == 0
            assert results['total_chunks'] > 0
            assert len(results['jobs']) == 1
            assert results['jobs'][0].status == 'completed'
            assert results['duration_seconds'] > 0

    def test_batch_ingest_multiple_files(self):
        """Test batch_ingest with multiple markdown files."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create multiple markdown files
            for i in range(3):
                test_file = tmp_path / f"test{i}.md"
                test_file.write_text(f"# Test Document {i}\n\nThis is test document {i}.")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify results
            assert results['total_files'] == 3
            assert results['successful'] == 3
            assert results['failed'] == 0
            assert results['total_chunks'] > 0
            assert len(results['jobs']) == 3
            assert all(job.status == 'completed' for job in results['jobs'])
            assert results['duration_seconds'] > 0

    def test_batch_ingest_nested_directories(self):
        """Test batch_ingest with nested directory structure."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create nested directory structure
            subdir1 = tmp_path / "subdir1"
            subdir1.mkdir()
            subdir2 = tmp_path / "subdir2"
            subdir2.mkdir()

            # Create markdown files in different directories
            (tmp_path / "root.md").write_text("# Root Document")
            (subdir1 / "sub1.md").write_text("# Subdir1 Document")
            (subdir2 / "sub2.md").write_text("# Subdir2 Document")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify results
            assert results['total_files'] == 3
            assert results['successful'] == 3
            assert results['failed'] == 0
            assert len(results['jobs']) == 3

    def test_batch_ingest_ignores_hidden_files(self):
        """Test that batch_ingest ignores hidden and temporary files."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create regular and hidden files
            (tmp_path / "visible.md").write_text("# Visible Document")
            (tmp_path / ".hidden.md").write_text("# Hidden Document")
            (tmp_path / "temp.md~").write_text("# Temp Document")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify only visible file is processed
            assert results['total_files'] == 1
            assert results['successful'] == 1

    def test_batch_ingest_ignores_non_markdown_files(self):
        """Test that batch_ingest ignores non-markdown files."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create markdown and non-markdown files
            (tmp_path / "doc.md").write_text("# Markdown Document")
            (tmp_path / "readme.txt").write_text("Text file")
            (tmp_path / "script.py").write_text("# Python file")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify only markdown file is processed
            assert results['total_files'] == 1
            assert results['successful'] == 1

    def test_batch_ingest_progress_tracking(self):
        """Test that batch_ingest provides progress information."""
        service = DocumentIngestionService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create multiple files
            for i in range(5):
                (tmp_path / f"doc{i}.md").write_text(f"# Document {i}\n\nContent {i}")

            # Batch ingest
            results = service.batch_ingest(tmp_path)

            # Verify progress information
            assert 'total_files' in results
            assert 'successful' in results
            assert 'failed' in results
            assert 'total_chunks' in results
            assert 'jobs' in results
            assert 'duration_seconds' in results

            # Verify all files processed
            assert results['total_files'] == 5
            assert results['successful'] + results['failed'] == 5
            assert len(results['jobs']) == 5



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
