"""
Document Ingestion Service for RAG Application

This module provides the DocumentIngestionService class that monitors documentation
storage for changes and triggers the ingestion pipeline. It handles file watching,
event processing, and coordinates the text preprocessing, embedding, and storage
components.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 18.1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Set, Dict, Any, TYPE_CHECKING
from queue import Queue
from threading import Thread, Event
from datetime import datetime, timedelta
from uuid import uuid4

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from sqlalchemy.orm import Session

from .text_preprocessor import TextPreprocessor
from .vector_search_service import VectorSearchService
from .database import DatabaseManager, get_db_manager, init_db
from .orm_models import Document as ORMDocument, IngestionJob as ORMIngestionJob
from .models import IngestionJob

if TYPE_CHECKING:
    from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MarkdownFileEventHandler(FileSystemEventHandler):
    """
    File system event handler for markdown files.
    
    Monitors file system events (create, modify, delete) for markdown files
    and adds them to a processing queue.
    
    Requirements: 1.1, 1.2, 1.3
    """
    
    def __init__(self, event_queue: Queue):
        """
        Initialize the event handler.
        
        Args:
            event_queue: Queue to store file system events for processing
        """
        super().__init__()
        self.event_queue = event_queue
        self._processing_files: Set[str] = set()
        logger.info("MarkdownFileEventHandler initialized")
    
    def _is_markdown_file(self, path: str) -> bool:
        """
        Check if the file is a markdown file.
        
        Args:
            path: File path to check
            
        Returns:
            True if file has .md extension, False otherwise
        """
        return path.lower().endswith('.md')
    
    def _should_process_event(self, event: FileSystemEvent) -> bool:
        """
        Determine if an event should be processed.
        
        Filters out:
        - Directory events
        - Non-markdown files
        - Temporary files
        
        Args:
            event: File system event
            
        Returns:
            True if event should be processed, False otherwise
        """
        # Skip directory events
        if event.is_directory:
            return False
        
        # Skip non-markdown files
        if not self._is_markdown_file(event.src_path):
            return False
        
        # Skip temporary files (starting with . or ending with ~)
        filename = Path(event.src_path).name
        if filename.startswith('.') or filename.endswith('~'):
            return False
        
        return True
    
    def on_created(self, event: FileSystemEvent):
        """
        Handle file creation events.
        
        When a markdown file is created, add it to the processing queue
        for ingestion.
        
        Validates Requirement 1.1: Detect new files within 10 seconds
        
        Args:
            event: File system event
        """
        if not self._should_process_event(event):
            return
        
        file_path = event.src_path
        logger.info(f"File created: {file_path}")
        
        # Add to processing queue
        self.event_queue.put(('created', file_path))
    
    def on_modified(self, event: FileSystemEvent):
        """
        Handle file modification events.
        
        When a markdown file is modified, add it to the processing queue
        for re-ingestion.
        
        Validates Requirement 1.2: Re-process modified files
        
        Args:
            event: File system event
        """
        if not self._should_process_event(event):
            return
        
        file_path = event.src_path
        logger.info(f"File modified: {file_path}")
        
        # Add to processing queue
        self.event_queue.put(('modified', file_path))
    
    def on_deleted(self, event: FileSystemEvent):
        """
        Handle file deletion events.
        
        When a markdown file is deleted, add it to the processing queue
        for cleanup (remove associated chunks).
        
        Validates Requirement 1.3: Remove chunks when files are deleted
        
        Args:
            event: File system event
        """
        if not self._should_process_event(event):
            return
        
        file_path = event.src_path
        logger.info(f"File deleted: {file_path}")
        
        # Add to processing queue
        self.event_queue.put(('deleted', file_path))


class DocumentIngestionService:
    """
    Document ingestion service for monitoring and processing markdown files.
    
    This service monitors a documentation directory for changes and triggers
    the ingestion pipeline. It handles:
    - File watching using watchdog library
    - Event queue management
    - Processing loop for file events
    - Coordination with text preprocessor, embedding service, and vector search
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 18.1
    """
    
    def __init__(
        self,
        text_preprocessor: Optional[TextPreprocessor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_search_service: Optional[VectorSearchService] = None,
        db_manager: Optional[DatabaseManager] = None,
        max_retries: int = 3,
        base_retry_delay: float = 2.0
    ):
        """
        Initialize the DocumentIngestionService.
        
        Sets up the event queue, file system observer, and processing thread.
        Initializes the ingestion pipeline components.
        
        Args:
            text_preprocessor: Text preprocessing component (creates if None)
            embedding_service: Embedding generation component (creates if None)
            vector_search_service: Vector search component (creates if None)
            db_manager: Database manager (uses global if None)
            max_retries: Maximum number of retry attempts for failed jobs (default: 3)
            base_retry_delay: Base delay in seconds for exponential backoff (default: 2.0)
        """
        self.event_queue: Queue = Queue()
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[MarkdownFileEventHandler] = None
        self.processing_thread: Optional[Thread] = None
        self.retry_thread: Optional[Thread] = None
        self.stop_event: Event = Event()
        self.watch_path: Optional[Path] = None
        
        # Retry configuration
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        
        # Initialize pipeline components
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        if embedding_service is not None:
            self.embedding_service = embedding_service
        else:
            import os
            embedding_url = os.getenv("EMBEDDING_SERVICE_URL")
            if embedding_url:
                from .embedding_client import EmbeddingClient
                self.embedding_service = EmbeddingClient(service_url=embedding_url)
            else:
                from .embedding_service import EmbeddingService
                self.embedding_service = EmbeddingService()
        self.vector_search_service = vector_search_service or VectorSearchService(db_manager)
        self.db_manager = db_manager or init_db()
        
        logger.info(
            f"DocumentIngestionService initialized with pipeline components "
            f"(max_retries={max_retries}, base_retry_delay={base_retry_delay}s)"
        )
    
    def watch_directory(self, path: Path) -> None:
        """
        Monitor directory for markdown file changes and trigger ingestion.
        
        Sets up a file system watcher using the watchdog library to monitor
        the specified directory for markdown file changes. Events are queued
        and processed by a separate thread.
        
        Preconditions:
        - path exists and is a directory
        - Process has read permissions on directory
        - Ingestion service is initialized
        
        Postconditions:
        - File system watcher is active
        - New/modified .md files trigger ingestion
        - Deleted files trigger cleanup
        - Watcher runs until explicitly stopped
        - All file events are logged
        
        Loop Invariants:
        - Watcher remains active throughout execution
        - Event queue is processed in order
        - No events are lost or duplicated
        
        Validates Requirements:
        - 1.1: Detect new files within 10 seconds
        - 1.2: Re-process modified files
        - 1.3: Remove chunks when files are deleted
        
        Args:
            path: Path to directory to monitor
            
        Raises:
            ValueError: If path does not exist or is not a directory
            RuntimeError: If watcher is already running
            
        Example:
            service = DocumentIngestionService()
            service.watch_directory(Path("/docs"))
            # Watcher is now active and processing events
        """
        # Validate preconditions
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        if self.observer is not None and self.observer.is_alive():
            raise RuntimeError("Watcher is already running")
        
        self.watch_path = path
        logger.info(f"Starting directory watch: {path}")
        
        # Step 1: Create event handler
        self.event_handler = MarkdownFileEventHandler(self.event_queue)
        
        # Step 2: Create and configure observer
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(path),
            recursive=True  # Monitor subdirectories
        )
        
        # Step 3: Start observer
        self.observer.start()
        logger.info(f"File system observer started for: {path}")
        
        # Step 4: Start event processing thread
        self.stop_event.clear()
        self.processing_thread = Thread(
            target=self._process_events,
            name="IngestionEventProcessor",
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Event processing thread started")
        
        # Step 5: Start retry processing thread
        self.start_retry_processor()
        
        logger.info(f"Directory watch active: {path}")
    
    def stop_watching(self) -> None:
        """
        Stop the directory watcher and event processing.
        
        Gracefully shuts down the file system observer and event processing
        thread. Waits for the current event to finish processing before
        stopping.
        
        Postconditions:
        - File system watcher is stopped
        - Event processing thread is stopped
        - Retry processing thread is stopped
        - All resources are cleaned up
        """
        logger.info("Stopping directory watch...")
        
        # Step 1: Signal processing threads to stop
        self.stop_event.set()
        
        # Step 2: Stop observer
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.observer = None
            logger.info("File system observer stopped")
        
        # Step 3: Wait for processing thread to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
            logger.info("Event processing thread stopped")
        
        # Step 4: Wait for retry thread to finish
        if self.retry_thread is not None:
            self.retry_thread.join(timeout=5.0)
            self.retry_thread = None
            logger.info("Retry processing thread stopped")
        
        logger.info("Directory watch stopped")
    
    def _process_events(self) -> None:
        """
        Event processing loop (runs in separate thread).
        
        Continuously processes events from the queue until stop_event is set.
        Each event triggers the appropriate action (ingest, re-ingest, or delete).
        
        This is an internal method that runs in a separate thread and should
        not be called directly. Use watch_directory() to start the watcher.
        
        Loop Invariants:
        - Events are processed in FIFO order
        - Each event is processed exactly once
        - Processing continues until stop_event is set
        """
        logger.info("Event processing loop started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for event with timeout to allow checking stop_event
                if not self.event_queue.empty():
                    event_type, file_path = self.event_queue.get(timeout=1.0)
                    
                    logger.info(f"Processing event: {event_type} - {file_path}")
                    
                    # Process event based on type
                    if event_type == 'created':
                        self._handle_file_created(file_path)
                    elif event_type == 'modified':
                        self._handle_file_modified(file_path)
                    elif event_type == 'deleted':
                        self._handle_file_deleted(file_path)
                    else:
                        logger.warning(f"Unknown event type: {event_type}")
                    
                    # Mark task as done
                    self.event_queue.task_done()
                else:
                    # No events, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                # Continue processing other events
        
        logger.info("Event processing loop stopped")
    
    def _handle_file_created(self, file_path: str) -> None:
        """
        Handle file creation event.
        
        Triggers ingestion for newly created markdown files.
        
        Args:
            file_path: Path to the created file
        """
        logger.info(f"Ingesting new file: {file_path}")
        try:
            job = self.ingest_document(Path(file_path))
            if job.status == 'completed':
                logger.info(f"Successfully ingested {job.chunks_created} chunks from {file_path}")
            else:
                logger.error(f"Ingestion failed for {file_path}: {job.error_message}")
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}", exc_info=True)
    
    def _handle_file_modified(self, file_path: str) -> None:
        """
        Handle file modification event.
        
        Triggers re-ingestion for modified markdown files.
        
        Args:
            file_path: Path to the modified file
        """
        logger.info(f"Re-ingesting modified file: {file_path}")
        try:
            job = self.ingest_document(Path(file_path))
            if job.status == 'completed':
                logger.info(f"Successfully re-ingested {job.chunks_created} chunks from {file_path}")
            else:
                logger.error(f"Re-ingestion failed for {file_path}: {job.error_message}")
        except Exception as e:
            logger.error(f"Error re-ingesting file {file_path}: {e}", exc_info=True)
    
    def _handle_file_deleted(self, file_path: str) -> None:
        """
        Handle file deletion event.
        
        Removes all chunks associated with the deleted file.
        
        Args:
            file_path: Path to the deleted file
        """
        logger.info(f"Removing chunks for deleted file: {file_path}")
        try:
            with self.db_manager.session_scope() as session:
                # Find document by file path
                document = session.query(ORMDocument).filter(
                    ORMDocument.file_path == file_path
                ).first()
                
                if document:
                    # Delete document (cascades to chunks)
                    session.delete(document)
                    logger.info(f"Deleted document and chunks for {file_path}")
                else:
                    logger.warning(f"No document found for deleted file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting document {file_path}: {e}", exc_info=True)
    
    def ingest_document(self, file_path: Path) -> IngestionJob:
        """
        Process a single document through the complete ingestion pipeline.
        
        Coordinates TextPreprocessor, EmbeddingService, and VectorSearchService
        to ingest a markdown document. Creates an IngestionJob for tracking,
        uses database transactions for atomic operations, and handles errors
        with logging and status updates.
        
        This implements the complete document ingestion pipeline from the design:
        1. Create ingestion job
        2. Read and parse markdown file
        3. Extract metadata
        4. Create/update document record
        5. Preprocess and chunk text
        6. Generate embeddings and store chunks
        7. Update document and job status
        
        Preconditions:
        - file_path exists and is readable
        - file_path points to valid markdown file
        - PostgreSQL database is accessible
        - Embedding service is initialized
        
        Postconditions:
        - Document is stored in database
        - All text chunks are vectorized and stored
        - IngestionJob status is 'completed' or 'failed'
        - If successful: chunks_created > 0
        
        Loop Invariants:
        - All processed chunks have valid embeddings
        - Database remains in consistent state (via transactions)
        
        Validates Requirements:
        - 1.4: Perform batch ingestion of all existing markdown files on startup
        - 1.5: Log errors and retry with exponential backoff when ingestion fails
        - 18.1: Ensure all chunks are stored or none are (atomic transactions)
        
        Args:
            file_path: Path to the markdown file to ingest
            
        Returns:
            IngestionJob with status and results
            
        Raises:
            ValueError: If file_path is invalid
            
        Example:
            service = DocumentIngestionService()
            job = service.ingest_document(Path("/docs/user-guide.md"))
            if job.status == 'completed':
                print(f"Ingested {job.chunks_created} chunks")
        """
        # Validate preconditions
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not file_path.suffix.lower() == '.md':
            raise ValueError(f"File is not a markdown file: {file_path}")
        
        # Step 1: Create ingestion job
        job_id = str(uuid4())
        job = IngestionJob(
            id=job_id,
            file_path=str(file_path),
            status='processing',
            started_at=datetime.now()
        )
        
        logger.info(f"Starting ingestion job {job_id} for {file_path}")
        
        try:
            # Step 2: Read and parse markdown file
            content = file_path.read_text(encoding='utf-8')
            if not content or not content.strip():
                raise ValueError("File is empty")
            
            logger.debug(f"Read {len(content)} characters from {file_path}")
            
            # Step 3: Extract metadata
            metadata = self.text_preprocessor.extract_metadata(content, str(file_path))
            logger.debug(f"Extracted metadata: {metadata}")
            
            # Step 4: Create or update document record (in transaction)
            with self.db_manager.session_scope() as session:
                # Check if document already exists (re-ingestion case)
                existing_doc = session.query(ORMDocument).filter(
                    ORMDocument.file_path == str(file_path)
                ).first()
                
                if existing_doc:
                    # Update existing document
                    logger.info(f"Updating existing document: {existing_doc.id}")
                    existing_doc.title = metadata.get('title', file_path.stem)
                    existing_doc.content = content
                    existing_doc.doc_metadata = metadata
                    existing_doc.updated_at = datetime.now()
                    existing_doc.ingestion_status = 'processing'
                    document_id = existing_doc.id
                    
                    # Delete old chunks (will be replaced)
                    session.query(ORMDocument).filter(
                        ORMDocument.id == document_id
                    ).first()  # Ensure document is in session
                else:
                    # Create new document
                    document = ORMDocument(
                        file_path=str(file_path),
                        title=metadata.get('title', file_path.stem),
                        content=content,
                        doc_metadata=metadata,
                        ingestion_status='processing'
                    )
                    session.add(document)
                    session.flush()  # Get the ID
                    document_id = document.id
                    logger.info(f"Created new document: {document_id}")
            
            # Step 5: Preprocess and chunk text
            logger.debug("Cleaning markdown content")
            cleaned_text = self.text_preprocessor.clean_markdown(content)
            
            logger.debug("Chunking text")
            chunks = self.text_preprocessor.chunk_text(
                cleaned_text,
                chunk_size=512,
                overlap=50
            )
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Step 6: Generate embeddings and store chunks (in transaction)
            chunks_created = 0
            
            # Use a single transaction for all chunk operations (atomicity)
            with self.db_manager.session_scope() as session:
                # Verify document exists in this session
                doc = session.query(ORMDocument).filter(
                    ORMDocument.id == document_id
                ).first()
                
                if not doc:
                    raise ValueError(f"Document {document_id} not found")
                
                # Process chunks in batches for efficiency
                batch_size = 10
                for batch_start in range(0, len(chunks), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]
                    
                    # Generate embeddings for batch
                    logger.debug(f"Generating embeddings for chunks {batch_start}-{batch_end}")
                    embeddings = self.embedding_service.embed_batch(batch_chunks)
                    
                    # Store each chunk with its embedding
                    for idx, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        chunk_index = batch_start + idx
                        
                        # Validate embedding
                        if embedding.shape[0] != 1024:
                            raise ValueError(
                                f"Invalid embedding dimension: {embedding.shape[0]}, expected 1024"
                            )
                        
                        # Store chunk using vector search service
                        # This handles upsert logic for re-ingestion
                        chunk_id = self.vector_search_service.store_embedding(
                            text=chunk_text,
                            vector=embedding,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            metadata={
                                'document_title': metadata.get('title', file_path.stem),
                                'file_path': str(file_path)
                            }
                        )
                        
                        chunks_created += 1
                        logger.debug(f"Stored chunk {chunk_index}: {chunk_id}")
                
                # Step 7: Update document status
                doc.ingestion_status = 'completed'
                doc.updated_at = datetime.now()
                
                logger.info(f"Successfully stored {chunks_created} chunks")
            
            # Update job status
            job.status = 'completed'
            job.completed_at = datetime.now()
            job.chunks_created = chunks_created
            
            logger.info(
                f"Ingestion job {job_id} completed: {chunks_created} chunks created"
            )
            
        except Exception as e:
            # Handle errors
            logger.error(f"Ingestion job {job_id} failed: {e}", exc_info=True)
            
            job.status = 'failed'
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            # Update document status if it exists
            try:
                with self.db_manager.session_scope() as session:
                    doc = session.query(ORMDocument).filter(
                        ORMDocument.file_path == str(file_path)
                    ).first()
                    
                    if doc:
                        doc.ingestion_status = 'failed'
                        doc.updated_at = datetime.now()
            except Exception as update_error:
                logger.error(f"Failed to update document status: {update_error}")
            
            # Queue for retry with exponential backoff
            self.queue_retry(job, str(e))
        
        finally:
            # Save ingestion job to database
            try:
                with self.db_manager.session_scope() as session:
                    orm_job = ORMIngestionJob(
                        id=uuid4(),  # Use new UUID for ORM
                        file_path=job.file_path,
                        status=job.status,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                        error_message=job.error_message,
                        chunks_created=job.chunks_created,
                        retry_count=job.retry_count,
                        max_retries=job.max_retries,
                        next_retry_at=job.next_retry_at
                    )
                    session.add(orm_job)
                    logger.debug(f"Saved ingestion job {job_id} to database")
            except Exception as save_error:
                logger.error(f"Failed to save ingestion job: {save_error}")
        
        return job
    
    def is_watching(self) -> bool:
        """
        Check if the watcher is currently active.
        
        Returns:
            True if watcher is running, False otherwise
        """
        return (
            self.observer is not None and
            self.observer.is_alive() and
            not self.stop_event.is_set()
        )
    def batch_ingest(self, directory_path: Path) -> Dict[str, Any]:
        """
        Process multiple documents in batch for startup processing.

        Scans the specified directory for all existing markdown files and
        processes them through the ingestion pipeline. Provides progress
        tracking and handles multiple documents efficiently.

        This method is designed for startup batch ingestion of all existing
        markdown files in a documentation directory. It processes files
        sequentially and provides detailed progress information.

        Preconditions:
        - directory_path exists and is a directory
        - Process has read permissions on directory
        - Ingestion service is initialized

        Postconditions:
        - All markdown files in directory are processed
        - Returns summary with success/failure counts
        - Each file has an associated IngestionJob

        Validates Requirements:
        - 1.4: Perform batch ingestion of all existing markdown files on startup

        Args:
            directory_path: Path to directory containing markdown files

        Returns:
            Dictionary with batch ingestion results:
            {
                'total_files': int,
                'successful': int,
                'failed': int,
                'total_chunks': int,
                'jobs': List[IngestionJob],
                'duration_seconds': float
            }

        Raises:
            ValueError: If directory_path is invalid

        Example:
            service = DocumentIngestionService()
            results = service.batch_ingest(Path("/docs"))
            print(f"Ingested {results['successful']}/{results['total_files']} files")
            print(f"Created {results['total_chunks']} chunks")
        """
        # Validate preconditions
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        logger.info(f"Starting batch ingestion from directory: {directory_path}")
        start_time = time.time()

        # Step 1: Scan directory for markdown files
        markdown_files = list(directory_path.rglob("*.md"))

        # Filter out hidden files and temporary files
        markdown_files = [
            f for f in markdown_files
            if not f.name.startswith('.') and not f.name.endswith('~')
        ]

        total_files = len(markdown_files)
        logger.info(f"Found {total_files} markdown files to process")

        if total_files == 0:
            logger.warning(f"No markdown files found in {directory_path}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_chunks': 0,
                'jobs': [],
                'duration_seconds': 0.0
            }

        # Step 2: Process files with progress tracking
        jobs = []
        successful = 0
        failed = 0
        total_chunks = 0

        for idx, file_path in enumerate(markdown_files, 1):
            logger.info(f"Processing file {idx}/{total_files}: {file_path}")

            try:
                # Ingest document
                job = self.ingest_document(file_path)
                jobs.append(job)

                # Update counters
                if job.status == 'completed':
                    successful += 1
                    total_chunks += job.chunks_created
                    logger.info(
                        f"✓ [{idx}/{total_files}] {file_path.name}: "
                        f"{job.chunks_created} chunks created"
                    )
                else:
                    failed += 1
                    logger.error(
                        f"✗ [{idx}/{total_files}] {file_path.name}: "
                        f"Failed - {job.error_message}"
                    )

            except Exception as e:
                # Handle unexpected errors
                failed += 1
                logger.error(
                    f"✗ [{idx}/{total_files}] {file_path.name}: "
                    f"Unexpected error - {e}",
                    exc_info=True
                )

                # Create failed job record
                job = IngestionJob(
                    id=str(uuid4()),
                    file_path=str(file_path),
                    status='failed',
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error_message=str(e),
                    chunks_created=0
                )
                jobs.append(job)

        # Step 3: Calculate duration and create summary
        duration = time.time() - start_time

        summary = {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'jobs': jobs,
            'duration_seconds': round(duration, 2)
        }

        # Log summary
        logger.info(
            f"Batch ingestion completed in {duration:.2f}s: "
            f"{successful} successful, {failed} failed, "
            f"{total_chunks} total chunks created"
        )

        return summary

    
    def get_queue_size(self) -> int:
        """
        Get the current size of the event queue.
        
        Returns:
            Number of events waiting to be processed
        """
        return self.event_queue.qsize()
    
    def calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay for retry attempts.
        
        Uses exponential backoff strategy: delay = base_delay * (2 ^ retry_count)
        
        Examples:
        - retry_count=0: 2.0 seconds
        - retry_count=1: 4.0 seconds
        - retry_count=2: 8.0 seconds
        - retry_count=3: 16.0 seconds
        
        Validates Requirements:
        - 1.5: Retry with exponential backoff when ingestion fails
        - 13.3: Queue jobs for retry with exponential backoff
        
        Args:
            retry_count: Number of previous retry attempts
            
        Returns:
            Delay in seconds before next retry
        """
        delay = self.base_retry_delay * (2 ** retry_count)
        logger.debug(f"Calculated retry delay for attempt {retry_count}: {delay}s")
        return delay
    
    def queue_retry(self, job: IngestionJob, error_message: str) -> None:
        """
        Queue a failed ingestion job for retry with exponential backoff.
        
        If the job has not exceeded max_retries, it will be queued for retry
        with an exponentially increasing delay. The job status is updated to
        'queued' and next_retry_at is set based on the exponential backoff
        calculation.
        
        If max_retries is exceeded, the job status remains 'failed' and no
        retry is scheduled.
        
        Validates Requirements:
        - 1.5: Log error and retry with exponential backoff when ingestion fails
        - 13.3: Queue jobs for retry with exponential backoff
        
        Args:
            job: The failed ingestion job
            error_message: Error message from the failed attempt
            
        Example:
            job = IngestionJob(...)
            service.queue_retry(job, "Connection timeout")
            # Job will be retried after exponential backoff delay
        """
        # Check if we should retry
        if job.retry_count >= self.max_retries:
            logger.warning(
                f"Job {job.id} exceeded max retries ({self.max_retries}), "
                f"marking as permanently failed: {job.file_path}"
            )
            job.status = 'failed'
            job.completed_at = datetime.now()
            return
        
        # Calculate next retry time with exponential backoff
        retry_delay = self.calculate_retry_delay(job.retry_count)
        next_retry_at = datetime.now() + timedelta(seconds=retry_delay)
        
        # Update job for retry
        job.status = 'queued'
        job.retry_count += 1
        job.next_retry_at = next_retry_at
        job.error_message = error_message
        job.completed_at = None  # Clear completion time for retry
        
        logger.info(
            f"Queued job {job.id} for retry {job.retry_count}/{self.max_retries} "
            f"at {next_retry_at.isoformat()} (delay: {retry_delay}s): {job.file_path}"
        )
        
        # Save updated job to database
        try:
            with self.db_manager.session_scope() as session:
                orm_job = session.query(ORMIngestionJob).filter(
                    ORMIngestionJob.file_path == job.file_path
                ).order_by(ORMIngestionJob.created_at.desc()).first()
                
                if orm_job:
                    orm_job.status = job.status
                    orm_job.retry_count = job.retry_count
                    orm_job.next_retry_at = job.next_retry_at
                    orm_job.error_message = job.error_message
                    orm_job.completed_at = job.completed_at
                    logger.debug(f"Updated job {job.id} in database for retry")
                else:
                    # Create new job record if not found
                    new_orm_job = ORMIngestionJob(
                        file_path=job.file_path,
                        status=job.status,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                        error_message=job.error_message,
                        chunks_created=job.chunks_created,
                        retry_count=job.retry_count,
                        max_retries=job.max_retries,
                        next_retry_at=job.next_retry_at
                    )
                    session.add(new_orm_job)
                    logger.debug(f"Created new job record for retry: {job.id}")
        except Exception as e:
            logger.error(f"Failed to save retry job to database: {e}", exc_info=True)
    
    def process_retry_jobs(self) -> None:
        """
        Process queued jobs that are ready for retry.
        
        This method runs in a separate thread and continuously checks for
        jobs that are queued for retry and whose next_retry_at time has
        passed. When found, it attempts to re-ingest the document.
        
        This is an internal method that runs in a separate thread and should
        not be called directly. It is started automatically when watch_directory()
        is called.
        
        Validates Requirements:
        - 1.5: Retry with exponential backoff when ingestion fails
        - 13.3: Queue jobs for retry with exponential backoff
        
        Loop Invariants:
        - Only processes jobs where next_retry_at <= current time
        - Each job is processed at most once per retry cycle
        - Processing continues until stop_event is set
        """
        logger.info("Retry processing loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check for jobs ready for retry
                with self.db_manager.session_scope() as session:
                    now = datetime.now()
                    
                    # Find jobs that are queued and ready for retry
                    retry_jobs = session.query(ORMIngestionJob).filter(
                        ORMIngestionJob.status == 'queued',
                        ORMIngestionJob.next_retry_at.isnot(None),
                        ORMIngestionJob.next_retry_at <= now
                    ).all()
                    
                    if retry_jobs:
                        logger.info(f"Found {len(retry_jobs)} jobs ready for retry")
                    
                    for orm_job in retry_jobs:
                        if self.stop_event.is_set():
                            break
                        
                        logger.info(
                            f"Retrying job (attempt {orm_job.retry_count}/{orm_job.max_retries}): "
                            f"{orm_job.file_path}"
                        )
                        
                        # Convert ORM job to dataclass
                        job = IngestionJob(
                            id=str(orm_job.id),
                            file_path=orm_job.file_path,
                            status=orm_job.status,
                            started_at=orm_job.started_at,
                            completed_at=orm_job.completed_at,
                            error_message=orm_job.error_message,
                            chunks_created=orm_job.chunks_created,
                            retry_count=orm_job.retry_count,
                            max_retries=orm_job.max_retries,
                            next_retry_at=orm_job.next_retry_at
                        )
                        
                        # Attempt ingestion
                        try:
                            file_path = Path(job.file_path)
                            
                            # Check if file still exists
                            if not file_path.exists():
                                logger.warning(
                                    f"File no longer exists, marking job as failed: {job.file_path}"
                                )
                                orm_job.status = 'failed'
                                orm_job.completed_at = datetime.now()
                                orm_job.error_message = "File no longer exists"
                                continue
                            
                            # Attempt ingestion
                            result_job = self.ingest_document(file_path)
                            
                            # Update ORM job with result
                            orm_job.status = result_job.status
                            orm_job.completed_at = result_job.completed_at
                            orm_job.chunks_created = result_job.chunks_created
                            orm_job.error_message = result_job.error_message
                            
                            if result_job.status == 'completed':
                                logger.info(
                                    f"Retry successful for {job.file_path}: "
                                    f"{result_job.chunks_created} chunks created"
                                )
                                orm_job.next_retry_at = None  # Clear retry time
                            else:
                                # Failed again, queue for another retry
                                logger.warning(
                                    f"Retry failed for {job.file_path}: {result_job.error_message}"
                                )
                                self.queue_retry(result_job, result_job.error_message or "Unknown error")
                                
                        except Exception as e:
                            logger.error(
                                f"Error during retry of {job.file_path}: {e}",
                                exc_info=True
                            )
                            # Queue for another retry
                            job.error_message = str(e)
                            self.queue_retry(job, str(e))
                
                # Sleep before checking again
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error in retry processing loop: {e}", exc_info=True)
                time.sleep(5.0)
        
        logger.info("Retry processing loop stopped")
    
    def start_retry_processor(self) -> None:
        """
        Start the retry processing thread.
        
        This method starts a background thread that continuously checks for
        and processes jobs that are ready for retry. It is called automatically
        by watch_directory() but can also be called manually if needed.
        
        Validates Requirements:
        - 1.5: Retry with exponential backoff when ingestion fails
        - 13.3: Queue jobs for retry with exponential backoff
        
        Raises:
            RuntimeError: If retry processor is already running
        """
        if self.retry_thread is not None and self.retry_thread.is_alive():
            raise RuntimeError("Retry processor is already running")
        
        logger.info("Starting retry processor thread")
        self.retry_thread = Thread(
            target=self.process_retry_jobs,
            name="IngestionRetryProcessor",
            daemon=True
        )
        self.retry_thread.start()
        logger.info("Retry processor thread started")

