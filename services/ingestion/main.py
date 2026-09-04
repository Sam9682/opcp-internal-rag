"""Document ingestion service."""

import sys
import os
from pathlib import Path

# Add parent directory to path to import shared modules as a package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from shared.document_ingestion_service import DocumentIngestionService
from shared.logging_config import configure_logging, get_logger
from shared.metrics import set_system_info
from shared.sentry_config import configure_sentry

# Configure structured logging
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
    service_name="ingestion-service"
)
logger = get_logger(__name__)


def main():
    """Main ingestion service loop."""
    logger.info("Document ingestion service starting...")
    
    # Configure Sentry
    configure_sentry(
        service_name="ingestion-service",
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("RELEASE", "1.0.0")
    )
    
    # Set system info for metrics
    set_system_info(
        service_name="ingestion-service",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )
    
    # Get documentation directory from environment or use default
    docs_dir = os.getenv('DOCS_DIR', '/docs')
    docs_path = Path(docs_dir)
    
    # Validate documentation directory
    if not docs_path.exists():
        logger.error(f"Documentation directory does not exist: {docs_path}")
        logger.info("Creating documentation directory...")
        docs_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService()
    logger.info("DocumentIngestionService initialized")
    
    # Start watching directory
    try:
        ingestion_service.watch_directory(docs_path)
        logger.info(f"Watching directory: {docs_path}")
        
        # Keep service running
        while ingestion_service.is_watching():
            import time
            time.sleep(1)
            
            # Log queue size periodically
            queue_size = ingestion_service.get_queue_size()
            if queue_size > 0:
                logger.debug(f"Event queue size: {queue_size}")
                
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error in ingestion service: {e}", exc_info=True)
    finally:
        # Graceful shutdown
        logger.info("Stopping ingestion service...")
        ingestion_service.stop_watching()
        logger.info("Ingestion service stopped")


if __name__ == "__main__":
    main()
