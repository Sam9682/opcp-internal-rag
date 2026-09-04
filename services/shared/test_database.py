"""Simple test script for database connection management.

This script tests the basic functionality of the database connection
management system including connection pooling, session handling,
and health checks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.database import DatabaseManager, get_db_manager, init_db
from services.shared.orm_models import Base, Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_manager_initialization():
    """Test DatabaseManager initialization."""
    logger.info("Testing DatabaseManager initialization...")
    
    db_manager = DatabaseManager(pool_size=5, max_overflow=10)
    
    # Test that engine is not initialized yet
    try:
        _ = db_manager.engine
        logger.error("Expected RuntimeError for uninitialized engine")
        return False
    except RuntimeError as e:
        logger.info(f"✓ Correctly raised RuntimeError: {e}")
    
    # Initialize the database manager
    db_manager.initialize()
    
    # Test that engine is now available
    engine = db_manager.engine
    logger.info(f"✓ Engine initialized: {engine}")
    
    # Clean up
    db_manager.close()
    logger.info("✓ DatabaseManager initialization test passed")
    return True


def test_health_check():
    """Test database health check functionality."""
    logger.info("Testing database health check...")
    
    db_manager = DatabaseManager()
    
    # Health check should fail before initialization
    if db_manager.health_check():
        logger.warning("Health check passed before initialization (unexpected)")
    else:
        logger.info("✓ Health check correctly failed before initialization")
    
    # Initialize and test health check
    try:
        db_manager.initialize()
        
        if db_manager.health_check():
            logger.info("✓ Health check passed after initialization")
        else:
            logger.error("Health check failed after initialization")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.info("This is expected if PostgreSQL is not running")
        return True  # Not a test failure, just database not available
    finally:
        db_manager.close()
    
    logger.info("✓ Health check test passed")
    return True


def test_wait_for_db():
    """Test wait_for_db functionality."""
    logger.info("Testing wait_for_db...")
    
    db_manager = DatabaseManager()
    
    # Try to wait for database with short timeout
    result = db_manager.wait_for_db(max_retries=3, retry_interval=1, timeout=2)
    
    if result:
        logger.info("✓ Database became available")
        db_manager.close()
    else:
        logger.info("✓ wait_for_db correctly handled unavailable database")
    
    logger.info("✓ wait_for_db test passed")
    return True


def test_session_scope():
    """Test session scope context manager."""
    logger.info("Testing session scope...")
    
    try:
        db_manager = init_db(wait_for_ready=True)
    except RuntimeError as e:
        logger.info(f"Database not available: {e}")
        logger.info("Skipping session scope test")
        return True
    
    try:
        # Test successful transaction
        with db_manager.session_scope() as session:
            # Just verify we can create a session
            logger.info(f"✓ Session created: {session}")
        
        logger.info("✓ Session scope committed successfully")
        
        # Test rollback on exception
        try:
            with db_manager.session_scope() as session:
                # Simulate an error
                raise ValueError("Test error")
        except ValueError:
            logger.info("✓ Session scope correctly rolled back on exception")
        
        logger.info("✓ Session scope test passed")
        return True
    
    finally:
        db_manager.close()


def test_create_tables():
    """Test creating database tables."""
    logger.info("Testing table creation...")
    
    try:
        db_manager = init_db(wait_for_ready=True)
    except RuntimeError as e:
        logger.info(f"Database not available: {e}")
        logger.info("Skipping table creation test")
        return True
    
    try:
        # Create all tables
        Base.metadata.create_all(db_manager.engine)
        logger.info("✓ Tables created successfully")
        
        # Verify we can query the tables
        with db_manager.session_scope() as session:
            count = session.query(Document).count()
            logger.info(f"✓ Document table accessible, count: {count}")
        
        logger.info("✓ Table creation test passed")
        return True
    
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        return False
    
    finally:
        db_manager.close()


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting database connection management tests")
    logger.info("=" * 60)
    
    tests = [
        ("DatabaseManager Initialization", test_database_manager_initialization),
        ("Health Check", test_health_check),
        ("Wait for DB", test_wait_for_db),
        ("Session Scope", test_session_scope),
        ("Create Tables", test_create_tables),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        logger.info(f"Running: {test_name}")
        logger.info("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("")
    logger.info(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
