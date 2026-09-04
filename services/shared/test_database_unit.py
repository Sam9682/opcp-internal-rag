"""Unit tests for database connection management (no database required).

These tests verify the database manager's initialization and configuration
without requiring an actual PostgreSQL connection.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.database import DatabaseManager, get_db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_manager_creation():
    """Test DatabaseManager can be created with custom configuration."""
    logger.info("Testing DatabaseManager creation...")
    
    db_manager = DatabaseManager(
        pool_size=15,
        max_overflow=25,
        pool_timeout=45,
        pool_recycle=7200,
        pool_pre_ping=False
    )
    
    assert db_manager.pool_size == 15, "pool_size not set correctly"
    assert db_manager.max_overflow == 25, "max_overflow not set correctly"
    assert db_manager.pool_timeout == 45, "pool_timeout not set correctly"
    assert db_manager.pool_recycle == 7200, "pool_recycle not set correctly"
    assert db_manager.pool_pre_ping == False, "pool_pre_ping not set correctly"
    assert db_manager._engine is None, "engine should not be initialized yet"
    assert db_manager._session_factory is None, "session_factory should not be initialized yet"
    
    logger.info("✓ DatabaseManager creation test passed")
    return True


def test_engine_access_before_init():
    """Test that accessing engine before initialization raises error."""
    logger.info("Testing engine access before initialization...")
    
    db_manager = DatabaseManager()
    
    try:
        _ = db_manager.engine
        logger.error("Expected RuntimeError for uninitialized engine")
        return False
    except RuntimeError as e:
        logger.info(f"✓ Correctly raised RuntimeError: {e}")
        return True


def test_session_factory_access_before_init():
    """Test that accessing session_factory before initialization raises error."""
    logger.info("Testing session_factory access before initialization...")
    
    db_manager = DatabaseManager()
    
    try:
        _ = db_manager.session_factory
        logger.error("Expected RuntimeError for uninitialized session_factory")
        return False
    except RuntimeError as e:
        logger.info(f"✓ Correctly raised RuntimeError: {e}")
        return True


def test_database_url_construction():
    """Test that database URL is constructed correctly from settings."""
    logger.info("Testing database URL construction...")
    
    db_manager = DatabaseManager()
    
    # The database_url should be constructed from settings
    assert db_manager.database_url is not None, "database_url should not be None"
    assert "postgresql://" in db_manager.database_url, "database_url should be PostgreSQL URL"
    
    logger.info(f"✓ Database URL: {db_manager.database_url}")
    return True


def test_get_db_manager_singleton():
    """Test that get_db_manager returns singleton instance."""
    logger.info("Testing get_db_manager singleton...")
    
    # Reset the global instance
    import services.shared.database as db_module
    db_module._db_manager = None
    
    # Get first instance
    db_manager1 = get_db_manager()
    
    # Get second instance
    db_manager2 = get_db_manager()
    
    # Should be the same instance
    assert db_manager1 is db_manager2, "get_db_manager should return singleton"
    
    logger.info("✓ get_db_manager singleton test passed")
    return True


def test_health_check_without_engine():
    """Test health check behavior when engine is not initialized."""
    logger.info("Testing health check without engine...")
    
    db_manager = DatabaseManager()
    
    # Health check should return False when engine is not initialized
    result = db_manager.health_check()
    
    assert result == False, "health_check should return False without engine"
    
    logger.info("✓ Health check without engine test passed")
    return True


def test_close_without_init():
    """Test that close() can be called safely without initialization."""
    logger.info("Testing close without initialization...")
    
    db_manager = DatabaseManager()
    
    # Should not raise an error
    try:
        db_manager.close()
        logger.info("✓ close() called successfully without initialization")
        return True
    except Exception as e:
        logger.error(f"close() raised unexpected error: {e}")
        return False


def main():
    """Run all unit tests."""
    logger.info("=" * 60)
    logger.info("Starting database connection management unit tests")
    logger.info("(No database connection required)")
    logger.info("=" * 60)
    
    tests = [
        ("DatabaseManager Creation", test_database_manager_creation),
        ("Engine Access Before Init", test_engine_access_before_init),
        ("Session Factory Access Before Init", test_session_factory_access_before_init),
        ("Database URL Construction", test_database_url_construction),
        ("get_db_manager Singleton", test_get_db_manager_singleton),
        ("Health Check Without Engine", test_health_check_without_engine),
        ("Close Without Init", test_close_without_init),
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
