"""Database connection management and session handling.

This module provides SQLAlchemy-based database connection management with:
- Connection pool configuration
- Session factory for creating database sessions
- Context managers for automatic transaction handling
- Health check functions to verify database connectivity
- Proper error handling and connection retry logic

Validates Requirement 16.2: Wait for PostgreSQL health checks before initializing services
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection pool and session manager.
    
    Provides connection pool management with proper configuration,
    session factory for creating database sessions, and health check
    functions to verify database connectivity.
    
    Validates Requirement 16.2: Health checks before service initialization
    """
    
    def __init__(
        self,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
    ):
        """Initialize database manager with connection pool configuration.
        
        Args:
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_timeout: Seconds to wait before giving up on getting a connection
            pool_recycle: Seconds after which to recycle connections
            pool_pre_ping: Enable connection health checks before using
        """
        settings = get_settings()
        self.database_url = settings.database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    def initialize(self) -> None:
        """Create database engine and session factory.
        
        Initializes the SQLAlchemy engine with connection pooling and
        creates a session factory for generating database sessions.
        
        Raises:
            OperationalError: If database connection cannot be established
        """
        if self._engine is None:
            try:
                # Create engine with connection pool
                self._engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_timeout=self.pool_timeout,
                    pool_recycle=self.pool_recycle,
                    pool_pre_ping=self.pool_pre_ping,
                    echo=False,  # Set to True for SQL query logging
                )
                
                # Add connection event listeners for logging
                event.listen(self._engine, "connect", self._on_connect)
                event.listen(self._engine, "checkout", self._on_checkout)
                
                # Create session factory
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,
                )
                
                logger.info(
                    f"Database engine initialized with pool_size={self.pool_size}, "
                    f"max_overflow={self.max_overflow}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize database engine: {e}")
                raise
    
    def _on_connect(self, dbapi_conn, connection_record):
        """Event handler called when a new connection is created."""
        logger.debug("New database connection created")
    
    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Event handler called when a connection is retrieved from the pool."""
        logger.debug("Connection checked out from pool")
    
    def close(self) -> None:
        """Close database engine and dispose of connection pool.
        
        Closes all connections in the pool and disposes of the engine.
        Should be called during application shutdown.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed and connection pool closed")
            self._engine = None
            self._session_factory = None
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine.
        
        Returns:
            SQLAlchemy Engine instance
            
        Raises:
            RuntimeError: If engine is not initialized
        """
        if self._engine is None:
            raise RuntimeError("Database engine not initialized. Call initialize() first.")
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory.
        
        Returns:
            SQLAlchemy sessionmaker instance
            
        Raises:
            RuntimeError: If session factory is not initialized
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized. Call initialize() first.")
        return self._session_factory
    
    def create_session(self) -> Session:
        """Create a new database session.
        
        Returns:
            SQLAlchemy Session instance
            
        Example:
            session = db_manager.create_session()
            try:
                # Use session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        """
        return self.session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope for database operations.
        
        Context manager that automatically handles session lifecycle:
        - Creates a new session
        - Commits on success
        - Rolls back on exception
        - Closes session in all cases
        
        Yields:
            SQLAlchemy Session instance
            
        Example:
            with db_manager.session_scope() as session:
                document = session.query(Document).first()
                document.title = "Updated Title"
                # Automatically commits on exit
        """
        session = self.create_session()
        try:
            yield session
            session.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            session.close()
            logger.debug("Session closed")
    
    def health_check(self, timeout: int = 5) -> bool:
        """Check database connectivity and health.
        
        Performs a simple query to verify the database is accessible
        and responding to queries.
        
        Args:
            timeout: Maximum seconds to wait for response
            
        Returns:
            True if database is healthy, False otherwise
            
        Example:
            if db_manager.health_check():
                print("Database is healthy")
            else:
                print("Database is not responding")
        """
        try:
            if self._engine is None:
                logger.warning("Cannot perform health check: engine not initialized")
                return False
            
            # Execute simple query with timeout
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.debug("Database health check passed")
            return True
        except OperationalError as e:
            logger.error(f"Database health check failed (operational error): {e}")
            return False
        except Exception as e:
            logger.error(f"Database health check failed (unexpected error): {e}")
            return False
    
    def wait_for_db(
        self,
        max_retries: int = 30,
        retry_interval: int = 2,
        timeout: int = 5
    ) -> bool:
        """Wait for database to become available.
        
        Repeatedly attempts to connect to the database until successful
        or maximum retries reached. Useful for waiting for PostgreSQL
        to start during application initialization.
        
        Validates Requirement 16.2: Wait for PostgreSQL health checks
        before initializing dependent services
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_interval: Seconds to wait between attempts
            timeout: Seconds to wait for each connection attempt
            
        Returns:
            True if database becomes available, False if max retries exceeded
            
        Example:
            if db_manager.wait_for_db(max_retries=30, retry_interval=2):
                print("Database is ready")
                db_manager.initialize()
            else:
                print("Database failed to become available")
                sys.exit(1)
        """
        logger.info(
            f"Waiting for database to become available "
            f"(max_retries={max_retries}, retry_interval={retry_interval}s)"
        )
        
        for attempt in range(1, max_retries + 1):
            try:
                # Try to initialize engine if not already done
                if self._engine is None:
                    self.initialize()
                
                # Perform health check
                if self.health_check(timeout=timeout):
                    logger.info(f"Database is available (attempt {attempt}/{max_retries})")
                    return True
                
            except OperationalError as e:
                logger.warning(
                    f"Database not available yet (attempt {attempt}/{max_retries}): {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error while waiting for database "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
            
            # Wait before next attempt (except on last attempt)
            if attempt < max_retries:
                time.sleep(retry_interval)
        
        logger.error(f"Database did not become available after {max_retries} attempts")
        return False
    
    def execute_raw_sql(self, sql: str, params: Optional[dict] = None) -> list:
        """Execute raw SQL query and return results.
        
        Useful for operations that don't fit well into ORM patterns,
        such as custom vector similarity searches.
        
        Args:
            sql: SQL query string (use :param for parameters)
            params: Dictionary of parameter values
            
        Returns:
            List of result rows as dictionaries
            
        Example:
            results = db_manager.execute_raw_sql(
                "SELECT * FROM documents WHERE title = :title",
                {"title": "My Document"}
            )
        """
        with self.session_scope() as session:
            result = session.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance.
    
    Returns:
        DatabaseManager singleton instance
        
    Example:
        db_manager = get_db_manager()
        db_manager.initialize()
        
        with db_manager.session_scope() as session:
            documents = session.query(Document).all()
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_db(wait_for_ready: bool = True) -> DatabaseManager:
    """Initialize database connection and wait for availability.
    
    Convenience function that creates database manager, waits for
    database to become available, and initializes the connection pool.
    
    Validates Requirement 16.2: Wait for PostgreSQL health checks
    before initializing dependent services
    
    Args:
        wait_for_ready: Whether to wait for database to become available
        
    Returns:
        Initialized DatabaseManager instance
        
    Raises:
        RuntimeError: If database does not become available
        
    Example:
        # In service startup code
        db_manager = init_db(wait_for_ready=True)
        # Now safe to use database
    """
    db_manager = get_db_manager()
    
    if wait_for_ready:
        if not db_manager.wait_for_db():
            raise RuntimeError("Database failed to become available")
    
    db_manager.initialize()
    return db_manager


# Convenience function for getting a session scope
def get_session() -> Generator[Session, None, None]:
    """Get a database session with automatic transaction management.
    
    Convenience function that wraps get_db_manager().session_scope()
    for easier imports and usage.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        from services.shared.database import get_session
        
        with get_session() as session:
            document = Document(title="Test", content="Content")
            session.add(document)
            # Automatically commits on exit
    """
    db_manager = get_db_manager()
    with db_manager.session_scope() as session:
        yield session
