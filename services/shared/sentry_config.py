"""
Sentry error tracking configuration.

This module provides centralized Sentry integration for error tracking
across all services with proper context and user information.
"""

import os
from typing import Optional, Dict, Any
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


def configure_sentry(
    service_name: str,
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
) -> None:
    """
    Configure Sentry SDK for error tracking.
    
    Args:
        service_name: Name of the service for tagging
        dsn: Sentry DSN (Data Source Name). If None, reads from SENTRY_DSN env var
        environment: Deployment environment (development, staging, production)
        release: Release version for tracking
        traces_sample_rate: Percentage of transactions to trace (0.0 to 1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
    """
    # Get DSN from parameter or environment
    sentry_dsn = dsn or os.getenv("SENTRY_DSN")
    
    # Skip initialization if no DSN is provided
    if not sentry_dsn:
        return
    
    # Get environment from parameter or environment variable
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    # Get release from parameter or environment variable
    rel = release or os.getenv("RELEASE", "unknown")
    
    # Configure logging integration
    logging_integration = LoggingIntegration(
        level=None,  # Capture all logs
        event_level=None  # Don't send logs as events (we'll use explicit capture)
    )
    
    # Configure integrations list
    integrations = [logging_integration]
    
    # Add SQLAlchemy integration only if sqlalchemy is installed
    try:
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        integrations.append(SqlalchemyIntegration())
    except ImportError:
        pass
    
    # Initialize Sentry
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=env,
        release=rel,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        integrations=integrations,
        # Set service name as a tag
        before_send=lambda event, hint: _add_service_context(event, service_name),
        # Send default PII (we'll control this per-event)
        send_default_pii=False,
    )


def _add_service_context(event: Dict[str, Any], service_name: str) -> Dict[str, Any]:
    """Add service context to all Sentry events."""
    if 'tags' not in event:
        event['tags'] = {}
    event['tags']['service'] = service_name
    return event


def set_user_context(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
    **extra: Any
) -> None:
    """
    Set user context for error tracking.
    
    Args:
        user_id: Unique user identifier
        username: Username
        email: User email (will be redacted if send_default_pii is False)
        **extra: Additional user context
    """
    user_data = {}
    
    if user_id:
        user_data['id'] = user_id
    if username:
        user_data['username'] = username
    if email:
        user_data['email'] = email
    
    user_data.update(extra)
    
    sentry_sdk.set_user(user_data)


def set_context(context_name: str, context_data: Dict[str, Any]) -> None:
    """
    Set additional context for error tracking.
    
    Args:
        context_name: Name of the context (e.g., 'query', 'document', 'conversation')
        context_data: Dictionary of context data
    """
    sentry_sdk.set_context(context_name, context_data)


def add_breadcrumb(
    message: str,
    category: Optional[str] = None,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a breadcrumb for error tracking.
    
    Breadcrumbs are a trail of events that led to an error.
    
    Args:
        message: Breadcrumb message
        category: Category of the breadcrumb (e.g., 'query', 'database', 'llm')
        level: Severity level (debug, info, warning, error, critical)
        data: Additional data
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


def capture_exception(
    error: Exception,
    level: str = "error",
    tags: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture an exception and send to Sentry.
    
    Args:
        error: Exception to capture
        level: Severity level (debug, info, warning, error, critical)
        tags: Additional tags for filtering
        extra: Additional context data
    
    Returns:
        Event ID if sent to Sentry, None otherwise
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level
        
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)
        
        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)
        
        return sentry_sdk.capture_exception(error)


def capture_message(
    message: str,
    level: str = "info",
    tags: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture a message and send to Sentry.
    
    Args:
        message: Message to capture
        level: Severity level (debug, info, warning, error, critical)
        tags: Additional tags for filtering
        extra: Additional context data
    
    Returns:
        Event ID if sent to Sentry, None otherwise
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level
        
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)
        
        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)
        
        return sentry_sdk.capture_message(message)


# Context managers for operation tracking
class SentryOperationContext:
    """Context manager for tracking operations in Sentry."""
    
    def __init__(
        self,
        operation_name: str,
        operation_type: str = "task",
        description: Optional[str] = None
    ):
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.description = description
        self.transaction = None
    
    def __enter__(self):
        # Start a transaction for performance monitoring
        self.transaction = sentry_sdk.start_transaction(
            op=self.operation_type,
            name=self.operation_name,
            description=self.description
        )
        self.transaction.__enter__()
        
        # Add breadcrumb
        add_breadcrumb(
            message=f"Started {self.operation_name}",
            category=self.operation_type,
            level="info"
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Operation failed
            add_breadcrumb(
                message=f"Failed {self.operation_name}: {exc_val}",
                category=self.operation_type,
                level="error"
            )
        else:
            # Operation succeeded
            add_breadcrumb(
                message=f"Completed {self.operation_name}",
                category=self.operation_type,
                level="info"
            )
        
        if self.transaction:
            self.transaction.__exit__(exc_type, exc_val, exc_tb)
        
        return False  # Don't suppress exceptions


def track_operation(operation_name: str, operation_type: str = "task", description: Optional[str] = None):
    """
    Decorator or context manager for tracking operations.
    
    Usage as context manager:
        with track_operation("process_query", "query"):
            # operation code
            pass
    
    Usage as decorator:
        @track_operation("process_query", "query")
        def process_query():
            # operation code
            pass
    """
    return SentryOperationContext(operation_name, operation_type, description)
