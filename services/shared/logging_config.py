"""
Structured logging configuration using structlog.

This module provides centralized logging configuration for all services
with structured output, context management, and separate security event logging.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    # Add service name if available from environment
    import os
    service_name = os.getenv("SERVICE_NAME", "unknown")
    event_dict["service"] = service_name
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Censor sensitive data from log entries."""
    # List of keys that might contain sensitive data
    sensitive_keys = ["password", "token", "api_key", "secret", "authorization"]
    
    for key in sensitive_keys:
        if key in event_dict:
            event_dict[key] = "***REDACTED***"
    
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    service_name: Optional[str] = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output logs in JSON format; otherwise use console format
        service_name: Name of the service for context
    """
    # Set service name in environment for context processor
    if service_name:
        import os
        os.environ["SERVICE_NAME"] = service_name
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_app_context,
        censor_sensitive_data,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate renderer based on output format
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def get_security_logger() -> structlog.stdlib.BoundLogger:
    """
    Get a logger specifically for security events.
    
    Security events are logged with a special marker for filtering
    and separate processing.
    
    Returns:
        Configured structlog logger for security events
    """
    logger = structlog.get_logger("security")
    return logger.bind(event_type="security")


# Convenience functions for common logging patterns
def log_operation_start(logger: structlog.stdlib.BoundLogger, operation: str, **context: Any) -> None:
    """Log the start of an operation with context."""
    logger.info(f"{operation} started", operation=operation, **context)


def log_operation_success(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration_ms: Optional[float] = None,
    **context: Any
) -> None:
    """Log successful completion of an operation."""
    log_data = {"operation": operation, **context}
    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms
    logger.info(f"{operation} completed", **log_data)


def log_operation_error(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    error: Exception,
    **context: Any
) -> None:
    """Log an operation error with exception details."""
    logger.error(
        f"{operation} failed",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
        exc_info=True
    )


def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    **context: Any
) -> None:
    """
    Log a security event.
    
    Args:
        event_type: Type of security event (e.g., "input_rejected", "unauthorized_access")
        severity: Severity level (info, warning, error, critical)
        description: Human-readable description of the event
        **context: Additional context data
    """
    security_logger = get_security_logger()
    log_func = getattr(security_logger, severity.lower(), security_logger.info)
    log_func(
        description,
        security_event_type=event_type,
        **context
    )
