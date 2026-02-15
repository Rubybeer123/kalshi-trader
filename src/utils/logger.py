"""Structured logging configuration."""

import logging
import sys
from typing import Any, Dict

import structlog


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: str = None
) -> None:
    """
    Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format (json or text)
        log_file: Optional file path for logging
    """
    # Configure standard library logging
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers
    )
    
    # Configure structlog
    shared_processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
    ]
    
    if format == "json":
        format_processor = structlog.processors.JSONRenderer()
    else:
        format_processor = structlog.dev.ConsoleRenderer()
    
    structlog.configure(
        processors=shared_processors + [format_processor],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Pre-configured loggers for common use cases
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)
