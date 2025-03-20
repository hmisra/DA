"""
Logging module for the DB Agent System.
Provides structured logging functionality for all components.
"""
import os
import sys
from typing import Dict, Any, Optional

from loguru import logger

from src.utils.config import config


def setup_logging() -> None:
    """Set up logging configuration for the application."""
    # Clear default handlers
    logger.remove()

    # Determine log level from config
    log_level = config.LOG_LEVEL

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        colorize=True,
    )

    # Add file handler for all logs
    logger.add(
        "logs/db_agent.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="1 week",
    )

    # Add file handler for errors only
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="1 month",
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging initialized with level: {log_level}")


class AgentLogger:
    """Logger class for agents with contextual information."""

    def __init__(self, agent_name: str, agent_type: str, context: Optional[Dict[str, Any]] = None):
        """Initialize the agent logger.

        Args:
            agent_name: Name of the agent
            agent_type: Type of the agent (e.g., "SQL Developer", "Data Analyst")
            context: Optional contextual information to include in logs
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.context = context or {}
        self.logger = logger.bind(
            agent_name=agent_name,
            agent_type=agent_type,
            **self.context,
        )

    def update_context(self, **kwargs) -> None:
        """Update the logger context with new values.

        Args:
            **kwargs: Key-value pairs to add to the context
        """
        self.context.update(kwargs)
        self.logger = self.logger.bind(**kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message.

        Args:
            message: Message to log
            **kwargs: Additional context for this log message
        """
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message.

        Args:
            message: Message to log
            **kwargs: Additional context for this log message
        """
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message.

        Args:
            message: Message to log
            **kwargs: Additional context for this log message
        """
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log an error message.

        Args:
            message: Message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context for this log message
        """
        self.logger.error(message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log a critical message.

        Args:
            message: Message to log
            exc_info: Whether to include exception info
            **kwargs: Additional context for this log message
        """
        self.logger.critical(message, exc_info=exc_info, **kwargs)


# Setup logging when the module is imported
setup_logging() 