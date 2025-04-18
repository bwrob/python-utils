"""Logger module for setting up and using a custom logger.

This module provides:
- A logger with both console and file handlers.
- Console handler uses RichHandler for better formatting.
- File handler logs all messages at DEBUG level to a rotating log file.
- Configurable log levels and file paths via environment variables.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# Default configurations
DEFAULT_LOGGER_NAME = "my_logger"
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "app.log")
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))  # 10 MB
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))


def get_logger(
    name: str | None = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: str = LOG_FILE_PATH,
    max_bytes: int = LOG_FILE_MAX_BYTES,
    backup_count: int = LOG_FILE_BACKUP_COUNT,
) -> logging.Logger:
    """Set up and return a logger with console and rotating file handlers."""
    logger = logging.getLogger(name or DEFAULT_LOGGER_NAME)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Console handler with RichHandler
    console_handler = RichHandler(
        level=log_level,
        show_level=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        omit_repeated_times=True,
        log_time_format="[%X]",
    )

    def show_only_debug(record: logging.LogRecord) -> bool:
        record.pathname = record.pathname.replace(str(Path.cwd()), "")
        return True

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_file,
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[{asctime}] [{levelname:8s}] {message} ({pathname} {lineno})",
            style="{",
            datefmt="%x %X",
        ),
    )
    file_handler.addFilter(show_only_debug)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = get_logger()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    try:
        1 / 0  # noqa: B018
    except ZeroDivisionError:
        # Log an exception with traceback
        logger.exception(
            "This is an exception message. It will be logged with traceback.",
        )
    logger.info("This is an info message after exception.")
    logger.debug("This is a debug message after exception.")
