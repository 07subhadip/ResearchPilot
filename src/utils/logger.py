"""
Centralized logging setup using loguru.

WHY LOGURU over Python's built-in logging:
- Built-in logging requires 10+ lines of boilerplate to set up properly
- Loguru does it in 3 lines
- Loguru auto-formats with colors, timestamps, and file/line info
- Loguru handles file rotation automatically
- Every serious Python project at companies like Stripe uses structured logging
"""

import sys
from loguru import logger
from config.settings import LOG_LEVEL, LOG_FILE, LOG_ROTATION, LOG_RETENTION


def setup_logger():
    """
    Configure loguru logger with both console and file output.

    Return the configured logger instance
    Call this once at application startup
    """

    # Remove the default logger handler
    # (it only prints to console with basic formatting)
    logger.remove()

    # ----------- Console Handler -----------
    # Prints colored, formatted logs to terminal
    # Format: 2024-01-15 10:23:45 | INFO | module:function:42 | Message
    logger.add(
        sys.stdout,
        level = LOG_LEVEL,
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize = True
    )

    # ----------- File Handler -----------
    # Writes all logs to file for debugging and monitoring
    # rotation="10 MB"      -> creates new file when current reaches 10MB
    # retention="7 days"    -> deletes log files oder than 7 days
    logger.add(
        LOG_FILE,
        level = 'DEBUG',
        format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation = LOG_ROTATION,
        retention = LOG_RETENTION,
        encoding = 'utf-8'
    ) 

    return logger



# Create the logger instance
# Other modules import this directly:
#   from src.utils.logger import get_logger
#   logger = get_logger(__name__)
def get_logger(name: str):
    """
    Get a named logger instance.
    The name appears in log output so you know which module logged what

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting ingestion...") 
    """
    return logger.bind(name = name)

# Initialize logger when this module is first imported
# setup_logger()