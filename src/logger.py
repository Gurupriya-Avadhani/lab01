# #!/usr/bin/env python3
# """
# Unified logging setup for MLOps pipeline stages.
# """
# import logging
# import sys
# from datetime import datetime
# class ColorFormatter(logging.Formatter):
# """Custom formatter with color for terminal output."""
# COLORS = {
# 'DEBUG': '\033[36m', # Cyan
# 'INFO': '\033[32m', # Green
# 'WARNING': '\033[33m', # Yellow
# 'ERROR': '\033[31m', # Red
# 'CRITICAL': '\033[35m', # Magenta
# }
# RESET = '\033[0m'
# def format(self, record):
# color = self.COLORS.get(record.levelname, '')
# record.levelname = f"{color}{record.levelname}{self.RESET}"
# return super().format(record)
# def setup_logger(name, level=logging.INFO):
# """
# Configure and return a logger.
# Args:
# name: Logger name (typically __name__)
# level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# Returns:
# Configured logger instance
# """
# logger = logging.getLogger(name)
# logger.setLevel(level)
# # Remove existing handlers (prevent duplicates)
# logger.handlers = []
# # Console handler with color
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(level)
# formatter = ColorFormatter(
# fmt='[%(levelname)s] %(name)s - %(message)s',
# datefmt='%Y-%m-%d %H:%M:%S'
# )
# ch.setFormatter(formatter)
# logger.addHandler(ch)
# return logger
# def configure_pipeline_logging():
# """Configure logging for entire pipeline."""
# # Suppress verbose library logs
# logging.getLogger('urllib3').setLevel(logging.WARNING)
# logging.getLogger('requests').setLevel(logging.WARNING)
# return setup_logger('MLOps.Pipeline', level=logging.INFO)
# # Example usage
# if __name__ == '__main__':
# logger = setup_logger(__name__)
# logger.debug("Debug message")
# logger.info("Pipeline started")
# logger.warning("This is a warning")
# logger.error("An error occurred")

#!/usr/bin/env python3
"""
Unified logging setup for MLOps pipeline stages.
"""

import logging
import sys
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Custom formatter with color for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        levelname = f"{color}{record.levelname}{self.RESET}"
        record.levelname = levelname
        return super().format(record)


def setup_logger(name, level=logging.INFO):
    """
    Configure and return a logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers (prevent duplicates)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with color
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    formatter = ColorFormatter(
        fmt='[%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.propagate = False  # Prevent double logging

    return logger


def configure_pipeline_logging():
    """Configure logging for entire pipeline."""
    # Suppress verbose library logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return setup_logger('MLOps.Pipeline', level=logging.INFO)


# Example usage
if __name__ == '__main__':
    logger = setup_logger(__name__, level=logging.DEBUG)

    logger.debug("Debug message")
    logger.info("Pipeline started")
    logger.warning("This is a warning")
    logger.error("An error occurred")