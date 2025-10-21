"""
Utility functions for MPC framework.
"""

import logging
import sys
from typing import Any, Optional


# Set up logging
def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Create logger
logger = logging.getLogger('pympc')


def LOG_DEBUG(message: str) -> None:
    """Log debug message."""
    logger.debug(message)


def LOG_INFO(message: str) -> None:
    """Log info message."""
    logger.info(message)


def LOG_WARN(message: str) -> None:
    """Log warning message."""
    logger.warning(message)


def LOG_ERROR(message: str) -> None:
    """Log error message."""
    logger.error(message)


# Initialize logging
setup_logging()
