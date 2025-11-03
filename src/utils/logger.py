"""
Logging utilities
"""
from loguru import logger
import sys
from pathlib import Path


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week"
):
    """
    Setup loguru logger

    Args:
        log_file: Path to log file (optional)
        level: Logging level
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    return logger
