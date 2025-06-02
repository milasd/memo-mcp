import logging
from typing import Optional

def setup_logging(log_level: int = logging.INFO, name: str = "memo_rag") -> logging.Logger:
    """
    Setup logging configuration for the Memo RAG system.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        name: The name of the logger.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
