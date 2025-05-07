import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Log file path
log_file_path = os.path.join(logs_dir, "deepresearch.log")


def setup_logger(name):
    """
    Configure a logger with both file and console handlers.

    Args:
        name: The name for the logger (typically __name__ from the calling module)

    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Only configure handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Log format
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler (with rotation to prevent huge log files)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
