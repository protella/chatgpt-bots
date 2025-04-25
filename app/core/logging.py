import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with rotating file handlers and optional console output.
    
    Args:
        name: The name of the logger
        level: The logging level (default: logging.INFO)
        
    Returns:
        A configured logging.Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    logs_dir.mkdir(exist_ok=True)
    
    # Common log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Regular log file handler (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        str(logs_dir / "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error log file handler (separate file for errors)
    error_handler = RotatingFileHandler(
        str(logs_dir / "error.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Console handler (optional, controlled by CONSOLE_LOGGING_ENABLED)
    console_logging_enabled = os.environ.get("CONSOLE_LOGGING_ENABLED", "false").lower() in (
        "true", "1", "yes", "y"
    )
    
    if console_logging_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger 