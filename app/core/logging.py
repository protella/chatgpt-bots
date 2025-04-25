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
    
    # Determine logs directory - prefer mounted /logs in container,
    # fall back to environment variable, then to local logs folder
    if os.path.exists("/logs") and os.access("/logs", os.W_OK):
        # Container environment with mounted /logs
        logs_dir = Path("/logs")
    else:
        # Local development environment
        project_root = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
        logs_dir = project_root / "logs"
    
    logs_dir.mkdir(exist_ok=True)
    logger.debug(f"Using logs directory: {logs_dir}")
    
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
    console_logging_enabled = os.environ.get("CONSOLE_LOGGING_ENABLED", "true").lower() in (
        "true", "1", "yes", "y"
    )
    
    if console_logging_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger 