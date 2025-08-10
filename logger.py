"""
Logging module for Slack Bot V2
Provides structured logging with different levels and modules
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
from config import config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "slack_bot",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with specified configuration
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Use config level if not specified, check for platform-specific levels
    if level is None:
        if "slack" in name.lower():
            level = config.slack_log_level
        elif "discord" in name.lower():
            level = config.discord_log_level
        elif "utils" in name.lower() or "openai" in name.lower():
            level = config.utils_log_level
        else:
            level = config.log_level
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    # Return early if logger already has handlers (avoid duplicates)
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    logs_dir = config.log_directory
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Console handler with colors (only if enabled)
    if config.console_logging_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handlers - always create these
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Main app log file (all levels)
    app_log_file = os.path.join(logs_dir, "app.log")
    app_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    app_handler.setFormatter(file_formatter)
    logger.addHandler(app_handler)
    
    # Error log file (ERROR and CRITICAL only)
    error_log_file = os.path.join(logs_dir, "error.log")
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    # Custom file handler if specified (in addition to defaults)
    if log_file:
        custom_handler = logging.FileHandler(log_file)
        custom_handler.setFormatter(file_formatter)
        logger.addHandler(custom_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to provide logging capabilities to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create a logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = setup_logger(
                name=f"slack_bot.{self.__class__.__name__}"
            )
        return self._logger
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def log_error(self, message: str, exc_info=False, **kwargs):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def log_critical(self, message: str, exc_info=False, **kwargs):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)


# Main logger instance
main_logger = setup_logger("slack_bot")


def log_session_start():
    """Log session start marker"""
    main_logger.info("=" * 60)
    main_logger.info(f"Session started at {datetime.now().isoformat()}")
    main_logger.info(f"Debug mode: {config.debug_mode}")
    main_logger.info(f"GPT Model: {config.gpt_model}")
    main_logger.info(f"Utility Model: {config.utility_model}")
    main_logger.info(f"Image Model: {config.image_model}")
    main_logger.info(f"Log Level: {config.log_level}")
    main_logger.info("=" * 60)


def log_session_end():
    """Log session end marker"""
    main_logger.info("=" * 60)
    main_logger.info(f"Session ended at {datetime.now().isoformat()}")
    main_logger.info("=" * 60)