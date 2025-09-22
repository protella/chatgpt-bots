"""
Logging module for Slack Bot V2
Provides structured logging with different levels and modules
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import queue
import threading
from typing import Dict, List
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    USE_CONCURRENT_HANDLER = True
except ImportError:
    USE_CONCURRENT_HANDLER = False

from config import config


# Thread-safe singleton for logger initialization
_logger_lock = threading.Lock()
_initialized_loggers: Dict[str, logging.Logger] = {}
_queue_listener: Optional[QueueListener] = None
_log_queue: Optional[queue.Queue] = None

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
    log_file: Optional[str] = None,
    use_queue: bool = True  # Use QueueHandler pattern by default
) -> logging.Logger:
    """
    Set up a logger with specified configuration

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        use_queue: Whether to use QueueHandler for thread-safe logging

    Returns:
        Configured logger instance
    """
    global _initialized_loggers, _queue_listener, _log_queue

    # Thread-safe check for existing logger
    with _logger_lock:
        if name in _initialized_loggers:
            return _initialized_loggers[name]

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

        # Clear any existing handlers to ensure clean setup
        logger.handlers.clear()
    
        # Create logs directory if it doesn't exist
        logs_dir = config.log_directory
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        # File formatter
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if use_queue and _log_queue is None:
            # Initialize the queue and listener once (shared across all loggers)
            _log_queue = queue.Queue(-1)  # Unbounded queue

            # Create the actual file handlers
            handlers = []

            # Console handler with colors (only if enabled)
            if config.console_logging_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = ColoredFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                handlers.append(console_handler)

            # Main app log file (all levels)
            app_log_file = os.path.join(logs_dir, "app.log")
            if USE_CONCURRENT_HANDLER:
                app_handler = ConcurrentRotatingFileHandler(
                    app_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            else:
                # Fallback to regular RotatingFileHandler with warning
                main_logger.warning("ConcurrentRotatingFileHandler not available, using standard RotatingFileHandler")
                app_handler = RotatingFileHandler(
                    app_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            app_handler.setFormatter(file_formatter)
            handlers.append(app_handler)

            # Error log file (ERROR and CRITICAL only)
            error_log_file = os.path.join(logs_dir, "error.log")
            if USE_CONCURRENT_HANDLER:
                error_handler = ConcurrentRotatingFileHandler(
                    error_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            else:
                error_handler = RotatingFileHandler(
                    error_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            handlers.append(error_handler)

            # Start the queue listener with all handlers
            _queue_listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
            _queue_listener.start()

        if use_queue and _log_queue:
            # Add a single QueueHandler to the logger
            queue_handler = QueueHandler(_log_queue)
            logger.addHandler(queue_handler)
        else:
            # Direct handler setup (less thread-safe but simpler)
            # Console handler with colors (only if enabled)
            if config.console_logging_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = ColoredFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

            # Main app log file (all levels)
            app_log_file = os.path.join(logs_dir, "app.log")
            if USE_CONCURRENT_HANDLER:
                app_handler = ConcurrentRotatingFileHandler(
                    app_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            else:
                app_handler = RotatingFileHandler(
                    app_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            app_handler.setFormatter(file_formatter)
            logger.addHandler(app_handler)

            # Error log file (ERROR and CRITICAL only)
            error_log_file = os.path.join(logs_dir, "error.log")
            if USE_CONCURRENT_HANDLER:
                error_handler = ConcurrentRotatingFileHandler(
                    error_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            else:
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

        # Store the logger in our thread-safe cache
        _initialized_loggers[name] = logger

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
    main_logger.info(f"Log Level: {config.log_level}")
    main_logger.info(f"GPT Model: {config.gpt_model}")
    main_logger.info(f"Utility Model: {config.utility_model}")
    main_logger.info(f"Image Model: {config.image_model}")
    main_logger.info("=" * 60)


def log_session_end():
    """Log session end marker"""
    main_logger.info("=" * 60)
    main_logger.info(f"Session ended at {datetime.now().isoformat()}")
    main_logger.info("=" * 60)

    # Stop the queue listener if it exists
    global _queue_listener
    if _queue_listener:
        _queue_listener.stop()
        _queue_listener = None