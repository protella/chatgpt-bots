import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Unset any existing log level environment variables to ensure .env values are used
if "CONSOLE_LOGGING_ENABLED" in os.environ:
    del os.environ["CONSOLE_LOGGING_ENABLED"]

# Load environment variables
load_dotenv()

# Define log level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Environment variables for logging:
# - Log levels: SLACK_LOG_LEVEL, DISCORD_LOG_LEVEL, BOT_LOG_LEVEL, UTILS_LOG_LEVEL
# - Console toggle: CONSOLE_LOGGING_ENABLED (TRUE/FALSE)


def get_log_level(level_name, default=logging.INFO):
    """
    Convert a string log level name to its numeric value.
    
    Args:
        level_name (str): The name of the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        default (int): The default log level to use if level_name is invalid.
        
    Returns:
        int: The numeric log level value.
    """
    return LOG_LEVELS.get(level_name.upper(), default)

# Configure logging
def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name (str): The name of the logger.
        log_level (int): The logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: The configured logger.
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Set the logger's level
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Check if console logging is enabled (default to TRUE if not specified)
    console_logging_enabled = os.environ.get("CONSOLE_LOGGING_ENABLED", "TRUE").upper() == "TRUE"
    
    # Create console handler if enabled
    if console_logging_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Create file handlers
    general_file_handler = RotatingFileHandler(
        'logs/app.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    general_file_handler.setLevel(log_level)
    general_file_handler.setFormatter(file_formatter)
    
    error_file_handler = RotatingFileHandler(
        'logs/error.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    
    # Add file handlers to logger
    logger.addHandler(general_file_handler)
    logger.addHandler(error_file_handler)
    
    # Ensure propagation is set to False to prevent duplicate logs
    logger.propagate = False
    
    return logger

def log_session_marker(logger, marker_type="START"):
    """
    Log a session marker to clearly separate different bot sessions in the logs.
    
    Args:
        logger (logging.Logger): The logger to use.
        marker_type (str): The type of marker (START or END).
    """
    separator = "="*80
    if marker_type.upper() == "START":
        logger.info(f"\n{separator}\n{logger.name.upper()} SERVICE STARTING - NEW SESSION\n{separator}")
    elif marker_type.upper() == "END":
        logger.info(f"\n{separator}\n{logger.name.upper()} SERVICE SHUTDOWN\n{separator}")
    else:
        logger.info(f"\n{separator}\n{marker_type}\n{separator}")

# Initialize logger references but don't create them yet
app_logger = None
slack_logger = None
discord_logger = None
bot_logger = None
utils_logger = None

def get_logger(name, log_level=None):
    """
    Get a logger by name, creating it if it doesn't exist.
    
    Args:
        name (str): The name of the logger.
        log_level (int, optional): The log level to use. If None, uses the existing logger's level.
        
    Returns:
        logging.Logger: The requested logger.
    """
    global app_logger, slack_logger, discord_logger, bot_logger, utils_logger
    
    # Check if we're requesting one of our predefined loggers
    if name == 'app' and app_logger is not None:
        return app_logger
    elif name == 'slack' and slack_logger is not None:
        return slack_logger
    elif name == 'discord' and discord_logger is not None:
        return discord_logger
    elif name == 'bot' and bot_logger is not None:
        return bot_logger
    elif name == 'utils' and utils_logger is not None:
        return utils_logger
    
    # If the logger doesn't exist or we're not requesting a predefined one,
    # create it with the specified log level or get the existing one
    logger = logging.getLogger(name)
    
    # If a log level was specified, set up the logger with that level
    if log_level is not None:
        logger = setup_logger(name, log_level)
        
        # Update our references if this is one of our predefined loggers
        if name == 'app':
            app_logger = logger
        elif name == 'slack':
            slack_logger = logger
        elif name == 'discord':
            discord_logger = logger
        elif name == 'bot':
            bot_logger = logger
        elif name == 'utils':
            utils_logger = logger
    
    return logger 