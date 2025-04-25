import logging
import os
from unittest import mock

import pytest

from app.core.logging import setup_logger


@pytest.fixture
def mock_logging_setup():
    """Mock all file operations in the logging module."""
    # Mock Path and mkdir
    with mock.patch('app.core.logging.Path', autospec=True) as mock_path:
        mock_path.return_value.mkdir.return_value = None
        # Mock RotatingFileHandler
        with mock.patch('app.core.logging.RotatingFileHandler') as mock_handler:
            # Return a mock handler that can be added to the logger
            mock_handler.return_value = mock.MagicMock(spec=logging.Handler)
            yield mock_handler


def test_setup_logger_creates_logger(mock_logging_setup):
    """Test that setup_logger creates a logger with the correct name and level."""
    logger = setup_logger("test_logger", level=logging.DEBUG)
    
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


def test_console_logging_enabled(mock_logging_setup):
    """Test that console logging is enabled when CONSOLE_LOGGING_ENABLED is set."""
    with mock.patch.dict(os.environ, {"CONSOLE_LOGGING_ENABLED": "true"}):
        with mock.patch('app.core.logging.logging.StreamHandler', autospec=True) as mock_stream:
            mock_stream.return_value = mock.MagicMock(spec=logging.Handler)
            logger = setup_logger("test_logger")
            
            # Verify StreamHandler was created
            assert mock_stream.called


def test_console_logging_disabled(mock_logging_setup):
    """Test that console logging can be disabled with environment variable."""
    with mock.patch.dict(os.environ, {"CONSOLE_LOGGING_ENABLED": "false"}):
        with mock.patch('app.core.logging.logging.StreamHandler', autospec=True) as mock_stream:
            logger = setup_logger("test_logger")
            
            # Verify StreamHandler was not created
            assert not mock_stream.called


def test_logger_file_handler_config(mock_logging_setup):
    """Test that file handlers are configured with correct parameters."""
    mock_handler = mock_logging_setup
    setup_logger("test_logger")
    
    # Check that handler was created at least twice
    assert mock_handler.call_count >= 2
    
    # Check the parameters
    for call in mock_handler.call_args_list:
        # Ensure maxBytes and backupCount are correct
        assert call[1].get('maxBytes') == 10 * 1024 * 1024  # 10 MB
        assert call[1].get('backupCount') == 5 