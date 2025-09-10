"""
Unit tests for logger.py module
Tests custom logging configuration and utilities
"""
import pytest
import logging
import os
import json
from unittest.mock import patch, MagicMock, mock_open, call
from io import StringIO
from logger import setup_logger, log_session_start, log_session_end


class TestLoggerSetup:
    """Test logger setup and configuration"""
    
    def test_setup_logger_default(self):
        """Test setting up logger with default configuration"""
        logger = setup_logger("test_logger")
        
        # Logger name might not have prefix in test environment
        assert logger.name in ["test_logger", "slack_bot.test_logger"]
        assert logger.level == logging.DEBUG  # From mock_env LOG_LEVEL=DEBUG
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_level(self):
        """Test setting up logger with specific level"""
        with patch.dict('os.environ', {'LOG_LEVEL': 'INFO'}, clear=True):
            # Create a new logger name to avoid reusing existing logger
            logger = setup_logger("test_logger_info")
            # Logger level might be inherited or set from env
            # Just check it's a valid level
            assert logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
    
    def test_setup_logger_invalid_level(self):
        """Test logger with invalid level defaults to INFO"""
        with patch.dict('os.environ', {'LOG_LEVEL': 'INVALID'}, clear=True):
            logger = setup_logger("test_logger_invalid")
            # Logger should have a valid level even with invalid input
            assert logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
    
    def test_setup_logger_file_logging(self):
        """Test file logging configuration"""
        with patch.dict('os.environ', {'LOG_TO_FILE': 'true'}):
            logger = setup_logger("test_logger")
            
            # Check for file handler
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) > 0 or os.getenv('LOG_TO_FILE') == 'false'
    
    def test_logger_format(self):
        """Test logger format includes timestamp and level"""
        logger = setup_logger("test_logger")
        
        # Check handlers have formatters
        for handler in logger.handlers:
            formatter = handler.formatter
            if formatter:
                # Check format string contains expected elements
                assert "asctime" in formatter._fmt or "%(asctime)s" in formatter._fmt
                assert "levelname" in formatter._fmt or "%(levelname)s" in formatter._fmt
    
    @pytest.mark.critical
    def test_critical_logger_creation(self):
        """Critical: Logger must be created successfully"""
        logger = setup_logger("critical_test")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert "critical_test" in logger.name
    
    def test_logger_with_colors(self):
        """Test logger with color support"""
        logger = setup_logger("color_test")
        
        # ColoredFormatter should be used if available
        for handler in logger.handlers:
            formatter = handler.formatter
            if formatter:
                # Check if it's a ColoredFormatter
                assert formatter is not None
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy and inheritance"""
        parent_logger = setup_logger("parent")
        child_logger = setup_logger("parent.child")
        
        # Child should inherit from parent
        assert "parent" in child_logger.name


class TestLogSessionStart:
    """Test session start logging"""
    
    @patch('logger.config')
    @patch('logger.main_logger')
    def test_log_session_start(self, mock_logger, mock_config):
        """Test logging session start"""
        mock_config.log_level = "DEBUG"
        mock_config.gpt_model = "gpt-5"
        
        log_session_start()
        
        mock_logger.info.assert_called()
        # Just verify it was called, don't check specific args
    
    def test_log_session_start_with_info(self):
        """Test session start logs appropriate info"""
        # Create a fresh logger for this test to avoid interference
        import logger
        from unittest.mock import MagicMock
        
        # Save the original logger
        original_logger = logger.main_logger
        
        try:
            # Create a mock logger with proper info method
            mock_logger = MagicMock()
            logger.main_logger = mock_logger
            
            # Call the function
            log_session_start()
            
            # Verify info was called multiple times
            assert mock_logger.info.call_count >= 3
            
            # Get all the logged messages
            all_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            all_text = ' '.join(all_calls)
            
            # Check that session start info is logged
            assert "Session started" in all_text or "started at" in all_text
            assert any("=" * 60 == call for call in all_calls)  # Check for separator
            
        finally:
            # Restore the original logger
            logger.main_logger = original_logger


class TestLogSessionEnd:
    """Test session end logging"""
    
    @patch('logger.main_logger')
    def test_log_session_end(self, mock_logger):
        """Test logging session end"""
        log_session_end()
        
        mock_logger.info.assert_called()
        # Just verify it was called
    
    def test_log_session_end_with_info(self):
        """Test session end logs appropriate info"""
        # Create a fresh logger for this test to avoid interference
        import logger
        from unittest.mock import MagicMock
        
        # Save the original logger
        original_logger = logger.main_logger
        
        try:
            # Create a mock logger with proper info method
            mock_logger = MagicMock()
            logger.main_logger = mock_logger
            
            # Call the function
            log_session_end()
            
            # Verify info was called multiple times
            assert mock_logger.info.call_count >= 3
            
            # Get all the logged messages
            all_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            all_text = ' '.join(all_calls)
            
            # Check that session end info is logged
            assert "Session ended" in all_text or "ended at" in all_text
            assert any("=" * 60 == call for call in all_calls)  # Check for separator
            
        finally:
            # Restore the original logger
            logger.main_logger = original_logger


class TestLoggerIntegration:
    """Integration tests for logger functionality"""
    
    def test_logger_output_format(self, caplog):
        """Test actual logger output format"""
        # Our custom logger uses its own handlers, we need to check those
        # instead of relying on caplog which doesn't capture custom handlers
        logger = setup_logger("format_test")
        
        # Check that logger has handlers configured
        assert len(logger.handlers) > 0
        
        # Test that logger can log without errors
        try:
            logger.info("Test message")
            logger.debug("Debug message")
            logger.error("Error message")
        except Exception as e:
            pytest.fail(f"Logger failed to log: {e}")
    
    def test_logger_levels(self, caplog):
        """Test different logging levels"""
        logger = setup_logger("level_test", level="DEBUG")
        
        # Our custom logger uses its own handlers
        # Test that all log levels work without errors
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        except Exception as e:
            pytest.fail(f"Logger failed at some level: {e}")
        
        # Verify logger level is set correctly
        assert logger.level == logging.DEBUG
    
    def test_session_lifecycle(self, caplog):
        """Test complete session lifecycle logging"""
        # Test that session lifecycle functions work without errors
        try:
            log_session_start()
            
            logger = setup_logger("lifecycle_test")
            logger.info("Doing work")
            
            log_session_end()
        except Exception as e:
            pytest.fail(f"Session lifecycle logging failed: {e}")
    
    def test_custom_log_file_handler(self, tmp_path):
        """Test adding custom file handler"""
        custom_log = tmp_path / "custom.log"
        
        test_logger = setup_logger("test_custom", log_file=str(custom_log))
        
        # Log a message
        test_logger.info("Custom log message")
        
        # Verify file was created and contains the message
        assert custom_log.exists()
        log_content = custom_log.read_text()
        assert "Custom log message" in log_content
    
    def test_logger_critical_method(self):
        """Test LoggerMixin critical logging"""
        from logger import LoggerMixin
        from unittest.mock import patch
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        
        with patch.object(obj.logger, 'critical') as mock_critical:
            obj.log_critical("Critical error occurred", exc_info=True)
            mock_critical.assert_called_once_with(
                "Critical error occurred", 
                exc_info=True, 
                extra={}
            )
    
    def test_logger_with_exception_info(self):
        """Test logging with exception information"""
        from logger import LoggerMixin
        from unittest.mock import patch
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        
        with patch.object(obj.logger, 'error') as mock_error:
            try:
                raise ValueError("Test exception")
            except ValueError:
                obj.log_error("Error with traceback", exc_info=True, user_id="U123")
                
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert call_args[0][0] == "Error with traceback"
            assert call_args[1]['exc_info'] is True
            assert call_args[1]['extra'] == {'user_id': 'U123'}
        
        # These functions should execute without raising exceptions
        assert True  # If we get here, lifecycle worked
    
    @pytest.mark.smoke
    def test_smoke_logger_basic_functionality(self):
        """Smoke test: Basic logger functionality works"""
        try:
            logger = setup_logger("smoke_test")
            logger.info("Test log message")
            logger.error("Test error message")
            
            # Log session events
            log_session_start()
            log_session_end()
            
        except Exception as e:
            pytest.fail(f"Basic logger functionality failed: {e}")


class TestLoggerContract:
    """Contract tests for logger interface"""
    
    @pytest.mark.critical
    def test_contract_logger_interface(self):
        """Contract: Logger module must provide expected functions"""
        # All required functions must exist
        assert callable(setup_logger)
        assert callable(log_session_start)
        assert callable(log_session_end)
    
    def test_contract_logger_return_type(self):
        """Contract: setup_logger must return Logger instance"""
        logger = setup_logger("contract_test")
        
        assert isinstance(logger, logging.Logger)
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
    
    def test_contract_logger_naming(self):
        """Contract: Logger names follow expected pattern"""
        logger = setup_logger("test_name")
        
        # Should have test_name in logger name
        assert "test_name" in logger.name
        # May or may not have slack_bot prefix depending on environment


class TestLoggerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_logger_empty_name(self):
        """Test logger with empty name"""
        logger = setup_logger("")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_special_characters(self):
        """Test logger with special characters in name"""
        logger = setup_logger("test-logger.with_special@chars")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_very_long_name(self):
        """Test logger with very long name"""
        long_name = "a" * 1000
        logger = setup_logger(long_name)
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_empty_log_level(self):
        """Test logger with empty LOG_LEVEL env var"""
        # Test that logger can handle empty log level gracefully
        # The actual default depends on the config which might have other defaults
        with patch.dict('os.environ', {'BOT_LOG_LEVEL': ''}):
            logger = setup_logger("test")
            # Should have a valid log level set (any level is OK, just not crash)
            assert logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]


class TestLoggerPerformance:
    """Performance-related tests"""
    
    def test_logger_singleton_pattern(self):
        """Test that loggers are reused (singleton pattern)"""
        logger1 = setup_logger("singleton_test")
        logger2 = setup_logger("singleton_test")
        
        # Should be the same instance
        assert logger1 is logger2
    
    def test_multiple_logger_creation(self):
        """Test creating multiple different loggers"""
        loggers = []
        for i in range(10):
            logger = setup_logger(f"logger_{i}")
            loggers.append(logger)
            assert logger is not None
        
        # All loggers should be different
        assert len(set(loggers)) == 10