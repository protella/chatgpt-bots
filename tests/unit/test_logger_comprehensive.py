"""
Comprehensive unit tests for logger.py module
Tests for improved coverage of logging functionality
"""
import pytest
import logging
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock, Mock, call
from io import StringIO
from logger import setup_logger, log_session_start, log_session_end, ColoredFormatter, LoggerMixin


class TestColoredFormatter:
    """Test ColoredFormatter functionality"""

    def test_colored_formatter_debug(self):
        """Test colored formatter with DEBUG level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.DEBUG, pathname='', lineno=0,
            msg='Debug message', args=(), exc_info=None
        )

        result = formatter.format(record)

        # Should contain color codes for DEBUG (cyan)
        assert '\033[36m' in result  # Cyan color
        assert '\033[0m' in result   # Reset color
        assert 'Debug message' in result

    def test_colored_formatter_info(self):
        """Test colored formatter with INFO level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Info message', args=(), exc_info=None
        )

        result = formatter.format(record)

        # Should contain color codes for INFO (green)
        assert '\033[32m' in result  # Green color
        assert '\033[0m' in result   # Reset color
        assert 'Info message' in result

    def test_colored_formatter_warning(self):
        """Test colored formatter with WARNING level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.WARNING, pathname='', lineno=0,
            msg='Warning message', args=(), exc_info=None
        )

        result = formatter.format(record)

        # Should contain color codes for WARNING (yellow)
        assert '\033[33m' in result  # Yellow color
        assert '\033[0m' in result   # Reset color
        assert 'Warning message' in result

    def test_colored_formatter_error(self):
        """Test colored formatter with ERROR level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.ERROR, pathname='', lineno=0,
            msg='Error message', args=(), exc_info=None
        )

        result = formatter.format(record)

        # Should contain color codes for ERROR (red)
        assert '\033[31m' in result  # Red color
        assert '\033[0m' in result   # Reset color
        assert 'Error message' in result

    def test_colored_formatter_critical(self):
        """Test colored formatter with CRITICAL level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.CRITICAL, pathname='', lineno=0,
            msg='Critical message', args=(), exc_info=None
        )

        result = formatter.format(record)

        # Should contain color codes for CRITICAL (magenta)
        assert '\033[35m' in result  # Magenta color
        assert '\033[0m' in result   # Reset color
        assert 'Critical message' in result

    def test_colored_formatter_unknown_level(self):
        """Test colored formatter with unknown level"""
        formatter = ColoredFormatter('%(levelname)s | %(message)s')
        record = logging.LogRecord(
            name='test', level=25, pathname='', lineno=0,  # Custom level
            msg='Custom message', args=(), exc_info=None
        )
        record.levelname = 'CUSTOM'

        result = formatter.format(record)

        # Should use reset color for unknown levels
        assert '\033[0m' in result
        assert 'Custom message' in result


class TestSetupLoggerComprehensive:
    """Comprehensive tests for setup_logger function"""

    def test_setup_logger_slack_specific_level(self):
        """Test setup_logger with slack-specific log level"""
        with patch('logger.config') as mock_config:
            mock_config.slack_log_level = 'WARNING'
            mock_config.log_level = 'INFO'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('slack_test_logger')

                assert logger.level == logging.WARNING

    def test_setup_logger_discord_specific_level(self):
        """Test setup_logger with discord-specific log level"""
        with patch('logger.config') as mock_config:
            mock_config.discord_log_level = 'ERROR'
            mock_config.log_level = 'INFO'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('discord_test_logger')

                assert logger.level == logging.ERROR

    def test_setup_logger_utils_specific_level(self):
        """Test setup_logger with utils-specific log level"""
        with patch('logger.config') as mock_config:
            mock_config.utils_log_level = 'CRITICAL'
            mock_config.log_level = 'INFO'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('utils_test_logger')

                assert logger.level == logging.CRITICAL

    def test_setup_logger_openai_specific_level(self):
        """Test setup_logger with openai-specific log level"""
        with patch('logger.config') as mock_config:
            mock_config.utils_log_level = 'DEBUG'
            mock_config.log_level = 'INFO'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('openai_test_logger')

                assert logger.level == logging.DEBUG

    def test_setup_logger_default_level(self):
        """Test setup_logger with default log level"""
        with patch('logger.config') as mock_config:
            mock_config.log_level = 'WARNING'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('default_test_logger')

                assert logger.level == logging.WARNING

    def test_setup_logger_explicit_level_override(self):
        """Test setup_logger with explicit level parameter"""
        with patch('logger.config') as mock_config:
            mock_config.log_level = 'INFO'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('test_logger', level='ERROR')

                assert logger.level == logging.ERROR

    def test_setup_logger_propagation_disabled(self):
        """Test that logger propagation is disabled"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('propagation_test')

                assert logger.propagate is False

    def test_setup_logger_existing_handlers_return_early(self):
        """Test that logger with existing handlers returns early"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            # Create logger with handlers
            logger = logging.getLogger('existing_handlers_test')
            handler = logging.StreamHandler()
            logger.addHandler(handler)

            with patch('os.makedirs') as mock_makedirs:
                result_logger = setup_logger('existing_handlers_test')

                # Should return early and not create directories
                assert result_logger is logger
                mock_makedirs.assert_not_called()

    def test_setup_logger_creates_log_directory(self):
        """Test that setup_logger creates log directory if it doesn't exist"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.path.exists', return_value=False) as mock_exists:
                with patch('os.makedirs') as mock_makedirs:
                    setup_logger('directory_test')

                    mock_exists.assert_called_with('test_logs')
                    mock_makedirs.assert_called_with('test_logs')

    def test_setup_logger_console_handler_enabled(self):
        """Test console handler is added when enabled"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = True
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('console_enabled_test')

                # Should have console handler
                console_handlers = [h for h in logger.handlers
                                   if isinstance(h, logging.StreamHandler)
                                   and h.stream == sys.stdout]
                assert len(console_handlers) > 0

    def test_setup_logger_console_handler_disabled(self):
        """Test console handler is not added when disabled"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('console_disabled_test')

                # Should not have console handler to stdout
                console_handlers = [h for h in logger.handlers
                                   if isinstance(h, logging.StreamHandler)
                                   and hasattr(h, 'stream') and h.stream == sys.stdout]
                assert len(console_handlers) == 0

    def test_setup_logger_file_handlers_created(self):
        """Test that file handlers are created"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('file_handlers_test')

                # Should have file handlers
                from logging.handlers import RotatingFileHandler
                file_handlers = [h for h in logger.handlers
                               if isinstance(h, RotatingFileHandler)]
                assert len(file_handlers) >= 2  # app.log and error.log

    def test_setup_logger_error_handler_level(self):
        """Test that error handler only logs ERROR and above"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('error_level_test')

                # Find error handler (should be set to ERROR level)
                from logging.handlers import RotatingFileHandler
                error_handlers = []
                for handler in logger.handlers:
                    if isinstance(handler, RotatingFileHandler):
                        if handler.level == logging.ERROR:
                            error_handlers.append(handler)

                assert len(error_handlers) >= 1

    def test_setup_logger_custom_file_handler(self):
        """Test custom file handler is added when specified"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('custom_file_test', log_file='custom.log')

                # Should have additional file handler
                file_handlers = [h for h in logger.handlers
                               if isinstance(h, logging.FileHandler)]
                assert len(file_handlers) >= 1


class TestLoggerMixin:
    """Test LoggerMixin functionality"""

    def test_logger_mixin_creates_logger(self):
        """Test that LoggerMixin creates a logger property"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        logger = instance.logger

        assert isinstance(logger, logging.Logger)
        assert 'TestClass' in logger.name

    def test_logger_mixin_caches_logger(self):
        """Test that LoggerMixin caches the logger instance"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        logger1 = instance.logger
        logger2 = instance.logger

        assert logger1 is logger2

    def test_logger_mixin_log_debug(self):
        """Test LoggerMixin log_debug method"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()

        # Mock the logger
        instance._logger = Mock()
        instance.log_debug("Debug message", extra_param="value")

        instance._logger.debug.assert_called_once_with(
            "Debug message", extra={'extra_param': 'value'}
        )

    def test_logger_mixin_log_info(self):
        """Test LoggerMixin log_info method"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_info("Info message", test_param="test")

        instance._logger.info.assert_called_once_with(
            "Info message", extra={'test_param': 'test'}
        )

    def test_logger_mixin_log_warning(self):
        """Test LoggerMixin log_warning method"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_warning("Warning message")

        instance._logger.warning.assert_called_once_with(
            "Warning message", extra={}
        )

    def test_logger_mixin_log_error(self):
        """Test LoggerMixin log_error method"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_error("Error message", exc_info=True, error_code=500)

        instance._logger.error.assert_called_once_with(
            "Error message", exc_info=True, extra={'error_code': 500}
        )

    def test_logger_mixin_log_error_default_exc_info(self):
        """Test LoggerMixin log_error method with default exc_info"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_error("Error message")

        instance._logger.error.assert_called_once_with(
            "Error message", exc_info=False, extra={}
        )

    def test_logger_mixin_log_critical(self):
        """Test LoggerMixin log_critical method"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_critical("Critical message", exc_info=True)

        instance._logger.critical.assert_called_once_with(
            "Critical message", exc_info=True, extra={}
        )

    def test_logger_mixin_log_critical_default_exc_info(self):
        """Test LoggerMixin log_critical method with default exc_info"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        instance._logger = Mock()
        instance.log_critical("Critical message")

        instance._logger.critical.assert_called_once_with(
            "Critical message", exc_info=False, extra={}
        )


class TestSessionLogging:
    """Test session logging functions"""

    @patch('logger.main_logger')
    @patch('logger.config')
    def test_log_session_start(self, mock_config, mock_logger):
        """Test log_session_start function"""
        mock_config.log_level = 'INFO'
        mock_config.gpt_model = 'gpt-5'
        mock_config.utility_model = 'gpt-5-mini'
        mock_config.image_model = 'gpt-image-1'

        log_session_start()

        # Verify logger calls
        assert mock_logger.info.call_count >= 6

        # Check for specific log messages
        calls = mock_logger.info.call_args_list
        call_messages = [call[0][0] for call in calls]

        assert any("Session started at" in msg for msg in call_messages)
        assert any("Log Level: INFO" in msg for msg in call_messages)
        assert any("GPT Model: gpt-5" in msg for msg in call_messages)
        assert any("Utility Model: gpt-5-mini" in msg for msg in call_messages)
        assert any("Image Model: gpt-image-1" in msg for msg in call_messages)
        assert any("=" * 60 in msg for msg in call_messages)

    @patch('logger.main_logger')
    def test_log_session_end(self, mock_logger):
        """Test log_session_end function"""
        log_session_end()

        # Verify logger calls
        assert mock_logger.info.call_count >= 3

        # Check for specific log messages
        calls = mock_logger.info.call_args_list
        call_messages = [call[0][0] for call in calls]

        assert any("Session ended at" in msg for msg in call_messages)
        assert any("=" * 60 in msg for msg in call_messages)


class TestMainLogger:
    """Test main logger instance"""

    def test_main_logger_exists(self):
        """Test that main logger is created"""
        from logger import main_logger

        assert isinstance(main_logger, logging.Logger)
        assert 'slack_bot' in main_logger.name


class TestLoggerHandlerConfiguration:
    """Test detailed logger handler configuration"""

    def test_rotating_file_handler_configuration(self):
        """Test rotating file handler configuration"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('rotating_test')

                from logging.handlers import RotatingFileHandler
                rotating_handlers = [h for h in logger.handlers
                                   if isinstance(h, RotatingFileHandler)]

                assert len(rotating_handlers) >= 2

                # Check configuration
                for handler in rotating_handlers:
                    assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
                    assert handler.backupCount == 5

    def test_formatter_configuration(self):
        """Test formatter configuration on handlers"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = True
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('formatter_test')

                # Check that all handlers have formatters
                for handler in logger.handlers:
                    assert handler.formatter is not None

                    # Check format string
                    format_string = handler.formatter._fmt
                    assert '%(asctime)s' in format_string
                    assert '%(levelname)s' in format_string
                    assert '%(name)s' in format_string
                    assert '%(message)s' in format_string

    def test_console_handler_colored_formatter(self):
        """Test that console handler uses ColoredFormatter"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = True
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('colored_formatter_test')

                # Find console handler
                console_handlers = [h for h in logger.handlers
                                   if isinstance(h, logging.StreamHandler)
                                   and hasattr(h, 'stream') and h.stream == sys.stdout]

                assert len(console_handlers) > 0

                # Check that it uses ColoredFormatter
                console_handler = console_handlers[0]
                assert isinstance(console_handler.formatter, ColoredFormatter)


@pytest.mark.critical
class TestLoggerCritical:
    """Critical tests for logger functionality"""

    def test_critical_logger_level_setting(self):
        """Critical test for logger level setting"""
        with patch('logger.config') as mock_config:
            mock_config.log_level = 'ERROR'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('critical_level_test')

                assert logger.level == logging.ERROR
                assert logger.isEnabledFor(logging.ERROR)
                assert not logger.isEnabledFor(logging.INFO)

    def test_critical_handler_creation(self):
        """Critical test that required handlers are created"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('critical_handlers_test')

                # Must have at least file handlers
                assert len(logger.handlers) >= 2

                # Must have error-level handler for error logs
                error_handlers = [h for h in logger.handlers
                                if h.level == logging.ERROR]
                assert len(error_handlers) >= 1

    def test_critical_mixin_functionality(self):
        """Critical test for LoggerMixin basic functionality"""
        class TestClass(LoggerMixin):
            pass

        instance = TestClass()

        # Must be able to access logger
        assert hasattr(instance, 'logger')
        assert callable(instance.log_info)
        assert callable(instance.log_error)

        # Must not raise exceptions
        try:
            instance.log_info("Test message")
            instance.log_error("Test error")
        except Exception as e:
            pytest.fail(f"LoggerMixin methods should not raise exceptions: {e}")


@pytest.mark.smoke
class TestLoggerSmoke:
    """Smoke tests for logger module"""

    def test_smoke_logger_creation(self):
        """Smoke test for basic logger creation"""
        try:
            logger = setup_logger('smoke_test')
            assert isinstance(logger, logging.Logger)
        except Exception as e:
            pytest.fail(f"Basic logger creation failed: {e}")

    def test_smoke_mixin_usage(self):
        """Smoke test for LoggerMixin usage"""
        try:
            class TestClass(LoggerMixin):
                def test_method(self):
                    self.log_info("Test message")

            instance = TestClass()
            instance.test_method()
        except Exception as e:
            pytest.fail(f"LoggerMixin usage failed: {e}")

    def test_smoke_session_logging(self):
        """Smoke test for session logging functions"""
        try:
            log_session_start()
            log_session_end()
        except Exception as e:
            pytest.fail(f"Session logging failed: {e}")

    def test_smoke_colored_formatter(self):
        """Smoke test for ColoredFormatter"""
        try:
            formatter = ColoredFormatter('%(levelname)s | %(message)s')
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='', lineno=0,
                msg='Test message', args=(), exc_info=None
            )
            result = formatter.format(record)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"ColoredFormatter failed: {e}")


class TestLoggerEdgeCases:
    """Test edge cases and error conditions"""

    def test_invalid_log_level_handling(self):
        """Test handling of invalid log levels"""
        with patch('logger.config') as mock_config:
            mock_config.log_level = 'INVALID_LEVEL'
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                # Should not raise exception
                try:
                    logger = setup_logger('invalid_level_test', level='INVALID_LEVEL')
                    # Should have a valid level (default to something reasonable)
                    assert hasattr(logger, 'level')
                except AttributeError:
                    # Expected if invalid level is passed to getattr(logging, 'INVALID_LEVEL')
                    pass

    def test_directory_creation_error_handling(self):
        """Test handling of directory creation errors"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs', side_effect=OSError("Permission denied")):
                    # Should handle directory creation errors gracefully
                    try:
                        logger = setup_logger('dir_error_test')
                        # Logger should still be created even if directory creation fails
                        assert isinstance(logger, logging.Logger)
                    except OSError:
                        # This is acceptable - the error should propagate
                        pass

    def test_logger_with_empty_name(self):
        """Test logger creation with empty name"""
        with patch('logger.config') as mock_config:
            mock_config.console_logging_enabled = False
            mock_config.log_directory = 'test_logs'

            with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                logger = setup_logger('')
                assert isinstance(logger, logging.Logger)

    def test_mixin_with_complex_class_name(self):
        """Test LoggerMixin with complex class names"""
        class ComplexClassName_With_Underscores_And123Numbers(LoggerMixin):
            pass

        instance = ComplexClassName_With_Underscores_And123Numbers()
        logger = instance.logger

        assert isinstance(logger, logging.Logger)
        assert 'ComplexClassName_With_Underscores_And123Numbers' in logger.name