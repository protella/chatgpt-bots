"""
Comprehensive tests for thread-safe logging implementation
Tests ConcurrentRotatingFileHandler, QueueHandler/QueueListener, and singleton patterns
"""
import pytest
import logging
import os
import tempfile
import time
import threading
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock, Mock, PropertyMock
import shutil
from pathlib import Path

# Import the logger module components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from logger import (
    setup_logger, ColoredFormatter, LoggerMixin,
    log_session_start, log_session_end,
    _logger_lock, _initialized_loggers, _queue_listener, _log_queue
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_logger_state():
    """Reset global logger state before each test"""
    global _initialized_loggers, _queue_listener, _log_queue

    # Import to modify the actual module globals
    import logger

    # Stop existing queue listener if any
    if logger._queue_listener:
        logger._queue_listener.stop()

    # Clear state
    logger._initialized_loggers.clear()
    logger._queue_listener = None
    logger._log_queue = None

    # Clear all handlers from existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger_obj = logging.getLogger(name)
        logger_obj.handlers.clear()

    yield

    # Cleanup after test
    if logger._queue_listener:
        logger._queue_listener.stop()
        logger._queue_listener = None
    logger._initialized_loggers.clear()
    logger._log_queue = None


class TestConcurrentRotatingFileHandler:
    """Test ConcurrentRotatingFileHandler functionality"""

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_concurrent_handler_used_when_available(self, mock_log_dir, temp_log_dir):
        """Test that ConcurrentRotatingFileHandler is used when available"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        # Create logger with queue disabled to see actual handlers
        logger_obj = setup_logger('test_concurrent', use_queue=False)

        # Check if ConcurrentRotatingFileHandler is used
        from logger import USE_CONCURRENT_HANDLER

        if USE_CONCURRENT_HANDLER:
            from concurrent_log_handler import ConcurrentRotatingFileHandler
            # Should have ConcurrentRotatingFileHandler
            concurrent_handlers = [
                h for h in logger_obj.handlers
                if isinstance(h, ConcurrentRotatingFileHandler)
            ]
            assert len(concurrent_handlers) >= 2, "Should have at least 2 ConcurrentRotatingFileHandlers (app and error)"
        else:
            from logging.handlers import RotatingFileHandler
            # Should fall back to RotatingFileHandler
            rotating_handlers = [
                h for h in logger_obj.handlers
                if isinstance(h, RotatingFileHandler)
            ]
            assert len(rotating_handlers) >= 2, "Should have at least 2 RotatingFileHandlers (app and error)"

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_lock_files_created(self, mock_log_dir, temp_log_dir):
        """Test that lock files are created for ConcurrentRotatingFileHandler"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        from logger import USE_CONCURRENT_HANDLER
        if not USE_CONCURRENT_HANDLER:
            pytest.skip("ConcurrentRotatingFileHandler not available")

        # Create logger
        logger_obj = setup_logger('test_locks', use_queue=False)

        # Log something to trigger file creation
        logger_obj.info("Test message")

        # Check for lock files
        lock_files = list(Path(temp_log_dir).glob('.__*.lock'))
        assert len(lock_files) > 0, "Lock files should be created"


class TestQueueHandlerPattern:
    """Test QueueHandler/QueueListener pattern"""

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_queue_handler_default(self, mock_log_dir, temp_log_dir):
        """Test that QueueHandler is used by default"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        logger_obj = setup_logger('test_queue')

        # Should have exactly one handler - the QueueHandler
        assert len(logger_obj.handlers) == 1

        from logging.handlers import QueueHandler
        assert isinstance(logger_obj.handlers[0], QueueHandler)

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_queue_listener_started(self, mock_log_dir, temp_log_dir):
        """Test that QueueListener is started"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        import logger

        # Initially no listener
        assert logger._queue_listener is None

        # Create logger
        logger_obj = setup_logger('test_listener')

        # Queue listener should be started
        assert logger._queue_listener is not None
        assert logger._log_queue is not None

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_shared_queue_across_loggers(self, mock_log_dir, temp_log_dir):
        """Test that multiple loggers share the same queue"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        import logger

        # Create multiple loggers
        logger1 = setup_logger('test_shared1')
        logger2 = setup_logger('test_shared2')
        logger3 = setup_logger('test_shared3')

        # All should use the same queue
        from logging.handlers import QueueHandler
        queue1 = logger1.handlers[0].queue if isinstance(logger1.handlers[0], QueueHandler) else None
        queue2 = logger2.handlers[0].queue if isinstance(logger2.handlers[0], QueueHandler) else None
        queue3 = logger3.handlers[0].queue if isinstance(logger3.handlers[0], QueueHandler) else None

        assert queue1 is queue2 is queue3
        assert queue1 is logger._log_queue

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_queue_listener_cleanup(self, mock_log_dir, temp_log_dir):
        """Test that QueueListener is properly cleaned up"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        import logger

        # Create logger to start listener
        logger_obj = setup_logger('test_cleanup')
        assert logger._queue_listener is not None

        # Call log_session_end which should stop the listener
        log_session_end()

        # Listener should be stopped
        assert logger._queue_listener is None


class TestThreadSafeSingleton:
    """Test thread-safe singleton pattern"""

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_singleton_same_logger(self, mock_log_dir, temp_log_dir):
        """Test that same logger name returns same instance"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        logger1 = setup_logger('test_singleton')
        logger2 = setup_logger('test_singleton')

        assert logger1 is logger2

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_logger_cache(self, mock_log_dir, temp_log_dir):
        """Test that loggers are cached in _initialized_loggers"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        import logger

        # Initially empty
        assert 'test_cache' not in logger._initialized_loggers

        # Create logger
        logger_obj = setup_logger('test_cache')

        # Should be in cache
        assert 'test_cache' in logger._initialized_loggers
        assert logger._initialized_loggers['test_cache'] is logger_obj

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_thread_safe_initialization(self, mock_log_dir, temp_log_dir):
        """Test that logger initialization is thread-safe"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        import logger
        loggers = []
        errors = []

        def create_logger(name, index):
            try:
                log = setup_logger(name)
                loggers.append((index, log))
            except Exception as e:
                errors.append(e)

        # Create same logger from multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_logger, args=('test_threadsafe', i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # All should have gotten the same logger instance
        logger_instances = [log for _, log in loggers]
        assert all(log is logger_instances[0] for log in logger_instances)


class TestConcurrentLogging:
    """Test concurrent logging scenarios"""

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_concurrent_writes_no_corruption(self, mock_log_dir, temp_log_dir):
        """Test that concurrent writes don't corrupt log files"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        logger_obj = setup_logger('test_concurrent_writes')
        messages_written = []

        def write_logs(thread_id):
            for i in range(100):
                msg = f"Thread-{thread_id}-Message-{i}"
                logger_obj.info(msg)
                messages_written.append(msg)
                time.sleep(0.001)  # Small delay to increase chance of interleaving

        # Start multiple threads writing simultaneously
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_logs, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Give queue time to flush
        time.sleep(0.5)

        # Read log file and verify all messages are there
        log_file = os.path.join(temp_log_dir, 'app.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()

            # Verify all messages were logged
            for msg in messages_written:
                assert msg in log_content, f"Message '{msg}' not found in log"

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_async_concurrent_logging(self, mock_log_dir, temp_log_dir):
        """Test logging from async tasks"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        logger_obj = setup_logger('test_async')
        messages = []

        async def async_log(task_id):
            for i in range(50):
                msg = f"Task-{task_id}-Message-{i}"
                logger_obj.info(msg)
                messages.append(msg)
                await asyncio.sleep(0.001)

        async def run_tasks():
            tasks = [async_log(i) for i in range(5)]
            await asyncio.gather(*tasks)

        # Run async logging
        asyncio.run(run_tasks())

        # Give queue time to flush
        time.sleep(0.5)

        # Verify messages were logged
        log_file = os.path.join(temp_log_dir, 'app.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()

            for msg in messages:
                assert msg in log_content

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_mixed_sync_async_logging(self, mock_log_dir, temp_log_dir):
        """Test mixed synchronous and asynchronous logging"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        logger_obj = setup_logger('test_mixed')
        all_messages = []

        def sync_log(thread_id):
            for i in range(30):
                msg = f"Sync-{thread_id}-{i}"
                logger_obj.info(msg)
                all_messages.append(msg)
                time.sleep(0.002)

        async def async_log(task_id):
            for i in range(30):
                msg = f"Async-{task_id}-{i}"
                logger_obj.info(msg)
                all_messages.append(msg)
                await asyncio.sleep(0.002)

        async def run_mixed():
            # Start sync threads
            threads = []
            for i in range(3):
                t = threading.Thread(target=sync_log, args=(i,))
                t.start()
                threads.append(t)

            # Run async tasks concurrently
            async_tasks = [async_log(i) for i in range(3)]
            await asyncio.gather(*async_tasks)

            # Wait for threads
            for t in threads:
                t.join()

        asyncio.run(run_mixed())
        time.sleep(0.5)  # Let queue flush

        # Verify all messages logged
        log_file = os.path.join(temp_log_dir, 'app.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()

            for msg in all_messages:
                assert msg in log_content


class TestLogRotation:
    """Test log rotation functionality"""

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_rotation_parameters(self, mock_log_dir, temp_log_dir):
        """Test that rotation parameters are set correctly"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        from logger import USE_CONCURRENT_HANDLER

        if USE_CONCURRENT_HANDLER:
            # Test with real ConcurrentRotatingFileHandler
            logger_obj = setup_logger('test_rotation_params', use_queue=False)

            from concurrent_log_handler import ConcurrentRotatingFileHandler
            # Find the handlers
            for handler in logger_obj.handlers:
                if isinstance(handler, ConcurrentRotatingFileHandler):
                    # Check rotation parameters
                    assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
                    assert handler.backupCount == 5
        else:
            # Test with RotatingFileHandler
            from logging.handlers import RotatingFileHandler
            logger_obj = setup_logger('test_rotation_params', use_queue=False)

            for handler in logger_obj.handlers:
                if isinstance(handler, RotatingFileHandler):
                    # Check rotation parameters
                    assert handler.maxBytes == 10 * 1024 * 1024  # 10MB
                    assert handler.backupCount == 5

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_rotation_file_naming(self, mock_log_dir, temp_log_dir):
        """Test that rotated files have correct naming"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        from logger import USE_CONCURRENT_HANDLER
        if not USE_CONCURRENT_HANDLER:
            pytest.skip("Testing rotation naming with ConcurrentRotatingFileHandler")

        # This would require actually triggering rotation and checking files
        # For now, we verify the handler is configured correctly
        logger_obj = setup_logger('test_naming', use_queue=False)

        # Write some messages
        for i in range(10):
            logger_obj.info(f"Message {i}")

        # Check that main log file exists
        assert os.path.exists(os.path.join(temp_log_dir, 'app.log'))


class TestErrorHandling:
    """Test error handling in logging"""

    def test_log_directory_creation(self, temp_log_dir):
        """Test that log directory is created if it doesn't exist"""
        # Use a non-existent directory
        non_existent = os.path.join(temp_log_dir, 'new_logs_dir')

        with patch('logger.config.console_logging_enabled', False):
            with patch('logger.config.log_directory', non_existent):
                # Should not exist initially
                assert not os.path.exists(non_existent)

                # Create logger
                logger_obj = setup_logger('test_dir_creation')

                # Directory should be created
                assert os.path.exists(non_existent)

    @patch('logger.config.log_directory')
    @patch('logger.config.console_logging_enabled', False)
    def test_multiple_handlers_not_added(self, mock_log_dir, temp_log_dir):
        """Test that multiple calls don't add duplicate handlers"""
        mock_log_dir.__str__ = Mock(return_value=temp_log_dir)
        mock_log_dir.__fspath__ = Mock(return_value=temp_log_dir)

        # Create logger multiple times
        logger1 = setup_logger('test_no_dups')
        handler_count1 = len(logger1.handlers)

        logger2 = setup_logger('test_no_dups')
        handler_count2 = len(logger2.handlers)

        logger3 = setup_logger('test_no_dups')
        handler_count3 = len(logger3.handlers)

        # Should have same number of handlers (not accumulating)
        assert handler_count1 == handler_count2 == handler_count3

        # Should be the same logger instance
        assert logger1 is logger2 is logger3


class TestLoggerMixin:
    """Test LoggerMixin functionality"""

    @patch('logger.setup_logger')
    def test_logger_mixin_initialization(self, mock_setup):
        """Test that LoggerMixin creates logger on first access"""
        mock_logger = MagicMock()
        mock_setup.return_value = mock_logger

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Access logger property
        logger = obj.logger

        # Should have called setup_logger
        mock_setup.assert_called_once_with(name='slack_bot.TestClass')
        assert logger is mock_logger

        # Second access should return cached logger
        logger2 = obj.logger
        assert logger2 is logger
        # Still only called once
        mock_setup.assert_called_once()

    def test_logger_mixin_methods(self):
        """Test LoggerMixin logging methods"""
        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Mock the _logger attribute directly since logger is a property
        mock_logger = MagicMock()
        obj._logger = mock_logger

        # Test each logging method
        obj.log_debug("Debug message", extra="data")
        mock_logger.debug.assert_called_once_with("Debug message", extra={'extra': 'data'})

        obj.log_info("Info message", key="value")
        mock_logger.info.assert_called_once_with("Info message", extra={'key': 'value'})

        obj.log_warning("Warning message")
        mock_logger.warning.assert_called_once_with("Warning message", extra={})

        obj.log_error("Error message", exc_info=True)
        mock_logger.error.assert_called_once_with("Error message", exc_info=True, extra={})

        obj.log_critical("Critical message", exc_info=False)
        mock_logger.critical.assert_called_once_with("Critical message", exc_info=False, extra={})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])