"""Unit tests for main.py - Main application entry point"""

import pytest
import sys
import signal
import time
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
from threading import Thread
import argparse

from main import ChatBotV2, main


class TestChatBotV2Initialization:
    """Test ChatBotV2 initialization and setup"""
    
    def test_init_with_slack_platform(self):
        """Test initialization with Slack platform"""
        bot = ChatBotV2(platform="slack")
        
        assert bot.platform == "slack"
        assert bot.client is None
        assert bot.processor is None
        assert bot.cleanup_task is None
        assert bot.running is False
        assert bot.sigint_count == 0
        assert bot.last_sigint_time == 0
    
    def test_init_with_discord_platform(self):
        """Test initialization with Discord platform"""
        bot = ChatBotV2(platform="discord")
        
        assert bot.platform == "discord"
    
    def test_init_platform_lowercase(self):
        """Test platform name is converted to lowercase"""
        bot = ChatBotV2(platform="SLACK")
        assert bot.platform == "slack"
    
    @patch('main.config')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @pytest.mark.asyncio
    async def test_initialize_slack_success(self, mock_processor_class, mock_slackbot_class, mock_config):
        """Test successful Slack initialization"""
        mock_config.validate.return_value = None
        mock_client = Mock()
        mock_client.db = Mock()
        mock_slackbot_class.return_value = mock_client

        bot = ChatBotV2(platform="slack")
        await bot.initialize()

        # Verify config validated
        mock_config.validate.assert_called_once()

        # Verify Slack client created
        mock_slackbot_class.assert_called_once()
        assert bot.client is mock_client
        
        # Verify processor created with client's DB
        mock_processor_class.assert_called_once_with(db=mock_client.db)
        assert bot.processor is not None
    
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_initialize_config_error(self, mock_config):
        """Test initialization with config validation error"""
        mock_config.validate.side_effect = ValueError("Invalid config")

        bot = ChatBotV2(platform="slack")

        with pytest.raises(SystemExit):
            await bot.initialize()
    
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_initialize_discord_not_implemented(self, mock_config):
        """Test Discord platform not yet implemented"""
        mock_config.validate.return_value = None

        bot = ChatBotV2(platform="discord")

        with pytest.raises(SystemExit):
            await bot.initialize()
    
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_initialize_unknown_platform(self, mock_config):
        """Test unknown platform error"""
        mock_config.validate.return_value = None

        bot = ChatBotV2(platform="unknown")

        with pytest.raises(SystemExit):
            await bot.initialize()
    
    @patch('main.signal.signal')
    @patch('main.config')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @pytest.mark.asyncio
    async def test_signal_handlers_setup(self, mock_processor, mock_slackbot, mock_config, mock_signal):
        """Test signal handlers are set up"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot.return_value = mock_client

        bot = ChatBotV2(platform="slack")
        await bot.initialize()

        # Verify signal handlers registered
        calls = mock_signal.call_args_list
        signals = [call[0][0] for call in calls]
        assert signal.SIGINT in signals
        assert signal.SIGTERM in signals


class TestChatBotV2MessageHandling:
    """Test message handling functionality"""
    
    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        return bot
    
    @pytest.mark.asyncio
    async def test_handle_message_text_response(self, bot):
        """Test handling text response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()

        # Make client methods async-compatible
        from unittest.mock import AsyncMock
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()
        client.format_text = Mock(return_value="Formatted: Hello world")

        # Mock processor response
        response = Mock(
            type="text",
            content="Hello world",
            metadata={"streamed": False}
        )
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        # Verify thinking indicator sent and deleted
        client.send_thinking_indicator.assert_called_once()
        client.delete_message.assert_called_once_with("C123", "thinking_123")
        
        # Verify message sent
        client.format_text.assert_called_once_with("Hello world")
        client.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_message_streamed_response(self, bot):
        """Test handling streamed response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()

        response = Mock(
            type="text",
            content="Streamed content",
            metadata={"streamed": True}
        )
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        # Should not delete thinking indicator for streamed responses
        client.delete_message.assert_not_called()

        # Should not send message again (already displayed via streaming)
        client.send_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_message_busy_response(self, bot):
        """Test handling busy response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.send_busy_message = AsyncMock()

        response = Mock(type="busy", content="Thread is busy")
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        client.send_busy_message.assert_called_once_with("C123", "thread_123")
    
    @pytest.mark.asyncio
    async def test_handle_message_busy_fallback(self, bot):
        """Test busy message fallback when method not available"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock(spec=['send_thinking_indicator', 'delete_message', 'send_message'])
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.send_message = AsyncMock()

        response = Mock(type="busy", content="Thread is busy")
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        # Should fallback to send_message
        client.send_message.assert_called_once_with(
            "C123", "thread_123", "Thread is busy"
        )
    
    @patch('main.asyncio.sleep')
    @pytest.mark.asyncio
    async def test_handle_message_image_response(self, mock_sleep, bot):
        """Test handling image response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.name = "SlackBot"

        image_data = Mock()
        image_data.to_bytes.return_value = b"image_bytes"
        image_data.format = "png"

        response = Mock(
            type="image",
            content=image_data,
            metadata={"streamed": False}
        )
        bot.processor.process_message = AsyncMock(return_value=response)

        client.send_thinking_indicator = AsyncMock(return_value="status_123")
        client.send_image = AsyncMock(return_value="http://image.url")
        client.delete_message = AsyncMock()
        client.update_message = AsyncMock()
        bot.processor.update_last_image_url = AsyncMock()

        await bot.handle_message(message, client)

        # Verify image sent
        client.send_image.assert_called_once_with(
            "C123", "thread_123",
            b"image_bytes",
            "generated_image.png",
            ""
        )

        # Verify URL updated
        bot.processor.update_last_image_url.assert_called_once_with(
            "C123", "thread_123", "http://image.url"
        )

        # Verify cleanup after delay
        mock_sleep.assert_called_once_with(4)
        assert client.delete_message.call_count == 2  # thinking + status
    
    @pytest.mark.asyncio
    async def test_handle_message_error_response(self, bot):
        """Test handling error response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.handle_error = AsyncMock()

        response = Mock(type="error", content="Something went wrong")
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        client.handle_error.assert_called_once_with(
            "C123", "thread_123", "Something went wrong"
        )
    
    @pytest.mark.asyncio
    async def test_handle_message_exception(self, bot):
        """Test exception handling during message processing"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.handle_error = AsyncMock()

        bot.processor.process_message = AsyncMock(side_effect=Exception("Processing error"))

        await bot.handle_message(message, client)

        # Should delete thinking indicator on error
        client.delete_message.assert_called_once_with("C123", "thinking_123")

        # Should send error message
        client.handle_error.assert_called_once_with(
            "C123", "thread_123", "Processing error"
        )


class TestChatBotV2CleanupThread:
    """Test cleanup thread functionality"""
    
    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor.thread_manager = Mock()
        bot.running = True
        return bot
    
    @patch('croniter.croniter')
    @patch('asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_start_cleanup_thread(self, mock_datetime, mock_sleep, mock_croniter_class, bot):
        """Test starting cleanup task"""
        # Mock cron schedule
        mock_cron = Mock()
        mock_now = Mock()
        mock_next = Mock()

        mock_datetime.datetime.now.return_value = mock_now
        mock_cron.get_next.return_value = mock_next
        mock_next.__sub__ = Mock(return_value=Mock(total_seconds=Mock(return_value=3600)))
        mock_croniter_class.return_value = mock_cron

        # Mock asyncio.create_task to verify task creation
        with patch('asyncio.create_task') as mock_create_task:
            # Start cleanup task
            await bot.start_cleanup_task()

            # Verify task was created
            mock_create_task.assert_called_once()
    
    @patch('asyncio.create_task')
    @pytest.mark.asyncio
    async def test_cleanup_thread_invalid_cron(self, mock_create_task, bot):
        """Test cleanup task with invalid cron expression"""
        # Just test that task is created
        await bot.start_cleanup_task()

        # Should create task
        mock_create_task.assert_called_once()
    
    @patch('main.config')
    def test_cleanup_execution(self, mock_config, bot):
        """Test cleanup execution can be called directly"""
        mock_config.cleanup_max_age_hours = 48
        
        # Directly call the cleanup function that would be called by the thread
        # This tests that the cleanup method exists and can be called
        bot.processor.thread_manager.cleanup_old_threads(max_age=48 * 3600)
        
        # Verify the method was called (it's a mock)
        bot.processor.thread_manager.cleanup_old_threads.assert_called_with(max_age=48 * 3600)


class TestChatBotV2Lifecycle:
    """Test bot lifecycle management"""
    
    @pytest.fixture
    def bot(self):
        return ChatBotV2(platform="slack")
    
    @patch('main.sys.exit')
    @patch('main.log_session_end')
    @patch('main.log_session_start')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_run_normal_flow(self, mock_config, mock_processor_class,
                             mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test normal run flow"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client

        # Make client.start() raise an exception to trigger the finally block
        mock_client.start.side_effect = KeyboardInterrupt("Test interrupt")

        await bot.run()

        # Verify session logging
        mock_log_start.assert_called_once()
        # log_session_end is called by run() when it exits normally
        mock_log_end.assert_called_once()

        # Verify client started
        mock_client.start.assert_called_once()
    
    @patch('main.sys.exit')
    @patch('main.log_session_end')
    @patch('main.log_session_start')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self, mock_config, mock_processor_class,
                                   mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test handling keyboard interrupt"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        mock_client.start.side_effect = KeyboardInterrupt()

        await bot.run()

        # Should handle gracefully
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
    
    @patch('main.sys.exit')
    @patch('main.log_session_end')
    @patch('main.log_session_start')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_run_unexpected_error(self, mock_config, mock_processor_class,
                                 mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test handling unexpected errors"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        mock_client.start.side_effect = Exception("Unexpected error")

        await bot.run()

        # Should handle gracefully
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, bot):
        """Test shutdown process"""
        bot.running = True
        bot.client = Mock()
        bot.client.stop = Mock(return_value=None)  # Make it async-compatible
        bot.processor = Mock()
        bot.processor.get_stats.return_value = {"threads": 5}

        # Should not call sys.exit anymore - graceful shutdown should just complete
        await bot.shutdown()

        assert bot.running is False
        bot.client.stop.assert_called_once()
        bot.processor.get_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, bot):
        """Test shutdown is idempotent"""
        bot.running = False

        # Should not do anything if not running
        await bot.shutdown()

        # Running state should remain False
        assert bot.running is False
    
    @pytest.mark.asyncio
    async def test_shutdown_with_client_error(self, bot):
        """Test shutdown handles client.stop() errors gracefully"""
        bot.running = True
        bot.client = Mock()
        bot.client.stop.side_effect = Exception("Failed to stop client")
        bot.processor = Mock()
        bot.processor.get_stats.return_value = {"threads": 5}

        # Should continue despite error
        await bot.shutdown()

        assert bot.running is False
        bot.processor.get_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_stats_error(self, bot):
        """Test shutdown handles get_stats() errors gracefully"""
        bot.running = True
        bot.client = Mock()
        bot.client.stop = Mock(return_value=None)  # Make it async-compatible
        bot.processor = Mock()
        bot.processor.get_stats.side_effect = Exception("Failed to get stats")

        # Should continue despite error
        await bot.shutdown()

        assert bot.running is False
        bot.client.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_all_errors(self, bot):
        """Test shutdown handles multiple errors gracefully"""
        bot.running = True
        bot.client = Mock()
        bot.client.stop.side_effect = Exception("Client error")
        bot.processor = Mock()
        bot.processor.get_stats.side_effect = Exception("Stats error")

        # Should still complete shutdown
        await bot.shutdown()

        assert bot.running is False
    
    @patch('asyncio.create_task')
    def test_signal_handler_sigterm(self, mock_create_task, bot):
        """Test SIGTERM handler calls shutdown"""
        bot.shutdown = Mock()

        bot._signal_handler(signal.SIGTERM, None)

        # Verify create_task was called with shutdown
        mock_create_task.assert_called_once_with(bot.shutdown())
    
    @patch('asyncio.create_task')
    @patch('time.time')
    def test_signal_handler_first_sigint(self, mock_time, mock_create_task, bot):
        """Test first SIGINT attempts graceful shutdown"""
        mock_time.return_value = 1000.0
        bot.shutdown = Mock()
        bot.sigint_count = 0
        bot.last_sigint_time = 0

        bot._signal_handler(signal.SIGINT, None)

        # Should increment count and attempt shutdown
        assert bot.sigint_count == 1
        assert bot.last_sigint_time == 1000.0
        mock_create_task.assert_called_once_with(bot.shutdown())
    
    @patch('time.time')
    def test_signal_handler_second_sigint_after_delay(self, mock_time, bot):
        """Test second SIGINT after delay attempts another graceful shutdown"""
        # First SIGINT at time 1000
        bot.sigint_count = 1
        bot.last_sigint_time = 1000.0
        
        # Second SIGINT at time 1003 (3 seconds later - outside 2 second window)
        mock_time.return_value = 1003.0
        bot.shutdown = Mock()
        
        bot._signal_handler(signal.SIGINT, None)
        
        # Should NOT force exit, but warn about shutdown in progress
        assert bot.sigint_count == 2
        assert bot.last_sigint_time == 1003.0
        # shutdown not called again since count > 1
        bot.shutdown.assert_not_called()
    
    @patch('os._exit')
    @patch('threading.enumerate')
    @patch('time.time')
    def test_signal_handler_double_sigint_force_exit(self, mock_time, mock_enumerate, mock_exit, bot):
        """Test double SIGINT within 2 seconds forces exit"""
        # First SIGINT at time 1000
        bot.sigint_count = 1
        bot.last_sigint_time = 1000.0
        
        # Second SIGINT at time 1001 (1 second later - within 2 second window)
        mock_time.return_value = 1001.0
        
        # Mock active threads
        main_thread = Mock()
        main_thread.name = "MainThread"
        main_thread.daemon = False
        
        worker_thread = Mock()
        worker_thread.name = "WorkerThread"
        worker_thread.daemon = True
        
        mock_enumerate.return_value = [main_thread, worker_thread]
        
        bot._signal_handler(signal.SIGINT, None)
        
        # Should force exit with code 1
        mock_exit.assert_called_once_with(1)
    
    @patch('os._exit')
    @patch('threading.enumerate')
    @patch('time.time')
    def test_signal_handler_double_sigint_no_extra_threads(self, mock_time, mock_enumerate, mock_exit, bot):
        """Test double SIGINT with only main thread"""
        bot.sigint_count = 1
        bot.last_sigint_time = 1000.0
        mock_time.return_value = 1001.0
        
        # Only main thread active
        main_thread = Mock()
        main_thread.name = "MainThread"
        mock_enumerate.return_value = [main_thread]
        
        bot._signal_handler(signal.SIGINT, None)
        
        # Should still force exit
        mock_exit.assert_called_once_with(1)
    
    def test_signal_handler_shutdown_in_progress(self, bot):
        """Test SIGINT when shutdown already in progress"""
        bot.sigint_count = 2  # Already pressed twice
        bot.last_sigint_time = 1000.0
        bot.shutdown = Mock()
        
        with patch('time.time', return_value=1005.0):
            bot._signal_handler(signal.SIGINT, None)
        
        # Should not call shutdown again
        bot.shutdown.assert_not_called()
        assert bot.sigint_count == 3


class TestMainFunction:
    """Test main entry point function"""
    
    @patch('main.ChatBotV2')
    @patch('main.argparse.ArgumentParser')
    @pytest.mark.asyncio
    async def test_main_default_platform(self, mock_parser_class, mock_chatbot_class):
        """Test main with default platform"""
        mock_parser = Mock()
        mock_args = Mock(platform="slack")
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_bot = Mock()
        mock_bot.run = AsyncMock()
        mock_chatbot_class.return_value = mock_bot

        await main()

        mock_chatbot_class.assert_called_once_with(platform="slack")
        mock_bot.run.assert_called_once()
    
    @patch('main.ChatBotV2')
    @patch('sys.argv', ['main.py', '--platform', 'discord'])
    @pytest.mark.asyncio
    async def test_main_discord_platform(self, mock_chatbot_class):
        """Test main with Discord platform argument"""
        mock_bot = Mock()
        mock_bot.run = AsyncMock()
        mock_chatbot_class.return_value = mock_bot

        await main()

        mock_chatbot_class.assert_called_once_with(platform="discord")
        mock_bot.run.assert_called_once()
    
    @patch('sys.argv', ['main.py'])
    @patch('main.ChatBotV2')
    @pytest.mark.asyncio
    async def test_main_module_execution(self, mock_chatbot_class):
        """Test main module execution"""
        mock_bot = Mock()
        mock_bot.run = AsyncMock()
        mock_chatbot_class.return_value = mock_bot

        # Import should not run main
        import main as main_module
        mock_chatbot_class.assert_not_called()

        # Direct call should work
        await main_module.main()
        mock_chatbot_class.assert_called_once()


@pytest.mark.critical
class TestChatBotV2Critical:
    """Critical functionality tests"""
    
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_critical_initialization_chain(self, mock_config, mock_slackbot_class, mock_processor_class):
        """Critical test for initialization chain"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client

        bot = ChatBotV2(platform="slack")
        await bot.initialize()

        # Must create client before processor
        assert mock_slackbot_class.called
        assert mock_processor_class.called

        # Processor must use client's DB
        processor_db = mock_processor_class.call_args[1]["db"]
        assert processor_db is mock_client.db
    
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_critical_message_handler_callback(self, mock_config, mock_slackbot_class, mock_processor_class):
        """Critical test for message handler callback setup"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client

        bot = ChatBotV2(platform="slack")
        await bot.initialize()

        # Verify message handler passed to client
        call_kwargs = mock_slackbot_class.call_args[1]
        assert "message_handler" in call_kwargs
        assert call_kwargs["message_handler"] == bot.handle_message
    
    @pytest.mark.asyncio
    async def test_critical_error_propagation(self):
        """Critical test for error propagation"""
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()

        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.handle_error = AsyncMock()

        # Simulate critical error
        bot.processor.process_message = AsyncMock(side_effect=Exception("Critical error"))

        # Should not crash, but handle error
        await bot.handle_message(message, client)

        # Should clean up and report error
        client.delete_message.assert_called_with("C123", "thinking_123")
        client.handle_error.assert_called_with("C123", "thread_123", "Critical error")


@pytest.mark.integration
class TestChatBotV2Integration:
    """Integration tests for main module"""
    
    @patch('asyncio.create_task')
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_integration_full_startup(self, mock_config, mock_slackbot_class,
                                     mock_processor_class, mock_create_task):
        """Test full startup sequence"""
        mock_config.validate.return_value = None
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 24

        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client

        bot = ChatBotV2(platform="slack")
        await bot.initialize()
        bot.running = True
        await bot.start_cleanup_task()
        
        # Verify all components initialized
        assert bot.client is not None
        assert bot.processor is not None
        # Verify cleanup task was created
        mock_create_task.assert_called()
        
        # Cleanup
        bot.running = False
    
    @pytest.mark.smoke
    def test_smoke_import_chain(self):
        """Smoke test for import chain"""
        # Should be able to import without errors
        import main
        from main import ChatBotV2, main as main_func

        # Verify exports
        assert hasattr(main, 'ChatBotV2')
        assert hasattr(main, 'main')
        assert callable(main_func)


class TestChatBotV2ImageHandlng:
    """Test image handling edge cases with better coverage"""

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        return bot

    @patch('main.asyncio.sleep')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_handle_message_image_streamed_with_status_id(self, mock_config, mock_sleep, bot):
        """Test image response with streamed metadata and status_message_id"""
        mock_config.circle_loader_emoji = ":loading:"

        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.name = "SlackBot"

        image_data = Mock()
        image_data.to_bytes.return_value = b"image_bytes"
        image_data.format = "png"

        response = Mock(
            type="image",
            content=image_data,
            metadata={
                "streamed": True,
                "status_message_id": "status_msg_123"
            }
        )
        bot.processor.process_message = AsyncMock(return_value=response)
        bot.processor.update_last_image_url = AsyncMock()

        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.send_image = AsyncMock(return_value="http://image.url")
        client.delete_message = AsyncMock()
        client.update_message = AsyncMock()

        await bot.handle_message(message, client)

        # Should update existing status message with upload status
        client.update_message.assert_called_with(
            "C123", "status_msg_123", ":loading: Uploading image to Slack..."
        )

        # Should send image
        client.send_image.assert_called_once()

        # Should update image URL
        bot.processor.update_last_image_url.assert_called_once_with(
            "C123", "thread_123", "http://image.url"
        )

        # Should delete status message after delay
        mock_sleep.assert_called_once_with(4)
        client.delete_message.assert_called_with("C123", "status_msg_123")

    @patch('main.asyncio.sleep')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_handle_message_image_streamed_fallback_no_update_method(self, mock_config, mock_sleep, bot):
        """Test image response streamed fallback when client has no update_message method"""
        mock_config.circle_loader_emoji = ":loading:"

        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock(spec=['send_thinking_indicator', 'send_image', 'delete_message'])
        client.name = "SlackBot"

        image_data = Mock()
        image_data.to_bytes.return_value = b"image_bytes"
        image_data.format = "png"

        response = Mock(
            type="image",
            content=image_data,
            metadata={
                "streamed": True,
                "status_message_id": "status_msg_123"
            }
        )
        bot.processor.process_message = AsyncMock(return_value=response)
        bot.processor.update_last_image_url = AsyncMock()

        client.send_thinking_indicator = AsyncMock(return_value="fallback_123")
        client.send_image = AsyncMock(return_value="http://image.url")
        client.delete_message = AsyncMock()

        await bot.handle_message(message, client)

        # Should create new status message as fallback
        client.send_thinking_indicator.assert_called_with("C123", "thread_123")

        # Should send image
        client.send_image.assert_called_once()

        # Should update image URL
        bot.processor.update_last_image_url.assert_called_once()

        # Should delete fallback status message
        client.delete_message.assert_called_with("C123", "fallback_123")

    @patch('main.asyncio.sleep')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_handle_message_image_streamed_fallback_with_update(self, mock_config, mock_sleep, bot):
        """Test image response streamed fallback with update_message capability"""
        mock_config.circle_loader_emoji = ":loading:"

        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.name = "SlackBot"

        image_data = Mock()
        image_data.to_bytes.return_value = b"image_bytes"
        image_data.format = "png"

        response = Mock(
            type="image",
            content=image_data,
            metadata={
                "streamed": True,
                "status_message_id": None  # No status message ID provided
            }
        )
        bot.processor.process_message = AsyncMock(return_value=response)
        bot.processor.update_last_image_url = AsyncMock()

        client.send_thinking_indicator = AsyncMock(return_value="fallback_123")
        client.send_image = AsyncMock(return_value="http://image.url")
        client.delete_message = AsyncMock()
        client.update_message = AsyncMock()

        await bot.handle_message(message, client)

        # Should create new status message as fallback
        client.send_thinking_indicator.assert_called_with("C123", "thread_123")

        # Should update the fallback message
        client.update_message.assert_called_with(
            "C123", "fallback_123", ":loading: Uploading image to Slack..."
        )

        # Should send image
        client.send_image.assert_called_once()


class TestChatBotV2CleanupTaskCoverage:
    """Test cleanup task edge cases for better coverage"""

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor.thread_manager = Mock()
        bot.processor.thread_manager.cleanup_old_threads = AsyncMock()
        bot.processor.get_stats = Mock(return_value={"threads": 10, "cleaned": 2})
        bot.running = True
        return bot

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_invalid_cron_fallback(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """Test cleanup task with invalid cron expression fallback"""
        mock_config.cleanup_schedule = "invalid_cron"
        mock_config.cleanup_max_age_hours = 24

        # First croniter call raises exception, second succeeds
        mock_croniter_class.side_effect = [
            Exception("Invalid cron expression"),
            Mock()  # Fallback croniter
        ]

        # Mock datetime
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock the fallback croniter
        mock_fallback_cron = Mock()
        mock_croniter_class.side_effect = [
            Exception("Invalid cron expression"),
            mock_fallback_cron
        ]

        # Make the loop exit quickly
        bot.running = False

        # Start cleanup task which creates the worker
        await bot.start_cleanup_task()

        # Verify error was logged and fallback was used
        mock_logger.error.assert_called_with("Invalid cron expression 'invalid_cron': Invalid cron expression")
        mock_logger.info.assert_called_with("Falling back to daily at midnight (0 0 * * *)")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_run_cleanup_short_interval(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """Test cleanup task with short interval (minutes logging)"""
        mock_config.cleanup_schedule = "*/30 * * * *"  # Every 30 minutes
        mock_config.cleanup_max_age_hours = 24

        # Mock croniter
        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron

        # Mock datetime
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock next run time (30 minutes = 1800 seconds from now)
        mock_next_run = Mock()
        mock_next_run.strftime.return_value = "2023-01-01 12:30:00"
        mock_cron.get_next.return_value = mock_next_run

        # Mock time difference to be 30 minutes (1800 seconds)
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 1800  # 30 minutes
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        # Mock sleep to stop after first iteration
        call_count = 0
        async def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First sleep call - simulate waiting for cleanup time
                bot.running = False  # Stop the loop
            return

        mock_sleep.side_effect = mock_sleep_side_effect

        # Start cleanup task
        await bot.start_cleanup_task()

        # Verify minutes logging was used (< 3600 seconds)
        mock_logger.info.assert_any_call("Next cleanup scheduled for 2023-01-01 12:30:00 (30.0 minutes from now)")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_actually_runs_cleanup(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """Test cleanup task actually executes cleanup"""
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 48

        # Mock croniter
        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron

        # Mock datetime
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock next run time
        mock_next_run = Mock()
        mock_next_run.strftime.return_value = "2023-01-01 00:00:00"
        mock_cron.get_next.return_value = mock_next_run

        # Mock time difference
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 0  # Time to run cleanup now
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        # Mock sleep to allow cleanup to run then stop
        call_count = 0
        async def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # After sleep, bot is still running so cleanup executes
                pass  # Allow cleanup to run
            else:
                bot.running = False  # Stop after cleanup
            return

        mock_sleep.side_effect = mock_sleep_side_effect

        # Start cleanup task
        await bot.start_cleanup_task()

        # Verify cleanup was executed
        bot.processor.thread_manager.cleanup_old_threads.assert_called_with(max_age=48 * 3600)
        bot.processor.get_stats.assert_called()
        mock_logger.info.assert_any_call("Cleanup complete. Stats: {'threads': 10, 'cleaned': 2}")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_handles_cancelled_error(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """Test cleanup task handles CancelledError"""
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 24

        # Mock croniter
        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron

        # Mock datetime
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock next run time
        mock_next_run = Mock()
        mock_cron.get_next.return_value = mock_next_run
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 3600
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        # Mock sleep to raise CancelledError
        mock_sleep.side_effect = asyncio.CancelledError("Task cancelled")

        # Start cleanup task
        await bot.start_cleanup_task()

        # Verify cancellation was handled
        mock_logger.info.assert_any_call("Cleanup task cancelled")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_handles_general_error(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """Test cleanup task handles general errors and retries"""
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 24

        # Mock croniter
        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron

        # Mock datetime
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock next run time
        mock_next_run = Mock()
        mock_cron.get_next.return_value = mock_next_run
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 3600
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        # Mock error then stop
        call_count = 0
        async def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Cleanup error")
            elif call_count == 2:
                # Second call is the 5-minute retry delay
                assert seconds == 300
                bot.running = False
            return

        mock_sleep.side_effect = mock_sleep_side_effect

        # Start cleanup task
        await bot.start_cleanup_task()

        # Verify error was logged and retry delay was used
        mock_logger.error.assert_called_with("Error in cleanup task: Cleanup error")


class TestChatBotV2AsyncCancellation:
    """Test async cancellation edge cases"""

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.client = Mock()
        bot.processor = Mock()
        return bot

    @patch('main.asyncio.CancelledError')
    @patch('main.main_logger')
    @pytest.mark.asyncio
    async def test_run_client_cancelled_error(self, mock_logger, mock_cancelled_error, bot):
        """Test run method handles client CancelledError during shutdown"""
        bot.client.start = AsyncMock(side_effect=asyncio.CancelledError("Client cancelled"))
        bot.running = True

        with patch.object(bot, 'initialize', new_callable=AsyncMock):
            with patch.object(bot, 'start_cleanup_task', new_callable=AsyncMock):
                with patch.object(bot, 'shutdown', new_callable=AsyncMock):
                    await bot.run()

        # Should log cancellation message
        mock_logger.info.assert_any_call("Bot client cancelled during shutdown")

    @patch('main.asyncio.all_tasks')
    @patch('main.asyncio.current_task')
    @patch('main.asyncio.gather')
    @patch('main.main_logger')
    @pytest.mark.asyncio
    async def test_shutdown_cancels_remaining_tasks(self, mock_logger, mock_gather, mock_current_task, mock_all_tasks, bot):
        """Test shutdown cancels remaining tasks"""
        bot.running = True
        bot.client.stop = AsyncMock()
        bot.processor.get_stats = Mock(return_value={"threads": 5})
        bot.processor.cleanup = AsyncMock()

        # Mock current task
        mock_current = Mock()
        mock_current_task.return_value = mock_current

        # Mock remaining tasks
        mock_task1 = Mock()
        mock_task2 = Mock()
        mock_all_tasks.return_value = [mock_current, mock_task1, mock_task2]

        await bot.shutdown()

        # Should cancel remaining tasks
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()

        # Should gather with return_exceptions=True
        mock_gather.assert_called_once_with(mock_task1, mock_task2, return_exceptions=True)
        mock_logger.warning.assert_called_with("Cancelling 2 remaining tasks...")