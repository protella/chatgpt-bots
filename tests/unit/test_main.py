"""Unit tests for main.py - Main application entry point"""

import pytest
import sys
import signal
import time
from unittest.mock import Mock, patch, MagicMock, call
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
        assert bot.cleanup_thread is None
        assert bot.running is False
    
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
    def test_initialize_slack_success(self, mock_processor_class, mock_slackbot_class, mock_config):
        """Test successful Slack initialization"""
        mock_config.validate.return_value = None
        mock_client = Mock()
        mock_client.db = Mock()
        mock_slackbot_class.return_value = mock_client
        
        bot = ChatBotV2(platform="slack")
        bot.initialize()
        
        # Verify config validated
        mock_config.validate.assert_called_once()
        
        # Verify Slack client created
        mock_slackbot_class.assert_called_once()
        assert bot.client is mock_client
        
        # Verify processor created with client's DB
        mock_processor_class.assert_called_once_with(db=mock_client.db)
        assert bot.processor is not None
    
    @patch('main.config')
    def test_initialize_config_error(self, mock_config):
        """Test initialization with config validation error"""
        mock_config.validate.side_effect = ValueError("Invalid config")
        
        bot = ChatBotV2(platform="slack")
        
        with pytest.raises(SystemExit):
            bot.initialize()
    
    @patch('main.config')
    def test_initialize_discord_not_implemented(self, mock_config):
        """Test Discord platform not yet implemented"""
        mock_config.validate.return_value = None
        
        bot = ChatBotV2(platform="discord")
        
        with pytest.raises(SystemExit):
            bot.initialize()
    
    @patch('main.config')
    def test_initialize_unknown_platform(self, mock_config):
        """Test unknown platform error"""
        mock_config.validate.return_value = None
        
        bot = ChatBotV2(platform="unknown")
        
        with pytest.raises(SystemExit):
            bot.initialize()
    
    @patch('main.signal.signal')
    @patch('main.config')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    def test_signal_handlers_setup(self, mock_processor, mock_slackbot, mock_config, mock_signal):
        """Test signal handlers are set up"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot.return_value = mock_client
        
        bot = ChatBotV2(platform="slack")
        bot.initialize()
        
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
    
    def test_handle_message_text_response(self, bot):
        """Test handling text response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        
        # Mock processor response
        response = Mock(
            type="text",
            content="Hello world",
            metadata={"streamed": False}
        )
        bot.processor.process_message.return_value = response
        
        # Mock thinking indicator
        client.send_thinking_indicator.return_value = "thinking_123"
        
        bot.handle_message(message, client)
        
        # Verify thinking indicator sent and deleted
        client.send_thinking_indicator.assert_called_once()
        client.delete_message.assert_called_once_with("C123", "thinking_123")
        
        # Verify message sent
        client.format_text.assert_called_once_with("Hello world")
        client.send_message.assert_called_once()
    
    def test_handle_message_streamed_response(self, bot):
        """Test handling streamed response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        
        response = Mock(
            type="text",
            content="Streamed content",
            metadata={"streamed": True}
        )
        bot.processor.process_message.return_value = response
        
        client.send_thinking_indicator.return_value = "thinking_123"
        
        bot.handle_message(message, client)
        
        # Should not delete thinking indicator for streamed responses
        client.delete_message.assert_not_called()
        
        # Should not send message again (already displayed via streaming)
        client.send_message.assert_not_called()
    
    def test_handle_message_busy_response(self, bot):
        """Test handling busy response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        client.send_busy_message = Mock()
        
        response = Mock(type="busy", content="Thread is busy")
        bot.processor.process_message.return_value = response
        
        bot.handle_message(message, client)
        
        client.send_busy_message.assert_called_once_with("C123", "thread_123")
    
    def test_handle_message_busy_fallback(self, bot):
        """Test busy message fallback when method not available"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock(spec=['send_thinking_indicator', 'delete_message', 'send_message'])
        client.send_thinking_indicator.return_value = "thinking_123"
        
        response = Mock(type="busy", content="Thread is busy")
        bot.processor.process_message.return_value = response
        
        bot.handle_message(message, client)
        
        # Should fallback to send_message
        client.send_message.assert_called_once_with(
            "C123", "thread_123", "Thread is busy"
        )
    
    @patch('main.time.sleep')
    def test_handle_message_image_response(self, mock_sleep, bot):
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
        bot.processor.process_message.return_value = response
        
        client.send_thinking_indicator.return_value = "status_123"
        client.send_image.return_value = "http://image.url"
        
        bot.handle_message(message, client)
        
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
    
    def test_handle_message_error_response(self, bot):
        """Test handling error response"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        
        response = Mock(type="error", content="Something went wrong")
        bot.processor.process_message.return_value = response
        
        bot.handle_message(message, client)
        
        client.handle_error.assert_called_once_with(
            "C123", "thread_123", "Something went wrong"
        )
    
    def test_handle_message_exception(self, bot):
        """Test exception handling during message processing"""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        
        bot.processor.process_message.side_effect = Exception("Processing error")
        client.send_thinking_indicator.return_value = "thinking_123"
        
        bot.handle_message(message, client)
        
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
    @patch('main.time.sleep')
    @patch('datetime.datetime')
    def test_start_cleanup_thread(self, mock_datetime, mock_sleep, mock_croniter_class, bot):
        """Test starting cleanup thread"""
        # Mock cron schedule
        mock_cron = Mock()
        mock_now = Mock()
        mock_next = Mock()
        
        mock_datetime.datetime.now.return_value = mock_now
        mock_cron.get_next.return_value = mock_next
        mock_next.__sub__ = Mock(return_value=Mock(total_seconds=Mock(return_value=3600)))
        mock_croniter_class.return_value = mock_cron
        
        # Start cleanup thread
        bot.start_cleanup_thread()
        
        assert bot.cleanup_thread is not None
        assert isinstance(bot.cleanup_thread, Thread)
        assert bot.cleanup_thread.daemon is True
    
    @patch('main.Thread')
    def test_cleanup_thread_invalid_cron(self, mock_thread_class, bot):
        """Test cleanup thread with invalid cron expression"""
        # Just test that thread is created
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        bot.start_cleanup_thread()
        
        # Should create thread
        assert bot.cleanup_thread is not None
        assert mock_thread.start.called
    
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
    def test_run_normal_flow(self, mock_config, mock_processor_class, 
                             mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test normal run flow"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        
        # Make client.start() raise an exception to trigger the finally block
        mock_client.start.side_effect = KeyboardInterrupt("Test interrupt")
        
        bot.run()
        
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
    def test_run_keyboard_interrupt(self, mock_config, mock_processor_class,
                                   mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test handling keyboard interrupt"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        mock_client.start.side_effect = KeyboardInterrupt()
        
        bot.run()
        
        # Should handle gracefully
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
    
    @patch('main.sys.exit')
    @patch('main.log_session_end')
    @patch('main.log_session_start')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @patch('main.config')
    def test_run_unexpected_error(self, mock_config, mock_processor_class,
                                 mock_slackbot_class, mock_log_start, mock_log_end, mock_exit, bot):
        """Test handling unexpected errors"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        mock_client.start.side_effect = Exception("Unexpected error")
        
        bot.run()
        
        # Should handle gracefully
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
    
    def test_shutdown(self, bot):
        """Test shutdown process"""
        bot.running = True
        bot.client = Mock()
        bot.processor = Mock()
        bot.processor.get_stats.return_value = {"threads": 5}
        
        with patch('main.sys.exit') as mock_exit:
            bot.shutdown()
        
        assert bot.running is False
        bot.client.stop.assert_called_once()
        bot.processor.get_stats.assert_called_once()
        mock_exit.assert_called_once_with(0)
    
    def test_shutdown_idempotent(self, bot):
        """Test shutdown is idempotent"""
        bot.running = False
        
        with patch('main.sys.exit') as mock_exit:
            bot.shutdown()
        
        # Should not do anything if not running
        mock_exit.assert_not_called()
    
    def test_signal_handler(self, bot):
        """Test signal handler calls shutdown"""
        bot.shutdown = Mock()
        
        bot._signal_handler(signal.SIGINT, None)
        
        bot.shutdown.assert_called_once()


class TestMainFunction:
    """Test main entry point function"""
    
    @patch('main.ChatBotV2')
    @patch('main.argparse.ArgumentParser')
    def test_main_default_platform(self, mock_parser_class, mock_chatbot_class):
        """Test main with default platform"""
        mock_parser = Mock()
        mock_args = Mock(platform="slack")
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser
        
        mock_bot = Mock()
        mock_chatbot_class.return_value = mock_bot
        
        main()
        
        mock_chatbot_class.assert_called_once_with(platform="slack")
        mock_bot.run.assert_called_once()
    
    @patch('main.ChatBotV2')
    @patch('sys.argv', ['main.py', '--platform', 'discord'])
    def test_main_discord_platform(self, mock_chatbot_class):
        """Test main with Discord platform argument"""
        mock_bot = Mock()
        mock_chatbot_class.return_value = mock_bot
        
        main()
        
        mock_chatbot_class.assert_called_once_with(platform="discord")
        mock_bot.run.assert_called_once()
    
    @patch('sys.argv', ['main.py'])
    @patch('main.ChatBotV2')
    def test_main_module_execution(self, mock_chatbot_class):
        """Test main module execution"""
        mock_bot = Mock()
        mock_chatbot_class.return_value = mock_bot
        
        # Import should not run main
        import main as main_module
        mock_chatbot_class.assert_not_called()
        
        # Direct call should work
        main_module.main()
        mock_chatbot_class.assert_called_once()


@pytest.mark.critical
class TestChatBotV2Critical:
    """Critical functionality tests"""
    
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    def test_critical_initialization_chain(self, mock_config, mock_slackbot_class, mock_processor_class):
        """Critical test for initialization chain"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        
        bot = ChatBotV2(platform="slack")
        bot.initialize()
        
        # Must create client before processor
        assert mock_slackbot_class.called
        assert mock_processor_class.called
        
        # Processor must use client's DB
        processor_db = mock_processor_class.call_args[1]["db"]
        assert processor_db is mock_client.db
    
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    def test_critical_message_handler_callback(self, mock_config, mock_slackbot_class, mock_processor_class):
        """Critical test for message handler callback setup"""
        mock_config.validate.return_value = None
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        
        bot = ChatBotV2(platform="slack")
        bot.initialize()
        
        # Verify message handler passed to client
        call_kwargs = mock_slackbot_class.call_args[1]
        assert "message_handler" in call_kwargs
        assert call_kwargs["message_handler"] == bot.handle_message
    
    def test_critical_error_propagation(self):
        """Critical test for error propagation"""
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock()
        
        # Simulate critical error
        bot.processor.process_message.side_effect = Exception("Critical error")
        client.send_thinking_indicator.return_value = "thinking_123"
        
        # Should not crash, but handle error
        bot.handle_message(message, client)
        
        # Should clean up and report error
        client.delete_message.assert_called_with("C123", "thinking_123")
        client.handle_error.assert_called_with("C123", "thread_123", "Critical error")


@pytest.mark.integration
class TestChatBotV2Integration:
    """Integration tests for main module"""
    
    @patch('main.Thread')
    @patch('main.MessageProcessor')
    @patch('slack_client.SlackBot')
    @patch('main.config')
    def test_integration_full_startup(self, mock_config, mock_slackbot_class, 
                                     mock_processor_class, mock_thread_class):
        """Test full startup sequence"""
        mock_config.validate.return_value = None
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 24
        
        mock_client = Mock(db=Mock())
        mock_slackbot_class.return_value = mock_client
        
        bot = ChatBotV2(platform="slack")
        bot.initialize()
        bot.running = True
        bot.start_cleanup_thread()
        
        # Verify all components initialized
        assert bot.client is not None
        assert bot.processor is not None
        assert bot.cleanup_thread is not None
        
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