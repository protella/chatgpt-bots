"""Unit tests for main.py - Main application entry point"""

import asyncio
import pytest
import signal
from unittest.mock import Mock, patch, AsyncMock

from config import config
from main import ChatBotV2, main


def _startup_db():
    """A db stub that can get through startup.

    ``initialize`` runs the legacy directives→policy migration before any Slack traffic and
    ABORTS if it fails — a channel left with rules in a column nothing reads would be silently
    disobeyed. So every test that initializes needs the call to succeed.
    """
    db = Mock()
    # Startup runs every state migration before Slack traffic and ABORTS if one fails — a channel
    # left behind has something its operator set that this build would silently ignore. So each of
    # them has to succeed for a test to get through initialize().
    for name in ("migrate_channel_directives_to_policy_async",
                 "migrate_participation_levels_to_binary_async",
                 "migrate_participation_prefs_to_policy_async"):
        setattr(db, name, AsyncMock(return_value=(0, 0)))
    # Startup also establishes the outbound-receipts epoch (fatal if it cannot be read back)
    # and reconciles dead-session rows before any Slack traffic.
    db.set_meta_if_absent_async = AsyncMock(return_value=True)
    db.get_meta_async = AsyncMock(return_value="1000.000000")
    db.finalize_dead_session_receipts_async = AsyncMock(return_value=0)
    db.get_pending_shares_async = AsyncMock(return_value=[])
    return db


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
    
    def test_init_with_other_platform(self):
        """Test initialization stores an arbitrary platform string"""
        bot = ChatBotV2(platform="matrix")
        
        assert bot.platform == "matrix"
    
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
        mock_client.db = _startup_db()
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
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @pytest.mark.asyncio
    async def test_boot_starts_the_admission_tokenizer_without_waiting_for_it(
            self, mock_processor_class, mock_slackbot_class, mock_config):
        """The cold-cache fetch belongs to boot, not to the first channel turn — but boot must not
        WAIT for a network round trip either, or a blackholed egress holds the process down."""
        import token_counter

        mock_config.validate.return_value = None
        mock_client = Mock()
        mock_client.db = _startup_db()
        mock_slackbot_class.return_value = mock_client

        with patch.object(token_counter, "wait_for_admission_encoder") as warm:
            await ChatBotV2(platform="slack").initialize()

        warm.assert_called_once_with(timeout=0)


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
    async def test_initialize_unsupported_platform_exits(self, mock_config):
        """Test unsupported platform names exit with an error"""
        mock_config.validate.return_value = None

        bot = ChatBotV2(platform="matrix")

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
        mock_client = Mock(db=_startup_db())
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
        
        # F11: main.py must NOT pre-format. send_message owns the single conversion
        # point (format_text is not idempotent — a second pass renders bold as italic),
        # so raw content flows straight through.
        client.format_text.assert_not_called()
        client.send_message.assert_called_once()
        assert client.send_message.call_args.args[2] == "Hello world"
    
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
    async def test_handle_message_queued_response_posts_nothing(self, bot):
        """Phase Q: a queued response posts NO message (no busy reply — retired)."""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock(spec=['send_thinking_indicator', 'delete_message', 'send_message'])
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()

        response = Mock(type="queued", content="", metadata={})
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        # Nothing posted; the (raced) thinking indicator is cleaned up.
        client.send_message.assert_not_called()
        client.delete_message.assert_called_once_with("C123", "thinking_123")

    @pytest.mark.asyncio
    async def test_handle_message_skips_indicator_when_conversation_busy(self, bot):
        """Phase Q: no thinking indicator flashes for a message that will queue."""
        message = Mock(channel_id="C123", thread_id="thread_123")
        client = Mock(spec=['send_thinking_indicator', 'delete_message', 'send_message'])
        client.send_thinking_indicator = AsyncMock(return_value="thinking_123")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()

        bot.processor.thread_manager = Mock()
        bot.processor.thread_manager.is_thread_processing = Mock(return_value=True)
        response = Mock(type="queued", content="", metadata={})
        bot.processor.process_message = AsyncMock(return_value=response)

        await bot.handle_message(message, client)

        client.send_thinking_indicator.assert_not_called()
        client.send_message.assert_not_called()
        client.delete_message.assert_not_called()
    
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

        # The error notice now carries the stale-send lease: it is terminal text, and on a turn
        # with no thinking surface it is the room's only word from us.
        assert client.handle_error.await_args.args == (
            "C123", "thread_123", "Something went wrong")
        assert "lease" in client.handle_error.await_args.kwargs
    
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

        # Should send a fixed friendly notice — the raw exception text must
        # never reach Slack (it stays in the logs)
        client.handle_error.assert_called_once()
        args = client.handle_error.call_args[0]
        assert args[0] == "C123"
        assert args[1] == "thread_123"
        assert "Processing error" not in args[2]
        assert "Something Went Wrong" in args[2]


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

    def test_blocking_db_ops_offloaded_to_thread(self):
        """F29: the ambient-artifact sweep and the DB backup are blocking (conn.backup can
        take hundreds of ms); they must run via asyncio.to_thread, not on the event loop."""
        import inspect
        import re
        import main

        src = inspect.getsource(main.ChatBotV2.start_cleanup_task)
        assert re.search(
            r"to_thread\(\s*self\.processor\.db\.delete_expired_ambient_artifacts", src), \
            "ambient-artifact sweep must be offloaded via asyncio.to_thread"
        assert re.search(
            r"to_thread\(\s*self\.processor\.db\.backup_database", src), \
            "database backup must be offloaded via asyncio.to_thread"


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
        mock_client = Mock(db=_startup_db())
        mock_slackbot_class.return_value = mock_client

        # No MCP servers — otherwise run() would try to create the health-probe
        # task from a MagicMock coroutine
        mock_processor_class.return_value.mcp_manager.has_mcp_servers.return_value = False

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
        mock_client = Mock(db=_startup_db())
        mock_slackbot_class.return_value = mock_client
        mock_processor_class.return_value.mcp_manager.has_mcp_servers.return_value = False
        mock_client.start.side_effect = KeyboardInterrupt()

        await bot.run()

        # Should handle gracefully — a signal/Ctrl-C stop is exit 0, never sys.exit(1) (F31).
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
        mock_exit.assert_not_called()
    
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
        mock_client = Mock(db=_startup_db())
        mock_slackbot_class.return_value = mock_client
        mock_processor_class.return_value.mcp_manager.has_mcp_servers.return_value = False
        mock_client.start.side_effect = Exception("Unexpected error")

        await bot.run()

        # F31: graceful shutdown still runs, but an UNEXPECTED fatal error must then
        # exit non-zero so a supervisor sees the failure (not a clean exit 0).
        mock_log_start.assert_called_once()
        mock_log_end.assert_called_once()
        mock_exit.assert_called_once_with(1)

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
        mock_client = Mock(db=_startup_db())
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
        mock_client = Mock(db=_startup_db())
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

        # Should clean up and report a sanitized error (raw text stays in logs)
        client.delete_message.assert_called_with("C123", "thinking_123")
        client.handle_error.assert_called_once()
        args = client.handle_error.call_args[0]
        assert args[0] == "C123"
        assert args[1] == "thread_123"
        assert "Critical error" not in args[2]


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

        mock_client = Mock(db=_startup_db())
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
        from main import main as main_func

        # Verify exports
        assert hasattr(main, 'ChatBotV2')
        assert hasattr(main, 'main')
        assert callable(main_func)


class TestChatBotV2CleanupTaskCoverage:
    """Test cleanup task edge cases for better coverage"""

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor.thread_manager = Mock()
        bot.processor.thread_manager.cleanup_old_threads = AsyncMock()
        bot.processor.db.cleanup_old_modal_sessions_async = AsyncMock()
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
        await bot.cleanup_task  # let the background worker run to completion

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
        await bot.cleanup_task  # let the background worker run to completion

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
        await bot.cleanup_task  # let the background worker run to completion

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
        await bot.cleanup_task  # let the background worker run to completion

        # Verify cancellation was handled
        mock_logger.info.assert_any_call("Cleanup task cancelled")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_takes_database_backup(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """The scheduled cleanup is the only steady-state backup trigger — without
        it backup_database() is called by the one-time migrations and never again,
        despite the documented 'automatic backups with 7-day retention'."""
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 48
        mock_config.tool_usage_retention_days = 30

        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron
        mock_datetime.datetime.now.return_value = Mock()

        mock_next_run = Mock()
        mock_next_run.strftime.return_value = "2023-01-01 00:00:00"
        mock_cron.get_next.return_value = mock_next_run
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 0
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        call_count = 0

        async def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                bot.running = False
            return

        mock_sleep.side_effect = mock_sleep_side_effect

        await bot.start_cleanup_task()
        await bot.cleanup_task

        # Untagged so cleanup_old_backups()'s 7-day retention prunes it
        bot.processor.db.backup_database.assert_called_once_with()
        mock_logger.info.assert_any_call(
            "Scheduled database backup complete (7-day retention)")

    @patch('main.config')
    @patch('main.main_logger')
    @patch('croniter.croniter')
    @patch('main.asyncio.sleep')
    @patch('datetime.datetime')
    @pytest.mark.asyncio
    async def test_cleanup_task_survives_backup_failure(self, mock_datetime, mock_sleep, mock_croniter_class, mock_logger, mock_config, bot):
        """A failing backup must never kill the cleanup worker or the bot."""
        mock_config.cleanup_schedule = "0 0 * * *"
        mock_config.cleanup_max_age_hours = 48
        mock_config.tool_usage_retention_days = 30
        bot.processor.db.backup_database.side_effect = OSError("No space left on device")

        mock_cron = Mock()
        mock_croniter_class.return_value = mock_cron
        mock_datetime.datetime.now.return_value = Mock()

        mock_next_run = Mock()
        mock_next_run.strftime.return_value = "2023-01-01 00:00:00"
        mock_cron.get_next.return_value = mock_next_run
        mock_time_diff = Mock()
        mock_time_diff.total_seconds.return_value = 0
        mock_next_run.__sub__ = Mock(return_value=mock_time_diff)

        call_count = 0

        async def mock_sleep_side_effect(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                bot.running = False
            return

        mock_sleep.side_effect = mock_sleep_side_effect

        await bot.start_cleanup_task()
        await bot.cleanup_task  # must not raise

        mock_logger.error.assert_any_call(
            "Scheduled database backup FAILED: No space left on device")
        # Cleanup itself still completed — the failure was contained
        bot.processor.get_stats.assert_called()
        mock_logger.info.assert_any_call("Cleanup complete. Stats: {'threads': 10, 'cleaned': 2}")

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
        await bot.cleanup_task  # let the background worker run to completion

        # Verify error was logged and retry delay was used
        mock_logger.error.assert_called_with("Error in cleanup task: Cleanup error")


class TestChatBotV2AsyncCancellation:
    """Test async cancellation edge cases"""

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.client = Mock()
        bot.processor = Mock()
        # No MCP servers — keeps run() from spawning a health-probe task
        # off a MagicMock coroutine
        bot.processor.mcp_manager.has_mcp_servers.return_value = False
        return bot

    @patch('main.main_logger')
    @pytest.mark.asyncio
    async def test_run_client_cancelled_error(self, mock_logger, bot):
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
    @patch('main.asyncio.gather', new_callable=AsyncMock)
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

class TestShutdownQuiescesTurns:
    """Spec §5: a receipt can only be written while the queue is open.

    Shutdown used to drain background tasks and close the receipt service while admitted turns
    were still running — so a turn that posted its answer during shutdown had its registration
    AND its settle refused. The message sat in Slack with nothing claiming it, and the rebuilt
    stream would never contain it.
    """

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.client = Mock()
        bot.client.stop = AsyncMock()
        bot.processor = Mock()
        bot.processor.get_stats = Mock(return_value={})
        bot.processor.cleanup = AsyncMock()
        bot.processor.thread_manager = None
        bot.processor.ambient_service = None
        bot.processor.drain_background_tasks = AsyncMock()
        bot.running = True
        return bot

    @pytest.mark.asyncio
    async def test_a_message_arriving_after_shutdown_is_refused_before_admission(self, bot):
        bot._admitting = False
        bot._watermarks = Mock()
        message = Mock(channel_id="C123", thread_id="10.0")

        await bot.handle_message(message, Mock())

        # No lease, no watermark movement, no turn.
        bot._watermarks.begin_turn.assert_not_called()
        bot.processor.process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_a_turn_it_already_admitted(self, bot):
        posted = []
        running = asyncio.Event()

        async def _turn():
            running.set()
            await asyncio.sleep(0.05)
            posted.append("reply")

        task = asyncio.ensure_future(_turn())
        bot.active_turns.add(task)
        await running.wait()

        await bot.shutdown()

        assert posted == ["reply"], "shutdown closed up underneath a running turn"
        assert bot._admitting is False

    @pytest.mark.asyncio
    async def test_shutdown_closes_ticket_issuance_only_once_ingress_is_quiet(self, bot):
        """Spec §1's shutdown contract, which is ORDERING and nothing else.

        The ticketless interval used to be structural: issuance closed, THEN the client stopped, so
        every callback Bolt was still dispatching admitted events that could take no ticket, and a
        failed index write for one of them could only be logged as lost. The order below removes the
        interval instead of accounting for it — ingress is proven quiet first, so the last event
        anything can admit still gets a ticket, and only then does issuance close. The worker drains
        after that (the database it repairs through is still open) and late receipts after that.
        """
        from message_processor import outbound_receipts
        from slack_client import admission_watermark
        from slack_client.event_handlers import registration

        order = []
        original_quiesce = bot._quiesce_turns
        original_callbacks = outbound_receipts.drain_channel_post_callbacks

        async def _quiesce(*a, **k):
            order.append("quiesce_turns")
            return await original_quiesce(*a, **k)

        async def _callbacks(*a, **k):
            order.append("drain_callbacks")
            return await original_callbacks(*a, **k)

        async def _drain_ingress(*a, **k):
            order.append("ingress_quiet")
            return registration.IngressDrain()

        bot.receipt_service = Mock()
        bot.receipt_service.shutdown = AsyncMock()
        bot.receipt_service.drain_late_arrivals = AsyncMock(
            side_effect=lambda *a, **k: order.append("late_receipts"))
        bot._quiesce_turns = _quiesce
        with patch.object(admission_watermark, "close_issuance",
                          side_effect=lambda: order.append("close_issuance")), \
             patch.object(outbound_receipts, "drain_channel_post_callbacks",
                          new=AsyncMock(side_effect=_callbacks)), \
             patch.object(registration, "drain_ingress_callbacks",
                          new=AsyncMock(side_effect=_drain_ingress)), \
             patch.object(admission_watermark, "shutdown",
                          new=AsyncMock(side_effect=lambda *a, **k: order.append("drain_worker"))):
            bot.client.stop = AsyncMock(side_effect=lambda: order.append("ingress_stop"))
            await bot.shutdown()

        assert order == ["quiesce_turns", "drain_callbacks", "ingress_stop", "ingress_quiet",
                         "close_issuance", "drain_worker", "late_receipts"]

    @pytest.mark.asyncio
    async def test_the_shutdown_phase_order_is_unchanged(self, bot):
        """T5. The coordinator used to stop between the late-receipt drain and DB teardown, and
        removing it must leave the FXC-pinned sequence byte-for-byte in place. Asserted as the
        WHOLE list rather than as an absence, because a removal that also reordered its
        neighbours would pass every "is it gone" check and still break the contract."""
        from message_processor import outbound_receipts
        from slack_client import admission_watermark
        from slack_client.event_handlers import registration

        order = []
        bot.receipt_service = Mock()
        bot.receipt_service.shutdown = AsyncMock()
        bot.receipt_service.drain_late_arrivals = AsyncMock(
            side_effect=lambda *a, **k: order.append("drain_late_arrivals"))
        with patch.object(admission_watermark, "close_issuance",
                          side_effect=lambda: order.append("close_issuance")), \
             patch.object(outbound_receipts, "drain_channel_post_callbacks",
                          new=AsyncMock()), \
             patch.object(registration, "drain_ingress_callbacks",
                          new=AsyncMock(side_effect=lambda *a, **k: (
                              order.append("drain_ingress_callbacks"),
                              registration.IngressDrain())[1])), \
             patch.object(admission_watermark, "shutdown",
                          new=AsyncMock(
                              side_effect=lambda *a, **k: order.append("watermark.shutdown"))):
            bot.client.stop = AsyncMock(side_effect=lambda: order.append("client.stop"))
            await bot.shutdown()

        assert order == ["client.stop", "drain_ingress_callbacks", "close_issuance",
                         "watermark.shutdown", "drain_late_arrivals"]
        assert not hasattr(bot, "snapshot_coordinator")

    @pytest.mark.asyncio
    async def test_a_callback_straddling_client_stop_is_drained_before_the_worker_stops(self, bot):
        """r2-5. `await client.stop()` returning is NOT quiescence: Bolt dispatches each event as
        its own task, and production stop force-marks sessions closed and abandons the handler's
        own close. The previous test replaced `client.stop` with an atomic mock, so it could not
        see that. Here a REAL callback is mid-flight when stop returns, and the guarantee is that
        it finishes before the index retry worker is taken away."""
        from slack_client import admission_watermark
        from slack_client.event_handlers import registration

        order = []
        released = asyncio.Event()

        @registration.track_ingress
        async def _callback():
            order.append("callback_start")
            await released.wait()
            order.append("callback_end")

        async def _stop():
            # Slack handed Bolt an event a moment before the socket went down. NOTHING yields
            # before stop() returns: the dispatch exists as a task but has not reached the wrapper,
            # which is precisely the state the old barrier read as "quiet". The test used to yield
            # here to make the callback visible — the barrier's own grace recheck is what has to
            # see it now.
            asyncio.ensure_future(_callback())
            order.append("ingress_stop")
            asyncio.get_running_loop().call_later(0.01, released.set)

        bot.client.stop = AsyncMock(side_effect=_stop)
        registration.ingress.reset()
        try:
            with patch.object(admission_watermark, "shutdown",
                              new=AsyncMock(side_effect=lambda *a, **k: order.append(
                                  "drain_worker"))):
                await bot.shutdown()
        finally:
            registration.ingress.reset()

        assert order == ["ingress_stop", "callback_start", "callback_end", "drain_worker"], order

    @pytest.mark.asyncio
    async def test_a_wedged_callback_is_named_and_cancelled_rather_than_waited_on_forever(
            self, bot, monkeypatch, caplog):
        """The bound is honest in both directions: shutdown does not hang, and it does not claim a
        drain it did not get. The straggler is named by its listener and cancelled — the next thing
        shutdown does is take away the worker that would have persisted its observation, so leaving
        it running would abandon a ticket pending with nothing left to resolve it. Cancellation
        WORKING is not the same as quiet being granted, and the log says which one happened."""
        import logging

        from slack_client.event_handlers import registration

        monkeypatch.setattr(config, "ingress_drain_timeout_seconds", 0.01, raising=False)

        @registration.track_ingress
        async def handle_wedged_listener():
            await asyncio.sleep(60)

        registration.ingress.reset()
        task = asyncio.ensure_future(handle_wedged_listener())
        await asyncio.sleep(0)
        try:
            with caplog.at_level(logging.CRITICAL):
                await bot.shutdown()
        finally:
            task.cancel()
            registration.ingress.reset()

        critical = "\n".join(r.getMessage() for r in caplog.records
                             if r.levelno >= logging.CRITICAL)
        assert "did not go quiet within the drain deadline" in critical
        assert "survived cancellation" not in critical, \
            "cancellation worked; shutdown must not report a residual it does not have"
        assert "handle_wedged_listener" in critical
        assert task.cancelled(), "the straggler was left running past the drain"

    @pytest.mark.asyncio
    async def test_a_file_deleted_arriving_after_receipts_close_still_lands(self, bot, tmp_path,
                                                                          monkeypatch):
        """r2-6. Receipt shutdown happens while the `file_deleted` listener is still registered, so
        a transient DB failure there used to try to retain its cleanup in a CLOSED queue and be
        refused — restoring the one-shot deletion loss the P1 lattice was built to close. Slack
        never re-sends the event, so "refused" means the pending row survives forever."""
        from database import DatabaseManager
        from message_processor import outbound_receipts as orx

        monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
        db = DatabaseManager(platform="slack")
        orx.reset_service()
        service = orx.install_service(db)
        bot.receipt_service = service
        await db.record_pending_share_async("T1", "C0BKX77NU66", "F1", "turn-1", None)
        reads = {"n": 0}
        real_read = db.get_pending_shares_async

        async def _flaky_read():
            reads["n"] += 1
            if reads["n"] == 1:
                raise RuntimeError("WAL writer contended")
            return await real_read()

        monkeypatch.setattr(db, "get_pending_shares_async", _flaky_read)
        try:
            async def _stop():
                # The listener fires here: receipts have already closed, ingress has not.
                await orx.delete_pending_shares_for_file(db, "F1")

            bot.client.stop = AsyncMock(side_effect=_stop)
            await bot.shutdown()

            rows = await real_read()
            assert not [r for r in rows or [] if r.get("file_id") == "F1"], \
                "the pending row for a deleted file outlived the process"
        finally:
            orx.reset_service()
            db.conn.close()

    @pytest.mark.asyncio
    async def test_a_callback_that_started_before_stop_still_gets_a_ticket_and_is_retained(
            self, bot, monkeypatch, caplog):
        """r3-8. The injection codex asked for, with the answer it asked for.

        A Bolt callback already in flight when the socket closes admits its event AFTER `stop()` has
        returned. Under the old order issuance was already shut, so `_admit` handed it no ticket and
        a failed index write could only be logged as permanently lost — an "accepted" loss on a path
        where nothing had actually accepted anything. Issuance now closes behind the barrier, so the
        event gets a ticket, its failure is RETAINED with a replay, and the retry worker repairs it
        while the database is still open.
        """
        import logging

        from slack_client import admission_watermark
        from slack_client.event_handlers import activity_index
        from slack_client.event_handlers import registration
        from slack_client.event_handlers.registration import _admit

        monkeypatch.setattr(admission_watermark.logger, "propagate", True)
        monkeypatch.setattr(activity_index, "_FEED_RETRY_BASE_SECONDS", 0)

        writes = {"n": 0}

        class _Host:
            self_team_id = "T1"
            bot_user_id = "UBOT"
            bot_id = "BBOT"
            app_id = "A1"

            def __init__(self):
                self.db = Mock()
                self.db.seed_channel_coverage_async = AsyncMock()
                self.db.record_thread_activity_async = AsyncMock(side_effect=self._write)

            async def _write(self, *a, **k):
                # Contended WAL for every in-line attempt; the background repair is what lands.
                writes["n"] += 1
                if writes["n"] <= activity_index._FEED_ATTEMPTS:
                    raise RuntimeError("WAL writer contended")

            def is_own_message(self, message):
                return False

        host = _Host()
        event = {"type": "message", "channel": "C1", "channel_type": "channel", "user": "U1",
                 "ts": "500.000100", "thread_ts": "499.000100", "text": "a late reply",
                 "event_ts": "500.000100"}
        seen = {}
        released = asyncio.Event()

        @registration.track_ingress
        async def _late_callback():
            await released.wait()
            seen["ticket"] = _admit(host, event)
            await activity_index.feed_thread_activity_index(host, event, ticket=seen["ticket"])

        async def _stop():
            # Bolt dispatched this before the socket went down; it admits after stop() returns.
            asyncio.ensure_future(_late_callback())
            asyncio.get_running_loop().call_soon(released.set)

        bot.client.stop = AsyncMock(side_effect=_stop)
        registration.ingress.reset()
        # An earlier shutdown in this module closed the module singleton's issuance for good.
        admission_watermark.watermark.reset()
        try:
            with caplog.at_level(logging.CRITICAL,
                                 logger=admission_watermark.logger.name):
                await bot.shutdown()
        finally:
            registration.ingress.reset()
            admission_watermark.watermark.reset()

        assert seen["ticket"] is not None, \
            "issuance closed before ingress was quiet — the ticketless interval is back"
        assert admission_watermark.watermark.ticket_state(seen["ticket"]) == \
            admission_watermark.REPAIRED, "the retained observation was never replayed"
        critical = "\n".join(r.getMessage() for r in caplog.records
                            if r.levelno >= logging.CRITICAL)
        assert "permanently lost" not in critical and "lost at shutdown" not in critical, critical

    @pytest.mark.asyncio
    async def test_an_overrunning_turn_is_cancelled_rather_than_holding_shutdown_open(self, bot):
        running = asyncio.Event()
        outcome = []

        async def _wedged():
            running.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                outcome.append("cancelled")
                raise

        task = asyncio.ensure_future(_wedged())
        bot.active_turns.add(task)
        await running.wait()

        await asyncio.wait_for(bot._quiesce_turns(timeout=0.05), timeout=5)

        assert outcome == ["cancelled"]
        assert task.done()

    @pytest.mark.asyncio
    async def test_a_turn_straddling_shutdown_still_gets_its_receipt(self, bot, tmp_path,
                                                                     monkeypatch):
        """The whole point of the ordering: quiesce, THEN close the queue."""
        from database import DatabaseManager
        from message_processor import outbound_receipts as orx

        monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
        db = DatabaseManager(platform="slack")
        orx.reset_service()
        service = orx.install_service(db)
        bot.receipt_service = service
        try:
            running = asyncio.Event()

            async def _turn():
                ledger = orx.ReceiptLedger("s:1", "T1", "C0BKX77NU66")
                running.set()
                await asyncio.sleep(0.05)          # the model is still writing
                await ledger.note_post("100.0")    # …and the answer lands mid-shutdown
                await orx.settle_ledger(ledger)

            task = asyncio.ensure_future(_turn())
            bot.active_turns.add(task)
            await running.wait()

            await bot.shutdown()

            row = await db.get_receipt_async("T1", "C0BKX77NU66", "100.0")
            assert row is not None, "the turn's own words were never claimed"
            assert row["state"] == "finalized"
        finally:
            orx.reset_service()
            db.conn.close()

    @pytest.mark.asyncio
    async def test_channel_intros_are_drained_before_receipts_close(self, bot):
        """Spec §5: the intro is a detached CLIENT-owned producer of real prose. Slack ingress
        stops at the very end of shutdown, long after the receipt queue — so the intro drain has
        to happen up front or its registration is refused."""
        from message_processor import outbound_receipts as orx

        order = []
        bot.client.drain_channel_intros = AsyncMock(
            side_effect=lambda *a, **k: order.append("intros"))
        bot.receipt_service = Mock()
        bot.receipt_service.shutdown = AsyncMock(
            side_effect=lambda *a, **k: order.append("receipts"))
        bot.client.stop = AsyncMock(side_effect=lambda *a, **k: order.append("client"))

        async def _drain_callbacks(*a, **k):
            order.append("callbacks")

        with patch.object(orx, "drain_channel_post_callbacks", _drain_callbacks):
            await bot.shutdown()

        # Every producer of durable channel prose is off the field before receipts close, and
        # Socket Mode — which outlives all of it — stops last.
        assert order == ["intros", "callbacks", "receipts", "client"]

    @pytest.mark.asyncio
    async def test_a_slow_protected_post_holds_the_whole_teardown_back(self, bot):
        """The real gate, not a stub. A post Slack may already have taken has to finish
        registering before receipts close, the database goes away, and the blanket task-cancel
        at the bottom of shutdown runs — so the drain returning IS the guarantee."""
        from message_processor import outbound_receipts as orx

        order = []
        release = asyncio.Event()
        bot.client.drain_channel_intros = AsyncMock()
        bot.receipt_service = Mock()
        bot.receipt_service.shutdown = AsyncMock(
            side_effect=lambda *a, **k: order.append("receipts"))
        bot.client.stop = AsyncMock(side_effect=lambda *a, **k: order.append("client"))

        async def _slow_pair():
            await release.wait()
            order.append("registered")

        task = asyncio.ensure_future(_slow_pair())
        orx.get_channel_post_gate().protect(task)

        seen_while_stalled = []

        async def _release_once_shutdown_is_waiting():
            await asyncio.sleep(0.05)
            # Shutdown is parked on the protected pair: nothing past the gate has run.
            seen_while_stalled.append(list(order))
            release.set()

        releaser = asyncio.ensure_future(_release_once_shutdown_is_waiting())
        # Awaited directly, never wrapped: shutdown's blanket cancel at the bottom spares only
        # the CURRENT task, so a wrapped call would cancel this test out from under itself.
        await bot.shutdown()
        await releaser

        assert seen_while_stalled == [[]], "receipts closed before the post had registered"
        assert order == ["registered", "receipts", "client"]
        assert task.done()

    @pytest.mark.asyncio
    async def test_a_callback_gate_failure_does_not_stop_shutdown(self, bot):
        from message_processor import outbound_receipts as orx

        async def _boom(*a, **k):
            raise RuntimeError("gate broken")

        with patch.object(orx, "drain_channel_post_callbacks", _boom):
            await bot.shutdown()

        # Shutdown ran all the way to the bottom rather than stopping at the broken gate.
        bot.client.stop.assert_awaited_once()
        bot.processor.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_client_without_an_intro_drain_is_not_an_error(self, bot):
        bot.client = Mock(spec=["stop"])
        bot.client.stop = AsyncMock()

        await bot.shutdown()

        bot.client.stop.assert_awaited_once()


class TestIngressBarrier:
    """r3-9: what makes `drain_ingress_callbacks` a BARRIER and not a hint.

    Every test here fails against the counter it replaced, and each one is a specific way that
    counter could report "ingress is quiet" while it wasn't: a zero read before a dispatch had
    reached the wrapper, a second callback riding out on the first one's idle edge, and a socket
    close that was still able to hand Bolt events.
    """

    @pytest.fixture(autouse=True)
    def clean_tracker(self):
        from slack_client.event_handlers import registration
        registration.ingress.reset()
        yield registration.ingress
        registration.ingress.reset()

    @pytest.mark.asyncio
    async def test_a_dispatch_that_has_not_reached_the_wrapper_is_not_quiet(self, clean_tracker):
        """The count is zero and the callback exists. Bolt creates each callback as its own task,
        so between `ensure_future` and the wrapper's first line the old barrier saw nothing at all
        and returned immediately — the grace recheck is what closes that."""
        from slack_client.event_handlers import registration

        finished = []

        @registration.track_ingress
        async def handle_scheduled():
            await asyncio.sleep(0.02)
            finished.append("done")

        asyncio.ensure_future(handle_scheduled())      # NOT yielded to
        assert clean_tracker.in_flight == 0

        assert await registration.drain_ingress_callbacks(timeout=2.0) == \
            registration.IngressDrain()
        assert finished == ["done"], "the barrier returned before the callback ever ran"

    @pytest.mark.asyncio
    async def test_a_second_callback_cannot_ride_out_on_the_first_ones_idle_edge(self,
                                                                                clean_tracker):
        """A leaves (the idle edge is set), B enters, and the waiter resumes. The old wait was
        already resolved and returned 0 without re-reading the count, so B — mid-DB-write — was
        invisible to a shutdown about to close the database."""
        from slack_client.event_handlers import registration

        order = []
        a_leaving = asyncio.Event()

        @registration.track_ingress
        async def handle_first():
            order.append("a_start")
            await a_leaving.wait()
            order.append("a_end")

        @registration.track_ingress
        async def handle_second():
            order.append("b_start")
            await asyncio.sleep(0.05)
            order.append("b_end")

        async def _first_then_second():
            await handle_first()
            # Same loop pass as A's release: B enters before the waiter is scheduled again.
            asyncio.ensure_future(handle_second())

        asyncio.ensure_future(_first_then_second())
        await asyncio.sleep(0)                          # let A enter, as Bolt would
        assert clean_tracker.in_flight == 1
        asyncio.get_running_loop().call_soon(a_leaving.set)

        assert await registration.drain_ingress_callbacks(timeout=2.0) == \
            registration.IngressDrain()
        assert order == ["a_start", "a_end", "b_start", "b_end"], order

    @pytest.mark.asyncio
    async def test_a_close_that_can_still_dispatch_is_waited_out_first(self, clean_tracker):
        """The socket-mode close is registered as a dispatcher, so quiescence is not even
        considered until it has finished — and a callback it delivers on its way out is caught."""
        from slack_client.event_handlers import registration

        order = []

        @registration.track_ingress
        async def handle_last_event():
            order.append("callback_start")
            await asyncio.sleep(0.02)
            order.append("callback_end")

        async def _closing_socket():
            await asyncio.sleep(0.03)
            order.append("close_done")
            asyncio.ensure_future(handle_last_event())  # Bolt's final delivery

        close_task = asyncio.ensure_future(_closing_socket())
        clean_tracker.track_dispatcher(close_task, name="the socket-mode handler close")

        assert await registration.drain_ingress_callbacks(timeout=2.0) == \
            registration.IngressDrain()
        assert order == ["close_done", "callback_start", "callback_end"], order

    @pytest.mark.asyncio
    async def test_a_wedged_dispatcher_is_cancelled_not_merely_reported(self, clean_tracker,
                                                                       caplog):
        """r4-1. A dispatcher that outlives the deadline is the ticketless interval itself: it is
        still able to hand Bolt an event, and the caller's very next act closes ticket issuance and
        takes away the repair worker. Naming it in the log left it running. It is CANCELLED here,
        and the verdict says quiet was taken rather than granted."""
        import logging

        from slack_client.event_handlers import registration

        async def _wedged_close():
            await asyncio.sleep(60)

        close_task = asyncio.ensure_future(_wedged_close())
        clean_tracker.track_dispatcher(close_task, name="the socket-mode handler close")
        try:
            with caplog.at_level(logging.CRITICAL):
                drain = await registration.drain_ingress_callbacks(timeout=0.05)
        finally:
            close_task.cancel()

        assert drain.gave_up is True, "a deadline the barrier missed must never read as quiet"
        assert drain.survived == (), "the close died on cancellation; nothing outlived it"
        assert close_task.cancelled(), "the close was left able to dispatch past the drain"
        critical = "\n".join(r.getMessage() for r in caplog.records
                             if r.levelno >= logging.CRITICAL)
        assert "socket-mode handler close" in critical
        assert "cancelling it" in critical

    @pytest.mark.asyncio
    async def test_a_callback_that_resists_cancellation_is_named_and_not_waited_on(
            self, clean_tracker, caplog, monkeypatch):
        """r4-1/r4-2. The post-cancel wait is bounded too, or one callback that swallows
        CancelledError hangs shutdown for good. That callback is a programming error, so it is named
        CRITICAL and reported as a survivor — the one residual the barrier cannot take away."""
        import logging

        from slack_client.event_handlers import registration

        monkeypatch.setattr(type(clean_tracker), "_CANCEL_GRACE", 0.05, raising=False)
        release = asyncio.Event()

        @registration.track_ingress
        async def handle_stubborn_listener():
            while not release.is_set():
                try:
                    await asyncio.wait_for(release.wait(), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    continue

        task = asyncio.ensure_future(handle_stubborn_listener())
        await asyncio.sleep(0)
        try:
            with caplog.at_level(logging.CRITICAL):
                drain = await registration.drain_ingress_callbacks(timeout=0.01)
        finally:
            release.set()
            await asyncio.wait([task], timeout=2.0)

        assert drain.gave_up is True
        assert drain.survived == ("handle_stubborn_listener",), drain
        critical = "\n".join(r.getMessage() for r in caplog.records
                             if r.levelno >= logging.CRITICAL)
        assert "survived cancellation" in critical
        assert "handle_stubborn_listener" in critical

    @pytest.mark.asyncio
    async def test_a_callback_handed_over_while_cancellation_unwinds_is_not_called_quiet(
            self, clean_tracker):
        """r5-1. Cancelling once and reporting the survivors of THAT snapshot was still a lie. A
        dispatcher unwinding from CancelledError can hand Bolt one final event, and that callback
        entered after the snapshot was taken — so it appeared in neither the victim list nor the
        verdict, and the drain reported nothing survived while a callback was mid-write. The cancel
        phase re-proves the zero, so the late callback is absorbed instead of ignored."""
        from slack_client.event_handlers import registration

        entered = []
        tasks: list = []

        @registration.track_ingress
        async def handle_parting_event():
            entered.append("entered")
            await asyncio.sleep(60)

        async def _wedged_close():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # Bolt's last delivery, made while the close unwinds — i.e. strictly after
                # `_give_up` listed its victims.
                tasks.append(asyncio.ensure_future(handle_parting_event()))
                raise

        close_task = asyncio.ensure_future(_wedged_close())
        clean_tracker.track_dispatcher(close_task, name="the socket-mode handler close")

        drain = await registration.drain_ingress_callbacks(timeout=0.05)

        assert entered == ["entered"], "the late callback never ran; the scenario did not happen"
        assert drain.gave_up is True
        # The invariant: an empty `survived` is a claim that nothing is left running, so the count
        # has to agree with it.
        assert drain.survived == (), drain
        assert clean_tracker.in_flight == 0, "reported quiet with a callback still in flight"
        assert tasks[0].done(), "the late callback outlived the drain that called itself quiet"

    @pytest.mark.asyncio
    async def test_a_parting_callback_that_resists_cancellation_is_reported(
            self, clean_tracker, monkeypatch):
        """The other half of r5-1: absorbed OR reported, never neither. A late callback that
        swallows CancelledError cannot be taken away, so it is named in `survived` — with the count
        that proves it."""
        from slack_client.event_handlers import registration

        monkeypatch.setattr(type(clean_tracker), "_CANCEL_GRACE", 0.3, raising=False)
        release = asyncio.Event()
        started = asyncio.Event()
        tasks: list = []

        @registration.track_ingress
        async def handle_stubborn_parting():
            started.set()
            while not release.is_set():
                try:
                    await asyncio.wait_for(release.wait(), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    continue

        async def _wedged_close():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                tasks.append(asyncio.ensure_future(handle_stubborn_parting()))
                raise

        close_task = asyncio.ensure_future(_wedged_close())
        clean_tracker.track_dispatcher(close_task, name="the socket-mode handler close")
        try:
            drain = await registration.drain_ingress_callbacks(timeout=0.05)
        finally:
            release.set()
            if tasks:
                await asyncio.wait(tasks, timeout=2.0)

        assert started.is_set()
        assert drain.gave_up is True
        assert drain.survived == ("handle_stubborn_parting",), drain

    def test_every_registered_bolt_callback_is_counted_including_settings(self):
        """r3-10. The settings callbacks were deliberately left untracked, so "Slack ingress is
        quiet" was a claim about only some of Slack's ingress — while a channel-settings submission
        writes to the very database the next lines close. The whole registration surface is walked
        here, not a list somebody remembered to update."""
        from slack_client.event_handlers.registration import SlackRegistrationMixin
        from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

        registered = {}

        class _RecordingApp:
            def _record(self, kind):
                def _decorator_factory(*a, **k):
                    def _decorator(fn):
                        registered[f"{kind}:{a[0] if a else ''}:{fn.__name__}"] = fn
                        return fn
                    return _decorator
                return _decorator_factory

            def __getattr__(self, kind):
                return self._record(kind)

        class _Host(SlackRegistrationMixin, SlackSettingsHandlersMixin):
            def __init__(self):
                self.app = _RecordingApp()

        _Host()._register_handlers()

        untracked = sorted(name for name, fn in registered.items()
                           if not getattr(fn, "__ingress_tracked__", False))
        assert not untracked, f"these Bolt callbacks are invisible to shutdown: {untracked}"
        # Sanity: the walk actually found both surfaces, so an empty result cannot pass vacuously.
        assert any(name.startswith("event:message:") for name in registered)
        assert any("settings" in name for name in registered)
        assert len(registered) > 20, registered

    @pytest.mark.asyncio
    async def test_the_callback_that_asked_for_shutdown_does_not_wait_for_itself(self,
                                                                                clean_tracker):
        """Shutdown is reachable from a Bolt callback (a settings action, a slash command). Its own
        entry is excluded, or the barrier would burn the whole deadline on every such teardown — and
        it still waits for everybody ELSE, which is the half that must not be lost with it."""
        from slack_client.event_handlers import registration

        finished = []

        @registration.track_ingress
        async def handle_other_event():
            await asyncio.sleep(0.05)
            finished.append("other")

        @registration.track_ingress
        async def handle_shutdown_request():
            asyncio.ensure_future(handle_other_event())
            return await registration.drain_ingress_callbacks(timeout=5.0)

        loop = asyncio.get_running_loop()
        started = loop.time()
        drain = await asyncio.wait_for(handle_shutdown_request(), timeout=2.0)
        elapsed = loop.time() - started

        assert drain == registration.IngressDrain()
        assert finished == ["other"], "the barrier stopped waiting for the other callback"
        assert elapsed < 1.0, f"the barrier waited out its own deadline ({elapsed:.2f}s)"


class TestSocketModeStopIsNotAbandoned:
    """What `client.stop()` RETURNING is allowed to mean (codex r2-5).

    The socket-mode close is the thing that stops Bolt dispatching. It hangs often enough that
    stop() only waits 0.1s for it — but it then walked away from the task entirely, so stop()
    returned while the socket could still be delivering events, and every caller downstream
    treated ingress as down. It is awaited on a real timeout now, and said out loud when the
    timeout is what happens.
    """

    def _host(self, close_coro):
        from slack_client.messaging import SlackMessagingMixin

        class _Host(SlackMessagingMixin):
            def __init__(self):
                self.handler = Mock()
                self.handler.client = None          # no sessions to force-close
                self.handler.close_async = Mock(side_effect=lambda: close_coro)
                self._start_task = Mock()
                self._start_task.done = Mock(return_value=True)
                self.app = None
                self._socket_liveness = None
                self.logged = []

            def log_info(self, msg): self.logged.append(("info", msg))
            def log_debug(self, msg): self.logged.append(("debug", msg))
            def log_warning(self, msg): self.logged.append(("warning", msg))
            def log_error(self, msg): self.logged.append(("error", msg))

        return _Host()

    @pytest.mark.asyncio
    async def test_a_slow_close_is_awaited_to_completion(self):
        done = []

        async def _slow_close():
            await asyncio.sleep(0.2)                # past the 0.1s first try
            done.append("closed")

        host = self._host(_slow_close())
        await host.stop()

        assert done == ["closed"], "stop() returned with the socket still closing"

    @pytest.mark.asyncio
    async def test_a_wedged_close_is_bounded_and_logged_as_an_error(self, monkeypatch):
        from slack_client.event_handlers import registration

        async def _wedged():
            await asyncio.sleep(60)

        coro = _wedged()
        host = self._host(coro)
        monkeypatch.setattr(type(host), "_HANDLER_CLOSE_TIMEOUT", 0.05, raising=False)
        registration.ingress.reset()
        try:
            await host.stop()

            # r3-9: and it is still registered with the ingress barrier, so the caller cannot
            # mistake this bounded wait for ingress being down. r4-1: the barrier then cancels it,
            # so "gave up" is the verdict rather than a count of things left running.
            assert (await registration.drain_ingress_callbacks(timeout=0.05)).gave_up is True
        finally:
            for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
                task.cancel()
            registration.ingress.reset()

        errors = [m for level, m in host.logged if level == "error"]
        assert any("may still be dispatching" in m for m in errors), host.logged


class TestTurnLedgerWiring:
    """CV8 §10: every CHANNEL turn opens with a `turn_start` and closes with exactly one
    `turn_outcome`, and the pair is what makes the turn population countable.

    The gate population cannot answer "how much of the bot's channel output did the gate cause",
    because the turns it never judged — mentions, thread continuations — leave no row in it. These
    two events are that denominator, and they are emitted from handle_message's own outer finally,
    which is the only place that runs for every turn no matter how it ended.
    """

    @pytest.fixture
    def sink(self, tmp_path):
        import json
        import logging

        from config import config
        from message_processor import participation_telemetry as pt

        named = logging.getLogger(pt._SINK_LOGGER_NAME)
        saved = named.handlers[:]
        pt.shutdown()
        with patch.object(config, "log_directory", str(tmp_path)), \
                patch.object(config, "enable_participation_telemetry", True):
            pt.initialize()

            def lines(event=None):
                pt._drain()
                path = tmp_path / pt.LOG_NAME
                rows = ([json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
                        if path.exists() else [])
                return [r for r in rows if event is None or r["event"] == event]

            try:
                yield lines
            finally:
                pt.shutdown()
        named.handlers = saved

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor.thread_manager = None
        return bot

    @staticmethod
    def _client():
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value=None)
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock(return_value="50.0")
        client.self_team_id = "T1"
        return client

    @staticmethod
    def _message(channel_id="C123", **meta):
        from base_client import Message
        payload = {"ts": "10.0", "wake_source": "mention"}
        payload.update(meta)
        return Message(text="ping", user_id="U1", channel_id=channel_id,
                       thread_id="10.0", metadata=payload)

    @pytest.mark.asyncio
    async def test_a_channel_turn_opens_and_closes_exactly_one_row(self, bot, sink):
        message = self._message()
        bot.processor.process_message = AsyncMock(return_value=Mock(
            type="text", content="pong", metadata={"streamed": False}))

        await bot.handle_message(message, self._client())

        starts, outcomes = sink("turn_start"), sink("turn_outcome")
        assert len(starts) == 1 and len(outcomes) == 1
        assert starts[0]["turn_id"] == outcomes[0]["turn_id"]   # the join key
        assert starts[0]["surface"] == "channel"
        assert starts[0]["gated"] is False                      # no gate_required on this message
        assert starts[0]["wake_source"] == "mention"
        assert outcomes[0]["kind"] == "reply"

    @pytest.mark.asyncio
    async def test_the_outcome_carries_the_destination_the_reply_actually_landed_on(self, bot,
                                                                                   sink):
        message = self._message()
        bot.processor.process_message = AsyncMock(return_value=Mock(
            type="text", content="pong", metadata={"streamed": False}))

        await bot.handle_message(message, self._client())

        row = sink("turn_outcome")[0]
        assert [d["first_ts"] for d in row["destinations"]] == ["50.0"]
        assert row["destinations"][0]["state"] == "committed"
        assert row["destinations"][0]["kind"] == "reply"
        assert row["chars"] == len("pong")

    @pytest.mark.asyncio
    async def test_a_turn_that_died_still_closes_its_row(self, bot, sink):
        """The finally is the point: an outcome missing for a turn that ran is indistinguishable
        from a turn that never started, and a crash is exactly when we want to know."""
        message = self._message()
        bot.processor.process_message = AsyncMock(side_effect=RuntimeError("boom"))
        client = self._client()
        client.handle_error = AsyncMock()

        await bot.handle_message(message, client)

        outcomes = sink("turn_outcome")
        # `error_unhandled`, not the `empty` a missing Response would imply: the turn row uses
        # the label the path that ended the turn chose, so it agrees with the gate terminal.
        assert len(outcomes) == 1 and outcomes[0]["kind"] == "error_unhandled"

    @pytest.mark.asyncio
    async def test_a_dm_turn_is_outside_the_turn_population(self, bot, sink):
        """Same discriminator receipts use. A DM has no channel stream and no receipt, so a row
        describing one would carry an H and a stream flag that mean nothing."""
        message = self._message(channel_id="D999")
        bot.processor.process_message = AsyncMock(return_value=Mock(
            type="text", content="pong", metadata={"streamed": False}))

        await bot.handle_message(message, self._client())

        assert sink("turn_start") == [] and sink("turn_outcome") == []

    @pytest.mark.asyncio
    async def test_a_gate_that_never_woke_leaves_no_turn_row(self, bot, sink):
        """A declined gate returns before the TurnRuntime exists, so there is no turn to report —
        and pairing holds because the start and the outcome are both downstream of that object."""
        message = self._message(gate_required=True)
        bot._run_participation_gate = AsyncMock(return_value=None)
        bot.processor.process_message = AsyncMock()

        await bot.handle_message(message, self._client())

        assert sink("turn_start") == [] and sink("turn_outcome") == []
        bot.processor.process_message.assert_not_called()


class TestSignalDrivenShutdown:
    """Rider D: a SIGTERM used to leave the ledger with no `session_end`.

    The signal handler started its OWN shutdown task while `run()` went on to finish and let the
    loop close — which cancelled that task partway, and the last thing it does is drain the
    telemetry sink. So every restart looked, in the one file that exists to tell those apart, like
    a crash. Now there is ONE shutdown and both paths await it.
    """

    @pytest.fixture
    def bot(self):
        return ChatBotV2(platform="slack")

    @pytest.mark.asyncio
    async def test_begin_shutdown_is_the_same_task_for_every_caller(self, bot):
        """Two signals, or a signal and run()'s finally: one shutdown, awaited by both."""
        with patch.object(bot, "shutdown", new_callable=AsyncMock) as shutdown:
            task = bot.begin_shutdown()
            assert bot.begin_shutdown() is task
            await task
        assert shutdown.await_count == 1

    @patch('main.sys.exit')
    @patch('main.log_session_start')
    @patch('slack_client.SlackBot')
    @patch('main.MessageProcessor')
    @patch('main.config')
    @pytest.mark.asyncio
    async def test_a_sigterm_shutdown_writes_exactly_one_session_end_after_the_drain(
            self, mock_config, mock_processor_class, mock_slackbot_class, mock_log_start,
            mock_exit, bot):
        """End to end through the REAL telemetry sink (pointed at this test's own directory by the
        autouse fixture): the row exists, exactly once, and it is the LAST line — which is what
        makes its absence meaningful for every other session."""
        import json
        import pathlib

        from message_processor import participation_telemetry

        mock_config.validate.return_value = None
        mock_client = Mock(db=_startup_db())
        mock_slackbot_class.return_value = mock_client
        mock_processor_class.return_value.mcp_manager.has_mcp_servers.return_value = False

        stop = asyncio.Event()

        async def _start():
            # The signal arrives while Socket Mode is running, delivered the way the OS delivers
            # it: from a handler with no loop of its own, which may only schedule.
            bot._signal_handler(signal.SIGTERM, None)
            await stop.wait()

        async def _stop():
            stop.set()

        mock_client.start = _start
        mock_client.stop = _stop

        # Awaited INLINE, as `main()` awaits it: run() shares the caller's task, so the shutdown's
        # final sweep sees the task that is waiting for it and must not cancel it.
        await bot.run()

        participation_telemetry._drain()
        path = pathlib.Path(config.log_directory) / participation_telemetry.LOG_NAME
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ends = [row for row in lines if row.get("event") == "session_end"]
        assert len(ends) == 1, f"expected exactly one session_end, got {len(ends)}"
        assert lines[-1]["event"] == "session_end", "it must be the last line of the session"

    @patch('main.asyncio.all_tasks')
    @patch('main.asyncio.current_task')
    @patch('main.asyncio.gather', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_the_final_sweep_spares_the_task_awaiting_the_shutdown(
            self, mock_gather, mock_current_task, mock_all_tasks, bot):
        """When a signal starts the shutdown, `run()` is a plain pending task from the sweep's
        point of view. Cancelling it would cancel the await keeping the loop alive for this very
        drain — and the thing lost would be the session_end above."""
        bot.running = True
        bot.client = Mock()
        bot.client.stop = AsyncMock()
        bot.processor = Mock()
        bot.processor.get_stats = Mock(return_value={})
        bot.processor.cleanup = AsyncMock()

        shutdown_task, run_task, stranger = Mock(), Mock(), Mock()
        bot._shutdown_task, bot._run_task = shutdown_task, run_task
        mock_current_task.return_value = shutdown_task
        mock_all_tasks.return_value = [shutdown_task, run_task, stranger]

        await bot.shutdown()

        stranger.cancel.assert_called_once()
        run_task.cancel.assert_not_called()
        shutdown_task.cancel.assert_not_called()


class TestTurnSettlementOrdering:
    """What this turn DISPATCHED settles before what it POSTED.

    A tool that outran its bound is still running when the loop returns, and it may still be
    posting. So the turn's finally cancels it and WAITS for the cancellation to land before the
    receipt ledger settles: settling first is exactly what leaves a delivered message of ours with
    nothing claiming it, permanently outside the stream we rebuild the room from.
    """

    @pytest.fixture
    def bot(self):
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor._persist_tool_provenance = Mock()
        return bot

    @staticmethod
    def _client():
        client = Mock()
        client.send_thinking_indicator = AsyncMock(return_value=None)
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock(return_value="50.0")
        client.self_team_id = "T1"
        return client

    @pytest.mark.asyncio
    async def test_a_straggling_flight_is_cancelled_before_the_ledger_settles(self, bot):
        from base_client import Message, Response
        from message_processor import outbound_receipts
        from tool_registry import ToolContext, ToolRegistry

        order = []
        captured = {}
        bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)

        async def _wedged_tool(ctx, args):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                order.append("flight cancelled")
                raise

        registry = ToolRegistry()
        registry.register({"type": "function", "name": "wedged", "parameters": {}},
                          _wedged_tool, timeout=0.02)

        async def _process(message, client, thinking_id=None, **kwargs):
            turn = captured["turn"]
            await registry.dispatch_all(
                ToolContext(channel_id="C1", thread_ts="10.0", turn=turn),
                [{"name": "wedged", "arguments": "{}", "call_id": "c1"}])
            assert turn.pending_tool_flights, "the tool outran its bound and is still running"
            return Response(type="text", content="", metadata={"posted": False})

        bot.processor.process_message = _process

        async def _settle(ledger, turn=None):
            order.append("ledger settled")

        with patch.object(outbound_receipts, "settle_ledger", _settle):
            await bot.handle_message(
                Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                        metadata={"ts": "10.0"}),
                self._client())

        assert order == ["flight cancelled", "ledger settled"]
        assert captured["turn"].pending_tool_flights == []

    @pytest.mark.asyncio
    async def test_a_cancellation_landing_on_the_finalizer_still_revokes_and_settles(self, bot):
        """The window this closes: a cancellation arriving while the turn is finishing its
        flights used to be caught and stepped over, so revocation never happened and the ledger
        settled anyway — leaving a shielded straggler free to take a lease and post AFTER
        settlement. The sequence is now one shielded unit that owns itself."""
        from base_client import Message, Response
        from message_processor import outbound_receipts
        from message_processor.turn_runtime import TurnRuntime
        from tool_registry import ToolContext, ToolRegistry

        order, captured = [], {}
        entered, hold, stuck = asyncio.Event(), asyncio.Event(), asyncio.Event()
        bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)

        async def _stubborn_tool(ctx, args):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                order.append("flight cancelled")
                await stuck.wait()          # refuses to die within the grace
            return {"ok": True}

        registry = ToolRegistry()
        registry.register({"type": "function", "name": "stubborn", "parameters": {}},
                          _stubborn_tool, timeout=0.02)

        async def _process(message, client, thinking_id=None, **kwargs):
            await registry.dispatch_all(
                ToolContext(channel_id="C1", thread_ts="10.0", turn=captured["turn"]),
                [{"name": "stubborn", "arguments": "{}", "call_id": "c1"}])
            return Response(type="text", content="", metadata={"posted": False})

        bot.processor.process_message = _process
        real_finish = TurnRuntime.finish_tool_flights

        async def _finish(self, *, grace=5.0):
            # Park INSIDE the unit, so the cancellation below lands on the outer await of the
            # finalizer — the exact seam the old code walked away from.
            entered.set()
            await hold.wait()
            return await real_finish(self, grace=0.05)

        async def _settle(ledger, turn=None):
            order.append(f"settled (revoked={turn.effects_revoked})")

        with patch.object(TurnRuntime, "finish_tool_flights", _finish), \
             patch.object(outbound_receipts, "settle_ledger", _settle):
            task = asyncio.ensure_future(bot.handle_message(
                Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                        metadata={"ts": "10.0"}),
                self._client()))
            await entered.wait()
            task.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            assert not task.done(), "the finalizer owns the turn's ending, not the canceller"
            hold.set()
            await task
            stuck.set()

        assert order == ["flight cancelled", "settled (revoked=True)"]
        turn = captured["turn"]
        assert turn.effects_revoked is True, "a straggler that outlived cancellation was revoked"
        await asyncio.gather(*(f.task for f in turn.pending_tool_flights),
                             return_exceptions=True)

    @pytest.mark.asyncio
    async def test_the_caller_never_returns_while_the_finalizer_is_still_running(self, bot):
        """There used to be a cap on how many cancellations the outer await would absorb. Past
        it, `handle_message` returned — releasing the thread's state, emitting the outcome and
        removing the turn — while the finalizer was still revoking and settling behind it. That
        state is the whole thing the shielded unit exists to prevent, so the wait is now bound by
        the unit FINISHING and by nothing else. Cancelled here far more times than any cap was."""
        from base_client import Message, Response
        from message_processor import outbound_receipts
        from message_processor.turn_runtime import TurnRuntime

        entered, hold = asyncio.Event(), asyncio.Event()
        settled = []

        async def _process(message, client, thinking_id=None, **kwargs):
            return Response(type="text", content="", metadata={"posted": False})

        bot.processor.process_message = _process

        async def _finish(self, *, grace=5.0):
            entered.set()
            await hold.wait()
            return ()

        async def _settle(ledger, turn=None):
            settled.append(True)

        with patch.object(TurnRuntime, "finish_tool_flights", _finish), \
             patch.object(outbound_receipts, "settle_ledger", _settle):
            task = asyncio.ensure_future(bot.handle_message(
                Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                        metadata={"ts": "10.0"}),
                self._client()))
            await entered.wait()
            for _ in range(25):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "the caller outlived the finalizer"
                assert not settled, "settlement had not happened yet"
            hold.set()
            await task

        assert settled == [True], "and the unit ran to completion exactly once"

    @pytest.mark.asyncio
    async def test_a_drain_that_fails_revokes_before_it_settles(self, bot):
        """A failure of `finish_tool_flights` ITSELF — not a tool's, which it absorbs. The turn
        can no longer say what is still running, and settling receipts around unknown state is how
        a post ends up outside every account of the turn. It settles anyway (the alternative is
        stranding rows nothing will ever revisit), but only after nothing can act any more."""
        from base_client import Message, Response
        from message_processor import outbound_receipts
        from message_processor.turn_runtime import TurnRuntime

        order, captured = [], {}
        bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)

        async def _process(message, client, thinking_id=None, **kwargs):
            return Response(type="text", content="", metadata={"posted": False})

        bot.processor.process_message = _process

        async def _finish(self, *, grace=5.0):
            raise RuntimeError("the flight table is broken")

        real_revoke = TurnRuntime.revoke_effects

        def _revoke(self, reason):
            order.append("revoked")
            real_revoke(self, reason)

        async def _settle(ledger, turn=None):
            order.append(f"settled (revoked={turn.effects_revoked})")

        with patch.object(TurnRuntime, "finish_tool_flights", _finish), \
             patch.object(TurnRuntime, "revoke_effects", _revoke), \
             patch.object(outbound_receipts, "settle_ledger", _settle):
            await bot.handle_message(
                Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                        metadata={"ts": "10.0"}),
                self._client())

        assert order == ["revoked", "settled (revoked=True)"]
        assert captured["turn"].effects_revoked is True

    @pytest.mark.asyncio
    async def test_a_revocation_that_fails_too_leaves_the_ledger_unsettled(self, bot, caplog):
        """The one state where finalizing is strictly worse than not. The turn cannot say what is
        still running AND cannot stop it, so a settle would record as accounted-for words a live
        task may still be adding to. The rows stay in flight for the next boot's dead-session
        reconcile, which is the only mechanism that can still tell the truth about them."""
        import logging

        from base_client import Message, Response
        from message_processor import outbound_receipts
        from message_processor.turn_runtime import TurnRuntime

        settled, captured = [], {}
        bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)

        async def _process(message, client, thinking_id=None, **kwargs):
            return Response(type="text", content="", metadata={"posted": False})

        bot.processor.process_message = _process

        async def _finish(self, *, grace=5.0):
            raise RuntimeError("the flight table is broken")

        def _revoke(self, reason):
            raise RuntimeError("and revocation is broken too")

        async def _settle(ledger, turn=None):
            settled.append(True)

        with patch.object(TurnRuntime, "finish_tool_flights", _finish), \
             patch.object(TurnRuntime, "revoke_effects", _revoke), \
             patch.object(outbound_receipts, "settle_ledger", _settle), \
             caplog.at_level(logging.CRITICAL):
            await bot.handle_message(
                Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                        metadata={"ts": "10.0"}),
                self._client())

        assert settled == [], "settling around unknown, unrevoked state is the receiptless post"
        critical = "\n".join(r.getMessage() for r in caplog.records
                             if r.levelno >= logging.CRITICAL)
        assert "UNSETTLED" in critical
        assert str(captured["turn"].turn_id) in critical, "named, so the rows can be found"


class TestShutdownRequestedDuringStartup:
    """A SIGTERM can land after the handlers are installed and before there is anything to stop.

    `initialize()` installs them; the run loop comes up afterwards. The request used to create a
    shutdown task that returned immediately (nothing was running yet) and CACHED it — so the bot
    finished starting, served happily, and every later shutdown request was handed that finished
    task. A supervisor's stop signal during a slow start left an unstoppable bot.
    """

    @pytest.fixture
    def bot(self):
        return ChatBotV2(platform="slack")

    @patch('main.log_session_start')
    @pytest.mark.asyncio
    async def test_a_signal_during_initialization_still_stops_the_bot(self, _log_start, bot):
        calls = []

        async def _initialize():
            # Exactly where the real one installs the handlers: inside initialize, before run()
            # has set `running`.
            bot._signal_handler(signal.SIGTERM, None)
            await asyncio.sleep(0)

        async def _shutdown():
            # Mirrors the real one's first line, which is the whole reason the request has to
            # outlive it.
            if not bot.running:
                calls.append("nothing to stop")
                return
            calls.append("stopped the bot")
            bot.running = False
            bot._shutdown_completed = True

        with patch.object(bot, "initialize", _initialize), \
             patch.object(bot, "shutdown", _shutdown), \
             patch.object(bot, "start_cleanup_task", AsyncMock()) as cleanup:
            await bot.run()

        assert calls == ["nothing to stop", "stopped the bot"]
        assert bot.running is False
        cleanup.assert_not_awaited(), "the bot never started serving"

    @patch('main.log_session_start')
    @pytest.mark.asyncio
    async def test_the_completed_shutdown_is_still_the_one_every_later_caller_gets(
            self, _log_start, bot):
        """The other half: once a shutdown has actually RUN, it is THE shutdown and a second
        request must not start another one."""
        calls = []

        async def _shutdown():
            calls.append(1)
            bot._shutdown_completed = True

        with patch.object(bot, "shutdown", _shutdown):
            first = bot.begin_shutdown()
            await first
            assert bot.begin_shutdown() is first
            await bot.begin_shutdown()

        assert calls == [1]


class TestWordsElsewhereIsNotAViolation:
    """[P3 live F1] The delivery contract check must not read a committed cross-thread post as
    'empty text without a terminal action' — the healthy words-elsewhere turn and a real
    violation were indistinguishable in the log."""

    def test_the_check_consults_the_committed_visible_action(self):
        import inspect

        import main as main_mod

        src = inspect.getsource(main_mod)
        anchor = src.index("Empty text response without a terminal action")
        window = src[max(0, anchor - 900):anchor]
        assert "visible_action_committed" in window, (
            "the contract-violation warning must be skipped when the turn committed its "
            "visible action elsewhere (cross-thread post)")
