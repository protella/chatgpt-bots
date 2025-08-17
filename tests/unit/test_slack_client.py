"""Unit tests for slack_client.py - Slack bot client implementation"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from slack_sdk.errors import SlackApiError
import base64
import json

from slack_client import SlackBot
from base_client import Message, Response
from openai_client import ImageData


class TestSlackBotInitialization:
    """Test SlackBot initialization and setup"""
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_init_creates_slack_app(self, mock_app_class, mock_db_class):
        """Test that SlackBot creates Slack app on init"""
        # Create bot
        bot = SlackBot()
        
        # Verify Slack app created
        mock_app_class.assert_called_once()
        assert bot.app is not None
        
        # Verify database created
        mock_db_class.assert_called_once_with(platform="slack")
        assert bot.db is not None
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_init_with_message_handler(self, mock_app_class, mock_db_class):
        """Test initialization with message handler"""
        handler = Mock()
        bot = SlackBot(message_handler=handler)
        
        assert bot.message_handler is handler
        assert bot.markdown_converter is not None
        assert bot.user_cache == {}
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_register_handlers(self, mock_app_class, mock_db_class):
        """Test Slack event handlers are registered"""
        mock_app = Mock()
        mock_app_class.return_value = mock_app
        
        bot = SlackBot()
        
        # Verify handlers registered
        assert mock_app.event.called
        # Should register app_mention and message handlers
        calls = mock_app.event.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "app_mention" in event_types
        assert "message" in event_types


class TestSlackBotUserManagement:
    """Test user info and timezone management"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.db = Mock()
            return bot
    
    def test_get_username_from_cache(self, bot):
        """Test getting username from memory cache"""
        # Setup cache
        bot.user_cache["U123"] = {
            "username": "test_user",
            "timezone": "America/New_York"
        }
        
        client = Mock()
        username = bot.get_username("U123", client)
        
        assert username == "test_user"
        # Should not call API
        client.users_info.assert_not_called()
    
    def test_get_username_from_database(self, bot):
        """Test getting username from database"""
        # Setup DB mock
        bot.db.get_or_create_user.return_value = {"username": "db_user"}
        bot.db.get_user_timezone.return_value = ("UTC", "UTC", 0)
        
        client = Mock()
        username = bot.get_username("U456", client)
        
        assert username == "db_user"
        assert "U456" in bot.user_cache
        assert bot.user_cache["U456"]["username"] == "db_user"
    
    def test_get_username_from_slack_api(self, bot):
        """Test fetching username from Slack API"""
        # Setup mocks
        bot.db.get_or_create_user.return_value = {}
        bot.db.get_user_timezone.return_value = None
        
        client = Mock()
        client.users_info.return_value = {
            "ok": True,
            "user": {
                "profile": {
                    "display_name": "Display Name",
                    "real_name": "Real Name"
                },
                "name": "username",
                "tz": "America/Los_Angeles",
                "tz_label": "PST",
                "tz_offset": -28800
            }
        }
        
        username = bot.get_username("U789", client)
        
        assert username == "Display Name"
        assert bot.user_cache["U789"]["username"] == "Display Name"
        assert bot.user_cache["U789"]["timezone"] == "America/Los_Angeles"
        
        # Should save to DB
        bot.db.save_user_info.assert_called_once()
    
    def test_get_username_api_failure(self, bot):
        """Test fallback when API fails"""
        bot.db.get_or_create_user.return_value = {}
        bot.db.get_user_timezone.return_value = None
        
        client = Mock()
        client.users_info.side_effect = Exception("API Error")
        
        username = bot.get_username("U999", client)
        
        # Should fallback to user ID
        assert username == "U999"
    
    def test_get_user_timezone(self, bot):
        """Test getting user timezone"""
        # From cache
        bot.user_cache["U123"] = {"timezone": "America/New_York"}
        tz = bot.get_user_timezone("U123", Mock())
        assert tz == "America/New_York"
        
        # From database
        bot.db.get_user_timezone.return_value = ("Europe/London", "GMT", 0)
        tz = bot.get_user_timezone("U456", Mock())
        assert tz == "Europe/London"
        
        # Default fallback
        bot.db.get_user_timezone.return_value = None
        bot.get_username = Mock(return_value="user")  # Mock to avoid API call
        tz = bot.get_user_timezone("U789", Mock())
        assert tz == "UTC"


class TestSlackBotMessageHandling:
    """Test message processing and handling"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.db = Mock()
            return bot
    
    def test_handle_slack_message(self, bot):
        """Test converting Slack event to Message"""
        handler = Mock()
        bot.message_handler = handler
        bot.get_username = Mock(return_value="test_user")
        bot.get_user_timezone = Mock(return_value="UTC")
        
        event = {
            "text": "Hello <@U123> bot",
            "user": "U456",
            "channel": "C789",
            "thread_ts": "123.456",
            "ts": "123.456"
        }
        
        client = Mock()
        bot._handle_slack_message(event, client)
        
        # Verify handler called
        handler.assert_called_once()
        message = handler.call_args[0][0]
        
        assert isinstance(message, Message)
        assert message.text == "Hello  bot"  # Mention cleaned
        assert message.user_id == "U456"
        assert message.channel_id == "C789"
        assert message.thread_id == "123.456"
    
    def test_handle_message_with_files(self, bot):
        """Test handling messages with file attachments"""
        handler = Mock()
        bot.message_handler = handler
        bot.get_username = Mock(return_value="user")
        bot.get_user_timezone = Mock(return_value="UTC")
        
        event = {
            "text": "Check this image",
            "user": "U123",
            "channel": "C456",
            "ts": "789.012",
            "files": [
                {
                    "id": "F123",
                    "name": "image.png",
                    "mimetype": "image/png",
                    "url_private": "https://files.slack.com/image.png"
                }
            ]
        }
        
        bot._handle_slack_message(event, Mock())
        
        message = handler.call_args[0][0]
        assert len(message.attachments) == 1
        assert message.attachments[0]["type"] == "image"
        assert message.attachments[0]["url"] == "https://files.slack.com/image.png"
    
    def test_skip_message_changed_events(self, bot):
        """Test that message_changed events are skipped"""
        handler = Mock()
        bot.message_handler = handler
        
        event = {
            "subtype": "message_changed",
            "text": "Edited message"
        }
        
        bot._handle_slack_message(event, Mock())
        
        # Should not call handler
        handler.assert_not_called()
    
    def test_clean_mentions(self, bot):
        """Test mention cleaning from text"""
        text = "Hello <@U123ABC> and <@U456DEF>, how are you?"
        cleaned = bot._clean_mentions(text)
        assert cleaned == "Hello  and , how are you?"


class TestSlackBotMessaging:
    """Test sending messages to Slack"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.app = Mock()
            bot.app.client = Mock()
            return bot
    
    def test_send_message_simple(self, bot):
        """Test sending a simple message"""
        bot.format_text = Mock(return_value="formatted text")
        
        result = bot.send_message("C123", "thread_123", "Hello world")
        
        assert result is True
        bot.app.client.chat_postMessage.assert_called_once_with(
            channel="C123",
            thread_ts="thread_123",
            text="formatted text"
        )
    
    def test_send_message_long_split(self, bot):
        """Test splitting long messages"""
        long_text = "x" * 5000  # Exceeds MAX_MESSAGE_LENGTH
        bot.format_text = Mock(return_value=long_text)
        bot._split_message = Mock(return_value=["chunk1", "chunk2", "chunk3"])
        
        result = bot.send_message("C123", "thread_123", long_text)
        
        assert result is True
        # Should be called multiple times for chunks
        assert bot.app.client.chat_postMessage.call_count == 3
        
        # Check pagination added
        first_call = bot.app.client.chat_postMessage.call_args_list[0]
        assert "Part 1/3" in first_call[1]["text"]
    
    def test_send_message_error_handling(self, bot):
        """Test error handling in send_message"""
        bot.format_text = Mock(return_value="text")
        bot.app.client.chat_postMessage.side_effect = SlackApiError(
            message="Error",
            response={"error": "channel_not_found"}
        )
        
        result = bot.send_message("C123", "thread_123", "Hello")
        
        assert result is False
    
    def test_split_message_logic(self, bot):
        """Test message splitting logic"""
        # Test paragraph splitting
        text = "Para 1\\n\\nPara 2\\n\\nPara 3"
        chunks = bot._split_message(text)
        assert len(chunks) >= 1
        
        # Test sentence splitting for long paragraphs
        long_para = ". ".join(["Sentence " + str(i) for i in range(200)])
        chunks = bot._split_message(long_para)
        assert all(len(chunk) <= bot.MAX_MESSAGE_LENGTH for chunk in chunks)
    
    def test_send_thinking_indicator(self, bot):
        """Test sending thinking indicator"""
        bot.app.client.chat_postMessage.return_value = {"ts": "msg_123"}
        
        result = bot.send_thinking_indicator("C123", "thread_123")
        
        assert result == "msg_123"
        bot.app.client.chat_postMessage.assert_called_once()
        call_text = bot.app.client.chat_postMessage.call_args[1]["text"]
        assert "Thinking..." in call_text
    
    def test_delete_message(self, bot):
        """Test deleting a message"""
        result = bot.delete_message("C123", "msg_123")
        
        assert result is True
        bot.app.client.chat_delete.assert_called_once_with(
            channel="C123",
            ts="msg_123"
        )
    
    def test_update_message(self, bot):
        """Test updating a message"""
        result = bot.update_message("C123", "msg_123", "Updated text")
        
        assert result is True
        bot.app.client.chat_update.assert_called_once_with(
            channel="C123",
            ts="msg_123",
            text="Updated text",
            mrkdwn=True
        )


class TestSlackBotImageHandling:
    """Test image upload and download functionality"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.app = Mock()
            bot.app.client = Mock()
            return bot
    
    def test_send_image_success(self, bot):
        """Test successful image upload"""
        bot.app.client.files_upload_v2.return_value = {
            "files": [{
                "url_private": "https://files.slack.com/image.png",
                "permalink": "https://slack.com/files/image"
            }]
        }
        
        image_data = b"fake_image_data"
        url = bot.send_image("C123", "thread_123", image_data, "test.png", "Caption")
        
        assert url == "https://files.slack.com/image.png"
        bot.app.client.files_upload_v2.assert_called_once_with(
            channel="C123",
            thread_ts="thread_123",
            file=image_data,
            filename="test.png",
            initial_comment="Caption"
        )
    
    def test_send_image_no_url(self, bot):
        """Test image upload with no URL in response"""
        bot.app.client.files_upload_v2.return_value = {"files": [{}]}
        
        url = bot.send_image("C123", "thread_123", b"data", "test.png")
        
        assert url is None
    
    def test_send_image_error(self, bot):
        """Test image upload error handling"""
        bot.app.client.files_upload_v2.side_effect = SlackApiError(
            message="Error",
            response={"error": "file_too_large"}
        )
        
        url = bot.send_image("C123", "thread_123", b"data", "test.png")
        
        assert url is None
    
    @patch('requests.get')
    def test_download_file_with_id(self, mock_requests_get, bot):
        """Test downloading file with file ID"""
        # Mock file info response
        bot.app.client.files_info.return_value = {
            "ok": True,
            "file": {
                "url_private": "https://files.slack.com/download"
            }
        }
        
        # Mock download response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"file_content"
        mock_response.headers = {"content-type": "image/png"}
        mock_requests_get.return_value = mock_response
        
        content = bot.download_file("https://files.slack.com/file", "F123")
        
        assert content == b"file_content"
        bot.app.client.files_info.assert_called_once_with(file="F123")
    
    @patch('requests.get')
    def test_download_file_extract_id(self, mock_requests_get, bot):
        """Test downloading file by extracting ID from URL"""
        url = "https://files.slack.com/files-pri/T123-F456ABC/image.png"
        
        bot.app.client.files_info.return_value = {
            "ok": True,
            "file": {"url_private": "https://download.url"}
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"content"
        mock_response.headers = {"content-type": "image/png"}
        mock_requests_get.return_value = mock_response
        
        content = bot.download_file(url)
        
        assert content == b"content"
        bot.app.client.files_info.assert_called_once_with(file="F456ABC")
    
    def test_extract_file_id_from_url(self, bot):
        """Test file ID extraction from URLs"""
        # files-pri format
        url1 = "https://files.slack.com/files-pri/T123-F456/file.png"
        assert bot.extract_file_id_from_url(url1) == "F456"
        
        # permalink format
        url2 = "https://team.slack.com/files/U123/F789/file.png"
        assert bot.extract_file_id_from_url(url2) == "F789"
        
        # Invalid URL
        url3 = "https://example.com/image.png"
        assert bot.extract_file_id_from_url(url3) is None


class TestSlackBotThreadHistory:
    """Test thread history retrieval"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.app = Mock()
            bot.app.client = Mock()
            return bot
    
    def test_get_thread_history_success(self, bot):
        """Test retrieving thread history"""
        bot.app.client.conversations_replies.return_value = {
            "messages": [
                {
                    "text": "User message",
                    "user": "U123",
                    "ts": "1.0"
                },
                {
                    "text": "Bot response",
                    "user": "U456",
                    "bot_id": "B789",
                    "ts": "2.0"
                }
            ]
        }
        
        messages = bot.get_thread_history("C123", "thread_123")
        
        assert len(messages) == 2
        assert messages[0].text == "User message"
        assert messages[0].metadata["is_bot"] is False
        assert messages[1].text == "Bot response"
        assert messages[1].metadata["is_bot"] is True
    
    def test_get_thread_history_skip_thinking(self, bot):
        """Test that thinking indicators are skipped"""
        bot.app.client.conversations_replies.return_value = {
            "messages": [
                {"text": "Real message", "user": "U123", "ts": "1.0"},
                {"text": "🤔 Thinking...", "user": "U456", "ts": "2.0"},
                {"text": ":warning: currently processing", "user": "U789", "ts": "3.0"}
            ]
        }
        
        messages = bot.get_thread_history("C123", "thread_123")
        
        # Should only include real message
        assert len(messages) == 1
        assert messages[0].text == "Real message"
    
    def test_get_thread_history_with_files(self, bot):
        """Test thread history with file attachments"""
        bot.app.client.conversations_replies.return_value = {
            "messages": [{
                "text": "Check this",
                "user": "U123",
                "ts": "1.0",
                "files": [{
                    "id": "F123",
                    "name": "doc.pdf",
                    "mimetype": "application/pdf",
                    "url_private": "https://files.slack.com/doc.pdf"
                }]
            }]
        }
        
        messages = bot.get_thread_history("C123", "thread_123")
        
        assert len(messages[0].attachments) == 1
        assert messages[0].attachments[0]["type"] == "file"
        assert messages[0].attachments[0]["mimetype"] == "application/pdf"


class TestSlackBotStreaming:
    """Test streaming functionality"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            bot = SlackBot()
            bot.app = Mock()
            bot.app.client = Mock()
            bot.markdown_converter = Mock()
            return bot
    
    def test_supports_streaming(self, bot):
        """Test streaming support check"""
        with patch('slack_client.config') as mock_config:
            mock_config.enable_streaming = True
            mock_config.slack_streaming = True
            assert bot.supports_streaming() is True
            
            mock_config.slack_streaming = False
            assert bot.supports_streaming() is False
    
    def test_get_streaming_config(self, bot):
        """Test getting streaming configuration"""
        with patch('slack_client.config') as mock_config:
            mock_config.streaming_update_interval = 1.0
            mock_config.streaming_min_interval = 0.5
            mock_config.streaming_max_interval = 5.0
            mock_config.streaming_buffer_size = 100
            mock_config.streaming_circuit_breaker_threshold = 5
            mock_config.streaming_circuit_breaker_cooldown = 60
            
            config = bot.get_streaming_config()
            
            assert config["platform"] == "slack"
            assert config["update_interval"] == 1.0
            assert config["min_interval"] == 0.5
            assert config["max_interval"] == 5.0
    
    def test_update_message_streaming_success(self, bot):
        """Test streaming message update"""
        bot.format_text = Mock(return_value="formatted")
        bot.app.client.chat_update.return_value = {"ok": True}
        
        result = bot.update_message_streaming("C123", "msg_123", "Update")
        
        assert result["success"] is True
        assert result["rate_limited"] is False
        bot.app.client.chat_update.assert_called_once()
    
    def test_update_message_streaming_rate_limit(self, bot):
        """Test handling rate limits during streaming"""
        error_response = Mock()
        error_response.status_code = 429
        error_response.headers = {"Retry-After": "30"}
        
        bot.format_text = Mock(return_value="text")
        bot.app.client.chat_update.side_effect = SlackApiError(
            message="Rate limited",
            response=error_response
        )
        
        result = bot.update_message_streaming("C123", "msg_123", "Update")
        
        assert result["success"] is False
        assert result["rate_limited"] is True
        assert result["retry_after"] == 30
    
    def test_update_message_streaming_skip_formatting(self, bot):
        """Test skipping format for pre-formatted messages"""
        # Enhanced prompt - should skip formatting
        text = "✨ Enhanced prompt here"
        bot.app.client.chat_update.return_value = {"ok": True}
        
        bot.update_message_streaming("C123", "msg_123", text)
        
        # Should not call format_text for enhanced prompts
        call_text = bot.app.client.chat_update.call_args[1]["text"]
        assert call_text == text  # Unchanged


class TestSlackBotErrorHandling:
    """Test error formatting and handling"""
    
    @pytest.fixture
    def bot(self):
        with patch('slack_client.App'), patch('slack_client.DatabaseManager'):
            return SlackBot()
    
    def test_format_error_message_simple(self, bot):
        """Test formatting simple error messages"""
        error = "Something went wrong"
        formatted = bot.format_error_message(error)
        
        assert ":warning:" in formatted
        assert "Oops! Something went wrong" in formatted
        assert "```Something went wrong```" in formatted
    
    def test_format_error_message_with_code(self, bot):
        """Test formatting error with error code"""
        error = "API failed. Error code: 429"
        formatted = bot.format_error_message(error)
        
        assert "*Error Code:* `429`" in formatted
    
    def test_format_error_message_openai_format(self, bot):
        """Test formatting OpenAI API error format"""
        error = "{'error': {'message': 'Rate limit exceeded', 'type': 'rate_limit_error'}}"
        formatted = bot.format_error_message(error)
        
        assert "rate_limit_error" in formatted
        assert "Rate limit exceeded" in formatted
        assert "Wait a moment and try again" in formatted
    
    def test_format_error_context_length(self, bot):
        """Test formatting context length error"""
        error = "context_length_exceeded: The conversation is too long"
        formatted = bot.format_error_message(error)
        
        assert "Start a new thread" in formatted
    
    def test_send_busy_message(self, bot):
        """Test sending busy message"""
        bot.send_message = Mock()
        
        bot.send_busy_message("C123", "thread_123")
        
        bot.send_message.assert_called_once()
        call_text = bot.send_message.call_args[0][2]
        assert ":warning:" in call_text
        assert "currently processing" in call_text


class TestSlackBotLifecycle:
    """Test bot start/stop lifecycle"""
    
    @patch('slack_client.SocketModeHandler')
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_start_bot(self, mock_app_class, mock_db_class, mock_handler_class):
        """Test starting the bot"""
        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler
        
        bot = SlackBot()
        bot.start()
        
        mock_handler_class.assert_called_once()
        mock_handler.start.assert_called_once()
        assert bot.handler is mock_handler
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_stop_bot(self, mock_app_class, mock_db_class):
        """Test stopping the bot"""
        bot = SlackBot()
        bot.handler = Mock()
        
        bot.stop()
        
        bot.handler.close.assert_called_once()
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_handle_response(self, mock_app_class, mock_db_class):
        """Test handling different response types"""
        bot = SlackBot()
        bot.send_message = Mock()
        bot.send_image = Mock(return_value="http://image.url")
        bot.format_error_message = Mock(return_value="Formatted error")
        
        # Text response
        text_response = Response(type="text", content="Hello")
        bot.handle_response("C123", "thread_123", text_response)
        bot.send_message.assert_called_with("C123", "thread_123", "Hello")
        
        # Image response
        image_data = ImageData(base64_data="data", format="png", prompt="test")
        image_response = Response(type="image", content=image_data)
        bot.handle_response("C123", "thread_123", image_response)
        bot.send_image.assert_called_once()
        
        # Error response
        error_response = Response(type="error", content="Error occurred")
        bot.handle_response("C123", "thread_123", error_response)
        assert bot.format_error_message.called


@pytest.mark.critical
class TestSlackBotCritical:
    """Critical functionality tests"""
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_critical_message_flow(self, mock_app_class, mock_db_class):
        """Critical test for complete message flow"""
        bot = SlackBot()
        handler = Mock()
        bot.message_handler = handler
        bot.get_username = Mock(return_value="user")
        bot.get_user_timezone = Mock(return_value="UTC")
        
        # Simulate incoming message
        event = {
            "text": "Test message",
            "user": "U123",
            "channel": "C456",
            "ts": "789.0"
        }
        
        bot._handle_slack_message(event, Mock())
        
        # Verify message created and passed to handler
        handler.assert_called_once()
        message = handler.call_args[0][0]
        assert message.text == "Test message"
        assert message.user_id == "U123"
        assert message.channel_id == "C456"
    
    @patch('slack_client.DatabaseManager')
    @patch('slack_client.App')
    def test_critical_error_recovery(self, mock_app_class, mock_db_class):
        """Critical test for error recovery"""
        bot = SlackBot()
        bot.app.client = Mock()
        
        # Simulate transient error then success
        bot.app.client.chat_postMessage.side_effect = [
            SlackApiError("Temporary error", response={"error": "timeout"}),
            {"ts": "123.456"}
        ]
        
        # First call fails
        result1 = bot.send_message("C123", "thread", "Test")
        assert result1 is False
        
        # Second call succeeds
        result2 = bot.send_message("C123", "thread", "Test")
        assert result2 is True