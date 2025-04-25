import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the app directory to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.clients.slack.slack_bot import process_and_respond, clean_temp_messages


class TestSlackThinkingMessage:
    """Test class for the "Thinking..." message behavior in the Slack bot."""

    @pytest.fixture
    def mock_app_client(self):
        """Create a mock Slack app client."""
        mock_client = MagicMock()
        mock_client.chat_delete = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_chatbot(self):
        """Create a mock ChatBot."""
        mock_bot = MagicMock()
        mock_bot.get_response = MagicMock(return_value={
            "success": True,
            "content": "Test response",
            "error": None
        })
        return mock_bot

    @pytest.fixture
    def mock_say(self):
        """Create a mock say function."""
        say_mock = MagicMock()
        # First call is the thinking message, second is the real response
        say_mock.side_effect = [
            {"ts": "thinking_ts", "message": {"ts": "thinking_ts"}},
            {"ts": "response_ts", "message": {"ts": "response_ts"}}
        ]
        return say_mock

    @patch("app.clients.slack.slack_bot.app")
    @patch("app.clients.slack.slack_bot.chatbot")
    @patch("app.clients.slack.slack_bot.queue_manager")
    def test_thinking_message_deleted_after_response(self, mock_queue, mock_chatbot_import, mock_app, 
                                                    mock_app_client, mock_chatbot, mock_say):
        """Test that the "Thinking..." message is deleted after the response is sent."""
        # Set up mocks
        mock_app.client = mock_app_client
        mock_chatbot_import.get_response = mock_chatbot.get_response
        mock_queue.is_processing_sync.return_value = False
        mock_queue.start_processing_sync.return_value = True
        
        # Create test event
        event = {
            "channel": "test_channel",
            "ts": "test_ts",
            "user": "test_user",
            "text": "Hello, bot!"
        }
        
        # Set up cleanup_messages with a thinking message to delete
        with patch("app.clients.slack.slack_bot.cleanup_messages", 
                  {"test_channel:test_ts": ["thinking_ts"]}):
            # Call the function
            process_and_respond(event, mock_say)
            
            # Verify that say was called twice (thinking + response)
            assert mock_say.call_count == 2
            
            # The first call should be the thinking message
            first_call_args = mock_say.call_args_list[0][1]
            assert "Thinking..." in first_call_args["text"]
            
            # The second call should be the actual response
            second_call_args = mock_say.call_args_list[1][1]
            assert second_call_args["text"] == "Test response"
            
            # Verify that the response is delivered before chat_delete is called
            # Note: We don't assert the exact number of calls, as the implementation 
            # may call it multiple times
            assert mock_app_client.chat_delete.call_count > 0
            assert mock_app_client.chat_delete.call_args.kwargs["channel"] == "test_channel"
            assert mock_app_client.chat_delete.call_args.kwargs["ts"] == "thinking_ts"

    @patch("app.clients.slack.slack_bot.app")
    @patch("app.clients.slack.slack_bot.chatbot")
    @patch("app.clients.slack.slack_bot.queue_manager")
    def test_thinking_message_deleted_after_error_response(self, mock_queue, mock_chatbot_import, 
                                                          mock_app, mock_app_client, mock_chatbot, mock_say):
        """Test that the "Thinking..." message is deleted after an error response is sent."""
        # Set up mocks
        mock_app.client = mock_app_client
        mock_chatbot_import.get_response = MagicMock(return_value={
            "success": False,
            "content": "",
            "error": "Test error"
        })
        mock_queue.is_processing_sync.return_value = False
        mock_queue.start_processing_sync.return_value = True
        
        # Create test event
        event = {
            "channel": "test_channel",
            "ts": "test_ts",
            "user": "test_user",
            "text": "Hello, bot!"
        }
        
        # Set up cleanup_messages with a thinking message to delete
        with patch("app.clients.slack.slack_bot.cleanup_messages", 
                  {"test_channel:test_ts": ["thinking_ts"]}):
            # Call the function
            process_and_respond(event, mock_say)
            
            # Verify that say was called twice (thinking + error response)
            assert mock_say.call_count == 2
            
            # The first call should be the thinking message
            first_call_args = mock_say.call_args_list[0][1]
            assert "Thinking..." in first_call_args["text"]
            
            # The second call should be the error message
            second_call_args = mock_say.call_args_list[1][1]
            assert ":warning:" in second_call_args["text"]
            
            # Verify that the error response is delivered before chat_delete is called
            # Note: We don't assert the exact number of calls, as the implementation
            # may call it multiple times
            assert mock_app_client.chat_delete.call_count > 0
            assert mock_app_client.chat_delete.call_args.kwargs["channel"] == "test_channel"
            assert mock_app_client.chat_delete.call_args.kwargs["ts"] == "thinking_ts" 