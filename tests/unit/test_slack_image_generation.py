import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the app directory to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.clients.slack.slack_bot import process_and_respond


class TestSlackImageGeneration:
    """Test class for image generation functionality in the Slack bot."""

    @pytest.fixture
    def mock_app_client(self):
        """Create a mock Slack app client."""
        mock_client = MagicMock()
        mock_client.chat_delete = MagicMock()
        mock_client.files_upload_v2 = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_say(self):
        """Create a mock say function."""
        say_mock = MagicMock()
        # First call is the generating message, second is the image upload
        say_mock.side_effect = [
            {"ts": "generating_ts", "message": {"ts": "generating_ts"}}
        ]
        return say_mock

    @patch("app.clients.slack.slack_bot.app")
    @patch("app.clients.slack.slack_bot.chatbot")
    @patch("app.clients.slack.slack_bot.is_image_request")
    @patch("app.clients.slack.slack_bot.queue_manager")
    @patch("app.core.image_service.create_optimized_prompt")
    @patch("app.core.image_service.generate_image")
    def test_image_generation_workflow(
        self, 
        mock_generate_image, 
        mock_create_prompt, 
        mock_queue, 
        mock_is_image_request,
        mock_chatbot, 
        mock_app, 
        mock_app_client,
        mock_say
    ):
        """Test that image generation workflow works correctly."""
        # Set up mocks
        mock_app.client = mock_app_client
        mock_queue.is_processing_sync.return_value = False
        mock_queue.start_processing_sync.return_value = True
        
        # Mock the image request detection to return True
        mock_is_image_request.return_value = True
        
        # Mock the prompt optimization
        mock_create_prompt.return_value = "Optimized prompt for a beautiful blue sky"
        
        # Mock the image generation
        mock_generate_image.return_value = (b"fake_image_data", "Revised prompt", False)
        
        # Create test event
        event = {
            "channel": "test_channel",
            "ts": "test_ts",
            "user": "test_user",
            "text": "Generate an image of a blue sky"
        }
        
        # Set up cleanup_messages with a thinking message to delete
        with patch("app.clients.slack.slack_bot.cleanup_messages", 
                  {"test_channel:test_ts": ["generating_ts"]}):
            # Call the function
            process_and_respond(event, mock_say)
            
            # Verify that say was called once with the generating message
            assert mock_say.call_count == 1
            
            # Check that the generating message was correct
            generating_msg_args = mock_say.call_args_list[0][1]
            assert "Generating image" in generating_msg_args["text"]
            
            # Verify image was generated with the optimized prompt
            mock_create_prompt.assert_called_once()
            mock_generate_image.assert_called_once_with(
                "Optimized prompt for a beautiful blue sky", 
                "test_ts", 
                mock_chatbot.get.return_value
            )
            
            # Verify the image was uploaded
            mock_app_client.files_upload_v2.assert_called_once()
            upload_args = mock_app_client.files_upload_v2.call_args[1]
            assert upload_args["channel"] == "test_channel"
            assert upload_args["thread_ts"] == "test_ts"
            assert upload_args["file"] == b"fake_image_data"
            
            # Verify the status message was cleaned up
            mock_app_client.chat_delete.assert_called_once_with(
                channel="test_channel", 
                ts="generating_ts"
            )
            
    @patch("app.clients.slack.slack_bot.app")
    @patch("app.clients.slack.slack_bot.chatbot")
    @patch("app.clients.slack.slack_bot.is_image_request")
    @patch("app.clients.slack.slack_bot.queue_manager")
    @patch("app.core.image_service.create_optimized_prompt")
    @patch("app.core.image_service.generate_image")
    def test_image_generation_error_handling(
        self, 
        mock_generate_image, 
        mock_create_prompt, 
        mock_queue, 
        mock_is_image_request,
        mock_chatbot, 
        mock_app, 
        mock_app_client,
        mock_say
    ):
        """Test that image generation errors are handled correctly."""
        # Set up mocks
        mock_app.client = mock_app_client
        mock_queue.is_processing_sync.return_value = False
        mock_queue.start_processing_sync.return_value = True
        
        # Mock the image request detection to return True
        mock_is_image_request.return_value = True
        
        # Mock the prompt optimization
        mock_create_prompt.return_value = "Optimized prompt for a beautiful blue sky"
        
        # Mock the image generation to fail
        mock_generate_image.return_value = (b"", None, True)
        
        # Create test event
        event = {
            "channel": "test_channel",
            "ts": "test_ts",
            "user": "test_user",
            "text": "Generate an image of a blue sky"
        }
        
        # Set up mock say to handle both generating message and error message
        mock_say.side_effect = [
            {"ts": "generating_ts", "message": {"ts": "generating_ts"}},
            {"ts": "error_ts", "message": {"ts": "error_ts"}}
        ]
        
        # Set up cleanup_messages with a thinking message to delete
        with patch("app.clients.slack.slack_bot.cleanup_messages", 
                  {"test_channel:test_ts": ["generating_ts"]}):
            # Call the function
            process_and_respond(event, mock_say)
            
            # Verify that say was called twice (generating + error)
            assert mock_say.call_count == 2
            
            # Check that the error message was sent
            error_msg_args = mock_say.call_args_list[1][1]
            assert ":warning:" in error_msg_args["text"]
            
            # Verify the status message was cleaned up
            mock_app_client.chat_delete.assert_called_once()
            
            # Verify image upload was NOT called
            mock_app_client.files_upload_v2.assert_not_called() 