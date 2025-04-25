import pytest
import os
import time
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock the Slack App class before importing any modules
mock_app = MagicMock()
mock_app.client = MagicMock()
mock_app.client.auth_test = MagicMock(return_value={"user_id": "U12345"})
mock_app.client.chat_delete = MagicMock()
mock_app.client.users_info = MagicMock(return_value={
    "user": {
        "profile": {
            "first_name": "Alex"
        }
    }
})
mock_app.client.conversations_replies = MagicMock(return_value={
    "messages": [
        {
            "user": "U12345",
            "text": "Hello bot!",
            "ts": "123.000"
        },
        {
            "user": "U12345BOT",
            "text": "Hi there! How can I help?",
            "ts": "123.001"
        }
    ]
})

# Create a patched App class that doesn't try to authenticate
class MockApp:
    def __init__(self, token=None):
        self.client = mock_app.client
        self.event_handlers = {}
        self.command_handlers = {}
        self.action_handlers = {}
        
    def event(self, event_type):
        def decorator(func):
            self.event_handlers[event_type] = func
            return func
        return decorator
        
    def command(self, command):
        def decorator(func):
            self.command_handlers[command] = func
            return func
        return decorator
        
    def action(self, action_id):
        def decorator(func):
            self.action_handlers[action_id] = func
            return func
        return decorator

# First just import the module for testing, without replacing anything
with patch.dict(os.environ, {
    "SLACK_BOT_TOKEN": "xoxb-test-token",
    "SLACK_APP_TOKEN": "xapp-test-token",
    "OPENAI_API_KEY": "sk-test-key"
}):
    with patch.object(sys, 'exit'):
        with patch('slack_bolt.App', MockApp):
            # Now we can safely import from slack_bot
            from app.clients.slack import slack_bot
            
            # Get the key functions we'll be testing
            handle_app_mention = slack_bot.handle_app_mention
            handle_message = slack_bot.handle_message
            handle_config_command = slack_bot.handle_config_command
            handle_reset_action = slack_bot.handle_reset_action
            process_and_respond = slack_bot.process_and_respond

@pytest.fixture
def mock_say():
    """Mock for Slack's say function."""
    return MagicMock(return_value={"ts": "123.456"})

@pytest.fixture
def mock_ack():
    """Mock for Slack's ack function."""
    return MagicMock()

@pytest.fixture
def mock_respond():
    """Mock for Slack's respond function."""
    return MagicMock()

@pytest.fixture
def app_mention_event():
    """Sample app_mention event data."""
    return {
        "channel": "C12345",
        "ts": "123.456",
        "user": "U12345",
        "text": "<@U12345> Hello, can you help me?",
    }

@pytest.fixture
def direct_message_event():
    """Sample direct message event data."""
    return {
        "channel": "D12345",
        "channel_type": "im",  # Direct message channel
        "ts": "123.456",
        "user": "U12345",
        "text": "Hello, can you help me?",
    }

@pytest.fixture
def command_body():
    """Sample slash command body."""
    return {
        "channel_id": "C12345",
        "user_id": "U12345",
        "text": "",
        "response_url": "https://hooks.slack.com/commands/1234/5678",
    }

@pytest.fixture
def action_body():
    """Sample action body for button clicks."""
    return {
        "user": {"id": "U12345"},
        "channel": {"id": "C12345"},
        "message": {
            "ts": "123.456",
            "text": "Thread ID: thread_123\nStatus: Active",
        },
    }

def test_handle_app_mention(app_mention_event, mock_say):
    """Test handling of app mention events."""
    # Patch the process_and_respond function to avoid full processing
    with patch('app.clients.slack.slack_bot.process_and_respond') as mock_process:
        # Call the event handler directly
        handle_app_mention(app_mention_event, mock_say)
        
        # Check that process_and_respond was called with correct args
        mock_process.assert_called_once_with(app_mention_event, mock_say)

def test_handle_app_mention_bot_message(app_mention_event, mock_say):
    """Test ignoring bot messages in app mentions."""
    # Add bot_id to make this a bot message
    app_mention_event["bot_id"] = "B12345"
    
    # Patch the process_and_respond function
    with patch('app.clients.slack.slack_bot.process_and_respond') as mock_process:
        # Call the event handler directly
        handle_app_mention(app_mention_event, mock_say)
        
        # process_and_respond should not be called for bot messages
        mock_process.assert_not_called()

def test_handle_direct_message(direct_message_event, mock_say):
    """Test handling of direct messages."""
    # Patch the process_and_respond function
    with patch('app.clients.slack.slack_bot.process_and_respond') as mock_process:
        # Call the event handler directly
        handle_message(direct_message_event, mock_say)
        
        # Check that process_and_respond was called with correct args
        mock_process.assert_called_once_with(direct_message_event, mock_say)

def test_handle_non_dm_message(mock_say):
    """Test that non-DM messages are ignored."""
    # Create a message event that's not a direct message
    channel_message = {
        "channel": "C12345",
        "channel_type": "channel",  # Not a direct message
        "ts": "123.456",
        "user": "U12345",
        "text": "Hello, can you help me?",
    }
    
    # Patch the process_and_respond function
    with patch('app.clients.slack.slack_bot.process_and_respond') as mock_process:
        # Call the event handler directly
        handle_message(channel_message, mock_say)
        
        # process_and_respond should not be called for non-DM messages
        mock_process.assert_not_called()

def test_config_command(mock_ack, command_body, mock_respond):
    """Test the /chatgpt-config-dev command handler."""
    # Patch the config_service
    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
        # Setup mock config to return test data
        mock_config.get.return_value = {
            "gpt_model": "gpt-4.1-2025-04-14",
            "temperature": 0.7,
            "top_p": 1.0,
            "max_output_tokens": 4096,
            "image_model": "gpt-4.1-2025-04-14",
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid",
            "number": 1,
            "detail": "auto"
        }
        
        # Call the command handler directly
        handle_config_command(mock_ack, command_body, mock_respond)
        
        # Check that the command was acknowledged
        mock_ack.assert_called_once()
        
        # Check that respond was called with blocks
        mock_respond.assert_called_once()
        args = mock_respond.call_args[0][0]
        assert "response_type" in args
        assert "blocks" in args
        assert len(args["blocks"]) > 0

def test_config_command_reset(mock_ack, command_body, mock_respond):
    """Test the /chatgpt-config-dev command with reset option."""
    # Add reset text to the command
    command_body["text"] = "reset"
    
    # Patch the config_service
    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
        # Call the command handler directly
        handle_config_command(mock_ack, command_body, mock_respond)
        
        # Check that reset was called
        mock_config.reset.assert_called_once()
        
        # Check that respond was called with the right message
        mock_respond.assert_called_once()
        args = mock_respond.call_args[0][0]
        assert "Configuration has been reset" in args["text"]

def test_reset_config_action(mock_ack, action_body, mock_respond):
    """Test the reset_config button action handler."""
    # Patch the config_service
    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
        # Call the action handler directly
        handle_reset_action(mock_ack, action_body, mock_respond)
        
        # Check that the action was acknowledged
        mock_ack.assert_called_once()
        
        # Check that reset was called
        mock_config.reset.assert_called_once()
        
        # Check that respond was called
        mock_respond.assert_called_once()

def test_image_processing(mock_say, direct_message_event):
    """Test processing messages with image attachments."""
    # Add image file to the event
    direct_message_event["files"] = [
        {
            "name": "test_image.jpg",
            "mimetype": "image/jpeg",
            "url_private": "https://files.slack.com/test_image.jpg"
        }
    ]
    
    # Mock all the required components
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager') as mock_queue:
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                        with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                            with patch('app.core.history.download_and_encode_image', return_value="base64encoded_image") as mock_download:
                                # Configure the mock behavior
                                mock_queue.is_processing_sync.return_value = False
                                mock_queue.start_processing_sync.return_value = True
                                mock_chatbot.get_response.return_value = {
                                    "success": True,
                                    "content": "I see an image of a cat."
                                }
                                mock_config.get.return_value = {}
                                mock_config.extract_config_from_text.return_value = None
                                
                                # Call process_and_respond
                                process_and_respond(direct_message_event, mock_say)
                                
                                # Check that the image was downloaded and encoded
                                mock_download.assert_called_once_with(
                                    "https://files.slack.com/test_image.jpg",
                                    "xoxb-test-token"
                                )
                                
                                # Check that the chatbot was called with the image
                                mock_chatbot.get_response.assert_called_once()
                                args, kwargs = mock_chatbot.get_response.call_args
                                assert len(kwargs["images"]) == 1
                                assert kwargs["images"][0] == "base64encoded_image"
                                
                                # Check that the response was sent
                                mock_say.assert_called_with(
                                    text="I see an image of a cat.",
                                    thread_ts="123.456"
                                )

def test_image_processing_error(mock_say, direct_message_event):
    """Test handling errors when processing images."""
    # Add image file to the event
    direct_message_event["files"] = [
        {
            "name": "test_image.jpg",
            "mimetype": "image/jpeg",
            "url_private": "https://files.slack.com/test_image.jpg"
        }
    ]
    
    # Mock all the required components
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager') as mock_queue:
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                        with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                            with patch('app.core.history.download_and_encode_image', side_effect=Exception("Download error")) as mock_download:
                                # Configure the mock behavior
                                mock_queue.is_processing_sync.return_value = False
                                mock_queue.start_processing_sync.return_value = True
                                mock_chatbot.get_response.return_value = {
                                    "success": True,
                                    "content": "I processed your text without the image."
                                }
                                mock_config.get.return_value = {}
                                mock_config.extract_config_from_text.return_value = None
                                
                                # Call process_and_respond
                                process_and_respond(direct_message_event, mock_say)
                                
                                # Check that download was attempted
                                mock_download.assert_called_once()
                                
                                # The error should be caught and processing should continue without the image
                                mock_chatbot.get_response.assert_called_once()
                                args, kwargs = mock_chatbot.get_response.call_args
                                assert len(kwargs["images"]) == 0
                                
                                # Check that the response was sent
                                mock_say.assert_called_with(
                                    text="I processed your text without the image.",
                                    thread_ts="123.456"
                                )

def test_intent_detection(mock_say, direct_message_event):
    """Test intent detection for image generation requests."""
    # Mock all the required components
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager') as mock_queue:
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                        with patch('app.clients.slack.slack_bot.is_image_request', return_value=True) as mock_intent:
                            with patch('app.clients.slack.slack_bot.remove_slack_mentions', return_value="Hello, can you help me?"):
                                # Configure the mock behavior
                                mock_queue.is_processing_sync.return_value = False
                                mock_queue.start_processing_sync.return_value = True
                                mock_chatbot.get_response.return_value = {
                                    "success": True,
                                    "content": "Here's the image you requested."
                                }
                                mock_config.get.return_value = {}
                                mock_config.extract_config_from_text.return_value = None
                                
                                # Call process_and_respond
                                process_and_respond(direct_message_event, mock_say)
                                
                                # Check that intent detection was called with the right message
                                # Note: We know it's called with the personalized message, so test for that
                                mock_intent.assert_called_once()
                                
                                # Check that the chatbot was called with the right input
                                mock_chatbot.get_response.assert_called_once()
                                
                                # Check that the response was sent
                                mock_say.assert_called_with(
                                    text="Here's the image you requested.",
                                    thread_ts="123.456"
                                )

def test_error_handling(mock_say, direct_message_event):
    """Test error handling in process_and_respond."""
    # Mock components with an error from the chatbot
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager') as mock_queue:
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                        with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                            # Configure mock to throw an error
                            mock_queue.is_processing_sync.return_value = False
                            mock_queue.start_processing_sync.return_value = True
                            mock_chatbot.get_response.return_value = {
                                "success": False,
                                "error": "API error"
                            }
                            mock_config.get.return_value = {}
                            mock_config.extract_config_from_text.return_value = None
                            
                            # Call process_and_respond
                            process_and_respond(direct_message_event, mock_say)
                            
                            # Should send an error message
                            mock_say.assert_called_with(
                                text=":warning: Something went wrong. Please try again or contact support.",
                                thread_ts="123.456"
                            )
                            
                            # Queue should be released
                            mock_queue.finish_processing_sync.assert_called_once_with(direct_message_event["ts"])

def test_exception_handling(mock_say, direct_message_event):
    """Test exception handling in process_and_respond."""
    # Mock components with an exception during processing
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager') as mock_queue:
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', side_effect=Exception("Test exception")):
                    # Configure mock
                    mock_queue.is_processing_sync.return_value = False
                    mock_queue.start_processing_sync.return_value = True
                    
                    # Call process_and_respond
                    process_and_respond(direct_message_event, mock_say)
                    
                    # Should send an error message
                    mock_say.assert_called_with(
                        text=":warning: Something went wrong. Please try again or contact support.",
                        thread_ts="123.456"
                    )
                    
                    # Queue should be released
                    mock_queue.finish_processing_sync.assert_called_once_with(direct_message_event["ts"]) 