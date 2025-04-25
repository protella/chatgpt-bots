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

# Create a patched App class that doesn't try to authenticate
class MockApp:
    def __init__(self, token=None):
        self.client = mock_app.client
        
    def event(self, event_type):
        def decorator(func):
            return func
        return decorator
        
    def command(self, command):
        def decorator(func):
            return func
        return decorator
        
    def action(self, action_id):
        def decorator(func):
            return func
        return decorator

# Patch modules
with patch.dict(os.environ, {
    "SLACK_BOT_TOKEN": "xoxb-test-token",
    "SLACK_APP_TOKEN": "xapp-test-token",
    "OPENAI_API_KEY": "sk-test-key"
}):
    with patch.object(sys, 'exit'):
        with patch('slack_bolt.App', MockApp):
            from app.core.queue import QueueManager
            # Now we can safely import from slack_bot
            from app.clients.slack import slack_bot
            
            # Replace the app with our mock
            slack_bot.app = mock_app
            
            # Access the functions we need to test
            clean_temp_messages = slack_bot.clean_temp_messages
            process_and_respond = slack_bot.process_and_respond
            queue_manager = slack_bot.queue_manager
            cleanup_messages = slack_bot.cleanup_messages

@pytest.fixture
def mock_say():
    """Mock for Slack's say function."""
    return MagicMock(return_value={"ts": "123.456"})

@pytest.fixture
def mock_queue_manager():
    """Mock for the QueueManager."""
    mock = MagicMock()
    mock.is_processing_sync = MagicMock(return_value=False)
    mock.start_processing_sync = MagicMock(return_value=True)
    mock.finish_processing_sync = MagicMock()
    return mock

@pytest.fixture
def event_data():
    """Sample Slack event data for testing."""
    return {
        "channel": "C12345",
        "ts": "123.456",
        "user": "U12345",
        "text": "Hello, can you help me?",
    }

@pytest.fixture
def threaded_event_data():
    """Sample Slack event data for a threaded message."""
    return {
        "channel": "C12345",
        "ts": "123.456",
        "thread_ts": "123.000",
        "user": "U12345",
        "text": "Hello, can you help me?",
    }

@pytest.mark.parametrize("is_busy", [True, False])
def test_busy_messaging(mock_say, mock_queue_manager, event_data, is_busy, monkeypatch):
    """Test busy messaging when thread is already being processed."""
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager', mock_queue_manager):
            with patch('app.clients.slack.slack_bot.clean_temp_messages') as mock_clean:
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                    with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                        with patch('app.clients.slack.slack_bot.rebuild_thread_history') as mock_rebuild:
                            with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                                # Configure the mock behavior
                                mock_queue_manager.is_processing_sync.return_value = is_busy
                                mock_chatbot.get_response.return_value = {
                                    "success": True,
                                    "content": "This is a response"
                                }
                                mock_config.get.return_value = {}
                                mock_config.extract_config_from_text.return_value = None
                                
                                # Call the function
                                process_and_respond(event_data, mock_say)
                                
                                # Check the behavior
                                if is_busy:
                                    # If busy, should send the busy message and not process further
                                    mock_say.assert_called_once_with(
                                        text="I'm still working on your last request. Please wait a moment and try again.",
                                        thread_ts="123.456"
                                    )
                                    mock_queue_manager.start_processing_sync.assert_not_called()
                                    mock_chatbot.get_response.assert_not_called()
                                else:
                                    # If not busy, should process normally
                                    mock_queue_manager.start_processing_sync.assert_called_once_with("123.456")
                                    assert mock_say.call_count >= 2  # Thinking + response
                                    mock_chatbot.get_response.assert_called_once()
                                    mock_queue_manager.finish_processing_sync.assert_called_once_with("123.456")

def test_personalization(mock_say, mock_queue_manager, event_data):
    """Test that user personalization is correctly added to messages."""
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.queue_manager', mock_queue_manager):
            with patch('app.clients.slack.slack_bot.clean_temp_messages'):
                with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex") as mock_get_user:
                    with patch('app.clients.slack.slack_bot.remove_slack_mentions', return_value="Hello, can you help me?"):
                        with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                            with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                                # Configure mocks
                                mock_config.get.return_value = {}
                                mock_config.extract_config_from_text.return_value = None
                                mock_chatbot.get_response.return_value = {
                                    "success": True,
                                    "content": "Hi Alex, I can help you!"
                                }
                                
                                # Call the function
                                process_and_respond(event_data, mock_say)
                                
                                # Check that personalization was added
                                mock_get_user.assert_called_once_with(mock_app.client, "U12345")
                                
                                # Verify the message sent to OpenAI includes the personalization tag
                                expected_text = "[username=Alex] Hello, can you help me?"
                                mock_chatbot.get_response.assert_called_once()
                                args, kwargs = mock_chatbot.get_response.call_args
                                assert kwargs["input_text"] == expected_text

def test_clean_temp_messages():
    """Test that temporary messages are properly cleaned up."""
    # Setup test data
    channel_id = "C12345"
    thread_ts = "123.456"
    key = f"{channel_id}:{thread_ts}"
    
    # Add some messages to clean up
    global cleanup_messages
    cleanup_messages[key] = ["111.111", "222.222", "333.333"]
    
    # Call the function
    clean_temp_messages(channel_id, thread_ts)
    
    # Check that the client's delete method was called for each message
    assert mock_app.client.chat_delete.call_count == 3
    
    # Check that the cleanup list is now empty
    assert key in cleanup_messages
    assert len(cleanup_messages[key]) == 0
    
    # Test error handling
    mock_app.client.chat_delete.side_effect = Exception("API error")
    cleanup_messages[key] = ["444.444"]
    
    # This should not raise an exception
    clean_temp_messages(channel_id, thread_ts)
    
    # List should still be empty after cleanup
    assert len(cleanup_messages[key]) == 0

def test_concurrent_threads(mock_say, event_data, threaded_event_data):
    """Test that different threads can be processed concurrently."""
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        with patch('app.clients.slack.slack_bot.clean_temp_messages'):
            with patch('app.clients.slack.slack_bot.get_user_info', return_value="Alex"):
                with patch('app.clients.slack.slack_bot.config_service') as mock_config:
                    with patch('app.clients.slack.slack_bot.is_image_request', return_value=False):
                        # Reset the queue manager to a real instance for this test
                        real_queue = QueueManager()
                        with patch('app.clients.slack.slack_bot.queue_manager', real_queue):
                            # Configure mocks
                            mock_config.get.return_value = {}
                            mock_config.extract_config_from_text.return_value = None
                            mock_chatbot.get_response.return_value = {
                                "success": True,
                                "content": "This is a response"
                            }
                            
                            # Start processing the first thread
                            assert real_queue.start_processing_sync("123.000")
                            
                            # Try to process the same thread again - should return busy
                            assert real_queue.is_processing_sync("123.000")
                            
                            # Different thread should be able to start processing
                            assert not real_queue.is_processing_sync("123.456")
                            assert real_queue.start_processing_sync("123.456")
                            
                            # Clean up
                            real_queue.finish_processing_sync("123.000")
                            real_queue.finish_processing_sync("123.456")
                            
                            # Both threads should now be available
                            assert not real_queue.is_processing_sync("123.000")
                            assert not real_queue.is_processing_sync("123.456") 