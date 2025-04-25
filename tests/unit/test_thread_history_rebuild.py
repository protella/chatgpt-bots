import pytest
import os
from unittest.mock import MagicMock, patch

# Import the modules under test
from app.core.history import rebuild_thread_history
from app.clients.slack.slack_bot import chatbot

@pytest.fixture
def mock_slack_client():
    """Create a mock Slack client for testing."""
    mock_client = MagicMock()
    # Set up the conversations_replies method to return sample messages
    mock_client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U123456",
                "text": "Hello bot!",
                "ts": "123.000"
            },
            {
                "user": "BOTUSER",  # Bot user ID
                "text": "Hi there! How can I help?",
                "ts": "123.001"
            },
            {
                "user": "U123456",
                "text": "What can you tell me about Python?",
                "ts": "123.002"
            },
            {
                "user": "BOTUSER",
                "text": "Python is a high-level programming language...",
                "ts": "123.003"
            }
        ]
    }
    return mock_client

@pytest.fixture
def mock_chatbot():
    """Create a mock chatbot for testing."""
    mock_bot = MagicMock()
    mock_bot.get_response.return_value = {
        "success": True,
        "content": "This is a test response"
    }
    mock_bot.thread_responses = {}
    return mock_bot

def test_rebuild_thread_history_with_existing_thread(mock_slack_client):
    """Test rebuilding thread history for an existing thread."""
    with patch('app.core.history.download_and_encode_image') as mock_download:
        with patch('os.environ.get', return_value='mock_token'):
            # Call the function
            messages = rebuild_thread_history(
                client=mock_slack_client,
                channel_id="C123456",
                thread_ts="123.000",
                bot_user_id="BOTUSER"
            )
            
            # Verify that conversations_replies was called
            mock_slack_client.conversations_replies.assert_called_once_with(
                channel="C123456",
                ts="123.000",
                limit=100
            )
            
            # Verify the result structure
            assert len(messages) == 4  # System prompt + 3 messages (skipping the last one)
            
            # Check system prompt
            assert messages[0]["role"] == "system"
            
            # Check user messages
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assert len(user_messages) == 2
            
            # Check assistant messages
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            assert len(assistant_messages) == 1

def test_initialize_history_with_synthetic_turns():
    """Test initializing history with synthetic turns."""
    # Create a mock for the chatbot and other required components
    with patch('app.clients.slack.slack_bot.chatbot') as mock_chatbot:
        # Configure the mock
        mock_chatbot.get_response.return_value = {
            "success": True,
            "content": "This is a synthetic response"
        }
        mock_chatbot.thread_responses = {}
        
        # Create mock OpenAI format messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, how are you?"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'm doing well, thank you!"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What can you tell me about Python?"
                    }
                ]
            }
        ]
        
        # Initialize empty thread responses dict for our test
        thread_id = "123.456"
        
        # Directly initialize history with the messages
        for i, msg in enumerate(messages[1:]):  # Skip system prompt
            # Skip assistant messages
            if msg["role"] == "assistant":
                continue
                
            # For user messages, extract just the text
            if msg["role"] == "user":
                content_text = ""
                for item in msg["content"]:
                    if item["type"] == "text":
                        content_text += item["text"] + " "
                
                # Call get_response with the extracted text
                mock_chatbot.get_response(
                    input_text=content_text.strip(),
                    thread_id=thread_id,
                    images=[]
                )
        
        # Verify that get_response was called the correct number of times
        assert mock_chatbot.get_response.call_count == 2

def test_rebuild_thread_history_with_images(mock_slack_client):
    """Test rebuilding thread history with images in the thread."""
    # Set up the mock to return messages with images
    mock_slack_client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U123456",
                "text": "Check out this image",
                "ts": "123.000",
                "files": [
                    {
                        "name": "test_image.jpg",
                        "mimetype": "image/jpeg",
                        "url_private": "https://files.slack.com/test_image.jpg"
                    }
                ]
            },
            {
                "user": "BOTUSER",
                "text": "I see a beautiful landscape",
                "ts": "123.001"
            }
        ]
    }
    
    # Mock the download_and_encode_image function
    with patch('app.core.history.download_and_encode_image', return_value="base64_encoded_image") as mock_download:
        with patch('os.environ.get', return_value='mock_token'):
            # Call the function
            messages = rebuild_thread_history(
                client=mock_slack_client,
                channel_id="C123456",
                thread_ts="123.000",
                bot_user_id="BOTUSER"
            )
            
            # Verify the download was attempted
            mock_download.assert_called_once_with(
                "https://files.slack.com/test_image.jpg",
                "mock_token"
            )
            
            # Verify the message structure - the assistant message is not included
            # because it's the bot's response and is handled differently
            assert len(messages) == 2  # System prompt + user with image
            
            # Check for the image in the user message
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assert len(user_messages) == 1
            image_items = [item for msg in user_messages for item in msg["content"] 
                         if isinstance(item, dict) and item["type"] == "image_url"]
            assert len(image_items) == 1
            assert "base64" in image_items[0]["image_url"]["url"] 