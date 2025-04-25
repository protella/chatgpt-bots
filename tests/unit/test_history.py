import pytest
from unittest.mock import MagicMock, patch
from app.core.history import rebuild_thread_history, remove_slack_mentions

def test_remove_slack_mentions():
    """Test that Slack mentions are properly removed from text."""
    # Simple case with one mention
    text = "Hello <@U123456>!"
    assert remove_slack_mentions(text) == "Hello!"
    
    # Multiple mentions
    text = "Hi <@U123456> and <@U234567>!"
    assert remove_slack_mentions(text) == "Hi and!"
    
    # No mentions
    text = "Just a regular message"
    assert remove_slack_mentions(text) == "Just a regular message"

@patch('app.core.history.download_and_encode_image')
def test_rebuild_thread_history(mock_download_image):
    """Test that thread history is rebuilt correctly."""
    # Set up mock return for image download
    mock_download_image.return_value = "base64_encoded_image_data"
    
    # Mock the Slack client
    mock_client = MagicMock()
    
    # Mock conversations_replies response
    mock_client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U123456",
                "text": "Hello bot!",
                "ts": "1234567890.123456"
            },
            {
                "user": "BOTUSER",  # Bot user ID
                "text": "Hi there! How can I help?",
                "ts": "1234567890.123457"
            },
            {
                "user": "U123456",
                "text": "Can you analyze this image?",
                "ts": "1234567890.123458",
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
                "text": "Thinking...",  # This should be skipped
                "ts": "1234567890.123459"
            },
            {
                "user": "U123456",
                "text": "Current message",  # This is the current message and should be skipped
                "ts": "1234567890.123460"
            }
        ]
    }
    
    # Run the function
    with patch('os.environ.get', return_value='mock_token'):
        messages = rebuild_thread_history(
            client=mock_client,
            channel_id="C123456",
            thread_ts="1234567890.123456",
            bot_user_id="BOTUSER"
        )
    
    # Verify the result
    assert len(messages) == 4  # System prompt + 2 user messages + 1 assistant message
    
    # Check that the system prompt is first
    assert messages[0]["role"] == "system"
    
    # Check that the user messages are included
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assert len(user_messages) == 2
    
    # Check that content is formatted correctly
    assert any(item.get("text") == "Hello bot!" for msg in user_messages for item in msg["content"] if item["type"] == "text")
    
    # Check that the image was processed
    image_items = [item for msg in user_messages for item in msg["content"] if item["type"] == "image_url"]
    assert len(image_items) == 1
    assert "base64" in image_items[0]["image_url"]["url"]
    
    # Verify "Thinking..." message was skipped
    thinking_messages = [
        msg for msg in messages 
        if msg["role"] == "user" and any("Thinking..." in item.get("text", "") for item in msg["content"] if item["type"] == "text")
    ]
    assert len(thinking_messages) == 0 