import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
import json

# Patch the setup_logger function before importing ChatBot
with patch('app.core.logging.setup_logger') as mock_logger:
    mock_logger.return_value = MagicMock()
    # Also patch the OpenAI client to avoid any actual API calls
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        from app.core.chatbot import ChatBot

# Sample test data
SAMPLE_TEXT_INPUT = "Hello, how are you?"
SAMPLE_THREAD_ID = "thread_12345"
SAMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock OpenAI response."""
    mock = MagicMock()
    mock.id = "resp_12345"
    mock.choices = [MagicMock()]
    mock.choices[0].message = MagicMock()
    mock.choices[0].message.content = "I'm doing well, thank you for asking!"
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 50
    mock.usage.completion_tokens = 30
    mock.usage.total_tokens = 80
    return mock

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch('openai.OpenAI', autospec=True) as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_completions = MagicMock()
        mock_client.chat.completions.create = mock_completions
        yield mock_client

@pytest.fixture
def chatbot(mock_openai_client) -> ChatBot:
    """Create a ChatBot instance with a mock API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        return ChatBot(api_key="test-api-key")

class TestChatBot:
    """Test suite for the ChatBot class."""

    def test_init_missing_api_key(self) -> None:
        """Test that initialization fails when API key is missing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            with patch('openai.OpenAI'):  # Prevent actual client creation
                with pytest.raises(ValueError) as exc_info:
                    ChatBot()
                assert "API key is required" in str(exc_info.value)

    def test_init_with_explicit_api_key(self) -> None:
        """Test initialization with an explicit API key."""
        with patch('openai.OpenAI'):  # Prevent actual client creation
            bot = ChatBot(api_key="explicit-test-key")
            assert bot.api_key == "explicit-test-key"

    def test_get_response_text_only(self, chatbot, mock_response, mock_openai_client) -> None:
        """Test getting a text-only response."""
        # Setup mock
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Call the method
        response = chatbot.get_response(SAMPLE_TEXT_INPUT, SAMPLE_THREAD_ID)
        
        # Verify the response
        assert response["success"] is True
        assert response["content"] == "I'm doing well, thank you for asking!"
        assert response["error"] is None
        
        # Verify the API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        
        # Verify parameters
        assert call_kwargs["model"] == "gpt-4.1-2025-04-14"
        assert call_kwargs["store"] is True
        assert "previous_response_id" not in call_kwargs
        
        # Verify messages format
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"  # System prompt
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) == 1  # Text only
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][0]["text"] == SAMPLE_TEXT_INPUT
        
        # Verify token usage tracking
        assert SAMPLE_THREAD_ID in chatbot.token_usage
        assert chatbot.token_usage[SAMPLE_THREAD_ID]["prompt_tokens"] == 50
        assert chatbot.token_usage[SAMPLE_THREAD_ID]["completion_tokens"] == 30
        assert chatbot.token_usage[SAMPLE_THREAD_ID]["total_tokens"] == 80
        
        # Verify response ID storage
        assert chatbot.thread_responses[SAMPLE_THREAD_ID] == "resp_12345"

    def test_get_response_with_images(self, chatbot, mock_response, mock_openai_client) -> None:
        """Test getting a response with image inputs."""
        # Setup mock
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Call with a single image
        response = chatbot.get_response(
            SAMPLE_TEXT_INPUT, 
            SAMPLE_THREAD_ID, 
            images=[SAMPLE_IMAGE_BASE64]
        )
        
        # Verify success
        assert response["success"] is True
        
        # Verify the API call
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        
        # Check message format with image
        user_message = call_kwargs["messages"][1]
        assert len(user_message["content"]) == 2  # Text + 1 image
        assert user_message["content"][0]["type"] == "text"
        assert user_message["content"][1]["type"] == "image_url"
        assert f"data:image/png;base64,{SAMPLE_IMAGE_BASE64}" == user_message["content"][1]["image_url"]["url"]
        assert user_message["content"][1]["image_url"]["detail"] == "auto"

    def test_get_response_with_multiple_images(self, chatbot, mock_response, mock_openai_client) -> None:
        """Test getting a response with multiple image inputs."""
        # Setup mock
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Multiple images
        images = [SAMPLE_IMAGE_BASE64, SAMPLE_IMAGE_BASE64, SAMPLE_IMAGE_BASE64]
        
        # Call with multiple images
        response = chatbot.get_response(
            SAMPLE_TEXT_INPUT, 
            SAMPLE_THREAD_ID, 
            images=images
        )
        
        # Verify success
        assert response["success"] is True
        
        # Verify the API call
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        
        # Check message format with images
        user_message = call_kwargs["messages"][1]
        assert len(user_message["content"]) == 4  # Text + 3 images
        assert user_message["content"][0]["type"] == "text"
        
        # Verify each image is included
        for i in range(1, 4):
            assert user_message["content"][i]["type"] == "image_url"
            assert f"data:image/png;base64,{SAMPLE_IMAGE_BASE64}" == user_message["content"][i]["image_url"]["url"]

    def test_conversation_continuation(self, chatbot, mock_response, mock_openai_client) -> None:
        """Test that conversation context is maintained between messages."""
        # Setup mock
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # First message in thread (sets up the conversation)
        chatbot.get_response(SAMPLE_TEXT_INPUT, SAMPLE_THREAD_ID)
        
        # Reset the mock to track the second call separately
        mock_openai_client.chat.completions.create.reset_mock()
        
        # Second message in the same thread
        follow_up_text = "What's the weather like today?"
        chatbot.get_response(follow_up_text, SAMPLE_THREAD_ID)
        
        # Verify the second API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        
        # Verify previous_response_id is included for the continuation
        assert "previous_response_id" in call_kwargs
        assert call_kwargs["previous_response_id"] == "resp_12345"
        
        # Verify system prompt is NOT included in the follow-up
        messages = call_kwargs["messages"]
        assert len(messages) == 1  # Only user message, no system prompt
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == follow_up_text

    def test_error_handling(self, chatbot, mock_openai_client) -> None:
        """Test that API errors are properly handled."""
        # Setup mock to raise an exception
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error: Rate limit exceeded")
        
        # Call the method
        response = chatbot.get_response(SAMPLE_TEXT_INPUT, SAMPLE_THREAD_ID)
        
        # Verify error handling
        assert response["success"] is False
        assert response["content"] == ""
        assert "API Error" in response["error"]

    def test_get_token_usage(self, chatbot) -> None:
        """Test retrieving token usage statistics."""
        # Setup mock data
        chatbot.token_usage[SAMPLE_THREAD_ID] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        # Get usage
        usage = chatbot.get_token_usage(SAMPLE_THREAD_ID)
        
        # Verify
        assert usage is not None
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        
        # Test non-existent thread
        assert chatbot.get_token_usage("non_existent_thread") is None 