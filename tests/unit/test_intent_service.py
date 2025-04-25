"""Unit tests for the intent detection service."""

import pytest
from unittest.mock import patch, MagicMock

from app.core.intent_service import is_image_request


class TestIntentService:
    """Test cases for the intent detection service."""

    @pytest.fixture
    def mock_openai_true(self):
        """Mock OpenAI client that returns 'True'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "True"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.return_value.chat.completions.create.return_value = mock_response
        
        return mock_client

    @pytest.fixture
    def mock_openai_false(self):
        """Mock OpenAI client that returns 'False'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "False"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.return_value.chat.completions.create.return_value = mock_response
        
        return mock_client

    @pytest.fixture
    def mock_openai_invalid(self):
        """Mock OpenAI client that returns invalid response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "I'm not sure about that."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.return_value.chat.completions.create.return_value = mock_response
        
        return mock_client

    @pytest.fixture
    def mock_openai_error(self):
        """Mock OpenAI client that raises an exception."""
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.side_effect = Exception("API error")
        
        return mock_client

    @patch('app.core.intent_service.OpenAI')
    def test_is_image_request_true(self, mock_openai, mock_openai_true):
        """Test that image requests are properly detected."""
        mock_openai.return_value = mock_openai_true.return_value
        
        config = {"temperature": 0.0}
        result = is_image_request("Draw me a cat", "thread123", config)
        
        assert result is True
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('app.core.intent_service.OpenAI')
    def test_is_image_request_false(self, mock_openai, mock_openai_false):
        """Test that non-image requests are properly detected."""
        mock_openai.return_value = mock_openai_false.return_value
        
        config = {"temperature": 0.0}
        result = is_image_request("What's the capital of France?", "thread123", config)
        
        assert result is False
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('app.core.intent_service.OpenAI')
    def test_is_image_request_invalid_response(self, mock_openai, mock_openai_invalid):
        """Test handling of invalid responses from OpenAI."""
        mock_openai.return_value = mock_openai_invalid.return_value
        
        config = {"temperature": 0.0}
        result = is_image_request("Tell me a joke", "thread123", config)
        
        # Should default to False for invalid responses
        assert result is False
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('app.core.intent_service.OpenAI')
    def test_is_image_request_error(self, mock_openai, mock_openai_error):
        """Test error handling when OpenAI raises an exception."""
        mock_openai.return_value = mock_openai_error.return_value
        
        config = {"temperature": 0.0}
        result = is_image_request("Draw me a forest", "thread123", config)
        
        # Should default to False on errors
        assert result is False
        mock_openai.return_value.chat.completions.create.assert_called_once() 