import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.image_service import create_optimized_prompt, generate_image_description


class TestImageHistory:
    """Tests for image history and description functionality."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Optimized prompt for testing"))]
        )
        return mock_client
    
    @patch("openai.OpenAI")
    def test_create_optimized_prompt_with_history(self, mock_openai):
        """Test that create_optimized_prompt uses conversation history."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A detailed blue sky with birds"))]
        )
        mock_openai.return_value = mock_client
        
        # Create test conversation history
        conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": [{"type": "text", "text": "Show me a blue sky"}]},
            {"role": "assistant", "content": "[GENERATED IMAGE DESCRIPTION: A clear blue sky]"},
            {"role": "user", "content": [{"type": "text", "text": "Add some birds to it"}]}
        ]
        
        # Call the function
        result = create_optimized_prompt(
            "Add more clouds", 
            "test_thread", 
            {"gpt_model": "gpt-4"},
            conversation_history
        )
        
        # Verify the result
        assert result == "A detailed blue sky with birds"
        
        # Verify that history was included in the API call
        messages_arg = mock_client.chat.completions.create.call_args[1]['messages']
        system_msg = messages_arg[0]['content']
        
        # Check that the system message contains parts of the conversation history
        assert "Previous conversation" in system_msg
        assert "blue sky" in system_msg
        assert "birds" in system_msg
    
    @patch("openai.OpenAI")
    def test_generate_image_description(self, mock_openai):
        """Test the image description generation function."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A serene blue sky with fluffy white clouds and birds flying"))]
        )
        mock_openai.return_value = mock_client
        
        # Call the function
        result = generate_image_description(
            "Blue sky with clouds and birds",
            "A detailed image of a blue sky with white fluffy clouds and birds soaring",
            "test_thread"
        )
        
        # Verify the result
        assert "[GENERATED IMAGE DESCRIPTION:" in result
        assert "serene blue sky" in result
        assert "fluffy white clouds" in result
        assert "birds" in result
        
        # Verify that the revised prompt was used
        prompt_arg = mock_client.chat.completions.create.call_args[1]['messages'][0]['content']
        assert "detailed image of a blue sky" in prompt_arg
    
    @patch("openai.OpenAI")
    def test_create_optimized_prompt_fallback(self, mock_openai):
        """Test that create_optimized_prompt works without history."""
        # Set up mock to raise an exception
        mock_openai.side_effect = Exception("Test error")
        
        # Call the function with no history
        result = create_optimized_prompt(
            "Show me a beach sunset", 
            "test_thread", 
            {"gpt_model": "gpt-4"}
        )
        
        # Verify it falls back to the original prompt
        assert result == "Show me a beach sunset" 