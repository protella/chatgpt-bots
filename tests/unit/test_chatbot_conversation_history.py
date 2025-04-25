import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.chatbot import ChatBot


class TestChatbotConversationHistory:
    """Test class for verifying ChatBot conversation history management."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            id="mock-response-id",
            choices=[MagicMock(message=MagicMock(content="Test assistant response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        return mock_client

    @pytest.fixture
    def chatbot(self, mock_openai_client):
        """Create a ChatBot instance with a mocked OpenAI client."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            bot = ChatBot(api_key="mock-api-key")
            return bot

    def test_new_conversation(self, chatbot, mock_openai_client):
        """Test starting a new conversation."""
        # Get a response for a new thread
        thread_id = "test_thread_1"
        result = chatbot.get_response("Hello", thread_id)
        
        # Verify the result
        assert result["success"] is True
        assert result["content"] == "Test assistant response"
        
        # Verify the conversation was stored
        assert thread_id in chatbot.conversations
        assert len(chatbot.conversations[thread_id]["messages"]) == 3  # system + user + assistant
        assert chatbot.conversations[thread_id]["response_id"] == "mock-response-id"
        
        # Check that the OpenAI API was called with the correct messages
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert len(calls) == 1
        call_kwargs = calls[0][1]
        assert len(call_kwargs["messages"]) == 2  # system + user (not assistant yet)
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"][0]["text"] == "Hello"

    def test_continuing_conversation(self, chatbot, mock_openai_client):
        """Test continuing an existing conversation with multiple turns."""
        # Start a conversation
        thread_id = "test_thread_2"
        chatbot.get_response("First message", thread_id)
        
        # Continue the conversation
        result = chatbot.get_response("Second message", thread_id)
        
        # Verify the result
        assert result["success"] is True
        
        # Verify the conversation history was updated correctly
        conversation = chatbot.conversations[thread_id]
        assert len(conversation["messages"]) == 5  # system + user1 + assistant1 + user2 + assistant2
        
        # Check system message
        assert conversation["messages"][0]["role"] == "system"
        
        # Check first user message
        assert conversation["messages"][1]["role"] == "user"
        assert conversation["messages"][1]["content"][0]["text"] == "First message"
        
        # Check first assistant response
        assert conversation["messages"][2]["role"] == "assistant"
        assert conversation["messages"][2]["content"] == "Test assistant response"
        
        # Check second user message
        assert conversation["messages"][3]["role"] == "user"
        assert conversation["messages"][3]["content"][0]["text"] == "Second message"
        
        # Check second assistant response
        assert conversation["messages"][4]["role"] == "assistant"
        assert conversation["messages"][4]["content"] == "Test assistant response"
        
        # Verify the second API call included all previous messages
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert len(calls) == 2
        second_call_kwargs = calls[1][1]
        assert len(second_call_kwargs["messages"]) == 4  # system + user1 + assistant1 + user2

    def test_initialize_from_history(self, chatbot):
        """Test initializing a conversation from existing message history."""
        thread_id = "test_thread_3"
        # Create some message history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Previous message 1"}]},
            {"role": "assistant", "content": "Previous response 1"},
            {"role": "user", "content": [{"type": "text", "text": "Previous message 2"}]},
            {"role": "assistant", "content": "Previous response 2"}
        ]
        
        # Initialize from history
        chatbot.initialize_from_history(thread_id, messages)
        
        # Verify the conversation was stored correctly
        assert thread_id in chatbot.conversations
        assert len(chatbot.conversations[thread_id]["messages"]) == 5
        assert chatbot.conversations[thread_id]["response_id"] is None
        
        # Continue the conversation
        result = chatbot.get_response("New message", thread_id)
        
        # Verify the result
        assert result["success"] is True
        
        # Verify the conversation history was updated correctly
        conversation = chatbot.conversations[thread_id]
        assert len(conversation["messages"]) == 7  # Previous 5 + new user + new assistant
        
        # Check that response_id was set
        assert conversation["response_id"] == "mock-response-id"

    def test_conversation_with_images(self, chatbot, mock_openai_client):
        """Test a conversation with image attachments."""
        thread_id = "test_thread_4"
        mock_image_data = "mock-base64-data"
        
        # Get a response with an image
        result = chatbot.get_response("Image description", thread_id, images=[mock_image_data])
        
        # Verify the result
        assert result["success"] is True
        
        # Verify the conversation with image was stored
        conversation = chatbot.conversations[thread_id]
        assert len(conversation["messages"]) == 3  # system + user + assistant
        
        # Check user message includes both text and image
        user_message = conversation["messages"][1]
        assert user_message["role"] == "user"
        assert len(user_message["content"]) == 2  # text + image
        assert user_message["content"][0]["type"] == "text"
        assert user_message["content"][1]["type"] == "image_url"
        assert mock_image_data in user_message["content"][1]["image_url"]["url"]
        
        # Verify the API call included the image
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"][1]["content"]) == 2  # text + image 

    def test_personalization_tags_handled_correctly(self, chatbot, mock_openai_client):
        """Test that personalization tags are correctly handled."""
        # Start a conversation with personalization tags
        thread_id = "test_thread_personalization"
        
        # For a normal message, personalization tags should be preserved
        result1 = chatbot.get_response("[username=Peter] First message", thread_id)
        
        # Check that normal message was sent with personalization tag intact
        first_call_kwargs = mock_openai_client.chat.completions.create.call_args_list[0][1]
        first_user_message = first_call_kwargs["messages"][1]
        assert "username=Peter" in first_user_message["content"][0]["text"]
        
        # For a repeat request, personalization tags should be removed
        result2 = chatbot.get_response("Now repeat this whole conversation back to me", thread_id)
        
        # Verify OpenAI was called with messages that don't have personalization tags
        second_call_kwargs = mock_openai_client.chat.completions.create.call_args_list[1][1]
        second_messages = second_call_kwargs["messages"]
        
        # The repeat request itself doesn't have tags, so shouldn't be filtered
        assert second_messages[3]["role"] == "user"
        assert "repeat this whole conversation" in second_messages[3]["content"][0]["text"]
        
        # But the first message should have had its tags removed for the repeat
        for i, message in enumerate(second_messages):
            if i == 1:  # First user message
                assert message["role"] == "user"
                assert "[username=Peter]" not in message["content"][0]["text"]
                assert "First message" in message["content"][0]["text"] 