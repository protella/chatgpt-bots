"""Test multi-user conversation support"""
import pytest
from unittest.mock import MagicMock, patch
from message_processor.base import MessageProcessor
from base_client import Message
from config import config


class TestMultiUserConversation:
    """Test multi-user conversation attribution"""
    
    @pytest.fixture
    def processor(self):
        """Create a MessageProcessor instance"""
        with patch('message_processor.DocumentHandler'):
            processor = MessageProcessor()
            return processor
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client"""
        client = MagicMock()
        client.name = "TestClient"
        client.get_thread_history.return_value = []
        return client
    
    def test_format_user_content_with_username(self, processor):
        """Test username formatting helper method"""
        # Create a message with username in metadata
        message = Message(
            text="Hello world",
            user_id="U123",
            channel_id="C123",
            thread_id="T123",
            metadata={"username": "Alice"}
        )
        
        # Test normal content
        formatted = processor._format_user_content_with_username("Hello world", message)
        assert formatted == "Alice: Hello world"
        
        # Test empty content
        formatted = processor._format_user_content_with_username("", message)
        assert formatted == "Alice:"
        
        # Test bracketed content
        formatted = processor._format_user_content_with_username("[uploaded image]", message)
        assert formatted == "Alice: [uploaded image]"
    
    def test_format_user_content_without_username(self, processor):
        """Test username formatting with missing username"""
        # Message without metadata
        message = Message(
            text="Hello world",
            user_id="U123",
            channel_id="C123",
            thread_id="T123"
        )
        
        formatted = processor._format_user_content_with_username("Hello world", message)
        assert formatted == "User: Hello world"
    
    def test_thread_rebuild_with_multiple_users(self, processor, mock_client):
        """Test thread rebuilding includes usernames from different users"""
        # Mock thread history with multiple users
        history = [
            Message(
                text="Hi there!",
                user_id="U001",
                channel_id="C123",
                thread_id="T123",
                metadata={"username": "Alice", "ts": "1001"}
            ),
            Message(
                text="Hello Alice!",
                user_id="BOT",
                channel_id="C123",
                thread_id="T123",
                metadata={"is_bot": True, "ts": "1002"}
            ),
            Message(
                text="How are you?",
                user_id="U002",
                channel_id="C123",
                thread_id="T123",
                metadata={"username": "Bob", "ts": "1003"}
            )
        ]
        
        mock_client.get_thread_history.return_value = history
        
        # Current message from third user
        current_message = Message(
            text="I have a question",
            user_id="U003",
            channel_id="C123",
            thread_id="T123",
            metadata={"username": "Charlie", "ts": "1004"}
        )
        
        # Get thread state (triggers rebuild)
        thread_state = processor._get_or_rebuild_thread_state(current_message, mock_client)
        
        # Check that messages include usernames
        messages = thread_state.messages
        
        # Find user messages (excluding system messages)
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        # Verify usernames are in content
        assert any("Alice:" in msg.get("content", "") for msg in user_messages)
        assert any("Bob:" in msg.get("content", "") for msg in user_messages)
    
    def test_document_handling_with_username(self, processor):
        """Test document handling includes username"""
        message = Message(
            text="Please analyze this document",
            user_id="U123",
            channel_id="C123",
            thread_id="T123",
            metadata={"username": "David"}
        )
        
        # The username should be included in the base text before document enhancement
        # This is handled in process_message, where base_text_with_username is created
        # We can test the format helper directly
        formatted = processor._format_user_content_with_username(
            "[Attempted to upload 2 document(s) - exceeded context limit]", 
            message
        )
        assert formatted == "David: [Attempted to upload 2 document(s) - exceeded context limit]"
    
    def test_image_handling_with_username(self, processor):
        """Test image/vision handling includes username"""
        message = Message(
            text="Generate an image of a cat",
            user_id="U123",
            channel_id="C123",
            thread_id="T123",
            metadata={"username": "Eve"}
        )
        
        # Test image generation prompt formatting
        formatted = processor._format_user_content_with_username("Generate an image of a cat", message)
        assert formatted == "Eve: Generate an image of a cat"
        
        # Test image upload breadcrumb
        formatted = processor._format_user_content_with_username(
            "[uploaded image(s) for analysis]",
            message
        )
        assert formatted == "Eve: [uploaded image(s) for analysis]"
    
    def test_conversation_continuity_with_usernames(self, processor):
        """Test that conversations maintain proper user attribution"""
        messages = []
        
        # Simulate multiple users adding messages
        users = [
            ("U001", "Alice", "What's the weather like?"),
            ("U002", "Bob", "I heard it's going to rain"),
            ("U003", "Charlie", "Do you have an umbrella?")
        ]
        
        for user_id, username, text in users:
            message = Message(
                text=text,
                user_id=user_id,
                channel_id="C123",
                thread_id="T123",
                metadata={"username": username}
            )
            
            formatted = processor._format_user_content_with_username(text, message)
            assert formatted.startswith(f"{username}:")
            messages.append({"role": "user", "content": formatted})
        
        # Verify all messages have proper attribution
        assert messages[0]["content"] == "Alice: What's the weather like?"
        assert messages[1]["content"] == "Bob: I heard it's going to rain"
        assert messages[2]["content"] == "Charlie: Do you have an umbrella?"