"""
Unit tests for base_client.py module
Tests the abstract base client for platform implementations
"""
import pytest
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from unittest.mock import MagicMock, patch, Mock
from base_client import BaseClient, Message, Response


class TestMessage:
    """Test Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a Message instance"""
        message = Message(
            text="Hello bot",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            metadata={"username": "testuser"}
        )
        
        assert message.text == "Hello bot"
        assert message.user_id == "U123"
        assert message.channel_id == "C456"
        assert message.thread_id == "T789"
        assert message.metadata["username"] == "testuser"
    
    def test_message_optional_fields(self):
        """Test Message with optional fields"""
        message = Message(
            text="Test",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )
        
        assert message.text == "Test"
        assert message.user_id == "U123"
        assert message.channel_id == "C456"
        assert message.thread_id == "T789"
        assert message.attachments == []  # Initialized as empty list
        assert message.metadata == {}  # Initialized as empty dict
    
    def test_message_with_empty_metadata(self):
        """Test Message with empty metadata dict"""
        message = Message(
            text="Test",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            metadata={}
        )
        
        assert message.metadata == {}


class TestResponse:
    """Test Response dataclass"""
    
    def test_response_creation(self):
        """Test creating a Response instance"""
        response = Response(
            type="text",
            content="Hello user",
            metadata={"message_id": "msg_123"}
        )
        
        assert response.type == "text"
        assert response.content == "Hello user"
        assert response.metadata["message_id"] == "msg_123"
    
    def test_response_types(self):
        """Test different response types"""
        # Text response
        text_response = Response(type="text", content="Hello")
        assert text_response.type == "text"
        
        # Image response
        image_response = Response(type="image", content="base64_data")
        assert image_response.type == "image"
        
        # Error response
        error_response = Response(type="error", content="Something went wrong")
        assert error_response.type == "error"
    
    def test_response_optional_metadata(self):
        """Test Response with optional metadata"""
        response = Response(
            type="text",
            content="Test"
        )
        
        assert response.type == "text"
        assert response.content == "Test"
        assert response.metadata == {}  # Initialized as empty dict


class TestBaseClient:
    """Test BaseClient abstract class"""
    
    class MockClient(BaseClient):
        """Concrete implementation for testing"""
        
        def __init__(self):
            super().__init__(name="MockClient")
            self.platform = "mock"
            self.posted_messages = []
            self.uploaded_images = []
        
        async def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None, **kwargs) -> Dict[str, Any]:
            """Mock send message"""
            self.posted_messages.append({
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts
            })
            return {"ts": f"msg_{len(self.posted_messages)}"}
        
        async def send_image(self, channel: str, image_data: bytes, filename: str, thread_ts: Optional[str] = None, **kwargs) -> Dict[str, Any]:
            """Mock send image"""
            self.uploaded_images.append({
                "channel": channel,
                "filename": filename,
                "thread_ts": thread_ts
            })
            return {"file": {"url_private": f"https://mock.com/{filename}"}}
        
        async def get_thread_history(self, channel: str, thread_ts: str, **kwargs) -> List[Dict[str, Any]]:
            """Mock get thread history"""
            return [
                {"user": "U123", "text": "Previous message", "ts": "123.456"}
            ]
        
        async def send_thinking_indicator(self, channel: str, thread_ts: Optional[str] = None) -> Optional[str]:
            """Mock send thinking indicator"""
            return "thinking_123"
        
        async def delete_message(self, channel: str, ts: str) -> bool:
            """Mock delete message"""
            return True
        
        async def download_file(self, url: str) -> bytes:
            """Mock download file"""
            return b"fake_file_content"
        
        def format_text(self, text: str, platform_specific: bool = True) -> str:
            """Mock format text"""
            return text
        
        async def post_message(self, channel_id: str, text: str, thread_ts: Optional[str] = None) -> str:
            """Compatibility method"""
            result = await self.send_message(channel_id, text, thread_ts)
            return result["ts"]
        
        async def upload_image(self, channel_id: str, image_data: bytes, filename: str, thread_ts: Optional[str] = None) -> str:
            """Compatibility method"""
            result = await self.send_image(channel_id, image_data, filename, thread_ts)
            return result["file"]["url_private"]
        
        async def fetch_thread_history(self, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
            """Compatibility method"""
            return await self.get_thread_history(channel_id, thread_ts)
        
        async def handle_event(self, event: Dict[str, Any]) -> None:
            """Mock handle event"""
            pass
        
        async def start(self):
            """Mock start"""
            pass
        
        async def stop(self):
            """Mock stop"""
            pass
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client instance"""
        return self.MockClient()
    
    def test_base_client_initialization(self, mock_client):
        """Test base client initialization"""
        assert mock_client.platform == "mock"
        assert mock_client.posted_messages == []
        assert mock_client.uploaded_images == []
    
    @pytest.mark.asyncio
    async def test_post_message(self, mock_client):
        """Test posting a message"""
        message_id = await mock_client.post_message(
            channel_id="C123",
            text="Hello",
            thread_ts="456.789"
        )
        
        assert message_id == "msg_1"
        assert len(mock_client.posted_messages) == 1
        assert mock_client.posted_messages[0]["channel"] == "C123"
        assert mock_client.posted_messages[0]["text"] == "Hello"
        assert mock_client.posted_messages[0]["thread_ts"] == "456.789"
    
    @pytest.mark.asyncio
    async def test_upload_image(self, mock_client):
        """Test uploading an image"""
        image_url = await mock_client.upload_image(
            channel_id="C123",
            image_data=b"fake_image_data",
            filename="test.png",
            thread_ts="456.789"
        )
        
        assert image_url == "https://mock.com/test.png"
        assert len(mock_client.uploaded_images) == 1
        assert mock_client.uploaded_images[0]["channel"] == "C123"
        assert mock_client.uploaded_images[0]["filename"] == "test.png"
    
    @pytest.mark.asyncio
    async def test_fetch_thread_history(self, mock_client):
        """Test fetching thread history"""
        history = await mock_client.fetch_thread_history(
            channel_id="C123",
            thread_ts="456.789"
        )
        
        assert len(history) == 1
        assert history[0]["user"] == "U123"
        assert history[0]["text"] == "Previous message"
    
    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented"""
        # Try to create BaseClient directly (should fail)
        with pytest.raises(TypeError):
            BaseClient()
    
    @pytest.mark.critical
    def test_critical_interface_contract(self, mock_client):
        """Critical: Ensure BaseClient interface is maintained"""
        # All required methods must exist
        assert hasattr(mock_client, 'post_message')
        assert hasattr(mock_client, 'upload_image')
        assert hasattr(mock_client, 'fetch_thread_history')
        assert hasattr(mock_client, 'handle_event')
        assert hasattr(mock_client, 'platform')
        
        # Methods must be callable
        assert callable(mock_client.post_message)
        assert callable(mock_client.upload_image)
        assert callable(mock_client.fetch_thread_history)
        assert callable(mock_client.handle_event)


class TestBaseClientScenarios:
    """Scenario tests for BaseClient usage"""
    
    @pytest.mark.asyncio
    async def test_scenario_conversation_flow(self):
        """Scenario: Handle a conversation flow"""
        client = TestBaseClient.MockClient()
        
        # User sends message
        user_message = Message(
            text="Hello bot",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )
        
        # Bot responds
        response_id = await client.post_message(
            channel_id=user_message.channel_id,
            text="Hello! How can I help?",
            thread_ts=user_message.thread_id
        )
        
        assert response_id == "msg_1"
        
        # User continues conversation
        followup = Message(
            text="Tell me about Python",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )
        
        # Fetch history to maintain context
        history = await client.fetch_thread_history(
            channel_id=followup.channel_id,
            thread_ts=followup.thread_id
        )
        
        assert len(history) > 0
        
        # Bot responds with context
        response_id2 = await client.post_message(
            channel_id=followup.channel_id,
            text="Python is a programming language...",
            thread_ts=followup.thread_id
        )
        
        assert response_id2 == "msg_2"
        assert len(client.posted_messages) == 2
    
    @pytest.mark.asyncio
    async def test_scenario_image_handling(self):
        """Scenario: Handle image generation and upload"""
        client = TestBaseClient.MockClient()
        
        # Generate image (mocked)
        image_data = b"fake_image_bytes"
        
        # Upload image
        image_url = await client.upload_image(
            channel_id="C123",
            image_data=image_data,
            filename="generated_sunset.png",
            thread_ts="T456"
        )
        
        assert image_url == "https://mock.com/generated_sunset.png"
        
        # Post message with image reference
        await client.post_message(
            channel_id="C123",
            text=f"Here's your sunset image: {image_url}",
            thread_ts="T456"
        )
        
        assert len(client.uploaded_images) == 1
        assert len(client.posted_messages) == 1


class TestBaseClientContract:
    """Contract tests for BaseClient implementations"""
    
    def test_contract_platform_implementations(self):
        """Contract: Platform implementations must follow BaseClient interface"""
        # This test would verify that SlackClient and DiscordClient
        # properly implement the BaseClient interface
        
        # Import statements would be here if we were testing actual implementations
        # from slack_client import SlackClient
        # from discord_client import DiscordClient
        
        # For now, just test that our mock follows the contract
        client = TestBaseClient.MockClient()
        
        # Must have platform identifier
        assert hasattr(client, 'platform')
        assert isinstance(client.platform, str)
        
        # Must implement all abstract methods
        assert hasattr(client, 'post_message')
        assert hasattr(client, 'upload_image')
        assert hasattr(client, 'fetch_thread_history')
        assert hasattr(client, 'handle_event')
    
    @pytest.mark.smoke
    def test_smoke_basic_client_operations(self):
        """Smoke test: Basic client operations work"""
        try:
            client = TestBaseClient.MockClient()
            
            # Can create messages
            msg = Message("Test", "U1", "C1", "T1")
            assert msg is not None
            
            # Can create responses
            resp = Response("text", "Test response")
            assert resp is not None
            
            # Client has required attributes
            assert client.platform is not None
            
        except Exception as e:
            pytest.fail(f"Basic client operations failed: {e}")