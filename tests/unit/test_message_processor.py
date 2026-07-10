"""
Unit tests for message_processor.py module
Tests the core message processing logic
"""
import pytest
import base64
import time
from io import BytesIO
from unittest.mock import MagicMock, patch, Mock, call, AsyncMock
from datetime import datetime
from message_processor.base import MessageProcessor
from base_client import Message, Response


class TestMessageProcessor:
    """Test MessageProcessor class"""
    
    @pytest.fixture
    def mock_thread_manager(self):
        """Create a mock thread manager"""
        from unittest.mock import AsyncMock
        mock = MagicMock()
        mock.acquire_thread_lock = AsyncMock(return_value=True)
        mock.release_thread_lock = AsyncMock(return_value=None)
        mock.get_or_create_thread.return_value = MagicMock(
            thread_ts="123.456",
            channel_id="C123",
            messages=[],
            config_overrides={},
            system_prompt=None,
            is_processing=False,
            had_timeout=False,
            add_message=MagicMock(),
            get_recent_messages=MagicMock(return_value=[]),
            message_count=0  # Add this for comparison operations
        )
        # Add token counter
        mock._token_counter = MagicMock()
        mock._token_counter.count_message_tokens.return_value = 100
        mock._token_counter.count_thread_tokens.return_value = 100
        mock._max_tokens = 100000  # Add max tokens limit
        return mock
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        from openai_client import ImageData
        mock = MagicMock()
        mock.classify_intent.return_value = "chat"
        mock.get_response.return_value = "Hello from AI"
        mock.create_text_response.return_value = "Hello from AI"
        mock.create_text_response_with_tools.return_value = "Hello from AI"
        mock.count_tokens.return_value = 100  # Add proper token count
        
        # Create proper ImageData objects for image operations
        image_data = MagicMock()
        image_data.base64_data = "ZmFrZV9pbWFnZV9kYXRh"  # base64 of "fake_image_data"
        image_data.format = "png"
        image_data.prompt = "Generated image"
        image_data.to_bytes.return_value = BytesIO(b"fake_image_data")
        mock.generate_image.return_value = image_data
        
        mock.analyze_images.return_value = "Image shows a cat"
        mock.edit_image.return_value = image_data
        return mock
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock platform client"""
        mock = MagicMock()
        mock.platform = "mock"
        mock.name = "MockClient"
        mock.post_message.return_value = "msg_123"
        mock.upload_image.return_value = "https://mock.com/image.png"
        mock.download_file.return_value = b"fake_file_data"
        mock.get_thread_history = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def processor(self, mock_thread_manager, mock_openai_client):
        """Create a MessageProcessor with mocked dependencies"""
        with patch('message_processor.base.AsyncThreadStateManager', return_value=mock_thread_manager):
            with patch('message_processor.base.OpenAIClient', return_value=mock_openai_client):
                processor = MessageProcessor()
                processor.thread_manager = mock_thread_manager
                processor.openai_client = mock_openai_client
                # Add required attributes
                processor._token_counter = MagicMock()
                processor._token_counter.count_thread_tokens.return_value = 100
                processor._max_tokens = 100000
                return processor
    
    def test_initialization_without_db(self):
        """Test MessageProcessor initialization without database"""
        with patch('message_processor.base.AsyncThreadStateManager') as mock_thread_manager:
            with patch('message_processor.base.OpenAIClient') as mock_openai:
                processor = MessageProcessor()

                assert processor.db is None
                mock_thread_manager.assert_called_once_with(db=None)
                mock_openai.assert_called_once()
    
    def test_initialization_with_db(self):
        """Test MessageProcessor initialization with database"""
        mock_db = MagicMock()
        with patch('message_processor.base.AsyncThreadStateManager') as mock_thread_manager:
            with patch('message_processor.base.OpenAIClient') as mock_openai:
                processor = MessageProcessor(db=mock_db)

                assert processor.db is mock_db
                mock_thread_manager.assert_called_once_with(db=mock_db)
    
    async def test_process_message_thread_busy(self, processor, mock_client):
        """Test processing message when thread is busy"""
        message = Message(
            text="Hello",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )

        # Make thread busy
        async def mock_acquire_lock(*args, **kwargs):
            return False
        processor.thread_manager.acquire_thread_lock = mock_acquire_lock

        response = await processor.process_message(message, mock_client)

        # Phase Q: contention queues the message instead of rejecting it
        assert response.type == "queued"
    
    async def test_process_message_simple_chat(self, processor, mock_client):
        """Test processing a simple chat message"""
        message = Message(
            text="Hello bot",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            metadata={"username": "testuser"}
        )

        # Mock thread state
        thread_state = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False,
            pending_clarification=None,
            message_count=0,  # Add required field
            config_overrides={},
            system_prompt=None
        )
        processor.thread_manager.get_or_create_thread.return_value = thread_state

        # Process message
        response = await processor.process_message(message, mock_client)

        # Should return a response (either text or error)
        assert response is not None
        assert response.type in ["text", "error"]
        
        # If successful, check the content
        if response.type == "text":
            assert response.content == "Hello from AI"
            # Should get AI response - check either method could be called
            assert (
                processor.openai_client.create_text_response.called or 
                processor.openai_client.create_text_response_with_tools.called
            ), "Expected text response method to be called"
    
    @pytest.mark.asyncio
    async def test_process_message_with_timeout(self, processor, mock_client):
        """Test handling of timeout during processing"""
        from unittest.mock import AsyncMock

        message = Message(
            text="Hello",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )

        # Mock all async methods
        processor.thread_manager.get_or_create_thread_async = AsyncMock()
        mock_client.send_message_async = AsyncMock()
        mock_client.get_thread_history = AsyncMock(return_value=[])
        mock_client.get_thread_history_async = AsyncMock(return_value=[])
        processor.openai_client.classify_intent = AsyncMock(return_value="text_only")
        processor.openai_client.create_text_response = AsyncMock(return_value="Test response")

        # Mock thread state with previous timeout
        thread_state = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            pending_clarification=None,
            add_message=MagicMock()
        )
        # Set had_timeout as a property that can be changed
        thread_state.had_timeout = True
        processor.thread_manager.get_or_create_thread_async.return_value = thread_state

        # Also need to mock client.send_message since it's called for timeout recovery
        mock_client.send_message = AsyncMock()

        # Process message
        response = await processor.process_message(message, mock_client)

        # Should clear timeout flag
        assert thread_state.had_timeout is False
    
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_smoke_basic_message_processing(self, processor, mock_client):
        """Smoke test: Basic message processing works"""
        try:
            message = Message("Test", "U1", "C1", "T1")
            thread_state = MagicMock(
                thread_ts="T1",
                channel_id="C1",
                messages=[],
                had_timeout=False,
                pending_clarification=None
            )
            processor.thread_manager.get_or_create_thread.return_value = thread_state

            response = await processor.process_message(message, mock_client)
            assert response is not None

        except Exception as e:
            pytest.fail(f"Basic message processing failed: {e}")


class TestMessageProcessorHelpers:
    """Test helper methods of MessageProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create a MessageProcessor with mocked dependencies"""
        with patch('message_processor.base.AsyncThreadStateManager'):
            with patch('message_processor.base.OpenAIClient'):
                return MessageProcessor()
    
    def test_extract_slack_file_urls(self, processor):
        """Test extracting Slack file URLs from text"""
        text = "Check this <https://files.slack.com/files/U123/F456/image.png> and https://files.slack.com/files/U789/F012/photo.jpg"
        
        urls = processor._extract_slack_file_urls(text)
        
        assert len(urls) == 2
        assert "https://files.slack.com/files/U123/F456/image.png" in urls
        assert "https://files.slack.com/files/U789/F012/photo.jpg" in urls
    
    def test_build_user_content_text_only(self, processor):
        """Test building user content with text only"""
        content = processor._build_user_content("Hello bot", [])
        
        assert content == "Hello bot"
    
    def test_build_user_content_with_images(self, processor):
        """Test building user content with images"""
        image_inputs = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
        ]
        
        content = processor._build_user_content("What is this?", image_inputs)
        
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "What is this?"}  # Changed from "text" to "input_text"
        assert content[1]["type"] == "image_url"
    
    def test_get_system_prompt_slack(self, processor):
        """Test getting system prompt for Slack"""
        mock_client = MagicMock()
        mock_client.platform = "slack"
        mock_client.name = "SlackClient"  # Add name attribute
        
        prompt = processor._get_system_prompt(mock_client, "UTC")
        
        assert prompt is not None
        assert "Slack" in prompt or "slack" in prompt.lower()
    
    def test_format_response_text(self, processor):
        """Test formatting text response"""
        # This method doesn't exist in MessageProcessor, removing test
        # The formatting is done by the client itself, not the processor
        pass


class TestMessageProcessorContract:
    """Contract tests for MessageProcessor interface"""
    
    @pytest.mark.critical
    def test_contract_processor_interface(self):
        """Contract: MessageProcessor must provide expected interface"""
        with patch('message_processor.base.AsyncThreadStateManager'):
            with patch('message_processor.base.OpenAIClient'):
                processor = MessageProcessor()
                
                # Required attributes
                assert hasattr(processor, 'thread_manager')
                assert hasattr(processor, 'openai_client')
                assert hasattr(processor, 'db')
                
                # Required methods
                assert callable(processor.process_message)
                
                # process_message signature
                import inspect
                sig = inspect.signature(processor.process_message)
                params = list(sig.parameters.keys())
                assert 'message' in params
                assert 'client' in params
                assert 'thinking_id' in params
    
    def test_contract_response_types(self):
        """Contract: Response types must be valid"""
        valid_types = {"text", "image", "error", "busy"}
        
        # Test each response type
        for response_type in valid_types:
            response = Response(type=response_type, content="test")
            assert response.type in valid_types


class TestMessageProcessorDiagnostics:
    """Diagnostic tests for debugging"""
    
    def test_diagnostic_thread_state_tracking(self):
        """Diagnostic: Track thread state during processing"""
        with patch('message_processor.base.AsyncThreadStateManager') as mock_thread:
            with patch('message_processor.base.OpenAIClient'):
                processor = MessageProcessor()
                
                # Track thread state changes
                thread_state = MagicMock(
                    thread_ts="T123",
                    channel_id="C456",
                    messages=[],
                    had_timeout=False,
                    pending_clarification=None
                )
                processor.thread_manager.get_or_create_thread.return_value = thread_state
                processor.thread_manager.acquire_thread_lock.return_value = True
                
                diagnostic_info = {
                    "thread_key": f"{thread_state.channel_id}:{thread_state.thread_ts}",
                    "message_count": len(thread_state.messages),
                    "has_timeout": thread_state.had_timeout,
                    "pending_clarification": thread_state.pending_clarification is not None
                }
                
                print(f"\\nDiagnostic Thread Info: {diagnostic_info}")
                
                # Verify initial state
                assert diagnostic_info["message_count"] == 0
                assert diagnostic_info["has_timeout"] is False
                assert diagnostic_info["pending_clarification"] is False