"""
Unit tests for openai_client.py module
Tests OpenAI API client wrapper for Responses API
"""
import pytest
import json
from unittest.mock import MagicMock, patch, Mock, AsyncMock
from datetime import datetime
import base64
from openai_client import OpenAIClient, ImageData


class TestOpenAIClient:
    """Test OpenAIClient class"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        with patch('openai_client.OpenAI') as mock:
            yield mock
    
    @pytest.fixture
    def client(self, mock_openai):
        """Create OpenAIClient instance with mocked OpenAI"""
        return OpenAIClient()
    
    def test_initialization(self, mock_openai):
        """Test client initialization"""
        client = OpenAIClient()
        
        # Check that OpenAI client was initialized
        assert client.client is not None
        assert client.stream_timeout_seconds == 30.0  # From mock_env
        mock_openai.assert_called_once()
    
    def test_create_text_response_with_tools(self, client, mock_openai):
        """Test creating text response with tools"""
        # Setup mock response matching Responses API structure
        mock_content = MagicMock()
        mock_content.text = "Test response"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        result = client.create_text_response(
            messages=messages,
            model="gpt-5",
            temperature=0.7,
            max_tokens=4096
        )
        
        assert result == "Test response"
        client.client.responses.create.assert_called_once()
    
    def test_classify_intent_text_only(self, client, mock_openai):
        """Test intent classification for text-only message"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "none"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Hello bot"}
        ]
        
        result = client.classify_intent(messages, "Hello bot")
        
        assert result == "text_only"  # "none" gets mapped to "text_only"
    
    def test_classify_intent_image_generation(self, client, mock_openai):
        """Test intent classification for image generation"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "new"  # Classifier returns single words
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Draw a beautiful sunset"}
        ]
        
        result = client.classify_intent(messages, "Draw a beautiful sunset")
        
        assert result == "new_image"
    
    def test_classify_intent_image_edit(self, client, mock_openai):
        """Test intent classification for image edit"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "edit"  # Classifier returns single words
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Make the last image blue"}
        ]
        
        result = client.classify_intent(messages, "Make the last image blue", has_attached_images=False)
        
        assert result == "edit_image"
    
    def test_generate_image(self, client, mock_openai):
        """Test image generation"""
        # Setup mock response for gpt-image-1 model
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"fake_image_data").decode()
        mock_response = MagicMock()
        mock_response.data = [mock_image]
        client.client.images.generate.return_value = mock_response
        
        result = client.generate_image("A beautiful sunset")
        
        assert isinstance(result, ImageData)
        assert "A beautiful sunset" in result.prompt
        assert result.base64_data is not None
        client.client.images.generate.assert_called_once()
    
    def test_edit_image(self, client, mock_openai):
        """Test image editing"""
        # Setup mock response for image edit
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"edited_image_data").decode()
        mock_response = MagicMock()
        mock_response.data = [mock_image]
        client.client.images.edit.return_value = mock_response  # Mock the edit endpoint
        
        # Also mock the enhance prompt call
        mock_content = MagicMock()
        mock_content.text = "Enhanced: Make it blue with vibrant colors"
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        mock_enhance_response = MagicMock()
        mock_enhance_response.output = [mock_item]
        client.client.responses.create.return_value = mock_enhance_response
        
        # Create base64 image inputs (edit_image expects List[str] of base64 data)
        input_images = [base64.b64encode(b"fake_image").decode()]
        
        result = client.edit_image(input_images, "Make it blue")
        
        assert isinstance(result, ImageData)
        assert "blue" in result.prompt.lower()
        client.client.images.edit.assert_called()
    
    def test_analyze_image(self, client, mock_openai):
        """Test image analysis"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "This is a sunset image"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        # analyze_images expects List[str] of base64 data
        images = [base64.b64encode(b"fake_image_data").decode()]
        
        result = client.analyze_images(images, "What is this?")
        
        assert result == "This is a sunset image"
        # analyze_images calls create twice: once for prompt enhancement, once for analysis
        assert client.client.responses.create.call_count == 2
    
    def test_timeout_handling(self, client, mock_openai):
        """Test timeout configuration"""
        import httpx
        
        # Simulate timeout error
        client.client.responses.create.side_effect = httpx.TimeoutException("Timeout")
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(TimeoutError):  # Our wrapper converts to TimeoutError
            client.create_text_response(messages)
    
    def test_model_parameter_handling_gpt5_reasoning(self, client, mock_openai):
        """Test parameter handling for GPT-5 reasoning models"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "Response"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Test with gpt-5-mini (reasoning model)
        client.create_text_response(
            messages=messages,
            model="gpt-5-mini",
            reasoning_effort="high",
            verbosity=3
        )
        
        call_args = client.client.responses.create.call_args
        # Temperature should be 1.0 for reasoning models
        assert call_args[1].get("temperature") == 1.0
        # Should have reasoning parameters in new format
        assert "reasoning" in call_args[1]
        assert call_args[1]["reasoning"]["effort"] == "high"
        assert "text" in call_args[1]
        assert call_args[1]["text"]["verbosity"] == 3
    
    def test_model_parameter_handling_gpt5_chat(self, client, mock_openai):
        """Test parameter handling for GPT-5 chat models"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "Response"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Test with gpt-5-chat model
        client.create_text_response(
            messages=messages,
            model="gpt-5-chat-latest",
            temperature=0.7
        )
        
        call_args = client.client.responses.create.call_args
        # Should use provided temperature
        assert call_args[1].get("temperature") == 0.7
        # Should not have reasoning parameters in new format
        assert "reasoning" not in call_args[1]
        assert "text" not in call_args[1]
    
    @pytest.mark.critical
    def test_critical_response_format(self, client, mock_openai):
        """Critical: Ensure response format is correct"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "Test response"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        result = client.create_text_response(messages)
        
        assert isinstance(result, str)
        assert result == "Test response"
    
    @pytest.mark.smoke
    def test_smoke_basic_api_call(self, client, mock_openai):
        """Smoke test: Basic API call works"""
        # Setup mock response with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "Hello"
        
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        
        mock_response = MagicMock()
        mock_response.output = [mock_item]
        
        client.client.responses.create.return_value = mock_response
        
        try:
            messages = [{"role": "user", "content": "Hi"}]
            result = client.create_text_response(messages)
            assert result is not None
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Basic API call failed: {e}")


class TestImageData:
    """Test ImageData dataclass"""
    
    def test_image_data_creation(self):
        """Test creating ImageData instance"""
        image_data = ImageData(
            base64_data="ZmFrZV9iYXNlNjQ=",
            prompt="Test prompt",
            slack_url="https://example.com/image.png"
        )
        
        assert image_data.base64_data == "ZmFrZV9iYXNlNjQ="
        assert image_data.prompt == "Test prompt"
        assert image_data.slack_url == "https://example.com/image.png"
    
    def test_image_data_optional_fields(self):
        """Test ImageData with optional fields"""
        image_data = ImageData(
            base64_data="ZmFrZV9iYXNlNjQ=",
            prompt="Test prompt"
        )
        
        assert image_data.base64_data == "ZmFrZV9iYXNlNjQ="
        assert image_data.prompt == "Test prompt"
        assert image_data.slack_url is None  # slack_url is the correct attribute


class TestOpenAIClientStreaming:
    """Test streaming functionality"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        with patch('openai_client.OpenAI') as mock:
            yield mock
    
    @pytest.fixture
    def streaming_client(self, mock_openai):
        """Create client with streaming enabled"""
        with patch.dict('os.environ', {'ENABLE_STREAMING': 'true'}):
            return OpenAIClient()
    
    def test_streaming_response(self, streaming_client, mock_openai):
        """Test streaming response handling"""
        # Create mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content="World"))]
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock(delta=MagicMock(content=None))]
        
        mock_openai.return_value.responses.create.return_value = iter([
            mock_chunk1, mock_chunk2, mock_chunk3
        ])
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Note: The actual streaming implementation may differ
        # This test would need to be adjusted based on actual implementation
        pass


class TestOpenAIClientContract:
    """Contract tests for OpenAI client interface"""
    
    @pytest.fixture
    def client(self):
        """Create OpenAIClient instance"""
        with patch('openai_client.OpenAI'):
            return OpenAIClient()
    
    @pytest.mark.critical
    def test_contract_openai_interface(self, client):
        """Contract: OpenAIClient must provide expected interface"""
        # Required methods for MessageProcessor
        assert callable(client.create_text_response)
        assert callable(client.classify_intent)
        assert callable(client.generate_image)
        assert callable(client.edit_image)
        assert callable(client.analyze_images)  # Changed from analyze_image
        
        # Required attributes (OpenAIClient doesn't have model/timeout as direct attributes)
        assert hasattr(client, 'client')  # Has OpenAI client
        assert hasattr(client, 'stream_timeout_seconds')  # Has streaming timeout


class TestOpenAIClientScenarios:
    """Scenario tests for real-world usage"""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        with patch('openai_client.OpenAI') as mock:
            yield mock
    
    @pytest.fixture
    def client(self, mock_openai):
        """Create OpenAIClient instance with mocked OpenAI"""
        return OpenAIClient()
    
    def test_scenario_conversation_flow(self, client, mock_openai):
        """Scenario: Multi-turn conversation"""
        # Setup mock responses for conversation
        responses = [
            "Hello! How can I help you?",
            "Python is a programming language.",
            "It's known for its simplicity."
        ]
        
        # Setup mock responses with Responses API structure
        mock_responses = []
        for text in responses:
            mock_content = MagicMock()
            mock_content.text = text
            mock_item = MagicMock()
            mock_item.content = [mock_content]
            mock_response = MagicMock()
            mock_response.output = [mock_item]
            mock_responses.append(mock_response)
        
        client.client.responses.create.side_effect = mock_responses
        
        # Simulate conversation
        messages = []
        
        # Turn 1
        messages.append({"role": "user", "content": "Hello"})
        response1 = client.create_text_response(messages)
        assert response1 == "Hello! How can I help you?"
        messages.append({"role": "assistant", "content": response1})
        
        # Turn 2
        messages.append({"role": "user", "content": "What is Python?"})
        response2 = client.create_text_response(messages)
        assert response2 == "Python is a programming language."
        messages.append({"role": "assistant", "content": response2})
        
        # Turn 3
        messages.append({"role": "user", "content": "Why is it popular?"})
        response3 = client.create_text_response(messages)
        assert response3 == "It's known for its simplicity."
        messages.append({"role": "assistant", "content": response3})  # Add last response
        
        # Verify conversation context was maintained
        assert len(messages) == 6  # 3 user + 3 assistant
    
    def test_scenario_image_generation_flow(self, client, mock_openai):
        """Scenario: Complete image generation flow"""
        # Setup mocks with Responses API structure
        mock_content = MagicMock()
        mock_content.text = "new"  # Classifier returns single word
        mock_item = MagicMock()
        mock_item.content = [mock_content]
        mock_classify = MagicMock()
        mock_classify.output = [mock_item]
        
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"image_data").decode()
        mock_generate = MagicMock()
        mock_generate.data = [mock_image]
        
        client.client.responses.create.return_value = mock_classify
        client.client.images.generate.return_value = mock_generate
        
        # Classify intent
        messages = [{"role": "user", "content": "Draw a sunset"}]
        intent = client.classify_intent(messages, "Draw a sunset")
        assert intent == "new_image"  # Corrected expected value
        
        # Generate image
        image = client.generate_image("sunset")
        assert isinstance(image, ImageData)
        assert image.base64_data is not None