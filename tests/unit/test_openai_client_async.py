"""
Unit tests for async OpenAI client
Tests async OpenAI API client wrapper for Responses API
"""
import pytest
import json
from unittest.mock import MagicMock, patch, Mock, AsyncMock, PropertyMock
from datetime import datetime
import base64
import asyncio
from openai_client import OpenAIClient, ImageData


class TestAsyncOpenAIClient:
    """Test async OpenAIClient class"""

    @pytest.fixture
    def mock_async_openai(self):
        """Mock AsyncOpenAI client"""
        with patch('openai_client.base.AsyncOpenAI') as mock:
            # Create a mock instance with async methods
            mock_instance = AsyncMock()
            mock_instance.close = AsyncMock()

            # Mock responses API
            mock_responses = AsyncMock()
            mock_instance.responses = mock_responses
            mock_instance.responses.create = AsyncMock()

            # Mock images API
            mock_images = AsyncMock()
            mock_instance.images = mock_images
            mock_instance.images.generate = AsyncMock()
            mock_instance.images.edit = AsyncMock()

            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_aiohttp(self):
        """Mock aiohttp session"""
        with patch('openai_client.base.aiohttp.ClientSession') as mock:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def client(self, mock_async_openai, mock_aiohttp):
        """Create OpenAIClient instance with mocked AsyncOpenAI"""
        return OpenAIClient()

    def test_initialization(self, mock_async_openai):
        """Test client initialization"""
        client = OpenAIClient()

        # Check that AsyncOpenAI client was initialized
        assert client.client is not None
        assert client.stream_timeout_seconds == 30.0  # From mock_env
        mock_async_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_text_response(self, client, mock_async_openai):
        """Test creating text response"""
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

        result = await client.create_text_response(
            messages=messages,
            model="gpt-5",
            temperature=0.7,
            max_tokens=4096
        )

        assert result == "Test response"
        client.client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_text_response_with_tools(self, client, mock_async_openai):
        """Test creating text response with tools"""
        # Setup mock response
        mock_content = MagicMock()
        mock_content.text = "Tool response"

        mock_item = MagicMock()
        mock_item.content = [mock_content]

        mock_response = MagicMock()
        mock_response.output = [mock_item]

        client.client.responses.create.return_value = mock_response

        messages = [{"role": "user", "content": "Use tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]

        result = await client.create_text_response_with_tools(
            messages=messages,
            tools=tools,
            model="gpt-5"
        )

        assert result == "Tool response"

    @pytest.mark.asyncio
    async def test_classify_intent(self, client, mock_async_openai):
        """Test intent classification"""
        # Setup mock response
        mock_content = MagicMock()
        mock_content.text = "none"

        mock_item = MagicMock()
        mock_item.content = [mock_content]

        mock_response = MagicMock()
        mock_response.output = [mock_item]

        client.client.responses.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello bot"}]

        result = await client.classify_intent(messages, "Hello bot")

        assert result == "text_only"  # "none" gets mapped to "text_only"

    @pytest.mark.asyncio
    async def test_generate_image(self, client, mock_async_openai):
        """Test image generation"""
        # Setup mock response for enhancement
        mock_enhance_content = MagicMock()
        mock_enhance_content.text = "Enhanced: A beautiful sunset"
        mock_enhance_item = MagicMock()
        mock_enhance_item.content = [mock_enhance_content]
        mock_enhance_response = MagicMock()
        mock_enhance_response.output = [mock_enhance_item]

        # Setup mock response for image generation
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"fake_image_data").decode()
        mock_gen_response = MagicMock()
        mock_gen_response.data = [mock_image]

        # First call is for enhancement, second for generation
        client.client.responses.create.return_value = mock_enhance_response
        client.client.images.generate.return_value = mock_gen_response

        result = await client.generate_image("A beautiful sunset")

        assert isinstance(result, ImageData)
        assert result.base64_data is not None
        client.client.images.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_images(self, client, mock_async_openai):
        """Test image analysis with streaming"""
        # Setup mock streaming response
        async def mock_stream():
            # Yield events like the real API
            mock_event = MagicMock()
            mock_event.type = "response.created"
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.output_text.delta"
            mock_event.delta = "This is "
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.output_text.delta"
            mock_event.delta = "an image"
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.done"
            yield mock_event

        # Mock prompt enhancement
        mock_enhance_content = MagicMock()
        mock_enhance_content.text = "Enhanced question"
        mock_enhance_item = MagicMock()
        mock_enhance_item.content = [mock_enhance_content]
        mock_enhance_response = MagicMock()
        mock_enhance_response.output = [mock_enhance_item]

        # First call is enhancement (non-streaming), second is analysis (streaming)
        client.client.responses.create.side_effect = [
            mock_enhance_response,  # Enhancement
            mock_stream()  # Analysis stream
        ]

        images = [base64.b64encode(b"fake_image_data").decode()]

        # Test with streaming callback
        chunks = []
        def callback(chunk):
            if chunk:
                chunks.append(chunk)

        result = await client.analyze_images(
            images,
            "What is this?",
            stream_callback=callback
        )

        assert result == "This is an image"
        assert chunks == ["This is ", "an image"]

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup"""
        await client.close()
        client.client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_api_call_timeout(self, client):
        """Test timeout handling in _safe_api_call"""
        async def slow_api_call():
            await asyncio.sleep(10)
            return "Should timeout"

        with pytest.raises(TimeoutError):
            await client._safe_api_call(
                slow_api_call,
                timeout_seconds=0.1,
                operation_type="test"
            )

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_critical_streaming_response(self, client, mock_async_openai):
        """Critical: Test streaming response handling"""
        # Setup mock streaming response
        async def mock_stream():
            mock_event = MagicMock()
            mock_event.type = "response.created"
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.output_text.delta"
            mock_event.delta = "Test "
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.output_text.delta"
            mock_event.delta = "streaming"
            yield mock_event

            mock_event = MagicMock()
            mock_event.type = "response.done"
            yield mock_event

        client.client.responses.create.return_value = mock_stream()

        chunks = []
        def callback(chunk):
            if chunk:
                chunks.append(chunk)

        result = await client.create_streaming_response(
            messages=[{"role": "user", "content": "Test"}],
            stream_callback=callback
        )

        assert result == "Test streaming"
        assert chunks == ["Test ", "streaming"]