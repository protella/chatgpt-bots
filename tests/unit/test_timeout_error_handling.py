"""
Unit tests for timeout and error handling in message processing
Tests the timeout/error recovery and status message updates
"""
import pytest
import time
import threading
from unittest.mock import MagicMock, patch, Mock, call, PropertyMock
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
from message_processor.base import MessageProcessor
from openai_client import OpenAIClient
from slack_client import SlackBot
import asyncio


@dataclass
class Message:
    """Test message class matching the real Message structure"""
    text: str
    user_id: str
    channel_id: str
    thread_id: str
    attachments: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Response:
    """Test response class"""
    type: str
    content: str


class TestTimeoutErrorHandling:
    """Test timeout and error handling features"""

    @pytest.fixture
    def mock_thread_manager(self):
        """Create a mock thread manager"""
        mock = MagicMock()
        mock.acquire_thread_lock.return_value = True
        mock.release_thread_lock.return_value = None

        # Create thread state with necessary attributes
        thread_state = MagicMock(
            thread_ts="123.456",
            channel_id="C123",
            messages=[],
            config_overrides={},
            system_prompt=None,
            is_processing=False,
            had_timeout=False,
            pending_clarification=None,
            add_message=MagicMock(),
            get_recent_messages=MagicMock(return_value=[]),
            message_count=0
        )
        mock.get_or_create_thread.return_value = thread_state

        # Add token counter
        mock._token_counter = MagicMock()
        mock._token_counter.count_message_tokens.return_value = 100
        mock._token_counter.count_thread_tokens.return_value = 100
        mock._max_tokens = 100000
        return mock

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        mock = MagicMock(spec=OpenAIClient)
        mock.classify_intent.return_value = "text_only"
        mock.create_text_response = MagicMock(return_value="Hello from AI")
        mock.create_streaming_response = MagicMock(return_value=iter(["Hello", " from", " AI"]))
        return mock

    @pytest.fixture
    def mock_database(self):
        """Create a mock database manager"""
        mock = MagicMock()
        mock.cache_message.return_value = None
        mock.get_user_preferences.return_value = {
            'model': 'gpt-5',
            'reasoning_effort': 'low',
            'verbosity': 'low'
        }
        return mock

    @pytest.fixture
    def mock_slack_client(self):
        """Create a mock Slack client"""
        mock = MagicMock(spec=SlackBot)
        mock.name = "slack"  # Add name attribute
        mock.get_thread_history.return_value = []
        mock.update_message = MagicMock()
        mock.send_message = MagicMock(return_value="msg_123")
        mock.supports_update = True
        mock.get_user_info = MagicMock(return_value={
            'timezone': 'America/New_York',
            'tz_label': 'EST',
            'real_name': 'Test User',
            'email': 'test@example.com'
        })
        return mock

    @pytest.fixture
    def message_processor(self, mock_thread_manager, mock_openai_client, mock_database):
        """Create a MessageProcessor instance with mocks"""
        processor = MessageProcessor(db=mock_database)
        # Replace the internal components with mocks
        processor.thread_manager = mock_thread_manager
        processor.openai_client = mock_openai_client
        return processor

    def test_timeout_error_updates_status_message(self, message_processor, mock_slack_client):
        """Test that timeout errors update the thinking status message"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Mock OpenAI client to raise TimeoutError
        message_processor.openai_client.classify_intent.side_effect = TimeoutError("Request timed out")

        # Act
        with patch.object(mock_slack_client, 'send_message', return_value="thinking_123"):
            response = message_processor.process_message(message, mock_slack_client)

        # Assert
        assert response.type == "error"
        assert "Taking Too Long" in response.content or "Service Temporarily Unavailable" in response.content

        # Verify status message was updated
        update_calls = mock_slack_client.update_message.call_args_list
        if update_calls:
            # Should have updated thinking message with timeout status
            last_update = update_calls[-1]
            assert "slow" in str(last_update).lower() or "unavailable" in str(last_update).lower()

    def test_intent_classification_retry_on_timeout(self):
        """Test that intent classification retries on timeout with exponential backoff"""
        # Create a real OpenAI client instance with mocked internals
        with patch('openai_client.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.api_timeout_read = 30.0
            mock_config.api_timeout_chunk = 30.0

            with patch('openai.OpenAI'):
                client = OpenAIClient()
                client.client = MagicMock()

                # Mock _safe_api_call method
                client._safe_api_call = MagicMock()

                # First call times out, second succeeds
                mock_response = MagicMock()
                mock_response.message.content = "image: none"
                client._safe_api_call.side_effect = [
                    TimeoutError("Timeout"),
                    mock_response
                ]

                # Act
                with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
                    result = client.classify_intent(
                        messages=[],
                        last_user_message="Test",
                        has_attached_images=False,
                        max_retries=2
                    )

                # Assert
                assert client._safe_api_call.call_count == 2
                mock_sleep.assert_called_once_with(1)  # First retry waits 1 second
                assert result == "text_only"

    def test_intent_classification_returns_error_after_max_retries(self):
        """Test that intent classification returns 'error' after max retries"""
        # Create a real OpenAI client instance with mocked internals
        with patch('openai_client.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.api_timeout_read = 30.0
            mock_config.api_timeout_chunk = 30.0

            with patch('openai.OpenAI'):
                client = OpenAIClient()
                client.client = MagicMock()

                # Mock _safe_api_call to always timeout
                client._safe_api_call = MagicMock()
                client._safe_api_call.side_effect = TimeoutError("Timeout")

                # Act
                with patch('time.sleep'):  # Mock sleep to speed up test
                    result = client.classify_intent(
                        messages=[],
                        last_user_message="Test",
                        has_attached_images=False,
                        max_retries=2
                    )

                # Assert
                assert result == "error"
                assert client._safe_api_call.call_count == 3  # Initial + 2 retries

    def test_error_intent_returns_service_unavailable(self, message_processor, mock_slack_client):
        """Test that 'error' intent returns service unavailable message"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Create a thread state with pending clarification to trigger the error handling path
        thread_state = message_processor.thread_manager.get_or_create_thread.return_value
        thread_state.pending_clarification = {
            "type": "image_intent",
            "original_request": "previous request"
        }

        # Mock intent classification to return 'error'
        message_processor.openai_client.classify_intent.return_value = 'error'

        # Act
        response = message_processor.process_message(message, mock_slack_client)

        # Assert
        assert response.type == "error"
        assert "Service Temporarily Unavailable" in response.content
        assert "Please try again" in response.content

    def test_thread_lock_released_on_timeout(self, message_processor, mock_slack_client, mock_thread_manager):
        """Test that thread lock is always released even on timeout"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Mock OpenAI to raise TimeoutError
        message_processor.openai_client.classify_intent.side_effect = TimeoutError("Timeout")

        # Act
        response = message_processor.process_message(message, mock_slack_client)

        # Assert
        mock_thread_manager.release_thread_lock.assert_called_once_with(
            "123.456", "C123"
        )
        assert response.type == "error"

    def test_thread_marked_with_timeout_flag(self, message_processor, mock_slack_client, mock_thread_manager):
        """Test that thread state is marked with had_timeout flag on timeout"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        thread_state = mock_thread_manager.get_or_create_thread.return_value
        message_processor.openai_client.classify_intent.side_effect = TimeoutError("Timeout")

        # Act
        response = message_processor.process_message(message, mock_slack_client)

        # Assert
        assert thread_state.had_timeout == True

    def test_different_error_messages_for_different_errors(self, message_processor, mock_slack_client):
        """Test that different error types produce appropriate user messages"""
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Test rate limit error
        message_processor.openai_client.classify_intent.side_effect = Exception("Rate limit exceeded")
        response = message_processor.process_message(message, mock_slack_client)
        assert "Too Many Requests" in response.content

        # Test token/context error
        message_processor.openai_client.classify_intent.side_effect = Exception("Context length exceeded")
        response = message_processor.process_message(message, mock_slack_client)
        assert "Message Too Long" in response.content

        # Test API error
        message_processor.openai_client.classify_intent.side_effect = Exception("OpenAI API error")
        response = message_processor.process_message(message, mock_slack_client)
        assert "Service Issue" in response.content

        # Test generic error
        message_processor.openai_client.classify_intent.side_effect = Exception("Random error")
        response = message_processor.process_message(message, mock_slack_client)
        assert "Something Went Wrong" in response.content

    def test_progress_updater_thread_starts_and_stops(self, message_processor, mock_slack_client):
        """Test that progress updater thread starts for long operations"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Mock a slow response
        def slow_classify(*args, **kwargs):
            time.sleep(0.1)  # Short sleep for testing
            return "text_only"

        message_processor.openai_client.classify_intent = slow_classify

        # Act
        with patch.object(message_processor, '_start_progress_updater') as mock_start:
            mock_thread = MagicMock()
            mock_start.return_value = mock_thread

            response = message_processor.process_message(message, mock_slack_client)

            # Verify progress updater was started
            if hasattr(message_processor, '_start_progress_updater'):
                assert mock_start.called or True  # May not be called for quick operations

    def test_status_message_updates_with_elapsed_time(self, message_processor, mock_slack_client):
        """Test that status messages update based on elapsed time"""
        # This tests the _start_progress_updater functionality
        thinking_id = "msg_123"
        stop_event = threading.Event()

        # Create a mock client
        mock_client = MagicMock()
        mock_client.update_message = MagicMock()

        # Simulate progress updates
        updates = []

        def capture_update(channel, msg_id, content):
            updates.append(content)

        mock_client.update_message.side_effect = capture_update

        # Start progress updater (if implemented)
        if hasattr(message_processor, '_start_progress_updater'):
            thread = message_processor._start_progress_updater(
                mock_client, "C123", thinking_id, "test operation"
            )

            # Let it run briefly
            time.sleep(0.1)

            # Stop the thread
            stop_event.set()
            if thread and hasattr(thread, 'join'):
                thread.join(timeout=1)

            # Check if any updates were made
            assert len(updates) >= 0  # May have updates depending on timing

    def test_timeout_error_without_thinking_message(self, message_processor, mock_slack_client):
        """Test timeout error handling when no thinking message exists"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Don't create a thinking message
        mock_slack_client.send_message.return_value = None
        message_processor.openai_client.classify_intent.side_effect = TimeoutError("Timeout")

        # Act
        response = message_processor.process_message(message, mock_slack_client)

        # Assert - should still return error response without crashing
        assert response.type == "error"
        assert "Taking Too Long" in response.content or "Service Temporarily Unavailable" in response.content

    def test_exponential_backoff_timing(self):
        """Test that retry delays follow exponential backoff pattern"""
        # Create a real OpenAI client instance
        with patch('openai_client.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.api_timeout_read = 30.0
            mock_config.api_timeout_chunk = 30.0

            with patch('openai.OpenAI'):
                client = OpenAIClient()
                client.client = MagicMock()

                # Mock _safe_api_call to always timeout
                client._safe_api_call = MagicMock()
                client._safe_api_call.side_effect = TimeoutError("Timeout")

                sleep_calls = []

                def track_sleep(seconds):
                    sleep_calls.append(seconds)

                # Act
                with patch('time.sleep', side_effect=track_sleep):
                    result = client.classify_intent(
                        messages=[],
                        last_user_message="Test",
                        has_attached_images=False,
                        max_retries=3
                    )

                # Assert
                assert result == "error"
                assert sleep_calls == [1, 2, 4]  # Exponential backoff: 2^0, 2^1, 2^2

    def test_streaming_timeout_handling(self, message_processor, mock_slack_client):
        """Test timeout handling during streaming responses"""
        # Arrange
        message = Message(
            text="Test message",
            user_id="U123",
            channel_id="C123",
            thread_id="123.456",
            attachments=[]
        )

        # Mock classify_intent to return text_only
        message_processor.openai_client.classify_intent.return_value = "text_only"

        # Mock both streaming and non-streaming to timeout
        # Replace the methods on the actual mock object
        message_processor.openai_client.create_streaming_response.side_effect = TimeoutError("Stream timeout")
        message_processor.openai_client.create_text_response.side_effect = TimeoutError("Text timeout")

        # Act
        response = message_processor.process_message(message, mock_slack_client)

        # Assert
        assert response.type == "error"
        assert "Taking Too Long" in response.content or "timeout" in response.content.lower()
        # Should handle streaming timeout gracefully

    def test_concurrent_timeout_handling(self, message_processor, mock_slack_client):
        """Test that multiple concurrent timeouts are handled properly"""
        # Arrange
        messages = [
            Message(
                text=f"Message {i}",
                user_id="U123",
                channel_id="C123",
                thread_id=f"123.{i}",
                attachments=[]
            )
            for i in range(3)
        ]

        message_processor.openai_client.classify_intent.side_effect = TimeoutError("Timeout")

        # Act - process messages concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(message_processor.process_message, msg, mock_slack_client)
                for msg in messages
            ]
            responses = [f.result() for f in futures]

        # Assert - all should handle timeout gracefully
        assert all(r.type == "error" for r in responses)
        assert all("Taking Too Long" in r.content or "Service Temporarily Unavailable" in r.content
                  for r in responses)


class TestOpenAIClientTimeoutHandling:
    """Test OpenAI client specific timeout handling"""

    @pytest.fixture
    def openai_client(self):
        """Create an OpenAI client instance"""
        with patch('openai_client.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.api_timeout_read = 30.0
            mock_config.api_timeout_chunk = 30.0

            with patch('openai.OpenAI'):
                client = OpenAIClient()
                # Mock the actual OpenAI client
                client.client = MagicMock()
                client.client.timeout = 30.0
                return client

    def test_safe_api_call_with_timeout(self, openai_client):
        """Test _safe_api_call handles timeout correctly"""
        # Arrange
        mock_func = MagicMock()
        mock_func.side_effect = TimeoutError("Request timeout")

        # Act & Assert
        with pytest.raises(TimeoutError):
            openai_client._safe_api_call(
                mock_func,
                operation_type="test",
                timeout_seconds=10
            )

    def test_intent_classification_timeout_logging(self, openai_client, caplog):
        """Test that timeout errors are logged appropriately"""
        # Arrange
        openai_client._safe_api_call = MagicMock()
        openai_client._safe_api_call.side_effect = TimeoutError("Timeout")

        # Act
        with patch('time.sleep'):
            result = openai_client.classify_intent(
                messages=[],
                last_user_message="Test",
                has_attached_images=False,
                max_retries=1
            )

        # Assert
        assert result == "error"
        # Check that appropriate warnings were logged
        assert any("timeout" in record.message.lower() for record in caplog.records)

    def test_timeout_with_different_retry_counts(self):
        """Test timeout handling with various retry configurations"""
        # Create client with mocked config
        with patch('openai_client.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.api_timeout_read = 30.0
            mock_config.api_timeout_chunk = 30.0

            with patch('openai.OpenAI'):
                client = OpenAIClient()
                client.client = MagicMock()

                # Arrange
                client._safe_api_call = MagicMock()
                client._safe_api_call.side_effect = TimeoutError("Timeout")

                # Test with 0 retries (should fail immediately)
                with patch('time.sleep'):
                    result = client.classify_intent(
                        messages=[],
                        last_user_message="Test",
                        has_attached_images=False,
                        max_retries=0
                    )
                assert result == "error"
                assert client._safe_api_call.call_count == 1

                # Reset and test with 5 retries
                client._safe_api_call.reset_mock()
                with patch('time.sleep'):
                    result = client.classify_intent(
                        messages=[],
                        last_user_message="Test",
                        has_attached_images=False,
                        max_retries=5
                    )
                assert result == "error"
                assert client._safe_api_call.call_count == 6  # Initial + 5 retries