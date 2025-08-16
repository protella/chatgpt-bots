"""
Pytest configuration and shared fixtures
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ['TESTING'] = 'true'

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing"""
    test_env = {
        'TESTING': 'true',
        'SLACK_BOT_TOKEN': 'xoxb-test-token',
        'SLACK_APP_TOKEN': 'xapp-test-token',
        'DISCORD_TOKEN': 'discord-test-token',
        'OPENAI_KEY': 'sk-test-key',
        'GPT_MODEL': 'gpt-5-chat-latest',
        'DEFAULT_REASONING_EFFORT': 'medium',
        'DEFAULT_VERBOSITY': '2',
        'UTILITY_REASONING_EFFORT': 'low',
        'UTILITY_VERBOSITY': '1',
        'ANALYSIS_REASONING_EFFORT': 'high',
        'ANALYSIS_VERBOSITY': '3',
        'DEFAULT_MAX_TOKENS': '4096',
        'DEFAULT_TEMPERATURE': '0.7',
        'STREAMING_UPDATE_INTERVAL': '2.0',
        'STREAMING_CIRCUIT_BREAKER_THRESHOLD': '5',
        'API_TIMEOUT_READ': '180',
        'API_TIMEOUT_STREAMING_CHUNK': '30',
        'LOG_LEVEL': 'DEBUG'
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env

@pytest.fixture
def mock_slack_client():
    """Mock Slack client"""
    client = MagicMock()
    client.conversations_history.return_value = {
        'ok': True,
        'messages': []
    }
    client.chat_postMessage.return_value = {
        'ok': True,
        'ts': '1234567890.123456'
    }
    return client

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = MagicMock()
    response = MagicMock()
    response.id = 'test-response-id'
    response.created = 1234567890
    response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    client.responses.create.return_value = response
    return client

@pytest.fixture
def sample_slack_message():
    """Sample Slack message for testing"""
    return {
        'type': 'message',
        'user': 'U123456',
        'text': 'Hello bot',
        'ts': '1234567890.123456',
        'channel': 'C123456',
        'thread_ts': None
    }

@pytest.fixture
def sample_thread_messages():
    """Sample thread messages for testing"""
    return [
        {
            'role': 'user',
            'content': 'Hello',
            'timestamp': '1234567890.123456'
        },
        {
            'role': 'assistant',
            'content': 'Hi there!',
            'timestamp': '1234567890.123457'
        }
    ]

@pytest.fixture
async def async_mock():
    """Helper for async mocking"""
    def _async_mock(return_value=None):
        future = asyncio.Future()
        future.set_result(return_value)
        return future
    return _async_mock

# Automatically use event loop for async tests
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()