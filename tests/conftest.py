"""
Pytest configuration and shared fixtures
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ['TESTING'] = 'true'

# Redirect ALL database/backup writes to a throwaway dir before config.py is
# imported (its singleton reads DATABASE_DIR at import time). Without this,
# any DatabaseManager built in a test writes tagged migration backups into the
# real data/backups/ — and backup retention cleanup could then delete real
# backups older than 7 days.
import tempfile  # noqa: E402
os.environ['DATABASE_DIR'] = tempfile.mkdtemp(prefix='pytest-bot-db-')

# DISARM THE DEV TURN BARRIERS FOR THE WHOLE SUITE, before config.py's load_dotenv() can hand
# them to us. An operator arming these in .env for a live battery arms them for EVERY process
# that imports config — pytest included — and then any test that drives a channel turn blocks at
# a real seam for DEV_TURN_BARRIERS_TIMEOUT (120s by default). The suite does not fail, it HANGS,
# which reads like an infrastructure problem rather than a stray env var. Measured: a full run
# stalled indefinitely at ~14s of CPU until these were cleared.
#
# A test that wants a barrier sets it explicitly with monkeypatch (see test_dev_barriers.py),
# so clearing the inherited value costs nothing and cannot mask a real one.
#
# SET TO EMPTY, NEVER pop(). `load_dotenv()` does not override a variable that is already
# present, but it will happily supply one that is ABSENT — so deleting these here just hands
# config.py's load_dotenv the .env value a moment later, and the seams end up armed anyway.
# An empty string is present-and-falsy, which is what `dev_barriers._enabled` treats as off.
for _dev_barrier_var in ('DEV_TURN_BARRIERS', 'DEV_TURN_BARRIERS_DIR', 'DEV_TURN_BARRIERS_TIMEOUT'):
    os.environ[_dev_barrier_var] = ''

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing"""
    test_env = {
        'TESTING': 'true',
        'SLACK_BOT_TOKEN': 'xoxb-test-token',
        'SLACK_APP_TOKEN': 'xapp-test-token',
        'OPENAI_KEY': 'sk-test-key',
        # GPT_MODEL and DEFAULT_VERBOSITY must be REAL values: config.validate() refuses to boot
        # on either one outside its allowlist, because the channel capability resolver falls back
        # to them and an unusable fallback would turn one bad channel row into a 400.
        'GPT_MODEL': 'gpt-5.6-sol',
        'DEFAULT_REASONING_EFFORT': 'medium',
        'DEFAULT_VERBOSITY': 'medium',
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
        'LOG_LEVEL': 'DEBUG',
        # Image generation settings
        'DEFAULT_IMAGE_QUALITY': 'auto',
        'DEFAULT_IMAGE_BACKGROUND': 'auto',
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

# The channel-post admission gate is process-wide, like the receipt service, and shutdown closes
# it permanently by design. Any test that drives ChatBotV2.shutdown() would otherwise leave it
# shut for every test after it, silently refusing their posts.
@pytest.fixture(autouse=True)
def _open_channel_post_gate():
    from message_processor import outbound_receipts

    outbound_receipts.reset_channel_post_gate()
    yield
    outbound_receipts.reset_channel_post_gate()


# The document extraction cache is a process-wide singleton keyed by file id alone, so one test
# file's fixture ids collide with another's across the whole session — and the collision is
# INVISIBLE: the later test gets a plausible string back and fails on its content, which reads as a
# bug in the code under test rather than in the run order. (Seen exactly that way: a DM attachment
# test warmed "F1" and a channel read_document test then asserted against the wrong document.)
# The participation ledger is a process-wide sink built from config.log_directory the first time
# anything records an event — so without this the suite writes its synthetic rows into the REAL
# ledger (301 of them, against channel "C123", seen after the P2 battery), where they are
# indistinguishable from production decisions in every later analysis.
#
# ORDER IS THE WHOLE FIXTURE. The open sink holds a file handle from the OLD directory, so it is
# drained and closed BEFORE the swap and again after it: closing after the restore would flush this
# test's rows into whatever directory came next, which is the leak in miniature.
@pytest.fixture(autouse=True)
def _isolated_participation_ledger(tmp_path, monkeypatch):
    from config import config
    from message_processor import participation_telemetry

    participation_telemetry.shutdown()
    monkeypatch.setattr(config, "log_directory", str(tmp_path / "ledger"), raising=False)
    yield
    participation_telemetry.shutdown()


@pytest.fixture(autouse=True)
def _empty_document_extraction_cache():
    from message_processor.document_tools import _extraction_cache

    _extraction_cache.clear()
    yield
    _extraction_cache.clear()
