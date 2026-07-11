"""Channel topic/purpose in default context.

Covers: the TTL-cached get_channel_context lookup (channels vs DMs, negative cache
on failure), the CHANNEL CONTEXT system-prompt section, and graceful absence
(no client support / DM / empty info → prompt unchanged).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slack_client.utilities import SlackUtilitiesMixin


@pytest.fixture
def message_processor():
    from message_processor import MessageProcessor
    with patch('message_processor.base.AsyncThreadStateManager', return_value=MagicMock()):
        with patch('message_processor.base.OpenAIClient', return_value=MagicMock()):
            return MessageProcessor()


class _Client(SlackUtilitiesMixin):
    def __init__(self, api_client):
        self.app = SimpleNamespace(client=api_client)

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _info_response(name="chatgpt-bot-test", topic="Github project for the bot: github.com/protella/chatgpt-bots",
                   purpose="Bot testing sandbox", is_im=False, num_members=12):
    channel = {
        "name": name,
        "topic": {"value": topic},
        "purpose": {"value": purpose},
        "is_im": is_im,
        "is_mpim": False,
    }
    if num_members is not None:
        channel["num_members"] = num_members
    return {"channel": channel}


# ---------------- get_channel_context ----------------

@pytest.mark.asyncio
async def test_get_channel_context_returns_metadata():
    api = SimpleNamespace(conversations_info=AsyncMock(return_value=_info_response()))
    ctx = await _Client(api).get_channel_context("C123")
    assert ctx == {
        "name": "chatgpt-bot-test",
        "topic": "Github project for the bot: github.com/protella/chatgpt-bots",
        "purpose": "Bot testing sandbox",
        "num_members": 12,
    }
    # F29: the count is requested from the API (needed for the people signal/suffix line)
    api.conversations_info.assert_awaited_once_with(channel="C123", include_num_members=True)


@pytest.mark.asyncio
async def test_get_channel_context_num_members_absent_is_none():
    # Defensive: an API payload without num_members surfaces None, never a KeyError.
    api = SimpleNamespace(conversations_info=AsyncMock(return_value=_info_response(num_members=None)))
    ctx = await _Client(api).get_channel_context("C123")
    assert ctx["num_members"] is None


@pytest.mark.asyncio
async def test_get_cached_channel_context_sync_peek():
    # The sync peek returns warmed cache data with no extra API call, and None before warmup.
    api = SimpleNamespace(conversations_info=AsyncMock(return_value=_info_response()))
    client = _Client(api)
    assert client.get_cached_channel_context("C123") is None  # cold
    await client.get_channel_context("C123")                  # warm it
    peeked = client.get_cached_channel_context("C123")
    assert peeked["num_members"] == 12
    assert api.conversations_info.await_count == 1             # peek made no call


@pytest.mark.asyncio
async def test_get_channel_context_caches_per_channel():
    api = SimpleNamespace(conversations_info=AsyncMock(return_value=_info_response()))
    client = _Client(api)
    await client.get_channel_context("C123")
    await client.get_channel_context("C123")
    assert api.conversations_info.await_count == 1  # second hit served from cache


@pytest.mark.asyncio
async def test_get_channel_context_dm_is_none():
    api = SimpleNamespace(conversations_info=AsyncMock(return_value=_info_response(is_im=True)))
    assert await _Client(api).get_channel_context("D123") is None


@pytest.mark.asyncio
async def test_get_channel_context_failure_is_none_and_negative_cached():
    api = SimpleNamespace(conversations_info=AsyncMock(side_effect=RuntimeError("boom")))
    client = _Client(api)
    assert await client.get_channel_context("C123") is None
    assert await client.get_channel_context("C123") is None
    assert api.conversations_info.await_count == 1  # negative cache absorbs the retry


@pytest.mark.asyncio
async def test_get_channel_context_unescapes_slack_entities():
    api = SimpleNamespace(conversations_info=AsyncMock(
        return_value=_info_response(topic="Q&amp;A: &lt;https://x.y&gt;", purpose="Dev &amp; Testing")))
    ctx = await _Client(api).get_channel_context("C123")
    assert ctx["topic"] == "Q&A: <https://x.y>"
    assert ctx["purpose"] == "Dev & Testing"


@pytest.mark.asyncio
async def test_get_channel_context_missing_channel_id():
    api = SimpleNamespace(conversations_info=AsyncMock())
    assert await _Client(api).get_channel_context(None) is None
    api.conversations_info.assert_not_called()


# ---------------- system prompt section ----------------

def _slack_client_stub():
    return SimpleNamespace(name="slack", tool_registry=None, user_cache={}, bot_user_id=None)


class TestChannelContextPromptSection:
    def test_section_rendered_with_topic_and_purpose(self, message_processor):
        prompt = message_processor._get_system_prompt(
            _slack_client_stub(),
            channel_info={"name": "eng-data", "topic": "Repo: github.com/x/y", "purpose": "Data eng chat"},
        )
        assert "--- CHANNEL CONTEXT ---" in prompt
        assert "#eng-data" in prompt
        assert "Channel topic: Repo: github.com/x/y" in prompt
        assert "Channel description: Data eng chat" in prompt

    def test_empty_topic_and_purpose_lines_omitted(self, message_processor):
        prompt = message_processor._get_system_prompt(
            _slack_client_stub(),
            channel_info={"name": "eng-data", "topic": "", "purpose": ""},
        )
        assert "#eng-data" in prompt
        assert "Channel topic:" not in prompt
        assert "Channel description:" not in prompt

    def test_no_channel_info_no_section(self, message_processor):
        prompt = message_processor._get_system_prompt(_slack_client_stub())
        assert "--- CHANNEL CONTEXT ---" not in prompt


# ---------------- _build_channel_info bridge ----------------

@pytest.mark.asyncio
async def test_build_channel_info_uses_client_lookup(message_processor):
    client = SimpleNamespace(get_channel_context=AsyncMock(return_value={"name": "x", "topic": "t", "purpose": ""}))
    assert await message_processor._build_channel_info(client, "C1") == {"name": "x", "topic": "t", "purpose": ""}


@pytest.mark.asyncio
async def test_build_channel_info_client_without_support(message_processor):
    assert await message_processor._build_channel_info(SimpleNamespace(), "C1") is None


@pytest.mark.asyncio
async def test_build_channel_info_swallows_errors(message_processor):
    client = SimpleNamespace(get_channel_context=AsyncMock(side_effect=RuntimeError("api down")))
    assert await message_processor._build_channel_info(client, "C1") is None
