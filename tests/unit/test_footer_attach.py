"""Attached settings chrome — the "⚙️ <model>" row rides the response message itself.

Covers: attachable_footer_blocks surface routing (channel → channel settings, DM →
user settings, disabled → None), the coordinator passing blocks to the LAST part's
stopStream only, maybe_post_response_footer standing down when the chrome was
attached, and the rebuild filter keeping content-bearing messages that carry the
chrome (only pure-chrome messages are skipped).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from slack_client.event_handlers import feedback
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import SlackMessagingMixin, _is_ui_helper_message
from slack_client.utilities import SlackUtilitiesMixin
from streaming.native_sink import NativeStreamCoordinator


class _Host(SlackMessagingMixin):
    def __init__(self):
        self.app = SimpleNamespace(client=SimpleNamespace(chat_postMessage=AsyncMock()))

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


# ---------------- attachable_footer_blocks: surface routing ----------------

def test_channel_gets_channel_settings_button(monkeypatch):
    monkeypatch.setattr(config, "enable_response_footer", True)
    blocks = _Host().attachable_footer_blocks("C1", "gpt-5.6-sol")
    assert blocks[0]["elements"][0]["action_id"] == "open_channel_settings"
    assert blocks[0]["elements"][0]["text"]["text"] == "⚙️ gpt-5.6-sol"


def test_dm_gets_user_settings_button(monkeypatch):
    # DMs have no channel settings — the same chrome routes to the personal modal.
    monkeypatch.setattr(config, "enable_response_footer", True)
    blocks = _Host().attachable_footer_blocks("D1", "gpt-5.6-sol")
    assert blocks[0]["elements"][0]["action_id"] == feedback.USER_SETTINGS_ACTION_ID
    assert blocks[0]["elements"][0]["text"]["text"] == "⚙️ gpt-5.6-sol"


def test_disabled_or_missing_channel_returns_none(monkeypatch):
    monkeypatch.setattr(config, "enable_response_footer", False)
    assert _Host().attachable_footer_blocks("C1", "m") is None
    monkeypatch.setattr(config, "enable_response_footer", True)
    assert _Host().attachable_footer_blocks(None, "m") is None


def test_model_falls_back_to_config_default(monkeypatch):
    monkeypatch.setattr(config, "enable_response_footer", True)
    blocks = _Host().attachable_footer_blocks("C1", None)
    assert blocks[0]["elements"][0]["text"]["text"] == f"⚙️ {config.gpt_model}"


# ---------------- coordinator: blocks ride the LAST part only ----------------

def _native_client(parts_ts):
    """Slack client stub whose startStream hands out successive ts values."""
    it = iter(parts_ts)
    return SimpleNamespace(
        chat_startStream=AsyncMock(side_effect=lambda **k: {"ts": next(it)}),
        chat_appendStream=AsyncMock(),
        chat_stopStream=AsyncMock(),
    )


class _CoordClient:
    """Carrier exposing begin_native_stream over a stubbed Slack client."""
    def __init__(self, client):
        self.app = SimpleNamespace(client=client)
        self._client = client

    def begin_native_stream(self, channel_id, thread_id):
        from slack_client.messaging import NativeStreamSession
        return NativeStreamSession(self._client, channel_id, thread_id, team_id="TEAM1")


@pytest.mark.asyncio
async def test_finalize_attaches_blocks_on_stop():
    client = _native_client(["1.1"])
    coord = NativeStreamCoordinator(_CoordClient(client), "C1", "T1", char_limit=10_000)
    assert await coord.start()
    chrome = [{"type": "actions", "elements": []}]
    assert await coord.finalize("short answer", blocks=chrome)
    stop_kwargs = client.chat_stopStream.await_args.kwargs
    assert stop_kwargs["blocks"] is chrome


@pytest.mark.asyncio
async def test_multipart_finalize_puts_blocks_on_final_part_only():
    client = _native_client(["1.1", "2.2"])
    coord = NativeStreamCoordinator(_CoordClient(client), "C1", "T1", char_limit=300)
    assert await coord.start()
    chrome = [{"type": "actions", "elements": []}]
    long_text = ("word " * 120).strip()  # ~600 chars → one roll
    assert await coord.finalize(long_text, blocks=chrome)
    stops = client.chat_stopStream.await_args_list
    assert len(stops) == 2
    assert "blocks" not in stops[0].kwargs          # rolled part: no chrome
    assert stops[1].kwargs["blocks"] is chrome      # final part: chrome


@pytest.mark.asyncio
async def test_finalize_without_blocks_sends_none_to_stop():
    client = _native_client(["1.1"])
    coord = NativeStreamCoordinator(_CoordClient(client), "C1", "T1", char_limit=10_000)
    assert await coord.start()
    assert await coord.finalize("short")
    assert "blocks" not in client.chat_stopStream.await_args.kwargs


# ---------------- separate footer stands down when attached ----------------

@pytest.mark.asyncio
async def test_separate_footer_skipped_when_attached(monkeypatch):
    monkeypatch.setattr(config, "enable_response_footer", True)
    host = _Host()
    msg = SimpleNamespace(channel_id="C1", thread_id="T1")
    resp = SimpleNamespace(type="text", content="hello",
                           metadata={"model": "gpt-5.6-sol", "footer_attached": True})
    await host.maybe_post_response_footer(msg, resp)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_separate_footer_still_posts_when_not_attached(monkeypatch):
    # Fallback contract: legacy/non-streamed channel paths keep the trailing footer.
    monkeypatch.setattr(config, "enable_response_footer", True)
    host = _Host()
    msg = SimpleNamespace(channel_id="C1", thread_id="T1")
    resp = SimpleNamespace(type="text", content="hello",
                           metadata={"model": "gpt-5.6-sol"})
    await host.maybe_post_response_footer(msg, resp)
    host.app.client.chat_postMessage.assert_awaited_once()


# ---------------- rebuild filter: pure chrome vs content + chrome ----------------

def _chrome_block(action_id="open_channel_settings"):
    return {"type": "actions", "elements": [
        {"type": "button", "text": {"type": "plain_text", "text": "⚙️ gpt-5.6-sol"},
         "action_id": action_id}]}


def test_pure_chrome_messages_still_skipped():
    # Separate footer posts: text is the model-label notification fallback.
    assert _is_ui_helper_message({"text": "gpt-5.6-sol", "blocks": [_chrome_block()]}) is True
    assert _is_ui_helper_message({"text": "Rate this response",
                                  "blocks": feedback.build_feedback_blocks("gpt-5.6-sol")}) is True
    assert _is_ui_helper_message({"text": "", "blocks": [_chrome_block()]}) is True


def test_content_with_attached_chrome_is_kept():
    msg = {"text": "The Q3 menu data shows a 12% rise in birria mentions.",
           "blocks": [_chrome_block()]}
    assert _is_ui_helper_message(msg) is False
    dm = {"text": "Here's your answer.", "blocks": [_chrome_block(feedback.USER_SETTINGS_ACTION_ID)]}
    assert _is_ui_helper_message(dm) is False


class _RebuildBot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Real get_thread_history against a mocked Slack client (same pattern as
    test_sender_classification)."""
    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k): pass
    log_debug = log_error = log_warning = log_info


@pytest.mark.asyncio
async def test_rebuild_keeps_response_carrying_attached_chrome():
    """REGRESSION: a real answer whose final part carries the attached Configure
    chrome must survive the history rebuild; the old filter dropped ANY message
    with a helper action_id and would have erased the bot's own answers."""
    b = _RebuildBot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "what's trending?"},
        {"ts": "2", "bot_id": "B07SELF", "user": "U07SELF",
         "text": "Birria is up 12% on menus this quarter.",
         "blocks": [_chrome_block()]},                      # answer + attached chrome
        {"ts": "3", "bot_id": "B07SELF", "user": "U07SELF",
         "text": "gpt-5.6-sol", "blocks": [_chrome_block()]},  # old separate footer
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )
    result = await b.get_thread_history("C1", "1")
    texts = [m.text for m in result]
    assert "Birria is up 12% on menus this quarter." in texts  # kept
    assert "gpt-5.6-sol" not in texts                          # pure chrome still skipped
