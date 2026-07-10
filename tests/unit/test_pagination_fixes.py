"""Pagination-review fixes (R1-R5, W1-W4): read-side fetch resilience and
write-side split/marker safety.

R1: 429s retried with Retry-After; terminal failure raises HistoryFetchError
    and the processor fails the turn loudly instead of answering with amnesia.
R2: split bot replies rebuild as ONE clean assistant turn (markers stripped).
R3: the placeholder skip-filter only drops OWN status messages.
R4: summary-tail rebuilds pass oldest= to conversations.replies.
R5: history tool reports has_more.
W1-W3: overflow completion posts its remainder; non-streaming splits are
    fence-aware, per-chunk isolated, and never cut inside a <entity>.
"""
import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from slack_sdk.errors import SlackApiError

from base_client import HistoryFetchError, Message
from message_markers import (
    CONTINUATION_HEAD,
    CONTINUATION_TRAILER,
    continuation_trailer,
    ends_with_continuation,
    entity_safe_cut,
    fence_safe_chunks,
    part_prefix,
    starts_as_continuation,
    strip_continuation_markers,
)
from slack_client.messaging import SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.utilities import SlackUtilitiesMixin
from message_processor.thread_management import ThreadManagementMixin


SELF_BOT_ID = "B07SELF"
SELF_USER_ID = "U07SELF"


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Minimal harness binding the real messaging mixins to a mocked client."""
    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.bot_id = SELF_BOT_ID
        self.bot_user_id = SELF_USER_ID
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def format_text(self, text): return text


def _slack_error(error="ratelimited", status=429, retry_after="2"):
    resp = MagicMock()
    resp.get = lambda key, default=None: {"error": error}.get(key, default)
    resp.status_code = status
    resp.headers = {"Retry-After": retry_after}
    return SlackApiError(message=error, response=resp)


# ---------------- R1: rate-limit retry + loud terminal failure ----------------

@pytest.mark.asyncio
async def test_ratelimit_retry_honors_retry_after_and_succeeds():
    b = _Bot()
    ok = {"messages": [], "response_metadata": {}}
    b.app.client.conversations_replies = AsyncMock(
        side_effect=[_slack_error(), _slack_error(), ok]
    )
    with patch("slack_client.messaging.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = await b._replies_page_with_retry({"channel": "C1", "ts": "1", "limit": 1000})
    assert result is ok
    assert mock_sleep.await_count == 2
    assert mock_sleep.await_args_list[0].args[0] == 2.0  # Retry-After honored


@pytest.mark.asyncio
async def test_terminal_ratelimit_raises_history_fetch_error():
    b = _Bot()
    b.app.client.conversations_replies = AsyncMock(side_effect=_slack_error())
    with patch("slack_client.messaging.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(HistoryFetchError):
            await b.get_thread_history("C1", "1")


@pytest.mark.asyncio
async def test_non_ratelimit_api_error_raises_immediately():
    b = _Bot()
    b.app.client.conversations_replies = AsyncMock(
        side_effect=_slack_error(error="channel_not_found", status=404)
    )
    with patch("slack_client.messaging.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(HistoryFetchError):
            await b.get_thread_history("C1", "1")
    mock_sleep.assert_not_awaited()  # no pointless retries on hard errors


@pytest.mark.asyncio
async def test_empty_thread_still_returns_empty_list_not_error():
    b = _Bot()
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": [], "response_metadata": {}}
    )
    assert await b.get_thread_history("C1", "1") == []


def test_processor_fails_history_fetch_loudly():
    """The processor must catch HistoryFetchError distinctly and return a
    user-facing error — never proceed to answer without context."""
    import message_processor.base as mp_base
    src = inspect.getsource(mp_base)
    assert "except HistoryFetchError" in src
    assert "Couldn't Load Conversation History" in src


# ---------------- R2: continuation markers + rebuild merge ----------------

def test_strip_continuation_markers_all_shapes():
    assert strip_continuation_markers(
        f"{part_prefix(2)}hello{continuation_trailer()}") == "hello"
    assert strip_continuation_markers("*Part 1/3*\n\nhello") == "hello"
    assert strip_continuation_markers(
        "hello\n\n*...continued in next message...*") == "hello"  # legacy shape
    assert strip_continuation_markers(f"{CONTINUATION_HEAD}\n\nworld") == "world"
    assert strip_continuation_markers("no markers here") == "no markers here"


def test_marker_detection_helpers():
    assert ends_with_continuation(f"text{continuation_trailer()}")
    assert not ends_with_continuation("text")
    assert starts_as_continuation(part_prefix(3) + "text")
    assert starts_as_continuation(f"{CONTINUATION_HEAD}\n\ntext")
    assert not starts_as_continuation("Continued discussion from standup")


class _Rebuilder(ThreadManagementMixin):
    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_warning(self, *a, **k): pass


def _self_msg(ts, text, reactions=None, attachments=None):
    return Message(text=text, user_id=SELF_USER_ID, channel_id="C1", thread_id="1",
                   attachments=attachments or [],
                   metadata={"ts": ts, "is_bot": True, "sender_type": "self",
                             "reactions": reactions})


def _human_msg(ts, text):
    return Message(text=text, user_id="U07HUMAN", channel_id="C1", thread_id="1",
                   metadata={"ts": ts, "is_bot": False, "sender_type": "human"})


def test_three_part_split_reply_rebuilds_as_one_turn():
    r = _Rebuilder()
    history = [
        _human_msg("1", "explain the deploy process"),
        _self_msg("2", f"Step one...{continuation_trailer()}",
                  reactions=[{"name": "eyes", "count": 1}]),
        _self_msg("3", f"{part_prefix(2)}Step two...{continuation_trailer()}"),
        _self_msg("4", f"{part_prefix(3)}Step three.",
                  attachments=[{"type": "image", "url": "http://x"}]),
        _human_msg("5", "thanks!"),
    ]
    merged = r._merge_continuation_history(history)
    assert [m.metadata["ts"] for m in merged] == ["1", "2", "5"]
    combined = merged[1]
    assert combined.text == "Step one...\n\nStep two...\n\nStep three."
    assert "Part" not in combined.text and "Continued" not in combined.text
    assert combined.attachments == [{"type": "image", "url": "http://x"}]
    assert combined.metadata["reactions"] == [{"name": "eyes", "count": 1}]


def test_merge_leaves_human_messages_alone():
    r = _Rebuilder()
    history = [
        _human_msg("1", f"I read this somewhere: {CONTINUATION_TRAILER}"),
        _human_msg("2", "*Part 2 (continued)*\n\nquoting a bot, lol"),
    ]
    merged = r._merge_continuation_history(history)
    assert len(merged) == 2
    assert merged[0].text.endswith(CONTINUATION_TRAILER)  # humans never stripped


def test_orphaned_marker_on_self_message_is_stripped():
    r = _Rebuilder()
    # Only the part-1 landed inside the fetch window; its trailer must not leak.
    merged = r._merge_continuation_history(
        [_self_msg("2", f"partial answer{continuation_trailer()}")]
    )
    assert merged[0].text == "partial answer"


# ---------------- R3: placeholder filter precision ----------------

@pytest.mark.asyncio
async def test_thinking_filter_only_skips_own_placeholders():
    b = _Bot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "I'm Thinking about the Q3 plan"},
        {"ts": "2", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID,
         "text": ":loading: Thinking..."},
        {"ts": "3", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID,
         "text": ":circle-loader: Rebuilding thread history from Slack..."},
        {"ts": "4", "user": "U07HUMAN", "text": "Thinking... maybe we should ship it"},
        {"ts": "5", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "Real answer"},
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )
    result = await b.get_thread_history("C1", "1")
    assert [m.metadata["ts"] for m in result] == ["1", "4", "5"]


# ---------------- R4: oldest= plumbing ----------------

@pytest.mark.asyncio
async def test_get_thread_history_passes_oldest():
    b = _Bot()
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": [], "response_metadata": {}}
    )
    await b.get_thread_history("C1", "1", oldest="1700000000.000100")
    kwargs = b.app.client.conversations_replies.await_args.kwargs
    assert kwargs["oldest"] == "1700000000.000100"


def test_rebuild_passes_boundary_as_oldest():
    import message_processor.thread_management as tm
    src = inspect.getsource(tm.ThreadManagementMixin._get_or_rebuild_thread_state)
    assert 'oldest=(summary_row["boundary_ts"]' in src


# ---------------- R5: has_more ----------------

@pytest.mark.asyncio
async def test_history_tool_reports_has_more():
    from slack_client.history_tool import SlackHistoryToolMixin

    class _Hist(SlackHistoryToolMixin):
        def __init__(self):
            self.app = MagicMock()
        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_error(self, *a, **k): pass
        def log_warning(self, *a, **k): pass

    h = _Hist()
    h._channel_is_accessible = AsyncMock(return_value=(True, "ok"))
    h.app.client.conversations_history = AsyncMock(return_value={
        "messages": [{"user": "U1", "ts": "1", "text": "x"}],
        "response_metadata": {"next_cursor": "abc"},
    })
    result = await h.fetch_history_tool("C1", limit=1)
    assert result["ok"] is True and result["has_more"] is True and "note" in result

    h.app.client.conversations_history = AsyncMock(return_value={
        "messages": [{"user": "U1", "ts": "1", "text": "x"}],
        "response_metadata": {},
    })
    result = await h.fetch_history_tool("C1", limit=5)
    assert result["has_more"] is False and "note" not in result


# ---------------- W1-W3: write-side split safety ----------------

def test_entity_safe_cut():
    text = "hello <@U0123456789> world"
    # Cutting inside the mention scans back to before '<'
    assert entity_safe_cut(text, 12) == 6
    # Normal cut point unaffected
    assert entity_safe_cut("plain text " * 10, 20) == 20
    # Whole-string-is-one-entity falls back to limit (no infinite loop)
    assert entity_safe_cut("<" + "a" * 100, 50) == 50
    # Short text: full length
    assert entity_safe_cut("short", 100) == 5


def test_fence_safe_chunks_close_and_reopen():
    code = "\n".join(f"line{i}" for i in range(60))
    text = f"intro paragraph\n\n```python\n{code}\n```\n\nafter"
    chunks = fence_safe_chunks(text, 200)
    assert len(chunks) > 1
    open_counts = [c.count("```") for c in chunks]
    assert all(n % 2 == 0 for n in open_counts), "every chunk must have balanced fences"
    # A continuation chunk that carries code must reopen with the language hint
    reopened = [c for c in chunks[1:] if c.startswith("```python")]
    assert reopened, "expected at least one chunk reopening the python fence"


def test_fence_safe_chunks_hard_wraps_oversized_fragment():
    blob = "x" * 5000  # no split boundaries at all
    chunks = fence_safe_chunks(blob, 1000)
    assert all(len(c) <= 1010 for c in chunks)
    assert "".join(chunks) == blob


@pytest.mark.asyncio
async def test_send_message_split_uses_continued_markers_and_isolates_failures():
    b = _Bot()
    calls = []

    async def post(channel, thread_ts, text):
        calls.append(text)
        if len(calls) == 2:
            raise _slack_error(error="msg_too_long", status=400)
        return {"ok": True}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)
    long_text = ("para " * 300 + "\n\n") * 8  # ~12k chars -> 4 chunks
    ok = await b.send_message("C1", "1", long_text)
    assert ok is True  # some chunks landed despite chunk-2 failure
    assert len(calls) >= 3, "failure on chunk 2 must not abort later chunks"
    assert calls[0].rstrip().endswith(CONTINUATION_TRAILER)
    assert calls[1].startswith(CONTINUATION_HEAD)
    assert "Part 1/" not in calls[0]  # old style retired


def test_overflow_completion_flush_handles_oversize():
    """W1: the current_part>1 completion flush must split, not silently truncate."""
    import message_processor.handlers.text as th
    src = inspect.getsource(th)
    marker = src.index("W1: the buffer can outgrow")
    window = src[marker:marker + 1800]
    assert "entity_safe_cut(final_part_text, 3800)" in window
    assert "send_message(" in window  # remainder actually posts
    assert "CONTINUATION_HEAD" in window
