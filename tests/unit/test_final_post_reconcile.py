"""Reconcile-before-retry on an ambiguous final-post transport failure.

The bug: a `final_post_only` turn posts its whole answer once, at the end, through
`send_message`. `chat.postMessage` has no server-side idempotency key, so when the request
REACHES Slack but its response times out (a non-SlackApiError transport exception), the old
code let the exception propagate → text.py marked `final_post_failed` → main.py re-posted the
same answer. Two identical replies.

The fix lives in `SlackMessagingMixin.send_message`: only the ambiguous branch (a
non-SlackApiError raised by the post) does new work — it queries recent history and, if our
own bot already posted this text, returns that ts as success so nothing re-posts. The happy
path and the definitive-failure (SlackApiError) path pay nothing.
"""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Host(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Messaging host with self-identity resolved and a fully stubbed Slack client."""

    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.app = SimpleNamespace(client=SimpleNamespace(
            chat_postMessage=AsyncMock(),
            conversations_replies=AsyncMock(),
            conversations_history=AsyncMock(),
        ))
        # Self-identity (normally resolved once via auth_test on start).
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = None
        self.self_team_id = None
        # No channel-pulse in these tests — _record_own_reply_pulse then no-ops.
        self.channel_pulse = None
        # Identity mrkdwn conversion so `format_text` is a passthrough — keeps the
        # sent-vs-stored comparison exact for plain-text fixtures.
        self.markdown_converter = SimpleNamespace(convert=lambda t: t)

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _recent_ts():
    """A Slack ts inside the reconcile window and AFTER the attempt start (a few seconds ago,
    comfortably within the ~5s clock-skew allowance)."""
    return f"{time.time() - 3:.6f}"


def _ts_ago(secs):
    """A Slack ts `secs` seconds in the past — for messages that predate the attempt start."""
    return f"{time.time() - secs:.6f}"


def _own_msg(host, text, ts=None):
    """A history entry authored by the bot itself, carrying `text` as Slack stored it."""
    return {"user": host.bot_user_id, "ts": ts or _recent_ts(), "text": host.format_text(text)}


# A ~250-char body: long enough that prefix matching (>=200 both sides) is permitted.
_LONG_BODY = "The deployment finished cleanly. " + ("All services are green and healthy. " * 6)


@pytest.mark.asyncio
async def test_happy_path_never_reconciles():
    """A successful post returns the ts and does NOT touch history at all."""
    host = _Host()
    host.app.client.chat_postMessage.return_value = {"ts": "111.1"}

    ts = await host.send_message("C1", "10.0", "the final answer")

    assert ts == "111.1"
    host.app.client.chat_postMessage.assert_awaited_once()
    host.app.client.conversations_replies.assert_not_called()
    host.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_slack_api_error_does_not_reconcile():
    """A definitive API rejection is swallowed to None as before — no reconcile."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = SlackApiError(
        "channel_not_found", {"error": "channel_not_found"})

    ts = await host.send_message("C1", "10.0", "the final answer")

    assert ts is None  # unchanged legacy behavior — outer handler returns None
    host.app.client.conversations_replies.assert_not_called()
    host.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_transport_error_message_found_returns_success_thread():
    """Ambiguous timeout + the reply IS in the thread → return its ts, post exactly once.

    A truthy return is what makes the caller record `posted`/`streamed` true, so main.py
    never re-posts. No duplicate chat.postMessage is issued by the reconcile path."""
    host = _Host()
    landed_ts = _recent_ts()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.return_value = {
        "messages": [
            {"user": "U999", "ts": _recent_ts(), "text": "someone else"},
            _own_msg(host, "the final answer", ts=landed_ts),
        ]
    }

    ts = await host.send_message("C1", "10.0", "the final answer")

    assert ts == landed_ts  # the landed message — treated as success
    host.app.client.chat_postMessage.assert_awaited_once()  # no re-post from here
    host.app.client.conversations_replies.assert_awaited_once()
    host.app.client.conversations_history.assert_not_called()
    # The query is scoped to the attempt window: `oldest` anchored near now, `inclusive` on.
    _, kwargs = host.app.client.conversations_replies.await_args
    assert kwargs["inclusive"] is True
    assert float(kwargs["oldest"]) >= time.time() - 30  # not the beginning of the thread


@pytest.mark.asyncio
async def test_transport_error_channel_target_uses_history():
    """A top-level channel reply (thread_id=None) reconciles via conversations.history."""
    host = _Host()
    landed_ts = _recent_ts()
    host.app.client.chat_postMessage.side_effect = ConnectionResetError("reset")
    host.app.client.conversations_history.return_value = {
        "messages": [_own_msg(host, "top-level channel answer", ts=landed_ts)]
    }

    ts = await host.send_message("C1", None, "top-level channel answer")

    assert ts == landed_ts
    host.app.client.conversations_history.assert_awaited_once()
    host.app.client.conversations_replies.assert_not_called()
    _, kwargs = host.app.client.conversations_history.await_args
    assert kwargs["inclusive"] is True
    assert float(kwargs["oldest"]) >= time.time() - 30


@pytest.mark.asyncio
async def test_transport_error_message_not_found_reports_failure():
    """Ambiguous timeout but the reply is NOT there → re-raise so the caller's retry runs.

    text.py's `except Exception` then sets final_post_failed and main.py posts once (correct:
    a missing answer is worse than a rare duplicate)."""
    host = _Host()
    boom = TimeoutError("response never arrived")
    host.app.client.chat_postMessage.side_effect = boom
    host.app.client.conversations_replies.return_value = {
        "messages": [
            # Our own message, freshly posted, but a genuinely DIFFERENT reply.
            _own_msg(host, "a completely different earlier reply", ts=_recent_ts()),
            {"user": "U999", "ts": _recent_ts(), "text": "the final answer"},  # not ours
        ]
    }

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "the final answer")

    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_transport_error_reconcile_query_also_fails_reports_failure():
    """If the reconcile query itself raises, treat as not-found → re-raise (retry allowed)."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.side_effect = SlackApiError(
        "ratelimited", {"error": "ratelimited"})

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "the final answer")

    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_own_message_outside_window_is_not_a_match():
    """A matching own message older than the reconcile window must NOT count as the landed post."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    # ts well before now - 120s.
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, "the final answer", ts="1.0")]
    }

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "the final answer")


@pytest.mark.asyncio
async def test_identical_prior_reply_before_attempt_is_not_a_match():
    """Finding 1: an IDENTICAL earlier reply that predates this attempt must not be mistaken for
    the landed post. The bot said the same thing 60s ago (inside the old 120s window) but before
    we tried to post now — the new post is genuinely lost and the caller must retry."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, "the final answer", ts=_ts_ago(60))]
    }

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "the final answer")

    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_short_prefix_superset_is_not_a_match():
    """Finding 1: a short own reply that is a prefix of the attempted text (or vice versa) must
    NOT match — prefix matching is permitted only when both strings are long. Here the bot posted
    a fresh "OK" and we are trying to post "OK, I updated the deployment now"."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, "OK", ts=_recent_ts())]
    }

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "OK, I updated the deployment now")

    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_prefix_still_matches():
    """Prefix matching remains available for long messages (Slack may append/trim chrome around
    the fallback text): both normalized strings are >= 200 chars, so a shared 200-char prefix is
    still a valid landed-post signal."""
    host = _Host()
    landed_ts = _recent_ts()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    # Stored copy carries an extra trailing chrome fragment — a superset of the attempted text.
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, _LONG_BODY + " (edited)", ts=landed_ts)]
    }

    ts = await host.send_message("C1", "10.0", _LONG_BODY)

    assert ts == landed_ts
    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_shared_prefix_then_diverging_is_not_a_match():
    """Finding 1 (deeper): two long replies that share a 200+ char prefix then DIVERGE must NOT
    match. The old code compared only the first 200 chars (`startswith(target[:200])`), so any two
    replies sharing that much boilerplate collided even when they diverged immediately after; the
    fix compares the WHOLE shorter string."""
    host = _Host()
    shared = "The nightly batch reconciled every ledger entry without error. " * 4  # ~248 chars
    assert len(shared) >= 200
    stored = shared + "Then it archived the logs and exited."
    attempted = shared + "Then it emailed the on-call engineer instead."
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, stored, ts=_recent_ts())]
    }

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", attempted)

    host.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_follows_cursor_match_on_second_page():
    """Finding 2: conversations.replies returns the EARLIEST in-window messages first, so with more
    than one page of in-window replies the freshly-posted tail can land on a later page. The
    reconcile must follow `response_metadata.next_cursor` and scan every page — with limit raised
    to 100 — not just the first."""
    host = _Host()
    landed_ts = _recent_ts()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    page1 = {
        "messages": [_own_msg(host, f"status update {i}", ts=_recent_ts()) for i in range(25)],
        "response_metadata": {"next_cursor": "CURSOR2"},
    }
    page2 = {
        "messages": [_own_msg(host, "the tail answer", ts=landed_ts)],
        "response_metadata": {"next_cursor": ""},
    }
    host.app.client.conversations_replies.side_effect = [page1, page2]

    ts = await host.send_message("C1", "10.0", "the tail answer")

    assert ts == landed_ts  # found on page 2
    assert host.app.client.conversations_replies.await_count == 2
    first_kwargs = host.app.client.conversations_replies.await_args_list[0].kwargs
    second_kwargs = host.app.client.conversations_replies.await_args_list[1].kwargs
    assert first_kwargs["limit"] == 100  # raised from the old 20
    assert second_kwargs["cursor"] == "CURSOR2"  # followed the page-1 cursor


@pytest.mark.asyncio
async def test_reconcile_query_error_on_second_page_reports_failure():
    """A query error ANYWHERE in the pagination (here on page 2) means 'not found' → re-raise so
    the caller's retry runs. A partial scan must never be treated as a definitive miss-then-match."""
    host = _Host()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    page1 = {
        "messages": [_own_msg(host, f"status update {i}", ts=_recent_ts()) for i in range(25)],
        "response_metadata": {"next_cursor": "CURSOR2"},
    }
    host.app.client.conversations_replies.side_effect = [
        page1, SlackApiError("ratelimited", {"error": "ratelimited"})]

    with pytest.raises(TimeoutError):
        await host.send_message("C1", "10.0", "the tail answer")

    assert host.app.client.conversations_replies.await_count == 2


@pytest.mark.asyncio
async def test_long_thread_query_is_anchored_to_the_attempt():
    """Finding 2: in a long thread, the reconcile query must be scoped to the attempt window
    (`oldest` + `inclusive`) so the freshly-posted tail is returned rather than the thread's
    first page. Assert the exact kwargs the SDK is called with."""
    host = _Host()
    landed_ts = _recent_ts()
    host.app.client.chat_postMessage.side_effect = TimeoutError("response never arrived")
    host.app.client.conversations_replies.return_value = {
        "messages": [_own_msg(host, "the tail answer", ts=landed_ts)]
    }

    ts = await host.send_message("C1", "10.0", "the tail answer")

    assert ts == landed_ts
    _, kwargs = host.app.client.conversations_replies.await_args
    assert kwargs["inclusive"] is True
    assert "oldest" in kwargs
    # The floor sits within a few seconds of now (attempt_start minus the small skew), never at
    # the beginning of the thread.
    assert float(kwargs["oldest"]) >= time.time() - 30


def test_normalize_for_match_unescapes_and_collapses_whitespace():
    """Whitespace runs and Slack HTML-escaping must not defeat the match."""
    n = _Host._normalize_for_match
    assert n("a &amp; b\n\n  c") == n("a & b c")
    assert n("<tag>  &lt;ok&gt;") == n("<tag> <ok>")
    assert n("") == ""
