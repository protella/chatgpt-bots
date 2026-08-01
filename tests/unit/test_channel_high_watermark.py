"""H: what a channel turn is allowed to see, and how the listeners establish it.

The watermark module's own mechanics are tested in test_admission_watermark.py. This file is
about the WIRING — the synchronous admission step in the raw listeners, and the H a turn ends up
pinning as a result. The ordering rule is the whole point: admission happens before the first
await, so a turn that starts after Slack handed us an event can never pin an H that excludes it.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import HistoryFetchError
from slack_client import admission_watermark
from slack_client.event_handlers import registration
from slack_client.event_handlers.registration import _admit

TEAM = "T1"
CH = "C0BKX77NU66"
DM = "D0AAAAA"


class _Client:
    def __init__(self):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_BOT"

    def is_own_message(self, msg):
        if not isinstance(msg, dict):
            return False
        return msg.get("bot_id") == self.bot_id or msg.get("user") == self.bot_user_id


@pytest.fixture(autouse=True)
def _clean():
    admission_watermark.watermark.reset()
    yield
    admission_watermark.watermark.reset()


@pytest.fixture
def client():
    return _Client()


def _msg(ts, **extra):
    return {"channel": CH, "ts": ts, "event_ts": ts, "user": "U1", "text": "hi", **extra}


# ------------------------------------------------------------------ admission

def test_a_plain_message_advances_the_watermark_and_takes_a_ticket(client):
    ticket = _admit(client, _msg("1700000100.000000"))
    assert admission_watermark.current(CH) == "1700000100.000000"
    assert ticket is not None and ticket.channel_id == CH


def test_the_watermark_only_moves_forward(client):
    _admit(client, _msg("1700000200.000000"))
    _admit(client, _msg("1700000100.000000"))
    assert admission_watermark.current(CH) == "1700000200.000000"


def test_unequal_width_timestamps_compare_numerically(client):
    _admit(client, _msg("1700000000.10"))
    _admit(client, _msg("1700000000.5"))
    assert admission_watermark.current(CH) == "1700000000.5"


def test_an_edit_advances_with_the_OUTER_event_ts(client):
    """The activity happened now; the ts being edited may be hours old."""
    event = {"channel": CH, "subtype": "message_changed", "event_ts": "1700009999.000000",
             "message": {"ts": "1700000001.000000", "user": "U1", "text": "fixed",
                         "edited": {"ts": "1700009999.000000"}}}
    _admit(client, event)
    assert admission_watermark.current(CH) == "1700009999.000000"


def test_a_deletion_advances_with_the_OUTER_event_ts(client):
    event = {"channel": CH, "subtype": "message_deleted", "event_ts": "1700009999.000000",
             "deleted_ts": "1700000001.000000",
             "previous_message": {"ts": "1700000001.000000", "user": "U1", "text": "gone"}}
    _admit(client, event)
    assert admission_watermark.current(CH) == "1700009999.000000"


def test_our_own_message_never_advances_h(client):
    assert _admit(client, _msg("1700000100.000000", bot_id="B_BOT")) is None
    assert admission_watermark.current(CH) is None


def test_our_own_streaming_edit_never_advances_h(client):
    event = {"channel": CH, "subtype": "message_changed", "event_ts": "1700009999.000000",
             "message": {"ts": "1700000001.000000", "bot_id": "B_BOT", "text": "partial"}}
    assert _admit(client, event) is None
    assert admission_watermark.current(CH) is None


def test_dms_are_skipped_entirely(client):
    assert _admit(client, {"channel": DM, "ts": "1700000100.000000", "user": "U1"}) is None
    assert admission_watermark.current(DM) is None


def test_an_item_only_event_uses_the_item_channel(client):
    """Shape tolerance only. No listener actually routes one here — see the test below."""
    ticket = _admit(client, {"item": {"channel": CH}, "event_ts": "1700000100.000000"})
    assert ticket is not None
    assert admission_watermark.current(CH) == "1700000100.000000"


def test_only_the_message_and_mention_listeners_admit():
    """Reactions do not feed the index or H: the stream renders reaction state from the fetched
    message, not from an event. A reaction listener that admitted would issue a ticket whose
    activity nothing indexes."""
    source = inspect.getsource(registration.SlackRegistrationMixin._register_handlers)
    admitting = [block.split("\n")[0].strip()
                 for block in source.split("async def ")[1:]
                 if "_admit(self, event)" in block.split("async def ")[0]]
    assert admitting == ["handle_app_mention(event, say, client):",
                        "handle_message(event, say, client):"]


def test_a_channelless_event_is_ignored(client):
    assert _admit(client, {"ts": "1700000100.000000"}) is None
    assert _admit(client, "not a dict") is None


def test_admission_is_synchronous(client):
    """If this ever becomes a coroutine, H stops being pinned before the first await."""
    assert not inspect.iscoroutinefunction(_admit)


def test_admission_survives_a_client_without_identity():
    ticket = _admit(MagicMock(is_own_message=MagicMock(side_effect=RuntimeError)),
                    _msg("1700000100.000000"))
    assert ticket is not None


# ------------------------------------------------------------------ the pin

def test_h_is_the_max_of_the_watermark_and_the_trigger(client):
    _admit(client, _msg("1700000100.000000"))
    assert admission_watermark.pin(CH, "1700000050.000000").h == "1700000100.000000"
    assert admission_watermark.pin(CH, "1700000900.000000").h == "1700000900.000000"


def test_the_frontier_is_captured_with_h(client):
    first = _admit(client, _msg("1700000100.000000"))
    pin = admission_watermark.pin(CH, "1700000100.000000")
    later = _admit(client, _msg("1700000200.000000"))
    assert pin.frontier == first.seq
    assert later.seq > pin.frontier
    # H is pinned, not live: the later event does not move what this turn will render.
    assert pin.h == "1700000100.000000"
    assert admission_watermark.current(CH) == "1700000200.000000"


def test_pinning_with_nothing_admitted_fails_closed(client):
    with pytest.raises(HistoryFetchError):
        admission_watermark.pin(CH, None)


# ------------------------------------------------------------------ listener ordering

def test_both_listeners_admit_before_their_first_await():
    """Read the source rather than the behaviour: the ordering is the invariant, and a future
    edit that moves `_admit` below an await would still pass a behavioural test on a quiet loop.
    """
    source = inspect.getsource(registration.SlackRegistrationMixin._register_handlers)
    for listener in ("async def handle_app_mention", "async def handle_message"):
        body = source.split(listener, 1)[1]
        admit_at = body.index("_admit(self, event)")
        first_await = body.index("await ")
        assert admit_at < first_await, listener


async def test_a_ticket_is_completed_by_the_feed(client, monkeypatch):
    from slack_client.event_handlers import activity_index
    client.db = MagicMock()
    client.db.seed_channel_coverage_async = AsyncMock(return_value=True)
    client.db.record_thread_activity_async = AsyncMock(return_value=None)
    client.classify_sender = lambda msg: "human"
    ticket = _admit(client, _msg("1700000200.000000", thread_ts="1700000100.000000"))
    await activity_index.feed_thread_activity_index(client, _msg(
        "1700000200.000000", thread_ts="1700000100.000000"), ticket=ticket)
    assert ticket.state == admission_watermark.OK
    await admission_watermark.drain(CH, ticket.seq, timeout=0.01)


async def test_a_failed_feed_degrades_the_channel_and_the_turn_fails_closed(client, monkeypatch):
    from slack_client.event_handlers import activity_index
    monkeypatch.setattr(activity_index, "_FEED_RETRY_BASE_SECONDS", 0)
    client.db = MagicMock()
    client.db.seed_channel_coverage_async = AsyncMock(return_value=True)
    client.db.record_thread_activity_async = AsyncMock(side_effect=RuntimeError("disk"))
    client.classify_sender = lambda msg: "human"
    event = _msg("1700000200.000000", thread_ts="1700000100.000000")
    ticket = _admit(client, event)
    await activity_index.feed_thread_activity_index(client, event, ticket=ticket)
    assert ticket.state == admission_watermark.FAILED
    assert admission_watermark.is_degraded(CH)
    with pytest.raises(HistoryFetchError):
        await admission_watermark.drain(CH, ticket.seq, timeout=0.01)
    await admission_watermark.watermark.shutdown(timeout=0.01)


async def test_an_event_the_index_ignores_still_completes_its_ticket(client):
    from slack_client.event_handlers import activity_index
    client.db = MagicMock()
    client.db.seed_channel_coverage_async = AsyncMock(return_value=True)
    ticket = _admit(client, {"channel": DM, "ts": "1700000100.000000", "user": "U1",
                             "channel_type": "im"})
    assert ticket is None
    event = _msg("1700000300.000000")
    ticket = _admit(client, event)
    await activity_index.feed_thread_activity_index(client, event, ticket=ticket)
    assert ticket.state == admission_watermark.OK


def test_deleting_our_own_message_never_advances_h(client):
    event = {"channel": CH, "subtype": "message_deleted", "event_ts": "1700009999.000000",
             "deleted_ts": "1700000001.000000",
             "previous_message": {"ts": "1700000001.000000", "bot_id": "B_BOT",
                                  "text": "our uploading indicator"}}
    assert _admit(client, event) is None
    assert admission_watermark.current(CH) is None


async def test_a_cancelled_feed_never_leaves_its_ticket_pending(client):
    from slack_client.event_handlers import activity_index
    client.db = MagicMock()
    client.db.seed_channel_coverage_async = AsyncMock(return_value=True)
    client.db.record_thread_activity_async = AsyncMock(side_effect=asyncio.CancelledError)
    client.classify_sender = lambda msg: "human"
    event = _msg("1700000200.000000", thread_ts="1700000100.000000")
    ticket = _admit(client, event)
    with pytest.raises(asyncio.CancelledError):
        await activity_index.feed_thread_activity_index(client, event, ticket=ticket)
    assert ticket.state == admission_watermark.FAILED
    await admission_watermark.watermark.shutdown(timeout=0.01)
