"""F10 — per-message timestamps in model context.

Covers the pure stamp helpers (deterministic render, UTC fallbacks, non-string /
unparseable passthrough, sender-tz resolution order), the rebuild-loop stamping of
self and non-self turns, warm-inbound stamping that renders identically to the later
rebuild, coexistence with the pinned [used tools:]/[reactions:] suffix order and
footer stripping, the never-stamped compaction summary head, and the flag-off
byte-identical guarantee.

These stamps are DM/legacy only as of P2. The pulse envelope was the other stamped surface and
it is retired; a channel turn renders the stream, whose header carries its own UTC minute stamp
(channel_stream.iso_minute). The last section pins that the two renderings stay distinct — one
sits inside a cached prefix, the other does not — while describing the same instant.

(The participation-prompt time-awareness assertion is gone with the prompt it pinned: the binary
wake gate's prompt is ten lines about one bit and says nothing about time, because it no longer
judges whether an exchange is still open. The RESPONDER's date/time context is unchanged and is
covered by the stamp tests above.)
"""
import re

import pytest
from unittest.mock import AsyncMock, MagicMock

from base_client import Message
from config import config
from message_processor.message_timestamps import (
    render_message_timestamp,
    sender_timezone,
    stamp_content,
)
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from thread_manager import AsyncThreadStateManager

# A stamp is a leading "[Weekday YYYY-MM-DD H:MM AM/PM TZ]" bracket.
_STAMP_RE = re.compile(r"^\[(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} "
                       r"\d{1,2}:\d{2} (AM|PM) \w+\]")


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, MessageUtilitiesMixin):
    def __init__(self, db=None):
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = None
        self.document_handler = None

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info

    def _update_status(self, *a, **k): pass


def _hist(ts, text, sender="human", reactions=None, user_timezone=None):
    md = {"ts": ts, "is_bot": sender == "self", "sender_type": sender,
          "bot_name": None, "username": "Peter", "reactions": reactions}
    if user_timezone is not None:
        md["user_timezone"] = user_timezone
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata=md)


def _incoming(ts="200.0", text="latest"):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata={"ts": ts})


def _client(history, user_cache=None):
    c = MagicMock()
    c.get_thread_history = AsyncMock(return_value=history)
    c.name = "slack"
    c.user_cache = user_cache or {}
    c.bot_user_id = "UBOT"
    return c


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture(autouse=True)
def _timestamps_on(monkeypatch):
    # Default the suite ON (independent of the ambient .env), tests toggle off explicitly.
    monkeypatch.setattr(config, "enable_message_timestamps", True)


# ------------------------------------------------------------------ pure helpers

def test_render_is_pure_and_deterministic():
    ts = "1783692913.675809"
    # Fixed ts + tz → exact string (12-hour, non zero-padded hour, minute precision).
    assert render_message_timestamp(ts, "America/New_York") == "[Fri 2026-07-10 10:15 AM EDT]"
    assert render_message_timestamp(ts, "UTC") == "[Fri 2026-07-10 2:15 PM UTC]"
    # Pure: no datetime.now() — repeated calls render byte-identically.
    assert render_message_timestamp(ts, "America/New_York") == \
        render_message_timestamp(ts, "America/New_York")


def test_render_utc_fallback_for_unknown_or_missing_tz():
    ts = "1783692913.675809"
    expected = "[Fri 2026-07-10 2:15 PM UTC]"
    assert render_message_timestamp(ts, None) == expected          # missing (e.g. other bot)
    assert render_message_timestamp(ts, "Not/AZone") == expected   # invalid IANA name
    assert render_message_timestamp(ts) == expected                # default arg


def test_render_unparseable_ts_returns_empty():
    assert render_message_timestamp(None) == ""
    assert render_message_timestamp("not-a-ts") == ""
    assert render_message_timestamp("", "UTC") == ""


def test_stamp_content_non_string_passthrough():
    payload = [{"type": "input_image"}]          # multimodal content list
    assert stamp_content(payload, "101.0", "UTC") is payload
    assert stamp_content({"x": 1}, "101.0", "UTC") == {"x": 1}
    assert stamp_content(None, "101.0", "UTC") is None


def test_stamp_content_prefixes_and_edge_cases():
    assert stamp_content("hi", "101.0", "UTC") == "[Thu 1970-01-01 12:01 AM UTC] hi"
    # Empty content → bare stamp (no trailing space).
    assert stamp_content("", "101.0", "UTC") == "[Thu 1970-01-01 12:01 AM UTC]"
    # Unparseable ts → content untouched (no-op), never a bare/empty prefix.
    assert stamp_content("hi", "bad-ts", "UTC") == "hi"


def test_sender_timezone_metadata_then_cache_then_utc():
    cache = {"U1": {"timezone": "Europe/Paris"}}
    # 1) explicit metadata user_timezone wins over the cache
    assert sender_timezone({"user_timezone": "America/Chicago"}, "U1", cache) == "America/Chicago"
    # 2) no metadata tz → in-memory user_cache (keyed by user_id, field 'timezone')
    assert sender_timezone({}, "U1", cache) == "Europe/Paris"
    assert sender_timezone(None, "U1", cache) == "Europe/Paris"
    # 3) nothing cached → UTC
    assert sender_timezone({}, "U1", {}) == "UTC"
    assert sender_timezone({}, None, cache) == "UTC"
    assert sender_timezone({}, "U2", cache) == "UTC"     # unknown user


# ------------------------------------------------------------------ rebuild path

@pytest.mark.asyncio
async def test_rebuild_stamps_self_and_non_self_turns(temp_db):
    proc = _Proc(db=temp_db)
    history = [
        _hist("1783692913.675809", "human question", sender="human"),
        _hist("1783693000.000000", "bot answer", sender="self"),
    ]
    # Non-self sender tz resolves from the client user_cache.
    client = _client(history, user_cache={"U1": {"timezone": "America/New_York"}})
    state = await proc._get_or_rebuild_thread_state(_incoming(), client)

    user_msg = next(m for m in state.messages if m["role"] == "user")
    bot_msg = next(m for m in state.messages if m["role"] == "assistant")
    # Non-self: "[stamp in sender tz] username: text"
    assert user_msg["content"] == "[Fri 2026-07-10 10:15 AM EDT] Peter: human question"
    # Self: stamped in UTC (bot's own turns fall back to UTC), no username prefix.
    assert bot_msg["content"] == "[Fri 2026-07-10 2:16 PM UTC] bot answer"


@pytest.mark.asyncio
async def test_rebuild_stamp_coexists_with_used_tools_reactions_and_footer_strip(temp_db, monkeypatch):
    # Self turn with an external _Used Tools:_ footer, a persisted provenance row, and a
    # reaction — the stamp must ride as a pure prefix without disturbing the pinned
    # footer-strip → [used tools:] → [reactions:] suffix order.
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.\n\n_Used Tools: web_search_", sender="self",
                     reactions=[{"name": "eyes", "count": 1, "users": ["U9"]}])]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert content == ("[Thu 1970-01-01 12:01 AM UTC] Answer.\n"
                       "[used tools: web_search]\n[reactions: :eyes: x1 (<@U9>)]")


@pytest.mark.asyncio
async def test_summary_head_is_never_stamped(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Earlier stuff.", "101.5", refs=[])
    proc = _Proc(db=temp_db)
    # 101.0 is behind the boundary (skipped); 102.0 is the fresh, stamped tail.
    history = [_hist("101.0", "old", sender="self"), _hist("102.0", "fresh tail")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    head = next(m for m in state.messages
                if (m.get("metadata") or {}).get("type") == proc.SUMMARY_HEAD_MARKER)
    assert head["content"].startswith("--- SUMMARY OF EARLIER CONVERSATION ---")
    assert not _STAMP_RE.match(head["content"])          # summary head stays timestamp-free
    # ...but the fresh tail turn IS stamped (proves the flag was on for this rebuild).
    tail = next(m for m in state.messages if m["role"] == "user")
    assert _STAMP_RE.match(tail["content"])


# ------------------------------------------------------------------ warm path

@pytest.mark.asyncio
async def test_warm_inbound_stamp_matches_later_rebuild(temp_db):
    # A single non-self human turn carrying its own ts + user_timezone. The warm helper
    # (live append) and the later rebuild render the SAME ts+tz, so the content is
    # byte-identical between the two paths (determinism across warm/cold).
    proc = _Proc(db=temp_db)
    msg = _hist("1783692913.675809", "hello there", sender="human",
                user_timezone="America/New_York")
    warm = proc._format_user_content_with_username("hello there", msg)

    state = await proc._get_or_rebuild_thread_state(_incoming(), _client([msg]))
    rebuilt = next(m for m in state.messages if m["role"] == "user")["content"]
    assert warm == rebuilt == "[Fri 2026-07-10 10:15 AM EDT] Peter: hello there"


def test_warm_utc_fallback_and_empty_content():
    proc = _Proc()
    # No user_timezone on metadata → UTC.
    msg = _hist("101.0", "", sender="human")
    assert proc._format_user_content_with_username("", msg) == "[Thu 1970-01-01 12:01 AM UTC] Peter:"


# --------------------------------------------------------------- flag-off parity

@pytest.mark.asyncio
async def test_flag_off_is_byte_identical_everywhere(temp_db, monkeypatch):
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    proc = _Proc(db=temp_db)

    # warm
    msg = _hist("150.0", "hi", sender="human", user_timezone="America/New_York")
    assert proc._format_user_content_with_username("hi", msg) == "Peter: hi"

    # rebuild — no stamp on any turn
    history = [_hist("150.0", "q", sender="human"), _hist("151.0", "a", sender="self")]
    client = _client(history, user_cache={"U1": {"timezone": "America/New_York"}})
    state = await proc._get_or_rebuild_thread_state(_incoming(), client)
    assert next(m for m in state.messages if m["role"] == "user")["content"] == "Peter: q"
    assert next(m for m in state.messages if m["role"] == "assistant")["content"] == "a"


# ------------------------------------------------------- these stamps are DM-only now
#
# The pulse envelope was the other stamped surface, and it is retired: a channel turn renders
# the stream instead, and the stream's own header carries a UTC minute stamp built from the
# shared timestamp parser's SECONDS field (channel_stream.iso_minute). These helpers are the
# DM/legacy renderer — a per-sender local time, 12-hour clock, weekday and tz label — and the two
# must not converge, because one is inside a cached prefix and the other is not.

def test_the_channel_stream_header_does_not_use_the_dm_stamp():
    from message_processor.channel_stream import iso_minute
    assert iso_minute("1783692913.675809") == "2026-07-10 14:15"
    assert not _STAMP_RE.match(iso_minute("1783692913.675809"))
    assert render_message_timestamp("1783692913.675809", "America/New_York") == (
        "[Fri 2026-07-10 10:15 AM EDT]")


def test_the_two_stamps_read_the_same_instant():
    """Different renderings, one ts: the stream is UTC and cache-stable, the DM stamp is local
    to the sender. A drift between them would be a real reporting bug, not a formatting one."""
    from message_processor.channel_stream import iso_minute
    assert iso_minute("1783692913.675809").endswith("14:15")
    assert "10:15 AM EDT" in render_message_timestamp("1783692913.675809", "America/New_York")
