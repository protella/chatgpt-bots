"""F7 — tool-use provenance.

Covers the pure helpers (gist/build/render/strip incl. the anti-shielding strip), the
DB layer (roundtrip, idempotent upsert, age sweep), the processor persistence seam
(enabled/disabled/empty/no-ts), and deterministic rebuild reinjection with the pinned
footer-strip → used-tools → reactions ordering, compaction-boundary skipping, config-off,
and silent DB failure.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from base_client import Message
from config import config
from message_processor import tool_provenance as tp
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from thread_manager import AsyncThreadStateManager


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


def _hist(ts, text, sender="human", reactions=None):
    return Message(
        text=text, user_id="U1", channel_id="C1", thread_id="100.0", attachments=[],
        metadata={"ts": ts, "is_bot": sender == "self", "sender_type": sender,
                  "bot_name": None, "username": "Peter", "reactions": reactions},
    )


def _incoming(ts="200.0", text="latest"):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata={"ts": ts})


def _client(history):
    c = MagicMock()
    c.get_thread_history = AsyncMock(return_value=history)
    c.name = "slack"
    c.user_cache = {}
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
def _provenance_on(monkeypatch):
    monkeypatch.setattr(config, "enable_tool_provenance", True)


# ------------------------------------------------------------------ pure helpers

def test_gist_from_arguments_scalars_and_containers():
    # COUNT keys keep numeric values; TS keys keep Slack-ts / plain-number values.
    assert tp.gist_from_arguments('{"limit": 50, "before": "169.5"}') == "limit=50, before=169.5"
    assert tp.gist_from_arguments('{"oldest": "1690000000.000100"}') == "oldest=1690000000.000100"
    assert tp.gist_from_arguments('{"ids": [1, 2, 3]}') == "ids=[3]"
    assert tp.gist_from_arguments("{}") == ""
    assert tp.gist_from_arguments("not json") == ""
    assert tp.gist_from_arguments(None) == ""


def test_gist_never_leaks_opaque_string_values():
    # M3/F6: a non-allowlisted value is user content (query/prompt/URL/token) and must NEVER
    # be persisted verbatim — only that a value was present (`<str>`). This holds even for
    # NUMBERS on non-allowlisted keys (a numeric token would otherwise leak) and for
    # allowlisted keys whose value fails its per-key validator.
    assert tp.gist_from_arguments('{"query": "secret search terms"}') == "query=<str>"
    assert tp.gist_from_arguments('{"emoji": "eyes"}') == "emoji=<str>"
    assert tp.gist_from_arguments('{"url": "https://x/y?token=abc"}') == "url=<str>"
    assert tp.gist_from_arguments('{"token": 123456}') == "token=<str>"          # numeric leak blocked
    assert tp.gist_from_arguments('{"before": "https://x/?t=secret"}') == "before=<str>"  # ts key, bad value
    assert tp.gist_from_arguments('{"limit": "50; DROP"}') == "limit=<str>"       # count key, non-numeric
    # Container shapes still summarize by kind+size (no leak).
    assert tp.gist_from_arguments('{"q": "hi", "opts": {"a": 1}}') == "q=<str>, opts={1}"
    # Numbers on COUNT keys and booleans anywhere are always safe.
    assert tp.gist_from_arguments('{"top_k": 5, "stream": true}') == "top_k=5, stream=True"


def test_gist_is_length_capped():
    big = '{"prompt": "' + "x" * 500 + '"}'
    assert len(tp.gist_from_arguments(big)) <= config.tool_provenance_gist_chars


def test_build_provenance_combines_local_then_external_and_caps():
    local = [{"name": "fetch_channel_history", "ok": True, "gist": "limit=50"},
             {"name": "react_to_message", "ok": True, "gist": "emoji=eyes"}]
    out = tp.build_provenance(local, ["web_search"])
    assert out == [
        {"tool_name": "fetch_channel_history", "gist": "limit=50"},
        {"tool_name": "react_to_message", "gist": "emoji=eyes"},
        {"tool_name": "web_search", "gist": ""},
    ]
    many = [{"name": f"t{i}", "ok": True, "gist": ""} for i in range(30)]
    assert len(tp.build_provenance(many, ["a", "b", "c"])) == config.tool_provenance_max_entries


def test_render_prefers_gists_then_degrades_to_names():
    short = [{"tool_name": "fetch_channel_history", "gist": "limit=50"},
             {"tool_name": "web_search", "gist": ""}]
    assert tp.render_used_tools_annotation(short) == \
        "[used tools: fetch_channel_history(limit=50), web_search]"
    # Long gists blow the budget → names only.
    long = [{"tool_name": f"tool_{i}", "gist": "x" * 40} for i in range(6)]
    rendered = tp.render_used_tools_annotation(long)
    assert "(" not in rendered and rendered.startswith("[used tools: tool_0, ")
    assert tp.render_used_tools_annotation([]) == ""


def test_render_annotation_carries_more_than_eight_entries():
    # F14: the record used to cap at 8; a 12-entry turn must now render every entry
    # (default budget 20 entries / 300-char line), so a late-but-load-bearing call isn't dropped.
    tools = [{"tool_name": f"tool_{i}", "gist": ""} for i in range(12)]
    rendered = tp.render_used_tools_annotation(tools)
    assert "tool_8" in rendered and "tool_11" in rendered
    assert rendered.count(",") == 11  # all 12 names present


def test_strip_footer_and_anti_shielding():
    assert tp.strip_used_tools_footer("hi\n\n_Tools Used: web_search_") == "hi"
    # A trailing [used tools:]/[reactions:] annotation must NOT shield the footer.
    shielded = "hi\n\n_Tools Used: web_search_\n[used tools: web_search]\n[reactions: :eyes: x1]"
    assert tp.strip_used_tools_footer(shielded) == "hi\n[used tools: web_search]\n[reactions: :eyes: x1]"
    assert tp.strip_used_tools_footer("no footer here") == "no footer here"
    assert tp.strip_used_tools_footer(None) is None


def test_strip_footer_still_catches_the_legacy_wording():
    """The footer was renamed "Used Tools" -> "Tools Used" on 2026-07-11, but Slack is the
    transcript: every reply posted before then still carries the old wording and comes back
    on every rebuild. If the stripper stops matching it, stale chrome leaks into model context."""
    assert tp.strip_used_tools_footer("hi\n\n_Used Tools: web_search_") == "hi"
    legacy_shielded = "hi\n\n_Used Tools: web_search_\n[used tools: web_search]"
    assert tp.strip_used_tools_footer(legacy_shielded) == "hi\n[used tools: web_search]"
    assert tp.strip_used_tools_footer("hi\n\n_Used Tools: a, b (failed: c)_") == "hi"


# ------------------------------------------------------------------ DB layer

@pytest.mark.asyncio
async def test_db_save_and_get_roundtrip(temp_db):
    tools = [{"tool_name": "fetch_channel_history", "gist": "limit=50"}]
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", tools)
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": tools}


@pytest.mark.asyncio
async def test_db_save_merges_not_last_write_wins(temp_db):
    # A re-persist for the same reply MERGES (union by tool_name) rather than clobbering,
    # so a second pass can't drop tools recorded by the first (one row, both tools).
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "b", "gist": ""}])
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": [{"tool_name": "a", "gist": ""}, {"tool_name": "b", "gist": ""}]}


@pytest.mark.asyncio
async def test_db_save_merge_upgrades_empty_gist(temp_db):
    # An empty gist for a tool is upgraded when a later pass supplies a non-empty one.
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": "limit=5"}])
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": [{"tool_name": "a", "gist": "limit=5"}]}


def test_merge_preserves_multiple_executions_dedupes_exact_only():
    from database import DatabaseManager as DM
    # Same tool, DIFFERENT gists = two real executions → both kept (ordered).
    out = DM._merge_tool_provenance(
        [{"tool_name": "fetch", "gist": "limit=5"}],
        [{"tool_name": "fetch", "gist": "limit=10"}])
    assert out == [{"tool_name": "fetch", "gist": "limit=5"},
                   {"tool_name": "fetch", "gist": "limit=10"}]
    # EXACT duplicate (same name AND gist) → deduped to one.
    out = DM._merge_tool_provenance(
        [{"tool_name": "fetch", "gist": "limit=5"}],
        [{"tool_name": "fetch", "gist": "limit=5"}])
    assert out == [{"tool_name": "fetch", "gist": "limit=5"}]
    # Empty-gist placeholder upgraded in place; a later empty is absorbed by the non-empty.
    out = DM._merge_tool_provenance(
        [{"tool_name": "fetch", "gist": ""}],
        [{"tool_name": "fetch", "gist": "limit=5"}, {"tool_name": "fetch", "gist": ""}])
    assert out == [{"tool_name": "fetch", "gist": "limit=5"}]


@pytest.mark.asyncio
async def test_db_age_sweep_deletes_old_rows(temp_db):
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    # Backdate the row well past the retention window.
    temp_db.conn.execute(
        "UPDATE message_tool_usage SET created_at = datetime('now', '-200 days')")
    temp_db.conn.commit()
    temp_db.delete_old_tool_usage(days=90)
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") == {}


@pytest.mark.asyncio
async def test_db_age_sweep_honors_config_retention_days(temp_db, monkeypatch):
    # F14: the sweep window comes from config.tool_usage_retention_days (wired at the
    # cleanup call site). A row aged 50 days survives the default 90 but not a 30-day window.
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    temp_db.conn.execute(
        "UPDATE message_tool_usage SET created_at = datetime('now', '-50 days')")
    temp_db.conn.commit()
    monkeypatch.setattr(config, "tool_usage_retention_days", 90, raising=False)
    temp_db.delete_old_tool_usage(days=config.tool_usage_retention_days)
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") != {}  # within 90d window
    monkeypatch.setattr(config, "tool_usage_retention_days", 30, raising=False)
    temp_db.delete_old_tool_usage(days=config.tool_usage_retention_days)
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") == {}  # past 30d window


@pytest.mark.asyncio
async def test_db_get_is_silent_on_missing_table(temp_db):
    temp_db.conn.execute("DROP TABLE message_tool_usage")
    temp_db.conn.commit()
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") == {}  # no raise


# ---------------------------------------------- delivered-ts selection (F5/F7)

def test_delivered_ts_native_path_uses_native_current_ts():
    from message_processor.handlers.text import _delivered_stream_ts
    from types import SimpleNamespace
    native = SimpleNamespace(current_ts="999.9")
    # Native finalize confirmed → the native stream's own message ts, NOT the (stale)
    # legacy current_message_id. (native_finalized already implies content delivery.)
    assert _delivered_stream_ts(native, True, "111.1", True) == "999.9"


def test_delivered_ts_legacy_path_uses_current_message_id():
    from message_processor.handlers.text import _delivered_stream_ts
    from types import SimpleNamespace
    # Native ran but did NOT finalize (fell back to legacy edits) → the final
    # current_message_id from the update loop — but ONLY when content was delivered.
    native = SimpleNamespace(current_ts="999.9")
    assert _delivered_stream_ts(native, False, "222.2", True) == "222.2"
    # No native coordinator at all (pure legacy streaming).
    assert _delivered_stream_ts(None, False, "333.3", True) == "333.3"


def test_delivered_ts_none_when_content_not_delivered():
    from message_processor.handlers.text import _delivered_stream_ts
    from types import SimpleNamespace
    # A placeholder/current id exists but EVERY content flush failed → not a real delivery,
    # so the legacy path returns None (no phantom posted/pulse/provenance).
    assert _delivered_stream_ts(None, False, "333.3", False) is None
    assert _delivered_stream_ts(SimpleNamespace(current_ts="9"), False, "333.3", False) is None
    # Nothing delivered at all → None.
    assert _delivered_stream_ts(None, False, None, False) is None


# ------------------------------------------------------------------ persistence seam

@pytest.mark.asyncio
async def test_persist_schedules_save_when_enabled():
    proc = _Proc(db=MagicMock(save_tool_usage_async=AsyncMock()))
    prov = [{"tool_name": "web_search", "gist": ""}]
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", prov)
    await asyncio.sleep(0)
    proc.db.save_tool_usage_async.assert_awaited_once_with("C1", "101.0", "C1:100.0", prov)


@pytest.mark.asyncio
async def test_persist_noop_when_disabled_empty_or_no_ts(monkeypatch):
    proc = _Proc(db=MagicMock(save_tool_usage_async=AsyncMock()))
    prov = [{"tool_name": "web_search", "gist": ""}]
    # disabled
    monkeypatch.setattr(config, "enable_tool_provenance", False)
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", prov)
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    # empty provenance (no tools ran)
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", [])
    # no ts (reaction-only / statusless turn)
    proc._persist_tool_provenance("C1", None, "C1:100.0", prov)
    await asyncio.sleep(0)
    proc.db.save_tool_usage_async.assert_not_awaited()


# ------------------------------------------------------------------ rebuild reinjection

@pytest.mark.asyncio
async def test_rebuild_annotates_matching_bot_message(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0",
        [{"tool_name": "fetch_channel_history", "gist": "limit=50"}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Here are the threads.", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    bot_msg = next(m for m in state.messages if m["role"] == "assistant")
    assert "[used tools: fetch_channel_history(limit=50)]" in bot_msg["content"]


@pytest.mark.asyncio
async def test_rebuild_ordering_footer_stripped_then_used_then_reactions(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.\n\n_Used Tools: web_search_", sender="self",
                     reactions=[{"name": "eyes", "count": 1, "users": ["U9"]}])]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "_Used Tools:" not in content  # external chrome stripped, not in model context
    # F10 stamp is a pure PREFIX — it rides ahead of the pinned suffix order without
    # disturbing footer-strip → [used tools:] → [reactions:] (ts 101.0, self turn → UTC).
    assert content == ("[Thu 1970-01-01 12:01 AM UTC] Answer.\n"
                       "[used tools: web_search]\n[reactions: :eyes: x1 (<@U9>)]")


@pytest.mark.asyncio
async def test_rebuild_is_deterministic_across_repeats(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    history = [_hist("101.0", "Answer.", sender="self")]
    first = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    second = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    a = next(m for m in first.messages if m["role"] == "assistant")["content"]
    b = next(m for m in second.messages if m["role"] == "assistant")["content"]
    assert a == b


@pytest.mark.asyncio
async def test_rebuild_skips_rows_behind_compaction_boundary(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Earlier stuff.", "101.5", refs=[])
    # A row for a message AT/behind the boundary — its message is excluded, never annotated.
    await temp_db.save_tool_usage_async("C1", "101.0", thread_key,
                                        [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "old", sender="self"), _hist("102.0", "fresh tail")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    joined = " ".join(str(m.get("content")) for m in state.messages)
    assert "used tools" not in joined


@pytest.mark.asyncio
async def test_rebuild_config_off_leaves_content_untouched(temp_db, monkeypatch):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    monkeypatch.setattr(config, "enable_tool_provenance", False)
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.\n\n_Used Tools: web_search_", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "used tools:" not in content  # no F7 annotation
    assert "_Used Tools: web_search_" in content  # footer untouched at rebuild (as today)


@pytest.mark.asyncio
async def test_rebuild_survives_db_provenance_read_failure(temp_db, monkeypatch):
    proc = _Proc(db=temp_db)
    monkeypatch.setattr(temp_db, "get_thread_tool_usage_async",
                        AsyncMock(side_effect=RuntimeError("boom")))
    history = [_hist("101.0", "Answer.", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    # No annotation, but the rebuild completes and the message is present.
    assert any("Answer." in str(m.get("content")) for m in state.messages)
