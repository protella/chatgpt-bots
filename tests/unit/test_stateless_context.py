"""Phase S — Slack-native context: retire the message mirror.

Covers: rebuild-always-fetches, the edited-message staleness regression, summary
head + tail composition with boundary dedup, refs preserved through compaction,
chunky compaction, the mirror-drop migration, image injection via metadata ts,
deterministic rebuild serialization (prompt-cache hygiene), date-only prefix +
minute-time suffix, the summary-aware system-prompt note, reactions annotations,
and prompt_cache_key plumbing.
"""
import asyncio
import json
import os
import sqlite3

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from base_client import Message
from config import config
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from message_processor.handlers.vision import VisionHandlerMixin
from thread_manager import AsyncThreadStateManager, ThreadState


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, VisionHandlerMixin, MessageUtilitiesMixin):
    """Minimal processor binding the real mixins."""

    def __init__(self, db=None, openai_client=None):
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = openai_client
        self.document_handler = None

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def _update_status(self, *a, **k): pass


def _hist(ts, text, user="U1", sender="human", username="Peter", reactions=None):
    return Message(
        text=text, user_id=user, channel_id="C1", thread_id="100.0",
        attachments=[],
        metadata={"ts": ts, "is_bot": sender == "self", "sender_type": sender,
                  "bot_name": None, "username": username, "reactions": reactions},
    )


def _incoming(ts="200.0", text="latest question"):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata={"ts": ts})


async def _inject(proc, messages, state):
    """Call _inject_image_analyses tolerating both sync and async signatures
    (a parallel async-audit workstream is converting mixin methods)."""
    result = proc._inject_image_analyses(messages, state)
    if hasattr(result, "__await__"):
        result = await result
    return result


def _client_with_history(history):
    client = MagicMock()
    client.get_thread_history = AsyncMock(return_value=history)
    client.name = "slack"
    client.user_cache = {}
    client.bot_user_id = "UBOT"
    return client


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


# ------------------------------------------------------------- rebuild always fetches

@pytest.mark.asyncio
async def test_cold_rebuild_always_fetches_from_slack(temp_db):
    proc = _Proc(db=temp_db)
    client = _client_with_history([_hist("101.0", "hello bot")])
    state = await proc._get_or_rebuild_thread_state(_incoming(), client)
    client.get_thread_history.assert_awaited_once()
    assert any("hello bot" in str(m.get("content")) for m in state.messages)


@pytest.mark.asyncio
async def test_warm_state_does_not_refetch(temp_db):
    proc = _Proc(db=temp_db)
    client = _client_with_history([_hist("101.0", "hello bot")])
    await proc._get_or_rebuild_thread_state(_incoming(), client)
    client.get_thread_history.reset_mock()
    await proc._get_or_rebuild_thread_state(_incoming(ts="201.0"), client)
    client.get_thread_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_staleness_regression_edited_message_appears(temp_db):
    """THE Phase S bug fix: an edit in Slack must be visible after a cold rebuild —
    no DB rows may resurrect the pre-edit text."""
    proc = _Proc(db=temp_db)
    client = _client_with_history([_hist("101.0", "original question")])
    await proc._get_or_rebuild_thread_state(_incoming(), client)

    # Simulate restart: fresh manager (in-memory state gone), same DB
    proc2 = _Proc(db=temp_db)
    client2 = _client_with_history([_hist("101.0", "EDITED question")])
    state = await proc2._get_or_rebuild_thread_state(_incoming(), client2)

    joined = json.dumps([m.get("content") for m in state.messages])
    assert "EDITED question" in joined
    assert "original question" not in joined


# ------------------------------------------- summary head + tail + boundary dedup

@pytest.mark.asyncio
async def test_summary_head_and_tail_composition(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Earlier: they planned the launch.", "101.5",
                                refs=[{"kind": "file", "value": "plan.pdf", "name": "plan.pdf"}])
    proc = _Proc(db=temp_db)
    history = [
        _hist("101.0", "old covered message"),     # <= boundary: excluded
        _hist("101.5", "boundary message"),        # == boundary: excluded
        _hist("102.0", "fresh tail message"),      # > boundary: included
    ]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client_with_history(history))

    head = state.messages[0]
    assert head["role"] == "developer"
    assert (head.get("metadata") or {}).get("type") == "thread_summary"
    assert "Earlier: they planned the launch." in head["content"]
    assert "plan.pdf" in head["content"]

    joined = json.dumps([m.get("content") for m in state.messages])
    assert "fresh tail message" in joined
    assert "old covered message" not in joined
    assert "boundary message" not in joined
    assert state.has_summary_head is True


# --------------------------------------------------- compaction + refs preservation

def _mock_openai(summary="ROLLED-UP SUMMARY"):
    oc = MagicMock()
    oc.create_text_response = AsyncMock(return_value=summary)
    return oc


@pytest.mark.asyncio
async def test_write_thread_summary_rolls_and_preserves_refs(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    proc = _Proc(db=temp_db, openai_client=_mock_openai("first summary"))
    state = ThreadState(thread_ts="100.0", channel_id="C1")

    dropped1 = [
        {"role": "user", "content": "see https://files.example/report.pdf",
         "metadata": {"ts": "101.0"}},
        {"role": "assistant", "content": "noted",
         "metadata": {"ts": "102.0", "type": "image_generation", "url": "https://img/x.png"}},
    ]
    await proc._write_thread_summary(state, thread_key, dropped1)

    row = temp_db.get_thread_summary(thread_key)
    assert row["summary_text"] == "first summary"
    assert row["boundary_ts"] == "102.0"
    ref_values = {r["value"] for r in row["refs"]}
    assert "https://files.example/report.pdf" in ref_values
    assert "https://img/x.png" in ref_values

    # Rolling: second span extends the same row, keeps prior refs, advances boundary
    proc.openai_client = _mock_openai("second rolled summary")
    dropped2 = [{"role": "user", "content": "also https://files.example/deck.pptx",
                 "metadata": {"ts": "103.0"}}]
    await proc._write_thread_summary(state, thread_key, dropped2)
    row2 = temp_db.get_thread_summary(thread_key)
    assert row2["summary_text"] == "second rolled summary"
    assert row2["boundary_ts"] == "103.0"
    values2 = {r["value"] for r in row2["refs"]}
    assert {"https://files.example/report.pdf", "https://img/x.png",
            "https://files.example/deck.pptx"} <= values2

    # Live state got exactly ONE summary head, updated in place
    heads = [m for m in state.messages
             if (m.get("metadata") or {}).get("type") == "thread_summary"]
    assert len(heads) == 1
    assert "second rolled summary" in heads[0]["content"]


@pytest.mark.asyncio
async def test_compaction_is_chunky_to_target(temp_db):
    """One compaction pass must land at/below TOKEN_COMPACTION_TARGET — not a small
    per-turn trim (which would bust the prefix cache every turn)."""
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    proc = _Proc(db=temp_db, openai_client=_mock_openai())
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    for i in range(20):
        state.messages.append({"role": "user", "content": f"filler message {i}",
                               "metadata": {"ts": f"10{i}.0"}})

    counter = MagicMock()
    counter.count_thread_tokens = lambda msgs: 100 * len(msgs)
    proc.thread_manager._token_counter = counter

    with patch.object(config, "get_model_token_limit", return_value=1000), \
         patch.object(config, "token_compaction_target", 0.7):
        processed = await proc._compact_thread_to_target(state, thread_key)

    assert processed > 0
    # target = 700 tokens = 7 messages (incl. the inserted summary head)
    assert counter.count_thread_tokens(state.messages) <= 700
    assert temp_db.get_thread_summary(thread_key) is not None


@pytest.mark.asyncio
async def test_summarizer_failure_uses_deterministic_fallback(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    oc = MagicMock()
    oc.create_text_response = AsyncMock(side_effect=RuntimeError("api down"))
    proc = _Proc(db=temp_db, openai_client=oc)
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    await proc._write_thread_summary(state, thread_key, [
        {"role": "user", "content": "hello", "metadata": {"ts": "101.0"}}])
    row = temp_db.get_thread_summary(thread_key)
    assert row["summary_text"] == "(Earlier messages were removed to manage context length.)"


# ----------------------------------------------------------------- migration

def test_mirror_drop_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager

    db = DatabaseManager(platform="slack")
    # Simulate a genuinely legacy database: messages table + old documents shape
    # (content column; its dead summary column is dropped by Phase S and the
    # LOAD-BEARING summary column is (re)created by the D2 migration).
    db.conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, role TEXT,
            content TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_ts TEXT, metadata_json TEXT)
    """)
    db.conn.execute("INSERT INTO messages (thread_id, role, content) VALUES ('C1:1','user','hi')")
    db.conn.execute("ALTER TABLE documents ADD COLUMN content TEXT")
    db.conn.close()

    # Restart: migration runs
    db2 = DatabaseManager(platform="slack")
    tables = {r[0] for r in db2.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "messages" not in tables
    assert "thread_summaries" in tables
    doc_cols = [c[1] for c in db2.conn.execute("PRAGMA table_info(documents)")]
    assert "content" not in doc_cols  # D2: no document content at rest
    assert "summary" in doc_cols      # D2's load-bearing summary column
    backups = os.listdir(str(tmp_path / "backups"))
    assert any("pre-v3-mirror-drop" in b for b in backups)

    # Idempotent: another restart takes no new backup
    db2.conn.close()
    db3 = DatabaseManager(platform="slack")
    assert len(os.listdir(str(tmp_path / "backups"))) == len(backups)
    db3.conn.close()


# ------------------------------------------------- image injection via metadata ts

@pytest.mark.asyncio
async def test_image_injection_uses_message_ts_metadata(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_image_metadata(thread_id=thread_key, url="https://img/cat.png",
                                image_type="uploaded", prompt="look at this",
                                analysis="A tabby cat on a desk.", message_ts="101.0")
    proc = _Proc(db=temp_db)
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    messages = [
        {"role": "user", "content": "look at this", "metadata": {"ts": "101.0"}},
        {"role": "assistant", "content": "Nice cat!", "metadata": {"ts": "102.0"}},
    ]
    enhanced = await _inject(proc, messages, state)
    assert len(enhanced) == 3
    assert enhanced[1]["role"] == "developer"
    assert "A tabby cat on a desk." in enhanced[1]["content"]


# ------------------------------------------- determinism (prompt-cache hygiene)

@pytest.mark.asyncio
async def test_two_rebuilds_serialize_identically(temp_db):
    """Prompt-cache hygiene: identical Slack fixtures must produce byte-identical
    context on every rebuild (covers name prefixes, reactions annotations, summary
    head, and image injection ordering)."""
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Stable summary.", "100.5",
                                refs=[{"kind": "link", "value": "https://x", "name": None}])
    temp_db.save_image_metadata(thread_id=thread_key, url="https://img/a.png",
                                image_type="uploaded", analysis="An image.",
                                message_ts="101.0")

    def fixtures():
        return [
            _hist("101.0", "check this image",
                  reactions=[{"name": "joy", "users": ["U2", "U1"], "count": 2},
                             {"name": "eyes", "users": ["U3"], "count": 1}]),
            _hist("102.0", "my reply", user="UBOT", sender="self"),
        ]

    async def one_rebuild():
        proc = _Proc(db=temp_db)
        state = await proc._get_or_rebuild_thread_state(_incoming(), _client_with_history(fixtures()))
        injected = await _inject(proc, state.messages, state)
        return json.dumps(injected, sort_keys=True)

    assert await one_rebuild() == await one_rebuild()


def test_reactions_annotation_deterministic_and_sorted():
    r_a = [{"name": "joy", "users": ["U2", "U1"], "count": 2},
           {"name": "eyes", "users": ["U3"], "count": 1}]
    r_b = [{"name": "eyes", "users": ["U3"], "count": 1},
           {"name": "joy", "users": ["U1", "U2"], "count": 2}]
    out_a = ThreadManagementMixin._render_reactions_annotation(r_a)
    out_b = ThreadManagementMixin._render_reactions_annotation(r_b)
    assert out_a == out_b
    assert out_a == "[reactions: :eyes: x1 (<@U3>); :joy: x2 (<@U1>, <@U2>)]"
    assert ThreadManagementMixin._render_reactions_annotation(None) == ""
    assert ThreadManagementMixin._render_reactions_annotation([]) == ""


@pytest.mark.asyncio
async def test_rebuild_includes_reactions_annotation(temp_db):
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "great news",
                     reactions=[{"name": "tada", "users": ["U9"], "count": 1}])]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client_with_history(history))
    joined = json.dumps([m.get("content") for m in state.messages])
    assert ":tada: x1 (<@U9>)" in joined


# --------------------------------------------- usage-driven token budgeting

def test_estimator_is_chars_over_four_no_tiktoken():
    import token_counter as tc
    assert not hasattr(tc, "tiktoken")
    counter = tc.TokenCounter("gpt-5.5")
    assert counter.count_tokens("x" * 400) == 100
    msg = {"role": "user", "content": "x" * 400}
    assert counter.count_message_tokens(msg) == 100 + 4 + 1  # content + overhead + role
    assert counter.count_thread_tokens([msg, msg]) == 2 * 105 + 3


def test_thread_state_usage_tracking():
    state = ThreadState(thread_ts="1.0", channel_id="C1")
    assert state.context_tokens == 0

    # Estimates accumulate as messages are added
    state.add_message("user", "x" * 400)
    est_after_one = state.context_tokens
    assert est_after_one > 0
    state.add_message("assistant", "y" * 400)
    assert state.context_tokens > est_after_one

    # The API's usage number REPLACES the estimate
    state.record_usage(input_tokens=5000, output_tokens=300)
    assert state.context_tokens == 5300

    # Next message increments on top of the authoritative number
    state.add_message("user", "z" * 400)
    assert state.context_tokens > 5300

    # Zero/None usage never wipes the tracked number
    state.record_usage(0, 0)
    assert state.context_tokens > 5300


@pytest.mark.asyncio
async def test_cleanup_trigger_uses_tracked_usage(temp_db):
    """The compaction trigger reads thread_state.context_tokens (usage-driven),
    not a recount of the messages."""
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    proc = _Proc(db=temp_db, openai_client=_mock_openai())
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    for i in range(10):
        # ~100 estimated tokens per message so the estimator agrees content exists
        state.messages.append({"role": "user", "content": f"msg {i} " + "x" * 390,
                               "metadata": {"ts": f"10{i}.0"}})

    with patch.object(config, "get_model_token_limit", return_value=1000), \
         patch.object(config, "token_cleanup_threshold", 0.9), \
         patch.object(config, "token_compaction_target", 0.7):
        # Tracked usage (authoritative) says we're tiny -> NO compaction, even
        # though a recount of the messages would say ~1000 tokens. This is the
        # trigger reading the tracked number, not recounting.
        state.context_tokens = 100
        await proc._async_post_response_cleanup(state, thread_key)
        assert temp_db.get_thread_summary(thread_key) is None
        assert len(state.messages) == 10

        # Tracked usage over threshold -> compaction runs and writes the summary
        state.context_tokens = 950
        await proc._async_post_response_cleanup(state, thread_key)
        assert temp_db.get_thread_summary(thread_key) is not None
        assert len(state.messages) < 10


def test_capture_usage_helper():
    from openai_client.api.responses import _capture_usage

    class _U:
        input_tokens = 1234
        output_tokens = 56

    class _R:
        usage = _U()

    sink = {}
    _capture_usage(sink, _R())
    assert sink == {"input_tokens": 1234, "output_tokens": 56}

    # None sink / response / usage are all safe no-ops
    _capture_usage(None, _R())
    sink2 = {}
    _capture_usage(sink2, None)
    _capture_usage(sink2, type("X", (), {"usage": None})())
    assert sink2 == {}


def test_context_length_error_detection():
    is_err = ThreadManagementMixin._is_context_length_error
    assert is_err(Exception("Error code: 400 - context_length_exceeded"))
    assert is_err(Exception("This model's maximum context length is 400000 tokens"))
    assert is_err(Exception("Input exceeds the context window of this model"))
    assert not is_err(Exception("rate_limit_exceeded"))
    assert not is_err(Exception("timeout"))


@pytest.mark.asyncio
async def test_compaction_rebaselines_tracked_estimate(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    proc = _Proc(db=temp_db, openai_client=_mock_openai())
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    for i in range(20):
        state.messages.append({"role": "user", "content": "filler " * 50,
                               "metadata": {"ts": f"10{i}.0"}})
    state.context_tokens = 999_999  # stale huge number

    with patch.object(config, "get_model_token_limit", return_value=1000), \
         patch.object(config, "token_compaction_target", 0.7):
        await proc._compact_thread_to_target(state, thread_key)

    # Tracked number was re-baselined from the compacted messages, not left stale
    assert state.context_tokens < 999_999
    assert state.context_tokens == proc.thread_manager._token_counter.count_thread_tokens(state.messages)


# ------------------------------------ system prompt: date-only prefix, time suffix

def _sys_prompt(proc, **kw):
    client = MagicMock()
    client.name = "slack"
    defaults = dict(user_timezone="UTC", model="gpt-5.5",
                    web_search_enabled=False, has_trimmed_messages=False)
    defaults.update(kw)
    return proc._get_system_prompt(client, defaults["user_timezone"], None, None, None,
                                   defaults["model"], defaults["web_search_enabled"],
                                   defaults["has_trimmed_messages"], None)


def test_system_prompt_is_date_only_no_minutes(temp_db):
    import re
    proc = _Proc(db=temp_db)
    prompt = _sys_prompt(proc)
    assert "Today's date:" in prompt
    assert not re.search(r"\d{1,2}:\d{2}", prompt), "minute-precision time busts the prefix cache"


def test_time_suffix_carries_minutes(temp_db):
    import re
    proc = _Proc(db=temp_db)
    suffix = proc._build_time_suffix_context("UTC")
    assert re.search(r"\d{1,2}:\d{2} [AP]M", suffix)
    assert suffix.startswith("[Current date and time:")


def test_summary_note_wording_is_stable(temp_db):
    proc = _Proc(db=temp_db)
    with_note = _sys_prompt(proc, has_trimmed_messages=True)
    without = _sys_prompt(proc, has_trimmed_messages=False)
    assert "has been summarized in a summary message above" in with_note
    assert "summarized" not in without
    # deterministic: no counts or timestamps in the note
    note = with_note.replace(without[:without.index("Today's date:")], "")
    assert "Note: The beginning of this conversation has been summarized" in with_note


# ----------------------------------------------------------- prompt_cache_key

@pytest.mark.asyncio
async def test_prompt_cache_key_passed_for_gpt55():
    from openai_client.api import responses as responses_api

    captured = {}

    class _FakeClient:
        pass

    class _Self:
        client = MagicMock()
        def log_debug(self, *a, **k): pass
        def log_info(self, *a, **k): pass
        def log_error(self, *a, **k): pass
        async def _safe_api_call(self, fn, operation_type=None, **params):
            captured.update(params)
            resp = MagicMock()
            resp.output = []
            return resp

    await responses_api.create_text_response(
        _Self(), messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.5", prompt_cache_key="C1:100.0")
    assert captured.get("prompt_cache_key") == "C1:100.0"
    assert captured.get("prompt_cache_retention") == "24h"

    captured.clear()
    await responses_api.create_text_response(
        _Self(), messages=[{"role": "user", "content": "hi"}],
        model="gpt-5-mini")
    assert "prompt_cache_key" not in captured
    assert "prompt_cache_retention" not in captured
