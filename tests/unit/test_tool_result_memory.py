"""F12 — tool-result memory for MCP calls (extends F7).

Covers the pure helpers (digest build with per-call + per-turn truncation, the
result-results render, the combined pinned-order render, and the used-tools line NOT
listing result entries), the mcp_call output capture on both Responses paths, the DB
merge preserving result_digest as a distinct class, deterministic rebuild reinjection in
the strip → used-tools → tool-results → reactions order, old rows rendering as today,
config-off, and the prompt instruction.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from base_client import Message
from config import config
from message_processor import tool_provenance as tp
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from message_processor.handlers.vision import VisionHandlerMixin
from thread_manager import AsyncThreadStateManager


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, VisionHandlerMixin, MessageUtilitiesMixin):
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
def _flags_on(monkeypatch):
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    monkeypatch.setattr(config, "enable_tool_result_memory", True)
    monkeypatch.setattr(config, "tool_result_digest_chars", 2000)
    monkeypatch.setattr(config, "tool_result_turn_chars", 6000)


# ------------------------------------------------------------------ pure helpers

def test_build_result_digests_basic_and_newline_flattened():
    out = tp.build_result_digests(
        [{"tool_name": "reportpro", "output": "Ice Cream\np.25 link=x"}], 2000, 6000)
    # newlines flattened to spaces (one annotation line; no injected [reactions:] lines)
    assert out == [{"tool_name": "reportpro", "result_digest": "Ice Cream p.25 link=x"}]


def test_build_result_digests_per_call_truncation():
    out = tp.build_result_digests(
        [{"tool_name": "srv", "output": "y" * 5000}], per_call_chars=100, per_turn_chars=6000)
    digest = out[0]["result_digest"]
    assert digest == "y" * 100 + tp.TRUNCATION_MARKER
    assert digest.endswith("… [truncated]")


def test_build_result_digests_per_turn_budget_drops_later_calls():
    # First two calls fill the 100-char turn budget; the third stores nothing.
    results = [
        {"tool_name": "a", "output": "a" * 60},
        {"tool_name": "b", "output": "b" * 60},
        {"tool_name": "c", "output": "c" * 60},
    ]
    out = tp.build_result_digests(results, per_call_chars=2000, per_turn_chars=100)
    names = [e["tool_name"] for e in out]
    assert names == ["a", "b"]  # 'c' arrives after the 100-char cap is spent


def test_build_result_digests_skips_empty_and_nameless():
    out = tp.build_result_digests(
        [{"tool_name": "s", "output": ""}, {"tool_name": "", "output": "x"},
         {"tool_name": "s", "output": "   "}], 2000, 6000)
    assert out == []


def test_used_tools_line_excludes_result_entries():
    tools = [
        {"tool_name": "reportpro", "gist": ""},                       # used-tools entry
        {"tool_name": "reportpro", "result_digest": "Ice Cream p.25"},  # F12 result entry
    ]
    # The [used tools:] line lists the server ONCE (from the used-tools entry), never the
    # result entry — otherwise the digest'd server would double-list.
    assert tp.render_used_tools_annotation(tools) == "[used tools: reportpro]"


def test_render_tool_results_annotation_one_line_per_digest():
    tools = [
        {"tool_name": "reportpro", "gist": ""},
        {"tool_name": "reportpro", "result_digest": "Ice Cream p.25"},
        {"tool_name": "srv2", "result_digest": "another result"},
    ]
    assert tp.render_tool_results_annotation(tools) == (
        "[tool results: reportpro → Ice Cream p.25]\n"
        "[tool results: srv2 → another result]")
    # No digests → empty (old F7 rows).
    assert tp.render_tool_results_annotation([{"tool_name": "a", "gist": "limit=5"}]) == ""


def test_render_provenance_annotations_pinned_order():
    tools = [
        {"tool_name": "reportpro", "gist": ""},
        {"tool_name": "reportpro", "result_digest": "Ice Cream p.25"},
    ]
    assert tp.render_provenance_annotations(tools) == (
        "[used tools: reportpro]\n[tool results: reportpro → Ice Cream p.25]")


def test_render_provenance_old_rows_render_as_today():
    # A pre-F12 row (names/gists only) renders exactly the used-tools line, no results block.
    tools = [{"tool_name": "fetch_channel_history", "gist": "limit=50"}]
    assert tp.render_provenance_annotations(tools) == \
        "[used tools: fetch_channel_history(limit=50)]"


# --------------------------------------------------------- capture on Responses paths

def _mcp_item(server_label, output, error=None):
    return SimpleNamespace(type="mcp_call", server_label=server_label,
                           output=output, error=error, content=None)


def test_capture_mcp_result_helper_streaming_and_nonstreaming_shape():
    from openai_client.api.responses import _capture_mcp_result
    sink = []
    _capture_mcp_result(sink, _mcp_item("reportpro", "Ice Cream link=x"), "reportpro")
    assert sink == [{"tool_name": "reportpro", "output": "Ice Cream link=x"}]
    # errored / empty / no-sink calls capture nothing
    _capture_mcp_result(sink, _mcp_item("reportpro", "boom", error={"code": 1}), "reportpro")
    _capture_mcp_result(sink, _mcp_item("reportpro", None), "reportpro")
    _capture_mcp_result(None, _mcp_item("reportpro", "x"), "reportpro")
    assert sink == [{"tool_name": "reportpro", "output": "Ice Cream link=x"}]
    # missing server_label falls back to "mcp"
    _capture_mcp_result(sink, _mcp_item(None, "y"), None)
    assert sink[-1] == {"tool_name": "mcp", "output": "y"}


@pytest.mark.asyncio
async def test_capture_nonstreaming_path_collects_output():
    """create_text_response_with_tools threads a completed mcp_call's output into the sink
    (exercises the real non-streaming output loop, not just the helper)."""
    from openai_client.api import responses as R

    fake = MagicMock()
    fake.log_info = fake.log_debug = fake.log_warning = fake.log_error = lambda *a, **k: None
    resp = SimpleNamespace(output=[_mcp_item("reportpro", "Ice Cream p.25 link=x")],
                           usage=None)
    fake._safe_api_call = AsyncMock(return_value=resp)

    sink = []
    result = await R.create_text_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "mcp"}], model="gpt-5.6-sol",
        return_metadata=True, mcp_results_sink=sink)
    assert sink == [{"tool_name": "reportpro", "output": "Ice Cream p.25 link=x"}]
    assert "reportpro" in result["tools_used"]


@pytest.mark.asyncio
async def test_capture_streaming_path_collects_output():
    """create_streaming_response_with_tools captures a completed mcp_call's output from the
    response.output_item.done event into the sink."""
    from openai_client.api import responses as R

    fake = MagicMock()
    fake.log_info = fake.log_debug = fake.log_warning = fake.log_error = lambda *a, **k: None
    fake._safe_api_call = AsyncMock(return_value=SimpleNamespace())

    events = [
        SimpleNamespace(type="response.output_item.done",
                        item=_mcp_item("reportpro", "Ice Cream p.25 link=x")),
        SimpleNamespace(type="response.completed", response=None),
    ]

    async def _iter(response, op):
        for e in events:
            yield e

    fake._safe_stream_iteration = _iter

    sink = []
    await R.create_streaming_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "mcp"}], stream_callback=lambda chunk: None,
        model="gpt-5.6-sol", mcp_results_sink=sink)
    assert sink == [{"tool_name": "reportpro", "output": "Ice Cream p.25 link=x"}]


# ------------------------------------------------------------------ DB layer

@pytest.mark.asyncio
async def test_db_roundtrip_preserves_result_digest(temp_db):
    tools = [{"tool_name": "reportpro", "gist": ""},
             {"tool_name": "reportpro", "result_digest": "Ice Cream p.25 link=x"}]
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", tools)
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": tools}


def test_merge_keeps_result_and_used_as_distinct_classes():
    from database import DatabaseManager as DM
    # A used-tools entry and a result entry for the SAME server never collapse; both kept,
    # used-tools first then results.
    out = DM._merge_tool_provenance(
        [{"tool_name": "srv", "gist": ""}],
        [{"tool_name": "srv", "result_digest": "R1"}])
    assert out == [{"tool_name": "srv", "gist": ""},
                   {"tool_name": "srv", "result_digest": "R1"}]


def test_merge_dedupes_result_by_name_and_digest_but_keeps_distinct_outputs():
    from database import DatabaseManager as DM
    out = DM._merge_tool_provenance(
        [{"tool_name": "srv", "result_digest": "R1"}],
        [{"tool_name": "srv", "result_digest": "R1"},   # exact dup → dropped
         {"tool_name": "srv", "result_digest": "R2"}])  # distinct output → kept
    assert out == [{"tool_name": "srv", "result_digest": "R1"},
                   {"tool_name": "srv", "result_digest": "R2"}]


def test_merge_result_entries_not_subject_to_used_tools_cap():
    from database import DatabaseManager as DM
    from config import config
    cap = config.tool_provenance_max_entries  # F14: env-backed (default 20)
    used = [{"tool_name": f"t{i}", "gist": ""} for i in range(cap + 5)]
    results = [{"tool_name": "srv", "result_digest": f"R{i}"} for i in range(5)]
    out = DM._merge_tool_provenance(used, results)
    used_out = [e for e in out if "result_digest" not in e]
    res_out = [e for e in out if "result_digest" in e]
    assert len(used_out) == cap  # used-tools capped at config.tool_provenance_max_entries
    assert len(res_out) == 5     # all char-bounded result digests survive


def test_merge_old_rows_unchanged():
    from database import DatabaseManager as DM
    # No result entries → behaves exactly like the F7 merge.
    out = DM._merge_tool_provenance(
        [{"tool_name": "fetch", "gist": ""}],
        [{"tool_name": "fetch", "gist": "limit=5"}, {"tool_name": "fetch", "gist": ""}])
    assert out == [{"tool_name": "fetch", "gist": "limit=5"}]


# ------------------------------------------------------------------ rebuild reinjection

@pytest.mark.asyncio
async def test_rebuild_renders_tool_results_block_after_used_tools(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0",
        [{"tool_name": "reportpro", "gist": ""},
         {"tool_name": "reportpro", "result_digest": "Ice Cream 2025-12-10 p.25 link=x"}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Found it.\n\n_Used Tools: reportpro_", sender="self",
                     reactions=[{"name": "eyes", "count": 1, "users": ["U9"]}])]
    # timestamps off for a clean assertion on the pinned suffix order
    from config import config as cfg
    orig = cfg.enable_message_timestamps
    cfg.enable_message_timestamps = False
    try:
        state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    finally:
        cfg.enable_message_timestamps = orig
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "_Used Tools:" not in content  # external footer stripped
    assert content == (
        "Found it.\n"
        "[used tools: reportpro]\n"
        "[tool results: reportpro → Ice Cream 2025-12-10 p.25 link=x]\n"
        "[reactions: :eyes: x1 (<@U9>)]")


@pytest.mark.asyncio
async def test_rebuild_is_deterministic_across_repeats(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0",
        [{"tool_name": "srv", "result_digest": "digest here"}])
    history = [_hist("101.0", "Answer.", sender="self")]
    first = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    second = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    a = next(m for m in first.messages if m["role"] == "assistant")["content"]
    b = next(m for m in second.messages if m["role"] == "assistant")["content"]
    assert a == b
    assert "[tool results: srv → digest here]" in a


@pytest.mark.asyncio
async def test_rebuild_old_row_renders_used_tools_only(temp_db):
    # A pre-F12 row (no result_digest) renders exactly as before — no tool-results block.
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.", sender="self")]
    from config import config as cfg
    orig = cfg.enable_message_timestamps
    cfg.enable_message_timestamps = False
    try:
        state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    finally:
        cfg.enable_message_timestamps = orig
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert content == "Answer.\n[used tools: web_search]"
    assert "tool results:" not in content


@pytest.mark.asyncio
async def test_rebuild_skips_result_block_behind_compaction_boundary(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Earlier.", "101.5", refs=[])
    await temp_db.save_tool_usage_async(
        "C1", "101.0", thread_key,
        [{"tool_name": "srv", "result_digest": "should never surface"}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "old", sender="self"), _hist("102.0", "fresh tail")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    joined = " ".join(str(m.get("content")) for m in state.messages)
    assert "tool results" not in joined
    assert "should never surface" not in joined


@pytest.mark.asyncio
async def test_footer_strip_fires_with_tool_results_block_present(temp_db):
    # F7-4 re-verify: the end-anchored footer strip still fires when BOTH annotation
    # blocks would follow it — the strip happens before annotations are appended, and the
    # anti-shielding lookahead only admits [used tools:]/[reactions:] lines, so the footer
    # can never be shielded.
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0",
        [{"tool_name": "reportpro", "gist": ""},
         {"tool_name": "reportpro", "result_digest": "kept"}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Body.\n\n_Used Tools: reportpro_", sender="self")]
    from config import config as cfg
    orig = cfg.enable_message_timestamps
    cfg.enable_message_timestamps = False
    try:
        state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    finally:
        cfg.enable_message_timestamps = orig
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "_Used Tools:" not in content
    assert content == ("Body.\n[used tools: reportpro]\n"
                       "[tool results: reportpro → kept]")


@pytest.mark.asyncio
async def test_rebuild_config_off_stores_no_results(temp_db, monkeypatch):
    # enable_tool_result_memory off: names-only provenance still renders, no results block.
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    monkeypatch.setattr(config, "enable_tool_result_memory", False)
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.", sender="self")]
    from config import config as cfg
    orig = cfg.enable_message_timestamps
    cfg.enable_message_timestamps = False
    try:
        state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    finally:
        cfg.enable_message_timestamps = orig
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    # rebuild still renders whatever is stored (render is flag-independent), but with the
    # flag off nothing writes result_digest rows in the first place — see capture tests.
    assert "[used tools: web_search]" in content


# ------------------------------------------------------------------ prompt instruction

def test_prompt_has_tool_results_trust_instruction():
    from prompts import SLACK_SYSTEM_PROMPT
    assert "[tool results:" in SLACK_SYSTEM_PROMPT
    # the retraction half is present
    assert "never retract" in SLACK_SYSTEM_PROMPT.lower()
