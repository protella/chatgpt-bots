"""Unit tests for F2 — explicit no-reply terminal-action contract.

Covers the tool-exposure gate (silence-capable routes only, config-off), the
once-materialized request config (shared dict never mutated), the tool-loop terminal
semantics (ends the turn's WORDS while every sibling still executes, and requires one of the
eight declared reasons — see `message_processor/terminal_actions.py` and
tests/unit/test_terminal_actions.py), and the F2-revision committed-text contract for
streaming (no_response_needed honored only while no visible text has streamed;
rejected-and-completed once a reply has begun).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import config
from tool_registry import ToolRegistry


def _no_reply_schema():
    return {
        "type": "function",
        "name": "no_response_needed",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]},
    }


def _registry_with_no_reply():
    reg = ToolRegistry()
    reg.register(
        _no_reply_schema(), AsyncMock(return_value={"ok": True}),
        enabled=lambda cfg: config.enable_no_reply_tool and bool(cfg.get("_silence_capable_turn")))
    return reg


# --------------------------------------------------------------------- exposure gate

def test_no_reply_hidden_without_silence_capable_flag():
    reg = _registry_with_no_reply()
    assert reg.schemas({}) == []
    assert reg.schemas({"_silence_capable_turn": True})[0]["name"] == "no_response_needed"


def test_no_reply_hidden_when_config_off(monkeypatch):
    monkeypatch.setattr(config, "enable_no_reply_tool", False)
    reg = _registry_with_no_reply()
    assert reg.schemas({"_silence_capable_turn": True}) == []


# --------------------------------------------------- _materialize_request_tools

class _MatHost:
    def __init__(self, registry):
        from message_processor.handlers.text import TextHandlerMixin
        for n in ("_materialize_request_tools", "_get_tool_registry"):
            setattr(self, n, getattr(TextHandlerMixin, n).__get__(self))
        self._client = SimpleNamespace(tool_registry=registry)


def _msg(gate_required=False, silence_capable=False, wake_source=None,
         routing_posture="channel_activity"):
    # The routing facts as a dispatcher stamps them: present on every message, and False on
    # the addressed routes rather than absent.
    md = {"ts": "1.1", "gate_required": gate_required, "silence_capable": silence_capable,
          "routing_posture": routing_posture}
    if wake_source:
        md["wake_source"] = wake_source
    return SimpleNamespace(metadata=md, channel_id="C1")


def test_materialize_gated_turn_exposes_tool_without_mutating_shared(mock_env):
    from prompts import CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    host = _MatHost(_registry_with_no_reply())
    shared = {"model": "gpt-5"}
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, shared, _msg(gate_required=True, silence_capable=True), tools_disabled=False)
    assert available is True
    assert suffix == CHANNEL_ACTIVITY_NO_REPLY_SUFFIX  # channel-activity wording
    assert request_config["_silence_capable_turn"] is True
    assert "_silence_capable_turn" not in shared  # copied, never mutated
    assert registry is not None


def test_materialize_prompted_turn_no_tool(mock_env):
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5"}, _msg(), tools_disabled=False)
    assert available is False
    assert suffix is None
    assert "_silence_capable_turn" not in request_config


def test_materialize_timeout_retry_drops_tool_and_paragraph(mock_env):
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5"}, _msg(gate_required=True, silence_capable=True),
        tools_disabled=True)
    # Retry disables the registry — so the tool AND the suffix paragraph fall away.
    assert registry is None
    assert available is False
    assert suffix is None


# ------------------------------------------------- F18 continuation exposure

def test_materialize_continuation_exposes_tool_with_thread_suffix(mock_env):
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX)
    host = _MatHost(_registry_with_no_reply())
    shared = {"model": "gpt-5"}
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, shared, _msg(silence_capable=True, wake_source="thread_continuation",
                                   routing_posture="thread_activity"),
        tools_disabled=False)
    assert available is True
    assert request_config["_silence_capable_turn"] is True  # drives the tool's enabled gate
    assert "_silence_capable_turn" not in shared  # copied, never mutated
    assert registry is not None
    # Thread wording on a thread reply — NOT the channel-activity paragraph.
    assert suffix == THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert suffix != CHANNEL_ACTIVITY_NO_REPLY_SUFFIX


def test_materialize_continuation_hidden_when_config_off(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_no_reply_tool", False)
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5"},
        _msg(silence_capable=True, wake_source="thread_continuation",
             routing_posture="thread_activity"), tools_disabled=False)
    assert available is False
    assert suffix is None


def test_materialize_mention_wake_no_tool(mock_env):
    """@-mention / name-summons turns (wake_source name_mention/app_mention) are genuinely
    addressed — no silence option, no suffix, tool hidden."""
    host = _MatHost(_registry_with_no_reply())
    for src in ("app_mention", "name_mention", "dm", "ambient"):
        registry, request_config, available, suffix = host._materialize_request_tools(
            host._client, {"model": "gpt-5"}, _msg(wake_source=src), tools_disabled=False)
        assert available is False, src
        assert suffix is None, src
        assert "_silence_capable_turn" not in request_config, src


def test_materialize_continuation_timeout_retry_drops_tool_and_paragraph(mock_env):
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5"},
        _msg(silence_capable=True, wake_source="thread_continuation"), tools_disabled=True)
    assert registry is None
    assert available is False
    assert suffix is None


# ------------------------------------------------------------- tool-loop terminal

class _LoopSelf:
    def log_info(self, *a, **k):
        pass

    log_warning = log_debug = log_error = log_info


class _FakeRegistry:
    """Records dispatched call names; returns ok for each."""
    def __init__(self, dispatched):
        self.dispatched = dispatched

    async def dispatch_all(self, ctx, calls):
        results = []
        for c in calls:
            self.dispatched.append(c.get("name"))
            results.append({"ok": True})
        return results


def _fc(name, call_id, arguments="{}"):
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


@pytest.mark.asyncio
async def test_tool_loop_no_reply_runs_every_sibling(monkeypatch):
    from openai_client.api import tool_loop
    scripts = [[
        _fc("no_response_needed", "1", '{"reason": "addressed_to_other"}'),
        _fc("react_to_message", "2", '{"emoji": "eyes"}'),
        _fc("remember_fact", "3", "{}"),
    ]]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        function_call_sink.extend(scripts.pop(0))
        return {"text": "should be suppressed", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)

    assert result["terminal_action"] == "no_reply"
    assert result["silence_reason"] == "addressed_to_other"
    assert result["text"] == ""
    # EVERY sibling runs. Silence ends the words, not the round: the memory write the model
    # chose is a decision it made, and dropping it was us overruling that decision silently.
    assert set(dispatched) == {"no_response_needed", "react_to_message", "remember_fact"}
    assert [c["name"] for c in result["local_tool_calls"]] == [
        "react_to_message", "remember_fact"]
    assert all(c["ok"] for c in result["local_tool_calls"])


@pytest.mark.asyncio
async def test_terminal_round_reacts_respect_global_cap(monkeypatch):
    # F6 fix (b) + F4 fix: react siblings in a no_reply terminal round still count
    # against MAX_TOOL_CALLS_PER_TURN, AND the terminal no_response_needed call itself
    # consumes one slot — so reacts are capped at remaining_budget - 1. With a cap of 2
    # that means terminal(1) + exactly 1 react = 2 total; the round never exceeds the cap.
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.config, "max_tool_calls_per_turn", 2)
    scripts = [[
        _fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
        _fc("react_to_message", "2", '{"emoji": "eyes"}'),
        _fc("react_to_message", "3", '{"emoji": "tada"}'),
        _fc("react_to_message", "4", '{"emoji": "thumbsup"}'),
    ]]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        function_call_sink.extend(scripts.pop(0))
        return {"text": "", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)
    assert result["terminal_action"] == "no_reply"
    # no_response_needed reserves one slot; only 1 sibling fits under the cap of 2.
    assert dispatched.count("react_to_message") == 1
    assert dispatched.count("no_response_needed") == 1


@pytest.mark.asyncio
async def test_terminal_round_suppresses_duplicate_no_reply(monkeypatch):
    # F4 fix: when the model emits multiple no_response_needed calls in one terminal
    # round, only the FIRST is dispatched (first wins); the duplicates are suppressed.
    from openai_client.api import tool_loop
    scripts = [[
        _fc("no_response_needed", "1", '{"reason": "not_relevant"}'),
        _fc("no_response_needed", "2", '{"reason": "duplicate"}'),
        _fc("react_to_message", "3", '{"emoji": "eyes"}'),
    ]]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        function_call_sink.extend(scripts.pop(0))
        return {"text": "", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)
    assert result["terminal_action"] == "no_reply"
    assert result["silence_reason"] == "not_relevant"  # first wins
    assert dispatched.count("no_response_needed") == 1  # duplicate suppressed
    assert dispatched.count("react_to_message") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    '{"reason": "it is not really for me"}',   # prose, the old contract
    '{"reason": ""}',                          # empty
    '{}',                                      # absent
    '{"reason": 7}',                           # not a string
    'not json at all',                         # unparseable
])
async def test_an_unrecognized_reason_is_rejected_never_coerced(monkeypatch, arguments):
    """The turn is NOT silenced and the reason is NOT rewritten to `other`. We do not know why
    the model wants to be quiet, and inventing a value would file our guess as its testimony."""
    from openai_client.api import tool_loop
    scripts = [[_fc("no_response_needed", "1", arguments)], []]
    state = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(scripts[min(i, len(scripts) - 1)])
        return {"text": "" if i == 0 else "fine, here you go", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)
    assert result.get("terminal_action") is None      # not silenced
    assert result.get("silence_reason") is None       # and nothing invented
    assert result["text"] == "fine, here you go"      # the model got another round
    assert "no_response_needed" not in dispatched     # rejected, never executed
    rejected = [c for c in result["local_tool_calls"] if c["name"] == "no_response_needed"]
    assert rejected and all(c["ok"] is False for c in rejected)


@pytest.mark.asyncio
async def test_an_invalid_terminal_still_runs_its_siblings_exactly_once(monkeypatch):
    from openai_client.api import tool_loop
    scripts = [[_fc("no_response_needed", "1", '{"reason": "made up"}'),
                _fc("react_to_message", "2", '{"emoji": "eyes"}')], []]
    state = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(scripts[min(i, len(scripts) - 1)])
        return {"text": "" if i == 0 else "done", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)
    assert dispatched.count("react_to_message") == 1   # ran, and only once
    assert result.get("terminal_action") is None


# ---------------------------------------------- F2 revision: streaming committed-text

class _RecordingSelf(_LoopSelf):
    def __init__(self):
        self.warnings = []

    def log_warning(self, msg, *a, **k):
        self.warnings.append(str(msg))


def _streaming_fake(rounds):
    """Fake create_streaming_response_with_tools driven by a list of (text, calls)."""
    state = {"n": 0}

    async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                             function_call_sink=None, tool_choice=None, **params):
        text, calls = rounds[min(state["n"], len(rounds) - 1)]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(calls)
        return text

    return fake_streaming


@pytest.mark.asyncio
async def test_streaming_no_reply_honored_when_no_text_committed(monkeypatch):
    # Round 1 calls no_response_needed with NO streamed text → silence is honored.
    from openai_client.api import tool_loop
    rounds = [("", [_fc("no_response_needed", "1", '{"reason": "addressed_to_other"}')])]
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _streaming_fake(rounds))
    out = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry([]),
        tool_context=None, stream_callback=lambda c: None)
    assert out["terminal_action"] == "no_reply"
    assert out["silence_reason"] == "addressed_to_other"
    assert out["text"] == ""


@pytest.mark.asyncio
async def test_streaming_no_reply_rejected_after_committed_text(monkeypatch):
    # Round 1 streams a visible preamble AND calls no_response_needed → INVALID: the call is
    # rejected, the loop continues, and round 2 completes the reply. WARNING logged.
    from openai_client.api import tool_loop
    rounds = [
        ("Sure, here's the answer", [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}')]),
        (" — all done.", []),
    ]
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _streaming_fake(rounds))
    host = _RecordingSelf()
    out = await tool_loop.create_streaming_response_with_tool_loop(
        host, messages=[], tools=[], registry=_FakeRegistry([]),
        tool_context=None, stream_callback=lambda c: None)
    assert out.get("terminal_action") is None          # NOT silenced
    # Default (non-aggregate) loop returns the completing round only; the chat handler, which
    # opts into aggregate_segments, is where the committed preamble is preserved on screen.
    assert out["text"] == " — all done."               # completing round's reply
    assert {"name": "no_response_needed", "ok": False, "gist": "reason=<str>"} in out["local_tool_calls"]
    assert any("after visible text" in w for w in host.warnings)


@pytest.mark.asyncio
async def test_streaming_no_reply_with_committed_text_runs_react_sibling(monkeypatch):
    # A react sibling in the rejected round still executes; the reply then completes.
    from openai_client.api import tool_loop
    rounds = [
        ("partial", [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
                     _fc("react_to_message", "2", '{"emoji": "eyes"}')]),
        ("done", []),
    ]
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _streaming_fake(rounds))
    dispatched = []
    out = await tool_loop.create_streaming_response_with_tool_loop(
        _RecordingSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None, stream_callback=lambda c: None)
    assert out["text"] == "done"                       # default loop: completing round only
    assert "react_to_message" in dispatched            # sibling ran
    assert "no_response_needed" not in dispatched       # rejected via override, not dispatched


# ------------------------------------------------- F8: cross-attempt committed text

@pytest.mark.asyncio
async def test_streaming_no_reply_rejected_when_prior_committed(monkeypatch):
    # F8: prior_committed seeds the committed-text signal, so even a fresh stream that
    # shows NO text this attempt rejects a no_response_needed — a prior attempt's partial
    # reply must never be orphaned as fake silence.
    from openai_client.api import tool_loop
    rounds = [("", [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}')]),
              ("recovered reply", [])]
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _streaming_fake(rounds))
    out = await tool_loop.create_streaming_response_with_tool_loop(
        _RecordingSelf(), messages=[], tools=[], registry=_FakeRegistry([]),
        tool_context=None, stream_callback=lambda c: None, prior_committed=True)
    assert out.get("terminal_action") is None
    assert out["text"] == "recovered reply"


@pytest.mark.asyncio
async def test_nonstreaming_no_reply_rejected_when_prior_committed(monkeypatch):
    # F8: the non-streaming loop honors no_reply normally, but when prior_committed is set
    # (a streaming attempt already exposed text) it rejects and forces completion instead.
    from openai_client.api import tool_loop
    scripts = [[_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}')], []]
    texts = ["", "finished reply"]
    state = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(scripts[min(i, len(scripts) - 1)])
        return {"text": texts[min(i, len(texts) - 1)], "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    host = _RecordingSelf()
    dispatched = []
    out = await tool_loop.create_text_response_with_tool_loop(
        host, messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None, prior_committed=True)
    assert out.get("terminal_action") is None
    assert out["text"] == "finished reply"
    assert "no_response_needed" not in dispatched   # rejected via override, not dispatched
    assert any("prior attempt" in w for w in host.warnings)


@pytest.mark.asyncio
async def test_run_tool_round_override_is_identity_not_call_id():
    # A degenerate round where a sibling shares the terminal's call_id must NOT misroute the
    # override — identity keying short-circuits only the specific terminal object.
    from openai_client.api import tool_loop
    terminal = _fc("no_response_needed", "dup", '{"reason": "nothing_to_add"}')
    sibling = _fc("react_to_message", "dup", '{"emoji": "eyes"}')  # same call_id on purpose
    sink = [terminal, sibling]
    local, dispatched = [], []
    await tool_loop._run_tool_round(
        _LoopSelf(), _FakeRegistry(dispatched), None, sink, [], local,
        result_overrides={id(terminal): tool_loop._INVALID_NO_REPLY_RESULT})
    assert dispatched == ["react_to_message"]                  # only the sibling dispatched
    assert local[0]["name"] == "no_response_needed" and local[0]["ok"] is False
    assert local[1]["name"] == "react_to_message" and local[1]["ok"] is True


# ------------------------------------------------- silent-stream cleanup (F2/F8)

def _cleanup_host():
    from message_processor.handlers.text import TextHandlerMixin
    host = SimpleNamespace(warnings=[], debugs=[])
    host._cleanup_silent_stream = TextHandlerMixin._cleanup_silent_stream.__get__(host)
    host.log_warning = lambda m, *a, **k: host.warnings.append(str(m))
    host.log_debug = lambda m, *a, **k: host.debugs.append(str(m))
    return host


@pytest.mark.asyncio
async def test_cleanup_silent_stream_deletes_both_distinct_messages():
    # A no_reply/reaction-only teardown must delete EVERY distinct message we created —
    # the original placeholder AND the stream/seed — not just current_message_id.
    host = _cleanup_host()
    deleted = []

    async def _del(ch, ts):
        deleted.append(ts)
        return True
    client = SimpleNamespace(delete_message=AsyncMock(side_effect=_del))
    native = SimpleNamespace(started=True, abandon=AsyncMock(return_value=True))
    await host._cleanup_silent_stream(client, "C1", native, "placeholder.1", "stream.2", "no_reply")
    native.abandon.assert_awaited_once()
    assert set(deleted) == {"placeholder.1", "stream.2"}


@pytest.mark.asyncio
async def test_cleanup_silent_stream_reports_abandon_failure_and_skips_none():
    host = _cleanup_host()
    deleted = []

    async def _del(ch, ts):
        deleted.append(ts)
        return True
    client = SimpleNamespace(delete_message=AsyncMock(side_effect=_del))
    native = SimpleNamespace(started=True, abandon=AsyncMock(return_value=False))  # stop failed
    # message_id None (status-only DM) → only the stream message is deleted, no None delete.
    await host._cleanup_silent_stream(client, "C1", native, None, "stream.2", "reaction-only")
    assert deleted == ["stream.2"]
    assert any("abandon failed" in w for w in host.warnings)


# --------------------------------------------------------------- contract paragraph

def test_channel_activity_paragraph_wording():
    from prompts import CHANNEL_ACTIVITY_NO_REPLY_SUFFIX as s
    # Silence is the default and needs no apology — the old "you joined uninvited" framing
    # asked the model to feel like an intruder rather than to judge whether it adds anything.
    assert "uninvited" not in s
    assert "Silence is the DEFAULT here" in s
    assert "could not easily get themselves" in s
    assert "no_response_needed" in s
    # C1: the mid-flight escape — an answer that would only admit a limitation is silence,
    # but a substantive answer isn't suppressed for including one, and a name-summoned turn
    # prefers a brief honest answer.
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "addressed by name, prefer a brief honest answer over silence" in s
    # The terminal contract the schema also states: silence ends the words, not the round.
    assert "ends your words, not your other actions" in s
    assert "wait for work you started yourself" in s
    assert s.startswith("[") and s.endswith("]")


def test_thread_activity_paragraph_wording():
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX as s)
    # Thread wording keeps the sticky-addressee rule: check the addressee yourself, the
    # hand-off sticks, and silence means silence (never a "staying quiet" placeholder).
    assert "check its addressee yourself" in s
    assert "STICKS" in s
    assert "silence means silence" in s
    assert "no_response_needed" in s
    assert "ends your words, not your other actions" in s
    assert "wait for work you started yourself" in s
    assert s != CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert s.startswith("[") and s.endswith("]")


def test_neither_paragraph_restates_the_silence_vocabulary():
    """The eight values live in the tool schema. Two copies of one vocabulary drift.

    Checked on the identifier-shaped values only — `other` is also an ordinary English word,
    and "another person" is prose, not a restatement of the enum."""
    from message_processor.terminal_actions import SILENCE_REASONS
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX)
    identifiers = [v for v in SILENCE_REASONS if "_" in v]
    for paragraph in (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX, THREAD_ACTIVITY_NO_REPLY_SUFFIX):
        assert not [v for v in identifiers if v in paragraph]
