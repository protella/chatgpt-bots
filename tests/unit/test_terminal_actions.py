"""The terminal-action contract: eight declared reasons, and silence that ends words only.

Two things are under test here, and they are the two things that were wrong before:

1. WHY the bot stayed quiet is now a closed vocabulary the model picks from, recorded verbatim.
   It is never inferred from the message, the reaction, the tools or the posture, and an
   unrecognized value is a rejected call rather than a quiet rewrite to `other` — otherwise the
   one column that reports the model's own judgment would be partly our guesses.

2. no_response_needed ends the turn's WORDS, not its effects. It used to drop every sibling
   except a reaction, which meant a memory write the model had decided to make, or a post into
   another thread, vanished because the model also chose to add no words here.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_processor.terminal_actions import (SILENCE_REASON_DESCRIPTIONS,
                                                SILENCE_REASONS, SilenceReason,
                                                is_valid_silence_reason,
                                                render_reason_guide)


# --------------------------------------------------------------------- the vocabulary

def test_the_literal_and_the_tuple_stay_in_lockstep():
    """The Literal has to be spelled out for the type checker, so it is a second copy — and a
    second copy that drifts would type-check code the validator rejects at runtime."""
    from typing import get_args
    assert get_args(SilenceReason) == SILENCE_REASONS


def test_every_value_is_defined_for_the_model():
    """The schema description IS the definition the model reads. A value with no description is
    a value it has to guess the meaning of, and guesses are what the enum exists to remove."""
    assert set(SILENCE_REASON_DESCRIPTIONS) == set(SILENCE_REASONS)
    assert all(SILENCE_REASON_DESCRIPTIONS[v].strip() for v in SILENCE_REASONS)


def test_the_reason_guide_carries_every_value_and_the_dispatched_work_prohibition():
    guide = render_reason_guide()
    for value in SILENCE_REASONS:
        assert value in guide
    # awaiting_context is the one value with a hard prohibition attached: a turn that dispatched
    # its own work and then "waits" for it is a turn that abandoned the work.
    assert "NEVER for work you dispatched yourself" in guide


@pytest.mark.parametrize("value", list(SILENCE_REASONS))
def test_every_declared_value_validates(value):
    assert is_valid_silence_reason(value) is True


@pytest.mark.parametrize("value", ["", None, 7, True, "Other", "nothing to add", ["other"],
                                   "addressed_to_other ", "unknown"])
def test_nothing_else_validates(value):
    """Strict about type as well as membership: `True` is not a reason, and Python would
    happily compare it against a tuple of strings without complaint."""
    assert is_valid_silence_reason(value) is False


# ------------------------------------------------------------------------- the schema

def _schema():
    from slack_client.messaging import SlackMessagingMixin
    return SlackMessagingMixin.get_no_reply_tool_schema(MagicMock())


def test_the_schema_offers_exactly_the_eight_values():
    params = _schema()["parameters"]
    assert params["properties"]["reason"]["enum"] == list(SILENCE_REASONS)
    assert params["required"] == ["reason"]
    # Closed: a model that invents a second argument is told so by the API, not silently obeyed.
    assert params["additionalProperties"] is False


def test_the_description_states_the_terminal_contract():
    description = _schema()["description"]
    # It ends the WORDS, in THIS conversation — not the turn's other effects, which is exactly
    # the misreading that used to be true of the implementation.
    assert "without posting a normal text reply in the current conversation" in description
    assert "does not cancel other tools" in description
    assert "never after writing one" in description
    # ...and the prohibition that has no other home.
    assert "wait for work you dispatched yourself" in description


# ------------------------------------------------------- the loop: siblings and validity

class _LoopSelf:
    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


class _Registry:
    """Records dispatch order and can fail a named tool, so a sibling's FAILURE can be told
    apart from a sibling that never ran."""

    def __init__(self, dispatched, failing=()):
        self.dispatched = dispatched
        self.failing = set(failing)

    async def dispatch_all(self, ctx, calls):
        results = []
        for c in calls:
            name = c.get("name")
            self.dispatched.append(name)
            results.append({"ok": name not in self.failing})
        return results


def _fc(name, call_id, arguments="{}"):
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


def _scripted(scripts, texts=("", "later text")):
    state = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(scripts[min(i, len(scripts) - 1)])
        return {"text": texts[min(i, len(texts) - 1)], "tools_used": []}

    return fake_create


@pytest.mark.asyncio
@pytest.mark.parametrize("value", list(SILENCE_REASONS))
async def test_every_declared_value_survives_the_loop_unchanged(monkeypatch, value):
    """Verbatim, all eight. The ledger's column is the model's own word for what it did."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1",
                                        '{"reason": "%s"}' % value)]]))
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry([]), tool_context=None)
    assert result["terminal_action"] == "no_reply"
    assert result["silence_reason"] == value


@pytest.mark.asyncio
async def test_an_arbitrary_side_effect_sibling_runs_there_is_no_allowlist(monkeypatch):
    """A tool nobody anticipated executes too. An allowlist here would be a second copy of the
    authorization rules that already live in each executor, quietly diverging from them."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
                                    _fc("some_future_tool", "2", '{"x": 1}'),
                                    _fc("post_to_thread", "3", '{"thread_ts": "9.9"}')]]))
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched), tool_context=None)
    assert set(dispatched) == {"no_response_needed", "some_future_tool", "post_to_thread"}
    assert result["terminal_action"] == "no_reply"
    assert {c["name"] for c in result["local_tool_calls"]} == {"some_future_tool",
                                                               "post_to_thread"}


@pytest.mark.asyncio
async def test_a_failed_sibling_does_not_cancel_the_silence(monkeypatch):
    """Silence was the model's decision about words; a sibling failing is not a reason to start
    talking, and re-opening the turn would post something nobody chose to say."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "not_relevant"}'),
                                    _fc("remember_fact", "2", "{}")]]))
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[],
        registry=_Registry(dispatched, failing={"remember_fact"}), tool_context=None)
    assert result["terminal_action"] == "no_reply"
    assert result["silence_reason"] == "not_relevant"
    failed = [c for c in result["local_tool_calls"] if c["name"] == "remember_fact"]
    assert failed and failed[0]["ok"] is False      # recorded as failed, not hidden


@pytest.mark.asyncio
async def test_streaming_and_non_streaming_agree_on_siblings_and_reason(monkeypatch):
    """Two loops, one contract. They have drifted before, and a silence that runs its siblings
    in one and drops them in the other is the same bug twice."""
    from openai_client.api import tool_loop
    calls = [_fc("no_response_needed", "1", '{"reason": "duplicate"}'),
             _fc("remember_fact", "2", "{}")]

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([list(calls)]))
    plain_dispatched = []
    plain = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(plain_dispatched),
        tool_context=None)

    async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                             function_call_sink=None, tool_choice=None, **params):
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend([dict(c) for c in calls])
        return ""

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_streaming)
    streamed_dispatched = []
    streamed = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(streamed_dispatched),
        tool_context=None, stream_callback=lambda c: None)

    assert plain["silence_reason"] == streamed["silence_reason"] == "duplicate"
    assert sorted(plain_dispatched) == sorted(streamed_dispatched)
    assert [c["name"] for c in plain["local_tool_calls"]] == \
           [c["name"] for c in streamed["local_tool_calls"]] == ["remember_fact"]


# ------------------------------------------------- what the room saw, when words were skipped

def _resp(**meta):
    return MagicMock(type="text", content="", metadata=meta)


def test_a_detached_surface_outranks_silence():
    """A background job posts its own card. A turn that started one and added no words is a
    DETACHED turn — filing it as silence would describe an empty room that has a status card
    ticking away in it."""
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="nothing_to_add",
              background_job_started=True), None) == "detached"
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="nothing_to_add"),
        TurnRuntime(visible_action_committed=True)) == "detached"


def test_reacted_instead_without_a_reaction_is_preserved_not_corrected():
    """The model said the emoji was its answer and no emoji landed. Both facts are recorded as
    they are: `silence` + reaction_visible false + the reason it gave. Correcting either one
    would erase the only evidence that the two disagree."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="reacted_instead"), None) == "silence"


def test_a_reaction_that_landed_makes_it_reaction_only_whatever_the_reason():
    from main import ChatBotV2
    for reason in ("reacted_instead", "nothing_to_add", "addressed_to_other"):
        assert ChatBotV2._classify_visible_action(
            _resp(terminal_action="no_reply", silence_reason=reason,
                  response_reaction_committed=True), None) == "reaction_only"


def test_a_bare_empty_turn_is_still_a_contract_violation():
    """Never inferred silence: posting nothing WITHOUT calling the terminal tool is the
    responder contract breaking, and it must stay distinguishable from a chosen silence."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(_resp(), None) == "empty"


# ---------------------------------------------------------------- post_to_thread is visible

@pytest.mark.asyncio
async def test_post_to_thread_marks_the_turn_as_having_produced_output():
    """Words went into the workspace, just not into this thread — a legitimate pairing with
    silence ("I answered over there"). Without this the ledger calls that turn a silence."""
    from message_processor.turn_runtime import TurnRuntime
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host.send_message = AsyncMock(return_value="99.9")
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)
    turn = TurnRuntime()
    ctx = SimpleNamespace(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)
    with patch("slack_client.messaging.config") as cfg:
        cfg.enable_post_to_thread_tool = True
        result = await host.execute_post_to_thread(ctx, {"thread_ts": "50.0", "text": "over here"})
    assert result["ok"] is True
    assert turn.visible_action_committed is True


@pytest.mark.asyncio
async def test_a_refused_post_marks_nothing():
    from message_processor.turn_runtime import TurnRuntime
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host.send_message = AsyncMock(return_value=None)     # Slack refused it
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)
    turn = TurnRuntime()
    ctx = SimpleNamespace(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)
    with patch("slack_client.messaging.config") as cfg:
        cfg.enable_post_to_thread_tool = True
        result = await host.execute_post_to_thread(ctx, {"thread_ts": "50.0", "text": "hi"})
    assert result["ok"] is False
    assert turn.visible_action_committed is False


# ------------------------------------------------------------ the settings tool's registration

def test_the_settings_tool_is_registered_even_with_the_engine_off(mock_env, monkeypatch):
    """The engine decides whether UNADDRESSED traffic is judged. It has nothing to say about
    whether a person who addressed us may change the settings — and with the engine off is
    exactly when someone is most likely to want to."""
    from config import config as cfg
    from slack_client.base import SlackBot
    monkeypatch.setattr(cfg, "enable_participation_engine", False, raising=False)
    monkeypatch.setattr(cfg, "enable_tool_loop", True, raising=False)

    # Every schema getter must return a REAL dict — register() reads a MagicMock as a
    # per-request schema factory and refuses it.
    host = MagicMock()
    for getter, name in (("get_history_tools_for_openai", None),
                         ("get_react_tool_schema", "react_to_message"),
                         ("get_search_tool_schema", "search_slack"),
                         ("get_post_to_thread_tool_schema", "post_to_thread"),
                         ("get_no_reply_tool_schema", "no_response_needed"),
                         ("get_emoji_search_tool_schema", "search_workspace_emoji"),
                         ("get_lookup_channel_tool_schema", "lookup_channel"),
                         ("get_resolve_channel_name_tool_schema", "resolve_channel_name")):
        schema = {"type": "function", "name": name, "parameters": {}}
        getattr(host, getter).return_value = (
            [{"type": "function", "name": "fetch_channel_history", "parameters": {}}]
            if name is None else schema)
    registry = SlackBot._build_tool_registry(host)
    assert "set_channel_participation" in {s["name"] for s in registry.schemas({})}


# ----------------------------------------------- what a silent turn still hands to delivery

def _terminal_processor(loop_result, *, background=False, sandbox_assets=(), artifacts=()):
    """The REAL _handle_text_response on a minimal host, driven to the terminal branch.

    The point is the metadata contract at the end of a silent turn: everything the turn
    PRODUCED still has to reach main.py, because silence was a decision about words.
    """
    from base_client import Response  # noqa: F401 — the handler builds one
    from message_processor.handlers.text import TextHandlerMixin

    host = MagicMock()
    host._handle_text_response = TextHandlerMixin._handle_text_response.__get__(host)
    host._is_reaction_only = MagicMock(return_value=False)
    host.db = None
    host.mcp_manager = MagicMock()
    host.mcp_manager.cache_discovered_tools_payload = MagicMock()

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return ""

    host._inject_image_analyses = _passthru
    host._pre_trim_messages_for_api = _passthru
    host._build_channel_info = _empty
    host._drop_dead_containers = _none
    host._resolve_ci_container = _none
    host._prepare_sandbox_tools = _none
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_suffix_context = MagicMock(return_value="")
    host._build_participant_roster = MagicMock(return_value="")
    host._build_tools_array = MagicMock(return_value=[{"type": "function", "name": "t"}])
    host._materialize_request_tools = MagicMock(
        return_value=(MagicMock(), {"model": "m"}, True, None))
    host._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=background,
        sandbox_image_assets=list(sandbox_assets),
        mounted_files=[{"digest": "d1"}],
    ))
    host._add_message_with_token_management = MagicMock()
    host._schedule_async_call = MagicMock()

    host.openai_client = MagicMock()
    host.openai_client.create_text_response_with_tool_loop = AsyncMock(
        return_value=loop_result)
    # The artifacts accumulator the handler collects container ids from.
    host._artifacts = list(artifacts)
    return host


@pytest.mark.asyncio
async def test_a_silent_turn_still_hands_over_everything_it_produced():
    """Files built in the sandbox, images made as ingredients, a job's card: silence ends the
    WORDS. A terminal response that dropped these swallowed the turn's own deliverables."""
    from base_client import Message
    from config import config as cfg

    host = _terminal_processor(
        {"text": "", "tools_used": [], "local_tool_calls": [],
         "terminal_action": "no_reply", "silence_reason": "nothing_to_add"},
        sandbox_assets=[{"image_data": b"x"}],
        artifacts=[{"container_id": "cont_1", "file_id": "f1", "filename": "a.png"}])
    message = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id="C1", thread_ts="10.0",
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None)

    async def fake_config(**kw):
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": False, "enable_code_interpreter": False}

    with patch.object(cfg, "get_thread_config_async", side_effect=fake_config):
        response = await host._handle_text_response(
            "hi", thread_state, MagicMock(), message, thinking_id=None,
            artifacts_acc=host._artifacts)

    meta = response.metadata
    assert meta["terminal_action"] == "no_reply"
    assert meta["silence_reason"] == "nothing_to_add"
    assert meta["artifact_containers"] == ["cont_1"]
    assert meta["sandbox_image_assets"] == [{"image_data": b"x"}]
    assert meta["mounted_digests"] == ["d1"]


@pytest.mark.asyncio
async def test_a_started_job_is_never_hidden_by_the_silence_that_accompanied_it():
    """Both facts ride the response. The job's card is what the room sees, so main.py must be
    able to see it too — and the model's own account of why it added no words survives."""
    from base_client import Message
    from config import config as cfg

    host = _terminal_processor(
        {"text": "", "tools_used": [], "local_tool_calls": [],
         "terminal_action": "no_reply", "silence_reason": "awaiting_context"},
        background=True,
        sandbox_assets=[{"image_data": b"y"}],
        artifacts=[{"container_id": "cont_2", "file_id": "f2", "filename": "b.csv"}])
    message = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id="C1", thread_ts="10.0",
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None)

    async def fake_config(**kw):
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": False, "enable_code_interpreter": False}

    with patch.object(cfg, "get_thread_config_async", side_effect=fake_config):
        response = await host._handle_text_response(
            "hi", thread_state, MagicMock(), message, thinking_id=None,
            artifacts_acc=host._artifacts)

    meta = response.metadata
    assert meta["background_job_started"] is True
    assert meta["terminal_action"] == "no_reply"
    assert meta["silence_reason"] == "awaiting_context"
    # This branch returns BEFORE the terminal branch, so it is the one that has to carry the
    # deliverables too — a job starting in the same round must not eat the chart.
    assert meta["artifact_containers"] == ["cont_2"]
    assert meta["sandbox_image_assets"] == [{"image_data": b"y"}]
    assert meta["mounted_digests"] == ["d1"]


# ------------------------------------------------- the reaction fact, on every terminal path

@pytest.mark.asyncio
async def test_the_react_tool_marks_the_turn_wherever_the_turn_ends():
    """The fact is recorded where the emoji LANDS, not per-Response.

    It used to be derived in each handler branch, and only the no-reply branch actually built
    the field — so a reaction-only turn and a reply that also reacted both told the ledger "no
    emoji" while the `added` reaction event sat right beside them in the same file."""
    from message_processor.turn_runtime import TurnRuntime
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host._reserve_and_react = AsyncMock(return_value={"ok": True})
    host.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(host)
    turn = TurnRuntime()
    ctx = SimpleNamespace(channel_id="C1", trigger_ts="10.0", thread_ts="10.0",
                          attempt_id=None, turn=turn)
    with patch("slack_client.messaging.config") as cfg:
        cfg.enable_reactions = True
        cfg.enable_react_tool = True
        cfg.reaction_emojis = []
        result = await host.execute_react_tool(ctx, {"emoji": "tada"})
    assert result["ok"] is True
    assert turn.reaction_committed is True


@pytest.mark.asyncio
async def test_a_refused_reaction_marks_nothing():
    from message_processor.turn_runtime import TurnRuntime
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host._reserve_and_react = AsyncMock(return_value={"ok": False, "error": "cap_reached"})
    host.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(host)
    turn = TurnRuntime()
    ctx = SimpleNamespace(channel_id="C1", trigger_ts="10.0", thread_ts="10.0",
                          attempt_id=None, turn=turn)
    with patch("slack_client.messaging.config") as cfg:
        cfg.enable_reactions = True
        cfg.enable_react_tool = True
        cfg.reaction_emojis = []
        await host.execute_react_tool(ctx, {"emoji": "tada"})
    assert turn.reaction_committed is False


def test_a_reply_that_also_reacted_is_classified_with_its_reaction():
    """A REPLY carries no `response_reaction_committed` — only the no-reply branch ever built
    that field. Reading the turn instead is what makes the emoji visible on every route."""
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime

    reacted = TurnRuntime(reaction_committed=True)
    # A reaction-only turn ending in silence is now correctly a reaction_only...
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="reacted_instead"),
        reacted) == "reaction_only"
    # ...and the turn's own fact is what says the 👀 was honored.
    assert ChatBotV2._produced_visible_output(
        _resp(terminal_action="no_reply", silence_reason="reacted_instead"), reacted) is True


# ------------------------------------------- a rejected terminal's duplicates are not executed

@pytest.mark.asyncio
async def test_duplicate_terminals_are_answered_not_executed_on_a_rejected_round(monkeypatch):
    """A turn cannot end twice. On a round the loop CONTINUES past, a duplicate used to be
    dispatched normally and answered ok — so the model read one terminal refused and another
    accepted while the turn carried on."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "made up"}'),
                                    _fc("no_response_needed", "2", '{"reason": "nothing_to_add"}'),
                                    _fc("react_to_message", "3", '{"emoji": "eyes"}')], []]))
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched), tool_context=None)
    assert "no_response_needed" not in dispatched          # neither of them ran
    assert dispatched.count("react_to_message") == 1       # the ordinary sibling still did
    terminals = [c for c in result["local_tool_calls"] if c["name"] == "no_response_needed"]
    assert len(terminals) == 2 and all(c["ok"] is False for c in terminals)
    assert result.get("terminal_action") is None           # and the turn continued


# ------------------------------------------- bookkeeping calls never take a productive slot

@pytest.mark.asyncio
async def test_a_free_sibling_does_not_eat_the_productive_slot(monkeypatch):
    """With one productive slot left and a round ordered [update_todos, remember_fact], a naive
    slice spends it on the status card and drops the memory write — the exact inversion the
    free-tool allowance exists to prevent."""
    from openai_client.api import tool_loop

    calls = [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
             _fc("update_todos", "2", "{}"),
             _fc("remember_fact", "3", "{}")]

    async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                             function_call_sink=None, tool_choice=None, **params):
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(calls)
        return ""

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_streaming)
    dispatched = []
    result = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched),
        tool_context=None, stream_callback=lambda c: None,
        max_tool_calls=2,                      # terminal reserves one → one productive slot
        free_tools=["update_todos"])
    assert result["terminal_action"] == "no_reply"
    assert "remember_fact" in dispatched       # the productive call got the slot
    assert "update_todos" in dispatched        # ...and the free one ran on its own allowance


@pytest.mark.parametrize("response", [
    MagicMock(type="error", content="boom", metadata={}),
    MagicMock(type="text", content="sorry, that died", metadata={"interrupted": True}),
    MagicMock(type="text", content="the answer", metadata={"posted": False}),
])
def test_a_reaction_survives_a_turn_that_then_failed(response):
    """The 👀 is a claim on work, and a placed emoji is work the reader can still see. An error,
    an interruption or a failed send afterwards is not the reaction's fault — retracting the eye
    there takes back a mark that is visibly still on the message."""
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime
    assert ChatBotV2._produced_visible_output(
        response, TurnRuntime(reaction_committed=True)) is True
    # ...and with no reaction, each of these still correctly retracts it.
    assert ChatBotV2._produced_visible_output(response, TurnRuntime()) is False


def test_a_queued_turn_never_claims_a_reaction():
    """A queued turn ran no tools, so nothing could have set the fact — the exit stays honest
    without a special case pretending to handle a state that cannot occur."""
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime
    queued = MagicMock(type="queued", content="", metadata={})
    assert TurnRuntime().reaction_committed is False
    assert ChatBotV2._produced_visible_output(queued, TurnRuntime()) is False


# ------------------------------------- the executor, not the schema, authorizes the silence

class _TerminalRegistry:
    """Dispatches through the REAL no-reply executor, so the loop's terminal contract is tested
    against the thing that actually decides it."""

    def __init__(self, turn, dispatched=None, failing=()):
        from slack_client.messaging import SlackMessagingMixin
        self.turn = turn
        self.dispatched = dispatched if dispatched is not None else []
        self.failing = set(failing)
        self._executor = SlackMessagingMixin.execute_no_reply_tool.__get__(MagicMock())

    async def dispatch_all(self, ctx, calls):
        results = []
        for c in calls:
            name = c.get("name")
            self.dispatched.append(name)
            if name == "no_response_needed":
                results.append(await self._executor(
                    SimpleNamespace(turn=self.turn), {}))
            else:
                results.append({"ok": name not in self.failing})
        return results


def _turn(silence_capable):
    return SimpleNamespace(silence_capable=silence_capable)


@pytest.mark.asyncio
async def test_the_executor_refuses_on_an_owed_words_turn(mock_env):
    """The channel surface exposes the tool statically, so the schema can no longer say which
    routes may be quiet. The executor does, and it fails CLOSED."""
    from slack_client.messaging import SlackMessagingMixin
    executor = SlackMessagingMixin.execute_no_reply_tool.__get__(MagicMock())
    owed = await executor(SimpleNamespace(turn=_turn(False)), {})
    assert owed["ok"] is False and owed["error"] == "reply_owed"
    # An absent turn is refused too — a context that never derived the fact cannot authorize it.
    assert (await executor(SimpleNamespace(turn=None), {}))["ok"] is False
    assert (await executor(SimpleNamespace(turn=_turn(True)), {}))["ok"] is True


@pytest.mark.asyncio
async def test_the_executor_refuses_when_the_feature_is_off(mock_env, monkeypatch):
    from config import config as cfg
    from slack_client.messaging import SlackMessagingMixin
    monkeypatch.setattr(cfg, "enable_no_reply_tool", False)
    executor = SlackMessagingMixin.execute_no_reply_tool.__get__(MagicMock())
    out = await executor(SimpleNamespace(turn=_turn(True)), {})
    assert out["ok"] is False and out["error"] == "disabled"


@pytest.mark.asyncio
async def test_a_refused_terminal_continues_the_loop_and_keeps_its_siblings(mock_env, monkeypatch):
    """The refusal is fed back as this call's output, the siblings keep theirs, and the model
    gets another round to write the reply it owes. Silencing the turn on a refused call would
    swallow an answer somebody is waiting for."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
                                    _fc("remember_fact", "2", "{}")], []],
                                  texts=("", "here you go")))
    registry = _TerminalRegistry(_turn(False))
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=registry, tool_context=None)
    assert result.get("terminal_action") is None
    assert result.get("silence_reason") is None
    assert result["text"] == "here you go"
    assert registry.dispatched.count("remember_fact") == 1      # the sibling ran, once
    terminal = [c for c in result["local_tool_calls"] if c["name"] == "no_response_needed"]
    assert terminal and terminal[0]["ok"] is False              # recorded as the refusal it is


@pytest.mark.asyncio
async def test_a_refused_terminal_replays_every_call_with_an_output(mock_env, monkeypatch):
    """The Responses API 400s on a function_call with no matching function_call_output, and the
    budget skips calls the terminal round would otherwise leave unanswered."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.config, "max_tool_calls_per_turn", 2)
    calls = [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
             _fc("react_to_message", "2", '{"emoji": "eyes"}'),
             _fc("react_to_message", "3", '{"emoji": "tada"}'),
             _fc("remember_fact", "4", "{}")]
    scripted = _scripted([list(calls), []], texts=("", "done"))
    rounds_seen = []

    async def _capture(client, messages, tools, return_metadata, function_call_sink,
                       tool_choice=None, **kw):
        rounds_seen.append(list(messages))
        return await scripted(client, messages, tools, return_metadata,
                              function_call_sink, tool_choice=tool_choice, **kw)

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", _capture)
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_TerminalRegistry(_turn(False)),
        tool_context=None)
    assert result.get("terminal_action") is None
    second_round = rounds_seen[1]
    call_ids = {i["call_id"] for i in second_round if i.get("type") == "function_call"}
    output_ids = {i["call_id"] for i in second_round
                  if i.get("type") == "function_call_output"}
    assert call_ids == output_ids == {"1", "2", "3", "4"}


@pytest.mark.asyncio
async def test_a_refused_terminal_charges_the_round_and_forces_the_answer_at_the_cap(
        mock_env, monkeypatch):
    """A refusal that cost nothing would let the model loop on the tool forever."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.config, "max_tool_rounds", 1)
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "duplicate"}')],
                                   [_fc("no_response_needed", "9", '{"reason": "duplicate"}')]],
                                  texts=("", "forced answer")))
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_TerminalRegistry(_turn(False)),
        tool_context=None)
    assert result.get("terminal_action") is None
    assert result["text"] == "forced answer"


@pytest.mark.asyncio
async def test_a_refused_terminal_replays_the_streamed_preamble(mock_env, monkeypatch):
    """Streaming: whatever the model said before calling the tool is already on screen. Without
    the replay the next round has no record of having spoken and says it again, into the same
    message."""
    from openai_client.api import tool_loop
    rounds = [("Sure — one sec.",
               [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}')]),
              (" here it is.", [])]
    seen = []
    state = {"n": 0}

    async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                             function_call_sink=None, tool_choice=None, **params):
        seen.append(list(messages))
        text, calls = rounds[min(state["n"], len(rounds) - 1)]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend([dict(c) for c in calls])
        return text

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_streaming)
    out = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_TerminalRegistry(_turn(False)),
        tool_context=None, stream_callback=lambda c: None)
    assert out.get("terminal_action") is None
    assert {"role": "assistant", "content": "Sure — one sec."} in seen[1]


@pytest.mark.asyncio
async def test_a_permitted_turn_still_ends_in_silence(mock_env, monkeypatch):
    """The other half of the same contract: an authorized executor success is still terminal."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "not_relevant"}'),
                                    _fc("react_to_message", "2", '{"emoji": "eyes"}')]]))
    registry = _TerminalRegistry(_turn(True))
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=registry, tool_context=None)
    assert result["terminal_action"] == "no_reply"
    assert result["silence_reason"] == "not_relevant"
    assert result["text"] == ""
    assert "react_to_message" in registry.dispatched
    # An ACCEPTED terminal is the turn's ending, not one of its actions.
    assert [c["name"] for c in result["local_tool_calls"]] == ["react_to_message"]


# --------------------------------------- moved from test_no_reply_tool.py (spec §14)

@pytest.mark.asyncio
async def test_the_terminal_reserves_a_slot_of_the_call_budget(monkeypatch):
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.config, "max_tool_calls_per_turn", 2)
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}'),
                                    _fc("react_to_message", "2", '{"emoji": "eyes"}'),
                                    _fc("react_to_message", "3", '{"emoji": "tada"}'),
                                    _fc("react_to_message", "4", '{"emoji": "thumbsup"}')]]))
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched), tool_context=None)
    assert result["terminal_action"] == "no_reply"
    assert dispatched.count("react_to_message") == 1
    assert dispatched.count("no_response_needed") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    '{"reason": "it is not really for me"}',   # prose, the old contract
    '{"reason": ""}',                          # empty
    '{}',                                      # absent
    '{"reason": 7}',                           # not a string
    'not json at all',                         # unparseable
])
async def test_an_unrecognized_reason_is_rejected_never_coerced(monkeypatch, arguments):
    """The turn is NOT silenced and the reason is NOT rewritten to `other`."""
    from openai_client.api import tool_loop
    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([[_fc("no_response_needed", "1", arguments)], []],
                                  texts=("", "fine, here you go")))
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched), tool_context=None)
    assert result.get("terminal_action") is None
    assert result.get("silence_reason") is None
    assert result["text"] == "fine, here you go"
    assert "no_response_needed" not in dispatched
    rejected = [c for c in result["local_tool_calls"] if c["name"] == "no_response_needed"]
    assert rejected and all(c["ok"] is False for c in rejected)


@pytest.mark.asyncio
async def test_streaming_silence_is_honored_only_before_visible_text(monkeypatch):
    from openai_client.api import tool_loop

    def _fake(rounds):
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            text, calls = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend([dict(c) for c in calls])
            return text

        return fake_streaming

    terminal = [_fc("no_response_needed", "1", '{"reason": "addressed_to_other"}')]
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _fake([("", terminal)]))
    quiet = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry([]), tool_context=None,
        stream_callback=lambda c: None)
    assert quiet["terminal_action"] == "no_reply" and quiet["text"] == ""

    # ...and once a reply has begun, the same call is invalid and the model completes it.
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        _fake([("Sure, here's the answer", terminal), (" — all done.", [])]))
    dispatched = []
    committed = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry(dispatched), tool_context=None,
        stream_callback=lambda c: None)
    assert committed.get("terminal_action") is None
    assert committed["text"] == " — all done."
    assert "no_response_needed" not in dispatched


@pytest.mark.asyncio
async def test_prior_committed_text_rejects_silence_on_both_loops(monkeypatch):
    """F8: a partial reply from an EARLIER attempt must never be orphaned as fake silence."""
    from openai_client.api import tool_loop
    terminal = [_fc("no_response_needed", "1", '{"reason": "nothing_to_add"}')]

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools",
                        _scripted([list(terminal), []], texts=("", "finished reply")))
    plain = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry([]), tool_context=None,
        prior_committed=True)
    assert plain.get("terminal_action") is None and plain["text"] == "finished reply"

    state = {"n": 0}
    rounds = [("", terminal), ("recovered reply", [])]

    async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                             function_call_sink=None, tool_choice=None, **params):
        text, calls = rounds[min(state["n"], len(rounds) - 1)]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend([dict(c) for c in calls])
        return text

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_streaming)
    streamed = await tool_loop.create_streaming_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_Registry([]), tool_context=None,
        stream_callback=lambda c: None, prior_committed=True)
    assert streamed.get("terminal_action") is None and streamed["text"] == "recovered reply"


@pytest.mark.asyncio
async def test_a_result_override_is_keyed_by_identity_not_call_id():
    """A degenerate round where a sibling shares the terminal's call_id must not misroute the
    override onto the sibling."""
    from openai_client.api import tool_loop
    terminal = _fc("no_response_needed", "dup", '{"reason": "nothing_to_add"}')
    sibling = _fc("react_to_message", "dup", '{"emoji": "eyes"}')
    sink = [terminal, sibling]
    local, dispatched = [], []
    await tool_loop._run_tool_round(
        _LoopSelf(), _Registry(dispatched), None, sink, [], local,
        result_overrides={id(terminal): tool_loop._INVALID_NO_REPLY_RESULT})
    assert dispatched == ["react_to_message"]
    assert local[0]["name"] == "no_response_needed" and local[0]["ok"] is False
    assert local[1]["name"] == "react_to_message" and local[1]["ok"] is True


# ------------------------------- the contract paragraphs (moved from test_no_reply_tool.py)

def test_channel_activity_paragraph_wording():
    from prompts import CHANNEL_ACTIVITY_NO_REPLY_SUFFIX as s
    assert "uninvited" not in s
    assert "Silence is the DEFAULT here" in s
    assert "could not easily get themselves" in s
    assert "no_response_needed" in s
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "addressed by name, prefer a brief honest answer over silence" in s
    assert "ends your words, not your other actions" in s
    assert "wait for work you started yourself" in s
    assert s.startswith("[") and s.endswith("]")


def test_thread_activity_paragraph_wording():
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX as s)
    assert "check its addressee yourself" in s
    assert "STICKS" in s
    assert "silence means silence" in s
    assert "no_response_needed" in s
    assert "ends your words, not your other actions" in s
    assert "wait for work you started yourself" in s
    assert s != CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert s.startswith("[") and s.endswith("]")


def test_neither_paragraph_restates_the_silence_vocabulary():
    """The eight values live in the tool schema. Two copies of one vocabulary drift."""
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX)
    identifiers = [v for v in SILENCE_REASONS if "_" in v]
    for paragraph in (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX, THREAD_ACTIVITY_NO_REPLY_SUFFIX):
        assert not [v for v in identifiers if v in paragraph]


# ------------------------- silent-stream teardown (moved from test_no_reply_tool.py)

def _cleanup_host():
    from message_processor.handlers.text import TextHandlerMixin
    host = SimpleNamespace(warnings=[], debugs=[])
    host._cleanup_silent_stream = TextHandlerMixin._cleanup_silent_stream.__get__(host)
    host.log_warning = lambda m, *a, **k: host.warnings.append(str(m))
    host.log_debug = lambda m, *a, **k: host.debugs.append(str(m))
    return host


@pytest.mark.asyncio
async def test_cleanup_silent_stream_deletes_both_distinct_messages():
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
    native = SimpleNamespace(started=True, abandon=AsyncMock(return_value=False))
    await host._cleanup_silent_stream(client, "C1", native, None, "stream.2", "reaction-only")
    assert deleted == ["stream.2"]
    assert any("abandon failed" in w for w in host.warnings)
