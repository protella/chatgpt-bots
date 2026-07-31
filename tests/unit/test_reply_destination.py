"""Where the reply goes — chosen by the model that writes it, stated once, never re-derived.

Placement used to be inferred by three separate authors that could not see the answer: a channel
setting, a gate verdict formed before the reply existed, and a "did any tool run" flag that
re-threaded a top-level reply after the fact. This is the replacement, and these tests hold the
three properties that make it an improvement rather than a rename:

1. The tool is offered ONLY where both destinations are genuinely legal. A DM, a thread, and a
   channel that forbids top-level replies have nothing to decide, and they behave exactly as
   they always have.
2. Nothing is shown ANYWHERE until the destination is settled. A placeholder in the wrong place
   is the bug the old design kept producing.
3. The choice is the model's, and it is never guessed: an invalid value is refused, a conflicting
   second call is refused, a call after the reply is up is refused, and an answer that arrives
   with no call at all is still delivered — in the default thread, with the miss recorded.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor.destination_tools import (SET_REPLY_DESTINATION,
                                                 destination_tool_available,
                                                 execute_set_reply_destination,
                                                 get_set_reply_destination_schema)
from message_processor.turn_runtime import (SELECTABLE_DESTINATIONS, TurnRuntime)


def _message(channel="C1", thread="10.0", ts="10.0", **meta):
    return SimpleNamespace(channel_id=channel, thread_id=thread, user_id="U1", text="hi",
                           attachments=None, metadata={"ts": ts, **meta})


# ------------------------------------------------------------------ which turns get the choice

@pytest.mark.parametrize("message, allowed, destination, source, selected", [
    # A DM has nowhere else to go.
    (_message(channel="D1"), True, "dm", "structural", True),
    # A reply inside a thread belongs to that thread.
    (_message(ts="11.0"), True, "thread", "structural", True),
    # The channel forbids top-level replies: settled in its settings.
    (_message(), False, "thread", "structural", True),
    # Both legal: open, and defaulting to the thread until the model says otherwise.
    (_message(), True, "thread", "default", False),
])
def test_the_route_decides_who_decides(message, allowed, destination, source, selected):
    turn = TurnRuntime.for_message(message, channel_post_allowed=allowed)
    assert turn.reply_destination == destination
    assert turn.destination_source == source
    assert turn.destination_selected is selected


def test_only_an_open_turn_shows_the_tool():
    """The `enabled` predicate reads a per-request flag the text handler stamps from the turn.
    Absent → not offered, which is the safe default: the reply lands where it always did."""
    assert destination_tool_available({"_destination_choice_open": True}) is True
    assert destination_tool_available({}) is False
    assert destination_tool_available({"_destination_choice_open": False}) is False


@pytest.mark.parametrize("message, allowed", [
    (_message(channel="D1"), True),      # DM
    (_message(ts="11.0"), True),         # in-thread
    (_message(), False),                 # top-level, channel posting forbidden
])
def test_a_settled_route_is_never_offered_the_tool(mock_env, message, allowed):
    from message_processor.handlers.text import TextHandlerMixin
    host = SimpleNamespace(
        _get_tool_registry=lambda client, cfg, surface="dm": None,
        _materialize_request_tools=None)
    host._materialize_request_tools = TextHandlerMixin._materialize_request_tools.__get__(host)
    turn = TurnRuntime.for_message(message, channel_post_allowed=allowed)
    _, request_config, _, suffix = host._materialize_request_tools(
        MagicMock(), {"model": "m"}, message, tools_disabled=False, turn=turn)
    assert "_destination_choice_open" not in request_config
    assert suffix is None


def test_an_open_turn_gets_the_flag_and_the_contract_paragraph(mock_env):
    from message_processor.handlers.text import TextHandlerMixin
    from prompts import DESTINATION_CONTRACT_SUFFIX
    from tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(get_set_reply_destination_schema(), AsyncMock(return_value={"ok": True}),
                      name=SET_REPLY_DESTINATION, enabled=destination_tool_available)
    host = SimpleNamespace()
    for name in ("_materialize_request_tools", "_get_tool_registry"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host))
    host._client = SimpleNamespace(tool_registry=registry)

    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    reg, request_config, _, suffix = host._materialize_request_tools(
        host._client, {"model": "m"}, message, tools_disabled=False, turn=turn)
    assert request_config["_destination_choice_open"] is True
    assert SET_REPLY_DESTINATION in {s["name"] for s in reg.schemas(request_config)}
    assert suffix == DESTINATION_CONTRACT_SUFFIX


def test_a_timeout_retry_drops_the_tool_and_its_paragraph_together(mock_env):
    """Same rule the silence tool follows: the retry runs the loop-less API, so a paragraph
    telling the model to call something it cannot see would be an instruction it must disobey."""
    from message_processor.handlers.text import TextHandlerMixin
    host = SimpleNamespace()
    for name in ("_materialize_request_tools", "_get_tool_registry"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host))
    host._client = SimpleNamespace(tool_registry=MagicMock())
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    registry, _, _, suffix = host._materialize_request_tools(
        host._client, {"model": "m"}, message, tools_disabled=True, turn=turn)
    assert registry is None and suffix is None


# --------------------------------------------------------------------------- the schema

def test_the_schema_offers_exactly_two_destinations():
    params = get_set_reply_destination_schema()["parameters"]
    assert params["properties"]["destination"]["enum"] == list(SELECTABLE_DESTINATIONS)
    assert params["properties"]["destination"]["enum"] == ["thread", "channel"]
    assert params["required"] == ["destination"]
    assert params["additionalProperties"] is False


def test_the_contract_paragraph_defaults_to_thread_when_balanced():
    """The rubric is REVERSED from the utility classifier this replaces, which answered
    "channel" whenever it was unsure — so every ambiguous long answer landed in the room."""
    from prompts import DESTINATION_CONTRACT_SUFFIX as s
    assert "call set_reply_destination exactly once" in s
    assert "If it is a close call, choose thread" in s


# ------------------------------------------------------------------- choosing, and refusing

@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["thread", "channel"])
async def test_a_valid_choice_is_recorded_on_the_turn(destination):
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    result = await execute_set_reply_destination(
        SimpleNamespace(turn=turn, message=message), {"destination": destination})
    assert result["ok"] is True
    assert turn.reply_destination == destination
    assert turn.destination_source == "model"
    assert turn.destination_selected is True
    assert turn.destination_contract_miss is False


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {}, {"destination": ""}, {"destination": "dm"}, {"destination": "Channel"},
    {"destination": "top-level"}, {"destination": None}, {"destination": True},
])
async def test_an_invalid_destination_is_refused_never_guessed(args):
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    result = await execute_set_reply_destination(
        SimpleNamespace(turn=turn, message=message), args)
    assert result["ok"] is False and result["error"] == "invalid_destination"
    # …and the turn is untouched: still open, still defaulting to the thread.
    assert turn.destination_selected is False
    assert turn.reply_destination == "thread"


@pytest.mark.asyncio
async def test_an_identical_repeat_is_idempotent():
    """Models re-state decisions. That is not a conflict, and refusing it would teach the model
    that its own choice failed."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    ctx = SimpleNamespace(turn=turn, message=message)
    assert (await execute_set_reply_destination(ctx, {"destination": "channel"}))["ok"] is True
    second = await execute_set_reply_destination(ctx, {"destination": "channel"})
    assert second["ok"] is True and second["idempotent"] is True
    assert turn.reply_destination == "channel"


@pytest.mark.asyncio
async def test_a_conflicting_second_choice_is_refused():
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    ctx = SimpleNamespace(turn=turn, message=message)
    await execute_set_reply_destination(ctx, {"destination": "channel"})
    result = await execute_set_reply_destination(ctx, {"destination": "thread"})
    assert result["ok"] is False and result["error"] == "destination_conflict"
    assert turn.reply_destination == "channel"      # the first choice stands


@pytest.mark.asyncio
async def test_no_choice_survives_the_reply_going_out():
    """After a surface exists the destination is a fact about Slack, not a preference: moving it
    would strand the message already posted."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    turn.lock_destination()
    result = await execute_set_reply_destination(
        SimpleNamespace(turn=turn, message=message), {"destination": "channel"})
    assert result["ok"] is False and result["error"] == "destination_locked"
    assert turn.reply_destination == "thread"


@pytest.mark.asyncio
async def test_a_context_with_no_turn_refuses_rather_than_lying():
    result = await execute_set_reply_destination(
        SimpleNamespace(turn=None, message=_message()), {"destination": "channel"})
    assert result["ok"] is False and result["error"] == "no_turn"


def test_a_missed_call_delivers_in_the_default_thread_and_records_the_miss():
    """Never drop a valid answer, and never infer the destination from it either."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    turn.settle_default_destination()
    assert turn.reply_destination == "thread"
    assert turn.destination_source == "default"
    assert turn.destination_contract_miss is True
    assert turn.resolve_reply_target(message) == "10.0"


def test_settling_never_overwrites_a_choice_the_model_made():
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    turn.select_destination("channel", message=message)
    turn.settle_default_destination()
    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"
    assert turn.destination_contract_miss is False


def test_a_productive_tool_never_moves_an_explicit_choice():
    """The `did_substantive_work` override is gone: a tool running is not evidence about where
    the answer belongs, and it used to silently overrule the only party who knows."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    turn.select_destination("channel", message=message)
    turn.visible_action_committed = True        # a detached producer ran
    turn.reaction_committed = True              # and a reaction landed
    assert turn.resolve_reply_target(message) is None
    assert turn.final_post_only is True
    assert not hasattr(turn, "did_substantive_work")


# ------------------------------------------------------------- no surface before the choice

def test_an_open_turn_shows_no_chrome_anywhere():
    """The whole point of deferring: a placeholder cannot be minted in a place the answer may
    not go, so an unresolved turn mints none in EITHER place."""
    assert TurnRuntime.for_message(_message(), channel_post_allowed=True).progress_enabled is False
    # …while a settled thread turn still shows its indicator exactly as before.
    assert TurnRuntime.for_message(_message(ts="11.0"),
                                   channel_post_allowed=True).progress_enabled is True


def test_the_free_tool_costs_no_productive_budget():
    """It is bookkeeping: it produces nothing and must never take a slot a real tool needs."""
    from message_processor.handlers.text import TextHandlerMixin
    import inspect
    src = inspect.getsource(TextHandlerMixin._handle_streaming_text_response)
    assert "free_tools=" in src and SET_REPLY_DESTINATION in src


# ---------------------------------------------------------------------- telemetry linkage

def test_only_the_responders_own_reply_carries_a_destination():
    """The field means where THIS turn's reply went. A silence went nowhere; a detached
    producer posts its own surface into the thread, bypassing the destination entirely, so
    reporting the turn's would describe a reply that was never sent."""
    from main import _DELIVERED_KINDS
    assert _DELIVERED_KINDS == {"reply", "delivery_failed", "interrupted"}
    for absent in ("silence", "reaction_only", "queued", "empty", "aborted", "error",
                   "detached", "none"):
        assert absent not in _DELIVERED_KINDS


def test_a_structural_destination_cannot_be_overwritten_by_the_tool():
    """The tool is not offered on a settled route — but the registry checks `enabled` when it
    builds the schema set, not again at dispatch. A call that arrives anyway is refused by the
    state rather than silently relocating a DM's reply."""
    for message, allowed in ((_message(channel="D1"), True),      # DM
                             (_message(ts="11.0"), True),         # inside a thread
                             (_message(), False)):                # top-level, posting forbidden
        turn = TurnRuntime.for_message(message, channel_post_allowed=allowed)
        before = turn.reply_destination
        result = turn.select_destination("channel", message=message)
        assert result["ok"] is False and result["error"] == "destination_not_open"
        assert turn.reply_destination == before


def test_a_posted_notice_settles_and_locks_the_thread():
    """A prior-timeout or failed-files notice posts BEFORE the model runs, into the thread. If
    the turn stayed open, the model could send the answer to the channel top level and split one
    turn across two surfaces."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    assert turn.destination_selected is False          # open, until the notice goes out

    turn.settle_structural_thread()

    assert turn.reply_destination == "thread"
    assert turn.destination_source == "structural"     # the route decided, not a defaulting model
    assert turn.destination_contract_miss is False     # …so it is not a contract miss
    assert turn.destination_selected is True and turn.destination_locked is True
    refused = turn.select_destination("channel", message=message)
    assert refused["ok"] is False


def test_settling_a_notice_never_moves_a_reply_that_is_already_out():
    """Locked means locked, whichever way the change comes from."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    turn.select_destination("channel", message=message)
    turn.lock_destination()
    turn.settle_structural_thread()
    assert turn.reply_destination == "channel"


# ======================================================== the streaming lifecycle (brief §3)
#
# The hard part: an eligible turn must create NO surface anywhere until the destination tool
# round has completed, then bind at the first word of the answer. Driven through the REAL
# streaming handler with a fake Slack, because the failure mode is a message appearing in the
# wrong place — which only shows up end to end.

from tests.unit.test_reply_surface import (  # noqa: E402 — the harness this reuses lives there
    FakeOpenAI, FakeSlack, _message as _surface_message, _processor, _run, _thread_state)


class _ChoosingOpenAI(FakeOpenAI):
    """Streams the destination tool call in round one (no visible text), then the answer.

    That is the real shape: the tool round completes silently, so by the time the first word of
    the answer arrives the destination is already settled."""

    def __init__(self, destination, chunks, turn, message):
        super().__init__(chunks)
        self._destination = destination
        self._turn = turn
        self._message = message

    async def _run(self, stream_callback, **kw):
        if self._destination is not None and self._turn is not None:
            self._turn.select_destination(self._destination, message=self._message)
            self._destination = None            # one tool round, like the contract asks
        return await super()._run(stream_callback, **kw)


@pytest.mark.asyncio
async def test_an_eligible_turn_that_chose_thread_streams_into_the_thread(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_ChoosingOpenAI("thread", ["Postgres ", "defaults."], turn, msg))

    await _run(processor, slack, msg, state, turn)

    assert turn.reply_destination == "thread" and turn.destination_source == "model"
    assert len(slack.streams) == 1, "a thread reply keeps its live reveal"
    assert slack.edits == []


@pytest.mark.asyncio
async def test_an_eligible_turn_that_chose_channel_posts_once_and_never_edits(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_ChoosingOpenAI("channel", ["Short ", "answer."], turn, msg))

    resp = await _run(processor, slack, msg, state, turn)

    assert turn.reply_destination == "channel"
    assert slack.streams == [], "Slack cannot stream top-level; we must not have tried"
    assert slack.edits == [], "an edit would brand the answer (edited) forever"
    assert len(slack.posts) == 1, f"expected exactly one finished post, got {slack.calls}"
    assert resp.metadata.get("posted") is True


@pytest.mark.asyncio
async def test_an_eligible_turn_shows_nothing_before_the_choice(monkeypatch):
    """The regression this whole lifecycle exists to prevent: a placeholder or a status in one
    place while the answer goes to the other."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    assert turn.progress_enabled is False                      # main.py mints no indicator
    processor = _processor(_ChoosingOpenAI("channel", ["hi"], turn, msg))

    await _run(processor, slack, msg, state, turn)

    assert [c for c in slack.calls if c[0] == "setStatus"] == []
    # Exactly one message total, and it is the finished answer — nothing was ever seeded.
    assert len(slack.live) == 1


@pytest.mark.asyncio
async def test_an_answer_with_no_choice_still_lands_in_the_default_thread(monkeypatch):
    """Never drop a valid answer over a missed tool call — and never guess from its text."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_ChoosingOpenAI(None, ["I just started talking."], turn, msg))

    await _run(processor, slack, msg, state, turn)

    assert turn.reply_destination == "thread"
    assert turn.destination_source == "default"
    assert turn.destination_contract_miss is True
    assert len(slack.streams) == 1                 # delivered, in the thread
    assert "".join(slack.live.values()).strip()


@pytest.mark.asyncio
async def test_the_destination_is_locked_once_the_reply_exists(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_ChoosingOpenAI("thread", ["done"], turn, msg))

    await _run(processor, slack, msg, state, turn)

    assert turn.destination_locked is True
    late = turn.select_destination("channel", message=msg)
    assert late["ok"] is False and late["error"] == "destination_locked"


@pytest.mark.asyncio
async def test_a_retry_never_resets_a_model_selected_destination(monkeypatch):
    """The MCP fallback re-enters the handler with the SAME TurnRuntime. If the retry re-opened
    the question, an answer could change places halfway through being delivered."""
    from config import config
    from tests.unit.test_reply_surface import MCP_ERROR
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    openai = _ChoosingOpenAI("channel", ["the answer"], turn, msg)
    openai.error = MCP_ERROR
    processor = _processor(openai)

    await _run(processor, slack, msg, state, turn)

    assert openai.attempts == 2, "the MCP failure should have retried"
    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"
    assert slack.edits == []


# ------------------------------------------------- the free tool costs nothing, on BOTH paths

def test_both_tool_loops_are_told_the_destination_tool_is_free():
    """It was free in the streaming loop only. The non-streaming loop charged every call, so a
    bookkeeping call there spent a slot a real tool needed — or displaced a sibling beside
    no_response_needed."""
    import inspect
    from message_processor.handlers.text import TextHandlerMixin
    for method in (TextHandlerMixin._handle_text_response,
                   TextHandlerMixin._handle_streaming_text_response):
        src = inspect.getsource(method)
        assert "free_tools=(SET_REPLY_DESTINATION,)" in src, method.__name__


@pytest.mark.asyncio
async def test_the_non_streaming_loop_charges_no_budget_for_a_free_call(monkeypatch):
    """A round of pure bookkeeping costs no round and no productive call — same rule the
    streaming loop already applied."""
    from openai_client.api import tool_loop

    scripts = [[{"type": "function_call", "name": SET_REPLY_DESTINATION, "call_id": "1",
                 "arguments": '{"destination": "channel"}'}],
               [{"type": "function_call", "name": "remember_fact", "call_id": "2",
                 "arguments": "{}"}],
               []]
    state = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(scripts[min(i, len(scripts) - 1)])
        return {"text": "" if i < 2 else "the answer", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    # A cap of ONE productive call: the bookkeeping round must not consume it, or the memory
    # write in the next round never runs.
    monkeypatch.setattr(tool_loop.config, "max_tool_calls_per_turn", 1)
    dispatched = []

    class _Registry:
        async def dispatch_all(self, ctx, calls):
            dispatched.extend(c.get("name") for c in calls)
            return [{"ok": True} for _ in calls]

    result = await tool_loop.create_text_response_with_tool_loop(
        SimpleNamespace(log_info=lambda *a, **k: None, log_debug=lambda *a, **k: None,
                        log_warning=lambda *a, **k: None, log_error=lambda *a, **k: None),
        messages=[], tools=[], registry=_Registry(), tool_context=None,
        free_tools=(SET_REPLY_DESTINATION,))

    assert dispatched == [SET_REPLY_DESTINATION, "remember_fact"]
    assert result["text"] == "the answer"


@pytest.mark.asyncio
async def test_a_free_sibling_never_displaces_a_real_one_beside_the_terminal(monkeypatch):
    """The non-streaming terminal path got no free-tool set either, so a bookkeeping call could
    take the single remaining slot away from the memory write the model also asked for."""
    from openai_client.api import tool_loop

    calls = [{"type": "function_call", "name": "no_response_needed", "call_id": "1",
              "arguments": '{"reason": "nothing_to_add"}'},
             {"type": "function_call", "name": SET_REPLY_DESTINATION, "call_id": "2",
              "arguments": '{"destination": "thread"}'},
             {"type": "function_call", "name": "remember_fact", "call_id": "3",
              "arguments": "{}"}]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        if function_call_sink is not None:
            function_call_sink.extend(calls)
        return {"text": "", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    monkeypatch.setattr(tool_loop.config, "max_tool_calls_per_turn", 2)  # terminal + one more
    dispatched = []

    class _Registry:
        async def dispatch_all(self, ctx, calls_):
            dispatched.extend(c.get("name") for c in calls_)
            return [{"ok": True} for _ in calls_]

    result = await tool_loop.create_text_response_with_tool_loop(
        SimpleNamespace(log_info=lambda *a, **k: None, log_debug=lambda *a, **k: None,
                        log_warning=lambda *a, **k: None, log_error=lambda *a, **k: None),
        messages=[], tools=[], registry=_Registry(), tool_context=None,
        free_tools=(SET_REPLY_DESTINATION,))

    assert result["terminal_action"] == "no_reply"
    assert "remember_fact" in dispatched, "the productive sibling kept its slot"
    assert SET_REPLY_DESTINATION in dispatched, "…and the free one ran on its own allowance"


# ------------------------------------------------------------------ locking on delivery

@pytest.mark.asyncio
async def test_a_non_streamed_reply_locks_its_destination_when_it_goes_out():
    """The streaming paths lock when they bind a surface; a non-streamed reply is posted by
    main.py, so that send is the same moment. Without it `destination_locked` would be false on
    a turn whose answer is already in the room."""
    from base_client import Message, Response
    from main import ChatBotV2

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = MagicMock(spec=[])
    captured = {}

    async def _process(message, client, thinking_id=None, turn=None):
        captured["turn"] = turn
        return Response(type="text", content="here you go",
                        metadata={"streamed": False, "model": "m"})

    app.processor.process_message = AsyncMock(side_effect=_process)
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock(return_value="99.9")
    client.format_text = lambda t: t
    client.maybe_post_response_footer = AsyncMock()
    client.clear_assistant_status = AsyncMock()

    await app.handle_message(
        Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                metadata={"ts": "10.0", "channel_post_allowed": True}), client)

    turn = captured["turn"]
    assert turn.destination_locked is True
    late = turn.select_destination("channel", message=None)
    assert late["ok"] is False and late["error"] == "destination_locked"


# ------------------------------------------- a notice settles the destination only if it LANDS

def _notice_processor():
    """The REAL process_message on a bare harness, driven far enough to reach the notices."""
    from message_processor.base import MessageProcessor

    class _Proc:
        process_message = MessageProcessor.process_message
        # The real notice helper, so this still exercises the send it is about.
        _post_prior_timeout_notice = MessageProcessor._post_prior_timeout_notice

        def __init__(self):
            from thread_manager import AsyncThreadStateManager
            self.thread_manager = AsyncThreadStateManager(db=None)
            self.db = None

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

    return _Proc()


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_ts", ["99.9", None])
async def test_a_channel_prior_timeout_notice_settles_before_the_request_is_measured(notice_ts):
    """[r3-3] A CHANNEL turn that owes this notice settles the destination BEFORE assembly, whether
    or not the notice then lands.

    It used to settle only on a confirmed send, which is the right rule when the settle FOLLOWS the
    send. Deferring the notice past admission (r2-11) inverted that: admission now pins the tool
    tuple and the suffix, so a settle afterwards left the measured request exposing
    `set_reply_destination` and saying nothing about the destination, while the request actually
    sent said `reply_destination=thread` and refused that tool at runtime — different bytes, and a
    prompt that contradicted its own lock.

    The cost of settling early is that a notice Slack drops still forces the reply into the thread.
    That is a reply in a slightly less convenient place; the alternative was an estimate that
    measured a request nobody sent."""
    import contextlib
    from base_client import Message

    proc = _notice_processor()
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "channel_post_allowed": True})
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    assert turn.destination_selected is False

    client = MagicMock()
    client.send_message = AsyncMock(return_value=notice_ts)

    async def _state(*a, **k):
        return SimpleNamespace(had_timeout=True, messages=[], channel_id="C1",
                               thread_ts="10.0", root_author=("U1", "human"),
                               config_overrides={}, current_model="gpt-5.6-sol",
                               participants={}, has_trimmed_messages=False)

    # A channel turn CREATES its state rather than rebuilding one, so both entry points answer
    # with the timed-out thread — the notice fires before either path is left behind.
    proc._get_or_rebuild_thread_state = _state
    proc.get_or_create_channel_thread_state = _state
    # r2-11: the notice now waits for the window and the admission, so the two steps in front of
    # it have to succeed for this test to reach it.
    proc._build_channel_turn_stream = AsyncMock(return_value=None)
    proc._admit_channel_request = AsyncMock()
    proc._process_attachments = AsyncMock(return_value=([], [], []))
    proc._build_user_content = MagicMock(return_value="q")
    with contextlib.suppress(Exception):   # the bare harness dies past the notice
        await proc.process_message(message, client=client, thinking_id=None, turn=turn)

    client.send_message.assert_awaited()                     # the notice was attempted
    assert turn.destination_selected is True
    assert turn.destination_locked is True
    assert turn.reply_destination == "thread"
    assert turn.destination_source == "structural"


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_ts,expected", [("99.9", "info"), (None, "warning")])
async def test_a_prior_timeout_notice_is_logged_as_what_it_did(notice_ts, expected):
    """[r4-7] `send_message` swallows a SlackApiError and returns None, and the line said "Notified
    user" either way — so the one path where nobody is told anything read exactly like the one where
    they were."""
    from base_client import Message

    proc = _notice_processor()
    logged = []
    proc.log_info = lambda msg, *a, **k: logged.append(("info", msg))
    proc.log_warning = lambda msg, *a, **k: logged.append(("warning", msg))
    client = MagicMock()
    client.send_message = AsyncMock(return_value=notice_ts)
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})

    await proc._post_prior_timeout_notice(message, client, None, "C1:10.0")

    assert [level for level, _ in logged] == [expected], logged
    assert "C1:10.0" in logged[0][1]
