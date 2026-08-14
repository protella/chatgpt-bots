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
                                                 DestinationMarkerReader,
                                                 consume_destination_marker,
                                                 destination_marker,
                                                 destination_tool_available,
                                                 execute_set_reply_destination,
                                                 get_set_reply_destination_schema,
                                                 parse_destination_marker)
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
    from message_processor.prompts import DESTINATION_CONTRACT_SUFFIX
    from message_processor.tool_registry import ToolRegistry

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
    from message_processor.prompts import DESTINATION_CONTRACT_SUFFIX as s
    assert "begin your reply with exactly" in s
    assert "If it is a close call, choose thread" in s


def test_the_contract_paragraph_spells_the_markers_the_parser_accepts():
    """The prompt and the parser have to agree on the literal, character for character — a
    paragraph teaching a grammar nothing accepts is a contract miss on every single turn."""
    from message_processor.prompts import DESTINATION_CONTRACT_SUFFIX as s
    for destination in SELECTABLE_DESTINATIONS:
        marker = destination_marker(destination)
        assert marker in s
        assert parse_destination_marker(f"{marker} x")[0] == destination
    # …and it says the marker is not part of the message, since the model can't see it removed.
    assert "removed before anyone sees the message" in s


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


def _marked(destination, *chunks):
    """The model's own first tokens: the marker, then the answer. W4's whole shape — no tool
    round, no extra call, the choice riding the reply that is already being written."""
    return FakeOpenAI([destination_marker(destination) + "\n\n", *chunks])


@pytest.mark.asyncio
async def test_an_eligible_turn_that_marked_thread_streams_into_the_thread(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_marked("thread", "Postgres ", "defaults."))

    await _run(processor, slack, msg, state, turn)

    assert turn.reply_destination == "thread" and turn.destination_source == "model"
    assert len(slack.streams) == 1, "a thread reply keeps its live reveal"
    assert slack.edits == []


@pytest.mark.asyncio
async def test_an_eligible_turn_that_marked_channel_posts_once_and_never_edits(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_marked("channel", "Short ", "answer."))

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
    processor = _processor(_marked("channel", "hi"))

    await _run(processor, slack, msg, state, turn)

    assert [c for c in slack.calls if c[0] == "setStatus"] == []
    # Exactly one message total, and it is the finished answer — nothing was ever seeded.
    assert len(slack.live) == 1


@pytest.mark.asyncio
async def test_an_answer_with_no_marker_still_lands_in_the_default_thread(monkeypatch):
    """Never drop a valid answer over a missed marker — and never guess from its text.

    W4 changed the SHAPE of this delivery and not its outcome: with no marker the destination is
    never selected, so the surface stays unbound and the words buffer to the end, then post once
    into the default thread. There is no earlier moment at which binding would be honest."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(FakeOpenAI(["I just started talking."]))

    await _run(processor, slack, msg, state, turn)

    assert turn.reply_destination == "thread"
    assert turn.destination_source == "default"
    assert turn.destination_contract_miss is True
    assert len(slack.posts) == 1                   # delivered, in the thread
    assert "I just started talking." in "".join(slack.live.values())


@pytest.mark.asyncio
async def test_the_destination_is_locked_once_the_reply_exists(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_marked("thread", "done"))

    await _run(processor, slack, msg, state, turn)

    assert turn.destination_locked is True
    late = turn.select_destination("channel", message=msg)
    assert late["ok"] is False and late["error"] == "destination_locked"


@pytest.mark.asyncio
async def test_a_retry_never_resets_a_model_selected_destination(monkeypatch):
    """The MCP fallback re-enters the handler with the SAME TurnRuntime. If the retry re-opened
    the question, an answer could change places halfway through being delivered — and the retry
    re-emits the marker, so the refusal has to come from the locked turn."""
    from config import config
    from tests.unit.test_reply_surface import MCP_ERROR
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    openai = _marked("channel", "the answer")
    openai.error = MCP_ERROR
    processor = _processor(openai)

    await _run(processor, slack, msg, state, turn)

    assert openai.attempts == 2, "the MCP failure should have retried"
    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"
    assert slack.edits == []


# ======================================================== W4: the first-token marker
#
# The tool round is gone. What replaces it is a marker the model writes into its own opening
# tokens, one parser that reads it, and one rule that parser applies everywhere. These tests hold
# the rule, the stripping, and the two things the rule exists to make impossible: a preamble read
# as a missing marker, and a marker a reader can see.

# ---------------------------------------------------------------------- the rule, one parser

def test_a_leading_marker_selects_and_disappears():
    assert parse_destination_marker("[[reply:channel]]\n\nShort answer.") == (
        "channel", "Short answer.")
    assert parse_destination_marker("[[reply:thread]] Long answer.") == ("thread", "Long answer.")


def test_a_marker_after_a_preamble_still_selects():
    """The rule is ANYWHERE, not leading. "Leading" is what the prompt asks of the model; a
    parser that required it would turn every preamble into a contract miss, and today's
    aggregation can put preamble text at the head of the canonical text."""
    assert parse_destination_marker("Let me look. [[reply:thread]] Here you go.") == (
        "thread", "Let me look. Here you go.")


def test_the_first_of_several_markers_wins_and_all_of_them_go():
    """A model that restates its choice has not made two of them. The first is the answer; the
    rest are text nobody may see."""
    destination, cleaned = parse_destination_marker(
        "[[reply:thread]] one [[reply:channel]] two [[reply:thread]] three")
    assert destination == "thread"
    assert cleaned == "one two three"
    assert "[[reply:" not in cleaned


def test_text_without_a_marker_comes_back_untouched():
    """The identity that keeps every non-selectable turn byte-identical: the parser runs on their
    canonical text too, and must not so much as trim it."""
    for text in ("", "plain answer", "  spaced  ", "brackets [like] this", None):
        destination, cleaned = parse_destination_marker(text)
        assert destination is None
        assert cleaned == (text or "")


def test_a_near_miss_is_not_a_marker():
    """The grammar is closed. An unrecognized destination is not a guess at what was meant, and
    it is not stripped either — mangling the model's prose to hide a typo helps nobody."""
    for text in ("[[reply:dm]] hi", "[[reply:Thread]] hi", "[reply:thread] hi", "[[reply]] hi"):
        assert parse_destination_marker(text) == (None, text)


# ----------------------------------------------------------- the rule, arriving in fragments

def _read(chunks, turn=None, message=None):
    reader = DestinationMarkerReader(turn, message)
    out = [reader.feed(c) for c in chunks]
    out.append(reader.flush())
    return reader, "".join(out)


def test_a_marker_split_across_chunks_is_still_one_marker():
    reader, text = _read(["[[re", "ply:cha", "nnel]]", "\n\nHel", "lo"])
    assert reader.destination == "channel"
    assert text == "Hello"
    assert "[[" not in text


@pytest.mark.parametrize("chunks", [
    ["[[reply:thread]] Here you go."],
    ["[[reply:thread]]", " Here you go."],
    ["[[reply:thread]]", " ", "Here ", "you go."],
    ["[[reply:thr", "ead]]\n", "\nHere you go."],
    ["Let me look. ", "[[reply:thread]] ", "Here you go."],
])
def test_the_stream_and_the_whole_string_agree(chunks):
    """The property that makes one parser genuinely one parser: whatever the chunk boundaries
    are, the reader emits exactly what a single parse of the joined text would have."""
    reader, streamed = _read(chunks)
    whole_destination, whole_text = parse_destination_marker("".join(chunks))
    assert reader.destination == whole_destination
    assert streamed == whole_text


def test_the_reader_holds_a_partial_marker_rather_than_showing_it():
    """Nothing partial ever goes downstream: a lone "[[" on screen for one tick is a marker the
    reader can see, which is the thing that must never happen."""
    reader = DestinationMarkerReader(None)
    assert reader.feed("[[reply:th") == ""
    assert reader.feed("read]] go") == "go"


def test_the_reader_selects_on_the_chunk_that_completed_the_marker():
    """Not the chunk after. The whole point is to bind the surface as early as the answer
    honestly allows, and a one-chunk delay is a round trip of dead air."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    reader = DestinationMarkerReader(turn, message)
    reader.feed("[[reply:channel]]")
    assert reader.destination == "channel"
    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"


def test_no_marker_is_never_a_verdict_before_the_end():
    """MISSING is a terminal verdict and nothing else. Mid-stream, "no marker yet" means the
    model is still writing — defaulting there is how a preamble would steal the choice."""
    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    reader = DestinationMarkerReader(turn, message)
    reader.feed("Let me check that for you. ")
    assert reader.destination is None
    assert turn.destination_selected is False          # nothing decided, nothing recorded
    assert turn.destination_contract_miss is False
    reader.flush()
    assert turn.destination_selected is False          # the reader never renders the verdict…
    turn.settle_default_destination()                  # …the terminal does
    assert turn.reply_destination == "thread"
    assert turn.destination_contract_miss is True


def test_the_choke_point_strips_even_where_nothing_may_be_chosen():
    """A turn with no choice still must not leak a marker into the room. Selection is refused;
    the strip is not conditional on it."""
    message = _message(channel="D1")
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)   # a DM: structural
    cleaned = consume_destination_marker("[[reply:channel]] hi", turn=turn, message=message)
    assert cleaned == "hi"
    assert turn.reply_destination == "dm" and turn.destination_source == "structural"


# ------------------------------------------------------- the tool is retired, statically

def test_the_channel_registry_does_not_carry_the_tool_and_the_dm_one_still_does():
    """Retirement is at REGISTRY CONSTRUCTION, not per turn: per-request `enabled` predicates are
    structurally ignored on the channel surface, so a per-turn gate there would have been a lie —
    and a schema set that varies within a channel is a cache fork."""
    from message_processor.destination_tools import register_destination_tools
    from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM, ToolRegistry

    registry = ToolRegistry()
    register_destination_tools(registry)
    open_cfg = {"_destination_choice_open": True}

    def names(cfg, surface):
        return {s["name"] for s in registry.schemas(cfg, surface=surface)}

    assert names(open_cfg, SURFACE_CHANNEL) == set()
    assert names({}, SURFACE_CHANNEL) == set()          # …on every config, not just this one
    assert SET_REPLY_DESTINATION in names(open_cfg, SURFACE_DM)
    assert SET_REPLY_DESTINATION not in names({}, SURFACE_DM)


def test_the_marker_paragraph_rides_a_turn_with_no_tool_to_call(mock_env):
    """The one contract paragraph here that is NOT keyed to a schema. It asks for text, and the
    channel surface — the only surface that ever has a destination to choose — has no tool."""
    from message_processor.destination_tools import register_destination_tools
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.prompts import DESTINATION_CONTRACT_SUFFIX
    from message_processor.tool_registry import SURFACE_CHANNEL, ToolRegistry

    registry = ToolRegistry()
    register_destination_tools(registry)
    registry.register({"type": "function", "name": "react_to_message", "parameters": {}},
                      AsyncMock())
    host = SimpleNamespace()
    for name in ("_materialize_request_tools", "_get_tool_registry"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host))
    host._client = SimpleNamespace(tool_registry=registry)

    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    reg, request_config, _, suffix = host._materialize_request_tools(
        host._client, {"model": "m"}, message, tools_disabled=False, turn=turn,
        surface=SURFACE_CHANNEL)

    exposed = {s["name"] for s in reg.schemas(request_config, surface=SURFACE_CHANNEL)}
    assert SET_REPLY_DESTINATION not in exposed
    assert suffix == DESTINATION_CONTRACT_SUFFIX


# ------------------------------------------------- the marker never survives the turn (4 sinks)

def _stored_assistant_texts(processor):
    """Everything the handler wrote into the thread state — sink 2, and the only thing thread
    compaction can ever be handed."""
    return [call.args[2] for call in
            processor._add_message_with_token_management.call_args_list
            if call.args[1] == "assistant"]


@pytest.mark.asyncio
async def test_no_marker_reaches_slack_on_either_destination(monkeypatch):
    """Sink 1. Every message the turn wrote, at every moment it wrote one — `history` keeps the
    mid-stream writes a final correction would otherwise cover for."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    for destination in SELECTABLE_DESTINATIONS:
        slack = FakeSlack(native=True)
        msg, state = _surface_message(), _thread_state()
        turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
        processor = _processor(_marked(destination, "The ", "answer."))

        resp = await _run(processor, slack, msg, state, turn)

        written = [text for _ts, text in slack.history]
        assert written, f"{destination}: nothing was written at all"
        for text in written:
            assert "[[reply:" not in text, f"{destination}: a marker reached Slack: {text!r}"
        assert "[[reply:" not in (resp.content or "")
        assert "The answer." in "".join(slack.live.values())


@pytest.mark.asyncio
async def test_no_marker_reaches_the_finalized_message_of_a_structural_turn(monkeypatch):
    """Sink 1, on the route that has nothing to choose. A DM or an in-thread reply is never
    offered a destination, so a marker there means nothing — but "means nothing" is not "is
    invisible". The native finalize commits the BUFFER, not the cleaned canonical text, so a
    stray marker that reached the buffer would sit in the finished Slack message forever.

    Stripping and selecting are separable, and only selecting is about the route: the reader
    runs on every streamed turn and `select_destination` refuses on its own."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)   # structural thread
    processor = _processor(FakeOpenAI(["[[reply:channel]] Threaded ", "answer."]))

    await _run(processor, slack, msg, state, turn)

    assert slack.streams, "this turn should have streamed natively"
    assert all("[[reply:" not in text for _ts, text in slack.history), slack.history
    finalized = "".join(slack.live.values())
    assert "Threaded answer." in finalized and "[[reply:" not in finalized
    # …and the stray marker moved nothing: this route was never open.
    assert turn.reply_destination == "thread"
    assert turn.destination_source == "structural"


@pytest.mark.asyncio
async def test_no_marker_reaches_slack_on_an_mcp_retry_of_a_chosen_turn(monkeypatch):
    """Sink 1, the attempt that nearly got away. An MCP failover re-enters the handler with the
    SAME turn, whose destination is already chosen — and the retry's model writes the marker
    again. The reader has to exist on that attempt too, because the native finalize commits the
    BUFFER, so a marker that streams into it is a marker that stays in the message."""
    from config import config
    from tests.unit.test_reply_surface import MCP_ERROR
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    openai = _marked("thread", "First try.")
    openai.error = MCP_ERROR
    openai.retry_chunks = ["[[reply:thread]] Second try."]
    processor = _processor(openai)

    await _run(processor, slack, msg, state, turn)

    assert openai.attempts == 2
    assert all("[[reply:" not in text for _ts, text in slack.history), slack.history
    assert "Second try." in "".join(slack.live.values())


@pytest.mark.asyncio
async def test_no_marker_reaches_the_thread_state_of_a_dm_turn(monkeypatch):
    """Sink 2. A DM stores its own assistant turn (a channel turn deliberately stores none — its
    words come back from Slack), and that stored text is what the next request replays."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg = _surface_message(channel="D1")
    state = _thread_state(channel="D1")
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)
    processor = _processor(FakeOpenAI(["[[reply:channel]] Stray ", "marker."]))

    await _run(processor, slack, msg, state, turn)

    stored = _stored_assistant_texts(processor)
    assert stored, "the DM turn stored nothing at all"
    assert all("[[reply:" not in text for text in stored), stored
    assert "Stray marker." in stored[0]
    # …and the stray marker moved nothing: a DM's destination was never open.
    assert turn.reply_destination == "dm" and turn.destination_source == "structural"


@pytest.mark.asyncio
async def test_no_marker_reaches_reconsideration_or_the_draft_it_returns(monkeypatch):
    """Sink 3, both directions. The draft handed to the reconsideration runner is quoted verbatim
    into a model request, and the revised text it hands back becomes the message — so the choke
    point has to sit on both sides of that call."""
    from config import config
    from message_processor.handlers import text as text_module
    from message_processor.stale_send_guard import StaleSendSuppressed
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    class _RefusingOnce(FakeSlack):
        def __init__(self):
            super().__init__(native=True)
            self.refused = False

        async def send_message(self, *a, **kw):
            if not self.refused:
                self.refused = True
                raise StaleSendSuppressed(surface="final_post")
            return await super().send_message(*a, **kw)

    seen = {}

    async def _capture(*, draft, deliver, **kw):
        seen["draft"] = draft
        # The runner's revised text is fresh model output and may carry a marker of its own.
        return await deliver("[[reply:thread]] The revised answer.")

    monkeypatch.setattr(text_module, "intercept_stale_send", _capture)

    slack = _RefusingOnce()
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    processor = _processor(_marked("channel", "The first answer."))

    resp = await _run(processor, slack, msg, state, turn)

    assert "[[reply:" not in seen["draft"], seen["draft"]
    assert "The first answer." in seen["draft"]
    assert "[[reply:" not in (resp.content or "")
    assert all("[[reply:" not in text for _ts, text in slack.history), slack.history
    assert "The revised answer." in "".join(slack.live.values())
    # The revised draft's marker changed nothing: the destination was locked when it bound.
    assert turn.reply_destination == "channel"


@pytest.mark.asyncio
async def test_no_marker_reaches_a_non_streamed_reconsidered_reply(monkeypatch):
    """Sink 3, on main.py's own delivery. A non-streamed reply is posted by main.py, so the
    reconsideration runner's revised text lands in `response.content` THERE — and from there it
    goes to Slack, to F7 persistence, and into the history the next turn rebuilds. It is fresh
    model output and can mint a marker the original draft never had."""
    import main as main_module
    from message_processor.client_contract import Message, Response
    from main import ChatBotV2
    from message_processor.stale_send_guard import StaleSendSuppressed

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = MagicMock(spec=[])
    captured = {}

    async def _process(message, client, thinking_id=None, turn=None):
        captured["turn"] = turn
        return Response(type="text", content="The first answer.",
                        metadata={"streamed": False, "model": "m"})

    app.processor.process_message = AsyncMock(side_effect=_process)

    posted = []

    async def _send(channel, thread, text, **kw):
        if not posted:
            posted.append(None)                      # the first attempt is refused, not failed
            raise StaleSendSuppressed(surface="final_post")
        posted.append(text)
        if kw.get("on_first_accept"):
            kw["on_first_accept"]("99.9")
        return "99.9"

    async def _capture(*, draft, deliver, **kw):
        captured["draft"] = draft
        return await deliver("[[reply:channel]] The revised answer.")

    monkeypatch.setattr(main_module, "intercept_stale_send", _capture)

    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock(side_effect=_send)
    client.format_text = lambda t: t
    client.maybe_post_response_footer = AsyncMock()
    client.clear_assistant_status = AsyncMock()

    await app.handle_message(
        Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                metadata={"ts": "10.0", "channel_post_allowed": True}), client)

    delivered = [t for t in posted if t]
    assert delivered, "the reconsidered reply never went out"
    assert all("[[reply:" not in text for text in delivered), delivered
    assert "The revised answer." in delivered[0]
    assert "[[reply:" not in captured["draft"]


@pytest.mark.asyncio
async def test_no_marker_reaches_the_compaction_summary_input(monkeypatch):
    """Sink 4. Compaction folds the stored turns into a rolling summary through the utility
    model — the last place a marker could be laundered back into the model's own memory of what
    it said."""
    from config import config
    from message_processor.thread_management import ThreadManagementMixin
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    slack = FakeSlack(native=True)
    msg = _surface_message(channel="D1")
    state = _thread_state(channel="D1")
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)
    processor = _processor(FakeOpenAI(["[[reply:channel]] What I said."]))
    await _run(processor, slack, msg, state, turn)
    stored = _stored_assistant_texts(processor)[0]

    sent = {}

    class _Summarizer:
        async def create_text_response(self, messages=None, **kw):
            sent["blocks"] = [m["content"] for m in (messages or [])]
            return "a summary"

    class _Compactor(ThreadManagementMixin):
        """The REAL summary writer — anything less would not be testing the sink."""

        def __init__(self):
            self.db = SimpleNamespace(get_thread_summary_async=AsyncMock(return_value=None),
                                      upsert_thread_summary_async=AsyncMock())
            self.openai_client = _Summarizer()

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

    dropped = [{"role": "assistant", "content": stored, "metadata": {"ts": "10.0"}}]
    await _Compactor()._write_thread_summary(
        SimpleNamespace(messages=[], has_summary_head=False), "D1:10.0", dropped)

    assert sent["blocks"], "the summarizer was never called"
    assert all("[[reply:" not in block for block in sent["blocks"]), sent["blocks"]
    assert "What I said." in "\n".join(sent["blocks"])


# ------------------------------------------------------- streaming gating and the contract miss

@pytest.mark.asyncio
async def test_nothing_posts_before_the_marker_arrives(monkeypatch):
    """The surface stays unbound while the model is still choosing. A preamble ahead of the
    marker buffers; it does not mint a message somewhere the answer may not go."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)

    seen_before_marker = {}

    class _Preambling(FakeOpenAI):
        async def _run(self, stream_callback, **kw):
            await stream_callback("Let me check that. ")
            seen_before_marker["calls"] = list(slack.calls)
            seen_before_marker["selected"] = turn.destination_selected
            await stream_callback("[[reply:thread]] Here you go.")
            await stream_callback(None)
            return "Let me check that. [[reply:thread]] Here you go."

    await _run(_processor(_Preambling([])), slack, msg, state, turn)

    assert seen_before_marker["calls"] == [], "a surface existed before the destination did"
    assert seen_before_marker["selected"] is False
    # …and the preamble was not lost: it is part of the answer, wherever the marker sent it.
    assert turn.reply_destination == "thread"
    delivered = "".join(slack.live.values())
    assert "Let me check that." in delivered and "Here you go." in delivered
    assert "[[reply:" not in delivered


@pytest.mark.asyncio
async def test_a_contract_miss_is_recorded_for_the_ledger(monkeypatch):
    """The battery metric. A miss is a PROMPT problem worth counting, never a delivery failure —
    so the answer goes out and the row says the model was asked and did not say."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)

    resp = await _run(_processor(FakeOpenAI(["No marker here."])), slack, msg, state, turn)

    assert turn.destination_contract_miss is True
    assert turn.destination_source == "default"
    assert resp.metadata.get("posted") is True, "a miss must never cost the answer"
    assert "No marker here." in "".join(slack.live.values())


@pytest.mark.asyncio
async def test_a_turn_with_no_words_owes_no_marker(monkeypatch):
    """"Begin your reply with the marker" is a rule about replies. A turn that ends without any
    has nothing to place, and charging it a contract miss would make the metric meaningless."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)
    turn.visible_action_committed = True            # the words went into another thread

    await _run(_processor(FakeOpenAI([])), slack, msg, state, turn)

    assert turn.destination_contract_miss is False
    assert turn.destination_selected is False


@pytest.mark.asyncio
async def test_a_non_selectable_turn_runs_the_callback_exactly_as_before(monkeypatch):
    """Byte-identical: a turn with nothing to choose builds no reader, so no text is held back
    and no chunk boundary moves. Driven through the thread route, which streams live."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)   # structural thread
    processor = _processor(FakeOpenAI(["Post", "gres ", "defaults."]))

    await _run(processor, slack, msg, state, turn)

    assert turn.destination_source == "structural"
    assert turn.destination_contract_miss is False
    assert len(slack.streams) == 1
    # Every chunk reached the surface as it arrived. (The tail repeat is the finalize rewriting
    # the finished text.)
    assert [text for _ts, text in slack.history][:3] == [
        "Post", "Postgres ", "Postgres defaults."]


@pytest.mark.asyncio
async def test_a_held_tail_never_migrates_across_a_local_tool_seam(monkeypatch):
    """THE ROUND BOUNDARY. A successful function-call round deliberately skips the terminal
    flush, so nothing releases the reader's held tail until the NEXT round's text has already
    arrived — and by then the seam is armed, so the held characters would land on the far side
    of the paragraph break.

    That is not a cosmetic slip. The buffer is what the native finalize commits, `join_segments`
    is the canonical text, and a native success never runs the correction that would reconcile
    them — so the room would read `A\\n\\n[B` while the history said `A[\\n\\nB`. The two must be
    the same bytes."""
    from config import config
    from message_processor.message_markers import join_segments
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    segments = ["Checking that.[", "Here you go."]

    class _TwoRounds(FakeOpenAI):
        """Round one's text, a local tool, then round two's — the shape the tool loop drives,
        including the flush it skips between rounds."""

        async def _run(self, stream_callback, **kw):
            self.attempts += 1
            tool_callback = kw.get("tool_callback")
            await stream_callback(segments[0])
            await tool_callback("local:remember_fact", "started")
            await tool_callback("local:remember_fact", "completed")
            await stream_callback(segments[1])
            await stream_callback(None)
            return join_segments(segments)

    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)

    resp = await _run(_processor(_TwoRounds([])), slack, msg, state, turn)

    canonical = join_segments(segments)
    assert canonical == "Checking that.[\n\nHere you go."     # the bracket ends round one
    assert "".join(slack.live.values()) == canonical
    assert resp.content == canonical


def test_rounds_are_parsed_alone_so_a_marker_cannot_eat_a_seam():
    """The seam between two rounds belongs to NEITHER of them, so the parser must never be shown
    it. Handed the joined string, a marker that ended round one swallows the separator that was
    inserted after it — and the streaming reader, flushed at every round boundary, does not. On a
    native turn that disagreement is permanent: the buffer is what Slack keeps."""
    from message_processor.message_markers import join_segments

    segments = ["Checking that.[[reply:thread]]", "Here you go."]
    joined = join_segments(segments)

    # What the naive whole-string parse does, and why it is wrong here.
    assert parse_destination_marker(joined)[1] == "Checking that.Here you go."
    # Per round, the seam survives — and the choice is still made.
    turn = TurnRuntime.for_message(_message(), channel_post_allowed=True)
    assert consume_destination_marker(None, turn=turn, message=_message(),
                                      segments=segments) == "Checking that.\n\nHere you go."
    assert turn.reply_destination == "thread" and turn.destination_source == "model"


def test_a_leading_markers_whitespace_still_vanishes_per_round():
    """The trailing-whitespace rule exists so a reply never opens with a blank line. Parsing per
    round must not cost that: within a round the marker still takes its own separator with it."""
    from message_processor.message_markers import join_segments

    segments = ["[[reply:channel]]\n\nThe short answer.", "And the follow-up."]
    assert consume_destination_marker(None, segments=segments) == join_segments(
        ["The short answer.", "And the follow-up."])
    assert not consume_destination_marker(None, segments=segments).startswith(("\n", " "))


@pytest.mark.asyncio
async def test_a_complete_marker_at_a_round_boundary_keeps_native_and_canonical_equal(monkeypatch):
    """Codex's reproducer, end to end. Round one's text ENDS on the marker and a local tool
    follows, so the marker sits exactly on the seam. All three views of the answer — the bytes
    Slack finalized, the canonical text, and the Response the caller gets — have to be the same
    string, with the paragraph break intact and the destination chosen."""
    from config import config
    from tests.unit.test_reply_surface import FakeToolLoopOpenAI, _processor_tools
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    slack = FakeSlack(native=True)
    # Round one's text ends ON the marker; round two follows the tool.
    openai = FakeToolLoopOpenAI(slack, ["Checking that.[[reply:thread]]"], ["Here you go."],
                                tool="local:remember_fact")
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)

    resp = await _run(_processor_tools(openai), slack, msg, state, turn)

    expected = "Checking that.\n\nHere you go."
    assert "".join(slack.live.values()) == expected, "the bytes Slack finalized"
    assert resp.content == expected, "the canonical text"
    assert all("[[reply:" not in text for _ts, text in slack.history), slack.history
    # The marker still did its job on the way past.
    assert turn.reply_destination == "thread" and turn.destination_source == "model"


@pytest.mark.asyncio
async def test_rebuilding_from_rounds_never_undoes_the_sanitizers(monkeypatch):
    """ORDER. The marker consumer rebuilds the canonical text from the RAW rounds, so anything
    cleaned before it would come back — a dead `sandbox:` link, live again, in the message, the
    thread state and the reconsideration input. The sanitizers run on the rebuilt text, which
    keeps them a single pass over the whole answer and just moves them one step later."""
    from config import config
    from tests.unit.test_reply_surface import FakeToolLoopOpenAI, _processor_tools
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    slack = FakeSlack(native=True)
    openai = FakeToolLoopOpenAI(
        slack,
        ["See [chart](sandbox:/mnt/data/c.png).[[reply:thread]]"],
        ["Done.\n[used tools: code_interpreter]"],
        tool="local:remember_fact")
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)

    resp = await _run(_processor_tools(openai), slack, msg, state, turn)

    # Sanitized: the dead sandbox link is a plain word again and the echoed footer is gone…
    assert "sandbox:/mnt/data" not in resp.content, resp.content
    assert "[used tools:" not in resp.content, resp.content
    # …and the seam the rebuild exists to protect is still there.
    assert resp.content.startswith("See chart.")
    assert "\n\nDone." in resp.content
    assert "[[reply:" not in resp.content
    assert turn.reply_destination == "thread" and turn.destination_source == "model"


def _non_streaming_loop(monkeypatch, rounds):
    """The real non-streaming tool loop over `rounds`, with a local tool between them."""
    from openai_client.api import tool_loop

    state_n = {"n": 0}

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        i = state_n["n"]
        state_n["n"] += 1
        if i == 0 and function_call_sink is not None:
            function_call_sink.append({"type": "function_call", "name": "remember_fact",
                                       "call_id": "1", "arguments": "{}"})
        return {"text": rounds[min(i, len(rounds) - 1)], "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)

    class _Registry:
        async def dispatch_all(self, ctx, calls):
            return [{"ok": True} for _ in calls]

    host = SimpleNamespace(log_info=lambda *a, **k: None, log_debug=lambda *a, **k: None,
                           log_warning=lambda *a, **k: None, log_error=lambda *a, **k: None)

    async def _run_loop(**kw):
        state_n["n"] = 0
        return await tool_loop.create_text_response_with_tool_loop(
            host, messages=[], tools=[], registry=_Registry(), tool_context=None, **kw)

    return _run_loop


@pytest.mark.asyncio
async def test_the_non_streaming_loop_aggregates_only_for_the_caller_that_asks(monkeypatch):
    """Chat wants the whole turn; deep research wants the report and not the "I'll go look…"
    that preceded it. Same opt-in the streaming twin has, and the rounds come back either way."""
    from message_processor.message_markers import join_segments

    rounds = ["[[reply:channel]] Checking.", "Done."]
    run_loop = _non_streaming_loop(monkeypatch, rounds)

    aggregated = await run_loop(aggregate_segments=True)
    assert aggregated["text"] == join_segments(rounds)
    assert aggregated["segments"] == rounds

    # The default — every internal consumer that reads `text` as a finished artifact.
    terminal_only = await run_loop()
    assert terminal_only["text"] == "Done."
    assert terminal_only["segments"] == rounds


@pytest.mark.asyncio
async def test_a_marker_written_before_a_tool_still_places_the_reply(monkeypatch):
    """A marker belongs in the model's opening tokens, which on a tool turn is round ONE — not
    at the head of the joined whole. Parsing per round is what finds it, and the preamble it was
    attached to survives with its seam."""
    from message_processor.message_markers import join_segments

    rounds = ["[[reply:channel]] Checking.", "Done."]
    result = await _non_streaming_loop(monkeypatch, rounds)(aggregate_segments=True)

    message = _message()
    turn = TurnRuntime.for_message(message, channel_post_allowed=True)
    cleaned = consume_destination_marker(result["text"], turn=turn, message=message,
                                         segments=result["segments"])

    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"
    assert turn.destination_contract_miss is False
    assert cleaned == join_segments(["Checking.", "Done."]) == "Checking.\n\nDone."


@pytest.mark.asyncio
async def test_the_non_streamed_turn_keeps_its_preamble_and_its_marked_destination(monkeypatch):
    """Owner decision 2026-08-08, end to end through the real non-streaming handler — the path
    that runs whenever streaming is off or unsupported.

    Two things at once, because they are the same fix: the preamble the model wrote before
    calling a tool is part of the answer the reader gets (it used to be dropped on the floor),
    and the marker it carried still places the reply. The canonicalization is byte-for-byte what
    the streaming path produces for the same rounds."""
    from config import config
    from message_processor.message_markers import join_segments
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)

    rounds = ["[[reply:channel]] Checking that for you.", "Done — it shipped Tuesday."]
    slack = FakeSlack(native=True)
    slack.supports_streaming = lambda: False        # force the non-streaming handler
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=True)

    class _LoopOpenAI:
        async def create_text_response_with_tool_loop(self, aggregate_segments=False, **kw):
            assert aggregate_segments is True, "chat must ask for the whole turn"
            return {"text": join_segments(rounds), "segments": list(rounds),
                    "tools_used": [], "local_tool_calls": []}

    processor = _processor(_LoopOpenAI())
    registry = MagicMock()
    processor._materialize_request_tools = MagicMock(return_value=(registry, {}, False, ""))
    processor._build_tools_array = MagicMock(
        return_value=[{"type": "function", "name": "remember_fact"}])
    processor._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[]))

    async def _noop(*a, **k):
        return None

    processor._prepare_sandbox_tools = _noop

    from tests.unit.channel_turn_harness import pin_channel_turn
    pin_channel_turn(turn, channel_id=msg.channel_id, trigger_ts="10.0",
                     origin_thread_ts=msg.thread_id, trigger_text="hi",
                     prepared=(registry, {}, False, "", None))
    resp = await processor._handle_text_response("hi", state, slack, msg, None, None, turn=turn)

    # The preamble is no longer lost, and the seam is the one join_segments produces.
    assert resp.content == "Checking that for you.\n\nDone — it shipped Tuesday."
    assert "[[reply:" not in resp.content
    # …and the marker in that same round placed the reply, with no miss recorded.
    assert turn.reply_destination == "channel"
    assert turn.destination_source == "model"
    assert turn.destination_contract_miss is False, "the model chose; this is no miss"


@pytest.mark.asyncio
async def test_the_streaming_loop_hands_back_the_rounds_it_joined(monkeypatch):
    """The contract the fix rests on: the loop returns its ROUNDS beside the joined text, so a
    caller that has to transform the text can do it before the seams exist. Absent unless the
    caller asked for the aggregate — a final-round-only consumer must never be handed rounds its
    `text` does not describe."""
    from message_processor.message_markers import join_segments
    from openai_client.api import tool_loop

    rounds = [["Checking that.[[reply:thread]]"], ["Here you go."]]
    state = {"n": 0}

    async def fake_stream(self, messages, tools, stream_callback=None, function_call_sink=None,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        chunks = rounds[min(i, len(rounds) - 1)]
        for c in chunks:
            await stream_callback(c)
        if i == 0 and function_call_sink is not None:
            function_call_sink.append({"type": "function_call", "name": "remember_fact",
                                       "call_id": "1", "arguments": "{}"})
        return "".join(chunks)

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_stream)

    class _Registry:
        async def dispatch_all(self, ctx, calls):
            return [{"ok": True} for _ in calls]

    host = SimpleNamespace(log_info=lambda *a, **k: None, log_debug=lambda *a, **k: None,
                           log_warning=lambda *a, **k: None, log_error=lambda *a, **k: None)

    result = await tool_loop.create_streaming_response_with_tool_loop(
        host, messages=[], tools=[], registry=_Registry(), tool_context=None,
        stream_callback=AsyncMock(), tool_callback=AsyncMock(), aggregate_segments=True)

    assert result["segments"] == ["Checking that.[[reply:thread]]", "Here you go."]
    assert result["text"] == join_segments(result["segments"])


@pytest.mark.asyncio
async def test_both_loops_canonicalize_the_same_rounds_identically(monkeypatch):
    """The point of the owner's change: which loop ran is no longer visible in the answer. Same
    rounds in, same canonical text and same rounds out — so the non-streamed reply reads exactly
    as the streamed one would have."""
    from openai_client.api import tool_loop

    rounds = ["Checking that.[[reply:thread]]", "Here you go."]
    non_streaming = await _non_streaming_loop(monkeypatch, rounds)(aggregate_segments=True)

    state = {"n": 0}

    async def fake_stream(self, messages, tools, stream_callback=None, function_call_sink=None,
                          tool_choice=None, **kw):
        i = state["n"]
        state["n"] += 1
        await stream_callback(rounds[min(i, len(rounds) - 1)])
        if i == 0 and function_call_sink is not None:
            function_call_sink.append({"type": "function_call", "name": "remember_fact",
                                       "call_id": "1", "arguments": "{}"})
        return rounds[min(i, len(rounds) - 1)]

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                        fake_stream)

    class _Registry:
        async def dispatch_all(self, ctx, calls):
            return [{"ok": True} for _ in calls]

    host = SimpleNamespace(log_info=lambda *a, **k: None, log_debug=lambda *a, **k: None,
                           log_warning=lambda *a, **k: None, log_error=lambda *a, **k: None)
    streaming = await tool_loop.create_streaming_response_with_tool_loop(
        host, messages=[], tools=[], registry=_Registry(), tool_context=None,
        stream_callback=AsyncMock(), tool_callback=AsyncMock(), aggregate_segments=True)

    assert non_streaming["text"] == streaming["text"]
    assert non_streaming["segments"] == streaming["segments"]
    # …and the marker step then produces the same answer from either of them.
    assert (consume_destination_marker(non_streaming["text"], segments=non_streaming["segments"])
            == consume_destination_marker(streaming["text"], segments=streaming["segments"])
            == "Checking that.\n\nHere you go.")


@pytest.mark.asyncio
async def test_bracketed_prose_survives_the_reader_intact(monkeypatch):
    """The reader now runs on EVERY streamed turn, and markers are made of brackets — so the
    thing to prove is that ordinary bracketed text is not mangled by the hold. A chunk boundary
    landing inside a markdown link is the realistic worst case."""
    from config import config
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    msg, state = _surface_message(), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)
    processor = _processor(FakeOpenAI(["See [", "the docs](http://x/y) and [", "this]."]))

    await _run(processor, slack, msg, state, turn)

    assert "See [the docs](http://x/y) and [this]." in "".join(slack.live.values())


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
    from message_processor.client_contract import Message, Response
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
            from message_processor.thread_manager import AsyncThreadStateManager
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
    from message_processor.client_contract import Message

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
    from message_processor.client_contract import Message

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
