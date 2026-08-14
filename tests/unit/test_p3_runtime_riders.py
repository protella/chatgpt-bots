"""P3a plumbing: the surface-keyed prompt selectors, and three riders.

THE POINT OF THE SELECTORS. Two surfaces need to say different things about one tool, and the words
land in a later wave — so what is tested here is the READ, not the wording: with the channel
constants empty, every surface renders exactly the bytes it renders today, and filling a constant
moves the channel surface and ONLY the channel surface. A test that asserted the words would have to
be rewritten the moment they arrive; these do not.

The riders are unrelated to each other and all three were found in the P2 live battery:

* the participation ledger is process-global, so the unit suite was writing into the REAL one;
* `stream_render` was emitted for builds that are not turns, whose rows join to nothing;
* SIGTERM produced no `session_end`, so every restart read as a crash in the one file that exists
  to tell those apart.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import message_processor.prompts as prompts
from config import config
from message_processor import channel_stream, participation_telemetry
from message_processor.destination_tools import register_destination_tools
from message_processor.utilities import MessageUtilitiesMixin, local_tools_guidance_for
from tests.unit import channel_turn_harness as harness
from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM, ToolRegistry


# ----------------------------------------------------------- the tool-etiquette selector

def test_the_selector_hands_each_surface_its_own_etiquette():
    """RETIRED TRIPWIRE (P3b). This asserted both surfaces still read the DM text — scaffolding for
    landing the selector byte-neutrally, and true only while the channel constant was empty. The
    invariant that outlives it: the channel text is the DM text MINUS the post_to_thread bullet
    (the one that tells the model to acknowledge in the origin thread), and the DM side is
    untouched. The full derivation is pinned in test_channel_restraint_prompts.py.

    W2 added the second half of the channel composition — the window guidance is INTERPOLATED
    onto the derived constant, because a shallow window is only safe if the model knows it is
    looking at one. So the channel surface is no longer the bare constant; what is still
    asserted here is that it CONTAINS it, and that the DM side did not move."""
    assert local_tools_guidance_for(SURFACE_DM) == prompts.LOCAL_TOOLS_GUIDANCE
    channel = local_tools_guidance_for(SURFACE_CHANNEL)
    assert channel.startswith(prompts.CHANNEL_LOCAL_TOOLS_GUIDANCE)
    assert channel != prompts.LOCAL_TOOLS_GUIDANCE
    assert "post_to_thread" in prompts.LOCAL_TOOLS_GUIDANCE
    assert "post_to_thread" not in channel


def test_a_filled_channel_constant_moves_only_the_channel_surface():
    """The constant is read off the module rather than bound at import, so editing it moves the
    channel surface and nothing else. The window guidance rides on top of whatever it holds."""
    with patch.object(prompts, "CHANNEL_LOCAL_TOOLS_GUIDANCE", "\n\n--- CHANNEL ETIQUETTE ---"):
        channel = local_tools_guidance_for(SURFACE_CHANNEL)
        assert channel.startswith("\n\n--- CHANNEL ETIQUETTE ---")
        assert channel.endswith(prompts.render_window_guidance())
        assert local_tools_guidance_for(SURFACE_DM) == prompts.LOCAL_TOOLS_GUIDANCE


def _prompt(surface, **over):
    proc = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    client = MagicMock()
    client.name = "Slack"
    kwargs = dict(tool_surface=surface, tools_available=True, channel_steering=None)
    kwargs.update(over)
    return proc._get_system_prompt(client, **kwargs)


def test_the_system_prompt_reads_the_selector_per_surface():
    with patch.object(prompts, "CHANNEL_LOCAL_TOOLS_GUIDANCE", "CHANNEL-ONLY-MARKER"):
        channel, dm = _prompt(SURFACE_CHANNEL), _prompt(SURFACE_DM)

    assert "CHANNEL-ONLY-MARKER" in channel
    assert "TOOLS ETIQUETTE" not in channel, "the channel text REPLACES the DM block"
    assert "CHANNEL-ONLY-MARKER" not in dm and "TOOLS ETIQUETTE" in dm


def test_the_dm_prompt_is_byte_identical_across_the_change():
    """The non-negotiable: nothing a DM turn sends may move in this wave — and the DM bytes must
    not move even once the channel constant IS filled."""
    baseline = _prompt(SURFACE_DM)
    with patch.object(prompts, "CHANNEL_LOCAL_TOOLS_GUIDANCE", "whatever lands later"):
        assert _prompt(SURFACE_DM) == baseline
    assert prompts.LOCAL_TOOLS_GUIDANCE in baseline


def test_a_turn_with_no_tools_gets_no_etiquette_on_either_surface():
    """The timeout fork sends no tools; a prompt that still taught tool etiquette would be
    describing things absent from its own request."""
    for surface in (SURFACE_DM, SURFACE_CHANNEL):
        with patch.object(prompts, "CHANNEL_LOCAL_TOOLS_GUIDANCE", "CHANNEL-ONLY-MARKER"):
            out = _prompt(surface, tools_available=False)
        assert "TOOLS ETIQUETTE" not in out and "CHANNEL-ONLY-MARKER" not in out


# ------------------------------------------------- the cross-thread conduct paragraph (§9)

def _registry_with_post_to_thread():
    reg = ToolRegistry()
    reg.register({"type": "function", "name": "post_to_thread", "parameters": {}}, AsyncMock())
    reg.register({"type": "function", "name": "no_response_needed", "parameters": {}},
                 AsyncMock(),
                 enabled=lambda cfg: bool(cfg.get("_silence_capable_turn")),
                 channel_enabled=lambda cfg: True)
    register_destination_tools(reg)
    return reg


def _materialize(surface, *, silence_capable, tools_disabled=False, registry=None,
                 destination_open=True):
    """Driven with a REAL turn: the destination paragraph only rides a turn that still has a
    destination to choose, and an order test composed without it grades a list too short to have
    the wrong order."""
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.turn_runtime import TurnRuntime

    host = MagicMock()
    host._materialize_request_tools = TextHandlerMixin._materialize_request_tools.__get__(host)
    host._get_tool_registry = TextHandlerMixin._get_tool_registry.__get__(host)
    client = MagicMock()
    client.tool_registry = (registry if registry is not None
                            else _registry_with_post_to_thread())
    message = SimpleNamespace(
        channel_id="C1" if surface == SURFACE_CHANNEL else "D1",
        metadata={"ts": "10.0", "sender_type": "human",
                  "silence_capable": silence_capable})
    turn = TurnRuntime()
    turn.destination_selected = not destination_open
    with patch.object(config, "enable_tool_loop", True), \
         patch.object(config, "enable_no_reply_tool", True):
        _reg, _cfg, _no_reply, contract = host._materialize_request_tools(
            client, {"model": "m"}, message, tools_disabled=tools_disabled, turn=turn,
            surface=surface)
    return contract or ""


CONDUCT = "[Closing a loop you were part of is legitimate. Post ONCE, in THAT thread.]"


@pytest.mark.parametrize("silence_capable", [True, False])
def test_the_conduct_paragraph_reaches_addressed_and_silence_capable_channel_turns(
        silence_capable):
    """A turn that lands work in another thread arrives ADDRESSED as often as not — somebody is
    usually talking to the bot when the thing that was owed elsewhere becomes answerable — and the
    restraint suffixes never reach an addressed turn. So this paragraph is keyed to the TOOL being
    exposed, not to a posture."""
    with patch.object(prompts, "CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX", CONDUCT):
        contract = _materialize(SURFACE_CHANNEL, silence_capable=silence_capable)

    assert CONDUCT in contract
    # Destination first, conduct next, restraint last: the destination contract ends "call
    # set_reply_destination, then answer", and after conduct has said to answer in the other
    # thread that is the wrong instruction to leave standing. (Ordering, not measured effect —
    # the remeasurement found no support for a recency effect on the row that tests restraint.)
    assert prompts.DESTINATION_CONTRACT_SUFFIX in contract
    assert contract.index(prompts.DESTINATION_CONTRACT_SUFFIX) < contract.index(CONDUCT)
    if silence_capable:
        assert prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX in contract
        assert contract.index(CONDUCT) < contract.index(prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX)
        assert contract.endswith(prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX)
        assert CONDUCT + "\n\n" in contract
    else:
        assert contract.endswith(CONDUCT)


def test_the_conduct_paragraph_never_rides_a_dm_turn():
    """The tool IS in this registry's DM schema set, so what is being tested is the SURFACE check
    and nothing else: a DM has no other thread of its own to post into, and the channel-wide
    conduct paragraph has no business in its prompt."""
    with patch.object(prompts, "CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX", CONDUCT):
        assert CONDUCT not in _materialize(SURFACE_DM, silence_capable=True)


def test_a_turn_that_cannot_post_cross_thread_is_never_told_to():
    """Two ways the tool can be absent, and neither may leave the instruction behind: the timeout
    retry drops the registry, and a registry without the tool simply does not have it."""
    empty = ToolRegistry()
    empty.register({"type": "function", "name": "react_to_message", "parameters": {}}, AsyncMock())
    with patch.object(prompts, "CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX", CONDUCT):
        assert CONDUCT not in _materialize(SURFACE_CHANNEL, silence_capable=True,
                                           tools_disabled=True)
        assert CONDUCT not in _materialize(SURFACE_CHANNEL, silence_capable=True, registry=empty)


def test_the_channel_contract_is_its_paragraphs_in_order():
    """RETIRED TRIPWIRE (P3b). This asserted the suffix was the restraint paragraph ALONE — P3a's
    proof that landing the plumbing moved no bytes, and true only while the conduct constant was
    empty. What replaces it is the whole composition, byte for byte, in the order it is sent:
    destination, conduct, restraint — plus the empty case, which is still the guarantee that an
    unfilled constant adds nothing."""
    assert _materialize(SURFACE_CHANNEL, silence_capable=True) == "\n\n".join([
        prompts.DESTINATION_CONTRACT_SUFFIX,
        prompts.CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX,
        prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX])
    # An ADDRESSED turn reaches no restraint suffix at all, so it ends on conduct.
    assert _materialize(SURFACE_CHANNEL, silence_capable=False) == "\n\n".join([
        prompts.DESTINATION_CONTRACT_SUFFIX,
        prompts.CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX])
    # A turn whose destination is already settled composes the same list without that paragraph.
    assert _materialize(SURFACE_CHANNEL, silence_capable=True, destination_open=False) == (
        prompts.CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX + "\n\n"
        + prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX)
    with patch.object(prompts, "CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX", ""):
        assert _materialize(SURFACE_CHANNEL, silence_capable=True,
                            destination_open=False) == prompts.CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
        assert _materialize(SURFACE_CHANNEL, silence_capable=False,
                            destination_open=False) == ""


# ------------------------------------------------- rider B: stream_render is a TURN event

def _stream():
    return harness.build_stream([harness.normalized("10.0", "hello")])


def _carrier(stream=None):
    """The emitter takes the BUILD RESULT, not the stream: `reselected`, `anchor_advanced` and
    the three page counts all postdate the bytes, so none of them can live on the stream."""
    return channel_stream.StreamBuildResult(
        stream=stream if stream is not None else _stream(),
        reselected=False, anchor_advanced=False,
        pages=channel_stream.PageCounts(history=1, reply=0, origin=0))


def test_a_build_with_no_turn_id_emits_nothing():
    """The dev probes and the out-of-process rebuilds that verify a pinned stream all come through
    the same builder. Their rows join to nothing and cannot be told from a turn whose id went
    missing, so they emit no row at all."""
    with patch.object(participation_telemetry, "stream_render") as emit:
        channel_stream._emit_stream_render(_carrier(), turn_id=None,
                                           origin_root_ts="10.0",
                                           trigger_ts="10.0")
    emit.assert_not_called()


def test_an_admitted_channel_turn_emits_exactly_one_row():
    """The other half: the guard must not be able to swallow the production path silently."""
    with patch.object(participation_telemetry, "stream_render") as emit:
        channel_stream._emit_stream_render(_carrier(), turn_id="turn-1",
                                           origin_root_ts="10.0",
                                           trigger_ts="10.0")
    assert emit.call_count == 1
    fields = emit.call_args.kwargs
    assert fields["turn_id"] == "turn-1"
    assert fields["stream_sha256"] == _stream().stream_sha256


# ------------------------------------------------- rider A: the ledger fixture isolates rows

def test_the_suite_writes_its_participation_rows_into_its_own_directory():
    """The autouse fixture in conftest.py is what makes this true; this is its assertion. Without
    it every unit run appended synthetic decisions to the real ledger, where nothing afterwards
    could tell them from production ones."""
    directory = pathlib.Path(config.log_directory)
    assert directory.name == "ledger"
    assert directory.is_absolute() and "pytest" in str(directory)
    assert directory.parent != pathlib.Path("logs").resolve()

    participation_telemetry.initialize()
    participation_telemetry.record("gate_start", channel_id="C_ISOLATION_PROBE")
    participation_telemetry._drain()

    written = (directory / participation_telemetry.LOG_NAME).read_text()
    assert "C_ISOLATION_PROBE" in written


def test_the_fixture_closes_the_sink_between_tests():
    """Ordering, which is the whole fixture: the sink is drained and closed BEFORE the swap, so a
    row can never be flushed into whichever directory came next."""
    participation_telemetry.initialize()
    participation_telemetry.record("gate_start", channel_id="C_SECOND_PROBE")
    participation_telemetry._drain()
    directory = pathlib.Path(config.log_directory)

    assert "C_ISOLATION_PROBE" not in (directory / participation_telemetry.LOG_NAME).read_text()
