"""Participation telemetry — the record of what the gate decided and what the room saw.

The point of these tests is not that a file gets written. It is that the ledger's two
structural promises hold:

1. A DECLINE stays distinguishable from a decision to stay quiet. A burst collapse, a provider
   error and a model that judged "not for me" all look identical in Slack, and only the record
   can tell them apart afterwards.
2. Every gate attempt ends in EXACTLY ONE terminal event. Not two (the outer gate and the
   responder both used to claim the end of the turn), and not zero (a cancelled turn used to
   vanish). A ledger that cannot count its own attempts cannot be the denominator of anything.

Also asserted: this can never cost a turn. Every entry point swallows its own failures, and
`record` only enqueues — the file write happens on the listener thread.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from message_processor import participation_telemetry as pt
from message_processor.participation import ParticipationEngine, WakeDecision
from tools import participation_ledger_check as plc


@pytest.fixture
def sink(tmp_path):
    """Point the ledger at a temp dir and hand back a reader for its lines.

    Both the module state AND the underlying logger's handlers have to be reset: the logger is
    looked up by name from logging's global registry, so a handler left over from a previous
    test would keep writing to that test's file and this one would read an empty list. Only OUR
    handlers are touched — pytest's logging plugin attaches its own capture handler directly to
    non-propagating loggers, and tearing that out would break capture for everything after.
    """
    named = logging.getLogger(pt._SINK_LOGGER_NAME)
    saved_handlers = named.handlers[:]
    pt.shutdown()
    with patch.object(config, "log_directory", str(tmp_path)), \
            patch.object(config, "enable_participation_telemetry", True):
        pt.initialize()

        def lines(event=None):
            pt._drain()   # the listener owns the file; reading without this races the thread
            path = tmp_path / pt.LOG_NAME
            if not path.exists():
                return []
            rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
            return [r for r in rows if event is None or r["event"] == event]

        try:
            yield lines
        finally:
            pt.shutdown()
    named.handlers = saved_handlers


# --------------------------------------------------------------------------- the record itself

def test_session_start_is_a_health_event_not_an_attempt(sink):
    """It marks a restart boundary, so it carries identity and nothing else. If it looked like
    an attempt, every "one terminal per attempt" check would fail on the first line of the file
    and every restart would inflate the denominator by one."""
    first = sink()[0]
    assert first["event"] == "session_start"
    for absent in ("attempt_id", "channel_id", "trigger_ts"):
        assert absent not in first
    assert first["session"] == pt.SESSION_ID
    assert first["gate_contract"] == pt.GATE_CONTRACT
    assert first["v"] == pt.CONTRACT_VERSION


def test_every_line_carries_the_process_identity(sink):
    """Restarts lose all in-memory participation state. Without a session marker, the first
    verdicts after a deploy are indistinguishable from a genuine change in the room."""
    pt.record("gate_start", channel_id="C1", trigger_ts="10.5")
    for row in sink():
        assert row["session"] == pt.SESSION_ID
        assert row["gate_contract"] == pt.GATE_CONTRACT
        assert row["v"] == pt.CONTRACT_VERSION


def test_record_writes_one_json_line_with_the_join_key(sink):
    pt.record("gate_start", channel_id="C1", trigger_ts="10.5", level="judicious")
    rows = sink("gate_start")
    assert len(rows) == 1
    row = rows[0]
    # channel_id + trigger_ts + attempt_id IS the turn key — events are never assembled in
    # memory, they are joined on those at analysis time.
    assert (row["channel_id"], row["trigger_ts"]) == ("C1", "10.5")
    assert row["level"] == "judicious"
    assert isinstance(row["at"], float)


def test_none_valued_fields_are_omitted_not_written_as_null(sink):
    """Absent and null must not both appear for the same key, or a group-by has two buckets
    meaning the same thing."""
    pt.record("gate_decision", channel_id="C1", trigger_ts="1.0", emoji=None, action="ignore")
    row = sink("gate_decision")[0]
    assert "emoji" not in row
    assert row["action"] == "ignore"


def test_flag_off_writes_nothing(sink):
    before = len(sink())
    with patch.object(config, "enable_participation_telemetry", False):
        pt.record("gate_start", channel_id="C1", trigger_ts="1.0")
    assert len(sink()) == before


def test_an_unserializable_value_still_produces_a_line(sink):
    """default=str, so a stray object degrades to its repr rather than losing the whole event."""
    pt.record("gate_decision", channel_id="C1", trigger_ts="1.0", verdict=object())
    assert "object at 0x" in sink("gate_decision")[0]["verdict"]


def test_record_never_raises_and_never_writes_before_initialization():
    """The contract every call site depends on: telemetry cannot cost a verdict, a reply or a
    reaction. A raise here would land inside the gate's except-clause and become silence.

    Also: with no sink, `record` drops the line instead of building one lazily. Lazy
    construction is what put a mkdir and a file open on the gate's hot path."""
    pt.shutdown()
    pt.record("gate_start", channel_id="C1", trigger_ts="1.0")   # no sink — must not raise

    boom = MagicMock()
    boom.info.side_effect = OSError("no space left on device")
    with patch.object(pt, "_sink", boom):
        pt.record("gate_start", channel_id="C1", trigger_ts="1.0")   # must not raise


def test_no_model_authored_prose_can_reach_the_ledger_at_all(sink):
    """RE-BASELINED TWICE, and the second time is the interesting one.

    This began as a test of the cap on the GATE's `reason` — prose the classifier wrote about a
    human's message, bounded because truncation lowers exposure. When the gate stopped writing a
    reason, the test was rewritten to assert the helper plus the absence. Now the HELPER is gone
    too, and that is the stronger position: a bound with no field to bound is an invitation to add
    one back without thinking about what it would carry. The exposure is removed rather than
    limited.

    So what is asserted is the shape of the door, not the size of the gap: `gate_decision` is
    keyword-only and closed, and there is nowhere to put prose even if somebody wanted to."""
    assert not hasattr(pt, "truncate_reason")

    pt.gate_decision("C1", "1.0", wake=False)
    assert "reason" not in sink("gate_decision")[0]
    with pytest.raises(TypeError):
        pt.gate_decision("C1", "1.0", wake=False, reason="x" * 250)


# ---------------------------------------------------------------- the attempt lifecycle helper

def _msg(**meta):
    m = {"ts": "10.0", "gate_required": True, "silence_capable": True,
         "participation_level": "judicious"}
    m.update(meta)
    return Message(text="anyone know the deploy status?", user_id="U1",
                   channel_id="C1", thread_id="10.0", metadata=m)


def test_the_second_terminal_call_for_an_attempt_is_a_no_op(sink):
    """The defect this guard exists for: the outer gate closed a decline AND the responder path
    closed the same turn, so half the population was counted twice."""
    message = _msg()
    pt.begin_attempt(message)
    assert pt.finish_attempt(message, "silence") is True
    assert pt.finish_attempt(message, "reply") is False
    assert [r["kind"] for r in sink("visible_action")] == ["silence"]


def test_a_redispatch_is_a_linked_second_attempt_not_a_duplicate(sink):
    """Phase Q re-runs the SAME Message object through the gate. A fresh pass can reach a
    different verdict, so it is a second attempt — but one that must stay traceable to the
    first, and one that must be able to close in its own right."""
    message = _msg()
    first = pt.begin_attempt(message)
    pt.finish_attempt(message, "silence")
    second = pt.begin_attempt(message)
    assert second != first
    assert message.metadata[pt.PARENT_KEY] == first
    assert pt.finish_attempt(message, "reply") is True   # the closed flag was reset
    rows = sink("visible_action")
    assert [r["kind"] for r in rows] == ["silence", "reply"]
    assert rows[1]["parent_attempt_id"] == first


def test_an_ungated_message_produces_no_terminal_event(sink):
    """THE SCOPE GUARD. A mention, a DM or a direct thread continuation is not a gate attempt;
    letting its outcome in would put rows with no decision behind them into a population
    documented as decisions."""
    assert pt.finish_attempt(_msg(), "reply") is False
    assert sink("visible_action") == []


# ------------------------------------------------------------- queued-batch linkage (v3)

def test_the_contract_version_says_the_event_set_changed():
    """v3 added the `queue_link` event; v4 turned `silence_reason` from prose into a declared
    enum and made `reaction_visible` unconditional; v5 dropped the ambiguous `placement` field
    for `destination` + `destination_source`; v6 added the `stale_suppressed` terminal kind and
    the `stale_send` diagnostic; v7 is the binary gate — `gate_decision` loses action, emoji,
    placement, reason, the staged findings, the overrules and the backoff taxonomy, and carries
    one bool plus four facts about the call; v8 adds the single-stream events and with them a
    SECOND population keyed by turn_id; v9 is stale reconsideration — `stale_send` gains
    `turn_id` and one-per-suppression-EVENT cardinality, and the `reconsider_start` /
    `reconsider_outcome` pair joins the turn population; v10 is edit_own_message —
    `turn_outcome` gains the always-present `edits` list and the destination kinds gain
    `correction_announcement`. Each is a change an analysis written
    against the older contract must be able to refuse.

    GATE_CONTRACT is asserted beside it because the two move independently — v2–v6 rows remain
    valid under their own contracts, and a reader has to be able to tell which one a row obeys.
    It deliberately did NOT move at v8, v9 or v10: the gate is unchanged, so its rows still
    pool."""
    assert pt.CONTRACT_VERSION == 10
    assert pt.GATE_CONTRACT == "binary-v1"


def test_a_decision_row_carries_the_bit_and_nothing_the_gate_does_not_decide(sink):
    """The v7 field set, pinned. Every name in the second list is a judgment the gate no longer
    makes, so a row carrying one would describe a decision nobody made."""
    pt.gate_decision("C1", "1.0", wake=True, attempt_id="A1", gate_ms=3100,
                     classifier_ms=740, source_count=3, newest_source_ts="1.2")
    row = sink("gate_decision")[0]
    assert row["wake"] is True
    assert row["model"] == config.utility_model
    assert (row["gate_ms"], row["classifier_ms"]) == (3100, 740)
    # The cohort this bit was decided over: without these, a wake on a five-message burst is
    # indistinguishable from a wake on one, which is the difference between the debounce working
    # and the debounce dropping things.
    assert (row["source_count"], row["newest_source_ts"]) == (3, "1.2")
    for retired in ("action", "emoji", "placement", "reason", "relation", "exchange_state",
                    "answerability", "overruled_by", "dimension", "durability", "scope",
                    "guidance", "memory_op", "structural_request", "burst_earlier"):
        assert retired not in row, retired


def test_a_gated_successor_names_the_attempt_it_absorbed_them_into(sink):
    successor = _msg()
    pt.stage_queue_links(successor, ["src-a", "src-b"])
    attempt = pt.begin_attempt(successor)        # the successor's own gate attempt
    pt.emit_queue_links(successor, gate_required=True)

    rows = sink("queue_link")
    assert [r["attempt_id"] for r in rows] == ["src-a", "src-b"]
    for row in rows:
        assert row["batched_into_attempt_id"] == attempt
        assert row["batched_into_gate_required"] is True
        assert row["batched_into_channel_id"] == "C1"
        assert row["batched_into_trigger_ts"] == "10.0"


def test_an_ungated_successor_omits_the_attempt_id_and_says_so(sink):
    """A mention/DM/continuation mints no attempt, and inventing one to make the link prettier
    would put a turn this ledger excludes into the population. The conversation key carries the
    linkage instead."""
    successor = _msg()
    pt.stage_queue_links(successor, ["src-a"])
    pt.emit_queue_links(successor, gate_required=False)

    row = sink("queue_link")[0]
    assert "batched_into_attempt_id" not in row
    assert row["batched_into_gate_required"] is False
    assert row["batched_into_channel_id"] == "C1"
    assert row["batched_into_trigger_ts"] == "10.0"


def test_links_are_written_once_and_never_inherited_by_the_next_batch(sink):
    successor = _msg()
    pt.stage_queue_links(successor, ["src-a"])
    pt.begin_attempt(successor)
    pt.emit_queue_links(successor, gate_required=True)
    pt.emit_queue_links(successor, gate_required=True)   # e.g. a redispatch of the same object
    assert len(sink("queue_link")) == 1


@pytest.mark.asyncio
async def test_a_gated_turn_writes_its_links_after_the_attempt_exists(sink, instant_gate):
    """Ordering, end to end: the successor's attempt is minted inside the gate, so the links
    cannot be written before it — and they are written even when the gate then stays silent."""
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    minted = {}

    async def _gate(message, client):
        minted["id"] = pt.begin_attempt(message)
        return None                      # the gate decided to stay quiet

    app._run_participation_gate = _gate
    message = _msg()
    pt.stage_queue_links(message, ["src-a"])
    await app.handle_message(message, MagicMock())

    row = sink("queue_link")[0]
    assert row["attempt_id"] == "src-a"
    assert row["batched_into_attempt_id"] == minted["id"]


@pytest.mark.asyncio
async def test_a_gated_turn_that_dies_after_minting_still_writes_its_links(sink, instant_gate):
    """The attempt is minted INSIDE the awaited gate. A cancellation on the way back out used to
    skip the emission entirely, leaving an attempt that absorbed queued messages, was recorded as
    `aborted`, and never said what it had taken on."""
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    minted = {}

    async def _gate(message, client):
        minted["id"] = pt.begin_attempt(message)
        raise asyncio.CancelledError()

    app._run_participation_gate = _gate
    message = _msg()
    pt.stage_queue_links(message, ["src-a"])
    with pytest.raises(asyncio.CancelledError):
        await app.handle_message(message, MagicMock())

    row = sink("queue_link")[0]
    assert row["attempt_id"] == "src-a"
    assert row["batched_into_attempt_id"] == minted["id"]


@pytest.mark.asyncio
async def test_a_gate_that_dies_before_minting_keeps_the_links_for_a_later_turn(sink,
                                                                                instant_gate):
    """The other half: with no attempt there is no successor to name, so nothing is written and
    the sources stay owed to whichever turn does run."""
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()

    async def _gate(message, client):
        raise RuntimeError("died on the way to the model")

    app._run_participation_gate = _gate
    message = _msg()
    pt.stage_queue_links(message, ["src-a"])
    with pytest.raises(RuntimeError):
        await app.handle_message(message, MagicMock())

    assert sink("queue_link") == []
    assert message.metadata[pt.BATCHED_SOURCES_KEY] == ["src-a"]


class _LockOnlyProcessor:
    """The REAL process_message on a bare harness — enough to reach the conversation lock and
    no further, which is exactly the boundary the ungated emission sits on."""
    from message_processor.base import MessageProcessor as _MP
    process_message = _MP.process_message

    def __init__(self, manager):
        self.thread_manager = manager
        self.db = None

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


@pytest.mark.asyncio
async def test_an_ungated_turn_claims_nothing_until_it_holds_the_lock(sink):
    """THE RACE. An ungated successor mints no attempt, so a link written when the turn was
    merely INTENDED can never be corrected downstream: if the turn then queues instead of
    answering, the ledger says an ungated turn absorbed messages it never answered. So the write
    waits for the lock — the point where the turn is genuinely running."""
    from thread_manager import AsyncThreadStateManager

    manager = AsyncThreadStateManager(db=None)
    proc = _LockOnlyProcessor(manager)
    message = _msg(gate_required=False, silence_capable=False)
    pt.stage_queue_links(message, ["src-a"])

    # Busy conversation: it queues instead of answering, and claims nothing.
    assert await manager.acquire_thread_lock("10.0", "C1") is True
    try:
        queued = await proc.process_message(message, client=MagicMock(), thinking_id=None)
    finally:
        await manager.release_thread_lock("10.0", "C1")
    assert queued.type == "queued"
    assert sink("queue_link") == []
    assert message.metadata[pt.BATCHED_SOURCES_KEY] == ["src-a"]   # still owed, not lost

    # Free conversation: the same message genuinely runs, and now the link is written.
    with contextlib.suppress(Exception):   # the bare harness dies past the lock; the write is done
        await proc.process_message(message, client=MagicMock(), thinking_id=None)
    row = sink("queue_link")[0]
    assert row["attempt_id"] == "src-a"
    assert "batched_into_attempt_id" not in row
    assert row["batched_into_gate_required"] is False
    assert pt.ATTEMPT_KEY not in message.metadata   # no attempt was minted to carry it


def test_a_queue_link_is_not_a_terminal_event(sink):
    """It says where a message's work went, not what the room saw. If it counted as an outcome
    the queued exclusion rule would be undone by the very event meant to explain it."""
    successor = _msg()
    pt.stage_queue_links(successor, ["src-a"])
    pt.begin_attempt(successor)
    pt.emit_queue_links(successor, gate_required=True)
    assert sink("visible_action") == []


# ------------------------------------------------------------------------ gate-terminal wiring

class _FakeClassifier:
    """An openai_client stand-in. `wake` may be True/False, None (the classifier's own
    no-usable-bit signal), or an exception instance to raise."""

    def __init__(self, wake):
        self._wake = wake
        self.calls = 0

    async def classify_wake(self, *, sources, channel_steering_text=None):
        self.calls += 1
        if isinstance(self._wake, Exception):
            raise self._wake
        return self._wake


def _app(wake, react_result=None):
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.participation_engine = ParticipationEngine(_FakeClassifier(wake))
    app.processor = MagicMock()
    app.processor.db = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    app.processor.db.upsert_channel_pref_memory = AsyncMock(return_value=7)
    client = MagicMock()
    client.channel_pulse = None
    # Explicitly absent, not a MagicMock: the engine pops edit context off the facade, and a
    # truthy mock store would hand back a fake context that suppresses the edit-supersession
    # check and rewrites the classifier prompt.
    client._edit_reply_ctx_map = None
    client._reserve_and_react = AsyncMock(
        return_value=react_result if react_result is not None else {"ok": True})
    return app, client


@pytest.fixture
def instant_gate(monkeypatch):
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)


def _terminals(lines):
    return lines("visible_action")


@pytest.mark.asyncio
async def test_a_supersession_ends_in_exactly_one_terminal_event(sink, instant_gate):
    app, client = _app(True)
    message = _msg()
    app.participation_engine.note_arrival("C1", "20.0", None, "U1")  # a newer message arrived

    assert await app._gate_verdict(message, client) is None

    declines = sink("gate_declined")
    assert [d["cause"] for d in declines] == ["superseded"]   # the old outer backstop is gone
    assert declines[0]["survivor_ts"] == "20.0"               # diagnostic detail lives here
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert (terminals[0]["kind"], terminals[0]["cause"]) == ("none", "superseded")


@pytest.mark.asyncio
async def test_an_edit_supersession_ends_in_exactly_one_terminal_event(sink, instant_gate):
    app, client = _app(True)
    message = _msg()
    app.participation_engine.supersede("C1", "10.0", thread_root="10.0", sender_id="U1")

    assert await app._gate_verdict(message, client) is None

    assert [d["cause"] for d in sink("gate_declined")] == ["edit_superseded"]
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert (terminals[0]["kind"], terminals[0]["cause"]) == ("none", "edit_superseded")


@pytest.mark.asyncio
async def test_a_classifier_failure_is_never_scored_as_a_verdict(sink, instant_gate):
    """The whole reason classify_participation now returns None. A forged {"action": "ignore"}
    made a bad afternoon at the provider indistinguishable from the bot exercising restraint —
    in a ledger whose entire job is to measure exactly that."""
    app, client = _app(None)   # the classifier's own fail-safe signal
    assert await app._gate_verdict(_msg(), client) is None

    assert sink("gate_decision") == []   # no manufactured decision
    assert [d["cause"] for d in sink("gate_declined")] == ["classifier_error"]
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert (terminals[0]["kind"], terminals[0]["cause"]) == ("none", "classifier_error")


@pytest.mark.asyncio
async def test_a_classifier_exception_records_its_type_and_its_latency(sink, instant_gate):
    app, client = _app(TimeoutError("upstream"))
    assert await app._gate_verdict(_msg(), client) is None
    decline = sink("gate_declined")[0]
    assert decline["cause"] == "classifier_error"
    assert decline["detail"] == "TimeoutError"
    # Measured around the model call alone. The gate's own wall time is mostly debounce, so
    # reading it as classifier latency would blame the provider for a delay we chose.
    assert "classifier_ms" in decline
    # WHICH model failed. Without it a utility-model swap can be judged on its verdicts but not
    # on its failure rate — and the failure rate is the half that decides whether it was worth it.
    assert decline["model"] == config.utility_model


@pytest.mark.asyncio
async def test_a_failure_after_the_decision_is_not_a_classifier_decline(sink, instant_gate):
    """The model decided; carrying the decision out is what broke. Filing that as a decline
    reports the gate as unable to judge while its decision sits two lines above in the same
    file — and inflates exactly the number ("how often does the classifier fail?") that would
    send someone looking at the wrong system.

    RE-BASELINED in how it is provoked, not in what it asserts. There used to be a rich action to
    fail at (`_place_gate_reaction` raising on the way to Slack); with the reaction and the backoff
    write deleted, the only work left after the bit is stamping the cohort on the message. So the
    metadata itself refuses the write — contrived, deliberately, because the SPLIT is the thing
    worth keeping: `decision_recorded` is what tells "the gate could not judge" apart from "the
    gate judged and the handoff broke", and an untested branch is how that distinction rots."""
    class _HostileMetadata(dict):
        def __setitem__(self, key, value):
            if key == "gate_sources":
                raise RuntimeError("slack down")
            super().__setitem__(key, value)

    app, client = _app(True)
    message = _msg()
    message.metadata = _HostileMetadata(message.metadata)
    assert await app._gate_verdict(message, client) is None

    assert sink("gate_decision")[0]["wake"] is True           # the decision IS on record
    decline = sink("gate_declined")[0]
    assert (decline["cause"], decline["detail"]) == ("action_error", "RuntimeError")
    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["cause"] == "action_error"


@pytest.mark.asyncio
async def test_a_failure_before_any_verdict_is_still_a_plain_error(sink, instant_gate):
    """The other side of the same split: nothing was decided, so this one really is a decline."""
    app, client = _app(True)
    app.participation_engine.note_arrival = MagicMock(side_effect=RuntimeError("boom"))
    assert await app._gate_verdict(_msg(), client) is None

    assert sink("gate_decision") == []
    assert sink("gate_declined")[0]["cause"] == "error"
    assert _terminals(sink)[0]["cause"] == "error"


@pytest.mark.asyncio
async def test_failure_and_a_real_no_wake_are_the_same_silence_and_different_rows(
        sink, instant_gate):
    """Behaviour equivalence, asserted rather than assumed: returning None from the classifier
    changed what we WRITE DOWN, and nothing about what the room sees.

    `none` vs `silence` is the whole point. Only a genuine wake=false is the model choosing to stay
    out; a provider outage is the gate never opening, and scoring it as restraint would corrupt the
    one number this ledger exists to produce."""
    app_failed, client = _app(None)
    app_quiet, client2 = _app(False)

    assert await app_failed._gate_verdict(_msg(), client) is None
    assert await app_quiet._gate_verdict(_msg(), client2) is None    # identical outcome

    kinds = [(r["kind"], r.get("cause")) for r in _terminals(sink)]
    assert kinds == [("none", "classifier_error"), ("silence", None)]
    # ...and only the second produced a decision row at all.
    assert [r["wake"] for r in sink("gate_decision")] == [False]


@pytest.mark.asyncio
async def test_a_decision_not_to_wake_is_a_decision_and_says_so(sink, instant_gate):
    app, client = _app(False)
    assert await app._gate_verdict(_msg(), client) is None

    decision = sink("gate_decision")[0]
    assert decision["wake"] is False
    assert decision["model"] == config.utility_model    # decision quality is model-specific
    assert "gate_ms" in decision and "classifier_ms" in decision
    assert decision["source_count"] == 1 and decision["newest_source_ts"] == "10.0"
    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "silence" and terminal["ended_by"] == "gate"
    # No silence_reason: that eight-value enum belongs to the RESPONDER, which can say why it chose
    # to stay quiet after seeing everything. The gate knows only that it did not open.
    assert "silence_reason" not in terminal
    # A gate-only outcome never woke the responder. Conflating "the gate acted" with "the gate
    # woke the bot" makes the wake rate unreadable.
    assert terminal["gate_woke"] is False and terminal["responder_started"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", [True, False, None])
async def test_the_gate_writes_no_reaction_row_on_any_outcome(sink, instant_gate, wake):
    """SEVEN TESTS COLLAPSE INTO THIS ONE, and the collapse is the commit.

    They covered the gate's own reactions in detail: a react verdict that landed (reaction_only), one
    whose emoji failed (none), one Slack said was already there (already_present), one the
    once-per-message stamp refused before Slack (already_stamped), and three backoff-ack variants
    (acked → reaction_only, feedback about reactions → silence, redispatched ack → not a silence).
    Every one of them described the gate placing an emoji before the responder ran — which is
    exactly what commit 6 deletes, along with `_place_gate_reaction`, `_backoff_ack`, and the
    `origin=gate` / `origin=backoff_ack` runtime emissions.

    So there is nothing left to parametrize over except the outcomes, and the assertion is that the
    room is untouched on all of them. `origin=work_claim` (TurnRuntime.claim_work) and
    `origin=responder` (the react tool) are unaffected and are covered in their own sections."""
    app, client = _app(wake)
    message = _msg()
    assert (await app._gate_verdict(message, client) is None) is (wake is not True)

    assert sink("reaction") == []
    client._reserve_and_react.assert_not_awaited()
    assert "participation_reaction_emoji" not in message.metadata
    # No decision row carries an emoji or a backoff taxonomy either — a reaction that is only
    # recorded and never placed is the same lie told in the ledger instead of the room.
    for row in sink("gate_decision"):
        for retired in ("emoji", "dimension", "durability", "scope", "memory_op",
                        "structural_request", "guidance"):
            assert retired not in row, retired


@pytest.mark.asyncio
async def test_the_gate_writes_nothing_durable_on_any_outcome(sink, instant_gate):
    """The other half of the deleted backoff path: a classifier that had seen ONE message used to
    write channel settings and a preference memory row. Participation feedback wakes the responder
    now, whose memory/settings tools own the write under commit-3 authorization — so the gate must
    touch neither database method on any outcome."""
    for wake in (True, False, None):
        app, client = _app(wake)
        await app._gate_verdict(_msg(), client)
        app.processor.db.upsert_channel_pref_memory.assert_not_called()
        app.processor.db.set_channel_settings_async.assert_not_called()


@pytest.mark.asyncio
async def test_a_speaking_verdict_leaves_the_attempt_open_for_the_responder(sink, instant_gate):
    """The gate hands the turn on, so it must NOT close it — the responder owns what the room
    finally saw, and a terminal here would be the double-count all over again.

    Through `_run_participation_gate`, not `_gate_verdict`: the wake is recorded once at the
    single point where a verdict leaves the gate, rather than once per fall-through branch."""
    app, client = _app(True)
    message = _msg()
    decision = await app._run_participation_gate(message, client)

    assert decision is not None and decision.wake is True
    assert _terminals(sink) == []
    assert message.metadata[pt.GATE_WOKE_KEY] is True
    assert message.metadata["gate_woke"] is True   # …and the public routing fact agrees


@pytest.mark.asyncio
async def test_the_engine_off_path_still_counts_as_an_attempt(sink):
    """An attempt that never reaches the model is still an attempt — and it needs a gate_start,
    not just an ending. gate_start IS the denominator, so an engine-off attempt that produced a
    decline and a terminal but no start contributed two numerators and nothing to divide by.
    (The OTHER engine-off drop, the pre-dispatch one in message_events, never reaches the gate
    and is documented as outside this population.)"""
    app, client = _app(True)
    with patch.object(config, "enable_participation_engine", False):
        assert await app._gate_verdict(_msg(), client) is None

    starts = sink("gate_start")
    assert len(starts) == 1 and starts[0]["level"] == "judicious"
    assert [d["cause"] for d in sink("gate_declined")] == ["engine_off"]
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert (terminals[0]["kind"], terminals[0]["cause"]) == ("none", "engine_off")
    # All three join on the same attempt.
    assert len({starts[0]["attempt_id"], sink("gate_declined")[0]["attempt_id"],
                terminals[0]["attempt_id"]}) == 1


@pytest.mark.asyncio
async def test_gate_start_records_the_posture_the_message_arrived_with(sink, instant_gate):
    app, client = _app(False)
    message = _msg(sender_type="human", wake_source="ambient", edit_reply_marker="99.9")
    message.metadata[pt.ATTEMPT_KEY] = "an-earlier-attempt"   # i.e. a queued redispatch

    await app._gate_verdict(message, client)

    start = sink("gate_start")[0]
    assert start["sender_type"] == "human"
    assert start["wake_source"] == "ambient"
    assert start["edit"] is True
    assert start["redispatch"] is True
    # DROPPED: `is_dm` was constant-false here (a DM never reaches the gate) and invited exactly
    # the wrong reading — that DMs are in this population and simply never judged.
    assert "is_dm" not in start


# ------------------------------------------------------------------ the responder's react tool

def _react_host(attempt_id="A1", result=None):
    from slack_client.messaging import SlackMessagingMixin
    host = MagicMock()
    host._reserve_and_react = AsyncMock(
        return_value=result if result is not None else {"ok": True})
    host.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(host)
    ctx = MagicMock(channel_id="C1", trigger_ts="10.0", thread_ts="10.0",
                    attempt_id=attempt_id)
    return host, ctx


@pytest.mark.asyncio
async def test_the_react_tool_keeps_its_target_apart_from_its_trigger(sink):
    """The model may react to an OLDER message. Collapsing the two silently reattributes those
    reactions to the message that merely caused the turn."""
    host, ctx = _react_host()
    await host.execute_react_tool(ctx, {"emoji": "tada", "ts": "5.0"})

    row = sink("reaction")[0]
    assert row["trigger_ts"] == "10.0"       # the message that woke us
    assert row["target_ts"] == "5.0"         # the message that got the emoji
    assert (row["origin"], row["result"]) == ("responder", "added")


@pytest.mark.asyncio
@pytest.mark.parametrize("args,setting,detail", [
    ({"emoji": "tada"}, ("enable_react_tool", False), "disabled"),
    ({"emoji": "NOT valid!"}, None, "invalid_emoji"),
    ({"emoji": "tada"}, ("reaction_emojis", ["joy"]), "emoji_not_allowed"),
])
async def test_a_refused_reaction_is_an_intent_worth_recording(sink, args, setting, detail,
                                                               monkeypatch):
    """A model that keeps asking for a disallowed emoji is a prompt problem. Before these were
    written down, the only refusal that left any trace was the one that reached Slack."""
    host, ctx = _react_host()
    if setting:
        monkeypatch.setattr(config, setting[0], setting[1], raising=False)
    out = await host.execute_react_tool(ctx, args)

    assert out["ok"] is False
    row = sink("reaction")[0]
    assert (row["result"], row["detail"]) == ("refused", detail)
    # WHAT it was aimed at, resolved before the gauntlet. These refusals used to record no
    # target at all, which read as "nothing was aimed at" even when the call named a message.
    assert row["target_ts"] == "10.0"


@pytest.mark.asyncio
async def test_a_refusal_remembers_the_older_message_it_named(sink, monkeypatch):
    """The model may aim at an OLDER message and still be refused. Resolving the target after
    the allowlist check threw that away, so a model repeatedly reacting at one specific message
    with a banned emoji was indistinguishable from one flailing at the trigger."""
    monkeypatch.setattr(config, "reaction_emojis", ["joy"], raising=False)
    host, ctx = _react_host()
    await host.execute_react_tool(ctx, {"emoji": "tada", "ts": "5.0"})

    row = sink("reaction")[0]
    assert (row["result"], row["detail"]) == ("refused", "emoji_not_allowed")
    assert (row["trigger_ts"], row["target_ts"]) == ("10.0", "5.0")


@pytest.mark.asyncio
async def test_a_react_tool_call_with_nowhere_to_aim_is_recorded(sink):
    host, ctx = _react_host()
    ctx.channel_id = None
    out = await host.execute_react_tool(ctx, {"emoji": "tada"})
    assert out["error"] == "no_target"
    assert sink("reaction")[0]["detail"] == "no_target"


@pytest.mark.asyncio
async def test_the_react_tool_writes_nothing_for_an_ungated_turn(sink):
    """A mention's reaction is a real reaction and simply not ours to count."""
    host, ctx = _react_host(attempt_id=None)
    await host.execute_react_tool(ctx, {"emoji": "tada"})
    await host.execute_react_tool(ctx, {"emoji": "NOT valid!"})
    assert sink("reaction") == []


# ---------------------------------------------------------------------------- the work claim

def _turn_and_client(reserve_result, lease):
    from message_processor.turn_runtime import TurnRuntime
    turn = TurnRuntime()
    client = MagicMock()
    client._reserve_and_react_owned = AsyncMock(return_value=(reserve_result, lease))
    return turn, client


@pytest.mark.asyncio
async def test_a_work_claim_counts_as_placed_only_when_it_holds_the_lease(sink, monkeypatch):
    """A lease is the only proof WE placed the 👀. `ok` without one means it was already up
    there — a previous turn's, or the model's own react tool — and recording that as `added`
    would inflate the claim rate with reactions the bot never made."""
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)

    turn, client = _turn_and_client({"ok": True}, {"token": "t"})
    await turn.claim_work(client, _msg(**{pt.ATTEMPT_KEY: "A1"}))
    assert sink("reaction")[-1]["result"] == "added"

    turn2, client2 = _turn_and_client({"ok": True, "idempotent": True}, None)
    await turn2.claim_work(client2, _msg(**{pt.ATTEMPT_KEY: "A2"}))
    assert sink("reaction")[-1]["result"] == "already_present"


@pytest.mark.asyncio
async def test_a_work_claim_that_never_reached_slack_is_a_failure(sink, monkeypatch):
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    turn, client = _turn_and_client(None, None)
    client._reserve_and_react_owned = AsyncMock(side_effect=RuntimeError("slack down"))

    await turn.claim_work(client, _msg(**{pt.ATTEMPT_KEY: "A1"}))

    row = sink("reaction")[-1]
    assert (row["result"], row["detail"]) == ("failed", "RuntimeError")


@pytest.mark.asyncio
async def test_a_retracted_claim_records_the_real_removal_outcome(sink, monkeypatch):
    """remove_owned_reaction refuses a stale lease and returns False, leaving the 👀 up. A row
    saying we took it back would describe the opposite of the room."""
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    from message_processor.turn_runtime import TurnRuntime

    for removed, expected in ((True, "removed"), (False, "remove_failed")):
        turn = TurnRuntime(ack_lease={"token": "t"}, ack_target_ts="10.0",
                           ack_channel_id="C1", ack_attempt_id="A1")
        client = MagicMock()
        client.remove_owned_reaction = AsyncMock(return_value=removed)
        await turn.settle_ack(client, produced_output=False)
        row = sink("reaction")[-1]
        assert (row["operation"], row["result"]) == ("remove", expected)


@pytest.mark.asyncio
async def test_the_work_claim_writes_nothing_for_an_ungated_turn(sink, monkeypatch):
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    turn, client = _turn_and_client({"ok": True}, {"token": "t"})
    await turn.claim_work(client, _msg())            # no attempt id stamped
    turn.ack_lease = {"token": "t"}
    client.remove_owned_reaction = AsyncMock(return_value=True)
    await turn.settle_ack(client, produced_output=False)
    assert sink("reaction") == []


# -------------------------------------------------- what the room saw, as one honest label

def _resp(rtype="text", content="", **meta):
    return MagicMock(type=rtype, content=content, metadata=meta)


@pytest.mark.parametrize("response,turn,expected", [
    (None, None, "empty"),
    (_resp("queued"), None, "queued"),
    (_resp("error", "boom"), None, "error"),
    (_resp(content="sorry, that failed", interrupted=True), None, "interrupted"),
    (_resp(terminal_action="no_reply", silence_reason="addressed_to_other"), None,
     "silence"),
    (_resp(reaction_only=True), None, "reaction_only"),
    (_resp(content="here you go"), None, "reply"),
    (_resp(content="", streamed=True, posted=True), None, "reply"),
    (_resp(content="", background_job_started=True), None, "detached"),
    (_resp(content=""), None, "empty"),
])
def test_visible_action_labels(response, turn, expected):
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(response, turn) == expected


def test_an_answer_slack_refused_is_not_a_reply():
    """`posted is False` with real content is a DELIVERY FAILURE — the opposite of silence. It
    used to be filed as `reply` because the content was non-empty, so every delivery outage
    read as the bot talking normally."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(
        _resp(content="here is the answer", posted=False), None) == "delivery_failed"
    assert ChatBotV2._classify_visible_action(
        _resp(content="here is the answer", posted=True), None) == "reply"


def test_a_silence_carrying_a_reaction_is_not_a_silence():
    """The emoji WAS the answer and the room saw one. Filing it as silence understated how
    often the bot participates without words — the very rate this ledger measures."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="reacted_instead",
              response_reaction_committed=True), None) == "reaction_only"


def test_a_queued_turn_is_never_an_empty_one():
    """It never ran; another turn owns the conversation. `empty` means the contract broke."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(_resp("queued"), None) == "queued"


def test_a_detached_producer_is_not_an_empty_turn():
    """generate_image posts the picture itself and hands back an empty Response. Read literally
    that is 'posted nothing' — which would file the most visible turn the bot can have as a
    contract violation."""
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime

    turn = TurnRuntime(visible_action_committed=True)
    assert ChatBotV2._classify_visible_action(_resp(content=""), turn) == "detached"
    # ...and with nothing committed, the same empty Response IS the violation.
    assert ChatBotV2._classify_visible_action(_resp(content=""), TurnRuntime()) == "empty"


def test_chosen_silence_and_broken_contract_never_share_a_label():
    """Identical in the room, opposite in meaning. If these collapsed, every failed turn would
    read as the bot exercising restraint — which is the exact claim the ledger has to test."""
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(
        _resp(terminal_action="no_reply", silence_reason="nothing_to_add"), None) == "silence"
    assert ChatBotV2._classify_visible_action(_resp(content="   "), None) == "empty"


# ----------------------------------------------------------------- wired into the live engine

@pytest.mark.asyncio
async def test_engine_records_a_supersession_decline(sink):
    """End-to-end through the real ParticipationEngine: a message overtaken during the debounce
    yields no verdict, and that nothing is the invisible half of the population."""
    engine = ParticipationEngine(MagicMock())
    engine.openai_client.classify_wake = AsyncMock(return_value=False)
    with patch.object(config, "participation_debounce_seconds", 0.0):
        engine.note_arrival("C1", "20.0", None, "U1")  # a newer message already arrived
        evaluation = await engine.evaluate(channel_id="C1", ts="10.0", text="hi",
                                           sender_id="U1", attempt_id="A1")

    assert evaluation.decision is None
    assert evaluation.decline_cause == "superseded"
    engine.openai_client.classify_wake.assert_not_awaited()   # nothing was even asked
    rows = sink("gate_declined")
    assert len(rows) == 1
    assert rows[0]["cause"] == "superseded"
    assert (rows[0]["trigger_ts"], rows[0]["attempt_id"]) == ("10.0", "A1")


@pytest.mark.asyncio
async def test_engine_reports_the_failure_and_still_fails_safe(sink):
    """An API error is still silence in the room — that has not changed — but it is no longer
    DRESSED as a decision.

    RE-BASELINED: the old assertion was `evaluation.verdict.action == "ignore"`, i.e. the engine
    manufactured a fail-safe verdict and the caller could not tell it from the model choosing to
    stay out. Now `decision` is None and `decline_cause` says which kind of nothing it was, so the
    caller closes the attempt as `none` instead of scoring a provider outage as restraint."""
    engine = ParticipationEngine(MagicMock())
    engine.openai_client.classify_wake = AsyncMock(side_effect=TimeoutError("upstream"))
    with patch.object(config, "participation_debounce_seconds", 0.0):
        evaluation = await engine.evaluate(channel_id="C1", ts="10.0", text="hi", sender_id="U1")

    assert evaluation.decision is None                 # no forged bit
    assert evaluation.decline_cause == "classifier_error"
    assert isinstance(evaluation.classifier_ms, int)   # measured even on the failing path
    assert sink("gate_declined")[0]["detail"] == "TimeoutError"
    assert sink("gate_decision") == []


# --------------------------------------------------------- the responder's own terminal event


def _responder_app(response, *, provenance_error=None):
    """A ChatBotV2 whose gate always hands the turn on and whose responder returns `response`."""
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = None
    app.processor.process_message = AsyncMock(return_value=response)
    app.processor._persist_tool_provenance = MagicMock(side_effect=provenance_error)
    return app


async def _run_responder(app, message):
    """Drive handle_message past a gate that woke the responder.

    The old `gate_emoji=` knob is gone with the thing it simulated: react_and_respond put an emoji
    on the message BEFORE the responder ran, and the terminal classifier had to be told about it
    separately. Every reaction is the responder's own now, so a test that wants one in the room
    puts `response_reaction_committed` on the RESPONSE — which is where the live path reads it."""
    from main import ChatBotV2

    async def _gate(msg, client):
        pt.begin_attempt(msg)
        pt.mark_gate_woke(msg)
        return WakeDecision(wake=True)

    app._run_participation_gate = _gate
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.send_message = AsyncMock(return_value="11.0")
    client.delete_message = AsyncMock()
    client.handle_error = AsyncMock()
    await ChatBotV2.handle_message(app, message, client)
    return client


@pytest.mark.asyncio
async def test_a_reaction_plus_a_declared_silence_is_not_a_silence(sink):
    """A turn that reacted and then declared "no words needed" left something in the room, so the
    terminal must not call it silence — and the reason it gave was the reason for using no WORDS,
    which is preserved rather than swallowed.

    RE-BASELINED as to WHOSE emoji it is: this was the react_and_respond case, where the gate put
    the emoji up before the responder ran. The gate places nothing now, so the same shape arises
    from the responder reacting through react_to_message and then vetoing — the identical terminal
    question with one fewer actor in it."""
    app = _responder_app(_resp(terminal_action="no_reply", silence_reason="nothing_to_add",
                               response_reaction_committed=True))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "reaction_only"
    assert terminal["silence_reason"] == "nothing_to_add"   # preserved, not swallowed
    assert terminal["reaction_visible"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["addressed_to_other", "nothing_to_add", "duplicate",
                                    "user_requested_silence", "awaiting_context", "other"])
async def test_the_declared_reason_reaches_the_ledger_verbatim(sink, reason):
    """End to end, all the way from the terminal tool's argument to the column. Verbatim: the
    ledger reports what the model said about itself, and a value we adjusted on the way through
    would make the whole column untrustworthy at exactly the points worth reading."""
    app = _responder_app(_resp(terminal_action="no_reply", silence_reason=reason))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "silence"
    assert terminal["silence_reason"] == reason
    assert terminal["reaction_visible"] is False


@pytest.mark.asyncio
async def test_reacted_instead_with_no_reaction_records_both_halves(sink):
    """The mismatch is the measurement: the model said the emoji was its answer, and no emoji
    is there. Neither fact is corrected against the other, so the disagreement is countable."""
    app = _responder_app(_resp(terminal_action="no_reply", silence_reason="reacted_instead"))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "silence"                    # nothing was in the room
    assert terminal["silence_reason"] == "reacted_instead"  # ...but this is what it claimed
    assert terminal["reaction_visible"] is False


@pytest.mark.asyncio
async def test_a_reply_that_also_reacted_says_so(sink):
    """`kind` names the words because they are the louder half — so the emoji half of a turn that
    both reacted and replied has to ride beside it or it leaves the terminal record entirely."""
    app = _responder_app(_resp(content="here you go", posted=True, streamed=True, model="m-1",
                               response_reaction_committed=True))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "reply" and terminal["reaction_visible"] is True
    # WHICH model wrote it: per-user and per-thread overrides mean two rows here can come from
    # different models, and a reply-quality comparison that pools them describes neither.
    assert terminal["model"] == "m-1"


@pytest.mark.asyncio
async def test_a_plain_reply_claims_no_reaction(sink):
    app = _responder_app(_resp(content="here you go", posted=True, streamed=True))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "reply"
    # Explicit false, not absent: "no emoji" and "nobody recorded whether there was one" are
    # different claims, and only one of them can be counted.
    assert terminal["reaction_visible"] is False


@pytest.mark.asyncio
async def test_no_response_object_is_its_own_contract_failure(sink):
    """`none` is the GATE ending a turn with nothing to show. A responder that hands back no
    Response at all is the responder contract breaking, which is a different bug in a different
    layer and used to share the gate's label."""
    app = _responder_app(None)
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert (terminal["kind"], terminal["detail"]) == ("empty", "no_response_object")


@pytest.mark.asyncio
async def test_a_delivered_reply_survives_a_crash_in_the_bookkeeping_after_it(sink):
    """Everything between the send and the close is bookkeeping. Filing a post-delivery raise as
    `error_unhandled` deletes a reply Slack is already showing from the talk rate — so a bad
    afternoon in the provenance path would read as a bot that stopped answering."""
    app = _responder_app(_resp(content="here you go"),
                         provenance_error=RuntimeError("db gone"))
    await _run_responder(app, _msg())

    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert terminals[0]["kind"] == "reply"
    assert terminals[0]["post_delivery_error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_a_crash_before_delivery_is_still_an_unhandled_error(sink):
    """The other side of the same guard: nothing landed, so nothing is being protected."""
    app = _responder_app(_resp(content="here you go"))
    app.processor.process_message = AsyncMock(side_effect=RuntimeError("model died"))
    await _run_responder(app, _msg())

    terminal = _terminals(sink)[0]
    assert (terminal["kind"], terminal["detail"]) == ("error_unhandled", "RuntimeError")


# ------------------------------------------------------------------- the sink's own lifecycle


def test_telemetry_off_opens_no_file_and_starts_no_thread(tmp_path):
    """config.py promises "off costs nothing but the record". An empty rotating log plus a
    parked listener thread is not nothing — and `record()` checking the flag only stopped the
    writing, not the opening."""
    named = logging.getLogger(pt._SINK_LOGGER_NAME)
    saved = named.handlers[:]
    pt.shutdown()
    try:
        with patch.object(config, "log_directory", str(tmp_path)), \
                patch.object(config, "enable_participation_telemetry", False):
            pt.initialize()
            assert pt._sink is None and pt._listener is None
            assert list(tmp_path.iterdir()) == []
    finally:
        pt.shutdown()
        named.handlers = saved


def test_the_build_stamp_identifies_the_code_and_rides_one_line_only(tmp_path):
    """A restart is not a deployment. Without a build revision the before/after boundary of a
    prompt change is not mechanically recoverable — and the gate prompt has already moved once
    without GATE_CONTRACT moving. It rides `session_start` alone; everything else joins on
    `session`."""
    named = logging.getLogger(pt._SINK_LOGGER_NAME)
    saved = named.handlers[:]
    pt.shutdown()
    try:
        with patch.object(config, "log_directory", str(tmp_path)), \
                patch.object(config, "enable_participation_telemetry", True), \
                patch.object(pt, "_build_revision", lambda: "cafe123"):
            pt.initialize()
            pt.record("gate_start", channel_id="C1", trigger_ts="1.0")
            pt.shutdown()
        rows = [json.loads(ln) for ln in
                (tmp_path / pt.LOG_NAME).read_text().splitlines() if ln.strip()]
        assert rows[0] == {**rows[0], "event": "session_start", "build": "cafe123"}
        assert [r for r in rows if r["event"] != "session_start" and "build" in r] == []
        # A graceful stop is visible IN THE FILE. Without it, a tail that simply ends cannot be
        # told apart from a crash that lost an unknown number of terminals.
        assert rows[-1]["event"] == "session_end"
        assert rows[-1]["session"] == pt.SESSION_ID
        assert "attempt_id" not in rows[-1]
    finally:
        pt.shutdown()
        named.handlers = saved


def test_a_build_revision_never_raises_and_never_hangs():
    """No git, no repo, a stale index lock: any of them yields None and boots anyway."""
    with patch.object(pt.subprocess, "run", side_effect=OSError("no git")):
        assert pt._build_revision() is None
    with patch.object(pt.subprocess, "run",
                      side_effect=pt.subprocess.TimeoutExpired("git", 1)):
        assert pt._build_revision() is None


def test_an_off_vocabulary_label_is_written_and_warned_about_once(sink):
    """Analysis is a group-by, so a typo does not fail anything — it invents a bucket and
    deflates the real one. Losing the line would be worse than logging an odd label, so the
    check only complains; and it complains once, or a hot-path typo buries app.log."""
    pt._warned_vocabulary.clear()
    message = _msg()
    pt.begin_attempt(message)
    with patch.object(pt, "logger") as log:
        pt.finish_attempt(message, "raply")            # a plausible typo of `reply`
        pt.begin_attempt(message)
        pt.finish_attempt(message, "raply")
        assert log.warning.call_count == 1
    assert [r["kind"] for r in sink("visible_action")] == ["raply", "raply"]
    pt._warned_vocabulary.clear()


def test_the_known_vocabulary_covers_what_the_code_actually_writes():
    """The sets are only worth having if they are the same list the call sites use."""
    assert {"reply", "silence", "reaction_only", "queued", "empty", "none"} <= pt.KINDS
    assert {"engine_off", "error", "action_error", "classifier_error"} <= pt.DECLINE_CAUSES
    assert pt.REACTION_OPERATIONS == {"add", "remove"}
    assert "remove_failed" in pt.REACTION_RESULTS
    assert "backoff_ack" in pt.REACTION_ORIGINS


def test_responder_bookkeeping_never_touches_an_ungated_message():
    """mark_responder_started sits on the path EVERY turn takes. Without the attempt-id guard a
    mention or a DM picked up telemetry-private state it can never produce a line from."""
    message = _msg()
    pt.mark_responder_started(message)
    assert pt.RESPONDER_STARTED_KEY not in message.metadata

    pt.begin_attempt(message)
    pt.mark_responder_started(message)
    assert message.metadata[pt.RESPONDER_STARTED_KEY] is True


@pytest.mark.asyncio
async def test_a_removal_that_raised_is_recorded_as_a_failed_removal(sink, monkeypatch):
    """The 👀 is still up and the lease is gone, so the claim is stranded. A lifecycle that ends
    with an add and no removal outcome reads as a claim that was HONORED — the exact opposite of
    a turn that promised work, produced none, and then failed to clean up after itself."""
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    from message_processor.turn_runtime import TurnRuntime

    turn = TurnRuntime(ack_lease={"token": "t"}, ack_target_ts="10.0",
                       ack_channel_id="C1", ack_attempt_id="A1")
    client = MagicMock()
    client.remove_owned_reaction = AsyncMock(side_effect=RuntimeError("slack down"))
    await turn.settle_ack(client, produced_output=False)

    row = sink("reaction")[-1]
    assert (row["operation"], row["result"]) == ("remove", "remove_failed")
    assert row["detail"] == "RuntimeError"


@pytest.mark.asyncio
async def test_an_aborted_turn_still_closes_its_attempt(sink, instant_gate):
    """A cancellation used to leave a gate_start with no ending — indistinguishable from a
    decision that simply has not been written yet."""
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = None      # the already-processing peek short-circuits
    message = _msg()

    async def _explode(msg, client):
        pt.begin_attempt(msg)
        raise asyncio.CancelledError()

    # Cancelled inside the GATE, i.e. before the responder try/except exists — the exact hole
    # the outer finally was added for.
    app._run_participation_gate = _explode
    with pytest.raises(asyncio.CancelledError):
        await ChatBotV2.handle_message(app, message, MagicMock())

    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["kind"] == "aborted"


# ------------------------------------------------------- v8: the turn population (CV8, §10)

def _turn(**kwargs):
    from message_processor.turn_runtime import TurnRuntime
    return TurnRuntime(**kwargs)


def test_a_turn_start_is_the_turn_populations_denominator_not_the_gates(sink):
    """Written for every channel turn, gated or not. That is the whole reason it exists beside
    `gate_start`: a mention and a thread continuation are turns nobody judged, and a ledger that
    could only count judged messages could never say what share of the bot's channel output the
    gate is responsible for."""
    pt.turn_start("C1", "10.0", turn_id="s:1", origin_thread_ts="9.0", surface="channel",
                  gated=False, wake_source="mention")
    row = sink("turn_start")[0]
    assert row["turn_id"] == "s:1" and row["gated"] is False
    assert (row["channel_id"], row["trigger_ts"], row["origin_thread_ts"]) == ("C1", "10.0", "9.0")
    assert row["surface"] == "channel" and row["wake_source"] == "mention"
    assert "attempt_id" not in row          # an ungated turn has none to name


def test_a_turn_outcome_reports_what_the_turn_accumulated(sink):
    """Assembled from the TurnRuntime rather than from a payload the caller composed, so the
    outer finally cannot report a destination set that disagrees with the handlers'."""
    from message_processor.turn_runtime import DEST_KIND_REPLY

    turn = _turn(turn_id="s:2", H="99.0", stream_build_present=True)
    turn.mark_destination_committed(first_ts="50.0", kind=DEST_KIND_REPLY, text="hello there",
                                   channel_id="C1", thread_root_ts="9.0")
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply") is True

    row = sink("turn_outcome")[0]
    assert row["turn_id"] == "s:2" and row["kind"] == "reply"
    assert row["H"] == "99.0" and row["stream_build_present"] is True
    assert row["chars"] == len("hello there")
    assert row["destinations"] == [{"channel_id": "C1", "thread_root_ts": "9.0",
                                   "first_ts": "50.0", "state": "committed",
                                   "chars": 11, "kind": "reply"}]
    assert "text" not in row["destinations"][0]   # the ledger records a length, never the reply


def test_a_turn_outcome_carries_the_destination_contract_miss(sink):
    """W4's battery metric, and it has to ride HERE. The `visible_action` copy belongs to the
    gate population, and the ungated channel turns — most of the selectable ones — never emit a
    terminal at all, so a miss rate read from that alone would be blind to exactly the traffic
    the marker was built for."""
    turn = _turn(turn_id="s:11")
    turn.destination_selected = False
    turn.settle_default_destination()          # words arrived, no marker ever did
    assert turn.destination_contract_miss is True

    pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply")

    assert sink("turn_outcome")[0]["destination_contract_miss"] is True


def test_a_turn_that_placed_its_own_reply_writes_no_miss(sink):
    """Written ONLY when true, the same convention the terminal copy follows: a column that is
    false on nearly every row costs bytes on every line to say nothing."""
    message = SimpleNamespace(thread_id="10.0")
    turn = _turn(turn_id="s:12")
    turn.destination_selected = False
    turn.destination_source = "default"
    turn.select_destination("channel", message=message)

    pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply")

    row = sink("turn_outcome")[0]
    assert "destination_contract_miss" not in row
    # …and neither does a turn that never had the choice in the first place.
    pt.emit_turn_outcome(_turn(turn_id="s:13"), channel_id="C1", trigger_ts="11.0", kind="reply")
    assert "destination_contract_miss" not in sink("turn_outcome")[1]


def test_a_turn_outcome_is_emitted_exactly_once(sink):
    """The same guard `finish_attempt` has, for the same reason: two emitters each believing they
    own the end of a turn is how the gate population came to be double-counted."""
    turn = _turn(turn_id="s:3")
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="silence") is True
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply") is False
    assert [r["kind"] for r in sink("turn_outcome")] == ["silence"]


def test_an_interrupted_stream_stays_observed_only_and_is_still_reported(sink):
    """[r4-6] A stream Slack accepted and that never finished is genuinely both facts: the room
    saw it, and it is not the answer. Reporting only committed records would hide the delivery;
    reporting it as committed would claim an answer that was never written."""
    from message_processor.turn_runtime import DEST_KIND_STREAM

    turn = _turn(turn_id="s:4", stream_build_present=True)
    turn.note_destination_observed(channel_id="C1", first_ts="50.0", kind=DEST_KIND_STREAM,
                                  thread_root_ts="9.0")
    pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="interrupted")

    row = sink("turn_outcome")[0]
    assert [d["state"] for d in row["destinations"]] == ["observed"]
    assert row["destinations"][0]["chars"] is None
    assert "chars" not in row                      # nothing committed, so no total to report
    assert turn.committed_destinations == []       # ...and memory extraction sees nothing


def test_a_fail_closed_turn_names_its_code_and_says_no_stream_was_built(sink):
    turn = _turn(turn_id="s:5", turn_error="coverage_not_ready", stream_build_present=False)
    pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="error")
    row = sink("turn_outcome")[0]
    assert row["error"] == "coverage_not_ready" and row["stream_build_present"] is False
    assert row["destinations"] == []               # written empty: silence and lost records differ


def test_a_turn_outcome_is_not_a_terminal_event(sink):
    """The two populations answer different questions. A turn_outcome counted as a terminal would
    put every ungated mention into a ledger documented as gate decisions."""
    message = _msg()
    pt.begin_attempt(message)
    pt.finish_attempt(message, "reply")
    pt.emit_turn_outcome(_turn(turn_id="s:6"), channel_id="C1", trigger_ts="10.0", kind="reply")
    assert len(sink("visible_action")) == 1
    assert len(sink("turn_outcome")) == 1


def test_emit_turn_outcome_never_raises_on_a_broken_turn(sink):
    """It runs in the outer finally of every turn. A raise there would turn one lost line into
    a turn that never released its lease."""
    assert pt.emit_turn_outcome(None, channel_id="C1", trigger_ts="1.0", kind="reply") is False
    # No __dict__, so even the once-only stamp fails: reported False, nothing written, no raise.
    assert pt.emit_turn_outcome(object(), channel_id="C1", trigger_ts="1.0", kind="reply") is False
    assert sink("turn_outcome") == []


# --------------------------------------------------------------------------- stream_render

def test_stream_render_carries_the_builders_own_fields(sink):
    """The payload is ChannelStream.stream_render_fields() passed through verbatim — the
    serializer owns what identifies a build, and restating those keys here would be a second
    thing to keep in step with it."""
    fields = {"channel_id": "C1", "snapshot_id": None, "generation": None, "boundary": "5.0",
              "floor_inclusive": True, "H": "99.0", "inventory_start_ts": "5.0",
              "serializer_version": 1, "serializer_config_hash": "cfg", "actor_map_hash": "am",
              "sidecar_versions_hash": "sc", "capability_profile_hash": "cp", "byte_count": 120,
              "message_count": 3, "stream_sha256": "deadbeef", "receipts_included_count": 1,
              "receipts_excluded_count": 2, "receipts_membership_hash": "mh"}
    pt.stream_render(turn_id="s:7", origin_thread_ts="9.0", trigger_ts="10.0", **fields)

    row = sink("stream_render")[0]
    assert row["turn_id"] == "s:7" and row["channel_id"] == "C1"
    assert (row["boundary"], row["floor_inclusive"], row["H"]) == ("5.0", True, "99.0")
    assert row["stream_sha256"] == "deadbeef" and row["message_count"] == 3
    assert row["receipts_membership_hash"] == "mh"
    # P2 pins no snapshot, and a None field is OMITTED rather than written as null — so a
    # group-by never gets an "absent" and a "null" bucket meaning the same thing.
    assert "snapshot_id" not in row and "generation" not in row


def test_a_real_build_emits_one_stream_render_that_matches_its_own_fields(sink):
    from tests.unit.channel_turn_harness import build_stream, normalized
    from message_processor.channel_stream import (PageCounts, StreamBuildResult,
                                                  _emit_stream_render)

    def _carrier(stream):
        return StreamBuildResult(stream=stream, reselected=False, anchor_advanced=False,
                                 pages=PageCounts(history=1, reply=0, origin=0))

    stream = build_stream([normalized("10.0", "hello")])
    _emit_stream_render(_carrier(stream), turn_id="s:8", origin_root_ts="10.0",
                        trigger_ts="10.0")
    row = sink("stream_render")[0]
    for key, value in stream.stream_render_fields().items():
        if value is None:
            assert key not in row
        else:
            assert row[key] == value


# ------------------------------------------------------------------- receipts and snapshots

def test_an_outbound_receipt_row_records_the_refusal_not_the_intent(sink):
    """`applied=false` with a reason is the interesting row: a register the lattice absorbed, a
    finalize a foreign turn owns and a demote with nothing to demote are identical from the call
    site and three different facts about the stream."""
    pt.outbound_receipt(channel_id="C1", message_ts="50.0", owner_turn_id="s:1", op="register",
                        prior_state="finalized", new_state="finalized", applied=False,
                        reason="absorbed_finalized")
    row = sink("outbound_receipt")[0]
    assert row["applied"] is False               # False is WRITTEN; only None is omitted
    assert row["reason"] == "absorbed_finalized"
    assert (row["op"], row["prior_state"], row["new_state"]) == (
        "register", "finalized", "finalized")
    assert row["owner_turn_id"] == "s:1" and row["message_ts"] == "50.0"


# --------------------------------------------------------------------------- model_response

def test_model_attempts_are_sequenced_per_turn_not_per_process(sink):
    """The question is "how many calls did THIS turn cost", which a process-wide counter cannot
    answer. Each tool-loop round is its own attempt: the loop issues one API call per round."""
    turn = _turn(turn_id="s:9")
    first = pt.ModelAttemptSink(turn=turn)
    first.close(first.open("gpt-5.6-sol"), status="ok", input_tokens=100, output_tokens=20,
                cached_input_tokens=64)
    forked = pt.ModelAttemptSink(turn=turn, fork_reason="mcp_retry")
    forked.close(forked.open("gpt-5.6-sol"), status="error", detail="APITimeoutError")

    rows = sink("model_response")
    assert [r["attempt_seq"] for r in rows] == [1, 2]
    assert rows[0]["status"] == "ok" and rows[0]["cached_input_tokens"] == 64
    assert rows[0]["input_tokens"] == 100 and rows[0]["output_tokens"] == 20
    assert "fork_reason" not in rows[0]                    # the first attempt is not a fork
    assert rows[1]["status"] == "error" and rows[1]["fork_reason"] == "mcp_retry"
    assert rows[1]["detail"] == "APITimeoutError"
    assert [a.attempt_seq for a in turn.model_attempts] == [1, 2]


def test_a_model_attempt_sink_never_raises_into_the_api_layer(sink):
    """It runs inside the request wrappers. A telemetry failure there would turn one lost line
    into a lost answer."""
    broken = pt.ModelAttemptSink(turn=object())
    assert broken.open("gpt-5.6-sol") is None
    broken.close(None, status="ok")          # nothing to close
    broken.close(object(), status="ok")      # an attempt whose fields cannot be written
    assert sink("model_response") == []


def test_the_v8_vocabularies_cover_what_the_code_writes():
    assert {"register", "promote", "finalize", "demote", "transfer", "delete",
            "reconcile_finalize", "pending_resolve"} == pt.RECEIPT_OPS
    assert {"absent", "in_flight", "finalized", "chrome"} == pt.RECEIPT_STATES
    assert pt.MODEL_RESPONSE_STATUSES == {"ok", "error"}
    assert pt.TURN_SURFACES == {"channel", "dm"}
    # turn_outcome reuses the terminal vocabulary on purpose: a turn and its gate attempt
    # describe the same room, and two vocabularies for one question make the rows uncomparable.
    assert {"reply", "silence", "interrupted", "error", "queued"} <= pt.KINDS


# --------------------------------------------------- one stream_render per BUILD, joined to it

class _StreamClient:
    """The narrowest client build_channel_stream needs: one history page, no replies."""

    def __init__(self):
        self.self_team_id = "T1"
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app = MagicMock()
        self.app.client.conversations_history = AsyncMock(return_value={
            "ok": True, "messages": [{"ts": "10.0", "text": "hi", "user": "U1"}],
            "has_more": False})
        self.app.client.conversations_replies = AsyncMock(return_value={"ok": True,
                                                                       "messages": []})

    def is_own_message(self, msg):
        return bool(msg) and msg.get("user") == self.bot_user_id

    def classify_sender(self, msg):
        return "self" if self.is_own_message(msg) else "human"

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25):
        return {uid: f"name-{uid}" for uid in ids}


def _stream_db(snapshot=None):
    db = MagicMock()
    db.read_channel_window_anchor_async = AsyncMock(return_value={
        "anchor": {"floor_ts": "5.0", "selection_version": 1},
        "inventory": {"inventory_start_ts": "5.0", "bootstrap_status": "complete",
                      "reason": "genesis"}})
    db.read_channel_discovery_roots_async = AsyncMock(return_value={
        "activity_roots": {}, "receipt_roots": ()})
    db.read_channel_sidecars_for_async = AsyncMock(return_value={
        "ids": [], "receipt_feature_epoch_ts": None, "receipts": [],
        "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
        "tool_usage": {}, "versions_hash": "h"})
    db.advance_channel_window_anchor_async = AsyncMock(return_value=True)
    db.clear_thread_dirty_async = AsyncMock(return_value=True)
    return db


@pytest.fixture
def _stream_singletons():
    from slack_client import actor_tail as actor_tail_module
    from slack_client import admission_watermark
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()
    yield
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()


@pytest.mark.asyncio
async def test_a_build_writes_one_stream_render_naming_its_turn(sink, _stream_singletons):
    """Once per BUILD, joined to the turn by turn_id — that join is the only way to ask which
    window an answer was written from."""
    from message_processor.channel_stream import build_channel_stream

    result = await build_channel_stream(
        client=_StreamClient(), db=_stream_db(), team_id="T1", channel_id="C1", h="99.0",
        turn_id="s:10", origin_root_ts="10.0", trigger_ts="10.0")
    stream = result.stream

    rows = sink("stream_render")
    assert len(rows) == 1
    assert rows[0]["turn_id"] == "s:10" and rows[0]["channel_id"] == "C1"
    assert rows[0]["origin_thread_ts"] == "10.0" and rows[0]["trigger_ts"] == "10.0"
    assert rows[0]["stream_sha256"] == stream.stream_sha256
    assert rows[0]["H"] == "99.0"


def test_base_py_hands_the_builder_the_turn_it_is_building_for():
    """Source-level, because the wiring is what breaks silently: a stream_render with no turn_id
    joins to nothing and the whole event stops answering the question it exists for."""
    import inspect

    from message_processor.base import MessageProcessor

    # BOTH frames: P4 moved the builder call itself into `_channel_stream_call` so the §1a
    # ordering above it stays readable. Following the call is the point — a module-wide substring
    # search would let a genuine future disconnection pass, which is the failure this test exists
    # to catch.
    source = (inspect.getsource(MessageProcessor._build_channel_turn_stream)
              + inspect.getsource(MessageProcessor._channel_stream_call))
    # `origin_root_ts` replaced `origin_thread_ts`: the origin thread is a BUILD INPUT now — it
    # is fetched, not selected out of an existing window — and the emitter reads that same value
    # rather than being handed the root a second time under a telemetry-only name.
    for kwarg in ("turn_id=", "origin_root_ts=", "trigger_ts="):
        assert kwarg in source
    # And the kwargs are on the BUILDER CALL, not merely somewhere in the two functions.
    call = inspect.getsource(MessageProcessor._channel_stream_call)
    assert call.count("build_channel_stream(") == 1
    for kwarg in ("turn_id=", "origin_root_ts=", "trigger_ts="):
        assert kwarg in call.split("build_channel_stream(", 1)[1]


_HASH = "a" * 64


def stream_render_row(**overrides):
    """A COMPLETE `stream_render` row — every §8 field, all of them mandatory.

    Shared rather than duplicated because two tests need it for opposite reasons: T8 asserts a
    real ledger passes the checker, and T54 asserts each rule by breaking exactly one field at a
    time. A minimal row would fail T8 the moment the fields became mandatory, so the fixture and
    the rule that makes it necessary land together.
    """
    row = {
        "v": 10, "at": 3.0, "session": "S", "gate_contract": "binary-v1",
        "event": "stream_render", "turn_id": "t1",
        "channel_id": "C1", "H": "1.9",
        "periphery_floor_ts": "1.0", "inventory_start_ts": "0.5",
        "inventory_state": "warm",
        "stream_sha256": _HASH, "union_sha256": _HASH, "serializer_config_hash": _HASH,
        "sidecar_versions_hash": _HASH, "actor_map_hash": _HASH,
        "receipts_membership_hash": _HASH, "capability_profile_hash": _HASH,
        "byte_count": 10, "origin_byte_count": 4, "message_count": 1, "origin_count": 1,
        "candidate_count": 3, "root_count": 1, "orphan_root_count": 0,
        "receipts_included_count": 1, "receipts_excluded_count": 0,
        "history_pages": 1, "reply_pages": 0, "origin_pages": 1,
        "selection_version": 1, "serializer_version": 3,
        "reselected": False, "anchor_advanced": False,
    }
    row.update(overrides)
    return row


def test_the_ledger_has_no_compaction_vocabulary(tmp_path):
    """T8. `compaction_snapshot` and the outbox that carried it are gone, and CONTRACT_VERSION
    did NOT move for it.

    That was the claim worth testing when the removal shipped inside v8: the turn population —
    turn_start, turn_outcome, stream_render, model_response, outbound_receipt — was unchanged in
    identity, and no completeness rule ever named the removed event, so no denominator anybody
    computes off a v8 ledger was invalidated. The version sits at 10 NOW because stale
    reconsideration (v9) and then edit_own_message (v10) each changed the contract for real —
    unlike this removal, which still earned no bump of its own.
    """
    assert pt.CONTRACT_VERSION == 10
    for name in ("compaction_snapshot", "SNAPSHOT_OPS", "OUTBOX_OPS", "BUILD_STATUSES",
                 "BUILD_REASON_STATUSES", "FIT_RESULTS", "OUTBOX_EVENT",
                 "canonical_body_bytes", "extract_canonical_body", "validate_outbox_body",
                 "emit_outbox_body", "flush_sync"):
        assert not hasattr(pt, name), f"participation_telemetry still exports {name}"

    # And the checker accepts a real ledger, with no build-before-publish rule left to impose.
    import json
    import subprocess
    import sys
    from pathlib import Path

    checker = Path(__file__).resolve().parents[2] / "tools" / "participation_ledger_check.py"
    source = checker.read_text(encoding="utf-8")
    for name in ("compaction_snapshot", "compaction_publish_without_build",
                 "compaction_publish_before_build", "_check_compaction_outbox", "OUTBOX_OPS"):
        assert name not in source, f"the checker still carries {name}"

    ledger = tmp_path / "participation.jsonl"
    rows = [
        {"v": 10, "at": 1.0, "session": "S", "gate_contract": "binary-v1",
         "event": "session_start", "build": "abc"},
        {"v": 10, "at": 2.0, "session": "S", "gate_contract": "binary-v1", "event": "turn_start",
         "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "t1", "surface": "channel",
         "gated": False},
        stream_render_row(),
        {"v": 10, "at": 4.0, "session": "S", "gate_contract": "binary-v1",
         "event": "turn_outcome", "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "t1",
         "kind": "silence", "detached_started": False, "stream_build_present": True,
         "destinations": [], "edits": []},
        {"v": 10, "at": 5.0, "session": "S", "gate_contract": "binary-v1",
         "event": "session_end"},
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = subprocess.run([sys.executable, str(checker), str(ledger), "--json"],
                            capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["violations"] == []


# ================================================ T54 — the stream_render checker contract (§8a)

def _run_checker(tmp_path, rows):
    """Run the REAL checker over a ledger built from `rows`, and return its violation names."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    checker = Path(__file__).resolve().parents[2] / "tools" / "participation_ledger_check.py"
    ledger = tmp_path / "participation.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = subprocess.run([sys.executable, str(checker), str(ledger), "--json"],
                            capture_output=True, text=True, cwd=str(tmp_path))
    payload = json.loads(result.stdout)
    return result.returncode, [v["name"] for v in payload["violations"]]


def _ledger(*, render=None, renders=None, stream_build_present=True, outcome_extra=None):
    """A complete one-turn session, with the stream_render row(s) under test substituted in."""
    if renders is None:
        renders = [] if render is None else [render]
    outcome = {"v": 10, "at": 4.0, "session": "S", "gate_contract": "binary-v1",
               "event": "turn_outcome", "channel_id": "C1", "trigger_ts": "1.0",
               "turn_id": "t1", "kind": "silence", "detached_started": False,
               "stream_build_present": stream_build_present, "destinations": [], "edits": []}
    outcome.update(outcome_extra or {})
    return [
        {"v": 10, "at": 1.0, "session": "S", "gate_contract": "binary-v1",
         "event": "session_start", "build": "abc"},
        {"v": 10, "at": 2.0, "session": "S", "gate_contract": "binary-v1",
         "event": "turn_start", "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "t1",
         "surface": "channel", "gated": False},
        *renders,
        outcome,
        {"v": 10, "at": 5.0, "session": "S", "gate_contract": "binary-v1",
         "event": "session_end"},
    ]


def test_the_stream_render_contract_is_enforced(tmp_path):
    """T54. Every §8a rule, asserted by BREAKING ONE FIELD AT A TIME against the real checker.

    The row is the only durable evidence of what the model was shown. A checker that accepts a
    malformed one turns a bad answer into an unexplainable one months later, which is when
    somebody actually reads these.
    """
    # The complete row passes — the baseline every case below deviates from by exactly one field.
    assert _run_checker(tmp_path, _ledger(render=stream_render_row())) == (0, [])

    # RULE 1 — THE HASHES. Six have no caller-omitted path, so empty is a defect.
    for field in ("stream_sha256", "union_sha256", "serializer_config_hash",
                  "sidecar_versions_hash", "actor_map_hash", "receipts_membership_hash"):
        code, names = _run_checker(tmp_path, _ledger(render=stream_render_row(**{field: ""})))
        assert code == 1 and "stream_render_bad_hash" in names, field
    # `capability_profile_hash` ALONE may be empty: the builder defaults it that way, so the
    # probe and every utility build legitimately emit one.
    assert _run_checker(
        tmp_path, _ledger(render=stream_render_row(capability_profile_hash=""))) == (0, [])
    # Wrong shape is a defect for all seven — uppercase is not lowercase hex, and 63 is not 64.
    for bad in (_HASH.upper(), _HASH[:63], "not-a-hash", 12345):
        code, names = _run_checker(
            tmp_path, _ledger(render=stream_render_row(stream_sha256=bad)))
        assert code == 1 and "stream_render_bad_hash" in names, bad

    # RULE 2 — THE COUNTS. Twelve non-negative ints; bools are not ints here.
    for field in ("byte_count", "origin_byte_count", "message_count", "origin_count",
                  "candidate_count", "root_count", "orphan_root_count",
                  "receipts_included_count", "receipts_excluded_count",
                  "history_pages", "reply_pages", "origin_pages"):
        code, names = _run_checker(tmp_path, _ledger(render=stream_render_row(**{field: -1})))
        assert code == 1 and "stream_render_bad_count" in names, field
        code, names = _run_checker(tmp_path, _ledger(render=stream_render_row(**{field: True})))
        assert code == 1 and "stream_render_bad_count" in names, field
    # THE THREE PAGE COUNTS ARE THREE FIELDS, never one sum: only separate counts can show the
    # history walk stayed inside its ceiling while the reply fan-out ran unbounded.
    row = stream_render_row()
    assert {"history_pages", "reply_pages", "origin_pages"} <= set(row)
    assert "periphery_pages" not in row

    # VERSIONS SIT OUTSIDE THE COUNT RULE — validated as ints and nothing more. A bound on them
    # would be a bound on how many times the format may change.
    for field in ("selection_version", "serializer_version"):
        assert _run_checker(
            tmp_path, _ledger(render=stream_render_row(**{field: 9999}))) == (0, [])
        code, names = _run_checker(tmp_path,
                                   _ledger(render=stream_render_row(**{field: "3"})))
        assert code == 1 and "stream_render_bad_count" in names, field

    # ROOT_COUNT <= MESSAGE_COUNT. Roots are a subset of the rendered message items, so this
    # catches a count computed over the wrong subject, which no type check would notice.
    code, names = _run_checker(
        tmp_path, _ledger(render=stream_render_row(root_count=5, message_count=2)))
    assert code == 1 and "stream_render_bad_count" in names

    # RULE 4 — BOOLEANS, and the two floor strings whose EMPTY VALUE IS A VALUE.
    for field in ("reselected", "anchor_advanced"):
        code, names = _run_checker(tmp_path,
                                   _ledger(render=stream_render_row(**{field: "false"})))
        assert code == 1 and "stream_render_bad_bool" in names, field
    for field in ("periphery_floor_ts", "inventory_start_ts"):
        assert _run_checker(
            tmp_path, _ledger(render=stream_render_row(**{field: ""}))) == (0, []), field
        code, names = _run_checker(tmp_path, _ledger(render=stream_render_row(**{field: 1.0})))
        assert code == 1 and "stream_render_bad_field" in names, field

    # INVENTORY_STATE — one of the SIX, and the list is closed.
    for state in ("absent", "cold", "warm", "limited_retention", "limited_depth", "unavailable"):
        assert _run_checker(
            tmp_path, _ledger(render=stream_render_row(inventory_state=state))) == (0, []), state
    code, names = _run_checker(
        tmp_path, _ledger(render=stream_render_row(inventory_state="pending")))
    assert code == 1 and "stream_render_bad_inventory_state" in names

    # PRESENCE — every mandatory field. Absence and None are ONE case, because record() omits
    # None-valued fields rather than writing null.
    for field in plc.STREAM_RENDER_MANDATORY:
        row = stream_render_row()
        row.pop(field)
        code, names = _run_checker(tmp_path, _ledger(render=row))
        assert code == 1 and "stream_render_missing_field" in names, field

    # RETIRED FIELDS ARE REJECTED, each one actually sent through the checker. Asserting the
    # constant lists them would prove only that I typed them; these rows prove the checker acts.
    for retired, value in (("snapshot_id", "snap-1"), ("generation", 4), ("boundary", "1.0"),
                           ("floor_inclusive", True), ("coverage_start_ts", "0.5"),
                           ("selection_result", "reanchored"), ("reanchored", True)):
        code, names = _run_checker(
            tmp_path, _ledger(render=stream_render_row(**{retired: value})))
        assert code == 1 and "stream_render_retired_field" in names, retired

    # RULE 3 — THE turn_error VOCABULARY, and retired codes are VIOLATIONS not grandfathered:
    # a fresh row carrying one means a producer survived the excision.
    for code_name in ("stream_data_invalid", "stream_over_budget", "history_fetch_failed",
                      "origin_fetch_failed"):
        assert _run_checker(tmp_path, _ledger(
            render=stream_render_row(),
            outcome_extra={"error": code_name})) == (0, []), code_name
    for retired in ("snapshot_unsupported", "coverage_not_ready", "invented_code"):
        code, names = _run_checker(tmp_path, _ledger(
            render=stream_render_row(), outcome_extra={"error": retired}))
        assert code == 1 and "turn_outcome_bad_error" in names, retired

    # RULE 5 — ALL THREE CARDINALITY DIRECTIONS.
    code, names = _run_checker(tmp_path, _ledger(render=None, stream_build_present=True))
    assert code == 1 and "stream_render_absent_for_build" in names
    code, names = _run_checker(tmp_path, _ledger(render=stream_render_row(),
                                                 stream_build_present=False))
    assert code == 1 and "stream_render_without_build" in names
    # TWO ROWS FOR ONE turn_id — the direction "exactly one" needs and presence cannot give.
    code, names = _run_checker(tmp_path, _ledger(
        renders=[stream_render_row(), stream_render_row(at=3.5)]))
    assert code == 1 and "stream_render_duplicate" in names


# ==================================== CV9 — stale reconsideration (§5 of the spec), now at CV10

def test_a_stale_send_row_joins_the_turn_population_by_turn_id(sink):
    """v9. The row's turn_id is what lets a turn's suppression EVENTS be counted beside its
    reconsider_start rows. An ungated turn still mints no attempt, and the absent optional is
    OMITTED, never null."""
    pt.stale_send("C1", "10.0", turn_id="t1", last_seen_ts="9.0", observed_latest_ts="11.0",
                  scope="thread", surface="reply", guard_mode="buffered")
    row = sink("stale_send")[0]
    assert row["turn_id"] == "t1"
    assert row["scope"] == "thread"           # scope[0]-only, unchanged at v9
    assert "attempt_id" not in row
    assert None not in row.values()


def test_a_reconsider_start_writes_the_literal_v9_keys(sink):
    """The wire key is `pass` — a Python keyword, so the helper takes `pass_number` and the
    grammar keeps the literal name. `scope` is the FULL three-part tuple, as a JSON list."""
    pt.reconsider_start("C1", "10.0", turn_id="t1", pass_number=1,
                        scope=("thread", "C1", "9.0"), observed_latest_ts="11.0",
                        attempt_id="A1", model_attempt_seq=3)
    row = sink("reconsider_start")[0]
    assert row["pass"] == 1
    assert "pass_number" not in row
    assert row["scope"] == ["thread", "C1", "9.0"]
    assert row["observed_latest_ts"] == "11.0"
    assert (row["turn_id"], row["attempt_id"], row["model_attempt_seq"]) == ("t1", "A1", 3)
    assert (row["channel_id"], row["trigger_ts"]) == ("C1", "10.0")


def test_unavailable_optional_reconsider_fields_are_omitted_not_null(sink):
    """The v9 null rule: an ungated channel turn has no attempt_id, a failed attempt-sink open
    has no model_attempt_seq, a skip has no forced and no error — every one of them is ABSENT
    from the line, matching record()'s drop-None behavior, so a reader never sees two buckets
    (absent and null) meaning 'unavailable'."""
    pt.reconsider_start("C1", "10.0", turn_id="t1", pass_number=2,
                        scope=("thread", "C1", "9.0"), observed_latest_ts="11.0")
    row = sink("reconsider_start")[0]
    assert "attempt_id" not in row and "model_attempt_seq" not in row
    assert None not in row.values()

    pt.reconsider_outcome("C1", "10.0", turn_id="t1", outcome="skipped", passes=2)
    out = sink("reconsider_outcome")[0]
    assert (out["outcome"], out["passes"]) == ("skipped", 2)
    for absent in ("forced", "error", "attempt_id"):
        assert absent not in out
    assert None not in out.values()


def test_a_reconsider_outcome_carries_its_conditionals_only_where_they_apply(sink):
    """`forced` rides only on posted outcomes — and False is WRITTEN, because only None means
    unavailable. `error` rides only on error_dropped, carrying the §4f subtype."""
    pt.reconsider_outcome("C1", "10.0", turn_id="t1", outcome="posted_asis", passes=1,
                          forced=False)
    pt.reconsider_outcome("C1", "10.0", turn_id="t2", outcome="error_dropped", passes=2,
                          error="delivery_failed")
    rows = sink("reconsider_outcome")
    assert rows[0]["forced"] is False and "error" not in rows[0]
    assert rows[1]["error"] == "delivery_failed" and "forced" not in rows[1]


def test_reconsider_facts_omit_inapplicable_keys():
    """as_payload() feeds turn_outcome's nested `reconsider` object, and nested values survive
    record()'s top-level drop-None untouched — so the omission has to happen HERE, or a null
    would reach the file."""
    from message_processor.turn_runtime import ReconsiderFacts

    assert ReconsiderFacts(outcome="posted_revised", passes=3, forced=True).as_payload() == {
        "outcome": "posted_revised", "passes": 3, "forced": True}
    assert ReconsiderFacts(outcome="posted_asis", passes=1, forced=False).as_payload() == {
        "outcome": "posted_asis", "passes": 1, "forced": False}
    assert ReconsiderFacts(outcome="error_dropped", passes=2,
                           error="context_rebuild").as_payload() == {
        "outcome": "error_dropped", "passes": 2, "error": "context_rebuild"}
    # Inapplicable keys are dropped even when SET: forced on a skip, error on a fuse drop.
    assert ReconsiderFacts(outcome="skipped", passes=1, forced=True,
                           error="model_failure").as_payload() == {
        "outcome": "skipped", "passes": 1}
    assert ReconsiderFacts(outcome="fuse_dropped", passes=5,
                           error="model_failure").as_payload() == {
        "outcome": "fuse_dropped", "passes": 5}
    # ...and a posted outcome whose forced was never recorded carries no null either.
    assert ReconsiderFacts(outcome="posted_asis", passes=1).as_payload() == {
        "outcome": "posted_asis", "passes": 1}


def test_a_turn_outcome_attaches_the_reconsider_facts_and_only_then(sink):
    """emit_turn_outcome reads TurnRuntime.reconsider — absent when no reconsideration ran, the
    as_payload() dict verbatim when one did."""
    from message_processor.turn_runtime import ReconsiderFacts

    plain = _turn(turn_id="s:30")
    pt.emit_turn_outcome(plain, channel_id="C1", trigger_ts="10.0", kind="reply")
    reconsidered = _turn(turn_id="s:31")
    reconsidered.reconsider = ReconsiderFacts(outcome="posted_revised", passes=2, forced=False)
    pt.emit_turn_outcome(reconsidered, channel_id="C1", trigger_ts="11.0", kind="reply")

    rows = sink("turn_outcome")
    assert "reconsider" not in rows[0]
    assert rows[1]["reconsider"] == {"outcome": "posted_revised", "passes": 2, "forced": False}


# -------------------------- the CV9 checker invariants, per §5 — unchanged and graded at CV10

def _v10(event, **fields):
    row = {"v": 10, "at": 3.0, "session": "S", "gate_contract": "binary-v1", "event": event}
    row.update(fields)
    return row


def _suppression(turn_id="t1", **over):
    row = _v10("stale_send", channel_id="C1", trigger_ts="1.0", turn_id=turn_id,
              last_seen_ts="1.0", observed_latest_ts="2.0", scope="thread",
              surface="reply", guard_mode="buffered")
    row.update(over)
    return row


def _start(pass_number, turn_id="t1", **over):
    row = _v10("reconsider_start", at=4.0 + pass_number, channel_id="C1", trigger_ts="1.0",
              turn_id=turn_id, scope=["thread", "C1", "1.0"], observed_latest_ts="2.0")
    row["pass"] = pass_number
    row.update(over)
    return row


def _outcome(outcome="skipped", passes=1, turn_id="t1", **over):
    row = _v10("reconsider_outcome", at=8.0, channel_id="C1", trigger_ts="1.0",
              turn_id=turn_id, outcome=outcome, passes=passes)
    row.update(over)
    return row


def _reconsider_session(*middle):
    """One UNGATED channel turn (no attempt_id anywhere — the join is turn_id alone) that ended
    stale_suppressed, with the reconsideration rows under test in the middle."""
    return [
        _v10("session_start", at=1.0, build="abc"),
        _v10("turn_start", at=2.0, channel_id="C1", trigger_ts="1.0", turn_id="t1",
            surface="channel", gated=False),
        *middle,
        _v10("turn_outcome", at=9.0, channel_id="C1", trigger_ts="1.0", turn_id="t1",
            kind="stale_suppressed", detached_started=False, stream_build_present=False,
            destinations=[], edits=[]),
        _v10("session_end", at=10.0),
    ]


def test_the_checker_accepts_a_healthy_ungated_reconsideration(tmp_path):
    """The baseline every violation case deviates from: two suppression events, two contiguous
    passes, one outcome whose `passes` is the started-pass count — joined by turn_id with no
    attempt_id on any row, because an ungated channel turn mints none."""
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _suppression(at=4.5), _start(2),
        _outcome(outcome="posted_asis", passes=2, forced=True)))
    assert (code, names) == (0, [])


def test_the_checker_accepts_the_passes_accounting_boundaries(tmp_path):
    """A fuse drop records 5 — five suppression events, five started passes; a cancellation
    before any pass started records 0. Both are legal `passes` values, and the checker must not
    invent a floor of 1 for them."""
    fuse = [_row for n in range(1, 6) for _row in (_suppression(at=3.0 + n), _start(n))]
    code, names = _run_checker(tmp_path, _reconsider_session(
        *fuse, _outcome(outcome="fuse_dropped", passes=5)))
    assert (code, names) == (0, [])

    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _outcome(outcome="cancelled", passes=0)))
    assert (code, names) == (0, [])


def test_pass_numbers_must_be_contiguous_from_one(tmp_path):
    # A gap.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _suppression(at=4.5), _start(3), _outcome(passes=2)))
    assert code == 1 and "reconsider_pass_not_contiguous" in names
    # A sequence that never started at 1.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(2), _outcome(passes=1)))
    assert code == 1 and "reconsider_pass_not_contiguous" in names
    # A repeat.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _suppression(at=4.5), _start(1), _outcome(passes=2)))
    assert code == 1 and "reconsider_pass_duplicate" in names


def test_a_turn_cannot_start_more_passes_than_it_had_suppression_events(tmp_path):
    """Every pass exists because a suppression event preceded it, and each suppression event
    writes exactly one stale_send row — so starts <= stale_sends, per turn, joined on turn_id."""
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _start(2), _outcome(passes=2)))
    assert code == 1 and "reconsider_start_exceeds_stale_send" in names


def test_at_most_one_reconsider_outcome_per_turn(tmp_path):
    """The once-per-turn gate makes a second runner invocation impossible, so a duplicate is an
    emitter defect in any file, fragment or not."""
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=1),
        _outcome(outcome="posted_asis", passes=1, forced=False, at=8.5)))
    assert code == 1 and "reconsider_outcome_duplicate" in names


def test_a_dm_stale_send_with_a_turn_id_and_no_turn_start_is_tolerated(tmp_path):
    """The ruled tolerance: DM turns take leases and emit stale_send rows carrying a turn_id,
    but DMs are outside the channel-turn population, so there is no turn_start to join — and
    that must never read as a violation."""
    rows = [
        _v10("session_start", at=1.0, build="abc"),
        _suppression(turn_id="dm-turn", at=2.0),
        _v10("session_end", at=3.0),
    ]
    code, names = _run_checker(tmp_path, rows)
    assert (code, names) == (0, [])


def test_the_reconsider_event_grammar_is_enforced(tmp_path):
    """One field broken at a time against the real checker, like the stream_render contract."""
    # Mandatory keys, on both events.
    for field in ("turn_id", "channel_id", "trigger_ts", "pass", "scope", "observed_latest_ts"):
        start = _start(1)
        start.pop(field)
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), start, _outcome(passes=1)))
        assert code == 1 and "reconsider_start_missing_field" in names, field
    for field in ("turn_id", "channel_id", "trigger_ts", "outcome", "passes"):
        outcome = _outcome(passes=1)
        outcome.pop(field)
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1), outcome))
        assert code == 1 and "reconsider_outcome_missing_field" in names, field

    # `pass` counts from 1; `scope` is the FULL three-part list; the seq is an int.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(0), _outcome(passes=1)))
    assert code == 1 and "reconsider_start_bad_pass" in names
    for bad_scope in ("thread", ["thread", "C1"], None):
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1, scope=bad_scope), _outcome(passes=1)))
        assert code == 1 and "reconsider_start_bad_scope" in names, bad_scope
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, model_attempt_seq="3"), _outcome(passes=1)))
    assert code == 1 and "reconsider_start_bad_field" in names

    # The outcome vocabulary is closed, and `passes` is a non-negative int.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(outcome="gave_up", passes=1)))
    assert code == 1 and "reconsider_outcome_bad_outcome" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=-1)))
    assert code == 1 and "reconsider_outcome_bad_passes" in names

    # The conditionals: forced only on posted outcomes, error only on error_dropped, and the
    # error subtype comes from the closed §4f set.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(outcome="skipped", passes=1, forced=True)))
    assert code == 1 and "reconsider_outcome_bad_conditional" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(outcome="skipped", passes=1, error="model_failure")))
    assert code == 1 and "reconsider_outcome_bad_conditional" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1),
        _outcome(outcome="error_dropped", passes=1, error="mystery")))
    assert code == 1 and "reconsider_outcome_bad_conditional" in names
    # ...and the legal shapes of both conditionals pass.
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1),
        _outcome(outcome="error_dropped", passes=1, error="guard_rearm_failed")))
    assert (code, names) == (0, [])


def test_a_v9_stale_send_must_carry_its_turn_id(tmp_path):
    """The pass-count invariant counts suppression EVENTS per turn, so a v9 row with no turn_id
    counts toward no turn and quietly shrinks the denominator every `reconsider_start` is
    measured against. Older rows are exempt for free — the checker skips anything below v9. The
    healthy direction is the DM-tolerance case above: a turn_id with nothing to join is fine, a
    missing one is not."""
    orphan = _suppression()
    orphan.pop("turn_id")
    code, names = _run_checker(tmp_path, [
        _v10("session_start", at=1.0, build="abc"),
        orphan,
        _v10("session_end", at=3.0),
    ])
    assert code == 1 and "stale_send_missing_turn_id" in names


def test_a_stale_send_turn_id_must_be_a_non_empty_string(tmp_path):
    """The type half of the same rule: an explicit null, a non-string or an empty string would
    all be dropped SILENTLY by the checker's own join index — a suppression event that counts
    toward no turn, shrinking the pass-count denominator with no finding. The grammar flags
    each by name instead."""
    for bad in (None, 7, ""):
        code, names = _run_checker(tmp_path, [
            _v10("session_start", at=1.0, build="abc"),
            _suppression(turn_id=bad),
            _v10("session_end", at=3.0),
        ])
        assert code == 1 and "stale_send_missing_turn_id" in names, bad


def test_turn_start_and_turn_outcome_null_turn_ids_are_violations_not_silent_drops(tmp_path):
    """The join-key rule covers EVERY turn-joined event, not only the three v9 rows with a
    reject_null mandatory check: a `turn_start.turn_id = null` plus `turn_outcome.turn_id =
    null` used to pass with zero violations — both rows silently discarded by the join index,
    a whole turn vanishing from every invariant with no finding."""
    for event, name in (("turn_start", "turn_start_missing_field"),
                        ("turn_outcome", "turn_outcome_missing_field")):
        code, names = _run_checker(tmp_path, [
            _v10("session_start", at=1.0, build="abc"),
            _v10("turn_start", at=1.5, turn_id=None if event == "turn_start" else "t1",
                channel_id="C1", surface="channel"),
            _v10("turn_outcome", at=2.0, turn_id=None if event == "turn_outcome" else "t1",
                kind="reply", destinations=[], stream_build_present=True, edits=[]),
            _v10("session_end", at=3.0),
        ])
        assert code == 1 and name in names, event


def test_a_reconsider_optional_present_with_explicit_null_is_a_violation(tmp_path):
    """No JSON nulls anywhere on the two v9 reconsider events — an unavailable OPTIONAL is
    omitted by record()'s drop-None, so a null that reached the file was written on purpose."""
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, attempt_id=None), _outcome(passes=1)))
    assert code == 1 and "reconsider_start_bad_field" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=1, attempt_id=None)))
    assert code == 1 and "reconsider_outcome_bad_field" in names


def test_an_explicit_null_on_a_reconsider_identity_field_is_a_missing_field(tmp_path):
    """No JSON nulls anywhere in the v9 grammar: record() omits None-valued fields, so a null
    that reaches the file means something wrote one on purpose. It reads as "unavailable" beside
    the absent encoding that already means that, so it fails under the SAME name as absence —
    one defect, one bucket."""
    # The baseline these deviate from, clean.
    assert _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=1))) == (0, [])

    for field in ("turn_id", "channel_id", "trigger_ts", "pass", "scope", "observed_latest_ts"):
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1, **{field: None}), _outcome(passes=1)))
        assert code == 1 and "reconsider_start_missing_field" in names, field
    for field in ("turn_id", "channel_id", "trigger_ts", "outcome", "passes"):
        nulled = _outcome(passes=1)
        nulled[field] = None
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1), nulled))
        assert code == 1 and "reconsider_outcome_missing_field" in names, field


def test_a_reconsider_scope_is_three_strings(tmp_path):
    """The full three-part suppressing scope is the pass's evidence, and a scope part that is not
    a string is not a scope part — a float ts here would compare unequal to every string ts the
    guard actually holds."""
    assert _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, scope=["thread", "C1", "1.0"]),
        _outcome(passes=1))) == (0, [])
    for bad in (["thread", "C1", 2.0], ["thread", None, "1.0"], [1, 2, 3]):
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1, scope=bad), _outcome(passes=1)))
        assert code == 1 and "reconsider_start_bad_scope" in names, bad


def test_unknown_keys_on_the_two_reconsider_events_fail(tmp_path):
    """The literal grammar is CLOSED. A field the emitter invented and no reader knows about is
    how a contract drifts out from under the tool grading it, so it fails rather than being
    ignored — while every legal optional stays silent."""
    # Every optional present at once: the direction a closed-grammar rule most easily breaks.
    assert _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, attempt_id="A1", model_attempt_seq=3),
        _outcome(outcome="posted_revised", passes=1, forced=True,
                 attempt_id="A1"))) == (0, [])
    assert _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, attempt_id="A1", model_attempt_seq=3),
        _outcome(outcome="error_dropped", passes=1, error="request_build",
                 attempt_id="A1"))) == (0, [])

    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1, reviewed_through="2.0"), _outcome(passes=1)))
    assert code == 1 and "reconsider_start_unknown_field" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=1, decision="skip")))
    assert code == 1 and "reconsider_outcome_unknown_field" in names


def test_posted_outcomes_require_forced_and_error_drops_require_a_subtype(tmp_path):
    """The other direction of the conditionals, and it applies to the EVENT only: the emitter
    writes forced=False on every posted outcome and the runner always names a §4f subtype, so an
    absence there is a lost fact rather than an omitted optional. The nested turn_outcome copy
    comes from as_payload(), which legally omits a forced that was never recorded."""
    for posted in ("posted_asis", "posted_revised"):
        assert _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1),
            _outcome(outcome=posted, passes=1, forced=False))) == (0, []), posted
        code, names = _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1), _outcome(outcome=posted, passes=1)))
        assert code == 1 and "reconsider_outcome_bad_conditional" in names, posted

    assert _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1),
        _outcome(outcome="error_dropped", passes=1, error="delivery_exception"))) == (0, [])
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(outcome="error_dropped", passes=1)))
    assert code == 1 and "reconsider_outcome_bad_conditional" in names

    # The asymmetry, pinned: the nested copy without `forced` is HEALTHY.
    rows = _reconsider_session(_suppression(), _start(1),
                               _outcome(outcome="posted_asis", passes=1, forced=False))
    rows[-2]["reconsider"] = {"outcome": "posted_asis", "passes": 1}
    assert _run_checker(tmp_path, rows) == (0, [])


def test_the_reconsider_error_vocabulary_is_the_eight_subtypes(tmp_path):
    """All eight §4f subtypes, each sent through the real checker. Asserting the constant would
    prove only that I typed them; the two the tightening ADDED are the ones that matter."""
    for subtype in ("context_rebuild", "model_failure", "admission_overflow", "delivery_failed",
                    "epoch_invalidated", "guard_rearm_failed", "request_build",
                    "delivery_exception"):
        assert _run_checker(tmp_path, _reconsider_session(
            _suppression(), _start(1),
            _outcome(outcome="error_dropped", passes=1, error=subtype))) == (0, []), subtype


def test_an_outcomes_passes_equals_its_joined_start_count(tmp_path):
    """`passes` IS the started-pass count, so a disagreement means one of the two is counting
    something else — and every later reading of the file inherits that arithmetic. The healthy
    exact match is the baseline above (two suppressions, two passes, passes=2)."""
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _outcome(passes=2)))
    assert code == 1 and "reconsider_outcome_passes_mismatch" in names
    code, names = _run_checker(tmp_path, _reconsider_session(
        _suppression(), _start(1), _suppression(at=4.5), _start(2), _outcome(passes=1)))
    assert code == 1 and "reconsider_outcome_passes_mismatch" in names

    # NOT graded when the turn's head is not in the input: a rotated file legitimately leaves the
    # early passes in participation.jsonl.1, and counting what survived would invent a defect.
    code, names = _run_checker(tmp_path, [
        _suppression(turn_id="rotated", at=2.0),
        _outcome(passes=3, turn_id="rotated"),
    ])
    assert (code, names) == (0, [])


def test_the_nested_turn_outcome_reconsider_payload_is_checked(tmp_path):
    """turn_outcome.reconsider follows the same rules as the event: closed outcome vocabulary,
    int passes, conditionals only where they apply — nested nulls would survive record(), so the
    checker is the tripwire for them."""
    def _with_facts(facts):
        rows = _reconsider_session(_suppression(), _start(1),
                                   _outcome(outcome="posted_asis", passes=1, forced=False))
        rows[-2]["reconsider"] = facts        # the turn_outcome row
        return rows

    code, names = _run_checker(
        tmp_path, _with_facts({"outcome": "posted_asis", "passes": 1, "forced": False}))
    assert (code, names) == (0, [])
    for bad in ({"outcome": "gave_up", "passes": 1},
                {"outcome": "skipped", "passes": None},
                {"outcome": "skipped", "passes": 1, "error": "model_failure"},
                {"outcome": "posted_asis", "passes": 1, "forced": None},
                # r2 finding 5 — the nested grammar is CLOSED, null-free, non-negative:
                {"outcome": "skipped", "passes": 1, "reviewed_through": "1.0"},
                {"outcome": "error_dropped", "passes": 1, "error": None},
                {"outcome": "skipped", "passes": -1},
                "posted_asis"):
        code, names = _run_checker(tmp_path, _with_facts(bad))
        assert code == 1 and "turn_outcome_reconsider_malformed" in names, bad


def test_the_checker_skips_v8_and_v9_rows_rather_than_grading_them(tmp_path):
    """The version moved, so the checker's older-contract rule now covers v9 as well as v8: a
    mixed file across the deploy is the normal state of the world, and grading v9 rows by v10
    rules — every one of which lacks `edits` — would invent violations out of correct
    history."""
    assert plc.CONTRACT_VERSION == 10
    rows = [
        {"v": 8, "at": 1.0, "session": "S8", "gate_contract": "binary-v1",
         "event": "turn_start", "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "old",
         "surface": "channel", "gated": False},   # v8, no outcome — must NOT be graded
        # A v9 turn, complete under ITS contract: turn_outcome has no `edits`, which is a v10
        # violation and a healthy v9 row. Skipped, so it produces nothing.
        {"v": 9, "at": 1.5, "session": "S9", "gate_contract": "binary-v1",
         "event": "turn_start", "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "older",
         "surface": "channel", "gated": False},
        {"v": 9, "at": 1.6, "session": "S9", "gate_contract": "binary-v1",
         "event": "turn_outcome", "channel_id": "C1", "trigger_ts": "1.0", "turn_id": "older",
         "kind": "silence", "detached_started": False, "stream_build_present": False,
         "destinations": []},
        _v10("session_start", at=2.0, build="abc"),
        _v10("session_end", at=3.0),
    ]
    code, names = _run_checker(tmp_path, rows)
    assert (code, names) == (0, [])


# ==================================== CV10 — edit_own_message (EDIT_OWN_MESSAGE.md §7)

def test_a_turn_outcome_always_carries_the_edits_list(sink):
    """v10: `edits` is written even when empty, for the same reason `destinations` is — a turn
    that edited nothing and a turn whose edit records were lost are not the same fact."""
    pt.emit_turn_outcome(_turn(turn_id="s:40"), channel_id="C1", trigger_ts="10.0", kind="reply")
    pt.turn_outcome("C1", "11.0", turn_id="s:41", kind="silence")
    rows = sink("turn_outcome")
    assert rows[0]["edits"] == []
    assert rows[1]["edits"] == []


def test_emit_turn_outcome_reads_turn_edits_with_the_exact_payload(sink):
    """One entry per EditRecord on `turn.edits`, the spec payload verbatim, over the REAL
    §11.6 lifecycle (a record exists only post-acceptance, so both states carry
    announcement_ts; only a committed record has error=None): None values are OMITTED —
    nested values survive record()'s top-level drop-None, so the omission must happen at
    assembly."""
    from types import SimpleNamespace

    committed = SimpleNamespace(channel_id="C1", target_ts="2.0", announcement_ts="3.0",
                                state="committed", error=None)
    partial = SimpleNamespace(channel_id="C1", target_ts="4.0", announcement_ts="5.0",
                              state="announcement_only",
                              error="epoch_refused_after_announcement")
    turn = SimpleNamespace(turn_id="s:42", destinations=[], turn_error=None, H=None,
                           stream_build_present=False, reconsider=None,
                           edits=[committed, partial])
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply")

    row = sink("turn_outcome")[0]
    assert row["edits"] == [
        {"channel_id": "C1", "target_ts": "2.0", "state": "committed",
         "announcement_ts": "3.0"},
        {"channel_id": "C1", "target_ts": "4.0", "state": "announcement_only",
         "announcement_ts": "5.0", "error": "epoch_refused_after_announcement"},
    ]
    for entry in row["edits"]:
        assert None not in entry.values()


def test_reconsider_and_edits_coexist_on_one_emitted_turn_outcome(sink):
    """The spec's coexistence ruling: a turn can be reconsidered AND edit an earlier message,
    and one row carries both facts."""
    from types import SimpleNamespace

    from message_processor.turn_runtime import ReconsiderFacts

    edit = SimpleNamespace(channel_id="C1", target_ts="2.0", announcement_ts="3.0",
                           state="committed", error=None)
    turn = SimpleNamespace(turn_id="s:43", destinations=[], turn_error=None, H=None,
                           stream_build_present=False,
                           reconsider=ReconsiderFacts(outcome="posted_asis", passes=1,
                                                      forced=False),
                           edits=[edit])
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="reply")
    row = sink("turn_outcome")[0]
    assert row["reconsider"] == {"outcome": "posted_asis", "passes": 1, "forced": False}
    assert row["edits"] == [{"channel_id": "C1", "target_ts": "2.0", "state": "committed",
                             "announcement_ts": "3.0"}]


# ------------------------------------------------------ the CV10 checker invariants, per §7

_ANNOUNCEMENT_DEST = {"channel_id": "C1", "thread_root_ts": "2.0", "first_ts": "3.0",
                      "state": "committed", "chars": 40, "kind": "correction_announcement"}


def _edit_entry(**over):
    entry = {"channel_id": "C1", "target_ts": "2.0", "announcement_ts": "3.0",
             "state": "committed"}
    entry.update(over)
    return entry


def _edit_session(edits, destinations=None):
    """One ungated channel turn whose turn_outcome carries the edits under test."""
    return [
        _v10("session_start", at=1.0, build="abc"),
        _v10("turn_start", at=2.0, channel_id="C1", trigger_ts="1.0", turn_id="t1",
             surface="channel", gated=False),
        _v10("turn_outcome", at=9.0, channel_id="C1", trigger_ts="1.0", turn_id="t1",
             kind="reply", detached_started=False, stream_build_present=False,
             destinations=list(destinations or []), edits=edits),
        _v10("session_end", at=10.0),
    ]


def test_a_v10_turn_outcome_with_empty_edits_is_healthy(tmp_path):
    assert _run_checker(tmp_path, _edit_session([])) == (0, [])


def test_a_v10_turn_outcome_without_edits_fails(tmp_path):
    """`edits` is MANDATORY in CV10 — always a list, empty when the turn edited nothing."""
    rows = _edit_session([])
    del rows[2]["edits"]
    code, names = _run_checker(tmp_path, rows)
    assert code == 1 and "turn_outcome_missing_field" in names


def test_the_edits_entry_grammar_is_enforced(tmp_path):
    """One field broken at a time against the real checker, like the reconsider grammar."""
    # The baseline every case deviates from: one committed edit whose announcement joins.
    assert _run_checker(tmp_path, _edit_session(
        [_edit_entry()], destinations=[_ANNOUNCEMENT_DEST])) == (0, [])
    # ...and the announcement_only lifecycle shape — announcement_ts AND error present
    # (§11.6: no `error` on committed is the ONLY legal omission; a bare announcement_only
    # is a lifecycle violation, covered by the lifecycle test below).
    assert _run_checker(tmp_path, _edit_session(
        [{"channel_id": "C1", "target_ts": "2.0", "state": "announcement_only",
          "announcement_ts": "3.0", "error": "stale_target_after_announcement"}],
        destinations=[_ANNOUNCEMENT_DEST])) == (0, [])

    # Not a list at all.
    code, names = _run_checker(tmp_path, _edit_session({"target_ts": "2.0"}))
    assert code == 1 and "turn_outcome_edits_not_list" in names
    # An entry that is not an object.
    code, names = _run_checker(tmp_path, _edit_session(["2.0"]))
    assert code == 1 and "turn_outcome_edit_malformed" in names
    # A state outside the two-value lifecycle.
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(state="silent")], destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_bad_state" in names
    # An unknown nested key — the grammar is CLOSED.
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(note="fixed a typo")], destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_malformed" in names
    # Explicit nulls — unavailable values are omitted, never null.
    for field in ("channel_id", "target_ts", "state", "announcement_ts", "error"):
        code, names = _run_checker(tmp_path, _edit_session(
            [_edit_entry(**{field: None})], destinations=[_ANNOUNCEMENT_DEST]))
        assert code == 1 and "turn_outcome_edit_malformed" in names, field
    # A missing mandatory key.
    for field in ("channel_id", "target_ts", "state"):
        entry = _edit_entry()
        entry.pop(field)
        code, names = _run_checker(tmp_path, _edit_session(
            [entry], destinations=[_ANNOUNCEMENT_DEST]))
        assert code == 1 and "turn_outcome_edit_malformed" in names, field


def test_the_edit_identity_fields_require_non_empty_strings(tmp_path):
    """§11.18 regression: `channel_id`, `target_ts`, `announcement_ts` and `error` must be
    NON-EMPTY strings — an empty or non-string value names no Slack coordinate and no
    failure code (the old grammar accepted error="")."""
    for field in ("channel_id", "target_ts", "announcement_ts"):
        code, names = _run_checker(tmp_path, _edit_session(
            [_edit_entry(**{field: ""})], destinations=[_ANNOUNCEMENT_DEST]))
        assert code == 1 and "turn_outcome_edit_malformed" in names, field
        code, names = _run_checker(tmp_path, _edit_session(
            [_edit_entry(**{field: 7})], destinations=[_ANNOUNCEMENT_DEST]))
        assert code == 1 and "turn_outcome_edit_malformed" in names, field
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(state="announcement_only", error="")],
        destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_malformed" in names
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(state="announcement_only", error=17)],
        destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_malformed" in names


def test_the_edit_lifecycle_is_sound(tmp_path):
    """§11.6: a record exists only once the disclosure was accepted, so BOTH states carry
    `announcement_ts`; `committed` carries no `error` and `announcement_only` always carries
    one. Each violation by its own name."""
    # A bare announcement_only — the shape the old grammar blessed — breaks BOTH rules.
    code, names = _run_checker(tmp_path, _edit_session(
        [{"channel_id": "C1", "target_ts": "2.0", "state": "announcement_only"}]))
    assert code == 1
    assert "turn_outcome_edit_announcement_missing" in names
    assert "turn_outcome_edit_error_missing" in names
    # A committed record without its announcement ts — announcement-first makes it a lie.
    entry = _edit_entry()
    entry.pop("announcement_ts")
    code, names = _run_checker(tmp_path, _edit_session([entry]))
    assert code == 1 and "turn_outcome_edit_announcement_missing" in names
    # committed ⇒ error absent.
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(error="update_failed_after_announcement")],
        destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_error_on_committed" in names
    # announcement_only ⇒ error present.
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry(state="announcement_only")], destinations=[_ANNOUNCEMENT_DEST]))
    assert code == 1 and "turn_outcome_edit_error_missing" in names


def test_an_edits_announcement_ts_must_join_a_committed_disclosure_destination(tmp_path):
    """The §7 join invariant: the disclosure post is a real destination the room saw, so an
    entry's announcement_ts that joins no committed correction_announcement destination means
    one of the two records is wrong about what was delivered."""
    # Healthy — both states join the same way; the join does not depend on the edit's state.
    assert _run_checker(tmp_path, _edit_session(
        [_edit_entry()], destinations=[_ANNOUNCEMENT_DEST])) == (0, [])
    assert _run_checker(tmp_path, _edit_session(
        [_edit_entry(state="announcement_only",
                     error="update_failed_after_announcement")],
        destinations=[_ANNOUNCEMENT_DEST])) == (0, [])

    # No destination at all.
    code, names = _run_checker(tmp_path, _edit_session([_edit_entry()]))
    assert code == 1 and "turn_outcome_edit_announcement_unjoined" in names
    # A destination of the right kind that never committed.
    observed = dict(_ANNOUNCEMENT_DEST, state="observed", chars=None)
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry()], destinations=[observed]))
    assert code == 1 and "turn_outcome_edit_announcement_unjoined" in names
    # A committed destination of the wrong kind.
    reply = dict(_ANNOUNCEMENT_DEST, kind="reply")
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry()], destinations=[reply]))
    assert code == 1 and "turn_outcome_edit_announcement_unjoined" in names
    # A committed disclosure at a DIFFERENT ts.
    elsewhere = dict(_ANNOUNCEMENT_DEST, first_ts="9.9")
    code, names = _run_checker(tmp_path, _edit_session(
        [_edit_entry()], destinations=[elsewhere]))
    assert code == 1 and "turn_outcome_edit_announcement_unjoined" in names
    # ...and the correction_announcement destination kind is legal on its own.
    assert _run_checker(tmp_path, _edit_session([], destinations=[_ANNOUNCEMENT_DEST])) \
        == (0, [])


def test_reconsider_and_edits_coexist_on_one_checked_turn_outcome(tmp_path):
    """The checker accepts the coexistence the emitter produces: one row carrying both the
    nested reconsider facts and a joined edit entry."""
    rows = _reconsider_session(_suppression(), _start(1),
                               _outcome(outcome="posted_asis", passes=1, forced=False))
    rows[-2]["reconsider"] = {"outcome": "posted_asis", "passes": 1, "forced": False}
    rows[-2]["kind"] = "reply"
    rows[-2]["destinations"] = [_ANNOUNCEMENT_DEST]
    rows[-2]["edits"] = [_edit_entry()]
    assert _run_checker(tmp_path, rows) == (0, [])
