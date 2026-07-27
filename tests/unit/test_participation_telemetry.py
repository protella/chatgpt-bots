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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from message_processor import participation_telemetry as pt
from message_processor.participation import ParticipationEngine


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


def test_a_model_authored_reason_is_bounded(sink):
    """Reasons are the classifier's prose about a human's message and may echo it. Same bound
    as the stored preference sentence — one shared privacy policy, not two."""
    long_reason = "x" * (pt.GUIDANCE_TRUNCATION_CHARS + 50)
    pt.gate_decision("C1", "1.0", MagicMock(action="ignore", reason=long_reason))
    written = sink("gate_decision")[0]["reason"]
    assert len(written) == pt.GUIDANCE_TRUNCATION_CHARS + 1   # + the ellipsis
    assert written.endswith("…")


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
    for `destination` + `destination_source`. Each is a change an analysis written against the
    older contract must be able to refuse."""
    assert pt.CONTRACT_VERSION == 5


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
    """An openai_client stand-in. `verdict` may be a dict, None (the failure signal), or an
    exception instance to raise."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    async def classify_participation(self, text, signals=None, **kwargs):
        self.calls += 1
        if isinstance(self._verdict, Exception):
            raise self._verdict
        return self._verdict


def _staged(action, **extra):
    v = {"action": action, "relation": "to_assistant", "exchange_state": "open",
         "answerability": "substantive"}
    v.update(extra)
    return v


def _app(verdict, react_result=None):
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.participation_engine = ParticipationEngine(_FakeClassifier(verdict))
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
    app, client = _app(_staged("respond"))
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
    app, client = _app(_staged("respond"))
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
async def test_a_failure_after_the_verdict_is_not_a_classifier_decline(sink, instant_gate):
    """The model decided; carrying the decision out is what broke. Filing that as a decline
    reports the gate as unable to judge while its verdict sits two lines above in the same
    file — and inflates exactly the number ("how often does the classifier fail?") that would
    send someone looking at the wrong system."""
    app, client = _app(_staged("react", emoji="tada"))

    async def _boom(*a, **k):
        raise RuntimeError("slack down")

    app._place_gate_reaction = _boom
    assert await app._gate_verdict(_msg(), client) is None

    assert sink("gate_decision")[0]["action"] == "react"      # the verdict IS on record
    decline = sink("gate_declined")[0]
    assert (decline["cause"], decline["detail"]) == ("action_error", "RuntimeError")
    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["cause"] == "action_error"


@pytest.mark.asyncio
async def test_a_failure_before_any_verdict_is_still_a_plain_error(sink, instant_gate):
    """The other side of the same split: nothing was decided, so this one really is a decline."""
    app, client = _app(_staged("respond"))
    app.participation_engine.note_arrival = MagicMock(side_effect=RuntimeError("boom"))
    assert await app._gate_verdict(_msg(), client) is None

    assert sink("gate_decision") == []
    assert sink("gate_declined")[0]["cause"] == "error"
    assert _terminals(sink)[0]["cause"] == "error"


@pytest.mark.asyncio
async def test_failure_and_a_real_ignore_are_the_same_silence_and_different_rows(
        sink, instant_gate):
    """Behaviour equivalence, asserted rather than assumed: returning None from the classifier
    changed what we WRITE DOWN, and nothing about what the room sees."""
    app_failed, client = _app(None)
    app_quiet, client2 = _app(_staged("ignore"))

    assert await app_failed._gate_verdict(_msg(), client) is None
    assert await app_quiet._gate_verdict(_msg(), client2) is None    # identical outcome

    kinds = [(r["kind"], r.get("cause")) for r in _terminals(sink)]
    assert kinds == [("none", "classifier_error"), ("silence", None)]


@pytest.mark.asyncio
async def test_a_quiet_verdict_is_a_decision_and_says_so(sink, instant_gate):
    app, client = _app(_staged("ignore"))
    assert await app._gate_verdict(_msg(), client) is None

    decision = sink("gate_decision")[0]
    assert decision["action"] == "ignore"
    assert decision["model"] == config.utility_model    # verdict quality is model-specific
    assert "gate_ms" in decision and "classifier_ms" in decision
    terminal = _terminals(sink)[0]
    assert terminal["kind"] == "silence" and terminal["ended_by"] == "gate"
    # A gate-only outcome never woke the responder. Conflating "the gate acted" with "the gate
    # woke the bot" makes the wake rate unreadable.
    assert terminal["gate_woke"] is False and terminal["responder_started"] is False


@pytest.mark.asyncio
async def test_a_react_verdict_that_lands_is_a_reaction_only_turn(sink, instant_gate):
    app, client = _app(_staged("react", emoji="tada"))
    assert await app._gate_verdict(_msg(), client) is None

    reaction = sink("reaction")[0]
    assert (reaction["operation"], reaction["result"]) == ("add", "added")
    assert reaction["origin"] == "gate" and reaction["emoji"] == "tada"
    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["kind"] == "reaction_only"


@pytest.mark.asyncio
async def test_a_react_verdict_whose_emoji_never_landed_showed_nothing(sink, instant_gate):
    """The emoji IS the turn. Filing a failed add as reaction_only would report a reaction that
    is not on the message."""
    app, client = _app(_staged("react", emoji="tada"),
                       react_result={"ok": False, "error": "reaction_cap"})
    assert await app._gate_verdict(_msg(), client) is None

    reaction = sink("reaction")[0]
    assert reaction["result"] == "failed" and reaction["detail"] == "reaction_cap"
    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["kind"] == "none"


@pytest.mark.asyncio
async def test_an_emoji_somebody_else_placed_is_not_one_we_added(sink, instant_gate):
    """`idempotent` from the reservation layer means the emoji is present, not that we put it
    there. Counting those as placements would credit the gate with reactions it never made."""
    app, client = _app(_staged("react", emoji="tada"),
                       react_result={"ok": True, "idempotent": True})
    await app._gate_verdict(_msg(), client)
    assert sink("reaction")[0]["result"] == "already_present"


@pytest.mark.asyncio
async def test_a_gate_reaction_refused_before_slack_is_still_recorded(sink, instant_gate):
    """A redispatch re-runs the same message and the once-per-message stamp refuses the second
    reaction. Invisible before this — which made "the gate stopped reacting" impossible to tell
    apart from "the gate stopped choosing emoji"."""
    app, client = _app(_staged("react", emoji="tada"))
    message = _msg(participation_reaction_emoji="eyes")   # a previous pass already reacted

    await app._gate_verdict(message, client)

    reaction = sink("reaction")[0]
    assert reaction["result"] == "already_present"
    assert reaction["detail"] == "already_stamped"
    client._reserve_and_react.assert_not_awaited()
    # ...and the TURN is still a reaction turn. The stamp is only ever written on a genuine
    # placement, so the emoji is on that message right now; filing this as `none` described an
    # untouched message that visibly has a reaction on it.
    terminals = _terminals(sink)
    assert len(terminals) == 1 and terminals[0]["kind"] == "reaction_only"


@pytest.mark.asyncio
async def test_a_backoff_whose_ack_was_already_stamped_is_not_a_silence(sink, instant_gate):
    """Same fact, the other branch: a redispatched backoff finds its ack already on the message.
    The feedback WAS visibly acknowledged, so the turn is not silent."""
    app, client = _app({"action": "backoff", "emoji": "ok_hand", "dimension": "replies",
                        "durability": "momentary", "scope": "thread", "memory_op": "none",
                        "structural_request": "none"})
    message = _msg(participation_reaction_emoji="ok_hand")
    assert await app._gate_verdict(message, client) is None

    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert terminals[0]["kind"] == "reaction_only" and terminals[0]["detail"] == "backoff"


@pytest.mark.asyncio
async def test_a_handled_backoff_with_an_ack_is_not_a_silent_turn(sink, instant_gate):
    app, client = _app({"action": "backoff", "emoji": "ok_hand", "dimension": "replies",
                        "durability": "momentary", "scope": "thread", "memory_op": "none",
                        "structural_request": "none"})
    assert await app._gate_verdict(_msg(), client) is None

    reaction = sink("reaction")[0]
    # Its own origin: a protocol acknowledgment is not a social reaction the classifier chose
    # for its own sake, and pooling them would flatter every emoji-diversity number.
    assert reaction["origin"] == "backoff_ack" and reaction["result"] == "added"
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert terminals[0]["kind"] == "reaction_only" and terminals[0]["detail"] == "backoff"
    # The taxonomy rides the DECISION. Without it a handled backoff is a bare reaction_only with
    # no way to ask what the feedback was about or whether it was meant to stick — the whole
    # question the taxonomy exists to answer.
    decision = sink("gate_decision")[0]
    assert (decision["dimension"], decision["durability"]) == ("replies", "momentary")
    assert (decision["scope"], decision["memory_op"]) == ("thread", "none")
    assert decision["structural_request"] == "none"
    # ...and never the free-text guidance, which is prose about a human's message.
    assert "guidance" not in decision


@pytest.mark.asyncio
async def test_a_handled_backoff_without_an_ack_is_a_silence(sink, instant_gate):
    """Feedback ABOUT reactions never gets acked with a reaction — acking "stop reacting" with
    a reaction is absurd — so that turn really did show the room nothing."""
    app, client = _app({"action": "backoff", "emoji": "ok_hand", "dimension": "reactions",
                        "durability": "momentary", "scope": "thread", "memory_op": "none",
                        "structural_request": "none"})
    assert await app._gate_verdict(_msg(), client) is None

    assert sink("reaction") == []
    terminals = _terminals(sink)
    assert len(terminals) == 1
    assert terminals[0]["kind"] == "silence" and terminals[0]["detail"] == "backoff"


@pytest.mark.asyncio
async def test_a_speaking_verdict_leaves_the_attempt_open_for_the_responder(sink, instant_gate):
    """The gate hands the turn on, so it must NOT close it — the responder owns what the room
    finally saw, and a terminal here would be the double-count all over again.

    Through `_run_participation_gate`, not `_gate_verdict`: the wake is recorded once at the
    single point where a verdict leaves the gate, rather than once per fall-through branch."""
    app, client = _app(_staged("respond"))
    message = _msg()
    verdict = await app._run_participation_gate(message, client)

    assert verdict is not None and verdict.action == "respond"
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
    app, client = _app(_staged("respond"))
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
    app, client = _app(_staged("ignore"))
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
    engine.openai_client.classify_participation = AsyncMock(return_value={"action": "ignore"})
    with patch.object(config, "participation_debounce_seconds", 0.0):
        engine.note_arrival("C1", "20.0", None, "U1")  # a newer message already arrived
        evaluation = await engine.evaluate(channel_id="C1", ts="10.0", text="hi",
                                           sender_id="U1", attempt_id="A1")

    assert evaluation.verdict is None
    assert evaluation.decline_cause == "superseded"
    rows = sink("gate_declined")
    assert len(rows) == 1
    assert rows[0]["cause"] == "superseded"
    assert (rows[0]["trigger_ts"], rows[0]["attempt_id"]) == ("10.0", "A1")


@pytest.mark.asyncio
async def test_engine_reports_the_failure_and_still_fails_safe(sink):
    """An API error still becomes an `ignore` verdict downstream — byte-identical silence — but
    the engine now says so in the return value, so the caller can close the attempt honestly
    instead of reading the cause off a log line."""
    engine = ParticipationEngine(MagicMock())
    engine.openai_client.classify_participation = AsyncMock(side_effect=TimeoutError("upstream"))
    with patch.object(config, "participation_debounce_seconds", 0.0):
        evaluation = await engine.evaluate(channel_id="C1", ts="10.0", text="hi", sender_id="U1")

    assert evaluation.verdict.action == "ignore"       # behaviour unchanged
    assert evaluation.decline_cause == "classifier_error"
    assert isinstance(evaluation.classifier_ms, int)
    assert sink("gate_declined")[0]["detail"] == "TimeoutError"


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


async def _run_responder(app, message, *, gate_emoji=None):
    """Drive handle_message past a gate that woke the responder. `gate_emoji` stands in for a
    react_and_respond that already put an emoji on the message."""
    from main import ChatBotV2

    async def _gate(msg, client):
        pt.begin_attempt(msg)
        pt.mark_gate_woke(msg)
        if gate_emoji:
            msg.metadata["participation_reaction_emoji"] = gate_emoji
        return MagicMock(placement="thread", reason=None, burst_earlier=None)

    app._run_participation_gate = _gate
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.send_message = AsyncMock(return_value="11.0")
    client.delete_message = AsyncMock()
    client.handle_error = AsyncMock()
    await ChatBotV2.handle_message(app, message, client)
    return client


@pytest.mark.asyncio
async def test_a_gate_reaction_plus_a_responder_veto_is_not_a_silence(sink):
    """react_and_respond puts the emoji up BEFORE the responder runs. If the responder then
    vetoes with no_response_needed, the terminal used to say `silence` about a message that
    visibly has a reaction on it — and the reason it gave was the reason for using no WORDS."""
    app = _responder_app(_resp(terminal_action="no_reply", silence_reason="nothing_to_add"))
    await _run_responder(app, _msg(), gate_emoji="tada")

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
    """`kind` names the words because they are the louder half — so the emoji half of a
    react_and_respond has to ride beside it or it leaves the terminal record entirely."""
    app = _responder_app(_resp(content="here you go", posted=True, streamed=True, model="m-1"))
    await _run_responder(app, _msg(), gate_emoji="tada")

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
