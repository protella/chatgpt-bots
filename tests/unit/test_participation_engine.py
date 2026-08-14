"""ParticipationEngine — the binary wake gate: level/mode mapping, debounce supersession,
conversation keying, COHORT coalescing, the main.py wiring around one bit, modal dual-write,
DB columns/migration, and the busy-rejection needs_refresh fix.

All stubbed I/O — no live bot, no legacy suite.

WHAT LEFT THIS FILE IN COMMIT 6, and why deleting rather than re-pointing was right:

* the verdict-validation suite (`validate_verdict`, `_apply_invariants`, emoji coercion,
  placement coercion, reason truncation, the staged-findings fail-closed rule). There is no
  verdict. `WakeDecision` has one field, a bool, produced by a strict json_schema — so the entire
  class of "the model sent something shaped wrong and we must repair it safely" cannot occur, and
  a test asserting how it is repaired would be testing dead code.
* the backoff/preference-write tests. A classifier that had seen one message wrote to the
  database; that path is gone (its taxonomy tests went with the file
  test_participation_backoff_taxonomy.py). Participation feedback now WAKES the responder, whose
  memory and settings tools own the write.
* the gate-reaction tests. The gate places nothing in the room, so "does a respond verdict also
  react" has no subject. The inverted guard survives below as one test, because a reaction
  appearing before the responder ran is the specific regression worth catching.
* the burst-carry tests as written. Carrying survives and is stronger — typed source records
  merged into the responder's real input instead of quoted prose — but `_MAX_BURST_CARRY`, the
  freshness window and the pending-map eviction are deleted on purpose: each could silently drop a
  message somebody sent, in the one structure whose whole job is not to lose it. The replacements
  assert that nothing is capped or dropped.
"""
from __future__ import annotations

import asyncio
import inspect
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_processor.client_contract import Message
from config import config
from database import DatabaseManager
from message_processor.participation import (
    LEVEL_TO_MODE, MODE_TO_LEVEL, VALID_LEVELS, GateEvaluation, ParticipationEngine,
    SourceMessage, WakeDecision, resolve_participation_level,
)


# ------------------------------------------------------------------ level resolution

class TestLevelResolution:
    def test_mode_mapping_round_trip(self):
        assert MODE_TO_LEVEL == {"off": "off", "tag_only": "mentions_only", "auto_respond": "on"}
        # Under a binary gate the mapping is a genuine bijection: every level has exactly one
        # legacy mode and vice versa. It was NOT one before — `judicious` and `active` shared
        # `auto_respond`, so a round trip through the legacy column silently rewrote `active`.
        for level, mode in LEVEL_TO_MODE.items():
            assert MODE_TO_LEVEL[mode] == level
        assert set(LEVEL_TO_MODE) == set(VALID_LEVELS)

    def test_level_truth_table(self, monkeypatch):
        """The whole resolution contract in one place: legal levels pass through untouched, each
        legacy response_mode maps to exactly one level, and the retired names are not levels."""
        # A global default that is NOT the answer to any of these cases, so a silent fallthrough
        # to it would show up as a wrong result instead of an accidentally-right one.
        monkeypatch.setattr(config, "channel_response_mode", "off", raising=False)
        for level in VALID_LEVELS:
            assert resolve_participation_level({"participation_level": level}) == level
        for mode, level in MODE_TO_LEVEL.items():
            assert resolve_participation_level({"response_mode": mode}) == level
        # `judicious` and `active` are retired, and a row still carrying one resolves to
        # mentions_only — NOT to whatever its legacy response_mode says.
        #
        # Absent and present-but-unrecognised are different questions. An absent level means the
        # channel never chose one, so the legacy mode is the honest answer. A level we cannot read
        # means the channel DID choose something, and falling through to `auto_respond` would
        # resolve it to `on` — turning an unreadable setting into the most talkative one available.
        # These rows should not exist (the startup migration rewrites them, and a process that
        # fails that migration refuses to start); if one does, quiet is the safe direction.
        assert "judicious" not in VALID_LEVELS and "active" not in VALID_LEVELS
        for retired in ("judicious", "active"):
            assert resolve_participation_level({"participation_level": retired}) == "mentions_only"
            assert resolve_participation_level(
                {"participation_level": retired,
                 "response_mode": "auto_respond"}) == "mentions_only"

    def test_participation_level_wins_over_mode(self):
        cs = {"participation_level": "on", "response_mode": "off"}
        assert resolve_participation_level(cs) == "on"

    def test_falls_back_to_row_mode(self):
        assert resolve_participation_level({"response_mode": "auto_respond"}) == "on"
        assert resolve_participation_level({"response_mode": "off"}) == "off"

    def test_falls_back_to_global_default(self, monkeypatch):
        monkeypatch.setattr(config, "channel_response_mode", "tag_only", raising=False)
        assert resolve_participation_level(None) == "mentions_only"
        monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
        assert resolve_participation_level({}) == "on"

    def test_garbage_degrades_safe(self, monkeypatch):
        monkeypatch.setattr(config, "channel_response_mode", "banana", raising=False)
        assert resolve_participation_level({"participation_level": "loud"}) == "mentions_only"


# ------------------------------------------------------------------- the shape of the output

class TestWakeDecisionShape:
    def test_the_decision_is_one_bit_and_nothing_else(self):
        """Tripwire, not a tautology. Every field the old verdict carried became a control bus —
        `action` branched the caller four ways, `emoji` placed a reaction, `reason` was forwarded
        into the responder's prompt and pre-argued the turn. A new field here is how that starts
        again, so the field set is pinned."""
        assert [f for f in WakeDecision.__dataclass_fields__] == ["wake"]
        assert WakeDecision(wake=True).wake is True
        # Frozen: nothing downstream may edit a decision on its way through.
        with pytest.raises(Exception):
            WakeDecision(wake=True).wake = False

    def test_the_evaluation_reports_which_kind_of_nothing(self):
        """`evaluate()` returning verdict-or-None left the caller unable to tell a cohort collapse
        from an edit cancellation from a provider outage — it could only find out by reading the
        engine's log lines, and the caller owns the turn's single terminal event.

        `source_files` is pinned here too, and it is not a reopening of the rich verdict: it carries
        no judgment and nothing branches on it. It is the cohort members' live file payloads, which
        exist nowhere else once their own dispatches end at the gate — the survivor's turn needs
        them to authorize a file id Slack has not propagated yet [r6-3]."""
        assert set(GateEvaluation.__dataclass_fields__) == {
            "decision", "decline_cause", "classifier_ms", "sources", "source_files"}
        empty = GateEvaluation()
        assert empty.decision is None and empty.decline_cause is None and empty.sources == ()
        assert empty.source_files == ()

    def test_a_source_knows_its_own_topology(self):
        # A thread ROOT is not a thread reply: its root ts is its own ts. This is what decides
        # whether the prompt says "a reply inside a thread" or "posted to the channel".
        assert SourceMessage(ts="10.0", thread_root_ts="10.0").is_thread_reply is False
        assert SourceMessage(ts="10.5", thread_root_ts="10.0").is_thread_reply is True
        assert SourceMessage(ts="10.0").is_thread_reply is False


# ------------------------------------------------------------------- debounce + cohorts

class _FakeClient:
    """The classifier half of the world: records every cohort it is asked to judge.

    `calls` counts calls STARTED and `completed` counts calls that ran to the end — two different
    numbers since W5b, because a speculative call can be cancelled in flight. `delay` is what makes
    that observable: a classifier that returns without ever awaiting cannot be interrupted, so a
    zero-delay fake would report every cancelled call as completed and quietly pass a suite that
    was measuring the wrong thing.

    Keep `delay` SHORTER than the debounce in tests about the one-live-speculation cap. Longer, and
    the sleeper's own wake-up cancels its call first — the ordinary superseded-abandon path, which
    predates the cap — so the count comes out right for a reason that has nothing to do with what
    the test claims to measure.
    """

    def __init__(self, wake=True, delay=0.0):
        self._wake = wake
        self._delay = delay
        self.calls = 0
        self.completed = 0
        self.cohorts = []          # one entry per call: the tuple of sources it saw
        self.steering = []

    async def classify_wake(self, *, sources, channel_steering_text=None):
        self.calls += 1
        self.cohorts.append(tuple(sources))
        self.steering.append(channel_steering_text)
        if self._delay:
            await asyncio.sleep(self._delay)
        self.completed += 1
        return self._wake

    @property
    def last_cohort(self):
        return self.cohorts[-1] if self.cohorts else ()


def _texts(cohort):
    return [s.text for s in cohort]


def _warm(engine, *, channel="C1", ts="0.5", thread_root=None, sender_id=None):
    """Make a stream WARM, so the message under test still gets a debounce window to sit in.

    W5c: a conversation with no arrival inside the last debounce window is judged immediately —
    there is no burst to collect. Every test about what the window COLLECTS therefore starts from
    an active stream, which is exactly one prior arrival."""
    engine.note_arrival(channel, ts, thread_root, sender_id)


class TestDebounceAndSupersession:
    @pytest.mark.asyncio
    async def test_rapid_fire_collapses_to_latest(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True, delay=0.02)
        engine = ParticipationEngine(fake)
        _warm(engine)
        first = asyncio.create_task(engine.evaluate(channel_id="C1", ts="1.0", text="line one"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(channel_id="C1", ts="2.0", text="line two"))
        r1, r2 = await asyncio.gather(first, second)
        assert r1.decision is None and r1.decline_cause == "superseded"
        assert r2.decision.wake is True
        # ONE model call COMPLETED for the burst, and one decision. The sleeper started a
        # speculative call inside its own window; the survivor's enrollment cancelled it while it
        # was still in flight, so it cost a request that was aborted rather than a whole judgment.
        assert fake.completed == 1
        assert fake.calls == 2               # started, and one of the two was cancelled early
        # ...and the superseded message is not lost: it is in the survivor's cohort.
        assert _texts(fake.last_cohort) == ["line one", "line two"]

    @pytest.mark.asyncio
    async def test_thread_message_survives_newer_message_in_other_thread(self, monkeypatch):
        """F21: supersession is conversation-scoped. A pending evaluation in thread A
        must NOT be dropped because thread B (or another conversation) posted something
        newer in the same channel during the debounce window."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True)
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="question in thread A", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="20.5", text="chatter in thread B", thread_root_ts="20.0"))
        ra, rb = await asyncio.gather(a, b)
        assert ra.decision.wake is True    # thread A still judged
        assert rb.decision.wake is True
        assert fake.calls == 2             # both conversations evaluated
        # Independent streams, so neither cohort may contain the other's message.
        assert _texts(fake.cohorts[0]) == ["question in thread A"]
        assert _texts(fake.cohorts[1]) == ["chatter in thread B"]

    @pytest.mark.asyncio
    async def test_thread_message_survives_newer_top_level(self, monkeypatch):
        """F21: a newer TOP-LEVEL message must not supersede a pending thread reply."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True)
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="thread question", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="30.0", text="unrelated top-level"))  # roots key as |top
        ra, rb = await asyncio.gather(a, b)
        assert ra.decision is not None and rb.decision is not None
        assert fake.calls == 2

    def test_conv_key_root_vs_reply(self):
        """A thread ROOT keys as top-level (thread_root == ts); its replies key by root.
        F27: top-level keys are per-sender; a thread reply key ignores sender."""
        assert ParticipationEngine._conv_key("C1", "10.0", "10.0", "U1") == "C1|top|U1"
        assert ParticipationEngine._conv_key("C1", "10.5", "10.0", "U1") == "C1|10.0"
        assert ParticipationEngine._conv_key("C1", "30.0", None, "U2") == "C1|top|U2"
        # no sender_id → "unknown" (back-compat default)
        assert ParticipationEngine._conv_key("C1", "30.0", None) == "C1|top|unknown"

    @pytest.mark.asyncio
    async def test_channels_debounce_independently(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.02, raising=False)
        fake = _FakeClient(wake=False)
        engine = ParticipationEngine(fake)
        r1, r2 = await asyncio.gather(
            engine.evaluate(channel_id="C1", ts="1.0", text="a"),
            engine.evaluate(channel_id="C2", ts="1.0", text="b"),
        )
        assert r1.decision.wake is False and r2.decision.wake is False
        assert fake.calls == 2

    @pytest.mark.asyncio
    async def test_a_classifier_failure_is_a_decline_not_a_decision(self, monkeypatch):
        """RE-BASELINED, and this is the behaviour change worth reading twice.

        The rich gate turned an API failure into a manufactured `{"action": "ignore"}` — silence
        either way, but it scored a provider outage as the model choosing restraint, in the same
        ledger field used to judge the model's judgment. Now there is no bit: `decision` is None,
        `decline_cause` is "classifier_error", and the caller ends the turn as `none` rather than
        `silence`. The failure is still measured (classifier_ms is recorded even on the failing
        path — a timeout's duration is the story)."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)

        class _Boom:
            async def classify_wake(self, *, sources, channel_steering_text=None):
                raise RuntimeError("api down")

        ev = await ParticipationEngine(_Boom()).evaluate(channel_id="C1", ts="1.0", text="x")
        assert ev.decision is None
        assert ev.decline_cause == "classifier_error"
        assert ev.classifier_ms is not None
        # The cohort still comes back, so the caller can catalogue what it could not answer.
        assert _texts(ev.sources) == ["x"]

    @pytest.mark.asyncio
    async def test_a_none_from_the_classifier_is_also_a_decline(self, monkeypatch):
        # classify_wake returns None for a refusal, an empty output, or a non-boolean payload.
        # Same outcome as an exception: no bit, therefore no decision.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)

        class _NoBit:
            async def classify_wake(self, *, sources, channel_steering_text=None):
                return None

        ev = await ParticipationEngine(_NoBit()).evaluate(channel_id="C1", ts="1.0", text="x")
        assert ev.decision is None and ev.decline_cause == "classifier_error"

    @pytest.mark.asyncio
    async def test_a_captionless_upload_declines_structurally(self, monkeypatch):
        """A cohort of nothing but wordless files: no question, no addressee, nothing for a wake
        decision to be ABOUT. Named as a structural outcome rather than dressed up as the model
        choosing silence — no classifier runs at all. The sources still come back, because
        declining to answer a wordless upload is not the same as forgetting it happened (the caller
        catalogues them)."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        fake = _FakeClient(wake=True)
        ev = await ParticipationEngine(fake).evaluate(
            channel_id="C1", ts="1.0", text="   ", attachments=["food.png (image)"])
        assert ev.decision is None and ev.decline_cause == "image_only"
        assert fake.calls == 0
        assert ev.sources[0].attachments == ("food.png (image)",)

    @pytest.mark.asyncio
    async def test_a_caption_anywhere_in_the_cohort_reaches_the_classifier(self, monkeypatch):
        # The rule is about the COHORT, not the survivor: someone who posts a file and then says
        # what it is has asked a question, and judging only the newest fragment would miss it.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True, delay=0.02)
        engine = ParticipationEngine(fake)
        _warm(engine, sender_id="U1")
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="what do we think?", sender_id="U1"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="2.0", text="", sender_id="U1",
            attachments=["poster.png (image)"]))
        _, survivor = await asyncio.gather(first, second)
        assert survivor.decline_cause is None
        assert survivor.decision.wake is True
        assert fake.completed == 1   # the sleeper's speculation was cancelled, not completed
        assert _texts(fake.last_cohort) == ["what do we think?", ""]

    def test_the_engine_has_no_pacing_rails(self):
        # F17: the hourly-cap hard rail is gone entirely, and the binary gate adds no replacement.
        # Pacing is not a gate question — the responder decides whether to speak.
        engine = ParticipationEngine(MagicMock())
        assert not hasattr(engine, "hourly_cap")
        assert not hasattr(engine, "over_throttle")


# ------------------------------------------------------- cohorts: nothing capped, nothing dropped

class TestCohortDelivery:
    @pytest.mark.asyncio
    async def test_different_authors_top_level_both_survive(self, monkeypatch):
        """F27: two DIFFERENT users' unrelated top-level messages within the debounce no
        longer collapse — each is the newest in its own per-sender stream, both answered,
        and neither cohort contains the other's message."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True)
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="alice question", sender_id="U1"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="2.0", text="bob question", sender_id="U2"))
        ra, rb = await asyncio.gather(a, b)
        assert ra.decision.wake is True and rb.decision.wake is True
        assert fake.calls == 2
        assert _texts(fake.cohorts[0]) == ["alice question"]
        assert _texts(fake.cohorts[1]) == ["bob question"]

    @pytest.mark.asyncio
    async def test_a_multi_message_top_level_cohort_reaches_the_model_once_in_order(
            self, monkeypatch):
        """Five sends from one person inside one debounce window: ONE model call, all five
        messages, oldest first, no duplication.

        This is the replacement for the old carry test, which asserted the newest THREE were kept
        (`_MAX_BURST_CARRY`). A cap here is a message the person actually sent that neither model
        ever sees, so there is no cap now and this asserts the whole burst arrives."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True, delay=0.02)
        engine = ParticipationEngine(fake)
        _warm(engine, sender_id="U1")
        tasks = []
        for i in range(1, 6):
            tasks.append(asyncio.create_task(engine.evaluate(
                channel_id="C1", ts=f"{i}.0", text=f"m{i}", sender_id="U1")))
            await asyncio.sleep(0.002)
        results = await asyncio.gather(*tasks)
        # ONE completed model call for the burst, judged on the whole of it. Each of the four
        # sleepers did START a speculative call, and each was cancelled by the next message's
        # enrollment within a couple of milliseconds — at most one speculation is ever live in a
        # conversation, so the burst costs one judgment plus four aborted requests.
        assert fake.completed == 1
        assert fake.calls == 5
        assert _texts(fake.last_cohort) == ["m1", "m2", "m3", "m4", "m5"]
        # exactly one survivor, and its GateEvaluation carries the same bundle for the responder
        survivors = [r for r in results if r.decision is not None]
        assert len(survivors) == 1
        assert _texts(survivors[0].sources) == ["m1", "m2", "m3", "m4", "m5"]
        assert all(r.decline_cause == "superseded" for r in results if r.decision is None)
        # Drained: the survivor took the bucket with it, so a later message starts a fresh cohort.
        assert "C1|top|U1" not in engine._cohorts

    @pytest.mark.asyncio
    async def test_a_cross_author_thread_cohort_reaches_the_model_once_in_order(self, monkeypatch):
        """RE-BASELINED. A thread collapses cross-author (F21 — the reply lands in-thread with full
        history), and the old code then THREW THE SUPERSEDED TEXT AWAY: carry was top-level-only
        because the prose render labelled the carried lines "the same sender", so keeping them would
        have misattributed. Typed records carry the sender, so the reason to drop them is gone —
        two people talking at once in a thread is exactly the case where the second message alone
        reads as a non sequitur."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True, delay=0.02)
        engine = ParticipationEngine(fake)
        _warm(engine, ts="10.4", thread_root="10.0", sender_id="U1")
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="alice's words", sender_id="U1",
            sender_name="Alice", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.6", text="bob's reply", sender_id="U2",
            sender_name="Bob", thread_root_ts="10.0"))
        r1, r2 = await asyncio.gather(first, second)
        assert r1.decision is None and r1.decline_cause == "superseded"
        assert r2.decision.wake is True
        assert fake.completed == 1   # Alice's speculation was cancelled the moment Bob enrolled
        assert _texts(fake.last_cohort) == ["alice's words", "bob's reply"]
        # Attribution is preserved per record, which is what makes the carry safe.
        assert [s.sender_name for s in fake.last_cohort] == ["Alice", "Bob"]
        assert "C1|10.0" not in engine._cohorts       # bucket drained (memory hygiene)

    def test_an_old_enrollment_is_still_carried(self):
        """RE-BASELINED, deliberately inverting the old assertion.

        The old `_collect_burst` dropped entries more than ~15s older than the survivor, calling
        them stale leftovers. That is a silent message-loss path: the only way a stale enrollment
        exists is that its own evaluation never ran, which is a bug to SEE, not to paper over — and
        the same window would drop a genuine slow burst. Age is no longer a reason to discard."""
        eng = ParticipationEngine(MagicMock())
        key = "C1|top|U1"
        eng._enroll_source(key, SourceMessage(ts="100.0", text="ancient"))
        eng._enroll_source(key, SourceMessage(ts="1000.0", text="recent"))
        eng._enroll_source(key, SourceMessage(ts="1001.0", text="survivor"))
        carried = eng._drain_cohort(key, "1001.0")
        assert _texts(carried) == ["ancient", "recent", "survivor"]
        assert key not in eng._cohorts          # bucket removed once drained

    def test_a_fileless_replacement_leaves_no_empty_payload_bucket(self):
        """[r7-4] Re-enrolling the same ts without files (an edit that removed the attachment)
        must remove the whole bucket when its last payload goes — a leftover `{key: {}}` would
        outlive the drained cohort forever, because the empty mapping short-circuits the taker."""
        eng = ParticipationEngine(MagicMock())
        key = "C1|top|U1"
        payload = {"type": "file", "id": "F9", "name": "data.csv"}
        eng._enroll_source(key, SourceMessage(ts="1.0", text="with file"),
                           file_payloads=(payload,))
        eng._enroll_source(key, SourceMessage(ts="1.0", text="file removed"))
        assert key not in eng._cohort_files
        carried = eng._drain_cohort(key, "1.0")
        assert eng._take_cohort_files(key, carried) == ()
        assert eng._cohorts == {} and eng._cohort_files == {}

    def test_a_newer_enrollment_is_left_for_its_own_survivor(self):
        # Strictly-newer entries belong to a later survivor: taking one would judge a message whose
        # own debounce has not finished.
        eng = ParticipationEngine(MagicMock())
        key = "C1|top|U1"
        for ts, text in (("1.0", "mine"), ("2.0", "not yet mine")):
            eng._enroll_source(key, SourceMessage(ts=ts, text=text))
        assert _texts(eng._drain_cohort(key, "1.0")) == ["mine"]
        assert list(eng._cohorts[key]) == ["2.0"]     # bucket kept, holding the newer one
        assert _texts(eng._drain_cohort(key, "2.0")) == ["not yet mine"]
        assert key not in eng._cohorts

    def test_re_enrolling_one_ts_replaces_rather_than_duplicates(self):
        # An edit keeps its original timestamp, so it re-enrolls the same ts. The cohort must hold
        # the CURRENT text, not two versions of one message.
        eng = ParticipationEngine(MagicMock())
        eng._enroll_source("K", SourceMessage(ts="1.0", text="draft"))
        eng._enroll_source("K", SourceMessage(ts="1.0", text="draft final"))
        assert _texts(eng._drain_cohort("K", "1.0")) == ["draft final"]

    def test_the_cohort_map_is_deliberately_unbounded(self):
        """A correctness requirement, not an oversight, so it is asserted rather than commented.

        The pending map used to evict the oldest bucket past `_MAX_PENDING_KEYS`. Eviction there
        discards an ENROLLED message — one somebody sent, waiting for its turn — and the busiest
        workspace is exactly where it would happen. Cancellation is explicit instead
        (`discard_source`), and buckets go away when their survivor drains them."""
        import message_processor.participation as participation
        assert not hasattr(participation, "_MAX_PENDING_KEYS")
        assert not hasattr(participation, "_MAX_BURST_CARRY")
        eng = ParticipationEngine(MagicMock())
        for i in range(2000):
            eng._enroll_source(f"C1|top|U{i}", SourceMessage(ts="1.0", text="x"))
        assert len(eng._cohorts) == 2000
        # The one bound that IS kept: supersession MARKS, which are bookkeeping about messages
        # already handled elsewhere, so dropping one can never lose something awaiting a turn.
        assert participation._MAX_SUPERSESSION_KEYS == 512

    def test_discard_source_forgets_an_abandoned_source(self):
        eng = ParticipationEngine(MagicMock())
        eng._enroll_source("C1|10.0", SourceMessage(ts="10.5", text="orphan"))
        eng.discard_source("C1", "10.5", thread_root="10.0")
        assert "C1|10.0" not in eng._cohorts     # it was the last one out, so the bucket goes too
        # Idempotent, and a missing channel/ts is a no-op rather than a crash.
        eng.discard_source("C1", "10.5", thread_root="10.0")
        eng.discard_source("", None)

    @pytest.mark.asyncio
    async def test_a_superseded_attempt_leaves_its_record_enrolled(self, monkeypatch):
        """The invariant the whole design rests on: being superseded must not delete you.

        Checked mid-flight rather than only through the survivor's cohort, because "the survivor
        happened to see it" and "the record was still there to be seen" are different facts, and
        only the second one holds when the survivor arrives late."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient(wake=True)
        engine = ParticipationEngine(fake)
        _warm(engine, sender_id="U1")
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="the real question", sender_id="U1"))
        await asyncio.sleep(0.01)
        # Advance the marker WITHOUT starting a second evaluation, so the first attempt is
        # superseded by a survivor that has not run yet.
        engine.note_arrival("C1", "2.0", None, "U1")
        ev = await first
        assert ev.decline_cause == "superseded"
        # W5b: a speculative call DID run inside the window and was discarded unused — nothing
        # enrolled to cancel it, since the survivor has not started yet, and only an enrollment
        # cancels. What the invariant is about is untouched: no decision came back, and the record
        # is still there for that survivor to collect.
        assert ev.decision is None and fake.calls == 1 and fake.completed == 1
        assert _texts(engine._drain_cohort("C1|top|U1", "2.0")) == ["the real question"]

    @pytest.mark.asyncio
    async def test_the_steering_bytes_are_passed_through_untouched(self, monkeypatch):
        # Commit 5's invariant: the gate inserts the caller's exact string. It does not render,
        # re-read, or reorder it — the responder's copy of this turn must be byte-identical.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        fake = _FakeClient(wake=True)
        steering = "Standing channel policy (instructions; follow these):\nonly deploys"
        await ParticipationEngine(fake).evaluate(
            channel_id="C1", ts="1.0", text="hi", channel_steering_text=steering)
        assert fake.steering == [steering]


# --------------------------------------------------------------- main.py gate wiring

def _make_app(wake=True, engine=True):
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    fake = _FakeClient(wake)
    app.participation_engine = ParticipationEngine(fake) if engine else None
    app.processor = MagicMock()
    app.processor.db = MagicMock()
    app.processor.db.get_channel_policy_async = AsyncMock(return_value=None)
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    app.processor.db.set_channel_settings_async = AsyncMock()
    app.processor.db.add_channel_memory_async = AsyncMock()
    app.processor._schedule_async_call = MagicMock()
    client = MagicMock()
    client.react = AsyncMock()
    client._reserve_and_react = AsyncMock(return_value={"ok": True})
    return app, client, fake


def _channel_msg(**meta):
    m = {"ts": "10.0", "gate_required": True, "silence_capable": True,
         "participation_level": "on"}
    m.update(meta)
    return Message(text="anyone know the deploy status?", user_id="U1",
                   channel_id="C1", thread_id="10.0", metadata=m)


class TestGateWiring:
    @pytest.fixture(autouse=True)
    def _no_debounce(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)

    @pytest.mark.asyncio
    async def test_engine_disabled_stays_silent(self, monkeypatch):
        # The inner `engine_off` terminal is kept as a race/test backstop even though
        # message_events also prefilters on the flag.
        monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
        app, client, fake = _make_app()
        assert await app._run_participation_gate(_channel_msg(), client) is None
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_no_engine_object_stays_silent(self):
        app, client, _ = _make_app(engine=False)
        assert await app._run_participation_gate(_channel_msg(), client) is None

    @pytest.mark.asyncio
    async def test_wake_true_hands_the_turn_on(self):
        app, client, fake = _make_app(wake=True)
        decision = await app._run_participation_gate(_channel_msg(), client)
        assert isinstance(decision, WakeDecision) and decision.wake is True
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_wake_false_ends_the_turn_at_the_gate(self):
        # RE-BASELINED: there is no `ignore` action to branch on. A genuine wake=false returns None
        # from the wiring, which is the caller's whole signal to stop.
        app, client, fake = _make_app(wake=False)
        assert await app._run_participation_gate(_channel_msg(), client) is None
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_name_hit_still_reaches_the_gate(self):
        # F17: a name-addressed message is judged like any other. Being named is not being
        # addressed, and the prefilter does not get to decide which it was.
        app, client, fake = _make_app(wake=True)
        decision = await app._run_participation_gate(
            _channel_msg(participation_name_hit=True), client)
        assert decision is not None and fake.calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wake", [True, False])
    async def test_the_gate_never_puts_a_reaction_in_the_room(self, wake, monkeypatch):
        """The inverted guard that replaces the whole gate-reaction suite (spec §3).

        The rich gate could place an emoji before the responder ran — so a responder that then
        chose silence had not actually left the room quiet, and every downstream label had to be
        told about the gate's emoji. On no outcome, with reactions fully enabled, may the gate
        touch the room."""
        monkeypatch.setattr(config, "enable_reactions", True, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "eyes"], raising=False)
        app, client, _ = _make_app(wake=wake)
        msg = _channel_msg()
        await app._run_participation_gate(msg, client)
        client._reserve_and_react.assert_not_awaited()
        client.react.assert_not_awaited()
        assert "participation_reaction_emoji" not in msg.metadata

    @pytest.mark.asyncio
    async def test_a_wake_stamps_the_cohort_for_the_responders_input(self, monkeypatch):
        """The replacement for `participation_burst_earlier`: typed records on the message, which
        the input builder merges into the real conversation — not prose metadata describing
        messages the model cannot see."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        app, client, fake = _make_app(wake=True)
        engine = app.participation_engine
        _warm(engine, sender_id="U1")
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="9.0", text="earlier thought", sender_id="U1"))
        await asyncio.sleep(0.005)
        msg = _channel_msg(ts="10.0")
        gate = asyncio.create_task(app._run_participation_gate(msg, client))
        await first
        decision = await gate
        assert decision is not None and decision.wake is True
        assert _texts(msg.metadata["gate_sources"]) == [
            "earlier thought", "anyone know the deploy status?"]
        # and none of the retired prose keys are stamped anywhere
        for retired in ("participation_reason", "participation_burst_earlier",
                        "participation_reaction_emoji", "participation_images"):
            assert retired not in msg.metadata

    @pytest.mark.asyncio
    async def test_a_silent_gate_still_catalogues_the_files(self):
        """Deciding not to REPLY is not deciding to FORGET. Live on the first run of this feature:
        four files landed seconds apart, the CSV's message was superseded, and the CSV ceased to
        exist as far as the bot was concerned — then the model correctly refused to build the
        report it could not read."""
        app, client, _ = _make_app(wake=False)
        msg = _channel_msg()
        msg.attachments = [{"type": "file", "id": "F1", "name": "data.csv"}]
        assert await app._run_participation_gate(msg, client) is None
        app.processor._schedule_async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_woken_turn_leaves_the_files_to_the_turn(self):
        # On a wake the turn does the richer job (extraction, summaries, descriptions) and
        # save_document is a plain INSERT — cataloguing here too would just duplicate the row.
        app, client, _ = _make_app(wake=True)
        msg = _channel_msg()
        msg.attachments = [{"type": "file", "id": "F1", "name": "data.csv"}]
        assert await app._run_participation_gate(msg, client) is not None
        app.processor._schedule_async_call.assert_not_called()

    def test_the_gate_reaction_and_backoff_machinery_is_gone(self):
        """AST-level tripwire (spec §9). Each of these was a way for a classifier that had seen one
        message to act on the workspace: place an emoji, acknowledge a backoff, write a preference
        row. Leaving any of them executable changes behaviour, which is why they die in this commit
        rather than in the cleanup one."""
        from main import ChatBotV2
        for gone in ("_place_gate_reaction", "_apply_backoff", "_backoff_ack",
                     "_apply_pref_memory", "_own_pref_row", "_is_own_dimension_pref",
                     "_pref_memory_content", "_is_unprompted_turn"):
            assert not hasattr(ChatBotV2, gone), gone
        # And the visible-action classifier no longer takes a gate-reaction argument: it can see
        # all of its own inputs, because nothing put an emoji in the room before it ran.
        params = list(inspect.signature(ChatBotV2._classify_visible_action).parameters)
        assert params == ["response", "turn"]


class TestPlacement:
    def _app_with_processor(self, response):
        from main import ChatBotV2
        app = ChatBotV2.__new__(ChatBotV2)
        app.participation_engine = None
        app.processor = MagicMock()
        app.processor.process_message = AsyncMock(return_value=response)
        app.processor.thread_manager = MagicMock(spec=[])  # no upload latch attrs
        client = MagicMock()
        client.send_thinking_indicator = AsyncMock(return_value="think.1")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()
        client.format_text = lambda t: t
        client.maybe_post_response_footer = AsyncMock()
        return app, client

    def _resp(self, text="answer"):
        r = MagicMock()
        r.type = "text"
        r.content = text
        r.metadata = {}
        return r

    @pytest.mark.asyncio
    async def test_an_eligible_turn_shows_nothing_until_the_model_places_it(self):
        """The allowance alone no longer sends a reply top-level: both destinations are legal,
        so the MODEL chooses, and until it does the turn shows nothing anywhere — no
        placeholder, in either location. Without a choice the answer takes the default thread.

        (The "(edited)" rule is why a channel-destined turn still writes nothing until the end:
        Slack can only stream into a thread, so a top-level placeholder could only become the
        answer by being chat.update-d, which brands the message "(edited)" forever.)
        """
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "channel_post_allowed": True})
        await app.handle_message(msg, client)
        client.send_thinking_indicator.assert_not_awaited()
        assert client.send_message.await_args.args[1] == "10.0"   # default thread
        client.maybe_post_response_footer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_default_threads_and_footer_posts(self):
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
        await app.handle_message(msg, client)
        # The turn's receipt ledger rides the indicator (spec §5); assert the destination.
        client.send_thinking_indicator.assert_awaited_once()
        assert client.send_thinking_indicator.await_args.args == ("C1", "10.0")
        assert client.send_message.await_args.args[1] == "10.0"
        client.maybe_post_response_footer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_thread_reply_never_moves_top_level_despite_setting(self):
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "11.0", "reply_in_channel": True})  # reply inside thread
        await app.handle_message(msg, client)
        assert client.send_message.await_args.args[1] == "10.0"

    @pytest.mark.asyncio
    async def test_a_woken_turn_carries_no_gate_prose_into_the_responder(self):
        """RE-BASELINED from two tests that asserted the gate's `burst_earlier` and `reason`
        landed in metadata for the wake envelope to render.

        `reason` is deleted outright — it was forwarded into the responder's prompt, where it
        pre-argued the turn and undercut the responder's own option to stay silent — and the burst
        is delivered as real conversation instead (see the gate-wiring cohort test). So the
        assertion inverts: a woken turn's metadata gains none of it."""
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        app._run_participation_gate = AsyncMock(return_value=WakeDecision(wake=True))
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "gate_required": True, "silence_capable": True})
        await app.handle_message(msg, client)
        assert client.send_message.await_args.args[1] == "10.0"
        for retired in ("participation_reason", "participation_burst_earlier",
                        "participation_reaction_emoji"):
            assert retired not in msg.metadata

    @pytest.mark.asyncio
    async def test_a_gate_that_does_not_wake_produces_no_reply_at_all(self):
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        app._run_participation_gate = AsyncMock(return_value=None)
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "gate_required": True, "silence_capable": True})
        await app.handle_message(msg, client)
        client.send_message.assert_not_awaited()
        app.processor.process_message.assert_not_awaited()


# ------------------------------------------------------- modal dual-write + DB columns

class TestDBAndModal:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.makedirs"):
                db = DatabaseManager("test")
                db.db_path = f"{tmpdir}/test.db"
                if getattr(db, "conn", None):
                    db.conn.close()
                db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.conn.execute("PRAGMA journal_mode=WAL")
                db.init_schema()
                yield db
                if getattr(db, "conn", None):
                    db.conn.close()

    def test_new_columns_set_get_preserve_clear(self, temp_db):
        temp_db.set_channel_settings("C1", participation_level="on",
                                     snoozed_until="2026-07-09T20:00:00+00:00")
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] == "on"
        assert row["snoozed_until"] == "2026-07-09T20:00:00+00:00"
        # omitted fields preserved
        temp_db.set_channel_settings("C1", verbosity="low")
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] == "on"
        assert row["snoozed_until"] == "2026-07-09T20:00:00+00:00"
        # explicit None clears
        temp_db.set_channel_settings("C1", participation_level=None, snoozed_until=None)
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] is None
        assert row["snoozed_until"] is None

    def test_muted_threads_set_get_preserve_clear(self, temp_db):
        # F15: muted_threads round-trips as a Python list (stored as JSON), is preserved
        # when omitted, and clears on None/[].
        temp_db.set_channel_settings("C1", muted_threads=["10.0", "20.5"])
        assert temp_db.get_channel_settings("C1")["muted_threads"] == ["10.0", "20.5"]
        temp_db.set_channel_settings("C1", verbosity="low")  # omitted → preserved
        assert temp_db.get_channel_settings("C1")["muted_threads"] == ["10.0", "20.5"]
        temp_db.set_channel_settings("C1", muted_threads=None)  # cleared
        assert temp_db.get_channel_settings("C1")["muted_threads"] == []
        # no row → empty list, never a crash
        assert temp_db.get_channel_settings("C2") is None

    def test_migration_adds_columns_to_legacy_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/legacy.db"
            conn = sqlite3.connect(path)
            conn.execute("""
                CREATE TABLE channel_settings (
                    channel_id TEXT PRIMARY KEY, response_mode TEXT DEFAULT 'tag_only',
                    directives TEXT, reply_in_channel BOOLEAN DEFAULT 0,
                    updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_by TEXT)
            """)
            conn.commit()
            conn.close()
            with patch("os.makedirs"):
                db = DatabaseManager("test")
                if getattr(db, "conn", None):
                    db.conn.close()
                db.db_path = path
                db.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.init_schema()  # runs migrations
                cols = [c[1] for c in db.conn.execute("PRAGMA table_info(channel_settings)")]
                assert "participation_level" in cols and "snoozed_until" in cols
                assert "muted_threads" in cols  # F15 migration
                db.conn.close()

    def test_modal_participation_select_no_snooze_block(self):
        # F15: the snooze early-resume control is retired — the modal never renders it.
        from slack_client.settings_modal import SettingsModal
        builder = SettingsModal.__new__(SettingsModal)
        view = builder.build_channel_settings_modal(
            "C1", {"participation_level": "on"}, "tag_only")
        blocks = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}
        sel = blocks["participation_block"]["element"]
        assert sel["action_id"] == "participation_level"
        assert sel["initial_option"]["value"] == "on"
        values = [o["value"] for o in sel["options"]]
        # Three levels + inherit. The retired restraint dials must not linger as pickable options:
        # a user selecting one would store a level nothing resolves.
        assert values == ["inherit", "mentions_only", "on", "off"]
        assert "snooze_block" not in blocks

    def test_modal_legacy_mode_row_maps_and_no_snooze_block(self):
        from slack_client.settings_modal import SettingsModal
        builder = SettingsModal.__new__(SettingsModal)
        view = builder.build_channel_settings_modal("C1", {"response_mode": "auto_respond"}, "tag_only")
        blocks = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}
        assert blocks["participation_block"]["element"]["initial_option"]["value"] == "on"
        assert "snooze_block" not in blocks


# ----------------------------------------------------- busy rejection → needs_refresh

class TestNeedsRefresh:
    def test_mark_and_consume_semantics(self):
        from message_processor.thread_manager import AsyncThreadStateManager
        mgr = AsyncThreadStateManager.__new__(AsyncThreadStateManager)
        mgr._needs_refresh = set()
        AsyncThreadStateManager.mark_needs_refresh(mgr, "C1:10.0")
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C1:10.0") is True
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C1:10.0") is False  # cleared
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C2:1.0") is False  # cold thread unaffected

    def test_contention_branch_queues_not_rejects(self):
        """Phase Q: lock contention enqueues (no busy rejection); needs_refresh is
        reserved for the loss paths (queue overflow / enqueue failure)."""
        from message_processor import base as mp_base
        src = inspect.getsource(mp_base.MessageProcessor.process_message)
        assert 'type="busy"' not in src
        queued_idx = src.index('type="queued"')
        assert "enqueue_pending" in src[:queued_idx]
        # overflow/failure still flags a transcript refetch
        from message_processor.thread_manager import AsyncThreadStateManager
        assert "mark_needs_refresh" in inspect.getsource(AsyncThreadStateManager.enqueue_pending)

    def test_rebuild_consumes_refresh_flag(self):
        from message_processor import thread_management as tm
        src = inspect.getsource(tm.ThreadManagementMixin._get_or_rebuild_thread_state)
        assert "consume_needs_refresh" in src
        # flag check must be able to flip should_rebuild for WARM threads
        assert src.index("consume_needs_refresh") < src.index("if should_rebuild")
