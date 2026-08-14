"""EDIT §2 — message-level authorization for edit_own_message (spec §9, authorization items).

Two sources feed `ToolContext.authorized_edit_targets`, and both are proof-shaped:

* §2a: the SERIALIZED channel stream — a message is a key only when the model was shown it THIS
  turn under the `assistant` role (receipt-vouched), its durable receipt is finalized AND of
  class `assistant_reply`, and it sits in the pinned channel;
* §2b: a read tool's OWN result — staged by the executor after it proves channel/ownership/
  receipt/exact-ts, committed by the registry only if the ts's own `"ts": …` field survives the
  clipped serialization the model receives, and enrolled onto the TURN so the mapping only
  widens at the top of the NEXT dispatch round.

Everything here fails CLOSED: legacy NULL-class rows, chrome, grandfathered pre-epoch messages,
conflicting duplicate snapshots and malformed anything all resolve to "not editable" — an empty
mapping, never None.
"""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from config import config
from message_processor.channel_stream import (
    SERIALIZER_VERSION,
    InventoryPin,
    PinnedTuple,
    ReceiptRec,
    SidecarPin,
    classify_chrome,
    serialize_stream,
    serializer_config_snapshot,
)
from message_processor.handlers.text import _authorized_edit_targets
from message_processor.turn_runtime import (
    DISCOVERY_SOURCES,
    RECEIPT_CLASS_ASSISTANT_REPLY,
    AuthorizedEditTarget,
    EditRecord,
    TurnRuntime,
)
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.normalizer import NormalizedMessage, parse_ts
from slack_client.search_tool import (
    SlackSearchToolMixin,
    _ChannelScan,
    build_search_query,
)
from message_processor.tool_registry import (
    StagedEditTarget,
    ToolContext,
    ToolRegistry,
    stage_discovered_edit_target,
)

TEAM = "T1"
CH = "C0BKX77NU66"
T0 = "1700000000.000100"   # a human message
T1 = "1700000060.000200"   # our own reply
T2 = "1700003600.000300"   # a second own message
ROOT = "1699999990.000050"
EDITED = "1700000100.000000"
H = "1700007200.000900"
COVERAGE = InventoryPin(start_ts="1699999940.000000", status="complete", reason="genesis")


# ------------------------------------------------------------------ harness (serializer-shaped)

def msg(ts, *, text="hello", sender="U1", sender_type="human", root=None, edited_ts=None,
        channel=CH) -> NormalizedMessage:
    return NormalizedMessage(
        team_id=TEAM, channel_id=channel, ts=ts, thread_root_ts=root, subtype=None,
        sender_id=sender, sender_type=sender_type, raw_bot_name=None, text=text,
        files=(), reactions=(), edited_ts=edited_ts, is_broadcast=False, is_tombstone=False,
        reply_count=None, latest_reply=None, mention_ids=(), origin="history")


def sidecars(*, receipts=(), epoch=None) -> SidecarPin:
    return SidecarPin(
        window=(COVERAGE.start_ts, True), receipts=tuple(receipts),
        receipt_feature_epoch_ts=epoch, coverage=COVERAGE, activity_roots=(),
        activity_event_ts=(), image_analyses=(), document_extractions=(),
        ambient_artifacts=(), tool_usage=(), versions_hash="sidecarhash")


def pinned(messages, *, cards=None, origin=(), origin_root=None) -> PinnedTuple:
    cards = cards if cards is not None else sidecars()
    floor = cards.window[0]
    cfg = serializer_config_snapshot()
    return PinnedTuple(
        team_id=TEAM, channel_id=CH, window=(floor, True), H=H,
        periphery_floor_ts=floor, selection_version=1,
        chrome_ts=classify_chrome(tuple(messages), chrome_markers=cfg["chrome_markers"]),
        fetch_snapshot=tuple(messages), sidecar_versions_hash=cards.versions_hash,
        actor_map=(("U1", "alice"),), actor_map_hash="actorhash",
        serializer_version=SERIALIZER_VERSION, serializer_config_hash="cfghash",
        capability_profile_hash="caphash", tool_schema_version="tools-1",
        coverage=COVERAGE, receipt_feature_epoch_ts=cards.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in cards.receipts),
        sidecars=cards, serializer_config=cfg,
        origin_root_ts=origin_root, origin_snapshot=tuple(origin))


def reply_receipt(ts, *, state="finalized", receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY,
                  thread_root_ts=ROOT) -> ReceiptRec:
    return ReceiptRec(ts=ts, state=state, turn_id="turn-1", thread_root_ts=thread_root_ts,
                      receipt_class=receipt_class)


def channel_turn(**over) -> TurnRuntime:
    turn = TurnRuntime()
    turn.channel_turn_context = SimpleNamespace(channel_id=CH)
    for name, value in over.items():
        setattr(turn, name, value)
    return turn


def target(ts, *, thread_root_ts=ROOT, edited_ts=None) -> AuthorizedEditTarget:
    return AuthorizedEditTarget(channel_id=CH, message_ts=ts, thread_root_ts=thread_root_ts,
                                edited_ts=edited_ts,
                                receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY)


# ================================================================= §2a: the stream mapping

def test_the_stream_mapping_holds_only_finalized_assistant_reply_receipts():
    """Only the exact conjunction authorizes: assistant role (receipt-vouched), finalized
    state, `assistant_reply` class. A legacy NULL-class row, a finalized correction
    announcement and an in-flight reply are each one predicate short — and a human message
    never enters at all."""
    cards = sidecars(receipts=[
        reply_receipt(T1),
        reply_receipt(T2, receipt_class=None),                              # legacy NULL-class
        reply_receipt("1700003660.000400", receipt_class="correction_announcement"),
        reply_receipt("1700003720.000500", state="in_flight"),
    ])
    stream = serialize_stream(pinned([
        msg(T0),                                                            # human
        msg(T1, text="we said this", sender="B0", sender_type="self"),
        msg(T2, text="legacy row", sender="B0", sender_type="self"),
        msg("1700003660.000400", text="Correction to …", sender="B0", sender_type="self"),
        msg("1700003720.000500", text="half written", sender="B0", sender_type="self"),
    ], cards=cards))
    assert set(stream.authorized_edit_targets) == {T1}
    assert stream.authorized_edit_targets[T1] == target(T1)


def test_chrome_never_authorizes_an_edit():
    """A chrome receipt keeps the message out of the stream entirely, so it can never appear in
    the mapping — and even a hand-crafted finalized row of class `chrome` fails the class test."""
    cards = sidecars(receipts=[
        reply_receipt(T1, state="chrome", receipt_class="chrome"),
        reply_receipt(T2, receipt_class="chrome"),   # finalized, but chrome-classed
    ])
    stream = serialize_stream(pinned([
        msg(T1, text="Thinking…", sender="B0", sender_type="self"),
        msg(T2, text="footer", sender="B0", sender_type="self"),
    ], cards=cards))
    assert dict(stream.authorized_edit_targets) == {}


def test_the_mapping_requires_the_pinned_channel():
    """A rendered own message whose metadata names another channel authorizes nothing — the
    edit tool posts into `ctx.channel_id`, and cross-channel editing is off the table."""
    cards = sidecars(receipts=[reply_receipt(T1)])
    stream = serialize_stream(pinned(
        [msg(T1, text="ours, elsewhere", sender="B0", sender_type="self", channel="C0OTHER")],
        cards=cards))
    assert dict(stream.authorized_edit_targets) == {}


def test_the_target_carries_the_receipts_thread_root_and_the_rendered_edited_ts():
    """`thread_root_ts` is the RECEIPT's (the value the disclosure lands under) and `edited_ts`
    is the rendered snapshot — None for a never-edited message, Slack's `edited.ts` otherwise."""
    cards = sidecars(receipts=[reply_receipt(T1, thread_root_ts=ROOT),
                               reply_receipt(T2, thread_root_ts=None)])
    stream = serialize_stream(pinned([
        msg(T1, text="edited later", sender="B0", sender_type="self", root=ROOT,
            edited_ts=EDITED),
        msg(T2, text="top level", sender="B0", sender_type="self"),
    ], cards=cards))
    assert stream.authorized_edit_targets[T1] == target(T1, edited_ts=EDITED)
    assert stream.authorized_edit_targets[T2] == target(T2, thread_root_ts=None)


def test_an_origin_duplicate_of_a_periphery_message_dedups_to_one_agreeing_target():
    """RULING-1 renders an origin message that also sits in the window TWICE; the mapping is
    per-message, so the duplicate appearances collapse to one key (they agree by construction —
    same pin, same receipt)."""
    own = msg(T1, text="both blocks", sender="B0", sender_type="self", root=ROOT)
    cards = sidecars(receipts=[reply_receipt(T1)])
    stream = serialize_stream(pinned([msg(T0), own], cards=cards,
                                     origin=[own], origin_root=ROOT))
    assert set(stream.authorized_edit_targets) == {T1}


def test_message_item_metadata_carries_edited_ts():
    """§2a's pinned fact rides the item metadata — the ts's own row, not a lookup elsewhere."""
    stream = serialize_stream(pinned([msg(T0, edited_ts=EDITED), msg(T1, sender="U2")]))
    by_ts = {item.metadata["ts"]: item for item in stream.message_items}
    assert by_ts[T0].metadata["edited_ts"] == EDITED
    assert by_ts[T1].metadata["edited_ts"] is None


# ================================================================= §2b: enrollment on the turn

def test_enrollment_requires_full_proof_and_refuses_everything_short_of_it():
    turn = channel_turn()
    good = dict(channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=None,
                receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack")
    assert turn.enroll_discovered_edit_target(**good) is True

    refusals = [
        dict(good, message_ts=T2, source="get_message_permalink"),   # not a staging tool
        dict(good, message_ts=T2, channel_id="C0OTHER"),             # not this turn's channel
        dict(good, message_ts=1700000060.0002),                      # ts is not a string
        dict(good, message_ts="yesterday"),                          # ts does not parse
        dict(good, message_ts=T2, receipt_class=None),               # legacy NULL class
        dict(good, message_ts=T2, receipt_class="system_notice"),    # non-reply class
        dict(good, message_ts=T2, edited_ts=1700000100.0),           # malformed edited snapshot
        dict(good, message_ts=T2, thread_root_ts="soon"),            # malformed root
    ]
    for kwargs in refusals:
        assert turn.enroll_discovered_edit_target(**kwargs) is False, kwargs
    assert set(turn.discovered_edit_targets) == {T1}


def test_the_frozen_read_property_cannot_widen_the_turn():
    turn = channel_turn()
    turn.enroll_discovered_edit_target(
        channel_id=CH, message_ts=T1, thread_root_ts=None, edited_ts=None,
        receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="fetch_channel_history")
    view = turn.discovered_edit_targets
    assert isinstance(view, MappingProxyType)
    with pytest.raises(TypeError):
        view[T2] = target(T2)  # type: ignore[index]
    # …and it is a snapshot copy: later reads are fresh, earlier ones do not mutate.
    assert set(turn.discovered_edit_targets) == {T1}


def test_duplicate_enrollments_must_agree_and_a_conflict_excludes_forever():
    turn = channel_turn()
    base = dict(channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=None,
                receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack")
    assert turn.enroll_discovered_edit_target(**base) is True
    # An agreeing re-proof is idempotent…
    assert turn.enroll_discovered_edit_target(**base) is False
    assert set(turn.discovered_edit_targets) == {T1}
    # …a disagreeing one EXCLUDES the target rather than choosing a snapshot…
    assert turn.enroll_discovered_edit_target(**dict(base, edited_ts=EDITED)) is False
    assert dict(turn.discovered_edit_targets) == {}
    # …and the original proof cannot resurrect it.
    assert turn.enroll_discovered_edit_target(**base) is False
    assert dict(turn.discovered_edit_targets) == {}
    assert T1 in turn.conflicted_edit_targets


# ================================================================= the combiner (text.py)

def test_missing_or_malformed_anything_resolves_to_the_empty_mapping_never_none():
    assert _authorized_edit_targets(None) == {}
    assert _authorized_edit_targets(TurnRuntime()) == {}                       # no stream
    broken = channel_turn(channel_stream=SimpleNamespace(authorized_edit_targets="junk"))
    assert _authorized_edit_targets(broken) == {}
    for value in (_authorized_edit_targets(None), _authorized_edit_targets(broken)):
        assert value is not None and len(value) == 0
    # The context's own default is the same fail-closed shape.
    assert ToolContext().authorized_edit_targets == {}
    assert ToolContext().authorized_edit_targets is not None


def test_the_combiner_unions_stream_and_discovered_targets():
    turn = channel_turn(channel_stream=SimpleNamespace(
        authorized_edit_targets={T1: target(T1, edited_ts=EDITED)}))
    turn.enroll_discovered_edit_target(
        channel_id=CH, message_ts=T2, thread_root_ts=None, edited_ts=None,
        receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="fetch_thread_messages")
    combined = _authorized_edit_targets(turn)
    assert combined == {T1: target(T1, edited_ts=EDITED),
                        T2: target(T2, thread_root_ts=None)}


def test_a_stream_discovery_disagreement_excludes_the_target():
    """One ts, two snapshots that disagree about `edited_ts` — the message changed between the
    stream build and the read. Neither snapshot is chosen; the ts is not editable this turn."""
    turn = channel_turn(channel_stream=SimpleNamespace(
        authorized_edit_targets={T1: target(T1, edited_ts=None)}))
    turn.enroll_discovered_edit_target(
        channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=EDITED,
        receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack")
    assert _authorized_edit_targets(turn) == {}


def test_an_agreeing_duplicate_across_sources_stays_one_entry():
    turn = channel_turn(channel_stream=SimpleNamespace(
        authorized_edit_targets={T1: target(T1)}))
    turn.enroll_discovered_edit_target(
        channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=None,
        receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack")
    assert _authorized_edit_targets(turn) == {T1: target(T1)}


def test_a_discovery_conflict_is_not_resurrected_by_the_stream():
    """Two tools disagreed, so the turn excluded the ts — a stream-rendered appearance of the
    same ts must not put it back on the table."""
    turn = channel_turn(channel_stream=SimpleNamespace(
        authorized_edit_targets={T1: target(T1)}))
    base = dict(channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=None,
                receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack")
    turn.enroll_discovered_edit_target(**base)
    turn.enroll_discovered_edit_target(**dict(base, edited_ts=EDITED))  # the conflict
    assert _authorized_edit_targets(turn) == {}


# ================================================================= §2b: staging tools

class _HistoryHost(SlackHistoryToolMixin):
    """The mixin with exactly the host surface the staging seam touches."""

    def __init__(self, *, own_ts=(), team=TEAM):
        self.self_team_id = team
        self._own_ts = set(own_ts)
        self.warnings = []

    def is_own_message(self, m):
        return m.get("ts") in self._own_ts

    def log_warning(self, message):
        self.warnings.append(message)


class _FakeDB:
    def __init__(self, receipts):
        self.receipts = receipts
        self.calls = []

    async def read_channel_sidecars_for_async(self, team_id, channel_id, message_ts,
                                              busy_timeout_ms=None):
        self.calls.append((team_id, channel_id, tuple(message_ts)))
        return {"receipts": self.receipts, "receipt_feature_epoch_ts": None}


def staging_ctx(*, channel_id=CH, db=None):
    return SimpleNamespace(channel_id=channel_id, db=db,
                           tool_flight=SimpleNamespace(staged_edit_targets=[]))


@pytest.mark.parametrize("thread_ts,expected_source", [
    (None, "fetch_channel_history"),
    (ROOT, "fetch_thread_messages"),
])
async def test_history_stages_a_proved_own_reply_under_its_own_tool_name(
        thread_ts, expected_source):
    host = _HistoryHost(own_ts={T1})
    db = _FakeDB([{"message_ts": T1, "state": "finalized",
                   "receipt_class": "assistant_reply", "thread_root_ts": ROOT}])
    ctx = staging_ctx(db=db)
    raw = [{"ts": T0, "text": "a human", "user": "U1"},
           {"ts": T1, "text": "ours", "bot_id": "B0", "edited": {"ts": EDITED}}]
    await host._stage_history_edit_targets(ctx, CH, raw, thread_ts)
    assert ctx.tool_flight.staged_edit_targets == [StagedEditTarget(
        channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=EDITED,
        receipt_class="assistant_reply", source=expected_source, field="ts")]
    assert expected_source in DISCOVERY_SOURCES
    # ONE batched read, over exactly the own messages in the returned window.
    assert db.calls == [(TEAM, CH, (T1,))]


async def test_history_stages_nothing_without_a_finalized_assistant_reply_receipt():
    """No row (grandfathered or lost), a legacy NULL-class row and an in-flight row each stage
    nothing — readability is not editability, and no inference covers for a missing class."""
    host = _HistoryHost(own_ts={T0, T1, T2})
    db = _FakeDB([
        {"message_ts": T1, "state": "finalized", "receipt_class": None,
         "thread_root_ts": None},                                       # legacy NULL class
        {"message_ts": T2, "state": "in_flight", "receipt_class": "assistant_reply",
         "thread_root_ts": None},                                       # not finalized
    ])                                                                  # T0: no row at all
    ctx = staging_ctx(db=db)
    raw = [{"ts": ts, "text": "ours", "bot_id": "B0"} for ts in (T0, T1, T2)]
    await host._stage_history_edit_targets(ctx, CH, raw, None)
    assert ctx.tool_flight.staged_edit_targets == []


async def test_history_stages_nothing_for_another_channel_or_foreign_messages():
    host = _HistoryHost(own_ts={T1})
    db = _FakeDB([{"message_ts": T1, "state": "finalized",
                   "receipt_class": "assistant_reply", "thread_root_ts": None}])
    # A fetch of a channel that is not the turn's own stages nothing…
    ctx = staging_ctx(db=db)
    await host._stage_history_edit_targets(ctx, "C0OTHER", [{"ts": T1, "bot_id": "B0"}], None)
    assert ctx.tool_flight.staged_edit_targets == []
    # …and a window holding only foreign messages never reads the database at all.
    ctx2 = staging_ctx(db=db)
    await host._stage_history_edit_targets(ctx2, CH, [{"ts": T0, "user": "U1"}], None)
    assert ctx2.tool_flight.staged_edit_targets == []
    assert db.calls == []


class _SearchHost(SlackSearchToolMixin):
    def __init__(self):
        self.warnings = []

    def log_warning(self, message):
        self.warnings.append(message)


def scan_with(receipt_rows):
    scan = _ChannelScan(query=build_search_query("anything"),
                        trigger_key=parse_ts(H), limit=5)
    scan.receipt_rows = receipt_rows
    return scan


def test_search_stages_a_proved_own_reply():
    host = _SearchHost()
    scan = scan_with({T1: {"message_ts": T1, "state": "finalized",
                           "receipt_class": "assistant_reply", "thread_root_ts": ROOT}})
    ctx = staging_ctx()
    payload = {"results": [
        {"channel": CH, "ts": T0, "thread_ts": None},
        {"channel": CH, "ts": T1, "thread_ts": ROOT},
    ]}
    kept = [({"ts": T0, "user": "U1"}, CH, [TEAM]),
            ({"ts": T1, "bot_id": "B0", "edited": {"ts": EDITED}}, CH, [TEAM])]
    host._stage_scan_edit_targets(ctx, CH, scan, payload, kept)
    assert ctx.tool_flight.staged_edit_targets == [StagedEditTarget(
        channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=EDITED,
        receipt_class="assistant_reply", source="search_slack", field="ts")]


def test_search_grandfathering_grants_nothing():
    """A pre-epoch own message is SEARCHABLE without a receipt row (the §S3 grandfather), but
    it has no row — and no row means no edit claim. The exact asymmetry §2b pins."""
    host = _SearchHost()
    scan = scan_with({})   # the batched read returned no rows: pre-epoch, or receipts lost
    ctx = staging_ctx()
    payload = {"results": [{"channel": CH, "ts": T1, "thread_ts": None}]}
    kept = [({"ts": T1, "bot_id": "B0"}, CH, [TEAM])]
    host._stage_scan_edit_targets(ctx, CH, scan, payload, kept)
    assert ctx.tool_flight.staged_edit_targets == []


def test_search_stages_nothing_for_legacy_or_unfinalized_rows():
    host = _SearchHost()
    scan = scan_with({
        T1: {"message_ts": T1, "state": "finalized", "receipt_class": None,
             "thread_root_ts": None},
        T2: {"message_ts": T2, "state": "in_flight", "receipt_class": "assistant_reply",
             "thread_root_ts": None},
    })
    ctx = staging_ctx()
    payload = {"results": [{"channel": CH, "ts": T1}, {"channel": CH, "ts": T2}]}
    kept = [({"ts": T1, "bot_id": "B0"}, CH, [TEAM]), ({"ts": T2, "bot_id": "B0"}, CH, [TEAM])]
    host._stage_scan_edit_targets(ctx, CH, scan, payload, kept)
    assert ctx.tool_flight.staged_edit_targets == []


# ================================================================= the registry seam

def make_registry_with(executors):
    registry = ToolRegistry()
    for name, executor in executors.items():
        registry.register({"type": "function", "name": name,
                           "parameters": {"type": "object", "properties": {}}}, executor)
    return registry


async def test_a_committed_claim_authorizes_the_next_round_never_its_own():
    """The §2b timing rule, driven through the REAL registry: a read that proves a target and an
    edit-shaped check in the SAME round see the round-start mapping; the NEXT round's restamp is
    where the proof takes effect."""
    turn = channel_turn()
    seen = []

    async def reader(ctx, args):
        stage_discovered_edit_target(
            ctx, channel_id=CH, message_ts=T1, thread_root_ts=ROOT, edited_ts=None,
            receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack", field="ts")
        return {"ok": True, "results": [{"ts": T1}]}

    async def checker(ctx, args):
        seen.append(dict(ctx.authorized_edit_targets))
        return {"ok": True}

    registry = make_registry_with({"reader": reader, "checker": checker})
    ctx = ToolContext(channel_id=CH, turn=turn)

    await registry.dispatch_all(ctx, [
        {"name": "reader", "arguments": {}, "call_id": "c1"},
        {"name": "checker", "arguments": {}, "call_id": "c2"},
    ])
    assert seen == [{}], "a same-round read and edit must not authorize each other"
    assert set(turn.discovered_edit_targets) == {T1}, "the claim committed onto the TURN"

    await registry.dispatch_all(ctx, [{"name": "checker", "arguments": {}, "call_id": "c3"}])
    assert seen[1] == {T1: target(T1)}, "the NEXT round's restamp carries the proof"


async def test_a_claim_whose_ts_is_clipped_away_grants_nothing(monkeypatch):
    """The exact `_survives_truncation` rule: the target's own `"ts": …` field must reach the
    model. A claim buried past the clip — however truthfully staged — authorizes nothing."""
    monkeypatch.setattr(config, "tool_result_max_chars", 200, raising=False)
    turn = channel_turn()

    async def reader(ctx, args):
        stage_discovered_edit_target(
            ctx, channel_id=CH, message_ts=T1, thread_root_ts=None, edited_ts=None,
            receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack", field="ts")
        return {"ok": True, "padding": "x" * 5000, "results": [{"ts": T1}]}

    registry = make_registry_with({"reader": reader})
    ctx = ToolContext(channel_id=CH, turn=turn)
    await registry.dispatch_all(ctx, [{"name": "reader", "arguments": {}, "call_id": "c1"}])
    assert dict(turn.discovered_edit_targets) == {}


async def test_the_digits_alone_in_prose_do_not_survive_the_clip_rule():
    """The pair check, not a substring check: a result that merely MENTIONS the timestamp in
    someone's text never enrolls the target — only the structured field does."""
    turn = channel_turn()

    async def reader(ctx, args):
        stage_discovered_edit_target(
            ctx, channel_id=CH, message_ts=T1, thread_root_ts=None, edited_ts=None,
            receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack", field="ts")
        return {"ok": True, "results": [{"text": f"someone quoted {T1} here"}]}

    registry = make_registry_with({"reader": reader})
    ctx = ToolContext(channel_id=CH, turn=turn)
    await registry.dispatch_all(ctx, [{"name": "reader", "arguments": {}, "call_id": "c1"}])
    assert dict(turn.discovered_edit_targets) == {}


def test_staging_without_a_flight_claims_nothing():
    ctx = SimpleNamespace(tool_flight=None)
    stage_discovered_edit_target(
        ctx, channel_id=CH, message_ts=T1, thread_root_ts=None, edited_ts=None,
        receipt_class=RECEIPT_CLASS_ASSISTANT_REPLY, source="search_slack", field="ts")
    # No flight, no staging area, no error — a hand-built context outside the registry.


# ================================================================= wave-B carriers exist

def test_the_edit_record_carriers_have_the_pinned_shape():
    """Wave B builds the executor on these exact names; pin them so a rename is a loud test
    failure here rather than a quiet AttributeError there."""
    turn = TurnRuntime()
    assert turn.edits == []
    record = EditRecord(channel_id=CH, target_ts=T1, announcement_ts=T2)
    assert (record.state, record.error) == ("announcement_only", None)
    turn.edits.append(record)
    assert turn.committed_edits == []
    record.state = "committed"
    record.error = None
    assert turn.committed_edits == [record]

    from message_processor.turn_runtime import DEST_KIND_CORRECTION_ANNOUNCEMENT
    assert DEST_KIND_CORRECTION_ANNOUNCEMENT == "correction_announcement"
