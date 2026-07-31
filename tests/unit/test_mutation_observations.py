"""Durable snapshot-invalidation observations on the canonical mutation path (P4 §1c, R0-3).

The activity index alone cannot carry this: it writes nothing for an unthreaded top-level edit
unless that ts is already an indexed root, and a separate listener would race the index for
ordering. So the SAME ticketed feed writes both, in ONE transaction — and the tests below are
mostly about what happens when only one of the two halves would have been written.

A1's `record_activity_and_mutation_async` is mocked here at its pinned signature (interfaces
§3.5); the fake enforces the real thing's two load-bearing properties — all-or-nothing, and
INSERT OR IGNORE on the unique key.
"""
import logging

import pytest

from database import DatabaseManager
from slack_client import admission_watermark
from slack_client.event_handlers.activity_index import (feed_own_mutation,
                                                        feed_thread_activity_index)
from slack_client.event_handlers.registration import _admit
from slack_client.utilities import SlackUtilitiesMixin

TEAM = "T1"
CH = "C1"
ROOT = "100.000100"
MUTATION_FIELDS = ("team_id", "channel_id", "subject_ts", "kind", "observation_identity",
                   "observed_at")


class _FakeDB:
    """The §3.5 accessor, plus the reads the index half arbitrates against."""

    def __init__(self):
        self.index_rows = {}
        self.mutations = []
        self._mutation_keys = set()
        self.known_roots = set()
        self.receipts = set()
        self.calls = []
        self.failures = 0

    # -- reads the feed makes before the write ------------------------------------------
    async def seed_channel_coverage_async(self, team_id, channel_id, ts):
        return None

    async def thread_activity_exists_async(self, team_id, channel_id, root_ts):
        return (team_id, channel_id, root_ts) in self.known_roots

    async def get_receipt_async(self, team_id, channel_id, ts):
        return {"ts": ts} if (channel_id, ts) in self.receipts else None

    async def record_thread_activity_async(self, *a, **k):
        raise AssertionError("the live feed must write through the one-transaction accessor")

    # -- interfaces §3.5 ------------------------------------------------------------------
    async def record_activity_and_mutation_async(self, *, observation=None, mutation=None):
        self.calls.append((observation, mutation))
        if self.failures > 0:
            # Raised BEFORE either half is applied: that is what makes the fake honest about
            # the transaction, and it is the whole point of the atomicity test below.
            self.failures -= 1
            raise RuntimeError("database is locked")
        if observation is not None:
            key = (observation["team_id"], observation["channel_id"], observation["root_ts"])
            row = self.index_rows.setdefault(key, {"dirty": False})
            row["reply_ts"] = observation["reply_ts"] or row.get("reply_ts")
            row["event_ts"] = observation["event_ts"] or row.get("event_ts")
            row["dirty"] = row["dirty"] or bool(observation["mark_dirty"])
        if mutation is not None:
            unique = (mutation["team_id"], mutation["channel_id"], mutation["subject_ts"],
                      mutation["kind"], mutation["observation_identity"])
            if None in unique:
                raise AssertionError(f"a NULL in the mutation unique key: {unique}")
            if unique in self._mutation_keys:
                return
            self._mutation_keys.add(unique)
            self.mutations.append(dict(mutation))


class _Client(SlackUtilitiesMixin):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.self_team_id = TEAM
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = "A123"


@pytest.fixture
def db():
    return _FakeDB()


@pytest.fixture
def client(db):
    return _Client(db)


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    """A real DatabaseManager, for the one test that proves this feed's row dict composes with
    A1's landed accessor and its actual unique key."""
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    instance = DatabaseManager(platform="slack")
    yield instance
    instance.conn.close()


@pytest.fixture
def wm():
    """The ticket singleton, reset afterwards: a degraded channel is process-wide state."""
    instance = admission_watermark.watermark
    instance.reset()
    yield instance
    instance.reset()


def _changed(message, event_ts="900.000100", channel=CH, **extra):
    event = {"type": "message", "subtype": "message_changed", "channel": channel,
             "channel_type": "channel", "message": message}
    if event_ts is not None:
        event["event_ts"] = event_ts
    event.update(extra)
    return event


def _deleted(previous, deleted_ts, event_ts="900.000100", channel=CH):
    return {"type": "message", "subtype": "message_deleted", "channel": channel,
            "channel_type": "channel", "previous_message": previous,
            "deleted_ts": deleted_ts, "event_ts": event_ts}


def _reply(ts="101.000100", root=ROOT, **extra):
    event = {"type": "message", "channel": CH, "channel_type": "channel", "user": "U1",
             "ts": ts, "thread_ts": root, "text": "a reply", "event_ts": ts}
    event.update(extra)
    return event


def _only(mutations):
    assert len(mutations) == 1, f"expected exactly one observation, got {mutations}"
    return mutations[0]


# ------------------------------------------------------- Appendix C test 2: NULL-free identity

async def test_owned_operation_with_no_slack_event_id_inserts_exactly_once(client, db):
    """Appendix C test 2. An operation WE performed carries no Slack event id, so its identity
    is the stable operation id — and SQLite treats NULLs as distinct, so a null one would make
    every replay a new row and defeat the unique key it is part of."""
    for _ in range(3):
        assert await feed_own_mutation(client, CH, "500.000100", "delete",
                                       operation_id="delete:C1:500.000100") is True

    row = _only(db.mutations)
    assert row["subject_ts"] == "500.000100"
    assert row["kind"] == "delete"
    assert row["observation_identity"] == "op:delete:C1:500.000100"
    assert all(row[field] is not None for field in MUTATION_FIELDS)
    assert len(db.calls) == 3, "each replay must reach the accessor and be ignored there"


async def test_owned_operation_retried_in_place_still_inserts_once(client, db):
    db.failures = 2
    assert await feed_own_mutation(client, CH, "500.000100", "edit",
                                   operation_id="update:C1:500.000100:a") is True
    _only(db.mutations)


async def test_owned_operation_with_no_slack_event_id_inserts_once_in_the_real_store(
        client, real_db):
    """Appendix C test 2 against A1's landed accessor and its real unique key, so the dict this
    feed builds is proven to compose with the schema rather than only with the fake."""
    client.db = real_db
    for _ in range(3):
        assert await feed_own_mutation(client, CH, "500.000100", "delete",
                                       operation_id="delete:C1:500.000100") is True
    rows = await real_db.mutation_observations_after_async(TEAM, CH, 0,
                                                          subject_ts_in=["500.000100"])
    assert len(rows) == 1
    assert rows[0]["observation_identity"] == "op:delete:C1:500.000100"
    assert rows[0]["kind"] == "delete"

    # A DIFFERENT operation on the same message is a second observation, not a collapsed one.
    assert await feed_own_mutation(client, CH, "500.000100", "delete",
                                   operation_id="delete:C1:500.000100:again") is True
    assert len(await real_db.mutation_observations_after_async(
        TEAM, CH, 0, subject_ts_in=["500.000100"])) == 2


@pytest.mark.parametrize("operation_id", [None, "", "   "])
async def test_owned_operation_without_a_usable_id_mints_one(client, db, operation_id):
    """An EMPTY identity defeats the unique key exactly as a NULL one does, and the accessor
    rejects it — so a blank operation id must mint a real one here, not pass the blank down."""
    assert await feed_own_mutation(client, CH, "500.000100", "edit",
                                   operation_id=operation_id) is True
    identity = _only(db.mutations)["observation_identity"]
    assert identity.startswith("op:") and len(identity) > len("op:")


async def test_a_blank_slack_event_id_never_becomes_the_identity(client, db):
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "edited"}, event_id=""))
    assert _only(db.mutations)["observation_identity"] == "900.000100"


async def test_a_lost_owned_operation_is_logged_CRITICAL_not_swallowed(client, db, caplog):
    """This path holds no ticket, so nothing will ever retry it — which is the reason the
    terminal failure is loud rather than the reason it could be quiet."""
    db.failures = 99
    with caplog.at_level(logging.CRITICAL):
        assert await feed_own_mutation(client, CH, "500.000100", "edit") is False
    assert db.mutations == []
    assert [r.levelno for r in caplog.records
            if "own mutation observation lost" in r.getMessage()] == [logging.CRITICAL]


@pytest.mark.parametrize("channel,subject_ts,kind", [
    ("D999", "500.000100", "edit"),      # a DM has no stream and no snapshot
    (CH, "not-a-timestamp", "edit"),
    (CH, "500.000100", "reaction"),
    (CH, None, "edit"),
])
async def test_owned_operation_refuses_what_it_cannot_record(client, db, channel, subject_ts,
                                                             kind):
    assert await feed_own_mutation(client, channel, subject_ts, kind) is False
    assert db.calls == []


# ----------------------------------------------------------------- ONE ticketed transaction

async def test_a_failing_write_commits_neither_half_and_fails_the_ticket(client, db, wm,
                                                                        caplog):
    """Driven through the real feed, not the helper: the ticket is the thing under test. A
    swallowed failure would certify to every turn in that window that the index caught up on
    an event nobody indexed."""
    db.known_roots.add((TEAM, CH, ROOT))
    db.failures = 99
    ticket = wm.issue(CH)
    with caplog.at_level(logging.CRITICAL):
        await feed_thread_activity_index(
            client, _changed({"ts": "101.000100", "thread_ts": ROOT, "user": "U1",
                              "text": "edited"}), ticket=ticket)

    assert db.index_rows == {}, "the index half committed without the mutation half"
    assert db.mutations == [], "the mutation half committed without the index half"
    assert ticket.state == "failed"
    assert wm.is_degraded(CH) is True


async def test_the_retained_replay_carries_both_halves(client, db, wm):
    """A replay that re-ran only the index half would complete the ticket while the snapshot
    store went on believing nothing had changed."""
    db.known_roots.add((TEAM, CH, ROOT))
    db.failures = 3
    ticket = wm.issue(CH)
    await feed_thread_activity_index(
        client, _changed({"ts": "101.000100", "thread_ts": ROOT, "user": "U1",
                          "text": "edited"}), ticket=ticket)
    assert ticket.state == "failed"

    assert await ticket.retry() is True
    assert db.index_rows[(TEAM, CH, ROOT)]["dirty"] is True
    assert _only(db.mutations)["subject_ts"] == "101.000100"


async def test_both_halves_ride_one_accessor_call(client, db, wm):
    db.known_roots.add((TEAM, CH, ROOT))
    ticket = wm.issue(CH)
    await feed_thread_activity_index(
        client, _changed({"ts": "101.000100", "thread_ts": ROOT, "user": "U1",
                          "text": "edited"}), ticket=ticket)

    assert len(db.calls) == 1
    observation, mutation = db.calls[0]
    assert observation["root_ts"] == ROOT
    assert mutation["subject_ts"] == "101.000100"
    assert ticket.state == "ok"


# --------------------------------------------------------------- the unthreaded top-level edit

async def test_unthreaded_top_level_edit_writes_a_mutation_with_no_index_row(client, db, wm):
    """THE case R0-3 exists for. The index writes nothing — the ts is not a known root and the
    payload carries no threading hint — and without the mutation half this edit would leave no
    durable trace at all, so a snapshot summarizing it would never be invalidated."""
    ticket = wm.issue(CH)
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "typo fixed"}),
        ticket=ticket)

    assert db.index_rows == {}
    assert db.calls[0][0] is None, "an index observation was passed for an unindexed root"
    row = _only(db.mutations)
    assert row["subject_ts"] == "500.000100"
    assert row["kind"] == "edit"
    assert ticket.state == "ok"


async def test_unthreaded_top_level_deletion_writes_a_mutation_too(client, db):
    await feed_thread_activity_index(
        client, _deleted({"ts": "500.000100", "user": "U1", "text": "gone"},
                         deleted_ts="500.000100"))
    assert db.index_rows == {}
    assert _only(db.mutations)["kind"] == "delete"


async def test_our_own_housekeeping_delete_still_records_a_mutation(client, db):
    """An anonymous deletion the receipt ledger says was ours: the index deliberately records
    nothing (it is not channel activity), but the message is still gone from the room."""
    db.known_roots.add((TEAM, CH, ROOT))
    db.receipts.add((CH, ROOT))
    await feed_thread_activity_index(
        client, _deleted({"ts": ROOT, "text": "gone", "reply_count": 2}, deleted_ts=ROOT))

    assert db.index_rows == {}
    assert _only(db.mutations)["subject_ts"] == ROOT


async def test_an_ordinary_reply_is_no_mutation(client, db):
    await feed_thread_activity_index(client, _reply())
    assert db.mutations == []
    assert db.index_rows[(TEAM, CH, ROOT)]["reply_ts"] == "101.000100"


async def test_a_dm_mutation_is_not_recorded(client, db):
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "x"}, channel="D999",
                         channel_type="im"))
    assert db.calls == []


# --------------------------------------------------------------------- own-message mutations

async def test_own_message_edit_records_without_advancing_h_or_waking(client, db, wm):
    """§1c: the non-wake invalidation feed. H must not move for our own edit — a turn that
    could wait on its own reply would deadlock — and no ticket is issued, so nothing downstream
    is woken by it. The observation is still written."""
    event = _changed({"ts": "500.000100", "user": "UBOT", "text": "our own answer, edited"})
    ticket = _admit(client, event)
    await feed_thread_activity_index(client, event, ticket=ticket)

    assert ticket is None, "an own-message mutation must take no readiness ticket"
    assert wm.current(CH) is None, "an own-message mutation advanced H"
    assert wm.is_degraded(CH) is False
    assert db.index_rows == {}
    row = _only(db.mutations)
    assert row["subject_ts"] == "500.000100"
    assert row["kind"] == "edit"


async def test_own_message_deletion_delivery_records_without_advancing_h(client, db, wm):
    event = _deleted({"ts": "500.000100", "bot_id": "BBOT", "text": "gone"},
                     deleted_ts="500.000100")
    ticket = _admit(client, event)
    await feed_thread_activity_index(client, event, ticket=ticket)

    assert ticket is None
    assert wm.current(CH) is None
    assert _only(db.mutations)["kind"] == "delete"


async def test_the_owned_operation_feed_never_touches_the_watermark(client, db, wm):
    await feed_own_mutation(client, CH, "500.000100", "delete", operation_id="delete:1")
    assert wm.current(CH) is None
    assert wm.pending_failures(CH) == []
    assert wm.is_degraded(CH) is False
    assert len(db.mutations) == 1


# ------------------------------------------------------- subject_ts and identity derivation

async def test_subject_ts_is_the_edited_message_never_the_envelope(client, db):
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "edited",
                          "edited": {"user": "U1", "ts": "888.000100"}}))
    row = _only(db.mutations)
    # The envelope's event_ts is when the edit happened; the subject is what changed.
    assert row["subject_ts"] == "500.000100"
    assert row["observation_identity"] == "900.000100"


async def test_identity_falls_back_to_the_nested_edited_ts(client, db):
    """The same fallback `registration._admit` honours: an edit with no outer event_ts is placed
    from `edited.ts`, and both readers get it from the one normalizer definition."""
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "edited",
                          "edited": {"user": "U1", "ts": "888.000100"}}, event_ts=None))
    row = _only(db.mutations)
    assert row["subject_ts"] == "500.000100"
    assert row["observation_identity"] == "888.000100"


async def test_identity_prefers_a_slack_event_id_when_the_delivery_carries_one(client, db):
    await feed_thread_activity_index(
        client, _changed({"ts": "500.000100", "user": "U1", "text": "edited"},
                         event_id="Ev0PV52K21"))
    assert _only(db.mutations)["observation_identity"] == "Ev0PV52K21"


async def test_identity_of_an_activity_less_mutation_is_still_derived_and_stable(client, db):
    """A shape with neither an event id nor any activity ts. The identity must still be a pure
    function of the payload: a clock reading or a uuid would insert a second row on every
    replay of the one event."""
    event = _changed({"ts": "500.000100", "user": "U1", "text": "edited"}, event_ts=None)
    await feed_thread_activity_index(client, event)
    await feed_thread_activity_index(client, event)
    assert _only(db.mutations)["observation_identity"] == "subject:500.000100"


async def test_deletion_subject_is_the_previous_message_ts(client, db):
    await feed_thread_activity_index(
        client, _deleted({"ts": "500.000100", "user": "U1", "text": "gone", "reply_count": 2},
                         deleted_ts="500.000100"))
    row = _only(db.mutations)
    assert row["subject_ts"] == "500.000100"
    assert row["kind"] == "delete"
    assert row["observation_identity"] == "900.000100"


async def test_a_tombstone_delivery_is_a_deletion(client, db):
    """Slack announces a deleted root as message_changed carrying a tombstone. The durable
    vocabulary has two values, and this one is a delete."""
    await feed_thread_activity_index(
        client, _changed({"ts": ROOT, "thread_ts": ROOT, "subtype": "tombstone",
                          "text": "This message was deleted."}))
    assert _only(db.mutations)["kind"] == "delete"


async def test_a_malformed_mutation_records_nothing_and_fails_its_ticket(client, db, wm,
                                                                        caplog):
    """A mutation whose subject cannot be read has no subject_ts, so there is no observation to
    write — and its outer event_ts already advanced H, so it must not report success either."""
    ticket = wm.issue(CH)
    with caplog.at_level(logging.CRITICAL):
        await feed_thread_activity_index(client, _changed(None), ticket=ticket)
    assert db.calls == []
    assert ticket.state == "failed"
