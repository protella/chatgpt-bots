"""The periphery floor and the re-anchor policy (SHALLOW_STREAM_RESPEC §2d, §4.2).

The floor is a POLICY decision — "this is where the recent window starts" — and deliberately not
a coverage claim, not a horizon, and not a statement that nothing exists below it. Two properties
make it safe to persist, and both are asserted here:

  * it only ever moves FORWARD at a given selection version, so a slow turn cannot drag a channel
    backwards into history it has already stopped rendering;
  * a write carrying a LOWER selection version is REJECTED, so an old process — a rolling
    restart, a straggler turn holding a pre-upgrade pin — cannot overwrite a newer-policy row.

The selection ALGORITHM that chooses the floor lands with the build; this file covers the store
it is persisted through.
"""
from __future__ import annotations

import pytest

from database import DatabaseManager

TEAM = "T1"
CH = "C0BKX77NU66"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="windowtest")
    yield db
    db.conn.close()


async def _anchor(db):
    return (await db.read_channel_window_anchor_async(TEAM, CH))["anchor"]


# ------------------------------------------------------------------ T31: the three version rules

async def test_the_anchor_upsert_rejects_a_stale_version_writer(temp_db):
    """T31. Three cases against real SQLite, because the whole rule is one SQL predicate and a
    mock would be asserting my own reading of it back to me."""
    # A fresh row is written.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "100.0", 1) is True
    assert await _anchor(temp_db) == {"floor_ts": "100.0", "selection_version": 1}

    # EQUAL version: forward moves, backward does not.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "200.0", 1) is True
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "150.0", 1) is False
    assert await _anchor(temp_db) == {"floor_ts": "200.0", "selection_version": 1}

    # LOWER version: REJECTED OUTRIGHT, and it leaves BOTH fields untouched. This is the case
    # the two-branch predicate exists for — a `!=` test would read "different" as "newer" and
    # let this stale writer drag the floor back to a policy no longer in force.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "999.0", 0) is False
    assert await _anchor(temp_db) == {"floor_ts": "200.0", "selection_version": 1}

    # HIGHER version: overwrites unconditionally — including BACKWARD, which is the point. A
    # policy change resets the floor, and the new policy's floor is wherever it says.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "50.0", 2) is True
    assert await _anchor(temp_db) == {"floor_ts": "50.0", "selection_version": 2}


async def test_the_floor_comparison_is_numeric_not_lexicographic(temp_db):
    """Slack timestamps are not fixed-width, so "999.123456" sorts AFTER "1000.5" as text. A
    string comparison here would refuse a legitimate forward move at every decade boundary."""
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "999.123456", 1) is True
    # Numerically forward, lexicographically backward.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "1000.5", 1) is True
    assert (await _anchor(temp_db))["floor_ts"] == "1000.5"
    # And the reverse is refused, though it sorts later as text.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "999.999999", 1) is False


async def test_an_equal_floor_is_not_a_move(temp_db):
    """Writing the floor a build already read reports False — `anchor_advanced` means the row
    MOVED, and a turn that re-derived the same floor moved nothing."""
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "300.0", 1) is True
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "300.0", 1) is False


async def test_two_writers_racing_forward_converge(temp_db):
    """Two turns re-anchoring concurrently: whichever order the writes land in, the stored floor
    ends at the NEWER value and never regresses. One statement, so neither has to read the
    other's write first."""
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "500.0", 1) is True
    # The older turn's write arrives second and is refused.
    assert await temp_db.advance_channel_window_anchor_async(TEAM, CH, "400.0", 1) is False
    assert (await _anchor(temp_db))["floor_ts"] == "500.0"


async def test_the_anchor_and_the_inventory_are_read_together(temp_db):
    """READ 1 stage 1 returns exactly two things, and an ABSENT inventory row is `None` — the
    representation the turn path already uses for "never swept"."""
    payload = await temp_db.read_channel_window_anchor_async(TEAM, CH)
    assert set(payload) == {"anchor", "inventory"}
    assert payload == {"anchor": None, "inventory": None}

    await temp_db.seed_channel_coverage_async(TEAM, CH, "10.0")
    await temp_db.advance_channel_window_anchor_async(TEAM, CH, "100.0", 1)

    payload = await temp_db.read_channel_window_anchor_async(TEAM, CH)
    assert payload["anchor"] == {"floor_ts": "100.0", "selection_version": 1}
    assert payload["inventory"]["inventory_start_ts"] == "10.0"
    assert payload["inventory"]["bootstrap_status"] == "pending"


async def test_stage_one_reads_no_activity_and_no_receipt_rows(temp_db):
    """The staging is the point (RULING-10): those need a window, and the window is not known
    until the history walk has run. A stage-1 read that returned them would be the stale-floor
    fan-out this split exists to prevent."""
    await temp_db.record_thread_activity_async(TEAM, CH, "50.0", reply_ts="60.0",
                                               event_ts="60.0")
    await temp_db.register_receipt_async(TEAM, CH, "70.0", "turn-1", "finalized",
                                         thread_root_ts="50.0")

    payload = await temp_db.read_channel_window_anchor_async(TEAM, CH)
    assert set(payload) == {"anchor", "inventory"}
    assert "activity_roots" not in payload and "receipt_roots" not in payload


# ------------------------------------------------------------------ stage 2: discovery only

async def test_discovery_returns_a_mapping_of_pinned_event_timestamps(temp_db):
    """`activity_roots` is a MAPPING, not a set, and the value is what compare-and-clear later
    compares against. A set could not express "this root's ts moved since I looked", and the
    clear would be unsafe."""
    await temp_db.record_thread_activity_async(TEAM, CH, "300.0", reply_ts="600.0",
                                               event_ts="600.0")
    got = await temp_db.read_channel_discovery_roots_async(
        TEAM, CH, floor_ts="400.0", high_ts="800.0")

    assert got["activity_roots"] == {"300.0": "600.0"}
    assert isinstance(got["receipt_roots"], tuple)


async def test_a_dirty_root_is_exempt_from_the_floor_but_not_from_h(temp_db):
    """FX1's findings F3/F4, preserved verbatim. An edit says nothing about where the mutated
    message sits, so a dirty root comes back until someone fetches it — but a mutation above this
    turn's frontier cannot appear in its stream, so scheduling a fetch for it would let a later
    event delay an already-admitted older turn."""
    # Dirty, and far BELOW the floor: still returned.
    await temp_db.record_thread_activity_async(TEAM, CH, "10.0", mark_dirty=True)
    # Dirty, but its event is ABOVE H: withheld.
    await temp_db.record_thread_activity_async(TEAM, CH, "20.0", event_ts="9999.0",
                                               mark_dirty=True)

    got = await temp_db.read_channel_discovery_roots_async(
        TEAM, CH, floor_ts="400.0", high_ts="800.0")
    assert "10.0" in got["activity_roots"], "a dirty root must be exempt from the floor"
    assert "20.0" not in got["activity_roots"], "a dirty root is NOT exempt from H"


async def test_receipt_roots_name_the_threads_we_have_posted_in(temp_db):
    """Our own posts name threads the history walk cannot surface — `conversations.history`
    returns only top-level messages. Predicated into the same window by the ts of the message
    each receipt describes."""
    await temp_db.register_receipt_async(TEAM, CH, "500.0", "turn-1", "finalized",
                                         thread_root_ts="120.0")
    # Outside the window: its message ts is above H.
    await temp_db.register_receipt_async(TEAM, CH, "9000.0", "turn-2", "finalized",
                                         thread_root_ts="130.0")

    got = await temp_db.read_channel_discovery_roots_async(
        TEAM, CH, floor_ts="400.0", high_ts="800.0")
    assert got["receipt_roots"] == ("120.0",)


async def test_discovery_windows_on_the_floor_it_is_given(temp_db):
    """The accessor windows on the floor the CALLER passes — which the builder sets to the
    EFFECTIVE floor, not the stored one. Binding it to a stale stored floor is what pulls months
    of roots out of the index and into reply fan-out."""
    for root, reply in (("100.0", "150.0"), ("500.0", "550.0"), ("700.0", "750.0")):
        await temp_db.record_thread_activity_async(TEAM, CH, root, reply_ts=reply,
                                                   event_ts=reply)

    stale = await temp_db.read_channel_discovery_roots_async(
        TEAM, CH, floor_ts="0.0", high_ts="800.0")
    effective = await temp_db.read_channel_discovery_roots_async(
        TEAM, CH, floor_ts="600.0", high_ts="800.0")

    assert set(stale["activity_roots"]) == {"100.0", "500.0", "700.0"}
    assert set(effective["activity_roots"]) == {"700.0"}, (
        "a narrower floor must bound the DB fan-out, not merely the history depth")


# ================================================ §2d's count rule, over the real selector

from message_processor.channel_stream import _select_floor          # noqa: E402
from slack_client.normalizer import ORIGIN_HISTORY, NormalizedMessage   # noqa: E402


def ev(ts, *, root=None, sender="U1", sender_type="human"):
    """One eligible event. `root` set (and different from `ts`) makes it a REPLY."""
    return NormalizedMessage(
        team_id=TEAM, channel_id=CH, ts=ts, thread_root_ts=root, subtype=None,
        sender_id=sender, sender_type=sender_type, raw_bot_name=None, text="x",
        files=(), reactions=(), edited_ts=None, is_broadcast=False, is_tombstone=False,
        reply_count=None, latest_reply=None, mention_ids=(), origin=ORIGIN_HISTORY)


def roots(n, *, start=1000):
    return [ev(f"{start + i}.000000") for i in range(n)]


def select(events, *, floor_read=None, target=50, ceiling=100, inverted=False):
    return _select_floor(events, floor_read=floor_read, target=target, ceiling=ceiling,
                         inverted=inverted)


@pytest.mark.parametrize("target", [15, 30])
def test_the_count_is_roots_and_the_floor_selects_events(target):
    """T24. THE COUNT-DEFINITION TEST. `N` roots plus ONE foreign thread carrying 49 replies:
    every root survives, all 49 replies ride along, and no reply moved the floor.

    The swamping problem was never that replies are unwelcome — it was that roots and replies
    competed for ONE budget, so a 49-reply thread could evict 49 roots and turn a picture of the
    room into a picture of one conversation."""
    ceiling = target * 2
    the_roots = roots(ceiling + 1)                     # one PAST the ceiling ⇒ re-anchor
    busy = the_roots[0].ts
    replies = [ev(f"{2000 + i}.000000", root=busy) for i in range(49)]

    floor = select(the_roots + replies, target=target, ceiling=ceiling)
    kept = [e for e in the_roots + replies if float(e.ts) >= float(floor)]

    # The floor lands on the OLDEST root among the newest TARGET — counted over ROOTS only.
    assert len([e for e in kept if not e.is_reply]) == target
    # Every reply is newer than the floor here, so all 49 ride along, uncounted.
    assert len([e for e in kept if e.is_reply]) == 49


def test_the_ceiling_is_a_root_ceiling():
    """T26. `R == CEILING` keeps the floor; `R == CEILING + 1` re-anchors. And a hundred replies
    under existing roots move NOTHING — a test that counts events passes the first half and
    fails the second."""
    at_ceiling = roots(100)
    assert select(at_ceiling, floor_read="900.000000") == "900.000000"

    over = roots(101)
    reanchored = select(over, floor_read="900.000000")
    assert reanchored != "900.000000"
    assert len([r for r in over if float(r.ts) >= float(reanchored)]) == 50

    # A HUNDRED REPLIES under those same roots move nothing.
    flood = at_ceiling + [ev(f"{5000 + i}.000000", root=at_ceiling[0].ts) for i in range(100)]
    assert select(flood, floor_read="900.000000") == "900.000000"


def test_the_floor_never_moves_backward():
    """T27. Deletes drop the root count below TARGET: the floor is UNCHANGED. Moving it back
    would re-fetch arbitrary older history to satisfy a number, and would invalidate the cached
    prefix in the one direction that buys nothing."""
    survivors = roots(3, start=2000)
    assert select(survivors, floor_read="1500.000000") == "1500.000000"


def test_a_cold_channel_renders_everything_it_has():
    """T28. Fewer than TARGET roots: no re-anchor, no failure, and the floor is the oldest
    eligible EVENT — event, not root, so a channel whose oldest reachable message is a reply
    starts THERE."""
    oldest_is_a_reply = [ev("900.000000", root="500.000000")] + roots(3, start=1000)
    assert select(oldest_is_a_reply) == "900.000000"


def test_an_empty_channel_selects_the_sentinel():
    """T29's selector half. Zero eligible events and no stored floor ⇒ the empty-floor sentinel,
    which is the empty STRING — falsy, a `str` so no type widens, and it can never collide with
    a real Slack ts. `parse_ts` is never called on it."""
    assert select([]) == ""
    # A STORED FLOOR IS NEVER RESET: a channel whose events were all deleted keeps it.
    assert select([], floor_read="1500.000000") == "1500.000000"


def test_replies_ride_along_uncounted_and_interleaved():
    """T32. 30 roots + 30 replies at TARGET=50: all 60 are in the window and the root count is
    30. A `thread_broadcast` counts as the REPLY it is, never as a root."""
    the_roots = roots(30, start=1000)
    replies = [ev(f"{3000 + i}.000000", root=the_roots[i].ts) for i in range(30)]
    floor = select(the_roots + replies, target=50, ceiling=100)

    kept = [e for e in the_roots + replies if not floor or float(e.ts) >= float(floor)]
    assert len(kept) == 60
    assert len([e for e in kept if not e.is_reply]) == 30


def test_a_floor_newer_than_h_keeps_the_pinned_floor():
    """T69's selector half. An inverted window renders from the floor it PINNED — not the
    sentinel, which is reserved for a channel with no eligible events at all."""
    assert select([], floor_read="9999.000000", inverted=True) == "9999.000000"


async def test_concurrent_reanchors_report_reselected_and_converge(temp_db):
    """T30. Two builds re-anchoring at once: BOTH report `reselected`, the stored floor ends at
    the newer value, and each build's bytes match ITS OWN pinned floor.

    The two fields answer different questions, which is why r1's single boolean could not serve.
    `reselected` is a pure property of the BUILD — this build chose a floor different from the
    one it read — so it is always decidable and never racy, and it is what the live battery
    grades. `anchor_advanced` reports whether the UPSERT actually moved the row, so under a race
    it may be true for either or both. Demanding "exactly one turn reports it" is a thing
    concurrency cannot guarantee.
    """
    read_floor = "1000.000000"
    await temp_db.advance_channel_window_anchor_async(TEAM, CH, read_floor, 1)

    # Two builds, each having read the SAME floor and each having selected a newer one.
    older_choice, newer_choice = "2000.000000", "3000.000000"
    assert select([], floor_read=read_floor) == read_floor         # no events ⇒ no reselection

    # Both are reselections: each chose a floor different from the one it read.
    assert older_choice != read_floor and newer_choice != read_floor

    # The writes land in either order and the store CONVERGES FORWARD, never regressing.
    first = await temp_db.advance_channel_window_anchor_async(TEAM, CH, newer_choice, 1)
    second = await temp_db.advance_channel_window_anchor_async(TEAM, CH, older_choice, 1)
    assert first is True, "the newer floor moved the row"
    assert second is False, "the older write is refused rather than dragging the floor back"
    assert (await _anchor(temp_db))["floor_ts"] == newer_choice

    # And in the reverse arrival order the store still ends at the newer value — BOTH writes
    # report having moved it, which is the case `anchor_advanced` is allowed to be true twice.
    await temp_db.advance_channel_window_anchor_async(TEAM, CH, "4000.000000", 2)
    a = await temp_db.advance_channel_window_anchor_async(TEAM, CH, "5000.000000", 2)
    b = await temp_db.advance_channel_window_anchor_async(TEAM, CH, "6000.000000", 2)
    assert a is True and b is True
    assert (await _anchor(temp_db))["floor_ts"] == "6000.000000"


# ============================================ T50 — a failed anchor write never fails the turn

class _AnchorClient:
    def __init__(self):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.bot_handle = "chatgpt-dev"
        from unittest.mock import AsyncMock, MagicMock
        self.app = MagicMock()
        self.app.client.conversations_history = AsyncMock(return_value={
            "ok": True, "messages": [{"ts": "1000.0", "text": "room", "user": "U1",
                                      "type": "message"}]})
        self.app.client.conversations_replies = AsyncMock(return_value={
            "ok": True, "messages": [{"ts": "1000.0", "text": "room", "user": "U1",
                                      "type": "message"}]})

    def is_own_message(self, msg):
        return bool(msg) and msg.get("user") == self.bot_user_id

    def classify_sender(self, msg):
        return "self" if self.is_own_message(msg) else "human"

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25, stats=None):
        if stats is not None:
            stats["remote_lookups"] = 0
            stats.setdefault("attempted_ids", set()).update(ids)
        return {uid: f"name-{uid}" for uid in ids}


class _AnchorDB:
    """The reads, plus an anchor write whose behaviour the test chooses."""

    def __init__(self, *, anchor=None, on_advance=None):
        self._anchor = anchor
        self._on_advance = on_advance
        self.advance_calls = []

    async def read_channel_window_anchor_async(self, team_id, channel_id):
        return {"anchor": self._anchor, "inventory": None}

    async def read_channel_discovery_roots_async(self, team_id, channel_id, *, floor_ts, high_ts):
        return {"activity_roots": {}, "receipt_roots": ()}

    async def read_channel_sidecars_for_async(self, team_id, channel_id, message_ts):
        return {"ids": sorted(message_ts), "receipt_feature_epoch_ts": None, "receipts": [],
                "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
                "tool_usage": {}, "versions_hash": "v" * 64}

    async def advance_channel_window_anchor_async(self, team_id, channel_id, floor_ts, version):
        self.advance_calls.append((floor_ts, version))
        if self._on_advance is not None:
            return self._on_advance()
        return True


async def _anchor_build(db, *, h="9999.0", turn_id="turn-1"):
    from unittest.mock import patch

    from message_processor import channel_stream as cs
    from message_processor import participation_telemetry as pt
    from slack_client import admission_watermark

    emitted = []

    async def _drain(channel_id, frontier, timeout=None):
        return None

    with patch.object(admission_watermark, "drain", _drain), \
         patch.object(pt, "stream_render", lambda **kw: emitted.append(kw)):
        result = await cs.build_channel_stream(
            client=_AnchorClient(), db=db, team_id=TEAM, channel_id=CH, h=h,
            origin_root_ts="1000.0", turn_id=turn_id, trigger_ts="1000.0")
    return result, emitted


async def test_an_anchor_write_failure_does_not_fail_the_turn(caplog):
    """T50. The three no-move TURN paths, and `anchor_advanced` is False on ALL of them.

    "We did not move it" and "we could not move it" are both honestly False — the distinction
    lives in the log, not in a telemetry boolean, because a build whose bytes are already
    correct has nothing to tell the model about a write it does not depend on.
    """
    def _raise():
        raise RuntimeError("database is locked")

    # PATH 1 — THE WRITE RAISES. The bytes are unchanged, a WARNING is logged, the turn stands.
    failing = _AnchorDB(anchor=None, on_advance=_raise)
    with caplog.at_level("WARNING"):
        failed_result, failed_emitted = await _anchor_build(failing)
    assert failing.advance_calls, "the build must have attempted the write"
    assert failed_result.anchor_advanced is False
    assert failed_emitted[0]["anchor_advanced"] is False
    assert any("anchor" in r.message for r in caplog.records), "a failure is a WARNING"

    # The next turn re-derives the SAME floor from the same world — nothing was lost by the
    # failed write, which is why it is a warning rather than a turn failure.
    again, _ = await _anchor_build(_AnchorDB(anchor=None), turn_id="turn-2")
    assert again.stream.periphery_floor_ts == failed_result.stream.periphery_floor_ts
    assert again.stream.stream_sha256 == failed_result.stream.stream_sha256

    # PATH 2 — THE UNCHANGED FLOOR. A stored floor the build re-selects is not written again.
    stored = {"floor_ts": "1000.0", "selection_version": 1}
    unchanged_db = _AnchorDB(anchor=stored)
    unchanged, unchanged_emitted = await _anchor_build(unchanged_db, turn_id="turn-3")
    assert unchanged.stream.periphery_floor_ts == "1000.0"
    assert unchanged_db.advance_calls == [], "an unchanged floor writes no row"
    assert unchanged.anchor_advanced is False
    assert unchanged_emitted[0]["anchor_advanced"] is False

    # PATH 3 — THE EMPTY FLOOR. A channel with no eligible events writes no row at all: the
    # sentinel is never persisted, and absence is the stored form of "no floor".
    class _EmptyClient(_AnchorClient):
        def __init__(self):
            super().__init__()
            from unittest.mock import AsyncMock
            self.app.client.conversations_history = AsyncMock(
                return_value={"ok": True, "messages": []})
            self.app.client.conversations_replies = AsyncMock(
                return_value={"ok": True, "messages": []})

    from unittest.mock import patch

    from message_processor import channel_stream as cs
    from message_processor import participation_telemetry as pt
    from slack_client import admission_watermark

    empty_db = _AnchorDB(anchor=None)
    empty_emitted = []

    async def _drain(channel_id, frontier, timeout=None):
        return None

    with patch.object(admission_watermark, "drain", _drain), \
         patch.object(pt, "stream_render", lambda **kw: empty_emitted.append(kw)):
        empty = await cs.build_channel_stream(
            client=_EmptyClient(), db=empty_db, team_id=TEAM, channel_id=CH, h="9999.0",
            origin_root_ts=None, turn_id="turn-4", trigger_ts=None)

    assert empty.stream.periphery_floor_ts == ""
    assert empty_db.advance_calls == [], "the empty-floor sentinel is never persisted"
    assert empty.anchor_advanced is False
    assert empty_emitted[0]["anchor_advanced"] is False


def test_the_probe_reports_no_anchor_advance_in_its_own_report(tmp_path, monkeypatch):
    """T50's PROBE path, asserted separately and read off the JSON REPORT — never a ledger row.

    A probe emits no `stream_render` at all, so asserting a field on one would be asserting a
    row that cannot exist.

    SYNCHRONOUS on purpose: the probe is a CLI, and its entry point owns its own event loop —
    driving it from inside one would be testing a shape production never runs.
    """
    from tests.unit.test_stream_probe import _drive

    code, _client, db, report, _out = _drive(monkeypatch, tmp_path, origins=["1700000100.000000"])
    assert code == 0
    assert report["anchor_advanced"] is False
    assert db.writes == [], "the probe never reaches the anchor write at all"
