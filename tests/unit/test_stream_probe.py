"""T55 — the read-only diagnostic replay (SHALLOW_STREAM_RESPEC §4.11).

The probe exists to inspect a real channel's build without being a turn, and its whole value
rests on one promise: it writes nothing durable. That promise is asserted here by TRAPPING WRITE
ACCESSORS BY NAME on a recording fake database, not by opening a file read-only — a read-only
handle proves only that a write would have failed, while this proves no write was ever reached.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import stream_probe

TEAM = "T1"
CH = "C0BKX77NU66"
ROOT_A = "1700000100.000000"
ROOT_B = "1700000300.000000"

pytestmark = pytest.mark.asyncio


def raw(ts, *, text="hi", user="U1", root=None, **extra):
    payload = {"ts": ts, "text": text, "user": user, "type": "message"}
    if root:
        payload["thread_ts"] = root
    payload.update(extra)
    return payload


class _Client:
    """The read surface the build actually uses, and nothing else.

    Every Slack WRITE method is deliberately absent rather than mocked: an attribute error on a
    method the probe should never call is a louder failure than a recorded call nobody asserts.
    """

    def __init__(self, *, history=None, replies=None):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.bot_handle = "chatgpt-dev"
        self.app = MagicMock()
        self.calls = []
        self.app.client.conversations_history = AsyncMock(
            side_effect=self._record("conversations_history",
                                     history or {"ok": True, "messages": []}))
        self.app.client.conversations_replies = AsyncMock(
            side_effect=self._record("conversations_replies",
                                     replies or {"ok": True, "messages": []}))

    def _record(self, name, payload):
        async def _call(**kwargs):
            self.calls.append((name, kwargs))
            if callable(payload):
                return payload(**kwargs)
            return payload
        return _call

    def is_own_message(self, msg):
        return bool(msg) and (msg.get("bot_id") == self.bot_id
                              or msg.get("user") == self.bot_user_id)

    def classify_sender(self, msg):
        if self.is_own_message(msg):
            return "self"
        if msg.get("bot_id") or msg.get("app_id"):
            return "other_bot"
        return "human"

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25, stats=None):
        if stats is not None:
            stats["remote_lookups"] = 0
            stats.setdefault("attempted_ids", set()).update(ids)
        return {uid: f"name-{uid}" for uid in ids}

    async def _ensure_self_identity(self):
        # THE REAL CLIENT CALLS `auth.test` HERE. Recording it keeps the fake honest: a fake
        # that silently skipped it would let the probe's read-call inventory claim two methods
        # while the real command used four, which is how the old docstring came to be false.
        self.calls.append(("auth_test", {}))
        self.self_team_id = TEAM
        return None


# Every write accessor the turn path can reach on a DatabaseManager. Trapping by NAME is the
# point: a probe that grew a new write would have to add its name here to stay green, which is
# a decision someone has to make deliberately rather than one that slips through.
TRAPPED_WRITES = (
    "advance_channel_window_anchor_async", "clear_thread_dirty_async",
    "record_thread_activity_async", "register_receipt_async", "save_document_async",
    "save_document_if_absent_async", "set_meta_if_absent_async",
    "advance_channel_coverage_async", "seed_channel_coverage_async",
    "acquire_coverage_sweep_async", "save_image_analysis_async",
    "record_message_tool_usage_async", "set_channel_setting_async",
)


class _RecordingDB:
    """The three READS the split-phase build performs, and a trap on every write."""

    def __init__(self, *, anchor=None, inventory=None, activity_roots=None, receipt_roots=()):
        self.writes = []
        self.reads = []
        self._anchor = anchor
        self._inventory = inventory
        self._activity_roots = dict(activity_roots or {})
        self._receipt_roots = tuple(receipt_roots)
        for name in TRAPPED_WRITES:
            setattr(self, name, self._trap(name))

    def _trap(self, name):
        async def _call(*args, **kwargs):
            self.writes.append(name)
            return True
        return _call

    async def read_channel_window_anchor_async(self, team_id, channel_id):
        self.reads.append("anchor")
        return {"anchor": self._anchor, "inventory": self._inventory}

    async def read_channel_discovery_roots_async(self, team_id, channel_id, *, floor_ts, high_ts):
        self.reads.append(("discovery", floor_ts, high_ts))
        return {"activity_roots": dict(self._activity_roots),
                "receipt_roots": tuple(self._receipt_roots)}

    async def read_channel_sidecars_for_async(self, team_id, channel_id, message_ts):
        self.reads.append(("sidecars", tuple(message_ts)))
        return {"ids": sorted(message_ts), "receipt_feature_epoch_ts": None, "receipts": [],
                "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
                "tool_usage": {}, "versions_hash": "h" * 64}


def _history_pages():
    return {"ok": True, "messages": [raw(ROOT_A, text="room chatter"),
                                     raw(ROOT_B, text="another root")]}


def _replies_for(**kwargs):
    root = kwargs.get("ts")
    if root == ROOT_A:
        return {"ok": True, "messages": [raw(ROOT_A, text="room chatter"),
                                         raw("1700000150.000000", text="a reply", root=ROOT_A)]}
    if root == ROOT_B:
        return {"ok": True, "messages": [raw(ROOT_B, text="another root")]}
    return {"ok": True, "messages": []}


def _drive(monkeypatch, tmp_path, *, origins, client=None, db=None):
    """Drive the REAL CLI — argument parsing, the phases, the report write, the exit code."""
    client = client or _Client(history=_history_pages(), replies=_replies_for)
    db = db or _RecordingDB(anchor=None, inventory=None)

    import slack_client.base as slack_base

    # NO `raising=False` ON EITHER. A patch that tolerates a missing attribute will happily
    # CREATE the name it meant to replace, so the test passes while the CLI imports something
    # that does not exist — which is exactly what happened here, and only the live run caught
    # it. Failing loudly on a rename is the whole value of naming the real symbol.
    monkeypatch.setattr(slack_base, "SlackBot", lambda: client)
    monkeypatch.setattr(stream_probe, "DatabaseManager", lambda platform=None: db)

    out = tmp_path / "report.json"
    argv = ["--channel", CH, "--out", str(out)]
    for origin in origins:
        argv += ["--origin", origin]
    code = stream_probe.main(argv)
    return code, client, db, json.loads(out.read_text()), out


def test_the_probe_writes_nothing(monkeypatch, tmp_path):
    """T55. ZERO durable writes — Slack, DB and ledger — and the report is the ONE file."""
    ledger = tmp_path / "participation.jsonl"
    ledger.write_text('{"v": 8, "event": "session_start"}\n')
    before = ledger.read_bytes()

    code, client, db, report, out = _drive(monkeypatch, tmp_path, origins=[ROOT_A])

    # THE DB TRAP. Not one write accessor was REACHED — which a read-only file handle could
    # never show, since it can only prove a write would have failed once attempted.
    assert db.writes == [], f"the probe reached write accessors: {db.writes}"
    # It did do its three reads, so the zero above is not the zero of a probe that did nothing.
    assert "anchor" in db.reads
    assert any(isinstance(r, tuple) and r[0] == "discovery" for r in db.reads)
    assert any(isinstance(r, tuple) and r[0] == "sidecars" for r in db.reads)

    # NO SLACK WRITES, and the READ inventory is exactly what the docstring claims — auth.test
    # for identity plus the two history readers. `users.info` rides inside `resolve_usernames`,
    # which this fake serves locally; it is a read either way.
    assert {name for name, _kwargs in client.calls} <= {
        "auth_test", "conversations_history", "conversations_replies"}
    assert ("auth_test", {}) in client.calls, "the real client resolves identity; so must this"
    assert any(name == "conversations_history" for name, _ in client.calls)

    # NO LEDGER WRITES — byte-identical before and after.
    assert ledger.read_bytes() == before

    # THE REPORT — §4.11's schema, and the two fields that say what kind of build this was.
    assert code == 0, report
    assert report["probe_version"] == stream_probe.PROBE_VERSION
    assert report["h_source"] == "wall_clock_upper_bound"
    assert report["frontier"] == 0
    assert report["channel_id"] == CH
    assert report["anchor_advanced"] is False
    for field in ("periphery_floor_ts", "selection_version", "root_count", "message_count",
                  "candidate_count", "orphan_root_count", "inventory_state", "history_pages",
                  "reply_pages", "origin_pages", "byte_count", "origin_byte_count",
                  "stream_sha256", "union_sha256", "reselected", "origin_root_ts",
                  "origin_count", "origin_complete", "origin_stability", "errors"):
        assert field in report, field
    # The report file is the ONE permitted write, and it is the command's whole output.
    assert out.exists()


def test_the_two_origin_mode_proves_the_shared_prefix(monkeypatch, tmp_path):
    """T55's render-equality mode: ONE periphery, TWO origin pins, in ONE process.

    Two separate invocations cannot prove this — their H values differ by construction, so their
    peripheries differ and any hash comparison between them is meaningless.
    """
    code, client, db, report, _out = _drive(monkeypatch, tmp_path, origins=[ROOT_A, ROOT_B])

    assert code == 0, report
    assert db.writes == []
    assert report["prefix_identical"] is True
    assert report["unions_differ"] is True
    assert len(report["origins"]) == 2
    assert [entry["origin_root_ts"] for entry in report["origins"]] == [ROOT_A, ROOT_B]

    # The per-origin fields have NO single value in this mode and are omitted from the top
    # level; `origin_pages` carries the SUM, so the page block still means "pages this spent".
    for field in ("origin_root_ts", "origin_count", "origin_complete", "origin_stability",
                  "union_sha256"):
        assert field not in report, field
    assert report["origin_pages"] == sum(e["origin_pages"] for e in report["origins"])
    # Every PERIPHERY field stays at top level, because there is exactly one periphery.
    for field in ("root_count", "message_count", "periphery_floor_ts", "history_pages",
                  "reply_pages", "byte_count", "stream_sha256"):
        assert field in report, field


def test_the_probe_calls_the_phases_exactly_once_each(monkeypatch, tmp_path):
    """T123. The probe's SHAPE: one shared periphery, N origins, and the composer never called.

    This is what the two-origin mode is FOR — the report's booleans are downstream of it. A probe
    that built two shared pins would report `prefix_identical: true` for two peripheries that
    merely happened to match, which proves nothing about the invariant.

    SYNCHRONOUS: the probe is a CLI and its entry point owns its own event loop.
    """
    from message_processor import channel_stream as cs

    calls = {"pin": 0, "origin_fetch": 0, "origin_pin": 0, "composer": 0}
    real_pin, real_fetch = cs.build_channel_pin, cs.fetch_origin_thread
    real_origin_pin, real_composer = cs.build_origin_pin, cs.build_channel_stream
    deadlines = []

    async def _pin(*a, **kw):
        calls["pin"] += 1
        deadlines.append(kw.get("deadline_at"))
        assert kw.get("probe") is True, "the probe must suppress the dirty compare-and-clear"
        return await real_pin(*a, **kw)

    async def _fetch(client, channel_id, root, h, budget, trigger_ts):
        calls["origin_fetch"] += 1
        deadlines.append(budget.deadline_at)
        assert trigger_ts is None, "a diagnostic replay has no trigger"
        return await real_fetch(client, channel_id, root, h, budget, trigger_ts)

    async def _origin_pin(*a, **kw):
        calls["origin_pin"] += 1
        return await real_origin_pin(*a, **kw)

    async def _composer(*a, **kw):
        calls["composer"] += 1
        return await real_composer(*a, **kw)

    monkeypatch.setattr(cs, "build_channel_pin", _pin)
    monkeypatch.setattr(cs, "fetch_origin_thread", _fetch)
    monkeypatch.setattr(cs, "build_origin_pin", _origin_pin)
    monkeypatch.setattr(cs, "build_channel_stream", _composer)
    monkeypatch.setattr(stream_probe, "build_channel_pin", _pin)
    monkeypatch.setattr(stream_probe, "fetch_origin_thread", _fetch)
    monkeypatch.setattr(stream_probe, "build_origin_pin", _origin_pin)

    code, _client, db, report, _out = _drive(monkeypatch, tmp_path, origins=[ROOT_A, ROOT_B])

    assert code == 0, report
    assert calls["pin"] == 1, "ONE shared periphery pin, whatever the origin count"
    assert calls["origin_fetch"] == 2 and calls["origin_pin"] == 2
    # THE COMPOSER IS NEVER CALLED — which is why the anchor persist and the telemetry emit are
    # unreachable rather than suppressed.
    assert calls["composer"] == 0
    assert db.writes == []
    # Every component shares ONE absolute deadline.
    assert deadlines and len(set(deadlines)) == 1, deadlines
    # And both origins are graded, with a verdict each.
    assert [e["origin_complete"] for e in report["origins"]] == [True, True]
    assert [e["origin_stability"] for e in report["origins"]] == ["stable", "stable"]


async def test_the_origin_verdict_separates_an_unstable_world_from_a_bad_build(monkeypatch,
                                                                               tmp_path):
    """T124. The membership/race table — the probe's most subtle logic, and the half that
    decides whether a mismatch is the BUILD's fault or the WORLD's.

    Only a mismatch that survives a build-side re-pin is reported as a build failure. The probe
    observes the world at a later instant than the build did, so blaming the build for a receipt
    that finalized after its pin would be the probe lying in the expensive direction.
    """
    from message_processor import channel_stream as cs

    shared = SimpleNamespace(team_id=TEAM, channel_id=CH,
                             serializer_config={"chrome_markers": ()})
    origin_fetch = SimpleNamespace(origin_root_ts=ROOT_A, messages=())
    stream = SimpleNamespace(origin_items=(), origin_count=0)

    async def _verdict(walks, build_sides):
        """Drive `_origin_verdict` with scripted membership answers."""
        walk_iter, build_iter = iter(walks), iter(build_sides)

        async def _walk(*a, **kw):
            got = next(walk_iter)
            return got, stream_probe._ts_sha256(got)

        async def _build_side(*a, **kw):
            got = next(build_iter)
            return got, stream_probe._ts_sha256(got)

        monkeypatch.setattr(stream_probe, "_independent_origin_walk", _walk)
        monkeypatch.setattr(stream_probe, "_build_side_membership", _build_side)
        errors = []
        return await stream_probe._origin_verdict(
            None, None, shared, origin_fetch, stream, h="9.0", deadline_at=1.0, errors=errors)

    built = []          # the build rendered nothing, so its membership hash is the empty one

    # ROW 0 — the walk agrees with the build: COMPLETE and STABLE.
    out = await _verdict(walks=[built], build_sides=[])
    assert (out["origin_complete"], out["origin_stability"]) == (True, "stable")

    # ROW 1 — the build-side RE-PIN now matches the walk: the world moved AFTER the build's pin,
    # and the build was honest at its pin. Inconclusive, never a failure.
    out = await _verdict(walks=[["1.0"], ["1.0"]], build_sides=[["1.0"]])
    assert out["origin_complete"] is None and out["origin_stability"] == "unstable"

    # ROW 2 — the two independent walks DISAGREE: Slack changed underneath the probe.
    out = await _verdict(walks=[["1.0"], ["1.0", "2.0"]], build_sides=[[]])
    assert out["origin_complete"] is None and out["origin_stability"] == "unstable"

    # ROW 3 — the re-pin reproduces the BUILD's own membership and it still differs from a
    # stable walk: the mismatch SURVIVES the re-pin, and only this is the build's fault.
    out = await _verdict(walks=[["1.0"], ["1.0"]], build_sides=[built])
    assert out["origin_complete"] is False and out["origin_stability"] == "stable"

    assert cs is not None
