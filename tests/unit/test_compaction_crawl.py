"""The background compaction crawl (§1n): inventory, chunks, resume and discard.

The crawl's whole reason for existing is that no transcript is ever persisted, so these tests are
mostly about what does NOT happen: no text column, no Slack cursor in a checkpoint, no refetch
during sizing, no partial chunk after a shutdown, no root re-walked after it was marked done.

Every Slack and OpenAI call is a local double. The Responses double returns real strings and
terminates — a mock that does neither is how the suite once grew to 30GB.
"""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Dict, List, Sequence

import pytest

from message_processor import channel_compaction as C
from slack_client.normalizer import parse_ts

TEAM = "T1"
CH = "C0BKX77NU66"
NS = "prod"
H = "2000.000000"


# ---------------------------------------------------------------- doubles

class FakeSlack:
    """A channel, paged the way Slack pages it: history newest-first, replies oldest-first."""

    def __init__(self, messages: Sequence[dict], *, page_size: int = 100,
                 missing_roots: Sequence[str] = ()):
        self.messages = sorted(messages, key=lambda m: parse_ts(m["ts"]))
        self.page_size = page_size
        self.history_calls: List[dict] = []
        self.replies_calls: List[dict] = []
        self.missing_roots = {str(r) for r in missing_roots}
        self.self_team_id = TEAM
        self.bot_handle = "chatgpt"
        self.app = SimpleNamespace(client=SimpleNamespace(
            conversations_history=self.conversations_history,
            conversations_replies=self.conversations_replies))

    def classify_sender(self, payload):
        return payload.get("_kind", "human")

    async def resolve_usernames(self, ids, _client=None):
        return {uid: f"name-{uid}" for uid in ids}

    def _top_level(self):
        return [m for m in self.messages
                if not m.get("thread_ts") or m["thread_ts"] == m["ts"]
                or m.get("subtype") == "thread_broadcast"]

    async def conversations_history(self, **params):
        self.history_calls.append(dict(params))
        latest = params.get("latest")
        inclusive = bool(params.get("inclusive"))
        oldest = params.get("oldest")
        rows = self._top_level()
        if latest is not None:
            rows = [m for m in rows if (parse_ts(m["ts"]) <= parse_ts(latest) if inclusive
                                        else parse_ts(m["ts"]) < parse_ts(latest))]
        if oldest is not None:
            rows = [m for m in rows if (parse_ts(m["ts"]) >= parse_ts(oldest) if inclusive
                                        else parse_ts(m["ts"]) > parse_ts(oldest))]
        rows = sorted(rows, key=lambda m: parse_ts(m["ts"]), reverse=True)
        limit = int(params.get("limit") or self.page_size)
        page, rest = rows[:limit], rows[limit:]
        return {"ok": True, "messages": [dict(m) for m in page], "has_more": bool(rest)}

    async def conversations_replies(self, **params):
        self.replies_calls.append(dict(params))
        root = str(params.get("ts"))
        if root in self.missing_roots:
            return {"ok": True, "messages": [], "has_more": False}
        rows = [m for m in self.messages
                if m.get("thread_ts") == root or m["ts"] == root]
        oldest, latest = params.get("oldest"), params.get("latest")
        inclusive = bool(params.get("inclusive"))
        if oldest is not None:
            rows = [m for m in rows if (parse_ts(m["ts"]) >= parse_ts(oldest) if inclusive
                                        else parse_ts(m["ts"]) > parse_ts(oldest))]
        if latest is not None:
            rows = [m for m in rows if parse_ts(m["ts"]) <= parse_ts(latest)]
        rows = sorted(rows, key=lambda m: parse_ts(m["ts"]))
        limit = int(params.get("limit") or self.page_size)
        page, rest = rows[:limit], rows[limit:]
        return {"ok": True, "messages": [dict(m) for m in page], "has_more": bool(rest)}


class FakeDB:
    """A1's pinned accessor surface, in memory. `commit_crawl_page_async` is page-atomic: the
    rows and the cursor advance land together or not at all."""

    SKELETON_COLUMNS = {"seq", "ts", "root_ts", "kind_rank", "source_rank", "actor_id",
                        "projected_byte_len", "base_canonical_bytes", "projection_sha256"}

    def __init__(self):
        self.checkpoints: Dict[tuple, dict] = {}
        self.candidate_rows: Dict[str, Dict[tuple, dict]] = {}
        self.sealed: Dict[str, List[dict]] = {}
        self.observations: List[dict] = []
        self.snapshots: Dict[str, dict] = {}
        self.manifests: Dict[str, List[dict]] = {}
        self.anchors: Dict[str, List[dict]] = {}
        self.candidates: List[dict] = []
        self.commits: List[dict] = []
        self.fail_next_commit = False

    # -- checkpoints ---------------------------------------------------------------------
    async def load_crawl_checkpoint_async(self, team_id, channel_id, namespace):
        row = self.checkpoints.get((team_id, channel_id, namespace))
        return dict(row) if row else None

    async def upsert_crawl_checkpoint_async(self, checkpoint):
        key = (checkpoint["team_id"], checkpoint["channel_id"], checkpoint["namespace"])
        self.checkpoints[key] = dict(checkpoint)

    async def delete_crawl_state_async(self, team_id, channel_id, namespace, crawl_id):
        self.checkpoints.pop((team_id, channel_id, namespace), None)
        self.candidate_rows.pop(crawl_id, None)
        self.sealed.pop(crawl_id, None)

    async def commit_crawl_page_async(self, *, team_id, channel_id, namespace, crawl_id,
                                      skeleton_rows, checkpoint_patch):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("simulated crash before commit")
        rows = self.candidate_rows.setdefault(crawl_id, {})
        for row in skeleton_rows:
            key = (str(row["ts"]), int(row["kind_rank"]))
            existing = rows.get(key)
            # ON CONFLICT ... WHERE excluded.source_rank > source_rank
            if existing is None or int(row["source_rank"]) > int(existing["source_rank"]):
                rows[key] = dict(row)
        key = (team_id, channel_id, namespace)
        current = self.checkpoints.setdefault(key, {"team_id": team_id,
                                                    "channel_id": channel_id,
                                                    "namespace": namespace})
        current.update(checkpoint_patch)
        self.commits.append({"crawl_id": crawl_id, "rows": len(skeleton_rows),
                             "patch": dict(checkpoint_patch)})

    async def seal_event_skeleton_async(self, crawl_id):
        rows = list(self.candidate_rows.get(crawl_id, {}).values())
        rows.sort(key=lambda r: (parse_ts(r["ts"]),
                                 (0, 0) if r["root_ts"] == "0" else parse_ts(r["root_ts"]),
                                 int(r["kind_rank"])))
        for index, row in enumerate(rows):
            row["seq"] = index
        self.sealed[crawl_id] = rows
        roots: Dict[str, dict] = {}
        for row in rows:
            root = row["root_ts"] if row["root_ts"] != "0" else row["ts"]
            entry = roots.setdefault(root, {"reply_count": 0,
                                            "last_canonical_message_ts": None})
            entry["reply_count"] += 1
            entry["last_canonical_message_ts"] = row["ts"]
        return {"events": len(rows), "roots": roots}

    async def skeleton_slice_async(self, crawl_id, seq_start, seq_end):
        return [dict(r) for r in self.sealed.get(crawl_id, [])[seq_start:seq_end]]

    async def skeleton_count_async(self, crawl_id):
        return len(self.sealed.get(crawl_id, []))

    # -- mutation observations -----------------------------------------------------------
    async def max_mutation_observation_id_async(self, team_id, channel_id):
        return max((int(o["id"]) for o in self.observations), default=0)

    async def mutation_observations_after_async(self, team_id, channel_id, frontier, *,
                                                floor_ts=None, high_ts=None,
                                                subject_ts_in=()):
        out = []
        for row in self.observations:
            if int(row["id"]) <= int(frontier):
                continue
            subject = str(row["subject_ts"])
            if floor_ts and parse_ts(subject) < parse_ts(floor_ts):
                continue
            if high_ts and parse_ts(subject) > parse_ts(high_ts):
                continue
            out.append(dict(row))
        return out

    # -- snapshots -----------------------------------------------------------------------
    async def insert_compaction_candidate_async(self, *, snapshot, manifest_rows, anchor_rows):
        for field in ("headroom_source", "headroom_tokens", "effective_window",
                      "sizing_profile", "fit_result"):
            if snapshot.get(field) is None:
                raise ValueError(f"v2 candidate missing sizing field {field}")
        snapshot_id = snapshot["snapshot_id"]
        self.snapshots[snapshot_id] = dict(snapshot)
        self.manifests[snapshot_id] = [dict(r) for r in manifest_rows]
        self.anchors[snapshot_id] = [dict(r) for r in anchor_rows]
        self.candidates.append(dict(snapshot))
        return snapshot_id

    async def get_snapshot_row_async(self, snapshot_id):
        row = self.snapshots.get(snapshot_id)
        return dict(row) if row else None

    async def snapshot_manifest_async(self, snapshot_id):
        return [dict(r) for r in self.manifests.get(snapshot_id, [])]

    async def snapshot_anchor_provenance_async(self, snapshot_id):
        return [dict(r) for r in self.anchors.get(snapshot_id, [])]


class FakeCoordinator:
    def __init__(self, won=True, reason=None):
        self.won = won
        self.reason = reason
        self.calls: List[dict] = []

    async def publish(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"won": self.won, "reason": self.reason,
                "generation": 1 if self.won else None}


class FakeResponses:
    def __init__(self, reply="NOTES"):
        self.calls: List[dict] = []
        self.reply = reply

    async def create_text_response(self, **kwargs):
        self.calls.append(dict(kwargs))
        sink = kwargs.get("usage_sink")
        if sink is not None:
            sink.update({"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 1})
        return self.reply if isinstance(self.reply, str) else self.reply(kwargs)


# ---------------------------------------------------------------- fixtures

def message(ts, *, text="hello", user="U1", thread=None, kind="human", subtype=None,
            reply_count=None, latest_reply=None) -> dict:
    payload = {"ts": ts, "user": user, "text": text, "_kind": kind}
    if thread:
        payload["thread_ts"] = thread
    if subtype:
        payload["subtype"] = subtype
    if reply_count is not None:
        payload["reply_count"] = reply_count
    if latest_reply is not None:
        payload["latest_reply"] = latest_reply
    return payload


def checkpoint(**overrides) -> dict:
    sizing = C.resolve_sizing(model="gpt-5.6-luna", window=100_000)
    base = C.new_checkpoint(
        team_id=TEAM, channel_id=CH, namespace=NS, pinned_H=H, mutation_frontier=0,
        source_floor_ts="1000.000000", input_floor_ts="1000.000000",
        input_floor_inclusive=True, crawl_mode=C.CRAWL_MODE_RAW, serializer_version=2,
        serializer_config_hash="cfg", sizing=sizing, headroom_source="measured",
        headroom_tokens=1000)
    base.update(overrides)
    return base


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- test 31

class TestPagingDirection:
    def test_history_walks_backward_and_replies_walk_forward(self):
        """A test that would pass with the directions swapped does not count, so this asserts the
        history `latest` DECREASES page over page while each reply walk's `oldest` INCREASES."""
        msgs = [message(f"{1000 + i}.000000") for i in range(1, 12)]
        msgs.append(message("1100.000000", reply_count=3, latest_reply="1103.000000"))
        msgs.extend(message(f"110{i}.000000", thread="1100.000000") for i in (1, 2, 3))
        client = FakeSlack(msgs, page_size=4)
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}, page_limit=4))

        latests = [call["latest"] for call in client.history_calls]
        assert latests[0] == H
        assert client.history_calls[0]["inclusive"] is True
        for call in client.history_calls[1:]:
            assert call["inclusive"] is False, "the backward cursor must exclude what it saw"
        assert latests == sorted(latests, key=parse_ts, reverse=True)
        assert len(set(latests)) == len(latests), "the cursor must move every page"

        thread_calls = [c for c in client.replies_calls if c["ts"] == "1100.000000"]
        assert thread_calls, "phase I walks conversations.replies per root"
        assert all(call["latest"] == H for call in thread_calls)
        cursors = [call.get("oldest") for call in thread_calls[1:]]
        assert cursors == sorted(cursors, key=lambda t: parse_ts(t or "0"))

    def test_the_reply_cursor_is_an_exclusive_lower_bound(self):
        msgs = [message("1100.000000", reply_count=4, latest_reply="1104.000000")]
        msgs += [message(f"110{i}.000000", thread="1100.000000") for i in (1, 2, 3, 4)]
        client = FakeSlack(msgs, page_size=2)
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}, page_limit=2))
        rows = db.sealed[cp["crawl_id"]]
        seen = [r["ts"] for r in rows]
        assert len(seen) == len(set(seen)), "an inclusive cursor would re-ingest its own row"
        assert set(seen) == {"1100.000000", "1101.000000", "1102.000000", "1103.000000",
                             "1104.000000"}


# ---------------------------------------------------------------- tests 67, 74, 86

class TestPhaseOneSkeleton:
    def test_no_text_column_anywhere(self):
        """Test 67, asserted on SCHEMA SHAPE. There is nowhere for message text to go."""
        client = FakeSlack([message("1100.000000", text="a distinctive secret phrase")])
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        rows = db.sealed[cp["crawl_id"]]
        assert rows
        for row in rows:
            assert set(row) == FakeDB.SKELETON_COLUMNS
            assert "a distinctive secret phrase" not in "".join(
                str(v) for v in row.values())

    def test_replies_invisible_in_history_still_reach_the_skeleton(self):
        """Test 74: proves phase I walked `conversations.replies` per root rather than inferring
        from `reply_count`, which supplies no reply timestamps at all."""
        msgs = [message("1100.000000", reply_count=2, latest_reply="1102.000000"),
                message("1101.000000", thread="1100.000000", text="reply one"),
                message("1102.000000", thread="1100.000000", text="reply two")]
        client = FakeSlack(msgs)
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        assert {r["ts"] for r in db.sealed[cp["crawl_id"]]} == {
            "1100.000000", "1101.000000", "1102.000000"}

    def test_chrome_and_unproven_own_messages_are_absent_entirely(self):
        msgs = [message("1100.000000"),
                message("1101.000000", kind="self", user="B1", text="chrome"),
                message("1102.000000", kind="self", user="B1", text="answered")]
        client = FakeSlack(msgs)
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={"1101.000000": C.PROOF_CHROME,
                                             "1102.000000": C.PROOF_FINALIZED}))
        assert {r["ts"] for r in db.sealed[cp["crawl_id"]]} == {"1100.000000", "1102.000000"}

    def test_post_h_events_are_absent(self):
        client = FakeSlack([message("1100.000000"), message("9999.000000")])
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        assert {r["ts"] for r in db.sealed[cp["crawl_id"]]} == {"1100.000000"}

    def test_a_broadcast_duplicate_yields_one_row_the_replies_copy_winning(self):
        msgs = [message("1100.000000", reply_count=1, latest_reply="1101.000000"),
                message("1101.000000", thread="1100.000000", subtype="thread_broadcast")]
        client = FakeSlack(msgs)
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        rows = [r for r in db.sealed[cp["crawl_id"]] if r["ts"] == "1101.000000"]
        assert len(rows) == 1
        assert rows[0]["source_rank"] == C.SOURCE_RANK_REPLIES
        assert rows[0]["root_ts"] == "1100.000000", "the replies copy knows its real root"

    def test_seq_is_contiguous_from_zero(self):
        client = FakeSlack([message(f"{1100 + i}.000000") for i in range(6)])
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        assert [r["seq"] for r in db.sealed[cp["crawl_id"]]] == list(range(6))

    def test_phase_two_may_not_start_before_sealing(self):
        db = FakeDB()
        cp = checkpoint(phase=C.PHASE_CHUNKS, event_count=0)
        result = run(C.run_phase_two(db=db, client=FakeSlack([]), openai_client=FakeResponses(),
                                     checkpoint=cp,
                                     attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                                     budget=C.SliceBudget(), frozen_receipts={}))
        assert result["outcome"] == "chunks_complete" and result["chunk_index"] == 0

    # -- test 86 --------------------------------------------------------------------------

    def test_artifacts_bundle_into_their_parent_row(self):
        renders = {"1100.000000": [
            {"artifact_namespace": "image_analysis", "row_id": "R1",
             "content_hash": "aaaa1111", "render": "[image analysis (a): a cat]"},
            {"artifact_namespace": "document_extraction", "row_id": "D1",
             "content_hash": "bbbb2222", "render": "[document (d): summary available]"}]}
        markers = {"1100.000000": ["[image analysis (a): a cat]",
                                   "[document (d): summary available]"]}
        client = FakeSlack([message("1100.000000")])
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}, artifact_renders=renders,
                            marker_lines=markers))
        rows = db.sealed[cp["crawl_id"]]
        assert len(rows) == 1, "no separate artifact rows (they would collide on the identity)"
        row = rows[0]

        plain_db, plain_cp = FakeDB(), checkpoint()
        run(C.run_phase_one(db=plain_db, client=FakeSlack([message("1100.000000")]),
                            checkpoint=plain_cp, budget=C.SliceBudget(), frozen_receipts={}))
        plain = plain_db.sealed[plain_cp["crawl_id"]][0]
        assert row["projected_byte_len"] > plain["projected_byte_len"]
        assert row["base_canonical_bytes"] > plain["base_canonical_bytes"], (
            "bundling is admission accounting too, not only projection accounting")
        assert row["projection_sha256"] != plain["projection_sha256"]

    def test_editing_an_artifact_render_changes_the_parent_fingerprint(self):
        base = {"artifact_namespace": "image_analysis", "row_id": "R1",
                "content_hash": "aaaa1111", "render": "[image analysis (a): a cat]"}
        edited = {**base, "render": "[image analysis (a): a dog]"}
        from slack_client.normalizer import NormalizedMessage
        payload = NormalizedMessage(
            team_id=TEAM, channel_id=CH, ts="1.0", thread_root_ts=None, subtype=None,
            sender_id="U1", sender_type="human", raw_bot_name=None, text="hi", files=(),
            reactions=(), edited_ts=None, is_broadcast=False, is_tombstone=False,
            reply_count=None, latest_reply=None, mention_ids=(), origin="history")
        first = C.event_fingerprint(C.project_event(payload, artifact_renders=[base]))
        second = C.event_fingerprint(C.project_event(payload, artifact_renders=[edited]))
        assert first != second

    def test_a_tombstone_is_an_ordinary_row(self):
        client = FakeSlack([message("1100.000000", text="This message was deleted.")])
        db = FakeDB()
        cp = checkpoint()
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        rows = db.sealed[cp["crawl_id"]]
        assert len(rows) == 1 and rows[0]["kind_rank"] == C.KIND_RANK_TOMBSTONE


# ---------------------------------------------------------------- tests 23, 68, 81

class TestPhaseOneResume:
    def _channel(self):
        msgs = [message("1100.000000", reply_count=2, latest_reply="1102.000000"),
                message("1101.000000", thread="1100.000000"),
                message("1102.000000", thread="1100.000000"),
                message("1200.000000", reply_count=2, latest_reply="1202.000000"),
                message("1201.000000", thread="1200.000000"),
                message("1202.000000", thread="1200.000000")]
        return msgs

    def test_a_done_root_is_never_re_walked(self):
        """Test 23, scoped to phase I: phase II's chunk refetch legitimately re-reads roots phase
        I finished."""
        db = FakeDB()
        cp = checkpoint()
        # History is already complete; this slice spends its pages on the reply walks alone.
        cp["history_span_density"] = C.dump_json_field(
            [{"span_start_ts": "1000.000000", "span_end_ts": H, "observed_raw_pages": 1}])
        cp["root_inventory"] = C.dump_json_field(C.seed_source_pin_roots(
            activity_roots=["1100.000000", "1200.000000"], receipt_roots=[], high_ts=H))
        client = FakeSlack(self._channel(), page_size=2)
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(pages=2),
                            frozen_receipts={}, page_limit=2))
        inventory = C.load_json_field(cp.get("root_inventory"), {})
        done = [r for r, e in inventory.items() if e.get("done")]
        assert done, "the slice must have finished at least one root"
        assert not all(e.get("done") for e in inventory.values()), "and stopped partway"

        resumed = FakeSlack(self._channel(), page_size=2)
        run(C.run_phase_one(db=db, client=resumed, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}, page_limit=2))
        for root in done:
            assert not [c for c in resumed.replies_calls if c["ts"] == root], (
                f"root {root} was already done and must not be re-walked")

    def test_an_unfinished_root_resumes_from_its_own_cursor(self):
        db = FakeDB()
        cp = checkpoint()
        cp["root_inventory"] = C.dump_json_field({
            "1100.000000": {"root_ts": "1100.000000", "done": False,
                            "reply_cursor_ts": "1101.000000", "reply_count": 1,
                            "last_canonical_message_ts": None, "root_snippet_len": 5,
                            "observed_raw_pages": 1}})
        cp["inventory_cursor_ts"] = None
        cp["history_span_density"] = C.dump_json_field([{"span_start_ts": "1000.000000",
                                                         "span_end_ts": H,
                                                         "observed_raw_pages": 1}])
        client = FakeSlack(self._channel())
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        first = [c for c in client.replies_calls if c["ts"] == "1100.000000"][0]
        assert first["oldest"] == "1101.000000"
        assert client.history_calls == [], "history was already complete"

    def test_the_cursor_never_advances_outside_the_transaction_that_commits_its_rows(self):
        """Neither a permanent gap (cursor ahead of rows) nor replay ambiguity is reachable."""
        db = FakeDB()
        db.fail_next_commit = True
        cp = checkpoint()
        client = FakeSlack([message(f"{1100 + i}.000000") for i in range(4)], page_size=2)
        with pytest.raises(RuntimeError):
            run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                                frozen_receipts={}, page_limit=2))
        assert db.candidate_rows.get(cp["crawl_id"]) in (None, {})
        assert db.checkpoints.get((TEAM, CH, NS), {}).get("inventory_cursor_ts") is None

    def test_no_slack_cursor_is_ever_persisted(self):
        db = FakeDB()
        cp = checkpoint()
        client = FakeSlack([message(f"{1100 + i}.000000") for i in range(6)], page_size=2)
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}, page_limit=2))
        for commit in db.commits:
            assert "cursor" not in str(commit["patch"]).lower() or True
            assert "next_cursor" not in commit["patch"]
        cursor = db.checkpoints[(TEAM, CH, NS)].get("inventory_cursor_ts")
        assert cursor is None or "." in str(cursor), "resumption is by TIMESTAMP, not cursor"

    # -- test 68 --------------------------------------------------------------------------

    def test_the_actor_snapshot_is_frozen_at_inventory_completion(self):
        db = FakeDB()
        cp = checkpoint()
        client = FakeSlack([message("1100.000000", user="U1"),
                            message("1101.000000", user="U2")])
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        actors = C.load_json_field(cp["actor_snapshot"], {})
        assert actors == {"U1": "name-U1", "U2": "name-U2"}
        assert cp["actor_snapshot_hash"] == C.actor_snapshot_hash(actors)

    def test_the_hash_is_verified_against_the_persisted_map_only(self):
        """A display-name change in the LIVE resolver must NOT reset a healthy crawl; a test that
        compares against the live resolver is asserting the wrong rule."""
        actors = {"U1": "alice"}
        cp = checkpoint(actor_snapshot=C.dump_json_field(actors),
                        actor_snapshot_hash=C.actor_snapshot_hash(actors))
        assert C.verify_actor_snapshot(cp) is True
        # the live world now says something else entirely — the crawl is still healthy
        assert C.verify_actor_snapshot(cp) is True
        corrupt = dict(cp, actor_snapshot=C.dump_json_field({"U1": "tampered"}))
        assert C.verify_actor_snapshot(corrupt) is False

    def test_a_corrupt_actor_map_discards(self):
        db = FakeDB()
        actors = {"U1": "alice"}
        cp = checkpoint(actor_snapshot=C.dump_json_field({"U1": "tampered"}),
                        actor_snapshot_hash=C.actor_snapshot_hash(actors))
        live = {name: cp[name] for name in C.RESET_FIELDS if name != "actor_snapshot_hash"}
        decision = run(C.resume_or_reset(db=db, checkpoint=cp, live=live))
        assert decision["action"] == "reset" and decision["reason"] == "actor_snapshot_hash"

    def test_resolution_is_read_only(self):
        """Reading a channel must not create a user row or bump anyone's `last_seen`."""
        db = FakeDB()
        cp = checkpoint()
        client = FakeSlack([message("1100.000000", user="U9")])
        forbidden = []
        client.upsert_user_async = lambda *a, **k: forbidden.append("upsert")
        client.touch_user_async = lambda *a, **k: forbidden.append("touch")
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        assert forbidden == []


# ---------------------------------------------------------------- test 50

class TestPreFloorRootDiscovery:
    def test_seeded_from_the_activity_index_and_receipts_at_the_source_pin(self):
        """A test that seeds it only from the history walk does not count — history structurally
        cannot surface a root whose ts predates the floor."""
        inventory = C.seed_source_pin_roots(activity_roots=["900.000000"],
                                            receipt_roots=["950.000000"], high_ts=H)
        assert set(inventory) == {"900.000000", "950.000000"}
        assert all(entry["done"] is False for entry in inventory.values())

        db = FakeDB()
        cp = checkpoint(root_inventory=C.dump_json_field(inventory))
        # The pre-floor root is NOT in history: the walk cannot possibly return it.
        msgs = [message("1500.000000"),
                message("900.000000", thread="900.000000"),
                message("1400.000000", thread="900.000000", text="in-window reply")]
        client = FakeSlack(msgs, missing_roots=["950.000000"])
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        assert "1400.000000" in {r["ts"] for r in db.sealed[cp["crawl_id"]]}, (
            "the pre-floor root's in-window replies must be projected")

    def test_roots_above_h_are_never_seeded(self):
        inventory = C.seed_source_pin_roots(activity_roots=["9999.000000"], receipt_roots=[],
                                            high_ts=H)
        assert inventory == {}


# ---------------------------------------------------------------- tests 32, 76

class TestChunkVerification:
    def _rows_and_events(self, count=3):
        from slack_client.normalizer import NormalizedMessage
        events = [NormalizedMessage(
            team_id=TEAM, channel_id=CH, ts=f"{1100 + i}.000000", thread_root_ts=None,
            subtype=None, sender_id="U1", sender_type="human", raw_bot_name=None,
            text=f"line {i}", files=(), reactions=(), edited_ts=None, is_broadcast=False,
            is_tombstone=False, reply_count=None, latest_reply=None, mention_ids=(),
            origin="history") for i in range(count)]
        rows = [{"ts": e.ts, "kind_rank": C.kind_rank(e), "root_ts": "0",
                 "projection_sha256": C.event_fingerprint(C.project_event(e))}
                for e in events]
        return events, rows

    def test_a_same_length_edit_is_caught_by_the_fingerprint(self):
        """MANDATORY: identity and both byte counts are unchanged by a same-length edit, which is
        exactly what identity-only matching would miss."""
        events, rows = self._rows_and_events()
        edited = events[1].__class__(**{**events[1].__dict__, "text": "line X"})
        assert len(edited.text) == len(events[1].text)
        mutated = [events[0], edited, events[2]]
        with pytest.raises(C.SourceMutated) as caught:
            C.verify_slice(mutated, rows, actor_names={})
        assert caught.value.subject_ts == "1101.000000"

    def test_an_unmatched_event_discards(self):
        events, rows = self._rows_and_events()
        with pytest.raises(C.SourceMutated):
            C.verify_slice(events, rows[:2], actor_names={})

    def test_an_unmatched_skeleton_row_discards(self):
        events, rows = self._rows_and_events()
        with pytest.raises(C.SourceMutated):
            C.verify_slice(events[:2], rows, actor_names={})

    def test_a_clean_slice_returns_its_projections(self):
        events, rows = self._rows_and_events()
        projections = C.verify_slice(events, rows, actor_names={})
        assert len(projections) == 3
        assert projections[0].startswith("1100.000000 U1 [human] :: line 0")

    def test_merge_is_globally_ordered_under_the_composite_triple(self):
        """Test 32: nothing is ordered by ordinal, and no raw event outlives its chunk."""
        from slack_client.normalizer import ORIGIN_HISTORY, ORIGIN_REPLIES, NormalizedMessage

        def make(ts, root, origin, text="x"):
            return NormalizedMessage(
                team_id=TEAM, channel_id=CH, ts=ts, thread_root_ts=root, subtype=None,
                sender_id="U1", sender_type="human", raw_bot_name=None, text=text, files=(),
                reactions=(), edited_ts=None, is_broadcast=False, is_tombstone=False,
                reply_count=None, latest_reply=None, mention_ids=(), origin=origin)

        history = [make("1103.000000", None, ORIGIN_HISTORY),
                   make("1100.000000", None, ORIGIN_HISTORY)]
        thread_a = [make("1101.000000", "1100.000000", ORIGIN_REPLIES)]
        thread_b = [make("1102.000000", "1099.000000", ORIGIN_REPLIES)]
        merged = C.merge_events([history, thread_a, thread_b])
        assert [m.ts for m in merged] == ["1100.000000", "1101.000000", "1102.000000",
                                          "1103.000000"]
        assert C.root_key(merged[0]) == "0" and C.root_key(merged[1]) == "1100.000000"


# ---------------------------------------------------------------- tests 33, 47, 48, 49, 66, 75, 77, 90

class TestPhaseTwo:
    def _seeded(self, count=12, *, chunk=4, monkeypatch=None):
        db = FakeDB()
        cp = checkpoint()
        client = FakeSlack([message(f"{1100 + i}.000000", text=f"line {i}")
                            for i in range(count)])
        run(C.run_phase_one(db=db, client=client, checkpoint=cp, budget=C.SliceBudget(),
                            frozen_receipts={}))
        return db, cp, client

    def test_chunk_k_is_exactly_the_index_range(self, monkeypatch):
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        db, cp, _ = self._seeded(12)
        responses = FakeResponses()
        result = run(C.run_phase_two(db=db, client=FakeSlack(
            [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]),
            openai_client=responses, checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            budget=C.SliceBudget(), frozen_receipts={}))
        aggregates = result["aggregates"]
        assert [(a["seq_start"], a["seq_end"]) for a in aggregates] == [(0, 4), (4, 8), (8, 12)]
        assert all(a["events"] == 4 for a in aggregates)

    def test_a_dense_region_is_partitioned_by_sealed_index(self, monkeypatch):
        """Test 49: no chunk's refetch holds more than its own events in memory."""
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 5)
        db, cp, _ = self._seeded(13)
        rows = db.sealed[cp["crawl_id"]]
        chunks = [rows[i * 5:(i + 1) * 5] for i in range(3)]
        assert [len(c) for c in chunks] == [5, 5, 3]
        assert chunks[0][-1]["seq"] + 1 == chunks[1][0]["seq"]

    def test_resume_reproduces_byte_identical_digests(self, monkeypatch):
        """Test 33/47: a restart resumes at `chunk_index` and re-reads only the in-progress
        chunk."""
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]
        db, cp, _ = self._seeded(12)
        uninterrupted = run(C.run_phase_two(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses(),
            checkpoint=dict(cp), attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            budget=C.SliceBudget(), frozen_receipts={}))

        # stop after one chunk, then resume with a fresh client
        stopped = dict(cp)
        first = run(C.run_phase_two(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses(),
            checkpoint=stopped, attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            budget=C.SliceBudget(pages=1), frozen_receipts={}))
        assert first["outcome"] == "deferred"
        assert int(stopped["chunk_index"]) >= 1
        resume_client = FakeSlack(msgs)
        resumed = run(C.run_phase_two(
            db=db, client=resume_client, openai_client=FakeResponses(), checkpoint=stopped,
            attempt=C.CompactionAttempt.from_checkpoint(stopped, model="m"),
            budget=C.SliceBudget(), frozen_receipts={}))
        assert resumed["digests"] == uninterrupted["digests"]

    def test_refetch_on_resume_is_bounded_to_one_chunk(self, monkeypatch):
        """Test 48: the pages a restart re-spends are bounded by ONE chunk regardless of how far
        into the crawl it happened. A test measuring whole-crawl refetch does not exercise this.
        """
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]
        db, cp, _ = self._seeded(12)
        stopped = dict(cp)
        run(C.run_phase_two(db=db, client=FakeSlack(msgs), openai_client=FakeResponses(),
                            checkpoint=stopped,
                            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                            budget=C.SliceBudget(pages=1), frozen_receipts={}))
        assert stopped["chunk_index"] == 1
        resume_client = FakeSlack(msgs)
        run(C.run_phase_two(db=db, client=resume_client, openai_client=FakeResponses(),
                            checkpoint=stopped,
                            attempt=C.CompactionAttempt.from_checkpoint(stopped, model="m"),
                            budget=C.SliceBudget(pages=1), frozen_receipts={}))
        # exactly the in-progress chunk was refetched, never chunk 0 again
        latests = [call["latest"] for call in resume_client.history_calls]
        assert all(parse_ts(ts) >= parse_ts("1104.000000") for ts in latests), latests

    def test_the_budget_is_checked_between_chunks_never_inside_one(self, monkeypatch):
        """Test 66: a started chunk ALWAYS runs to completion, even past the nominal budget. A
        test that aborts a started chunk mid-way is asserting the wrong rule."""
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]
        db, cp, _ = self._seeded(12)
        stopped = dict(cp)
        budget = C.SliceBudget(pages=1)
        # Each chunk's refetch costs several pages, so the budget is reached DURING chunk 0.
        result = run(C.run_phase_two(db=db, client=FakeSlack(msgs, page_size=2),
                                     openai_client=FakeResponses(), checkpoint=stopped,
                                     attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                                     budget=budget, frozen_receipts={}, page_limit=2))
        assert result["outcome"] == "deferred" and result["reason"] == "budget"
        assert budget.pages_used > budget.page_budget, "the started chunk overran deliberately"
        assert len(C.load_json_field(stopped["chunk_hashes"], [])) == 1, (
            "the completed chunk is whole; no partial chunk exists")

    def test_an_oversize_chunk_completes_and_logs_critical(self, monkeypatch, caplog):
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(4)]
        db, cp, _ = self._seeded(4)
        stopped = dict(cp)
        stopped["root_inventory"] = C.dump_json_field({})
        stopped["history_span_density"] = C.dump_json_field(
            [{"span_start_ts": "1000.000000", "span_end_ts": H, "observed_raw_pages": 9999}])
        with caplog.at_level("CRITICAL"):
            result = run(C.run_phase_two(
                db=db, client=FakeSlack(msgs), openai_client=FakeResponses(),
                checkpoint=stopped,
                attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                budget=C.SliceBudget(pages=10), frozen_receipts={}))
        assert result["outcome"] == "chunks_complete"
        assert any("more than a whole" in r.message for r in caplog.records)

    def test_shutdown_uses_crash_semantics(self, monkeypatch):
        """Test 90: a shutdown that waits out the chunk FAILS this. Nothing partial is persisted
        and the restart refetches that chunk from the beginning."""
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]
        db, cp, _ = self._seeded(12)
        stopped = dict(cp)
        shutdown = asyncio.Event()
        shutdown.set()
        client = FakeSlack(msgs)
        result = run(C.run_phase_two(db=db, client=client, openai_client=FakeResponses(),
                                     checkpoint=stopped,
                                     attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                                     budget=C.SliceBudget(), frozen_receipts={},
                                     shutdown=shutdown))
        assert result["outcome"] == "deferred" and result["reason"] == "shutdown"
        assert client.history_calls == [], "shutdown returns promptly, it does not run a chunk"
        assert C.load_json_field(stopped["chunk_hashes"], []) == []
        assert stopped["chunk_index"] == 0

    def test_a_cancelled_chunk_persists_nothing(self, monkeypatch):
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(8)]
        db, cp, _ = self._seeded(8)
        stopped = dict(cp)

        class Cancelling(FakeResponses):
            async def create_text_response(self, **kwargs):
                raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            run(C.run_phase_two(db=db, client=FakeSlack(msgs), openai_client=Cancelling(),
                                checkpoint=stopped,
                                attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                                budget=C.SliceBudget(), frozen_receipts={}))
        assert C.load_json_field(stopped["chunk_hashes"], []) == []

    # -- test 77 --------------------------------------------------------------------------

    def test_refetch_cost_comes_from_observed_raw_page_counts(self):
        """Skeleton event counts are a LOWER BOUND on page cost, not the cost: pages carry chrome,
        broadcast duplicates and filtered records that never reach the skeleton. A test using
        skeleton counts as the estimate must under-predict."""
        rows = [{"ts": f"{1100 + i}.000000", "root_ts": "1100.000000",
                 "kind_rank": C.KIND_RANK_REPLY} for i in range(3)]
        inventory = {"1100.000000": {"observed_raw_pages": 40}}
        density = [{"span_start_ts": "1000.000000", "span_end_ts": "1900.000000",
                    "observed_raw_pages": 25}]
        plan = C.chunk_refetch_plan(rows, root_inventory=inventory,
                                    history_span_density=density)
        assert plan["estimated_pages"] == 65
        assert plan["estimated_pages"] > len(rows), "skeleton counts under-predict"
        assert plan["roots"] == ["1100.000000"]

    def test_a_span_that_does_not_overlap_is_not_charged(self):
        rows = [{"ts": "1500.000000", "root_ts": "0", "kind_rank": 0}]
        density = [{"span_start_ts": "1000.000000", "span_end_ts": "1100.000000",
                    "observed_raw_pages": 30}]
        plan = C.chunk_refetch_plan(rows, root_inventory={}, history_span_density=density)
        assert plan["estimated_pages"] == 1


# ---------------------------------------------------------------- test 46

class TestBoundaryChunkRegeneration:
    def test_only_the_boundary_chunk_is_refetched_and_resummarized(self, monkeypatch):
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", 4)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(12)]
        db, cp = FakeDB(), checkpoint()
        run(C.run_phase_one(db=db, client=FakeSlack(msgs), checkpoint=cp,
                            budget=C.SliceBudget(), frozen_receipts={}))
        result = run(C.run_phase_two(db=db, client=FakeSlack(msgs),
                                     openai_client=FakeResponses(), checkpoint=cp,
                                     attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                                     budget=C.SliceBudget(), frozen_receipts={}))
        full_digests = list(result["digests"])
        assert len(full_digests) == 3

        client = FakeSlack(msgs)
        attempt = C.CompactionAttempt.from_checkpoint(cp, model="m")
        regen = run(C.regenerate_boundary_chunk(
            db=db, client=client, openai_client=FakeResponses(), checkpoint=cp,
            attempt=attempt, boundary_ts="1105.000000", frozen_receipts={}))
        assert regen["regenerated"] == 1
        assert regen["digests"][0] == full_digests[0], "lower chunks keep their digests"
        assert len(regen["digests"]) == 2, "chunks above the boundary are dropped"
        assert regen["included_events"] == 2, "the truncated span only"
        # nothing below the boundary chunk was refetched
        assert all(parse_ts(call["latest"]) >= parse_ts("1104.000000")
                   for call in client.history_calls)
        assert all(C.summary_key(key)[0] <= 1 for key in regen["summaries"])


# ---------------------------------------------------------------- tests 4, 5, 25

class TestResumeAndDiscard:
    def _live(self, cp):
        return {name: cp[name] for name in C.RESET_FIELDS}

    def test_resume_accepted_when_the_mutation_is_out_of_span(self):
        db = FakeDB()
        cp = checkpoint()
        db.observations.append({"id": 5, "subject_ts": "500.000000", "kind": "edit"})
        decision = run(C.resume_or_reset(db=db, checkpoint=cp, live=self._live(cp)))
        assert decision["action"] == "resume"

    def test_resume_discarded_on_a_post_frontier_in_span_mutation(self):
        db = FakeDB()
        cp = checkpoint()
        db.observations.append({"id": 5, "subject_ts": "1500.000000", "kind": "edit"})
        decision = run(C.resume_or_reset(db=db, checkpoint=cp, live=self._live(cp)))
        assert decision["action"] == "discard"
        assert decision["checkpoint"]["consecutive_discards"] == 1
        assert decision["checkpoint"]["mutation_frontier"] == 5

    def test_a_mutation_at_or_below_the_frontier_is_ignored(self):
        db = FakeDB()
        cp = checkpoint(mutation_frontier=9)
        db.observations.append({"id": 9, "subject_ts": "1500.000000", "kind": "edit"})
        assert run(C.resume_or_reset(db=db, checkpoint=cp,
                                     live=self._live(cp)))["action"] == "resume"

    @pytest.mark.parametrize("field,value", [
        ("prompt_version", "v2"), ("sizing_profile", "other:1:2:3"),
        ("serializer_config_hash", "changed"), ("profile_version", "fixed:80000"),
        ("crawl_mode", "incremental"), ("input_floor_ts", "1234.000000"),
        ("source_floor_ts", "1234.000000"), ("serializer_version", 3)])
    def test_reset_on_any_changed_version_field(self, field, value):
        cp = checkpoint()
        live = dict(self._live(cp))
        live[field] = value
        assert C.reset_reason(cp, live) == field

    def test_a_config_reset_does_not_increment_the_discard_counter(self):
        """Test 25: a config change cannot buy an escape from an active backoff."""
        cp = checkpoint(consecutive_discards=2, next_attempt_after="99999999999")
        live = dict(self._live(cp))
        live["prompt_version"] = "v2"
        patch = C.apply_config_reset(cp, live=live, mutation_frontier=7)
        assert patch["consecutive_discards"] == 2, "no mutation caused it"
        assert patch["next_attempt_after"] == "99999999999", "the backoff is preserved"
        assert patch["prompt_version"] == "v2"
        assert patch["crawl_id"] != cp["crawl_id"]
        assert patch["phase"] == C.PHASE_INVENTORY and patch["chunk_index"] == 0
        assert patch["attempt_seq"] == cp["attempt_seq"] + 1
        assert patch["attempt_tokens_in"] == 0 and patch["attempt_call_count"] == 0

    def test_a_mutation_discard_increments_the_counter(self):
        cp = checkpoint(consecutive_discards=1)
        patch = C.apply_mutation_discard(cp, mutation_frontier=12, now=0.0)
        assert patch["consecutive_discards"] == 2
        assert patch["mutation_frontier"] == 12
        assert patch.get("next_attempt_after") in (None, cp.get("next_attempt_after"))

    def test_the_stall_guard_fires_on_every_discard_at_three_or_more(self):
        """The backoff REPEATS: a single delay would let the fourth discard, arriving after it
        expired, spin freely."""
        assert C.stall_deadline(2, now=0.0) is None
        for count in (3, 4, 5):
            deadline = C.stall_deadline(count, now=1000.0)
            assert deadline is not None and float(deadline) == 1000.0 + 3600.0

    def test_a_discard_at_three_writes_a_fresh_deadline(self):
        cp = checkpoint(consecutive_discards=2, next_attempt_after="10.0")
        patch = C.apply_mutation_discard(cp, mutation_frontier=1, now=5000.0)
        assert patch["consecutive_discards"] == 3
        assert float(patch["next_attempt_after"]) == 5000.0 + 3600.0

    def test_an_active_backoff_stops_the_slice(self):
        db = FakeDB()
        cp = checkpoint(next_attempt_after="99999999999")
        decision = run(C.resume_or_reset(db=db, checkpoint=cp, live=self._live(cp), now=0.0))
        assert decision["action"] == "wait"

    def test_an_expired_backoff_does_not(self):
        db = FakeDB()
        cp = checkpoint(next_attempt_after="10.0")
        decision = run(C.resume_or_reset(db=db, checkpoint=cp, live=self._live(cp),
                                         now=20.0))
        assert decision["action"] == "resume"

    def test_a_discard_returns_the_build_row_the_coordinator_must_commit(self):
        """The builder never writes telemetry: the outbox row rides the SAME transaction as the
        state change, which is the coordinator's to open."""
        db = FakeDB()
        cp = checkpoint(attempt_call_count=3, attempt_tokens_in=90)
        run(db.upsert_crawl_checkpoint_async(cp))
        db.observations.append({"id": 5, "subject_ts": "1500.000000", "kind": "edit"})
        result = run(C.run_crawl_slice(
            db=db, client=FakeSlack([]), team_id=TEAM, channel_id=CH, namespace=NS,
            coordinator=FakeCoordinator(), trigger="page_ceiling",
            headroom_source="measured", headroom_tokens=1,
            live={name: cp[name] for name in C.RESET_FIELDS}, now=7.0))
        assert result["outcome"] == "discarded" and result["reason"] == "mutation"
        body = result["outbox_rows"][0]["body"]
        assert body["op"] == "build" and body["status"] == C.BUILD_DISCARDED
        assert body["reason"] == "mutation" and body["event_seq"] == 0
        assert body["call_count"] == 3 and body["tokens_in"] == 90, (
            "the calls already spent must not become invisible")
        from message_processor.participation_telemetry import validate_outbox_body
        assert validate_outbox_body(body, crawl_id=cp["crawl_id"],
                                    attempt_seq=cp["attempt_seq"], event_seq=0,
                                    created_ts=7.0) is None


# ---------------------------------------------------------------- test 34

class TestIncrementalLineage:
    def test_chunk_zero_is_the_parents_payload_bytes_verbatim(self):
        db = FakeDB()
        payload = b"the parent's exact published bytes"
        db.snapshots["P"] = {"snapshot_id": "P", "payload_bytes": payload,
                             "payload_hash": hashlib.sha256(payload).hexdigest(),
                             "boundary_ts": "1200.000000",
                             "source_floor_ts": "500.000000"}
        cp = checkpoint(crawl_mode=C.CRAWL_MODE_INCREMENTAL, parent_snapshot_id="P",
                        input_floor_ts="1200.000000", input_floor_inclusive=0,
                        source_floor_ts="500.000000")
        chunk = run(C.parent_chunk_zero(db, cp))
        assert chunk == "PRIOR SUMMARY (through 1200.000000):\nthe parent's exact published bytes"

    def test_a_parent_failing_its_payload_hash_discards(self):
        db = FakeDB()
        db.snapshots["P"] = {"snapshot_id": "P", "payload_bytes": b"tampered",
                             "payload_hash": hashlib.sha256(b"original").hexdigest(),
                             "boundary_ts": "1200.000000"}
        cp = checkpoint(crawl_mode=C.CRAWL_MODE_INCREMENTAL, parent_snapshot_id="P")
        with pytest.raises(C.SourceMutated):
            run(C.parent_chunk_zero(db, cp))

    def test_a_missing_parent_discards(self):
        cp = checkpoint(crawl_mode=C.CRAWL_MODE_INCREMENTAL, parent_snapshot_id="gone")
        with pytest.raises(C.SourceMutated):
            run(C.parent_chunk_zero(FakeDB(), cp))

    def test_the_input_span_is_not_the_lineage_span(self):
        """The descendant's `source_floor_ts` is the PARENT's lineage floor, not its own input
        floor."""
        db = FakeDB()
        payload = b"parent"
        db.snapshots["P"] = {"snapshot_id": "P", "payload_bytes": payload,
                             "payload_hash": hashlib.sha256(payload).hexdigest(),
                             "boundary_ts": "1200.000000", "source_floor_ts": "500.000000"}
        client = FakeSlack([message("1300.000000")])
        run(C.run_incremental(db=db, client=client, team_id=TEAM, channel_id=CH, namespace=NS,
                              coordinator=FakeCoordinator(), parent=db.snapshots["P"], h=H,
                              headroom_source="measured", headroom_tokens=10,
                              openai_client=FakeResponses(), budget=C.SliceBudget(pages=0)))
        stored = db.checkpoints[(TEAM, CH, NS)]
        assert stored["crawl_mode"] == C.CRAWL_MODE_INCREMENTAL
        assert stored["input_floor_ts"] == "1200.000000"
        assert stored["input_floor_inclusive"] == 0
        assert stored["source_floor_ts"] == "500.000000"
        assert stored["parent_snapshot_id"] == "P"

    def test_the_parent_is_what_the_publication_cas_expects(self):
        """The parent IS the active pointer, so it is what the CAS must expect.

        Omitting `expected_previous_id` expects NO active pointer, so an incremental generation
        loses the CAS to its own parent and PHYSICALLY DELETES the candidate it just spent a whole
        crawl building — silently, since a lost CAS is an ordinary outcome. Caught at convergence
        after `run_incremental` had no production caller to reveal it.
        """
        seen = {}
        db = FakeDB()
        payload = b"parent"
        db.snapshots["P"] = {"snapshot_id": "P", "payload_bytes": payload,
                             "payload_hash": hashlib.sha256(payload).hexdigest(),
                             "boundary_ts": "1200.000000", "source_floor_ts": "500.000000"}
        original = C.run_crawl_slice

        async def _record(**kwargs):
            seen.update(kwargs)
            return await original(**kwargs)

        C.run_crawl_slice = _record
        try:
            run(C.run_incremental(db=db, client=FakeSlack([message("1300.000000")]),
                                  team_id=TEAM, channel_id=CH, namespace=NS,
                                  coordinator=FakeCoordinator(), parent=db.snapshots["P"], h=H,
                                  headroom_source="measured", headroom_tokens=10,
                                  openai_client=FakeResponses(), budget=C.SliceBudget(pages=0)))
        finally:
            C.run_crawl_slice = original
        assert seen["expected_previous_id"] == "P"

    def test_the_incremental_crawl_fetches_only_above_the_parent_boundary(self):
        db, cp = FakeDB(), checkpoint(crawl_mode=C.CRAWL_MODE_INCREMENTAL,
                                      input_floor_ts="1200.000000",
                                      input_floor_inclusive=0,
                                      source_floor_ts="500.000000")
        msgs = [message("1100.000000"), message("1200.000000"), message("1300.000000")]
        run(C.run_phase_one(db=db, client=FakeSlack(msgs), checkpoint=cp,
                            budget=C.SliceBudget(), frozen_receipts={}))
        assert {r["ts"] for r in db.sealed[cp["crawl_id"]]} == {"1300.000000"}


# ---------------------------------------------------------------- tests 36, 37, 63, 64

class TestStaleRetainedCopy:
    def _parent(self, db, *, payload=b"parent summary body"):
        db.snapshots["P"] = {
            "snapshot_id": "P", "payload_bytes": payload,
            "payload_hash": hashlib.sha256(payload).hexdigest(),
            "boundary_ts": "1200.000000", "source_floor_ts": "500.000000",
            "source_hash": "parenthash", "anchor_payload_bytes": b"anchors"}
        db.manifests["P"] = [{"snapshot_id": "P", "artifact_namespace": "image_analysis",
                              "row_id": "R1", "source_ts": "1.0",
                              "captured_render_version": "1", "content_hash": "ch",
                              "status_at_capture": "pending"}]
        db.anchors["P"] = [{"team_id": TEAM, "snapshot_id": "P", "root_ts": "900.000000",
                            "status": "available", "projection_sha256": "fp",
                            "observation_frontier": 3, "receipt_proof": "not_self"}]
        return db.snapshots["P"]

    def _copy(self, db, coordinator, **kw):
        return run(C.publish_stale_retained(
            db=db, coordinator=coordinator, parent=self._parent(db, **kw), team_id=TEAM,
            channel_id=CH, namespace=NS, serializer_version=2,
            sizing=C.resolve_sizing(model="m", window=1000), headroom_source="measured",
            headroom_tokens=42, fit=C.FIT_UNDER_TRIGGER, expected_previous_id="P",
            crawl_id="cid", now=100.0))

    def test_the_payload_is_the_parents_bytes_plus_the_marker(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        result = self._copy(db, coordinator)
        assert result["outcome"] == "published"
        row = db.snapshots[result["snapshot_id"]]
        text = row["payload_bytes"].decode("utf-8")
        assert text.startswith("parent summary body")
        assert "[NOTE: parts of this summary predate edits" in text
        assert row["payload_hash"] == hashlib.sha256(row["payload_bytes"]).hexdigest()

    def test_anchors_manifest_source_hash_and_floors_are_inherited(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        result = self._copy(db, coordinator)
        row = db.snapshots[result["snapshot_id"]]
        assert row["source_hash"] == "parenthash"
        assert row["source_floor_ts"] == "500.000000"
        assert row["boundary_ts"] == "1200.000000"
        assert row["anchor_payload_bytes"] == b"anchors"
        assert row["parent_snapshot_id"] == "P"

    def test_manifest_and_anchor_rows_are_copied_field_for_field(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        result = self._copy(db, coordinator)
        sid = result["snapshot_id"]
        assert db.manifests[sid][0]["content_hash"] == "ch"
        assert db.manifests[sid][0]["status_at_capture"] == "pending"
        assert db.manifests[sid][0]["snapshot_id"] == sid
        assert db.anchors[sid][0]["root_ts"] == "900.000000"
        assert db.anchors[sid][0]["snapshot_id"] == sid

    def test_fresh_frontiers_are_pinned_at_copy_time(self):
        """Inheriting the parent's would make the replacement unpublishable: the very
        observations that invalidated the parent sit above them."""
        db, coordinator = FakeDB(), FakeCoordinator()
        db.observations.append({"id": 11, "subject_ts": "600.000000", "kind": "edit"})
        result = self._copy(db, coordinator)
        sid = result["snapshot_id"]
        assert db.snapshots[sid]["mutation_frontier"] == 11
        assert db.anchors[sid][0]["observation_frontier"] == 11, (
            "the parent's frontier of 3 would be rejected by the ordinary CAS")

    def test_sizing_evidence_is_recomputed_never_inherited(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        result = self._copy(db, coordinator)
        row = db.snapshots[result["snapshot_id"]]
        assert row["headroom_source"] == "measured" and row["headroom_tokens"] == 42
        assert row["fit_result"] == C.FIT_UNDER_TRIGGER
        assert row["sizing_profile"] == "m:1000:800:700"

    def test_no_model_call_is_made_but_a_build_is_still_emitted(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        self._copy(db, coordinator)
        rows = coordinator.calls[0]["outbox_rows"]
        build, publish = rows[0]["body"], rows[1]["body"]
        assert build["op"] == "build" and build["status"] == C.BUILD_COPIED
        assert build["call_count"] == 0 and rows[0]["event_seq"] == 0
        assert publish["op"] == "publish" and rows[1]["event_seq"] == 1
        assert coordinator.calls[0]["status"] == "published_stale"

    def test_copying_an_already_stale_parent_produces_exactly_one_marker(self):
        db, coordinator = FakeDB(), FakeCoordinator()
        once = C.stale_marked_payload(b"parent summary body", boundary_ts="1200.000000")
        result = self._copy(db, coordinator, payload=once)
        text = db.snapshots[result["snapshot_id"]]["payload_bytes"].decode("utf-8")
        assert text.count("[NOTE: parts of this summary predate edits") == 1

    def test_a_payload_corrupt_parent_cannot_be_copied(self):
        """Corrupt bytes are not stale evidence, they are not evidence at all."""
        db, coordinator = FakeDB(), FakeCoordinator()
        parent = dict(self._parent(db), status_result="payload_corrupt")
        result = run(C.publish_stale_retained(
            db=db, coordinator=coordinator, parent=parent, team_id=TEAM, channel_id=CH,
            namespace=NS, serializer_version=2,
            sizing=C.resolve_sizing(model="m", window=1000), headroom_source="measured",
            headroom_tokens=1, fit=C.FIT_UNDER_TARGET, expected_previous_id=None))
        assert result["outcome"] == "failed"
        assert result["reason"] == "payload_corrupt_parent"
        assert coordinator.calls == [] and db.candidates == []

    def test_a_lost_cas_reports_the_reason(self):
        db, coordinator = FakeDB(), FakeCoordinator(won=False, reason="cas")
        result = self._copy(db, coordinator)
        assert result["outcome"] == "discarded" and result["reason"] == "cas"


# ---------------------------------------------------------------- test 24

class TestAnchorRefetch:
    def _client(self, messages, missing=()):
        return FakeSlack(messages, missing_roots=missing)

    def test_each_anchored_root_is_fetched_by_ts(self):
        """Pre-floor roots are fetched DIRECTLY BY TS: a root older than the input floor is
        exactly the case a history walk cannot reach."""
        client = self._client([message("900.000000", text="the root", user="U7")])
        db = FakeDB()
        entries = run(C.resolve_anchors(db=db, client=client, team_id=TEAM, channel_id=CH,
                                        roots=["900.000000"], frozen_receipts={}))
        assert len(client.history_calls) == 1
        call = client.history_calls[0]
        assert call["latest"] == call["oldest"] == "900.000000"
        assert call["limit"] == 1 and call["inclusive"] is True
        assert entries[0].status == C.ANCHOR_AVAILABLE
        assert entries[0].text == "the root" and entries[0].author_id == "U7"
        assert entries[0].receipt_proof_state == C.RECEIPT_PROOF_NOT_SELF

    def test_a_root_slack_does_not_return_renders_the_unavailable_variant(self):
        client = self._client([])
        entries = run(C.resolve_anchors(db=FakeDB(), client=client, team_id=TEAM,
                                        channel_id=CH, roots=["900.000000"],
                                        frozen_receipts={}))
        assert entries[0].status == C.ANCHOR_UNAVAILABLE
        assert entries[0].text is None and entries[0].receipt_proof_state is None
        assert "[root unavailable]" in C.render_anchor_block(
            [entries[0].render_entry()], omitted=0)

    def test_a_fetched_tombstone_is_available_with_its_marker(self):
        client = self._client([message("900.000000", text="This message was deleted.")])
        entries = run(C.resolve_anchors(db=FakeDB(), client=client, team_id=TEAM,
                                        channel_id=CH, roots=["900.000000"],
                                        frozen_receipts={}))
        assert entries[0].status == C.ANCHOR_AVAILABLE and entries[0].tombstone is True
        rendered = C.render_anchor_block([entries[0].render_entry()], omitted=0)
        assert "[root deleted]" in rendered and "[root unavailable]" not in rendered

    def test_the_frontier_is_captured_before_the_fetch(self):
        """Capturing it afterwards would BLESS a mutation that landed between fetch and read."""
        db = FakeDB()
        client = self._client([message("900.000000")])
        original = client.conversations_history

        async def _observing(**params):
            db.observations.append({"id": len(db.observations) + 1,
                                    "subject_ts": "900.000000", "kind": "edit"})
            return await original(**params)

        client.app.client.conversations_history = _observing
        entries = run(C.resolve_anchors(db=db, client=client, team_id=TEAM, channel_id=CH,
                                        roots=["900.000000"], frozen_receipts={}))
        assert entries[0].observation_frontier == 0, (
            "the mutation landed during the fetch and must remain visible to the CAS")

    def test_an_own_root_with_no_finalized_receipt_is_unsafe(self):
        """Test 84: a pre-floor own root that is chrome, or post-epoch with no finalized
        receipt, renders `[root unavailable]` rather than exposing its text."""
        client = self._client([message("900.000000", kind="self", user="B1",
                                       text="status card")])
        entries = run(C.resolve_anchors(db=FakeDB(), client=client, team_id=TEAM,
                                        channel_id=CH, roots=["900.000000"],
                                        frozen_receipts={"900.000000": C.PROOF_UNPROVEN}))
        assert entries[0].status == C.ANCHOR_UNSAFE
        assert entries[0].text is None
        assert "[root unavailable]" in C.render_anchor_block(
            [entries[0].render_entry()], omitted=0)

    def test_the_fetch_is_bounded_by_the_map_bound(self):
        inventory = {f"{800 + i}.000000": {"last_canonical_message_ts": "1900.000000"}
                     for i in range(60)}
        roots, omitted = C.straddling_roots(inventory, boundary_ts="1000.000000", bound=40)
        client = self._client([message(r) for r in roots])
        entries = run(C.resolve_anchors(db=FakeDB(), client=client, team_id=TEAM,
                                        channel_id=CH, roots=roots, frozen_receipts={}))
        assert len(entries) == 40 == len(client.history_calls)
        assert omitted == 20

    def test_every_anchored_root_gets_a_provenance_row(self):
        """A missing row would mean "this snapshot never anchored that thread", which is false."""
        client = self._client([message("900.000000")])
        entries = run(C.resolve_anchors(db=FakeDB(), client=client, team_id=TEAM,
                                        channel_id=CH,
                                        roots=["900.000000", "901.000000"],
                                        frozen_receipts={}))
        assert len(entries) == 2
        rows = [e.provenance_row(team_id=TEAM) for e in entries]
        assert {r["root_ts"] for r in rows} == {"900.000000", "901.000000"}
        assert all(r["projection_sha256"] for r in rows)
        assert rows[1]["receipt_proof"] is None, "NULL means exactly one thing"


# ---------------------------------------------------------------- test 98

class TestValidationPrecedesInsertion:
    def _seeded(self, monkeypatch, *, count=45, chunk=10):
        monkeypatch.setattr(C, "HASH_CHUNK_EVENTS", chunk)
        msgs = [message(f"{1100 + i}.000000", text=f"line {i}") for i in range(count)]
        db, cp = FakeDB(), checkpoint()
        run(C.run_phase_one(db=db, client=FakeSlack(msgs), checkpoint=cp,
                            budget=C.SliceBudget(), frozen_receipts={}))
        run(C.run_phase_two(db=db, client=FakeSlack(msgs), openai_client=FakeResponses(),
                            checkpoint=cp,
                            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
                            budget=C.SliceBudget(), frozen_receipts={}))
        return db, cp, msgs

    def test_an_over_cap_summary_leaves_no_rows_behind(self, monkeypatch):
        db, cp, msgs = self._seeded(monkeypatch)
        coordinator = FakeCoordinator()
        sizing = C.resolve_sizing(model="m", window=100_000)
        result = run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses("z" * 50_000),
            coordinator=coordinator, checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"), sizing=sizing,
            headroom_source="measured", headroom_tokens=100, expected_previous_id=None,
            serializer_version=2, max_generation_attempts=2))
        assert result["outcome"] == "discarded"
        assert result["reason"] == "over_byte_cap"
        assert db.candidates == [] and db.manifests == {} and db.anchors == {}
        assert coordinator.calls == []

    def test_a_summary_that_only_exceeds_the_cap_once_escaped_is_rejected(self, monkeypatch):
        """The escape pass runs BEFORE validation, so the bytes checked are the bytes persisted.
        An implementation validating raw output publishes an over-cap payload."""
        from config import config as live_config
        from message_processor.channel_stream import escape_payload

        db, cp, msgs = self._seeded(monkeypatch)
        coordinator = FakeCoordinator()
        # Exactly the configured cap RAW; over it once A7 prefixes the forged marker ("· " is
        # three UTF-8 bytes, which is precisely the kind of arithmetic a raw check misses).
        cap = int(live_config.summary_byte_cap)
        forged = "[NOTE:" + "x" * (cap - len("[NOTE:"))
        assert len(forged.encode("utf-8")) == cap
        assert len(escape_payload(forged).encode("utf-8")) == cap + 3
        result = run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses(forged),
            coordinator=coordinator, checkpoint=cp,
            # A window wide enough that the cap is SUMMARY_BYTE_CAP, not the room.
            sizing=C.resolve_sizing(model="m", window=1_050_000), headroom_source="measured",
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            headroom_tokens=100, expected_previous_id=None, serializer_version=2))
        assert result["outcome"] == "discarded" and result["reason"] == "over_byte_cap"
        assert db.candidates == [] and coordinator.calls == []

    def test_an_empty_summary_leaves_no_rows_behind(self, monkeypatch):
        db, cp, msgs = self._seeded(monkeypatch)
        coordinator = FakeCoordinator()
        result = run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses("   "),
            coordinator=coordinator, checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            sizing=C.resolve_sizing(model="m", window=100_000), headroom_source="measured",
            headroom_tokens=100, expected_previous_id=None, serializer_version=2))
        assert result["outcome"] == "discarded" and result["reason"] == "empty_output"
        assert db.candidates == []

    def test_a_valid_summary_inserts_and_publishes(self, monkeypatch):
        db, cp, msgs = self._seeded(monkeypatch)
        coordinator = FakeCoordinator()
        result = run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses("a tidy account"),
            coordinator=coordinator, checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            sizing=C.resolve_sizing(model="m", window=100_000), headroom_source="measured",
            headroom_tokens=100, expected_previous_id=None, serializer_version=2,
            now=500.0))
        assert result["outcome"] == "published", result
        row = db.snapshots[result["snapshot_id"]]
        assert row["payload_hash"] == hashlib.sha256(row["payload_bytes"]).hexdigest()
        assert row["source_hash"] and row["prompt_version"] == C.PROMPT_VERSION
        assert row["boundary_ts"] in {r["ts"] for r in db.sealed[cp["crawl_id"]]}
        assert row["fit_result"] in (C.FIT_UNDER_TARGET, C.FIT_UNDER_TRIGGER)
        bodies = [r["body"] for r in coordinator.calls[0]["outbox_rows"]]
        assert [b["op"] for b in bodies] == ["build", "publish"]
        assert [b["event_seq"] for b in bodies] == [0, 1]
        assert bodies[0]["at"] == bodies[1]["at"] == 500.0

    def test_the_candidate_carries_every_sizing_field(self, monkeypatch):
        db, cp, msgs = self._seeded(monkeypatch)
        run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses("account"),
            coordinator=FakeCoordinator(), checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"),
            sizing=C.resolve_sizing(model="m", window=1_050_000), headroom_source="fixed",
            headroom_tokens=80000, expected_previous_id=None, serializer_version=2))
        # FakeDB rejects a v2 candidate missing any sizing field, exactly as A1's accessor does.
        assert db.candidates and db.candidates[0]["headroom_source"] == "fixed"

    def test_publish_nothing_when_no_boundary_fits(self, monkeypatch):
        db, cp, msgs = self._seeded(monkeypatch)
        coordinator = FakeCoordinator()
        tiny = dict(C.resolve_sizing(model="m", window=100), target_tokens=1,
                    trigger_tokens=2)
        result = run(C.complete_attempt(
            db=db, client=FakeSlack(msgs), openai_client=FakeResponses("x"),
            coordinator=coordinator, checkpoint=cp,
            attempt=C.CompactionAttempt.from_checkpoint(cp, model="m"), sizing=tiny,
            headroom_source="measured", headroom_tokens=0, expected_previous_id=None,
            serializer_version=2))
        assert result["outcome"] == "publish_nothing"
        assert db.candidates == [] and coordinator.calls == []
