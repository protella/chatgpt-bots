"""The compaction telemetry outbox primitives — plan §1l, mandated test 100.

WHAT IS ACTUALLY BEING PROTECTED HERE. A compaction state change and its telemetry event must
commit together, and they live in two different stores (SQLite and a JSONL file). The outbox row
bridges them, and it is DELETED once the event is acknowledged — so every property below is about
what may safely be deleted:

1. THE TWO-LAYER SPLIT. The CANONICAL BODY is replay-stable identity: sessionless, serialized once
   from persisted facts, byte-identical on every reconstruction. The CV8 ENVELOPE is emission
   provenance: exactly `v`, `session`, `gate_contract`, added at emission and nothing else. Compare
   whole lines instead of bodies and every honest post-restart replay reads as a conflict.
2. VALIDATION IS NOT "PARSES AS JSON". A payload can be perfectly well-formed and still be unsafe
   to acknowledge: the wrong identity, or a `publish` at sequence 0, would be emitted and DELETED,
   permanently breaking the ordering guarantee with nothing left to inspect.
3. ACKNOWLEDGEMENT IS NOT ENQUEUEING. The turn-path writer hands records to a daemon listener and
   returns; a row deleted on that promise is lost whenever the process dies with the write still
   queued — the exact failure the outbox exists to prevent.

The drainer itself (batching, backoff, the poison-row halt, boot and shutdown ordering) is the
coordinator's, and its cases — 100(a)-(i) — are tested there. What lives here is the vocabulary,
the serializer, the validator, the acknowledged flush seam and the ledger checker.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from config import config
from message_processor import participation_telemetry as pt

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "participation_ledger_check.py"

ABSENT = object()      # in an override dict: delete the key rather than set it


# ------------------------------------------------------------------------------ the harness

@pytest.fixture
def sink(tmp_path):
    """The real ledger, pointed at a temp dir, with a reader for its lines.

    The named logger lives in logging's global registry, so its handlers are saved and restored:
    a handler left behind would keep writing into a previous test's file.
    """
    named = logging.getLogger(pt._SINK_LOGGER_NAME)
    saved = named.handlers[:]
    pt.shutdown()
    with patch.object(config, "log_directory", str(tmp_path)), \
            patch.object(config, "enable_participation_telemetry", True):
        pt.initialize()
        path = tmp_path / pt.LOG_NAME

        def lines(event=None, *, drain=True):
            if drain:
                pt._drain()
            if not path.exists():
                return []
            rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
            return [r for r in rows if event is None or r.get("event") == event]

        lines.path = path
        try:
            yield lines
        finally:
            pt.shutdown()
    named.handlers = saved


def build_body(**over):
    """A valid `op=build` canonical body. `event_seq` is 0 because a build always is."""
    body = {
        "event": "compaction_snapshot", "crawl_id": "cr-1", "attempt_seq": 1, "event_seq": 0,
        "team_id": "T1", "channel_id": "C1", "namespace": "prod", "at": 1_700_000_000.5,
        "op": "build", "model": "gpt-5.6-luna", "tokens_in": 90_000, "tokens_out": 4_200,
        "cached_input_tokens": 61_000, "call_count": 9, "status": "ok",
    }
    if over.get("status") in ("failed", "discarded"):
        body["reason"] = "reduce_call_failed"
    body.update(over)
    return {key: value for key, value in body.items() if value is not ABSENT}


def publish_body(**over):
    body = {
        "event": "compaction_snapshot", "crawl_id": "cr-1", "attempt_seq": 1, "event_seq": 1,
        "team_id": "T1", "channel_id": "C1", "namespace": "prod", "at": 1_700_000_001.25,
        "op": "publish", "snapshot_id": "snap-7", "generation": 4, "boundary_ts": "1699.0",
        "fit_result": "under_target", "serializer_version": 2,
    }
    body.update(over)
    return {key: value for key, value in body.items() if value is not ABSENT}


def clause(body, **row):
    """The failing clause name, with the row's own identity defaulting to the body's."""
    source = body if isinstance(body, dict) else {}
    return pt.validate_outbox_body(
        body,
        crawl_id=row.get("crawl_id", source.get("crawl_id")),
        attempt_seq=row.get("attempt_seq", source.get("attempt_seq")),
        event_seq=row.get("event_seq", source.get("event_seq")),
        created_ts=row.get("created_ts", source.get("at")))


def run_checker(*paths, extra=("--json",)):
    """The CLI, by path, from a directory that is NOT the repo: "runs against a jsonl copied off a
    box with no venv" is a requirement, not a preference."""
    argv = [sys.executable, str(CHECKER), *[str(p) for p in paths], *extra]
    result = subprocess.run(argv, capture_output=True, text=True,
                            cwd=str(Path(paths[0]).parent))
    assert result.stderr == "", result.stderr
    return result.returncode, json.loads(result.stdout)


def checker_module():
    """The checker imported in-process, for the one assertion that compares its serializer with
    the emitter's. It imports nothing from the repo, which is what makes this safe."""
    spec = importlib.util.spec_from_file_location("_ledger_check", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def envelope(body, *, session="S1"):
    """One emitted line: the body flattened under exactly the three wrapper keys."""
    line = {"v": pt.CONTRACT_VERSION, "at": body["at"], "session": session,
            "gate_contract": pt.GATE_CONTRACT}
    line.update({k: v for k, v in body.items() if k != "at"})
    return line


def write_ledger(tmp_path, rows, name="participation.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def names(payload):
    return set(payload["violation_counts"])


# =========================================================================== the one serializer

def test_the_canonical_serializer_is_the_pinned_call_exactly():
    """100(j) pins it. Three places use it — the bytes stored in the row, the insert-conflict
    comparison, the checker's extraction — and any two differing would make identical bodies
    compare unequal over key order or escaping alone."""
    body = build_body()
    assert pt.canonical_body_bytes(body) == json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_key_order_and_dict_identity_do_not_change_the_bytes():
    """The retry path REBUILDS the body; nothing guarantees it builds the keys in the same order."""
    body = build_body()
    shuffled = {key: body[key] for key in reversed(list(body))}
    assert shuffled is not body and list(shuffled) != list(body)
    assert pt.canonical_body_bytes(shuffled) == pt.canonical_body_bytes(body)


def test_non_ascii_is_serialized_as_itself_not_as_escapes():
    """`ensure_ascii=False` is part of the pinned call. A body escaped one way and compared
    against the same body escaped the other differs on nothing that means anything."""
    body = build_body(reason="résumé too long", status="failed")
    assert "résumé".encode("utf-8") in pt.canonical_body_bytes(body)


def test_a_reconstructed_body_is_byte_identical_across_a_simulated_restart(sink):
    """100(m) at this layer. An uncertain commit is retried by REBUILDING the body from the
    persisted terminal facts, not by holding the original object — so the bytes must still match
    when the reconstructing process is a DIFFERENT session. A test that re-serializes the original
    object proves nothing: it passes even for an implementation stamping now() or a session id."""
    facts = {"crawl_id": "cr-9", "attempt_seq": 2, "created_ts": 1_700_000_444.125,
             "tokens_in": 1_000, "call_count": 3}

    def reconstruct():
        return build_body(crawl_id=facts["crawl_id"], attempt_seq=facts["attempt_seq"],
                          at=facts["created_ts"], tokens_in=facts["tokens_in"],
                          call_count=facts["call_count"])

    original = pt.canonical_body_bytes(reconstruct())
    with patch.object(pt, "SESSION_ID", "a-different-process"):
        replay = pt.canonical_body_bytes(reconstruct())
    assert replay == original
    assert b"session" not in original      # THERE IS NO SESSION FIELD IN THE CANONICAL BODY
    assert b"gate_contract" not in original


# ============================================================== validate_outbox_body — clause 1-6

def test_a_valid_build_and_a_valid_publish_pass():
    assert clause(build_body()) is None
    assert clause(publish_body()) is None
    assert clause(build_body(status="copied", call_count=0, tokens_in=0, tokens_out=0,
                             cached_input_tokens=0)) is None
    assert clause(build_body(status="discarded")) is None


@pytest.mark.parametrize("body", ["a string", 7, None, ["compaction_snapshot"]])
def test_clause_1_a_non_object_payload_is_not_an_event(body):
    assert pt.validate_outbox_body(body, crawl_id="cr-1", attempt_seq=1, event_seq=0,
                                   created_ts=1.0) == "not_object"


@pytest.mark.parametrize("event", ["model_response", "stream_render", "", ABSENT, 8])
def test_clause_2_only_one_event_rides_this_table(event):
    assert clause(build_body(event=event)) == "event"


def test_clause_3_a_payload_identity_that_is_not_the_rows_own_is_refused():
    """THE CLAUSE THAT MATTERS MOST. The row is deleted after acknowledgement and the checker reads
    only the ledger, so a payload naming a different identity would be deduped under a triple the
    row never had — and the row itself is gone."""
    assert clause(build_body(), crawl_id="cr-other") == "identity"
    assert clause(build_body(), attempt_seq=2) == "identity"
    assert clause(publish_body(), event_seq=2) == "identity"
    assert clause(build_body(crawl_id=ABSENT)) == "identity"
    assert clause(build_body(attempt_seq="1"), attempt_seq="1") == "identity"
    assert clause(build_body(attempt_seq=True), attempt_seq=True) == "identity"


def test_clause_3_accepts_only_a_byte_equal_crawl_id():
    assert clause(build_body(crawl_id="CR-1"), crawl_id="cr-1") == "identity"


@pytest.mark.parametrize("field", ["team_id", "channel_id", "namespace", "op", "model",
                                   "tokens_in", "tokens_out", "cached_input_tokens",
                                   "call_count", "status"])
def test_clause_4_every_required_build_field_is_required(field):
    assert clause(build_body(**{field: ABSENT})) == "fields"


@pytest.mark.parametrize("field", ["snapshot_id", "generation", "boundary_ts", "fit_result",
                                   "serializer_version"])
def test_clause_4_every_required_publish_field_is_required(field):
    assert clause(publish_body(**{field: ABSENT})) == "fields"


@pytest.mark.parametrize("field,value", [("tokens_in", "90000"), ("tokens_out", 4.2),
                                          ("call_count", True), ("model", 5),
                                          ("cached_input_tokens", None)])
def test_clause_4_a_mistyped_field_is_as_bad_as_a_missing_one(field, value):
    assert clause(build_body(**{field: value})) == "fields"


def test_clause_4_the_token_names_are_authoritative():
    """100(n). `input_tokens` / `output_tokens` belong to model_response and its per-call turn
    population. A schema carrying them here must FAIL — one vocabulary, no aliases."""
    aliased = build_body(tokens_in=ABSENT, tokens_out=ABSENT)
    aliased["input_tokens"] = 90_000
    aliased["output_tokens"] = 4_200
    assert clause(aliased) == "fields"
    # And the alias is refused even when the authoritative names are also present, so a body
    # cannot quietly carry both and let a reader pick.
    both = build_body()
    both["input_tokens"] = 90_000
    assert clause(both) == "fields"


def test_clause_4_a_publish_carries_no_token_fields():
    """The cost belongs to the attempt that produced the summary; duplicating it on the
    publication would double-count any aggregate that sums the ledger."""
    for field in ("tokens_in", "tokens_out", "cached_input_tokens", "call_count", "model"):
        assert clause(publish_body(**{field: 1})) == "fields"


def test_clause_4_reason_is_required_exactly_when_the_attempt_did_not_succeed():
    for status in ("failed", "discarded"):
        assert clause(build_body(status=status, reason=ABSENT)) == "fields"
        assert clause(build_body(status=status)) is None
    for status in ("ok", "copied"):
        assert clause(build_body(status=status, reason="nothing went wrong")) == "fields"
        assert clause(build_body(status=status)) is None


@pytest.mark.parametrize("status", ["succeeded", "OK", "", 1, ABSENT])
def test_clause_4_a_build_status_outside_the_vocabulary_is_refused(status):
    assert clause(build_body(status=status)) == "fields"


@pytest.mark.parametrize("fit", ["under_budget", "", 2, ABSENT])
def test_clause_4_a_publish_fit_result_outside_the_vocabulary_is_refused(fit):
    assert clause(publish_body(fit_result=fit)) == "fields"


@pytest.mark.parametrize("op", ["read", "invalidate", "stale_retained", "", ABSENT])
def test_clause_4_only_outbox_routed_ops_can_be_stored(op):
    """A turn-path event has no crawl attempt to be keyed by, so it cannot live in a table keyed
    by one. Direct-write ops are direct precisely because that identity does not exist."""
    assert clause(build_body(op=op)) == "fields"


def test_clause_5_a_publish_is_never_at_sequence_zero():
    """THE OTHER CLAUSE THAT MATTERS MOST. A publish stored at 0 reaches the ledger with no build
    before it — the precise failure `event_seq` exists to prevent — and is then deleted."""
    assert clause(publish_body(event_seq=0)) == "routing"


def test_clause_5_a_build_is_always_at_sequence_zero():
    assert clause(build_body(event_seq=1)) == "routing"
    assert clause(build_body(event_seq=2)) == "routing"


@pytest.mark.parametrize("key", ["v", "session", "gate_contract"])
def test_clause_6_a_wrapper_key_in_the_body_is_refused(key):
    """100(j3), first of three points. A body carrying emission provenance would let a persisted
    value overwrite the provenance of the process actually writing the line. There is deliberately
    no merge-precedence rule: refusing is the only correct answer, so it is the only rule."""
    assert clause(build_body(**{key: "anything"})) == "wrapper"
    assert clause(publish_body(**{key: "anything"})) == "wrapper"


@pytest.mark.parametrize("at", [float("nan"), float("inf"), float("-inf")])
def test_clause_6_a_non_finite_at_is_refused(at):
    """`json.dumps` writes bare NaN / Infinity, which no strict JSON reader accepts back."""
    assert clause(build_body(at=at), created_ts=at) == "wrapper"


def test_clause_6_a_corrupted_but_finite_at_is_caught_on_read():
    """100(j2)/(k), the case insert-time validation cannot see. A post-landing corruption that
    replaces `at` with a DIFFERENT finite float passes every other clause — so the row's own
    column is the reference value, and the check runs on read as well as at insert."""
    body = build_body(at=1_700_000_000.5)
    assert clause(body, created_ts=1_700_000_000.5) is None
    assert clause(body, created_ts=1_700_000_000.75) == "wrapper"   # one quarter-second adrift


@pytest.mark.parametrize("at", ["1700000000.5", ABSENT, None, True])
def test_clause_6_at_stays_numeric(at):
    """Rendering it as a string is a silent type change on a graded field, and would break
    anything parsing `at` numerically."""
    assert clause(build_body(at=at), created_ts=1_700_000_000.5) == "wrapper"


def test_an_integer_at_equal_to_the_row_is_accepted():
    """Numeric equality, not type identity: `1700000000 == 1700000000.0`."""
    assert clause(build_body(at=1_700_000_000), created_ts=1_700_000_000.0) is None


def test_the_clause_names_are_the_pinned_six():
    """Three other agents branch on these strings."""
    seen = {
        clause("not a dict", crawl_id="cr-1", attempt_seq=1, event_seq=0, created_ts=1.0),
        clause(build_body(event="turn_start")),
        clause(build_body(), crawl_id="other"),
        clause(build_body(model=ABSENT)),
        clause(build_body(event_seq=3), event_seq=3),
        clause(build_body(session="S1")),
    }
    assert seen == {"not_object", "event", "identity", "fields", "routing", "wrapper"}


# ============================================================ emission: the acknowledged seam

def test_emission_adds_exactly_three_fields_and_takes_at_and_event_from_the_body(sink):
    """The five CV8 envelope fields, assembled from two layers: `v`, `session` and `gate_contract`
    are emission provenance; `at` and `event` come from the body because both must survive a
    replay."""
    body = build_body()
    assert pt.emit_outbox_body(body) is True
    line = sink("compaction_snapshot", drain=False)[0]
    assert set(line) - set(body) == {"v", "session", "gate_contract"}
    assert set(body) - set(line) == set()
    assert (line["v"], line["gate_contract"]) == (pt.CONTRACT_VERSION, pt.GATE_CONTRACT)
    assert line["session"] == pt.SESSION_ID
    assert line["event"] == "compaction_snapshot"


def test_the_emitted_at_is_the_bodys_own_and_stays_a_float(sink):
    """100(j2). An implementation stamping `time.time()` at emission passes the type check and
    fails this equality one."""
    body = build_body(at=1_600_000_000.125)
    assert pt.emit_outbox_body(body) is True
    line = sink("compaction_snapshot", drain=False)[0]
    assert isinstance(line["at"], float) and not isinstance(line["at"], bool)
    assert line["at"] == 1_600_000_000.125


def test_a_replay_from_another_session_differs_only_in_its_envelope(sink):
    """The point of the split: the body is the same row re-emitted, the envelope is a different
    process. Both lines must strip back to identical canonical bytes."""
    body = build_body()
    assert pt.emit_outbox_body(body) is True
    with patch.object(pt, "SESSION_ID", "second-process"):
        assert pt.emit_outbox_body(dict(body)) is True
    first, second = [json.dumps(r) for r in sink("compaction_snapshot", drain=False)]
    assert first != second                                        # the sessions differ
    assert pt.extract_canonical_body(first) == pt.extract_canonical_body(second)


@pytest.mark.parametrize("key", ["v", "session", "gate_contract"])
def test_emission_refuses_a_wrapper_collision_and_writes_nothing(sink, key):
    """100(j3), third point. No merge occurs and no precedence rule exists — the row takes the
    poison path instead, so a human sees it rather than a silently rewritten line."""
    with patch.object(pt, "logger") as log:
        assert pt.emit_outbox_body(build_body(**{key: "hijack"})) is False
        assert log.critical.call_count == 1
    assert sink("compaction_snapshot") == []


def test_the_bytes_are_durable_before_emission_returns(sink):
    """THE WHOLE SEAM. `record()` hands the line to a daemon listener and returns; this must not
    return until the bytes are in the file, or a row deleted on its word is lost to a crash with
    the write still queued."""
    assert pt.emit_outbox_body(build_body()) is True
    assert len(sink("compaction_snapshot", drain=False)) == 1   # read WITHOUT draining


def test_a_write_still_sitting_in_the_queue_is_not_an_acknowledgement(sink):
    """100(b) at this layer, made deterministic by STALLING THE LISTENER. An implementation that
    returns as soon as `record()` has enqueued reports success here while the bytes are still in
    the queue — and the drainer deletes the row that was the only remaining copy."""
    handler = pt._listener.handlers[0]
    released = threading.Event()
    original = handler.emit

    def stalled(record):
        released.wait(5.0)
        original(record)

    with patch.object(handler, "emit", stalled):
        assert pt.emit_outbox_body(build_body(), timeout=0.2) is False
        assert sink("compaction_snapshot", drain=False) == []    # nothing reached the file
        released.set()
    pt._drain()
    assert len(sink("compaction_snapshot", drain=False)) == 1    # and nothing was lost, either


def test_no_acknowledgement_means_no_success(sink):
    """A False is the drainer's instruction to keep the row. The line may well have been written;
    what was not obtained is the promise that it was."""
    with patch.object(pt, "flush_sync", return_value=False):
        assert pt.emit_outbox_body(build_body()) is False


def test_a_dead_listener_is_not_an_acknowledgement(sink):
    """Waiting on a queue nobody drains would burn the whole timeout and then lie about it."""
    worker = pt._listener._thread
    with patch.object(worker, "is_alive", return_value=False):
        assert pt.flush_sync(timeout=0.1) is False
        assert pt.emit_outbox_body(build_body()) is False


def test_flush_and_emission_refuse_when_the_sink_was_never_opened():
    """Rows are RETAINED while telemetry is disabled or unavailable — a durable backlog, not
    best-effort. Reporting success here would delete them."""
    pt.shutdown()
    assert pt.flush_sync(timeout=0.1) is False
    assert pt.emit_outbox_body(build_body()) is False


def test_telemetry_disabled_emits_nothing_and_acknowledges_nothing(sink):
    with patch.object(config, "enable_participation_telemetry", False):
        assert pt.emit_outbox_body(build_body()) is False
    assert sink("compaction_snapshot") == []


def test_emission_never_raises_into_the_drainer(sink):
    """It runs inside a loop holding a database row. A telemetry failure there must cost the
    acknowledgement, never the process."""
    assert pt.emit_outbox_body("not a body") is False
    assert pt.emit_outbox_body({"event": "compaction_snapshot", "at": object()}) is False


def test_the_turn_path_still_only_enqueues(sink):
    """TURN-PATH ENQUEUEING IS NOT CHANGED. The acknowledged seam is an ADDITIONAL entry point
    used by the drainer alone; putting a synchronous flush on the gate's path would put a file
    write inside the one call the participation decision is waiting on."""
    with patch.object(pt, "flush_sync") as flush:
        pt.compaction_snapshot(op="read", channel_id="C1", snapshot_id="s1")
        pt.gate_start("C1", "1.0", attempt_id="a1")
        assert flush.call_count == 0
    assert len(sink("compaction_snapshot")) == 1


# ============================================================ extraction, on both sides

def test_extraction_strips_exactly_the_three_wrapper_keys():
    body = build_body()
    line = json.dumps(envelope(body))
    assert pt.extract_canonical_body(line) == pt.canonical_body_bytes(body)


def test_extraction_is_not_a_substring_or_a_whole_line_comparison():
    """A comparison that skips the canonical RE-SERIALIZATION differs on key order or `\\uXXXX`
    escaping alone. Both of those are true of the emitted line here, and neither means anything."""
    body = build_body(reason="naïve", status="failed")
    line = json.dumps(envelope(body))                    # ensure_ascii=True, insertion order
    assert "\\u00ef" in line                              # the file escapes it
    assert "naïve".encode("utf-8") in pt.extract_canonical_body(line)   # the body does not
    assert pt.extract_canonical_body(line) == pt.canonical_body_bytes(body)
    assert pt.extract_canonical_body(line) != line.encode("utf-8")


def test_extraction_refuses_a_line_that_is_not_an_object():
    with pytest.raises(ValueError):
        pt.extract_canonical_body("[1, 2]")
    with pytest.raises(ValueError):
        pt.extract_canonical_body("{not json")


def test_the_checker_and_the_emitter_serialize_identically():
    """The checker is stdlib-only by requirement — it runs where the evidence is, on a box with no
    venv and no repo — so its serializer is RESTATED rather than imported. Restated code drifts;
    this is the assertion that says so the moment it does."""
    checker = checker_module()
    body = build_body(reason="ünicode ✓", status="failed")
    line = json.dumps(envelope(body))
    assert checker.extract_canonical_body(line) == pt.extract_canonical_body(line)
    assert checker.canonical_body_bytes(body) == pt.canonical_body_bytes(body)
    assert tuple(checker.WRAPPER_KEYS) == tuple(pt.CANONICAL_BODY_KEYS)
    assert set(checker.BUILD_STATUSES) == set(pt.BUILD_STATUSES)
    assert set(checker.OUTBOX_OPS) == set(pt.OUTBOX_OPS)
    assert set(checker.SNAPSHOT_OPS) == set(pt.SNAPSHOT_OPS)
    assert set(checker.FIT_RESULTS) == set(pt.FIT_RESULTS)


# ============================================================ the ledger checker, invariant 6

def test_a_real_emitted_attempt_passes_the_checker(sink, tmp_path):
    """THE ROUND TRIP: the emitter's own bytes, the real envelope, graded by the real CLI."""
    assert pt.emit_outbox_body(build_body()) is True
    assert pt.emit_outbox_body(publish_body()) is True
    pt.shutdown()
    code, payload = run_checker(sink.path)
    assert (code, payload["violations"], payload["verdict"]) == (0, [], "pass")
    assert payload["counts"]["compaction_outbox_events"] == 2
    assert payload["counts"]["compaction_builds"] == 1


def test_a_replayed_line_is_expected_and_never_a_violation(tmp_path):
    """100(c)+(j). A crash between emit and delete replays the row, so DUPLICATE LINES ARE
    EXPECTED: the checker counts DISTINCT triples, not lines. The replay carries a different
    `session`, exactly as an honest post-restart drain does — a checker comparing whole JSONL
    lines flags every legitimate replay and must fail here."""
    body = build_body()
    ledger = write_ledger(tmp_path, [envelope(body, session="S1"),
                                     envelope(body, session="S2-after-restart")])
    code, payload = run_checker(ledger)
    assert (code, payload["violations"]) == (0, [])
    assert payload["counts"]["compaction_outbox_events"] == 1        # one identity
    assert payload["counts"]["compaction_outbox_replayed_lines"] == 1


def test_one_identity_carrying_two_different_bodies_is_reported(tmp_path):
    """100(j). Counting distinct triples and stopping there would silently absorb two DIFFERENT
    events wearing one identity — a payload rebuilt from changed state, or an identity reused
    across attempts — hiding inside the very mechanism meant to make duplicates safe."""
    ledger = write_ledger(tmp_path, [envelope(build_body(tokens_in=90_000)),
                                     envelope(build_body(tokens_in=91_000))])
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_outbox_body_conflict" in names(payload)


def test_a_body_differing_only_in_key_order_is_not_a_conflict(tmp_path):
    """The re-serialization is what makes the comparison deterministic. Without it two identical
    bodies written in different key orders read as a contract violation."""
    body = build_body()
    reordered = {key: body[key] for key in reversed(list(body))}
    ledger = write_ledger(tmp_path, [envelope(body), envelope(reordered, session="S2")])
    code, payload = run_checker(ledger)
    assert (code, payload["violations"]) == (0, [])


def test_a_publish_with_no_build_anywhere_is_a_violation(tmp_path):
    """The cardinality rule the outbox exists for: at least one build precedes any publish."""
    code, payload = run_checker(write_ledger(tmp_path, [envelope(publish_body())]))
    assert code == 1 and "compaction_publish_without_build" in names(payload)


def test_a_publish_emitted_before_its_own_build_is_a_violation(tmp_path):
    """100(a) at the ledger layer. The drainer emits in `outbox_seq` order, so a publish arriving
    first means the ordering guarantee broke — an implementation direct-emitting the publish
    while the build waits in the outbox must fail here."""
    ledger = write_ledger(tmp_path, [envelope(publish_body()), envelope(build_body())])
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_publish_before_build" in names(payload)


def test_a_build_from_another_attempt_does_not_satisfy_a_publish(tmp_path):
    """The rule is per ATTEMPT. A build for attempt 1 says nothing about attempt 2's publication."""
    ledger = write_ledger(tmp_path, [envelope(build_body(attempt_seq=1)),
                                     envelope(publish_body(attempt_seq=2))])
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_publish_without_build" in names(payload)


def test_a_copy_publication_satisfies_the_rule_with_a_zero_call_build(tmp_path):
    """MANDATED TEST 64, end to end. A stale-retained copy IS a publication, so it emits the pair
    like any other and its build records that no model call was made."""
    ledger = write_ledger(tmp_path, [
        envelope(build_body(status="copied", call_count=0, tokens_in=0, tokens_out=0,
                            cached_input_tokens=0)),
        envelope(publish_body())])
    code, payload = run_checker(ledger)
    assert (code, payload["violations"]) == (0, [])


def test_a_failed_attempt_emits_a_build_alone_and_that_is_complete(tmp_path):
    """MANDATED TEST 13 at the ledger layer. Nothing was published, so there is no publish event —
    and no rule may demand one."""
    ledger = write_ledger(tmp_path, [envelope(build_body(status="failed"))])
    code, payload = run_checker(ledger)
    assert (code, payload["violations"]) == (0, [])
    assert payload["counts"]["compaction_builds"] == 1


def test_the_direct_write_ops_carry_no_completeness_rule(tmp_path):
    """DELIBERATE ACCEPTANCE. `op=read`, `op=invalidate` and stale-retained copies are synchronous
    outcomes of a DB transaction, not products of a crawl attempt: they have no identity to be
    keyed by and CAN BE LOST to a crash in the gap. Their durable truth is the database row. No
    "every invalidation has an event" assertion exists or may be added."""
    ledger = write_ledger(tmp_path, [
        envelope({"event": "compaction_snapshot", "at": 1.0, "op": "read", "channel_id": "C1",
                  "snapshot_id": "s1", "generation": 4}),
        envelope({"event": "compaction_snapshot", "at": 2.0, "op": "invalidate",
                  "channel_id": "C1", "snapshot_id": "s1"}),
        envelope({"event": "compaction_snapshot", "at": 3.0, "op": "stale_retained",
                  "channel_id": "C1", "snapshot_id": "s2"})])
    code, payload = run_checker(ledger)
    assert (code, payload["violations"]) == (0, [])
    assert payload["counts"]["compaction_outbox_events"] == 0


@pytest.mark.parametrize("field", ["crawl_id", "attempt_seq", "event_seq"])
def test_an_outbox_event_without_its_identity_is_a_violation(tmp_path, field):
    """100(h). The row is deleted after acknowledgement and the checker reads only the JSONL, so
    an identity living solely in dropped columns is one nothing can count."""
    ledger = write_ledger(tmp_path, [envelope(build_body(**{field: ABSENT}))])
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_snapshot_missing_identity" in names(payload)


def test_a_misplaced_op_is_a_violation_in_the_ledger_too(tmp_path):
    ledger = write_ledger(tmp_path, [envelope(publish_body(event_seq=0))])
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_snapshot_misplaced_op" in names(payload)
    ledger = write_ledger(tmp_path, [envelope(build_body(event_seq=1))], name="second.jsonl")
    code, payload = run_checker(ledger)
    assert code == 1 and "compaction_snapshot_misplaced_op" in names(payload)


def test_the_checker_refuses_the_retired_token_names(tmp_path):
    """100(n) on the checker side: the vocabularies are restated here, so they are graded here."""
    aliased = build_body(tokens_in=ABSENT, tokens_out=ABSENT)
    aliased["input_tokens"] = 90_000
    aliased["output_tokens"] = 4_200
    code, payload = run_checker(write_ledger(tmp_path, [envelope(aliased)]))
    assert code == 1 and "compaction_snapshot_missing_field" in names(payload)


def test_the_checker_reports_a_publish_carrying_token_fields(tmp_path):
    body = publish_body()
    body["tokens_in"] = 90_000
    code, payload = run_checker(write_ledger(tmp_path, [envelope(build_body()), envelope(body)]))
    assert code == 1 and "compaction_publish_carries_tokens" in names(payload)


def test_the_checker_grades_the_reason_rule(tmp_path):
    """A failed generation must not be invisible; a successful one must not carry an empty excuse."""
    silent_failure = build_body(status="failed")
    silent_failure.pop("reason")
    code, payload = run_checker(write_ledger(tmp_path, [envelope(silent_failure)]))
    assert code == 1 and "compaction_snapshot_reason_rule" in names(payload)
    code, payload = run_checker(
        write_ledger(tmp_path, [envelope(build_body(reason="but it worked"))], name="b.jsonl"))
    assert code == 1 and "compaction_snapshot_reason_rule" in names(payload)


def test_a_build_status_outside_the_vocabulary_is_a_violation(tmp_path):
    code, payload = run_checker(
        write_ledger(tmp_path, [envelope(build_body(status="succeeded"))]))
    assert code == 1 and "compaction_snapshot_bad_status" in names(payload)


def _refuse(token):
    raise ValueError(f"{token} is not JSON")


def test_an_at_that_is_not_finite_never_survives_to_a_ledger_line():
    """Why finiteness is a clause and not a nicety: `json.dumps` writes a bare `NaN` token, which
    Python reads back happily and no strict JSON reader accepts at all. The validator is the only
    thing standing between a corrupted row and a ledger line half the world cannot parse."""
    body = build_body(at=float("nan"))
    assert clause(body, created_ts=float("nan")) == "wrapper"
    assert "NaN" in json.dumps(body)
    assert math.isnan(json.loads(json.dumps(body))["at"])       # Python is the lenient reader
    with pytest.raises(ValueError):
        json.loads(json.dumps(body), parse_constant=_refuse)    # a strict one is not
