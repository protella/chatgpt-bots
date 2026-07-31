#!/usr/bin/env python3
"""Validate a `participation.jsonl` ledger against telemetry contract CV8.

WHY THIS EXISTS. The P2 live battery decides whether a scenario passed by reading this ledger, so
the ledger's structure is load-bearing evidence and not a debugging convenience. A join that
silently does not join turns a battery pass into a guess. This says so, by name, with a line
number, and with an exit code a script can believe.

WHY NO IMPORTS. It runs where the evidence is: a jsonl copied off a box with no venv, no repo and
no `config`. Stdlib only, deliberately — which means the vocabularies below are RESTATED from
`message_processor/participation_telemetry.py` rather than imported. That duplication is the price
of being runnable at all, so it is confined to ONE block: a contract bump edits the block and
`CONTRACT_VERSION`, and nothing else in this file knows a value.

WHAT "REQUIRED" MEANS HERE. The emitter's `record()` OMITS None-valued fields rather than writing
nulls, so absence is the normal representation of "not applicable" and a naive required-keys check
would cry wolf on every healthy ledger. A field is treated as mandatory below only when the
emitter writes it unconditionally (a plain `bool(...)`, a required keyword) or when the row is
meaningless without it — an unjoinable row is a defect, not a shrug. Each MANDATORY tuple carries
its reason.

VIOLATION vs WARNING. A violation means the contract was broken and someone has to look. A
warning means the FILE is incomplete in a way the contract predicts:
  * the last session has no `session_end` — a crash truncates the tail, so its unmatched
    `turn_start`s are missing records, not missing outcomes;
  * a session whose `session_start` is not in the input — the file rotated and we are reading a
    fragment, so an orphan join means the other half is in `participation.jsonl.1`;
  * a `gate_start` with no terminal — the emitter documents this as evidence about the SINK
    (a lost line, a crash with records queued), which analysis must count rather than fail on.

DUPLICATES ARE NOT A DEFECT. Compaction telemetry is delivered from an outbox that deletes a row
only after acknowledgement, so a crash between the two replays the line: the contract is
EXACTLY-ONCE BY IDENTITY, AT-LEAST-ONCE BY DELIVERY. Those events are counted by DISTINCT
`(crawl_id, attempt_seq, event_seq)`, and a replay differing only in its envelope is expected. One
identity carrying two different CANONICAL BODIES is the defect, and that is what gets reported.

Usage:
    python3 tools/participation_ledger_check.py logs/participation.jsonl [more.jsonl ...]

Exit: 0 clean, 1 violations found, 2 nothing readable to check.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# ===========================================================================================
# THE CONTRACT, RESTATED. The one block to edit on a contract bump. Mirrors
# message_processor/participation_telemetry.py — keep the names identical so a diff is readable.
# ===========================================================================================

CONTRACT_VERSION = 8
GATE_CONTRACT = "binary-v1"

# On every line, at every version the emitter has ever written... except that `gate_contract`
# arrived with v7, so the full envelope is only asserted on the lines we actually grade.
ENVELOPE_FIELDS = ("v", "at", "session", "gate_contract", "event")

# `visible_action.kind`, reused verbatim by `turn_outcome.kind` — a turn and its gate attempt
# describe the same room, and two vocabularies for one question would make the rows uncomparable.
KINDS = frozenset({
    "reply", "delivery_failed", "silence", "reaction_only", "detached", "queued",
    "interrupted", "error", "error_unhandled", "aborted", "empty", "none",
    "stale_suppressed",
})
TURN_SURFACES = frozenset({"channel", "dm"})
# turn_outcome.destinations[] — DestinationRecord.as_payload()
DESTINATION_STATES = frozenset({"observed", "committed"})
DESTINATION_KINDS = frozenset({"reply", "stream", "split", "post_to_thread", "reconciled"})
RECEIPT_OPS = frozenset({
    "register", "promote", "finalize", "demote", "transfer", "delete", "reconcile_finalize",
    "pending_resolve",
})
RECEIPT_STATES = frozenset({"absent", "in_flight", "finalized", "chrome"})
SNAPSHOT_OPS = frozenset({"read", "publish", "invalidate", "stale_retained", "build"})
MODEL_RESPONSE_STATUSES = frozenset({"ok", "error"})

# The two compaction ops that ride the TELEMETRY OUTBOX, and are therefore the only ones with an
# identity, a cardinality contract and a delivery order. `read`, `invalidate` and stale-retained
# copies are DIRECT WRITES and BEST-EFFORT BY DESIGN — a committed invalidation can still lose its
# ledger line to a crash in the gap. NO COMPLETENESS RULE IS IMPOSED ON THEM, here or anywhere:
# their durable truth is the database row, and a missing line costs a breadcrumb, not correctness.
OUTBOX_OPS = frozenset({"build", "publish"})
BUILD_STATUSES = frozenset({"ok", "failed", "discarded", "copied"})
BUILD_REASON_STATUSES = frozenset({"failed", "discarded"})
FIT_RESULTS = frozenset({"under_target", "under_trigger"})
# The CV8 envelope keys, stripped to recover the canonical body. Emission adds EXACTLY these three;
# `at` and `event` come from the body, because both must survive a replay.
WRAPPER_KEYS = ("v", "session", "gate_contract")
# The identity triple, in the PAYLOAD and not only in the outbox key — the row is deleted after
# acknowledgement, so an identity in dropped columns is one this checker cannot see.
IDENTITY_FIELDS = (("crawl_id", str), ("attempt_seq", int), ("event_seq", int))
# THE AUTHORITATIVE TOKEN NAMES: tokens_in / tokens_out, never input_tokens / output_tokens.
BUILD_FIELDS = (("model", str), ("tokens_in", int), ("tokens_out", int),
                ("cached_input_tokens", int), ("call_count", int), ("status", str))
# NO TOKEN FIELDS ON `publish` — the cost belongs to the attempt that produced the summary.
PUBLISH_FIELDS = (("snapshot_id", str), ("generation", int), ("boundary_ts", str),
                  ("fit_result", str), ("serializer_version", int))

# THE TERMINAL POPULATION IS THIS AND NOTHING ELSE. `turn_outcome` is not a terminal event (see
# the emitter's v8 note), so it must never reach the visible_action index — invariant 3 checks
# that separation rather than trusting this constant.
TERMINAL_EVENT = "visible_action"
TURN_EVENTS = frozenset({"turn_start", "turn_outcome", "stream_render", "model_response"})

# Mandatory fields, with the reason absence is a defect rather than a legal omission.
MANDATORY: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "turn_start": (
        ("turn_id", "the whole turn population joins on it; a row without one counts nothing"),
    ),
    "turn_outcome": (
        ("turn_id", "pairs the outcome to its start"),
        ("kind", "the event exists to say what the room saw"),
        ("destinations", "written even when empty — a silent turn and a lost record differ"),
        ("stream_build_present", "always a bool from the emitter; drives invariant 2"),
    ),
    "stream_render": (
        ("turn_id", "a build nobody's turn owns cannot be read as evidence about a turn"),
    ),
    "model_response": (
        ("turn_id", "unjoinable without it"),
        ("attempt_seq", "the per-turn contiguity contract is uncheckable without it"),
        ("status", "the one fact the event exists for"),
    ),
    "outbound_receipt": (
        ("channel_id", "a receipt names a message in a channel or it names nothing"),
        ("message_ts", "as above"),
        ("owner_turn_id", "the receipt lattice is keyed by owner; unattributable without it"),
        ("op", "required keyword on the emitter"),
        ("applied", "emitter writes bool(applied) unconditionally"),
    ),
    "compaction_snapshot": (
        ("op", "required keyword on the emitter"),
        ("channel_id", "a pointer belongs to one channel; every emitter has it"),
    ),
}
# Legal absences, stated so nobody re-adds them to MANDATORY without reading this:
#   outbound_receipt.prior_state/new_state/reason — omitted when None; `absent` is a real state,
#       so a missing one is "no transition recorded", checked only against the vocabulary.
#   compaction_snapshot.snapshot_id/generation/boundary_ts/serializer_version — read off the
#       pointer dict, which need not carry them on a DIRECT-WRITE op (`read`, `invalidate`,
#       stale-retained). The OUTBOX-routed ops are the exception and are graded against their
#       literal per-op schema: an outbox row is deleted once acknowledged, so a payload missing a
#       field is a fact nothing can recover afterwards.
#   turn_outcome.chars/error/H/attempt_id — chars is None on a turn that delivered no text,
#       error only on the four fail-closed codes, attempt_id only on a GATED turn.
#   model_response.model/token counts — a call that raised before the response has none.
#   destinations[].thread_root_ts/chars — nullable INSIDE the list: nested nulls survive,
#       because record() only strips top-level Nones.
DESTINATION_FIELDS = ("channel_id", "thread_root_ts", "first_ts", "state", "chars", "kind")
DESTINATION_NULLABLE = frozenset({"thread_root_ts", "chars"})


# ===========================================================================================
# findings
# ===========================================================================================

class Finding(NamedTuple):
    name: str
    file: str
    line: int          # 0 when the finding is about a file or a join rather than a line
    detail: str


class Row(NamedTuple):
    file: str
    file_index: int
    line: int
    obj: Dict[str, Any]

    @property
    def event(self) -> str:
        value = self.obj.get("event")
        return value if isinstance(value, str) else ""

    @property
    def session(self) -> str:
        value = self.obj.get("session")
        return value if isinstance(value, str) else ""


class Report:
    """Everything the run learned. Violations fail; warnings are printed and forgiven."""

    def __init__(self) -> None:
        self.violations: List[Finding] = []
        self.warnings: List[Finding] = []
        self.counts: Counter = Counter()
        self.events: Counter = Counter()
        self.files: List[str] = []
        self.sessions_seen: List[str] = []
        self.sessions_closed: set = set()
        self.sessions_opened: set = set()
        self.order: Dict[str, int] = {}

    def fail(self, name: str, row: Optional[Row], detail: str = "",
             file: str = "", line: int = 0) -> None:
        self.violations.append(Finding(name, row.file if row else file,
                                      row.line if row else line, detail))

    def warn(self, name: str, row: Optional[Row], detail: str = "",
             file: str = "", line: int = 0) -> None:
        self.warnings.append(Finding(name, row.file if row else file,
                                     row.line if row else line, detail))

    def file_order(self, path: str) -> int:
        return self.order.get(path, 0)


# ===========================================================================================
# reading
# ===========================================================================================

def _read_file(path: str, file_index: int, report: Report) -> List[Row]:
    """One JSON object per line. A bad line is a NAMED violation, never a traceback: the whole
    point of this tool is to survive the file it is complaining about."""
    rows: List[Row] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        report.fail("file_unreadable", None, str(exc), file=path)
        return rows
    for lineno, raw in enumerate(raw_lines, start=1):
        text = raw.strip()
        if not text:
            report.counts["blank_lines"] += 1
            continue
        report.counts["lines_read"] += 1
        try:
            obj = json.loads(text)
        except ValueError as exc:
            report.counts["unparsable_lines"] += 1
            report.fail("malformed_line", None, f"not JSON: {exc}", file=path, line=lineno)
            continue
        if not isinstance(obj, dict):
            report.counts["unparsable_lines"] += 1
            report.fail("malformed_line", None, f"not a JSON object: {type(obj).__name__}",
                        file=path, line=lineno)
            continue
        rows.append(Row(path, file_index, lineno, obj))
    return rows


def _grade_version(row: Row, report: Report) -> bool:
    """True when this line is a v8 line we are entitled to grade.

    Older lines are SKIPPED, not failed: the file rotates across a deploy, so a mixed file is
    the normal state of the world and grading v7 rows by v8 rules would invent violations out of
    correct history. A version we do not know is refused in the other direction — silently
    grading a FUTURE contract by these rules is how a checker starts lying.
    """
    version = row.obj.get("v")
    if not isinstance(version, int) or isinstance(version, bool) or version > CONTRACT_VERSION:
        report.counts["ungraded_lines"] += 1
        report.fail("unknown_contract_version", row, f"v={version!r} (this tool grades "
                                                     f"v{CONTRACT_VERSION})")
        return False
    if version < CONTRACT_VERSION:
        report.counts["skipped_older_contract"] += 1
        report.counts[f"skipped_v{version}"] += 1
        return False
    return True


# ===========================================================================================
# invariant 4 — the envelope
# ===========================================================================================

def _check_envelope(row: Row, report: Report) -> None:
    """Identity on every line. Without `session` a burst of odd rows after a deploy cannot be
    told from a change in the room; without `gate_contract` two incompatible gates get pooled."""
    for field in ENVELOPE_FIELDS:
        if field not in row.obj:
            report.fail("missing_envelope_field", row, f"no {field!r}")
    contract = row.obj.get("gate_contract")
    if "gate_contract" in row.obj and contract != GATE_CONTRACT:
        report.fail("bad_gate_contract", row,
                    f"{contract!r} on a v{CONTRACT_VERSION} line (expected {GATE_CONTRACT!r})")


# ===========================================================================================
# invariant 5 — payload spot checks
# ===========================================================================================

def _check_mandatory(row: Row, report: Report, name: str) -> None:
    for field, why in MANDATORY.get(row.event, ()):
        if field not in row.obj:
            report.fail(name, row, f"{row.event} has no {field!r} — {why}")


def _check_vocabulary(row: Row, report: Report, field: str, allowed: frozenset,
                      name: str, *, required: bool) -> None:
    """A typo does not fail the emitter — it invents a bucket and deflates the real one. That is
    exactly the failure a group-by cannot see, so it is checked here instead."""
    if field not in row.obj:
        if required:
            report.fail(name, row, f"{row.event} has no {field!r}")
        return
    value = row.obj.get(field)
    if value not in allowed:
        report.fail(name, row, f"{field}={value!r} not in {{{', '.join(sorted(allowed))}}}")


def _check_turn_start(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "turn_start_missing_field")
    if "surface" in row.obj and row.obj.get("surface") not in TURN_SURFACES:
        report.fail("turn_start_bad_surface", row, f"surface={row.obj.get('surface')!r}")


def _check_turn_outcome(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "turn_outcome_missing_field")
    _check_vocabulary(row, report, "kind", KINDS, "turn_outcome_bad_kind", required=False)
    if "stream_build_present" in row.obj \
            and not isinstance(row.obj.get("stream_build_present"), bool):
        report.fail("turn_outcome_missing_field", row,
                    f"stream_build_present={row.obj.get('stream_build_present')!r} is not a bool")
    if "destinations" not in row.obj:
        return
    destinations = row.obj.get("destinations")
    if not isinstance(destinations, list):
        report.fail("turn_outcome_destinations_not_list", row,
                    f"destinations is {type(destinations).__name__}")
        return
    for index, destination in enumerate(destinations):
        _check_destination(row, report, index, destination)


def _check_destination(row: Row, report: Report, index: int, destination: Any) -> None:
    """One place this turn's words landed. Nulls are legal INSIDE the list (nested Nones are not
    stripped), so the key must be present and only some of the values may be null."""
    where = f"destinations[{index}]"
    if not isinstance(destination, dict):
        report.fail("turn_outcome_destination_malformed", row,
                    f"{where} is {type(destination).__name__}, not an object")
        return
    for field in DESTINATION_FIELDS:
        if field not in destination:
            report.fail("turn_outcome_destination_malformed", row, f"{where} has no {field!r}")
        elif destination.get(field) is None and field not in DESTINATION_NULLABLE:
            report.fail("turn_outcome_destination_malformed", row, f"{where}.{field} is null")
    if "state" in destination and destination.get("state") not in DESTINATION_STATES:
        report.fail("turn_outcome_destination_bad_state", row,
                    f"{where}.state={destination.get('state')!r}")
    if "kind" in destination and destination.get("kind") not in DESTINATION_KINDS:
        report.fail("turn_outcome_destination_bad_kind", row,
                    f"{where}.kind={destination.get('kind')!r}")


def _check_stream_render(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "stream_render_missing_turn_id")


def _check_model_response(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "model_response_missing_field")
    _check_vocabulary(row, report, "status", MODEL_RESPONSE_STATUSES,
                      "model_response_bad_status", required=False)
    sequence = row.obj.get("attempt_seq")
    if "attempt_seq" in row.obj and (not isinstance(sequence, int) or isinstance(sequence, bool)):
        report.fail("model_response_missing_field", row,
                    f"attempt_seq={sequence!r} is not an int")


def _check_outbound_receipt(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "outbound_receipt_missing_field")
    _check_vocabulary(row, report, "op", RECEIPT_OPS, "outbound_receipt_bad_op", required=False)
    for field in ("prior_state", "new_state"):
        _check_vocabulary(row, report, field, RECEIPT_STATES, "outbound_receipt_bad_state",
                          required=False)
    applied = row.obj.get("applied")
    if "applied" in row.obj and not isinstance(applied, bool):
        # A truthy string here would make every "did it land?" count silently correct-looking.
        report.fail("outbound_receipt_applied_not_bool", row, f"applied={applied!r}")
    if applied is True and "new_state" not in row.obj:
        report.fail("outbound_receipt_applied_without_new_state", row,
                    "a transition that applied must say what state it produced")
    if applied is False and "reason" not in row.obj:
        # Not a contract break, but `applied=false` with no reason is the row that answers
        # nothing: the lattice refused and the record does not say why.
        report.warn("outbound_receipt_refused_without_reason", row, "applied=false, no reason")


def _typed(value: Any, kind: type) -> bool:
    """Present and of the declared type. A bool is refused where an int is required: it is an int
    to Python and a different fact to every reader of the ledger."""
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, kind)


def _check_compaction_snapshot(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "compaction_snapshot_missing_field")
    _check_vocabulary(row, report, "op", SNAPSHOT_OPS, "compaction_snapshot_bad_op",
                      required=False)
    op = row.obj.get("op")
    if op not in OUTBOX_OPS:
        return   # a direct-write op: no identity, no schema beyond the pointer, no completeness
    for field, kind in IDENTITY_FIELDS:
        if not _typed(row.obj.get(field), kind):
            report.fail("compaction_snapshot_missing_identity", row,
                        f"op={op} has {field}={row.obj.get(field)!r} — the dedup key is the "
                        f"payload's (crawl_id, attempt_seq, event_seq)")
    for field, kind in (BUILD_FIELDS if op == "build" else PUBLISH_FIELDS):
        if not _typed(row.obj.get(field), kind):
            report.fail("compaction_snapshot_missing_field", row,
                        f"op={op} has {field}={row.obj.get(field)!r}, expected {kind.__name__}")
    event_seq = row.obj.get("event_seq")
    # THE ROUTING TABLE: a build is ALWAYS at 0, a publish NEVER is. A publish emitted at 0 would
    # be acknowledged and deleted, permanently breaking the order the sequence exists to keep.
    if isinstance(event_seq, int) and not isinstance(event_seq, bool):
        if op == "build" and event_seq != 0:
            report.fail("compaction_snapshot_misplaced_op", row,
                        f"op=build at event_seq={event_seq} (a build is always at 0)")
        if op == "publish" and event_seq == 0:
            report.fail("compaction_snapshot_misplaced_op", row,
                        "op=publish at event_seq=0 (the build belongs there)")
    if op == "build":
        _check_vocabulary(row, report, "status", BUILD_STATUSES,
                          "compaction_snapshot_bad_status", required=False)
        needs_reason = row.obj.get("status") in BUILD_REASON_STATUSES
        if needs_reason and "reason" not in row.obj:
            report.fail("compaction_snapshot_reason_rule", row,
                        f"status={row.obj.get('status')!r} with no reason — a failed generation "
                        f"must not be invisible")
        if not needs_reason and "reason" in row.obj and row.obj.get("status") in BUILD_STATUSES:
            report.fail("compaction_snapshot_reason_rule", row,
                        f"status={row.obj.get('status')!r} carries a reason")
    else:
        _check_vocabulary(row, report, "fit_result", FIT_RESULTS,
                          "compaction_snapshot_bad_fit_result", required=False)
        present = [f for f in ("tokens_in", "tokens_out", "cached_input_tokens", "call_count")
                   if f in row.obj]
        if present:
            # Double-counting waiting to happen: any aggregate summing the ledger would charge
            # the attempt's cost twice.
            report.fail("compaction_publish_carries_tokens", row,
                        f"op=publish carries {present} — the cost belongs to its build")


PAYLOAD_CHECKS = {
    "turn_start": _check_turn_start,
    "turn_outcome": _check_turn_outcome,
    "stream_render": _check_stream_render,
    "model_response": _check_model_response,
    "outbound_receipt": _check_outbound_receipt,
    "compaction_snapshot": _check_compaction_snapshot,
}


# ===========================================================================================
# invariants 1-3 — the joins
# ===========================================================================================

class Joins:
    """The indexes the three join invariants are computed from.

    `terminals` is built from `visible_action` AND NOTHING ELSE. That is invariant 3's actual
    content: `turn_outcome` describes a turn, `visible_action` closes a gate attempt, and the two
    populations answer different questions. Their denominators are reported side by side and are
    never divided into each other.
    """

    def __init__(self) -> None:
        self.turn_starts: Dict[str, List[Row]] = defaultdict(list)
        self.turn_outcomes: Dict[str, List[Row]] = defaultdict(list)
        self.stream_renders: Dict[str, List[Row]] = defaultdict(list)
        self.model_responses: Dict[str, List[Row]] = defaultdict(list)
        self.gate_starts: Dict[str, List[Row]] = defaultdict(list)
        self.terminals: Dict[str, List[Row]] = defaultdict(list)
        self.terminal_rows: List[Row] = []

    def add(self, row: Row) -> None:
        event = row.event
        if event == TERMINAL_EVENT:
            self.terminal_rows.append(row)
            attempt_id = row.obj.get("attempt_id")
            if isinstance(attempt_id, str) and attempt_id:
                self.terminals[attempt_id].append(row)
            return
        if event == "gate_start":
            attempt_id = row.obj.get("attempt_id")
            if isinstance(attempt_id, str) and attempt_id:
                self.gate_starts[attempt_id].append(row)
            return
        turn_id = row.obj.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return
        if event == "turn_start":
            self.turn_starts[turn_id].append(row)
        elif event == "turn_outcome":
            self.turn_outcomes[turn_id].append(row)
        elif event == "stream_render":
            self.stream_renders[turn_id].append(row)
        elif event == "model_response":
            self.model_responses[turn_id].append(row)


def _first_at(row: Row, other: Row) -> str:
    """Point at the earlier row without repeating a long path the reader already has."""
    return f"line {other.line}" if other.file == row.file else f"{other.file}:{other.line}"


def _fragment_sessions(report: Report) -> set:
    """Sessions whose `session_start` is NOT in the input — we are reading a rotated fragment, so
    the other half of every join is in `participation.jsonl.1`. Their orphans are warnings."""
    return {session for session in report.sessions_seen
            if session not in report.sessions_opened and session}


def _tail_session(rows: List[Row]) -> str:
    """The session the file ends in. `session_end` rides the same queue as everything else, so
    its absence HERE is the one signal that a run was cut off rather than stopped."""
    for row in reversed(rows):
        if row.session:
            return row.session
    return ""


def _check_turn_pairing(joins: Joins, report: Report, *,
                        tail_session: str, fragments: set) -> None:
    """INVARIANT 1: exactly one turn_outcome per turn_start, joined on turn_id.

    Three different failures, reported separately because they have three different causes: an
    unmatched start is a turn that ended without saying what it did (a raise past the outer
    finally); an unmatched outcome is an outcome for a turn that never opened (a second emitter,
    or a rotated head); a duplicate is two emitters believing they own the end of the turn —
    exactly the bug `finish_attempt`'s closed-flag was added to stop on the gate side.
    """
    open_tail = tail_session and tail_session not in report.sessions_closed
    for turn_id, starts in sorted(joins.turn_starts.items()):
        if len(starts) > 1:
            report.fail("turn_start_duplicate", starts[-1],
                        f"turn_id={turn_id} opened {len(starts)}x "
                        f"(first at {_first_at(starts[-1], starts[0])})")
        outcomes = joins.turn_outcomes.get(turn_id, [])
        if len(outcomes) > 1:
            report.fail("turn_outcome_duplicate", outcomes[-1],
                        f"turn_id={turn_id} has {len(outcomes)} outcomes "
                        f"(first at {_first_at(outcomes[-1], outcomes[0])})")
        elif not outcomes:
            row = starts[0]
            if open_tail and row.session == tail_session:
                # A crash truncates the file. The turn may well have finished; its record did not.
                report.warn("turn_outcome_missing", row,
                            f"turn_id={turn_id} — tail of session {row.session} with no "
                            f"session_end (records lost, not outcomes)")
                report.counts["truncated_tail_unmatched_turn_start"] += 1
            else:
                report.fail("turn_outcome_missing", row, f"turn_id={turn_id} never ended")
    for turn_id, outcomes in sorted(joins.turn_outcomes.items()):
        if turn_id in joins.turn_starts:
            continue
        row = outcomes[0]
        detail = f"turn_id={turn_id} has no turn_start"
        if row.session in fragments:
            report.warn("turn_outcome_orphan", row, f"{detail} (session head not in input)")
        else:
            report.fail("turn_outcome_orphan", row, detail)


def _check_stream_render_joins(joins: Joins, report: Report, *, fragments: set) -> None:
    """INVARIANT 2: a build and its turn agree, in both directions.

    `stream_build_present` is the turn's own claim that the room was rendered. If the claim is
    true and no build was written, the turn's `kind` is being read as a judgment about a room
    that may never have been assembled; if the claim is false and a build exists, the turn
    rendered and then reported that it had not. Either way the fail-closed accounting is wrong.
    """
    for turn_id, renders in sorted(joins.stream_renders.items()):
        if turn_id in joins.turn_starts:
            continue
        row = renders[0]
        detail = f"turn_id={turn_id} has no turn_start"
        if row.session in fragments:
            report.warn("stream_render_orphan_turn", row, f"{detail} (session head not in input)")
        else:
            report.fail("stream_render_orphan_turn", row, detail)
    for turn_id, outcomes in sorted(joins.turn_outcomes.items()):
        row = outcomes[0]
        claimed = row.obj.get("stream_build_present")
        renders = joins.stream_renders.get(turn_id, [])
        if claimed is True and not renders:
            detail = f"turn_id={turn_id} says stream_build_present but rendered nothing"
            if row.session in fragments:
                # The build is written BEFORE the outcome, so a head-truncated file legitimately
                # has the outcome without it. Only the other direction stays a violation.
                report.warn("stream_render_absent_for_build", row,
                            f"{detail} (session head not in input)")
            else:
                report.fail("stream_render_absent_for_build", row, detail)
        elif claimed is False and renders:
            report.fail("stream_render_without_build", row,
                        f"turn_id={turn_id} says no stream build, but {len(renders)} "
                        f"stream_render(s) exist (first at {_first_at(row, renders[0])})")


def _check_model_attempts(joins: Joins, report: Report) -> None:
    """`attempt_seq` starts at 1 and is contiguous per turn_id — the sequence is what makes "this
    turn cost four API calls" true rather than a lower bound.

    Only checked for turns whose `turn_start` we can see: a rotated file can cut a turn's head
    off, and a sequence that legitimately starts at 3 must not be graded as a gap.
    """
    for turn_id, responses in sorted(joins.model_responses.items()):
        if turn_id not in joins.turn_starts:
            report.warn("model_response_orphan_turn", responses[0],
                        f"turn_id={turn_id} has no turn_start (sequence not graded)")
            continue
        sequence = [r.obj.get("attempt_seq") for r in responses]
        numbers = [n for n in sequence if isinstance(n, int) and not isinstance(n, bool)]
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        if duplicates:
            report.fail("model_response_attempt_seq_duplicate", responses[-1],
                        f"turn_id={turn_id} repeats attempt_seq {duplicates}")
        expected = list(range(1, len(set(numbers)) + 1))
        if sorted(set(numbers)) != expected:
            report.fail("model_response_attempt_seq_not_contiguous", responses[0],
                        f"turn_id={turn_id} has attempt_seq {sorted(numbers)}, expected "
                        f"{expected}")


# ===========================================================================================
# invariant 6 — the compaction outbox: exactly-once BY IDENTITY, at-least-once BY DELIVERY
# ===========================================================================================

def canonical_body_bytes(body: Dict[str, Any]) -> bytes:
    """THE ONE SERIALIZER, restated from message_processor/participation_telemetry.py — the same
    call, byte for byte. Two serializations would make identical bodies compare unequal over key
    order, whitespace or `\\uXXXX` escaping alone, which is the entire comparison."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def extract_canonical_body(line: str) -> bytes:
    """Recover the body from a flattened JSONL line: parse, remove EXACTLY the three wrapper keys,
    re-serialize canonically. The body is flattened into the emitted object, so its original bytes
    are not recoverable by parsing alone."""
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"ledger line is not a JSON object: {type(obj).__name__}")
    return body_bytes(obj)


def body_bytes(obj: Dict[str, Any]) -> bytes:
    """The canonical body of an already-parsed line."""
    return canonical_body_bytes({k: v for k, v in obj.items() if k not in WRAPPER_KEYS})


def _check_compaction_outbox(rows: List[Row], report: Report) -> None:
    """THE HONEST CONTRACT: EXACTLY-ONCE BY IDENTITY, AT-LEAST-ONCE BY DELIVERY.

    A crash between emit and delete replays the row, so DUPLICATE LINES ARE EXPECTED and are not
    an error — this counts DISTINCT `(crawl_id, attempt_seq, event_seq)` triples, not lines. The
    replay's ENVELOPE legitimately differs (`session` above all: it is a different process), which
    is why every comparison here is over the CANONICAL BODY. A checker comparing whole JSONL lines
    would report every honest replay as a conflict.

    BUT one triple mapping to DIFFERING BODIES IS A CONTRACT VIOLATION and is reported: two
    different events wearing one identity — a payload rebuilt from changed state, or an identity
    reused across attempts — is a real defect hiding inside the very mechanism that makes
    duplicates safe.

    And the cardinality rule the outbox exists for: AT LEAST ONE `build` PRECEDES ANY `publish` of
    the same attempt. The drainer emits in `outbox_seq` order, so a publish that arrives first
    means the ordering guarantee broke, not that the file is odd.
    """
    seen: Dict[Tuple[Any, Any, Any], Tuple[Row, bytes]] = {}
    builds: Dict[Tuple[Any, Any], int] = {}
    publishes: List[Tuple[int, Row, Tuple[Any, Any]]] = []
    for order, row in enumerate(rows):
        if row.event != "compaction_snapshot":
            continue
        op = row.obj.get("op")
        if op not in OUTBOX_OPS:
            continue
        triple = tuple(row.obj.get(field) for field, _ in IDENTITY_FIELDS)
        if not all(_typed(row.obj.get(f), k) for f, k in IDENTITY_FIELDS):
            continue   # already failed as a missing identity; it cannot be deduped
        body = body_bytes(row.obj)
        first = seen.get(triple)
        if first is None:
            seen[triple] = (row, body)
        elif first[1] != body:
            report.fail("compaction_outbox_body_conflict", row,
                        f"identity {triple} carries two different canonical bodies "
                        f"(first at {_first_at(row, first[0])})")
        else:
            report.counts["compaction_outbox_replayed_lines"] += 1
            continue   # an honest replay: same identity, same body, a different session
        attempt = (triple[0], triple[1])
        if op == "build":
            builds.setdefault(attempt, order)
        else:
            publishes.append((order, row, attempt))
    for order, row, attempt in publishes:
        build_order = builds.get(attempt)
        if build_order is None:
            report.fail("compaction_publish_without_build", row,
                        f"attempt {attempt} published with no op=build in the ledger")
        elif build_order > order:
            report.fail("compaction_publish_before_build", row,
                        f"attempt {attempt} published before its build "
                        f"(build at {_first_at(row, rows[build_order])})")
    report.counts["compaction_outbox_events"] = len(seen)
    report.counts["compaction_builds"] = len(builds)


def _check_terminal_invariant(joins: Joins, report: Report, rows: List[Row], *,
                              fragments: set) -> None:
    """INVARIANT 3: at most one `visible_action` per attempt_id, and every one of them has a
    `gate_start`.

    The one-terminal rule is the reason any participation rate can be believed: the outer gate
    and the responder both think they own the end of a turn, and two terminals double-count one
    message while zero loses it. `gate_start` is the DENOMINATOR — a terminal with no start
    describes an attempt that, as far as the file is concerned, never entered the gate.

    A `gate_start` with no terminal is NOT failed: the emitter documents it as evidence about the
    sink (a lost line, a crash with records still queued), so it is counted and reported.
    """
    # The two populations, kept apart on purpose — and checked, not assumed. This can only fire
    # if someone teaches the indexer that a turn event closes an attempt.
    independent = [r for r in rows if r.event == TERMINAL_EVENT]
    leaked = [r for r in joins.terminal_rows if r.event in TURN_EVENTS]
    if leaked or len(independent) != len(joins.terminal_rows):
        report.fail("turn_event_in_terminal_population", leaked[0] if leaked else None,
                    f"{len(joins.terminal_rows)} rows counted as terminals but "
                    f"{len(independent)} visible_action rows exist")
    for row in joins.terminal_rows:
        attempt_id = row.obj.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            report.fail("visible_action_missing_attempt_id", row,
                        f"kind={row.obj.get('kind')!r} closes no attempt")
    for attempt_id, terminals in sorted(joins.terminals.items()):
        if len(terminals) > 1:
            report.fail("visible_action_duplicate", terminals[-1],
                        f"attempt_id={attempt_id} closed {len(terminals)}x "
                        f"(first at {_first_at(terminals[-1], terminals[0])})")
        if attempt_id not in joins.gate_starts:
            row = terminals[0]
            detail = f"attempt_id={attempt_id} has no gate_start"
            if row.session in fragments:
                report.warn("visible_action_orphan_attempt", row,
                            f"{detail} (session head not in input)")
            else:
                report.fail("visible_action_orphan_attempt", row, detail)
    for attempt_id, starts in sorted(joins.gate_starts.items()):
        if attempt_id not in joins.terminals:
            report.counts["gate_start_without_terminal"] += 1
            report.warn("gate_start_without_terminal", starts[0],
                        f"attempt_id={attempt_id} — evidence about the sink, not the gate")


# ===========================================================================================
# the run
# ===========================================================================================

def check_paths(paths: List[str]) -> Report:
    report = Report()
    rows: List[Row] = []
    for index, path in enumerate(paths):
        report.files.append(path)
        report.order[path] = index
        rows.extend(_read_file(path, index, report))
    if not rows and not report.counts["lines_read"] \
            and not any(f.name == "file_unreadable" for f in report.violations):
        # A battery must not read "no violations" off a ledger with nothing in it.
        report.fail("empty_ledger", None, "no lines to check", file=paths[0] if paths else "")

    graded: List[Row] = []
    for row in rows:
        if not _grade_version(row, report):
            continue
        graded.append(row)
        report.counts["graded_lines"] += 1
        event = row.event
        report.events[event or "<missing>"] += 1
        if row.session and row.session not in report.sessions_seen:
            report.sessions_seen.append(row.session)
        if event == "session_start":
            report.sessions_opened.add(row.session)
        elif event == "session_end":
            report.sessions_closed.add(row.session)
        _check_envelope(row, report)
        if not event:
            continue
        checker = PAYLOAD_CHECKS.get(event)
        if checker is not None:
            checker(row, report)

    joins = Joins()
    for row in graded:
        joins.add(row)

    fragments = _fragment_sessions(report)
    tail = _tail_session(graded)
    report.counts["gate_start"] = len(joins.gate_starts)
    report.counts["turn_start"] = len(joins.turn_starts)
    report.counts["visible_action"] = len(joins.terminal_rows)
    report.counts["turn_outcome"] = sum(len(v) for v in joins.turn_outcomes.values())
    _check_turn_pairing(joins, report, tail_session=tail, fragments=fragments)
    _check_stream_render_joins(joins, report, fragments=fragments)
    _check_model_attempts(joins, report)
    _check_terminal_invariant(joins, report, graded, fragments=fragments)
    _check_compaction_outbox(graded, report)

    report.violations.sort(key=lambda f: (report.file_order(f.file), f.line, f.name))
    report.warnings.sort(key=lambda f: (report.file_order(f.file), f.line, f.name))
    report.counts["sessions"] = len(report.sessions_seen)
    report.counts["tail_session_open"] = int(bool(tail and tail not in report.sessions_closed))
    report.counts["fragment_sessions"] = len(fragments)
    return report


# ===========================================================================================
# output
# ===========================================================================================

def _where(finding: Finding) -> str:
    if finding.file and finding.line:
        return f"{finding.file}:{finding.line}"
    return finding.file or "-"


def human_report(report: Report, out) -> None:
    counts, events = report.counts, report.events
    print(f"participation ledger check — contract CV{CONTRACT_VERSION} ({GATE_CONTRACT})",
          file=out)
    print(f"  files    : {', '.join(report.files) or '-'}", file=out)
    print(f"  lines    : {counts['lines_read']} read, {counts['graded_lines']} graded at "
          f"v{CONTRACT_VERSION}, {counts['skipped_older_contract']} skipped (older contract), "
          f"{counts['unparsable_lines']} unparsable", file=out)
    older = sorted(k for k in counts if k.startswith("skipped_v"))
    if older:
        print("  skipped  : " + ", ".join(f"{k[8:]}={counts[k]}" for k in older), file=out)
    print(f"  sessions : {counts['sessions']} ({len(report.sessions_closed)} with session_end"
          + (", tail session OPEN — crash or still running" if counts["tail_session_open"]
             else "")
          + (f", {counts['fragment_sessions']} head-truncated" if counts["fragment_sessions"]
             else "") + ")", file=out)
    # Two populations, two denominators. Printed together and labelled because dividing one into
    # the other is the single easiest way to publish a wrong participation rate.
    print("  denominators (DO NOT divide one into the other):", file=out)
    print(f"      gate_start   {counts['gate_start']:>6}  gate attempts  "
          f"(visible_action: {counts['visible_action']})", file=out)
    print(f"      turn_start   {counts['turn_start']:>6}  channel turns  "
          f"(turn_outcome: {counts['turn_outcome']})", file=out)
    if counts["compaction_outbox_events"]:
        print(f"  compaction: {counts['compaction_outbox_events']} distinct outbox identities "
              f"({counts['compaction_builds']} builds, "
              f"{counts['compaction_outbox_replayed_lines']} replayed lines — expected)", file=out)
    if events:
        print("  events   : " + ", ".join(f"{name}={events[name]}"
                                          for name in sorted(events)), file=out)
    if report.warnings:
        print(f"\nWARNINGS ({len(report.warnings)}) — expected incompleteness, not failures:",
              file=out)
        for finding in report.warnings:
            print(f"  {_where(finding)}  {finding.name}: {finding.detail}", file=out)
    if report.violations:
        print(f"\nVIOLATIONS ({len(report.violations)}):", file=out)
        for finding in report.violations:
            print(f"  {_where(finding)}  {finding.name}: {finding.detail}", file=out)
        tally = Counter(f.name for f in report.violations)
        print("  by name  : " + ", ".join(f"{n}={c}" for n, c in sorted(tally.items())),
              file=out)
    verdict = "FAIL" if report.violations else "PASS"
    print(f"\nverdict: {verdict} ({len(report.violations)} violation(s), "
          f"{len(report.warnings)} warning(s))", file=out)


def json_report(report: Report) -> Dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "gate_contract": GATE_CONTRACT,
        "ok": not report.violations,
        "verdict": "fail" if report.violations else "pass",
        "files": list(report.files),
        "counts": dict(sorted(report.counts.items())),
        "events": dict(sorted(report.events.items())),
        "denominators": {"gate_start": report.counts["gate_start"],
                         "visible_action": report.counts["visible_action"],
                         "turn_start": report.counts["turn_start"],
                         "turn_outcome": report.counts["turn_outcome"]},
        "sessions": {"seen": list(report.sessions_seen),
                     "closed": sorted(report.sessions_closed),
                     "opened": sorted(report.sessions_opened)},
        "violation_counts": dict(sorted(Counter(f.name
                                                for f in report.violations).items())),
        "violations": [f._asdict() for f in report.violations],
        "warnings": [f._asdict() for f in report.warnings],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="participation_ledger_check",
        description=f"Validate a participation.jsonl ledger against contract "
                    f"CV{CONTRACT_VERSION}. Exits 1 on any violation.")
    parser.add_argument("paths", nargs="+", metavar="LEDGER",
                        help="participation.jsonl files (rotations too, oldest first)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="machine-readable report on stdout (same verdict)")
    parser.add_argument("--quiet", action="store_true",
                        help="violations and verdict only; nothing at all when clean "
                             "(no effect with --json)")
    args = parser.parse_args(argv)

    report = check_paths(args.paths)
    if args.as_json:
        json.dump(json_report(report), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.quiet:
        for finding in report.violations:
            print(f"{_where(finding)}  {finding.name}: {finding.detail}")
        if report.violations:
            print(f"verdict: FAIL ({len(report.violations)} violation(s))")
    else:
        human_report(report, sys.stdout)

    if any(f.name == "file_unreadable" for f in report.violations):
        return 2   # pointed at something we could not open: a usage problem, not a ledger one
    return 1 if report.violations else 0


if __name__ == "__main__":
    sys.exit(main())
