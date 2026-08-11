#!/usr/bin/env python3
"""Validate a `participation.jsonl` ledger against telemetry contract CV10.

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

Usage:
    python3 tools/participation_ledger_check.py logs/participation.jsonl [more.jsonl ...]

Exit: 0 clean, 1 violations found, 2 nothing readable to check.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# ===========================================================================================
# THE CONTRACT, RESTATED. The one block to edit on a contract bump. Mirrors
# message_processor/participation_telemetry.py — keep the names identical so a diff is readable.
# ===========================================================================================

CONTRACT_VERSION = 10
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
# turn_outcome.destinations[] — DestinationRecord.as_payload(). `correction_announcement`
# (v10) is the executor-synthesized disclosure post of an edit_own_message transaction,
# recorded as a committed destination (DEST_KIND_CORRECTION_ANNOUNCEMENT in turn_runtime).
DESTINATION_STATES = frozenset({"observed", "committed"})
DESTINATION_KINDS = frozenset({"reply", "stream", "split", "post_to_thread", "reconciled",
                               "correction_announcement"})
RECEIPT_OPS = frozenset({
    "register", "promote", "finalize", "demote", "transfer", "delete", "reconcile_finalize",
    "pending_resolve",
})
RECEIPT_STATES = frozenset({"absent", "in_flight", "finalized", "chrome"})
MODEL_RESPONSE_STATUSES = frozenset({"ok", "error"})
# v9 — how one reconsideration runner invocation ended, and the ALL EIGHT §4f error subtypes an
# `error_dropped` may carry. Posted outcomes assert physical Slack acceptance, nothing more.
RECONSIDER_OUTCOMES = frozenset({
    "posted_asis", "posted_revised", "skipped", "fuse_dropped", "error_dropped", "cancelled",
})
RECONSIDER_ERRORS = frozenset({
    "context_rebuild", "model_failure", "admission_overflow", "delivery_failed",
    "epoch_invalidated", "guard_rearm_failed", "request_build", "delivery_exception",
})

# The LITERAL key grammar of the two v9 events: envelope plus these and nothing else. An unknown
# key is failed rather than ignored — a field the emitter invented and no reader knows about is
# how a contract drifts out from under the tool that is supposed to be grading it.
RECONSIDER_START_FIELDS = frozenset(ENVELOPE_FIELDS) | {
    "turn_id", "channel_id", "trigger_ts", "attempt_id", "pass", "scope", "observed_latest_ts",
    "model_attempt_seq",
}
RECONSIDER_OUTCOME_FIELDS = frozenset(ENVELOPE_FIELDS) | {
    "turn_id", "channel_id", "trigger_ts", "attempt_id", "outcome", "passes", "forced", "error",
}
# The nested `turn_outcome.reconsider` copy — ReconsiderFacts.as_payload() verbatim, so its key
# grammar is closed too: these four and nothing else, with inapplicable keys OMITTED, never null.
RECONSIDER_NESTED_FIELDS = frozenset({"outcome", "passes", "forced", "error"})

# v10 — turn_outcome.edits[], one entry per EditRecord (Docs/specs/EDIT_OWN_MESSAGE.md §7,
# lifecycle per §11.6). The nested grammar is CLOSED and null-free, and the lifecycle is
# SOUND: a record exists only once the disclosure was accepted (the shielded transaction runs
# to completion), so BOTH states always carry `announcement_ts`; `committed` (disclosure AND
# update landed) carries no `error`, `announcement_only` (the disclosure landed, the update
# did not) always carries one. `error` is therefore the only conditional key, and only on
# `committed` may it be absent.
EDIT_FIELDS = frozenset({"channel_id", "target_ts", "announcement_ts", "state", "error"})
EDIT_REQUIRED_FIELDS = ("channel_id", "target_ts", "state")
EDIT_STATES = frozenset({"announcement_only", "committed"})

# THE TERMINAL POPULATION IS THIS AND NOTHING ELSE. `turn_outcome` is not a terminal event (see
# the emitter's v8 note), so it must never reach the visible_action index — invariant 3 checks
# that separation rather than trusting this constant.
TERMINAL_EVENT = "visible_action"
TURN_EVENTS = frozenset({"turn_start", "turn_outcome", "stream_render", "model_response",
                         "reconsider_start", "reconsider_outcome"})

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
        ("edits", "v10: written even when empty — a turn that edited nothing and a turn whose "
                  "edit records were lost are not the same fact"),
    ),
    "stream_render": (
        ("turn_id", "a build nobody's turn owns cannot be read as evidence about a turn"),
    ),
    "model_response": (
        ("turn_id", "unjoinable without it"),
        ("attempt_seq", "the per-turn contiguity contract is uncheckable without it"),
        ("status", "the one fact the event exists for"),
    ),
    "reconsider_start": (
        ("turn_id", "the primary turn-population join; a pass nobody's turn owns counts nothing"),
        ("channel_id", "the runner always has the conversation key"),
        ("trigger_ts", "as above"),
        ("pass", "the contiguity contract is uncheckable without it"),
        ("scope", "the full three-part suppressing scope is the pass's evidence"),
        ("observed_latest_ts", "the message the pass is reconsidering against"),
    ),
    "reconsider_outcome": (
        ("turn_id", "pairs the outcome to its passes and its turn"),
        ("channel_id", "the runner always has the conversation key"),
        ("trigger_ts", "as above"),
        ("outcome", "the one fact the event exists for"),
        ("passes", "the started-pass accounting is unreadable without it"),
    ),
    "stale_send": (
        ("turn_id", "v9 rows carry it; the pass-count invariant counts suppression events per "
                    "turn, and a row without one counts toward no turn"),
    ),
    "outbound_receipt": (
        ("channel_id", "a receipt names a message in a channel or it names nothing"),
        ("message_ts", "as above"),
        ("owner_turn_id", "the receipt lattice is keyed by owner; unattributable without it"),
        ("op", "required keyword on the emitter"),
        ("applied", "emitter writes bool(applied) unconditionally"),
    ),
}
# Legal absences, stated so nobody re-adds them to MANDATORY without reading this:
#   outbound_receipt.prior_state/new_state/reason — omitted when None; `absent` is a real state,
#       so a missing one is "no transition recorded", checked only against the vocabulary.
#   turn_outcome.chars/error/H/attempt_id — chars is None on a turn that delivered no text,
#       error only on the four fail-closed codes (TURN_ERRORS below: stream_data_invalid,
#       stream_over_budget, history_fetch_failed, origin_fetch_failed), attempt_id only on a
#       GATED turn.
#   model_response.model/token counts — a call that raised before the response has none.
#   reconsider_start.attempt_id/model_attempt_seq — ungated turns (channel or DM) mint no
#       attempt, and a failed attempt-sink open omits the seq (telemetry never blocks the model
#       call). DM turns emit these events too — see `_check_reconsiderations`.
#   reconsider_outcome.attempt_id — as above; `forced` rides only on posted outcomes and
#       `error` only on `error_dropped` — their PRESENCE elsewhere is the violation, absence
#       is the normal encoding.
#   stale_send.turn_id — NOT a legal absence: v9 rows REQUIRE it (MANDATORY above), because the
#       pass-count invariant reads the per-turn suppression count. Older rows stay exempt for
#       free — _grade_version skips anything below v9. A row carrying a turn_id with no
#       turn_start to join is TOLERATED at the JOIN level (a DM ledger written before DM turns
#       joined the population has exactly that shape); it is the missing FIELD that fails, not
#       the missing join.
#   turn_outcome.reconsider — present only when a reconsideration ran; nested keys follow the
#       reconsider_outcome rules (forced posted-only, error error_dropped-only, no nulls).
#   turn_outcome.edits[].announcement_ts — ALWAYS present, in BOTH states: announcement-first
#       means a record exists only once the disclosure was accepted, so there is no legal
#       absence. `error` is the entry's only conditional key (announcement_only always carries
#       one, committed never); the entry itself never carries an explicit null anywhere.
#   destinations[].thread_root_ts/chars — nullable INSIDE the list: nested nulls survive,
#       because record() only strips top-level Nones.
DESTINATION_FIELDS = ("channel_id", "thread_root_ts", "first_ts", "state", "chars", "kind")
DESTINATION_NULLABLE = frozenset({"thread_root_ts", "chars"})

# ---------------------------------------------------------------- the stream_render contract
# Enumerated rather than inferred: a checker author who has to work these out from the emitter
# will work out different ones, and the field set is the whole evidence base for how the room
# was rendered.

# THE HASH RULE — 64 lowercase hex, with ONE exception. `capability_profile_hash` may also be
# empty: the builder's own signature defaults it that way, so every caller that does not resolve
# a thread config — the diagnostic probe, any utility build — legitimately emits "". The other
# six are computed from the stream itself and have no caller-omitted path, so empty is a defect.
STREAM_RENDER_HASHES = ("stream_sha256", "union_sha256", "serializer_config_hash",
                        "sidecar_versions_hash", "actor_map_hash", "receipts_membership_hash",
                        "capability_profile_hash")
HASH_MAY_BE_EMPTY = frozenset({"capability_profile_hash"})
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

# THE COUNT RULE — non-negative ints, and its scope is exact.
STREAM_RENDER_COUNTS = ("byte_count", "origin_byte_count", "message_count", "origin_count",
                        "candidate_count", "root_count", "orphan_root_count",
                        "receipts_included_count", "receipts_excluded_count",
                        "history_pages", "reply_pages", "origin_pages")
# VERSION ints sit OUTSIDE it: they are identifiers that happen to be numbers, and a bound on
# them would be a bound on how many times the format may change.
STREAM_RENDER_VERSIONS = ("selection_version", "serializer_version")
STREAM_RENDER_BOOLS = ("reselected", "anchor_advanced")
# Strings whose EMPTY VALUE IS A VALUE, not an absence: "" is the empty-floor sentinel, and for
# the inventory it means the row is absent.
STREAM_RENDER_STRINGS = ("channel_id", "H", "periphery_floor_ts", "inventory_start_ts")
INVENTORY_STATES = frozenset({"absent", "cold", "warm", "limited_retention", "limited_depth",
                              "unavailable"})
# Every field above is MANDATORY on every row. `origin_thread_ts` and `trigger_ts` are the only
# optional ones — a turn with no origin root, or no trigger, emits neither. Absence and None are
# ONE case here, because record() omits None-valued fields rather than writing null.
STREAM_RENDER_MANDATORY = (STREAM_RENDER_STRINGS + STREAM_RENDER_HASHES
                           + STREAM_RENDER_COUNTS + STREAM_RENDER_VERSIONS
                           + STREAM_RENDER_BOOLS + ("inventory_state",))

# THE FAIL-CLOSED VOCABULARY, by enumeration. Three survive W1's excision and W2 adds the
# fourth. RETIRED CODES ARE VIOLATIONS, NOT GRANDFATHERED: `snapshot_unsupported` and
# `coverage_not_ready` have no producer any more, so a fresh row carrying one means a producer
# survived the excision — exactly the defect this check exists to catch. Validating a current
# ledger and reading a historical one are different activities.
TURN_ERRORS = frozenset({"stream_data_invalid", "stream_over_budget", "history_fetch_failed",
                         "origin_fetch_failed"})

# RETIRED FIELDS, rejected rather than ignored. Each described the compaction-era stream — a
# boundary with an inclusivity flag, a snapshot id, a coverage floor, a single `reanchored`
# boolean now split into two. A CURRENT row carrying one means a producer survived the excision,
# which is exactly the defect worth failing on; tolerating them to stay readable against old
# files would make the checker unable to detect the thing it exists to detect.
STREAM_RENDER_RETIRED = ("snapshot_id", "generation", "boundary", "floor_inclusive",
                         "coverage_start_ts", "selection_result", "reanchored")


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
    """True when this line carries the contract version this tool grades.

    Older lines are SKIPPED, not failed: the file rotates across a deploy, so a mixed file is
    the normal state of the world and grading v9 rows by v10 rules would invent violations out
    of correct history. A version we do not know is refused in the other direction — silently
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

def _check_mandatory(row: Row, report: Report, name: str, *, reject_null: bool = False) -> None:
    """`reject_null` is opt-in, and only the three v9 events ask for it (the two reconsider
    events and `stale_send`, whose mandated `turn_id` join key follows the same no-null rule).
    Everywhere else
    absence is the emitter's own encoding of "unavailable" and a null has never been produced, so
    testing for one would be a rule about a shape no producer can make. On these two events the
    grammar says it out loud (§5): no JSON nulls anywhere, so a present-but-null identity field is
    the same defect as a missing one and gets the same violation name."""
    for field, why in MANDATORY.get(row.event, ()):
        if field not in row.obj:
            report.fail(name, row, f"{row.event} has no {field!r} — {why}")
        elif reject_null and row.obj.get(field) is None:
            report.fail(name, row, f"{row.event} has {field!r} = null — {why}")


def _check_join_key_string(row: Row, report: Report, name: str, *,
                           null_reported_elsewhere: bool = False) -> None:
    """TYPE enforcement on the mandated `turn_id` join key: `Joins.add` indexes only non-empty
    strings, so a null, non-string, or empty-string value would otherwise be DROPPED silently —
    a row the join invariants should count would count toward no turn, with no finding. The
    grammar flags it here first, so the join-side drop is a filter over already-failed rows,
    never a silent one. On the three v9 events whose `reject_null` mandatory check already
    reports an explicit null, the caller passes `null_reported_elsewhere=True` so one defect
    earns one finding."""
    if "turn_id" not in row.obj:
        return
    value = row.obj.get("turn_id")
    if value is None:
        if not null_reported_elsewhere:
            report.fail(name, row, f"{row.event} has turn_id = null — the row joins no turn")
    elif not isinstance(value, str) or not value:
        report.fail(name, row, f"{row.event} has turn_id={value!r} — not a non-empty string, "
                               f"so the row joins no turn")


def _check_no_explicit_nulls(row: Row, report: Report, fields: Tuple[str, ...],
                             name: str) -> None:
    """The v9 no-null rule on OPTIONAL fields (§5): an unavailable optional is OMITTED by the
    emitter's drop-None behavior, so a null that reached the file was written on purpose — a
    second encoding of a fact the grammar already encodes by absence, and a defect."""
    for field in fields:
        if field in row.obj and row.obj.get(field) is None:
            report.fail(name, row, f"{row.event} has {field!r} = null — unavailable optionals "
                                   f"are omitted, never null")


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
    _check_join_key_string(row, report, "turn_start_missing_field")
    if "surface" in row.obj and row.obj.get("surface") not in TURN_SURFACES:
        report.fail("turn_start_bad_surface", row, f"surface={row.obj.get('surface')!r}")


def _check_turn_outcome(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "turn_outcome_missing_field")
    _check_join_key_string(row, report, "turn_outcome_missing_field")
    _check_vocabulary(row, report, "kind", KINDS, "turn_outcome_bad_kind", required=False)
    if "stream_build_present" in row.obj \
            and not isinstance(row.obj.get("stream_build_present"), bool):
        report.fail("turn_outcome_missing_field", row,
                    f"stream_build_present={row.obj.get('stream_build_present')!r} is not a bool")
    # The fail-closed code, when one is present. Absence stays legal — most turns do not fail —
    # but a code outside the enumerated four is either a typo inventing a bucket or a retired
    # producer that survived the excision, and both are worth failing on.
    _check_vocabulary(row, report, "error", TURN_ERRORS, "turn_outcome_bad_error",
                      required=False)
    # v9. The nested reconsideration facts, when a reconsideration ran. Nested values survive
    # record()'s top-level drop-None, so a null INSIDE this object is an emitter defect.
    if "reconsider" in row.obj:
        facts = row.obj.get("reconsider")
        if not isinstance(facts, dict):
            report.fail("turn_outcome_reconsider_malformed", row,
                        f"reconsider is {type(facts).__name__}, not an object")
        else:
            # The nested grammar is CLOSED (the four as_payload() keys) and null-free: nested
            # values survive record()'s top-level drop-None, so an explicit null in here was
            # written on purpose and an unknown key is contract drift the event-side
            # closed-grammar check would never see.
            for key in sorted(facts):
                if key not in RECONSIDER_NESTED_FIELDS:
                    report.fail("turn_outcome_reconsider_malformed", row,
                                f"reconsider carries unknown key {key!r} — the nested grammar "
                                f"is closed (outcome, passes, forced, error)")
                elif facts.get(key) is None:
                    report.fail("turn_outcome_reconsider_malformed", row,
                                f"reconsider.{key} is null — inapplicable keys are omitted, "
                                f"never null")
            outcome = facts.get("outcome")
            if outcome not in RECONSIDER_OUTCOMES:
                report.fail("turn_outcome_reconsider_malformed", row,
                            f"reconsider.outcome={outcome!r} not in "
                            f"{{{', '.join(sorted(RECONSIDER_OUTCOMES))}}}")
            nested_passes = _as_int(facts.get("passes"))
            if nested_passes is None or nested_passes < 0:
                report.fail("turn_outcome_reconsider_malformed", row,
                            f"reconsider.passes={facts.get('passes')!r} is not a "
                            f"non-negative int")
            _check_reconsider_conditionals(row, report, facts, outcome,
                                           "turn_outcome_reconsider_malformed",
                                           where="reconsider.")
    destinations = row.obj.get("destinations")
    if "destinations" in row.obj:
        if not isinstance(destinations, list):
            report.fail("turn_outcome_destinations_not_list", row,
                        f"destinations is {type(destinations).__name__}")
        else:
            for index, destination in enumerate(destinations):
                _check_destination(row, report, index, destination)
    _check_edits(row, report, destinations if isinstance(destinations, list) else [])


def _check_edits(row: Row, report: Report, destinations: List[Any]) -> None:
    """v10: the turn's edit_own_message record, and its join to the disclosure destination.

    The entry grammar is closed and null-free (EDIT_FIELDS above). The JOIN INVARIANT is checked
    here rather than in a join pass because both sides live on this one row: every entry's
    `announcement_ts` — ALWAYS present, announcement-first means a record exists only once the
    disclosure was accepted — must name a COMMITTED `correction_announcement` destination in
    this turn's `destinations` — the disclosure is a real post the room saw, so an announcement
    ts that joins no destination means one of the two records is lying about what was
    delivered."""
    if "edits" not in row.obj:
        return   # absence is already a mandatory-field violation
    edits = row.obj.get("edits")
    if not isinstance(edits, list):
        report.fail("turn_outcome_edits_not_list", row,
                    f"edits is {type(edits).__name__} — always a list in CV10, empty when the "
                    f"turn edited nothing")
        return
    announced = {destination.get("first_ts") for destination in destinations
                 if isinstance(destination, dict)
                 and destination.get("kind") == "correction_announcement"
                 and destination.get("state") == "committed"}
    for index, edit in enumerate(edits):
        where = f"edits[{index}]"
        if not isinstance(edit, dict):
            report.fail("turn_outcome_edit_malformed", row,
                        f"{where} is {type(edit).__name__}, not an object")
            continue
        for field in sorted(edit):
            if field not in EDIT_FIELDS:
                report.fail("turn_outcome_edit_malformed", row,
                            f"{where} carries unknown key {field!r} — the nested grammar is "
                            f"closed (channel_id, target_ts, announcement_ts, state, error)")
            elif edit.get(field) is None:
                report.fail("turn_outcome_edit_malformed", row,
                            f"{where}.{field} is null — unavailable values are omitted, "
                            f"never null")
        for field in EDIT_REQUIRED_FIELDS:
            if field not in edit:
                report.fail("turn_outcome_edit_malformed", row, f"{where} has no {field!r}")
        state = edit.get("state")
        if "state" in edit and state not in EDIT_STATES:
            report.fail("turn_outcome_edit_bad_state", row,
                        f"{where}.state={state!r} not in "
                        f"{{{', '.join(sorted(EDIT_STATES))}}}")
        # §11.18: the identity fields name real Slack coordinates and `error` names a real
        # failure code — a present-but-empty string satisfies neither, so all four require a
        # NON-EMPTY string (None is already the null violation above).
        for field in ("channel_id", "target_ts", "announcement_ts"):
            if field in edit and edit.get(field) is not None \
                    and (not isinstance(edit.get(field), str) or not edit.get(field)):
                report.fail("turn_outcome_edit_malformed", row,
                            f"{where}.{field}={edit.get(field)!r} is not a non-empty string "
                            f"— the edit-entry identity fields name real Slack coordinates")
        if "error" in edit and edit.get("error") is not None \
                and (not isinstance(edit.get("error"), str) or not edit.get("error")):
            report.fail("turn_outcome_edit_malformed", row,
                        f"{where}.error={edit.get('error')!r} is not a non-empty string")
        # §11.6 lifecycle soundness: records exist only post-acceptance, so BOTH states carry
        # the disclosure ts; `error` rides announcement_only ALWAYS and committed NEVER.
        if "announcement_ts" not in edit:
            report.fail("turn_outcome_edit_announcement_missing", row,
                        f"{where} has no announcement_ts — an edit record exists only once "
                        f"the disclosure was accepted, so both states carry its ts")
        if state == "committed" and "error" in edit:
            report.fail("turn_outcome_edit_error_on_committed", row,
                        f"{where}.error={edit.get('error')!r} on a committed edit — a "
                        f"committed record carries no error")
        if state == "announcement_only" and "error" not in edit:
            report.fail("turn_outcome_edit_error_missing", row,
                        f"{where} is announcement_only with no error — the update did not "
                        f"land, and the record must say why")
        announcement_ts = edit.get("announcement_ts")
        if announcement_ts is not None and announcement_ts not in announced:
            report.fail("turn_outcome_edit_announcement_unjoined", row,
                        f"{where}.announcement_ts={announcement_ts!r} joins no committed "
                        f"correction_announcement destination on this turn — the disclosure "
                        f"post is a real destination, so an unjoined ts means one of the two "
                        f"records is wrong about what was delivered")


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
    """The FULL field contract. This row is the only durable evidence of what the model was
    shown, so a field that is quietly absent or the wrong type makes the record unreadable at
    exactly the moment someone is trying to explain a bad answer."""
    _check_mandatory(row, report, "stream_render_missing_turn_id")
    _check_join_key_string(row, report, "stream_render_missing_turn_id")

    for field in STREAM_RENDER_MANDATORY:
        # Absence and None are ONE case: record() omits None-valued fields rather than writing
        # null, so a presence test is already a non-None test and must not also test for null.
        if field not in row.obj:
            report.fail("stream_render_missing_field", row,
                        f"stream_render has no {field!r} — every §8 field is mandatory except "
                        "origin_thread_ts and trigger_ts")

    for field in STREAM_RENDER_RETIRED:
        if field in row.obj:
            report.fail("stream_render_retired_field", row,
                        f"stream_render carries retired field {field!r} — it was removed with "
                        "the compaction-era stream, so a fresh row holding one means a producer "
                        "survived the excision")

    for field in STREAM_RENDER_HASHES:
        if field not in row.obj:
            continue
        value = row.obj.get(field)
        if value == "" and field in HASH_MAY_BE_EMPTY:
            continue
        if not isinstance(value, str) or not _HEX64.match(value):
            report.fail("stream_render_bad_hash", row,
                        f"{field}={value!r} is not 64 lowercase hex characters")

    for field in STREAM_RENDER_COUNTS:
        if field not in row.obj:
            continue
        value = row.obj.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            report.fail("stream_render_bad_count", row,
                        f"{field}={value!r} is not a non-negative int")

    for field in STREAM_RENDER_VERSIONS:
        if field not in row.obj:
            continue
        value = row.obj.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            report.fail("stream_render_bad_count", row, f"{field}={value!r} is not an int")

    for field in STREAM_RENDER_BOOLS:
        if field in row.obj and not isinstance(row.obj.get(field), bool):
            report.fail("stream_render_bad_bool", row,
                        f"{field}={row.obj.get(field)!r} is not a bool")

    for field in STREAM_RENDER_STRINGS:
        if field in row.obj and not isinstance(row.obj.get(field), str):
            report.fail("stream_render_bad_field", row,
                        f"{field}={row.obj.get(field)!r} is not a string")

    _check_vocabulary(row, report, "inventory_state", INVENTORY_STATES,
                      "stream_render_bad_inventory_state", required=False)

    # The rendered window can never hold more roots than it holds messages: roots are a SUBSET
    # of the periphery's message items, so this catches a count computed over the wrong subject
    # — the pre-filter root count, say — which no type check would notice.
    included = row.obj.get("root_count")
    total = row.obj.get("message_count")
    if isinstance(included, int) and isinstance(total, int) and included > total:
        report.fail("stream_render_bad_count", row,
                    f"root_count={included} exceeds message_count={total}")


def _check_model_response(row: Row, report: Report) -> None:
    _check_mandatory(row, report, "model_response_missing_field")
    _check_join_key_string(row, report, "model_response_missing_field")
    _check_vocabulary(row, report, "status", MODEL_RESPONSE_STATUSES,
                      "model_response_bad_status", required=False)
    sequence = row.obj.get("attempt_seq")
    if "attempt_seq" in row.obj and (not isinstance(sequence, int) or isinstance(sequence, bool)):
        report.fail("model_response_missing_field", row,
                    f"attempt_seq={sequence!r} is not an int")


def _check_reconsider_conditionals(row: Row, report: Report, obj: Dict[str, Any],
                                   outcome: Any, name: str, *, where: str = "",
                                   required: bool = False) -> None:
    """The two conditional keys, shared between the event and the nested turn_outcome copy.

    `forced` rides ONLY on posted outcomes and `error` ONLY on `error_dropped`, so PRESENCE in
    the wrong place is a defect on BOTH copies. No JSON nulls anywhere in this grammar.

    `required` — the other direction — is asserted on the EVENT ONLY. There the emitter always
    has the fact: it writes `forced=False` on every posted outcome and the runner always names a
    §4f subtype when it drops. The nested `turn_outcome.reconsider` copy comes from
    `ReconsiderFacts.as_payload()`, which legally omits a `forced` that was never recorded, so
    demanding it there would fail a healthy row."""
    if "forced" in obj:
        if outcome not in ("posted_asis", "posted_revised"):
            report.fail(name, row, f"{where}forced={obj.get('forced')!r} on "
                                   f"outcome={outcome!r} — forced rides only on posted outcomes")
        elif not isinstance(obj.get("forced"), bool):
            report.fail(name, row, f"{where}forced={obj.get('forced')!r} is not a bool")
    elif required and outcome in ("posted_asis", "posted_revised"):
        report.fail(name, row, f"{where}outcome={outcome!r} has no {'forced'!r} — the emitter "
                               f"writes forced=False on every posted outcome, so its absence is "
                               f"a lost fact, not an omitted optional")
    if "error" in obj:
        if outcome != "error_dropped":
            report.fail(name, row, f"{where}error={obj.get('error')!r} on outcome={outcome!r} "
                                   f"— error rides only on error_dropped")
        elif obj.get("error") not in RECONSIDER_ERRORS:
            report.fail(name, row, f"{where}error={obj.get('error')!r} not in "
                                   f"{{{', '.join(sorted(RECONSIDER_ERRORS))}}}")
    elif required and outcome == "error_dropped":
        report.fail(name, row, f"{where}error_dropped has no {'error'!r} — the runner always "
                               f"has a §4f subtype, so a drop that names none says nothing")


def _check_unknown_fields(row: Row, report: Report, allowed: frozenset, name: str) -> None:
    """The literal grammar, enforced closed. Reported one key at a time so the name of the
    invented field is in the finding rather than a count of them."""
    for field in sorted(row.obj):
        if field not in allowed:
            report.fail(name, row, f"{row.event} carries unknown field {field!r} — the v9 "
                                   f"grammar for this event is closed")


def _check_reconsider_start(row: Row, report: Report) -> None:
    """One reconsideration pass. `pass` counts from 1 (contiguity is a join invariant); `scope`
    is the FULL three-part suppressing scope as a JSON list of three STRINGS — unlike
    stale_send's scope[0]."""
    _check_mandatory(row, report, "reconsider_start_missing_field", reject_null=True)
    _check_join_key_string(row, report, "reconsider_start_bad_field", null_reported_elsewhere=True)
    _check_no_explicit_nulls(row, report, ("attempt_id", "model_attempt_seq"),
                             "reconsider_start_bad_field")
    _check_unknown_fields(row, report, RECONSIDER_START_FIELDS, "reconsider_start_unknown_field")
    number = row.obj.get("pass")
    typed_number = _as_int(number)
    if "pass" in row.obj and (typed_number is None or typed_number < 1):
        report.fail("reconsider_start_bad_pass", row,
                    f"pass={number!r} is not an int counting from 1")
    scope = row.obj.get("scope")
    if "scope" in row.obj and (not isinstance(scope, list) or len(scope) != 3
                               or not all(isinstance(part, str) for part in scope)):
        report.fail("reconsider_start_bad_scope", row,
                    f"scope={scope!r} is not the full three-part scope as a JSON list of "
                    f"three strings")
    seq = row.obj.get("model_attempt_seq")
    if "model_attempt_seq" in row.obj and not _typed(seq, int):
        report.fail("reconsider_start_bad_field", row,
                    f"model_attempt_seq={seq!r} is not an int")


def _check_reconsider_outcome(row: Row, report: Report) -> None:
    """How one runner invocation ended. `passes` is the started-pass count — a fuse drop
    records 5, a failure or cancellation the passes started by then, so zero is legal."""
    _check_mandatory(row, report, "reconsider_outcome_missing_field", reject_null=True)
    _check_join_key_string(row, report, "reconsider_outcome_bad_field", null_reported_elsewhere=True)
    _check_no_explicit_nulls(row, report, ("attempt_id",), "reconsider_outcome_bad_field")
    _check_unknown_fields(row, report, RECONSIDER_OUTCOME_FIELDS,
                          "reconsider_outcome_unknown_field")
    outcome = row.obj.get("outcome")
    _check_vocabulary(row, report, "outcome", RECONSIDER_OUTCOMES,
                      "reconsider_outcome_bad_outcome", required=False)
    passes = row.obj.get("passes")
    typed_passes = _as_int(passes)
    if "passes" in row.obj and (typed_passes is None or typed_passes < 0):
        report.fail("reconsider_outcome_bad_passes", row,
                    f"passes={passes!r} is not a non-negative int")
    _check_reconsider_conditionals(row, report, row.obj, outcome,
                                   "reconsider_outcome_bad_conditional", required=True)


def _check_stale_send(row: Row, report: Report) -> None:
    """A gate-population diagnostic still — the only payload rule is the v9 join key, and the
    key is checked as a KEY: present, non-null (`reject_null` — the no-null rule covers this
    event's one mandated field), and a non-empty string, so a defective one fails by name here
    rather than being dropped silently at the join. Rows below v9 never reach here
    (_grade_version skips them), so the migration needs no grandfathering."""
    _check_mandatory(row, report, "stale_send_missing_turn_id", reject_null=True)
    _check_join_key_string(row, report, "stale_send_missing_turn_id", null_reported_elsewhere=True)


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


def _as_int(value: Any) -> Optional[int]:
    """The value when `_typed(value, int)` holds, else None — the same rule as a narrowing the
    type checker can see through, for the checks that go on to COMPARE the value."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


PAYLOAD_CHECKS = {
    "turn_start": _check_turn_start,
    "turn_outcome": _check_turn_outcome,
    "stream_render": _check_stream_render,
    "model_response": _check_model_response,
    "outbound_receipt": _check_outbound_receipt,
    "reconsider_start": _check_reconsider_start,
    "reconsider_outcome": _check_reconsider_outcome,
    "stale_send": _check_stale_send,
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
        # v9 — the reconsideration joins, all keyed by turn_id. stale_send is indexed here TOO
        # (by the turn_id v9 rows carry), because the pass-count invariant reads it; it stays a
        # gate-population diagnostic everywhere else.
        self.reconsider_starts: Dict[str, List[Row]] = defaultdict(list)
        self.reconsider_outcomes: Dict[str, List[Row]] = defaultdict(list)
        self.stale_sends: Dict[str, List[Row]] = defaultdict(list)

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
            # NOT a silent discard: every event indexed below mandates `turn_id`, and the
            # payload checks have already failed a missing / null / non-string / empty one by
            # name (`_check_mandatory` + `_check_join_key_string`). This filter only keeps an
            # already-flagged row from polluting the join keys.
            return
        if event == "turn_start":
            self.turn_starts[turn_id].append(row)
        elif event == "turn_outcome":
            self.turn_outcomes[turn_id].append(row)
        elif event == "stream_render":
            self.stream_renders[turn_id].append(row)
        elif event == "model_response":
            self.model_responses[turn_id].append(row)
        elif event == "reconsider_start":
            self.reconsider_starts[turn_id].append(row)
        elif event == "reconsider_outcome":
            self.reconsider_outcomes[turn_id].append(row)
        elif event == "stale_send":
            self.stale_sends[turn_id].append(row)


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
        # EXACTLY ONE, and "exactly" is not enforced by a rule that only checks presence. The
        # timeout tool-drop retry reuses the pinned stream and emits nothing, and no rebuild path
        # exists that could produce a second row — so two rows for one turn means either a second
        # build nobody intended or a duplicated write, and both make every count derived from
        # this population wrong.
        if len(renders) > 1:
            report.fail("stream_render_duplicate", renders[1],
                        f"turn_id={turn_id} has {len(renders)} stream_render rows; a turn "
                        f"renders once (first at {_first_at(renders[1], renders[0])})")
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


def _is_dm_row(row: Row) -> bool:
    """A DM's rows, by the only discriminator a ledger line carries: its channel id.

    "D…" is the conversation itself; "U…" and "W…" are the RECIPIENT ids a DM is sometimes
    addressed by (a Slack user id is "U…" on a normal workspace and "W…" on Enterprise Grid, and
    outbound DMs post to either), so all three read as DM. Nothing else on a row distinguishes
    the surface — `turn_start.surface` says it outright, but only that one event carries it."""
    channel_id = row.obj.get("channel_id")
    return isinstance(channel_id, str) and channel_id[:1] in ("D", "U", "W")


def _dm_head_is_present(row: Row, fragments: set) -> bool:
    """Can a DM's sequence be graded with no `turn_start` to prove its head is in the input?

    Yes — UNLESS we are reading a rotated fragment of its session, in which case pass 1 (or
    attempt 1) may be sitting in `participation.jsonl.1` and a sequence that legitimately starts
    at 3 would be failed as a gap. This is the same exemption a channel turn gets; a DM just
    reaches it by a different route, because the row that would otherwise prove the head is
    present is the one it may not have.
    """
    return row.session not in fragments


def _turn_is_dm(joins: "Joins", turn_id: str) -> bool:
    """Is this TURN a DM's? Asked of the turn rather than of one row, because the event that
    needs the answer — `model_response` — carries no `channel_id` at all. Any joined row that
    does carry one answers for the whole turn; they all name the same conversation."""
    for index in (joins.turn_starts, joins.turn_outcomes, joins.reconsider_starts,
                  joins.reconsider_outcomes, joins.stale_sends):
        for row in index.get(turn_id, ()):
            if row.obj.get("channel_id") is not None:
                return _is_dm_row(row)
    return False


def _check_model_attempts(joins: Joins, report: Report, *, fragments: set) -> None:
    """`attempt_seq` starts at 1 and is contiguous per turn_id — the sequence is what makes "this
    turn cost four API calls" true rather than a lower bound.

    Only checked for turns whose `turn_start` we can see: a rotated file can cut a turn's head
    off, and a sequence that legitimately starts at 3 must not be graded as a gap.

    A DM's rows are graded WITHOUT that head, because a ledger written before DM turns joined
    the turn population has no `turn_start` to join them to and the one attempt sequence a DM
    ever produces would go permanently unchecked. The ROTATION exemption still applies to them
    (`_dm_head_is_present`) — it is the missing turn_start that DMs are excused from, never the
    missing head.
    """
    for turn_id, responses in sorted(joins.model_responses.items()):
        if turn_id not in joins.turn_starts and not (
                _turn_is_dm(joins, turn_id)
                and _dm_head_is_present(responses[0], fragments)):
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


def _check_reconsiderations(joins: Joins, report: Report, *, fragments: set) -> None:
    """The v9 join invariants, all on turn_id: pass numbers contiguous from 1; a turn's
    `reconsider_start` count never exceeds its `stale_send` count (every pass exists because a
    suppression event preceded it, and each suppression event writes exactly one row); at most
    one `reconsider_outcome` per turn (the once-per-turn gate makes a second invocation
    impossible, so a duplicate is a defect in any file, fragment or not); and an outcome's
    `passes` EQUALS the number of `reconsider_start` rows joined to that turn — the field is a
    count of the passes this invocation started, so a disagreement means one of the two is
    counting something else, which is exactly the arithmetic every later reading of the file
    would inherit.

    DELIBERATELY NOT HERE: no cross-join of posted outcomes to `turn_outcome` kind or to F7 —
    F7 provenance is DB state with no ledger source, and the posted-outcome ⇒ ordinary-reply
    correspondence is unit/integration-mandated instead. And a `stale_send` row that carries a
    `turn_id` with no `turn_start` to join is TOLERATED: a DM ledger written before DM turns
    joined the population has exactly that shape, and it is not a defect.

    DM RECONSIDERATIONS ARE GRADED (STALE_SUPPRESSION_RECONSIDERATION ruling 8). A DM's draft
    goes through the same runner, so its passes obey the same arithmetic, and a DM row set may
    have no `turn_start` beside it — declining to grade those would leave the DM population
    unchecked. They are recognized by channel id (`_is_dm_row`) and graded in place.

    ROTATION IS STILL EXEMPT, on both surfaces. A file whose `session_start` is missing is a
    fragment: its early passes may be in `participation.jsonl.1`, so a sequence legitimately
    starting at 3 is not graded as a gap. What DMs are excused from is the missing `turn_start`,
    never the missing head.
    """
    for turn_id, starts in sorted(joins.reconsider_starts.items()):
        if turn_id not in joins.turn_starts and not (
                _is_dm_row(starts[0]) and _dm_head_is_present(starts[0], fragments)):
            report.warn("reconsider_start_orphan_turn", starts[0],
                        f"turn_id={turn_id} has no turn_start (passes not graded)")
            continue
        numbers = [n for n in (r.obj.get("pass") for r in starts)
                   if isinstance(n, int) and not isinstance(n, bool)]
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        if duplicates:
            report.fail("reconsider_pass_duplicate", starts[-1],
                        f"turn_id={turn_id} repeats pass {duplicates}")
        expected = list(range(1, len(set(numbers)) + 1))
        if sorted(set(numbers)) != expected:
            report.fail("reconsider_pass_not_contiguous", starts[0],
                        f"turn_id={turn_id} has passes {sorted(numbers)}, expected {expected}")
        suppressions = len(joins.stale_sends.get(turn_id, []))
        if len(starts) > suppressions:
            report.fail("reconsider_start_exceeds_stale_send", starts[-1],
                        f"turn_id={turn_id} has {len(starts)} reconsider_start rows but only "
                        f"{suppressions} stale_send row(s) — every pass exists because a "
                        f"suppression event preceded it")
    for turn_id, outcomes in sorted(joins.reconsider_outcomes.items()):
        if len(outcomes) > 1:
            report.fail("reconsider_outcome_duplicate", outcomes[-1],
                        f"turn_id={turn_id} has {len(outcomes)} reconsider_outcome rows "
                        f"(first at {_first_at(outcomes[-1], outcomes[0])}) — at most one "
                        f"runner invocation per turn")
        if turn_id not in joins.turn_starts and not (
                _is_dm_row(outcomes[0]) and _dm_head_is_present(outcomes[0], fragments)):
            report.warn("reconsider_outcome_orphan_turn", outcomes[0],
                        f"turn_id={turn_id} has no turn_start")
            continue
        # A malformed `passes` is already a payload violation; comparing it here too would only
        # report the same defect twice under a name that suggests a different one.
        passes = _as_int(outcomes[0].obj.get("passes"))
        started = len(joins.reconsider_starts.get(turn_id, []))
        if passes is not None and passes >= 0 and passes != started:
            report.fail("reconsider_outcome_passes_mismatch", outcomes[0],
                        f"turn_id={turn_id} records passes={passes} but {started} "
                        f"reconsider_start row(s) joined — `passes` is the count of passes "
                        f"this invocation started")


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
    # THREE populations now, and the DM one is counted apart for the same reason the gate and
    # turn denominators are: DM turns joined the turn population when their drafts started being
    # reconsidered, and a talk rate that pools a channel's turns with a DM's describes neither.
    report.counts["turn_start_dm"] = sum(
        1 for rows in joins.turn_starts.values() if _is_dm_row(rows[0]))
    report.counts["turn_outcome_dm"] = sum(
        1 for rows in joins.turn_outcomes.values() for row in rows if _is_dm_row(row))
    _check_turn_pairing(joins, report, tail_session=tail, fragments=fragments)
    _check_stream_render_joins(joins, report, fragments=fragments)
    _check_model_attempts(joins, report, fragments=fragments)
    _check_reconsiderations(joins, report, fragments=fragments)
    _check_terminal_invariant(joins, report, graded, fragments=fragments)

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
    channel_starts = counts["turn_start"] - counts["turn_start_dm"]
    channel_outcomes = counts["turn_outcome"] - counts["turn_outcome_dm"]
    print(f"      turn_start   {channel_starts:>6}  channel turns  "
          f"(turn_outcome: {channel_outcomes})", file=out)
    print(f"      turn_start   {counts['turn_start_dm']:>6}  DM turns       "
          f"(turn_outcome: {counts['turn_outcome_dm']})", file=out)
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
                         "turn_start": report.counts["turn_start"] - report.counts["turn_start_dm"],
                         "turn_outcome": (report.counts["turn_outcome"]
                                          - report.counts["turn_outcome_dm"]),
                         "turn_start_dm": report.counts["turn_start_dm"],
                         "turn_outcome_dm": report.counts["turn_outcome_dm"]},
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
