"""One record of what the participation system decided, and what the room saw.

WHY THIS EXISTS. Every question worth asking about participation — does it react to the right
messages, does it talk when it should stay quiet, is its emoji choice varied or is it stuck on
one joke — is a question about a POPULATION of decisions. Today that population cannot be
recovered from anything we keep. `app.log` carries the woken turns at INFO and the rest at
DEBUG, interleaved with everything else the bot does, in a file that rotates away every few
hours of real traffic.

The declines are the important half and the missing half. A gate that never wakes leaves no
trace anywhere: no message posted, no reaction placed, no row written. So the only evidence
that survives is evidence of the times we DID something, which is precisely the sample that
cannot tell us whether we are doing it too often. (Anthropic's own bot has this blind spot
structurally — its model is never invoked on the messages its gate declined, so it can describe
its policy but not its own miss rate. We own our gate, so we can just write the declines down.)

WHAT THIS IS NOT. It changes no behaviour and gates nothing. Every entry point swallows its own
exceptions and returns None: a telemetry failure must never cost a verdict, a reply or a
reaction. Nothing here is read back by the bot at runtime — the file is for us.

TWO POPULATIONS, NEVER ONE. Since v8 this file records both, and they must never be divided into
each other (the split is spelled out at `turn_start` below).

The GATE-ATTEMPT population is keyed by `attempt_id` and covers `gate_start`, `gate_decision`,
`gate_declined`, `reaction`, `visible_action` and the `stale_send`/`queue_link` diagnostics. An
attempt begins when `_run_participation_gate` mints an id; mentions, DMs and direct thread
continuations mint none and are NOT IN IT — no gate_start, no decision, no terminal, no reactions,
no work claims. The one place an ungated turn is named at all is `queue_link`, and only ever as the
DESTINATION of somebody else's queued work (`batched_into_channel_id` / `batched_into_trigger_ts`,
with `batched_into_gate_required=false` and no attempt id, because there is none). That names a
turn; it does not enrol it. Neither do the messages dropped BEFORE dispatch (level=off, own
message, non-semantic subtype, and the engine-off short-circuit in message_events) — they never
reach the gate, so any rate over this population has "messages that entered the gate" as its
denominator and nothing wider. The `gate_declined(cause=engine_off)` event covers only the OTHER
engine-off path, the one inside the gate itself.

The CHANNEL-TURN population is keyed by `turn_id` and covers `turn_start`, `turn_outcome`,
`stream_render`, `model_response` and `outbound_receipt`. EVERY channel
turn is in it, gated or not — a mention and a continuation write turn rows while writing no gate
rows at all — so its denominator is `turn_start` and is strictly wider than `gate_start`.

EVENTS.
  session_start   the sink opened. A health marker and a restart boundary, NOT an attempt: it
                  deliberately carries no attempt_id / channel_id / trigger_ts, and analysis
                  must exclude it when counting attempts. It ALONE carries `build` (the git
                  revision this process is running); every other line joins to it via `session`.
  session_end     the sink closed on purpose. Its absence at the tail of a session is the only
                  way to tell a graceful stop from a crash FROM THE FILE ALONE — see the
                  terminal-invariant note below, which depends on knowing which of the two
                  happened.
  gate_start      a message entered the gate. THE DENOMINATOR — emitted for EVERY minted
                  attempt, including the engine-off short-circuit inside the gate.
  gate_declined   DIAGNOSTIC, never terminal. Says why no verdict was acted on (superseded,
                  edit_superseded, classifier_error, engine_off, error, action_error) with
                  detail the terminal event does not carry. It is always accompanied by a
                  visible_action.

                  `error` and `action_error` are deliberately separate: `error` is a failure
                  BEFORE any verdict existed (the gate blew up on the way to the model), while
                  `action_error` is a failure AFTER a verdict was recorded, while carrying it
                  out. Only the first is a decline of the classifier; filing the second as one
                  would report the gate as failing to decide when it decided fine and the
                  reaction (or the handoff) is what broke.
  gate_decision   the model's own verdict, INCLUDING `ignore` — a decision, not a decline.
  stale_send      DIAGNOSTIC, never terminal. One per suppression: the guard refused to create
                  a surface because a newer message had already arrived. Emitted for UNGATED
                  turns too (a mention, a DM, a continuation), which carry no attempt_id —
                  those rows are outside the gate population and must be excluded from any
                  rate whose denominator is `gate_start`, exactly like the turns they describe.
  queue_link      DIAGNOSTIC, never terminal, and NEVER part of any denominator. One line per
                  queued attempt that a later catch-up turn absorbed: `attempt_id` is the
                  QUEUED source, and the `batched_into_*` fields name the successor turn that
                  answered on its behalf. See the linkage rule below.
  reaction        one attempt at one emoji: operation (add/remove) x result (added /
                  already_present / refused / failed / removed / remove_failed).
  turn_start      a CHANNEL turn opened (v8). Keyed by `turn_id`, not `attempt_id`: every channel
                  turn writes one, including the mentions and continuations no gate judged. It is
                  the denominator of the TURN population, which is not the gate population — the
                  two must never be divided into each other.
  turn_outcome    what that turn ended up doing, exactly once per turn_start, assembled in the
                  outer finally from the turn's own accumulated state. NOT a terminal event in the
                  `visible_action` sense; it describes a turn, not a gate attempt.
  stream_render   one channel stream BUILD: the pinned window, the hashes that identify it, and
                  what it cost in bytes. Joins to its turn by `turn_id`.
  model_response  one Responses API attempt on a channel turn, success or failure.
  outbound_receipt one receipt state transition (spec §5), from the transition's own result rather
                  than from what the caller intended.
  visible_action  THE SOLE TERMINAL EVENT. Exactly one per attempt in a HEALTHY EMITTED LEDGER,
                  guaranteed by `finish_attempt`, which flips a flag on the message and no-ops
                  on any second call.

                  That guarantee is about the CODE PATH, not about the file: a disabled sink, a
                  serialization failure, or a crash with records still queued loses whatever was
                  never written, and `finish_attempt` marks the attempt closed before it
                  enqueues, so `abort_attempt` will not retry it. Analysis must therefore COUNT
                  unmatched `gate_start`s (and check for a `session_end`) rather than assume
                  every start has a terminal — an attempt with zero of these is evidence about
                  the sink, and only two of them is evidence about this module.

ANALYSIS RULE FOR `queued`. A terminal of `kind=queued` is NOT an outcome — it means another
turn already owned the conversation and this message's work continues inside a LATER attempt
(the catch-up turn, which carries its own attempt_id). Counting it in a react / talk / silence
denominator scores one message's judgment as a silence and then scores it again under the turn
that actually answered. EXCLUDE it — that rule is unchanged and still binding.

LINKAGE. Which later attempt covered a queued one is now on the record. The message the drain
REUSES as its trigger is linked by `parent_attempt_id` (same Message, re-gated, so the second
attempt names the first). Every OTHER member of the drained batch is folded into that trigger's
turn and never gets a turn of its own, so it is linked by a `queue_link` line: join a queued
terminal to its successor by `attempt_id`, then read `batched_into_attempt_id`. When the
successor is an ungated turn — a mention, a DM, a thread continuation — there IS no successor
attempt, and the line carries `batched_into_gate_required=false` with the successor's
channel/trigger key instead: the fate is recoverable, just not as an attempt id. No successor
attempt is ever minted early to make the link prettier.

SHAPE. One JSON object per line, in `logs/participation.jsonl`. Events are NOT assembled into a
turn record in memory; each line stands alone and carries `attempt_id` (plus `channel_id` +
`trigger_ts`), so a turn is reconstructed by joining on that key at analysis time. `queue_link`
is the ONE deliberate exception: it carries its source's `attempt_id` but not that source's
coordinates, because the coordinates are already on every other line the source produced and the
join answers the question this event exists for. Its `channel_id`/`trigger_ts`-shaped fields are
prefixed `batched_into_` and describe the SUCCESSOR, not the subject of the line — an analysis
that treats them as the line's own coordinates would file each queued message under the turn
that absorbed it. That means no
in-flight state to leak, no unbounded dict keyed by messages that may never come back, and no
ordering assumption — a crash mid-turn loses nothing that was already written. A redispatch of
the same Message gets a FRESH attempt_id and carries `parent_attempt_id` pointing at the one it
replaces, so a re-gate is a linked second attempt rather than a duplicate of the first.

NOT AN ARCHIVE. The file rotates under the same policy as app.log (logger.LOG_ROTATION_*), so
old generations are eventually deleted, not kept. Any analysis of more than the very recent past
has to read `participation.jsonl.1` … `.5` as well, and has to accept that history older than
those five generations is gone.

PRIVACY. Nothing model-authored is written here at all. The rich gate wrote a `reason` and a
`guidance` — summaries of a human's message — and bounded them with a shared truncation, which
lowers exposure without promising none. The binary gate produces neither, so the exposure is gone
rather than bounded, and the truncation helper went with it: a bound with no field to bound is an
invitation to add one back without thinking about what it would carry.

`v` is the contract version. Bump it when a field changes MEANING, and equally when event
CARDINALITY or TERMINAL SEMANTICS change (one visible_action per attempt is part of the
contract, not an implementation detail) — so a later analysis can refuse to average two
incompatible definitions together.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any, Dict, List, Optional

from config import config
from runtime_identity import SESSION_ID
from logger import (LOG_ROTATION_BACKUP_COUNT, LOG_ROTATION_MAX_BYTES,
                    USE_CONCURRENT_HANDLER, setup_logger)

# The app log, for this module's own health — never for the JSON lines themselves.
logger = setup_logger(name="slack_bot.ParticipationTelemetry")

# Bump on a semantic change to an existing field, on a change to which events an attempt
# produces, or on a change to what "terminal" means. Adding a new key is backwards compatible:
# an older reader ignores it, and a newer reader sees it missing on old lines.
# v2: one mandatory terminal visible_action per attempt; attempt_id/session identity; the
#     reaction event's placed-bool replaced by operation + result.
# v3: the `queue_link` event — an attempt may now produce a line that is neither a decision nor
#     an outcome, so event cardinality per attempt changed.
# v4: `silence_reason` changed MEANING — model-authored prose became one of eight declared
#     values (message_processor.terminal_actions), so v3 and v4 rows cannot be grouped together;
#     and `reaction_visible` became explicit on every responder terminal rather than present
#     only on replies, so its absence no longer means the same thing either.
# v5: the ambiguous `placement` field is gone FROM THE TERMINAL EVENT, replaced there by
#     `destination` + `destination_source`. It said "thread" for every turn that produced no
#     words at all, so half the column described a decision nobody had made; the new pair is
#     written only on terminals that delivered the responder's own reply, and says who decided
#     as well as where it went. NOTE: `gate_decision` still carries its own `placement` — the
#     gate's pre-answer verdict, which no longer routes anything and is removed with the rest of
#     the rich verdict in the binary-gate commit.
# v6: the `stale_suppressed` terminal kind and the `stale_send` diagnostic event — a turn can
#     now end because the conversation moved past it, which is neither a silence nor a delivery
#     failure and had no representation before.
# v7: the BINARY gate. `gate_decision` carries one bit and four facts about the call; every rich
#     field is gone — action, emoji, placement, reason, the staged findings, the overrules and the
#     whole backoff taxonomy — because the gate no longer decides any of it. A `wake=false` is a
#     terminal `silence` with NO `silence_reason`: that enum belongs to the responder, which can
#     say why it chose quiet after seeing everything, where the gate knows only that it did not
#     open. `classifier_error` no longer arrives with a manufactured decision beside it.
# v8: the SINGLE STREAM (Docs/SINGLE_STREAM_SPEC.md §10). Five new events — turn_start,
#     turn_outcome, stream_render, model_response, outbound_receipt — and
#     with them a SECOND population keyed by `turn_id` rather than `attempt_id`, covering every
#     CHANNEL turn including the ungated ones the gate population deliberately excludes. Event
#     cardinality per attempt therefore changed, which is what earns the bump; `visible_action`
#     itself is untouched, keeps its vocabulary, and remains the sole terminal of a gate attempt.
#     A `turn_outcome` is NOT a terminal event and must never be counted as one: the two
#     populations answer different questions and their rows join, at most, by attempt_id.
# v8, amended — `compaction_snapshot` and the telemetry outbox that carried it are REMOVED with
#     the compaction machinery itself. THE VERSION DOES NOT MOVE: the turn population's remaining
#     events are unchanged in identity, and no completeness rule ever named the removed event, so
#     nothing that reads a v8 ledger has a denominator this invalidates. v8 rows written before
#     this stay valid and readable; they simply carry an event nothing emits any more.
# v9: STALE RECONSIDERATION (Docs/SINGLE_STREAM_SPEC.md §10, CV9 addendum). `stale_send` gains
#     `turn_id` and its CARDINALITY generalizes from one-per-suppression to one per suppression
#     EVENT — the initiating refusal, each per-pass re-race inside the reconsideration runner,
#     and a post-run once-gate suppression each emit exactly one row. Two new turn-population
#     events, `reconsider_start` (one per pass) and `reconsider_outcome` (at most one per runner
#     invocation), join on `turn_id`; `turn_outcome` may carry a nested `reconsider` payload;
#     `model_response` gains the `stale_reconsideration` fork reason. Unavailable optional
#     fields are OMITTED, never null — the generic drop-None rule is part of this grammar.
# v10: EDIT_OWN_MESSAGE (Docs/SINGLE_STREAM_SPEC.md §10, CV10 addendum). `turn_outcome` gains
#     `edits` — ALWAYS present as a list (empty when the turn edited nothing), one entry per
#     `TurnRuntime.edits` EditRecord: {"channel_id", "target_ts", "announcement_ts",
#     "state": "announcement_only"|"committed"[, "error"]} — a record exists only once the
#     disclosure was ACCEPTED (spec §11.6/§11.18), so `announcement_ts` is ALWAYS present;
#     only `error` is optional, OMITTED per the drop-None rule (rides announcement_only
#     always, committed never). `reconsider` and `edits` may coexist on one row. The
#     destination kind vocabulary gains `correction_announcement` (the executor-synthesized
#     disclosure post, recorded as a committed destination); every CV9 reconsideration field
#     and invariant is preserved unchanged.
CONTRACT_VERSION = 10

# WHICH gate produced these lines. The rich multi-signal classifier was "rich-v1"; the one-bit
# wake gate is "binary-v1". A different gate is a different population even at the same
# CONTRACT_VERSION, and the two must never be pooled just because some field names line up —
# a rich `ignore` and a binary `wake=false` are not the same event, and the rate of one says
# nothing about the rate of the other. v2–v6 rows stay valid under their own contracts.
GATE_CONTRACT = "binary-v1"

# One per process (runtime_identity), so a ledger line and a receipt row from the same run name
# the same session. Restarts lose all in-memory participation state, so a line's session is what
# tells a burst of odd verdicts after a deploy apart from a genuine change in the room.

# A RESTART IS NOT A DEPLOYMENT. `session` separates two runs; it cannot say whether the second
# run is the same code as the first, and the interesting comparisons here — did the prompt change
# make it quieter? — are exactly the ones that straddle a code change. The gate prompt has
# already moved once without GATE_CONTRACT moving, so the contract string cannot be relied on as
# a build boundary either. The git revision can, and it is one cheap subprocess at startup.
#
# Written on `session_start` ONLY: it is constant for the process, and repeating it on every line
# would grow the file for a fact that a join on `session` already answers.
_BUILD_REVISION: Optional[str] = None
# Deliberately short and deliberately named: this runs on the startup path, and a git call that
# hangs (a network-backed worktree, a stale index lock) must cost the boot a moment, not a
# minute. Any failure at all — no git, no repo, non-zero exit, timeout — is simply "unknown".
BUILD_REVISION_TIMEOUT_S = 2.0

# The VOCABULARY. Analysis is a group-by, so a typo does not fail anything — it silently invents
# a bucket and quietly deflates the real one. These sets do not gate behaviour (a line with an
# unknown value is still written: losing the record is worse than recording an odd label); they
# make the drift audible in app.log the first time it happens.
KINDS = frozenset({
    "reply", "delivery_failed", "silence", "reaction_only", "detached", "queued",
    "interrupted", "error", "error_unhandled", "aborted", "empty", "none",
    "stale_suppressed",
})
DECLINE_CAUSES = frozenset({
    "superseded", "edit_superseded", "classifier_error", "engine_off", "error", "action_error",
    # A gate-routed cohort of nothing but captionless images. Structural, not a judgment: there is
    # no text to decide about, so no classifier runs and no responder wakes. The ambient worker
    # still analyses the pictures and the files are still catalogued — declining to answer a
    # wordless upload is not the same as forgetting it happened.
    "image_only",
})
REACTION_OPERATIONS = frozenset({"add", "remove"})
REACTION_RESULTS = frozenset({
    "added", "already_present", "refused", "failed", "removed", "remove_failed",
})
# `gate` and `backoff_ack` are RETIRED: the binary gate places nothing in the room, so a runtime
# row can only be `responder` (a social reaction the answering model chose) or `work_claim` (the
# 👀 staked by slow work, which was never a gate reaction). The retired names stay in the
# vocabulary because v2–v6 history is full of them and a reader of those rows must not be told
# they are invalid.
REACTION_ORIGINS = frozenset({"gate", "responder", "work_claim", "backoff_ack"})

# v8 vocabularies. Same rule as above: an odd value is written and warned about, never dropped.
# `turn_outcome.kind` deliberately reuses KINDS — a turn and its gate attempt describe the same
# room, and two vocabularies for one question would make the rows uncomparable.
TURN_SURFACES = frozenset({"channel", "dm"})
# The receipt ops, matching outbound_receipts' lattice plus the two that have no lattice op:
# `reconcile_finalize` (boot, per row) and `pending_resolve` (an upload's share ts arriving).
RECEIPT_OPS = frozenset({
    "register", "promote", "finalize", "demote", "transfer", "delete", "reconcile_finalize",
    "pending_resolve",
})
# `absent` is a real prior state: a finalize may insert a row that was never registered.
RECEIPT_STATES = frozenset({"absent", "in_flight", "finalized", "chrome"})
MODEL_RESPONSE_STATUSES = frozenset({"ok", "error"})
# v9. How one reconsideration runner invocation ended. `posted_asis`/`posted_revised` assert
# PHYSICAL Slack acceptance of the first surface, not finalized turn accounting.
RECONSIDER_OUTCOMES = frozenset({
    "posted_asis", "posted_revised", "skipped", "fuse_dropped", "error_dropped", "cancelled",
})

# Warn once per unknown value, not once per line: a typo in a hot path would otherwise fill
# app.log with the same sentence and bury whatever else is happening.
_warned_vocabulary: set = set()

LOG_NAME = "participation.jsonl"
_SINK_LOGGER_NAME = "participation_telemetry"
# Marks the handlers WE own, on a logger whose handler list we do not exclusively control:
# pytest's logging plugin attaches its own capture handler directly to non-propagating loggers.
_OURS = "_participation_sink"

# --- attempt bookkeeping, carried on message.metadata ---
# Public: other modules read this to decide whether they are inside a gated attempt at all.
ATTEMPT_KEY = "participation_attempt_id"
# Private: the ledger's own scratch space on the same dict. Underscored so nothing else reads
# them as behaviour, and so a Message dumped into a log is obviously not carrying config.
PARENT_KEY = "_pt_parent_attempt_id"
CLOSED_KEY = "_pt_attempt_closed"
GATE_WOKE_KEY = "_pt_gate_woke"
RESPONDER_STARTED_KEY = "_pt_responder_started"
# The queued attempts a drained batch folded into the message carrying this key. Staged by the
# drain (which knows the sources) and emitted by the successor turn (which alone knows what it
# became), so nothing has to pre-mint an attempt id to have something to point at.
BATCHED_SOURCES_KEY = "_pt_batched_source_attempt_ids"

_lock = threading.Lock()
_sink: Optional[logging.Logger] = None
_queue: Optional[queue.Queue] = None
_listener: Optional[QueueListener] = None
_init_failed = False


# --------------------------------------------------------------------------- the sink

def _soft_check(value: Any, allowed: frozenset, label: str) -> None:
    """Note an off-vocabulary value and carry on. Never raises, never gates.

    A telemetry contract that refuses to write the odd line protects the schema by destroying
    the evidence — and the odd line is usually the interesting one. So this only complains."""
    try:
        if value is None or value in allowed:
            return
        key = (label, str(value))
        if key in _warned_vocabulary:
            return
        _warned_vocabulary.add(key)
        logger.warning(
            f"Participation telemetry: unknown {label} {value!r} (line still written) — "
            f"known values: {', '.join(sorted(allowed))}")
    except Exception:  # noqa: BLE001 — a vocabulary check must never cost a line
        pass


def _build_revision() -> Optional[str]:
    """The git revision this process is running, or None. Best effort, and quiet about failure.

    Not a version string: the repo is the deployment unit here, and `git rev-parse` is the only
    identity that is true without anyone remembering to bump anything."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=BUILD_REVISION_TIMEOUT_S)
        if result.returncode != 0:
            return None
        return (result.stdout or "").strip() or None
    except Exception:  # noqa: BLE001 — no git, no repo, or too slow: the ledger still opens
        return None


def initialize() -> None:
    """Open the ledger. Called once at startup, from ChatBotV2.initialize().

    EAGER, deliberately. Building this lazily on the first `record()` put a directory creation
    and a file open on the gate's hot path — inside the one call the whole participation
    decision is waiting on — and did it again after every failure. Doing it at startup means
    the first message of the day pays nothing, and a broken log directory is known before any
    traffic arrives rather than discovered one gate at a time.

    Its own file and its own handler: sharing app.log would put these lines behind that file's
    rotation, and ten minutes of DEBUG traffic would age out a week of decisions.
    `propagate = False` keeps the JSON out of app.log and off the console. The write itself
    happens on the listener thread, so `record()` only ever enqueues.

    Idempotent, and silent about a second call. On failure it warns ONCE and disables itself:
    a log that cannot be opened is not worth a warning per message.

    OFF MEANS OFF. With the flag false this returns before creating anything — no directory, no
    file, no listener thread. `record()` checks the flag too, but that only stops the writing;
    the promise in config.py is that the feature costs nothing when disabled, and an empty
    rotating log plus a parked thread is not nothing.

    The module globals are published BEFORE the listener starts, so a failure in between leaves
    a listener that `shutdown()` can still find and stop rather than an orphan nobody holds a
    reference to. The cost of that order is a window where `record()` enqueues into a queue no
    thread is draining yet — harmless, because the listener drains from the head on start.
    A crash before `run()` reaches its shutdown still loses whatever is queued at that instant:
    QueueListener's thread is a daemon and dies with the process, unflushed.
    """
    global _sink, _queue, _listener, _init_failed, _BUILD_REVISION
    if not getattr(config, "enable_participation_telemetry", True):
        return
    if _sink is not None or _init_failed:
        return
    with _lock:
        if _sink is not None or _init_failed:
            return
        try:
            logs_dir = getattr(config, "log_directory", None) or "logs"
            os.makedirs(logs_dir, exist_ok=True)
            path = os.path.join(logs_dir, LOG_NAME)
            # Rotation is app.log's policy, imported rather than restated: one retention
            # answer in the repo. These lines are a fraction of app.log's volume, so the same
            # ceiling buys far more history.
            handler: logging.Handler
            if USE_CONCURRENT_HANDLER:
                from concurrent_log_handler import ConcurrentRotatingFileHandler
                handler = ConcurrentRotatingFileHandler(
                    path, maxBytes=LOG_ROTATION_MAX_BYTES,
                    backupCount=LOG_ROTATION_BACKUP_COUNT)
            else:
                handler = RotatingFileHandler(
                    path, maxBytes=LOG_ROTATION_MAX_BYTES,
                    backupCount=LOG_ROTATION_BACKUP_COUNT)
            handler.setFormatter(logging.Formatter("%(message)s"))  # the line IS the JSON
            setattr(handler, _OURS, True)

            event_queue: queue.Queue = queue.Queue(-1)  # unbounded, like logger.py's
            listener = QueueListener(event_queue, handler, respect_handler_level=True)

            sink = logging.getLogger(_SINK_LOGGER_NAME)
            sink.setLevel(logging.INFO)
            sink.propagate = False
            for existing in [h for h in sink.handlers if getattr(h, _OURS, False)]:
                sink.removeHandler(existing)   # a re-init must not double-write every line
            queue_handler = QueueHandler(event_queue)
            setattr(queue_handler, _OURS, True)
            sink.addHandler(queue_handler)

            # Published first, started second: see the docstring. A raise from start() below
            # then leaves something shutdown() can reach.
            _queue, _listener, _sink = event_queue, listener, sink
            listener.start()
            _BUILD_REVISION = _build_revision()
        except Exception as e:  # noqa: BLE001
            _init_failed = True
            logger.warning(f"Participation telemetry disabled (sink unavailable): {e}")
            return
    # Outside the lock: record() takes no lock, but keeping the emit out of the critical
    # section is what makes that true by construction rather than by inspection.
    record("session_start", build=_BUILD_REVISION)


def shutdown() -> None:
    """Drain and close the ledger. Idempotent; safe to call when it was never opened.

    `QueueListener.stop()` enqueues a sentinel and joins the thread, so everything already
    queued is written first — which matters because the events most worth keeping (the one
    terminal event per attempt) are the last ones produced.

    Emits `session_end` BEFORE the stop, and that ordering is the point: it rides the same queue
    as everything else, so it can only be the last line of a session that actually drained. A
    file whose final session has no `session_end` ended some other way, and its tail is missing
    an unknown number of terminals — which is the difference between "the bot went quiet" and
    "we stopped recording".
    """
    global _sink, _queue, _listener, _init_failed
    if _sink is not None:
        record("session_end")
    with _lock:
        sink, listener = _sink, _listener
        _sink = _queue = _listener = None
        _init_failed = False   # a later initialize() may legitimately retry
    if sink is not None:
        for handler in [h for h in sink.handlers if getattr(h, _OURS, False)]:
            sink.removeHandler(handler)
    if listener is not None:
        try:
            listener.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Participation telemetry listener stop failed: {e}")
        for handler in listener.handlers:
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass


def _drain() -> None:
    """Block until everything enqueued so far has been written. For tests only.

    The listener owns the file, so a test that reads the file immediately after `record()`
    is racing a thread. `Queue.join()` is exact here: QueueListener calls `task_done()` for
    every record it handles.
    """
    event_queue, listener = _queue, _listener
    if event_queue is None or listener is None:
        return
    worker = getattr(listener, "_thread", None)
    if worker is None or not worker.is_alive():
        return   # nothing will ever drain it; joining would hang the test suite
    event_queue.join()
    for handler in listener.handlers:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass


def record(event: str, *, channel_id: Optional[str] = None,
           trigger_ts: Optional[str] = None, **fields: Any) -> None:
    """Append one event. Never raises, and never blocks: this only enqueues.

    `fields` are event-specific and free-form; keep the key names stable, because the analysis
    is a grep and a group-by, not a schema migration. Values are coerced through `json.dumps`
    with `default=str` so an unexpected object degrades to its repr instead of losing the line.
    None-valued fields are OMITTED rather than written as null, so a group-by never gets two
    buckets — absent and null — meaning the same thing.
    """
    if not getattr(config, "enable_participation_telemetry", True):
        return
    try:
        # Inside the try, deliberately. "Never raises" has to hold structurally rather than by
        # the good behaviour of the functions below — an exception escaping here lands in the
        # gate's except-clause and becomes silence.
        sink = _sink
        if sink is None:
            return   # not initialized (or initialization failed): drop the line, cost nothing
        payload: dict = {
            "v": CONTRACT_VERSION,
            "at": time.time(),
            "session": SESSION_ID,
            "gate_contract": GATE_CONTRACT,
            "event": event,
        }
        if channel_id is not None:
            payload["channel_id"] = channel_id
        if trigger_ts is not None:
            payload["trigger_ts"] = trigger_ts
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        sink.info(json.dumps(payload, default=str))
    except Exception as e:  # noqa: BLE001 — a lost line is never worth a lost turn
        logger.debug(f"Participation telemetry write failed: {e}")


# ------------------------------------------------------------------ the attempt lifecycle

def begin_attempt(message: Any) -> Optional[str]:
    """Mint this gate attempt's id and stamp it on the message. Returns the id.

    A redispatch (Phase Q) re-runs the SAME Message object through the gate, so an id may
    already be stamped. That is a second attempt, not a continuation: it gets a fresh id, the
    old one rides on as `parent_attempt_id`, and the terminal flag is cleared so the new
    attempt can close in its own right. Without the reset a redispatch would inherit the first
    attempt's closed flag and silently produce no terminal event at all.
    """
    try:
        meta = getattr(message, "metadata", None)
        if not isinstance(meta, dict):
            return None
        previous = meta.get(ATTEMPT_KEY)
        attempt_id = uuid.uuid4().hex
        meta[ATTEMPT_KEY] = attempt_id
        if previous:
            meta[PARENT_KEY] = previous
        meta.pop(CLOSED_KEY, None)
        meta.pop(GATE_WOKE_KEY, None)
        meta.pop(RESPONDER_STARTED_KEY, None)
        return attempt_id
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Participation attempt id not minted: {e}")
        return None


def detach_attempt(message: Any) -> Optional[str]:
    """Strip a CLOSED gate attempt off a message that is about to run as an ungated turn, and
    return the id that was removed (None when there was nothing to detach).

    One caller: the queue drain, when a batch it folded together contains an answer already owed,
    so the successor turn runs without re-gating. That turn is now an ungated route — the same
    shape as a DM or an @mention — but the trigger is still carrying the attempt from its earlier
    gated pass, already closed with `queued`. Left in place it corrupts two things at once: every
    reaction this turn places is attributed to a closed attempt, and `finish_attempt` sees the
    closed flag and writes NO terminal event at all, so the turn the room actually saw is missing
    from the ledger entirely.

    Detaching rather than re-minting is deliberate. A fresh id here would put a gate attempt into
    a ledger documented as gate attempts, for a turn no gate judged. Ungated routes mint nothing,
    and this turn is one of them; the returned id is staged as an absorbed source instead, so the
    earlier attempt still says what became of it."""
    try:
        meta = getattr(message, "metadata", None)
        if not isinstance(meta, dict):
            return None
        detached = meta.pop(ATTEMPT_KEY, None)
        for key in (CLOSED_KEY, GATE_WOKE_KEY, RESPONDER_STARTED_KEY, PARENT_KEY):
            meta.pop(key, None)
        return detached or None
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Participation attempt not detached: {e}")
        return None


def attempt_id_for(message: Any) -> Optional[str]:
    """This message's live attempt id, or None when it was never gated.

    THE SCOPE GUARD. Every reaction and every terminal event in this ledger is conditional on
    this returning something: a mention, a DM or a direct thread continuation is not a gate
    attempt, and letting its rows in would put outcomes with no decision behind them into a
    population documented as decisions.
    """
    meta = getattr(message, "metadata", None)
    if not isinstance(meta, dict):
        return None
    return meta.get(ATTEMPT_KEY) or None


def mark_gate_woke(message: Any) -> None:
    """The rich gate handed this message to the FULL responder — respond, react_and_respond, or
    a backoff that falls through for a settings change. A gate-only reaction does NOT set this:
    the gate acted, but it never woke the responder, and conflating the two makes the wake rate
    unreadable."""
    _mark(message, GATE_WOKE_KEY)


def mark_responder_started(message: Any) -> None:
    """The processor is about to be invoked. Set immediately before the call, so a turn that
    died on the way there (thinking indicator, queue peek) is distinguishable from one that
    reached the model and failed."""
    _mark(message, RESPONDER_STARTED_KEY)


def _mark(message: Any, key: str) -> None:
    """Stamp one of this module's private flags — but ONLY inside a gate attempt.

    `mark_responder_started` sits on the path every turn takes, so without the attempt-id guard
    a mention or a DM picked up telemetry bookkeeping it can never produce a line from: state
    with no reader, on messages this ledger is documented as excluding."""
    try:
        meta = getattr(message, "metadata", None)
        if isinstance(meta, dict) and meta.get(ATTEMPT_KEY):
            meta[key] = True
    except Exception:  # noqa: BLE001
        pass


def finish_attempt(message: Any, kind: str, **fields: Any) -> bool:
    """Close this attempt with the ONE outcome the room saw. Returns True if it closed it.

    Every terminal path goes through here — gate-terminal silences, gate reactions, handled
    backoffs, declines, responder outcomes, unhandled exceptions, aborts — and the second call
    for an attempt is a no-op. That is the whole point: the outer gate and the responder path
    both believe they own the end of the turn, and before this guard existed a gate decline was
    counted twice while an aborted turn was counted not at all.

    No attempt id (a mention, a DM, a direct continuation) means no event: see `attempt_id_for`.
    """
    try:
        meta = getattr(message, "metadata", None)
        if not isinstance(meta, dict):
            return False
        attempt_id = meta.get(ATTEMPT_KEY)
        if not attempt_id or meta.get(CLOSED_KEY):
            return False
        meta[CLOSED_KEY] = True
        _soft_check(kind, KINDS, "visible_action kind")
        visible_action(
            getattr(message, "channel_id", None),
            meta.get("ts") or getattr(message, "thread_id", None),
            kind=kind,
            attempt_id=attempt_id,
            parent_attempt_id=meta.get(PARENT_KEY),
            gate_woke=bool(meta.get(GATE_WOKE_KEY)),
            responder_started=bool(meta.get(RESPONDER_STARTED_KEY)),
            **fields)
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Participation attempt not closed: {e}")
        return False


def stage_queue_links(trigger: Any, source_attempt_ids: Any) -> None:
    """Remember, on the message a drain chose as its trigger, which queued attempts it is
    absorbing. Nothing is written here: at drain time the successor turn does not exist yet, and
    minting its attempt id early to have something to write would put a start in the ledger for
    a turn that may never run.

    MERGES rather than overwrites. A trigger can arrive already carrying sources it inherited
    from an earlier drain and never got to answer (it was queued again before it ran), and
    replacing that list would drop exactly the messages whose lineage is hardest to recover."""
    try:
        meta = getattr(trigger, "metadata", None)
        if not isinstance(meta, dict):
            return
        merged = list(meta.get(BATCHED_SOURCES_KEY) or [])
        for source_id in (source_attempt_ids or []):
            if source_id and str(source_id) not in merged:
                merged.append(str(source_id))
        if merged:
            meta[BATCHED_SOURCES_KEY] = merged
    except Exception as e:  # noqa: BLE001 — a lost link is never worth a lost turn
        logger.debug(f"Participation queue links not staged: {e}")


def take_staged_links(message: Any) -> list:
    """The sources staged on a message that it never got to answer, removed as they are handed
    over. A queued successor is absorbed into a LATER trigger, and its unanswered inheritance
    has to travel with it — otherwise the chain ends at a turn that never ran."""
    try:
        meta = getattr(message, "metadata", None)
        if not isinstance(meta, dict):
            return []
        return list(meta.pop(BATCHED_SOURCES_KEY, None) or [])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Participation queue links not forwarded: {e}")
        return []


def emit_queue_links(message: Any, *, gate_required: bool) -> None:
    """Write the staged links now that this turn is genuinely running, and clear them so a turn
    that is itself queued and drained again links its next batch, not this one.

    CALLED AT THE POINT OF NO RETURN, not at the point of intent. A gated turn calls it once the
    gate has minted its attempt; an ungated one once it holds the conversation lock. Anything
    earlier can claim that a turn absorbed messages it then queued without answering — and an
    ungated turn has no attempt id, so nothing downstream could correct the record.

    Leaves the links STAGED (and writes nothing) when a gated turn has no attempt id yet: the
    attempt was never minted, so there is no successor to name, and the sources are still owed
    to whichever turn eventually runs."""
    try:
        meta = getattr(message, "metadata", None)
        if not isinstance(meta, dict):
            return
        sources = meta.get(BATCHED_SOURCES_KEY)
        if not sources:
            return
        successor_id = meta.get(ATTEMPT_KEY) if gate_required else None
        if gate_required and not successor_id:
            return   # cancelled/raised before the attempt existed — keep them for a later turn
        meta.pop(BATCHED_SOURCES_KEY, None)
        for source_attempt_id in sources:
            queue_link(source_attempt_id,
                       batched_into_attempt_id=successor_id,
                       batched_into_channel_id=getattr(message, "channel_id", None),
                       batched_into_trigger_ts=(meta.get("ts")
                                                or getattr(message, "thread_id", None)),
                       batched_into_gate_required=bool(gate_required))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Participation queue links not written: {e}")


def abort_attempt(message: Any) -> bool:
    """The turn left without closing its attempt: a cancellation, an early raise on the way to
    the gate, a thinking-indicator failure. Runs from a `finally` that wraps everything from
    before the gate call onward, so the ledger has no attempts that simply stop mid-sentence."""
    return finish_attempt(message, "aborted")


# ------------------------------------------------------------------------------ the events

def gate_start(channel_id: Optional[str], trigger_ts: Optional[str],
               *, attempt_id: Optional[str] = None, **posture: Any) -> None:
    """A message entered the gate. This is the DENOMINATOR — without it a decline that returns
    early is indistinguishable from a message that was never gated at all.

    `posture` is what was true about the message BEFORE the model saw it (level, thread_reply,
    name_hit, sender_is_bot, sender_type, wake_source, has_images, has_attachments, redispatch,
    edit), so a rate can be sliced by the conditions rather than only by the outcome."""
    record("gate_start", channel_id=channel_id, trigger_ts=trigger_ts,
           attempt_id=attempt_id, **posture)


def gate_declined(channel_id: Optional[str], trigger_ts: Optional[str],
                  cause: str, *, attempt_id: Optional[str] = None, **fields: Any) -> None:
    """DIAGNOSTIC detail for an attempt that produced no acted-on verdict: engine off, superseded
    during the debounce, cancelled by an edit, the classifier failing, or an error in the gate.

    NEVER terminal — the attempt is closed by its `visible_action`, which echoes the same cause
    with none of the detail. This event exists because the terminal event cannot carry a
    survivor_ts or an exception type without turning into a grab bag.

    Distinct from a verdict of `ignore`, which IS a decision: conflating the two would read a
    burst collapse as the model choosing silence. Equally distinct from `action_error`, which is
    not a decline at all in the classifier sense — a verdict existed and the ACTION failed. See
    the cause list in the module docstring."""
    _soft_check(cause, DECLINE_CAUSES, "gate_declined cause")
    record("gate_declined", channel_id=channel_id, trigger_ts=trigger_ts,
           cause=cause, attempt_id=attempt_id, **fields)


def gate_decision(channel_id: Optional[str], trigger_ts: Optional[str], *,
                  wake: bool, attempt_id: Optional[str] = None,
                  gate_ms: Optional[int] = None,
                  classifier_ms: Optional[int] = None,
                  source_count: Optional[int] = None,
                  newest_source_ts: Optional[str] = None) -> None:
    """The gate's decision: one bit, and four facts about the call that produced it.

    NOT emitted when the classifier failed — that is a decline, and there is no decision to
    record. (The rich gate manufactured a fail-safe verdict in that case and logged it here,
    which scored a provider outage as the model choosing restraint.)

    `gate_ms` is TOTAL gate wall time and includes the debounce sleep; `classifier_ms` is the model
    call alone. Only the second is a latency number about the model, and reading the first as one
    blames the provider for a debounce we chose.

    `source_count` and `newest_source_ts` describe the COHORT this bit was decided over — how many
    messages were coalesced, and the newest one included. Without them a wake on a five-message
    burst is indistinguishable from a wake on one message, which is the difference between the
    debounce working and the debounce dropping things.

    What is deliberately absent: an action, an emoji, a placement, a reason, the staged findings,
    the overrules, the backoff taxonomy. The gate decides none of those, so recording them would
    describe a decision nobody made. A reason is doubly excluded — it was free prose about a
    person's message, and it is not written anywhere in this build.
    """
    record(
        "gate_decision", channel_id=channel_id, trigger_ts=trigger_ts,
        attempt_id=attempt_id,
        wake=bool(wake),
        # WHICH model produced it. The utility model changes independently of this contract, and a
        # decision-quality comparison across a model swap is meaningless without it.
        model=getattr(config, "utility_model", None),
        gate_ms=gate_ms,
        classifier_ms=classifier_ms,
        source_count=source_count,
        newest_source_ts=newest_source_ts,
    )


def stale_send(channel_id: Optional[str], trigger_ts: Optional[str], *,
               attempt_id: Optional[str] = None, turn_id: Optional[str] = None,
               last_seen_ts: Any = None,
               observed_latest_ts: Any = None, scope: Optional[str] = None,
               surface: Optional[str] = None, guard_mode: Optional[str] = None) -> None:
    """One suppression EVENT by the stale-send guard. DIAGNOSTIC, never terminal.

    CARDINALITY (v9): one row per suppression EVENT, not per suppressed turn — the initiating
    refusal, each per-pass re-race inside the reconsideration runner, and a post-run once-gate
    suppression each emit exactly one row. Single-owner rule: the runner emits for every
    suppression it handles and marks the exception `telemetry_recorded`; an UNMARKED suppression
    is emitted by main.py's terminal catch, so no suppression is counted twice or lost.

    Emitted for every suppression, INCLUDING on turns the gate never judged — mentions, DMs and
    thread continuations mint no attempt, so those rows carry no `attempt_id` and belong to no
    gate population. Any rate computed against `gate_start` has to exclude them, for the same
    reason the ledger excludes those turns everywhere else.

    `turn_id` (v9) joins the row to the CHANNEL-TURN population, so a turn's suppression rows
    can be counted beside its `reconsider_start`s. DM suppressions carry one too, with no
    channel `turn_start` to join — the checker tolerates those rows by name.

    The terminal event says what the room saw; this says why, and with what evidence: the ts
    this turn had accounted for, the newer one that overtook it, which scope noticed, which
    surface was refused, and whether the turn was buffering to a single send or streaming."""
    record("stale_send", channel_id=channel_id, trigger_ts=trigger_ts, attempt_id=attempt_id,
           turn_id=turn_id,
           last_seen_ts=last_seen_ts, observed_latest_ts=observed_latest_ts,
           scope=scope, surface=surface, guard_mode=guard_mode)


def reconsider_start(channel_id: Optional[str], trigger_ts: Optional[str], *,
                     turn_id: Optional[str], pass_number: int, scope: Any = None,
                     observed_latest_ts: Any = None, attempt_id: Optional[str] = None,
                     model_attempt_seq: Optional[int] = None) -> None:
    """One reconsideration pass opened (v9). Emitted via the structured-decision wrapper's
    `on_attempt_open` callback, after `ModelAttemptSink.open()` and before the request.

    `turn_id` is the primary turn-population join — mandatory in the grammar. `pass_number`
    lands on the wire as the literal key `pass`, counting from 1 and contiguous per turn.
    `scope` is the suppressing scope as the FULL three-part tuple, written as a JSON list —
    unlike `stale_send`, which keeps its `scope[0]`-only field. `attempt_id` is absent on
    ungated channel turns; `model_attempt_seq` is absent when the attempt sink failed to open
    (telemetry never blocks the model call). Unavailable optional fields are OMITTED — the
    drop-None rule, never a null."""
    record("reconsider_start", channel_id=channel_id, trigger_ts=trigger_ts,
           turn_id=turn_id, attempt_id=attempt_id,
           scope=list(scope) if scope is not None else None,
           observed_latest_ts=observed_latest_ts,
           model_attempt_seq=model_attempt_seq,
           **{"pass": pass_number})


def reconsider_outcome(channel_id: Optional[str], trigger_ts: Optional[str], *,
                       turn_id: Optional[str], outcome: str, passes: int,
                       attempt_id: Optional[str] = None, forced: Optional[bool] = None,
                       error: Optional[str] = None) -> None:
    """How one reconsideration runner invocation ended (v9). At most one per invocation;
    exactly one on every non-cancelled path.

    `passes` is the number of `reconsider_start` events THIS invocation emitted — a fuse drop
    records 5, a failure or cancellation records the passes started by then. `forced` rides
    only on posted outcomes (`posted_asis`/`posted_revised`) and is written even when False;
    `error` rides only on `error_dropped`, carrying the §4f subtype. Both are OMITTED when
    inapplicable — no nulls. A posted outcome asserts PHYSICAL Slack acceptance of the first
    surface, not finalized turn accounting."""
    _soft_check(outcome, RECONSIDER_OUTCOMES, "reconsider_outcome outcome")
    record("reconsider_outcome", channel_id=channel_id, trigger_ts=trigger_ts,
           turn_id=turn_id, attempt_id=attempt_id, outcome=outcome, passes=passes,
           forced=forced, error=error)


def queue_link(source_attempt_id: str, *, batched_into_attempt_id: Optional[str],
               batched_into_channel_id: Optional[str],
               batched_into_trigger_ts: Optional[str],
               batched_into_gate_required: bool) -> None:
    """One queued attempt, and the later turn that answered on its behalf.

    DIAGNOSTIC and never terminal: the queued attempt already closed itself with `kind=queued`,
    and this adds no second outcome to it. It must stay out of every denominator — counting it
    would put one message in the population twice, which is the exact mistake the `queued`
    exclusion rule exists to prevent.

    `attempt_id` is the SOURCE (the queued message), not the successor, so the join from a
    queued terminal is a lookup on the key it already carries. The successor is described by the
    `batched_into_*` fields, and `batched_into_attempt_id` is absent whenever the successor was
    an ungated turn — there is no attempt to name, and inventing one would put a mention or a DM
    into a ledger documented as excluding them."""
    record("queue_link", attempt_id=source_attempt_id,
           batched_into_attempt_id=batched_into_attempt_id,
           batched_into_channel_id=batched_into_channel_id,
           batched_into_trigger_ts=batched_into_trigger_ts,
           batched_into_gate_required=batched_into_gate_required)


def reaction(channel_id: Optional[str], trigger_ts: Optional[str], *,
             operation: str, result: str, origin: str, emoji: Optional[str],
             target_ts: Optional[str] = None, attempt_id: Optional[str] = None,
             detail: Any = None) -> None:
    """One attempt at one emoji, and what became of it.

    `operation` is add | remove. `result` is added | already_present | refused | failed for an
    add, removed | remove_failed for a remove. The distinctions are load-bearing:

    * `already_present` is NOT `added`. Slack's `already_reacted` and our own reservation guard
      both report success for an emoji somebody else put there; counting those as placements
      would credit the gate with reactions it did not make and flatten every diversity number.
    * `refused` (we declined before calling Slack: reactions off, invalid name, not on the
      allowlist, no target) is not `failed` (we called and it did not land). One is policy and
      one is an outage, and they need different fixes.

    `origin` is which decision chose it. In THIS build only two are emitted: `responder` (the
    model called react_to_message) and `work_claim` (the 👀 staked by slow work). `gate` and
    `backoff_ack` are v2–v6 history — the rich gate could place a reaction itself, and could
    acknowledge participation feedback with one; the binary gate puts nothing in the room. The
    names stay legal so a reader of those older rows is not told they are invalid.

    Diversity has to be measured per origin, and pooling across the contract boundary is exactly
    the mistake to avoid: the old gate picked blind from one prompt, the responder can search the
    catalog, and the work claim is a fixed operational marker that is not taste at all.

    `target_ts` is the message the emoji went ON, kept separate from `trigger_ts` (the message
    that caused the turn). The react tool may target an older message, and collapsing the two
    silently reattributes those reactions to the wrong message."""
    _soft_check(operation, REACTION_OPERATIONS, "reaction operation")
    _soft_check(result, REACTION_RESULTS, "reaction result")
    _soft_check(origin, REACTION_ORIGINS, "reaction origin")
    record("reaction", channel_id=channel_id, trigger_ts=trigger_ts,
           operation=operation, result=result, origin=origin, emoji=emoji,
           target_ts=target_ts, attempt_id=attempt_id, detail=detail)


def visible_action(channel_id: Optional[str], trigger_ts: Optional[str], *,
                   kind: str, **fields: Any) -> None:
    """THE TERMINAL EVENT. Prefer `finish_attempt`, which is the only caller that can guarantee
    exactly one of these per attempt.

    `ended_by` names the STAGE THAT CLOSED THE ATTEMPT — `gate`, `responder`, or `stale_guard`
    (the send guard refused the turn's first surface because a newer message had arrived) — and
    nothing more. It is not a claim that that stage made a judgment: engine-off, a classifier failure, a
    gate action that raised, and a queued turn all close at a stage without anyone deciding
    anything. It was called `decided_by`, which read as "who decided", and every one of those
    rows was a lie in that reading. To ask who DECIDED, read `gate_decision` (a verdict exists
    only when the model produced one) or `gate_woke` (the gate handed the turn on).

    Other fields the terminal may carry:
      reaction_visible    was one of OUR emoji on the message when the turn ended (the gate's or
                          the responder's). Explicit true/false on EVERY responder terminal, not
                          only on replies: read beside `silence_reason` it is what makes a model
                          that declared `reacted_instead` and then placed no reaction visible as
                          a mismatch. Neither field is corrected against the other — the
                          disagreement IS the measurement, and a ledger that quietly reconciled
                          them could never report it.
      post_delivery_error the words landed and something AFTER the send raised. `kind` keeps the
                          delivered outcome — Slack shows the reply, so the ledger must too —
                          and this says the turn was not clean.
      detached_started    a producer owned a surface AND the turn errored afterwards.
      destination         WHERE the words went: thread | channel | dm. Present only on a
                          terminal that delivered something — a silence has no destination, and
                          the field it replaces (`placement`) claimed one anyway.
      destination_source  WHO decided: `structural` (the route left no choice — a DM, a reply
                          inside a thread, a channel that forbids top-level replies), `model`
                          (it called set_reply_destination), or `default` (it was offered the
                          choice and produced words without making one).
      destination_contract_miss
                          written ONLY when true: the model had the choice and skipped it. The
                          answer was still delivered, in the default thread — this is a prompt
                          problem worth counting, not a delivery failure.
      silence_reason      the model's own account of why it used no words: ONE of the eight
                          values in message_processor.terminal_actions, recorded verbatim, on
                          reaction_only and detached as well as silence. Never inferred from the
                          text, the reaction, the tools or the posture, and an unrecognized value
                          never reaches here — it is rejected at the tool call, so this column
                          only ever holds something the model actually chose.

    `kind` is what the room actually saw:
      reply            words went out
      delivery_failed  words were written and the send did not land — the opposite of silence,
                       and previously filed as `reply` because content was non-empty
      silence          a deliberate no-reply (the gate's `ignore`, or the terminal tool)
      reaction_only    an emoji WAS the answer — a gate `react` that landed, a handled backoff
                       whose ack landed, or a no-reply turn that committed a reaction
      detached         a producer that owns its own surface (generate_image posts the picture,
                       a background job posts its status card) so the Response is empty by design
      queued           never ran; another turn owns this conversation
      interrupted      died partway; the thread got an apology, not an answer
      error            the turn errored (delivery of the error notice itself is NOT observable
                       from here, so this says "we tried to apologize", not "they saw it")
      error_unhandled  the exception escaped all the way to handle_message's own except-clause
      aborted          the attempt left without ever closing itself — cancellation, or a raise
                       before the responder
      empty            the RESPONDER produced nothing usable: it posted nothing and never called
                       the terminal tool, or it handed back no Response object at all
                       (detail=no_response_object). A contract violation, and the single most
                       important thing here to keep apart from a chosen silence — they are
                       identical in the room and opposite in meaning
      stale_suppressed the turn had an answer and the guard refused to create its first
                       surface, because a newer message in the same conversation arrived while
                       it was writing. NOT a silence: nobody chose it and there is no silence
                       reason. NOT delivery_failed: Slack was never called, so nothing failed.
                       Only when the suppression is the whole story — a turn that still shows a
                       reaction or a detached surface takes that label and carries
                       `reply_stale_suppressed` beside it
      none             the GATE ended the turn without any visible act at all. Kept apart from
                       `empty` on purpose: one is a decision path completing with nothing to
                       show, the other is the responder contract breaking
    """
    record("visible_action", channel_id=channel_id, trigger_ts=trigger_ts, kind=kind, **fields)


# --------------------------------------------------------------- v8: the turn population

# The exactly-once guard for turn_outcome, stamped on the TurnRuntime itself. main.py's outer
# finally cannot run twice, so this is insurance against a second emitter appearing later — the
# same reason `finish_attempt` flips a flag rather than trusting its callers.
TURN_OUTCOME_KEY = "_pt_turn_outcome_emitted"


def turn_start(channel_id: Optional[str], trigger_ts: Optional[str], *, turn_id: Optional[str],
               origin_thread_ts: Optional[str] = None, surface: Optional[str] = None,
               gated: bool = False, attempt_id: Optional[str] = None,
               wake_source: Optional[str] = None) -> None:
    """A channel turn opened. THE DENOMINATOR OF THE TURN POPULATION.

    Emitted for every channel turn, gated or not, which is exactly what makes it a different
    denominator from `gate_start`: a mention and a thread continuation are turns with no attempt,
    and a ledger that could only count judged messages could never say what share of the bot's
    channel output the gate is responsible for. `gated` is that split, on the row.
    """
    _soft_check(surface, TURN_SURFACES, "turn surface")
    record("turn_start", channel_id=channel_id, trigger_ts=trigger_ts, turn_id=turn_id,
           origin_thread_ts=origin_thread_ts, surface=surface, gated=bool(gated),
           attempt_id=attempt_id, wake_source=wake_source)


def turn_outcome(channel_id: Optional[str], trigger_ts: Optional[str], *,
                 turn_id: Optional[str], kind: str, destinations: Optional[List[Dict]] = None,
                 chars: Optional[int] = None, detached_started: bool = False,
                 error: Optional[str] = None, H: Optional[str] = None,
                 stream_build_present: bool = False,
                 attempt_id: Optional[str] = None,
                 reconsider: Optional[Dict[str, Any]] = None,
                 edits: Optional[List[Dict[str, Any]]] = None,
                 destination_contract_miss: Optional[bool] = None) -> None:
    """What one channel turn ended up doing. Exactly one per `turn_start`.

    `destinations` is the OBSERVED set — every surface Slack accepted, whether or not it reached
    its final text — because that is what the room saw, with ONE ruled exception: the zero-chunk
    truncation notice. When every first-chunk attempt fails, `send_message` may still post a
    truncation warning — a finalized turn-owned receipt — without committing the lease; that
    notice's ts never reaches `turn.destinations` (no `on_first_accept`, `first_ts=None`), so it
    is absent here even though Slack accepted it. Pre-existing on every zero-chunk send, and
    RULED accepted rather than plumbed into accounting (Docs/specs/STALE_RECONSIDERATION.md
    §4a, r5-2). Memory extraction reads the COMMITTED
    subset instead (main.py), and the difference between the two lists is precisely an interrupted
    reply. It is written even when empty: a silent turn and a turn whose records were lost are not
    the same fact.

    `reconsider` (v9) is the nested facts of the turn's reconsideration run —
    `{outcome, passes[, forced][, error]}`, ReconsiderFacts.as_payload() verbatim — absent when
    no reconsideration ran. Inapplicable keys are omitted inside it, never null: nested values
    survive `record()`'s top-level drop-None untouched.

    `edits` (v10) is ALWAYS present as a list — written even when empty, for the same reason
    `destinations` is: a turn that edited nothing and a turn whose edit records were lost are
    not the same fact. One entry per EditRecord on the turn:
    `{"channel_id", "target_ts", "announcement_ts", "state":
    "announcement_only"|"committed"[, "error"]}` — a record exists only once the disclosure
    was ACCEPTED (spec §11.6/§11.18), so `announcement_ts` is ALWAYS present; only `error`
    is OMITTED when unavailable, never null (nested values survive record()'s top-level
    drop-None, so the omission happens at assembly). Every entry's `announcement_ts` joins a
    committed `correction_announcement` destination in this row's `destinations` — the
    disclosure post is a real destination the room saw. `reconsider` and `edits` may coexist
    on one row.

    `error` is one of the FOUR fail-closed codes (stream_over_budget, history_fetch_failed,
    stream_data_invalid, and origin_fetch_failed — a partial origin thread is a different
    conversation, not a smaller one), absent on a turn that did not fail that way.
    `coverage_not_ready` and `snapshot_unsupported` were retired with the machinery that raised
    them; rows in older ledgers still carry them and stay valid under their own contract, while
    a FRESH row carrying one is a violation the checker reports.
    `stream_build_present` says whether the room was actually rendered — a turn that failed before
    the fetch answered from nothing, and its `kind` must not be read as a judgment.

    `destination_contract_miss` (W4) is the battery metric for the first-token marker: the model
    was offered both destinations, produced words, and wrote no marker — so the answer went to
    the default thread and this says the PROMPT missed, not the delivery. Written ONLY when
    true, the same convention the `visible_action` copy of this field follows and for the same
    reason: a column that is false on nearly every row costs bytes on every line to say nothing.
    It rides HERE as well as on `finish_attempt` because the two populations do not overlap where
    it matters — `finish_attempt` belongs to the gate population, and the ungated channel turns
    that make up most of the selectable ones never emit one, so a miss rate read from the
    terminal alone would be blind to exactly the turns the marker was built for.
    """
    _soft_check(kind, KINDS, "turn_outcome kind")
    record("turn_outcome", channel_id=channel_id, trigger_ts=trigger_ts, turn_id=turn_id,
           attempt_id=attempt_id, kind=kind,
           destinations=list(destinations or []),
           chars=chars, detached_started=bool(detached_started), error=error, H=H,
           stream_build_present=bool(stream_build_present), reconsider=reconsider,
           edits=list(edits or []),
           destination_contract_miss=True if destination_contract_miss else None)


def emit_turn_outcome(turn: Any, *, channel_id: Optional[str], trigger_ts: Optional[str],
                      kind: str, detached_started: bool = False,
                      attempt_id: Optional[str] = None) -> bool:
    """Assemble a turn's outcome from the turn itself. True when this call emitted it.

    Reads the TurnRuntime rather than being handed a payload, so the outer finally cannot report
    a destination set that disagrees with the one the handlers actually recorded. Duck-typed on
    purpose: this module must not import turn_runtime, which imports it.
    """
    try:
        if turn is None or getattr(turn, TURN_OUTCOME_KEY, False):
            return False
        setattr(turn, TURN_OUTCOME_KEY, True)
        records = list(getattr(turn, "destinations", None) or [])
        payloads = [r.as_payload() for r in records if hasattr(r, "as_payload")]
        sizes = [p.get("chars") for p in payloads if p.get("chars") is not None]
        # v9. The reconsideration facts the runner stamped on the runtime, absent when none ran.
        facts = getattr(turn, "reconsider", None)
        reconsider = (facts.as_payload()
                      if facts is not None and hasattr(facts, "as_payload") else None)
        # v10. One entry per EditRecord on `turn.edits` (spec §7) — the exact payload, with
        # `announcement_ts` and `error` OMITTED when None: nested values survive record()'s
        # top-level drop-None, so the no-null rule has to be applied here.
        edits: List[Dict[str, Any]] = []
        for edit in (getattr(turn, "edits", None) or []):
            entry: Dict[str, Any] = {
                "channel_id": getattr(edit, "channel_id", None),
                "target_ts": getattr(edit, "target_ts", None),
                "state": getattr(edit, "state", None),
            }
            if getattr(edit, "announcement_ts", None) is not None:
                entry["announcement_ts"] = edit.announcement_ts
            if getattr(edit, "error", None) is not None:
                entry["error"] = edit.error
            edits.append(entry)
        turn_outcome(
            channel_id, trigger_ts,
            turn_id=getattr(turn, "turn_id", None), kind=kind, destinations=payloads,
            chars=sum(sizes) if sizes else None, detached_started=detached_started,
            error=getattr(turn, "turn_error", None), H=getattr(turn, "H", None),
            stream_build_present=bool(getattr(turn, "stream_build_present", False)),
            attempt_id=attempt_id, reconsider=reconsider, edits=edits,
            # W4. Read off the runtime like everything else here, so the row cannot disagree
            # with the turn that settled it.
            destination_contract_miss=bool(
                getattr(turn, "destination_contract_miss", False)))
        return True
    except Exception as e:  # noqa: BLE001 — a lost line is never worth a lost turn
        logger.debug(f"Participation turn outcome not written: {e}")
        return False


def stream_render(*, turn_id: Optional[str], origin_thread_ts: Optional[str] = None,
                  trigger_ts: Optional[str] = None, **fields: Any) -> None:
    """One channel stream build, identified by its hashes.

    `**fields` is ChannelStream.stream_render_fields() verbatim — the pinned window, the SEVEN
    hashes that make two builds comparable, and the byte/message cost. Passing the dict through
    rather than restating its keys here is deliberate: the serializer owns what identifies a
    build, and a second enumeration of those keys is a second thing to keep in step.
    """
    channel_id = fields.pop("channel_id", None)
    record("stream_render", channel_id=channel_id, trigger_ts=trigger_ts, turn_id=turn_id,
           origin_thread_ts=origin_thread_ts, **fields)


def model_response(*, turn_id: Optional[str], attempt_seq: Optional[int],
                   model: Optional[str] = None, fork_reason: Optional[str] = None,
                   status: Optional[str] = None, cached_input_tokens: Optional[int] = None,
                   input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                   detail: Any = None) -> None:
    """One Responses API attempt on a channel turn — the call, not the turn.

    Written on failure as well as success, because the question this answers is "how many calls
    did this turn cost and why", and an attempt that raised is exactly the expensive kind. Each
    tool-loop round is its own attempt: the loop issues one API call per round. A channel turn's
    reconsideration passes are attempts of the SAME turn with `fork_reason` =
    `stale_reconsideration` (v9), so their tokens and failures land here beside the responder's.

    `cached_input_tokens` is the provider's own `input_tokens_details.cached_tokens`, which is the
    only evidence that the pinned-prefix cache key is doing anything at all.
    """
    _soft_check(status, MODEL_RESPONSE_STATUSES, "model_response status")
    record("model_response", turn_id=turn_id, attempt_seq=attempt_seq, model=model,
           fork_reason=fork_reason, status=status, cached_input_tokens=cached_input_tokens,
           input_tokens=input_tokens, output_tokens=output_tokens, detail=detail)


@dataclass
class ModelAttemptSink:
    """The per-turn carrier that gets `model_response` out of the API layer.

    Threaded alongside `usage_sink` through the five request-issuing wrappers, so the layer that
    knows a call happened does not also have to know what a turn is. `turn` sequences the attempts
    (TurnRuntime.next_model_attempt); `fork_reason` says why this handler entry is issuing calls at
    all and is fixed for every call it makes. None of these methods raise.
    """

    turn: Any
    fork_reason: Optional[str] = None

    def open(self, model: Optional[str] = None) -> Any:
        try:
            return self.turn.next_model_attempt(model=model, fork_reason=self.fork_reason)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Model attempt not opened: {e}")
            return None

    def close(self, attempt: Any, *, status: str, input_tokens: Optional[int] = None,
              output_tokens: Optional[int] = None, cached_input_tokens: Optional[int] = None,
              detail: Any = None) -> None:
        try:
            if attempt is None:
                return
            attempt.status = status
            attempt.input_tokens = input_tokens
            attempt.output_tokens = output_tokens
            attempt.cached_input_tokens = cached_input_tokens
            model_response(
                turn_id=getattr(self.turn, "turn_id", None),
                attempt_seq=getattr(attempt, "attempt_seq", None),
                model=getattr(attempt, "model", None),
                fork_reason=getattr(attempt, "fork_reason", None),
                status=status, cached_input_tokens=cached_input_tokens,
                input_tokens=input_tokens, output_tokens=output_tokens, detail=detail)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Model attempt not closed: {e}")


def outbound_receipt(*, channel_id: Optional[str], message_ts: Optional[str],
                     owner_turn_id: Optional[str], op: str,
                     prior_state: Optional[str] = None, new_state: Optional[str] = None,
                     applied: bool = False, reason: Optional[str] = None) -> None:
    """One receipt state transition (spec §5), as the DATABASE resolved it.

    Written from the transition's own result rather than from the caller's intent, which is the
    whole point: a register the lattice absorbed, a finalize a foreign turn already owns and a
    demote that found nothing to demote all look identical from the call site and are three
    different facts about the stream. `applied=false` with a `reason` is the interesting row.
    """
    _soft_check(op, RECEIPT_OPS, "outbound_receipt op")
    _soft_check(prior_state, RECEIPT_STATES, "outbound_receipt prior_state")
    _soft_check(new_state, RECEIPT_STATES, "outbound_receipt new_state")
    record("outbound_receipt", channel_id=channel_id, message_ts=message_ts,
           owner_turn_id=owner_turn_id, op=op, prior_state=prior_state, new_state=new_state,
           applied=bool(applied), reason=reason)
