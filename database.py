"""
SQLite Database Manager for ChatGPT Bots
Provides persistent storage for threads, messages, images, documents, and user preferences
"""

import sqlite3
import aiosqlite
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Dict, Iterable, List, Any, Tuple, cast
import logging
import asyncio
from config import dev_epoch_fence_requested
from logger import LoggerMixin

logger = logging.getLogger(__name__)

# Sentinel distinguishing "argument omitted → preserve existing value" from an explicit
# None (→ clear the column to NULL). Used by the channel_settings setters so the settings
# modal's "inherit from global default" selection stores NULL rather than a literal string.
_UNSET = object()

# bot_meta key holding the outbound-receipts feature epoch (unix seconds, "%.6f"). Own-messages
# posted before it are grandfathered into the channel stream without a receipt. Written once,
# ever, via set_meta_if_absent_async — see the bot_meta design note in init_schema.
OUTBOUND_RECEIPTS_EPOCH_KEY = "outbound_receipts_feature_epoch_ts"

# bot_meta keys recording that the two one-shot respec §3 migrations have run.
_P4A_CLEANUP_KEY = "p4a_compaction_schema_dropped"
_COVERAGE_RENAME_KEY = "channel_coverage_renamed_to_inventory_at"

# Receipt states, in the order the state machine allows: chrome and in_flight may promote,
# finalized is absorbing.
_RECEIPT_STATES = ("in_flight", "finalized", "chrome")

# Receipt CLASSES (EDIT_OWN_MESSAGE spec §4): what KIND of surface a receipt describes, distinct
# from its lifecycle state. Closed enum, validated here in Python rather than by a CHECK
# constraint so legacy rows keep their NULL — a NULL class is "stamped before the feature
# existed" and is ineligible for anything class-gated; it is NEVER inferred from state, owner,
# text or epoch. Every producer stamps a class explicitly; the only sanctioned transitions are
# placeholder promotion (chrome → assistant_reply) and retry demotion (back to chrome) — any
# other same-message class conflict fails closed and logs.
_RECEIPT_CLASSES = ("assistant_reply", "correction_announcement", "system_notice",
                    "background_job", "artifact", "chrome")


def _check_receipt_class(receipt_class: Optional[str]) -> Optional[str]:
    """Validate one class value. None is legal at this layer (legacy/defensive rows)."""
    if receipt_class is not None and receipt_class not in _RECEIPT_CLASSES:
        raise ValueError(f"invalid receipt class: {receipt_class!r}")
    return receipt_class

# Ids per `IN (...)` statement in the render pin. SQLite's default variable limit is 999 and a
# channel turn can legitimately pin more identities than that, so the read chunks rather than
# letting a platform limit decide whether a turn renders.
_SIDECAR_ID_CHUNK = 500


def _ts_sort(raw: Any) -> float:
    """A Slack ts as a float, for re-sorting rows that arrived across several chunks.

    Sorting only — never a window comparison. The accessors compare timestamps as REAL in SQL,
    and this exists to restore the ORDER BY that chunking split, so a one-chunk read and a
    five-chunk read over the same ids return the same order.
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class TransitionResult:
    """What a receipt write actually did, not what its caller intended.

    A bool answered "did the row end up mine", which is the same answer for a finalize the lattice
    absorbed, a row a foreign turn holds and a chrome registration the state machine refused —
    three different facts about the stream, and the refusals are the interesting ones (telemetry
    §5, `participation_telemetry.outbound_receipt`).

    `prior_state` speaks the ledger's vocabulary (`absent | in_flight | finalized | chrome`), where
    `absent` means no row existed. `__bool__` is a safety net for a call site that slipped through
    the audit, NOT the intended read: every caller reads `.applied`.
    """

    applied: bool
    prior_state: Optional[str] = None
    new_state: Optional[str] = None
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.applied

# A sweep token is abandoned if its holder has not heartbeat within this window.
_COVERAGE_SWEEP_STALE_MINUTES = 10

# The ONE thread-activity merge rule: both ts columns only ever move forward, and the advisory
# count never moves downward.
_ACTIVITY_UPSERT_SQL = """
    INSERT INTO channel_thread_activity
        (team_id, channel_id, root_ts, last_observed_reply_ts, advisory_reply_count,
         last_index_event_ts, dirty, updated_ts)
    VALUES (:team, :ch, :root, :reply_ts, :count, :event_ts, :dirty, CURRENT_TIMESTAMP)
    ON CONFLICT(team_id, channel_id, root_ts) DO UPDATE SET
        last_observed_reply_ts = CASE
            WHEN excluded.last_observed_reply_ts IS NULL
                THEN channel_thread_activity.last_observed_reply_ts
            WHEN channel_thread_activity.last_observed_reply_ts IS NULL
                THEN excluded.last_observed_reply_ts
            WHEN CAST(excluded.last_observed_reply_ts AS REAL)
                 > CAST(channel_thread_activity.last_observed_reply_ts AS REAL)
                THEN excluded.last_observed_reply_ts
            ELSE channel_thread_activity.last_observed_reply_ts END,
        last_index_event_ts = CASE
            WHEN excluded.last_index_event_ts IS NULL
                THEN channel_thread_activity.last_index_event_ts
            WHEN channel_thread_activity.last_index_event_ts IS NULL
                THEN excluded.last_index_event_ts
            WHEN CAST(excluded.last_index_event_ts AS REAL)
                 > CAST(channel_thread_activity.last_index_event_ts AS REAL)
                THEN excluded.last_index_event_ts
            ELSE channel_thread_activity.last_index_event_ts END,
        advisory_reply_count = CASE
            WHEN excluded.advisory_reply_count IS NULL
                THEN channel_thread_activity.advisory_reply_count
            WHEN channel_thread_activity.advisory_reply_count IS NOT NULL
                 AND excluded.advisory_reply_count
                     < channel_thread_activity.advisory_reply_count
                THEN channel_thread_activity.advisory_reply_count
            WHEN channel_thread_activity.last_index_event_ts IS NOT NULL
                 AND (:obs_ts IS NULL
                      OR CAST(:obs_ts AS REAL)
                         < CAST(channel_thread_activity.last_index_event_ts AS REAL))
                THEN channel_thread_activity.advisory_reply_count
            ELSE excluded.advisory_reply_count END,
        dirty = CASE
            WHEN excluded.dirty = 1 OR channel_thread_activity.dirty = 1 THEN 1
            ELSE 0 END,
        updated_ts = CURRENT_TIMESTAMP
"""


def _activity_upsert_params(team_id, channel_id, root_ts, reply_ts, reply_count, event_ts,
                            mark_dirty):
    """Bind one activity observation. A count with no latest_reply marks the root dirty:
    we know there are replies but not where."""
    if reply_count is not None and reply_count > 0 and not reply_ts:
        mark_dirty = True
    return {
        "team": team_id,
        "ch": channel_id,
        "root": str(root_ts),
        "reply_ts": str(reply_ts) if reply_ts else None,
        "count": reply_count,
        "event_ts": str(event_ts) if event_ts else None,
        "obs_ts": str(event_ts or reply_ts) if (event_ts or reply_ts) else None,
        "dirty": 1 if mark_dirty else 0,
    }

# The one terminal coverage verdict that can reverse (re-invite, unarchive), so it is the only
# one reset_channel_coverage_async will demote. Mirrored by activity_index._UNAVAILABLE_REASON.
_COVERAGE_UNAVAILABLE_REASON = "unavailable"

# F51c — cap on late-artifact addenda folded onto ONE thread's compaction summary head, so a
# pathological channel can't bloat the head unboundedly. Each note is already length-capped at
# render time; this caps the COUNT. Single source of truth for both the completion-time
# (ambient service) and compaction-time (thread management) capture paths.
_MAX_SUMMARY_ADDENDA_PER_THREAD = 20

# The channel-only scope for the documents uniqueness rule. DM thread ids start with "D"/"U"/"W";
# a channel thread key is "<C…|G…>:<thread_ts>". Restricting BOTH the dedup pass and the partial
# index to that scope is deliberate: a DM re-attach is a genuinely new observation and its rows
# must keep behaving exactly as they always have, while a channel turn re-ingests its origin
# thread every single turn and would otherwise add a row per turn forever.
_CHANNEL_DOCS_PREDICATE = ("file_id IS NOT NULL "
                           "AND (thread_id LIKE 'C%' OR thread_id LIKE 'G%')")
_CHANNEL_DOCS_INDEX = "idx_documents_channel_message_file"

# The unattended-file placeholder. `catalog_unattended` records a file nobody read yet with this
# summary so the bytes stay REACHABLE; it is not a summary, and treating it as one would have the
# serializer tell the model a document has been read when it has not, and the dedup migration
# keep a placeholder over a real summary. ONE definition — the writer's literal, the migration's
# predicate and the serializer's eligibility rule are all this.
UNATTENDED_SUMMARY_TEMPLATE = "Shared in this conversation ({name}). Not yet read."
_UNATTENDED_PREFIX = "Shared in this conversation ("
_UNATTENDED_SUFFIX = "). Not yet read."
# The same shape as a SQL LIKE pattern, for the one rule that has to hold inside a statement:
# an upgrading write may not replace a real summary with the placeholder.
_UNATTENDED_LIKE = f"{_UNATTENDED_PREFIX}%{_UNATTENDED_SUFFIX}"


def is_unattended_summary(summary: Optional[str]) -> bool:
    """True for the placeholder `catalog_unattended` writes instead of a real summary."""
    text = (summary or "").strip()
    return (text.startswith(_UNATTENDED_PREFIX) and text.endswith(_UNATTENDED_SUFFIX)
            and len(text) > len(_UNATTENDED_PREFIX) + len(_UNATTENDED_SUFFIX))


# Shared hash/normalize contract for channel-memory reconciliation. The settings modal builder,
# the submit handler, and reconcile_channel_memory_from_textarea_async ALL route content through
# these two functions so a content hash computed at modal-open matches one recomputed at submit —
# that identity is how a seeded row is matched (keep), missed (delete), or changed (conflict).
def normalize_memory_line(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines) to a single space and strip ends.

    Blank or whitespace-only input (and None) returns "". This is the single normalization all
    three call sites share, so a legacy multi-line fact and its single-line textarea rendering
    hash to the same value.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def merged_policy_text(current: str, incoming: str) -> Optional[str]:
    """Both texts, or one, or neither — the rule for folding an incoming policy into a stored one.

    Used by the directives migration, the preference-row migration, and by a legacy modal's
    submission — the only paths that CANNOT know what the stored policy is (two predate it, one
    predates the field). Guessing which text the operator meant is not their call, and silently
    dropping either is how a live rule disappears — so both survive as separate lines,
    deduplicated only on exact equality after whitespace normalization.

    Dedup is per LINE, not per blob. One incoming text was enough while the directives column was
    the only legacy source: one column, one rule, and `current` was whatever the operator typed.
    The preference migration folds SEVERAL legacy rows into the same policy one after another, so
    by the second fold `current` is already multi-line — and a blob comparison would happily
    append a line that sits verbatim two lines above it. The rule itself is unchanged (exact
    equality after whitespace normalization); only the unit it applies to. Every single-line case,
    which is every pre-existing caller's case, folds exactly as it always did.

    Returns None when there is nothing to store. Note the asymmetry with a deliberate REPLACE: a
    writer that can see the current policy replaces it wholesale; a writer that cannot, merges.
    """
    current = (current or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return current or None
    if not current:
        return incoming
    # `current` is kept verbatim — it is what a human last saw and may contain its own blank
    # lines or indentation. Only the INCOMING lines are filtered and normalized.
    seen = {normalize_memory_line(line) for line in current.splitlines()}
    seen.discard("")
    additions: List[str] = []
    for line in incoming.splitlines():
        key = normalize_memory_line(line)
        if not key or key in seen:
            continue                        # the same rule arriving twice
        seen.add(key)                       # …and twice within one incoming text
        additions.append(line.strip())
    if not additions:
        return current
    return "\n".join([current, *additions])


def memory_content_hash(text: str) -> str:
    """Stable short identity for a memory line: sha256 hexdigest of the normalized text, [:16].

    Paired with normalize_memory_line so text that is equal after normalization hashes equal.
    """
    return hashlib.sha256(normalize_memory_line(text).encode()).hexdigest()[:16]


# --- the dev-only epoch overlay (Docs/specs/EPOCH_FENCE_SPEC.md §7.1d, §7.1h-bis) -----------
#
# THE DERIVED-STATE RULE. A production row may persist under a fence IFF it records an OBSERVABLE
# FACT about a real Slack object — the message really was posted, the thread really has a reply —
# because those objects exist whether or not a fence is up. POLICY, SELECTION and PREFERENCE state
# is OVERLAID: it is the state that decides what a LATER run sees, so carrying it across a battery
# boundary is exactly how one run contaminates the next. Receipts, the activity index, artifacts
# and the coverage inventory therefore still write through; settings, standing policy, channel
# memory, `channel_window_anchor` and `response_feedback` are redirected below.
#
# The redirect lives at the ACCESSOR rather than at each call site: it is one place instead of
# dozens, and it reaches the reads inside the channel-stream build without that module needing to
# know a fence exists. An unfenced process runs identical code to before — the guard is a single
# empty-dict test inside `overlay_for_channel`, and that dict cannot become non-empty without
# `DEV_EPOCH_FENCE_ENABLE`.

_EPOCH_UNRESOLVED = object()
_epoch_hooks: Dict[str, Any] = {}


def _epoch_hook(name: str):
    """One attribute of the fence module, or None when no fence is requested.

    THE FLAG IS TESTED BEFORE THE IMPORT, on every call, and that ordering is the whole point:
    without it these accessors would pull `message_processor.epoch_fence` into `sys.modules` the
    first time anybody read a channel setting, defeating the import gates in `main.py` and
    `slack_client/messaging.py`. With `DEV_EPOCH_FENCE_ENABLE` empty the fence module is never
    imported through the database layer either, so flag-off really is import-inert everywhere.

    Resolution is cached, the flag test is NOT — the check runs ahead of the cache, so turning the
    flag off mid-process (which the harness demonstrations do) is honoured rather than answered
    from a stale binding. An import failure caches None: a dev harness must never break a
    production read, and `main.initialize` is what refuses to start when the flag asked for one.
    """
    if not dev_epoch_fence_requested():
        return None
    hook = _epoch_hooks.get(name, _EPOCH_UNRESOLVED)
    if hook is _EPOCH_UNRESOLVED:
        try:
            from message_processor import epoch_fence
        except Exception:  # noqa: BLE001
            hook = None
        else:
            hook = getattr(epoch_fence, name, None)
        _epoch_hooks[name] = hook
    return hook


def _epoch_overlay(channel_id, team_id=None):
    """The overlay a durable channel read/write must be served from, or None."""
    hook = _epoch_hook("overlay_for_channel")
    return hook(channel_id, team_id) if hook is not None else None


def _epoch_memory_row(memory_id):
    """The overlay owning a `channel_memory` row id, or None. See `overlay_by_memory_id`."""
    hook = _epoch_hook("overlay_by_memory_id")
    return hook(memory_id) if hook is not None else None


def _epoch_feedback_row(message_ts, user_id, source):
    """The overlay holding a `response_feedback` row, or None. See `overlay_by_feedback`."""
    hook = _epoch_hook("overlay_by_feedback")
    return hook(message_ts, user_id, source) if hook is not None else None


def _epoch_fence_up():
    """True when ANY fence is installed. The cheap gate for the channel-less accessors."""
    hook = _epoch_hook("any_fence_installed")
    return bool(hook()) if hook is not None else False


def _epoch_refuse_production_write(channel_id, what: str) -> None:
    """Raise when `channel_id` is fenced.

    `update_channel_memory_async(id)`, `delete_channel_memory_async(id)` and
    `delete_response_feedback_async(ts, user, source)` name a ROW and no channel, so the overlay
    band cannot route them: an id BELOW the band is a production row, and a production row in a
    FENCED channel is one that predates the battery. Writing it would break the promise that no
    production policy or preference row moves while a fence is up — and it is the one leak the
    band cannot see. Rows in unfenced channels fall straight through; the fence is one channel's
    fence.
    """
    if not _epoch_fence_up() or not channel_id:
        return
    if _epoch_overlay(channel_id) is None:
        return
    EpochEffectRefused = _epoch_hook("EpochEffectRefused")
    if EpochEffectRefused is None:      # unreachable — a fence is up, so the module resolved
        return
    raise EpochEffectRefused(
        f"{what} names a PRODUCTION row in fenced channel {channel_id}; it predates the battery "
        f"and the fence does not rewrite production rows. Nothing was changed.")


def _epoch_settings_fields(**fields):
    """Drop the `_UNSET` sentinels so the overlay applies the SAME partial semantics as the
    production write: only the fields present in the call are written, and a modal submission that
    omits a block preserves the overlay's value for it."""
    return {k: v for k, v in fields.items() if v is not _UNSET}


def _check_clauses(ddl: str) -> List[str]:
    """Every `CHECK (...)` clause in a CREATE TABLE statement, each as its own string.

    Parenthesis-balanced rather than regex, because the constraints this reads contain a nested
    group of their own (`... IN (0, 1)`) and a naive match would stop at the first `)`. Splitting
    the DDL into per-clause strings is what lets a caller ask which COLUMN a check belongs to;
    searching the raw text instead conflates a constrained column with the next one declared
    after it.
    """
    clauses: List[str] = []
    idx = 0
    while True:
        idx = ddl.find("CHECK", idx)
        if idx < 0:
            return clauses
        idx += len("CHECK")
        while idx < len(ddl) and ddl[idx].isspace():
            idx += 1
        if idx >= len(ddl) or ddl[idx] != "(":
            continue                      # the word "CHECK" outside a constraint; keep looking
        start, depth = idx, 0
        while idx < len(ddl):
            if ddl[idx] == "(":
                depth += 1
            elif ddl[idx] == ")":
                depth -= 1
                if depth == 0:
                    idx += 1
                    break
            idx += 1
        clauses.append(ddl[start:idx])


def _decode_muted_threads(raw) -> List[str]:
    """Parse the channel_settings.muted_threads JSON column into a list of thread ts
    strings. Malformed/absent → empty list (fail open — a bad blob never silences)."""
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(t) for t in val] if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def _encode_muted_threads(val) -> Optional[str]:
    """Serialize a list of thread ts strings for storage. None/empty → NULL."""
    if not val:
        return None
    try:
        return json.dumps([str(t) for t in val])
    except (ValueError, TypeError):
        return None


# The participation vocabulary the binary gate retired, and the single value it retired into.
#
# Frozen here on purpose rather than imported from message_processor.participation: a migration
# describes a database as it WAS, and its two legacy names no longer appear in VALID_LEVELS at
# all. Sourcing the target from the live tuple would be worse still — a later rename of the
# vocabulary would silently rewrite what this historical migration claims to have written.
#
# `judicious` and `active` were two dials on a gate that weighed how much value was enough to
# speak. The binary gate does not ask that question, so the two names describe one behavior and
# collapse into it. `off`, `mentions_only` and NULL (inherit the global default) keep their exact
# meaning and are not touched.
_LEGACY_PARTICIPATION_LEVELS = ("judicious", "active")
_PARTICIPATION_LEVEL_ON = "on"

# The author marker the retired backoff writer stamped on its preference rows — the same string as
# channel_steering.PREF_AUTHOR_PREFIX and as the partial unique index's own predicate.
#
# A frozen copy rather than an import, for the same reason as the levels above: this is the name
# rows in an OLD database carry, and the constant it mirrors belongs to a writer that the binary
# gate deletes. A migration that imports its legacy vocabulary from live code stops being able to
# read old data the moment that code is cleaned up — and it would also drag the whole
# message_processor import graph into database.py at migration time, which is a lot of surface for
# one string. If the two ever disagree, the DB's own index predicate is the tiebreaker.
_LEGACY_PREF_AUTHOR_PREFIX = "participation_engine:pref:"

# channel_settings columns whose write is a real, attributed structural edit. Touching any of
# them bumps updated_ts/updated_by; a write that touches only the non-structural columns
# (snoozed_until, the deprecated muted_threads) leaves authorship untouched — so a background
# housekeeping write never looks like the last human to edit the channel's response settings.
#
# `directives` is NOT here, and takes no parameter below: the channel's standing rules live in
# the reserved policy row now (see message_processor/channel_steering.py). The COLUMN survives so
# the migration has something to read on an old database; nothing writes it, and nothing but the
# migration reads it.
_CHANNEL_SETTINGS_STRUCTURAL = (
    "response_mode", "reply_in_channel",
    "participation_level", "model", "reasoning_effort", "verbosity",
    "ambient_memory",
    "enable_web_search", "enable_mcp", "image_model",
)


def _build_channel_settings_write(channel_id, response_mode=_UNSET,
                                  reply_in_channel=_UNSET, participation_level=_UNSET,
                                  snoozed_until=_UNSET, muted_threads=_UNSET,
                                  model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                                  ambient_memory=_UNSET,
                                  enable_web_search=_UNSET, enable_mcp=_UNSET,
                                  image_model=_UNSET, updated_by=None):
    """Build the atomic upsert for a partial channel_settings write.

    Returns ``(sql, params)`` or ``None`` when no field was provided (caller no-ops).
    Pure — no I/O — so both the sync and async setters share one implementation and one behavior.

    Design (fixes the mute-clobber incident):
    - Only explicitly-provided columns are written. Untouched columns are NEVER rewritten, so a
      partial write cannot clobber another field (no read-modify-write of the whole row → no race
      with a concurrent modal save).
    - Inheritance-capable columns that carry a non-NULL table default (response_mode → 'tag_only',
      reply_in_channel → 0) are pinned to NULL on a FRESH insert unless explicitly provided, so a
      partial write never materializes a downgraded default over the live global config. Cleared
      inheritance fields store NULL (never a copied runtime default), so global-config changes keep
      being inherited.
    - updated_ts/updated_by bump ONLY when a structural field changed
      (see ``_CHANNEL_SETTINGS_STRUCTURAL``).
    - reply_in_channel: explicit None → NULL (inherit); True/False → 1/0.
    """
    provided: Dict[str, Any] = {}
    if response_mode is not _UNSET:
        provided["response_mode"] = response_mode
    if reply_in_channel is not _UNSET:
        provided["reply_in_channel"] = (
            None if reply_in_channel is None else (1 if reply_in_channel else 0))
    if participation_level is not _UNSET:
        provided["participation_level"] = participation_level
    if snoozed_until is not _UNSET:
        provided["snoozed_until"] = snoozed_until
    if muted_threads is not _UNSET:
        # Deprecated inert JSON column — nothing reads it anymore (the per-thread mute mechanism
        # was removed). Kept only so an explicit write can still clear it to NULL.
        provided["muted_threads"] = _encode_muted_threads(muted_threads)
    if model is not _UNSET:
        provided["model"] = model
    if reasoning_effort is not _UNSET:
        provided["reasoning_effort"] = reasoning_effort
    if verbosity is not _UNSET:
        provided["verbosity"] = verbosity
    if ambient_memory is not _UNSET:
        # F51 opt-out: explicit None → NULL (inherit config.enable_ambient_memory); True/False → 1/0.
        provided["ambient_memory"] = (
            None if ambient_memory is None else (1 if ambient_memory else 0))
    # The capability tri-states (§6.1): explicit None → NULL (inherit the global default),
    # True/False → 1/0. Normalizing to exactly 1/0 here is what keeps the column's CHECK
    # satisfiable — a caller handing us a stray truthy value stores 1, never that value.
    if enable_web_search is not _UNSET:
        provided["enable_web_search"] = (
            None if enable_web_search is None else (1 if enable_web_search else 0))
    if enable_mcp is not _UNSET:
        provided["enable_mcp"] = (
            None if enable_mcp is None else (1 if enable_mcp else 0))
    if image_model is not _UNSET:
        provided["image_model"] = image_model

    if not provided:
        return None

    structural_provided = [c for c in provided if c in _CHANNEL_SETTINGS_STRUCTURAL]
    changed_structural = bool(structural_provided)
    # On the UPDATE (conflict) branch, "structural change" means a real VALUE change, not merely
    # "a structural field was supplied": writing the SAME value must preserve updated_ts/updated_by
    # so an idempotent structural write (a re-save of unchanged settings, a mute-path no-op) never
    # rewrites who last edited the channel. `IS NOT` is SQLite's null-safe inequality, so an
    # inherit(NULL)→NULL write also reads as unchanged.
    change_cond = " OR ".join(
        f"channel_settings.{c} IS NOT excluded.{c}" for c in structural_provided)

    insert_cols = ["channel_id"]
    params: List[Any] = [channel_id]
    update_assignments: List[str] = []

    # Pin the non-NULL-default inheritance columns to NULL on a fresh insert unless provided,
    # so a partial insert inherits from global config instead of freezing a downgraded default.
    for col in ("response_mode", "reply_in_channel"):
        insert_cols.append(col)
        params.append(provided.get(col))
        if col in provided:
            update_assignments.append(f"{col}=excluded.{col}")

    for col, val in provided.items():
        if col in ("response_mode", "reply_in_channel"):
            continue
        insert_cols.append(col)
        params.append(val)
        update_assignments.append(f"{col}=excluded.{col}")

    # Attribute authorship only on a structural change. On a fresh insert updated_ts still gets
    # its column default (a row must have a created stamp); the "don't bump" rule guards UPDATEs.
    # An anonymous structural write (updated_by=None) still stamps the change time but preserves
    # the prior author rather than erasing it.
    insert_cols.append("updated_by")
    params.append(updated_by if changed_structural else None)
    if changed_structural:
        # Bump the stamp/author ONLY when a provided structural column's value actually differs
        # from the stored row (see change_cond) — a same-value write leaves attribution intact.
        update_assignments.append(
            f"updated_ts=CASE WHEN ({change_cond}) THEN CURRENT_TIMESTAMP ELSE updated_ts END")
        if updated_by is not None:
            update_assignments.append(
                f"updated_by=CASE WHEN ({change_cond}) THEN excluded.updated_by ELSE updated_by END")

    placeholders = ", ".join(["?"] * len(insert_cols))
    cols_sql = ", ".join(insert_cols)
    if update_assignments:
        conflict = f"ON CONFLICT(channel_id) DO UPDATE SET {', '.join(update_assignments)}"
    else:
        # Only non-structural columns AND the row already exists → nothing changes there;
        # but a fresh insert still needs to land, so keep the INSERT and no-op the conflict.
        conflict = "ON CONFLICT(channel_id) DO NOTHING"
    sql = f"INSERT INTO channel_settings ({cols_sql}) VALUES ({placeholders}) {conflict}"
    return sql, params


class DatabaseManager(LoggerMixin):
    """
    Manages SQLite database operations for bot persistence.
    Each platform gets its own database file.
    """
    
    def __init__(self, platform: str = "slack"):
        """
        Initialize database connection for the specified platform.

        Args:
            platform: Platform name (e.g. "slack")
        """
        self.platform = platform

        # Get database directory from config
        from config import BotConfig
        config = BotConfig()
        self.db_dir = config.database_dir

        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(f"{self.db_dir}/backups", exist_ok=True)

        # Connect to platform-specific database
        self.db_path = f"{self.db_dir}/{platform}.db"
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None  # Autocommit mode
        )
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        
        # Initialize schema
        self.init_schema()
        
        logger.info(f"Database initialized for {platform} at {self.db_path}")

        # For async operations, we'll create connections as needed
        self._async_db_semaphore = asyncio.Semaphore(10)  # Limit concurrent async connections
    
    def init_schema(self):
        """Create database tables if they don't exist."""
        
        # Threads table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for threads
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_thread 
            ON threads(channel_id, thread_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_activity 
            ON threads(last_activity)
        """)
        
        # NOTE (Phase S): there is deliberately NO messages table. Slack is the only
        # transcript — context is always rebuilt from conversations.replies. The DB keeps
        # only what Slack doesn't have: config, memory, derived artifacts (images/documents),
        # and thread_summaries (compaction state). See Docs/CHANNEL_TEAMMATE_REDESIGN_PLAN.md §5b.

        # Thread summaries table — rolling compaction store for long threads.
        # summary_text covers everything at or before boundary_ts; refs_json preserves
        # structured references (files/images/links) from the summarized span.
        # preserved_ts_json (F3): Slack ts of messages kept in live context (images/
        # summarized docs — _should_preserve_message) that sit AT/BEHIND boundary_ts.
        # They are neither summarized nor in the fetched tail, so a cold rebuild would
        # drop them; recording their ts lets rebuild fetch + re-admit them across the
        # boundary. Refs/metadata only — never transcript content (CLAUDE.md §5b).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_summaries (
                thread_id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                boundary_ts TEXT NOT NULL,
                refs_json TEXT,
                preserved_ts_json TEXT,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)

        # F51c — late-artifact addenda to the compaction summary. An ambient artifact (a slow
        # link fetch, a deferred vision job) can complete AFTER its source message has already
        # been folded into a thread's compaction summary — or a message with a long-ready
        # artifact can be compacted later. Either way the derived note would vanish: it never
        # lived in thread_state.messages (injection is transient per API call), the summary was
        # written without it, and the compacted message no longer returns in the rebuilt tail.
        # These rows carry that late/folded derivation forward — the rebuild concatenates them
        # onto the summary head. Bounded per thread (_MAX_SUMMARY_ADDENDA_PER_THREAD); UNIQUE
        # per (thread, source, kind, ref) so the completion path and the compaction path can't
        # double-record the same note. Deterministic order (source_ts, id) for cache stability.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_summary_addenda (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                source_ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                ref TEXT NOT NULL,
                note TEXT NOT NULL,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(thread_id, source_ts, kind, ref)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_addenda_thread
            ON thread_summary_addenda(thread_id, source_ts, id)
        """)

        # Track 1 — persistent per-channel "recent channel narrative" summary. A cached,
        # throttled, background-generated sketch of what a channel is about (purpose, who's
        # active, recurring topics/vocabulary, ongoing work), read by BOTH the participation
        # classifier and the main response agent for background "grasp" of the room. Like the
        # rest of the DB it is a DERIVED artifact, never a transcript (Slack is the source of
        # truth). One row per channel, keyed by channel_id ONLY — every read/write is strictly
        # WHERE channel_id = ? (no workspace fallback), preserving the shipped scope-guard
        # boundary. `built_through_ts` is the newest source message folded in (the refresh
        # boundary); `source_message_count` is how many messages were actually fed to the model
        # (NOT a lifetime count). `invalidated_at` is set when an in-window edit/delete makes the
        # cache untrustworthy — both agents stop injecting until a background rebuild clears it.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_summaries (
                channel_id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                built_through_ts TEXT NOT NULL,
                source_message_count INTEGER NOT NULL,
                generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                invalidated_at TIMESTAMP
            )
        """)

        # Track 4 — channel join intro idempotency. One row per channel records the lifecycle of
        # the one-time "I've been added here" intro so it is posted EXACTLY once, surviving Slack
        # event refires AND a crash between posting and recording intro_ts. `status` is the durable
        # lease: 'pending' (an attempt owns it — a refire must skip), 'posted' (done — never repost),
        # 'failed' (an attempt died — a genuine refire may retry, guarded by a history reconcile so a
        # post-then-crash can't double-post). `event_id` is the member_joined_channel event_ts that
        # started it (debugging only); `intro_ts` is the posted message's ts. Keyed by channel_id
        # ONLY — no transcript content lives here (Slack is the source of truth).
        # `owner_token` is minted per acquiring attempt: the failure handler downgrades a 'pending'
        # row to 'failed' ONLY when its own token matches, so a task that failed before/without
        # winning the lease can never steal a CONCURRENT attempt's live lease.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_introductions (
                channel_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                prepared_text TEXT,
                event_id TEXT,
                intro_ts TEXT,
                owner_token TEXT,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # V3 channel-teammate: a first-time user's @mention in a CHANNEL is answered with channel +
        # default settings (no DM-first onboarding gate), and we silently DM them the settings button
        # exactly once so they can tune personal prefs if they want. This table is that "once" guard.
        # It MUST be durable: in-memory guards die on restart and this bot rebuilds from scratch, so a
        # session-only set would re-DM the same newcomer after every deploy — the very noise we're
        # removing. One row per user; presence == already nudged. No transcript content lives here.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_onboarding_nudges (
                slack_user_id TEXT PRIMARY KEY,
                nudged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # F32: thread-scoped code-interpreter containers. One OpenAI container per thread, so
        # the model's sandbox state (files in /mnt/data, loaded dataframes) survives the turn
        # boundary within a conversation.
        #
        # `published_files_json` is NOT bookkeeping fluff — it is a correctness guard. A reused
        # container's listing still contains every file from earlier turns, so without a durable
        # record of what we already uploaded, a bot restart mid-conversation would re-post turn
        # 1's chart on turn 2 (the in-memory dedupe dies with the process). It lives here, next
        # to the container id, because it is meaningless once the container is gone.
        #
        # No FK: PRAGMA foreign_keys is never enabled, and a container can outlive its threads
        # row. Rows are swept by age in the daily cleanup instead.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_containers (
                thread_id TEXT PRIMARY KEY,
                container_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_files_json TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_containers_last_used
            ON thread_containers(last_used_at)
        """)

        # Images table (no base64 storage)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                message_ts TEXT,  -- Links image to specific message
                image_type TEXT,
                prompt TEXT,
                analysis TEXT,
                original_analysis TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for images
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_images 
            ON images(thread_id, created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_url 
            ON images(url)
        """)
        
        # Documents table — summary + metadata + Slack ref ONLY (user hard rule,
        # CLAUDE.md pitfall 6a): full content is never at rest. The file lives on
        # Slack's CDN (file_id/url_private) and is re-derived in memory on demand
        # via the read_document tool.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                summary TEXT,
                file_id TEXT,
                url_private TEXT,
                size_bytes INTEGER,
                page_structure TEXT,
                total_pages INTEGER,
                metadata_json TEXT,
                message_ts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for documents
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_documents 
            ON documents(thread_id, created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_filename 
            ON documents(filename)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_message
            ON documents(message_ts)
        """)

        # F51 — Ambient artifacts. Derived summaries for images/links/files posted ambiently
        # (or addressed) in a channel/thread, kept in the running context even when the bot
        # doesn't respond. CHANNEL + source-ts keyed (NOT the colon thread key that locked out
        # the incident lookup). conversation_ts is the thread root (= source_ts for a top-level
        # message, NOT nullable) so thread retrieval + compaction stay deterministic without
        # ever splitting a colon-composed key. summary/model are NULL for pending/failed rows.
        # Slack stays the only transcript — this holds ONLY derivations + refs, never message
        # text mirrors, never image bytes (CLAUDE.md pitfall 4). Reuse is SAME-CHANNEL only.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ambient_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                source_ts TEXT NOT NULL,
                conversation_ts TEXT NOT NULL,
                kind TEXT NOT NULL,               -- 'image' | 'link' | 'file'
                ref TEXT NOT NULL,                -- Slack file id, or normalized URL
                title TEXT,
                summary TEXT,                     -- NULL until ready
                model TEXT,
                status TEXT NOT NULL DEFAULT 'pending',  -- pending|ready|failed|blocked|omitted
                derivation_source TEXT,           -- gate_vision|vision_worker|fetch|unfurl|document
                content_type TEXT,
                error_code TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fetched_at TIMESTAMP,
                expires_at TIMESTAMP,
                UNIQUE(channel_id, source_ts, kind, ref)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_source
            ON ambient_artifacts(channel_id, source_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_conversation
            ON ambient_artifacts(channel_id, conversation_ts, source_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_ref
            ON ambient_artifacts(channel_id, kind, ref, status, fetched_at)
        """)

        # Tool-use provenance (F7): compact per-reply record of the tools the bot invoked
        # (names + arg-derived gists only, NO results/content), keyed by the reply's Slack
        # ts. Reinjected as a "[used tools: …]" annotation on rebuild so the model can
        # recall its own past tool use. Deliberately NO foreign key: the ON DELETE CASCADE
        # path is dead (PRAGMA foreign_keys is never enabled) — rows are swept by age via
        # delete_old_tool_usage() instead. UNIQUE(channel, ts) makes re-persist idempotent.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS message_tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                message_ts TEXT NOT NULL,
                thread_key TEXT NOT NULL,
                tools_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(channel_id, message_ts)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_usage_thread
            ON message_tool_usage(thread_key)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_usage_created
            ON message_tool_usage(created_at)
        """)

        # Users table with timezone support
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                real_name TEXT,
                email TEXT,
                config_json TEXT,
                timezone TEXT DEFAULT 'UTC',
                tz_label TEXT,
                tz_offset INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User preferences table for settings modal
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                slack_user_id TEXT PRIMARY KEY,
                slack_email TEXT,

                -- Model settings
                model TEXT DEFAULT 'gpt-5.6-sol',
                reasoning_effort TEXT DEFAULT 'medium',
                verbosity TEXT DEFAULT 'low',
                temperature REAL DEFAULT 0.8,
                top_p REAL DEFAULT 1.0,

                -- Feature toggles
                enable_web_search BOOLEAN DEFAULT 1,
                enable_mcp BOOLEAN DEFAULT 1,
                enable_streaming BOOLEAN DEFAULT 1,

                -- Image settings
                image_model TEXT DEFAULT 'gpt-image-2',
                image_size TEXT DEFAULT '1024x1024',
                image_quality TEXT DEFAULT 'auto',
                image_background TEXT DEFAULT 'auto',
                input_fidelity TEXT DEFAULT 'high',
                vision_detail TEXT DEFAULT 'auto',

                -- Metadata
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                settings_completed BOOLEAN DEFAULT 0,

                FOREIGN KEY (slack_user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create index for email lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_prefs_email
            ON user_preferences(slack_email)
        """)

        # Modal sessions table for temporary modal state storage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS modal_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                modal_type TEXT DEFAULT 'settings',
                state TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create indexes for modal sessions
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modal_session_user
            ON modal_sessions(user_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modal_session_created
            ON modal_sessions(created_at)
        """)

        # MCP tools cache table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mcp_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_label TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                description TEXT,
                input_schema TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(server_label, tool_name)
            )
        """)

        # Create index for mcp_tools
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mcp_server_label
            ON mcp_tools(server_label)
        """)

        # Phase 7: per-channel response settings.
        # No row for a channel => global defaults apply (no behavior change).
        # `directives` is LEGACY and inert: the channel's standing rules moved to the reserved
        # policy row. The column stays so migrate_channel_directives_to_policy_async has
        # something to read on a database that predates the move; nothing else touches it.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_settings (
                channel_id TEXT PRIMARY KEY,
                response_mode TEXT DEFAULT 'tag_only',
                directives TEXT,
                reply_in_channel BOOLEAN DEFAULT 0,
                participation_level TEXT,
                snoozed_until TEXT,
                muted_threads TEXT,
                model TEXT,
                reasoning_effort TEXT,
                verbosity TEXT,
                ambient_memory INTEGER,
                enable_web_search INTEGER
                    CHECK (enable_web_search IS NULL OR enable_web_search IN (0, 1)),
                enable_mcp INTEGER
                    CHECK (enable_mcp IS NULL OR enable_mcp IN (0, 1)),
                image_model TEXT,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """)

        # Per-channel durable memory (Phase 9). scope='channel' rows are private to that channel;
        # scope='workspace' rows are shared (read-mostly, admin/manual writes only).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'channel',
                content TEXT NOT NULL,
                author TEXT,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_memory_lookup ON channel_memory (scope, channel_id)"
        )

        # NOTE: there was an `emoji_usage` table here — a workspace-wide reaction tally that
        # ranked the custom-emoji shortlist injected into the old rich participation gate. The
        # gate is now one bit and reads no shortlist, so nothing writes or reads it. We stop
        # CREATING it and deliberately do NOT drop it: an existing installation keeps an
        # orphaned, content-free table (name + count, no channel/ts/author), and a DROP would
        # be a destructive migration bought for a few kilobytes.

        # Response feedback (Phase H): thumbs signal from native feedback buttons and
        # from +1/-1 reactions on the bot's own messages. One row per
        # (message, user, source); a changed thumb updates the row in place.
        # The participation engine may read per-channel ratios later.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS response_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                thread_ts TEXT,
                message_ts TEXT NOT NULL,
                user_id TEXT NOT NULL,
                signal INTEGER NOT NULL CHECK (signal IN (-1, 1)),
                source TEXT NOT NULL CHECK (source IN ('button', 'reaction')),
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (message_ts, user_id, source)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_response_feedback_channel "
            "ON response_feedback (channel_id, created_ts)"
        )

        # Single-stream P1 (Docs/SINGLE_STREAM_SPEC.md) ---------------------------------------

        # Process-independent key/value facts the bot needs across restarts. The receipts
        # feature epoch lives here: own-messages older than it are grandfathered into the
        # stream, so the value must be written exactly ONCE ever (set_meta_if_absent) — a
        # rewrite would silently re-grandfather every message posted since.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Spec §5. One row per durable message WE posted. An own-message enters the channel
        # stream ONLY with a `finalized` receipt; `chrome` is permanent exclusion (placeholders,
        # status cards, footers) and `in_flight` means the producing turn is still editing.
        # turn_id encodes the owning session ("{session}:{seq}"), which is what makes boot
        # reconciliation possible: a dead session's in_flight rows are final by definition.
        # thread_root_ts is the destination root for thread posts — evidence for pre-boundary
        # root discovery, NEVER a wake source.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS outbound_receipts (
                team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                message_ts TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('in_flight', 'finalized', 'chrome')),
                thread_root_ts TEXT,
                receipt_class TEXT,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finalized_ts TIMESTAMP,
                PRIMARY KEY (team_id, channel_id, message_ts)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbound_receipts_state "
            "ON outbound_receipts (state)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbound_receipts_channel_state "
            "ON outbound_receipts (team_id, channel_id, state)"
        )

        # An uploaded file's SHARE ts is not known when files_upload_v2 returns — only the
        # file_id is. This row is written in that gap so a crash before resolution leaves a
        # retryable record instead of a message that can never earn a receipt. Rows are deleted
        # only by successful resolution (atomically, with the finalize) or by a Slack-confirmed
        # deletion; every resolution FAILURE keeps the row for boot retry.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_share_receipts (
                team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                owner_turn_id TEXT NOT NULL,
                thread_root_ts TEXT,
                receipt_class TEXT,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, channel_id, file_id)
            )
        """)

        # Spec §4 live index: which roots have replies we would otherwise never see. A
        # pre-boundary root with a post-boundary reply cannot be surfaced by
        # conversations.history (the parent keeps its original ts), so this index is the only
        # route to it. Both ts columns are monotonic hints — advisory_reply_count may cause an
        # extra fetch but must never suppress one. `dirty` is sticky: an edit/delete under a
        # root forces a refetch until the reader clears it against the exact event ts it saw.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_thread_activity (
                team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                root_ts TEXT NOT NULL,
                last_observed_reply_ts TEXT,
                advisory_reply_count INTEGER,
                last_index_event_ts TEXT,
                dirty INTEGER NOT NULL DEFAULT 0,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, channel_id, root_ts)
            )
        """)

        # How far back this channel's ROOT INVENTORY reaches — which thread roots exist and when
        # each last saw a reply. It never gates the stream and never renders content.
        # There are deliberately NO persisted pagination cursors — inventory_start_ts IS the
        # resume point (a restart re-pages from latest=inventory_start_ts, inclusive=false), so
        # it may only ever move BACKWARD and only after a page is fully processed. sweep_token
        # is the single-worker claim; a holder proves liveness with heartbeat_ts.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_coverage (
                team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                inventory_start_ts TEXT NOT NULL,
                bootstrap_status TEXT NOT NULL
                    CHECK (bootstrap_status IN ('pending', 'running', 'complete', 'limited')),
                inventory_reason TEXT,
                sweep_token TEXT,
                heartbeat_ts TIMESTAMP,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, channel_id)
            )
        """)

        # The persisted periphery floor. NOT a coverage claim and NOT a horizon: it is the policy
        # decision "this is where the recent window starts", and it only ever moves FORWARD.
        # selection_version invalidates it when the selection POLICY changes, independently of
        # the serializer grammar — a policy change must be able to move a floor without
        # necessarily changing a rendered byte.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_window_anchor (
                team_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                floor_ts TEXT NOT NULL,
                selection_version INTEGER NOT NULL,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, channel_id)
            )
        """)

        self.conn.commit()

        # Run migrations for existing databases
        self._run_migrations()

    @contextmanager
    def _migration_step(self, name: str):
        """Run one migration phase in isolation.

        A raising phase is logged LOUDLY (named, with a traceback) and the
        remaining phases still run. Previously the whole migration body sat under
        a single try/except, so one bad step silently skipped every later step and
        the bot then served traffic on a half-migrated schema.
        """
        try:
            yield
        except Exception as e:
            self.log_error(f"DB: Migration step '{name}' FAILED: {e}", exc_info=True)

    def _is_pre_v3_database(self) -> bool:
        """True when this database still has the pre-v3 (v2.x) shape.

        Cheap, read-only, and deliberately conservative — it must NOT fire on a
        brand-new database (init_schema's CREATE TABLE IF NOT EXISTS block runs
        immediately before the migrations, so a fresh DB already has `documents`
        and `user_preferences`) nor on an already-migrated one (second boot).

        Legacy signals, any one of which is decisive:
        - the `messages` mirror table exists (dropped by the v3 mirror-drop)
        - `documents.content` exists (dropped by the v3 doc-content-drop)
        - `user_preferences` is missing the `gpt56_migrated` sentinel AND already
          holds rows. The sentinel is added by the migration, not by CREATE TABLE,
          so a fresh DB also lacks it — the row check is what distinguishes real
          user preferences (about to be bulk-overwritten) from an empty new table.
        """
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='messages'
        """)
        if cursor.fetchone():
            return True

        cursor = self.conn.execute("PRAGMA table_info(documents)")
        if any(col[1] == 'content' for col in cursor.fetchall()):
            return True

        cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
        up_columns = [col[1] for col in cursor.fetchall()]
        if up_columns and 'gpt56_migrated' not in up_columns:
            cursor = self.conn.execute("SELECT 1 FROM user_preferences LIMIT 1")
            if cursor.fetchone():
                return True

        return False

    def _run_migrations(self):
        """Run database migrations to update schema for existing databases.

        Each phase is isolated by `_migration_step` so a failure is loud and
        contained instead of silently skipping every later phase.
        """
        # BEFORE every other step, and required-critical: each step below and the running bot
        # assume the P4a schema is gone. Deliberately ahead of the pre-v3 backup — a pre-v3
        # database predates P4a entirely, so there is nothing there for this to drop.
        self._migrate_drop_compaction_schema()

        # Rollback path FIRST: snapshot the database before any migration writes to
        # it. The gpt-5.6 swap below bulk-overwrites every user's model/effort, and
        # the two destructive drops each take their own tagged backup only AFTER
        # that swap has already run — so without this, no backup can restore what
        # users actually picked. Runs at most once per database (see detection).
        with self._migration_step("pre-v3 backup"):
            if self._is_pre_v3_database():
                self.log_info(
                    "DB: Pre-v3 database detected — backup tagged pre-v3-upgrade before migrating"
                )
                self.backup_database(tag="pre-v3-upgrade")

        with self._migration_step("channel_settings columns"):
            # Phase F: participation_level + snoozed_until on channel_settings
            cursor = self.conn.execute("PRAGMA table_info(channel_settings)")
            cs_columns = [col[1] for col in cursor.fetchall()]
            if cs_columns and 'participation_level' not in cs_columns:
                self.log_info("DB: Adding participation_level column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN participation_level TEXT")
                self.conn.commit()
            if cs_columns and 'snoozed_until' not in cs_columns:
                self.log_info("DB: Adding snoozed_until column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN snoozed_until TEXT")
                self.conn.commit()
            # F15: muted_threads (JSON list) — threads permanently opted out of unprompted
            # participation via a "butt out" backoff. Replaces the snoozed_until timer rail.
            if cs_columns and 'muted_threads' not in cs_columns:
                self.log_info("DB: Adding muted_threads column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN muted_threads TEXT")
                self.conn.commit()
            # Shared per-channel model/effort/verbosity overrides (NULL = inherit)
            for col in ("model", "reasoning_effort", "verbosity"):
                if cs_columns and col not in cs_columns:
                    self.log_info(f"DB: Adding {col} column to channel_settings")
                    self.conn.execute(f"ALTER TABLE channel_settings ADD COLUMN {col} TEXT")
                    self.conn.commit()
            # F51: per-channel ambient-memory opt-out (NULL = inherit ENABLE_AMBIENT_MEMORY;
            # 0 = memory off for this channel, distinct from participation `off`).
            if cs_columns and 'ambient_memory' not in cs_columns:
                self.log_info("DB: Adding ambient_memory column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN ambient_memory INTEGER")
                self.conn.commit()

        with self._migration_step("images.message_ts"):
            # Check if message_ts column exists in images table
            cursor = self.conn.execute("PRAGMA table_info(images)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'message_ts' not in columns:
                self.log_info("DB: Adding message_ts column to images table")
                self.conn.execute("""
                    ALTER TABLE images
                    ADD COLUMN message_ts TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added message_ts column")

        with self._migration_step("thread_summaries.preserved_ts_json"):
            # F3: preserved-message ts refs behind the compaction boundary (see CREATE TABLE).
            cursor = self.conn.execute("PRAGMA table_info(thread_summaries)")
            ts_columns = [col[1] for col in cursor.fetchall()]
            if ts_columns and 'preserved_ts_json' not in ts_columns:
                self.log_info("DB: Adding preserved_ts_json column to thread_summaries")
                self.conn.execute("ALTER TABLE thread_summaries ADD COLUMN preserved_ts_json TEXT")
                self.conn.commit()
                self.log_info("DB: Successfully added preserved_ts_json column")

        with self._migration_step("users.real_name"):
            # Check if real_name column exists in users table
            cursor = self.conn.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'real_name' not in columns:
                self.log_info("DB: Adding real_name column to users table")
                self.conn.execute("""
                    ALTER TABLE users
                    ADD COLUMN real_name TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added real_name column")

        with self._migration_step("user_preferences.custom_instructions"):
            # Check if custom_instructions column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'custom_instructions' not in columns:
                self.log_info("DB: Adding custom_instructions column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN custom_instructions TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added custom_instructions column")

        with self._migration_step("users.email"):
            # Check if email column exists in users table
            cursor = self.conn.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'email' not in columns:
                self.log_info("DB: Adding email column to users table")
                self.conn.execute("""
                    ALTER TABLE users
                    ADD COLUMN email TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added email column")

        with self._migration_step("user_preferences table"):
            # Check if user_preferences table exists
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='user_preferences'
            """)
            if not cursor.fetchone():
                self.log_info("DB: Creating user_preferences table")
                self.conn.execute("""
                    CREATE TABLE user_preferences (
                        slack_user_id TEXT PRIMARY KEY,
                        slack_email TEXT,

                        -- Model settings
                        model TEXT DEFAULT 'gpt-5.6-sol',
                        reasoning_effort TEXT DEFAULT 'medium',
                        verbosity TEXT DEFAULT 'low',
                        temperature REAL DEFAULT 0.8,
                        top_p REAL DEFAULT 1.0,

                        -- Feature toggles
                        enable_web_search BOOLEAN DEFAULT 1,
                        enable_streaming BOOLEAN DEFAULT 1,

                        -- Image settings
                        image_model TEXT DEFAULT 'gpt-image-2',
                        image_size TEXT DEFAULT '1024x1024',
                        image_quality TEXT DEFAULT 'auto',
                        image_background TEXT DEFAULT 'auto',
                        input_fidelity TEXT DEFAULT 'high',
                        vision_detail TEXT DEFAULT 'auto',

                        -- Metadata
                        created_at INTEGER DEFAULT (strftime('%s', 'now')),
                        updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                        settings_completed BOOLEAN DEFAULT 0,

                        FOREIGN KEY (slack_user_id) REFERENCES users(user_id)
                    )
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_prefs_email
                    ON user_preferences(slack_email)
                """)
                self.conn.commit()
                self.log_info("DB: Successfully created user_preferences table")

        with self._migration_step("user_preferences.image_quality"):
            # Check if image_quality column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_quality' not in columns:
                self.log_info("DB: Adding image_quality column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_quality TEXT DEFAULT 'auto'
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added image_quality column")

        with self._migration_step("user_preferences.image_background"):
            # Check if image_background column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_background' not in columns:
                self.log_info("DB: Adding image_background column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_background TEXT DEFAULT 'auto'
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added image_background column")

        with self._migration_step("user_preferences.enable_mcp"):
            # Check if enable_mcp column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'enable_mcp' not in columns:
                self.log_info("DB: Adding enable_mcp column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN enable_mcp BOOLEAN DEFAULT 1
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added enable_mcp column")

        with self._migration_step("user_preferences.image_model"):
            # Check if image_model column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_model' not in columns:
                self.log_info("DB: Adding image_model column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_model TEXT DEFAULT 'gpt-image-2'
                """)
                # Explicitly set all existing rows to gpt-image-2. The DEFAULT clause
                # above already does this on SQLite, but make the one-time bulk swap
                # explicit so the intent is unambiguous and the row count gets logged.
                # This runs exactly once (the surrounding `if` block guarantees it).
                cursor = self.conn.execute(
                    "UPDATE user_preferences SET image_model = 'gpt-image-2'"
                )
                row_count = cursor.rowcount
                self.conn.commit()
                self.log_info(
                    f"DB: Successfully added image_model column and migrated "
                    f"{row_count} existing user(s) to gpt-image-2"
                )

        with self._migration_step("gpt-5.5 swap"):
            # One-time bulk swap: migrate every user still on a pre-5.5 model to gpt-5.5.
            # Gated by a sentinel migration marker so it ran exactly once back when older
            # models were still selectable. (Superseded by the normalizer below, kept so
            # the column exists on databases created between the two migrations.)
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'gpt55_migrated' not in columns:
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN gpt55_migrated INTEGER DEFAULT 0
                """)
                cursor = self.conn.execute("""
                    UPDATE user_preferences
                    SET model = 'gpt-5.5', gpt55_migrated = 1
                    WHERE gpt55_migrated = 0
                """)
                swapped = cursor.rowcount
                self.conn.commit()
                self.log_info(
                    f"DB: One-time migration — swapped {swapped} user(s) to gpt-5.5"
                )

        self._migrate_gpt56()

        with self._migration_step("settings_completed backfill"):
            # One-time backfill: mark long-standing users as settings_completed.
            # Earlier versions of the bot only flipped settings_completed=True when
            # the user saved with "global" scope. Users who only ever saved thread-scope
            # configs kept getting the "Please configure your settings" warning on every
            # DM. Backfill anyone whose row was created more than 24h ago — if they've
            # been around that long, they know the bot exists and don't need the gate.
            cursor = self.conn.execute("""
                UPDATE user_preferences
                SET settings_completed = 1
                WHERE settings_completed = 0
                  AND created_at IS NOT NULL
                  AND created_at < (strftime('%s', 'now') - 86400)
            """)
            backfilled = cursor.rowcount
            if backfilled:
                self.conn.commit()
                self.log_info(
                    f"DB: Backfilled settings_completed=1 for {backfilled} pre-existing user(s)"
                )

        with self._migration_step("mirror drop"):
            # Phase S one-time cleanup: drop the message mirror. Slack is the only
            # transcript now — context is always rebuilt from conversations.replies.
            # Guarded on table existence so it runs exactly once per database; the
            # tagged backup is the rollback path.
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='messages'
            """)
            if cursor.fetchone():
                self.backup_database(tag="pre-v3-mirror-drop")
                size_before = os.path.getsize(self.db_path)
                cursor = self.conn.execute("SELECT COUNT(*) FROM messages")
                row_count = cursor.fetchone()[0]
                self.conn.execute("DROP TABLE IF EXISTS messages")
                # Drop the never-read LEGACY documents.summary column (dead since
                # day one). Only on the legacy table shape (content column present)
                # — the D2 schema has a NEW, load-bearing summary column that the
                # D2 migration below (re)creates and populates.
                # ALTER ... DROP COLUMN needs SQLite 3.35+; degrade gracefully below it.
                try:
                    cursor = self.conn.execute("PRAGMA table_info(documents)")
                    doc_columns = [col[1] for col in cursor.fetchall()]
                    if 'summary' in doc_columns and 'content' in doc_columns:
                        self.conn.execute("ALTER TABLE documents DROP COLUMN summary")
                except Exception as col_err:
                    self.log_warning(f"DB: Could not drop documents.summary column: {col_err}")
                self.conn.execute("VACUUM")
                size_after = os.path.getsize(self.db_path)
                self.log_info(
                    f"DB: Mirror-drop migration complete — removed {row_count} cached message "
                    f"row(s), reclaimed {max(0, size_before - size_after):,} bytes "
                    f"(backup tagged pre-v3-mirror-drop in {self.db_dir}/backups)"
                )

        with self._migration_step("doc-content drop"):
            # Doc-architecture (D2) one-time cleanup: drop documents.content.
            # Same hard rule as the mirror drop — no file/document content at rest;
            # rows keep summary + metadata + the Slack CDN ref. Guarded on the
            # content column existing so it runs exactly once per database.
            cursor = self.conn.execute("PRAGMA table_info(documents)")
            doc_columns = [col[1] for col in cursor.fetchall()]
            if 'content' in doc_columns:
                self.backup_database(tag="pre-v3-doc-content-drop")
                size_before = os.path.getsize(self.db_path)
                # Ensure the new columns exist before synthesizing summaries
                for col_name, col_type in (("summary", "TEXT"), ("file_id", "TEXT"),
                                           ("url_private", "TEXT"), ("size_bytes", "INTEGER")):
                    if col_name not in doc_columns:
                        self.conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
                # Mechanical summary synthesis for legacy rows: labeled excerpt of
                # the stored content (cheap, safe; rows are ≤30 days old).
                cursor = self.conn.execute("""
                    UPDATE documents
                    SET summary = '[excerpt of original — full document available via read_document]' || char(10)
                                  || substr(content, 1, 1500)
                    WHERE (summary IS NULL OR summary = '') AND content IS NOT NULL
                """)
                synthesized = cursor.rowcount
                try:
                    self.conn.execute("ALTER TABLE documents DROP COLUMN content")
                    self.conn.execute("VACUUM")
                    size_after = os.path.getsize(self.db_path)
                    self.log_info(
                        f"DB: Doc-content-drop migration complete — synthesized {synthesized} "
                        f"summary(ies), reclaimed {max(0, size_before - size_after):,} bytes "
                        f"(backup tagged pre-v3-doc-content-drop in {self.db_dir}/backups)"
                    )
                except Exception as col_err:
                    # SQLite < 3.35 can't DROP COLUMN; content stays but is never
                    # read or written again. Log loudly — this violates the
                    # no-content-at-rest rule until SQLite is upgraded.
                    self.log_warning(f"DB: Could not drop documents.content column: {col_err}")
                self.conn.commit()

        with self._migration_step("mcp_tools table"):
            # Check if mcp_tools table exists
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='mcp_tools'
            """)
            if not cursor.fetchone():
                self.log_info("DB: Creating mcp_tools table")
                self.conn.execute("""
                    CREATE TABLE mcp_tools (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server_label TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        description TEXT,
                        input_schema TEXT,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(server_label, tool_name)
                    )
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mcp_server_label
                    ON mcp_tools(server_label)
                """)
                self.conn.commit()
                self.log_info("DB: Successfully created mcp_tools table")

        # Deliberately OUTSIDE _migration_step: this one must fail startup. Every later channel
        # document write depends on the index existing, and a bot running without it silently
        # accumulates one duplicate row per turn per file — which is exactly the failure the
        # index exists to prevent, made invisible.
        self._migrate_channel_document_uniqueness()

        # Also OUTSIDE _migration_step: the sweep's own accessors read the renamed columns, so
        # a database left on the old names would fail every inventory read after this boot.
        self._migrate_coverage_to_inventory()

        # Also OUTSIDE _migration_step (respec §6.1, REQUIRED-CRITICAL): every channel turn reads
        # these three columns, and a bot serving traffic without them would run every channel on
        # the global defaults while the modal cheerfully reported the channel's own settings.
        self._migrate_channel_capability_columns()

    # ---------------------------------------------------- channel capability columns (respec §6.1)

    _CHANNEL_CAPABILITY_DDL = (
        ("enable_web_search",
         "INTEGER CHECK (enable_web_search IS NULL OR enable_web_search IN (0, 1))"),
        ("enable_mcp",
         "INTEGER CHECK (enable_mcp IS NULL OR enable_mcp IN (0, 1))"),
        ("image_model", "TEXT"),
    )

    # The canonical `channel_settings` shape, in `init_schema` order. The rebuild below recreates
    # the table from this, so it must stay in step with the CREATE TABLE — which is why the
    # rebuild refuses to run at all if the live table holds a column this list does not name.
    _CHANNEL_SETTINGS_COLUMNS = (
        "channel_id", "response_mode", "directives", "reply_in_channel", "participation_level",
        "snoozed_until", "muted_threads", "model", "reasoning_effort", "verbosity",
        "ambient_memory", "enable_web_search", "enable_mcp", "image_model",
        "updated_ts", "updated_by",
    )

    def _migrate_channel_capability_columns(self):
        """Give `channel_settings` the three capability columns, WITH their constraints.
        FAILS STARTUP.

        `_migration_step` logs and continues, which is the wrong contract here: the capability
        profile is what a channel turn RUNS on, so a half-migrated schema does not announce
        itself — it quietly serves every channel the workspace defaults while the settings modal
        shows, and accepts, per-channel values that go nowhere.

        THREE steps, because "the column exists" and "the column is constrained" are different
        claims and only the first is fixable with ALTER:

        1. ADD COLUMN for anything missing; the CHECK rides along, which SQLite permits there.
        2. If a boolean column exists WITHOUT its CHECK — a database built by an intermediate
           revision, where step 1 sees the name and skips it — REBUILD the table. SQLite cannot
           attach a constraint to an existing column, and without this §6.1's claim that the
           storage layer makes `2` unreachable would simply be false for those databases.
        3. VERIFY, and raise otherwise. This runs regardless of how the earlier steps went, so a
           later edit that wraps them in a try/except still cannot get past it.

        The read-side validation in `BotConfig._channel_capability_profile` stays either way: it
        is what covers rows written BEFORE the constraint existed, which SQLite never revalidates.
        """
        cursor = self.conn.execute("PRAGMA table_info(channel_settings)")
        live = [col[1] for col in cursor.fetchall()]
        existing = set(live)

        # PREFLIGHT, before the first ALTER. A column that ALREADY exists unconstrained is one this
        # boot will have to rebuild for, and the rebuild refuses a table carrying a column the
        # canonical list cannot name. Deciding that here is what makes the refusal leave the schema
        # byte-intact: ADD COLUMN is autocommitted on this connection, so a refusal raised after
        # one had run would already have rewritten `sqlite_master`, and the operator would be
        # looking at a table nobody finished migrating. Columns added BELOW carry their CHECK with
        # them and never trigger a rebuild, so this is the whole of the condition.
        if any(column in existing for column in self._channel_settings_missing_checks()):
            self._refuse_unknown_columns(live)

        added = []
        for column, ddl in self._CHANNEL_CAPABILITY_DDL:
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE channel_settings ADD COLUMN {column} {ddl}")
                added.append(column)
        if added:
            self.conn.commit()
            self.log_info(f"DB: added channel_settings capability columns: {', '.join(added)}")

        if self._channel_settings_missing_checks():
            self._rebuild_channel_settings_with_checks()

        cursor = self.conn.execute("PRAGMA table_info(channel_settings)")
        present = {col[1] for col in cursor.fetchall()}
        missing = [column for column, _ in self._CHANNEL_CAPABILITY_DDL if column not in present]
        if missing:
            raise RuntimeError(
                f"channel_settings is missing the capability column(s) {', '.join(missing)}; "
                f"refusing to start rather than serve a partial channel capability profile")

    def _refuse_unknown_columns(self, live: List[str]) -> None:
        """Fail the boot when `channel_settings` carries a column the canonical list cannot name.

        Called TWICE and deliberately so: once as a preflight before any ALTER runs, and once
        inside the rebuild itself. The preflight is what keeps the refusal byte-intact on a
        database that also needs columns added; the second call is the rail that cannot be walked
        past, so no future caller can reach the copy without having been asked this question.
        Both are cheap and the answer cannot change between them.
        """
        unknown = [column for column in live if column not in self._CHANNEL_SETTINGS_COLUMNS]
        if not unknown:
            return
        message = (
            f"channel_settings carries unrecognised column(s) {', '.join(unknown)}, so it cannot "
            f"be rebuilt without dropping them; refusing to start rather than drop data or serve "
            f"an unconstrained capability profile. The table is untouched.")
        self.log_error(f"DB: {message}")
        raise RuntimeError(message)

    def _channel_settings_missing_checks(self) -> List[str]:
        """The boolean capability columns whose CHECK constraint is absent from the live DDL.

        Read off `sqlite_master.sql`, which is the statement the table was actually created with —
        there is no PRAGMA that reports per-column CHECKs.

        PER CLAUSE, not per DDL. An earlier version searched all the text following the first
        `CHECK`, which reads as constrained any column merely DECLARED after a constrained one:
        a table where `enable_web_search` had its check and `enable_mcp` did not answered "both
        fine" and booted unrepaired, because the string "enable_mcp" appeared downstream in its
        own plain declaration. Each column now has to be named inside a CHECK clause of its own.
        """
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='channel_settings'"
        ).fetchone()
        clauses = _check_clauses((row[0] if row else "") or "")
        return [column for column, spec in self._CHANNEL_CAPABILITY_DDL
                if "CHECK" in spec and not any(column in clause for clause in clauses)]

    def _rebuild_channel_settings_with_checks(self):
        """Recreate `channel_settings` so its CHECK constraints exist. FAILS STARTUP.

        The standard SQLite table rebuild, and cheap here: `channel_settings` is a small config
        table with no foreign keys, no triggers, and no index beyond the primary key's own.

        Columns are copied BY NAME from the canonical list, so a live table carrying a column that
        list does not know has no safe outcome available here and the boot FAILS. Three options
        and only one of them is honest: copying the known columns drops the unrecognised one and
        its data; leaving the table alone and continuing returns a bot serving traffic on an
        unconstrained table, which is precisely what REQUIRED-CRITICAL and §6.1's storage
        guarantee forbid; refusing to start leaves the table and every row byte-intact for an
        operator to look at. Nothing is dropped and nothing is half-migrated — same fail-closed
        rule the fence's enabled path follows.

        Any stored boolean outside {0, 1} is normalized to NULL on the way across, because the new
        table's CHECK would otherwise reject the copy and fail the boot: those values are already
        read as "inherit" by the resolver, so this writes down what the bot was doing anyway.
        """
        cursor = self.conn.execute("PRAGMA table_info(channel_settings)")
        live = [col[1] for col in cursor.fetchall()]
        self._refuse_unknown_columns(live)

        carried = [column for column in self._CHANNEL_SETTINGS_COLUMNS if column in live]
        bool_columns = [column for column, spec in self._CHANNEL_CAPABILITY_DDL
                        if "CHECK" in spec and column in carried]
        # A boolean outside {0, 1} becomes NULL; everything else copies verbatim.
        select_list = ", ".join(
            (f"CASE WHEN {c} IN (0, 1) THEN {c} ELSE NULL END" if c in bool_columns else c)
            for c in carried)

        rogue = 0
        for column in bool_columns:
            rogue += self.conn.execute(
                f"SELECT COUNT(*) FROM channel_settings "
                f"WHERE {column} IS NOT NULL AND {column} NOT IN (0, 1)").fetchone()[0]

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("""
                CREATE TABLE channel_settings_new (
                    channel_id TEXT PRIMARY KEY,
                    response_mode TEXT DEFAULT 'tag_only',
                    directives TEXT,
                    reply_in_channel BOOLEAN DEFAULT 0,
                    participation_level TEXT,
                    snoozed_until TEXT,
                    muted_threads TEXT,
                    model TEXT,
                    reasoning_effort TEXT,
                    verbosity TEXT,
                    ambient_memory INTEGER,
                    enable_web_search INTEGER
                        CHECK (enable_web_search IS NULL OR enable_web_search IN (0, 1)),
                    enable_mcp INTEGER
                        CHECK (enable_mcp IS NULL OR enable_mcp IN (0, 1)),
                    image_model TEXT,
                    updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT
                )
            """)
            columns_sql = ", ".join(carried)
            self.conn.execute(
                f"INSERT INTO channel_settings_new ({columns_sql}) "
                f"SELECT {select_list} FROM channel_settings")
            self.conn.execute("DROP TABLE channel_settings")
            self.conn.execute("ALTER TABLE channel_settings_new RENAME TO channel_settings")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self.log_info(
            "DB: rebuilt channel_settings to attach the capability CHECK constraints"
            + (f"; normalized {rogue} out-of-range boolean value(s) to NULL" if rogue else ""))

    # -------------------------------------------------------------- inventory rename (respec §3.5)

    def _record_migration_key(self, key: str) -> None:
        """Mark a one-shot migration done. Idempotent; commits on its own."""
        self.conn.execute("INSERT OR IGNORE INTO bot_meta (key, value) VALUES (?, ?)",
                          (key, datetime.now().isoformat()))
        self.conn.commit()

    def _migrate_drop_compaction_schema(self):
        """Drop the P4a compaction schema and the P1 snapshot store, once (respec §3.3).

        The stream builder no longer reads a snapshot pointer, so a stale row cannot fail a turn —
        but it CAN keep megabytes of summary payload blobs and crawl skeletons in a database whose
        every writer and reader was removed in the same wave. Nothing dropped here is a transcript,
        a config row, or anything Slack does not still hold.
        """
        if self.get_meta(_P4A_CLEANUP_KEY):
            return
        # ATOMIC: the drops and the completion marker commit together or not at all. A partial
        # drop that recorded the marker would leave orphaned tables nothing will ever revisit;
        # a partial drop that did NOT record it is simply retried on the next boot.
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for table in ("compaction_cancellation_intent", "pending_recompaction",
                          "compaction_telemetry_outbox", "compaction_event_skeleton",
                          "compaction_crawl_checkpoints", "snapshot_anchor_provenance",
                          "snapshot_capture_manifest", "snapshot_mutation_observations",
                          "channel_snapshot_pointer", "channel_snapshots"):
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            # IN THE SAME TRANSACTION as the drops.
            self.conn.execute("INSERT OR IGNORE INTO bot_meta (key, value) VALUES (?, ?)",
                              (_P4A_CLEANUP_KEY, datetime.now().isoformat()))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _migrate_coverage_to_inventory(self):
        """Rename channel_coverage's coverage_* columns to inventory_* (respec §3.5).

        The sweep now builds a ROOT INVENTORY, not a coverage floor, and the column names are
        what a reader of this schema goes by. SQLite cannot rename a column on the versions this
        database must still open, so it is the standard table rebuild in ONE transaction.

        FAILS STARTUP on error, unlike _migration_step: every inventory accessor below reads the
        new names, so a half-renamed table serves traffic that cannot read its own index.
        """
        if self.get_meta(_COVERAGE_RENAME_KEY):
            return
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(channel_coverage)")}
        if not columns or "inventory_start_ts" in columns:
            self._record_migration_key(_COVERAGE_RENAME_KEY)
            return

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("""
                CREATE TABLE channel_coverage_new (
                    team_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    inventory_start_ts TEXT NOT NULL,
                    bootstrap_status TEXT NOT NULL
                        CHECK (bootstrap_status IN ('pending', 'running', 'complete', 'limited')),
                    inventory_reason TEXT,
                    sweep_token TEXT,
                    heartbeat_ts TIMESTAMP,
                    updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (team_id, channel_id)
                )
            """)
            self.conn.execute(
                "INSERT INTO channel_coverage_new "
                "SELECT team_id, channel_id, coverage_start_ts, bootstrap_status, "
                "       coverage_reason, sweep_token, heartbeat_ts, updated_ts "
                "FROM channel_coverage")
            self.conn.execute("DROP TABLE channel_coverage")
            self.conn.execute("ALTER TABLE channel_coverage_new RENAME TO channel_coverage")
            # IN THE SAME TRANSACTION as the rebuild: a marker that outlived a rolled-back
            # rename would skip the retry that fixes it.
            self.conn.execute("INSERT OR IGNORE INTO bot_meta (key, value) VALUES (?, ?)",
                              (_COVERAGE_RENAME_KEY, datetime.now().isoformat()))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self.log_info("DB: renamed channel_coverage coverage_* columns to inventory_*")

    def _migrate_channel_document_uniqueness(self):
        """Dedup channel document rows, then make the duplication impossible. FAILS STARTUP.

        Order matters: the index cannot be created while duplicates exist, and the duplicates
        cannot be resolved by merging (the table stores summary/page_structure/total_pages/
        metadata — no extracted text — so there is nothing to merge and a merged row would be a
        record of an observation nobody made). Whole-row wins: keep the newest row that carries
        a REAL summary, because the same file is routinely written twice — once by
        `catalog_unattended` with a placeholder, once by the turn that actually read it — and
        the placeholder must never outlive the summary.
        """
        cursor = self.conn.execute(
            f"""
            SELECT thread_id, message_ts, file_id, COUNT(*) AS n FROM documents
            WHERE {_CHANNEL_DOCS_PREDICATE} AND message_ts IS NOT NULL
            GROUP BY thread_id, message_ts, file_id HAVING n > 1
            """)
        groups = [(r["thread_id"], r["message_ts"], r["file_id"]) for r in cursor.fetchall()]
        removed = 0
        for thread_id, message_ts, file_id in groups:
            # `rowid` must be ALIASED: `documents.id` is an INTEGER PRIMARY KEY, so it IS the
            # rowid and sqlite3.Row names the selected column "id" — reading row["rowid"] then
            # raises, and this migration fails startup by design.
            rows = self.conn.execute(
                "SELECT rowid AS rid, summary, created_at FROM documents "
                "WHERE thread_id = ? AND message_ts = ? AND file_id = ? "
                "ORDER BY created_at, rowid",
                (thread_id, message_ts, file_id)).fetchall()
            real = [r for r in rows if (r["summary"] or "").strip()
                    and not is_unattended_summary(r["summary"])]
            keep = (real or rows)[-1]["rid"]
            for row in rows:
                if row["rid"] != keep:
                    self.conn.execute("DELETE FROM documents WHERE rowid = ?", (row["rid"],))
                    removed += 1
        self.conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {_CHANNEL_DOCS_INDEX} "
            f"ON documents (thread_id, message_ts, file_id) WHERE {_CHANNEL_DOCS_PREDICATE}")
        self.conn.commit()
        if removed:
            self.log_info(
                f"DB: removed {removed} duplicate channel document row(s) across "
                f"{len(groups)} file(s) before creating {_CHANNEL_DOCS_INDEX}")

    def _migrate_gpt56(self):
        """GPT-5.6 model-lineup migration (2026-07-09).

        Two parts, both safe to run on every startup:
        1. ONE-TIME (sentinel `gpt56_migrated` column, same pattern as the
           gpt55/gpt-image-2 swaps): move EVERYONE's default model to
           gpt-5.6-sol with medium reasoning. Users can re-customize globally
           and per channel/thread afterward — this only resets the default.
        2. EVERY-STARTUP normalizer: only gpt-5.6-sol/terra/luna and gpt-5.5
           are selectable; any other stored model (user prefs or per-thread
           overrides) coerces to gpt-5.6-sol, and stored reasoning efforts a
           model rejects are clamped (`minimal` is a 400 on 5.6 -> none;
           `max` doesn't exist on 5.5 -> xhigh). Guarantees the API layer
           never receives a dropped model name or an unsupported effort.

        Both parts are individually isolated: a failure in the one-time swap must
        not take the every-startup normalizers down with it, since those are what
        keep the API layer from ever seeing a dropped model or a rejected effort.
        """
        with self._migration_step("gpt-5.6 migration"):
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'gpt56_migrated' not in columns:
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN gpt56_migrated INTEGER DEFAULT 0
                """)
                cursor = self.conn.execute("""
                    UPDATE user_preferences
                    SET model = 'gpt-5.6-sol', reasoning_effort = 'medium', gpt56_migrated = 1
                    WHERE gpt56_migrated = 0
                """)
                swapped = cursor.rowcount
                self.conn.commit()
                self.log_info(
                    f"DB: One-time GPT-5.6 migration — swapped {swapped} user(s) to "
                    f"gpt-5.6-sol with medium reasoning"
                )

        with self._migration_step("gpt-5.6 normalizers"):
            supported = "('gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.5')"
            cursor = self.conn.execute(f"""
                UPDATE user_preferences
                SET model = 'gpt-5.6-sol'
                WHERE model IS NOT NULL AND model NOT IN {supported}
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Normalized {cursor.rowcount} user(s) from dropped models to gpt-5.6-sol"
                )
            cursor = self.conn.execute("""
                UPDATE user_preferences
                SET reasoning_effort = 'none'
                WHERE model LIKE 'gpt-5.6%' AND reasoning_effort = 'minimal'
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Clamped reasoning minimal->none for {cursor.rowcount} user(s) on 5.6 models"
                )
            cursor = self.conn.execute("""
                UPDATE user_preferences
                SET reasoning_effort = 'xhigh'
                WHERE model = 'gpt-5.5' AND reasoning_effort = 'max'
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Clamped reasoning max->xhigh for {cursor.rowcount} user(s) on gpt-5.5"
                )
            cursor = self.conn.execute(f"""
                UPDATE threads
                SET config_json = json_set(config_json, '$.model', 'gpt-5.6-sol')
                WHERE config_json IS NOT NULL
                  AND json_extract(config_json, '$.model') IS NOT NULL
                  AND json_extract(config_json, '$.model') NOT IN {supported}
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Normalized {cursor.rowcount} thread override(s) to gpt-5.6-sol"
                )
            cursor = self.conn.execute("""
                UPDATE threads
                SET config_json = json_set(config_json, '$.reasoning_effort', 'none')
                WHERE config_json IS NOT NULL
                  AND json_extract(config_json, '$.model') LIKE 'gpt-5.6%'
                  AND json_extract(config_json, '$.reasoning_effort') = 'minimal'
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Clamped {cursor.rowcount} thread override(s) minimal->none on 5.6 models"
                )
            cursor = self.conn.execute("""
                UPDATE threads
                SET config_json = json_set(config_json, '$.reasoning_effort', 'xhigh')
                WHERE config_json IS NOT NULL
                  AND json_extract(config_json, '$.model') = 'gpt-5.5'
                  AND json_extract(config_json, '$.reasoning_effort') = 'max'
            """)
            if cursor.rowcount:
                self.log_info(
                    f"DB: Clamped {cursor.rowcount} thread override(s) max->xhigh on gpt-5.5"
                )
            self.conn.commit()

        with self._migration_step("participation redesign: memory cleanup"):
            self._migrate_participation_redesign()

        with self._migration_step("drop channel_thread_mutes table"):
            # The per-thread mute mechanism was removed entirely. Drop the normalized table Layer
            # 0 created. Separately keyed from the participation-redesign step above (not folded
            # into it) so it is isolated by its own try/except and runs on every boot — including
            # the already-migrated live DB, which still has the table. DROP TABLE IF EXISTS is
            # idempotent: a no-op on a fresh DB that never created it and on every re-run.
            self.conn.execute("DROP TABLE IF EXISTS channel_thread_mutes")
            self.conn.commit()

        with self._migration_step("outbound_receipts.receipt_class"):
            # EDIT_OWN_MESSAGE spec §4: nullable class column. Existing rows stay NULL —
            # legacy, ineligible for anything class-gated, never backfilled or inferred.
            cursor = self.conn.execute("PRAGMA table_info(outbound_receipts)")
            if not any(col[1] == "receipt_class" for col in cursor.fetchall()):
                self.conn.execute(
                    "ALTER TABLE outbound_receipts ADD COLUMN receipt_class TEXT")
                self.conn.commit()

        with self._migration_step("pending_share_receipts.receipt_class"):
            # Same ruling for the pending-share sidecar: the class rides the pending row so a
            # share resolves into a receipt with its class present, never inferred at resolve.
            cursor = self.conn.execute("PRAGMA table_info(pending_share_receipts)")
            if not any(col[1] == "receipt_class" for col in cursor.fetchall()):
                self.conn.execute(
                    "ALTER TABLE pending_share_receipts ADD COLUMN receipt_class TEXT")
                self.conn.commit()

    def _migrate_participation_redesign(self):
        """Layer 0 of the participation-backoff redesign. Idempotent and re-runnable.

        Runs on EVERY init (``_migration_step`` is only a try/except, not a one-time guard), so
        every step below MUST converge to the same state on a re-run and MUST NOT clobber rows
        the running system now owns.

        1. Clear the legacy JSON channel_settings.muted_threads column. The per-thread mute
           mechanism was removed, so there is no longer any table to migrate the entries into —
           the column is dead weight. Nothing reads it anymore, so nulling it is inert for the
           running system, and clearing converges on the same state on every re-run. (The table
           itself is dropped by a separate, later migration step.)
        2. Delete the old auto-written "butt out / raise the bar" channel-memory facts
           (author LIKE 'participation_engine:%') — BUT NOT the new per-dimension preference
           markers (author LIKE 'participation_engine:pref:%'), which are live state the
           redesign writes and must survive every restart. The old generic facts kept the
           classifier suppressed channel-wide after the fix; the pref markers replace them.
        3. Collapse any duplicate preference markers to one row per (channel, dimension) and
           enforce that with a partial UNIQUE index — the invariant upsert_channel_pref_memory
           relies on. Done here (not in init_schema) so the dedup runs BEFORE the UNIQUE index
           is created and a re-run never trips over pre-existing duplicates.

        Structural channel_settings columns are deliberately NOT rewritten: the mute clobber
        overwrote updated_by, so we cannot prove the stored values were implicit, and the one
        affected channel was already reset by hand.
        """
        # 1. CLEAR the legacy JSON muted_threads column. The per-thread mute mechanism was
        #    removed, so there is nothing to copy anywhere — just null out the inert column so no
        #    stale blob lingers. This is the only writer of the column and it converges on the
        #    same state every re-run.
        cleared = self.conn.execute(
            "UPDATE channel_settings SET muted_threads = NULL "
            "WHERE muted_threads IS NOT NULL AND muted_threads != ''"
        )
        if cleared.rowcount:
            self.log_info(
                f"DB: Cleared legacy muted_threads JSON on {cleared.rowcount} channel(s)")

        # 2. Remove the stale severe participation-engine memory facts — but PRESERVE the
        #    per-dimension preference markers (participation_engine:pref:*), which are live
        #    redesign state, not stale suppression facts.
        deleted = self.conn.execute(
            "DELETE FROM channel_memory "
            "WHERE author LIKE 'participation_engine:%' "
            "AND author NOT LIKE 'participation_engine:pref:%'"
        )
        if deleted.rowcount:
            self.log_info(
                f"DB: Removed {deleted.rowcount} stale participation-engine memory fact(s)")

        # 3. One preference marker per (channel, dimension). Collapse any duplicates BEFORE
        #    creating the partial UNIQUE index the upsert relies on.
        #    SHOULD-FIX 2: the marker is a CHANNEL-scope row (the upsert only ever writes/reads
        #    scope='channel'), so both the dedupe and the index predicate must be scoped to
        #    'channel' — otherwise a same-named WORKSPACE row could be swept by the dedupe or
        #    collide with a valid channel marker in the (channel_id, author) unique index. And
        #    keep the FRESHEST row (greatest updated_ts, id as the tie-breaker), not merely the
        #    highest id: the upsert refreshes a marker in place with a new updated_ts, so a
        #    later-refreshed but lower-id row must win over a stale higher-id duplicate.
        #    COALESCE guards any legacy NULL updated_ts (falls back to created_ts, then '').
        self.conn.execute(
            "DELETE FROM channel_memory "
            "WHERE author LIKE 'participation_engine:pref:%' AND scope = 'channel' "
            "AND id NOT IN ("
            "  SELECT m.id FROM channel_memory m "
            "  WHERE m.author LIKE 'participation_engine:pref:%' AND m.scope = 'channel' "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM channel_memory m2 "
            "    WHERE m2.author LIKE 'participation_engine:pref:%' AND m2.scope = 'channel' "
            "    AND m2.channel_id = m.channel_id AND m2.author = m.author "
            "    AND (COALESCE(m2.updated_ts, m2.created_ts, '') > COALESCE(m.updated_ts, m.created_ts, '') "
            "      OR (COALESCE(m2.updated_ts, m2.created_ts, '') = COALESCE(m.updated_ts, m.created_ts, '') "
            "          AND m2.id > m.id))))"
        )
        # Drop first so a re-run REPLACES an index created under the old (scope-agnostic)
        # predicate; CREATE ... IF NOT EXISTS alone would silently keep the stale predicate.
        # The RESERVED POLICY ROW: at most one per channel. `scope` already distinguishes row
        # KINDS ('channel' private facts vs 'workspace' shared ones), so 'policy' extends that
        # column rather than introducing a parallel notion of kind. A partial unique index is
        # what makes "reserved" true in the storage rather than only in the code that writes it.
        self.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_policy")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_memory_policy "
            "ON channel_memory (channel_id) WHERE scope = 'policy'"
        )
        self.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_pref_marker")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_memory_pref_marker "
            "ON channel_memory (channel_id, author) "
            "WHERE author LIKE 'participation_engine:pref:%' AND scope = 'channel'"
        )
        self.conn.commit()

    # Thread operations
    def get_or_create_thread(self, thread_id: str, channel_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Get existing thread or create new one with user defaults.
        
        Args:
            thread_id: Thread identifier (channel_id:thread_ts format)
            channel_id: Channel ID
            user_id: Optional user ID to copy defaults from
            
        Returns:
            Thread data dictionary
        """
        self.log_debug(f"DB: get_or_create_thread - thread={thread_id}, channel={channel_id}, user={user_id}")
        
        # Try to get existing thread
        cursor = self.conn.execute(
            "SELECT * FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update last activity
            self.update_thread_activity(thread_id)
            self.log_debug(f"DB: Found existing thread {thread_id}")
            return dict(row)
        
        # Create new thread
        thread_ts = thread_id.split(":", 1)[1] if ":" in thread_id else thread_id
        
        # Get user config if user_id provided
        config = {}
        if user_id:
            user_config = self.get_user_config(user_id)
            if user_config:
                config = user_config
                self.log_debug(f"DB: Applied user config for {user_id} to new thread")
        
        try:
            self.conn.execute("""
                INSERT INTO threads (thread_id, channel_id, thread_ts, config_json)
                VALUES (?, ?, ?, ?)
            """, (thread_id, channel_id, thread_ts, json.dumps(config) if config else None))
            
            self.log_info(f"DB: Created new thread {thread_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to create thread {thread_id} - {e}", exc_info=True)
            raise
        
        return self.get_or_create_thread(thread_id, channel_id)
    def save_thread_config(self, thread_id: str, config: Dict):
        """
        Save thread configuration.
        
        Args:
            thread_id: Thread identifier
            config: Configuration dictionary
        """
        self.conn.execute("""
            UPDATE threads 
            SET config_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        """, (json.dumps(config), thread_id))
        
        logger.debug(f"Saved config for thread {thread_id}")
    
    def get_thread_config(self, thread_id: str) -> Optional[Dict]:
        """
        Get thread configuration.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Configuration dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT config_json FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        
        if row and row["config_json"]:
            return json.loads(row["config_json"])
        
        return None
    
    def get_channel_settings(self, channel_id: str) -> Optional[Dict]:
        """Get per-channel settings (Phase 7). Returns a dict or None if the channel has no row."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            fenced = store.channel_settings(key)
            return {k: v for k, v in fenced.items() if k != "ambient_memory"}
        cursor = self.conn.execute(
            "SELECT response_mode, reply_in_channel, participation_level, "
            "snoozed_until, muted_threads, model, reasoning_effort, verbosity, "
            "enable_web_search, enable_mcp, image_model, updated_ts, updated_by "
            "FROM channel_settings WHERE channel_id = ?",
            (channel_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "response_mode": row["response_mode"],
            # The two capability booleans are handed back RAW, not through bool(): the CHECK
            # constraint arrived after the column, so a legacy row can still hold `2` or `-1`,
            # and `BotConfig._channel_capability_profile` is the layer that recognises those as
            # unusable and falls back loudly. Coercing here would make that guard unreachable
            # and turn a stored `-1` into "web search enabled".
            "enable_web_search": row["enable_web_search"],
            "enable_mcp": row["enable_mcp"],
            "image_model": row["image_model"],
            # NULL reply_in_channel stays None (inherit → resolves to config.reply_in_channel_default
            # at read time). Collapsing NULL to False here erased the inherit distinction.
            "reply_in_channel": (None if row["reply_in_channel"] is None
                                 else bool(row["reply_in_channel"])),
            "participation_level": row["participation_level"],
            "snoozed_until": row["snoozed_until"],
            "muted_threads": _decode_muted_threads(row["muted_threads"]),
            "model": row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "verbosity": row["verbosity"],
            "updated_ts": row["updated_ts"],
            "updated_by": row["updated_by"],
        }

    def set_channel_settings(self, channel_id: str, response_mode=_UNSET,
                             reply_in_channel=_UNSET,
                             participation_level=_UNSET, snoozed_until=_UNSET,
                             muted_threads=_UNSET,
                             model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                             ambient_memory=_UNSET,
                             enable_web_search=_UNSET, enable_mcp=_UNSET, image_model=_UNSET,
                             updated_by: Optional[str] = None):
        """Upsert per-channel settings (Phase 7; Phase F adds participation_level/snoozed_until).

        Atomic partial write: ONLY the explicitly-provided fields are written — omitted fields are
        preserved untouched (never rewritten, so no clobber and no race with a concurrent save).
        An explicit value sets a field; an explicit None CLEARS it (→ NULL) so the modal's "inherit
        from global default" stores NULL rather than a copied default (NULL then resolves to the
        global default at read time). updated_ts/updated_by bump only when a STRUCTURAL field
        changed. muted_threads is a deprecated inert JSON column (nothing reads it — the
        per-thread mute mechanism was removed); it takes a Python list, None/[] clears it.
        """
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            store.set_channel_settings(key, **_epoch_settings_fields(
                response_mode=response_mode, reply_in_channel=reply_in_channel,
                participation_level=participation_level, snoozed_until=snoozed_until,
                muted_threads=muted_threads, model=model, reasoning_effort=reasoning_effort,
                verbosity=verbosity, ambient_memory=ambient_memory,
                enable_web_search=enable_web_search, enable_mcp=enable_mcp,
                image_model=image_model, updated_by=updated_by))
            return
        built = _build_channel_settings_write(
            channel_id, response_mode=response_mode,
            reply_in_channel=reply_in_channel, participation_level=participation_level,
            snoozed_until=snoozed_until, muted_threads=muted_threads, model=model,
            reasoning_effort=reasoning_effort, verbosity=verbosity,
            ambient_memory=ambient_memory, enable_web_search=enable_web_search,
            enable_mcp=enable_mcp, image_model=image_model, updated_by=updated_by)
        if built is None:
            return
        sql, params = built
        if ambient_memory is False:
            # Track 1: opting out must purge the derived narrative ATOMICALLY with the settings
            # write. The sync connection is autocommit (isolation_level=None), so two separate
            # execute()s each self-commit and a trailing commit() binds nothing — take an explicit
            # BEGIN IMMEDIATE … COMMIT (ROLLBACK on error), mirroring the async path's idiom.
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute(sql, params)
                self.conn.execute(
                    "DELETE FROM channel_summaries WHERE channel_id = ?", (channel_id,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        else:
            self.conn.execute(sql, params)
            self.conn.commit()
        logger.debug(f"Saved channel_settings for {channel_id}")

    # --- Per-channel memory (Phase 9) ---
    def get_channel_memory(self, channel_id: str) -> List[Dict]:
        """Return durable memory visible to a channel: its own channel-scope rows + shared
        workspace-scope rows. A channel NEVER sees another channel's channel-scope rows."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            return store.memory(key)
        cursor = self.conn.execute(
            "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
            "FROM channel_memory WHERE (scope = 'channel' AND channel_id = ?) OR scope = 'workspace' "
            "ORDER BY updated_ts ASC",
            (channel_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # --- the reserved policy row ---------------------------------------------------------
    #
    # Deliberately NOT reachable through get_channel_memory / add_channel_memory: those are the
    # generic fact API the memory tools and the settings textarea speak, and the policy row must
    # be untouchable from all of them. Its own accessors are the only way in, and the two
    # authorized writers (the settings modal and set_channel_participation) are their only
    # callers.

    def get_channel_policy(self, channel_id: str) -> Optional[Dict]:
        """The channel's standing policy row, or None. Never returns facts."""
        cursor = self.conn.execute(
            "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
            "FROM channel_memory WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
            (channel_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_channel_policy(self, channel_id: str, content: Optional[str],
                           author: Optional[str] = None) -> None:
        """REPLACE the whole policy, or DELETE it when the text is blank.

        Replace, never append or patch: a standing policy is a single statement of how the bot
        should behave here, and merging a new one into an old one would silently accumulate
        contradictions nobody chose. Blank means the operator cleared it, which is a deletion —
        an empty row would render an empty instructions heading."""
        text = (content or "").strip()
        if not text:
            self.conn.execute(
                "DELETE FROM channel_memory WHERE scope = 'policy' AND channel_id = ?",
                (channel_id,))
            self.conn.commit()
            return
        # The partial unique index makes this an upsert rather than a race between readers.
        self.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) "
            "VALUES (?, 'policy', ?, ?) "
            "ON CONFLICT (channel_id) WHERE scope = 'policy' "
            "DO UPDATE SET content = excluded.content, author = excluded.author, "
            "              updated_ts = CURRENT_TIMESTAMP",
            (channel_id, text, author))
        self.conn.commit()

    def add_channel_memory(self, channel_id: str, content: str, scope: str = "channel",
                           author: Optional[str] = None) -> int:
        """Insert a memory row; returns the new id."""
        cursor = self.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            (channel_id, scope, content, author)
        )
        self.conn.commit()
        logger.debug(f"Added channel_memory for {channel_id} (scope={scope})")
        return cast(int, cursor.lastrowid)

    def update_channel_memory(self, memory_id: int, content: str):
        """Update an existing memory row's content (and updated_ts)."""
        self.conn.execute(
            "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP WHERE id = ?",
            (content, memory_id)
        )
        self.conn.commit()

    def delete_channel_memory(self, memory_id: int):
        """Delete a memory row (manual forget / cap eviction)."""
        self.conn.execute("DELETE FROM channel_memory WHERE id = ?", (memory_id,))
        self.conn.commit()

    # --- Response feedback (Phase H) ---
    def record_response_feedback(self, channel_id: str, thread_ts: Optional[str],
                                 message_ts: str, user_id: str, signal: int,
                                 source: str) -> None:
        """Upsert one feedback signal. A user changing their thumb (same message,
        same source) updates the existing row rather than adding a second vote."""
        self.conn.execute("""
            INSERT INTO response_feedback (channel_id, thread_ts, message_ts, user_id, signal, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_ts, user_id, source) DO UPDATE SET
                signal=excluded.signal,
                updated_ts=CURRENT_TIMESTAMP
        """, (channel_id, thread_ts, message_ts, user_id, signal, source))
        self.conn.commit()
        logger.debug(f"Recorded response feedback {signal:+d} ({source}) on {channel_id}:{message_ts}")

    def delete_response_feedback(self, message_ts: str, user_id: str, source: str) -> None:
        """Remove a feedback row (future reaction_removed handling)."""
        self.conn.execute(
            "DELETE FROM response_feedback WHERE message_ts = ? AND user_id = ? AND source = ?",
            (message_ts, user_id, source)
        )
        self.conn.commit()

    def get_channel_feedback_ratio(self, channel_id: str, days: int = 30):
        """(positive, negative, ratio) for a channel's recent feedback.

        ratio is positive/(positive+negative), or None when there's no feedback —
        callers must treat None as "no signal", not as zero. Read-only plumbing for
        the participation engine; not wired into decisions yet."""
        cursor = self.conn.execute(
            "SELECT "
            "  SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END) AS positive, "
            "  SUM(CASE WHEN signal < 0 THEN 1 ELSE 0 END) AS negative "
            "FROM response_feedback "
            "WHERE channel_id = ? AND created_ts >= datetime('now', ?)",
            (channel_id, f"-{int(days)} days")
        )
        row = cursor.fetchone()
        positive = row["positive"] or 0
        negative = row["negative"] or 0
        total = positive + negative
        return positive, negative, (positive / total if total else None)

    def update_thread_activity(self, thread_id: str):
        """
        Update thread's last activity timestamp.
        
        Args:
            thread_id: Thread identifier
        """
        self.conn.execute("""
            UPDATE threads 
            SET last_activity = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        """, (thread_id,))
    
    # Message operations
    
    # Thread summary operations (Phase S — rolling compaction store)
    def get_thread_summary(self, thread_id: str) -> Optional[Dict]:
        """
        Get the compaction summary row for a thread, if one exists.

        Returns:
            Dict with summary_text, boundary_ts, refs (parsed list), preserved_ts
            (parsed list), updated_ts — or None.
        """
        cursor = self.conn.execute(
            "SELECT * FROM thread_summaries WHERE thread_id = ?", (thread_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        summary = dict(row)
        summary["refs"] = json.loads(summary["refs_json"]) if summary.get("refs_json") else []
        # NULL (legacy row, written before the column existed) stays None = "unknown", which the
        # rebuild fails safe on; an explicit "[]" means verified-empty (fast tail-only path). F3.
        _pts = summary.get("preserved_ts_json")
        summary["preserved_ts"] = json.loads(_pts) if _pts is not None else None
        return summary

    def save_thread_summary(self, thread_id: str, summary_text: str, boundary_ts: str,
                            refs: Optional[List[Dict]] = None,
                            preserved_ts: Optional[List[str]] = None):
        """
        Upsert the compaction summary for a thread (one row per thread, rolling).

        Args:
            thread_id: Thread identifier
            summary_text: Summary covering everything at or before boundary_ts
            boundary_ts: Slack ts of the newest message covered by the summary
            refs: Structured refs (files/images/links) from the summarized span
            preserved_ts: Slack ts of preserved messages (images/summarized docs) kept
                in live context but sitting at/behind boundary_ts (F3)
        """
        self.conn.execute("""
            INSERT INTO thread_summaries
                (thread_id, summary_text, boundary_ts, refs_json, preserved_ts_json, updated_ts)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                boundary_ts = excluded.boundary_ts,
                refs_json = excluded.refs_json,
                preserved_ts_json = excluded.preserved_ts_json,
                updated_ts = CURRENT_TIMESTAMP
        """, (thread_id, summary_text, boundary_ts,
              json.dumps(refs) if refs else None,
              # `is not None`, not truthiness: an explicit [] must persist as "[]" (verified
              # empty), distinct from NULL (legacy/unknown). F3.
              json.dumps(preserved_ts) if preserved_ts is not None else None))
        self.log_info(f"DB: Saved thread summary for {thread_id} (boundary_ts={boundary_ts})")

    def delete_thread_summary(self, thread_id: str):
        """Delete the compaction summary for a thread (and its late-artifact addenda —
        PRAGMA foreign_keys is never enabled, so the cascade is explicit)."""
        self.conn.execute("DELETE FROM thread_summaries WHERE thread_id = ?", (thread_id,))
        self.conn.execute("DELETE FROM thread_summary_addenda WHERE thread_id = ?", (thread_id,))

    # Thread summary addenda (F51c — late-artifact context folded onto the summary head)
    async def add_thread_summary_addendum_async(
        self, thread_id: str, channel_id: str, source_ts: str, kind: str, ref: str, note: str,
        *, cap: int = _MAX_SUMMARY_ADDENDA_PER_THREAD,
    ) -> bool:
        """Record a late/folded ambient-artifact note against a thread's compaction summary.

        Idempotent on (thread_id, source_ts, kind, ref) so the completion-time path and the
        compaction-time path can't double-record the same note. Bounded: at most `cap` addenda
        per thread (a pathological channel can't bloat the summary head). Returns True when a
        row was actually inserted."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT 1 FROM thread_summary_addenda "
                "WHERE thread_id = ? AND source_ts = ? AND kind = ? AND ref = ?",
                (thread_id, source_ts, kind, ref)) as cur:
                if await cur.fetchone():
                    return False  # already recorded (idempotent) — doesn't re-count against cap
            async with db.execute(
                "SELECT COUNT(*) FROM thread_summary_addenda WHERE thread_id = ?",
                (thread_id,)) as cur:
                row = await cur.fetchone()
                if row and int(row[0]) >= int(cap):
                    return False  # cap reached — drop silently rather than bloat the head
            await db.execute(
                "INSERT OR IGNORE INTO thread_summary_addenda "
                "(thread_id, channel_id, source_ts, kind, ref, note) VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, channel_id, source_ts, kind, ref, note))
            await db.commit()
        return True

    async def get_thread_summary_addenda_async(self, thread_id: str) -> List[Dict]:
        """Late-artifact addenda for a thread, deterministically ordered (source_ts, id) so
        every rebuild serializes the summary head identically (prompt-cache hygiene). source_ts
        is a numeric Slack ts stored as TEXT, so order by its REAL value, not string collation."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM thread_summary_addenda WHERE thread_id = ? "
                "ORDER BY CAST(source_ts AS REAL) ASC, id ASC", (thread_id,)) as cursor:
                return [dict(row) async for row in cursor]

    # Image operations
    def save_image_metadata(self, thread_id: str, url: str, image_type: str,
                           prompt: Optional[str] = None, analysis: Optional[str] = None,
                           original_analysis: Optional[str] = None, metadata: Optional[Dict] = None,
                           message_ts: Optional[str] = None):
        """
        Save image metadata (NO base64 data).
        
        Args:
            thread_id: Thread identifier
            url: Image URL
            image_type: Type of image (uploaded/generated/edited)
            prompt: Full generation/edit prompt
            analysis: Full vision analysis
            original_analysis: For edited images, the pre-edit analysis
            metadata: Additional metadata
            message_ts: Message timestamp to link image to specific message
        """
        self.log_debug(f"DB: Saving image - thread={thread_id}, url={url[:100]}, "
                      f"type={image_type}, has_analysis={bool(analysis)}, "
                      f"analysis_len={len(analysis) if analysis else 0}, "
                      f"prompt_len={len(prompt) if prompt else 0}")
        
        try:
            # Merge-preserving upsert (F1): a later write over the same URL (the
            # post-refresh Slack rebuild saves the uploaded file with an empty caption,
            # and the ledger issues its own upsert) must NOT erase the non-empty
            # prompt/analysis/type/generation_id an earlier write already recorded.
            # Existing non-empty values win over incoming empties; message_ts fills in.
            self.conn.execute("""
                INSERT INTO images
                (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    image_type = COALESCE(NULLIF(images.image_type, ''), excluded.image_type),
                    prompt = COALESCE(NULLIF(images.prompt, ''), excluded.prompt),
                    analysis = COALESCE(NULLIF(images.analysis, ''), excluded.analysis),
                    original_analysis = COALESCE(NULLIF(images.original_analysis, ''), excluded.original_analysis),
                    metadata_json = COALESCE(NULLIF(images.metadata_json, ''), excluded.metadata_json),
                    message_ts = COALESCE(excluded.message_ts, images.message_ts)
            """, (thread_id, url, image_type, prompt, analysis, original_analysis,
                  json.dumps(metadata) if metadata else None, message_ts))

            self.log_info(f"DB: Successfully saved image metadata for {url[:50]}... in thread {thread_id}")

        except sqlite3.IntegrityError as e:
            self.log_warning(f"DB: Image metadata already exists for {url[:50]}...: {e}")
        except Exception as e:
            self.log_error(f"DB: Failed to save image metadata - {e}", exc_info=True)
            raise
    def get_image_analysis_by_url(self, thread_id: str, url: str) -> Optional[Dict]:
        """
        Get image analysis by URL (thread-isolated).
        
        Args:
            thread_id: Thread identifier
            url: Image URL
            
        Returns:
            Image metadata dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ? AND url = ?
        """, (thread_id, url))
        
        row = cursor.fetchone()
        if row:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            return img
        
        return None
    
    def get_images_by_message(self, thread_id: str, message_ts: str) -> List[Dict]:
        """
        Get images associated with a specific message.
        
        Args:
            thread_id: Thread identifier
            message_ts: Message timestamp
            
        Returns:
            List of image metadata dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ? AND message_ts = ?
            ORDER BY created_at ASC
        """, (thread_id, message_ts))
        
        images = []
        for row in cursor:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            images.append(img)
        
        return images
    
    def find_thread_images(self, thread_id: str, image_type: Optional[str] = None) -> List[Dict]:
        """
        Find all images for a thread.
        
        Args:
            thread_id: Thread identifier
            image_type: Optional filter by image type
            
        Returns:
            List of image metadata dictionaries
        """
        if image_type:
            cursor = self.conn.execute("""
                SELECT * FROM images 
                WHERE thread_id = ? AND image_type = ?
                ORDER BY created_at ASC
            """, (thread_id, image_type))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM images 
                WHERE thread_id = ?
                ORDER BY created_at ASC
            """, (thread_id,))
        
        images = []
        for row in cursor:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            images.append(img)
        
        return images
    
    def get_latest_thread_image(self, thread_id: str) -> Optional[Dict]:
        """
        Get the most recent image for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Image metadata dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (thread_id,))
        
        row = cursor.fetchone()
        if row:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            return img
        
        return None
    
    # User preferences operations
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """
        Get user preferences for settings modal.
        
        Args:
            user_id: Slack user ID
            
        Returns:
            User preferences dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM user_preferences WHERE slack_user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            prefs = dict(row)
            # Convert SQLite boolean (0/1) to Python boolean
            prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
            prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
            prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
            prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
            return prefs
        
        return None
    
    def create_default_user_preferences(self, user_id: str, email: Optional[str] = None) -> Dict:
        """
        Create default user preferences based on environment variables.
        
        Args:
            user_id: Slack user ID
            email: Optional user email
            
        Returns:
            Dictionary of default preferences
        """
        from config import config
        
        defaults = {
            'slack_user_id': user_id,
            'slack_email': email,
            'model': config.gpt_model,
            'reasoning_effort': config.default_reasoning_effort,
            'verbosity': config.default_verbosity,
            'temperature': config.default_temperature,
            'top_p': config.default_top_p,
            'enable_web_search': config.enable_web_search,
            'enable_streaming': config.enable_streaming,
            'image_model': config.image_model,
            'image_size': config.default_image_size,
            'image_quality': config.default_image_quality,
            'image_background': config.default_image_background,
            'input_fidelity': config.default_input_fidelity,
            'vision_detail': config.default_detail_level,
            'settings_completed': False
        }

        try:
            # Insert with defaults
            self.conn.execute("""
                INSERT INTO user_preferences
                (slack_user_id, slack_email, model, reasoning_effort, verbosity,
                 temperature, top_p, enable_web_search, enable_streaming,
                 image_model, image_size, image_quality, image_background, input_fidelity, vision_detail, settings_completed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email, defaults['model'],
                defaults['reasoning_effort'], defaults['verbosity'],
                defaults['temperature'], defaults['top_p'],
                1 if defaults['enable_web_search'] else 0,
                1 if defaults['enable_streaming'] else 0,
                defaults['image_model'],
                defaults['image_size'], defaults['image_quality'],
                defaults['image_background'], defaults['input_fidelity'],
                defaults['vision_detail'], 0
            ))
            
            self.log_info(f"DB: Created default preferences for user {user_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to create default preferences for {user_id} - {e}")
            
        return defaults
    
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: Slack user ID
            preferences: Dictionary of preferences to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build dynamic UPDATE query based on provided fields
            updates = []
            values = []
            
            for field in ['model', 'reasoning_effort', 'verbosity', 'temperature',
                         'top_p', 'image_size', 'image_quality', 'image_background',
                         'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    updates.append(f"{field} = ?")
                    values.append(preferences[field])
            
            # Handle boolean fields
            for field in ['enable_web_search', 'enable_streaming', 'enable_mcp']:
                if field in preferences:
                    updates.append(f"{field} = ?")
                    values.append(1 if preferences[field] else 0)
            
            # Always update timestamp
            updates.append("updated_at = strftime('%s', 'now')")
            
            # Add user_id for WHERE clause
            values.append(user_id)
            
            query = f"""
                UPDATE user_preferences 
                SET {', '.join(updates)}
                WHERE slack_user_id = ?
            """
            
            self.conn.execute(query, values)
            self.log_info(f"DB: Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            self.log_error(f"DB: Failed to update preferences for {user_id} - {e}")
            return False
    
    # Document operations
    
    def save_document(self, thread_id: str, filename: str, mime_type: str,
                     summary: Optional[str] = None, file_id: Optional[str] = None,
                     url_private: Optional[str] = None, size_bytes: Optional[int] = None,
                     page_structure: Optional[Dict] = None,
                     total_pages: Optional[int] = None,
                     metadata: Optional[Dict] = None, message_ts: Optional[str] = None):
        """
        Save document summary + metadata + Slack CDN ref. Full content is NEVER
        persisted (CLAUDE.md pitfall 6a) — it is re-derived in memory on demand.

        On the CHANNEL surface this is an atomic whole-row UPGRADE rather than a plain insert. A
        channel turn writes the same file twice by design: the stream's origin ingester records
        the unattended placeholder before admission, and finalization then writes the extraction
        it just paid a utility model for. The second write is the one that read the file, so it
        replaces the row whole — a merge would record an observation nobody made, and the plain
        insert this used to be hit the channel unique index and failed the turn AFTER the
        summarization spend.

        DM behaviour is unchanged, structurally rather than by a branch: the ON CONFLICT clause
        names the channel-scoped partial index, and a DM row has no entry in it, so no conflict
        can arise and the statement is the INSERT it always was.

        The one write that must not win is the placeholder — `catalog_unattended` comes through
        here too, and a real summary must never be replaced by "not yet read" (the same rule the
        dedup migration applies to rows that already exist).

        Args:
            thread_id: Thread identifier
            filename: Original filename
            mime_type: Document MIME type
            summary: Attach-time summary (the only content-bearing field)
            file_id: Slack file id (read_document lookup key)
            url_private: Slack CDN URL for authenticated re-download
            size_bytes: Original file size
            page_structure: Optional page/sheet structure info as dict
            total_pages: Total page/sheet count
            metadata: Additional metadata (size, author, etc.)
            message_ts: Message timestamp to link document to specific message
        """
        self.log_debug(f"DB: Saving document - thread={thread_id}, filename={filename}, "
                      f"summary_len={len(summary) if summary else 0}, pages={total_pages}")

        try:
            self.conn.execute(f"""
                INSERT INTO documents
                (thread_id, filename, mime_type, summary, file_id, url_private,
                 size_bytes, page_structure, total_pages, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (thread_id, message_ts, file_id)
                    WHERE {_CHANNEL_DOCS_PREDICATE}
                DO UPDATE SET
                    filename = excluded.filename, mime_type = excluded.mime_type,
                    summary = excluded.summary, url_private = excluded.url_private,
                    size_bytes = excluded.size_bytes,
                    page_structure = excluded.page_structure,
                    total_pages = excluded.total_pages,
                    metadata_json = excluded.metadata_json
                WHERE NOT (? = 1
                           AND TRIM(COALESCE(documents.summary, '')) <> ''
                           AND documents.summary NOT LIKE ?)
            """, (thread_id, filename, mime_type, summary, file_id, url_private,
                  size_bytes,
                  json.dumps(page_structure) if page_structure else None,
                  total_pages,
                  json.dumps(metadata) if metadata else None, message_ts,
                  1 if is_unattended_summary(summary) else 0, _UNATTENDED_LIKE))

            # Update thread activity
            self.update_thread_activity(thread_id)
            
            self.log_info(f"DB: Successfully saved document {filename} for thread {thread_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to save document {filename} - {e}", exc_info=True)
            raise

    def restore_document_derived(self, thread_id: str, filename: str, *,
                                 summary: Optional[str] = None,
                                 page_structure: Optional[Dict] = None,
                                 total_pages: Optional[int] = None,
                                 size_bytes: Optional[int] = None,
                                 message_ts: Optional[str] = None) -> int:
        """Re-hydrate a SLIMMED document row IN PLACE and return the rows updated.

        Retention (delete_old_documents) nulls a row's derived bulk (summary/page_structure/
        metadata) but keeps the Slack ref. When a rebuild re-derives that content it must UPDATE
        the preserved row, not INSERT a second one: the `documents` table has no
        UNIQUE(thread_id, filename) constraint, so a fresh save_document each retention/rebuild
        cycle accumulates duplicate reference rows. Returns 0 when no matching row exists — the
        caller then falls back to inserting (a genuinely legacy, never-stored document)."""
        try:
            cursor = self.conn.execute("""
                UPDATE documents
                SET summary = ?, page_structure = ?, total_pages = ?,
                    size_bytes = COALESCE(?, size_bytes),
                    message_ts = COALESCE(?, message_ts)
                WHERE thread_id = ? AND filename = ?
            """, (summary,
                  json.dumps(page_structure) if page_structure else None,
                  total_pages, size_bytes, message_ts, thread_id, filename))
            if cursor.rowcount:
                self.update_thread_activity(thread_id)
                self.log_info(f"DB: Re-hydrated slimmed document {filename} for thread {thread_id}")
            return cursor.rowcount or 0
        except Exception as e:
            self.log_error(f"DB: Failed to restore document {filename} - {e}", exc_info=True)
            raise

    def get_thread_documents(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all documents for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Optional limit on number of documents returned
            
        Returns:
            List of document dictionaries
        """
        query = """
            SELECT * FROM documents 
            WHERE thread_id = ? 
            ORDER BY created_at ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query, (thread_id,))
        documents = []
        
        for row in cursor:
            doc = dict(row)
            if doc.get("page_structure"):
                doc["page_structure"] = json.loads(doc["page_structure"])
            if doc.get("metadata_json"):
                doc["metadata"] = json.loads(doc["metadata_json"])
                del doc["metadata_json"]
            documents.append(doc)
        
        return documents
    
    def get_document_by_filename(self, thread_id: str, filename: str) -> Optional[Dict]:
        """
        Get a specific document by filename within a thread.
        
        Args:
            thread_id: Thread identifier
            filename: Document filename
            
        Returns:
            Document dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM documents 
            WHERE thread_id = ? AND filename = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (thread_id, filename))
        
        row = cursor.fetchone()
        if row:
            doc = dict(row)
            if doc.get("page_structure"):
                doc["page_structure"] = json.loads(doc["page_structure"])
            if doc.get("metadata_json"):
                doc["metadata"] = json.loads(doc["metadata_json"])
                del doc["metadata_json"]
            return doc
        
        return None
    
    def delete_old_documents(self, days: int = 90):
        """SLIM document-extraction rows older than `days` (retention sweep) — do NOT delete them.

        The reference row (filename, thread/channel key, Slack file_id/url_private, mime_type,
        size, timestamps) is PRESERVED so read_document and thread rebuilds can always re-resolve
        and re-extract the file from Slack on demand. Only the bulky DERIVED fields (summary,
        page_structure, metadata) are nulled. This is the fix for the compaction-boundary gap:
        a document behind a compaction boundary is never recreated by a rebuild, so DELETING its
        row made a 100-day-old-but-still-in-Slack file unresolvable (`document_not_found`) even
        though the summary head still referenced it. Slimming ages out the bulk while keeping the
        row reachable indefinitely. created_at defaults to CURRENT_TIMESTAMP (UTC), so the cutoff
        is computed IN SQL with datetime('now', …) — a Python datetime.now() cutoff would be LOCAL
        time and skew the retention window on non-UTC hosts. Same trap as delete_old_tool_usage."""
        cursor = self.conn.execute("""
            UPDATE documents
            SET summary = NULL, page_structure = NULL, metadata_json = NULL
            WHERE created_at < datetime('now', ?)
              AND (summary IS NOT NULL OR page_structure IS NOT NULL OR metadata_json IS NOT NULL)
        """, (f"-{int(days)} days",))

        if cursor.rowcount > 0:
            self.log_info(f"DB: Slimmed {cursor.rowcount} documents older than {days} days "
                          "(refs kept for on-demand re-extraction, derived bulk cleared)")

    def delete_old_tool_usage(self, days: int = 90):
        """Delete tool-use provenance rows older than `days` (F7 retention sweep).

        The ON DELETE CASCADE path is dead (PRAGMA foreign_keys is never enabled), so
        message_tool_usage gets this explicit age sweep instead, wired into the scheduled
        cleanup worker. created_at defaults to CURRENT_TIMESTAMP (UTC), so the cutoff is
        computed in SQL with datetime('now', …) — a Python datetime.now() cutoff would be
        LOCAL time and skew the retention window on non-UTC hosts."""
        cursor = self.conn.execute("""
            DELETE FROM message_tool_usage
            WHERE created_at < datetime('now', ?)
        """, (f"-{int(days)} days",))

        if cursor.rowcount > 0:
            self.log_info(f"DB: Cleaned up {cursor.rowcount} tool-usage rows older than {days} days")

    # F32: thread-scoped code-interpreter containers
    #
    # Every staleness cutoff below is computed IN SQL with datetime('now', …). last_used_at
    # defaults to CURRENT_TIMESTAMP, which SQLite writes in UTC — a Python datetime.now()
    # cutoff would be LOCAL time and, on this host (UTC-4), would judge every container
    # four hours fresher than it is. Same trap as delete_old_tool_usage above.

    # EVERY mutation below is conditional on `container_id`, not just `thread_id`. A row can be
    # rebound to a NEW container at any moment (the old one expired, a turn recreated it), and a
    # thread_id-only write then lands on the wrong container: the daily reaper, having selected
    # stale container X, would delete the row for its live replacement Y; and a late publication
    # for X would write X's file ids into Y's dedupe list, suppressing Y's real artifacts.
    _CONTAINER_PUBLISHED_CAP = 2048  # ~8x what one 20-minute container could plausibly hold

    @staticmethod
    def _container_row(row) -> Optional[Dict]:
        if not row:
            return None
        result = dict(row)
        try:
            result["published_files"] = json.loads(result.get("published_files_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            result["published_files"] = []
        return result

    def get_thread_container(self, thread_id: str) -> Optional[Dict]:
        """The thread's container binding, regardless of age.

        Deliberately NOT age-filtered. Age belongs to container *selection* only. The dedupe
        record must stay readable for as long as the binding exists: a single turn can run
        longer than the reuse window (a tool loop with slow tools), and if publication could no
        longer read its own published-file list it would re-post every earlier artifact still
        sitting in the container.
        """
        cursor = self.conn.execute("""
            SELECT thread_id, container_id, published_files_json, created_at, last_used_at
            FROM thread_containers WHERE thread_id = ?
        """, (thread_id,))
        return self._container_row(cursor.fetchone())

    def get_fresh_thread_container(self, thread_id: str, reuse_minutes: int) -> Optional[Dict]:
        """The thread's container, but ONLY if we used it within `reuse_minutes`.

        A row older than that is not returned: the container has almost certainly idle-expired
        (20-minute API ceiling), and handing OpenAI a dead id would fail the whole turn.
        Callers treat None as "create a fresh one".
        """
        cursor = self.conn.execute("""
            SELECT thread_id, container_id, published_files_json, created_at, last_used_at
            FROM thread_containers
            WHERE thread_id = ? AND last_used_at > datetime('now', ?)
        """, (thread_id, f"-{int(reuse_minutes)} minutes"))
        return self._container_row(cursor.fetchone())

    def save_thread_container(self, thread_id: str, container_id: str):
        """Bind a NEW container to a thread, clearing the published-file record.

        The reset is deliberate: a new container starts empty, so nothing has been published
        out of it yet. Carrying the old ids forward would let a stale id suppress a genuinely
        new artifact that happened to reuse it.
        """
        self.conn.execute("""
            INSERT INTO thread_containers (thread_id, container_id, created_at, last_used_at,
                                           published_files_json)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '[]')
            ON CONFLICT(thread_id) DO UPDATE SET
                container_id = excluded.container_id,
                created_at = CURRENT_TIMESTAMP,
                last_used_at = CURRENT_TIMESTAMP,
                published_files_json = '[]'
        """, (thread_id, container_id))
        self.conn.commit()

    def adopt_thread_container(self, thread_id: str, container_id: str) -> Optional[str]:
        """W3: bind a container the model ALREADY ran code in — but never over a live binding.

        Compare-and-set, in one statement: the insert lands only when the thread has no binding
        at all. Adoption observes an id that arrived mid-turn, so by the time we write it a
        concurrent turn may already have bound a newer container of its own; overwriting that
        would point the next turn at a sandbox its files are not in, and would reset the newer
        binding's published-file record on the way past.

        Returns the container_id the thread is bound to AFTER the attempt — ours when we won,
        the incumbent's when we lost, None when the row could not be read back. The caller keeps
        using the id it observed either way: that is where this turn's files actually are.
        """
        self.conn.execute("""
            INSERT INTO thread_containers (thread_id, container_id, created_at, last_used_at,
                                           published_files_json)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '[]')
            ON CONFLICT(thread_id) DO NOTHING
        """, (thread_id, container_id))
        self.conn.commit()
        cursor = self.conn.execute(
            "SELECT container_id FROM thread_containers WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        return row["container_id"] if row else None

    def touch_thread_container(self, thread_id: str, container_id: str):
        """Mark this container as used now (keeps it inside the reuse window)."""
        self.conn.execute("""
            UPDATE thread_containers SET last_used_at = CURRENT_TIMESTAMP
            WHERE thread_id = ? AND container_id = ?
        """, (thread_id, container_id))
        self.conn.commit()

    def add_published_container_files(self, thread_id: str, container_id: str,
                                      file_ids: List[str]):
        """Durably record container file ids already handled, so they are never posted twice.

        Holds two kinds of id, and treats them identically because their effect is identical —
        "not eligible for publication": files we uploaded to Slack, and files already sitting in
        the container when a turn started (the baseline). Without this record a bot restart
        mid-conversation re-posts every earlier artifact still in the reused container.
        """
        if not file_ids:
            return
        cursor = self.conn.execute(
            "SELECT published_files_json FROM thread_containers "
            "WHERE thread_id = ? AND container_id = ?", (thread_id, container_id))
        row = cursor.fetchone()
        if not row:
            # The row was rebound to a different container while this turn ran. These ids belong
            # to a container that no longer backs this thread; writing them would corrupt the
            # new container's dedupe list.
            return
        try:
            existing = json.loads(row["published_files_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            existing = []
        merged = list(dict.fromkeys([*existing, *file_ids]))[-self._CONTAINER_PUBLISHED_CAP:]
        self.conn.execute("""
            UPDATE thread_containers SET published_files_json = ?
            WHERE thread_id = ? AND container_id = ?
        """, (json.dumps(merged), thread_id, container_id))
        self.conn.commit()

    def delete_thread_container(self, thread_id: str, container_id: Optional[str] = None):
        """Forget a container binding (it expired, or the API told us it is gone).

        `container_id` scopes the delete so a slow reaper cannot drop a binding that a live turn
        has already replaced. Omitted only where the caller genuinely means "whatever is bound".
        """
        if container_id is None:
            self.conn.execute("DELETE FROM thread_containers WHERE thread_id = ?", (thread_id,))
        else:
            self.conn.execute(
                "DELETE FROM thread_containers WHERE thread_id = ? AND container_id = ?",
                (thread_id, container_id))
        self.conn.commit()

    def get_expired_thread_containers(self, older_than_minutes: int) -> List[Dict]:
        """Rows whose container is certainly dead — for the daily reap."""
        cursor = self.conn.execute("""
            SELECT thread_id, container_id FROM thread_containers
            WHERE last_used_at <= datetime('now', ?)
        """, (f"-{int(older_than_minutes)} minutes",))
        return [dict(r) for r in cursor.fetchall()]

    # User operations

    def get_or_create_user(self, user_id: str, username: Optional[str] = None) -> Dict:
        """
        Get existing user or create new one with defaults.
        
        Args:
            user_id: User identifier
            username: Optional username
            
        Returns:
            User data dictionary
        """
        # Try to get existing user
        cursor = self.conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update last seen
            self.conn.execute("""
                UPDATE users SET last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (user_id,))
            return dict(row)
        
        # Create new user with defaults from config
        from config import BotConfig
        config = BotConfig()
        
        default_config = {
            "model": config.gpt_model,
            "temperature": config.default_temperature,
            "top_p": config.default_top_p,
            "reasoning_effort": config.default_reasoning_effort,
            "verbosity": config.default_verbosity,
            "image_model": config.image_model,
            "image_size": config.default_image_size,
            "image_quality": config.default_image_quality,
            "image_background": config.default_image_background,
            "input_fidelity": config.default_input_fidelity,
            "detail_level": config.default_detail_level
        }
        
        self.conn.execute("""
            INSERT INTO users (user_id, username, config_json)
            VALUES (?, ?, ?)
        """, (user_id, username, json.dumps(default_config)))
        
        return self.get_or_create_user(user_id, username)
    
    def update_user_config(self, user_id: str, config: Dict):
        """
        Update user configuration.
        
        Args:
            user_id: User identifier
            config: Configuration dictionary
        """
        self.conn.execute("""
            UPDATE users 
            SET config_json = ?, last_seen = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (json.dumps(config), user_id))
        
        logger.debug(f"Updated config for user {user_id}")
    
    def get_user_config(self, user_id: str) -> Optional[Dict]:
        """
        Get user configuration.
        
        Args:
            user_id: User identifier
            
        Returns:
            Configuration dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT config_json FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row and row["config_json"]:
            return json.loads(row["config_json"])
        
        return None
    
    def save_user_info(self, user_id: str, username: Optional[str] = None, real_name: Optional[str] = None,
                       email: Optional[str] = None, timezone: Optional[str] = None,
                       tz_label: Optional[str] = None, tz_offset: Optional[int] = None):
        """
        Save comprehensive user information.
        
        Args:
            user_id: User identifier
            username: Display/username
            real_name: User's real name
            email: User's email address
            timezone: Timezone string
            tz_label: Timezone label
            tz_offset: Offset in seconds from UTC
        """
        # Build update query dynamically based on provided fields
        updates: List[str] = []
        params: List[Any] = []
        
        if username is not None:
            updates.append("username = ?")
            params.append(username)
        if real_name is not None:
            updates.append("real_name = ?")
            params.append(real_name)
        
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        if timezone is not None:
            updates.append("timezone = ?")
            params.append(timezone)
        if tz_label is not None:
            updates.append("tz_label = ?")
            params.append(tz_label)
        if tz_offset is not None:
            updates.append("tz_offset = ?")
            params.append(tz_offset)
        
        if updates:
            updates.append("last_seen = CURRENT_TIMESTAMP")
            params.append(user_id)
            
            query = f"""
                UPDATE users 
                SET {', '.join(updates)}
                WHERE user_id = ?
            """
            self.conn.execute(query, params)
            
            logger.debug(f"Updated user info for {user_id}: username={username}, real_name={real_name}, tz={timezone}")
    
    def save_user_timezone(self, user_id: str, timezone: str, 
                          tz_label: Optional[str] = None, tz_offset: Optional[int] = None):
        """
        Save user timezone information (kept for compatibility).
        
        Args:
            user_id: User identifier
            timezone: Timezone string
            tz_label: Timezone label
            tz_offset: Offset in seconds from UTC
        """
        self.save_user_info(user_id, timezone=timezone, tz_label=tz_label, tz_offset=tz_offset)
    
    def get_user_timezone(self, user_id: str) -> Optional[Tuple[str, str, int]]:
        """
        Get user timezone information.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (timezone, tz_label, tz_offset) or None
        """
        cursor = self.conn.execute(
            "SELECT timezone, tz_label, tz_offset FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return (row["timezone"], row["tz_label"], row["tz_offset"])
        
        return None
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive user information including email.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user info or None
        """
        cursor = self.conn.execute(
            "SELECT username, real_name, email, timezone, tz_label, tz_offset FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                'username': row["username"],
                'real_name': row["real_name"],
                'email': row["email"],
                'timezone': row["timezone"],
                'tz_label': row["tz_label"],
                'tz_offset': row["tz_offset"]
            }
        
        return None
    
    # Config hierarchy resolution
    
    def get_effective_config(self, thread_id: str, user_id: str) -> Dict:
        """
        Get effective configuration merging BotConfig -> User -> Thread.
        
        Args:
            thread_id: Thread identifier
            user_id: User identifier
            
        Returns:
            Merged configuration dictionary
        """
        from config import BotConfig
        bot_config = BotConfig()
        
        # Start with bot defaults
        effective = {
            "model": bot_config.gpt_model,
            "temperature": bot_config.default_temperature,
            "top_p": bot_config.default_top_p,
            "reasoning_effort": bot_config.default_reasoning_effort,
            "verbosity": bot_config.default_verbosity,
            "image_model": bot_config.image_model,
            "image_size": bot_config.default_image_size,
            "image_quality": bot_config.default_image_quality,
            "image_background": bot_config.default_image_background,
            "input_fidelity": bot_config.default_input_fidelity,
            "detail_level": bot_config.default_detail_level
        }
        
        # Apply user config
        user_config = self.get_user_config(user_id)
        if user_config:
            effective.update(user_config)
        
        # Apply thread config
        thread_config = self.get_thread_config(thread_id)
        if thread_config:
            effective.update(thread_config)
        
        return effective
    
    # Maintenance operations
    
    # =============================
    # MODAL SESSION MANAGEMENT
    # =============================

    def create_modal_session(self, session_id: str, user_id: str, state: Dict, modal_type: str = 'settings') -> bool:
        """
        Create a new modal session.

        Args:
            session_id: Unique session identifier (UUID)
            user_id: User ID who owns this session
            state: Initial state dictionary
            modal_type: Type of modal (default 'settings')

        Returns:
            True if created successfully
        """
        try:
            self.conn.execute("""
                INSERT INTO modal_sessions (session_id, user_id, modal_type, state)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, modal_type, json.dumps(state)))
            self.log_debug(f"Created modal session {session_id} for user {user_id}")
            return True
        except Exception as e:
            self.log_error(f"Failed to create modal session: {e}")
            return False

    def get_modal_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve modal session state.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        try:
            cursor = self.conn.execute("""
                SELECT state FROM modal_sessions
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            self.log_error(f"Failed to get modal session: {e}")
            return None

    def update_modal_session(self, session_id: str, state: Dict) -> bool:
        """
        Update modal session state.

        Args:
            session_id: Session identifier
            state: Updated state dictionary

        Returns:
            True if updated successfully
        """
        try:
            cursor = self.conn.execute("""
                UPDATE modal_sessions
                SET state = ?, updated_at = strftime('%s', 'now')
                WHERE session_id = ?
            """, (json.dumps(state), session_id))

            if cursor.rowcount > 0:
                self.log_debug(f"Updated modal session {session_id}")
                return True
            return False
        except Exception as e:
            self.log_error(f"Failed to update modal session: {e}")
            return False

    def delete_modal_session(self, session_id: str) -> bool:
        """
        Delete a modal session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        try:
            cursor = self.conn.execute("""
                DELETE FROM modal_sessions
                WHERE session_id = ?
            """, (session_id,))

            if cursor.rowcount > 0:
                self.log_debug(f"Deleted modal session {session_id}")
                return True
            return False
        except Exception as e:
            self.log_error(f"Failed to delete modal session: {e}")
            return False

    def cleanup_old_modal_sessions(self, hours: int = 24):
        """
        Clean up modal sessions older than specified hours.

        Args:
            hours: Number of hours to retain sessions (default 24)
        """
        try:
            cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())

            cursor = self.conn.execute("""
                DELETE FROM modal_sessions
                WHERE created_at < ?
            """, (cutoff,))

            if cursor.rowcount > 0:
                self.log_info(f"Cleaned up {cursor.rowcount} modal sessions older than {hours} hours")
        except Exception as e:
            self.log_error(f"Failed to cleanup modal sessions: {e}")

    # Async versions for modal sessions
    async def create_modal_session_async(self, session_id: str, user_id: str, state: Dict, modal_type: str = 'settings') -> bool:
        """Async version of create_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        INSERT INTO modal_sessions (session_id, user_id, modal_type, state)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, user_id, modal_type, json.dumps(state)))
                    await db.commit()
                    self.log_debug(f"Created modal session {session_id} for user {user_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to create modal session (async): {e}")
                    return False

    async def get_modal_session_async(self, session_id: str) -> Optional[Dict]:
        """Async version of get_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    async with db.execute("""
                        SELECT state FROM modal_sessions
                        WHERE session_id = ?
                    """, (session_id,)) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return json.loads(row[0])
                    return None
                except Exception as e:
                    self.log_error(f"Failed to get modal session (async): {e}")
                    return None

    async def update_modal_session_async(self, session_id: str, state: Dict) -> bool:
        """Async version of update_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        UPDATE modal_sessions
                        SET state = ?, updated_at = strftime('%s', 'now')
                        WHERE session_id = ?
                    """, (json.dumps(state), session_id))
                    await db.commit()
                    self.log_debug(f"Updated modal session {session_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to update modal session (async): {e}")
                    return False

    async def delete_modal_session_async(self, session_id: str) -> bool:
        """Async version of delete_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        DELETE FROM modal_sessions
                        WHERE session_id = ?
                    """, (session_id,))
                    await db.commit()
                    self.log_debug(f"Deleted modal session {session_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to delete modal session (async): {e}")
                    return False

    def backup_database(self, tag: Optional[str] = None):
        """Create timestamped backup of database.

        Args:
            tag: Optional label inserted before the timestamp (e.g. a migration name),
                 producing {platform}_{tag}_{timestamp}.db. Kept before the timestamp so
                 cleanup_old_backups' date parsing (last two underscore parts) still works.
        """
        # Checkpoint WAL file before backup
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{tag}_" if tag else ""
        backup_path = f"{self.db_dir}/backups/{self.platform}_{label}{timestamp}.db"
        
        # Use SQLite backup API
        backup_conn = sqlite3.connect(backup_path)
        with backup_conn:
            self.conn.backup(backup_conn)
        backup_conn.close()
        
        logger.info(f"Created backup: {backup_path}")
        
        # Clean up old backups
        self.cleanup_old_backups()
    
    def cleanup_old_backups(self):
        """Remove SCHEDULED backups older than 7 days.

        Only untagged nightly backups ({platform}_{date}_{time}.db) are pruned.
        Tagged backups ({platform}_{tag}_{date}_{time}.db — pre-v3-upgrade and the
        two migration drops) are an operator's only rollback path out of an
        irreversible upgrade, and retention must never eat them: the nightly backup
        calls this on every run, so a 7-day sweep would delete the pre-upgrade
        snapshot exactly one week after the upgrade. They are removed by hand.
        """
        cutoff = datetime.now() - timedelta(days=7)
        # Untagged shape only: platform_YYYYMMDD_HHMMSS.db — anything with an extra
        # segment carries a tag and is kept.
        scheduled = re.compile(rf"^{re.escape(self.platform)}_(\d{{8}})_(\d{{6}})\.db$")

        for filename in os.listdir(f"{self.db_dir}/backups"):
            match = scheduled.match(filename)
            if not match:
                continue
            try:
                file_date = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
                if file_date < cutoff:
                    os.remove(f"{self.db_dir}/backups/{filename}")
                    logger.info(f"Removed old backup: {filename}")

            except Exception as e:
                logger.warning(f"Error processing backup file {filename}: {e}")
    
    def cleanup_old_threads(self):
        """Remove threads older than 3 months."""
        cutoff = datetime.now() - timedelta(days=90)
        
        cursor = self.conn.execute("""
            DELETE FROM threads 
            WHERE last_activity < ?
        """, (cutoff,))
        
        if cursor.rowcount > 0:
            logger.info(f"Cleaned up {cursor.rowcount} threads older than 3 months")
    
    # =============================
    # ASYNC VERSIONS OF CORE METHODS
    # =============================

    async def _get_async_connection(self):
        """Get an async database connection with semaphore control."""
        await self._async_db_semaphore.acquire()
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )
            conn.row_factory = aiosqlite.Row  # Enable column access by name

            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

            return conn
        finally:
            self._async_db_semaphore.release()

    async def get_thread_summary_async(self, thread_id: str) -> Optional[Dict]:
        """Async version of get_thread_summary."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM thread_summaries WHERE thread_id = ?", (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                summary = dict(row)
                summary["refs"] = json.loads(summary["refs_json"]) if summary.get("refs_json") else []
                # NULL (legacy row) -> None = "unknown" (rebuild fails safe); "[]" -> verified
                # empty (fast tail-only path). F3.
                _pts = summary.get("preserved_ts_json")
                summary["preserved_ts"] = json.loads(_pts) if _pts is not None else None
                return summary

    async def save_thread_summary_async(self, thread_id: str, summary_text: str, boundary_ts: str,
                                        refs: Optional[List[Dict]] = None,
                                        preserved_ts: Optional[List[str]] = None):
        """Async version of save_thread_summary (upsert, rolling)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                INSERT INTO thread_summaries
                    (thread_id, summary_text, boundary_ts, refs_json, preserved_ts_json, updated_ts)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    boundary_ts = excluded.boundary_ts,
                    refs_json = excluded.refs_json,
                    preserved_ts_json = excluded.preserved_ts_json,
                    updated_ts = CURRENT_TIMESTAMP
            """, (thread_id, summary_text, boundary_ts,
                  json.dumps(refs) if refs else None,
                  # `is not None`: an explicit [] persists as "[]" (verified empty), distinct
                  # from NULL (legacy/unknown). F3.
                  json.dumps(preserved_ts) if preserved_ts is not None else None))
            await db.commit()
        self.log_info(f"DB: Saved thread summary for {thread_id} (boundary_ts={boundary_ts}, async)")

    # --- Track 1: per-channel "recent channel narrative" summary CRUD --------------------
    # Every query is strictly WHERE channel_id = ? — NO workspace-scope fallback, so one
    # channel's narrative can never be read/written under another's id (scope-guard boundary).

    async def get_channel_summary_async(self, channel_id: str) -> Optional[Dict]:
        """The cached channel narrative row for one channel, or None. Per-channel scope only."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT channel_id, summary_text, built_through_ts, source_message_count, "
                "generated_at, invalidated_at FROM channel_summaries WHERE channel_id = ?",
                (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def save_channel_summary_async(self, channel_id: str, summary_text: str,
                                         built_through_ts: str, source_message_count: int):
        """Upsert a freshly REBUILT channel narrative (one row per channel). Because a rebuild
        is always from a fresh snapshot, this CLEARS invalidated_at and bumps generated_at —
        a saved summary is by definition current and injectable again.

        CONDITIONAL on the channel not being explicitly opted out: the INSERT...SELECT writes
        nothing when channel_settings.ambient_memory = 0, so a build that raced a settings change
        to ambient_memory=False can never resurrect a summary for an opted-out channel. Returns
        True when a row was written/updated."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cur = await db.execute("""
                INSERT INTO channel_summaries
                    (channel_id, summary_text, built_through_ts, source_message_count,
                     generated_at, invalidated_at)
                SELECT ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL
                WHERE NOT EXISTS (
                    SELECT 1 FROM channel_settings
                    WHERE channel_id = ? AND ambient_memory = 0
                )
                ON CONFLICT(channel_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    built_through_ts = excluded.built_through_ts,
                    source_message_count = excluded.source_message_count,
                    generated_at = CURRENT_TIMESTAMP,
                    invalidated_at = NULL
            """, (channel_id, summary_text, str(built_through_ts), int(source_message_count),
                  channel_id))
            await db.commit()
            wrote = cur.rowcount != 0
        if wrote:
            self.log_info(
                f"DB: Saved channel summary for {channel_id} "
                f"(built_through_ts={built_through_ts}, msgs={source_message_count}, async)")
        else:
            self.log_debug(
                f"DB: Skipped channel summary save for {channel_id} — channel opted out of ambient memory")
        return wrote

    async def invalidate_channel_summary_async(self, channel_id: str):
        """Mark the cache invalid (an in-window edit/delete touched a summarized message) so
        both agents STOP injecting it until a background rebuild clears the flag. No-op when no
        row exists. Per-channel scope only."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "UPDATE channel_summaries SET invalidated_at = CURRENT_TIMESTAMP "
                "WHERE channel_id = ?", (channel_id,))
            await db.commit()

    async def delete_channel_summary_async(self, channel_id: str):
        """Delete a channel's cached narrative (per-channel ambient-memory opt-out / cleanup).
        Per-channel scope only."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("DELETE FROM channel_summaries WHERE channel_id = ?", (channel_id,))
            await db.commit()

    # --- Track 4: channel join intro lease + lifecycle ----------------------------------
    # Idempotency for the one-time join intro. Every query is WHERE channel_id = ? (per-channel).

    async def try_acquire_channel_intro_lease_async(self, channel_id: str,
                                                    event_id: Optional[str] = None) -> Dict[str, Any]:
        """Atomically claim the right to post THIS channel's join intro. Returns a dict:
          {"acquired": bool, "status": str|None, "prior_status": str|None, "owner_token": str|None}.

        We win (acquired=True, with a freshly minted owner_token) when there was no row, or the
        previous attempt is 'failed' and we re-claim it. `prior_status` reports what existed BEFORE
        this claim (None = truly fresh, nothing to double-post; 'failed' = re-acquired, a prior
        attempt existed and MAY have posted — so the caller must reconcile before reposting). We
        lose (acquired=False) when a row already sits in 'pending' or 'posted' — and `status`
        reports which, so the caller can tell "another attempt owns it / may have crashed mid-post"
        (pending → RECONCILE for our marker) from "already done" (posted → just skip).

        The claim is a SINGLE statement (INSERT ... SELECT WHERE NOT EXISTS ... ON CONFLICT DO
        UPDATE) so there is no check-then-act race: a concurrent second caller either sees our
        fresh 'pending' row (NOT EXISTS fails → rowcount 0 → loses) or wrote first (we lose). The
        prior-status read is a same-connection SELECT before the write — informational only, so its
        (harmless) staleness under contention never affects the atomic claim."""
        owner_token = uuid.uuid4().hex
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT status FROM channel_introductions WHERE channel_id = ?", (channel_id,)
            ) as c0:
                prior = await c0.fetchone()
            prior_status = prior["status"] if prior else None
            cur = await db.execute("""
                INSERT INTO channel_introductions (channel_id, status, event_id, owner_token, updated_at)
                SELECT ?, 'pending', ?, ?, CURRENT_TIMESTAMP
                WHERE NOT EXISTS (
                    SELECT 1 FROM channel_introductions
                    WHERE channel_id = ? AND status IN ('pending', 'posted')
                )
                ON CONFLICT(channel_id) DO UPDATE SET
                    status = 'pending',
                    event_id = excluded.event_id,
                    owner_token = excluded.owner_token,
                    updated_at = CURRENT_TIMESTAMP
            """, (channel_id, event_id, owner_token, channel_id))
            acquired = cur.rowcount != 0
            await db.commit()
            if acquired:
                return {"acquired": True, "status": "pending",
                        "prior_status": prior_status, "owner_token": owner_token}
            async with db.execute(
                "SELECT status FROM channel_introductions WHERE channel_id = ?", (channel_id,)
            ) as c2:
                row = await c2.fetchone()
            return {"acquired": False,
                    "status": (row["status"] if row else None),
                    "prior_status": prior_status, "owner_token": None}

    async def mark_channel_intro_posted_async(self, channel_id: str,
                                             intro_ts: Optional[str]) -> None:
        """Record that the intro was posted (or reconciled from history): status 'posted' + the
        message ts. Idempotent — a repeat call just refreshes intro_ts/updated_at. Safe to call
        without owning the lease: finding our marker in history is itself proof it was posted."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO channel_introductions (channel_id, status, intro_ts, updated_at)
                VALUES (?, 'posted', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(channel_id) DO UPDATE SET
                    status = 'posted',
                    intro_ts = excluded.intro_ts,
                    updated_at = CURRENT_TIMESTAMP
            """, (channel_id, str(intro_ts) if intro_ts is not None else None))
            await db.commit()

    async def mark_channel_intro_failed_async(self, channel_id: str,
                                             owner_token: Optional[str] = None) -> None:
        """Flag a failed attempt so a genuine later refire may retry — but ONLY the 'pending' row
        THIS attempt owns (WHERE status='pending' AND owner_token=?). This never clobbers a
        'posted' row (a late error can't reopen a sent intro) and never steals a CONCURRENT
        attempt's live lease (its token differs). Called only by a task that actually acquired the
        lease; a missing token matches nothing (no-op), failing safe."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "UPDATE channel_introductions SET status = 'failed', updated_at = CURRENT_TIMESTAMP "
                "WHERE channel_id = ? AND status = 'pending' AND owner_token IS ?",
                (channel_id, owner_token))
            await db.commit()

    async def get_channel_intro_async(self, channel_id: str) -> Optional[Dict]:
        """The channel's intro lifecycle row (status/intro_ts/…), or None. Per-channel scope."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT channel_id, status, prepared_text, event_id, intro_ts, owner_token, "
                "updated_at FROM channel_introductions WHERE channel_id = ?", (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def claim_channel_onboarding_nudge_async(self, user_id: str) -> bool:
        """Atomically claim the one-time channel-onboarding settings DM for this user.

        Returns True EXACTLY ONCE per user — the caller that wins the claim sends the silent DM.
        Concurrent @mentions and every later interaction (including after a restart) get False, so
        the newcomer is never re-DM'd. `INSERT OR IGNORE` + rowcount is the atomic test-and-set;
        there is no check-then-act race. If the send then fails, the caller rolls the claim back via
        clear_channel_onboarding_nudge_async so a later interaction can retry."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cur = await db.execute(
                "INSERT OR IGNORE INTO channel_onboarding_nudges (slack_user_id) VALUES (?)",
                (user_id,))
            await db.commit()
            return cur.rowcount == 1

    async def clear_channel_onboarding_nudge_async(self, user_id: str) -> None:
        """Release a claimed nudge (used only when the DM send failed) so a future channel
        interaction can retry the one-time settings DM. No-op if no row exists."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "DELETE FROM channel_onboarding_nudges WHERE slack_user_id = ?", (user_id,))
            await db.commit()

    async def update_thread_activity_async(self, thread_id: str):
        """
        Async version of update_thread_activity.

        Args:
            thread_id: Thread identifier
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE threads
                SET last_activity = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (thread_id,))

            await db.commit()

    async def save_image_metadata_async(self, thread_id: str, url: str, image_type: str,
                                       prompt: Optional[str] = None, analysis: Optional[str] = None,
                                       original_analysis: Optional[str] = None, metadata: Optional[Dict] = None,
                                       message_ts: Optional[str] = None):
        """
        Async version of save_image_metadata (NO base64 data).

        Args:
            thread_id: Thread identifier
            url: Image URL
            image_type: Type of image (uploaded/generated/edited)
            prompt: Full generation/edit prompt
            analysis: Full vision analysis
            original_analysis: For edited images, the pre-edit analysis
            metadata: Additional metadata
            message_ts: Message timestamp to link image to specific message
        """
        self.log_debug(f"DB: Async saving image - thread={thread_id}, url={url[:100]}, "
                      f"type={image_type}, has_analysis={bool(analysis)}, "
                      f"analysis_len={len(analysis) if analysis else 0}, "
                      f"prompt_len={len(prompt) if prompt else 0}")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")

                # Merge-preserving upsert (F1): see save_image_metadata. A later empty
                # write (rebuild with empty caption, ledger upsert) must not erase the
                # prompt/analysis/type/generation_id an earlier write recorded.
                await db.execute("""
                    INSERT INTO images
                    (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        thread_id = excluded.thread_id,
                        image_type = COALESCE(NULLIF(images.image_type, ''), excluded.image_type),
                        prompt = COALESCE(NULLIF(images.prompt, ''), excluded.prompt),
                        analysis = COALESCE(NULLIF(images.analysis, ''), excluded.analysis),
                        original_analysis = COALESCE(NULLIF(images.original_analysis, ''), excluded.original_analysis),
                        metadata_json = COALESCE(NULLIF(images.metadata_json, ''), excluded.metadata_json),
                        message_ts = COALESCE(excluded.message_ts, images.message_ts)
                """, (thread_id, url, image_type, prompt, analysis, original_analysis,
                      json.dumps(metadata) if metadata else None, message_ts))

                await db.commit()

                self.log_info(f"DB: Successfully saved image metadata for {url[:50]}... in thread {thread_id}")

        except Exception as e:
            self.log_error(f"DB: Failed to save image metadata async - {e}", exc_info=True)
            raise

    @staticmethod
    def _merge_tool_provenance(existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge two provenance lists (existing first, then new), ORDER-preserving.

        Two entry classes coexist: F7 used-tools entries (name + gist) and F12 result-digest
        entries (name + result_digest). They never collapse into each other.

        Used-tools: a tool that genuinely ran more than once (same name, different gists) is
        kept as multiple entries; only EXACT duplicates (same name AND gist) are deduped; an
        empty-gist placeholder is UPGRADED in place by a later non-empty gist for the same
        tool. Capped at config.tool_provenance_max_entries (F14; default 20, was 8) so the
        persisted row honors the same budget build_provenance applies.

        Result-digests (F12): deduped by (name, digest) so re-persist is idempotent but two
        distinct outputs from the same server are both kept. Already char-bounded at capture,
        so NOT subject to the used-tools entry cap; appended AFTER the used-tools entries
        (matches the pinned [used tools:] → [tool results:] render order). Old rows (no
        result_digest) merge exactly as before."""
        used: List[Dict] = []
        results: List[Dict] = []
        seen_results = set()
        for entry in list(existing or []) + list(new or []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool_name")
            if not name:
                continue
            digest = entry.get("result_digest")
            if digest:
                key = (name, digest)
                if key in seen_results:
                    continue  # exact (name, digest) duplicate — idempotent re-persist
                seen_results.add(key)
                results.append({"tool_name": name, "result_digest": digest})
                continue
            gist = entry.get("gist") or ""
            if any(m["tool_name"] == name and m["gist"] == gist for m in used):
                continue  # exact duplicate — dedupe
            if gist.strip():
                placeholder = next(
                    (m for m in used if m["tool_name"] == name and not m["gist"].strip()), None)
                if placeholder is not None:
                    placeholder["gist"] = gist  # upgrade empty placeholder in place
                    continue
            elif any(m["tool_name"] == name and m["gist"].strip() for m in used):
                continue  # empty gist already covered by a non-empty entry for this tool
            used.append({"tool_name": name, "gist": gist})
        from config import config
        return used[:int(getattr(config, "tool_provenance_max_entries", 20))] + results

    async def save_tool_usage_async(self, channel_id: str, message_ts: str,
                                    thread_key: str, tools: List[Dict]) -> None:
        """Persist a reply's tool-use provenance (F7), keyed by channel+ts.

        `tools` is the compact [{"tool_name","gist"}] record (names + arg gists), with
        optional F12 [{"tool_name","result_digest"}] MCP result-memory entries appended.
        Idempotent on (channel_id, message_ts): a re-persist MERGES with the existing row
        (union by tool_name, preferring a non-empty gist) rather than last-write-wins, so a
        second pass can't drop tools recorded by the first. Best-effort — the caller wraps
        this so a DB failure never blocks the reply."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                # BEGIN IMMEDIATE takes the write lock up front so the read-modify-write
                # (SELECT existing → merge → UPSERT) is atomic against a concurrent
                # persist for the same reply — otherwise two passes could each read the
                # old row and the second would clobber the first's merged tools.
                await db.execute("BEGIN IMMEDIATE")
                existing: List[Dict] = []
                async with db.execute(
                    "SELECT tools_json FROM message_tool_usage WHERE channel_id = ? AND message_ts = ?",
                    (channel_id, message_ts)
                ) as cur:
                    row = await cur.fetchone()
                if row and row[0]:
                    try:
                        parsed = json.loads(row[0])
                        if isinstance(parsed, list):
                            existing = parsed
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing = []
                merged = self._merge_tool_provenance(existing, tools)
                await db.execute("""
                    INSERT INTO message_tool_usage
                        (channel_id, message_ts, thread_key, tools_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel_id, message_ts) DO UPDATE SET
                        thread_key = excluded.thread_key,
                        tools_json = excluded.tools_json
                """, (channel_id, message_ts, thread_key, json.dumps(merged)))
                await db.commit()
        except Exception as e:
            self.log_debug(f"DB: save_tool_usage_async failed (non-fatal): {e}")

    async def get_thread_tool_usage_async(self, thread_key: str) -> Dict[str, List[Dict]]:
        """Batch-fetch a thread's tool-use provenance for rebuild reinjection (F7).

        Returns {message_ts: [{"tool_name","gist"}, …]}. Empty on any error so a missing
        table / read failure degrades to no annotations rather than breaking the rebuild."""
        result: Dict[str, List[Dict]] = {}
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                async with db.execute(
                    "SELECT message_ts, tools_json FROM message_tool_usage WHERE thread_key = ?",
                    (thread_key,)
                ) as cursor:
                    async for row in cursor:
                        try:
                            parsed = json.loads(row["tools_json"])
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if isinstance(parsed, list):
                            result[row["message_ts"]] = parsed
        except Exception as e:
            self.log_debug(f"DB: get_thread_tool_usage_async failed (non-fatal): {e}")
        return result

    async def save_thread_config_async(self, thread_id: str, config: Dict):
        """
        Async version of save_thread_config.

        Args:
            thread_id: Thread identifier
            config: Configuration dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE threads
                SET config_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (json.dumps(config), thread_id))

            await db.commit()
            logger.debug(f"Saved config for thread {thread_id} (async)")

    async def get_channel_settings_async(self, channel_id: str) -> Optional[Dict]:
        """Async version of get_channel_settings."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            return store.channel_settings(key)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT response_mode, reply_in_channel, participation_level, "
                "snoozed_until, muted_threads, model, reasoning_effort, verbosity, "
                "ambient_memory, enable_web_search, enable_mcp, image_model, "
                "updated_ts, updated_by "
                "FROM channel_settings WHERE channel_id = ?",
                (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "response_mode": row["response_mode"],
                    # F51 opt-out: NULL → None (inherit config.enable_ambient_memory at read time).
                    "ambient_memory": (None if row["ambient_memory"] is None
                                       else bool(row["ambient_memory"])),
                    # RAW, deliberately — see the sync twin. A legacy pre-CHECK `2` must reach
                    # the capability resolver as `2` so it can be refused.
                    "enable_web_search": row["enable_web_search"],
                    "enable_mcp": row["enable_mcp"],
                    "image_model": row["image_model"],
                    # NULL stays None (inherit → config.reply_in_channel_default at read time).
                    "reply_in_channel": (None if row["reply_in_channel"] is None
                                         else bool(row["reply_in_channel"])),
                    "participation_level": row["participation_level"],
                    "snoozed_until": row["snoozed_until"],
                    "muted_threads": _decode_muted_threads(row["muted_threads"]),
                    "model": row["model"],
                    "reasoning_effort": row["reasoning_effort"],
                    "verbosity": row["verbosity"],
                    "updated_ts": row["updated_ts"],
                    "updated_by": row["updated_by"],
                }

    async def set_channel_settings_async(self, channel_id: str, response_mode=_UNSET,
                                         reply_in_channel=_UNSET,
                                         participation_level=_UNSET, snoozed_until=_UNSET,
                                         muted_threads=_UNSET,
                                         model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                                         ambient_memory=_UNSET,
                                         enable_web_search=_UNSET, enable_mcp=_UNSET,
                                         image_model=_UNSET,
                                         updated_by: Optional[str] = None):
        """Async version of set_channel_settings (Phase F adds participation_level/snoozed_until).

        Atomic partial write — ONLY provided fields are written, omitted fields preserved (no
        read-modify-write of the whole row, so no race with a concurrent modal save). Explicit None
        CLEARS to NULL (inherit); updated_ts/updated_by bump only on a STRUCTURAL change.
        muted_threads is a deprecated inert JSON column (nothing reads it — the per-thread mute
        mechanism was removed).
        """
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            store.set_channel_settings(key, **_epoch_settings_fields(
                response_mode=response_mode, reply_in_channel=reply_in_channel,
                participation_level=participation_level, snoozed_until=snoozed_until,
                muted_threads=muted_threads, model=model, reasoning_effort=reasoning_effort,
                verbosity=verbosity, ambient_memory=ambient_memory,
                enable_web_search=enable_web_search, enable_mcp=enable_mcp,
                image_model=image_model, updated_by=updated_by))
            return
        built = _build_channel_settings_write(
            channel_id, response_mode=response_mode,
            reply_in_channel=reply_in_channel, participation_level=participation_level,
            snoozed_until=snoozed_until, muted_threads=muted_threads, model=model,
            reasoning_effort=reasoning_effort, verbosity=verbosity,
            ambient_memory=ambient_memory, enable_web_search=enable_web_search,
            enable_mcp=enable_mcp, image_model=image_model, updated_by=updated_by)
        if built is None:
            return
        sql, params = built
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(sql, params)
            # Track 1: turning ambient_memory OFF purges the derived channel narrative in the SAME
            # transaction, so an in-flight summary build can't leave a row behind for a channel that
            # just opted out (the summary upsert is likewise blocked while ambient_memory = 0).
            if ambient_memory is False:
                await db.execute(
                    "DELETE FROM channel_summaries WHERE channel_id = ?", (channel_id,))
            await db.commit()
            logger.debug(f"Saved channel_settings for {channel_id} (async)")

    async def set_channel_settings_and_policy_async(self, channel_id: str,
                                                    policy: Optional[str],
                                                    author: Optional[str] = None,
                                                    **settings) -> None:
        """Write structural channel settings AND the standing policy as ONE transaction.

        A single instruction — "only speak up about deploys, and keep it to threads" — is one
        decision, and half of it landing is a channel configured in a way nobody asked for. The
        two writes therefore share a transaction rather than being two calls that happen to run
        next to each other.

        `policy` follows set_channel_policy_async: text REPLACES the whole policy, blank DELETES
        it. `settings` are the same keyword fields set_channel_settings_async takes; passing none
        of them writes only the policy.
        """
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            store.set_channel_settings(key, **_epoch_settings_fields(updated_by=author,
                                                                     **settings))
            store.set_steering(key, policy)
            return
        built = _build_channel_settings_write(channel_id, updated_by=author, **settings)
        text = (policy or "").strip()
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                if built is not None:
                    sql, params = built
                    await db.execute(sql, params)
                if not text:
                    await db.execute(
                        "DELETE FROM channel_memory WHERE scope = 'policy' AND channel_id = ?",
                        (channel_id,))
                else:
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'policy', ?, ?) "
                        "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                        "DO UPDATE SET content = excluded.content, author = excluded.author, "
                        "              updated_ts = CURRENT_TIMESTAMP",
                        (channel_id, text, author))
                await db.execute("COMMIT")
            except Exception:
                await db.execute("ROLLBACK")
                raise
        logger.debug(f"Saved channel_settings + policy for {channel_id} (async)")

    # --- Per-channel memory (Phase 9), async variants ---
    async def get_channel_memory_async(self, channel_id: str) -> List[Dict]:
        """Async version of get_channel_memory (channel-scope for this channel + shared workspace)."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            return store.memory(key)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
                "FROM channel_memory WHERE (scope = 'channel' AND channel_id = ?) OR scope = 'workspace' "
                "ORDER BY updated_ts ASC",
                (channel_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_channel_policy_async(self, channel_id: str) -> Optional[Dict]:
        """Async version of get_channel_policy. Never returns facts."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            return store.steering_row(key)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
                "FROM channel_memory WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def set_channel_policy_async(self, channel_id: str, content: Optional[str],
                                       author: Optional[str] = None) -> None:
        """Async version of set_channel_policy: REPLACE the whole policy, or DELETE when blank.

        Replace, never append or patch — a standing policy is a single statement of how the bot
        behaves here, and merging a new one into an old one accumulates contradictions nobody
        chose. Blank means the operator cleared it, which is a deletion: an empty row would
        render an empty instructions heading."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            store.set_steering(key, content)
            return
        text = (content or "").strip()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            if not text:
                await db.execute(
                    "DELETE FROM channel_memory WHERE scope = 'policy' AND channel_id = ?",
                    (channel_id,))
            else:
                # The partial unique index turns this into an upsert rather than a race.
                await db.execute(
                    "INSERT INTO channel_memory (channel_id, scope, content, author) "
                    "VALUES (?, 'policy', ?, ?) "
                    "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                    "DO UPDATE SET content = excluded.content, author = excluded.author, "
                    "              updated_ts = CURRENT_TIMESTAMP",
                    (channel_id, text, author))
            await db.commit()

    async def set_channel_policy_if_unchanged_async(self, channel_id: str,
                                                    expected_hash: Optional[str],
                                                    content: Optional[str],
                                                    author: Optional[str] = None) -> bool:
        """Compare-and-swap the policy: replace it ONLY if it still hashes to `expected_hash`.

        Returns True when the write landed, False when someone else changed the policy first.

        The comparison and the write are ONE transaction on purpose. The settings modal replaces
        the policy wholesale, so a save from a modal opened before someone else edited it would
        revert their rule — and an emptied box would revert it to nothing at all. Reading the
        hash in one statement and writing in another leaves exactly that gap open, just narrower:
        a concurrent save landing between the two is silently overwritten by whichever modal
        submits second. `expected_hash` is memory_content_hash of the policy the writer was
        shown, or "" / None when it was shown no policy at all.
        """
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            # The CAS runs against the OVERLAY's hash, not production's: a fenced modal save must
            # keep the concurrency guard rather than fail against a hash the real row still holds.
            store, key = overlay
            return store.set_steering_if_unchanged(key, content, expected_hash=expected_hash,
                                                   hasher=memory_content_hash)
        expected = expected_hash or ""
        text = (content or "").strip()
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            # IMMEDIATE, not deferred: the write lock is taken before the read, so a concurrent
            # writer serializes behind this rather than racing inside it.
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT content FROM channel_memory "
                    "WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                    (channel_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                stored = ((row["content"] if row else "") or "").strip()
                current_hash = memory_content_hash(stored) if stored else ""
                if current_hash != expected:
                    await db.execute("ROLLBACK")
                    return False
                if stored == text:
                    # Nothing changed. Writing anyway would re-attribute the policy to whoever
                    # opened the modal and clicked Save without touching the box — the same
                    # "field supplied is not value changed" rule the channel-settings writer
                    # follows for updated_by.
                    await db.execute("COMMIT")
                    return True
                if not text:
                    await db.execute(
                        "DELETE FROM channel_memory WHERE scope = 'policy' AND channel_id = ?",
                        (channel_id,))
                else:
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'policy', ?, ?) "
                        "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                        "DO UPDATE SET content = excluded.content, author = excluded.author, "
                        "              updated_ts = CURRENT_TIMESTAMP",
                        (channel_id, text, author))
                await db.execute("COMMIT")
                return True
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    async def merge_channel_policy_async(self, channel_id: str, content: Optional[str],
                                         author: Optional[str] = None) -> Optional[str]:
        """Fold `content` into the stored policy per ``merged_policy_text``, in ONE transaction.

        For the writers that cannot see the current policy and therefore must not replace it: the
        directives migration's own case, and a modal opened before the policy field existed. A
        blank `content` is NOT a clear — a writer that cannot see the policy cannot mean "delete
        it". Returns the stored text afterwards (None if there was and is nothing).
        """
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT content FROM channel_memory "
                    "WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                    (channel_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                stored = ((row["content"] if row else "") or "").strip()
                merged = merged_policy_text(stored, content or "")
                if merged and merged != stored:
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'policy', ?, ?) "
                        "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                        "DO UPDATE SET content = excluded.content, author = excluded.author, "
                        "              updated_ts = CURRENT_TIMESTAMP",
                        (channel_id, merged, author))
                await db.execute("COMMIT")
                return merged
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    async def update_channel_fact_async(self, memory_id: int, content: str) -> bool:
        """Revise an ORDINARY channel fact. Returns True when a row actually changed.

        The scope and author live in the WHERE clause, not in a caller's check, because the
        callers that most need the guarantee are the ones least able to enforce it: the fallback
        extractor hands this an id a utility model produced, and an id it never saw is either a
        hallucination or stale. A policy row or one of the gate's preference markers matches
        nothing here, so the write is a no-op rather than a silent overwrite of steering.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            from message_processor.channel_steering import PREF_AUTHOR_PREFIX
            cursor = await db.execute(
                "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP "
                "WHERE id = ? AND scope = 'channel' "
                "AND (author IS NULL OR author NOT LIKE ? || '%')",
                (content, memory_id, PREF_AUTHOR_PREFIX))
            await db.commit()
            return (cursor.rowcount or 0) > 0

    async def migrate_channel_directives_to_policy_async(self) -> tuple:
        """Move every nonempty `channel_settings.directives` into its channel's policy row.

        IDEMPOTENT, and safe to interrupt. Each channel is one transaction: the policy row is
        written FIRST and the directives column nulled only after that write succeeds, so a
        failure anywhere leaves the operator's rules exactly where they were. A rerun finds no
        nonempty directives and does nothing.

        A channel that somehow has BOTH keeps both texts as separate lines in a stable order,
        deduplicated only on exact equality after whitespace normalization. Guessing which one
        the operator meant is not this migration's call, and silently dropping either is how a
        live rule disappears.

        Returns ``(migrated, failed)``. A nonzero `failed` is NOT survivable at startup: the
        legacy readers are gone, so a channel whose directives are still sitting in the column
        has an operator rule that nothing will obey until a later run happens to succeed. The
        caller aborts on it — see main.ChatBotV2.initialize."""
        migrated = 0
        failed = 0
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT channel_id, directives FROM channel_settings "
                "WHERE directives IS NOT NULL AND TRIM(directives) != ''"
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                channel_id = row["channel_id"]
                directives = (row["directives"] or "").strip()
                if not directives:
                    continue
                try:
                    async with db.execute(
                        "SELECT content FROM channel_memory "
                        "WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                        (channel_id,)
                    ) as pol_cursor:
                        existing = await pol_cursor.fetchone()
                    current = (existing["content"] or "").strip() if existing else ""
                    content = merged_policy_text(current, directives)
                    if current and content != current:
                        logger.warning(
                            f"DB: channel {channel_id} had BOTH a policy row and legacy "
                            f"directives — preserving both; an operator should review")
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'policy', ?, 'migration:channel_directives') "
                        "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                        "DO UPDATE SET content = excluded.content, "
                        "              updated_ts = CURRENT_TIMESTAMP",
                        (channel_id, content))
                    # ONLY after the policy write lands: the column is the fallback until it isn't.
                    await db.execute(
                        "UPDATE channel_settings SET directives = NULL WHERE channel_id = ?",
                        (channel_id,))
                    await db.commit()
                    migrated += 1
                except Exception as e:  # noqa: BLE001 — every channel gets its own attempt
                    failed += 1
                    try:
                        await db.rollback()
                    except Exception:
                        pass
                    logger.error(
                        f"DB: directives→policy migration failed for {channel_id} "
                        f"({type(e).__name__}); its directives are untouched and its operator "
                        f"rule will NOT be obeyed until this succeeds")
        if migrated:
            logger.info(
                f"DB: migrated channel directives to policy rows for {migrated} channel(s)")
        return migrated, failed

    async def migrate_participation_levels_to_binary_async(self) -> tuple:
        """Collapse the retired participation levels onto the binary gate's single ON value.

        `judicious` and `active` meant the same thing the moment the gate became one bit — see
        _LEGACY_PARTICIPATION_LEVELS. A channel left on either name would be read by a gate whose
        vocabulary no longer contains it, and an unknown level falls back to the global default:
        an operator who deliberately turned this channel ON could find it inheriting `off`.

        IDEMPOTENT and safe to interrupt. One UPDATE per channel, each its own transaction, so a
        channel that cannot be written costs only itself — the same per-channel bargain the
        directives migration makes. A rerun matches no legacy names and does nothing.

        `response_mode` is deliberately LEFT ALONE. It is dual-written by the settings modal and
        its legacy mapping (auto_respond ↔ on) already reads correctly, so rewriting it would
        change nothing except the one column a rollback to the previous release still reads.

        updated_ts/updated_by are not bumped either: this is housekeeping, not an edit, and
        stamping it would make the migration look like the last human to touch the channel's
        settings — the same rule the channel_settings writer follows for non-structural columns.

        Returns ``(migrated, failed)``, counting CHANNELS, so the startup caller can abort on
        `failed` exactly as it does for the directives migration.
        """
        placeholders = ", ".join("?" for _ in _LEGACY_PARTICIPATION_LEVELS)
        migrated = 0
        failed = 0
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            # LOWER(TRIM(...)) on both sides: the modal only ever wrote these lowercase, but a
            # value hand-edited into the DB is exactly the kind of row that would otherwise be
            # left behind to fall back to the global default.
            async with db.execute(
                f"SELECT channel_id, participation_level FROM channel_settings "
                f"WHERE participation_level IS NOT NULL "
                f"AND LOWER(TRIM(participation_level)) IN ({placeholders})",
                _LEGACY_PARTICIPATION_LEVELS
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                channel_id = row["channel_id"]
                try:
                    await db.execute(
                        f"UPDATE channel_settings SET participation_level = ? "
                        f"WHERE channel_id = ? "
                        f"AND LOWER(TRIM(participation_level)) IN ({placeholders})",
                        (_PARTICIPATION_LEVEL_ON, channel_id, *_LEGACY_PARTICIPATION_LEVELS))
                    await db.commit()
                    migrated += 1
                except Exception as e:  # noqa: BLE001 — every channel gets its own attempt
                    failed += 1
                    try:
                        await db.rollback()
                    except Exception:
                        pass
                    logger.error(
                        f"DB: participation-level migration failed for {channel_id} "
                        f"({type(e).__name__}); it is still on "
                        f"'{row['participation_level']}', which the binary gate cannot read and "
                        f"will treat as the global default")
        if migrated:
            logger.info(
                f"DB: collapsed legacy participation levels to "
                f"'{_PARTICIPATION_LEVEL_ON}' for {migrated} channel(s)")
        return migrated, failed

    async def migrate_participation_prefs_to_policy_async(self) -> tuple:
        """Move every legacy backoff preference row into its channel's reserved policy row.

        The rich gate recorded a "stop doing X here" verdict as a channel_memory row authored
        ``participation_engine:pref:<dimension>`` and rendered it as an instruction. That writer is
        gone with the binary gate, and so is the section that rendered those rows — so their text
        is now steering that nothing reads. It is a standing instruction a person actually gave
        ("react less in here"), which is precisely what the reserved policy row is for, so that is
        where it goes.

        COPY, VERIFY, THEN DELETE — in ONE transaction per channel. The policy row is written
        first and the legacy rows dropped only after that write lands, so an interruption anywhere
        leaves every preference exactly where it was rather than half-moved. Grouped by channel
        because a single transaction for the whole database would let one unwritable channel
        discard every other channel's preferences with it.

        Merging goes through ``merged_policy_text`` so this cannot drift from the directives
        migration: existing policy lines are kept verbatim, preference text is appended as its own
        line, and a line already present (after whitespace normalization) is not repeated. Nothing
        is ever truncated or summarized — a preference the bot has been obeying must survive the
        move intact. The rows are folded in id order, so the same database always produces the same
        policy text, which is what keeps the rendered steering block byte-stable.

        Blank preference rows are deleted without a policy write: they carry nothing to preserve,
        and leaving them behind would make a rerun report work forever.

        IDEMPOTENT. A rerun finds no ``participation_engine:pref:*`` rows and does nothing.

        Returns ``(migrated, failed)``, counting CHANNELS, so the startup caller can abort on
        `failed` exactly as it does for the directives migration. A nonzero `failed` is not
        survivable: nothing renders those rows any more, so the channel is quietly disobeying an
        instruction it was given until a later run succeeds.
        """
        # The author marker IS the identity of these rows — the same predicate the classifier
        # (channel_steering.is_pref_row) and the writer's unique index key on, so what this
        # migration moves cannot drift from what the code called a preference. Scope is
        # deliberately NOT in the predicate for the same reason.
        migrated = 0
        failed = 0
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            # An enumeration pass, and ONLY that: which channels have work. Every value this
            # migration actually merges is re-read inside the per-channel lock below, because
            # between this scan and that write a person can save the settings modal or the model
            # can call the policy tool — and a merge built on a policy read out here would quietly
            # revert theirs, having already deleted the only other copy of the text.
            async with db.execute(
                "SELECT DISTINCT channel_id FROM channel_memory "
                "WHERE author LIKE ? || '%' ORDER BY channel_id",
                (_LEGACY_PREF_AUTHOR_PREFIX,)
            ) as cursor:
                channel_ids = [row["channel_id"] for row in await cursor.fetchall()]
            for channel_id in channel_ids:
                try:
                    # IMMEDIATE: the write lock is taken BEFORE the reads, so a concurrent policy
                    # writer serializes behind this rather than racing inside it.
                    await db.execute("BEGIN IMMEDIATE")
                    async with db.execute(
                        "SELECT id, content FROM channel_memory "
                        "WHERE channel_id = ? AND author LIKE ? || '%' ORDER BY id",
                        (channel_id, _LEGACY_PREF_AUTHOR_PREFIX)
                    ) as pref_cursor:
                        pref_rows = await pref_cursor.fetchall()
                    if not pref_rows:
                        # Migrated by another process (or another run) between the scan and the
                        # lock. Nothing to do, and nothing to report as failure.
                        await db.execute("COMMIT")
                        continue
                    async with db.execute(
                        "SELECT content FROM channel_memory "
                        "WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                        (channel_id,)
                    ) as pol_cursor:
                        existing = await pol_cursor.fetchone()
                    current = (existing["content"] or "").strip() if existing else ""
                    content = current
                    # id order: deterministic, and it preserves the order the preferences were
                    # actually recorded in, which is the closest thing to the operator's own
                    # sequence that survives.
                    for pref in pref_rows:
                        content = merged_policy_text(content, pref["content"] or "") or ""
                    ids = [pref["id"] for pref in pref_rows]
                    if content and content != current:
                        # On a fresh row the migration signs it; on an existing one the author is
                        # left as it is, because the operator who wrote that policy is still its
                        # author — we only appended to it. Same shape as the directives migration.
                        await db.execute(
                            "INSERT INTO channel_memory (channel_id, scope, content, author) "
                            "VALUES (?, 'policy', ?, 'migration:participation_prefs') "
                            "ON CONFLICT (channel_id) WHERE scope = 'policy' "
                            "DO UPDATE SET content = excluded.content, "
                            "              updated_ts = CURRENT_TIMESTAMP",
                            (channel_id, content))
                        # Read the policy back before anything is dropped. The legacy rows are the
                        # only remaining copy of this text, so "the write succeeded" has to be
                        # observed, not assumed.
                        async with db.execute(
                            "SELECT content FROM channel_memory "
                            "WHERE scope = 'policy' AND channel_id = ? LIMIT 1",
                            (channel_id,)
                        ) as verify_cursor:
                            stored = await verify_cursor.fetchone()
                        if not stored or (stored["content"] or "") != content:
                            raise RuntimeError("policy row did not take the merged text")
                    # ONLY the rows just copied, and only now.
                    await db.execute(
                        f"DELETE FROM channel_memory "
                        f"WHERE id IN ({', '.join('?' for _ in ids)}) AND author LIKE ? || '%'",
                        (*ids, _LEGACY_PREF_AUTHOR_PREFIX))
                    await db.execute("COMMIT")
                    migrated += 1
                except Exception as e:  # noqa: BLE001 — every channel gets its own attempt
                    failed += 1
                    try:
                        await db.execute("ROLLBACK")
                    except Exception:
                        pass
                    logger.error(
                        f"DB: participation-preference→policy migration failed for {channel_id} "
                        f"({type(e).__name__}); its preference rows are untouched, and nothing "
                        f"renders them any more, so those instructions will NOT be obeyed until "
                        f"this succeeds")
        if migrated:
            logger.info(
                f"DB: migrated legacy participation preferences into policy rows for "
                f"{migrated} channel(s)")
        return migrated, failed

    async def add_channel_memory_async(self, channel_id: str, content: str, scope: str = "channel",
                                       author: Optional[str] = None) -> int:
        """Async version of add_channel_memory; returns the new id."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            return store.add_memory(key, content, scope=scope, author=author)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                (channel_id, scope, content, author)
            )
            await db.commit()
            return cast(int, cursor.lastrowid)

    async def update_channel_memory_async(self, memory_id: int, content: str):
        """Async version of update_channel_memory."""
        overlay = _epoch_memory_row(memory_id)
        if overlay is not None:
            store, key = overlay
            store.update_memory(key, int(memory_id), content)
            return
        if _epoch_fence_up():
            _epoch_refuse_production_write(
                await self._channel_of_memory_row(memory_id), "update_channel_memory")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP WHERE id = ?",
                (content, memory_id)
            )
            await db.commit()

    async def delete_channel_memory_async(self, memory_id: int):
        """Async version of delete_channel_memory."""
        overlay = _epoch_memory_row(memory_id)
        if overlay is not None:
            store, key = overlay
            store.delete_memory(key, int(memory_id))
            return
        if _epoch_fence_up():
            _epoch_refuse_production_write(
                await self._channel_of_memory_row(memory_id), "delete_channel_memory")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("DELETE FROM channel_memory WHERE id = ?", (memory_id,))
            await db.commit()

    async def upsert_channel_pref_memory(self, channel_id: str, marker_author: str,
                                         content: str, max_rows: Optional[int] = None
                                         ) -> Optional[int]:
        """Atomically write-or-refresh the SINGLE per-channel participation preference marker
        (author == ``marker_author``, e.g. ``participation_engine:pref:reactions``), returning
        its row id — or ``None`` when a new marker is declined because the channel is at the
        memory-row cap.

        Why a bespoke helper (participation redesign, SHOULD-FIX #8): the old
        read-all-then-insert in _apply_pref_memory raced (two concurrent "react less" verdicts
        both saw "no marker" and both INSERTed a duplicate) and an ``update:<id>`` path could
        leave the row authored by something other than the marker. This does the existence
        check, the cap check, and the write inside ONE ``BEGIN IMMEDIATE`` transaction (the
        connection is opened in autocommit so the explicit transaction is ours to control), so
        concurrent callers serialize and converge on exactly one marker row per (channel,
        dimension) — an invariant also pinned by the partial UNIQUE index
        idx_channel_memory_pref_marker. The stored/updated row's author is ALWAYS the marker.

        The cap mirrors remember_fact: at MEMORY_MAX_ROWS with no marker yet, decline rather
        than evict a human's memory. An existing marker is always refreshed (a refresh frees no
        slot and adds no row, so the cap never blocks it).
        """
        if not channel_id or not marker_author:
            return None
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            # The preference marker is POLICY — it steers how the bot behaves in the room — so it
            # is overlaid rather than written through. The spec's full design REFUSED this writer
            # instead, with an honest tool error; slim overlays it, because the per-case reset
            # already denies it the thing refusal was protecting (a case rewriting its own
            # participation settings and carrying that into the next one).
            store, key = overlay
            for row in store.memory(key):
                if row["author"] == marker_author and row["scope"] == "channel":
                    store.update_memory(key, row["id"], content)
                    return row["id"]
            return store.add_memory(key, content, scope="channel", author=marker_author)
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT id FROM channel_memory "
                    "WHERE channel_id = ? AND author = ? AND scope = 'channel' LIMIT 1",
                    (channel_id, marker_author),
                ) as cur:
                    existing = await cur.fetchone()
                if existing is not None:
                    row_id = existing[0]
                    await db.execute(
                        "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP "
                        "WHERE id = ?",
                        (content, row_id),
                    )
                    await db.execute("COMMIT")
                    return row_id
                if max_rows is not None:
                    # ORDINARY facts only. The cap exists to stop remembered facts from crowding
                    # out the prompt; steering rows are not facts, and counting the engine's own
                    # markers against it would let a channel's preferences lock out the very
                    # thing the cap protects.
                    from message_processor.channel_steering import PREF_AUTHOR_PREFIX
                    async with db.execute(
                        "SELECT COUNT(*) FROM channel_memory "
                        "WHERE scope = 'channel' AND channel_id = ? "
                        "AND (author IS NULL OR author NOT LIKE ? || '%')",
                        (channel_id, PREF_AUTHOR_PREFIX),
                    ) as cur:
                        (count,) = cast(Any, await cur.fetchone())
                    if count >= max(1, int(max_rows)):
                        await db.execute("ROLLBACK")
                        return None
                cur = await db.execute(
                    "INSERT INTO channel_memory (channel_id, scope, content, author) "
                    "VALUES (?, 'channel', ?, ?)",
                    (channel_id, content, marker_author),
                )
                await db.execute("COMMIT")
                return cur.lastrowid
            except Exception:
                await db.execute("ROLLBACK")
                raise

    async def reconcile_channel_memory_from_textarea_async(
        self, channel_id: str, seed: list, lines: list, author: str, max_rows: int
    ) -> dict:
        """Reconcile channel-scope memory against an edited textarea in ONE atomic transaction.

        The settings modal renders channel memory as a multiline textarea (one note per line). On
        submit the handler passes:
          - ``seed``: the ``[memory_id, content_hash]`` pairs captured at modal-OPEN (channel
            scope, non-blank rows only) — the exact snapshot the user edited.
          - ``lines``: the submitted textarea lines, already ``normalize_memory_line``-d with
            blanks dropped, deduped, order preserved (re-normalized defensively here).

        Keep / delete / add, all inside one ``BEGIN IMMEDIATE`` transaction (mirrors
        ``upsert_channel_pref_memory``'s autocommit-plus-explicit-txn style) so a concurrent modal
        save serializes and a partial failure rolls back — never a half-applied edit:
          - KEEP a seed whose hash still appears in the textarea → untouched (author preserved).
          - DELETE a seed whose hash left the textarea, but ONLY if the row still exists AND its
            current content still hashes to the seed hash (unchanged since open). If it changed
            since open, count a ``conflict`` and leave it (never clobber a concurrent edit). If it
            is already gone, silently skip (never resurrect a row deleted elsewhere).
          - ADD each textarea line matching NO seed hash AND no surviving channel row's content
            (dedup vs unseeded rows). ``max_rows`` counts ALL remaining channel rows (current −
            deletes + adds-so-far); overflow lines are skipped and counted in ``over_cap``.

        Returns ``{'deleted': [ids], 'added': [contents], 'conflicts': int, 'over_cap': int}``.
        Never raises on empty seed/lines: a fully-blanked box (``lines=[]``) deletes every
        still-unchanged seeded row and adds nothing.
        """
        result: Dict[str, Any] = {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}
        if not channel_id:
            return result

        # Defensive re-normalize + dedup by hash, order preserved (the handler already did this;
        # a second pass keeps the method correct when called directly, e.g. from tests).
        norm_lines: List[str] = []
        line_hashes: set = set()
        for raw in (lines or []):
            n = normalize_memory_line(raw)
            if not n:
                continue
            h = memory_content_hash(n)
            if h in line_hashes:
                continue
            line_hashes.add(h)
            norm_lines.append(n)

        cap = max(1, int(max_rows)) if max_rows is not None else None

        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            # The same keep / delete / add decision against the overlay's rows. The concurrency
            # guards the production path carries — BEGIN IMMEDIATE, the changed-since-open
            # conflict check — have no counterpart here: one battery drives one modal, and the
            # overlay is process-local, so there is no second writer to serialize against.
            store, key = overlay
            from message_processor.channel_steering import PREF_AUTHOR_PREFIX
            current: Dict[Any, str] = {r["id"]: r["content"] for r in store.memory(key)
                       if r["scope"] == "channel"
                       and not (r["author"] or "").startswith(PREF_AUTHOR_PREFIX)}
            seed_ids = {int(mid): h for mid, h in (seed or [])}
            for mem_id, seed_hash in seed_ids.items():
                if seed_hash in line_hashes or mem_id not in current:
                    continue
                if memory_content_hash(current[mem_id]) != seed_hash:
                    result["conflicts"] += 1
                    continue
                store.delete_memory(key, mem_id)
                current.pop(mem_id, None)
                result["deleted"].append(mem_id)
            existing = {memory_content_hash(c) for c in current.values()}
            for line in norm_lines:
                if memory_content_hash(line) in existing:
                    continue
                if cap is not None and len(current) >= cap:
                    result["over_cap"] += 1
                    continue
                new_id = store.add_memory(key, line, scope="channel", author=author)
                current[new_id] = line
                existing.add(memory_content_hash(line))
                result["added"].append(line)
            return result

        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                # 1. Snapshot the current ORDINARY channel-scope rows → {id: content}.
                #    The engine's preference markers are excluded on purpose: they are steering,
                #    they were never shown in the textarea, and a row the user could not see must
                #    not be deletable by an edit to that box — nor may it eat the cap.
                from message_processor.channel_steering import PREF_AUTHOR_PREFIX
                async with db.execute(
                    "SELECT id, content FROM channel_memory "
                    "WHERE scope = 'channel' AND channel_id = ? "
                    "AND (author IS NULL OR author NOT LIKE ? || '%')",
                    (channel_id, PREF_AUTHOR_PREFIX),
                ) as cur:
                    current = {
                        row["id"]: row["content"] for row in await cur.fetchall()
                    }

                # 2-4. Walk the seed once: keep, delete-if-unchanged, or record a conflict.
                deleted_ids: List[Any] = []
                seed_hashes: set = set()
                for entry in (seed or []):
                    try:
                        mem_id, seed_hash = entry[0], entry[1]
                    except (TypeError, IndexError, KeyError):
                        continue
                    seed_hashes.add(seed_hash)
                    if seed_hash in line_hashes:
                        continue  # KEEP — still present in the textarea, leave untouched.
                    cur_content = current.get(mem_id)
                    if cur_content is None:
                        continue  # Already deleted elsewhere — nothing to do, no conflict.
                    if memory_content_hash(cur_content) == seed_hash:
                        await db.execute(
                            "DELETE FROM channel_memory WHERE id = ?", (mem_id,))
                        deleted_ids.append(mem_id)
                        current.pop(mem_id, None)
                    else:
                        # Changed elsewhere since open — never clobber the concurrent edit.
                        result["conflicts"] += 1

                # 5. ADD. Dedup a new line against every seed hash (its KEEP row already covers it)
                #    and against the content of every channel row that survived deletion.
                surviving_hashes: set = {
                    memory_content_hash(c) for c in current.values()
                }
                remaining = len(current)
                added: List[str] = []
                for n in norm_lines:
                    h = memory_content_hash(n)
                    if h in seed_hashes or h in surviving_hashes:
                        continue
                    if cap is not None and remaining >= cap:
                        result["over_cap"] += 1
                        continue
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'channel', ?, ?)",
                        (channel_id, n, author),
                    )
                    added.append(n)
                    surviving_hashes.add(h)
                    remaining += 1

                await db.execute("COMMIT")
                result["deleted"] = deleted_ids
                result["added"] = added
                return result
            except Exception:
                await db.execute("ROLLBACK")
                raise

    # --- Response feedback (Phase H) ---
    async def record_response_feedback_async(self, channel_id: str, thread_ts: Optional[str],
                                             message_ts: str, user_id: str, signal: int,
                                             source: str) -> None:
        """Async version of record_response_feedback (upsert per message/user/source)."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            # PREFERENCE state: a 👍 on a test message is not a judgement about production, and the
            # battery clicks these buttons deliberately. Those clicks must not enter the real
            # signal, so the overlay holds them and the production table is untouched.
            FeedbackRow = _epoch_hook("FeedbackRow")
            store, key = overlay
            store.add_feedback(key, FeedbackRow(
                channel_id=channel_id, thread_ts=thread_ts, message_ts=message_ts,
                user_id=user_id, signal=int(signal), source=source,
                created_ts=f"{time.time():.6f}"))
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO response_feedback (channel_id, thread_ts, message_ts, user_id, signal, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_ts, user_id, source) DO UPDATE SET
                    signal=excluded.signal,
                    updated_ts=CURRENT_TIMESTAMP
            """, (channel_id, thread_ts, message_ts, user_id, signal, source))
            await db.commit()

    async def delete_response_feedback_async(self, message_ts: str, user_id: str, source: str) -> None:
        """Async version of delete_response_feedback."""
        overlay = _epoch_feedback_row(message_ts, user_id, source)
        if overlay is not None:
            store, key = overlay
            store.delete_feedback(key, message_ts, user_id, source)
            return
        if _epoch_fence_up():
            _epoch_refuse_production_write(
                await self._channel_of_feedback_row(message_ts, user_id, source),
                "delete_response_feedback")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "DELETE FROM response_feedback WHERE message_ts = ? AND user_id = ? AND source = ?",
                (message_ts, user_id, source)
            )
            await db.commit()

    async def get_channel_feedback_ratio_async(self, channel_id: str, days: int = 30):
        """Async version of get_channel_feedback_ratio."""
        overlay = _epoch_overlay(channel_id)
        if overlay is not None:
            store, key = overlay
            rows = store.feedback(key)
            positive = sum(1 for r in rows if r.signal > 0)
            negative = sum(1 for r in rows if r.signal < 0)
            total = positive + negative
            return positive, negative, (positive / total if total else None)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT "
                "  SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END) AS positive, "
                "  SUM(CASE WHEN signal < 0 THEN 1 ELSE 0 END) AS negative "
                "FROM response_feedback "
                "WHERE channel_id = ? AND created_ts >= datetime('now', ?)",
                (channel_id, f"-{int(days)} days")
            ) as cursor:
                row = cast(Any, await cursor.fetchone())
        positive = row["positive"] or 0
        negative = row["negative"] or 0
        total = positive + negative
        return positive, negative, (positive / total if total else None)

    async def get_or_create_user_async(self, user_id: str, username: Optional[str] = None) -> Dict:
        """
        Async version of get_or_create_user.

        Args:
            user_id: User identifier
            username: Optional username

        Returns:
            User data dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            # Try to get existing user
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update last seen
                await db.execute("""
                    UPDATE users SET last_seen = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
                await db.commit()
                return dict(row)

            # Create new user
            await db.execute("""
                INSERT INTO users (user_id, username, created_at, last_seen)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (user_id, username))

            await db.commit()

            # Return the created user
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {}

    async def get_user_info_async(self, user_id: str) -> Optional[Dict]:
        """
        Async version of get_user_info.

        Args:
            user_id: User identifier

        Returns:
            User info dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_user_infos_async(self, user_ids) -> Dict[str, Dict]:
        """BF2 bulk read: {user_id: row-dict} for the given ids in ONE connection, chunked to
        stay under SQLite's bound-variable limit. Read-only; mirrors get_user_info_async's row
        shape. Ids with no row are simply absent from the result (never created)."""
        ids = [u for u in dict.fromkeys(user_ids) if u]
        if not ids:
            return {}
        out: Dict[str, Dict] = {}
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            for i in range(0, len(ids), 500):  # under SQLite's ~999 variable cap
                chunk = ids[i:i + 500]
                placeholders = ",".join("?" for _ in chunk)
                async with db.execute(
                    f"SELECT * FROM users WHERE user_id IN ({placeholders})", chunk
                ) as cursor:
                    async for row in cursor:
                        d = dict(row)
                        out[d["user_id"]] = d
        return out

    async def get_all_users_async(self) -> list:
        """F29: all persisted user_info rows (user_id/username/real_name/email/tz), for
        resolving a name → id when lookup_user is called with a name rather than a Slack id.
        Read-only; returns a list of dicts (empty on any failure)."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                async with db.execute(
                    "SELECT user_id, username, real_name, email, timezone, tz_label "
                    "FROM users"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"DB: get_all_users_async failed: {e}")
            return []

    async def get_user_preferences_async(self, user_id: str) -> Optional[Dict]:
        """
        Async version of get_user_preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM user_preferences WHERE slack_user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    prefs = dict(row)
                    # Convert SQLite boolean (0/1) to Python boolean
                    prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
                    prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
                    prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
                    prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
                    return prefs
                return None

    async def create_default_user_preferences_async(self, user_id: str, email: str) -> Dict:
        """
        Async version of create_default_user_preferences.

        Args:
            user_id: User identifier
            email: User email

        Returns:
            Created preferences dictionary
        """
        from config import BotConfig
        config = BotConfig()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            # Create default preferences
            await db.execute("""
                INSERT OR REPLACE INTO user_preferences (
                    slack_user_id, slack_email,
                    model, temperature, top_p,
                    enable_web_search, enable_streaming,
                    reasoning_effort, verbosity,
                    image_model, image_size, image_quality, image_background,
                    input_fidelity, vision_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email,
                config.gpt_model, config.default_temperature, config.default_top_p,
                1 if config.enable_web_search else 0,
                1 if config.enable_streaming else 0,
                config.default_reasoning_effort, config.default_verbosity,
                config.image_model,
                config.default_image_size, config.default_image_quality, config.default_image_background,
                config.default_input_fidelity, config.default_detail_level
            ))

            await db.commit()

            # Return the created preferences
            async with db.execute(
                "SELECT * FROM user_preferences WHERE slack_user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    prefs = dict(row)
                    # Convert SQLite boolean (0/1) to Python boolean
                    prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
                    prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
                    prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
                    prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
                    return prefs
                return {}

    async def find_thread_images_async(self, thread_id: str, image_type: Optional[str] = None) -> List[Dict]:
        """Async version of find_thread_images."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            params: Tuple[Any, ...]
            if image_type:
                query = ("SELECT * FROM images WHERE thread_id = ? AND image_type = ? "
                         "ORDER BY created_at ASC")
                params = (thread_id, image_type)
            else:
                query = "SELECT * FROM images WHERE thread_id = ? ORDER BY created_at ASC"
                params = (thread_id,)

            async with db.execute(query, params) as cursor:
                images = []
                async for row in cursor:
                    img = dict(row)
                    if img.get("metadata_json"):
                        img["metadata"] = json.loads(img["metadata_json"])
                        del img["metadata_json"]
                    images.append(img)
                return images

    async def get_images_by_message_async(self, thread_id: str, message_ts: str) -> List[Dict]:
        """Async version of get_images_by_message."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM images WHERE thread_id = ? AND message_ts = ? ORDER BY created_at ASC",
                (thread_id, message_ts)
            ) as cursor:
                images = []
                async for row in cursor:
                    img = dict(row)
                    if img.get("metadata_json"):
                        img["metadata"] = json.loads(img["metadata_json"])
                        del img["metadata_json"]
                    images.append(img)
                return images

    # ---------------------------------------------------------------- F51 ambient artifacts
    #
    # Channel + source-ts keyed derivations for ambiently-seen images/links/files. Slack stays
    # the only transcript: these hold summaries + refs ONLY. Reuse is same-channel by design —
    # cross-channel reuse could leak private-channel/DM-derived content elsewhere.

    async def insert_pending_ambient_artifact(
        self, *, channel_id: str, source_ts: str, conversation_ts: str, kind: str, ref: str,
        content_type: Optional[str] = None, derivation_source: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[Dict]:
        """Claim (or observe) an ambient artifact occurrence. Idempotent by
        (channel_id, source_ts, kind, ref): a re-offer of the same occurrence does NOT clobber an
        existing row (singleflight — a ready summary survives). Returns the row as it stands AFTER
        the call (dict), so the caller can see whether it is already `ready`/`pending`/etc."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO ambient_artifacts
                    (channel_id, source_ts, conversation_ts, kind, ref, status,
                     content_type, derivation_source, expires_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(channel_id, source_ts, kind, ref) DO NOTHING
            """, (channel_id, source_ts, conversation_ts, kind, ref,
                  content_type, derivation_source, expires_at))
            await db.commit()
            async with db.execute("""
                SELECT * FROM ambient_artifacts
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (channel_id, source_ts, kind, ref)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def set_ambient_artifact_ready(
        self, *, channel_id: str, source_ts: str, kind: str, ref: str,
        title: Optional[str], summary: str, model: Optional[str],
        derivation_source: str, content_type: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> None:
        """Mark an artifact ready with its derived summary. Only writes a row that exists (the
        pending occurrence was claimed first) — status flips to `ready`, fetched_at stamped."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                UPDATE ambient_artifacts
                SET status = 'ready', title = ?, summary = ?, model = ?,
                    derivation_source = ?, content_type = COALESCE(?, content_type),
                    error_code = NULL, updated_at = CURRENT_TIMESTAMP,
                    fetched_at = CURRENT_TIMESTAMP, expires_at = COALESCE(?, expires_at)
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (title, summary, model, derivation_source, content_type, expires_at,
                  channel_id, source_ts, kind, ref))
            await db.commit()

    async def set_ambient_artifact_status(
        self, *, channel_id: str, source_ts: str, kind: str, ref: str,
        status: str, error_code: Optional[str] = None, increment_attempt: bool = False,
        derivation_source: Optional[str] = None,
    ) -> None:
        """Persist a terminal/interim status (failed/blocked/omitted/pending) with an honest
        error_code — the house rule is no silent drops."""
        attempt_sql = "attempt_count = attempt_count + 1," if increment_attempt else ""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(f"""
                UPDATE ambient_artifacts
                SET status = ?, error_code = ?, {attempt_sql}
                    derivation_source = COALESCE(?, derivation_source),
                    updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (status, error_code, derivation_source,
                  channel_id, source_ts, kind, ref))
            await db.commit()

    async def get_ambient_artifacts_for_messages(
        self, channel_id: str, source_ts_list: List[str],
        statuses: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """ONE batched query (never N+1) mapping source_ts -> [artifact rows], for rendering a
        whole thread/page of history at once. Same-channel scoped. Deterministic order (id ASC)."""
        if not channel_id or not source_ts_list:
            return {}
        # De-dup and cap placeholders defensively.
        uniq = list(dict.fromkeys(str(t) for t in source_ts_list if t))
        if not uniq:
            return {}
        placeholders = ",".join("?" for _ in uniq)
        status_filter = ""
        params: List[Any] = [channel_id, *uniq]
        if statuses:
            status_filter = f" AND status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        out: Dict[str, List[Dict]] = {}
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(f"""
                SELECT * FROM ambient_artifacts
                WHERE channel_id = ? AND source_ts IN ({placeholders}){status_filter}
                ORDER BY source_ts ASC, id ASC
            """, params) as cursor:
                async for row in cursor:
                    r = dict(row)
                    out.setdefault(r["source_ts"], []).append(r)
        return out

    async def find_reusable_ambient_summary(
        self, channel_id: str, kind: str, ref: str, *, fresh_after: Optional[str] = None,
    ) -> Optional[Dict]:
        """A ready summary for the same ref IN THE SAME CHANNEL, optionally requiring
        fetched_at >= fresh_after (ISO/SQL datetime) so a stale link re-fetches. Newest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            query = ("SELECT * FROM ambient_artifacts WHERE channel_id = ? AND kind = ? "
                     "AND ref = ? AND status = 'ready' AND summary IS NOT NULL")
            params: List[Any] = [channel_id, kind, ref]
            if fresh_after:
                query += " AND (fetched_at IS NULL OR fetched_at >= ?)"
                params.append(fresh_after)
            query += " ORDER BY fetched_at DESC, id DESC LIMIT 1"
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def delete_ambient_artifacts_by_source(self, channel_id: str, source_ts: str) -> int:
        """Purge all artifacts for a (channel, source message) — message_deleted/edit lifecycle.

        ALSO purges the ambient image analyses the vision worker dual-wrote into `images`
        (marked metadata `{"ambient": true}`, message_ts == source_ts, thread_id under this
        channel). Without this the deleted/edited image's description survives in the ledger and
        keeps being injected — the exact leak the retention/deletion path is meant to close."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE channel_id = ? AND source_ts = ?",
                (channel_id, source_ts))
            # Exact structured match: the dual-written ambient row carries metadata
            # {"ambient":true,"channel_id":...}. json_extract avoids matching {"ambient":false}
            # or {"description":"ambient"}; json_valid guards any legacy non-JSON metadata.
            await db.execute(
                "DELETE FROM images WHERE message_ts = ? AND metadata_json IS NOT NULL "
                "AND json_valid(metadata_json) "
                "AND json_extract(metadata_json, '$.ambient') = 1 "
                "AND json_extract(metadata_json, '$.channel_id') = ?",
                (source_ts, channel_id))
            # F51c: a late-artifact addendum for this source message must die with it, or a
            # deleted/edited message's derived note keeps riding the summary head.
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE channel_id = ? AND source_ts = ?",
                (channel_id, source_ts))
            await db.commit()
            return cursor.rowcount or 0

    async def delete_ambient_artifacts_by_ref(self, channel_id: str, kind: str, ref: str) -> int:
        """Purge artifacts for a specific ref (file_deleted lifecycle — a Slack file removed)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE channel_id = ? AND kind = ? AND ref = ?",
                (channel_id, kind, ref))
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE channel_id = ? AND kind = ? AND ref = ?",
                (channel_id, kind, ref))
            await db.commit()
            return cursor.rowcount or 0

    async def delete_ambient_artifacts_by_file_id(self, file_id: str) -> int:
        """Purge image/file artifacts derived from a Slack file id, workspace-wide (file_deleted).
        A file id is globally unique, so no channel scope is needed."""
        if not file_id:
            return 0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE ref = ? AND kind IN ('image','file')",
                (file_id,))
            # Exact structured match on the file id stored in metadata — NOT a url substring LIKE
            # (an id that is a substring of another url would cross-delete the wrong image).
            await db.execute(
                "DELETE FROM images WHERE metadata_json IS NOT NULL AND json_valid(metadata_json) "
                "AND json_extract(metadata_json, '$.ambient') = 1 "
                "AND json_extract(metadata_json, '$.file_id') = ?",
                (file_id,))
            # F51c: file id is the addendum ref for image/file kinds — purge those too.
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE ref = ? AND kind IN ('image','file')",
                (file_id,))
            await db.commit()
            return cursor.rowcount or 0

    async def get_pending_ambient_artifacts(self, limit: int = 200) -> List[Dict]:
        """Rows still `pending` — interrupted work to resume on restart. Oldest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM ambient_artifacts WHERE status = 'pending' "
                "ORDER BY created_at ASC LIMIT ?", (int(limit),)) as cursor:
                return [dict(row) async for row in cursor]

    def delete_expired_ambient_artifacts(self, days: int = 30) -> List[str]:
        """Retention sweep (sync, mirrors delete_old_tool_usage) — wired into the scheduled
        cleanup worker. Uses expires_at when set, else falls back to a created_at age cutoff.
        Cutoffs computed IN SQL (UTC) like the other sweeps, never a local Python datetime.

        Returns the DISTINCT thread keys (`channel_id:thread_ts`) whose late-artifact addenda were
        deleted. F51d: the sweep clears the addenda from the DB, but an ACTIVE warm thread still
        holds an in-memory summary head carrying the expired note and would keep sending it
        indefinitely; the cleanup worker marks each returned thread for refresh so its next turn
        rebuilds without the note."""
        # F51c: retire the aged artifacts' late-artifact addenda in the SAME operation. A row's
        # derived note otherwise lingers indefinitely in the summary head after its artifact ages
        # out — and keeps occupying one of the per-thread addenda cap slots. Match on the artifact
        # identity (channel_id + source_ts + kind + ref); the row-value IN subquery uses the SAME
        # cutoff predicate and MUST run BEFORE the artifacts are deleted (afterwards the subquery
        # would find nothing to match).
        # Capture the affected thread keys BEFORE the delete — the same identity subquery finds
        # nothing once the artifacts are gone. thread_summary_addenda.thread_id is already stored
        # as the full `channel_id:thread_ts` key, so it is the mark_needs_refresh key verbatim.
        affected = self.conn.execute("""
            SELECT DISTINCT thread_id FROM thread_summary_addenda
            WHERE (channel_id, source_ts, kind, ref) IN (
                SELECT channel_id, source_ts, kind, ref FROM ambient_artifacts
                WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
                   OR (expires_at IS NULL AND created_at < datetime('now', ?))
            )
        """, (f"-{int(days)} days",)).fetchall()
        affected_thread_keys = [row[0] for row in affected]
        self.conn.execute("""
            DELETE FROM thread_summary_addenda
            WHERE (channel_id, source_ts, kind, ref) IN (
                SELECT channel_id, source_ts, kind, ref FROM ambient_artifacts
                WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
                   OR (expires_at IS NULL AND created_at < datetime('now', ?))
            )
        """, (f"-{int(days)} days",))
        cursor = self.conn.execute("""
            DELETE FROM ambient_artifacts
            WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
               OR (expires_at IS NULL AND created_at < datetime('now', ?))
        """, (f"-{int(days)} days",))
        # Retention must also reach the dual-written ambient image analyses (metadata
        # `{"ambient": true}`) — they have no expires_at column, so age them by created_at with
        # the same window. Addressed uploads (no ambient marker) are untouched.
        img_cursor = self.conn.execute("""
            DELETE FROM images
            WHERE metadata_json IS NOT NULL AND json_valid(metadata_json)
              AND json_extract(metadata_json, '$.ambient') = 1
              AND created_at < datetime('now', ?)
        """, (f"-{int(days)} days",))
        if cursor.rowcount > 0 or img_cursor.rowcount > 0:
            self.log_info(f"DB: Cleaned up {cursor.rowcount} ambient artifacts + "
                          f"{img_cursor.rowcount} ambient image analyses (retention {days}d)")
        return affected_thread_keys

    async def get_thread_documents_async(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Async version of get_thread_documents."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            query = "SELECT * FROM documents WHERE thread_id = ? ORDER BY created_at ASC"
            if limit:
                query += f" LIMIT {int(limit)}"

            async with db.execute(query, (thread_id,)) as cursor:
                documents = []
                async for row in cursor:
                    doc = dict(row)
                    if doc.get("page_structure"):
                        doc["page_structure"] = json.loads(doc["page_structure"])
                    if doc.get("metadata_json"):
                        doc["metadata"] = json.loads(doc["metadata_json"])
                        del doc["metadata_json"]
                    documents.append(doc)
                return documents

    async def find_channel_images_async(self, channel_id: str, within_hours: Optional[int] = None,
                                        limit: int = 50) -> List[Dict]:
        """Images shared anywhere in a channel, NEWEST FIRST.

        The image twin of get_channel_documents_async, and it exists for the same reason: a
        thread_id is "channel:thread", so a conversation that fragments across roots hides its own
        images from itself. That is chronic in DMs, where every top-level message is its own root —
        send a picture, then ask about it in the next message, and the picture is in another
        "thread".

        Same privacy boundary as the document lookup: a prefix LIKE on ``channel_id + ':'``, and
        channel ids are alphanumeric (no LIKE metacharacters), so this cannot escape the channel.
        `within_hours` bounds it in time; None means no bound.
        """
        params: List[Any] = [f"{channel_id}:%"]
        where = "thread_id LIKE ?"
        if within_hours is not None:
            where += " AND created_at >= datetime('now', ?)"
            params.append(f"-{int(within_hours)} hours")
        params.append(int(limit))

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                f"SELECT * FROM images WHERE {where} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ) as cursor:
                images = []
                async for row in cursor:
                    img = dict(row)
                    if img.get("metadata_json"):
                        img["metadata"] = json.loads(img["metadata_json"])
                        del img["metadata_json"]
                    images.append(img)
                return images

    async def get_channel_documents_async(self, channel_id: str) -> List[Dict]:
        """All documents shared anywhere in a channel (F22 channel-wide access).

        thread_id is stored as "channel:thread"; a channel's documents are every row
        whose thread_id starts with ``channel_id + ':'``. Channel ids are alphanumeric
        (no LIKE metacharacters), so a plain prefix LIKE is safe and cannot escape the
        channel — the privacy boundary is same-channel-only. Same row shape and
        created_at ASC ordering as get_thread_documents_async."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM documents WHERE thread_id LIKE ? ORDER BY created_at ASC",
                (f"{channel_id}:%",),
            ) as cursor:
                documents = []
                async for row in cursor:
                    doc = dict(row)
                    if doc.get("page_structure"):
                        doc["page_structure"] = json.loads(doc["page_structure"])
                    if doc.get("metadata_json"):
                        doc["metadata"] = json.loads(doc["metadata_json"])
                        del doc["metadata_json"]
                    documents.append(doc)
                return documents

    async def get_document_by_filename_async(self, thread_id: str, filename: str) -> Optional[Dict]:
        """Async version of get_document_by_filename (newest matching row)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                """SELECT * FROM documents
                   WHERE thread_id = ? AND filename = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (thread_id, filename),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                doc = dict(row)
                if doc.get("page_structure"):
                    doc["page_structure"] = json.loads(doc["page_structure"])
                if doc.get("metadata_json"):
                    doc["metadata"] = json.loads(doc["metadata_json"])
                    del doc["metadata_json"]
                return doc

    async def get_document_for_rebuild_async(self, thread_id: str, filename: str,
                                             file_id: Optional[str] = None,
                                             message_ts: Optional[str] = None) -> Optional[Dict]:
        """Resolve the document row for a SPECIFIC historical upload during rebuild (F12).

        get_document_by_filename_async is newest-wins by filename, so two same-named
        uploads in one thread both resolve to the newest file's summary/file_id after a
        restart. Disambiguate with the identity we have in hand for the message we're
        replaying: the Slack file_id (exact, survives same-name collisions), else the
        nearest upload at/before this message's ts. Falls back to newest-by-filename so a
        legacy row (no file_id, no message_ts) still resolves as before."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            row = None
            # 1. Exact Slack file_id — one upload, one row (renames don't fool it).
            if file_id:
                async with db.execute(
                    """SELECT * FROM documents
                       WHERE thread_id = ? AND file_id = ?
                       ORDER BY created_at DESC, id DESC LIMIT 1""",
                    (thread_id, file_id),
                ) as cursor:
                    row = await cursor.fetchone()
            # 2. Nearest same-named upload at/before this message's ts. Slack ts sort
            #    lexically only when equal-width, so compare as REAL.
            if row is None and message_ts:
                async with db.execute(
                    """SELECT * FROM documents
                       WHERE thread_id = ? AND filename = ? AND message_ts IS NOT NULL
                         AND CAST(message_ts AS REAL) <= CAST(? AS REAL)
                       ORDER BY CAST(message_ts AS REAL) DESC, id DESC LIMIT 1""",
                    (thread_id, filename, message_ts),
                ) as cursor:
                    row = await cursor.fetchone()
            # 3. Legacy fallback: newest row for the filename.
            if row is None:
                async with db.execute(
                    """SELECT * FROM documents
                       WHERE thread_id = ? AND filename = ?
                       ORDER BY created_at DESC, id DESC LIMIT 1""",
                    (thread_id, filename),
                ) as cursor:
                    row = await cursor.fetchone()

            if not row:
                return None
            doc = dict(row)
            if doc.get("page_structure"):
                doc["page_structure"] = json.loads(doc["page_structure"])
            if doc.get("metadata_json"):
                doc["metadata"] = json.loads(doc["metadata_json"])
                del doc["metadata_json"]
            return doc

    async def get_or_create_thread_async(self, thread_id: str, channel_id: str,
                                         user_id: Optional[str] = None) -> Dict:
        """Async wrapper for get_or_create_thread.

        The sync method is multi-step (lookup, activity touch, user-config copy,
        insert, recursive re-read); duplicating it in aiosqlite risks divergence,
        so it runs unchanged on a worker thread (the shared connection is created
        with check_same_thread=False and WAL handles concurrency).
        """
        return await asyncio.to_thread(self.get_or_create_thread, thread_id, channel_id, user_id)

    async def cleanup_old_modal_sessions_async(self, hours: int = 24):
        """Async version of cleanup_old_modal_sessions."""
        try:
            cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                cursor = await db.execute(
                    "DELETE FROM modal_sessions WHERE created_at < ?", (cutoff,)
                )
                await db.commit()
                if cursor.rowcount > 0:
                    self.log_info(f"Cleaned up {cursor.rowcount} modal sessions older than {hours} hours")
        except Exception as e:
            self.log_error(f"Failed to cleanup modal sessions: {e}")

    # F32 container async wrappers. asyncio.to_thread over the sync methods, matching
    # get_or_create_thread_async — these run on the message path, so they must not block
    # the event loop on SQLite.

    async def get_thread_container_async(self, thread_id: str) -> Optional[Dict]:
        return await asyncio.to_thread(self.get_thread_container, thread_id)

    async def get_fresh_thread_container_async(self, thread_id: str,
                                               reuse_minutes: int) -> Optional[Dict]:
        return await asyncio.to_thread(self.get_fresh_thread_container, thread_id, reuse_minutes)

    async def save_thread_container_async(self, thread_id: str, container_id: str):
        return await asyncio.to_thread(self.save_thread_container, thread_id, container_id)

    async def adopt_thread_container_async(self, thread_id: str,
                                           container_id: str) -> Optional[str]:
        return await asyncio.to_thread(self.adopt_thread_container, thread_id, container_id)

    async def touch_thread_container_async(self, thread_id: str, container_id: str):
        return await asyncio.to_thread(self.touch_thread_container, thread_id, container_id)

    async def add_published_container_files_async(self, thread_id: str, container_id: str,
                                                  file_ids: List[str]):
        return await asyncio.to_thread(
            self.add_published_container_files, thread_id, container_id, file_ids)

    async def delete_thread_container_async(self, thread_id: str,
                                            container_id: Optional[str] = None):
        return await asyncio.to_thread(self.delete_thread_container, thread_id, container_id)

    async def get_expired_thread_containers_async(self, older_than_minutes: int) -> List[Dict]:
        return await asyncio.to_thread(self.get_expired_thread_containers, older_than_minutes)

    async def get_user_timezone_async(self, user_id: str) -> Optional[str]:
        """
        Async version of get_user_timezone.

        Args:
            user_id: User identifier

        Returns:
            Timezone string or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT timezone FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row["timezone"] if row and row["timezone"] else None

    async def save_user_info_async(self, user_id: str, username: str, real_name: str, email: str,
                                   timezone: Optional[str] = None, tz_label: Optional[str] = None,
                                   tz_offset: Optional[int] = None):
        """
        Async version of save_user_info.

        Args:
            user_id: User identifier
            username: Username
            real_name: Real name
            email: Email address
            timezone: Optional timezone
            tz_label: Optional timezone label
            tz_offset: Optional timezone offset
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE users
                SET username = ?, real_name = ?, email = ?, timezone = ?, last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (username, real_name, email, timezone, user_id))

            await db.commit()

    async def update_user_preferences_async(self, user_id: str, preferences: Dict) -> bool:
        """
        Async version of update_user_preferences.

        Args:
            user_id: User identifier
            preferences: Preferences to update

        Returns:
            True if update successful
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            # Build dynamic update query
            update_fields = []
            values = []

            # Handle regular fields
            for field in ['model', 'reasoning_effort', 'verbosity', 'temperature',
                         'top_p', 'image_model', 'image_size', 'image_quality', 'image_background',
                         'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    update_fields.append(f"{field} = ?")
                    values.append(preferences[field])

            # Handle boolean fields - convert to integers for SQLite
            for field in ['enable_web_search', 'enable_streaming', 'enable_mcp']:
                if field in preferences:
                    update_fields.append(f"{field} = ?")
                    values.append(1 if preferences[field] else 0)

            if not update_fields:
                return False

            values.append(user_id)
            query = f"""
                UPDATE user_preferences
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE slack_user_id = ?
            """

            await db.execute(query, values)
            await db.commit()
            return True

    async def get_thread_config_async(self, thread_id: str) -> Optional[Dict]:
        """
        Async version of get_thread_config.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread config dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT config_json FROM threads WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row and row["config_json"]:
                return json.loads(row["config_json"])

            return None

    # =====================================================================================
    # Single-stream P1 (Docs/SINGLE_STREAM_SPEC.md §4/§5): meta, outbound receipts,
    # thread-activity index, the root inventory.
    #
    # Two rules hold across every accessor below. Slack timestamps compare as REAL, never as
    # TEXT ("999.123456" sorts after "1000.5" as text). NULL-safety is spelled out with CASE
    # instead of scalar MAX(), which returns NULL if ANY argument is NULL and would erase a
    # known value the first time a blind write lands.
    # =====================================================================================

    # The house busy timeout: how long a stream accessor waits for a locked database before it
    # gives up. Five seconds is right for a turn-path read that has nothing better to do than
    # wait — and WRONG for a caller that is already inside a shorter deadline of its own.
    STREAM_BUSY_TIMEOUT_MS = 5000

    @asynccontextmanager
    async def _stream_conn(self, *, busy_timeout_ms: Optional[int] = None):
        """Per-call connection with the house pragmas: autocommit, WAL, 5s busy timeout.

        `busy_timeout_ms` CAPS THE LOCK WAIT for a caller that is running against a deadline.
        Cancelling such a read from outside is not enough: `asyncio.wait_for` cancels the
        coroutine and then waits for it to unwind, and unwinding drains whatever SQLite work is
        already queued — so a read blocked on a lock can overshoot its caller's deadline by the
        whole busy timeout (codex verify, P2). The bound has to be inside the connection.
        Coerced through `int()` because a PRAGMA cannot take a bound parameter.
        """
        timeout_ms = (self.STREAM_BUSY_TIMEOUT_MS if busy_timeout_ms is None
                      else max(0, int(busy_timeout_ms)))
        # THE CONNECT TIMEOUT AND THE PRAGMA, IN THAT ORDER, AND BOTH ARE NEEDED. `sqlite3`
        # applies its own 5-second default busy timeout the moment the connection opens, so the
        # FIRST statement — `journal_mode`, which takes a lock — would wait five seconds on a
        # locked database however small the PRAGMA that follows it. Measured: capping only the
        # pragma left a 0.3s-budgeted read blocking for 5.004s. `timeout=` bounds the opening
        # statements, the PRAGMA bounds every statement after them.
        async with aiosqlite.connect(self.db_path, isolation_level=None,
                                     timeout=timeout_ms / 1000.0) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(f"PRAGMA busy_timeout={timeout_ms}")
            await db.execute("PRAGMA journal_mode=WAL")
            yield db

    # --- bot_meta -----------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        """Read one bot_meta value (sync: boot paths run before the event loop)."""
        cursor = self.conn.execute("SELECT value FROM bot_meta WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    async def get_meta_async(self, key: str) -> Optional[str]:
        """Async twin of get_meta."""
        async with self._stream_conn() as db:
            async with db.execute("SELECT value FROM bot_meta WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                return row["value"] if row else None

    async def set_meta_async(self, key: str, value: str) -> None:
        """Upsert a bot_meta value. NOT for epoch-class keys — see set_meta_if_absent_async."""
        async with self._stream_conn() as db:
            await db.execute(
                "INSERT INTO bot_meta (key, value, updated_ts) "
                "VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "                               updated_ts = CURRENT_TIMESTAMP",
                (key, value))

    async def set_meta_if_absent_async(self, key: str, value: str) -> bool:
        """Write `key` only if it has never been written; True when this call wrote it.

        The only legal writer for epoch-class keys: rewriting the receipts epoch would
        re-grandfather every own-message posted since the first boot.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "INSERT OR IGNORE INTO bot_meta (key, value) VALUES (?, ?)", (key, value))
            return bool(cursor.rowcount)

    # --- outbound receipts (spec §5) -----------------------------------------------------

    async def register_receipt_async(self, team_id: str, channel_id: str, message_ts: str,
                                     turn_id: str, state: str,
                                     thread_root_ts: Optional[str] = None, *,
                                     receipt_class: Optional[str]) -> TransitionResult:
        """Claim a receipt for a message we posted. `applied` when this call owns the result.

        Every conflict resolves in favor of not losing a stronger claim:
        `finalized` is absorbing (a late registration never demotes it), a row held by a
        different turn is left alone and logged loudly, and chrome→in_flight promotion is
        same-owner only. Registering chrome over an in_flight row is refused —
        demote_receipt_chrome_async is the only legal path down.

        `receipt_class` (spec §4) is a REQUIRED keyword — every producer says what kind of
        surface this is, and only plumbing that genuinely has no claim passes None (the row
        then stays ineligible for anything class-gated). NULL is IMMUTABLE (spec §11.2): it is
        never filled by a later same-owner claim and never promoted over — a legacy row can
        never become editable. Placeholder promotion (chrome → assistant_reply, with the state
        move, over a stored `chrome` class only) is the one sanctioned class change; any other
        same-message class conflict fails CLOSED to NULL (spec §11.1) — the state transition is
        refused, the row's class is NULLed (the ineligible terminal state),
        `reason="class_conflict"`, and an ERROR names both classes.

        Each of those refusals returns its own `reason`, because from the call site they are
        indistinguishable and the ledger has to tell them apart.
        """
        if state not in _RECEIPT_STATES:
            raise ValueError(f"invalid receipt state: {state!r}")
        _check_receipt_class(receipt_class)
        key = (team_id, channel_id, message_ts)
        async with self._stream_conn() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT turn_id, state, thread_root_ts, receipt_class "
                    "FROM outbound_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND message_ts = ?", key
                ) as cursor:
                    row = await cursor.fetchone()

                if row is None:
                    await db.execute(
                        "INSERT INTO outbound_receipts "
                        "(team_id, channel_id, message_ts, turn_id, state, thread_root_ts, "
                        " receipt_class, finalized_ts) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, "
                        "        CASE WHEN ? = 'finalized' THEN CURRENT_TIMESTAMP END)",
                        (*key, turn_id, state, thread_root_ts, receipt_class, state))
                    await db.execute("COMMIT")
                    self.log_debug(f"Receipt {channel_id}/{message_ts} → {state} "
                                   f"[{receipt_class}] ({turn_id})")
                    return TransitionResult(True, "absent", state, "inserted")

                current = row["state"]
                current_class = row["receipt_class"]

                # Class arbitration BEFORE anything else, so a refused registration changes
                # nothing but the class itself — not the root fill below, not the state. The
                # one sanctioned change is the placeholder promotion: a stored `chrome` class
                # becoming the answer. NULL is IMMUTABLE (spec §11.2): a NULL row promotes
                # with class still NULL, so a legacy row can never become editable.
                promotion = (current == "chrome" and state == "in_flight"
                             and receipt_class == "assistant_reply"
                             and current_class == "chrome")
                if (receipt_class is not None and current_class is not None
                        and receipt_class != current_class and not promotion):
                    # Spec §11.1: a conflict fails closed to NULL — the ineligible terminal
                    # state. Nothing editable can be produced by a conflict.
                    await db.execute(
                        "UPDATE outbound_receipts SET receipt_class = NULL "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?", key)
                    await db.execute("COMMIT")
                    self.log_error(
                        f"Receipt {channel_id}/{message_ts} class conflict: row was "
                        f"{current_class!r}, {turn_id} claimed {receipt_class!r} — class "
                        f"NULLed (ineligible), registration refused")
                    return TransitionResult(False, current, current, "class_conflict")

                # A message's destination root is a property of the message, not of the owner,
                # so a later observation may FILL a NULL — it may never clear a known one.
                if thread_root_ts and not row["thread_root_ts"]:
                    await db.execute(
                        "UPDATE outbound_receipts SET thread_root_ts = ? "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                        (thread_root_ts, *key))

                if current == "finalized":
                    await db.execute("COMMIT")
                    self.log_debug(
                        f"Receipt {channel_id}/{message_ts} already finalized; "
                        f"{state} registration by {turn_id} absorbed")
                    return TransitionResult(
                        False, "finalized", "finalized", "absorbed_finalized")
                if row["turn_id"] != turn_id:
                    await db.execute("COMMIT")
                    self.log_warning(
                        f"Receipt {channel_id}/{message_ts} is held by {row['turn_id']} "
                        f"({current}); refusing {state} registration by {turn_id}")
                    return TransitionResult(False, current, current, "foreign_owner")
                if current == "in_flight" and state == "chrome":
                    await db.execute("COMMIT")
                    self.log_warning(
                        f"Receipt {channel_id}/{message_ts} in_flight; chrome registration by "
                        f"{turn_id} refused (demote_receipt_chrome_async is the only path down)")
                    return TransitionResult(
                        False, "in_flight", "in_flight", "chrome_over_in_flight")
                # The class this row ends the call with: a promotion rewrites it WITH the
                # state (atomically — same transaction); otherwise the stored value —
                # including an immutable NULL (spec §11.2, no same-owner fill) — is never
                # touched.
                class_to_write = receipt_class if promotion else current_class
                if class_to_write != current_class:
                    await db.execute(
                        "UPDATE outbound_receipts SET receipt_class = ? "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                        (class_to_write, *key))
                if current != state:
                    await db.execute(
                        "UPDATE outbound_receipts SET state = ?, finalized_ts = "
                        "  CASE WHEN ? = 'finalized' THEN CURRENT_TIMESTAMP ELSE finalized_ts END "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                        (state, state, *key))
                    self.log_debug(
                        f"Receipt {channel_id}/{message_ts} {current}→{state} "
                        f"[{class_to_write}] ({turn_id})")
                    await db.execute("COMMIT")
                    return TransitionResult(True, current, state, "transitioned")
                await db.execute("COMMIT")
                return TransitionResult(True, current, state, "unchanged")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    async def register_chrome_async(self, team_id: str, channel_id: str, message_ts: str,
                                    owner_turn_id: str,
                                    thread_root_ts: Optional[str] = None, *,
                                    receipt_class: Optional[str]) -> TransitionResult:
        """Register a permanently excluded surface (placeholder, status card, footer).

        Chrome STATE and chrome CLASS are orthogonal: a research status card is chrome-state
        with class `background_job`, a placeholder is chrome-state with class `chrome`."""
        return await self.register_receipt_async(
            team_id, channel_id, message_ts, owner_turn_id, "chrome", thread_root_ts,
            receipt_class=receipt_class)

    async def transfer_receipt_async(self, team_id: str, channel_id: str, message_ts: str,
                                     expected_owner: str,
                                     new_owner: str) -> TransitionResult:
        """Hand a CHROME row to another owner. Refused for any other state.

        Only chrome transfers: a surface that has never carried conversational content (a
        placeholder another turn is about to reuse) is the only thing an owner may give away.
        The state never moves, so both ends of the transition are `chrome` — what changes is
        the owner, which the event carries separately.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "UPDATE outbound_receipts SET turn_id = ? "
                "WHERE team_id = ? AND channel_id = ? AND message_ts = ? "
                "  AND turn_id = ? AND state = 'chrome'",
                (new_owner, team_id, channel_id, message_ts, expected_owner))
            moved = bool(cursor.rowcount)
        if moved:
            self.log_debug(
                f"Receipt {channel_id}/{message_ts} chrome transferred "
                f"{expected_owner}→{new_owner}")
        else:
            self.log_warning(
                f"Receipt {channel_id}/{message_ts} transfer refused "
                f"(expected chrome owned by {expected_owner})")
        return TransitionResult(
            moved, "chrome", "chrome",
            "transferred" if moved else "not_chrome_or_foreign")

    async def finalize_receipts_async(self, team_id: str, channel_id: str,
                                      records: Iterable[Tuple[str, Optional[str],
                                                              Optional[str]]],
                                      turn_id: str) -> List[TransitionResult]:
        """Finalize a turn's posts as ONE unit.
        `records` = [(message_ts, thread_root_ts|None, receipt_class|None)].

        Missing rows are INSERTED already finalized (their registration may have been lost to a
        failed write) and carry their destination root AND class. A row owned by another turn is
        left alone; a NULL root is filled and a known one never cleared. An EXISTING row's class
        is never touched at all — NULL is immutable (spec §11.2), so a NULL class stays NULL —
        and a record claiming a DIFFERENT class than the row already carries fails CLOSED to
        NULL for that record (spec §11.1): no state change, the row's class is NULLed (the
        ineligible terminal state), and an ERROR names both classes. Class arbitration is
        ORTHOGONAL to the ownership refusal (spec §11.12): a FOREIGN-owner record with a
        conflicting class still NULLs the row's class — its `reason` stays "foreign_owner"
        (the state refusal) while a same-owner conflict reports "class_conflict".

        ONE result per INPUT record, in the input order — callers zip the two lists to emit a
        per-message event, so a skipped record would silently reattribute every event after it.
        The prior row is read inside the SAME transaction as the upsert; the upsert's own rowcount
        is what decides `applied`, so the conflict rule stays the single arbiter.
        """
        given = list(records)
        if not any(ts for ts, _root, _cls in given):
            return [TransitionResult(False, None, None, "no_message_ts") for _ in given]
        for _ts, _root, cls in given:
            _check_receipt_class(cls)
        results: List[TransitionResult] = []
        async with self._stream_conn() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                for ts, root, cls in given:
                    if not ts:
                        results.append(TransitionResult(False, None, None, "no_message_ts"))
                        continue
                    message_ts = str(ts)
                    async with db.execute(
                        "SELECT state, turn_id, receipt_class FROM outbound_receipts "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                        (team_id, channel_id, message_ts)
                    ) as cursor:
                        prior = await cursor.fetchone()
                    cursor = await db.execute(
                        """
                        INSERT INTO outbound_receipts
                            (team_id, channel_id, message_ts, turn_id, state, thread_root_ts,
                             receipt_class, created_ts, finalized_ts)
                        VALUES (?, ?, ?, ?, 'finalized', ?, ?,
                                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT(team_id, channel_id, message_ts) DO UPDATE SET
                            state = 'finalized',
                            finalized_ts = COALESCE(outbound_receipts.finalized_ts,
                                                    CURRENT_TIMESTAMP),
                            thread_root_ts = COALESCE(excluded.thread_root_ts,
                                                      outbound_receipts.thread_root_ts),
                            receipt_class = outbound_receipts.receipt_class
                        WHERE outbound_receipts.turn_id = excluded.turn_id
                          AND (outbound_receipts.receipt_class IS NULL
                               OR excluded.receipt_class IS NULL
                               OR outbound_receipts.receipt_class = excluded.receipt_class)
                        """,
                        (team_id, channel_id, message_ts, turn_id, root, cls))
                    applied = bool(cursor.rowcount)
                    prior_state = prior["state"] if prior is not None else "absent"
                    if not applied:
                        if (prior is not None
                                and cls is not None and prior["receipt_class"] is not None
                                and prior["receipt_class"] != cls):
                            # Spec §11.1/§11.12: the conflict fails closed to NULL — the state
                            # stays where it was, and the class lands in the ineligible
                            # terminal state so nothing editable can come out of the
                            # disagreement. Ownership does NOT shield the class: a foreign
                            # owner's conflicting claim NULLs it too (class arbitration is
                            # orthogonal to the state refusal, which keeps its own reason).
                            await db.execute(
                                "UPDATE outbound_receipts SET receipt_class = NULL "
                                "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                                (team_id, channel_id, message_ts))
                            self.log_error(
                                f"Receipt {channel_id}/{message_ts} class conflict at "
                                f"finalize: row was {prior['receipt_class']!r}, {turn_id} "
                                f"claimed {cls!r} — class NULLed (ineligible), record refused")
                            same_owner = prior["turn_id"] == turn_id
                            results.append(TransitionResult(
                                False, prior_state, prior_state,
                                "class_conflict" if same_owner else "foreign_owner"))
                            continue
                        results.append(TransitionResult(
                            False, prior_state, prior_state, "foreign_owner"))
                    elif prior is None:
                        results.append(TransitionResult(
                            True, "absent", "finalized", "inserted"))
                    elif prior_state == "finalized":
                        results.append(TransitionResult(
                            True, "finalized", "finalized", "already_finalized"))
                    else:
                        results.append(TransitionResult(
                            True, prior_state, "finalized", "finalized"))
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        written = sum(1 for r in results if r.applied)
        self.log_debug(
            f"Receipts finalized for {turn_id} in {channel_id}: {written}/{len(results)}")
        return results

    async def demote_receipt_chrome_async(self, team_id: str, channel_id: str, message_ts: str,
                                          turn_id: str) -> TransitionResult:
        """in_flight → chrome, same owner only. The only legal way down.

        For a surface whose conversational content was overwritten by scaffolding (a
        "Retrying without…" notice replacing a partial answer).

        A refusal names no prior state: the guarded UPDATE never read one, and guessing
        `in_flight` for a row that was chrome, finalized or foreign would put an invented
        transition on the record.

        The retry demotion maps the CLASS back too (spec §4): the surface stopped carrying an
        answer, so whatever class the promotion stamped reverts to `chrome`, atomically with
        the state — but only when a class was ever stamped. A legacy NULL stays NULL: the
        demotion is a mapping-back, never a first stamping.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "UPDATE outbound_receipts SET state = 'chrome', "
                "  receipt_class = CASE WHEN receipt_class IS NULL THEN NULL "
                "                       ELSE 'chrome' END "
                "WHERE team_id = ? AND channel_id = ? AND message_ts = ? "
                "  AND turn_id = ? AND state = 'in_flight'",
                (team_id, channel_id, message_ts, turn_id))
            demoted = bool(cursor.rowcount)
        if demoted:
            self.log_debug(f"Receipt {channel_id}/{message_ts} in_flight→chrome ({turn_id})")
            return TransitionResult(True, "in_flight", "chrome", "demoted")
        return TransitionResult(False, None, None, "not_in_flight_or_foreign")

    async def delete_receipt_async(self, team_id: str, channel_id: str,
                                   message_ts: str) -> TransitionResult:
        """Drop a receipt. Callers must have CONFIRMED the Slack deletion first.

        The read and the delete share ONE transaction, so the state the event reports is the
        state this call actually removed rather than whatever a separate read happened to see.
        """
        key = (team_id, channel_id, message_ts)
        async with self._stream_conn() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT state FROM outbound_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND message_ts = ?", key
                ) as cursor:
                    row = await cursor.fetchone()
                cursor = await db.execute(
                    "DELETE FROM outbound_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND message_ts = ?", key)
                deleted = bool(cursor.rowcount)
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        prior = row["state"] if row is not None else "absent"
        return TransitionResult(deleted, prior, "absent",
                                "deleted" if deleted else "no_row")

    async def get_channel_receipts_async(self, team_id: str, channel_id: str) -> List[Dict]:
        """Every receipt for one channel, oldest first."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT team_id, channel_id, message_ts, turn_id, state, thread_root_ts, "
                "       receipt_class, created_ts, finalized_ts FROM outbound_receipts "
                "WHERE team_id = ? AND channel_id = ? ORDER BY CAST(message_ts AS REAL)",
                (team_id, channel_id)
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def get_receipt_async(self, team_id: str, channel_id: str,
                                message_ts: str) -> Optional[Dict]:
        """One receipt row, or None."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT team_id, channel_id, message_ts, turn_id, state, thread_root_ts, "
                "       receipt_class, created_ts, finalized_ts FROM outbound_receipts "
                "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                (team_id, channel_id, message_ts)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def finalize_dead_session_receipts_async(self, live_session_id: str) -> List[Dict]:
        """Boot reconciliation: finalize every in_flight row owned by a DEAD session.

        Whatever Slack holds for those messages IS the final content — the process that could
        still have edited them is gone. Matching on the "{session}:" prefix with substr rather
        than LIKE keeps the comparison exact (no wildcard escaping).

        Returns the rows it moved, identified, so the caller can put one event per recovered
        message on the record; a count could only ever say that some existed. The SELECT shares
        the UPDATE's transaction, so the list is exactly what this call finalized.
        """
        prefix = f"{live_session_id}:"
        predicate = "state = 'in_flight' AND substr(turn_id, 1, ?) <> ?"
        params = (len(prefix), prefix)
        async with self._stream_conn() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT team_id, channel_id, message_ts, turn_id FROM outbound_receipts "
                    f"WHERE {predicate}", params
                ) as cursor:
                    moved = [dict(row) for row in await cursor.fetchall()]
                await db.execute(
                    "UPDATE outbound_receipts "
                    "SET state = 'finalized', finalized_ts = CURRENT_TIMESTAMP "
                    f"WHERE {predicate}", params)
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        if moved:
            self.log_info(
                f"Receipts: finalized {len(moved)} in_flight row(s) from dead sessions")
        return moved

    # --- pending share receipts ---------------------------------------------------------

    async def record_pending_share_async(self, team_id: str, channel_id: str, file_id: str,
                                         owner_turn_id: str,
                                         thread_root_ts: Optional[str] = None, *,
                                         receipt_class: Optional[str]) -> bool:
        """Record an uploaded file whose share ts is not known yet. First writer owns it.

        `receipt_class` rides the pending row end-to-end (spec §4): the resolve copies it into
        the receipt, so an artifact share finalizes WITH its class present, never inferred.

        A SECOND claim for the same file is refused (first writer owns it) — and when the
        second claim names a DIFFERENT class, that conflict is detected and ERROR-logged
        (spec §11.1); the first claim stands untouched."""
        _check_receipt_class(receipt_class)
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "INSERT OR IGNORE INTO pending_share_receipts "
                "(team_id, channel_id, file_id, owner_turn_id, thread_root_ts, receipt_class) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (team_id, channel_id, file_id, owner_turn_id, thread_root_ts, receipt_class))
            claimed = bool(cursor.rowcount)
            if not claimed:
                async with db.execute(
                    "SELECT owner_turn_id, receipt_class FROM pending_share_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND file_id = ?",
                    (team_id, channel_id, file_id)
                ) as read:
                    prior = await read.fetchone()
                if (prior is not None and receipt_class is not None
                        and prior["receipt_class"] is not None
                        and prior["receipt_class"] != receipt_class):
                    self.log_error(
                        f"Pending share {file_id} second claim class conflict: first claim "
                        f"({prior['owner_turn_id']}) carries {prior['receipt_class']!r}, "
                        f"{owner_turn_id} claimed {receipt_class!r} — the first claim stands")
            return claimed

    async def resolve_pending_share_async(self, team_id: str, channel_id: str, file_id: str,
                                          message_ts: str) -> bool:
        """Finalize the share's receipt and drop the pending row in ONE transaction.

        Crash-atomic on purpose: a finalize that committed without the delete would replay on
        the next boot, and a delete that committed without the finalize would strand the share
        outside the stream forever. Idempotent — an already-resolved file is a no-op success.
        """
        async with self._stream_conn() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT owner_turn_id, thread_root_ts, receipt_class "
                    "FROM pending_share_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND file_id = ?",
                    (team_id, channel_id, file_id)
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await db.execute("COMMIT")
                    return True

                # The class the pending row carried resolves WITH the share (spec §4). If the
                # ts already carries a DIFFERENT class, the conflict fails closed to NULL
                # (spec §11.1) — the ineligible terminal state, ERROR-logged with both classes.
                # The STATE resolution itself still lands, and the pending row is still
                # deleted below (terminal — refusing the resolve would retry the same
                # conflicted row on every boot forever). An EXISTING row's class is NEVER
                # stamped by the resolve (spec §11.12): the upsert keeps whatever the row
                # holds — a NULL (legacy or conflict-poisoned) stays NULL, a matching class
                # is unchanged — and only a row this resolve CREATES carries the pending class.
                async with db.execute(
                    "SELECT receipt_class FROM outbound_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                    (team_id, channel_id, str(message_ts))
                ) as cursor:
                    existing = await cursor.fetchone()
                conflicted = (existing is not None and existing["receipt_class"] is not None
                              and row["receipt_class"] is not None
                              and existing["receipt_class"] != row["receipt_class"])
                if conflicted:
                    self.log_error(
                        f"Pending share {file_id} class conflict at resolve: receipt "
                        f"{channel_id}/{message_ts} is {existing['receipt_class']!r}, pending "
                        f"row carried {row['receipt_class']!r} — class NULLed (ineligible), "
                        f"resolution still lands")

                await db.execute(
                    """
                    INSERT INTO outbound_receipts
                        (team_id, channel_id, message_ts, turn_id, state, thread_root_ts,
                         receipt_class, created_ts, finalized_ts)
                    VALUES (?, ?, ?, ?, 'finalized', ?, ?,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(team_id, channel_id, message_ts) DO UPDATE SET
                        state = 'finalized',
                        finalized_ts = COALESCE(outbound_receipts.finalized_ts, CURRENT_TIMESTAMP),
                        thread_root_ts = COALESCE(excluded.thread_root_ts,
                                                  outbound_receipts.thread_root_ts),
                        receipt_class = outbound_receipts.receipt_class
                    """,
                    (team_id, channel_id, str(message_ts), row["owner_turn_id"],
                     row["thread_root_ts"], row["receipt_class"]))
                if conflicted:
                    # After the state resolution so the outcome is unambiguous whatever the
                    # upsert kept: the conflicted class ends NULL, in the same transaction.
                    await db.execute(
                        "UPDATE outbound_receipts SET receipt_class = NULL "
                        "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                        (team_id, channel_id, str(message_ts)))
                await db.execute(
                    "DELETE FROM pending_share_receipts "
                    "WHERE team_id = ? AND channel_id = ? AND file_id = ?",
                    (team_id, channel_id, file_id))
                await db.execute("COMMIT")
                self.log_debug(
                    f"Pending share {file_id} resolved to {channel_id}/{message_ts} "
                    f"({row['owner_turn_id']})")
                return True
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    async def delete_pending_share_async(self, team_id: str, channel_id: str,
                                         file_id: str) -> bool:
        """Drop a pending share. ONLY after Slack confirms the file/share is gone.

        A resolution failure — auth error, file_not_found race, timeout, exhausted polling —
        must RETAIN the row so boot recovery can retry it.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "DELETE FROM pending_share_receipts "
                "WHERE team_id = ? AND channel_id = ? AND file_id = ?",
                (team_id, channel_id, file_id))
            return bool(cursor.rowcount)

    async def get_pending_shares_async(self, team_id: Optional[str] = None) -> List[Dict]:
        """Unresolved shares, oldest first (boot recovery reads them all)."""
        sql = ("SELECT team_id, channel_id, file_id, owner_turn_id, thread_root_ts, "
               "receipt_class, created_ts FROM pending_share_receipts")
        params: Tuple = ()
        if team_id:
            sql += " WHERE team_id = ?"
            params = (team_id,)
        sql += " ORDER BY created_ts"
        async with self._stream_conn() as db:
            async with db.execute(sql, params) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    # --- thread-activity index (spec §4) -------------------------------------------------

    async def record_thread_activity_async(self, team_id: str, channel_id: str, root_ts: str,
                                           reply_ts: Optional[str] = None,
                                           reply_count: Optional[int] = None,
                                           event_ts: Optional[str] = None,
                                           mark_dirty: bool = False) -> None:
        """Idempotent observation of activity under one root. Safe to call twice per event.

        Both ts columns only ever move forward, and the advisory count only moves on an
        observation at least as recent as the last one indexed and never downward — a reply
        count that arrives out of order must not talk the index out of a fetch. A count with no
        latest_reply to go with it marks the root dirty: we know there are replies but not where.
        """
        params = _activity_upsert_params(team_id, channel_id, root_ts, reply_ts, reply_count,
                                         event_ts, mark_dirty)
        async with self._stream_conn() as db:
            await db.execute(_ACTIVITY_UPSERT_SQL, params)

    async def clear_thread_dirty_async(self, team_id: str, channel_id: str, root_ts: str,
                                       if_event_ts_equals: Optional[str]) -> bool:
        """Compare-and-clear: only clears if no newer event landed since the reader looked.

        `IS` compares NULL-safely, so a row that has never carried an event ts clears against
        an explicit None.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "UPDATE channel_thread_activity SET dirty = 0, updated_ts = CURRENT_TIMESTAMP "
                "WHERE team_id = ? AND channel_id = ? AND root_ts = ? "
                "  AND last_index_event_ts IS ?",
                (team_id, channel_id, str(root_ts),
                 str(if_event_ts_equals) if if_event_ts_equals else None))
            return bool(cursor.rowcount)

    async def delete_thread_activity_if_unchanged_async(
            self, team_id: str, channel_id: str, root_ts: str,
            if_event_ts_equals: Optional[str]) -> bool:
        """Compare-and-DELETE one activity row. Returns True when the row actually went.

        The mirror of compare-and-clear above, for a root the fan-out proved is GONE from Slack:
        `conversations.replies` answered `thread_not_found` and nothing visible vouches for it.
        Without the delete the dead row is exempt from the floor, re-fires the same refused fetch
        on every turn, and the channel can never build again — which is not a hypothetical, it is
        the state that took #dev-ops down.

        THE COMPARE IS ON `COALESCE(last_index_event_ts, last_observed_reply_ts)`, which is the
        value READ 1 pinned in `activity_roots` — not on `last_index_event_ts` alone, which is
        what compare-and-clear uses and would miss a row whose only timestamp is a reply
        observation. If it MOVED, something learned about this root after the pin and the row
        stays for the next turn to reassess.

        `IS` compares NULL-safely, so a bootstrap hint carrying no timestamp at all deletes
        against an explicit None. The delete is recoverable by construction: any future event on
        the root re-creates the row through the index.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "DELETE FROM channel_thread_activity "
                "WHERE team_id = ? AND channel_id = ? AND root_ts = ? "
                "  AND COALESCE(last_index_event_ts, last_observed_reply_ts) IS ?",
                (team_id, channel_id, str(root_ts),
                 str(if_event_ts_equals) if if_event_ts_equals else None))
            return bool(cursor.rowcount)

    async def get_thread_activity_async(self, team_id: str, channel_id: str,
                                        since_ts: Optional[str] = None) -> List[Dict]:
        """Roots with activity after `since_ts`, PLUS every dirty root regardless of ts.

        A dirty root's activity is an edit or a deletion, whose position in time says nothing
        about where the mutated message sits — it always comes back until it is cleared.
        """
        async with self._stream_conn() as db:
            async with db.execute(
                """
                SELECT team_id, channel_id, root_ts, last_observed_reply_ts,
                       advisory_reply_count, last_index_event_ts, dirty, updated_ts
                FROM channel_thread_activity
                WHERE team_id = :team AND channel_id = :ch
                  AND (:since IS NULL
                       OR dirty = 1
                       OR (last_observed_reply_ts IS NOT NULL
                           AND CAST(last_observed_reply_ts AS REAL) > CAST(:since AS REAL))
                       OR (last_index_event_ts IS NOT NULL
                           AND CAST(last_index_event_ts AS REAL) > CAST(:since AS REAL)))
                ORDER BY CAST(root_ts AS REAL)
                """,
                {"team": team_id, "ch": channel_id,
                 "since": str(since_ts) if since_ts else None}
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def thread_activity_exists_async(self, team_id: str, channel_id: str,
                                           root_ts: str) -> bool:
        """Is this ts already known to the index as a root? Primary-key probe.

        Slack roots often advertise threading only through `reply_count`/`latest_reply`, and an
        edit or deletion payload may carry neither — the index's own memory of the root is then
        the only evidence that a plain-looking top-level ts is actually a thread parent.
        """
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT 1 FROM channel_thread_activity "
                "WHERE team_id = ? AND channel_id = ? AND root_ts = ? LIMIT 1",
                (team_id, channel_id, str(root_ts))
            ) as cursor:
                return await cursor.fetchone() is not None

    # --- coverage (spec §4) --------------------------------------------------------------

    async def seed_channel_coverage_async(self, team_id: str, channel_id: str,
                                          start_ts: str) -> bool:
        """Create the coverage row with a concrete horizon. Never moves an existing one."""
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "INSERT OR IGNORE INTO channel_coverage "
                "(team_id, channel_id, inventory_start_ts, bootstrap_status) "
                "VALUES (?, ?, ?, 'pending')",
                (team_id, channel_id, str(start_ts)))
            return bool(cursor.rowcount)

    async def acquire_coverage_sweep_async(self, team_id: str, channel_id: str,
                                           token: str) -> bool:
        """Claim the sweep for one channel. One worker at a time, terminal rows never again.

        A claim is takeable when nobody holds it or the holder stopped heartbeating
        (>10 min). `complete`/`limited` rows are finished — recompaction and refresh are not
        P1 — so the predicate excludes them and a claim can never be resurrected by a restart.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                f"""
                UPDATE channel_coverage
                SET sweep_token = ?, bootstrap_status = 'running',
                    heartbeat_ts = CURRENT_TIMESTAMP, updated_ts = CURRENT_TIMESTAMP
                WHERE team_id = ? AND channel_id = ?
                  AND bootstrap_status IN ('pending', 'running')
                  AND (sweep_token IS NULL
                       OR heartbeat_ts IS NULL
                       OR heartbeat_ts
                          < datetime('now', '-{_COVERAGE_SWEEP_STALE_MINUTES} minutes'))
                """,
                (token, team_id, channel_id))
            return bool(cursor.rowcount)

    async def heartbeat_coverage_sweep_async(self, team_id: str, channel_id: str,
                                             token: str) -> bool:
        """Prove the sweep holder is alive without advancing coverage.

        A worker parked on a page ceiling or a Retry-After sleep still holds its claim; without
        this its heartbeat would go stale and another worker would take the channel from it.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                "UPDATE channel_coverage SET heartbeat_ts = CURRENT_TIMESTAMP "
                "WHERE team_id = ? AND channel_id = ? AND sweep_token = ?",
                (team_id, channel_id, token))
            return bool(cursor.rowcount)

    async def advance_channel_coverage_async(self, team_id: str, channel_id: str, token: str,
                                             new_start_ts: Optional[str], status: str,
                                             reason: Optional[str] = None) -> bool:
        """Extend coverage backward and/or set the bootstrap status. Token-guarded.

        inventory_start_ts is the resume point, so it only ever moves BACKWARD and only for a
        page that was fully processed. A terminal status (`complete`/`limited`) is never talked
        back down to `running` — a channel that reached genesis or Slack's retention wall stays
        there for the rest of P1.
        """
        if status not in ("pending", "running", "complete", "limited"):
            raise ValueError(f"invalid bootstrap_status: {status!r}")
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                UPDATE channel_coverage SET
                    inventory_start_ts = CASE
                        WHEN :new_start IS NULL THEN inventory_start_ts
                        WHEN CAST(:new_start AS REAL) < CAST(inventory_start_ts AS REAL)
                            THEN :new_start
                        ELSE inventory_start_ts END,
                    bootstrap_status = CASE
                        WHEN bootstrap_status IN ('complete', 'limited')
                             AND :status NOT IN ('complete', 'limited')
                            THEN bootstrap_status
                        ELSE :status END,
                    inventory_reason = CASE
                        WHEN bootstrap_status IN ('complete', 'limited')
                             AND :status NOT IN ('complete', 'limited')
                            THEN inventory_reason
                        ELSE :reason END,
                    heartbeat_ts = CURRENT_TIMESTAMP,
                    updated_ts = CURRENT_TIMESTAMP
                WHERE team_id = :team AND channel_id = :ch AND sweep_token = :token
                """,
                {"team": team_id, "ch": channel_id, "token": token,
                 "new_start": str(new_start_ts) if new_start_ts else None,
                 "status": status, "reason": reason})
            return bool(cursor.rowcount)

    async def reset_channel_coverage_async(self, team_id: str, channel_id: str) -> bool:
        """Hand an unreachable channel back to the sweep. True when a verdict was cleared.

        `limited`/'unavailable' is the one terminal state that can reverse — the bot gets
        re-invited, the channel gets unarchived — and neither `advance` (which refuses to talk a
        terminal status down) nor `acquire` (which never looks at terminal rows) can express the
        demote. A retention wall, a depth cap and `complete` are facts about history rather than
        reachability, so the predicate excludes them and a finished channel is never re-walked.
        Untokened by design: the caller is a join event, not a claim holder, and clearing
        sweep_token here is exactly the takeover. inventory_start_ts is left alone so an
        interrupted backward walk resumes where it stopped.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                UPDATE channel_coverage
                SET bootstrap_status = 'pending', inventory_reason = NULL, sweep_token = NULL,
                    heartbeat_ts = NULL, updated_ts = CURRENT_TIMESTAMP
                WHERE team_id = ? AND channel_id = ?
                  AND bootstrap_status = 'limited' AND inventory_reason = ?
                """,
                (team_id, channel_id, _COVERAGE_UNAVAILABLE_REASON))
            return bool(cursor.rowcount)

    async def get_channel_coverage_async(self, team_id: str,
                                         channel_id: str) -> Optional[Dict]:
        """The declared horizon for one channel, or None when it has never been seeded."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT team_id, channel_id, inventory_start_ts, bootstrap_status, "
                "       inventory_reason, sweep_token, heartbeat_ts, updated_ts "
                "FROM channel_coverage WHERE team_id = ? AND channel_id = ?",
                (team_id, channel_id)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    # --- channel stream sidecars (spec §3) ------------------------------------------------
    #
    # BOUNDARY SEMANTICS, once, for everything below.
    #
    # A channel turn renders the window (floor_ts, floor_inclusive] .. H. There are exactly two
    # kinds of floor and they differ ONLY in that flag:
    #   * genesis — floor is the coverage row's `inventory_start_ts`, INCLUSIVE. That ts is the
    #     oldest message the sweep actually processed, so excluding it would drop a real message
    #     nothing else will ever show us.
    #   * an explicit floor passed by the caller, whose `floor_inclusive` flag it supplies.
    # H is always an INCLUSIVE upper bound: it is a ts this process admitted, so the message at
    # H exists and belongs in the window.
    #
    # Every comparison is numeric — `CAST(x AS REAL)` in SQL, `parse_ts` in Python — never a
    # string compare. Slack timestamps are not fixed-width, so "1752600000.5" > "1752600000.10"
    # lexicographically and the two orderings disagree exactly at the boundary this window is
    # defined by.

    # --- the window anchor + the two staged reads (SHALLOW_STREAM_RESPEC §4.2, §4.6) -------

    async def advance_channel_window_anchor_async(
            self, team_id: str, channel_id: str, floor_ts: str,
            selection_version: int) -> bool:
        """Move the periphery floor forward. True when the row actually moved.

        THREE version rules, and the middle one is the whole point:

          * HIGHER incoming version -> overwrite unconditionally (a policy change resets the
            floor, which is what SELECTION_VERSION exists to do);
          * EQUAL version           -> the floor may only move FORWARD;
          * LOWER incoming version  -> REJECTED outright.

        The lower-version rejection is what stops an old process — a rolling restart, a straggler
        turn holding a pre-upgrade pin — from overwriting a newer-version row and dragging the
        floor back to a policy that is no longer in force. ONE statement, so two concurrent turns
        converge without either reading the other's write first.

        The empty-floor sentinel never reaches here: a build with no floor writes no row.
        """
        overlay = _epoch_overlay(channel_id, team_id)
        if overlay is not None:
            # SELECTION state, and the easiest to miss: every build advances it, so a battery
            # would leave the channel's floor wherever the last case left it and the NEXT
            # battery's first turn would render a window shaped by the previous run.
            store, key = overlay
            return store.advance_window_anchor(key, floor_ts, selection_version)
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO channel_window_anchor
                    (team_id, channel_id, floor_ts, selection_version, updated_ts)
                VALUES (:team, :ch, :floor, :ver, CURRENT_TIMESTAMP)
                ON CONFLICT(team_id, channel_id) DO UPDATE SET
                    floor_ts          = excluded.floor_ts,
                    selection_version = excluded.selection_version,
                    updated_ts        = CURRENT_TIMESTAMP
                WHERE excluded.selection_version > channel_window_anchor.selection_version
                   OR (excluded.selection_version = channel_window_anchor.selection_version
                       AND CAST(excluded.floor_ts AS REAL)
                           > CAST(channel_window_anchor.floor_ts AS REAL))
                """,
                {"team": team_id, "ch": channel_id, "floor": str(floor_ts),
                 "ver": int(selection_version)})
            return bool(cursor.rowcount)

    async def read_channel_window_anchor_async(self, team_id: str,
                                               channel_id: str) -> Dict[str, Any]:
        """READ 1 STAGE 1. Anchor + inventory, and BOTH ARE PINNED.

        One transaction returning exactly two things:
          * the window anchor row -> {"floor_ts", "selection_version"} or None
          * the inventory row     -> {"inventory_start_ts", "bootstrap_status", "reason"} or
            **None when the row is ABSENT**

        BOTH RENDER. The anchor becomes the periphery floor, which is pre-breakpoint bytes; the
        inventory state becomes the horizon's index clause. They are frozen into the pin at the
        moment they are read and are NEVER re-read for the rest of the turn.

        It reads NO activity and NO receipt rows: those need a window, and the window is not
        known until the history walk has run. Stage 2 below owns them.
        """
        payload: Dict[str, Any] = {"anchor": None, "inventory": None}
        # Under a fence the ANCHOR comes from the overlay and the INVENTORY still comes from
        # production: the inventory records an observable fact — how far the index has actually
        # walked this channel — while the anchor decides what the next run's window contains.
        # The overlay's anchor starts as None, so a fenced channel's first turn takes the cold
        # path and derives its own floor instead of inheriting whatever the room was doing before.
        overlay = _epoch_overlay(channel_id, team_id)
        async with self._stream_conn() as db:
            await db.execute("BEGIN DEFERRED")
            try:
                if overlay is not None:
                    store, key = overlay
                    payload["anchor"] = store.window_anchor(key)
                else:
                    async with db.execute(
                        "SELECT floor_ts, selection_version FROM channel_window_anchor "
                        "WHERE team_id = ? AND channel_id = ?",
                        (team_id, channel_id)
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is not None:
                        payload["anchor"] = {"floor_ts": row["floor_ts"],
                                             "selection_version": int(row["selection_version"])}

                async with db.execute(
                    "SELECT inventory_start_ts, bootstrap_status, inventory_reason "
                    "FROM channel_coverage WHERE team_id = ? AND channel_id = ?",
                    (team_id, channel_id)
                ) as cursor:
                    row = await cursor.fetchone()
                if row is not None:
                    payload["inventory"] = {
                        "inventory_start_ts": row["inventory_start_ts"],
                        "bootstrap_status": row["bootstrap_status"],
                        "reason": row["inventory_reason"],
                    }
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        return payload

    async def read_channel_discovery_roots_async(self, team_id: str, channel_id: str, *,
                                                 floor_ts: str,
                                                 high_ts: str) -> Dict[str, Any]:
        """READ 1 STAGE 2. DISCOVERY ONLY — nothing here is ever rendered.

        Returns `{"activity_roots": {root_ts: pinned event ts | None},
                  "receipt_roots": (root_ts, ...)}`.

        **`activity_roots` IS A MAPPING, not a set**, and the value is what compare-and-clear
        compares against: the event ts THIS read pinned for that root
        (`COALESCE(last_index_event_ts, last_observed_reply_ts)`, None for a bootstrap
        reply-count hint with no event ts). A root's dirty flag is cleared only when the stored
        value still equals the one pinned here — a root whose ts MOVED saw a mutation this fetch
        did not, and must stay dirty. A set could not express that and the clear would be unsafe.

        `floor_ts` is the EFFECTIVE floor — the stored floor only when the history walk actually
        reached it, otherwise the walk's early-stop floor. Windowing on the STORED floor would
        let a stale anchor pull months of roots out of the index and into reply fan-out, which is
        the deep-walk failure this design replaces, reached through the database instead of
        through Slack.

        Staleness here costs at most one redundant reply fetch, or one reply the next turn picks
        up. It can never produce wrong bytes — which is what makes it discovery and not a pin.
        """
        bounds = {"team": team_id, "ch": channel_id, "floor": str(floor_ts),
                  "high": str(high_ts)}

        def _window_sql(column: str) -> str:
            return (f"CAST({column} AS REAL) >= CAST(:floor AS REAL) "
                    f"AND CAST({column} AS REAL) <= CAST(:high AS REAL)")

        activity_roots: Dict[str, Optional[str]] = {}
        receipt_roots: List[str] = []
        async with self._stream_conn() as db:
            await db.execute("BEGIN DEFERRED")
            try:
                # The landed activity predicate, verbatim: a dirty row keeps its exemption from
                # the FLOOR (an edit says nothing about where the mutated message sits, so the
                # root must come back until someone fetches it) but NOT from H — a mutation
                # above this turn's frontier cannot appear in its stream.
                async with db.execute(
                    f"""
                    SELECT root_ts, last_observed_reply_ts, last_index_event_ts
                    FROM channel_thread_activity
                    WHERE team_id = :team AND channel_id = :ch
                      AND ((dirty = 1
                            AND (COALESCE(last_index_event_ts, last_observed_reply_ts) IS NULL
                                 OR CAST(COALESCE(last_index_event_ts,
                                                  last_observed_reply_ts) AS REAL)
                                    <= CAST(:high AS REAL)))
                           OR (last_observed_reply_ts IS NOT NULL
                               AND {_window_sql('last_observed_reply_ts')})
                           OR (last_index_event_ts IS NOT NULL
                               AND {_window_sql('last_index_event_ts')}))
                    ORDER BY CAST(root_ts AS REAL)
                    """,
                    bounds
                ) as cursor:
                    for row in await cursor.fetchall():
                        pinned = row["last_index_event_ts"] or row["last_observed_reply_ts"]
                        activity_roots[str(row["root_ts"])] = (
                            str(pinned) if pinned is not None else None)

                # Our own posts name the threads we are already part of. Predicated into the
                # SAME window as everything else, by the ts of the message each receipt
                # describes (FX1 finding F3, preserved).
                async with db.execute(
                    f"""
                    SELECT DISTINCT thread_root_ts
                    FROM outbound_receipts
                    WHERE team_id = :team AND channel_id = :ch
                      AND thread_root_ts IS NOT NULL
                      AND {_window_sql('message_ts')}
                    ORDER BY CAST(thread_root_ts AS REAL)
                    """,
                    bounds
                ) as cursor:
                    receipt_roots = [str(r["thread_root_ts"]) for r in await cursor.fetchall()]
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        return {"activity_roots": activity_roots, "receipt_roots": tuple(receipt_roots)}

    async def read_channel_sidecars_for_async(self, team_id: str, channel_id: str,
                                              message_ts: Sequence[str],
                                              busy_timeout_ms: Optional[int] = None
                                              ) -> Dict[str, Any]:
        """READ 2. THE RENDER PIN: one transaction, EXACT identities, nothing else.

        Replaces the `(floor, high]` window form on the turn path. The window form dragged in
        every artifact row between a possibly-ancient origin root and H; the identities are what
        the build actually renders.

        ITS SUBJECT IS CANDIDATES, NOT SELECTIONS: every normalized periphery candidate plus
        every normalized origin message. Eligibility needs receipt state and the receipt epoch,
        which live here, so a pin restricted to already-selected ids could never be built at all.

        The builder calls it TWICE — once over the periphery candidate ids, once over the
        origin-only ids — with the shared call's rows winning any overlap. Each call is still ONE
        transaction over an exact id list, which is the property this signature exists to give.

        The `IN (...)` lists are CHUNKED at 500 ids per statement inside the one transaction, so
        SQLite's variable limit is never the thing that decides a turn.

        `busy_timeout_ms` is for a caller under a deadline of its own — the search scan reads
        receipt evidence with whatever is left of its fetch budget, and a five-second lock wait
        would outlive it. Omitted, the house timeout applies and the turn path is unchanged.
        """
        ids = sorted({str(ts) for ts in (message_ts or ()) if ts})
        payload: Dict[str, Any] = {
            "ids": ids, "receipt_feature_epoch_ts": None, "receipts": [],
            "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
            "tool_usage": {}, "versions_hash": "",
        }

        def _chunks() -> List[List[str]]:
            return [ids[i:i + _SIDECAR_ID_CHUNK] for i in range(0, len(ids), _SIDECAR_ID_CHUNK)]

        async with self._stream_conn(busy_timeout_ms=busy_timeout_ms) as db:
            await db.execute("BEGIN DEFERRED")
            try:
                async with db.execute(
                    "SELECT value FROM bot_meta WHERE key = ?",
                    (OUTBOUND_RECEIPTS_EPOCH_KEY,)
                ) as cursor:
                    row = await cursor.fetchone()
                payload["receipt_feature_epoch_ts"] = row["value"] if row else None

                if ids:
                    thread_prefix = f"{channel_id}:%"
                    receipts: List[Dict] = []
                    images: List[Dict] = []
                    documents: List[Dict] = []
                    ambient: List[Dict] = []
                    usage: Dict[str, List[Dict]] = {}
                    for chunk in _chunks():
                        marks = ",".join("?" * len(chunk))

                        async with db.execute(
                            f"SELECT message_ts, turn_id, state, thread_root_ts, receipt_class "
                            f"FROM outbound_receipts "
                            f"WHERE team_id = ? AND channel_id = ? AND message_ts IN ({marks}) "
                            f"ORDER BY CAST(message_ts AS REAL)",
                            (team_id, channel_id, *chunk)
                        ) as cursor:
                            receipts.extend(dict(r) for r in await cursor.fetchall())

                        async with db.execute(
                            f"SELECT id, thread_id, message_ts, url, image_type, analysis, "
                            f"       metadata_json "
                            f"FROM images "
                            f"WHERE thread_id LIKE ? AND message_ts IN ({marks}) "
                            f"ORDER BY CAST(message_ts AS REAL), id",
                            (thread_prefix, *chunk)
                        ) as cursor:
                            for r in await cursor.fetchall():
                                row_dict = dict(r)
                                raw = row_dict.pop("metadata_json", None)
                                try:
                                    row_dict["metadata"] = json.loads(raw) if raw else None
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    row_dict["metadata"] = None
                                images.append(row_dict)

                        async with db.execute(
                            f"SELECT id, thread_id, message_ts, filename, mime_type, file_id, "
                            f"       summary "
                            f"FROM documents "
                            f"WHERE thread_id LIKE ? AND message_ts IN ({marks}) "
                            f"ORDER BY CAST(message_ts AS REAL), id",
                            (thread_prefix, *chunk)
                        ) as cursor:
                            documents.extend(dict(r) for r in await cursor.fetchall())

                        async with db.execute(
                            f"SELECT id, source_ts, conversation_ts, kind, ref, title, summary, "
                            f"       status, derivation_source "
                            f"FROM ambient_artifacts "
                            f"WHERE channel_id = ? AND source_ts IN ({marks}) "
                            f"ORDER BY CAST(source_ts AS REAL), id",
                            (channel_id, *chunk)
                        ) as cursor:
                            ambient.extend(dict(r) for r in await cursor.fetchall())

                        async with db.execute(
                            f"SELECT message_ts, tools_json FROM message_tool_usage "
                            f"WHERE channel_id = ? AND message_ts IN ({marks}) "
                            f"ORDER BY CAST(message_ts AS REAL), id",
                            (channel_id, *chunk)
                        ) as cursor:
                            for r in await cursor.fetchall():
                                try:
                                    parsed = json.loads(r["tools_json"])
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    continue
                                if isinstance(parsed, list):
                                    usage[str(r["message_ts"])] = parsed

                    # Chunking splits the ORDER BY, so the merged lists are re-sorted into the
                    # accessor's own order — a single-chunk read and a five-chunk read over the
                    # same ids must be byte-comparable.
                    payload["receipts"] = sorted(receipts, key=lambda r: _ts_sort(r["message_ts"]))
                    payload["image_analyses"] = sorted(
                        images, key=lambda r: (_ts_sort(r["message_ts"]), r["id"]))
                    payload["document_extractions"] = sorted(
                        documents, key=lambda r: (_ts_sort(r["message_ts"]), r["id"]))
                    payload["ambient_artifacts"] = sorted(
                        ambient, key=lambda r: (_ts_sort(r["source_ts"]), r["id"]))
                    payload["tool_usage"] = usage
                await db.execute("COMMIT")
            except Exception:
                try:
                    await db.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        payload["versions_hash"] = self._sidecar_versions_hash(payload)
        return payload

    @staticmethod
    def _sidecar_versions_hash(payload: Dict) -> str:
        """SHA-256 over the EXACT render-relevant contents of a sidecar read.

        Row ids, timestamps of insertion and anything else the serializer never renders are
        excluded on purpose: the hash exists so a turn whose rendered bytes changed is a cache
        miss, and a hash that moved because a row was touched without changing what it renders
        would throw away every cache hit for nothing.
        """
        # THE PINNED GRAMMAR: the epoch, the id list, and the rows that actually render.
        # `coverage`, `window` and `activity` are GONE — all three were discovery, and discovery
        # moved to READ 1. They happened to be inert in this payload, so keeping them changed no
        # hash value; but a hash whose material includes fields its own contract removed is one
        # edit away from meaning something nobody intended, and "inert today" is not a property
        # anyone should have to re-verify to read this function.
        material = {
            "epoch": payload.get("receipt_feature_epoch_ts"),
            # The read carries its subject: "these rows, for these messages".
            "ids": list(payload.get("ids") or []),
            # receipt_class is render-relevant material: it decides §2a edit authorization for
            # a rendered own-message, so a class change must be a cache miss.
            "receipts": [[r.get("message_ts"), r.get("state"), r.get("turn_id"),
                          r.get("thread_root_ts"), r.get("receipt_class")]
                         for r in payload.get("receipts") or []],
            "images": [[r.get("message_ts"), r.get("url"), r.get("analysis"),
                        (r.get("metadata") or {}).get("filename")
                        if isinstance(r.get("metadata"), dict) else None]
                       for r in payload.get("image_analyses") or []],
            "documents": [[r.get("message_ts"), r.get("filename"), r.get("file_id"),
                           r.get("summary")]
                          for r in payload.get("document_extractions") or []],
            "ambient": [[r.get("source_ts"), r.get("kind"), r.get("ref"), r.get("status"),
                         r.get("derivation_source"), r.get("title"), r.get("summary")]
                        for r in payload.get("ambient_artifacts") or []],
            "tools": sorted((ts, json.dumps(tools, sort_keys=True))
                            for ts, tools in (payload.get("tool_usage") or {}).items()),
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def save_document_if_absent_async(self, thread_id: str, filename: str, mime_type: str,
                                            *, summary: Optional[str] = None,
                                            file_id: Optional[str] = None,
                                            url_private: Optional[str] = None,
                                            size_bytes: Optional[int] = None,
                                            page_structure: Optional[Dict] = None,
                                            total_pages: Optional[int] = None,
                                            metadata: Optional[Dict] = None,
                                            message_ts: Optional[str] = None) -> bool:
        """Idempotent document write for the CHANNEL origin ingester. True when it inserted.

        `save_document` upgrades a channel row in place and inserts plainly on the DM path, where
        a re-attach genuinely is a new observation. A channel turn re-ingests its origin thread on
        EVERY turn, so a plain insert would add one row per turn forever. This one leans on the
        same channel-scoped unique partial index and does nothing on conflict.

        A NULL file_id cannot conflict (SQLite treats NULLs as distinct), so those rows use
        lookup-before-insert under BEGIN IMMEDIATE instead. That lookup matches on
        (thread_id, filename, message_ts) and deliberately IGNORES file_id, so a NULL-key write
        also refuses when a row for the same share already exists WITH a Slack file id. That is
        the intended idempotency, not a near-miss: the two rows would describe the same file on
        the same message, and the one that already knows its file id is strictly better — it is
        the one `read_document` and `mount_file` can actually reach the bytes through.
        """
        columns = (thread_id, filename, mime_type, summary, file_id, url_private, size_bytes,
                   json.dumps(page_structure) if page_structure else None, total_pages,
                   json.dumps(metadata) if metadata else None, message_ts)
        async with self._stream_conn() as db:
            if file_id is None:
                await db.execute("BEGIN IMMEDIATE")
                try:
                    async with db.execute(
                        "SELECT 1 FROM documents WHERE thread_id = ? AND filename = ? "
                        "  AND (message_ts IS ?) LIMIT 1",
                        (thread_id, filename, message_ts)
                    ) as cursor:
                        if await cursor.fetchone():
                            await db.execute("COMMIT")
                            return False
                    await db.execute(
                        "INSERT INTO documents (thread_id, filename, mime_type, summary, "
                        " file_id, url_private, size_bytes, page_structure, total_pages, "
                        " metadata_json, message_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)", columns)
                    await db.execute("COMMIT")
                    return True
                except Exception:
                    try:
                        await db.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
            cursor = await db.execute(
                "INSERT INTO documents (thread_id, filename, mime_type, summary, file_id, "
                " url_private, size_bytes, page_structure, total_pages, metadata_json, "
                " message_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                f"ON CONFLICT (thread_id, message_ts, file_id) WHERE {_CHANNEL_DOCS_PREDICATE} "
                "DO NOTHING", columns)
            return bool(cursor.rowcount)

    # MCP tool caching methods
    def save_mcp_tool(self, server_label: str, tool_name: str, description: Optional[str] = None, input_schema: Optional[str] = None):
        """
        Save or update an MCP tool in the cache.

        Args:
            server_label: MCP server label
            tool_name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON string)
        """
        try:
            self.conn.execute("""
                INSERT INTO mcp_tools (server_label, tool_name, description, input_schema)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(server_label, tool_name) DO UPDATE SET
                    description = excluded.description,
                    input_schema = excluded.input_schema,
                    last_verified = CURRENT_TIMESTAMP
            """, (server_label, tool_name, description, input_schema))
            self.conn.commit()
            self.log_debug(f"DB: Cached MCP tool {server_label}:{tool_name}")
        except Exception as e:
            self.log_error(f"DB: Error caching MCP tool: {e}", exc_info=True)

    def get_mcp_tools(self, server_label: Optional[str] = None) -> List[Dict]:
        """
        Get cached MCP tools, optionally filtered by server.

        Args:
            server_label: Optional server label to filter by

        Returns:
            List of tool dictionaries
        """
        try:
            if server_label:
                cursor = self.conn.execute("""
                    SELECT server_label, tool_name, description, input_schema,
                           discovered_at, last_verified
                    FROM mcp_tools
                    WHERE server_label = ?
                    ORDER BY server_label, tool_name
                """, (server_label,))
            else:
                cursor = self.conn.execute("""
                    SELECT server_label, tool_name, description, input_schema,
                           discovered_at, last_verified
                    FROM mcp_tools
                    ORDER BY server_label, tool_name
                """)

            tools = []
            for row in cursor.fetchall():
                tools.append({
                    'server_label': row[0],
                    'tool_name': row[1],
                    'description': row[2],
                    'input_schema': row[3],
                    'discovered_at': row[4],
                    'last_verified': row[5]
                })
            return tools
        except Exception as e:
            self.log_error(f"DB: Error retrieving MCP tools: {e}", exc_info=True)
            return []

    def clear_mcp_tools(self, server_label: Optional[str] = None):
        """
        Clear cached MCP tools, optionally for a specific server.

        Args:
            server_label: Optional server label to clear tools for (clears all if not provided)
        """
        try:
            if server_label:
                self.conn.execute("DELETE FROM mcp_tools WHERE server_label = ?", (server_label,))
                self.log_info(f"DB: Cleared MCP tools for server {server_label}")
            else:
                self.conn.execute("DELETE FROM mcp_tools")
                self.log_info("DB: Cleared all MCP tools")
            self.conn.commit()
        except Exception as e:
            self.log_error(f"DB: Error clearing MCP tools: {e}", exc_info=True)

    async def _channel_of_memory_row(self, memory_id) -> Optional[str]:
        """Which channel a `channel_memory` row belongs to. Only ever called with a fence up."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT channel_id FROM channel_memory WHERE id = ?", (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row["channel_id"] if row else None

    async def _channel_of_feedback_row(self, message_ts: str, user_id: str,
                                       source: str) -> Optional[str]:
        """Which channel a `response_feedback` row belongs to. Only ever called with a fence up."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT channel_id FROM response_feedback "
                " WHERE message_ts = ? AND user_id = ? AND source = ?",
                (message_ts, user_id, source)
            ) as cursor:
                row = await cursor.fetchone()
                return row["channel_id"] if row else None

    # --- the dev-only epoch fence lease (Docs/specs/EPOCH_FENCE_SPEC.md §7.1b) ------------
    #
    # ONE table, and it is created HERE rather than in init_schema on purpose: every method below
    # is reached only from the fence watcher, which only exists when DEV_EPOCH_FENCE_ENABLE is
    # set. A production process never calls any of them, so it never creates the table.
    #
    # All three timestamp columns are TEXT holding Slack-style ts strings, compared with
    # slack_client.normalizer.parse_ts — one comparator for every time in this system, so a lease
    # check can never disagree with a window check. They are NOT declared TIMESTAMP: SQLite gives
    # that NUMERIC affinity, and a numeric-looking ts coerced to REAL loses its exact fractional
    # representation. The ONE exception is the expiry compare below, which uses CAST(… AS REAL)
    # against a 300-second window — a coarse "has this expired" where sixth-decimal rounding
    # cannot change the answer.

    async def ensure_epoch_fence_schema_async(self) -> None:
        """Create the fence lease table. Called only by the flag-gated watcher."""
        async with self._stream_conn() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS epoch_fence_lease (
                    team_id       TEXT NOT NULL,
                    channel_id    TEXT NOT NULL,
                    lease_id      TEXT NOT NULL,
                    owner_token   TEXT NOT NULL,
                    state         TEXT NOT NULL CHECK (state IN ('armed', 'active', 'closing',
                                                                 'released', 'invalidated')),
                    applied_command_id INTEGER,
                    test_epoch_id TEXT,
                    start_ts      TEXT,
                    heartbeat_ts  TEXT NOT NULL,
                    expiry_ts     TEXT NOT NULL,
                    created_ts    TEXT NOT NULL,
                    PRIMARY KEY (team_id, channel_id)
                )
            """)

    async def acquire_epoch_lease_async(self, team_id: str, channel_id: str, lease_id: str,
                                        owner_token: str, now_ts: str, expiry_ts: str,
                                        command_id: Optional[int] = None) -> bool:
        """CAS acquire. True only when a row actually changed.

        Acquisition succeeds when ANY of three things holds, and the whole thing is ONE statement:
        no row exists (the INSERT), the stored row is 'released', or the stored expiry has passed
        whatever the state. The released case is load-bearing — without it the first battery would
        permanently burn the only allowlisted channel.

        `state='invalidated'` is NOT acquirable. It is cleared only by an explicit release, which
        moves it to 'released' and makes the second rule apply: an invalidated battery must be
        looked at, not silently overwritten by the next one.

        Acquisition lands in 'armed', never 'active' — there is no case open yet, and
        authorize_effect refuses channel work until the first advance installs one. Reacquisition
        CLEARS test_epoch_id/start_ts so a new battery never inherits the previous one's case, and
        REWRITES created_ts: a takeover mints a new lease_id, so it IS a new lease and carrying the
        displaced lease's creation time forward would misdate it.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                INSERT INTO epoch_fence_lease
                    (team_id, channel_id, lease_id, owner_token, state,
                     test_epoch_id, start_ts, heartbeat_ts, expiry_ts, created_ts,
                     applied_command_id)
                VALUES (:team, :ch, :lease, :token, 'armed', NULL, NULL, :now, :expiry, :now, :cmd)
                ON CONFLICT(team_id, channel_id) DO UPDATE SET
                    lease_id = excluded.lease_id, owner_token = excluded.owner_token,
                    state = 'armed', test_epoch_id = NULL, start_ts = NULL,
                    heartbeat_ts = excluded.heartbeat_ts, expiry_ts = excluded.expiry_ts,
                    created_ts = excluded.created_ts,
                    applied_command_id = excluded.applied_command_id
                WHERE epoch_fence_lease.state = 'released'
                   OR (epoch_fence_lease.state != 'invalidated'
                       AND CAST(epoch_fence_lease.expiry_ts AS REAL)
                           <= CAST(excluded.heartbeat_ts AS REAL))
                """,
                {"team": team_id, "ch": channel_id, "lease": lease_id, "token": owner_token,
                 "now": now_ts, "expiry": expiry_ts, "cmd": command_id})
            return bool(cursor.rowcount)

    async def heartbeat_epoch_lease_async(self, team_id: str, channel_id: str, owner_token: str,
                                          now_ts: str, expiry_ts: str) -> bool:
        """Compare-and-bump on `owner_token`. False on a token mismatch.

        It does NOT revive an EXPIRED lease: the harness must re-acquire, because the expiry is
        the dead-man's switch and reviving past it would let a harness that came back from the
        dead resume a battery whose overlays are gone.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                UPDATE epoch_fence_lease
                   SET heartbeat_ts = :now, expiry_ts = :expiry
                 WHERE team_id = :team AND channel_id = :ch
                   AND owner_token = :token
                   AND state IN ('armed', 'active', 'closing')
                   AND CAST(expiry_ts AS REAL) > CAST(:now AS REAL)
                """,
                {"team": team_id, "ch": channel_id, "token": owner_token,
                 "now": now_ts, "expiry": expiry_ts})
            return bool(cursor.rowcount)

    async def advance_epoch_subepoch_async(self, team_id: str, channel_id: str, owner_token: str,
                                           test_epoch_id: str, start_ts: str, expiry_ts: str,
                                           command_id: Optional[int] = None) -> bool:
        """Install a new case. CAS on `owner_token` AND a live, unexpired, non-invalidated row."""
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                UPDATE epoch_fence_lease
                   SET state = 'active', test_epoch_id = :case, start_ts = :start,
                       heartbeat_ts = :start, expiry_ts = :expiry,
                       applied_command_id = :cmd
                 WHERE team_id = :team AND channel_id = :ch
                   AND owner_token = :token
                   AND state IN ('armed', 'active', 'closing')
                   AND CAST(expiry_ts AS REAL) > CAST(:start AS REAL)
                """,
                {"team": team_id, "ch": channel_id, "token": owner_token, "case": test_epoch_id,
                 "start": start_ts, "expiry": expiry_ts, "cmd": command_id})
            return bool(cursor.rowcount)

    async def release_epoch_lease_async(self, team_id: str, channel_id: str, owner_token: str,
                                        command_id: Optional[int] = None) -> bool:
        """Compare-and-release. Clears the case columns so the next acquire starts clean.

        A release from 'invalidated' is the ONLY way out of that state, and it is deliberate: a
        human must look at an invalidated battery before the channel takes another one.
        """
        async with self._stream_conn() as db:
            cursor = await db.execute(
                """
                UPDATE epoch_fence_lease
                   SET state = 'released', test_epoch_id = NULL, start_ts = NULL,
                       applied_command_id = :cmd
                 WHERE team_id = :team AND channel_id = :ch AND owner_token = :token
                """,
                {"team": team_id, "ch": channel_id, "token": owner_token, "cmd": command_id})
            return bool(cursor.rowcount)

    async def read_epoch_lease_async(self, team_id: str, channel_id: str) -> Optional[Dict]:
        """The lease row, or None. THE ROW IS THE TRUTH for every replay decision."""
        async with self._stream_conn() as db:
            async with db.execute(
                "SELECT * FROM epoch_fence_lease WHERE team_id = ? AND channel_id = ?",
                (team_id, channel_id)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def invalidate_stale_epoch_leases_async(self, now_ts: str) -> List[Dict]:
        """Boot recovery: INVALIDATE every battery a restart interrupted, and return what it hit.

        The lease row is durable and the runtime state is not, so a live unexpired row at boot
        means a battery whose overlays died with the previous process. An EXPIRED row is left
        alone — it is already reacquirable and needs no human.
        """
        async with self._stream_conn() as db:
            async with db.execute(
                """
                SELECT * FROM epoch_fence_lease
                 WHERE state IN ('armed', 'active', 'closing')
                   AND CAST(expiry_ts AS REAL) > CAST(? AS REAL)
                """,
                (now_ts,)
            ) as cursor:
                rows = [dict(r) for r in await cursor.fetchall()]
            for row in rows:
                await db.execute(
                    "UPDATE epoch_fence_lease SET state = 'invalidated' "
                    " WHERE team_id = ? AND channel_id = ?",
                    (row["team_id"], row["channel_id"]))
            return rows

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed for {self.platform}")