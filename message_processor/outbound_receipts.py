"""Outbound receipts (spec §5): which of our own messages may re-enter the channel stream.

An own-bot message is admitted to the rebuilt stream ONLY with a `finalized` receipt. Chrome
(placeholders, status cards, footers, checklists) is permanently excluded, and an `in_flight`
row means "this session is still writing it". So every durable post we make into a
C/G-prefixed conversation has to say which of those it is, at the moment it is made.

Ownership: a turn owns `{SESSION_ID}:{seq}`, a detached job owns `{SESSION_ID}:job:{job_id}`,
and lifecycle posts nobody's turn made own `{SESSION_ID}:sys`. Dead-session rows finalize at
boot — whatever Slack holds for them IS the final content once the writer is gone.

The upload crash window is accepted and documented: Slack does not hand out a file id before
files_upload_v2 returns, so a crash between "upload succeeded" and "pending row written"
loses the share. Boot recovery can only retry rows that exist.

Share-ts resolution is TIME-BUDGETED, not a fixed attempt count: files.info is polled on a
backoff until `IMAGE_SHARE_TS_TIMEOUT_SECONDS` is spent (config.image_share_ts_timeout_seconds),
because a 429 or a slow share is worth another poll and an attempt cap turns a slow day into
lost provenance. Exhausting the budget RETAINS the pending row for boot recovery.

Failed DB writes coalesce per (team, channel, ts) and drain in the background for the process
lifetime. A deletion tombstone absorbs everything after it — a queued registration must never
resurrect a message Slack has already confirmed gone. A process that dies with a non-empty
queue permanently omits those rows; boot reconcile finalizes rows that EXIST, it cannot
reconstruct a registration that only ever lived in memory.

CLASSIFICATION INVENTORY (every durable post site; "exempt" = DM/ephemeral, no row possible).
The `stamp` column is the EDIT_OWN_MESSAGE §4 receipt_class every producer passes EXPLICITLY —
the ledger API takes it as a required keyword, there is no default and no inference.

  site                                             owner    state lifecycle        stamp
  ------------------------------------------------------------------------------------------
  send_message single post                         turn     in_flight -> finalized caller's
                                                                                   (reply sites
                                                                                   stamp
                                                                                   assistant_reply)
  send_message split chunks 1..N                   turn     in_flight each, unit   caller's
                                                            finalize
  send_message split-abort truncation notice       turn     finalized              system_notice
  _reconcile_uncertain_post late ts                turn     late register ->       caller's
                                                            finalize
  send_message_get_ts legacy seed                  turn     in_flight -> finalized caller's
  native stream part 1..N (chat.startStream)       turn     in_flight each, unit   assistant_reply
                                                            finalize
  native stream abandon()                          turn     delete after confirm   --
  legacy overflow part (_post_overflow_part)       turn     in_flight -> finalized assistant_reply
  thinking placeholder                             turn     chrome -> promote on   chrome; promote
                                                            first answer edit      maps to
                                                                                   assistant_reply
  "Retrying without ..." overwrite of a partial    turn     demote to chrome       maps back to
                                                                                   chrome
  terminal error / timeout / interruption notice   turn     finalized              system_notice
  prior-timeout notice (process_message)           turn     finalized              system_notice
  status card post/update (research)               job      chrome                 background_job
  progress checklist message                       turn     chrome                 background_job
  response footer post                             turn     chrome                 chrome
  post_to_thread                                   turn     finalized + root       assistant_reply
  research findings / failure note                 job      finalized              background_job
  detached image upload (publish_image)            producer finalized via pending  artifact
                                                            share
  artifact upload (send_file)                      producer finalized via pending  artifact
                                                            share
  produced-image review comment                    job      in_flight -> finalized artifact
                                                            (§11.3: job output beside
                                                            the share, not an editable
                                                            reply)
  image-gen handler notices                        turn     finalized              system_notice
  channel-join hello + findings                    sys      finalized              system_notice
  channel welcome / reminder / settings-button     sys      chrome                 chrome
  settings "saved" confirmations in a channel      sys      finalized              system_notice
  correction disclosure (edit_own_message)         turn     finalized              correction_
                                                                                   announcement
  ephemerals, DM targets, assistant greetings      --       exempt (structural)    --
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

from logger import setup_logger
from runtime_identity import SESSION_ID
# Both are leaf modules (logger/config/runtime_identity only), so a top-level import here cannot
# close a cycle back through the package.
from message_processor import dev_barriers, participation_telemetry

# The repo's configured hierarchy, not `getLogger(__name__)`. That name sits outside `slack_bot.*`
# and so lands on the unconfigured root logger: in a normal `python3 slackbot.py` run every
# transition line below went nowhere, and the only trace of a receipt was the DatabaseManager
# mirror one layer down (live battery F4).
logger = setup_logger(name="slack_bot.OutboundReceipts")

STATE_IN_FLIGHT = "in_flight"
STATE_FINALIZED = "finalized"
STATE_CHROME = "chrome"

RECEIPT_KINDS = (STATE_CHROME, STATE_FINALIZED)

# Receipt CLASSES (EDIT_OWN_MESSAGE spec §4): what KIND of surface a message is, orthogonal to
# its lifecycle state above (a research status card is chrome-STATE with class background_job).
# A closed enum, validated in Python; only `assistant_reply` rows are ever edit-eligible.
# Legacy rows carry NULL and are ineligible for anything class-gated — never inferred.
CLASS_ASSISTANT_REPLY = "assistant_reply"
CLASS_CORRECTION_ANNOUNCEMENT = "correction_announcement"
CLASS_SYSTEM_NOTICE = "system_notice"
CLASS_BACKGROUND_JOB = "background_job"
CLASS_ARTIFACT = "artifact"
CLASS_CHROME = "chrome"

RECEIPT_CLASSES = (CLASS_ASSISTANT_REPLY, CLASS_CORRECTION_ANNOUNCEMENT, CLASS_SYSTEM_NOTICE,
                   CLASS_BACKGROUND_JOB, CLASS_ARTIFACT, CLASS_CHROME)


def _checked_class(receipt_class: Any, *, site: str, message_ts: Any = None) -> Optional[str]:
    """The ledger-side enum gate. Never raises into a posting path.

    A producer passing an unknown class is a bug worth shouting about, but refusing the WRITE
    would put the delivered message outside the stream forever — the module's worst failure.
    So the row is written with a NULL class (ineligible for anything class-gated, exactly like
    a legacy row: fail closed on the class, never on the delivery record) and the bug is
    logged as an error. An explicit None gets the same loud line: every producer is required
    to say what it posted.
    """
    if receipt_class in RECEIPT_CLASSES:
        return receipt_class
    logger.error(
        "Receipt class %r at %s for ts=%s is not a valid class — recording NULL "
        "(the row stays ineligible for anything class-gated)",
        receipt_class, site, message_ts, stack_info=receipt_class is not None)
    return None


def receipt_has_class(receipt: Any, required_class: str) -> bool:
    """Whether a receipt row carries EXACTLY `required_class` (spec §4 eligibility).

    The single answer to "is this row class-gated in": a legacy row (receipt_class IS NULL)
    is NEVER eligible, and nothing here infers a class from state, owner, text, provenance or
    the pre-epoch grandfather. `receipt` may be a DB row dict or anything with a
    `receipt_class` attribute; anything else is ineligible.
    """
    if required_class not in RECEIPT_CLASSES:
        raise ValueError(f"invalid receipt class: {required_class!r}")
    if isinstance(receipt, dict):
        value = receipt.get("receipt_class")
    else:
        value = getattr(receipt, "receipt_class", None)
    return value == required_class

# How long a turn's settle may hold the outer finally before we give up on it and let the
# drain worker carry the rows. Shielded, so the settle itself continues either way.
SETTLE_TIMEOUT_SECONDS = 10.0

_DRAIN_INTERVAL_SECONDS = 2.0

# How many Slack-confirmed file deletions the tombstone remembers. Bounded because the set would
# otherwise grow for the life of the process; generous because forgetting one costs a permanently
# unresolvable pending row, and an upload cannot complete thousands of deletions later.
_DELETED_FILE_MEMORY = 4096

# How long shutdown waits for a Slack callback that is mid-post before cancelling it. Short: a
# callback is a notice, not a turn, and Slack has its own 3s ack budget anyway.
CALLBACK_DRAIN_TIMEOUT_SECONDS = 10.0

# How often the wait on a post Slack may already have accepted says so. It is a WATCHDOG, not a
# budget: that wait has no deadline (see `_settle_protected`), and this only makes a slow one
# visible instead of silent.
PROTECTED_POST_WATCHDOG_SECONDS = 15.0


def sys_owner() -> str:
    return f"{SESSION_ID}:sys"


def job_owner(job_id: Any) -> str:
    return f"{SESSION_ID}:job:{job_id}"


_turn_seq = 0


def next_turn_id() -> str:
    """`{session}:{seq}` — the session half is what dead-session reconciliation matches on."""
    global _turn_seq
    _turn_seq += 1
    return f"{SESSION_ID}:{_turn_seq}"


def receipts_apply(channel_id: Optional[str]) -> bool:
    """Receipts exist only for channel-shaped conversations (spec §8 surface ruling).

    Imported lazily: the Slack package pulls in the transport that calls back into here.
    """
    if not channel_id:
        return False
    from slack_client.utilities import is_dm_conversation
    return not is_dm_conversation(channel_id)


# --- the retry lattice ------------------------------------------------------------------

# Ops, strongest last. A stronger op absorbs a weaker one queued for the same message.
_RANK = {"chrome": 0, "register": 1, "promote": 1, "demote": 1, "pending_share": 1,
         "finalize": 2, "delete": 3, "delete_share": 3}

# Slack confirmed the thing is gone; nothing queued behind one may write a row for it again.
_TOMBSTONES = frozenset({"delete", "delete_share"})


class _PoisonedClass:
    """Spec §11.12: the DISTINCT conflict-poisoned class sentinel.

    Plain None could not carry the verdict: None also means "no claim yet", so a third claim
    arriving after a conflict looked like a first fill and quietly refilled the class the
    conflict had just voided. This object is truthy, equal only to itself, and sticky in every
    merge — once a conflict stamps it, no later claim refills the class. It reaches the
    database as NULL (`_Op.db_class`), the ineligible terminal state, and nothing else.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<receipt class conflict-poisoned>"


POISONED_CLASS = _PoisonedClass()


def _merged_class(survivor: "_Op", other: "_Op") -> Any:
    """The class the SURVIVING op carries after a lattice merge (spec §11.1/§11.12).

    Two REAL classes disagreeing is a producer bug: the survivor fails closed to the distinct
    POISONED_CLASS sentinel (writes as NULL, the ineligible terminal state). The poison is
    STICKY — once either side carries it the merge answers poison, so a third claim after a
    conflict can never refill the class. Otherwise the known class fills the unknown one.
    """
    a, b = survivor.receipt_class, other.receipt_class
    if a is POISONED_CLASS or b is POISONED_CLASS:
        return POISONED_CLASS
    if a and b and a != b:
        logger.error(
            "Receipt class conflict in the lattice for %s/%s: %r vs %r — class poisoned "
            "(ineligible; writes as NULL)",
            survivor.channel_id, survivor.file_id or survivor.message_ts, a, b)
        return POISONED_CLASS
    return a or b

# Lattice kind → ledger op (participation_telemetry.RECEIPT_OPS). `chrome` and `register` are one
# op there: both are a registration, and the row's `new_state` already says which surface it made.
# The share ops are absent on purpose — they touch pending_share_receipts, not outbound_receipts,
# and an event for one would put a row with no receipt state into the receipt population.
_EVENT_OPS = {"register": "register", "promote": "promote", "chrome": "register",
              "demote": "demote", "finalize": "finalize", "delete": "delete"}


def _emit_transition(*, channel_id: Optional[str], message_ts: Optional[str], owner: Optional[str],
                     op: str, result: Any = None, applied: Optional[bool] = None,
                     prior_state: Optional[str] = None, new_state: Optional[str] = None,
                     reason: Optional[str] = None) -> None:
    """Put one transition on the ledger. Never raises, never changes a return value.

    Reads the TransitionResult by duck-typing rather than importing it: this module is on the
    posting path and must not gain a hard dependency on database.py for a log line. An explicit
    keyword wins over the result object, for the two ops that have no accessor result of their own.
    """
    try:
        participation_telemetry.outbound_receipt(
            channel_id=channel_id, message_ts=message_ts, owner_turn_id=owner, op=op,
            prior_state=(prior_state if prior_state is not None
                         else getattr(result, "prior_state", None)),
            new_state=(new_state if new_state is not None
                       else getattr(result, "new_state", None)),
            applied=bool(applied if applied is not None
                         else getattr(result, "applied", False)),
            reason=(reason if reason is not None else getattr(result, "reason", None)))
    except Exception as e:  # noqa: BLE001 — a lost line is never worth a lost message
        logger.debug("Receipt telemetry skipped for %s/%s: %s", channel_id, message_ts, e)


@dataclass
class _Op:
    kind: str
    team_id: str
    channel_id: str
    message_ts: str
    owner: str
    thread_root_ts: Optional[str] = None
    # Spec §4: the class the producer stamped, riding the queued write so a retried
    # registration lands with the same claim the original made. None only for ops that carry
    # no claim of their own (delete, demote — the DB maps the demotion class itself).
    # POISONED_CLASS (spec §11.12) after a lattice conflict — sticky, never refilled.
    receipt_class: Optional[Any] = None
    # Set only for pending-share ops, which have no message ts yet — the file id is the whole
    # handle Slack gave us.
    file_id: Optional[str] = None

    @property
    def rank(self) -> int:
        return _RANK.get(self.kind, 0)

    @property
    def db_class(self) -> Optional[str]:
        """What the DATABASE is told: a conflict-poisoned class writes as NULL — the
        ineligible terminal state — and the sentinel itself never leaves the lattice."""
        return None if self.receipt_class is POISONED_CLASS else self.receipt_class

    @property
    def key(self) -> Tuple[str, str, str]:
        """Queue identity. File-keyed ops share the queue with ts-keyed ones, so the namespace
        prefix keeps a file id from ever colliding with a message ts."""
        if self.file_id:
            return (self.team_id, self.channel_id, f"file:{self.file_id}")
        return (self.team_id, self.channel_id, self.message_ts)


@dataclass
class ReceiptService:
    """Owns the DB handle, the coalescing queue and the drain worker.

    Every write goes through `apply`, which tries the DB immediately and queues the op on
    failure. Nothing here raises into a posting path: a receipt is bookkeeping about a message
    that is already in the room.
    """

    db: Any = None
    _queue: Dict[Tuple[str, str, str], _Op] = field(default_factory=dict, repr=False)
    _drain_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _tasks: set = field(default_factory=set, repr=False)
    _resolvers: set = field(default_factory=set, repr=False)
    _accepting: bool = True
    # The op each key is mid-write on, and the keys claimed out from under such a write. A
    # pending-share drain and a share RESOLUTION can touch one file at the same moment: the
    # resolver pops the queued op and finalizes the share, while the drain — already awaiting its
    # DB write — commits a pending row for a share that is now resolved, and that row outlives
    # the process. The write cannot be un-awaited, so it is compensated instead.
    _draining: Dict[Tuple[str, str, str], _Op] = field(default_factory=dict, repr=False)
    _revoked: set = field(default_factory=set, repr=False)
    # Files Slack has confirmed deleted. A per-key queue tombstone cannot cover this: the upload
    # completion and the `file_deleted` listener are unordered, so the deletion can land while
    # there is nothing queued to absorb it, and the registration that arrives afterwards writes a
    # pending row for a file that no longer exists — unresolvable, retried and logged critically
    # on every boot from then on. This outlives the queue entry, so a LATE registration is
    # refused rather than absorbed.
    _deleted_files: "OrderedDict[str, None]" = field(default_factory=OrderedDict, repr=False)

    # --- queue ---------------------------------------------------------------------------

    @property
    def accepting(self) -> bool:
        return self._accepting

    def note_file_deleted(self, file_id: Any) -> None:
        """Remember a Slack-confirmed deletion for the rest of the process."""
        fid = str(file_id)
        self._deleted_files.pop(fid, None)
        self._deleted_files[fid] = None
        while len(self._deleted_files) > _DELETED_FILE_MEMORY:
            self._deleted_files.popitem(last=False)

    def file_is_deleted(self, file_id: Any) -> bool:
        return str(file_id) in self._deleted_files

    def _refuse(self, op: _Op) -> None:
        """After the final drain the queue is closed. A producer arriving here is plumbing that
        outlived its shutdown, so it is logged rather than silently queued behind a worker that
        will never run again."""
        logger.error("Receipt %s for %s/%s arrived after receipt shutdown and was refused",
                     op.kind, op.channel_id, op.file_id or op.message_ts)

    def _enqueue(self, op: _Op, internal: bool = False) -> None:
        # `internal` marks compensation for a write this service itself just made — never a new
        # producer. It is the one thing a closed queue must still take: refusing it would leave
        # the very row the compensation exists to undo sitting in the database.
        if not self._accepting and not internal:
            self._refuse(op)
            return
        key = op.key
        existing = self._queue.get(key)
        if existing is not None:
            if existing.kind in _TOMBSTONES:
                return
            if op.rank < existing.rank:
                if op.thread_root_ts and not existing.thread_root_ts:
                    existing.thread_root_ts = op.thread_root_ts
                existing.receipt_class = _merged_class(existing, op)
                return
            if op.kind == "pending_share" and existing.kind == "pending_share":
                # Spec §11.12/§11.20: queued pending-share claims are UNCONDITIONALLY
                # first-writer, mirroring the database's rule. An identical re-claim is an
                # idempotent no-op; ANY differing second claim — class (None and poisoned
                # included), owner or root — is dropped with the ERROR, because an equal-rank
                # replacement would hand the file's ownership and class to the later claim.
                if (op.owner == existing.owner
                        and op.receipt_class == existing.receipt_class
                        and op.thread_root_ts == existing.thread_root_ts):
                    return
                logger.error(
                    "Pending share %s second claim differs in the lattice: first "
                    "claim (%s) carries %r, %s claimed %r — the first claim stands",
                    op.file_id, existing.owner, existing.receipt_class, op.owner,
                    op.receipt_class)
                return
            if existing.thread_root_ts and not op.thread_root_ts:
                op.thread_root_ts = existing.thread_root_ts
            op.receipt_class = _merged_class(op, existing)
        self._queue[key] = op
        self._ensure_drain()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def _ensure_drain(self) -> None:
        if not self._accepting:
            return
        if self._drain_task is not None and not self._drain_task.done():
            return
        try:
            self._drain_task = asyncio.get_running_loop().create_task(self._drain_worker())
        except RuntimeError:
            self._drain_task = None

    async def _drain_worker(self) -> None:
        try:
            while self._queue:
                await asyncio.sleep(_DRAIN_INTERVAL_SECONDS)
                await self.drain_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.critical("Receipt drain worker died with %s; queued rows retained", e)

    async def drain_once(self) -> int:
        """One pass over the queue. Ops that fail again stay queued for a later drain."""
        drained = 0
        for key, op in list(self._queue.items()):
            if self._queue.get(key) is not op:
                continue
            self._draining[key] = op
            try:
                ok = await self._write(op)
            finally:
                self._draining.pop(key, None)
            revoked = key in self._revoked
            self._revoked.discard(key)
            if ok:
                if self._queue.get(key) is op:
                    del self._queue[key]
                drained += 1
                if revoked and op.kind == "pending_share":
                    # Claimed mid-write: the share resolved while this row was being committed,
                    # so the row now describes work that is already finalized. Undo it through
                    # the lattice, which retries until the delete actually lands.
                    self._enqueue(_Op("delete_share", op.team_id, op.channel_id, "",
                                      op.owner, file_id=op.file_id), internal=True)
            else:
                logger.critical(
                    "Receipt %s for %s/%s still failing; retained for a later drain",
                    op.kind, op.channel_id, op.message_ts)
        return drained

    # --- writes --------------------------------------------------------------------------

    @staticmethod
    async def _delete_shares_by_file(db: Any, file_id: Optional[str]) -> None:
        """Drop every pending row for one file, wherever it lives.

        The addressed form needs a team and a channel; Slack's `file_deleted` carries neither,
        so the rows have to be found first. Raising is the point — the caller turns it into a
        retained lattice op, because the event is one-shot and a transient read failure would
        otherwise lose the cleanup for the life of the database.
        """
        rows = await db.get_pending_shares_async()
        for row in rows or []:
            if row.get("file_id") != file_id:
                continue
            await db.delete_pending_share_async(
                row.get("team_id"), row.get("channel_id"), file_id)

    async def _write(self, op: _Op) -> bool:
        """True when the write COMPLETED, false only when it raised.

        A refused transition is not a failed one: `applied=False` is an answer, and retrying it
        would ask the same question of the same rows forever. The distinction is the whole reason
        the accessors return a result object, so it must not leak back into this return value.
        """
        db = self.db
        if db is None:
            return False
        result: Any = None
        try:
            if op.kind == "register":
                result = await db.register_receipt_async(
                    op.team_id, op.channel_id, op.message_ts, op.owner, STATE_IN_FLIGHT,
                    op.thread_root_ts, receipt_class=op.db_class)
            elif op.kind == "promote":
                result = await db.register_receipt_async(
                    op.team_id, op.channel_id, op.message_ts, op.owner, STATE_IN_FLIGHT,
                    op.thread_root_ts, receipt_class=op.db_class)
            elif op.kind == "chrome":
                result = await db.register_chrome_async(
                    op.team_id, op.channel_id, op.message_ts, op.owner, op.thread_root_ts,
                    receipt_class=op.db_class)
            elif op.kind == "demote":
                result = await db.demote_receipt_chrome_async(
                    op.team_id, op.channel_id, op.message_ts, op.owner)
            elif op.kind == "finalize":
                unit = await db.finalize_receipts_async(
                    op.team_id, op.channel_id,
                    [(op.message_ts, op.thread_root_ts, op.db_class)], op.owner)
                result = unit[0] if isinstance(unit, (list, tuple)) and unit else None
            elif op.kind == "delete":
                result = await db.delete_receipt_async(
                    op.team_id, op.channel_id, op.message_ts)
            elif op.kind == "pending_share":
                if self.file_is_deleted(op.file_id):
                    # The deletion landed while this write sat in the queue. Writing it now would
                    # resurrect a row for a file that is gone; dropping the op IS the success.
                    logger.debug("Pending share %s dropped — the file is already gone",
                                 op.file_id)
                    return True
                await db.record_pending_share_async(
                    op.team_id, op.channel_id, op.file_id, op.owner, op.thread_root_ts,
                    receipt_class=op.db_class)
                # Same overlap as the direct write: a deletion can stamp and scan while this
                # one is in flight. Checked again now that it has committed.
                await _drop_share_row_if_deleted(db, self, op.team_id, op.channel_id,
                                                 op.file_id)
            elif op.kind == "delete_share":
                if op.team_id and op.channel_id:
                    await db.delete_pending_share_async(op.team_id, op.channel_id, op.file_id)
                else:
                    await self._delete_shares_by_file(db, op.file_id)
            else:
                logger.error("Unknown receipt op %r", op.kind)
                return True
        except Exception as e:  # noqa: BLE001 — bookkeeping never breaks a delivered message
            logger.warning("Receipt %s failed for %s/%s: %s",
                           op.kind, op.channel_id, op.message_ts, e)
            return False
        event_op = _EVENT_OPS.get(op.kind)
        if event_op is not None:
            _emit_transition(channel_id=op.channel_id, message_ts=op.message_ts,
                             owner=op.owner, op=event_op, result=result)
        logger.debug("Receipt %s %s/%s owner=%s root=%s",
                     op.kind, op.channel_id, op.file_id or op.message_ts, op.owner,
                     op.thread_root_ts)
        return True

    async def apply(self, op: _Op) -> bool:
        if not self._accepting:
            self._refuse(op)
            return False
        key = op.key
        queued = self._queue.get(key)
        if queued is not None:
            # Something for this message is already waiting; keep strict per-key ordering by
            # merging into the queue rather than racing it to the DB.
            self._enqueue(op)
            return False
        if await self._write(op):
            return True
        self._enqueue(op)
        return False

    async def finalize_unit(self, team_id: str, channel_id: str,
                            records: List[Tuple[str, Optional[str], Optional[str]]],
                            owner: str) -> bool:
        """Finalize a turn's posts as ONE unit; per-receipt fallback keeps the queue honest.

        `records` = [(message_ts, thread_root_ts|None, receipt_class|None)] — the class each
        post was stamped with rides the finalize, so a lost registration is inserted WITH its
        class rather than as an unclassifiable row."""
        db = self.db
        if db is None or not records:
            return False
        if not self._accepting:
            self._refuse(_Op("finalize", team_id, channel_id, records[0][0], owner))
            return False
        pending = [r for r in records
                   if self._queue.get((team_id, channel_id, r[0])) is None]
        deferred = [r for r in records if r not in pending]
        for ts, root, cls in deferred:
            self._enqueue(_Op("finalize", team_id, channel_id, ts, owner, root, cls))
        if not pending:
            return False
        try:
            results = await db.finalize_receipts_async(team_id, channel_id, pending, owner)
        except Exception as e:  # noqa: BLE001
            logger.warning("Receipt unit finalize failed for %s: %s", channel_id, e)
            for ts, root, cls in pending:
                self._enqueue(_Op("finalize", team_id, channel_id, ts, owner, root, cls))
            return False
        # One event per message, zipped by position — the accessor returns one result per input
        # record in input order precisely so this attribution cannot slip.
        rows = list(results) if isinstance(results, (list, tuple)) else []
        for (ts, _root, _cls), result in zip(pending, rows):
            _emit_transition(channel_id=channel_id, message_ts=ts, owner=owner, op="finalize",
                             result=result)
        for ts, root, _cls in pending:
            logger.debug("Receipt finalize %s/%s owner=%s root=%s", channel_id, ts, owner, root)
        return True

    # --- lifecycle -----------------------------------------------------------------------

    def track(self, task: asyncio.Task) -> None:
        """Strong ref on a detached settle so it can't be garbage-collected mid-write."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def track_resolver(self, task: asyncio.Task) -> None:
        """Strong ref on a share-ts poll. Cancelled at shutdown rather than awaited: it can run
        for the whole files.info budget, and a cancelled resolution leaves the pending row
        exactly where boot recovery expects to find it."""
        self._resolvers.add(task)
        task.add_done_callback(self._resolvers.discard)

    def queue_pending_share(self, *, team_id: str, channel_id: str, file_id: str, owner: str,
                            thread_root_ts: Optional[str] = None,
                            receipt_class: Optional[str] = None) -> None:
        """The pending row could not be written. Retain the provenance in the lattice: without
        it a live-process SQLite blip loses the upload's owner for good — unless Slack has
        already said the file is gone, in which case there is nothing left to be provenance for.
        The producer's class rides the queued op (spec §4) so the retried row — and any
        resolution consumed straight off the queue — still carries it."""
        if self.file_is_deleted(file_id):
            return
        self._enqueue(_Op("pending_share", str(team_id), str(channel_id), "", owner,
                          thread_root_ts, receipt_class, file_id=str(file_id)))

    def queued_pending_shares(self, file_id: str) -> List[_Op]:
        """Pending-share writes still waiting on the queue for one file id. Slack's
        `file_deleted` names only the file, and a write that never reached the database has no
        row to look the rest up from."""
        return [op for op in self._queue.values()
                if op.kind == "pending_share" and op.file_id == str(file_id)]

    def take_pending_share(self, team_id: str, channel_id: str,
                           file_id: str) -> Optional[_Op]:
        """Remove and return a queued pending-share write, if one is still waiting.

        A write already in flight counts as taken: the caller knows the share ts NOW, so its
        finalize is the later, authoritative fact. The key is marked revoked so the drain
        compensates whatever it is about to commit.
        """
        key = (str(team_id), str(channel_id), f"file:{file_id}")
        op = self._queue.pop(key, None)
        in_flight = self._draining.get(key)
        if in_flight is not None and in_flight.kind == "pending_share":
            self._revoked.add(key)
            if op is None:
                op = in_flight
        return op

    def queue_share_deletion(self, *, team_id: str, channel_id: str, file_id: str,
                             internal: bool = True) -> None:
        """Slack confirmed the file is gone. Tombstones any queued write for it.

        ALWAYS retainable, which is why `internal` defaults True here. Two reasons, and they are
        the same reason: this op compensates a row this service itself wrote, and its trigger —
        Slack's `file_deleted` — is one-shot. Slack never sends it twice, so a deletion refused
        because the queue had closed is not deferred, it is lost, and the pending row it was meant
        to remove is then retried and logged critically on every boot for the life of the database.
        That is precisely the hole the P1 lattice exists to close, and receipt shutdown happens
        while the `file_deleted` listener is still registered — so the closed queue must take it.
        `drain_late_arrivals()` is what then writes it.
        """
        self._enqueue(_Op("delete_share", str(team_id), str(channel_id), "", sys_owner(),
                          file_id=str(file_id)), internal=internal)

    async def shutdown(self) -> None:
        """Stop the producers, stop the worker, then drain what is left — in that order, so the
        final drain still has a DB to write to and nothing is writing alongside it.

        `_accepting` stays True until the settles have run: the tracked tasks are producers that
        were already in flight when shutdown began, and refusing them would throw away the rows
        this drain exists to write.

        The WORKER goes down before the final drain, not after. Running both meant two passes
        writing the same queue at once — and the loser's compensating `delete_share` (a revoked
        mid-drain pending share) could arrive to find the queue already closed.
        """
        for task in list(self._resolvers):
            task.cancel()
        if self._resolvers:
            await asyncio.gather(*list(self._resolvers), return_exceptions=True)
        for task in list(self._tasks):
            if not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=SETTLE_TIMEOUT_SECONDS)
                except Exception:  # noqa: BLE001
                    pass
        # A settle that outran its shield is STILL RUNNING. Left alone it would enqueue behind a
        # queue that is about to close and a worker that will never run again — the rows it was
        # carrying refused, its messages permanently outside the stream. It is stopped here,
        # while there is still a queue to catch whatever it managed to write.
        stragglers = [t for t in list(self._tasks) if not t.done()]
        if stragglers:
            logger.warning("Cancelling %d receipt settle(s) that outran their shield",
                           len(stragglers))
            for task in stragglers:
                task.cancel()
            await asyncio.gather(*stragglers, return_exceptions=True)
        # The worker stops FIRST, while the queue is still open, so anything it is mid-write on
        # can still compensate itself.
        task = self._drain_task
        self._drain_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
        self._accepting = False
        try:
            await self.drain_once()
            # The final drain can compensate itself too (a share resolved out from under a
            # pending-share write). That op is enqueued internally, so one more pass carries it.
            if self._queue:
                await self.drain_once()
        except Exception as e:  # noqa: BLE001
            logger.warning("Final receipt drain failed: %s", e)
        if self._queue:
            logger.critical(
                "Receipts: %d queued row(s) never reached the database; those messages are "
                "permanently omitted from the stream", len(self._queue))

    async def drain_late_arrivals(self) -> int:
        """One more pass for ops that arrived AFTER `shutdown()` closed the queue.

        Only always-retainable ops can be here: a `delete_share` from the `file_deleted` listener,
        which stays registered until Slack ingress stops — well after receipts close. The queue
        accepts it (see `queue_share_deletion`) but nothing drains a closed queue, so this is the
        call that finishes the job. It must run once ingress is genuinely quiet and while the
        database is still open; both are true at exactly one point in shutdown.
        """
        if not self._queue:
            return 0
        drained = 0
        try:
            drained = await self.drain_once()
        except Exception as e:  # noqa: BLE001
            logger.warning("Late receipt drain failed: %s", e)
        if self._queue:
            logger.critical(
                "Receipts: %d late row(s) never reached the database; a deleted file's pending "
                "share will be retried and logged on every boot until it is cleaned up",
                len(self._queue))
        return drained


_service: Optional[ReceiptService] = None


def install_service(db: Any) -> ReceiptService:
    """Composition root wiring. Idempotent — a second call rebinds the db on the same queue."""
    global _service
    if _service is None:
        _service = ReceiptService(db=db)
    else:
        _service.db = db
        _service._accepting = True
    return _service


def get_service() -> Optional[ReceiptService]:
    return _service


def reset_service() -> None:
    global _service
    _service = None


# --- admission for callback-borne channel posts -------------------------------------------


@dataclass
class ChannelPostGate:
    """Admission for Slack callbacks that post durable channel content.

    A turn is not the only thing that puts our own prose in a room. Bolt callbacks do too — a
    settings confirmation, an onboarding notice — and Socket Mode stays connected until the very
    end of shutdown, long after the receipt queue closes. A callback that posted in that window
    had its registration refused: the message sat in the channel with nothing claiming it, and
    the rebuilt stream would never contain it.

    So a refused callback does not post AT ALL. Better unsent than unreceipted: a notice nobody
    sees costs one piece of setup chrome, while a post nobody can account for corrupts the
    record of the conversation for as long as the channel exists.

    Entries are counted per task, so a callback that enters twice (a notice inside a notice)
    leaves only when its outermost frame does.

    The post and its registration are ONE unit (`protect`). A cancellation landing between them
    would recreate the exact failure this gate exists to prevent, from the other direction:
    Slack has accepted the message, and the row that keeps it inside the stream never gets
    written. So that pair runs as its own task, is never cancelled, and the drain waits for it.
    """

    _admitting: bool = True
    _active: Dict[Any, int] = field(default_factory=dict, repr=False)
    _protected: set = field(default_factory=set, repr=False)

    @property
    def admitting(self) -> bool:
        return self._admitting

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def protected_count(self) -> int:
        return len(self._protected)

    def protect(self, task: Any) -> None:
        """Strong ref on a post-through-registration pair. Awaited by the drain, never cancelled."""
        self._protected.add(task)
        task.add_done_callback(self._protected.discard)

    def enter(self, site: str) -> bool:
        if not self._admitting:
            logger.warning(
                "Channel post at %s refused — receipts are closing, so it was never sent", site)
            return False
        task = asyncio.current_task() if _has_running_loop() else None
        if task is not None:
            self._active[task] = self._active.get(task, 0) + 1
        return True

    def leave(self) -> None:
        task = asyncio.current_task() if _has_running_loop() else None
        if task is None:
            return
        remaining = self._active.get(task, 0) - 1
        if remaining > 0:
            self._active[task] = remaining
        else:
            self._active.pop(task, None)

    async def drain(self, timeout: float = CALLBACK_DRAIN_TIMEOUT_SECONDS) -> None:
        """Close admission, wait out the callbacks already inside, then settle what Slack has
        already accepted."""
        self._admitting = False
        current = asyncio.current_task()
        pending = [t for t in list(self._active) if t is not current and not t.done()]
        if pending:
            logger.info("Draining %d channel-posting callback(s)...", len(pending))
            _done, still_running = await asyncio.wait(pending, timeout=max(0.0, timeout))
            if still_running:
                logger.warning("Cancelling %d channel-posting callback(s) that did not finish",
                               len(still_running))
                for task in still_running:
                    task.cancel()
                await asyncio.gather(*still_running, return_exceptions=True)
        await self._settle_protected()

    async def _settle_protected(self) -> None:
        """Wait for every post-through-registration pair. TO COMPLETION — no deadline.

        Cancelling a callback does not cancel a post Slack may already have taken, and giving up
        on the registration behind one is the same thing as never having shielded it: the message
        is in the room and nothing accounts for it. Everything downstream of this — the receipt
        queue closing, the database going away, main's blanket task-cancel — assumes these are
        finished, so this is where they finish.

        An unbounded wait is only safe because a pair terminates on its own, which holds by
        construction on all three legs:

        * the post. Bolt builds its per-request AsyncWebClient with `session=` copied from an
          app client that never sets one, so slack_sdk takes the no-running-session branch and
          builds a fresh `aiohttp.ClientSession(timeout=ClientTimeout(total=30))` per call. The
          only default retry handler is AsyncConnectionErrorRetryHandler(max_retry_count=1), and
          there is no 429 handler, so the ceiling is two attempts plus a sub-second jittered
          backoff — about a minute, worst case.
        * the registration. Every receipt accessor opens through `_stream_conn`, which sets
          `PRAGMA busy_timeout=5000`; a contended write fails after five seconds rather than
          waiting on a lock.
        * a registration that fails is retained as a lattice op instead of retried inline. That
          enqueue lands in an OPEN queue because this drain runs before `ReceiptService.shutdown`
          (main.py orders it so, and test_main asserts the order), and the final drain carries it.

        The watchdog only makes a slow wait visible; it never ends one.
        """
        waited = 0.0
        while True:
            protected = [t for t in list(self._protected) if not t.done()]
            if not protected:
                return
            if not waited:
                logger.info("Settling %d post(s) Slack may already have accepted...",
                            len(protected))
            _done, unfinished = await asyncio.wait(
                protected, timeout=PROTECTED_POST_WATCHDOG_SECONDS)
            if unfinished:
                waited += PROTECTED_POST_WATCHDOG_SECONDS
                logger.warning(
                    "Still waiting on %d channel post(s) to finish registering after %.0fs — "
                    "shutdown will not close receipts until they do",
                    len(unfinished), waited)


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


_channel_post_gate = ChannelPostGate()


def get_channel_post_gate() -> ChannelPostGate:
    return _channel_post_gate


def reset_channel_post_gate() -> None:
    global _channel_post_gate
    _channel_post_gate = ChannelPostGate()


@asynccontextmanager
async def channel_post_admission(site: str):
    """Bracket a callback's durable channel post. Yields False when it must not post."""
    gate = get_channel_post_gate()
    admitted = gate.enter(site)
    try:
        yield admitted
    finally:
        if admitted:
            gate.leave()


async def post_then_register(work: Any) -> Any:
    """Run one post-and-register coroutine as an uninterruptible unit.

    Admission alone is not enough. A callback cancelled by the drain can be cancelled BETWEEN
    Slack accepting the message and its receipt being written — which is the same unaccounted
    message the gate exists to prevent, arriving from the other direction. So the pair runs as
    its own task, the caller only ever awaits a shield over it, and the gate holds a reference
    and waits for it. Cancelling the caller abandons the wait, never the work.
    """
    task = asyncio.ensure_future(work)
    get_channel_post_gate().protect(task)
    return await asyncio.shield(task)


async def drain_channel_post_callbacks(
        timeout: float = CALLBACK_DRAIN_TIMEOUT_SECONDS) -> None:
    await get_channel_post_gate().drain(timeout)


# --- the per-owner ledger ---------------------------------------------------------------


class ReceiptLedger:
    """One owner's outbound posts for one channel. Never raises into a posting path.

    `note_post` claims a message as this owner's conversational output; `settle` finalizes
    every such message that is still in flight, as a unit. Chrome is registered and forgotten
    — it is excluded from the stream permanently, so it has nothing to settle.
    """

    def __init__(self, owner_id: str, team_id: Optional[str], channel_id: Optional[str],
                 service: Optional[ReceiptService] = None,
                 barrier_eligible: bool = False):
        self.owner_id = owner_id
        self.team_id = str(team_id or "")
        self.channel_id = str(channel_id or "")
        self._service = service
        # ts -> (thread_root_ts, receipt_class): what settle will finalize each post AS.
        self._in_flight: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._settled = False
        # Whether this ledger may hold a turn still at the dev `post_partial_post` seam. OPT-IN:
        # only the responder turn's own ledger qualifies. A detached job and the sys owner post on
        # their own schedule, and pausing one of those would freeze an unrelated surface while the
        # battery waits for a turn that is not coming.
        self._barrier_eligible = bool(barrier_eligible)
        self._barrier_fired = False

    @property
    def service(self) -> Optional[ReceiptService]:
        return self._service if self._service is not None else get_service()

    @property
    def _svc(self) -> ReceiptService:
        """`service` on the far side of an `active` check, where it is non-None by definition.

        TYPING ONLY. `cast` is a no-op, so a caller that somehow got here without the check still
        gets the same AttributeError it always did — into the same `except` that swallowed it."""
        return cast(ReceiptService, self.service)

    @property
    def active(self) -> bool:
        """False for DMs, unknown surfaces, an unknown team, or an unconfigured service."""
        svc = self.service
        return bool(self.team_id and receipts_apply(self.channel_id)
                    and svc is not None and svc.db is not None)

    @property
    def pending_ts(self) -> List[str]:
        return list(self._in_flight)

    def _op(self, kind: str, ts: str, root: Optional[str] = None,
            receipt_class: Optional[str] = None) -> _Op:
        return _Op(kind, self.team_id, self.channel_id, str(ts), self.owner_id, root,
                   receipt_class)

    async def note_post(self, ts: Optional[str], kind: str = STATE_IN_FLIGHT,
                        thread_root_ts: Optional[str] = None, *,
                        receipt_class: Optional[str]) -> None:
        """Claim a durable post. `kind` chrome registers excluded chrome instead.

        `receipt_class` (spec §4) is REQUIRED — every producer says what kind of surface it
        posted; there is no default and nothing downstream infers one. An invalid value is
        recorded as NULL (class-ineligible) and logged loudly rather than losing the row."""
        if not ts or not self.active:
            return
        receipt_class = _checked_class(receipt_class, site="note_post", message_ts=ts)
        try:
            if kind == STATE_CHROME:
                await self.note_chrome(ts, thread_root_ts, receipt_class=receipt_class)
                return
            if kind == STATE_FINALIZED:
                await self._svc.apply(
                    self._op("finalize", ts, thread_root_ts, receipt_class))
                self._in_flight.pop(str(ts), None)
                return
            if self._settled:
                # DEFENCE IN DEPTH, and a tripwire. An IN-FLIGHT registration here writes a row
                # this ledger will never finalize — a message of ours sitting outside the stream
                # forever — so it is refused and logged loudly instead. (Chrome and an
                # already-finalized claim are handled above and stay legal: neither leaves an
                # in-flight row behind.) It should be unreachable: a post is now made under an
                # EFFECT LEASE that covers this call, and settlement waits for held leases before
                # it settles anything. If this line ever fires, that chain is broken.
                logger.error(
                    "Receipt refused for %s/%s: this turn's ledger (%s) has already settled — the "
                    "post is outside the stream and its effect was not leased",
                    self.channel_id, ts, self.owner_id)
                return
            if str(ts) in self._in_flight:
                # Already claimed this turn; a streaming tick is not a new message. A
                # duplicate claim naming a DIFFERENT class is a producer bug: the class
                # fails closed to NULL (spec §11.1) — locally, so settle cannot re-stamp
                # it, and in the row, by forwarding the conflicting claim so the DB
                # arbitration NULLs it and shouts.
                prior_root, prior_class = self._in_flight[str(ts)]
                if (prior_class is not None and receipt_class is not None
                        and receipt_class != prior_class):
                    logger.error(
                        "Receipt class conflict at note_post for %s/%s: %r claimed over "
                        "%r — class NULLed (ineligible)",
                        self.channel_id, ts, receipt_class, prior_class)
                    self._in_flight[str(ts)] = (prior_root, None)
                    await self._svc.apply(
                        self._op("register", ts, prior_root, receipt_class))
                return
            self._in_flight[str(ts)] = (thread_root_ts, receipt_class)
            await self._svc.apply(self._op("register", ts, thread_root_ts, receipt_class))
            await self._partial_post_barrier(ts)
        except Exception as e:  # noqa: BLE001
            logger.debug("Receipt note_post skipped for %s: %s", ts, e)

    async def note_chrome(self, ts: Optional[str],
                          thread_root_ts: Optional[str] = None, *,
                          receipt_class: Optional[str]) -> None:
        """Register excluded chrome. `receipt_class` is REQUIRED (spec §4): chrome STATE says
        "never in the stream", the class says what KIND of chrome — a research status card is
        chrome-state with class background_job, a placeholder is class chrome."""
        if not ts or not self.active:
            return
        receipt_class = _checked_class(receipt_class, site="note_chrome", message_ts=ts)
        try:
            await self._svc.apply(self._op("chrome", ts, thread_root_ts, receipt_class))
        except Exception as e:  # noqa: BLE001
            logger.debug("Receipt note_chrome skipped for %s: %s", ts, e)

    async def promote(self, ts: Optional[str],
                      thread_root_ts: Optional[str] = None) -> None:
        """A chrome surface this owner holds is about to carry the answer.

        The class is NOT a parameter here: promotion IS the spec §4 atomic mapping
        chrome → assistant_reply, and the DB applies it with the state move in one
        transaction. Nothing else a promotion could claim is legal."""
        if not ts or not self.active:
            return
        try:
            if str(ts) in self._in_flight:
                return  # promoted once; every later edit grows a surface we already own
            self._in_flight[str(ts)] = (thread_root_ts, CLASS_ASSISTANT_REPLY)
            await self._svc.apply(
                self._op("promote", ts, thread_root_ts, CLASS_ASSISTANT_REPLY))
            # The legacy-placeholder path reaches its first conversational surface HERE, not in
            # note_post: the message already exists as chrome and this is the edit that makes it
            # an answer. Without this the seam would only ever catch native/split posts.
            await self._partial_post_barrier(ts)
        except Exception as e:  # noqa: BLE001
            logger.debug("Receipt promote skipped for %s: %s", ts, e)

    async def _partial_post_barrier(self, ts: Any) -> None:
        """Pause at the dev seam on this ledger's FIRST in_flight surface, after the row is
        written and before anything finalizes — the only window in which "an in-flight surface is
        excluded from the stream a concurrent turn builds" is observable from outside.

        The flag flips before the await, so two posts racing into the same turn cannot both hold
        the seam. A hard no-op in production (dev_barriers reads DEV_TURN_BARRIERS).
        """
        if not self._barrier_eligible or self._barrier_fired:
            return
        self._barrier_fired = True
        try:
            await dev_barriers.post_partial_post(
                channel_id=self.channel_id, message_ts=str(ts), owner=self.owner_id)
        except Exception as e:  # noqa: BLE001 — a dev seam never costs a delivered message
            logger.debug("Partial-post barrier skipped for %s: %s", ts, e)

    async def demote(self, ts: Optional[str]) -> None:
        """Scaffolding overwrote the words this surface carried ("Retrying without…")."""
        if not ts or not self.active:
            return
        try:
            self._in_flight.pop(str(ts), None)
            await self._svc.apply(self._op("demote", ts))
        except Exception as e:  # noqa: BLE001
            logger.debug("Receipt demote skipped for %s: %s", ts, e)

    async def abort(self, ts: Optional[str]) -> None:
        """The message is GONE from Slack and the caller confirmed it."""
        if not ts or not self.active:
            return
        try:
            self._in_flight.pop(str(ts), None)
            await self._svc.apply(self._op("delete", ts))
        except Exception as e:  # noqa: BLE001
            logger.debug("Receipt abort skipped for %s: %s", ts, e)

    async def settle(self) -> int:
        """Finalize everything still in flight, as one unit. Idempotent."""
        if self._settled or not self.active:
            self._settled = True
            return 0
        self._settled = True
        records = [(ts, root, cls) for ts, (root, cls) in self._in_flight.items()]
        self._in_flight.clear()
        if not records:
            return 0
        try:
            await self._svc.finalize_unit(
                self.team_id, self.channel_id, records, self.owner_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Receipt settle failed for %s: %s", self.channel_id, e)
        return len(records)


def ledger_for(owner_id: str, team_id: Optional[str], channel_id: Optional[str],
               barrier_eligible: bool = True) -> Optional[ReceiptLedger]:
    """A ledger, or None when this conversation can never carry receipts.

    Barrier-eligible by default because the responder turn is this function's only production
    caller; the detached route below opts out explicitly.
    """
    if not receipts_apply(channel_id):
        return None
    return ReceiptLedger(owner_id, team_id, channel_id, barrier_eligible=barrier_eligible)


def ledger_for_job(job_id: Any, team_id: Optional[str],
                   channel_id: Optional[str]) -> Optional[ReceiptLedger]:
    """Detached producers own their OWN ledger — they outlive the turn that started them, so they
    are never the turn a battery is holding at the partial-post seam."""
    return ledger_for(job_owner(job_id), team_id, channel_id, barrier_eligible=False)


async def settle_ledger(ledger: Optional[ReceiptLedger], turn: Any = None) -> None:
    """Settle under a shield: a cancelled turn must not strand its own live rows.

    The task is created explicitly and handed to the service rather than left anonymous inside
    `shield`: on timeout this function stops waiting, and an untracked task with nothing holding
    it can be collected mid-write.

    `turn` (optional) is waited on FIRST, to COMPLETION, for any effect lease it still holds.
    Settlement never proceeds with a lease open: a lease is held across an accepted post and the
    `note_post` that claims it, so settling through one is precisely the race that leaves a
    delivered message unclaimed forever. The wait is unbounded on purpose — each lease body is
    already bounded by the transport it runs under — and a failure of the wait ITSELF is allowed
    to propagate rather than be logged and stepped over: there is no state in which "the wait for
    live effects broke" is a reason to finalize this turn's receipts anyway.
    """
    if turn is not None:
        waiter = getattr(turn, "wait_for_effects", None)
        held: Any = waiter() if waiter is not None else None
        if hasattr(held, "__await__"):
            await held
    if ledger is None:
        return
    task = asyncio.ensure_future(ledger.settle())
    svc = ledger.service
    if svc is not None:
        svc.track(task)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=SETTLE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Receipt settle timed out for %s; the drain worker owns it now",
                       ledger.channel_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Receipt settle error for %s: %s", ledger.channel_id, e)


# --- transport intent -------------------------------------------------------------------


async def record_transport_post(*, team_id: Optional[str], channel_id: Optional[str],
                                message_ts: Optional[str], receipts: Optional[ReceiptLedger],
                                receipt_kind: Optional[str] = None,
                                receipt_class: Optional[str],
                                thread_root_ts: Optional[str] = None,
                                site: str = "transport") -> None:
    """The receipt intent contract for one durable post.

    A ledger claims it for its owner. A bare `receipt_kind` covers lifecycle posts no turn
    made. NEITHER is a plumbing bug, not a design: it is logged with a stack and the message
    is registered FINALIZED under the sys owner, because silently excluding real words from
    the stream is the one failure nobody would ever notice.

    `receipt_class` (spec §4) is a REQUIRED keyword with no default: the producer says what
    kind of surface this post is. A missing/invalid class is logged loudly and the row is
    written with NULL — class-ineligible, exactly like a legacy row — never inferred.
    """
    if not message_ts:
        return
    if receipts is not None:
        await receipts.note_post(message_ts, receipt_kind or STATE_IN_FLIGHT, thread_root_ts,
                                 receipt_class=receipt_class)
        return
    if not receipts_apply(channel_id):
        return
    svc = get_service()
    if svc is None or svc.db is None:
        return
    if receipt_kind not in RECEIPT_KINDS:
        logger.error(
            "Durable channel post at %s reached transport with no receipt intent "
            "(%s/%s) — registering it finalized under the sys owner",
            site, channel_id, message_ts, stack_info=True)
        receipt_kind = STATE_FINALIZED
    ledger = ReceiptLedger(sys_owner(), team_id, channel_id, service=svc)
    if receipt_kind == STATE_CHROME:
        await ledger.note_chrome(message_ts, thread_root_ts, receipt_class=receipt_class)
    else:
        await ledger.note_post(message_ts, STATE_FINALIZED, thread_root_ts,
                               receipt_class=receipt_class)


# --- uploads ----------------------------------------------------------------------------


async def record_pending_share(db: Any, *, team_id: Optional[str], channel_id: Optional[str],
                               file_id: Optional[str], owner_turn_id: str,
                               thread_root_ts: Optional[str] = None,
                               receipt_class: Optional[str]) -> bool:
    """Written the instant files_upload_v2 hands back a file id, before resolution starts.

    A row that is already there is success — the file id is unique per upload, so the only way
    to meet one is a retry of this same write. A raising DB is NOT: the accepted crash window
    covers a dead process, not a live one that hit a busy database, so the write is retained in
    the lattice and either drains later or is short-circuited by the resolution below.

    A file Slack has already confirmed deleted gets NO row. The two listeners are unordered, so
    this can genuinely arrive after the deletion — and the row it would write can never resolve.
    The tombstone is read TWICE, because one read cannot cover an OVERLAP: a deletion that
    starts after the first check can look for rows before this write commits, find none, and
    finish — leaving the row behind it. See `_drop_share_row_if_deleted`.
    """
    if not (db and file_id and receipts_apply(channel_id) and team_id):
        return False
    receipt_class = _checked_class(receipt_class, site="record_pending_share",
                                   message_ts=file_id)
    svc = get_service()
    if svc is not None and svc.file_is_deleted(file_id):
        logger.info("Upload %s completed after Slack confirmed the file gone — no pending row",
                    file_id)
        return False
    try:
        await db.record_pending_share_async(
            str(team_id), str(channel_id), str(file_id), owner_turn_id, thread_root_ts,
            receipt_class=receipt_class)
    except Exception as e:  # noqa: BLE001
        logger.warning("Pending share record failed for %s (queued for retry): %s", file_id, e)
        if svc is not None:
            svc.queue_pending_share(team_id=str(team_id), channel_id=str(channel_id),
                                    file_id=str(file_id), owner=owner_turn_id,
                                    thread_root_ts=thread_root_ts,
                                    receipt_class=receipt_class)
        return False
    if await _drop_share_row_if_deleted(db, svc, team_id, channel_id, file_id):
        return False
    return True


async def _drop_share_row_if_deleted(db: Any, svc: Optional[ReceiptService], team_id: Any,
                                     channel_id: Any, file_id: Any) -> bool:
    """Post-commit compensation. True when the row we just wrote has been taken away again.

    The deletion path stamps its tombstone BEFORE it looks for rows, and that ordering is what
    makes two reads enough. Whenever a deletion overlaps a registration, one of them sees the
    other: either the deletion's scan runs after this write commits (it finds the row and
    removes it), or the deletion stamped before this check (and the row is removed here). The
    window where the deletion scans an uncommitted write and this check runs before the stamp
    does not exist, because the stamp comes first.

    A failed delete is retained in the lattice rather than dropped — a pending row for a file
    Slack has destroyed can never resolve, and would be retried and logged critically on every
    boot for the life of the database. Retained INTERNALLY, because the lattice's own drain is
    one of the callers: during the final drain a normal enqueue would be refused, and the row
    this compensation exists to remove would survive the process.
    """
    if svc is None or not svc.file_is_deleted(file_id):
        return False
    logger.info("Pending share %s written into a deletion — removing it again", file_id)
    try:
        await db.delete_pending_share_async(str(team_id), str(channel_id), str(file_id))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not remove the pending row for deleted file %s (queued): %s",
                       file_id, e)
        svc.queue_share_deletion(team_id=str(team_id), channel_id=str(channel_id),
                                 file_id=str(file_id), internal=True)
    return True


async def resolve_pending_share(db: Any, *, team_id: Optional[str],
                                channel_id: Optional[str], file_id: Optional[str],
                                message_ts: Optional[str]) -> bool:
    """Finalize the share's receipt and drop the pending row, atomically (A1).

    A failure RETAINS the row: boot recovery retries it. Only a Slack-confirmed deletion of
    the file may remove a pending row without resolving it.

    If the pending row never reached the database (queued above), the queued write is consumed
    here instead of drained: the share ts is known now, so the message is finalized directly
    rather than writing a row only to delete it — and resolve_pending_share_async would treat
    the missing row as already-resolved and write nothing at all.
    """
    if not (db and file_id and message_ts and receipts_apply(channel_id) and team_id):
        return False
    svc = get_service()
    queued = (svc.take_pending_share(str(team_id), str(channel_id), str(file_id))
              if svc is not None else None)
    if queued is not None:
        # The queued op carries the producer's class (spec §4), so a share whose pending row
        # never reached the database still finalizes WITH its class present.
        # cast: a queued op only exists because `svc` handed one over just above.
        ok = await cast(ReceiptService, svc).apply(
            _Op("finalize", str(team_id), str(channel_id), str(message_ts),
                queued.owner, queued.thread_root_ts, queued.receipt_class))
        # A false here means the finalize was queued, not refused — the share is still outside the
        # stream, so the transition genuinely has not happened yet.
        _emit_transition(channel_id=channel_id, message_ts=message_ts, owner=queued.owner,
                         op="pending_resolve", applied=bool(ok), new_state=STATE_FINALIZED,
                         reason="queued_finalize" if ok else "write_queued")
        return ok
    try:
        ok = bool(await db.resolve_pending_share_async(
            str(team_id), str(channel_id), str(file_id), str(message_ts)))
    except Exception as e:  # noqa: BLE001
        logger.warning("Pending share resolve failed for %s (row retained): %s", file_id, e)
        return False
    _emit_transition(channel_id=channel_id, message_ts=message_ts, owner=None,
                     op="pending_resolve", applied=ok, new_state=STATE_FINALIZED,
                     reason="resolved" if ok else "not_resolved")
    return ok


def schedule_share_resolution(client: Any, db: Any, *, team_id: Optional[str],
                              channel_id: Optional[str], file_id: Optional[str],
                              site: str = "upload") -> Optional[asyncio.Task]:
    """Start the poll that turns an uploaded file into a finalized receipt.

    Image delivery runs its OWN resolve, because there one poll feeds three consumers (the
    "Uploading…" hold, F7 provenance and the receipt). Every other upload has no such consumer,
    so without this the file sits pending — absent from the rebuilt stream — until a restart
    recovers it.

    Detached: the file is already posted, and provenance must never be able to delay it.
    """
    resolve = getattr(client, "resolve_file_share_ts", None)
    if not (resolve is not None and db is not None and file_id
            and receipts_apply(channel_id) and team_id):
        return None
    svc = get_service()
    if svc is None or svc.db is None or not svc.accepting:
        return None

    async def _run() -> None:
        share_ts = await resolve(channel_id, file_id)
        if not share_ts:
            logger.warning(
                "Share ts never resolved for %s (%s) — its pending row is retained for boot "
                "recovery", file_id, site)
            return
        await resolve_pending_share(db, team_id=team_id, channel_id=channel_id,
                                    file_id=file_id, message_ts=share_ts)

    coro = _run()
    try:
        task = asyncio.ensure_future(coro)
    except RuntimeError:  # no running loop (sync test context)
        coro.close()
        return None
    svc.track_resolver(task)
    return task


async def delete_pending_shares_for_file(db: Any, file_id: Optional[str]) -> int:
    """Slack confirmed a file is gone: drop any pending row waiting to resolve it.

    Slack's `file_deleted` event carries only the file id, so the rows are looked up rather
    than addressed. Without this, an unresolvable deleted file is retried and logged critically
    on every boot for the life of the database.

    The tombstone is stamped FIRST, before any lookup. The upload-completion and `file_deleted`
    listeners are unordered, so the registration for this file may not have happened yet — and a
    cleanup that only removes what it can currently see is beaten by the one that arrives after.

    This runs from a live Slack listener, and Slack ingress outlives receipt shutdown, so a
    deletion can land here with the queue already closed. Every retain below is therefore
    always-retainable (`queue_share_deletion`), and shutdown's `drain_late_arrivals()` writes it.
    """
    if db is None or not file_id:
        return 0
    svc = get_service()
    removed = 0
    if svc is not None:
        svc.note_file_deleted(file_id)
        for op in svc.queued_pending_shares(file_id):
            svc.queue_share_deletion(team_id=op.team_id, channel_id=op.channel_id,
                                     file_id=file_id)
            removed += 1
    try:
        rows = await db.get_pending_shares_async()
    except Exception as e:  # noqa: BLE001
        # One transient read used to end it here, and Slack never sends `file_deleted` twice —
        # so the rows survived forever, retried and logged critically on every boot. The lattice
        # holds a file-wide deletion instead and retries it until it lands.
        logger.warning("Pending share cleanup could not read its rows (queued for retry): %s", e)
        if svc is not None:
            svc.queue_share_deletion(team_id="", channel_id="", file_id=file_id)
        return removed
    for row in rows or []:
        if row.get("file_id") != file_id:
            continue
        team_id, channel_id = row.get("team_id"), row.get("channel_id")
        if svc is not None:
            svc.queue_share_deletion(team_id=team_id, channel_id=channel_id, file_id=file_id)
        try:
            await db.delete_pending_share_async(team_id, channel_id, file_id)
            if svc is not None:
                svc.take_pending_share(team_id, channel_id, file_id)
            removed += 1
        except Exception as e:  # noqa: BLE001 — the tombstone above keeps it queued
            logger.debug("Pending share deletion failed for %s: %s", file_id, e)
    return removed


async def delete_receipt_for(*, team_id: Optional[str], channel_id: Optional[str],
                             message_ts: Optional[str], site: str = "raw_delete") -> None:
    """Drop the receipt for a message whose Slack deletion is CONFIRMED.

    For the raw post sites that delete their own surfaces without a ledger in hand. Routed
    through the service so the deletion tombstones anything still queued for that message.
    """
    if not (message_ts and receipts_apply(channel_id) and team_id):
        return
    svc = get_service()
    if svc is None or svc.db is None:
        return
    try:
        await svc.apply(_Op("delete", str(team_id), str(channel_id), str(message_ts),
                            sys_owner()))
    except Exception as e:  # noqa: BLE001
        logger.debug("Receipt deletion skipped at %s for %s: %s", site, message_ts, e)


async def recover_pending_shares(db: Any, client: Any) -> int:
    """Boot: retry every leftover pending share. A row that still can't resolve is kept."""
    resolve = getattr(client, "resolve_file_share_ts", None)
    if db is None or resolve is None:
        return 0
    try:
        rows = await db.get_pending_shares_async()
    except Exception as e:  # noqa: BLE001
        logger.warning("Pending share recovery could not read its rows: %s", e)
        return 0
    resolved = 0
    for row in rows or []:
        channel_id = row.get("channel_id")
        file_id = row.get("file_id")
        try:
            share_ts = await resolve(channel_id, file_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Pending share %s could not be resolved at boot: %s", file_id, e)
            continue
        if not share_ts:
            logger.critical(
                "Pending share %s in %s still has no share ts — the row is retained and that "
                "file stays outside the stream until it resolves", file_id, channel_id)
            continue
        if await resolve_pending_share(db, team_id=row.get("team_id"), channel_id=channel_id,
                                       file_id=file_id, message_ts=share_ts):
            resolved += 1
    if resolved:
        logger.info("Receipts: recovered %d pending share(s) at boot", resolved)
    return resolved


async def reconcile_dead_sessions(db: Any, live_session_id: Optional[str] = None) -> int:
    """Boot: finalize the previous session's in_flight rows, and say which messages they were.

    One event per recovered message rather than one for the batch: a count cannot tell a clean
    restart from a crash that stranded a specific reply, and those rows are the only evidence
    either way. The DB error is NOT swallowed — boot treats a failed reconcile as fatal, and a
    caller that cannot tell failure from "nothing to do" would start into a channel whose own
    half of the conversation is invisible.
    """
    rows = await db.finalize_dead_session_receipts_async(live_session_id or SESSION_ID)
    if not isinstance(rows, (list, tuple)):
        # A count, not the rows: no per-row detail exists to write, and the caller still wants
        # the number.
        try:
            return int(rows or 0)
        except (TypeError, ValueError):
            return 0
    for row in rows:
        get = row.get if isinstance(row, dict) else (lambda _key: None)
        _emit_transition(channel_id=get("channel_id"), message_ts=get("message_ts"),
                         owner=get("turn_id"), op="reconcile_finalize", applied=True,
                         prior_state=STATE_IN_FLIGHT, new_state=STATE_FINALIZED,
                         reason="dead_session")
    return len(rows)


async def establish_epoch(db: Any) -> str:
    """Write the grandfathering epoch once, then read it back.

    FATAL on failure by contract (the caller exits): without an epoch the rebuild cannot tell
    a legacy own-message from one this build simply failed to register, and it would either
    replay chrome forever or drop real replies.
    """
    from database import OUTBOUND_RECEIPTS_EPOCH_KEY
    import time as _time

    await db.set_meta_if_absent_async(OUTBOUND_RECEIPTS_EPOCH_KEY, f"{_time.time():.6f}")
    value = await db.get_meta_async(OUTBOUND_RECEIPTS_EPOCH_KEY)
    if not value:
        raise RuntimeError("outbound receipts epoch could not be established")
    return value
