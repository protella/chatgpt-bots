"""Thread-activity index feed + channel coverage bootstrap (spec §4).

The index answers one question for the stream builder: which roots have replies we have not
rendered yet. Pre-boundary roots can only be discovered this way — `history(oldest=boundary)`
never surfaces them, because the parent keeps its original ts.

Two feeds:
  * live — module-level hook called from BOTH raw Slack listeners, before every
    participation/listening/subtype filter and after the own-message check;
  * backfill — one persistent per-channel worker paging conversations.history backward to
    Slack's retention wall or the configured depth, so the retained-root inventory exists
    for channels that were busy before the bot ever booted.
"""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from config import config
from logger import setup_logger
from slack_client import admission_watermark
from slack_client.history_fetch import (HistoryPageError, HistoryPageInvalid, PageResult,
                                        fetch_page, retry_after_seconds, slack_error_code)
from slack_client.normalizer import (KIND_DELETE, KIND_EDIT, KIND_MESSAGE, KIND_TOMBSTONE,
                                     MUTATION_SUBTYPES, NormalizedEvent, TimestampError,
                                     mutation_kind, mutation_observation_identity,
                                     mutation_subject_ts, normalize_slack_event, parse_ts,
                                     secondary_ts)
from slack_client.utilities import is_dm_conversation

# Must go through setup_logger: handlers are attached to `slack_bot.*` loggers with
# propagate=False, so a bare getLogger(__name__) writes to NOWHERE — every coverage sweep and
# index line in this module would be invisible in a normal run.
logger = setup_logger(name="slack_bot.ActivityIndex")

_IDENTITY_POLL_SECONDS = 0.5
# In-place retries before a channel is declared degraded. A contended WAL writer is the common
# failure and it clears in milliseconds; degrading a channel is the expensive answer.
_FEED_ATTEMPTS = 3
_FEED_RETRY_BASE_SECONDS = 0.05
_RESUME_BASE_SECONDS = 60.0
_RESUME_JITTER_SECONDS = 15.0
_DISCOVERY_PAGE_LIMIT = 200
_CONVERSATION_TYPES = "public_channel,private_channel,mpim"
TERMINAL_COVERAGE_STATUSES = ("complete", "limited")
_TERMINAL_STATUSES = TERMINAL_COVERAGE_STATUSES
# The channel itself is unreachable — persisted as `limited`/'unavailable' so the worker stops
# and no other process keeps thinking the sweep is in progress. Unlike a retention or depth wall
# this one can reverse (re-invite, unarchive), so `reset_channel_coverage` reclaims it.
_CHANNEL_GONE_ERRORS = frozenset({"channel_not_found", "not_in_channel", "is_archived"})
_UNAVAILABLE_REASON = "unavailable"
# App-level refusals: nothing about THIS channel is settled, and reinstalling a scope or a
# token fixes every one of them. Persisting a terminal state here would outlive the fix, so
# the worker abandons the pass and the stale-heartbeat window brings the channel back.
_APP_LEVEL_ERRORS = frozenset({
    "missing_scope", "invalid_auth", "account_inactive", "access_denied",
})


class ActivityObservation(NamedTuple):
    """One normalized index observation. `root_ts` None means the event is real channel
    activity worth seeding coverage for, but carries nothing the index records.

    `root_if_indexed` marks a root_ts that is a SUSPICION rather than a fact: a mutated
    top-level message with no threading hints at all. It is recorded only if the index already
    knows that ts as a root, so a plain top-level edit still writes nothing.

    `owner_probe_ts` is the ts of a DELETED message whose payload named no sender: the caller
    must ask the receipt ledger whether it was ours before recording anything.
    """
    team_id: str
    channel_id: str
    kind: str
    root_ts: Optional[str]
    reply_ts: Optional[str]
    event_ts: Optional[str]
    mark_dirty: bool
    sender_type: str
    root_if_indexed: bool = False
    owner_probe_ts: Optional[str] = None


def _ts_key(raw: Any) -> Optional[Tuple[int, int]]:
    """The shared comparator's key, or None when the ts cannot be read.

    There is ONE comparator (normalizer.parse_ts) and this is not a second one: a float conversion
    decides boundary cases by rounding, and coverage compares timestamps at exactly the boundary.
    """
    try:
        return parse_ts(raw)
    except TimestampError:
        return None


# NormalizedEvent.kind → the index's own vocabulary. A tombstone is a deletion as far as
# activity goes: the message is gone, and its thread's replies are not.
_KIND_NAMES = {KIND_MESSAGE: "new", KIND_EDIT: "edited", KIND_DELETE: "deleted",
               KIND_TOMBSTONE: "deleted"}


def observation_from_event(event: NormalizedEvent) -> Optional[ActivityObservation]:
    """NormalizedEvent → index observation, or None to skip. Pure."""
    kind = _KIND_NAMES.get(event.kind)
    if kind is None:
        return None
    message = event.message
    sender_type = message.sender_type if message is not None else "human"
    if sender_type == "self":
        return None

    def _nothing() -> ActivityObservation:
        return ActivityObservation(event.team_id, event.channel_id, kind, None, None, None,
                                   False, sender_type)

    if message is None:
        return _nothing()
    if kind == "new":
        if not message.is_reply:
            return _nothing()
        return ActivityObservation(event.team_id, event.channel_id, "new",
                                   message.thread_root_ts, message.ts, message.ts,
                                   False, sender_type)
    # A mutated message with no thread_ts of its own is its own root — a suspicion the caller
    # arbitrates against the index (root_if_indexed).
    root_ts = message.thread_root_ts or message.ts
    return ActivityObservation(event.team_id, event.channel_id, kind, root_ts, None,
                               event.activity_ts, True, sender_type, event.root_if_indexed,
                               event.owner_probe_ts)


def mutation_from_event(event: NormalizedEvent, *,
                        observed_at: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """NormalizedEvent → one durable snapshot-invalidation observation (spec §1c), or None.

    EVERY `message_changed` and `message_deleted` in a channel produces one, including the
    unthreaded top-level edit the activity index writes nothing for and including our own
    messages: the record says a summarized message changed, and a snapshot covering it is wrong
    from that moment whoever did it. `observed_at` is when WE saw it, so it is fixed once by the
    caller and carried across every replay rather than re-read from the clock.
    """
    kind = mutation_kind(event)
    if kind is None:
        return None
    return {
        "team_id": event.team_id,
        "channel_id": event.channel_id,
        "subject_ts": event.subject_ts,
        "kind": kind,
        "observation_identity": mutation_observation_identity(event),
        "observed_at": observed_at or f"{time.time():.6f}",
    }


def _event_channel(event: Any) -> Optional[str]:
    if not isinstance(event, dict):
        return None
    return event.get("channel") or (event.get("item") or {}).get("channel")


def _event_subject_ts(event: Any) -> Optional[str]:
    """The ts a malformed-event log should name — the subject's, which is the one that failed."""
    if not isinstance(event, dict):
        return None
    subject_ts = mutation_subject_ts(event)
    if subject_ts:
        return subject_ts
    if event.get("subtype") not in MUTATION_SUBTYPES and event.get("ts"):
        return str(event["ts"])
    return event.get("event_ts")


def normalize_activity_event(client: Any, event: Any) -> Optional[ActivityObservation]:
    """Raw Slack event → index observation, or None to skip.

    A thin derivation over the one canonical normalizer (slack_client/normalizer.py): the
    effective-message resolution, sender classification, DM exclusion and anonymous-deletion
    probe all live there now, so the stream, the actor tail and this index cannot disagree about
    what an event was.
    """
    try:
        normalized = normalize_slack_event(client, event)
    except ValueError as e:
        # A malformed ts or an unresolvable mutation. This pure helper has no ticket to fail, so it
        # logs and returns None; the LISTENER path (feed_thread_activity_index) is the one that
        # holds the ticket and fails the observation.
        logger.warning(f"activity index could not normalize an event: {e}")
        return None
    if normalized is None:
        return None
    return observation_from_event(normalized)


def _seeded_channels(client: Any) -> Set[str]:
    seen = getattr(client, "_coverage_seeded_channels", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            client._coverage_seeded_channels = seen
        except Exception:  # noqa: BLE001
            pass
    return seen


async def _seed_coverage(client: Any, db: Any, team_id: str, channel_id: str) -> None:
    """Lazy seed: the first time we see a channel, give it a concrete horizon to walk back
    from. INSERT OR IGNORE, so an existing row's coverage is never disturbed."""
    seen = _seeded_channels(client)
    if channel_id in seen:
        return
    seeder = getattr(db, "seed_channel_coverage_async", None)
    if not callable(seeder):
        return
    seen.add(channel_id)
    try:
        await seeder(team_id, channel_id, f"{time.time():.6f}")
    except Exception:
        seen.discard(channel_id)
        raise


async def reset_channel_coverage(client: Any, channel_id: str) -> bool:
    """Reactivation hook for a channel we had written off: seed it if it is new, and clear the
    `unavailable` verdict so the sweep can claim it again. True when a verdict was cleared."""
    db = getattr(client, "db", None)
    team_id = getattr(client, "self_team_id", None)
    if db is None or not team_id or not channel_id or is_dm_conversation(channel_id):
        return False
    resetter = getattr(db, "reset_channel_coverage_async", None)
    if not callable(resetter):
        return False
    team_id = str(team_id)
    try:
        await _seed_coverage(client, db, team_id, channel_id)
        reactivated = bool(await resetter(team_id, channel_id))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"coverage reset failed for {channel_id}: {e}")
        return False
    if reactivated:
        boot = getattr(client, "_coverage_bootstrap", None)
        reclaim = getattr(boot, "reclaim", None)
        if callable(reclaim):
            reclaim(team_id, channel_id)
    return reactivated


async def _deletion_was_ours(db: Any, observation: ActivityObservation) -> bool:
    """The receipt ledger is the oracle for an anonymous deletion: a row in ANY state means we
    posted that ts, so removing it is our own housekeeping and not channel activity — the
    "Uploading…" indicator behind every image post is the loudest case.

    Accepted gap: a message posted before the receipts epoch has no row, so deleting it still
    reads as somebody else's. The epoch bounds that and it decays as the epoch recedes.
    """
    getter = getattr(db, "get_receipt_async", None)
    if not callable(getter):
        return False
    try:
        receipt = await getter(observation.team_id, observation.channel_id,
                               observation.owner_probe_ts)
    except Exception as e:  # noqa: BLE001
        # Fail OPEN, matching the file's standing bias toward an extra fetch over a missed root.
        logger.warning(f"receipt oracle unavailable for an anonymous deletion: {e}")
        return False
    return receipt is not None


async def _index_row(db: Any, client: Any,
                     observation: ActivityObservation) -> Optional[Dict[str, Any]]:
    """The index half of one feed, arbitrated: the row to upsert, or None to write nothing.

    The arbitration reads (the receipt oracle, the is-this-a-known-root probe) happen HERE,
    outside the write, so the write itself stays the one transaction §1c requires.
    """
    await _seed_coverage(client, db, observation.team_id, observation.channel_id)
    if observation.root_ts is None:
        return None
    if observation.owner_probe_ts and await _deletion_was_ours(db, observation):
        return None
    if observation.root_if_indexed:
        probe = getattr(db, "thread_activity_exists_async", None)
        if not callable(probe) or not await probe(
                observation.team_id, observation.channel_id, observation.root_ts):
            return None
    return {
        "team_id": observation.team_id,
        "channel_id": observation.channel_id,
        "root_ts": observation.root_ts,
        "reply_ts": observation.reply_ts,
        "event_ts": observation.event_ts,
        "mark_dirty": observation.mark_dirty,
    }


async def _apply_observation(client: Any, observation: Optional[ActivityObservation],
                             mutation: Optional[Dict[str, Any]] = None) -> None:
    """The DB half of one feed: the index upsert and the mutation observation, ONE transaction.

    Raises on a write failure — the ticket needs to know. The two halves commit together because
    a replay that re-ran only the index half would leave the snapshot store believing nothing
    changed, and a snapshot published in that gap would summarize a message it no longer matches.

    Either half may be absent and the other still writes: an unthreaded top-level edit records no
    index row at all, and an ordinary reply is no mutation.
    """
    db = getattr(client, "db", None)
    if db is None:
        return
    index_row = await _index_row(db, client, observation) if observation is not None else None
    if index_row is None and mutation is None:
        return
    await db.record_activity_and_mutation_async(observation=index_row, mutation=mutation)


async def feed_thread_activity_index(client: Any, event: Any, *, ticket: Any = None) -> None:
    """Index + invalidation hook for the raw listeners. Never raises — a failure here must never
    cost a turn.

    It does, however, REPORT. A turn whose window contains this event waits on `ticket`, so a
    swallowed failure that told nobody would let that turn render a stream with a thread quietly
    missing from it. Three in-place attempts first (a busy WAL writer is the common case and
    retrying beats degrading a channel), then the ticket is failed with the normalized event
    retained for the background repair worker.

    OFFLINE RESIDUAL (§1c): this feed only sees what Slack delivers to a running process. A
    mutation landing AFTER PUBLICATION while we are down leaves no observation and nothing later
    re-reads that span — see the compaction module docstring for the scoped statement.
    """
    normalized = None
    observation = None
    mutation = None
    try:
        normalized = normalize_slack_event(client, event)
        observation = observation_from_event(normalized) if normalized else None
        mutation = mutation_from_event(normalized) if normalized else None
    except ValueError as e:
        # A ts the shared comparator cannot read, or a known mutation whose subject is missing.
        # Nothing can be recorded and nothing can be replayed, so completing this ticket OK would
        # tell every turn in its window that the index is caught up on an event we could not place
        # at all. The outer event_ts has already advanced H by this point, which is exactly why
        # this must fail rather than pass quietly.
        admission_watermark.fail_observation(
            ticket, channel_id=_event_channel(event), ts=_event_subject_ts(event),
            reason=f"an indexable event could not be normalized: {e}")
        return
    except Exception as e:  # noqa: BLE001
        # NOT logged and walked past. Falling through here reached `complete_ok`, so a normalizer
        # bug or an unforeseen payload shape certified that the index had caught up on an event
        # nobody indexed — the exact silent drop the ticket exists to prevent, and the opposite of
        # fail-closed. The harmless outcomes are the DECLARED ones: an unknown or declined kind
        # comes back as None below, and a known-but-unreadable one raises ValueError above.
        # Nothing is retained for replay: re-normalizing the same payload raises the same way.
        logger.exception("thread-activity index normalization raised unexpectedly")
        admission_watermark.fail_observation(
            ticket, channel_id=_event_channel(event), ts=_event_subject_ts(event),
            reason=f"unexpected {type(e).__name__} while normalizing an indexable event: {e}")
        return
    if observation is None and mutation is None:
        admission_watermark.complete_ok(ticket)
        return

    async def _replay() -> bool:
        # BOTH halves, always: a replay of the index half alone would complete the ticket while
        # the mutation nobody recorded stayed unrecorded. Both writes are idempotent.
        try:
            await _apply_observation(client, observation, mutation)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"thread-activity index replay failed: {e}")
            return False

    last: Optional[BaseException] = None
    for attempt in range(_FEED_ATTEMPTS):
        try:
            await _apply_observation(client, observation, mutation)
            admission_watermark.complete_ok(ticket)
            return
        except asyncio.CancelledError as e:
            # A cancelled listener callback must not leave the ticket pending: every later turn
            # whose frontier includes it would wait out the drain timeout and fail closed.
            admission_watermark.complete_failed(ticket, event=normalized, retry=_replay)
            raise e
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < _FEED_ATTEMPTS - 1:
                await asyncio.sleep(_FEED_RETRY_BASE_SECONDS * (2 ** attempt))
    logger.warning(f"thread-activity index feed failed: {last}")
    admission_watermark.complete_failed(ticket, event=normalized, retry=_replay)


async def feed_own_mutation(client: Any, channel_id: Optional[str], subject_ts: Optional[str],
                            kind: str, *, operation_id: Optional[str] = None) -> bool:
    """Record a mutation WE performed (spec §1c own-message mutations). NON-WAKE.

    It advances no watermark, takes no ticket and reaches no gate: our own edit is not channel
    activity a turn may wait on, and H moving for it would let a turn wait on its own reply. What
    it is, is evidence that a message a snapshot may have summarized no longer says what the
    snapshot says it said.

    `operation_id` must be UNIQUE PER OPERATION and stable only across RETRIES of that one
    operation — it is the never-null observation identity, so a value reused by a later edit
    would collapse into the first row and that later edit would invalidate nothing. Omitting it
    mints one, which is correct for a one-shot call and wrong for a caller that retries.

    Never raises: a chrome delete must not be able to break the turn that made it.
    """
    db = getattr(client, "db", None)
    team_id = getattr(client, "self_team_id", None)
    if db is None or not team_id or not channel_id or not subject_ts:
        return False
    if is_dm_conversation(channel_id):
        return False
    if kind not in ("edit", "delete"):
        logger.warning(f"own mutation feed refused an unknown kind: {kind!r}")
        return False
    try:
        parse_ts(subject_ts)
    except TimestampError:
        logger.warning(f"own mutation feed got an unreadable subject ts: {subject_ts!r}")
        return False
    mutation = {
        "team_id": str(team_id),
        "channel_id": str(channel_id),
        "subject_ts": str(subject_ts),
        "kind": kind,
        # An EMPTY identity defeats the unique key exactly as a NULL one does, so a blank id
        # mints a real one rather than being passed down.
        "observation_identity": f"op:{(operation_id or '').strip() or uuid.uuid4().hex}",
        "observed_at": f"{time.time():.6f}",
    }
    last: Optional[BaseException] = None
    for attempt in range(_FEED_ATTEMPTS):
        try:
            await db.record_activity_and_mutation_async(observation=None, mutation=mutation)
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < _FEED_ATTEMPTS - 1:
                await asyncio.sleep(_FEED_RETRY_BASE_SECONDS * (2 ** attempt))
    # No ticket to fail — this path never took one, so nothing will retry it. CRITICAL for
    # exactly that reason. Best-effort is defensible because the DELIVERY feed covers the same
    # edit whenever the socket is up and repairs its own write failures through its ticket: for
    # both to fail at once the delivery must never have arrived, which is the §1c OFFLINE
    # RESIDUAL. This is a sibling of that documented residual, not an independent gap — do not
    # "fix" it into something that can fail a turn.
    logger.critical(
        f"own mutation observation lost for {channel_id}/{subject_ts} ({kind}): {last}")
    return False


class _SweepTokenLost(Exception):
    """Another worker took this channel's claim."""


class ChannelCoverageBootstrap:
    """Background sweep that extends each channel's coverage backward.

    One persistent worker per claimed channel holds its sweep token to a terminal state: a
    page-ceiling pause parks and resumes with the claim still held, so a deep channel is
    covered across passes instead of one unbounded burst. The concurrency semaphore bounds
    channels doing Slack work at once and is released across every sleep.
    """

    def __init__(self, client: Any, db: Any = None, cfg: Any = None):
        self.client = client
        self.db: Any = db if db is not None else getattr(client, "db", None)
        self.config = cfg if cfg is not None else config
        self._task: Optional[asyncio.Task] = None
        self._workers: Dict[Tuple[str, str], asyncio.Task] = {}
        self._settled: Set[Tuple[str, str]] = set()
        self._discovered = False
        self._semaphore = asyncio.Semaphore(
            max(1, int(self.config.coverage_sweep_concurrency)))
        self._stopping = False
        # Event handlers reach the sweep through the client, which is the only object they hold.
        try:
            client._coverage_bootstrap = self
        except Exception:  # noqa: BLE001
            pass

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is not None or self.db is None:
            return
        self._task = asyncio.create_task(self._supervise())
        self._task.add_done_callback(self._log_task_error)

    async def stop(self) -> None:
        self._stopping = True
        tasks = [t for t in ([self._task] + list(self._workers.values())) if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._task = None

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(f"coverage bootstrap task error: {exc}")

    # -- supervision ----------------------------------------------------------------------

    async def _supervise(self) -> None:
        team_id = await self._await_identity()
        if not team_id:
            return
        while not self._stopping:
            if not self._discovered:
                # A workspace that was rate-limited or briefly unreachable at boot would
                # otherwise be discovered never; every tick retries until one walk completes.
                self._discovered = await self._discover_channels(team_id)
            for key, task in list(self._workers.items()):
                if task.done():
                    self._workers.pop(key, None)
            for channel_id in sorted(_seeded_channels(self.client)):
                key = (team_id, channel_id)
                if key in self._workers or key in self._settled:
                    continue
                if is_dm_conversation(channel_id):
                    continue
                worker = asyncio.create_task(self._sweep_channel(team_id, channel_id))
                worker.add_done_callback(self._log_task_error)
                self._workers[key] = worker
            await asyncio.sleep(self._resume_delay())

    async def _await_identity(self) -> Optional[str]:
        """Both halves of our identity are required: team_id keys every row, and bot_user_id
        proves auth.test landed (without it own-message checks would silently pass)."""
        while not self._stopping:
            if getattr(self.client, "bot_user_id", None) and \
                    getattr(self.client, "self_team_id", None):
                return str(self.client.self_team_id)
            await asyncio.sleep(_IDENTITY_POLL_SECONDS)
        return None

    async def _discover_channels(self, team_id: str) -> bool:
        """Seed every conversation we are a member of. True when the walk is done with — a
        transient failure returns False so the next scheduler tick tries again. A channel whose
        seed failed makes the whole walk incomplete: it is quiet by definition (nothing else
        would ever put it in the seeded set), so a retry is the only way it gets covered.

        The page cap (`history_page_ceiling` pages of 200) bounds the membership walk; hitting
        it is not transient, so it counts as done rather than retrying the same prefix forever.
        """
        getter = self._web_method("users_conversations")
        if getter is None:
            return True
        complete = True
        cursor = ""
        for _ in range(max(1, int(self.config.history_page_ceiling))):
            kwargs: Dict[str, Any] = {
                "types": _CONVERSATION_TYPES,
                "limit": _DISCOVERY_PAGE_LIMIT,
                "exclude_archived": True,
            }
            if cursor:
                kwargs["cursor"] = cursor
            try:
                resp = await getter(**kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"coverage bootstrap: users.conversations failed ({e}); "
                    "retrying on the next tick")
                return False
            for entry in ((resp or {}).get("channels") or []):
                if not isinstance(entry, dict):
                    continue
                channel_id = entry.get("id")
                if not isinstance(channel_id, str) or entry.get("is_archived"):
                    continue
                if is_dm_conversation(channel_id):
                    continue
                try:
                    await _seed_coverage(self.client, self.db, team_id, channel_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"coverage seed failed for {channel_id}: {e}")
                    complete = False
            cursor = ((resp or {}).get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                return complete
        logger.warning(
            "coverage bootstrap: users.conversations stopped at the "
            f"{self.config.history_page_ceiling}-page cap "
            f"({self.config.history_page_ceiling * _DISCOVERY_PAGE_LIMIT} conversations); "
            "raise HISTORY_PAGE_CEILING for a larger workspace")
        return complete

    # -- per-channel sweep ----------------------------------------------------------------

    async def _sweep_channel(self, team_id: str, channel_id: str) -> None:
        token = uuid.uuid4().hex
        try:
            acquired = await self.db.acquire_coverage_sweep_async(team_id, channel_id, token)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"coverage claim failed for {channel_id}: {e}")
            return
        if not acquired:
            await self._settle_if_terminal(team_id, channel_id)
            return
        logger.info(f"coverage sweep claimed {channel_id}")
        try:
            while not self._stopping:
                outcome = await self._sweep_pass(team_id, channel_id, token)
                if outcome == "park":
                    await self._park(team_id, channel_id, token)
                    continue
                if outcome == "gone":
                    # Settle ONLY on a persisted verdict: an unreachable channel that stayed
                    # `running` in the DB would be a worker this process never retries and a row
                    # every other process still thinks is in progress. A failed write leaves it
                    # unsettled so the stale-heartbeat window brings it back.
                    if await self._mark_unavailable(team_id, channel_id, token):
                        self._settled.add((team_id, channel_id))
                elif outcome == "terminal":
                    self._settled.add((team_id, channel_id))
                # Everything else leaves the row `running` and stays out of _settled, so the
                # stale-heartbeat window is what brings the channel back.
                return
        except _SweepTokenLost:
            logger.info(f"coverage sweep for {channel_id} lost its claim")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"coverage sweep for {channel_id} stopped: {e}")

    async def _mark_unavailable(self, team_id: str, channel_id: str, token: str) -> bool:
        try:
            return bool(await self.db.advance_channel_coverage_async(
                team_id, channel_id, token, None, "limited", _UNAVAILABLE_REASON))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"coverage could not be marked unavailable for {channel_id}: {e}")
            return False

    def reclaim(self, team_id: str, channel_id: str) -> None:
        """Forget a settled channel so the next supervisor tick spawns a worker for it again."""
        self._settled.discard((team_id, channel_id))
        _seeded_channels(self.client).add(channel_id)

    async def _settle_if_terminal(self, team_id: str, channel_id: str) -> None:
        try:
            row = await self.db.get_channel_coverage_async(team_id, channel_id)
        except Exception:  # noqa: BLE001
            return
        if row and row.get("bootstrap_status") in _TERMINAL_STATUSES:
            self._settled.add((team_id, channel_id))

    async def _sweep_pass(self, team_id: str, channel_id: str, token: str) -> str:
        """One pass: up to `history_page_ceiling` pages walked backward from the persisted
        coverage_start_ts. Returns terminal | gone | park | abandon | lost | retry."""
        row = await self.db.get_channel_coverage_async(team_id, channel_id)
        if not row:
            return "retry"
        if row.get("bootstrap_status") in _TERMINAL_STATUSES:
            return "terminal"
        if row.get("sweep_token") != token:
            return "lost"

        latest = row.get("coverage_start_ts")
        # The depth wall as a comparator key, so the page's oldest ts is judged by the same
        # arithmetic the window predicate uses rather than by float rounding.
        floor = parse_ts(f"{time.time() - max(1, int(self.config.coverage_bootstrap_days)) * 86400:.6f}")
        cursor = ""
        for _ in range(max(1, int(self.config.history_page_ceiling))):
            page, failure = await self._fetch_page(team_id, channel_id, token, latest, cursor)
            if page is None:
                return failure or "park"
            async with self._semaphore:
                oldest, page_complete = await self._process_page(
                    team_id, channel_id, list(page.messages))
            if not page_complete:
                # Coverage stays where it is. The row is left `running`, so the channel reads as
                # NOT bootstrapped and its turns fail closed — which is the truth: there is a
                # record in its history we cannot place, and a horizon past it would be a lie
                # that no later sweep would ever correct.
                logger.error(
                    f"coverage sweep for {channel_id} found a history record it could not "
                    "place (missing or unreadable timestamp); coverage is left where it was "
                    "rather than claiming a horizon past it")
                return "abandon"

            is_limited = page.is_limited
            has_more = page.has_more
            next_cursor = page.next_cursor

            if is_limited:
                status, reason = "limited", "retention"
            elif not has_more and not next_cursor:
                status, reason = "complete", "genesis"
            elif oldest is not None and (_ts_key(oldest) or (0, 0)) <= floor:
                status, reason = "limited", "depth_config"
            else:
                status, reason = "running", None

            # Page-atomic: coverage only ever moves to the oldest ts of a page whose every
            # message was already recorded.
            advanced = await self.db.advance_channel_coverage_async(
                team_id, channel_id, token, oldest, status, reason)
            if not advanced:
                return "lost"
            if status != "running":
                logger.info(f"coverage sweep for {channel_id} finished: {status}/{reason}")
                return "terminal"
            if next_cursor:
                cursor = next_cursor
                continue
            # No cursor to follow: walk on from the page's own oldest ts instead.
            if oldest is None:
                # Slack says there is more but hands us neither a message nor a cursor. We
                # cannot honestly call the channel covered, so leave it `running` and let the
                # stale window retry rather than freezing a mismatched pair of states.
                logger.warning(
                    f"coverage sweep for {channel_id} got an empty page with has_more and no "
                    "cursor; leaving coverage running for a later retry")
                return "abandon"
            cursor = ""
            latest = oldest
        return "park"

    async def _fetch_page(self, team_id: str, channel_id: str, token: str,
                          latest: Optional[str],
                          cursor: str) -> Tuple[Optional[PageResult], Optional[str]]:
        """One backward page via the shared pager, with the sweep's own park-and-heartbeat sleep.

        `inclusive=False` is the sweep's half of the pager's one parameter: it resumes from a ts
        it has already processed, so including it would re-walk the same page forever. The turn
        path is inclusive and applies its own window predicate.
        """
        getter = self._web_method("conversations_history")
        if getter is None:
            logger.error(
                "coverage bootstrap: no conversations.history method on this client; "
                "coverage stays running")
            return None, "abandon"

        async def _guarded(**kwargs: Any) -> Any:
            # Held only while a page is actually in flight; every sleep happens outside it.
            async with self._semaphore:
                return await getter(**kwargs)

        async def _sleeper(delay: float) -> None:
            await self._park(team_id, channel_id, token, delay)

        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": max(1, int(self.config.history_page_size)),
            "inclusive": False,
        }
        if latest:
            params["latest"] = str(latest)
        if cursor:
            params["cursor"] = cursor
        try:
            page = await fetch_page(_guarded, params,
                                    attempts=int(self.config.fetch_retry_attempts),
                                    sleeper=_sleeper, require_ts=True,
                                    label=f"coverage history {channel_id}")
        except HistoryPageInvalid as e:
            # Slack answered with something that is not a page, or with a record this sweep cannot
            # place. Same verdict as a malformed record found during processing (below): coverage
            # is left exactly where it was, because a horizon over records we never read is a
            # claim no later pass would ever correct.
            logger.error(
                f"coverage history for {channel_id} came back unreadable ({e}); coverage is left "
                "where it was rather than claiming a horizon past it")
            return None, "abandon"
        except HistoryPageError as e:
            outcome = _failure_outcome(e.code)
            if outcome != "park":
                logger.warning(f"coverage history refused for {channel_id}: {e.code}")
            return None, outcome
        except asyncio.CancelledError:
            raise
        except _SweepTokenLost:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"coverage history error for {channel_id}: {e}")
            return None, "park"
        return page, None

    async def _process_page(self, team_id: str, channel_id: str,
                            messages: List[Dict[str, Any]]
                            ) -> Tuple[Optional[str], bool]:
        """Record every parent hint on the page and return (oldest ts, page was complete).

        `complete` is False when any record on the page could not be processed — not a dict, no
        ts, or a ts the comparator cannot read, in ANY of the timestamp-bearing fields. Coverage
        may not move past a record we skipped: coverage_start_ts is the claim "everything at or
        after this is in the index", and advancing over an unreadable record turns a gap into a
        horizon nothing will ever revisit.

        The secondary timestamps are checked for the same reason and are not a lesser case: a row
        keyed on an unreadable `thread_ts`, or holding an unreadable `latest_reply`, is a poisoned
        index row — discovered, never placeable, and sitting under a horizon that says the page it
        came from was fully recorded.

        A `latest_reply` is persisted whenever Slack sends one, even with no positive
        `reply_count` beside it. It used to require the count, and the count is the field Slack
        omits on a trimmed or otherwise unusual parent — so a root whose only advertisement was
        `latest_reply` was recorded as nothing at all, and the pre-boundary thread it named
        stayed invisible to every stream that followed. A count with no latest_reply is the
        mirror case: replies exist, their position is unknown, which is what dirty means.

        Parents are recorded regardless of author: a thread rooted on our own post still holds
        other people's replies, and the inventory would lose it otherwise.
        """
        oldest: Optional[str] = None
        oldest_key: Optional[Tuple[int, int]] = None
        complete = True
        for message in messages:
            if not isinstance(message, dict):
                complete = False
                continue
            ts = message.get("ts")
            if not ts:
                complete = False
                continue
            key = _ts_key(ts)
            if key is None:
                # An unreadable ts is not recorded either: a root keyed on a ts no window
                # predicate can compare would be discovered and then never placed.
                complete = False
                continue
            if oldest_key is None or key < oldest_key:
                oldest, oldest_key = str(ts), key
            count = message.get("reply_count")
            positive_count = count if isinstance(count, int) and count > 0 else None
            try:
                latest_reply = secondary_ts(message, "latest_reply")
                thread_ts = secondary_ts(message, "thread_ts")
            except ValueError:
                # Nothing is written for this parent: the row it would produce is exactly the
                # poisoned one described above. The page is incomplete, so coverage stays put and
                # this record is walked again rather than buried under a horizon. A field that is
                # PRESENT and empty lands here too — reading `""` as "no thread" is how coverage
                # advances over a thread it never recorded.
                complete = False
                continue
            if positive_count is None and not latest_reply:
                continue
            await self.db.record_thread_activity_async(
                team_id, channel_id, str(thread_ts or ts),
                reply_ts=str(latest_reply) if latest_reply else None,
                reply_count=positive_count, mark_dirty=not latest_reply)
        return oldest, complete

    async def _park(self, team_id: str, channel_id: str, token: str,
                    delay: Optional[float] = None) -> None:
        """Sleep with the claim held and the semaphore released. A heartbeat that no longer
        matches means somebody else owns the channel now — stop rather than fight for it."""
        try:
            alive = await self.db.heartbeat_coverage_sweep_async(team_id, channel_id, token)
        except Exception:  # noqa: BLE001
            alive = True
        if not alive:
            raise _SweepTokenLost(channel_id)
        await asyncio.sleep(self._resume_delay() if delay is None else max(0.0, delay))

    # -- helpers --------------------------------------------------------------------------

    def _resume_delay(self) -> float:
        return _RESUME_BASE_SECONDS + random.uniform(-_RESUME_JITTER_SECONDS,
                                                     _RESUME_JITTER_SECONDS)

    def _web_method(self, name: str):
        app = getattr(self.client, "app", None)
        web = getattr(app, "client", None) if app is not None else None
        getter = getattr(web, name, None)
        if callable(getter):
            return getter
        getter = getattr(self.client, name, None)
        return getter if callable(getter) else None


def _failure_outcome(err: Any) -> str:
    if err in _CHANNEL_GONE_ERRORS:
        return "gone"
    if err in _APP_LEVEL_ERRORS:
        return "abandon"
    return "park"


# Error-shape helpers moved to slack_client/history_fetch.py with the pager; re-exported so an
# existing caller (and the sweep's own tests) keeps working.
_slack_error = slack_error_code
_retry_after = retry_after_seconds
