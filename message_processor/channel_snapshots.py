"""The compaction coordinator and its lifecycle (plan §1f, §1l, §1m).

SAME-TURN COMPACTION DOES NOT EXIST. All three trigger paths enqueue BACKGROUND work and return
immediately: no turn awaits a compaction, re-pins a winner under its original H, or rebuilds its
stream. That one ruling is what makes everything below straightforward — there is exactly one
canonical stream build per turn, and the coordinator never has a turn waiting on it.

What lives here:

- SELECTION AND PINNING for the turn path (§1b), including the retirement-pending set new
  selection refuses.
- PUBLICATION, whose channel lock is scoped strictly inside `publish()` (§1m deadlock fix:
  `channel_lock()` is non-reentrant, so a caller holding it and a publish that takes it would
  wedge; `lock_held=True` is how the one caller that already holds it says so).
- THE THREE TRIGGER PATHS, all background, all arbitrating through the dormant
  `pending_recompaction` row BEFORE any task is created. Suppressing only the obligation's own
  recompaction would achieve nothing: paths 1-3 would enqueue an ordinary compaction for the same
  channel and walk past the backoff by the front door.
- SINGLE-FLIGHT per `(team, channel, namespace)`, with a COMPLETION HOOK that re-runs row
  arbitration when a task vacates the slot, so a successor obligation does not sit idle on a quiet
  channel waiting for a trigger that may never come.
- THREE-PHASE RETIREMENT (§1f) and the R0-5 rollback, which are the two paths that delete a
  published generation out from under live readers — and must not.
- THE TELEMETRY DRAINER (§1l), which emits outbox rows in `outbox_seq` order and deletes a row
  only after durable acknowledgement.

ACCEPTED RESIDUAL, stated where the module can be read: a mutation that occurs AFTER PUBLICATION
while this process is offline writes no observation row, and nothing later re-reads the span to
notice it. Everything else is covered — a crawl fingerprints its sources, so an edit before or
during one is caught there.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config import config
from database import PROD_NAMESPACE
from slack_client import admission_watermark

logger = logging.getLogger(__name__)

SERIALIZER_VERSION = 2

# The §1m trigger paths, named rather than numbered at the call sites.
PATH_NEAR_TRIGGER = 1        # admitted, close to the ceiling; the turn already answered
PATH_OVER_BUDGET = 2         # StreamOverBudgetError after a COMPLETE fetch; the turn failed closed
PATH_PAGE_CEILING = 3        # page-ceiling HistoryFetchError, before any estimate exists
# §1b's `raw_rebuild_required` / `payload_corrupt` also trigger recompaction, and like path 3 they
# fire before any estimate exists — so they size the same way. An alias rather than a fourth path:
# the plan names three, and inventing a fourth would put a state in telemetry it never defined.
PATH_SELECTION_REFUSED = PATH_PAGE_CEILING

# The two selection results that are never handed to the builder: they retain identity for
# telemetry and for the publication CAS, and they trigger recompaction. They are not renderable.
REFUSED_RESULTS = ("raw_rebuild_required", "payload_corrupt")
# Results that carry a snapshot the turn renders from.
RENDERED_RESULTS = ("pinned", "pinned_stale")

DORMANT_BACKOFF_SECONDS = 3600.0

DRAIN_BATCH = 50
DRAIN_BACKOFF_START = 30.0
DRAIN_BACKOFF_CEILING = 600.0

# How many worker slices one background task will run before yielding the slot. A slice that
# exhausts its budget reschedules and is NOT a failure (§1n), so this bounds the task, not the
# attempt: the attempt survives across slices, restarts and boots until it publishes or is
# discarded.
MAX_SLICES_PER_TASK = 64


class ChannelSnapshotCoordinator:
    """One per process, constructed in application composition and injected.

    Pins are in-memory refcounts by design: a pin means "something in THIS process is still
    rendering from this snapshot", which is not a fact that survives a restart — after one,
    nothing is reading anything. A second instance would happily delete a snapshot the first has
    pinned, which is why main.py builds exactly one and treats a failure to build it as fatal.
    """

    def __init__(self, db, retain_generations: Optional[int] = None,
                 retain_days: Optional[int] = None, *, client: Any = None,
                 openai_client: Any = None, retry_delay: float = 5.0,
                 max_retries: int = 2):
        self.db = db
        self.client = client
        self.openai_client = openai_client
        self.retain_generations = (config.snapshot_retain_generations
                                   if retain_generations is None else retain_generations)
        self.retain_days = (config.snapshot_retain_days
                            if retain_days is None else retain_days)
        self.retry_delay = float(retry_delay)
        self.max_retries = int(max_retries)
        self._channel_locks: Dict[tuple, asyncio.Lock] = {}
        # Guards the refcount map AND is held across the sweep's delete, so a pin taken while
        # the sweep is mid-flight can never lose to it.
        self._pin_lock = asyncio.Lock()
        self._pins: Dict[str, int] = defaultdict(int)
        # Lineages marked retirement-pending (§1f phase 1). New selection refuses them from the
        # moment the mark is set, which is BEFORE the readers have drained.
        self._retiring: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

        # --- §1m coordinator state -------------------------------------------------------
        # Single-flight, keyed (team, channel, namespace).
        self._tasks: Dict[Tuple[str, str, str], asyncio.Task] = {}
        # False before start() and from the first moment of stop(). The completion hook checks it
        # before spawning, so a task vacating during shutdown cannot start work the drain passed.
        self._accepting = False
        self._closing = asyncio.Event()
        self._drain_task: Optional[asyncio.Task] = None
        # CRITICALs the plan bounds to once per boot.
        self._stalled: Set[Tuple[str, str, str]] = set()
        self._poison_logged: Set[int] = set()
        # The §1c pending-invalidation resolution point per channel: observations at or below it
        # have already been turned into invalidations by this process.
        self._resolved_frontier: Dict[Tuple[str, str], int] = {}
        # The obligated pair the last task for a channel actually ATTEMPTED. The completion hook
        # compares against it so the handoff fires for a SUCCESSOR and never for a rerun.
        self._attempted: Dict[Tuple[str, str, str], Tuple[str, int]] = {}
        # The §1m revalidation machine. Keyed (team, channel, namespace, serializer_version);
        # `owed` is DERIVED from the published row's `headroom_source`, so it survives a restart
        # in the only direction that is safe — a lost `complete` costs one re-measurement, while a
        # lost `owed` would lose the obligation entirely.
        self._revalidation: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        self._revalidation_lock = asyncio.Lock()

    # ------------------------------------------------------------------ locks and pins

    def channel_lock(self, team_id: str, channel_id: str) -> asyncio.Lock:
        """The per-channel lock. Retirement holds it across its three phases; publication takes
        it itself, briefly, and NOBODY holds it across a call into `publish()` — see §1m."""
        key = (team_id, channel_id)
        lock = self._channel_locks.get(key)
        if lock is None:
            lock = self._channel_locks[key] = asyncio.Lock()
        return lock

    async def pin(self, snapshot_id: str) -> None:
        """Hold a snapshot against the sweep for as long as something is reading it."""
        async with self._pin_lock:
            self._pins[snapshot_id] += 1

    async def unpin(self, snapshot_id: str) -> None:
        async with self._pin_lock:
            if snapshot_id in self._pins:
                self._pins[snapshot_id] -= 1
                if self._pins[snapshot_id] <= 0:
                    del self._pins[snapshot_id]

    @asynccontextmanager
    async def pinned(self, snapshot_id: str):
        """Pin for the duration of a block — the shape every reader should use."""
        await self.pin(snapshot_id)
        try:
            yield snapshot_id
        finally:
            await self.unpin(snapshot_id)

    def pinned_ids(self) -> List[str]:
        """Snapshot ids with a live reader right now."""
        return [sid for sid, count in self._pins.items() if count > 0]

    def refused_lineage(self, team_id: str, channel_id: str,
                        namespace: str = PROD_NAMESPACE) -> Tuple[str, ...]:
        """The retirement-pending set selection must refuse for this scope."""
        return tuple(sorted(self._retiring.get((team_id, channel_id, namespace), ())))

    # ------------------------------------------------------------------ selection (§1b)

    async def select_and_pin(self, team_id: str, channel_id: str,
                             serializer_version: int = SERIALIZER_VERSION,
                             max_boundary: Optional[str] = None, *,
                             namespace: str = PROD_NAMESPACE) -> Dict[str, Any]:
        """The §1b result dict, with the chosen snapshot pinned when there is one to pin.

        Selection and the pin happen under `_pin_lock` — the same lock the sweep deletes against
        — so a snapshot cannot be chosen and then deleted before its refcount lands. Two steps is
        the classic version of that bug: the turn reads a valid pointer, the sweep runs, and the
        turn renders from a row that no longer exists.

        ONLY a rendered result is pinned. `raw_rebuild_required` and `payload_corrupt` retain
        their identity for telemetry and the publication CAS but are never read, so pinning them
        would hold rows against the sweep that nothing will ever render.
        """
        async with self._pin_lock:
            result = await self.db.select_snapshot_for_pin_async(
                team_id, channel_id, namespace, int(serializer_version), max_boundary,
                refused_lineage=self.refused_lineage(team_id, channel_id, namespace))
            if result.get("result") in RENDERED_RESULTS and result.get("snapshot_id"):
                self._pins[result["snapshot_id"]] += 1
        return result

    async def get_active(self, team_id: str, channel_id: str,
                         serializer_version: int = SERIALIZER_VERSION
                         ) -> Optional[Dict[str, Any]]:
        """The published snapshot for this scope, or None at genesis."""
        return await self.db.get_active_snapshot_async(team_id, channel_id, serializer_version)

    async def invalidate(self, snapshot_id: str) -> bool:
        """Mark one generation invalid. Recompaction is triggered, never implicit."""
        return await self.db.invalidate_snapshot_async(snapshot_id)

    async def insert_candidate(self, team_id: str, channel_id: str, serializer_version: int,
                               boundary_ts: str, summary_text: str,
                               root_anchors: Optional[List[Dict[str, Any]]] = None) -> str:
        """Store an unpublished v1-shaped candidate and return its opaque id."""
        snapshot_id = uuid.uuid4().hex
        await self.db.insert_channel_snapshot_async(
            snapshot_id, team_id, channel_id, serializer_version, boundary_ts, summary_text,
            root_anchors)
        return snapshot_id

    # ------------------------------------------------------- pending invalidation (§1a step 3)

    async def resolve_pending_invalidation(self, team_id: str, channel_id: str, *,
                                           namespace: str = PROD_NAMESPACE) -> List[str]:
        """Turn every mutation observation this process has not yet acted on into invalidations.

        Runs AFTER the frontier drain and BEFORE selection (§1a): an observation that landed while
        the drain was waiting must reach the pointer before the turn reads it, or the turn selects
        a generation a durable decision has already condemned.

        Never only the active generation — `affected_snapshot_ids_async` returns every generation
        whose summarized span covers the subject OR whose rendered anchor roots include it, because
        falling back to an ancestor that summarized the same source would silently restore the lie.
        """
        key = (team_id, channel_id)
        frontier = self._resolved_frontier.get(key, 0)
        observations = await self.db.mutation_observations_after_async(
            team_id, channel_id, frontier)
        if not observations:
            return []
        invalidated: List[str] = []
        seen: Set[str] = set()
        high = frontier
        for observation in observations:
            high = max(high, int(observation.get("id") or 0))
            subject_ts = str(observation.get("subject_ts") or "")
            if not subject_ts or subject_ts in seen:
                continue
            seen.add(subject_ts)
            for snapshot_id in await self.db.affected_snapshot_ids_async(
                    team_id, channel_id, namespace, subject_ts):
                if await self.db.invalidate_snapshot_async(snapshot_id):
                    invalidated.append(snapshot_id)
                    self._emit(op="invalidate", channel_id=channel_id, snapshot_id=snapshot_id,
                               reason=str(observation.get("kind") or "mutation"))
        self._resolved_frontier[key] = high
        if invalidated:
            logger.info(
                f"{channel_id}: invalidated {len(invalidated)} generation(s) on pending mutation "
                f"observations up to id {high}")
        return invalidated

    # ------------------------------------------------------------------ publication (§1d)

    async def publish(self, *, team_id: str, channel_id: str, serializer_version: int,
                      snapshot_id: str, expected_previous_id: Optional[str],
                      source_floor_ts: str, boundary_ts: str, mutation_frontier: int,
                      namespace: str = PROD_NAMESPACE, current_profile: Optional[str] = None,
                      status: str = "published",
                      outbox_rows: Sequence[Dict[str, Any]] = (),
                      satisfy: Optional[Dict[str, Any]] = None,
                      dormancy: Optional[Dict[str, Any]] = None,
                      crawl_id: Optional[str] = None,
                      lock_held: bool = False) -> Dict[str, Any]:
        """The whole publication CAS, with the channel lock scoped strictly inside (§1m).

        `channel_lock()` is NOT reentrant. A caller that already holds it — retirement, which holds
        it across all three of its phases — passes `lock_held=True` rather than deadlocking on its
        own lock. Every other caller passes nothing and the lock is taken here, briefly.

        `current_profile` defaults to the channel's CURRENT EFFECTIVE profile, resolved at the
        moment of publication: the candidate's `sizing_profile` must still equal it, checked inside
        the transaction, because a profile can change after the last chunk boundary has passed and
        no boundary is left to cancel at.
        """
        if current_profile is None:
            current_profile = (await self.sizing_for(team_id, channel_id))["sizing_profile"]

        async def _publish() -> Dict[str, Any]:
            return await self.db.publish_compaction_candidate_async(
                team_id=team_id, channel_id=channel_id, namespace=namespace,
                serializer_version=int(serializer_version), snapshot_id=snapshot_id,
                expected_previous_id=expected_previous_id, source_floor_ts=source_floor_ts,
                boundary_ts=boundary_ts, mutation_frontier=int(mutation_frontier),
                current_profile=current_profile, status=status, outbox_rows=outbox_rows,
                satisfy=satisfy, dormancy=dormancy, crawl_id=crawl_id)

        if lock_held:
            result = await _publish()
        else:
            async with self.channel_lock(team_id, channel_id):
                result = await _publish()
        if result.get("won"):
            self._stalled.discard((team_id, channel_id, namespace))
        return result

    # ------------------------------------------------------------------ retirement (§1f)

    async def retire_lineage(self, *, team_id: str, channel_id: str,
                             lineage_ids: Sequence[str],
                             expected_active_id: Optional[str],
                             expected_generation: Optional[int] = None,
                             namespace: str = PROD_NAMESPACE,
                             serializer_version: int = SERIALIZER_VERSION,
                             drain_timeout: float = 30.0) -> Dict[str, Any]:
        """THREE PHASES, and the TWO LOCKS do different jobs.

        `unpin()` acquires the PIN lock, so waiting for refcounts to fall while holding THAT lock
        can never succeed — the readers we are waiting for cannot get in to release. But
        publication is serialized by the CHANNEL lock, not the pin lock, so a retirement guarding
        pins alone could have the active pointer moved underneath it mid-drain.

        So the CHANNEL LOCK is held across all three phases, and the PIN LOCK is taken only
        briefly, to set the mark:

        1. mark retirement-pending (pin lock, released immediately) — new selection refuses the
           lineage from this moment;
        2. let existing readers drain, pin lock FREE so `unpin()` proceeds, channel lock still
           held so no publication moves the pointer;
        3. the guarded transaction.

        CANCELLATION CLEANUP IS AN OUTER `finally`: whenever the deletion transaction did NOT
        commit, the mark comes off. Without it a cancelled retirement leaves a snapshot that
        exists, is valid, and that nothing in this process will ever read again.
        """
        ids = [str(s) for s in lineage_ids if s]
        if not ids:
            return {"ok": False, "restored": None, "reason": "empty_lineage"}
        scope = (team_id, channel_id, namespace)
        committed = False
        async with self.channel_lock(team_id, channel_id):
            try:
                async with self._pin_lock:              # phase 1
                    self._retiring[scope].update(ids)
                drained = await self._await_drain(ids, timeout=drain_timeout)   # phase 2
                if not drained:
                    logger.warning(
                        f"{channel_id}: retirement of {len(ids)} generation(s) abandoned — a "
                        f"reader did not release within {drain_timeout:g}s")
                    return {"ok": False, "restored": None, "reason": "pinned"}
                result = await self.db.retire_snapshot_lineage_async(   # phase 3
                    team_id=team_id, channel_id=channel_id, namespace=namespace,
                    serializer_version=int(serializer_version), lineage_ids=ids,
                    expected_active_id=expected_active_id,
                    expected_generation=expected_generation)
                committed = bool(result.get("ok"))
                return result
            finally:
                if not committed:
                    async with self._pin_lock:
                        self._retiring[scope].difference_update(ids)
                else:
                    async with self._pin_lock:
                        self._retiring[scope].difference_update(ids)
                        if not self._retiring[scope]:
                            self._retiring.pop(scope, None)

    async def _await_drain(self, ids: Sequence[str], *, timeout: float,
                           poll: float = 0.02) -> bool:
        """Wait for every id's refcount to reach zero, WITHOUT holding the pin lock."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            async with self._pin_lock:
                live = [sid for sid in ids if self._pins.get(sid, 0) > 0]
            if not live:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll)

    async def rollback_generation(self, *, team_id: str, channel_id: str,
                                  expected_snapshot_id: str,
                                  namespace: str = PROD_NAMESPACE,
                                  serializer_version: int = SERIALIZER_VERSION,
                                  drain_timeout: float = 30.0) -> Dict[str, Any]:
        """R0-5: retire ONE rejected published generation and restore what came before it.

        The same three-phase protocol, because it is the same hazard: a live reader mid-render and
        a concurrent publication are both able to make a single-statement rollback wrong. Leaving
        the generation merely `invalidated` would not restore pre-crawl behaviour either — §1b
        classifies an invalidated generation with no valid ancestor as `raw_rebuild_required`,
        which restarts the very compaction the owner rejected.

        NO PRODUCTION MESSAGE IS EDITED OR DELETED by a rollback, and nothing here posts.
        """
        scope = (team_id, channel_id, namespace)
        committed = False
        async with self.channel_lock(team_id, channel_id):
            try:
                async with self._pin_lock:
                    self._retiring[scope].add(expected_snapshot_id)
                if not await self._await_drain([expected_snapshot_id], timeout=drain_timeout):
                    return {"ok": False, "restored": None, "reason": "pinned"}
                result = await self.db.rollback_published_generation_async(
                    team_id=team_id, channel_id=channel_id, namespace=namespace,
                    serializer_version=int(serializer_version),
                    expected_snapshot_id=expected_snapshot_id)
                committed = bool(result.get("ok"))
                return result
            finally:
                async with self._pin_lock:
                    self._retiring[scope].discard(expected_snapshot_id)
                    if not self._retiring[scope]:
                        self._retiring.pop(scope, None)
                if committed:
                    logger.warning(
                        f"{channel_id}: rolled back published generation {expected_snapshot_id}")

    # ------------------------------------------------------------------ sizing / profile

    async def sizing_for(self, team_id: str, channel_id: str) -> Dict[str, Any]:
        """The channel's CURRENT EFFECTIVE sizing profile (§1m).

        Four parts — model, window, trigger, target — because a single integer cannot be compared
        across a model, window OR threshold change, and a publication proved to fit under a looser
        target says nothing about fit under the stricter one that raised the obligation.
        """
        from message_processor.channel_compaction import resolve_sizing
        model = await self._effective_model(channel_id)
        return resolve_sizing(model=model, window=config.get_model_token_limit(model))

    async def _effective_model(self, channel_id: str) -> str:
        from message_processor.utilities import effective_request_model
        profile: Dict[str, Any] = {}
        getter = getattr(self.db, "get_channel_settings_async", None)
        if callable(getter):
            try:
                row = await getter(channel_id) or {}
            except Exception as e:  # noqa: BLE001 — a settings read never blocks compaction
                logger.debug(f"{channel_id}: channel settings unread for sizing: {e}")
                row = {}
            # A NULL column means INHERIT, and a key present with None would defeat the
            # resolver's own defaulting.
            profile = {key: row.get(key) for key in ("model", "enable_web_search")
                       if row.get(key) is not None}
        return effective_request_model(profile) or config.gpt_model

    # ------------------------------------------------------------------ triggers (§1m 1-3)

    def trigger(self, *, team_id: str, channel_id: str, namespace: str = PROD_NAMESPACE,
                path: int, **evidence: Any) -> None:
        """THE one entry point for the three trigger paths. Background, and it returns at once.

        Single-flight is claimed SYNCHRONOUSLY here: registering the task before the first await
        is what makes two triggers arriving in the same tick produce one task rather than two that
        both discover an empty map. Everything that decides whether work actually happens —
        obsolescence reconciliation, the dormancy arbitration, the revival CAS — runs inside that
        one task.
        """
        if not self._accepting:
            return
        key = (team_id, channel_id, namespace)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.ensure_future(self._run(key, path=int(path), evidence=dict(evidence)))
        self._tasks[key] = task
        task.add_done_callback(functools.partial(self._done_callback, key))

    def _done_callback(self, key: Tuple[str, str, str], task: "asyncio.Task") -> None:
        self._on_task_done(key, task)

    def _on_task_done(self, key: Tuple[str, str, str], task: "asyncio.Task") -> None:
        """The task vacates its single-flight slot HERE, which is the only moment guaranteed to
        happen — so this is where the successor handoff lives (§1m, test 95 f2)."""
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.warning(f"{key[1]}: compaction task ended with {error!r}")
        # COORDINATOR CLOSING ⇒ spawn nothing. Checked first, so a task vacating during shutdown
        # cannot start fresh work the drain has already passed.
        if not self._accepting:
            return
        asyncio.ensure_future(self._completion_hook(key))

    async def _completion_hook(self, key: Tuple[str, str, str]) -> None:
        """Re-run ROW ARBITRATION when a slot is vacated. It NEVER STARTS BLIND.

        "The newer obligation stays scheduled" is not enough on its own: the stale task occupied
        the slot, so the successor could not start while it ran, and waiting for the next trigger
        could leave a ready obligation idle indefinitely on a quiet channel.

        - ACTIVE current-profile obligation ⇒ start it immediately.
        - DORMANT ⇒ start NOTHING. Its deadline is honoured exactly as §1m requires; a completion
          hook that started it would be a backoff bypass with a different name.
        - ABSENT or SATISFIED ⇒ nothing.

        AND ONLY FOR A GENUINELY NEWER OBLIGATION. The hook exists for the SUCCESSOR case — a
        newer obligated pair enqueued while the stale attempt was still running — so it starts
        only when the row names a pair the finished task did not attempt. Restarting the SAME pair
        would be a spin: the attempt that just failed to satisfy it would be run again with
        nothing changed, forever, and the backoff that is supposed to bound that lives in the
        publish-nothing terminal rather than here.
        """
        if not self._accepting or (key in self._tasks and not self._tasks[key].done()):
            return
        team_id, channel_id, namespace = key
        try:
            row = await self.db.load_pending_recompaction_async(team_id, channel_id, namespace)
            if not row or row.get("state") != "active":
                return
            profile = (await self.sizing_for(team_id, channel_id))["sizing_profile"]
            if profile not in (row.get("requirements") or {}):
                return
            pair = (str(row.get("obligated_snapshot_id") or ""),
                    int(row.get("obligated_generation") or 0))
            if self._attempted.get(key) == pair:
                return
            self.trigger(team_id=team_id, channel_id=channel_id, namespace=namespace,
                         path=PATH_NEAR_TRIGGER, successor=True,
                         obligation=dict(row))
        except Exception as e:  # noqa: BLE001 — a handoff failure is never a turn's problem
            logger.warning(f"{channel_id}: compaction completion hook failed: {e}")

    async def arbitrate(self, key: Tuple[str, str, str], *,
                        profile: str) -> Dict[str, Any]:
        """Should this trigger produce work at all? (§1m, tests 95a/95b/95g/95h.)

        EVERY trigger path arrives here BEFORE any compaction is created. Before the deadline the
        answer is ENQUEUE NOTHING AT ALL — not the obligation's recompaction, and not a separate
        ordinary or background compaction either. At or after it, ONE CAS revives EXACTLY ONE
        task and the loser does nothing.
        """
        team_id, channel_id, namespace = key
        row = await self.db.load_pending_recompaction_async(team_id, channel_id, namespace)
        if row is None:
            return {"run": True, "reason": "no_obligation", "row": None}

        # RECONCILIATION POINT 2 — task acquisition, checked as the task is claimed so a task
        # cannot start against a profile that changed while it sat in the map.
        retired = await self.reconcile_profile(team_id, channel_id, namespace=namespace,
                                               current_profile=profile)
        if retired:
            row = await self.db.load_pending_recompaction_async(team_id, channel_id, namespace)
            if row is None:
                return {"run": True, "reason": "obsolete_cleared", "row": None}

        if row.get("state") == "dormant":
            if row.get("malformed"):
                # FAIL CLOSED. Reading a malformed row as active would let it bypass the backoff,
                # which is the one outcome this machinery exists to prevent.
                return {"run": False, "reason": "malformed", "row": row}
            deadline = row.get("next_attempt_after")
            if not deadline or datetime.now().isoformat() < str(deadline):
                return {"run": False, "reason": "dormant", "row": row}
            won = await self.db.cas_pending_recompaction_async(
                team_id=team_id, channel_id=channel_id, namespace=namespace,
                expect_state="dormant", new_state="active", expect_profile_key=profile,
                deadline_passed=True)
            if not won:
                return {"run": False, "reason": "lost_revival", "row": row}
            row = await self.db.load_pending_recompaction_async(team_id, channel_id, namespace)
            return {"run": True, "reason": "revived", "row": row}
        return {"run": True, "reason": "active", "row": row}

    async def reconcile_profile(self, team_id: str, channel_id: str, *,
                                namespace: str = PROD_NAMESPACE,
                                current_profile: Optional[str] = None) -> List[str]:
        """RETIRE requirement-map entries keyed to a profile that no longer exists.

        Called at all THREE reconciliation points (§1m): boot hydration, task acquisition, and —
        the one a hydration-only implementation misses — a LIVE CHANNEL SETTINGS CHANGE through
        the modal write path, which changes the channel model at runtime.

        A RUNNING old-profile task is cancelled at its next CHUNK BOUNDARY, and the CANCELLATION
        INTENT is written BEFORE the requirement is removed, in the same transaction: otherwise a
        crash in between leaves boot with an orphan checkpoint and no obligation to explain it.
        """
        if current_profile is None:
            current_profile = (await self.sizing_for(team_id, channel_id))["sizing_profile"]
        row = await self.db.load_pending_recompaction_async(team_id, channel_id, namespace)
        obsolete = sorted(k for k in ((row or {}).get("requirements") or {})
                          if k != current_profile)
        if not obsolete:
            return []
        checkpoint = await self.db.load_crawl_checkpoint_async(team_id, channel_id, namespace)
        if checkpoint and str(checkpoint.get("sizing_profile") or "") != current_profile:
            await self.db.write_cancellation_intent_async(
                {"team_id": team_id, "channel_id": channel_id, "namespace": namespace,
                 "crawl_id": str(checkpoint["crawl_id"]),
                 "obligated_snapshot_id": str(row.get("obligated_snapshot_id") or ""),
                 "reason": "obsolete_profile",
                 "created_ts": datetime.now().isoformat()},
                retire_keys=obsolete)
            self._cancel_running(team_id, channel_id, namespace)
            return obsolete
        return await self.db.reconcile_pending_profiles_async(
            team_id=team_id, channel_id=channel_id, namespace=namespace,
            current_profile=current_profile)

    def _cancel_running(self, team_id: str, channel_id: str, namespace: str) -> None:
        """Signal a running old-profile task. The task observes this at its next CHUNK BOUNDARY —
        chunks stay atomic, since a half-chunk is exactly what that rule exists to prevent."""
        task = self._tasks.get((team_id, channel_id, namespace))
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------ the background run

    async def _run(self, key: Tuple[str, str, str], *, path: int,
                   evidence: Dict[str, Any]) -> Dict[str, Any]:
        team_id, channel_id, namespace = key
        try:
            sizing = await self.sizing_for(team_id, channel_id)
            decision = await self.arbitrate(key, profile=sizing["sizing_profile"])
            row = decision.get("row") or {}
            self._attempted[key] = (str(row.get("obligated_snapshot_id") or ""),
                                    int(row.get("obligated_generation") or 0))
            if not decision["run"]:
                logger.debug(f"{channel_id}: compaction trigger {path} enqueued nothing "
                             f"({decision['reason']})")
                return decision
            return await self._compact(key, path=path, evidence=evidence, sizing=sizing,
                                       obligation=decision.get("row"))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — reported; the channel keeps failing closed
            logger.warning(f"{channel_id}: background compaction failed: {e}", exc_info=True)
            return {"outcome": "failed", "reason": str(e)}

    async def _compact(self, key: Tuple[str, str, str], *, path: int,
                       evidence: Dict[str, Any], sizing: Dict[str, Any],
                       obligation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """One obligation, run to a terminal state: published, discarded, or publish-nothing.

        The CHANNEL LOCK IS NOT HELD HERE. `publish()` takes it itself and holds it only for the
        CAS; holding it across the whole crawl would block retirement for the length of a
        multi-minute background job and, worse, deadlock against the publish inside it.
        """
        from message_processor import channel_compaction as compaction
        team_id, channel_id, namespace = key
        headroom_source, headroom_tokens = self._headroom(path, evidence)
        attempts = 0
        outcome: Dict[str, Any] = {"outcome": "failed", "reason": "not_started"}

        while attempts <= self.max_retries and not self._closing.is_set():
            attempts += 1
            h = self._fresh_h(channel_id, evidence)
            if h is None:
                return {"outcome": "failed", "reason": "no_watermark"}
            selection = await self.db.select_snapshot_for_pin_async(
                team_id, channel_id, namespace, SERIALIZER_VERSION, h,
                refused_lineage=self.refused_lineage(team_id, channel_id, namespace))
            parent = (selection.get("snapshot")
                      if selection.get("result") in RENDERED_RESULTS else None)
            expected_previous = selection.get("snapshot_id") if parent else None
            checkpoint = await self.db.load_crawl_checkpoint_async(team_id, channel_id, namespace)

            common = dict(db=self.db, client=self.client, team_id=team_id,
                          channel_id=channel_id, namespace=namespace, coordinator=self,
                          openai_client=self.openai_client, shutdown=self._closing)
            if checkpoint is None:
                checkpoint = await self._new_checkpoint(
                    key, h=h, sizing=sizing, parent=parent,
                    headroom_source=headroom_source, headroom_tokens=headroom_tokens)
                if checkpoint is None:
                    return {"outcome": "failed", "reason": "no_coverage"}
            outcome = await self._drive_slices(
                compaction, common, trigger=str(path), sizing=sizing,
                headroom_source=headroom_source, headroom_tokens=headroom_tokens,
                expected_previous=expected_previous)

            await self._commit_outbox(outcome)
            result = outcome.get("outcome")
            if result == "published":
                await self._after_publication(key, outcome)
                return outcome
            if result in ("deferred", "publish_nothing"):
                break
            # `discarded` / `failed` — a bounded retry, then the terminal below.
            if attempts <= self.max_retries and self.retry_delay:
                await asyncio.sleep(self.retry_delay)

        if outcome.get("outcome") == "deferred":
            return outcome
        await self._terminal_publish_nothing(key, outcome=outcome, sizing=sizing,
                                             obligation=obligation)
        return outcome

    async def _drive_slices(self, compaction, common: Dict[str, Any], *, trigger: str,
                            sizing: Dict[str, Any], headroom_source: str,
                            headroom_tokens: int,
                            expected_previous: Optional[str]) -> Dict[str, Any]:
        """Run worker slices until the crawl reaches a terminal state or the coordinator closes.

        The budget bounds ONE SLICE, not the attempt (§1n): a generation attempt spans every
        resumption until it publishes, fails or is discarded, so a slice that runs out of pages
        reschedules — which is not a failure and does not end the attempt.
        """
        outcome: Dict[str, Any] = {"outcome": "deferred", "reason": "not_started"}
        live = await self._live_identity(common["team_id"], common["channel_id"],
                                         common["namespace"], sizing=sizing)
        for _slice in range(MAX_SLICES_PER_TASK):
            if self._closing.is_set():
                return outcome
            outcome = await compaction.run_crawl_slice(
                trigger=trigger, headroom_source=headroom_source,
                headroom_tokens=headroom_tokens, sizing=sizing, live=live,
                expected_previous_id=expected_previous,
                budget=compaction.SliceBudget(),
                serializer_version=SERIALIZER_VERSION, **common)
            if outcome.get("outcome") != "deferred":
                return outcome
            if outcome.get("reason") == "next_attempt_after":
                return outcome      # an active backoff, not an exhausted slice: stop, don't spin
            await self._commit_outbox(outcome)
        return outcome

    async def _live_identity(self, team_id: str, channel_id: str, namespace: str, *,
                             sizing: Dict[str, Any]) -> Dict[str, Any]:
        """The live values the §1n step-2 reset list is compared against.

        Passed on EVERY slice: without it `run_crawl_slice` skips the resume ladder entirely, and
        a crawl would neither notice a config change nor discard on an in-span mutation.
        """
        from message_processor.channel_compaction import PROMPT_VERSION, profile_version
        checkpoint = await self.db.load_crawl_checkpoint_async(team_id, channel_id, namespace)
        source = str((checkpoint or {}).get("headroom_source") or "")
        tokens = int((checkpoint or {}).get("headroom_tokens") or 0)
        return {
            "serializer_version": SERIALIZER_VERSION,
            "serializer_config_hash": self._config_hash(),
            "prompt_version": PROMPT_VERSION,
            "sizing_profile": sizing["sizing_profile"],
            # Headroom is a property of the TRIGGER PATH that opened the crawl, not of live
            # config, so it is compared against the checkpoint's own value — otherwise every
            # measured crawl would reset itself against the fixed default on its next slice.
            "profile_version": profile_version(headroom_source=source, headroom_tokens=tokens),
        }

    @staticmethod
    def _config_hash() -> str:
        from message_processor.channel_stream import _stable_hash, serializer_config_snapshot
        return _stable_hash(serializer_config_snapshot())

    async def _new_checkpoint(self, key: Tuple[str, str, str], *, h: str,
                              sizing: Dict[str, Any], parent: Optional[Dict[str, Any]],
                              headroom_source: str,
                              headroom_tokens: int) -> Optional[Dict[str, Any]]:
        """A fresh crawl: INCREMENTAL from a live parent, RAW otherwise.

        The input span is NOT the lineage span (§1n). An incremental crawl fetches only
        `(parent_boundary_ts, pinned_H]` while the snapshot it produces still claims the PARENT's
        `source_floor_ts` — its lineage covers everything the parent covered. Conflating them makes
        an incremental crawl either double-summarize old history or behave as a raw rebuild.
        """
        from message_processor import channel_compaction as compaction
        team_id, channel_id, namespace = key
        if parent is not None:
            source_floor = str(parent.get("source_floor_ts"))
            input_floor, inclusive = str(parent.get("boundary_ts")), False
            mode, parent_id = "incremental", parent.get("snapshot_id")
        else:
            coverage = await self.db.get_channel_coverage_async(team_id, channel_id)
            source_floor = str((coverage or {}).get("coverage_start_ts") or "")
            if not source_floor:
                logger.info(
                    f"{channel_id}: no coverage row yet; a crawl has no floor to start from")
                return None
            input_floor, inclusive = source_floor, True
            mode, parent_id = "raw", None
        frontier = await self.db.max_mutation_observation_id_async(team_id, channel_id)
        renders, receipts = await self._freeze_source_pin(team_id, channel_id, h)
        checkpoint = compaction.new_checkpoint(
            team_id=team_id, channel_id=channel_id, namespace=namespace, pinned_H=h,
            mutation_frontier=int(frontier), source_floor_ts=source_floor,
            input_floor_ts=input_floor, input_floor_inclusive=inclusive, crawl_mode=mode,
            serializer_version=SERIALIZER_VERSION,
            serializer_config_hash=self._config_hash(), sizing=sizing,
            headroom_source=headroom_source, headroom_tokens=headroom_tokens,
            parent_snapshot_id=parent_id, frozen_renders=renders, frozen_receipts=receipts)
        await self.db.upsert_crawl_checkpoint_async(dict(checkpoint))
        return checkpoint

    async def _freeze_source_pin(self, team_id: str, channel_id: str,
                                 h: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """FROZEN AT CRAWL START (§1n): the artifact RENDER BYTES and the receipt PROOF STATES.

        Both are frozen because completed chunks already depend on them and a restart must not mix
        two points in time — §1i forbids a live manifest query outright. The renders are the exact
        BYTES, not merely the row fields: Appendix B embeds those bytes in the projection, so a
        restart holding only row ids and hashes could not reproduce the projection it already
        hashed.

        The bytes come from `channel_stream.artifact_render_bytes`, the SINGLE producer, and their
        SHA-256 becomes the manifest's `content_hash`. Late-artifact evidence re-renders through
        the same function and compares that hash to decide whether a row changed since capture, so
        suppression is BY BYTE IDENTITY — a second producer drifting by one character would make
        every already-summarized artifact look changed and render it again as "late" evidence the
        summary already contains.
        """
        import hashlib

        from message_processor.channel_compaction import freeze_receipts
        from message_processor.channel_stream import artifact_render_bytes
        sidecars = await self.db.read_channel_sidecars_async(team_id, channel_id, h)
        renders: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for namespace, key, ts_key in (
                ("image_analysis", "image_analyses", "message_ts"),
                ("document_extraction", "document_extractions", "message_ts"),
                ("ambient_artifact", "ambient_artifacts", "source_ts")):
            for row in sidecars.get(key) or ():
                body = artifact_render_bytes(namespace, row)
                if body is None:
                    continue          # not evidence worth an item at all
                source_ts = str(row.get(ts_key) or "")
                renders[source_ts].append({
                    "artifact_namespace": namespace,
                    "row_id": str(row.get("id") or row.get("row_id") or source_ts),
                    "source_ts": source_ts,
                    "render": body,
                    "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    "captured_render_version": "1",
                    "status_at_capture": str(row.get("status") or "complete"),
                })
        receipts = freeze_receipts(
            sidecars.get("receipts") or (),
            epoch_ts=sidecars.get("receipt_feature_epoch_ts"))
        return {ts: rows for ts, rows in renders.items()}, receipts

    def _headroom(self, path: int, evidence: Dict[str, Any]) -> Tuple[str, int]:
        """MEASURED where possible, fixed only where necessary (§1n).

        Path 2 has real admission components from its triggering turn, so it owes no revalidation.
        Path 3 fires before any estimate exists, so it records the fixed heuristic — which is made
        safe by a RECORDED OBLIGATION rather than by hope.
        """
        measured = evidence.get("headroom_tokens")
        if path in (PATH_NEAR_TRIGGER, PATH_OVER_BUDGET) and measured is not None:
            return "measured", int(measured)
        return "fixed", int(config.crawl_fixed_headroom_tokens)

    def _fresh_h(self, channel_id: str, evidence: Dict[str, Any]) -> Optional[str]:
        """EXECUTION PINS A FRESH H (§1m). The obligation records WHY a recompaction is owed, not
        a stale ceiling; running against the H that existed when it was enqueued would summarize
        to a boundary the channel has long since moved past."""
        try:
            return admission_watermark.pin(channel_id, evidence.get("h")).h
        except Exception as e:  # noqa: BLE001 — a channel with no admitted ts has no window
            logger.debug(f"{channel_id}: no H for a background compaction: {e}")
            return None

    async def _commit_outbox(self, outcome: Dict[str, Any]) -> None:
        """Rows the builder HANDED BACK, committed here. The builder never writes telemetry."""
        rows = outcome.get("outbox_rows") or ()
        if not rows:
            return
        try:
            await self.db.insert_outbox_rows_async(list(rows))
            outcome["outbox_rows"] = []
        except Exception as e:  # noqa: BLE001 — a poisoned body must not lose the state change
            logger.error(f"Compaction telemetry row rejected: {e}")

    async def _after_publication(self, key: Tuple[str, str, str],
                                 outcome: Dict[str, Any]) -> None:
        team_id, channel_id, namespace = key
        self._stalled.discard(key)
        try:
            await self.sweep()
        except Exception as e:  # noqa: BLE001 — retention is not correctness
            logger.debug(f"{channel_id}: post-publication sweep skipped: {e}")
        self._emit(op="publish", channel_id=channel_id,
                   snapshot_id=outcome.get("snapshot_id"),
                   generation=outcome.get("generation"),
                   boundary_ts=outcome.get("boundary_ts"),
                   serializer_version=SERIALIZER_VERSION)
        from message_processor import dev_barriers
        await dev_barriers.pre_resume_after_compaction(
            channel_id=channel_id, compaction_id=outcome.get("crawl_id") or
            outcome.get("snapshot_id"), snapshot_id=outcome.get("snapshot_id"),
            generation=outcome.get("generation"))

    async def _terminal_publish_nothing(self, key: Tuple[str, str, str], *,
                                        outcome: Dict[str, Any], sizing: Dict[str, Any],
                                        obligation: Optional[Dict[str, Any]]) -> None:
        """PUBLISH-NOTHING GETS ITS OWN DURABLE TERMINAL TRANSACTION, AND IT TERMINATES THE CRAWL.

        An attempt can exhaust its retries without publishing anything at all, and then there is no
        publication transaction to host the `active → dormant` move. Without this, the single case
        where the channel is MOST stuck is the one that stays active and is retried at every boot.

        The crawl state must die with the attempt, because a revived attempt starts a NEW crawl
        with a FRESH `H`: a surviving checkpoint still pins the OLD one, and the §1n resume ladder
        does not treat a changed H as a reason to discard.
        """
        team_id, channel_id, namespace = key
        checkpoint = await self.db.load_crawl_checkpoint_async(team_id, channel_id, namespace)
        crawl_id = str((checkpoint or {}).get("crawl_id") or outcome.get("crawl_id") or "")
        profile = sizing["sizing_profile"]
        deadline = (datetime.now() + timedelta(seconds=DORMANT_BACKOFF_SECONDS)).isoformat()
        rows = list(outcome.get("outbox_rows") or ())
        pair = None
        if obligation:
            pair = (str(obligation.get("obligated_snapshot_id") or ""),
                    int(obligation.get("obligated_generation") or 0))

        terminal = getattr(self.db, "terminal_publish_nothing_async", None)
        if callable(terminal):
            result = await terminal(
                team_id=team_id, channel_id=channel_id, namespace=namespace, crawl_id=crawl_id,
                expect_state="active", expect_pair=pair, expect_profile_key=profile,
                dormant_profile_key=profile, next_attempt_after=deadline,
                outbox_rows=rows, candidate_id=outcome.get("snapshot_id"))
            outcome["outbox_rows"] = []
            mismatch = result.get("mismatch")
            if mismatch == "profile":
                # The task really is sized for a configuration that no longer exists.
                await self.reconcile_profile(team_id, channel_id, namespace=namespace,
                                             current_profile=profile)
                return
            if mismatch:
                # SAME-PROFILE supersession: discard ONLY the stale attempt. The newer obligation
                # is left untouched and stays scheduled — treating this as obsolescence would
                # retire a live, current-profile obligation because an older attempt finished late.
                logger.info(f"{channel_id}: a stale compaction attempt was discarded; the newer "
                            f"obligation stays scheduled")
                return
        else:
            logger.critical(
                f"CRITICAL: {channel_id} publish-nothing terminal ran WITHOUT its atomic accessor "
                f"(database.terminal_publish_nothing_async is missing); a crash between these "
                f"writes leaves an active obligation with its backoff unwritten")
            await self._commit_outbox(outcome)
            if obligation:
                await self.db.cas_pending_recompaction_async(
                    team_id=team_id, channel_id=channel_id, namespace=namespace,
                    expect_state="active", new_state="dormant", expect_profile_key=profile,
                    expect_pair=pair, dormant_profile_key=profile,
                    next_attempt_after=deadline)
            if crawl_id:
                await self.db.delete_crawl_state_async(team_id, channel_id, namespace, crawl_id)

        if key not in self._stalled:
            self._stalled.add(key)
            logger.critical(
                f"CRITICAL: {channel_id} is COMPACTION-STALLED — the minimum retained tail leaves "
                f"no candidate that gets the complete request under trigger "
                f"({outcome.get('reason') or 'no_fit'}). Turns there keep failing closed; nothing "
                f"further is attempted until a trigger arrives at or after {deadline}")

    # ------------------------------------------------------ revalidation machine (§1m)

    async def revalidation_state(self, team_id: str, channel_id: str, *,
                                 namespace: str = PROD_NAMESPACE,
                                 serializer_version: int = SERIALIZER_VERSION,
                                 snapshot: Optional[Dict[str, Any]] = None) -> str:
        """`owed` | `claimed` | `complete` for the snapshot a turn just pinned.

        `owed` is DERIVED from the published row's own `headroom_source`, which is durable, rather
        than from a flag that would have to be written at publication and could be lost. A
        publication made under `headroom_source='fixed'` owes a fit revalidation against real
        admission components; one made under `measured` never does.
        """
        if not snapshot or str(snapshot.get("headroom_source") or "") != "fixed":
            return "complete"
        key = (team_id, channel_id, namespace, int(serializer_version))
        pair = (str(snapshot.get("snapshot_id")), int(snapshot.get("generation") or 0))
        entry = self._revalidation.get(key)
        if entry is None or entry.get("pair") != pair:
            return "owed"
        if entry["state"] == "claimed":
            # DEAD-CLAIM RECOVERY: a claimant that crashed mid-measurement does not strand the
            # obligation forever.
            if time.monotonic() - entry["claimed_at"] > float(config.revalidation_claim_ttl):
                return "owed"
        return entry["state"]

    async def revalidate(self, team_id: str, channel_id: str, *, snapshot: Dict[str, Any],
                         evidence: Any, namespace: str = PROD_NAMESPACE,
                         serializer_version: int = SERIALIZER_VERSION) -> Dict[str, Any]:
        """The `owed → claimed → complete` move for ONE turn (§1m, tests 55/72/79).

        EVERY CAS predicate names the obligated `(snapshot_id, generation)` pair. Without that
        binding a claimant measuring snapshot A could race the publication of a new fixed-headroom
        snapshot B and then mark B's obligation complete, discharging a revalidation that never
        happened.

        MEASUREMENT HAPPENS UNDER THE CLAIM, and an over-target result — INCLUDING the band
        between target and trigger, where the ordinary paths would enqueue nothing and where a
        mis-sized fixed profile hides — ENQUEUES DURABLY FIRST and only then moves to `complete`.
        Clearing first would lose the revalidation permanently if the process died in between.

        LOSERS DO NOTHING: they neither re-verify nor enqueue, so a burst of concurrent turns
        cannot produce a burst of compactions.
        """
        key = (team_id, channel_id, namespace, int(serializer_version))
        pair = (str(snapshot.get("snapshot_id")), int(snapshot.get("generation") or 0))
        if str(snapshot.get("headroom_source") or "") != "fixed":
            return {"claimed": False, "reason": "not_owed"}

        async with self._revalidation_lock:
            entry = self._revalidation.get(key)
            stale_claim = (entry is not None and entry.get("pair") == pair
                           and entry["state"] == "claimed"
                           and time.monotonic() - entry["claimed_at"]
                           > float(config.revalidation_claim_ttl))
            if entry is not None and entry.get("pair") == pair and not stale_claim:
                return {"claimed": False, "reason": entry["state"]}
            self._revalidation[key] = {"pair": pair, "state": "claimed",
                                       "claimed_at": time.monotonic()}

        enqueued = False
        try:
            fits = bool(evidence is not None and getattr(evidence, "admitted", False)
                        and evidence.total_tokens <= evidence.target_tokens)
            if not fits and evidence is not None:
                await self.db.merge_pending_recompaction_async(
                    team_id=team_id, channel_id=channel_id, namespace=namespace,
                    profile_key=evidence.sizing_profile,
                    required_headroom=int(evidence.fixed_tokens),
                    obligated_snapshot_id=pair[0], obligated_generation=pair[1],
                    reason="fixed_headroom_revalidation")
                enqueued = True
        except Exception:
            async with self._revalidation_lock:
                # The obligation stays recoverable: the claim reverts rather than closing over a
                # measurement that did not complete.
                self._revalidation.pop(key, None)
            raise

        async with self._revalidation_lock:
            current = self._revalidation.get(key)
            if current is not None and current.get("pair") == pair:
                current["state"] = "complete"
        if enqueued:
            self.trigger(team_id=team_id, channel_id=channel_id, namespace=namespace,
                         path=PATH_NEAR_TRIGGER,
                         headroom_tokens=int(getattr(evidence, "fixed_tokens", 0)))
        return {"claimed": True, "enqueued": enqueued}

    # ------------------------------------------------------------------ sweep

    async def sweep(self) -> int:
        """Retention pass. Re-reads the pin set under the lock it deletes against, and PROTECTS
        every live crawl checkpoint's parent — without that the sweep can retire the one lineage
        an in-progress incremental crawl needs, forcing it back to raw on a channel whose Slack
        retention may not support one."""
        protected: List[str] = []
        getter = getattr(self.db, "live_checkpoint_parent_ids_async", None)
        if callable(getter):
            protected = list(await getter() or [])
        async with self._pin_lock:
            pinned = self.pinned_ids()
            return await self.db.sweep_snapshots_async(
                pinned, retain_generations=self.retain_generations,
                retain_days=self.retain_days, protected_ids=protected)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Boot: recover cancellations, hydrate the task map, then open for triggers.

        BOOT HYDRATION IS NOT A TRIGGER. An expired dormant row STAYS DORMANT here: treating
        startup as a trigger would turn every restart into a free revival, and a crash-looping
        process into exactly the spin the backoff exists to prevent.
        """
        self._closing.clear()
        await self.recover_cancellations()
        self._accepting = True
        await self.hydrate()
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.ensure_future(self._drain_loop())

    async def stop(self) -> None:
        """Close to new work and let the running tasks unwind.

        The accepting flag drops FIRST, so the completion hook of anything finishing during the
        drain spawns nothing. A task cancelled here takes CRASH SEMANTICS deliberately (§1n test
        90): the in-flight chunk is refetched after restart, which costs one chunk and returns
        shutdown promptly, rather than waiting out a chunk that may be minutes long.
        """
        self._accepting = False
        self._closing.set()
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._drain_task is not None:
            self._drain_task.cancel()
            await asyncio.gather(self._drain_task, return_exceptions=True)
            self._drain_task = None

    async def hydrate(self) -> int:
        """Rebuild the task map from `pending_recompaction` (§1m).

        The durable row is why the headroom fields are carried: hydration must be able to size the
        job without redoing the measurement that produced it. An ACTIVE row means "this should be
        running", so hydration starts it — that is also the repair for the CAS-to-registration
        seam, where a cancellation between the committed revival and task registration would
        otherwise leave an active row with no live task.
        """
        rows = await self.db.all_pending_recompactions_async()
        started = 0
        for row in rows:
            team_id = str(row.get("team_id"))
            channel_id = str(row.get("channel_id"))
            namespace = str(row.get("namespace") or PROD_NAMESPACE)
            try:
                profile = (await self.sizing_for(team_id, channel_id))["sizing_profile"]
                # RECONCILIATION POINT 1.
                await self.reconcile_profile(team_id, channel_id, namespace=namespace,
                                             current_profile=profile)
                current = await self.db.load_pending_recompaction_async(
                    team_id, channel_id, namespace)
            except Exception as e:  # noqa: BLE001 — one bad channel never stops the boot
                logger.warning(f"{channel_id}: obligation not hydrated: {e}")
                continue
            if not current or current.get("state") != "active":
                continue          # dormant stays dormant: hydration is not a trigger
            if profile not in (current.get("requirements") or {}):
                continue
            self.trigger(team_id=team_id, channel_id=channel_id, namespace=namespace,
                         path=PATH_NEAR_TRIGGER, hydrated=True)
            started += 1
        if started:
            logger.info(f"Compaction coordinator hydrated {started} active obligation(s)")
        return started

    async def recover_cancellations(self) -> int:
        """AN INTENT PLUS AN ORPHAN CHECKPOINT IS A DETERMINISTIC INSTRUCTION: finish the discard.

        Through THE SAME atomic accessor the live chunk-boundary cleanup calls — one transaction,
        one code path. That is what makes intent-without-checkpoint genuinely impossible rather
        than merely unlikely, and giving the two paths separate implementations is exactly how that
        divergence gets introduced. Recovery is idempotent, so a crash during it replays safely.
        """
        try:
            intents = await self.db.all_cancellation_intents_async()
        except Exception as e:  # noqa: BLE001 — never block a boot on recovery
            logger.warning(f"Cancellation intents unread at boot: {e}")
            return 0
        finished = 0
        for intent in intents:
            try:
                if await self.finish_discard(
                        team_id=str(intent["team_id"]), channel_id=str(intent["channel_id"]),
                        namespace=str(intent["namespace"]), crawl_id=str(intent["crawl_id"])):
                    finished += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Cancellation recovery for {intent.get('crawl_id')}: {e}")
        if finished:
            logger.info(f"Finished {finished} cancelled compaction attempt(s) at boot")
        return finished

    async def finish_discard(self, *, team_id: str, channel_id: str, namespace: str,
                             crawl_id: str, outbox_rows: Sequence[Dict[str, Any]] = (),
                             candidate_id: Optional[str] = None) -> bool:
        """THE shared cleanup. Live chunk-boundary cancellation and boot recovery both call this
        and nothing else, so the two can never diverge (test 101c)."""
        return await self.db.finish_cancellation_discard_async(
            team_id=team_id, channel_id=channel_id, namespace=namespace, crawl_id=crawl_id,
            outbox_rows=list(outbox_rows), candidate_id=candidate_id)

    # ------------------------------------------------------------------ the drainer (§1l)

    async def drain_outbox(self, *, bounded: bool = False, timeout: float = 30.0) -> int:
        """Emit outbox rows in `outbox_seq` ORDER and delete each only AFTER acknowledgement.

        DELIVERY ORDER IS `outbox_seq`, NOT THE IDENTITY TRIPLE: `crawl_id` is a random uuid4, so
        identity order is not time order — a drainer ordering by the triple emits BACKWARD the
        first time a lexicographically smaller crawl is inserted between batches.

        THE DRAIN HALTS AT THE FIRST UNACKNOWLEDGED ROW, GLOBALLY. Skipping ahead after a failure
        puts `publish` in the ledger with no `build` before it — the precise failure `event_seq`
        exists to prevent — and buys nothing, because the sink is unavailable and the next row will
        fail too. Rows inserted during the backoff take higher `outbox_seq` values and wait.

        A POISONED ROW IS NOT A SINK OUTAGE: it halts ordered emission, STAYS IN PLACE, and logs
        CRITICAL bounded to once per row per boot. Retrying it every 30 seconds forever would bury
        the one message that identifies it while presenting a permanent defect as a transient
        outage. `bounded` is the shutdown form — one pass, no retry loop.
        """
        from message_processor.participation_telemetry import emit_outbox_body
        emitted = 0
        while True:
            try:
                batch = await self.db.read_outbox_batch_async(DRAIN_BATCH)
            except Exception as e:  # noqa: BLE001 — a closed DB at shutdown is not a data loss
                logger.warning(f"Compaction telemetry outbox unreadable: {e}")
                return emitted
            if not batch:
                return emitted
            for row in batch:
                clause = row.get("invalid")
                if clause:
                    self._log_poison(row, clause)
                    return emitted          # halt, and the row STAYS
                if not emit_outbox_body(row["body"], timeout=timeout):
                    return emitted          # halt at the first unacknowledged row, globally
                await self.db.delete_outbox_row_async(int(row["outbox_seq"]))
                emitted += 1
            if bounded or len(batch) < DRAIN_BATCH:
                return emitted

    def _log_poison(self, row: Dict[str, Any], clause: str) -> None:
        seq = int(row.get("outbox_seq") or 0)
        if seq in self._poison_logged:
            return
        self._poison_logged.add(seq)
        logger.critical(
            f"CRITICAL: compaction telemetry outbox row {seq} fails validation clause {clause!r} "
            f"(crawl_id={row.get('crawl_id')!r} attempt_seq={row.get('attempt_seq')!r} "
            f"event_seq={row.get('event_seq')!r}). Ordered emission is HALTED and the row is left "
            f"in place for inspection; repair or delete it and draining resumes on its own.")

    async def _drain_loop(self) -> None:
        """Backoff starts at 30s and DOUBLES to a 10-minute ceiling; a productive pass resets it.

        THE HALT IS A PROPERTY OF THE ROW'S CONTENT, NEVER A LATCHED PROCESS STATE — the halted
        row is re-read on the ordinary cadence, so an administratively repaired or deleted row
        resumes draining with NO RESTART.
        """
        backoff = DRAIN_BACKOFF_START
        while not self._closing.is_set():
            try:
                emitted = await self.drain_outbox()
                backoff = DRAIN_BACKOFF_START if emitted else min(
                    backoff * 2, DRAIN_BACKOFF_CEILING)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Compaction telemetry drain pass failed: {e}")
                backoff = min(backoff * 2, DRAIN_BACKOFF_CEILING)
            try:
                await asyncio.wait_for(self._closing.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------ telemetry

    @staticmethod
    def _emit(**fields: Any) -> None:
        """A DIRECT-WRITE compaction op (§1l routing table). Best-effort by design: these are
        observability, and the DB rows are the durable truth. The outbox exists for the two ops
        that carry a cardinality contract, which is a different thing from an audit log."""
        try:
            from message_processor import participation_telemetry
            participation_telemetry.compaction_snapshot(
                **{k: v for k, v in fields.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"compaction_snapshot not emitted: {e}")
