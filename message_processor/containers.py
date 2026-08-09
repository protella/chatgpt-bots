"""F32: thread-scoped code-interpreter containers.

One OpenAI container per Slack thread, its id persisted in `thread_containers` and reused
across turns, so the sandbox keeps its working state within a conversation: the CSV the model
cleaned up on turn 1 is still sitting in /mnt/data on turn 2 when the user says "now chart it".

**The 20-minute ceiling is the API's, not ours.** `expires_after.minutes` must be <= 20 —
asking for 60 returns HTTP 400 ("integer above maximum value. Expected a value <= 20"). There
is no way to hold a container longer, so "persistent" here means *warm within an active
conversation*, and a revived thread always gets a fresh, empty one. That is exactly the
recreate-on-demand behaviour the design calls for; it just isn't a choice we get to make.

Three failure modes drive the shape of this module:

* **A dead id fails the whole turn.** Handing OpenAI an expired container id returns 404 and the
  user gets an error instead of an answer. Reuse is therefore *confirmed* with a retrieve()
  first, and the DB's reuse window (< TTL) keeps us away from the edge. The window is necessary
  but NOT sufficient: `last_used_at` records when *we* last used the id, which is not when the
  container was last *active* — a failed API call never touched it. Only retrieve() knows. And
  even that is not a lease, so `openai_client.container_errors` catches the residual mid-turn
  404 at the Responses-call boundary.
* **The listing is CUMULATIVE.** A reused container still holds every file from every earlier
  turn. Anything already in it when a turn starts is, by definition, not this turn's output —
  hence the baseline snapshot in `get_or_create`. Without it, ten leftover files from turn 1
  would consume turn 2's publication cap and the chart the user just asked for would be dropped.
* **Nothing here may break code interpreter.** Every failure degrades to `{"type": "auto"}`
  (a fresh throwaway container, the pre-F32 behaviour) rather than dropping the tool. Losing
  sandbox continuity is a bad turn; losing the sandbox is a broken feature.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

from config import config
from logger import setup_logger
# Canonical home is openai_client (message_processor imports it, so the reverse would cycle).
# Re-exported here because this is where callers naturally look for it.
from openai_client.container_errors import AUTO_CONTAINER, adoption_blocked, is_container_gone

__all__ = ["AUTO_CONTAINER", "ContainerManager", "adoption_blocked", "is_container_gone",
           "publication_lock", "wait_for_publication"]

# Must go through setup_logger: the app attaches handlers to `slack_bot.*` loggers and sets
# propagate=False, so a bare getLogger(__name__) writes to NOWHERE. Every warning in this
# module — expired containers, failed creates — would be invisible in production.
logger = setup_logger(name="slack_bot.Containers")

_API_TIMEOUT = 20.0


# --- Publication latch ------------------------------------------------------------------
#
# The thread lock is released when `process_message` returns, but artifacts are listed,
# downloaded and uploaded AFTER that, from main.py. So turn A can still be publishing while
# turn B is already running code in the SAME persistent container. Two things then go wrong:
# A's listing picks up B's half-written file and posts it under A's answer, and B's baseline
# misses the files A has not yet recorded, so both turns upload them.
#
# This latch closes that window: publication holds it, and the next turn's container resolution
# waits on it. It is per-thread, so unrelated conversations never block each other.
_publication_locks: Dict[str, asyncio.Lock] = {}
_publication_waiters: Dict[str, int] = {}


def publication_lock(thread_key: str) -> asyncio.Lock:
    """The publication latch for one thread. Created on demand."""
    lock = _publication_locks.get(thread_key)
    if lock is None:
        lock = asyncio.Lock()
        _publication_locks[thread_key] = lock
    _publication_waiters[thread_key] = _publication_waiters.get(thread_key, 0) + 1
    return lock


def release_publication_lock(thread_key: str) -> None:
    """Drop the bookkeeping for a finished waiter, and the lock itself once nobody holds it.

    Without the prune, one Lock per thread accumulates for the life of the process.
    """
    remaining = _publication_waiters.get(thread_key, 1) - 1
    if remaining > 0:
        _publication_waiters[thread_key] = remaining
        return
    _publication_waiters.pop(thread_key, None)
    lock = _publication_locks.get(thread_key)
    if lock is not None and not lock.locked():
        _publication_locks.pop(thread_key, None)


async def wait_for_publication(thread_key: str) -> None:
    """Block until the previous turn in this thread has finished publishing its artifacts.

    Cheap when uncontended (the common case): no lock exists, so this returns immediately.
    """
    lock = _publication_locks.get(thread_key)
    if lock is None or not lock.locked():
        return
    logger.debug(f"Waiting for the previous turn's artifact publication in {thread_key}")
    async with lock:
        pass


class ContainerManager:
    """Resolves the code-interpreter container for a thread. Never raises."""

    def __init__(self, openai_client: Any, db: Any = None):
        self.openai_client = openai_client
        self.db = db

    @property
    def _raw(self):
        return self.openai_client.client

    async def _create(self, thread_key: str) -> Optional[str]:
        """Mint a container for this thread and bind it. Returns None on failure."""
        try:
            container = await asyncio.wait_for(
                self._raw.containers.create(
                    # The name is what an operator sees in the OpenAI dashboard; make it
                    # traceable back to the Slack thread that owns it.
                    name=f"slackbot-{thread_key}"[:120],
                    expires_after={
                        "anchor": "last_active_at",
                        "minutes": config.code_interpreter_container_ttl_minutes,
                    },
                ),
                timeout=_API_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001 — fall back to `auto`, never break the tool
            logger.warning(f"Could not create container for {thread_key}: {e}")
            return None

        container_id = getattr(container, "id", None)
        if not container_id:
            return None

        if self.db is not None:
            try:
                await self.db.save_thread_container_async(thread_key, container_id)
            except Exception as e:  # noqa: BLE001
                # The container is real and usable — we just won't remember it next turn.
                logger.warning(f"Container {container_id} created but not persisted: {e}")

        logger.info(f"Created container {container_id} for thread {thread_key}")
        return container_id

    async def _is_alive(self, container_id: str) -> bool:
        """Confirm the container still exists before we bet a turn on it."""
        try:
            got = await asyncio.wait_for(
                self._raw.containers.retrieve(container_id), timeout=_API_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            if is_container_gone(e):
                logger.info(f"Container {container_id} has expired")
            else:
                # Network blip, timeout, 5xx — we genuinely don't know. Treat as dead: a fresh
                # container costs one API call, while a wrong "alive" costs the user's turn.
                logger.warning(f"Could not verify container {container_id}: {e}")
            return False
        return getattr(got, "status", None) == "running"

    async def _snapshot_baseline(self, thread_key: str, container_id: str) -> None:
        """Mark everything already in a reused container as "not this turn's".

        The listing is cumulative, so without this a turn inherits every file the model ever
        wrote in this thread. They would compete for the publication cap — ten leftovers from
        turn 1 can crowd out the one chart the user just asked for — and any gap in the durable
        record (a restart, an evicted id) would re-post them outright.

        Costs one extra listing call on reuse turns. That is worth it: it is exact, needs no
        clock comparison between our host and OpenAI's, and self-heals a lost dedupe record.
        """
        existing: List[str] = []
        try:
            async def _walk():
                pager = self._raw.containers.files.list(container_id=container_id)
                async for f in pager:
                    if getattr(f, "source", None) != "assistant":
                        continue
                    fid = getattr(f, "id", "")
                    if fid:
                        existing.append(fid)

            await asyncio.wait_for(_walk(), timeout=_API_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            # Failing OPEN here would re-post old files; failing closed only risks missing a
            # file this turn writes, and the model can be asked again. Neither is great, so:
            # keep the turn going, and let the durable record do what it can.
            logger.warning(f"Could not baseline container {container_id}: {e}")
            return

        if existing:
            await self.remember_published(thread_key, container_id, existing)
            logger.debug(f"Baselined {len(existing)} pre-existing file(s) in {container_id}")

    async def _live_binding(self, thread_key: str) -> Optional[str]:
        """This thread's persistent container, if it has one that is still alive.

        The reuse path in full: settle the previous turn's publication, read the binding, prove
        the container still exists, touch it, baseline what it already holds. Returns None when
        the thread is unbound or its container is gone (the dead binding is dropped on the way
        out, so the published-file record dies with the container it describes).
        """
        if not thread_key or self.db is None:
            # No thread identity (or no DB) means nothing to scope a container TO.
            return None

        # The previous turn may still be uploading out of this very container. Let it finish, so
        # our baseline sees a settled container and we cannot race it into publishing twice.
        await wait_for_publication(thread_key)

        try:
            row = await self.db.get_fresh_thread_container_async(
                thread_key, config.code_interpreter_container_reuse_minutes)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Container lookup failed for {thread_key}: {e}")
            row = None

        if row and row.get("container_id"):
            container_id = row["container_id"]
            if await self._is_alive(container_id):
                try:
                    await self.db.touch_thread_container_async(thread_key, container_id)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Could not touch container row for {thread_key}: {e}")
                await self._snapshot_baseline(thread_key, container_id)
                logger.debug(f"Reusing container {container_id} for thread {thread_key}")
                return container_id
            # Dead: drop the binding so the published-file record dies with it. Keeping the row
            # would let a stale cfile id suppress a genuinely new artifact in the replacement.
            await self.invalidate(thread_key, container_id)

        return None

    async def get_or_create(self, thread_key: str) -> Union[str, Dict[str, str]]:
        """The container to hand the code_interpreter tool for this thread.

        W3 — **an unbound thread is NOT worth a create here.** The client-side
        `containers.create` this used to run costs 0.7–4.0s of dead time before the request is
        even sent, on every first turn of every conversation, whether or not the model ends up
        running a line of code. `{"type": "auto"}` costs the API ~0.3s and blocks nothing, so a
        thread with no live binding starts there and the id is ADOPTED the moment the model
        actually runs something (`adopt`, driven from the tool loop's round boundaries).

        Reuse is unchanged: a thread that already has a live container still gets it back, which
        is what makes "now chart that file again" work across turns.

        Callers that genuinely need an addressable sandbox up front — a research job with files
        to mount — want `create_explicit`, not this.
        """
        return await self._live_binding(thread_key) or AUTO_CONTAINER

    async def create_explicit(self, thread_key: str) -> Union[str, Dict[str, str]]:
        """Reuse this thread's container, or MINT one now — the pre-W3 `get_or_create`.

        Blocking and worth it only where the caller cannot proceed without an addressable id:
        a build phase has files to mount into the sandbox and a listing to read back out of it,
        and `auto` gives it neither. Still degrades to `AUTO_CONTAINER` rather than raising —
        losing sandbox continuity is a bad turn, losing the sandbox is a broken feature.
        """
        bound = await self._live_binding(thread_key)
        if bound:
            return bound
        created = await self._create(thread_key or "")
        return created if created else AUTO_CONTAINER

    # --- W3 adoption ----------------------------------------------------------------------
    #
    # Per-thread single flight. Adoption races two ways: a round boundary and a bridge tool can
    # both discover an unbound thread in the same instant, and two bridge calls in one round run
    # concurrently by construction (dispatch_all gathers them). Without a lock that is two real
    # containers created for one turn, one of them immediately orphaned.
    # Class-level, like the publication latch is module-level: two ContainerManager instances
    # for one thread would otherwise each hold their own idea of the lock and neither would
    # exclude the other. Pruned when the last waiter leaves — the count is taken BEFORE the
    # await, so an entry can never be dropped out from under someone queued on it.
    _sandbox_locks: Dict[str, asyncio.Lock] = {}
    _sandbox_lock_waiters: Dict[str, int] = {}

    @classmethod
    @asynccontextmanager
    async def _sandbox_lock(cls, thread_key: str) -> AsyncIterator[None]:
        lock = cls._sandbox_locks.get(thread_key)
        if lock is None:
            lock = asyncio.Lock()
            cls._sandbox_locks[thread_key] = lock
        cls._sandbox_lock_waiters[thread_key] = cls._sandbox_lock_waiters.get(thread_key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = cls._sandbox_lock_waiters.get(thread_key, 1) - 1
            if remaining > 0:
                cls._sandbox_lock_waiters[thread_key] = remaining
            else:
                cls._sandbox_lock_waiters.pop(thread_key, None)
                cls._sandbox_locks.pop(thread_key, None)

    async def _adopt_locked(self, thread_key: str, container_id: str) -> Optional[str]:
        """The CAS write itself. Caller holds `_sandbox_lock(thread_key)`."""
        try:
            bound = await self.db.adopt_thread_container_async(thread_key, container_id)
        except Exception as e:  # noqa: BLE001 — a lost binding costs continuity, never the turn
            logger.warning(f"Could not adopt container {container_id} for {thread_key}: {e}")
            return None
        if bound == container_id:
            logger.info(f"Adopted container {container_id} for thread {thread_key}")
        else:
            logger.info(
                f"Not adopting {container_id} for {thread_key}: already bound to {bound}")
        return bound

    async def adopt(self, thread_key: str, container_id: str) -> Optional[str]:
        """Bind a container the model is ALREADY running code in.

        **No baseline listing.** `_snapshot_baseline` exists because a REUSED container arrives
        holding files from earlier turns that are not this turn's output. An adopted one was
        created for this turn and we have been in it since it was empty, so everything in it is
        ours — a baseline call here would cost a round-trip to mark nothing, and on a slow
        listing it could mark this turn's own chart as pre-existing and drop it.

        Returns the container the thread is bound to afterwards, which is NOT necessarily ours:
        the DB write is a compare-and-set that never overwrites a live binding (see
        `database.adopt_thread_container`). The turn keeps using the id it observed regardless —
        that is where its files are — so the return value is for the record, not for routing.
        """
        if not thread_key or not container_id or self.db is None:
            return None
        async with self._sandbox_lock(thread_key):
            return await self._adopt_locked(thread_key, container_id)

    async def bridge_container(self, thread_key: str, holder: Any) -> Optional[str]:
        """An addressable container for a turn that started on `auto`, created on demand.

        `mount_file` and `create_image_asset` need an id to push bytes into. On an auto turn
        there is none until the model runs code, and these tools are precisely the ones that run
        BEFORE it does. So the first one to ask pays for the create, fills the turn's shared
        holder, and the tool loop pins the id into the next round's declaration.

        The holder is re-checked INSIDE the lock: sibling calls in the same round get one
        container between them, not one each.
        """
        if not thread_key or holder is None:
            return None
        async with self._sandbox_lock(thread_key):
            existing = getattr(holder, "container_id", None)
            if existing:
                return cast(str, existing)
            created = await self.create_explicit(thread_key)
            if not isinstance(created, str):
                logger.warning(f"No addressable container available for {thread_key}")
                return None
            if self.db is not None:
                await self._adopt_locked(thread_key, created)
            holder.container_id = created
            return created

    async def invalidate(self, thread_key: str, container_id: Optional[str] = None) -> None:
        """Forget a container binding. Best-effort; never raises.

        Scoped by container_id so a stale invalidation cannot unbind a container that a newer
        turn has already put in its place.
        """
        if self.db is None or not thread_key:
            return
        try:
            await self.db.delete_thread_container_async(thread_key, container_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not invalidate container for {thread_key}: {e}")

    async def get_published_files(self, thread_key: str, container_id: str) -> List[str]:
        """Container file ids that must NOT be published: already uploaded, or pre-existing.

        Read WITHOUT an age filter. A single turn can outlive the reuse window (a tool loop with
        slow tools), and a dedupe list that goes unreadable mid-turn means re-posting every
        earlier artifact in the container.

        Returns [] when the row now points at a different container — those ids describe a
        sandbox that no longer backs this thread.
        """
        if self.db is None or not thread_key or not container_id:
            return []
        try:
            row = await self.db.get_thread_container_async(thread_key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read published files for {thread_key}: {e}")
            return []
        if not row or row.get("container_id") != container_id:
            return []
        return list(row.get("published_files") or [])

    async def remember_published(self, thread_key: str, container_id: str,
                                 file_ids: List[str]) -> None:
        """Record ids as handled. No-ops for an ephemeral (`auto`) container, which has no row —
        writing them would poison a persistent binding that these files never came from."""
        if self.db is None or not thread_key or not container_id or not file_ids:
            return
        try:
            await self.db.add_published_container_files_async(thread_key, container_id, file_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not record published files for {thread_key}: {e}")

    async def reap(self) -> int:
        """Delete containers whose threads have gone quiet, and drop their rows.

        Wired into the daily cleanup worker. The containers themselves have already
        idle-expired by then (20-minute ceiling) — this is mostly row hygiene, and the API
        delete is best-effort precisely because a 404 is the *expected* outcome. Returns the
        number of rows reaped.
        """
        if self.db is None:
            return 0

        # A container idle past its TTL is gone by definition. Double it for margin so we
        # never reap a container a live turn is still using.
        cutoff = max(1, config.code_interpreter_container_ttl_minutes) * 2
        try:
            rows = await self.db.get_expired_thread_containers_async(cutoff)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Container reap query failed: {e}")
            return 0

        reaped = 0
        for row in rows:
            container_id = row.get("container_id")
            thread_key = row.get("thread_id")
            if container_id:
                try:
                    await asyncio.wait_for(
                        self._raw.containers.delete(container_id), timeout=_API_TIMEOUT)
                except Exception as e:  # noqa: BLE001
                    # Almost always a 404 — it already expired on its own. Not worth a warning.
                    logger.debug(f"Container {container_id} already gone: {e}")
            try:
                # Scoped to the container we actually selected: a turn may have rebound this
                # thread to a live container while we were deleting the stale one.
                await self.db.delete_thread_container_async(thread_key, container_id)
                reaped += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not drop container row {thread_key}: {e}")

        if reaped:
            logger.info(f"Reaped {reaped} expired code-interpreter container(s)")
        return reaped
