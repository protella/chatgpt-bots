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
from typing import Any, Dict, List, Optional, Union

from config import config
from logger import setup_logger
# Canonical home is openai_client (message_processor imports it, so the reverse would cycle).
# Re-exported here because this is where callers naturally look for it.
from openai_client.container_errors import AUTO_CONTAINER, is_container_gone

__all__ = ["AUTO_CONTAINER", "ContainerManager", "is_container_gone",
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

    async def get_or_create(self, thread_key: str) -> Union[str, Dict[str, str]]:
        """The container to hand the code_interpreter tool for this thread.

        Returns a container id, or `AUTO_CONTAINER` when we could not get a persistent one —
        the tool stays enabled either way.
        """
        if not thread_key or self.db is None:
            # No thread identity (or no DB) means nothing to scope a container TO.
            return await self._create_or_auto(thread_key or "")

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

        return await self._create_or_auto(thread_key)

    async def _create_or_auto(self, thread_key: str) -> Union[str, Dict[str, str]]:
        created = await self._create(thread_key)
        return created if created else AUTO_CONTAINER

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
