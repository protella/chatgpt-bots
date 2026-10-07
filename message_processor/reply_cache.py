"""One fetch of a thread's replies, shared — across a burst's concurrent stream builds and the
reconsideration rebuilds that follow them.

THE FAILURE. Every channel stream build re-fetched the replies of every thread root in its
window: about sixty `conversations.replies` calls per build in a busy channel. Three quick
messages from one person meant three builds plus two reconsideration rebuilds inside a minute,
Slack's Tier 3 limit answered 429, and the newest turn and both rebuilds failed closed — the
person who asked three times got nothing.

THE CACHE. Per channel, per thread root: the replies the last fetch returned, keyed by what the
thread looked like when they were fetched — the parent's `latest_reply` and `reply_count` from
the `conversations.history` page, and the activity index's pinned event ts for the root. A build
reuses an entry only when that key is unchanged, the root is not dirty in the index, and the
entry's fetch window covers what this build needs. Anything else is fetched.

WHAT CAN CHANGE A THREAD WITHOUT MOVING THE KEY, and how each is caught:
  * someone else edits, deletes or reacts to a message — the inbound event drops the entries
    that hold it (`forget`, fed from the raw event listeners);
  * WE post into, edit, delete, stream into or react to a message — Bolt never shows us our own
    events, so the Slack client's own mutation methods drop the entries (`install_on_client`);
  * a mutation lands WHILE a fetch is in flight — that fetch's result is never stored, and never
    handed to a request made after the mutation (that request fetches fresh); the request that
    started it, which predates the mutation, keeps it like any fetch racing an edit.

SINGLE FLIGHT. Builds that ask for the same root under the same key at the same time share one
request, when its window covers theirs. The request is its own task and every requester awaits it
shielded, so it SURVIVES the cancellation of the turn that started it — a superseded turn's
half-done fetches land in the cache for the newer turn instead of being thrown away. It stays
bounded by the starting build's own fetch budget.

BOUNDS — no invented number. A channel keeps only the roots of its most recent build window
(`retain`); channels are LRU-bounded by the existing PARTICIPATION_ACTIVITY_LRU_MAX.

A CACHE, NOT STATE. Process memory only: a restart starts empty and every build reads Slack.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from config import config
from logger import setup_logger
from slack_client.normalizer import TimestampError, parse_ts

logger = setup_logger(name="slack_bot.ReplyCache")

# (parent latest_reply, parent reply_count, activity-index pinned event ts)
ReplyKey = Tuple[Optional[str], Optional[int], Optional[str]]

SOURCE_FETCHED = "fetched"
SOURCE_CACHE = "cache"
SOURCE_SHARED = "shared"


def _key_newest(key: ReplyKey) -> Optional[str]:
    """The newest ts the key vouches for — what a fetch window must reach to hold every reply."""
    stamps = [s for s in (key[0], key[2]) if s]
    if not stamps:
        return None
    try:
        return max(stamps, key=parse_ts)
    except TimestampError:
        return None


def cacheable(key: ReplyKey) -> bool:
    """Only a key that names a reply ts can tell an unchanged thread from a changed one."""
    return _key_newest(key) is not None


def _covers(floor_ts: str, high: str, want_floor: str, key: ReplyKey) -> bool:
    """Does a fetch made over [floor_ts, high] hold every reply a build over [want_floor, …]
    needs? It must reach as far back, and as far forward as the newest reply the key names
    (nothing newer exists while the key is unchanged)."""
    newest = _key_newest(key)
    if newest is None:
        return False
    try:
        return (parse_ts(floor_ts) <= parse_ts(want_floor)
                and parse_ts(newest) <= parse_ts(high))
    except TimestampError:
        return False


@dataclass(frozen=True)
class _Entry:
    key: ReplyKey
    floor_ts: str
    high: str
    messages: Tuple[Any, ...]
    message_ts: FrozenSet[str]


@dataclass
class _Flight:
    task: "asyncio.Future[List[Any]]"
    key: ReplyKey
    floor_ts: str
    high: str
    # Every mutation the channel saw while this fetch was in flight, as (channel seq, ts).
    mutations: List[Tuple[int, str]] = field(default_factory=list)
    # A mutation named this root itself: no request may join it any more.
    invalid: bool = False

    async def join(self) -> List[Any]:
        """Await the shared fetch without owning it: a cancelled requester leaves it running."""
        return list(await asyncio.shield(self.task))

    def stale_for(self, root_ts: str, messages: List[Any], since: Optional[int] = None) -> bool:
        """Did a mutation (up to channel seq `since`, or any) touch the root or a message this
        fetch returned?"""
        stamps = {str(getattr(m, "ts", "")) for m in messages} | {root_ts}
        return any(ts in stamps and (since is None or seq <= since)
                   for seq, ts in self.mutations)


@dataclass
class _Channel:
    entries: Dict[str, _Entry] = field(default_factory=dict)
    flights: Dict[Tuple[str, ReplyKey], _Flight] = field(default_factory=dict)
    seq: int = 0


class ReplyCache:
    def __init__(self) -> None:
        self._channels: "OrderedDict[str, _Channel]" = OrderedDict()

    def _channel(self, channel_id: str) -> _Channel:
        channel = self._channels.get(channel_id)
        if channel is None:
            channel = self._channels[channel_id] = _Channel()
            limit = max(1, int(getattr(config, "participation_activity_lru_max", 1024)))
            while len(self._channels) > limit:
                self._channels.popitem(last=False)
        else:
            self._channels.move_to_end(channel_id)
        return channel

    async def get(self, channel_id: str, root_ts: str, key: ReplyKey, *, floor_ts: str,
                  high: str, dirty: bool,
                  fetch: Callable[[], Awaitable[List[Any]]]) -> Tuple[List[Any], str]:
        """The replies of one root and where they came from (fetched / cache / shared).

        Cached and shared results carry the window they were FETCHED over; the caller narrows
        them to its own window."""
        channel = self._channel(channel_id)
        reusable = cacheable(key) and not dirty
        if reusable:
            entry = channel.entries.get(root_ts)
            if entry is not None and entry.key == key and _covers(entry.floor_ts, entry.high,
                                                                  floor_ts, key):
                return list(entry.messages), SOURCE_CACHE
            flight = channel.flights.get((root_ts, key))
            if (flight is not None and not flight.invalid
                    and _covers(flight.floor_ts, flight.high, floor_ts, key)):
                asked_at = channel.seq
                shared = await flight.join()
                if not flight.stale_for(root_ts, shared, since=asked_at):
                    return shared, SOURCE_SHARED
                # Invalidated before this request was made: fetch fresh.
        if not reusable:
            # Nothing anyone else could share or reuse (a dirty root, a key that cannot tell
            # versions apart): fetched in the caller's own task, cancelled with it.
            return list(await fetch()), SOURCE_FETCHED
        flight = _Flight(task=asyncio.ensure_future(fetch()), key=key, floor_ts=floor_ts,
                         high=high)
        displaced = channel.flights.get((root_ts, key))
        if displaced is not None:
            # Out of the map it stops hearing invalidations, so it must never land.
            displaced.invalid = True
        channel.flights[(root_ts, key)] = flight
        flight.task.add_done_callback(
            lambda task: self._land(channel, root_ts, flight, task))
        return await flight.join(), SOURCE_FETCHED

    @staticmethod
    def _land(channel: _Channel, root_ts: str, flight: _Flight,
              task: "asyncio.Future[List[Any]]") -> None:
        """A shared fetch finished, whoever is still waiting for it: store what it read unless a
        mutation of one of its messages raced it. Runs as the task's done-callback, so it lands
        even when every requester has been cancelled."""
        if channel.flights.get((root_ts, flight.key)) is flight:
            channel.flights.pop((root_ts, flight.key), None)
        if task.cancelled() or task.exception() is not None:
            return
        messages = list(task.result())
        if flight.invalid or flight.stale_for(root_ts, messages):
            return
        stamps = {str(getattr(m, "ts", "")) for m in messages} | {root_ts}
        channel.entries[root_ts] = _Entry(
            key=flight.key, floor_ts=flight.floor_ts, high=flight.high,
            messages=tuple(messages), message_ts=frozenset(stamps))

    def forget(self, channel_id: Optional[str], ts: Optional[str]) -> None:
        """A message changed (edited, deleted, reacted to, replied under): drop every entry that
        holds it, and remember it against fetches still in flight. Synchronous; never raises."""
        if not channel_id or not ts:
            return
        channel = self._channels.get(str(channel_id))
        if channel is None:
            return
        value = str(ts)
        channel.seq += 1
        for (root, _key), flight in list(channel.flights.items()):
            flight.mutations.append((channel.seq, value))
            if root == value:
                # The root itself changed: the flight can never be shared again.
                flight.invalid = True
                channel.flights.pop((root, _key), None)
        for root in [r for r, e in channel.entries.items()
                     if r == value or value in e.message_ts]:
            channel.entries.pop(root, None)

    def retain(self, channel_id: str, roots: Iterable[str]) -> None:
        """Keep only the roots of this channel's most recent build window."""
        channel = self._channels.get(channel_id)
        if channel is None:
            return
        keep = set(roots)
        for root in [r for r in channel.entries if r not in keep]:
            channel.entries.pop(root, None)

    def reset(self) -> None:
        self._channels.clear()


_CACHE = ReplyCache()


def cache() -> ReplyCache:
    return _CACHE


def forget(channel_id: Optional[str], ts: Optional[str]) -> None:
    _CACHE.forget(channel_id, ts)


def note_inbound_event(event: Any) -> None:
    """Feed one raw Slack event (message or reaction) into the cache's invalidation. Called
    synchronously from the raw listeners; never raises."""
    try:
        if not isinstance(event, dict):
            return
        if event.get("type") in ("reaction_added", "reaction_removed"):
            item = event.get("item") or {}
            forget(item.get("channel"), item.get("ts"))
            return
        channel_id = event.get("channel")
        subtype = event.get("subtype") or ""
        if subtype == "message_deleted":
            prev = event.get("previous_message") or {}
            forget(channel_id, event.get("deleted_ts") or prev.get("ts"))
            forget(channel_id, prev.get("thread_ts"))
            return
        payload = event.get("message") if subtype == "message_changed" else event
        if not isinstance(payload, dict):
            return
        forget(channel_id, payload.get("ts"))
        thread_ts = payload.get("thread_ts")
        if thread_ts and str(thread_ts) != str(payload.get("ts")):
            forget(channel_id, thread_ts)
    except Exception as e:  # noqa: BLE001 — a missed invalidation must never cost the event
        logger.debug(f"reply cache invalidation skipped: {e}")


# Our own mutations, read off the Slack call's arguments: the message ts it changes, and the
# thread root a new message or upload lands under.
_OWN_MUTATIONS = ("chat_postMessage", "chat_update", "chat_delete", "chat_startStream",
                  "chat_appendStream", "chat_stopStream", "reactions_add", "reactions_remove",
                  "files_completeUploadExternal")


def _forget_own(kwargs: Dict[str, Any]) -> None:
    channel_id = kwargs.get("channel") or kwargs.get("channel_id")
    for name in ("ts", "timestamp", "thread_ts"):
        forget(channel_id, kwargs.get(name))


def install_on_client(web_client: Any) -> None:
    """Wrap the Slack client's own message mutations so each drops the cache entries it touches
    — before the call (a fetch racing it is not stored) and after it (a fetch that finished in
    between is dropped). Bolt never delivers our own events, so this is the only place they are
    seen. Idempotent."""
    if web_client is None or getattr(web_client, "_reply_cache_installed", False):
        return
    for name in _OWN_MUTATIONS:
        found = getattr(web_client, name, None)
        if not callable(found):
            continue
        original: Callable[..., Any] = found

        async def wrapper(*args: Any, _original: Callable[..., Any] = original,
                          **kwargs: Any) -> Any:
            _forget_own(kwargs)
            try:
                return await _original(*args, **kwargs)
            finally:
                _forget_own(kwargs)

        setattr(web_client, name, wrapper)
    try:
        web_client._reply_cache_installed = True
    except Exception:  # noqa: BLE001 — a client that refuses attributes simply re-wraps
        pass


__all__ = ["ReplyCache", "ReplyKey", "cache", "cacheable", "forget", "install_on_client",
           "note_inbound_event", "SOURCE_CACHE", "SOURCE_FETCHED", "SOURCE_SHARED"]
