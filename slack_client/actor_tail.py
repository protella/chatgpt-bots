"""Per-thread actor tail — who spoke in a thread, not what they said (spec §2).

Extracted from ChannelPulse, which is retired. The content ring is gone; the stream is the room
now. What survives is structural and load-bearing: `thread_has_other_bot` is how the
deterministic 1:1-thread continuation fast path — the route that answers with no gate involved
at all — learns that a second agent is present beyond the replies fast path's first page.

Two writers, and they must not fight. The live feed records events as they arrive; a turn's
stream fetch hydrates the same window from Slack. A generation counter arbitrates: the fetch
captures the channel's generation before it starts, every live mutation bumps it, and a
reconcile whose generation no longer matches SKIPS. Live wins, and the next turn re-hydrates.

Root asymmetry, preserved deliberately: a root is filed under its OWN ts (`thread_ts or ts`), so
a root and its replies land in the same thread bucket. Filing a root under None — which is what
the raw Slack payload suggests — is how `thread_has_other_bot` silently stopped matching.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import config
from slack_client.normalizer import NormalizedMessage, in_window, parse_ts


@dataclass(frozen=True)
class ActorEntry:
    ts: str
    is_bot: bool
    sender_type: str


@dataclass(frozen=True)
class TailRecord:
    """One actor observation, with the thread it belongs to."""
    root_ts: str
    ts: str
    is_bot: bool
    sender_type: str


def tail_record(message: NormalizedMessage) -> TailRecord:
    return TailRecord(root_ts=message.root_ts, ts=message.ts,
                      is_bot=message.sender_type != "human",
                      sender_type=message.sender_type)


def _depth() -> int:
    return int(getattr(config, "participation_thread_tail", 15))


def _threads_max() -> int:
    return int(getattr(config, "actor_tail_threads_max", 50))


def _channels_max() -> int:
    return int(getattr(config, "actor_tail_channels_max", 30))


class ActorTail:
    """Per-channel, per-thread actor rings with LRU bounds and a generation counter."""

    def __init__(self) -> None:
        # channel -> thread root -> ts -> entry. OrderedDicts carry the LRU order.
        self._tails: "OrderedDict[str, OrderedDict[str, Dict[str, ActorEntry]]]" = OrderedDict()
        # channel -> ts -> thread root, so a removal by ts alone can find its bucket.
        self._index: Dict[str, Dict[str, str]] = {}
        self._generations: Dict[str, int] = {}

    # -- generation ------------------------------------------------------------------------

    def generation(self, channel_id: Optional[str]) -> int:
        return self._generations.get(channel_id or "", 0)

    def _bump(self, channel_id: str) -> None:
        self._generations[channel_id] = self._generations.get(channel_id, 0) + 1

    # -- live feed -------------------------------------------------------------------------

    def record(self, channel_id: Optional[str], *, ts: Optional[str], root_ts: Optional[str],
               is_bot: bool, sender_type: str) -> None:
        """Record one actor. Idempotent on (channel, ts) — the message and app_mention
        listeners both see a mention, and one of them is not a second speaker."""
        if not channel_id or not ts or _depth() <= 0:
            return
        try:
            parse_ts(ts)
        except ValueError:
            return
        bucket = root_ts or ts
        self._store(channel_id, bucket, ActorEntry(str(ts), bool(is_bot), sender_type))
        self._bump(channel_id)

    def record_message(self, message: NormalizedMessage) -> None:
        rec = tail_record(message)
        self.record(message.channel_id, ts=rec.ts, root_ts=rec.root_ts,
                    is_bot=rec.is_bot, sender_type=rec.sender_type)

    def remove(self, channel_id: Optional[str], ts: Optional[str]) -> bool:
        """Forget one ts (a deletion, or a reaction receipt being taken back)."""
        if not channel_id or not ts:
            return False
        index = self._index.setdefault(channel_id, {})
        bucket = index.pop(str(ts), None)
        if bucket is None:
            return False
        chan = self._tails.get(channel_id)
        entries = chan.get(bucket) if chan is not None else None
        removed = bool(entries and entries.pop(str(ts), None) is not None)
        if chan is not None and entries is not None and not entries:
            chan.pop(bucket, None)
        if removed:
            self._bump(channel_id)
        return removed

    # -- reader ----------------------------------------------------------------------------

    def thread_has_other_bot(self, channel_id: Optional[str], root_ts: Optional[str]) -> bool:
        """True when the thread's ring holds a message from a bot that is not us.

        The sole reader of the ring, and the reason it still exists at all.
        """
        entries = self._bucket(channel_id, root_ts)
        if not entries:
            return False
        return any(e.is_bot and e.sender_type != "self" for e in entries.values())

    def entries(self, channel_id: Optional[str],
                root_ts: Optional[str]) -> Tuple[ActorEntry, ...]:
        """Test/diagnostic view: one thread's ring, oldest first."""
        entries = self._bucket(channel_id, root_ts)
        if not entries:
            return ()
        return tuple(sorted(entries.values(), key=lambda e: parse_ts(e.ts)))

    def _bucket(self, channel_id: Optional[str],
                root_ts: Optional[str]) -> Optional[Dict[str, ActorEntry]]:
        if not channel_id or not root_ts:
            return None
        chan = self._tails.get(channel_id)
        return chan.get(str(root_ts)) if chan is not None else None

    # -- turn-path hydration ----------------------------------------------------------------

    def reconcile_window(self, channel_id: str, records: Iterable[TailRecord], *,
                         window: Tuple[Any, bool, Any],
                         expected_generation: int) -> bool:
        """Replace the covered interval with what the fetch actually saw.

        Returns False without touching anything when the channel's generation has moved: a live
        event landed while we were fetching, so what we hold is newer than what we fetched. The
        next turn re-hydrates.
        """
        if self.generation(channel_id) != expected_generation:
            return False
        floor_ts, floor_inclusive, high = window
        chan = self._tails.get(channel_id)
        if chan is not None:
            for bucket in list(chan.keys()):
                entries = chan[bucket]
                for ts in list(entries.keys()):
                    try:
                        covered = in_window(ts, floor_ts, floor_inclusive, high)
                    except ValueError:
                        covered = False
                    if covered:
                        entries.pop(ts, None)
                        (self._index.get(channel_id) or {}).pop(ts, None)
                if not entries:
                    chan.pop(bucket, None)
        for rec in records:
            self._store(channel_id, rec.root_ts or rec.ts,
                        ActorEntry(str(rec.ts), bool(rec.is_bot), rec.sender_type))
        return True

    # -- storage ---------------------------------------------------------------------------

    def _store(self, channel_id: str, bucket: str, entry: ActorEntry) -> None:
        chan = self._tails.get(channel_id)
        if chan is None:
            chan = self._tails[channel_id] = OrderedDict()
        self._tails.move_to_end(channel_id)
        entries = chan.get(bucket)
        if entries is None:
            entries = chan[bucket] = {}
        chan.move_to_end(bucket)
        entries[entry.ts] = entry
        self._index.setdefault(channel_id, {})[entry.ts] = bucket
        self._trim_thread(channel_id, bucket, entries)
        self._evict(channel_id, chan)

    def _trim_thread(self, channel_id: str, bucket: str,
                     entries: Dict[str, ActorEntry]) -> None:
        keep = max(1, _depth() + 2)
        if len(entries) <= keep:
            return
        ordered: List[str] = sorted(entries.keys(), key=parse_ts)
        for ts in ordered[:len(ordered) - keep]:
            entries.pop(ts, None)
            (self._index.get(channel_id) or {}).pop(ts, None)

    def _evict(self, channel_id: str,
               chan: "OrderedDict[str, Dict[str, ActorEntry]]") -> None:
        while len(chan) > max(1, _threads_max()):
            bucket, entries = chan.popitem(last=False)
            index = self._index.get(channel_id) or {}
            for ts in entries:
                index.pop(ts, None)
        while len(self._tails) > max(1, _channels_max()):
            evicted, _ = self._tails.popitem(last=False)
            self._index.pop(evicted, None)
            self._generations.pop(evicted, None)

    def reset(self) -> None:
        """Test seam."""
        self._tails.clear()
        self._index.clear()
        self._generations.clear()


# The module-level singleton. W2b wires the live feed onto it; nothing redefines these
# structures.
actor_tail = ActorTail()


def record(channel_id: Optional[str], *, ts: Optional[str], root_ts: Optional[str],
           is_bot: bool, sender_type: str) -> None:
    actor_tail.record(channel_id, ts=ts, root_ts=root_ts, is_bot=is_bot,
                      sender_type=sender_type)


def record_message(message: NormalizedMessage) -> None:
    actor_tail.record_message(message)


def remove(channel_id: Optional[str], ts: Optional[str]) -> bool:
    return actor_tail.remove(channel_id, ts)


def thread_has_other_bot(channel_id: Optional[str], root_ts: Optional[str]) -> bool:
    return actor_tail.thread_has_other_bot(channel_id, root_ts)


def generation(channel_id: Optional[str]) -> int:
    return actor_tail.generation(channel_id)


def reconcile_window(channel_id: str, records: Iterable[TailRecord], *,
                     window: Tuple[Any, bool, Any], expected_generation: int) -> bool:
    return actor_tail.reconcile_window(channel_id, records, window=window,
                                       expected_generation=expected_generation)
