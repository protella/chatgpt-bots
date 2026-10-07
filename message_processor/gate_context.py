"""What was just said in a channel — a short recent-context ring for the wake gate.

THE GAP. The gate decides from the debounce cohort alone. A correction ("no, wait, not that
one — the other"), an addition ("and this too") or a follow-up to something the assistant just
answered reads as a bare fragment without the exchange it belongs to, and a fragment is easy to
sleep. This module holds the exchange: a few recent messages per channel, so a later wiring can
show the gate a short "Recent conversation before these messages" block.

THE SOURCE. An in-memory ring per channel, fed from two places and nowhere else:

  inbound   every channel message event that reaches the raw Slack listeners — including the
            ones the gate goes on to sleep — plus their edits, deletions and tombstones
            (`slack_client/event_handlers/registration.py`).
  outbound  our own replies as Slack accepted them (`slack_client/messaging.py`), because Bolt
            drops our own message events and the assistant's last answer is exactly the context
            a follow-up needs.

A GIST, NOT A TRANSCRIPT. The gate block caps every message at a few hundred characters, so
the text kept here is whatever the feeding call already had in hand — no reconstruction, no
Slack lookup, no await. The transcript is Slack's; this is a hint for one boolean decision.

COLD IS FINE. Restart = empty ring = empty block = the gate decides exactly as it does with no
context. Nothing here is persisted, and nothing on the gate's path may ever wait on it.

BOUNDS. A fixed per-channel capacity (`PARTICIPATION_GATE_CONTEXT_MAX`, oldest ts evicted
first) and an LRU over channels reusing `PARTICIPATION_ACTIVITY_LRU_MAX`. DMs are never recorded:
the gate never runs there, since every DM message gets its own turn.

ORDERING. Slack ts comparisons go through `stale_send_guard.ts_key` / `is_newer` — numeric,
never lexical — so this module and the send guard agree about what "older" means.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from config import config
from message_processor.message_timestamps import render_message_timestamp
from message_processor.stale_send_guard import is_newer, ts_key

SENDER_HUMAN = "human"
SENDER_SELF = "self"
SENDER_OTHER_BOT = "other_bot"

RECENT_CONTEXT_HEADER = ("Recent conversation before these messages (context only — decide about "
                         "the messages below, not these):")

# The in-flight fact (Part A2): one line per turn in this conversation that is writing a reply
# right now. Literal from the A/B harness, like the block format below.
INFLIGHT_PREFIX = "The assistant is currently writing a reply to: "

# Message subtypes that are somebody SAYING something. Everything else Slack sends through the
# message listener — joins, topic changes, pins, reminders — is room lifecycle, not
# conversation. An allowlist, so a subtype Slack adds later stays out until someone decides.
RECORDABLE_SUBTYPES = frozenset({"", "bot_message", "file_share", "thread_broadcast",
                                 "me_message"})


@dataclass(frozen=True)
class GateContextEntry:
    """One remembered message.

    `version` is the message's `edited.ts` (None for an unedited post). It is what makes
    "latest edit wins" hold whatever order Slack delivers in: an app_mention twin carrying the
    ORIGINAL text can arrive after the edit, and it must not put the old words back."""

    ts: str
    thread_ts: Optional[str]
    sender_id: Optional[str]
    sender_kind: str
    text: str
    # Names and kinds only (`participation.describe_attachment`), the gate's own format.
    attachments: Tuple[str, ...] = ()
    reply_count: Optional[int] = None
    sender_name: Optional[str] = None
    version: Optional[str] = None
    # A thread reply also sent to the channel: it reads as a reply, and it is a channel post too.
    is_broadcast: bool = False

    @property
    def is_thread_reply(self) -> bool:
        """A reply inside a thread. A root carries `thread_ts == ts` once it has replies, and
        is still a channel post."""
        return bool(self.thread_ts) and ts_key(self.thread_ts) != ts_key(self.ts)


def _is_dm(channel_id: str) -> bool:
    return channel_id.startswith("D")


def _capacity() -> int:
    return int(getattr(config, "participation_gate_context_max", 40))


def _channels_max() -> int:
    return max(1, int(getattr(config, "participation_activity_lru_max", 1024)))


class GateContextRing:
    """Per-channel bounded rings of recent messages, with an LRU over channels.

    Synchronous throughout and single-threaded by construction: every writer runs on the event
    loop, with no await inside any method here."""

    def __init__(self) -> None:
        # channel -> ts -> entry. The outer OrderedDict carries the channel LRU order; the inner
        # dict is unordered (eviction picks the numerically oldest ts, not the first inserted).
        self._channels: "OrderedDict[str, Dict[str, GateContextEntry]]" = OrderedDict()

    # -- writers ---------------------------------------------------------------------------

    def record(self, channel_id: Optional[str], entry: GateContextEntry, *,
               from_mention: bool = False) -> bool:
        """Insert or update one message, deduplicated on (channel, ts). Returns True when the
        ring changed.

        Latest edit wins: an entry is replaced unless the one held carries a NEWER edit. An
        app_mention delivery is a twin of a message event, so it may only replace on a strictly
        newer edit — an equal one is the same words, and an older one is the stale original."""
        if not channel_id or not entry.ts or _is_dm(str(channel_id)):
            return False
        capacity = _capacity()
        if capacity <= 0:
            return False
        channel_id = str(channel_id)
        chan = self._channels.get(channel_id)
        existing = chan.get(entry.ts) if chan is not None else None
        if existing is not None:
            if from_mention:
                if not is_newer(entry.version, existing.version):
                    return False
            elif is_newer(existing.version, entry.version):
                return False
        if chan is None:
            chan = self._channels[channel_id] = {}
        self._channels.move_to_end(channel_id)
        chan[entry.ts] = entry
        while len(chan) > capacity:
            oldest = min(chan, key=ts_key)
            chan.pop(oldest, None)
        while len(self._channels) > _channels_max():
            self._channels.popitem(last=False)
        return True

    def replace_text(self, channel_id: Optional[str], ts: Optional[str], text: str) -> bool:
        """Our own edit landed: the held entry's words change, nothing else does. A message the
        ring does not hold is left alone — an edit says nothing about which thread it is in."""
        if not channel_id or not ts:
            return False
        chan = self._channels.get(str(channel_id))
        existing = chan.get(str(ts)) if chan is not None else None
        if chan is None or existing is None:
            return False
        chan[str(ts)] = replace(existing, text=text or "")
        return True

    def remove(self, channel_id: Optional[str], ts: Optional[str]) -> bool:
        """A deletion or a tombstone: the message is gone from the room, so it is gone here."""
        if not channel_id or not ts:
            return False
        chan = self._channels.get(str(channel_id))
        if chan is None:
            return False
        removed = chan.pop(str(ts), None) is not None
        if not chan:
            self._channels.pop(str(channel_id), None)
        return removed

    # -- reader ----------------------------------------------------------------------------

    def recent_for(self, channel_id: Optional[str], *, before_ts: Optional[str],
                   thread_root_ts: Optional[str] = None,
                   limit: int) -> List[GateContextEntry]:
        """The recent messages the responder's stream would show before `before_ts`.

        Selection and ORDER are the wake-gate A/B harness's, which is what the shipped numbers
        were measured on (burst follow-ups §A4):

        Top-level message (no `thread_root_ts`, or one equal to `before_ts`): the newest `limit`
        preceding channel posts (a thread broadcast counts as one), oldest first.

        Thread reply: the thread — its root when the ring still holds it, then its preceding
        replies, oldest first. When the thread alone fills `limit` it is the newest `limit` of
        it (an old root falls away like any older message). Otherwise the newest preceding
        channel posts fill the room left, and come FIRST, ahead of the thread — not interleaved
        by time.

        Strictly older than `before_ts`, at most `limit` entries — and only what the bounded
        ring holds: an evicted root or a cold ring just means less context."""
        if not channel_id or limit <= 0 or before_ts is None:
            return []
        chan = self._channels.get(str(channel_id))
        if not chan:
            return []
        older = [e for e in chan.values() if is_newer(before_ts, e.ts)]
        top = sorted((e for e in older if not e.is_thread_reply or e.is_broadcast),
                     key=lambda e: ts_key(e.ts))
        root = str(thread_root_ts) if thread_root_ts else None
        if root is None or ts_key(root) == ts_key(before_ts):
            return top[-limit:]
        root_key = ts_key(root)
        thread = sorted((e for e in older
                         if ts_key(e.ts) == root_key
                         or (e.is_thread_reply and ts_key(e.thread_ts) == root_key)),
                        key=lambda e: ts_key(e.ts))
        if len(thread) >= limit:
            return thread[-limit:]
        in_thread = {e.ts for e in thread}
        fill = [e for e in top if e.ts not in in_thread]
        return fill[-(limit - len(thread)):] + thread

    def get(self, channel_id: Optional[str], ts: Optional[str]) -> Optional[GateContextEntry]:
        """One held message by (channel, ts), or None — the in-flight line's trigger text."""
        if not channel_id or not ts:
            return None
        chan = self._channels.get(str(channel_id))
        return chan.get(str(ts)) if chan is not None else None

    def entries(self, channel_id: Optional[str]) -> List[GateContextEntry]:
        """Test/diagnostic view: one channel's ring, oldest first."""
        chan = self._channels.get(str(channel_id or ""))
        if not chan:
            return []
        return sorted(chan.values(), key=lambda e: ts_key(e.ts))

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    def reset(self) -> None:
        """Test seam."""
        self._channels.clear()


# THE process-wide ring. Module-level like `slack_client.actor_tail.actor_tail` and
# `slack_client.admission_watermark`: its writers are the raw Slack listeners and the transport,
# which share no object a turn could hand them, and its reader is the gate.
ring = GateContextRing()


def record(channel_id: Optional[str], entry: GateContextEntry, *,
           from_mention: bool = False) -> bool:
    return ring.record(channel_id, entry, from_mention=from_mention)


def replace_text(channel_id: Optional[str], ts: Optional[str], text: str) -> bool:
    return ring.replace_text(channel_id, ts, text)


def remove(channel_id: Optional[str], ts: Optional[str]) -> bool:
    return ring.remove(channel_id, ts)


def recent_for(channel_id: Optional[str], *, before_ts: Optional[str],
               thread_root_ts: Optional[str] = None, limit: int) -> List[GateContextEntry]:
    return ring.recent_for(channel_id, before_ts=before_ts, thread_root_ts=thread_root_ts,
                           limit=limit)


def get(channel_id: Optional[str], ts: Optional[str]) -> Optional[GateContextEntry]:
    return ring.get(channel_id, ts)


def record_assistant_reply(channel_id: Optional[str], thread_ts: Optional[str],
                           ts: Optional[str], text: Optional[str],
                           sender_id: Optional[str] = None) -> bool:
    """One of our own replies, as Slack accepted it. A blank text records nothing: an emptied
    stream (a reaction-only turn's abandoned native stream) said nothing to remember."""
    if not ts or not (text or "").strip():
        return False
    return ring.record(channel_id, GateContextEntry(
        ts=str(ts), thread_ts=str(thread_ts) if thread_ts else None, sender_id=sender_id,
        sender_kind=SENDER_SELF, text=text or ""))


# -- rendering -------------------------------------------------------------------------------

def _cap(text: str, char_cap: int) -> str:
    """Trimmed, then at most `char_cap` characters plus an ellipsis — the harness's cap."""
    text = (text or "").strip()
    return text if len(text) <= char_cap else text[:char_cap].rstrip() + "…"


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _speaker(entry: GateContextEntry) -> str:
    if entry.sender_kind == SENDER_SELF:
        return "the assistant"
    who = entry.sender_name or entry.sender_id or "someone"
    if entry.sender_kind == SENDER_OTHER_BOT:
        who += " (a bot)"
    return who


def _render_entry(entry: GateContextEntry, char_cap: int) -> str:
    """One message as a small block, in the gate's own speaker/timestamp style
    (`openai_client.api.responses._render_wake_source`): who, a UTC stamp from the shared pure
    helper, and whether it was a thread reply or a channel post; then its text; then its files.
    Byte-for-byte the A/B harness's rendering."""
    header = _speaker(entry)
    stamp = render_message_timestamp(entry.ts, "UTC")
    if stamp:
        header += f" {stamp}"
    header += (" — a reply inside a thread" if entry.is_thread_reply
               else " — posted to the channel")
    lines = [header, _cap(entry.text, char_cap) or "(no text)"]
    if entry.attachments:
        lines.append("Attached: " + ", ".join(entry.attachments))
    return "\n".join(lines)


def render_recent_context(entries: Sequence[GateContextEntry], *, char_cap: int) -> str:
    """The labelled recent-context block, or "" when there is nothing to show — so an empty or
    cold ring leaves the gate request exactly as it is without one."""
    if not entries:
        return ""
    return (RECENT_CONTEXT_HEADER + "\n\n"
            + "\n\n".join(_render_entry(e, char_cap) for e in entries))


def render_inflight_line(entry: GateContextEntry, *, char_cap: int) -> str:
    """One turn writing a reply right now, named by the message it is answering:
    `The assistant is currently writing a reply to: [<speaker> <time>] <text>` — the text on one
    line and capped. The harness's line, byte for byte."""
    stamp = render_message_timestamp(entry.ts, "UTC").strip("[]")
    tag = f"{_speaker(entry)} {stamp}".strip()
    return f"{INFLIGHT_PREFIX}[{tag}] {_cap(_one_line(entry.text), char_cap)}"
