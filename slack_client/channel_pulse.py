"""ChannelPulse — per-channel ambient awareness (Phase E).

An in-memory ring buffer of recent channel messages, fed by EVERY channel
message event (including ones the bot ignores and its own posts). Consumers:
the wake classifier / ParticipationEngine (channel context for verdicts) and
the response-context envelope (peripheral vision while replying).

Deliberate design constraints:
- Process-lifetime only. NO DB persistence — this is a peripheral-vision
  cache, not a source of truth (Slack is the transcript; see plan §5b). On
  cold start each channel lazily backfills ONCE via conversations.history.
- Single-asyncio-loop safe: plain dict/deque state, no locks held across
  awaits (the only await is the backfill API call, guarded by a flag set
  before awaiting so concurrent wakes can't double-fetch).
- Deterministic rendering: given the same buffer state, render_envelope()
  returns byte-identical text (cache hygiene; the envelope is volatile and
  therefore always injected at the SUFFIX, never the system prompt).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

TEXT_TRUNCATE = 300
THREAD_LABEL_WORDS = 6


class ChannelPulse:
    def __init__(self, size: int = 30):
        self.size = max(1, int(size))
        self._buffers: Dict[str, deque] = {}
        self._backfilled: set = set()
        # Rolling unprompted-reply timestamps per channel (Phase F consumes this).
        self._bot_replies: Dict[str, deque] = {}
        # channel -> {thread_ts: first-words label} for envelope thread naming.
        self._thread_labels: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ feed

    @staticmethod
    def _is_dm(channel_id: Optional[str]) -> bool:
        return bool(channel_id) and channel_id.startswith("D")

    def record(self, channel_id: str, *, ts: str, thread_ts: Optional[str],
               user_id: Optional[str], display_name: Optional[str],
               sender_type: str, text: str, is_bot: bool) -> None:
        """Feed one message event into the channel's ring. DMs are excluded."""
        if not channel_id or self._is_dm(channel_id) or not ts:
            return
        entry = {
            "ts": ts,
            # Slack sets thread_ts == ts on thread roots; normalize roots/top-level to None
            "thread_ts": thread_ts if (thread_ts and thread_ts != ts) else None,
            "user_id": user_id,
            "display_name": display_name or user_id or ("bot" if is_bot else "unknown"),
            "sender_type": sender_type,
            "text": (text or "")[:TEXT_TRUNCATE],
            "is_bot": is_bot,
        }
        buf = self._buffers.setdefault(channel_id, deque(maxlen=self.size))
        buf.append(entry)
        # Top-level messages label any thread that grows under them.
        if entry["thread_ts"] is None and entry["text"]:
            labels = self._thread_labels.setdefault(channel_id, {})
            labels[ts] = " ".join(entry["text"].split()[:THREAD_LABEL_WORDS])

    async def ensure_backfill(self, channel_id: str, client, bot) -> None:
        """Seed the ring with ONE conversations.history call, once per channel
        per process. `bot` supplies classify_sender/get_username-style helpers."""
        if not channel_id or self._is_dm(channel_id) or channel_id in self._backfilled:
            return
        self._backfilled.add(channel_id)  # set BEFORE awaiting: no concurrent double-fetch
        try:
            result = await client.conversations_history(channel=channel_id, limit=self.size)
            messages = list(reversed(result.get("messages", [])))  # oldest -> newest
        except Exception:
            return  # best-effort; live events keep feeding the ring regardless
        existing = {e["ts"] for e in self._buffers.get(channel_id, ())}
        for m in messages:
            if m.get("subtype") or m.get("ts") in existing:
                continue
            sender_type = bot.classify_sender(m) if hasattr(bot, "classify_sender") else "human"
            uid = m.get("user")
            name = m.get("username")
            if not name and uid and uid in getattr(bot, "user_cache", {}):
                name = bot.user_cache[uid].get("real_name")
            self.record(
                channel_id,
                ts=m.get("ts"),
                thread_ts=m.get("thread_ts"),
                user_id=uid,
                display_name=name,
                sender_type=sender_type,
                text=m.get("text", ""),
                is_bot=sender_type != "human",
            )
        # Backfill arrives out of live order; re-sort the ring by ts once.
        buf = self._buffers.get(channel_id)
        if buf:
            ordered = sorted(buf, key=lambda e: float(e["ts"]))
            buf.clear()
            buf.extend(ordered[-self.size:])

    # ------------------------------------------------ participation stats (Phase F)

    def record_bot_reply(self, channel_id: str, ts: str, unprompted: bool,
                         now: Optional[float] = None) -> None:
        if self._is_dm(channel_id) or not unprompted:
            return
        dq = self._bot_replies.setdefault(channel_id, deque(maxlen=200))
        dq.append(now if now is not None else time.time())

    def unprompted_count_last_hour(self, channel_id: str,
                                   now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        dq = self._bot_replies.get(channel_id)
        if not dq:
            return 0
        cutoff = now - 3600
        return sum(1 for t in dq if t >= cutoff)

    # -------------------------------------------------------------- envelope

    def render_envelope(self, channel_id: str, exclude_thread_ts: Optional[str] = None,
                        max_lines: int = 15) -> str:
        """Deterministic compact rendering of recent channel activity,
        oldest -> newest, EXCLUDING messages that belong to exclude_thread_ts
        (that thread is already the model's full context)."""
        buf = self._buffers.get(channel_id)
        if not buf or max_lines <= 0:
            return ""
        labels = self._thread_labels.get(channel_id, {})
        lines: List[str] = []
        for e in buf:
            root = e["thread_ts"] or e["ts"]
            if exclude_thread_ts and root == exclude_thread_ts:
                continue
            if e["thread_ts"]:
                label = labels.get(e["thread_ts"])
                where = f'in thread "{label}…"' if label else "in a thread"
            else:
                where = "top-level"
            lines.append(f'- {e["display_name"]} ({where}): {e["text"]}')
        if not lines:
            return ""
        lines = lines[-max_lines:]
        return (
            "[Recent channel activity — peripheral context only; reply to the "
            "conversation you were addressed in]\n" + "\n".join(lines)
        )
