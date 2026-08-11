"""
Native Slack streaming coordinator (Phase G).

Drives chat.startStream/appendStream/stopStream for the streaming handlers via the
client's NativeStreamSession. The legacy sink EDITS one message per tick (display-safe
text, temporary fence closing, loader emoji); the native sink is APPEND-ONLY, so this
coordinator owns the differences:

- ticks append raw cumulative markdown (Slack renders progressively; no loader emoji,
  no temporary fence closing);
- when a part outgrows the per-message limit it "rolls": closes any open code fence,
  appends the continuation trailer, stops the stream, and starts a new native message
  whose base is the part prefix (+ reopened fence). Markers come from message_markers
  in their markdown-flavored forms, which Slack stores as the exact canonical mrkdwn
  shapes the rebuild-side merger (_merge_continuation_history) strips — do NOT inline
  marker strings here (R2 context-pollution bug);
- any failure flips ``failed`` and the caller falls back to the legacy
  update_message_streaming edit loop on ``current_ts``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from message_processor.stale_send_guard import StaleSendSuppressed

from message_markers import (
    _fence_state,
    entity_safe_cut,
    part_prefix_markdown,
)


def find_stream_split(text: str, limit: int, floor: int = 0) -> int:
    """Best split index in (floor, limit] — paragraph, then sentence, then newline,
    then word boundary, then an entity-safe hard cut. ``floor`` is the number of
    characters an append-only sink has already sent (can't be unsent)."""
    floor = max(0, min(floor, len(text)))
    if limit <= floor:
        return floor
    search_start = max(floor, limit - 500)
    for probe, offset in (("\n\n", 2), (". ", 2), ("\n", 1), (" ", 1)):
        idx = text.rfind(probe, search_start, limit)
        if idx > floor:
            return idx + offset
    return max(entity_safe_cut(text, limit), floor) or limit


class NativeStreamCoordinator:
    """Multi-part native streaming with shared continuation markers.

    Usage: ``start()`` once on first content; ``update(raw)`` per tick with the
    CURRENT part's cumulative raw text — returns ``(ok, overflow)`` where a non-None
    overflow means the part rolled and the caller must reset its buffer to exactly
    that text; ``finalize(raw, suffix)`` at the end appends the tail (+ attribution
    suffix) and stops the stream.
    """

    def __init__(self, client, channel_id: str, thread_ts: Optional[str],
                 char_limit: int, logger=None, user_id: Optional[str] = None,
                 receipts=None):
        self._client = client
        # chat.startStream MINTS a message, so every part is a durable post this turn owns.
        # Registered as it is minted, never at lease-commit time: the lease says the turn may
        # speak, the receipt says the room has words it will keep (spec §5).
        self._receipts = receipts
        self.channel = channel_id
        self.thread_ts = thread_ts
        # Triggering user's id — chat.startStream needs it as recipient_user_id for
        # channel streaming; forwarded to begin_native_stream on start() and roll.
        self.user_id = user_id
        self.char_limit = max(200, char_limit)
        self._log = logger or (lambda msg: None)
        # The client's NativeStreamSession, minted by begin_native_stream; `_client` is duck-typed
        # (real bot or a test double), so the session is too.
        self.session: Any = None
        self.base = ""       # non-buffer prefix of the current part (part prefix + fence reopen)
        self.part = 1
        self.failed = False
        self.finished = False
        self.part_ts: list = []  # ts of every native message created (finished + current)
        # HIDDEN-BUFFER MODE. The stale-send guard refused `start()`, so chat.startStream was
        # never called and this coordinator owns no message at all. It is deliberately NOT
        # `failed`: `failed` means "Slack let us down, fall back to the legacy edit loop", and
        # that fallback would post the very answer the guard refused. Every write path below
        # checks `hidden` first and does nothing, so the generation runs to completion with no
        # visible surface and the finished draft goes to reconsideration instead.
        self.suppression: Optional[Exception] = None

    @property
    def hidden(self) -> bool:
        """Was the first surface refused before Slack was ever called?"""
        return self.suppression is not None

    @property
    def active(self) -> bool:
        return (not self.failed and not self.finished and not self.hidden
                and self.session is not None and self.session.active)

    @property
    def started(self) -> bool:
        return self.session is not None

    @property
    def current_ts(self) -> Optional[str]:
        return self.session.ts if self.session is not None else None

    def _sent_raw_len(self) -> int:
        """Chars of the current part's RAW text already appended (base excluded)."""
        if self.session is None:
            return 0
        return max(0, len(self.session._sent) - len(self.base))

    async def start(self, lease: Any = None) -> bool:
        """`lease` (stale guard): chat.startStream MINTS the reply message, so it is a first
        answer surface and the session checks the lease before making the call."""
        try:
            self.session = self._client.begin_native_stream(self.channel, self.thread_ts, user_id=self.user_id)
            ok = await self.session.start(lease=lease)
        except StaleSendSuppressed as suppressed:
            # Not a sink failure: the conversation moved on and the stream was deliberately
            # never opened. Swallowed here it would look like Slack refusing us, and the caller
            # would quietly fall back to the legacy edit loop and post the stale answer anyway.
            # Remembered rather than merely re-raised: this exact object is the one the
            # reconsideration machinery is identity-bound to, and `hidden` is what keeps every
            # later write path (tool flush, finalize, abandon) from touching Slack.
            self.suppression = suppressed
            raise
        except Exception as e:  # noqa: BLE001 - best-effort sink, never fatal
            self._log(f"native coordinator start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
            await self._note_part(self.session.ts)
        else:
            self.failed = True
        return not self.failed

    async def _note_part(self, ts: str) -> None:
        """Claim one native part. Never raises — the message is already in the room."""
        if self._receipts is None:
            return
        try:
            # Spec §4: every native stream part is the answer itself — assistant_reply.
            await self._receipts.note_post(ts, thread_root_ts=self.thread_ts,
                                           receipt_class="assistant_reply")
        except Exception as e:  # noqa: BLE001
            self._log(f"native part receipt failed: {e}")

    async def update(self, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Append the tail of the current part's cumulative raw text; roll on overflow."""
        if self.hidden:
            # Nothing to append to and nothing to fail: report "not written" WITHOUT flipping
            # `failed`, which is the flag that hands the turn to the legacy edit loop.
            return False, None
        if not self.active:
            self.failed = True
            return False, None
        try:
            if len(raw_text) <= self.char_limit:
                ok = await self.session.update(self.base + raw_text)
                if not ok:
                    self.failed = True
                return ok, None
            return await self._roll(raw_text)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator update error: {e}")
            self.failed = True
            return False, None

    async def _roll(self, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Close the current part and open the next one."""
        split = find_stream_split(raw_text, self.char_limit, floor=self._sent_raw_len())
        first = raw_text[:split]
        if not await self.session.update(self.base + first):
            self.failed = True
            return False, None
        # Fence continuity: append-only means we close the fence by APPENDING "```",
        # then reopen it (with the language hint) in the next part's base.
        in_block, lang = _fence_state(self.base + first)
        closing = "\n```" if in_block else ""
        # No trailing "Continued in next message..." (user directive 2026-07-11): the next
        # part's "Part N (continued)" header marks the seam by itself, and the rebuild
        # merger fires on EITHER side's marker (thread_management merge is OR).
        finished = await self.session.finish(final_text=self.base + first + closing)
        overflow = raw_text[split:].lstrip("\n")
        if not finished:
            self.failed = True
            return False, overflow
        self.part += 1
        self.base = part_prefix_markdown(self.part) + (f"```{lang}\n" if in_block else "")
        try:
            self.session = self._client.begin_native_stream(self.channel, self.thread_ts, user_id=self.user_id)
            ok = await self.session.start(self.base + overflow)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator roll-start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
            await self._note_part(self.session.ts)
            self._log(f"native stream rolled to part {self.part} (fence reopened: {in_block})")
            return True, overflow
        self.failed = True
        return False, overflow

    async def finalize(self, final_raw: str, suffix: str = "", blocks=None) -> bool:
        """Append any remaining tail (+ suffix, e.g. tools attribution) and stop.

        ``blocks`` (e.g. the channel Configure chrome) ride the LAST part's stopStream
        only — intermediate rolled parts close without them, so a multi-part response
        carries the chrome exactly once, at the very end.

        Returns False if anything failed — the caller should fall back to the legacy
        final-correction edit on ``current_ts`` (and post any chrome separately)."""
        if self.hidden:
            # The refused stream has no message to finalize. Returning False here does NOT mean
            # "fall back and post it anyway": the caller's hidden branch owns this turn's
            # delivery, through reconsideration.
            return False
        if self.session is None or self.finished:
            return False
        try:
            text = final_raw
            while self.active and len(text) + len(suffix) > self.char_limit:
                ok, overflow = await self._roll(text)
                if overflow is None or not ok:
                    return False
                text = overflow
            if not self.active:
                return False
            ok = await self.session.finish(final_text=self.base + text + suffix, blocks=blocks)
            if ok:
                self.finished = True
            else:
                self.failed = True
            return ok
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator finalize error: {e}")
            self.failed = True
            return False

    async def abandon(self) -> bool:
        """Stop the stream without appending anything (e.g. reaction-only turns).

        Returns False if the stop call failed (the caller should treat the stream message
        as possibly-lingering and fall back to an explicit delete)."""
        if self.hidden:
            # No stream was ever opened, so there is nothing to stop and nothing lingering.
            self.finished = True
            return True
        ok = True
        if self.session is not None and self.session.active:
            try:
                ok = await self.session.finish()
            except Exception as e:  # noqa: BLE001
                self._log(f"native coordinator abandon error: {e}")
                ok = False
        self.finished = True
        return ok
