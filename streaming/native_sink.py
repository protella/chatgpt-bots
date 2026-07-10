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

from typing import Optional, Tuple

from message_markers import (
    _fence_state,
    continuation_trailer_markdown,
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
                 char_limit: int, logger=None):
        self._client = client
        self.channel = channel_id
        self.thread_ts = thread_ts
        self.char_limit = max(200, char_limit)
        self._log = logger or (lambda msg: None)
        self.session = None
        self.base = ""       # non-buffer prefix of the current part (part prefix + fence reopen)
        self.part = 1
        self.failed = False
        self.finished = False
        self.part_ts: list = []  # ts of every native message created (finished + current)

    @property
    def active(self) -> bool:
        return (not self.failed and not self.finished
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

    async def start(self) -> bool:
        try:
            self.session = self._client.begin_native_stream(self.channel, self.thread_ts)
            ok = await self.session.start()
        except Exception as e:  # noqa: BLE001 - best-effort sink, never fatal
            self._log(f"native coordinator start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
        else:
            self.failed = True
        return not self.failed

    async def update(self, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Append the tail of the current part's cumulative raw text; roll on overflow."""
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
        """Close the current part with the continuation trailer and open the next one."""
        split = find_stream_split(raw_text, self.char_limit, floor=self._sent_raw_len())
        first = raw_text[:split]
        if not await self.session.update(self.base + first):
            self.failed = True
            return False, None
        # Fence continuity: append-only means we close the fence by APPENDING "```",
        # then reopen it (with the language hint) in the next part's base.
        in_block, lang = _fence_state(self.base + first)
        closing = "\n```" if in_block else ""
        finished = await self.session.finish(
            final_text=self.base + first + closing + continuation_trailer_markdown()
        )
        overflow = raw_text[split:].lstrip("\n")
        if not finished:
            self.failed = True
            return False, overflow
        self.part += 1
        self.base = part_prefix_markdown(self.part) + (f"```{lang}\n" if in_block else "")
        try:
            self.session = self._client.begin_native_stream(self.channel, self.thread_ts)
            ok = await self.session.start(self.base + overflow)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator roll-start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
            self._log(f"native stream rolled to part {self.part} (fence reopened: {in_block})")
            return True, overflow
        self.failed = True
        return False, overflow

    async def finalize(self, final_raw: str, suffix: str = "") -> bool:
        """Append any remaining tail (+ suffix, e.g. tools attribution) and stop.

        Returns False if anything failed — the caller should fall back to the legacy
        final-correction edit on ``current_ts``."""
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
            ok = await self.session.finish(final_text=self.base + text + suffix)
            if ok:
                self.finished = True
            else:
                self.failed = True
            return ok
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator finalize error: {e}")
            self.failed = True
            return False

    async def abandon(self) -> None:
        """Stop the stream without appending anything (e.g. reaction-only turns)."""
        if self.session is not None and self.session.active:
            try:
                await self.session.finish()
            except Exception as e:  # noqa: BLE001
                self._log(f"native coordinator abandon error: {e}")
        self.finished = True
