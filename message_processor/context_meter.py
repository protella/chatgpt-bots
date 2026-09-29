"""The context meter's carrier (CONTEXT_METER_SPEC §3.2, §3.4).

OpenAI's own input-token count is the only number a size decision is made on (OWNER-A). A
`MeterHook` rides one turn's requests as the optional `meter` kwarg of every main-turn wrapper in
openai_client/api/responses.py and tool_loop.py — never into SDK kwargs or usage dicts — and each
wrapper calls it in one fixed order:

    final kwargs built → await preflight(kwargs, round) → _open_attempt → seq = dispatched(kwargs,
    round) → _safe_api_call → usage(seq, usage)

`preflight` is the only step that can stop a request, and it runs before any attempt is opened,
so an overflow it finds leaves no ledger row and sends nothing. `dispatched` starts the PARALLEL
count (owner: measured "all along", never in the reply's critical path) and `usage` records what
the response itself reported. Both land in `ThreadState.record_measure`, whose ordering rules keep
a slow count from overwriting a newer truth.

A DETACHED hook (`thread_state=None`: research build rounds, reconsideration, utility calls) never
writes a thread's meter and never preflights; its parallel count is logged only.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from config import config
from logger import setup_logger

logger = setup_logger(name="slack_bot.ContextMeter")


class ContextOverLimit(Exception):
    """The request is bigger than the model's usable window.

    Raised by preflight (`tokens` = the awaited count) or by a wrapper converting the API's own
    TOKEN context-length 400 (`tokens=None`). Either way it carries the wrapper's FINAL kwargs,
    so recovery can count the failing request when it has no measure of its own (R3-5), and the
    round it happened on, which decides whether recovery is allowed at all (§3.5)."""

    def __init__(self, tokens: Optional[int], limit: int, *,
                 kwargs: Optional[Dict[str, Any]] = None, round_index: int = 0,
                 source: str = "preflight", recoverable: Optional[bool] = None):
        self.tokens = tokens
        self.limit = limit
        self.kwargs = kwargs
        self.round_index = round_index
        self.source = source
        # Was the TURN still before its first completed request when this happened? Stamped by
        # the hook from turn-scoped state, so a loop re-entry's local round 0 cannot reopen it.
        self.recoverable = (round_index == 0) if recoverable is None else bool(recoverable)
        # The measure recovery apportions against — `tokens` when known, filled in by recovery
        # (R3-5) when not.
        self.measure: Optional[int] = tokens
        shown = f"{tokens:,}" if tokens is not None else "unknown"
        super().__init__(f"context over limit ({source}): {shown} > {limit:,} tokens")


class ContextIrreducible(Exception):
    """Recovery proved the overflow cannot be compacted away: what is left is the turn's own
    content. The one size outcome with its own copy (base.py's "Message Too Long")."""


def is_context_length_error(error: BaseException) -> bool:
    """The API's TOKEN context-window rejection — and only that. A per-field length cap or an
    HTTP 413 is a different failure and never reaches compaction (C-12)."""
    text = str(error).lower()
    return ("context_length_exceeded" in text
            or "maximum context length" in text
            or "context window" in text
            or getattr(error, "code", None) == "context_length_exceeded")


def usable_limit(model: Optional[str]) -> int:
    """The model's usable input window — the one resolver every size decision reads."""
    return config.get_model_token_limit(model or config.gpt_model)


class MeterHook:
    """One turn's (or one detached caller's) view of the context meter.

    `client` is the OpenAIClient (`count_input_tokens`). `schedule` is the processor's
    `_schedule_async_call` — strong reference, logged failure — and carries the parallel counts;
    without one, no parallel count runs. `heavy` is the handler's flag for a turn that carries
    attachments, documents, images or a batched cohort (preflight rule b). `on_threshold(model)`
    fires once the turn is finished for an accepted measure at or over the cleanup threshold
    (§3.7) — immediately for a measure that lands after `finish()`.
    """

    def __init__(self, *, client: Any,
                 schedule: Optional[Callable[[Awaitable[Any]], Any]] = None,
                 thread_state: Any = None, key: str = "",
                 heavy: bool = False,
                 on_threshold: Optional[Callable[[Optional[str]], None]] = None):
        self.client = client
        self.schedule = schedule
        self.thread_state = thread_state
        self.key = key or "(detached)"
        self.heavy = heavy
        self.on_threshold = on_threshold
        self._local_seq = 0
        self._dispatches: Dict[int, Tuple[int, Optional[str]]] = {}
        # The count preflight awaited, held for the dispatch it measured (by identity), so the
        # same request is not counted twice.
        self._preflight: Optional[Tuple[Dict[str, Any], Any]] = None
        self._turn_open = True
        self._threshold_model: Optional[str] = None
        self._threshold_pending = False
        # TURN-scoped (the hook lives on the turn): once any request of this turn has completed —
        # and with it any tool round or effect that followed — an overflow is no longer the
        # turn's first request, whatever round a re-entered loop thinks it is on (§3.5).
        self.effects_committed = False

    @property
    def detached(self) -> bool:
        return self.thread_state is None

    # ------------------------------------------------------------------ the three calls

    async def preflight(self, create_kwargs: Dict[str, Any], round_index: int) -> None:
        """Round 0 only, and awaited only when the margin may be gone (§3.4 rules a–d).

        Raises `ContextOverLimit` for an awaited count over the limit. An unknown or incomplete
        count under the limit proceeds: best effort, and the API is the final judge."""
        if self.detached or round_index != 0 or self.effects_committed:
            return
        model = create_kwargs.get("model")
        limit = usable_limit(model)
        current = self.thread_state.current_measure(model)
        if current is None:
            reason = "no measure"
        elif self.heavy:
            reason = "attachments"
        elif current[0] >= limit * config.token_cleanup_threshold:
            reason = "over threshold"
        elif not current[1]:
            reason = "incomplete measure"
        else:
            return
        result = await self.client.count_input_tokens(create_kwargs)
        self._preflight = (create_kwargs, result)
        logger.debug(f"Context meter {self.key}: preflight count ({reason}) = {result.tokens}")
        if result.tokens is not None and result.tokens > limit:
            raise ContextOverLimit(result.tokens, limit, kwargs=create_kwargs,
                                   round_index=round_index, source="preflight",
                                   recoverable=True)

    def dispatched(self, create_kwargs: Dict[str, Any], round_index: int) -> int:
        """Register a request that is about to be sent; start its parallel count. Returns the
        dispatch seq that `usage` must be called with."""
        model = create_kwargs.get("model")
        if self.detached:
            self._local_seq += 1
            seq, generation = self._local_seq, 0
        else:
            seq = self.thread_state.allocate_dispatch_seq()
            generation = self.thread_state.meter_generation
        self._dispatches[seq] = (generation, model)
        pre, self._preflight = self._preflight, None
        if pre is not None and pre[0] is create_kwargs and pre[1].tokens is not None:
            self._accept(seq, pre[1].tokens, pre[1].complete, "count")
            return seq
        if self.schedule is not None:
            from openai_client.api.token_count import count_body
            try:
                self.schedule(self._parallel_count(count_body(create_kwargs), seq))
            except Exception as e:  # noqa: BLE001 — the meter never costs a request
                logger.warning(f"Context meter {self.key}: parallel count not scheduled: {e}")
        return seq

    def usage(self, seq: int, usage_dict: Optional[Dict[str, Any]]) -> None:
        """The response's own `input_tokens` for dispatch `seq`. A missing usage records nothing."""
        if not usage_dict:
            return
        tokens = usage_dict.get("input_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            return
        self._accept(seq, tokens, True, "usage")

    def overflow_from(self, error: BaseException, create_kwargs: Dict[str, Any],
                      round_index: int) -> Optional[ContextOverLimit]:
        """A wrapper's real TOKEN context-length 400, re-expressed as `ContextOverLimit` carrying
        the final kwargs (R3-5). None for any other error, and always None on a detached hook —
        nothing above a detached caller runs this turn's recovery."""
        if self.detached:
            return None
        if isinstance(error, ContextOverLimit):
            return error                     # already the meter's (a retry's own preflight)
        if not is_context_length_error(error):
            return None
        return ContextOverLimit(None, usable_limit(create_kwargs.get("model")),
                                kwargs=create_kwargs, round_index=round_index, source="api",
                                recoverable=round_index == 0 and not self.effects_committed)

    def request_completed(self) -> None:
        """A request of this turn came back whole: from here on an overflow is not recoverable."""
        self.effects_committed = True

    def finish(self) -> None:
        """The turn is over. A measure already at the threshold triggers now; later ones will
        trigger as they land."""
        self._turn_open = False
        if self._threshold_pending:
            self._threshold_pending = False
            self._fire_threshold(self._threshold_model)

    # ------------------------------------------------------------------ internals

    async def _parallel_count(self, body: Dict[str, Any], seq: int) -> None:
        try:
            result = await self.client.count_input_tokens(body)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Context meter {self.key}: parallel count failed: {e}")
            return
        if result.tokens is None:
            return
        self._accept(seq, result.tokens, result.complete, "count")

    def _accept(self, seq: int, tokens: int, complete: bool, source: str) -> None:
        generation, model = self._dispatches.get(seq, (-1, None))
        limit = usable_limit(model)
        if not self.detached:
            if not self.thread_state.record_measure(tokens, complete, seq, generation, model,
                                                    source):
                return
        pct = (tokens / limit) if limit else 0.0
        suffix = "" if complete else ", incomplete"
        logger.info(f"Context meter {self.key}: {tokens:,} tokens ({source}{suffix}) of "
                    f"{limit:,} usable ({pct:.1%})")
        if self.detached or self.on_threshold is None:
            return
        if tokens >= limit * config.token_cleanup_threshold:
            if self._turn_open:
                self._threshold_pending = True
                self._threshold_model = model
            else:
                self._fire_threshold(model)
        elif self._turn_open:
            # A newer accepted measure below the threshold supersedes an earlier one above it.
            self._threshold_pending = False

    def _fire_threshold(self, model: Optional[str]) -> None:
        callback = self.on_threshold
        if callback is None:
            return
        try:
            callback(model)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Context meter {self.key}: threshold trigger failed: {e}")


__all__ = ["ContextIrreducible", "ContextOverLimit", "MeterHook", "is_context_length_error",
           "usable_limit"]
