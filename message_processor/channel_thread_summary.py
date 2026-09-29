"""The channel ORIGIN summary (CONTEXT_METER_SPEC §3.6, R3-1..R3-5).

A long channel thread outgrows the window long before the periphery does. OWNER-D/E: the bot keeps
the room's recent activity at its normal depth and summarizes the THREAD instead — the earlier
part of the origin thread is replaced, inside the post-breakpoint origin block, by one synthetic
summary item. Slack stays the transcript; the summary is a derived artifact stored in the existing
`thread_summaries` row (key `channel_id:thread_ts`), which a channel row marks by carrying a
`source_fingerprint` and never taking addenda.

THE ROW IS VALID ONLY FOR THE MESSAGES IT DESCRIBES. Its fingerprint is taken over the stable
fields of the root and every covered message, so an edit, a delete or an insertion inside the
covered span makes the row describe a thread that no longer exists; the pin-time judgment
(`judge_row`) then renders the thread in full and the turn path removes the row.

TWO WRITERS, ONE SUMMARY-ONLY LOCK. A background compaction (after a turn crossed the threshold)
and a foreground one (a turn's own overflow recovery, on its held turn lock) may both run. Each
snapshots the thread's `summary_generation`; a commit, and an invalidation, happen under a
per-thread asyncio.Lock that is never a turn lock, and a commit lands only if the generation is
unchanged — so the later writer wins only when nothing moved in between, and a `stale` result is
harmless.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from config import clamp_effort, config
from logger import setup_logger
from message_processor.channel_stream import (ChannelStream, OriginSummary, PageCounts,
                                              SidecarPin, StreamBuildResult, _emit_stream_render,
                                              _resolve_self_role, iso_minute, serialize_stream)
from message_processor.context_meter import usable_limit
from slack_client.normalizer import NormalizedMessage, TimestampError, parse_ts, sanitize_name

logger = setup_logger(name="slack_bot.ChannelThreadSummary")

FINGERPRINT_VERSION = "v1"

OUTCOME_COMMITTED = "committed"
OUTCOME_SUMMARY_FAILED = "summary_failed"
OUTCOME_IRREDUCIBLE = "irreducible"
OUTCOME_STALE = "stale"


def thread_key(channel_id: str, root_ts: str) -> str:
    return f"{channel_id}:{root_ts}"


# ------------------------------------------------------------------ pure: eligibility + coverage

def eligible_origin(messages: Sequence[NormalizedMessage], *, sidecars: SidecarPin,
                    receipt_feature_epoch_ts: Optional[str],
                    chrome_ts: FrozenSet[str]) -> List[NormalizedMessage]:
    """The origin messages a summary may speak for, ts-ordered.

    The PURE role rule decides (C-19): a foreign message always qualifies, one of ours only when
    `_resolve_self_role` gives it a role. The serializer's receipt-membership recording is never
    run for these — a candidate is not a render."""
    out = [m for m in messages
           if m.sender_type != "self"
           or _resolve_self_role(m, receipts=sidecars,
                                 receipt_feature_epoch_ts=receipt_feature_epoch_ts,
                                 chrome_ts=chrome_ts) is not None]
    return sorted(out, key=lambda m: parse_ts(m.ts))


def protected_set(eligible: Sequence[NormalizedMessage], supplied: FrozenSet[str],
                  newest: int) -> FrozenSet[str]:
    """What no summary may cover: the caller's trigger ts plus the newest `newest` eligible
    messages (TOKEN_TRIM_MESSAGE_COUNT)."""
    tail = [m.ts for m in eligible][-newest:] if newest > 0 else []
    return frozenset(supplied) | frozenset(tail)


def covered_messages(eligible: Sequence[NormalizedMessage], root_ts: str,
                     boundary_ts: str) -> List[NormalizedMessage]:
    """Eligible messages strictly after the root and at/before the boundary. The root is never
    covered: it is summarizer context and renders verbatim."""
    root, boundary = parse_ts(root_ts), parse_ts(boundary_ts)
    return [m for m in eligible
            if m.ts != root_ts and root < parse_ts(m.ts) <= boundary]


def fingerprint(root: Optional[NormalizedMessage],
                covered: Sequence[NormalizedMessage]) -> str:
    """`v1:` + sha256 over the STABLE fields of the root and every covered message, ts order.
    Reactions, sidecar evidence and resolved names are deliberately absent — they move without
    the conversation moving."""
    subjects = ([root] if root is not None else []) + sorted(covered,
                                                             key=lambda m: parse_ts(m.ts))
    payload = [[m.ts, m.edited_ts or "", m.sender_id, m.text, [f.id for f in m.files]]
               for m in subjects]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{FINGERPRINT_VERSION}:{hashlib.sha256(blob).hexdigest()}"


def canonical_line(message: NormalizedMessage, actor_names: Mapping[str, str]) -> str:
    """One source message as the summarizer reads it — stable fields only."""
    sender = message.sender_id or "unknown"
    name = sanitize_name(actor_names.get(sender, "")) or sender
    line = f"[{iso_minute(message.ts)}] {name}: {message.text}"
    names = [f.name for f in message.files if f.name]
    if names:
        line += f" (files: {', '.join(names)})"
    return line


@dataclass(frozen=True)
class Judgement:
    """A stored row, judged against one fetched origin: the summary to freeze onto the pin, or
    `invalid` with the (boundary_ts, source_fingerprint) that was judged — the identity R3-3's
    delete re-checks before removing anything."""
    summary: Optional[OriginSummary] = None
    invalid: bool = False
    judged: Optional[Tuple[str, Optional[str]]] = None


def judge_row(row: Optional[Mapping[str, Any]], *, origin_root_ts: str,
              origin_snapshot: Sequence[NormalizedMessage], eligible: Sequence[NormalizedMessage],
              h: str, protected: FrozenSet[str]) -> Judgement:
    """PURE. Valid iff boundary ≤ H, no protected ts ≤ boundary, and the fingerprint recomputed
    from the fetched origin matches (C-7, C-27). Anything else — a DM-style row with no
    fingerprint, an unparseable boundary — is invalid, never guessed at.

    The ROOT is exempt from the protected test: it is never covered by construction, so a
    root-triggered turn (whose protected trigger IS the root) must not invalidate every row."""
    if not row:
        return Judgement()
    boundary = str(row.get("boundary_ts") or "")
    stored = row.get("source_fingerprint")
    judged = (boundary, stored)
    try:
        edge = parse_ts(boundary)
        if edge > parse_ts(h) or any(parse_ts(ts) <= edge for ts in protected
                                     if ts != origin_root_ts):
            return Judgement(invalid=True, judged=judged)
    except TimestampError:
        return Judgement(invalid=True, judged=judged)
    covered = covered_messages(eligible, origin_root_ts, boundary)
    root = next((m for m in origin_snapshot if m.ts == origin_root_ts), None)
    if not covered or stored != fingerprint(root, covered):
        return Judgement(invalid=True, judged=judged)
    later = [m.ts for m in eligible if m.ts != origin_root_ts and parse_ts(m.ts) > edge]
    return Judgement(summary=OriginSummary(
        text=str(row.get("summary_text") or ""), boundary_ts=boundary,
        covered_ts=frozenset(m.ts for m in covered),
        first_uncovered_ts=later[0] if later else None, source_fingerprint=stored))


async def load_judgement(db: Any, *, channel_id: str, origin_root_ts: Optional[str],
                         origin_snapshot: Sequence[NormalizedMessage], eligible:
                         Sequence[NormalizedMessage], h: str,
                         protected: FrozenSet[str]) -> Judgement:
    """READ-ONLY (R3-2): load the row and judge it. Never writes, never deletes — the caller that
    owns the turn acts on `invalid`. A read failure is "no summary", logged."""
    reader = getattr(db, "get_thread_summary_async", None)
    if not origin_root_ts or reader is None:
        return Judgement()
    try:
        row = await reader(thread_key(channel_id, origin_root_ts))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — the thread renders in full without it
        logger.warning(f"thread summary read failed for {channel_id}:{origin_root_ts}: {e}")
        return Judgement()
    if not isinstance(row, Mapping):
        return Judgement()
    return judge_row(row, origin_root_ts=origin_root_ts, origin_snapshot=origin_snapshot,
                     eligible=eligible, h=h, protected=protected)


# ------------------------------------------------------------------ generation + locks (R3-3)

_locks: Dict[str, asyncio.Lock] = {}
_generations: Dict[str, int] = {}
# {thread_key: (summary_generation at launch, running task)} — background coalescing.
_running: Dict[str, Tuple[int, "asyncio.Task[Any]"]] = {}


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = _locks[key] = asyncio.Lock()
    return lock


def summary_generation(key: str) -> int:
    return _generations.get(key, 0)


def _bump(key: str) -> None:
    _generations[key] = summary_generation(key) + 1


async def invalidate_if_unchanged(db: Any, key: str,
                                  judged: Optional[Tuple[str, Optional[str]]]) -> bool:
    """R3-3 invalidation: under the summary lock, re-read the row and delete it only if its
    boundary and fingerprint still equal what was judged invalid; then bump the generation.
    A row that changed since (a newer commit) is left alone. Returns whether it deleted."""
    if db is None or judged is None:
        return False
    async with _lock_for(key):
        row = await db.get_thread_summary_async(key)
        if not row or (str(row.get("boundary_ts") or ""), row.get("source_fingerprint")) != judged:
            return False
        await db.delete_thread_summary_async(key)
        _bump(key)
    logger.info(f"Thread summary for {key} no longer matches its thread — deleted")
    return True


# ------------------------------------------------------------------ compaction


@dataclass(frozen=True)
class CompactionResult:
    outcome: str
    row: Optional[Dict[str, Any]] = None


def _origin_request_items(stream: ChannelStream) -> List[Dict[str, Any]]:
    return [{"role": item["role"], "content": item["content"]}
            for item in stream.origin_input_items()]


def _summarizer_params(root: Optional[NormalizedMessage], prior: Optional[str],
                       span: Sequence[NormalizedMessage],
                       actor_names: Mapping[str, str]) -> Dict[str, Any]:
    from message_processor.prompts import CHANNEL_THREAD_SUMMARY_PROMPT
    from openai_client.base import _build_request_params

    parts = []
    if root is not None:
        parts.append("THREAD OPENING MESSAGE:\n" + canonical_line(root, actor_names))
    if prior:
        parts.append("EXISTING SUMMARY OF EARLIER MESSAGES:\n" + prior)
    parts.append("MESSAGES TO FOLD IN:\n"
                 + "\n".join(canonical_line(m, actor_names) for m in span))
    return _build_request_params(
        model=config.utility_model,
        input_items=[{"role": "user", "content": "\n\n".join(parts)}],
        system_prompt=CHANNEL_THREAD_SUMMARY_PROMPT,
        max_output_tokens=config.default_max_tokens,
        reasoning_effort=clamp_effort(config.utility_model, config.utility_reasoning_effort),
        verbosity=config.utility_verbosity,
    )


def nominate_span(candidates: Sequence[NormalizedMessage], *, protected: FrozenSet[str],
                  rendered_bytes: Mapping[str, int], tokens_per_byte: float,
                  needed: float) -> List[NormalizedMessage]:
    """C-3: oldest-first, apportioning the origin's official count by rendered bytes, until the
    apportioned sum reaches `needed` or the next candidate is protected."""
    span: List[NormalizedMessage] = []
    shed = 0.0
    for message in candidates:
        if message.ts in protected:
            break
        span.append(message)
        shed += rendered_bytes.get(message.ts, 0) * tokens_per_byte
        if shed >= needed:
            break
    return span


async def compact_origin(*, openai_client: Any, db: Any, stream: ChannelStream,
                         thread_state: Any, measure: Optional[int],
                         model: Optional[str]) -> CompactionResult:
    """One compaction of a channel origin thread. Outcomes: `committed` (row written, meter
    invalidated) | `summary_failed` (nothing written) | `irreducible` (no candidate left: every
    eligible message after the root is protected) | `stale` (the generation moved)."""
    pinned = stream.pinned
    root_ts = pinned.origin_root_ts
    if not root_ts or db is None:
        return CompactionResult(OUTCOME_SUMMARY_FAILED)
    key = thread_key(pinned.channel_id, root_ts)
    snapshot = summary_generation(key)

    eligible = eligible_origin(pinned.origin_snapshot, sidecars=pinned.sidecars,
                               receipt_feature_epoch_ts=pinned.receipt_feature_epoch_ts,
                               chrome_ts=frozenset(pinned.chrome_ts))
    summary = pinned.origin_summary
    root_key = parse_ts(root_ts)
    after = parse_ts(summary.boundary_ts) if summary is not None else root_key
    candidates = [m for m in eligible if m.ts != root_ts and parse_ts(m.ts) > after]
    if not candidates or candidates[0].ts in pinned.protected_ts:
        return CompactionResult(OUTCOME_IRREDUCIBLE)

    limit = usable_limit(model)
    target = int(limit * config.token_compaction_target)
    needed = (measure if measure is not None else limit) - target
    if needed <= 0:
        return CompactionResult(OUTCOME_SUMMARY_FAILED)

    # The origin block's own official count, apportioned over its rendered bytes.
    origin_items = _origin_request_items(stream)
    counted = await openai_client.count_input_tokens({"model": model, "input": origin_items})
    total_bytes = sum(len(str(i["content"]).encode("utf-8")) for i in origin_items)
    if counted.tokens is None or total_bytes <= 0:
        logger.warning(f"Origin count unavailable for {key} — no summary")
        return CompactionResult(OUTCOME_SUMMARY_FAILED)
    rendered = {str(item.metadata.get("ts")): len(item.content.encode("utf-8"))
                for item in stream.origin_items if item.metadata.get("ts")}
    span = nominate_span(candidates, protected=pinned.protected_ts, rendered_bytes=rendered,
                         tokens_per_byte=counted.tokens / total_bytes, needed=needed)

    root = next((m for m in pinned.origin_snapshot if m.ts == root_ts), None)
    # The root's CONTENT reaches the summarizer only if the pure role rule admits it: an
    # excluded root of ours (in flight, chrome, unregistered) must not return as user context
    # through the summary. Its identity still enters the fingerprint below.
    root_text_source = root if root is not None and any(m.ts == root_ts for m in eligible) \
        else None
    prior = summary.text if summary is not None else None
    actor_names = pinned.actor_names
    utility_limit = usable_limit(config.utility_model)
    while True:
        params = _summarizer_params(root_text_source, prior, span, actor_names)
        sized = await openai_client.count_input_tokens(params)
        if sized.tokens is None:
            logger.warning(f"Summarizer request for {key} could not be counted — no summary")
            return CompactionResult(OUTCOME_SUMMARY_FAILED)
        if sized.tokens <= utility_limit:
            break
        if len(span) <= 1:
            logger.warning(f"Summarizer request for {key} is over the utility window even for "
                           f"one message ({sized.tokens:,} > {utility_limit:,}) — no summary")
            return CompactionResult(OUTCOME_SUMMARY_FAILED)
        span = span[:len(span) // 2]            # the oldest half is kept

    try:
        response = await openai_client._safe_api_call(
            openai_client.client.responses.create, operation_type="utility_call", **params)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — a failed summary leaves the thread in full
        logger.warning(f"Thread summarization failed for {key}: {e}")
        return CompactionResult(OUTCOME_SUMMARY_FAILED)
    text = getattr(response, "output_text", None)
    if getattr(response, "status", None) != "completed" or not isinstance(text, str) \
            or not text.strip():
        logger.warning(f"Thread summarization for {key} returned no usable summary "
                       f"(status={getattr(response, 'status', None)})")
        return CompactionResult(OUTCOME_SUMMARY_FAILED)

    boundary = span[-1].ts
    covered = covered_messages(eligible, root_ts, boundary)
    row: Dict[str, Any] = {"summary_text": text.strip(), "boundary_ts": boundary,
                           "source_fingerprint": fingerprint(root, covered)}
    # The candidate must pass the SAME judgment every later pin will apply, or it is a row this
    # turn could never install and no later turn could keep.
    if judge_row(row, origin_root_ts=root_ts, origin_snapshot=pinned.origin_snapshot,
                 eligible=eligible, h=pinned.H, protected=pinned.protected_ts).summary is None:
        logger.warning(f"Thread summary candidate for {key} does not judge valid — not written")
        return CompactionResult(OUTCOME_SUMMARY_FAILED)
    # The row this pin was judged against: a commit from this pin is valid only while that row —
    # or its absence — is still what is stored. A newer summary written from a fresher pin is
    # never replaced by one built from an older transcript (no backward boundary moves).
    expected = ((summary.boundary_ts, summary.source_fingerprint) if summary is not None
                else None)
    try:
        async with _lock_for(key):
            stored = await db.get_thread_summary_async(key)
            identity = ((str(stored.get("boundary_ts") or ""), stored.get("source_fingerprint"))
                        if stored else None)
            if summary_generation(key) != snapshot or identity != expected:
                logger.info(f"Thread summary for {key} is stale (the stored row moved) — "
                            "not written")
                return CompactionResult(OUTCOME_STALE)
            await db.save_thread_summary_async(
                key, row["summary_text"], boundary, [], [],
                source_fingerprint=row["source_fingerprint"])
            _bump(key)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — a failed write leaves the thread in full
        logger.warning(f"Thread summary for {key} could not be written: {e}")
        return CompactionResult(OUTCOME_SUMMARY_FAILED)
    if thread_state is not None:
        thread_state.invalidate_measure()
    logger.info(f"Thread summary for {key} committed: {len(covered)} message(s) through "
                f"{boundary}")
    return CompactionResult(OUTCOME_COMMITTED, row)


def schedule_background(processor: Any, *, stream: ChannelStream, thread_state: Any,
                        model: Optional[str]) -> bool:
    """§3.7: background compaction after a turn, outside any turn lock, DB-row commit only.
    A trigger while one runs for the same thread is dropped (coalesced). Returns whether a task
    was started."""
    root_ts = stream.pinned.origin_root_ts
    if not root_ts:
        return False
    key = thread_key(stream.pinned.channel_id, root_ts)
    held = _running.get(key)
    if held is not None and not held[1].done():
        logger.debug(f"Background summary for {key} already running — trigger coalesced")
        return False
    current = thread_state.current_measure(model) if thread_state is not None else None
    task = processor._schedule_async_call(compact_origin(
        openai_client=processor.openai_client, db=processor.db, stream=stream,
        thread_state=thread_state, measure=current[0] if current is not None else None,
        model=model))
    if isinstance(task, asyncio.Task):
        _running[key] = (summary_generation(key), task)

        def _retire(done: "asyncio.Task[Any]", _key: str = key) -> None:
            held = _running.get(_key)
            if held is not None and held[1] is done:
                del _running[_key]

        task.add_done_callback(_retire)
    return True


# ------------------------------------------------------------------ R3-1: install after commit

def rebuild_with_row(stream: ChannelStream, row: Mapping[str, Any]) -> Optional[ChannelStream]:
    """The committed row frozen onto the SAME origin pin (no Slack refetch, periphery and pinned
    evidence reused), re-serialized. None when the row does not judge valid against it."""
    pinned = stream.pinned
    if not pinned.origin_root_ts:
        return None
    eligible = eligible_origin(pinned.origin_snapshot, sidecars=pinned.sidecars,
                               receipt_feature_epoch_ts=pinned.receipt_feature_epoch_ts,
                               chrome_ts=frozenset(pinned.chrome_ts))
    judgement = judge_row(row, origin_root_ts=pinned.origin_root_ts,
                          origin_snapshot=pinned.origin_snapshot, eligible=eligible,
                          h=pinned.H, protected=pinned.protected_ts)
    if judgement.summary is None:
        return None
    return serialize_stream(pinned.with_origin_summary(judgement.summary))


def install_rebuilt_stream(turn: Any, stream: ChannelStream) -> None:
    """THE one install: the turn's stream and its context move TOGETHER, and the rebuild is
    recorded as the turn's next build (`build_seq + 1`)."""
    from message_processor.channel_request import fresh_turn_context

    ctx = turn.channel_turn_context
    turn.channel_stream = stream
    turn.channel_turn_context = fresh_turn_context(ctx, stream)
    turn.stream_build_seq = int(getattr(turn, "stream_build_seq", 0)) + 1
    _emit_stream_render(
        StreamBuildResult(stream=stream, reselected=False, anchor_advanced=False,
                          pages=PageCounts(history=0, reply=0, origin=0)),
        turn_id=getattr(turn, "turn_id", None), origin_root_ts=stream.pinned.origin_root_ts,
        trigger_ts=getattr(ctx, "trigger_ts", None), build_seq=turn.stream_build_seq)


__all__ = ["CompactionResult", "Judgement", "OUTCOME_COMMITTED", "OUTCOME_IRREDUCIBLE",
           "OUTCOME_STALE", "OUTCOME_SUMMARY_FAILED", "canonical_line", "compact_origin",
           "covered_messages", "eligible_origin", "fingerprint", "install_rebuilt_stream",
           "invalidate_if_unchanged", "judge_row", "load_judgement", "nominate_span",
           "protected_set", "rebuild_with_row", "schedule_background", "summary_generation",
           "thread_key"]
