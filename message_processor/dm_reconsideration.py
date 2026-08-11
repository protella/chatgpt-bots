"""A DM's draft gets reconsidered too — over the DM's own surface.

WHY THIS EXISTS. `reconsideration.py` rebuilds a pure CHANNEL snapshot and re-assembles the
canonical channel request over it. A DM has neither: no channel stream, no
`ChannelTurnContext`, no assembler. So DM suppressions used to rethrow — the draft died
silently, which is exactly the failure the reconsideration runner was built to end. The
2026-08-10 parallel-load incident hit it three times in one burst.

WHAT A DM SNAPSHOT HAS TO BE. Not "rebuild the thread this turn was answering": the message
that suppressed the draft is very often a NEW top-level DM, living under a different thread
root entirely, and a rebuild that cannot see it can never verify it — which fails closed and
drops the draft for a second time. So the snapshot is the DM SURFACE: the recent top-level
timeline across roots (`conversations.history`), plus the origin thread's own replies
(`conversations.replies`), merged in timestamp order.

PURE, in the same sense §4c means it for channels: it reads Slack and writes nothing. No warm
state, no document processing, no summarization, no persistence, no telemetry — none of the
stateful DM history machinery in `thread_management.py` is touched. The one thing it must not
invent is the REQUEST: the instructions and sampling settings come from `DMTurnContext`,
pinned by the handler when it built the original request, so the reconsideration call re-asks
with the turn's own evidence rather than a reconstruction of it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import config
from message_processor.channel_request import estimate_admission
from message_processor.stale_send_guard import ts_key
from openai_client.api.responses import STALE_RECONSIDERATION_RESPONSE_FORMAT
from slack_client.history_fetch import FetchBudget, iter_pages, page_messages


# ------------------------------------------------------------------ the pinned turn context


@dataclass(frozen=True)
class DMTurnContext:
    """What a DM turn cannot re-derive after its request has been sent.

    Deliberately tiny beside `ChannelTurnContext`: a DM reconsideration rebuilds the room from
    Slack, so the only things worth pinning are the ones a rebuild would have to GUESS — the
    exact instructions the turn spoke under, its sampling settings, and the trigger it was
    answering (whose text is quoted back to the model, and whose position in a reconsideration
    snapshot is no longer "the newest message").
    """

    channel_id: str
    trigger_ts: Optional[str]
    origin_root_ts: Optional[str]
    trigger_text: str
    requester_name: str
    instructions: str
    thread_config: Dict[str, Any] = field(default_factory=dict)
    prompt_cache_key: Optional[str] = None


def pin_dm_turn_context(turn: Any, message: Any, *, thread_config: Dict[str, Any],
                        instructions: str,
                        prompt_cache_key: Optional[str] = None) -> None:
    """Record the DM request's evidence on the turn. Idempotent per turn: an MCP retry or a
    non-streaming fallback re-enters the same handler and re-pins the same facts, and the LAST
    request the turn actually sent is the one reconsideration should re-ask under."""
    if turn is None or message is None:
        return
    meta = getattr(message, "metadata", None) or {}
    turn.dm_turn_context = DMTurnContext(
        channel_id=str(getattr(message, "channel_id", "") or ""),
        trigger_ts=meta.get("ts"),
        origin_root_ts=getattr(message, "thread_id", None) or meta.get("ts"),
        trigger_text=getattr(message, "text", "") or "",
        requester_name=str(meta.get("user_real_name") or meta.get("username")
                           or getattr(message, "user_id", None) or "unknown"),
        instructions=instructions or "",
        thread_config=dict(thread_config or {}),
        prompt_cache_key=prompt_cache_key)


# ------------------------------------------------------------------------------ the snapshot


@dataclass(frozen=True)
class DMSnapshotItem:
    """One message of the rebuilt DM surface.

    `metadata` carries exactly the keys the shared reviewed-through extraction reads
    (`reconsideration.reviewed_through_map` / `suppressing_ts_present`), under the same names
    the channel serializer uses — which is what lets one guard-facing implementation serve both
    surfaces instead of two that can drift apart.
    """

    metadata: Dict[str, Any]
    role: str
    content: str


@dataclass(frozen=True)
class DMSurfaceSnapshot:
    """The rebuilt DM surface: top-level timeline + the origin thread, both read-only.

    Named `message_items` / `origin_items` on purpose — the extraction functions in
    `reconsideration.py` take either surface's snapshot and never learn which one they got.
    """

    message_items: Tuple[DMSnapshotItem, ...] = ()
    origin_items: Tuple[DMSnapshotItem, ...] = ()

    def input_items(self) -> List[Dict[str, Any]]:
        """The conversation as API input, oldest first, deduped by ts.

        Our own messages ride as `assistant` and everyone else's as `user`, which is the shape
        a DM request has always had. Each non-self message keeps a `[name ts=…]` header so the
        model can find the trigger the reconsideration item NAMES — a reconsideration snapshot
        is the one stream whose newest message is not the trigger, so the positional convention
        every normal turn relies on is unavailable here.
        """
        merged: Dict[str, DMSnapshotItem] = {}
        for item in (*self.message_items, *self.origin_items):
            ts = item.metadata.get("ts")
            if ts and ts not in merged:
                merged[ts] = item
        ordered = sorted(merged.values(), key=lambda i: ts_key(i.metadata.get("ts")))
        return [{"role": item.role, "content": item.content} for item in ordered]


def _render(client: Any, raw: Dict[str, Any], *, channel_id: str,
            requester_name: str) -> Optional[DMSnapshotItem]:
    """One raw Slack message → one snapshot item, or None when it carries no timestamp (a
    message we cannot place in the timeline is one we cannot let advance a scope either)."""
    ts = raw.get("ts")
    if not ts:
        return None
    from slack_client.normalizer import canonical_sender_id

    classify = getattr(client, "classify_sender", None)
    sender_type = classify(raw) if callable(classify) else "human"
    sender_id = canonical_sender_id(client, raw)
    thread_root = raw.get("thread_ts") or ts
    text = (raw.get("text") or "").strip() or "(no text)"
    if sender_type == "self":
        return DMSnapshotItem(
            metadata={"ts": str(ts), "channel_id": channel_id, "sender_id": sender_id,
                      "sender_type": sender_type, "thread_root_ts": str(thread_root)},
            role="assistant", content=text)
    who = requester_name if sender_type == "human" else (sender_id or "unknown")
    header = f"[{who} ts={ts}]"
    if str(thread_root) != str(ts):
        header = f"[{who} ts={ts} in thread {thread_root}]"
    return DMSnapshotItem(
        metadata={"ts": str(ts), "channel_id": channel_id, "sender_id": sender_id,
                  "sender_type": sender_type, "thread_root_ts": str(thread_root)},
        role="user", content=f"{header} {text}")


def _web(client: Any, name: str) -> Optional[Any]:
    """The raw web method, behind the bot facade or on the client itself (the same lookup
    `channel_stream._web` does; duplicated rather than imported so this module does not pull in
    the channel builder)."""
    app = getattr(client, "app", None)
    web = getattr(app, "client", None) if app is not None else None
    method = getattr(web, name, None)
    if callable(method):
        return method
    method = getattr(client, name, None)
    return method if callable(method) else None


# A user id resolves to the same IM conversation for the life of the workspace, so the lookup is
# cached for the process — the same reasoning the bot's own identity cache runs on. Only
# SUCCESSES are cached: a transient failure must not blind the process until restart.
_IM_CHANNEL_IDS: Dict[str, str] = {}


async def resolve_dm_conversation_id(client: Any, channel_id: str) -> str:
    """The `D…` conversation id to READ from.

    Outbound posting accepts a bare user id (`chat.postMessage` opens the IM for you), so a DM
    turn's channel id is sometimes `U…` — or `W…` on Enterprise Grid. The READ APIs have no such
    shortcut: `conversations.history` on a user id returns `channel_not_found`, which would
    surface as a context-rebuild failure and drop the draft for a second time. `conversations.open`
    is the documented lookup and is idempotent for a conversation that already exists — and this
    one does, because a turn is answering in it.

    Raises when it cannot be resolved: the caller treats that as a context-rebuild failure and
    the suppression stands. Never guesses.
    """
    if not channel_id:
        raise RuntimeError("a DM snapshot needs a conversation id")
    if channel_id.startswith("D"):
        return channel_id
    cached = _IM_CHANNEL_IDS.get(channel_id)
    if cached:
        return cached
    opener = _web(client, "conversations_open")
    if opener is None:
        raise RuntimeError(f"cannot resolve the DM conversation for {channel_id}: "
                           "no conversations.open method on this client")
    response = await opener(users=channel_id)
    resolved = ((response or {}).get("channel") or {}).get("id")
    if not resolved:
        raise RuntimeError(f"conversations.open returned no channel id for {channel_id}")
    _IM_CHANNEL_IDS[channel_id] = str(resolved)
    return str(resolved)


async def build_dm_reconsideration_snapshot(*, client: Any, channel_id: str,
                                            origin_root_ts: Optional[str],
                                            requester_name: str = "unknown",
                                            limit: Optional[int] = None) -> DMSurfaceSnapshot:
    """Read the DM surface as it stands NOW. Writes nothing; any failure propagates and the
    runner classifies it as a context-rebuild failure (§4f) — never as a reason to post.

    The two fetches share ONE absolute deadline, exactly as the channel snapshot's phases do,
    so a slow surface cannot cost the turn two full budgets in series.
    """
    history = _web(client, "conversations_history")
    replies = _web(client, "conversations_replies")
    if history is None:
        raise RuntimeError("no conversations.history method on this client")

    read_id = await resolve_dm_conversation_id(client, channel_id)
    ceiling = int(limit or config.history_tool_max_messages)
    deadline_at = asyncio.get_running_loop().time() + float(config.fetch_retry_total_seconds)
    budget = FetchBudget(deadline_at=deadline_at, page_ceiling=None)

    async def _history() -> List[Dict[str, Any]]:
        # EXACTLY ONE PAGE. The newest `ceiling` messages are the surface being reviewed, and a
        # long-lived DM always hands back a `next_cursor` with them — that is a normal page, not
        # an anomaly. Draining it would read the whole DM back to its first message; capping the
        # pager at one page instead turns the trailing cursor into a fetch ERROR, which the
        # runner classifies as `context_rebuild` and drops the draft over. So: take the first
        # page and stop asking. `iter_pages` validates each page BEFORE yielding it, so the
        # genuine anomalies (a claim of more with no cursor, an empty page claiming more) still
        # raise here.
        async for page in iter_pages(history, channel_id=read_id, limit=ceiling,
                                     budget=budget, label="dm history"):
            return list(page)
        return []

    async def _origin() -> List[Dict[str, Any]]:
        if not origin_root_ts or replies is None:
            return []
        return await page_messages(replies, channel_id=read_id, budget=budget,
                                   label="dm origin thread",
                                   extra_params={"ts": str(origin_root_ts)})

    history_task = asyncio.ensure_future(_history())
    origin_task = asyncio.ensure_future(_origin())
    try:
        raw_history, raw_origin = await asyncio.gather(history_task, origin_task)
    except BaseException:
        for task in (history_task, origin_task):
            if not task.done():
                task.cancel()
        raise

    def _items(rows: Sequence[Dict[str, Any]]) -> Tuple[DMSnapshotItem, ...]:
        # Rendered under the ORIGINAL channel id, not the one we read from. The lease's scopes
        # were built from whatever the message carried, and the reviewed-through extraction
        # compares scope tuples — swapping in the resolved id here would make every scope miss
        # and the rearm fail closed.
        rendered = [_render(client, row, channel_id=channel_id, requester_name=requester_name)
                    for row in rows if isinstance(row, dict)]
        return tuple(item for item in rendered if item is not None)

    return DMSurfaceSnapshot(message_items=_items(raw_history),
                             origin_items=_items(raw_origin))


# ------------------------------------------------------------------------------- the surface


class DMReconsiderSurface:
    """The DM half of the surface adapter the runner drives (ruling 6).

    Same three questions the channel surface answers — which model, what does the room look
    like now, and what request re-asks the question — and nothing else. The decision loop, the
    rearm, the fuse and the telemetry all stay in `reconsideration.py`, shared.
    """

    label = "dm"

    def __init__(self, *, processor: Any, client: Any, message: Any, turn: Any,
                 ctx: DMTurnContext):
        self._processor = processor
        self._client = client
        self._message = message
        self._turn = turn
        self.ctx = ctx
        self.model: Any = None

    # --- request_build boundary -------------------------------------------------------------

    def prepare(self) -> None:
        from message_processor.reconsideration import select_reconsideration_model

        self.model = select_reconsideration_model(self._turn, self.ctx.thread_config)

    # --- context_rebuild boundary -----------------------------------------------------------

    async def rebuild(self) -> DMSurfaceSnapshot:
        return await build_dm_reconsideration_snapshot(
            client=self._client, channel_id=self.ctx.channel_id,
            origin_root_ts=self.ctx.origin_root_ts,
            requester_name=self.ctx.requester_name)

    # --- request_build boundary -------------------------------------------------------------

    def trigger_line(self) -> str:
        text = (self.ctx.trigger_text or "").strip() or "(no text)"
        return f"[{self.ctx.requester_name} ts={self.ctx.trigger_ts}] {text}"

    def build_request(self, snapshot: DMSurfaceSnapshot, *, pass_number: int,
                      draft: str) -> Any:
        """The DM request over the fresh snapshot, plus the ONE appended developer item — the
        same grammar §4d fixes for channels, over the surface a DM actually has. The
        instructions and the sampling settings are the turn's OWN, pinned when it built its
        request; nothing about the question is written fresh here."""
        from message_processor.reconsideration import PreparedDecision, reconsideration_item

        items: List[Dict[str, Any]] = snapshot.input_items()
        items.append(reconsideration_item(pass_number, draft, self.trigger_line()))
        estimate = estimate_admission(
            instructions=self.ctx.instructions, input_items=items, tools=None,
            raw_document_texts=(), native_file_bounds=(), model=self.model,
            response_format=STALE_RECONSIDERATION_RESPONSE_FORMAT)
        cfg = self.ctx.thread_config or {}
        return PreparedDecision(
            instructions=self.ctx.instructions, api_items=items, estimate=estimate,
            params={"reasoning_effort": cfg.get("reasoning_effort"),
                    "verbosity": cfg.get("verbosity"),
                    "max_output_tokens": cfg.get("max_tokens"),
                    "temperature": cfg.get("temperature"),
                    "prompt_cache_key": self.ctx.prompt_cache_key})


__all__ = ["DMTurnContext", "DMSnapshotItem", "DMSurfaceSnapshot", "DMReconsiderSurface",
           "build_dm_reconsideration_snapshot", "pin_dm_turn_context",
           "resolve_dm_conversation_id"]
