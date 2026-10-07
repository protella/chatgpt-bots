"""A stale draft is a decision, not a drop (Docs/specs/STALE_RECONSIDERATION.md).

THE INCIDENT. A turn finished a correct 163-char answer after 55 seconds of real work, and the
StaleSendGuard suppressed the final post because another bot's interim message had advanced the
thread. The answer was silently discarded. The guard was right that the turn had no right to post
UNEXAMINED; it was wrong that the room should get nothing. So on suppression of a COMPLETE
buffered draft, the MAIN responder model looks at the updated conversation beside what it was
about to say and decides: post (after one more staleness check), force-post (waiving that check,
on the record), or skip.

THE SHAPE. The site keeps delivering; this runner only decides. Each covered delivery site, on
`StaleSendSuppressed` of its first surface, calls `intercept_stale_send` with a DELIVERY CLOSURE
`deliver(text) -> Optional[str]` that replaces the site's canonical text and re-runs the site's
own send — so every post-send bookkeeping path the site already owns (destination commit, F7
persistence, footer, receipts, splits) reads the chosen text with no second code path. The runner
never posts independently and owns NO cleanup: on every non-posted ending except
`delivery_failed` and `delivery_exception` — the two that return None to the site instead,
because a turn whose delivery visibly (or possibly) half-happened must never be filed as
"suppressed, room saw nothing" — it re-raises the newest suppression (marked, §5) so the site's
except path and main.py's terminal handling run exactly once, unmodified.

THE SURFACES. The loop below is surface-agnostic (STALE_SUPPRESSION_RECONSIDERATION ruling 6):
a small adapter answers "which model", "what does the room look like now" and "what request
re-asks the question", and the channel and the DM each answer it their own way — the channel
with its pure stream snapshot and the canonical assembler, the DM with a pure DM-SURFACE
snapshot (`dm_reconsideration.py`) that spans thread roots, because the message that suppressed
a DM draft usually arrived as a new top-level DM under a different root. Everything else — the
decision, the rearm, the fuse, the telemetry — is shared, and neither surface can drift from it.

THE LOOP (§4e — owner ruling: model judgment, no policy cap). Pass N rebuilds a pure snapshot of
the surface, assembles the ENTIRE normal request over it in no-tools mode, appends ONE
developer item quoting the current draft, and asks for a structured {decision, text}. A `post`
re-arms the guard through per-scope reviewed-through timestamps COMPUTED BY THIS MODULE from the
serialized snapshot — never supplied by the model — and delivers; a fresh suppression from that
send begins pass N+1. `RECONSIDER_FUSE_PASSES` is a malfunction backstop, not policy: the fuse
fires only when a SIXTH pass would begin, and a working model ends the loop rationally long
before that.

REDO (burst follow-ups R1-B/R2/R3). When no newer message has an owner of its own — no newer
responder running, no newer turn still deciding, nothing newer queued or in dispatch behind a
catch-up — and the turn has a hosted tool to re-answer with, the decision pass may also answer
`redo` (+ an optional interim line). The interim
posts at once through its own receipted send, which only VALIDATES the suppression; then a TOOLED
pass re-answers the trigger and the newer messages together over the same snapshot, with the
turn's hosted tools, and its text takes the ordinary rearm → deliver path. One tooled pass per
invocation; ownership is re-read after the redo decision and after the tooled pass, and a newer
owner found there means one more tool-free decision pass (redo not offered), never a blind skip.

THE ONCE-PER-TURN GATE is `TurnRuntime.reconsider`: the runner stamps it before every return,
every rethrow and cancellation propagation, and the interception wrappers rethrow immediately —
UNMARKED — any suppression observed while it is set, so main.py's terminal catch emits that
suppression's own `stale_send` row exactly once. Re-races INSIDE the loop are passes, not new
invocations, and never consult the gate.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, cast

from config import config
from logger import setup_logger
from message_processor import participation_telemetry
from message_processor.channel_request import (TOOL_MODE_HOSTED_ONLY,
                                               assemble_channel_request,
                                               capability_profile_hash, fresh_turn_context,
                                               hosted_only_profile, reconsideration_profile,
                                               to_input_items, tool_schema_version)
from message_processor.channel_stream import build_reconsideration_snapshot
from message_processor.context_meter import usable_limit
from message_processor.stale_send_guard import (Scope, StaleSendSuppressed, TurnSendLease,
                                                scopes_for, ts_key)
from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM
from message_processor.turn_runtime import DEST_KIND_RECONSIDER_INTERIM, ReconsiderFacts
from message_processor.utilities import effective_request_model
from slack_client.normalizer import TimestampError
from openai_client.api.responses import build_reconsideration_create_kwargs
from message_processor.prompts import (RECONSIDERATION_INSTRUCTION,
                                       RECONSIDERATION_NEWER_RESPONDERS,
                                       RECONSIDERATION_REDO, RECONSIDERATION_TOOLED)

logger = setup_logger(name="slack_bot.Reconsideration")

# §4e: the exact fuse boundary. Passes 1..5 each run to a completed decision (a pass-5
# `force_post` delivers); the fuse fires only when a SIXTH pass would begin. A malfunction
# backstop — our loop is harness recursion spending a full model call per pass — never policy.
RECONSIDER_FUSE_PASSES = 5

# The epoch fence's refusal (dev-only mechanism; new-post closures only, §4a r3-6). Imported
# defensively the way slack_client/messaging.py holds it: if the module is unavailable the
# except-clause below simply never matches.
try:  # pragma: no cover - import shape, not behavior
    from message_processor.epoch_fence import EpochEffectRefused as _EPOCH_REFUSED
except Exception:  # noqa: BLE001 - the fence is dev-only machinery
    class _EPOCH_REFUSED(Exception):  # type: ignore[no-redef]
        """Stands in when the fence module is unavailable; never raised."""


# --------------------------------------------------------------------- request construction


def select_reconsideration_model(turn: Any, thread_config: Optional[Dict[str, Any]]) -> Any:
    """§4d model precedence: the LAST `ModelAttempt.model` if an attempt exists AND its model
    value is truthy, else `effective_request_model()` over the pinned profile."""
    attempts = list(getattr(turn, "model_attempts", None) or ())
    if attempts and getattr(attempts[-1], "model", None):
        return attempts[-1].model
    return effective_request_model(thread_config)


def draft_fence(draft: str) -> str:
    """A backtick fence that cannot occur in the draft: one backtick longer than the draft's
    longest backtick run, floored at the standard three (§4d)."""
    longest = max((len(run) for run in re.findall(r"`+", draft or "")), default=0)
    return "`" * max(3, longest + 1)


# Hosted tools are recorded under their API names; the item reads better in words.
_HOSTED_TOOL_LABELS = {"web_search": "web search", "code_interpreter": "code interpreter"}


def tools_used_summary(turn: Any) -> List[str]:
    """The tools this turn actually ran while producing its draft — names only, no counts.

    Two turn-owned records feed it. `provenance_tool_calls` holds one entry per LOCAL call,
    including calls answered without being run (marked `dispatched: False` by the tool loop);
    only the ones that ran count. `provenance_external_tools` holds the hosted names (web search,
    the sandbox, MCP) but also every local name the loop merged, run or not — so any name that
    appears among the local records is left to the local side to decide.
    """
    local_records = [r for r in (getattr(turn, "provenance_tool_calls", None) or ())
                     if isinstance(r, dict) and r.get("name")]
    local_names = {str(r["name"]) for r in local_records}
    names: List[str] = []
    for name in getattr(turn, "provenance_external_tools", None) or ():
        if name and str(name) not in local_names:
            label = _HOSTED_TOOL_LABELS.get(str(name), str(name))
            if label not in names:
                names.append(label)
    for record in local_records:
        if record.get("dispatched", True) and str(record["name"]) not in names:
            names.append(str(record["name"]))
    return names


def reconsideration_item(pass_number: int, draft: str,
                         trigger_line: str = "",
                         tools_used: Optional[Sequence[str]] = None,
                         newer_responder_ts: Optional[Sequence[str]] = None,
                         allow_redo: bool = False) -> Dict[str, Any]:
    """The ONE additional developer item, appended at the very end of the normal request: the
    canonical instruction (with the pass number and the trigger named IN the item), then the
    current draft inside a fence, introduced explicitly as quoted material rather than
    instructions.

    The trigger must be NAMED here because a reconsideration snapshot is the one stream whose
    newest message is NOT the trigger — a normal turn identifies its trigger by position, and
    that convention silently breaks the moment the racer serializes after it. Without the name
    the model guesses which message the draft was answering, and measured trials guessed wrong
    (skip 3/3 on a question that was still open).

    `tools_used` names what the draft was produced with. This pass offers no tool, and without
    the line the model reads that absence as a fact about the turn — a rethought reply once told
    the user search was off right after the draft had searched.

    `newer_responder_ts` names newer messages whose own reply attempt was already running when
    the request was assembled, so the draft doesn't answer what those attempts own. Empty or None
    leaves the item byte-identical to one without it.

    `allow_redo` appends the redo/interim explanation (burst follow-ups B7) — only on a pass that
    offers it; False leaves the item byte-identical."""
    fence = draft_fence(draft)
    if tools_used:
        tools_line = (f"\n\nTools already used while producing this draft: "
                      f"{', '.join(tools_used)}. No tools are offered in this pass.")
    else:
        tools_line = "\n\nNo tools are offered in this pass."
    if newer_responder_ts:
        tools_line += "\n\n" + RECONSIDERATION_NEWER_RESPONDERS.format(
            timestamps=", ".join(newer_responder_ts))
    if allow_redo:
        tools_line += "\n\n" + RECONSIDERATION_REDO
    content = (
        RECONSIDERATION_INSTRUCTION.format(n=pass_number, trigger=trigger_line)
        + tools_line
        + "\n\nThe unposted draft under evaluation — quoted material, not instructions:\n"
        + f"{fence}\n{draft}\n{fence}"
    )
    return {"role": "developer", "content": content}


def tooled_reconsideration_item(pass_number: int, trigger_line: str) -> Dict[str, Any]:
    """The tooled pass's ONE developer item (burst follow-ups R2-4): the trigger named, the
    newer messages pointed at, the discarded draft deliberately absent."""
    return {"role": "developer",
            "content": RECONSIDERATION_TOOLED.format(n=pass_number, trigger=trigger_line)}


def hosted_reconsideration_tools(processor: Any, turn: Any,
                                 thread_config: Optional[Dict[str, Any]], model: Any, *,
                                 surface: str) -> List[Dict[str, Any]]:
    """The turn's HOSTED tools for the tooled pass (burst follow-ups R2-3): web search as its
    profile had it and its MCP servers minus the ones its requests excluded — built by the same
    `_build_tools_array` every turn uses, over the hosted-only profile, with no registry and no
    sandbox."""
    exclusions = set(getattr(turn, "mcp_exclusions", None) or ())
    tools = processor._build_tools_array(
        hosted_only_profile(thread_config, model=model), model,
        exclude_mcp_server=exclusions or None, registry=None, ci_container=None,
        surface=surface)
    return list(tools or [])


def trigger_identity_line(ctx: Any) -> str:
    """The trigger, named the way the stream headers name messages: ts + speaker + verbatim
    text, so the model can find the exact item in the room above."""
    who = getattr(ctx.requester, "real_name", None) or getattr(
        ctx.requester, "user_id", None) or "unknown"
    text = (ctx.trigger_text or "").strip() or "(no text)"
    return f"[{who} ts={ctx.trigger_ts}] {text}"


def build_reconsideration_request(*, processor: Any, client: Any, ctx: Any, model: Any,
                                  pass_number: int, draft: str,
                                  reply_destination: Optional[str] = None,
                                  tools_used: Optional[Sequence[str]] = None,
                                  newer_responder_ts: Optional[Sequence[str]] = None,
                                  allow_redo: bool = False
                                  ) -> Tuple[Any, List[Dict[str, Any]]]:
    """§4d request grammar, literally: the ENTIRE normal assembled channel request over the
    (fresh) context, unchanged and in its existing order, in no-tools mode — then the one
    appended developer item. Returns (request, api_items). PURE and synchronous: what the
    request costs is OpenAI's count of the final create kwargs, taken by the runner inside its
    `request_build` boundary (CONTEXT_METER §3.10)."""
    request = assemble_channel_request(
        processor=processor, client=client, ctx=ctx, model=model, tools=[],
        request_config=None, contract_suffix=None, registry=None,
        reply_destination=reply_destination, no_tools=True)
    extra = reconsideration_item(pass_number, draft, trigger_identity_line(ctx),
                                 tools_used=tools_used,
                                 newer_responder_ts=newer_responder_ts,
                                 allow_redo=allow_redo)
    api_items = [*to_input_items(request), extra]
    return request, api_items


def build_tooled_reconsideration_request(*, processor: Any, client: Any, ctx: Any, model: Any,
                                         pass_number: int, tools: List[Dict[str, Any]],
                                         reply_destination: Optional[str] = None
                                         ) -> Tuple[Any, List[Dict[str, Any]]]:
    """The tooled pass's request (burst follow-ups R2-3): the same canonical assembly over the
    same fresh context in hosted-only mode, then the tooled item. PURE and synchronous, like its
    tool-free twin."""
    request = assemble_channel_request(
        processor=processor, client=client, ctx=ctx, model=model, tools=tools,
        request_config=None, contract_suffix=None, registry=None,
        reply_destination=reply_destination, tool_mode=TOOL_MODE_HOSTED_ONLY)
    api_items = [*to_input_items(request),
                 tooled_reconsideration_item(pass_number, trigger_identity_line(ctx))]
    return request, api_items


# --------------------------------------------------------------------- reviewed-through


def _snapshot_items(stream: Any) -> Tuple[Any, ...]:
    """EXACTLY the extraction set of §4a: the serialized `message_items` and `origin_items` of
    the fresh stream. Fetched-but-filtered candidates, embedded root snippets and framing items
    (whose metadata carries no ts) never advance a scope."""
    return (*getattr(stream, "message_items", ()), *getattr(stream, "origin_items", ()))


def suppressing_ts_present(stream: Any, observed_latest_ts: Any) -> bool:
    """The suppressing message must itself be in the rebuilt input — missing, deleted, malformed
    or filtered means fail closed (§4a / §4f `context_rebuild`)."""
    if observed_latest_ts is None:
        return False
    target = str(observed_latest_ts)
    return any(item.metadata.get("ts") == target for item in _snapshot_items(stream))


def suppressing_ts_summarized(stream: Any, observed_latest_ts: Any) -> bool:
    """Is the suppressing message inside the fresh snapshot's thread summary? It never should be
    — the protected rule keeps the newest messages out of every summary (CONTEXT_METER §3.10) —
    so True is a broken invariant, and a review of a message the model sees only as a summary
    cannot claim to have read it."""
    summary = getattr(getattr(stream, "pinned", None), "origin_summary", None)
    if summary is None or observed_latest_ts is None:
        return False
    target = str(observed_latest_ts)
    if target in summary.covered_ts:
        return True
    try:
        return ts_key(target) <= ts_key(summary.boundary_ts)
    except TimestampError:
        return True


def reviewed_through_map(lease: TurnSendLease, stream: Any) -> Dict[Scope, str]:
    """Per-scope reviewed-through evidence, computed by trusted runtime code (§4a): for each of
    the lease's scopes, the greatest inbound ts among the serialized items belonging to it,
    floored at the scope's current effective baseline. A scope with nothing new simply doesn't
    advance; a scope with no value at all is omitted, and the rearm preconditions then fail
    closed."""
    reviewed: Dict[Scope, str] = {}
    items = _snapshot_items(stream)
    for scope in lease.scopes:
        best = lease._effective_baseline(scope)
        for item in items:
            meta = item.metadata
            ts = meta.get("ts")
            if not ts:
                continue
            if meta.get("sender_type") == "self":
                # §4a: INBOUND only, matching the sites' inbound accounting
                # (handlers/text.py's advance-last-seen skips sender_type == "self"). Our own
                # serialized posts are not evidence the room was reviewed; if the human
                # suppressor is absent, the scope floors at its baseline and rearm fails closed.
                continue
            member = scopes_for(meta.get("channel_id"), ts, meta.get("thread_root_ts"),
                                meta.get("sender_id"))
            if scope in member and (best is None or ts_key(ts) > ts_key(best)):
                best = ts
        if best is not None:
            reviewed[scope] = str(best)
    return reviewed


# --------------------------------------------------------------------- surface adapters


@dataclass
class PreparedDecision:
    """One pass's request, whatever surface built it: what to send. What it costs is counted
    by the runner over the final create kwargs, never estimated here."""

    instructions: str
    api_items: List[Dict[str, Any]]
    params: Dict[str, Any] = field(default_factory=dict)
    # The pass's mode, read by BOTH the count and the create: the redo schema, and — on the
    # tooled pass only — the hosted tools (None on every tool-free pass).
    allow_redo: bool = False
    tools: Optional[List[Dict[str, Any]]] = None


class ChannelReconsiderSurface:
    """The channel half of the adapter: the pure channel snapshot and the canonical assembler,
    exactly as §4c/§4d fix them. Nothing here is new — it is the runner's original body, moved
    behind the same three questions the DM surface answers (STALE_SUPPRESSION_RECONSIDERATION
    ruling 6), so the decision loop below no longer knows which surface it is serving."""

    label = "channel"

    def __init__(self, *, processor: Any, client: Any, message: Any, turn: Any, ctx: Any):
        self._processor = processor
        self._client = client
        self._message = message
        self._turn = turn
        self.ctx = ctx
        self.model: Any = None
        self._profile_hash = ""
        self._schema_version = ""

    def prepare(self) -> None:
        self.model = select_reconsideration_model(self._turn, self.ctx.thread_config)
        profile = reconsideration_profile(self.ctx.thread_config, model=self.model)
        self._profile_hash = capability_profile_hash(profile)
        self._schema_version = tool_schema_version(None, profile)

    async def rebuild(self) -> Any:
        snapshot = await build_reconsideration_snapshot(
            client=self._client, db=getattr(self._processor, "db", None),
            team_id=self.ctx.team_id, channel_id=self.ctx.channel_id,
            trigger_ts=self.ctx.trigger_ts, origin_root_ts=self.ctx.origin_thread_ts,
            capability_profile_hash=self._profile_hash,
            tool_schema_version=self._schema_version,
            reach_tools=(),
            drain_timeout=getattr(config, "index_drain_timeout_seconds", None))
        return snapshot.stream

    def _reply_destination(self) -> Optional[str]:
        return (getattr(self._turn, "reply_destination", None)
                if getattr(self._turn, "destination_selected", False) else None)

    def _params(self, request: Any) -> Dict[str, Any]:
        cfg = self.ctx.thread_config
        return {"reasoning_effort": cfg.get("reasoning_effort"),
                "verbosity": cfg.get("verbosity"),
                "max_output_tokens": cfg.get("max_tokens"),
                "temperature": cfg.get("temperature"),
                "prompt_cache_key": request.prompt_cache_key}

    def build_request(self, stream: Any, *, pass_number: int, draft: str,
                      newer_responder_ts: Optional[Sequence[str]] = None,
                      allow_redo: bool = False) -> PreparedDecision:
        fresh_ctx = fresh_turn_context(self.ctx, stream)
        request, api_items = build_reconsideration_request(
            processor=self._processor, client=self._client, ctx=fresh_ctx, model=self.model,
            pass_number=pass_number, draft=draft,
            reply_destination=self._reply_destination(),
            tools_used=tools_used_summary(self._turn),
            newer_responder_ts=newer_responder_ts,
            allow_redo=allow_redo)
        return PreparedDecision(instructions=request.instructions, api_items=api_items,
                                params=self._params(request), allow_redo=allow_redo)

    # The canonical assembler has a hosted-only mode, so a channel turn can always re-ask.
    supports_tooled = True

    def build_tooled_request(self, stream: Any, *, pass_number: int) -> PreparedDecision:
        """The tooled pass over the SAME snapshot the redo decision read (burst follow-ups
        R1-B 3), with the turn's hosted tools."""
        fresh_ctx = fresh_turn_context(self.ctx, stream)
        tools = hosted_reconsideration_tools(self._processor, self._turn,
                                             self.ctx.thread_config, self.model,
                                             surface=SURFACE_CHANNEL)
        request, api_items = build_tooled_reconsideration_request(
            processor=self._processor, client=self._client, ctx=fresh_ctx, model=self.model,
            pass_number=pass_number, tools=tools, reply_destination=self._reply_destination())
        return PreparedDecision(instructions=request.instructions, api_items=api_items,
                                params=self._params(request), tools=list(request.tools or []))


def surface_for(*, processor: Any, client: Any, message: Any, turn: Any,
                channel_turn: bool) -> Optional[Any]:
    """Which surface's fresh context this turn reconsiders over — or None, meaning it cannot
    reconsider at all and the suppression stands (fail closed).

    A channel turn needs its pinned `ChannelTurnContext`; a DM needs the `DMTurnContext` its
    handler pinned when it built the request. A turn carrying neither is one whose request we
    would have to INVENT to re-ask it, and inventing the question is not reconsideration."""
    if channel_turn:
        ctx = getattr(turn, "channel_turn_context", None)
        if ctx is None:
            return None
        return ChannelReconsiderSurface(processor=processor, client=client, message=message,
                                        turn=turn, ctx=ctx)
    dm_ctx = getattr(turn, "dm_turn_context", None)
    if dm_ctx is None or not getattr(dm_ctx, "channel_id", None):
        return None
    from message_processor.dm_reconsideration import DMReconsiderSurface

    return DMReconsiderSurface(processor=processor, client=client, message=message, turn=turn,
                               ctx=dm_ctx)


# --------------------------------------------------------------------- ownership (redo)


def redo_offerable(lease: TurnSendLease, processor: Any, channel_id: Optional[str]) -> bool:
    """May this pass offer `redo`? Only while no newer message in the lease's scopes has an
    owner of its own (burst follow-ups R1-B 4, R2-2, R3-1): no newer responder running, no newer
    turn still deciding, and nothing newer queued or in dispatch behind a Phase Q catch-up.
    Synchronous reads — nothing waits."""
    if lease.newer_responding_ts() or lease.newer_deciding_ts():
        return False
    manager = getattr(processor, "thread_manager", None)
    pending = getattr(manager, "pending_in_scope", None)
    if callable(pending) and pending(channel_id, lease.scopes, lease.ceiling_ts):
        return False
    return True


def newer_owner_ts(lease: TurnSendLease, processor: Any,
                   channel_id: Optional[str]) -> List[str]:
    """Every newer message that has an owner of its own, by ts — the newer-responder guidance
    for the decision pass that follows a failed ownership recheck (R3-2)."""
    found: Dict[str, None] = {ts: None for ts in (*lease.newer_responding_ts(),
                                                  *lease.newer_deciding_ts())}
    manager = getattr(processor, "thread_manager", None)
    pending = getattr(manager, "pending_ts_in_scope", None)
    if callable(pending):
        for ts in pending(channel_id, lease.scopes, lease.ceiling_ts) or ():
            found[str(ts)] = None
    return sorted(found, key=ts_key)


async def post_reconsider_interim(*, client: Any, message: Any, turn: Any,
                                  lease: TurnSendLease, expected: StaleSendSuppressed,
                                  text: str) -> Optional[str]:
    """The interim line (burst follow-ups B6, R2-1): a SEPARATE receipted send. It validates
    the live suppression and changes nothing on the lease — no waiver, no rearm, no first
    surface — so the answer still has to pass the rearm → deliver path. Committed as a
    `reconsider_interim` destination, which channel memory never reads."""
    lease.validate_suppression(expected)
    target = turn.resolve_reply_target(message)
    meta: Dict[str, Any] = {}
    posted = await client.send_message(  # unleased-ok: interim note — validate_suppression above is its check, and the answer's own surface stays leased (R2-1)
        message.channel_id, target, text, meta_out=meta,
        receipts=getattr(turn, "receipt_ledger", None), receipt_class="assistant_reply")
    if posted:
        delivery = meta.get("delivery")
        turn.mark_destination_committed(
            first_ts=posted, kind=DEST_KIND_RECONSIDER_INTERIM,
            text=delivery.text if delivery is not None else text,
            complete=delivery.complete if delivery is not None else True,
            channel_id=message.channel_id, thread_root_ts=target)
    return posted


# --------------------------------------------------------------------- interception


async def intercept_stale_send(*, processor: Any, client: Any, message: Any, turn: Any,
                               lease: Optional[TurnSendLease],
                               suppressed: StaleSendSuppressed, draft: str,
                               deliver: Callable[..., Awaitable[Optional[str]]],
                               channel_turn: bool = True) -> Optional[str]:
    """The site wrapper (§4b). `channel_turn` no longer decides WHETHER a draft is reviewed —
    only which surface provides the fresh context (STALE_SUPPRESSION_RECONSIDERATION ruling 6):
    a DM's draft is reviewed over a pure DM-surface snapshot, through this same runner.

    A turn with no lease, no runtime, or no pinned context for its surface still rethrows —
    fail closed. So does a suppression observed while `turn.reconsider` is already set (the
    once-per-turn gate), rethrown immediately and UNMARKED so main.py's terminal catch emits
    its `stale_send` row exactly once: no second runner, no second outcome, no lost row."""
    if turn is None or lease is None:
        raise suppressed
    if getattr(turn, "reconsider", None) is not None:
        raise suppressed
    surface = surface_for(processor=processor, client=client, message=message, turn=turn,
                          channel_turn=channel_turn)
    if surface is None:
        raise suppressed
    return await reconsider_stale_draft(
        processor=processor, client=client, message=message, turn=turn, lease=lease,
        suppressed=suppressed, draft=draft, deliver=deliver, surface=surface)


async def reconsider_stale_draft(*, processor: Any, client: Any, message: Any, turn: Any,
                                 lease: TurnSendLease, suppressed: StaleSendSuppressed,
                                 draft: str,
                                 deliver: Callable[..., Awaitable[Optional[str]]],
                                 surface: Optional[Any] = None) -> Optional[str]:
    """The runner (§4b–§4f). Decide-only: it never posts the answer (delivery is the site's
    closure) and owns no cleanup (rethrow endings hand the turn back to today's stale terminal
    path). Its one post of its own is the redo interim, through a separate receipted send.

    Returns the delivered first-surface ts on a posted ending, or None on the
    `delivery_failed` and `delivery_exception` endings — the TWO non-posted endings that do not
    rethrow, because a turn that visibly posted a truncation notice (or whose delivery raised
    after Slack may already have accepted content) must never be classified as "suppressed,
    room saw nothing"; the site's own failed-delivery accounting owns the state (§4b r5-1,
    §4f review r8). Every other non-posted ending emits its outcome, marks the newest
    suppression `telemetry_recorded`, and re-raises it.

    `deliver(text)` re-runs the site's send over a tool-free pass's text. The tooled pass's
    text is delivered as `deliver(text, extra_tools=…, extra_results=…)` — the hosted tools it
    completed, which the site merges into its attribution and provenance (R3-5).

    `surface` provides the fresh context and the request over it (channel or DM). Omitted, it
    defaults to this turn's channel surface — the shape every existing caller passes.
    """
    channel_id = getattr(message, "channel_id", None)
    trigger_ts = (getattr(message, "metadata", None) or {}).get("ts")
    turn_id = getattr(turn, "turn_id", None)
    attempt_id = participation_telemetry.attempt_id_for(message)
    guard_mode = getattr(turn, "guard_mode", None)

    initial_draft = draft
    current_draft = draft
    current = suppressed          # the newest StaleSendSuppressed; re-remembered per re-race
    passes = 0                    # passes begun == reconsider_start rows this invocation emitted
    # The next pass begins with a NEW suppression event (and writes its stale_send row). False
    # only for the tooled pass and an ownership recheck, which re-decide the same suppression.
    new_suppression = True
    redo_decided = False          # `redo` is offered until the first time it is chosen
    tooled_next = False           # the next pass is the tooled pass, over the redo's snapshot
    redo_stream: Any = None
    redo_reviewed: Dict[Scope, str] = {}
    recheck = False               # this decision pass follows a failed ownership recheck
    tooled_ran = False
    draft_from_tooled = False     # the current draft is (derived from) the tooled pass's text
    draft_tools: Tuple[str, ...] = ()
    draft_results: Tuple[Dict[str, Any], ...] = ()

    def _emit_suppression(exc: StaleSendSuppressed) -> None:
        """One `stale_send` row per suppression EVENT the runner handles (§5 single-owner rule),
        emitted BEFORE any await or snapshot work of its pass and marked so the terminal catch
        never double-counts it."""
        exc.telemetry_recorded = True
        participation_telemetry.stale_send(
            channel_id, trigger_ts, attempt_id=attempt_id, turn_id=turn_id,
            last_seen_ts=exc.last_seen_ts, observed_latest_ts=exc.observed_latest_ts,
            scope=exc.scope[0] if exc.scope else None, surface=exc.surface,
            guard_mode=guard_mode)

    def _finish(outcome: str, *, forced: Optional[bool] = None,
                error: Optional[str] = None) -> None:
        """Stamp the once-per-turn gate, then emit the outcome — exactly one per invocation."""
        redone: Optional[bool] = None
        if tooled_ran:
            redone = draft_from_tooled and outcome in ("posted_asis", "posted_revised")
        turn.reconsider = ReconsiderFacts(outcome=outcome, passes=passes, forced=forced,
                                          error=error, redone=redone)
        participation_telemetry.reconsider_outcome(
            channel_id, trigger_ts, turn_id=turn_id, outcome=outcome, passes=passes,
            attempt_id=attempt_id, forced=forced, error=error, redone=redone)

    def _hosted_tools_available() -> bool:
        """Ruling 3: redo is offered only when the tooled pass would have at least one hosted
        tool — web search on in the turn's profile, or an MCP server left after exclusions."""
        kind = SURFACE_DM if getattr(surface, "label", None) == "dm" else SURFACE_CHANNEL
        return bool(hosted_reconsideration_tools(
            processor, turn, getattr(getattr(surface, "ctx", None), "thread_config", None),
            getattr(surface, "model", None), surface=kind))

    def _give_up(error: str) -> StaleSendSuppressed:
        """§4f give-up: outcome `error_dropped` with the subtype, rethrow the newest suppression
        (already marked when its row was emitted)."""
        _finish("error_dropped", error=error)
        return current

    if surface is None:
        surface = ChannelReconsiderSurface(
            processor=processor, client=client, message=message, turn=turn,
            ctx=getattr(turn, "channel_turn_context", None))
    try:
        while True:
            if new_suppression:
                _emit_suppression(current)
                new_suppression = False
            if passes >= RECONSIDER_FUSE_PASSES:
                # A SIXTH pass would begin. Malfunction backstop, never policy (§4e).
                logger.error(
                    f"Reconsideration fuse: {passes} passes completed on {channel_id} and the "
                    f"draft is still racing — dropping (newest supersession "
                    f"{current.observed_latest_ts} in {current.scope})")
                _finish("fuse_dropped")
                raise current
            pass_number = passes + 1
            tooled = tooled_next
            tooled_next = False

            # ---- pass N: model selection and profile (§4d). A failure HERE is a programming
            # failure — `request_build`, never `context_rebuild`: it must not masquerade as a
            # Slack-history failure (§4f review r8). ------------------------------------------
            try:
                if getattr(surface, "ctx", None) is None:
                    raise RuntimeError(f"{getattr(surface, 'label', 'unknown')} turn reached "
                                       "reconsideration with no pinned context")
                surface.prepare()
            except asyncio.CancelledError:
                raise
            except Exception as selection_error:  # noqa: BLE001 — §4f: never post unexamined
                logger.error(f"Reconsideration request preparation failed on {channel_id}: "
                             f"{selection_error}")
                raise _give_up("request_build") from selection_error

            if tooled:
                # ---- the tooled pass (R1-B 3): the SAME snapshot the redo decision read, so
                # its presence checks already hold and its reviewed-through map is reused. ----
                fresh_stream = redo_stream
                reviewed = redo_reviewed
                newer_count = 0
                try:
                    prepared = surface.build_tooled_request(fresh_stream,
                                                            pass_number=pass_number)
                    counted = await processor.openai_client.count_input_tokens(
                        build_reconsideration_create_kwargs(prepared, model=surface.model))
                    limit = usable_limit(surface.model)
                except asyncio.CancelledError:
                    raise
                except Exception as assembly_error:  # noqa: BLE001 — never post unexamined
                    logger.error(f"Tooled reconsideration request assembly failed on "
                                 f"{channel_id}: {assembly_error}")
                    raise _give_up("request_build") from assembly_error
            else:
                # ---- the pure snapshot (§4c). `context_rebuild` is EXACTLY the Slack-history
                # failures: snapshot build, deadline miss, missing suppressing ts. ------------
                try:
                    fresh_stream = await surface.rebuild()
                except asyncio.CancelledError:
                    raise
                except Exception as rebuild_error:  # noqa: BLE001 — §4f: never post unexamined
                    logger.error(f"Reconsideration context rebuild failed on {channel_id}: "
                                 f"{rebuild_error}")
                    raise _give_up("context_rebuild") from rebuild_error

                # ---- the request over it (§4d). Assembly and counting failures are
                # `request_build` again — the snapshot already succeeded. The count is taken
                # over the FINAL create kwargs the decision call sends (R3-6) and consumed
                # inside the same boundary (§4f: request_build covers model selection through
                # assembly AND the size check), so nothing about it escapes unclassified. ----
                try:
                    present = suppressing_ts_present(fresh_stream, current.observed_latest_ts)
                    reviewed = reviewed_through_map(lease, fresh_stream)
                    # Sampled per pass, after the rebuild and right before assembly: which newer
                    # messages already have their own responder running — and, after a failed
                    # ownership recheck, every newer message that has an owner (R3-2).
                    newer_responders = (newer_owner_ts(lease, processor, channel_id) if recheck
                                        else lease.newer_responding_ts())
                    newer_count = len(newer_responders)
                    # Redo needs two more fuse slots: the tooled pass, and the recheck a newer
                    # owner may force after it — a redo that cannot finish is never offered.
                    allow_redo = (not redo_decided
                                  and pass_number + 2 <= RECONSIDER_FUSE_PASSES
                                  and bool(getattr(surface, "supports_tooled", False))
                                  and redo_offerable(lease, processor, channel_id)
                                  and _hosted_tools_available())
                    prepared = surface.build_request(fresh_stream, pass_number=pass_number,
                                                     draft=current_draft,
                                                     newer_responder_ts=newer_responders,
                                                     allow_redo=allow_redo)
                    counted = await processor.openai_client.count_input_tokens(
                        build_reconsideration_create_kwargs(prepared, model=surface.model))
                    limit = usable_limit(surface.model)
                except asyncio.CancelledError:
                    raise
                except Exception as assembly_error:  # noqa: BLE001 — never post unexamined
                    logger.error(f"Reconsideration request assembly failed on {channel_id}: "
                                 f"{assembly_error}")
                    raise _give_up("request_build") from assembly_error
                if suppressing_ts_summarized(fresh_stream, current.observed_latest_ts):
                    logger.error(
                        f"Reconsideration snapshot on {channel_id} carries the suppressing ts "
                        f"{current.observed_latest_ts} inside its thread summary — the "
                        "protected rule failed; dropping")
                    raise _give_up("request_build")
                if not present:
                    # The suppressing message is missing, deleted, malformed or filtered — the
                    # review cannot claim to have covered it. Fail closed (§4a).
                    logger.warning(
                        f"Reconsideration snapshot on {channel_id} does not contain the "
                        f"suppressing ts {current.observed_latest_ts} — dropping")
                    raise _give_up("context_rebuild")
            if counted.tokens is None or not counted.complete:
                # A detached count that could not describe the whole request: unlike a turn's
                # preflight, a review that cannot be sized fails closed (CONTEXT_METER §3.10).
                logger.warning(
                    f"Reconsideration request on {channel_id} could not be counted "
                    f"(tokens={counted.tokens}, complete={counted.complete}) — dropping")
                raise _give_up("request_build")
            if counted.tokens > limit:
                logger.warning(
                    f"Reconsideration request on {channel_id} over the context window "
                    f"({counted.tokens:,} of {limit:,} tokens) — dropping")
                raise _give_up("admission_overflow")

            # ---- the decision call (§4d): a NEW ModelAttempt of the SAME turn ---------------
            sink = participation_telemetry.ModelAttemptSink(
                turn=turn, fork_reason="stale_reconsideration")

            def _on_attempt_open(seq: Optional[int], _pass: int = pass_number,
                                 _exc: StaleSendSuppressed = current,
                                 _newer: int = newer_count, _tooled: bool = tooled,
                                 _recheck: bool = recheck and not tooled) -> None:
                nonlocal passes, tooled_ran
                passes = _pass
                if _tooled:
                    # From here the tooled pass is on the record, so every ending — a failed
                    # call included — reports `redone` (False unless its text is delivered).
                    tooled_ran = True
                participation_telemetry.reconsider_start(
                    channel_id, trigger_ts, turn_id=turn_id, pass_number=_pass,
                    scope=_exc.scope, observed_latest_ts=_exc.observed_latest_ts,
                    attempt_id=attempt_id, model_attempt_seq=seq,
                    newer_responder_count=_newer, tooled=_tooled, recheck=_recheck)

            try:
                decision = await processor.openai_client.create_reconsideration_decision(
                    input_items=prepared.api_items,
                    instructions=prepared.instructions,
                    model=surface.model,
                    attempt_sink=sink,
                    on_attempt_open=_on_attempt_open,
                    allow_redo=prepared.allow_redo,
                    tools=prepared.tools,
                    **prepared.params)
            except asyncio.CancelledError:
                raise
            except Exception as model_error:  # noqa: BLE001 — timeout/refusal/schema alike
                logger.error(f"Reconsideration decision call failed on {channel_id}: "
                             f"{model_error}")
                raise _give_up("model_failure") from model_error

            # ---- act on it (§4d semantics, plus redo) ----------------------------------------
            if tooled:
                if decision.tools_used:
                    # The hosted calls ran whatever happens next (R2-5).
                    turn.note_external_tools(list(decision.tools_used))
                if decision.decision == "skip":
                    _finish("skipped")
                    raise current
                # Never the discarded draft: the tooled pass's text is required (R2-4).
                current_draft = cast(str, decision.text)
                draft_from_tooled = True
                draft_tools, draft_results = decision.tools_used, decision.tool_results
                if not redo_offerable(lease, processor, channel_id):
                    # Check (c): a newer message gained an owner while the tooled pass ran. Its
                    # text becomes the draft and one more tool-free pass decides (R3-2).
                    logger.info(f"Tooled reconsideration on {channel_id}: a newer message "
                                "gained its own owner — re-deciding without tools")
                    recheck = True
                    continue
            elif decision.decision == "redo":
                redo_decided = True
                if not redo_offerable(lease, processor, channel_id):
                    # Check (b): an owner appeared while the decision ran — no interim, and the
                    # decision is re-run with redo off the table (R3-2).
                    logger.info(f"Reconsideration redo on {channel_id}: a newer message gained "
                                "its own owner — re-deciding without redo")
                    recheck = True
                    continue
                recheck = False
                if decision.interim:
                    try:
                        await post_reconsider_interim(client=client, message=message, turn=turn,
                                                      lease=lease, expected=current,
                                                      text=decision.interim)
                    except asyncio.CancelledError:
                        raise
                    except StaleSendSuppressed:
                        raise      # an unleased send never raises one; never swallowed if it did
                    except Exception as interim_error:  # noqa: BLE001 — the redo still runs
                        logger.warning(f"Reconsideration interim on {channel_id} did not post: "
                                       f"{interim_error}")
                tooled_next = True
                redo_stream, redo_reviewed = fresh_stream, reviewed
                continue
            else:
                recheck = False
                if decision.decision == "skip":
                    _finish("skipped")
                    raise current
                if (decision.text is not None
                        and decision.text.strip() != current_draft.strip()):
                    current_draft = decision.text
            forced = decision.decision == "force_post"
            try:
                if forced:
                    lease.force_after_reconsideration(current)
                else:
                    lease.rearm_after_reconsideration(reviewed, current)
            except ValueError as rearm_error:
                logger.error(f"Reconsideration guard transition refused on {channel_id}: "
                             f"{rearm_error}")
                raise _give_up("guard_rearm_failed") from rearm_error

            # ---- deliver, through the site's own closure (§4b) -------------------------------
            try:
                if draft_from_tooled:
                    delivered_ts = await deliver(current_draft, extra_tools=draft_tools,
                                                 extra_results=draft_results)
                else:
                    delivered_ts = await deliver(current_draft)
            except StaleSendSuppressed as re_race:
                # A re-race is the next pass, not an escape: its stale_send row is emitted at
                # the top of the loop, and THIS exception becomes `expected` for the next rearm.
                current = re_race
                new_suppression = True
                continue
            except _EPOCH_REFUSED as fence_error:
                logger.error(f"Epoch fence refused the reconsidered delivery on {channel_id}: "
                             f"{fence_error}")
                raise _give_up("epoch_invalidated") from fence_error
            except asyncio.CancelledError:
                raise
            except Exception as deliver_error:  # noqa: BLE001 — §4f delivery_exception
                # A NON-ENUMERATED exception from the site's closure (§4f review r8). Physical
                # acceptance may be UNKNOWN — the exception may have fired after Slack accepted
                # content — so this takes the same mandated no-rescue site path as
                # `delivery_failed`: cancel any live force waiver (it never survives its
                # delivery, §4a), stamp the gate, emit the outcome, and return None. No stale
                # rethrow, no fallback resend, no error notice, no artifact rescue.
                logger.error(f"Reconsidered delivery raised on {channel_id}: {deliver_error}",
                             exc_info=True)
                lease.cancel_force_waiver()
                _finish("error_dropped", error="delivery_exception")
                return None

            if delivered_ts is None:
                # §4f delivery_failed: emit the outcome and return None to the site — NO stale
                # rethrow (a visible truncation notice must not be filed as "room saw nothing"),
                # no legacy rescue (the site's r5-1 flag), and main.py's artifact gates suppress
                # both collections off the non-posted `turn.reconsider`.
                _finish("error_dropped", error="delivery_failed")
                return None

            outcome = ("posted_asis" if current_draft.strip() == initial_draft.strip()
                       else "posted_revised")
            # The posted outcome asserts PHYSICAL Slack acceptance of the first surface,
            # nothing more (r3-16); site bookkeeping proceeds normally after the return.
            _finish(outcome, forced=forced)
            return delivered_ts
    except asyncio.CancelledError:
        # §4f: propagate. The runner created no surface of its own; a delivery Slack already
        # accepted may stand exactly as under today's cancellation semantics (r3-5). Outcome
        # `cancelled` is best-effort and at-most-once — a posted/failed outcome already stamped
        # wins.
        if getattr(turn, "reconsider", None) is None:
            _finish("cancelled")
        raise


__all__ = ["RECONSIDER_FUSE_PASSES", "intercept_stale_send", "reconsider_stale_draft",
           "build_reconsideration_request", "build_tooled_reconsideration_request",
           "reconsideration_item", "tooled_reconsideration_item", "draft_fence",
           "tools_used_summary", "hosted_reconsideration_tools", "redo_offerable",
           "newer_owner_ts", "post_reconsider_interim",
           "reviewed_through_map", "suppressing_ts_present", "select_reconsideration_model",
           "ChannelReconsiderSurface", "PreparedDecision", "surface_for"]
