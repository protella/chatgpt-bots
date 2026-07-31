"""Plumbing for the two-tier participation scenario harness (spec §15).

The scenarios themselves — and the expected-outcome table the owner reviews — live in
`test_participation_scenarios.py`. This module is only the machinery that puts a scenario in
front of the production code and reads back what happened.

Two tiers, two entry points:

* `run_wake_trial` drives the real `OpenAIClient.classify_wake` over production
  `SourceMessage` records and a steering block rendered by the production renderer. Nothing
  else: no stream, no tools, no destination.

* `run_responder_trial` forces admission and runs the production request assembler
  (`_assemble_channel_attempt` → `assemble_channel_request`) over a real serialized
  `ChannelStream`, then the production tool loop, then grades the OBSERVABLE OUTCOME.

WHAT IS REAL HERE, because a harness that fakes the thing under test proves nothing:
the serializer, the pinned tuple, the assembler, the system prompt, the restraint and terminal
contract paragraphs, every tool SCHEMA the channel surface exposes, the tool loop, the
`no_response_needed` and `set_reply_destination` executors (they only write turn state), and
the API call itself.

WHAT IS SUBSTITUTED: the Slack and database EFFECTS. Every other executor is replaced by a
recorder that writes the call into an in-memory sink and answers with a plausible success. So
a reaction or a memory write is observable and costs nothing outside this process. Nothing here
can reach Slack.

ONE EXECUTOR IS NOT SUBSTITUTED. `post_to_thread` runs the PRODUCTION
`execute_post_to_thread` with only its transport replaced — one `send_message` method. Everything
that decides whether the post is legitimate is therefore real: the same-thread rail, the
authorization check against the turn's frozen `trusted_thread_roots`, the effect lease, the
destination bookkeeping. A recorder that answered `{"ok": True}` would have graded the model on a
tool that cannot refuse anything, which is the one thing the cross-thread rows are for.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

from base_client import Message
from config import config
from message_processor import channel_request
from message_processor.channel_request import to_input_items
from message_processor.channel_steering import render_snapshot
from message_processor.channel_stream import ReceiptRec
from message_processor.routing_facts import stamp_routing_facts
from message_processor.turn_runtime import DESTINATION_CHANNEL, TurnRuntime
from tests.unit.channel_turn_harness import (build_stream, normalized, sidecars, thread_config)
from tool_registry import SURFACE_CHANNEL, ToolContext, ToolRegistry

# --------------------------------------------------------------------------- outcomes

SILENCE = "silence"
REACTION_ONLY = "reaction_only"
IN_THREAD_REPLY = "in_thread_reply"
CHANNEL_REPLY = "channel_reply"
CROSS_THREAD_POST = "cross_thread_post"
DETACHED_EFFECT = "detached_effect"
CONTRACT_VIOLATION = "contract_violation"

OUTCOMES = (SILENCE, REACTION_ONLY, IN_THREAD_REPLY, CHANNEL_REPLY, CROSS_THREAD_POST,
            DETACHED_EFFECT, CONTRACT_VIOLATION)

# Tools that post their own surface and are expected to leave the turn's words empty.
DETACHED_TOOLS = ("generate_image", "start_background_job")

# --------------------------------------------------------------------------- the room

BOT_ID = "UBOT"
BOT_NAME = "ChatGPT"
CHANNEL = "C1"
TEAM = "T1"


@dataclass(frozen=True)
class Say:
    """One message in the room, as prose plus the facts the serializer renders.

    `thread` is the root ts a reply hangs under; None is top-level. `kind` is the sender
    classification the whole pipeline branches on — "self" is us (it renders as `assistant`
    once its receipt is pinned), "other_bot" is another app, "human" is a person.
    """

    ts: str
    who: str
    text: str
    kind: str = "human"
    thread: Optional[str] = None


@dataclass(frozen=True)
class Room:
    """A channel's visible history for one scenario, plus who is in it."""

    says: Tuple[Say, ...]
    actors: Dict[str, str]                      # user/bot id -> display name
    num_members: int = 7
    topic: Optional[str] = None
    participation_level: str = "judicious"
    reply_in_channel: bool = True


def actor_id(room: Room, name: str) -> str:
    for user_id, display in room.actors.items():
        if display == name:
            return user_id
    raise KeyError(f"{name!r} is not in this room's actor map")


def build_room_stream(room: Room, *, through: Optional[str] = None):
    """A real ChannelStream over the room — the production serializer actually runs.

    `through` is H, and it TRIMS: a turn's stream can never contain a message that arrived after
    the trigger it is answering, because H pins at admission and is never refreshed. Passing the
    trigger's ts is what lets one room serve several scenarios at different points in its own
    history without any of them reading its future.

    Every `self` message gets a finalized receipt, which is what makes it render as `assistant`:
    the role comes from the receipt map, never from the sender id.

    THREAD ROOTS ARE SLACK'S, not ours. Slack stamps `thread_ts` on a reply, and on a root only
    once that root HAS a reply; a top-level message nobody answered carries none at all. The
    harness has to match that exactly, because `ChannelStream.trusted_thread_roots` — the allowlist
    `post_to_thread` authorizes against — is read from this field. Rooting every message at itself
    (which this did) made every top-level line a legal cross-thread target, so the executor's
    refusal could never fire and the cross-thread rows would have been graded against a tool that
    accepts anything.
    """
    high = through or max(say.ts for say in room.says)
    says = [say for say in room.says if float(say.ts) <= float(high)]
    answered = {say.thread for say in says if say.thread}
    messages = [
        normalized(say.ts, say.text, sender_id=actor_id(room, say.who),
                   sender_type=say.kind, channel_id=CHANNEL, team_id=TEAM,
                   thread_root_ts=(say.thread or (say.ts if say.ts in answered else None)),
                   raw_bot_name=(say.who if say.kind != "human" else None))
        for say in says
    ]
    receipts = tuple(
        ReceiptRec(ts=say.ts, state="finalized", turn_id=f"turn-{index}",
                   thread_root_ts=say.thread or say.ts)
        for index, say in enumerate(says) if say.kind == "self")
    return build_stream(messages, h=high, channel_id=CHANNEL, team_id=TEAM,
                        actor_map=tuple(sorted(room.actors.items())),
                        pinned_sidecars=sidecars(receipts=receipts))


def steering_snapshot(*, policy: Optional[str] = None, facts: Sequence[str] = (),
                      workspace_facts: Sequence[str] = ()):
    """A steering snapshot rendered by the PRODUCTION renderer, so the harness cannot invent a
    shape the live gate and responder never see."""
    rows: List[Dict[str, Any]] = []
    for index, fact in enumerate(facts, start=1):
        rows.append({"id": index, "content": fact, "scope": "channel"})
    for index, fact in enumerate(workspace_facts, start=len(rows) + 1):
        rows.append({"id": index, "content": fact, "scope": "workspace"})
    return render_snapshot({"content": policy} if policy else None, rows)


# --------------------------------------------------------------------------- tier 1

async def run_wake_trial(client: Any, sources: Sequence[Any],
                         steering: Any = None) -> Optional[bool]:
    """One gate decision. None means the gate produced nothing usable — not a decision, and the
    engine turns it into a decline, so the caller must not score it as a sleep."""
    return await client.classify_wake(
        sources=list(sources),
        channel_steering_text=(steering.gate_text if steering is not None else None))


# --------------------------------------------------------------------------- tier 2 plumbing

_PASSTHROUGH_EXECUTORS = ("no_response_needed", "set_reply_destination")

# What a recorded call answers with. Every one of these is honest about the harness rather than
# pretending to have data: a read tool that invented results would be grading the model on
# fiction. Absent from this table means `{"ok": True}`.
_RECORDED_RESULTS: Dict[str, Dict[str, Any]] = {
    "react_to_message": {"ok": True},
    "search_slack": {"ok": True, "matches": [], "message": "No matches."},
    "fetch_thread_messages": {"ok": True, "messages": []},
    "fetch_channel_history": {"ok": True, "messages": []},
    "fetch_pinned_messages": {"ok": True, "messages": []},
    "list_canvases": {"ok": True, "canvases": []},
    "list_channel_members": {"ok": True, "members": []},
    "generate_image": {"ok": True, "started": True},
    "start_background_job": {"ok": True, "started": True},
    "remember_fact": {"ok": True, "id": 1},
}


class _SchemaHost:
    """Enough of a Slack client for the real schema getters, and none of its I/O."""

    name = "Slack"
    bot_user_id = BOT_ID

    def __init__(self):
        self.workspace_emojis = SimpleNamespace(
            get_custom_emoji_names=lambda: ["shipit", "party-parrot"])

    def log_debug(self, *args, **kwargs):
        pass

    log_info = log_warning = log_error = log_debug


def _schema_host():
    from slack_client.channel_lookup_tool import SlackChannelLookupToolMixin
    from slack_client.history_tool import SlackHistoryToolMixin
    from slack_client.messaging import SlackMessagingMixin
    from slack_client.search_tool import SlackSearchToolMixin

    class Host(SlackMessagingMixin, SlackHistoryToolMixin, SlackChannelLookupToolMixin,
               SlackSearchToolMixin, _SchemaHost):
        pass

    return Host()


class FakeTransport:
    """Slack's send path, replaced at its narrowest point: one method, one accepted post.

    Bound as `self` to the PRODUCTION `execute_post_to_thread`, so everything that decides whether
    a post is legitimate still runs — the config gate, the same-thread rail, the authorization check
    against the turn's frozen root set, the effect lease, `note_destination_observed` and
    `mark_destination_committed`. What is gone is the socket.

    `posts` is what the room would actually be able to read afterwards, one entry per accepted post
    with the thread it landed in — so a fan-out and a wrong target are different findings.

    The one thing it does NOT reproduce is Slack's formatting pass: the real `Delivery.text` is the
    formatted string Slack accepted, and this reports the source markdown. Nothing here grades text,
    only where it went, so the difference is deliberate rather than overlooked.
    """

    name = "Slack"
    bot_user_id = BOT_ID

    def __init__(self):
        self.posts: List[Dict[str, Any]] = []

    def log_debug(self, *args, **kwargs):
        pass

    log_info = log_warning = log_error = log_debug

    async def send_message(self, channel_id: str, thread_id: str, text: str,
                           blocks=None, meta_out: Optional[dict] = None,
                           username: Optional[str] = None, lease: Any = None,
                           surface: str = "final_post", receipts: Any = None,
                           receipt_kind: Optional[str] = None,
                           on_first_accept=None) -> Optional[str]:
        from slack_client.messaging import Delivery

        # The stale-send guard is part of the production path and stays part of it: the real
        # send_message authorizes before it posts, and a lease that refused would have to refuse
        # here too.
        if lease is not None:
            lease.authorize(surface)
        posted_ts = f"9{len(self.posts) + 1:03d}.000100"
        # Fired BEFORE the return, exactly where the real send fires it — the executor records the
        # destination from this callback, so a harness that skipped it would grade a turn whose
        # ledger never mentioned the post.
        if on_first_accept is not None:
            on_first_accept(posted_ts)
        if meta_out is not None:
            meta_out["delivery"] = Delivery(first_ts=posted_ts, text=text, complete=True,
                                            parts_delivered=1, parts_total=1)
        self.posts.append({"channel_id": channel_id, "thread_ts": thread_id, "text": text,
                           "ts": posted_ts, "surface": surface})
        return posted_ts


def _real_post_to_thread(sink, transport: FakeTransport):
    """The production executor, unbound, with `transport` as its `self`."""
    from slack_client.messaging import SlackMessagingMixin

    async def run(ctx, args):
        result = await SlackMessagingMixin.execute_post_to_thread(transport, ctx, args)
        sink.append({"name": "post_to_thread", "arguments": args, "result": result})
        return result

    return run


def recording_registry(sink: List[Dict[str, Any]],
                       transport: Optional[FakeTransport] = None) -> ToolRegistry:
    """The production channel tool surface with its effects redirected into `sink`.

    Schemas, gates and registration order come from `SlackBot._build_tool_registry` — the model
    reads exactly the descriptions production writes. Only the executors change, and only for
    the tools that would reach Slack or the database: `no_response_needed` and
    `set_reply_destination` keep their real ones, because all they do is write turn state, and
    that state is precisely what the outcome is read from — and `post_to_thread` keeps its real one
    on top of `transport`, because a recorder cannot refuse an unauthorized target and refusing is
    half of what the cross-thread rows measure.
    """
    from slack_client.base import SlackBot

    real = SlackBot._build_tool_registry(_schema_host())
    out = ToolRegistry()
    for name, tool in real._tools.items():
        if name in _PASSTHROUGH_EXECUTORS:
            executor = _passthrough(sink, name, tool["executor"])
        elif name == "post_to_thread" and transport is not None:
            executor = _real_post_to_thread(sink, transport)
        else:
            executor = _recorder(sink, name)
        out._tools[name] = dict(tool, executor=executor)
    return out


def _passthrough(sink, name, executor):
    async def run(ctx, args):
        result = await executor(ctx, args)
        sink.append({"name": name, "arguments": args, "result": result})
        return result

    return run


def _recorder(sink, name):
    async def run(ctx, args):
        result = dict(_RECORDED_RESULTS.get(name, {"ok": True}))
        sink.append({"name": name, "arguments": args, "result": result})
        return result

    return run


def processor_host():
    """A processor stand-in bound to the REAL assembler, prompt builder and tools array."""
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.utilities import MessageUtilitiesMixin

    host = MagicMock()
    for attr in ("_assemble_channel_attempt", "_channel_prepared_tools",
                 "_materialize_request_tools", "_build_tools_array", "_get_tool_registry"):
        setattr(host, attr, getattr(TextHandlerMixin, attr).__get__(host))
    for attr in ("_get_system_prompt", "_build_time_suffix_context",
                 "_build_message_with_documents"):
        setattr(host, attr, getattr(MessageUtilitiesMixin, attr).__get__(host))
    # Nothing is in flight in a scenario, and both notes are DB reads in production.
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    return host


def platform_client(registry: ToolRegistry):
    client = MagicMock()
    client.name = "Slack"
    client.bot_user_id = BOT_ID
    client.tool_registry = registry
    client.workspace_emojis = SimpleNamespace(
        get_custom_emoji_names=lambda: ["shipit", "party-parrot"])
    return client


# --------------------------------------------------------------------------- one responder trial

@dataclass
class TrialResult:
    outcome: str
    text: str
    effects: List[str] = field(default_factory=list)
    silence_reason: Optional[str] = None
    destination: Optional[str] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)
    detail: Optional[str] = None
    # Cross-thread facts, for the five assertions below. `posts` is what LANDED (the transport
    # accepted it); `post_attempts` is every target the model aimed at, refused ones included, so a
    # fan-out that the executor blocked is still visible; `trusted_roots` is the allowlist that was
    # actually in force, which is what makes a pass mean something.
    posts: List[Dict[str, Any]] = field(default_factory=list)
    post_attempts: List[Dict[str, Any]] = field(default_factory=list)
    trusted_roots: Optional[frozenset] = None


def _committed(calls, name) -> bool:
    return any(c["name"] == name and (c["result"] or {}).get("ok") for c in calls)


def _post_attempts(calls) -> List[Dict[str, Any]]:
    return [{"thread_ts": (c["arguments"] or {}).get("thread_ts"),
             "ok": bool((c["result"] or {}).get("ok")),
             "error": (c["result"] or {}).get("error")}
            for c in calls if c["name"] == "post_to_thread"]


def cross_thread_failures(result: TrialResult, *, target: str) -> List[str]:
    """The five things one correct cross-thread turn has to satisfy, each as its own finding.

    Every one of them is a way a turn can post into the right thread and still be wrong, and none
    of them is visible in the OUTCOME LABEL — `cross_thread_post` is returned by a turn that posted
    into a stranger's thread, posted twice, or pasted the answer in both places.

    "Zero origin prose" is prose, not every surface: a REACTION in the origin thread is allowed
    (the conduct paragraph offers it as the ending), and the owner's 2026-07-29 ruling permits
    reactions generally. What may not happen is WORDS in the thread the turn started in.
    """
    failures: List[str] = []
    # 0. The pass must not be vacuous. If authorization was never in force — no stream, or a
    #    stand-in that degraded to legacy — then "the executor accepted the target" is not a fact
    #    about the target at all.
    if result.trusted_roots is None:
        failures.append("authorization was NOT in force on this turn — a pass proves nothing")
    elif target not in result.trusted_roots:
        failures.append(f"the scenario's target {target} is not in the turn's trusted roots "
                        f"{sorted(result.trusted_roots)} — the room cannot reach it")
    # 1. Exactly one post.
    if len(result.posts) != 1:
        failures.append(f"{len(result.posts)} post(s) landed, expected exactly 1")
    # 2. In the expected thread.
    landed = sorted({p["thread_ts"] for p in result.posts})
    if landed and landed != [target]:
        failures.append(f"landed in {landed}, expected [{target!r}]")
    # 3. No extra target, including ones the executor refused — aiming at a second thread is
    #    fan-out whether or not it was allowed through.
    aimed = sorted({a["thread_ts"] for a in result.post_attempts})
    if aimed and aimed != [target]:
        failures.append(f"aimed at {aimed}, expected [{target!r}]")
    # 4. Authorization actually said yes.
    refused = [(a["thread_ts"], a["error"]) for a in result.post_attempts if not a["ok"]]
    if refused:
        failures.append(f"the executor refused: {refused}")
    # 5. Nothing was said in the origin.
    if result.text:
        failures.append(f"origin prose was delivered: {result.text[:120]!r}")
    return failures


def classify_outcome(result: Dict[str, Any], calls: List[Dict[str, Any]],
                     turn: TurnRuntime) -> Tuple[str, Optional[str]]:
    """The turn's observable outcome, and a note when the label is a violation.

    Read from what the turn DID, never from what the text says about itself. The ordering picks
    the most consequential surface: a cross-thread post is the loudest thing a turn can do, a
    detached job owns the turn's output even when the model also typed an ack (production drops
    that ack), and words in the origin conversation come next.
    """
    text = (result.get("text") or "").strip()
    silent = result.get("terminal_action") == "no_reply"

    if text and silent:
        return CONTRACT_VIOLATION, "declared silence with visible reply text"
    if _committed(calls, "post_to_thread"):
        return CROSS_THREAD_POST, None
    if any(_committed(calls, name) for name in DETACHED_TOOLS):
        return DETACHED_EFFECT, None
    if text:
        return ((CHANNEL_REPLY if turn.reply_destination == DESTINATION_CHANNEL
                 else IN_THREAD_REPLY), None)
    if _committed(calls, "react_to_message"):
        return REACTION_ONLY, None
    if silent:
        return SILENCE, None
    return CONTRACT_VIOLATION, "no words, no reaction and no declared silence"


async def run_responder_trial(openai_client: Any, *, room: Room, trigger: Say,
                              steering: Any = None,
                              silence_capable: bool = True,
                              addressed: bool = False,
                              web_search: bool = False,
                              code_interpreter: bool = False,
                              wake_source: Optional[str] = "channel_activity",
                              model: Optional[str] = None) -> TrialResult:
    """One responder turn, graded on what it did.

    `trigger` must be one of the room's own messages — the thing the coordinates block names as
    this turn's trigger. Admission is FORCED: the gate does not run, so a scenario measures the
    responder's own restraint rather than the gate's.
    """
    sink: List[Dict[str, Any]] = []
    transport = FakeTransport()
    registry = recording_registry(sink, transport)
    client = platform_client(registry)
    host = processor_host()

    trigger_id = actor_id(room, trigger.who)
    origin_thread = trigger.thread or trigger.ts
    message = Message(text=trigger.text, user_id=trigger_id, channel_id=CHANNEL,
                      thread_id=origin_thread,
                      metadata={"ts": trigger.ts, "username": trigger.who,
                                "user_real_name": trigger.who, "sender_type": trigger.kind,
                                "mentioned_self": addressed})
    stamp_routing_facts(message, gate_required=not addressed,
                        silence_capable=silence_capable, addressed=addressed,
                        ts=trigger.ts, thread_ts=origin_thread)
    if not addressed:
        message.metadata["gate_woke"] = True
    turn = TurnRuntime.for_message(
        message, channel_post_allowed=bool(room.reply_in_channel and trigger.thread is None))

    cfg = thread_config(model=model or config.gpt_model,
                        temperature=config.default_temperature,
                        max_tokens=config.default_max_tokens,
                        reasoning_effort=config.default_reasoning_effort,
                        verbosity=config.default_verbosity,
                        enable_web_search=web_search,
                        enable_code_interpreter=code_interpreter)
    # The tool exposure and its matching contract paragraph, resolved exactly where base.py
    # resolves them, then pinned on the turn the way the admission estimate pins them.
    turn.channel_prepared = (*host._materialize_request_tools(
        client, cfg, message, tools_disabled=False, turn=turn, surface=SURFACE_CHANNEL), None)

    stream = build_room_stream(room, through=trigger.ts)
    root_author = next((s.who for s in room.says if s.ts == origin_thread), None)
    ctx = channel_request.ChannelTurnContext(
        stream=stream,
        steering=steering if steering is not None else steering_snapshot(),
        thread_config=cfg, channel_id=CHANNEL, team_id=TEAM, trigger_ts=trigger.ts,
        origin_thread_ts=origin_thread, trigger_text=trigger.text,
        canonical_files=channel_request.canonical_files_from_stream(stream),
        origin_participants={actor_id(room, s.who): s.who for s in room.says
                             if (s.thread or s.ts) == origin_thread
                             and float(s.ts) <= float(trigger.ts)},
        requester=channel_request.RequesterFacts(
            user_id=trigger_id, real_name=trigger.who, sender_type=trigger.kind,
            is_root_author=(root_author == trigger.who)),
        channel_info={"participation_level": room.participation_level,
                      "reply_in_channel": room.reply_in_channel,
                      **({"topic": room.topic} if room.topic else {})},
        num_members=room.num_members, wake_source=wake_source)
    turn.channel_stream = stream
    turn.channel_turn_context = ctx
    turn.stream_build_present = True
    turn.H = stream.pinned.H

    request, _prepared, reg, request_config, _no_reply, _suffix, _container = (
        await host._assemble_channel_attempt(client, message, SimpleNamespace(), turn, cfg,
                                             cfg["model"],
                                             thread_key=f"{CHANNEL}:{origin_thread}"))
    # Resolved by the PRODUCTION helper, off the turn — the harness must not be able to hand the
    # executor a wider allowlist than a real turn would get.
    from message_processor.handlers.text import _trusted_thread_roots

    trusted = _trusted_thread_roots(turn)
    tool_ctx = ToolContext(channel_id=CHANNEL, thread_ts=origin_thread, trigger_ts=trigger.ts,
                           user_id=trigger_id, client=client, db=None, is_dm=False,
                           turn=turn, message=message, thread_config=request_config,
                           canonical_files=ctx.canonical_files,
                           requester_is_human=(trigger.kind == "human"),
                           trusted_thread_roots=trusted,
                           structural_change_authorized=bool(
                               request_config.get("_structural_change_authorized")))
    result = await openai_client.create_text_response_with_tool_loop(
        messages=to_input_items(request), tools=request.tools, registry=reg,
        tool_context=tool_ctx, free_tools=("set_reply_destination",), model=cfg["model"],
        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
        system_prompt=request.instructions, reasoning_effort=cfg["reasoning_effort"],
        verbosity=cfg["verbosity"], store=False, prompt_cache_key=request.prompt_cache_key,
        layout="channel")

    outcome, detail = classify_outcome(result, sink, turn)
    return TrialResult(outcome=outcome, text=(result.get("text") or "").strip(),
                       effects=[c["name"] for c in sink],
                       silence_reason=result.get("silence_reason"),
                       destination=turn.reply_destination, calls=sink, detail=detail,
                       posts=list(transport.posts), post_attempts=_post_attempts(sink),
                       trusted_roots=trusted)


# --------------------------------------------------------------------------- running trials

_TRANSPORT_ERRORS = (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)
_TRANSPORT_MODULES = ("openai", "httpx", "httpcore")


def is_transport_error(exc: BaseException) -> bool:
    """Did the provider fail, rather than the code under test?

    The distinction is the same one `classify_wake` already draws by returning None: a request
    that never completed is not a decision, and scoring it as one would file a provider outage
    under the model's name. A harness or production bug — a TypeError, a missing attribute —
    must still fail the run loudly, so only transport shapes qualify.
    """
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    module = (type(exc).__module__ or "").split(".")[0]
    return module in _TRANSPORT_MODULES


async def gather_trials(coroutine_factories, *, concurrency: int = 6,
                        retries: int = 1) -> List[Any]:
    """Run every trial with a bounded fan-out, retrying provider failures.

    Trials are independent — separate turns, separate registries, separate sinks — so the only
    reason to bound them is the provider. A transport failure is retried `retries` times and then
    returned as the exception, for the caller to report and exclude; anything else is returned
    immediately, because a bug should not be given a second chance to look intermittent.
    """
    limiter = asyncio.Semaphore(concurrency)

    async def guarded(factory):
        async with limiter:
            for attempt in range(retries + 1):
                try:
                    return await factory()
                except Exception as exc:                 # noqa: BLE001 — a trial, not the suite
                    if attempt >= retries or not is_transport_error(exc):
                        return exc
        return None                                      # unreachable

    return await asyncio.gather(*(guarded(f) for f in coroutine_factories))
