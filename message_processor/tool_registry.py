"""Platform-agnostic local-tool registry for the Responses API function-call loop (Phase A).

The registry maps function-tool schemas to async executors. The loop
(``openai_client/api/tool_loop.py``) collects ``function_call`` items from a response,
dispatches them here, and feeds ``function_call_output`` items back to the model.

Executors receive a ``ToolContext`` (per-request platform state) and the parsed
arguments dict, and return a JSON-serializable dict. They must never raise to the
loop: dispatch wraps every failure (unknown tool, bad args, timeout, exception) into
an ``{"ok": False, "error": ...}`` result so a tool problem degrades the answer, not
the response.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Mapping, Optional

from config import config
from logger import setup_logger

if TYPE_CHECKING:  # a module-scope import would cycle through slack_client.base
    from message_processor.turn_runtime import AuthorizedEditTarget

logger = setup_logger(name="slack_bot.ToolRegistry")

# The two tool surfaces (spec §3a). "dm" is the legacy, per-request-dynamic one; "channel" is
# the cache-stable one, where a tool's shape is a function of (channel, channel config, bot
# version) and nothing else.
SURFACE_DM = "dm"
SURFACE_CHANNEL = "channel"


@dataclass
class SandboxHolder:
    """W3: the turn's addressable code-interpreter container, held BY REFERENCE.

    Turns now start on `{"type": "auto"}` — no client-side `containers.create` on the critical
    path — so the id is not known when the context is built. It arrives one of two ways, and both
    happen mid-turn: the model runs code and the container id shows up in the artifacts sink
    (adoption), or a bridge tool needs somewhere to push bytes and mints one (`ensure`).

    It has to be an OBJECT, not a string field. A round's calls run on shallow per-call copies of
    the ToolContext, so a string written by `mount_file` would land in that call's private copy
    and be invisible to its siblings and to the loop that must pin the id into the next round's
    declaration. One holder, shared by every copy, is the only version of the answer.

    `manager` is a `message_processor.containers.ContainerManager` — duck-typed on purpose, since
    message_processor imports this module and the reverse would cycle. Absent (a background job's
    hand-built context, a test) simply means no bridge: `ensure` answers None and the executor
    says so honestly.
    """
    container_id: Optional[str] = None
    manager: Any = None
    thread_key: Optional[str] = None

    async def ensure(self) -> Optional[str]:
        """The addressable container, minting one if this turn started on `auto`.

        Single-flight per thread key inside the manager, so two bridge calls in the same round
        produce ONE container: the winner fills this holder and the losers read it back.
        """
        if self.container_id:
            return self.container_id
        if self.manager is None or not self.thread_key:
            return None
        return await self.manager.bridge_container(self.thread_key, self)

    def replace(self, container_id: Optional[str]) -> None:
        """Point the turn at a DIFFERENT container (`reset_sandbox`).

        For the same reason the holder is an object at all: the per-call copies and the tool loop
        share THIS instance, so the swap has to happen in it. Writing a new id anywhere else would
        leave the loop pinning the abandoned sandbox into the next round's declaration, and the
        model would be told to open files in a container it can no longer reach.
        """
        self.container_id = container_id or None


@dataclass
class ToolContext:
    """Per-request state passed to every executor (built by the message processor)."""
    channel_id: Optional[str] = None
    thread_ts: Optional[str] = None
    trigger_ts: Optional[str] = None      # ts of the message we're answering
    # The participation gate's id for THIS attempt, when the turn came from the gate at all
    # (None for mentions, DMs and direct thread continuations). Tools that place a reaction
    # record it so the row joins to the decision that produced the turn — and so the ledger
    # can suppress rows for turns that had no gate decision behind them.
    attempt_id: Optional[str] = None
    action_token: Optional[str] = None    # from the triggering Slack event (search API)
    user_id: Optional[str] = None         # triggering user (provenance for memory writes)
    client: Any = None                    # platform client (e.g. SlackBot)
    db: Any = None
    is_dm: bool = False
    # F30: exposed by the tool loop so a detached job (start_deep_research) can snapshot the
    # CURRENT turn's full conversation by deep-copying `current_input` at call time. The
    # developer prompt rides separately in `system_prompt`; `model` is the thread's model.
    processor: Any = None                 # MessageProcessor (openai_client, scheduling, thread_manager)
    current_input: Optional[List[Any]] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    # F30.1: set True by execute_start_background_job on a successful start, so the turn's
    # finalizer (handlers/text.py) can DROP the model's ack reply — the research status card
    # the job posts is the acknowledgment. Read back from the same context the loop shares.
    background_job_started: bool = False
    # F34: image tools. `thread_config` carries the resolved per-user settings (the image
    # MODEL is read from here and is NOT model-selectable). `container_id` is the SAME
    # persistent code-interpreter container already placed in the tools array — the image
    # asset tool must never resolve its own, or it would mount bytes into a container the
    # model cannot see. `image_catalog` is this turn's allowlist of editable images, so an
    # invented image id is rejected rather than silently editing the wrong picture.
    thread_config: Optional[Dict[str, Any]] = None
    container_id: Optional[str] = None
    # F15: the SAME list the API layer records dead container ids into (container_gone_sink).
    # A persistent container confirmed alive at turn start can idle-expire between tool rounds;
    # when it does, `_create_with_container_recovery` retries that one API call against a fresh
    # ephemeral sandbox and appends the dead id here. `container_id` above still points at the
    # corpse, so a within-round mount_file / create_image_asset would push bytes into a
    # container the model can no longer see. Executors call `container_recycled()` and fail
    # fast instead. Shared by reference, so the append the API makes is visible here with no
    # extra plumbing between rounds.
    container_gone_sink: Optional[List[str]] = None
    image_catalog: Optional[List[Dict[str, Any]]] = None
    # `view_image` stages re-attached EARLIER images here; the tool loop drains them into a
    # user-role message so the model actually sees the pixels on the next round. Shared by
    # reference (like container_gone_sink) so no extra plumbing is needed between rounds.
    # USER role, never developer: these are untrusted user-supplied image bytes.
    pending_vision_parts: Optional[List[Dict[str, Any]]] = None
    # Urls of the images whose pixels ALREADY ride this turn. The image catalog is built after
    # the answered message's attachments are persisted, so those images are IN it — without this,
    # view_image would happily re-attach a picture already in front of the model.
    current_image_urls: Optional[List[str]] = None
    # Set True by a detached image generation, so the finalizer can drop the model's ack
    # reply the same way deep research does — the posted image IS the acknowledgment.
    image_generation_started: bool = False
    # Paths mounted into the container by create_image_asset this turn. If the turn ends
    # having published nothing, these are rescued to the thread rather than vanishing with
    # the container (see handlers/text.py) — a silent no-output turn is the worst failure.
    sandbox_image_assets: Optional[List[Dict[str, Any]]] = None
    # F35: mount_file. `thread_files` is this turn's allowlist of mountable files (images AND
    # documents behind one opaque id space) — the same authorization rule as `image_catalog`:
    # only ids we advertised resolve. `mounted_files` records what actually went into the
    # sandbox, so (a) a second mount of the same file is a no-op, and (b) the artifact
    # publisher can refuse to post a user's own file back at them, even byte-copied.
    thread_files: Optional[List[Dict[str, Any]]] = None
    mounted_files: Optional[List[Dict[str, Any]]] = None
    # Participation redesign: True only when a HUMAN authored the message AND this turn genuinely
    # reached the responder (handlers.text `_structural_change_authorized`, from the routing facts
    # sender_type / gate_required / gate_woke — never the loose name-hit regex). It gates the
    # structural set_channel_participation tool; the canvas-delete tool has its own parallel,
    # stricter signal, `_canvas_delete_authorized`. Left False (fail-closed), so a non-human
    # sender, an unclassified sender, or a message that needed the gate and never woke it — the
    # injection / hallucination / "being talked about ≠ talked to" vector — can never flip
    # channel settings even if the model emits the call.
    structural_change_authorized: bool = False
    # The parallel, stricter signal for the irreversible canvas-delete tool: a HUMAN sender AND a
    # genuine current-message address (a real <@bot> mention, or a DM). It used to live only in
    # the tool's per-turn schema gate, which the channel surface structurally ignores — so the
    # authorization moved here, where the executor checks it on BOTH surfaces. Fail-closed False.
    canvas_delete_authorized: bool = False
    # F38: the turn's presentation + work-claim state (message_processor.turn_runtime).
    # A slow local tool calls `await ctx.turn.claim_work(ctx.client, ctx.message)` once its
    # arguments and capacity checks have PASSED and immediately before the slow part starts —
    # never on entry, or a rejected call would flash a 👀 it is about to retract. A tool that
    # posts its own surface (a background job's card, a detached image) sets
    # `ctx.turn.visible_action_committed = True` so the turn counts as having produced output
    # even though its Response carries no text.
    turn: Any = None
    message: Any = None
    # --- user-scoped channel authorization (2026-07) ---
    # Every channel-read tool must prove that BOTH the bot AND `user_id` are in the target
    # conversation before any content is returned. Two fields serve that check:
    #
    # `origin_membership_attested` is a NON-INHERITABLE attestation that this context was built
    # from a genuine Slack-delivered HUMAN message event whose channel is `channel_id` and whose
    # sender is `user_id` — i.e. Slack itself just proved the requester is in that conversation.
    # It is set ONLY in handlers/text.py from origin markers stamped at the real event entry
    # points, and it authorizes skipping the membership lookup for THAT ONE conversation. It must
    # stay False on every synthetic/replayed/detached context (settings' welcome replay, edit
    # re-dispatch, background jobs, scheduled work), because `channel_id` alone is forgeable:
    # a stale context can name a channel the user has since left. Default False = full check.
    #
    # `channel_access_memo` is a REQUEST-SCOPED single-flight memo (decision futures + the
    # requester's conversation set) that dies with this context. Deliberately NOT a TTL/process
    # cache: a positive answer must never outlive the request that earned it, or someone removed
    # from a private channel keeps reading it. Its only job is to coalesce the identical
    # membership walks that a round's parallel tool calls (dispatch_all) would each run.
    #
    # `requester_is_human` says `user_id` belongs to a PERSON, not another app's bot user.
    # Bot-authored messages are deliberately processed (bot<->bot works), and a bot posting
    # through its own token carries a genuine U… id, so `user_id` alone cannot carry the
    # policy. Defaults False and is set only where `user_id` is set from a classified message,
    # so the two can never disagree; a hand-built context has neither and is denied twice over.
    origin_membership_attested: bool = False
    requester_is_human: bool = False
    channel_access_memo: Optional[Dict[Any, Any]] = None
    # Every file the pinned channel window saw, `file_id -> {filename, mime_type, url_private,
    # kind, channel_id, message_ts, thread_root_ts}`. Built from the stream's own fetch snapshot
    # plus the live payloads of any cohort member Slack had not propagated yet, so a document
    # dropped in ANOTHER thread of this channel is readable on the turn it is asked about rather
    # than one turn later, once a `documents` row exists.
    #
    # AUTHORIZATION, not convenience: it names only ids the stream actually rendered, so it can
    # widen what `read_document` reaches without widening it to arbitrary ids. Empty on DM turns,
    # which have no channel window and keep the `documents`-table path verbatim.
    canonical_files: Optional[Dict[str, Dict[str, Any]]] = None
    # The thread roots this turn's stream actually SHOWED the model, frozen at pin time — the
    # allowlist `post_to_thread` authorizes a cross-thread target against. It is deliberately
    # three-valued in spirit: None means "no channel stream" (a DM, a background agent, a
    # hand-built context) and keeps the legacy behavior verbatim; a frozenset — including an EMPTY
    # one — means channel enforcement, because a turn whose stream rendered no thread has no
    # thread it may post into. It comes from the SERIALIZED stream, not the fetch snapshot: what
    # the model was shown is what it may act on, and the excluded/in-flight messages a snapshot
    # still carries were withheld from it on purpose.
    trusted_thread_roots: Optional[frozenset] = None
    # EDIT §2. The exact own messages this turn may edit, keyed by message_ts. An edit is
    # authorized iff the target ts is a key. UNLIKE the roots above there is no None sentinel:
    # a missing or malformed source resolves to the EMPTY mapping, never None, because "no
    # stream" must fail an edit closed where it keeps a legacy post path open. Restamped from
    # the turn once at the top of every dispatch round, so a read and an edit in the SAME round
    # can never authorize each other.
    authorized_edit_targets: Mapping[str, "AuthorizedEditTarget"] = field(default_factory=dict)
    # --- per-call identity (spec §6 d2) ---------------------------------------------------
    # These four are set on a SHALLOW PER-CALL COPY of the context, never on the shared one: a
    # round's calls run CONCURRENTLY (dispatch_all gathers them), so a single mutable "current
    # call" field would name whichever sibling wrote to it last. Everything above is shared BY
    # REFERENCE across those copies — the vision staging, the mounted-file record, the membership
    # memo — so the bookkeeping stays in one place and only the identity is per-call.
    tool_call_id: Optional[str] = None
    # THE resolved bound for this call and the monotonic instant it expires, stamped once at
    # dispatch. Generic tools and the synchronous image tools resolve very different values, so
    # anything downstream that needs the bound reads THIS rather than re-deriving one from config.
    tool_timeout: Optional[float] = None
    tool_deadline: Optional[float] = None
    # This call's single-flight entry (message_processor.turn_runtime.ToolFlight). An executor
    # calls `tool_flight.mark_launched()` immediately before it issues the side-effect request it
    # cannot take back. None when the call carries no id, which is the legacy no-dedup path.
    tool_flight: Any = None
    # W3: the turn-owned container reference (see SandboxHolder). It outranks `container_id`,
    # which now only carries a container that was already bound when the context was built —
    # a background job's own sandbox, a hand-built context. On a chat turn the holder starts
    # empty and is filled by adoption or by the first bridge call.
    sandbox: Optional[SandboxHolder] = None

    def sandbox_container_id(self) -> Optional[str]:
        """The addressable container this turn is ACTUALLY using, without creating one."""
        holder = self.sandbox
        if holder is not None and holder.container_id:
            return holder.container_id
        return self.container_id

    async def ensure_sandbox(self) -> Optional[str]:
        """…and mint one if the turn started on `auto`. None means there is nowhere to push to.

        W3: `mount_file` / `create_image_asset` are now offered on auto turns, because refusing
        them for want of an id the turn deliberately did not pay for would cost the feature to
        save the latency. They call this instead; the tool loop pins whatever it produces into
        the next round's declaration, so the model can see what was left there.
        """
        return await self.sandbox.ensure() if self.sandbox is not None else self.container_id

    def container_recycled(self) -> bool:
        """F15: True when this turn's container died mid-turn and was recorded dead.

        A tool that pushes bytes into the persistent sandbox (mount_file, create_image_asset)
        checks this before uploading — a recycled container is a dead drop the model cannot
        read back, so failing fast is honest where a silent write is not."""
        cid = self.sandbox_container_id()
        return bool(cid and self.container_gone_sink and cid in self.container_gone_sink)


Executor = Callable[[ToolContext, Dict[str, Any]], Awaitable[Dict[str, Any]]]


# Containers the executors SHARE across a round. They are created here, before the per-call copies
# exist, so no executor ever has to install one — an assign-if-None inside an executor would write
# into its own copy, and two siblings that each installed a list would keep one of them.
_SHARED_CONTAINERS = (("pending_vision_parts", list), ("sandbox_image_assets", list),
                      ("mounted_files", list), ("channel_access_memo", dict))

# Monotonic flags an executor sets to tell the HANDLER what happened (the ack reply it must drop,
# the surface a detached producer owns). They are set on the per-call copy, so they are adopted
# back onto the shared context when the call ends — including when it ends by raising, which is
# how a suppressed cross-thread post still reports the image it had already started.
_PER_CALL_FLAGS = ("background_job_started", "image_generation_started")


def _ensure_shared_containers(ctx: Any) -> None:
    for name, factory in _SHARED_CONTAINERS:
        try:
            if getattr(ctx, name, None) is None:
                setattr(ctx, name, factory())
        except Exception:  # noqa: BLE001 — a read-only stand-in context keeps today's behavior
            continue


def _adopt_per_call_flags(parent: Any, call_ctx: Any) -> None:
    if parent is call_ctx:
        return
    for name in _PER_CALL_FLAGS:
        try:
            if getattr(call_ctx, name, False) and not getattr(parent, name, False):
                setattr(parent, name, True)
        except Exception:  # noqa: BLE001
            continue


def _call_fingerprint(name: str, args: Dict[str, Any]) -> str:
    """What a call id is a name FOR. Two calls with one id and different fingerprints are two
    different questions, and the second must never be answered with the first one's result."""
    try:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        payload = repr(args)
    return hashlib.sha256(f"{name}\x00{payload}".encode("utf-8")).hexdigest()


class ToolRegistry:
    """Name → (schema, executor, enabled-gate). Gates are evaluated per request."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        schema: Any,
        executor: Executor,
        enabled: Optional[Callable[[dict], bool]] = None,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        dynamic: bool = False,
        channel_schema: Any = None,
        channel_enabled: Optional[Callable[[dict], bool]] = None,
    ) -> None:
        """Register a tool.

        ``schema`` is either a static dict or a FACTORY ``(thread_config) -> dict``, for a
        tool whose shape depends on the request (F34: the image tools' legal option values
        differ by the user's selected image model, and their description names the user's
        saved defaults). A factory must be given an explicit ``name``, since there is no
        dict to read it from, and must declare ``dynamic=True`` — per-request shape is now a
        DM-surface-only privilege, and an undeclared factory would silently reintroduce it.

        ``channel_schema`` / ``channel_enabled`` are the channel surface's answers to the same
        two questions. The channel schema must not vary with the request (A3a's ``_static``
        variants accept and ignore the config so the registry can call them uniformly), and
        ``channel_enabled`` may read only channel-config-stable facts. A tool with neither
        keeps its static base schema and is exposed unconditionally there.
        """
        if callable(schema):
            if not dynamic:
                raise ValueError(
                    f"Tool {name or '<unnamed>'}: a callable schema is per-request and must be "
                    "registered with dynamic=True")
            if not name:
                raise ValueError("A schema factory must be registered with an explicit name")
        else:
            name = schema.get("name")
            if not name:
                raise ValueError("Tool schema must have a 'name'")
        # timeout=None → the shared config.tool_call_timeout. A tool with a heavier
        # worst case (e.g. read_document, which may download + render + OCR a scan)
        # sets its own longer bound so the generic 20s cap can't abort it.
        self._tools[name] = {"schema": schema, "executor": executor,
                             "enabled": enabled, "timeout": timeout,
                             "dynamic": bool(dynamic),
                             "channel_schema": channel_schema,
                             "channel_enabled": channel_enabled}

    def schemas(self, thread_config: Optional[dict] = None,
                surface: str = SURFACE_DM) -> List[Dict[str, Any]]:
        """Schemas of the tools enabled for this request (a failing gate hides the tool).

        ``surface="dm"`` is the legacy behavior, verbatim: per-request ``enabled`` gates and
        per-request schema factories.

        ``surface="channel"`` reads the channel pair instead — ``channel_schema`` (falling back
        to the static base schema) gated by ``channel_enabled`` alone. Per-turn ``enabled``
        callables are structurally ignored there: a gate that varies within a channel is exactly
        the cache fork §3a exists to remove, so authorization moves into the executors.

        A gate or factory that raises hides its tool rather than failing the turn — fail-closed
        omission, logged with the tool's name so a broken schema is not silently invisible.
        """
        out = []
        cfg = thread_config or {}
        channel = surface == SURFACE_CHANNEL
        for name, tool in self._tools.items():
            try:
                gate = tool["channel_enabled"] if channel else tool["enabled"]
                if gate is not None and not gate(cfg):
                    continue
                schema = tool["schema"]
                if channel and tool["channel_schema"] is not None:
                    schema = tool["channel_schema"]
                if callable(schema):
                    schema = schema(cfg)
                    if not schema:
                        continue
                out.append(schema)
            except Exception as e:  # noqa: BLE001 — a broken schema costs its tool, not the turn
                logger.error(
                    f"tool schema omitted: name={name} surface={surface} error={e}",
                    exc_info=True)
                continue
        return out

    def has_tools(self, thread_config: Optional[dict] = None,
                  surface: str = SURFACE_DM) -> bool:
        return bool(self.schemas(thread_config, surface=surface))

    async def dispatch(self, ctx: ToolContext, name: str, arguments: Any,
                       call_id: Optional[str] = None) -> Dict[str, Any]:
        """Run one tool call.

        Returns a result for everything a tool can do wrong — an unknown name, bad arguments, a
        timeout, a bug — so a broken tool never kills the response. The ONE exception it lets
        through is `StaleSendSuppressed`: a guarded tool declining to post because the
        conversation moved on is control flow, and reporting it as a tool error would have the
        model retry the post the guard just refused.

        ``call_id`` is the model's own id for this call, and with a turn to hold it, it is the
        in-turn retry key: the FIRST dispatch owns the work and every later one with the same id
        receives that same work's outcome instead of doing it again. Absent, there is nothing to
        dedup on — distinct calls must never collapse onto one another just because neither
        carried an id — but the turn still takes ownership of the WORK, under an anonymous
        flight, so an id-less call is drained, cancelled and revoked with every other one. With
        no turn able to hold a flight at all, the legacy bounded-and-cancelled path is kept
        verbatim."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": "unknown_tool", "message": f"No tool named '{name}'."}

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "bad_arguments", "message": "Arguments were not valid JSON."}
        else:
            args = arguments or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "bad_arguments", "message": "Arguments must be a JSON object."}

        timeout = tool.get("timeout")
        if timeout is None:
            timeout = config.tool_call_timeout

        turn = getattr(ctx, "turn", None)
        if not call_id and turn is not None and hasattr(turn, "open_anonymous_tool_flight"):
            # No id to key on, so no dedup — but the turn still has to OWN this call, or a
            # sibling's suppression propagating out of the round leaves it running with nothing
            # to drain, cancel or revoke it, and it can post after the receipts have settled.
            flight = self._open_anonymous_flight(turn, name=name, timeout=timeout)
            if flight is None:
                logger.error(f"anonymous tool flight could not be opened: name={name} — "
                             "refusing to run the call untracked")
                return {"ok": False, "error": "flight_unavailable",
                        "message": (f"Tool '{name}' could not be started safely. Nothing ran — "
                                    "try the call again.")}
            return await self._fly_call(tool, ctx, args, name=name, flight=flight,
                                        created=True, turn=turn, call_id=None)
        if call_id and turn is not None and hasattr(turn, "open_tool_flight"):
            opened = self._open_flight(turn, name=name, args=args, call_id=call_id,
                                       timeout=timeout)
            if opened is None:
                # A REAL turn and a REAL call id, and the protection could not be opened. Running
                # the tool anyway is unprotected execution of exactly the calls this key exists
                # to make once — chosen at the moment the bookkeeping is known to be broken. It
                # fails closed instead: the model is told plainly and nothing irreversible runs.
                logger.error(
                    f"tool flight could not be opened: id={call_id} name={name} — refusing to "
                    "run the call unprotected")
                return {"ok": False, "error": "flight_unavailable",
                        "message": (f"Tool '{name}' could not be started safely. Nothing ran — "
                                    "try the call again.")}
            else:
                flight, created = opened
                if flight is None:
                    # The same id, describing something else. Serving the first result here would
                    # answer a question nobody asked; re-running under a key already spent would
                    # be the double-dispatch this whole mechanism exists to prevent.
                    logger.error(
                        f"tool call id reused for a different call: id={call_id} name={name}")
                    return {"ok": False, "error": "duplicate_call_id",
                            "message": ("This call re-used the id of a different call already "
                                        "made this turn. Nothing was run — make the call again.")}
                return await self._fly_call(tool, ctx, args, name=name, flight=flight,
                                           created=created, turn=turn, call_id=call_id)

        return await self._run_bounded(tool, ctx, args, name=name, timeout=timeout)

    async def _run_bounded(self, tool: Dict[str, Any], ctx: Any, args: Dict[str, Any], *,
                           name: str, timeout: float) -> Dict[str, Any]:
        """The legacy execution: bounded inline, cancelled when the bound expires."""
        try:
            return await asyncio.wait_for(tool["executor"](ctx, args), timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout",
                    "message": f"Tool '{name}' timed out after {timeout:.0f}s."}
        except Exception as e:  # noqa: BLE001 — a tool bug must not kill the response
            # A guarded tool (post_to_thread) can decline to post because the conversation moved
            # on. That is control flow, not a tool bug: reported as `execution_error` the model
            # would read it as a broken tool and try again, which is the one thing a suppression
            # must never cause. Imported lazily — this module sits UNDER message_processor in
            # the import graph, so naming it at module scope is a cycle.
            from message_processor.stale_send_guard import StaleSendSuppressed
            if isinstance(e, StaleSendSuppressed):
                raise
            return {"ok": False, "error": "execution_error", "message": str(e)[:500]}

    @staticmethod
    def _open_flight(turn: Any, *, name: str, args: Dict[str, Any], call_id: str,
                     timeout: float) -> Optional[tuple]:
        """The turn's answer for this call id, or None when the turn could not give one.

        None is a PLUMBING FAILURE on a turn that has the mechanism (the caller has already
        checked that), and the caller refuses the call rather than running it unprotected. A
        runtime without `open_tool_flight` at all — a stand-in object in a test, an older turn —
        never reaches here and keeps the legacy path verbatim."""
        try:
            return turn.open_tool_flight(call_id=call_id, tool_name=name,
                                        fingerprint=_call_fingerprint(name, args),
                                        timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"tool flight unavailable for {name}: {e}")
            return None

    @staticmethod
    def _open_anonymous_flight(turn: Any, *, name: str,
                               timeout: float) -> Optional[Any]:
        """Lifecycle ownership for an id-less call, or None when the turn could not give one.

        None is a PLUMBING FAILURE on a turn that HAS the mechanism (the caller has already
        checked), and the caller refuses the call rather than running it untracked. A raise, a
        None and a value that is not a `ToolFlight` are the SAME failure: the mechanism is
        present and there is no flight, so nothing would drain, cancel or revoke the call. Only a
        turn without the API at all keeps the legacy bounded path."""
        from message_processor.turn_runtime import ToolFlight  # lazy: cycle at module scope
        try:
            flight = turn.open_anonymous_tool_flight(tool_name=name, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"anonymous tool flight unavailable for {name}: {e}")
            return None
        return flight if isinstance(flight, ToolFlight) else None

    async def _fly_call(self, tool: Dict[str, Any], ctx: Any, args: Dict[str, Any], *,
                        name: str, flight: Any, created: bool, turn: Any,
                        call_id: Optional[str]) -> Dict[str, Any]:
        """Await this call's ONE execution, bounded by the deadline stamped when it was opened."""
        label = call_id or "no id"
        if created:
            _ensure_shared_containers(ctx)
            call_ctx = self._per_call_context(ctx, call_id=call_id, flight=flight)
            if call_ctx is None:
                # Without the per-call stamp the executor has no flight to mark launched, so the
                # one thing standing between a replayed call id and a second picture is gone.
                # Refused rather than run: nothing has happened yet, and this is the last moment
                # that is still true.
                self._abandon(turn, flight)
                logger.error(f"per-call context unavailable for {name} (call {label}) — "
                             "refusing to run the call unprotected")
                return {"ok": False, "error": "flight_unavailable",
                        "message": (f"Tool '{name}' could not be started safely. Nothing ran — "
                                    "try the call again.")}
            invocation = self._invoke(tool, ctx, call_ctx, args)
            try:
                turn.launch_tool_flight(flight, invocation)
            except Exception as e:  # noqa: BLE001 — could not own the work; refuse it
                # Nobody took the coroutine, so nobody will ever await it: closed here, by the
                # caller that made it. Left open it becomes a "coroutine was never awaited"
                # warning attached to a turn that is otherwise reporting itself honestly.
                invocation.close()
                self._abandon(turn, flight)
                logger.error(f"tool flight not launched for {name} (call {label}): {e}")
                return {"ok": False, "error": "flight_unavailable",
                        "message": (f"Tool '{name}' could not be started safely. Nothing ran — "
                                    "try the call again.")}
        task = getattr(flight, "task", None)
        if task is None:
            return {"ok": False, "error": "execution_error",
                    "message": f"Tool '{name}' could not be started."}
        try:
            # SHIELDED: a caller that gives up — this bound expiring, the round being cancelled —
            # must not abort an effect already in flight, or a posted message loses the receipt
            # that makes it ours. Cancellation is the outer finally's job, and it is established
            # there rather than requested here.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=flight.remaining())
        except asyncio.TimeoutError:
            # The shielded execution is still running and may yet produce a result — one nothing
            # will deliver. THE TIMEOUT IS SELECTED HERE, so this is a path that commits nothing:
            # whatever that call staged describes a result the model is not getting (§2g).
            logger.warning(
                f"tool '{name}' (call {label}) outran its {flight.timeout:.0f}s bound; the "
                "turn will settle it before its receipts")
            return {"ok": False, "error": "timeout",
                    "message": f"Tool '{name}' timed out after {flight.timeout:.0f}s."}
        except asyncio.CancelledError:
            # THE ONE EXECUTION was cancelled (the turn settling its flights), and we were not.
            # That is a result, not a cancellation of the round: whoever is waiting here still
            # owes the model an output for this call, and a duplicate arriving after the fact was
            # never cancelled at all. Our OWN cancellation still propagates — `cancelling()` is
            # what tells the two apart.
            current = asyncio.current_task()
            if task.cancelled() and not (current is not None and current.cancelling()):
                return {"ok": False, "error": "cancelled",
                        "message": (f"Tool '{name}' was stopped before it finished; it was not "
                                    "run again.")}
            raise
        except Exception as e:  # noqa: BLE001 — a tool bug must not kill the response
            from message_processor.stale_send_guard import StaleSendSuppressed
            if isinstance(e, StaleSendSuppressed):
                raise
            return {"ok": False, "error": "execution_error", "message": str(e)[:500]}
        # THE COMMIT POINT (§2g). This line is reached only when THIS result is the one going to
        # the model: every other exit above selected something else — a timeout, a cancellation,
        # an error — and each of them returns without committing. Nothing before here can grant
        # authority, and nothing after here can take it back, so "the model was shown it" is a
        # fact about control flow rather than a guess made from a clock while the answer was
        # still undecided.
        self._commit_staged_roots(flight, turn, result)
        self._commit_staged_edit_targets(flight, turn, result)
        return result

    @staticmethod
    def _commit_staged_roots(flight: Any, turn: Any, result: Any) -> None:
        """Grant authority to the staged roots whose own field survives into what is delivered.

        Two questions, both answerable only here: WAS this result selected (yes — see the call
        site), and WHICH OF ITS PARTS reached the model. The second is decided against the
        SERIALIZED, CLIPPED bytes the loop will hand back, because the tail of a long result is
        cut off and a root nobody can read is a root nobody was shown.

        Reads the claims off the FLIGHT, so EVERY waiter that selects the real result commits
        them — including a duplicate that joined a flight whose original waiter was cancelled or
        timed out. Committing twice is harmless: enrollment is a set, and a root already in it
        re-enrols to no effect.

        Never raises: a call that returned content must not fail because bookkeeping did.
        """
        staged = getattr(flight, "staged_roots", None) if flight is not None else None
        if not staged:
            return
        enroll = getattr(turn, "enroll_discovered_root", None)
        if enroll is None:
            return
        try:
            clipped = serialize_tool_result(result)
            for root in staged:
                if not isinstance(root, StagedRoot) or not _survives_truncation(clipped, root):
                    continue
                enroll(channel_id=root.channel_id, root_ts=root.root_ts, source=root.source)
        except Exception as e:  # noqa: BLE001 — authority is never load-bearing for the answer
            logger.debug(f"staged roots not committed: {e}")

    @staticmethod
    def _commit_staged_edit_targets(flight: Any, turn: Any, result: Any) -> None:
        """EDIT §2b: enroll the staged edit targets whose own `"ts"` field survives into what is
        delivered — the same commit moment, the same clipped bytes and the same field-pair rule
        as the staged roots, because "the model saw the exact message" is one question.

        Enrollment lands on the TURN, and the round's context is only re-stamped at the top of
        the NEXT `dispatch_all` — which is what keeps a same-round read and edit from authorizing
        each other. Never raises: a call that returned content must not fail over bookkeeping.
        """
        staged = getattr(flight, "staged_edit_targets", None) if flight is not None else None
        if not staged:
            return
        enroll = getattr(turn, "enroll_discovered_edit_target", None)
        if enroll is None:
            return
        try:
            clipped = serialize_tool_result(result)
            for target in staged:
                if not isinstance(target, StagedEditTarget):
                    continue
                if not _field_pair_survives(clipped, target.field, target.message_ts):
                    continue
                enroll(channel_id=target.channel_id, message_ts=target.message_ts,
                       thread_root_ts=target.thread_root_ts, edited_ts=target.edited_ts,
                       receipt_class=target.receipt_class, source=target.source)
        except Exception as e:  # noqa: BLE001 — authority is never load-bearing for the answer
            logger.debug(f"staged edit targets not committed: {e}")

    @staticmethod
    def _abandon(turn: Any, flight: Any) -> None:
        """Give the call id back after a plumbing failure that ran nothing. Never after a
        launch — the turn refuses that, and it is right to."""
        try:
            turn.abandon_tool_flight(flight)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"tool flight not abandoned: {e}")

    @staticmethod
    def _per_call_context(ctx: Any, *, call_id: Optional[str], flight: Any) -> Any:
        """A shallow copy carrying this call's identity, or None when it cannot be made.

        None is fatal to the call by design. The stamp is how the executor reaches
        `mark_launched`, so an unstamped context is a call running with its duplicate protection
        silently disabled — and the tools this guards are the ones where a second run is a second
        picture, a second bill and a second post."""
        try:
            call_ctx = copy.copy(ctx)
        except Exception as e:  # noqa: BLE001
            logger.error(f"per-call context copy failed: {e}")
            return None
        try:
            # An id-less call gets no identity stamp — there is none to give, and inventing one
            # would put a synthetic id in front of an executor that reads it as the model's.
            if call_id:
                call_ctx.tool_call_id = str(call_id)
            call_ctx.tool_flight = flight
            call_ctx.tool_timeout = flight.timeout
            call_ctx.tool_deadline = flight.deadline
        except Exception as e:  # noqa: BLE001
            logger.error(f"per-call context not stamped: {e}")
            return None
        return call_ctx

    @staticmethod
    async def _invoke(tool: Dict[str, Any], parent_ctx: Any, call_ctx: Any,
                      args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await tool["executor"](call_ctx, args)
        finally:
            _adopt_per_call_flags(parent_ctx, call_ctx)

    @staticmethod
    def _restamp_trusted_roots(ctx: Any) -> None:
        """§2g. Re-resolve this ROUND's `post_to_thread` allowlist from the turn, ONCE, before
        any of the round's calls run.

        This is where "per round" is made true. The context is built once per handler request and
        rides every round by reference, so without this a root a tool result proved in round N
        would be visible in no round at all. `dispatch_all` runs exactly once per round — for the
        ordinary round AND for the no-reply terminal round — which makes the timing a property of
        the code rather than a comment.

        AT THE TOP, never per call: a set refreshed alongside each call would let a search and a
        post issued in the SAME round authorize each other, which is "authorized because the model
        named it in the same breath" — the thing `execute_post_to_thread`'s own comment forbids.

        Two things it will not do. It never re-stamps a context with no turn to resolve against —
        there is nothing to re-read, and a hand-built context's explicit set is its own answer.
        And it never replaces a set with `None`, because `None` is the widest authorization there
        is and this is a refresh, not a widening.
        """
        turn = getattr(ctx, "turn", None)
        if turn is None:
            return
        try:
            # Lazy: handlers.text imports this module, so the pair cycles at module scope.
            from message_processor.handlers.text import _trusted_thread_roots
            resolved = _trusted_thread_roots(turn)
            if resolved is None:
                return
            ctx.trusted_thread_roots = resolved
        except Exception as e:  # noqa: BLE001 — a refresh that fails leaves the round's own set
            logger.debug(f"trusted thread roots not re-stamped for this round: {e}")

    @staticmethod
    def _restamp_edit_targets(ctx: Any) -> None:
        """EDIT §2b. Re-resolve this ROUND's edit-target mapping from the turn, ONCE, before any
        of the round's calls run — the exact discipline `_restamp_trusted_roots` states, for the
        same reason: at the top, never per call, so a read and an edit issued in the SAME round
        cannot authorize each other. A context with no turn keeps its explicit mapping, and a
        refresh that fails leaves the round's own mapping in place (the mapping itself is never
        None — the combiner fails to EMPTY, not to wide)."""
        turn = getattr(ctx, "turn", None)
        if turn is None:
            return
        try:
            from message_processor.handlers.text import _authorized_edit_targets
            resolved = _authorized_edit_targets(turn)
            if not isinstance(resolved, Mapping):
                return
            ctx.authorized_edit_targets = resolved
        except Exception as e:  # noqa: BLE001
            logger.debug(f"authorized edit targets not re-stamped for this round: {e}")

    async def dispatch_all(self, ctx: ToolContext, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a round's calls in parallel; result order matches ``calls``.

        The call's own id rides along, because THIS is the seam a duplicate dispatch would arrive
        at: the loop hands the ids it already has, and the registry decides whether the work has
        been done before."""
        self._restamp_trusted_roots(ctx)
        self._restamp_edit_targets(ctx)
        return list(await asyncio.gather(
            *(self.dispatch(ctx, c.get("name", ""), c.get("arguments"), c.get("call_id"))
              for c in calls)
        ))


def serialize_tool_result(result: Any) -> str:
    """JSON-encode an executor result, truncated to TOOL_RESULT_MAX_CHARS for the model."""
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        s = str(result)
    cap = config.tool_result_max_chars
    if len(s) > cap:
        s = s[:cap] + " …[truncated]"
    return s


@dataclass(frozen=True)
class StagedRoot:
    """One root a tool result CLAIMS, before anything has granted it authority.

    `field` is the name of the structured key the root was read from — `thread_ts` for a search
    hit or a thread fetch, `ts` for a history entry with replies. It is carried because the
    survival check below is about that FIELD surviving, not about the digits appearing somewhere.
    """

    channel_id: Any
    root_ts: Any
    source: str
    field: str


def stage_discovered_root(ctx: Any, *, channel_id: Any, root_ts: Any, source: str,
                          field: str) -> None:
    """An executor CLAIMS a root. It does not get one.

    §2g's authority rule, split in two so that no part of it is decided by a component that
    cannot know the answer. An executor cannot know whether its result will be the one the model
    receives — it is still running when that is decided — so it may only say "this result names
    this root". The registry commits, at the one moment the answer is known.

    The claim is recorded on the FLIGHT — the execution — not on this call's context, because a
    flight can have more than one waiter and only the first has a context. A context with no
    flight stages nothing and is not an error: that is a hand-built context outside the registry,
    and a call the registry never selected a result for is a call that never authorized anything.
    """
    staged = getattr(getattr(ctx, "tool_flight", None), "staged_roots", None)
    if not isinstance(staged, list):
        return
    staged.append(StagedRoot(channel_id=channel_id, root_ts=root_ts, source=source, field=field))


def _survives_truncation(clipped: str, root: StagedRoot) -> bool:
    """Did this root's OWN STRUCTURED FIELD reach the model?

    `serialize_tool_result` clips the function output, so the tail of a long result never arrives.
    The test is that the root's own `"<field>": "<ts>"` pair is present in the clipped bytes — the
    pair as JSON writes it, not the timestamp on its own.

    THE DIFFERENCE IS NOT COSMETIC. A bare search for the digits accepts a root whose entry was
    clipped away entirely, as long as some earlier entry happened to CONTAIN those digits — an
    author id, a quoted timestamp in somebody's message text. That authorizes a thread the model
    was never shown, sourced from prose, which is the exact thing §2g exists to refuse. A field
    pair cannot be forged from message text: JSON escapes the quotes inside a string value, so
    text reading `"thread_ts": "…"` serializes as `\\"thread_ts\\": \\"…\\"` and does not match.
    """
    return _field_pair_survives(clipped, root.field, root.root_ts)


def _field_pair_survives(clipped: str, field_name: str, value: Any) -> bool:
    """The one rule, factored so the staged roots and the staged edit targets (EDIT §2b) cannot
    apply two different readings of "survived the clip"."""
    try:
        pair = json.dumps({field_name: value}, ensure_ascii=False, default=str)[1:-1]
    except Exception:  # noqa: BLE001 — unserializable claim, no authority
        return False
    return bool(pair) and pair in clipped


@dataclass(frozen=True)
class StagedEditTarget:
    """EDIT §2b. One exact own message a read tool's result CLAIMS is editable, before anything
    has granted it authority.

    The executor that stages it has already proved the §2b preconditions — result from the
    turn's own channel, a raw Slack message that is OURS, a finalized `assistant_reply` receipt,
    and the exact ts present in the returned result — and the registry commits the claim only if
    its own `"<field>": "<ts>"` pair survives the clipped serialization the model receives.

    `source` is the tool name that returned it; `field` is the structured field the ts survived
    in, for the same survival rule `StagedRoot` carries its field for.
    """

    channel_id: str
    message_ts: str
    thread_root_ts: Optional[str]
    edited_ts: Optional[str]
    receipt_class: str
    source: str
    field: str


def stage_discovered_edit_target(ctx: Any, *, channel_id: str, message_ts: str,
                                 thread_root_ts: Optional[str], edited_ts: Optional[str],
                                 receipt_class: str, source: str, field: str) -> None:
    """An executor CLAIMS an edit target. It does not get one.

    The same split as `stage_discovered_root`, for the same reason: the executor cannot know
    whether its result will be the one the model receives, so it may only record the claim on
    the FLIGHT; `ToolRegistry` commits the subset whose own field survives the delivered,
    clipped payload, and `TurnRuntime.enroll_discovered_edit_target` re-validates on the way
    in. A context with no flight stages nothing and is not an error.
    """
    staged = getattr(getattr(ctx, "tool_flight", None), "staged_edit_targets", None)
    if not isinstance(staged, list):
        return
    staged.append(StagedEditTarget(channel_id=channel_id, message_ts=message_ts,
                                   thread_root_ts=thread_root_ts, edited_ts=edited_ts,
                                   receipt_class=receipt_class, source=source, field=field))
