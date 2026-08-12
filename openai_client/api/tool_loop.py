"""Local function-call loop for the Responses API (redesign Phase A — the keystone).

Wraps the existing ``create_text_response_with_tools`` / ``create_streaming_response_with_tools``
calls in a loop: collect ``function_call`` items → dispatch through the ToolRegistry
(parallel, timeout-guarded) → append ``function_call`` + ``function_call_output`` items to the
input → re-invoke. Local tools compose with server-side tools (web_search, MCP) in the same
``tools`` array.

Caps: ``MAX_TOOL_ROUNDS`` rounds / ``MAX_TOOL_CALLS_PER_TURN`` total calls. On cap, one final
round runs with ``tool_choice="none"`` so the model must answer with what it has.

Streaming: intermediate rounds stream through the same callback, but their text deltas are
suppressed inside ``create_streaming_response_with_tools`` once a function_call appears in the
round (pre-tool preamble); ``tool_callback(f"local:{name}", ...)`` drives the status line while
tools run, and only the final round's text reaches the user.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, cast

from config import config
from logger import setup_logger
from message_markers import join_segments
from openai_client.container_errors import (adoption_blocked, demote_container_tools,
                                            pin_container_tools)
from tool_registry import ToolContext, ToolRegistry, serialize_tool_result

from . import responses as responses_api

logger = setup_logger(name="slack_bot.ToolLoop")


def _call_ok(result: Any) -> bool:
    """A tool result counts as successful unless it explicitly says ok=False."""
    return not (isinstance(result, dict) and result.get("ok") is False)


def _function_calls(sink: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The dispatchable function_call entries of a round's sink (reasoning items excluded)."""
    return [e for e in sink if e.get("type", "function_call") == "function_call"]


def _note_turn_tool_call(tool_context: Any, record: Dict[str, Any]) -> None:
    """Mirror one dispatched call onto the TURN (§5.4a amendment), if there is a turn to tell.

    The loop's accumulator is only readable by a caller the loop RETURNS to. A cross-thread post
    commits mid-loop and cannot be retracted, so the record of the tools that produced it must
    not depend on a later round succeeding. Best-effort: a context with no turn — a background
    job, a hand-built context in a test — records nothing and is not an error, and bookkeeping
    never breaks a round.
    """
    note = getattr(getattr(tool_context, "turn", None), "note_tool_call", None)
    if note is None:
        return
    try:
        note(record)
    except Exception:  # noqa: BLE001 — a ledger write never fails a tool round
        pass


async def _run_tool_round(
    self,
    registry: ToolRegistry,
    tool_context: ToolContext,
    sink: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    local_tool_calls: List[Dict[str, Any]],
    tool_callback: Optional[Callable[[str, str], Any]] = None,
    result_overrides: Optional[Dict[int, Any]] = None,
) -> None:
    """Dispatch one round's calls, then replay the round's items (reasoning items in
    place, each function_call followed by its function_call_output) onto the input.

    ``result_overrides`` (id(call) -> result) short-circuits specific calls: they are NOT
    dispatched and their given result is fed back instead — used to reject an invalid
    no_response_needed (F2) while still running its siblings. Keyed by OBJECT IDENTITY, not
    call_id: OpenAI normally returns a unique non-empty call_id per function_call, but a
    degenerate/malformed round could repeat or omit them, which would misroute the override
    to a sibling; identity is always exact (the same call dicts flow through `sink`)."""

    async def _notify(tool_id: str, status: str) -> None:
        if not tool_callback:
            return
        try:
            result = tool_callback(tool_id, status)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as e:  # noqa: BLE001 — status UI must never break the loop
            self.log_warning(f"Tool callback error for {tool_id}: {e}")

    from message_processor.tool_provenance import gist_from_arguments

    overrides = result_overrides or {}
    calls = _function_calls(sink)
    for call in calls:
        await _notify(f"local:{call.get('name')}", "started")

    dispatch_calls = [c for c in calls if id(c) not in overrides]
    dispatched = await registry.dispatch_all(tool_context, dispatch_calls)
    dispatched_by_id = {id(c): r for c, r in zip(dispatch_calls, dispatched)}
    result_by_id = {}
    for call in calls:
        oid = id(call)
        result = overrides[oid] if oid in overrides else dispatched_by_id.get(oid)
        ok = _call_ok(result)
        # F7: capture a short arg-derived gist alongside name/ok (provenance; no results).
        # WRITTEN DOWN TWICE, ON PURPOSE (§5.4a amendment): this accumulator only reaches the
        # handler if the loop RETURNS, and a cross-thread post that already landed cannot have
        # its provenance depend on that. `_note_turn_tool_call` puts the same record on the turn,
        # which outlives the loop.
        record = {"name": call.get("name"), "ok": ok,
                  "gist": gist_from_arguments(call.get("arguments"))}
        local_tool_calls.append(record)
        _note_turn_tool_call(tool_context, record)
        self.log_info(f"Local tool '{call.get('name')}' -> {'ok' if ok else 'error'}")
        result_by_id[id(call)] = result
        await _notify(f"local:{call.get('name')}", "completed")

    _replay_round_items(sink, input_items, result_by_id, tool_context)


def _replay_round_items(sink: List[Dict[str, Any]], input_items: List[Dict[str, Any]],
                        result_by_id: Dict[int, Any], tool_context: ToolContext) -> None:
    """Append a dispatched round's items to the running input.

    Split out of ``_run_tool_round`` because the terminal-silence path dispatches its own round
    (its budget rules differ) and, when the executor REFUSES the terminal, has to hand the model
    the same replay so the loop can continue — a function_call left without a matching
    function_call_output earns a 400 on the next request."""
    # Replay in encounter order — reasoning models require their reasoning items to
    # precede the paired function_call when the conversation is replayed statelessly.
    for entry in sink:
        if entry.get("type") == "reasoning":
            if entry.get("item"):
                input_items.append(entry["item"])
            continue
        input_items.append({
            "type": "function_call",
            "call_id": entry.get("call_id"),
            "name": entry.get("name"),
            "arguments": entry.get("arguments") or "{}",
        })
        input_items.append({
            "type": "function_call_output",
            "call_id": entry.get("call_id"),
            "output": serialize_tool_result(result_by_id.get(id(entry))),
        })

    # `view_image` re-attaches an EARLIER thread image (whose pixels never rode this turn — only
    # the answered message's attachments become input_image parts). Drain what it staged into ONE
    # user message so the model can actually SEE it on the next round.
    #
    # Placement and role both matter:
    #  * AFTER every function_call/function_call_output pair — never between a reasoning item and
    #    the function_call it belongs to, which reasoning models require to stay adjacent.
    #  * USER role, not developer: the bytes are untrusted user-supplied content, the same
    #    boundary the stored image descriptions already respect.
    # The `_image_id` bookkeeping key is stripped here: our dicts do double duty, and the API
    # 400s on unknown keys inside a content part.
    staged = getattr(tool_context, "pending_vision_parts", None) or []
    fresh = [r for r in staged if r.get("_ready") and not r.get("_replayed")]
    if fresh:
        content: List[Dict[str, Any]] = []
        for reservation in fresh:
            reservation["_replayed"] = True
            content.extend(reservation.get("parts") or [])
        if content:
            input_items.append({"role": "user", "content": content})


_PreRoundCallback = Callable[[], Awaitable[List[Dict[str, Any]]]]


async def _inject_pre_round_items(self, callback: Optional[_PreRoundCallback],
                                  input_items: List[Dict[str, Any]]) -> None:
    """Give the caller one chance per round to append items to the running input.

    ONCE PER WHILE-ITERATION, not once per HTTP attempt. A container-recovery retry
    (`responses.py`) re-sends the very list this appended to, so injecting any deeper would put
    the same items in front of the model twice; from up here a retry simply carries what is
    already there. Free rounds and the forced-final `tool_choice="none"` round are rounds too —
    the last one is a job's final chance to hear a correction before it writes its answer.

    Placement is the whole contract: the previous round's `_replay_round_items` has already run,
    so items land AFTER complete function_call/function_call_output pairs and never between a
    reasoning item and the call it belongs to.

    The callback is AWAITED, so it must be an async callable — awaiting is what makes an
    `async def` with no awaits in its body legal, which is the shape a caller needs to pop a
    queue and return with nothing able to interleave. A plain `def` handed in here is refused
    loudly rather than half-applied: awaiting its list raises, and the round goes out unsteered
    with a warning instead of the returned coroutine being read as a malformed item list.

    `except Exception`, deliberately not `BaseException`: a cancelled job must still cancel, so
    CancelledError propagates. A return that is not a list of item dicts is dropped WHOLE rather
    than filtered — half an injection is a worse round than an unsteered one — and the round
    goes out regardless.
    """
    if callback is None:
        return
    try:
        items: Any = await callback()
    except Exception as e:  # noqa: BLE001 — a caller's bookkeeping never fails a round
        self.log_warning(f"Pre-round input callback failed: {e}")
        return
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        self.log_warning(
            f"Pre-round input callback returned {type(items).__name__}, not a list of item "
            "dicts — skipping injection this round")
        return
    if items:
        input_items.extend(items)
        self.log_info(f"Injected {len(items)} pre-round input item(s)")


def _observed_container(tool_context: Any, artifacts_sink: Any) -> Optional[str]:
    """The container this turn's code actually ran in, ignoring any that has since DIED.

    `_note_container` records every container a `code_interpreter_call` reports, so after a
    mid-turn container death the sink holds both the corpse and the ephemeral sandbox the
    recovery retry ran in. The corpse is in `container_gone_sink`; naming it again would 404 the
    next request. Of what survives, the FIRST id wins: it is where the earliest files are.
    """
    from message_processor.artifacts import collect_container_ids
    gone = set(getattr(tool_context, "container_gone_sink", None) or ())
    return next((cid for cid in collect_container_ids(artifacts_sink) if cid not in gone), None)


async def _adopt_and_pin(tool_context: Any, artifacts_sink: Any,
                         tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """W3: bind the container this turn is really in, and name it from here on.

    Turns start on `{"type": "auto"}`, so the first request asks for a sandbox without knowing
    which one it will get. The id arrives during the round — `_note_container` puts it in the
    artifacts sink as the `code_interpreter_call` streams — and from then on it must be named
    explicitly. Leave the declaration on `auto` and the next request provisions a SECOND
    container: the chart round 1 wrote would be in a sandbox round 2 cannot open.

    Called TWICE per round, because the id can appear on either side of the dispatch and each
    call is a cheap no-op once the holder is filled:

      * After the response, before local calls dispatch. This is what stops a MIXED round — a
        hosted `code_interpreter_call` in container A alongside a `mount_file` in the same
        response — from binding the wrong sandbox: the bridge tool would otherwise find an empty
        holder, mint container B, and strand A's files outside the thread's binding.
      * Before the next request. A bridge tool that ran with no hosted call beside it fills the
        holder DURING dispatch, after the response-side check has already run, and its container
        still has to be named in the next declaration.

    Three things happen, in this order:
      * EVICT a corpse. A container that died mid-turn is in `container_gone_sink`, and the
        recovery that rescued that one call demoted only ITS retry — the loop's array still names
        the dead id. Left alone, every remaining round 404s and mints another recovery sandbox.
      * ADOPT — the id becomes the thread's persistent binding, so tomorrow's "revise that deck"
        finds the file. Skipped while the sink is flagged: after a recovery the surviving id
        names a sandbox the API minted for one retried call, which must never outlive the turn.
      * PIN — the id replaces `auto` in the array. Deliberately NOT conditional on adoption. A
        recovery sandbox is where the model IS, so the rest of the turn belongs in it even though
        the DB must never hear about it; likewise a bridge tool's own container, already in the
        holder, and an id whose adoption lost the CAS race.

    Returns the tools array to send next, the SAME object when nothing changed. Never raises:
    losing continuity is a worse turn, not a failed one.
    """
    holder = getattr(tool_context, "sandbox", None)
    if holder is None:
        return tools
    try:
        current = holder.container_id
        if isinstance(current, str) and current and tool_context.container_recycled():
            logger.info(f"Container {current} died mid-turn — this turn moves off it")
            holder.container_id = None
            tools = demote_container_tools(tools)[0] or tools
        if not holder.container_id:
            container_id = _observed_container(tool_context, artifacts_sink)
            if container_id is None:
                return tools
            if adoption_blocked(artifacts_sink):
                # Pinned for the rest of the turn, never written down. One turn, one sandbox —
                # even when the sandbox is one we would not choose to keep.
                logger.debug(f"Staying in recovery container {container_id} without adopting it")
            else:
                manager = holder.manager
                if manager is not None and holder.thread_key:
                    await manager.adopt(holder.thread_key, container_id)
            holder.container_id = container_id
        return pin_container_tools(tools, holder.container_id) or tools
    except Exception as e:  # noqa: BLE001 — never fail a round over bookkeeping
        logger.warning(f"Container adoption skipped: {e}")
        return tools


def _merge_used(tools_used_all: List[str], round_used: List[Any],
                tool_context: Any = None) -> None:
    """Merge a round's used-tool names into the loop's accumulator AND onto the turn.

    THE TURN IS THE COPY THAT SURVIVES (§5.4a exit-path amendment, codex round-2 #1). This
    accumulator is a local of a loop that has to RETURN for anyone to read it, so a round that
    fails after a cross-thread post has already landed takes the external names with it — the
    same way it used to take the local call records. Every merge point routes through here, so
    mirroring it once covers all of them.
    """
    for name in round_used:
        if name not in tools_used_all:
            tools_used_all.append(name)
    note = getattr(getattr(tool_context, "turn", None), "note_external_tools", None)
    if note is None:
        return
    try:
        note(list(round_used or ()))
    except Exception:  # noqa: BLE001 — a ledger write never fails a round
        pass


def _replay_committed_text(input_items: List[Dict[str, Any]], text: str) -> None:
    """Replay a STREAMING round's pre-tool preamble as an assistant turn.

    In the streaming loop a round's text is only suppressed once a function_call appears —
    whatever the model said BEFORE calling the tool has already streamed to Slack. Without
    replaying it, the next round sees no record of having spoken, so the model says the same
    thing again, and the repeat lands in the SAME streamed message: the user reads
    "Making that now. Making that now."

    Deliberately NOT done in the non-streaming loop: there, an intermediate round's text is
    discarded rather than shown, so the model repeating it in the final round is exactly
    right — it is the only copy the user ever sees.

    Appended BEFORE the round's items, so it never lands between a reasoning item and the
    function_call it belongs to (reasoning models require that pair to stay adjacent).
    """
    if (text or "").strip():
        input_items.append({"role": "assistant", "content": text})


# F37: a "free" (bookkeeping) round costs no budget, but it is still a round — a model that
# loops on update_todos and nothing else must terminate. Free rounds get their own ceiling at
# this multiple of the productive cap.
_FREE_ROUND_CEILING = 2
# ...and a ceiling WITHIN a round, enforced BEFORE dispatch. A round's calls run in parallel, so
# without this a single round can fire fifty update_todos at once: fifty concurrent executors and
# fifty Slack updates. Capping only the totals stops the NEXT round — far too late, the storm has
# already happened. A rewrite-the-whole-list tool never needs more than one call in a round; two
# is slack. Excess calls are not dispatched, but they ARE answered (see _EXCESS_FREE_RESULT):
# the Responses API 400s on a function_call with no matching function_call_output.
_FREE_CALLS_PER_ROUND = 2
_EXCESS_FREE_RESULT = {
    "ok": False,
    "error": "too_many_calls_this_round",
    "message": ("Not run: you called this bookkeeping tool several times in one round. It "
                "replaces the whole list in a single call — make one call with the final state."),
}


# --- F2: no_response_needed terminal action ---

_NO_REPLY_TOOL = "no_response_needed"

# F2: fed back when no_response_needed is called AFTER visible reply text already
# streamed to Slack — the call is invalid; the model must finish the reply instead.
_INVALID_NO_REPLY_RESULT = {
    "ok": False,
    "error": "invalid_no_response_needed",
    "message": ("Invalid: you already began a visible reply — complete the reply instead "
                "of calling no_response_needed."),
}


# Fed back to the SECOND and later no_response_needed calls of one round. A turn cannot end
# twice, so only the first is ever considered — but on a round where the first is rejected the
# loop continues, and every call it continues past needs an output the model can read. Without
# this they were dispatched normally and answered ok, so the model saw one terminal refused and
# another accepted while the turn kept going.
_DUPLICATE_TERMINAL_RESULT = {
    "ok": False,
    "error": "duplicate_terminal_call",
    "message": ("Not run: you called this more than once in one round. One call ends the turn — "
                "make a single call, or reply normally."),
}


def _invalid_silence_reason_result() -> Dict[str, Any]:
    """Fed back when the terminal call's `reason` is absent or not one of the eight.

    The turn is NOT ended: an unrecognized reason means we do not know why the model wants to
    be quiet, and guessing one would put our inference in the ledger as its testimony. It gets
    told the vocabulary and gets another round to decide — including the option of deciding to
    speak after all."""
    from message_processor.terminal_actions import SILENCE_REASONS
    return {
        "ok": False,
        "error": "invalid_silence_reason",
        "message": ("Not run: `reason` must be exactly one of "
                    f"{', '.join(SILENCE_REASONS)}. Call it again with one of those values, "
                    "or reply normally."),
    }


def _free_call_overrides(calls: List[Dict[str, Any]], free_names: set,
                        remaining_allowance: int) -> Dict[int, Any]:
    """Refuse the free calls in this round that exceed the per-round burst cap or the remaining
    free allowance — BEFORE they are dispatched. Returns result_overrides for _run_tool_round,
    which answers them without running them (a function_call left without a
    function_call_output earns a 400 on the next request).

    Shared by BOTH loops on purpose. It lived inside the streaming one, and the non-streaming
    loop consequently had no notion of a free call at all — so a bookkeeping call there spent
    the same budget a real tool does."""
    if not free_names:
        return {}
    allowed = min(_FREE_CALLS_PER_ROUND, max(0, remaining_allowance))
    overrides: Dict[int, Any] = {}
    taken = 0
    for c in calls:
        if c.get("name") not in free_names:
            continue
        taken += 1
        if taken > allowed:
            overrides[id(c)] = _EXCESS_FREE_RESULT
    return overrides


def _no_reply_call(calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((c for c in calls if c.get("name") == _NO_REPLY_TOOL), None)


def _silence_reason(call: Dict[str, Any]) -> Optional[str]:
    """The call's declared silence reason, or None when it is not one of the eight.

    No sanitizing and no coercion: the value is a closed enum, so it either IS a member — in
    which case it is already exactly what the ledger stores — or the call is invalid. Rewriting
    an unrecognized value to `other` would file our guess under the model's name, in the one
    column that exists to record the model's own account of itself."""
    import json
    from message_processor.terminal_actions import is_valid_silence_reason
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            args = {}
    reason = (args or {}).get("reason") if isinstance(args, dict) else None
    return reason if is_valid_silence_reason(reason) else None


def _terminal_overrides(calls: List[Dict[str, Any]], terminal_call: Dict[str, Any],
                        terminal_result: Dict[str, Any]) -> Dict[int, Any]:
    """Short-circuit results for EVERY terminal call in a round the loop is continuing past.

    The first one carries the reason it was rejected. The rest are duplicates: they must not be
    dispatched (a turn cannot end twice, and the executor would answer them ok), and they cannot
    simply be dropped either — the Responses API 400s on a function_call with no matching
    output, and a model reading one refusal beside one success learns the wrong lesson."""
    overrides: Dict[int, Any] = {id(terminal_call): terminal_result}
    for call in calls:
        if call is not terminal_call and call.get("name") == _NO_REPLY_TOOL:
            overrides[id(call)] = _DUPLICATE_TERMINAL_RESULT
    return overrides


_OVER_BUDGET_RESULT = {
    "ok": False,
    "error": "over_budget",
    "message": ("Not run: this turn's tool budget was already spent. Work with what you have."),
}


def _over_budget_overrides(calls: List[Dict[str, Any]], free_names: set,
                           remaining_allowance: int) -> Dict[int, Any]:
    """Refuse the PRODUCTIVE calls in this round that exceed the turn's REMAINING call
    allowance — BEFORE they are dispatched. Returns result_overrides for _run_tool_round.

    Same shape, and the same reason, as ``_free_call_overrides``: a round's calls dispatch in
    PARALLEL, so charging them afterwards stops only the NEXT round — by which time a single
    response carrying a dozen `import_web_image` calls has already run a dozen fetches, vision
    calls and Slack uploads against a budget of three. The cap has to bind before dispatch.

    Order is the round's own call order: the first N run, the excess is refused. Refused, not
    dropped — a function_call left without a matching function_call_output earns a 400 on the
    next request, so the excess gets ``_OVER_BUDGET_RESULT`` fed back instead.

    Free (bookkeeping) calls are invisible to this: they have their own allowance and must never
    be displaced by, or displace, productive work. ``_charge`` bills only what actually RAN, so
    the calls suppressed here cost nothing — they did nothing."""
    allowed = max(0, int(remaining_allowance))
    overrides: Dict[int, Any] = {}
    taken = 0
    for c in calls:
        if c.get("name") in free_names:
            continue
        taken += 1
        if taken > allowed:
            overrides[id(c)] = _OVER_BUDGET_RESULT
    return overrides


async def _handle_no_reply_terminal(
    self,
    registry: ToolRegistry,
    tool_context: ToolContext,
    sink: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    terminal_call: Dict[str, Any],
    silence_reason: str,
    tools_used_all: List[str],
    local_tool_calls: List[Dict[str, Any]],
    remaining_budget: Optional[int] = None,
    result_overrides: Optional[Dict[int, Any]] = None,
    tool_callback: Optional[Callable[[str, str], Any]] = None,
    free_tools: Optional[Iterable[str]] = None,
    committed_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Terminal round: no_response_needed ends the turn's WORDS. It does not cancel the round.

    It used to: every sibling except react_to_message was dropped before dispatch. That read
    the tool as "abort the turn", and the effects it silently discarded were ones the model had
    already decided on — a memory write it judged worth keeping, a cross-thread post, an image.
    The model was choosing "say nothing HERE", and we were hearing "do nothing anywhere". So
    every sibling now runs through the ordinary parallel dispatch, and silence is honored after
    they finish, whatever they returned: a failed sibling is not a reason to start talking.

    There is deliberately NO allowlist. A tool that must not run on an unaddressed turn is
    refused by its own executor (that is where authorization lives, and it is checked whether or
    not this turn ends in silence); an allowlist here would be a second, quietly diverging copy
    of those rules.

    Duplicate no_response_needed calls are still suppressed — the FIRST valid one wins, and the
    turn cannot end twice. Budget is unchanged: PRODUCTIVE siblings count against
    MAX_TOOL_CALLS_PER_TURN and the terminal call itself reserves one slot of the remaining
    budget. ``free_tools`` (F37 bookkeeping calls) are budgeted separately by the caller and
    must never take a productive slot — with one slot left and a round ordered
    [update_todos, remember_fact], a naive slice spends it on the status card and drops the
    memory write, which is the exact inversion the free-tool allowance exists to prevent.

    ``result_overrides`` (id(call) -> result) passes through the caller's pre-dispatch
    suppressions (the excess-bookkeeping rule) so that mechanism keeps working here too.

    Silence is honored ONLY when the terminal EXECUTOR succeeds. The tool is statically exposed
    on the channel surface (§3a), so the executor — not the schema — is what knows whether this
    route may stay quiet; a refusal means the turn owes words. Returns None then, having replayed
    the whole round (every call answered, including the ones budget skipped) so the caller can
    charge the round and continue the loop with the refusal in front of the model."""
    overrides = result_overrides or {}
    free_names = {str(n) for n in (free_tools or ())}
    # Duplicates of the terminal are dropped, never executed: only `terminal_call` ends the turn.
    siblings = [c for c in calls
                if c is not terminal_call and c.get("name") != _NO_REPLY_TOOL]
    productive = [c for c in siblings if c.get("name") not in free_names]
    free_siblings = [c for c in siblings if c.get("name") in free_names]
    if remaining_budget is None:
        sibling_budget = len(productive)
    else:
        sibling_budget = max(0, int(remaining_budget) - 1)  # the terminal reserves one slot
    # Free calls are already capped per round by the caller's excess-bookkeeping suppression
    # (the excess ones arrive here inside `overrides`, answered but never dispatched).
    allowed_ids = {id(c) for c in productive[:sibling_budget]} | {id(c) for c in free_siblings}
    exec_calls = [c for c in calls if c is terminal_call or id(c) in allowed_ids]
    exec_ids = {id(c) for c in exec_calls}
    skipped = [c.get("name") for c in calls if id(c) not in exec_ids]
    if skipped:
        self.log_info(f"{_NO_REPLY_TOOL} terminal — over budget or duplicate, not run: {skipped}")

    async def _notify(tool_id: str, status: str) -> None:
        if not tool_callback:
            return
        try:
            result = tool_callback(tool_id, status)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as e:  # noqa: BLE001 — status UI must never break the loop
            self.log_warning(f"Tool callback error for {tool_id}: {e}")

    from message_processor.tool_provenance import gist_from_arguments

    for call in exec_calls:
        await _notify(f"local:{call.get('name')}", "started")
    dispatch_calls = [c for c in exec_calls if id(c) not in overrides]
    results = await registry.dispatch_all(tool_context, dispatch_calls)
    dispatched_by_id = {id(c): r for c, r in zip(dispatch_calls, results)}
    result_by_id: Dict[int, Any] = {}
    terminal_result: Any = None
    for call in exec_calls:
        result = overrides.get(id(call), dispatched_by_id.get(id(call)))
        result_by_id[id(call)] = result
        # Every executed sibling is recorded — the handler reads this list to learn what the
        # turn actually did (a landed reaction, a post that went out), and a silent turn whose
        # effects are missing from it is a turn the ledger describes as having done nothing.
        ok = _call_ok(result)
        if call is terminal_call:
            terminal_result = result
            # A refused terminal is an executed call like any other and belongs in the record;
            # an ACCEPTED one is the turn's ending, not one of its actions, and stays out.
            if not ok:
                refused = {"name": call.get("name"), "ok": False,
                           "gist": gist_from_arguments(call.get("arguments"))}
                local_tool_calls.append(refused)
                _note_turn_tool_call(tool_context, refused)
        else:
            record = {"name": call.get("name"), "ok": ok,
                      "gist": gist_from_arguments(call.get("arguments"))}
            local_tool_calls.append(record)
            _note_turn_tool_call(tool_context, record)
            self.log_info(f"Local tool '{call.get('name')}' -> {'ok' if ok else 'error'}")
        await _notify(f"local:{call.get('name')}", "completed")
    _merge_used(tools_used_all, [c.get("name") for c in exec_calls if c.get("name")],
                tool_context)

    if not _call_ok(terminal_result):
        # The executor refused (an owed-words route, the tool switched off). The turn is NOT
        # silenced: the refusal goes back as this call's output, the siblings keep theirs, and
        # the loop continues so the model answers. Calls the budget skipped are answered too —
        # unanswered function_calls 400 the next request.
        self.log_warning(
            f"{_NO_REPLY_TOOL} refused by its executor — the turn owes a reply; continuing")
        for call in calls:
            result_by_id.setdefault(id(call), _OVER_BUDGET_RESULT)
        _replay_committed_text(input_items, committed_text)
        _replay_round_items(sink, input_items, result_by_id, tool_context)
        return None

    self.log_info(f"{_NO_REPLY_TOOL}: ending turn without words — reason: {silence_reason}")
    return {
        "text": "",
        "tools_used": tools_used_all,
        "local_tool_calls": local_tool_calls,
        "terminal_action": "no_reply",
        "silence_reason": silence_reason,
    }


async def create_text_response_with_tool_loop(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    registry: ToolRegistry,
    tool_context: ToolContext,
    prior_committed: bool = False,
    free_tools: Optional[Iterable[str]] = None,
    aggregate_segments: bool = False,
    **params: Any,
) -> Dict[str, Any]:
    """Non-streaming response with local tool execution.

    ``free_tools`` (F37) names BOOKKEEPING calls — they produce nothing a user can see, so they
    do not spend the productive budget and never displace a real tool. Same rule the streaming
    loop applies, and it is here because it was NOT: this loop charged every call, so a
    bookkeeping call on the non-streaming path quietly cost a slot a real one needed. A round of
    nothing BUT free calls costs no round either, but free rounds have their own ceiling, so
    "free" can never mean "unbounded".

    Returns {"text", "segments", "tools_used", "local_tool_calls"} — ``local_tool_calls`` is the
    ordered [{"name", "ok"}] record of every local call (e.g. for reaction-only detection).

    ``aggregate_segments`` mirrors the streaming twin exactly: OPT-IN, and it makes ``text`` the
    seam-joined whole turn instead of the terminal round alone. The chat handler opts in (owner
    decision 2026-08-08), so a preamble written before a tool ran reaches the reader here as
    well — it used to be dropped on the floor, and this path shows nothing until the end, so
    there was never a half-written line on screen to give the loss away. It stays opt-in for the
    same reason the twin's does: a caller that reads ``text`` as a finished ARTIFACT rather than
    as a conversation wants the last round and not the "I'll go look…" in front of it, and the
    default must not decide that for it.

    ``segments`` is the rounds either way, so a caller that must TRANSFORM the text can work per
    round and never meet a seam this loop inserted (W4's marker parser).

    ``prior_committed`` (F8): True when an EARLIER attempt this turn (e.g. a streaming
    attempt that failed mid-reply) already exposed visible text. A no_response_needed on
    this attempt would then orphan that partial as fake silence, so it is REJECTED and the
    model is forced to finish the reply — mirroring the streaming loop's committed-text rule.
    """
    input_items: List[Dict[str, Any]] = list(messages)
    tools_used_all: List[str] = []
    local_tool_calls: List[Dict[str, Any]] = []
    tool_choice: Optional[str] = None
    rounds = 0
    total_calls = 0
    free_names = {str(n) for n in (free_tools or ())}
    free_rounds = 0
    free_calls = 0
    free_rounds_cap = max(1, config.max_tool_rounds * _FREE_ROUND_CEILING)
    free_calls_cap = max(1, config.max_tool_calls_per_turn * _FREE_ROUND_CEILING)

    def _charge(calls: List[Dict[str, Any]], suppressed: Dict[int, Any]) -> None:
        """Bill a round. A round of PURE bookkeeping costs no round and no productive call;
        anything else is fully charged, though the free calls riding in a mixed round stay
        free. Only calls that actually RAN are billed — a suppressed one did no work."""
        nonlocal rounds, total_calls, free_rounds, free_calls
        ran = [c for c in calls if id(c) not in suppressed]
        free = [c for c in ran if c.get("name") in free_names]
        productive = [c for c in ran if c.get("name") not in free_names]
        free_calls += len(free)
        if not productive and calls:
            free_rounds += 1
            return
        rounds += 1
        total_calls += len(productive)

    def _capped() -> bool:
        return (rounds >= config.max_tool_rounds
                or total_calls >= config.max_tool_calls_per_turn
                or free_rounds >= free_rounds_cap or free_calls >= free_calls_cap)

    # F30: expose this turn's live input (stable reference — appended in place across rounds)
    # plus the developer prompt/model, so a detached job can snapshot the full context by copy.
    if tool_context is not None:
        tool_context.current_input = input_items
        tool_context.system_prompt = params.get("system_prompt")
        tool_context.model = params.get("model")

    # Every round's text, in order — the same list, and under `aggregate_segments` the same
    # seam-joined result, as the streaming twin builds.
    segments: List[str] = []

    while True:
        # W3, request side: picks up a container a BRIDGE tool created during the previous
        # round's dispatch, which lands in the holder after the response-side check has run.
        tools = await _adopt_and_pin(tool_context, params.get("artifacts_sink"), tools)
        sink: List[Dict[str, Any]] = []
        # return_metadata=True makes this a metadata dict, which the declared `-> str` omits.
        result = cast(Dict[str, Any], await responses_api.create_text_response_with_tools(
            self,
            messages=input_items,
            tools=tools,
            return_metadata=True,
            function_call_sink=sink,
            tool_choice=tool_choice,
            **params,
        ))
        _merge_used(tools_used_all, result.get("tools_used") or [], tool_context)
        if result.get("text"):
            segments.append(result["text"])
        # W3: after the response, before this round's local calls dispatch — so a bridge tool in
        # a mixed round finds the hosted container instead of minting a rival one, and so the
        # array the NEXT request sends already names it.
        tools = await _adopt_and_pin(tool_context, params.get("artifacts_sink"), tools)

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": (join_segments(segments) if aggregate_segments
                         else result.get("text", "")),
                # The rounds this turn ran, in order. Under `aggregate_segments` these are the
                # pieces `text` was joined from, BEFORE the seams went between them; otherwise
                # `text` is the terminal one and these are the rest.
                "segments": list(segments),
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            silence_reason = _silence_reason(terminal_call)
            if silence_reason is not None and not prior_committed:
                suppressed = _free_call_overrides(calls, free_names,
                                                  free_calls_cap - free_calls)
                outcome = await _handle_no_reply_terminal(
                    self, registry, tool_context, sink, input_items, calls, terminal_call,
                    silence_reason, tools_used_all, local_tool_calls,
                    remaining_budget=config.max_tool_calls_per_turn - total_calls,
                    result_overrides=suppressed or None, free_tools=free_tools)
                if outcome is not None:
                    return outcome
                # The executor refused: the round ran, its outputs are on the input, and the
                # turn owes words. Charge it and go round again.
                _charge(calls, suppressed)
                if _capped():
                    self.log_warning(
                        f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — "
                        "forcing final answer")
                    tool_choice = "none"
                continue
            # Either the reason is not one of the eight, or a prior attempt already exposed
            # visible text (honoring silence now would orphan it). Reject the terminal (feed
            # the matching error back), run every sibling, and continue so the model decides
            # again with the contract in front of it.
            if silence_reason is None:
                self.log_warning(
                    f"{_NO_REPLY_TOOL} called with an unrecognized reason — rejecting; "
                    "the model must pick one of the declared values or reply")
                terminal_result = _invalid_silence_reason_result()
            else:
                self.log_warning(
                    f"{_NO_REPLY_TOOL} called after a prior attempt already exposed text — "
                    "rejecting; model must complete the reply")
                terminal_result = _INVALID_NO_REPLY_RESULT
            # Rejecting the terminal does not exempt the round from the turn's budget: the
            # siblings are ordinary productive work and dispatch in parallel like any other
            # round's. The terminal itself reserves a slot here exactly as it does on the
            # honored path, and `_terminal_overrides` still wins for that call below.
            suppressed = {
                **_free_call_overrides(calls, free_names, free_calls_cap - free_calls),
                **_over_budget_overrides(calls, free_names,
                                         config.max_tool_calls_per_turn - total_calls),
            }
            _charge(calls, suppressed)
            await _run_tool_round(
                self, registry, tool_context, sink, input_items, local_tool_calls,
                result_overrides={**suppressed,
                                  **_terminal_overrides(calls, terminal_call, terminal_result)})
            _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")],
                        tool_context)
            if _capped():
                self.log_warning(
                    f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — forcing final answer")
                tool_choice = "none"
            continue

        suppressed = _free_call_overrides(calls, free_names, free_calls_cap - free_calls)
        over_budget = _over_budget_overrides(calls, free_names,
                                             config.max_tool_calls_per_turn - total_calls)
        if over_budget:
            self.log_warning(
                f"Suppressed {len(over_budget)} productive call(s) past this turn's remaining "
                f"budget ({config.max_tool_calls_per_turn - total_calls}) — not dispatched; the "
                "round runs what the allowance covers and the rest are answered over_budget")
        suppressed = {**suppressed, **over_budget}
        _charge(calls, suppressed)
        await _run_tool_round(self, registry, tool_context, sink, input_items, local_tool_calls,
                              result_overrides=suppressed or None)
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")],
                        tool_context)

        if _capped():
            self.log_warning(
                f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — forcing final answer"
            )
            tool_choice = "none"


async def create_streaming_response_with_tool_loop(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    registry: ToolRegistry,
    tool_context: ToolContext,
    stream_callback: Callable[[Optional[str]], Any],
    tool_callback: Optional[Callable[[str, str], Any]] = None,
    prior_committed: bool = False,
    max_tool_rounds: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
    tool_choice: Optional[str] = None,
    free_tools: Optional[Iterable[str]] = None,
    aggregate_segments: bool = False,
    pre_round_input_callback: Optional[_PreRoundCallback] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Streaming response with local tool execution.

    Returns {"text", "tools_used", "local_tool_calls"}. Intermediate (tool) rounds don't
    stream text to the user; the final round streams normally and fires the completion flush.

    ``tool_choice`` seeds the FIRST round only; the loop still forces ``"none"`` on the final
    round as before. F37 passes ``"required"`` so the delivery-plan call cannot answer in prose
    and deliver nothing. It must be a named parameter, not a ``**params`` passthrough — the
    round call already sends ``tool_choice=`` explicitly, so a duplicate in ``params`` raises
    TypeError.

    ``prior_committed`` (F8): seeds the committed-text signal True when an EARLIER attempt
    this turn already exposed visible text (e.g. an MCP-failure retry after a partial
    reply), so a no_response_needed on this attempt is rejected rather than orphaning that
    partial as fake silence.

    ``max_tool_rounds`` / ``max_tool_calls`` override the config chat-turn caps for callers
    with a different round economy (F30.2: the research job spends a round per milestone
    report, so the 4-round chat default would strangle it).

    ``free_tools`` (F37) names BOOKKEEPING tools that must not compete with productive work for
    the budget. A round whose calls are ALL free costs neither a round nor a call. The caps
    exist to stop a runaway loop from billing forever; a status-card update is not the thing
    they are guarding against, and leaving it on the meter means a chatty todo list starves the
    build phase of the `mount_file` / `create_image_asset` calls it actually needs. Free rounds
    still have a ceiling of their own (``_FREE_ROUND_CEILING`` × the cap) so "free" can never
    mean "unbounded": a model looping on update_todos alone is a runaway too, just a cheaper one.
    A MIXED round (bookkeeping + real work) is fully productive — only the free calls in it ride
    free, so the round is charged normally.

    ``pre_round_input_callback`` is an ASYNC callable awaited once per round, immediately before
    the request; the list of input items it returns is appended to the running input. Generic,
    but scoped: a background job is the intended caller, folding in the mid-run notes that
    arrived while the previous round was working. See ``_inject_pre_round_items`` for the
    guarantees it is holding to — once per ROUND and not per HTTP attempt, after the round's
    replayed tool pairs, and never able to fail the round.
    """
    rounds_cap = int(max_tool_rounds) if max_tool_rounds is not None else config.max_tool_rounds
    calls_cap = (int(max_tool_calls) if max_tool_calls is not None
                 else config.max_tool_calls_per_turn)
    free_names = {str(n) for n in (free_tools or ())}
    free_rounds_cap = max(1, rounds_cap * _FREE_ROUND_CEILING)
    free_calls_cap = max(1, calls_cap * _FREE_ROUND_CEILING)
    budget = {"rounds": 0, "calls": 0, "free_rounds": 0, "free_calls": 0}

    def _suppress_excess_free(calls: List[Dict[str, Any]]) -> Dict[int, Any]:
        return _free_call_overrides(calls, free_names,
                                    free_calls_cap - budget["free_calls"])

    def _suppress_over_budget(calls: List[Dict[str, Any]]) -> Dict[int, Any]:
        return _over_budget_overrides(calls, free_names, calls_cap - budget["calls"])

    def _charge(calls: List[Dict[str, Any]], suppressed: Dict[int, Any]) -> None:
        """Bill a round. A round of PURE bookkeeping costs no round and no productive call;
        anything else is fully charged (the free calls riding in a mixed round are still free,
        the round is not).

        Free CALLS are counted too, not just free rounds — but only the ones that actually RAN.
        A suppressed call did no work, so billing it would let a burst exhaust the allowance
        without ever executing. It cannot loop on that forever: a round of nothing but free
        calls is still a free ROUND, and those have their own ceiling."""
        ran = [c for c in calls if id(c) not in suppressed]
        free = [c for c in ran if c.get("name") in free_names]
        productive = [c for c in ran if c.get("name") not in free_names]
        budget["free_calls"] += len(free)
        if not productive and calls:
            budget["free_rounds"] += 1
            return
        budget["rounds"] += 1
        budget["calls"] += len(productive)

    def _capped() -> bool:
        return (budget["rounds"] >= rounds_cap or budget["calls"] >= calls_cap
                or budget["free_rounds"] >= free_rounds_cap
                or budget["free_calls"] >= free_calls_cap)
    input_items: List[Dict[str, Any]] = list(messages)
    tools_used_all: List[str] = []
    local_tool_calls: List[Dict[str, Any]] = []
    # F30: expose this turn's live input (stable reference) + developer prompt/model so a
    # detached job can snapshot the full context by copy (see create_text_response_with_tool_loop).
    if tool_context is not None:
        tool_context.current_input = input_items
        tool_context.system_prompt = params.get("system_prompt")
        tool_context.model = params.get("model")
    # F2: track whether any visible reply text has streamed to Slack this turn. The round's
    # returned text is exactly what was forwarded to the stream callback (pre-tool-call
    # preamble is committed; post-call text is suppressed), so this is the committed-text
    # signal that decides whether a no_response_needed call is valid. Seeded by
    # prior_committed so a cross-attempt partial (F8) also counts as committed.
    visible_committed = bool(prior_committed)
    # Every round's visible text, in order — a pre-tool preamble and the post-tool text are
    # SEPARATE rounds. ``aggregate_segments`` (the chat handler) returns the seam-joined whole so
    # the thread remembers exactly what Slack showed instead of just the last round's "Fixed."
    # It is OPT-IN: internal consumers that treat this as a final-round-only stream — deep
    # research reads result["text"] as the report and never shows the intermediate "I'll search…"
    # preambles — must keep getting only the last round, or those preambles leak into the report.
    segments: List[str] = []

    while True:
        # W3, request side: picks up a container a BRIDGE tool created during the previous
        # round's dispatch, which lands in the holder after the response-side check has run.
        tools = await _adopt_and_pin(tool_context, params.get("artifacts_sink"), tools)
        # Last seam before the request goes out: everything the previous round produced is
        # already on the input, so a caller can add to it here (a job's mid-run steering notes)
        # and know the items sit after complete tool pairs.
        await _inject_pre_round_items(self, pre_round_input_callback, input_items)
        sink: List[Dict[str, Any]] = []
        text = await responses_api.create_streaming_response_with_tools(
            self,
            messages=input_items,
            tools=tools,
            stream_callback=stream_callback,
            tool_callback=tool_callback,
            function_call_sink=sink,
            tool_choice=tool_choice,
            **params,
        )
        # A seeded tool_choice (F37: "required") seeds the FIRST round ONLY. Left set, it would
        # force the SAME tool again on the next round — the model would be made to re-answer a
        # question it had already answered, and the second answer would overwrite the first.
        # "none" is the loop's own terminal state and must survive.
        if tool_choice not in (None, "none"):
            tool_choice = None
        if text:
            # Keep even a whitespace-only round: join_segments drops only truly empty ("")
            # segments, but a "\n" is real committed text the handler's buffer also keeps —
            # dropping it here would desync the returned aggregate from the Slack display.
            segments.append(text)
        if (text or "").strip():
            visible_committed = True
        # W3: after the response, before this round's local calls dispatch — so a bridge tool in
        # a mixed round finds the hosted container instead of minting a rival one, and so the
        # array the NEXT request sends already names it.
        tools = await _adopt_and_pin(tool_context, params.get("artifacts_sink"), tools)

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": join_segments(segments) if aggregate_segments else text,
                # The rounds this turn ran, in order — the SAME contract the non-streaming twin
                # returns, so a caller reads one key and not two spellings of it. Under
                # `aggregate_segments` these are exactly the pieces `text` was joined from,
                # BEFORE the seams went between them; otherwise `text` is the terminal one and
                # these are the rest.
                #
                # A consumer that has to transform the text — W4's marker parser is the one —
                # must work per round and re-join, because the seams belong to no round: a
                # transform applied to the joined string can consume a separator this loop
                # inserted, and then the caller's own per-round rendering (the streaming buffer)
                # and this text stop agreeing about the finished answer.
                "segments": list(segments),
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            silence_reason = _silence_reason(terminal_call)
            if silence_reason is not None and not visible_committed:
                # Nothing visible has posted yet — honor the terminal (silent turn), after
                # its siblings have run. The excess-bookkeeping suppression rides along so
                # that rule holds on this round exactly as it does on any other.
                suppressed = _suppress_excess_free(calls)
                outcome = await _handle_no_reply_terminal(
                    self, registry, tool_context, sink, input_items, calls, terminal_call,
                    silence_reason, tools_used_all, local_tool_calls,
                    remaining_budget=calls_cap - budget["calls"],
                    result_overrides=suppressed or None, tool_callback=tool_callback,
                    free_tools=free_tools, committed_text=text)
                if outcome is not None:
                    return outcome
                # The executor refused: the round ran (its committed preamble and outputs are
                # already on the input) and the turn owes words. Charge it and continue.
                _charge(calls, suppressed)
                if _capped():
                    self.log_warning(
                        f"Tool loop cap hit ({budget['rounds']} rounds / "
                        f"{budget['calls']} calls) — forcing final answer")
                    tool_choice = "none"
                continue
            # Either the reason is not one of the eight, or a visible reply already began (in
            # which case silence is INVALID — the model must finish what the room is reading).
            # Reject the terminal, run the siblings, and CONTINUE. WARNING = contract friction.
            if silence_reason is None:
                self.log_warning(
                    f"{_NO_REPLY_TOOL} called with an unrecognized reason — rejecting; "
                    "the model must pick one of the declared values or reply")
                terminal_result = _invalid_silence_reason_result()
            else:
                self.log_warning(
                    f"{_NO_REPLY_TOOL} called after visible text already streamed — rejecting; "
                    "model must complete the reply")
                terminal_result = _INVALID_NO_REPLY_RESULT
            # Rejecting the terminal does not exempt the round from the turn's budget: the
            # siblings are ordinary productive work and dispatch in parallel like any other
            # round's. The terminal itself reserves a slot here exactly as it does on the
            # honored path, and `_terminal_overrides` still wins for that call below.
            suppressed = {**_suppress_excess_free(calls), **_suppress_over_budget(calls)}
            _charge(calls, suppressed)
            _replay_committed_text(input_items, text)
            await _run_tool_round(
                self, registry, tool_context, sink, input_items, local_tool_calls, tool_callback,
                result_overrides={**suppressed,
                                  **_terminal_overrides(calls, terminal_call, terminal_result)})
            _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")],
                        tool_context)
            if _capped():
                self.log_warning(
                    f"Tool loop cap hit ({budget['rounds']} rounds / "
                    f"{budget['calls']} calls) — forcing final answer"
                )
                tool_choice = "none"
            continue

        suppressed = _suppress_excess_free(calls)
        if suppressed:
            self.log_warning(
                f"Suppressed {len(suppressed)} excess bookkeeping call(s) in one round — "
                "not dispatched; the model is told to make a single call")
        over_budget = _suppress_over_budget(calls)
        if over_budget:
            self.log_warning(
                f"Suppressed {len(over_budget)} productive call(s) past this turn's remaining "
                f"budget ({calls_cap - budget['calls']}) — not dispatched; the round runs what "
                "the allowance covers and the rest are answered over_budget")
        suppressed = {**suppressed, **over_budget}
        _charge(calls, suppressed)
        _replay_committed_text(input_items, text)
        await _run_tool_round(
            self, registry, tool_context, sink, input_items, local_tool_calls, tool_callback,
            result_overrides=suppressed or None,
        )
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")],
                        tool_context)

        if _capped():
            self.log_warning(
                f"Tool loop cap hit ({budget['rounds']} rounds / "
                f"{budget['calls']} calls) — forcing final answer"
            )
            tool_choice = "none"
