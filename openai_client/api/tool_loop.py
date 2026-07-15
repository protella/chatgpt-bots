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

from typing import Any, Callable, Dict, Iterable, List, Optional

from config import config
from message_markers import join_segments
from tool_registry import ToolContext, ToolRegistry, serialize_tool_result

from . import responses as responses_api


def _call_ok(result: Any) -> bool:
    """A tool result counts as successful unless it explicitly says ok=False."""
    return not (isinstance(result, dict) and result.get("ok") is False)


def _function_calls(sink: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The dispatchable function_call entries of a round's sink (reasoning items excluded)."""
    return [e for e in sink if e.get("type", "function_call") == "function_call"]


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
        local_tool_calls.append({"name": call.get("name"), "ok": ok,
                                 "gist": gist_from_arguments(call.get("arguments"))})
        self.log_info(f"Local tool '{call.get('name')}' -> {'ok' if ok else 'error'}")
        result_by_id[id(call)] = result
        await _notify(f"local:{call.get('name')}", "completed")

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


def _merge_used(tools_used_all: List[str], round_used: List[str]) -> None:
    for name in round_used:
        if name not in tools_used_all:
            tools_used_all.append(name)


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
_REACT_TOOL = "react_to_message"

# F2: fed back when no_response_needed is called AFTER visible reply text already
# streamed to Slack — the call is invalid; the model must finish the reply instead.
_INVALID_NO_REPLY_RESULT = {
    "ok": False,
    "error": "invalid_no_response_needed",
    "message": ("Invalid: you already began a visible reply — complete the reply instead "
                "of calling no_response_needed."),
}


def _no_reply_call(calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((c for c in calls if c.get("name") == _NO_REPLY_TOOL), None)


def _sanitize_reason(call: Dict[str, Any]) -> str:
    """Extract + sanitize the no_response_needed reason (control-stripped, length-capped)."""
    import json
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            args = {}
    reason = (args or {}).get("reason", "") if isinstance(args, dict) else ""
    reason = "".join(ch if ch.isprintable() else " " for ch in str(reason)).strip()
    return reason[:300]


async def _handle_no_reply_terminal(
    self,
    registry: ToolRegistry,
    tool_context: ToolContext,
    calls: List[Dict[str, Any]],
    terminal_call: Dict[str, Any],
    tools_used_all: List[str],
    local_tool_calls: List[Dict[str, Any]],
    remaining_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Terminal round: no_response_needed ends the turn. Only sibling react_to_message
    calls execute (filtered BEFORE dispatch — dispatch_all runs a round concurrently);
    other side-effect calls are suppressed with a logged skip. Returns a no_reply outcome
    with the (sanitized) reason; nothing is posted.

    F6 fix (b): react siblings still count against MAX_TOOL_CALLS_PER_TURN — the terminal
    branch runs before the loop's own cap check, so apply the remaining global budget to
    the react calls here.

    F4 fix: the terminal no_response_needed call itself consumes ONE slot of the remaining
    budget, so react siblings are capped at remaining_budget - 1 (floor 0) — the round can
    never exceed the cap by that one terminal call. Duplicate no_response_needed calls are
    suppressed: only the FIRST (``terminal_call``) is honored (first wins)."""
    react_calls = [c for c in calls if c.get("name") == _REACT_TOOL]
    if remaining_budget is None:
        react_budget = len(react_calls)
    else:
        react_budget = max(0, int(remaining_budget) - 1)  # reserve one slot for the terminal
    allowed_react_ids = {id(c) for c in react_calls[:react_budget]}
    # Only the first terminal call runs (identity match); duplicate no_response_needed
    # calls are dropped into `skipped` below and never dispatched.
    exec_calls = [c for c in calls
                  if c is terminal_call or id(c) in allowed_react_ids]
    exec_ids = {id(c) for c in exec_calls}
    skipped = [c.get("name") for c in calls if id(c) not in exec_ids]
    if skipped:
        self.log_info(f"{_NO_REPLY_TOOL} terminal — suppressing sibling calls: {skipped}")
    from message_processor.tool_provenance import gist_from_arguments

    results = await registry.dispatch_all(tool_context, exec_calls)
    for call, result in zip(exec_calls, results):
        if call.get("name") == _REACT_TOOL:
            local_tool_calls.append({"name": call.get("name"), "ok": _call_ok(result),
                                     "gist": gist_from_arguments(call.get("arguments"))})
    _merge_used(tools_used_all, [c.get("name") for c in exec_calls if c.get("name")])
    reason = _sanitize_reason(terminal_call)
    self.log_info(f"{_NO_REPLY_TOOL}: ending turn without posting — reason: {reason!r}")
    return {
        "text": "",
        "tools_used": tools_used_all,
        "local_tool_calls": local_tool_calls,
        "terminal_action": "no_reply",
        "reason": reason,
    }


async def create_text_response_with_tool_loop(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    registry: ToolRegistry,
    tool_context: ToolContext,
    prior_committed: bool = False,
    **params: Any,
) -> Dict[str, Any]:
    """Non-streaming response with local tool execution.

    Returns {"text", "tools_used", "local_tool_calls"} — ``local_tool_calls`` is the
    ordered [{"name", "ok"}] record of every local call (e.g. for reaction-only detection).

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
    # F30: expose this turn's live input (stable reference — appended in place across rounds)
    # plus the developer prompt/model, so a detached job can snapshot the full context by copy.
    if tool_context is not None:
        tool_context.current_input = input_items
        tool_context.system_prompt = params.get("system_prompt")
        tool_context.model = params.get("model")

    while True:
        sink: List[Dict[str, Any]] = []
        result = await responses_api.create_text_response_with_tools(
            self,
            messages=input_items,
            tools=tools,
            return_metadata=True,
            function_call_sink=sink,
            tool_choice=tool_choice,
            **params,
        )
        _merge_used(tools_used_all, result.get("tools_used") or [])

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": result.get("text", ""),
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            if not prior_committed:
                return await _handle_no_reply_terminal(
                    self, registry, tool_context, calls, terminal_call,
                    tools_used_all, local_tool_calls,
                    remaining_budget=config.max_tool_calls_per_turn - total_calls)
            # A prior attempt already exposed visible text — honoring silence now would
            # orphan it. Reject the terminal (feed an error back), run any siblings, and
            # continue so the model completes the reply.
            self.log_warning(
                f"{_NO_REPLY_TOOL} called after a prior attempt already exposed text — "
                "rejecting; model must complete the reply")
            rounds += 1
            total_calls += len(calls)
            await _run_tool_round(
                self, registry, tool_context, sink, input_items, local_tool_calls,
                result_overrides={id(terminal_call): _INVALID_NO_REPLY_RESULT})
            _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])
            if rounds >= config.max_tool_rounds or total_calls >= config.max_tool_calls_per_turn:
                self.log_warning(
                    f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — forcing final answer")
                tool_choice = "none"
            continue

        rounds += 1
        total_calls += len(calls)
        await _run_tool_round(self, registry, tool_context, sink, input_items, local_tool_calls)
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])

        if rounds >= config.max_tool_rounds or total_calls >= config.max_tool_calls_per_turn:
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
    """
    rounds_cap = int(max_tool_rounds) if max_tool_rounds is not None else config.max_tool_rounds
    calls_cap = (int(max_tool_calls) if max_tool_calls is not None
                 else config.max_tool_calls_per_turn)
    free_names = {str(n) for n in (free_tools or ())}
    free_rounds_cap = max(1, rounds_cap * _FREE_ROUND_CEILING)
    free_calls_cap = max(1, calls_cap * _FREE_ROUND_CEILING)
    budget = {"rounds": 0, "calls": 0, "free_rounds": 0, "free_calls": 0}

    def _suppress_excess_free(calls: List[Dict[str, Any]]) -> Dict[int, Any]:
        """Refuse the free calls in this round that exceed the burst cap or the remaining free
        allowance — BEFORE they are dispatched. Returns result_overrides for _run_tool_round,
        which answers them without running them (a function_call left without a
        function_call_output earns a 400 on the next request)."""
        if not free_names:
            return {}
        allowed = min(_FREE_CALLS_PER_ROUND, max(0, free_calls_cap - budget["free_calls"]))
        overrides: Dict[int, Any] = {}
        taken = 0
        for c in calls:
            if c.get("name") not in free_names:
                continue
            taken += 1
            if taken > allowed:
                overrides[id(c)] = _EXCESS_FREE_RESULT
        return overrides

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

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": join_segments(segments) if aggregate_segments else text,
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            if not visible_committed:
                # Nothing visible has posted yet — honor the terminal (silent turn).
                return await _handle_no_reply_terminal(
                    self, registry, tool_context, calls, terminal_call,
                    tools_used_all, local_tool_calls,
                    remaining_budget=calls_cap - budget["calls"])
            # A visible reply already began: no_response_needed is INVALID. Reject it
            # (feed an error back), run any siblings, and CONTINUE so the model completes
            # the reply into the same streamed message. WARNING = contract friction.
            self.log_warning(
                f"{_NO_REPLY_TOOL} called after visible text already streamed — rejecting; "
                "model must complete the reply")
            suppressed = _suppress_excess_free(calls)
            _charge(calls, suppressed)
            _replay_committed_text(input_items, text)
            await _run_tool_round(
                self, registry, tool_context, sink, input_items, local_tool_calls, tool_callback,
                result_overrides={**suppressed,
                                  id(terminal_call): _INVALID_NO_REPLY_RESULT})
            _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])
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
        _charge(calls, suppressed)
        _replay_committed_text(input_items, text)
        await _run_tool_round(
            self, registry, tool_context, sink, input_items, local_tool_calls, tool_callback,
            result_overrides=suppressed or None,
        )
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])

        if _capped():
            self.log_warning(
                f"Tool loop cap hit ({budget['rounds']} rounds / "
                f"{budget['calls']} calls) — forcing final answer"
            )
            tool_choice = "none"
