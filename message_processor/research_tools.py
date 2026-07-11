"""F30 — background deep-dive research jobs (F30.1 — Claude-parity status card).

A local tool (``start_deep_research``) for questions that genuinely need multi-source
investigation. Instead of answering inline, the model kicks off a background job, the thread
lock releases (chat keeps flowing), and a sourced findings report lands in the SAME thread
minutes later. Mirrors the background image-generation pattern
(``message_processor/handlers/image_gen.py``): snapshot context → detach an asyncio task →
deliver through the normal send path → cancel/await on shutdown.

F30.1: the model's ack reply is SUPPRESSED (handlers/text.py drops it, cued by
``ToolContext.deep_research_started``). In its place the detached job posts a live-updating
status card — the SAME labelled identity as the findings — then reads
"✓ Reported findings below." right before the report.

F30.2 (Claude-parity card content): the card's body lines are MODEL-AUTHORED milestones —
the job carries ONE local tool, report_progress, and the model ticks goal-level
accomplishments ("Searched X — found Y.") as it works. Raw web_search/MCP completions no
longer add body lines; they bump live activity counters in the card's context line
("todos as of H:MM · N web searches · …"). The headline ⏳ flips to ✅/❌ on finalize.

The job runs a streaming TOOL LOOP internally (web_search + configured MCP servers +
report_progress; round budget DEEP_RESEARCH_MAX_TOOL_ROUNDS) — never streamed to Slack; the
stream is consumed only to observe tool activity (driving the card) and to accumulate the
final text + tools_used. It then posts the report with a compact provenance trailer.
Errors/timeouts finalize the card and post an honest one-line failure note — never silent.
"""
from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from config import clamp_effort, config
from tool_registry import ToolContext, ToolRegistry

# Process-lifetime flag: set once a labelled findings post fails (likely a missing
# chat:write.customize scope) so we stop attempting the username override and post plainly
# for the rest of the process. Stored on the processor instance (see _deliver_findings).
_RESEARCH_LABEL_DISABLED_ATTR = "_research_label_disabled"

# Developer instruction appended to the snapshot when the detached job runs.
_RESEARCH_JOB_INSTRUCTION = (
    "Execute the following research task thoroughly and independently. This runs as a "
    "DETACHED background job: the user is not watching in real time and cannot answer "
    "follow-ups, so do not ask clarifying questions — investigate and report.\n\n"
    "TASK:\n{task}\n\n"
    "As you work, keep the live status card honest with the report_progress tool: call it "
    "once each time a major research goal completes, with a short past-tense line covering "
    "what you did and what it yielded. Aim for 2-5 milestones across the whole job — "
    "goal-level accomplishments, never one call per search — and report each one as you go; "
    "finish all milestone reporting BEFORE you start writing the findings report.\n\n"
    "Cross-check multiple independent sources before stating a conclusion. Produce a clear, "
    "well-structured findings report that leads with the direct answer, supports each key "
    "claim with a source and a link, notes where sources disagree, and states honestly what "
    "remains uncertain or could not be verified. End with a short list of the sources/links "
    "you relied on."
)


def get_start_deep_research_schema() -> dict:
    """Function-tool schema for start_deep_research (channels and DMs both allowed)."""
    return {
        "type": "function",
        "name": "start_deep_research",
        "description": (
            "Kick off a BACKGROUND deep-research job for a question that genuinely needs "
            "multi-source investigation — validating a claim, 'dig into X', a multi-part "
            "factual question — where a sourced, cross-checked report clearly beats a quick "
            "answer. The job runs after this turn ends and posts its findings back to THIS "
            "thread in a few minutes. Do NOT use it for anything a single web_search answers "
            "inline, or for opinions/chit-chat. Calling it posts a live status card that "
            "acknowledges the request on its own — your reply text will NOT be posted, so "
            "write NOTHING after the call and no preamble before it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The research question, restated FULLY and self-contained. The job runs "
                        "detached from this conversation, so include every detail it needs "
                        "(entities, constraints, what a good answer must cover) — do not rely on "
                        "conversational context that isn't restated here."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "SHORT topic tag for the research byline — 2-5 words, under 30 "
                        "characters, e.g. 'fast-casual 2026 performance'. A tag, not a "
                        "sentence: it renders as '[research: <label>]' next to every post "
                        "from this job, and Slack hard-truncates long bylines."
                    ),
                },
            },
            "required": ["task"],
        },
    }


def get_report_progress_schema() -> dict:
    """Function-tool schema for report_progress — the research job's ONLY local tool (F30.2).

    The model ticks goal-level milestones onto the live status card as it works, so the card
    reads as accomplished goals ('Searched X — found Y.') rather than a raw tool-call log."""
    return {
        "type": "function",
        "name": "report_progress",
        "description": (
            "Add a completed milestone to the live status card the user is watching. Call it "
            "each time a major research goal completes — NOT once per search. Pass one short "
            "past-tense line stating what you investigated and what it yielded, e.g. "
            "'Searched regulatory dockets and trade press for the final rule — found the "
            "March 2026 Federal Register entry.' Report milestones as you go, and finish all "
            "of them BEFORE you start writing the findings report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "milestone": {
                    "type": "string",
                    "description": ("One short past-tense line: the goal that completed and "
                                    "what it yielded."),
                },
            },
            "required": ["milestone"],
        },
    }


def _gist(task: str, limit: int = 60) -> str:
    """Compact single-line gist of the task for logs / acks / the label."""
    g = " ".join((task or "").split())
    if len(g) > limit:
        return g[:limit - 1].rstrip() + "…"
    return g


def _fmt_duration(seconds: float) -> str:
    """'Xm Ys' (or 'Ys' under a minute) for the findings trailer."""
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _research_label(processor, label_source: str) -> Optional[str]:
    """The chat.postMessage username override for the labelled research surfaces (status card
    AND findings), or None when labelling is off/disabled — so both carry the SAME identity.

    ``label_source`` is the model's short topic tag when it gave one (Claude-parity byline),
    else the full task text. Keeps the full bracketed label within Slack's 50-char
    server-side username cap. Returns None when ENABLE_RESEARCH_LABEL is off, or once a
    labelled post has failed this process (``_RESEARCH_LABEL_DISABLED_ATTR`` — likely a
    missing chat:write.customize scope)."""
    if not getattr(config, "enable_research_label", True):
        return None
    if getattr(processor, _RESEARCH_LABEL_DISABLED_ATTR, False):
        return None
    alias = (config.bot_name_aliases or ["bot"])[0]
    # Slack truncates chat.postMessage usernames at 50 chars SERVER-SIDE, so the gist budget
    # is whatever keeps the full label, bracket included, within 50.
    gist_budget = 50 - len(alias) - len(" [research: ") - 1
    return f"{alias} [research: {_gist(label_source, max(gist_budget, 8))}]"


# --- F30.1: live status card ---------------------------------------------------------------

# CONSTANT notification fallback across every chat.update — Slack badges "(edited)" only when
# the top-level `text` changes, so keeping it fixed lets blocks-only updates stay unbadged.
_CARD_FALLBACK_TEXT = "Deep research in progress…"
_CARD_MAX_TODO_LINES = 10          # visible todo lines before the tail collapses to "+N more"
_MILESTONE_GIST_CHARS = 300        # per-milestone gist cap (goal lines are 1-2 sentences)
# Headline budget derived from Slack's REAL limit — a section block's text caps at 3000
# chars — minus the worst-case rest of the body: (_CARD_MAX_TODO_LINES - 2) full milestone
# lines ("✓ " + gist), the "+N more" overflow line, the terminal line, newlines, and the
# status emoji (~150 chars of slack for those tail bits). Truncation only when the task
# genuinely can't fit, never at an arbitrary width.
_SECTION_TEXT_LIMIT = 3000
_CARD_HEADLINE_CHARS = (_SECTION_TEXT_LIMIT
                        - (_CARD_MAX_TODO_LINES - 2) * (_MILESTONE_GIST_CHARS + 2)
                        - 150)


def _card_throttle_s() -> float:
    """Card chat.update floor. Not a new magic number: Slack's real constraint on message
    updates is ~1/sec per message, already encoded as STREAMING_MIN_INTERVAL — the same knob
    the streaming path throttles on. Floor of 1.0 enforced."""
    return max(1.0, float(getattr(config, "streaming_min_interval", 1.0) or 1.0))


def _now_label(clock: Callable[[], time.struct_time] = time.localtime) -> str:
    """Sender-local 'H:MM AM/PM' stamp for the card's context line."""
    return time.strftime("%I:%M %p", clock()).lstrip("0")


class _ResearchCard:
    """Claude-parity live status card for a background research job (F30.1/F30.2).

    Posts ONE blocks message — a section block whose mrkdwn is the headline (⏳ + task gist,
    flipping to ✅/❌ on finalize) plus the model-authored milestone lines (report_progress),
    and a context block "todos as of H:MM AM/PM · N web searches · …" carrying live activity
    counters — with the same labelled identity as the findings, then updates it in place.
    Milestones are goal-level lines the model writes; raw tool events only bump the counters,
    so the card never degenerates into a per-search log. chat.update is THROTTLED to at most
    one per ``_card_throttle_s()`` (STREAMING_MIN_INTERVAL) with a TRAILING flush, so the last event is never left
    unshown. The notification `text` is CONSTANT (``_CARD_FALLBACK_TEXT``) across every update
    so Slack never badges "(edited)". Every op is best-effort: a card failure is logged and
    swallowed — it must NEVER kill the research job."""

    def __init__(self, *, processor, client, channel_id: str, thread_root: str, task: str,
                 label: Optional[str],
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = asyncio.sleep,
                 now_label: Callable[[], str] = _now_label):
        self.processor = processor
        self.client = client
        self.channel_id = channel_id
        self.thread_root = thread_root
        self.label = label
        self.ts: Optional[str] = None
        self._clock = clock
        self._sleep = sleep
        self._now_label = now_label
        self._task_gist = _gist(task, _CARD_HEADLINE_CHARS)
        self._status_emoji = "⏳"
        self._milestones: List[str] = []
        self._web_searches = 0
        self._mcp_calls: Dict[str, int] = {}
        self._terminal: Optional[str] = None
        self._dirty = False
        self._closed = False
        self._last_update: Optional[float] = None
        self._flush_task: Optional[asyncio.Future] = None
        self._lock = asyncio.Lock()

    # --- rendering ---
    def _visible_lines(self) -> List[str]:
        lines = [f"{self._status_emoji} {self._task_gist}"] + self._milestones
        if len(lines) > _CARD_MAX_TODO_LINES:
            # Loud, counted tail collapse — never a silent cap.
            hidden = len(lines) - (_CARD_MAX_TODO_LINES - 1)
            lines = lines[:_CARD_MAX_TODO_LINES - 1] + [f"… +{hidden} more milestones"]
        if self._terminal:
            lines.append(self._terminal)
        return lines

    def _context_line(self) -> str:
        """'todos as of H:MM' + live activity counters — the mechanical tool events live
        here as counts, not as body lines."""
        parts = [f"todos as of {self._now_label()}"]
        if self._web_searches:
            n = self._web_searches
            parts.append(f"{n} web search{'es' if n != 1 else ''}")
        for label, n in self._mcp_calls.items():
            parts.append(f"{n} {label} call{'s' if n != 1 else ''}")
        return " · ".join(parts)

    def _blocks(self) -> List[Dict[str, Any]]:
        return [
            {"type": "section",
             "text": {"type": "mrkdwn", "text": "\n".join(self._visible_lines())}},
            {"type": "context",
             "elements": [{"type": "mrkdwn", "text": self._context_line()}]},
        ]

    # --- posting / observing ---
    async def start(self) -> None:
        """Post the card immediately on job start. Labelled first; on a labelled-post failure
        (missing scope?) remember it process-wide and fall back to an unlabeled card — it still
        posts. A no-op when the client can't post cards."""
        client = self.client
        if not hasattr(client, "post_status_card"):
            return
        blocks = self._blocks()
        ts = None
        if self.label:
            ts = await client.post_status_card(
                self.channel_id, self.thread_root, _CARD_FALLBACK_TEXT, blocks,
                username=self.label)
            if ts is None:
                setattr(self.processor, _RESEARCH_LABEL_DISABLED_ATTR, True)
                self.processor.log_info(
                    "Research card label post failed (missing chat:write.customize?) — "
                    "posting unlabeled for the rest of this process")
                self.label = None
        if ts is None:
            ts = await client.post_status_card(
                self.channel_id, self.thread_root, _CARD_FALLBACK_TEXT, blocks)
        self.ts = ts

    async def add_milestone(self, text: str) -> None:
        """A model-authored goal line (report_progress) — the card's body content."""
        line = " ".join((text or "").split())
        if not line:
            return
        self._milestones.append(f"✓ {_gist(line, _MILESTONE_GIST_CHARS)}")
        await self._request_update()

    async def note_web_search(self) -> None:
        self._web_searches += 1
        await self._request_update()

    async def note_mcp(self, label: Optional[str]) -> None:
        key = label or "MCP"
        self._mcp_calls[key] = self._mcp_calls.get(key, 0) + 1
        await self._request_update()

    async def finalize_success(self) -> None:
        await self._finalize("✓ Reported findings below.", "✅")

    async def finalize_failure(self, reason: str) -> None:
        await self._finalize(f"✗ hit a wall: {_gist(reason, 80)}", "❌")

    async def finalize_cancelled(self) -> None:
        await self._finalize("✗ cancelled (bot shutting down)", "❌")

    # --- throttled update machinery ---
    async def _request_update(self) -> None:
        """Coalesce updates: flush now when the throttle window has elapsed, else schedule a
        SINGLE trailing flush that renders the latest state (never leaving the card stale)."""
        if self.ts is None or self._closed:
            return
        self._dirty = True
        now = self._clock()
        throttle = _card_throttle_s()
        if self._last_update is None or (now - self._last_update) >= throttle:
            await self._flush()
        elif self._flush_task is None or self._flush_task.done():
            delay = throttle - (now - self._last_update)
            self._flush_task = asyncio.ensure_future(self._delayed_flush(delay))

    async def _delayed_flush(self, delay: float) -> None:
        try:
            await self._sleep(delay)
        except asyncio.CancelledError:
            return
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if self.ts is None or self._closed or not self._dirty:
                return
            self._dirty = False
            self._last_update = self._clock()
            await self._safe_update(self._blocks())

    async def _finalize(self, terminal_line: str, status_emoji: str = "✅") -> None:
        """Force the final state out (bypassing the throttle) and close the card so any pending
        trailing flush becomes a no-op. Flips the headline ⏳ to the overall outcome emoji."""
        self._terminal = terminal_line
        self._status_emoji = status_emoji
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.ts is None:
                return
            await self._safe_update(self._blocks())

    async def _safe_update(self, blocks: List[Dict[str, Any]]) -> None:
        if not hasattr(self.client, "update_status_card"):
            return
        try:
            await self.client.update_status_card(
                self.channel_id, self.ts, _CARD_FALLBACK_TEXT, blocks)
        except Exception as e:  # noqa: BLE001 — card ops never break the research
            self.processor.log_debug(f"Research card update failed: {e}")


async def _consume_research_stream(processor, *, messages: List[Dict[str, Any]],
                                   tools: List[Dict[str, Any]], registry: ToolRegistry,
                                   tool_context: ToolContext, model: str,
                                   system_prompt: Optional[str], effort: str, verbosity: str,
                                   card: Optional["_ResearchCard"]) -> Dict[str, Any]:
    """Run the job's Responses tool loop as an INTERNAL stream (never streamed to Slack).

    F30.2: this is the streaming TOOL LOOP, not a single call — ``registry`` carries the
    job's one local tool (report_progress), so each model-authored milestone costs a round.
    The round budget is DEEP_RESEARCH_MAX_TOOL_ROUNDS (the chat-turn cap of 4 would strangle
    the 2-5 milestones the job instruction asks for). Accumulates the final round's text,
    feeds observed web_search/MCP completions to the status card as activity counters, and
    rebuilds ``tools_used`` from those same events (report_progress deliberately excluded —
    the trailer attributes research sources, not card bookkeeping). Returns
    ``{"text", "tools_used"}``."""
    observed: List[str] = []

    async def _stream_cb(_chunk):  # text deltas accumulate inside the API call, not posted
        return None

    async def _on_event(ev: Dict[str, Any]):
        kind = ev.get("kind")
        if kind == "web_search":
            if "web_search" not in observed:
                observed.append("web_search")
            if card is not None:
                await card.note_web_search()
        elif kind == "mcp":
            label = ev.get("server_label") or "mcp"
            if label not in observed:
                observed.append(label)
            if card is not None:
                await card.note_mcp(ev.get("server_label"))

    rounds_cap = max(1, int(getattr(config, "deep_research_max_tool_rounds", 10) or 10))
    result = await processor.openai_client.create_streaming_response_with_tool_loop(
        messages=messages, tools=tools, registry=registry, tool_context=tool_context,
        stream_callback=_stream_cb, tool_callback=None, tool_event_callback=_on_event,
        max_tool_rounds=rounds_cap, max_tool_calls=rounds_cap,
        model=model, system_prompt=system_prompt, reasoning_effort=effort,
        verbosity=verbosity, store=False)
    return {"text": (result.get("text") or "") if isinstance(result, dict) else (result or ""),
            "tools_used": observed}


async def execute_start_deep_research(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executor: enforce the per-thread cap, snapshot the current turn's context by copy, and
    detach the job. Posts nothing itself — returns a structured result the model relays as its
    one-line ack. Never raises (the loop wraps failures anyway; we return {"ok": False,...})."""
    if not config.enable_deep_research:
        return {"ok": False, "error": "disabled", "message": "Deep research is disabled."}
    task = (args.get("task") or "").strip()
    if not task:
        return {"ok": False, "error": "missing_task", "message": "A research task is required."}
    # Optional short byline tag (Claude-parity): a 2-5 word topic beats a truncated task gist
    # inside Slack's 50-char username cap. Whitespace-collapsed; _research_label still
    # bracket-safe-gists it, so an overlong tag degrades gracefully.
    label_hint = " ".join((args.get("label") or "").split())
    processor = getattr(ctx, "processor", None)
    client = getattr(ctx, "client", None)
    if processor is None or client is None:
        return {"ok": False, "error": "unavailable",
                "message": "Deep research isn't available right now."}

    channel_id = ctx.channel_id
    # Post findings back into the thread this turn is in; if triggered top-level, thread under
    # the triggering message (its ts becomes the thread root).
    thread_root = ctx.thread_ts or ctx.trigger_ts
    thread_key = f"{channel_id}:{thread_root}"
    tm = getattr(processor, "thread_manager", None)

    # Per-thread cap — friendly, structured rejection the model can relay (no global cap).
    cap = max(1, int(getattr(config, "deep_research_max_per_thread", 2)))
    in_flight = (tm.research_in_flight_count(thread_key)
                 if tm is not None and hasattr(tm, "research_in_flight_count") else 0)
    if in_flight >= cap:
        return {"ok": False, "error": "too_many_research_jobs",
                "message": (f"There {'is' if in_flight == 1 else 'are'} already {in_flight} "
                            f"research job{'s' if in_flight != 1 else ''} running in this thread "
                            f"(max {cap}). Let one finish before starting another.")}

    # Snapshot the CURRENT turn's full conversation input by DEEP COPY so mutations after this
    # call can't change what the job sees (CLAUDE.md: always full context, never a window).
    snapshot = copy.deepcopy(list(getattr(ctx, "current_input", None) or []))
    system_prompt = getattr(ctx, "system_prompt", None)
    model = getattr(ctx, "model", None) or config.gpt_model

    job_id = uuid4().hex[:12]
    if tm is not None and hasattr(tm, "register_research"):
        tm.register_research(thread_key, job_id, _gist(task))
    coro = _run_deep_research_job(
        processor=processor, client=client, channel_id=channel_id,
        thread_root=thread_root, thread_key=thread_key, job_id=job_id,
        task=task, label_hint=label_hint, snapshot=snapshot,
        system_prompt=system_prompt, model=model)
    try:
        task_handle = processor._schedule_async_call(coro)
    except Exception as e:  # scheduling failed — the job will never run; clear the registry
        coro.close()  # dispose the never-scheduled coroutine (no unawaited-coroutine warning)
        if tm is not None and hasattr(tm, "finish_research"):
            tm.finish_research(thread_key, job_id)
        processor.log_error(f"Failed to schedule deep research for {thread_key}: {e}", exc_info=True)
        return {"ok": False, "error": "schedule_failed",
                "message": "Couldn't start the research job. Please try again."}
    if task_handle is not None and tm is not None and hasattr(tm, "attach_research_task"):
        tm.attach_research_task(thread_key, job_id, task_handle)
    processor.log_info(f"Deep research {job_id} started for {thread_key}: {_gist(task)!r}")
    # F30.1: signal the turn's finalizer to DROP the model's ack reply — the status card the
    # job posts is the acknowledgment. Set on the shared ToolContext the tool loop exposes.
    try:
        ctx.deep_research_started = True
    except Exception:  # noqa: BLE001 — a defensive guard; ctx is a mutable dataclass
        pass
    return {"ok": True, "status": "started", "task": _gist(task)}


async def _run_deep_research_job(*, processor, client, channel_id: str, thread_root: str,
                                 thread_key: str, job_id: str, task: str,
                                 snapshot: List[Dict[str, Any]], system_prompt: Optional[str],
                                 model: str, label_hint: str = "") -> None:
    """The detached job: post the live status card, run ONE Responses call consumed as an
    INTERNAL stream (web_search + MCP, no local tools) to drive the card + accumulate the
    report, finalize the card, then deliver the report (or an honest failure note) to the
    originating thread. The card's final update always lands BEFORE the report so the thread
    reads top-down: card(done) → report."""
    started = time.monotonic()
    effort = clamp_effort(model, getattr(config, "deep_research_reasoning_effort", "high") or "high")
    verbosity = getattr(config, "deep_research_verbosity", "medium") or "medium"
    timeout_s = float(getattr(config, "deep_research_timeout", 600) or 600)
    tm = getattr(processor, "thread_manager", None)
    processor.log_info(
        f"Deep research {job_id} running for {thread_key} (model={model}, effort={effort}): "
        f"{_gist(task)!r}")
    # F30.1: the live status card is the acknowledgment (the model's ack reply was suppressed).
    # Post it immediately; it uses the same label as the findings (unlabeled if disabled).
    # The byline prefers the model's short topic tag over a truncated task gist.
    label_source = label_hint or task
    card = _ResearchCard(processor=processor, client=client, channel_id=channel_id,
                         thread_root=thread_root, task=task,
                         label=_research_label(processor, label_source))
    try:
        await card.start()
    except Exception as e:  # noqa: BLE001 — a card failure must never kill the research
        processor.log_debug(f"Research card start failed: {e}")
    # F30.2: the job's ONLY local tool — report_progress ticks model-authored milestones
    # onto the card. A job-local registry + context, never the chat registry.
    async def _report_progress(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        line = (args.get("milestone") or "").strip()
        if not line:
            return {"ok": False, "error": "missing_milestone",
                    "message": "A milestone line is required."}
        await card.add_milestone(line)
        return {"ok": True}

    job_registry = ToolRegistry()
    job_registry.register(get_report_progress_schema(), _report_progress)
    job_ctx = ToolContext(channel_id=channel_id, thread_ts=thread_root,
                          trigger_ts=thread_root, client=client, processor=processor)
    try:
        # snapshot + an appended developer instruction to execute the task.
        job_input: List[Dict[str, Any]] = list(snapshot)
        job_input.append({"role": "developer",
                          "content": _RESEARCH_JOB_INSTRUCTION.format(task=task)})
        # web_search + configured MCP servers + report_progress (the job's one local tool).
        # web_search is FORCED into the job's tools regardless of the global toggle — this
        # tool IS web research; a truthy MCP-only array must not silently strip it (Codex
        # review find).
        tools = processor._build_tools_array({}, model, registry=None) or []
        if not any(t.get("type") == "web_search" for t in tools):
            tools.append({"type": "web_search"})
        tools.append(get_report_progress_schema())
        # Internal streaming consumption bounds the WHOLE tool loop by the deep-research
        # timeout.
        result = await asyncio.wait_for(
            _consume_research_stream(
                processor, messages=job_input, tools=tools, registry=job_registry,
                tool_context=job_ctx, model=model, system_prompt=system_prompt,
                effort=effort, verbosity=verbosity, card=card),
            timeout=timeout_s,
        )
        text = (result.get("text") or "").strip()
        tools_used = result.get("tools_used") or []
        elapsed = time.monotonic() - started
        if not text:
            processor.log_warning(f"Deep research {job_id} produced no text")
            await card.finalize_failure("the research came back empty")
            await _deliver_failure(client, channel_id, thread_root,
                                   "the research came back empty")
            return
        # Visible tool attribution (same spirit as the F7 "Used Tools" footer on normal
        # turns — the findings post is out-of-band, so it carries its own).
        tool_bit = f" · tools: {', '.join(tools_used)}" if tools_used else ""
        trailer = f"\n\n_deep research · {_fmt_duration(elapsed)} · effort {effort}{tool_bit}_"
        # Card done BEFORE the findings post — the thread reads card(done) → report.
        await card.finalize_success()
        posted = await _deliver_findings(processor, client, channel_id, thread_root,
                                         text + trailer, label_source)
        if not posted:
            # send_message swallows Slack errors into None — a failed post must not be
            # logged as a completed job, and the thread deserves a note (best-effort; if
            # posting is broken the note fails too, but the ERROR log stays loud).
            processor.log_error(
                f"Deep research {job_id} finished but the findings post FAILED for {thread_key}")
            await _deliver_failure(client, channel_id, thread_root,
                                   "the findings were ready but posting them to Slack failed")
            return
        processor.log_info(f"Deep research {job_id} completed for {thread_key} in {elapsed:.1f}s")
    except asyncio.CancelledError:
        # Shutdown/cancel — best-effort final card update, then re-raise so the task cancels
        # and finally cleans up the registry.
        try:
            await card.finalize_cancelled()
        except Exception:  # noqa: BLE001
            pass
        raise
    except asyncio.TimeoutError:
        processor.log_warning(f"Deep research {job_id} timed out after {timeout_s:.0f}s")
        await card.finalize_failure("it ran past the time limit before finishing")
        await _deliver_failure(client, channel_id, thread_root,
                               "it ran past the time limit before finishing")
    except Exception as e:  # noqa: BLE001 — a job failure must post an honest note, never crash
        processor.log_error(f"Deep research {job_id} failed for {thread_key}: {e}", exc_info=True)
        reason = str(e)[:200] or "an unexpected error"
        await card.finalize_failure(reason)
        await _deliver_failure(client, channel_id, thread_root, reason)
    finally:
        if tm is not None and hasattr(tm, "finish_research"):
            tm.finish_research(thread_key, job_id)


async def _deliver_findings(processor, client, channel_id: str, thread_root: str,
                            text: str, label_source: str) -> Optional[str]:
    """Post the findings through the normal send path (inherits markdown conversion +
    record_own_reply pulse recording). Optionally label the post with a chat.postMessage
    username override built from ``label_source`` (the model's short topic tag, falling back
    to the task text); on any failure (likely missing chat:write.customize) disable the label
    for the rest of the process and retry the plain path — delivery must never break."""
    label = _research_label(processor, label_source)
    posted = None
    if label:
        try:
            posted = await client.send_message(channel_id, thread_root, text, username=label)
        except Exception as e:  # noqa: BLE001
            processor.log_debug(f"Labelled research post raised: {e}")
            posted = None
        if not posted:
            setattr(processor, _RESEARCH_LABEL_DISABLED_ATTR, True)
            processor.log_info(
                "Research label post failed (missing chat:write.customize?) — falling back to "
                "plain posts for the rest of this process")
    if not posted:
        posted = await client.send_message(channel_id, thread_root, text)
    return posted


async def _deliver_failure(client, channel_id: str, thread_root: str, reason: str) -> None:
    """Post an honest one-line failure note to the originating thread. Best-effort."""
    try:
        await client.send_message(
            channel_id, thread_root,
            f"⚠️ That deep-research job hit a wall: {reason}. Try asking again.")
    except Exception:
        pass


def register_research_tools(registry: ToolRegistry) -> None:
    """Register start_deep_research. Default (short) tool timeout — the executor only spawns
    the background task, it doesn't run the research itself."""
    registry.register(get_start_deep_research_schema(), execute_start_deep_research)
