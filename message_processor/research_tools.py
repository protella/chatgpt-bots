"""F30 — background deep-dive research jobs.

A local tool (``start_deep_research``) for questions that genuinely need multi-source
investigation. Instead of answering inline, the model kicks off a background job: it acks in
one short line, the thread lock releases (chat keeps flowing), and a sourced findings report
lands in the SAME thread minutes later. Mirrors the background image-generation pattern
(``message_processor/handlers/image_gen.py``): snapshot context → detach an asyncio task →
deliver through the normal send path → cancel/await on shutdown.

The tool itself posts NOTHING (the model's own one-liner is the ack). The detached job makes
ONE non-streaming Responses call with web_search + configured MCP servers and NO local tools,
then posts the report with a compact provenance trailer. Errors/timeouts post an honest
one-line failure note — never silent.
"""
from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Dict, List, Optional
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
            "inline, or for opinions/chit-chat. After calling it, reply with ONE short line "
            "telling the requester the findings will land here shortly — no fake progress."
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
            },
            "required": ["task"],
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


async def execute_start_deep_research(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executor: enforce the per-thread cap, snapshot the current turn's context by copy, and
    detach the job. Posts nothing itself — returns a structured result the model relays as its
    one-line ack. Never raises (the loop wraps failures anyway; we return {"ok": False,...})."""
    if not config.enable_deep_research:
        return {"ok": False, "error": "disabled", "message": "Deep research is disabled."}
    task = (args.get("task") or "").strip()
    if not task:
        return {"ok": False, "error": "missing_task", "message": "A research task is required."}
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
        task=task, snapshot=snapshot, system_prompt=system_prompt, model=model)
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
    return {"ok": True, "status": "started", "task": _gist(task)}


async def _run_deep_research_job(*, processor, client, channel_id: str, thread_root: str,
                                 thread_key: str, job_id: str, task: str,
                                 snapshot: List[Dict[str, Any]], system_prompt: Optional[str],
                                 model: str) -> None:
    """The detached job: ONE non-streaming Responses call (web_search + MCP, no local tools),
    then deliver the report (or an honest failure note) to the originating thread."""
    started = time.monotonic()
    effort = clamp_effort(model, getattr(config, "deep_research_reasoning_effort", "high") or "high")
    verbosity = getattr(config, "deep_research_verbosity", "medium") or "medium"
    timeout_s = float(getattr(config, "deep_research_timeout", 600) or 600)
    tm = getattr(processor, "thread_manager", None)
    processor.log_info(
        f"Deep research {job_id} running for {thread_key} (model={model}, effort={effort}): "
        f"{_gist(task)!r}")
    try:
        # snapshot + an appended developer instruction to execute the task.
        job_input: List[Dict[str, Any]] = list(snapshot)
        job_input.append({"role": "developer",
                          "content": _RESEARCH_JOB_INSTRUCTION.format(task=task)})
        # web_search + configured MCP servers, NO local tools (registry=None). web_search is
        # FORCED into the job's tools regardless of the global toggle — this tool IS web
        # research; a truthy MCP-only array must not silently strip it (Codex review find).
        tools = processor._build_tools_array({}, model, registry=None) or []
        if not any(t.get("type") == "web_search" for t in tools):
            tools.append({"type": "web_search"})
        result = await asyncio.wait_for(
            processor.openai_client.create_text_response_with_tools(
                messages=job_input,
                tools=tools,
                model=model,
                system_prompt=system_prompt,
                reasoning_effort=effort,
                verbosity=verbosity,
                store=False,
                return_metadata=True,
            ),
            timeout=timeout_s,
        )
        text = result.get("text") if isinstance(result, dict) else str(result or "")
        text = (text or "").strip()
        elapsed = time.monotonic() - started
        if not text:
            processor.log_warning(f"Deep research {job_id} produced no text")
            await _deliver_failure(client, channel_id, thread_root,
                                   "the research came back empty")
            return
        trailer = f"\n\n_deep research · {_fmt_duration(elapsed)} · effort {effort}_"
        posted = await _deliver_findings(processor, client, channel_id, thread_root,
                                         text + trailer, task)
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
        # Shutdown/cancel — stay quiet and let finally clean up the registry.
        raise
    except asyncio.TimeoutError:
        processor.log_warning(f"Deep research {job_id} timed out after {timeout_s:.0f}s")
        await _deliver_failure(client, channel_id, thread_root,
                               "it ran past the time limit before finishing")
    except Exception as e:  # noqa: BLE001 — a job failure must post an honest note, never crash
        processor.log_error(f"Deep research {job_id} failed for {thread_key}: {e}", exc_info=True)
        await _deliver_failure(client, channel_id, thread_root, str(e)[:200] or "an unexpected error")
    finally:
        if tm is not None and hasattr(tm, "finish_research"):
            tm.finish_research(thread_key, job_id)


async def _deliver_findings(processor, client, channel_id: str, thread_root: str,
                            text: str, task: str) -> Optional[str]:
    """Post the findings through the normal send path (inherits markdown conversion +
    record_own_reply pulse recording). Optionally label the post with a chat.postMessage
    username override; on any failure (likely missing chat:write.customize) disable the label
    for the rest of the process and retry the plain path — delivery must never break."""
    label = None
    if (getattr(config, "enable_research_label", True)
            and not getattr(processor, _RESEARCH_LABEL_DISABLED_ATTR, False)):
        alias = (config.bot_name_aliases or ["bot"])[0]
        label = f"{alias} [research: {_gist(task, 40)}]"
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
