"""Code-interpreter container failure detection and recovery.

Lives here, not in `message_processor`, purely for layering: `message_processor` imports
`openai_client`, so the reverse would be a cycle. `message_processor.containers` re-exports
`is_container_gone` so callers there keep a natural import.

A persistent container id can die between the moment we verified it and the moment a Responses
call actually uses it — the tool loop makes one call per round, with minutes of tool work in
between. When that happens the API 404s and the user gets an error instead of an answer, which
is never an acceptable price for a sandbox nicety. `demote_container_tools` rewrites the tools
array to `{"type": "auto"}` so the call can be retried once against a fresh throwaway container.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

AUTO_CONTAINER: Dict[str, str] = {"type": "auto"}

# The API's own words for a dead container, e.g.
#   Container with id 'cntr_6a53…' not found.
# This is the ONLY reliable signal on the streaming path — see below.
_GONE_RE = re.compile(r"container with id\b.*\bnot found", re.IGNORECASE | re.DOTALL)


def is_container_gone(exc: Exception) -> bool:
    """Does this exception mean the container id we sent no longer exists?

    Two shapes, and the second one cost us a live bug. Non-streaming calls and
    `containers.retrieve()` raise `NotFoundError` with `status_code == 404`. But a container that
    dies mid-STREAM surfaces from the SSE iterator as a bare `openai.APIError` with **no
    status_code at all** — gating on 404 alone silently returned False there, the designed
    recovery never fired, and the turn only survived by falling through the generic
    non-streaming fallback (leaving an ERROR traceback and a Slack streaming_state_conflict
    behind). So match the message too.

    Deliberately NOT a bare "container" substring check: an unrelated 404 must never unbind a
    healthy container. The message pattern is specific enough to be safe on its own.
    """
    text = str(exc)
    if _GONE_RE.search(text):
        return True
    # Belt-and-braces for a 404 phrased some other way; still requires it to be container-shaped.
    return getattr(exc, "status_code", None) == 404 and "container" in text.lower()


def persistent_container_ids(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    """The explicit (string) container ids riding this tools array.

    `{"type": "auto"}` is a dict, so it is not one of these — only an id we chose and persisted
    can go stale, and only those are worth invalidating.
    """
    ids: List[str] = []
    for tool in tools or []:
        try:
            if tool.get("type") != "code_interpreter":
                continue
            container = tool.get("container")
            if isinstance(container, str) and container:
                ids.append(container)
        except AttributeError:  # a malformed tool entry is not our problem here
            continue
    return ids


def demote_container_tools(
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """Swap every explicit container id for `auto`, so a retry cannot hit the same dead id.

    Returns (tools, changed). `changed` is False when there was nothing to demote — in which case
    the 404 was not about a container we chose, and retrying would just fail identically.
    """
    if not tools:
        return tools, False
    out: List[Dict[str, Any]] = []
    changed = False
    for tool in tools:
        if (isinstance(tool, dict) and tool.get("type") == "code_interpreter"
                and isinstance(tool.get("container"), str)):
            out.append({**tool, "container": dict(AUTO_CONTAINER)})
            changed = True
        else:
            out.append(tool)
    return out, changed
