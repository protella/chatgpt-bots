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


def pin_container_tools(
    tools: Optional[List[Dict[str, Any]]],
    container_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """W3: bind every code_interpreter declaration to an explicit id — the inverse of demote.

    A turn now starts on `{"type": "auto"}`. The moment we learn which container the model is
    actually in (adoption) or make one for a bridge tool (mount_file / create_image_asset), the
    NEXT round's declaration has to name it: left on `auto`, the following request would provision
    a SECOND sandbox, and the file the model just wrote — or the one we just mounted for it —
    would be in a container it can no longer reach.

    Returns the SAME list object when nothing changed, so a caller can rebind unconditionally at
    every round boundary without copying the array (and without disturbing prompt caching).
    """
    if not tools or not container_id:
        return tools
    out: List[Dict[str, Any]] = []
    changed = False
    for tool in tools:
        if (isinstance(tool, dict) and tool.get("type") == "code_interpreter"
                and tool.get("container") != container_id):
            out.append({**tool, "container": container_id})
            changed = True
        else:
            out.append(tool)
    return out if changed else tools


# W3: the recovery path's veto on adoption, written into the attempt's artifacts sink.
#
# When `_create_with_container_recovery` engages it retries the call against a fresh EPHEMERAL
# sandbox. Any container id observed after that belongs to a throwaway the recovery minted, and
# binding it would make the next turn's "chart that file again" reach for a container that
# expires in minutes. An observed id cannot say which of the two it is, so the recovery raises a
# FLAG instead — set before the retry is issued, read by both adoption checkpoints.
_ADOPTION_BLOCKED_KEY = "adoption_blocked"


def mark_adoption_blocked(artifacts_sink: Optional[List[Dict[str, Any]]]) -> None:
    """Veto adoption for the rest of this turn. Never raises — bookkeeping, not delivery."""
    if artifacts_sink is None:
        return
    try:
        artifacts_sink.append({_ADOPTION_BLOCKED_KEY: True})
    except Exception:  # noqa: BLE001 — a sink we cannot write to must not fail the retry
        pass


def adoption_blocked(artifacts_sink: Optional[List[Dict[str, Any]]]) -> bool:
    """Has container recovery run on this turn? Then no observed id may be bound."""
    for entry in (artifacts_sink or []):
        try:
            if entry.get(_ADOPTION_BLOCKED_KEY):
                return True
        except AttributeError:  # a malformed sink entry is not a veto
            continue
    return False


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
