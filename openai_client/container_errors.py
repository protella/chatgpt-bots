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
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from config import config


def auto_container() -> Dict[str, str]:
    """The `{"type": "auto"}` declaration — an ephemeral sandbox the API provisions for us.

    A FACTORY, not a constant, because the size rides along: `{"type": "auto"}` alone lands the
    model in a 1 GB container (probed), which is where the kernel death this module now detects
    came from. A module-level dict would freeze whatever config said at import time, and every
    caller mutating a shared dict is its own bug.
    """
    return {"type": "auto", "memory_limit": config.code_interpreter_memory_limit}


# The API's own words for a dead container, e.g.
#   Container with id 'cntr_6a53…' not found.
# This is the ONLY reliable signal on the streaming path — see below.
_GONE_RE = re.compile(r"container with id\b.*\bnot found", re.IGNORECASE | re.DOTALL)

# The API's words for a sandbox whose kernel died, e.g.
#   {"type": "invalid_request_error", "message": "There was an issue with your request. …"}
# Generic by design on their side, so the detection below adds every other constraint it can.
_WEDGED_RE = re.compile(r"there was an issue with your request", re.IGNORECASE)


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


def is_container_wedged(exc: Exception) -> bool:
    """The sandbox is alive but can no longer run code: every exec fails the same generic way.

    Measured 2026-09-14: a kernel OOM inside a container leaves it reporting `status=running`
    forever, while every later `code_interpreter` exec fails in 2-5s with a bare `APIError`
    carrying `invalid_request_error` and the message below. A build's retry loop then re-entered
    the same corpse and burned its remaining attempts in seconds.

    This cannot PROVE a kernel death — the API's message is generic — so it is a SUSPICION, and
    the callers make a false positive cheap (one replacement per build, and the work in the old
    sandbox is banked before it is abandoned). Everything that can narrow it, does:

      * a status_code of None **or 400**. The status-less shape is the SSE stream's bare `APIError`
        and the RuntimeError `responses.py` builds out of a `response.failed` event. 400 was
        excluded at first on the reasoning that a genuine bad request arrives that way — right
        about the risk, wrong about the facts: measured 2026-09-15, the SAME wedged container
        called NON-streaming raises `BadRequestError(status_code=400)` carrying exactly this body,
        so excluding 400 meant the fallback path recovered nothing.
      * a mapping body must say `invalid_request_error`, and — this is what keeps a real bad
        request out — must name NO `param` and NO `code`. A malformed request says which part it
        objected to (`Unknown parameter: 'input[3].content[1].source'`); the wedge has
        `param: null, code: null` and a message that names nothing at all.
        `Mapping`, not `dict`: the SDK is free to hand back any mapping, and gating on `dict`
        alone let one whose type was `server_error` skip the restriction entirely and pass.
      * not a dead container, which has its own recovery.

    The callers' engage gate does the rest: recovery only runs when the request actually named an
    explicit container, and a false positive costs one replacement.
    """
    if getattr(exc, "status_code", None) not in (None, 400):
        return False
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        if body.get("type") != "invalid_request_error":
            return False
        if body.get("param") is not None or body.get("code") is not None:
            return False
        # A mapping that carries no usable message says nothing either way, so judge the
        # exception's own text rather than silently passing on an empty string.
        text = str(body.get("message") or "") or str(exc)
    else:
        text = str(exc)
    if not _WEDGED_RE.search(text):
        return False
    return not is_container_gone(exc)


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


# A container we suspect is wedged, recorded on the turn the same way adoption_blocked is.
#
# A wedge is invisible to everything that asks the API: `containers.retrieve()` answers `running`
# and the files are still listable. So the only place the knowledge lives is here, and it has to
# reach the two things that would otherwise walk straight back into the corpse — adoption (which
# pins the FIRST observed id, wedged or not) and the non-streaming fallback (whose own
# `containers_gone` list starts empty and cannot know).
_WEDGED_IDS_KEY = "wedged_container_ids"


def mark_container_wedged(artifacts_sink: Optional[List[Dict[str, Any]]],
                          ids: Optional[List[str]]) -> None:
    """Record ids as unusable for the rest of this turn. Never raises — bookkeeping."""
    if artifacts_sink is None or not ids:
        return
    try:
        artifacts_sink.append({_WEDGED_IDS_KEY: [cid for cid in ids if cid]})
    except Exception:  # noqa: BLE001 — a sink we cannot write to must not fail the recovery
        pass


def wedged_container_ids(artifacts_sink: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Every id marked wedged this turn, in encounter order."""
    out: List[str] = []
    for entry in (artifacts_sink or []):
        try:
            marked = entry.get(_WEDGED_IDS_KEY)
        except AttributeError:  # a malformed sink entry marks nothing
            continue
        if not isinstance(marked, list):
            continue
        for cid in marked:
            if isinstance(cid, str) and cid and cid not in out:
                out.append(cid)
    return out


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
            out.append({**tool, "container": auto_container()})
            changed = True
        else:
            out.append(tool)
    return out, changed
