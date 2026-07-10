"""Tool-use provenance (F7) — compact, deterministic records of the tools the bot
invoked on a turn, and the rendering/stripping helpers shared by the text handlers and
the thread-rebuild path.

The record is names + short arg-derived gists ONLY — never tool results or content
(CLAUDE.md derived-artifact rules). It is persisted keyed by the reply's Slack ts and
reinjected as a `[used tools: …]` annotation so the model can recall its own past tool
use instead of confabulating about it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# The external `_Used Tools:_` attribution footer (handlers/text.py appends this to the
# VISIBLE message). It is deliberately user-facing chrome and must never reach model
# context. END-anchored, but tolerant of trailing `[used tools: …]` / `[reactions: …]`
# annotations so a following annotation can never shield the footer from stripping (F7-4).
_USED_TOOLS_FOOTER_RE = re.compile(r'\n\n_Used Tools:.+?_(?=\n\[|\s*$)')

# Budgets (spec F7): <=8 entries/turn, gist <=~80 chars, annotation <=~160 chars.
MAX_PROVENANCE_ENTRIES = 8
MAX_GIST_CHARS = 80
MAX_ANNOTATION_CHARS = 160


def strip_used_tools_footer(content: Any) -> Any:
    """Remove the external `_Used Tools:_` footer from an assistant message body.

    A no-op for non-strings and for content without the footer. Used both at API-send
    time (keep external chrome out of model context) and BEFORE appending the F7
    annotation on rebuild/warm append (so the annotation is appended to already-clean
    text and the ordering stays: strip footer → [used tools:] → reactions)."""
    if not isinstance(content, str):
        return content
    return _USED_TOOLS_FOOTER_RE.sub('', content)


def gist_from_arguments(arguments: Any) -> str:
    """Deterministic short arg summary for a tool call, e.g. ``limit=50, before=169…``.

    Names are NOT included (the record carries tool_name separately). Nested values are
    summarized by kind+size, scalars are stringified and per-value capped; the whole gist
    is capped at MAX_GIST_CHARS. Returns "" when there are no usable args."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except (json.JSONDecodeError, ValueError):
            return ""
    if not isinstance(arguments, dict) or not arguments:
        return ""
    parts: List[str] = []
    for key, value in arguments.items():
        if isinstance(value, dict):
            rendered = f"{{{len(value)}}}"
        elif isinstance(value, list):
            rendered = f"[{len(value)}]"
        else:
            rendered = str(value)
        rendered = rendered.replace("\n", " ").replace("\r", " ").strip()
        if len(rendered) > 30:
            rendered = rendered[:27] + "…"
        parts.append(f"{key}={rendered}")
        if len(", ".join(parts)) >= MAX_GIST_CHARS:
            break
    return ", ".join(parts)[:MAX_GIST_CHARS]


def build_provenance(local_tool_calls: Optional[List[Dict[str, Any]]],
                     external_names: Optional[List[str]]) -> List[Dict[str, str]]:
    """Assemble the per-turn provenance list ``[{"tool_name", "gist"}]``.

    Local tool calls (the confabulation risk — history fetches, reactions, memory ops)
    come first with their arg-derived gists; external/built-in names (web_search, MCP)
    follow with empty gists (server-side calls expose no args here). Capped at
    MAX_PROVENANCE_ENTRIES."""
    out: List[Dict[str, str]] = []
    for call in local_tool_calls or []:
        name = call.get("name")
        if not name:
            continue
        out.append({"tool_name": str(name), "gist": (call.get("gist") or "")[:MAX_GIST_CHARS]})
    for name in external_names or []:
        if name:
            out.append({"tool_name": str(name), "gist": ""})
    return out[:MAX_PROVENANCE_ENTRIES]


def render_used_tools_annotation(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render the reinjected annotation line, e.g.
    ``[used tools: fetch_channel_history(limit=50), web_search]``.

    Gists are included when the whole line fits MAX_ANNOTATION_CHARS; otherwise it
    degrades to names only. Pure function of the (immutable) rows, so every rebuild
    renders identically (determinism invariant F7-5)."""
    if not tools:
        return ""
    names: List[str] = []
    with_gist: List[str] = []
    for entry in tools:
        name = entry.get("tool_name")
        if not name:
            continue
        names.append(str(name))
        gist = (entry.get("gist") or "").strip()
        with_gist.append(f"{name}({gist})" if gist else str(name))
    if not names:
        return ""
    full = f"[used tools: {', '.join(with_gist)}]"
    if len(full) <= MAX_ANNOTATION_CHARS:
        return full
    return f"[used tools: {', '.join(names)}]"
