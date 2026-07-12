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

from config import config

# The external `_Used Tools:_` attribution footer (handlers/text.py appends this to the
# VISIBLE message). It is deliberately user-facing chrome and must never reach model
# context. END-anchored: the remainder after the footer must be ONLY optional
# `[used tools: …]` / `[reactions: …]` annotation lines then end-of-string, so a
# following annotation can never shield the footer from stripping (F7-4) while an
# unrelated trailing bracket line no longer triggers a false match.
# Matches BOTH wordings: the footer reads "_Tools Used: …_" as of 2026-07-11, but Slack is the
# transcript and every reply posted before that still carries the old "_Used Tools: …_". Rebuilds
# read those old messages back, so the stripper must keep catching them or the stale chrome starts
# leaking into model context.
_USED_TOOLS_FOOTER_RE = re.compile(
    r'\n\n_(?:Tools Used|Used Tools):.+?_(?=(?:\n\[(?:used tools|reactions):[^\n\]]*\])*\s*$)')

# The model ECHOES its own annotations. `[used tools: …]` / `[tool results: …]` are injected into
# context as a record of what it previously did — and having seen them on every prior assistant
# turn, it happily writes one itself, which then ships to the user as gibberish chrome (observed
# live: a reply that read "56,088\n[used tools: code_interpreter]"). These lines are OURS to write
# and never the model's, so they are stripped from generated text unconditionally.
_PROVENANCE_ECHO_RE = re.compile(
    r'\n*^\[(?:used tools|tool results):[^\n\]]*\]\s*$', re.MULTILINE)


def strip_provenance_echo(text: str) -> str:
    """Remove any `[used tools: …]` / `[tool results: …]` line the MODEL wrote itself."""
    if not text or "[used tools:" not in text and "[tool results:" not in text:
        return text
    return _PROVENANCE_ECHO_RE.sub("", text).strip()


# Attribution answers "where did this information come from" — so it lists EXTERNAL sources only.
# code_interpreter is internal processing (the model doing arithmetic in a sandbox), not a source,
# and surfacing it just adds noise to every computed answer.
ATTRIBUTION_HIDDEN_TOOLS = {"code_interpreter"}


def visible_attribution_tools(tools_used) -> List[str]:
    """The tools worth telling the user about: external data sources, not internal plumbing."""
    return [t for t in (tools_used or []) if t not in ATTRIBUTION_HIDDEN_TOOLS]

# Budgets (F7, now env-backed per F14): entries/turn (config.tool_provenance_max_entries,
# default 20), gist chars (config.tool_provenance_gist_chars, default 80), annotation line
# budget (config.tool_provenance_line_budget, default 300). Read at CALL time via the tiny
# accessors below so annotation rendering stays a pure function of (row, config) — config is
# boot-constant, so rebuild determinism holds, and tests can monkeypatch the singleton.
def _max_provenance_entries() -> int:
    return int(getattr(config, "tool_provenance_max_entries", 20))


def _max_gist_chars() -> int:
    return int(getattr(config, "tool_provenance_gist_chars", 80))


def _max_annotation_chars() -> int:
    return int(getattr(config, "tool_provenance_line_budget", 300))

# F12: marker appended to a per-call MCP result digest when it is cut to the char cap.
TRUNCATION_MARKER = "… [truncated]"

# Structural arg keys whose values describe the SHAPE of a call (pagination/sizing/
# time-window) and are safe to persist — but ONLY when the value passes a per-key type
# check, so a caller can't smuggle content through a whitelisted key (e.g.
# before="https://x/?token=secret"). Every other value — non-allowlisted keys of ANY type
# (incl. numbers like token=123456), or allowlisted keys failing validation — is redacted
# to `<str>`, per the derived-artifacts rule (CLAUDE.md). Booleans are always safe.
#   * COUNT keys: value must be a real number (int/float, not bool).
#   * TS keys: value must look like a Slack ts / plain number (^\d+(\.\d+)?$).
_COUNT_GIST_KEYS = frozenset({
    "limit", "count", "max", "max_results", "n", "top_k", "k", "size", "num", "num_results",
    "days", "hours", "minutes", "page", "offset", "depth",
})
_TS_GIST_KEYS = frozenset({
    "oldest", "latest", "before", "after", "since", "until", "start", "end",
})
_TS_VALUE_RE = re.compile(r"^\d+(\.\d+)?$")


def _gist_render_value(key: str, value: Any) -> str:
    """Render ONE arg value for the gist, redacting anything that could carry content.

    Booleans and validated structural values pass through; everything else → `<str>`."""
    k = str(key).lower()
    if isinstance(value, bool):
        return str(value)  # booleans never carry content
    if isinstance(value, dict):
        return f"{{{len(value)}}}"
    if isinstance(value, list):
        return f"[{len(value)}]"
    if k in _COUNT_GIST_KEYS and isinstance(value, (int, float)):
        return str(value)  # count-like: only a real number is safe
    if k in _TS_GIST_KEYS and _TS_VALUE_RE.match(str(value)):
        return str(value)  # ts-like: only Slack-ts / plain-number shape is safe
    # Non-allowlisted key (any type, incl. numeric tokens), or an allowlisted key whose
    # value failed validation → NEVER the value itself.
    return "<str>"


def strip_used_tools_footer(content: Any) -> Any:
    """Remove the external `_Tools Used:_` footer from an assistant message body.

    Also removes the legacy `_Used Tools:_` wording, which is still present on every reply
    posted before 2026-07-11 and comes back on every rebuild (Slack is the transcript).

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
    is capped at config.tool_provenance_gist_chars. Returns "" when there are no usable args."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except (json.JSONDecodeError, ValueError):
            return ""
    if not isinstance(arguments, dict) or not arguments:
        return ""
    max_gist = _max_gist_chars()
    parts: List[str] = []
    for key, value in arguments.items():
        rendered = _gist_render_value(key, value).replace("\n", " ").replace("\r", " ").strip()
        if len(rendered) > 30:
            rendered = rendered[:27] + "…"
        parts.append(f"{key}={rendered}")
        if len(", ".join(parts)) >= max_gist:
            break
    return ", ".join(parts)[:max_gist]


def build_provenance(local_tool_calls: Optional[List[Dict[str, Any]]],
                     external_names: Optional[List[str]]) -> List[Dict[str, str]]:
    """Assemble the per-turn provenance list ``[{"tool_name", "gist"}]``.

    Local tool calls (the confabulation risk — history fetches, reactions, memory ops)
    come first with their arg-derived gists; external/built-in names (web_search, MCP)
    follow with empty gists (server-side calls expose no args here). Capped at
    config.tool_provenance_max_entries."""
    max_gist = _max_gist_chars()
    out: List[Dict[str, str]] = []
    for call in local_tool_calls or []:
        name = call.get("name")
        if not name:
            continue
        out.append({"tool_name": str(name), "gist": (call.get("gist") or "")[:max_gist]})
    for name in external_names or []:
        if name:
            out.append({"tool_name": str(name), "gist": ""})
    return out[:_max_provenance_entries()]


def render_used_tools_annotation(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render the reinjected annotation line, e.g.
    ``[used tools: fetch_channel_history(limit=50), web_search]``.

    Gists are included when the whole line fits config.tool_provenance_line_budget; otherwise
    it degrades to names only. Pure function of the (immutable) rows and boot-constant config,
    so every rebuild renders identically (determinism invariant F7-5)."""
    if not tools:
        return ""
    names: List[str] = []
    with_gist: List[str] = []
    for entry in tools:
        # F12: result-digest entries are a distinct class rendered by
        # render_tool_results_annotation — never list them on the [used tools:] line.
        if entry.get("result_digest"):
            continue
        name = entry.get("tool_name")
        if not name:
            continue
        names.append(str(name))
        gist = (entry.get("gist") or "").strip()
        with_gist.append(f"{name}({gist})" if gist else str(name))
    if not names:
        return ""
    full = f"[used tools: {', '.join(with_gist)}]"
    if len(full) <= _max_annotation_chars():
        return full
    return f"[used tools: {', '.join(names)}]"


def build_result_digests(mcp_results: Optional[List[Dict[str, Any]]],
                         per_call_chars: int, per_turn_chars: int) -> List[Dict[str, str]]:
    """Build the F12 result-digest entries ``[{"tool_name", "result_digest"}]`` from the
    captured ``mcp_call`` outputs (``[{"tool_name", "output"}]``, capture order).

    ONLY MCP outputs reach this — local Slack-fetch/read_document results never do
    (CLAUDE.md content rules; the caller passes MCP outputs only). Each output is
    newline-flattened (kept to one annotation line, and so a digest can't smuggle a fake
    ``[reactions: …]`` line into context), truncated to ``per_call_chars`` with a
    ``… [truncated]`` marker, and the turn is bounded by ``per_turn_chars`` in first-come
    order — once the running total reaches the cap, later calls store NO digest."""
    out: List[Dict[str, str]] = []
    used = 0
    for entry in mcp_results or []:
        name = entry.get("tool_name")
        output = entry.get("output")
        if not name or not output:
            continue
        if used >= per_turn_chars:
            break  # turn budget spent — later calls store no digest
        text = str(output).replace("\r", " ").replace("\n", " ").strip()
        if not text:
            continue
        if len(text) > per_call_chars:
            text = text[:per_call_chars] + TRUNCATION_MARKER
        out.append({"tool_name": str(name), "result_digest": text})
        used += len(text)
    return out


async def build_result_digests_summarized(
    mcp_results: Optional[List[Dict[str, Any]]],
    openai_client: Any,
    per_call_chars: int,
    per_turn_chars: int,
    input_chars: int,
) -> List[Dict[str, str]]:
    """F16 capture-time variant of :func:`build_result_digests`: an MCP output longer than
    ``per_call_chars`` is SUMMARIZED once (utility model, low effort) instead of hard-cut,
    so the URL/figure/title that made it worth keeping survives.

    Restructure honesty: the pure/deterministic budget + truncation logic stays in the sync
    :func:`build_result_digests`. This async pre-pass ONLY does the utility calls — it
    replaces each overlong output with its single-line summary (already under the cap) and
    hands the result to the pure builder, which then applies the unchanged per-turn budget in
    capture order (summaries count toward it) and truncates anything left over. Determinism
    holds because summarization happens ONCE here at persist time; the stored digest is
    immutable and rebuild never re-summarizes (F7-5).

    Fallback is total: on ANY summarizer error/timeout, an empty return, or a summary that
    is still over the cap, the ORIGINAL output is passed through unchanged so the pure
    builder applies today's ``… [truncated]`` behavior. The summarizer is fed at most the
    first ``input_chars`` of each output (budget guard). Never raises; never blocks on
    outputs that don't need summarizing."""
    prepared: List[Dict[str, Any]] = []
    for entry in mcp_results or []:
        name = entry.get("tool_name")
        output = entry.get("output")
        if not name or not output:
            prepared.append(entry)  # let the pure builder skip it uniformly
            continue
        text = str(output)
        if len(text) <= per_call_chars:
            prepared.append(entry)  # fits already — no utility call (verbatim path)
            continue
        summary = None
        try:
            summary = await openai_client.summarize_tool_result(text[:input_chars], per_call_chars)
        except Exception:
            summary = None  # defensive: the client contract is non-raising, but never trust it
        if summary:
            flat = str(summary).replace("\r", " ").replace("\n", " ").strip()
            # A summary that overshoots the cap is a failed summary — fall back to truncation
            # of the ORIGINAL output rather than truncating a lossy paraphrase.
            if flat and len(flat) <= per_call_chars:
                prepared.append({"tool_name": name, "output": flat})
                continue
        prepared.append(entry)  # summarizer failed/overlong → original → pure truncation
    return build_result_digests(prepared, per_call_chars, per_turn_chars)


def render_tool_results_annotation(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render the F12 reinjected block — one ``[tool results: <tool_name> → <digest>]``
    line per stored MCP digest, joined deterministically.

    Pure function of the immutable rows (F7-5 standard). Entries without a
    ``result_digest`` (the used-tools entries, and every old pre-F12 row) yield nothing."""
    if not tools:
        return ""
    lines: List[str] = []
    for entry in tools:
        digest = entry.get("result_digest")
        name = entry.get("tool_name")
        if not digest or not name:
            continue
        lines.append(f"[tool results: {name} → {digest}]")
    return "\n".join(lines)


def render_provenance_annotations(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Combined reinjection block in pinned order: ``[used tools: …]`` first, then the
    F12 ``[tool results: …]`` lines. The reactions annotation (rebuild-only) follows this
    block at the call site, keeping the strip → used-tools → tool-results → reactions
    order. Pure function of the immutable rows."""
    parts: List[str] = []
    used = render_used_tools_annotation(tools)
    if used:
        parts.append(used)
    results = render_tool_results_annotation(tools)
    if results:
        parts.append(results)
    return "\n".join(parts)
