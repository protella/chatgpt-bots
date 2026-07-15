"""
Shared continuation-marker and message-splitting helpers.

Single source of truth for the "Continued..." markers used when a long reply is
split across multiple Slack messages, plus the helpers that keep splits safe
(code fences closed/reopened, Slack <entities> never cut mid-token) and the
rebuild-side strippers that merge split parts back into ONE assistant turn.

Writers (slack_client/messaging.py, message_processor/handlers/text.py) and the
rebuild reader (message_processor/thread_management.py) must both import from
here so the marker shapes can never drift apart. Phase G's native streaming
path must build on these same helpers.
"""
import re
from typing import List, Tuple

# Invisible tag (F1) appended to the bot's own progress-checklist messages so the
# history rebuild can recognize and drop them — their "✓ …" rendering doesn't match the
# ":emoji: text" self-status filter, and if a rebuild fires mid-generation (the checklist
# is still visible) they must never replay as an assistant turn. Three INVISIBLE
# SEPARATORs (U+2063): render nothing, survive chat.update, and can't collide with human
# text. Recognized on read; users never see it.
CHECKLIST_STATUS_MARKER = "⁣⁣⁣"


def is_checklist_status_text(text: str) -> bool:
    """True if `text` carries the checklist marker (an own progress-checklist message)."""
    return bool(text) and CHECKLIST_STATUS_MARKER in text


def segment_separator(prev: str, nxt: str) -> str:
    """The boundary between two consecutive streamed round-segments — a pre-tool preamble and
    the post-tool text of the next round.

    Each local-tool round is its own API call, and the streaming buffer simply concatenates
    the rounds, so without a separator the seam jams: "…under Super Heavy." + "Fixed." renders
    as "Super Heavy.Fixed." Insert a paragraph break, but ONLY when the model didn't already
    leave whitespace on either side (so we never double-space a seam it wrote itself). The
    separator is APPENDED between segments — never inserted retroactively before already-sent
    text — so an append-only sink stays append-only. Used in two places that must agree: the
    handler injects it into the live buffer (Slack display) and the tool loop joins its
    round-segments with it (the returned/persisted canonical text), so what the user sees and
    what the thread remembers are the same string."""
    if not prev or not nxt:
        return ""
    if prev[-1].isspace() or nxt[0].isspace():
        return ""
    return "\n\n"


def join_segments(segments: List[str]) -> str:
    """Join round-segments into the one canonical response, applying ``segment_separator`` at
    every seam. Empty segments (tool-only rounds) contribute nothing and never leave a dangling
    separator behind."""
    out = ""
    for seg in segments:
        if not seg:
            continue
        out += segment_separator(out, seg) + seg
    return out


# Canonical marker set ("Continued..." style — user preference).
_TRAILER_BODY = "Continued in next message..."
_HEAD_BODY = "...continued"
CONTINUATION_TRAILER = f"*{_TRAILER_BODY}*"
CONTINUATION_HEAD = f"*{_HEAD_BODY}*"

# Legacy/alternate shapes still present in Slack history; recognized on read, never
# written by the mrkdwn writers. The markdown-flavored variants cover the native
# streaming path (chat.startStream markdown_text): Slack converts markdown **bold**
# to stored mrkdwn *bold* (canonical), but the raw/italic forms are recognized too
# in case a message is stored unconverted.
_LEGACY_TRAILERS = (
    "*...continued in next message...*",  # messaging-layer backup truncation
    f"**{_TRAILER_BODY}**",               # markdown bold, stored verbatim
    f"_{_TRAILER_BODY}_",                 # markdown italic converted to mrkdwn
)
_TRAILER_VARIANTS = (CONTINUATION_TRAILER,) + _LEGACY_TRAILERS
_HEAD_VARIANTS = (CONTINUATION_HEAD, f"**{_HEAD_BODY}**", f"_{_HEAD_BODY}_")
# "*Part 2 (continued)*" (streaming) and "*Part 1/3*" (old non-streaming) prefixes;
# [*_]{1,2} accepts the markdown-written (native streaming) sibling shapes.
_PART_PREFIX_RE = re.compile(r"^\s*[*_]{1,2}Part \d+(?: \(continued\)|\s*/\s*\d+)[*_]{1,2}\s*\n+")


def part_prefix(part: int) -> str:
    """Prefix for the Nth (N>1) message of a split reply."""
    return f"*Part {part} (continued)*\n\n"


def continuation_trailer() -> str:
    """Trailer appended to a message that continues in the next one."""
    return f"\n\n{CONTINUATION_TRAILER}"


def part_prefix_markdown(part: int) -> str:
    """part_prefix for markdown-input sinks (chat.startStream markdown_text).

    Slack converts markdown **bold** to mrkdwn *bold* on store, so this lands in
    history as exactly ``part_prefix(part)`` — the shape the rebuild merger strips.
    """
    return f"**Part {part} (continued)**\n\n"


def continuation_trailer_markdown() -> str:
    """continuation_trailer for markdown-input sinks; stores as the canonical shape."""
    return f"\n\n**{_TRAILER_BODY}**"


def ends_with_continuation(text: str) -> bool:
    if not text:
        return False
    stripped = text.rstrip()
    return any(stripped.endswith(t) for t in _TRAILER_VARIANTS)


def starts_as_continuation(text: str) -> bool:
    if not text:
        return False
    if _PART_PREFIX_RE.match(text):
        return True
    stripped = text.lstrip()
    return any(stripped.startswith(h) for h in _HEAD_VARIANTS)


def strip_continuation_markers(text: str) -> str:
    """Remove part prefixes and continuation trailers/heads from a message body."""
    if not text:
        return text
    t = _PART_PREFIX_RE.sub("", text)
    lstripped = t.lstrip()
    for head in _HEAD_VARIANTS:
        if lstripped.startswith(head):
            t = lstripped[len(head):].lstrip("\n ")
            break
    stripped = t.rstrip()
    changed = True
    while changed:
        changed = False
        for trailer in _TRAILER_VARIANTS:
            if stripped.endswith(trailer):
                stripped = stripped[: -len(trailer)].rstrip()
                changed = True
    return stripped


def entity_safe_cut(text: str, limit: int) -> int:
    """Largest cut index <= limit that doesn't split a Slack <entity> (mention/URL).

    Falls back to `limit` when the text before it is one giant unclosed entity
    (pathological; better a broken entity than an infinite loop).
    """
    if len(text) <= limit:
        return len(text)
    lt = text.rfind("<", 0, limit)
    if lt > 0 and text.rfind(">", lt, limit) == -1:
        return lt
    return limit


def _fence_state(text: str) -> Tuple[bool, str]:
    """Walk ``` fences in text (assumed to start outside a block); return (open?, lang)."""
    in_block = False
    lang = ""
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            if in_block:
                in_block, lang = False, ""
            else:
                in_block, lang = True, s[3:].strip()
    return in_block, lang


def fence_safe_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into <=chunk_size pieces at paragraph/sentence boundaries,
    hard-wrapping oversized fragments entity-safely, then close/reopen code
    fences across the seams so no chunk ever renders a shattered block.
    Markers are NOT added here — callers own presentation."""
    if chunk_size <= 10:
        chunk_size = 10

    # Pass 1: raw chunks on paragraph, then sentence boundaries; hard-wrap leftovers.
    raw_chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            raw_chunks.append(current.strip())
        current = ""

    def append_fragment(fragment: str):
        nonlocal current
        while len(fragment) > chunk_size:
            flush()
            cut = entity_safe_cut(fragment, chunk_size)
            if cut <= 0:
                cut = chunk_size
            raw_chunks.append(fragment[:cut].strip())
            fragment = fragment[cut:]
        if len(current) + len(fragment) + 2 <= chunk_size:
            current += fragment + "\n\n"
        else:
            flush()
            current = fragment + "\n\n"

    for para in text.split("\n\n"):
        if len(para) > chunk_size:
            # Sentence-level attempt before hard wrapping.
            for sentence in para.replace(". ", ".\n").split("\n"):
                append_fragment(sentence)
        else:
            append_fragment(para)
    flush()

    # Pass 2: fence continuity across seams.
    result: List[str] = []
    open_block, open_lang = False, ""
    for chunk in raw_chunks:
        body = (f"```{open_lang}\n" if open_block else "") + chunk
        open_block, open_lang = _fence_state(body)
        if open_block:
            body += "\n```"
        result.append(body)
    return result
