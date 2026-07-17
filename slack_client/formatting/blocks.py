"""Render the content Slack delivers OUTSIDE a message's `text` field (F48).

Slack puts real user content in places our ingest never read. A TSV pasted into the
composer arrives as a Block Kit `table` block inside `attachments[]` with NO `files`
entry at all — so a message whose `text` is "what about this?" carried an 18-row table
the bot could not see. The same array carries link unfurls, quoted/forwarded messages,
and legacy webhook posts (Jira/GitHub/Drive) whose entire payload is in `fields[]`.

This is a PURE renderer. It takes a Slack message dict plus the caller's primary text
and returns plain text; it makes NO ownership decisions. Callers know `sender_type` and
MUST NOT pass our own bot's messages — our status cards, deep-research cards and welcome
chrome live in exactly these fields and would come back as "evidence" (the F47
attribution bug). Real bot answers always have canonical top-level text, so skipping
supplementary extraction for `sender_type == "self"` loses nothing.

Rendering mirrors what Slack itself puts in `text` — raw `<@U123>` mentions, `:emoji:`,
`<url|label>` links — so the caller's existing mention policy applies to supplementary
text exactly as it does to primary text. Callers must therefore combine RAW, THEN clean;
appending after the mention pass leaves `<@U…>` raw inside table cells.

Everything here fails open PER NODE: a bad cell yields an empty cell, never an erased
table (the real incident payload contains a literal JSON `null` cell). Every cap is
announced with an honest marker — no silent truncation.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default per-message supplementary budget. Callers that serialize into a tighter
# surface (the ChannelPulse ring) pass their own; everything else uses this, so a live
# message and the same message rebuilt from history serialize IDENTICALLY.
SUPPLEMENTARY_CHAR_BUDGET = 12000

# Slack's own legal maxima for a table block; the aggregate cell cap bounds the rendered
# body regardless of how many rows/columns a malformed payload claims.
_MAX_TABLE_ROWS = 100
_MAX_CELLS_PER_ROW = 20
_TABLE_CELL_CHAR_BUDGET = 10000
# Bounds malformed/adversarial nesting (e.g. attachments -> message_blocks -> …).
_MAX_DEPTH = 4
# Rich text nests deeper than block structure for legitimate reasons — a quoted message
# with a bulleted list is section -> list -> item-section -> text, ~5 levels — so it gets
# its own fresh, deeper counter. Total work stays bounded by _MAX_RICH_ELEMENTS.
_MAX_RICH_DEPTH = 12
_MAX_NODES = 200
_MAX_RICH_ELEMENTS = 200
_MAX_FIELDS = 50
# Below this a bracketed label plus any content plus a marker cannot coexist honestly.
_MIN_BUDGET = 120

# Provenance labels. Quoted/table/unfurl text must never flatten into the speaker's own
# words — a quoted third party becoming a claim by the uploader is an attribution bug.
_LABEL_TABLE = "Slack table"
_LABEL_BLOCK = "Slack message block"
_LABEL_ATTACHMENT = "Slack attachment"
_LABEL_UNFURL = "Link preview"
_LABEL_QUOTE = "Quoted Slack message"
_LABEL_FALLBACK = "Attachment fallback"

# `fallback` is a trap: on the real table payload its value is the literal
# "[no preview available]" — NOT the table content and NOT equal to `text`, so a
# "skip fallback when it equals text" rule never fires and the noise string reaches
# the model. These are dropped on sight (compared canonicalized).
_FALLBACK_PLACEHOLDERS = frozenset(
    {
        "[no preview available]",
        "no preview available",
        "(no preview available)",
        "[no text]",
        "[attachment]",
        "[image]",
    }
)

# Top-level `blocks` we never render: `rich_text` is a faithful duplicate of `text`
# (rendering it doubles the prompt), the rest are interactive/decorative chrome. NESTED
# rich_text (table cells, quoted messages) has no such duplicate and IS rendered.
_SKIP_TOP_BLOCKS = frozenset(
    {"rich_text", "divider", "actions", "input", "context_actions"}
)

_LINK_LABEL_RE = re.compile(r"<([^<>|]*)\|([^<>]*)>")
_LINK_BARE_RE = re.compile(r"<([^<>|]+)>")
_EMPHASIS = str.maketrans("", "", "*_~`")


def _canonical(text: Any) -> str:
    """Canonical form used ONLY for dedupe equality — never for output.

    Normalizes encoding differences so the same content in two shapes compares equal:
    HTML entities, Slack `<url|label>` links (-> visible label), mrkdwn emphasis (an
    attachment's `text` says "*bold*" where its `message_blocks` rich_text says "bold"),
    whitespace and case.
    """
    if not text:
        return ""
    s = html.unescape(str(text))
    s = _LINK_LABEL_RE.sub(lambda m: m.group(2) or m.group(1), s)
    s = _LINK_BARE_RE.sub(lambda m: m.group(1), s)
    s = s.translate(_EMPHASIS)
    return " ".join(s.split()).strip().lower()


def _flat(text: Any) -> str:
    """Single-line rendering for a table cell: the ` | ` row format must stay parseable."""
    return " ".join(str(text or "").split())


def _string(value: Any) -> str:
    """A display string, or "" for anything that isn't one. Never coerces dicts/lists."""
    return value.strip() if isinstance(value, str) else ""


def _text_object(obj: Any) -> str:
    """Slack composition object: {"type": "mrkdwn"|"plain_text", "text": "…"}."""
    return _string(obj.get("text")) if isinstance(obj, dict) else ""


@dataclass
class _Part:
    """One atomic candidate. Dedupe compares whole parts, so parts must be small:
    a field is a part, an unfurl title is a part — never a pre-joined blob."""

    label: str
    text: str
    # Table parts only: rows (header first) stay cuttable so a tight budget yields
    # "header + first rows + marker" instead of dropping the table whole.
    rows: Optional[List[str]] = field(default=None)
    rows_dropped: int = 0


class _Ctx:
    """Shared traversal budget. `dropped` counts structured nodes we refused to visit,
    so the caller can say so honestly instead of silently rendering less."""

    def __init__(self, max_nodes: int = _MAX_NODES) -> None:
        self.nodes_left = max_nodes
        self.dropped = 0

    def take(self) -> bool:
        if self.nodes_left <= 0:
            self.dropped += 1
            return False
        self.nodes_left -= 1
        return True


# --------------------------------------------------------------------- rich text


def _render_rich_text(
    elements: Any, ctx: _Ctx, depth: int, budget: Optional[List[int]] = None
) -> str:
    """Render a rich_text element tree the way Slack renders it into `text`.

    Styles (bold/italic) are dropped — the model needs the words, and emphasis markers
    inside table cells are noise. Structure (lists, quotes, code fences) is kept.
    """
    if depth > _MAX_RICH_DEPTH or not isinstance(elements, list):
        return ""
    if budget is None:
        budget = [_MAX_RICH_ELEMENTS]
    out: List[str] = []
    for el in elements:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        try:
            out.append(_render_rich_node(el, ctx, depth, budget))
        except Exception:
            logger.debug("blocks: rich_text node render failed", exc_info=True)
    joined = "".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _render_rich_node(el: Any, ctx: _Ctx, depth: int, budget: List[int]) -> str:
    if not isinstance(el, dict):
        return ""
    t = el.get("type")
    if t == "rich_text_section":
        return _render_rich_text(el.get("elements"), ctx, depth + 1, budget)
    if t == "rich_text_list":
        ordered = el.get("style") == "ordered"
        items = []
        for i, item in enumerate(el.get("elements") or []):
            body = _render_rich_text([item], ctx, depth + 1, budget)
            if body:
                items.append(f"{i + 1}. {body}" if ordered else f"- {body}")
        return "\n" + "\n".join(items) + "\n" if items else ""
    if t == "rich_text_quote":
        body = _render_rich_text(el.get("elements"), ctx, depth + 1, budget)
        if not body:
            return ""
        return "\n" + "\n".join(f"> {ln}" for ln in body.split("\n")) + "\n"
    if t == "rich_text_preformatted":
        body = _render_rich_text(el.get("elements"), ctx, depth + 1, budget)
        return f"\n```\n{body}\n```\n" if body else ""
    # Leaves. Mentions stay in RAW Slack syntax so the caller's mention policy resolves
    # them exactly as it resolves the ones in `text`.
    if t == "text":
        return str(el.get("text") or "")
    if t == "link":
        url = str(el.get("url") or "")
        label = str(el.get("text") or "")
        if not url:
            return label
        return f"<{url}|{label}>" if label and label != url else url
    if t == "user":
        uid = el.get("user_id")
        return f"<@{uid}>" if uid else ""
    if t == "channel":
        cid = el.get("channel_id")
        return f"<#{cid}>" if cid else ""
    if t == "usergroup":
        gid = el.get("usergroup_id")
        return f"<!subteam^{gid}>" if gid else ""
    if t == "broadcast":
        rng = el.get("range")
        return f"<!{rng}>" if rng else ""
    if t == "emoji":
        name = el.get("name")
        return f":{name}:" if name else ""
    if t == "date":
        return str(el.get("fallback") or el.get("timestamp") or "")
    return ""


# ------------------------------------------------------------------------ tables


def _render_cell(cell: Any, ctx: _Ctx, depth: int) -> str:
    """One table cell. Fails open to "" — the real payload has a literal JSON `null`
    cell (row 10, col 4), and `cell.get("type")` on it raises AttributeError. A single
    bad cell must never erase the table around it."""
    try:
        if cell is None:
            return ""
        if isinstance(cell, bool):
            return ""
        if isinstance(cell, (str, int, float)):
            return _flat(cell)
        if not isinstance(cell, dict):
            return ""
        ctype = cell.get("type")
        # Header cells are styled, so Slack sends them as rich_text while data cells are
        # raw_text. A naive cell["text"] read returns None for every header cell and
        # silently drops the whole header row, leaving the model unnamed columns.
        if ctype == "rich_text" or (
            ctype is None and isinstance(cell.get("elements"), list)
        ):
            return _flat(_render_rich_text(cell.get("elements"), ctx, 1))
        # raw_text carries `text`; raw_number and future scalar cells are read from the
        # documented display fields only — never by scraping arbitrary JSON.
        for key in ("text", "value", "number"):
            v = cell.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, (str, int, float)):
                return _flat(v)
        if isinstance(cell.get("elements"), list):
            return _flat(_render_rich_text(cell.get("elements"), ctx, 1))
        return ""
    except Exception:
        logger.debug("blocks: table cell render failed", exc_info=True)
        return ""


def _table_part(block: Dict[str, Any], ctx: _Ctx) -> Optional[_Part]:
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    lines: List[str] = []
    dropped = 0
    cell_chars = 0
    for idx, row in enumerate(rows):
        if len(lines) >= _MAX_TABLE_ROWS or cell_chars >= _TABLE_CELL_CHAR_BUDGET:
            dropped += len(rows) - idx
            break
        try:
            cells = row if isinstance(row, list) else []
            kept = cells[:_MAX_CELLS_PER_ROW]
            rendered = [_render_cell(c, ctx, 1) for c in kept]
            if len(cells) > len(kept):
                rendered.append(_omitted_marker(len(cells) - len(kept), "more column"))
            line = " | ".join(rendered)
        except Exception:
            logger.debug("blocks: table row render failed", exc_info=True)
            dropped += 1
            continue
        lines.append(line)
        cell_chars += len(line)
    if not lines:
        return None
    return _Part(
        label=_LABEL_TABLE,
        text=_table_text(lines, dropped),
        rows=lines,
        rows_dropped=dropped,
    )


def _omitted_marker(n: int, noun: str) -> str:
    return f"[… {n:,} {noun}{'' if n == 1 else 's'} omitted]"


def _table_text(rows: List[str], dropped: int, keep: Optional[int] = None) -> str:
    kept = rows if keep is None else rows[:keep]
    omitted = dropped + (len(rows) - len(kept))
    body = "\n".join(kept)
    if omitted:
        body = (
            f"{body}\n{_omitted_marker(omitted, 'more table row')}"
            if body
            else _omitted_marker(omitted, "more table row")
        )
    return body


# ------------------------------------------------------------------------ blocks


def _collect_blocks(
    blocks: Any,
    ctx: _Ctx,
    depth: int,
    parts: List[_Part],
    *,
    skip_top: bool = False,
    label_override: Optional[str] = None,
) -> None:
    if not isinstance(blocks, list) or depth > _MAX_DEPTH:
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if skip_top and block.get("type") in _SKIP_TOP_BLOCKS:
            continue
        if not ctx.take():
            return
        try:
            _render_block(block, ctx, depth, parts, label_override=label_override)
        except Exception:
            logger.debug("blocks: block render failed", exc_info=True)


def _render_block(
    block: Dict[str, Any],
    ctx: _Ctx,
    depth: int,
    parts: List[_Part],
    *,
    label_override: Optional[str] = None,
) -> None:
    btype = block.get("type")
    label = label_override or _LABEL_BLOCK

    def add(value: str, prefix: str = "") -> None:
        if value:
            parts.append(_Part(label=label, text=f"{prefix}{value}"))

    if btype == "table":
        part = _table_part(block, ctx)
        if part:
            if label_override:
                part.label = label_override
            parts.append(part)
        return
    if btype == "rich_text":
        add(_render_rich_text(block.get("elements"), ctx, 1))
        return
    if btype == "section":
        add(_text_object(block.get("text")))
        for f in (block.get("fields") or [])[:_MAX_FIELDS]:
            if not ctx.take():
                return
            add(_text_object(f))
        return
    if btype == "header":
        add(_text_object(block.get("text")))
        return
    if btype == "markdown":
        add(_string(block.get("text")))
        return
    if btype == "image":
        add(_text_object(block.get("title")))
        add(_string(block.get("alt_text")), "Image: ")
        return
    if btype == "video":
        add(_string(block.get("title")) or _text_object(block.get("title")))
        add(_string(block.get("description")) or _text_object(block.get("description")))
        add(_string(block.get("title_url")))
        return
    if btype == "context":
        for el in (block.get("elements") or [])[:_MAX_FIELDS]:
            if not isinstance(el, dict):
                continue
            if not ctx.take():
                return
            if el.get("type") == "image":
                add(_string(el.get("alt_text")), "Image: ")
            else:
                add(_text_object(el))
        return
    # Unknown/undocumented block types are skipped deliberately. Scraping arbitrary
    # JSON strings out of them would duplicate content and can leak opaque state
    # (button `value`s carry serialized context).


# ------------------------------------------------------------------- attachments


def _is_unfurl(att: Dict[str, Any]) -> bool:
    return bool(
        att.get("service_name")
        or att.get("title_link")
        or att.get("from_url")
        or att.get("original_url")
        or att.get("app_unfurl_url")
    )


def _render_attachment(att: Any, ctx: _Ctx, depth: int, parts: List[_Part]) -> None:
    if not isinstance(att, dict):
        return
    quoted = bool(att.get("message_blocks"))
    label = (
        _LABEL_QUOTE
        if quoted
        else (_LABEL_UNFURL if _is_unfurl(att) else _LABEL_ATTACHMENT)
    )

    def add(value: Any, prefix: str = "") -> None:
        s = _string(value)
        if s:
            parts.append(_Part(label=label, text=f"{prefix}{s}"))

    add(att.get("pretext"))
    add(att.get("author_name"), "Author: ")
    add(att.get("title"))
    # Unprefixed: a bare URL is self-evident, and it must stay able to dedupe against a
    # primary text that is only that link.
    add(att.get("title_link"))
    add(att.get("text"))
    for f in (att.get("fields") or [])[:_MAX_FIELDS]:
        if not isinstance(f, dict):
            continue
        if not ctx.take():
            break
        # Legacy webhook posts (Jira/GitHub/Drive) put their whole payload here with
        # `text` empty. The title rides in the part text so sibling fields sharing a
        # value ("Branch: main", "Target: main") stay distinct under dedupe.
        title, value = _string(f.get("title")), _string(f.get("value"))
        if title and value:
            add(f"{title}: {value}")
        else:
            add(value or title)
    # Where the table actually lives on the real incident payload.
    _collect_blocks(att.get("blocks"), ctx, depth + 1, parts)
    for mb in att.get("message_blocks") or []:
        if not isinstance(mb, dict):
            continue
        if not ctx.take():
            break
        inner = (
            (mb.get("message") or {}).get("blocks")
            if isinstance(mb.get("message"), dict)
            else None
        )
        _collect_blocks(inner, ctx, depth + 1, parts, label_override=_LABEL_QUOTE)
    add(att.get("footer"), "Footer: ")
    add(att.get("service_name"), "Via: ")
    add(att.get("image_url"), "Image: ")
    # Fallback is LOWEST priority but is NOT discarded: "Build #123 failed" can be the
    # only copy of the news while `text` says "See logs". Dedupe drops it when it merely
    # repeats something already rendered.
    fb = _string(att.get("fallback"))
    if fb and _canonical(fb) not in _FALLBACK_PLACEHOLDERS:
        parts.append(_Part(label=_LABEL_FALLBACK, text=fb))


# --------------------------------------------------------------------- assembly


def _dedupe(parts: List[_Part], primary_text: str) -> List[_Part]:
    """Drop ONLY exact canonical matches — never substrings. A title that merely appears
    inside a longer sentence is kept: a small duplicate is safer than deleting content.
    """
    seen = {_canonical(primary_text)}
    seen.discard("")
    out = []
    for p in parts:
        key = _canonical(p.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _assemble(parts: List[_Part], budget: int, dropped_nodes: int) -> str:
    chunks: List[str] = []
    used = 0
    cur_label: Optional[str] = None
    stopped: Optional[int] = None
    # Reserve room for the truncation marker up front, but only if we might need it:
    # an extraction that fits whole must not pay for a marker it will never print.
    # _measure omits per-label header costs, so it can under-report; reserve extra when a
    # dropped-node marker will also be appended after budgeting so both fit inside `budget`.
    full = _measure(parts)
    reserve = 0 if full <= budget else 96
    if dropped_nodes:
        reserve += 64

    for i, part in enumerate(parts):
        new_label = part.label != cur_label
        sep = "" if not chunks else ("\n\n" if new_label else "\n")
        head = f"[{part.label}]\n" if new_label else ""
        cost = len(sep) + len(head) + len(part.text)
        if used + cost <= budget - reserve:
            chunks.append(sep + head + part.text)
            used += cost
            cur_label = part.label
            continue
        # Doesn't fit. A table is cuttable at ROW boundaries, so a tight budget still
        # yields header + first rows + an honest marker rather than nothing.
        if part.rows:
            room = budget - reserve - used - len(sep) - len(head)
            fitted = _fit_table(part, room)
            if fitted:
                chunks.append(sep + head + fitted)
                used += len(sep) + len(head) + len(fitted)
                stopped = i + 1
                break
        stopped = i
        break

    out = "".join(chunks)
    if stopped is not None and stopped < len(parts):
        omitted = _measure(parts[stopped:])
        marker = (
            f"[Supplementary Slack content truncated; {omitted:,} characters omitted]"
        )
        out = f"{out}\n\n{marker}" if out else marker
    if dropped_nodes:
        out = (
            f"{out}\n{_omitted_marker(dropped_nodes, 'more Slack content item')}"
            if out
            else _omitted_marker(dropped_nodes, "more Slack content item")
        )
    out = out.strip()
    # Strict-bound safety net: the label-aware costs and post-budget markers can, on an
    # adversarial many-label/many-node payload, edge past `budget`. Clamp hard so the promised
    # bound holds, cutting at a whitespace boundary when one is near the end.
    if len(out) > budget:
        clamp = "\n[Supplementary Slack content truncated to budget]"
        keep = max(0, budget - len(clamp))
        head = out[:keep]
        sp = head.rfind(" ")
        if sp >= keep - 40:  # prefer a nearby word boundary, but never lose much content
            head = head[:sp]
        out = (head.rstrip() + clamp).strip()
    return out


def _measure(parts: List[_Part]) -> int:
    return sum(len(p.text) + 2 for p in parts)


def _fit_table(part: _Part, room: int) -> str:
    """Largest header-first prefix of the table that fits, with its own row marker."""
    if not part.rows or room < _MIN_BUDGET:
        return ""
    for keep in range(len(part.rows), 0, -1):
        text = _table_text(part.rows, part.rows_dropped, keep=keep)
        if len(text) <= room:
            return text
    return ""


def extract_supplementary_text(
    msg: Any, *, primary_text: str = "", budget: int = SUPPLEMENTARY_CHAR_BUDGET
) -> str:
    """Render the content Slack delivers outside `msg["text"]` as plain text.

    `primary_text` is the caller's RAW primary text; it seeds dedupe so an unfurl title
    that merely repeats the message is not sent twice. `budget` caps the result — every
    cut announces itself.

    Returns "" when there is nothing to add. Callers own the chrome decision: do NOT
    call this for `sender_type == "self"`.
    """
    if not isinstance(msg, dict) or budget < _MIN_BUDGET:
        return ""
    try:
        ctx = _Ctx()
        parts: List[_Part] = []
        # Top-level `blocks` skip rich_text: it is a faithful duplicate of `text`.
        _collect_blocks(msg.get("blocks"), ctx, 1, parts, skip_top=True)
        for att in msg.get("attachments") or []:
            if not ctx.take():
                break
            try:
                _render_attachment(att, ctx, 1, parts)
            except Exception:
                logger.debug("blocks: attachment render failed", exc_info=True)
        parts = _dedupe(parts, primary_text)
        if not parts and not ctx.dropped:
            return ""
        return _assemble(parts, budget, ctx.dropped)
    except Exception:
        logger.debug("blocks: supplementary extraction failed", exc_info=True)
        return ""


def extract_unfurls(msg: Any) -> List[Dict[str, str]]:
    """F51: structured link-preview associations Slack attached to a message.

    `extract_supplementary_text` renders unfurl text into message content but exposes no
    url→preview mapping. F51's link fetcher needs one: when a live fetch is blocked (paywall,
    SSRF, timeout), the preview Slack already resolved is the fallback summary — but ONLY when
    its url normalizes to the fetched link. Returns [{"url", "title", "text"}] for each unfurl
    attachment that carries an associated URL. Best-effort; never raises."""
    out: List[Dict[str, str]] = []
    if not isinstance(msg, dict):
        return out
    for att in msg.get("attachments") or []:
        if not isinstance(att, dict) or not _is_unfurl(att):
            continue
        url = _string(att.get("original_url") or att.get("from_url")
                      or att.get("title_link") or att.get("app_unfurl_url"))
        if not url:
            continue
        title = _string(att.get("title"))
        text = _string(att.get("text") or att.get("pretext"))
        out.append({"url": url, "title": title, "text": text})
    return out
