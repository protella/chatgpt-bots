"""F36 — Slack canvases the bot can actually manage: create, read, edit, list.

A canvas is the right home for something a thread keeps coming back to — a running spec, a
launch checklist, a summary that gets amended — where a chat message would be buried in an hour
and a posted file would fork into `_final_v3`. It is the one surface in Slack that is meant to
be EDITED rather than appended to, so it is where durable work belongs.

Probed live against the API (2026-07-12), because the docs leave the important parts out:

* **We create the CHANNEL canvas** (`conversations.canvases.create`), not a standalone one
  (`canvases.create`). Only the channel canvas gets a TAB at the top of the channel, and a tab is
  the whole game: Slack posts no message when a canvas is shared, so an untabbed canvas appears
  NOWHERE in the channel — not in history, not in the rebuilt transcript, nowhere a human or the
  model would trip over it. Standalone canvases cannot be pinned (`pins.add` refuses both the
  share ts and the file id) and their "bookmark" is write-only (see `execute_create_channel_canvas`).
* **`title` is undocumented and absent from the slack_sdk signature — and it works**, because the
  signature takes `**kwargs`. It labels the tab. Pass it or the canvas is `Untitled` FOREVER: there
  is no rename (`files.rename` / `canvases.setTitle` / `conversations.canvases.setTitle` are all
  `unknown_method`, `files.edit` is `not_allowed_token_type`), so creation is the only chance.
* It is **not idempotent** — call it twice and the channel has two canvases and two tabs. The
  create tool is a factory that vanishes once a canvas exists, so the mistake is unmakeable.
* `properties.canvas` on `conversations.info` is **null even when a channel canvas exists**. The
  real record is `properties.tabs` — and a tab OUTLIVES its canvas for a while, so it has to be
  cross-checked against a live file list (`_channel_canvas_id`).
* A canvas IS a file — which is why `files.info` describes it and `files.list(types="canvases")`
  enumerates it. Read/edit/delete work the same on any canvas, so the tools below still handle
  standalone canvases a human made.
* **There is no `canvases.read`.** Content comes back by downloading `url_private`, and it
  arrives as **HTML**, not the markdown that went in. Round-tripping therefore needs a
  converter; `_html_to_markdown` below is it.
* Editing a specific passage is a two-step: `canvases.sections.lookup(contains_text=…)` hands
  back a section id (`temp:C:…`), which `canvases.edit(operation="replace", section_id=…)` then
  targets. Without the lookup the only operations are insert-at-start/end.
* A *standalone* canvas is visible to NOBODY until `canvases.access.set(channel_ids=[…])`. The
  channel canvas needs no such call — it belongs to the channel and Slack shares it in on
  creation (`files.info` shows the channel under `shares`, `source: CHANNEL_TAB`).

**Authorization.** A canvas id is a Slack file id, and a file id from another channel is still a
valid file id. So editing is gated on the canvas being shared into THIS channel — checked live
against `files.info`, not against a list we built earlier. Editing the wrong canvas silently
rewrites someone else's work, and unlike a bad chat message there is no version of that a user
would ever see coming.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Set

from config import config
from logger import setup_logger
from tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.CanvasTools")

# A canvas is a document, not a chat message: generous, but not unbounded.
MAX_MARKDOWN_CHARS = 12000
# How much of a canvas we hand back on a read. Enough to reason about and rewrite.
MAX_READ_CHARS = 12000
# How many canvases to advertise. The ones anyone means are the recent ones.
MAX_LIST = 15

# Serializes the check-then-create in execute_create_channel_canvas. Sibling tool calls in one
# round run concurrently (tool_registry gathers them), so two create_channel_canvas calls could
# both pass the "does one exist?" check and each create a canvas — and a duplicate channel canvas
# (with its own permanent tab) can never be removed. One create at a time, workspace-wide; canvas
# creation is rare enough that a single lock costs nothing.
_channel_canvas_create_lock = asyncio.Lock()
# What Slack calls a canvas with no title — which is every channel canvas, permanently.
UNTITLED = "Untitled"

# The markdown a canvas ACTUALLY renders, probed line by line against the API rather than assumed.
# It is nearly GFM. The near-misses are what matter — and note checkboxes DO work: this file used
# to claim they didn't, and the bot was told so, which is exactly how you end up with agendas
# nobody can tick.
CANVAS_MARKDOWN_HELP = (
    "Canvas markdown: headings (`#`–`###`), **bold**, _italic_, ~~strike~~, `code`, "
    "[links](url), bullet lists, numbered lists, CHECKLISTS (`- [ ] todo` / `- [x] done`), "
    "tables (`| a | b |` with a `| --- |` row), blockquotes (`>`), code fences, `---` rules, and "
    "nested lists (indent 4 spaces).\n"
    "Two things to LEAN ON:\n"
    "- Anything that gets ticked off — a meeting agenda, action items, a launch checklist — is a "
    "CHECKLIST (`- [ ] item`), not plain bullets. That is what the boxes are for, and people tick "
    "them during the meeting.\n"
    "- Anything with repeating fields — owners, dates, statuses, options compared side by side — "
    "is a TABLE, not a run of bullets.\n"
    "Two things to AVOID:\n"
    "- Never nest one KIND of list inside another (a `- [ ]` item under a plain `-` bullet). "
    "Slack rejects the whole write outright, so the edit is lost, not degraded. Keep a nested "
    "list the same type as its parent.\n"
    "- Images are silently dropped from canvas markdown — post a picture in the thread instead."
)


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    # Logged, not just returned: the tool loop records only "-> error", so without this a
    # refusal is invisible in the logs and indistinguishable from a Slack outage.
    logger.info(f"canvas tool refused: {code} — {message}")
    return {"ok": False, "error": code, "message": message, **extra}


def _web(ctx: ToolContext):
    """The Slack WebClient underneath the platform client."""
    client = getattr(ctx, "client", None)
    app = getattr(client, "app", None)
    return getattr(app, "client", None) if app is not None else None


# Markdown scaffolding that `read_canvas` ADDS and the canvas itself does not contain.
_MD_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+)?(?:\[[ xX]\]\s+)?")
_MD_MARKS_RE = re.compile(r"[*_`]+")

# ONLY the list markers. A heading's `##` must survive into a replacement — strip it and the
# heading silently demotes to a paragraph — but a list bullet must not, because it forks the
# list. Different jobs, different regexes.
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?:\[[ xX]\]\s+)?")


def _replacement_for_section(markdown: str) -> str:
    """Strip the list bullet off a replacement so it lands IN the list, not beside it.

    A section IS the list item, so its replacement is the item's content. Hand Slack a bullet
    and it parses a whole new list: verified live, replacing item "beta" with `- [x] beta`
    DELETED it from the list and appended a fresh one-item list at the bottom of the canvas.
    The same replacement without the bullet swaps in place, keeping the item's own id.

    The model naturally passes the bullet — it is quoting the markdown `read_canvas` gave it —
    so this normalises rather than refuses.
    """
    stripped = (markdown or "").strip()
    if "\n" in stripped:            # a genuine multi-block replacement: leave it alone
        return markdown
    return _LIST_PREFIX_RE.sub("", stripped, count=1) or markdown


def _searchable(text: str) -> str:
    """Turn a quote from `read_canvas` back into something `contains_text` can find.

    The round trip is lossy in a way that bites on the very first edit. `read_canvas` renders a
    canvas as MARKDOWN — so a list item comes back as `- Launch lead — Dana`. The model, quite
    reasonably, quotes exactly that as `find_text`. But Slack's `contains_text` searches the
    canvas's TEXT, where the bullet is structure rather than content, and there is no `- ` in
    it. The lookup misses, the tool refuses, and the model has to guess what we wanted.

    Verified live: `'- Launch lead — Dana'` → section_not_found; the same string without the
    bullet → matched, first time.
    """
    out = _MD_PREFIX_RE.sub("", text or "")
    out = _MD_MARKS_RE.sub("", out)
    return " ".join(out.split())


# --- HTML -> markdown (canvas content only ever comes back as HTML) ------------------------

_BLOCK_PREFIX = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ",
                 "h5": "##### ", "h6": "###### ", "blockquote": "> "}

# A canvas list is a `<div data-section-style=N>` wrapping a `<ul>`, and N — NOT the tag, which is
# always `ul` — is what says which KIND of list it is. Probed live:
BULLET_LIST, NUMBERED_LIST, CHECK_LIST = "5", "6", "7"
_LIST_STYLES = {BULLET_LIST, NUMBERED_LIST, CHECK_LIST}


def _html_to_markdown(html: str) -> str:
    """Turn canvas HTML back into the markdown the model wrote.

    Deliberately small. Canvas HTML is a closed set, but it is a WEIRD one, and reading it wrong
    is not a cosmetic problem — the model edits what it reads. Probed live:

    * A checklist is `<div data-section-style='7'>`, and a ticked item is `<li class='checked'>`.
      An UNTICKED item carries no marker at all, so keying off the item (as this did) made an
      untouched checklist read back as a plain bullet list — the model then cannot tell a checklist
      from a bullet list, nor see what is already done. The list's STYLE is the signal.
    * Links come back as `<lnk href=…>`, not `<a>`. Reading only `a` dropped every link silently.
    * A code block comes back as `<p class="prettyprint">`, not `<pre>`.
    * Tables are real `<table>` markup whose cells hold `<p>` — so walking every `<p>` in the
      document (as this did) shredded a table into a run of loose paragraphs.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # noqa: BLE001 — degrade to raw text rather than lose the read
        # beautifulsoup4 is a hard requirement (see requirements.in); reaching here means a
        # broken install, and the model would silently edit raw HTML it misread as markdown.
        logger.error("beautifulsoup4 is not installed — canvas HTML cannot be parsed, returning "
                     "raw HTML. Reinstall dependencies (pip install --require-hashes -r "
                     "requirements.txt); canvas reads/edits are unreliable until then.")
        return html

    soup = BeautifulSoup(html or "", "html.parser")

    def inline(node) -> str:
        out = []
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                out.append(str(child))
            elif name in ("strong", "b"):
                out.append(f"**{inline(child)}**")
            elif name in ("em", "i"):
                out.append(f"_{inline(child)}_")
            elif name in ("del", "s", "strike"):
                out.append(f"~~{inline(child)}~~")
            elif name == "code":
                out.append(f"`{inline(child)}`")
            elif name in ("a", "lnk"):        # canvases emit <lnk>, not <a>
                out.append(f"[{inline(child)}]({child.get('href', '')})")
            elif name == "br":
                out.append("\n")
            else:
                out.append(inline(child))
        return " ".join("".join(out).split())

    def render_list(container, style: str, lines: List[str]) -> None:
        for i, li in enumerate(container.find_all("li"), start=1):
            text = inline(li).strip()
            if not text:
                continue
            if style == CHECK_LIST:
                # No marker means UNCHECKED — the ticked ones are the ones that say so.
                box = "[x]" if "checked" in (li.get("class") or []) else "[ ]"
                lines.append(f"- {box} {text}")
            elif style == NUMBERED_LIST:
                lines.append(f"{i}. {text}")
            else:
                lines.append(f"- {text}")
        lines.append("")

    def render_table(table, lines: List[str]) -> None:
        rows = []
        for tr in table.find_all("tr"):
            cells = [" ".join(inline(td).split()) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        lines.append("")

    lines: List[str] = []
    root = soup.find("div", class_="quip-canvas-content") or soup

    # Walk the TOP-LEVEL blocks only. Recursing would visit a table's cell paragraphs and a
    # list's items a second time, as loose lines.
    for el in root.find_all(True, recursive=False):
        name = el.name
        if name == "div":
            style = str(el.get("data-section-style") or "")
            if style in _LIST_STYLES:
                render_list(el, style, lines)
            else:                                    # an unknown wrapper: render what's inside
                for sub in el.find_all(["ul", "ol"]):
                    render_list(sub, BULLET_LIST, lines)
        elif name in ("ul", "ol"):
            render_list(el, NUMBERED_LIST if name == "ol" else BULLET_LIST, lines)
        elif name == "table":
            render_table(el, lines)
        elif name == "hr":
            lines.extend(["---", ""])
        elif name == "pre":
            lines.extend(["```", el.get_text(), "```", ""])
        elif name == "p" and "prettyprint" in (el.get("class") or []):
            lines.extend(["```", inline(el).strip(), "```", ""])
        elif name == "blockquote":
            for p in el.find_all("p") or [el]:
                text = inline(p).strip()
                if text:
                    lines.append(f"> {text}")
            lines.append("")
        else:
            text = inline(el).strip()
            if text:
                lines.append(_BLOCK_PREFIX.get(name, "") + text)
                lines.append("")

    out = "\n".join(lines).strip()
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out


# What a real canvas body has and a Slack login page does not. `download_file` normally rejects
# an HTML body outright, because for an image HTML means the auth failed and Slack served a
# login screen instead of a 401. Canvases have to opt out of that guard — their content IS
# html — so they take on the job of telling a canvas apart from a login page themselves.
_CANVAS_MARKER = "quip-canvas-content"


async def _download_canvas_markdown(client, web, canvas_id: str) -> Optional[str]:
    """Download a canvas and convert it back to markdown. There is no read API."""
    if web is None or client is None:
        return None
    try:
        info = await _async(web.files_info, file=canvas_id)
        url = (info.get("file") or {}).get("url_private")
        if not url:
            return None
        raw = await client.download_file(url, canvas_id, allow_html=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Canvas read failed for {canvas_id}: {e}")
        return None
    if not raw:
        logger.warning(f"Canvas {canvas_id} came back empty")
        return None

    html = raw.decode("utf-8", errors="replace")
    if _CANVAS_MARKER not in html:
        # We asked for HTML, so we no longer get the guard that would have caught a login page.
        logger.warning(f"Canvas {canvas_id} did not return canvas content (auth problem?)")
        return None
    try:
        return _html_to_markdown(html)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Canvas HTML conversion failed for {canvas_id}: {e}")
        return None


async def _fetch_canvas_markdown(ctx: ToolContext, canvas_id: str) -> Optional[str]:
    return await _download_canvas_markdown(ctx.client, _web(ctx), canvas_id)


async def _async(fn, **kwargs):
    """slack_sdk's WebClient here is the AsyncWebClient under Bolt's async app."""
    result = fn(**kwargs)
    if hasattr(result, "__await__"):
        result = await result
    return result


async def _canvas_file(ctx: ToolContext, canvas_id: str) -> Optional[Dict[str, Any]]:
    """The canvas's file record — but ONLY if the canvas is really in THIS channel.

    The authorization check, and it is deliberately a LIVE one. A canvas id is just a file id:
    one from another channel is syntactically perfect and would edit happily. Since every canvas
    we create is shared into its channel on creation, "shared here" is exactly the right test —
    and unlike a snapshot taken at the start of the turn, it also covers a canvas the model
    created moments ago.

    It hands back the whole record rather than a yes/no because the caller wants the `permalink`
    out of it anyway, and this is the same `files.info` call that would have to fetch it.
    """
    web = _web(ctx)
    if web is None:
        return None
    try:
        info = await _async(web.files_info, file=canvas_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Canvas access check failed for {canvas_id}: {e}")
        return None
    f = info.get("file") or {}
    if f.get("filetype") not in ("quip", "canvas"):
        return None
    # Slack files report their channels in THREE places and the right one depends on the
    # channel's type: `channels` for public, `groups` for private, `ims` for DMs. Checking only
    # `channels` reads a private channel as "not shared here" and refuses every edit — verified
    # live, in a private channel, where `channels` came back empty and `groups` held the id.
    for bucket in ("channels", "groups", "ims"):
        if ctx.channel_id in (f.get(bucket) or []):
            return f
    shares = (f.get("shares") or {})
    for bucket in ("public", "private"):
        if ctx.channel_id in (shares.get(bucket) or {}):
            return f
    return None


async def _shared_into_channel(ctx: ToolContext, canvas_id: str) -> bool:
    return await _canvas_file(ctx, canvas_id) is not None


async def _permalink(web, canvas_id: str) -> Optional[str]:
    """The canvas's URL. It comes from `files.info` and is never CONSTRUCTED — the workspace
    domain and team id are both in it (`https://<workspace>.slack.com/docs/<team>/<file_id>`), so
    building one from the id would be a guess, and a guessed link is a broken link.

    Best-effort by design: the edit has already landed by the time anyone wants a link, so a
    failure here costs the confirmation its link, not the user their work.
    """
    if web is None:
        return None
    try:
        info = await _async(web.files_info, file=canvas_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not get a permalink for {canvas_id}: {e}")
        return None
    return (info.get("file") or {}).get("permalink") or None


def _link_hint(url: Optional[str], title: Optional[str]) -> str:
    """Tell the model to put the canvas's link in its reply — and hand it the exact markdown.

    The URL is in the result's `url` field either way, but a model that has to assemble a link
    from a field is a model that can mistype one, and a broken link to a document is worse than
    no link at all. So the finished markdown goes in the instruction, ready to copy.
    """
    if not url:
        return ""
    label = (title or "").strip() or "the canvas"
    return f" Link it so they can jump straight there: [{label}]({url})"


# --- the channel's canvas catalog ----------------------------------------------------------
#
# Without this the model does not know a single canvas EXISTS. Slack posts no message when a
# canvas is shared, so canvases never appear in the rebuilt history either — the only signal
# was the word "canvas" in a tool description. So "update our devops call agenda to discuss
# failed deploys" had nothing to connect "agenda" to: no keyword, no ids, nothing. It would have
# had to guess that a canvas might exist and call list_canvases on a hunch.
#
# Same fix as `mount_file` and `edit_image`: put the ids in front of it as a literal enum, with
# titles beside them, so the ask matches a thing it can already see. `list_canvases` survives as
# a way to refresh, but the model should rarely need it.
#
# Cost: unlike the file catalogs (a DB read), this is a Slack API call. So it is cached per
# channel with a short TTL — canvas lists change rarely, and our own create/edit invalidates.
CATALOG_KEY = "_canvas_catalog"
_CATALOG_TTL = 300.0
_catalog_cache: Dict[str, Any] = {}


def _invalidate_catalog(channel_id: Optional[str]) -> None:
    _catalog_cache.pop(channel_id or "", None)


async def _first_heading(client, web, canvas_id: str) -> Optional[str]:
    """The canvas's own top heading — the only name a channel canvas has."""
    md = await _download_canvas_markdown(client, web, canvas_id)
    if not md:
        return None
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("#"):
            name = line.lstrip("#").strip()
            if name:
                return name[:80]
        elif line:
            break        # real content before any heading: it hasn't got one
    return None


async def _file_is_live(web, file_id: str, *, strict: bool = False) -> bool:
    """Is this canvas still a real file? `files.info` answers precisely — a deleted one comes back
    `file_deleted` — which `files.list` cannot, because it is EVENTUALLY CONSISTENT in both
    directions: it keeps a deleted canvas for a while, and it does not yet know about one created
    seconds ago."""
    try:
        info = await _async(web.files_info, file=file_id)
    except Exception as e:  # noqa: BLE001
        err = str(getattr(getattr(e, "response", None), "get", lambda _: "")("error") or e)
        if "file_deleted" in err or "file_not_found" in err:
            return False
        if strict:
            raise                     # a Slack outage is not evidence the canvas is gone
        logger.warning(f"Liveness check failed for {file_id}: {e}")
        return False
    return bool((info.get("file") or {}).get("id"))


async def _channel_canvas_id(web, channel_id: str, live_ids: Set[str],
                             *, strict: bool = False) -> Optional[str]:
    """The id of THE channel canvas — the one Slack pins as a tab — or None if there isn't one.

    Two things had to be probed live, because neither is in the docs:

    1. `properties.canvas` is NULL even on a channel that demonstrably has a channel canvas. The
       real record is `properties.tabs`, where the canvas appears as `{"type": "canvas",
       "data": {"file_id": …}}`. So the tab IS the channel canvas.
    2. **A tab outlives its canvas** — delete the canvas and the tab lingers — so a tab alone is
       not proof. But `files.list` is eventually consistent in BOTH directions, so it is not proof
       either: a canvas created seconds ago is missing from it. Taking absence as death was a real
       bug — right after the bot made the agenda, the catalog decided the tab was stale, dropped
       the canvas, and offered `create_channel_canvas` again while `edit_canvas` had no id to aim
       at. So `files.list` is only the fast path; anything it doesn't list is settled by
       `files.info`, which answers `file_deleted` precisely.

    `strict` is for callers where "I could not tell" must NOT read as "there isn't one". Building
    a catalog can shrug off a failed lookup; deciding whether the thing about to be irreversibly
    deleted is the channel's own document cannot — swallowing the error there turns a Slack
    outage into a licence to delete. (A test caught exactly that.)
    """
    try:
        info = await _async(web.conversations_info, channel=channel_id)
    except Exception as e:  # noqa: BLE001 — no tab info just means "no channel canvas"
        if strict:
            raise
        logger.warning(f"Channel canvas lookup failed for {channel_id}: {e}")
        return None
    props = (info.get("channel") or {}).get("properties") or {}
    candidates = []
    canvas_prop = props.get("canvas") or {}
    if canvas_prop.get("file_id"):
        candidates.append(canvas_prop["file_id"])
    for tab in props.get("tabs") or []:
        if tab.get("type") == "canvas":
            fid = (tab.get("data") or {}).get("file_id")
            if fid:
                candidates.append(fid)
    for fid in candidates:
        if fid in live_ids:
            return fid
        if await _file_is_live(web, fid, strict=strict):
            return fid                # brand new: files.list simply hasn't caught up yet
    return None


async def build_catalog(client, channel_id: str, *, now: Optional[float] = None
                        ) -> List[Dict[str, Any]]:
    """The canvases in this channel, id + title, with the channel canvas marked. Never raises —
    no catalog just means the model has to look them up, which is where we started."""
    if not client or not channel_id or not getattr(config, "enable_canvas_tools", True):
        return []

    stamp = now if now is not None else time.monotonic()
    cached = _catalog_cache.get(channel_id)
    if cached and (stamp - cached["at"]) < _CATALOG_TTL:
        return cached["entries"]

    app = getattr(client, "app", None)
    web = getattr(app, "client", None) if app is not None else None
    if web is None:
        logger.warning(f"Canvas catalog: no Slack web client on {type(client).__name__}")
        return []
    try:
        res = await _async(web.files_list, channel=channel_id, types="canvases", limit=MAX_LIST)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Canvas catalog lookup failed for {channel_id}: {e}")
        return []

    entries = [{"canvas_id": f["id"], "title": (f.get("title") or "").strip() or UNTITLED,
                "is_channel_canvas": False}
               for f in (res.get("files") or []) if f.get("id")]

    ch_id = await _channel_canvas_id(web, channel_id, {e["canvas_id"] for e in entries})
    if ch_id and ch_id not in {e["canvas_id"] for e in entries}:
        # files.list hasn't caught up with a canvas we just made. Leaving it out would hide the
        # channel's own document from read_canvas and edit_canvas for the next few minutes —
        # exactly when the turn that created it wants to keep working on it.
        title = UNTITLED
        try:
            info = await _async(web.files_info, file=ch_id)
            title = ((info.get("file") or {}).get("title") or "").strip() or UNTITLED
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not title the fresh channel canvas {ch_id}: {e}")
        entries.insert(0, {"canvas_id": ch_id, "title": title, "is_channel_canvas": False})

    for e in entries:
        if e["canvas_id"] == ch_id:
            e["is_channel_canvas"] = True
            if e["title"] == UNTITLED:
                # We always create with a title, so this is the salvage path: a canvas made
                # before we passed one, or by something else. It can never be renamed, and a
                # document called "Untitled" is one no ask can match — so fall back to its top
                # heading. One small fetch, at most one canvas, behind the same 5-minute cache.
                heading = await _first_heading(client, web, e["canvas_id"])
                if heading:
                    e["title"] = heading

    _catalog_cache[channel_id] = {"at": stamp, "entries": entries}
    logger.info(f"Canvas catalog for {channel_id}: {[e['title'] for e in entries]} "
                f"(channel canvas: {ch_id or 'none'})")
    return entries


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """One line per canvas. The channel canvas is called out by ROLE rather than by name, because
    it does not have a usable one: `conversations.canvases.create` takes no title (see
    `execute_create_channel_canvas`), so Slack reports it as "Untitled" forever. Its identity to
    a reader is "the channel's document", which is exactly what we say."""
    out = []
    for e in entries:
        title = e.get("title") or UNTITLED
        if e.get("is_channel_canvas"):
            role = ("the channel canvas — this channel's pinned document"
                    if title == UNTITLED else
                    f"{title} (the channel canvas — this channel's pinned document)")
            out.append(f"{e['canvas_id']} — {role}")
        else:
            out.append(f"{e['canvas_id']} — {title}")
    return "\n".join(out)


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["canvas_id"] for e in (entries or []) if e.get("canvas_id")]


def channel_canvas_id(entries: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    for e in entries or []:
        if e.get("is_channel_canvas") and e.get("canvas_id"):
            return e["canvas_id"]
    return None


def _catalog(thread_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return (thread_config or {}).get(CATALOG_KEY) or []


# --- schemas ------------------------------------------------------------------------------

def get_create_channel_canvas_schema(thread_config: Optional[Dict[str, Any]] = None
                                     ) -> Optional[Dict[str, Any]]:
    """Offered ONLY when the channel has no canvas yet.

    "Create if not exists" is a rule the model would otherwise have to remember and would
    eventually forget — and forgetting is expensive here, because `conversations.canvases.create`
    is NOT idempotent: called twice it cheerfully makes a SECOND channel canvas with a SECOND tab
    (verified live). So the rule is enforced by the schema instead: once a channel canvas exists
    this tool disappears, and `edit_canvas` — which already has the id in its enum — is the only
    way to write to it.
    """
    if channel_canvas_id(_catalog(thread_config)):
        return None
    return {
        "type": "function",
        "name": "create_channel_canvas",
        "description": (
            "Create this channel's canvas — a living document pinned as a TAB at the top of the "
            "channel, editable later by you or by anyone else.\n\n"
            "Reach for it when the thing you are producing will be RETURNED TO: a running spec, a "
            "standing agenda, a checklist, an onboarding guide, a plan that will change. A chat "
            "message is buried within the hour and a posted file forks into `_final_v3`; the "
            "channel canvas stays put, stays editable, and is the one document everybody can "
            "find.\n\n"
            "There is exactly ONE canvas per channel and this channel has none yet, so what you "
            "write is its starting content. From then on you extend it with edit_canvas rather "
            "than making another.\n\n"
            "Do NOT use it as a fancy way to answer a question — if the reply is just an answer, "
            "write the answer. And do not use it for generated data files (a chart, a workbook, a "
            "deck): those are files, and the code sandbox already delivers them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "The canvas title — it labels the tab at the top of the channel. It can "
                        "NEVER be changed afterwards (Slack has no rename), so make it say what "
                        "the document is: 'DevOps Call Agenda', not 'Notes'."
                    ),
                },
                "markdown": {
                    "type": "string",
                    "description": (
                        "The starting content. Do NOT open it with the title again as a heading "
                        "— Slack already renders the title above the content, so a restated one "
                        "just says the document's name twice. Start with the content itself.\n\n"
                        + CANVAS_MARKDOWN_HELP
                    ),
                },
            },
            "required": ["title", "markdown"],
            "additionalProperties": False,
        },
    }


def get_read_canvas_schema(thread_config: Optional[Dict[str, Any]] = None
                          ) -> Optional[Dict[str, Any]]:
    entries = _catalog(thread_config)
    ids = valid_ids(entries)
    if not ids:
        return None          # nothing to read: hide the tool rather than invite a guess
    return {
        "type": "function",
        "name": "read_canvas",
        "description": (
            "Read a canvas in this channel, as markdown. Use it before editing one — you cannot "
            "sensibly revise a document you have not read, and it may have been changed by "
            "someone else since you last saw it.\n\n"
            "Canvases in this channel:\n" + catalog_lines(entries)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "canvas_id": {"type": "string", "enum": ids},
            },
            "required": ["canvas_id"],
            "additionalProperties": False,
        },
    }


def get_list_canvases_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_canvases",
        "description": (
            "List the canvases in this channel, with their ids and titles. Use it to find the "
            "canvas the user means before reading or editing one."
        ),
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    }


def get_edit_canvas_schema(thread_config: Optional[Dict[str, Any]] = None
                          ) -> Optional[Dict[str, Any]]:
    entries = _catalog(thread_config)
    ids = valid_ids(entries)
    if not ids:
        return None          # nothing to edit
    return {
        "type": "function",
        "name": "edit_canvas",
        "description": (
            "Edit a canvas in this channel.\n\n"
            "DEFAULT TO ADDING. A canvas is a record and its old entries are the point of it, so "
            "new material is an insert — never a rewrite of what is there. Reach for a replace_* "
            "or delete_section only when the ask is to CHANGE or REMOVE something specific that "
            "already exists (tick a box, correct a figure, drop a line). A recurring meeting is a "
            "rolling log: `prepend` the new date's section so the newest is on top and every "
            "previous meeting survives below it, untouched.\n\n"
            "ADDING TO AN EXISTING LIST IS `replace_list`, NOT AN INSERT. This is the one that "
            "bites: an insert can only place a WHOLE BLOCK (a heading, a paragraph, a new list) "
            "— it can never put an item inside a list that already exists. Insert `- [ ] new item` "
            "next to a list and Slack builds a SECOND, separate one-item list beside it. So to add "
            "items to a list that is already there, use `replace_list` and pass the whole list back "
            "with the new items included.\n\n"
            "- `append` / `prepend` — add a block at the very end or the very start.\n"
            "- `insert_after` / `insert_before` — add a block next to the block `find_text` "
            "matches. This is how you put something in the MIDDLE: to add material under an "
            "existing heading, insert_after that heading. Never do this to grow a list (see "
            "above).\n"
            "- `replace_section` — rewrite the ONE BLOCK `find_text` matches: a heading, a "
            "paragraph, or a single list item. A section is not a region: to change three lines, "
            "make three calls.\n"
            "- `replace_list` — rewrite a WHOLE list at once: `find_text` is any line of it, "
            "`markdown` is the complete new list. This is how you ADD items to a list, REMOVE "
            "items from one, REORDER it, and **the only way to tick a checkbox** (a box cannot be "
            "ticked item-by-item — replacing `- [x] beta` on its own would tear that item out of "
            "the list). Pass every item back with its box as it should END UP (`- [x]` for done, "
            "`- [ ]` for not), leaving the items you are not changing exactly as they are.\n"
            "- `delete_section` — remove the ONE BLOCK `find_text` matches. Use it to take out a "
            "line the user asked you to drop, or to clean up something you yourself put in the "
            "wrong place. It needs no markdown. Never delete anything else.\n\n"
            "READ THE CANVAS FIRST. `find_text` must be text you have actually seen in it. If it "
            "matches more than one block you will be told how many; pass `occurrence` to say which "
            "one you mean, counting from the top.\n\n"
            + CANVAS_MARKDOWN_HELP +
            "\n\nCanvases in this channel:\n" + catalog_lines(entries)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "canvas_id": {"type": "string", "enum": ids},
                "operation": {"type": "string",
                              "enum": ["append", "prepend", "insert_after", "insert_before",
                                       "replace_section", "replace_list", "delete_section"]},
                "markdown": {
                    "type": "string",
                    "description": ("The new content. For replace_list, the COMPLETE new list — "
                                    "every item, not just the changed ones. Not needed for "
                                    "delete_section."),
                },
                "find_text": {
                    "type": "string",
                    "description": ("The block to act on: what to rewrite (replace_section), "
                                    "remove (delete_section), insert next to (insert_after / "
                                    "insert_before), or any line of the list to rewrite "
                                    "(replace_list). Required for all of those."),
                },
                "occurrence": {
                    "type": "integer",
                    "description": ("Which match to act on when find_text appears more than once, "
                                    "counting from the top of the canvas (1 = the first). Only "
                                    "needed when the tool tells you the text is ambiguous — that "
                                    "is how you remove a duplicated line."),
                },
            },
            "required": ["canvas_id", "operation"],
            "additionalProperties": False,
        },
    }


# --- executors ----------------------------------------------------------------------------

def _drop_repeated_title(title: str, markdown: str) -> str:
    """Slack renders the canvas TITLE as a heading above the content, and the model — reasonably —
    also opens the document with `# <the same title>`. The reader then sees the name twice. Only a
    leading heading that repeats the title is dropped; a different one is the document's own first
    section and is left alone."""
    lines = (markdown or "").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        heading = line.lstrip("#").strip()
        if line.lstrip().startswith("#") and _norm(heading) == _norm(title):
            return "\n".join(lines[i + 1:]).lstrip("\n")
        return markdown
    return markdown


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


async def execute_create_channel_canvas(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Create the channel canvas — the one canvas Slack pins as a tab.

    Why this and not `canvases.create`: a standalone canvas cannot be pinned. Probed live —
        pins.add(timestamp=<the share ts>)  -> message_not_found  (a canvas share isn't a message)
        pins.add(file=<canvas id>)          -> no_item_specified  (Slack dropped file pinning)
        bookmarks.add(link=<permalink>)     -> ok, but comes back type="file", is NOT returned by
                                               bookmarks.list, and bookmarks.remove refuses it
                                               with invalid_bookmark_type — a write-only bookmark
    ...and it still produces no tab. `conversations.canvases.create` is the ONLY call that yields
    a real `{"type": "canvas"}` entry in the channel's tab bar, which is the thing that makes a
    canvas findable at all: Slack posts no message when a canvas is shared, so an untabbed canvas
    appears nowhere in the channel — not in history, not in search-by-eye.

    `title` is UNDOCUMENTED and is not in the slack_sdk signature — which takes `**kwargs`, so it
    reaches the API anyway. It works, and it is the only chance to set one: the tab is labelled
    with it, and afterwards there is no way to rename a canvas at all (`files.rename`,
    `canvases.setTitle`, `conversations.canvases.setTitle` are each `unknown_method`; `files.edit`
    is `not_allowed_token_type` for a bot). Omit it and the canvas is "Untitled" forever, which is
    a document no ask can ever match — so the title is required, not optional.
    """
    title = (args.get("title") or "").strip()
    markdown = (args.get("markdown") or "").strip()
    if not title or not markdown:
        return _err("missing_content", "A canvas needs a title and some markdown.")
    if len(markdown) > MAX_MARKDOWN_CHARS:
        return _err("too_long",
                    f"That is {len(markdown)} characters; the limit is {MAX_MARKDOWN_CHARS}.")

    web = _web(ctx)
    if web is None or not ctx.channel_id:
        return _err("unavailable", "Canvases aren't available right now.")

    # Last line of defence against a second channel canvas: the schema hides this tool once one
    # exists, but the schema is built from a catalog that can be up to _CATALOG_TTL stale, and
    # conversations.canvases.create is NOT idempotent — a second call means a second tab, forever.
    # The whole check-then-create is serialized so two sibling calls gathered in one round cannot
    # both pass the existence check and each create a canvas.
    async with _channel_canvas_create_lock:
        try:
            listed = await _async(web.files_list, channel=ctx.channel_id, types="canvases",
                                  limit=MAX_LIST)
            live = {f["id"] for f in (listed.get("files") or []) if f.get("id")}
            existing = await _channel_canvas_id(web, ctx.channel_id, live)
        except Exception as e:  # noqa: BLE001 — fail CLOSED: a duplicate canvas is unrecoverable
            logger.error(f"Could not verify whether a channel canvas already exists: {e}",
                         exc_info=True)
            return _err("check_failed",
                        "I couldn't check whether this channel already has a canvas, so I didn't "
                        "create one — a duplicate channel canvas can't be undone. Try again in a "
                        "moment.")
        if existing:
            return _err("already_exists",
                        "This channel already has a canvas. Edit it with edit_canvas instead of "
                        "creating another — a channel is meant to have exactly one.",
                        canvas_id=existing)

        try:
            created = await _async(
                web.conversations_canvases_create, channel_id=ctx.channel_id, title=title,
                document_content={"type": "markdown",
                                  "markdown": _drop_repeated_title(title, markdown)})
            canvas_id = created.get("canvas_id")
        except Exception as e:  # noqa: BLE001
            logger.error(f"conversations.canvases.create failed: {e}", exc_info=True)
            return _err("create_failed", f"Slack refused to create the canvas: {e}")

        if not canvas_id:
            return _err("create_failed", "Slack created no canvas id.")

        # No canvases.access.set here, unlike a standalone canvas: the channel canvas belongs to
        # the channel, so Slack shares it in on creation (files.info shows the channel under
        # `shares` with source CHANNEL_TAB) and everyone who can see the channel can already see
        # it. Invalidate inside the lock so the next check (ours or a sibling's) SEES this canvas.
        _invalidate_catalog(ctx.channel_id)

    # F46: a canvas was really written — deliverable work that does not call claim_work. Force
    # a top-level channel reply into a thread at final-post time (resolve_reply_target).
    _turn = getattr(ctx, "turn", None)
    if _turn is not None:
        _turn.mark_substantive_work()

    url = await _permalink(web, canvas_id)
    logger.info(f"Created the channel canvas {canvas_id} ({title!r}) in {ctx.channel_id}")
    return {"ok": True, "canvas_id": canvas_id, "title": title, "is_channel_canvas": True,
            "url": url,
            "message": ("The channel canvas is created, and Slack has pinned it as a tab at the "
                        "top of the channel. Tell the user it exists and what is in it; don't "
                        "paste its contents back." + _link_hint(url, title))}


async def execute_list_canvases(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    web = _web(ctx)
    if web is None or not ctx.channel_id:
        return _err("unavailable", "Canvases aren't available right now.")
    try:
        res = await _async(web.files_list, channel=ctx.channel_id, types="canvases",
                           limit=MAX_LIST)
    except Exception as e:  # noqa: BLE001
        return _err("list_failed", f"Could not list canvases: {e}")

    canvases = [{"canvas_id": f.get("id"), "title": f.get("title") or "(untitled)",
                 "updated": f.get("updated")}
                for f in (res.get("files") or []) if f.get("id")]
    return {"ok": True, "canvases": canvases, "count": len(canvases)}


async def execute_read_canvas(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    canvas_id = (args.get("canvas_id") or "").strip()
    if not canvas_id:
        return _err("missing_canvas_id", "A canvas_id is required.")
    canvas = await _canvas_file(ctx, canvas_id)
    if canvas is None:
        return _err("not_in_this_channel",
                    f"{canvas_id} is not a canvas in this channel.")

    markdown = await _fetch_canvas_markdown(ctx, canvas_id)
    if markdown is None:
        return _err("read_failed", "Could not read that canvas.")

    truncated = len(markdown) > MAX_READ_CHARS
    # `url` rides along unasked-for: a read is often the prelude to talking ABOUT the canvas, and
    # the link is the one thing the markdown itself cannot tell the model.
    return {"ok": True, "canvas_id": canvas_id,
            "url": canvas.get("permalink"),
            "markdown": markdown[:MAX_READ_CHARS],
            "truncated": truncated}


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+(?:\[[ xX]\]\s*)?|\d+[.)]\s+)(.*\S)\s*$")
_CHECKBOX_RE = re.compile(r"^\s*[-*+]\s*\[[ xX]\]")
MAX_LIST_ITEMS = 40          # a rewrite costs one API call per item; a runaway is a bug, not a doc


def _list_block_around(markdown: str, needle: str) -> Optional[List[str]]:
    """The contiguous run of list items containing `needle`, as their bare texts, in order."""
    items: List[Optional[str]] = []
    blocks: List[List[str]] = []
    for line in (markdown or "").splitlines():
        m = _LIST_ITEM_RE.match(line)
        if m:
            items.append(m.group(1))
        else:
            if items:
                blocks.append([i for i in items if i])
                items = []
    if items:
        blocks.append([i for i in items if i])

    for block in blocks:
        if any(needle in _searchable(item) for item in block):
            return block
    return None


async def _rewrite_list(ctx: ToolContext, canvas_id: str, find_text: str, markdown: str,
                        *, url: Optional[str] = None,
                        title: Optional[str] = None) -> Dict[str, Any]:
    """Rewrite a whole list in place — the ONLY way to tick a checkbox.

    A checkbox cannot be toggled by editing its own section. Probed live, every route:

      replace(item, "- [x] beta")      -> the item LEAVES the list and a new one-item list is
                                          appended after it. beta jumps out of order, silently.
      insert_after(item, "- [x] beta") -> same: a new list block, not a sibling item.
      replace(item, "beta")            -> replaces in place, but the tick state is untouched —
                                          so a "mark it done" quietly does nothing at all.

    Any markdown carrying list syntax becomes a NEW list block; only bare text lands in place.
    So the unit of a tick is the LIST, not the item:

      1. replace the FIRST item's section with the entire new list -> the new list is planted
         directly after the old container;
      2. delete every remaining old item -> the old container empties and disappears.

    The rebuilt list ends up exactly where the old one was — verified with content on both sides
    of it. (`canvases.edit` takes ONE change per call, so this is 1 + N calls, not a batch.)
    """
    web = _web(ctx)
    current = await _fetch_canvas_markdown(ctx, canvas_id)
    if current is None:
        return _err("read_failed", "Could not read the canvas to rewrite its list.")

    block = _list_block_around(current, _searchable(find_text))
    if not block:
        return _err("list_not_found",
                    f"No list in that canvas contains {_searchable(find_text)!r}. Read the canvas "
                    f"and quote a line you can actually see in the list you mean.")
    if len(block) > MAX_LIST_ITEMS:
        return _err("list_too_long",
                    f"That list has {len(block)} items; {MAX_LIST_ITEMS} is the most this can "
                    f"rewrite safely.")

    # Resolve every existing item to its section BEFORE changing anything: a half-rewritten list
    # is worse than a refused one, and an ambiguous item would delete the wrong line.
    section_ids: List[str] = []
    for item in block:
        needle = _searchable(item)
        if not needle or not re.search(r"\w", needle):
            continue
        try:
            found = await _async(web.canvases_sections_lookup, canvas_id=canvas_id,
                                 criteria={"contains_text": needle})
            sections = found.get("sections") or []
        except Exception as e:  # noqa: BLE001
            return _err("lookup_failed", f"Could not search the canvas: {e}")
        if len(sections) != 1:
            # This is where the bot used to get STUCK. Once it had accidentally duplicated a line,
            # every rewrite of that list was ambiguous, and it had no way to remove the duplicate
            # either — so it just kept apologising. Name the escape hatch.
            return _err(
                "ambiguous_list_item",
                f"{needle!r} appears {len(sections)} times in this canvas, so the list cannot be "
                f"rewritten safely — a rewrite would hit the wrong copy. If those are accidental "
                f"duplicates, remove the unwanted one first with operation='delete_section', "
                f"find_text={needle!r} and the occurrence (1–{len(sections)}, counting from the "
                f"top of the canvas) of the copy that should go; then rewrite the list.")
        section_ids.append(sections[0]["id"])

    if not section_ids:
        return _err("list_not_found", "That list has no addressable items.")

    try:
        await _async(web.canvases_edit, canvas_id=canvas_id,
                     changes=[{"operation": "replace", "section_id": section_ids[0],
                               "document_content": {"type": "markdown", "markdown": markdown}}])
    except Exception as e:  # noqa: BLE001
        logger.error(f"canvases.edit (list rewrite) failed for {canvas_id}: {e}", exc_info=True)
        return _err("edit_failed", f"Slack refused the edit: {e}")

    # From here the new list EXISTS. Every leftover we fail to delete is a duplicate line the user
    # can see, so this is loud — but it is not a failure of the edit, which already landed.
    stranded = 0
    for sid in section_ids[1:]:
        try:
            await _async(web.canvases_edit, canvas_id=canvas_id,
                         changes=[{"operation": "delete", "section_id": sid}])
        except Exception as e:  # noqa: BLE001
            stranded += 1
            logger.error(f"Canvas {canvas_id}: could not delete superseded list item {sid}: {e}")

    _invalidate_catalog(ctx.channel_id)
    logger.info(f"Rewrote a {len(section_ids)}-item list in canvas {canvas_id} "
                f"(stranded={stranded})")
    if stranded:
        return {"ok": True, "canvas_id": canvas_id, "operation": "replace_list",
                "stranded_items": stranded, "url": url,
                "message": (f"The list was rewritten, but {stranded} old item(s) could not be "
                            f"removed and are still in the canvas. Say so plainly."
                            + _link_hint(url, title))}
    return {"ok": True, "canvas_id": canvas_id, "operation": "replace_list", "url": url,
            "message": ("The list is rewritten. Say what changed; don't paste the whole canvas."
                        + _link_hint(url, title))}


async def _resolve_section(ctx: ToolContext, canvas_id: str, find_text: str,
                          occurrence: Optional[int]):
    """`find_text` -> exactly ONE section id, or a refusal. Returns (section_id, error).

    Ambiguity used to be a dead end: "matches 2 passages, be more specific" is impossible advice
    when the two passages are IDENTICAL — which is exactly the state the bot lands in after it
    duplicates a line. It could not rewrite the list (ambiguous) and could not delete the
    duplicate (also ambiguous), so the mess was permanent. `occurrence` is the way out, and it is
    well-defined because `canvases.sections.lookup` returns matches in DOCUMENT ORDER (probed:
    deleting sections[0] removes the topmost one). The model can read the canvas, count, and say
    which. Still no guessing — with several matches and no occurrence, this refuses.
    """
    needle = _searchable(find_text)
    # A needle with no word in it ("-", "**") is not a search — it is punctuation that would
    # match half the document and overwrite whichever half Slack returned first.
    if not needle or not re.search(r"\w", needle):
        return None, _err("missing_find_text",
                          "find_text must contain some actual text from the canvas.")
    try:
        found = await _async(_web(ctx).canvases_sections_lookup, canvas_id=canvas_id,
                             criteria={"contains_text": needle})
        sections = found.get("sections") or []
    except Exception as e:  # noqa: BLE001
        return None, _err("lookup_failed", f"Could not search the canvas: {e}")

    if not sections:
        return None, _err("section_not_found",
                          f"No passage in that canvas contains {needle!r}. Read the canvas and "
                          f"quote text you can actually see in it.")
    if occurrence is not None:
        if not 1 <= occurrence <= len(sections):
            return None, _err("bad_occurrence",
                              f"occurrence={occurrence}, but {len(sections)} block(s) contain "
                              f"{needle!r}. They are numbered 1–{len(sections)} from the top.")
        return sections[occurrence - 1]["id"], None
    if len(sections) > 1:
        # Ambiguity is the caller's to resolve. Picking one would be a coin flip on which of the
        # user's paragraphs gets overwritten — but now there IS a way to resolve it.
        return None, _err("ambiguous_find_text",
                          f"{needle!r} matches {len(sections)} blocks in the canvas, numbered "
                          f"1–{len(sections)} from the top. Either use text that appears in only "
                          f"one of them, or pass occurrence to say which you mean (if these are "
                          f"accidental duplicates, delete_section with the occurrence of the "
                          f"unwanted one is how you clean that up).",
                          matches=len(sections))
    return sections[0]["id"], None


def _anchor_is_list_item(markdown: str, needle: str) -> bool:
    """Is the block `needle` names a LIST ITEM (rather than a heading or paragraph)?"""
    return any(_LIST_ITEM_RE.match(line) and needle in _searchable(line)
               for line in (markdown or "").splitlines())


async def execute_edit_canvas(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    canvas_id = (args.get("canvas_id") or "").strip()
    operation = (args.get("operation") or "").strip()
    markdown = (args.get("markdown") or "").strip()
    find_text = (args.get("find_text") or "").strip()
    occurrence = args.get("occurrence")
    if isinstance(occurrence, str) and occurrence.strip().isdigit():
        occurrence = int(occurrence)
    if not isinstance(occurrence, int):
        occurrence = None

    if not canvas_id:
        return _err("missing_content", "A canvas_id is required.")
    # Every operation but one WRITES something, and demanding markdown for the one that doesn't
    # was its own bug: with no delete operation to reach for, the model tried to delete a line by
    # replacing it with nothing, got "markdown is required", and had no move left. (Seen live,
    # three times in one thread, ending in "I can't safely remove them from here".)
    if operation != "delete_section" and not markdown:
        return _err("missing_content", f"{operation or 'That operation'} needs markdown to write.")
    if len(markdown) > MAX_MARKDOWN_CHARS:
        return _err("too_long",
                    f"That is {len(markdown)} characters; the limit is {MAX_MARKDOWN_CHARS}.")
    canvas = await _canvas_file(ctx, canvas_id)
    if canvas is None:
        # Not a guess about what they meant: a file id from another channel is syntactically
        # perfect, and rewriting the wrong document is not something a user sees coming.
        return _err("not_in_this_channel", f"{canvas_id} is not a canvas in this channel.")
    url = canvas.get("permalink")
    title = (canvas.get("title") or "").strip()

    web = _web(ctx)
    content = {"type": "markdown", "markdown": markdown}

    # A canvas SECTION is one block — a heading, a paragraph, a single list item — not a region.
    # So a find_text spanning several lines can never match anything, and the model reaches for
    # exactly that: it reads the canvas, sees a Steps list, and quotes the whole list to "replace
    # the Steps section". Say what the unit actually is, rather than refusing with a not-found it
    # cannot learn from.
    if find_text and "\n" in find_text.strip():
        return _err(
            "find_text_spans_multiple_lines",
            "A canvas section is ONE block — a heading, a paragraph, or a single list item — so "
            "find_text must name one line, not a region. To rewrite a whole list, use "
            "operation='replace_list' with any single line of it as find_text; to change several "
            "unrelated lines, call edit_canvas once per line.")

    if operation == "append":
        changes = [{"operation": "insert_at_end", "document_content": content}]
    elif operation == "prepend":
        changes = [{"operation": "insert_at_start", "document_content": content}]
    elif operation == "replace_list":
        if not find_text:
            return _err("missing_find_text",
                        "replace_list needs find_text — any line of the list you mean.")
        return await _rewrite_list(ctx, canvas_id, find_text, markdown, url=url, title=title)
    elif operation in ("insert_after", "insert_before", "replace_section", "delete_section"):
        if not find_text:
            return _err("missing_find_text",
                        f"{operation} needs find_text — the block to act on.")

        if operation in ("insert_after", "insert_before") and _LIST_ITEM_RE.match(markdown):
            # An insert cannot put an item INSIDE a list. Probed: insert_after a list item lands
            # the content after the WHOLE list, and list markdown becomes a SECOND, separate list
            # sitting beside the first. That is precisely the bug this refusal exists to stop —
            # the bot tried to add agenda items next to a checklist and produced a stray one-item
            # list below it, which it then could not clean up. Only refuse when the anchor really
            # is a list item, though: putting a NEW block after a list is legitimate.
            current = await _fetch_canvas_markdown(ctx, canvas_id)
            if current and _anchor_is_list_item(current, _searchable(find_text)):
                return _err(
                    "use_replace_list",
                    "You cannot insert an item into a list that already exists — Slack would "
                    "build a second, separate list beside it. To add items to that list, use "
                    "operation='replace_list' with any line of it as find_text and the COMPLETE "
                    "new list (old items plus new, each box as it should end up) as markdown.")

        # Ticking a box is NOT an in-place edit. A replacement carrying `- [x]` makes Slack build
        # a NEW list: the item leaves its place in the document and reappears at the bottom. And
        # stripping the box (which `_replacement_for_section` does, correctly, for a plain bullet)
        # would land the text in place with its tick UNCHANGED — a "mark it done" that silently
        # does nothing. Neither is acceptable, so refuse and name the operation that works.
        if operation == "replace_section" and _CHECKBOX_RE.match(markdown):
            return _err(
                "use_replace_list",
                "A checkbox cannot be ticked in place — editing one item's section either moves "
                "it to the bottom of the canvas or leaves the tick untouched. Use "
                "operation='replace_list' and pass the WHOLE list, with each item's box as it "
                "should end up.")

        section_id, error = await _resolve_section(ctx, canvas_id, find_text, occurrence)
        if error:
            return error
        if operation == "delete_section":
            changes = [{"operation": "delete", "section_id": section_id}]
        elif operation == "replace_section":
            changes = [{"operation": "replace", "section_id": section_id,
                        "document_content": {"type": "markdown",
                                             "markdown": _replacement_for_section(markdown)}}]
        else:
            changes = [{"operation": operation, "section_id": section_id,
                        "document_content": content}]
    else:
        return _err("bad_operation", f"Unknown operation {operation!r}.")

    try:
        await _async(web.canvases_edit, canvas_id=canvas_id, changes=changes)
    except Exception as e:  # noqa: BLE001
        logger.error(f"canvases.edit failed for {canvas_id}: {e}", exc_info=True)
        return _err("edit_failed", f"Slack refused the edit: {e}")

    # F46: a canvas edit is deliverable work (no claim_work here) — thread a top-level reply.
    _turn = getattr(ctx, "turn", None)
    if _turn is not None:
        _turn.mark_substantive_work()

    _invalidate_catalog(ctx.channel_id)
    logger.info(f"Edited canvas {canvas_id} ({operation})")
    return {"ok": True, "canvas_id": canvas_id, "operation": operation, "url": url,
            "message": ("The canvas is updated. Say what changed; don't paste the whole canvas."
                        + _link_hint(url, title))}


def get_delete_canvas_schema(thread_config: Optional[Dict[str, Any]] = None
                            ) -> Optional[Dict[str, Any]]:
    entries = _catalog(thread_config)
    # The channel canvas is NOT deletable and is left out of the enum entirely. It is the
    # channel's own document and its tab is part of the furniture — "clear the agenda" means
    # rewrite it, never destroy it — and deleting it would strand the tab (Slack keeps showing
    # one for a dead canvas). Emptying it is an edit; there is no ask that needs this.
    ids = [e["canvas_id"] for e in entries
           if e.get("canvas_id") and not e.get("is_channel_canvas")]
    if not ids:
        return None
    listable = [e for e in entries if not e.get("is_channel_canvas")]
    return {
        "type": "function",
        "name": "delete_canvas",
        "description": (
            "Delete a canvas from this channel. IRREVERSIBLE — the document and its history are "
            "gone, for everyone, and it may be someone else's work.\n\n"
            "Only ever do this when a person has just asked you to delete THIS specific canvas. "
            "Never tidy up on your own initiative, never delete something merely because it "
            "looks stale or obsolete, and if there is any doubt about which canvas they meant, "
            "ask instead of guessing.\n\n"
            "The channel canvas cannot be deleted and is not listed here — to clear it out, "
            "rewrite it with edit_canvas.\n\n"
            "Canvases in this channel:\n" + catalog_lines(listable)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "canvas_id": {"type": "string", "enum": ids},
            },
            "required": ["canvas_id"],
            "additionalProperties": False,
        },
    }


async def execute_delete_canvas(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    canvas_id = (args.get("canvas_id") or "").strip()
    if not canvas_id:
        return _err("missing_canvas_id", "A canvas_id is required.")
    # Defense-in-depth (mirrors participation_tools): `_delete_enabled` already withholds the tool
    # unless a HUMAN directly addressed the bot, but this is the one irreversible canvas op, so
    # re-check the raw sender classification off ctx.message and refuse a NON-human author outright.
    # A bot-authored @mention (dispatched to the main handler un-gated) must never reach an
    # irreversible delete even if the tool were somehow offered. Absent classification → rely on the
    # schema gate (never fail closed on paths that omit it).
    msg = getattr(ctx, "message", None)
    sender_type = (getattr(msg, "metadata", None) or {}).get("sender_type") if msg is not None else None
    if sender_type is not None and sender_type != "human":
        return _err("not_human_sender", "A canvas can only be deleted at a person's request.")
    if not await _shared_into_channel(ctx, canvas_id):
        return _err("not_in_this_channel", f"{canvas_id} is not a canvas in this channel.")

    web = _web(ctx)
    if web is None:
        return _err("unavailable", "Canvases aren't available right now.")

    # Re-checked at execution, not just in the schema: the enum is built from a catalog that can
    # be stale, and an id is not authorization. Slack would happily delete the channel canvas.
    try:
        listed = await _async(web.files_list, channel=ctx.channel_id, types="canvases",
                              limit=MAX_LIST)
        live = {f["id"] for f in (listed.get("files") or []) if f.get("id")}
        is_channel_canvas = canvas_id == await _channel_canvas_id(
            web, ctx.channel_id, live, strict=True)
    except Exception as e:  # noqa: BLE001 — a failed check must not become a licence to delete
        logger.warning(f"Could not confirm {canvas_id} is not the channel canvas: {e}")
        return _err("check_failed",
                    "Could not confirm that isn't the channel canvas, so it was left alone.")
    if is_channel_canvas:
        return _err("is_channel_canvas",
                    "That is the channel canvas — the channel's own pinned document — and it "
                    "cannot be deleted. If it needs clearing out, rewrite it with edit_canvas.")

    try:
        await _async(web.canvases_delete, canvas_id=canvas_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"canvases.delete failed for {canvas_id}: {e}", exc_info=True)
        return _err("delete_failed", f"Slack refused to delete it: {e}")

    _invalidate_catalog(ctx.channel_id)
    # WARNING, not info: this is the one canvas operation nobody can undo, and if it ever fires
    # when it shouldn't have, the log is the only record that it happened at all.
    logger.warning(f"DELETED canvas {canvas_id} from {ctx.channel_id} "
                   f"(requested by {ctx.user_id})")
    return {"ok": True, "canvas_id": canvas_id, "deleted": True,
            "message": "The canvas is gone. Confirm it briefly; there is nothing to link to."}


def _enabled(_cfg: dict) -> bool:
    return bool(getattr(config, "enable_canvas_tools", True))


def _delete_enabled(cfg: dict) -> bool:
    """Deletion is offered ONLY when a PERSON directly addressed the bot in this message.

    The danger was never the user asking — "delete that canvas" is an ordinary request and they
    own their channel. The danger is the bot deciding to tidy up: on a turn nobody addressed it
    in, it is reading along and inferring what would be helpful, and a tool that irreversibly
    destroys a shared document has no business being on the table for that.

    Keyed off `_canvas_delete_authorized` (handlers.text `_materialize_request_tools`): a HUMAN
    sender AND a genuine current-message address — a real <@bot> mention or a DM. A bare name-hit
    does NOT qualify (the `participation_name_hit` regex also fires on a message that merely QUOTES
    the bot's name), and a NON-human other_bot @mention does not either. Absent → fail CLOSED: a
    destructive tool must never default to available on a config that skipped the derivation.
    """
    if not getattr(config, "enable_canvas_tools", True):
        return False
    if not getattr(config, "enable_canvas_delete", True):
        return False
    return bool(cfg.get("_canvas_delete_authorized", False))


def register_canvas_tools(registry: ToolRegistry) -> None:
    """create_channel_canvas / list / read / edit / delete.

    Every one of these is a FACTORY, and that is what encodes the lifecycle. The channel's
    canvases ride IN the schemas as an enum, so the model knows they exist without anyone saying
    the word "canvas" — and the set of tools on offer states the truth about the channel:

      no canvas yet   -> only create_channel_canvas   (nothing to read, edit or delete)
      canvas exists   -> read / edit / delete, and create DISAPPEARS, so a second channel
                         canvas (and a second tab, forever) is not a mistake that can be made

    Delete is the odd one out twice over: it excludes the channel canvas from its enum, and it is
    withheld unless a PERSON directly addressed the bot in this message (see `_delete_enabled`), so
    the model can honour a genuine "delete that canvas" but can never decide to tidy up a channel it
    was only listening in.
    """
    registry.register(get_create_channel_canvas_schema, execute_create_channel_canvas,
                      enabled=_enabled, name="create_channel_canvas")
    registry.register(get_list_canvases_schema(), execute_list_canvases, enabled=_enabled)
    registry.register(get_read_canvas_schema, execute_read_canvas,
                      enabled=_enabled, name="read_canvas")
    registry.register(get_edit_canvas_schema, execute_edit_canvas,
                      enabled=_enabled, name="edit_canvas")
    registry.register(get_delete_canvas_schema, execute_delete_canvas,
                      enabled=_delete_enabled, name="delete_canvas")
