"""Canvas HTML — the one representation Slack will hand back, and how to read it.

There is no `canvases.read`. Content comes back by downloading `url_private`, and it arrives as
HTML rather than the markdown that went in, so anything that wants to READ a canvas needs the
converter below. Two callers now want it — the canvas TOOLS (`message_processor/canvas_tools.py`,
round-tripping a canvas the bot manages) and the DOCUMENT path (`document_handler.parse_canvas`,
reading a canvas someone dropped in a thread) — which is why it lives here rather than in either
of them.

The HTML facts encoded here were probed live against the API; the notes on `html_to_markdown`
say which ones and why they matter.
"""
from __future__ import annotations

from typing import List

from logger import setup_logger

logger = setup_logger(name="slack_bot.CanvasContent")

# Slack's mimetype for a canvas file. A canvas IS a file, so it arrives on messages like any
# other attachment — this is the only thing that identifies it as a canvas.
CANVAS_MIMETYPE = "application/vnd.slack-docs"

# What a real canvas body has and a Slack login page does not. `download_file` normally rejects
# an HTML body outright, because for an image HTML means the auth failed and Slack served a
# login screen instead of a 401. Canvases have to opt out of that guard — their content IS
# html — so they take on the job of telling a canvas apart from a login page themselves.
CANVAS_MARKER = "quip-canvas-content"

_BLOCK_PREFIX = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ",
                 "h5": "##### ", "h6": "###### ", "blockquote": "> "}

# A canvas list is a `<div data-section-style=N>` wrapping a `<ul>`, and N — NOT the tag, which is
# always `ul` — is what says which KIND of list it is. Probed live:
BULLET_LIST, NUMBERED_LIST, CHECK_LIST = "5", "6", "7"
_LIST_STYLES = {BULLET_LIST, NUMBERED_LIST, CHECK_LIST}


def html_to_markdown(html: str) -> str:
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
