"""The thread's mountable-file catalog (F35).

Until now the only bytes that could ever reach the code-interpreter sandbox were images the
bot itself generated (``create_image_asset``). Everything a *user* shared could be SEEN —
images ride the turn as ``input_image`` parts, documents are text-extracted into the prompt —
but never *used*: "build a PDF from these four screenshots" was structurally impossible, and
"analyse this 50k-row CSV" forced the model to retype the data into its own source code.

This catalog is the id space that fixes it. It unions the two stores the thread already keeps
(the image table and the document table) behind one opaque id, so ``mount_file`` is a single
tool rather than a pair the model has to choose between by guessing a file's type.

The authorization rule is ``image_catalog``'s, for the same reason: the ids are advertised to
the model as a literal ``enum`` and re-validated against the turn's snapshot before any bytes
move. A syntactically valid id is not authorization. A file from another thread cannot resolve,
and neither can one the model invented.
"""
from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from logger import setup_logger

logger = setup_logger(name="slack_bot.ThreadFiles")

# How many files the model may choose among. Newest-first, so the ones anyone refers to are
# the ones present; a thread with 300 attachments must not blow up the tool schema.
MAX_CATALOG = 20

# Longest blurb we put next to an id — enough to tell two screenshots apart, not a wall.
_DESC_CHARS = 100

_IMAGE_FALLBACK_MIME = "image/png"


def image_file_id(row_id: Any) -> str:
    return f"file_img_{row_id}"


def document_file_id(row_id: Any) -> str:
    return f"file_doc_{row_id}"


def _filename_from_url(url: str, default: str = "image.png") -> str:
    """Best-effort filename for an image row, which stores a URL and no name."""
    try:
        name = os.path.basename(unquote(urlparse(url).path))
    except Exception:  # noqa: BLE001 — a malformed URL costs a nicer name, nothing else
        name = ""
    return name or default


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "no description available"
    return text[:_DESC_CHARS] + ("…" if len(text) > _DESC_CHARS else "")


def _human_size(size: Any) -> str:
    try:
        n = int(size or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _doc_origin(row: Dict[str, Any]) -> str:
    """"uploaded" (the user shared it) vs "generated" (we built it earlier in this thread).

    ``metadata`` may arrive as a dict or as the raw JSON text the column stores, depending on
    which accessor loaded the row — and it is only a label either way, so a parse failure just
    means we call the file uploaded.
    """
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            import json
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            meta = None
    if isinstance(meta, dict) and meta.get("source") == "generated":
        return "generated"
    return "uploaded"


async def build_catalog(db, thread_key: str) -> List[Dict[str, Any]]:
    """Every file in this thread the sandbox could mount, newest first, capped.

    Never raises: no catalog simply means ``mount_file`` is not offered this turn.
    """
    if not db or not thread_key:
        return []

    entries: List[Dict[str, Any]] = []

    try:
        images = await db.find_thread_images_async(thread_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Image lookup failed for {thread_key}: {e}")
        images = []

    for row in images or []:
        row_id, url = row.get("id"), row.get("url")
        if row_id is None or not url:
            continue
        origin = row.get("image_type") or "image"
        entries.append({
            "file_id": image_file_id(row_id),
            "kind": "image",
            "origin": origin,
            "filename": _filename_from_url(url),
            "mime_type": mimetypes.guess_type(url)[0] or _IMAGE_FALLBACK_MIME,
            "size_bytes": None,
            "url": url,
            "slack_file_id": None,
            "description": _clip(row.get("analysis") or row.get("prompt") or ""),
            "created_at": row.get("created_at"),
        })

    try:
        docs = await db.get_thread_documents_async(thread_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Document lookup failed for {thread_key}: {e}")
        docs = []

    for row in docs or []:
        row_id = row.get("id")
        url = row.get("url_private")
        slack_id = row.get("file_id")
        # A row with neither ref predates on-demand access — its bytes are unreachable, so
        # offering it would only produce a mount that fails.
        if row_id is None or (not url and not slack_id):
            continue
        entries.append({
            "file_id": document_file_id(row_id),
            "kind": "document",
            "origin": _doc_origin(row),
            "filename": row.get("filename") or "document",
            "mime_type": row.get("mime_type") or "application/octet-stream",
            "size_bytes": row.get("size_bytes"),
            "url": url,
            "slack_file_id": slack_id,
            "description": _clip(row.get("summary") or ""),
            "created_at": row.get("created_at"),
        })

    # Both stores return oldest-first. The model reasons about "the ones I just sent".
    entries.sort(key=lambda e: (e.get("created_at") or ""), reverse=True)

    # One Slack file, one catalog entry. The same upload can be written twice — the unattended
    # catalog records it when we stay quiet, and a turn records it again if it later processes
    # the same message — and `save_document` is a plain INSERT, so both rows survive. Offering
    # the model two ids for one file wastes an enum slot and invites it to mount the thing
    # twice. Newest row wins, since it carries whatever richer metadata arrived later.
    deduped: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in entries:
        key = entry.get("slack_file_id") or entry.get("url")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped[:MAX_CATALOG]


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """The human-readable half of the enum — what each id actually is."""
    lines = []
    for entry in entries:
        bits = [entry["filename"], entry["mime_type"]]
        size = _human_size(entry.get("size_bytes"))
        if size:
            bits.append(size)
        lines.append(f"{entry['file_id']} — {', '.join(bits)}: {entry['description']}")
    return "\n".join(lines)


async def catalog_unattended(processor, client, message) -> None:
    """Record the files on a message the bot decided NOT to answer.

    Cataloguing used to be a side effect of replying, so a file shared in a message we stayed
    quiet about — or one superseded while the participation gate was debouncing — was gone for
    good: no document row, no image row, and therefore invisible to `read_document`, to
    `mount_file`, and to the model. Slack still had it. We just could not see it.

    This is metadata only: the Slack ref and enough to name the file. No bytes are stored (the
    no-content-at-rest rule), and no extraction or visual description is done — those are the
    expensive parts and they belong to a turn we are actually running. The point is simply that
    the file remains REACHABLE, so a later "use the CSV I posted earlier" can still find it.

    Never raises: failing to catalog costs a file, not the bot.
    """
    db = getattr(processor, "db", None)
    attachments = list(getattr(message, "attachments", None) or [])
    if db is None or not attachments:
        return

    thread_key = f"{message.channel_id}:{message.thread_id}"
    try:
        known_docs = {d.get("file_id") for d in (await db.get_thread_documents_async(thread_key)) or []}
        known_images = {i.get("url") for i in (await db.find_thread_images_async(thread_key)) or []}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read the existing catalog for {thread_key}: {e}")
        known_docs, known_images = set(), set()

    message_ts = (getattr(message, "metadata", None) or {}).get("ts")
    for att in attachments:
        url = att.get("url") or att.get("url_private")
        file_id = att.get("file_id") or att.get("id")
        name = att.get("filename") or att.get("name") or "file"
        mime = att.get("mimetype") or att.get("mime_type") or ""
        if not url:
            continue
        try:
            if att.get("type") == "image" or mime.startswith("image/"):
                if url in known_images:
                    continue
                await db.save_image_metadata_async(
                    thread_id=thread_key, url=url, image_type="uploaded",
                    prompt="", analysis="",
                    metadata={"source": "unattended", "filename": name},
                    message_ts=message_ts)
            else:
                if file_id and file_id in known_docs:
                    continue
                db.save_document(
                    thread_id=thread_key, filename=name,
                    mime_type=mime or "application/octet-stream",
                    summary=f"Shared in this conversation ({name}). Not yet read.",
                    file_id=file_id, url_private=url,
                    size_bytes=att.get("size"),
                    metadata={"source": "uploaded", "cataloged": "unattended"},
                    message_ts=message_ts)
            logger.info(f"Catalogued unattended file {name} for {thread_key}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not catalog {name} for {thread_key}: {e}")


def resolve(entries: Optional[List[Dict[str, Any]]], file_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an id against THIS TURN's snapshot. Only ids we put in front of the model
    resolve — a valid-looking id is not permission to read the bytes behind it."""
    for entry in entries or []:
        if entry.get("file_id") == file_id:
            return entry
    return None


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["file_id"] for e in (entries or []) if e.get("file_id")]
