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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import unquote, urlparse

from message_processor.canvas_content import CANVAS_MIMETYPE
from message_processor.image_catalog import DM_LOOKBACK_HOURS
from database import UNATTENDED_SUMMARY_TEMPLATE
from logger import setup_logger

logger = setup_logger(name="slack_bot.ThreadFiles")

# How many files the model may choose among. Newest-first, so the ones anyone refers to are
# the ones present; a thread with 300 attachments must not blow up the tool schema.
MAX_CATALOG = 20

# Longest blurb we put next to an id — enough to tell two screenshots apart, not a wall.
_DESC_CHARS = 100

_IMAGE_FALLBACK_MIME = "image/png"

# The marker a widened entry carries, and the words the model sees next to its id. Kept as its
# own key rather than folded into `origin`, which already means something else here ("uploaded"
# vs "generated") and is read by callers that do not care where a file came from.
_DM_SCOPE = "earlier in this DM"


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


def _image_entry(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One image row as a catalog entry, or None when its bytes are unreachable.

    One builder for both scopes, so a widened entry is the same shape as a strict one by
    construction rather than by two loops agreeing.
    """
    row_id, url = row.get("id"), row.get("url")
    if row_id is None or not url:
        return None
    return {
        "file_id": image_file_id(row_id),
        "kind": "image",
        "origin": row.get("image_type") or "image",
        "filename": _filename_from_url(url),
        "mime_type": mimetypes.guess_type(url)[0] or _IMAGE_FALLBACK_MIME,
        "size_bytes": None,
        "url": url,
        "slack_file_id": None,
        "description": _clip(row.get("analysis") or row.get("prompt") or ""),
        "created_at": row.get("created_at"),
    }


def _document_entry(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One document row as a catalog entry, or None when it cannot be mounted."""
    row_id = row.get("id")
    url = row.get("url_private")
    slack_id = row.get("file_id")
    # A row with neither ref predates on-demand access — its bytes are unreachable, so
    # offering it would only produce a mount that fails.
    if row_id is None or (not url and not slack_id):
        return None
    # A canvas's bytes are HTML, which is not a useful build input — the model reads a
    # canvas through history or read_document instead.
    if row.get("mime_type") == CANVAS_MIMETYPE:
        return None
    return {
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
    }


def _dedup_key(entry: Dict[str, Any]) -> Any:
    return entry.get("slack_file_id") or entry.get("url")


async def build_catalog(db, thread_key: str) -> List[Dict[str, Any]]:
    """Every file in this thread the sandbox could mount, newest first, capped.

    Never raises: no catalog simply means ``mount_file`` is not offered this turn.

    In a DM the scope is widened to the rest of the conversation, exactly as
    ``image_catalog.build_catalog`` widens the edit catalog and for the same reason: Slack makes
    every top-level DM message its own thread root, so four images generated a minute ago sit
    under a different key than "now build me a deck from those". Strictly keyed, ``mount_file``
    had nothing to offer and the sandbox saw an empty /mnt/data.
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
        entry = _image_entry(row)
        if entry:
            entries.append(entry)

    try:
        docs = await db.get_thread_documents_async(thread_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Document lookup failed for {thread_key}: {e}")
        docs = []

    for row in docs or []:
        entry = _document_entry(row)
        if entry:
            entries.append(entry)

    # Both stores return oldest-first. The model reasons about "the ones I just sent".
    entries.sort(key=lambda e: (e.get("created_at") or ""), reverse=True)

    # One Slack file, one catalog entry. The same upload can be written twice — the unattended
    # catalog records it when we stay quiet, and a turn records it again if it later processes
    # the same message — and `save_document` is a plain INSERT, so both rows survive. Offering
    # the model two ids for one file wastes an enum slot and invites it to mount the thing
    # twice. Newest row wins, since it carries whatever richer metadata arrived later.
    deduped: List[Dict[str, Any]] = []
    seen: Set[Any] = set()
    seen_ids: Set[str] = set()
    for entry in entries:
        key = _dedup_key(entry)
        if key in seen:
            continue
        seen.add(key)
        seen_ids.add(entry["file_id"])
        deduped.append(entry)

    # Strict entries are already deduped and already hold their places, so the widening only
    # fills what is left of the cap: this thread's own files always win.
    deduped.extend(await _dm_widening(db, thread_key, seen, seen_ids,
                                      room=MAX_CATALOG - len(deduped)))
    return deduped[:MAX_CATALOG]


async def _dm_widening(db, thread_key: str, seen: Set[Any], seen_ids: Set[str], *,
                       room: int) -> List[Dict[str, Any]]:
    """Recent files from elsewhere in the SAME DM, newest first.

    DMs only, and one DM only: both lookups are a prefix match on this channel id, the same
    boundary ``read_document``'s channel-wide fallback and ``image_catalog``'s widening already
    use. It never reaches another channel, another DM, or another person.

    Channels are deliberately left strict. There a thread IS a real conversation boundary, and
    offering files from other threads would be a genuine scope change rather than a repair.

    Never raises: any failure below costs a narrower catalog, never the turn.
    """
    # Thread keys contain colons on BOTH sides (channel:thread_ts) — split once, from the left.
    channel_id = thread_key.split(":", 1)[0]
    if room <= 0 or not channel_id.startswith("D"):
        return []

    # Both stores are read BEFORE anything competes for the room. Filling from images first and
    # asking documents for the remainder meant a DM holding 20 recent pictures could never offer
    # the CSV uploaded one message ago — the room was gone before the document query ran. So the
    # two candidate lists are merged and ranked by time, and recency decides.
    candidates = await _widen_images(db, channel_id) + await _widen_documents(db, channel_id)
    candidates.sort(key=lambda e: _parse_stamp(e.get("created_at")) or _OLDEST, reverse=True)

    widened = _collect(candidates, seen, seen_ids, room)
    if widened:
        logger.debug(f"DM file catalog widened by {len(widened)} for {channel_id}")
    return widened


async def _widen_images(db, channel_id: str) -> List[Dict[str, Any]]:
    """Every image candidate from elsewhere in this DM. Time-bounded in SQL."""
    if not hasattr(db, "find_channel_images_async"):
        return []
    try:
        rows = await db.find_channel_images_async(
            channel_id, within_hours=DM_LOOKBACK_HOURS, limit=MAX_CATALOG * 2)
    except Exception as e:  # noqa: BLE001 — a failed widening is just a narrower catalog
        logger.debug(f"DM image widening failed for {channel_id}: {e}")
        return []
    return [e for e in (_image_entry(r) for r in rows or []) if e is not None]


async def _widen_documents(db, channel_id: str) -> List[Dict[str, Any]]:
    """Every document candidate from elsewhere in this DM.

    Unlike the image lookup, ``get_channel_documents_async`` takes no time bound, so the same
    lookback window is applied here instead of in SQL.
    """
    if not hasattr(db, "get_channel_documents_async"):
        return []
    try:
        rows = await db.get_channel_documents_async(channel_id)
    except Exception as e:  # noqa: BLE001 — a failed widening is just a narrower catalog
        logger.debug(f"DM document widening failed for {channel_id}: {e}")
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DM_LOOKBACK_HOURS)
    recent = (r for r in (rows or []) if _within(r.get("created_at"), cutoff))
    return [e for e in (_document_entry(r) for r in recent) if e is not None]


def _collect(candidates: List[Dict[str, Any]], seen: Set[Any], seen_ids: Set[str],
             room: int) -> List[Dict[str, Any]]:
    """Mark and keep candidates until the room runs out, skipping anything the strict catalog
    already has and anything already taken here. Both dedup keys matter: the same Slack file can
    be a duplicate row (same url/file_id) and the same DB row can be reached by both the strict
    and the widened query (same catalog id)."""
    widened: List[Dict[str, Any]] = []
    for entry in candidates:
        key = _dedup_key(entry)
        if key in seen or entry["file_id"] in seen_ids:
            continue
        entry["scope"] = _DM_SCOPE
        seen.add(key)
        seen_ids.add(entry["file_id"])
        widened.append(entry)
        if len(widened) >= room:
            break
    return widened


# Where a row with no usable timestamp sorts: last. A file we cannot date is not one to rank
# ahead of a file we can.
_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _parse_stamp(created_at: Any) -> Optional[datetime]:
    """A row's ``created_at`` as an aware datetime, or None when it cannot be read."""
    if isinstance(created_at, datetime):
        stamp = created_at
    else:
        text = str(created_at or "").strip()
        if not text:
            return None
        # SQLite writes "2026-09-09 05:44:00"; other writers use the ISO "T".
        text = text.replace(" ", "T", 1)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            return None
    if stamp.tzinfo is None:
        # CURRENT_TIMESTAMP is UTC, and so is the image lookup's datetime('now').
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _within(created_at: Any, cutoff: datetime) -> bool:
    """Is this row inside the DM lookback window?

    Unparseable or missing timestamps are OUT, which is what the SQL comparison the image
    lookup uses does with them too — a row we cannot date is not a row we can call recent.
    """
    stamp = _parse_stamp(created_at)
    return stamp is not None and stamp >= cutoff


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """The human-readable half of the enum — what each id actually is."""
    lines = []
    for entry in entries:
        bits = [entry["filename"], entry["mime_type"]]
        size = _human_size(entry.get("size_bytes"))
        if size:
            bits.append(size)
        # Say so when a file came from another message in this DM rather than this exchange —
        # "the deck I just sent" and "that CSV from earlier" are different requests.
        marker = f" [{entry['scope']}]" if entry.get("scope") else ""
        lines.append(f"{entry['file_id']}{marker} — {', '.join(bits)}: {entry['description']}")
    return "\n".join(lines)


EVIDENCE_HEADER = "Files in this thread (mount_file ids):"


def catalog_evidence_lines(entries: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """The file section of a channel turn's tool-evidence block, one entry per line.

    Same text the mount_file schema used to carry, moved out of the cached prefix.
    """
    entries = entries or []
    if not entries:
        return [EVIDENCE_HEADER, "(none)"]
    return [EVIDENCE_HEADER] + catalog_lines(entries).split("\n")


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
                    summary=UNATTENDED_SUMMARY_TEMPLATE.format(name=name),
                    file_id=file_id, url_private=url,
                    size_bytes=att.get("size"),
                    metadata={"source": "uploaded", "cataloged": "unattended"},
                    message_ts=message_ts)
            logger.info(f"Catalogued unattended file {name} for {thread_key}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not catalog {name} for {thread_key}: {e}")


def resolve(entries: Optional[List[Dict[str, Any]]], file_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an id against THIS TURN's snapshot. Only ids we put in front of the model
    resolve — a valid-looking id is not permission to read the bytes behind it.

    Slack's own `F…` id for the same entry resolves too. On the channel surface `mount_file`
    is the static schema, which cannot carry a per-thread enum, and the `F…` id is sitting
    right there in the message the file arrived on — so the first mount of a channel turn was
    reliably spent being refused and retried. This is not a widening: the alias matches the
    SAME entries this turn already offered, so an id from another thread still resolves to
    nothing. Catalog handles are matched first, so a shadowed id keeps its own entry.
    """
    for entry in entries or []:
        if entry.get("file_id") == file_id:
            return entry
    for entry in entries or []:
        if entry.get("slack_file_id") and entry.get("slack_file_id") == file_id:
            return entry
    return None


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["file_id"] for e in (entries or []) if e.get("file_id")]
