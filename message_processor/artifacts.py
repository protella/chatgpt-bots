"""F32: publish code-interpreter artifacts into the thread.

The model runs Python in an OpenAI-hosted container (`code_interpreter`). Whatever files it
writes there, we upload into the Slack thread:

    list container -> download bytes -> validate -> dedupe -> files_upload_v2 -> persist ref

**The container LISTING is the only publication source.** Two things follow from that, both
verified live against the API:

* We do not use `container_file_citation` annotations. They only appear when the model writes
  a `sandbox:/mnt/data/...` markdown link — which we forbid, because such links are dead in
  Slack (the file rides as an upload instead). With that instruction the model cites nothing.
  A display-only chart the model never saved still shows up in the listing (as
  `/mnt/data/cfile_<id>.png`), so the listing is a strict SUPERSET of what citations would
  give us. Trusting a citation would also let the model hand back a file it does not own —
  see the source filter below — so citations buy nothing and cost safety.
* Files are filtered to `source == "assistant"`. A user's own attachment mounts into the very
  same container as `source == "user"`; without this filter we would upload someone's
  spreadsheet straight back into the channel. The filter fails CLOSED: an absent or unknown
  source is skipped.

Nothing in here may raise into the turn: the text answer has already posted by the time
artifacts publish, so a failure here must degrade to "fewer files", never to a broken answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import config
from logger import setup_logger
from message_processor.containers import publication_lock, release_publication_lock

# Must go through setup_logger: handlers are attached to `slack_bot.*` loggers with
# propagate=False, so a bare getLogger(__name__) writes to NOWHERE. Every rejection notice here
# ("artifact refused", "content does not match .xlsx", "cap reached") would be invisible — and
# silent artifact failures are exactly what made the v1 bug so expensive to find.
logger = setup_logger(name="slack_bot.Artifacts")

# The container writes under /mnt/data and the model may link it with a `sandbox:` URI. Those
# links are dead to the user, so they're stripped from the reply text.
_SANDBOX_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*sandbox:[^)]*\)")
_SANDBOX_BARE_RE = re.compile(r"\(?sandbox:/\S+?\)?(?=[\s.,;)]|$)")

# OpenAI names a display-only image after its own file id. Readable names beat internal ids.
_CFILE_NAME_RE = re.compile(r"^cfile_[0-9a-f]{6,}", re.IGNORECASE)

_MAGIC: Dict[str, List[bytes]] = {
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "xlsx": [b"PK\x03\x04"],
    "docx": [b"PK\x03\x04"],
    "pptx": [b"PK\x03\x04"],
}
_OOXML_EXTS = {"xlsx", "docx", "pptx"}
_TEXT_EXTS = {"csv", "tsv", "json", "txt", "md"}
_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}

# The main-part content type an OOXML package MUST declare for the extension the model claimed.
# Checking only that `[Content_Types].xml` exists proves nothing — any zip can carry a file by
# that name. These are the strings that actually make a package a workbook/document/deck.
_OOXML_MAIN_PART = {
    "xlsx": "spreadsheetml.sheet.main+xml",
    "docx": "wordprocessingml.document.main+xml",
    "pptx": "presentationml.presentation.main+xml",
}
# Active content. We will not hand a colleague a macro-bearing file, whatever it is named.
_OOXML_FORBIDDEN_PARTS = ("vbaproject.bin",)
_OOXML_FORBIDDEN_TYPES = ("macroenabled", "ms-office.vbaproject", "ms-excel.sheet.binary")

_EXT_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "webp": "image/webp", "pdf": "application/pdf", "csv": "text/csv",
    "tsv": "text/tab-separated-values", "json": "application/json", "txt": "text/plain",
    "md": "text/markdown",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_API_TIMEOUT = 30.0

# Container file ids already published by this process — the fast path for the case that is now
# the NORM rather than the exception: containers are thread-scoped and reused across turns, so a
# turn-2 listing still contains turn-1's chart.
#
# This cache alone is NOT sufficient. It dies with the process, and a bot restart mid-conversation
# leaves the container alive (up to 20 min) with its files intact — turn 2 would then re-upload
# everything from turn 1. The durable record lives in `thread_containers.published_files_json`
# and is merged in below; this stays as the in-process fast path.
_PUBLISHED_MEMORY = 512
_published_file_ids: "OrderedDict[str, None]" = OrderedDict()


def _remember_published(file_id: str) -> None:
    _published_file_ids[file_id] = None
    while len(_published_file_ids) > _PUBLISHED_MEMORY:
        _published_file_ids.popitem(last=False)


@dataclass
class ArtifactRef:
    """One file the model wrote in the container."""
    container_id: str
    file_id: str
    filename: str
    # The listing reports this for some files and null for others (verified live), so it is a
    # cheap early rejection, never a guarantee. The streaming download is what actually bounds
    # memory.
    size_bytes: Optional[int] = None


def collect_container_ids(sink: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Unique code-interpreter container ids seen this turn, in encounter order.

    Normally there is exactly one — the thread's persistent container, used by every round of
    the tool loop. It stays a LIST because that is not guaranteed: a turn that falls back to
    `auto` gets a FRESH container per API call (verified live), and the tool loop makes one call
    per round. Keeping only the last id would silently drop an earlier round's chart.
    """
    seen: List[str] = []
    for entry in (sink or []):
        try:
            cid = entry.get("container_id")
            if cid and cid not in seen:
                seen.append(cid)
        except Exception:  # noqa: BLE001
            continue
    return seen


def strip_sandbox_links(text: str) -> str:
    """Remove `sandbox:/mnt/data/...` links — they 404 for the user.

    `[Download chart.png](sandbox:/mnt/data/chart.png)` -> `chart.png`, so the sentence still
    reads and the real file arrives as an upload. Cheap and unconditional: called even when no
    artifacts were captured, so a stray link can never reach anyone.
    """
    if not text or "sandbox:" not in text:
        return text
    out = _SANDBOX_LINK_RE.sub(lambda m: m.group(1) or "", text)
    out = _SANDBOX_BARE_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sanitize_filename(name: str, fallback_index: int = 1) -> Optional[str]:
    """Basename-only, control-stripped, length-capped, extension-allowlisted.

    Returns None when the file must not be published. The MODEL names these files, so the name
    is untrusted input: no paths, no traversal, no control characters.
    """
    raw = (name or "").strip()
    raw = raw.replace("\\", "/").split("/")[-1]
    raw = "".join(ch for ch in raw if ch.isprintable()).strip()
    raw = raw.lstrip(".")
    if not raw or "." not in raw:
        return None

    stem, _, ext = raw.rpartition(".")
    ext = ext.lower()
    if ext not in config.artifact_allowed_extensions:
        return None

    if _CFILE_NAME_RE.match(stem):
        stem = f"output_{fallback_index}"

    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", stem).strip(" .-_")
    stem = re.sub(r"\s+", " ", stem)
    if not stem:
        stem = f"output_{fallback_index}"
    return f"{stem[:80]}.{ext}"


def _ooxml_ok(data: bytes, ext: str) -> bool:
    """A real OOXML package OF THE CLAIMED TYPE, and not a macro carrier.

    ZIP magic alone is worthless — any archive renamed `.xlsx` passes it. So is the mere
    presence of `[Content_Types].xml`: a zip can contain a file with that name and nothing else.
    We parse it and require the main-part content type that actually defines the format, so a
    `.docx` cannot arrive wearing an `.xlsx` name (Slack would render a preview for a file Word
    can't open), and a renamed archive of arbitrary junk cannot pass at all.
    """
    expected = _OOXML_MAIN_PART.get(ext)
    if not expected:
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            lowered = [n.lower() for n in names]
            if "[Content_Types].xml" not in names:
                return False
            if any(n.endswith(bad) for n in lowered for bad in _OOXML_FORBIDDEN_PARTS):
                return False
            declared = zf.read("[Content_Types].xml").decode("utf-8", "replace").lower()
    except Exception:  # noqa: BLE001 — a corrupt archive is simply not publishable
        return False

    if any(bad in declared for bad in _OOXML_FORBIDDEN_TYPES):
        return False
    return expected in declared


def _magic_ok(data: bytes, ext: str) -> bool:
    """Do the bytes actually match the extension the model claimed?"""
    if not data:
        return False
    if ext == "webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    expected = _MAGIC.get(ext)
    if expected:
        if not any(data.startswith(sig) for sig in expected):
            return False
        if ext in _OOXML_EXTS:
            return _ooxml_ok(data, ext)
        return True
    if ext in _TEXT_EXTS:
        if b"\x00" in data[:1024]:
            return False
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


async def resolve_container_artifacts(openai_client: Any,
                                      container_ids: List[str]) -> List[ArtifactRef]:
    """List the files the model WROTE, across every container this turn touched.

    Paginates: a turn that wrote many files must not have the one the user asked for hidden
    behind a page boundary. Never raises — a container that already idle-expired yields
    nothing, which costs files, not the reply.
    """
    refs: List[ArtifactRef] = []
    seen_ids = set()
    for container_id in container_ids:
        try:
            raw = openai_client.client

            async def _walk(cid=container_id):
                pager = raw.containers.files.list(container_id=cid)
                async for f in pager:  # auto-paginates
                    # Fail CLOSED: only files the assistant itself wrote. A user's own
                    # attachment mounts here too, and is never ours to re-publish.
                    if getattr(f, "source", None) != "assistant":
                        continue
                    fid = getattr(f, "id", "")
                    if not fid or fid in seen_ids:
                        continue
                    seen_ids.add(fid)
                    path = getattr(f, "path", "") or ""
                    size = getattr(f, "bytes", None)
                    refs.append(ArtifactRef(container_id=cid, file_id=fid,
                                            filename=path.rsplit("/", 1)[-1],
                                            size_bytes=size if isinstance(size, int) else None))

            await asyncio.wait_for(_walk(), timeout=_API_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not list container {container_id}: {e}")
    return refs


async def _stream_download(raw: Any, ref: ArtifactRef, max_bytes: int) -> Optional[bytes]:
    """Pull the body in chunks, ABORTING the moment it exceeds the cap.

    The plain `content.retrieve()` buffers the entire body before returning, so a check made
    afterwards is not a limit at all — the model can write a multi-gigabyte file and we would
    already be holding it. Streaming means an oversized artifact costs one chunk, not the file.
    """
    buf = io.BytesIO()
    total = 0
    async with raw.containers.files.content.with_streaming_response.retrieve(
            ref.file_id, container_id=ref.container_id) as resp:
        declared = 0
        try:
            declared = int((resp.headers or {}).get("content-length") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > max_bytes:
            logger.warning(
                f"Artifact refused before download ({declared} bytes > cap): {ref.filename}")
            return None

        async for chunk in resp.iter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                logger.warning(
                    f"Artifact refused mid-download (exceeded {config.artifact_max_mb}MB cap): "
                    f"{ref.filename}")
                return None
            buf.write(chunk)
    return buf.getvalue() or None


async def _download(openai_client: Any, ref: ArtifactRef, max_bytes: int) -> Optional[bytes]:
    """Fetch one container file's bytes, refusing anything over the cap. Never raises."""
    # Free rejection when the listing knows the size. It often does not (`bytes` comes back null
    # for assistant-written files), so this is an optimization — the streaming read below is the
    # actual bound.
    if ref.size_bytes is not None and ref.size_bytes > max_bytes:
        logger.warning(
            f"Artifact refused from listing ({ref.size_bytes} bytes > cap): {ref.filename}")
        return None
    try:
        raw = openai_client.client
        return await asyncio.wait_for(
            _stream_download(raw, ref, max_bytes), timeout=_API_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"Artifact download timed out: {ref.filename}")
        return None
    except Exception as e:  # noqa: BLE001 — an expired container must not kill the reply
        logger.warning(f"Artifact download failed for {ref.filename}: {e}")
        return None


async def _persist(db: Any, *, ext: str, filename: str, thread_key: str,
                   upload: Dict[str, Any], size: int, message_ts: Optional[str]) -> None:
    """Record the Slack ref (never the bytes) so the model can find its own artifact later.

    Images go to the image table and documents to the document table — routed by type, because
    `read_document` runs its input through DocumentHandler, which has no image parser. Filing a
    chart as a "document" would make it *look* re-readable and fail at read time.
    """
    if db is None:
        return
    try:
        url = upload.get("url_private")
        if ext in _IMAGE_EXTS:
            await db.save_image_metadata_async(
                thread_id=thread_key,
                url=url,
                image_type="generated",
                prompt="",
                analysis="",
                metadata={"source": "code_interpreter", "filename": filename},
            )
        else:
            db.save_document(
                thread_id=thread_key,
                filename=filename,
                mime_type=_EXT_MIME.get(ext, "application/octet-stream"),
                summary=f"Generated by the assistant ({filename}).",
                file_id=upload.get("file_id"),
                url_private=url,
                size_bytes=size,
                metadata={"source": "generated", "tool": "code_interpreter"},
                message_ts=message_ts,
            )
    except Exception as e:  # noqa: BLE001 — the file is already in Slack; never un-post it
        logger.warning(f"Artifact published to Slack but not persisted ({filename}): {e}")


async def publish_artifacts(
    *,
    openai_client: Any,
    client: Any,
    channel_id: str,
    thread_id: str,
    thread_key: str,
    container_ids: Optional[List[str]] = None,
    db: Any = None,
    message_ts: Optional[str] = None,
    container_manager: Any = None,
) -> List[Dict[str, Any]]:
    """Publish everything the model wrote THIS turn. Returns the published descriptors.

    Holds the thread's publication latch for its whole body. The thread lock is already gone by
    the time this runs (the reply has posted, `process_message` has returned), so without the
    latch the NEXT turn could be writing files into the same persistent container while we list
    it — and we would post its half-finished work under the previous answer.
    """
    lock = publication_lock(thread_key)
    try:
        async with lock:
            return await _publish_locked(
                openai_client=openai_client, client=client, channel_id=channel_id,
                thread_id=thread_id, thread_key=thread_key, container_ids=container_ids,
                db=db, message_ts=message_ts, container_manager=container_manager)
    finally:
        release_publication_lock(thread_key)


async def _publish_locked(
    *,
    openai_client: Any,
    client: Any,
    channel_id: str,
    thread_id: str,
    thread_key: str,
    container_ids: Optional[List[str]],
    db: Any,
    message_ts: Optional[str],
    container_manager: Any,
) -> List[Dict[str, Any]]:
    """The publication body. Runs under the thread's publication latch.

    The cap counts ACCEPTED uploads, not candidates: an intermediate file the model happened to
    write must not consume the budget the real deliverable needed.
    """
    refs = await resolve_container_artifacts(openai_client, container_ids or [])

    # Containers are thread-scoped and reused, so the listing is cumulative: it holds every file
    # the model has written in this thread, not just this turn's. Suppress anything already
    # handled — uploaded by this process (fast path), or recorded durably by an earlier one
    # (which is what survives a restart mid-conversation, and what carries the turn baseline).
    already: set = set(_published_file_ids)
    if container_manager is not None:
        for cid in {r.container_id for r in refs}:
            try:
                already.update(await container_manager.get_published_files(thread_key, cid))
            except Exception as e:  # noqa: BLE001 — worst case we re-post; never fail the turn
                logger.warning(f"Could not load published-file record for {thread_key}: {e}")

    refs = [r for r in refs if r.file_id not in already]
    if not refs:
        return []

    max_bytes = config.artifact_max_mb * 1024 * 1024
    cap = config.artifact_max_files
    published: List[Dict[str, Any]] = []
    published_ids: Dict[str, List[str]] = {}
    seen_hashes = set()
    skipped = 0

    for index, ref in enumerate(refs, start=1):
        if len(published) >= cap:
            skipped = len(refs) - index + 1
            break

        filename = sanitize_filename(ref.filename, fallback_index=index)
        if not filename:
            logger.warning(f"Artifact rejected (name/extension not allowed): {ref.filename!r}")
            continue

        data = await _download(openai_client, ref, max_bytes)
        if not data:
            continue

        if len(data) > max_bytes:
            logger.warning(
                f"Artifact rejected (>{config.artifact_max_mb}MB): {filename} ({len(data)} bytes)")
            continue

        ext = filename.rpartition(".")[2].lower()
        if not _magic_ok(data, ext):
            logger.warning(f"Artifact rejected (content does not match .{ext}): {filename}")
            continue

        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes:
            # e.g. the model saved a chart AND displayed it — same bytes, one upload.
            logger.info(f"Artifact skipped (duplicate content): {filename}")
            continue
        seen_hashes.add(digest)

        try:
            upload = await client.send_file(
                channel_id=channel_id, thread_id=thread_id,
                file_data=io.BytesIO(data), filename=filename,
            )
        except Exception as e:  # noqa: BLE001 — publish_artifacts must never raise
            logger.warning(f"Artifact upload raised for {filename}: {e}")
            continue

        if not upload or not upload.get("file_id"):
            # A response without a usable identity is not a success: we could neither find the
            # file again nor persist a meaningful ref.
            logger.warning(f"Artifact upload returned no usable file identity: {filename}")
            continue

        logger.info(f"Artifact published: {filename} ({len(data)} bytes)")
        _remember_published(ref.file_id)
        published_ids.setdefault(ref.container_id, []).append(ref.file_id)
        published.append({"filename": filename, "size_bytes": len(data), **upload})

        await _persist(db, ext=ext, filename=filename, thread_key=thread_key,
                       upload=upload, size=len(data), message_ts=message_ts)

    # Durably remember what went out, per container, so the next turn's listing doesn't re-post
    # it — and so a restart in between doesn't either. Keyed by container because an `auto`
    # fallback container has no row of its own, and its ids must never be written into the
    # thread's persistent binding.
    if published_ids and container_manager is not None:
        for cid, ids in published_ids.items():
            try:
                await container_manager.remember_published(thread_key, cid, ids)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not record published files for {thread_key}: {e}")

    if skipped > 0:
        # Never silently truncate — a dropped deliverable the user asked for should be visible
        # in the logs, not inferred from its absence.
        logger.warning(f"Artifact cap ({cap}) reached — {skipped} further file(s) not published")

    return published
