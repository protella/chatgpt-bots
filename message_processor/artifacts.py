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
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from config import config
from logger import setup_logger
from message_processor.containers import publication_lock, release_publication_lock
from message_processor.tool_provenance import strip_provenance_echo

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
    # A plain archive: local-file-header (PK\x03\x04), empty-archive (PK\x05\x06), or spanned
    # (PK\x07\x08) magic. An advertised "archive" deliverable is delivered as a .zip.
    "zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
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


# Web-search citation markers. The Responses API wraps them in Private Use Area delimiters —
# U+E200 cite U+E202 turn12search1 U+E201 — which are invisible to the model's own eye and
# meaningless to Slack. They are NOT harmless: the PUA codepoints get rendered downstream as
# arbitrary emoji, so a cited sentence reaches the user as
#   "…one-million-token context. cite:ship:turn12search1:walking:"
# The payload sits BETWEEN the delimiters, so stripping the PUA characters alone would be worse
# than doing nothing — it would leave the literal text "citeturn12search1" behind. The whole
# span has to go.
_CITATION_RE = re.compile("\ue200.*?\ue201", re.DOTALL)
_PUA_RE = re.compile("[\ue200-\ue20f]")


def strip_citation_markers(text: str) -> str:
    """Remove the Responses API's PUA-delimited web-search citations.

    Verified against a live report. The raw text carries:
        \\ue200 cite \\ue202 turn9search0 \\ue202 turn9search1 \\ue201
    and by the time it reaches Slack the inner delimiters have already been mapped to emoji
    shortcodes, so the user reads "...recurrent approaches. cite:ship:turn9search0:walking:".
    Strip the whole span at the SOURCE, before any emoji conversion can get to it.
    """
    if not text:
        return text
    out = _CITATION_RE.sub("", text)
    # Belt and braces: an unpaired delimiter would still render as a stray emoji.
    return _PUA_RE.sub("", out)


# A markdown link or a bare sandbox path that is still being written. We cannot yet tell what
# it will become, and a native stream is APPEND-ONLY — once a character is sent it cannot be
# unsent — so the tail is held back until it resolves.
_OPEN_LINKISH_RE = re.compile(r"\[[^\]\n]*\]\([^)\n]*$|\[[^\]\n]*$|\[[^\]\n]*\]$")


def stream_safe_text(text: str, final: bool = False) -> str:
    """What is safe to APPEND to an append-only native stream right now.

    ``strip_sandbox_links`` alone was not enough, and this is why: it ran only at finalize,
    while the stream appends as it goes. By the time we stripped the link it was already in
    Slack, the stripped text was SHORTER than what had been sent, so the delta was empty and
    the dead link simply stayed — a clickable "Download the deck" that leads nowhere, sitting
    right above the real file. Verified live.

    So the strip has to happen at the APPEND, and it has to be PREFIX-STABLE: what we send for
    a prefix of the text must remain a prefix of what we send for the whole text, or the sink's
    sent-length bookkeeping desynchronizes and text duplicates or vanishes. Hence no ``.strip()``
    and no whitespace collapsing here (``strip_sandbox_links`` does both, which is fine for a
    one-shot non-streaming post and fatal mid-stream), and hence the hold-back: anything that
    could still turn into a sandbox link waits until we can see what it actually is.

    Whatever is held back is not lost — ``final=True`` releases it, and by then any real
    sandbox link has been removed for good.
    """
    if not text:
        return text
    # Complete `[label](sandbox:/…)` links collapse to their label on both paths, so the two
    # agree about them. Completed citation spans vanish on both paths, likewise.
    out = _SANDBOX_LINK_RE.sub(lambda m: m.group(1) or "", text)
    out = _CITATION_RE.sub("", out)

    if final:
        out = _SANDBOX_BARE_RE.sub("", out)
        out = _PUA_RE.sub("", out)
        return strip_provenance_echo(out)

    # Mid-stream, a bare path CANNOT be told apart from a path still being typed — the regex
    # happily matches `sandbox:/mnt/data/deck.` as if it were whole, and substituting it away
    # left a dangling `[Download it].` that sailed straight through the hold-back and into
    # Slack. So don't substitute at all here: cut at the FIRST unresolved mention and wait.
    idx = out.find("sandbox:")
    if idx != -1:
        out = out[:idx]
    # A citation span still arriving is the same problem: its opener has landed but its closer
    # has not, so the span cannot be matched yet. Emit up to the opener and wait — appending
    # `citeturn9` and only THEN learning it was a citation is unfixable, because
    # the stream is append-only.
    cut = out.find("")
    if cut != -1:
        out = out[:cut]
    out = strip_provenance_echo(out)
    # ...then cut back over any markdown-link syntax left dangling in front of it (or still
    # being written), since it may yet turn out to wrap a sandbox path.
    return _OPEN_LINKISH_RE.sub("", out)


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


async def _download(openai_client: Any, ref: ArtifactRef, max_bytes: int,
                    timeout: float = _API_TIMEOUT) -> Optional[bytes]:
    """Fetch one container file's bytes, refusing anything over the cap. Never raises.

    ``timeout`` lets the caller shrink the per-file bound below ``_API_TIMEOUT`` — when a whole
    staging phase is on a deadline, the LAST download must not be free to run a full 30s past it.
    """
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
            _stream_download(raw, ref, max_bytes), timeout=timeout)
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
    ledger_key: Optional[str] = None,
    suppress_digests: Optional[Iterable[str]] = None,
    expect_filenames: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Publish everything the model wrote THIS turn. Returns the published descriptors.

    Holds the publication latch for its whole body. The thread lock is already gone by the time
    this runs (the reply has posted, `process_message` has returned), so without the latch the
    NEXT turn could be writing files into the same persistent container while we list it — and
    we would post its half-finished work under the previous answer.

    `ledger_key` scopes the *container* concerns — the latch and the already-published record —
    while `thread_key` stays the *thread* concern: which conversation the DB rows belong to.
    They are the same string for a normal turn. A background job separates them: it builds in
    its OWN container (sharing the thread's would let a chat turn's baseline snapshot silently
    mark the job's half-written deck as "already published", and it would never be posted), but
    the deck it produces still belongs to the thread the user asked in, or tomorrow's "revise
    that deck" would not be able to find it.

    `suppress_digests` refuses to publish bytes we ourselves put INTO the container — a mounted
    input is an ingredient. `containers.files.create` already marks uploads `source="user"` and
    the listing only yields `"assistant"` files, so this is the second lock on the same door:
    it also catches a model that copies the user's spreadsheet to a new name and thereby makes
    an assistant-owned, byte-identical twin of a file they already have.

    `expect_filenames` is the DECLARED deliverable manifest, and when it is present it is the
    whole answer: the user asked for a PDF, so they get the PDF and nothing else. Everything
    else in the container is working material by definition. Only a background build knows this
    up front — a chat turn has no manifest and must fall back to the heuristics below.
    """
    ledger = ledger_key or thread_key
    lock = publication_lock(ledger)
    try:
        async with lock:
            return await _publish_locked(
                openai_client=openai_client, client=client, channel_id=channel_id,
                thread_id=thread_id, thread_key=thread_key, container_ids=container_ids,
                db=db, message_ts=message_ts, container_manager=container_manager,
                ledger_key=ledger, suppress_digests=set(suppress_digests or ()),
                expect_filenames=[f.lower() for f in (expect_filenames or ())])
    finally:
        release_publication_lock(ledger)


# Zip-backed document formats. An image placed into one of these is stored as a plain zip
# entry, byte-for-byte, which is what makes ingredient detection exact rather than a guess.
_COMPOSITE_EXTS = {"pptx", "docx", "xlsx"}

# Documents that are NOT zips, so the exact byte-match above cannot see inside them. A PDF
# re-encodes an image on the way in (DCTDecode/FlateDecode), so the chart embedded in the
# document and the loose chart in the container share no bytes at all.
_OPAQUE_COMPOSITE_EXTS = {"pdf"}

# Every format whose whole point is to CONTAIN the other files.
_DOCUMENT_EXTS = _COMPOSITE_EXTS | _OPAQUE_COMPOSITE_EXTS


def _embedded_member_hashes(candidates: List[Dict[str, Any]]) -> set:
    """sha256 of every file embedded inside the composite documents we're about to publish.

    Used to tell a deliverable apart from an ingredient. python-pptx/docx store an added
    picture as an unmodified zip entry, so the hash matches the loose file exactly. If the
    model re-encodes an image on the way in, the hashes won't match and the loose copy still
    goes out — this is a precise suppression, never a guess at what looks like a leftover.
    """
    hashes: set = set()
    for candidate in candidates:
        if candidate["ext"] not in _COMPOSITE_EXTS:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(candidate["data"])) as zf:
                for info in zf.infolist():
                    if info.is_dir() or info.file_size == 0:
                        continue
                    hashes.add(hashlib.sha256(zf.read(info)).hexdigest())
        except Exception as e:  # noqa: BLE001 — a corrupt zip must not stop publication
            logger.debug(f"Could not inspect {candidate['filename']} for embedded files: {e}")
    return hashes


def _base_name(ref: ArtifactRef) -> str:
    return ref.filename.rsplit("/", 1)[-1].lower()


def _prioritize_declared(refs: List[ArtifactRef],
                         expect_filenames: List[str]) -> List[ArtifactRef]:
    """Move refs that match the declared manifest to the FRONT of the listing.

    The download budget below is spent in listing order, and the listing is the container's
    creation order — so a deliverable written AFTER >=budget intermediates would never be
    downloaded, and `_select_candidates` could then never pick it. Ordering (not budget size)
    is the fix.

    Tiers, because an exact name is a stronger signal than a shared extension:
      * tier 0 — the file selection will actually claim: an exact-name match, or (for an entry
        with no exact match) the NEWEST same-extension ref, i.e. the LAST one in listing order.
        Resolving "newest" from the listing here — not from the downloaded subset later — is what
        guarantees the exact file selection wants is inside the budget even when the same-ext
        drafts outnumber it.
      * tier 1 — the other same-extension drafts (ahead of unrelated intermediates, behind the
        one that will be claimed).
      * tier 2 — everything else.
    The sort is STABLE, so creation order is preserved within each tier.
    """
    if not expect_filenames:
        return refs
    names = set(expect_filenames)
    manifest_exts = {f.rpartition(".")[2] for f in expect_filenames if "." in f}

    # Which declared entries already have an exact-name match? Only entries WITHOUT one fall back
    # to an extension, and each such entry promotes its newest (last-in-listing) same-ext ref.
    present = {f for f in expect_filenames if any(_base_name(r) == f for r in refs)}
    promoted: set = set()
    for fname in expect_filenames:
        if fname in present or "." not in fname:
            continue
        ext = fname.rpartition(".")[2]
        same = [r for r in refs
                if _base_name(r).rpartition(".")[2] == ext and id(r) not in promoted]
        if same:
            promoted.add(id(same[-1]))  # last in listing order = newest

    def _tier(ref: ArtifactRef) -> int:
        base = _base_name(ref)
        if base in names or id(ref) in promoted:
            return 0
        if base.rpartition(".")[2] in manifest_exts:
            return 1
        return 2

    return sorted(refs, key=_tier)


async def _gather_candidates(
    *,
    openai_client: Any,
    container_ids: Optional[List[str]],
    container_manager: Any,
    ledger_key: str,
    expect_filenames: Optional[List[str]] = None,
    deadline: Optional[float] = None,
) -> tuple:
    """Phase 1 of publication: list the container(s), drop what already went out, and download
    + validate everything that's left. Returns ``(candidates, skipped)``.

    Bytes land in memory and stay there — never on disk (CLAUDE.md). Downloading BEFORE any
    decision is deliberate: whether a chart is a deliverable or an ingredient of the deck beside
    it can only be answered by looking inside the deck.

    ``deadline`` (a ``time.monotonic()`` value) time-bounds the download loop from the INSIDE, so
    a slow container yields whatever was staged before the budget ran out instead of losing
    everything to an outer cancel. Per-file bounding (``_API_TIMEOUT``) still applies; this only
    stops starting NEW downloads once the budget is spent. F14's manifest prioritization runs
    first, so the files that survive a truncated budget are the declared deliverables.
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
                already.update(await container_manager.get_published_files(ledger_key, cid))
            except Exception as e:  # noqa: BLE001 — worst case we re-post; never fail the turn
                logger.warning(f"Could not load published-file record for {ledger_key}: {e}")

    refs = [r for r in refs if r.file_id not in already]
    if not refs:
        return [], 0

    # Stamp each ref with its LISTING position (creation order) BEFORE prioritization reshuffles
    # them, so the "newest same-ext" fallback in _select_candidates resolves against the true
    # listing order rather than the downloaded subset.
    listing_order = {id(r): i for i, r in enumerate(refs)}

    # A declared deliverable must be downloaded even if it was written after a budget's worth of
    # intermediates — so pull the manifest matches to the front before the budget bites.
    refs = _prioritize_declared(refs, expect_filenames or [])

    max_bytes = config.artifact_max_mb * 1024 * 1024
    cap = config.artifact_max_files
    skipped = 0

    # Fetch and validate every candidate BEFORE deciding what to send, because that decision
    # is not per-file: a chart that was embedded into a deck is an ingredient, not a
    # deliverable, and we can only know that by looking at the deck. Bounded so a runaway loop
    # writing hundreds of files can't turn into hundreds of downloads.
    download_budget = max(cap * 4, cap)
    candidates: List[Dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        if len(candidates) >= download_budget:
            skipped = len(refs) - index + 1
            break

        # How much of the time budget is left. <= 0 means STOP starting downloads — hand back what
        # we already staged (the declared deliverables sorted to the front, so they are what made
        # it) rather than nothing.
        remaining = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                skipped = len(refs) - index + 1
                logger.warning(
                    f"Artifact gather hit its time budget after {len(candidates)} file(s) — "
                    f"{skipped} container file(s) never examined")
                break

        filename = sanitize_filename(ref.filename, fallback_index=index)
        if not filename:
            logger.warning(f"Artifact rejected (name/extension not allowed): {ref.filename!r}")
            continue

        # Bound THIS download by whatever is left of the budget (exact, no floor), so an in-flight
        # read can't sail a full _API_TIMEOUT past the deadline.
        dl_timeout = _API_TIMEOUT if remaining is None else min(_API_TIMEOUT, remaining)
        data = await _download(openai_client, ref, max_bytes, timeout=dl_timeout)
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

        candidates.append({"ref": ref, "filename": filename, "ext": ext, "data": data,
                           "digest": hashlib.sha256(data).hexdigest(),
                           "order": listing_order.get(id(ref), index)})

    return candidates, skipped


def _declared_candidate_ids(candidates: List[Dict[str, Any]],
                            expect_filenames: List[str]) -> set:
    """Which candidates actually satisfy the declared manifest, matched entry by entry.

    For each declared filename: every EXACT-name match counts (the model may write the same name
    more than once, and all such copies were literally asked for). Only when an entry has NO
    exact-name match does the EXTENSION fallback engage — and then it claims just ONE candidate,
    the newest same-extension file not already claimed. That "one best" rule is the whole point:
    a shared extension is a weak signal, so a swarm of `.pptx` drafts must not ALL count as the
    one `deck.pptx` that was asked for (which would exempt them all from the superseded-draft
    filter and publish every draft). Creation order is preserved by the caller within an
    extension, so the last is the finished one.
    """
    declared: set = set()
    lowered = [(c, c["filename"].lower()) for c in candidates]

    # Pass 1 — exact names. An entry satisfied here never reaches the extension fallback.
    exact_hit: set = set()
    for fname in expect_filenames:
        hits = [c for c, low in lowered if low == fname]
        if hits:
            exact_hit.add(fname)
            declared.update(id(c) for c in hits)

    # Pass 2 — extension fallback, one file per still-unsatisfied entry: the NEWEST same-ext
    # candidate, resolved by LISTING order (`order`, stamped in _gather_candidates before the
    # prioritized download reshuffles things). Falling back to the candidate's position keeps the
    # direct-call/test path sensible when no listing order was stamped.
    for fname in expect_filenames:
        if fname in exact_hit or "." not in fname:
            continue
        ext = fname.rpartition(".")[2]
        same_ext = [(i, c) for i, c in enumerate(candidates)
                    if c["ext"] == ext and id(c) not in declared]
        if same_ext:
            _, chosen = max(same_ext, key=lambda ic: (ic[1].get("order", ic[0]), ic[0]))
            declared.add(id(chosen))
    return declared


def _select_candidates(candidates: List[Dict[str, Any]], *, suppress_digests: set,
                       expect_filenames: List[str]) -> List[Dict[str, Any]]:
    """Phase 2: decide which downloaded candidates are DELIVERABLES rather than working
    material. Pure selection, no I/O — which is what lets a background job run it, show the
    result to the model, and upload only later (F37).

    Every rule below answers one question: the model wrote this file, but did the user ask for
    it?
    """
    # A .pptx/.docx/.xlsx is a zip, and an image embedded into one is stored as a verbatim zip
    # entry — so the deck itself tells us which of the other files were merely its ingredients.
    # Ask for a deck and you should get a deck, not a deck plus the loose charts that went into
    # it. (The prompt also tells the model to embed from memory; this is the part that does not
    # depend on the model complying.)
    embedded = _embedded_member_hashes(candidates)

    # A DECLARED manifest beats every heuristic below: the caller knows exactly which files were
    # asked for, so anything else in the container is working material. Matched on name, and on
    # extension too — the model does not always honour the filename it was given, and delivering
    # the right document under a slightly wrong name beats delivering nothing.
    #
    # A matched candidate is a DECLARED deliverable, and the heuristics below (superseded-draft
    # pruning, document-image suppression) exist to guess a chat turn's intent — they must never
    # fire on a file the caller explicitly asked for. A job declaring report.pdf AND
    # social-card.png would otherwise lose the PNG the moment the PDF made it a "document turn".
    declared_ids: set = set()
    if expect_filenames:
        declared_ids = _declared_candidate_ids(candidates, expect_filenames)
        keep = [c for c in candidates if id(c) in declared_ids]
        if keep:
            dropped = [c["filename"] for c in candidates if id(c) not in declared_ids]
            if dropped:
                # Never silently truncate — say what was held back and why.
                logger.info(f"Artifacts held back (not a declared deliverable): {dropped}")
            candidates = keep
        else:
            # Nothing matched. Publishing the intermediates would be worse than useless — it
            # would look like the deliverable. Say so loudly and hand back the raw candidates,
            # so a mis-named deck still reaches the user rather than vanishing.
            logger.warning(
                f"No candidate matched the declared deliverables {expect_filenames}; "
                f"falling back to {[c['filename'] for c in candidates]}")

    # A model that revises its work leaves the draft behind. It wrote Board_Ready.pdf, thought
    # better of it, wrote Board_Brief.pdf — and the user got both, with no way to tell which was
    # the real one. The listing is in creation order, so within one extension the LAST document
    # is the finished one; the earlier ones are the drafts it moved on from.
    #
    # Scoped to DOCUMENT types on purpose. "Give me a chart and a cleaned CSV" is a normal ask
    # and must keep working; "give me two different PDFs in one turn" is not, and the log says
    # plainly what was held back if it ever happens.
    by_ext: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["ext"] in _DOCUMENT_EXTS:
            by_ext.setdefault(candidate["ext"], []).append(candidate)
    superseded = {id(c) for group in by_ext.values() if len(group) > 1 for c in group[:-1]
                  if id(c) not in declared_ids}
    if superseded:
        logger.info(
            "Artifacts held back (superseded drafts of the same document type): "
            f"{[c['filename'] for c in candidates if id(c) in superseded]}")
        candidates = [c for c in candidates if id(c) not in superseded]

    # Is a DOCUMENT going out this turn? Then the loose images beside it are the pictures that
    # went into it, and the user asked for the document.
    #
    # The exact hash check above catches this for zip formats, but it cannot for a PDF (which
    # re-encodes what it embeds) and it cannot for a chart the model merely DISPLAYED (a
    # display render is a separate rasterisation, so its bytes never match the embedded copy).
    # Both holes were live: a PDF shipped with its own two 1MB charts posted next to it.
    #
    # Note this stays scoped to turns that produce a document. A turn that just draws a chart
    # publishes it, exactly as before — being asked for a picture and being asked for a report
    # that contains pictures are different requests.
    publishing_document = any(c["ext"] in _DOCUMENT_EXTS for c in candidates)

    accepted: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    for candidate in candidates:
        filename = candidate["filename"]
        ext, digest = candidate["ext"], candidate["digest"]

        if digest in suppress_digests:
            # Byte-identical to something WE mounted into the container. The user already has
            # this file — they gave it to us. Posting it back is noise at best.
            logger.info(f"Artifact suppressed (this is a mounted input, not output): {filename}")
            continue

        if digest in embedded:
            logger.info(
                f"Artifact suppressed (embedded in a document we're publishing): {filename}")
            continue

        if publishing_document and ext in _IMAGE_EXTS and id(candidate) not in declared_ids:
            logger.info(
                f"Artifact suppressed (an ingredient of the document being published): {filename}")
            continue

        if digest in seen_hashes:
            # e.g. the model saved a chart AND displayed it — same bytes, one upload.
            logger.info(f"Artifact skipped (duplicate content): {filename}")
            continue
        seen_hashes.add(digest)

        accepted.append(candidate)

    return accepted


async def _upload_candidates(
    accepted: List[Dict[str, Any]],
    *,
    client: Any,
    channel_id: str,
    thread_id: str,
    thread_key: str,
    db: Any,
    message_ts: Optional[str],
    container_manager: Any,
    ledger_key: str,
    skipped: int = 0,
) -> List[Dict[str, Any]]:
    """Phase 3: upload the selected files to Slack and persist their refs.

    The cap counts ACCEPTED uploads, not candidates: an intermediate file the model happened to
    write must not consume the budget the real deliverable needed.
    """
    cap = config.artifact_max_files
    published: List[Dict[str, Any]] = []
    published_ids: Dict[str, List[str]] = {}

    for candidate in accepted:
        if len(published) >= cap:
            skipped += 1
            continue

        ref, filename = candidate["ref"], candidate["filename"]
        ext, data = candidate["ext"], candidate["data"]

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
                await container_manager.remember_published(ledger_key, cid, ids)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not record published files for {ledger_key}: {e}")

    if skipped > 0:
        # Never silently truncate — a dropped deliverable the user asked for should be visible
        # in the logs, not inferred from its absence.
        logger.warning(f"Artifact cap ({cap}) reached — {skipped} further file(s) not published")

    return published


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
    ledger_key: str,
    suppress_digests: set,
    expect_filenames: List[str],
) -> List[Dict[str, Any]]:
    """The publication body — gather, select, upload. Runs under the publication latch."""
    candidates, skipped = await _gather_candidates(
        openai_client=openai_client, container_ids=container_ids,
        container_manager=container_manager, ledger_key=ledger_key,
        expect_filenames=expect_filenames)
    if not candidates:
        return []
    accepted = _select_candidates(candidates, suppress_digests=suppress_digests,
                                  expect_filenames=expect_filenames)
    return await _upload_candidates(
        accepted, client=client, channel_id=channel_id, thread_id=thread_id,
        thread_key=thread_key, db=db, message_ts=message_ts,
        container_manager=container_manager, ledger_key=ledger_key, skipped=skipped)


# --- F37: staged publication (background jobs) ----------------------------------------------


@dataclass
class StagedArtifact:
    """A built file, downloaded and validated, waiting for someone to decide its fate.

    Staging exists so a background job's container can DIE the moment it finishes building. The
    API caps a container's idle life at 20 minutes; if the model that decides what to ship had
    to reach back into the container, a slow finalize — or a queue, or a retry — would lose the
    deliverable. So we pull the bytes out first and hold them in memory (never disk, CLAUDE.md),
    and the container is free to expire.

    ``artifact_id`` is opaque and application-issued on purpose. The model selects what to
    publish BY ID, never by filename: filenames are model-authored and therefore hallucinable,
    and a selection contract that silently matched the wrong file — or nothing — would ship the
    wrong deliverable with total confidence.
    """
    artifact_id: str
    filename: str
    ext: str
    size_bytes: int
    candidate: Dict[str, Any]   # the internal candidate dict (carries the bytes + ref)

    def manifest_entry(self) -> Dict[str, Any]:
        """What the model is shown. The bytes never go anywhere near the prompt."""
        return {"artifact_id": self.artifact_id, "filename": self.filename,
                "kind": self.ext, "size_bytes": self.size_bytes}


async def stage_artifacts(
    *,
    openai_client: Any,
    container_ids: Optional[List[str]] = None,
    container_manager: Any = None,
    ledger_key: str,
    suppress_digests: Optional[Iterable[str]] = None,
    expect_filenames: Optional[Iterable[str]] = None,
    time_budget: Optional[float] = None,
) -> List[StagedArtifact]:
    """Gather + select + hold in memory. Publishes NOTHING. Never raises.

    ``time_budget`` (seconds) bounds the download loop from the inside: on expiry the files
    staged SO FAR are returned, never discarded. This replaces an outer ``wait_for`` that
    cancelled the whole coroutine and lost every file a slow container had already produced.
    """
    lock = publication_lock(ledger_key)
    try:
        async with lock:
            # Start the clock once we hold the latch — lock contention must not eat the budget
            # the downloads need.
            deadline = (time.monotonic() + time_budget) if time_budget else None
            candidates, skipped = await _gather_candidates(
                openai_client=openai_client, container_ids=container_ids,
                container_manager=container_manager, ledger_key=ledger_key,
                expect_filenames=[f.lower() for f in (expect_filenames or ())],
                deadline=deadline)
            if skipped:
                # Never silently truncate: the model is about to be shown a manifest, and a file
                # missing from it is a file it cannot choose to publish.
                logger.warning(f"Staging for {ledger_key} stopped at the download budget — "
                               f"{skipped} container file(s) never examined")
            if not candidates:
                return []
            accepted = _select_candidates(
                candidates, suppress_digests=set(suppress_digests or ()),
                expect_filenames=[f.lower() for f in (expect_filenames or ())])
    except Exception as e:  # noqa: BLE001 — a staging failure costs the files, not the job
        logger.error(f"Artifact staging failed for {ledger_key}: {e}", exc_info=True)
        return []
    finally:
        release_publication_lock(ledger_key)

    staged = [
        StagedArtifact(artifact_id=f"art_{i}", filename=c["filename"], ext=c["ext"],
                       size_bytes=len(c["data"]), candidate=c)
        for i, c in enumerate(accepted, start=1)
    ]
    if staged:
        logger.info(f"Staged {len(staged)} artifact(s) for {ledger_key}: "
                    f"{[(s.artifact_id, s.filename) for s in staged]}")
    return staged


async def publish_staged(
    staged: List[StagedArtifact],
    artifact_ids: Iterable[str],
    *,
    client: Any,
    channel_id: str,
    thread_id: str,
    thread_key: str,
    db: Any = None,
    message_ts: Optional[str] = None,
    container_manager: Any = None,
    ledger_key: str,
) -> List[Dict[str, Any]]:
    """Upload the staged artifacts the model asked for, in the order it asked for them.

    An id that matches nothing is DROPPED and logged loudly — never resolved to a
    "close enough" file. Publishing the wrong deliverable confidently is worse than publishing
    none, and unlike a filename match there is no honest fallback available here.
    """
    by_id = {s.artifact_id: s for s in staged}
    chosen: List[Dict[str, Any]] = []
    seen: set = set()
    for artifact_id in artifact_ids or ():
        key = str(artifact_id).strip()
        target = by_id.get(key)
        if target is None:
            logger.warning(f"Delivery plan named an unknown artifact id {artifact_id!r} "
                           f"(have: {sorted(by_id)}) — dropped")
            continue
        if key in seen:
            # ["art_1", "art_1"] would otherwise upload and persist the same file twice.
            logger.info(f"Delivery plan named {key} more than once — publishing it once")
            continue
        seen.add(key)
        chosen.append(target.candidate)
    if not chosen:
        return []

    lock = publication_lock(ledger_key)
    try:
        async with lock:
            return await _upload_candidates(
                chosen, client=client, channel_id=channel_id, thread_id=thread_id,
                thread_key=thread_key, db=db, message_ts=message_ts,
                container_manager=container_manager, ledger_key=ledger_key)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Staged publication failed for {ledger_key}: {e}", exc_info=True)
        return []
    finally:
        release_publication_lock(ledger_key)
