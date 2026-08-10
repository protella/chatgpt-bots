"""Model-invoked document access (Phase D2 of the channel-teammate redesign).

Documents are never at rest (CLAUDE.md pitfall 6a): the DB row holds
summary + metadata + the Slack CDN ref, and this tool re-derives the full
text ON DEMAND — authenticated download into memory, BytesIO extraction,
return the requested slice. A process-lifetime bounded LRU of extracted
text (never persisted, gone on restart) makes iterating on one document
cheap.

Deleted Slack file ⇒ download fails ⇒ ``{"ok": False, "error": "file_deleted"}``
— a privacy feature: deleting a file in Slack genuinely removes its content
from the bot's reach; only the labeled summary row remains.

Executors never raise: every failure is an ``{"ok": False, ...}`` result.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from canvas_content import CANVAS_MIMETYPE
from config import config
from document_handler import DocumentHandler
from tool_registry import ToolContext, ToolRegistry

# One slice of document text per tool round — big enough to be useful,
# bounded so a huge doc can't blow the tool-result cap.
SLICE_CHARS = 4000
# Query mode: context window around each match, and max windows per call.
QUERY_WINDOW_CHARS = 600
QUERY_MAX_MATCHES = 3

# Shared extractor instance (stateless besides config; BytesIO-only by contract).
_document_handler = DocumentHandler()


class ExtractionCache:
    """Process-lifetime bounded LRU of extracted document text.

    Keyed by Slack file_id. NEVER persisted — entries live in memory only and
    die on eviction or restart (no-content-at-rest rule).
    """

    def __init__(self, max_entries: int):
        self.max_entries = max(1, max_entries)
        self._entries: "OrderedDict[str, str]" = OrderedDict()

    def get(self, file_id: str) -> Optional[str]:
        text = self._entries.get(file_id)
        if text is not None:
            self._entries.move_to_end(file_id)
        return text

    def put(self, file_id: str, text: str) -> None:
        self._entries[file_id] = text
        self._entries.move_to_end(file_id)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop everything. The cache is a process-wide singleton keyed by file id ALONE — no
        thread and no channel — which is correct because a Slack file id is globally unique and
        because `read_document` authorizes against the turn's own catalog BEFORE it ever consults
        this. Nothing here is a permission. But that global key means the entries outlive whatever
        put them there, so anything that needs a clean process (a test file, a privacy purge) needs
        a way to say so rather than reaching into `_entries`."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


_extraction_cache = ExtractionCache(config.doc_extraction_cache_size)


def get_read_document_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "read_document",
        "description": (
            "Read the full content of a document shared ANYWHERE in this channel — the current "
            "conversation is checked first, then every other thread. No summary needs to be in "
            "context: a filename seen in an attachment note (\"[+1 file: report.pdf]\"), in "
            "fetched history, or mentioned in chat is enough. Document summaries in context are "
            "SUMMARIES — use this tool whenever you need specific figures, quotes, table values, "
            "or sections a summary doesn't literally contain. Provide query to search inside the "
            "document, or offset to read sequentially."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Slack file id from the document summary header (preferred when known).",
                },
                "filename": {
                    "type": "string",
                    "description": "Document filename — from a summary, an attachment note, fetched "
                                   "history, or chat (used when file_id is unknown).",
                },
                "query": {
                    "type": "string",
                    "description": "Case-insensitive text to locate; returns surrounding context windows.",
                },
                "offset": {
                    "type": "integer",
                    "description": f"Character offset to read from (returns ~{SLICE_CHARS} chars).",
                },
            },
            "required": [],
        },
    }


def _resolve_document(docs: List[Dict[str, Any]], file_id: Optional[str],
                      filename: Optional[str]) -> Optional[Dict[str, Any]]:
    """Find the newest matching document row for this thread."""
    if file_id:
        for doc in reversed(docs):
            if doc.get("file_id") == file_id:
                return doc
    if filename:
        want = filename.strip().lower()
        for doc in reversed(docs):
            have = (doc.get("filename") or "").lower()
            if have == want or have.endswith("/" + want):
                return doc
        # Loose fallback: substring match, newest first
        for doc in reversed(docs):
            if want in (doc.get("filename") or "").lower():
                return doc
    if not file_id and not filename and docs:
        # No selector: default to the most recent document in the thread
        return docs[-1]
    return None


def _resolve_canonical_file(canonical: Optional[Dict[str, Dict[str, Any]]],
                            file_id: Optional[str],
                            filename: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve against the pinned channel window's file catalog.

    Only IMAGES are excluded — `read_document` extracts text, and an image has none to extract;
    `view_image` is the tool for those. Filename matching is exact-then-suffix-then-substring, the
    same ladder `_resolve_document` uses, so a model that read a name out of a stream item's file
    marker gets the same behavior it would get from a summary header.
    """
    entries = [e for e in (canonical or {}).values() if e.get("kind") != "image"]
    if not entries:
        return None
    if file_id:
        for entry in entries:
            if entry.get("file_id") == file_id:
                return entry
    if filename:
        want = filename.strip().lower()
        for match in (lambda have: have == want,
                      lambda have: have.endswith("/" + want),
                      lambda have: want in have):
            for entry in entries:
                if match((entry.get("filename") or "").lower()):
                    return entry
    return None


def _query_slices(text: str, query: str) -> List[Dict[str, Any]]:
    """Case-insensitive search returning up to QUERY_MAX_MATCHES context windows."""
    matches: List[Dict[str, Any]] = []
    lowered = text.lower()
    needle = query.lower()
    start = 0
    while len(matches) < QUERY_MAX_MATCHES:
        pos = lowered.find(needle, start)
        if pos == -1:
            break
        window_start = max(0, pos - QUERY_WINDOW_CHARS)
        window_end = min(len(text), pos + len(needle) + QUERY_WINDOW_CHARS)
        matches.append({
            "position": pos,
            "context": text[window_start:window_end],
        })
        start = pos + len(needle)
    return matches


def extraction_warning(extracted: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why this extractor result is not the document — or ``None`` when it is.

    Three distinct ways the extractor hands back truthy text that is not the file's text, and
    they do NOT share a signal:

    ``error`` — a failure it recovered from. ``warning`` — a lower-fidelity fallback path (the
    DOCX xml-parsing and pandoc routes, a lossy character decode). Both are explicit.

    The third is not, and it is the dangerous one: a SCANNED PDF whose text was never recovered.
    ``parse_pdf_structured`` marks the scan with ``is_image_based``/``requires_ocr``, and when
    OCR is off, unavailable, or fruitless it substitutes an explanatory note — "[Note: This PDF
    appears to be a scanned document…]" — as the content, with no error and no warning attached.
    Read literally that is a truthy, clean-looking extraction whose entire content is a status
    message about itself. Cached, it becomes the document as far as every later read is
    concerned; handed to a revision master, the build faithfully "revises" a status message
    into the deliverable and drops the real document.

    ``ocr_text_used`` is the ONLY field that means the content string carries text recovered
    from the scan. ``ocr_processed`` deliberately does not qualify: it marks that page IMAGES
    were sent to a vision model, while the content string it accompanies was REPLACED by a
    one-line note about that having happened.

    Returning the reason rather than a bool so callers can surface something truthy — for a
    scan placeholder there is no upstream message to forward, and a `None` "warning" would sail
    straight through the master's rejection check.
    """
    if not extracted:
        return "no extraction result"
    error = extracted.get("error")
    if error:
        return str(error)
    warning = extracted.get("warning")
    if warning:
        return str(warning)
    if ((extracted.get("requires_ocr") or extracted.get("is_image_based"))
            and not extracted.get("ocr_text_used")):
        return ("scanned document: OCR did not recover the text, so this content is an "
                "explanatory note rather than the document")
    return None


def extraction_is_clean(extracted: Optional[Dict[str, Any]]) -> bool:
    """Is this extractor result safe to CACHE and to hand a revision master?

    One definition, because the invariant has to hold everywhere the LRU is written — the
    reader, the master path, and the two ingestion warmers in `utilities.py`.
    """
    return extraction_warning(extracted) is None


async def load_document_text(client: Any, doc: Dict[str, Any], *,
                             bypass_cache: bool = False,
                             on_miss: Optional[Callable[[], Awaitable[None]]] = None,
                             ) -> Dict[str, Any]:
    """Fetch + extract ONE document row's full text, in memory, for whoever asks.

    Hoisted out of ``execute_read_document`` so a second caller — the background job's
    revision master (``research_tools._load_revision_master``) — gets the identical
    download/extract path rather than a parallel copy of it that drifts.

    Two things the two callers genuinely disagree about, and nothing else:

    ``bypass_cache`` — the reader wants the LRU (iterating on one document is the common
    case). A master must never read it: the cache is not purged when a file is deleted in
    Slack, so a cached entry could resurrect content the user removed, and a revision master
    has to be as fresh as Slack is anyway.

    ``on_miss`` — the reader stakes its 👀 only when it is about to do real work (F38), which
    is exactly a cache miss. Awaited after the source-ref check and before the download, so a
    row that can never be fetched does not claim anything.

    Returns ``{"content": str, "cached": bool}`` — plus a truthy ``"warning"`` whenever the
    text came back DEGRADED, in any of the three senses ``extraction_warning`` recognises: an
    extractor ``error`` it recovered from, an extractor ``warning`` from a lower-fidelity
    fallback, or a scanned PDF whose text was never recovered (marked only by
    ``requires_ocr``/``is_image_based`` with no ``ocr_text_used``, and carrying an explanatory
    note as its content). Otherwise ``{"error": code}`` with ``code`` one of ``no_source_ref``,
    ``download_failed``, ``file_deleted``, ``extraction_failed`` — the last two carrying a
    ``detail``.

    Degraded content is NEVER cached: caching it would let a later read serve a placeholder,
    a partial extraction, or a note about a scan as a clean hit with the signal stripped off.
    """
    doc_file_id = doc.get("file_id")
    url_private = doc.get("url_private")
    if not url_private and not doc_file_id:
        # Before the cache and before on_miss: a row with no way to fetch it claims nothing.
        return {"error": "no_source_ref"}

    cache_key = cast(str, doc_file_id or url_private)  # one of the two is present (checked above)
    if not bypass_cache:
        cached = _extraction_cache.get(cache_key)
        if cached is not None:
            return {"content": cached, "cached": True}

    if on_miss is not None:
        await on_miss()
    try:
        # A canvas body IS html, so it has to opt out of the login-page guard that would
        # otherwise reject it; parse_canvas makes that check itself.
        data = await client.download_file(
            url_private, doc_file_id,
            allow_html=(doc.get("mime_type") == CANVAS_MIMETYPE))
    except Exception as e:  # noqa: BLE001 — every failure is a result, never a raise
        return {"error": "download_failed", "detail": str(e)}
    if not data:
        # Deleted-in-Slack is indistinguishable from never-there — by design,
        # deletion removes the content from the bot's reach.
        return {"error": "file_deleted"}

    extracted = await _document_handler.safe_extract_content_async(
        data, doc.get("mime_type") or "application/octet-stream",
        doc.get("filename") or "document",
        # Text slices only: page images are useless in a tool result, but OCR TEXT
        # rescues image-only/scanned PDFs that yield nothing from local extraction.
        ocr_images=False, ocr_text=True)
    text = (extracted or {}).get("content")
    if not text:
        return {"error": "extraction_failed",
                "detail": (extracted or {}).get("error", "no content extracted")}
    result: Dict[str, Any] = {"content": text, "cached": False}
    degraded = extraction_warning(extracted)
    if degraded is None:
        _extraction_cache.put(cache_key, text)
    else:
        # Every flavour of degradation surfaced under ONE name, always truthy, so callers have
        # one thing to check: the reader ignores it (partial text beats no text), a master
        # rejects on it.
        result["warning"] = degraded
    return result


async def execute_read_document(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Download from Slack CDN (memory only) -> extract (BytesIO) -> return the slice."""
    file_id = (args.get("file_id") or "").strip() or None
    filename = (args.get("filename") or "").strip() or None
    query = (args.get("query") or "").strip() or None
    offset = args.get("offset")

    thread_key = f"{ctx.channel_id}:{ctx.thread_ts}"
    try:
        docs = await ctx.db.get_thread_documents_async(thread_key)
    except Exception as e:
        return {"ok": False, "error": f"document_lookup_failed: {e}"}

    # F22: resolve against the CURRENT thread first (same-name in both threads → this
    # thread's wins); on a miss, fall back channel-wide (newest match) so a file dropped
    # in another conversation in this channel is still readable. Same channel only — the
    # channel-wide lookup prefix-matches thread_id on channel_id, never crossing channels.
    origin: Optional[str] = None
    doc = _resolve_document(docs or [], file_id, filename)
    channel_docs: Optional[List[Dict[str, Any]]] = None
    if not doc:
        try:
            channel_docs = await ctx.db.get_channel_documents_async(ctx.channel_id)
        except Exception as e:
            return {"ok": False, "error": f"document_lookup_failed: {e}"}
        doc = _resolve_document(channel_docs or [], file_id, filename)
        if doc:
            origin = "shared in another conversation in this channel"
    if not doc:
        # Last resort, and the one that makes "[+1 file: report.pdf]" in another thread's stream
        # item into a real offer: the pinned channel window's own file catalog. There is no
        # `documents` row yet — nobody has read this file — so the summary/page fields simply do
        # not exist, and the fields that matter (the CDN ref and the mimetype) come straight off
        # the FileRef the fetch normalized. Fetch-on-demand into memory, no DB write: cataloguing
        # here would persist a row for a file the model may not even end up using.
        doc = _resolve_canonical_file(getattr(ctx, "canonical_files", None), file_id, filename)
        if doc:
            origin = "shared elsewhere in this channel"
    if not doc:
        known = [d.get("filename") for d in (channel_docs or docs or [])][-5:]
        known += [entry.get("filename")
                  for entry in (getattr(ctx, "canonical_files", None) or {}).values()
                  if entry.get("filename") not in known][:5]
        return {"ok": False, "error": "document_not_found",
                "known_documents": known,
                "hint": "Pass a filename from known_documents, an attachment note, or fetched "
                        "history — any file shared in this channel is reachable, no summary needed."}

    async def _claim_work() -> None:
        # F38: a cache MISS means a real download plus extraction (OCR on a scanned PDF is
        # genuinely slow) — stake the 👀. A cache hit returns instantly and claims nothing.
        turn = getattr(ctx, "turn", None)
        if turn is not None:
            await turn.claim_work(ctx.client, getattr(ctx, "message", None))

    loaded = await load_document_text(ctx.client, doc, on_miss=_claim_work)
    error = loaded.get("error")
    if error == "no_source_ref":
        return {"ok": False, "error": "document_has_no_source_ref",
                "hint": "This document predates on-demand access; only its summary is available."}
    if error == "download_failed":
        return {"ok": False, "error": f"download_failed: {loaded.get('detail')}"}
    if error == "file_deleted":
        return {"ok": False, "error": "file_deleted",
                "hint": "The file is no longer available in Slack; only its summary remains."}
    if error:
        return {"ok": False, "error": "extraction_failed", "detail": loaded.get("detail")}
    # Truthy content is success and an accompanying extractor warning is ignored — the reader
    # would rather hand back partial text than nothing. Only a MASTER is stricter than this.
    text = cast(str, loaded.get("content"))

    total = len(text)
    base = {"ok": True, "filename": doc.get("filename"), "total_chars": total}
    if origin:
        # Channel-wide hit: tell the model the file came from elsewhere so it attributes
        # honestly ("from a file shared in another thread") rather than implying it was here.
        base["origin"] = origin

    if query:
        matches = _query_slices(text, query)
        base["query"] = query
        base["matches"] = matches
        if not matches:
            # F25: a literal-substring miss must not dead-end the call — hand back the
            # document start (the whole document when it fits one slice) so the model
            # can answer or navigate instead of concluding the value isn't in the file.
            base["content"] = text[:SLICE_CHARS]
            base["has_more"] = SLICE_CHARS < total
            if base["has_more"]:
                base["next_offset"] = SLICE_CHARS
                base["note"] = ("No literal match for the query; the document START is "
                                "included — read on with offset or retry a shorter term.")
            else:
                base["note"] = ("No literal match for the query; the FULL document "
                                "content is included above.")
        return base

    start = max(0, int(offset or 0))
    slice_text = text[start:start + SLICE_CHARS]
    base["offset"] = start
    base["content"] = slice_text
    base["has_more"] = (start + SLICE_CHARS) < total
    if base["has_more"]:
        base["next_offset"] = start + SLICE_CHARS
    return base


def register_document_tools(registry: ToolRegistry) -> None:
    """Register read_document (gated on ENABLE_READ_DOCUMENT_TOOL by the caller)."""
    # Longer per-tool timeout than the generic 20s: a scanned-PDF read may download +
    # render + OCR, which the shared cap would abort before the ExtractionCache ever fills.
    registry.register(get_read_document_schema(), execute_read_document,
                      timeout=config.read_document_timeout)
