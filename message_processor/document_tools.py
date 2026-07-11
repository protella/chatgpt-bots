"""Model-invoked document access (Phase D2 of the channel-teammate redesign).

Documents are never at rest (CLAUDE.md pitfall 6a): the DB row holds
summary + metadata + the Slack CDN ref, and this tool re-derives the full
text ON DEMAND — authenticated download into memory, BytesIO extraction,
return the requested slice. A process-lifetime bounded LRU of extracted
text (never persisted, gone on restart — same lifecycle class as
ChannelPulse) makes iterating on one document cheap.

Deleted Slack file ⇒ download fails ⇒ ``{"ok": False, "error": "file_deleted"}``
— a privacy feature: deleting a file in Slack genuinely removes its content
from the bot's reach; only the labeled summary row remains.

Executors never raise: every failure is an ``{"ok": False, ...}`` result.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

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

    def __len__(self) -> int:
        return len(self._entries)


_extraction_cache = ExtractionCache(config.doc_extraction_cache_size)


def get_read_document_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "read_document",
        "description": (
            "Read the full content of a document shared in this channel (current conversation "
            "checked first). Document "
            "summaries in context are SUMMARIES — use this tool whenever you need specific "
            "figures, quotes, table values, or sections a summary doesn't literally contain. "
            "Provide query to search inside the document, or offset to read sequentially."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Slack file id from the document summary header (preferred).",
                },
                "filename": {
                    "type": "string",
                    "description": "Document filename (used when file_id is unknown).",
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
        known = [d.get("filename") for d in (channel_docs or docs or [])][-5:]
        return {"ok": False, "error": "document_not_found",
                "known_documents": known,
                "hint": "Use the filename or file_id from the document summary in context."}

    doc_file_id = doc.get("file_id")
    url_private = doc.get("url_private")
    if not url_private and not doc_file_id:
        return {"ok": False, "error": "document_has_no_source_ref",
                "hint": "This document predates on-demand access; only its summary is available."}

    cache_key = doc_file_id or url_private
    text = _extraction_cache.get(cache_key)
    if text is None:
        try:
            data = await ctx.client.download_file(url_private, doc_file_id)
        except Exception as e:
            return {"ok": False, "error": f"download_failed: {e}"}
        if not data:
            # Deleted-in-Slack is indistinguishable from never-there — by design,
            # deletion removes the content from the bot's reach.
            return {"ok": False, "error": "file_deleted",
                    "hint": "The file is no longer available in Slack; only its summary remains."}
        extracted = await _document_handler.safe_extract_content_async(
            data, doc.get("mime_type") or "application/octet-stream",
            doc.get("filename") or "document",
            ocr_images=False)  # tool returns text slices; page images are useless here
        text = (extracted or {}).get("content")
        if not text:
            return {"ok": False, "error": "extraction_failed",
                    "detail": (extracted or {}).get("error", "no content extracted")}
        _extraction_cache.put(cache_key, text)

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
            base["note"] = ("No literal match; try a shorter or different term, "
                            "or read sequentially with offset.")
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
    registry.register(get_read_document_schema(), execute_read_document)
