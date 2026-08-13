"""``search_stored_knowledge`` — what this conversation ALREADY worked out, looked up by keyword.

Every document read and every image analysed in a channel leaves a derived row behind: a summary
in `documents`, a description in `images`. Until now nothing could ASK those rows a question. The
model could only meet them by accident — if the pinned window happened to render the right
message, or if someone re-uploaded the file. "Which screenshot showed the 500 error?" and "that
pricing sheet from March" therefore dead-ended in a channel that demonstrably held the answer.

Three properties make the results honest rather than merely present:

* **It searches DERIVED text, never bodies.** `documents` holds a summary + metadata + the Slack
  ref and never the file's content (CLAUDE.md pitfall 6a), so a miss here does NOT prove the
  document never said it. The description says so in as many words, and `read_document` is the
  tool that re-derives real content.
* **Handles are actionable or they are labelled.** A document hit carries the Slack file id and
  filename `read_document` genuinely resolves. An image hit carries its analysis and a permalink
  and NO viewing id: `view_image` resolves only ids from this turn's own catalog, so a synthesised
  `img_*` would be a handle that always breaks.
* **The canonical read gate runs first.** Not "the requester is in this channel by construction" —
  that holds for a live message event and fails for the synthetic, replayed and detached contexts
  the registry also serves (tool_registry.py:164).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from config import config
from logger import setup_logger
from message_processor.document_tools import QUERY_WINDOW_CHARS
from tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.KnowledgeTools")

# The search tool's convention, so two keyword searches in one turn cannot disagree about what
# "limit" means (slack_client/search_tool.py:585 / :629).
DEFAULT_LIMIT = 10
MAX_LIMIT = 20


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


def _clamp_limit(limit: Any) -> int:
    """1-20, default 10 — `_clamp_search_limit`'s behaviour, including its coercion of junk."""
    try:
        return max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _thread_ts_of(thread_id: Any, channel_id: str) -> Optional[str]:
    """The thread half of a ``channel:thread`` key.

    Thread keys contain colons (CLAUDE.md pitfall 3), so this strips the known channel prefix
    rather than splitting on the delimiter and hoping. A row whose key does not carry this
    channel's prefix returns None — it cannot happen through the channel-scoped queries, and
    silently mislabelling it if it ever did would be worse than an absent field.
    """
    if not isinstance(thread_id, str):
        return None
    prefix = f"{channel_id}:"
    if not thread_id.startswith(prefix):
        return None
    rest = thread_id[len(prefix):].strip()
    return rest or None


def _snippet(text: Optional[str], query: str) -> Optional[str]:
    """A context window around the first case-insensitive hit, or the head of the text.

    The window is `read_document`'s own QUERY_WINDOW_CHARS, so a snippet from a search and a
    slice from a read are the same size of evidence. A None return means there was nothing to
    quote — the match was in the filename, and inventing a quote from a summary that does not
    contain the term would misrepresent why the row matched.
    """
    if not text:
        return None
    pos = text.lower().find(query.lower())
    if pos == -1:
        head = text[:QUERY_WINDOW_CHARS].strip()
        return (head + "…") if len(text) > QUERY_WINDOW_CHARS else (head or None)
    start = max(0, pos - QUERY_WINDOW_CHARS)
    end = min(len(text), pos + len(query) + QUERY_WINDOW_CHARS)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _contains(text: Optional[str], query: str) -> bool:
    return bool(text) and query.lower() in str(text).lower()


def _document_hit(row: Dict[str, Any], query: str, channel_id: str) -> Dict[str, Any]:
    filename = row.get("filename") or "document"
    summary = row.get("summary")
    # Which column earned the row, so the model can weigh a real summary match against a
    # filename that merely happens to contain the word.
    matched_summary = _contains(summary, query)
    hit: Dict[str, Any] = {
        "kind": "document",
        "filename": filename,
        "matched_in": "summary" if matched_summary else "filename",
        "shared_at": row.get("created_at"),
    }
    file_id = row.get("file_id")
    if file_id:
        # THE actionable handle: read_document resolves a file_id (exactly, surviving same-name
        # collisions) or a filename, channel-wide (document_tools.py:319).
        hit["file_id"] = file_id
    thread_ts = _thread_ts_of(row.get("thread_id"), channel_id)
    if thread_ts:
        hit["thread_ts"] = thread_ts
    if row.get("message_ts"):
        hit["message_ts"] = row["message_ts"]
    # QUOTE THE COLUMN THAT MATCHED, as the image hit does: on a filename-only match the summary
    # is about something else entirely ("pricing.xlsx" holding a holiday calendar), and its head
    # shipped under `summary_snippet` reads as the evidence for a match it had no part in.
    snippet = _snippet(summary, query) if matched_summary else None
    if snippet:
        hit["summary_snippet"] = snippet
    elif summary and str(summary).strip():
        hit["note"] = ("Matched on the filename; the stored summary does not mention the query. "
                       "Use read_document to see what this file actually says.")
    else:
        hit["note"] = "Matched on the filename; no summary was stored for this file."
    return hit


def _image_hit(row: Dict[str, Any], query: str, channel_id: str) -> Dict[str, Any]:
    # QUOTE THE COLUMN THAT MATCHED. The query hit `analysis` OR `original_analysis` (an edited
    # image's pre-edit description), so preferring `analysis` unconditionally would hand back a
    # snippet with no trace of the search term whenever the pre-edit text is what matched —
    # evidence for a hit the evidence does not support. Current analysis still wins a tie.
    analysis = row.get("analysis")
    original = row.get("original_analysis")
    if not _contains(analysis, query) and _contains(original, query):
        analysis = original
        matched_in = "pre-edit description"
    else:
        matched_in = "description"
    hit: Dict[str, Any] = {
        "kind": "image",
        "image_kind": row.get("image_type") or "image",
        "matched_in": matched_in,
        "shared_at": row.get("created_at"),
    }
    thread_ts = _thread_ts_of(row.get("thread_id"), channel_id)
    if thread_ts:
        hit["thread_ts"] = thread_ts
    if row.get("message_ts"):
        hit["message_ts"] = row["message_ts"]
    snippet = _snippet(analysis, query)
    if snippet:
        hit["analysis_snippet"] = snippet
    return hit


async def _permalinks(client: Any, channel_id: str,
                      timestamps: List[str]) -> Dict[str, str]:
    """`chat.getPermalink` for the image hits, best effort — the search tool's own idiom
    (search_tool.py:1355): bounded by the result limit and the same concurrency knob, and any
    failure is simply a missing link rather than a failed search."""
    out: Dict[str, str] = {}
    if not timestamps:
        return out
    web = getattr(getattr(client, "app", None), "client", None)
    method = getattr(web, "chat_getPermalink", None)
    if not callable(method):
        return out
    semaphore = asyncio.Semaphore(max(1, int(config.search_reply_fetch_concurrency)))

    async def _one(ts: str) -> None:
        async with semaphore:
            try:
                resp = await method(channel=channel_id, message_ts=ts)
            except Exception:  # noqa: BLE001 — a link is an enrichment, never a result
                return
            link = resp.get("permalink") if resp is not None else None
            if isinstance(link, str) and link:
                out[ts] = link

    await asyncio.gather(*(_one(ts) for ts in dict.fromkeys(timestamps)))
    return out


def get_search_stored_knowledge_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "search_stored_knowledge",
        "description": (
            "Keyword-search what you have ALREADY read or looked at in this conversation: the "
            "SUMMARIES and filenames of documents shared here, and the DESCRIPTIONS of images "
            "and screenshots shared here. Covers the whole current channel (in a DM, that DM) "
            "across every thread in it, newest first — including files from conversations that "
            "are no longer in front of you.\n\n"
            "Reach for it whenever the history suggests the answer already exists rather than "
            "needing to be worked out again: 'which screenshot showed the 500 error?', 'that "
            "pricing sheet from March', 'didn't someone post a diagram of this?'. Use it on your "
            "own judgment, not only when asked — and when someone does ask outright, this is the "
            "tool for it.\n\n"
            "IMPORTANT — this searches DERIVED text, not file contents. Document bodies are "
            "never stored; only a summary is. So a miss does NOT mean the document lacks the "
            "thing — it means no summary mentioned it. When you have reason to think a file "
            "holds the answer, open it with read_document (which re-reads the real file) instead "
            "of concluding from an empty result here.\n\n"
            "Document hits return a file_id and filename you can pass straight to read_document — "
            "that one tool, not any other file tool: the sandbox mounts work off handles offered "
            "on the turn itself, so what you get back here will not open one. Image hits are "
            "INFORMATIONAL: they give you the stored description "
            "and a link to the message, and there is no id to view the picture with, so answer "
            "from the description or point the person at the link."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The words to match, as they would appear in a summary or an image "
                        "description. Matching is literal substring, case-insensitive — prefer a "
                        "short distinctive term ('500 error', 'pricing') over a whole question."
                    ),
                },
                "limit": {"type": "integer", "description": "Max results (1-20, default 10)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


async def execute_search_stored_knowledge(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Authorize → query both stores → shape the hits. Never raises: a failure is a result."""
    try:
        return await _search(ctx, args)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — an executor failure is a tool result, not a turn's end
        logger.error(f"search_stored_knowledge failed: {e}", exc_info=True)
        return _err("search_failed", "Could not search what's been shared here.")


async def _search(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from slack_client.history_tool import ACCESS_DENIED_MESSAGE

    raw_query = args.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
    if not query:
        return _err("missing_query", "Say what to search the shared files and images for.")
    limit = _clamp_limit(args.get("limit"))

    channel_id = getattr(ctx, "channel_id", None)
    db = getattr(ctx, "db", None)
    client = getattr(ctx, "client", None)
    if not isinstance(channel_id, str) or not channel_id or db is None or client is None:
        return _err("unavailable", "Searching what's been shared isn't available right now.")

    # THE gate, before any stored text is handed back. `ctx.channel_id` is not proof of anything
    # on a synthetic, replayed or detached context, and these rows are the derived contents of
    # a conversation the requester may not be in.
    authorize = getattr(client, "_authorize_channel_read", None)
    if not callable(authorize):
        return _err("unavailable", "Searching what's been shared isn't available right now.")
    verdict, reason = await authorize(channel_id, ctx)
    if verdict != "ALLOW":
        logger.warning(f"search_stored_knowledge {verdict.lower()} for "
                       f"channel={channel_id} reason={reason}")
        return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

    docs, images, unavailable = await _gather_rows(db, channel_id, query, limit)
    if unavailable == "both":
        return _err("lookup_failed", "Couldn't read what's been shared here just now.")

    hits: List[Dict[str, Any]] = [_document_hit(row, query, channel_id) for row in docs]
    image_hits = [_image_hit(row, query, channel_id) for row in images]
    links = await _permalinks(client, channel_id,
                             [h["message_ts"] for h in image_hits if h.get("message_ts")])
    for hit in image_hits:
        link = links.get(hit.get("message_ts") or "")
        if link:
            hit["permalink"] = link
    hits.extend(image_hits)
    # Newest first ACROSS both stores — the two queries each sort their own rows, and a merged
    # list that ran documents-then-images would call an old file the most recent thing here.
    hits.sort(key=lambda h: str(h.get("shared_at") or ""), reverse=True)
    hits = hits[:limit]

    result: Dict[str, Any] = {
        "ok": True,
        "query": query,
        "results": hits,
        "count": len(hits),
    }
    if unavailable:
        # Never a silent partial: a half-searched store that reported "0 results" would read as
        # proof of absence.
        result["incomplete"] = (
            f"The stored {unavailable} could not be searched this time, so this result covers "
            f"only the other kind. Don't treat it as proof nothing exists.")
    if not hits:
        result["note"] = (
            "Nothing shared in this channel has a stored summary or description matching that. "
            "Document contents are not stored, so this does not prove no file says it — if you "
            "know which file to look in, read_document opens the real thing. A different or "
            "shorter search term may also match.")
    else:
        result["how_to_use"] = (
            "Document hits: pass file_id or filename to read_document for the real content. "
            "Image hits: the description and permalink are all there is — there is no id that "
            "will open the picture, so answer from the description or share the link.")
    return result


async def _gather_rows(db: Any, channel_id: str, query: str,
                       limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                            Optional[str]]:
    """Both stores concurrently. Returns (documents, images, what-was-unavailable) — a store that
    failed is NAMED rather than folded into an empty result."""
    # Annotated rather than tuple-unpacked: `gather(..., return_exceptions=True)` gives the
    # checker no per-element type to resolve, and each element is genuinely rows-or-exception.
    gathered: List[Any] = list(await asyncio.gather(
        db.search_channel_documents_async(channel_id, query, limit),
        db.search_channel_image_analyses_async(channel_id, query, limit),
        return_exceptions=True,
    ))
    doc_result, image_result = gathered[0], gathered[1]
    docs: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    failed: List[str] = []
    if isinstance(doc_result, BaseException):
        logger.warning(f"search_stored_knowledge document lookup failed: {doc_result}")
        failed.append("documents")
    else:
        docs = list(doc_result or [])
    if isinstance(image_result, BaseException):
        logger.warning(f"search_stored_knowledge image lookup failed: {image_result}")
        failed.append("image descriptions")
    else:
        images = list(image_result or [])
    if len(failed) == 2:
        return [], [], "both"
    return docs, images, (failed[0] if failed else None)


def register_knowledge_tools(registry: ToolRegistry) -> None:
    """Register search_stored_knowledge on both surfaces. One static schema — what it can reach
    is the channel, not anything that varies per turn — and no feature flag: reading back what
    the bot already derived in this very conversation is plain recall, not a capability."""
    registry.register(get_search_stored_knowledge_schema(), execute_search_stored_knowledge)
