"""``export_conversation`` — a whole Slack conversation into the sandbox as JSONL.

The history tools return a WINDOW: `fetch_channel_history` is the newest N messages and there is
no cursor to follow. That is right for "what were we just saying" and structurally wrong for the
work that keeps coming up — an incident report, an audit, "summarize everything since March".
Live evidence: asked for a report over two channels, the bot correctly identified that COLLECTION
was the only thing it could not do; the scripts that eventually did it cursor-paginated
`conversations.history` + `conversations.replies` and staged JSONL for analysis.

So the BOT PROCESS collects and the SANDBOX analyses. The model's context never sees the
transcript — it sees a path, a count and a date span — and the file it computes over is complete
rather than sampled.

Three properties are load-bearing:

* **The same authorization gate as every other channel read.** `_authorize_channel_read` (both
  the requester and the bot in the conversation), and a refusal is the one canonical
  `ACCESS_DENIED_MESSAGE` — never a variant that says which of the reasons applied.
* **Nothing touches disk and nothing touches the DB.** Slack stays the only transcript; the
  export exists in memory on the way to the container and nowhere else.
* **Complete or refused, never quietly partial.** Messages are deduped globally by ts, a bounded
  export walks the threads whose replies fall in range even when their root does not, and an
  export larger than one transfer is CHUNKED rather than truncated.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from message_processor.client_contract import HistoryFetchError
from config import config
from logger import setup_logger
from message_processor import file_mount
from slack_client.history_fetch import iter_pages
from slack_client.utilities import ACTOR_REMOTE_LOOKUP_DEFAULT
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.ExportTool")

# A full-channel export is minutes of paging, not seconds, so it cannot live under the registry's
# 20s default. Local tool time runs BETWEEN API rounds — it is not competing with the stream
# timeout — so the bound is generous rather than clever.
EXPORT_TIMEOUT_S = 600.0

# Slack's own maximum for a history/replies page, and what the collection scripts used.
_PAGE_LIMIT = 200

# Between pages. The pacing that ran two channels end to end without a single 429; the retry
# ladder in `fetch_page` honors Retry-After when one arrives anyway.
_PAGE_PAUSE_S = 1.2

# One transfer's ceiling is `artifact_max_mb` (the same bound `mount_file` moves bytes under),
# so a bigger export becomes several parts.
_PART_TEMPLATE = "export-part-{:03d}.jsonl"


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


def _max_transfer_bytes() -> int:
    return int(config.artifact_max_mb) * 1024 * 1024


def _web(client: Any, name: str) -> Optional[Callable[..., Any]]:
    """The Slack web method off a client that may be the bot or a bare web client (tests)."""
    app = getattr(client, "app", None)
    web = getattr(app, "client", None) if app is not None else None
    method = getattr(web, name, None)
    if callable(method):
        return method
    method = getattr(client, name, None)
    return method if callable(method) else None


def _ts_arg(value: Any) -> Optional[str]:
    """A Slack ts bound as the string Slack wants, or None. Never raises on model input."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if isinstance(value, str) and value.strip():
        try:
            return f"{float(value.strip()):.6f}"
        except ValueError:
            return None
    return None


def _ts_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_export_conversation_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "export_conversation",
        "description": (
            "Export a Slack conversation's FULL history into the code sandbox as a JSONL file "
            "(one message per line: ts, thread_ts, author, sender type, text, reactions, file "
            "names) so your code can read every message rather than a recent window. Returns the "
            "/mnt/data path(s), how many messages and threads it holds, and the date span.\n\n"
            "Use it when a task needs complete coverage of a conversation — an incident report, "
            "an audit, activity over a period, counting or ranking anything across a channel. "
            "Analyse the file with code; do not page history into your context and do not answer "
            "from a search sample. Same access rule as the history tools: only a conversation "
            "both you and the person who asked are in.\n\n"
            "A big channel takes MINUTES of paging. When the export is feeding a report or a "
            "built file, prefer start_background_job with mode 'build' and let the job export "
            "there, so nobody watches a blank reply while it collects.\n\n"
            "The export is an INGREDIENT: it is not posted to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": ("Slack channel ID. Omit for the CURRENT conversation; never "
                                    "guess an ID you have not seen."),
                },
                "oldest": {
                    "type": "string",
                    "description": ("Only messages at or after this Slack ts (e.g. "
                                    "'1740787200.000000'). Omit for everything Slack still "
                                    "holds."),
                },
                "latest": {
                    "type": "string",
                    "description": "Only messages at or before this Slack ts. Omit for up to now.",
                },
                "include_threads": {
                    "type": "boolean",
                    "description": ("Fetch the replies under every threaded message (default "
                                    "true). False exports top-level messages only."),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    }


async def execute_export_conversation(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Authorize → page → serialize → stage. Never raises: every failure is a result."""
    try:
        return await _export(ctx, args)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — an executor failure is a tool result, not a turn's end
        logger.error(f"export_conversation failed: {e}", exc_info=True)
        return _err("export_failed", "Could not export that conversation.")


async def _export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from slack_client.history_tool import ACCESS_DENIED_MESSAGE

    raw_channel = args.get("channel_id")
    channel_id = (raw_channel.strip() if isinstance(raw_channel, str) and raw_channel.strip()
                  else getattr(ctx, "channel_id", None))
    oldest = _ts_arg(args.get("oldest"))
    latest = _ts_arg(args.get("latest"))
    raw_threads = args.get("include_threads", True)
    include_threads = (raw_threads if isinstance(raw_threads, bool)
                       else str(raw_threads).strip().lower() not in ("false", "0", "no"))

    client = getattr(ctx, "client", None)
    processor = getattr(ctx, "processor", None)
    if client is None or processor is None:
        return _err("unavailable", "Exporting isn't available right now.")

    # THE gate — the canonical one, so this tool cannot become the loose door into a conversation
    # the requester may not read. A DENY and a REDIRECT are byte-identical to the model on
    # purpose (see history_tool): a refusal that varied would answer "does this channel exist?".
    authorize = getattr(client, "_authorize_channel_read", None)
    if not callable(authorize):
        return _err("unavailable", "Exporting isn't available right now.")
    verdict, reason = await authorize(channel_id, ctx)
    if verdict != "ALLOW":
        logger.warning(f"export_conversation {verdict.lower()} for "
                       f"channel={channel_id or '-'} reason={reason}")
        return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

    history = _web(client, "conversations_history")
    replies = _web(client, "conversations_replies")
    if history is None or (include_threads and replies is None):
        return _err("unavailable", "Slack history isn't reachable from here right now.")

    container_id = await ctx.ensure_sandbox()
    if not container_id:
        return _err("sandbox_unavailable",
                    "There is no code sandbox to export into on this turn.")
    # F15: a container that idle-expired earlier this turn is a dead drop — the export would be
    # invisible to the code that was going to read it.
    if ctx.container_recycled():
        return _err("container_recycled",
                    "The code sandbox was recycled mid-turn, so the export can't be placed in "
                    "it. Ask again and it will be set up fresh.")

    # Past every rejection, and a walk of minutes is about to start: the honest moment for the 👀.
    turn = getattr(ctx, "turn", None)
    if turn is not None:
        try:
            await turn.claim_work(client, getattr(ctx, "message", None))
        except Exception:  # noqa: BLE001 — presentation never breaks the export
            pass

    started = time.monotonic()
    try:
        messages, thread_roots = await _collect(
            history, replies, channel_id=str(channel_id), oldest=oldest, latest=latest,
            include_threads=include_threads)
    except HistoryFetchError as e:
        logger.warning(f"export_conversation paging failed for {channel_id}: {e}")
        return _err("history_unavailable",
                    "Slack stopped returning history part-way through, so the export would have "
                    "been incomplete. Nothing was staged — try again.")

    if not messages:
        return _err("empty_export",
                    "That conversation has no messages in the range asked for.")

    names = await _resolve_authors(client, messages)
    lines = [_serialize(client, m, names, str(channel_id)) for m in messages]
    parts = _chunk(lines, _max_transfer_bytes())

    staged: List[Dict[str, Any]] = []
    for index, blob in enumerate(parts, 1):
        filename = _PART_TEMPLATE.format(index)
        record = await file_mount.stage_bytes(
            ctx, container_id, filename, blob,
            source_id=f"export:{channel_id}:{index}:{int(started)}")
        if record is None:
            return _err("stage_failed",
                        f"Collected {len(messages)} messages but could not place "
                        f"{filename} in the sandbox.")
        staged.append({"path": record["path"], "filename": record["filename"],
                       "size_bytes": len(blob), "gzipped": bool(record.get("gzipped"))})

    total_bytes = sum(p["size_bytes"] for p in staged)
    span_oldest = messages[0].get("ts")
    span_latest = messages[-1].get("ts")
    logger.info(f"export_conversation staged {len(messages)} messages "
                f"({len(thread_roots)} threads, {total_bytes} bytes, {len(staged)} part(s)) "
                f"from {channel_id} in {time.monotonic() - started:.1f}s")

    note = (
        "Read the file(s) with code — one JSON object per line, oldest first, fields: ts, "
        "thread_ts, user, sender ('human' | 'other_bot' | 'self'), text, and reactions/files "
        "where the message has them. Everything in the range is here; it has NOT been posted "
        "to the user.")
    if any(p["gzipped"] for p in staged):
        note += (" A part marked gzipped is GZIP-COMPRESSED in the container — open those with "
                 "the `gzip` module rather than as plain text.")
    return {
        "ok": True,
        "channel": channel_id,
        "paths": [p["path"] for p in staged],
        "files": staged,
        "format": "jsonl",
        "message_count": len(messages),
        "thread_count": len(thread_roots),
        "oldest_ts": span_oldest,
        "latest_ts": span_latest,
        "size_bytes": total_bytes,
        "message": note,
    }


async def _collect(history: Callable[..., Any], replies: Optional[Callable[..., Any]], *,
                   channel_id: str, oldest: Optional[str], latest: Optional[str],
                   include_threads: bool) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Every message in the window, deduped by ts and sorted oldest-first.

    Two things make this complete rather than nearly complete. Messages are deduped GLOBALLY:
    `conversations.replies` re-includes the thread root, which history already returned, and a
    transcript that repeats a line reads as somebody repeating themselves. And when `oldest` is
    set, history is paged PAST it rather than bounded by it: a reply from last week can hang off
    a root from last year, and a bounded history call cannot see that root at all. Any root whose
    `latest_reply` lands in range is walked; the window predicate then keeps only the replies
    that belong.
    """
    seen: Set[str] = set()
    collected: List[Dict[str, Any]] = []
    roots: List[str] = []
    thread_roots: Set[str] = set()
    low = _ts_float(oldest)
    high = _ts_float(latest)

    def _in_window(ts: Any) -> bool:
        value = _ts_float(ts)
        if value is None:
            return False
        if low is not None and value < low:
            return False
        return not (high is not None and value > high)

    def _keep(msg: Dict[str, Any]) -> None:
        ts = msg.get("ts")
        if not isinstance(ts, str) or ts in seen or not _in_window(ts):
            return
        seen.add(ts)
        collected.append(msg)

    # `oldest` is withheld from the API only when the thread walk needs the older roots; with
    # include_threads off there is nothing to rescue and the cheap bounded walk is correct.
    api_oldest = None if (oldest and include_threads) else oldest
    pages = 0
    async for page in iter_pages(history, channel_id=channel_id, oldest=api_oldest,
                                 latest=latest, inclusive=True, limit=_PAGE_LIMIT,
                                 label="export history"):
        pages += 1
        for msg in page:
            _keep(msg)
            if include_threads and msg.get("reply_count") and _thread_touches(msg, low):
                ts = msg.get("ts")
                if isinstance(ts, str):
                    roots.append(ts)
        logger.debug(f"export_conversation: {channel_id} history page {pages} "
                     f"({len(collected)} kept, {len(roots)} threads queued)")
        await asyncio.sleep(_PAGE_PAUSE_S)

    if include_threads and replies is not None:
        for root_ts in roots:
            async for page in iter_pages(replies, channel_id=channel_id, oldest=oldest,
                                         latest=latest, inclusive=True, limit=_PAGE_LIMIT,
                                         extra_params={"ts": root_ts},
                                         label="export replies"):
                for msg in page:
                    before = len(collected)
                    _keep(msg)
                    if len(collected) > before:
                        thread_roots.add(root_ts)
                await asyncio.sleep(_PAGE_PAUSE_S)

    collected.sort(key=lambda m: _ts_float(m.get("ts")) or 0.0)
    return collected, thread_roots


def _thread_touches(msg: Dict[str, Any], low: Optional[float]) -> bool:
    """Does this root's thread reach into the window? Unbounded exports take every thread."""
    if low is None:
        return True
    newest = _ts_float(msg.get("latest_reply")) or _ts_float(msg.get("ts"))
    return newest is not None and newest >= low


async def _resolve_authors(client: Any, messages: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Display names for the human authors, read-only and in the resolver's own batch size.

    Reading history must not create user rows or bump `last_seen`, which is exactly what
    `resolve_usernames` guarantees. Its remote budget is per CALL, so an export with more
    speakers than one batch is resolved in successive batches (each hit is cached in the client
    for the ones after it) rather than leaving the overflow as raw ids.
    """
    resolver = getattr(client, "resolve_usernames", None)
    if not callable(resolver):
        return {}
    api_client = getattr(getattr(client, "app", None), "client", None)
    ids = list(dict.fromkeys(
        m.get("user") for m in messages if m.get("user") and not m.get("bot_id")))
    names: Dict[str, str] = {}
    for start in range(0, len(ids), ACTOR_REMOTE_LOOKUP_DEFAULT):
        batch = ids[start:start + ACTOR_REMOTE_LOOKUP_DEFAULT]
        try:
            names.update(await resolver(batch, api_client) or {})
        except Exception as e:  # noqa: BLE001 — an unresolved id stays raw, which is honest
            logger.debug(f"export_conversation: name resolution failed: {e}")
            break
    return names


def _serialize(client: Any, msg: Dict[str, Any], names: Dict[str, str],
               channel_id: str) -> bytes:
    """One message as one JSONL line."""
    user = msg.get("user")
    author = user or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown")
    if user and not msg.get("bot_id"):
        author = names.get(user, author)
    try:
        sender = client.classify_sender(msg)
    except Exception:  # noqa: BLE001 — identity not wired (a test double); the text still exports
        sender = "human"
    entry: Dict[str, Any] = {
        "ts": msg.get("ts"),
        "thread_ts": msg.get("thread_ts"),
        "user": author,
        "sender": sender,
        "text": _text(client, msg, channel_id),
    }
    reactions = msg.get("reactions")
    if isinstance(reactions, list):
        entry["reactions"] = [
            {"emoji": r.get("name"), "count": r.get("count") or len(r.get("users") or []),
             "users": r.get("users") or []}
            for r in reactions if isinstance(r, dict) and r.get("name")
        ]
    files = msg.get("files")
    if isinstance(files, list):
        named = [{"name": f.get("name"), "mimetype": f.get("mimetype")}
                 for f in files if isinstance(f, dict) and f.get("name")]
        if named:
            entry["files"] = named
    return (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _text(client: Any, msg: Dict[str, Any], channel_id: str) -> str:
    """The message's text INCLUDING what Slack delivered outside it (table blocks, unfurls,
    webhook attachment fields) — the same reader the history tool uses, so an exported message
    cannot say less than the same message read by tool."""
    reader = getattr(client, "_text_with_supplementary", None)
    if callable(reader):
        try:
            return reader(msg, channel_id)
        except Exception:  # noqa: BLE001
            pass
    return msg.get("text") or ""


def _chunk(lines: Sequence[bytes], max_bytes: int) -> List[bytes]:
    """Whole lines packed into transfer-sized parts. NEVER truncates: a single line longer than
    the ceiling is its own part, because dropping a message is worse than a big transfer."""
    parts: List[bytes] = []
    current = bytearray()
    for line in lines:
        if current and len(current) + len(line) > max_bytes:
            parts.append(bytes(current))
            current = bytearray()
        current.extend(line)
    if current:
        parts.append(bytes(current))
    return parts


def register_export_tool(registry: ToolRegistry) -> None:
    """Register export_conversation on both surfaces, gated on the sandbox being on — with
    nowhere to put the file there is no tool. The explicit timeout is the point: a full channel
    is minutes of paging and the registry's 20s default would abort it mid-walk."""
    registry.register(get_export_conversation_schema(), execute_export_conversation,
                      enabled=file_mount.sandbox_enabled,
                      channel_enabled=file_mount.sandbox_enabled,
                      timeout=EXPORT_TIMEOUT_S)
