"""``mount_file`` — put a thread file's real bytes into the code-interpreter sandbox (F35).

The model can already SEE what the user shared (images ride the turn as ``input_image``,
documents are text-extracted into the prompt) but until now it could not USE it: the only
bytes that ever reached ``/mnt/data`` were images the bot generated itself. So "turn these
four screenshots and the thread into a PDF" was structurally impossible, and "analyse this
50k-row CSV" degraded into the model retyping the data as a Python literal.

This tool is the missing bridge, and it is deliberately LAZY: mounting costs a download and a
container write, and most attachments are only ever read or looked at. The model asks for the
bytes when it actually needs to compute on them.

Two properties are load-bearing:

* **Bytes never touch disk** (CLAUDE.md pitfall 6a). Slack CDN → memory → container, and the
  BytesIO is dropped on the way out. We persist the mount's *path*, never its content.
* **A mounted file is an INGREDIENT, not a deliverable.** ``containers.files.create`` marks
  uploads ``source="user"``, and the artifact publisher only ever considers ``"assistant"``
  files — so a user's own spreadsheet cannot be posted back at them. We also record each
  mount's digest so that a model which merely *copies* an input to a new name (making an
  assistant-owned, byte-identical twin) still cannot round-trip it into the channel.

**The upload endpoint judges CONTENT, not filenames** (measured live 2026-08-12). Raw SVG bytes
are 400-refused — "You uploaded an invalid file", no error code — under ``logo.svg``, ``.xml``,
``.dat``, ``.bin`` and ``.svg.txt`` alike, while the SAME bytes gzip-compressed or base64-encoded
go straight in, as do plain text, JSON, PNG and HTML. No rename talks it round, so ``stage_bytes``
does not try to: it offers the real bytes, and on a 400 retries ONCE through gzip, which the
sandbox undoes in one line. We keep no allowlist and sniff nothing on our side — a format that
fails both ways fails honestly.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from config import config
from logger import setup_logger
from message_processor import thread_files
from tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.FileMount")

# Stashed on the per-request config so the schema FACTORY can see them (a factory only ever
# receives thread_config). Mirrors image_tools' CI_CONTAINER_KEY / CATALOG_KEY.
FILES_KEY = "_thread_files"

# What is already in which container, keyed by (container_id, file_id) — process-lifetime and
# bounded, the same lifecycle class as the artifact publisher's LRU. It must be keyed by
# CONTAINER, not by thread: that is precisely what makes the "come back after lunch" case work.
# Within a live container, round two's mount of the same CSV is a no-op. When the container has
# since expired, the thread's next turn gets a NEW id, every key misses, and the assets are
# re-mounted into the fresh sandbox — which is the whole recovery story, and it falls out for
# free rather than needing a rebuild protocol. A process restart just costs one re-upload.
_MOUNTS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_MOUNTS_MAX = 256

# Reuse the artifact ceiling: it is the same question in the other direction — how large a
# file are we willing to move between Slack and a container in one hop.
def _max_bytes() -> int:
    return config.artifact_max_mb * 1024 * 1024


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


def get_mount_file_schema(thread_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Offered whenever there is something to mount.

    Returning ``None`` hides the tool, and with no files there is nothing to name. It used to
    hide on an ``auto`` container too — no addressable id, nowhere to push bytes. W3 made every
    turn start on ``auto``, so that test would have hidden the tool on the first turn of every
    conversation, which is exactly the turn a user drops a spreadsheet into. The executor mints
    an addressable container on demand instead (``ToolContext.ensure_sandbox``), and the tool
    loop names it in the next round's declaration.
    """
    cfg = thread_config or {}
    entries = cfg.get(FILES_KEY) or []
    ids = thread_files.valid_ids(entries)
    if not ids:
        return None

    return {
        "type": "function",
        "name": "mount_file",
        "description": (
            "Copy a file shared in this thread into the code-interpreter sandbox so your code "
            "can open its REAL bytes. Returns the /mnt/data path.\n\n"
            "Use this before code_interpreter whenever you need the actual file — analysing a "
            "spreadsheet or CSV with pandas, embedding a user's image into a deck or PDF, "
            "editing an existing Office document, OCR, format conversion, or bundling files "
            "into an archive. You do NOT need it merely to read, summarise, or look at "
            "something: you can already see images and document text directly.\n\n"
            "Never retype a file's contents into your code as a literal — mount it and read it. "
            "Mounting is idempotent: calling it twice returns the same path.\n\n"
            "A mounted file is an INGREDIENT. It is not posted to the user, and copying it "
            "unchanged to a new name will not deliver it either — only files you genuinely "
            "create are published.\n\n"
            "Files available in this thread:\n" + thread_files.catalog_lines(entries)
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "enum": ids,
                    "description": "Which thread file to mount.",
                },
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    }


def get_mount_file_schema_static(thread_config: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, Any]:
    """Channel-surface mount_file: no id enum, no catalog text, no container gating.

    ``thread_config`` is accepted and IGNORED so the registry can call it like a factory; the
    output never varies. Both facts the dynamic factory gated on — is there a sandbox, is there
    anything to mount — are per-thread, so the executor answers them instead
    (``sandbox_unavailable`` / ``unknown_file_id`` with the valid ids).
    """
    return {
        "type": "function",
        "name": "mount_file",
        "description": (
            "Copy a file shared in this thread into the code-interpreter sandbox so your code "
            "can open its REAL bytes. Returns the /mnt/data path.\n\n"
            "Use this before code_interpreter whenever you need the actual file — analysing a "
            "spreadsheet or CSV with pandas, embedding a user's image into a deck or PDF, "
            "editing an existing Office document, OCR, format conversion, or bundling files "
            "into an archive. You do NOT need it merely to read, summarise, or look at "
            "something: you can already see images and document text directly.\n\n"
            "Never retype a file's contents into your code as a literal — mount it and read it. "
            "Mounting is idempotent: calling it twice returns the same path.\n\n"
            "A mounted file is an INGREDIENT. It is not posted to the user, and copying it "
            "unchanged to a new name will not deliver it either — only files you genuinely "
            "create are published.\n\n"
            "Ids come from the file catalog in this turn's evidence; an id that is not listed "
            "there does not resolve, and mounting needs a live code sandbox."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Which thread file to mount, by id from the catalog.",
                },
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    }


def _mount_key(container_id: str, file_id: str) -> str:
    return f"{container_id}|{file_id}"


def _remember_mount(key: str, record: Dict[str, Any]) -> None:
    _MOUNTS[key] = record
    _MOUNTS.move_to_end(key)
    while len(_MOUNTS) > _MOUNTS_MAX:
        _MOUNTS.popitem(last=False)


def _recall_mount(key: str) -> Optional[Dict[str, Any]]:
    record = _MOUNTS.get(key)
    if record is not None:
        _MOUNTS.move_to_end(key)
    return record


async def _download(client, entry: Dict[str, Any]) -> Optional[bytes]:
    """Authenticated fetch from Slack, into memory only.

    A deleted Slack file is indistinguishable from one that was never there — by design:
    deleting a file in Slack genuinely removes its content from the bot's reach.
    """
    try:
        return await client.download_file(entry.get("url"), entry.get("slack_file_id"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Mount download failed for {entry.get('file_id')}: {e}")
        return None


async def execute_mount_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve → download → push into the container → hand back the path."""
    file_id = (args.get("file_id") or "").strip()
    if not file_id:
        return _err("missing_file_id", "A file_id is required.")

    processor = getattr(ctx, "processor", None)
    client = getattr(ctx, "client", None)
    if processor is None or client is None:
        return _err("unavailable", "File mounting isn't available right now.")
    # W3: the turn may have started on `auto`, in which case this mints the addressable
    # container and shares it with the loop and with sibling calls. None means we genuinely
    # could not get one — say so rather than uploading into nowhere.
    container_id = await ctx.ensure_sandbox()
    if not container_id:
        return _err("sandbox_unavailable",
                    "There is no code sandbox to mount into on this turn.")
    # F15: the sandbox idle-expired earlier this turn and was rebuilt as an ephemeral one the
    # model now runs code in. `container_id` still names the corpse, so mounting into it would
    # be invisible. Fail fast — a recycled sandbox is not a place to leave a file.
    if ctx.container_recycled():
        return _err("container_recycled",
                    "The code sandbox was recycled mid-turn, so this file can't be mounted "
                    "into it. Ask again and it will be set up fresh.")

    entry = thread_files.resolve(ctx.thread_files, file_id)
    if entry is None:
        # An unresolvable id is either invented or from another thread. Say so rather than
        # guessing at what was meant — mounting the wrong file silently corrupts whatever gets
        # built from it. On the channel surface there is no enum to fall back on, so an empty
        # catalog arrives here and is answered as the empty catalog it is.
        valid = thread_files.valid_ids(ctx.thread_files)
        return _err("unknown_file_id",
                    (f"{file_id} is not a file in this thread."
                     if valid else "There are no files in this thread to mount."),
                    valid_file_ids=valid)

    if ctx.mounted_files is None:
        ctx.mounted_files = []

    # Idempotent: the model is told it can re-call this, a tool round's calls run in parallel,
    # and across turns the container may still hold the file from last round. The cache is
    # keyed by container, so a still-live sandbox skips the upload while an expired one (a new
    # id) correctly re-mounts.
    key = _mount_key(container_id, file_id)
    cached = _recall_mount(key)
    if cached is not None:
        # Re-record on THIS turn's context too, or the publisher would lose the digest and
        # could post the user's own file back at them.
        if not any(m.get("key") == key for m in ctx.mounted_files):
            ctx.mounted_files.append(cached)
        hit = _mount_result(cached)
        hit["already_mounted"] = True
        if not cached.get("gzipped"):
            hit["message"] = "Already in the sandbox from earlier — open it from this path."
        return hit

    # F38: past every rejection AND past the cache hit above (which returns in microseconds) —
    # a real Slack download plus a container upload is about to happen. This is the honest
    # moment to stake the 👀.
    turn = getattr(ctx, "turn", None)
    if turn is not None:
        await turn.claim_work(client, getattr(ctx, "message", None))

    data = await _download(client, entry)
    if not data:
        return _err("file_unavailable",
                    f"Could not fetch {entry['filename']} from Slack. It may have been deleted.")

    if len(data) > _max_bytes():
        return _err("file_too_large",
                    f"{entry['filename']} is {len(data) / (1024 * 1024):.1f} MB, over the "
                    f"{config.artifact_max_mb} MB mount limit.")

    # The upload itself is `stage_bytes` — one place where bytes enter a container, so a user's
    # own SVG gets the same gzip transport a fetched one does instead of the raw 400. The
    # idempotency cache stays HERE, on top: it is mount_file's contract, not the uploader's.
    record = await stage_bytes(ctx, container_id, entry["filename"], data, source_id=file_id)
    if record is None:
        return _err("mount_failed",
                    f"Could not place {_safe_name(entry['filename'])} in the sandbox.")

    _remember_mount(key, record)
    logger.info(f"Mounted {file_id} ({record['filename']}, {len(data)} bytes) "
                f"at {record['path']}")

    return _mount_result(record, size_bytes=len(data), mime_type=entry["mime_type"])


def _mount_result(record: Dict[str, Any], *, size_bytes: Optional[int] = None,
                  mime_type: Optional[str] = None) -> Dict[str, Any]:
    """The success payload, shared by the fresh mount and the cache hit — so a file that could
    only be carried gzip-wrapped says so on EVERY round, not just the one that uploaded it."""
    result: Dict[str, Any] = {
        "ok": True,
        "path": record["path"],
        "filename": record["filename"],
        "message": ("Open it from this path in your next code_interpreter call. It has NOT "
                    "been posted to the user."),
    }
    if size_bytes is not None:
        result["size_bytes"] = size_bytes
    if mime_type is not None:
        result["mime_type"] = mime_type
    if record.get("gzipped"):
        result["gzipped"] = True
        result["message"] = (
            "The sandbox refused this file's bytes as-is, so it is mounted GZIP-COMPRESSED at "
            "this path. Decompress it first in your code (the `gzip` module), then work on the "
            "real bytes. It has NOT been posted to the user.")
    return result


async def stage_bytes(ctx: ToolContext, container_id: str, filename: str, data: bytes, *,
                      source_id: str) -> Optional[Dict[str, Any]]:
    """Push in-memory bytes into `container_id` and record them the way a mount is recorded.

    The other two tools that put bytes in the sandbox from OUTSIDE the conversation — the channel
    export and the web fetch — need exactly what the tail of ``execute_mount_file`` does: a safe
    filename, one ``containers.files.create``, and a digest on ``ctx.mounted_files`` so the
    artifact publisher cannot post the ingredient back out. That record shape is this module's,
    so the helper lives here rather than being copied twice.

    Returns None when the upload failed or the API returned no path; the caller owns the wording
    of its own failure. A record with ``gzipped`` True means the bytes are in the container
    COMPRESSED and the caller must say so — see ``_upload`` for why. Deliberately NOT cached in
    ``_MOUNTS``: a thread file is the same bytes every time, while a re-export or a re-fetch is a
    request for the CURRENT ones.
    """
    processor = getattr(ctx, "processor", None)
    if processor is None:
        return None
    safe = _safe_name(filename)
    try:
        raw = processor.openai_client.client
    except AttributeError:
        return None

    payload = data
    gzipped = False
    try:
        created = await _upload(raw, container_id, safe, payload)
    except Exception as e:  # noqa: BLE001
        if not _is_bad_request(e):
            logger.error(f"Staging {safe} failed ({container_id}): {e}", exc_info=True)
            return None
        # Refused on CONTENT — try the one transport that is not content (see below).
        payload = await asyncio.to_thread(gzip.compress, data)
        safe = f"{safe}.gz"
        gzipped = True
        logger.info(f"The sandbox refused {filename!r} as-is; retrying gzip-wrapped as {safe}")
        try:
            created = await _upload(raw, container_id, safe, payload)
        except Exception as e2:  # noqa: BLE001
            logger.error(f"Staging {safe} failed after gzip-wrapping ({container_id}): {e2}",
                         exc_info=True)
            return None
    path = getattr(created, "path", None)
    if not path:
        logger.error(f"The sandbox accepted {safe} but returned no path ({container_id})")
        return None
    record: Dict[str, Any] = {
        "key": _mount_key(container_id, source_id),
        "file_id": source_id,
        "path": path,
        "filename": safe,
        "container_file_id": getattr(created, "id", None),
        # The digest of what is ACTUALLY in the container, so the publisher's byte-identical
        # check works against the file that exists there.
        "digest": hashlib.sha256(payload).hexdigest(),
        "gzipped": gzipped,
    }
    if gzipped:
        # …and the digest of what it decompresses to, because the obvious first thing the model
        # does with a wrapped ingredient is write the real bytes out beside it. That copy is
        # still the ingredient and still must not be posted.
        record["digest_raw"] = hashlib.sha256(data).hexdigest()
    if ctx.mounted_files is None:
        ctx.mounted_files = []
    ctx.mounted_files.append(record)
    return record


async def _upload(raw: Any, container_id: str, filename: str, payload: bytes) -> Any:
    buf = io.BytesIO(payload)
    buf.name = filename
    return await raw.containers.files.create(container_id=container_id, file=buf)


def _is_bad_request(exc: BaseException) -> bool:
    """A 400 from the container upload — the API refusing the CONTENT.

    Matched by status AND by class name, the same belt-and-braces `is_container_gone` uses: the
    SDK raises `openai.BadRequestError` (status_code 400), but an exception that reaches here
    through a wrapper or a stub may carry only one of the two.
    """
    if getattr(exc, "status_code", None) == 400:
        return True
    return type(exc).__name__ == "BadRequestError"


def _safe_name(name: str) -> str:
    """Strip anything that could escape /mnt/data or confuse the container's filesystem."""
    cleaned = "".join(c for c in (name or "") if c.isprintable() and c not in '/\\:*?"<>|')
    cleaned = cleaned.strip().lstrip(".") or "file"
    return cleaned[:120]


def mounted_digests(ctx: ToolContext) -> List[str]:
    """Digests of everything mounted this run — the publisher refuses to post these back.

    A gzip-wrapped staging contributes TWO: the compressed bytes sitting in the container, and
    what they decompress to, which is what a model writes out beside them before working on it.
    """
    out: List[str] = []
    for m in (getattr(ctx, "mounted_files", None) or []):
        for key in ("digest", "digest_raw"):
            digest = m.get(key)
            if digest:
                out.append(digest)
    return out


def sandbox_enabled(thread_config: Optional[Dict[str, Any]] = None) -> bool:
    """Is the code sandbox switched on for THIS turn?

    Resolved exactly the way `_build_tools_array` and `_resolve_ci_container` resolve it —
    per-thread override first, then the global. Reading only the global was survivable while the
    tool also required an addressable container id, because a turn with code interpreter off
    never had one. W3 removed that second condition: `ensure_sandbox` will now happily mint and
    BIND a container for a turn whose request carries no `code_interpreter` declaration at all,
    so the file would go into a sandbox no model could ever open. This is the gate that stops it.

    It deliberately does NOT ask whether the container is addressable yet. Sandbox on with an
    `auto` container is the normal state of a first turn, and the tool belongs there.
    """
    cfg = thread_config or {}
    return bool(cfg.get("enable_code_interpreter", config.enable_code_interpreter))


def register_file_mount_tools(registry: ToolRegistry) -> None:
    """Register mount_file. A schema FACTORY (the legal ids depend on the thread), so the
    name is explicit. Generous timeout: a mount is a Slack download plus a container upload."""
    registry.register(get_mount_file_schema, execute_mount_file,
                      name="mount_file", enabled=sandbox_enabled,
                      timeout=float(getattr(config, "read_document_timeout", 60.0)) + 30.0,
                      dynamic=True, channel_schema=get_mount_file_schema_static,
                      channel_enabled=sandbox_enabled)
