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
"""
from __future__ import annotations

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
    """Offered only when there is somewhere to mount TO and something to mount.

    Returning ``None`` hides the tool, which is the honest state: with an ``auto`` container
    we have no addressable id to push bytes into, and with no files there is nothing to name.
    """
    cfg = thread_config or {}
    from message_processor.image_tools import CI_CONTAINER_KEY

    container = cfg.get(CI_CONTAINER_KEY)
    if not isinstance(container, str) or not container:
        return None

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
    container_id = getattr(ctx, "container_id", None)
    if processor is None or client is None:
        return _err("unavailable", "File mounting isn't available right now.")
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
        # The ids are an enum built from this turn's snapshot; an unresolvable one is either
        # invented or from another thread. Say so rather than guessing at what was meant —
        # mounting the wrong file silently corrupts whatever gets built from it.
        return _err("unknown_file_id",
                    f"{file_id} is not a file in this thread.",
                    valid_file_ids=thread_files.valid_ids(ctx.thread_files))

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
        return {"ok": True, "path": cached["path"], "filename": cached["filename"],
                "already_mounted": True,
                "message": "Already in the sandbox from earlier — open it from this path."}

    # F38: past every rejection AND past the cache hit above (which returns in microseconds) —
    # a real Slack download plus a container upload is about to happen. This is the honest
    # moment to stake the 👀.
    turn = getattr(ctx, "turn", None)
    if turn is not None:
        turn.mark_substantive_work()  # F46: real download + mount ⇒ thread a top-level reply
        await turn.claim_work(client, getattr(ctx, "message", None))

    data = await _download(client, entry)
    if not data:
        return _err("file_unavailable",
                    f"Could not fetch {entry['filename']} from Slack. It may have been deleted.")

    if len(data) > _max_bytes():
        return _err("file_too_large",
                    f"{entry['filename']} is {len(data) / (1024 * 1024):.1f} MB, over the "
                    f"{config.artifact_max_mb} MB mount limit.")

    filename = _safe_name(entry["filename"])
    try:
        raw = processor.openai_client.client
        buf = io.BytesIO(data)
        buf.name = filename
        created = await raw.containers.files.create(container_id=container_id, file=buf)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Mount upload failed ({container_id}): {e}", exc_info=True)
        return _err("mount_failed", f"Could not place {filename} in the sandbox.")

    # The API assigns the path; assuming /mnt/data/<name> would be a guess.
    path = getattr(created, "path", None)
    if not path:
        return _err("mount_failed", f"The sandbox accepted {filename} but returned no path.")

    record = {
        "key": key,
        "file_id": file_id,
        "path": path,
        "filename": filename,
        "container_file_id": getattr(created, "id", None),
        "digest": hashlib.sha256(data).hexdigest(),
    }
    ctx.mounted_files.append(record)
    _remember_mount(key, record)
    logger.info(f"Mounted {file_id} ({filename}, {len(data)} bytes) at {path}")

    return {
        "ok": True,
        "path": path,
        "filename": filename,
        "size_bytes": len(data),
        "mime_type": entry["mime_type"],
        "message": ("Open it from this path in your next code_interpreter call. It has NOT "
                    "been posted to the user."),
    }


def _safe_name(name: str) -> str:
    """Strip anything that could escape /mnt/data or confuse the container's filesystem."""
    cleaned = "".join(c for c in (name or "") if c.isprintable() and c not in '/\\:*?"<>|')
    cleaned = cleaned.strip().lstrip(".") or "file"
    return cleaned[:120]


def mounted_digests(ctx: ToolContext) -> List[str]:
    """Digests of everything mounted this run — the publisher refuses to post these back."""
    return [m["digest"] for m in (getattr(ctx, "mounted_files", None) or []) if m.get("digest")]


def register_file_mount_tools(registry: ToolRegistry) -> None:
    """Register mount_file. A schema FACTORY (the legal ids depend on the thread), so the
    name is explicit. Generous timeout: a mount is a Slack download plus a container upload."""
    registry.register(get_mount_file_schema, execute_mount_file,
                      name="mount_file",
                      timeout=float(getattr(config, "read_document_timeout", 60.0)) + 30.0)
