"""``fetch_url_to_sandbox`` — the missing transport between the web and the sandbox.

The container has cairosvg, svglib, LibreOffice, ffmpeg and PIL, and NO network egress. The
fetcher has egress and every SSRF guard we own, and until now it could only hand back extracted
TEXT (`fetch_url`) or verified PIXELS (`import_web_image`). So an official logo published as SVG
was unreachable from both ends: the fetch refused it as an unsupported type, the image import
refused it as not-an-image, and the tools that could have converted it in a second could not be
handed the bytes.

This tool is that hop: hardened fetch in the bot process, bytes straight into `/mnt/data`, and the
conversion happens where the conversion tools already are. Nothing about the fetch is relaxed to
make it work — same validation, same redirect handling, same `link_fetch_max_bytes` cap, same
timeouts. What is dropped is the MIME allowlist, which only ever existed because the caller was
about to read the body as text.

The staged file is an INGREDIENT (`ctx.mounted_files` carries its digest, so the publisher will
not post it back out). Whatever the model BUILDS from it publishes normally.
"""
from __future__ import annotations

import asyncio
import posixpath
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import message_processor.ingestion.ambient_fetch as ambient_fetch
from config import config
from logger import setup_logger
from message_processor import file_mount
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.FetchToSandbox")

# Enough of a name to be recognizable in code; the API assigns the real path either way.
_FALLBACK_NAME = "download"

# Only for a URL whose path names no file at all ("https://host/"). A wrong extension on a file
# nobody is going to open by extension is noise; an honest one helps `soffice`/`cairosvg` pick a
# filter without the model having to rename it first.
_EXT_BY_TYPE = {
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "text/html": ".html",
    "application/json": ".json",
    "application/zip": ".zip",
}


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


def get_fetch_url_to_sandbox_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "fetch_url_to_sandbox",
        "description": (
            "Download a file from an http(s) URL straight into the code sandbox and return its "
            "/mnt/data path. Any content type — SVG and other vector art, Office and legacy "
            "spreadsheet formats, archives, media, data files — the bytes are never read into "
            "your context.\n\n"
            "This is how an asset gets CONVERTED: the sandbox has no internet, so bytes have to "
            "be pushed in. Fetch it here, then convert it with code (cairosvg or svglib for SVG, "
            "soffice for Office and legacy formats, PIL or ffmpeg for raster and media) and post "
            "the result you built. Look at what you converted before you post it.\n\n"
            "Some formats can only be carried in gzip-compressed — the result says so when that "
            "happens; decompress it in the sandbox before using the file.\n\n"
            "What you fetch here is an INGREDIENT and can never be posted from the sandbox: a "
            "file that comes back unchanged — including a straight copy under a new name — is "
            "refused by the publisher, so only a genuinely TRANSFORMED output (converted, "
            "resized, composited, built into a document) reaches the user. To deliver a web "
            "image AS-IS, do not route it through here at all: import_web_image posts it "
            "directly and checks the pixels first. For a page you need to READ, use fetch_url."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "The http(s) URL of the file to download."},
                "filename": {"type": "string",
                             "description": ("Optional name for the sandbox file. Defaults to the "
                                             "name in the URL.")},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    }


def _derive_filename(url: str, content_type: Optional[str], override: Any) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()
    name = ""
    try:
        name = posixpath.basename(urlsplit(url).path or "")
    except ValueError:
        name = ""
    if name:
        return name
    family = (content_type or "").split(";", 1)[0].strip().lower()
    return _FALLBACK_NAME + _EXT_BY_TYPE.get(family, "")


async def execute_fetch_url_to_sandbox(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch → sandbox → path. Never raises: every failure is a result."""
    try:
        return await _fetch_to_sandbox(ctx, args)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — an executor failure is a tool result, not a turn's end
        logger.error(f"fetch_url_to_sandbox failed: {e}", exc_info=True)
        return _err("fetch_failed", "Could not stage that URL in the sandbox.")


async def _fetch_to_sandbox(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        return _err("missing_url", "A url is required.")
    url = url.strip()

    if getattr(ctx, "processor", None) is None:
        return _err("unavailable", "Fetching into the sandbox isn't available right now.")

    # A real download is about to happen; claim the work before it, not on entry.
    turn = getattr(ctx, "turn", None)
    if turn is not None:
        try:
            await turn.claim_work(getattr(ctx, "client", None), getattr(ctx, "message", None))
        except Exception:  # noqa: BLE001 — presentation never breaks the fetch
            pass

    # The fetch comes FIRST, before a container is minted: a URL that turns out to be blocked,
    # oversized or dead should not have paid for a sandbox nobody will use.
    result = await ambient_fetch.fetch_url(
        url,
        max_bytes=int(config.link_fetch_max_bytes),
        connect_timeout=float(config.link_fetch_connect_timeout_s),
        read_timeout=float(config.link_fetch_read_timeout_s),
        total_timeout=float(config.link_fetch_total_timeout_s),
        max_redirects=int(config.link_fetch_max_redirects),
        max_chars=0,
        raw_mode=True)
    # Redacted: a signed URL carries its bearer token in the query string, and nothing
    # un-redacted may be logged or handed back to the model.
    source_ref = ambient_fetch.redact_url(result.final_url or url)
    if result.kind != "bytes" or not result.raw_bytes:
        return _err(result.error_code or "fetch_failed",
                    "Could not download that URL.",
                    detail=result.error_detail, source_url=source_ref)

    data = result.raw_bytes
    container_id = await ctx.ensure_sandbox()
    if not container_id:
        return _err("sandbox_unavailable",
                    "There is no code sandbox to fetch into on this turn.")
    # F15: the sandbox died mid-turn and was rebuilt as an ephemeral one — writing into the
    # corpse would leave a file the model can never open.
    if ctx.container_recycled():
        return _err("container_recycled",
                    "The code sandbox was recycled mid-turn, so this file can't be placed in "
                    "it. Ask again and it will be set up fresh.")

    filename = _derive_filename(result.final_url or url, result.content_type,
                               args.get("filename"))
    record = await file_mount.stage_bytes(ctx, container_id, filename, data,
                                          source_id=f"fetch:{source_ref}")
    if record is None:
        return _err("stage_failed", f"Downloaded {filename} but could not place it in the "
                                    "sandbox.")
    logger.info(f"fetch_url_to_sandbox staged {record['filename']} "
                f"({len(data)} bytes, {result.content_type or 'unknown type'}) at "
                f"{record['path']}")
    staged = {
        "ok": True,
        "path": record["path"],
        "filename": record["filename"],
        "content_type": result.content_type,
        "size_bytes": len(data),
        "source_url": source_ref,
        "message": ("Open it from this path in your next code_interpreter call. It has NOT been "
                    "posted to the user — post what you build from it."),
    }
    if record.get("gzipped"):
        # Said plainly, because the model is about to hand this path to a converter that will
        # fail on gzip bytes and report the ASSET as broken rather than the wrapper.
        staged["gzipped"] = True
        staged["message"] = (
            "The sandbox refused these bytes as-is, so the file is staged GZIP-COMPRESSED at "
            "this path. Decompress it first in your code (the `gzip` module: "
            "`gzip.open(path, 'rb').read()`, or write it out with the wrapper removed), then "
            "convert the real bytes. It has NOT been posted to the user — post what you build "
            "from it.")
    return staged


def register_fetch_to_sandbox_tool(registry: ToolRegistry) -> None:
    """Register fetch_url_to_sandbox (the caller gates on ENABLE_LINK_FETCH — the SSRF guard
    lives in the fetcher). Gated here on the sandbox being on: with nowhere to put the bytes
    there is no tool. The timeout is the fetch's own budget plus room for the container upload."""
    registry.register(get_fetch_url_to_sandbox_schema(), execute_fetch_url_to_sandbox,
                      enabled=file_mount.sandbox_enabled,
                      channel_enabled=file_mount.sandbox_enabled,
                      timeout=float(config.link_fetch_total_timeout_s) + 30.0)
