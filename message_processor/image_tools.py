"""Image generation and editing as first-class tools (F34).

This replaces the pre-flight intent classifier's image branches. The classifier had to
decide, before the model saw anything, whether a turn was "an image request" — one
irreversible choice for the whole turn. That router is what drew a chart with invented
numbers when someone said "chart this CSV", and it made composition impossible: a turn could
generate an image OR run code, never both.

As tools, the model decides in context, can call more than one, and can feed an image into
the sandbox. Three tools, deliberately named so their execution contracts are hard to
confuse:

* ``generate_image`` — DETACHED. The image is the deliverable; it posts itself into the
  thread when it is ready and the conversation stays usable meanwhile. This is the common case.
* ``create_image_asset`` — SYNCHRONOUS. The image is an *ingredient*: the bytes are pushed
  into the thread's code-interpreter container at ``/mnt/data`` and nothing is posted. This is
  what makes "build me a deck with a cover image and charts from this CSV" a single workflow.
* ``edit_image`` — SYNCHRONOUS. Edits images the thread already has, chosen by opaque id from
  a catalog we hand the model, and posts the result.

Every executor is a plain registry executor, so a background agent running its own
ToolRegistry gets image generation for free — it just supplies its own ToolContext.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import config, pipeline_status
from logger import setup_logger
from message_processor import image_catalog
from message_processor.image_service import (
    FORMATS,
    NAMED_SIZES,
    QUALITIES,
    backgrounds_for,
    defaults_sentence,
    image_model_for,
    is_v2,
    resolve_settings,
    supports_input_fidelity,
)

logger = setup_logger(name="slack_bot.ImageTools")

# Request-config keys the text handler stashes so a schema factory (which only ever sees
# thread_config) can shape the tool to this turn.
CI_CONTAINER_KEY = "_ci_container"
CATALOG_KEY = "_image_catalog"

# A filename the model picks lands in a shell-adjacent sandbox path. Keep it boring.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_ASSETS_PER_TURN = 4

# Image calls are expensive and slow, and `dispatch_all` runs a round's calls CONCURRENTLY —
# a model that emits several create_image_asset/edit_image calls in one round would otherwise
# fire them all at once. This bounds the SYNCHRONOUS tools, which do their API call inline.
#
# It deliberately does NOT cover detached generate_image: that call happens later, inside
# `_finish_image_generation_background`, after the turn (and the thread lock) is gone. Holding
# a process-wide semaphore across a detached job would let one thread's queue stall every
# other thread's synchronous image work. Detached generations are bounded per-thread by
# MAX_CONCURRENT_IMAGE_GENERATIONS, as they were before F34.
_image_semaphore: Optional[asyncio.Semaphore] = None


def _semaphore() -> asyncio.Semaphore:
    global _image_semaphore
    if _image_semaphore is None:
        _image_semaphore = asyncio.Semaphore(max(1, config.max_concurrent_image_generations))
    return _image_semaphore


def _reset_semaphore_for_tests() -> None:
    global _image_semaphore
    _image_semaphore = None


# --- schemas -----------------------------------------------------------------------------

def _overrides_schema(thread_config: Dict[str, Any]) -> Dict[str, Any]:
    """The option space for the SELECTED image model.

    Built per request, because the legal values differ by model: gpt-image-2 has no
    transparent background and takes arbitrary WxH sizes. Advertising `transparent` and then
    silently coercing it to `auto` would teach the model it got something it did not.

    The image model itself is deliberately absent — it is the user's choice and the one hard
    constraint, so the model cannot express a different one even by accident.
    """
    model = image_model_for(thread_config)
    size: Dict[str, Any] = {
        "type": "string",
        "description": "Image dimensions.",
    }
    if is_v2(model):
        size["description"] = (
            f"Image dimensions: one of {', '.join(NAMED_SIZES)}, or a custom WxH. Each side "
            "must be DIVISIBLE BY 16 (so 1920x1080 is invalid), between 256 and 3840 wide "
            "and 256 and 2160 tall, and no more extreme than 3:1. For a 16:9 slide use "
            "1536x864. A near-miss is snapped to the nearest valid size.")
    else:
        size["enum"] = list(NAMED_SIZES)

    props: Dict[str, Any] = {
        "size": size,
        "quality": {"type": "string", "enum": list(QUALITIES),
                    "description": "Rendering quality. Higher costs more and takes longer."},
        "background": {"type": "string", "enum": backgrounds_for(model),
                       "description": "Background treatment."},
        "format": {"type": "string", "enum": list(FORMATS), "description": "Output file format."},
    }
    if supports_input_fidelity(model):
        props["input_fidelity"] = {
            "type": "string", "enum": ["low", "high"],
            "description": ("Editing only: how closely to preserve the source image. "
                            "'high' keeps faces/logos/layout intact."),
        }
    return {
        "type": "object",
        "description": ("Task-specific departures from the user's saved defaults. OMIT THIS "
                        "ENTIRELY unless the task has a concrete reason to differ — a wide "
                        "image for a title slide, an opaque background for print. Do not "
                        "restate the defaults."),
        "properties": props,
        "additionalProperties": False,
    }


def get_generate_image_schema(thread_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "generate_image",
        "description": (
            "Create a NEW image from a description and post it into this Slack thread. "
            "Runs in the background: this returns immediately and the image arrives on its "
            "own, so the conversation stays usable while it renders.\n\n"
            "Use this when the image IS what the user asked for.\n\n"
            "Do NOT use this when the image is an ingredient for other work in this same "
            "turn — a picture to place in a slide deck, a logo to composite, a frame to "
            "process with code. Use create_image_asset for that instead; it hands the bytes "
            "to the code sandbox rather than posting them.\n\n"
            "Do NOT use this to chart or plot data. Charts are computed from real numbers "
            "with code_interpreter; an image model would draw a convincing chart with "
            "invented values.\n\n"
            f"The user's saved image settings are: {defaults_sentence(thread_config)}. "
            "They apply automatically.\n\n"
            "Say one short line to the user acknowledging you are making it (e.g. \"Making "
            "that now — it'll land here shortly.\"). Do not describe the image you are about "
            "to make; they will see it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": ("What the image should depict, in the user's terms. It is "
                                    "rewritten into a fuller prompt for you automatically — "
                                    "do not pad it with style boilerplate."),
                },
                "overrides": _overrides_schema(thread_config),
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    }


def get_create_image_asset_schema(thread_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "create_image_asset",
        "description": (
            "Create an image and place it in the code sandbox at /mnt/data so "
            "code_interpreter can USE it — as a slide image, a composited layer, an input to "
            "processing. It is NOT posted to Slack. Only the files your code writes get "
            "published, so the image reaches the user through whatever you build with it "
            "(a .pptx, a .docx, a composite .png).\n\n"
            "This BLOCKS until the image exists, which takes a while. Call it BEFORE the "
            "code_interpreter call that consumes it and WAIT for the path it returns — tool "
            "calls made in the same round cannot see each other's results.\n\n"
            "If the user just wants an image, use generate_image instead — it does not block "
            "and it posts the image for you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What the image should depict."},
                "filename": {
                    "type": "string",
                    "description": ("Filename to save it under in /mnt/data, e.g. "
                                    "'cover.png'. Use something your code can refer to."),
                },
                "overrides": _overrides_schema(thread_config),
            },
            "required": ["prompt", "filename"],
            "additionalProperties": False,
        },
    }


def get_edit_image_schema(thread_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Built from THIS TURN's catalog: the ids are a literal enum, so the model cannot name
    an image that does not exist. No catalog → no tool (there is nothing to edit)."""
    entries = (thread_config or {}).get(CATALOG_KEY) or []
    ids = image_catalog.valid_ids(entries)
    if not ids:
        return None
    return {
        "type": "function",
        "name": "edit_image",
        "description": (
            "Edit, restyle, or combine image(s) already in this thread, and post the result "
            "into the thread. Give the id(s) of the image(s) to work from; pass several ids "
            "to combine them into one image.\n\n"
            "Images available in this thread:\n"
            f"{image_catalog.catalog_lines(entries)}\n\n"
            "If the user's reference is ambiguous and picking wrong would waste real work, "
            "ask them which one rather than guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_image_ids": {
                    "type": "array",
                    "description": "The image(s) to edit, by id, from the list above.",
                    "items": {"type": "string", "enum": ids},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "prompt": {
                    "type": "string",
                    "description": "The change to make, in the user's terms.",
                },
                "overrides": _overrides_schema(thread_config),
            },
            "required": ["source_image_ids", "prompt"],
            "additionalProperties": False,
        },
    }


# --- shared helpers ----------------------------------------------------------------------

def _effective_config(thread_config: Optional[Dict[str, Any]],
                      overrides: Optional[Dict[str, Any]]) -> tuple:
    """Fold overrides onto the user's prefs and return a thread_config carrying the result,
    so downstream code (which reads thread_config) needs no new parameter."""
    settings, rejected = resolve_settings(thread_config, overrides)
    cfg = dict(thread_config or {})
    cfg.update({
        "image_model": settings["model"],
        "image_size": settings["size"],
        "image_quality": settings["quality"],
        "image_background": settings["background"],
        "image_format": settings["format"],
        "image_compression": settings["compression"],
        "input_fidelity": settings["input_fidelity"],
    })
    return settings, rejected, cfg


def _thread_key(ctx) -> str:
    return f"{ctx.channel_id}:{ctx.thread_ts}"


def _err(error: str, message: str, **extra) -> Dict[str, Any]:
    out = {"ok": False, "error": error, "message": message}
    out.update(extra)
    return out


def _moderation_blocked(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("moderation_blocked" in text or "safety system" in text
            or "content policy" in text)


_MODERATION_MESSAGE = (
    "The image was refused by the content safety filter. This often happens with brand "
    "names, real people, or other protected content. Tell the user plainly and offer to try "
    "a rephrased description.")


# --- generate_image (detached) -----------------------------------------------------------

async def execute_generate_image(ctx, args: Dict[str, Any]) -> Dict[str, Any]:
    """Detach a background generation and return immediately. The thread lock releases at
    the end of this turn; the image posts itself later. Mirrors start_deep_research."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return _err("bad_arguments", "A prompt is required.")

    processor = getattr(ctx, "processor", None)
    client = getattr(ctx, "client", None)
    if processor is None or client is None:
        return _err("unavailable", "Image generation is not available in this context.")

    thread_key = _thread_key(ctx)
    tm = processor.thread_manager

    # Per-thread cap, checked under the turn's lock (this executor runs inside it).
    in_flight = len(tm.generations_in_flight(thread_key))
    if in_flight >= config.max_concurrent_image_generations:
        return _err(
            "at_capacity",
            f"{in_flight} image(s) are already generating in this thread — the limit is "
            f"{config.max_concurrent_image_generations}. Tell the user to wait for one to land.")

    settings, rejected, effective_cfg = _effective_config(ctx.thread_config, args.get("overrides"))

    from message_processor.progress import ProgressChecklist
    # prefer_message=True is NOT the config default, and is forced here on purpose: this job is
    # DETACHED. Slack's assistant-status surface auto-clears the moment the turn's reply posts,
    # which is roughly a minute before the image is actually ready — so a status-only checklist
    # would vanish instantly and the whole generation would run with no indicator at all. A
    # detached job needs a surface that outlives the turn that started it, exactly like deep
    # research's status card. The synchronous tools below keep the ephemeral status, because
    # for them the turn IS still open.
    checklist = ProgressChecklist(
        client, ctx.channel_id, ctx.thread_ts, prefer_message=True)
    try:
        # No "enhancing prompt" step: the enhancement still happens inside generate_image,
        # it is simply not the user's business (it is our internal processing, like the
        # tools we ran). One honest line about the thing they actually asked for.
        await checklist.step(pipeline_status("generating_image", "Generating image…"),
                             done_text="Generated image")
    except Exception as e:  # noqa: BLE001 — a status hiccup must not block the generation
        logger.debug(f"checklist start failed: {e}")

    generation_id = uuid4().hex[:12]
    tm.register_generation(thread_key, generation_id, prompt[:200])
    try:
        task = processor._schedule_async_call(processor._finish_image_generation_background(
            client=client, channel_id=ctx.channel_id, thread_id=ctx.thread_ts,
            thread_key=thread_key, prompt=prompt, enhance=True,
            conversation_history=None, thread_config=effective_cfg,
            checklist=checklist, generating_id=None, generation_id=generation_id,
            message_ts=ctx.trigger_ts, unprompted=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to schedule background generation for {thread_key}: {e}",
                     exc_info=True)
        tm.finish_generation(thread_key, generation_id)
        await processor._abort_checklist(checklist, client, ctx.channel_id, ctx.thread_ts)
        return _err("schedule_failed", "The image generation could not be started.")

    if task is not None:
        tm.attach_generation_task(thread_key, generation_id, task)
    # Claim the upload latch NOW, while the turn still holds the thread lock, so a fast
    # follow-up "edit it" cannot win the lock and target a stale image (F1 TOCTOU fix).
    try:
        tm.mark_upload_started(thread_key, generation_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to mark upload started for {thread_key}: {e}")

    ctx.image_generation_started = True
    logger.info(f"Detached image generation {generation_id} for {thread_key} "
                f"(model={settings['model']} size={settings['size']})")

    result = {
        "ok": True,
        "status": "generating",
        "message": ("The image is being generated and will post into the thread on its own. "
                    "Acknowledge briefly; do not describe the image."),
        "settings": {k: settings[k] for k in ("model", "size", "quality", "background")},
    }
    if rejected:
        result["ignored_overrides"] = rejected
    return result


# --- create_image_asset (synchronous, into the sandbox) ----------------------------------

def _safe_filename(name: str, fmt: str) -> str:
    base = _SAFE_NAME_RE.sub("_", (name or "").strip()).strip("._-")
    if not base:
        base = "image"
    # Keep the extension honest: the bytes are `fmt`, whatever the model called the file.
    base = base.rsplit(".", 1)[0][:64]
    ext = "jpg" if fmt == "jpeg" else fmt
    return f"{base}.{ext}"


async def execute_create_image_asset(ctx, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synchronously and push the bytes into the thread's persistent container.

    The container id MUST be the one already placed in the tools array for this turn — the
    model can only see files in the container it is actually running code in.
    """
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return _err("bad_arguments", "A prompt is required.")

    processor = getattr(ctx, "processor", None)
    if processor is None:
        return _err("unavailable", "Image generation is not available in this context.")

    container_id = getattr(ctx, "container_id", None)
    if not container_id:
        return _err(
            "sandbox_unavailable",
            "There is no code sandbox for this thread right now, so an image cannot be "
            "placed in it. Use generate_image to post an image to the thread instead.")

    assets = ctx.sandbox_image_assets if ctx.sandbox_image_assets is not None else []
    if len(assets) >= _MAX_ASSETS_PER_TURN:
        return _err("at_capacity",
                    f"Already created {len(assets)} sandbox images this turn (limit "
                    f"{_MAX_ASSETS_PER_TURN}). Work with the ones you have.")

    settings, rejected, _ = _effective_config(ctx.thread_config, args.get("overrides"))
    filename = _safe_filename(args.get("filename") or "image", settings["format"])

    try:
        async with _semaphore():
            image_data = await processor.openai_client.generate_image(
                prompt=prompt,
                model=settings["model"],
                size=settings["size"],
                quality=settings["quality"],
                background=settings["background"],
                format=settings["format"],
                compression=settings["compression"],
                enhance_prompt=True,
                conversation_history=None,
            )
    except Exception as e:  # noqa: BLE001 — a tool must never raise into the loop
        if _moderation_blocked(e):
            return _err("moderation_blocked", _MODERATION_MESSAGE)
        logger.error(f"create_image_asset generation failed: {e}", exc_info=True)
        return _err("generation_failed", "The image could not be generated.")

    path = await mount_image_in_container(
        processor.openai_client, container_id, filename, image_data)
    if not path:
        return _err("mount_failed",
                    "The image was generated but could not be placed in the sandbox.")

    if ctx.sandbox_image_assets is None:
        ctx.sandbox_image_assets = []
    ctx.sandbox_image_assets.append({
        "path": path,
        "filename": filename,
        "prompt": prompt,
        "enhanced_prompt": getattr(image_data, "prompt", "") or prompt,
        "image_data": image_data,
    })

    logger.info(f"Mounted generated image at {path} in {container_id}")
    result = {
        "ok": True,
        "path": path,
        "format": settings["format"],
        "size": settings["size"],
        "message": ("The image is now at this path inside the sandbox. Open it from there in "
                    "your next code_interpreter call. It has NOT been posted to the user."),
    }
    if rejected:
        result["ignored_overrides"] = rejected
    return result


async def mount_image_in_container(openai_client, container_id: str, filename: str,
                                   image_data) -> Optional[str]:
    """Push bytes INTO a container. Files land as ``source="user"``, so the artifact
    publisher correctly does not re-publish them — an ingredient is not a deliverable.

    Only possible because container ids are persisted: under ``{"type": "auto"}`` the id is
    unknown until after the call, so there is nothing to push into.
    """
    try:
        raw = openai_client.client
        buf = image_data.to_bytes()
        buf.name = filename
        created = await raw.containers.files.create(container_id=container_id, file=buf)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Container file upload failed ({container_id}): {e}", exc_info=True)
        return None

    path = getattr(created, "path", None)
    if not path:
        # The API assigns the path; without one the model has no way to open the file.
        logger.error(f"Container file upload returned no path ({container_id})")
        return None
    return path


# --- edit_image (synchronous, into the thread) -------------------------------------------

async def execute_edit_image(ctx, args: Dict[str, Any]) -> Dict[str, Any]:
    """Edit thread images by catalog id and post the result.

    The ids are validated against THIS TURN's snapshot: a syntactically valid id is not
    authorization. There is no "most recent" fallback — editing the wrong image is an
    expensive, irreversible side effect, so an unresolvable id is an error the model can
    recover from, not a guess.
    """
    ids = args.get("source_image_ids") or []
    prompt = (args.get("prompt") or "").strip()
    if not isinstance(ids, list) or not ids:
        return _err("bad_arguments", "At least one source_image_id is required.")
    if not prompt:
        return _err("bad_arguments", "A prompt describing the edit is required.")

    processor = getattr(ctx, "processor", None)
    client = getattr(ctx, "client", None)
    if processor is None or client is None:
        return _err("unavailable", "Image editing is not available in this context.")

    entries = ctx.image_catalog or []
    resolved = []
    for image_id in ids[:8]:
        entry = image_catalog.resolve(entries, str(image_id))
        if entry is None:
            return _err("unknown_image_id",
                        f"No image {image_id!r} in this thread.",
                        valid_image_ids=image_catalog.valid_ids(entries))
        resolved.append(entry)

    thread_key = _thread_key(ctx)
    settings, rejected, _ = _effective_config(ctx.thread_config, args.get("overrides"))

    # Download the sources from Slack. They are never held on disk — bytes in, bytes out.
    b64_images: List[str] = []
    for entry in resolved:
        data = await _download_b64(client, entry["url"])
        if data is None:
            return _err("source_unavailable",
                        f"Could not fetch {entry['image_id']} from Slack.")
        b64_images.append(data)

    # The edit-prompt enhancer wants a description of what it is editing. The catalog
    # already carries the stored analysis, so the old flow's extra vision round-trip
    # (download → analyze → enhance) collapses to a lookup on the common path.
    description = next((e.get("analysis") for e in resolved if e.get("analysis")), None)

    from message_processor.progress import ProgressChecklist
    checklist = ProgressChecklist(
        client, ctx.channel_id, ctx.thread_ts,
        prefer_message=config.progress_checklist_prefer_message)
    try:
        await checklist.step(pipeline_status("editing_image", "Editing image…"),
                             done_text="Edited image")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"checklist start failed: {e}")

    try:
        async with _semaphore():
            image_data = await processor.openai_client.edit_image(
                input_images=b64_images,
                prompt=prompt,
                model=settings["model"],
                image_description=description,
                input_fidelity=settings["input_fidelity"],
                quality=settings["quality"],
                background=settings["background"],
                output_format=settings["format"],
                output_compression=settings["compression"],
                enhance_prompt=True,
                conversation_history=None,
            )
    except Exception as e:  # noqa: BLE001
        await processor._abort_checklist(checklist, client, ctx.channel_id, ctx.thread_ts)
        if _moderation_blocked(e):
            return _err("moderation_blocked", _MODERATION_MESSAGE)
        logger.error(f"edit_image failed for {thread_key}: {e}", exc_info=True)
        return _err("edit_failed", "The image could not be edited.")

    from message_processor.image_delivery import publish_image
    file_url = await publish_image(
        processor=processor, client=client, channel_id=ctx.channel_id,
        thread_id=ctx.thread_ts, thread_key=thread_key, image_data=image_data,
        checklist=checklist, generation_id=None, prompt=image_data.prompt,
        db=ctx.db, thread_manager=processor.thread_manager, unprompted=False,
        message_ts=ctx.trigger_ts, image_type="edited",
    )
    if not file_url:
        return _err("post_failed", "The edited image was created but could not be posted.")

    processor.thread_manager.mark_needs_refresh(thread_key)
    logger.info(f"Edited image posted for {thread_key} from {[e['image_id'] for e in resolved]}")

    result = {
        "ok": True,
        "status": "posted",
        "message": ("The edited image has been posted to the thread. Acknowledge briefly; do "
                    "not describe it."),
        "sources": [e["image_id"] for e in resolved],
    }
    if rejected:
        result["ignored_overrides"] = rejected
    return result


async def _download_b64(client, url: str) -> Optional[str]:
    """Slack URL → base64, in memory only (files never touch disk)."""
    try:
        raw = await client.download_file(url)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Image download failed ({url}): {e}")
        return None
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            base64.b64decode(raw, validate=True)
            return raw
        except (binascii.Error, ValueError):
            return None
    return base64.b64encode(raw).decode("utf-8")


# --- registration ------------------------------------------------------------------------

def _tools_enabled(cfg: dict) -> bool:
    return True


def _asset_tool_enabled(cfg: dict) -> bool:
    """Only offered when the sandbox exists AND has an addressable container. Under
    ``{"type": "auto"}`` the id is unknown until after the call, so there is nothing to push
    bytes into — offering the tool would guarantee a failed call."""
    if not config.enable_code_interpreter:
        return False
    return isinstance(cfg.get(CI_CONTAINER_KEY), str) and bool(cfg.get(CI_CONTAINER_KEY))


def register_image_tools(registry) -> None:
    # Image calls take far longer than the generic tool timeout; a synchronous one must be
    # allowed to finish. The detached one returns instantly and needs no extra room.
    sync_timeout = float(config.api_timeout_image) + 60.0
    registry.register(get_generate_image_schema, execute_generate_image,
                      enabled=_tools_enabled, name="generate_image")
    registry.register(get_create_image_asset_schema, execute_create_image_asset,
                      enabled=_asset_tool_enabled, timeout=sync_timeout,
                      name="create_image_asset")
    registry.register(get_edit_image_schema, execute_edit_image,
                      enabled=_tools_enabled, timeout=sync_timeout, name="edit_image")
