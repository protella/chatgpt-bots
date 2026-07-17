from __future__ import annotations

import asyncio
import time
from typing import Optional, TypeGuard

from config import config
from message_processor.progress import ProgressChecklist

# Longest enhanced prompt we'll put under an image. It is a caption, not an essay.
_CAPTION_CHARS = 700


def _enhanced_prompt_caption(image_data, prompt: str) -> str:
    """The enhanced prompt, as the image's caption — only when SHOW_ENHANCED_PROMPT is on.

    Enhancement always RUNS (it is most of what makes the picture good); this only controls
    whether the user has to look at it. It used to be shown unconditionally, as its own noisy
    block above every image, which is why the default here is off.
    """
    if not config.show_enhanced_prompt:
        return ""
    enhanced = (getattr(image_data, "prompt", "") or "").strip()
    # Nothing to show if the enhancer was a no-op — repeating the user's own words back at
    # them is not a feature.
    if not enhanced or enhanced == (prompt or "").strip():
        return ""
    if len(enhanced) > _CAPTION_CHARS:
        enhanced = enhanced[:_CAPTION_CHARS].rstrip() + "…"
    return f"_Enhanced prompt: {enhanced}_"


async def publish_image(
    *,
    processor,
    client,
    channel_id: str,
    thread_id: str,
    thread_key: str,
    image_data,
    checklist: Optional[ProgressChecklist],
    generation_id: Optional[str],  # None on the legacy sync path
    prompt: str,
    db,
    thread_manager,
    unprompted: bool,
    message_ts: Optional[str] = None,
    image_type: str = "generated",
    provenance_tool: Optional[str] = None,
) -> Optional[str]:
    """Single owner of image delivery for both the background job and the sync path:
    checklist "Uploading…" transition, upload, falsey-URL = failure, persistence,
    asset-ledger update, checklist completion, and unprompted participation accounting.
    Returns the file URL, or None on failure.

    Upload and persistence are separated: once send_image returns a URL the image IS
    posted, so a later persistence failure is logged but never un-posts it (the user is
    not told it failed, accounting still counts it). Persistence writes the DB row
    DIRECTLY on both paths (merge-preserving upsert) so it never depends on finding a
    mutable in-memory breadcrumb that a mid-flight refresh may have wiped; the sync path
    additionally refreshes the warm breadcrumb (URL + analysis) via update_last_image_url.

    The upload latch is NOT released here — its lifecycle is owned by the caller
    (generation-ID-conditional in the background job's finally; main.py's finally on the
    sync path) so a watchdog-cleared zombie can't signal a newer job's upload done.

    ``provenance_tool`` (F7) is the name of the tool that actually made this image, and is
    passed ONLY by callers that know it. The bot's text reply gets a provenance row keyed on
    the reply's ts; the image never did, and on a SILENT image turn — the model calls
    generate_image and says nothing — there is no reply ts at all, so the turn's provenance
    was dropped outright and the model would later deny having made its own image.

    Resolving the image's own ts answers TWO questions, so it is started once and shared: it
    keys the provenance row, and — because the share record is what makes the image visible —
    it is also the signal that the "Uploading…" indicator can finally come down. The
    provenance half stays detached (nothing may delay delivery for an invisible DB row); the
    indicator half is awaited here, bounded by image_indicator_hold_seconds, because a
    checklist that completes before its image exists is the gap this is here to close. Neither
    can un-post the image, which is already in the thread by then.
    """
    if checklist is not None:
        try:
            await checklist.step("Uploading…", done_text="Uploaded")
        except Exception as e:  # noqa: BLE001 — a status hiccup must not abort delivery
            processor.log_debug(f"checklist upload step failed: {e}")

    # 1) Upload. A returned URL means the image is POSTED — nothing below may flip it back.
    file_url: Optional[str] = None
    upload_meta: dict = {}
    try:
        file_url = await client.send_image(
            channel_id,
            thread_id,
            image_data.to_bytes(),
            f"generated_image.{image_data.format}",
            _enhanced_prompt_caption(image_data, prompt),
            meta_out=upload_meta,
        )
    except Exception as e:  # noqa: BLE001
        processor.log_error(f"Image upload failed for {thread_key}: {e}", exc_info=True)
        file_url = None
    # Stamped here, not at the hold below: upload-return is the moment the OLD code called it
    # done and dropped the indicator, so it is the origin the visible gap is measured from.
    uploaded_at = time.monotonic()

    if not file_url:
        if checklist is not None:
            try:
                await checklist.fail("Upload failed")
            except Exception:
                pass
        return None

    # 2) Persist — the image is already posted; failures here are logged, never un-post it.
    prompt_text = prompt or getattr(image_data, "prompt", "") or ""
    try:
        if db:
            meta = {"timestamp": time.time()}
            if generation_id:
                meta["generation_id"] = generation_id
            # Direct, breadcrumb-INDEPENDENT write (a Phase-Q refresh between lock release
            # and upload can wipe the sync path's breadcrumb before we get here). The
            # upsert is merge-preserving, so a later breadcrumb/rebuild write can enrich
            # analysis without clobbering prompt/type/generation_id.
            await db.save_image_metadata_async(
                thread_id=thread_key,
                url=file_url,
                image_type=image_type,
                prompt=prompt_text,
                analysis="",
                metadata=meta,
            )
    except Exception as e:  # noqa: BLE001
        processor.log_error(
            f"Image DB persist failed for {thread_key} (image WAS posted at {file_url}): {e}",
            exc_info=True)
    # Warm in-memory state: sync path refreshes the breadcrumb URL (+ analysis enrichment)
    # for immediate "edit it" targeting; background path has no breadcrumb, so it just
    # records a metadata-only ledger entry.
    try:
        if generation_id is None:
            await processor.update_last_image_url(channel_id, thread_id, file_url)
        else:
            _update_ledger(thread_manager, thread_id, prompt_text, file_url)
    except Exception as e:  # noqa: BLE001
        processor.log_debug(f"warm-state image update failed: {e}")

    # Participation accounting: only a real posted image on an unprompted channel turn.
    if unprompted and channel_id and not channel_id.startswith("D"):
        pulse = getattr(client, "channel_pulse", None)
        if pulse is not None:
            try:
                pulse.record_bot_reply(channel_id, message_ts, unprompted=True)
            except Exception as e:  # noqa: BLE001
                processor.log_debug(f"participation stat record failed: {e}")

    # Two consumers, one question ("has Slack actually SHARED the file yet?"), one resolve.
    holding = checklist is not None and checklist.surface != "none"
    share_task = _start_share_resolve(
        client, channel_id, upload_meta.get("file_id"),
        wanted=holding or _wants_provenance(provenance_tool))

    try:
        if checklist is not None:
            # Hold "Uploading…" until the image is really on screen (see _wait_for_share).
            if holding:
                await _wait_for_share(processor, share_task, uploaded_at)
            try:
                await checklist.complete(delete_after=4)
            except Exception as e:  # noqa: BLE001
                processor.log_debug(f"checklist completion failed: {e}")
    finally:
        # In a finally because the resolve is SHIELDED: nothing else will stop it, so every
        # exit — including a cancel landing in complete() above — has to hand it to the only
        # code that either owns it or kills it.
        _schedule_image_provenance(
            processor=processor, share_task=share_task, thread_key=thread_key,
            channel_id=channel_id, provenance_tool=provenance_tool)

    return file_url


def _wants_provenance(provenance_tool: Optional[str]) -> TypeGuard[str]:
    """Whether an F7 row is wanted at all. No tool named means the legacy classifier-routed
    path, where attributing the image to a tool would be a fabrication.

    A TypeGuard, not a bool: saying yes is precisely the claim that a tool WAS named, so the
    callers below get that narrowing for free instead of re-testing for it.
    """
    return bool(config.enable_tool_provenance and provenance_tool)


def _start_share_resolve(client, channel_id: Optional[str], file_id: Optional[str],
                         *, wanted: bool):
    """Begin resolving the posted image's own message ts, or None if nobody can/should.

    Started here rather than inside either consumer because BOTH want the answer and it costs
    a multi-second poll: the indicator waits on it to know when the image became visible, and
    F7 provenance needs the ts to key its row. Running it twice would double the API calls to
    learn the same fact.
    """
    resolve = getattr(client, "resolve_file_share_ts", None)  # non-Slack clients lack it
    if not (wanted and resolve is not None and file_id and channel_id):
        return None
    try:
        return asyncio.ensure_future(resolve(channel_id, file_id))
    except Exception:  # noqa: BLE001 — a non-awaitable stand-in must not break delivery
        return None


async def _wait_for_share(processor, share_task, uploaded_at: float) -> None:
    """Keep the progress indicator up until Slack has really shared the image.

    The old cushion was ``delete_after=4`` — a hardcoded guess that has been dead code since
    the setStatus migration anyway (ProgressChecklist.complete only honors delete_after on the
    *message* surface, and the composer status is now the usual one). Measured live 2026-07-16:
    files_upload_v2 returns at ~0.25s but the share message only goes live at ~2.0-2.2s, so
    completing on the upload left the user watching nothing happen for ~2s — every indicator
    gone, no image yet. `shares` populating tracks that moment within ~0.1s, so waiting on it
    ends the indicator on an EVENT ("it's on screen now") instead of a number, and self-tunes
    to a slow day or a big picture instead of being too long for one case and too short for
    another.

    SHIELDED and separately bounded on purpose: this deadline belongs to the indicator, which
    is visible and must never hang, while the resolve keeps its own longer budget for
    provenance, which is invisible and can afford to wait. Expiring here just restores the old
    behavior; it never delays the image, which is already posted.
    """
    if share_task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(share_task),
                               timeout=max(0.0, float(config.image_indicator_hold_seconds)))
        # Logged because this is the only place the real number is observable, and it is the
        # number IMAGE_INDICATOR_HOLD_SECONDS has to cover. Worth keeping: it is measured per
        # image, and image size and Slack's mood both move it.
        processor.log_info(f"Image visible {time.monotonic() - uploaded_at:.2f}s after upload; "
                           "indicator held until then")
    except asyncio.TimeoutError:
        processor.log_info(
            f"Image not visible {time.monotonic() - uploaded_at:.2f}s after upload — completing "
            "the indicator anyway (raise IMAGE_INDICATOR_HOLD_SECONDS if this is common)")
    except asyncio.CancelledError:
        # Shutdown: shield would otherwise leave the resolve running with no awaiter.
        share_task.cancel()
        raise
    except Exception as e:  # noqa: BLE001 — resolve swallows its own errors; belt and braces
        processor.log_debug(f"image share wait failed: {e}")


def _schedule_image_provenance(*, processor, share_task, channel_id: Optional[str],
                               thread_key: str, provenance_tool: Optional[str]) -> None:
    """Fire-and-forget the F7 provenance row for the image message itself.

    Detached because the resolve may still be running: the indicator stops WATCHING it at
    image_indicator_hold_seconds, but provenance is invisible and keeps the full
    image_share_ts_timeout_seconds budget, and publish_image must not stay behind for a DB
    row nobody is looking at. Nothing here can un-post the image: a failed resolve, a failed
    persist, or a client that can't resolve at all just leaves it with the provenance it has
    always had (none).

    With provenance off (or no tool named) this costs zero API calls of its OWN — but note
    that is no longer the same as zero API calls: the indicator holds on the same resolve, and
    it holds whether or not F7 is on, because the visible gap is a UX bug rather than a
    provenance feature.
    """
    if not (_wants_provenance(provenance_tool) and channel_id and share_task is not None):
        # Nobody wants the answer: don't leave the poll running for its full budget. Only
        # reachable when the indicator started it and provenance is off — with both off,
        # _start_share_resolve never made a task at all.
        if share_task is not None:
            share_task.cancel()
        return
    coro = _persist_image_provenance(
        processor=processor, share_task=share_task, channel_id=channel_id,
        thread_key=thread_key, provenance_tool=provenance_tool)
    try:
        processor._schedule_async_call(coro)
    except Exception as e:  # noqa: BLE001
        # Constructed but never scheduled, so nothing will ever await it — close it here or
        # Python warns "coroutine was never awaited" from the GC, out of any useful context.
        coro.close()
        share_task.cancel()
        processor.log_debug(f"image provenance scheduling skipped: {e}")


async def _persist_image_provenance(*, processor, share_task, channel_id: str,
                                    thread_key: str, provenance_tool: str) -> None:
    from message_processor.tool_provenance import build_provenance
    # Not guarded: resolve_file_share_ts is documented never to raise, so if one ever does it
    # is a real defect and belongs in _schedule_async_call's error log, not swallowed here.
    share_ts = await share_task
    if not share_ts:
        return
    # The gist stays empty on purpose: F7 redacts content-bearing arg values anyway, and the
    # prompt is content — it already lives in the images table.
    processor._persist_tool_provenance(
        channel_id, share_ts, thread_key,
        build_provenance([{"name": provenance_tool, "ok": True, "gist": ""}], None))


def _update_ledger(thread_manager, thread_ts: str, prompt: str, file_url: str) -> None:
    """Append a metadata-only ledger entry (no base64 in memory, no second DB write —
    the DB row was already written above)."""
    try:
        ledger = thread_manager.get_or_create_asset_ledger(thread_ts)
        ledger.images.append({
            "data": None,
            "prompt": prompt,
            "timestamp": time.time(),
            "slack_url": file_url,
            "source": "generated",
            "original_url": None,
        })
    except Exception:
        pass
