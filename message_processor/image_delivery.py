from __future__ import annotations

import time
from typing import Optional

from message_processor.progress import ProgressChecklist


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
    """
    if checklist is not None:
        try:
            await checklist.step("Uploading…", done_text="Uploaded")
        except Exception as e:  # noqa: BLE001 — a status hiccup must not abort delivery
            processor.log_debug(f"checklist upload step failed: {e}")

    # 1) Upload. A returned URL means the image is POSTED — nothing below may flip it back.
    file_url: Optional[str] = None
    try:
        file_url = await client.send_image(
            channel_id,
            thread_id,
            image_data.to_bytes(),
            f"generated_image.{image_data.format}",
            "",
        )
    except Exception as e:  # noqa: BLE001
        processor.log_error(f"Image upload failed for {thread_key}: {e}", exc_info=True)
        file_url = None

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

    if checklist is not None:
        try:
            await checklist.complete(delete_after=4)
        except Exception as e:  # noqa: BLE001
            processor.log_debug(f"checklist completion failed: {e}")

    return file_url


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
