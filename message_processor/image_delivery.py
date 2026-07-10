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
) -> Optional[str]:
    """Single owner of image delivery for both the background job and the config-off
    sync path: checklist "Uploading…" transition, upload, falsey-URL = failure,
    persistence, asset-ledger update, checklist completion, and unprompted participation
    accounting. Returns the file URL, or None on failure.

    Persistence splits on generation_id: the background path (id set) has no in-memory
    breadcrumb, so it writes the DB row directly; the legacy sync path (id None) keeps
    today's breadcrumb-driven update_last_image_url (which also writes the DB row).

    The upload latch is NOT released here — its lifecycle is owned by the caller
    (generation-ID-conditional in the background job's finally; main.py's finally on the
    sync path) so a watchdog-cleared zombie can't signal a newer job's upload done.
    """
    if checklist is not None:
        try:
            await checklist.step("Uploading…", done_text="Uploaded")
        except Exception as e:  # noqa: BLE001 — a status hiccup must not abort delivery
            processor.log_debug(f"checklist upload step failed: {e}")

    file_url: Optional[str] = None
    try:
        file_url = await client.send_image(
            channel_id,
            thread_id,
            image_data.to_bytes(),
            f"generated_image.{image_data.format}",
            "",
        )
        if file_url:
            if generation_id is None:
                # Legacy sync path — breadcrumb exists; warm state + DB via the helper.
                await processor.update_last_image_url(channel_id, thread_id, file_url)
            else:
                # Background path — no breadcrumb; write the DB row directly (must not
                # depend on finding a mutable in-memory breadcrumb) and update the ledger
                # in memory only (a persisting ledger upsert would fight this write).
                if db:
                    await db.save_image_metadata_async(
                        thread_id=thread_key,
                        url=file_url,
                        image_type="generated",
                        prompt=prompt,
                        analysis="",
                        metadata={"generation_id": generation_id, "timestamp": time.time()},
                    )
                _update_ledger(thread_manager, thread_id, prompt, file_url)
    except Exception as e:  # noqa: BLE001
        processor.log_error(f"Image upload/persist failed for {thread_key}: {e}", exc_info=True)
        file_url = None

    if not file_url:
        if checklist is not None:
            try:
                await checklist.fail("Upload failed")
            except Exception:
                pass
        return None

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
