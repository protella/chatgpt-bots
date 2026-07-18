"""The thread's editable-image catalog (F34).

Editing used to work by asking a utility model "which of these images did the user mean?"
and, failing that, silently defaulting to the most recent one. Editing the wrong image is an
expensive, irreversible side effect, so that guess is gone. Instead every image the thread
knows about gets a stable opaque id (``img_<row id>``), the ids are advertised to the model
as a literal enum on the edit tool, and the executor re-validates the chosen id against this
turn's snapshot before touching anything. An id the model invents cannot resolve, and an id
belonging to another thread cannot either.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from logger import setup_logger

logger = setup_logger(name="slack_bot.ImageCatalog")

# How many images the model may choose among. The newest are the ones anyone refers to;
# a 200-image thread must not blow the tool schema up.
MAX_CATALOG = 12

# Longest description we put next to an id. Enough to disambiguate, not a wall of text.
_DESC_CHARS = 110


def image_id_for(row_id: Any) -> str:
    return f"img_{row_id}"


def _describe(entry: Dict[str, Any]) -> str:
    """The blurb next to an id. Prefer what the image IS (its analysis) over what was asked
    for (its prompt) — an uploaded image has no prompt, and a generated one's prompt is the
    enhanced text, which is long and reads nothing like the picture."""
    text = (entry.get("analysis") or entry.get("prompt") or "").strip()
    text = " ".join(text.split())
    if not text:
        return "no description available"
    return text[:_DESC_CHARS] + ("…" if len(text) > _DESC_CHARS else "")


async def build_catalog(db, thread_key: str) -> List[Dict[str, Any]]:
    """This thread's images, newest first, capped. Never raises — no catalog just means the
    edit tool is not offered this turn."""
    if not db or not thread_key:
        return []
    try:
        rows = await db.find_thread_images_async(thread_key)
    except Exception as e:  # noqa: BLE001 — the turn must survive a catalog failure
        logger.warning(f"Image catalog lookup failed for {thread_key}: {e}")
        return []

    entries: List[Dict[str, Any]] = []
    for row in rows or []:
        row_id, url = row.get("id"), row.get("url")
        if row_id is None or not url:
            continue
        entries.append({
            "image_id": image_id_for(row_id),
            "url": url,
            "kind": row.get("image_type") or "image",
            "prompt": row.get("prompt") or "",
            "analysis": row.get("analysis") or "",
            "created_at": row.get("created_at"),
        })

    # find_thread_images returns oldest-first; the model reasons about "the last one".
    entries.reverse()
    return entries[:MAX_CATALOG]


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """The human-readable half of the enum — what each id actually is."""
    lines = []
    for i, e in enumerate(entries):
        marker = " (most recent)" if i == 0 else ""
        lines.append(f"{e['image_id']}{marker} — {e['kind']}: {_describe(e)}")
    return "\n".join(lines)


async def catalog_uploads(processor, thread_key: str, attachments: List[Dict[str, Any]],
                          image_inputs: List[str], message_ts: Optional[str] = None) -> None:
    """Store a canonical visual description for each image the user just uploaded.

    This is the one genuinely load-bearing thing the old vision handler did, and it survives
    the classifier's removal — but as a background side effect, not a routing decision. The
    main model already SEES the uploaded images (they ride the turn as input_image parts), so
    answering the user needs no vision round-trip. What it cannot do is remember them: a
    later "edit the screenshot I sent" or "what was in that chart?" needs a durable
    description, and the thread's rebuilt-from-Slack transcript carries only a URL.

    It describes what the image IS, not what was asked about it. The old handler stored the
    model's ANSWER as the analysis ("yes, the total is wrong"), which is useless as an edit
    source later.

    Never raises: a failed description costs a weaker catalog entry, not the turn.
    """
    if not processor.db or not attachments or not image_inputs:
        return

    # ONE description PER image, keyed by that image's own url. A single aggregate call over all
    # uploads returned ONE blurb that was then saved as the analysis of EVERY image — so three
    # uploaded screenshots became three IDENTICAL catalog entries, and a later "edit the second
    # one" had nothing to disambiguate on (and could edit the wrong picture, the exact expensive,
    # irreversible mistake this catalog exists to prevent). Describing each image on its own keeps
    # edit-target resolution unambiguous.
    #
    # Each part carries its own Slack url (utilities._process_attachments stores it on the part),
    # so we key off that rather than a separately-built url list — the two can drift when an image
    # is skipped (oversized/undecodable) and is absent from image_inputs but present in the url
    # list, which would misattribute descriptions.
    from prompts import IMAGE_ANALYSIS_PROMPT

    cataloged = 0
    for part in image_inputs:
        url = part.get("url") if isinstance(part, dict) else None
        if not url:
            continue
        try:
            description = await processor.openai_client.analyze_images(
                images=[part],
                question=IMAGE_ANALYSIS_PROMPT,
                enhance_prompt=False,
            )
        except Exception as e:  # noqa: BLE001 — a failed description costs an entry, not the turn
            logger.warning(f"Upload cataloging failed for {thread_key} ({url}): {e}")
            continue
        if not description:
            continue
        try:
            await processor.db.save_image_metadata_async(
                thread_id=thread_key,
                url=url,
                image_type="uploaded",
                prompt="",
                analysis=description,
                metadata={"cataloged": True},
                message_ts=message_ts,
            )
            cataloged += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to persist catalog entry for {url}: {e}")
    if cataloged:
        logger.info(f"Cataloged {cataloged} uploaded image(s) for {thread_key}")


def resolve(entries: Optional[List[Dict[str, Any]]], image_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an id against THIS TURN's snapshot. A syntactically valid id is not
    authorization: only ids we put in front of the model resolve."""
    for entry in entries or []:
        if entry.get("image_id") == image_id:
            return entry
    return None


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["image_id"] for e in (entries or []) if e.get("image_id")]
