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


# How far back the DM widening below reaches. A DM has no thread structure to bound "this
# conversation", so time is the only honest boundary — a picture from last week is not what
# "edit that image" means.
DM_LOOKBACK_HOURS = 24


def _entry(row: Dict[str, Any], origin: Optional[str] = None) -> Optional[Dict[str, Any]]:
    row_id, url = row.get("id"), row.get("url")
    if row_id is None or not url:
        return None
    entry = {
        "image_id": image_id_for(row_id),
        "url": url,
        "kind": row.get("image_type") or "image",
        "prompt": row.get("prompt") or "",
        "analysis": row.get("analysis") or "",
        "created_at": row.get("created_at"),
    }
    if origin:
        entry["origin"] = origin
    return entry


async def build_catalog(db, thread_key: str) -> List[Dict[str, Any]]:
    """This thread's images, newest first, capped. Never raises — no catalog just means the
    edit tool is not offered this turn.

    In a DM the thread is widened to the whole conversation (see below): Slack makes every
    top-level DM message its own thread root, so a picture sent as one message and the request
    about it sent as the next land under different keys. Strictly scoped, that meant `edit_image`
    and `view_image` were not even OFFERED on the second message — and the model, left with
    `generate_image` as the only image tool, re-imagined the picture from scratch instead of
    editing it.
    """
    if not db or not thread_key:
        return []
    try:
        rows = await db.find_thread_images_async(thread_key)
    except Exception as e:  # noqa: BLE001 — the turn must survive a catalog failure
        logger.warning(f"Image catalog lookup failed for {thread_key}: {e}")
        return []

    entries: List[Dict[str, Any]] = []
    for row in rows or []:
        entry = _entry(row)
        if entry:
            entries.append(entry)

    # find_thread_images returns oldest-first; the model reasons about "the last one".
    entries.reverse()

    entries.extend(await _dm_widening(db, thread_key, {e["url"] for e in entries},
                                      room=MAX_CATALOG - len(entries)))
    return entries[:MAX_CATALOG]


async def _dm_widening(db, thread_key: str, seen: set, *, room: int) -> List[Dict[str, Any]]:
    """Recent images from elsewhere in the SAME DM, newest first.

    DMs only, and one DM only: the lookup is a prefix match on this channel id, the same
    boundary `read_document`'s channel-wide fallback already uses. It never reaches another
    channel, another DM, or another person — a one-to-one DM is a single conversation that Slack
    happens to split into roots, not a scope the bot is crossing.

    Channels are deliberately left strict. There a thread IS a real conversation boundary, and
    offering images from other threads would be a genuine scope change rather than a repair.
    """
    # Thread keys contain colons on BOTH sides (channel:thread_ts) — split once, from the left.
    channel_id = thread_key.split(":", 1)[0]
    if room <= 0 or not channel_id.startswith("D"):
        return []
    if not hasattr(db, "find_channel_images_async"):
        return []
    try:
        rows = await db.find_channel_images_async(
            channel_id, within_hours=DM_LOOKBACK_HOURS, limit=MAX_CATALOG * 2)
    except Exception as e:  # noqa: BLE001 — a failed widening is just a narrower catalog
        logger.debug(f"DM image widening failed for {channel_id}: {e}")
        return []

    widened: List[Dict[str, Any]] = []
    for row in rows or []:
        if row.get("url") in seen:
            continue
        entry = _entry(row, origin="earlier in this DM")
        if not entry:
            continue
        seen.add(entry["url"])
        widened.append(entry)
        if len(widened) >= room:
            break
    if widened:
        logger.debug(f"DM image catalog widened by {len(widened)} for {channel_id}")
    return widened


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """The human-readable half of the enum — what each id actually is."""
    lines = []
    for i, e in enumerate(entries):
        marker = " (most recent)" if i == 0 else ""
        # Say so when an image came from another message in this DM rather than this exchange —
        # "the image I just sent" and "that chart from earlier" are different requests.
        if e.get("origin"):
            marker += f" [{e['origin']}]"
        lines.append(f"{e['image_id']}{marker} — {e['kind']}: {_describe(e)}")
    return "\n".join(lines)


async def catalog_uploads(processor, thread_key: str, image_inputs: List[Dict[str, Any]],
                          message_ts: Optional[str] = None) -> None:
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
    if not processor.db or not image_inputs:
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

    # Anything that ALREADY has a description does not need a second one.
    #
    # Two writers land in `images.analysis`: this function, and the participation gate's per-image
    # `image_observations` (dual-written by the ambient artifact store). The upsert is
    # merge-preserving — the first non-empty write wins — so when both ran, one description was
    # computed and then silently discarded. The loser was usually this one, and this is the
    # expensive side: the gate's observation rides a classifier call that had to happen anyway,
    # while every call below is a dedicated primary-model vision request. So read first and
    # describe only what is genuinely undescribed.
    described = set()
    try:
        for row in (await processor.db.find_thread_images_async(thread_key)) or []:
            if row.get("url") and (row.get("analysis") or "").strip():
                described.add(row["url"])
    except Exception as e:  # noqa: BLE001 — an unreadable catalog just means we describe again
        logger.debug(f"Could not read existing descriptions for {thread_key}: {e}")

    cataloged = 0
    reused = 0
    for part in image_inputs:
        if not isinstance(part, dict):
            continue
        # An image pulled from a LINK carries `original_url`; only ATTACHMENTS carry `url`
        # (utilities._process_attachments). Reading `url` alone meant every link-borne image was
        # skipped here and never described at all — it entered the catalog as "no description
        # available" and stayed that way.
        url = part.get("url") or part.get("original_url")
        if not url:
            continue
        if url in described:
            reused += 1
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
            # Guard the rest of THIS call too: the same url can appear twice in one turn's parts
            # (an attachment that is also linked in the text).
            described.add(url)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to persist catalog entry for {url}: {e}")
    if cataloged or reused:
        logger.info(f"Cataloged {cataloged} uploaded image(s) for {thread_key}"
                    + (f" ({reused} already described — no vision call spent)" if reused else ""))


def resolve(entries: Optional[List[Dict[str, Any]]], image_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an id against THIS TURN's snapshot. A syntactically valid id is not
    authorization: only ids we put in front of the model resolve."""
    for entry in entries or []:
        if entry.get("image_id") == image_id:
            return entry
    return None


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["image_id"] for e in (entries or []) if e.get("image_id")]
