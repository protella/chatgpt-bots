"""This conversation's image catalog (F34).

Editing used to work by asking a utility model "which of these images did the user mean?"
and, failing that, silently defaulting to the most recent one. Editing the wrong image is an
expensive, irreversible side effect, so that guess is gone. Instead every image the conversation
knows about gets a stable opaque id (``img_<row id>``), the ids are put in front of the model in
the tool descriptions and the turn's evidence, and the EXECUTOR re-validates the chosen id before
touching anything. That executor check is the authorization — the edit schemas no longer carry an
id enum (see `image_tools`) — so an id the model invented cannot resolve, and neither can one
belonging to another CHANNEL.

The snapshot reaches past the current root — the whole DM, or the whole channel — because the
conversation does (see ``build_catalog``). A channel is a shared workspace, so an image someone
else posted in it is as editable as one from this exchange [OWNER 2026-09-16]: "edit that chart
someone posted" is what a colleague does there, not a boundary to defend.

``MAX_CATALOG`` bounds this ADVERTISED list and nothing else. What is REACHABLE in the channel is
the index: ``search_stored_knowledge`` finds any image description in the channel and hands back
an ``img_*`` id, which ``view_image`` resolves by a channel-scoped lookup whether or not the cap
let it into the list (``image_view._resolve_in_channel``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from logger import setup_logger
from message_processor.tool_registry import SURFACE_CHANNEL

logger = setup_logger(name="slack_bot.ImageCatalog")

# How many images the model may choose among. The newest are the ones anyone refers to;
# a 200-image thread must not blow the tool schema up.
MAX_CATALOG = 12

# Longest description we put next to an id. Enough to disambiguate, not a wall of text.
_DESC_CHARS = 110

# The kinds whose `prompt` column holds an ENHANCED generation prompt worth reusing verbatim.
# An uploaded (or imported) image has no prompt of ours to reuse.
_PROMPTED_KINDS = ("generated", "edited")


_ID_PREFIX = "img_"


def image_id_for(row_id: Any) -> str:
    return f"{_ID_PREFIX}{row_id}"


def row_id_for(image_id: Any) -> Optional[int]:
    """The `images.id` behind an `img_<n>` handle, or None if it is not one.

    `image_id_for`'s inverse, and STRICT on purpose: only the exact lowercase prefix followed by
    digits. It is what turns a handle the model emitted into a database lookup, so `IMG_7`,
    `img_3x` and `../img_7` must all come back None rather than nearly-parsing into some row.
    """
    text = str(image_id or "").strip()
    if not text.startswith(_ID_PREFIX):
        return None
    digits = text[len(_ID_PREFIX):]
    if not (digits.isascii() and digits.isdigit()):
        return None
    try:
        return int(digits)
    except ValueError:  # pragma: no cover — isdigit() already guarantees this parses
        return None


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
DM_LOOKBACK_HOURS = 24 * 7  # one week [OWNER 2026-09-09]; was 24h, which cut off a same-day build's inputs

# What a widened row is labelled as, per surface. The label is the whole difference — a widened
# row is an ordinary catalog entry, editable like any other (ruling 10).
_CHANNEL_ORIGIN = "elsewhere in this channel"
_DM_ORIGIN = "earlier in this DM"


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


def entry_from_row(row: Dict[str, Any],
                   origin: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A catalog-shaped entry for a row fetched OUTSIDE `build_catalog`.

    `view_image` resolves an id the turn's list does not carry by going to the channel's rows
    directly (ruling 12), and what it gets back has to be the same shape everything downstream
    already reads — so the shape stays defined in exactly one place. None for a row with no id
    or no url, on the same terms as a row inside the list.
    """
    return _entry(row, origin=origin)


def _recency_key(row: Dict[str, Any]) -> Tuple[str, int]:
    """Newest-last ordering key for a raw image row.

    `images.created_at` has SECOND precision, so two images from the same second tie and the
    order between them would be whatever the merge happened to produce — which is exactly the
    pair a user means by "the last one". The autoincrement id breaks the tie, and it is the
    honest insertion order (both DB queries carry the same `id` tie-break).
    """
    try:
        row_id = int(row.get("id") or 0)
    except (TypeError, ValueError):
        row_id = 0
    return (str(row.get("created_at") or ""), row_id)


async def build_catalog(db, thread_key: str, *, surface: str) -> List[Dict[str, Any]]:
    """This conversation's images, newest first, capped. Never raises — no catalog just means
    the edit tool is not offered this turn.

    The current root is never the whole story, and the reason differs by surface, so the caller
    passes the surface rather than having it guessed from the key (ruling 2):

    * DM — Slack makes every top-level DM message its own thread root, so a picture sent as one
      message and the request about it sent as the next land under different keys. Strictly
      scoped, that meant `edit_image` and `view_image` were not even OFFERED on the second
      message, and the model, left with `generate_image` as the only image tool, re-imagined the
      picture from scratch instead of editing it.
    * CHANNEL — a screenshot pasted as its own channel message is captured by ambient memory and
      rendered into the channel stream as `[image analysis (ref): …]`, so the model can SEE that
      it exists while a strictly-scoped catalog left `view_image` unable to resolve it. Asked
      about it from another root, the bot said it could not reopen the image and web-searched
      for the source instead.

    Thread rows and widened rows are merged into ONE recency order BEFORE the cap (ruling 3),
    so index 0 — the entry `catalog_lines` labels "(most recent)" — really is the newest image.
    """
    if not db or not thread_key:
        return []
    try:
        thread_rows = list(await db.find_thread_images_async(thread_key) or [])
    except Exception as e:  # noqa: BLE001 — the turn must survive a catalog failure
        logger.warning(f"Image catalog lookup failed for {thread_key}: {e}")
        return []

    # (row, origin). The current root carries no origin qualifier at all (ruling 5).
    merged: List[Tuple[Dict[str, Any], Optional[str]]] = [(row, None) for row in thread_rows]
    seen: Set[str] = {row["url"] for row in thread_rows if row.get("url")}
    merged.extend(await _widening(db, thread_key, seen, surface=surface))

    # One global order, newest first, and only THEN the cap.
    merged.sort(key=lambda item: _recency_key(item[0]), reverse=True)

    entries: List[Dict[str, Any]] = []
    for row, origin in merged:
        entry = _entry(row, origin=origin)
        if entry:
            entries.append(entry)
        if len(entries) >= MAX_CATALOG:
            break
    return entries


async def _widening(db, thread_key: str, seen: Set[str], *, surface: str
                    ) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Rows from elsewhere in the SAME conversation, as (row, origin) pairs.

    One channel only: the lookup is a prefix match on this channel id, the same boundary
    `read_document`'s channel-wide fallback already uses. It never reaches another channel,
    another DM, or another person.

    A channel is unbounded in time (ruling 1): `MAX_CATALOG` is the only bound, because a quiet
    channel's three-week-old screenshot is still the one being asked about. A DM keeps its own
    time bound, which is a separate owner decision about what "edit that image" means in an
    unthreaded DM.
    """
    # Thread keys contain colons on BOTH sides (channel:thread_ts) — split once, from the left.
    channel_id = thread_key.split(":", 1)[0]
    if not channel_id or not hasattr(db, "find_channel_images_async"):
        return []
    if surface == SURFACE_CHANNEL:
        origin, within_hours = _CHANNEL_ORIGIN, None
    else:
        origin, within_hours = _DM_ORIGIN, DM_LOOKBACK_HOURS
    try:
        rows = await db.find_channel_images_async(
            channel_id, within_hours=within_hours, limit=MAX_CATALOG * 2)
    except Exception as e:  # noqa: BLE001 — a failed widening is just a narrower catalog
        logger.debug(f"Image catalog widening failed for {channel_id}: {e}")
        return []

    widened: List[Tuple[Dict[str, Any], Optional[str]]] = []
    for row in rows or []:
        url = row.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        widened.append((row, origin))
    if widened:
        logger.debug(f"Image catalog widened by {len(widened)} for {channel_id} ({surface})")
    return widened


def _reusable_prompt(entry: Dict[str, Any]) -> str:
    """The full enhanced prompt behind a generated/edited image, collapsed to ONE line.

    "Generate that again with the same prompt" was impossible: the enhanced prompt lives in
    `images.prompt`, rebuilt history redacts tool arguments, and the catalog line showed only the
    110-char description. So the model had nothing to reuse and re-enhanced from scratch, getting
    a different picture. It is carried in full — a paraphrase is not the same prompt.

    Single-line is a hard requirement, not tidiness: `catalog_evidence_lines` splits
    `catalog_lines` on newlines and would otherwise scatter one image across several entries.

    Uncapped, deliberately. A 2000-char ceiling used to guard against a pathological row, but a
    truncated prompt is not the same prompt — "generate that again" would silently re-enhance
    the missing tail and hand back a different picture, which is the exact failure this line
    exists to stop.
    """
    if (entry.get("kind") or "") not in _PROMPTED_KINDS:
        return ""
    text = " ".join((entry.get("prompt") or "").split())
    if not text:
        return ""
    return text


def catalog_lines(entries: List[Dict[str, Any]]) -> str:
    """The human-readable half of the enum — what each id actually is."""
    lines = []
    for i, e in enumerate(entries):
        marker = " (most recent)" if i == 0 else ""
        # Say so when an image came from another message in this conversation rather than this
        # exchange — "the image I just sent" and "that chart from earlier" are different
        # requests. Ruling 5: the current root carries no qualifier at all.
        if e.get("origin"):
            marker += f" [{e['origin']}]"
        line = f"{e['image_id']}{marker} — {e['kind']}: {_describe(e)}"
        prompt = _reusable_prompt(e)
        if prompt:
            line += (' · generation prompt (reuse verbatim for "same prompt again"; edits '
                     f'describe only the change): "{prompt}"')
        lines.append(line)
    return "\n".join(lines)


# Scope-neutral (ruling 5): the catalog reaches the whole channel now, so "this thread" was a
# lie the model could read as a boundary on what it may name.
EVIDENCE_HEADER = "Images in this conversation (edit_image / view_image ids):"

# Said wherever the list is shown, because a list presented as complete is how a model decides an
# image it cannot see does not exist — and then web-searches for a screenshot that is sitting in
# the channel [OWNER 2026-09-16]. `MAX_CATALOG` may bound this list only on the condition that the
# list says so. ONE LINE: `catalog_evidence_lines` splits on newlines.
INDEX_NOTE = ("Only the most recent images are listed — older ones anywhere in this channel are "
              "not, and are found with search_stored_knowledge, whose image_id works here as an "
              "id like any other.")


def catalog_evidence_lines(entries: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """The image section of a channel turn's tool-evidence block, one entry per line.

    Same text the edit/view schemas used to carry, moved out of the cached prefix. An empty
    catalog is stated rather than omitted: the tools are on the channel surface either way, and
    "there are none" is the fact that stops the model naming one — paired with `INDEX_NOTE`, so
    "none recently" is never read as "none in this channel".
    """
    entries = entries or []
    if not entries:
        return [EVIDENCE_HEADER, "(none)", INDEX_NOTE]
    return [EVIDENCE_HEADER] + catalog_lines(entries).split("\n") + [INDEX_NOTE]


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
    from message_processor.prompts import IMAGE_ANALYSIS_PROMPT

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


async def resolve_in_channel(ctx: Any, image_id: str) -> Optional[Dict[str, Any]]:
    """An id the turn's catalog does not carry, looked up among THIS channel's image rows.

    `MAX_CATALOG` caps the list the tools advertise. Letting it also cap what is REACHABLE was
    never a reviewed decision [OWNER 2026-09-16]: `search_stored_knowledge` indexes every image
    description in the channel and already finds the three-week-old screenshot, and the handle it
    returned used to be useless — its own docstring said so, and the reason it gave was the
    `view_image` executor. So the cap bounds convenience only.

    Shared by `view_image` and the edit path, which reach it under DIFFERENT authority and that
    difference is not here: `view_image` calls this for any id (a wrong view costs a round), while
    the edit executors call it only for an id the turn's catalog or a search this turn handed the
    model (a wrong edit posts a wrong picture publicly and irreversibly). This function is the
    channel BOUNDARY, not the permission.

    `get_channel_image_by_id_async` carries `find_channel_images_async`'s boundary verbatim — the
    row's thread key must start with this channel's id — so this reaches other threads in this one
    channel and nothing else. Another channel's image, another DM's image and an id that never
    existed are all the same answer: None.

    Never raises: a lookup failure is an unresolved id, not a failed turn.
    """
    row_id = row_id_for(image_id)
    channel_id = (getattr(ctx, "channel_id", None) or "").strip()
    db = getattr(getattr(ctx, "processor", None), "db", None)
    if row_id is None or not channel_id or db is None:
        return None
    lookup = getattr(db, "get_channel_image_by_id_async", None)
    if not callable(lookup):
        return None
    try:
        row = await lookup(channel_id, row_id)
    except Exception as e:  # noqa: BLE001 — an unresolvable id is a tool result, not a turn's end
        logger.warning(f"Channel-wide image lookup failed for {image_id}: {e}")
        return None
    if not row:
        return None
    # No origin label: the entry is used for its url right now and never rendered in a list, and
    # the row may as easily be one the cap dropped from THIS root as one from another thread.
    entry = entry_from_row(row)
    if entry:
        logger.info(f"Resolved {image_id} outside this turn's catalog ({channel_id}) — "
                    "from the channel's image rows")
    return entry


def seen_in_search(ctx: Any, image_id: str) -> bool:
    """Whether a `search_stored_knowledge` result this TURN put this id in front of the model.

    The edit path's widening (ruling 15) rests entirely on this: an id the model READ is not an
    id it invented, and an invented `img_N` that happened to resolve would post a wrong edited
    picture. Recorded by `knowledge_tools` from the hits it actually returns.
    """
    return bool(image_id) and image_id in (getattr(ctx, "searched_image_ids", None) or ())


def record_searched_ids(ctx: Any, image_ids: List[str]) -> None:
    """Remember the ids a search just showed the model, for `seen_in_search`.

    Appends to the SHARED list the registry installs (`_SHARED_CONTAINERS`) so a search in one
    round authorizes an edit in the next; installs one only if there is none, because assigning
    into a per-call copy would strand the record on a context nobody reads again. Never raises:
    failing to record costs a later refusal, not this search's result.
    """
    wanted = [i for i in image_ids if i]
    if not wanted:
        return
    try:
        seen = getattr(ctx, "searched_image_ids", None)
        if seen is None:
            seen = []
            ctx.searched_image_ids = seen
        for image_id in wanted:
            if image_id not in seen:
                seen.append(image_id)
    except Exception as e:  # noqa: BLE001 — a search result must survive a bookkeeping failure
        logger.debug(f"Could not record searched image ids: {e}")


def resolve(entries: Optional[List[Dict[str, Any]]], image_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an id against THIS TURN's snapshot. A syntactically valid id is not
    authorization: only ids we put in front of the model resolve."""
    for entry in entries or []:
        if entry.get("image_id") == image_id:
            return entry
    return None


def valid_ids(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [e["image_id"] for e in (entries or []) if e.get("image_id")]
