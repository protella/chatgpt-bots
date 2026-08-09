"""The ONE canonical Slack normalizer (spec §2).

Every consumer that used to read raw Slack dicts — the stream fetch, the live activity index,
the actor tail, the coverage sweep's parent hints, ambient artifact attribution and the
channel-surface history tool — reads these frozen records instead. One skip-set, one sender
classification, one timestamp comparator.

Self-message admission is deliberately NOT decided here. The normalizer REPRESENTS our own
messages and never returns None for them, because the consumers disagree about what to do with
one: the admission watermark, the live index and the actor tail exclude self; the coverage
sweep's parent hints include self roots; the stream includes self and lets the receipt rule
decide the ROLE at serialization time. A normalizer that filtered self would make three of
those five wrong and there would be no way to tell which.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import config
from slack_client.formatting.blocks import extract_supplementary_text
from slack_client.formatting.text import extract_mention_ids
from slack_client.utilities import is_dm_conversation

logger = logging.getLogger(__name__)

TOMBSTONE_TEXT = "this message was deleted."

# Subtypes that are not semantic messages: membership churn, topic churn, and Slack's
# reply-count notification. `tombstone` is deliberately ABSENT — a deleted root is evidence
# about the room, so it is represented (is_tombstone) and the serializer renders it.
# `message_changed`/`message_deleted` are envelopes, not messages: they are classified by
# normalize_slack_event and never reach normalize_slack_message.
SKIP_SUBTYPES = frozenset({
    "message_replied",
    "channel_join", "channel_leave", "channel_topic", "channel_purpose",
    "channel_name", "channel_archive", "channel_unarchive",
    "group_join", "group_leave", "bot_add", "bot_remove",
    "reminder_add", "pinned_item", "unpinned_item",
})

# Envelope subtypes: the subject is nested, not the payload itself.
MUTATION_SUBTYPES = frozenset({"message_changed", "message_deleted"})

ORIGIN_HISTORY = "history"
ORIGIN_REPLIES = "replies"
ORIGIN_LIVE = "live"

KIND_MESSAGE = "message"
KIND_EDIT = "edit"
KIND_DELETE = "delete"
KIND_TOMBSTONE = "tombstone"

# Every key Slack uses to name a sender. A deletion payload carrying none of them tells us
# nothing about who deleted what, which is what sends the caller to the receipt ledger.
_IDENTITY_KEYS = ("user", "bot_id", "app_id", "api_app_id")

_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(\|[^>]*)?>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class TimestampError(ValueError):
    """A Slack ts that cannot be parsed. Every turn-path caller fails closed on it."""


class MalformedEventError(ValueError):
    """A KNOWN event kind whose subject cannot be read.

    The distinction this carries is the whole point: an UNKNOWN or declined kind returns None and
    is harmless (a join notice records nothing and must not take a channel out of service), while
    an edit or a deletion we cannot resolve is an event that HAPPENED. Its outer `event_ts` has
    already advanced H, so completing its readiness ticket successfully would tell every turn in
    that window that the index is caught up on a mutation nobody indexed.
    """


def parse_ts(raw: Any) -> Tuple[int, int]:
    """A Slack ts as (seconds, microseconds) — the shared comparator's key.

    Integer fields, never a float: `1752600000.000001` and `1752600000.000002` are distinct
    messages and comparing them as floats at the window boundary decides inclusion by rounding.
    The seconds field is also what renders the header's minute stamp, for the same reason.
    """
    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise TimestampError("empty timestamp")
    whole, _, frac = text.partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise TimestampError(f"unparseable timestamp: {raw!r}")
    return int(whole), int((frac + "000000")[:6]) if frac else 0


_ABSENT = object()


def secondary_ts(payload: Dict[str, Any], key: str) -> Optional[str]:
    """A ts field that is ALLOWED to be missing (`thread_ts`, `latest_reply`) — validated when it
    is not. Returns the value as a string, or None when the field is genuinely absent.

    Absent, or explicitly null, means absent. Present-and-falsey does not: `""`, `0` and `false`
    are values Slack put in the field, and reading them as "no thread" is how a reply gets rendered
    as a top-level message and how a coverage sweep advances past a thread it never recorded. So
    they raise, and every caller already fails closed on that.
    """
    raw = payload.get(key, _ABSENT)
    if raw is _ABSENT or raw is None:
        return None
    if not raw:
        raise TimestampError(f"{key} is present but empty: {raw!r}")
    parse_ts(raw)
    return str(raw)


def ts_key(raw: Any) -> Tuple[int, int]:
    """Sort/compare key. Alias of parse_ts, named for how it reads at call sites."""
    return parse_ts(raw)


def ts_lt(a: Any, b: Any) -> bool:
    return parse_ts(a) < parse_ts(b)


def ts_le(a: Any, b: Any) -> bool:
    return parse_ts(a) <= parse_ts(b)


def ts_max(a: Optional[Any], b: Optional[Any]) -> Optional[str]:
    """Numeric max of two ts strings; either may be None."""
    if a is None:
        return None if b is None else str(b)
    if b is None:
        return str(a)
    return str(a) if parse_ts(a) >= parse_ts(b) else str(b)


def in_window(ts: Any, floor_ts: Any, floor_inclusive: bool, high: Any) -> bool:
    """The window predicate: (floor_ts, floor_inclusive) lower bound, ts <= H upper bound."""
    value = parse_ts(ts)
    low = parse_ts(floor_ts)
    if value < low or (value == low and not floor_inclusive):
        return False
    return value <= parse_ts(high)


def sanitize_field(text: Any) -> str:
    """Control-strip a rendered field. CRLF/CR fold to \\n; other control chars go."""
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", value)


def sanitize_name(text: Any) -> str:
    """A name/snippet/filename: control-stripped, then bracket- and quote-free so it cannot
    forge a header or a marker line."""
    value = sanitize_field(text).replace("\n", " ")
    for ch in ("[", "]", '"'):
        value = value.replace(ch, "")
    return value.strip()


@dataclass(frozen=True)
class FileRef:
    id: Optional[str]
    name: str
    mimetype: str
    size: Optional[int]
    url_private: Optional[str]
    kind: str  # "image" | "file"


@dataclass(frozen=True)
class ReactionRec:
    name: str
    count: int
    mine: bool = False


@dataclass(frozen=True)
class NormalizedMessage:
    team_id: str
    channel_id: str
    ts: str
    thread_root_ts: Optional[str]
    subtype: Optional[str]
    sender_id: Optional[str]
    sender_type: str  # "human" | "other_bot" | "self"
    raw_bot_name: Optional[str]
    text: str
    files: Tuple[FileRef, ...]
    reactions: Tuple[ReactionRec, ...]
    edited_ts: Optional[str]
    is_broadcast: bool
    is_tombstone: bool
    reply_count: Optional[int]
    latest_reply: Optional[str]
    mention_ids: Tuple[str, ...]
    origin: str

    @property
    def is_reply(self) -> bool:
        return bool(self.thread_root_ts) and self.thread_root_ts != self.ts

    @property
    def root_ts(self) -> str:
        """The thread this message belongs to — itself when it is the root."""
        return self.thread_root_ts or self.ts


@dataclass(frozen=True)
class NormalizedEvent:
    kind: str  # message | edit | delete | tombstone
    team_id: str
    channel_id: str
    subject_ts: str
    activity_ts: Optional[str]
    root_if_indexed: bool
    owner_probe_ts: Optional[str]
    deleted_ts: Optional[str]
    message: Optional[NormalizedMessage]


def _sender_type(client: Any, payload: Dict[str, Any]) -> str:
    """Classification, fail-open. An identity that is not wired yet must not cost the read: a
    normalizer that raised here would turn a history fetch into an error dict."""
    classify = getattr(client, "classify_sender", None)
    if callable(classify):
        try:
            return classify(payload) or "human"
        except Exception:  # noqa: BLE001
            pass
    own = getattr(client, "is_own_message", None)
    if callable(own):
        try:
            if own(payload):
                return "self"
        except Exception:  # noqa: BLE001
            pass
    if payload.get("bot_id") or payload.get("app_id") or payload.get("api_app_id"):
        if str(payload.get("bot_id") or "") in (config.dev_treat_bot_ids_as_human or []):
            return "human"
        return "other_bot"
    return "human"


def _file_refs(payload: Dict[str, Any]) -> Tuple[FileRef, ...]:
    refs: List[FileRef] = []
    for entry in payload.get("files") or []:
        if not isinstance(entry, dict):
            continue
        mimetype = str(entry.get("mimetype") or "")
        size = entry.get("size")
        refs.append(FileRef(
            id=str(entry["id"]) if entry.get("id") else None,
            name=str(entry.get("name") or entry.get("title") or "file"),
            mimetype=mimetype,
            size=int(size) if isinstance(size, int) else None,
            url_private=str(entry["url_private"]) if entry.get("url_private") else None,
            kind="image" if mimetype.startswith("image/") else "file",
        ))
    return tuple(refs)


def _reactions(payload: Dict[str, Any], self_user_id: Optional[str]) -> Tuple[ReactionRec, ...]:
    """`mine` is a strict-shape membership test, fail-closed on identity.

    Slack puts the authenticated user in a reaction's `users` array whenever that user reacted,
    even when other reactor ids are truncated — so `count > len(users)` with us present is still
    `mine=True`, and our absence is a genuine non-reaction rather than a truncation. A `users`
    that is not a list/tuple is never tested, so a string cannot match by substring nor a dict by
    key. Without `self_user_id` every reaction is `mine=False`; absent identity never raises.
    """
    recs: List[ReactionRec] = []
    for entry in payload.get("reactions") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        count = entry.get("count")
        users = entry.get("users")
        if not isinstance(count, int):
            count = len(users or [])
        mine = False
        if isinstance(users, (list, tuple)):
            mine = bool(self_user_id) and any(
                u == self_user_id for u in users if isinstance(u, str))
        recs.append(ReactionRec(name=str(entry["name"]), count=int(count), mine=mine))
    return tuple(recs)


def _bot_name(payload: Dict[str, Any]) -> Optional[str]:
    profile = payload.get("bot_profile")
    raw = payload.get("username") or (profile.get("name") if isinstance(profile, dict) else None)
    name = sanitize_name(raw) if raw else ""
    return name or None


def canonical_sender_id(client: Any, payload: Dict[str, Any]) -> Optional[str]:
    """Who posted this, as an id — preferring one a mention can actually name.

    A peer app in "agent mode" posts with no `user` field at all, so its only ids are the B
    (bot OBJECT) and A (app) ones. Those used to become the actor id verbatim, and a B id offered
    to the model as a mention target renders as raw `<@B…>` text in the channel. When the client
    has already resolved that bot's USER id (bots.info, primed by the async fetch seams) the actor
    is named by that instead, so the same app reads identically whichever mode it posted in.

    The B id remains the fallback: an actor we cannot name properly is still an actor, and losing
    it would be worse than naming it awkwardly. A client with no cache at all (a test double, a
    non-Slack caller) keeps the original behavior exactly."""
    user = payload.get("user")
    if user:
        return str(user)
    bot_id = payload.get("bot_id")
    if bot_id:
        lookup = getattr(client, "bot_user_id_for", None)
        if callable(lookup):
            try:
                resolved = lookup(bot_id)
            except Exception:  # noqa: BLE001 — a cache peek must never break normalization
                resolved = None
            if isinstance(resolved, str) and resolved:
                return resolved
        return str(bot_id)
    app_id = payload.get("app_id")
    return str(app_id) if app_id else None


async def prime_bot_actor_ids(client: Any, payloads: Iterable[Any]) -> None:
    """THE ASYNC HALF of `canonical_sender_id`, for seams that fetch before they normalize.

    `normalize_slack_message` is sync and can only read a cache someone else filled. This is the
    call that fills it: hand it the raw payloads about to be normalized and every peer bot in the
    batch that named itself only by a B id gets resolved once, for the process.

    Never raises and never fails a fetch — a client without the cache (a test double, a non-Slack
    caller) is a no-op, and a bots.info that errors leaves the batch normalizing exactly as it did
    before this existed."""
    primer = getattr(client, "prime_bot_user_ids", None)
    if not callable(primer):
        return
    try:
        await primer(payloads)
    except Exception as e:  # noqa: BLE001 — priming is an optimization, never a precondition
        logger.debug(f"bot actor id priming skipped: {e}")


def _is_tombstone(payload: Dict[str, Any]) -> bool:
    if payload.get("subtype") == "tombstone":
        return True
    return sanitize_field(payload.get("text")).strip().lower() == TOMBSTONE_TEXT


def normalize_slack_message(client: Any, payload: Any, *, channel_id: Optional[str] = None,
                            origin: str = ORIGIN_HISTORY,
                            team_id: Optional[str] = None,
                            allow_any_subtype: bool = False) -> Optional[NormalizedMessage]:
    """One Slack message dict → NormalizedMessage, or None when the subtype is not a message.

    Never returns None for our own messages (see the module docstring). Raises TimestampError on a
    malformed OR ABSENT ts, so a turn-path caller fails closed instead of silently dropping a
    message it cannot place in the window. Those two cases are the same case: a payload with no ts
    is no more placeable than one with "yesterday" in the field, and a message the window predicate
    cannot judge must not be silently omitted from a stream that claims to be the whole room.
    Declining a SUBTYPE still returns None — that is a decision, not a gap.

    `allow_any_subtype` is for a MUTATION's nested subject, where the skip-set does not apply:
    an edited or deleted message is a message whatever its subtype says, and the subtype is
    also where `tombstone` and `thread_broadcast` live — dropping or hiding it there is how the
    record's flags stopped agreeing with the event's kind.
    """
    if not isinstance(payload, dict):
        return None
    subtype = payload.get("subtype")
    if not allow_any_subtype and (subtype in SKIP_SUBTYPES or subtype in MUTATION_SUBTYPES):
        return None
    ts = payload.get("ts")
    parse_ts(ts)

    resolved_channel = channel_id or payload.get("channel")
    if not resolved_channel:
        return None
    resolved_team = team_id or getattr(client, "self_team_id", None) or ""

    sender_type = _sender_type(client, payload)
    raw_text = str(payload.get("text") or "")
    # F48: content Slack delivers outside `text` (pasted tables, webhook fields) is real
    # awareness — but never for our OWN messages, whose status cards live in exactly those
    # fields and would replay as somebody's evidence (the F47 attribution bug).
    if sender_type != "self":
        supplementary = extract_supplementary_text(payload, primary_text=raw_text)
        if supplementary:
            raw_text = f"{raw_text}\n\n{supplementary}" if raw_text.strip() else supplementary
    text = sanitize_field(raw_text)

    # The root ts is a comparator key too — the inventory places threads by it and the index keys
    # rows on it. An unreadable one used to ride through as a string, so a thread could be
    # discovered under a ts no window predicate could ever judge.
    thread_root = secondary_ts(payload, "thread_ts")
    edited = payload.get("edited")
    edited_ts = edited.get("ts") if isinstance(edited, dict) else None
    if edited_ts:
        parse_ts(edited_ts)
    latest_reply = secondary_ts(payload, "latest_reply")
    reply_count = payload.get("reply_count")

    return NormalizedMessage(
        team_id=str(resolved_team),
        channel_id=str(resolved_channel),
        ts=str(ts),
        thread_root_ts=thread_root,
        subtype=str(subtype) if subtype else None,
        sender_id=canonical_sender_id(client, payload),
        sender_type=sender_type,
        raw_bot_name=_bot_name(payload),
        text=text,
        files=_file_refs(payload),
        reactions=_reactions(payload, getattr(client, "bot_user_id", None)),
        edited_ts=str(edited_ts) if edited_ts else None,
        is_broadcast=(subtype == "thread_broadcast"),
        is_tombstone=_is_tombstone(payload),
        reply_count=reply_count if isinstance(reply_count, int) else None,
        latest_reply=latest_reply,
        # Raw, never cleaned: the actor map renders them, so bot-authored and human-authored
        # text finally read the same way (the get_thread_history cleaning gap).
        mention_ids=tuple(extract_mention_ids(text)),
        origin=origin,
    )


def mutation_activity_ts(event: Dict[str, Any]) -> Optional[str]:
    """When a mutation envelope's activity happened — the ONE definition of it.

    Read by the admission step (`registration._admit`) as well as by `normalize_slack_event`,
    because the two computed it separately once and it cost a whole class of permanent failure:
    admission refused an edit that carried no outer `event_ts`, putting the channel out of service,
    while the normalizer placed the same edit perfectly well from the nested `edited.ts`.

    Never the subject's ts — the message being edited or deleted may be hours old and the activity
    is now. Returns None for anything that is not a mutation, and None when the envelope names no
    activity time at all (a tombstone with neither field; the consumer records no event_ts for it).
    """
    subtype = event.get("subtype")
    if subtype not in MUTATION_SUBTYPES:
        return None
    outer = event.get("event_ts")
    if outer:
        return str(outer)
    if subtype != "message_changed":
        return None
    subject = event.get("message")
    nested = subject.get("edited") if isinstance(subject, dict) else None
    nested_ts = nested.get("ts") if isinstance(nested, dict) else None
    return str(nested_ts) if nested_ts else None


def mutation_subject(event: Dict[str, Any]) -> Any:
    """A mutation envelope's nested SUBJECT payload — the ONE definition of where it lives.

    `message` for a change, `previous_message` for a deletion, None for anything else. Returned
    raw (it may be missing or not a dict): the callers disagree about what an unreadable subject
    means — normalize_slack_event raises MalformedEventError, a log helper just wants a ts.
    """
    subtype = event.get("subtype")
    if subtype == "message_changed":
        return event.get("message")
    if subtype == "message_deleted":
        return event.get("previous_message")
    return None


def mutation_subject_ts(event: Dict[str, Any]) -> Optional[str]:
    """The ts of the message that was edited or deleted — never the envelope's own.

    Computed HERE and nowhere else: a second reader that fell back to `event_ts` would name when
    the edit happened rather than what it changed, and the two disagree on every mutation.
    """
    subject = mutation_subject(event)
    if not isinstance(subject, dict):
        return None
    ts = subject.get("ts")
    return str(ts) if ts else None


def _has_reply_hints(payload: Dict[str, Any]) -> bool:
    count = payload.get("reply_count")
    return bool(payload.get("latest_reply")) or (isinstance(count, int) and count > 0)


def normalize_slack_event(client: Any, event: Any) -> Optional[NormalizedEvent]:
    """Raw Slack listener payload → NormalizedEvent, or None to skip.

    Resolves the EFFECTIVE message dict first (an edit's or a deletion's subject is nested),
    because every downstream question — who sent it, which thread it belongs to, whether it is
    ours — is about that message and not about the envelope. DMs are skipped: they have no
    stream, no index and no receipts.

    None and a raise mean two different things, and the caller's fail-closed handling depends on
    which it gets. None is a DECISION: not our concern (a DM, another workspace), or a kind that
    records nothing. `MalformedEventError` is a known MUTATION we could not resolve — its subject
    missing, not a dict, or carrying no timestamp — which is a real event with a real activity ts
    that nothing can be recorded for. `TimestampError` is the same verdict for a ts that will not
    parse. Both are ValueError, which is what every listener already fails the observation on.
    """
    if not isinstance(event, dict):
        return None
    team_id = getattr(client, "self_team_id", None)
    if not team_id:
        return None
    channel_id = event.get("channel") or (event.get("item") or {}).get("channel")
    if not isinstance(channel_id, str) or not channel_id:
        return None
    if is_dm_conversation(channel_id, event.get("channel_type")):
        return None

    subtype = event.get("subtype")
    deleted_ts = None
    if subtype == "message_changed":
        effective = mutation_subject(event)
        if not isinstance(effective, dict):
            raise MalformedEventError(
                f"message_changed in {channel_id} carries no message to index")
        kind = KIND_TOMBSTONE if _is_tombstone(effective) else KIND_EDIT
        activity_ts = mutation_activity_ts(event)
    elif subtype == "message_deleted":
        # deleted_ts identifies WHICH message went away; the activity happened at event_ts.
        effective = mutation_subject(event)
        if not isinstance(effective, dict):
            raise MalformedEventError(
                f"message_deleted in {channel_id} carries no previous_message to index")
        kind = KIND_DELETE
        activity_ts = mutation_activity_ts(event)
        deleted_ts = event.get("deleted_ts")
    else:
        effective = event
        kind = KIND_MESSAGE
        activity_ts = None

    subject_ts = effective.get("ts")
    if not subject_ts:
        if kind != KIND_MESSAGE:
            # A mutation names its subject by ts and by nothing else. With none there is no row to
            # write and no way to say which message changed, so this fails the observation rather
            # than reporting an indexed edit that never happened.
            raise MalformedEventError(
                f"{subtype} in {channel_id} names no subject timestamp")
        # A plain event with no ts of its own (an `item`-shaped envelope) records nothing here.
        # It is not silently forgiven either: the watermark could not advance for it, so
        # registration._admit has already failed its observation.
        return None
    parse_ts(subject_ts)
    if activity_ts:
        parse_ts(activity_ts)

    root_if_indexed = False
    if kind != KIND_MESSAGE and not effective.get("thread_ts"):
        # A root usually carries no thread_ts of its own; Slack advertises its threading
        # through reply_count/latest_reply, and a tombstone only exists for a message that had
        # replies at all. With none of those hints the ts is only a root if the index already
        # knows it as one, which the consumer checks.
        root_if_indexed = not (kind == KIND_TOMBSTONE or _has_reply_hints(effective))

    # A deletion that names nobody defeats the own-message check — it fails OPEN and our own
    # housekeeping deletes read as somebody else's activity. Hand the consumer the ts to
    # arbitrate against the receipt ledger. Edits are excluded: their nested message always
    # carries its author.
    probe = None
    if kind in (KIND_DELETE, KIND_TOMBSTONE) and not any(
            effective.get(k) for k in _IDENTITY_KEYS):
        probe = str(subject_ts)

    # A plain message with a non-message subtype yields no record: the actor tail and the stream
    # must not see a join notice as somebody speaking. Consumers that only want the channel
    # touched (lazy coverage seeding) still get the event, with message=None.
    #
    # A MUTATION's subject is a message whatever envelope carried it, so the skip-set is waived
    # rather than the subtype removed. Removing it also removed the two flags that live there —
    # a subtype-only tombstone came back is_tombstone False while the EVENT said kind=tombstone,
    # and a deleted thread_broadcast came back is_broadcast False. Consumers read different
    # halves of that pair, so the two must agree.
    message = normalize_slack_message(client, effective, channel_id=channel_id,
                                      origin=ORIGIN_LIVE, team_id=str(team_id),
                                      allow_any_subtype=kind != KIND_MESSAGE)

    return NormalizedEvent(
        kind=kind,
        team_id=str(team_id),
        channel_id=channel_id,
        subject_ts=str(subject_ts),
        activity_ts=str(activity_ts) if activity_ts else None,
        root_if_indexed=root_if_indexed,
        owner_probe_ts=probe,
        deleted_ts=str(deleted_ts) if deleted_ts else None,
        message=message,
    )


def is_own_event(client: Any, event: NormalizedEvent) -> bool:
    """Consumer-side self filter (watermark, live index, actor tail). The normalizer never
    applies it — see the module docstring."""
    if event.message is not None:
        return event.message.sender_type == "self"
    return False


def render_mentions(text: str, actor_names: Dict[str, str]) -> str:
    """Replace `<@ID>` / `<@ID|label>` with `@{name}` from the frozen actor map; an
    unresolved id renders as `@{raw_id}` so an addressee marker never vanishes."""
    if not text or "<@" not in text:
        return text

    def _sub(match: "re.Match[str]") -> str:
        raw_id = match.group(1)
        return f"@{actor_names.get(raw_id) or raw_id}"

    return _MENTION_RE.sub(_sub, text)
