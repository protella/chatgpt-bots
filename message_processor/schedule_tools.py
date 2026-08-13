"""T1 — scheduled messages: the only future the bot actually has.

Everything else this process does is bounded by a turn. A restart, a deploy, a crash — and any
"I'll remind you at 9" that lives in memory is simply gone, with nobody to notice it never
happened. `chat.scheduleMessage` hands the promise to SLACK, which holds it server-side and posts
it whether or not we are running. That is the whole point of these three tools, and it is why the
descriptions tell the model that a reminder means calling one of them rather than saying it will.

Three facts shape the code below:

* **There is no model call at delivery time.** Slack posts the text VERBATIM, as the bot. So the
  text is written at schedule time as the bot's own future post ("Reminder for <@U…>: take the
  cake out") — not as an instruction to a future self that will never read it.
* **`post_at` is an instant, not a phrase.** The SCHEMA takes epoch seconds or ISO-8601, and the
  MODEL does the language→datetime step ("tomorrow at 9" is a language problem, and it has the
  conversation's context to solve it). The executor only resolves the one thing the model cannot
  know: an ISO string with no offset is read in the REQUESTER's own timezone
  (`get_user_timezone_async` — the IANA string; the sync sibling returns a tuple and is the wrong
  anchor). The resolved UTC instant is echoed back in the result so a timezone mistake is caught
  by the person, this turn, instead of at 3am.
* **Slack owns the limits.** 30 pending per channel, nothing in the past or more than 120 days
  out, and no cancelling within about a minute of the post time. Those are STATED in the
  descriptions and enforced by Slack: nothing here pre-validates a bound of its own invention, and
  a rejection comes back carrying Slack's own error name.

Scope is the current conversation and nothing else — the channel the request came from, the
current thread if it came from one, the DM if it was a DM. There is no cross-channel scheduling,
which keeps "who will see this" exactly the question a normal reply already answers.

Receipts: a scheduled delivery arrives later as an own-message that no turn is around to claim, so
`schedule_message` pre-registers the expectation and `outbound_receipts` finalizes it when the
message appears; cancelling drops the expectation. Without that the bot posts a reminder it can
never afterwards see. See the "scheduled deliveries" section of `outbound_receipts.py`.

Executors never raise: every failure is an ``{"ok": False, "error": ...}`` result.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from logger import setup_logger
from message_processor.outbound_receipts import (CLASS_ASSISTANT_REPLY,
                                                 expect_scheduled_delivery,
                                                 forget_scheduled_delivery)
from tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.ScheduleTools")


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    # Logged as well as returned: the tool loop records only "-> error", so an unlogged refusal is
    # indistinguishable from a Slack outage when someone reads the logs afterwards.
    logger.info(f"schedule tool refused: {code} — {message}")
    return {"ok": False, "error": code, "message": message, **extra}


def _web(ctx: ToolContext) -> Any:
    """The Slack WebClient underneath the platform client."""
    client = getattr(ctx, "client", None)
    app = getattr(client, "app", None)
    return getattr(app, "client", None) if app is not None else None


async def _async(fn: Any, **kwargs: Any) -> Any:
    """slack_sdk's WebClient here is the AsyncWebClient under Bolt's async app."""
    result = fn(**kwargs)
    if hasattr(result, "__await__"):
        result = await result
    return result


def _slack_error(exc: Exception) -> str:
    """Slack's own error name for a failed call, falling back to the exception text.

    The API error name (`time_in_past`, `time_too_far`, `msg_too_long`,
    `invalid_scheduled_message_id`) IS the classification and is the part the model can act on;
    `str(e)` on a SlackApiError is a response dump that says the same thing less usefully.
    """
    resp = getattr(exc, "response", None)
    getter = getattr(resp, "get", None) if resp is not None else None
    if callable(getter):
        name = getter("error")
        if isinstance(name, str) and name:
            return name
    return str(exc)[:200]


# --- time ----------------------------------------------------------------------------------

async def _requester_zone(ctx: ToolContext) -> Optional[str]:
    """The requester's IANA zone, or None when we genuinely do not know it.

    Two sources, in order: what we already stored for them, then Slack's live profile. NOT
    `client.get_user_timezone` — that helper indexes the async accessor's return as if it were the
    sync one's tuple, so a cached zone comes back as its own first letter.

    None is a real answer and the caller must not paper over it with UTC: silently reading "9am"
    as UTC posts a reminder at 3am for someone in Chicago, and the mistake is invisible until it
    happens.
    """
    user_id = getattr(ctx, "user_id", None)
    if not user_id:
        return None
    db = getattr(ctx, "db", None)
    if db is not None:
        try:
            zone = await db.get_user_timezone_async(user_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"stored timezone lookup failed for {user_id}: {e}")
            zone = None
        if isinstance(zone, str) and zone.strip():
            return zone.strip()
    web = _web(ctx)
    if web is None:
        return None
    try:
        res = await _async(web.users_info, user=user_id)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"users.info timezone lookup failed for {user_id}: {_slack_error(e)}")
        return None
    zone = ((res.get("user") or {}) or {}).get("tz")
    return zone.strip() if isinstance(zone, str) and zone.strip() else None


def _parse_epoch(raw: Any) -> Optional[int]:
    """`post_at` as epoch seconds, when it was given that way. None means "not an epoch"."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        # A bare integer string is an epoch; anything with a '-' or 'T' is a date.
        if text.isdigit():
            return int(text)
    return None


async def _resolve_post_at(ctx: ToolContext, raw: Any) -> Tuple[Optional[int], Optional[str],
                                                                Optional[Dict[str, Any]]]:
    """(epoch seconds, the zone it was read in, error result). Exactly one of the first/last is set.

    An offset-bearing ISO string and an epoch both name an instant on their own, so no zone is
    consulted for them and the returned zone is only used to echo a local time back.
    """
    epoch = _parse_epoch(raw)
    zone_name = await _requester_zone(ctx)
    if epoch is not None:
        return epoch, zone_name, None
    if not isinstance(raw, str) or not raw.strip():
        return None, None, _err("bad_post_at",
                                "post_at must be epoch seconds or an ISO-8601 datetime "
                                "(e.g. 2026-08-13T09:00:00).")
    text = raw.strip()
    try:
        when = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None, None, _err(
            "bad_post_at",
            f"I couldn't read {text!r} as a date and time. Use ISO-8601 "
            "(2026-08-13T09:00:00, or 2026-08-13T09:00:00-05:00 to state the offset yourself) "
            "or epoch seconds.")
    if when.tzinfo is None:
        # The one thing the model cannot resolve: whose 9am this is.
        if not zone_name:
            return None, None, _err(
                "timezone_unknown",
                "I don't know this person's timezone, so a bare local time is ambiguous. Pass "
                "post_at with an explicit UTC offset (2026-08-13T09:00:00-05:00) or as epoch "
                "seconds — or ask them which timezone they mean.")
        try:
            when = when.replace(tzinfo=ZoneInfo(zone_name))
        except (ZoneInfoNotFoundError, ValueError) as e:
            return None, None, _err(
                "timezone_unknown",
                f"Their stored timezone ({zone_name!r}) isn't one I can use ({e}). Pass post_at "
                "with an explicit UTC offset or as epoch seconds.")
    return int(when.timestamp()), zone_name, None


def _utc_iso(epoch: Any) -> Optional[str]:
    try:
        moment = datetime.datetime.fromtimestamp(float(epoch), tz=datetime.timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _local_iso(epoch: Any, zone_name: Optional[str]) -> Optional[str]:
    """The same instant in the requester's zone — the form they would recognise as "9am"."""
    if not zone_name:
        return None
    try:
        zone = ZoneInfo(zone_name)
        moment = datetime.datetime.fromtimestamp(float(epoch), tz=zone)
    except (ZoneInfoNotFoundError, TypeError, ValueError, OSError, OverflowError):
        return None
    return moment.strftime("%Y-%m-%d %H:%M %Z")


def _when(epoch: Any, zone_name: Optional[str]) -> Dict[str, Any]:
    """The three ways of saying one instant that a result carries."""
    out: Dict[str, Any] = {"post_at": int(float(epoch)), "post_at_utc": _utc_iso(epoch)}
    local = _local_iso(epoch, zone_name)
    if local:
        out["post_at_local"] = local
        out["timezone"] = zone_name
    return out


# --- schemas -------------------------------------------------------------------------------

_LIMITS = ("Slack holds the schedule, not this process, so it survives restarts. It keeps up to "
           "30 pending messages per conversation, refuses a time in the past or more than 120 "
           "days out, and may refuse one only seconds away; a scheduled message cannot be "
           "cancelled in the last minute or so before it posts.")


def get_schedule_message_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "schedule_message",
        "description": (
            "Post a message into THIS conversation at a future time. This is the only way you can "
            "make anything happen later — you are not running between turns, so a reminder you "
            "merely promise is a reminder nobody gets. If someone asks you to remind them, ping "
            "them, or follow up at some point, call this; never say you will and stop there.\n\n"
            "Slack posts the text exactly as written, as you, with no chance for you to rewrite "
            "it first — there is no model call at delivery time. So write it as the finished "
            "future post: '<@U123> reminder: take the cake out of the oven', not 'remind them "
            "about the cake'. Mention the person you are reminding, or they may never see it.\n\n"
            "On request only. A message nobody asked for arrives in a conversation with no "
            "context and cannot be taken back once it posts, so never schedule one on your own "
            "initiative, and schedule one message rather than a series unless they asked for the "
            "series.\n\n"
            "It goes to this conversation only (this thread when you are in one) — there is no "
            "scheduling into another channel. It cannot be edited afterwards: to change one, "
            "cancel it with cancel_scheduled_message and schedule the new version.\n\n"
            + _LIMITS + "\n\n"
            "The result echoes back the exact time it resolved to. Say that time back to the "
            "person in your reply, in their words, so a misunderstanding is caught now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": ("The message to post, written as your own future post. Slack "
                                    "sends it verbatim."),
                },
                "post_at": {
                    "type": "string",
                    "description": (
                        "When to post it: an ISO-8601 datetime (2026-08-13T09:00:00) or epoch "
                        "seconds. You do the reading of 'tomorrow at 9' or 'in two hours' into a "
                        "datetime — you know today's date and what they meant. A datetime with no "
                        "UTC offset is read in the REQUESTER's own timezone; add the offset "
                        "(2026-08-13T09:00:00-05:00) when you mean a specific one."
                    ),
                },
                "thread": {
                    "type": "boolean",
                    "description": (
                        "Where it lands. Defaults to this thread when the request came from one, "
                        "and to the top level otherwise. Pass false to put a reminder at the top "
                        "level of the channel where it is hard to miss; pass true to keep it "
                        "inside the thread it belongs to."
                    ),
                },
            },
            "required": ["text", "post_at"],
            "additionalProperties": False,
        },
    }


def get_list_scheduled_messages_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_scheduled_messages",
        "description": (
            "List the messages you have scheduled to post in this conversation but which have not "
            "posted yet, each with the id cancel_scheduled_message takes, when it will post, and "
            "its text.\n\n"
            "Use it when someone asks what is queued up, when they want to change or call off "
            "something you scheduled, or before scheduling something that may already be "
            "scheduled. Only PENDING ones exist here — once a scheduled message posts it is an "
            "ordinary message in the conversation, not something this can find."
        ),
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    }


def get_cancel_scheduled_message_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "cancel_scheduled_message",
        "description": (
            "Call off a message you scheduled in this conversation, so it never posts.\n\n"
            "Use it when someone says the reminder is no longer needed, or as the first half of "
            "changing one: a scheduled message cannot be edited, so cancel it and schedule the "
            "new version.\n\n"
            "The id comes from list_scheduled_messages or from the schedule_message result — it "
            "is opaque and cannot be guessed. Slack refuses a cancellation in the last minute or "
            "so before the message posts, so a last-second change of mind may be too late; if it "
            "posts anyway, say so rather than pretending it was stopped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scheduled_message_id": {
                    "type": "string",
                    "description": ("The id of the scheduled message, from "
                                    "list_scheduled_messages or schedule_message."),
                },
            },
            "required": ["scheduled_message_id"],
            "additionalProperties": False,
        },
    }


# --- executors -----------------------------------------------------------------------------

def _thread_for(ctx: ToolContext, requested: Any) -> Optional[str]:
    """The thread_ts a scheduled post should carry, or None for the top level.

    Default = where this turn's own reply would go: inside the thread when the request came from
    one (`thread_ts` differs from the triggering message), at the top level otherwise. An explicit
    `thread` argument overrides that in either direction, except where the channel's settings
    forbid a top-level reply — a scheduled post is still a post, and it does not get to route
    around a setting a live reply obeys.
    """
    thread_ts = getattr(ctx, "thread_ts", None)
    trigger_ts = getattr(ctx, "trigger_ts", None)
    in_thread = bool(thread_ts) and thread_ts != trigger_ts
    wanted = in_thread if not isinstance(requested, bool) else requested
    if not wanted:
        meta = getattr(getattr(ctx, "message", None), "metadata", None) or {}
        if meta.get("channel_post_allowed") is False and thread_ts:
            return str(thread_ts)
        return None
    return str(thread_ts) if thread_ts else None


async def execute_schedule_message(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return _err("missing_text", "A scheduled message needs the text to post.")
    text = text.strip()
    channel_id = getattr(ctx, "channel_id", None)
    web = _web(ctx)
    if web is None or not channel_id:
        return _err("unavailable", "Scheduling isn't available right now.")

    epoch, zone_name, failure = await _resolve_post_at(ctx, args.get("post_at"))
    if failure is not None:
        return failure
    # cast-free: `failure is None` is the branch where _resolve_post_at returned an epoch.
    post_at = int(epoch or 0)

    thread_ts = _thread_for(ctx, args.get("thread"))
    # The same outbound hygiene every other post of ours gets (mrkdwn, mention encoding), so a
    # scheduled message does not read as the one bot post with raw markdown in it.
    body = text
    formatter = getattr(getattr(ctx, "client", None), "format_text", None)
    if callable(formatter):
        try:
            body = formatter(text)
        except Exception as e:  # noqa: BLE001 — formatting is polish, never a reason not to post
            logger.debug(f"format_text failed for a scheduled message: {e}")
            body = text

    try:
        res = await _async(web.chat_scheduleMessage, channel=channel_id, post_at=str(post_at),
                           text=body, **({"thread_ts": thread_ts} if thread_ts else {}))
    except Exception as e:  # noqa: BLE001 — the tool contract is a dict, never a raise
        err = _slack_error(e)
        logger.warning(f"chat.scheduleMessage refused in {channel_id} for {post_at}: {err}")
        return _err("schedule_failed", f"Slack refused to schedule it: {err}",
                    **_when(post_at, zone_name))

    scheduled_id = res.get("scheduled_message_id") or res.get("id")
    # Slack echoes the time it actually accepted; trust its copy over ours for the wait.
    accepted_at = res.get("post_at") or post_at
    expect_scheduled_delivery(
        team_id=getattr(getattr(ctx, "client", None), "self_team_id", None),
        channel_id=channel_id, scheduled_message_id=scheduled_id, text=body,
        post_at=accepted_at, receipt_class=CLASS_ASSISTANT_REPLY,
        # The reconciler reads the delivery back out of Slack, and a thread reply is not in the
        # channel listing — it can only be found by replying-thread. This is the one moment
        # anything knows which thread the post went into.
        thread_root_ts=thread_ts)

    logger.info(f"Scheduled message {scheduled_id} in {channel_id} for {accepted_at} "
                f"(requested by {ctx.user_id})")
    return {"ok": True, "scheduled_message_id": scheduled_id, "channel": channel_id,
            "thread_ts": thread_ts, "text": text, **_when(accepted_at, zone_name),
            "message": ("It's queued with Slack and will post even if I restart. Tell them what "
                        "it will say and WHEN — including the time above in their own timezone — "
                        "so a mistake is caught now rather than when it posts.")}


def _entry(raw: Any, zone_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """One pending scheduled message, reduced to what a caller can act on."""
    if not isinstance(raw, dict) or raw.get("id") in (None, ""):
        return None
    return {"scheduled_message_id": str(raw.get("id")),
            "text": raw.get("text") or "",
            **_when(raw.get("post_at") or 0, zone_name)}


async def execute_list_scheduled_messages(ctx: ToolContext,
                                          args: Dict[str, Any]) -> Dict[str, Any]:
    channel_id = getattr(ctx, "channel_id", None)
    web = _web(ctx)
    if web is None or not channel_id:
        return _err("unavailable", "Scheduling isn't available right now.")
    zone_name = await _requester_zone(ctx)
    try:
        res = await _async(web.chat_scheduledMessages_list, channel=channel_id)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        return _err("list_failed", f"Slack could not list the scheduled messages: {err}")

    scheduled: List[Dict[str, Any]] = []
    for raw in (res.get("scheduled_messages") or []):
        # Slack scopes the listing by `channel` already; the filter is here so a future paging or
        # transport change cannot quietly widen a tool that promises "this conversation".
        if isinstance(raw, dict) and raw.get("channel_id") and raw["channel_id"] != channel_id:
            continue
        entry = _entry(raw, zone_name)
        if entry is not None:
            scheduled.append(entry)
    scheduled.sort(key=lambda item: item.get("post_at") or 0)
    return {"ok": True, "count": len(scheduled), "scheduled_messages": scheduled,
            "message": ("These are the ones that have NOT posted yet. Anything already posted is "
                        "an ordinary message in this conversation."
                        if scheduled else
                        "Nothing is scheduled to post in this conversation.")}


async def execute_cancel_scheduled_message(ctx: ToolContext,
                                           args: Dict[str, Any]) -> Dict[str, Any]:
    raw_id = args.get("scheduled_message_id")
    if isinstance(raw_id, (int, float)) and not isinstance(raw_id, bool):
        raw_id = str(int(raw_id))
    if not isinstance(raw_id, str) or not raw_id.strip():
        return _err("missing_id", "A scheduled_message_id is required.")
    scheduled_id = raw_id.strip()
    channel_id = getattr(ctx, "channel_id", None)
    web = _web(ctx)
    if web is None or not channel_id:
        return _err("unavailable", "Scheduling isn't available right now.")

    try:
        # `channel` is what confines this to the current conversation: Slack refuses an id that
        # was not scheduled here, so no id from elsewhere can be cancelled through this tool.
        await _async(web.chat_deleteScheduledMessage, channel=channel_id,
                     scheduled_message_id=scheduled_id)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        logger.warning(f"chat.deleteScheduledMessage refused for {scheduled_id} in "
                       f"{channel_id}: {err}")
        return _err("cancel_failed", f"Slack refused to cancel it: {err}",
                    scheduled_message_id=scheduled_id)

    # The expectation goes with it — nothing is coming to satisfy it now.
    forget_scheduled_delivery(scheduled_id)
    logger.info(f"Cancelled scheduled message {scheduled_id} in {channel_id} "
                f"(requested by {ctx.user_id})")
    return {"ok": True, "scheduled_message_id": scheduled_id, "cancelled": True,
            "message": "It won't post. Confirm briefly."}


def register_schedule_tools(registry: ToolRegistry) -> None:
    """schedule_message / list_scheduled_messages / cancel_scheduled_message, both surfaces.

    Static schemas and no feature flag, like `pin_message` and the bookmark tools: what is pending
    is not a per-turn fact, and putting it in a description would fork the cached prefix every time
    a reminder posts. `list_scheduled_messages` is how the model finds out.

    Both surfaces on purpose — "remind me tomorrow" is at least as common in a DM as in a channel,
    and the DM schedules into that DM.
    """
    registry.register(get_schedule_message_schema(), execute_schedule_message)
    registry.register(get_list_scheduled_messages_schema(), execute_list_scheduled_messages)
    registry.register(get_cancel_scheduled_message_schema(), execute_cancel_scheduled_message)
