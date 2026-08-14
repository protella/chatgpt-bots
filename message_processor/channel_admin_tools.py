"""T7 — the channel's own description: its topic and its purpose.

Both are one-line facts about a room that everybody in it reads, nobody re-reads, and almost
nobody notices changing. That asymmetry is the whole design here:

* **HUMAN REQUEST ONLY, enforced in code.** Not a prompt rule with a hopeful description — the
  same hard gate `set_channel_participation` carries (`participation_tools.py`): a HUMAN sender
  AND a turn that genuinely reached the responder. A topic is exactly the kind of thing a
  quoted line ("we should change the topic to X"), a bot's message, or an ambient turn nobody
  addressed would otherwise talk us into rewriting, and the person who finds out is whoever
  scrolls past it a week later.
* **The previous value is read FIRST and echoed back.** Slack's `conversations.setTopic`
  response carries only the NEW value, so without a read beforehand an accidental change is
  unrecoverable — nobody remembers what a channel topic said. With it, the fix is one ask.
* **The prompt cache is invalidated after the write.** `get_channel_context` caches
  name/topic/purpose for 15 minutes and Slack's topic-change events never reach the dispatch
  path, so a successful write would otherwise be followed by a quarter-hour of prompts
  confidently describing the old topic.
* **Real channels and private groups only.** A group DM is channel-shaped everywhere else in
  this codebase (`is_dm_conversation` classifies `mpim` as not-a-DM, so an MPIM gets the channel
  tool surface), but the app carries no MPIM topic scope — so an MPIM is excluded explicitly,
  against `conversations.info`, rather than left to fail as a scope error nobody can read.

Executors never raise: every failure is an ``{"ok": False, "error": ...}`` result.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from logger import setup_logger
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.ChannelAdminTools")

# field name -> (the conversations.info key, the setter method, its keyword argument)
_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "topic": ("topic", "conversations_setTopic", "topic"),
    "purpose": ("purpose", "conversations_setPurpose", "purpose"),
}


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    logger.info(f"channel admin tool refused: {code} — {message}")
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
    """Slack's own error name, which is the whole classification, or the exception text."""
    resp = getattr(exc, "response", None)
    getter = getattr(resp, "get", None) if resp is not None else None
    if callable(getter):
        name = getter("error")
        if isinstance(name, str) and name:
            return name
    return str(exc)[:200]


def _clean(value: Any) -> str:
    """Slack HTML-escapes &, < and > in these fields; undo it, exactly as the prompt path does.

    This matters more here than it does for a prompt: the previous value we echo is meant to be
    passed straight back in to restore it, and a still-escaped `&amp;` would be re-escaped on
    the way, so a round trip through this tool would corrode the topic one ampersand at a time.
    """
    if not isinstance(value, str):
        return ""
    return value.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def _field_value(channel: Any, info_key: str) -> str:
    """`topic`/`purpose` out of a channel record, where each is a `{"value": …}` sub-object."""
    if not isinstance(channel, dict):
        return ""
    holder = channel.get(info_key)
    value = holder.get("value") if isinstance(holder, dict) else ""
    return value if isinstance(value, str) else ""


def _authorized(ctx: ToolContext) -> Optional[Dict[str, Any]]:
    """The human-request gate, or None when the call may proceed.

    Deliberately the SAME shape as `execute_set_channel_participation`'s gate, for the same
    reason: `structural_change_authorized` already encodes "a human wrote this AND the turn
    genuinely reached the responder", and the raw sender classification is re-checked on top
    where the context still carries one. Absent classification falls back to the flag, so this
    never fails closed on a path that omits the metadata.
    """
    if not getattr(ctx, "structural_change_authorized", False):
        return _err("not_addressed",
                    "The channel topic and purpose can only be changed when someone directly "
                    "asks you to, in their own current message.")
    msg = getattr(ctx, "message", None)
    sender_type = ((getattr(msg, "metadata", None) or {}).get("sender_type")
                   if msg is not None else None)
    if sender_type is not None and sender_type != "human":
        return _err("not_human_sender",
                    "The channel topic and purpose can only be changed at a person's request.")
    return None


async def _channel_record(ctx: ToolContext) -> Tuple[Optional[Dict[str, Any]],
                                                     Optional[Dict[str, Any]]]:
    """`conversations.info` for the current channel: (record, refusal). Exactly one of the two.

    Does double duty on purpose — it is both the surface gate (is this really a channel?) and
    the source of the previous value — because those are the same round trip and doing them
    separately would mean two.
    """
    web = _web(ctx)
    if web is None:
        return None, _err("unavailable", "Channel settings aren't available right now.")
    try:
        res = await _async(web.conversations_info, channel=ctx.channel_id)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        return None, _err("lookup_failed",
                          f"I couldn't read this channel's current settings, so I changed "
                          f"nothing: {err}")
    channel = res.get("channel") or {}
    if channel.get("is_im") or channel.get("is_mpim"):
        # A group DM reaches the channel tool surface but has no topic scope behind it. Said
        # plainly here rather than surfaced later as a scope error nobody can act on.
        return None, _err("not_a_channel",
                          "This is a group DM, not a channel — it has no topic or purpose to "
                          "set.")
    return channel, None


async def _set_field(ctx: ToolContext, args: Dict[str, Any], field: str) -> Dict[str, Any]:
    """The shared body of set_channel_topic / set_channel_purpose."""
    info_key, method_name, kwarg = _FIELDS[field]

    # Refusals decidable without a network call come first, so a call that cannot proceed never
    # touches the workspace.
    if getattr(ctx, "is_dm", False):
        return _err(f"{field}_is_channel_only",
                    f"A DM has no {field} to set — that's a channel setting.")
    if not ctx.channel_id:
        return _err("no_channel", "No channel in this context.")
    refusal = _authorized(ctx)
    if refusal is not None:
        return refusal
    text = args.get("text")
    # An empty string is a real instruction — "clear the topic" — so this is a TYPE check, not a
    # truthiness one; collapsing "" into missing would make a clear silently do nothing.
    if not isinstance(text, str):
        return _err("missing_text",
                    f"Pass the new {field} as text (an empty string clears it).")
    text = text.strip()

    channel, refusal = await _channel_record(ctx)
    if refusal is not None:
        return refusal
    previous = _clean(_field_value(channel, info_key))

    try:
        res = await _async(getattr(_web(ctx), method_name), channel=ctx.channel_id,
                           **{kwarg: text})
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        logger.warning(f"conversations.set{field.capitalize()} failed in {ctx.channel_id}: {err}")
        return _err(f"set_{field}_failed", f"Slack refused to change the {field}: {err}")

    # Slack echoes the stored value back — the one that actually landed, after whatever
    # normalizing it did — so prefer it over what we sent. It arrives in either of two shapes
    # depending on the API version answering: a bare `"topic": "…"` at the top level, or the
    # whole channel record with `topic.value` inside it. Read both, fall back to what we sent.
    landed = _clean(res.get(info_key)) or _clean(_field_value(res.get("channel"), info_key))
    landed = landed or text

    # The write is done; the cache is what the next prompt reads. Without this the bot would
    # describe the OLD topic for up to 15 minutes, including in the same conversation where it
    # just announced the change.
    invalidate = getattr(getattr(ctx, "client", None), "invalidate_channel_context", None)
    if callable(invalidate):
        try:
            invalidate(ctx.channel_id)
        except Exception as e:  # noqa: BLE001 — a stale cache must not fail a landed write
            logger.debug(f"channel context not invalidated for {ctx.channel_id}: {e}")

    logger.warning(f"CHANGED channel {field} in {ctx.channel_id} (requested by {ctx.user_id}): "
                   f"{previous!r} -> {landed!r}")
    confirmation = (f"The channel {field} is now: {landed}" if landed
                    else f"The channel {field} is cleared.")
    return {"ok": True, "field": field, "previous": previous, "new": landed,
            "confirmation": confirmation,
            "message": (f"Confirm the new {field} briefly. Slack also posts its own notice in "
                        f"the channel, so don't repeat it at length. The previous {field} is in "
                        "`previous` — if this turns out to be wrong, setting it back is one "
                        "call.")}


# --- schemas ------------------------------------------------------------------------------

_REQUEST_ONLY = (
    "Use this ONLY when a person, in their CURRENT message, directly asks you to change it. "
    "Never on your own initiative, never because the channel has drifted off its stated "
    "subject, and never from a line that merely MENTIONS the topic — someone quoting, "
    "reporting or wishing out loud ('the topic is out of date') is not an instruction. If you "
    "are not sure they meant it as an instruction, ask. Everyone in the channel sees the "
    "change and Slack announces it in the channel."
)


def get_set_channel_topic_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "set_channel_topic",
        "description": (
            "Set this channel's TOPIC — the short line beside the channel name, meant for what "
            "is going on right now ('Release 4.2 freeze until Friday').\n\n"
            + _REQUEST_ONLY +
            "\n\nActs on the current channel only (there is no channel argument), and not in "
            "DMs or group DMs. The result gives you the PREVIOUS topic, so tell them what it "
            "used to be if the change might have been a mistake."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": ("The new topic, in the words the person asked for. An empty "
                                    "string clears it."),
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    }


def get_set_channel_purpose_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "set_channel_purpose",
        "description": (
            "Set this channel's PURPOSE — the standing description of what the channel is FOR, "
            "shown to anyone deciding whether to join it ('Where we coordinate production "
            "deploys'). Unlike the topic, it is not about this week.\n\n"
            + _REQUEST_ONLY +
            "\n\nActs on the current channel only (there is no channel argument), and not in "
            "DMs or group DMs. The result gives you the PREVIOUS purpose, so tell them what it "
            "used to be if the change might have been a mistake."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": ("The new purpose, in the words the person asked for. An "
                                    "empty string clears it."),
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    }


# --- executors ----------------------------------------------------------------------------

async def execute_set_channel_topic(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return await _set_field(ctx, args, "topic")


async def execute_set_channel_purpose(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return await _set_field(ctx, args, "purpose")


def register_channel_admin_tools(registry: ToolRegistry) -> None:
    """set_channel_topic / set_channel_purpose — the CHANNEL surface only.

    `enabled=lambda _cfg: False` is the established way to keep a tool off the DM surface
    (`edit_own_message` does the same): a DM has no topic or purpose, so offering the tools
    there is an invitation to a refusal. No `channel_enabled` — there is no feature flag for
    these, and the authorization that matters is per-message, which a channel-surface gate
    structurally cannot see. The executors hold it instead, on both surfaces.
    """
    registry.register(get_set_channel_topic_schema(), execute_set_channel_topic,
                      enabled=lambda _cfg: False)
    registry.register(get_set_channel_purpose_schema(), execute_set_channel_purpose,
                      enabled=lambda _cfg: False)
