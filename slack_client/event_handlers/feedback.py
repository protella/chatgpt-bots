"""
Phase H: response feedback — native feedback buttons + passive reaction ingestion.

Two signal sources, one sink (the ``response_feedback`` table):

- **Feedback buttons** — Slack's native ``feedback_buttons`` element (AI-apps surface,
  slack-sdk 3.43+), rendered inside a ``context_actions`` block on a small trailing
  message under DM/assistant-thread responses (channels keep their compact Configure
  footer instead — pixels matter there). Clicks arrive as ordinary block_actions.
- **Reactions** — ``reaction_added`` events on the bot's OWN messages, mapped through a
  tiny emoji table. Purely passive recording: no LLM call, no reply, debug logs only.

The participation engine can later read ``get_channel_feedback_ratio`` — deliberately
not wired into decisions yet.
"""
from __future__ import annotations

import os
from collections import OrderedDict

from config import config

# action_id for the feedback_buttons element. Also used by the history-rebuild filter
# to skip feedback-strip messages (like the footer's open_channel_settings).
FEEDBACK_ACTION_ID = "response_feedback"

# action_id for the DM strip's "⚙️ <model>" button → opens the USER settings modal
# (the channel footer's open_channel_settings opens the per-channel modal instead).
# Also in the rebuild filter's skip set.
USER_SETTINGS_ACTION_ID = "open_user_settings"

# Button values → signal. Slack echoes the clicked button's value in the action payload.
_BUTTON_VALUES = {"good": 1, "bad": -1}

# Reaction name → signal. Names arrive without colons and may carry a skin-tone
# suffix ("+1::skin-tone-4"). Everything not listed is ignored — reactions are an
# ambient social signal, only unambiguous thumbs count as feedback.
_REACTION_SIGNALS = {
    "+1": 1, "thumbsup": 1, "thumbsup_all": 1,
    "-1": -1, "thumbsdown": -1,
}


def feedback_enabled() -> bool:
    """Feature flag, defensive: config attr when present, env fallback, default ON."""
    val = getattr(config, "enable_feedback_buttons", None)
    if val is not None:
        return bool(val)
    return os.getenv("ENABLE_FEEDBACK_BUTTONS", "true").lower() == "true"


# Threads that already got a thumbs pair this process — feedback buttons show on the
# FIRST response of a conversation thread only (user feedback 2026-07-09: "we don't
# need the feedback after every response"). Reactions stay the always-available
# signal; a restart forgetting this set costs at most one extra thumbs row per thread.
_FEEDBACK_OFFERED: "OrderedDict[str, None]" = OrderedDict()
_FEEDBACK_OFFERED_MAX = 500


def should_offer_feedback(channel_id: str, thread_ts: str | None) -> bool:
    """True exactly once per (channel, thread) per process lifetime."""
    key = f"{channel_id}:{thread_ts or ''}"
    if key in _FEEDBACK_OFFERED:
        return False
    _FEEDBACK_OFFERED[key] = None
    while len(_FEEDBACK_OFFERED) > _FEEDBACK_OFFERED_MAX:
        _FEEDBACK_OFFERED.popitem(last=False)
    return True


def _reset_feedback_offers() -> None:
    """Test hook: forget which threads were offered feedback."""
    _FEEDBACK_OFFERED.clear()


def build_feedback_blocks(model_label: str | None = None, *, offer_feedback: bool = True) -> list:
    """The DM/assistant response strip: a compact "⚙️ <model>" button that opens the
    user settings modal (Claude-style subtext row), plus — on the FIRST response of a
    thread only (offer_feedback) — the native feedback buttons.

    Why a regular actions button and not an icon_button inside context_actions:
    Slack's icon_button enum currently accepts ONLY "trash" (verified live
    2026-07-09 against chat.postMessage; docs agree) — no gear/settings icon
    exists yet. Revisit when more icons ship.
    """
    blocks = []
    if offer_feedback:
        blocks.append(
            {
                "type": "context_actions",
                "elements": [
                    {
                        "type": "feedback_buttons",
                        "action_id": FEEDBACK_ACTION_ID,
                        "positive_button": {
                            "text": {"type": "plain_text", "text": "Good response"},
                            "accessibility_label": "Mark this response as good",
                            "value": "good",
                        },
                        "negative_button": {
                            "text": {"type": "plain_text", "text": "Bad response"},
                            "accessibility_label": "Mark this response as bad",
                            "value": "bad",
                        },
                    }
                ],
            }
        )
    label = (model_label or getattr(config, "gpt_model", "") or "").strip()
    if label:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": f"⚙️ {label}"},
                        "action_id": USER_SETTINGS_ACTION_ID,
                    }
                ],
            }
        )
    return blocks


def reaction_signal(reaction_name: str) -> int | None:
    """Map a reaction name (possibly with ::skin-tone suffix) to +1/-1, else None."""
    base = (reaction_name or "").split("::", 1)[0]
    return _REACTION_SIGNALS.get(base)


async def handle_feedback_action(client_self, ack, body) -> None:
    """block_actions handler for the feedback buttons. Ack fast, record, thank quietly.

    Never raises — feedback must not be able to break the event loop.
    """
    try:
        await ack()
    except Exception as e:  # noqa: BLE001
        client_self.log_debug(f"feedback ack failed: {e}")
    try:
        action = (body.get("actions") or [{}])[0]
        signal = _BUTTON_VALUES.get(action.get("value"))
        if signal is None:
            client_self.log_debug(f"feedback action with unknown value: {action.get('value')!r}")
            return
        channel_id = (body.get("channel") or {}).get("id")
        user_id = (body.get("user") or {}).get("id")
        message = body.get("message") or {}
        message_ts = message.get("ts") or (body.get("container") or {}).get("message_ts")
        thread_ts = message.get("thread_ts") or message_ts
        if not (channel_id and user_id and message_ts):
            client_self.log_debug("feedback action missing channel/user/message — ignored")
            return
        await client_self.db.record_response_feedback_async(
            channel_id=channel_id, thread_ts=thread_ts, message_ts=message_ts,
            user_id=user_id, signal=signal, source="button",
        )
        client_self.log_debug(
            f"feedback button recorded: {signal:+d} from {user_id} in {channel_id}"
        )
        # Quiet acknowledgment; best-effort (some surfaces reject ephemerals).
        try:
            await client_self.app.client.chat_postEphemeral(
                channel=channel_id, user=user_id, thread_ts=thread_ts,
                text="Thanks for the feedback!",
            )
        except Exception as e:  # noqa: BLE001
            client_self.log_debug(f"feedback thanks ephemeral failed: {e}")
    except Exception as e:  # noqa: BLE001
        client_self.log_debug(f"feedback action handling failed: {e}")


async def ingest_reaction(client_self, event) -> None:
    """Passive reaction_added ingestion: thumbs on the BOT'S OWN messages → sink.

    No LLM involvement, no replies, debug logging only. Never raises.
    """
    try:
        signal = reaction_signal(event.get("reaction", ""))
        if signal is None:
            return
        # Only feedback about our own messages counts.
        bot_user_id = getattr(client_self, "bot_user_id", None)
        if not bot_user_id or event.get("item_user") != bot_user_id:
            return
        user_id = event.get("user")
        item = event.get("item") or {}
        channel_id = item.get("channel")
        message_ts = item.get("ts")
        if not (user_id and channel_id and message_ts) or item.get("type", "message") != "message":
            return
        await client_self.db.record_response_feedback_async(
            channel_id=channel_id,
            thread_ts=None,  # reaction_added doesn't carry thread_ts; message_ts is the stable key
            message_ts=message_ts, user_id=user_id, signal=signal, source="reaction",
        )
        client_self.log_debug(
            f"reaction feedback recorded: {signal:+d} ({event.get('reaction')}) "
            f"from {user_id} on {channel_id}:{message_ts}"
        )
    except Exception as e:  # noqa: BLE001
        client_self.log_debug(f"reaction feedback ingestion failed: {e}")
