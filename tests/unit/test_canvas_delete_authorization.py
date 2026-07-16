"""Authorization provenance for the irreversible canvas-delete tool.

Deleting a canvas is destructive and public, so the tool is offered ONLY when a HUMAN directly
addressed the bot in the CURRENT message. These are END-TO-END tests: they run the REAL signal
derivation in the text handler (`_materialize_request_tools`) from a Message's metadata, then feed
the resulting request_config into the REAL schema gate (`canvas_tools._delete_enabled`) and the
REAL registry. Nothing injects the authorization flag by hand — the point is to prove the wiring,
since the two historical bypasses (the same class we closed for set_channel_participation) lived
precisely in how the signal was computed:

  (a) the raw `participation_name_hit` regex also fires on a message that merely QUOTES / talks
      ABOUT the bot's name ("Alice said 'ChatGPT, delete the canvas'"), not a genuine summons;
  (b) an `other_bot` @mention is dispatched to the main handler un-gated, so a bare `not unprompted`
      authorized a NON-human sender to enable an irreversible delete.

The final authorization expression is:
    _canvas_delete_authorized =
        (sender_type == "human") AND (mentioned_self OR is_dm)
where `sender_type`/`mentioned_self` are stamped in _event_to_message and `is_dm` is derived from
the channel id. A bare name-hit does NOT authorize — a name-addressed delete must carry a real
<@bot> mention (or be a DM, where every message is addressed to the bot). Absent/failed sender
classification fails CLOSED.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from base_client import Message
from message_processor import canvas_tools as ct
from message_processor.handlers.text import TextHandlerMixin
from tool_registry import ToolRegistry

CHANNEL = "C04QDHE8W8M"
DM = "D07PETERDM"
CATALOG = [{"canvas_id": "F1", "title": "Old notes"}]  # one deletable (non-channel) canvas


class _Handler(TextHandlerMixin):
    """Minimal host for the mixin method under test (it only needs `self.db`)."""

    def __init__(self):
        self.db = None


def _delete_offered(meta, *, channel_id=CHANNEL):
    """Derive the request_config the SAME way production does, then ask the REAL gate + registry
    whether delete_canvas is on the table. Returns (authorized_flag, gate_bool, in_schema_set)."""
    msg = Message(text="delete that canvas", user_id="U07PETER",
                  channel_id=channel_id, thread_id="100.0", metadata=dict(meta))
    # tools_disabled=True short-circuits the registry lookup but still runs the flag derivation.
    _reg, request_config, _n, _s = _Handler()._materialize_request_tools(
        MagicMock(), {}, msg, tools_disabled=True)
    cfg = {ct.CATALOG_KEY: CATALOG, **request_config}
    registry = ToolRegistry()
    ct.register_canvas_tools(registry)
    in_schema = "delete_canvas" in {s["name"] for s in registry.schemas(cfg)}
    return request_config.get("_canvas_delete_authorized"), ct._delete_enabled(cfg), in_schema


# --------------------------------------------------------------------------- NOT enabled

def test_other_bot_mention_does_not_enable_delete():
    # Bypass (b): a REAL @mention, but the author is another bot (dispatched un-gated). The old
    # `not unprompted` authorized the non-human sender; the human-sender requirement refuses it.
    flag, gate, in_schema = _delete_offered({"sender_type": "other_bot", "mentioned_self": True})
    assert flag is False and gate is False and in_schema is False


def test_self_authored_turn_does_not_enable_delete():
    # The bot's own message must never authorize it to delete a canvas.
    flag, gate, in_schema = _delete_offered({"sender_type": "self", "mentioned_self": True})
    assert flag is False and gate is False and in_schema is False


def test_quoted_or_ambient_name_drop_does_not_enable_delete():
    # Bypass (a): a name-hit with NO real <@bot> mention and NO DM — "someone talked ABOUT the
    # bot". The old signal authorized it (name regex); the new one does not.
    flag, gate, in_schema = _delete_offered(
        {"sender_type": "human", "mentioned_self": False,
         "participation_check": True, "participation_name_hit": True})
    assert flag is False and gate is False and in_schema is False


def test_human_ambient_respond_turn_does_not_enable_delete():
    # An ordinary ambient respond turn (human, no mention, not a DM) must NOT expose delete even
    # though the model is answering — the model is acting on its own initiative here.
    flag, gate, in_schema = _delete_offered(
        {"sender_type": "human", "mentioned_self": False, "participation_check": True})
    assert flag is False and gate is False and in_schema is False


def test_absent_sender_classification_fails_closed():
    # classify_sender can return None before bot identity is wired; a genuine <@bot> mention with
    # an unknown sender still fails closed — a destructive tool withheld is the safe default.
    flag, gate, in_schema = _delete_offered({"mentioned_self": True})
    assert flag is False and gate is False and in_schema is False


# --------------------------------------------------------------------------- enabled

def test_genuine_human_mention_enables_delete():
    # A real human @mention: sender_type human AND a real mentioned_self → authorized, tool offered.
    flag, gate, in_schema = _delete_offered({"sender_type": "human", "mentioned_self": True})
    assert flag is True and gate is True and in_schema is True


def test_human_dm_enables_delete():
    # Every message in a DM is addressed to the bot, so "delete that canvas" there is a genuine
    # request even without a literal @mention.
    flag, gate, in_schema = _delete_offered(
        {"sender_type": "human", "mentioned_self": False}, channel_id=DM)
    assert flag is True and gate is True and in_schema is True


def test_name_addressed_participation_turn_with_a_real_mention_enables_delete():
    # A name-addressed request the classifier engaged with (participation turn, name-hit) IS served
    # when it carries a genuine <@bot> mention. The differentiator from the quoted/ambient case
    # above is `mentioned_self` (a deterministic parse-time fact), NOT the bare name-hit regex —
    # exactly the intended tightening.
    flag, gate, in_schema = _delete_offered(
        {"sender_type": "human", "mentioned_self": True,
         "participation_check": True, "participation_name_hit": True})
    assert flag is True and gate is True and in_schema is True
