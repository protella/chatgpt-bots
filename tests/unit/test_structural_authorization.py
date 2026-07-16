"""BLOCKER #3 (round 3) — authorization provenance for set_channel_participation.

The gated structural tool may fire ONLY when a HUMAN directly addressed the bot for it. These
are END-TO-END tests: they run the REAL signal derivation in the text handler
(`_materialize_request_tools` → `_build_tool_context`) from a Message's metadata, then feed the
resulting ToolContext into the REAL executor and assert allowed/refused. Nothing injects the
authorization flag by hand — the point is to prove the wiring, since the two historical bypasses
lived precisely in how the flag was computed:

  (a) the raw `participation_name_hit` regex also fires on a message that merely QUOTES/mentions
      the bot's name ("Alice said 'ChatGPT, only reply when tagged'"), not a genuine summons;
  (b) an `other_bot` @mention is dispatched to the main handler un-gated, so a bare `not
      unprompted` authorized a NON-human sender.

The final authorization expression is:
    structural_change_authorized =
        (sender_type == "human") AND (mentioned_self OR gate_authorized_structural)
where `sender_type`/`mentioned_self` are stamped in _event_to_message and
`gate_authorized_structural` is stamped by the participation gate (main._apply_backoff) when the
classifier judged the CURRENT message an explicit structural request.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from message_processor.handlers.text import TextHandlerMixin
from message_processor.participation_tools import execute_set_channel_participation
from tool_registry import ToolContext

CHANNEL = "C04QDHE8W8M"


class _Handler(TextHandlerMixin):
    """Minimal host for the two mixin methods under test (they only need `self.db`)."""

    def __init__(self, db):
        self.db = db


def _writable_db(before=None, after=None):
    """A db whose get_channel_settings_async returns `before` then `after` (mirrors the tool test)."""
    db = MagicMock()
    db.get_channel_settings_async = AsyncMock(
        side_effect=[before or {}, after if after is not None else (before or {})])
    db.set_channel_settings_async = AsyncMock()
    return db


def _ctx_from_metadata(meta, db):
    """Derive a ToolContext the SAME way production does: real materialize + real build."""
    handler = _Handler(db)
    msg = Message(text="only reply when I tag you", user_id="U07PETER",
                  channel_id=CHANNEL, thread_id="100.0", metadata=dict(meta))
    # tools_disabled=True short-circuits the registry lookup but still runs the flag derivation.
    _reg, request_config, _nra, _sfx = handler._materialize_request_tools(
        MagicMock(), {}, msg, tools_disabled=True)
    return handler._build_tool_context(msg, MagicMock(), request_config)


async def _run(meta, db):
    ctx = _ctx_from_metadata(meta, db)
    return ctx, await execute_set_channel_participation(ctx, {"participation": "mentions_only"})


# ------------------------------------------------------------------- refused end-to-end

@pytest.mark.asyncio
async def test_quoted_or_ambient_name_hit_is_refused():
    # Bypass (a): a name-hit with NO real <@bot> mention and NO structural judgment from the
    # classifier — "someone talked ABOUT the bot". The old signal authorized it (name regex);
    # the new one does not.
    db = _writable_db(before={"participation_level": "judicious"})
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "participation_check": True, "participation_name_hit": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bot_authored_mention_is_refused():
    # Bypass (b): a REAL @mention, but the author is another bot (dispatched un-gated). The old
    # `not unprompted` authorized the non-human sender; the human-sender requirement refuses it.
    db = _writable_db(before={"participation_level": "judicious"})
    ctx, res = await _run({"sender_type": "other_bot", "mentioned_self": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_unaddressed_non_structural_respond_turn_is_refused():
    # An ordinary ambient respond turn (human, no mention, classifier did NOT flag a structural
    # request) must NOT authorize a settings change even though the model is answering.
    db = _writable_db(before={"participation_level": "judicious"})
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False, "participation_check": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


# ------------------------------------------------------------------- allowed end-to-end

@pytest.mark.asyncio
async def test_genuine_human_mention_is_allowed():
    # A real human @mention: sender_type human AND a real mentioned_self → authorized, write goes.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    ctx, res = await _run({"sender_type": "human", "mentioned_self": True}, db)
    assert ctx.structural_change_authorized is True
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_unaddressed_structural_request_is_allowed():
    # "only reply when I tag you": a human ambient turn with no literal <@bot> mention, but the
    # classifier judged it an explicit structural request (gate stamped gate_authorized_structural).
    # That semantic judgment authorizes without a mention.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "participation_check": True, "gate_authorized_structural": True}, db)
    assert ctx.structural_change_authorized is True
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()


# ------------------------------------------------------------------- defense-in-depth

@pytest.mark.asyncio
async def test_executor_refuses_non_human_sender_even_if_flag_set():
    # Belt-and-suspenders: even if the authorization flag were somehow True, the executor reads the
    # raw sender classification off ctx.message and refuses a non-human author outright.
    db = _writable_db(before={"participation_level": "judicious"})
    msg = Message(text="x", user_id="B1", channel_id=CHANNEL, thread_id="100.0",
                  metadata={"sender_type": "other_bot"})
    ctx = ToolContext(channel_id=CHANNEL, user_id="B1", db=db, is_dm=False,
                      structural_change_authorized=True, message=msg)
    res = await execute_set_channel_participation(ctx, {"participation": "off"})
    assert res["ok"] is False and res["error"] == "not_human_sender"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_allows_human_sender_on_message_metadata():
    # The mirror: a human author on ctx.message with the flag set passes the defense-in-depth check.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "off", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    msg = Message(text="x", user_id="U07PETER", channel_id=CHANNEL, thread_id="100.0",
                  metadata={"sender_type": "human"})
    ctx = ToolContext(channel_id=CHANNEL, user_id="U07PETER", db=db, is_dm=False,
                      structural_change_authorized=True, message=msg)
    res = await execute_set_channel_participation(ctx, {"participation": "off"})
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()
