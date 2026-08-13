"""T7 — set_channel_topic / set_channel_purpose.

What actually matters here is everything that happens AROUND the one-line write:

* the human-request gate, which is the `set_channel_participation` gate and refuses the three
  shapes it refuses — no authorization, a non-human sender, and a DM;
* reading the PREVIOUS value first, because Slack's mutation response carries only the new one
  and nobody remembers what a channel topic used to say;
* excluding group DMs, which are channel-shaped everywhere else in this codebase and have no
  topic scope behind them;
* invalidating the channel-context cache, without which the bot would describe the old topic
  for the next fifteen minutes — including in the reply announcing the change.

Every refusal is additionally proven to have written nothing.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from message_processor import channel_admin_tools as cat
from tool_registry import SURFACE_CHANNEL, ToolContext, ToolRegistry

CH = "C1000000000"
DM = "D1000000000"
USER = "U_REQUESTER"
OLD_TOPIC = "Release 4.1 freeze"
NEW_TOPIC = "Release 4.2 freeze until Friday"


def _channel(**overrides):
    record = {"id": CH, "name": "some-channel", "is_channel": True,
              "topic": {"value": OLD_TOPIC}, "purpose": {"value": "Coordinating deploys"}}
    record.update(overrides)
    return record


def _ctx(*, channel=CH, is_dm=False, authorized=True, sender_type="human",
         info=None, set_result=None):
    web = SimpleNamespace(
        conversations_info=AsyncMock(
            return_value={"ok": True, "channel": info if info is not None else _channel()}),
        conversations_setTopic=AsyncMock(
            return_value=set_result if set_result is not None
            else {"ok": True, "topic": NEW_TOPIC}),
        conversations_setPurpose=AsyncMock(
            return_value=set_result if set_result is not None
            else {"ok": True, "purpose": "New purpose"}),
    )
    client = MagicMock()
    client.app.client = web
    client.invalidate_channel_context = MagicMock()
    message = SimpleNamespace(metadata={"sender_type": sender_type} if sender_type else {})
    return ToolContext(channel_id=channel, thread_ts="1700000000.000100", user_id=USER,
                       client=client, is_dm=is_dm, message=message,
                       structural_change_authorized=authorized)


def _web(ctx):
    return ctx.client.app.client


def _wrote_nothing(ctx):
    return (_web(ctx).conversations_setTopic.await_count == 0
            and _web(ctx).conversations_setPurpose.await_count == 0)


def _api_error(error="restricted_action"):
    return SlackApiError("boom", {"ok": False, "error": error})


# ================================================================================ schemas

def test_schemas_are_well_formed():
    for schema in (cat.get_set_channel_topic_schema(), cat.get_set_channel_purpose_schema()):
        assert schema["type"] == "function"
        assert schema["parameters"]["required"] == ["text"]
        assert set(schema["parameters"]["properties"]) == {"text"}


def test_schemas_state_the_request_only_rule():
    for schema in (cat.get_set_channel_topic_schema(), cat.get_set_channel_purpose_schema()):
        text = schema["description"].lower()
        assert "directly asks you to" in text
        assert "own initiative" in text


def test_registration_is_channel_surface_only():
    """A DM has no topic; offering the tool there is an invitation to a refusal."""
    registry = ToolRegistry()
    cat.register_channel_admin_tools(registry)
    assert {s["name"] for s in registry.schemas({})} == set()
    assert {s["name"] for s in registry.schemas({}, surface=SURFACE_CHANNEL)} == {
        "set_channel_topic", "set_channel_purpose"}


# ================================================================================ the gate

@pytest.mark.asyncio
async def test_dm_is_refused_before_any_slack_call():
    ctx = _ctx(channel=DM, is_dm=True)
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "topic_is_channel_only"
    assert _web(ctx).conversations_info.await_count == 0
    assert _wrote_nothing(ctx)


@pytest.mark.asyncio
async def test_unaddressed_turn_is_refused():
    """A quoted line, an ambient turn, a message that needed the gate and never woke it."""
    ctx = _ctx(authorized=False)
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "not_addressed"
    assert _wrote_nothing(ctx)


@pytest.mark.asyncio
async def test_non_human_sender_is_refused_even_when_authorized():
    ctx = _ctx(sender_type="bot")
    out = await cat.execute_set_channel_purpose(ctx, {"text": "New purpose"})
    assert out["ok"] is False and out["error"] == "not_human_sender"
    assert _wrote_nothing(ctx)


@pytest.mark.asyncio
async def test_absent_sender_classification_falls_back_to_the_flag():
    """Paths that omit the metadata must not fail closed — the flag already encodes a human."""
    ctx = _ctx(sender_type=None)
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_missing_channel_is_refused():
    ctx = _ctx(channel=None)
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "no_channel"


@pytest.mark.parametrize("args", [{}, {"text": None}, {"text": 42}])
@pytest.mark.asyncio
async def test_non_string_text_is_refused_without_calling_slack(args):
    ctx = _ctx()
    out = await cat.execute_set_channel_topic(ctx, args)
    assert out["ok"] is False and out["error"] == "missing_text"
    assert _web(ctx).conversations_info.await_count == 0


@pytest.mark.asyncio
async def test_group_dm_is_excluded():
    """An MPIM is channel-shaped (is_dm is False) but the app has no MPIM topic scope."""
    ctx = _ctx(info=_channel(is_channel=False, is_mpim=True))
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "not_a_channel"
    assert _wrote_nothing(ctx)


# ================================================================================ the write

@pytest.mark.asyncio
async def test_topic_write_echoes_the_previous_value():
    ctx = _ctx()
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is True
    assert out["field"] == "topic"
    assert out["previous"] == OLD_TOPIC
    assert out["new"] == NEW_TOPIC
    _web(ctx).conversations_info.assert_awaited_once_with(channel=CH)
    _web(ctx).conversations_setTopic.assert_awaited_once_with(channel=CH, topic=NEW_TOPIC)


@pytest.mark.asyncio
async def test_purpose_write_uses_the_purpose_endpoint():
    ctx = _ctx()
    out = await cat.execute_set_channel_purpose(ctx, {"text": "New purpose"})
    assert out["ok"] is True and out["field"] == "purpose"
    assert out["previous"] == "Coordinating deploys"
    assert out["new"] == "New purpose"
    _web(ctx).conversations_setPurpose.assert_awaited_once_with(channel=CH, purpose="New purpose")
    assert _web(ctx).conversations_setTopic.await_count == 0


@pytest.mark.asyncio
async def test_empty_string_clears_rather_than_being_treated_as_missing():
    ctx = _ctx(set_result={"ok": True, "topic": ""})
    out = await cat.execute_set_channel_topic(ctx, {"text": ""})
    assert out["ok"] is True and out["new"] == ""
    assert out["previous"] == OLD_TOPIC
    assert "cleared" in out["confirmation"]
    _web(ctx).conversations_setTopic.assert_awaited_once_with(channel=CH, topic="")


@pytest.mark.asyncio
async def test_previous_value_is_unescaped_so_it_can_be_pasted_back():
    """Slack escapes &/</> on the way out; re-sending the escaped form corrodes the topic."""
    ctx = _ctx(info=_channel(topic={"value": "Sales &amp; Ops &lt;paused&gt;"}))
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["previous"] == "Sales & Ops <paused>"


@pytest.mark.asyncio
async def test_new_value_is_read_from_a_channel_shaped_response_too():
    """Slack answers with either a bare `topic` or the whole channel record."""
    ctx = _ctx(set_result={"ok": True, "channel": _channel(topic={"value": "Normalized"})})
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["new"] == "Normalized"


@pytest.mark.asyncio
async def test_new_value_falls_back_to_what_we_sent():
    ctx = _ctx(set_result={"ok": True})
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["new"] == NEW_TOPIC


@pytest.mark.asyncio
async def test_cache_is_invalidated_after_a_successful_write():
    ctx = _ctx()
    await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    ctx.client.invalidate_channel_context.assert_called_once_with(CH)


@pytest.mark.asyncio
async def test_cache_is_not_invalidated_when_nothing_was_written():
    ctx = _ctx(authorized=False)
    await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    ctx.client.invalidate_channel_context.assert_not_called()


@pytest.mark.asyncio
async def test_a_failing_invalidation_does_not_fail_a_landed_write():
    ctx = _ctx()
    ctx.client.invalidate_channel_context.side_effect = RuntimeError("cache is gone")
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is True


# ================================================================================ failures

@pytest.mark.asyncio
async def test_unreadable_current_settings_write_nothing():
    """Without the previous value the change is unrecoverable, so it does not happen."""
    ctx = _ctx()
    _web(ctx).conversations_info.side_effect = _api_error("channel_not_found")
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "lookup_failed"
    assert "channel_not_found" in out["message"]
    assert _wrote_nothing(ctx)


@pytest.mark.asyncio
async def test_slack_refusal_surfaces_its_own_error_name():
    ctx = _ctx()
    _web(ctx).conversations_setTopic.side_effect = _api_error("missing_scope")
    out = await cat.execute_set_channel_topic(ctx, {"text": NEW_TOPIC})
    assert out["ok"] is False and out["error"] == "set_topic_failed"
    assert "missing_scope" in out["message"]
    ctx.client.invalidate_channel_context.assert_not_called()


@pytest.mark.asyncio
async def test_no_executor_ever_raises():
    ctx = ToolContext(channel_id=CH, client=None, structural_change_authorized=True)
    for execute in (cat.execute_set_channel_topic, cat.execute_set_channel_purpose):
        out = await execute(ctx, {"text": NEW_TOPIC})
        assert out["ok"] is False and out["error"] == "unavailable"
