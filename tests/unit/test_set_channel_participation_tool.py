"""Decision #4 — the gated set_channel_participation tool.

Covers: schema shape + explicit-only description guardrails, DM refusal, channel/db
preconditions, at-least-one-field requirement, enum validation, the atomic partial write
(only named fields; response_mode kept in lockstep with participation_level), old+new
effective-settings resolution (including NULL reply_in_channel inheriting the global
default), and the confirmation line. The DB is mocked; no real API/DB calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from config import config
from message_processor.participation_tools import (
    execute_set_channel_participation, get_set_channel_participation_schema,
    register_participation_tools)
from tool_registry import ToolContext, ToolRegistry

CHANNEL = "C04QDHE8W8M"


def _db(before=None, after=None):
    """A db mock whose get_channel_settings_async returns `before` first, then `after`."""
    db = MagicMock()
    db.get_channel_settings_async = AsyncMock(side_effect=[before, after if after is not None else before])
    db.set_channel_settings_async = AsyncMock()
    return db


def _ctx(db, **kw):
    # Default to an ADDRESSED turn (structural_change_authorized=True): these cases exercise the
    # authorized path. BLOCKER #3's refusal on an unaddressed turn is covered explicitly below.
    defaults = dict(channel_id=CHANNEL, thread_ts="1.0", trigger_ts="1.0",
                    user_id="U07PETER", db=db, is_dm=False, structural_change_authorized=True)
    defaults.update(kw)
    return ToolContext(**defaults)


# --------------------------------------------------------------- schema / guardrails

def test_schema_shape_and_enums():
    s = get_set_channel_participation_schema()
    assert s["type"] == "function" and s["name"] == "set_channel_participation"
    props = s["parameters"]["properties"]
    assert props["participation"]["enum"] == ["mentions_only", "judicious", "active", "off"]
    assert props["placement"]["enum"] == ["threads_only", "channel_allowed"]
    # current-channel-only: no channel_id parameter is exposed
    assert "channel_id" not in props
    # nothing is required at the schema level (at least one is enforced in the executor)
    assert not s["parameters"].get("required")


def test_description_states_explicit_only_guardrails():
    desc = get_set_channel_participation_schema()["description"].lower()
    assert "explicit" in desc and "current" in desc
    # must never be inferred from these sources
    for forbidden in ("memory", "history", "quoted", "attachment"):
        assert forbidden in desc
    assert "not available in dms" in desc


# --------------------------------------------------------------- authorization (BLOCKER #3)

@pytest.mark.asyncio
async def test_unaddressed_turn_refused():
    # The hard, in-code gate: on a turn the bot was NOT directly addressed on (an injected,
    # hallucinated, or quoted call), the settings write must be refused — no DB write.
    db = _db(before={"participation_level": "judicious"})
    res = await execute_set_channel_participation(
        _ctx(db, structural_change_authorized=False), {"participation": "mentions_only"})
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_addressed_turn_authorizes_the_write():
    # The same call on a directly-addressed turn goes through and writes.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _db(before=before, after=after)
    res = await execute_set_channel_participation(
        _ctx(db, structural_change_authorized=True), {"participation": "mentions_only"})
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_authorization_defaults_closed_when_flag_absent():
    # A ToolContext that never set the flag (older construction path) fails closed — refused.
    db = _db(before={})
    ctx = ToolContext(channel_id=CHANNEL, user_id="U07PETER", db=db, is_dm=False)
    res = await execute_set_channel_participation(ctx, {"participation": "off"})
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


# --------------------------------------------------------------- preconditions

@pytest.mark.asyncio
async def test_dm_refused():
    db = _db()
    res = await execute_set_channel_participation(
        _ctx(db, is_dm=True), {"participation": "mentions_only"})
    assert res["ok"] is False and res["error"] == "participation_is_channel_only"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_requires_at_least_one_field():
    db = _db(before={})
    res = await execute_set_channel_participation(_ctx(db), {})
    assert res["ok"] is False and res["error"] == "bad_arguments"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_enums_rejected():
    db = _db(before={})
    r1 = await execute_set_channel_participation(_ctx(db), {"participation": "loud"})
    r2 = await execute_set_channel_participation(_ctx(db), {"placement": "everywhere"})
    assert r1["error"] == "bad_arguments" and r2["error"] == "bad_arguments"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_channel_and_no_db_refused():
    assert (await execute_set_channel_participation(
        _ctx(_db(), channel_id=None), {"participation": "off"}))["error"] == "no_channel"
    assert (await execute_set_channel_participation(
        _ctx(None), {"participation": "off"}))["error"] == "settings_unavailable"


# --------------------------------------------------------------- happy paths / atomicity

@pytest.mark.asyncio
async def test_participation_write_is_atomic_and_lockstep(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _db(before=before, after=after)
    res = await execute_set_channel_participation(_ctx(db), {"participation": "mentions_only"})
    assert res["ok"] is True
    kwargs = db.set_channel_settings_async.await_args.kwargs
    # ONLY the named field (+ its lockstep response_mode) is written; placement is untouched.
    assert kwargs["participation_level"] == "mentions_only"
    assert kwargs["response_mode"] == "tag_only"  # LEVEL_TO_MODE[mentions_only]
    assert "reply_in_channel" not in kwargs
    assert kwargs["updated_by"] == "U07PETER"
    assert res["old"]["participation"] == "judicious"
    assert res["new"]["participation"] == "mentions_only"


@pytest.mark.asyncio
async def test_placement_only_write_leaves_participation_untouched():
    before = {"participation_level": "active", "reply_in_channel": True}
    after = {"participation_level": "active", "reply_in_channel": False}
    db = _db(before=before, after=after)
    res = await execute_set_channel_participation(_ctx(db), {"placement": "threads_only"})
    kwargs = db.set_channel_settings_async.await_args.kwargs
    assert kwargs["reply_in_channel"] is False
    assert "participation_level" not in kwargs and "response_mode" not in kwargs
    assert res["new"]["placement"] == "threads_only"
    assert "threads only" in res["confirmation"].lower()


@pytest.mark.asyncio
async def test_both_fields_and_active_maps_to_auto_respond():
    before = {"participation_level": "judicious", "reply_in_channel": False}
    after = {"participation_level": "active", "reply_in_channel": True}
    db = _db(before=before, after=after)
    res = await execute_set_channel_participation(
        _ctx(db), {"participation": "active", "placement": "channel_allowed"})
    kwargs = db.set_channel_settings_async.await_args.kwargs
    assert kwargs["participation_level"] == "active"
    assert kwargs["response_mode"] == "auto_respond"
    assert kwargs["reply_in_channel"] is True
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_null_reply_in_channel_inherits_default_in_effective(monkeypatch):
    # A row with reply_in_channel=None must resolve to the GLOBAL default, not "threads_only".
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    before = {"participation_level": None, "reply_in_channel": None}
    after = {"participation_level": "off", "reply_in_channel": None}
    db = _db(before=before, after=after)
    res = await execute_set_channel_participation(_ctx(db), {"participation": "off"})
    # before: participation inherited from global auto_respond → judicious; placement from default True
    assert res["old"] == {"participation": "judicious", "placement": "channel_allowed"}
    assert res["new"]["participation"] == "off"


@pytest.mark.asyncio
async def test_no_op_change_reports_no_change():
    before = {"participation_level": "judicious", "reply_in_channel": True}
    db = _db(before=before, after=before)
    res = await execute_set_channel_participation(_ctx(db), {"participation": "judicious"})
    assert res["ok"] is True
    assert "nothing changed" in res["confirmation"].lower()


# --------------------------------------------------------------- registration gating

def test_registered_only_with_engine_on(monkeypatch):
    reg = ToolRegistry()
    register_participation_tools(reg)
    assert "set_channel_participation" in [s["name"] for s in reg.schemas({})]
