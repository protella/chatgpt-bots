"""T1 — scheduled messages: schedule, list, cancel, and the receipt that survives the wait.

Four things here are worth testing and the rest is plumbing:

* **Timezone resolution**, because getting it wrong posts a 9am reminder at 3am and nobody finds
  out until it happens. A bare local time is read in the REQUESTER's zone, a time carrying an
  offset is left alone, and an unknown zone REFUSES rather than guessing UTC.
* **Scope**, because "current conversation only" is the whole safety story: every Slack call
  carries this channel, and the thread is the turn's own thread.
* **The pending receipt**, because a delivery arrives days later as an own-message that every
  gate drops. Scheduling registers the expectation, the delivery finalizes a real receipt row,
  and cancelling drops it — so a cancelled schedule cannot leave one behind forever.
* **Refusals**, because Slack owns the limits (30 pending, 120 days, no last-minute cancel) and
  the executor must surface Slack's own error name instead of inventing bounds of its own.

Everything runs the real executors against a mocked Slack transport.
"""
from __future__ import annotations

import asyncio
import datetime
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from database import DatabaseManager
from message_processor import outbound_receipts as orx
from message_processor import schedule_tools as st
from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM, ToolContext, ToolRegistry

TEAM = "T1"
CH = "C0BKX77NU66"
DM = "D0000001"
USER = "U_REQUESTER"
ZONE = "America/Chicago"
SMID = "Q1234567890"

# 2026-08-13 09:00 in America/Chicago (CDT, UTC-5) is 14:00 UTC.
NAIVE_ISO = "2026-08-13T09:00:00"
NAIVE_EPOCH = 1786629600
FUTURE_EPOCH = 1786629600


@pytest.fixture(autouse=True)
def _clean_registry():
    orx.reset_scheduled_deliveries()
    yield
    orx.reset_scheduled_deliveries()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture
def service(temp_db):
    orx.reset_service()
    svc = orx.install_service(temp_db)
    yield svc
    orx.reset_service()


def _web(*, schedule=None, listing=None, cancel=None, tz=ZONE, history=None, replies=None):
    return SimpleNamespace(
        chat_scheduleMessage=AsyncMock(return_value=schedule if schedule is not None else {
            "ok": True, "scheduled_message_id": SMID, "channel": CH, "post_at": FUTURE_EPOCH}),
        chat_scheduledMessages_list=AsyncMock(return_value=listing if listing is not None else {
            "ok": True, "scheduled_messages": []}),
        chat_deleteScheduledMessage=AsyncMock(return_value=cancel if cancel is not None
                                              else {"ok": True}),
        users_info=AsyncMock(return_value={"ok": True, "user": {"id": USER, "tz": tz}}),
        conversations_history=AsyncMock(return_value=history if history is not None
                                        else {"ok": True, "messages": []}),
        conversations_replies=AsyncMock(return_value=replies if replies is not None
                                        else {"ok": True, "messages": []}),
    )


def _ctx(*, channel=CH, thread_ts="1700000000.000100", trigger_ts=None, web=None,
         stored_zone=ZONE, channel_post_allowed=True, db=None):
    """A ToolContext whose `client.app.client` is the mocked Slack web client.

    `trigger_ts` defaults to a DIFFERENT ts than `thread_ts`, i.e. a turn that is already inside
    a thread — the shape the thread default keys off.
    """
    web = web if web is not None else _web()
    client = SimpleNamespace(app=SimpleNamespace(client=web), self_team_id=TEAM,
                             format_text=lambda text: text)
    if db is None:
        db = SimpleNamespace(get_user_timezone_async=AsyncMock(return_value=stored_zone))
    message = SimpleNamespace(metadata={"channel_post_allowed": channel_post_allowed})
    return ToolContext(channel_id=channel, thread_ts=thread_ts,
                       trigger_ts=trigger_ts if trigger_ts is not None else "1700000000.000900",
                       user_id=USER, client=client, db=db,
                       is_dm=str(channel).startswith("D"), message=message)


def _slack(ctx):
    return ctx.client.app.client


def _api_error(error="time_in_past"):
    return SlackApiError("boom", {"ok": False, "error": error})


def _delivery(ts, text, *, channel=CH, thread_ts=None):
    """The Slack message event a delivered scheduled post WOULD arrive as, if Bolt let it.

    It does not — `ignoring_self_events` drops own-bot message events before any handler — so this
    only exercises the kept own-message path, never the mechanism that runs in production.
    """
    event = {"channel": channel, "ts": ts, "text": text, "user": "UBOT", "bot_id": "BSELF"}
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


def _human(ts, text="anything at all", *, channel=CH):
    """A message event that DOES reach handlers — the reconciler's only trigger."""
    return {"channel": channel, "ts": ts, "text": text, "user": "U_SOMEONE"}


def _overdue(seconds_ago=600):
    """A post_at the wall clock has already passed (the reconciler asks `time.time()`)."""
    return int(time.time()) - seconds_ago


# ================================================================================ schemas

def test_schemas_are_well_formed():
    for schema in (st.get_schedule_message_schema(), st.get_list_scheduled_messages_schema(),
                   st.get_cancel_scheduled_message_schema()):
        assert schema["type"] == "function"
        assert schema["name"]
        assert schema["description"]
        assert schema["parameters"]["type"] == "object"


def test_schedule_schema_states_slacks_limits_and_the_verbatim_rule():
    """The limits are Slack's and the code never re-checks them, so the description must say
    them — and the model must know nothing rewrites the text at delivery time."""
    description = st.get_schedule_message_schema()["description"]
    assert "30" in description and "120 days" in description
    assert "verbatim" in description or "exactly as written" in description
    assert "no model call at delivery time" in description
    assert "cannot be edited" in description


def test_post_at_description_teaches_the_timezone_contract():
    post_at = st.get_schedule_message_schema()["parameters"]["properties"]["post_at"]
    assert "ISO-8601" in post_at["description"]
    assert "REQUESTER" in post_at["description"]


def test_registration_exposes_all_three_on_both_surfaces():
    registry = ToolRegistry()
    st.register_schedule_tools(registry)
    wanted = {"schedule_message", "list_scheduled_messages", "cancel_scheduled_message"}
    for surface in (SURFACE_DM, SURFACE_CHANNEL):
        assert wanted <= {schema["name"] for schema in registry.schemas({}, surface=surface)}


# ================================================================================ time

@pytest.mark.asyncio
async def test_a_bare_local_time_resolves_in_the_requesters_zone():
    ctx = _ctx()
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": NAIVE_ISO})
    assert result["ok"] is True
    assert _slack(ctx).chat_scheduleMessage.await_args.kwargs["post_at"] == str(NAIVE_EPOCH)
    assert result["post_at_utc"] == "2026-08-13T14:00:00Z"
    assert result["timezone"] == ZONE


@pytest.mark.asyncio
async def test_an_explicit_offset_is_taken_at_face_value():
    ctx = _ctx()
    await st.execute_schedule_message(
        ctx, {"text": "cake", "post_at": "2026-08-13T09:00:00+00:00"})
    sent = int(_slack(ctx).chat_scheduleMessage.await_args.kwargs["post_at"])
    assert sent == int(datetime.datetime(2026, 8, 13, 9, 0,
                                         tzinfo=datetime.timezone.utc).timestamp())


@pytest.mark.asyncio
async def test_epoch_seconds_pass_through_as_the_instant_they_are():
    ctx = _ctx()
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert _slack(ctx).chat_scheduleMessage.await_args.kwargs["post_at"] == str(FUTURE_EPOCH)
    assert result["post_at"] == FUTURE_EPOCH


@pytest.mark.asyncio
async def test_an_unknown_timezone_refuses_a_bare_local_time_instead_of_guessing_utc():
    """The one failure that is worse than not scheduling: a silent 6-hour error."""
    ctx = _ctx(stored_zone=None, web=_web(tz=None))
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": NAIVE_ISO})
    assert result["ok"] is False
    assert result["error"] == "timezone_unknown"
    _slack(ctx).chat_scheduleMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_slacks_profile_answers_when_nothing_is_stored():
    ctx = _ctx(stored_zone=None)
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": NAIVE_ISO})
    assert result["ok"] is True
    assert result["post_at"] == NAIVE_EPOCH
    _slack(ctx).users_info.assert_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_post_at_is_a_clean_refusal():
    ctx = _ctx()
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": "tomorrow at 9"})
    assert result["ok"] is False
    assert result["error"] == "bad_post_at"
    _slack(ctx).chat_scheduleMessage.assert_not_awaited()


# ================================================================================ scope

@pytest.mark.asyncio
async def test_a_thread_turn_schedules_into_that_thread_by_default():
    ctx = _ctx()
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    kwargs = _slack(ctx).chat_scheduleMessage.await_args.kwargs
    assert kwargs["channel"] == CH
    assert kwargs["thread_ts"] == "1700000000.000100"


@pytest.mark.asyncio
async def test_a_top_level_turn_schedules_at_the_top_level():
    ctx = _ctx(thread_ts="1700000000.000900", trigger_ts="1700000000.000900")
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert "thread_ts" not in _slack(ctx).chat_scheduleMessage.await_args.kwargs


@pytest.mark.asyncio
async def test_thread_false_leaves_the_thread_for_the_top_level():
    ctx = _ctx()
    await st.execute_schedule_message(
        ctx, {"text": "cake", "post_at": FUTURE_EPOCH, "thread": False})
    assert "thread_ts" not in _slack(ctx).chat_scheduleMessage.await_args.kwargs


@pytest.mark.asyncio
async def test_a_threads_only_channel_keeps_a_scheduled_post_in_the_thread():
    """A scheduled post is still a post: it does not get to route around a setting a live reply
    obeys."""
    ctx = _ctx(channel_post_allowed=False)
    await st.execute_schedule_message(
        ctx, {"text": "cake", "post_at": FUTURE_EPOCH, "thread": False})
    assert _slack(ctx).chat_scheduleMessage.await_args.kwargs["thread_ts"] == "1700000000.000100"


@pytest.mark.asyncio
async def test_a_dm_schedules_into_the_dm():
    ctx = _ctx(channel=DM, thread_ts=None, trigger_ts="1700000000.000900")
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert result["ok"] is True
    assert _slack(ctx).chat_scheduleMessage.await_args.kwargs["channel"] == DM


# ================================================================================ refusals

@pytest.mark.asyncio
async def test_slacks_own_error_is_what_a_rejected_schedule_reports():
    web = _web()
    web.chat_scheduleMessage = AsyncMock(side_effect=_api_error("time_too_far"))
    ctx = _ctx(web=web)
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert result["ok"] is False
    assert result["error"] == "schedule_failed"
    assert "time_too_far" in result["message"]


@pytest.mark.asyncio
async def test_empty_text_never_reaches_slack():
    ctx = _ctx()
    result = await st.execute_schedule_message(ctx, {"text": "  ", "post_at": FUTURE_EPOCH})
    assert result["ok"] is False
    assert result["error"] == "missing_text"
    _slack(ctx).chat_scheduleMessage.assert_not_awaited()


# ================================================================================ list + cancel

@pytest.mark.asyncio
async def test_listing_returns_ids_and_times_oldest_first():
    listing = {"ok": True, "scheduled_messages": [
        {"id": "Q2", "channel_id": CH, "post_at": FUTURE_EPOCH + 600, "text": "later"},
        {"id": "Q1", "channel_id": CH, "post_at": FUTURE_EPOCH, "text": "sooner"},
    ]}
    ctx = _ctx(web=_web(listing=listing))
    result = await st.execute_list_scheduled_messages(ctx, {})
    assert result["ok"] is True
    assert [entry["scheduled_message_id"] for entry in result["scheduled_messages"]] == ["Q1", "Q2"]
    assert result["scheduled_messages"][0]["post_at_utc"] == "2026-08-13T14:00:00Z"
    assert _slack(ctx).chat_scheduledMessages_list.await_args.kwargs["channel"] == CH


@pytest.mark.asyncio
async def test_listing_never_reports_another_conversations_message():
    listing = {"ok": True, "scheduled_messages": [
        {"id": "Q9", "channel_id": "C_OTHER", "post_at": FUTURE_EPOCH, "text": "elsewhere"},
    ]}
    ctx = _ctx(web=_web(listing=listing))
    result = await st.execute_list_scheduled_messages(ctx, {})
    assert result["scheduled_messages"] == []


@pytest.mark.asyncio
async def test_cancelling_names_this_channel_and_the_id():
    ctx = _ctx()
    result = await st.execute_cancel_scheduled_message(ctx, {"scheduled_message_id": SMID})
    assert result["ok"] is True
    kwargs = _slack(ctx).chat_deleteScheduledMessage.await_args.kwargs
    assert kwargs == {"channel": CH, "scheduled_message_id": SMID}


@pytest.mark.asyncio
async def test_a_too_late_cancellation_reports_slacks_refusal():
    web = _web()
    web.chat_deleteScheduledMessage = AsyncMock(
        side_effect=_api_error("invalid_scheduled_message_id"))
    ctx = _ctx(web=web)
    result = await st.execute_cancel_scheduled_message(ctx, {"scheduled_message_id": SMID})
    assert result["ok"] is False
    assert "invalid_scheduled_message_id" in result["message"]


@pytest.mark.asyncio
async def test_a_numeric_id_is_accepted_as_the_string_slack_wants():
    """`chat.scheduledMessages.list` renders ids as bare numbers, so the model can hand one back
    as an integer."""
    ctx = _ctx()
    await st.execute_cancel_scheduled_message(ctx, {"scheduled_message_id": 1298393284})
    assert (_slack(ctx).chat_deleteScheduledMessage.await_args.kwargs["scheduled_message_id"]
            == "1298393284")


# ================================================================================ receipts

@pytest.mark.asyncio
async def test_scheduling_registers_the_expectation_and_delivery_finalizes_a_receipt(service,
                                                                                     temp_db):
    ctx = _ctx()
    result = await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert result["ok"] is True

    ts = f"{FUTURE_EPOCH}.000100"
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(ts, "cake")) == SMID

    row = await temp_db.get_receipt_async(TEAM, CH, ts)
    assert row["state"] == "finalized"
    assert row["receipt_class"] == orx.CLASS_ASSISTANT_REPLY


@pytest.mark.asyncio
async def test_an_unmatched_own_message_gets_no_receipt(service, temp_db):
    """The registry must not adopt whatever the bot happens to post next."""
    ctx = _ctx()
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    ts = f"{FUTURE_EPOCH}.000100"
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(ts, "something else entirely")) is None
    assert await temp_db.get_receipt_async(TEAM, CH, ts) is None


@pytest.mark.asyncio
async def test_the_same_text_posted_before_the_schedule_fires_is_not_the_delivery(service):
    """Slack does not deliver early, so an earlier ts is an ordinary reply — and it already has
    its own receipt."""
    ctx = _ctx()
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    early = f"{FUTURE_EPOCH - 60}.000100"
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(early, "cake")) is None
    # …and the expectation survives for the real delivery.
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(f"{FUTURE_EPOCH}.000100", "cake")) == SMID


@pytest.mark.asyncio
async def test_a_delivery_in_another_channel_is_not_ours(service):
    ctx = _ctx()
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(f"{FUTURE_EPOCH}.000100", "cake",
                                      channel="C_OTHER")) is None


@pytest.mark.asyncio
async def test_cancelling_drops_the_expectation_so_it_cannot_linger(service):
    ctx = _ctx()
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    await st.execute_cancel_scheduled_message(ctx, {"scheduled_message_id": SMID})
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(f"{FUTURE_EPOCH}.000100", "cake")) is None


@pytest.mark.asyncio
async def test_a_dm_schedule_registers_nothing(service):
    """Receipts do not exist in DMs, so there is nothing to expect there."""
    ctx = _ctx(channel=DM, thread_ts=None, trigger_ts="1700000000.000900")
    await st.execute_schedule_message(ctx, {"text": "cake", "post_at": FUTURE_EPOCH})
    assert orx._scheduled_deliveries == {}


@pytest.mark.asyncio
async def test_a_restart_relearns_pending_schedules_from_slack_once(service, temp_db):
    """Slack holds the schedule, so the expectation is recoverable without a table of ours."""
    web = _web(listing={"ok": True, "scheduled_messages": [
        {"id": SMID, "channel_id": CH, "post_at": FUTURE_EPOCH, "text": "cake"}]})
    assert await orx.rehydrate_scheduled_deliveries(web, team_id=TEAM) == 1
    # Once per process: a second event does not re-list.
    assert await orx.rehydrate_scheduled_deliveries(web, team_id=TEAM) == 0
    assert web.chat_scheduledMessages_list.await_count == 1

    ts = f"{FUTURE_EPOCH}.000100"
    assert await orx.finalize_scheduled_delivery(
        team_id=TEAM, event=_delivery(ts, "cake")) == SMID
    assert (await temp_db.get_receipt_async(TEAM, CH, ts))["state"] == "finalized"


@pytest.mark.asyncio
async def test_a_delivery_arriving_mid_listing_waits_for_it(service, temp_db):
    """THE first-event race: boot's listing is still in flight when the delivery lands. The event
    must JOIN that listing — a rehydrate it merely skips leaves it matching an empty registry, and
    by then Slack no longer lists the message as pending, so nothing can recover it."""
    gate = asyncio.Event()

    async def _slow_list(**kwargs):
        await gate.wait()
        return {"ok": True, "scheduled_messages": [
            {"id": SMID, "channel_id": CH, "post_at": FUTURE_EPOCH, "text": "cake"}]}

    web = _web()
    web.chat_scheduledMessages_list = AsyncMock(side_effect=_slow_list)
    assert orx.start_scheduled_rehydrate(web, team_id=TEAM) is not None
    # Single-flight: boot owns the one listing, and nothing starts a second.
    assert orx.start_scheduled_rehydrate(web, team_id=TEAM) is None

    host = _host()
    host.app = SimpleNamespace(client=web)
    ts = f"{FUTURE_EPOCH}.000100"
    hook = asyncio.create_task(host._finalize_scheduled_delivery(_delivery(ts, "cake")))
    await asyncio.sleep(0)
    assert await temp_db.get_receipt_async(TEAM, CH, ts) is None, "the hook finalized too early"

    gate.set()
    await hook
    assert (await temp_db.get_receipt_async(TEAM, CH, ts))["state"] == "finalized"


@pytest.mark.asyncio
async def test_a_failed_rehydrate_never_raises_into_the_ingress():
    web = _web()
    web.chat_scheduledMessages_list = AsyncMock(side_effect=_api_error("ratelimited"))
    assert await orx.rehydrate_scheduled_deliveries(web, team_id=TEAM) == 0


# ================================================================================ the ingress hook

def _host(web=None):
    """The listener seam on a bare host — no ambient service, no DB, no processor."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class Host(SlackMessageEventsMixin):
        def __init__(self):
            self.app = SimpleNamespace(client=web if web is not None else _web())
            self.self_team_id = TEAM

        def is_own_message(self, event):
            return event.get("bot_id") == "BSELF"

        def log_debug(self, *a, **k):
            pass

    return Host()


@pytest.mark.asyncio
async def test_the_seam_finalizes_a_delivered_schedule(service, temp_db):
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=FUTURE_EPOCH,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)
    ts = f"{FUTURE_EPOCH}.000100"
    await _host()._finalize_scheduled_delivery(_delivery(ts, "cake"))
    assert (await temp_db.get_receipt_async(TEAM, CH, ts))["state"] == "finalized"


@pytest.mark.asyncio
async def test_the_seam_ignores_edits_and_deletions(service):
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=FUTURE_EPOCH,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)
    event = _delivery(f"{FUTURE_EPOCH}.000100", "cake")
    event["subtype"] = "message_changed"
    await _host()._finalize_scheduled_delivery(event)
    assert SMID in orx._scheduled_deliveries


# ====================================================== reconciling from the events that DO arrive
#
# Bolt's `ignoring_self_events` middleware drops own-bot message events before any handler runs
# (measured live 2026-08-13), so the delivery NEVER announces itself. The next human message in
# the channel is the trigger, and Slack's own history is where the delivered message is found.


@pytest.mark.asyncio
async def test_a_human_message_after_post_at_reconciles_the_delivery(service, temp_db):
    """THE mechanism. One history read, and the receipt lands on the ts Slack actually posted."""
    post_at = _overdue()
    delivered_ts = f"{post_at + 2}.000100"
    web = _web(history={"ok": True, "messages": [
        {"ts": f"{post_at + 300}.000200", "text": "did that go out?", "user": "U_SOMEONE"},
        {"ts": delivered_ts, "text": "cake", "bot_id": "BSELF"},
    ]})
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=post_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)

    await _host(web)._finalize_scheduled_delivery(_human(f"{post_at + 300}.000200",
                                                         "did that go out?"))

    assert web.conversations_history.await_count == 1
    assert web.conversations_replies.await_count == 0
    assert web.conversations_history.await_args.kwargs["channel"] == CH
    row = await temp_db.get_receipt_async(TEAM, CH, delivered_ts)
    assert row["state"] == "finalized"
    assert row["receipt_class"] == orx.CLASS_ASSISTANT_REPLY
    # Consumed: a second human message must not probe for it again.
    assert SMID not in orx._scheduled_deliveries


@pytest.mark.asyncio
async def test_a_channel_with_nothing_overdue_never_calls_slack(service):
    """The seam runs on EVERY message event, so the ordinary case has to be free."""
    web = _web()
    host = _host(web)
    # Settle boot's one listing first — after it, this is the steady state every event lands in.
    await orx.rehydrate_scheduled_deliveries(web, team_id=TEAM)
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=int(time.time()) + 3600,  # due in an hour
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)
    web.chat_scheduledMessages_list.reset_mock()

    await host._finalize_scheduled_delivery(_human(f"{int(time.time())}.000100"))
    await host._finalize_scheduled_delivery(_human(f"{int(time.time())}.000200",
                                                   channel="C_ELSEWHERE"))

    assert web.conversations_history.await_count == 0
    assert web.conversations_replies.await_count == 0
    assert web.chat_scheduledMessages_list.await_count == 0
    assert SMID in orx._scheduled_deliveries


@pytest.mark.asyncio
async def test_a_delivery_scheduled_into_a_thread_is_found_by_replies(service, temp_db):
    """`conversations.history` does not return thread replies, so a threaded expectation that
    looked there would never find its own delivery."""
    post_at = _overdue()
    root = "1700000000.000100"
    delivered_ts = f"{post_at + 2}.000100"
    web = _web(replies={"ok": True, "messages": [
        {"ts": root, "text": "remind us about cake", "user": USER},
        {"ts": delivered_ts, "text": "cake", "bot_id": "BSELF", "thread_ts": root},
    ]})
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=post_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY, thread_root_ts=root)

    await _host(web)._finalize_scheduled_delivery(_human(f"{post_at + 60}.000300"))

    assert web.conversations_history.await_count == 0
    assert web.conversations_replies.await_args.kwargs["ts"] == root
    row = await temp_db.get_receipt_async(TEAM, CH, delivered_ts)
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == root


@pytest.mark.asyncio
async def test_scheduling_into_a_thread_records_the_thread_for_the_reconciler(service):
    """The executor is the only thing that ever knows which thread the post went into."""
    await st.execute_schedule_message(_ctx(), {"text": "cake", "post_at": FUTURE_EPOCH})
    assert orx._scheduled_deliveries[SMID].thread_root_ts == "1700000000.000100"


@pytest.mark.asyncio
async def test_a_human_repeating_the_text_is_not_the_delivery(service, temp_db):
    """The probe reads a whole channel window, so authorship is the difference between finalizing
    our post and finalizing somebody else's."""
    post_at = _overdue()
    human_ts = f"{post_at + 2}.000100"
    web = _web(history={"ok": True, "messages": [{"ts": human_ts, "text": "cake",
                                                  "user": "U_SOMEONE"}]})
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=post_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)

    await _host(web)._finalize_scheduled_delivery(_human(f"{post_at + 5}.000300"))

    assert await temp_db.get_receipt_async(TEAM, CH, human_ts) is None
    assert SMID in orx._scheduled_deliveries


@pytest.mark.asyncio
async def test_an_empty_probe_is_retried_inside_the_grace_and_abandoned_after_it(service):
    """Slack does not deliver to the millisecond, so an early miss must not be believed — and a
    delivery that never appears must not cost a Slack call per human message forever."""
    post_at = _overdue()
    web = _web()

    assert await orx.reconcile_overdue_scheduled(
        web, team_id=TEAM, channel_id=CH, now=post_at + 1) == 0
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=post_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)

    await orx.reconcile_overdue_scheduled(web, team_id=TEAM, channel_id=CH, now=post_at + 1)
    assert web.conversations_history.await_count == 1
    assert not orx._scheduled_deliveries[SMID].abandoned

    grace = orx._SCHEDULED_PROBE_GRACE_SECONDS
    await orx.reconcile_overdue_scheduled(web, team_id=TEAM, channel_id=CH, now=post_at + grace)
    assert web.conversations_history.await_count == 2
    assert orx._scheduled_deliveries[SMID].abandoned

    await orx.reconcile_overdue_scheduled(web, team_id=TEAM, channel_id=CH,
                                          now=post_at + grace + 60)
    assert web.conversations_history.await_count == 2, "an abandoned entry is never probed again"


@pytest.mark.asyncio
async def test_two_identical_schedules_cannot_claim_one_delivery(service, temp_db):
    """Same words, same channel, one post delivered so far. Both expectations see the SAME window,
    so without a claim the pair would write one ts twice and leave the other delivery with no
    receipt at all. The older schedule takes the older post; the other stays owed."""
    first_at = _overdue(1200)
    second_at = _overdue(60)
    delivered_ts = f"{second_at + 5}.000100"
    web = _web(history={"ok": True, "messages": [
        {"ts": delivered_ts, "text": "cake", "bot_id": "BSELF"}]})
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id="Q_FIRST",
                                  text="cake", post_at=first_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id="Q_SECOND",
                                  text="cake", post_at=second_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)

    assert await orx.reconcile_overdue_scheduled(web, team_id=TEAM, channel_id=CH) == 1

    assert (await temp_db.get_receipt_async(TEAM, CH, delivered_ts))["state"] == "finalized"
    assert list(orx._scheduled_deliveries) == ["Q_SECOND"]
    assert not orx._scheduled_deliveries["Q_SECOND"].abandoned, "still waiting on its own post"


@pytest.mark.asyncio
async def test_a_cancelled_receipt_write_keeps_the_expectation(monkeypatch):
    """The expectation is the only record that a delivery is owed. Dropping it before the write
    means a cancellation between the two loses it for good — nothing in the database, nothing in
    memory, and no way for a later pass to try again."""
    post_at = _overdue()
    web = _web(history={"ok": True, "messages": [
        {"ts": f"{post_at + 2}.000100", "text": "cake", "bot_id": "BSELF"}]})
    orx.expect_scheduled_delivery(team_id=TEAM, channel_id=CH, scheduled_message_id=SMID,
                                  text="cake", post_at=post_at,
                                  receipt_class=orx.CLASS_ASSISTANT_REPLY)

    async def _cancelled(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(orx, "record_transport_post", _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await orx.reconcile_overdue_scheduled(web, team_id=TEAM, channel_id=CH)

    assert SMID in orx._scheduled_deliveries
