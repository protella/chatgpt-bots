"""T5 — delete_own_message and remove_reaction: the two things the bot can take back.

Both tools are about UNDOING something the bot itself did, and each carries one correction that
is easy to get wrong and expensive to get wrong live:

* **delete_own_message** proves the author with an exact Slack read BEFORE `chat.delete`, and
  drops the outbound receipt AFTER it. That order is not cosmetic: a receipt dropped first would,
  on a failed delete, leave a live message with no row describing it — a state nothing downstream
  can tell from an ordinary post. It is CHANNELS ONLY, and permanent.
* **remove_reaction** cannot go through `remove_owned_reaction`: `react_to_message` settles its
  lease into an UNOWNED slot the instant the reaction lands, so there is never a lease left to
  present by the time anyone asks for a removal. The raw call is what runs — and the guard slot
  has to go with the reaction, or the guard would answer a later honest re-add of the same emoji
  with idempotent success for a reaction that is no longer there.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from slack_sdk.errors import SlackApiError

from slack_client.messaging import SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin
from message_processor.tool_registry import ToolContext

TEAM = "T1"
CH = "C1"
DM = "D1"
BOT_USER = "U_BOT"
BOT_ID = "B_BOT"
APP_ID = "A_APP"
ROOT_TS = "1700000000.000100"
TS = "1700000060.000200"
REPLY_TS = "1700000099.000700"


# ------------------------------------------------------------------------------- harness

def _host(own=True, in_thread=False):
    """The REAL schema getters, shims and executors on a mocked Slack transport."""
    s = MagicMock()
    s.self_team_id = TEAM
    s.bot_user_id, s.bot_id, s.app_id = BOT_USER, BOT_ID, APP_ID
    # The guard is created lazily in production; a MagicMock would otherwise auto-invent one.
    s._reaction_guard = OrderedDict()
    s._reaction_guard_ts = {}
    s._REACTION_GUARD_MAX = SlackMessagingMixin._REACTION_GUARD_MAX
    s._REACTION_GUARD_RECENCY_S = SlackMessagingMixin._REACTION_GUARD_RECENCY_S
    s._REMOVING = SlackMessagingMixin._REMOVING
    s._is_committed = SlackMessagingMixin._is_committed
    for name in ("get_delete_own_message_tool_schema", "execute_delete_own_message",
                 "_execute_delete_own_message", "_read_deletable_message",
                 "_drop_own_message_receipt", "_read_exact_message",
                 "get_remove_reaction_tool_schema", "execute_remove_reaction_tool",
                 "_clear_reaction_guard_slot", "_reserve_and_react",
                 "_reserve_and_react_owned", "_reserve_once", "_react_add",
                 "_trim_reaction_guard", "settle_reaction_lease"):
        setattr(s, name, getattr(SlackMessagingMixin, name).__get__(s))
    s.is_own_message = SlackUtilitiesMixin.is_own_message.__get__(s)

    author = ({"bot_id": BOT_ID, "app_id": APP_ID, "user": BOT_USER} if own
              else {"user": "U_HUMAN"})
    message = dict(author, ts=TS, text="the earlier post")
    s.app.client.conversations_history = AsyncMock(
        return_value={"messages": [] if in_thread else [message]})
    s.app.client.conversations_replies = AsyncMock(
        return_value={"messages": [message] if in_thread else []})
    s.app.client.chat_delete = AsyncMock(return_value={"ok": True})
    s.app.client.reactions_add = AsyncMock(return_value={"ok": True})
    s.app.client.reactions_remove = AsyncMock(return_value={"ok": True})
    return s


def _ctx(channel=CH, is_dm=False, thread_ts=ROOT_TS, trigger_ts=TS):
    return ToolContext(channel_id=channel, thread_ts=thread_ts, trigger_ts=trigger_ts,
                       is_dm=is_dm, client=None)


def _api_error(error="message_not_found"):
    return SlackApiError("boom", {"ok": False, "error": error})


@pytest.fixture
def receipts(monkeypatch):
    """Records every receipt deletion, in call order with the Slack deletes."""
    calls: list = []
    order: list = []

    async def _fake(*, team_id, channel_id, message_ts, site="raw_delete"):
        calls.append({"team_id": team_id, "channel_id": channel_id,
                      "message_ts": message_ts, "site": site})
        order.append("receipt")

    monkeypatch.setattr("message_processor.outbound_receipts.delete_receipt_for", _fake)
    return {"calls": calls, "order": order}


def _watch_delete(host, order, result=None, error=None):
    async def _delete(**kwargs):
        order.append("chat_delete")
        if error is not None:
            raise error
        return result or {"ok": True}
    host.app.client.chat_delete = AsyncMock(side_effect=_delete)


# ================================================================= T5a: delete_own_message

@pytest.mark.unit
class TestDeleteSchema:
    def test_schema_shape(self):
        schema = _host().get_delete_own_message_tool_schema()
        assert schema["type"] == "function"
        assert schema["name"] == "delete_own_message"
        assert list(schema["parameters"]["properties"]) == ["message_ts"]
        assert schema["parameters"]["required"] == ["message_ts"]

    def test_the_description_says_permanent_and_on_request_only(self):
        text = _host().get_delete_own_message_tool_schema()["description"]
        assert "PERMANENT" in text and "no undo" in text
        assert "explicitly asks" in text
        assert "never delete on your own initiative" in text.lower()
        # The alternative the owner endorsed, named where the model will read it.
        assert "edit_own_message" in text


@pytest.mark.unit
class TestDeleteRefusals:
    async def test_a_dm_context_is_refused_before_any_slack_call(self):
        host = _host()
        result = await host.execute_delete_own_message(_ctx(channel=DM, is_dm=True),
                                                       {"message_ts": TS})
        assert result["ok"] is False and result["error"] == "channel_only"
        host.app.client.chat_delete.assert_not_awaited()
        host.app.client.conversations_history.assert_not_awaited()

    async def test_a_d_channel_is_refused_even_without_the_flag(self):
        host = _host()
        result = await host.execute_delete_own_message(_ctx(channel=DM), {"message_ts": TS})
        assert result["error"] == "channel_only"
        host.app.client.chat_delete.assert_not_awaited()

    async def test_someone_elses_message_is_never_deleted(self):
        host = _host(own=False)
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})
        assert result["ok"] is False and result["error"] == "not_own_message"
        host.app.client.chat_delete.assert_not_awaited()

    async def test_an_unreadable_target_deletes_nothing(self):
        host = _host()
        host.app.client.conversations_history = AsyncMock(return_value={"messages": []})
        host.app.client.conversations_replies = AsyncMock(return_value={"messages": []})
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})
        assert result["ok"] is False and result["error"] == "message_not_found"
        host.app.client.chat_delete.assert_not_awaited()

    @pytest.mark.parametrize("value", [None, 12, ["1700000060.000200"], {"ts": TS}, "not-a-ts"])
    async def test_a_malformed_ts_never_reaches_slack(self, value):
        host = _host()
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": value})
        assert result["ok"] is False and result["error"] == "invalid_ts"
        host.app.client.chat_delete.assert_not_awaited()
        host.app.client.conversations_history.assert_not_awaited()

    async def test_no_channel_context(self):
        host = _host()
        result = await host.execute_delete_own_message(_ctx(channel=None), {"message_ts": TS})
        assert result["error"] == "no_channel_context"

    async def test_an_unresolved_bot_identity_authorizes_nothing(self):
        # is_own_message fails closed: nothing proves the author, so nothing is deleted.
        host = _host()
        host.bot_user_id = host.bot_id = host.app_id = None
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})
        assert result["error"] == "not_own_message"
        host.app.client.chat_delete.assert_not_awaited()


@pytest.mark.unit
class TestDeleteOutcomes:
    async def test_deletes_the_message_then_drops_its_receipt(self, receipts):
        host = _host()
        _watch_delete(host, receipts["order"])

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result == {"ok": True, "deleted": True, "message_ts": TS,
                          "message": "That message is permanently deleted."}
        assert host.app.client.chat_delete.await_args.kwargs == {"channel": CH, "ts": TS}
        # BINDING ORDER: the Slack delete is confirmed first, and only then does the row go.
        assert receipts["order"] == ["chat_delete", "receipt"]
        assert receipts["calls"] == [{"team_id": TEAM, "channel_id": CH, "message_ts": TS,
                                      "site": "delete_own_message"}]

    async def test_a_thread_reply_is_read_from_the_thread(self, receipts):
        # conversations.history does not contain thread replies at all, so a reply would be
        # unprovable — and therefore undeletable — without the replies read.
        host = _host(in_thread=True)

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result["ok"] is True
        host.app.client.conversations_replies.assert_awaited_once()
        assert host.app.client.conversations_replies.await_args.kwargs["ts"] == ROOT_TS

    async def test_a_top_level_message_of_another_thread_falls_back_to_the_timeline(self,
                                                                                    receipts):
        host = _host()  # the thread read comes back empty; the channel timeline holds it
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})
        assert result["ok"] is True
        host.app.client.conversations_replies.assert_awaited_once()
        host.app.client.conversations_history.assert_awaited_once()

    async def test_an_already_gone_message_is_the_end_state_asked_for(self, receipts):
        host = _host()
        _watch_delete(host, receipts["order"], error=_api_error("message_not_found"))

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result["ok"] is True and result["deleted"] is False
        # The row still described a post that is not there, so it goes too.
        assert receipts["order"] == ["chat_delete", "receipt"]

    async def test_a_named_slack_error_is_reported_and_keeps_the_receipt(self, receipts):
        host = _host()
        _watch_delete(host, receipts["order"], error=_api_error("cant_delete_message"))

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result["ok"] is False and result["error"] == "cant_delete_message"
        assert receipts["calls"] == []

    async def test_an_unanswered_delete_is_outcome_unknown_not_a_retry(self, receipts):
        # A retry could delete a message somebody posted in the meantime; an unknown outcome is
        # the honest answer and the receipt stays, because it may still be accurate.
        host = _host()
        _watch_delete(host, receipts["order"], error=aiohttp.ClientError("no answer"))

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result["ok"] is False and result["error"] == "outcome_unknown"
        assert receipts["calls"] == []
        assert host.app.client.chat_delete.await_count == 1

    async def test_a_timeout_is_outcome_unknown_too(self, receipts):
        host = _host()
        _watch_delete(host, receipts["order"], error=asyncio.TimeoutError())
        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})
        assert result["error"] == "outcome_unknown"

    async def test_a_failing_receipt_deletion_never_unlands_the_delete(self, monkeypatch):
        async def _boom(**kwargs):
            raise RuntimeError("db down")
        monkeypatch.setattr("message_processor.outbound_receipts.delete_receipt_for", _boom)
        host = _host()

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result["ok"] is True and result["deleted"] is True

    async def test_an_unexpected_failure_is_a_dict_not_a_raise(self):
        host = _host()
        host.app.client.conversations_history = AsyncMock(side_effect=None)
        host._read_deletable_message = AsyncMock(side_effect=RuntimeError("boom"))

        result = await host.execute_delete_own_message(_ctx(), {"message_ts": TS})

        assert result == {"ok": False, "error": "delete_failed",
                          "message": "Could not delete that message."}


@pytest.mark.unit
class TestDeleteRegistrationStance:
    def test_channel_only_registration_mirrors_edit_own_message(self):
        """The registration the tool must be given: hidden on the DM surface by an `enabled`
        gate that is always False, exposed on the channel surface with no channel gate."""
        from message_processor.tool_registry import SURFACE_CHANNEL, ToolRegistry

        host = _host()
        registry = ToolRegistry()
        registry.register(host.get_delete_own_message_tool_schema(),
                          host.execute_delete_own_message,
                          enabled=lambda _cfg: False)

        assert {s["name"] for s in registry.schemas({})} == set()
        assert {s["name"] for s in registry.schemas({}, surface=SURFACE_CHANNEL)} == {
            "delete_own_message"}


# =================================================================== T5b: remove_reaction

@pytest.mark.unit
class TestRemoveReactionSchema:
    def test_schema_shape(self):
        schema = _host().get_remove_reaction_tool_schema()
        assert schema["name"] == "remove_reaction"
        assert set(schema["parameters"]["properties"]) == {"emoji", "ts"}
        assert schema["parameters"]["required"] == ["emoji"]
        # No enum: an allowlist governs what may be PLACED, and an emoji already on a message
        # must stay removable after that list changes underneath it.
        assert "enum" not in schema["parameters"]["properties"]["emoji"]

    def test_the_description_says_it_only_reaches_our_own(self):
        text = _host().get_remove_reaction_tool_schema()["description"]
        assert "never remove anyone else's" in text


@pytest.mark.unit
class TestRemoveReaction:
    async def test_removes_the_emoji_from_the_answered_message_by_default(self):
        host = _host()
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert result["ok"] is True and result["removed"] is True
        assert host.app.client.reactions_remove.await_args.kwargs == {
            "channel": CH, "name": "eyes", "timestamp": TS}

    async def test_colons_are_tolerated_and_another_ts_can_be_named(self):
        host = _host()
        result = await host.execute_remove_reaction_tool(
            _ctx(), {"emoji": ":tada:", "ts": REPLY_TS})
        assert result["ok"] is True
        assert host.app.client.reactions_remove.await_args.kwargs["name"] == "tada"
        assert host.app.client.reactions_remove.await_args.kwargs["timestamp"] == REPLY_TS

    async def test_a_reaction_that_was_not_there_is_not_a_failure(self):
        host = _host()
        host.app.client.reactions_remove = AsyncMock(side_effect=_api_error("no_reaction"))

        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert result["ok"] is True and result["removed"] is False

    async def test_a_named_slack_error_is_reported(self):
        host = _host()
        host.app.client.reactions_remove = AsyncMock(side_effect=_api_error("invalid_name"))
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})
        assert result["ok"] is False and result["error"] == "invalid_name"

    async def test_an_unexpected_failure_is_a_dict_not_a_raise(self):
        host = _host()
        host.app.client.reactions_remove = AsyncMock(side_effect=RuntimeError("boom"))
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})
        assert result["ok"] is False and result["error"] == "reaction_remove_failed"

    @pytest.mark.parametrize("emoji", ["", "  ", ":", "not an emoji", None, 7])
    async def test_a_malformed_emoji_never_reaches_slack(self, emoji):
        host = _host()
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": emoji})
        assert result["ok"] is False and result["error"] == "invalid_emoji"
        host.app.client.reactions_remove.assert_not_awaited()

    async def test_no_target_is_refused(self):
        host = _host()
        result = await host.execute_remove_reaction_tool(
            _ctx(thread_ts=None, trigger_ts=None), {"emoji": "eyes"})
        assert result["ok"] is False and result["error"] == "no_target"
        host.app.client.reactions_remove.assert_not_awaited()

    async def test_reactions_switched_off_refuses_before_slack(self, monkeypatch):
        from config import config as cfg
        monkeypatch.setattr(cfg, "enable_reactions", False)
        host = _host()
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})
        assert result["ok"] is False and result["error"] == "disabled"
        host.app.client.reactions_remove.assert_not_awaited()


@pytest.mark.unit
class TestRemoveReactionGuard:
    """The correction that makes this tool honest rather than merely functional."""

    async def test_a_later_re_add_of_the_same_emoji_really_calls_slack(self):
        """Without clearing the slot, the guard would answer the re-add with idempotent success
        for a reaction this tool had just taken off — the room would see nothing."""
        host = _host()
        placed = await host._reserve_and_react(CH, TS, "eyes")
        assert placed["ok"] is True and host.app.client.reactions_add.await_count == 1

        await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})
        again = await host._reserve_and_react(CH, TS, "eyes")

        assert again["ok"] is True
        assert again.get("idempotent") is not True
        assert host.app.client.reactions_add.await_count == 2

    async def test_the_slot_and_its_key_are_dropped_once_nothing_is_left(self):
        host = _host()
        await host._reserve_and_react(CH, TS, "eyes")
        assert (CH, TS) in host._reaction_guard

        await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert (CH, TS) not in host._reaction_guard
        assert (CH, TS) not in host._reaction_guard_ts

    async def test_a_sibling_emoji_on_the_same_message_is_left_alone(self):
        host = _host()
        await host._reserve_and_react(CH, TS, "eyes")
        await host._reserve_and_react(CH, TS, "tada")

        await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert set(host._reaction_guard[(CH, TS)]) == {"tada"}

    async def test_a_no_reaction_answer_still_corrects_the_bookkeeping(self):
        # Slack says it is not there and the guard says it is: the guard is the one that is
        # wrong, and a stale committed slot is exactly what refuses a later honest re-add.
        host = _host()
        await host._reserve_and_react(CH, TS, "eyes")
        host.app.client.reactions_remove = AsyncMock(side_effect=_api_error("no_reaction"))

        await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert (CH, TS) not in host._reaction_guard

    async def test_a_failed_removal_leaves_the_guard_intact(self):
        host = _host()
        await host._reserve_and_react(CH, TS, "eyes")
        host.app.client.reactions_remove = AsyncMock(side_effect=_api_error("invalid_name"))

        await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        # The emoji may well still be up there; forgetting it would let the cap be exceeded.
        assert "eyes" in host._reaction_guard[(CH, TS)]

    async def test_an_add_in_flight_is_refused_rather_than_raced(self):
        host = _host()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        host._reaction_guard[(CH, TS)] = {"eyes": fut}

        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})

        assert result["ok"] is False and result["error"] == "reaction_busy"
        host.app.client.reactions_remove.assert_not_awaited()
        assert host._reaction_guard[(CH, TS)]["eyes"] is fut
        fut.set_result(True)

    async def test_no_guard_entry_at_all_is_simply_removed(self):
        host = _host()
        result = await host.execute_remove_reaction_tool(_ctx(), {"emoji": "eyes"})
        assert result["ok"] is True
        host.app.client.reactions_remove.assert_awaited_once()
