"""Phase 5/6 + 2.5 — channel-listening decision logic, reply placement, and bot-in-roster.

These exercise the real decision code in SlackMessageEventsMixin with stubbed I/O, so they
assert the SAFE-by-default behavior the keystone is supposed to ship.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.utilities import build_roster_text
from slack_client.event_handlers.message_events import (
    SlackMessageEventsMixin, _attachment_descriptors)
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    """Minimal harness exposing the real channel-decision logic with stubbed logging/I/O."""

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()

    async def _fake_event_to_message(event, client):
        # Mirror the real _event_to_message file plumbing (both the @-mention and
        # channel-listening paths share it), so gate tests can assert the dispatched
        # Message carries files from a file_share event.
        attachments = []
        for f in event.get("files", []) or []:
            mimetype = f.get("mimetype", "")
            attachments.append({
                "type": "image" if mimetype.startswith("image/") else "file",
                "url": f.get("url_private"),
                "id": f.get("id"),
                "name": f.get("name"),
                "mimetype": mimetype,
            })
        return Message(
            text=event.get("text", ""),
            user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=attachments,
            metadata={"ts": event.get("ts")},
        )

    bot._event_to_message = _fake_event_to_message
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "hello there", "channel_type": "channel"}
    e.update(kw)
    return e


@pytest.fixture
def tag_only(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "tag_only", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT", "ChatGPT-Dev"], raising=False)


@pytest.mark.asyncio
async def test_own_message_by_user_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(user="UBOT", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_own_message_by_bot_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(bot_id="BBOT", user=None, text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_subtype_skipped(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(subtype="channel_join", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_message_changed_subtype_excluded_from_gate(tag_only):
    # F14: non-content subtypes (edits/deletes) still never drive a response.
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="message_changed", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_file_share_reaches_gate_and_plumbs_files(tag_only):
    # F14: an image/file upload arrives as subtype 'file_share' — it must proceed
    # through the response gate (was dropped before) AND carry its files onto the
    # dispatched Message so intent classification can route the vision/document flow.
    bot = _make_bot()
    file_meta = {
        "id": "F123", "name": "poster.png", "mimetype": "image/png",
        "url_private": "https://files.slack.com/poster.png",
    }
    await bot._handle_channel_message(
        _evt(subtype="file_share", text="ChatGPT good marketing material?",
             files=[file_meta]),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_name_hit") is True
    assert msg.attachments and msg.attachments[0]["type"] == "image"
    assert msg.attachments[0]["id"] == "F123"


def test_bot_message_has_content_distinguishes_webhook_from_chrome():
    # F48: a webhook bot post carries its payload in attachment fields with EMPTY text — that is
    # real content. Bare chrome (empty text, no files, no supplementary) is not.
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin as M
    webhook = {"subtype": "bot_message", "text": "",
               "attachments": [{"fields": [{"title": "Branch", "value": "main"}],
                                "fallback": "Build #123 failed"}]}
    assert M._bot_message_has_content(webhook) is True
    assert M._bot_message_has_content({"subtype": "bot_message", "text": ""}) is False
    assert M._bot_message_has_content({"subtype": "bot_message", "text": "hi"}) is True


@pytest.mark.asyncio
async def test_supplementary_bot_message_reaches_gate(monkeypatch):
    # F48 acceptance case (MUST-FIX 10): a Jira/GitHub webhook bot_message with empty text but
    # meaningful attachment fields must NOT be dropped at the subtype gate — it reaches the
    # participation engine like any other content message.
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="bot_message", user=None, bot_id="OTHERBOT", text="",
             attachments=[{"fields": [{"title": "Branch", "value": "main"}],
                           "fallback": "Build #123 failed"}]),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("gate_required") is True
    assert msg.metadata.get("participation_sender_bot") is True


@pytest.mark.asyncio
async def test_bare_chrome_bot_message_still_dropped(monkeypatch):
    # The other half: a content-free bot post (no text, no files, no supplementary) still drops
    # at the subtype gate — the widening is for real content only.
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="bot_message", user=None, bot_id="OTHERBOT", text=""), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_edit_and_delete_mark_thread_needs_refresh():
    # Blocker 1: a message edit/delete must flag the affected thread's warm ThreadState for
    # rebuild, or it can keep answering from the deleted/pre-edit content.
    from types import SimpleNamespace

    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    refreshed = []
    tm = SimpleNamespace(mark_needs_refresh=lambda k: refreshed.append(k))

    class _DB:
        async def delete_ambient_artifacts_by_source(self, c, t):
            return 0

    class _Svc:
        def offer_event(self, ev, cl):
            pass

    class _Host(SlackMessageEventsMixin):
        def __init__(self):
            self.processor = SimpleNamespace(ambient_service=_Svc(), thread_manager=tm)
            self.db = _DB()

        def is_own_message(self, e):
            return False

        def log_debug(self, *a, **k):
            pass

    host = _Host()
    # delete: refresh keyed on the deleted message's thread root
    await host._ambient_ingest(
        {"subtype": "message_deleted", "channel": "C1", "deleted_ts": "5.0",
         "previous_message": {"ts": "5.0", "thread_ts": "1.0"}}, object())
    assert "C1:1.0" in refreshed
    # edit: refresh keyed on the edited message's thread root
    await host._ambient_ingest(
        {"subtype": "message_changed", "channel": "C1",
         "message": {"ts": "6.0", "thread_ts": "2.0", "text": "edited text"}}, object())
    assert "C1:2.0" in refreshed


@pytest.mark.asyncio
async def test_tombstoned_root_takes_deletion_path_not_edit_path():
    # Deleting a root that has (or had) replies does NOT arrive as message_deleted —
    # Slack tombstones it via message_changed (nested subtype "tombstone", text
    # "This message was deleted."). Treated as an edit, the tombstone text reached ambient
    # memory as content and the edit-triggered engine ran on it (live 2026-07-18: the model
    # then "remembered" deleted threads as still visible). It must take the deletion path:
    # purge + refresh, no offer, no edit engine.
    from types import SimpleNamespace

    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    refreshed, purged, offered, edit_dispatches = [], [], [], []
    tm = SimpleNamespace(mark_needs_refresh=lambda k: refreshed.append(k))

    class _DB:
        async def delete_ambient_artifacts_by_source(self, c, t):
            purged.append((c, t))
            return 0

    class _Svc:
        def offer_event(self, ev, cl):
            offered.append(ev)

    class _Host(SlackMessageEventsMixin):
        def __init__(self):
            self.processor = SimpleNamespace(ambient_service=_Svc(), thread_manager=tm)
            self.db = _DB()

        def is_own_message(self, e):
            return False

        def log_debug(self, *a, **k):
            pass

        def _maybe_edit_triggered_reply(self, event, client):
            edit_dispatches.append(event)

    host = _Host()
    # canonical shape: nested subtype "tombstone"
    await host._ambient_ingest(
        {"subtype": "message_changed", "channel": "C1",
         "message": {"ts": "7.0", "subtype": "tombstone",
                     "text": "This message was deleted."}}, object())
    # text-only fallback shape (no nested subtype)
    await host._ambient_ingest(
        {"subtype": "message_changed", "channel": "C1",
         "message": {"ts": "8.0", "thread_ts": "8.0",
                     "text": "This message was deleted."}}, object())
    assert purged == [("C1", "7.0"), ("C1", "8.0")]         # derived artifacts purged
    assert "C1:7.0" in refreshed and "C1:8.0" in refreshed  # warm threads rebuilt
    assert offered == []                                    # never offered to ambient memory
    assert edit_dispatches == []                            # edit engine never runs on it


def test_attachment_descriptors_name_and_kind_per_file():
    """Re-baselined for the binary gate: this used to assert one prose sentence
    ("1 image, 1 file (chart.png, notes.pdf)") that this helper assembled and the gate prompt
    pasted in whole — i.e. an event handler was writing part of a prompt. It now returns one
    "name (kind)" descriptor per file and the renderer does the rendering.

    Names and types only, and no content: the gate it feeds never opens anything. Empty tuple
    rather than None for "no files", so callers can iterate unconditionally."""
    assert _attachment_descriptors(None) == ()
    assert _attachment_descriptors([]) == ()
    assert _attachment_descriptors(
        [{"name": "food.png", "mimetype": "image/png"}]) == ("food.png (image)",)
    assert _attachment_descriptors([
        {"name": "report.pdf", "mimetype": "application/pdf"},
        {"name": "data.csv", "mimetype": "text/csv"},
    ]) == ("report.pdf (file)", "data.csv (file)")
    # Mixed kinds keep SLACK'S order, not a kind-sorted one: the descriptors are per-message
    # facts, and reordering them would misstate what was uploaded when.
    assert _attachment_descriptors([
        {"name": "chart.png", "mimetype": "image/png"},
        {"name": "notes.pdf", "mimetype": "application/pdf"},
    ]) == ("chart.png (image)", "notes.pdf (file)")
    # A file with no name at all still gets a descriptor — the gate learning "something was
    # attached" is the point, and a dropped entry would understate the message.
    assert _attachment_descriptors([{"mimetype": "image/png"}]) == ("file (image)",)


@pytest.mark.asyncio
async def test_file_share_sets_participation_attachments_signal(tag_only):
    # F14b end-to-end: a file_share that reaches the gate carries its attachment DESCRIPTORS in
    # metadata (names and types, never pixels) so the gate is not blind to the uploaded artifact.
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="file_share", text="ChatGPT good marketing material?",
             files=[{"id": "F1", "name": "poster.png", "mimetype": "image/png",
                     "url_private": "https://files.slack.com/poster.png"}]),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("gate_required") is True
    assert msg.metadata.get("participation_attachments") == ("poster.png (image)",)


@pytest.mark.asyncio
async def test_thread_broadcast_subtype_reaches_gate(tag_only):
    # F14: a thread reply also broadcast to channel arrives as 'thread_broadcast' —
    # real content, so it reaches the gate (engine judges the name-hit).
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="thread_broadcast", text="ChatGPT what do you think?",
             thread_ts="50.0", ts="60.0"),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_name_hit") is True


@pytest.mark.asyncio
async def test_real_event_to_message_extracts_files(tag_only):
    # F14: the SHARED _event_to_message (both the @-mention and channel paths call it)
    # extracts event files into attachments — proving the channel path plumbs files
    # identically to the mention path. Image mimetypes classify as 'image', others 'file'.
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot._clean_mentions = lambda t: t
    bot.get_username = AsyncMock(return_value="Human")
    bot.get_user_timezone = AsyncMock(return_value="UTC")
    bot.classify_sender = lambda e: "human"
    event = _evt(
        subtype="file_share", text="what do we think?",
        files=[
            {"id": "F1", "name": "poster.png", "mimetype": "image/png",
             "url_private": "https://files.slack.com/poster.png"},
            {"id": "F2", "name": "brief.pdf", "mimetype": "application/pdf",
             "url_private": "https://files.slack.com/brief.pdf"},
        ])
    msg = await bot._event_to_message(event, bot.app.client if hasattr(bot, "app") else MagicMock())
    assert [a["type"] for a in msg.attachments] == ["image", "file"]
    assert [a["id"] for a in msg.attachments] == ["F1", "F2"]


@pytest.mark.asyncio
async def test_off_mode_never_responds(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "off", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT help me"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_unaddressed_ignored(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_name_hit_routes_to_engine(tag_only):
    # Revised contract: a name-in-text hit is a SIGNAL, not a verdict — the engine
    # decides addressed vs merely-discussed vs same-named public product.
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("channel_listen") is True
    assert msg.metadata.get("gate_required") is True
    assert msg.metadata.get("participation_name_hit") is True


@pytest.mark.asyncio
async def test_tag_only_name_hit_engine_disabled_falls_back_deterministic(tag_only, monkeypatch):
    # With the engine off, the legacy deterministic name wake keeps working.
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("gate_required") is not True


@pytest.mark.asyncio
async def test_explicit_mention_is_deduped(tag_only):
    # An <@UBOT> mention is already delivered via the app_mention event; channel path must skip.
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="<@UBOT> hello"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_auto_respond_gate_routes_unaddressed(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("gate_required") is True
    # auto_respond ≡ "on" now. `judicious` and `active` were two dials on a gate that weighed
    # "is this worth saying"; the binary gate does not ask that, so both migrated to one level.
    assert msg.metadata.get("participation_level") == "on"


@pytest.mark.asyncio
async def test_engine_disabled_makes_auto_respond_mentions_only(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    # unaddressed → ignored with zero model cost
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_not_called()
    # addressed by name → still responds directly (pre-F tag_only behavior preserved)
    await bot._handle_channel_message(_evt(ts="101.1", text="ChatGPT what's up?"), bot.app.client)
    bot.message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_one_to_one_continuation_responds_with_no_mute_lookup(monkeypatch):
    # The per-thread mute mechanism was removed: a "stay out of this thread" is now a no-op, so an
    # untagged HUMAN reply in a genuinely 1:1 thread still continues deterministically. Crucially
    # the routing consults NO mute state — the pre-gate mute lookup is gone.
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))  # bot present, one human, 1:1
    bot.db = MagicMock()
    bot.db.is_thread_muted_async = AsyncMock(return_value=True)  # even a "would-be muted" thread

    async def _cs(channel_id):
        return {"response_mode": "auto_respond"}

    bot._get_channel_settings = _cs
    await bot._handle_channel_message(
        _evt(text="and what about q4?", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("wake_source") == "thread_continuation"
    # no mute lookup happened anywhere on the path
    bot.db.is_thread_muted_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_channel_dispatch_stamps_the_ts_the_listener_admitted(tag_only):
    """H is pinned against what this process has already ADMITTED, and for an ordinary post that is
    the message's own ts. Stamping it at dispatch keeps the one place that knows which ts the
    listener observed — the handler — as the place that says so."""
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT what about q4?"), bot.app.client)
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata["trigger_admission_ts"] == "100.1"


@pytest.mark.asyncio
async def test_a_mention_dispatch_stamps_the_admission_ts(tag_only):
    bot = _make_bot()
    bot.db = MagicMock()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._get_channel_settings = AsyncMock(return_value=None)
    bot._post_settings_button_if_new_thread = AsyncMock()
    await bot._handle_slack_message(
        _evt(text="<@UBOT> what about q4?", ts="300.5"), bot.app.client,
        wake_source="app_mention")
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata["trigger_admission_ts"] == "300.5"


@pytest.mark.asyncio
async def test_a_dm_dispatch_carries_no_admission_stamp():
    """A DM has no channel stream to pin, and DM request shape is frozen — the key must not
    appear on that path at all."""
    bot = _make_bot()
    bot.db = MagicMock()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._get_channel_settings = AsyncMock(return_value=None)
    bot._maybe_set_assistant_thread_title = AsyncMock()
    bot._post_settings_button_if_new_thread = AsyncMock()
    await bot._handle_slack_message(
        _evt(channel="D1", channel_type="im", ts="200.5"), bot.app.client, wake_source="dm")
    msg = bot.message_handler.call_args[0][0]
    assert "trigger_admission_ts" not in msg.metadata


@pytest.mark.asyncio
async def test_participation_level_off_row_silences_channel(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()

    async def _cs(channel_id):
        return {"response_mode": "auto_respond", "participation_level": "off"}

    bot._get_channel_settings = _cs
    await bot._handle_channel_message(_evt(text="ChatGPT help"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_thread_reply_one_on_one_responds(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
    await bot._handle_channel_message(_evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_thread_reply_multiparty_unaddressed_ignored(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 3, 0))
    await bot._handle_channel_message(_evt(text="sounds good to me", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_thread_with_other_agent_is_not_a_continuation(tag_only):
    # A second bot/agent in the thread means untagged replies may be for IT —
    # no deterministic continuation (this is the Claude-in-the-test-channel bug:
    # one human + two agents looked "1:1" when only humans were counted).
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 1))
    await bot._handle_channel_message(_evt(text="sounds good", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_actor_tail_cancels_a_continuation_the_replies_scan_called_one_on_one(tag_only):
    """The replies fast path reads only Slack's oldest page, so a second agent deeper in a long
    thread is invisible to it — `_thread_participation` genuinely reports 1:1 here. The actor tail
    saw that agent arrive live, and it has to be able to cancel a route that answers with no gate
    involved at all."""
    from slack_client import actor_tail

    actor_tail.actor_tail.reset()
    try:
        bot = _make_bot()
        bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
        await bot._handle_channel_message(
            _evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
        bot.message_handler.assert_called_once()          # tail empty → the fast path stands

        actor_tail.record("C1", ts="55.0", root_ts="50.0", is_bot=True,
                          sender_type="other_bot")
        bot.message_handler.reset_mock()
        await bot._handle_channel_message(
            _evt(text="and what about saturday?", thread_ts="50.0", ts="61.0"), bot.app.client)
        bot.message_handler.assert_not_called()           # gate-judged instead, and tag_only → silent
    finally:
        actor_tail.actor_tail.reset()


@pytest.mark.asyncio
async def test_bot_sender_never_direct_continuation(tag_only):
    # Another bot replying in our 1:1 thread must not get a judgment-free response.
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="I agree with the plan.",
               thread_ts="50.0", ts="60.0")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_not_called()
    bot._thread_participation.assert_not_called()  # not even consulted for bot senders


@pytest.mark.asyncio
async def test_bot_sender_name_hit_routes_to_engine_with_signal(tag_only):
    # Bot-to-bot is allowed — but only via the engine's judgment, with the
    # sender-is-bot signal attached.
    bot = _make_bot()
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="ChatGPT, what does the data say?")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("gate_required") is True
    assert msg.metadata.get("participation_sender_bot") is True


@pytest.mark.asyncio
async def test_bot_sender_name_hit_engine_disabled_stays_silent(tag_only, monkeypatch):
    # With the engine off there is no judgment available, so a bot naming us
    # must not trigger the legacy deterministic wake (loop seed).
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="ChatGPT, ping")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_reply_placed_in_thread(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT ping", ts="77.7"), bot.app.client)
    msg = bot.message_handler.call_args[0][0]
    assert msg.thread_id == "77.7"  # top-level wake → reply in a thread rooted at the message


def test_text_mentions_bot_name_whole_word(tag_only):
    """The prefilter's ONE job: does configured name-addressing make this message eligible for
    dispatch. It decides nothing semantic — not relevance, wake, reply, silence, placement or
    settings — which is why "the chatgptithon event" only needs to be a non-match here rather than
    understood. The routing it gates is pinned above: unaddressed traffic in a mentions_only
    (tag_only) channel never dispatches, a name hit reaches the gate."""
    bot = _make_bot()
    assert bot._text_mentions_bot_name("hey ChatGPT can you help")
    assert bot._text_mentions_bot_name("CHATGPT-DEV go")
    assert not bot._text_mentions_bot_name("the chatgptithon event")  # whole-word match only
    assert not bot._text_mentions_bot_name("no name here")


@pytest.mark.asyncio
async def test_thread_participation_counts_humans_and_bot(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(return_value={"messages": [
        {"user": "UBOT", "bot_id": "BBOT"},  # self
        {"user": "UHUMAN1"},
        {"user": "UHUMAN2"},
        {"user": "UHUMAN1"},  # dup human
        {"user": "UCLAUDE", "bot_id": "BCLAUDE"},  # another agent
        {"user": "UCLAUDE", "bot_id": "BCLAUDE"},  # dup agent
    ]})
    bot_present, humans, other_bots = await bot._thread_participation("C1", "50.0")
    assert bot_present is True
    assert humans == 2
    assert other_bots == 1


@pytest.mark.asyncio
async def test_thread_participation_handles_api_error(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(side_effect=RuntimeError("boom"))
    assert await bot._thread_participation("C1", "50.0") == (False, 0, 0)


def test_default_config_is_safe(monkeypatch):
    # OUT OF THE BOX: the bot must not auto-listen, and the default channel mode is
    # tag_only. Build a fresh config with the env keys absent — the module singleton
    # may reflect a real .env (e.g. the dev box enables listening for live testing).
    monkeypatch.delenv("ENABLE_CHANNEL_LISTENING", raising=False)
    monkeypatch.delenv("CHANNEL_RESPONSE_MODE", raising=False)
    from config import BotConfig
    fresh = BotConfig()
    assert fresh.enable_channel_listening is False
    assert fresh.channel_response_mode == "tag_only"


def test_bot_with_real_user_id_lands_in_roster():
    # Phase 2.5: another bot that posts with a real user_id can be tagged via the roster.
    txt = build_roster_text({"U123": "Peter", "U999": "Claude"}, user_cache={}, bot_user_id="UBOT")
    assert "<@U999>" in txt
    # The "bot"/"unknown" placeholder ids are excluded (cannot <@>-tag a bot_id).
    txt2 = build_roster_text({"bot": "Bot", "U123": "Peter"}, bot_user_id="UBOT")
    assert "<@bot>" not in txt2
    assert "<@U123>" in txt2
