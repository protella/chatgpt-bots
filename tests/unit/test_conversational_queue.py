"""Phase Q — conversational queueing (busy rejection retired).

Messages arriving while a conversation is mid-processing queue on the state manager
and are answered by the finishing turn's drain hook as ONE batched catch-up turn.
Covers: queue primitives, the process_message contention path, the drain/dispatch
hook, gate-order eligibility, needs_refresh interplay, and busy retirement gates.
"""
import pathlib
from unittest.mock import Mock, AsyncMock, patch

import pytest

from base_client import Message
from config import config
from thread_manager import AsyncThreadStateManager
from message_processor.base import MessageProcessor


REPO = pathlib.Path(__file__).resolve().parents[2]


def _msg(text, user="U1", channel="C123", thread="111.0", ts=None, username=None):
    return Message(
        text=text, user_id=user, channel_id=channel, thread_id=thread,
        attachments=[], metadata={"ts": ts or thread, "username": username or user},
    )


@pytest.fixture
def manager():
    return AsyncThreadStateManager(db=None)


# --- Queue primitives (real AsyncThreadStateManager) ---

class TestQueuePrimitives:
    def test_enqueue_and_count(self, manager):
        assert manager.pending_count("C123:111.0") == 0
        assert manager.enqueue_pending("C123:111.0", _msg("a")) is True
        assert manager.enqueue_pending("C123:111.0", _msg("b")) is True
        assert manager.pending_count("C123:111.0") == 2

    def test_pop_batch_fifo_ordering(self, manager):
        key = "C123:111.0"
        for i in range(5):
            manager.enqueue_pending(key, _msg(f"m{i}"))
        batch = manager.pop_pending_batch(key, 10)
        assert [m.text for m in batch] == ["m0", "m1", "m2", "m3", "m4"]
        assert manager.pending_count(key) == 0

    def test_pop_batch_respects_max_batch_and_leaves_remainder(self, manager):
        key = "C123:111.0"
        for i in range(7):
            manager.enqueue_pending(key, _msg(f"m{i}"))
        batch = manager.pop_pending_batch(key, 3)
        assert [m.text for m in batch] == ["m0", "m1", "m2"]
        assert manager.pending_count(key) == 4  # remainder drains next turn

    def test_enqueue_does_not_set_needs_refresh(self, manager):
        """Queued messages aren't lost — no refetch storm from normal queueing."""
        key = "C123:111.0"
        manager.enqueue_pending(key, _msg("a"))
        assert manager.consume_needs_refresh(key) is False

    def test_max_pending_drops_and_flags_refresh(self, manager):
        key = "C123:111.0"
        with patch.object(config, "queue_max_pending", 3):
            for i in range(3):
                assert manager.enqueue_pending(key, _msg(f"m{i}")) is True
            assert manager.enqueue_pending(key, _msg("overflow")) is False
        assert manager.pending_count(key) == 3
        # Dropped from warm state → transcript refetch flagged (Slack still has it)
        assert manager.consume_needs_refresh(key) is True

    def test_dm_and_channel_parity(self, manager):
        """The queue is keyed on channel:thread — DMs, threads, channels identical."""
        for key in ("D08XYZ:222.0", "C123:111.0"):
            manager.enqueue_pending(key, _msg("hello", channel=key.split(":")[0]))
            assert manager.pending_count(key) == 1
            assert len(manager.pop_pending_batch(key, 10)) == 1

    def test_is_thread_processing_peek(self, manager):
        assert manager.is_thread_processing("111.0", "C123") is False


# --- Contention path: process_message enqueues + returns silent 'queued' ---

class _StubProcessor:
    """Binds the REAL process_message onto a minimal harness."""
    process_message = MessageProcessor.process_message
    _dispatch_pending_batch = MessageProcessor._dispatch_pending_batch

    def __init__(self, manager):
        self.thread_manager = manager
        self.db = None

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


class TestContentionPath:
    @pytest.mark.asyncio
    async def test_enqueue_while_locked_returns_queued_silently(self, manager):
        proc = _StubProcessor(manager)
        msg = _msg("second message")
        # Hold the lock as if a turn were in flight
        assert await manager.acquire_thread_lock("111.0", "C123") is True
        try:
            response = await proc.process_message(msg, client=Mock(), thinking_id=None)
        finally:
            await manager.release_thread_lock("111.0", "C123")

        assert response.type == "queued"
        assert response.content == ""  # nothing for main.py to post
        assert manager.pending_count("C123:111.0") == 1
        assert manager.pop_pending_batch("C123:111.0", 10)[0].text == "second message"
        # Normal queueing must NOT flag a refetch
        assert manager.consume_needs_refresh("C123:111.0") is False


# --- Drain/dispatch hook ---

def _drain_proc(manager):
    proc = _StubProcessor(manager)
    proc._format_user_content_with_username = lambda content, m: f"{m.metadata.get('username')}: {content}"
    proc._add_message_with_token_management = Mock()
    proc._schedule_async_call = Mock()
    return proc


class TestDrainDispatch:
    @pytest.mark.asyncio
    async def test_empty_queue_is_noop_without_linger(self, manager):
        proc = _drain_proc(manager)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()) as slept:
            await proc._dispatch_pending_batch(_msg("done"), Mock(), "C123:111.0")
        slept.assert_not_awaited()
        proc._schedule_async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_batch_three_senders_one_dispatch(self, manager):
        """3 queued messages from 3 senders → earlier two appended attributed,
        LAST becomes the trigger, exactly ONE re-dispatch."""
        key = "C123:111.0"
        state = Mock()
        manager.get_thread_async = AsyncMock(return_value=state)
        for user, text in (("alice", "what's the ETA?"), ("bob", "and the budget?"), ("carol", "thoughts?")):
            manager.enqueue_pending(key, _msg(text, user=user, username=user, ts=f"{user}.ts"))

        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()  # coroutine fn stand-in; scheduled, not awaited

        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()) as slept:
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        slept.assert_awaited_once_with(config.queue_drain_linger_seconds)
        # Earlier two appended individually with attribution + their ts
        appended = [c.args for c in proc._add_message_with_token_management.call_args_list]
        assert [a[2] for a in appended] == ["alice: what's the ETA?", "bob: and the budget?"]
        # One dispatch; trigger is the LAST message, marked with the batch size
        proc._schedule_async_call.assert_called_once()
        trigger = client.message_handler.call_args.args[0]
        assert trigger.text == "thoughts?"
        assert trigger.metadata["queued_batch_size"] == 3
        assert manager.pending_count(key) == 0

    @pytest.mark.asyncio
    async def test_linger_configurable_and_zero_skips_sleep(self, manager):
        key = "C123:111.0"
        manager.enqueue_pending(key, _msg("a"))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch.object(config, "queue_drain_linger_seconds", 0.0), \
             patch("message_processor.base.asyncio.sleep", new=AsyncMock()) as slept:
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        slept.assert_not_awaited()
        proc._schedule_async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_sustained_burst_drains_over_successive_turns(self, manager):
        """Loop-until-empty is emergent: remainder beyond QUEUE_MAX_BATCH drains
        when the NEXT turn's finally-hook fires."""
        key = "C123:111.0"
        manager.get_thread_async = AsyncMock(return_value=Mock())
        for i in range(7):
            manager.enqueue_pending(key, _msg(f"m{i}", ts=f"{i}.0"))
        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()

        with patch.object(config, "queue_max_batch", 5), \
             patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("turn1"), client, key)   # batch of 5
            assert manager.pending_count(key) == 2
            await proc._dispatch_pending_batch(_msg("turn2"), client, key)   # batch of 2
            assert manager.pending_count(key) == 0

        assert proc._schedule_async_call.call_count == 2
        first, second = [c.args[0] for c in client.message_handler.call_args_list]
        assert first.metadata["queued_batch_size"] == 5
        assert second.metadata["queued_batch_size"] == 2
        assert first.text == "m4" and second.text == "m6"  # FIFO preserved across turns

    @pytest.mark.asyncio
    async def test_no_handler_flags_refresh_instead_of_losing_messages(self, manager):
        key = "C123:111.0"
        manager.enqueue_pending(key, _msg("a"))
        proc = _drain_proc(manager)
        client = Mock(spec=[])  # no message_handler
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_not_called()
        assert manager.consume_needs_refresh(key) is True

    @pytest.mark.asyncio
    async def test_single_queued_message_dispatches_without_state_appends(self, manager):
        key = "C123:111.0"
        manager.enqueue_pending(key, _msg("solo", ts="9.9"))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._add_message_with_token_management.assert_not_called()  # its own turn appends it
        trigger = client.message_handler.call_args.args[0]
        assert trigger.text == "solo" and trigger.metadata["queued_batch_size"] == 1


def _edit_msg(text, ts, *, participation_check=False, edit_marker=None, channel="C123",
              thread="111.0"):
    md = {"ts": ts, "username": "u"}
    if participation_check:
        md["participation_check"] = True
    if edit_marker is not None:
        md["edit_reply_marker"] = edit_marker
    return Message(text=text, user_id="U1", channel_id=channel, thread_id=thread,
                   attachments=[], metadata=md)


class TestEditStaleDrop:
    """F52: a stale PRE-EDIT participation dispatch that slipped into the busy queue is dropped
    at drain (it would otherwise re-run the gate on stale text and post a duplicate), while the
    edit's own dispatch and genuinely different messages survive."""

    def _client(self, registry):
        client = Mock()
        client.message_handler = Mock()
        client.edit_dispatch_marker = lambda ch, ts: registry.get(f"{ch}|{ts}")
        return client

    @pytest.mark.asyncio
    async def test_stale_pre_edit_dispatch_dropped_survivor_kept(self, manager):
        key = "C123:111.0"
        # ts 200 was edited and handled; the edit's own dispatch carries marker "M".
        registry = {"C123|200.0": "M"}
        # Stale pre-edit engine respond (participation_check, no marker) for the SAME ts.
        manager.enqueue_pending(key, _edit_msg("does anyone remember?", "200.0",
                                               participation_check=True))
        # A genuinely different queued message (different ts) — must survive.
        manager.enqueue_pending(key, _edit_msg("unrelated question", "201.0",
                                               participation_check=True))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = self._client(registry)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_called_once()
        trigger = client.message_handler.call_args.args[0]
        assert trigger.text == "unrelated question"   # the stale one was dropped
        # The different message survived as the sole trigger (batch size 1 after the drop).
        assert trigger.metadata["queued_batch_size"] == 1

    @pytest.mark.asyncio
    async def test_edits_own_marked_dispatch_survives(self, manager):
        key = "C123:111.0"
        registry = {"C123|200.0": "M"}
        # The edit's OWN engine re-dispatch: same ts, carries the matching marker → kept.
        manager.enqueue_pending(key, _edit_msg("review the Q3 numbers", "200.0",
                                               participation_check=True, edit_marker="M"))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = self._client(registry)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_called_once()
        assert client.message_handler.call_args.args[0].text == "review the Q3 numbers"

    @pytest.mark.asyncio
    async def test_addressed_turn_never_dropped(self, manager):
        """An addressed (app_mention/DM) queued turn carries no participation_check — even for a
        registered edit ts it is never dropped."""
        key = "C123:111.0"
        registry = {"C123|200.0": "M"}
        manager.enqueue_pending(key, _edit_msg("<@UBOT> what's up", "200.0",
                                               participation_check=False))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = self._client(registry)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_called_once()
        assert client.message_handler.call_args.args[0].text == "<@UBOT> what's up"


# --- Gate order: participation-ignored messages never reach the queue ---

class TestGateOrder:
    @pytest.mark.asyncio
    async def test_gate_ignored_channel_message_never_processes_or_queues(self):
        from main import ChatBotV2
        bot = ChatBotV2(platform="slack")
        bot.processor = Mock()
        bot.processor.process_message = AsyncMock()
        bot._run_participation_gate = AsyncMock(return_value=None)  # engine said ignore

        message = Mock(channel_id="C123", thread_id="111.0")
        message.metadata = {"participation_check": True, "ts": "111.0"}
        await bot.handle_message(message, Mock())

        bot.processor.process_message.assert_not_called()  # → nothing could enqueue


# --- Busy retirement: source gates ---

class TestBusyRetirement:
    def _runtime_sources(self):
        for rel in ("main.py", "message_processor/base.py", "slack_client/messaging.py",
                    "base_client.py", "thread_manager.py"):
            yield rel, (REPO / rel).read_text()

    def test_no_busy_response_constructed_or_handled(self):
        for rel, src in self._runtime_sources():
            assert 'type="busy"' not in src and "type='busy'" not in src, rel
            assert "send_busy_message" not in src, rel

    def test_queued_type_exists_and_is_handled(self):
        assert 'type="queued"' in (REPO / "message_processor/base.py").read_text()
        assert '"queued"' in (REPO / "main.py").read_text()


# --- F10: earlier batch messages' attachments are processed, not dropped ---

def _attach_msg(text, *, attachments, user="alice", ts="a.ts", channel="C123", thread="111.0"):
    return Message(text=text, user_id=user, channel_id=channel, thread_id=thread,
                   attachments=attachments, metadata={"ts": ts, "username": user})


class TestBatchAttachments:
    """F10: an earlier queued message (not the trigger) carrying attachments used to be
    appended as TEXT ONLY — its documents got no save_document row (unreachable by
    read_document/mount_file) and its images rode only ambient dual-write. The drain now runs
    the SAME attachment pipeline the trigger turn runs for every batched message."""

    @pytest.mark.asyncio
    async def test_earlier_message_documents_processed_and_folded(self, manager):
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("see attached",
                              attachments=[{"type": "file", "name": "report.pdf"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("and the summary?", user="bob", username="bob", ts="b.ts"))

        proc = _drain_proc(manager)
        doc = {"filename": "report.pdf", "summary": "Q3 numbers"}
        proc._process_attachments = AsyncMock(return_value=([], [doc], []))
        proc._build_message_with_documents = Mock(
            side_effect=lambda text, docs: f"{text} [+doc:{docs[0]['filename']}]")

        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": True})):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        # The earlier message's attachments went through the SAME pipeline the trigger uses,
        # keyed on THAT message and the resolved per-thread CI setting.
        proc._process_attachments.assert_awaited_once()
        assert proc._process_attachments.await_args.args[0] is earlier
        assert proc._process_attachments.await_args.kwargs["code_interpreter_enabled"] is True
        # Its document summary was folded into that message's appended content.
        appended = [c.args[2] for c in proc._add_message_with_token_management.call_args_list]
        assert appended[0] == "alice: see attached [+doc:report.pdf]"

    @pytest.mark.asyncio
    async def test_earlier_message_images_catalogued(self, manager):
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("look at this",
                              attachments=[{"type": "image", "name": "shot.png",
                                            "url": "http://x/shot.png"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("thoughts?", user="bob", username="bob", ts="b.ts"))

        proc = _drain_proc(manager)
        img_inputs = [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]
        proc._process_attachments = AsyncMock(return_value=(img_inputs, [], []))
        proc._build_message_with_documents = Mock()

        client = Mock()
        client.message_handler = Mock()
        # Sync Mock (not the auto-detected AsyncMock) so no un-awaited coroutine is created:
        # _schedule_async_call is itself a Mock here and would never await a real coroutine.
        catalog = Mock(return_value=None)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": False})), \
             patch("message_processor.base.image_catalog.catalog_uploads", new=catalog):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        # A durable visual description was scheduled for the earlier image (trigger parity).
        catalog.assert_called_once()
        cat_args = catalog.call_args.args
        assert cat_args[2] == earlier.attachments and cat_args[3] == img_inputs
        # No documents → the document folder is never invoked.
        proc._build_message_with_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_earlier_images_and_failures_carried_to_trigger(self, manager):
        """T2-10: earlier messages' image parts AND attachment failures are stashed on the
        trigger's metadata so its turn can show the images and acknowledge the failures."""
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("pic plus a broken file",
                              attachments=[{"type": "image", "name": "a.png", "url": "u"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("go", user="bob", username="bob", ts="b.ts"))

        proc = _drain_proc(manager)
        img = {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        fail = {"name": "broken.pdf", "error": "download_failed"}
        proc._process_attachments = AsyncMock(return_value=([img], [], [fail]))
        proc._build_message_with_documents = Mock()
        catalog = Mock(return_value=None)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": False})), \
             patch("message_processor.base.image_catalog.catalog_uploads", new=catalog):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata["batched_image_inputs"] == [img]
        assert trigger.metadata["batched_unsupported_files"] == [fail]

    @pytest.mark.asyncio
    async def test_no_batched_keys_when_earlier_messages_have_no_attachments(self, manager):
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        manager.enqueue_pending(key, _msg("first", user="a", username="a", ts="a.ts"))
        manager.enqueue_pending(key, _msg("second", user="b", username="b", ts="b.ts"))
        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        trigger = client.message_handler.call_args.args[0]
        assert "batched_image_inputs" not in trigger.metadata
        assert "batched_unsupported_files" not in trigger.metadata

    @pytest.mark.asyncio
    async def test_no_attachments_skips_thread_config_resolution(self, manager):
        """The common attachment-free batch must NOT pay for a thread-config resolution."""
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        manager.enqueue_pending(key, _msg("first", user="a", username="a", ts="a.ts"))
        manager.enqueue_pending(key, _msg("second", user="b", username="b", ts="b.ts"))

        proc = _drain_proc(manager)
        proc._process_attachments = AsyncMock()
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async", new=AsyncMock()) as gtc:
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        gtc.assert_not_awaited()
        proc._process_attachments.assert_not_awaited()
