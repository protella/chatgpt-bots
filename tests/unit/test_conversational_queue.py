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


def _edit_msg(text, ts, *, gate_required=False, edit_marker=None, channel="C123",
              thread="111.0"):
    md = {"ts": ts, "username": "u"}
    if gate_required:
        md["gate_required"] = True
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
        # Stale pre-edit engine respond (gate-routed, no marker) for the SAME ts.
        manager.enqueue_pending(key, _edit_msg("does anyone remember?", "200.0",
                                               gate_required=True))
        # A genuinely different queued message (different ts) — must survive.
        manager.enqueue_pending(key, _edit_msg("unrelated question", "201.0",
                                               gate_required=True))
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
                                               gate_required=True, edit_marker="M"))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = self._client(registry)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_called_once()
        assert client.message_handler.call_args.args[0].text == "review the Q3 numbers"

    @pytest.mark.asyncio
    async def test_addressed_turn_never_dropped(self, manager):
        """An addressed (app_mention/DM) queued turn is not gate-routed — even for a
        registered edit ts it is never dropped."""
        key = "C123:111.0"
        registry = {"C123|200.0": "M"}
        manager.enqueue_pending(key, _edit_msg("<@UBOT> what's up", "200.0",
                                               gate_required=False))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        client = self._client(registry)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        proc._schedule_async_call.assert_called_once()
        assert client.message_handler.call_args.args[0].text == "<@UBOT> what's up"


# --- Queued-batch linkage: which later turn covered a queued message ---

class TestQueueLinkStaging:
    """A batch member that is not the trigger never gets a turn of its own — its work happens
    inside the successor's. Before this, that message's gate attempt ended in `queued` and
    nothing anywhere said which reply eventually covered it."""

    @pytest.mark.asyncio
    async def test_drain_stages_the_absorbed_attempt_ids_on_the_trigger(self, manager):
        from message_processor import participation_telemetry as pt
        key = "C123:111.0"
        manager.get_thread_async = AsyncMock(return_value=Mock())
        earlier, later = _msg("what's the ETA?", ts="1.0"), _msg("and the budget?", ts="2.0")
        first_id = pt.begin_attempt(earlier)
        second_id = pt.begin_attempt(later)
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, later)
        manager.enqueue_pending(key, _msg("thoughts?", ts="3.0"))   # the trigger, ungated

        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata[pt.BATCHED_SOURCES_KEY] == [first_id, second_id]

    @pytest.mark.asyncio
    async def test_a_requeued_successor_carries_its_inheritance_to_the_next_drain(self, manager):
        """THE RACE this staging exists to survive. An ungated successor absorbs a queued
        attempt, then hits a busy conversation and queues itself. It mints no attempt id, so if
        its inheritance stopped there, nothing downstream could ever say which turn covered
        those messages — the ledger would simply lose them."""
        from message_processor import participation_telemetry as pt
        key = "C123:111.0"
        successor = _msg("<@UBOT> and the budget?", ts="2.0")
        pt.stage_queue_links(successor, ["src-a"])       # handed to it by an earlier drain

        # It never runs: the conversation is busy, so it queues instead of answering.
        assert await manager.acquire_thread_lock("111.0", "C123") is True
        try:
            queued = await _StubProcessor(manager).process_message(
                successor, client=Mock(), thinking_id=None)
        finally:
            await manager.release_thread_lock("111.0", "C123")
        assert queued.type == "queued"
        assert successor.metadata[pt.BATCHED_SOURCES_KEY] == ["src-a"]   # claimed nothing

        # The next drain absorbs it — and the inheritance travels with it, onto the new trigger.
        manager.get_thread_async = AsyncMock(return_value=Mock())
        manager.enqueue_pending(key, _msg("thoughts?", ts="3.0"))
        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.text == "thoughts?"
        assert trigger.metadata[pt.BATCHED_SOURCES_KEY] == ["src-a"]
        assert pt.BATCHED_SOURCES_KEY not in successor.metadata   # handed over, not duplicated

    @pytest.mark.asyncio
    async def test_a_trigger_keeps_its_own_inheritance_when_it_absorbs_more(self, manager):
        """Staging MERGES. A trigger can already be carrying sources it never answered, and
        overwriting that list would drop exactly the messages hardest to recover."""
        from message_processor import participation_telemetry as pt
        key = "C123:111.0"
        manager.get_thread_async = AsyncMock(return_value=Mock())
        absorbed = _msg("what's the ETA?", ts="1.0")
        absorbed_id = pt.begin_attempt(absorbed)
        manager.enqueue_pending(key, absorbed)
        trigger_msg = _msg("thoughts?", ts="2.0")
        pt.stage_queue_links(trigger_msg, ["src-old"])   # inherited from an earlier drain
        manager.enqueue_pending(key, trigger_msg)

        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata[pt.BATCHED_SOURCES_KEY] == ["src-old", absorbed_id]

    @pytest.mark.asyncio
    async def test_ungated_batch_members_stage_nothing(self, manager):
        """Mentions and DMs mint no attempt, so there is nothing to link — and no empty key
        left behind for the successor to emit from."""
        from message_processor import participation_telemetry as pt
        key = "C123:111.0"
        manager.get_thread_async = AsyncMock(return_value=Mock())
        manager.enqueue_pending(key, _msg("<@UBOT> ping", ts="1.0"))
        manager.enqueue_pending(key, _msg("thoughts?", ts="2.0"))

        proc = _drain_proc(manager)
        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert pt.BATCHED_SOURCES_KEY not in trigger.metadata


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
        message.metadata = {"gate_required": True, "silence_capable": True,
                            "ts": "111.0"}
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
    async def test_earlier_message_documents_processed_and_folded_on_a_dm(self, manager):
        """A DM has no admission step, so the drain keeps the shipped sequencing verbatim:
        summarize now, fold the summary into the appended content."""
        key = "D08XYZ:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("see attached", channel="D08XYZ",
                              attachments=[{"type": "file", "name": "report.pdf"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("and the summary?", channel="D08XYZ",
                                          user="bob", username="bob", ts="b.ts"))

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
            await proc._dispatch_pending_batch(_msg("done", channel="D08XYZ"), client, key)

        # The earlier message's attachments went through the SAME pipeline the trigger uses,
        # keyed on THAT message and the resolved per-thread CI setting.
        proc._process_attachments.assert_awaited_once()
        assert proc._process_attachments.await_args.args[0] is earlier
        assert proc._process_attachments.await_args.kwargs["code_interpreter_enabled"] is True
        assert proc._process_attachments.await_args.kwargs["defer_document_summaries"] is False
        # Its document summary was folded into that message's appended content.
        appended = [c.args[2] for c in proc._add_message_with_token_management.call_args_list]
        assert appended[0] == "alice: see attached [+doc:report.pdf]"

    @pytest.mark.asyncio
    async def test_earlier_channel_documents_are_staged_for_the_admitted_turn(self, manager):
        """[r5-2] The summary is a Responses API call, and on a CHANNEL catch-up nothing may be
        spent before the turn is admitted. So the drain defers it and hands the staged entries to
        the trigger. The fold goes with it: what it could render now is an excerpt (there is no
        summary yet) into ThreadState.messages, a list the channel request never sends."""
        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("see attached",
                              attachments=[{"type": "file", "name": "report.pdf"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("and the summary?", user="bob", username="bob", ts="b.ts"))

        proc = _drain_proc(manager)
        doc = {"filename": "report.pdf", "summary": None, "_persist": {}}
        proc._process_attachments = AsyncMock(return_value=([], [doc], []))
        proc._build_message_with_documents = Mock()

        client = Mock()
        client.message_handler = Mock()
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": True})):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        assert proc._process_attachments.await_args.kwargs["defer_document_summaries"] is True
        proc._build_message_with_documents.assert_not_called()
        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata["batched_deferred_documents"] == [doc]
        appended = [c.args[2] for c in proc._add_message_with_token_management.call_args_list]
        assert appended[0] == "alice: see attached"

    @pytest.mark.asyncio
    async def test_earlier_message_images_catalogued_on_a_dm(self, manager):
        key = "D08XYZ:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg("look at this", channel="D08XYZ",
                              attachments=[{"type": "image", "name": "shot.png",
                                            "url": "http://x/shot.png"}], ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("thoughts?", channel="D08XYZ",
                                          user="bob", username="bob", ts="b.ts"))

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
            await proc._dispatch_pending_batch(_msg("done", channel="D08XYZ"), client, key)

        # A durable visual description was scheduled for the earlier image (trigger parity).
        catalog.assert_called_once()
        cat_args = catalog.call_args.args
        # The image PARTS are what gets cataloged — the urls are read off the parts themselves,
        # so a link-borne image (no attachment behind it) is described too.
        assert cat_args[2] == img_inputs
        # No documents → the document folder is never invoked.
        proc._build_message_with_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_earlier_channel_images_are_catalogued_only_by_the_admitted_turn(self, manager):
        """[r5-2] The description is a vision call on this bot's account, so a channel catch-up
        stages it too — grouped per source ts, because the description is stored against the
        message that actually carried the image."""
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
        catalog = Mock(return_value=None)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": False})), \
             patch("message_processor.base.image_catalog.catalog_uploads", new=catalog):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        catalog.assert_not_called()
        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata["batched_catalog_uploads"] == [("a.ts", img_inputs)]
        # The parts still ride to the trigger so the model can SEE them (T2-10 is unchanged).
        assert trigger.metadata["batched_image_inputs"] == img_inputs

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
    async def test_a_queued_channel_batch_spends_nothing_before_admission(self, manager):
        """[r5-2] The contract at the seam it was broken at, with the REAL attachment pipeline: a
        queued channel message carrying a document and an image is downloaded and extracted, and
        the OpenAI client records not one call. Both of the descriptions it owes are staged for the
        catch-up turn, which runs them once its request has been measured and accepted."""
        import base64
        import io

        from PIL import Image

        from message_processor.utilities import MessageUtilitiesMixin as U

        png = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 0, 0)).save(png, format="PNG")
        png_bytes = png.getvalue()

        class _RecordingOpenAI:
            """Every Responses API entry point, with a log of the ones that were reached."""

            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                async def _call(*_a, **_k):
                    self.calls.append(name)
                    return "a model said something"
                return _call

        key = "C123:111.0"
        state = Mock()
        state.config_overrides = {}
        manager.get_thread_async = AsyncMock(return_value=state)
        earlier = _attach_msg(
            "the report and a screenshot",
            attachments=[{"type": "file", "name": "q3.pdf", "id": "F1",
                          "mimetype": "application/pdf", "url": "https://x/q3.pdf", "size": 13},
                         {"type": "image", "name": "shot.png", "id": "F2",
                          "mimetype": "image/png", "url": "https://x/shot.png",
                          "size": len(png_bytes)}],
            ts="a.ts")
        manager.enqueue_pending(key, earlier)
        manager.enqueue_pending(key, _msg("what do you make of it?", user="bob",
                                          username="bob", ts="b.ts"))

        proc = _drain_proc(manager)
        recorder = _RecordingOpenAI()
        proc.openai_client = recorder
        proc.db = None
        proc._update_status = Mock()
        proc.image_url_handler = Mock(max_image_size=20 * 1024 * 1024)
        proc.image_url_handler.process_urls_from_text = AsyncMock(return_value=([], []))

        class _Handler:
            max_document_size = 50 * 1024 * 1024

            def is_document_file(self, name, mimetype):
                return True

            async def safe_extract_content_async(self, data, mimetype, name, **kw):
                return {"content": "the whole report", "total_pages": 2}

        proc.document_handler = _Handler()
        for name in ("_process_attachments", "_stage_document_summary",
                     "_summarize_document_for_attach", "_build_message_with_documents",
                     "_apply_scanned_pdf_ocr", "_extract_slack_file_urls",
                     "_native_file_eligible"):
            setattr(proc, name, getattr(U, name).__get__(proc))

        client = Mock()
        client.download_file = AsyncMock(
            side_effect=lambda url, *a, **k: (png_bytes if "shot" in url else b"%PDF-1.4 data"))
        client.message_handler = Mock()
        catalog = Mock(return_value=None)
        with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
             patch.object(config, "get_thread_config_async",
                          new=AsyncMock(return_value={"enable_code_interpreter": False})), \
             patch("message_processor.base.image_catalog.catalog_uploads", new=catalog):
            await proc._dispatch_pending_batch(_msg("done"), client, key)

        assert recorder.calls == [], f"the drain spent a Responses call: {recorder.calls}"
        catalog.assert_not_called()
        trigger = client.message_handler.call_args.args[0]
        staged = trigger.metadata["batched_deferred_documents"]
        assert [d["filename"] for d in staged] == ["q3.pdf"]
        assert staged[0]["summary"] is None and "_persist" in staged[0]
        carried = trigger.metadata["batched_catalog_uploads"]
        assert [ts for ts, _images in carried] == ["a.ts"]
        assert carried[0][1][0]["image_url"].startswith(
            f"data:image/png;base64,{base64.b64encode(png_bytes).decode()[:8]}")

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


@pytest.mark.asyncio
async def test_the_admitted_catch_up_turn_catalogues_the_carried_images():
    """[r5-2] The far end of the staging. The drain refused to describe the earlier messages'
    images, because a description is a vision call and its turn had not been admitted yet — so the
    turn has to do it, at the same point it catalogues its own uploads. Grouped per source ts: the
    description is stored against the message that actually carried the image."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from base_client import Response
    from message_processor.base import MessageProcessor
    from message_processor.turn_runtime import TurnRuntime

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        proc = MessageProcessor()

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False)

    async def _state(*a, **k):
        return state

    proc.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    proc.thread_manager.release_thread_lock = AsyncMock()
    proc._get_or_rebuild_thread_state = _state
    proc.get_or_create_channel_thread_state = _state
    proc._build_channel_turn_stream = AsyncMock(return_value=None)
    proc._admit_channel_request = AsyncMock()
    proc._handle_text_response = AsyncMock(return_value=Response(type="text", content="ok"))
    proc._build_channel_memory_text = AsyncMock(return_value="")
    proc._build_channel_info = AsyncMock(return_value="")
    proc._process_attachments = AsyncMock(return_value=([], [], []))
    proc._schedule_async_call = Mock()

    carried = [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]
    message = Message(text="what do you make of it?", user_id="U1", channel_id="C1",
                      thread_id="10.0", attachments=[],
                      metadata={"ts": "20.0", "queued_batch_size": 2,
                                "batched_image_inputs": carried,
                                "batched_catalog_uploads": [("15.0", carried)]})
    catalog = Mock(return_value=None)
    with patch("message_processor.base.image_catalog.catalog_uploads", new=catalog):
        try:
            await proc.process_message(
                message, MagicMock(), None,
                turn=TurnRuntime.for_message(message, channel_post_allowed=False))
        except Exception:
            pass    # anything past the cataloguing point is another test's business

    catalog.assert_called_once()
    args = catalog.call_args.args
    assert args[2] == carried and args[3] == "15.0"


@pytest.mark.asyncio
async def test_a_queued_messages_file_is_authorized_by_the_catch_up_turn(manager):
    """[r6-3] The absent-source contract, through the production path only.

    A message carrying a CSV queues behind a running turn. The drain folds it into one catch-up
    turn — and Slack has not propagated it into the window that turn fetches, so the stream cannot
    say the file exists. Its id still has to reach `canonical_files`, or the turn answers the
    question with the numbers unreadable (the live failure the cohort machinery exists to prevent).

    Nothing here stages `batched_file_refs`: the drain writes it off the queued message's own event
    payload, and admission reads it. That is the whole point of the test — the reader had no
    producer, and a hand-seeded fixture could not have told us.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from message_processor.turn_runtime import TurnRuntime
    from tests.unit.channel_turn_harness import (no_tools_prepared, normalized,
                                                 pin_channel_turn, steering, thread_config)

    key = "C1:10.0"
    state = Mock()
    state.config_overrides = {}
    manager.get_thread_async = AsyncMock(return_value=state)
    csv = {"type": "file", "name": "data.csv", "id": "F9", "mimetype": "text/csv",
           "url": "https://files.slack.com/files-pri/T1-F9/data.csv", "size": 12}
    manager.enqueue_pending(key, _attach_msg("here are the numbers", attachments=[csv],
                                             channel="C1", thread="10.0", ts="15.0"))
    manager.enqueue_pending(key, _msg("what do the numbers say?", channel="C1", thread="10.0",
                                      user="bob", username="bob", ts="20.0"))

    drain = _drain_proc(manager)
    drain._process_attachments = AsyncMock(return_value=([], [], []))
    client = Mock()
    client.message_handler = Mock()
    with patch("message_processor.base.asyncio.sleep", new=AsyncMock()), \
         patch.object(config, "get_thread_config_async",
                      new=AsyncMock(return_value={"enable_code_interpreter": False})):
        await drain._dispatch_pending_batch(_msg("done", channel="C1", thread="10.0"), client, key)

    trigger = client.message_handler.call_args.args[0]
    assert "batched_file_refs" in trigger.metadata, "the drain must stage the live payload itself"

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        proc = MessageProcessor()
    proc.db = None
    proc._update_status = MagicMock()
    proc._build_channel_info = AsyncMock(return_value=None)
    proc._build_tools_array = MagicMock(return_value=None)
    proc._get_system_prompt = MagicMock(return_value="SYSTEM")
    proc._prepare_channel_turn_tools = AsyncMock(return_value=no_tools_prepared())

    turn = TurnRuntime()
    # The window Slack returned: the question, and no sign of the message that carried the file.
    pin_channel_turn(turn, messages=[normalized("20.0", "what do the numbers say?")],
                     trigger_ts="20.0", prepared=no_tools_prepared())
    await proc._admit_channel_request(
        trigger, MagicMock(), turn, SimpleNamespace(channel_id="C1", thread_ts="10.0"),
        thread_config(), None, stream=turn.channel_stream, steering=steering(),
        image_inputs=[], file_inputs=[], document_inputs=[],
        batched_image_inputs=[], batched_images_omitted=0)

    ctx = turn.channel_turn_context
    assert "F9" in ctx.canonical_files, "the absent source's file was never authorized"
    assert ctx.canonical_files["F9"]["filename"] == "data.csv"
    assert ctx.canonical_files["F9"]["message_ts"] == "15.0"      # fetchable at its real coordinates
    assert [ref.id for source in ctx.cohort_sources for ref in source.files] == ["F9"]
