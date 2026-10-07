"""Phase Q — conversational queueing (busy rejection retired).

Messages arriving while a conversation is mid-processing queue on the state manager
and are answered by the finishing turn's drain hook as ONE batched catch-up turn.
Covers: queue primitives, the process_message contention path, the drain/dispatch
hook, gate-order eligibility, needs_refresh interplay, and busy retirement gates.
"""
import pathlib
from unittest.mock import Mock, AsyncMock, patch

import pytest

from message_processor.client_contract import Message
from config import config
from message_processor.thread_manager import AsyncThreadStateManager
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


# --- Burst follow-ups R2-2 / R3-1: what a Phase Q catch-up already owns ---

def _scoped(ts, *, thread=None, sender="U1", channel="C123"):
    """A queued message carrying the identity the stale guard scopes by."""
    message = _msg(f"m{ts}", user=sender, channel=channel, thread=thread or ts, ts=ts)
    message.metadata["sender_id"] = sender
    return message


class TestPendingInScope:
    TURN_SCOPES = (("thread", "C123", "100.0"), ("top", "C123", "U1"))

    def test_queued_newer_messages_in_an_overlapping_scope_are_owned(self, manager):
        manager.enqueue_pending("C123:100.0", _scoped("101.0", thread="100.0", sender="U2"))
        manager.enqueue_pending("C123:102.0", _scoped("102.0"))                 # same sender, top
        manager.enqueue_pending("C123:103.0", _scoped("103.0", sender="U9"))     # someone else
        manager.enqueue_pending("C999:104.0", _scoped("104.0", channel="C999"))  # other channel
        assert manager.pending_ts_in_scope("C123", self.TURN_SCOPES, "100.0") == [
            "101.0", "102.0"]
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is True
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "102.0") is False  # not newer

    def test_a_popped_batch_stays_owned_until_its_trigger_is_admitted(self, manager):
        key = "C123:100.0"
        manager.enqueue_pending(key, _scoped("101.0", thread="100.0", sender="U2"))
        manager.enqueue_pending(key, _scoped("102.0", thread="100.0", sender="U3"))
        batch = manager.pop_pending_batch(key, 10)
        assert manager.pending_count(key) == 0
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is True
        manager.bind_dispatch(batch[-1], batch)
        manager.end_dispatch_for(batch[-1])                      # begin_turn on the trigger
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is False

    def test_an_older_dispatch_never_releases_a_newer_dispatch_of_the_same_trigger(
            self, manager):
        """R4 #5: the trigger lost the lock race, was requeued and re-dispatched; the first
        dispatch's ending must not release the second's ownership."""
        key = "C123:100.0"
        trigger = _scoped("101.0", thread="100.0", sender="U2")
        manager.enqueue_pending(key, trigger)
        first = manager.bind_dispatch(trigger, manager.pop_pending_batch(key, 10))
        manager.end_dispatch_for(trigger)                    # admitted, then requeued
        manager.enqueue_pending(key, trigger)
        second = manager.bind_dispatch(trigger, manager.pop_pending_batch(key, 10))
        assert first is not second
        manager.end_dispatch_for(trigger, first)             # the old task finally ends
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is True
        manager.end_dispatch_for(trigger, second)
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is False

    @pytest.mark.asyncio
    async def test_a_drain_with_no_handler_releases_what_it_popped(self, manager):
        key = "C123:100.0"
        manager.enqueue_pending(key, _scoped("101.0", thread="100.0", sender="U2"))
        proc = _drain_proc(manager)
        client = Mock(spec=[])                                   # no message_handler
        with patch.object(config, "queue_drain_linger_seconds", 0.0):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["cancelled_before_start", "raised"])
    async def test_a_scheduled_catch_up_releases_the_batch_however_it_ends(self, manager,
                                                                         ending):
        """The task's done-callback releases ownership — including a cancellation before its
        first step, which no `finally` inside the task could see."""
        import asyncio
        from message_processor.utilities import MessageUtilitiesMixin

        key = "C123:100.0"
        manager.enqueue_pending(key, _scoped("101.0", thread="100.0", sender="U2"))
        manager.get_thread_async = AsyncMock(return_value=Mock())
        proc = _drain_proc(manager)
        proc._schedule_async_call = MessageUtilitiesMixin._schedule_async_call.__get__(proc)
        client = Mock()
        client.message_handler = AsyncMock(side_effect=RuntimeError("refused before admission"))
        with patch.object(config, "queue_drain_linger_seconds", 0.0):
            await proc._dispatch_pending_batch(_msg("done"), client, key)
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is True
        (task,) = proc._background_tasks
        if ending == "cancelled_before_start":
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)                                   # let done-callbacks run
        assert manager.pending_in_scope("C123", self.TURN_SCOPES, "100.0") is False
        assert client.message_handler.await_count == (0 if ending == "cancelled_before_start"
                                                      else 1)


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

def _start_scheduled(coro):
    """Stand-in for `_schedule_async_call`: take the catch-up's FIRST step, which is where the
    handler is now invoked (so its call is recorded), then discard the coroutine — the stub
    handlers here are plain Mocks and nothing awaits their result."""
    try:
        coro.send(None)
    except (StopIteration, TypeError):
        pass
    finally:
        coro.close()


def _drain_proc(manager):
    proc = _StubProcessor(manager)
    proc._format_user_content_with_username = lambda content, m: f"{m.metadata.get('username')}: {content}"
    proc._add_message_with_token_management = Mock()
    proc._schedule_async_call = Mock(side_effect=_start_scheduled)
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
                    "message_processor/client_contract.py",
                    "message_processor/thread_manager.py"):
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

    from message_processor.client_contract import Response
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



# --- A burst of DMs, end to end through the real process_message and its cleanup ---

DM = "D08XYZ"
DM_KEY = f"{DM}:111.0"


def _dm(ts):
    """A DM in the conversation keyed DM_KEY, from the one sender (U1)."""
    return _scoped(ts, thread="111.0", channel=DM)


@pytest.fixture
def dm_env():
    with patch.object(config, "queue_drain_linger_seconds", 0.0), \
         patch("message_processor.base.channel_steering.load_snapshot",
               new=AsyncMock(return_value=None)), \
         patch("message_processor.base.channel_steering.stamp"):
        yield


def _burst_proc(manager, reply):
    """The REAL process_message on a DM; `reply(message)` stands in for the model turn. Returns
    the processor, its client, and what each catch-up turn returned."""
    from unittest.mock import MagicMock

    from message_processor.client_contract import Response

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        proc = MessageProcessor()
    proc.db = None
    proc.thread_manager = manager

    def _state(*a, **k):
        return manager.get_thread("111.0", DM)            # created by the lock acquisition

    async def _handle(_content, _state, _client, message, *a, **k):
        await reply(message)
        return Response(type="text", content="ok")

    proc._get_or_rebuild_thread_state = AsyncMock(side_effect=_state)
    proc._process_attachments = AsyncMock(return_value=([], [], []))
    proc._handle_text_response = AsyncMock(side_effect=_handle)
    client = MagicMock()
    client.triggers = []
    catch_up: list = []

    async def _handler(trigger, _client):
        client.triggers.append(trigger)
        catch_up.append(await proc.process_message(trigger, _client, None))

    client.message_handler = _handler
    return proc, client, catch_up


async def _leased(proc, client, marks, message, leases):
    """One turn as main.py runs it: a send lease opened in the turn's own task."""
    from message_processor.turn_runtime import TurnRuntime

    lease = leases[message.metadata["ts"]] = marks.begin_turn(message)
    turn = TurnRuntime.for_message(message)
    turn.send_lease, lease.turn = lease, turn
    try:
        return await proc.process_message(message, client, None, turn=turn)
    finally:
        lease.close()


async def _settle(proc, catch_up):
    import asyncio
    for _ in range(50):
        if catch_up:
            break
        await asyncio.sleep(0)
    await asyncio.gather(*list(getattr(proc, "_background_tasks", ())), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["clean", "cancelled_in_release"])
async def test_the_catch_up_turn_is_admitted_after_the_finished_turn_releases_the_lock(
        manager, dm_env, ending):
    """Turn 1 runs, a follow-up queues behind it, and turn 1's drain hands it to a catch-up. That
    turn must find the lock FREE and run — one that starts before the release finds it held,
    re-enqueues itself, and nobody is left to drain it. Also when the ending turn is cancelled
    during its lock release: the release still completes and the catch-up still starts."""
    import asyncio

    queued, answered = [], []

    async def _reply(message):
        if message.text == "m111.0":
            # A follow-up lands while this turn holds the lock: it queues.
            queued.append(await proc.process_message(_dm("112.0"), client, None))
        answered.append(message.text)

    proc, client, catch_up = _burst_proc(manager, _reply)
    release = manager.release_thread_lock
    releasing = asyncio.Event()
    finish_release = asyncio.Event()

    async def _slow_release(*a, **k):
        releasing.set()
        await finish_release.wait()
        await release(*a, **k)

    if ending == "cancelled_in_release":
        manager.release_thread_lock = _slow_release

    turn = asyncio.ensure_future(proc.process_message(_dm("111.0"), client, None))
    if ending == "cancelled_in_release":
        await releasing.wait()
        turn.cancel()                                      # an early stand-down in cleanup
        await asyncio.gather(turn, return_exceptions=True)
        assert turn.cancelled()
        finish_release.set()
    else:
        await turn
    await _settle(proc, catch_up)

    assert [r.type for r in queued] == ["queued"]
    assert [r.type for r in catch_up] == ["text"], "the catch-up re-queued instead of running"
    assert answered == ["m111.0", "m112.0"]
    assert manager.pending_count(DM_KEY) == 0
    assert manager.is_thread_processing("111.0", DM) is False


@pytest.mark.asyncio
async def test_a_message_arriving_as_the_turn_ends_with_nothing_queued_is_still_answered(
        manager, dm_env):
    """The drain found nothing queued, so no catch-up exists. A message already runnable at that
    instant must not slip in between the last pending check and the unlock — it would queue
    behind a turn that has already decided there is nothing left to drain."""
    import asyncio

    answered, late = [], []

    async def _reply(message):
        if message.text == "m111.0":
            late.append(asyncio.ensure_future(proc.process_message(_dm("112.0"), client, None)))
        answered.append(message.text)

    proc, client, catch_up = _burst_proc(manager, _reply)
    await proc.process_message(_dm("111.0"), client, None)
    (follow_up,) = late
    response = await follow_up
    await _settle(proc, catch_up)

    assert response.type == "text", "the follow-up queued with nobody left to drain it"
    assert answered == ["m111.0", "m112.0"]
    assert manager.pending_count(DM_KEY) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_started", [False, True])
async def test_a_same_sender_follow_up_stands_down_the_running_turn_if_eligible(
        manager, dm_env, tool_started):
    """Threads and DMs get the top-level behavior: the same sender's follow-up queues behind the
    running turn and stands it down, and ONE catch-up answers both. A turn with a local tool in
    flight is not eligible — it finishes, and the catch-up answers the rest."""
    import asyncio

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    in_model, go = asyncio.Event(), asyncio.Event()
    answered = []

    async def _reply(message):
        if message.text == "m111.0":
            in_model.set()
            await go.wait()                               # the model call
        answered.append(message.text)

    proc, client, catch_up = _burst_proc(manager, _reply)
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await in_model.wait()
    if tool_started:
        leases["111.0"].tools_started = True
    queued = await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
    go.set()
    await asyncio.gather(first, return_exceptions=True)
    await _settle(proc, catch_up)

    assert queued.type == "queued"
    assert [r.type for r in catch_up] == ["text"]
    if tool_started:
        assert leases["111.0"].preempted_by is None and first.result().type == "text"
        assert answered == ["m111.0", "m112.0"]
    else:
        assert leases["111.0"].preempted_by == "112.0" and first.cancelled()
        assert answered == ["m112.0"]                     # one reply, from the catch-up
    assert manager.is_thread_processing("111.0", DM) is False


@pytest.mark.asyncio
async def test_a_turn_still_acquiring_the_lock_is_never_stood_down_by_a_queued_follow_up(
        manager, dm_env):
    """A turn holding the lock but not yet inside the cleanup that releases it must not be
    cancelled by a queued follow-up: it would never release or drain. It runs, the catch-up
    answers the follow-up — and a cancellation in that window releases the lock regardless."""
    import asyncio

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    answered = []

    async def _reply(message):
        answered.append(message.text)

    proc, client, catch_up = _burst_proc(manager, _reply)
    acquiring, go = asyncio.Event(), asyncio.Event()
    create = manager.get_or_create_thread_async

    async def _slow_create(*a, **k):
        if not go.is_set():
            acquiring.set()
            await go.wait()
        return await create(*a, **k)

    manager.get_or_create_thread_async = _slow_create
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await acquiring.wait()                                # lock held, cleanup not yet armed
    queued = await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
    assert queued.type == "queued" and leases["111.0"].preempted_by is None
    go.set()
    assert (await first).type == "text"
    await _settle(proc, catch_up)
    assert [r.type for r in catch_up] == ["text"]
    assert answered == ["m111.0", "m112.0"]

    acquiring.clear()
    go.clear()
    acquire = asyncio.ensure_future(manager.acquire_thread_lock("111.0", DM))
    await acquiring.wait()
    acquire.cancel()
    await asyncio.gather(acquire, return_exceptions=True)
    assert manager.is_thread_processing("111.0", DM) is False


@pytest.mark.asyncio
async def test_a_stood_down_message_joins_the_catch_up_as_its_oldest_member(manager, dm_env):
    """The stood-down turn's message is not merely context the catch-up might fetch: it rejoins
    the queue ahead of the follow-up and is drained like any queued message — once, in order,
    even though its own turn had already put it in warm state."""
    import asyncio

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    in_model = asyncio.Event()

    async def _reply(message):
        if message.text == "m111.0":
            state = manager.get_thread("111.0", DM)
            proc._add_message_with_token_management(state, "user", "U1: m111.0",
                                                    message_ts="111.0")
            in_model.set()
            await asyncio.Event().wait()                  # the model call; stood down here

    proc, client, catch_up = _burst_proc(manager, _reply)
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await in_model.wait()
    await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
    await asyncio.gather(first, return_exceptions=True)
    await _settle(proc, catch_up)

    (trigger,) = client.triggers
    assert trigger.text == "m112.0" and trigger.metadata["queued_batch_size"] == 2
    warm = manager.get_thread("111.0", DM).messages
    assert [(m.get("metadata") or {}).get("ts") for m in warm] == ["111.0"]


@pytest.mark.asyncio
async def test_a_stood_down_mention_hands_its_words_and_file_to_an_ungated_successor(manager):
    """A channel-thread @mention stood down by the same sender's thread continuation: the
    continuation is ungated and silence-capable, so the catch-up could have ended in silence on
    a direct request. The mention's obligation and its file ride the catch-up regardless."""
    from message_processor.routing_facts import GATE_REQUIRED, SILENCE_CAPABLE

    key = "C1:10.0"
    manager.get_thread_async = AsyncMock(return_value=None)
    csv = {"type": "file", "name": "data.csv", "id": "F9", "mimetype": "text/csv",
           "url": "https://files.slack.com/files-pri/T1-F9/data.csv", "size": 12}
    mention = _attach_msg("<@UBOT> what do these say?", attachments=[csv], user="U1",
                          channel="C1", thread="10.0", ts="15.0")
    mention.metadata.update({GATE_REQUIRED: False, SILENCE_CAPABLE: False})
    follow_up = _msg("and the totals?", channel="C1", thread="10.0", ts="20.0")
    follow_up.metadata.update({GATE_REQUIRED: False, SILENCE_CAPABLE: True})
    manager.enqueue_pending(key, follow_up)
    assert manager.requeue_superseded(key, mention, "20.0") is True
    assert manager.requeue_superseded(key, mention, "99.0") is False   # covered elsewhere

    drain = _drain_proc(manager)
    client = Mock()
    client.message_handler = Mock()
    with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
        await drain._dispatch_pending_batch(_msg("done", channel="C1", thread="10.0"), client, key)

    trigger = client.message_handler.call_args.args[0]
    assert trigger is follow_up and trigger.metadata["queued_batch_size"] == 2
    assert trigger.metadata[SILENCE_CAPABLE] is False, "the mention's words were dropped"
    assert "batched_file_refs" in trigger.metadata, "the mention's file never reached the turn"


@pytest.mark.asyncio
async def test_a_turn_taking_its_lock_is_never_stood_down_by_the_channel_path(manager, dm_env):
    """A newer same-sender turn elsewhere (the channel path's preemption) must not cancel a turn
    that holds its lock but has not reached the cleanup that releases it and drains: a reply
    queued behind that turn would sit there forever."""
    import asyncio

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    answered = []

    async def _reply(message):
        answered.append(message.text)

    proc, client, catch_up = _burst_proc(manager, _reply)
    acquiring, go = asyncio.Event(), asyncio.Event()
    create = manager.get_or_create_thread_async

    async def _slow_create(*a, **k):
        if not go.is_set():
            acquiring.set()
            await go.wait()
        return await create(*a, **k)

    manager.get_or_create_thread_async = _slow_create
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await acquiring.wait()                                 # lock held, cleanup not yet armed
    other = _scoped("112.0", thread="111.0", sender="U2", channel=DM)
    assert (await asyncio.ensure_future(
        _leased(proc, client, marks, other, leases))).type == "queued"

    async def _newer_elsewhere():
        newer = marks.begin_turn(_scoped("113.0", channel=DM))   # same sender, own conversation
        try:
            return newer.preempt_older_same_sender()
        finally:
            newer.close()

    assert await asyncio.ensure_future(_newer_elsewhere()) == []
    go.set()
    assert (await first).type == "text"
    await _settle(proc, catch_up)
    assert [r.type for r in catch_up] == ["text"]
    assert answered == ["m111.0", "m112.0"]


@pytest.mark.asyncio
async def test_a_turn_stood_down_mid_rebuild_leaves_the_catch_up_to_fetch_history(
        manager, dm_env):
    """After a restart the conversation's state is cold. A turn stood down while still rebuilding
    it must not let the drain's appends make that state look warm: the catch-up fetches the
    history, or it answers with only the burst and none of the conversation before it."""
    import asyncio
    from types import SimpleNamespace

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    proc, client, catch_up = _burst_proc(manager, AsyncMock())
    rebuilding, fetched = asyncio.Event(), []

    async def _state(message, *a, **k):
        if message.text == "m111.0":
            rebuilding.set()
            await asyncio.Event().wait()                  # the Slack fetch; stood down here
        warm = manager.get_thread("111.0", DM)
        fetched.append(not warm.messages or manager.consume_needs_refresh(DM_KEY))
        return SimpleNamespace(had_timeout=False, messages=[], thread_ts="111.0", channel_id=DM,
                               root_author=("U1", "human"), config_overrides={},
                               participants={}, current_model=None, has_trimmed_messages=False)

    proc._get_or_rebuild_thread_state = _state
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await rebuilding.wait()
    await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
    await asyncio.gather(first, return_exceptions=True)
    await _settle(proc, catch_up)

    assert leases["111.0"].preempted_by == "112.0"
    assert fetched == [True], "the catch-up trusted a cold state the drain had appended to"


@pytest.mark.asyncio
async def test_a_second_stand_down_keeps_what_the_first_catch_up_inherited(manager):
    """A (with a file) is absorbed into B's catch-up; B is then stood down by C. C's catch-up
    must still carry A as a source and A's file payload — not just B's own."""
    from message_processor.channel_request import cohort_sources_from_message

    key = "C1:10.0"
    manager.get_thread_async = AsyncMock(return_value=None)
    csv = {"type": "file", "name": "data.csv", "id": "F9", "mimetype": "text/csv",
           "url": "https://files.slack.com/files-pri/T1-F9/data.csv", "size": 12}
    first = _attach_msg("here are the numbers", attachments=[csv], user="U1",
                        channel="C1", thread="10.0", ts="15.0")
    second = _msg("what do they say?", channel="C1", thread="10.0", ts="20.0")
    third = _msg("and the totals?", channel="C1", thread="10.0", ts="25.0")
    drain = _drain_proc(manager)
    client = Mock()
    client.message_handler = Mock()
    done = _msg("done", channel="C1", thread="10.0")
    with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
        manager.enqueue_pending(key, first)
        manager.enqueue_pending(key, second)
        await drain._dispatch_pending_batch(done, client, key)          # B's catch-up
        manager.enqueue_pending(key, third)
        assert manager.requeue_superseded(key, second, "25.0") is True  # B stood down by C
        await drain._dispatch_pending_batch(second, client, key)        # C's catch-up

    trigger = client.message_handler.call_args.args[0]
    assert trigger is third
    assert {s.ts for s in cohort_sources_from_message(trigger)} == {"15.0", "20.0"}
    files = {e["ts"]: e["attachments"] for e in trigger.metadata["batched_file_refs"]}
    assert files["15.0"][0]["id"] == "F9", "the first message's file was dropped"


@pytest.mark.asyncio
async def test_a_follow_up_the_full_queue_rejected_never_stands_the_turn_down(manager, dm_env):
    """A follow-up dropped by a full queue is not going to be answered by any catch-up, so it
    must not stand the running turn down — that would lose both."""
    import asyncio

    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    in_model, go = asyncio.Event(), asyncio.Event()

    async def _reply(message):
        if message.text == "m111.0":
            in_model.set()
            await go.wait()

    proc, client, catch_up = _burst_proc(manager, _reply)
    with patch.object(config, "queue_max_pending", 1):
        first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
        await in_model.wait()
        manager.enqueue_pending(DM_KEY, _scoped("111.5", thread="111.0", sender="U2", channel=DM))
        await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
        assert leases["111.0"].preempted_by is None
        go.set()
        assert (await first).type == "text"
        await _settle(proc, catch_up)
    assert [t.text for t in client.triggers] == ["m111.5"]


@pytest.mark.asyncio
async def test_a_refetch_missing_the_stood_down_message_still_carries_it(manager, dm_env):
    """Cold DM: A is stood down before its history loaded, so the catch-up refetches — and Slack's
    snapshot does not have A yet. A's question still reaches the model: put back once, in ts
    order, after the history that came before it."""
    import asyncio

    from message_processor.client_contract import Response
    from message_processor.stale_send_guard import ConversationWatermarks

    marks, leases = ConversationWatermarks(), {}
    proc, client, catch_up = _burst_proc(manager, AsyncMock())
    rebuilding, seen = asyncio.Event(), []

    async def _state(message, *a, **k):
        if message.text == "m111.0":
            rebuilding.set()
            await asyncio.Event().wait()                  # the Slack fetch; stood down here
        warm = manager.get_thread("111.0", DM)
        assert manager.consume_needs_refresh(DM_KEY)
        warm.messages.clear()                             # the refetch: A has not propagated
        warm.add_message("user", "U1: earlier", message_ts="100.0")
        return warm

    async def _handle(_content, state, *a, **k):
        seen.append([((m.get("metadata") or {}).get("ts"), m["content"].split("] ")[-1])
                     for m in state.messages])
        return Response(type="text", content="ok")

    proc._get_or_rebuild_thread_state = _state
    proc._handle_text_response = AsyncMock(side_effect=_handle)
    first = asyncio.ensure_future(_leased(proc, client, marks, _dm("111.0"), leases))
    await rebuilding.wait()
    await asyncio.ensure_future(_leased(proc, client, marks, _dm("112.0"), leases))
    await asyncio.gather(first, return_exceptions=True)
    await _settle(proc, catch_up)

    assert seen == [[("100.0", "U1: earlier"), ("111.0", "U1: m111.0")]]


@pytest.mark.asyncio
async def test_a_requeued_catch_up_trigger_keeps_the_history_it_already_carried(manager):
    """Catch-up B carries A's history, loses the lock race and queues again behind C. The next
    drain keeps B as trigger: A's saved history must survive alongside C's."""
    key = DM_KEY
    state = Mock()
    state.config_overrides = {}
    manager.get_thread_async = AsyncMock(return_value=state)
    drain = _drain_proc(manager)
    client = Mock()
    client.message_handler = Mock()
    a, b, c = _dm("111.0"), _dm("112.0"), _dm("113.0")
    with patch("message_processor.base.asyncio.sleep", new=AsyncMock()):
        manager.enqueue_pending(key, a)
        manager.enqueue_pending(key, b)
        await drain._dispatch_pending_batch(_dm("110.0"), client, key)    # B carries A
        manager.enqueue_pending(key, c)
        manager.enqueue_pending(key, b)                                    # B lost the race
        await drain._dispatch_pending_batch(_dm("110.5"), client, key)

    assert client.message_handler.call_args.args[0] is b
    assert [ts for ts, _content in b.metadata["batched_history"]] == ["111.0", "113.0"]
