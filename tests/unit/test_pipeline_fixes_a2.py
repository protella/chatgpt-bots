"""Workstream A2 pre-release fixes — pipeline base + text handler.

Covers:
  F7  — timeout retry preserves the multipart user_content (image/file parts) AND pops the
        stale user turn the first attempt appended (no duplicate user message).
  F21 — a failed overflow Part-2 post retries once, then aborts WITHOUT overwriting Part 1
        (via the extracted _post_overflow_part helper).
  F35 — after a failed native ROLL the legacy fallback targets a NEW message, never the
        finished part (via the extracted _legacy_fallback_target helper).
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from base_client import Message, Response
from thread_manager import AsyncThreadStateManager
from message_processor.base import MessageProcessor
from message_processor.handlers.text import TextHandlerMixin, _legacy_fallback_target


# ----------------------------- F35: _legacy_fallback_target -----------------------------

class TestLegacyFallbackTarget:
    def test_failed_roll_forces_new_message(self):
        # overflow present => the current part is finished; never edit it -> None (new msg)
        assert _legacy_fallback_target("remainder", "PART1_TS", "PART1_TS") is None

    def test_failed_roll_ignores_native_ts_entirely(self):
        # even if the sink reports a ts, a rolled+failed part must not be reused
        assert _legacy_fallback_target("x", "SOME_TS", "OTHER") is None

    def test_non_roll_inert_keeps_native_ts(self):
        # no roll: keep editing the live current part
        assert _legacy_fallback_target(None, "LIVE_TS", "OLD") == "LIVE_TS"

    def test_non_roll_falls_back_to_current_id_when_native_ts_missing(self):
        assert _legacy_fallback_target(None, None, "OLD") == "OLD"


# ----------------------------- F21: _post_overflow_part -----------------------------

class _OverflowProc:
    _post_overflow_part = TextHandlerMixin._post_overflow_part

    def log_warning(self, *a, **k):
        pass


def _ok(ts):
    return {"success": True, "ts": ts}


class TestPostOverflowPart:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds_no_retry(self):
        client = Mock()
        client.send_message_get_ts = AsyncMock(return_value=_ok("P2"))
        proc = _OverflowProc()
        with patch("message_processor.handlers.text.asyncio.sleep", new=AsyncMock()) as slept:
            ts = await proc._post_overflow_part(client, "C1", "T1", "Part 2 …")
        assert ts == "P2"
        client.send_message_get_ts.assert_awaited_once()
        slept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_first_failure(self):
        client = Mock()
        client.send_message_get_ts = AsyncMock(
            side_effect=[{"success": False, "error": "timeout"}, _ok("P2b")])
        proc = _OverflowProc()
        with patch("message_processor.handlers.text.asyncio.sleep", new=AsyncMock()) as slept:
            ts = await proc._post_overflow_part(client, "C1", "T1", "Part 2 …")
        assert ts == "P2b"
        assert client.send_message_get_ts.await_count == 2
        slept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_both_attempts_fail_returns_none(self):
        client = Mock()
        client.send_message_get_ts = AsyncMock(return_value={"success": False})
        proc = _OverflowProc()
        with patch("message_processor.handlers.text.asyncio.sleep", new=AsyncMock()):
            ts = await proc._post_overflow_part(client, "C1", "T1", "Part 2 …")
        assert ts is None
        assert client.send_message_get_ts.await_count == 2


# ----------------------------- F7: timeout retry preserves multipart + pops stale turn ---

class _TState:
    def __init__(self):
        self.messages = []
        self.channel_id = "C1"
        self.thread_ts = "T1"
        self.current_model = None
        self.config_overrides = {}
        self.participants = {}
        self.had_timeout = False
        self.has_trimmed_messages = False
        self.root_author = ("U1", "human")
        self.channel_directives = None
        self.system_prompt = None


class _TimeoutHarness:
    """Binds the REAL process_message; stubs collaborators so the turn reaches
    _handle_text_response, times out once, and retries."""

    process_message = MessageProcessor.process_message
    _dispatch_pending_batch = MessageProcessor._dispatch_pending_batch  # no-op on empty queue

    def __init__(self, manager, thread_state):
        self.thread_manager = manager
        self.db = None
        self._thread_state = thread_state
        self._schedule_async_call = Mock()
        self._add_message_with_token_management = Mock()
        self.calls = []
        self._handle_text_response = AsyncMock(side_effect=self._handle_side_effect)

    # logging
    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass

    # collaborators
    async def _get_or_rebuild_thread_state(self, message, client, thinking_id):
        return self._thread_state

    def _get_system_prompt(self, *a, **k):
        return "sys"

    def _build_participant_roster(self, *a, **k):
        return ""

    async def _build_channel_memory_text(self, channel_id):
        return None

    async def _build_channel_info(self, client, channel_id):
        return None

    async def _process_attachments(self, message, client, thinking_id, code_interpreter_enabled=None):
        # one image, no documents, no unsupported files
        return ([{"type": "input_image", "image_url": "data:image/png;base64,AAAA",
                  "source": "attachment", "filename": "shot.png", "url": "u", "file_id": "f"}], [], [])

    def _build_user_content(self, text, image_inputs, file_inputs):
        # genuine multipart: text part + one image part
        parts = [{"type": "input_text", "text": text}]
        parts += [{"type": "input_image", "image_url": p["image_url"]} for p in (image_inputs or [])]
        return parts

    def _build_message_with_documents(self, text, docs):
        return text

    def _update_status(self, *a, **k):
        pass

    async def _handle_side_effect(self, user_content, thread_state, client, message, thinking_id,
                                  retry_count=0, **kw):
        self.calls.append({"user_content": user_content, "retry_count": retry_count})
        # Simulate text.py appending the user turn to thread state on EVERY attempt.
        thread_state.messages.append({"role": "user", "content": "Alice: hi [image]"})
        if retry_count == 0:
            err = TimeoutError("openai slow")
            err.operation_type = "text_normal"
            raise err
        return Response(type="text", content="recovered")


@pytest.fixture
def manager():
    m = AsyncThreadStateManager(db=None)
    m._token_counter = Mock()
    m._token_counter.count_message_tokens = Mock(return_value=10)
    m._token_counter.count_thread_tokens = Mock(return_value=10)
    return m


def _img_msg():
    # attachments=[] so the async image-catalog schedule is skipped (keeps the test coroutine-free);
    # _process_attachments still yields an image so user_content is genuinely multipart.
    return Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1",
                   attachments=[], metadata={"ts": "T1", "username": "Alice"})


class TestTimeoutRetryPreservesMultipart:
    @pytest.mark.asyncio
    async def test_retry_reuses_multipart_user_content_and_pops_stale_turn(self, manager):
        state = _TState()
        proc = _TimeoutHarness(manager, state)

        client = Mock(spec=[])  # no update_message -> retry status block is skipped
        resp = await proc.process_message(_img_msg(), client=client, thinking_id=None)

        # The turn recovered on retry.
        assert resp.type == "text" and resp.content == "recovered"
        # Two attempts: the first timed out, the second was the retry.
        assert [c["retry_count"] for c in proc.calls] == [0, 1]
        # F7 (multipart preserved): the retry got the ORIGINAL multipart user_content — the
        # SAME list object, carrying the input_image part — not the plain enhanced_text string.
        retry_content = proc.calls[1]["user_content"]
        assert isinstance(retry_content, list)
        assert any(part.get("type") == "input_image" for part in retry_content)
        assert proc.calls[0]["user_content"] is retry_content
        # F7 (no duplicate turn): the first attempt's appended user message was popped before
        # the retry re-appended it, so exactly ONE user turn remains.
        user_turns = [m for m in state.messages if m.get("role") == "user"]
        assert len(user_turns) == 1


# ----------------------------- Wiring: _build_tool_context container_gone_sink (F15) -----

class _CtxProc:
    _build_tool_context = TextHandlerMixin._build_tool_context
    db = None


def test_build_tool_context_wires_container_gone_sink():
    """F15: the ToolContext must hold the SAME dead-container list the API extends, so an
    executor sees its container die mid-turn (container_recycled fail-fast)."""
    proc = _CtxProc()
    msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1",
                  attachments=[], metadata={"ts": "T1"})
    sink: list = []
    tc = proc._build_tool_context(msg, Mock(), {"model": "x"}, "cont-123",
                                  turn=None, container_gone_sink=sink)
    assert tc.container_gone_sink is sink          # same object, not a copy
    assert tc.container_id == "cont-123"
    # the recovery view is live: appending a dead id later is visible through the context
    sink.append("cont-123")
    assert tc.container_recycled() is True


def test_build_tool_context_container_gone_sink_defaults_none():
    proc = _CtxProc()
    msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1",
                  attachments=[], metadata={"ts": "T1"})
    tc = proc._build_tool_context(msg, Mock(), {"model": "x"}, "cont-123")
    assert tc.container_gone_sink is None


# ----------------------------- too_large failed-file notice branch -----------------------

class TestFailedFilesNoticeTooLarge:
    def test_too_large_gets_honest_size_line_not_download_advice(self):
        notice = MessageProcessor._build_failed_files_notice([
            {"name": "big.pdf", "too_large": True,
             "size_bytes": 60 * 1024 * 1024, "limit_bytes": 50 * 1024 * 1024},
        ])
        assert "File Too Large" in notice
        assert "big.pdf" in notice
        assert "60.0MB" in notice and "50.0MB" in notice
        # must NOT be misrouted through the generic download bucket
        assert "Couldn't Download" not in notice

    def test_missing_sizes_degrade_to_question_mark(self):
        notice = MessageProcessor._build_failed_files_notice([
            {"name": "big.pdf", "too_large": True},
        ])
        assert "File Too Large" in notice and "?" in notice

    def test_download_failure_still_its_own_bucket(self):
        notice = MessageProcessor._build_failed_files_notice([
            {"name": "x.pdf", "error": "download_failed"},
        ])
        assert "Couldn't Download" in notice
        assert "File Too Large" not in notice

    def test_too_large_and_unsupported_coexist(self):
        notice = MessageProcessor._build_failed_files_notice([
            {"name": "big.pdf", "too_large": True,
             "size_bytes": 60 * 1024 * 1024, "limit_bytes": 50 * 1024 * 1024},
            {"name": "weird.xyz", "mimetype": "application/x-thing"},
        ])
        assert "File Too Large" in notice and "Unsupported File Type" in notice


# ----------- T2-10: trigger turn folds earlier-batch images + failures into its API turn -----

class _MergeHarness:
    """Binds the REAL process_message; stubs collaborators so a catch-up trigger runs to
    completion and we can inspect the multipart user_content and the messages it added."""

    process_message = MessageProcessor.process_message
    _dispatch_pending_batch = MessageProcessor._dispatch_pending_batch

    def __init__(self, manager, thread_state, attach_result):
        self.thread_manager = manager
        self.db = None
        self._thread_state = thread_state
        self._attach_result = attach_result           # (images, docs, unsupported)
        self._schedule_async_call = Mock()
        self.added = []                                # (role, content)
        self.captured = {}
        self._handle_text_response = AsyncMock(side_effect=self._handle)

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass

    async def _get_or_rebuild_thread_state(self, message, client, thinking_id):
        return self._thread_state

    def _get_system_prompt(self, *a, **k):
        return "sys"

    def _build_participant_roster(self, *a, **k):
        return ""

    async def _build_channel_memory_text(self, channel_id):
        return None

    async def _build_channel_info(self, client, channel_id):
        return None

    async def _process_attachments(self, message, client, thinking_id, code_interpreter_enabled=None):
        return self._attach_result

    def _build_user_content(self, text, image_inputs, file_inputs):
        parts = [{"type": "input_text", "text": text}]
        parts += [{"type": "input_image", "image_url": p.get("image_url", "x")}
                  for p in (image_inputs or [])]
        return parts

    def _build_message_with_documents(self, text, docs):
        return text

    def _format_user_content_with_username(self, content, message):
        username = (message.metadata or {}).get("username", "User")
        return f"{username}: {content}"

    def _build_failed_files_notice(self, files):
        return MessageProcessor._build_failed_files_notice(files)

    def _update_status(self, *a, **k):
        pass

    def _add_message_with_token_management(self, thread_state, role, content, **kw):
        self.added.append((role, content))

    async def _handle(self, user_content, thread_state, client, message, thinking_id, **kw):
        self.captured["user_content"] = user_content
        return Response(type="text", content="ok")


def _mk_image(i):
    return {"type": "input_image", "image_url": f"data:image/png;base64,IMG{i}",
            "source": "attachment", "filename": f"img{i}.png", "url": f"u{i}", "file_id": f"f{i}"}


def _trigger(batched_images=None, batched_unsupported=None, text="catch up"):
    md = {"ts": "T1", "username": "Alice"}
    if batched_images is not None:
        md["batched_image_inputs"] = batched_images
    if batched_unsupported is not None:
        md["batched_unsupported_files"] = batched_unsupported
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="T1",
                   attachments=[], metadata=md)


def _image_parts(user_content):
    return [p for p in user_content if p.get("type") == "input_image"]


def _text_part(user_content):
    return next(p["text"] for p in user_content if p.get("type") == "input_text")


class TestBatchedImageMerge:
    @pytest.mark.asyncio
    async def test_under_cap_folds_all_earlier_images_no_note(self, manager):
        own = [_mk_image(0)]
        batched = [_mk_image(1), _mk_image(2)]
        proc = _MergeHarness(manager, _TState(), (own, [], []))
        await proc.process_message(_trigger(batched_images=batched), client=Mock(spec=[]), thinking_id=None)
        uc = proc.captured["user_content"]
        assert len(_image_parts(uc)) == 3          # 1 own + 2 earlier
        assert "couldn't be attached" not in _text_part(uc)

    @pytest.mark.asyncio
    async def test_over_cap_prefers_own_and_notes_omission(self, manager):
        own = [_mk_image(i) for i in range(9)]
        batched = [_mk_image(100 + i) for i in range(3)]
        proc = _MergeHarness(manager, _TState(), (own, [], []))
        await proc.process_message(_trigger(batched_images=batched), client=Mock(spec=[]), thinking_id=None)
        uc = proc.captured["user_content"]
        img_parts = _image_parts(uc)
        # capped at 10: all 9 own kept, exactly 1 earlier image fills the last slot
        assert len(img_parts) == 10
        own_urls = {p["image_url"] for p in own}
        assert own_urls.issubset({p["image_url"] for p in img_parts})
        # 2 earlier images dropped → the model is told
        assert "2 image(s) from earlier messages" in _text_part(uc)

    @pytest.mark.asyncio
    async def test_earlier_failures_reach_failed_files_notice(self, manager):
        # trigger has text but no own attachments; the earlier failure must still be acknowledged
        proc = _MergeHarness(manager, _TState(), ([], [], []))
        fail = {"name": "broken.pdf", "error": "download_failed"}
        client = Mock()
        client.send_message = AsyncMock()
        await proc.process_message(_trigger(batched_unsupported=[fail]),
                                   client=client, thinking_id=None)
        assistant_msgs = [c for (r, c) in proc.added if r == "assistant"]
        assert any("broken.pdf" in m and "Couldn't Download" in m for m in assistant_msgs)
        # the turn still proceeds to answer (text present)
        assert "user_content" in proc.captured

    @pytest.mark.asyncio
    async def test_mixed_request_actually_posts_notice_to_user(self, manager):
        """Residual T2-10: on a MIXED request (some files OK, some failed) the notice was only
        recorded in thread state, never delivered — the model believed it acknowledged the
        failure while the user saw nothing. It must genuinely reach the user."""
        own = [_mk_image(0)]                          # one image processed OK
        fail = {"name": "broken.pdf", "error": "download_failed"}  # one file failed
        proc = _MergeHarness(manager, _TState(), (own, [], [fail]))
        client = Mock()
        client.send_message = AsyncMock()
        msg = Message(text="here you go", user_id="U1", channel_id="C1", thread_id="T1",
                      attachments=[], metadata={"ts": "T1", "username": "Alice"})
        await proc.process_message(msg, client=client, thinking_id=None)

        # The notice was POSTED to the user, not merely recorded in-memory.
        client.send_message.assert_awaited_once()
        posted = client.send_message.await_args.kwargs["text"]
        assert "broken.pdf" in posted and "Couldn't Download" in posted
        # Recorded in state too, so the model's context matches what was delivered.
        assert any("broken.pdf" in c for (r, c) in proc.added if r == "assistant")
        # The turn still proceeds to generate the real reply (image was fine).
        assert "user_content" in proc.captured
        assert len(_image_parts(proc.captured["user_content"])) == 1

    @pytest.mark.asyncio
    async def test_normal_message_without_batched_metadata_is_unaffected(self, manager):
        own = [_mk_image(0)]
        proc = _MergeHarness(manager, _TState(), (own, [], []))
        # a plain message — no batched_* keys
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1",
                      attachments=[], metadata={"ts": "T1", "username": "Alice"})
        await proc.process_message(msg, client=Mock(spec=[]), thinking_id=None)
        uc = proc.captured["user_content"]
        assert len(_image_parts(uc)) == 1
        assert "couldn't be attached" not in _text_part(uc)
