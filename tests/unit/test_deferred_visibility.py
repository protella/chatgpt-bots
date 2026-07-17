"""F38 — a turn that might say nothing shows nothing until it commits.

The bug, as reported: the bot would flash a "Thinking…" indicator in the channel and then
vanish without replying. The classifier was never the culprit — it returns before any UI. The
flash came from the SECOND decider: after the gate says "respond" (or on a thread continuation,
which skips the gate entirely), main.py posted the indicator and only THEN did the model run,
still free to call `no_response_needed`.

So a silence-capable turn now defers every speculative surface. What makes that non-trivial is
that `thinking_id=None` already meant "status-only surface" — so simply not posting the
placeholder would have re-routed phase updates to setStatus, which renders a thinking status
AND auto-opens the thread. That would have moved the flash, not removed it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor.turn_runtime import TurnRuntime


def _message(*, channel="C1", thread="10.0", ts="10.0", **meta):
    from base_client import Message
    m = Message(text="hey has anyone looked at the Q3 numbers", user_id="U1",
                channel_id=channel, thread_id=thread, metadata={"ts": ts, **meta})
    return m


# --------------------------------------------------------------- the predicate

def test_ambient_channel_message_is_silence_capable(monkeypatch):
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime.for_message(_message(participation_check=True), "10.0")
    assert turn.silence_capable is True
    assert turn.progress_enabled is False       # show nothing until it commits


def test_thread_continuation_is_silence_capable(monkeypatch):
    """The one that bit the user live: a 1:1 thread reply skips the gate entirely, so the
    model is the ONLY decider — and it can still bow out."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime.for_message(
        _message(wake_source="thread_continuation"), "10.0")
    assert turn.silence_capable is True
    assert turn.progress_enabled is False


def test_an_addressed_turn_still_shows_progress(monkeypatch):
    """A DM or an @-mention always gets an answer, so it keeps its indicator."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime.for_message(_message(wake_source="app_mention"), "10.0")
    assert turn.silence_capable is False
    assert turn.progress_enabled is True


def test_no_reply_tool_disabled_means_the_turn_always_answers(monkeypatch):
    """With the tool off the model CANNOT stay silent, so there is nothing to defer for."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    turn = TurnRuntime.for_message(_message(participation_check=True), "10.0")
    assert turn.silence_capable is False
    assert turn.progress_enabled is True


# --------------------------------------------------------------- no speculative chrome

@pytest.mark.asyncio
async def test_a_silent_ambient_turn_posts_nothing_at_all(monkeypatch):
    """End to end: gate says respond, model says no_reply → the channel sees NOTHING.

    Not a placeholder, not a composer status, not a status clear (which would itself
    auto-open the thread to say the bot is done doing nothing)."""
    from main import ChatBotV2
    from base_client import Response

    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    handler = ChatBotV2.__new__(ChatBotV2)
    handler.processor = SimpleNamespace(thread_manager=None)
    handler._run_participation_gate = AsyncMock(
        return_value=SimpleNamespace(placement="thread", reason="worth answering",
                                     burst_earlier=None))
    handler.processor.process_message = AsyncMock(return_value=Response(
        type="text", content="",
        metadata={"terminal_action": "no_reply", "reason": "not for me", "posted": False}))

    client = MagicMock()
    client.send_thinking_indicator = AsyncMock()
    client.set_assistant_status = AsyncMock()
    client.clear_assistant_status = AsyncMock()
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock()
    client.channel_pulse = None

    await handler.handle_message(
        _message(participation_check=True, wake_source="ambient"), client)

    client.send_thinking_indicator.assert_not_awaited()
    client.set_assistant_status.assert_not_awaited()
    client.clear_assistant_status.assert_not_awaited()
    client.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_addressed_turn_still_gets_its_indicator(monkeypatch):
    """The regression guard on the other side: don't defer what was never at risk."""
    from main import ChatBotV2
    from base_client import Response

    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    handler = ChatBotV2.__new__(ChatBotV2)
    handler.processor = SimpleNamespace(thread_manager=None)
    handler.processor.process_message = AsyncMock(return_value=Response(
        type="text", content="Revenue came in at 4.2M.", metadata={"posted": True}))
    handler._is_unprompted_turn = MagicMock(return_value=False)

    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value="T1")
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock()
    client.format_text = MagicMock(side_effect=lambda t: t)
    client.channel_pulse = None

    await handler.handle_message(_message(wake_source="app_mention"), client)
    client.send_thinking_indicator.assert_awaited_once()


# --------------------------------------------------------------- _update_status

def test_update_status_is_a_no_op_on_a_deferred_turn():
    """THE trap: `thinking_id is None` used to be sufficient proof of "status-only surface".
    On a deferred turn it means the opposite — say nothing."""
    from message_processor.utilities import MessageUtilitiesMixin

    host = MagicMock()
    real = MessageUtilitiesMixin._update_status
    client = MagicMock()
    client.set_assistant_status = MagicMock()

    deferred = TurnRuntime(silence_capable=True, progress_enabled=False)
    real(host, client, "C1", None, "Searching the web…", thread_id="10.0", turn=deferred)
    host._schedule_async_call.assert_not_called()

    # ...but a genuine status-only turn (a DM) still routes to the composer status.
    allowed = TurnRuntime(silence_capable=False, progress_enabled=True)
    real(host, client, "D1", None, "Searching the web…", thread_id="10.0", turn=allowed)
    host._schedule_async_call.assert_called_once()


# --------------------------------------------------------------- the compaction box

def test_the_context_usage_box_is_gone():
    """It posted a public ASCII box of token counts and "tips" into the thread, over a thing
    the user never asked about and cannot act on. Compaction is behind-the-scenes work — the
    bot has no business narrating it, and the state it needed to is gone with it."""
    import inspect
    from message_processor.base import MessageProcessor
    src = inspect.getsource(MessageProcessor.process_message)
    # The box was assembled inline and sent with client.send_message. Neither the flag that
    # tracked it nor the banner text survives.
    assert "has_shown_80_percent_warning" not in src
    assert "Tips for optimal performance" not in src
    assert "of available context" not in src

    from thread_manager import ThreadState
    assert not hasattr(ThreadState(thread_ts="1.0", channel_id="C1"),
                       "has_shown_80_percent_warning")


@pytest.mark.asyncio
async def test_a_deferred_non_streaming_turn_sets_no_status(monkeypatch):
    """The leak codex caught, and the nastiest one: it recreates the ORIGINAL bug.

    With streaming off, `_handle_text_response` runs its own pre-generation status updates.
    Those call `_update_status` with `thinking_id=None` (deferred, so no placeholder was
    posted) — and without the turn, that signature reads as "status-only surface" and routes
    to `set_assistant_status`, which renders a thinking status AND auto-opens the thread. The
    turn may then say nothing at all. Exactly the flash we set out to remove.
    """
    from message_processor.handlers.text import TextHandlerMixin
    import types

    calls = []

    host = MagicMock()
    host._update_status = MagicMock(side_effect=lambda *a, **kw: calls.append((a, kw)))
    host._inject_image_analyses = AsyncMock(side_effect=lambda m, _ts: m)
    host._pre_trim_messages_for_api = AsyncMock(side_effect=lambda m, **kw: m)
    host._build_channel_info = AsyncMock(return_value=None)
    host._materialize_request_tools = MagicMock(return_value=(None, {}, False, None))
    host.log_debug = host.log_info = host.log_warning = host.log_error = MagicMock()

    client = MagicMock()
    client.supports_streaming = MagicMock(return_value=False)   # streaming OFF
    msg = _message(participation_check=True)

    async def fake_config(**kw):
        return {"enable_streaming": False, "model": "gpt-5.6-sol"}

    thread_state = MagicMock()
    thread_state.messages = []
    deferred = TurnRuntime(silence_capable=True, progress_enabled=False)

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        bound = types.MethodType(TextHandlerMixin._handle_text_response, host)
        try:
            await bound("hello", thread_state, client, msg, thinking_id=None, turn=deferred)
        except Exception:
            pass   # the non-streaming path runs into the mocks; we only care how far it got

    assert calls, "expected the non-streaming path to attempt a status update"
    for _args, kwargs in calls:
        assert kwargs.get("turn") is deferred, (
            "a pre-generation status update did not carry the turn — on a deferred turn it "
            "will fall through to set_assistant_status and re-open the thread")


# --------------------------------------------------------------- reply placement

def _creation_calls(func):
    """Every call in `func` that MINTS a new Slack message, with the argument it uses as the
    thread target."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("send_message", "send_message_get_ts"):
            continue
        if len(node.args) >= 2:
            out.append(ast.unparse(node.args[1]))
    return out


def test_every_streaming_message_creation_uses_the_reply_target():
    """Codex's #1 predicted failure: forgetting one of the secondary creation sites.

    The lazy seed, the mid-stream overflow parts and the zero-chunk final post each mint a
    message. They used to target `message.thread_id` — the thread the TRIGGER lives in — and
    got away with it only because a placeholder already existed in the right place. Take the
    placeholder away and a top-level channel reply lands inside a thread instead. Every one of
    them must now use the turn's chosen destination.

    F46: the zero-chunk final post now targets the RESOLVED destination — `effective_target =
    turn.resolve_reply_target(message)`, i.e. `reply_target` refined by the substantive-work
    thread override (a top-level channel reply that did real work threads under the trigger).
    The seed / overflow parts still use `reply_target` directly. What must NEVER come back is
    `message.thread_id` (the thread the trigger merely lives in).
    """
    import inspect
    from message_processor.handlers.text import TextHandlerMixin

    targets = _creation_calls(TextHandlerMixin._handle_streaming_text_response)
    assert targets, "expected the streaming handler to create messages"
    assert all(t in ("reply_target", "effective_target") for t in targets), (
        f"a streaming message-creation site targets neither reply_target nor the resolved "
        f"effective_target (a regression to message.thread_id?): {targets}")
    assert "effective_target" in targets, (
        "the final-post site must use the resolved effective_target (F46 late thread override)")
    src = inspect.getsource(TextHandlerMixin._handle_streaming_text_response)
    assert "effective_target = " in src and "resolve_reply_target(message)" in src, (
        "effective_target must be bound to the turn's resolved reply destination")


def test_the_reply_target_is_the_turns_chosen_destination(monkeypatch):
    """None means top-level in the channel — not "the thread the trigger was in"."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    top_level = TurnRuntime.for_message(
        _message(participation_check=True), None)   # main.py chose channel placement
    assert top_level.reply_thread_id is None

    threaded = TurnRuntime.for_message(
        _message(participation_check=True), "10.0")
    assert threaded.reply_thread_id == "10.0"


# --------------------------------------------------------------- retry ownership

def test_the_mcp_retry_inherits_the_lazily_created_message():
    """Codex's #2 predicted failure: losing lazy-surface ownership across a recursive retry.

    An MCP failure keeps streaming and re-enters the handler. If the retry doesn't know about
    the message this attempt already created, it sees "no placeholder", seeds a SECOND one,
    and the turn posts its answer twice — which is exactly the class of bug the user hit with
    the duplicated status cards."""
    import ast
    import inspect
    import textwrap
    from message_processor.handlers.text import TextHandlerMixin

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(TextHandlerMixin._handle_streaming_text_response)))
    retries = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_handle_text_response"
    ]
    assert retries, "expected the streaming handler to have a retry path"
    for call in retries:
        kwargs = {k.arg for k in call.keywords}
        assert "lazy_surface_ts" in kwargs, (
            "a retry from the streaming handler does not carry the lazily-created message "
            "forward — it will seed a second one and answer twice")


def test_the_model_is_still_told_about_compaction():
    """The user is not told; the MODEL still is. That distinction is the whole point — the
    fact belongs in the system prompt, not in the channel."""
    import inspect
    from message_processor.utilities import MessageUtilitiesMixin
    src = inspect.getsource(MessageUtilitiesMixin._get_system_prompt)
    assert "has_trimmed_messages" in src


# --------------------------------------------------------------- the prior-timeout notice

@pytest.mark.asyncio
async def _timeout_notice_shown_for(turn, monkeypatch):
    """Drive the real process_message with a thread that previously timed out; report whether
    the recovery notice was posted."""
    from base_client import Response
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()

    state = SimpleNamespace(had_timeout=True, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={})
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()

    async def _state(*a, **k):
        return state

    p._get_or_rebuild_thread_state = _state
    # Stop the turn the moment the notice decision is behind us.
    p._handle_text_response = AsyncMock(return_value=Response(type="text", content="ok"))
    p._build_channel_memory_text = AsyncMock(return_value="")
    p._build_channel_info = AsyncMock(return_value="")
    p._process_attachments = AsyncMock(return_value=([], [], []))

    client = MagicMock()
    client.send_message = AsyncMock()

    try:
        await p.process_message(_message(participation_check=True), client, None, turn=turn)
    except Exception:
        pass   # anything downstream of the notice is not this test's business

    posted = [c for c in client.send_message.await_args_list
              if "never finished" in str(c)]
    return bool(posted), state.had_timeout


@pytest.mark.asyncio
async def test_prior_timeout_notice_is_held_back_on_a_silent_turn(monkeypatch):
    """It posts BEFORE the model decides, so on a turn that may say nothing it would announce
    a dead answer and then fall silent. The flag still clears — a stale one would make a later
    prompted turn describe an old failure as "my last answer"."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime(silence_capable=True, progress_enabled=False, reply_thread_id="10.0")

    shown, still_flagged = await _timeout_notice_shown_for(turn, monkeypatch)

    assert not shown, "a turn that may say nothing must not announce a dead answer first"
    assert not still_flagged, "the flag must clear either way, or a later turn re-announces it"


@pytest.mark.asyncio
async def test_prior_timeout_notice_still_shows_on_a_top_level_reply(monkeypatch):
    """F39: a top-level channel reply sets progress_enabled False — it may write NOTHING before
    its finished answer, or Slack brands the answer "(edited)". Keying this notice on that flag
    swallowed it on turns that were always going to answer. It is not speculative chrome: it is
    a standalone post, never edited into anything. Key it on SILENCE, not on progress."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime(silence_capable=False, progress_enabled=False,
                       reply_thread_id=None, final_post_only=True)

    shown, still_flagged = await _timeout_notice_shown_for(turn, monkeypatch)

    assert shown, ("an addressed top-level turn always answers — it must still tell the user "
                   "its last answer died")
    assert not still_flagged
