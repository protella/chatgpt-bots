"""F39 — how many messages does a turn leave behind, and does Slack call any of them "(edited)"?

Every previous test of `_handle_streaming_text_response` asserted on `inspect.getsource(...)`
strings. Those tests passed while the handler shipped a duplicate-reply bug for months: one of
them (`test_mcp_streaming_retry_keeps_its_partial`) asserted the BUG as if it were the contract.
A grep cannot tell you how many messages are on screen when the turn ends.

So this file drives the real handler against a fake Slack that tracks LIVE messages — created
minus deleted — and asserts the two things a reader of the channel actually experiences:

    1. exactly ONE message survives a turn, however many attempts it took, and
    2. a top-level channel reply is POSTED ONCE, never edited into existence.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config import config
from message_processor.base import MessageProcessor
from message_processor.turn_runtime import TurnRuntime
from streaming import NativeStreamCoordinator  # noqa: F401  (imported by the handler)


# --------------------------------------------------------------------- fake Slack

class FakeNativeSession:
    """Mirrors NativeStreamSession's contract, including Slack's hard requirement that
    chat.startStream have a thread_ts (docs.slack.dev: `thread_ts` is a REQUIRED argument)."""

    def __init__(self, slack, channel, thread):
        self.slack, self.channel, self.thread = slack, channel, thread
        self.ts = None
        self.active = False
        self._sent = ""

    async def start(self, initial_text: str = "") -> bool:
        if not self.thread or self.slack.native_start_fails:
            self.active = False
            return False          # top-level (or a missing scope): Slack refuses to stream
        self.ts = self.slack._mint("native")
        self.slack.calls.append(("startStream", self.ts))
        self._sent = initial_text or ""
        self.active = True
        return True

    async def update(self, cumulative: str) -> bool:
        if not self.active or self.slack.refuse_append:
            return False        # every mid-stream append fails: nothing is "delivered" yet
        delta = cumulative[len(self._sent):] if cumulative.startswith(self._sent) else cumulative
        if delta:
            self.slack.calls.append(("appendStream", self.ts))
            self.slack._write(self.ts, cumulative)
            self._sent = cumulative
        return True

    async def finish(self, final_text=None, blocks=None) -> bool:
        self.slack.calls.append(("stopStream", self.ts))
        if final_text is not None:
            # stopStream appends its tail; the message shows everything sent so far.
            self.slack._write(self.ts, self._sent + final_text
                              if not final_text.startswith(self._sent) else final_text)
        self.active = False
        return True


class FakeSlack:
    """Records every call and, crucially, tracks which messages are still LIVE."""

    def __init__(self, native: bool = False, refuse_delete: bool = False,
                 refuse_post: bool = False, native_start_fails: bool = False,
                 refuse_append: bool = False):
        self.calls = []              # ordered (verb, ts) log
        self.live = {}               # ts -> text, only messages currently on screen
        self.text = {}               # ts -> latest text ever written (live or not)
        self.history = []            # every (ts, text) ever WRITTEN — `text` keeps only the last
        self.native = native
        self.refuse_delete = refuse_delete
        self.refuse_post = refuse_post
        self.native_start_fails = native_start_fails
        self.refuse_append = refuse_append
        self._n = 0

    def _mint(self, kind: str) -> str:
        self._n += 1
        ts = f"{kind}-{self._n}"
        self.live[ts] = ""
        return ts

    def _write(self, ts: str, text: str) -> None:
        """Whatever is on screen for `ts` right now — but only while it IS on screen."""
        self.text[ts] = text
        self.history.append((ts, text))
        if ts in self.live:
            self.live[ts] = text

    # -- capabilities
    def supports_streaming(self):
        return True

    def get_streaming_config(self):
        return {"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0}

    def supports_native_streaming(self):
        return self.native

    def begin_native_stream(self, channel, thread, user_id=None):
        return FakeNativeSession(self, channel, thread)

    # -- writes
    async def send_message_get_ts(self, channel, thread, text):
        if self.refuse_post:
            return {"success": False}
        ts = self._mint("seed")
        self.calls.append(("postMessage", ts))
        self._write(ts, text)
        return {"success": True, "ts": ts}

    MAX_MESSAGE_LENGTH = 3900

    async def send_message(self, channel, thread, text, blocks=None, meta_out=None,
                           username=None):
        """Splits like the real client, and returns the FIRST chunk's ts (or None on failure —
        the real one swallows SlackApiError, which is exactly the silent-loss hole)."""
        if self.refuse_post:
            return None
        first = None
        for i in range(0, len(text), self.MAX_MESSAGE_LENGTH):
            ts = self._mint("post")
            self.calls.append(("postMessage", ts))
            self._write(ts, text[i:i + self.MAX_MESSAGE_LENGTH])
            first = first or ts
        return first

    async def update_message(self, channel, ts, text):
        if ts not in self.live:
            return False
        self.calls.append(("update", ts))
        self._write(ts, text)
        return True

    async def update_message_streaming(self, channel, ts, text):
        if ts not in self.live:
            return {"success": False, "error": "message_not_found"}
        self.calls.append(("update", ts))
        self._write(ts, text)
        return {"success": True}

    async def delete_message(self, channel, ts):
        if self.refuse_delete:
            return False
        if ts not in self.live:
            return False
        self.calls.append(("delete", ts))
        del self.live[ts]
        return True

    # -- misc surface the handler pokes
    def format_text(self, t):
        return t

    async def set_assistant_status(self, channel, thread, status=""):
        self.calls.append(("setStatus", status))

    def _record_own_reply_pulse(self, *a, **k):
        pass

    # -- assertions
    @property
    def edits(self):
        return [ts for verb, ts in self.calls if verb == "update"]

    @property
    def posts(self):
        return [ts for verb, ts in self.calls if verb == "postMessage"]

    @property
    def streams(self):
        return [ts for verb, ts in self.calls if verb == "startStream"]


class FakeOpenAI:
    """Streams `chunks`, then either raises or completes. `raises_once` fails only the first
    attempt, so a retry can succeed — exactly the MCP-failover shape."""

    def __init__(self, chunks, error=None, raises_once=True, retry_chunks=None):
        self.chunks = chunks
        self.error = error
        self.raises_once = raises_once
        self.retry_chunks = retry_chunks if retry_chunks is not None else chunks
        self.attempts = 0

    async def _run(self, stream_callback, **kw):
        self.attempts += 1
        first = self.attempts == 1
        chunks = self.chunks if first else self.retry_chunks
        for c in chunks:
            await stream_callback(c)
        if self.error is not None and (first or not self.raises_once):
            raise self.error
        await stream_callback(None)     # the client emits the completion signal
        return "".join(chunks)

    async def create_streaming_response(self, messages=None, stream_callback=None, **kw):
        return await self._run(stream_callback, **kw)

    # the non-streaming fallback (used when the failure is NOT an MCP one)
    async def create_text_response(self, **kw):
        self.attempts += 1
        return "".join(self.retry_chunks)

    async def _create_text_response_with_timeout(self, **kw):
        return await self.create_text_response(**kw)


# --------------------------------------------------------------------- harness

def _message(channel="C1", thread="10.0", **meta):
    return SimpleNamespace(
        channel_id=channel, thread_id=thread, user_id="U1", text="hi",
        attachments=None, metadata={"ts": "10.0", **meta},
    )


def _thread_state(channel="C1", thread="10.0"):
    return SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        channel_id=channel, thread_ts=thread, current_model="gpt-5.6-sol",
        config_overrides={}, has_summary_head=False, channel_directives=None,
        record_usage=MagicMock(), last_usage=None,
    )


def _processor(openai):
    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.openai_client = openai
    p.db = None

    async def _passthru(m, *a, **k):
        return m

    p._add_message_with_token_management = MagicMock()
    p._inject_image_analyses = _passthru
    p._pre_trim_messages_for_api = _passthru
    p._get_system_prompt = MagicMock(return_value="sys")
    p._build_participant_roster = MagicMock(return_value="")
    p._build_suffix_context = MagicMock(return_value="")
    p._build_tools_array = MagicMock(return_value=[])          # no tools -> plain stream
    p._materialize_request_tools = MagicMock(return_value=(None, {}, False, ""))
    p._persist_tool_provenance = MagicMock()
    p._schedule_async_call = MagicMock()

    async def _none(*a, **k):
        return None

    async def _empty_str(*a, **k):
        return ""

    p._build_channel_info = _empty_str
    p._async_post_response_cleanup = _none
    p._drop_dead_containers = _none
    p._resolve_ci_container = _none
    return p


async def _run(processor, slack, message, thread_state, turn, thinking_id=None):
    return await processor._handle_streaming_text_response(
        "hi", thread_state, slack, message, thinking_id, None, turn=turn)


def _surfaces_on_screen(slack, resp) -> int:
    """What the READER ends up seeing — which is not the same as what this handler posted.

    A non-streaming fallback does NOT post its own answer: it hands the text back and main.py
    posts it (`elif not response.metadata.get("streamed")`). So counting only the handler's own
    messages hides exactly the duplicate we are hunting: an undeleted partial sitting next to
    the answer main.py is about to send. Count both.
    """
    meta = resp.metadata or {}
    pending = 1 if (not meta.get("streamed") and (resp.content or "").strip()) else 0
    return len(slack.live) + pending


MCP_ERROR = Exception("Error calling MCP server: 'datassential' failed")


# ============================================================ 1. one surface survives

@pytest.mark.asyncio
async def test_native_mcp_retry_leaves_exactly_one_message(monkeypatch):
    """THE BUG. Native streaming mints its reply with chat.startStream. An MCP failure retried
    WITH streaming, but the retry never learned about the message the first attempt had already
    created — so it minted a second one, while the abandoned partial (holding answer text) was
    deliberately not deleted. Two messages, same answer. The "42 / 42" duplicate, again."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    openai = FakeOpenAI(["The answer ", "is 42."], error=MCP_ERROR)
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")          # threaded -> native streams

    resp = await _run(processor, slack, msg, ts, turn)

    assert openai.attempts == 2, "the MCP failure should have retried"
    assert _surfaces_on_screen(slack, resp) == 1, (
        f"the turn left {_surfaces_on_screen(slack, resp)} messages on screen: {slack.live}")
    assert "42" in list(slack.live.values())[0]


@pytest.mark.asyncio
async def test_native_stands_down_when_it_cannot_clear_the_old_surface(monkeypatch):
    """chat.startStream MINTS a message, so the old surface must be gone BEFORE it runs. If the
    delete fails and we start anyway, the turn owns two live messages. Rather than that, native
    stands down and keeps streaming into the surface we already have."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True, refuse_delete=True)
    openai = FakeOpenAI(["hello"])
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")
    placeholder = slack._mint("placeholder")

    resp = await _run(processor, slack, msg, ts, turn, thinking_id=placeholder)

    assert not slack.streams, "native must not mint a message it cannot pair with a delete"
    assert _surfaces_on_screen(slack, resp) == 1 and placeholder in slack.live
    assert "hello" in slack.live[placeholder]


@pytest.mark.asyncio
async def test_a_multi_part_native_answer_is_fully_reconciled(monkeypatch):
    """A rolled stream owns N messages, not one. The old cleanup deleted only `current_ts`, so
    parts 1..N-1 survived holding answer text and the retry duplicated them."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    # Long enough to roll past the ~3060-char part limit, then fail.
    openai = FakeOpenAI(["x" * 2000, "y" * 2000], error=MCP_ERROR, retry_chunks=["done"])
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    resp = await _run(processor, slack, msg, ts, turn)

    assert len(slack.streams) >= 2, "expected the first attempt to roll into a second part"
    assert _surfaces_on_screen(slack, resp) == 1, (
        f"an abandoned multi-part stream left messages behind: {slack.live}")


@pytest.mark.asyncio
async def test_a_legacy_seed_after_a_failed_native_start_is_still_reconciled(monkeypatch):
    """`NativeStreamCoordinator.started` is `session is not None`, and the session is assigned
    BEFORE start() is awaited — so a FAILED chat.startStream still reports started=True with an
    empty part_ts. Read the owned-surface ledger as "native parts, ELSE the legacy seed" and
    this turn owns nothing: the seed the legacy loop then created is never reconciled, survives
    the fallback, and the answer posts twice. The ledger must be a UNION, not a choice."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True, native_start_fails=True)
    openai = FakeOpenAI(["partial answer"], error=ValueError("boom"))   # non-MCP -> fallback
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    resp = await _run(processor, slack, msg, ts, turn)

    assert not slack.streams, "startStream was supposed to fail"
    assert _surfaces_on_screen(slack, resp) == 1, (
        f"the legacy seed survived the fallback — the answer lands on screen twice "
        f"(live={slack.live}, handing back={resp.content!r})")


@pytest.mark.asyncio
async def test_when_deletes_fail_no_surface_is_left_holding_a_half_answer(monkeypatch):
    """Fail-closed used to rewrite only survivors[0]. With a multi-part stream whose deletes all
    fail, that left the rest still showing partial ANSWER text (and a reset keeper still
    promising a retry that was never coming). We can't delete them — deleting is what failed —
    but we can make sure not one of them still reads like an answer."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True, refuse_delete=True)
    openai = FakeOpenAI(["x" * 2000, "y" * 2000], error=MCP_ERROR, retry_chunks=["done"])
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    resp = await _run(processor, slack, msg, ts, turn)

    assert resp.metadata.get("interrupted") is True, "it must not post a second answer"
    for ts_, text in slack.live.items():
        assert "xxxx" not in text and "yyyy" not in text, (
            f"{ts_} is still showing a partial answer: {text[:60]!r}")
        assert "Retrying without" not in text, (
            f"{ts_} still promises a retry that is not coming: {text[:60]!r}")


@pytest.mark.asyncio
async def test_a_long_top_level_answer_is_split_not_truncated(monkeypatch):
    """The final-post-only path returns before the streaming overflow logic entirely, so a long
    answer's only splitter is the one inside send_message. Prove the handler actually routes
    through it rather than handing Slack a 6k-char message to truncate."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True)
    long_answer = "para. " * 1200                      # ~7200 chars, well past Slack's limit
    processor = _processor(FakeOpenAI([long_answer]))
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, None)

    resp = await _run(processor, slack, msg, ts, turn)

    assert slack.edits == [], "still no edits, however long the answer"
    assert resp.metadata["posted"] is True
    delivered = "".join(slack.live.values())
    assert len(delivered) >= len(long_answer) * 0.9, (
        f"the answer was truncated: sent {len(long_answer)} chars, "
        f"{len(delivered)} reached Slack across {len(slack.live)} message(s)")


@pytest.mark.asyncio
async def test_the_cleanup_edit_counts_as_a_delivery(monkeypatch):
    """The subtlest one. If every mid-stream append fails, `visible_content_delivered` is still
    False — yet the error handler's "strip the loading indicator" edit then writes the buffered
    PARTIAL ANSWER onto the surface. That edit is the first successful delivery, and nothing
    recorded it. So a failed delete produced a survivor holding half an answer, the fail-closed
    guard (which asks that flag) declined to fire, and the fallback posted the answer beside it.
    """
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True, refuse_append=True, refuse_delete=True)
    openai = FakeOpenAI(["the partial answer"], error=ValueError("boom"))   # -> fallback
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    resp = await _run(processor, slack, msg, ts, turn)

    assert any("partial answer" in t for _, t in slack.history), (
        "precondition: the cleanup edit should have written the partial answer to Slack")
    assert _surfaces_on_screen(slack, resp) == 1, (
        f"the answer is on screen twice — the partial the cleanup edit delivered, plus the "
        f"fallback's full answer (live={slack.live})")
    assert not any("partial answer" in t for t in slack.live.values()), (
        "the partial answer is still on screen")


# ============================================================ 2. no "(edited)" at top level

@pytest.mark.asyncio
async def test_a_top_level_channel_reply_is_posted_once_never_edited(monkeypatch):
    """The "(edited)" bug. Slack can only stream into a THREAD, so a top-level channel reply
    fell to the legacy loop: post a stub, then chat.update it as the text arrives. Every one of
    those edits brands the message "(edited)" forever — which is why ours carried the marker and
    Claude's, posted once and finished, did not."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True)
    openai = FakeOpenAI(["Postgres defaults ", "to READ COMMITTED."])
    processor = _processor(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, None)            # main.py chose top-level placement
    assert turn.final_post_only

    resp = await _run(processor, slack, msg, ts, turn)

    assert slack.edits == [], f"a top-level reply was edited into existence: {slack.calls}"
    assert slack.streams == [], "Slack cannot stream top-level; we must not have tried"
    assert len(slack.posts) == 1, f"expected exactly one post, got {slack.calls}"
    assert "READ COMMITTED" in list(slack.live.values())[0]
    assert resp.metadata["posted"] is True


@pytest.mark.asyncio
async def test_a_top_level_turn_shows_no_status_and_no_placeholder(monkeypatch):
    """Nothing may be conjured mid-turn — a composer status would render a thinking line AND
    auto-open a thread under a message we are about to answer at the top level."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True)
    processor = _processor(FakeOpenAI(["hi"]))
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, None)

    await _run(processor, slack, msg, ts, turn)

    assert not turn.progress_enabled
    assert [c for c in slack.calls if c[0] == "setStatus"] == []


@pytest.mark.asyncio
async def test_a_threaded_reply_still_streams(monkeypatch):
    """Scope check: the fix must not cost threads their live reveal — they CAN stream natively,
    so they keep doing it, and native streaming never marks a message edited."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    slack = FakeSlack(native=True)
    processor = _processor(FakeOpenAI(["still ", "streaming"]))
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")
    assert not turn.final_post_only

    await _run(processor, slack, msg, ts, turn)

    assert len(slack.streams) == 1
    assert slack.edits == []


@pytest.mark.asyncio
async def test_a_failed_final_post_hands_the_answer_back_instead_of_dropping_it(monkeypatch):
    """The one-shot post is now the ONLY delivery for these turns, and send_message swallows
    SlackApiError and returns None. If we still claimed `streamed`, main.py would never re-post
    it and the answer would vanish silently."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True, refuse_post=True)
    processor = _processor(FakeOpenAI(["the answer"]))
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, None)

    resp = await _run(processor, slack, msg, ts, turn)

    assert not slack.live, "nothing should have reached Slack"
    assert resp.metadata.get("streamed") is False, (
        "a turn that delivered nothing must not claim it streamed — main.py would drop the text")
    assert "posted" not in resp.metadata, (
        "leave `posted` unset so main.py derives it from the send it is about to do; an explicit "
        "False would also retract the 👀 from a turn that does end up answering")
    assert "the answer" in resp.content
    assert "tool_provenance" in resp.metadata, (
        "main.py persists F7 provenance FROM THE METADATA after its own send — we never got a "
        "ts to persist against ourselves. Without it the rescued answer keeps its text but "
        "loses every tool attribution")


# ============================================================ 3. the placement rule

def test_final_post_only_is_exactly_top_level_in_a_channel(monkeypatch):
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)

    assert TurnRuntime.for_message(_message(channel="C1"), None).final_post_only
    assert not TurnRuntime.for_message(_message(channel="C1"), "10.0").final_post_only, \
        "a threaded reply can stream natively — leave it alone"
    assert not TurnRuntime.for_message(_message(channel="D1"), None).final_post_only, \
        "a DM is a conversation, not a public channel; it keeps its live reveal"


# ============================================================ 4. the pre-tool preamble flush

class FakeToolLoopOpenAI:
    """Drives the streaming FUNCTION-CALL loop the way the real wrapper does.

    Round 1 streams a preamble and then — because the round produced a function call — BREAKS
    WITHOUT sending the None completion signal (responses.py suppresses it on a tool round). The
    tool dispatches; round 2 streams the post-tool text and ends with the real None. `snapshot`
    records what was on the native surface at the instant the tool started — the whole point of
    the fix is that a finished preamble is on screen BEFORE a ~minute-long edit blocks the loop,
    instead of frozen at its first word until the tool returns.
    """

    def __init__(self, slack, preamble, post_tool, tool="local:edit_image"):
        self.slack, self.preamble, self.post_tool, self.tool = slack, preamble, post_tool, tool
        self.snapshot = None
        self.attempts = 0

    async def create_streaming_response_with_tool_loop(self, stream_callback=None,
                                                       tool_callback=None, aggregate_segments=False,
                                                       **kw):
        self.attempts += 1
        for c in self.preamble:
            await stream_callback(c)
        # saw_function_call -> the wrapper does NOT signal completion here (responses.py:930).
        await tool_callback(self.tool, "started")
        self.snapshot = dict(self.slack.text)          # what the reader sees mid-edit
        await tool_callback(self.tool, "completed")
        for c in self.post_tool:
            await stream_callback(c)
        await stream_callback(None)                    # final round: the real completion signal
        # Honour aggregate_segments exactly like the real loop: chat opts in and gets the
        # seam-joined whole (matching what streamed to Slack); everything else gets last-round.
        from message_markers import join_segments
        pre, post = "".join(self.preamble), "".join(self.post_tool)
        text = join_segments([pre, post]) if aggregate_segments else post
        return {"text": text,
                "local_tool_calls": [{"name": self.tool.split(":", 1)[-1], "ok": True}],
                "tools_used": [], "terminal_action": None, "reason": None}


def _processor_tools(openai):
    """A processor whose streaming turn takes the LOCAL-TOOL loop path (tools + registry)."""
    p = _processor(openai)
    registry = MagicMock()
    p._materialize_request_tools = MagicMock(return_value=(registry, {}, False, ""))
    p._build_tools_array = MagicMock(return_value=[{"type": "function", "name": "edit_image"}])
    p._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[]))

    async def _noop(*a, **k):
        return None

    p._prepare_sandbox_tools = _noop
    return p


def _held_streaming_config():
    """A cadence that never fires a time-based append (update_interval far in the future) but
    keeps the rate-limiter circuit closed — so round-1 text stays BUFFERED-BUT-UNFLUSHED until
    something forces it. This is the fast-preamble-then-blocking-tool shape, deterministically."""
    return {"update_interval": 100.0, "buffer_size": 100000, "min_interval": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["local:edit_image", "local:create_image_asset"])
async def test_a_blocking_tools_preamble_reaches_slack_before_it_runs(monkeypatch, tool):
    """THE BUG. Native appends are token-driven, and the wrapper skips the None signal on a tool
    round — so a round's preamble sat frozen at whatever the cadence last flushed (one word:
    "Yep") for the entire ~minute the tool ran, and only appeared once the NEXT round streamed.
    Every blocking synchronous tool must find the preamble already on screen when it dispatches —
    parametrized so dropping either member of _PRE_TOOL_FLUSH_TOOLS fails here."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    from message_processor import file_mount
    monkeypatch.setattr(file_mount, "mounted_digests", lambda tc: [], raising=False)
    slack = FakeSlack(native=True)
    slack.get_streaming_config = _held_streaming_config
    preamble = ["Yep — fixing ", "the chopsticks ", "under Super Heavy."]
    openai = FakeToolLoopOpenAI(slack, preamble, ["Fixed. Landed clean."], tool=tool)
    processor = _processor_tools(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")            # threaded -> native streams

    await _run(processor, slack, msg, ts, turn)

    mid_edit = "".join((openai.snapshot or {}).values())
    assert "Yep — fixing the chopsticks under Super Heavy." in mid_edit, (
        f"the preamble was still frozen when {tool} dispatched — the reader watches a "
        f"half-written sentence for the whole run: {mid_edit!r}")


@pytest.mark.asyncio
async def test_a_detached_or_background_tools_ack_is_not_flushed_early(monkeypatch):
    """Scoping guard. The flush is ONLY for tools that block the loop. `start_background_job`
    returns fast and DELIBERATELY withholds its short ack so the job's live status card owns the
    acknowledgment (F30.1) — that suppression keys on `visible_content_delivered` still being
    False. Force-flushing its preamble at dispatch would set the flag and strand both the ack and
    the card, so a non-blocking tool must NOT trigger the early flush."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    from message_processor import file_mount
    monkeypatch.setattr(file_mount, "mounted_digests", lambda tc: [], raising=False)
    slack = FakeSlack(native=True)
    slack.get_streaming_config = _held_streaming_config
    openai = FakeToolLoopOpenAI(slack, ["On it — kicking ", "off that build now."],
                                ["Done."], tool="local:start_background_job")
    processor = _processor_tools(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    await _run(processor, slack, msg, ts, turn)

    # The buffer always flushes its FIRST chunk (last_update_time seeds at 0), so the head is on
    # screen either way — the discriminator is the buffered TAIL, which only a force-flush pushes.
    mid = "".join((openai.snapshot or {}).values())
    assert "On it" in mid, "precondition: the head flushed on normal cadence"
    assert "off that build now" not in mid, (
        f"a non-blocking tool's buffered ack tail was force-flushed before dispatch — this is "
        f"exactly what breaks background-job ack suppression: {mid!r}")


@pytest.mark.asyncio
async def test_the_seam_between_a_preamble_and_the_post_tool_text_is_not_jammed(monkeypatch):
    """"…under Super Heavy." + "Fixed." must not render as "Super Heavy.Fixed." on screen. A
    local-tool round ends a text segment, so the next round's first words earn a paragraph seam —
    the same rule the tool loop uses to join its canonical aggregate, so display and memory match."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    from message_processor import file_mount
    monkeypatch.setattr(file_mount, "mounted_digests", lambda tc: [], raising=False)
    slack = FakeSlack(native=True)
    openai = FakeToolLoopOpenAI(slack, ["Fixing the chopsticks under Super Heavy."], ["Fixed."])
    processor = _processor_tools(openai)
    msg, ts = _message(), _thread_state()
    turn = TurnRuntime.for_message(msg, "10.0")

    await _run(processor, slack, msg, ts, turn)

    final = "".join(slack.live.values())
    assert "Super Heavy.\n\nFixed." in final, f"the seam is jammed: {final!r}"
    assert "Super Heavy.Fixed" not in final


def test_the_source_no_longer_claims_an_mcp_retry_keeps_its_partial():
    """The retired grep-test asserted `not failed_mcp_server` as the guard that SKIPPED deleting
    the native partial on an MCP retry. That guard WAS the bug. If it ever comes back, the
    behavioural tests above are the ones that must fail — this only pins the regression."""
    import inspect
    from message_processor.handlers.text import TextHandlerMixin
    src = inspect.getsource(TextHandlerMixin._handle_streaming_text_response)
    assert "and native_coord.current_ts and not failed_mcp_server" not in src
    assert re.search(r"owned\s*:\s*List\[str\]", src), "expected the owned-surface ledger"
