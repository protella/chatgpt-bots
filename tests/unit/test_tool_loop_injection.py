"""`pre_round_input_callback` — the seam a running job's mid-run updates come in through.

The parameter is generic and the tool loop knows nothing about jobs, but the guarantees it holds
to are all shaped by that caller: an update the conversation sent must reach the model exactly
once, on a complete input, without a callback of the caller's ever being able to fail a round or
outlive a cancellation. Everything below pins one of those.

All stubbed I/O: the streaming call is scripted, no live OpenAI. The container-recovery case is
the exception in one respect — it drives the real `_create_with_container_recovery` and fakes
only the response parsing, because "once per round, not once per HTTP attempt" is a claim about
production's retry and a test that loops twice by itself would prove nothing about it.

What this file deliberately does NOT establish: that the provider ACCEPTS developer/user items
appended after a replayed tool round. That is a wire fact, and the spec designates the round's
live pass (a steered live job) as its evidence — a mock that accepts whatever it is handed can
only ever agree with us.
"""
import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client.api import tool_loop
from message_processor.tool_registry import ToolContext, ToolRegistry


# --------------------------------------------------------------------------- helpers

class _Client:
    """Minimal OpenAIClient stand-in — the loop functions only use log_*."""
    def __init__(self):
        self.warnings = []

    def log_info(self, *a, **k): pass

    def log_debug(self, *a, **k): pass

    def log_warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def log_error(self, *a, **k): pass


async def _ok(args):
    return {"ok": True, "echo": args}


def _registry(*names):
    reg = ToolRegistry()
    for name in names or ("echo",):
        reg.register({"type": "function", "name": name, "description": "t",
                      "parameters": {"type": "object"}},
                     lambda ctx, args: _ok(args))
    return reg


def _call(name="echo", call_id="c1", arguments='{"x": 1}'):
    return {"call_id": call_id, "name": name, "arguments": arguments}


def _note(text="drop the pricing section"):
    """The dev+user pair the job-side callback builds, in miniature."""
    return [{"role": "developer", "content": "A mid-run update follows as the next user message."},
            {"role": "user", "content": text}]


class _ScriptedStream:
    """Scripted `create_streaming_response_with_tools`: one entry per round.

    Snapshots the input list it was handed at the moment of the call, so a test can ask what the
    model would actually have seen on that round rather than only what the loop returned.
    """
    def __init__(self, rounds):
        self.rounds = rounds            # [(text, [calls]), ...]
        self.snapshots = []             # one deep copy per round, in order

    async def __call__(self, client, messages, tools, stream_callback, tool_callback=None,
                       function_call_sink=None, tool_choice=None, **params):
        text, calls = self.rounds[min(len(self.snapshots), len(self.rounds) - 1)]
        self.snapshots.append(copy.deepcopy(list(messages)))
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(copy.deepcopy(calls))
        return text


_DEAD = "cntr_dead"
_RECOVERY = "cntr_recovery"


def _explicit_tools(container_id):
    return [{"type": "function", "name": "echo", "parameters": {}},
            {"type": "code_interpreter", "container": container_id}]


def _ci_container(tools):
    return next(t["container"] for t in tools if t["type"] == "code_interpreter")


def _holder_ctx(container_id=None, gone_sink=None):
    from message_processor.tool_registry import SandboxHolder
    ctx = ToolContext(sandbox=SandboxHolder(
        container_id=container_id, manager=MagicMock(adopt=AsyncMock()), thread_key="C1:1.1"))
    ctx.container_gone_sink = gone_sink if gone_sink is not None else []
    return ctx


def _gone(container_id):
    exc = Exception(f"Container with id '{container_id}' not found.")
    exc.status_code = 404  # what is_container_gone reads
    return exc


class _RecoveringStream:
    """Drives the REAL `_create_with_container_recovery` for each round.

    Only the response parsing is faked. The first wire request names the thread's explicit
    container and 404s exactly the way an idle-expired sandbox does, so the demote, the retry,
    the `container_gone_sink` entry and the adoption veto are all produced by production code
    rather than asserted into existence here. Modelled on the harness in `test_tool_loop.py`;
    kept local because that file is not mine to import from.

    Records, per WIRE request: the container declared (`sent`), the input the request carried
    (`attempts`), and how many times the injection callback had fired by then (`fired_at`).
    """

    def __init__(self, rounds, fired):
        self.rounds = rounds            # [(text, [calls])]
        self.fired = fired              # the callback's invocation log, shared with the test
        self.round_index = -1
        self.sent = []
        self.attempts = []
        self.fired_at = []

    async def __call__(self, client, messages, tools, stream_callback, tool_callback=None,
                       function_call_sink=None, tool_choice=None, artifacts_sink=None,
                       container_gone_sink=None, **params):
        self.round_index += 1

        # The real call builds its request payload ONCE and the recovery retry re-sends it with
        # only the tools demoted — so `input` is the same list object on both attempts, which is
        # exactly the property under test.
        request_params = {"tools": tools, "model": "m", "input": list(messages)}

        async def _safe(_create, operation_type=None, **sent_params):
            declared = _ci_container(sent_params["tools"])
            self.sent.append(declared)
            self.attempts.append(copy.deepcopy(sent_params["input"]))
            self.fired_at.append(len(self.fired))
            if declared == _DEAD:
                raise _gone(_DEAD)
            return MagicMock(output=[], usage=None)

        transport = MagicMock()
        transport._safe_api_call = AsyncMock(side_effect=_safe)
        transport.log_warning = lambda *a, **k: None
        await tool_loop.responses_api._create_with_container_recovery(
            transport, request_params, "streaming",
            container_gone_sink=container_gone_sink, artifacts_sink=artifacts_sink)

        # The model ran in whichever sandbox the SUCCESSFUL request used; an `auto` declaration
        # means the API minted one and reported its real id back.
        served = self.sent[-1]
        if artifacts_sink is not None:
            artifacts_sink.append({"container_id": served if isinstance(served, str)
                                   else _RECOVERY})

        text, calls = self.rounds[min(self.round_index, len(self.rounds) - 1)]
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(copy.deepcopy(calls))
        return text


async def _run(monkeypatch, stream, client=None, **kwargs):
    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools", stream)
    registry = kwargs.pop("registry", None) or _registry()
    return await tool_loop.create_streaming_response_with_tool_loop(
        client or _Client(), messages=[{"role": "user", "content": "write the doc"}],
        tools=[], registry=registry, tool_context=ToolContext(),
        stream_callback=lambda c: None, **kwargs)


def _counting_callback(items_per_round=None):
    """A callback that records each invocation and returns the scripted items for that round.

    `async def` with nothing awaited inside is the pinned shape: the loop awaits it, and the
    body runs start to finish with nothing able to interleave — which is how the job side pops
    its note queue and returns without a suspension point in between.
    """
    fired = []

    async def callback():
        n = len(fired)
        fired.append(n)
        if items_per_round is None:
            return []
        return items_per_round(n)

    return callback, fired


# --------------------------------------------------------------------------- once per round

class TestFiresOncePerRound:
    @pytest.mark.asyncio
    async def test_fires_on_every_round_including_the_forced_final_one(self, monkeypatch):
        """The cap round is the last chance a job has to hear a correction before it writes its
        answer, so `tool_choice="none"` must not skip the seam."""
        stream = _ScriptedStream([("", [_call()]), ("done", [])])
        callback, fired = _counting_callback()
        await _run(monkeypatch, stream, pre_round_input_callback=callback, max_tool_rounds=1)
        # Round 1 spent the one-round budget; round 2 went out with tool_choice="none".
        assert len(stream.snapshots) == 2
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_fires_on_a_pure_free_round(self, monkeypatch):
        """A bookkeeping-only round costs no budget, but it is still a round the job is working
        through — an update that arrived during it must not wait for a productive one."""
        stream = _ScriptedStream([("", [_call("update_todos", "c0", "{}")]),
                                  ("", [_call()]),
                                  ("done", [])])
        callback, fired = _counting_callback()
        await _run(monkeypatch, stream, registry=_registry("echo", "update_todos"),
                   free_tools=["update_todos"], max_tool_rounds=4,
                   pre_round_input_callback=callback)
        assert len(stream.snapshots) == 3
        assert len(fired) == 3

    @pytest.mark.asyncio
    async def test_container_recovery_retry_does_not_inject_twice(self, monkeypatch):
        """Recovery re-sends the SAME request from inside the responses call, below this loop.
        Injecting per HTTP attempt would put the update in front of the model twice; from the
        top of the round the retry simply carries what is already there.

        The second attempt is produced by the REAL `_create_with_container_recovery`, not by a
        test loop: the first wire request names the thread's explicit container and 404s the way
        a dead sandbox does, and production code demotes the tools and re-sends. Only the
        response parsing is faked.
        """
        fired = []

        async def callback():
            fired.append(len(fired))
            return _note()

        stream = _RecoveringStream([("", [_call()]), ("done", [])], fired)
        gone_sink: list = []
        artifacts: list = []
        monkeypatch.setattr(tool_loop.responses_api,
                            "create_streaming_response_with_tools", stream)
        await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[{"role": "user", "content": "write the doc"}],
            tools=_explicit_tools(_DEAD), registry=_registry(),
            tool_context=_holder_ctx(container_id=_DEAD, gone_sink=gone_sink),
            stream_callback=lambda c: None, artifacts_sink=artifacts,
            container_gone_sink=gone_sink, pre_round_input_callback=callback)

        # Production recovery really ran: the corpse is recorded and adoption is vetoed.
        from openai_client.container_errors import adoption_blocked
        assert gone_sink == [_DEAD]
        assert adoption_blocked(artifacts) is True
        # Round 1 died and retried, round 2 went out clean: three wire requests, two rounds.
        assert stream.sent == [_DEAD, {"type": "auto"}, _RECOVERY]
        assert len(fired) == 2

        # Round 1's two attempts carried the identical input, with the note on it exactly once,
        # and no callback ran in between (one injection stands behind both attempts).
        first, retry = stream.attempts[0], stream.attempts[1]
        assert first == retry
        assert [i.get("content") for i in retry].count(_note()[1]["content"]) == 1
        assert stream.fired_at == [1, 1, 2]


# --------------------------------------------------------------------------- where they land

class TestPlacement:
    @pytest.mark.asyncio
    async def test_items_are_on_the_input_of_the_round_that_drained_them(self, monkeypatch):
        stream = _ScriptedStream([("", [_call()]), ("done", [])])
        callback, _fired = _counting_callback(
            lambda n: _note("add a risks section") if n == 1 else [])
        await _run(monkeypatch, stream, pre_round_input_callback=callback)
        first = [i.get("content") for i in stream.snapshots[0]]
        second = [i.get("content") for i in stream.snapshots[1]]
        assert "add a risks section" not in first
        assert "add a risks section" in second

    @pytest.mark.asyncio
    async def test_items_land_after_the_replayed_tool_pairs(self, monkeypatch):
        """A function_call and its function_call_output must stay adjacent, and a reasoning item
        must stay adjacent to the call it belongs to — so the seam is after the whole replay."""
        stream = _ScriptedStream([("", [_call()]), ("done", [])])
        callback, _fired = _counting_callback(lambda n: _note() if n == 1 else [])
        await _run(monkeypatch, stream, pre_round_input_callback=callback)
        snap = stream.snapshots[1]
        types = [i.get("type") for i in snap]
        pair = types.index("function_call")
        assert types[pair + 1] == "function_call_output"
        injected = [i for i, item in enumerate(snap)
                    if item.get("content") == _note()[1]["content"]]
        assert injected and min(injected) > pair + 1

    @pytest.mark.asyncio
    async def test_items_keep_their_order(self, monkeypatch):
        stream = _ScriptedStream([("done", [])])
        callback, _fired = _counting_callback(
            lambda n: [{"role": "user", "content": "first"},
                       {"role": "user", "content": "second"}])
        await _run(monkeypatch, stream, pre_round_input_callback=callback)
        contents = [i.get("content") for i in stream.snapshots[0]]
        assert contents[-2:] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_the_awaited_items_are_what_reaches_the_model(self, monkeypatch):
        """The callback type is async: what gets injected is the AWAITED result, never the
        coroutine object the call itself returns."""
        stream = _ScriptedStream([("done", [])])

        async def callback():
            return _note("swap the chart for a table")

        await _run(monkeypatch, stream, pre_round_input_callback=callback)
        contents = [i.get("content") for i in stream.snapshots[0]]
        assert "swap the chart for a table" in contents
        assert all(isinstance(item, dict) for item in stream.snapshots[0])


# --------------------------------------------------------------------------- failure modes

class TestNeverFailsTheRound:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, monkeypatch):
        """`except Exception`, never `BaseException`: a cancelled job has to actually cancel."""
        stream = _ScriptedStream([("done", [])])

        async def callback():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _run(monkeypatch, stream, pre_round_input_callback=callback)

    @pytest.mark.asyncio
    async def test_a_raising_callback_is_swallowed_and_the_round_goes_out(self, monkeypatch):
        stream = _ScriptedStream([("done", [])])
        client = _Client()

        async def callback():
            raise RuntimeError("the ledger is on fire")

        out = await _run(monkeypatch, stream, client=client, pre_round_input_callback=callback)
        assert out["text"] == "done"
        assert len(stream.snapshots) == 1
        assert any("Pre-round input callback failed" in w for w in client.warnings)

    @pytest.mark.asyncio
    async def test_a_sync_callable_is_refused_rather_than_half_applied(self, monkeypatch):
        """The type is async, period. A plain `def` slipped in here must fail loudly on the
        await — the alternative is its returned list being mistaken for something injectable
        (or, with the old isawaitable dance, a coroutine silently read as malformed and every
        update no-opping without anyone noticing)."""
        stream = _ScriptedStream([("done", [])])
        client = _Client()
        out = await _run(monkeypatch, stream, client=client,
                         pre_round_input_callback=lambda: _note())
        assert out["text"] == "done"
        assert stream.snapshots[0] == [{"role": "user", "content": "write the doc"}]
        assert any("Pre-round input callback failed" in w for w in client.warnings)

    @pytest.mark.parametrize("bad", [
        None,
        {"role": "user", "content": "a bare item, not a list"},
        "a string",
        [{"role": "user", "content": "fine"}, "not a dict"],
    ])
    @pytest.mark.asyncio
    async def test_a_malformed_return_is_dropped_whole(self, monkeypatch, bad):
        """Half an injection is a worse round than an unsteered one, so a list with one bad
        element loses its good elements too — and the round still goes out. The check is on the
        AWAITED value, which is the only thing the loop ever looks at."""
        stream = _ScriptedStream([("done", [])])
        client = _Client()

        async def callback():
            return bad

        out = await _run(monkeypatch, stream, client=client,
                         pre_round_input_callback=callback)
        assert out["text"] == "done"
        assert stream.snapshots[0] == [{"role": "user", "content": "write the doc"}]
        assert sum("not a list of item dicts" in w for w in client.warnings) == 1

    @pytest.mark.asyncio
    async def test_an_empty_list_injects_nothing_and_warns_about_nothing(self, monkeypatch):
        stream = _ScriptedStream([("done", [])])
        client = _Client()

        async def callback():
            return []

        await _run(monkeypatch, stream, client=client, pre_round_input_callback=callback)
        assert stream.snapshots[0] == [{"role": "user", "content": "write the doc"}]
        assert client.warnings == []


# --------------------------------------------------------------------------- opt-in

class TestNoCallback:
    @pytest.mark.asyncio
    async def test_none_callback_is_byte_identical_to_not_passing_one(self, monkeypatch):
        rounds = [("", [_call()]), ("done", [])]
        absent = _ScriptedStream(copy.deepcopy(rounds))
        explicit = _ScriptedStream(copy.deepcopy(rounds))
        client = _Client()
        without = await _run(monkeypatch, absent, client=client)
        with_none = await _run(monkeypatch, explicit, client=client,
                               pre_round_input_callback=None)
        assert absent.snapshots == explicit.snapshots
        assert without == with_none
        assert client.warnings == []
