"""The wake gate's recent-context ring (burst follow-ups round, Part A infrastructure).

Inbound semantics through the raw-listener feeder, the read API's selection rules, the bounds,
the rendered block, and the transport's outbound recording.
"""
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor import gate_context
from message_processor.gate_context import (RECENT_CONTEXT_HEADER, GateContextEntry,
                                            render_recent_context)
from slack_client.event_handlers.registration import _feed_gate_context
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.markdown_converter import MarkdownConverter
from slack_client.messaging import NativeStreamSession, SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin
from tests.fixtures.people import ROSTER

DANA, JAMIE, RILEY = ROSTER[0], ROSTER[1], ROSTER[2]
CH = "C0TEST01"


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """The real messaging/identity mixins over a mocked Slack web client."""

    MAX_MESSAGE_LENGTH = 3900

    def __init__(self) -> None:
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MarkdownConverter(platform="slack")
        self.user_cache: Dict[str, Any] = {"U1": {"username": DANA}, "U2": {"username": JAMIE}}

    def log_info(self, *a: Any, **k: Any) -> None:
        pass

    log_debug = log_warning = log_error = log_info


@pytest.fixture(autouse=True)
def _fresh_ring():
    gate_context.ring.reset()
    yield
    gate_context.ring.reset()


def _msg(ts: str, text: str, *, user: str = "U1", thread_ts: Optional[str] = None,
         edited: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {"type": "message", "channel": CH, "channel_type": "channel",
                             "user": user, "ts": ts, "text": text}
    if thread_ts:
        event["thread_ts"] = thread_ts
    if edited:
        event["edited"] = {"user": user, "ts": edited}
    event.update(extra)
    return event


def _changed(ts: str, text: str, *, edited: str, user: str = "U1") -> Dict[str, Any]:
    return {"type": "message", "subtype": "message_changed", "channel": CH,
            "channel_type": "channel", "event_ts": edited,
            "message": {"user": user, "ts": ts, "text": text,
                        "edited": {"user": user, "ts": edited}}}


def _texts(channel: str = CH) -> Dict[str, str]:
    return {e.ts: e.text for e in gate_context.ring.entries(channel)}


def _entry(ts: str, *, thread_ts: Optional[str] = None, kind: str = "human",
           text: str = "hi", **kw: Any) -> GateContextEntry:
    return GateContextEntry(ts=ts, thread_ts=thread_ts, sender_id="U1", sender_kind=kind,
                            text=text, **kw)


# ------------------------------------------------------------------ inbound semantics

def test_dedup_on_channel_and_ts_and_identity_from_cache():
    bot = _Bot()
    _feed_gate_context(bot, _msg("100.000001", "weather in the capital?"), from_mention=False)
    _feed_gate_context(bot, _msg("100.000001", "weather in the capital?"), from_mention=True)
    entries = gate_context.ring.entries(CH)
    assert len(entries) == 1
    assert entries[0].sender_name == DANA and entries[0].sender_kind == "human"


def test_latest_edit_wins_and_a_mention_twin_never_overwrites_it():
    bot = _Bot()
    _feed_gate_context(bot, _msg("100.0", "not that city"), from_mention=False)
    _feed_gate_context(bot, _changed("100.0", "the other city", edited="105.0"),
                       from_mention=False)
    assert _texts()["100.0"] == "the other city"
    # The app_mention copy of the ORIGINAL post arrives late: the edit stays.
    _feed_gate_context(bot, _msg("100.0", "not that city"), from_mention=True)
    assert _texts()["100.0"] == "the other city"
    # A stale message_changed (older edit) does not roll it back either.
    _feed_gate_context(bot, _changed("100.0", "first draft", edited="103.0"),
                       from_mention=False)
    assert _texts()["100.0"] == "the other city"
    # A newer edit carried by an app_mention does win.
    _feed_gate_context(bot, _msg("100.0", "final words", edited="109.0"), from_mention=True)
    assert _texts()["100.0"] == "final words"


def test_delete_and_tombstone_remove():
    bot = _Bot()
    _feed_gate_context(bot, _msg("100.0", "one"), from_mention=False)
    _feed_gate_context(bot, _msg("101.0", "two"), from_mention=False)
    _feed_gate_context(bot, _msg("102.0", "three"), from_mention=False)
    _feed_gate_context(bot, {"type": "message", "subtype": "message_deleted", "channel": CH,
                             "channel_type": "channel", "deleted_ts": "100.0",
                             "previous_message": {"ts": "100.0", "user": "U1"}},
                       from_mention=False)
    _feed_gate_context(bot, {"type": "message", "subtype": "message_changed", "channel": CH,
                             "channel_type": "channel",
                             "message": {"subtype": "tombstone", "ts": "101.0",
                                         "text": "This message was deleted."}},
                       from_mention=False)
    assert list(_texts()) == ["102.0"]


def test_lifecycle_subtypes_and_dms_are_not_recorded():
    bot = _Bot()
    _feed_gate_context(bot, _msg("100.0", "joined", subtype="channel_join"), from_mention=False)
    _feed_gate_context(bot, {**_msg("101.0", "dm text"), "channel": "D0DM",
                             "channel_type": "im"}, from_mention=False)
    assert gate_context.ring.channel_count == 0


def test_other_bots_and_self_are_classified():
    bot = _Bot()
    _feed_gate_context(bot, {"type": "message", "subtype": "bot_message", "channel": CH,
                             "channel_type": "channel", "bot_id": "B0PEER", "ts": "100.0",
                             "text": "peer says", "username": "Claude"}, from_mention=False)
    _feed_gate_context(bot, _msg("101.0", "our own", user="U07SELF"), from_mention=False)
    kinds = {e.ts: (e.sender_kind, e.sender_name) for e in gate_context.ring.entries(CH)}
    assert kinds["100.0"] == ("other_bot", "Claude")
    assert kinds["101.0"][0] == "self"


def test_feeder_never_raises():
    class _Broken:
        def classify_sender(self, msg: Any) -> str:
            raise RuntimeError("boom")

    _feed_gate_context(_Broken(), _msg("100.0", "x"), from_mention=False)
    _feed_gate_context(_Broken(), None, from_mention=False)
    assert gate_context.ring.channel_count == 0


# ------------------------------------------------------------------ recent_for selection

def _seed_room() -> None:
    for e in (_entry("100.0", text="top A"),
              _entry("110.0", text="root B", reply_count=2),
              _entry("111.0", thread_ts="110.0", text="reply B1"),
              _entry("115.0", text="top C"),
              _entry("116.0", thread_ts="110.0", text="reply B2"),
              _entry("120.0", text="top D"),
              _entry("121.0", thread_ts="999.0", text="other thread")):
        gate_context.record(CH, e)


def test_recent_for_a_top_level_message_is_preceding_top_level_only():
    _seed_room()
    got = gate_context.recent_for(CH, before_ts="120.0", limit=10)
    assert [e.text for e in got] == ["top A", "root B", "top C"]
    # strictly older: the message itself is never its own context
    assert all(e.ts != "120.0" for e in got)
    assert [e.text for e in gate_context.recent_for(CH, before_ts="120.0", limit=2)] == [
        "root B", "top C"]
    # a thread_root_ts equal to before_ts is a top-level message too
    assert gate_context.recent_for(CH, before_ts="120.0", thread_root_ts="120.0",
                                   limit=10) == got


def test_recent_for_a_thread_reply_is_top_level_fill_then_the_thread():
    """The A/B harness's selection and order: the thread (root + replies, oldest first); when it
    does not fill `limit`, the newest preceding channel posts fill the room AHEAD of it."""
    _seed_room()
    got = gate_context.recent_for(CH, before_ts="117.0", thread_root_ts="110.0", limit=10)
    assert [e.text for e in got] == ["top A", "top C", "root B", "reply B1", "reply B2"]
    # the thread alone fills the limit: its newest entries, the old root falls away, no fill
    tight = gate_context.recent_for(CH, before_ts="117.0", thread_root_ts="110.0", limit=2)
    assert [e.text for e in tight] == ["reply B1", "reply B2"]
    # room for one fill entry: the newest preceding top-level post, first
    four = gate_context.recent_for(CH, before_ts="117.0", thread_root_ts="110.0", limit=4)
    assert [e.text for e in four] == ["top C", "root B", "reply B1", "reply B2"]


def test_recent_for_before_ts_is_numeric_and_strict():
    gate_context.record(CH, _entry("9.5", text="nine"))
    gate_context.record(CH, _entry("10.0", text="ten"))
    assert [e.text for e in gate_context.recent_for(CH, before_ts="10.0", limit=5)] == ["nine"]
    assert [e.text for e in gate_context.recent_for(CH, before_ts="10.000001", limit=5)] == [
        "nine", "ten"]
    assert gate_context.recent_for(CH, before_ts="10.0", limit=0) == []
    assert gate_context.recent_for("C0COLD", before_ts="10.0", limit=5) == []


def test_recent_for_degrades_when_the_root_was_evicted():
    gate_context.record(CH, _entry("200.0", text="top"))
    gate_context.record(CH, _entry("201.0", thread_ts="150.0", text="reply"))
    got = gate_context.recent_for(CH, before_ts="202.0", thread_root_ts="150.0", limit=5)
    assert [e.text for e in got] == ["top", "reply"]


# ------------------------------------------------------------------ bounds

def test_per_channel_capacity_evicts_the_oldest_ts(monkeypatch):
    monkeypatch.setattr(config, "participation_gate_context_max", 3, raising=False)
    for ts in ("104.0", "101.0", "103.0", "102.0", "100.0"):
        gate_context.record(CH, _entry(ts))
    assert [e.ts for e in gate_context.ring.entries(CH)] == ["102.0", "103.0", "104.0"]


def test_capacity_zero_records_nothing(monkeypatch):
    monkeypatch.setattr(config, "participation_gate_context_max", 0, raising=False)
    assert gate_context.record(CH, _entry("100.0")) is False
    assert gate_context.ring.channel_count == 0


def test_channel_lru_reuses_the_activity_bound(monkeypatch):
    monkeypatch.setattr(config, "participation_activity_lru_max", 2, raising=False)
    gate_context.record("C0A", _entry("1.0"))
    gate_context.record("C0B", _entry("1.0"))
    gate_context.record("C0A", _entry("2.0"))   # C0A is now most recent
    gate_context.record("C0C", _entry("1.0"))   # evicts C0B
    assert gate_context.ring.channel_count == 2
    assert gate_context.ring.entries("C0B") == []
    assert len(gate_context.ring.entries("C0A")) == 2


def test_config_defaults():
    from config import BotConfig

    loaded = BotConfig()
    assert loaded.participation_gate_context_max == 40
    assert loaded.participation_gate_context_messages == 8
    assert loaded.participation_gate_context_chars == 400


# ------------------------------------------------------------------ rendering

def test_render_empty_is_empty_string():
    assert render_recent_context([], char_cap=400) == ""


def test_render_format_labels_and_cap():
    """The harness's block, byte for byte: header, blank line, then one block per message —
    speaker + UTC stamp + topology, the trimmed text capped at C chars plus an ellipsis, an
    Attached line — separated by blank lines."""
    entries = [
        GateContextEntry(ts="1700000000.000100", thread_ts=None, sender_id="U1",
                         sender_kind="human", text="  weather in\nthe capital?  ",
                         sender_name=DANA, reply_count=2),
        GateContextEntry(ts="1700000005.000100", thread_ts="1700000000.000100",
                         sender_id="U07SELF", sender_kind="self", text="It is cloudy."),
        GateContextEntry(ts="1700000009.000100", thread_ts=None, sender_id="B0PEER",
                         sender_kind="other_bot", text="x" * 50, sender_name="Claude",
                         attachments=("chart.png (image)", "notes.pdf (file)")),
        GateContextEntry(ts="1700000010.000100", thread_ts=None, sender_id="U3",
                         sender_kind="human", text="", sender_name=RILEY),
    ]
    assert render_recent_context(entries, char_cap=20) == (
        RECENT_CONTEXT_HEADER + "\n\n"
        f"{DANA} [Tue 2023-11-14 10:13 PM UTC] — posted to the channel\n"
        "weather in\nthe capit…\n\n"
        "the assistant [Tue 2023-11-14 10:13 PM UTC] — a reply inside a thread\n"
        "It is cloudy.\n\n"
        "Claude (a bot) [Tue 2023-11-14 10:13 PM UTC] — posted to the channel\n"
        + "x" * 20 + "…\n"
        "Attached: chart.png (image), notes.pdf (file)\n\n"
        f"{RILEY} [Tue 2023-11-14 10:13 PM UTC] — posted to the channel\n"
        "(no text)")


def test_inflight_line_is_one_line_and_capped():
    entry = GateContextEntry(ts="1700000000.000100", thread_ts=None, sender_id="U1",
                             sender_kind="human", text="weather in\n  the capital right now?",
                             sender_name=DANA)
    assert gate_context.render_inflight_line(entry, char_cap=14) == (
        f"The assistant is currently writing a reply to: [{DANA} Tue 2023-11-14 10:13 PM UTC] "
        "weather in the…")


# ------------------------------------------------------------------ outbound recording

@pytest.mark.asyncio
async def test_an_assistant_reply_send_is_recorded_and_chrome_is_not():
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(side_effect=[
        {"ok": True, "ts": "300.0"}, {"ok": True, "ts": "301.0"}, {"ok": True, "ts": "302.0"}])
    await b.send_message(CH, "290.0", "the combined answer", receipt_class="assistant_reply")
    await b.send_message(CH, "290.0", "Working on it…", receipt_class="chrome")
    await b.send_message(CH, "290.0", "a notice")
    entries = gate_context.ring.entries(CH)
    assert [(e.ts, e.sender_kind, e.thread_ts) for e in entries] == [("300.0", "self", "290.0")]
    assert "combined answer" in entries[0].text
    assert entries[0].sender_id == "U07SELF"


@pytest.mark.asyncio
async def test_each_split_part_is_recorded_with_its_own_ts():
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(side_effect=[
        {"ok": True, "ts": f"40{i}.0"} for i in range(10)])
    long_text = ("para " * 300 + "\n\n") * 8
    await b.send_message(CH, None, long_text, receipt_class="assistant_reply")
    calls = b.app.client.chat_postMessage.await_count
    assert calls > 1
    assert [e.ts for e in gate_context.ring.entries(CH)] == [f"40{i}.0" for i in range(calls)]


@pytest.mark.asyncio
async def test_native_stream_records_only_the_finished_text():
    b = _Bot()
    client = MagicMock()
    client.chat_startStream = AsyncMock(return_value={"ts": "500.0"})
    client.chat_appendStream = AsyncMock(return_value={"ok": True})
    client.chat_stopStream = AsyncMock(return_value={"ok": True})
    session = NativeStreamSession(client, CH, "490.0", team_id="T1", user_id="U1", owner=b)
    assert await session.start("Kyiv is")
    await session.update("Kyiv is cloudy")
    assert gate_context.ring.entries(CH) == []      # partial updates are never recorded
    assert await session.finish(final_text="Kyiv is cloudy, 12C.")
    entries = gate_context.ring.entries(CH)
    assert [(e.ts, e.text, e.sender_kind) for e in entries] == [
        ("500.0", "Kyiv is cloudy, 12C.", "self")]


@pytest.mark.asyncio
async def test_delete_and_own_edit_update_the_ring():
    b = _Bot()
    gate_context.record_assistant_reply(CH, "290.0", "300.0", "first answer")
    gate_context.record_assistant_reply(CH, "290.0", "301.0", "second answer")
    assert gate_context.replace_text(CH, "300.0", "corrected answer")
    assert gate_context.replace_text(CH, "999.0", "unknown") is False
    b.app.client.chat_delete = AsyncMock(return_value={"ok": True})
    assert await b.delete_message(CH, "301.0")
    assert _texts() == {"300.0": "corrected answer"}


@pytest.mark.asyncio
async def test_a_ring_failure_never_breaks_the_send(monkeypatch):
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "300.0"})

    def _boom(*a: Any, **k: Any) -> bool:
        raise RuntimeError("ring down")

    monkeypatch.setattr(gate_context, "record_assistant_reply", _boom)
    assert await b.send_message(CH, None, "answer", receipt_class="assistant_reply") == "300.0"


# ------------------------------------------------------------------ the gate request (Part A)

class _CapturingLLM:
    """The OpenAIClient `self` classify_wake is bound to: captures the request body."""

    def __init__(self) -> None:
        self.client = MagicMock()
        self.kwargs: Dict[str, Any] = {}

    async def _safe_api_call(self, *a: Any, **k: Any) -> Any:
        self.kwargs = {key: v for key, v in k.items() if key != "operation_type"}
        content = MagicMock()
        content.text = '{"wake": true}'
        return MagicMock(status="completed", output=[MagicMock(content=[content])])

    def log_debug(self, *a: Any, **k: Any) -> None:
        pass

    log_warning = log_debug


def _cohort() -> Any:
    from message_processor.participation import SourceMessage

    return (SourceMessage(ts="1700000020.000100", text="no wait, the other city",
                          sender_id="U1", sender_name=DANA, sender_type="human"),)


def _todays_request(sources: Any) -> Dict[str, Any]:
    """The gate request exactly as it was before Part A, written out literally."""
    from config import clamp_effort
    from message_processor.prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT
    from openai_client.api.responses import _render_wake_source

    blocks = [_render_wake_source(s, index=i, total=len(sources)) for i, s in enumerate(sources)]
    return {
        "model": config.utility_model,
        "input": [
            {"role": "developer", "content": WAKE_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user",
             "content": "Messages to decide about, oldest first:\n\n" + "\n\n".join(blocks)},
        ],
        "max_output_tokens": max(2048, config.utility_max_tokens),
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "wake_decision", "strict": True,
                            "schema": {"type": "object",
                                       "properties": {"wake": {"type": "boolean"}},
                                       "required": ["wake"], "additionalProperties": False}}},
        "temperature": 1.0,
        "reasoning": {"effort": clamp_effort(config.utility_model,
                                             config.participation_reasoning_effort)},
    }


@pytest.mark.asyncio
async def test_classify_wake_disabled_is_byte_identical_to_today(monkeypatch):
    import json

    from openai_client.api.responses import classify_wake

    monkeypatch.setattr(config, "participation_gate_context_messages", 0, raising=False)
    llm = _CapturingLLM()
    sources = _cohort()
    assert await classify_wake(llm, sources=sources, recent_context="",
                               inflight_lines=None) is True
    assert json.dumps(llm.kwargs) == json.dumps(_todays_request(sources))


@pytest.mark.asyncio
async def test_classify_wake_enabled_carries_context_inflight_and_both_sentences(monkeypatch):
    from message_processor.prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT
    from openai_client.api.responses import classify_wake

    monkeypatch.setattr(config, "participation_gate_context_messages", 8, raising=False)
    asked = GateContextEntry(ts="1700000000.000100", thread_ts=None, sender_id="U1",
                             sender_kind="human", text="weather in the capital?",
                             sender_name=DANA)
    recent = render_recent_context([asked], char_cap=400)
    inflight = gate_context.render_inflight_line(asked, char_cap=400)
    llm = _CapturingLLM()
    sources = _cohort()
    await classify_wake(llm, sources=sources, recent_context=recent, inflight_lines=inflight)

    today = _todays_request(sources)
    # Order: context block, then the in-flight line, then today's "Messages to decide about".
    assert llm.kwargs["input"][1]["content"] == (
        recent + "\n\n" + inflight + "\n\n" + today["input"][1]["content"])
    # The A3 sentence as its own bullet right after the first "Wake it when:" bullet, and the
    # framing sentence appended to the opening paragraph — nothing else in the prompt moves.
    followup = ("A message that corrects, narrows, adds to or follows up on something the "
                "assistant is answering or has just answered is part of that exchange, and "
                "wakes it.")
    framing = ("The recent conversation is there so you can tell what the messages below are "
               "responding to; it is not itself what you are deciding about.")
    first_bullet = "- someone is talking to it, or about something it is expected to handle;\n"
    opening = "whether to run the assistant on the messages below."
    assert WAKE_CLASSIFIER_SYSTEM_PROMPT.count(first_bullet) == 1
    assert WAKE_CLASSIFIER_SYSTEM_PROMPT.count(opening) == 1
    assert llm.kwargs["input"][0]["content"] == (
        WAKE_CLASSIFIER_SYSTEM_PROMPT
        .replace(first_bullet, first_bullet + "- " + followup + "\n")
        .replace(opening, opening + " " + framing))
    # Everything but the input is today's request.
    assert {k: v for k, v in llm.kwargs.items() if k != "input"} == {
        k: v for k, v in today.items() if k != "input"}


@pytest.mark.asyncio
async def test_gate_verdict_degrades_to_empty_context_when_the_ring_fails(monkeypatch):
    from main import ChatBotV2
    from message_processor.client_contract import Message
    from message_processor.participation import GateEvaluation

    monkeypatch.setattr(config, "participation_gate_context_messages", 8, raising=False)

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("ring down")

    monkeypatch.setattr(gate_context, "recent_for", _boom)
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.db = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    app.processor.db.get_channel_policy_async = AsyncMock(return_value=None)
    app.participation_engine = MagicMock()
    app.participation_engine.evaluate = AsyncMock(
        return_value=GateEvaluation(decline_cause="superseded"))
    app.participation_engine.note_arrival = MagicMock()
    message = Message(text="no wait, the other city", user_id="U1", channel_id=CH,
                      thread_id="1700000020.000100",
                      metadata={"ts": "1700000020.000100", "sender_id": "U1",
                                "gate_required": True, "participation_level": "on"})

    assert await app._gate_verdict(message, MagicMock()) is None
    builder = app.participation_engine.evaluate.await_args.kwargs["gate_context_builder"]
    assert builder is not None
    assert builder(_cohort()) == ("", "")
