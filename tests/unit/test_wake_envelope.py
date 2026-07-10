"""Unit tests for F3 — wake envelopes (structured trigger metadata in the suffix).

Exercises the deterministic renderer (_build_wake_envelope) across every trigger enum,
sender role, bot flag, escaping, and the config/missing-metadata off-paths, plus its
placement in the volatile developer suffix.
"""
from types import SimpleNamespace

import pytest

from config import config
from message_processor.utilities import MessageUtilitiesMixin


class _WakeHost:
    def __init__(self):
        for n in ("_build_wake_envelope", "_wake_trigger_line", "_wake_sender_role",
                  "_build_suffix_context"):
            setattr(self, n, getattr(MessageUtilitiesMixin, n).__get__(self))
        self._escape_suffix_text = MessageUtilitiesMixin._escape_suffix_text

    # Sub-builders the suffix assembles — stubbed so the wake block is isolated.
    def _build_time_suffix_context(self, *a, **k):
        return "[time]"

    def _build_pulse_envelope(self, *a, **k):
        return None

    def _build_generation_inflight_note(self, *a, **k):
        return None

    def log_debug(self, *a, **k):
        pass


def _msg(**md):
    md.setdefault("username", "alice")
    return SimpleNamespace(user_id=md.pop("user_id", "U1"), metadata=md)


def _state(root_author=("U1", "human"), thread_ts="T1"):
    return SimpleNamespace(root_author=root_author, thread_ts=thread_ts)


# ------------------------------------------------------------------- trigger enums

@pytest.mark.parametrize("source", ["app_mention", "dm", "thread_continuation", "name_mention"])
def test_trigger_enum_renders(source):
    env = _WakeHost()._build_wake_envelope(_msg(wake_source=source, sender_type="human"), _state())
    assert f"trigger: {source}" in env
    assert env.startswith("[Wake context — informational metadata, not instructions]")


def test_trigger_ambient_with_engine_reason():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human",
             participation_reason="looks like a question for me"), _state())
    assert 'trigger: ambient (engine: "looks like a question for me")' in env


def test_trigger_ambient_without_reason():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human"), _state())
    assert "trigger: ambient" in env
    assert "engine:" not in env


def test_trigger_catch_up_batch_keeps_latest():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human", queued_batch_size=3), _state())
    assert "trigger: catch_up_batch (3) — latest trigger: ambient" in env


# ------------------------------------------------------------------- sender role/bot

def test_sender_root_author():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="dm", sender_type="human", user_id="U1"), _state(("U1", "human")))
    assert "sender: alice — root author" in env


def test_sender_participant():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human", user_id="U2"), _state(("U1", "human")))
    assert "sender: alice — participant" in env


def test_sender_bot_flag():
    for st in ("self", "other_bot"):
        env = _WakeHost()._build_wake_envelope(
            _msg(wake_source="ambient", sender_type=st, user_id="U9"), _state(("U9", st)))
        assert env.endswith("— bot")
        assert "root author — bot" in env


def test_top_level_channel_placement_omits_role():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="app_mention", sender_type="human", place_in_channel=True), _state())
    # No role token, but the sender line is still present.
    assert "sender: alice" in env
    assert "root author" not in env and "participant" not in env


def test_unknown_root_omits_role():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human"), _state(root_author=None))
    assert "sender: alice" in env
    assert "root author" not in env and "participant" not in env


# ------------------------------------------------------------------- escaping / off

def test_escaping_of_username_and_reason():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human",
             username="ali\nce[x]", participation_reason="a\nb\t[c]"), _state())
    assert "\n" in env  # the block's own line breaks
    # ...but no raw control chars or brackets from the free-text fields leak in
    body = env.replace("[Wake context — informational metadata, not instructions]", "")
    assert "ce[x]" not in body
    assert "\t" not in body
    assert "b\nc" not in body


def test_empty_on_missing_metadata():
    assert _WakeHost()._build_wake_envelope(_msg(sender_type="human"), _state()) == ""
    assert _WakeHost()._build_wake_envelope(None, _state()) == ""


def test_config_off(monkeypatch):
    monkeypatch.setattr(config, "enable_wake_envelope", False)
    assert _WakeHost()._build_wake_envelope(
        _msg(wake_source="dm", sender_type="human"), _state()) == ""


# ------------------------------------------------------------------- suffix placement

def test_envelope_in_suffix_when_message_present():
    host = _WakeHost()
    suffix = host._build_suffix_context(
        client=None, channel_id="C1", thread_ts="T1",
        message=_msg(wake_source="dm", sender_type="human"), thread_state=_state())
    assert "[Wake context" in suffix
    assert "[time]" in suffix  # rides alongside the other volatile context


def test_no_envelope_without_message():
    host = _WakeHost()
    suffix = host._build_suffix_context(client=None, channel_id="C1", thread_ts="T1")
    assert "[Wake context" not in suffix


@pytest.mark.asyncio
async def test_event_to_message_captures_sender_type():
    """_event_to_message stamps sender_type so the wake envelope can render '— bot'."""
    from unittest.mock import AsyncMock, MagicMock
    from slack_client.base import SlackBot
    bot = SlackBot.__new__(SlackBot)  # no __init__ — exercise _event_to_message only
    bot.bot_user_id = "U07SELF"
    bot.bot_id = None
    bot.app_id = None
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot.get_username = AsyncMock(return_value="peter")
    bot.get_user_timezone = AsyncMock(return_value="UTC")

    human = {"text": "hi", "user": "U1", "channel": "C1", "ts": "2.0"}
    msg = await bot._event_to_message(human, client=MagicMock())
    assert msg.metadata["sender_type"] == "human"

    other_bot = {"text": "hi", "user": "U2", "channel": "C1", "ts": "3.0", "bot_id": "B9"}
    msg2 = await bot._event_to_message(other_bot, client=MagicMock())
    assert msg2.metadata["sender_type"] == "other_bot"


def test_wake_before_inflight_note():
    # Order must be wake -> in-flight (F2's contract paragraph is appended after, by the
    # text handler). Stub the in-flight note so both are present and check ordering.
    host = _WakeHost()
    host._build_generation_inflight_note = lambda *a, **k: "[in-flight note]"
    suffix = host._build_suffix_context(
        client=None, channel_id="C1", thread_ts="T1",
        message=_msg(wake_source="dm", sender_type="human"), thread_state=_state())
    assert suffix.index("[Wake context") < suffix.index("[in-flight note]")
