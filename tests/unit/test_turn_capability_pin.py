"""One turn, one capability profile (P1 spec §3b).

The composed thread config is not advisory: it picks the model the turn trims against, decides
whether an attachment is mounted for the sandbox, and settles which tools are on the table. Every
retry path re-enters the handlers — the streaming/non-streaming fork, the context-length retry,
the MCP fallback, the timeout retry — and each of those used to re-read the table. A settings
change landing between two of those reads produced ONE request with old-model trimming and
attachment decisions carrying new-model tools, prompting and model.

So the profile is composed once at admission and pinned on the TurnRuntime; everything after
reads the pin.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from message_processor.handlers.text import pinned_thread_config
from message_processor.turn_runtime import TurnRuntime

pytestmark = pytest.mark.asyncio

CH = "C0BKX77NU66"
DM = "D0000001"

OLD = {"model": "gpt-5.6-sol", "enable_code_interpreter": False, "enable_streaming": False}
NEW = {"model": "gpt-5.5", "enable_code_interpreter": True, "enable_streaming": False}


def _message(channel_id=CH):
    return Message(text="hi", user_id="U1", channel_id=channel_id, thread_id="10.0",
                   metadata={"ts": "10.0"})


def _shifting_config(profiles):
    """A config source whose answer CHANGES between calls — someone edited the channel's
    settings while this turn was running."""
    seen = []

    async def _read(**kwargs):
        seen.append(kwargs)
        return profiles[min(len(seen) - 1, len(profiles) - 1)]

    return _read, seen


async def test_a_settings_change_mid_turn_cannot_split_the_profile():
    read, seen = _shifting_config([OLD, NEW])
    turn = TurnRuntime()
    proc = SimpleNamespace(db=None)
    state = SimpleNamespace(config_overrides={})

    with patch.object(config, "get_thread_config_async", read):
        first = await pinned_thread_config(proc, state, _message(), True, turn=turn)
        # …the streaming fork, a context-length retry, an MCP fallback, the timeout retry.
        rereads = [await pinned_thread_config(proc, state, _message(), True, turn=turn)
                   for _ in range(4)]

    assert len(seen) == 1, "the table was read more than once for one turn"
    assert all(r is first for r in rereads)
    assert first["model"] == "gpt-5.6-sol"


async def test_the_profile_is_pinned_on_the_turn_itself():
    read, _seen = _shifting_config([OLD])
    turn = TurnRuntime()
    with patch.object(config, "get_thread_config_async", read):
        resolved = await pinned_thread_config(
            SimpleNamespace(db=None), SimpleNamespace(config_overrides={}),
            _message(), True, turn=turn)
    assert turn.capability_profile is resolved


async def test_without_a_turn_the_profile_is_composed_per_call():
    """Direct-to-handler callers have nothing to pin to; they behave exactly as before."""
    read, seen = _shifting_config([OLD, NEW])
    proc, state = SimpleNamespace(db=None), SimpleNamespace(config_overrides={})
    with patch.object(config, "get_thread_config_async", read):
        first = await pinned_thread_config(proc, state, _message(), True)
        second = await pinned_thread_config(proc, state, _message(), True)
    assert len(seen) == 2
    assert (first["model"], second["model"]) == ("gpt-5.6-sol", "gpt-5.5")


@pytest.mark.parametrize("channel_id,channel_turn", [(CH, True), (DM, False)])
async def test_the_surface_discriminator_rides_the_one_read(channel_id, channel_turn):
    read, seen = _shifting_config([OLD])
    with patch.object(config, "get_thread_config_async", read):
        await pinned_thread_config(SimpleNamespace(db=None),
                                   SimpleNamespace(config_overrides={}),
                                   _message(channel_id), channel_turn, turn=TurnRuntime())
    assert seen[0]["channel_turn"] is channel_turn
    assert seen[0]["channel_id"] == channel_id


# --------------------------------------------------------------- through the pipeline


class _StopTurn(Exception):
    """Ends the turn at a known point rather than letting a test catch bare Exception."""


def _processor(db=None):
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.db = db
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    p.thread_manager._token_counter.count_thread_tokens = MagicMock(return_value=0)
    p.thread_manager._token_counter.count_message_tokens = MagicMock(return_value=0)

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id=CH,
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False,
                            channel_directives=None)

    async def _state_for(*a, **k):
        return state

    async def _noop(*a, **k):
        return None

    p._get_or_rebuild_thread_state = _state_for
    # A channel turn creates its state and pins a whole-channel stream before the handler runs.
    # Both are seams here: what is under test sits upstream of them, and the profile the stream
    # build is stamped with is the very pin being asserted.
    p.get_or_create_channel_thread_state = _state_for
    p._build_channel_turn_stream = _noop
    p._admit_channel_request = _noop
    p._process_attachments = AsyncMock(return_value=([], [], []))
    return p


async def test_admission_pins_the_profile_the_handler_and_its_retries_then_read():
    """The read at admission and every handler read afterwards are the same dict — even though
    the table answers differently the second time it is asked."""
    read, seen = _shifting_config([OLD, NEW])
    p = _processor()
    captured = {}

    async def _handler(*args, **kwargs):
        turn = kwargs["turn"]
        captured["at_admission"] = turn.capability_profile
        # Stand in for a retry re-entering the handler after the settings changed.
        captured["on_retry"] = await pinned_thread_config(
            p, SimpleNamespace(config_overrides={}), _message(), True, turn=turn)
        raise _StopTurn()

    p._handle_text_response = _handler
    client = MagicMock()
    client.send_message = AsyncMock()
    client.self_team_id = "T1"

    with patch.object(config, "get_thread_config_async", read):
        await p.process_message(_message(), client, None)

    assert len(seen) == 1
    assert captured["at_admission"] is captured["on_retry"]
    assert captured["at_admission"]["model"] == "gpt-5.6-sol"


# --------------------------------------------------------------- the DM discriminator


async def test_an_outbound_dm_target_is_a_dm_everywhere():
    """Spec §8: an outbound DM is posted with channel=<user_id>, so a "U"/"W" id names the same
    surface a "D" id does. Both the turn's destination and the tool context read the shared
    discriminator rather than testing the "D" prefix themselves."""
    from message_processor.handlers.text import is_dm_channel
    from message_processor.turn_runtime import DESTINATION_DM, DESTINATION_THREAD

    for channel_id in ("D0000001", "U0000001", "W0000001"):
        message = Message(text="hi", user_id="U1", channel_id=channel_id, thread_id="10.0",
                          metadata={"ts": "10.0"})
        turn = TurnRuntime.for_message(message, channel_post_allowed=True)
        assert turn.reply_destination == DESTINATION_DM, channel_id
        assert is_dm_channel(channel_id) is True

    channel = Message(text="hi", user_id="U1", channel_id=CH, thread_id="10.0",
                      metadata={"ts": "10.0"})
    assert TurnRuntime.for_message(channel).reply_destination == DESTINATION_THREAD
    assert is_dm_channel(CH) is False


async def test_an_absent_channel_id_is_not_a_dm_destination():
    """Unchanged: a message with no conversation id has no DM to be in, and treating it as one
    would settle a destination on nothing."""
    from message_processor.handlers.text import is_dm_channel
    from message_processor.turn_runtime import DESTINATION_THREAD

    message = Message(text="hi", user_id="U1", channel_id=None, thread_id="10.0",
                      metadata={"ts": "10.0"})
    assert TurnRuntime.for_message(message).reply_destination == DESTINATION_THREAD
    assert is_dm_channel(None) is False
