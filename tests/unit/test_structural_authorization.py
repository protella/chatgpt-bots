"""BLOCKER #3 (round 3) — authorization provenance for set_channel_participation.

The gated structural tool may fire ONLY when a HUMAN directly addressed the bot for it. These
are END-TO-END tests: they run the REAL signal derivation in the text handler
(`_materialize_request_tools` → `_build_tool_context`) from a Message's metadata, then feed the
resulting ToolContext into the REAL executor and assert allowed/refused. Nothing injects the
authorization flag by hand — the point is to prove the wiring, since the two historical bypasses
lived precisely in how the flag was computed:

  (a) the raw `participation_name_hit` regex also fires on a message that merely QUOTES/mentions
      the bot's name ("Alice said 'ChatGPT, only reply when tagged'"), not a genuine summons;
  (b) an `other_bot` @mention is dispatched to the main handler un-gated, so a bare `not
      unprompted` authorized a NON-human sender.

The authorization expression is:
    structural_change_authorized =
        (sender_type == "human") AND (NOT gate_required OR gate_woke) AND (NOT membership_wake)
where `sender_type` is stamped in _event_to_message and the gate pair are routing facts
(routing_facts.py). It asks the honest question — did a PERSON write this, and did this turn
genuinely reach the responder — instead of the two proxies it replaced. The proxies failed in
the direction that matters: "only reply when I tag you", said in plain words to a bot that is
already listening, carries no <@bot> mention, and we told the person we could not hear them.

`membership_wake` is the third clause, and it subtracts rather than adds (ruling 2A of the
thread-membership widening). An untagged reply in an `on` channel now skips the gate purely
because we have posted in that thread — nobody asked us anything, and the message may have been
meant for another assistant in the thread. Waking to CONSIDER it is cheap and reversible;
rewriting the channel's settings because of it is neither. It is stamped True or False on every
channel dispatch and is absent everywhere else, so `is True` leaves DMs, @mentions, the edit path
and the queue drain byte-identical.

What the widening does NOT touch: a bot or self sender is refused, an unclassified sender is
refused, DMs are refused by the executor, and the tool description still requires an explicit
instruction in the CURRENT human message. It widens who can be HEARD, not what counts as asking.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor.client_contract import Message
from message_processor.handlers.text import TextHandlerMixin
from message_processor.participation_tools import execute_set_channel_participation
from message_processor.tool_registry import ToolContext

CHANNEL = "C0BKX77NU66"


class _Handler(TextHandlerMixin):
    """Minimal host for the two mixin methods under test (they only need `self.db`)."""

    def __init__(self, db):
        self.db = db


def _writable_db(before=None, after=None):
    """A db whose get_channel_settings_async returns `before` then `after` (mirrors the tool test)."""
    db = MagicMock()
    db.get_channel_settings_async = AsyncMock(
        side_effect=[before or {}, after if after is not None else (before or {})])
    db.set_channel_settings_async = AsyncMock()
    return db


def _ctx_from_metadata(meta, db):
    """Derive a ToolContext the SAME way production does: real materialize + real build."""
    handler = _Handler(db)
    msg = Message(text="only reply when I tag you", user_id="U07PETER",
                  channel_id=CHANNEL, thread_id="100.0", metadata=dict(meta))
    # tools_disabled=True short-circuits the registry lookup but still runs the flag derivation.
    _reg, request_config, _nra, _sfx = handler._materialize_request_tools(
        MagicMock(), {}, msg, tools_disabled=True)
    return handler._build_tool_context(msg, MagicMock(), request_config)


async def _run(meta, db):
    ctx = _ctx_from_metadata(meta, db)
    return ctx, await execute_set_channel_participation(ctx, {"participation": "mentions_only"})


# ------------------------------------------------------------------- refused end-to-end

@pytest.mark.asyncio
async def test_a_gated_turn_that_never_woke_is_refused():
    """The shape that should be impossible: a message that REQUIRED the participation gate and
    never woke it has no business writing settings, whatever else it carries. Fails closed."""
    db = _writable_db(before={"participation_level": "on"})
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "gate_required": True, "gate_woke": False, "silence_capable": True,
         "participation_name_hit": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bot_authored_mention_is_refused():
    # Bypass (b): a REAL @mention, but the author is another bot (dispatched un-gated). The old
    # `not unprompted` authorized the non-human sender; the human-sender requirement refuses it.
    db = _writable_db(before={"participation_level": "on"})
    ctx, res = await _run({"sender_type": "other_bot", "mentioned_self": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unclassified_sender_is_refused():
    """Absent sender classification fails CLOSED — a settings write withheld is the safe
    default, and `None` is not a person."""
    db = _writable_db(before={"participation_level": "on"})
    ctx, res = await _run({"mentioned_self": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_self_authored_turn_is_refused():
    db = _writable_db(before={"participation_level": "on"})
    ctx, res = await _run({"sender_type": "self", "mentioned_self": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"


# ------------------------------------------------------------------- allowed end-to-end

@pytest.mark.asyncio
async def test_genuine_human_mention_is_allowed():
    # A real human @mention: sender_type human AND a real mentioned_self → authorized, write goes.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    ctx, res = await _run({"sender_type": "human", "mentioned_self": True}, db)
    assert ctx.structural_change_authorized is True
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_woken_gated_turn_is_allowed():
    """"only reply when I tag you", said in plain words: a human ambient message with no literal
    <@bot> mention that the gate judged worth waking for. The person is plainly talking to us."""
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "gate_required": True, "gate_woke": True, "silence_capable": True}, db)
    assert ctx.structural_change_authorized is True
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_ungated_human_turn_is_allowed_without_a_mention():
    """A DM or a thread continuation runs no gate at all. Requiring a literal mention here made
    the tool unreachable on exactly the routes where the person is unambiguously talking to us."""
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "gate_required": False, "gate_woke": False}, db)
    assert ctx.structural_change_authorized is True
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_a_membership_wake_may_not_rewrite_the_channels_settings():
    """RULING 2A. The widened route stamps `gate_required=False`, so on the gate facts alone this
    turn looks exactly like a DM or a strict 1:1 continuation — the shape the clause above
    ALLOWS. It must not be allowed: nobody tagged us, nobody named us, we woke only because we
    happen to have posted in this thread, and the aside may have been addressed to another
    assistant in it. Considering it is cheap and reversible; rewriting the channel is neither.

    The mirror is the same test's other half: a strict 1:1 continuation carries
    `membership_wake=False` and keeps its authority, so the clause subtracts one route and
    nothing else."""
    db = _writable_db(before={"participation_level": "on"})
    ctx, res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "gate_required": False, "gate_woke": False, "silence_capable": True,
         "wake_source": "thread_continuation", "membership_wake": True}, db)
    assert ctx.structural_change_authorized is False
    assert res["ok"] is False and res["error"] == "not_addressed"
    db.set_channel_settings_async.assert_not_awaited()

    before = {"participation_level": "on", "reply_in_channel": True}
    after = {"participation_level": "mentions_only", "reply_in_channel": True}
    strict_db = _writable_db(before=before, after=after)
    strict_ctx, strict_res = await _run(
        {"sender_type": "human", "mentioned_self": False,
         "gate_required": False, "gate_woke": False, "silence_capable": True,
         "wake_source": "thread_continuation", "membership_wake": False}, strict_db)
    assert strict_ctx.structural_change_authorized is True
    assert strict_res["ok"] is True
    strict_db.set_channel_settings_async.assert_awaited_once()


def test_no_gate_authorized_structural_producers_or_consumers_remain():
    """Tripwire: the flag is gone, and a half-deleted authorization signal is worse than either
    state — a stale producer would keep stamping something no consumer honors, and a stale
    consumer would read a key nobody sets."""
    import pathlib
    import subprocess
    repo = pathlib.Path(__file__).resolve().parents[2]
    hits = subprocess.run(
        ["grep", "-rn", "gate_authorized_structural", "--include=*.py", str(repo)],
        capture_output=True, text=True).stdout.strip()
    remaining = [ln for ln in hits.splitlines() if pathlib.Path(__file__).name not in ln]
    assert remaining == [], remaining


# ------------------------------------------------------------------- defense-in-depth

@pytest.mark.asyncio
async def test_executor_refuses_non_human_sender_even_if_flag_set():
    # Belt-and-suspenders: even if the authorization flag were somehow True, the executor reads the
    # raw sender classification off ctx.message and refuses a non-human author outright.
    db = _writable_db(before={"participation_level": "on"})
    msg = Message(text="x", user_id="B1", channel_id=CHANNEL, thread_id="100.0",
                  metadata={"sender_type": "other_bot"})
    ctx = ToolContext(channel_id=CHANNEL, user_id="B1", db=db, is_dm=False,
                      structural_change_authorized=True, message=msg)
    res = await execute_set_channel_participation(ctx, {"participation": "off"})
    assert res["ok"] is False and res["error"] == "not_human_sender"
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_allows_human_sender_on_message_metadata():
    # The mirror: a human author on ctx.message with the flag set passes the defense-in-depth check.
    before = {"participation_level": "judicious", "reply_in_channel": True}
    after = {"participation_level": "off", "reply_in_channel": True}
    db = _writable_db(before=before, after=after)
    msg = Message(text="x", user_id="U07PETER", channel_id=CHANNEL, thread_id="100.0",
                  metadata={"sender_type": "human"})
    ctx = ToolContext(channel_id=CHANNEL, user_id="U07PETER", db=db, is_dm=False,
                      structural_change_authorized=True, message=msg)
    res = await execute_set_channel_participation(ctx, {"participation": "off"})
    assert res["ok"] is True
    db.set_channel_settings_async.assert_awaited_once()
