"""EDIT §3/§5/§6/§7 — edit_own_message: the two-effect transaction, disclosure first.

The tool overwrites ONE of the bot's own finalized replies, and the design premise is that a
silent edit must be UNREACHABLE: the executor synthesizes the public disclosure and posts it
BEFORE the overwrite, the whole sequence runs under one process-wide keyed lock inside one
effect lease, and every way the second half can fail leaves the first half standing and says so.

What this file drives, against the REAL executor with a mocked Slack world:

* the exact §3 schema and its channel-only registration;
* all six §5 outcome rows and all four §5 runtime-exception rows;
* the §6 shapes: plain text, the exact bot-footer rebuild (actions block byte-preserved),
  every unsupported-shape refusal, continuation markers reapplied byte-for-byte, and both
  length caps (an edit is never split or truncated; the synthesized announcement must fit one
  message and never enters the split path);
* the §7 plumbing: one edit attempt per turn, the keyed lock spanning the WHOLE transaction,
  shielded completion after launch, duplicate-call-id single flight, `visible_action_committed`,
  `turn.edits` → `turn_outcome.edits` (a committed entry always carries `announcement_ts` —
  announcement-first guarantees it existed), destination-provenance persistence from
  `turn.edits`, the empty-response `reply` classification, and the channel-memory exclusion.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from markdown_converter import MarkdownConverter
from message_markers import extract_continuation_markers
from message_processor.stale_send_guard import StaleSendSuppressed
from message_processor.turn_runtime import (DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                            EDIT_STATE_ANNOUNCEMENT_ONLY, EDIT_STATE_COMMITTED,
                                            AuthorizedEditTarget, EditRecord, TurnRuntime)
from slack_client import messaging
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import Delivery, SlackMessagingMixin
from slack_client.utilities import strip_citations
from slack_sdk.errors import SlackApiError
from tool_registry import SURFACE_CHANNEL, ToolContext, ToolRegistry

TEAM = "T1"
CH = "C1"
ROOT = "1700000000.000100"
TS = "1700000060.000200"
EDITED = "1700000100.000000"
PERMALINK = "https://example.slack.com/archives/C1/p1700000060000200"
NEW_TEXT = "The cap is 48000 dollars."
NOTE = "the renewals cap is 48000, not 36000"
ARGS = {"message_ts": TS, "new_text": NEW_TEXT, "correction_note": NOTE}


# ------------------------------------------------------------------------------- harness

class _World:
    """One target message's live Slack state — mutable, so a test can change it mid-flight.

    `events` is the transaction's observable order: "read" per exact read, "post" per accepted
    announcement, "update" per chat.update ATTEMPT (raising ones included)."""

    def __init__(self, message):
        self.message = message
        self.events: list = []
        self.update_kwargs: list = []
        self.update_attempts = 0
        self.update_raises = None
        self.on_update_raise = None

    def snapshot(self):
        return copy.deepcopy(self.message)


def _plain_message(text="The old cap is 36000 dollars.", edited=None, **extra):
    message = {"ts": TS, "text": text, "bot_id": "BBOT", "thread_ts": ROOT, **extra}
    if edited:
        message["edited"] = {"ts": edited}
    return message


FOOTER_ACTIONS = {
    "type": "actions",
    "elements": [{"type": "button", "text": {"type": "plain_text", "text": "⚙️ gpt-5.6-sol"},
                  "action_id": "open_channel_settings"}],
}


def _footer_message(text="Old inline answer.", edited=None):
    return {"ts": TS, "text": text, "bot_id": "BBOT", "thread_ts": ROOT,
            **({"edited": {"ts": edited}} if edited else {}),
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                copy.deepcopy(FOOTER_ACTIONS),
            ]}


def _db(rows=None):
    default = [{"message_ts": TS, "state": "finalized", "receipt_class": "assistant_reply",
                "thread_root_ts": ROOT}]
    return SimpleNamespace(read_channel_sidecars_for_async=AsyncMock(
        return_value={"receipts": default if rows is None else rows}))


def _host(world: _World):
    """The REAL executor + helpers + formatting on a mocked Slack transport."""
    s = MagicMock()
    s.MAX_MESSAGE_LENGTH = 3900
    s._SECTION_TEXT_LIMIT = SlackMessagingMixin._SECTION_TEXT_LIMIT
    s._FOOTER_INLINE_MAX = SlackMessagingMixin._FOOTER_INLINE_MAX
    s._EDIT_UNAUTHORIZED = SlackMessagingMixin._EDIT_UNAUTHORIZED
    s.self_team_id = TEAM
    for name in ("execute_edit_own_message", "_execute_edit_own_message",
                 "_edit_transaction_body",
                 "_edit_preflight_receipt", "_read_exact_message",
                 "_update_edited_message", "_reconcile_uncertain_update", "_edit_permalink",
                 "get_edit_own_message_tool_schema"):
        setattr(s, name, getattr(SlackMessagingMixin, name).__get__(s))
    s._classify_edit_shape = SlackMessagingMixin._classify_edit_shape
    s._live_edited_ts = SlackMessagingMixin._live_edited_ts
    s.format_text = SlackFormattingMixin.format_text.__get__(s)
    s._encode_mentions = lambda t: t
    s.markdown_converter = MarkdownConverter(platform="slack")
    s.is_own_message = lambda m: bool(isinstance(m, dict) and m.get("bot_id") == "BBOT")
    s.get_message_permalink_tool = AsyncMock(
        return_value={"ok": True, "permalink": PERMALINK})

    async def _history(**kwargs):
        world.events.append("read")
        return {"messages": [world.snapshot()]}

    async def _replies(**kwargs):
        world.events.append("read")
        return {"messages": [world.snapshot()]}

    async def _update(**kwargs):
        world.events.append("update")
        world.update_attempts += 1
        if world.update_raises is not None:
            if world.on_update_raise is not None:
                world.on_update_raise()
            raise world.update_raises
        world.update_kwargs.append(kwargs)
        world.message["text"] = kwargs.get("text")
        if "blocks" in kwargs:
            world.message["blocks"] = kwargs["blocks"]
        world.message["edited"] = {"ts": "1700009999.000000"}
        return {"ok": True, "ts": kwargs.get("ts")}

    s.app.client.conversations_history = _history
    s.app.client.conversations_replies = _replies
    s.app.client.chat_update = _update
    s.app.client.chat_delete = AsyncMock()

    s.posts = []

    async def send_message(channel_id, thread_id, text, blocks=None, meta_out=None,
                           username=None, lease=None, surface="final_post", receipts=None,
                           receipt_kind=None, receipt_class=None, on_first_accept=None):
        if lease is not None:
            lease.authorize(surface)
        ts = f"9{len(s.posts) + 1:03d}.000100"
        formatted = s.format_text(strip_citations(text))
        # §11.19: through the REAL acceptance seam — `_note_first_accept` swallows callback
        # exceptions exactly as the production transport does, so a test can never pass by
        # letting a callback crash propagate where the real path would have eaten it.
        messaging._note_first_accept(on_first_accept, ts)
        if meta_out is not None:
            meta_out["delivery"] = Delivery(first_ts=ts, text=formatted, complete=True,
                                            parts_delivered=1, parts_total=1)
        world.events.append("post")
        s.posts.append({"channel_id": channel_id, "thread_ts": thread_id, "raw": text,
                        "text": formatted, "ts": ts, "surface": surface,
                        "receipt_kind": receipt_kind, "receipt_class": receipt_class})
        return ts

    s.send_message = send_message
    return s


def _target(ts=TS, *, edited_ts=None, channel=CH):
    return AuthorizedEditTarget(channel_id=channel, message_ts=ts, thread_root_ts=ROOT,
                                edited_ts=edited_ts, receipt_class="assistant_reply")


def _ctx(host, *, turn=None, edited_ts=None, targets=None, db=None, channel=CH, is_dm=False):
    mapping = targets if targets is not None else {TS: _target(edited_ts=edited_ts)}
    return ToolContext(channel_id=channel, thread_ts=ROOT, trigger_ts="1700000200.000100",
                       client=host, db=db if db is not None else _db(), is_dm=is_dm,
                       turn=turn if turn is not None else TurnRuntime(),
                       authorized_edit_targets=mapping)


def _expected_replacement(host, live_text, new_text=NEW_TEXT):
    prefix, _body, suffix = extract_continuation_markers(live_text)
    return f"{prefix}{host.format_text(strip_citations(new_text))}{suffix}"


# ==================================================================== §3: schema + registration

def test_schema_is_exactly_the_spec_json():
    world = _World(_plain_message())
    schema = _host(world).get_edit_own_message_tool_schema()
    assert schema["type"] == "function" and schema["name"] == "edit_own_message"
    assert schema["parameters"] == {
        "type": "object",
        "properties": {
            "message_ts": {"type": "string",
                           "description": ("Exact ts of one editable assistant message shown "
                                           "in this channel stream or returned by a read tool "
                                           "this turn.")},
            "new_text": {"type": "string",
                         "description": ("Complete corrected replacement body in normal "
                                         "markdown. Omit continuation markers and footer "
                                         "chrome; the tool preserves those.")},
            "correction_note": {"type": "string",
                                "description": ("A concise public description of the specific "
                                                "fact or detail being corrected.")},
        },
        "required": ["message_ts", "new_text", "correction_note"],
    }
    # No model-facing channel or thread parameter, no announcement text, no old-text echo.
    assert set(schema["parameters"]["properties"]) == {"message_ts", "new_text",
                                                       "correction_note"}


def test_registered_beside_post_to_thread_and_hidden_on_the_dm_surface():
    """One static schema for the channel surface; the DM surface never sees the tool at all
    (`enabled=lambda _: False`), and there is no feature flag around the registration."""
    from slack_client.base import SlackBot

    src = inspect.getsource(SlackBot._build_tool_registry)
    assert "get_edit_own_message_tool_schema" in src
    assert "self.execute_edit_own_message" in src
    assert "enabled=lambda _cfg: False" in src

    world = _World(_plain_message())
    registry = ToolRegistry()
    registry.register(_host(world).get_edit_own_message_tool_schema(), AsyncMock(),
                      enabled=lambda _cfg: False)
    assert "edit_own_message" not in {s["name"] for s in registry.schemas()}
    assert "edit_own_message" in {s["name"]
                                  for s in registry.schemas(surface=SURFACE_CHANNEL)}


def test_the_tool_is_budgeted_not_free():
    """§3: free tools are bookkeeping with no visible output; this one performs two visible
    mutations, so it must never ride a `free_tools` set."""
    import message_processor.handlers.text as text_module

    for line in inspect.getsource(text_module).splitlines():
        if "free_tools=" in line:
            assert "edit_own_message" not in line


async def test_the_executor_rejects_dm_contexts_in_depth():
    world = _World(_plain_message())
    host = _host(world)
    for ctx in (_ctx(host, is_dm=True), _ctx(host, channel="D123",
                                             targets={TS: _target(channel="D123")})):
        out = await host.execute_edit_own_message(ctx, dict(ARGS))
        assert out["ok"] is False and out["error"] == "channel_only"
    assert host.posts == [] and world.events == []


# ==================================================================== §3: pre-effect validation

async def test_missing_or_empty_arguments_are_refused():
    world = _World(_plain_message())
    host = _host(world)
    cases = [({"new_text": NEW_TEXT, "correction_note": NOTE}, "missing_message_ts"),
             ({"message_ts": TS, "new_text": "   ", "correction_note": NOTE},
              "empty_new_text"),
             ({"message_ts": TS, "new_text": NEW_TEXT, "correction_note": " "},
              "empty_correction_note")]
    for args, error in cases:
        out = await host.execute_edit_own_message(_ctx(host), args)
        assert out["ok"] is False and out["error"] == error, args
    assert world.events == [] and host.posts == []


async def test_unauthorized_target_reveals_nothing():
    """§2c: one refusal, byte-identical for a ts that exists and one that never did — a probe
    learns nothing, and nothing is read or posted on the way to saying so."""
    world = _World(_plain_message())
    host = _host(world)
    real_but_unauthorized = await host.execute_edit_own_message(
        _ctx(host, targets={}), dict(ARGS))
    invented = await host.execute_edit_own_message(
        _ctx(host), dict(ARGS, message_ts="1699999999.999999"))
    assert real_but_unauthorized == invented
    assert invented == {"ok": False, "error": "unauthorized_target",
                        "message": ("That message was not an editable reply shown to you "
                                    "this turn.")}
    assert world.events == [] and host.posts == []


async def test_a_malformed_mapping_fails_closed():
    world = _World(_plain_message())
    host = _host(world)
    ctx = _ctx(host)
    ctx.authorized_edit_targets = None  # malformed ⇒ EMPTY mapping, never None-wide
    out = await host.execute_edit_own_message(ctx, dict(ARGS))
    assert out["error"] == "unauthorized_target"


# ==================================================================== §5: the six outcome rows

async def test_success_row_announcement_first_then_update():
    """Row 2 — and the transaction's observable ORDER: preflight read, announcement, a SECOND
    exact read, then the update. The disclosure is synthesized (never the model's words), lands
    in the target's own thread, and the EditRecord commits carrying the announcement ts."""
    world = _World(_plain_message())
    original_text = world.message["text"]
    host = _host(world)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))

    assert out == {"ok": True, "message_ts": TS, "announcement_ts": "9001.000100",
                   "announcement_posted": True, "edited": True}
    assert world.events == ["read", "post", "read", "update"]
    # The synthesized disclosure, verbatim — permalink and note, nothing else.
    assert host.posts[0]["raw"] == f"Correction to [my earlier message]({PERMALINK}): {NOTE}"
    assert host.posts[0]["thread_ts"] == ROOT          # receipt.thread_root_ts, not top-level
    assert host.posts[0]["receipt_kind"] == "finalized"
    assert host.posts[0]["receipt_class"] == "correction_announcement"
    # The dedicated updater wrote the formatted replacement, plain shape: text only.
    assert world.update_kwargs[0]["channel"] == CH and world.update_kwargs[0]["ts"] == TS
    assert world.update_kwargs[0]["text"] == _expected_replacement(host, original_text)
    assert "blocks" not in world.update_kwargs[0]
    # §7 accounting: committed record carrying the announcement, visible action, destination.
    assert len(turn.edits) == 1
    record = turn.edits[0]
    assert (record.state, record.error) == (EDIT_STATE_COMMITTED, None)
    assert record.announcement_ts == "9001.000100" and record.target_ts == TS
    assert turn.visible_action_committed is True
    landed = [r for r in turn.committed_destinations
              if r.kind == DEST_KIND_CORRECTION_ANNOUNCEMENT]
    assert len(landed) == 1 and landed[0].first_ts == record.announcement_ts
    assert landed[0].thread_root_ts == ROOT


async def test_the_replacement_text_carries_no_bot_object_mention():
    """A `<@B…>` is not a mention — Slack renders it as raw text. The model can be handed one by
    a peer app that posts in agent mode, so the transaction that overwrites a live reply must not
    be the path that publishes it."""
    world = _World(_plain_message())
    host = _host(world)
    host.bot_user_id_for = lambda bot_id: {"B07KNOWN": "U07KNOWN"}.get(bot_id)
    args = dict(ARGS, new_text="<@B07KNOWN> and <@B07STRANGER> both answered.")
    out = await host.execute_edit_own_message(_ctx(host, turn=TurnRuntime()), args)
    assert out["ok"] is True
    written = world.update_kwargs[0]["text"]
    assert "<@B" not in written
    assert "<@U07KNOWN>" in written and "@bot" in written


async def test_announcement_failed_row():
    """Row 1: Slack rejected the disclosure — no chat.update, no EditRecord, honest error."""
    world = _World(_plain_message())
    host = _host(world)
    host.send_message = AsyncMock(return_value=None)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "announcement_failed"
    assert world.update_attempts == 0
    assert turn.edits == [] and turn.committed_destinations == []


async def test_stale_target_after_announcement_row():
    """Row 3: the target changed between the announcement and the update. The disclosure stays
    (its wording is true either way), NOTHING is overwritten, and the partial tells the model
    not to post a second correction."""
    world = _World(_plain_message())
    host = _host(world)
    inner_send = host.send_message

    async def send_then_race(*args, **kwargs):
        ts = await inner_send(*args, **kwargs)
        world.message["edited"] = {"ts": "1700008888.000000"}   # another actor's edit lands
        return ts

    host.send_message = send_then_race
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))

    assert out["ok"] is False and out["error"] == "stale_target_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert "another correction" in out["message"]
    assert world.update_attempts == 0, "no overwrite after a failed second read"
    assert world.events == ["read", "post", "read"], "the second exact read is what caught it"
    record = turn.edits[0]
    assert record.state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert record.error == "stale_target_after_announcement"
    assert turn.visible_action_committed is True, "the disclosure is in the room"


async def test_update_failed_after_announcement_row():
    """Row 4: a definitive API rejection. The correction notice stays visible — there is no
    compensating deletion — and the record stays announcement_only with the exact error."""
    world = _World(_plain_message())
    host = _host(world)
    world.update_raises = SlackApiError("nope", {"error": "fatal_error"})
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "update_failed_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    host.app.client.chat_delete.assert_not_awaited()
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert turn.edits[0].error == "update_failed_after_announcement"
    assert world.update_attempts == 1, "no blind retry of a definitive rejection"


async def test_ambiguous_update_reconciles_on_content_never_on_edited_ts():
    """Row 5, the landed half: the transport died but the write reached Slack. Equality is the
    exact formatted text — the re-fetch here carries a DIFFERENT edited.ts than any snapshot,
    which must not matter — and the reconciled success commits. Exactly one update attempt."""
    world = _World(_plain_message())
    host = _host(world)
    expected = _expected_replacement(host, world.message["text"])

    def landed_anyway():
        world.message["text"] = expected
        world.message["edited"] = {"ts": "1700007777.000000"}

    world.update_raises = TimeoutError("socket dropped mid-response")
    world.on_update_raise = landed_anyway
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is True and out["edited"] is True
    assert world.update_attempts == 1, "reconcile is a READ, never a blind retry"
    assert turn.edits[0].state == EDIT_STATE_COMMITTED and turn.edits[0].error is None


async def test_ambiguous_update_reconciles_a_block_id_only_difference():
    """§11.7/§11.18 regression (direct block-ID-only difference): the ambiguous-update
    equality is "exact modulo Slack-minted block_ids". A footer reply whose re-fetched
    blocks differ from the rebuilt ones ONLY by the `block_id` keys Slack stamps on storage
    reconciles as SUCCESS — one update attempt, committed record."""
    world = _World(_footer_message())
    host = _host(world)
    formatted = host.format_text(strip_citations(NEW_TEXT))

    def landed_with_minted_ids():
        world.message["text"] = formatted
        world.message["blocks"] = [
            {"type": "section", "block_id": "SlkMint1",
             "text": {"type": "mrkdwn", "text": formatted}},
            {**copy.deepcopy(FOOTER_ACTIONS), "block_id": "SlkMint2"},
        ]
        world.message["edited"] = {"ts": "1700007777.000000"}

    world.update_raises = TimeoutError("socket dropped mid-response")
    world.on_update_raise = landed_with_minted_ids
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is True and out["edited"] is True
    assert world.update_attempts == 1, "reconcile is a READ, never a blind retry"
    assert turn.edits[0].state == EDIT_STATE_COMMITTED and turn.edits[0].error is None


async def test_ambiguous_update_with_unmatched_content_is_unknown():
    """Row 5, the other half: the re-fetch shows the OLD text, so the outcome stays unknown —
    reported as such, announcement standing, still no retry."""
    world = _World(_plain_message())
    host = _host(world)
    world.update_raises = TimeoutError("socket dropped mid-response")
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False
    assert out["error"] == "update_outcome_unknown_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert world.update_attempts == 1
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY


async def test_cancellation_after_launch_completes_the_shielded_transaction():
    """Row 6: a caller that stops waiting does not stop the effect. The transaction body is
    leased and shielded, so a cancellation after the announcement still runs the update and the
    accounting — and the same call id can never relaunch it (its flight stays owned)."""
    world = _World(_plain_message())
    host = _host(world)
    gate = asyncio.Event()
    inner_send = host.send_message

    async def gated_send(*args, **kwargs):
        ts = await inner_send(*args, **kwargs)
        await gate.wait()
        return ts

    host.send_message = gated_send
    turn = TurnRuntime()
    ctx = _ctx(host, turn=turn)
    task = asyncio.create_task(host.execute_edit_own_message(ctx, dict(ARGS)))
    for _ in range(50):
        if host.posts:
            break
        await asyncio.sleep(0)
    assert host.posts, "the announcement is out — the irreversible half has started"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    await turn.wait_for_effects()
    assert world.update_attempts == 1 and world.update_kwargs
    assert turn.edits[0].state == EDIT_STATE_COMMITTED
    assert turn.edits[0].announcement_ts == host.posts[0]["ts"]


async def test_a_duplicate_call_id_joins_the_first_flight_and_relaunches_nothing():
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    ctx = _ctx(host, turn=turn)
    registry = ToolRegistry()
    registry.register(host.get_edit_own_message_tool_schema(), host.execute_edit_own_message)
    first = await registry.dispatch(ctx, "edit_own_message", dict(ARGS), call_id="c1")
    second = await registry.dispatch(ctx, "edit_own_message", dict(ARGS), call_id="c1")
    assert first["ok"] is True and second == first
    assert len(host.posts) == 1 and world.update_attempts == 1
    assert len(turn.edits) == 1


# ==================================================================== §5: runtime exceptions

async def test_launch_not_recorded_means_no_mutation_and_no_record():
    world = _World(_plain_message())
    host = _host(world)

    class _BrokenFlight:
        staged_edit_targets: list = []

        def mark_launched(self):
            raise RuntimeError("the flight ledger is broken")

    turn = TurnRuntime()
    ctx = _ctx(host, turn=turn)
    ctx.tool_flight = _BrokenFlight()
    out = await host.execute_edit_own_message(ctx, dict(ARGS))
    assert out["ok"] is False and out["error"] == "launch_not_recorded"
    assert host.posts == [] and world.update_attempts == 0
    assert turn.edits == []


async def test_effect_revoked_means_nothing_was_attempted():
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    turn.revoke_effects("straggler resisted cancellation")
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert world.events == [] and host.posts == []
    assert turn.edits == []


async def test_stale_send_suppressed_re_raises_unchanged():
    """The announcement's leased send refused because the room moved on. Control flow, not a
    tool error: re-raised UNCHANGED (matching post_to_thread), no announcement, no edit, no
    EditRecord."""
    world = _World(_plain_message())
    host = _host(world)
    suppression = StaleSendSuppressed(scope=("thread", CH, ROOT), last_seen_ts=ROOT,
                                      observed_latest_ts="1700000300.000100",
                                      surface="edit_own_message")
    host.send_message = AsyncMock(side_effect=suppression)
    turn = TurnRuntime()
    with pytest.raises(StaleSendSuppressed) as caught:
        await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert caught.value is suppression, "re-raised unchanged, not wrapped or rebuilt"
    assert world.update_attempts == 0
    assert turn.edits == [] and turn.destinations == []


async def test_epoch_refused_after_announcement_is_an_announcement_only_partial(monkeypatch):
    """§5's fourth exception row: the fence refuses the UPDATE after the disclosure landed. The
    result is the partial (`announcement_posted: true, edited: false`), the record stays
    announcement_only with the exact error, and chat.update is never reached."""

    class _Refused(Exception):
        pass

    def _refuse(client, channel_id, site):
        if site == "edit_own_message:update":
            raise _Refused(site)

    monkeypatch.setattr(messaging, "_epoch_authorize", _refuse)
    monkeypatch.setattr(messaging, "_EPOCH_REFUSED", _Refused)
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "epoch_refused_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert world.update_attempts == 0, "the fence sits immediately before chat.update"
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert turn.edits[0].error == "epoch_refused_after_announcement"


async def test_the_backstop_is_phase_aware_pre_announcement():
    """§11.8, first phase: an unexpected exception BEFORE the disclosure posts is the §3
    `edit_failed` backstop — Slack untouched, no EditRecord, and the model may fall back to
    a normal correction."""
    world = _World(_plain_message())
    host = _host(world)
    host._edit_preflight_receipt = AsyncMock(side_effect=RuntimeError("preflight bug"))
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "edit_failed"
    assert "announcement_posted" not in out
    assert host.posts == [] and world.update_attempts == 0
    assert turn.edits == []


async def test_the_backstop_is_phase_aware_post_acceptance():
    """§11.8, second phase: an unexpected exception AFTER the disclosure was accepted must
    never read as a bare failure that invites a duplicate correction — the result carries
    `announcement_posted: true`, and the EditRecord stays announcement_only with the error."""
    world = _World(_plain_message())
    host = _host(world)
    real_read = host._read_exact_message
    calls = {"n": 0}

    async def read_then_break(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:   # the post-announcement second read — past acceptance
            raise RuntimeError("accounting bug after acceptance")
        return await real_read(*args, **kwargs)

    host._read_exact_message = read_then_break
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "edit_failed_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert out["announcement_ts"] == host.posts[0]["ts"]
    assert "another correction" in out["message"]
    assert world.update_attempts == 0
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert turn.edits[0].error == "edit_failed_after_announcement"
    assert turn.visible_action_committed is True
    committed = [d for d in turn.destinations
                 if getattr(d, "first_ts", None) == host.posts[0]["ts"]]
    assert committed, "the CV10 join: a landed disclosure has a committed destination"
    assert committed[0].kind == DEST_KIND_CORRECTION_ANNOUNCEMENT, "the disclosure is in the room"


async def test_accounting_is_crash_ordered_past_destination_bookkeeping():
    """§11.10: the EditRecord append and visible_action_committed land IMMEDIATELY on
    acceptance, outside the swallowed region — destination/telemetry bookkeeping raising
    changes nothing, and the update still proceeds."""
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()

    def broken_destination(**kwargs):
        raise RuntimeError("destination ledger broke")

    turn.mark_destination_committed = broken_destination
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is True and out["edited"] is True
    assert world.update_attempts == 1, "the update still proceeded"
    assert len(turn.edits) == 1
    assert turn.edits[0].state == EDIT_STATE_COMMITTED
    assert turn.visible_action_committed is True


async def test_the_backstop_covers_argument_coercion():
    """§11.14/§11.18 regression: the backstop begins at the executor's FIRST line — a
    numeric, list or dict new_text (model-supplied junk) returns `edit_failed`, never an
    AttributeError out of the coercion, and Slack is untouched."""
    world = _World(_plain_message())
    host = _host(world)
    for bad in (12345, ["a", "list"], {"a": "dict"}):
        turn = TurnRuntime()
        out = await host.execute_edit_own_message(
            _ctx(host, turn=turn),
            {"message_ts": TS, "new_text": bad, "correction_note": NOTE})
        assert out["ok"] is False and out["error"] == "edit_failed", bad
        assert turn.edits == []
    assert host.posts == [] and world.events == []


async def test_citation_only_arguments_fail_the_emptiness_checks():
    """§11.17/§11.18 regression: strip_citations runs on BOTH texts BEFORE the emptiness
    checks, so citation-only input fails as empty_new_text / empty_correction_note — a
    disclosure can never post naming nothing. Slack untouched."""
    world = _World(_plain_message())
    host = _host(world)
    citation = "cite:ship:turn0:walking:"
    assert strip_citations(citation).strip() == "", "the premise of the case"
    out = await host.execute_edit_own_message(
        _ctx(host), {"message_ts": TS, "new_text": citation, "correction_note": NOTE})
    assert out["ok"] is False and out["error"] == "empty_new_text"
    out = await host.execute_edit_own_message(
        _ctx(host), {"message_ts": TS, "new_text": NEW_TEXT, "correction_note": citation})
    assert out["ok"] is False and out["error"] == "empty_correction_note"
    assert host.posts == [] and world.events == []


async def test_a_send_crash_after_acceptance_leaves_the_record_standing():
    """§11.15/§11.18 regression (acceptance-before-send-return crash): Slack accepts the
    disclosure — the `on_first_accept` seam fires — and send_message then raises during its
    own receipt bookkeeping, BEFORE returning. The EditRecord was already created AT the
    seam, so the result is the partial contract, never a bare edit_failed, and the record
    and `visible_action_committed` stand."""
    world = _World(_plain_message())
    host = _host(world)
    inner_send = host.send_message

    async def crashing_send(*args, **kwargs):
        await inner_send(*args, **kwargs)  # posts and fires on_first_accept
        raise RuntimeError("receipt bookkeeping broke after acceptance")

    host.send_message = crashing_send
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "edit_failed_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert out["announcement_ts"] == host.posts[0]["ts"]
    assert world.update_attempts == 0, "the transaction stopped — no overwrite"
    assert len(turn.edits) == 1
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert turn.edits[0].error == "edit_failed_after_announcement"
    assert turn.edits[0].announcement_ts == host.posts[0]["ts"]
    assert turn.visible_action_committed is True
    committed = [d for d in turn.destinations
                 if getattr(d, "first_ts", None) == host.posts[0]["ts"]]
    assert committed, "the CV10 join: a landed disclosure has a committed destination"
    assert committed[0].kind == DEST_KIND_CORRECTION_ANNOUNCEMENT


async def test_announcement_outcome_unknown_when_the_send_raises_before_acceptance():
    """§11.16/§11.18: the announcement send raises BEFORE the acceptance seam fires (an
    ambiguous transport failure whose reconciliation found nothing — send_message's
    reconciliation rethrow). The transaction STOPS: no update (silent-edit safety), no
    EditRecord, `ok:false` with announcement_outcome_unknown, and the message says the
    notice may already be visible and allows at most ONE normal follow-up, never a retry
    of the edit."""
    world = _World(_plain_message())
    host = _host(world)
    host.send_message = AsyncMock(side_effect=TimeoutError("socket died mid-post"))
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "announcement_outcome_unknown"
    assert "announcement_posted" not in out
    assert "may or may not have posted" in out["message"]
    assert "ONE normal follow-up" in out["message"]
    assert world.update_attempts == 0
    assert turn.edits == [] and turn.visible_action_committed is False
    assert turn.destinations == []


async def test_a_swallowed_callback_crash_is_repaired_in_lock():
    """§11.19: the acceptance callback itself raises INSIDE the real `_note_first_accept`
    seam — which swallows it, so the send returns success with the accounting cut short at
    its very first assignment. The post-send in-lock verification repairs the pre-created
    record into `turn.edits` before any update, and the transaction still commits."""

    class _FlakyEdits(list):
        raised = False

        def append(self, item):
            if not self.raised:
                self.raised = True
                raise RuntimeError("edits ledger broke at the seam")
            super().append(item)

    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    turn.edits = _FlakyEdits()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert turn.edits.raised, "the callback DID crash at the seam — the premise of the case"
    assert out["ok"] is True and out["edited"] is True
    assert world.update_attempts == 1, "verification-and-repair ran BEFORE the update"
    assert len(turn.edits) == 1, "the record was repaired into turn.edits"
    assert turn.edits[0].state == EDIT_STATE_COMMITTED and turn.edits[0].error is None
    assert turn.edits[0].announcement_ts == host.posts[0]["ts"]
    assert turn.visible_action_committed is True


async def test_announcement_time_epoch_refusal_is_outcome_unknown(monkeypatch):
    """§11.21: the epoch fence refuses the ANNOUNCEMENT send itself — driven through the
    REAL send_message, whose `_epoch_authorize` fence sits immediately before dispatch —
    and the classification answers `announcement_outcome_unknown`: nothing posted, no
    update, no EditRecord."""

    class _Refused(Exception):
        pass

    def _refuse(client, channel_id, site):
        if site == "send_message:edit_own_message":
            raise _Refused(site)

    monkeypatch.setattr(messaging, "_epoch_authorize", _refuse)
    monkeypatch.setattr(messaging, "_EPOCH_REFUSED", _Refused)
    world = _World(_plain_message())
    host = _host(world)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)
    host._record_receipt = SlackMessagingMixin._record_receipt.__get__(host)
    host._compose_reply_with_footer = SlackMessagingMixin._compose_reply_with_footer.__get__(
        host)
    host._split_message = SlackMessagingMixin._split_message.__get__(host)
    host.app.client.chat_postMessage = AsyncMock()
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "announcement_outcome_unknown"
    assert "announcement_posted" not in out
    host.app.client.chat_postMessage.assert_not_awaited()
    assert world.update_attempts == 0
    assert turn.edits == [] and turn.visible_action_committed is False


async def test_a_deterministic_local_send_exception_is_announcement_failed():
    """§11.21: an announcement-send exception that is NO transport ambiguity — a local bug
    raised before anything could reach Slack — maps to the §5 `announcement_failed` row:
    Slack untouched, no EditRecord, and the model may fall back to a normal correction."""
    world = _World(_plain_message())
    host = _host(world)
    host.send_message = AsyncMock(side_effect=TypeError("bad payload composition"))
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is False and out["error"] == "announcement_failed"
    assert "announcement_posted" not in out
    assert "Post a normal correction message instead" in out["message"]
    assert world.update_attempts == 0
    assert turn.edits == [] and turn.visible_action_committed is False


async def test_a_raising_context_property_returns_edit_failed():
    """§11.22: the backstop wraps from the executor's LITERAL first line — even the ctx
    reads run inside it, so a context whose `channel_id` property raises answers
    `edit_failed`, never a traceback, with Slack untouched."""
    world = _World(_plain_message())
    host = _host(world)

    class _HostileCtx:
        @property
        def channel_id(self):
            raise RuntimeError("context backing store broke")

    out = await host.execute_edit_own_message(_HostileCtx(), dict(ARGS))
    assert out == {"ok": False, "error": "edit_failed",
                   "message": "Could not edit that message."}
    assert host.posts == [] and world.events == []


# ==================================================================== §6: shapes and length

async def test_preflight_stale_and_shape_and_permalink_rows():
    """The pre-effect refusals of §5's preflight, each with Slack untouched."""
    # Changed edited.ts vs the authorized snapshot ⇒ stale_target.
    world = _World(_plain_message(edited="1700000500.000000"))
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host, edited_ts=None), dict(ARGS))
    assert out["error"] == "stale_target" and host.posts == []
    # An unreadable / vanished message ⇒ stale_target.
    world2 = _World(_plain_message())
    host2 = _host(world2)

    async def _gone(**kwargs):
        world2.events.append("read")
        return {"messages": []}

    host2.app.client.conversations_replies = _gone
    out = await host2.execute_edit_own_message(_ctx(host2), dict(ARGS))
    assert out["error"] == "stale_target" and host2.posts == []
    # Permalink failure ⇒ permalink_failed, before anything posts.
    world3 = _World(_plain_message())
    host3 = _host(world3)
    host3.get_message_permalink_tool = AsyncMock(return_value={"ok": False,
                                                               "error": "message_not_found"})
    out = await host3.execute_edit_own_message(_ctx(host3), dict(ARGS))
    assert out["error"] == "permalink_failed" and host3.posts == []
    # A receipt that no longer re-proves (legacy NULL class) ⇒ the unauthorized refusal.
    world4 = _World(_plain_message())
    host4 = _host(world4)
    ctx = _ctx(host4, db=_db([{"message_ts": TS, "state": "finalized",
                               "receipt_class": None, "thread_root_ts": ROOT}]))
    out = await host4.execute_edit_own_message(ctx, dict(ARGS))
    assert out["error"] == "unauthorized_target" and host4.posts == []


async def test_footer_reply_rebuild_preserves_the_actions_block_byte_for_byte():
    world = _World(_footer_message())
    host = _host(world)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is True
    blocks = world.update_kwargs[0]["blocks"]
    assert len(blocks) == 2
    formatted = host.format_text(strip_citations(NEW_TEXT))
    assert blocks[0] == {"type": "section", "text": {"type": "mrkdwn", "text": formatted}}
    assert blocks[1] == FOOTER_ACTIONS, "the existing actions block, byte-for-byte"
    assert world.update_kwargs[0]["text"] == formatted, "top-level fallback text updated too"


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(files=[{"id": "F1", "name": "report.pdf"}]),
    lambda m: m.update(attachments=[{"fallback": "webhook payload"}]),
    # §11.25: ONE rich_text block is Slack's own plain-message mirror and therefore the
    # PLAIN shape — the unsupported representative here is rich_text alongside a section,
    # which no plain post ever materializes.
    lambda m: m.update(blocks=[{"type": "rich_text", "elements": []},
                               {"type": "section", "text": {"type": "mrkdwn", "text": "x"}}]),
    lambda m: m.update(blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "x"}},
                               {"type": "section", "text": {"type": "mrkdwn", "text": "y"}}]),
    lambda m: m.update(blocks=[{"type": "actions", "elements": [
        {"type": "button", "action_id": "research_cancel"}]}]),
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"}},
        {"type": "actions", "elements": [
            {"type": "button", "action_id": "open_channel_settings"},
            {"type": "button", "action_id": "response_feedback"}]}]),
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"}},
        {"type": "context", "elements": []},
        copy.deepcopy(FOOTER_ACTIONS)]),
    lambda m: m.update(blocks=[{"type": "actions", "elements": []}]),
    # §11.4's strict-shape rows: the section may carry ONLY type/text (+ block_id) with
    # mrkdwn text; the actions block exactly ONE open_channel_settings BUTTON.
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"},
         "accessory": {"type": "image", "image_url": "https://x/i.png", "alt_text": "i"}},
        copy.deepcopy(FOOTER_ACTIONS)]),
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"},
         "fields": [{"type": "mrkdwn", "text": "f"}]},
        copy.deepcopy(FOOTER_ACTIONS)]),
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"}},
        {"type": "actions", "elements": [
            {"type": "button", "action_id": "open_channel_settings"},
            {"type": "button", "action_id": "open_channel_settings"}]}]),
    lambda m: m.update(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": "x"}},
        {"type": "actions", "elements": [
            {"type": "static_select", "action_id": "open_channel_settings"}]}]),
], ids=["files", "attachments", "rich_text_plus_section", "two_sections", "unknown_action",
        "extra_footer_element", "three_blocks", "empty_actions",
        "section_accessory", "section_fields", "duplicate_settings_buttons",
        "non_button_element"])
async def test_every_unsupported_shape_is_refused(mutate):
    message = _plain_message()
    mutate(message)
    world = _World(message)
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host), dict(ARGS))
    assert out["ok"] is False and out["error"] == "unsupported_message_shape"
    assert host.posts == [] and world.update_attempts == 0


async def test_continuation_markers_are_reapplied_byte_for_byte():
    """A mid-split part keeps its seams, and a LEGACY marker form keeps its legacy bytes —
    never rewritten to the canonical shape."""
    prefix = "*Part 2 (continued)*\n\n"
    suffix = "\n\n**Continued in next message...**"      # legacy markdown-bold trailer
    world = _World(_plain_message(text=f"{prefix}the old middle of the answer{suffix}"))
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host), dict(ARGS))
    assert out["ok"] is True
    formatted = host.format_text(strip_citations(NEW_TEXT))
    assert world.update_kwargs[0]["text"] == f"{prefix}{formatted}{suffix}"
    assert world.update_kwargs[0]["text"].endswith("**Continued in next message...**")


def test_extract_continuation_markers_is_byte_preserving():
    cases = [
        "no markers at all",
        "*Part 2 (continued)*\n\nbody text",
        "...body\n\n*Continued in next message...*",
        "body\n\n**Continued in next message...**",
        "_Continued in next message..._",
        "*...continued*\n\nbody\n\n*Continued in next message...*\n\n"
        "*Continued in next message...*",
        "",
    ]
    for text in cases:
        prefix, body, suffix = extract_continuation_markers(text)
        assert prefix + body + suffix == text, text
    prefix, body, suffix = extract_continuation_markers(
        "**Part 3 (continued)**\n\nthe body\n\n**Continued in next message...**")
    assert prefix == "**Part 3 (continued)**\n\n"        # legacy prefix bytes intact
    assert body == "the body"
    assert suffix == "\n\n**Continued in next message...**"
    assert extract_continuation_markers("plain") == ("", "plain", "")


async def test_replacement_over_the_cap_is_never_split_or_truncated():
    world = _World(_plain_message())
    host = _host(world)
    out = await host.execute_edit_own_message(
        _ctx(host), dict(ARGS, new_text="x" * 4000))
    assert out["ok"] is False and out["error"] == "replacement_too_long"
    assert host.posts == [] and world.update_attempts == 0


async def test_no_change_is_refused_before_anything_posts():
    world = _World(_plain_message(text="The cap is 48000 dollars."))
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host), dict(ARGS))
    assert out["ok"] is False and out["error"] == "no_change"
    assert host.posts == []


async def test_announcement_over_the_cap_is_refused_before_posting():
    world = _World(_plain_message())
    host = _host(world)
    out = await host.execute_edit_own_message(
        _ctx(host), dict(ARGS, correction_note="n" * 4000))
    assert out["ok"] is False and out["error"] == "announcement_too_long"
    assert host.posts == [] and world.update_attempts == 0


async def test_the_announcement_posts_as_one_message_through_the_real_send(monkeypatch):
    """The synthesized disclosure rides the REAL send_message and must fit ONE message: a
    single chat.postMessage, no continuation chrome, receipt class riding through."""
    from message_processor import outbound_receipts

    booked = {}

    async def _record(**kwargs):
        booked.update(kwargs)

    monkeypatch.setattr(outbound_receipts, "record_transport_post", _record)
    world = _World(_plain_message())
    host = _host(world)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)
    host._record_receipt = SlackMessagingMixin._record_receipt.__get__(host)
    host._compose_reply_with_footer = SlackMessagingMixin._compose_reply_with_footer.__get__(
        host)
    host._split_message = SlackMessagingMixin._split_message.__get__(host)
    host.app.client.chat_postMessage = AsyncMock(return_value={"ts": "905.000100"})
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(
        _ctx(host, turn=turn), dict(ARGS, correction_note="n" * 3000))
    assert out["ok"] is True and out["announcement_ts"] == "905.000100"
    assert host.app.client.chat_postMessage.await_count == 1, "one message, never the splitter"
    posted_text = host.app.client.chat_postMessage.await_args.kwargs["text"]
    assert "continued" not in posted_text.lower()
    assert booked["receipt_class"] == "correction_announcement"
    assert booked["receipt_kind"] == "finalized"


@pytest.mark.parametrize("new_text", ["y" * 200, "a\nb\nc\nd"],
                         ids=["over_180_chars", "over_two_newlines"])
async def test_footer_replacement_that_no_longer_fits_inline_is_refused(new_text):
    world = _World(_footer_message())
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host), dict(ARGS, new_text=new_text))
    assert out["ok"] is False and out["error"] == "edit_too_long_for_inline_footer"
    assert host.posts == [] and world.update_attempts == 0


# ==================================================================== §7: lock + one per turn

async def test_the_keyed_lock_spans_the_whole_transaction():
    """Preflight, both mutations and the accounting all run under the (team, channel, ts) lock:
    with the lock held elsewhere, the executor cannot even READ; once inside, every mutation
    happens with the lock held."""
    world = _World(_plain_message())
    host = _host(world)
    lock = messaging._edit_transaction_lock(TEAM, CH, TS)
    inner_send = host.send_message

    async def send_under_lock(*args, **kwargs):
        assert lock.locked(), "the announcement posts inside the keyed lock"
        return await inner_send(*args, **kwargs)

    host.send_message = send_under_lock
    inner_update = host.app.client.chat_update

    async def update_under_lock(**kwargs):
        assert lock.locked(), "the update runs inside the same keyed lock"
        return await inner_update(**kwargs)

    host.app.client.chat_update = update_under_lock
    await lock.acquire()
    try:
        task = asyncio.create_task(host.execute_edit_own_message(_ctx(host), dict(ARGS)))
        for _ in range(20):
            await asyncio.sleep(0)
        assert world.events == [], "not even the preflight read runs while the lock is held"
    finally:
        lock.release()
    out = await task
    assert out["ok"] is True and world.events == ["read", "post", "read", "update"]


async def test_the_lock_map_prunes_uncontended_entries():
    """§11.11: the keyed map drops an entry when its transaction ends uncontended, so it
    never grows monotonically with the messages ever edited — while a CONTENDED entry
    survives exactly as long as someone holds or awaits it."""
    key = (TEAM, CH, TS)
    world = _World(_plain_message())
    host = _host(world)
    out = await host.execute_edit_own_message(_ctx(host), dict(ARGS))
    assert out["ok"] is True
    assert key not in messaging._EDIT_TRANSACTION_LOCKS
    # A failed transaction prunes on the way out too.
    world2 = _World(_plain_message())
    host2 = _host(world2)
    host2.send_message = AsyncMock(return_value=None)
    failed = await host2.execute_edit_own_message(_ctx(host2), dict(ARGS))
    assert failed["error"] == "announcement_failed"
    assert key not in messaging._EDIT_TRANSACTION_LOCKS
    # Contended: held elsewhere, the entry stays; released and finished, it goes.
    world3 = _World(_plain_message())
    host3 = _host(world3)
    lock = messaging._edit_transaction_lock(TEAM, CH, TS)
    await lock.acquire()
    try:
        task = asyncio.create_task(host3.execute_edit_own_message(_ctx(host3), dict(ARGS)))
        for _ in range(20):
            await asyncio.sleep(0)
        assert key in messaging._EDIT_TRANSACTION_LOCKS
    finally:
        lock.release()
    out3 = await task
    assert out3["ok"] is True
    assert key not in messaging._EDIT_TRANSACTION_LOCKS


async def test_two_runtimes_race_one_wins_one_gets_stale_target():
    """The §5 race, with TWO SEPARATE TurnRuntime instances contending for one message: the
    winner edits; the loser's own preflight re-read (inside the lock, after the winner) sees a
    changed edited.ts and refuses with stale_target — it never announces and never overwrites."""
    world = _World(_plain_message())
    host_a, host_b = _host(world), _host(world)
    turn_a, turn_b = TurnRuntime(), TurnRuntime()
    results = await asyncio.gather(
        host_a.execute_edit_own_message(_ctx(host_a, turn=turn_a), dict(ARGS)),
        host_b.execute_edit_own_message(_ctx(host_b, turn=turn_b), dict(ARGS)))
    outcomes = sorted((r.get("error") or "ok") for r in results)
    assert outcomes == ["ok", "stale_target"], results
    assert len(host_a.posts) + len(host_b.posts) == 1, "one announcement in the room"
    assert world.update_attempts == 1, "one overwrite, ever"
    assert len(turn_a.edits) + len(turn_b.edits) == 1


async def test_two_calls_in_one_turn_yield_edit_already_attempted():
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    first = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert first["ok"] is True
    second = await host.execute_edit_own_message(
        _ctx(host, turn=turn), dict(ARGS, new_text="A different correction."))
    assert second["ok"] is False and second["error"] == "edit_already_attempted"
    assert len(host.posts) == 1 and world.update_attempts == 1
    assert len(turn.edits) == 1
    # …and the reservation holds even when the FIRST attempt failed after reserving.
    world2 = _World(_plain_message())
    host2 = _host(world2)
    host2.send_message = AsyncMock(return_value=None)
    turn2 = TurnRuntime()
    failed = await host2.execute_edit_own_message(_ctx(host2, turn=turn2), dict(ARGS))
    assert failed["error"] == "announcement_failed"
    again = await host2.execute_edit_own_message(_ctx(host2, turn=turn2), dict(ARGS))
    assert again["error"] == "edit_already_attempted"


# ==================================================================== §7: telemetry + plumbing

def _turn_outcome_row(turn):
    from message_processor import participation_telemetry as pt

    turn.stream_build_present = True
    turn.H = "1700000200.000100"
    pt.initialize()
    assert pt.emit_turn_outcome(turn, channel_id=CH, trigger_ts="1700000200.000100",
                                kind="reply") is True
    pt._drain()
    rows = [json.loads(line) for line in
            (pathlib.Path(config.log_directory) / pt.LOG_NAME).read_text().splitlines()
            if line.strip()]
    return [r for r in rows if r["event"] == "turn_outcome"][-1]


async def test_turn_outcome_edits_payload_and_the_announcement_join():
    """CV10's row half: `edits` carries the exact payload (a committed entry ALWAYS carries
    `announcement_ts`, no `error` key), and the announcement joins a committed
    correction_announcement destination whose first_ts equals it."""
    world = _World(_plain_message())
    host = _host(world)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    row = _turn_outcome_row(turn)
    assert row["edits"] == [{"channel_id": CH, "target_ts": TS, "state": "committed",
                             "announcement_ts": out["announcement_ts"]}]
    joined = [d for d in row["destinations"]
              if d["kind"] == DEST_KIND_CORRECTION_ANNOUNCEMENT
              and d["state"] == "committed"
              and d["first_ts"] == out["announcement_ts"]]
    assert joined, row["destinations"]


async def test_turn_outcome_announcement_only_partial_keeps_the_error():
    world = _World(_plain_message())
    host = _host(world)
    inner_send = host.send_message

    async def send_then_race(*args, **kwargs):
        ts = await inner_send(*args, **kwargs)
        world.message["edited"] = {"ts": "1700008888.000000"}
        return ts

    host.send_message = send_then_race
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    row = _turn_outcome_row(turn)
    assert row["edits"] == [{"channel_id": CH, "target_ts": TS,
                             "state": "announcement_only",
                             "announcement_ts": out["announcement_ts"],
                             "error": "stale_target_after_announcement"}]


def test_destination_provenance_reads_turn_edits(monkeypatch):
    """§7's provenance timing. The loop appends the tool's OWN record only after dispatch
    returns, so the executor cannot persist it — the destination-provenance routine (clean exit
    AND finally) reads `turn.edits`: the edited TARGET ts only for state==committed
    (union-merged into its old row), the ANNOUNCEMENT ts for both states."""
    from message_processor.handlers.text import TextHandlerMixin

    monkeypatch.setattr(config, "enable_tool_provenance", True, raising=False)
    host = MagicMock()
    host._persist_destination_provenance = (
        TextHandlerMixin._persist_destination_provenance.__get__(host))
    host._persist_tool_provenance = MagicMock()
    turn = TurnRuntime()
    committed = EditRecord(channel_id=CH, target_ts=TS, announcement_ts="901.000100",
                           state=EDIT_STATE_COMMITTED)
    partial = EditRecord(channel_id=CH, target_ts="1700000061.000300",
                         announcement_ts="902.000100",
                         error="update_failed_after_announcement")
    turn.edits.extend([committed, partial])
    turn.mark_destination_committed(first_ts="901.000100",
                                    kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                    text="Correction …", channel_id=CH, thread_root_ts=ROOT)
    turn.mark_destination_committed(first_ts="902.000100",
                                    kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                    text="Correction …", channel_id=CH,
                                    thread_root_ts="1700000061.000300")
    # The loop's after-dispatch record — the exact entry an executor-side persist would omit.
    turn.note_tool_call({"name": "edit_own_message", "ok": True, "gist": ""})

    host._persist_destination_provenance(turn, None, None)

    calls = host._persist_tool_provenance.call_args_list
    persisted = {c.args[1] for c in calls}
    assert persisted == {TS, "901.000100", "902.000100"}, \
        "committed target + BOTH announcements; the partial's target is never persisted"
    by_ts = {c.args[1]: c for c in calls}
    assert by_ts[TS].args[2] == f"{CH}:{ROOT}", "keyed into the thread both surfaces live in"
    for call in calls:
        assert any(t.get("tool_name") == "edit_own_message" for t in call.args[3]), \
            "the tool's own after-dispatch record rides every row"


def test_empty_response_with_committed_announcement_classifies_as_reply():
    """main.py §7/§11.5: the disclosure is the turn's own words in the room, so a turn that
    committed one is a `reply` — not a `detached` producer. The override runs BEFORE the
    terminal no_reply ⇒ detached return, so the TERMINAL branch (real terminal metadata) is
    driven here, alongside the empty-response branch."""
    from main import ChatBotV2

    def _announced_turn():
        turn = TurnRuntime()
        turn.visible_action_committed = True
        turn.mark_destination_committed(first_ts="901.000100",
                                        kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                        text="Correction …", channel_id=CH,
                                        thread_root_ts=ROOT)
        return turn

    # The TERMINAL branch: the model ended with no_response_needed after the edit — the
    # committed disclosure outranks the visible_action_committed ⇒ detached return.
    terminal = SimpleNamespace(type="text",
                               metadata={"terminal_action": "no_reply",
                                         "terminal_reason": "no_value_to_add"},
                               content="")
    assert ChatBotV2._classify_visible_action(terminal, _announced_turn()) == "reply"
    # Terminal control: the same terminal turn with only a detached surface stays detached.
    control = TurnRuntime()
    control.visible_action_committed = True
    assert ChatBotV2._classify_visible_action(terminal, control) == "detached"

    # The empty-response branch (no terminal call): same override, same control.
    empty = SimpleNamespace(type="text", metadata={}, content="")
    assert ChatBotV2._classify_visible_action(empty, _announced_turn()) == "reply"
    control2 = TurnRuntime()
    control2.visible_action_committed = True
    assert ChatBotV2._classify_visible_action(empty, control2) == "detached"


def test_correction_announcements_are_excluded_from_channel_memory(monkeypatch):
    """main.py §7, beside the cross-thread exclusion: the disclosure is not this exchange's
    answer, and remembering it as one would store a conversation nobody had."""
    from main import ChatBotV2
    from message_processor.turn_runtime import DEST_KIND_REPLY

    monkeypatch.setattr(config, "enable_memory_extraction_fallback", True, raising=False)
    host = MagicMock()
    host._schedule_channel_memory = ChatBotV2._schedule_channel_memory.__get__(host)
    turn = TurnRuntime()
    turn.stream_build_present = True
    turn.mark_destination_committed(first_ts="900.000100", kind=DEST_KIND_REPLY,
                                    text="the real answer", channel_id=CH,
                                    thread_root_ts=ROOT)
    turn.mark_destination_committed(first_ts="901.000100",
                                    kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                    text="Correction to my earlier message: 48",
                                    channel_id=CH, thread_root_ts=ROOT)
    message = SimpleNamespace(channel_id=CH, text="what was asked")
    host._schedule_channel_memory(message, turn)
    exchange = host.processor.extract_channel_memory_from_exchange.call_args.args
    assert exchange[2] == "the real answer", "the announcement never enters the exchange"

    # A turn whose ONLY committed words are the disclosure schedules nothing at all.
    host2 = MagicMock()
    host2._schedule_channel_memory = ChatBotV2._schedule_channel_memory.__get__(host2)
    turn2 = TurnRuntime()
    turn2.stream_build_present = True
    turn2.mark_destination_committed(first_ts="901.000100",
                                     kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                     text="Correction to my earlier message: 48",
                                     channel_id=CH, thread_root_ts=ROOT)
    host2._schedule_channel_memory(SimpleNamespace(channel_id=CH, text="asked"), turn2)
    host2.processor.extract_channel_memory_from_exchange.assert_not_called()


async def test_falsey_and_nonstring_arguments_are_edit_failed_not_empty():
    """§11.14/§11.22 (review r4): TYPE-checked before coercion. A falsey non-string
    new_text ([] / 0 / {} / False) must be `edit_failed`, never laundered into
    `empty_new_text` by an `or ""`; a non-string correction_note (list / int) must be
    `edit_failed`, never str()-coerced into a POSTABLE repr. Slack untouched in every case."""
    world = _World(_plain_message())
    host = _host(world)
    for field, bad in (("new_text", []), ("new_text", 0), ("new_text", {}),
                       ("new_text", False), ("correction_note", ["wrong fact"]),
                       ("correction_note", 17)):
        turn = TurnRuntime()
        args = dict(ARGS)
        args[field] = bad
        out = await host.execute_edit_own_message(_ctx(host, turn=turn), args)
        assert out["ok"] is False and out["error"] == "edit_failed", (field, bad, out)
        assert turn.edits == [] and turn.visible_action_committed is False
    assert host.posts == [] and world.update_attempts == 0, "Slack untouched"


async def test_combined_callback_crash_and_post_accept_raise_still_accounts():
    """§11.19 (review r4): the COMBINED failure — the acceptance callback's append crashes
    (swallowed by the real `_note_first_accept`, so accepted['record'] never set) AND the
    send then raises after Slack accepted (post-acceptance `lease.commit()` here). The
    visible announcement must not escape accounting or read as `announcement_failed`: the
    classifier recovers the accepted ts from `meta_out['delivery'].first_ts`, repairs the
    record in-lock, and answers with the partial post-announcement contract."""

    class _FlakyEdits(list):
        raised = False

        def append(self, item):
            if not self.raised:
                self.raised = True
                raise RuntimeError("edits ledger broke at the seam")
            super().append(item)

    world = _World(_plain_message())
    host = _host(world)
    orig_send = host.send_message

    async def crashing_send(*a, **k):
        # The real seam runs (acceptance callback fired-and-swallowed, delivery reported),
        # THEN the send raises — the production window between acceptance and return.
        await orig_send(*a, **k)
        raise RuntimeError("post-acceptance bookkeeping broke")

    host.send_message = crashing_send
    turn = TurnRuntime()
    turn.edits = _FlakyEdits()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert turn.edits.raised, "the callback DID crash — half the premise"
    assert out["ok"] is False
    assert out["error"] == "edit_failed_after_announcement"
    assert out["announcement_posted"] is True and out["edited"] is False
    assert len(host.posts) == 1, "exactly the one visible announcement"
    assert world.update_attempts == 0, "the target was never edited"
    assert len(turn.edits) == 1, "the record was repaired despite both failures"
    assert turn.edits[0].announcement_ts == host.posts[0]["ts"]
    assert turn.edits[0].state == EDIT_STATE_ANNOUNCEMENT_ONLY
    assert turn.edits[0].error == "edit_failed_after_announcement"
    assert turn.visible_action_committed is True
    committed = [d for d in turn.destinations
                 if getattr(d, "first_ts", None) == host.posts[0]["ts"]]
    assert committed, "the CV10 join: a landed disclosure has a committed destination"
    assert committed[0].kind == DEST_KIND_CORRECTION_ANNOUNCEMENT


async def test_slacks_materialized_rich_text_block_is_the_plain_shape():
    """§11.25 (live-found): Slack MATERIALIZES one `rich_text` block on every plain
    chat.postMessage, so read-back never shows "no blocks" — the first live edit refused a
    plain one-liner as unsupported. One rich_text block classifies as plain and the edit
    proceeds text-only (Slack regenerates the mirror block on update)."""
    message = _plain_message()
    message["blocks"] = [{
        "type": "rich_text", "block_id": "07M",
        "elements": [{"type": "rich_text_section",
                      "elements": [{"type": "text",
                                    "text": "The old cap is 36000 dollars."}]}],
    }]
    world = _World(message)
    host = _host(world)
    turn = TurnRuntime()
    out = await host.execute_edit_own_message(_ctx(host, turn=turn), dict(ARGS))
    assert out["ok"] is True and out["edited"] is True
    assert world.update_attempts == 1
    assert "blocks" not in world.update_kwargs[0], "text-only update — Slack remints the block"
    # Two rich_text blocks are NOT the plain shape.
    message2 = _plain_message()
    message2["blocks"] = [dict(message["blocks"][0]), dict(message["blocks"][0])]
    world2 = _World(message2)
    host2 = _host(world2)
    out2 = await host2.execute_edit_own_message(_ctx(host2, turn=TurnRuntime()), dict(ARGS))
    assert out2["ok"] is False and out2["error"] == "unsupported_message_shape"


async def test_the_refusal_shim_logs_the_error_name_and_translates_nothing():
    """§11.25 observability: every refusal names itself in the log (a quiet refusal cost a
    live debugging round), successful results log nothing, and control-flow exceptions pass
    through untranslated."""
    world = _World(_plain_message())
    host = _host(world)
    host.log_info = MagicMock()
    out = await host.execute_edit_own_message(
        _ctx(host, turn=TurnRuntime()),
        {"message_ts": "1700000099.999999", "new_text": NEW_TEXT, "correction_note": NOTE})
    assert out["error"] == "unauthorized_target"
    host.log_info.assert_called_once()
    assert "unauthorized_target" in host.log_info.call_args[0][0]

    host.log_info.reset_mock()
    ok = await host.execute_edit_own_message(_ctx(host, turn=TurnRuntime()), dict(ARGS))
    assert ok["ok"] is True
    for call in host.log_info.call_args_list:
        assert "refused" not in call[0][0]

    async def raising_inner(ctx, args):
        raise StaleSendSuppressed(surface="edit_announcement")

    host._execute_edit_own_message = raising_inner
    with pytest.raises(StaleSendSuppressed):
        await host.execute_edit_own_message(_ctx(host, turn=TurnRuntime()), dict(ARGS))
