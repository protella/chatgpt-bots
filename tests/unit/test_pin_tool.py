"""PIN §3/§4/§5 — pin_message: pin and unpin ONE message of this conversation, on request.

The tool is deliberately small, and everything interesting about it is what it refuses to do:
it type-checks every model-supplied value before any coercion (so a malformed argument never
reaches Slack), it serializes same-message calls on the edit transaction lock, and when Slack
gives an answer that cannot be trusted it READS the pins back exactly once rather than guessing
or retrying — an unreadable state is `outcome_unknown`, never an invented success.

What this file drives, against the REAL executor with a mocked Slack world:

* the §3 schema, whole-dict;
* every §4 refusal, each proven to have touched no Slack method at all;
* the §4.6 outcome rows, including the idempotent ones Slack reports as errors;
* the §4.7 reconciliation contract: its four action/state outcomes, each of its four triggers,
  exactly one read and no second mutation, and the difference between a malformed BODY (which
  is unknowable) and a malformed ITEM (which is merely skipped);
* the §4.4 lock: one in-flight Slack call per ts, and the map entry pruned on both exits.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from slack_sdk.errors import SlackApiError

from slack_client import messaging
from slack_client.messaging import SlackMessagingMixin
from message_processor.tool_registry import SURFACE_CHANNEL, ToolContext, ToolRegistry

TEAM = "T1"
CH = "C1"
DM = "D1"
TS = "1700000060.000200"
REPLY_TS = "1700000099.000700"
ARGS = {"action": "pin", "ts": TS}


# ------------------------------------------------------------------------------- harness

def _host():
    """The REAL schema getter + shim + executor on a mocked Slack transport."""
    s = MagicMock()
    s.self_team_id = TEAM
    for name in ("get_pin_message_tool_schema", "execute_pin_message", "_execute_pin_message"):
        setattr(s, name, getattr(SlackMessagingMixin, name).__get__(s))
    s.app.client.pins_add = AsyncMock(return_value={"ok": True})
    s.app.client.pins_remove = AsyncMock(return_value={"ok": True})
    s.app.client.pins_list = AsyncMock(return_value={"ok": True, "items": []})
    return s


def _ctx(channel=CH):
    return ToolContext(channel_id=channel, thread_ts="1700000000.000100", client=None)


def _api_error(error="message_not_found"):
    """A SlackApiError carrying a response the executor can read `error` off — the SDK's own
    responses answer `.get`, which is the only thing §4.6 relies on."""
    return SlackApiError("boom", {"ok": False, "error": error})


def _pins(*items):
    return {"ok": True, "items": list(items)}


def _pinned(ts=TS):
    return {"type": "message", "created": 1700000061, "message": {"ts": ts, "text": "hi"}}


def _no_slack_calls(host):
    return (host.app.client.pins_add.await_count == 0
            and host.app.client.pins_remove.await_count == 0
            and host.app.client.pins_list.await_count == 0)


# ====================================================================== §3: schema + registration

def test_schema_is_exactly_the_spec_dict():
    schema = _host().get_pin_message_tool_schema()
    assert schema == {
        "type": "function",
        "name": "pin_message",
        "description": (
            "Pin a message to this conversation's pinned items, or unpin one — only when "
            "someone here asks you to. The target must be a message in THIS conversation. "
            "Slack shows who pinned what, and a pin is a shared surface: never pin on your "
            "own initiative, and never pin to make a point. Never call this just to "
            "check or refresh a pin — if nothing needs to change, make no call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["pin", "unpin"],
                           "description": ("pin adds the message to the pinned items; unpin "
                                           "removes it.")},
                "ts": {"type": "string",
                       "description": ("Timestamp (ts) of the target message in THIS "
                                       "conversation, exactly as an id you can see this "
                                       "turn: from the turn coordinates, a message header "
                                       "in the stream, or a tool result about this "
                                       "conversation. Never guess or derive one. A thread "
                                       "reply's ts pins that reply itself.")},
            },
            "required": ["action", "ts"],
        },
    }
    # No model-facing channel parameter: the conversation comes from the ToolContext, which is
    # what makes Slack's `message_not_found` a structural confinement rather than a hint.
    assert set(schema["parameters"]["properties"]) == {"action", "ts"}
    assert schema["parameters"]["required"] == ["action", "ts"]


def test_registered_on_both_surfaces_with_no_gate():
    from slack_client.base import SlackBot

    src = inspect.getsource(SlackBot._build_tool_registry)
    assert ("registry.register(self.get_pin_message_tool_schema(), self.execute_pin_message)"
            in src)

    registry = ToolRegistry()
    registry.register(_host().get_pin_message_tool_schema(), AsyncMock())
    assert "pin_message" in {s["name"] for s in registry.schemas()}
    assert "pin_message" in {s["name"] for s in registry.schemas(surface=SURFACE_CHANNEL)}


def test_the_tool_is_budgeted_not_free():
    """§2: free tools are bookkeeping. This one mutates a shared surface, so it must never ride
    a `free_tools` set — it spends a round like any other real tool."""
    import message_processor.handlers.text as text_module

    for line in inspect.getsource(text_module).splitlines():
        if "free_tools=" in line:
            assert "pin_message" not in line


# ============================================================================ §4.5/§4.6: outcomes

async def test_pin_happy_path():
    host = _host()
    out = await host.execute_pin_message(_ctx(), {"action": "pin", "ts": TS})
    assert out == {"ok": True, "action": "pin", "ts": TS}
    host.app.client.pins_add.assert_awaited_once_with(channel=CH, timestamp=TS)
    assert host.app.client.pins_remove.await_count == 0


async def test_unpin_happy_path():
    host = _host()
    out = await host.execute_pin_message(_ctx(), {"action": "unpin", "ts": TS})
    assert out == {"ok": True, "action": "unpin", "ts": TS}
    host.app.client.pins_remove.assert_awaited_once_with(channel=CH, timestamp=TS)
    assert host.app.client.pins_add.await_count == 0


@pytest.mark.parametrize("action, error, expected", [
    ("pin", "already_pinned", {"ok": True, "action": "pin", "ts": TS, "note": "already pinned"}),
    ("unpin", "no_pin", {"ok": True, "action": "unpin", "ts": TS, "note": "was not pinned"}),
    ("unpin", "not_pinned", {"ok": True, "action": "unpin", "ts": TS, "note": "was not pinned"}),
])
async def test_the_goal_state_already_holds(action, error, expected):
    """Slack reports "it is already like that" as an error; the caller's intent is satisfied,
    so these are successes with a note — and no read is needed to say so."""
    host = _host()
    getattr(host.app.client, f"pins_{'add' if action == 'pin' else 'remove'}").side_effect = \
        _api_error(error)
    out = await host.execute_pin_message(_ctx(), {"action": action, "ts": TS})
    assert out == expected
    assert host.app.client.pins_list.await_count == 0


async def test_a_named_api_error_is_returned_as_itself():
    host = _host()
    host.app.client.pins_add.side_effect = _api_error("message_not_found")
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": False, "error": "message_not_found"}
    assert host.app.client.pins_list.await_count == 0


@pytest.mark.parametrize("response", [None, {"ok": False}, {"ok": False, "error": 500},
                                      {"ok": False, "error": ""}])
async def test_an_api_error_with_no_readable_name_reconciles(response):
    """§4.6: a SlackApiError we cannot NAME may still have landed, so it is ambiguous — the one
    thing it must never be is a flat failure."""
    host = _host()
    host.app.client.pins_add.side_effect = SlackApiError("boom", response)
    host.app.client.pins_list.return_value = _pins(_pinned())
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": True, "action": "pin", "ts": TS,
                   "note": "confirmed pinned after an ambiguous response"}
    assert host.app.client.pins_list.await_count == 1


async def test_an_unexpected_exception_is_pin_failed_and_logged_once():
    host = _host()
    host.app.client.pins_add.side_effect = RuntimeError("transport exploded")
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": False, "error": "pin_failed"}
    assert host.log_error.call_count == 1
    assert host.log_info.call_count == 0


async def test_cancellation_propagates():
    host = _host()
    host.app.client.pins_add.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await host.execute_pin_message(_ctx(), dict(ARGS))


# ============================================================== §4.1/§4.2/§4.3: refusals, no call

@pytest.mark.parametrize("args, error", [
    ({"action": 7, "ts": TS}, "invalid_action"),
    ({"action": None, "ts": TS}, "invalid_action"),
    ({"action": "", "ts": TS}, "invalid_action"),
    ({"action": "   ", "ts": TS}, "invalid_action"),
    ({"action": "delete", "ts": TS}, "invalid_action"),
    ({"action": "pin", "ts": 1700000060.0002}, "invalid_ts"),
    ({"action": "pin", "ts": None}, "invalid_ts"),
    ({"action": "pin", "ts": "yesterday"}, "invalid_ts"),
    ({"action": "pin", "ts": ""}, "invalid_ts"),
])
async def test_malformed_arguments_are_refused_before_any_slack_call(args, error):
    host = _host()
    out = await host.execute_pin_message(_ctx(), args)
    assert out == {"ok": False, "error": error}
    assert _no_slack_calls(host)


async def test_no_channel_context_is_refused_before_any_slack_call():
    host = _host()
    out = await host.execute_pin_message(SimpleNamespace(channel_id=None), dict(ARGS))
    assert out == {"ok": False, "error": "no_channel_context"}
    assert _no_slack_calls(host)


async def test_an_epoch_refusal_is_not_logged_a_second_time(monkeypatch):
    """§2: the epoch helper already warns. The shim stays quiet about this one so a fenced
    workspace produces exactly one line per refused write, not two."""
    seen = []

    def _refused(client, channel_id, site):
        seen.append(site)
        client.log_warning(f"Epoch fence refused {site}: fenced")
        return True

    monkeypatch.setattr(messaging, "_epoch_refused", _refused)
    host = _host()
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": False, "error": "workspace_unavailable"}
    assert seen == ["pins_add"]
    assert _no_slack_calls(host)
    assert host.log_warning.call_count == 1   # the helper's, and only the helper's
    assert host.log_info.call_count == 0

    host = _host()
    await host.execute_pin_message(_ctx(), {"action": "unpin", "ts": TS})
    assert seen[-1] == "pins_remove"


async def test_the_shim_names_an_ordinary_refusal_exactly_once():
    host = _host()
    host.app.client.pins_add.side_effect = _api_error("message_not_found")
    await host.execute_pin_message(_ctx(), dict(ARGS))
    assert host.log_info.call_count == 1
    assert "message_not_found" in host.log_info.call_args[0][0]
    assert host.log_error.call_count == 0


# ================================================================================ targets

async def test_a_thread_reply_ts_reaches_slack_unchanged():
    """A reply's own ts pins THAT reply — the executor never substitutes the thread root."""
    host = _host()
    out = await host.execute_pin_message(_ctx(), {"action": "pin", "ts": f"  {REPLY_TS}  "})
    assert out == {"ok": True, "action": "pin", "ts": REPLY_TS}
    host.app.client.pins_add.assert_awaited_once_with(channel=CH, timestamp=REPLY_TS)


async def test_a_dm_conversation_executes_normally():
    host = _host()
    out = await host.execute_pin_message(_ctx(channel=DM), dict(ARGS))
    assert out == {"ok": True, "action": "pin", "ts": TS}
    host.app.client.pins_add.assert_awaited_once_with(channel=DM, timestamp=TS)


# ========================================================================= §4.7: reconciliation

@pytest.mark.parametrize("trigger", [
    _api_error("internal_error"),
    _api_error("fatal_error"),
    asyncio.TimeoutError(),
    aiohttp.ClientError(),
])
async def test_every_ambiguous_trigger_reconciles(trigger):
    host = _host()
    host.app.client.pins_add.side_effect = trigger
    host.app.client.pins_list.return_value = _pins(_pinned())
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": True, "action": "pin", "ts": TS,
                   "note": "confirmed pinned after an ambiguous response"}


@pytest.mark.parametrize("action, items, expected", [
    ("pin", (_pinned(),), {"ok": True, "action": "pin", "ts": TS,
                           "note": "confirmed pinned after an ambiguous response"}),
    ("pin", (), {"ok": False, "error": "pin_failed"}),
    ("unpin", (), {"ok": True, "action": "unpin", "ts": TS,
                   "note": "confirmed removed after an ambiguous response"}),
    ("unpin", (_pinned(),), {"ok": False, "error": "unpin_failed"}),
])
async def test_the_four_reconciled_outcomes(action, items, expected):
    host = _host()
    mutation = host.app.client.pins_add if action == "pin" else host.app.client.pins_remove
    mutation.side_effect = _api_error("internal_error")
    host.app.client.pins_list.return_value = _pins(*items)
    out = await host.execute_pin_message(_ctx(), {"action": action, "ts": TS})
    assert out == expected
    # EXACTLY one read, and never a second attempt at the mutation itself.
    assert host.app.client.pins_list.await_count == 1
    assert mutation.await_count == 1


async def test_reconciliation_ignores_pins_belonging_to_other_messages():
    host = _host()
    host.app.client.pins_add.side_effect = _api_error("internal_error")
    host.app.client.pins_list.return_value = _pins(_pinned(ts=REPLY_TS))
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": False, "error": "pin_failed"}


@pytest.mark.parametrize("body", [
    RuntimeError("pins.list is down"),
    {"ok": False, "error": "ratelimited"},
    {"ok": True},
    {"ok": True, "items": "nope"},
    None,
    "not a response at all",
])
async def test_an_unreadable_answer_is_outcome_unknown(body):
    """A body we cannot parse is the one case with no honest verdict. Never a guess, and never
    a retry: the read gets one chance, and failing it says so."""
    host = _host()
    host.app.client.pins_add.side_effect = _api_error("internal_error")
    if isinstance(body, Exception):
        host.app.client.pins_list.side_effect = body
    else:
        host.app.client.pins_list.return_value = body
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": False, "error": "outcome_unknown"}
    assert host.app.client.pins_list.await_count == 1
    assert host.app.client.pins_add.await_count == 1


async def test_a_malformed_items_entry_is_skipped_not_fatal():
    """The distinction §4.7 draws: a malformed BODY is unknowable, a malformed ENTRY is junk
    beside the real answer — skipping it is what the history walk does with the same shape."""
    host = _host()
    host.app.client.pins_add.side_effect = _api_error("internal_error")
    host.app.client.pins_list.return_value = _pins(
        "junk", {"type": "file"}, {"message": None}, {"message": "nope"}, _pinned())
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": True, "action": "pin", "ts": TS,
                   "note": "confirmed pinned after an ambiguous response"}


async def test_reconciliation_reads_a_real_slack_response_object():
    """The SDK returns an AsyncSlackResponse, which is NOT a dict — reading it as one would
    turn every real reconciliation into `outcome_unknown`."""
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

    host = _host()
    host.app.client.pins_add.side_effect = _api_error("internal_error")
    host.app.client.pins_list.return_value = AsyncSlackResponse(
        client=None, http_verb="GET", api_url="https://slack.com/api/pins.list",
        req_args={}, data={"ok": True, "items": [_pinned()]}, headers={}, status_code=200)
    out = await host.execute_pin_message(_ctx(), dict(ARGS))
    assert out == {"ok": True, "action": "pin", "ts": TS,
                   "note": "confirmed pinned after an ambiguous response"}


# ================================================================================ §4.4: the lock

async def test_two_calls_at_one_ts_never_overlap_at_slack():
    host = _host()
    state = {"in_flight": 0, "peak": 0}

    async def _slow_add(**kwargs):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return {"ok": True}

    host.app.client.pins_add = AsyncMock(side_effect=_slow_add)
    results = await asyncio.gather(
        host.execute_pin_message(_ctx(), dict(ARGS)),
        host.execute_pin_message(_ctx(), dict(ARGS)))
    assert state["peak"] == 1
    assert all(r == {"ok": True, "action": "pin", "ts": TS} for r in results)


async def test_authorization_is_checked_at_the_mutation_point_not_at_entry(monkeypatch):
    """Diff-review r1: a turn parked behind the keyed lock must not spend an authorization
    that went stale while it waited — the epoch check runs INSIDE the lock, at the mutation
    point, so a fence that closed during the wait still refuses the write."""
    fenced = {"refused": False}
    monkeypatch.setattr(messaging, "_epoch_refused",
                        lambda client, channel_id, site: fenced["refused"])
    host = _host()
    key = (TEAM, CH, TS)
    lock = messaging._edit_transaction_lock(*key)
    await lock.acquire()
    try:
        task = asyncio.create_task(host.execute_pin_message(_ctx(), dict(ARGS)))
        await asyncio.sleep(0.01)   # parked on the lock; the guard would still authorize
        fenced["refused"] = True    # the epoch closes while it waits
    finally:
        lock.release()
    out = await task
    assert out == {"ok": False, "error": "workspace_unavailable"}
    assert _no_slack_calls(host)


async def test_the_lock_entry_is_pruned_on_both_exits():
    key = (TEAM, CH, TS)
    host = _host()
    await host.execute_pin_message(_ctx(), dict(ARGS))
    assert key not in messaging._EDIT_TRANSACTION_LOCKS

    host = _host()
    host.app.client.pins_add.side_effect = RuntimeError("transport exploded")
    with pytest.raises(RuntimeError):
        await host._execute_pin_message(_ctx(), dict(ARGS))
    assert key not in messaging._EDIT_TRANSACTION_LOCKS
