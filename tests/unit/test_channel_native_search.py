"""SLACK_CHANNEL_NATIVE_SEARCH (spec §D): an @mentioned channel turn's search_slack merges
Slack's own index (assistant.search.context) into the in-channel scan.

Eligibility, order, the strict-before-trigger fence, self-hit drop, dedupe, the global limit,
read-only index hits, failure ⇒ scan unchanged, and flag-off / DM bytes pinned.
"""
import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from config import config
from message_processor import prompts
from slack_client import search_tool as search_mod
from slack_client.utilities import SlackUtilitiesMixin
from message_processor.tool_registry import ToolContext
from message_processor.fail_cards import FAIL_RETRY_RERUN
from tests.unit.test_search_tool import CH, _FakeSlack, _m, _ScanBot, _scan_ctx

TRIGGER = "500.000000"
OTHER = "C0OTHERPUB1"


class _IndexSlack(_FakeSlack):
    """The scan's fake plus an assistant index that answers from a fixed hit list."""

    def __init__(self, *, index_hits=None, index_raises=None, index_delay=0.0, **kw):
        kw.setdefault("info", {"is_channel": True, "is_private": False, "is_member": True,
                               "is_mpim": False, "is_im": False})
        super().__init__(**kw)
        self.index_hits = list(index_hits or [])
        self.index_raises = index_raises
        self.index_delay = index_delay
        self.index_calls = []

    async def api_call(self, method, data=None, **kwargs):
        self.index_calls.append((method, dict(data or {})))
        if self.index_delay:
            await asyncio.sleep(self.index_delay)
        if self.index_raises is not None:
            raise self.index_raises
        return {"ok": True, "results": {"messages": list(self.index_hits)}}


class _NativeBot(_ScanBot):
    is_own_message = SlackUtilitiesMixin.is_own_message

    def __init__(self, client):
        super().__init__(client)
        self.bot_id = "B_BOT"
        self.app_id = "A_BOT"
        self.resolve_usernames = AsyncMock(
            side_effect=lambda ids, _client: {i: f"name-{i}" for i in ids})


def _hit(channel_id, ts, text, *, author="U_HUMAN", **extra):
    hit = {"channel_id": channel_id, "message_ts": ts, "content": text,
           "author_user_id": author, "team_id": "T_TEST",
           "permalink": f"https://example.slack.com/archives/{channel_id}/p{ts.replace('.', '')}"}
    hit.update(extra)
    return hit


def _eligible_ctx(**meta_overrides):
    meta = {"wake_source": "app_mention", "action_token": "tok-mention", "ts": TRIGGER}
    meta.update(meta_overrides)
    ctx = _scan_ctx(trigger_ts=TRIGGER, message=SimpleNamespace(metadata=meta))
    ctx.action_token = "tok-mention"
    return ctx


def _scan_client(**kw):
    # One scan hit at 300 in the current channel.
    return _IndexSlack(history=[{"messages": [_m("300.000000", "cert renewal for Kestwood")]}],
                       **kw)


async def _run(client, ctx=None, *, limit=None):
    bot = _NativeBot(client)
    ctx = ctx if ctx is not None else _eligible_ctx()
    args = {"query": "cert renewal"}
    if limit is not None:
        args["limit"] = limit
    payload = await bot.execute_search_tool(ctx, args)
    return bot, ctx, payload


@pytest.fixture
def native_on(monkeypatch):
    monkeypatch.setattr(config, "slack_channel_native_search", True)


# --- merge --------------------------------------------------------------------------------

async def test_an_eligible_mention_merges_index_hits_after_the_scan(native_on):
    client = _scan_client(index_hits=[
        _hit(OTHER, "400.000000", "cert renewal notes", channel_name="ops-notes",
             thread_ts="390.000000")])
    _bot, ctx, payload = await _run(client)
    assert payload["ok"] is True
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [
        (CH, "300.000000"), (OTHER, "400.000000")]
    index_hit = payload["results"][1]
    assert index_hit["channel_name"] == "ops-notes"
    assert index_hit["author"] == "name-U_HUMAN"
    assert "channel_name" not in payload["results"][0]
    assert payload["count"] == 2
    assert payload["coverage"]["index_hits"] == 1
    # Workspace reach, on the mention's own token.
    method, data = client.index_calls[0]
    assert method == "assistant.search.context"
    assert data["action_token"] == "tok-mention"
    # Index hits never stage write authority: the thread_ts it carries enrolled nothing.
    assert index_hit["thread_ts"] == "390.000000"
    assert ctx.tool_flight.staged_roots == []


async def test_the_fence_drops_late_and_malformed_hits_before_author_lookup(native_on):
    client = _scan_client(index_hits=[
        _hit(OTHER, "600.000000", "cert renewal later", author="U_LATE"),
        _hit(OTHER, TRIGGER, "cert renewal same instant", author="U_SAME"),
        _hit(OTHER, "not-a-ts", "cert renewal garbled", author="U_BAD"),
        {"channel_id": OTHER, "content": "cert renewal no ts", "author_user_id": "U_NONE"},
        _hit(OTHER, "450.000000", "cert renewal kept", author="U_KEPT"),
    ])
    bot, _ctx, payload = await _run(client)
    assert [r["ts"] for r in payload["results"]] == ["300.000000", "450.000000"]
    looked_up = {i for call in bot.resolve_usernames.await_args_list for i in call.args[0]}
    assert looked_up.isdisjoint({"U_LATE", "U_SAME", "U_BAD", "U_NONE"})


async def test_the_bots_own_hits_are_dropped(native_on):
    client = _scan_client(index_hits=[
        _hit(OTHER, "410.000000", "cert renewal by me", author="U_BOT"),
        _hit(OTHER, "420.000000", "cert renewal by my app", author=None, bot_id="B_BOT"),
        _hit(OTHER, "425.000000", "cert renewal by my app id", author=None,
             api_app_id="A_BOT"),
        _hit(OTHER, "430.000000", "cert renewal by a person"),
    ])
    _bot, _ctx, payload = await _run(client)
    assert [r["ts"] for r in payload["results"]] == ["300.000000", "430.000000"]


async def test_index_hits_dedupe_against_the_scan_by_channel_and_ts(native_on):
    client = _scan_client(index_hits=[
        _hit(CH, "300.000000", "cert renewal for Kestwood"),     # the scan already has it
        _hit(OTHER, "300.000000", "cert renewal elsewhere"),     # same ts, other channel: kept
        _hit(OTHER, "300.000000", "cert renewal elsewhere"),     # repeated in the index itself
    ])
    _bot, _ctx, payload = await _run(client)
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [
        (CH, "300.000000"), (OTHER, "300.000000")]
    assert payload["coverage"]["index_hits"] == 1


async def test_the_global_limit_keeps_scan_hits_first(native_on):
    client = _scan_client(index_hits=[
        _hit(OTHER, "401.000000", "cert renewal a"),
        _hit(OTHER, "402.000000", "cert renewal b"),
        _hit(OTHER, "403.000000", "cert renewal c"),
    ])
    _bot, _ctx, payload = await _run(client, limit=2)
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [
        (CH, "300.000000"), (OTHER, "401.000000")]
    assert payload["count"] == 2


async def test_a_hit_whose_channel_identities_disagree_is_dropped(native_on):
    private = "C0PRIVATE01"
    client = _scan_client(index_hits=[
        # Current channel at top level, another channel nested and in the link: dropped.
        _hit(CH, "410.000000", "cert renewal leak", channel={"id": private, "name": "secret"},
             permalink=f"https://slack/archives/{private}/p410000000"),
        # Link alone disagrees: dropped.
        _hit(OTHER, "420.000000", "cert renewal link",
             permalink=f"https://slack/archives/{private}/p420000000"),
        # cid alone disagrees: dropped.
        _hit(OTHER, "425.000000", "cert renewal cid",
             permalink=f"https://slack/archives/{OTHER}/p425000000?cid={private}"),
        # Everything agrees: kept, labelled from the same channel.
        _hit(OTHER, "430.000000", "cert renewal clean", channel={"id": OTHER, "name": "ops"},
             permalink=f"https://slack/archives/{OTHER}/p430000000?cid={OTHER}"),
    ])
    _bot, _ctx, payload = await _run(client)
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [
        (CH, "300.000000"), (OTHER, "430.000000")]
    assert payload["results"][1]["channel_name"] == "ops"


async def test_a_withheld_copy_does_not_shadow_a_later_valid_copy(native_on):
    client = _scan_client(index_hits=[
        _hit(OTHER, "440.000000", "cert renewal", context_team_id="T_FOREIGN"),  # contradictory
        _hit(OTHER, "440.000000", "cert renewal"),
    ])
    _bot, _ctx, payload = await _run(client)
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [
        (CH, "300.000000"), (OTHER, "440.000000")]


# --- failure ⇒ scan unchanged ---------------------------------------------------------------

async def _scan_only_payload():
    _bot, _ctx, payload = await _run(_scan_client(), _eligible_ctx(wake_source="ambient"))
    return payload


async def test_an_index_failure_returns_the_scan_unchanged(native_on):
    client = _scan_client(index_raises=RuntimeError("index down"))
    _bot, _ctx, payload = await _run(client)
    assert client.index_calls
    assert payload == await _scan_only_payload()
    assert "index_hits" not in payload["coverage"]


async def test_an_index_timeout_returns_the_scan_unchanged(native_on, monkeypatch):
    monkeypatch.setattr(search_mod, "_NATIVE_INDEX_BUDGET_SECONDS", 0.05)
    client = _scan_client(index_delay=2.0, index_hits=[_hit(OTHER, "400.000000", "cert renewal")])
    _bot, _ctx, payload = await _run(client)
    assert payload == await _scan_only_payload()


async def test_a_refused_history_read_never_reaches_the_index(native_on, monkeypatch):
    response = AsyncMock()
    response.get = lambda k, d=None: {"error": "not_in_channel"}.get(k, d)
    response.headers = {}
    client = _IndexSlack(
        history=[{"raise": SlackApiError(message="not_in_channel", response=response)}],
        index_hits=[_hit(CH, "400.000000", "cert renewal")])
    monkeypatch.setattr(config, "fetch_retry_attempts", 1)
    _bot, _ctx, payload = await _run(client)
    assert payload["coverage"]["stopped_reason"].startswith("history_error")
    assert client.index_calls == []
    assert payload["results"] == []


async def test_the_index_never_outlives_the_tool_timeout(native_on, monkeypatch):
    # No tool budget left after the margin: the index is not attempted at all.
    monkeypatch.setattr(config, "search_tool_timeout_seconds", 0.5)
    client = _scan_client(index_hits=[_hit(OTHER, "400.000000", "cert renewal")])
    _bot, _ctx, payload = await _run(client)
    assert client.index_calls == []
    assert "index_hits" not in payload["coverage"]
    # A little left: the index gets only that, not the full 5 s, and the scan survives it.
    monkeypatch.setattr(config, "search_tool_timeout_seconds", 1.3)
    client = _scan_client(index_delay=3.0, index_hits=[_hit(OTHER, "400.000000", "cert renewal")])
    started = asyncio.get_running_loop().time()
    _bot, _ctx, payload = await _run(client)
    assert asyncio.get_running_loop().time() - started < 1.3
    assert client.index_calls
    assert [r["ts"] for r in payload["results"]] == ["300.000000"]
    assert "index_hits" not in payload["coverage"]


# --- eligibility ⇒ scan only ------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    "no_token", "catch_up_batch", "one_message_catch_up", "fail_retry_rerun",
    "carried_batch_history", "synthetic_edit_dispatch",
    "ambient_wake", "inherited_token", "mpim"])
async def test_an_ineligible_turn_runs_the_scan_only(native_on, monkeypatch, case):
    ctx = _eligible_ctx()
    info = None
    if case == "no_token":
        ctx.action_token = None
        ctx.message.metadata["action_token"] = None
    elif case == "catch_up_batch":
        ctx.message.metadata["queued_batch_size"] = 3
    elif case == "one_message_catch_up":
        ctx.message.metadata["queued_batch_size"] = 1
    elif case == "fail_retry_rerun":
        ctx.message.metadata[FAIL_RETRY_RERUN] = True
    elif case == "carried_batch_history":
        ctx.message.metadata["batched_history"] = [("450.000000", "earlier")]
    elif case == "synthetic_edit_dispatch":
        # Not Slack-delivered: no attestation, so the scan walks membership (stubbed as a member).
        ctx.origin_membership_attested = False
        monkeypatch.setattr(_NativeBot, "_requester_conversations",
                            AsyncMock(return_value=({CH}, True)), raising=False)
    elif case == "ambient_wake":
        ctx.message.metadata["wake_source"] = "ambient"
    elif case == "inherited_token":
        ctx.message.metadata["action_token"] = "tok-someone-else"
    elif case == "mpim":
        info = {"is_mpim": True, "is_group": True, "is_private": True, "is_member": True}
    client = _scan_client(index_hits=[_hit(OTHER, "400.000000", "cert renewal")],
                          **({"info": info} if info else {}))
    _bot, _ctx, payload = await _run(client, ctx)
    assert client.index_calls == []
    assert [r["ts"] for r in payload["results"]] == ["300.000000"]
    assert "index_hits" not in payload["coverage"]


# --- flag off and DM bytes, pinned ------------------------------------------------------------

_CHANNEL_DESCRIPTION_FLAG_OFF = (
    "Keyword search of THIS channel — the messages it still holds, including the replies "
    "inside its threads. Use distinctive words you expect to appear in the message itself "
    "(a name, a figure, a project word), not a paraphrase of the question: this matches "
    "words, not meaning, and a message qualifies on any one of them. It reads this channel "
    "only, never the wider workspace, and never anything posted after the message you are "
    "answering. Each result carries its ts and, when it sits in a thread, its thread_ts — a "
    "result with no thread_ts is not a thread, so call fetch_thread_messages if you need to "
    "read or answer in one. The result also reports how much of the channel it managed to "
    "read: partial coverage is not the same as no matches, and a recent reply under a "
    "thread root older than that span is not reachable this way.")

# sha256 of the pre-feature channel etiquette and window guidance (git HEAD before §D).
_CHANNEL_GUIDANCE_SHA = "13388ef46de3e0c42a07055992a7f982f049a42d852660b81f9cfcf21dbadfb9"
_WINDOW_GUIDANCE_SHA = "874580568fc0c2266f9a1a2011a3ccb14c6925b293b0b317ac28331e152d4fb6"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def test_flag_off_bytes_are_unchanged():
    assert config.slack_channel_native_search is False
    bot = _NativeBot(_scan_client())
    assert bot.get_search_tool_channel_schema()["description"] == _CHANNEL_DESCRIPTION_FLAG_OFF
    assert _sha(prompts.CHANNEL_LOCAL_TOOLS_GUIDANCE) == _CHANNEL_GUIDANCE_SHA
    assert _sha(prompts.render_window_guidance()) == _WINDOW_GUIDANCE_SHA
    # An otherwise eligible mention never touches the index.
    client = _scan_client(index_hits=[_hit(OTHER, "400.000000", "cert renewal")])
    _bot, _ctx, payload = await _run(client)
    assert client.index_calls == []
    assert [(r["channel"], r["ts"]) for r in payload["results"]] == [(CH, "300.000000")]
    assert "index_hits" not in payload["coverage"]


async def test_dm_bytes_are_unchanged_with_the_flag_on(native_on):
    bot = _NativeBot(_IndexSlack(index_hits=[
        _hit("C0DMSRC0001", "400.000000", "cert renewal", channel_name="ops-notes")]))
    assert bot.get_search_tool_schema()["description"] == (
        "Search Slack messages the bot is allowed to see (workspace-wide or the "
        "current channel). Use for finding older discussions, decisions, or context "
        "outside the current thread; prefer fetch_thread_messages/fetch_channel_history "
        "for things in the current conversation. Each result carries its source channel "
        "id; pass that to resolve_channel_name to show the channel's name.")
    ctx = ToolContext(channel_id="D0DMCHAN001", thread_ts="1.0", trigger_ts=TRIGGER,
                      action_token="tok-dm", is_dm=True)
    payload = await bot.execute_search_tool(ctx, {"query": "cert renewal"})
    assert payload == {
        "ok": True, "query": "cert renewal", "scope": "workspace", "count": 1,
        "results": [{"channel": "C0DMSRC0001", "ts": "400.000000", "author": "name-U_HUMAN",
                     "text": "cert renewal",
                     "permalink": "https://example.slack.com/archives/C0DMSRC0001/p400000000"}]}


def test_flag_on_states_the_wider_reach_in_both_places(native_on):
    bot = _NativeBot(_scan_client())
    description = bot.get_search_tool_channel_schema()["description"]
    assert "other public channels" in description
    assert "never the wider workspace" not in description
    guidance = prompts._derive_channel_tools_guidance(native_search=True)
    assert "its results can also include other public channels" in guidance
    assert "it cannot look into other channels from there" not in guidance
    assert "other public channels" in prompts.render_window_guidance()
    # The DM etiquette is untouched.
    assert "it cannot look into other channels from there" in prompts.LOCAL_TOOLS_GUIDANCE
