"""User-scoped authorization for every channel-read tool (2026-07 security fix).

The bug this locks down: the old gate asked only whether the BOT was in a channel. So a
private channel the bot had been added to leaked to anyone who could name it, and — because
`conversations.info` on an IM carries no `is_private` key at all — `ch.get("is_private", False)`
classified *someone else's DM with the bot* as a public channel and handed over its contents.

The policy now: content is returned ONLY when BOTH the bot AND the requesting user are in the
target conversation, decided from Slack's own booleans (never an id prefix — private channels
here carry C- prefixes), with every denial rendered identically so the refusal can't be used as
an existence oracle.

Also covers `lookup_channel` (name → id, requester-scoped) and the resolver's skip-list for ids
users.info can never resolve.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from config import config
from slack_client.channel_lookup_tool import SlackChannelLookupToolMixin
from slack_client.history_tool import (ACCESS_DENIED_MESSAGE, CHANNEL_READ_TOOLS,
                                       SlackHistoryToolMixin)
from slack_client.search_tool import SlackSearchToolMixin
from tool_registry import ToolContext

REQUESTER = "U_ASKER"
OUTSIDER = "U_SOMEONE_ELSE"


class _Bot(SlackHistoryToolMixin, SlackChannelLookupToolMixin, SlackSearchToolMixin):
    """SlackBot stand-in: the three read-surface mixins under test over a mocked async Web API.
    The real SlackBot mixes history + lookup + search on ONE instance, so the delivery-audience
    gate defined in the history mixin is reachable as self.… from lookup and search."""

    def __init__(self):
        self.app = MagicMock()
        self.app.client = MagicMock()
        self.self_team_id = "T_WORKSPACE"
        self.bot_user_id = "U_BOT"
        for method in ("conversations_info", "conversations_history", "conversations_replies",
                       "conversations_list", "conversations_members", "users_conversations",
                       "chat_getPermalink", "pins_list", "users_info", "api_call"):
            setattr(self.app.client, method, AsyncMock())
        # Sensible "nothing here" defaults; individual tests override.
        self.app.client.conversations_history.return_value = {"messages": []}
        self.app.client.conversations_replies.return_value = {"messages": []}
        self.app.client.chat_getPermalink.return_value = {"permalink": "https://x/p1"}
        self.app.client.pins_list.return_value = {"items": []}
        self.app.client.conversations_list.return_value = {"ok": True, "channels": []}
        self.warnings = []

    def log_warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def log_debug(self, *a, **k):
        pass

    log_info = log_error = log_debug


@pytest.fixture
def bot():
    return _Bot()


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setattr(config, "enable_history_tools", True)
    monkeypatch.setattr(config, "history_tool_max_messages", 50)


def _ctx(user_id=REQUESTER, channel_id="C_TARGET", attested=False, human=True, **kw):
    """The default context is a PERSON asking. `human=False` models another app's bot, whose
    user id is real but whose membership must not authorize anything."""
    return ToolContext(channel_id=channel_id, user_id=user_id,
                       origin_membership_attested=attested,
                       requester_is_human=human, **kw)


def _info(**flags):
    """A conversations.info response. Only the keys a test passes are present — the point of
    several cases is what Slack does NOT send."""
    return {"ok": True, "channel": dict(flags)}


def _user_convos(*ids, cursor=None):
    """A users.conversations page. `cursor` set → Slack says there is more."""
    page = {"ok": True, "channels": [{"id": i, "name": i.lower()} for i in ids]}
    if cursor:
        page["response_metadata"] = {"next_cursor": cursor}
    return page


def _member_of(bot_, *ids):
    bot_.app.client.users_conversations.return_value = _user_convos(*ids)


# --------------------------------------------------------------- public channels

@pytest.mark.asyncio
async def test_public_channel_both_members_allowed(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    _member_of(bot, "C_TARGET")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "hello"}]}

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["ok"] is True
    assert res["messages"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_public_channel_requester_not_member_denied(bot):
    """The product decision: public is NOT a free pass. The bot being in #general does not
    entitle a non-member to read it through the bot."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    _member_of(bot, "C_SOMETHING_ELSE")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_public_channel_bot_not_member_denied(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=False)
    _member_of(bot, "C_TARGET")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_history.assert_not_called()


# --------------------------------------------------------------- private channels

@pytest.mark.asyncio
async def test_private_channel_both_members_allowed(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_TARGET")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "secret-ok"}]}

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["ok"] is True and res["messages"][0]["text"] == "secret-ok"


@pytest.mark.asyncio
async def test_private_channel_bot_only_denied(bot):
    """The original leak: bot in the private channel, requester not. No content, ever."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_OTHER")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"
    assert "messages" not in res
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_private_channel_requester_only_denied(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=False)
    _member_of(bot, "C_TARGET")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_group_object_requires_both(bot):
    """A legacy `is_group` private channel takes the same path as is_private."""
    bot.app.client.conversations_info.return_value = _info(is_group=True, is_member=True)
    _member_of(bot, "C_OTHER")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"


# --------------------------------------------------------------- DMs

@pytest.mark.asyncio
async def test_own_dm_allowed(bot):
    """An IM carries no is_private and no is_member — it is identified by is_im, and the
    participant is the `user` field."""
    bot.app.client.conversations_info.return_value = _info(
        is_im=True, is_open=True, user=REQUESTER)
    # Measured live: users.conversations(user=X, types=im) on the bot token returns exactly
    # the one DM X shares with the bot, so an unattested DM is authorized by the walk too.
    _member_of(bot, "D_MINE")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": REQUESTER, "ts": "1.1", "text": "my own dm"}]}

    res = await bot.fetch_history_tool("D_MINE", ctx=_ctx(channel_id="D_MINE"))

    assert res["ok"] is True and res["messages"][0]["text"] == "my own dm"
    # An UNATTESTED DM pays the membership walk like anything else. It could be decided from
    # is_im/`user` alone, but then a nonexistent D-id would fail fast while someone else's real
    # DM paid a walk first — and reply latency would answer "does this DM exist?". Uniform cost
    # for every unattested target is the point.
    assert bot.app.client.users_conversations.await_count == 1


@pytest.mark.asyncio
async def test_attested_dm_costs_no_membership_walk(bot):
    """The live case: Slack delivered this person's DM message, so their side is already
    proven and the walk is skipped — the common path stays one call, not a pagination."""
    bot.app.client.conversations_info.return_value = _info(
        is_im=True, is_open=True, user=REQUESTER)
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": REQUESTER, "ts": "1.1", "text": "my own dm"}]}

    res = await bot.fetch_history_tool(
        "D_MINE", ctx=_ctx(channel_id="D_MINE", attested=True))

    assert res["ok"] is True
    bot.app.client.users_conversations.assert_not_called()


@pytest.mark.asyncio
async def test_other_peoples_dm_denied(bot):
    """THE regression: conversations.info on an IM has no `is_private` key, so the old
    `ch.get("is_private", False)` read it as a public channel and served the transcript."""
    bot.app.client.conversations_info.return_value = _info(
        is_im=True, is_open=True, user=OUTSIDER)

    res = await bot.fetch_history_tool("D_THEIRS", ctx=_ctx(channel_id="D_THEIRS"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_dm_with_deactivated_account_denied(bot):
    """A deactivated user cannot start a fresh turn, so only a stale/replayed context can
    present one. Fail closed even though the ids match."""
    bot.app.client.conversations_info.return_value = _info(
        is_im=True, user=REQUESTER, is_user_deleted=True)

    res = await bot.fetch_history_tool("D_MINE", ctx=_ctx(channel_id="D_MINE"))

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_contradictory_type_flags_denied(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_im=True, is_mpim=True, user=REQUESTER)

    res = await bot.fetch_history_tool("D_WEIRD", ctx=_ctx(channel_id="D_WEIRD"))

    assert res["error"] == "not_accessible"


@pytest.mark.asyncio
async def test_empty_channel_object_denied(bot):
    bot.app.client.conversations_info.return_value = {"ok": True, "channel": {}}

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"


# --------------------------------------------------------------- group DMs

@pytest.mark.asyncio
async def test_mpim_member_allowed(bot):
    """Slack omits is_member on an mpim; the bot token only surfaces group DMs it is in, so
    the requester-side list carries the proof."""
    bot.app.client.conversations_info.return_value = _info(is_mpim=True, is_private=True)
    _member_of(bot, "G_GROUP")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": REQUESTER, "ts": "1.1", "text": "group chat"}]}

    res = await bot.fetch_history_tool("G_GROUP", ctx=_ctx(channel_id="G_GROUP"))

    assert res["ok"] is True and res["messages"][0]["text"] == "group chat"


@pytest.mark.asyncio
async def test_mpim_non_member_denied(bot):
    bot.app.client.conversations_info.return_value = _info(is_mpim=True, is_private=True)
    _member_of(bot, "G_SOME_OTHER_GROUP")

    res = await bot.fetch_history_tool("G_GROUP", ctx=_ctx(channel_id="G_GROUP"))

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_history.assert_not_called()


# --------------------------------------------------------------- requester identity

@pytest.mark.asyncio
async def test_no_user_id_denies_everything(bot):
    """A context with no requester (background job, replay, hand-built) reads nothing —
    never a fallback to the installer, a service identity, or 'the last human here'."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    _member_of(bot, "C_TARGET")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx(user_id=None))

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_info.assert_not_called()
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_no_context_at_all_denies(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)

    res = await bot.fetch_history_tool("C_TARGET")

    assert res["error"] == "not_accessible"


# --------------------------------------------------------------- origin attestation

@pytest.mark.asyncio
async def test_attestation_skips_membership_lookup(bot):
    """Slack delivering the turn's message from this channel already proves membership."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    bot.app.client.users_conversations.side_effect = AssertionError("must not be called")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": REQUESTER, "ts": "1.1", "text": "current channel"}]}

    res = await bot.fetch_history_tool(
        "C_TARGET", ctx=_ctx(channel_id="C_TARGET", attested=True))

    assert res["ok"] is True


@pytest.mark.asyncio
async def test_attestation_does_not_cover_other_channels(bot):
    """It exempts exactly the conversation it was minted for — not every id the turn names."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_TARGET")

    res = await bot.fetch_history_tool(
        "C_ELSEWHERE", ctx=_ctx(channel_id="C_TARGET", attested=True))

    assert res["error"] == "not_accessible"


@pytest.mark.asyncio
async def test_attestation_still_requires_bot_membership(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=False)

    res = await bot.fetch_history_tool(
        "C_TARGET", ctx=_ctx(channel_id="C_TARGET", attested=True))

    assert res["error"] == "not_accessible"


def test_attestation_defaults_false_on_every_hand_built_context():
    """Detached/synthetic contexts must never inherit it. The research/background jobs and the
    settings replay all construct contexts positionally-by-keyword like this."""
    assert ToolContext().origin_membership_attested is False
    assert ToolContext(channel_id="C1", user_id="U1").origin_membership_attested is False


@pytest.mark.asyncio
async def test_synthetic_context_gets_full_check(bot):
    """A replayed context naming a channel the user has since left is denied — the id match
    alone buys nothing without the attestation flag."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_LEFT_SINCE")

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx(channel_id="C_TARGET"))

    assert res["error"] == "not_accessible"


def test_attest_message_origin_requires_live_human_event():
    """The stamp itself: only a human event whose channel AND user match the message."""
    from base_client import Message
    from slack_client.event_handlers.message_events import attest_message_origin

    def _msg():
        return Message(text="hi", user_id=REQUESTER, channel_id="C1", thread_id="1.0")

    live = _msg()
    attest_message_origin(live, {"user": REQUESTER, "channel": "C1"}, "human")
    assert live.metadata["origin_event_verified"] is True
    assert live.metadata["origin_channel_id"] == "C1"

    for event, sender in (
        ({"user": REQUESTER, "channel": "C_OTHER"}, "human"),   # channel disagrees
        ({"user": OUTSIDER, "channel": "C1"}, "human"),         # user disagrees
        ({"user": REQUESTER, "channel": "C1"}, "other_bot"),    # not a human
        ({"channel": "C1"}, "human"),                           # no user
        ({"user": REQUESTER}, "human"),                         # no channel
    ):
        m = _msg()
        attest_message_origin(m, event, sender)
        assert "origin_event_verified" not in m.metadata, event


# --------------------------------------------------------------- failure modes

@pytest.mark.asyncio
async def test_slack_error_denies(bot):
    bot.app.client.conversations_info.side_effect = SlackApiError(
        "boom", {"error": "channel_not_found"})

    res = await bot.fetch_history_tool("C_NOPE", ctx=_ctx())

    assert res["error"] == "not_accessible"
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_membership_lookup_error_denies(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    bot.app.client.users_conversations.side_effect = SlackApiError("x", {"error": "ratelimited"})

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"


@pytest.mark.asyncio
async def test_membership_not_ok_response_denies(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    bot.app.client.users_conversations.return_value = {"ok": False, "error": "missing_scope"}

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert res["error"] == "not_accessible"


@pytest.mark.asyncio
async def test_pagination_exhaustion_denies_and_is_not_called_non_membership(bot):
    """Every page returns a cursor, so the walk hits its cap. That is authorization
    UNAVAILABLE, not proof of non-membership: deny, bounded, and log it as unverified."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    bot.app.client.users_conversations.return_value = _user_convos("C_A", cursor="MORE")
    ctx = _ctx()

    res = await bot.fetch_history_tool("C_TARGET", ctx=ctx)

    assert res["error"] == "not_accessible"
    import slack_client.history_tool as ht
    assert bot.app.client.users_conversations.await_count == ht._USER_CONVOS_MAX_PAGES
    allowed, reason = await bot._channel_is_accessible("C_TARGET", ctx)
    assert allowed is False
    assert reason == "requester_membership_unverified"   # NOT "requester_not_member"


@pytest.mark.asyncio
async def test_short_page_is_not_the_end(bot):
    """A page smaller than the limit still carries a cursor: follow it. The match lives on
    page 2, and a walk that stopped early would deny a legitimate member."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    bot.app.client.users_conversations.side_effect = [
        _user_convos("C_A", cursor="PAGE2"),
        _user_convos("C_TARGET"),
    ]

    allowed, reason = await bot._channel_is_accessible("C_TARGET", ctx=_ctx())

    assert allowed is True and reason == "both_members"
    assert bot.app.client.users_conversations.await_count == 2


@pytest.mark.asyncio
async def test_partial_scan_can_still_prove_membership(bot):
    """Cap-truncated is only fatal when the id was NOT found: an id we did see is proof."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    bot.app.client.users_conversations.return_value = _user_convos("C_TARGET", cursor="MORE")

    allowed, _ = await bot._channel_is_accessible("C_TARGET", ctx=_ctx())

    assert allowed is True


@pytest.mark.asyncio
async def test_membership_uses_users_conversations_not_the_roster(bot):
    """conversations.members would be O(channel size) and needs the roster of a channel we
    may not be entitled to enumerate."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    _member_of(bot, "C_TARGET")

    await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    bot.app.client.conversations_members.assert_not_called()
    kwargs = bot.app.client.users_conversations.await_args.kwargs
    assert kwargs["user"] == REQUESTER
    assert kwargs["limit"] == 200
    assert kwargs["exclude_archived"] is False
    assert "im" in kwargs["types"] and "mpim" in kwargs["types"]
    assert "private_channel" in kwargs["types"]


# --------------------------------------------------------------- uniform refusal

@pytest.mark.asyncio
async def test_every_denial_returns_an_identical_payload(bot):
    """No reason codes, no channel id, no wording differences — otherwise the refusal itself
    answers "does this private channel exist?" one probe at a time."""
    async def _deny(configure):
        b = _Bot()
        configure(b)
        return await b.fetch_history_tool("C_PROBE", ctx=_ctx(channel_id="C_PROBE"))

    def _nonexistent(b):
        b.app.client.conversations_info.side_effect = SlackApiError(
            "x", {"error": "channel_not_found"})

    def _bot_not_member(b):
        b.app.client.conversations_info.return_value = _info(
            is_channel=True, is_private=True, is_member=False)
        _member_of(b, "C_PROBE")

    def _requester_not_member(b):
        b.app.client.conversations_info.return_value = _info(
            is_channel=True, is_private=True, is_member=True)
        _member_of(b, "C_OTHER")

    def _malformed(b):
        b.app.client.conversations_info.return_value = {"ok": True, "channel": {}}

    def _capped(b):
        b.app.client.conversations_info.return_value = _info(
            is_channel=True, is_private=True, is_member=True)
        b.app.client.users_conversations.return_value = _user_convos("C_A", cursor="MORE")

    def _lookup_failed(b):
        b.app.client.conversations_info.return_value = _info(
            is_channel=True, is_private=True, is_member=True)
        b.app.client.users_conversations.side_effect = RuntimeError("network gone")

    def _someone_elses_dm(b):
        b.app.client.conversations_info.return_value = _info(is_im=True, user=OUTSIDER)

    results = [await _deny(c) for c in (_nonexistent, _bot_not_member, _requester_not_member,
                                        _malformed, _capped, _lookup_failed, _someone_elses_dm)]

    expected = {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    for res in results:
        assert res == expected
    assert all(r == results[0] for r in results)


@pytest.mark.asyncio
async def test_refusal_never_names_the_channel_or_the_reason(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_OTHER")

    res = await bot.fetch_history_tool("C_SECRET_PROJECT", ctx=_ctx())

    assert "reason" not in res
    assert "C_SECRET_PROJECT" not in res["message"]
    for leak in ("not_member", "private", "exist", "member of"):
        assert leak not in res["message"].replace("both you and the bot are in", "")


@pytest.mark.asyncio
async def test_denial_reasons_are_logged_even_though_hidden(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=False)
    _member_of(bot, "C_TARGET")      # the requester IS in it; only the bot is not

    await bot.fetch_history_tool("C_TARGET", ctx=_ctx())

    assert any("bot_not_member" in w for w in bot.warnings)


# --------------------------------------------------------- the requester must be a person

@pytest.mark.asyncio
async def test_another_bots_membership_does_not_authorize(bot):
    """Bot-authored messages are deliberately processed (bot<->bot works), and an app posting
    through its own token carries a REAL U… id in event["user"] — so `user_id` alone cannot
    carry the policy. Otherwise: bot A asks us in channel Y to read channel X that A and we
    share, and X lands in front of whoever is in Y."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_TARGET")          # the other bot really is a member

    res = await bot.fetch_history_tool("C_TARGET", ctx=_ctx(human=False))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    # Refused on identity alone — we never even ask Slack on a bot's behalf.
    bot.app.client.users_conversations.assert_not_called()


@pytest.mark.asyncio
async def test_a_bot_requester_cannot_resolve_channel_names_either():
    b = _lookup_bot([{"id": "C_MENU", "name": "menu-insights"}], infos={"C_MENU": _SHARED})

    res = await b.execute_lookup_channel(_ctx(human=False), {"name": "menu-insights"})

    assert res["ok"] is False and res["error"] == "not_accessible"


def test_tool_context_defaults_deny_a_non_human_requester():
    """Fail closed: a hand-built context (background job, replay) is not a person until
    something proves otherwise at a real event entry point."""
    assert ToolContext().requester_is_human is False
    assert ToolContext(user_id=REQUESTER).requester_is_human is False


# --------------------------------------------------------------- every tool is gated

@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(CHANNEL_READ_TOOLS))
async def test_every_channel_read_tool_is_gated(bot, tool_name):
    """The regression net: each registered channel-read tool must reach the authorization
    hook and return the uniform refusal when it denies. A tool added later that skips the
    gate fails here."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_OTHER")
    args = {"channel_id": "C_TARGET", "message_ts": "1.0", "thread_ts": "1.0", "limit": 5}

    res = await bot.dispatch_history_tool_call(tool_name, args, _ctx())

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    for reader in ("conversations_history", "conversations_replies", "chat_getPermalink",
                   "pins_list"):
        getattr(bot.app.client, reader).assert_not_called()


def test_schema_list_and_gated_set_agree(bot):
    """If a new schema is added without adding it to CHANNEL_READ_TOOLS, dispatch refuses it
    outright (it can never route past the gate) and this test says so out loud."""
    assert {s["name"] for s in bot.get_history_tools_for_openai()} == set(CHANNEL_READ_TOOLS)


@pytest.mark.asyncio
async def test_tool_not_in_the_gated_set_cannot_dispatch(bot):
    res = await bot.dispatch_history_tool_call("fetch_everything", {"channel_id": "C1"}, _ctx())
    assert res["error"] == "unknown_tool"


def test_tool_descriptions_state_the_both_members_rule(bot):
    for schema in bot.get_history_tools_for_openai():
        text = schema["description"].lower()
        assert "both" in text and "bot" in text, schema["name"]


@pytest.mark.asyncio
async def test_dispatch_gate_applies_to_the_current_channel_too(bot):
    """No `channel_id == ctx.channel_id` shortcut: an unattested context is checked like any
    other, so a stale one can't read a channel the user has left."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_OTHER")

    res = await bot.dispatch_history_tool_call("fetch_channel_history", {}, _ctx())

    assert res["error"] == "not_accessible"


# --------------------------------------------------------------- request-scoped memo

@pytest.mark.asyncio
async def test_a_cancelled_waiter_does_not_poison_the_shared_future(bot):
    """Awaiting a bare future propagates the WAITER's cancellation into the shared one. A
    single tool round timing out would then leave a cancelled future in the memo, and every
    later authorization in the request would raise CancelledError — a BaseException, so it
    escapes the fail-closed handlers and aborts the whole turn instead of denying."""
    release = asyncio.Event()

    async def _slow_users_conversations(**kwargs):
        await release.wait()
        return _user_convos("C_TARGET")

    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    bot.app.client.users_conversations.side_effect = _slow_users_conversations
    ctx = _ctx()

    owner = asyncio.ensure_future(bot._channel_is_accessible("C_TARGET", ctx))
    await asyncio.sleep(0)
    waiter = asyncio.ensure_future(bot._channel_is_accessible("C_TARGET", ctx))
    await asyncio.sleep(0)

    waiter.cancel()                      # e.g. this tool call hit its timeout
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    assert (await owner)[0] is True                                  # owner unharmed
    assert (await bot._channel_is_accessible("C_TARGET", ctx))[0] is True   # memo still usable


@pytest.mark.asyncio
async def test_memo_coalesces_concurrent_scans_into_one_walk(bot):
    """A round's tool calls run under asyncio.gather. Without single-flight (sharing the
    in-flight future, not just the result) each would launch the same pagination walk."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_users_conversations(**kwargs):
        started.set()
        await release.wait()
        return _user_convos("C_A", "C_B")

    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    bot.app.client.users_conversations.side_effect = _slow_users_conversations
    ctx = _ctx()

    task = asyncio.gather(
        bot._channel_is_accessible("C_A", ctx),
        bot._channel_is_accessible("C_B", ctx),
        bot._channel_is_accessible("C_A", ctx),
    )
    await started.wait()
    release.set()
    results = await task

    assert [r[0] for r in results] == [True, True, True]
    assert bot.app.client.users_conversations.await_count == 1     # ONE walk, not three
    assert bot.app.client.conversations_info.await_count == 2      # one per distinct channel


@pytest.mark.asyncio
async def test_memo_is_request_scoped_not_a_ttl_cache(bot):
    """A fresh context re-verifies from scratch: a positive answer must never outlive the
    request that earned it, or someone removed from a channel keeps reading it."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot, "C_TARGET")

    assert (await bot._channel_is_accessible("C_TARGET", _ctx()))[0] is True
    _member_of(bot, "C_OTHER")           # they have since left
    assert (await bot._channel_is_accessible("C_TARGET", _ctx()))[0] is False


@pytest.mark.asyncio
async def test_memo_keys_are_scoped_per_requester(bot):
    """One person's allow must never satisfy another person's check."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    ctx = _ctx()
    _member_of(bot, "C_TARGET")
    assert (await bot._channel_is_accessible("C_TARGET", ctx))[0] is True

    other = ToolContext(channel_id="C_TARGET", user_id=OUTSIDER,
                        channel_access_memo=ctx.channel_access_memo)
    _member_of(bot, "C_NOT_THEIRS")
    assert (await bot._channel_is_accessible("C_TARGET", other))[0] is False


@pytest.mark.asyncio
async def test_missing_memo_target_still_authorizes(bot):
    """A context we cannot hang a memo on (a bare mock) must re-verify, never fail open."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=True)
    ctx = MagicMock()

    allowed, _ = await bot._channel_is_accessible("C_TARGET", ctx)

    assert allowed is False      # a MagicMock user_id is not a usable requester identity


# --------------------------------------------------------------- lookup_channel

def _lookup_bot(user_channels, public_channels=(), infos=None):
    b = _Bot()
    b.app.client.users_conversations.return_value = {
        "ok": True, "channels": list(user_channels)}
    b.app.client.conversations_list.return_value = {
        "ok": True, "channels": list(public_channels)}
    table = infos or {}

    async def _info_for(channel, **kw):
        return {"ok": True, "channel": table.get(channel, {})}

    b.app.client.conversations_info.side_effect = _info_for
    return b


_SHARED = {"is_channel": True, "is_private": True, "is_member": True}
_BOT_ABSENT = {"is_channel": True, "is_private": True, "is_member": False}


@pytest.mark.asyncio
async def test_lookup_resolves_an_exact_name():
    b = _lookup_bot([{"id": "C_MENU", "name": "menu-insights", "is_private": True}],
                    infos={"C_MENU": _SHARED})

    # DM surface: lookup resolves normally. In a multi-user surface the delivery gate would
    # withhold this private match (see test_lookup_in_channel_withholds_private_match).
    res = await b.execute_lookup_channel(_ctx(is_dm=True), {"name": "menu-insights"})

    assert res == {"ok": True, "id": "C_MENU", "name": "menu-insights", "is_private": True}


@pytest.mark.asyncio
async def test_lookup_normalizes_hash_and_case():
    b = _lookup_bot([{"id": "C_MENU", "name": "menu-insights"}], infos={"C_MENU": _SHARED})

    res = await b.execute_lookup_channel(_ctx(is_dm=True), {"name": "  #Menu-Insights "})

    assert res["ok"] is True and res["id"] == "C_MENU"


@pytest.mark.asyncio
async def test_lookup_matches_name_normalized_too():
    b = _lookup_bot([{"id": "C_X", "name_normalized": "menu-insights"}],
                    infos={"C_X": _SHARED})

    res = await b.execute_lookup_channel(_ctx(is_dm=True), {"name": "menu-insights"})

    assert res["ok"] is True and res["id"] == "C_X"


@pytest.mark.asyncio
async def test_lookup_reports_ambiguity_instead_of_guessing():
    b = _lookup_bot(
        [{"id": "C_ONE", "name": "menu-insights"}, {"id": "C_TWO", "name": "menu-insights"}],
        infos={"C_ONE": _SHARED, "C_TWO": _SHARED})

    res = await b.execute_lookup_channel(_ctx(is_dm=True), {"name": "menu-insights"})

    assert res["ok"] is False and res["error"] == "ambiguous"
    assert {c["id"] for c in res["candidates"]} == {"C_ONE", "C_TWO"}


@pytest.mark.asyncio
async def test_lookup_requires_bot_membership_via_conversations_info():
    """users.conversations omits is_member, so a name the REQUESTER can see is not yet an id
    we may hand out — the bot's side is proved separately."""
    b = _lookup_bot([{"id": "C_THEIRS", "name": "menu-insights"}],
                    infos={"C_THEIRS": _BOT_ABSENT})

    res = await b.execute_lookup_channel(_ctx(), {"name": "menu-insights"})

    assert res["error"] == "not_found"
    assert "C_THEIRS" not in str(res)
    b.app.client.conversations_info.assert_awaited()


@pytest.mark.asyncio
async def test_lookup_never_enumerates_the_channel_directory():
    """conversations.list(types=private_channel) would leak names like #acme-layoffs to anyone
    who can ask the bot a question. The directory is not consulted AT ALL: under the
    both-members rule a channel absent from the requester's own list is refused anyway, so
    scanning could only add latency and a false "I couldn't finish looking"."""
    b = _lookup_bot([], public_channels=[{"id": "C_PUB", "name": "general"}],
                    infos={"C_PUB": {"is_channel": True, "is_private": False, "is_member": True}})
    b.app.client.users_conversations.return_value = _user_convos("C_OTHER")

    res = await b.execute_lookup_channel(_ctx(), {"name": "general"})

    b.app.client.conversations_list.assert_not_called()
    # …and a public channel reachable only from the directory stays unresolvable.
    assert res["ok"] is False and res["error"] == "not_found"


@pytest.mark.asyncio
async def test_lookup_private_channel_not_shared_is_invisible():
    """A private channel the requester isn't in never enters the candidate set at all."""
    b = _lookup_bot([{"id": "C_MINE", "name": "my-channel"}], infos={"C_MINE": _SHARED})

    res = await b.execute_lookup_channel(_ctx(), {"name": "someone-elses-private"})

    assert res["error"] == "not_found"


@pytest.mark.asyncio
async def test_lookup_incomplete_scan_says_so_and_never_says_no_such_channel():
    b = _lookup_bot([], infos={})
    b.app.client.users_conversations.return_value = _user_convos("C_A", cursor="MORE")

    res = await b.execute_lookup_channel(_ctx(), {"name": "menu-insights"})

    assert res["error"] == "incomplete"
    assert "INCOMPLETE" in res["message"]
    # Phrased as what it CANNOT conclude, and never as the not-found answer.
    assert "can't tell you that channel doesn't exist" in res["message"]
    assert "No channel by that name" not in res["message"]


@pytest.mark.asyncio
async def test_lookup_without_a_requester_is_refused():
    b = _lookup_bot([{"id": "C_MENU", "name": "menu-insights"}], infos={"C_MENU": _SHARED})

    res = await b.execute_lookup_channel(_ctx(user_id=None), {"name": "menu-insights"})

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    b.app.client.users_conversations.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_rejects_an_empty_name():
    b = _lookup_bot([])
    res = await b.execute_lookup_channel(_ctx(), {"name": "  #  "})
    assert res["error"] == "bad_arguments"


@pytest.mark.asyncio
async def test_lookup_shares_the_membership_walk_with_the_guard():
    """Both surfaces read one memoized walk per request, not one each."""
    b = _lookup_bot([{"id": "C_MENU", "name": "menu-insights"}], infos={"C_MENU": _SHARED})
    ctx = _ctx(is_dm=True)  # DM surface so the private match resolves; the walk-sharing is the point

    await b._channel_is_accessible("C_MENU", ctx)
    await b.execute_lookup_channel(ctx, {"name": "menu-insights"})

    assert b.app.client.users_conversations.await_count == 1


def test_lookup_schema_is_registered_with_a_required_name():
    b = _Bot()
    schema = b.get_lookup_channel_tool_schema()
    assert schema["name"] == "lookup_channel"
    assert schema["parameters"]["required"] == ["name"]


# --------------------------------------------------------------- roster tool

@pytest.mark.asyncio
async def test_list_channel_members_is_gated():
    """A roster says who is inside a conversation — same gate as its contents."""
    from message_processor.people_tools import execute_list_channel_members

    bot_ = _Bot()
    bot_.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot_, "C_OTHER")
    ctx = ToolContext(channel_id="C_TARGET", user_id=REQUESTER, client=bot_, is_dm=False)

    res = await execute_list_channel_members(ctx, {})

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    bot_.app.client.conversations_members.assert_not_called()


@pytest.mark.asyncio
async def test_list_channel_members_allowed_for_a_shared_channel():
    from message_processor.people_tools import execute_list_channel_members

    bot_ = _Bot()
    bot_.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(bot_, "C_TARGET")
    bot_.app.client.conversations_members.return_value = {"ok": True, "members": ["U1"]}
    bot_.get_username = AsyncMock(return_value="alice")
    ctx = ToolContext(channel_id="C_TARGET", user_id=REQUESTER, client=bot_, is_dm=False,
                      requester_is_human=True)

    res = await execute_list_channel_members(ctx, {})

    assert res["ok"] is True and res["members"] == [{"id": "U1", "name": "alice"}]


@pytest.mark.asyncio
async def test_list_channel_members_fails_closed_without_a_gate():
    """A client that doesn't expose the gate (another platform, a bare mock) gets nothing."""
    from message_processor.people_tools import execute_list_channel_members

    ctx = ToolContext(channel_id="C_TARGET", user_id=REQUESTER,
                      client=SimpleNamespace(app=SimpleNamespace(client=MagicMock())),
                      is_dm=False)

    res = await execute_list_channel_members(ctx, {})

    assert res["error"] == "not_accessible"


# --------------------------------------------------------------- resolver skip-list

@pytest.mark.asyncio
async def test_resolver_skips_ids_users_info_can_never_resolve():
    """assistant.search.context returns "U00" as the author of some results; users.info 404s
    on it, burning a slot in the resolver's remote budget. Bot OBJECT ids (B…) are the wrong
    argument for users.info too — a bot's user identity is a separate U/W id."""
    from slack_client.utilities import SlackUtilitiesMixin

    class _Resolver(SlackUtilitiesMixin):
        def __init__(self):
            self.user_cache = {}
            self.db = MagicMock()
            self.db.get_user_infos_async = AsyncMock(return_value={})

        def log_debug(self, *a, **k):
            pass

    looked_up = []

    async def _users_info(user):
        looked_up.append(user)
        return {"ok": True, "user": {"name": f"name-{user}", "profile": {}}}

    api = MagicMock()
    api.users_info = AsyncMock(side_effect=_users_info)

    out = await _Resolver().resolve_usernames(
        ["U00", "B01BOTOBJ", "W012345678", "U0123456789ABC", "U1"], api)

    assert looked_up == ["W012345678", "U0123456789ABC", "U1"]
    assert "U00" not in out and "B01BOTOBJ" not in out
    # W-prefixed ids and longer-than-usual U ids are real users: no length/regex guard, because
    # Slack has lengthened ids before and warns apps not to assume their shape.
    assert out["W012345678"] == "name-W012345678"
    assert out["U0123456789ABC"] == "name-U0123456789ABC"


@pytest.mark.asyncio
async def test_resolver_skip_happens_before_the_db_pass():
    from slack_client.utilities import SlackUtilitiesMixin

    class _Resolver(SlackUtilitiesMixin):
        def __init__(self):
            self.user_cache = {}
            self.db = MagicMock()
            self.db.get_user_infos_async = AsyncMock(return_value={})

        def log_debug(self, *a, **k):
            pass

    r = _Resolver()
    api = MagicMock()
    api.users_info = AsyncMock(return_value={"ok": True, "user": {"name": "x", "profile": {}}})

    await r.resolve_usernames(["U00", "B01BOTOBJ"], api)

    r.db.get_user_infos_async.assert_not_called()
    api.users_info.assert_not_called()


# ============================================================ delivery-audience gate (Option B)
#
# A second layer on TOP of retrieval: retrieval decides whether the REQUESTER may read a
# conversation; delivery decides whether that content may be spoken into the CURRENT reply's
# audience. In a DM the audience is the asker alone (full power); in any multi-user surface only
# the current channel or a public-internal source may be delivered — everything else is withheld
# behind the SAME generic refusal, so a redirect can't be told from a denial.

def _set_infos(bot_, table):
    """conversations.info answers from a channel→flags table (per-channel responses, since the
    delivery gate fetches info for both the source and the current-channel destination)."""
    async def _f(channel=None, **kw):
        return {"ok": True, "channel": table.get(channel, {})}
    bot_.app.client.conversations_info.side_effect = _f


_INTERNAL_HERE = {"is_channel": True, "is_private": False, "is_member": True}


@pytest.mark.asyncio
async def test_dm_surface_delivers_non_current_private_source(bot):
    """In a DM the audience is the asker alone, so a private channel they and the bot share is
    fully readable even though it is not the current conversation — no delivery filtering."""
    _set_infos(bot, {"C_PRIV": {"is_channel": True, "is_private": True, "is_member": True}})
    _member_of(bot, "C_PRIV")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "private but mine to read"}]}

    res = await bot.fetch_history_tool("C_PRIV", ctx=_ctx(channel_id="D_MINE", is_dm=True))

    assert res["ok"] is True and res["messages"][0]["text"] == "private but mine to read"


@pytest.mark.asyncio
async def test_multi_user_delivers_current_channel_source(bot):
    """The current channel is always deliverable into its own audience (even when private)."""
    _set_infos(bot, {"C_HERE": {"is_channel": True, "is_private": True, "is_member": True}})
    _member_of(bot, "C_HERE")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "current channel content"}]}

    res = await bot.fetch_history_tool("C_HERE", ctx=_ctx(channel_id="C_HERE"))

    assert res["ok"] is True and res["messages"][0]["text"] == "current channel content"


@pytest.mark.asyncio
async def test_multi_user_delivers_public_internal_source(bot):
    """A public-internal channel is deliverable into a multi-user surface even when it is not the
    current channel — it is readable by any workspace member by design."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE, "C_PUB": _INTERNAL_HERE})
    _member_of(bot, "C_PUB")
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "public knowledge"}]}

    res = await bot.fetch_history_tool("C_PUB", ctx=_ctx(channel_id="C_HERE"))

    assert res["ok"] is True and res["messages"][0]["text"] == "public knowledge"


@pytest.mark.asyncio
async def test_multi_user_redirects_private_non_current_source(bot):
    """A private channel both of us are in, but NOT the current one, is withheld from a channel
    audience — retrieval passes, delivery redirects, no content is read."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE,
                     "C_PRIV": {"is_channel": True, "is_private": True, "is_member": True}})
    _member_of(bot, "C_PRIV")

    res = await bot.fetch_history_tool("C_PRIV", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_multi_user_redirects_dm_source(bot):
    """The requester's own DM with the bot must not be read aloud into a channel."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE, "D_MINE": {"is_im": True, "user": REQUESTER}})
    _member_of(bot, "D_MINE")

    res = await bot.fetch_history_tool("D_MINE", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_multi_user_redirects_mpim_source(bot):
    """A group DM is not public-internal, so it is withheld from a channel audience."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE, "G_GROUP": {"is_mpim": True, "is_private": True}})
    _member_of(bot, "G_GROUP")

    res = await bot.fetch_history_tool("G_GROUP", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_multi_user_redirects_externally_shared_source(bot):
    """An externally/cross-org shared channel is not "the workspace", so it is withheld even
    though is_private is false and both of us are members."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE,
                     "C_SHARED": {"is_channel": True, "is_private": False, "is_member": True,
                                  "is_ext_shared": True}})
    _member_of(bot, "C_SHARED")

    res = await bot.fetch_history_tool("C_SHARED", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_ext_shared_destination_forces_current_source_only(bot):
    """Even a public-internal source is withheld when the CURRENT channel is externally shared:
    delivering internal content there would expose it to the external org (codex r3 #2)."""
    _set_infos(bot, {"C_HERE": {"is_channel": True, "is_private": False, "is_member": True,
                                "is_ext_shared": True},
                     "C_PUB": _INTERNAL_HERE})
    _member_of(bot, "C_PUB")

    res = await bot.fetch_history_tool("C_PUB", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_redirect_payload_is_byte_identical_to_denial(bot):
    """A redirect (member, private source in a channel) and a denial (non-member) must be
    indistinguishable to the model — same dict, byte for byte."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE,
                     "C_PRIV": {"is_channel": True, "is_private": True, "is_member": True}})
    _member_of(bot, "C_PRIV")
    redirect = await bot.fetch_history_tool("C_PRIV", ctx=_ctx(channel_id="C_HERE"))

    denial_bot = _Bot()
    denial_bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True)
    _member_of(denial_bot, "C_ELSEWHERE")   # requester is NOT in C_PRIV → denied at retrieval
    denial = await denial_bot.fetch_history_tool("C_PRIV", ctx=_ctx(channel_id="C_HERE"))

    assert redirect == denial
    assert redirect == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_limited_access_source_is_withheld_though_public(bot):
    """codex r3 #1: a limited-access / record-backed channel is NOT workspace-public even when
    is_private is false — membership there is gated — so it is withheld from a channel audience."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE,
                     "C_LTD": {"is_channel": True, "is_private": False, "is_member": True,
                               "is_limited_access": True}})
    _member_of(bot, "C_LTD")

    res = await bot.fetch_history_tool("C_LTD", ctx=_ctx(channel_id="C_HERE"))

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}


@pytest.mark.asyncio
async def test_destination_forces_current_only_fail_closed_and_shared():
    """codex r3 #2: the destination check forces current-source-only when it can't be classified
    (API error / malformed) and when the current channel is org-shared / pending-ext-shared; a
    normal internal channel does NOT force it (costs no internal capability)."""
    b_err = _Bot()
    b_err.app.client.conversations_info.side_effect = SlackApiError("x", {"error": "ratelimited"})
    assert await b_err._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True

    b_bad = _Bot()
    _set_infos(b_bad, {"C_HERE": {}})
    assert await b_bad._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True

    b_org = _Bot()
    _set_infos(b_org, {"C_HERE": {"is_channel": True, "is_org_shared": True}})
    assert await b_org._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True

    b_pend = _Bot()
    _set_infos(b_pend, {"C_HERE": {"is_channel": True, "is_pending_ext_shared": True}})
    assert await b_pend._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True

    b_ok = _Bot()
    _set_infos(b_ok, {"C_HERE": _INTERNAL_HERE})
    assert await b_ok._destination_forces_current_only(_ctx(channel_id="C_HERE")) is False


def test_classify_source_missing_is_private_is_not_public(bot):
    """codex r3 #3: is_private must be present AND explicitly False. Missing or None → not
    public (a genuine public channel sends is_private:false explicitly, verified live)."""
    assert bot._classify_source_public({"is_channel": True, "is_member": True}, _ctx()) is False
    assert bot._classify_source_public({"is_channel": True, "is_private": None}, _ctx()) is False
    assert bot._classify_source_public({"is_channel": True, "is_private": False}, _ctx()) is True


@pytest.mark.asyncio
async def test_search_foreign_team_hit_dropped_even_if_current_channel(bot):
    """codex r3 #4: a hit whose team_id differs from the bot's is dropped BEFORE the
    current-channel exemption — a cross-workspace result claiming the current channel id can't
    ride it."""
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        {"channel_id": "C_HERE", "team_id": "T_OTHER", "message_ts": "1.1", "content": "foreign"},
    ]}}

    out = await bot.execute_search_tool(_ctx(channel_id="C_HERE", action_token="tok"), {"query": "x"})

    assert out["ok"] is True and out["count"] == 0


@pytest.mark.asyncio
async def test_source_public_memo_keyed_by_team(bot):
    """codex r3 #4: the source-public verdict is keyed by team_id, so a same-channel-id verdict is
    never reused across differing workspaces — a foreign team_id makes an otherwise-public channel
    non-deliverable."""
    _set_infos(bot, {"C_PUB": _INTERNAL_HERE})
    ctx = _ctx(channel_id="C_HERE")

    assert await bot._source_is_public("C_PUB", ctx, source_team_id="T_OTHER") is False
    assert await bot._source_is_public("C_PUB", ctx, source_team_id="T_WORKSPACE") is True


@pytest.mark.asyncio
async def test_search_channel_surface_drops_non_deliverable_hits(bot):
    """Multi-user surface: keep current-channel + public-internal hits; drop the rest SILENTLY
    (no note, no count of what was removed). Also proves the defensive parse of a string-typed
    `channel` neither crashes nor false-positives."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE,   # normal internal destination (delivery not locked)
                     "C_PUB": _INTERNAL_HERE,
                     "C_PRIV": {"is_channel": True, "is_private": True, "is_member": True}})
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        {"channel": "C_HERE", "message_ts": "1.1", "content": "here as a string channel"},
        {"channel_id": "C_PUB", "message_ts": "1.2", "content": "public elsewhere"},
        {"channel_id": "C_PRIV", "message_ts": "1.3", "content": "private elsewhere"},
    ]}}

    out = await bot.execute_search_tool(_ctx(channel_id="C_HERE", action_token="tok"), {"query": "x"})

    kept = {(r["channel"], r["text"]) for r in out["results"]}
    assert kept == {("C_HERE", "here as a string channel"), ("C_PUB", "public elsewhere")}
    assert out["count"] == 2
    assert "note" not in out  # the private hit vanished with no trace of its existence


@pytest.mark.asyncio
async def test_search_dm_surface_keeps_everything(bot):
    """A DM is the asker's own audience: search keeps full reach and never classifies sources."""
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        {"channel_id": "C_PRIV", "message_ts": "1.1", "content": "private"},
        {"channel_id": "C_ANOTHER", "message_ts": "1.2", "content": "another"},
    ]}}

    out = await bot.execute_search_tool(
        _ctx(channel_id="D_MINE", is_dm=True, action_token="tok"), {"query": "x"})

    assert out["count"] == 2
    bot.app.client.conversations_info.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_in_channel_withholds_private_match():
    """A private match in a multi-user surface returns the generic refusal — no id, no name, no
    is_private — so the reply never confirms the channel exists to the audience."""
    b = _lookup_bot([{"id": "C_PRIV", "name": "secret-room", "is_private": True}],
                    infos={"C_PRIV": _SHARED, "C_HERE": _INTERNAL_HERE})

    res = await b.execute_lookup_channel(_ctx(channel_id="C_HERE"), {"name": "secret-room"})

    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    assert "C_PRIV" not in str(res) and "secret-room" not in str(res)
    assert "is_private" not in res and "id" not in res


@pytest.mark.asyncio
async def test_lookup_in_dm_resolves_private_match():
    """A DM is the asker's own audience, so lookup resolves the same private channel normally."""
    b = _lookup_bot([{"id": "C_PRIV", "name": "secret-room", "is_private": True}],
                    infos={"C_PRIV": _SHARED})

    res = await b.execute_lookup_channel(_ctx(is_dm=True), {"name": "secret-room"})

    assert res == {"ok": True, "id": "C_PRIV", "name": "secret-room", "is_private": True}


@pytest.mark.asyncio
async def test_lookup_in_channel_resolves_public_internal_match():
    """A public-internal match IS deliverable into a channel audience, so it resolves there."""
    b = _lookup_bot([{"id": "C_PUB", "name": "general"}],
                    infos={"C_PUB": _INTERNAL_HERE, "C_HERE": _INTERNAL_HERE})

    res = await b.execute_lookup_channel(_ctx(channel_id="C_HERE"), {"name": "general"})

    assert res["ok"] is True and res["id"] == "C_PUB"


# ------------------------------------------------ codex r4: fail-safe consistency hardening

@pytest.mark.asyncio
async def test_search_in_ext_shared_channel_delivers_only_current(bot):
    """FIX 1: search delegates to the canonical `_delivery_allowed`, so an externally-shared
    CURRENT channel locks delivery to current-channel hits — a public-internal non-current hit is
    dropped, closing the internal→external-org leak history and lookup already close."""
    _set_infos(bot, {"C_HERE": {"is_channel": True, "is_private": False, "is_member": True,
                                "is_ext_shared": True},
                     "C_PUB": _INTERNAL_HERE})
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        {"channel_id": "C_HERE", "message_ts": "1.1", "content": "current"},
        {"channel_id": "C_PUB", "message_ts": "1.2", "content": "public elsewhere"},
    ]}}

    out = await bot.execute_search_tool(_ctx(channel_id="C_HERE", action_token="tok"), {"query": "x"})

    assert [r["channel"] for r in out["results"]] == ["C_HERE"]
    assert out["count"] == 1


def test_classify_source_fails_closed_when_bot_team_unknown():
    """FIX 2: with self_team_id unknown we can't prove any channel is our-workspace-internal, so
    even a textbook public channel is NOT classified public."""
    b = _Bot()
    b.self_team_id = None
    assert b._classify_source_public(_INTERNAL_HERE, _ctx()) is False


@pytest.mark.asyncio
async def test_dest_forces_current_only_fail_closed_gaps():
    """FIX 3: (a) unknown bot team → current-only; (b) a nonempty-but-partial destination object
    carrying no recognizable type flag is unclassifiable → current-only."""
    b_noteam = _Bot()
    b_noteam.self_team_id = None
    _set_infos(b_noteam, {"C_HERE": _INTERNAL_HERE})
    assert await b_noteam._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True

    b_partial = _Bot()
    _set_infos(b_partial, {"C_HERE": {"num_members": 3}})   # nonempty, but no type/share flags
    assert await b_partial._destination_forces_current_only(_ctx(channel_id="C_HERE")) is True


@pytest.mark.asyncio
async def test_search_hit_with_contradictory_team_ids_dropped(bot):
    """FIX 4: a hit that claims two different workspaces is self-contradictory → dropped, even
    when its channel id equals the current channel."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE})
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        {"channel_id": "C_HERE", "team_id": "T_WORKSPACE",
         "channel": {"id": "C_HERE", "context_team_id": "T_OTHER"},
         "message_ts": "1.1", "content": "contradictory"},
    ]}}

    out = await bot.execute_search_tool(_ctx(channel_id="C_HERE", action_token="tok"), {"query": "x"})

    assert out["count"] == 0


@pytest.mark.asyncio
async def test_search_skips_non_dict_result_entries(bot):
    """FIX 5: a malformed hit (a string/None in the results list) is skipped, not crashed on."""
    _set_infos(bot, {"C_HERE": _INTERNAL_HERE})
    bot.app.client.api_call.return_value = {"ok": True, "results": {"messages": [
        "garbage",
        None,
        {"channel_id": "C_HERE", "message_ts": "1.1", "content": "real"},
    ]}}

    out = await bot.execute_search_tool(_ctx(channel_id="C_HERE", action_token="tok"), {"query": "x"})

    assert out["count"] == 1 and out["results"][0]["text"] == "real"


# --------------------------------------------------------------- resolve_channel_name (id -> name)

def _info_side(bot_, mapping):
    """conversations.info keyed by channel; an id absent from the map raises channel_not_found
    (what Slack returns for a private conversation the bot can't see)."""
    from slack_sdk.errors import SlackApiError

    def _side(channel=None, **kw):
        entry = mapping.get(channel)
        if entry is None:
            raise SlackApiError("not_found", {"error": "channel_not_found"})
        return {"ok": True, "channel": entry}

    bot_.app.client.conversations_info.side_effect = _side


@pytest.mark.asyncio
async def test_resolve_public_channel_names_without_membership(bot):
    """The fix: a public channel's name is workspace-visible, so it resolves from a DM even
    though neither side is a member (the search-result / pizza case)."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, is_member=False, name="ai-discussion")
    res = await bot.execute_resolve_channel_name(
        _ctx(channel_id="D_MINE", is_dm=True), {"channel_id": "C_PUBLIC"})
    assert res == {"ok": True, "id": "C_PUBLIC", "name": "ai-discussion"}


@pytest.mark.asyncio
async def test_resolve_public_channel_in_channel_surface(bot):
    """Public source is deliverable into a (non-shared) channel too, so its name resolves there."""
    _info_side(bot, {
        "C_HERE": {"is_channel": True, "is_private": False},                 # destination: internal
        "C_PUBLIC": {"is_channel": True, "is_private": False, "name": "announcements"},
    })
    res = await bot.execute_resolve_channel_name(
        _ctx(channel_id="C_HERE", is_dm=False), {"channel_id": "C_PUBLIC"})
    assert res == {"ok": True, "id": "C_PUBLIC", "name": "announcements"}


@pytest.mark.asyncio
async def test_resolve_private_source_in_channel_withheld(bot):
    """A private channel the requester isn't in refuses with the generic message — the real name
    never appears anywhere in the payload."""
    _info_side(bot, {
        "C_HERE": {"is_channel": True, "is_private": False},
        "C_SECRET": {"is_channel": True, "is_private": True, "is_member": True,
                     "name": "project-secret"},
    })
    _member_of(bot, "C_HERE")  # requester is NOT in C_SECRET
    res = await bot.execute_resolve_channel_name(
        _ctx(channel_id="C_HERE", is_dm=False), {"channel_id": "C_SECRET"})
    assert res == {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
    assert "project-secret" not in str(res)


@pytest.mark.asyncio
async def test_resolve_private_channel_both_members_in_dm(bot):
    """In a DM (audience = the asker), a private channel BOTH share resolves via the strict gate."""
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=True, is_member=True, name="leadership")
    _member_of(bot, "C_SHARED")
    res = await bot.execute_resolve_channel_name(
        _ctx(channel_id="D_MINE", is_dm=True), {"channel_id": "C_SHARED"})
    assert res == {"ok": True, "id": "C_SHARED", "name": "leadership"}


@pytest.mark.asyncio
async def test_resolve_refuses_non_human_requester(bot):
    bot.app.client.conversations_info.return_value = _info(
        is_channel=True, is_private=False, name="anything")
    res = await bot.execute_resolve_channel_name(
        _ctx(channel_id="D_MINE", is_dm=True, human=False), {"channel_id": "C_PUBLIC"})
    assert res["error"] == "not_accessible"
    assert "name" not in res


@pytest.mark.asyncio
async def test_resolve_empty_id_is_bad_arguments(bot):
    res = await bot.execute_resolve_channel_name(_ctx(is_dm=True), {"channel_id": "  "})
    assert res["error"] == "bad_arguments"
