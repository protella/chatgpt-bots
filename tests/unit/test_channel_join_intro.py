"""Track 4 — channel join behavior (one-time intro + participation how-to).

Covers the trigger rules (bot's OWN join only; other joins / DMs / MPIMs ignored), the feature
flag, detachment (the event handler schedules a background task and returns without awaiting the
build/post), idempotency (a durable per-channel lease + a crash-then-retry reconcile via the Slack
message-metadata marker), the participation-state wording variants (on / mentions_only /
off — with `off` using the "won't respond even to tags" wording + Configure and NO plain-English
tuning line), the empty-channel case (omit the read + offers, still post the how-to), and the
Configure button + metadata marker on the posted message.

All I/O stubbed — no live bot, no network.
"""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from database import DatabaseManager
from message_processor.channel_summary import ChannelSummaryService
from slack_client.event_handlers.channel_join import (
    _HELLO_TEXT,
    CHANNEL_INTRO_METADATA_EVENT_TYPE,
    SlackChannelJoinMixin,
)
from slack_client.event_handlers.registration import SlackRegistrationMixin


# --------------------------------------------------------------------------- fixtures/helpers

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.makedirs"):
            db = DatabaseManager("test")
            db.db_path = f"{tmpdir}/test.db"
            if getattr(db, "conn", None):
                db.conn.close()
            db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
            db.conn.row_factory = sqlite3.Row
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.init_schema()
            yield db
            if getattr(db, "conn", None):
                db.conn.close()


class _JoinBot(SlackChannelJoinMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _make_bot(temp_db, *, summary=None, compose="This channel is about X. I can help with Y.",
              history=None, info=None):
    """A workflow-ready bot: real temp DB (so the lease is genuinely exercised) + mocked Slack
    web client + a fake processor exposing build_for_intro + the compose openai client."""
    bot = _JoinBot.__new__(_JoinBot)
    bot.bot_user_id = "UBOT"
    bot.db = temp_db
    bot.channel_pulse = None

    app = MagicMock()
    app.client = MagicMock()
    app.client.conversations_info = AsyncMock(
        return_value=info or {"channel": {"is_channel": True}})
    app.client.conversations_history = AsyncMock(return_value={"messages": list(history or [])})
    # Slack's real chat_postMessage returns a SlackResponse (Mapping-like, NOT a dict). Mirror that
    # with a SimpleNamespace-backed .get() so an isinstance(resp, dict) regression is caught here.
    _post_resp = SimpleNamespace(get=lambda k, d=None: {"ok": True, "ts": "1700.0001"}.get(k, d))
    app.client.chat_postMessage = AsyncMock(return_value=_post_resp)
    bot.app = app

    oc = SimpleNamespace(create_text_response=AsyncMock(return_value=compose))

    async def _build_for_intro(channel_id, *, client=None, pulse=None):
        return summary

    svc = SimpleNamespace(build_for_intro=_build_for_intro, openai_client=oc)
    bot.processor = SimpleNamespace(openai_client=oc, channel_summary_service=svc)
    return bot


def _actions_button(blocks):
    """Return the first action button dict in `blocks`, or None."""
    for b in blocks or []:
        if b.get("type") == "actions":
            for el in b.get("elements", []):
                if el.get("type") == "button":
                    return el
    return None


def _marker_message(ts="1699.5", user="UBOT"):
    return {"ts": ts, "user": user,
            "metadata": {"event_type": CHANNEL_INTRO_METADATA_EVENT_TYPE, "event_payload": {}}}


def _hello_kwargs(mock):
    """The top-level HELLO post (first chat_postMessage call)."""
    return mock.call_args_list[0].kwargs


def _findings_kwargs(mock):
    """The threaded FINDINGS reply (second chat_postMessage call)."""
    return mock.call_args_list[1].kwargs


# ------------------------------------------------------------------ registration + trigger rules

class _RegBot(SlackRegistrationMixin, SlackChannelJoinMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass

    def __init__(self):
        self.bot_user_id = "UBOT"
        self._handle_slack_message = AsyncMock()
        self._handle_channel_message = AsyncMock()
        self._register_settings_handlers = MagicMock()
        self._handlers = {}
        self.app = MagicMock()

        def _event(name):
            def _decorator(fn):
                self._handlers[name] = fn
                return fn
            return _decorator

        self.app.event = _event
        self._register_handlers()


def test_member_joined_channel_is_registered():
    bot = _RegBot()
    assert "member_joined_channel" in bot._handlers


def _trigger_bot():
    bot = _JoinBot.__new__(_JoinBot)
    bot.bot_user_id = "UBOT"
    bot._spawn_channel_intro = MagicMock()
    return bot


@pytest.mark.asyncio
async def test_fires_only_for_bot_own_join():
    bot = _trigger_bot()
    await bot._handle_member_joined_channel(
        {"user": "UBOT", "channel": "C1", "event_ts": "1.1"}, client=None)
    bot._spawn_channel_intro.assert_called_once()


@pytest.mark.asyncio
async def test_ignores_other_user_join():
    bot = _trigger_bot()
    await bot._handle_member_joined_channel(
        {"user": "UHUMAN", "channel": "C1"}, client=None)
    bot._spawn_channel_intro.assert_not_called()


@pytest.mark.asyncio
async def test_skips_dm_channel_prefix():
    bot = _trigger_bot()
    await bot._handle_member_joined_channel(
        {"user": "UBOT", "channel": "D9"}, client=None)
    bot._spawn_channel_intro.assert_not_called()


@pytest.mark.asyncio
async def test_flag_off_no_spawn(monkeypatch):
    monkeypatch.setattr(config, "enable_channel_join_intro", False)
    bot = _trigger_bot()
    await bot._handle_member_joined_channel(
        {"user": "UBOT", "channel": "C1"}, client=None)
    bot._spawn_channel_intro.assert_not_called()


@pytest.mark.asyncio
async def test_handler_detaches_and_does_not_await_inline():
    bot = _JoinBot.__new__(_JoinBot)
    bot.bot_user_id = "UBOT"
    ran = {"v": False}

    async def fake_run(channel_id, client, event_id=None):
        ran["v"] = True

    bot._run_channel_join_intro = fake_run
    await bot._handle_member_joined_channel(
        {"user": "UBOT", "channel": "C1", "event_ts": "1.1"}, client=None)
    # Detached: scheduled as a background task, NOT awaited inline — it hasn't run yet.
    assert ran["v"] is False
    tasks = list(getattr(bot, "_channel_intro_tasks", []))
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert ran["v"] is True


# ------------------------------------------------------------------ MPIM / DM exclusion (async)

@pytest.mark.asyncio
async def test_skips_mpim_group_dm(temp_db):
    bot = _make_bot(temp_db, summary="S", info={"channel": {"is_mpim": True}})
    await bot._run_channel_join_intro("G1", bot.app.client, "1.1")
    bot.app.client.chat_postMessage.assert_not_called()
    # No lease row created for a non-channel — the MPIM guard runs before the lease.
    assert await temp_db.get_channel_intro_async("G1") is None


@pytest.mark.asyncio
async def test_mpim_guard_fails_closed_on_error(temp_db):
    # Bug 3: an errored conversations.info must NOT fall through to posting into a possible MPIM.
    bot = _make_bot(temp_db, summary="S")
    bot.app.client.conversations_info = AsyncMock(side_effect=RuntimeError("api down"))
    await bot._run_channel_join_intro("G1", bot.app.client, "1.1")
    bot.app.client.chat_postMessage.assert_not_called()
    assert await temp_db.get_channel_intro_async("G1") is None


@pytest.mark.asyncio
async def test_mpim_guard_fails_closed_on_ambiguous_info(temp_db):
    # An info response that does not POSITIVELY identify a real channel (no is_channel/is_group/
    # is_private, and not obviously an MPIM either) is treated as unsafe → skip.
    bot = _make_bot(temp_db, summary="S", info={"channel": {}})
    await bot._run_channel_join_intro("G1", bot.app.client, "1.1")
    bot.app.client.chat_postMessage.assert_not_called()
    assert await temp_db.get_channel_intro_async("G1") is None


@pytest.mark.asyncio
async def test_private_channel_is_a_real_channel(temp_db):
    # Positive identification also covers private channels (is_group / is_private).
    bot = _make_bot(temp_db, summary="S", info={"channel": {"is_group": True, "is_private": True}})
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")
    assert bot.app.client.chat_postMessage.await_count == 2  # hello + findings


# ------------------------------------------------------------------ happy path + idempotency

@pytest.mark.asyncio
async def test_posts_hello_then_threaded_findings_with_button_and_marker(temp_db):
    bot = _make_bot(temp_db, summary="Channel about vendor pricing.")
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    # TWO posts: a top-level hello, then a threaded findings reply beneath it.
    assert bot.app.client.chat_postMessage.await_count == 2
    hello = _hello_kwargs(bot.app.client.chat_postMessage)
    findings = _findings_kwargs(bot.app.client.chat_postMessage)

    # 1) HELLO — top-level (no thread_ts), carries the reconcile marker, NO Configure button.
    assert hello["channel"] == "C1"
    assert "thread_ts" not in hello
    assert hello["metadata"]["event_type"] == CHANNEL_INTRO_METADATA_EVENT_TYPE
    assert _actions_button(hello.get("blocks")) is None

    # 2) FINDINGS — threaded under the hello's ts, carries the read + Configure button, no marker.
    assert findings["thread_ts"] == "1700.0001"
    assert "metadata" not in findings
    btn = _actions_button(findings["blocks"])
    assert btn is not None and btn["action_id"] == "open_channel_settings"
    assert "vendor pricing" in findings["text"] or "help" in findings["text"]

    # Lease recorded as posted with the HELLO's ts (the anchor).
    row = await temp_db.get_channel_intro_async("C1")
    assert row["status"] == "posted" and row["intro_ts"] == "1700.0001"


@pytest.mark.asyncio
async def test_second_call_does_not_repost(temp_db):
    bot = _make_bot(temp_db, summary="S")
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")
    assert bot.app.client.chat_postMessage.await_count == 2  # hello + findings
    # Refire: the lease is 'posted' → skip before any history scan.
    await bot._run_channel_join_intro("C1", bot.app.client, "2.2")
    assert bot.app.client.chat_postMessage.await_count == 2


@pytest.mark.asyncio
async def test_hello_is_deterministic_and_anchors_marker(temp_db):
    # Even with NO summary (empty channel → no model call), the deterministic hello still posts,
    # carries the marker, and its ts is what mark_posted records as the anchor.
    bot = _make_bot(temp_db, summary=None)
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    bot.processor.openai_client.create_text_response.assert_not_called()  # deterministic hello
    hello = _hello_kwargs(bot.app.client.chat_postMessage)
    assert hello["text"] == _HELLO_TEXT
    assert hello["metadata"]["event_type"] == CHANNEL_INTRO_METADATA_EVENT_TYPE
    assert "thread_ts" not in hello
    # Findings threaded under the hello; anchor ts is the hello's.
    assert _findings_kwargs(bot.app.client.chat_postMessage)["thread_ts"] == "1700.0001"
    assert (await temp_db.get_channel_intro_async("C1"))["intro_ts"] == "1700.0001"


@pytest.mark.asyncio
async def test_findings_reply_failure_does_not_repost_hello(temp_db):
    # The hello is the durable anchor. If the FINDINGS reply fails, the hello still stands and the
    # lease is marked posted (its ts), so a later refire reconciles instead of reposting the hello.
    bot = _make_bot(temp_db, summary="S")
    hello_resp = SimpleNamespace(get=lambda k, d=None: {"ok": True, "ts": "1700.0001"}.get(k, d))
    bot.app.client.chat_postMessage = AsyncMock(
        side_effect=[hello_resp, RuntimeError("findings down")])

    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")  # must not raise

    assert bot.app.client.chat_postMessage.await_count == 2  # hello ok, findings attempted+failed
    row = await temp_db.get_channel_intro_async("C1")
    assert row["status"] == "posted" and row["intro_ts"] == "1700.0001"


@pytest.mark.asyncio
async def test_hard_crash_pending_lease_reconciles_via_marker(temp_db):
    # The hard-crash case: a prior attempt POSTED (its marker is in history) then died BEFORE
    # mark_posted, so the lease is stuck 'pending'. A refire cannot ACQUIRE the pending lease, but
    # it must STILL reconcile from history and adopt the existing intro rather than skip-and-risk
    # a later double post.
    await temp_db.try_acquire_channel_intro_lease_async("C1", "0")   # left 'pending' (crash)
    bot = _make_bot(temp_db, summary="S", history=[_marker_message(ts="1699.5")])
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    bot.app.client.chat_postMessage.assert_not_called()
    row = await temp_db.get_channel_intro_async("C1")
    assert row["status"] == "posted" and row["intro_ts"] == "1699.5"


@pytest.mark.asyncio
async def test_pending_lease_without_marker_skips_without_reposting(temp_db):
    # A genuinely live 'pending' lease (another attempt is mid-flight, nothing in history yet):
    # the refire must skip — never post a second intro on top of a running attempt.
    await temp_db.try_acquire_channel_intro_lease_async("C1", "0")   # 'pending', no marker
    bot = _make_bot(temp_db, summary="S", history=[])
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")
    bot.app.client.chat_postMessage.assert_not_called()
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "pending"


@pytest.mark.asyncio
async def test_reacquired_failed_lease_reconciles_before_reposting(temp_db):
    # The soft-crash case: a prior attempt POSTED then errored on mark_posted and marked itself
    # 'failed'. The retry RE-ACQUIRES the failed lease, then reconciles history and adopts the
    # existing intro instead of reposting.
    lease = await temp_db.try_acquire_channel_intro_lease_async("C1", "0")  # 'pending'
    await temp_db.mark_channel_intro_failed_async("C1", lease["owner_token"])  # 'failed'

    bot = _make_bot(temp_db, summary="S", history=[_marker_message(ts="1699.5")])
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    bot.app.client.chat_postMessage.assert_not_called()
    row = await temp_db.get_channel_intro_async("C1")
    assert row["status"] == "posted" and row["intro_ts"] == "1699.5"


@pytest.mark.asyncio
async def test_reacquired_failed_lease_does_not_post_when_history_unavailable(temp_db):
    # Gap 1: after RE-acquiring a 'failed' lease, a history-fetch ERROR is indistinguishable from
    # "marker absent" — so we must NOT repost (the prior attempt may already have posted). Leave it
    # 'failed' so a later refire with readable history can retry.
    lease = await temp_db.try_acquire_channel_intro_lease_async("C1", "0")
    await temp_db.mark_channel_intro_failed_async("C1", lease["owner_token"])
    bot = _make_bot(temp_db, summary="S")
    bot.app.client.conversations_history = AsyncMock(side_effect=RuntimeError("history down"))

    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    bot.app.client.chat_postMessage.assert_not_called()
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "failed"


@pytest.mark.asyncio
async def test_fresh_acquire_posts_even_when_history_unavailable(temp_db):
    # A truly fresh acquire (no prior attempt) has nothing to double-post, so the reconcile is
    # skipped entirely — a transient history error must NOT block the first-ever intro.
    bot = _make_bot(temp_db, summary="S")
    bot.app.client.conversations_history = AsyncMock(side_effect=RuntimeError("history down"))

    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    assert bot.app.client.chat_postMessage.await_count == 2  # hello + findings
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "posted"


# ------------------------------------------------------------------ empty channel + offers

@pytest.mark.asyncio
async def test_empty_channel_omits_read_still_posts_howto(temp_db):
    bot = _make_bot(temp_db, summary=None)  # build_for_intro → None (empty/new channel)
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")

    # Hello still posts; findings still posts (how-to + button), just no read/offers.
    assert bot.app.client.chat_postMessage.await_count == 2
    # No summary → NO model compose call, so no invented read/offers.
    bot.processor.openai_client.create_text_response.assert_not_called()
    findings = _findings_kwargs(bot.app.client.chat_postMessage)
    assert findings["thread_ts"] == "1700.0001"
    # The how-to still posts, with the Configure button.
    assert "chime in when I can add something concrete" in findings["text"] \
        or "mentions only" in findings["text"] \
        or "Participation is currently off" in findings["text"]
    assert _actions_button(findings["blocks"])["action_id"] == "open_channel_settings"


@pytest.mark.asyncio
async def test_read_and_offers_come_only_from_grounded_summary(temp_db):
    # When there IS a summary, whatever the grounded compose returns rides verbatim into the
    # threaded findings — the bot never fabricates offers on its own.
    bot = _make_bot(temp_db, summary="Weekly paper club.",
                    compose="This is where the team discusses ML papers.")
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")
    bot.processor.openai_client.create_text_response.assert_awaited_once()
    text = _findings_kwargs(bot.app.client.chat_postMessage)["text"]
    assert "This is where the team discusses ML papers." in text


# ------------------------------------------------------------------ participation-state wording

def test_howto_on():
    txt = SlackChannelJoinMixin._participation_howto("on")
    assert "chime in when I can add something concrete" in txt
    assert "You can tune that in plain English" in txt


def test_howto_unknown_level_uses_the_quietest_speaking_wording():
    # An unrecognized level must never over-promise. resolve_participation_level degrades an
    # unknown to mentions_only, and the wording follows it rather than claiming full participation.
    txt = SlackChannelJoinMixin._participation_howto("judicious")
    assert "mentions only" in txt


def test_howto_mentions_only():
    txt = SlackChannelJoinMixin._participation_howto("mentions_only")
    assert "mentions only" in txt and "stay quiet unless you tag or name me" in txt
    assert "You can tune that in plain English" in txt


def test_howto_off_says_wont_respond_even_to_tags_and_omits_tuning():
    txt = SlackChannelJoinMixin._participation_howto("off")
    assert "won't respond even to tags" in txt
    assert "Use Configure to turn me on." in txt
    # `off` must NOT invite plain-English tuning (it can't reach the bot while off) and must
    # NOT use the softer "quiet unless asked" framing.
    assert "You can tune that in plain English" not in txt
    assert "quiet unless asked" not in txt


@pytest.mark.asyncio
@pytest.mark.parametrize("level,needle", [
    ("on", "chime in when I can add something concrete"),
    ("mentions_only", "mentions only"),
    ("off", "won't respond even to tags"),
])
async def test_end_to_end_wording_matches_channel_level(temp_db, level, needle):
    await temp_db.set_channel_settings_async("C1", participation_level=level, updated_by="U")
    bot = _make_bot(temp_db, summary="Some channel.")
    await bot._run_channel_join_intro("C1", bot.app.client, "1.1")
    # The participation how-to now lives in the threaded findings reply.
    text = _findings_kwargs(bot.app.client.chat_postMessage)["text"]
    assert needle in text
    if level == "off":
        assert "You can tune that in plain English" not in text


# ------------------------------------------------------------------ lease primitive

@pytest.mark.asyncio
async def test_lease_primitive_semantics(temp_db):
    # First claim wins (with a token); a second while 'pending' loses and reports the live status.
    first = await temp_db.try_acquire_channel_intro_lease_async("C1", "1")
    assert first["acquired"] is True and first["owner_token"]
    second = await temp_db.try_acquire_channel_intro_lease_async("C1", "2")
    assert second["acquired"] is False and second["status"] == "pending"
    # Posted → still locked, and the loser learns it's 'posted' (so the caller skips, not reconcile).
    await temp_db.mark_channel_intro_posted_async("C1", "9.9")
    third = await temp_db.try_acquire_channel_intro_lease_async("C1", "3")
    assert third["acquired"] is False and third["status"] == "posted"
    # mark_failed must NOT downgrade a posted intro (a late error can't reopen it).
    await temp_db.mark_channel_intro_failed_async("C1", first["owner_token"])
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "posted"


@pytest.mark.asyncio
async def test_failed_lease_can_be_reacquired(temp_db):
    lease = await temp_db.try_acquire_channel_intro_lease_async("C1", "1")
    await temp_db.mark_channel_intro_failed_async("C1", lease["owner_token"])
    # A genuine later refire may retry a failed attempt.
    assert (await temp_db.try_acquire_channel_intro_lease_async("C1", "2"))["acquired"] is True


# ------------------------------------------------------------------ Track 1 build coordination

def _summary_cfg(**over):
    base = dict(
        enable_channel_summaries=True, channel_summary_source_max=200,
        channel_summary_refresh_msgs=50, channel_summary_ttl_hours=24,
        channel_summary_max_chars=2000, channel_summary_input_max_chars=50000,
        channel_summary_max_output_tokens=600, channel_summary_failure_cooldown_hours=1,
        channel_summary_global_concurrency=2, utility_model="gpt-5.6-luna",
        utility_reasoning_effort="none", utility_verbosity="low", bot_name_aliases=["ChatGPT"],
    )
    base.update(over)
    return SimpleNamespace(**base)


class _SummaryClient:
    """Slack facade stand-in for ChannelSummaryService: conversations_history + classify_sender."""
    def __init__(self, messages, classify="human"):
        web = SimpleNamespace()
        web.conversations_history = AsyncMock(return_value={"messages": list(messages)})
        self.app = SimpleNamespace(client=web)
        self.user_cache = {}
        self._classify = classify

    def classify_sender(self, m):
        return self._classify


class _Pulse:
    """Minimal ChannelPulse stand-in: count_since(None)=total, count_since(ts)=newer."""
    def __init__(self, newer, total):
        self._n, self._t = newer, total

    def count_since(self, cid, after=None, *, exclude_self=False, top_level_only=False):
        return self._t if after is None else self._n


@pytest.mark.asyncio
async def test_posted_intro_excluded_from_next_summary_build(temp_db):
    # Bug 6: the bot's own join intro carries the metadata marker and must NOT be ingested into the
    # next Track 1 channel summary (even though its generated opener won't match _is_join_intro and
    # classifies as ordinary content here).
    captured = {"messages": None}

    async def _gen(**kwargs):
        captured["messages"] = kwargs.get("messages")
        return "NARRATIVE"

    stub = SimpleNamespace(create_text_response=AsyncMock(side_effect=_gen))
    svc = ChannelSummaryService(db=temp_db, openai_client=stub, config=_summary_cfg())
    client = _SummaryClient([
        {"ts": "1700.0", "user": "UALICE", "text": "Alice discusses roadmap planning"},
        {"ts": "1699.5", "user": "UBOT",
         "text": "Here is what this channel is about and how I can help",
         "metadata": {"event_type": CHANNEL_INTRO_METADATA_EVENT_TYPE, "event_payload": {}}},
    ])
    await svc._build("C1", client, None, 0)

    user_block = captured["messages"][1]["content"]
    assert "roadmap planning" in user_block            # real content ingested
    assert "how I can help" not in user_block           # the intro marker excluded it
    # And include_all_metadata was requested so the marker was present to filter on.
    assert client.app.client.conversations_history.call_args.kwargs.get("include_all_metadata") is True


@pytest.mark.asyncio
async def test_build_returns_none_when_save_rejected_by_optout(temp_db):
    # Bug 5: if ambient memory is disabled so the conditional save writes nothing, _build must NOT
    # return the generated text (it would otherwise leak an unsaved summary into the intro).
    await temp_db.set_channel_settings_async("C1", ambient_memory=False, updated_by="U")
    stub = SimpleNamespace(create_text_response=AsyncMock(return_value="NARRATIVE"))
    svc = ChannelSummaryService(db=temp_db, openai_client=stub, config=_summary_cfg())
    client = _SummaryClient([{"ts": "1700.0", "user": "UALICE", "text": "some real content"}])

    result = await svc._build("C1", client, None, 0)
    assert result is None
    assert await temp_db.get_channel_summary_async("C1") is None  # nothing persisted either


@pytest.mark.asyncio
async def test_build_for_intro_serializes_concurrent_builds(temp_db):
    # Bug 4: build_for_intro must not bypass Track 1's one-build-per-channel guarantee. Two
    # concurrent calls generate ONCE — the second waits on the build lock, re-reads, and reuses.
    calls = {"n": 0}

    async def _gen(**kwargs):
        calls["n"] += 1
        await asyncio.sleep(0)  # yield so both coroutines are genuinely in-flight
        return "NARRATIVE"

    stub = SimpleNamespace(create_text_response=AsyncMock(side_effect=_gen))
    svc = ChannelSummaryService(db=temp_db, openai_client=stub, config=_summary_cfg())
    client = _SummaryClient([{"ts": "1700.0", "user": "UALICE", "text": "roadmap chatter"}])

    r1, r2 = await asyncio.gather(
        svc.build_for_intro("C1", client=client, pulse=None),
        svc.build_for_intro("C1", client=client, pulse=None),
    )
    assert r1 == "NARRATIVE" and r2 == "NARRATIVE"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_detached_refresh_skips_rebuild_when_fresh_summary_appears_under_lock(temp_db):
    # Gap 2: _decide_and_build decides pre-lock, but must RE-READ + RE-DECIDE after acquiring the
    # build lock — a join-intro build that saved a fresh summary while the refresh waited on the
    # lock makes the rebuild redundant. Simulate that by returning "no row" at the pre-lock decide
    # and a fresh row under the lock; the rebuild must be skipped.
    stub = SimpleNamespace(create_text_response=AsyncMock(return_value="X"))
    svc = ChannelSummaryService(db=temp_db, openai_client=stub, config=_summary_cfg())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    fresh_row = {"summary_text": "already-fresh", "built_through_ts": "1700.0",
                 "generated_at": now, "invalidated_at": None}
    reads = {"n": 0}

    async def fake_get(cid):
        reads["n"] += 1
        return None if reads["n"] == 1 else fresh_row  # absent pre-lock, fresh once under the lock

    temp_db.get_channel_summary_async = fake_get
    svc._build = AsyncMock()

    await svc._decide_and_build("C1", client=None, pulse=_Pulse(newer=1, total=5))

    svc._build.assert_not_called()   # redundant rebuild skipped
    assert reads["n"] == 2           # decided pre-lock (None) then RE-decided under the lock (fresh)


@pytest.mark.asyncio
async def test_mark_failed_does_not_steal_a_concurrent_pending_lease(temp_db):
    # Bug 2: a task that FAILED (before/without winning the lease) must not downgrade the live
    # 'pending' lease a DIFFERENT attempt owns. mark_failed is token-scoped, so a stale/foreign
    # token is a no-op.
    live = await temp_db.try_acquire_channel_intro_lease_async("C1", "owner")  # the real owner
    await temp_db.mark_channel_intro_failed_async("C1", "some-other-token")     # a loser's token
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "pending"
    # The true owner can still fail its own lease.
    await temp_db.mark_channel_intro_failed_async("C1", live["owner_token"])
    assert (await temp_db.get_channel_intro_async("C1"))["status"] == "failed"
