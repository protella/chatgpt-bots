"""search_slack — BOTH backends (CHANNEL_SEARCH_REBUILD).

The DM half is Phase B's assistant.search.context path, unchanged and asserted byte-identical:
schema shape, action_token plumbing (event → metadata → ToolContext), executor success mapping,
missing-token fallback, the SEARCH_CHANNEL_TYPES code gate, API-error wrapping, registry gating.

The CHANNEL/MPIM half is the bot-token in-channel scan: matching, the trigger and receipt
exclusions, reply coverage and its budgets, the honest coverage block, the result contract, and
the surface split that keeps the two apart.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from slack_sdk.errors import SlackApiError

from config import BotConfig, config
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.search_tool import (SearchBackend, SlackSearchToolMixin, build_search_query,
                                      normalize_search_text, score_search_text,
                                      search_backend_for)
from tool_registry import ToolContext, ToolRegistry


class _Bot(SlackSearchToolMixin, SlackHistoryToolMixin):
    # The real SlackBot mixes search + history on ONE instance; the delivery-audience gate that
    # search now consults (self._source_is_public / self._bot_team_id) lives in the history mixin.
    def __init__(self):
        self.app = MagicMock()
        self.self_team_id = "T_TEST"
        self.bot_user_id = "U_BOT"

    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _ctx(**kw):
    defaults = dict(channel_id="C0BKX77NU66", thread_ts="1.0", trigger_ts="1.0",
                    action_token="tok-123")
    defaults.update(kw)
    return ToolContext(**defaults)


def _dm_ctx(**kw):
    """A DM context — the assistant-context backend's surface."""
    kw.setdefault("is_dm", True)
    return _ctx(**kw)


def _api_response(messages):
    return {"ok": True, "results": {"messages": messages}}


# --- schema ---

def test_schema_shape():
    schema = _Bot().get_search_tool_schema()
    assert schema["type"] == "function"
    assert schema["name"] == "search_slack"
    props = schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"query"}
    assert set(props["scope"]["enum"]) == {"channel", "workspace"}
    assert "limit" in props


def test_the_dm_schema_is_untouched_by_the_channel_rebuild():
    """DM tool bytes are a contract, and the channel rebuild must not have moved one of them."""
    dm = _Bot().get_search_tool_schema()
    assert dm == {
        "type": "function",
        "name": "search_slack",
        "description": (
            "Search Slack messages the bot is allowed to see (workspace-wide or the "
            "current channel). Use for finding older discussions, decisions, or context "
            "outside the current thread; prefer fetch_thread_messages/fetch_channel_history "
            "for things in the current conversation. Each result carries its source channel "
            "id; pass that to resolve_channel_name to show the channel's name."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "scope": {
                    "type": "string",
                    "enum": ["channel", "workspace"],
                    "description": "Limit results to the current channel, or search the whole workspace (default).",
                },
                "limit": {"type": "integer", "description": "Max results (1-20, default 10)."},
            },
            "required": ["query"],
        },
    }
    assert "thread_ts" not in json.dumps(dm)


def test_the_channel_schema_is_keyword_worded_current_channel_and_scopeless():
    """§S8. The channel surface describes the tool it actually has: a keyword scan of THIS
    channel, with no workspace reach to ask for and no `scope` to ask it with."""
    channel = _Bot().get_search_tool_channel_schema()
    props = channel["parameters"]["properties"]
    assert set(props) == {"query", "limit"}
    assert "scope" not in json.dumps(channel)
    description = channel["description"].lower()
    assert "keyword" in description
    assert "this channel" in description
    # No promise of workspace reach anywhere in the bytes the model reads.
    assert "workspace" in description and "never the wider workspace" in description
    # It still teaches the two things a channel result carries.
    assert "thread_ts" in channel["description"]
    assert "coverage" in description or "how much of the channel" in description


def test_the_channel_schema_is_static_and_ignores_the_thread_config():
    """A channel schema that varied with the request would fork the cached prefix."""
    bot = _Bot()
    assert (bot.get_search_tool_channel_schema({"model": "a"})
            == bot.get_search_tool_channel_schema({"model": "b"})
            == bot.get_search_tool_channel_schema())


def test_the_channel_schema_change_moves_the_pinned_tool_digest():
    """§S8's cache clause: the digest that pins a channel turn's tool set is computed FROM the
    schemas, so the rewritten description cannot be mistaken for the cached old one."""
    from message_processor.channel_request import tool_schema_version

    bot = _Bot()
    registry = ToolRegistry()
    registry.register(bot.get_search_tool_schema(), bot.execute_search_tool,
                      channel_schema=bot.get_search_tool_channel_schema)
    now = tool_schema_version(registry, {})

    legacy = ToolRegistry()
    legacy.register(bot.get_search_tool_schema(), bot.execute_search_tool,
                    channel_schema=lambda cfg=None: bot.get_search_tool_schema())
    assert now != tool_schema_version(legacy, {})


# --- action_token plumbing ---

@pytest.mark.asyncio
async def test_event_to_message_captures_action_token():
    """_event_to_message must copy the event's action_token into metadata."""
    from slack_client.base import SlackBot
    bot = SlackBot.__new__(SlackBot)  # no __init__ — we only exercise _event_to_message
    bot.bot_user_id = "U07SELF"
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot.get_username = AsyncMock(return_value="peter")
    bot.get_user_timezone = AsyncMock(return_value="UTC")

    event = {"text": "find that thread", "user": "U1", "channel": "C1", "ts": "2.0",
             "action_token": "tok-evt"}
    msg = await bot._event_to_message(event, client=MagicMock())
    assert msg.metadata["action_token"] == "tok-evt"

    event_without = {"text": "hi", "user": "U1", "channel": "C1", "ts": "3.0"}
    msg2 = await bot._event_to_message(event_without, client=MagicMock())
    assert msg2.metadata["action_token"] is None


def test_tool_context_built_from_metadata():
    """The Phase A ToolContext builder passes metadata['action_token'] through."""
    from message_processor.handlers.text import TextHandlerMixin  # noqa: F401 — import proves wiring exists
    import inspect
    import message_processor.handlers.text as text_mod
    src = inspect.getsource(text_mod)
    assert 'action_token=meta.get("action_token")' in src


# --- executor ---

@pytest.mark.asyncio
async def test_search_success_maps_results():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([
        {"channel_id": "C09", "message_ts": "100.1", "author_user_id": "U9",
         "content": "we decided fridays", "permalink": "https://x/p1"},
        {"channel": {"id": "C08"}, "ts": "90.2", "user": "U8", "text": "older note"},
    ]))
    # DM surface: full reach, so the field-mapping + request plumbing this test covers isn't
    # touched by the delivery-audience filter (which is exercised in test_channel_scope_guard.py).
    out = await bot.execute_search_tool(_ctx(is_dm=True), {"query": "demo day"})
    assert out["ok"] is True and out["count"] == 2
    first = out["results"][0]
    # DM BYTES ARE PRE-W3, EXACTLY. `thread_ts` is a CHANNEL-surface field (§5.2): it exists to
    # make a root a legal post_to_thread target, and a DM has no stream, no allowlist and nothing
    # to do with it. Pinned as a whole dict on purpose — an extra key here is a silent change to
    # what every DM turn reads.
    assert first == {"channel": "C09", "ts": "100.1", "author": "U9",
                     "text": "we decided fridays", "permalink": "https://x/p1"}
    # THE WHOLE REQUEST DICT, not three fields (codex review #10). "DM request bytes unchanged"
    # is a promise about every key: `query`, `content_types`, `limit` and the key SET itself all
    # decide what Slack searches and what it charges for, and a partial assertion would let any
    # of them move silently.
    kwargs = bot.app.client.api_call.call_args
    assert kwargs.args[0] == "assistant.search.context"
    assert kwargs.kwargs["data"] == {
        "query": "demo day",
        "action_token": "tok-123",
        "channel_types": "public_channel,private_channel",
        "content_types": "messages",
        "limit": 10,
        "context_channel_id": "C0BKX77NU66",
    }


@pytest.mark.asyncio
async def test_scope_channel_filters_other_channels():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([
        {"channel_id": "C0BKX77NU66", "message_ts": "1.1", "content": "here"},
        {"channel_id": "C_OTHER", "message_ts": "1.2", "content": "elsewhere"},
    ]))
    out = await bot.execute_search_tool(_dm_ctx(), {"query": "x", "scope": "channel"})
    assert out["count"] == 1 and out["results"][0]["channel"] == "C0BKX77NU66"


@pytest.mark.asyncio
async def test_scope_channel_constrains_query_at_api(monkeypatch):
    """F22: channel scope must append `in:<#CHANNEL_ID>` so the API constrains at the
    source, not just the post-filter (a workspace-wide top-N could miss the channel)."""
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_dm_ctx(), {"query": "budget", "scope": "channel"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["query"]
    assert sent == "budget in:<#C0BKX77NU66>"


@pytest.mark.asyncio
async def test_scope_workspace_query_unmodified():
    """F22: workspace scope must NOT inject an in:<#...> operator."""
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_dm_ctx(), {"query": "budget", "scope": "workspace"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["query"]
    assert sent == "budget"
    # and the human-facing echo keeps the original query
    out = await bot.execute_search_tool(_dm_ctx(), {"query": "budget"})
    assert out["query"] == "budget"


@pytest.mark.asyncio
async def test_missing_token_falls_back():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock()
    out = await bot.execute_search_tool(_dm_ctx(action_token=None), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_unavailable"
    assert "fetch_channel_history" in out["hint"]
    bot.app.client.api_call.assert_not_called()


@pytest.mark.asyncio
async def test_empty_query_rejected():
    out = await _Bot().execute_search_tool(_dm_ctx(), {"query": "  "})
    assert out["ok"] is False and out["error"] == "bad_arguments"


# --- channel-type gate ---

@pytest.mark.asyncio
async def test_channel_types_gate_excludes_im_by_default():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["channel_types"]
    assert "im" not in sent.split(",") and "mpim" not in sent.split(",")


@pytest.mark.asyncio
async def test_channel_types_gate_env_widening_and_validation():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    with patch.object(config, "search_channel_types", ["public_channel", "im", "bogus_type"]):
        await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["channel_types"]
    assert sent == "public_channel,im"  # bogus filtered, im honored when configured


@pytest.mark.asyncio
async def test_no_valid_channel_types_disables_search():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock()
    with patch.object(config, "search_channel_types", ["bogus"]):
        out = await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_disabled"
    bot.app.client.api_call.assert_not_called()


# --- error wrapping ---

def _slack_error(err):
    resp = MagicMock()
    resp.get = lambda k, d=None: {"error": err}.get(k, d)
    return SlackApiError(message=err, response=resp)


@pytest.mark.asyncio
async def test_token_error_becomes_search_unavailable():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(side_effect=_slack_error("invalid_action_token"))
    out = await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_unavailable"


@pytest.mark.asyncio
async def test_api_error_wrapped_never_raises():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(side_effect=_slack_error("ratelimited"))
    out = await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "ratelimited"

    bot.app.client.api_call = AsyncMock(side_effect=RuntimeError("boom"))
    out2 = await bot.execute_search_tool(_dm_ctx(), {"query": "x"})
    assert out2["ok"] is False and out2["error"] == "exception"


# --- registry gating + guidance ---

def test_registry_gating():
    bot = _Bot()
    registry = ToolRegistry()
    if config.enable_search_tool:
        registry.register(bot.get_search_tool_schema(), bot.execute_search_tool)
    names = [s["name"] for s in registry.schemas()]
    assert ("search_slack" in names) == config.enable_search_tool

    # default is enabled
    assert config.enable_search_tool is True


def test_limit_clamping():
    clamp = SlackSearchToolMixin._clamp_search_limit
    assert clamp(None) == 10
    assert clamp("nope") == 10
    assert clamp(0) == 1
    assert clamp(500) == 20


def test_guidance_mentions_search():
    from prompts import LOCAL_TOOLS_GUIDANCE
    assert "search_slack" in LOCAL_TOOLS_GUIDANCE
    # BF1: the guidance is availability-conditional now — when search_slack is not among the
    # available tools (no action_token), fall to the fetch tools without comment.
    assert "When search_slack is available" in LOCAL_TOOLS_GUIDANCE
    assert "not among the available tools" in LOCAL_TOOLS_GUIDANCE


# =====================================================================================
# The CHANNEL backend — the bot-token in-channel scan (CHANNEL_SEARCH_REBUILD §S1-S11)
# =====================================================================================

from slack_client import search_tool as search_mod  # noqa: E402
from tool_registry import serialize_tool_result  # noqa: E402

CH = "C0BKX77NU66"


def _m(ts, text, *, user="U_HUMAN", thread_ts=None, reply_count=None, latest_reply=None,
       **extra):
    """One raw Slack message, shaped as conversations.history returns it."""
    msg = {"ts": ts, "text": text, "user": user, "team": "T_TEST"}
    if thread_ts:
        msg["thread_ts"] = thread_ts
    if reply_count:
        msg["reply_count"] = reply_count
    if latest_reply:
        msg["latest_reply"] = latest_reply
    msg.update(extra)
    return msg


def _self_m(ts, text, **extra):
    return _m(ts, text, user="U_BOT", **extra)


class _FakeSlack:
    """A Slack client that answers ONLY what the scan is allowed to call.

    Anything else raises AttributeError naturally (it is a plain class, not a mock), which is how
    "the channel path never posts, never lists users, never touches the assistant index" is
    enforced rather than asserted after the fact.
    """

    def __init__(self, *, history=None, replies=None, info=None):
        self.history_pages = history if history is not None else [{"messages": []}]
        self.reply_pages = replies or {}
        self.info = info or {"is_channel": True, "is_private": False, "is_member": True}
        self.history_calls = []
        self.reply_calls = []
        self.permalink_calls = []
        self.permalink_fails = False

    async def conversations_info(self, **kwargs):
        return {"ok": True, "channel": dict(self.info)}

    @staticmethod
    def _page(pages, cursor):
        index = int(cursor) if cursor else 0
        if index >= len(pages):
            return {"ok": True, "messages": []}
        page = pages[index]
        if page.get("raise") is not None:
            raise page["raise"]
        out = {"ok": True, "messages": list(page.get("messages") or [])}
        if "has_more" in page or "next_cursor" in page:
            if page.get("has_more"):
                out["has_more"] = True
            if page.get("next_cursor"):
                out["response_metadata"] = {"next_cursor": page["next_cursor"]}
        elif index + 1 < len(pages):
            out["has_more"] = True
            out["response_metadata"] = {"next_cursor": str(index + 1)}
        return out

    async def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self._page(self.history_pages, kwargs.get("cursor"))

    async def conversations_replies(self, **kwargs):
        self.reply_calls.append(kwargs)
        pages = self.reply_pages.get(kwargs.get("ts"), [{"messages": []}])
        return self._page(pages, kwargs.get("cursor"))

    async def chat_getPermalink(self, channel, message_ts):
        self.permalink_calls.append((channel, message_ts))
        if self.permalink_fails:
            raise SlackApiError(message="nope", response=MagicMock())
        return {"ok": True, "permalink": f"https://slack/{channel}/p{message_ts}"}

    async def api_call(self, *args, **kwargs):  # pragma: no cover - the assertion IS the point
        raise AssertionError(
            "the channel backend called the assistant index: " + str(args))


class _FakeDb:
    """Receipt evidence only. Every OTHER database call is a failure — the scan may READ the
    receipt rows and must never write a transcript row."""

    def __init__(self, *, receipts=None, epoch=None, fail=False):
        self.receipts = dict(receipts or {})
        self.epoch = epoch
        self.fail = fail
        self.reads = []
        self.busy_timeouts = []

    async def read_channel_sidecars_for_async(self, team_id, channel_id, message_ts,
                                              busy_timeout_ms=None):
        ids = list(message_ts or ())
        self.reads.append((team_id, channel_id, ids))
        self.busy_timeouts.append(busy_timeout_ms)
        if self.fail:
            raise RuntimeError("database unavailable")
        wanted = set(ids)
        return {
            "receipt_feature_epoch_ts": self.epoch,
            "receipts": [{"message_ts": ts, "state": state}
                         for ts, state in self.receipts.items() if ts in wanted],
        }

    def __getattr__(self, name):
        raise AssertionError(f"the scan reached for the database method {name!r}")


class _Flight:
    def __init__(self):
        self.staged_roots = []


class _ScanBot(_Bot):
    def __init__(self, client):
        super().__init__()
        self.app = MagicMock()
        self.app.client = client
        self.resolve_usernames = AsyncMock(return_value={"U_HUMAN": "dana"})

    def classify_sender(self, msg):
        if msg.get("user") == self.bot_user_id:
            return "self"
        if msg.get("bot_id"):
            return "other_bot"
        return "human"


def _scan_ctx(*, db=None, trigger_ts="500.000000", channel_id=CH, **kw):
    ctx = ToolContext(channel_id=channel_id, thread_ts=None, trigger_ts=trigger_ts,
                      action_token=None, user_id="U_HUMAN", db=db, is_dm=False,
                      requester_is_human=True, origin_membership_attested=True, **kw)
    ctx.tool_flight = _Flight()
    return ctx


async def _run_scan(client, *, query="cert renewal", ctx=None, args=None, db=None):
    bot = _ScanBot(client)
    ctx = ctx if ctx is not None else _scan_ctx(db=db)
    payload = await bot.execute_search_tool(ctx, {**{"query": query}, **(args or {})})
    return bot, ctx, payload


# --- matching (§S2, pure) -------------------------------------------------------------

def test_exact_phrase_outranks_a_token_match():
    query = build_search_query("loading dock resurfacing")
    phrase = score_search_text(query, normalize_search_text(
        "the loading dock resurfacing is booked"))
    tokens = score_search_text(query, normalize_search_text(
        "resurfacing the yard and the dock, plus loading bays"))
    assert phrase[0] == 1 and tokens[0] == 0
    assert phrase > tokens


def test_matching_is_or_not_and():
    """A query never has to appear in full: one distinctive token qualifies."""
    query = build_search_query("which supplier quoted for the resurfacing")
    assert score_search_text(query, normalize_search_text("Kestwood rang about the dock")) is None
    assert score_search_text(query, normalize_search_text("the resurfacing bid landed")) is not None
    # …and more distinct tokens ranks higher than fewer.
    one = score_search_text(query, normalize_search_text("the resurfacing bid landed"))
    two = score_search_text(query, normalize_search_text("the supplier for the resurfacing"))
    assert two > one


def test_tokens_match_on_word_boundaries_only():
    query = build_search_query("cert")
    assert score_search_text(query, normalize_search_text("the cert expires")) is not None
    assert score_search_text(query, normalize_search_text("certificate authority")) is None
    # The phrase test is bounded too — `cert` must not hit inside `certificate`.
    assert build_search_query("cert").phrase_hit("certificate authority") is False


def test_case_and_unicode_are_normalized():
    query = build_search_query("Café RÉSUMÉ")
    assert score_search_text(query, normalize_search_text("café résumé attached")) is not None
    # NFKC folds the fullwidth forms onto the same tokens.
    assert score_search_text(build_search_query("BUDGET"),
                             normalize_search_text("ＢＵＤＧＥＴ approved")) is not None


def test_comma_numbers_and_zero_cents_match_the_same_figure():
    query = build_search_query("41,770")
    for written in ("we paid 41770 in the end", "the quote was $41,770", "invoice for 41,770.00"):
        assert score_search_text(query, normalize_search_text(written)) is not None, written


def test_non_zero_cents_are_a_different_figure():
    query = build_search_query("41,770")
    assert score_search_text(query, normalize_search_text("they quoted $41,770.50")) is None


def test_a_stopword_only_query_qualifies_on_the_phrase_alone():
    query = build_search_query("what did we do about it")
    assert query.content_tokens == ()
    assert score_search_text(query, normalize_search_text(
        "So what did we do about it in the end?")) is not None
    assert score_search_text(query, normalize_search_text("we did nothing")) is None


def test_the_stopword_set_is_pinned_member_for_member():
    """§S2's golden (codex review #8). The set decides which words can never make a message
    qualify, so it is retrieval policy, not an implementation detail — and a set that only ever
    grows by accident is how a search quietly stops finding "no" or "one" or "back". Pinned in
    full so that editing it is a deliberate two-place change, here and in the module.

    TWO PROPERTIES BEYOND THE MEMBERSHIP, because they are the ones a careless edit breaks:
    every member is already normalized (an entry with a capital or a space could never match a
    token), and NO MEMBER IS A NUMBER — a figure is usually the most distinctive thing in a
    question, and stopwording one would be a retrieval bug nobody would think to look for.
    """
    from slack_client.search_tool import _SEARCH_STOPWORDS

    assert _SEARCH_STOPWORDS == frozenset({
        "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "anyone", "are",
        "as", "at", "back", "be", "because", "been", "before", "being", "but", "by", "can",
        "could", "did", "do", "does", "doing", "done", "down", "each", "even", "ever", "every",
        "for", "from", "get", "got", "had", "has", "have", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "know", "like", "me",
        "might", "mine", "more", "most", "much", "must", "my", "no", "nor", "not", "now", "of",
        "off", "on", "one", "only", "or", "other", "our", "ours", "out", "over", "own", "really",
        "said", "same", "say", "says", "she", "should", "so", "some", "still", "such", "than",
        "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "thing",
        "things", "this", "those", "to", "too", "up", "us", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
        "your", "yours",
    })
    assert all(word == normalize_search_text(word) for word in _SEARCH_STOPWORDS)
    assert not any(any(ch.isdigit() for ch in word) for word in _SEARCH_STOPWORDS)


@pytest.mark.asyncio
async def test_recency_breaks_an_equal_score_tie():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "the resurfacing quote"),
        _m("200.000000", "the resurfacing quote"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="resurfacing quote")
    assert [r["ts"] for r in payload["results"]] == ["300.000000", "200.000000"]


# --- exclusions (§S3) -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_trigger_and_anything_after_it_are_excluded():
    """The message we are answering is usually the newest and strongest match for its own
    question, and nothing posted after it was part of the room this turn reasons about."""
    client = _FakeSlack(history=[{"messages": [
        _m("600.000000", "cert renewal — who is it with?"),   # after the trigger
        _m("500.000000", "cert renewal — who is it with?"),   # IS the trigger
        _m("400.000000", "cert renewal is stuck with Kestwood"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [r["ts"] for r in payload["results"]] == ["400.000000"]
    assert payload["coverage"]["messages_scanned"] == 1
    # Asked of Slack as well as enforced locally.
    assert client.history_calls[0]["latest"] == "500.000000"
    assert client.history_calls[0]["inclusive"] is False


@pytest.mark.asyncio
async def test_a_finalized_own_message_is_searchable():
    client = _FakeSlack(history=[{"messages": [
        _self_m("400.000000", "the cert renewal is with Kestwood, quoted 41,770"),
    ]}])
    db = _FakeDb(receipts={"400.000000": "finalized"}, epoch="1.000000")
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", db=db)
    assert payload["count"] == 1
    assert payload["coverage"]["messages_scanned"] == 1
    assert db.reads and db.reads[0][2] == ["400.000000"]


@pytest.mark.asyncio
async def test_an_in_flight_own_message_is_excluded():
    """A direct scan can see the half-written reply the old index's lag hid — and that reply,
    being an answer to this very question, is exactly what would rank first."""
    client = _FakeSlack(history=[{"messages": [
        _self_m("450.000000", "cert renewal: looking into it now"),
        _m("400.000000", "cert renewal is stuck with Kestwood"),
    ]}])
    db = _FakeDb(receipts={"450.000000": "in_flight"}, epoch="1.000000")
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", db=db)
    assert [r["ts"] for r in payload["results"]] == ["400.000000"]
    assert payload["coverage"]["messages_scanned"] == 1


@pytest.mark.asyncio
async def test_own_messages_without_readable_evidence_are_excluded():
    """No database, or a database that will not answer, excludes every own message — the same
    direction the channel stream fails in."""
    messages = [{"messages": [_self_m("400.000000", "cert renewal with Kestwood")]}]
    _b1, _c1, no_db = await _run_scan(_FakeSlack(history=messages), db=None)
    _b2, _c2, broken = await _run_scan(_FakeSlack(history=messages), db=_FakeDb(fail=True))
    assert no_db["count"] == 0 and no_db["coverage"]["messages_scanned"] == 0
    assert broken["count"] == 0 and broken["coverage"]["messages_scanned"] == 0

    # A READ THAT FAILED IS A GAP, AND THE NOTE MUST NAME THE RIGHT SYSTEM. A failure of the
    # local receipt ledger used to fall through to the generic "stopped after a Slack API
    # failure" tail, which is a false explanation — and the one sentence a person actually
    # reads, so it would send an operator to the Slack dashboard for a SQLite problem.
    assert broken["coverage"]["complete"] is False
    assert broken["coverage"]["stopped_reason"] == "receipt_read_failed"
    # THE WHOLE SENTENCE, AS A LITERAL. A substring check passes with the conclusion deleted,
    # and "so none of them were searched" is the load-bearing half — it is what tells a reader
    # that own messages are missing from THESE results rather than merely that a read failed
    # somewhere. Written out here rather than recomputed from `coverage_note`, so this test is
    # an independent oracle for the wording and not a mirror of the code that produces it.
    assert broken["coverage"]["note"] == (
        "Searched 0 messages across 0 threads in this channel, newest activity first; could not "
        "read the receipt evidence for its own messages, so none of them were searched. Some of "
        "this channel was NOT read, so absence here is not evidence of absence in the channel.")

    # No database at all is structural, not a failure: own messages are simply never searchable
    # on such a context, and the scan does not claim to have been cut short.
    assert no_db["coverage"]["stopped_reason"] is None
    assert no_db["coverage"]["complete"] is True


@pytest.mark.asyncio
async def test_a_pre_epoch_own_message_is_grandfathered_but_chrome_is_not():
    """Grandfathering admits what we SAID, never our own furniture.

    THE CHROME HALF USES A REAL CHROME SHAPE (codex review #9): ":emoji: Thinking…" is what the
    placeholder actually looks like, and `is_self_chrome_message` is the same classifier the
    channel stream uses. It is written so that it MATCHES THE QUERY — otherwise it would be
    excluded for saying nothing relevant and the test would prove nothing about chrome.
    """
    answer = _self_m("400.000000", "cert renewal with Kestwood")
    chrome = _self_m("410.000000", ":hourglass_flowing_sand: Thinking... about the cert renewal")
    db = _FakeDb(receipts={}, epoch="450.000000")   # both messages predate the epoch
    _bot, _ctx_, payload = await _run_scan(
        _FakeSlack(history=[{"messages": [chrome, answer]}]), query="cert renewal", db=db)
    assert [r["ts"] for r in payload["results"]] == ["400.000000"], (
        "the reply is grandfathered; the placeholder is not a thing we said")
    # The chrome message is not counted as searched either — it was never readable content.
    assert payload["coverage"]["messages_scanned"] == 1

    # …and the same reply AFTER the epoch, with no row, is not evidence of a reply at all.
    db_after = _FakeDb(receipts={}, epoch="100.000000")
    _b2, _c2, later = await _run_scan(_FakeSlack(history=[{"messages": [
        _self_m("400.000000", "cert renewal with Kestwood")]}]), db=db_after)
    assert later["count"] == 0


@pytest.mark.asyncio
async def test_a_scan_with_no_usable_trigger_is_refused_before_it_fetches():
    """§S3's fence is a PRECONDITION (codex review #6). An absent or unparseable trigger used to
    become "no bound", which let a direct or replayed context search the present — including the
    message being answered. It is now an honest refusal that names a fallback, and Slack is never
    called at all."""
    for trigger in (None, "", "not-a-timestamp"):
        client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}])
        bot = _ScanBot(client)
        out = await bot.execute_search_tool(_scan_ctx(trigger_ts=trigger), {"query": "cert"})
        assert out["ok"] is False, trigger
        assert out["error"] == "search_unavailable"
        assert "fetch_channel_history" in out["hint"]
        assert client.history_calls == []


@pytest.mark.asyncio
async def test_an_unreadable_message_is_skipped_and_the_coverage_says_so():
    """codex review #2. A message the scan cannot read is a HOLE, not a decision.

    Skipping it is still right — one bad payload must not end a scan — but the block may not
    then claim it read everything. A DECLINED subtype (a join notice) is the other case and is
    not a hole at all: it is not a message, and coverage stays complete.
    """
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal is with Kestwood"),
        _m("abc", "cert renewal, but nobody can place it in time"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [r["ts"] for r in payload["results"]] == ["300.000000"]
    assert payload["coverage"]["complete"] is False
    assert payload["coverage"]["stopped_reason"] == "history_data_invalid"
    assert "could not read" in payload["coverage"]["note"]

    joined = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal is with Kestwood"),
        _m("290.000000", "has joined the channel", subtype="channel_join"),
    ]}])
    _b2, _c2, declined = await _run_scan(joined, query="cert renewal")
    assert declined["coverage"]["complete"] is True, "a join notice is not a message, not a gap"
    assert declined["coverage"]["messages_scanned"] == 1


@pytest.mark.asyncio
async def test_a_non_string_timestamp_is_refused_at_the_scan_boundary():
    """codex review #4. `secondary_ts` stringifies whatever it is handed, so a JSON number in
    `thread_ts` would arrive looking like a root and flow into `stage_discovered_root` — widening
    where this turn may post, off a value Slack never sends. Refused here, at the boundary, with
    the shared normalizer left alone. Same rule for `ts`, the identity the model quotes back."""
    for bad in ({"thread_ts": 300.0001}, {"ts": 300.0001}):
        message = _m("300.000000", "cert renewal is with Kestwood")
        message.update(bad)
        client = _FakeSlack(history=[{"messages": [message]}])
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
        assert payload["count"] == 0, bad
        assert payload["coverage"]["stopped_reason"] == "history_data_invalid"


@pytest.mark.asyncio
async def test_an_undeliverable_candidate_cannot_evict_a_deliverable_one():
    """codex review #3. The pool holds `limit` entries, so WHEN the delivery rule runs decides
    whether a channel with a perfectly good answer gets one.

    The foreign-workspace message is NEWER and therefore ranks first. Filtered after the
    competition, it would take the only seat and then be dropped, and the tool would report no
    matches. Checked at qualification time, it never competes.
    """
    client = _FakeSlack(history=[{"messages": [
        _m("400.000000", "cert renewal is with Kestwood", team="T_ELSEWHERE"),
        _m("300.000000", "cert renewal is with Kestwood"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", args={"limit": 1})
    assert [r["ts"] for r in payload["results"]] == ["300.000000"]


# --- coverage (§S4, §S7) --------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_reply_matches_when_its_root_matches_nothing():
    """The whole defect: the fact lives in a reply under an ordinary root."""
    client = _FakeSlack(
        history=[{"messages": [_m("300.000000", "weekly infra sync", reply_count=1,
                                  latest_reply="310.000000")]}],
        replies={"300.000000": [{"messages": [
            _m("300.000000", "weekly infra sync", reply_count=1, latest_reply="310.000000"),
            _m("310.000000", "cert renewal is stuck with Kestwood — they quoted 41,770"),
        ]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal 41,770")
    assert [r["ts"] for r in payload["results"]] == ["310.000000"]
    assert payload["results"][0]["thread_ts"] is None or payload["coverage"]["threads_scanned"] == 1
    assert payload["coverage"]["threads_scanned"] == 1
    assert payload["coverage"]["complete"] is True


@pytest.mark.asyncio
async def test_every_reply_bearing_root_is_fetched_not_only_matching_ones():
    client = _FakeSlack(
        history=[{"messages": [
            _m("300.000000", "weekly infra sync", reply_count=1, latest_reply="330.000000"),
            _m("200.000000", "lunch thread", reply_count=1, latest_reply="220.000000"),
            _m("100.000000", "a bare message with no replies"),
        ]}],
        replies={"300.000000": [{"messages": [_m("330.000000", "nothing relevant")]}],
                 "200.000000": [{"messages": [
                     _m("220.000000", "cert renewal is with Kestwood")]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    fetched = [call["ts"] for call in client.reply_calls]
    assert fetched == ["300.000000", "200.000000"]  # newest activity first
    assert "100.000000" not in fetched
    assert [r["ts"] for r in payload["results"]] == ["220.000000"]


@pytest.mark.asyncio
async def test_a_thread_broadcast_discovers_a_root_older_than_the_span():
    client = _FakeSlack(
        history=[{"messages": [
            _m("400.000000", "also mentioning it here", thread_ts="9.000000",
               subtype="thread_broadcast"),
        ]}],
        replies={"9.000000": [{"messages": [
            _m("9.000000", "an old root"),
            _m("10.000000", "cert renewal is with Kestwood"),
        ]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [call["ts"] for call in client.reply_calls] == ["9.000000"]
    assert [r["ts"] for r in payload["results"]] == ["10.000000"]


@pytest.mark.asyncio
async def test_both_pagers_follow_strict_cursors():
    client = _FakeSlack(
        history=[{"messages": [_m("300.000000", "sync", reply_count=1)]},
                 {"messages": [_m("200.000000", "cert renewal one")]}],
        replies={"300.000000": [
            {"messages": [_m("310.000000", "nothing")]},
            {"messages": [_m("320.000000", "cert renewal two")]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert client.history_calls[1]["cursor"] == "1"
    assert client.reply_calls[1]["cursor"] == "1"
    assert {r["ts"] for r in payload["results"]} == {"200.000000", "320.000000"}
    assert payload["coverage"]["history_pages"] == 2
    assert payload["coverage"]["reply_pages"] == 2


@pytest.mark.asyncio
async def test_messages_scanned_is_deduplicated_across_root_and_reply_pages():
    """A reply-bearing root arrives twice — in the history page and as replies' first entry."""
    root = _m("300.000000", "weekly sync", reply_count=1, latest_reply="310.000000")
    client = _FakeSlack(history=[{"messages": [root]}],
                        replies={"300.000000": [{"messages": [root, _m("310.000000", "hi")]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert payload["coverage"]["messages_scanned"] == 2


@pytest.mark.asyncio
async def test_the_history_page_ceiling_stops_the_walk_honestly():
    pages = [{"messages": [_m(f"{400 - i}.000000", "chatter")], "has_more": True,
              "next_cursor": str(i + 1)} for i in range(5)]
    client = _FakeSlack(history=pages)
    with patch.object(config, "search_history_page_ceiling", 2):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage["history_pages"] == 2
    assert coverage["stopped_reason"] == "history_page_ceiling"
    assert coverage["complete"] is False
    assert "history-page budget" in coverage["note"]


@pytest.mark.asyncio
async def test_the_reply_page_ceiling_is_global_across_roots():
    roots = [_m(f"{300 - i}.000000", "sync", reply_count=1) for i in range(5)]
    client = _FakeSlack(
        history=[{"messages": roots}],
        replies={r["ts"]: [{"messages": [_m(f"{r['ts']}1", "chatter")]}] for r in roots})
    with patch.object(config, "search_reply_page_ceiling", 2), \
            patch.object(config, "search_reply_fetch_concurrency", 1):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage["reply_pages"] == 2
    assert coverage["stopped_reason"] == "reply_page_ceiling"
    assert coverage["complete"] is False


@pytest.mark.asyncio
async def test_the_per_thread_ceiling_stops_one_pathological_thread():
    deep = [{"messages": [_m(f"31{i}.000000", "chatter")], "has_more": True,
             "next_cursor": str(i + 1)} for i in range(9)]
    client = _FakeSlack(
        history=[{"messages": [_m("300.000000", "sync", reply_count=99)]}],
        replies={"300.000000": deep})
    with patch.object(config, "search_reply_per_thread_page_ceiling", 3):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage["reply_pages"] == 3
    assert coverage["stopped_reason"] == "thread_page_ceiling"
    assert coverage["complete"] is False


@pytest.mark.asyncio
async def test_one_absolute_deadline_is_shared_by_every_budget_and_every_await(monkeypatch):
    """§S5's one-deadline rule, over the WHOLE call — not just the two pagers (codex #11).

    The pagers were the easy half. The awaits AFTER the fetch are the ones that can blow the
    outer 20-second tool timeout and take the honest coverage block down with them: the sidecar
    receipt read, username resolution, and permalink enrichment. Each is asserted to run inside
    the same window here, by measuring the timeout each one was actually given.
    """
    built = []
    real = search_mod.FetchBudget

    class _Spy(real):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(search_mod, "FetchBudget", _Spy)

    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal"),
        _self_m("290.000000", "cert renewal, we said"),
    ]}])
    db = _FakeDb(receipts={"290.000000": "finalized"}, epoch="1.000000")
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", db=db)

    assert len(built) == 2
    assert built[0]["deadline_at"] is not None
    assert built[0]["deadline_at"] == built[1]["deadline_at"]
    # Separate page ceilings, one clock.
    assert built[0]["page_ceiling"] == config.search_history_page_ceiling
    assert built[1]["page_ceiling"] == config.search_reply_page_ceiling
    assert payload["count"] == 2, "the receipt read ran, so the own message came back too"


@pytest.fixture
def locked_db(tmp_path, monkeypatch):
    """A REAL database, and a second connection holding an exclusive file lock on it.

    `locking_mode=EXCLUSIVE` plus a write is what blocks a READER under WAL — an ordinary
    `BEGIN EXCLUSIVE` would not, because WAL readers do not wait for a writer. Measured: a
    reader with `busy_timeout=200` fails at 0.201s and one with 1000 at 1.002s, so the busy
    timeout is exactly what decides how long the receipt read can overshoot.
    """
    import sqlite3

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager

    manager = DatabaseManager(platform="slack")
    # The manager's own SYNC handle is closed first, purely so this process can take the
    # exclusive lock at all: in production the lock would belong to another process (a backup, a
    # second bot), which is what this fixture is standing in for. The scan does not touch that
    # handle — `read_channel_sidecars_for_async` opens its own aiosqlite connection.
    manager.conn.close()
    locker = sqlite3.connect(str(manager.db_path))
    locker.execute("PRAGMA journal_mode=WAL")
    locker.execute("PRAGMA locking_mode=EXCLUSIVE")
    locker.execute("BEGIN IMMEDIATE")
    locker.execute("INSERT OR REPLACE INTO bot_meta (key, value) VALUES ('lock-probe', '1')")
    try:
        yield manager
    finally:
        locker.rollback()
        locker.close()


@pytest.mark.asyncio
async def test_a_locked_database_cannot_drag_the_receipt_read_past_the_deadline(locked_db):
    """codex verify P2, against a GENUINELY LOCKED database rather than a sleep.

    `asyncio.wait_for` cancels the read at expiry and then waits for it to unwind, and unwinding
    drains the SQLite work already queued — so with the connection's house busy timeout (5s) the
    receipt read overshoots the scan's whole budget by an order of magnitude, and the outer tool
    timeout fires before the coverage block is ever built. A cancellation-friendly `asyncio.sleep`
    cannot show this; a real lock can, and does.

    The bound therefore lives INSIDE the connection: the accessor's busy timeout is capped by
    what is left of the fetch budget. The outcome is the already-ruled one — no evidence means
    own messages are excluded — and the coverage block says the scan did not finish.
    """
    import time as _time

    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal is with Kestwood"),
        _self_m("290.000000", "cert renewal, we said so"),
    ]}])
    bot = _ScanBot(client)

    started = _time.monotonic()
    with patch.object(config, "search_fetch_total_seconds", 0.3):
        payload = await bot.execute_search_tool(
            _scan_ctx(db=locked_db), {"query": "cert renewal"})
    elapsed = _time.monotonic() - started

    assert elapsed < 2.0, (
        "the receipt read waited on the lock for the connection's house timeout, not for what "
        "was left of the scan's budget")
    assert payload["ok"] is True
    assert [r["ts"] for r in payload["results"]] == ["300.000000"], (
        "no readable evidence means our own message is not searchable")
    coverage = payload["coverage"]
    assert coverage["complete"] is False
    # WHICH of the two guards wins the race is not fixed — the busy timeout and the outer
    # `wait_for` are measured off the same instant — so the assertion is that the NOTE explains
    # whichever one fired, and never blames Slack for a local lock.
    assert coverage["stopped_reason"] in ("deadline", "receipt_read_failed")
    assert coverage["note"] == search_mod.coverage_note(
        complete=False, messages=coverage["messages_scanned"],
        threads=coverage["threads_scanned"], stopped_reason=coverage["stopped_reason"])
    assert "Slack API failure" not in coverage["note"]
    assert ("receipt evidence" in coverage["note"]
            or "time budget" in coverage["note"])


@pytest.mark.asyncio
async def test_the_receipt_read_is_skipped_rather_than_run_with_a_useless_lock_wait():
    """The floor. Below `_RECEIPT_READ_MIN_SECONDS` the read is not attempted at all: a busy
    timeout of a couple of milliseconds fails against any momentarily-locked database, and the
    outcome is the same either way — so the last of the clock is not spent manufacturing a lock
    error. Above the floor, the read runs with the REMAINING budget as its cap, never the
    connection's house default."""
    client = _FakeSlack(history=[{"messages": [_self_m("290.000000", "cert renewal, we said")]}])
    db = _FakeDb(receipts={"290.000000": "finalized"}, epoch="1.000000")

    with patch.object(config, "search_fetch_total_seconds", 0.001):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", db=db)
    assert db.reads == [], "no budget left, so no read was attempted"
    assert payload["coverage"]["stopped_reason"] == "deadline"

    healthy = _FakeDb(receipts={"290.000000": "finalized"}, epoch="1.000000")
    _b2, _c2, ok = await _run_scan(
        _FakeSlack(history=[{"messages": [_self_m("290.000000", "cert renewal, we said")]}]),
        query="cert renewal", db=healthy)
    assert ok["count"] == 1
    assert healthy.busy_timeouts and all(
        0 < ms <= config.search_fetch_total_seconds * 1000 for ms in healthy.busy_timeouts)


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", ["receipts", "usernames", "permalinks"])
async def test_every_post_fetch_await_is_bounded_by_the_scan_deadline(slow):
    """§S5 over the awaits that happen AFTER the pages (codex reviews #1 and #11).

    The pagers were always bounded. These three were not, and they are the ones that can spend
    the 20-second tool timeout and take the honest coverage block down with them — a scan that
    did all its work and then died in `users.info` returns NOTHING, which is the exact failure
    the coverage block exists to replace.

    EACH CASE ASSERTS THE DEGRADATION, not just the clock, because "it returned quickly" alone
    would still pass if the await were dropped entirely. Out of time means: own messages
    excluded (fail closed, as §S3 fails), raw author ids, and a null permalink.
    """
    import time as _time

    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal is with Kestwood"),
        _self_m("290.000000", "cert renewal, we said so"),
    ]}])
    db = _FakeDb(receipts={"290.000000": "finalized"}, epoch="1.000000")

    async def _crawl(*a, **k):
        await asyncio.sleep(5)
        return {}

    bot = _ScanBot(client)
    if slow == "receipts":
        db.read_channel_sidecars_for_async = _crawl
    elif slow == "usernames":
        bot.resolve_usernames = _crawl
    else:
        client.chat_getPermalink = _crawl

    started = _time.monotonic()
    with patch.object(config, "search_fetch_total_seconds", 0.25):
        payload = await bot.execute_search_tool(_scan_ctx(db=db), {"query": "cert renewal"})
    elapsed = _time.monotonic() - started

    assert elapsed < 2.0, f"the {slow} await outlived the scan's own deadline"
    assert payload["ok"] is True
    if slow == "receipts":
        # Fail closed: no evidence means our own message is not searchable, and the block says
        # the scan did not finish.
        assert [r["ts"] for r in payload["results"]] == ["300.000000"]
        assert payload["coverage"]["stopped_reason"] == "deadline"
    elif slow == "usernames":
        # A less readable answer, never a lost one.
        assert [r["author"] for r in payload["results"]] == ["U_HUMAN", "U_BOT"]
    else:
        assert payload["count"] == 2
        assert all(r["permalink"] is None for r in payload["results"])


@pytest.mark.asyncio
async def test_a_spent_deadline_returns_honest_incomplete_coverage():
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}])
    with patch.object(config, "search_fetch_total_seconds", 0.0):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage["stopped_reason"] == "deadline"
    assert coverage["complete"] is False
    assert payload["count"] == 0
    assert "not evidence of absence" in coverage["note"]


@pytest.mark.asyncio
async def test_a_rate_limit_names_the_failure_in_the_coverage_block():
    response = MagicMock()
    response.get = lambda k, d=None: {"error": "ratelimited"}.get(k, d)
    response.headers = {}
    client = _FakeSlack(history=[{"raise": SlackApiError(message="429", response=response)}])
    with patch.object(config, "fetch_retry_attempts", 1):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage["stopped_reason"] == "history_error:ratelimited"
    assert coverage["complete"] is False
    assert "Slack API failure" in coverage["note"]


@pytest.mark.asyncio
async def test_a_cursor_anomaly_returns_incomplete_coverage_not_an_empty_answer():
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")],
                                  "has_more": True}])  # claims more, hands us no cursor
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert payload["ok"] is True
    assert payload["coverage"]["complete"] is False
    assert payload["coverage"]["stopped_reason"].startswith("history_error")


@pytest.mark.asyncio
async def test_one_failing_thread_leaves_the_rest_of_the_scan_intact():
    response = MagicMock()
    response.get = lambda k, d=None: {"error": "thread_not_found"}.get(k, d)
    response.headers = {}
    client = _FakeSlack(
        history=[{"messages": [
            _m("300.000000", "sync", reply_count=1, latest_reply="330.000000"),
            _m("200.000000", "other", reply_count=1, latest_reply="220.000000"),
        ]}],
        replies={"300.000000": [{"raise": SlackApiError(message="gone", response=response)}],
                 "200.000000": [{"messages": [_m("220.000000", "cert renewal with Kestwood")]}]})
    with patch.object(config, "search_reply_fetch_concurrency", 1):
        _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [r["ts"] for r in payload["results"]] == ["220.000000"]
    assert payload["coverage"]["stopped_reason"] == "reply_error:thread_not_found"
    assert payload["coverage"]["complete"] is False


@pytest.mark.asyncio
async def test_a_complete_scan_says_so_and_counts_exactly():
    client = _FakeSlack(
        history=[{"messages": [_m("300.000000", "sync", reply_count=1, latest_reply="310.000000"),
                               _m("290.000000", "cert renewal here")]}],
        replies={"300.000000": [{"messages": [
            _m("300.000000", "sync", reply_count=1, latest_reply="310.000000"),
            _m("310.000000", "a reply")]}]})
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    coverage = payload["coverage"]
    assert coverage == {
        "complete": True, "messages_scanned": 3, "threads_scanned": 1,
        "history_pages": 1, "reply_pages": 1, "stopped_reason": None,
        "note": coverage["note"],
    }
    assert "Searched 3 messages across 1 thread" in coverage["note"]
    # §S6 addendum / codex review #5: a COMPLETE scan claims completeness over the messages it is
    # PERMITTED to return, never over the room. Our own unfinalized posts and anyone writing from
    # another workspace are withheld by design, so "that was not said here" would be false in
    # exactly the rooms — Slack Connect ones — where a wrong answer costs the most.
    note = coverage["note"]
    assert "not the same as never said in this room" in note
    assert "another workspace" in note
    assert "was not said here" not in note


# --- the result contract (§S6) --------------------------------------------------------

@pytest.mark.asyncio
async def test_the_result_entry_shape_and_author_rendering():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal is with Kestwood", thread_ts="290.000000"),
    ]}])
    bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert payload["results"] == [{
        "channel": CH,
        "ts": "300.000000",
        "author": "dana",
        "text": "cert renewal is with Kestwood",
        "permalink": f"https://slack/{CH}/p300.000000",
        "thread_ts": "290.000000",
    }]
    # Resolved read-only, in one batch, over the RETAINED set only.
    bot.resolve_usernames.assert_awaited_once()
    assert bot.resolve_usernames.await_args.args[0] == ["U_HUMAN"]


@pytest.mark.asyncio
async def test_a_failed_permalink_never_invalidates_a_match():
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}])
    client.permalink_fails = True
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert payload["count"] == 1
    assert payload["results"][0]["permalink"] is None


@pytest.mark.asyncio
async def test_delivery_allowed_is_invoked_for_every_candidate():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal one"), _m("290.000000", "cert renewal two")]}])
    bot = _ScanBot(client)
    ctx = _scan_ctx()
    seen = []
    original = bot._delivery_allowed

    async def _spy(target, context, source_team_id=None):
        seen.append((target, source_team_id))
        return await original(target, context, source_team_id)

    bot._delivery_allowed = _spy
    payload = await bot.execute_search_tool(ctx, {"query": "cert renewal"})
    assert payload["count"] == 2
    # Once for the pre-scan authorization, once per retained candidate — each carrying the
    # stamped team identity so a foreign one cannot ride the current-channel exemption.
    assert seen.count((CH, "T_TEST")) == 2
    assert (CH, None) in seen


@pytest.mark.asyncio
async def test_a_candidate_contradicting_itself_about_its_workspace_is_dropped():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal one", team_id="T_OTHER"),   # two distinct team ids
        _m("290.000000", "cert renewal two"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [r["ts"] for r in payload["results"]] == ["290.000000"]


@pytest.mark.asyncio
async def test_a_foreign_workspace_candidate_is_dropped():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal one", team="T_ELSEWHERE"),
        _m("290.000000", "cert renewal two"),
    ]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert [r["ts"] for r in payload["results"]] == ["290.000000"]


@pytest.mark.asyncio
async def test_the_channel_is_always_the_trusted_current_channel():
    client = _FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal", channel="C_LIAR")]}])
    _bot, _ctx_, payload = await _run_scan(
        client, query="cert renewal", args={"channel_id": "C_ELSEWHERE"})
    assert payload["results"][0]["channel"] == CH


@pytest.mark.asyncio
async def test_a_retained_root_is_staged_and_survives_serialization():
    """W3/§2g: search-to-action needs the `thread_ts` PAIR to reach the model."""
    client = _FakeSlack(
        history=[{"messages": [_m("300.000000", "sync", reply_count=1,
                                  latest_reply="310.000000")]}],
        replies={"300.000000": [{"messages": [
            _m("300.000000", "sync", reply_count=1, latest_reply="310.000000"),
            _m("310.000000", "cert renewal with Kestwood", thread_ts="300.000000")]}]})
    _bot, ctx, payload = await _run_scan(client, query="cert renewal")
    assert payload["results"][0]["thread_ts"] == "300.000000"
    staged = ctx.tool_flight.staged_roots
    assert [(s.channel_id, s.root_ts, s.source, s.field) for s in staged] == [
        (CH, "300.000000", "search_slack", "thread_ts")]
    assert '"thread_ts": "300.000000"' in serialize_tool_result(payload)


@pytest.mark.asyncio
async def test_the_scan_never_writes_anything():
    """The database is READ for receipts and never anything else; `_FakeDb` fails any other
    call and `_FakeSlack` has no write methods at all."""
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal"),
                                               _self_m("290.000000", "cert renewal reply")]}])
    db = _FakeDb(receipts={"290.000000": "finalized"}, epoch="1.000000")
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", db=db)
    assert payload["ok"] is True
    assert len(db.reads) == 1


# --- surfaces (§S1, §S8, §S9) ---------------------------------------------------------

def test_the_backend_selector_splits_on_the_surface_alone():
    assert search_backend_for(_dm_ctx()) is SearchBackend.ASSISTANT_CONTEXT
    assert search_backend_for(_ctx(is_dm=False)) is SearchBackend.IN_CHANNEL_SCAN
    # An MPIM is channel-shaped: `is_dm` is stamped from is_dm_conversation, which says so.
    from slack_client.utilities import is_dm_conversation
    assert is_dm_conversation("G0123MPIM", "mpim") is False
    assert is_dm_conversation("D0123", "im") is True
    assert search_backend_for(_ctx(channel_id="G0123MPIM", is_dm=False)) is (
        SearchBackend.IN_CHANNEL_SCAN)
    # The reserved shape-2 seam exists and is not selectable.
    assert SearchBackend.SERVICE_INDEX.value == "service_index"
    assert search_backend_for(_ctx()) is not SearchBackend.SERVICE_INDEX


@pytest.mark.asyncio
async def test_an_ambient_channel_turn_searches_without_an_action_token():
    """The first live defect: an unmentioned turn carries no action_token and could not search
    at all. The channel backend never asks for one."""
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal with Kestwood")]}])
    ctx = _scan_ctx()
    assert ctx.action_token is None
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", ctx=ctx)
    assert payload["ok"] is True and payload["count"] == 1


@pytest.mark.asyncio
async def test_the_channel_path_never_calls_the_assistant_index():
    """`_FakeSlack.api_call` raises; reaching it at all fails the test."""
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal")
    assert payload["ok"] is True
    assert payload["scope"] == "channel"


@pytest.mark.asyncio
async def test_an_explicit_non_channel_scope_is_refused():
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}])
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal",
                                           args={"scope": "workspace"})
    assert payload["ok"] is False and payload["error"] == "bad_arguments"
    assert client.history_calls == []
    # An explicit `channel` is simply what it already does.
    _b2, _c2, ok = await _run_scan(_FakeSlack(history=[{"messages": [
        _m("300.000000", "cert renewal")]}]), query="cert renewal", args={"scope": "channel"})
    assert ok["ok"] is True


@pytest.mark.asyncio
async def test_the_channel_read_gate_still_refuses_an_unauthorized_channel():
    client = _FakeSlack(history=[{"messages": [_m("300.000000", "cert renewal")]}],
                        info={"is_channel": True, "is_private": True, "is_member": False})
    ctx = _scan_ctx()
    _bot, _ctx_, payload = await _run_scan(client, query="cert renewal", ctx=ctx)
    assert payload["ok"] is False and payload["error"] == "not_accessible"
    assert client.history_calls == []


@pytest.mark.asyncio
async def test_the_dm_surface_still_runs_the_assistant_backend():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([
        {"channel_id": "C09", "message_ts": "100.1", "author_user_id": "U9",
         "content": "we decided fridays", "permalink": "https://x/p1"}]))
    out = await bot.execute_search_tool(_dm_ctx(), {"query": "demo day"})
    assert out["results"][0] == {"channel": "C09", "ts": "100.1", "author": "U9",
                                 "text": "we decided fridays", "permalink": "https://x/p1"}
    assert "coverage" not in out
    assert bot.app.client.api_call.await_args.args[0] == "assistant.search.context"


# --- config (§S5) ---------------------------------------------------------------------

def test_the_search_budgets_default_to_their_named_tuned_values(mock_env):
    cfg = BotConfig()
    assert cfg.search_history_page_ceiling == cfg.history_page_ceiling == 50
    assert cfg.search_reply_page_ceiling == 50
    assert cfg.search_reply_per_thread_page_ceiling == 10
    assert cfg.search_history_page_size == cfg.history_page_size == 200
    assert cfg.search_reply_fetch_concurrency == cfg.reply_fetch_concurrency == 4
    assert cfg.search_fetch_total_seconds == 8
    assert cfg.search_tool_timeout_seconds == cfg.tool_call_timeout == 20


def test_an_invalid_budget_relationship_is_rejected_at_boot(mock_env, monkeypatch):
    monkeypatch.setenv("SEARCH_FETCH_TOTAL_SECONDS", "30")
    monkeypatch.setenv("SEARCH_TOOL_TIMEOUT_SECONDS", "20")
    with pytest.raises(ValueError, match="SEARCH_FETCH_TOTAL_SECONDS"):
        BotConfig()
    monkeypatch.setenv("SEARCH_FETCH_TOTAL_SECONDS", "0")
    with pytest.raises(ValueError, match="SEARCH_FETCH_TOTAL_SECONDS"):
        BotConfig()
    # Equal is rejected too: the outer timeout must never be the thing that fires.
    monkeypatch.setenv("SEARCH_FETCH_TOTAL_SECONDS", "20")
    with pytest.raises(ValueError, match="SEARCH_FETCH_TOTAL_SECONDS"):
        BotConfig()


def test_the_search_tool_is_registered_with_its_own_timeout():
    import inspect
    from slack_client import base as base_mod
    src = inspect.getsource(base_mod)
    assert "timeout=config.search_tool_timeout_seconds" in src
