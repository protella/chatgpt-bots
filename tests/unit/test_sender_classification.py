"""Phase 1 — sender classification + multi-bot history role mapping.

Covers the new logic added for human/self/other_bot detection and the metadata that drives
the thread-history role fix (own bot -> assistant; humans AND other bots -> user). These are
standalone targeted tests; the legacy suite is not exercised here.

Also home to the bot OBJECT id -> bot USER id resolver and the outbound `<@B…>` guard: both are
answers to the same question this file is about — which id names the bot that posted, and which
of them Slack will actually render.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from slack_client.markdown_converter import MarkdownConverter
from slack_client.utilities import BOT_USER_ID_CACHE_MAX, SlackUtilitiesMixin
from slack_client.messaging import SlackMessagingMixin, rewrite_bot_object_mentions
from slack_client.formatting.text import SlackFormattingMixin


# --- Lightweight carriers binding the real mixin methods (no full SlackBot needed) ---

class _Ident:
    """Minimal object exposing the real classify_sender / is_own_message."""
    is_own_message = SlackUtilitiesMixin.is_own_message
    classify_sender = SlackUtilitiesMixin.classify_sender

    def __init__(self, bot_id=None, bot_user_id=None, app_id=None):
        self.bot_id = bot_id
        self.bot_user_id = bot_user_id
        self.app_id = app_id


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Minimal harness to exercise the real get_thread_history against a mocked client."""
    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_warning(self, *a, **k): pass


SELF_BOT_ID = "B07SELF"
SELF_USER_ID = "U07SELF"


# --- classify_sender / is_own_message ---

def test_human_message_is_human():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"user": "U07HUMAN", "text": "hi"}
    assert b.classify_sender(msg) == "human"
    assert b.is_own_message(msg) is False


def test_own_message_by_bot_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "mine"}
    assert b.is_own_message(msg) is True
    assert b.classify_sender(msg) == "self"


def test_own_message_by_user_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    assert b.classify_sender({"user": SELF_USER_ID, "text": "mine"}) == "self"


def test_own_message_by_app_id():
    b = _Ident(app_id="A07SELF")
    assert b.is_own_message({"app_id": "A07SELF"}) is True
    assert b.is_own_message({"api_app_id": "A07SELF"}) is True


def test_other_bot_by_bot_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"bot_id": "B07OTHER", "username": "Claude", "user": "U07X"}
    assert b.classify_sender(msg) == "other_bot"
    assert b.is_own_message(msg) is False


def test_other_bot_by_app_id_only():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    # app-posted message without subtype=="bot_message" must still be detected
    assert b.classify_sender({"app_id": "A07OTHER", "text": "x"}) == "other_bot"


def test_non_dict_defaults_human():
    b = _Ident(bot_id=SELF_BOT_ID)
    assert b.classify_sender(None) == "human"
    assert b.is_own_message(None) is False


def test_dev_allowlisted_bot_id_classifies_human(monkeypatch):
    # User-token (xoxp) harness posts carry the app's bot_id even though a human wrote them;
    # the DEV_TREAT_BOT_IDS_AS_HUMAN allowlist restores the truth. Empty in prod.
    from config import config as cfg
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"bot_id": "B07HARNESS", "app_id": "A07HARNESS", "user": "U07PETER", "text": "hi"}
    assert b.classify_sender(msg) == "other_bot"
    monkeypatch.setattr(cfg, "dev_treat_bot_ids_as_human", ["B07HARNESS"])
    assert b.classify_sender(msg) == "human"
    # Self-detection wins over the allowlist, and other bots stay bots.
    assert b.classify_sender({"bot_id": SELF_BOT_ID}) == "self"
    assert b.classify_sender({"bot_id": "B07OTHER"}) == "other_bot"
    # app_id-only messages (no bot_id) never match the bot_id allowlist.
    assert b.classify_sender({"app_id": "A07OTHER"}) == "other_bot"


# --- get_thread_history: metadata that feeds the role mapping ---

@pytest.mark.asyncio
async def test_get_thread_history_sets_sender_metadata():
    b = _Bot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "<@U07SELF> hello"},          # human
        {"ts": "2", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "my reply"},  # self
        {"ts": "3", "bot_id": "B07OTHER", "username": "Claude", "text": "from claude"},  # other bot
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )

    result = await b.get_thread_history("C1", "1")
    by_ts = {m.metadata["ts"]: m for m in result}

    # human
    assert by_ts["1"].metadata["sender_type"] == "human"
    assert by_ts["1"].metadata["is_bot"] is False
    assert by_ts["1"].metadata["bot_name"] is None
    assert by_ts["1"].text == "hello"  # mention stripped for humans

    # self
    assert by_ts["2"].metadata["sender_type"] == "self"
    assert by_ts["2"].metadata["is_bot"] is True

    # other bot — carries its display name for user-role prefixing
    assert by_ts["3"].metadata["sender_type"] == "other_bot"
    assert by_ts["3"].metadata["is_bot"] is True
    assert by_ts["3"].metadata["bot_name"] == "Claude"


@pytest.mark.asyncio
@pytest.mark.parametrize("footer_blocks", [
    # Current compact footer: single actions row, model name inside the button
    [
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "⚙️ gpt-5.5"},
             "action_id": "open_channel_settings"},
        ]},
    ],
    # Legacy two-row footer (context line + Configure button) — still present in old
    # channel history, must stay skipped
    [
        {"type": "context", "elements": [{"type": "mrkdwn", "text": ":robot_face: gpt-5.5"}]},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "⚙️ Configure"},
             "action_id": "open_channel_settings"},
        ]},
    ],
])
async def test_get_thread_history_skips_response_footer(footer_blocks):
    """The Configure footer (a separate own-bot message) must not enter rebuilt history,
    or every channel exchange would gain a bogus assistant turn saying just the model name."""
    b = _Bot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "what model are you?"},
        {"ts": "2", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "I'm running gpt-5.5."},
        {"ts": "3", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "gpt-5.5",
         "blocks": footer_blocks},  # the footer message
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )

    result = await b.get_thread_history("C1", "1")
    assert [m.metadata["ts"] for m in result] == ["1", "2"]  # footer (ts=3) skipped


# --- bot OBJECT id -> bot USER id, cached ---

class _Resolver(SlackUtilitiesMixin):
    """The real resolver over a mocked bots.info."""

    def __init__(self, response=None, error=None):
        super().__init__()
        self.app = MagicMock()
        if error is not None:
            self.app.client.bots_info = AsyncMock(side_effect=error)
        else:
            self.app.client.bots_info = AsyncMock(return_value=response)

    def log_debug(self, *a, **k): pass
    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
async def test_a_bot_id_is_resolved_once_and_then_served_from_memory():
    r = _Resolver({"ok": True, "bot": {"id": "B07PEER", "user_id": "U07PEER"}})
    assert await r.resolve_bot_user_id("B07PEER") == "U07PEER"
    assert await r.resolve_bot_user_id("B07PEER") == "U07PEER"
    await r.prime_bot_user_ids([{"bot_id": "B07PEER", "text": "hi"}])
    assert r.app.client.bots_info.await_count == 1
    assert r.bot_user_id_for("B07PEER") == "U07PEER"


@pytest.mark.asyncio
async def test_a_bot_slack_cannot_name_is_negatively_cached():
    r = _Resolver({"ok": True, "bot": {"id": "B07PEER"}})   # no user_id in the answer
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert r.app.client.bots_info.await_count == 1
    assert r.bot_user_id_for("B07PEER") is None


@pytest.mark.asyncio
async def test_a_non_user_shaped_answer_is_kept_as_a_miss():
    """A B-shaped "user id" is not a mention target — caching it would feed the outbound guard
    the very id it exists to keep out of the channel."""
    r = _Resolver({"ok": True, "bot": {"user_id": "B07SOMETHINGELSE"}})
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert r.bot_user_id_for("B07PEER") is None
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert r.app.client.bots_info.await_count == 1     # the miss itself is cached


@pytest.mark.asyncio
async def test_a_transient_api_error_is_swallowed_and_never_cached():
    """A timeout is not an answer. Caching it would blind the process to a resolvable bot until
    restart, so the id stays unknown and a later batch may try again — bounded by the batch
    budget, never by a permanent wrong answer."""
    r = _Resolver(error=RuntimeError("bots.info exploded"))
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert "B07PEER" not in r._bot_user_ids()
    assert await r.resolve_bot_user_id("B07PEER") is None
    assert r.app.client.bots_info.await_count == 2


@pytest.mark.asyncio
async def test_one_bot_id_resolved_concurrently_spends_one_call():
    """The shared window and the origin thread fetch at the same time, and the search fan-out
    likewise — without single flight they each pay for the same answer."""
    r = _Resolver({"ok": True, "bot": {"user_id": "U07PEER"}})
    slow = r.app.client.bots_info

    async def _answer(**kwargs):
        await asyncio.sleep(0.01)                       # long enough for the others to queue
        return {"ok": True, "bot": {"user_id": "U07PEER"}}

    slow.side_effect = _answer
    results = await asyncio.gather(*(r.resolve_bot_user_id("B07PEER") for _ in range(4)))
    assert results == ["U07PEER"] * 4
    assert slow.await_count == 1
    assert r._bot_id_flights() == {}                    # the flight is cleared when it lands


@pytest.mark.asyncio
async def test_waiters_share_a_failing_flight_and_a_later_call_still_retries():
    """A failure is not cached, so waiters that re-checked only the cache would each go on to
    make their own serialized call after one timeout. They share the flight's outcome instead —
    and a call that starts after it finished is free to try again."""
    r = _Resolver(error=RuntimeError("bots.info timed out"))

    async def _fail(**kwargs):
        await asyncio.sleep(0.01)                       # long enough for the others to queue
        raise RuntimeError("bots.info timed out")

    r.app.client.bots_info.side_effect = _fail
    results = await asyncio.gather(*(r.resolve_bot_user_id("B07PEER") for _ in range(4)))
    assert results == [None] * 4
    assert r.app.client.bots_info.await_count == 1
    assert "B07PEER" not in r._bot_user_ids()

    r.app.client.bots_info.side_effect = None
    r.app.client.bots_info.return_value = {"ok": True, "bot": {"user_id": "U07PEER"}}
    assert await r.resolve_bot_user_id("B07PEER") == "U07PEER"
    assert r.app.client.bots_info.await_count == 2


@pytest.mark.asyncio
async def test_two_concurrent_batches_each_keep_their_own_pinned_entries():
    """A batch spends most of its time WAITING on somebody else's flight. Its pins have to hold
    across that wait, or the resolving batch's cache write evicts what the waiting one knows."""
    r = _Resolver({"ok": True, "bot": {"user_id": "U07SHARED"}})
    cache = r._bot_user_ids()
    cache["B07A_KEEP"] = "U07A"                         # oldest — the first eviction victims
    cache["B07B_KEEP"] = "U07B"
    for filler in range(BOT_USER_ID_CACHE_MAX - 2):
        cache[f"B07FILL{filler}"] = f"U07FILL{filler}"

    async def _answer(**kwargs):
        await asyncio.sleep(0.01)                       # long enough for the batches to overlap
        return {"ok": True, "bot": {"user_id": "U07SHARED"}}

    r.app.client.bots_info.side_effect = _answer
    await asyncio.gather(
        r.prime_bot_user_ids([{"bot_id": "B07A_KEEP"}, {"bot_id": "B07SHARED"}]),
        r.prime_bot_user_ids([{"bot_id": "B07B_KEEP"}, {"bot_id": "B07SHARED"}]))
    assert r.app.client.bots_info.await_count == 1      # one flight, both batches
    assert r.bot_user_id_for("B07A_KEEP") == "U07A"
    assert r.bot_user_id_for("B07B_KEEP") == "U07B"
    assert r.bot_user_id_for("B07SHARED") == "U07SHARED"


@pytest.mark.asyncio
async def test_an_all_pinned_overflow_drains_back_to_the_bound():
    """Exceeding the cap when everything is pinned is a plateau, not a new ceiling."""
    r = _Resolver({"ok": True, "bot": {"user_id": "U07NEW"}})
    cache = r._bot_user_ids()
    for filler in range(BOT_USER_ID_CACHE_MAX):
        cache[f"B07FILL{filler}"] = f"U07FILL{filler}"
    everything = [{"bot_id": key} for key in list(cache)] + [{"bot_id": "B07NEW1"}]
    await r.prime_bot_user_ids(everything)              # nothing evictable → one over the bound
    assert len(cache) == BOT_USER_ID_CACHE_MAX + 1
    await r.prime_bot_user_ids([{"bot_id": "B07NEW2"}])  # pins released → drains
    assert len(cache) == BOT_USER_ID_CACHE_MAX
    assert r.bot_user_id_for("B07NEW1") == "U07NEW" and r.bot_user_id_for("B07NEW2") == "U07NEW"


@pytest.mark.asyncio
async def test_a_full_cache_never_evicts_the_batch_it_is_priming():
    """The batch's own keys are pinned: an entry it already counted as known must still be there
    when the caller normalizes a moment later."""
    r = _Resolver({"ok": True, "bot": {"user_id": "U07NEW"}})
    cache = r._bot_user_ids()
    cache["B07OLD"] = "U07OLD"                          # the batch's already-known bot
    for filler in range(BOT_USER_ID_CACHE_MAX):         # ... and a cache with no room left
        cache[f"B07FILL{filler}"] = f"U07FILL{filler}"
    await r.prime_bot_user_ids([{"bot_id": "B07OLD"}, {"bot_id": "B07NEW"}])
    assert r.bot_user_id_for("B07OLD") == "U07OLD"
    assert r.bot_user_id_for("B07NEW") == "U07NEW"


@pytest.mark.asyncio
async def test_a_later_failure_never_overwrites_a_resolved_user_id():
    r = _Resolver({"ok": True, "bot": {"user_id": "U07PEER"}})
    assert await r.resolve_bot_user_id("B07PEER") == "U07PEER"
    r.app.client.bots_info.side_effect = RuntimeError("rate limited")
    await r.prime_bot_user_ids([{"bot_id": "B07PEER"}])
    assert r.bot_user_id_for("B07PEER") == "U07PEER"


@pytest.mark.asyncio
async def test_priming_only_spends_calls_on_user_less_bot_payloads():
    r = _Resolver({"ok": True, "bot": {"user_id": "U07PEER"}})
    await r.prime_bot_user_ids([
        {"user": "U07HUMAN", "text": "a human"},                 # names its actor already
        {"user": "U07OTHER", "bot_id": "B07HASUSER"},            # a bot that names its user
        {"bot_id": "B07AGENT", "username": "Peer", "text": "x"},  # agent mode: bot_id only
        {"bot_id": "B07AGENT", "text": "same bot again"},        # deduped
        "not a payload",
    ])
    assert [c.kwargs["bot"] for c in r.app.client.bots_info.await_args_list] == ["B07AGENT"]
    assert r.bot_user_id_for("B07AGENT") == "U07PEER"
    assert r.bot_user_id_for("B07HASUSER") is None   # never looked up, never cached


# --- outbound: `<@B…>` must never reach Slack ---

class _Poster(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """The real send path over a mocked Slack web client."""

    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.bot_id = SELF_BOT_ID
        self.bot_user_id = SELF_USER_ID
        self.app_id = None
        self.self_team_id = "T07"
        self.app = MagicMock()
        self.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1.1"})
        self.app.client.chat_update = AsyncMock(return_value={"ok": True})
        self.markdown_converter = MarkdownConverter(platform="slack")
        self._bot_user_id_cache = {"B07KNOWN": "U07KNOWN"}

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info


def test_a_resolvable_bot_mention_is_rewritten_to_the_user_id():
    b = _Poster()
    assert rewrite_bot_object_mentions(b, "ask <@B07KNOWN> about it") == \
        "ask <@U07KNOWN> about it"
    assert rewrite_bot_object_mentions(b, "ask <@B07KNOWN|peer> about it") == \
        "ask <@U07KNOWN> about it"


def test_an_unresolvable_bot_mention_loses_the_mention_syntax():
    assert rewrite_bot_object_mentions(_Poster(), "ask <@B07STRANGER> about it") == \
        "ask @bot about it"


def test_text_without_a_bot_mention_is_untouched():
    b = _Poster()
    for text in ("", "plain words", "hi <@U07HUMAN> and <@W07GRID>", "a B07KNOWN id in prose"):
        assert rewrite_bot_object_mentions(b, text) == text


def test_quoted_code_and_links_are_left_exactly_as_written():
    """Code is what the answer is SHOWING the room — a log line, a payload, the id under
    discussion. Rewriting inside it changes what the bot claims it saw."""
    b = _Poster()
    fenced = "look:\n```\nsender: <@B07KNOWN>\n```\nand <@B07KNOWN> replied"
    assert rewrite_bot_object_mentions(b, fenced) == \
        "look:\n```\nsender: <@B07KNOWN>\n```\nand <@U07KNOWN> replied"
    inline = "the raw field was `<@B07STRANGER>` in the payload"
    assert rewrite_bot_object_mentions(b, inline) == inline
    # A Slack link form can never match: the pattern anchors on `<@`, which a link never has.
    linked = "see <https://example.com/bots/B07KNOWN|the bot page>"
    assert rewrite_bot_object_mentions(b, linked) == linked


async def _stream(cumulative_pieces):
    """Drive a real NativeStreamSession over the pieces (given as DELTAS) and return exactly what
    reached Slack."""
    b = _Poster()
    web = b.app.client
    web.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
    web.chat_appendStream = AsyncMock(return_value={"ok": True})
    web.chat_stopStream = AsyncMock(return_value={"ok": True})
    session = b.begin_native_stream("C1", "1.0", user_id="U07HUMAN")
    assert await session.start("") is True
    cumulative = ""
    for piece in cumulative_pieces:
        cumulative += piece
        await session.update(cumulative)
    await session.finish(cumulative)
    return "".join(
        call.kwargs.get("markdown_text") or ""
        for call in web.chat_appendStream.await_args_list + web.chat_stopStream.await_args_list)


@pytest.mark.asyncio
async def test_a_mention_split_across_native_stream_deltas_is_still_caught():
    """Native streaming sends DELTAS: `<@B07KNOWN>` can arrive as two halves, and rewriting each
    delta on its own would see neither. The fragment is held until it resolves."""
    b = _Poster()
    web = b.app.client
    web.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
    web.chat_appendStream = AsyncMock(return_value={"ok": True})
    web.chat_stopStream = AsyncMock(return_value={"ok": True})
    session = b.begin_native_stream("C1", "1.0", user_id="U07HUMAN")
    assert await session.start("") is True
    await session.update("ask <@B07")
    await session.update("ask <@B07KNOWN> now")
    await session.finish("ask <@B07KNOWN> now, and <@B07STRANGER>")
    streamed = "".join(
        call.kwargs.get("markdown_text") or ""
        for call in web.chat_appendStream.await_args_list + web.chat_stopStream.await_args_list)
    assert streamed == "ask <@U07KNOWN> now, and @bot"


@pytest.mark.asyncio
async def test_a_fence_opened_in_one_delta_still_protects_the_next():
    """Code-span state is CUMULATIVE across deltas: a per-chunk scan would rewrite inside a block
    it never saw open."""
    streamed = await _stream(["```\n", "sender: <@B07KNOWN>\n", "```\n", "and <@B07KNOWN> spoke"])
    assert streamed == "```\nsender: <@B07KNOWN>\n```\nand <@U07KNOWN> spoke"


@pytest.mark.asyncio
async def test_a_fence_delimiter_split_across_deltas_is_still_a_fence():
    """The delimiter itself arrives in pieces. Held until it is whole, in both directions:
    the fence that OPENS the block and the one that closes it."""
    streamed = await _stream(["`", "``\n", "sender: <@B07KNOWN>\n", "`", "``\n",
                              "and <@B07KNOWN> spoke"])
    assert streamed == "```\nsender: <@B07KNOWN>\n```\nand <@U07KNOWN> spoke"


@pytest.mark.asyncio
async def test_an_overlong_literal_inside_a_fence_is_left_exactly_as_written():
    """Classification runs BEFORE neutralization: a quoted `<@B…` is content the answer is
    showing the room, and breaking its bracket would edit what it says it saw."""
    label = "x" * 140
    streamed = await _stream(["```\n", f"raw: <@B07KNOWN|{label}", "\n```"])
    assert streamed == f"```\nraw: <@B07KNOWN|{label}\n```"


@pytest.mark.asyncio
async def test_an_overlong_unfinished_mention_is_neutralized_before_release():
    """Past the hold bound the fragment is released — but never as written: a `<@B…` that goes
    out intact becomes a real mention the moment a later delta brings the `>`."""
    label = "x" * 140
    streamed = await _stream([f"see <@B07KNOWN|{label}", ">"])
    assert "<@B" not in streamed and "<@" not in streamed
    assert streamed == f"see @B07KNOWN|{label}>"


@pytest.mark.asyncio
async def test_the_send_path_posts_no_bot_object_mention():
    b = _Poster()
    await b.send_message("C1", "1.0", "<@B07KNOWN> and <@B07STRANGER> both spoke",
                         receipt_class="assistant_reply")
    posted = b.app.client.chat_postMessage.await_args.kwargs["text"]
    assert "<@U07KNOWN>" in posted and "@bot" in posted and "<@B" not in posted


@pytest.mark.asyncio
async def test_the_generic_update_path_posts_no_bot_object_mention():
    """`update_message` — the error/timeout notice writer. The MODEL-driven edit goes through
    execute_edit_own_message, covered against the real transaction in test_edit_own_message.py."""
    b = _Poster()
    assert await b.update_message("C1", "1.1", "<@B07STRANGER> spoke") is True
    assert b.app.client.chat_update.await_args.kwargs["text"] == "@bot spoke"


# --- role mapping contract (as implemented in thread_management rebuild) ---

def _role_for(sender_type):
    """Mirror of the rule in ThreadManagementMixin._get_or_rebuild_thread_state:
    only our own messages are assistant turns; everyone else is a user turn."""
    return "assistant" if sender_type == "self" else "user"


@pytest.mark.parametrize("sender_type,expected", [
    ("self", "assistant"),
    ("other_bot", "user"),
    ("human", "user"),
])
def test_role_mapping_contract(sender_type, expected):
    assert _role_for(sender_type) == expected
