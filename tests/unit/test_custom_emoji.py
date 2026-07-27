"""AREA C — workspace custom emojis available to the model.

Covers the whole C surface:
- WorkspaceEmojiCache.refresh(): parse emoji.list KEYS (incl. alias entries), valid_emoji_name
  filter, sort + dedupe; retain last-good on error; empty only when never fetched.
- get_custom_emoji_names(): SYNC + stale-ok — returns last-good immediately, schedules exactly ONE
  background refresh when expired, never raises (incl. no running loop).
- react_to_message factory: REACTION_EMOJIS empty → a POINTER to search_workspace_emoji (not a
  name list), appearing only when the workspace has customs and tracking the live cache;
  REACTION_EMOJIS set → enum allowlist, customs suppressed, search tool not registered at all.
- WorkspaceEmojiCache.search(): lexical ranking over the full catalog, single-char tokens dropped.
- reactions.add uses the bare name; unknown fails soft. Config defaults + .env.example docs.

THE GATE IS NO LONGER A CONSUMER OF ANY OF THIS. The emoji shortlist and the `_coerce_emoji`
repair step existed because the rich gate PICKED an emoji and placed it itself, before the
responder ran. It picks nothing now — it returns one bit — so the palette has no gate to feed and
a coerced emoji has no placer. Those tests are inverted into tripwires below; the CACHE keeps its
own unit coverage, because the RESPONDER still reacts and `search_workspace_emoji` still answers
from the same catalog by NAME.

The observed-usage TALLY that ranked the shortlist (and its DB persistence) is gone outright —
ranking only ever served the gate prompt. See the tripwire at the foot of this file.

(Both name lists used to be an ALPHABETICAL prefix of ~1,400 names, so the model only ever saw
"000, 1password_icon, 2605732e-82a0-46b9-b1e0-ecc4f250eb35, 4cats_q, alabama…" — prompt noise it
could not use. The responder searches on demand instead, which is the surviving half of that fix.)

All in-memory; no network/DB.
"""
from __future__ import annotations

import asyncio
import inspect
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config, valid_emoji_name
from message_processor.participation import ParticipationEngine
from slack_client.channel_pulse import ChannelPulse
from slack_client.messaging import SlackMessagingMixin, WorkspaceEmojiCache
from tool_registry import ToolContext


# =============================================================== WorkspaceEmojiCache

def _emoji_client(emoji_return):
    api = SimpleNamespace(emoji_list=AsyncMock(return_value=emoji_return))
    client = SimpleNamespace(app=SimpleNamespace(client=api), log_debug=lambda *a, **k: None)
    return client, api


@pytest.mark.asyncio
async def test_refresh_parses_alias_keys_filters_and_dedupes():
    resp = {"ok": True, "emoji": {
        "party_parrot": "https://x/pp.gif",   # real custom
        "shipit": "alias:rocket",             # alias — the KEY is the name reactions.add accepts
        "bad name!": "https://x/x.png",        # invalid shorthand → filtered
        "party": "https://x/p.png",
        ":party:": "https://x/p2.png",         # normalizes to 'party' → deduped with the above
    }}
    client, _ = _emoji_client(resp)
    cache = WorkspaceEmojiCache(client)
    names = await cache.refresh()
    assert names == ("party", "party_parrot", "shipit")   # sorted, deduped, filtered
    assert cache.get_custom_emoji_names() == ("party", "party_parrot", "shipit")
    assert cache._expiry > 0.0                            # TTL set


@pytest.mark.asyncio
async def test_refresh_retains_last_good_on_error():
    client, api = _emoji_client({"ok": True, "emoji": {"aa": "u", "bb": "u"}})
    cache = WorkspaceEmojiCache(client)
    assert await cache.refresh() == ("aa", "bb")
    # emoji:read missing / any API error → the last-good tuple is kept, never wiped to empty.
    api.emoji_list = AsyncMock(side_effect=RuntimeError("emoji:read missing"))
    assert await cache.refresh() == ("aa", "bb")
    assert cache._expiry > 0.0                            # TTL still reset (back off, don't hammer)


@pytest.mark.asyncio
async def test_refresh_empty_only_when_never_fetched():
    client, api = _emoji_client(None)
    api.emoji_list = AsyncMock(side_effect=RuntimeError("boom"))
    cache = WorkspaceEmojiCache(client)
    assert await cache.refresh() == ()                   # never succeeded → empty tuple


@pytest.mark.asyncio
async def test_getter_returns_last_good_and_schedules_one_refresh(monkeypatch):
    monkeypatch.setattr(config, "workspace_emoji_ttl_seconds", 3600, raising=False)
    client, api = _emoji_client({"ok": True, "emoji": {"aa": "u"}})
    cache = WorkspaceEmojiCache(client)
    await cache.refresh()
    assert cache.get_custom_emoji_names() == ("aa",)     # fresh → no scheduling
    assert api.emoji_list.await_count == 1

    # Expire it and change what the server would return.
    cache._expiry = 0.0
    api.emoji_list = AsyncMock(return_value={"ok": True, "emoji": {"bb": "u"}})
    # A sync burst returns the LAST-GOOD tuple immediately and schedules exactly ONE refresh.
    got = [cache.get_custom_emoji_names() for _ in range(3)]
    assert all(g == ("aa",) for g in got)                # stale-ok, non-blocking
    assert cache._refreshing is True                     # guard set synchronously on first call
    # Let the scheduled background refresh run.
    for _ in range(5):
        await asyncio.sleep(0)
    assert cache.get_custom_emoji_names() == ("bb",)     # refreshed in the background
    assert api.emoji_list.await_count == 1               # the burst scheduled exactly one refresh
    assert cache._refreshing is False


def test_getter_no_running_loop_returns_current_without_raising():
    client, api = _emoji_client({"ok": True, "emoji": {"aa": "u"}})
    cache = WorkspaceEmojiCache(client)
    # No running loop (sync context), never fetched → returns the empty tuple, schedules nothing.
    assert cache.get_custom_emoji_names() == ()
    api.emoji_list.assert_not_awaited()


def test_startup_warms_cache_and_base_wires_it():
    import inspect
    from slack_client import base
    start_src = inspect.getsource(SlackMessagingMixin.start)
    assert "workspace_emojis" in start_src and "refresh()" in start_src
    assert "WorkspaceEmojiCache(self)" in inspect.getsource(base)


# =============================================================== react_to_message factory

class _MutableCache:
    """A workspace_emojis stub whose name tuple can change at runtime (no restart)."""
    def __init__(self, names):
        self._names = tuple(names)

    def set(self, names):
        self._names = tuple(names)

    def get_custom_emoji_names(self):
        return self._names


def _react_host(cache):
    s = MagicMock()
    s.workspace_emojis = cache
    # Must be bound explicitly: on a bare MagicMock the call would return a truthy Mock, so the
    # "no customs → no search pointer" case would pass for the wrong reason.
    s._custom_emoji_available = SlackMessagingMixin._custom_emoji_available.__get__(s)
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    return s


def _emoji_field(host):
    return host.get_react_tool_schema()["parameters"]["properties"]["emoji"]


def test_react_schema_points_at_search_not_a_name_list(monkeypatch):
    # The schema used to inline an ALPHABETICAL prefix of the custom names. With ~1,400 emoji
    # in a real workspace that spent ~600 chars per request on "000, 1password_icon, 4cats_q,
    # alabama…" — a slice the model could not use and had to read anyway. It now points at
    # search_workspace_emoji, which reaches all of them on demand for a fraction of the tokens.
    monkeypatch.setattr(config, "reaction_emojis", [])
    emoji = _emoji_field(_react_host(_MutableCache(["party_parrot", "shipit"])))
    assert "enum" not in emoji                            # an enum would forbid standard emoji
    assert "search_workspace_emoji" in emoji["description"]
    # the names themselves must NOT be inlined any more
    assert "party_parrot" not in emoji["description"]
    assert "shipit" not in emoji["description"]


def test_react_schema_omits_search_pointer_without_customs(monkeypatch):
    # No customs (or no emoji:read scope) → don't advertise a tool with nothing to find.
    monkeypatch.setattr(config, "reaction_emojis", [])
    emoji = _emoji_field(_react_host(_MutableCache([])))
    assert "search_workspace_emoji" not in emoji["description"]


def test_react_schema_reflects_live_cache_without_restart(monkeypatch):
    # The factory is still called per request, so the pointer appears/disappears with the
    # live cache rather than at process start.
    monkeypatch.setattr(config, "reaction_emojis", [])
    cache = _MutableCache([])
    host = _react_host(cache)
    assert "search_workspace_emoji" not in _emoji_field(host)["description"]
    cache.set(["shipit"])                                 # refreshed at runtime
    assert "search_workspace_emoji" in _emoji_field(host)["description"]


def test_react_schema_enum_suppresses_customs(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", ":eyes:"])
    emoji = _emoji_field(_react_host(_MutableCache(["party_parrot"])))
    assert emoji["enum"] == ["thumbsup", "eyes"]          # allowlist is the hard constraint
    assert "party_parrot" not in emoji["description"]     # customs never injected over it


# ================================================ the palette is not a gate input any more

@pytest.mark.asyncio
async def test_the_gate_cannot_be_handed_a_palette(monkeypatch):
    """INVERTED from four tests that fed the gate a ranked palette and checked what it received.

    They asserted the workspace-wide observed-usage list arrived ranked, aggregated across
    channels, empty before anything was observed, and suppressed under an allowlist. All four
    described an input to a decision the gate no longer makes — WHICH emoji to place — so there is
    no successor assertion, only the absence. Asserted at the signature, because a swallowed kwarg
    would let a caller believe a palette had been sent."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    from message_processor.participation import ParticipationEngine as _Engine

    class _Client:
        async def classify_wake(self, *, sources, channel_steering_text=None):
            return False

    engine = _Engine(_Client())
    with pytest.raises(TypeError):
        await engine.evaluate(channel_id="C1", ts="1.0", text="hi",
                              workspace_custom_emojis=["party_parrot"])


def test_nothing_builds_a_palette_for_the_gate_any_more(monkeypatch):
    """The producer side. main.py used to rank the tally and cap it on the gate's hot path, and the
    classifier rendered it as an emoji shortlist. Both are gone — and so, now, is the tally behind
    them (see the tripwire at the foot of this file); what survives is the CATALOG, which the
    responder's react tool and search_workspace_emoji answer from by name."""
    import inspect
    import main
    from openai_client.api import responses

    for src in (inspect.getsource(main), inspect.getsource(responses.classify_wake)):
        assert "workspace_custom_emojis" not in src
        assert "participation_custom_emoji_cap" not in src
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT as p
    for retired in ("Allowed reaction emoji", "actually reacts with", "do not stretch one to fit",
                    "any standard Slack emoji name"):
        assert retired not in p, retired


# =============================================================== _coerce_emoji + executor

class TestTheEmojiRepairStepIsGoneWithThePlacer:
    def test_the_engine_no_longer_coerces_an_emoji(self):
        """`_coerce_emoji` took the classifier's chosen emoji and repaired it — stripped colons,
        fell back to the first allowlisted name, rejected junk. It existed because the GATE placed
        the reaction; nothing in the gate chooses an emoji now, so there is nothing to repair.

        The behaviour it protected did not disappear, it moved: the responder's react tool does its
        own colon-stripping, charset check and allowlist enforcement, which is asserted in the
        executor tests immediately below and in test_participation_telemetry.py's refusal rows."""
        assert not hasattr(ParticipationEngine, "_coerce_emoji")

    @pytest.mark.asyncio
    async def test_the_responder_path_still_strips_colons_and_obeys_the_allowlist(self,
                                                                                 monkeypatch):
        # The surviving half, in one place: a custom name passes with its colons stripped when
        # there is no allowlist, and an off-list name is refused when there is one.
        monkeypatch.setattr(config, "enable_reactions", True, raising=False)
        monkeypatch.setattr(config, "enable_react_tool", True, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        host = MagicMock()
        host.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(host)
        host._reserve_and_react = AsyncMock(return_value={"ok": True})
        ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="123.4")
        assert (await host.execute_react_tool(ctx, {"emoji": ":party_parrot:"}))["ok"] is True
        host._reserve_and_react.assert_awaited_once_with("C1", "123.4", "party_parrot")

        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
        host._reserve_and_react.reset_mock()
        out = await host.execute_react_tool(ctx, {"emoji": "party_parrot"})
        # Refused outright rather than silently rewritten to thumbsup: the gate coerced because it
        # had to place SOMETHING, and the responder can simply not react.
        assert out["ok"] is False and out["error"] == "emoji_not_allowed"
        host._reserve_and_react.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_accepts_custom_and_passes_bare_name(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_react_tool", True)
    monkeypatch.setattr(config, "reaction_emojis", [])
    s = MagicMock()
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = AsyncMock(return_value={"ok": True})
    ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="123.4")
    out = await s.execute_react_tool(ctx, {"emoji": ":party_parrot:"})
    assert out["ok"] is True
    # colons stripped; a name unknown to the standard set is NOT rejected against the customs.
    s._reserve_and_react.assert_awaited_once_with("C1", "123.4", "party_parrot")


@pytest.mark.asyncio
async def test_executor_rejects_syntactic_garbage(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_react_tool", True)
    monkeypatch.setattr(config, "reaction_emojis", [])
    s = MagicMock()
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = AsyncMock()
    ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="123.4")
    out = await s.execute_react_tool(ctx, {"emoji": "not valid!"})
    assert out["ok"] is False and out["error"] == "invalid_emoji"
    s._reserve_and_react.assert_not_awaited()


@pytest.mark.asyncio
async def test_react_add_strips_colons_to_bare_name(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    s = MagicMock()
    s.app.client.reactions_add = AsyncMock()
    s._react_add = SlackMessagingMixin._react_add.__get__(s)
    ok, added = await s._react_add("C1", "1.0", ":party_parrot:")
    assert ok and added
    s.app.client.reactions_add.assert_awaited_once_with(
        channel="C1", name="party_parrot", timestamp="1.0")


@pytest.mark.asyncio
async def test_react_add_unknown_emoji_fails_soft(monkeypatch):
    from slack_sdk.errors import SlackApiError
    monkeypatch.setattr(config, "enable_reactions", True)
    s = MagicMock()
    s.app.client.reactions_add = AsyncMock(
        side_effect=SlackApiError("invalid_name", response={"error": "invalid_name"}))
    s._react_add = SlackMessagingMixin._react_add.__get__(s)
    ok, added = await s._react_add("C1", "1.0", "nonexistent_custom")
    assert ok is False and added is False                # never raises


# =============================================================== config + prompt guidance

def test_config_custom_emoji_defaults_and_documented():
    assert config.workspace_emoji_ttl_seconds == 3600
    example = pathlib.Path(".env.example").read_text()
    assert "WORKSPACE_EMOJI_TTL_SECONDS" in example
    # REACT_TOOL_CUSTOM_EMOJI_CAP is gone: it only ever capped an ALPHABETICAL slice of the
    # custom names for the react schema, and keeping a deliberately off-path selector around
    # "for future callers" just invites its accidental return.
    #
    # PARTICIPATION_CUSTOM_EMOJI_CAP and EMOJI_USAGE_FLUSH_SECONDS join it for the same reason:
    # one sized the ranked shortlist the rich gate was handed, the other paced the tally behind
    # that ranking. Neither has a consumer now, and an orphaned knob reads like a live one.
    for gone_attr, gone_key in (
        ("react_tool_custom_emoji_cap", "REACT_TOOL_CUSTOM_EMOJI_CAP"),
        ("participation_custom_emoji_cap", "PARTICIPATION_CUSTOM_EMOJI_CAP"),
        ("emoji_usage_flush_seconds", "EMOJI_USAGE_FLUSH_SECONDS"),
    ):
        assert not hasattr(config, gone_attr)
        assert gone_key not in example


def test_local_tools_guidance_mentions_workspace_custom_emoji():
    from prompts import LOCAL_TOOLS_GUIDANCE
    g = LOCAL_TOOLS_GUIDANCE.lower()
    assert "custom emoji" in g
    assert "standard slack emoji" in g


def test_valid_emoji_name_accepts_custom_shorthand():
    # The custom names surfaced everywhere share the standard emoji charset.
    assert valid_emoji_name("party_parrot") and valid_emoji_name("shipit")
    assert not valid_emoji_name("bad name!")


# =============================================================== search (the discovery half)

def _search_cache(names):
    c = WorkspaceEmojiCache.__new__(WorkspaceEmojiCache)
    c._names = tuple(sorted(names))
    c._expiry = float("inf")      # never expired → the getter won't try to schedule a refresh
    c._refreshing = False
    return c


def test_search_exact_name_wins_outright():
    cache = _search_cache(["shipit", "shipit-parrot-gif", "unrelated"])
    got = cache.search("shipit")
    assert got[0] == "shipit"
    assert "unrelated" not in got


def test_search_tiers_token_then_prefix_then_substring():
    # "ship" is a whole TOKEN of ship-of-theseus, only a PREFIX of shipit, and merely a
    # SUBSTRING of battleship — which is exactly the order they should come back in.
    cache = _search_cache([
        "battleship", "ship-of-theseus", "shipit", "shipit-parrot-gif", "unrelated"])
    got = cache.search("ship")
    assert got[0] == "ship-of-theseus"                       # token hit beats everything below
    assert got.index("shipit") < got.index("battleship")     # prefix beats substring
    # within the prefix tier the shorter name wins: `shipit` before `shipit-parrot-gif`
    assert got.index("shipit") < got.index("shipit-parrot-gif")
    assert "unrelated" not in got


def test_search_ignores_punctuation_and_separator_style():
    cache = _search_cache(["party_parrot", "absolutecinema"])
    assert "party_parrot" in cache.search("party-parrot")   # _ vs - must not matter
    assert cache.search("absolute cinema") == ["absolutecinema"]  # spaces vs concatenation


def test_search_drops_single_char_tokens():
    # REGRESSION: the stray "a" in "celebrate a win" is also a token of `alphabet-white-a`,
    # which floated alphabet junk to the top of every multi-word query.
    cache = _search_cache(["alphabet-white-a", "celebrate-happy", "a-team"])
    got = cache.search("celebrate a win")
    assert got == ["celebrate-happy"]
    assert "alphabet-white-a" not in got


def test_search_empty_query_and_empty_catalog_are_safe():
    assert _search_cache(["shipit"]).search("") == []
    assert _search_cache(["shipit"]).search("   ") == []
    assert _search_cache([]).search("anything") == []


def test_search_respects_limit():
    cache = _search_cache([f"fire-{i}" for i in range(30)])
    assert len(cache.search("fire", limit=5)) == 5
    assert cache.search("fire", limit=0) == []


# =============================================================== search_workspace_emoji tool

def _search_host(names):
    s = MagicMock()
    s.workspace_emojis = _search_cache(names)
    s.log_debug = lambda *a, **k: None
    s.get_emoji_search_tool_schema = SlackMessagingMixin.get_emoji_search_tool_schema.__get__(s)
    s.execute_emoji_search_tool = SlackMessagingMixin.execute_emoji_search_tool.__get__(s)
    return s


def test_search_tool_schema_tells_the_model_to_stay_quiet():
    # A reaction-only turn STREAMS: a narrated "let me look for an emoji…" would commit
    # visible text and destroy the wordless reaction the search exists to enable.
    d = _search_host(["shipit"]).get_emoji_search_tool_schema()["description"]
    assert "silently" in d
    assert "do NOT need it for standard Slack emoji" in d or "not need it for standard" in d


@pytest.mark.asyncio
async def test_search_tool_returns_matches(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "enable_reactions", True, raising=False)
    monkeypatch.setattr(config, "enable_react_tool", True, raising=False)
    out = await _search_host(["shipit", "nope"]).execute_emoji_search_tool(None, {"query": "ship it"})
    assert out["ok"] is True and out["matches"] == ["shipit"]


@pytest.mark.asyncio
async def test_search_tool_refuses_under_an_allowlist(monkeypatch):
    # Defence in depth: the tool isn't even registered when an allowlist is set, but discovery
    # must never become authorization if it somehow is.
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
    monkeypatch.setattr(config, "enable_reactions", True, raising=False)
    monkeypatch.setattr(config, "enable_react_tool", True, raising=False)
    out = await _search_host(["shipit"]).execute_emoji_search_tool(None, {"query": "ship"})
    assert out["ok"] is False and out["error"] == "allowlist_active"


@pytest.mark.asyncio
async def test_search_tool_never_raises_on_a_broken_cache(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "enable_reactions", True, raising=False)
    monkeypatch.setattr(config, "enable_react_tool", True, raising=False)
    host = _search_host(["shipit"])
    host.workspace_emojis.search = MagicMock(side_effect=RuntimeError("boom"))
    out = await host.execute_emoji_search_tool(None, {"query": "ship"})
    assert out["ok"] is True and out["matches"] == []      # degrades to "use a standard emoji"


def test_search_tool_not_registered_under_an_allowlist():
    import inspect
    from slack_client import base
    src = inspect.getsource(base.SlackBot._build_tool_registry)
    assert "search_workspace_emoji" in src
    # registration sits behind the same no-allowlist guard as the customs themselves
    assert "if not (config.reaction_emojis or [])" in src


# =============================================================== the shortlist is gone

def test_no_ranked_custom_emoji_shortlist_survives():
    """DELETED, and the deletion is the contract.

    Roughly twenty tests lived here for an observed-usage TALLY and its DB persistence: ranked
    by count, workspace-wide, DM-excluded, decremented on removal, hydrated at startup, flushed
    on a timer and once at shutdown, protected against double-counting history on restart. All
    of it fed ONE consumer — the ranked custom-emoji shortlist rendered into the old rich gate
    prompt. The binary gate returns one bit and renders no palette, so the ranking has no reader
    and the persistence has nothing worth persisting.

    The CATALOG and search survive and keep their own coverage above: the responder still reaches
    for a custom emoji and still finds it by NAME, which never needed a popularity ranking.

    No destructive migration goes with this. The `emoji_usage` table is no longer CREATED on a
    fresh schema; an existing installation keeps its orphaned copy, because dropping it would
    destroy data on upgrade to save a few kilobytes."""
    from database import DatabaseManager

    for gone in ("top_custom_reactions", "reaction_vocab_snapshot", "hydrate_reaction_vocab"):
        assert not hasattr(ChannelPulse, gone)
    for gone in ("load_emoji_usage_async", "save_emoji_usage_async"):
        assert not hasattr(DatabaseManager, gone)
    # The fresh-schema CREATE is gone; the absence of a DROP is the load-bearing half.
    src = inspect.getsource(DatabaseManager.init_schema)
    assert "CREATE TABLE IF NOT EXISTS emoji_usage" not in src
    assert "DROP TABLE" not in src.upper()


@pytest.mark.asyncio
async def test_backfill_still_seeds_per_message_social_proof():
    # What the backfill's reaction seeding was ALSO doing, and the half that survives: a cold
    # ring must show what the room already reacted to on a message, not just what it reacts to
    # after the next live event.
    client = SimpleNamespace(conversations_history=AsyncMock(return_value={"messages": [
        {"ts": "1.0", "user": "U1", "text": "deploy is green",
         "reactions": [{"name": "shipit", "count": 3}, {"name": "tada", "count": 1}]},
        {"ts": "2.0", "user": "U2", "text": "nice"},
    ]}))
    bot = SimpleNamespace(classify_sender=lambda m: "human", user_cache={})
    pulse = ChannelPulse()
    await pulse.ensure_backfill("C1", client, bot)
    assert pulse.render_reactions("C1", "1.0") == "[reactions: 3\u00d7 shipit, 1\u00d7 tada]"
    assert pulse.render_reactions("C1", "2.0") == ""
