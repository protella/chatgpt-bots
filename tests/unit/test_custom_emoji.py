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
- Classifier plumbing: the "Custom emoji THIS WORKSPACE actually reacts with…" line renders only
  when there is no allowlist; the main.py gate feeds the WORKSPACE-WIDE observed-usage palette
  (aggregated across every channel, DMs excluded — per-channel left quiet rooms with nothing).

Both name lists used to be an ALPHABETICAL prefix of ~1,400 names, so the model only ever saw
"000, 1password_icon, 2605732e-82a0-46b9-b1e0-ecc4f250eb35, 4cats_q, alabama…" — prompt noise it
could not use. The responder now searches on demand; the gate (one tool-free call, emoji placed
directly) gets what the workspace actually reacts with. Neither ever falls back to alphabetical.
- _coerce_emoji stays permissive (standard OR custom), reactions.add uses the bare name, unknown
  fails soft. Config defaults (3600/32/64) + .env.example documentation.

All in-memory; no network/DB.
"""
from __future__ import annotations

import asyncio
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
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


# =============================================================== classifier plumbing

async def _classifier_prompt(signals):
    """Render classify_participation's user-message content with a stubbed API call."""
    from openai_client.api import responses as responses_api
    captured = {}

    async def _fake(self, fn, *, operation_type, **params):
        captured["input"] = params["input"]
        return SimpleNamespace(output=[])

    host = MagicMock()
    host._safe_api_call = _fake.__get__(host)
    host.classify_participation = responses_api.classify_participation.__get__(host)
    await host.classify_participation(text="hi", signals=signals)
    return captured["input"][1]["content"]


@pytest.mark.asyncio
async def test_customs_are_offered_as_names_with_no_popularity_claim(monkeypatch):
    """Custom names are worth surfacing; a ranking over them is not, at any list length.

    Measured 2026-07-26 in the shared test channel, both assistants answering the same room:
    ranking these by observed use and calling them "this team's own vocabulary" concentrated our
    social reactions onto 14 distinct emoji across 38, with :dumpster-fire: alone at 24%.
    Anthropic's bot spread 43 reactions over 32 distinct names, most-used at 7% — and it has no
    ranked palette at all, just an opt-in name lookup. So the tally no longer steers anything and
    the list carries no ordering claim. Names the model cannot guess exist (:absolutecinema:) are
    the entire remaining value.

    A previous revision gated the framing on a hand-picked count of observed names. That was an
    arbitrary threshold and is gone; there is one framing at every length."""
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    for names in (["squirrel", "dumpster-fire", "absolutecinema"],
                  ["party_parrot", "shipit", "rocket", "facepalm", "yay", "boom", "sadpanda",
                   "tada2", "yolo"]):
        prompt = await _classifier_prompt({"workspace_custom_emojis": names})
        assert ", ".join(names) in prompt
        # No claim that the order means anything, or that these are what the team favours.
        assert "most-used first" not in prompt
        assert "this team's own vocabulary" not in prompt
        assert "prefer one of these" not in prompt
        assert "do not stretch one to fit" in prompt
        # Standard emoji stay on the table, so the model can still react with anything apt.
        assert "any standard Slack emoji name (shorthand, no colons)" in prompt


@pytest.mark.asyncio
async def test_classifier_omits_customs_when_allowlist_set(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
    prompt = await _classifier_prompt({"workspace_custom_emojis": ["party_parrot"]})
    assert "Allowed reaction emoji (choose one): thumbsup" in prompt
    assert "actually reacts with" not in prompt


@pytest.mark.asyncio
async def test_classifier_no_customs_line_when_none(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    prompt = await _classifier_prompt({})
    assert "any standard Slack emoji name (shorthand, no colons)" in prompt
    assert "actually reacts with" not in prompt


def _gate_app(monkeypatch, customs):
    """A gate wired to a fake engine, plus the declines it produced.

    `evaluate` returns a GateEvaluation, which is what the engine actually returns — a bare
    ParticipationVerdict makes the gate raise on `.decline_cause` and swallow it as silence, so
    these tests passed while asserting the classifier inputs of a call the real code never
    completes. `declines` exists so each test can prove the gate ran its real path.
    """
    from main import ChatBotV2
    from message_processor.participation import GateEvaluation, ParticipationVerdict
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr("message_processor.canvas_tools.build_catalog",
                        AsyncMock(return_value=[]))
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    captured = {}
    declines = []

    async def _eval(**kw):
        captured.update(kw)
        return GateEvaluation(verdict=ParticipationVerdict(action="ignore"))

    def _decline(channel_id=None, trigger_ts=None, cause=None, **fields):
        declines.append(cause)

    monkeypatch.setattr("message_processor.participation_telemetry.gate_declined", _decline)
    app.participation_engine = MagicMock()
    app.participation_engine.evaluate = _eval
    app.participation_engine.note_arrival = MagicMock()
    client = MagicMock()
    client.channel_pulse = None
    client.get_channel_context = AsyncMock(return_value={})
    client.workspace_emojis = _MutableCache(customs)
    msg = Message(text="x", user_id="U1", channel_id="C1", thread_id="10.0",
                  metadata={"ts": "10.0", "participation_check": True,
                            "participation_level": "judicious"})
    return app, client, msg, captured, declines


@pytest.mark.asyncio
async def test_gate_feeds_observed_customs_ranked_by_use(monkeypatch):
    # The gate's palette is what the WORKSPACE reacts with, most-used first — not an
    # alphabetical slice. `rare` is alphabetically first and least used; it must come LAST.
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "participation_custom_emoji_cap", 3, raising=False)
    app, client, msg, captured, declines = _gate_app(
        monkeypatch, ["rare", "shipit", "party_parrot", "elsewhere"])
    pulse = ChannelPulse()
    for _ in range(5):
        pulse.add_reaction("C1", "10.0", "shipit")
    for _ in range(3):
        pulse.add_reaction("C1", "11.0", "party_parrot")
    pulse.add_reaction("C1", "12.0", "rare")
    pulse.add_reaction("C1", "12.5", "rare")
    pulse.add_reaction("C1", "13.0", "thumbsup")     # standard → not a custom, filtered out
    pulse.add_reaction("CZ", "14.0", "elsewhere")    # another channel DOES count (workspace-wide)
    client.channel_pulse = pulse
    assert await app._gate_verdict(msg, client) is None          # ignore verdict → silent
    # cap=3 → the three most-used. `elsewhere` (1 use, another channel) IS counted but ranks
    # 4th, so it falls outside the cap; the next test proves cross-channel aggregation directly.
    assert captured["workspace_custom_emojis"] == ["shipit", "party_parrot", "rare"]
    assert declines == []            # the real gate path ran, not its except-clause


@pytest.mark.asyncio
async def test_gate_palette_includes_other_channels(monkeypatch):
    # Same setup, cap raised: the emoji seen only in ANOTHER channel now appears. This is the
    # whole point of the workspace-wide tally — a channel with no reactions of its own still
    # gets the team's vocabulary instead of nothing.
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "participation_custom_emoji_cap", 10, raising=False)
    app, client, msg, captured, declines = _gate_app(monkeypatch, ["elsewhere"])
    pulse = ChannelPulse()
    pulse.add_reaction("CZ", "14.0", "elsewhere")        # never seen in C1, the gated channel
    client.channel_pulse = pulse
    assert await app._gate_verdict(msg, client) is None
    assert captured["workspace_custom_emojis"] == ["elsewhere"]
    assert declines == []


@pytest.mark.asyncio
async def test_gate_sends_no_customs_when_nothing_observed_yet(monkeypatch):
    # The point of the rewrite: with no observed usage the gate sends NOTHING rather than
    # falling back to an alphabetical slice. Standard emoji are the fallback; junk is not.
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    app, client, msg, captured, declines = _gate_app(monkeypatch, [f"c{i}" for i in range(10)])
    client.channel_pulse = ChannelPulse()                        # nothing observed yet
    assert await app._gate_verdict(msg, client) is None
    assert captured["workspace_custom_emojis"] == []
    assert declines == []


@pytest.mark.asyncio
async def test_gate_omits_customs_when_allowlist_set(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
    app, client, msg, captured, declines = _gate_app(monkeypatch, ["party_parrot"])
    assert await app._gate_verdict(msg, client) is None
    assert captured["workspace_custom_emojis"] == []             # never injected over an allowlist
    assert declines == []


# =============================================================== _coerce_emoji + executor

class TestCoerceEmojiPermissive:
    def test_standard_and_custom_accepted_without_allowlist(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        assert ParticipationEngine._coerce_emoji({"emoji": "joy"}, True) == "joy"
        # A workspace custom name (same charset) is accepted; a valid standard emoji is NEVER
        # rejected just because it isn't in the custom set.
        assert ParticipationEngine._coerce_emoji({"emoji": ":party_parrot:"}, True) == "party_parrot"
        assert ParticipationEngine._coerce_emoji({"emoji": "bad name!"}, True) is None

    def test_allowlist_still_constrains(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
        assert ParticipationEngine._coerce_emoji({"emoji": "party_parrot"}, True) == "thumbsup"
        assert ParticipationEngine._coerce_emoji({"emoji": "party_parrot"}, False) is None


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
    assert config.participation_custom_emoji_cap == 32
    assert config.emoji_usage_flush_seconds == 300
    example = pathlib.Path(".env.example").read_text()
    for key in ("WORKSPACE_EMOJI_TTL_SECONDS", "PARTICIPATION_CUSTOM_EMOJI_CAP",
                "EMOJI_USAGE_FLUSH_SECONDS"):
        assert key in example
    # REACT_TOOL_CUSTOM_EMOJI_CAP is gone: it only ever capped an ALPHABETICAL slice of the
    # custom names for the react schema, and keeping a deliberately off-path selector around
    # "for future callers" just invites its accidental return.
    assert not hasattr(config, "react_tool_custom_emoji_cap")
    assert "REACT_TOOL_CUSTOM_EMOJI_CAP" not in example


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


# =============================================================== observed-usage palette

def test_top_custom_reactions_ranks_by_count_and_filters_to_customs():
    pulse = ChannelPulse()
    for _ in range(4):
        pulse.add_reaction("C1", "1.0", "shipit")
    pulse.add_reaction("C1", "2.0", "party_parrot")
    pulse.add_reaction("C1", "3.0", "thumbsup")            # standard, not in `allowed`
    got = pulse.top_custom_reactions(allowed=["shipit", "party_parrot"], limit=10)
    assert got == ["shipit", "party_parrot"]


def test_top_custom_reactions_aggregates_across_channels():
    # WORKSPACE-wide on purpose. Scoped per-channel, a quiet or freshly-joined channel got an
    # empty palette — which is most channels most of the time, and after every restart.
    pulse = ChannelPulse()
    for _ in range(3):
        pulse.add_reaction("C1", "1.0", "shipit")
    pulse.add_reaction("C2", "2.0", "party_parrot")        # a DIFFERENT channel still counts
    assert pulse.top_custom_reactions(allowed=["shipit", "party_parrot"]) == [
        "shipit", "party_parrot"]


def test_top_custom_reactions_empty_before_anything_is_observed():
    assert ChannelPulse().top_custom_reactions(allowed=["shipit"]) == []


def test_top_custom_reactions_ignores_dm_reactions():
    # DMs are excluded from the tally, same guard as the social-proof store.
    pulse = ChannelPulse()
    pulse.add_reaction("D123", "1.0", "shipit")
    assert pulse.top_custom_reactions(allowed=["shipit"]) == []


def test_top_custom_reactions_decrements_on_removal():
    pulse = ChannelPulse()
    pulse.add_reaction("C1", "1.0", "shipit")
    pulse.add_reaction("C1", "2.0", "party_parrot")
    pulse.remove_reaction("C1", "1.0", "shipit")
    assert pulse.top_custom_reactions(allowed=["shipit", "party_parrot"]) == ["party_parrot"]


def test_top_custom_reactions_drops_emoji_deleted_since_they_were_observed():
    pulse = ChannelPulse()
    pulse.add_reaction("C1", "1.0", "since_deleted")
    assert pulse.top_custom_reactions(allowed=["shipit"]) == []


@pytest.mark.asyncio
async def test_backfill_seeds_reactions_from_history():
    # conversations.history already carries reactions; without seeding them a FIRST-EVER run has
    # an empty palette until someone reacts live. Exercised for real rather than by reading the
    # source, because the interesting behaviour is the interaction with hydration below.
    client = SimpleNamespace(conversations_history=AsyncMock(return_value={"messages": [
        {"ts": "1.0", "user": "U1", "text": "deploy is green",
         "reactions": [{"name": "shipit", "count": 3}, {"name": "tada", "count": 1}]},
        {"ts": "2.0", "user": "U2", "text": "nice"},
    ]}))
    bot = SimpleNamespace(classify_sender=lambda m: "human", user_cache={})
    pulse = ChannelPulse()
    await pulse.ensure_backfill("C1", client, bot)
    assert pulse.reaction_vocab_snapshot() == {"shipit": 3, "tada": 1}


@pytest.mark.asyncio
async def test_backfill_does_not_double_count_a_persisted_tally():
    # THE RESTART BUG. Hydration happens in start() before the socket opens; each channel's
    # backfill then re-reads the same historical reactions. Counting them again inflated every
    # emoji in recent history by its whole count on EVERY restart (measured 100 -> 104).
    client = SimpleNamespace(conversations_history=AsyncMock(return_value={"messages": [
        {"ts": "1.0", "user": "U1", "text": "deploy is green",
         "reactions": [{"name": "shipit", "count": 4}]},
    ]}))
    bot = SimpleNamespace(classify_sender=lambda m: "human", user_cache={})
    pulse = ChannelPulse()
    pulse.hydrate_reaction_vocab({"shipit": 100})        # what the last run persisted
    await pulse.ensure_backfill("C1", client, bot)
    assert pulse.reaction_vocab_snapshot() == {"shipit": 100}
    # a LIVE reaction still counts — only history replays are suppressed
    pulse.add_reaction("C1", "9.0", "shipit")
    assert pulse.reaction_vocab_snapshot() == {"shipit": 101}


# =============================================================== persistence of the tally

def test_hydrate_merges_and_never_shrinks_observed_counts():
    # The backfill fires on a channel's first message and can beat the DB load. Rehydrating
    # must not roll the tally BACKWARDS to whatever the last flush happened to catch.
    pulse = ChannelPulse()
    for _ in range(4):
        pulse.add_reaction("C1", "1.0", "shipit")            # observed live: 4
    pulse.hydrate_reaction_vocab({"shipit": 2, "party_parrot": 9})
    snap = pulse.reaction_vocab_snapshot()
    assert snap["shipit"] == 4                               # kept the larger, not the persisted 2
    assert snap["party_parrot"] == 9                         # and gained what it had not seen


def test_hydrate_tolerates_junk():
    pulse = ChannelPulse()
    pulse.hydrate_reaction_vocab({"": 5, "ok": "not-a-number", ":fine:": 3, "neg": -2, "z": 0})
    assert pulse.reaction_vocab_snapshot() == {"fine": 3}     # colons stripped, junk dropped
    pulse.hydrate_reaction_vocab(None)                        # no-op, must not raise
    assert pulse.reaction_vocab_snapshot() == {"fine": 3}


def test_snapshot_is_a_copy_and_content_free():
    pulse = ChannelPulse()
    pulse.add_reaction("C1", "1.0", "shipit")
    snap = pulse.reaction_vocab_snapshot()
    snap["shipit"] = 999                                      # mutating the copy must not leak
    assert pulse.reaction_vocab_snapshot()["shipit"] == 1
    # name -> count only: no channel, ts, author or text can be reconstructed from it
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in
               pulse.reaction_vocab_snapshot().items())


@pytest.mark.asyncio
async def test_emoji_usage_round_trips_through_the_db(tmp_path):
    import sqlite3
    from database import DatabaseManager
    db = DatabaseManager("test")
    db.db_path = str(tmp_path / "t.db")          # same repoint the DB suite's fixture uses
    db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
    db.init_schema()
    await db.save_emoji_usage_async({"shipit": 4, "party_parrot": 1, "gone": 0})
    assert await db.load_emoji_usage_async() == {"shipit": 4, "party_parrot": 1}  # zero pruned
    # absolute counts: a later save REPLACES rather than accumulating, and drops missing names
    await db.save_emoji_usage_async({"shipit": 6})
    assert await db.load_emoji_usage_async() == {"shipit": 6}
    await db.save_emoji_usage_async({})
    assert await db.load_emoji_usage_async() == {}


@pytest.mark.asyncio
async def test_flush_and_load_never_raise_without_db_or_pulse():
    host = MagicMock()
    host.log_debug = lambda *a, **k: None
    host.channel_pulse = None
    host.db = None
    host._load_emoji_usage = SlackMessagingMixin._load_emoji_usage.__get__(host)
    host.flush_emoji_usage = SlackMessagingMixin.flush_emoji_usage.__get__(host)
    await host._load_emoji_usage()
    await host.flush_emoji_usage()


@pytest.mark.asyncio
async def test_flush_survives_a_failing_db():
    host = MagicMock()
    host.log_debug = lambda *a, **k: None
    host.channel_pulse = ChannelPulse()
    host.channel_pulse.add_reaction("C1", "1.0", "shipit")
    host.db = MagicMock()
    host.db.save_emoji_usage_async = AsyncMock(side_effect=RuntimeError("locked"))
    host.flush_emoji_usage = SlackMessagingMixin.flush_emoji_usage.__get__(host)
    await host.flush_emoji_usage()          # must not propagate


def test_shutdown_flushes_the_tally_after_ingress_stops():
    # A wiring guard, deliberately by inspection: stop() tears down sockets and sessions that are
    # impractical to stand up here. What it pins is the ORDER — the final snapshot must come
    # after the handler teardown, or a reaction arriving late in shutdown is observed and never
    # persisted. It also pins that the periodic task is cancelled rather than left running.
    import inspect
    src = inspect.getsource(SlackMessagingMixin.stop)
    assert "_emoji_flush_task" in src and "task.cancel()" in src
    assert src.count("flush_emoji_usage") == 1, "the final flush should happen exactly once"
    assert src.index("_emoji_flush_task") < src.index("await self.flush_emoji_usage()")
    assert src.index("self.handler") < src.index("await self.flush_emoji_usage()")
