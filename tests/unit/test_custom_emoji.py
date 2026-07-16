"""AREA C — workspace custom emojis available to the model.

Covers the whole C surface:
- WorkspaceEmojiCache.refresh(): parse emoji.list KEYS (incl. alias entries), valid_emoji_name
  filter, sort + dedupe; retain last-good on error; empty only when never fetched.
- get_custom_emoji_names(): SYNC + stale-ok — returns last-good immediately, schedules exactly ONE
  background refresh when expired, never raises (incl. no running loop).
- react_to_message factory: REACTION_EMOJIS empty → budgeted customs in the emoji DESCRIPTION
  (not an enum), reflecting a refreshed cache without restart, respecting count + char budgets;
  REACTION_EMOJIS set → enum allowlist and customs suppressed.
- Classifier plumbing: the "Workspace custom emoji you may also choose…" line renders only when
  there is no allowlist; the main.py gate feeds a deterministic capped list, and only when empty.
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
    s._CUSTOM_EMOJI_CHAR_BUDGET = SlackMessagingMixin._CUSTOM_EMOJI_CHAR_BUDGET
    s._budgeted_custom_emoji_names = SlackMessagingMixin._budgeted_custom_emoji_names.__get__(s)
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    return s


def _emoji_field(host):
    return host.get_react_tool_schema()["parameters"]["properties"]["emoji"]


def test_react_schema_lists_customs_in_description_no_enum(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [])
    emoji = _emoji_field(_react_host(_MutableCache(["party_parrot", "shipit"])))
    assert "enum" not in emoji                            # an enum would forbid standard emoji
    assert "custom emoji" in emoji["description"]
    assert "party_parrot" in emoji["description"] and "shipit" in emoji["description"]


def test_react_schema_reflects_live_cache_without_restart(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [])
    cache = _MutableCache(["party_parrot"])
    host = _react_host(cache)
    assert "party_parrot" in _emoji_field(host)["description"]
    cache.set(["shipit"])                                 # refreshed at runtime
    d2 = _emoji_field(host)["description"]
    assert "shipit" in d2 and "party_parrot" not in d2


def test_react_schema_enum_suppresses_customs(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", ":eyes:"])
    emoji = _emoji_field(_react_host(_MutableCache(["party_parrot"])))
    assert emoji["enum"] == ["thumbsup", "eyes"]          # allowlist is the hard constraint
    assert "party_parrot" not in emoji["description"]     # customs never injected over it


def test_react_schema_respects_count_cap(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [])
    monkeypatch.setattr(config, "react_tool_custom_emoji_cap", 3, raising=False)
    names = [f"c{i}" for i in range(10)]                  # short → count cap binds
    desc = _emoji_field(_react_host(_MutableCache(names)))["description"]
    assert [n for n in names if n in desc] == ["c0", "c1", "c2"]


def test_react_schema_respects_char_budget(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [])
    monkeypatch.setattr(config, "react_tool_custom_emoji_cap", 100, raising=False)
    names = [f"emoji_{i:02d}_" + "x" * 90 for i in range(20)]   # ~99 chars each → char budget binds
    desc = _emoji_field(_react_host(_MutableCache(names)))["description"]
    listed = [n for n in names if n in desc]
    assert 0 < len(listed) < 20                           # the ~600-char budget dropped most
    assert names[0] in desc and names[-1] not in desc


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
async def test_classifier_renders_customs_when_no_allowlist(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    prompt = await _classifier_prompt({"workspace_custom_emojis": ["party_parrot", "shipit"]})
    assert ("Workspace custom emoji you may also choose when one fits: party_parrot, shipit"
            in prompt)
    assert "standard Slack emoji remain allowed" in prompt


@pytest.mark.asyncio
async def test_classifier_omits_customs_when_allowlist_set(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
    prompt = await _classifier_prompt({"workspace_custom_emojis": ["party_parrot"]})
    assert "Allowed reaction emoji (choose one): thumbsup" in prompt
    assert "Workspace custom emoji you may also choose" not in prompt


@pytest.mark.asyncio
async def test_classifier_no_customs_line_when_none(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    prompt = await _classifier_prompt({})
    assert "any standard Slack emoji name (shorthand, no colons)" in prompt
    assert "Workspace custom emoji you may also choose" not in prompt


def _gate_app(monkeypatch, customs):
    from main import ChatBotV2
    from message_processor.participation import ParticipationVerdict
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr("message_processor.canvas_tools.build_catalog",
                        AsyncMock(return_value=[]))
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    captured = {}

    async def _eval(**kw):
        captured.update(kw)
        return ParticipationVerdict(action="ignore")

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
    return app, client, msg, captured


@pytest.mark.asyncio
async def test_gate_feeds_capped_customs_when_no_allowlist(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "participation_custom_emoji_cap", 3, raising=False)
    app, client, msg, captured = _gate_app(monkeypatch, [f"c{i}" for i in range(10)])
    assert await app._gate_verdict(msg, client) is None          # ignore verdict → silent
    assert captured["workspace_custom_emojis"] == ["c0", "c1", "c2"]  # deterministic capped slice


@pytest.mark.asyncio
async def test_gate_omits_customs_when_allowlist_set(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
    app, client, msg, captured = _gate_app(monkeypatch, ["party_parrot"])
    assert await app._gate_verdict(msg, client) is None
    assert captured["workspace_custom_emojis"] == []             # never injected over an allowlist


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
    assert config.react_tool_custom_emoji_cap == 64
    example = pathlib.Path(".env.example").read_text()
    for key in ("WORKSPACE_EMOJI_TTL_SECONDS", "PARTICIPATION_CUSTOM_EMOJI_CAP",
                "REACT_TOOL_CUSTOM_EMOJI_CAP"):
        assert key in example


def test_local_tools_guidance_mentions_workspace_custom_emoji():
    from prompts import LOCAL_TOOLS_GUIDANCE
    g = LOCAL_TOOLS_GUIDANCE.lower()
    assert "custom emoji" in g
    assert "standard slack emoji" in g


def test_valid_emoji_name_accepts_custom_shorthand():
    # The custom names surfaced everywhere share the standard emoji charset.
    assert valid_emoji_name("party_parrot") and valid_emoji_name("shipit")
    assert not valid_emoji_name("bad name!")
