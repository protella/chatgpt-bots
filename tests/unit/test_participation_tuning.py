"""Participation tuning + 3 bug fixes (2026-07-21).

Covers the prompt-wording contracts (A1 value floor, A2 open-question rule, B banter
rule, C1 mid-flight escape, C2 truthfulness sentence), the C1 real-event composition,
and the three bug fixes: BF1 (search_slack gated on the event's action_token), BF2
(username resolution in rebuilt history and tool-returned histories), and BF3 (pulse
envelope observability).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from prompts import (
    PARTICIPATION_SYSTEM_PROMPT,
    SLACK_SYSTEM_PROMPT,
    NO_REPLY_CONTRACT_SUFFIX,
)
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.search_tool import SlackSearchToolMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.utilities import SlackUtilitiesMixin
from tool_registry import ToolContext, ToolRegistry


# =========================================================== prompt wording (A1/A2/B/C1/C2)

def test_a1_value_floor_present():
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "Judge value by whether the assistant can supply the kind of answer requested" in p
    assert "sketch the honest lead of the reply" in p
    assert "A capability disclaimer plus a pointer, a restatement, or an unsolicited summary" in p
    assert "Requests for human firsthand experience or human action/authority" in p
    assert "An internal-fact question is a respond case only when" in p


def test_a1_value_floor_exempts_direct_summons():
    """Live regression (2026-07-21): the floor swallowed "chatgpt, do you know X?" —
    a bare-name summon is gated, so the floor must not reach it or the responder's
    honest-answer contract (C1) never runs."""
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "This floor applies only AFTER the addressee is resolved" in p
    assert "genuinely and currently put to THIS assistant by one of its own names" in p
    assert "being left on read is worse than a one-line" in p
    # the carve-out must not re-open the name-drop hole the addressee rules close
    assert "A name-drop, a quotation, or the assistant as a topic of discussion is not a summons" in p


def test_a2_open_question_rule_replaced():
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "resets the addressee, but does not by itself justify a response" in p
    assert "The source must match the question" in p
    assert 'Do not turn "anyone tried X?" into an unsolicited web summary' in p
    # the old open-question wording is gone
    assert "a colleague with the data at hand would speak up" not in p
    assert "that those tools can answer directly is a respond case" not in p


def test_b_banter_rule_replaced():
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "Playful banter or teasing genuinely aimed AT the assistant is participation-worthy" in p
    assert "a single emoji carries the beat, choose react" in p
    assert "merely talking about the assistant stays theirs" in p
    # old banter wording is gone
    assert "is a respond case, not marginal-value noise to ignore" not in p


def test_c1_mid_flight_escape_present():
    s = NO_REPLY_CONTRACT_SUFFIX
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "addressed by name, prefer a brief honest answer over silence" in s
    assert s.endswith("over silence.]")  # still one bracketed paragraph


def test_c2_truthfulness_sentence_present():
    s = SLACK_SYSTEM_PROMPT
    assert "don't fake familiarity" in s
    assert "a confident wrong guess reads far worse than either" in s


# =========================================== C1 composition on the real name-mention turn

def _no_reply_schema():
    return {"type": "function", "name": "no_response_needed",
            "parameters": {"type": "object",
                           "properties": {"reason": {"type": "string"}},
                           "required": ["reason"]}}


def _registry_no_reply_and_search():
    reg = ToolRegistry()
    reg.register(
        _no_reply_schema(), AsyncMock(return_value={"ok": True}),
        enabled=lambda cfg: config.enable_no_reply_tool and bool(cfg.get("_unprompted_turn")))
    reg.register(
        {"type": "function", "name": "search_slack", "parameters": {}},
        AsyncMock(return_value={"ok": True}),
        enabled=lambda cfg: bool(cfg.get("_slack_search_available")))
    return reg


class _MatHost:
    """Binds the real _materialize_request_tools onto a bare host (same pattern as
    test_no_reply_tool)."""
    def __init__(self, registry):
        from message_processor.handlers.text import TextHandlerMixin
        for n in ("_materialize_request_tools", "_get_tool_registry"):
            setattr(self, n, getattr(TextHandlerMixin, n).__get__(self))
        self._client = SimpleNamespace(tool_registry=registry)


def _msg(**md_extra):
    md = {"ts": "1.1"}
    md.update(md_extra)
    return SimpleNamespace(metadata=md, channel_id="C1")


def test_name_mention_turn_exposes_no_reply_suffix_and_tool(mock_env):
    # The real bare-name hit sets participation_check=True, participation_name_hit=True,
    # wake_source="name_mention" together (message_events ~709/717). That composition still
    # receives the F2 suffix + no_response_needed, so the C1 mid-flight escape reaches it.
    host = _MatHost(_registry_no_reply_and_search())
    msg = _msg(participation_check=True, participation_name_hit=True, wake_source="name_mention")
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, msg, tools_disabled=False)
    assert available is True
    assert suffix == NO_REPLY_CONTRACT_SUFFIX
    names = {s["name"] for s in registry.schemas(request_config)}
    assert "no_response_needed" in names


# ========================================================== BF1 — search_slack action_token gate

def test_materialize_sets_slack_search_available_from_action_token(mock_env):
    host = _MatHost(_registry_no_reply_and_search())
    _, cfg_on, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(participation_check=True, action_token="tok"), False)
    assert cfg_on["_slack_search_available"] is True

    _, cfg_off, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(participation_check=True), False)
    assert cfg_off["_slack_search_available"] is False


def test_search_schema_present_only_with_action_token(mock_env):
    host = _MatHost(_registry_no_reply_and_search())
    reg_on, cfg_on, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(participation_check=True, action_token="tok"), False)
    assert "search_slack" in {s["name"] for s in reg_on.schemas(cfg_on)}

    reg_off, cfg_off, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(participation_check=True), False)
    assert "search_slack" not in {s["name"] for s in reg_off.schemas(cfg_off)}


@pytest.mark.asyncio
async def test_search_tool_runtime_still_refuses_without_token():
    # Defense in depth: even if a schema slips through, the executor refuses tokenless calls.
    class _Bot(SlackSearchToolMixin):
        def __init__(self):
            self.app = MagicMock()

        def log_info(self, *a, **k): pass
        log_debug = log_warning = log_error = log_info

    out = await _Bot().execute_search_tool(ToolContext(action_token=None), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_unavailable"


# ==================================================== BF1 — end-to-end dispatch matrix

def _slack_tool_mock():
    """Schema-provider stand-in for SlackBot._build_tool_registry (mirrors test_tool_loop):
    every schema getter returns a REAL dict so register() doesn't read a MagicMock as a
    per-request schema factory."""
    s = MagicMock()
    s.get_history_tools_for_openai.return_value = [
        {"type": "function", "name": "fetch_channel_history", "parameters": {}}]
    s.get_react_tool_schema.return_value = {
        "type": "function", "name": "react_to_message", "parameters": {}}
    s.get_search_tool_schema.return_value = {
        "type": "function", "name": "search_slack", "parameters": {}}
    s.get_post_to_thread_tool_schema.return_value = {
        "type": "function", "name": "post_to_thread", "parameters": {}}
    s.get_no_reply_tool_schema.return_value = {
        "type": "function", "name": "no_response_needed", "parameters": {}}
    return s


def _real_registry_with_search(monkeypatch):
    """Registry built by the REAL SlackBot._build_tool_registry, so search_slack carries the
    action_token predicate from base.py (BF1) — not a re-declared copy."""
    from slack_client.base import SlackBot
    for gate in ("enable_history_tools", "enable_reactions", "enable_react_tool",
                 "enable_search_tool", "enable_channel_memory", "enable_post_to_thread_tool",
                 "enable_read_document_tool", "enable_people_tools", "enable_deep_research"):
        monkeypatch.setattr(config, gate, True)
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
    return SlackBot._build_tool_registry(_slack_tool_mock())


@pytest.mark.parametrize("label, md, channel_id, expect_search", [
    ("app_mention", {"mentioned_self": True, "action_token": "tok"}, "C1", True),
    ("dm", {"action_token": "tok"}, "D1", True),
    ("gated_channel", {"participation_check": True}, "C1", False),
    ("thread_continuation", {"wake_source": "thread_continuation"}, "C1", False),
])
def test_bf1_dispatch_matrix(mock_env, monkeypatch, label, md, channel_id, expect_search):
    # metadata → _materialize_request_tools (stamps _slack_search_available from action_token)
    # → the real registry predicate. search_slack is exposed exactly when the event carries a
    # token (@mention channel events + DMs), hidden on unmentioned/continuation turns.
    registry = _real_registry_with_search(monkeypatch)
    host = _MatHost(registry)
    msg = SimpleNamespace(metadata={"ts": "1.1", **md}, channel_id=channel_id)
    _, request_config, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, msg, tools_disabled=False)
    names = {s["name"] for s in registry.schemas(request_config)}
    assert ("search_slack" in names) is expect_search


# =============================================== BF2 — read-only, batched username resolution

def _mock_db(db_users=None):
    """A DB whose user read is the BULK get_user_infos_async (read-only, one connection). The
    write methods that would create rows / bump last_seen are present so tests can assert they
    stay UNCALLED."""
    db = MagicMock()
    users = db_users or {}

    async def _infos(user_ids):
        return {uid: users[uid] for uid in user_ids if uid in users}

    db.get_user_infos_async = AsyncMock(side_effect=_infos)
    db.get_or_create_user_async = AsyncMock()   # a WRITE — reading must never call it
    db.save_user_info_async = AsyncMock()        # a WRITE — reading must never call it
    return db


def _mock_api(remote_names=None, calls=None):
    """A Slack client whose users.info records every id it's asked for, so tests can pin the
    remote-lookup budget and negative caching."""
    names = remote_names or {}
    sink = calls if calls is not None else []

    async def _users_info(user):
        sink.append(user)
        if user in names:
            return {"ok": True, "user": {"name": names[user],
                                         "profile": {"display_name": names[user]}}}
        return {"ok": False, "error": "user_not_found"}

    api = MagicMock()
    api.users_info = AsyncMock(side_effect=_users_info)
    return api


class _Resolver(SlackUtilitiesMixin):
    def __init__(self, db_users=None):
        self.user_cache = {}
        self.db = _mock_db(db_users)

    def log_debug(self, *a, **k): pass
    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
async def test_resolver_memory_cache_hit_no_db_no_remote():
    h = _Resolver()
    h.user_cache["U1"] = {"username": "alice"}
    calls = []
    out = await h.resolve_usernames(["U1"], _mock_api(calls=calls))
    assert out == {"U1": "alice"}
    h.db.get_user_infos_async.assert_not_called()   # cache hit: no DB read at all
    assert calls == []


@pytest.mark.asyncio
async def test_resolver_db_hit_is_read_only():
    h = _Resolver(db_users={"U2": {"username": "bob"}})
    calls = []
    out = await h.resolve_usernames(["U2"], _mock_api(calls=calls))
    assert out == {"U2": "bob"}
    assert calls == []                                  # DB hit needs no remote call
    h.db.get_or_create_user_async.assert_not_called()   # reading never creates rows
    h.db.save_user_info_async.assert_not_called()
    assert h.user_cache["U2"]["username"] == "bob"      # warmed for the rest of the request


@pytest.mark.asyncio
async def test_resolver_remote_fetch_warms_cache_no_db_write():
    h = _Resolver()
    calls = []
    out = await h.resolve_usernames(["U3"], _mock_api({"U3": "carol"}, calls))
    assert out == {"U3": "carol"}
    assert calls == ["U3"]
    h.db.get_or_create_user_async.assert_not_called()
    h.db.save_user_info_async.assert_not_called()
    assert h.user_cache["U3"]["username"] == "carol"    # memory cache only, no DB persistence


@pytest.mark.asyncio
async def test_resolver_repeated_failure_one_remote_attempt():
    h = _Resolver()
    calls = []
    out = await h.resolve_usernames(["U9", "U9", "U9"], _mock_api(calls=calls))
    assert out == {}                    # unresolved -> omitted; caller keeps the raw id
    assert calls == ["U9"]              # deduped + negative-cached: exactly one attempt


@pytest.mark.asyncio
async def test_resolver_budget_caps_remote_lookups():
    h = _Resolver()
    calls = []
    ids = [f"U{i}" for i in range(30)]  # all unknown -> all misses
    out = await h.resolve_usernames(ids, _mock_api(calls=calls), max_remote_lookups=25)
    assert out == {}
    assert calls == ids[:25]            # over-budget ids stay raw, no extra remote calls


@pytest.mark.asyncio
async def test_resolver_budget_is_deterministic_by_input_order():
    # Blocker 2: the budget resolves the FIRST N in INPUT order — never a hash-random subset.
    # Same ids in two different orders => each resolves its own first N, deterministically.
    ids = [f"U{i}" for i in range(10)]
    calls_a = []
    await _Resolver().resolve_usernames(ids, _mock_api(calls=calls_a), max_remote_lookups=3)
    assert calls_a == ["U0", "U1", "U2"]
    calls_b = []
    await _Resolver().resolve_usernames(
        list(reversed(ids)), _mock_api(calls=calls_b), max_remote_lookups=3)
    assert calls_b == ["U9", "U8", "U7"]


@pytest.mark.asyncio
async def test_resolver_db_read_failure_degrades_to_remote():
    h = _Resolver()
    h.db.get_user_infos_async = AsyncMock(side_effect=RuntimeError("db down"))
    calls = []
    out = await h.resolve_usernames(["U1"], _mock_api({"U1": "alice"}, calls))
    assert out == {"U1": "alice"}       # DB error -> falls through to remote, still resolves
    assert calls == ["U1"]


class _HistHarness(SlackHistoryToolMixin, SlackUtilitiesMixin):
    def __init__(self, db_users=None, remote_names=None):
        self.app = MagicMock()
        self.app.client.conversations_info = AsyncMock(
            return_value={"channel": {"is_private": False, "is_member": True}})
        self.app.client.users_conversations = AsyncMock(
            return_value={"ok": True, "channels": [{"id": "C_PUBLIC", "name": "c_public"}]})
        self.app.client.conversations_history = AsyncMock()
        self.app.client.conversations_replies = AsyncMock()
        self.remote_calls = []
        self.app.client.users_info = _mock_api(remote_names, self.remote_calls).users_info
        self.user_cache = {}
        self.db = _mock_db(db_users)

    def log_debug(self, *a, **k): pass
    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
async def test_history_tool_resolves_authors_read_only():
    h = _HistHarness(db_users={"U1": {"username": "alice"}}, remote_names={"U2": "bob"})
    h.app.client.conversations_history.return_value = {"messages": [
        {"user": "U1", "ts": "1.1", "text": "bout time"},
        {"user": "U1", "ts": "1.2", "text": "again"},
        {"user": "U2", "ts": "1.3", "text": "hi"},
    ]}
    ctx = ToolContext(channel_id="C_PUBLIC", user_id="U_ASKER", requester_is_human=True)
    res = await h.fetch_history_tool("C_PUBLIC", ctx=ctx)
    assert [m["user"] for m in res["messages"]] == ["alice", "alice", "bob"]
    assert h.remote_calls == ["U2"]                    # U1 from DB; U2 once remotely; deduped
    h.db.get_or_create_user_async.assert_not_called()  # reading history creates no user rows
    h.db.save_user_info_async.assert_not_called()


@pytest.mark.asyncio
async def test_history_tool_unknown_author_falls_back_to_id():
    h = _HistHarness()  # no DB rows, no remote names -> nothing resolves
    h.app.client.conversations_history.return_value = {"messages": [
        {"user": "U404", "ts": "1.1", "text": "who dis"},
    ]}
    ctx = ToolContext(channel_id="C_PUBLIC", user_id="U_ASKER", requester_is_human=True)
    res = await h.fetch_history_tool("C_PUBLIC", ctx=ctx)
    assert res["messages"][0]["user"] == "U404"


class _SearchHarness(SlackSearchToolMixin, SlackUtilitiesMixin):
    def __init__(self, db_users=None, remote_names=None):
        self.app = MagicMock()
        self.remote_calls = []
        self.app.client.users_info = _mock_api(remote_names, self.remote_calls).users_info
        self.user_cache = {}
        self.db = _mock_db(db_users)

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info


@pytest.mark.asyncio
async def test_search_tool_resolves_authors_read_only():
    bot = _SearchHarness(db_users={"U9": {"username": "carol"}})
    bot.app.client.api_call = AsyncMock(return_value={"ok": True, "results": {"messages": [
        {"channel_id": "C09", "message_ts": "100.1", "author_user_id": "U9",
         "content": "we decided fridays", "permalink": "https://x/p1"},
        {"channel_id": "C09", "message_ts": "100.2", "author_user_id": "U9",
         "content": "still fridays", "permalink": "https://x/p2"},
    ]}})
    # DM surface: full reach, so this author-resolution test isn't touched by the
    # delivery-audience filter (exercised in test_channel_scope_guard.py).
    out = await bot.execute_search_tool(
        ToolContext(channel_id="C04", thread_ts="1.0", trigger_ts="1.0", action_token="tok",
                    is_dm=True),
        {"query": "demo day"})
    assert out["ok"] is True
    assert [r["author"] for r in out["results"]] == ["carol", "carol"]
    assert bot.remote_calls == []                       # served from DB, deduped
    bot.db.get_or_create_user_async.assert_not_called()  # searching creates no user rows


class _RebuildHarness(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Real get_thread_history + the real resolve_usernames against a mocked Slack client/DB."""
    def __init__(self, db_users=None, remote_names=None):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.remote_calls = []
        self.app.client.users_info = _mock_api(remote_names, self.remote_calls).users_info
        self.markdown_converter = MagicMock()
        self.user_cache = {}
        self.db = _mock_db(db_users)

    def log_info(self, *a, **k): pass
    log_debug = log_error = log_warning = log_info


@pytest.mark.asyncio
async def test_rebuild_resolves_author_and_prewarms_mentions():
    b = _RebuildHarness(db_users={"U2": {"username": "bob"}}, remote_names={"U1": "alice"})
    b.app.client.conversations_replies = AsyncMock(return_value={
        "messages": [{"ts": "1", "user": "U2", "text": "<@U1> bout time"}],
        "response_metadata": {},
    })
    result = await b.get_thread_history("C1", "1")
    msg = result[0]
    assert msg.metadata["username"] == "bob"    # author resolved from the read-only DB
    assert "@alice" in msg.text                  # mention resolved via the batched prewarm
    assert "U1" not in msg.text                  # raw id gone from the body
    assert b.remote_calls == ["U1"]              # only the uncached mention hit users.info
    b.db.get_or_create_user_async.assert_not_called()  # rebuild creates no user rows


@pytest.mark.asyncio
async def test_rebuild_orders_ids_root_then_newest_first():
    # Blocker 2 at a production call site: the ordered id list is root author first, then authors
    # newest→oldest — so when the budget bites, the root + recent speakers are the ones resolved.
    b = _RebuildHarness(remote_names={"U2": "root", "U3": "mid", "U4": "newest"})
    b.app.client.conversations_replies = AsyncMock(return_value={
        "messages": [                      # conversations.replies is ascending (root first)
            {"ts": "1", "user": "U2", "text": "root"},
            {"ts": "2", "user": "U3", "text": "mid"},
            {"ts": "3", "user": "U4", "text": "newest"},
        ],
        "response_metadata": {},
    })
    await b.get_thread_history("C1", "1")
    assert b.remote_calls == ["U2", "U4", "U3"]  # root, then newest→oldest


@pytest.mark.asyncio
async def test_rebuild_completes_with_raw_id_when_resolution_fails():
    # BF2 (c): a total resolution failure (DB + Slack both down) degrades to the raw id and
    # NEVER aborts the rebuild — Slack is the only transcript, so [] would be amnesia.
    b = _RebuildHarness()
    b.db.get_user_infos_async = AsyncMock(side_effect=RuntimeError("db down"))
    b.app.client.users_info = AsyncMock(side_effect=RuntimeError("slack down"))
    b.app.client.conversations_replies = AsyncMock(return_value={
        "messages": [{"ts": "1", "user": "U2", "text": "bout time"}],
        "response_metadata": {},
    })
    result = await b.get_thread_history("C1", "1")
    assert len(result) == 1                      # rebuild completed
    assert result[0].user_id == "U2"             # raw id preserved
    assert result[0].metadata["username"] is None  # unresolved -> downstream uses the id


@pytest.mark.asyncio
async def test_rebuild_always_sets_username_key():
    # Blocker 1 contract: get_thread_history ALWAYS emits the "username" KEY (human authors get
    # a name or None; bot/self get None), so the rebuild consumer treats its presence as proof
    # the batch resolve ran and never re-resolves per-message.
    b = _RebuildHarness(db_users={"U2": {"username": "bob"}})
    b.app.client.conversations_replies = AsyncMock(return_value={
        "messages": [
            {"ts": "1", "user": "U2", "text": "hi"},                       # human
            {"ts": "2", "bot_id": "B99", "username": "Webhook", "text": "beep"},  # other bot
        ],
        "response_metadata": {},
    })
    result = await b.get_thread_history("C1", "1")
    assert all("username" in m.metadata for m in result)   # key present on every message
    assert result[0].metadata["username"] == "bob"
    assert result[1].metadata["username"] is None          # bot author -> None, key still present


# ============================================================ BF3 — pulse envelope observability

def _pulse_entry(ts, text="hello", thread_ts=None, name="Alice", sender="human"):
    return dict(ts=ts, thread_ts=thread_ts, user_id="U1", display_name=name,
                sender_type=sender, text=text, is_bot=sender != "human")


def test_render_envelope_with_meta_counts_survivors(monkeypatch):
    from slack_client.channel_pulse import ChannelPulse
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    p = ChannelPulse(size=10)
    p.record("C1", **_pulse_entry("1.0"))
    p.record("C1", **_pulse_entry("2.0"))
    p.record("C1", **_pulse_entry("3.0", thread_ts="T"))  # excluded by exclude_thread_ts
    p.record("C1", **_pulse_entry("4.0"))
    p.record("C1", **_pulse_entry("5.0"))
    text, count, first_ts, last_ts = p.render_envelope_with_meta(
        "C1", exclude_thread_ts="T", max_lines=2)
    # span/count reflect exactly the entries that survive exclusion AND max_lines truncation
    assert count == 2
    assert (first_ts, last_ts) == ("4.0", "5.0")
    # timestamps are config-off, so the span could NOT have been parsed from the text
    assert "4.0" not in text and "5.0" not in text
    # the thin wrapper returns exactly the meta variant's text
    assert p.render_envelope("C1", exclude_thread_ts="T", max_lines=2) == text


def test_render_envelope_with_meta_empty_channel():
    from slack_client.channel_pulse import ChannelPulse
    assert ChannelPulse(size=5).render_envelope_with_meta("C1") == ("", 0, None, None)


def test_build_pulse_envelope_logs_span(monkeypatch):
    from message_processor.utilities import MessageUtilitiesMixin
    from slack_client.channel_pulse import ChannelPulse
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    monkeypatch.setattr(config, "channel_pulse_envelope_max", 5)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_pulse_entry("1.0"))
    pulse.record("C1", **_pulse_entry("2.0"))

    class _Host:
        def __init__(self):
            self._build_pulse_envelope = MessageUtilitiesMixin._build_pulse_envelope.__get__(self)
            self.logs = []

        def log_debug(self, msg, *a, **k):
            self.logs.append(msg)

    h = _Host()
    env = h._build_pulse_envelope(SimpleNamespace(channel_pulse=pulse), "C1", None)
    assert env  # non-empty envelope injected
    assert any("Pulse envelope injected: channel=C1 lines=2 span=1.0→2.0" in m for m in h.logs)
