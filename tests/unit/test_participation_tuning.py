"""Participation tuning + 3 bug fixes (2026-07-21).

Covers the surviving prompt-wording contracts (C1 mid-flight escape, C2 truthfulness sentence),
the C1 real-event composition, and two bug fixes: BF1 (search_slack gated on the event's
action_token) and BF2 (username resolution in rebuilt history and tool-returned histories). BF3
(pulse envelope observability) covered `render_envelope_with_meta` / `_build_pulse_envelope`,
both retired with `slack_client/channel_pulse.py` — the channel stream replaced the pulse ring
entirely, so that section is gone with no successor to re-point it at.

MOST OF THE PROMPT SECTION IS GONE, and it went with the prompt it described. Four tests here
(A1 value floor, A1's direct-summons exemption, A2's open-question rule, B's banter reversal) each
pinned sentences of PARTICIPATION_SYSTEM_PROMPT — the rich gate's staged
addressee/exchange-state/answerability rubric. The binary gate decides one bit and does not weigh
whether a reply is worth making, so there is no successor sentence to re-point them at; asserting
one would be inventing a contract. What DID survive is the part that was never about wording but
about levers: the clauses below must not come back. The positive contract for the new prompt lives
in tests/unit/test_wake_classifier.py.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from prompts import (
    LOCAL_TOOLS_GUIDANCE,
    SLACK_SYSTEM_PROMPT,
    WAKE_CLASSIFIER_SYSTEM_PROMPT,
    CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
)
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.search_tool import SlackSearchToolMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.utilities import SlackUtilitiesMixin
from tool_registry import ToolContext, ToolRegistry


# =========================================================== prompt wording (C1/C2)


def test_c1_mid_flight_escape_present():
    """C1 survived the P2 §9 rewrite. That rewrite recast the whole paragraph around full channel
    visibility, and this escape is orthogonal to it — how honest a real answer has to be, not
    whether this turn is yours. The rest of the paragraph's contract lives in
    tests/unit/test_channel_restraint_prompts.py."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "being addressed by name outranks this whole test" in s
    assert s.endswith("]")  # still one bracketed paragraph
    # ...and it coexists with the full-visibility framing rather than having replaced it.
    assert "The stream is the room, not an invitation." in s


def test_c2_truthfulness_sentence_present():
    s = SLACK_SYSTEM_PROMPT
    assert "don't fake familiarity" in s
    assert "a confident wrong guess reads far worse than either" in s


def test_the_voice_paragraph_keeps_the_tone_boundary():
    """The tuning wave's tone rule, pinned as CONTENT rather than as a paragraph.

    Nothing else can catch its deletion. The banter scenarios grade whether the bot SPEAKS, and a
    reply that makes a coworker the punchline scores exactly like a kind one — the 2026-08-03
    incident reply would have passed every row in the corpus. So the boundary is pinned here, on
    its load-bearing nouns rather than its sentences, which leaves the wording free to be edited
    and still fails if the rule is dropped. Live review stays the behavioural authority.
    """
    voice = SLACK_SYSTEM_PROMPT.split("\n\n")[1]
    assert voice.startswith("Voice:")
    assert "warmth" in voice and "cruelty" in voice
    for target in ("competence", "character", "vulnerability"):
        assert target in voice, f"the tone boundary no longer names {target}"
    assert "at their own expense" in voice, "the self-deprecation half of the rule is gone"
    # …and the boundary did not swallow what it qualifies: teasing aimed at the bot still gets a
    # beat back, which `banter-aimed-at-self` measures at the behavioural level.
    assert "give it back in kind" in voice and "comeback" in voice


def test_the_emoji_bullet_keeps_the_shared_moment_guidance():
    """Same argument, the reactions ruling. `team-welcome` is a measure row BY DESIGN — the form a
    welcome takes is the model's call — so deleting this guidance fails nothing in the corpus.

    Pinned to the emoji bullet itself, not to the file: guidance about reactions living in some
    other bullet is not the contract, and the wording is generic on purpose (no occasion list).
    """
    bullet = next(line for line in LOCAL_TOOLS_GUIDANCE.splitlines()
                  if line.startswith("- Emoji reactions:"))
    for needle in ("marking a moment", "fitting reaction", "warm line", "not both by default"):
        assert needle in bullet, f"the reaction guidance no longer says {needle!r}"


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
        enabled=lambda cfg: config.enable_no_reply_tool and bool(cfg.get("_silence_capable_turn")))
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
    # The real bare-name hit stamps a gate-routed, silence-capable turn together with
    # participation_name_hit=True and wake_source="name_mention" (message_events ~709/717).
    # That composition still receives the F2 suffix + no_response_needed, so the C1 mid-flight
    # escape reaches it.
    host = _MatHost(_registry_no_reply_and_search())
    msg = _msg(gate_required=True, silence_capable=True,
               participation_name_hit=True, wake_source="name_mention")
    registry, request_config, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, msg, tools_disabled=False)
    assert available is True
    assert suffix == CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    names = {s["name"] for s in registry.schemas(request_config)}
    assert "no_response_needed" in names


# ========================================================== BF1 — search_slack action_token gate

def test_materialize_sets_slack_search_available_from_action_token(mock_env):
    host = _MatHost(_registry_no_reply_and_search())
    _, cfg_on, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"},
        _msg(gate_required=True, silence_capable=True, action_token="tok"), False)
    assert cfg_on["_slack_search_available"] is True

    _, cfg_off, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(gate_required=True, silence_capable=True), False)
    assert cfg_off["_slack_search_available"] is False


def test_search_schema_present_only_with_action_token(mock_env):
    host = _MatHost(_registry_no_reply_and_search())
    reg_on, cfg_on, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"},
        _msg(gate_required=True, silence_capable=True, action_token="tok"), False)
    assert "search_slack" in {s["name"] for s in reg_on.schemas(cfg_on)}

    reg_off, cfg_off, _, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(gate_required=True, silence_capable=True), False)
    assert "search_slack" not in {s["name"] for s in reg_off.schemas(cfg_off)}


@pytest.mark.asyncio
async def test_search_tool_runtime_still_refuses_without_token():
    # Defense in depth ON THE DM SURFACE: even if a schema slips through, the assistant-context
    # executor refuses tokenless calls. The gate is a DM fact now — a CHANNEL turn runs the
    # bot-token in-channel scan, which needs no action_token and must NOT refuse without one
    # (CHANNEL_SEARCH_REBUILD §S9), so the surface is stated here rather than defaulted.
    class _Bot(SlackSearchToolMixin):
        def __init__(self):
            self.app = MagicMock()

        def log_info(self, *a, **k): pass
        log_debug = log_warning = log_error = log_info

    out = await _Bot().execute_search_tool(
        ToolContext(action_token=None, is_dm=True), {"query": "x"})
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
    s.get_edit_own_message_tool_schema.return_value = {
        "type": "function", "name": "edit_own_message", "parameters": {}}
    s.get_pin_message_tool_schema.return_value = {
        "type": "function", "name": "pin_message", "parameters": {}}
    s.get_no_reply_tool_schema.return_value = {
        "type": "function", "name": "no_response_needed", "parameters": {}}
    s.get_emoji_search_tool_schema.return_value = {
        "type": "function", "name": "search_workspace_emoji", "parameters": {}}
    s.get_lookup_channel_tool_schema.return_value = {
        "type": "function", "name": "lookup_channel", "parameters": {}}
    s.get_resolve_channel_name_tool_schema.return_value = {
        "type": "function", "name": "resolve_channel_name", "parameters": {}}
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
    ("gated_channel", {"gate_required": True, "silence_capable": True}, "C1", False),
    ("thread_continuation", {"silence_capable": True,
                             "wake_source": "thread_continuation"}, "C1", False),
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
    # NEWEST-first, the order conversations.history really returns; the tool inverts it so the
    # model reads the channel in discourse order.
    h.app.client.conversations_history.return_value = {"messages": [
        {"user": "U2", "ts": "1.3", "text": "hi"},
        {"user": "U1", "ts": "1.2", "text": "again"},
        {"user": "U1", "ts": "1.1", "text": "bout time"},
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


@pytest.mark.asyncio
async def test_name_resolution_budget_favours_the_newest_speakers():
    """resolve_usernames spends a bounded number of remote lookups and leaves the overflow as raw
    ids, so the budget has to go to the most RECENT speakers. The 2026-07 grounding fix made the
    result oldest-first, which — if author collection had simply followed display order — would
    have handed the budget to the oldest authors and left the newest messages showing raw Slack
    ids. Author order is decoupled from display order for exactly that reason."""
    h = _HistHarness(remote_names={"U_OLD": "olderton", "U_NEW": "newman"})
    h.app.client.conversations_history.return_value = {"messages": [
        {"user": "U_NEW", "ts": "9.0", "text": "most recent"},     # newest-first, as Slack returns
        {"user": "U_OLD", "ts": "1.0", "text": "ancient"},
    ]}
    ctx = ToolContext(channel_id="C_PUBLIC", user_id="U_ASKER", requester_is_human=True)
    res = await h.fetch_history_tool("C_PUBLIC", ctx=ctx)
    # Display order is oldest-first...
    assert [m["text"] for m in res["messages"]] == ["ancient", "most recent"]
    # ...while the newest speaker is the FIRST id offered to the budgeted resolver.
    assert h.remote_calls[0] == "U_NEW"


def test_no_special_case_and_no_reply_lever_returns_to_the_gate_prompt():
    """The reverted levers — one still retired, one deliberately brought back NARROWED.

    Two separate reversals, both about the same mistake — teaching the gate to WANT an outcome:

    * the room-humour lean. The owner asked for a slight bias toward reacting to funny posts;
      measured live it fired on 5 of 5 jokes and the verdict was that the bot reacts "the same way
      over and over". A per-topic special case is the wrong lever, and the reference implementation
      has no humour clause at all. This one is STILL retired, and still asserted as an absence.
    * the banter licence. "Playful teasing aimed AT the assistant is participation-worthy... an
      invitation to play along" is why the bot answered "Chatgpt, you are right!" 52 seconds after
      being told to hush. Removing it took that scenario from 0/6 to passing.

    OWNER REVERSAL, 2026-08-11: the second lever is back, and this test records it rather than
    guarding against it. The owner ruled that banter about the assistant is partly the assistant's
    own moment and should wake it. What returned is NOT the old licence: the old clause told the
    gate that teasing was "participation-worthy" and "an invitation to play along" — an appetite
    for an outcome, which is what produced the 52-second incident. The new criterion is one-sided
    and evidence-bound: it wakes only when the shown messages make it THIS assistant, it says in
    the same breath that the woken turn may be a reaction or nothing at all, and it explicitly
    sleeps through general-AI/other-bot banter and ambiguous "the bot". The closing doubt rule
    carries the same carve-out, so the two do not contradict.

    The 52-second scenario itself is not re-litigated here; a hush is an instruction the responder
    holds, and the gate's job is still one bit. The emoji-vs-nothing preference the humour lean
    qualified remains the RESPONDER's to make, which is why nothing positive is asserted about it.
    """
    p = WAKE_CLASSIFIER_SYSTEM_PROMPT
    for lever in ("Room-wide humour", "lean slightly toward reacting", "prefer nothing"):
        assert lever not in p, f"a retired participation lever is back in the gate prompt: {lever}"
    # The narrowed successor to the banter licence, as it now reads. Asserted positively: absence
    # of assistant-directed wake language is no longer the contract.
    assert "the room is talking about the assistant ITSELF" in p
    assert "This means THIS assistant, unmistakably, from what the messages themselves show" in p
    assert "it may still answer with just a reaction, or nothing" in p
    assert ("banter that merely mentions bots or AI in general, or some other bot, with nothing "
            "asked of anyone, stays ordinary banter and sleeps") in p
    assert ("leave it alone; only banter unmistakably about this assistant itself wakes it, "
            "as above") in p
