"""The channel tool surface: one tools array per channel, whoever is asking (spec §3a).

The thing being protected is the cached prefix. Tools are part of it, so a schema that varies
with the requester, the thread, or what happens to be in a catalog forks the cache for every
turn after it — and the fork is invisible: the request succeeds, it just costs full price.

So the channel surface answers three questions differently from the DM surface:

* the schema is a function of (channel, channel config, bot version) and nothing else;
* the gates are channel-config-stable predicates, and per-turn `enabled=` callables are
  structurally ignored;
* everything the schemas used to carry — the ids, the saved defaults, the live emoji cache —
  arrives as untrusted evidence in the request body instead.

DM turns keep the old dynamic surface verbatim, so both are asserted side by side.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor import canvas_tools, file_mount, image_tools
from message_processor.handlers.text import (TOOL_EVIDENCE_HEADER,
                                             TOOL_EVIDENCE_MAX_CHARS,
                                             TOOL_EVIDENCE_TRUNCATED,
                                             TextHandlerMixin,
                                             build_tool_evidence_block)
from tool_registry import SURFACE_CHANNEL, SURFACE_DM, ToolContext, ToolRegistry


# --------------------------------------------------------------------------- fixtures

def _slack_host():
    """A SlackBot stand-in whose schema getters return REAL dicts (a bare MagicMock is
    callable, and register() reads a callable as a per-request factory)."""
    from slack_client.messaging import SlackMessagingMixin
    s = MagicMock()
    s.get_history_tools_for_openai.return_value = [
        {"type": "function", "name": "fetch_thread_messages", "parameters": {}}]
    for getter, name in (("get_search_tool_schema", "search_slack"),
                         ("get_post_to_thread_tool_schema", "post_to_thread"),
                         ("get_edit_own_message_tool_schema", "edit_own_message"),
                         ("get_no_reply_tool_schema", "no_response_needed"),
                         ("get_emoji_search_tool_schema", "search_workspace_emoji"),
                         ("get_lookup_channel_tool_schema", "lookup_channel"),
                         ("get_resolve_channel_name_tool_schema", "resolve_channel_name")):
        getattr(s, getter).return_value = {"type": "function", "name": name, "parameters": {}}
    # W3: search_slack now registers a CHANNEL schema too (only that surface returns thread_ts),
    # and register() reads a bare MagicMock as a factory returning a mock — which would leave the
    # channel surface exposing a tool with a MagicMock for a name.
    s.get_search_tool_channel_schema = lambda cfg=None: {
        "type": "function", "name": "search_slack", "parameters": {}}
    # react_to_message is the one tool whose two variants genuinely differ, so both are real.
    s.workspace_emojis = SimpleNamespace(get_custom_emoji_names=lambda: ["shipit"])
    s._custom_emoji_available = SlackMessagingMixin._custom_emoji_available.__get__(s)
    s._react_tool_schema = SlackMessagingMixin._react_tool_schema.__get__(s)
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    s.get_react_tool_schema_static = SlackMessagingMixin.get_react_tool_schema_static.__get__(s)
    return s


def _registry():
    from slack_client.base import SlackBot
    return SlackBot._build_tool_registry(_slack_host())


def _tools_host(mcp_tools=()):
    host = SimpleNamespace()
    host._build_tools_array = TextHandlerMixin._build_tools_array.__get__(host)
    host._as_mcp_exclusion_set = TextHandlerMixin._as_mcp_exclusion_set
    host.log_debug = lambda *a, **k: None
    host.log_info = lambda *a, **k: None
    host.mcp_manager = SimpleNamespace(
        has_mcp_servers=lambda: bool(mcp_tools),
        get_tools_for_openai=lambda: [dict(t) for t in mcp_tools])
    return host


class _FakeDB:
    def __init__(self, prefs=None, channel_settings=None):
        self._prefs = prefs or {}
        self._channel = channel_settings

    def get_user_preferences(self, user_id):
        return self._prefs.get(user_id)

    def get_channel_settings(self, channel_id):
        return self._channel


def _channel_config(*, user_id="U1", prefs=None, overrides=None, channel_settings=None,
                    catalogs=True, container="cntr_thread_a", extras=None):
    """A channel turn's request_config, built the way the handler builds it."""
    cfg = config.get_thread_config(overrides=overrides or {}, user_id=user_id,
                                   db=_FakeDB(prefs or {}, channel_settings),
                                   channel_id="C1", channel_turn=True)
    cfg[image_tools.CI_CONTAINER_KEY] = container
    cfg[image_tools.CATALOG_KEY] = ([{"image_id": "img_1", "kind": "upload",
                                      "description": "a chart"}] if catalogs else [])
    cfg[file_mount.FILES_KEY] = ([{"file_id": "fil_1", "filename": "q3.csv",
                                   "mime_type": "text/csv", "size_bytes": 2048,
                                   "description": "quarterly numbers"}] if catalogs else [])
    cfg[canvas_tools.CATALOG_KEY] = ([{"canvas_id": "F1", "title": "Agenda",
                                       "is_channel_canvas": True}] if catalogs else [])
    cfg.update(extras or {})
    return cfg


def _names(schemas):
    return [s["name"] for s in schemas]


# ---------------------------------------------------- the matrix: one array per channel

def test_the_channel_tools_array_is_the_same_for_every_thread_and_every_requester():
    """Two origin threads × two requesters × catalog present/absent. The code-interpreter
    container is the ONE documented fork (§3a); everything else must be identical, byte for
    byte, or the channel's cached prefix is per-person."""
    registry = _registry()
    host = _tools_host()
    model = "gpt-5.6-sol"

    matrix = {}
    for thread, container in (("A", "cntr_a"), ("B", "cntr_b")):
        for requester, prefs in (("U1", {"U1": {"model": "gpt-5.5",
                                                "reasoning_effort": "low",
                                                "image_size": "1024x1024"}}),
                                 ("U2", {"U2": {"model": "gpt-5.6-luna",
                                                "verbosity": "high",
                                                "image_size": "auto"}})):
            for catalogs in (True, False):
                cfg = _channel_config(
                    user_id=requester, prefs=prefs, catalogs=catalogs, container=container,
                    # Per-turn facts that USED to move schemas around. On this surface they
                    # must move nothing at all.
                    extras={"_silence_capable_turn": (thread == "A"),
                            "_destination_choice_open": (requester == "U1"),
                            "_slack_search_available": catalogs,
                            "_canvas_delete_authorized": (thread == "B")},
                    overrides={"model": "gpt-5.5"} if thread == "B" else None)
                matrix[(thread, requester, catalogs)] = host._build_tools_array(
                    cfg, model, registry=registry, ci_container=container,
                    surface=SURFACE_CHANNEL)

    def _without_container(tools):
        return [t for t in tools if t.get("type") != "code_interpreter"]

    reference = matrix[("A", "U1", True)]
    for key, tools in matrix.items():
        assert _without_container(tools) == _without_container(reference), key

    containers = {t["container"] for tools in matrix.values() for t in tools
                  if t.get("type") == "code_interpreter"}
    assert containers == {"cntr_a", "cntr_b"}   # the fork exists, and it is the only one


def test_the_channel_capability_profile_ignores_the_requester_and_the_thread():
    """§3b: the capability keys leave the hierarchy on a channel turn. Two people with opposite
    settings, and a thread override on top, still run the same machine."""
    a = _channel_config(user_id="U1", prefs={"U1": {"model": "gpt-5.5",
                                                    "enable_web_search": False}})
    b = _channel_config(user_id="U2", prefs={"U2": {"model": "gpt-5.6-luna",
                                                    "enable_web_search": True}},
                        overrides={"model": "gpt-5.5", "enable_mcp": True})
    for key in ("model", "reasoning_effort", "verbosity", "enable_web_search",
                "enable_mcp", "image_model", "enable_code_interpreter"):
        assert a[key] == b[key], key


# ---------------------------------------------------------- the four documented cache forks

def test_fork_one_the_code_interpreter_container_is_thread_scoped(monkeypatch):
    monkeypatch.setattr(config, "enable_code_interpreter", True)
    host = _tools_host()
    cfg = _channel_config()
    one = host._build_tools_array(cfg, "gpt-5.6-sol", ci_container="cntr_a",
                                  surface=SURFACE_CHANNEL)
    two = host._build_tools_array(cfg, "gpt-5.6-sol", ci_container="cntr_b",
                                  surface=SURFACE_CHANNEL)
    assert one != two
    assert [t for t in one if t.get("type") == "code_interpreter"][0]["container"] == "cntr_a"


def test_fork_two_an_mcp_failure_retry_drops_the_failed_server(monkeypatch):
    monkeypatch.setattr(config, "mcp_enabled_default", True)
    host = _tools_host(mcp_tools=[{"type": "mcp", "server_label": "alpha"},
                                  {"type": "mcp", "server_label": "beta"}])
    cfg = _channel_config()
    cfg["enable_mcp"] = True
    full = host._build_tools_array(cfg, "gpt-5.6-sol", surface=SURFACE_CHANNEL)
    retry = host._build_tools_array(cfg, "gpt-5.6-sol", exclude_mcp_server="alpha",
                                    surface=SURFACE_CHANNEL)
    def labels(tools):
        return {t.get("server_label") for t in tools if t.get("type") == "mcp"}

    assert labels(full) == {"alpha", "beta"}
    assert labels(retry) == {"beta"}


def test_fork_three_the_timeout_retry_drops_the_local_tools_entirely(mock_env):
    """`tools_disabled` nulls the registry, so the retry request carries no local schemas —
    and the contract paragraphs go with them."""
    host = _materialize_host(_registry())
    _, _, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"},
        _message(silence_capable=True), tools_disabled=True, surface=SURFACE_CHANNEL)
    assert available is False and suffix is None


def test_fork_four_the_model_fallback_changes_the_mcp_eligibility(monkeypatch):
    """MCP rides only on gpt-5*, so a fallback to another family legitimately changes the
    array. Documented, and the only model-driven difference."""
    monkeypatch.setattr(config, "mcp_enabled_default", True)
    host = _tools_host(mcp_tools=[{"type": "mcp", "server_label": "alpha"}])
    cfg = _channel_config()
    cfg["enable_mcp"] = True
    assert any(t.get("type") == "mcp"
               for t in host._build_tools_array(cfg, "gpt-5.6-sol", surface=SURFACE_CHANNEL))
    assert not any(t.get("type") == "mcp"
                   for t in host._build_tools_array(cfg, "o4-mini", surface=SURFACE_CHANNEL))


# ------------------------------------------------------------- the DM surface is untouched

def test_the_dm_surface_is_still_dynamic():
    """The factories still fire there, and the ids still ride as enums — the whole point of
    keeping two surfaces is that DM turns are byte-identical to before."""
    registry = _registry()
    cfg = _channel_config(catalogs=True)
    dm = {s["name"]: s for s in registry.schemas(cfg, surface=SURFACE_DM)}
    assert "edit_image" in dm
    enum = dm["edit_image"]["parameters"]["properties"]["source_image_ids"]["items"]["enum"]
    assert enum == ["img_1"]
    assert dm["mount_file"]["parameters"]["properties"]["file_id"]["enum"] == ["fil_1"]

    channel = {s["name"]: s for s in registry.schemas(cfg, surface=SURFACE_CHANNEL)}
    assert "enum" not in channel["edit_image"]["parameters"]["properties"]["source_image_ids"]["items"]
    assert "enum" not in channel["mount_file"]["parameters"]["properties"]["file_id"]


def test_an_empty_catalog_removes_a_dm_tool_but_never_a_channel_one():
    """The DM factories hide themselves when there is nothing to name. On the channel surface
    the tool stays and the executor answers honestly — a tool set that depends on a catalog is
    a tool set that changes mid-conversation."""
    registry = _registry()
    empty = _channel_config(catalogs=False)
    assert "edit_image" not in _names(registry.schemas(empty, surface=SURFACE_DM))
    assert "view_image" not in _names(registry.schemas(empty, surface=SURFACE_DM))
    assert "mount_file" not in _names(registry.schemas(empty, surface=SURFACE_DM))
    channel = _names(registry.schemas(empty, surface=SURFACE_CHANNEL))
    assert {"edit_image", "view_image", "mount_file"} <= set(channel)


def test_the_react_description_follows_config_not_the_live_emoji_cache():
    """The cache warms after start and can empty on an API failure. On the DM surface that is
    fine; on the channel surface it would rewrite a cached prefix under a running process."""
    from slack_client.messaging import SlackMessagingMixin
    host = _slack_host()
    warm = host.get_react_tool_schema_static()
    host.workspace_emojis = SimpleNamespace(get_custom_emoji_names=lambda: [])
    host._custom_emoji_available = SlackMessagingMixin._custom_emoji_available.__get__(host)
    cold = host.get_react_tool_schema_static()
    assert warm == cold
    assert "search_workspace_emoji" in warm["parameters"]["properties"]["emoji"]["description"]
    # ...while the DM factory still tracks the cache, exactly as it did.
    assert "search_workspace_emoji" not in (
        host.get_react_tool_schema()["parameters"]["properties"]["emoji"]["description"])


def test_a_configured_allowlist_still_constrains_the_channel_schema(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", ":eyes:"])
    schema = _slack_host().get_react_tool_schema_static()
    assert schema["parameters"]["properties"]["emoji"]["enum"] == ["thumbsup", "eyes"]


# ------------------------------------------------------------------- gates and registration

def test_per_turn_gates_are_structurally_ignored_on_the_channel_surface():
    reg = ToolRegistry()
    reg.register({"type": "function", "name": "gated", "parameters": {}},
                 AsyncMock(), enabled=lambda cfg: cfg.get("_per_turn", False))
    assert reg.schemas({}, surface=SURFACE_DM) == []
    assert _names(reg.schemas({}, surface=SURFACE_CHANNEL)) == ["gated"]
    # A channel gate, when declared, is honoured — it just may only read stable facts.
    reg.register({"type": "function", "name": "off", "parameters": {}},
                 AsyncMock(), channel_enabled=lambda cfg: False)
    assert "off" not in _names(reg.schemas({}, surface=SURFACE_CHANNEL))


def test_has_tools_answers_per_surface():
    reg = ToolRegistry()
    reg.register({"type": "function", "name": "gated", "parameters": {}},
                 AsyncMock(), enabled=lambda cfg: cfg.get("_per_turn", False))
    assert reg.has_tools({}) is False
    assert reg.has_tools({}, surface=SURFACE_CHANNEL) is True
    assert reg.has_tools({"_per_turn": True}) is True


def test_a_callable_schema_must_declare_itself_dynamic():
    """Per-request shape is a DM privilege now. An undeclared factory would reintroduce it
    silently, which is exactly how the cache forks got in."""
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="dynamic=True"):
        reg.register(lambda cfg: {"name": "x"}, AsyncMock(), name="x")
    reg.register(lambda cfg: {"type": "function", "name": "x", "parameters": {}},
                 AsyncMock(), name="x", dynamic=True)
    assert _names(reg.schemas({})) == ["x"]


def test_a_factory_that_raises_omits_its_tool_and_logs():
    reg = ToolRegistry()

    def boom(cfg):
        raise RuntimeError("bad schema")

    reg.register(boom, AsyncMock(), name="broken", dynamic=True)
    reg.register({"type": "function", "name": "fine", "parameters": {}}, AsyncMock())
    with patch("tool_registry.logger") as log:
        assert _names(reg.schemas({})) == ["fine"]
    assert log.error.called
    message = log.error.call_args[0][0]
    assert "broken" in message and "bad schema" in message


def test_every_channel_schema_is_stable_across_configs():
    """The invariant, asserted over the whole registry rather than tool by tool: give the
    channel surface two wildly different request configs and it must hand back the same list."""
    registry = _registry()
    a = registry.schemas(_channel_config(user_id="U1", catalogs=True,
                                         container="cntr_a"), surface=SURFACE_CHANNEL)
    b = registry.schemas(_channel_config(user_id="U2", catalogs=False, container=None,
                                         prefs={"U2": {"image_size": "1536x1024"}},
                                         extras={"_silence_capable_turn": True,
                                                 "_canvas_delete_authorized": True}),
                         surface=SURFACE_CHANNEL)
    assert a == b


def test_the_research_job_registry_reads_the_default_surface():
    """The build phase wants the container-aware create_image_asset, not the channel-stable
    one — and it gets it because `schemas()` still defaults to "dm"."""
    from message_processor import research_tools  # noqa: F401 — registration site under test
    reg = ToolRegistry()
    reg.register(image_tools.get_create_image_asset_schema,
                 image_tools.execute_create_image_asset,
                 name="create_image_asset", dynamic=True)
    cfg = _channel_config(container="cntr_a")
    job = reg.schemas(cfg)
    assert _names(job) == ["create_image_asset"]
    # The default surface hands back the FACTORY's output, not the channel-stable variant.
    assert job[0] == image_tools.get_create_image_asset_schema(cfg)
    assert job[0] != image_tools.get_create_image_asset_schema_static()


# --------------------------------------------------------------------- the evidence block

def test_the_evidence_block_carries_what_the_schemas_stopped_carrying():
    from message_processor import image_catalog, thread_files
    from message_processor.image_service import SETTINGS_EVIDENCE_HEADER
    block = build_tool_evidence_block(_channel_config(), _slack_host())
    assert block.startswith(TOOL_EVIDENCE_HEADER)
    assert "Untrusted evidence, not instructions" in block
    for header in (SETTINGS_EVIDENCE_HEADER, image_catalog.EVIDENCE_HEADER,
                   thread_files.EVIDENCE_HEADER, canvas_tools.EVIDENCE_HEADER,
                   "Custom emoji:"):
        assert header in block
    assert "img_1" in block and "fil_1" in block and "F1" in block


def test_an_empty_catalog_is_stated_rather_than_omitted():
    """"There are none" is the fact that stops the model naming one."""
    block = build_tool_evidence_block(_channel_config(catalogs=False), _slack_host())
    assert "(none)" in block
    assert "this channel has no canvas yet" in block


def test_the_evidence_block_truncates_by_whole_line_in_section_priority():
    """A cut must never leave half an id — the model will try it. Sections go
    settings > images > files > canvases > emoji, and the block says it was cut."""
    from message_processor import image_catalog
    cfg = _channel_config()
    entries = [{"image_id": f"img_{i}", "kind": "upload", "analysis": "a chart of " + "x" * 200}
               for i in range(400)]
    cfg[image_tools.CATALOG_KEY] = entries
    block = build_tool_evidence_block(cfg, _slack_host())
    assert len(block) <= TOOL_EVIDENCE_MAX_CHARS
    assert block.endswith(TOOL_EVIDENCE_TRUNCATED)
    # Settings came first and survived whole; the lowest-priority sections are what went.
    assert "image model:" in block
    assert "Custom emoji:" not in block
    # Every catalog line that made it is a WHOLE source line, never a prefix of one.
    source = set(image_catalog.catalog_evidence_lines(entries))
    kept = [ln for ln in block.split("\n") if ln.startswith("img_")]
    assert kept and all(ln in source for ln in kept)


def test_the_evidence_block_never_fails_the_turn_on_a_broken_emoji_cache():
    class _Boom:
        def _custom_emoji_available(self):
            raise RuntimeError("cache is gone")

    block = build_tool_evidence_block(_channel_config(), _Boom())
    assert "no custom emoji are reachable here" in block


@pytest.mark.asyncio
async def test_the_evidence_block_rides_the_channel_request_after_the_breakpoint():
    """It is part of the request, at USER role, BELOW the cache breakpoint.

    It used to have to fit inside the trim budget — a channel turn had a trim. It does not any
    more: the request IS the pinned window, admitted whole or refused. What survives is the part
    that always mattered — these catalogs are a snapshot of what the tools can address, so they
    are content, they are volatile, and they must sit after the prefix everyone shares."""
    from tests.unit.channel_turn_harness import item_texts, pin_channel_turn
    from message_processor.channel_request import to_input_items
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.turn_runtime import TurnRuntime

    host = MagicMock()
    host._assemble_channel_attempt = TextHandlerMixin._assemble_channel_attempt.__get__(host)
    host._channel_prepared_tools = TextHandlerMixin._channel_prepared_tools.__get__(host)
    host._build_tools_array = MagicMock(return_value=None)
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_time_suffix_context = MagicMock(return_value="[time]")
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    host._build_message_with_documents = MagicMock(side_effect=lambda t, d: t)

    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=(_registry(), _channel_config(), False, None, None))
    message = SimpleNamespace(channel_id="C1", thread_id="10.0", user_id="U1", text="hi",
                              attachments=None, metadata={"ts": "10.0"})
    request, *_ = await host._assemble_channel_attempt(
        _slack_host(), message, SimpleNamespace(), turn,
        {"model": "gpt-5.6-sol"}, "gpt-5.6-sol", thread_key="C1:10.0")

    items = to_input_items(request)
    evidence = [i for i in items
                if isinstance(i.get("content"), str)
                and i["content"].startswith(TOOL_EVIDENCE_HEADER)]
    assert len(evidence) == 1
    assert evidence[0]["role"] == "user"        # untrusted evidence, never developer voice
    # BELOW the breakpoint: the end marker carries it, and the evidence comes after.
    marker = next(n for n, i in enumerate(items) if _has_breakpoint(i))
    assert items.index(evidence[0]) > marker
    # The channel cache key is the CHANNEL's, so two requesters share one prefix.
    assert request.prompt_cache_key == "chan:T1:C1"
    assert "[time]" in item_texts(items)[-1]     # ...and the developer suffix is last


def _has_breakpoint(item):
    content = item.get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("prompt_cache_breakpoint") for p in content)


@pytest.mark.asyncio
async def test_a_dm_turn_builds_no_evidence_block_and_resolves_per_user():
    from base_client import Message
    from message_processor.handlers.text import TextHandlerMixin

    seen = {}
    host = MagicMock()
    host._handle_text_response = TextHandlerMixin._handle_text_response.__get__(host)
    host._turn_surface = TextHandlerMixin._turn_surface
    host.db = None
    host.mcp_manager = MagicMock()
    host._is_reaction_only = MagicMock(return_value=False)

    async def _trim(messages, *a, **k):
        seen["messages"] = list(messages)
        return list(messages)

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return ""

    host._pre_trim_messages_for_api = _trim
    host._inject_image_analyses = _passthru
    host._build_channel_info = _empty
    host._build_channel_summary_block = _none
    host._drop_dead_containers = _none
    host._resolve_ci_container = _none
    host._prepare_sandbox_tools = _none
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_suffix_context = MagicMock(return_value="")
    host._build_participant_roster = MagicMock(return_value="")
    host._build_tools_array = MagicMock(return_value=None)
    host._materialize_request_tools = MagicMock(return_value=(None, {"model": "m"}, False, None))
    host._add_message_with_token_management = MagicMock()
    host.openai_client = MagicMock()
    host.openai_client.create_text_response = AsyncMock(return_value="hi")

    message = Message(text="hi", user_id="U1", channel_id="D1", thread_id="10.0",
                      metadata={"ts": "10.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id="D1", thread_ts="10.0",
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None)

    async def fake_config(**kw):
        seen.setdefault("channel_turn", []).append(kw.get("channel_turn"))
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": False, "enable_code_interpreter": False}

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        await host._handle_text_response("hi", thread_state, MagicMock(), message,
                                         thinking_id=None)

    assert not [m for m in seen["messages"]
                if isinstance(m.get("content"), str)
                and m["content"].startswith(TOOL_EVIDENCE_HEADER)]
    assert not any(seen["channel_turn"])
    # …and the DM keeps the legacy request layout.
    layout = host.openai_client.create_text_response.await_args.kwargs["layout"]
    assert layout == "legacy"


def test_the_surface_ruling_maps_ids_to_surfaces():
    surface = TextHandlerMixin._turn_surface
    for cid in ("D123", "U123", "W123"):
        assert surface(SimpleNamespace(channel_id=cid)) == SURFACE_DM
    for cid in ("C123", "G123"):
        assert surface(SimpleNamespace(channel_id=cid)) == SURFACE_CHANNEL


# ------------------------------------------------ the executors that took over from schemas

def _message(silence_capable=False, sender_type="human", mentioned_self=False,
             channel_id="C1"):
    return SimpleNamespace(
        metadata={"ts": "1.1", "silence_capable": silence_capable,
                  "routing_posture": "channel_activity", "sender_type": sender_type,
                  "mentioned_self": mentioned_self, "gate_required": False},
        channel_id=channel_id, thread_id="1.1", user_id="U1")


def _materialize_host(registry):
    host = SimpleNamespace()
    for name in ("_materialize_request_tools", "_get_tool_registry", "_build_tool_context"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host))
    host.db = None
    host._client = SimpleNamespace(tool_registry=registry)
    return host


def test_the_canvas_delete_authorization_reaches_the_tool_context(mock_env):
    """A3a moved the check into the executor, which left it refusing everywhere until the
    context carried the fact. This is the wire."""
    host = _materialize_host(_registry())
    addressed = _message(mentioned_self=True)
    _, request_config, _, _ = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, addressed, tools_disabled=False,
        surface=SURFACE_CHANNEL)
    assert request_config["_canvas_delete_authorized"] is True
    ctx = host._build_tool_context(addressed, MagicMock(), request_config)
    assert ctx.canvas_delete_authorized is True

    overheard = _message(mentioned_self=False)
    _, cfg2, _, _ = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, overheard, tools_disabled=False,
        surface=SURFACE_CHANNEL)
    assert host._build_tool_context(overheard, MagicMock(), cfg2).canvas_delete_authorized is False


def test_a_context_that_never_derived_the_fact_cannot_delete():
    """Fail-closed by construction: the dataclass default is False, so a hand-built or
    replayed context is refused without anyone remembering to check."""
    assert ToolContext().canvas_delete_authorized is False


@pytest.mark.asyncio
async def test_delete_canvas_runs_again_once_the_context_carries_the_authorization():
    from message_processor.canvas_tools import execute_delete_canvas
    unauthorized = ToolContext(channel_id="C1", message=SimpleNamespace(
        metadata={"sender_type": "human"}))
    refused = await execute_delete_canvas(unauthorized, {"canvas_id": "F1"})
    assert refused["ok"] is False and refused["error"] == "not_authorized"

    authorized = ToolContext(channel_id="C1", canvas_delete_authorized=True,
                             message=SimpleNamespace(metadata={"sender_type": "human"}))
    out = await execute_delete_canvas(authorized, {"canvas_id": "F1"})
    # It gets past authorization now and fails on the NEXT gate (this canvas isn't here),
    # which is the proof the interim block is gone.
    assert out["error"] != "not_authorized"


def test_the_channel_surface_exposes_the_static_terminal_and_destination_tools(mock_env):
    """Both used to be per-turn schema gates. The routes they guarded are per-turn facts, so
    the schemas went static and the executors took the decision (see
    tests/unit/test_terminal_actions.py for the loop contract)."""
    registry = _registry()
    quiet_turn = _channel_config(extras={"_silence_capable_turn": True,
                                         "_destination_choice_open": True})
    owed_turn = _channel_config()
    for cfg in (quiet_turn, owed_turn):
        names = _names(registry.schemas(cfg, surface=SURFACE_CHANNEL))
        assert "no_response_needed" in names
        assert "set_reply_destination" in names
        assert "search_slack" in names
    # DM keeps the gates.
    assert "no_response_needed" not in _names(registry.schemas(owed_turn, surface=SURFACE_DM))
    assert "no_response_needed" in _names(registry.schemas(quiet_turn, surface=SURFACE_DM))


def test_the_contract_paragraph_still_follows_the_route_not_the_schema(mock_env):
    """The tool is statically present on every channel turn now, so schema presence can no
    longer decide who is told they may stay quiet — the routing fact does."""
    from prompts import CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    host = _materialize_host(_registry())
    _, _, available, suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, _message(silence_capable=True),
        tools_disabled=False, surface=SURFACE_CHANNEL)
    assert available is True and suffix == CHANNEL_ACTIVITY_NO_REPLY_SUFFIX

    _, _, owed_available, owed_suffix = host._materialize_request_tools(
        host._client, {"model": "gpt-5.6-sol"}, _message(silence_capable=False),
        tools_disabled=False, surface=SURFACE_CHANNEL)
    assert owed_available is False and owed_suffix is None


@pytest.mark.asyncio
async def test_the_timeout_retry_carries_no_evidence_block_and_advertises_no_tools():
    """That attempt drops the local tools entirely. Evidence about tools that are not on the
    table is just tokens on a request that already timed out once — and the PROMPT has to agree:
    it used to ask the client-wide registry rather than the attempt, so it still taught the
    local-tool and canvas etiquette for tools absent from its own request."""
    from tests.unit.channel_turn_harness import no_tools_prepared, pin_channel_turn
    from message_processor.channel_request import to_input_items
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.turn_runtime import TurnRuntime

    host = MagicMock()
    host._assemble_channel_attempt = TextHandlerMixin._assemble_channel_attempt.__get__(host)
    host._channel_prepared_tools = TextHandlerMixin._channel_prepared_tools.__get__(host)
    host._prepare_channel_turn_tools = AsyncMock(
        return_value=no_tools_prepared(_channel_config()))
    host._build_tools_array = MagicMock(return_value=None)
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_time_suffix_context = MagicMock(return_value="[time]")
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    host._build_message_with_documents = MagicMock(side_effect=lambda t, d: t)

    turn = TurnRuntime()
    # No `prepared` pin: the retry resolves its OWN exposure, which is the point.
    pin_channel_turn(turn)
    message = SimpleNamespace(channel_id="C1", thread_id="10.0", user_id="U1", text="hi",
                              attachments=None, metadata={"ts": "10.0"})
    request, *_ = await host._assemble_channel_attempt(
        _slack_host(), message, SimpleNamespace(), turn,
        {"model": "gpt-5.6-sol"}, "gpt-5.6-sol", thread_key="C1:10.0", tools_disabled=True)

    assert not [i for i in to_input_items(request)
                if isinstance(i.get("content"), str)
                and i["content"].startswith(TOOL_EVIDENCE_HEADER)]
    # The prompt was told what THIS attempt sends: no registry, so no tool etiquette.
    assert host._get_system_prompt.call_args.kwargs["tools_available"] is False
    # …and the retry never reuses the turn's pinned exposure, which still has its tools.
    assert turn.channel_prepared is None


# ============================== T51 — the cache key, and the four forks (§2c)

def test_the_cache_key_stays_channel_scoped_and_forks_only_four_ways(monkeypatch, mock_env):
    """T51. The key is the CHANNEL's, and exactly four documented things fork the prefix.

    Inspecting two keys proves nothing on its own — two calls to a pure function of
    `(team_id, channel_id)` were always going to agree. What makes the claim mean something is
    DRIVING each of the four documented forks and showing no fifth arises from a channel turn.
    """
    from message_processor.channel_request import prompt_cache_key
    from message_processor.channel_stream import OriginFetch, build_origin_pin, serialize_stream
    from tests.unit.channel_turn_harness import build_stream, normalized

    # THE KEY IS THE CHANNEL'S, for both origins of one periphery — the T16 shape. Two threads
    # in one channel share a cache entry; that is the whole point of the stable prefix.
    assert prompt_cache_key("T1", "C1") == prompt_cache_key("T1", "C1")
    assert prompt_cache_key("T1", "C1") != prompt_cache_key("T1", "C2")
    assert prompt_cache_key("T2", "C1") != prompt_cache_key("T1", "C1")

    room = [normalized("1.0", "room chatter"), normalized("2.0", "more", sender_id="U2")]
    a = build_stream(room, origin_root_ts="10.0",
                     origin_messages=[normalized("10.0", "thread A")])
    b = build_stream(room, origin_root_ts="20.0",
                     origin_messages=[normalized("20.0", "thread B")])
    # Same channel, two origins: ONE key, and the pre-breakpoint bytes are identical while the
    # unions differ. A key that varied by origin would give every thread its own cache entry.
    assert prompt_cache_key("T1", "C1") == prompt_cache_key("T1", "C1")
    assert a.stream_sha256 == b.stream_sha256
    assert a.union_sha256 != b.union_sha256
    assert OriginFetch and build_origin_pin and serialize_stream  # the split-phase API is real

    # THE FOUR FORKS, each DRIVEN rather than described.
    test_fork_one_the_code_interpreter_container_is_thread_scoped(monkeypatch)
    test_fork_two_an_mcp_failure_retry_drops_the_failed_server(monkeypatch)
    test_fork_three_the_timeout_retry_drops_the_local_tools_entirely(mock_env)
    test_fork_four_the_model_fallback_changes_the_mcp_eligibility(monkeypatch)

    # AND NO FIFTH. Vary the requester, the thread and the per-user prefs — everything a channel
    # turn can differ by that is NOT one of the four — and the tools array is byte-identical.
    registry = _registry()
    host = _tools_host()
    reference = None
    for requester, thread in (("U1", "A"), ("U2", "B"), ("U3", "C")):
        cfg = _channel_config(user_id=requester, container="cntr_same")
        tools = host._build_tools_array(cfg, "gpt-5.6-sol", registry=registry,
                                        ci_container="cntr_same", surface=SURFACE_CHANNEL)
        if reference is None:
            reference = tools
        assert tools == reference, f"{requester}/{thread} forked the prefix on its own"


def test_a_profile_change_moves_the_capability_hash_once():
    """T107 (respec §6.2). The capability pin already derives from `CHANNEL_CAPABILITY_KEYS`, so
    the three new channel columns enter it for free: changing one gives the channel a cold prefix
    for exactly one turn, and changing nothing leaves the hash where it was.

    Both halves matter. A hash blind to the new columns would let a channel that just turned web
    search off keep serving the cached prefix of a bot that still had it; a hash that moved on
    every turn would forfeit the cache the pin exists to protect."""
    from message_processor.channel_request import capability_profile_hash

    stored = {"model": "gpt-5.6-terra", "enable_web_search": 1}
    first = capability_profile_hash(_channel_config(channel_settings=dict(stored)))
    # Same profile, a different requester and a different thread: still one machine, one hash.
    assert first == capability_profile_hash(
        _channel_config(user_id="U2", channel_settings=dict(stored),
                        container="cntr_thread_b"))

    for changed in ({"enable_web_search": 0}, {"enable_mcp": 0}, {"image_model": "gpt-image-1"}):
        moved = capability_profile_hash(
            _channel_config(channel_settings={**stored, **changed}))
        assert moved != first, changed
