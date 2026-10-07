"""Reconsideration request truth (STALE_RECONSIDERATION §4d): the no-tools assembly mode and
the mandated structured-decision wrapper.

Every surface that CLAIMS a capability has to describe the request actually sent: the horizon
reach text, the system instructions, the capability suffix and hash, the tool-schema digest —
all with MCP/canvas/image off and the model set to the one called — plus `registry=None`,
`contract_suffix=None` and `tools=[]` on the wire. The wrapper's schema, sampling pins and
`on_attempt_open` seam are pinned here too. The exact ordered item grammar of the final payload
belongs to the runner's integration tests, not this file.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import message_processor.prompts as prompts
from config import config
from message_processor import channel_request, channel_stream
from openai_client import base as openai_base
from openai_client.api import responses as responses_api

TEAM = "T1"
CH = "C1"


# ------------------------------------------------------------------ helpers


def _everything_on_config():
    from tests.unit.channel_turn_harness import thread_config

    cfg = thread_config(enable_web_search=True, enable_code_interpreter=True,
                        model="gpt-5.6-sol")
    cfg["enable_mcp"] = True
    cfg["enable_canvas_tools"] = True
    cfg["image_model"] = "gpt-image-2"
    return cfg


def _ctx(cfg):
    from tests.unit.channel_turn_harness import build_stream, normalized

    stream = build_stream([normalized("10.0")], channel_id=CH, team_id=TEAM)
    return channel_request.ChannelTurnContext(
        stream=stream, steering=SimpleNamespace(developer_policy=None, user_facts=None),
        thread_config=cfg, channel_id=CH, team_id=TEAM, trigger_ts="10.0",
        origin_thread_ts=None,
        requester=channel_request.RequesterFacts(user_id="U1", real_name="Alice",
                                                 sender_type="human"))


def _processor():
    processor = MagicMock()
    processor._get_system_prompt.return_value = "SYSTEM-PROMPT"
    processor._build_time_suffix_context.return_value = "[time: pinned]"
    processor._build_generation_inflight_note.return_value = None
    processor._build_research_inflight_note.return_value = None
    return processor


# ------------------------------------------------------------------ the normalized profile


def test_reconsideration_profile_sets_every_tool_field_off_and_names_the_called_model():
    cfg = _everything_on_config()
    profile = channel_request.reconsideration_profile(cfg, model="gpt-5.5")
    # Literal False, SET rather than deleted — absent keys fall back to process config.
    assert profile["enable_web_search"] is False
    assert profile["enable_code_interpreter"] is False
    assert profile["enable_mcp"] is False
    assert profile["enable_canvas_tools"] is False
    assert profile["image_model"] is None
    assert profile["model"] == "gpt-5.5"
    # Non-mutating: the pinned profile is untouched.
    assert cfg["enable_web_search"] is True and cfg["model"] == "gpt-5.6-sol"


def test_capability_hash_names_the_called_model_and_no_tool(monkeypatch):
    from message_processor.utilities import effective_request_model

    # With web search OFF the effective model is the called one even when WEB_SEARCH_MODEL
    # would reroute a tool-bearing turn.
    monkeypatch.setattr(config, "web_search_model", "gpt-5.6-terra")
    tools_on = _everything_on_config()
    profile = channel_request.reconsideration_profile(tools_on, model="gpt-5.5")
    assert effective_request_model(profile) == "gpt-5.5"

    # The hash describes THIS request: two originals differing only in their tool fields and
    # stored model normalize to the SAME hash, and it is not the original profile's hash.
    tools_off = dict(tools_on, enable_web_search=False, enable_code_interpreter=False,
                     enable_mcp=False, enable_canvas_tools=False, image_model=None,
                     model="gpt-5.6-luna")
    same = channel_request.reconsideration_profile(tools_off, model="gpt-5.5")
    assert (channel_request.capability_profile_hash(profile)
            == channel_request.capability_profile_hash(same))
    assert (channel_request.capability_profile_hash(profile)
            != channel_request.capability_profile_hash(tools_on))


def test_tool_schema_digest_claims_nothing_and_never_consults_mcp():
    class _ExplodingMCP:
        def get_server_labels(self):
            raise AssertionError("MCP manager consulted with enable_mcp=False")

    profile = channel_request.reconsideration_profile(_everything_on_config(), model="gpt-5.5")
    assert channel_request.hosted_tool_digest_entries(profile, _ExplodingMCP()) == []
    assert channel_request.tool_schema_version(None, profile,
                                               mcp_manager=_ExplodingMCP()) == ""


def test_snapshot_reach_default_is_empty_and_renders_no_reach_tool():
    # The pure stream build receives reach_tools=() (§4d) — the seam's default — and an empty
    # reach set renders NO reach clause, so the horizon claims no tool.
    seam = inspect.signature(channel_stream.build_reconsideration_snapshot)
    assert seam.parameters["reach_tools"].default == ()
    assert channel_stream.render_reach_clause(()) == ""
    horizon = channel_stream.render_horizon(floor_ts="", inventory_state="complete",
                                            reach_tools=())
    for name in prompts.REACH_TOOLS:
        assert name not in horizon


# ------------------------------------------------------------------ the no-tools assembly


def test_no_tools_assembly_claims_no_tool_on_every_surface():
    ctx = _ctx(_everything_on_config())
    processor = _processor()
    client = SimpleNamespace(bot_user_id="U_BOT")
    registry = MagicMock()  # would grow a catalog if consulted

    request = channel_request.assemble_channel_request(
        processor=processor, client=client, ctx=ctx, model="gpt-5.5",
        tools=[{"type": "web_search"}], request_config={"enable_web_search": True},
        contract_suffix="[CONTRACT]", registry=registry, no_tools=True)

    # The wire: no tool at all.
    assert request.tools == []
    # System instructions were built with every hosted tool off and no local registry.
    call = processor._get_system_prompt.call_args
    assert call.args[6] is False                                # enable_web_search
    assert call.kwargs["code_interpreter_enabled"] is False
    assert call.kwargs["tools_available"] is False              # registry=None
    assert call.kwargs["tools_structurally_withheld"] is True   # no settings claim either
    # The developer suffix: capability lines name the called model and claim no tool — not even
    # "off", which would read as the user's setting — and the tool contract never materializes.
    developer = request.input_items[-1]
    assert developer["role"] == "developer"
    assert "model: gpt-5.5" in developer["content"]
    assert "web search" not in developer["content"]
    assert "code interpreter" not in developer["content"]
    assert "[CONTRACT]" not in developer["content"]
    registry.schemas.assert_not_called()


def test_no_tools_assembly_equals_manual_normalization():
    """The mode IS the normalization: forcing the profile/registry/contract/tools by hand over
    the same context produces byte-identical items — except the one thing only the mode knows,
    that the tools are withheld by design, so its suffix states no web-search setting."""
    cfg = _everything_on_config()
    ctx = _ctx(cfg)
    processor = _processor()
    client = SimpleNamespace(bot_user_id="U_BOT")

    via_mode = channel_request.assemble_channel_request(
        processor=processor, client=client, ctx=ctx, model="gpt-5.5",
        tools=[{"type": "web_search"}], request_config=cfg,
        contract_suffix="[CONTRACT]", registry=MagicMock(), no_tools=True)

    profile = channel_request.reconsideration_profile(cfg, model="gpt-5.5")
    manual_ctx = replace(ctx, thread_config=profile)
    manual = channel_request.assemble_channel_request(
        processor=processor, client=client, ctx=manual_ctx, model="gpt-5.5",
        tools=[], request_config=profile, contract_suffix=None, registry=None)

    assert via_mode.input_items[:-1] == manual.input_items[:-1]
    off_line = next(line for line in manual.input_items[-1]["content"].split("\n")
                    if line.startswith("web search: off"))
    assert via_mode.input_items[-1] == dict(
        manual.input_items[-1],
        content=manual.input_items[-1]["content"].replace("\n" + off_line.rstrip("]"), ""))
    assert via_mode.instructions == manual.instructions
    assert via_mode.tools == manual.tools == []


def _real_prompt_processor():
    from message_processor.utilities import MessageUtilitiesMixin

    processor = _processor()
    processor._get_system_prompt = MessageUtilitiesMixin._get_system_prompt.__get__(processor)
    return processor


def _slack_client():
    return SimpleNamespace(name="Slack", bot_user_id="U_BOT", tool_registry=None)


def test_a_no_tools_request_states_no_web_search_setting_even_when_it_is_off():
    """A pass that offers no tool by design never claims a settings state. Live: the rethought
    reply told the user web search was off in their settings right after the draft had searched."""
    from tests.unit.channel_turn_harness import thread_config

    ctx = _ctx(thread_config(enable_web_search=False, model="gpt-5.6-sol"))
    request = channel_request.assemble_channel_request(
        processor=_real_prompt_processor(), client=_slack_client(), ctx=ctx, model="gpt-5.5",
        tools=[], request_config=None, contract_suffix=None, no_tools=True)

    suffix = request.input_items[-1]["content"]
    for text in (request.instructions, suffix):
        assert "disabled" not in text
        assert "web search: off" not in text
        assert config.settings_slash_command not in text


def test_a_real_web_search_off_setting_still_renders_the_user_settings_text():
    from tests.unit.channel_turn_harness import thread_config

    ctx = _ctx(thread_config(enable_web_search=False, model="gpt-5.6-sol"))
    request = channel_request.assemble_channel_request(
        processor=_real_prompt_processor(), client=_slack_client(), ctx=ctx, model="gpt-5.6-sol",
        tools=[], request_config=ctx.thread_config, contract_suffix=None)

    assert "Web search is currently disabled" in request.instructions
    assert config.settings_slash_command in request.instructions
    suffix = request.input_items[-1]["content"]
    assert "web search: off" in suffix and config.settings_slash_command in suffix


def test_the_reconsideration_item_names_the_tools_the_draft_already_used():
    from message_processor.reconsideration import reconsideration_item

    item = reconsideration_item(1, "the draft", "[Alice ts=10.0] hi",
                                tools_used=["web search", "fetch_url"])
    assert ("Tools already used while producing this draft: web search, fetch_url. "
            "No tools are offered in this pass.") in item["content"]


def test_the_reconsideration_item_names_newer_responders_only_when_given():
    from message_processor.reconsideration import reconsideration_item

    base = reconsideration_item(1, "the draft", "[Alice ts=10.0] hi", tools_used=["web search"])
    for empty in (None, []):
        assert reconsideration_item(1, "the draft", "[Alice ts=10.0] hi",
                                    tools_used=["web search"],
                                    newer_responder_ts=empty) == base
    item = reconsideration_item(1, "the draft", "[Alice ts=10.0] hi", tools_used=["web search"],
                                newer_responder_ts=["11.0", "12.0"])
    sentence = prompts.RECONSIDERATION_NEWER_RESPONDERS.format(timestamps="11.0, 12.0")
    content = item["content"]
    assert sentence in content
    assert (content.index("No tools are offered in this pass.") < content.index(sentence)
            < content.index("The unposted draft under evaluation"))


# ------------------------------------------------------------------ the counted body


def test_the_counted_body_carries_the_response_format_the_call_sends():
    """R3-6: the runner sizes a pass by counting `build_reconsideration_create_kwargs` — so the
    structured-output format the decision call sends is part of what is counted, and survives
    the counter's allowlist."""
    from message_processor.reconsideration import PreparedDecision
    from openai_client.api.token_count import count_body

    prepared = PreparedDecision(
        instructions="hi", api_items=[{"role": "user", "content": "x"}],
        params={"reasoning_effort": "medium", "verbosity": "medium",
                "max_output_tokens": 4000, "temperature": 1.0, "prompt_cache_key": "k"})
    kwargs = responses_api.build_reconsideration_create_kwargs(prepared, model="gpt-5.6-sol")
    assert kwargs["text"]["format"] == responses_api.STALE_RECONSIDERATION_RESPONSE_FORMAT
    assert count_body(kwargs)["text"]["format"] == (
        responses_api.STALE_RECONSIDERATION_RESPONSE_FORMAT)


# ------------------------------------------------------------------ the wrapper


class _FakeAPI:
    """A terminating, real-string mock of the OpenAI client surface the wrapper touches."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.timeouts = []
        self.warnings = []
        self.client = SimpleNamespace(responses=SimpleNamespace(create=self._create))

    async def _create(self, **params):
        self.calls.append(params)
        if self._error is not None:
            raise self._error
        return self._response

    async def _safe_api_call(self, api_method, *args, timeout_seconds=None,
                             operation_type="general", **kwargs):
        self.timeouts.append(timeout_seconds)
        return await api_method(*args, **kwargs)

    def log_debug(self, *args, **kwargs):
        pass

    def log_info(self, *args, **kwargs):
        pass

    def log_error(self, *args, **kwargs):
        pass

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))


def _response(payload=None, *, parts=None, status="completed", incomplete_reason=None):
    if parts is None:
        parts = [SimpleNamespace(type="output_text", text=json.dumps(payload))]
    return SimpleNamespace(
        status=status,
        incomplete_details=(SimpleNamespace(reason=incomplete_reason)
                            if incomplete_reason else None),
        output=[SimpleNamespace(type="message", content=parts)],
        usage=None)


async def _decide(fake, **overrides):
    kwargs = dict(input_items=[{"role": "user", "content": "stream"}],
                  instructions="INSTR", model="gpt-5.6-sol", reasoning_effort="medium",
                  verbosity="low", max_output_tokens=512, temperature=0.7,
                  prompt_cache_key="chan:T1:C1")
    kwargs.update(overrides)
    return await responses_api.create_reconsideration_decision(fake, **kwargs)


async def test_decision_null_text_and_whitespace_text_are_both_null():
    fake = _FakeAPI(response=_response({"decision": "post", "text": None}))
    decision = await _decide(fake)
    assert (decision.decision, decision.text) == ("post", None)

    fake = _FakeAPI(response=_response({"decision": "force_post", "text": "  \n\t "}))
    decision = await _decide(fake)
    assert (decision.decision, decision.text) == ("force_post", None)


async def test_decision_revised_text_survives_unstripped():
    fake = _FakeAPI(response=_response({"decision": "post", "text": "revised answer\n"}))
    decision = await _decide(fake)
    assert decision.text == "revised answer\n"


async def test_skip_with_text_drops_the_text_and_logs_the_anomaly_without_it():
    fake = _FakeAPI(response=_response({"decision": "skip", "text": "stray draft body"}))
    decision = await _decide(fake)
    assert (decision.decision, decision.text) == ("skip", None)
    assert any("skip" in warning for warning in fake.warnings)
    assert all("stray draft body" not in warning for warning in fake.warnings)


async def test_extra_properties_are_schema_invalid():
    fake = _FakeAPI(response=_response(
        {"decision": "post", "text": None, "reviewed_through": "1.0"}))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "schema_invalid"


async def test_junk_wrapped_json_is_schema_invalid_never_brace_hunted():
    """The ENTIRE stripped structured output must be the JSON document — surrounding prose is
    never salvaged by extracting from first '{' to last '}'."""
    fake = _FakeAPI(response=_response(parts=[SimpleNamespace(
        type="output_text",
        text='junk {"decision": "post", "text": null} junk')]))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "schema_invalid"


@pytest.mark.parametrize("bad_text", [12345, {"decision": "post", "text": None},
                                      0, False, [], {}])
async def test_a_non_string_output_text_is_schema_invalid_never_a_typeerror(bad_text):
    """A numeric or object `output_text.text` is schema-invalid model output — it must be
    classified by the wrapper, not escape as a TypeError the runner would misfile."""
    fake = _FakeAPI(response=_response(parts=[SimpleNamespace(type="output_text",
                                                              text=bad_text)]))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "schema_invalid"


async def test_refusal_is_a_model_failure():
    fake = _FakeAPI(response=_response(
        parts=[SimpleNamespace(type="refusal", refusal="cannot decide")]))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "refusal"


async def test_a_refusal_dominates_parseable_text_beside_it():
    """A refusal part wins even when a valid decision rides in the same output — the model's
    stated unwillingness is never overridden by text that also happens to parse."""
    fake = _FakeAPI(response=_response(parts=[
        SimpleNamespace(type="refusal", refusal="cannot decide"),
        SimpleNamespace(type="output_text",
                        text=json.dumps({"decision": "post", "text": None}))]))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "refusal"


async def test_only_output_text_content_is_parsed():
    """A non-structured content type carrying a `text` attribute is not the model's decision:
    with nothing else in the output the call fails rather than parsing it."""
    fake = _FakeAPI(response=_response(parts=[SimpleNamespace(
        type="reasoning_text",
        text=json.dumps({"decision": "post", "text": None}))]))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail in ("schema_invalid", "empty")


async def test_incomplete_is_a_model_failure():
    fake = _FakeAPI(response=_response({"decision": "post", "text": None},
                                       status="incomplete",
                                       incomplete_reason="max_output_tokens"))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "incomplete"


@pytest.mark.parametrize("status", [None, "in_progress"])
async def test_a_missing_or_unknown_status_is_rejected(status):
    """Only 'completed' proceeds — an absent status is not success, and an unrecognized one
    is not waved through either."""
    fake = _FakeAPI(response=_response({"decision": "post", "text": None}, status=status))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide(fake)
    assert exc.value.detail == "incomplete"


# ------------------------------------------------------------------ sampling pins


async def test_reasoning_effort_pins_temperature_to_one_and_sends_the_mandated_envelope():
    fake = _FakeAPI(response=_response({"decision": "skip", "text": None}))
    await _decide(fake, reasoning_effort="high", temperature=0.3)
    params = fake.calls[0]
    assert params["temperature"] == 1.0
    assert "top_p" not in params
    assert params["store"] is False
    assert params["tools"] == []
    assert params["reasoning"] == {"effort": "high"}
    assert params["max_output_tokens"] == 512
    assert params["prompt_cache_key"] == "chan:T1:C1"
    assert params["instructions"] == "INSTR"
    assert params["text"]["verbosity"] == "low"
    assert params["text"]["format"] == {
        "type": "json_schema",
        "name": "stale_reconsideration_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "text"],
            "properties": {"decision": {"enum": ["post", "force_post", "skip"]},
                           "text": {"type": ["string", "null"]}},
        },
    }
    assert fake.timeouts == [config.api_timeout_read]


async def test_effort_none_passes_pinned_temperature_and_builder_resolved_top_p():
    fake = _FakeAPI(response=_response({"decision": "post", "text": None}))
    await _decide(fake, reasoning_effort="none", temperature=0.4)
    params = fake.calls[0]
    assert params["temperature"] == 0.4
    assert params["top_p"] == config.default_top_p
    assert params["reasoning"] == {"effort": "none"}
    assert params["store"] is False


# ------------------------------------------------------------------ the attempt-open seam


class _Sink:
    def __init__(self, attempt):
        self._attempt = attempt
        self.opened = []
        self.closed = []

    def open(self, model=None):
        self.opened.append(model)
        return self._attempt

    def close(self, attempt, *, status, input_tokens=None, output_tokens=None,
              cached_input_tokens=None, detail=None):
        self.closed.append((attempt, status, detail))


async def test_on_attempt_open_gets_the_seq_after_open_and_before_the_request():
    attempt = SimpleNamespace(attempt_seq=7, fork_reason=None)
    sink = _Sink(attempt)
    fake = _FakeAPI(response=_response({"decision": "post", "text": None}))
    seen = []

    def callback(seq):
        # Invoked AFTER open() (the sink already recorded it) and BEFORE the request.
        assert sink.opened and fake.calls == []
        seen.append(seq)

    decision = await _decide(fake, attempt_sink=sink, on_attempt_open=callback)
    assert seen == [7]
    assert decision.decision == "post"


async def test_open_returning_none_hands_the_callback_none_and_the_call_proceeds():
    class _BrokenSink(_Sink):
        def open(self, model=None):
            self.opened.append(model)
            return None

    sink = _BrokenSink(None)
    fake = _FakeAPI(response=_response({"decision": "skip", "text": None}))
    seen = []
    decision = await _decide(fake, attempt_sink=sink, on_attempt_open=seen.append)
    assert seen == [None]
    assert decision.decision == "skip"
    assert fake.calls


async def test_a_raising_callback_never_blocks_the_call():
    fake = _FakeAPI(response=_response({"decision": "post", "text": None}))

    def callback(seq):
        raise RuntimeError("telemetry exploded")

    decision = await _decide(fake, on_attempt_open=callback)
    assert decision.decision == "post"
    assert fake.calls
    assert any("on_attempt_open" in warning for warning in fake.warnings)


# ------------------------------------------------------------------ base.py exposure


def test_wrapper_is_exposed_through_openai_base():
    assert hasattr(openai_base.OpenAIClient, "create_reconsideration_decision")
    assert openai_base.ReconsiderationDecision is responses_api.ReconsiderationDecision
    assert (openai_base.ReconsiderationDecisionError
            is responses_api.ReconsiderationDecisionError)
    assert (openai_base.STALE_RECONSIDERATION_RESPONSE_FORMAT
            is responses_api.STALE_RECONSIDERATION_RESPONSE_FORMAT)
    assert (openai_base.STALE_RECONSIDERATION_DECISION_SCHEMA
            is responses_api.STALE_RECONSIDERATION_DECISION_SCHEMA)


# ------------------------------------------------------------------ burst follow-ups: redo + tooled


def _prepared(**mode):
    from message_processor.reconsideration import PreparedDecision

    return PreparedDecision(
        instructions="INSTR", api_items=[{"role": "user", "content": "stream"}],
        params={"reasoning_effort": "medium", "verbosity": "low", "max_output_tokens": 512,
                "temperature": 0.7, "prompt_cache_key": "chan:T1:C1"}, **mode)


async def _decide_via_base(fake, prepared):
    """Through the OpenAIClient wrapper (codex R1 #6), with the prepared pass's own mode."""
    return await openai_base.OpenAIClient.create_reconsideration_decision(
        fake, input_items=prepared.api_items, instructions=prepared.instructions,
        model="gpt-5.6-sol", allow_redo=prepared.allow_redo, tools=prepared.tools,
        **prepared.params)


@pytest.mark.parametrize("mode", [{}, {"allow_redo": True},
                                  {"tools": [{"type": "web_search"}]}])
async def test_the_count_and_the_create_send_the_same_body_in_every_mode(mode):
    """B8 / R2-3: one prepared object feeds both — the counted kwargs ARE the sent kwargs."""
    prepared = _prepared(**mode)
    payload = ({"decision": "skip", "text": None, "interim": None} if mode.get("allow_redo")
               else {"decision": "skip", "text": None})
    fake = _FakeAPI(response=_response(payload))
    await _decide_via_base(fake, prepared)
    counted = responses_api.build_reconsideration_create_kwargs(prepared, model="gpt-5.6-sol")
    assert fake.calls[0] == counted


async def test_a_request_that_does_not_offer_redo_is_byte_identical_to_today():
    legacy = await _decide(_FakeAPI(response=_response({"decision": "skip", "text": None})))
    assert legacy.interim is None and legacy.tools_used == ()
    fake = _FakeAPI(response=_response({"decision": "skip", "text": None}))
    await _decide_via_base(fake, _prepared())
    assert fake.calls[0]["text"]["format"] == responses_api.STALE_RECONSIDERATION_RESPONSE_FORMAT
    assert fake.calls[0]["tools"] == [] and "include" not in fake.calls[0]


async def test_an_offered_redo_sends_the_redo_schema_and_keeps_its_interim():
    fake = _FakeAPI(response=_response(
        {"decision": "redo", "text": "stale words", "interim": "Re-checking with Kyiv too."}))
    decision = await _decide_via_base(fake, _prepared(allow_redo=True))
    assert fake.calls[0]["text"]["format"] == (
        responses_api.STALE_RECONSIDERATION_REDO_RESPONSE_FORMAT)
    assert fake.calls[0]["text"]["format"]["schema"]["properties"]["decision"]["enum"] == [
        "post", "force_post", "skip", "redo"]
    assert (decision.decision, decision.text) == ("redo", None)     # text on redo dropped
    assert decision.interim == "Re-checking with Kyiv too."


async def test_an_interim_on_any_other_decision_is_dropped():
    fake = _FakeAPI(response=_response(
        {"decision": "post", "text": "revised", "interim": "should not post"}))
    decision = await _decide_via_base(fake, _prepared(allow_redo=True))
    assert (decision.decision, decision.text, decision.interim) == ("post", "revised", None)
    assert any("interim" in warning for warning in fake.warnings)


@pytest.mark.parametrize("payload,allow_redo", [
    ({"decision": "redo", "text": None}, False),                    # redo never offered
    ({"decision": "redo", "text": None, "interim": None}, False),   # wrong key set too
    ({"decision": "post", "text": None}, True),                     # offered mode needs interim
])
async def test_the_parse_requires_the_modes_exact_shape(payload, allow_redo):
    fake = _FakeAPI(response=_response(payload))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide_via_base(fake, _prepared(allow_redo=allow_redo))
    assert exc.value.detail == "schema_invalid"


async def test_the_tooled_pass_sends_the_hosted_tools_and_harvests_what_completed():
    output = [
        SimpleNamespace(type="web_search_call", status="completed",
                        action=SimpleNamespace(type="search", query="kyiv weather now",
                                               sources=None)),
        SimpleNamespace(type="web_search_call", status="failed", action=None),
        SimpleNamespace(type="mcp_call", status="completed", server_label="menus",
                        error=None, output="42 items"),
        SimpleNamespace(type="mcp_call", status="completed", server_label="broken",
                        error="boom", output=None),
        SimpleNamespace(type="message", content=[SimpleNamespace(
            type="output_text", annotations=[],
            text=json.dumps({"decision": "post", "text": "Kyiv 9°C, London 14°C."}))]),
    ]
    fake = _FakeAPI(response=SimpleNamespace(status="completed", incomplete_details=None,
                                             output=output, usage=None))
    tools = [{"type": "web_search"}, {"type": "mcp", "server_label": "menus"}]
    decision = await _decide_via_base(fake, _prepared(tools=tools))

    params = fake.calls[0]
    assert params["tools"] == tools
    assert params["include"] == ["web_search_call.results"]
    assert params["text"]["format"] == responses_api.STALE_RECONSIDERATION_RESPONSE_FORMAT
    assert (decision.decision, decision.text) == ("post", "Kyiv 9°C, London 14°C.")
    assert decision.tools_used == ("web_search", "menus")
    assert {r["tool_name"] for r in decision.tool_results} == {"web_search", "menus"}


@pytest.mark.parametrize("text", [None, "   "])
async def test_a_tooled_post_without_text_is_schema_invalid(text):
    """R2-4: there is no draft to fall back to."""
    fake = _FakeAPI(response=_response({"decision": "post", "text": text}))
    with pytest.raises(responses_api.ReconsiderationDecisionError) as exc:
        await _decide_via_base(fake, _prepared(tools=[{"type": "web_search"}]))
    assert exc.value.detail == "schema_invalid"


def test_the_redo_explanation_rides_the_item_only_when_offered():
    from message_processor.reconsideration import reconsideration_item

    base = reconsideration_item(1, "the draft", "[Alice ts=10.0] hi")
    assert reconsideration_item(1, "the draft", "[Alice ts=10.0] hi", allow_redo=False) == base
    offered = reconsideration_item(1, "the draft", "[Alice ts=10.0] hi", allow_redo=True)
    assert prompts.RECONSIDERATION_REDO in offered["content"]
    assert prompts.RECONSIDERATION_REDO not in base["content"]
    # B7: every tool-free pass carries the honesty rule.
    assert "never re-attach the draft's findings" in base["content"]


def test_hosted_only_assembly_offers_exactly_the_hosted_tools():
    """R2-3: web search as the user set it and the given hosted list on the wire; no sandbox,
    no canvases, no local tools, no tool contract."""
    ctx = _ctx(_everything_on_config())
    processor = _processor()
    registry = MagicMock()
    hosted = [{"type": "web_search"}, {"type": "mcp", "server_label": "menus"}]
    request = channel_request.assemble_channel_request(
        processor=processor, client=SimpleNamespace(bot_user_id="U_BOT"), ctx=ctx,
        model="gpt-5.5", tools=hosted, request_config={"enable_web_search": True},
        contract_suffix="[CONTRACT]", registry=registry, tool_mode="hosted_only")

    assert request.tools == hosted
    call = processor._get_system_prompt.call_args
    assert call.args[6] is True                                 # web search, as set
    assert call.kwargs["code_interpreter_enabled"] is False
    assert call.kwargs["tools_available"] is False
    assert call.kwargs["tools_structurally_withheld"] is False
    developer = request.input_items[-1]["content"]
    assert "model: gpt-5.5" in developer
    assert "web search: available" in developer
    assert "code interpreter" not in developer
    assert "[CONTRACT]" not in developer
    registry.schemas.assert_not_called()
