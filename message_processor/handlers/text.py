from __future__ import annotations

import asyncio
import re
from typing import Any, List, Optional

from base_client import BaseClient, Message, Response
from config import config, pipeline_status
from prompts import NO_REPLY_CONTRACT_SUFFIX, CONTINUATION_NO_REPLY_SUFFIX
from message_markers import (
    CONTINUATION_HEAD,
    continuation_trailer,
    entity_safe_cut,
    part_prefix,
    segment_separator,
)
from streaming import FenceHandler, NativeStreamCoordinator, RateLimitManager, StreamingBuffer
from tool_registry import ToolContext
from message_processor import canvas_tools, file_mount, image_catalog, image_tools, thread_files
from message_processor.artifacts import (collect_container_ids, stream_safe_text, strip_citation_markers,
                                         strip_sandbox_links)
from message_processor.containers import AUTO_CONTAINER
from message_processor.tool_provenance import (strip_provenance_echo,
                                               visible_attribution_tools)
from openai_client.container_errors import is_container_gone, persistent_container_ids
from message_processor.tool_provenance import (
    build_provenance,
    build_result_digests,
    build_result_digests_summarized,
    render_provenance_annotations,
    strip_used_tools_footer,
)


def _delivered_stream_ts(native_coord, native_finalized: bool,
                         current_message_id: Optional[str],
                         content_delivered: bool) -> Optional[str]:
    """The ACTUAL delivered message ts for a streamed reply — the key F5/F7 must use.

    NOT the original placeholder `message_id` (None on native status-only streams, a
    DELETED placeholder on native fallback). Native path (finalize confirmed): the native
    stream's current ts. Legacy/fallback path: the final `current_message_id` — but ONLY
    when ``content_delivered`` is True. A placeholder/current id can exist even though
    EVERY content flush failed; returning it then would fake a delivery (phantom
    posted=True + pulse/provenance). Returns None when nothing visible actually landed."""
    if native_coord is not None and native_finalized:
        return native_coord.current_ts
    return current_message_id if content_delivered else None


def _legacy_fallback_target(overflow: Optional[str], native_current_ts: Optional[str],
                            current_message_id: Optional[str]) -> Optional[str]:
    """F35: after the native sink fails mid-stream, decide which message id the LEGACY
    fallback continues editing.

    A failed ROLL (``overflow`` is not None) means the current native part already received
    its first portion and was closed/abandoned, while the buffer now holds ONLY the remainder.
    Editing that finished part with the remainder would overwrite its first ~3000 chars, so
    return None to force the legacy path to SEED A NEW continuation message.

    A non-roll inert failure (``overflow`` is None) leaves the current native message as the
    live surface, so keep editing it (its ts, falling back to the existing id) — nothing
    visible is lost."""
    if overflow is not None:
        return None
    return native_current_ts or current_message_id


def native_stream_place_in_channel(message: Message) -> bool:
    """Whether the native-stream coordinator should target the channel top level
    (thread_ts None) instead of the thread — this must MATCH where main.py will
    actually post the reply, or chat.startStream is created against the wrong target.

    main.py stamps its final placement decision (which honors the participation
    engine's per-message placement verdict) into metadata["place_in_channel"]; that
    stamp WINS. Recomputing from the channel's reply_in_channel setting alone ignored
    a "thread" verdict on a top-level trigger: the reply threaded, but the coordinator
    was built with thread=None, so native streaming silently fell back to legacy (and
    the F8 attached footer never rode — the reply grew a separate footer message).
    The recompute survives only as a fallback for paths that bypass the stamp."""
    meta = message.metadata or {}
    if "place_in_channel" in meta:
        return bool(meta.get("place_in_channel"))
    is_top_level_trigger = meta.get("ts") == message.thread_id
    return (
        bool(meta.get("reply_in_channel")) and is_top_level_trigger
        and bool(message.channel_id) and not message.channel_id.startswith("D")
    )


# F38: hosted tools whose "started" event means real, slow work is now under way — the point
# at which a 👀 is honest. Everything else the callback fires for is deliberately absent:
# `mcp` discovery (`discovering_tools`) is plumbing that runs before the model has decided to
# call anything, and `local:*` events fire the instant a call is DISPATCHED — before its
# arguments are validated, before a duplicate background job is rejected — so claiming there
# would flash an eye on a call that never happened. Slow local tools claim from inside their
# own executors, once they know they are really going to do the work.
#
# KNOWN GAP, and it is structural rather than an oversight: these events only exist while
# STREAMING. A non-streaming turn resolves its hosted tools server-side inside one response,
# emitting nothing to react to, so a non-streaming web search claims no 👀. Local slow tools
# still claim on both paths (their executors run in both loops). Streaming is the default, and
# the alternative — polling, or claiming optimistically before the tools run — would put the
# eye back on a guess, which is the thing we just removed.
_WORK_CLAIM_HOSTED_TOOLS = frozenset({
    "web_search", "file_search", "code_interpreter", "image_generation",
})

# Local tools whose round emits a real pre-tool preamble ("Making that…") that must not freeze.
# Native appends are token-driven and the wrapper skips the None completion signal on a tool
# round, so a buffered preamble sits frozen — at whatever the cadence last flushed — until the
# NEXT round streams. edit_image/create_image_asset block the loop ~a minute; generate_image is
# detached, but its dispatch still does synchronous Slack work (stakes 👀, posts the status card)
# and the round boundary strands the preamble's tail across it either way. All three push the
# preamble before dispatch. `start_background_job` is deliberately EXCLUDED: it withholds its
# short ack so the job's live status card owns the acknowledgment (F30.1, keyed on
# visible_content_delivered staying False) — flushing there would strand both the ack and the card.
_PRE_TOOL_FLUSH_TOOLS = frozenset({
    "local:edit_image", "local:create_image_asset", "local:generate_image",
})


def _claims_work(tool_type: str, status: str) -> bool:
    """Does this tool event mean the bot has committed to real work?"""
    if status == "started" and tool_type in _WORK_CLAIM_HOSTED_TOOLS:
        return True
    # An MCP call signals "calling" (its "discovering_tools" phase is not a call at all).
    return status == "calling" and (tool_type == "mcp" or tool_type.startswith("mcp:"))


def _hosted_or_mcp_used(tools_used) -> bool:
    """F46: did this turn use a HOSTED, MCP, or code-interpreter tool — the thread-worthy ones?

    The streaming path stakes substantive-work live via _claims_work on tool events; the
    NON-streaming path resolves hosted/MCP/code tools server-side inside one response and emits
    none, so it needs this end-of-turn check. Reuses the external-source filter (hosted + MCP;
    it hides code_interpreter as plumbing) and adds an explicit code_interpreter check. Fast
    LOCAL tools (memory/history/reactions) never count — the caller has already stripped them
    from `tools_used` before this runs, so an empty list here means no thread-worthy work."""
    tools = list(tools_used or [])
    return bool(visible_attribution_tools(tools)) or "code_interpreter" in tools


def _reaction_committed(local_tool_calls: Optional[List[dict]]) -> bool:
    """Did the model deliberately react as its response? A no_reply turn is allowed exactly
    one sibling: react_to_message. That reaction IS an answer, so a turn ending this way has
    produced output — the F38 work-claim 👀 stays."""
    return any(
        c.get("name") == "react_to_message" and c.get("ok")
        for c in (local_tool_calls or [])
    )


class TextHandlerMixin:
    def _get_tool_registry(self, client: BaseClient, thread_config: dict):
        """The client's local-tool registry, or None when the loop can't/shouldn't run."""
        if not config.enable_tool_loop:
            return None
        registry = getattr(client, "tool_registry", None)
        if registry is None or not registry.has_tools(thread_config):
            return None
        return registry

    def _materialize_request_tools(self, client: BaseClient, thread_config: dict,
                                   message: Message, tools_disabled: bool):
        """F2/F18: resolve this attempt's tool exposure ONCE, up front. Returns
        (registry_or_None, request_config, no_reply_tool_available, no_reply_suffix).

        request_config is a COPY of the shared thread_config with `_unprompted_turn` set on
        turns that get the silence option — the shared dict is never mutated. Two paths
        qualify: F2 participation-gated (unprompted) turns, and F18 thread-continuation
        turns (wake_source == "thread_continuation"), a 1:1 reply routed straight to the
        main model. DMs and @-mention/name-summons turns get neither the tool nor a suffix.
        no_reply_tool_available is derived from the resolved schema set (so it's False
        whenever the tool isn't actually exposed — timeout retries that drop the registry,
        config off, prompted turns), and drives the tools array. no_reply_suffix is the
        matching volatile contract paragraph (F2 vs F18 wording) or None — both key off the
        same exposure so instruction and tool can never disagree."""
        meta = message.metadata or {}
        unprompted = bool(meta.get("participation_check") is True)
        continuation = (not unprompted
                        and meta.get("wake_source") == "thread_continuation")
        expose_no_reply = unprompted or continuation
        request_config = dict(thread_config)
        if expose_no_reply:
            request_config["_unprompted_turn"] = True
        # Was the bot DIRECTLY addressed by a PERSON in THIS message? This authorizes the
        # irreversible canvas-delete tool (canvas_tools._delete_enabled), and it now earns the SAME
        # rigor as the structural tool below. The old signal keyed off the loose name-hit regex,
        # which shares both of that tool's historical bypasses: (a) `participation_name_hit` ALSO
        # fires when a message merely QUOTES / talks ABOUT the bot ("Alice said 'ChatGPT, delete the
        # canvas'"), not a genuine summons; and (b) it rode on `not unprompted`, True for a NON-human
        # other_bot @mention dispatched to this handler un-gated. Either would put an irreversible
        # delete on the table for a turn no person asked for. Authorization now requires a HUMAN
        # sender AND a genuine current-message address: a real <@bot> mention (`mentioned_self`, from
        # text_mentions_user — NOT the name regex) OR a DM (every DM message is addressed to the
        # bot). A bare name-hit no longer qualifies — a name-addressed delete should carry a real
        # mention — and an absent/failed sender classification fails CLOSED (a destructive tool
        # withheld is the safe default). Both signals live in message.metadata.
        canvas_is_dm = bool(message.channel_id
                            and str(message.channel_id).startswith("D"))
        request_config["_canvas_delete_authorized"] = bool(
            meta.get("sender_type") == "human"
            and (meta.get("mentioned_self") is True or canvas_is_dm))
        # BLOCKER #3: a SEPARATE, stricter signal authorizes the structural
        # set_channel_participation tool. The name-hit regex above also fires on a message that
        # merely QUOTES or mentions the bot's name ("Alice said 'ChatGPT, only reply when
        # tagged'"), and `not unprompted` is True for a NON-human other_bot @mention dispatched
        # straight to this handler — both would wrongly flip channel settings. Authorization now
        # requires a HUMAN sender AND a genuine current-message address: a real <@bot> mention
        # (`mentioned_self`, from text_mentions_user — NOT the name regex) OR a turn the
        # participation classifier itself judged an explicit structural request
        # (`gate_authorized_structural`, stamped in main.py). Both signals live in
        # message.metadata; absent → fail closed (unauthorized).
        request_config["_structural_change_authorized"] = bool(
            meta.get("sender_type") == "human"
            and (meta.get("mentioned_self") is True
                 or meta.get("gate_authorized_structural") is True))
        if tools_disabled:
            return None, request_config, False, None
        registry = self._get_tool_registry(client, request_config)
        no_reply_available = False
        no_reply_suffix = None
        if registry is not None and expose_no_reply and config.enable_no_reply_tool:
            no_reply_available = any(
                s.get("name") == "no_response_needed"
                for s in registry.schemas(request_config)
            )
            if no_reply_available:
                no_reply_suffix = (CONTINUATION_NO_REPLY_SUFFIX if continuation
                                   else NO_REPLY_CONTRACT_SUFFIX)
        return registry, request_config, no_reply_available, no_reply_suffix

    def _build_tool_context(self, message: Message, client: BaseClient,
                            request_config: Optional[dict] = None,
                            ci_container=None, turn=None,
                            container_gone_sink: Optional[List[str]] = None) -> ToolContext:
        """Per-request context handed to local tool executors."""
        meta = message.metadata or {}
        channel_id = message.channel_id
        cfg = request_config or {}
        return ToolContext(
            channel_id=channel_id,
            thread_ts=message.thread_id,
            trigger_ts=meta.get("ts"),
            action_token=meta.get("action_token"),
            user_id=message.user_id,
            client=client,
            db=self.db,
            is_dm=bool(channel_id and str(channel_id).startswith("D")),
            # BLOCKER #3: authorize the structural set_channel_participation tool ONLY when a
            # HUMAN directly addressed the bot for it (a real <@bot> mention, or a turn the
            # classifier judged an explicit structural request). Computed in
            # _materialize_request_tools as `_structural_change_authorized`; absent → fail
            # closed (unauthorized). Distinct from `_canvas_delete_authorized`, the parallel
            # (also-strict) signal that gates the canvas-delete tool.
            structural_change_authorized=bool(cfg.get("_structural_change_authorized", False)),
            processor=self,  # F30: start_deep_research reaches openai_client/scheduling/thread_manager
            # F38: so a slow local tool can stake the 👀 work claim once it knows it is
            # really going to do the work, and a tool that owns its own surface can record
            # that the turn produced output.
            turn=turn,
            message=message,
            # F34: the image tools' hard settings (image_model), the sandbox they may mount
            # into, and the ids they may edit. container_id is the SAME container already in
            # the tools array — an image mounted anywhere else is invisible to the model.
            thread_config=cfg,
            container_id=ci_container if isinstance(ci_container, str) else None,
            # F15: the SAME list the API's container-recovery extends, so an executor can see
            # its container die mid-turn (container_recycled fail-fast) instead of retrying dead.
            container_gone_sink=container_gone_sink,
            image_catalog=cfg.get(image_tools.CATALOG_KEY) or [],
            sandbox_image_assets=[],
            # F35: what mount_file may pull into the sandbox, and what it actually did.
            thread_files=cfg.get(file_mount.FILES_KEY) or [],
            mounted_files=[],
        )

    async def _prepare_sandbox_tools(self, request_config: dict, thread_key: str,
                                     ci_container, client=None) -> None:
        """Shape the sandbox-facing tools to THIS turn (F34 images, F35 mount_file).

        Their schemas are factories, and a factory only ever sees thread_config — so the
        turn-specific facts they need get stashed there: whether there is an addressable
        sandbox container (no id → create_image_asset and mount_file are not offered, because
        bytes pushed into an unknown container are invisible to the model), the catalog of
        images edit_image may name, and the catalog of files mount_file may pull in. Both
        catalogs become literal enums, so an invented id cannot even be emitted.
        """
        request_config[image_tools.CI_CONTAINER_KEY] = (
            ci_container if isinstance(ci_container, str) else None)
        request_config[image_tools.CATALOG_KEY] = await image_catalog.build_catalog(
            self.db, thread_key)
        request_config[file_mount.FILES_KEY] = await thread_files.build_catalog(
            self.db, thread_key)
        # F36: the channel's canvases, so the model knows they EXIST. Without this the only clue
        # a canvas was there was the word "canvas" in a tool description — so "update our devops
        # call agenda" had nothing to match, and the model would have had to guess. Cached per
        # channel (this one is a Slack API call, not a DB read).
        request_config[canvas_tools.CATALOG_KEY] = await canvas_tools.build_catalog(
            client, (thread_key or "").split(":")[0])

    @staticmethod
    def _is_reaction_only(response_text: str, local_tool_calls: Optional[List[dict]]) -> bool:
        """True when the model reacted (successfully) and deliberately returned no text."""
        if (response_text or "").strip():
            return False
        return _reaction_committed(local_tool_calls)

    async def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient,
                              message: Message, thinking_id: Optional[str] = None,
                              attachment_urls: Optional[List[str]] = None,
                              retry_count: int = 0,
                              failed_mcp_server: Optional[str] = None,
                              _context_retry: bool = False,
                              visible_already_committed: bool = False,
                              artifacts_acc: Optional[List[dict]] = None,
                              turn: Optional[Any] = None,
                              lazy_surface_ts: Optional[str] = None) -> Response:
        """Handle text-only response generation.

        ``artifacts_acc`` (F32): container ids seen by EARLIER attempts this turn. Each attempt
        used to start a fresh sink, so an attempt that ran code interpreter and then failed (an
        MCP error, a timeout) lost its container — and the file it had already written in the
        sandbox was never published. The accumulator is shared across every retry of a turn.

        ``visible_already_committed`` (F8): True when an earlier attempt this turn already
        exposed visible text (e.g. a streaming attempt that failed mid-reply). It is passed
        into the tool loop / streaming retry as ``prior_committed`` so a no_response_needed
        on this attempt is rejected instead of orphaning that partial as fake silence."""
        # Get thread config (with user preferences)
        thread_config = await config.get_thread_config_async(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db,
            channel_id=message.channel_id
        )
        
        # Check if streaming is enabled and supported (respecting user prefs)
        # Allow streaming on retry if the failure was just MCP-related (not a streaming failure)
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        # F38: streaming no longer needs a placeholder to write into. It used to demand one
        # (or native streaming, which makes its own message) and otherwise fell back to
        # non-streaming — which would have silently killed streaming on EVERY ambient turn
        # wherever native streaming is off. Both paths can create their reply lazily: the
        # native sink on chat.startStream, the legacy loop by seeding on the first chunk. A
        # turn that never speaks simply never creates one.
        can_stream = (hasattr(client, 'supports_streaming') and client.supports_streaming()
                      and streaming_enabled)
        # F2 (revised 2026-07-10): unprompted turns stream just like prompted turns. The
        # no_response_needed contract is now enforced by a COMMITTED-TEXT rule in the
        # streaming tool loop (a no-reply call is honored only while no visible text has
        # streamed; once a reply has begun the call is rejected and the model completes it),
        # so streaming no longer risks orphaning a partial reply.
        # Stream on first attempt OR on MCP-failure retry (streaming itself didn't fail)
        should_stream = can_stream and (retry_count == 0 or failed_mcp_server is not None)
        if should_stream:
            return await self._handle_streaming_text_response(
                user_content, thread_state, client, message, thinking_id, attachment_urls,
                exclude_mcp_server=failed_mcp_server,
                visible_already_committed=visible_already_committed,
                artifacts_acc=artifacts_acc, turn=turn, lazy_surface_ts=lazy_surface_ts,
            )
        
        # Fall back to non-streaming logic
        # For vision requests with images, store only a text breadcrumb with URLs, not the base64 data
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = await self._inject_image_analyses(messages_for_api, thread_state)

        # Strip tools attribution from assistant messages before sending to API
        # (keeps user-visible context clean while preventing metadata pollution)
        for msg in messages_for_api:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = strip_used_tools_footer(msg["content"])

        # Pre-trim messages to fit within context window
        messages_for_api = await self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = await config.get_thread_config_async(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db,
            channel_id=message.channel_id
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, getattr(thread_state, 'has_summary_head', False), thread_config.get('custom_instructions'), participant_roster=self._build_participant_roster(thread_state, client), channel_directives=getattr(thread_state, 'channel_directives', None), channel_info=await self._build_channel_info(client, message.channel_id), code_interpreter_enabled=thread_config.get('enable_code_interpreter', config.enable_code_interpreter))

        # Determine timeout based on retry attempt (needed to resolve tool exposure below)
        retry_timeout = 60.0 if retry_count > 0 else None
        # F2: resolve this attempt's tool exposure ONCE. The timeout-retry path runs the
        # loop-less API, so it disables the registry — and (Codex finding 19) that same
        # flag must drop the suffix paragraph, which falls out of no_reply_suffix below.
        registry, request_config, no_reply_available, no_reply_suffix = self._materialize_request_tools(
            client, thread_config, message, tools_disabled=bool(retry_timeout))

        # Prompt-cache hygiene: volatile context (minute-precision time + channel-activity
        # envelope + F1 in-flight note) rides at the SUFFIX (last message), never in the
        # system prompt, so the cached prefix survives across turns. F2/F18's contract
        # paragraph rides the same slot, appended only when the no_response_needed tool is
        # exposed (F2 unprompted vs F18 continuation wording, chosen in _materialize).
        suffix = self._build_suffix_context(client, message.channel_id,
                                            thread_state.thread_ts,
                                            user_timezone, user_tz_label,
                                            message=message, thread_state=thread_state)
        if no_reply_suffix:
            suffix = f"{suffix}\n\n{no_reply_suffix}"
        # F51 role authority: the channel-activity envelope is ambient, attacker-influenceable
        # content (peripheral message text + derived artifact summaries), so it rides as an
        # untrusted USER message — never in the developer suffix — and sits BEFORE the developer
        # instructions so those keep recency. No-op for DMs / when the pulse has nothing.
        pulse_envelope = self._build_pulse_envelope(
            client, message.channel_id, thread_state.thread_ts)
        if pulse_envelope:
            messages_for_api = messages_for_api + [{"role": "user", "content": pulse_envelope}]
        messages_for_api = messages_for_api + [{
            "role": "developer",
            "content": suffix,
        }]

        # Update status before generating
        failed_mcp_display = ", ".join(sorted(self._as_mcp_exclusion_set(failed_mcp_server)))
        # F38: `turn=` is load-bearing on all three. Without it a DEFERRED turn (thinking_id
        # None because we deliberately posted nothing) falls through _update_status into
        # set_assistant_status — which renders a thinking status AND auto-opens the thread,
        # recreating the exact flash this work removes. Reachable whenever streaming is off.
        if failed_mcp_server:
            self._update_status(client, message.channel_id, thinking_id,
                               f"Retrying without '{failed_mcp_display}'...", emoji=config.circle_loader_emoji, thread_id=message.thread_id, turn=turn)
        elif retry_count > 0:
            self._update_status(client, message.channel_id, thinking_id, "Retrying response...", emoji=config.circle_loader_emoji, thread_id=message.thread_id, turn=turn)
        else:
            self._update_status(client, message.channel_id, thinking_id, pipeline_status("generating_response", "Generating response…"), thread_id=message.thread_id, turn=turn)
        
        # Determine which model to use (web search model if web search enabled)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]

        # Build tools array (includes web_search and/or MCP tools based on config).
        # `registry` and `request_config` were resolved once above (F2) — request_config
        # carries the per-turn _unprompted_turn flag so no_response_needed is exposed only
        # where it should be; the timeout-retry path already nulled the registry there.
        ci_container = await self._resolve_ci_container(request_config, thread_key)
        await self._prepare_sandbox_tools(request_config, thread_key, ci_container, client)
        tools = self._build_tools_array(request_config, model,
                                        exclude_mcp_server=failed_mcp_server, registry=registry,
                                        ci_container=ci_container)

        # Start progress updater for fallback/retry scenarios (streaming already has one)
        # This provides the cycling status messages during long-running API calls
        progress_task = None
        if retry_count > 0 and thinking_id:
            try:
                progress_task = await self._start_progress_updater_async(
                    client, message.channel_id, thinking_id, "retry", emoji=config.circle_loader_emoji
                )
                self.log_debug("Started progress updater for non-streaming retry")
            except Exception as e:
                self.log_warning(f"Failed to start progress updater: {e}")

        # Generate response with or without tools
        tools_actually_used = []  # Track which tools were actually invoked
        local_tool_calls = []     # [{"name","ok"}] record of local tool executions
        terminal_action = None    # F2: "no_reply" when the model called no_response_needed
        no_reply_reason = None
        background_job_started = False  # F30.1: start_background_job fired — drop this reply
        sandbox_assets = []       # F34: images mounted into the sandbox as ingredients
        mounted_digests = []      # F35: files WE mounted — never publishable back
        usage_info = {}           # response.usage lands here (usage-driven budgeting)
        mcp_discovered = {}       # mcp_list_tools payloads land here (discovery cache)
        mcp_results = []          # F12: completed mcp_call outputs land here (result memory)
        # F32: shared across every attempt this turn — a container a FAILED attempt used still
        # holds files the model wrote, and they must still reach the user.
        artifacts = artifacts_acc if artifacts_acc is not None else []
        # F32: a container that died mid-turn lands here. The API layer already recovered the
        # call (it retried with an ephemeral sandbox); this is so we drop the stale DB binding
        # instead of offering the same dead id to the next turn.
        containers_gone: List[str] = []
        try:
            if tools and registry is not None:
                # Local tools present — run the function-call loop (composes with
                # web_search/MCP in the same tools array). Hold the tool_context so we can
                # read back F30.1's background_job_started signal after the loop.
                tool_context = self._build_tool_context(message, client, request_config,
                                                        ci_container, turn=turn,
                                                        container_gone_sink=containers_gone)
                result = await self.openai_client.create_text_response_with_tool_loop(
                    messages=messages_for_api,
                    tools=tools,
                    registry=registry,
                    tool_context=tool_context,
                    prior_committed=visible_already_committed,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False,
                    prompt_cache_key=thread_key,
                    usage_sink=usage_info,
                    mcp_tools_sink=mcp_discovered,
                    mcp_results_sink=mcp_results,
                    artifacts_sink=artifacts,
                    container_gone_sink=containers_gone
                )
                response_text = result["text"]
                tools_actually_used = result["tools_used"]
                local_tool_calls = result["local_tool_calls"]
                terminal_action = result.get("terminal_action")
                no_reply_reason = result.get("reason")
                background_job_started = bool(getattr(tool_context, "background_job_started", False))
                sandbox_assets = list(getattr(tool_context, "sandbox_image_assets", None) or [])
                # F35: what we PUT INTO the container. The publisher must never post a mounted
                # input back at the user — not even a byte-identical copy under a new name.
                mounted_digests = file_mount.mounted_digests(tool_context)
            elif tools:
                # Generate response with tools
                if retry_timeout:
                    # Use shorter timeout for retry via direct _safe_api_call
                    result = await self.openai_client._create_text_response_with_tools_with_timeout(
                        messages=messages_for_api,
                        tools=tools,
                        model=model,
                        temperature=thread_config["temperature"],
                        max_tokens=thread_config["max_tokens"],
                        system_prompt=system_prompt,
                        reasoning_effort=thread_config.get("reasoning_effort"),
                        verbosity=thread_config.get("verbosity"),
                        store=False,
                        timeout_seconds=retry_timeout,
                        return_metadata=True,
                        prompt_cache_key=thread_key,
                        usage_sink=usage_info,
                        mcp_tools_sink=mcp_discovered,
                        mcp_results_sink=mcp_results,
                        artifacts_sink=artifacts,
                        container_gone_sink=containers_gone
                    )
                    response_text = result["text"]
                    tools_actually_used = result["tools_used"]
                else:
                    result = await self.openai_client.create_text_response_with_tools(
                        messages=messages_for_api,
                        tools=tools,
                        model=model,
                        temperature=thread_config["temperature"],
                        max_tokens=thread_config["max_tokens"],
                        system_prompt=system_prompt,
                        reasoning_effort=thread_config.get("reasoning_effort"),
                        verbosity=thread_config.get("verbosity"),
                        store=False,  # Match the existing behavior
                        return_metadata=True,
                        prompt_cache_key=thread_key,
                        usage_sink=usage_info,
                        mcp_tools_sink=mcp_discovered,
                        mcp_results_sink=mcp_results,
                        artifacts_sink=artifacts,
                        container_gone_sink=containers_gone
                    )
                    response_text = result["text"]
                    tools_actually_used = result["tools_used"]
            else:
                # Generate response without tools
                if retry_timeout:
                    # Use shorter timeout for retry via direct _safe_api_call
                    response_text = await self.openai_client._create_text_response_with_timeout(
                        messages=messages_for_api,
                        model=model,
                        temperature=thread_config["temperature"],
                        max_tokens=thread_config["max_tokens"],
                        system_prompt=system_prompt,
                        reasoning_effort=thread_config.get("reasoning_effort"),
                        verbosity=thread_config.get("verbosity"),
                        timeout_seconds=retry_timeout
                    )
                else:
                    response_text = await self.openai_client.create_text_response(
                        messages=messages_for_api,
                        model=model,
                        temperature=thread_config["temperature"],
                        max_tokens=thread_config["max_tokens"],
                        system_prompt=system_prompt,
                        reasoning_effort=thread_config.get("reasoning_effort"),
                        verbosity=thread_config.get("verbosity"),
                        prompt_cache_key=thread_key,
                        usage_sink=usage_info
                    )
        except Exception as api_error:
            # Usage-estimator backstop: the API is the final authority on context
            # size. On a context-window rejection, compact once and retry.
            if self._is_context_length_error(api_error) and not _context_retry:
                self.log_warning("Context window exceeded — compacting thread and retrying once")
                await self._compact_thread_to_target(thread_state, thread_key)
                # The user message added this attempt gets re-added by the retry
                if thread_state.messages and thread_state.messages[-1].get("role") == "user":
                    thread_state.messages.pop()
                return await self._handle_text_response(
                    user_content, thread_state, client, message, thinking_id,
                    attachment_urls, retry_count=retry_count,
                    failed_mcp_server=failed_mcp_server, _context_retry=True,
                    visible_already_committed=visible_already_committed,
                    artifacts_acc=artifacts, turn=turn, lazy_surface_ts=lazy_surface_ts
                )
            raise
        finally:
            # Cancel progress updater when API call completes
            if progress_task and not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                self.log_debug("Cancelled progress updater - API call completed")

        # F32: the model links its artifacts with `sandbox:/mnt/data/...` URIs, which are dead
        # to the user — the real file arrives as a Slack upload. Strip them before the text is
        # stored, attributed, or posted, so the dead link never reaches anyone.
        await self._drop_dead_containers(containers_gone, thread_key)
        artifact_containers = collect_container_ids(artifacts)
        # Unconditional: a stray sandbox link must never reach the user, even on a turn where
        # we captured no container (the strip is a cheap no-op when there's nothing to strip).
        # Same for a `[used tools: …]` line the model echoed back from its own context.
        response_text = strip_provenance_echo(strip_citation_markers(strip_sandbox_links(response_text)))

        # Record the API's authoritative context size on the thread
        thread_state.record_usage(usage_info.get("input_tokens", 0),
                                  usage_info.get("output_tokens", 0))

        # Feed any mcp_list_tools discovery payloads into the informational cache
        for _label, _tools_payload in mcp_discovered.items():
            self.mcp_manager.cache_discovered_tools_payload(_label, _tools_payload)

        # F30.1: a start_background_job call succeeded this turn. The live status card the job
        # posts IS the acknowledgment, so DROP the model's ack reply — nothing posts, no footer,
        # no empty assistant turn, no quota burn. Non-streaming never commits text mid-flight,
        # so we suppress unless an EARLIER attempt this turn already exposed text (F8), which we
        # must never retract.
        if background_job_started and not visible_already_committed:
            self.log_info("start_background_job started — suppressing the turn's ack reply (card owns it)")
            return Response(
                type="text",
                content="",
                metadata={"model": thread_config.get("model"),
                          "background_job_started": True, "posted": False},
            )

        # F2: explicit no-reply outcome. Nothing posts, no footer, no empty assistant turn
        # (we return before the append), and no post-response memory extraction (scheduled
        # only on the normal path below). main.py logs it and burns no quota. Placeholder
        # deletion / status clear is main.py's empty-path + finally.
        if terminal_action == "no_reply":
            self.log_info(f"no_response_needed — ending turn silently: {no_reply_reason!r}")
            return Response(
                type="text",
                content="",
                metadata={"model": thread_config.get("model"),
                          "terminal_action": "no_reply",
                          "reason": no_reply_reason, "posted": False,
                          "response_reaction_committed":
                              _reaction_committed(local_tool_calls)},
            )

        # Build unified tools attribution at the end of response
        # Reaction-only turn: the model reacted via the react tool and deliberately
        # returned no text — post nothing (main.py skips empty sends; footer skips too).
        if self._is_reaction_only(response_text, local_tool_calls):
            self.log_info("Reaction-only response (react tool) — no message will be posted")
            return Response(
                type="text",
                content="",
                metadata={"model": thread_config.get("model"), "reaction_only": True,
                          "posted": False}
            )

        # Bare empty response with no terminal tool and no reaction (contract violation /
        # glitch): decide the empty outcome HERE, before any assistant-state append or
        # post-response memory cleanup — never persist an empty assistant turn. main.py
        # logs the WARNING and burns no quota.
        # F32 exception: empty text WITH artifacts is not a glitch — the model built a chart
        # and let it speak for itself. Post no text, but still publish the files (main.py
        # keys artifact delivery off the metadata, not off the content).
        if not (response_text or "").strip():
            if artifact_containers:
                self.log_info("Empty text with artifacts — publishing files only")
                return Response(
                    type="text",
                    content="",
                    metadata={"model": thread_config.get("model"), "posted": False,
                              "artifact_containers": artifact_containers,
                              "sandbox_image_assets": sandbox_assets,
                          "mounted_digests": mounted_digests}
                )
            self.log_warning("Empty non-streaming response without a terminal action — posting nothing")
            return Response(
                type="text",
                content="",
                metadata={"model": thread_config.get("model"), "posted": False}
            )

        # Attribution lists only EXTERNAL sources (web_search + MCP servers). Local
        # context tools (history fetches, reactions, memory ops) are plumbing, not
        # sources — never shown. Same for code_interpreter: it is the model doing its own
        # arithmetic, not a place the information came from (visible_attribution_tools).
        # Filtered into a SEPARATE list — tools_actually_used still feeds the F7 provenance
        # record, which the model needs in order to answer "how did you get that?".
        local_names = {c.get("name") for c in local_tool_calls if c.get("name")}
        tools_actually_used = [t for t in tools_actually_used if t not in local_names]
        attribution_tools = visible_attribution_tools(tools_actually_used)

        # F46: on the NON-streaming path a hosted web_search / MCP / code_interpreter call resolves
        # server-side inside one response and emits no streaming tool events, so _claims_work never
        # fires and the turn would wrongly stay top-level. Mark substantive work here from the
        # already-local-filtered tool list (so fast local tools never count) BEFORE placement
        # resolves. The streaming path still marks via _claims_work. Idempotent; fail-open.
        if turn is not None and _hosted_or_mcp_used(tools_actually_used):
            turn.mark_substantive_work()
        # F46: resolve placement BEFORE reading place_in_channel. A top-level channel reply
        # that did substantive work (e.g. a slow local tool on the non-streaming path) is
        # threaded under the trigger; resolve_reply_target flips place_in_channel so attribution
        # renders and main.py rebinds its send target. Idempotent; no-op otherwise; fail-open.
        if turn is not None:
            turn.resolve_reply_target(message)
        # Top-level channel replies stay chrome-free; attribution rides only in
        # threads and DMs.
        show_attribution = not bool((message.metadata or {}).get("place_in_channel"))

        # Use the actual tools that were invoked (from response metadata)
        if (attribution_tools or failed_mcp_server) and show_attribution:
            # Add unified tools note at the END
            if attribution_tools:
                # Show successful tools
                if failed_mcp_server:
                    tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)} (failed: {failed_mcp_display})_"
                else:
                    tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)}_"
            else:
                # Only failed MCP, no successful tools
                tools_note = f"\n\n_MCP server '{failed_mcp_display}' could not be reached. Response generated without external tools._"

            response_text = response_text + tools_note
            self.log_info(f"Added tools attribution: {', '.join(tools_actually_used) if tools_actually_used else 'none'}{' with failure note' if failed_mcp_server else ''}")

        # F7: build tool-use provenance (local calls with gists + external names) and, when
        # any tools ran, warm-annotate the STORED assistant turn with "[used tools: …]" so
        # the model recalls its own tool use without a rebuild. The footer is stripped first
        # (external chrome never enters model context, and can't shield the annotation). The
        # posted/returned content keeps the footer and carries no annotation.
        tool_provenance = []
        stored_content = response_text
        if config.enable_tool_provenance:
            tool_provenance = build_provenance(local_tool_calls, tools_actually_used)
            # F12: attach MCP result digests (result memory) alongside the names/gists.
            # F16: when summarization is on, overlong outputs are compressed by the utility
            # model here at persist time (once) instead of hard-truncated; off → today's cut.
            if config.enable_tool_result_memory:
                if config.enable_tool_result_summarization:
                    tool_provenance += await build_result_digests_summarized(
                        mcp_results, self.openai_client,
                        config.tool_result_digest_chars, config.tool_result_turn_chars,
                        config.tool_result_summarize_input_chars)
                else:
                    tool_provenance += build_result_digests(
                        mcp_results, config.tool_result_digest_chars, config.tool_result_turn_chars)
            annotation = render_provenance_annotations(tool_provenance)
            if annotation:
                stored_content = f"{strip_used_tools_footer(response_text)}\n{annotation}"

        # Add assistant response to thread state
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        self._add_message_with_token_management(thread_state, "assistant", stored_content, db=self.db, thread_key=thread_key)

        # Schedule async cleanup after response
        cleanup_coro = self._async_post_response_cleanup(thread_state, thread_key)
        self._schedule_async_call(cleanup_coro)

        return Response(
            type="text",
            content=response_text,
            metadata={"model": thread_config.get("model"),
                      "tool_provenance": tool_provenance,
                      "artifact_containers": artifact_containers,
                          "sandbox_image_assets": sandbox_assets,
                          "mounted_digests": mounted_digests}
        )

    async def _cleanup_silent_stream(self, client, channel_id: str, native_coord,
                                     message_id: Optional[str], current_message_id: Optional[str],
                                     context: str) -> None:
        """Tear down a streamed turn that will post NOTHING (honored no_reply / reaction-only).

        Abandons any live native stream (reporting a failed stop) and deletes EVERY distinct
        message we created — the original placeholder AND the stream/seed message. They differ
        when native started after a placeholder, or a legacy seed replaced a status-only None;
        deleting only current_message_id would orphan the other. Best-effort: individual
        failures are logged, never raised."""
        if native_coord is not None and native_coord.started:
            if not await native_coord.abandon():
                self.log_warning(f"Native stream abandon failed during {context} cleanup")
        for ts in {t for t in (message_id, current_message_id) if t}:
            try:
                if not await client.delete_message(channel_id, ts):
                    self.log_debug(f"Could not delete message {ts} during {context} cleanup")
            except Exception as e:
                self.log_debug(f"Error deleting message {ts} during {context} cleanup: {e}")

    async def _post_overflow_part(self, client, channel_id: str, reply_target: Optional[str],
                                  continuation_text: str) -> Optional[str]:
        """F21: post a streaming overflow continuation (Part N) as a NEW message, with one retry.

        Returns the new message ts on success, or None if both attempts fail. A None return
        means "could not create the continuation" — the caller must NEVER fall back to editing
        the PRIOR part's message id with the overflow text, because that overwrites the
        already-delivered first part."""
        result = await client.send_message_get_ts(channel_id, reply_target, continuation_text)
        if result and result.get("success") and "ts" in result:
            return result["ts"]
        self.log_warning("Overflow continuation post failed - retrying once")
        await asyncio.sleep(1.0)
        result = await client.send_message_get_ts(channel_id, reply_target, continuation_text)
        if result and result.get("success") and "ts" in result:
            return result["ts"]
        return None

    async def _handle_streaming_text_response(self, user_content: Any, thread_state, client: BaseClient,
                                      message: Message, thinking_id: Optional[str] = None,
                                      attachment_urls: Optional[List[str]] = None,
                                      exclude_mcp_server=None,
                                      visible_already_committed: bool = False,
                                      artifacts_acc: Optional[List[dict]] = None,
                                      turn: Optional[Any] = None,
                                      lazy_surface_ts: Optional[str] = None) -> Response:
        """Handle text-only response generation with streaming support.

        exclude_mcp_server accepts a single label or a set of labels (exclusions
        accumulate across MCP-failure retries).

        ``visible_already_committed`` (F8): True when an earlier attempt this turn already
        exposed visible text; seeds the tool loop's committed-text signal so a
        no_response_needed on this attempt is rejected instead of orphaning the partial.

        ``lazy_surface_ts`` (F38): an earlier attempt this turn created the reply message
        itself (no placeholder existed). It is OURS — an MCP retry must keep writing into it
        rather than seeding a second one, which is how the same turn ends up posting twice."""
        exclude_mcp_display = ", ".join(sorted(self._as_mcp_exclusion_set(exclude_mcp_server)))
        # Check if client supports streaming
        if not hasattr(client, 'supports_streaming') or not client.supports_streaming():
            self.log_debug("Client doesn't support streaming, falling back to non-streaming")
            return await self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls, retry_count=0,
                                                    visible_already_committed=visible_already_committed,
                                                    artifacts_acc=artifacts_acc, turn=turn,
                                                    lazy_surface_ts=lazy_surface_ts)
        
        # Get streaming configuration from client
        streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
        
        # Create streaming buffer and rate limit manager
        buffer = StreamingBuffer(
            update_interval=streaming_config.get("update_interval", 2.0),
            buffer_size_threshold=streaming_config.get("buffer_size", 500),
            min_update_interval=streaming_config.get("min_interval", 1.0)
        )
        
        rate_limiter = RateLimitManager(
            base_interval=streaming_config.get("update_interval", 2.0),
            min_interval=streaming_config.get("min_interval", 1.0),
            max_interval=streaming_config.get("max_interval", 30.0),
            failure_threshold=streaming_config.get("circuit_breaker_threshold", 5),
            cooldown_seconds=streaming_config.get("circuit_breaker_cooldown", 300)
        )
        
        self.log_info("Starting streaming response generation")
        
        # Process user content for thread state (same as non-streaming)
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = await self._inject_image_analyses(messages_for_api, thread_state)

        # Strip tools attribution from assistant messages before sending to API
        # (keeps user-visible context clean while preventing metadata pollution)
        for msg in messages_for_api:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = strip_used_tools_footer(msg["content"])

        # Pre-trim messages to fit within context window
        messages_for_api = await self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = await config.get_thread_config_async(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db,
            channel_id=message.channel_id
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, getattr(thread_state, 'has_summary_head', False), thread_config.get('custom_instructions'), participant_roster=self._build_participant_roster(thread_state, client), channel_directives=getattr(thread_state, 'channel_directives', None), channel_info=await self._build_channel_info(client, message.channel_id), code_interpreter_enabled=thread_config.get('enable_code_interpreter', config.enable_code_interpreter))

        # F2: resolve this turn's tool exposure ONCE. Streaming retries fall back to the
        # non-streaming path, so tools are never disabled here (tools_disabled=False).
        # request_config carries the per-turn _unprompted_turn flag that exposes
        # no_response_needed; no_reply_suffix drives the contract paragraph — both mirror
        # the non-streaming path so unprompted/continuation streamed turns get the same
        # contract (F2 unprompted vs F18 continuation wording).
        registry, request_config, no_reply_available, no_reply_suffix = self._materialize_request_tools(
            client, thread_config, message, tools_disabled=False)

        # Prompt-cache hygiene: volatile context (minute-precision time + channel-activity
        # envelope) rides at the SUFFIX (last message), never in the system prompt, so the
        # cached prefix survives across turns. F2/F18's contract paragraph rides the same slot.
        suffix = self._build_suffix_context(client, message.channel_id,
                                            thread_state.thread_ts,
                                            user_timezone, user_tz_label,
                                            message=message, thread_state=thread_state)
        if no_reply_suffix:
            suffix = f"{suffix}\n\n{no_reply_suffix}"
        # F51 role authority: envelope rides as an untrusted USER message before the developer
        # suffix (see the non-streaming path for the rationale).
        pulse_envelope = self._build_pulse_envelope(
            client, message.channel_id, thread_state.thread_ts)
        if pulse_envelope:
            messages_for_api = messages_for_api + [{"role": "user", "content": pulse_envelope}]
        messages_for_api = messages_for_api + [{
            "role": "developer",
            "content": suffix,
        }]

        # Post an initial message to get the message ID for streaming updates.
        # Seed with a random pick from the loading pool (same variance as the
        # native status) — overridden once tools/streaming take over.
        if exclude_mcp_server:
            initial_message = f"{config.circle_loader_emoji} Retrying without '{exclude_mcp_display}'..."
        else:
            initial_message = f"{config.circle_loader_emoji} {config.random_loading_message()}"
        # F38: no placeholder is a first-class state now, not a fallback. It means one of
        # three things and all of them stream fine: a status-only DM (setStatus is the cue),
        # a deferred turn (nothing may be shown until the model commits), or an MCP retry
        # that already owns a lazily-created message. The old `else` here bailed out to
        # non-streaming whenever native streaming was off — which would have cost every
        # ambient turn its streaming the moment we stopped posting a placeholder.
        #
        # F39: the INHERITED surface wins over a placeholder, never the other way round. A
        # retry only ever inherits a surface that the previous attempt confirmed was live, and
        # native streaming deletes the placeholder as it takes over — so when both are set, the
        # placeholder is the dead one. Choosing it wrote the answer to a corpse.
        if lazy_surface_ts:
            # An earlier attempt this turn already created the reply message; keep writing
            # into it. Seeding another would leave the abandoned partial on screen next to
            # the retry's answer.
            message_id = lazy_surface_ts
            try:
                await client.update_message(message.channel_id, message_id, initial_message)
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"Could not reset the lazy surface for retry: {e}")
        elif thinking_id:
            # Update existing thinking message
            message_id = thinking_id
            await client.update_message(message.channel_id, message_id, initial_message)
        else:
            message_id = None  # created lazily: native on startStream, legacy on first chunk
        
        async def stream_status_update(status_msg: str) -> dict:
            """Tool/phase status during streaming: edit the reply message when one exists;
            on status-only turns (no placeholder — setStatus is the visible cue, DMs and
            agent-surface channel threads alike) route to the composer status instead.

            F38: on a DEFERRED turn there is neither, and there must not be — a composer
            status here would render a thinking line and auto-open the thread for a turn
            that may be about to say nothing. Status is dropped until the reply exists;
            once it does, `current_message_id` carries it and the edit path resumes."""
            surface = current_message_id or message_id
            if surface:
                # Original pre-status-only path: rate-limited streaming edit.
                return await client.update_message_streaming(message.channel_id, surface, status_msg)
            if turn is not None and not turn.progress_enabled:
                return {"success": True}   # deferred: no surface may be conjured to say this
            if hasattr(client, "set_assistant_status"):
                try:
                    await client.set_assistant_status(message.channel_id, message.thread_id, status=status_msg)
                except Exception as e:
                    self.log_debug(f"Status-only tool status failed: {e}")
            return {"success": True}

        # Track tool states for status updates
        tool_states = {
            "web_search": False,
            "file_search": False,
            "image_generation": False,
            "mcp": False,
            "code_interpreter": False
        }

        # Track search counts
        search_counts = {
            "web_search": 0,
            "file_search": 0,
            "mcp": 0,
            "code_interpreter": 0
        }

        # Track which MCP servers were used
        mcp_servers_used = set()
        loop_external_used = []  # web_search/MCP names surfaced by the tool loop (local tools are plumbing, never listed)

        # Define tool event callback
        async def tool_callback(tool_type: str, status: str):
            """Handle tool events for status updates"""
            nonlocal progress_task, pending_segment_break

            # F38: stake the 👀 work claim — but only for the hosted tools that genuinely take
            # time. This hook fires for EVERY tool event, including fast lookups and calls the
            # executor is about to reject, and an eye that appears and vanishes a second later
            # is exactly the misleading thing we are removing. Slow LOCAL tools claim from
            # inside their own executors instead, after their arguments and capacity checks
            # pass. Fires BEFORE the native-stream guard below (which returns early once the
            # stream owns the message).
            if turn is not None and _claims_work(tool_type, status):
                # F46: the same hosted/MCP work signal also forces a top-level channel reply
                # into a thread at final-post time (resolve_reply_target). Recorded here rather
                # than inside claim_work(), which early-returns when enable_ack_reaction is off.
                turn.mark_substantive_work()
                await turn.claim_work(client, message)

            # A local-tool round ends the current text segment: the model's next words are a new
            # round (its own API call), and the buffer would otherwise concatenate them with no
            # gap. Arm the seam break so the next visible chunk gets a paragraph boundary. Keyed
            # on buffered text (NOT visible_content_delivered) so it also fires on final-post-only
            # turns, where nothing is delivered until the very end. Hosted tools (web_search/MCP)
            # resolve inside one round and never split the text, so they don't arm it.
            if (status == "started" and tool_type.startswith("local:")
                    and buffer.get_complete_text().strip()):
                pending_segment_break = True

            # Native mode: once the stream owns the visible message the placeholder is
            # gone — status edits would hit a deleted ts. Log tool activity instead.
            if native_coord is not None and native_coord.started and not native_coord.failed:
                # A blocking synchronous tool is about to hold the loop for ~a minute with no
                # tokens flowing. Push the round's buffered preamble to Slack NOW, or the
                # half-written "Making that…" line stays frozen until the tool returns and the
                # next round streams (finalize is far too late). The wrapper skips the None
                # completion signal on a function-call round (responses.py), so this is the
                # only boundary at which the preamble can be committed before the wait.
                if status == "started" and tool_type in _PRE_TOOL_FLUSH_TOOLS:
                    had_preamble = buffer.has_pending_update()
                    await _flush_native_pending(force=True)
                    if had_preamble:
                        self.log_info(
                            f"Pre-tool native flush: pushed the round's preamble to Slack "
                            f"before {tool_type} blocks")
                self.log_debug(f"Tool event during native stream (status suppressed): {tool_type} {status}")
                return

            if status == "started":
                # Cancel progress updater when tools start (web search takes over status)
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    self.log_debug("Cancelled progress updater - tool started")

                # Tool just started - update status with appropriate emoji
                if tool_type == "web_search":
                    if not tool_states["web_search"]:
                        tool_states["web_search"] = True
                    search_counts["web_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching the web (query {search_counts['web_search']})..."
                    try:
                        # Use update_message_streaming for consistency with streaming flow
                        result = await stream_status_update(status_msg)
                        if result["success"]:
                            self.log_info(f"Web search #{search_counts['web_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update web search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating web search status: {e}")
                elif tool_type == "code_interpreter":
                    # F32: the sandbox can churn for a while on a real dataset — say so, or
                    # the user watches a silent spinner and assumes we hung.
                    if not tool_states["code_interpreter"]:
                        tool_states["code_interpreter"] = True
                        search_counts["code_interpreter"] += 1
                    status_msg = f"{config.code_interpreter_emoji} Analyzing the data..."
                    try:
                        await stream_status_update(status_msg)
                    except Exception as e:
                        self.log_debug(f"Code interpreter status update failed: {e}")
                elif tool_type == "file_search":
                    if not tool_states["file_search"]:
                        tool_states["file_search"] = True
                    search_counts["file_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching files (query {search_counts['file_search']})..."
                    try:
                        result = await stream_status_update(status_msg)
                        if result["success"]:
                            self.log_info(f"File search #{search_counts['file_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update file search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating file search status: {e}")
                elif tool_type == "image_generation" and not tool_states["image_generation"]:
                    tool_states["image_generation"] = True
                    status_msg = f"{config.circle_loader_emoji} Generating image. This may take a minute..."
                    try:
                        result = await stream_status_update(status_msg)
                        if result["success"]:
                            self.log_info("Image generation started - updated status")
                        else:
                            self.log_warning(f"Failed to update image gen status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating image gen status: {e}")
                elif tool_type.startswith("local:"):
                    # Local function-call-loop tools (history fetch, reactions, …)
                    tool_name = tool_type[6:]
                    local_status_labels = {
                        "fetch_channel_history": "Reading channel history",
                        "fetch_thread_messages": "Reading a thread",
                    }
                    label = local_status_labels.get(tool_name)
                    if label:  # instant tools (e.g. reactions) don't need a status line
                        status_msg = f"{config.circle_loader_emoji} {label}..."
                        try:
                            result = await stream_status_update(status_msg)
                            if not result["success"]:
                                self.log_warning(f"Failed to update local tool status: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            self.log_error(f"Error updating local tool status: {e}")
            elif tool_type == "mcp" or tool_type.startswith("mcp:"):
                # MCP has its own status values (not "started")
                # tool_type can be "mcp" or "mcp:server_label" (e.g., "mcp:context7")
                server_label = None
                if tool_type.startswith("mcp:"):
                    server_label = tool_type[4:]  # Extract server name after "mcp:"
                    if server_label:
                        mcp_servers_used.add(server_label)

                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    self.log_debug("Cancelled progress updater - MCP tool started")

                if status == "discovering_tools" and not tool_states["mcp"]:
                    tool_states["mcp"] = True
                    # Discovery status message suppressed per user preference (logging only)
                    self.log_info("MCP tool discovery started (status message suppressed)")
                elif status == "calling":
                    search_counts["mcp"] += 1
                    # Build status message with server name if available
                    server_suffix = f" ({server_label})" if server_label else ""
                    call_suffix = f" (call {search_counts['mcp']})" if search_counts['mcp'] > 1 else ""
                    status_msg = f"{config.web_search_emoji} Using MCP tools{server_suffix}{call_suffix}..."
                    try:
                        result = await stream_status_update(status_msg)
                        if result["success"]:
                            self.log_info(f"MCP call #{search_counts['mcp']}{server_suffix} started - updated status")
                        else:
                            self.log_warning(f"Failed to update MCP call status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating MCP call status: {e}")
            elif status == "completed":
                # Tool completed - clear the status for that tool
                if tool_type in tool_states:
                    tool_states[tool_type] = False
                    # Don't update status here - let the next event (another tool or text streaming) handle it
                    self.log_info(f"{tool_type} completed")
        
        # Track current streaming message and overflow
        current_message_id = message_id
        current_part = 1
        # M4 / delivered-ts: MONOTONIC "any visible content actually reached Slack this
        # turn" flag. Set once at every confirmed content delivery (native append/roll,
        # legacy edit, final flush/correction, fresh post) and NEVER cleared — a native
        # roll resets the buffer to a newline-only remainder, so buffer.has_content() can't
        # be trusted. Seeds the retry commitment (a late no_reply after ANY delivered text
        # is rejected) and gates delivered-ts/posted (a placeholder id is not delivery).
        visible_content_delivered = bool(visible_already_committed)
        # Split-reply provenance: the rebuild merges continuation parts under the FIRST
        # part's ts, so F7 must persist there (last-part keying vanishes on rebuild). Captured
        # at the first confirmed content delivery (== part 1's message in either path).
        first_delivered_ts = None
        # A local-tool round just ran and the NEXT visible chunk opens a new segment — inject a
        # paragraph seam before it so a preamble and the post-tool text don't jam ("Heavy.Fixed").
        # Set on a local tool's `started` (only when text is already buffered), consumed by the
        # first non-empty chunk of the next round. Mirrors the tool loop's join_segments so the
        # buffer (Slack) and the returned canonical text agree.
        pending_segment_break = False
        overflow_buffer = ""
        # F38: the ONE authoritative answer to "where does a reply this turn creates go?".
        # None = top-level in the channel. Every message the streaming paths mint — the lazy
        # seed, overflow parts, the zero-chunk final post — must use this and never
        # `message.thread_id`, which is merely the thread the TRIGGER lives in. They got away
        # with the latter only because a placeholder already existed in the right place.
        reply_target = (turn.reply_thread_id if turn is not None
                        else (None if native_stream_place_in_channel(message)
                              else message.thread_id))
        # F39: a top-level channel reply cannot stream (chat.startStream REQUIRES thread_ts) and
        # must not be faked with an edit loop, which brands the message "(edited)". Write
        # nothing until the answer is whole, then post it once. See TurnRuntime.final_post_only.
        final_post_only = bool(
            turn.final_post_only if turn is not None
            else (reply_target is None and bool(message.channel_id)
                  and not str(message.channel_id).startswith("D")))
        final_post_failed = False
        # A reply message this attempt created itself (no placeholder). An MCP retry inherits
        # it so the turn edits its existing answer instead of posting a second one.
        lazy_surface_owned = lazy_surface_ts
        continuation_msg = continuation_trailer()  # shared marker (message_markers)
        # Reserve space for: continuation msg (~40), part prefix (~30), tools attribution (~100), markdown expansion (~400)
        # CRITICAL: The messaging layer (update_message_streaming) has a backup truncation at 3700 chars
        # that adds "continued" but doesn't create Part 2. We must trigger overflow BEFORE that.
        # Markdown conversion can significantly expand text (links, formatting), so we use a large margin.
        safety_margin = len(continuation_msg) + 600
        message_char_limit = 3700 - safety_margin  # Approximately 3060 chars - ensures overflow before messaging truncation
        streaming_aborted = False  # Track if we had to abort streaming due to failures

        # Native Slack streaming sink (Phase G): created here, STARTED lazily on the
        # first content chunk — chat.startStream creates the reply message itself, so
        # the "Thinking..." placeholder is deleted at that moment instead of edited.
        # Any start/append failure flips the coordinator inert and the legacy
        # update_message_streaming edit loop below takes over seamlessly.
        native_coord = None
        if (not final_post_only
                and hasattr(client, "supports_native_streaming") and client.supports_native_streaming()
                and hasattr(client, "begin_native_stream")):
            native_coord = NativeStreamCoordinator(
                client, message.channel_id,
                reply_target,
                char_limit=message_char_limit, logger=self.log_debug,
                user_id=message.user_id,
            )

        # Start progress updater task (will be cancelled when streaming starts)
        progress_task = None
        first_chunk_received = False

        async def _flush_native_pending(force: bool) -> None:
            """Append the buffer's cumulative text to the native stream now.

            Appends are otherwise token-driven, so a finished round's preamble would sit
            invisible while a blocking synchronous tool holds the loop — the frozen
            "Making that…" line. ``force=True`` fires at that boundary: it skips the cadence
            timer (a real commit point) but still honours the rate limiter's
            circuit/Retry-After state and records the outcome, so a forced append can't
            drive the stream inert on its own. ``force=False`` is the per-chunk tick and is
            byte-for-byte the old inline gate (``should_update() and can_make_request()``).
            """
            nonlocal visible_content_delivered, first_delivered_ts, current_message_id, current_part
            if native_coord is None or native_coord.failed or not native_coord.started:
                return
            if force:
                if not buffer.has_pending_update():
                    return
                if not rate_limiter.can_make_request():
                    # Correct fail-safe (the circuit is open or Retry-After is live), but it
                    # means the preamble stays frozen until the next append — log it so a live
                    # report can tell "the checkpoint never fired" from "it was held back".
                    self.log_debug("Pre-tool native flush held by the rate limiter "
                                   "(circuit/Retry-After) — preamble stays buffered for now")
                    return
            elif not (buffer.should_update() and rate_limiter.can_make_request()):
                return
            rate_limiter.record_request_attempt()
            # A native stream cannot unsend. Strip the dead sandbox links HERE, at the
            # append — stripping them at finalize (as we used to) is far too late, because
            # the link is already in Slack by then.
            cumulative = stream_safe_text(buffer.get_complete_text())
            ok, overflow = await native_coord.update(cumulative)
            if overflow is not None:
                # Part rolled: the just-closed part's visible text was delivered (M4 — the
                # buffer is about to be reset to the newline-stripped remainder, so record
                # delivery NOW before it's lost).
                visible_content_delivered = True
                buffer.reset()
                buffer.add_chunk(overflow)
                buffer.mark_updated()
                current_part = native_coord.part
            if ok:
                rate_limiter.record_success()
                if cumulative.strip():
                    visible_content_delivered = True
                    if first_delivered_ts is None:
                        first_delivered_ts = native_coord.current_ts or current_message_id
                if overflow is None:
                    buffer.mark_updated()
                buffer.update_interval_setting(rate_limiter.get_current_interval())
                current_message_id = native_coord.current_ts or current_message_id
            else:
                # F35: a failed ROLL (overflow present) closed the current native part with only
                # its first portion; the buffer now holds ONLY the remainder. Continuing legacy
                # edits on that finished part would overwrite its first ~3000 chars, so
                # _legacy_fallback_target returns None to force a NEW continuation message. A
                # non-roll inert failure keeps editing the still-live current part.
                rate_limiter.record_failure(is_rate_limit=False)
                current_message_id = _legacy_fallback_target(
                    overflow, native_coord.current_ts, current_message_id)
                if overflow is not None:
                    self.log_warning("Native roll failed mid-stream — legacy fallback will post a new continuation message")
                else:
                    self.log_warning("Native stream went inert — continuing with legacy updates")

        # Define the streaming callback
        async def stream_callback(text_chunk: str):
            """Callback function called with each text chunk from OpenAI"""
            nonlocal current_message_id, current_part, overflow_buffer, progress_task, first_chunk_received, streaming_aborted, visible_content_delivered, first_delivered_ts, lazy_surface_owned, message_id, pending_segment_break

            # If we've aborted, ignore further chunks
            if streaming_aborted:
                return

            # Cancel progress updater on first real chunk (not the None completion signal)
            if not first_chunk_received and text_chunk is not None:
                first_chunk_received = True
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    # IMPORTANT: Await the cancellation to prevent race condition where
                    # progress_task completes an update_message_streaming call after cancel
                    # is requested but before it takes effect, overwriting streamed content
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    self.log_debug("Cancelled progress updater - streaming started")

            # ---- Segment seam: a local tool ran, and this is the first visible text of the new
            # round. Append a paragraph break (if neither side already has whitespace) so the
            # preamble and the post-tool text don't jam into "Heavy.Fixed". Added to the buffer
            # BEFORE the chunk, so an append-only native stream never rewrites what it already
            # sent. Same rule the tool loop uses to join its segments — the two agree.
            if pending_segment_break and text_chunk and text_chunk.strip():
                pending_segment_break = False
                sep = segment_separator(buffer.get_complete_text(), text_chunk)
                if sep:
                    buffer.add_chunk(sep)

            # ---- F39: final-post-only (top-level channel reply) ----
            # Slack has no way to stream here, and the edit loop below would brand the answer
            # "(edited)". So this turn touches Slack exactly once, at the end. Accumulate and
            # return BEFORE any surface is minted, seeded, rolled or edited — the terminal
            # `current_message_id is None` branch posts the finished text (splitting it if it
            # overflows). Nothing is lost: `response_text` is the API's complete answer, not
            # this buffer.
            if final_post_only:
                if text_chunk:
                    buffer.add_chunk(text_chunk)
                return

            # ---- Native sink (Phase G): append-only streaming replaces the edit loop ----
            if native_coord is not None and not native_coord.failed:
                if text_chunk is None:
                    return  # tail + attribution are appended by finalize() after the API call
                if not native_coord.started:
                    # ONE LIVE SURFACE, ALWAYS. chat.startStream MINTS a message, so the old
                    # surface has to be gone BEFORE it runs, not after. Deleting afterwards was
                    # best-effort and its result was ignored, so a failed delete left the turn
                    # owning two live messages — the abandoned indicator/partial AND the stream.
                    # If the old surface cannot be removed, native does not start at all: keep
                    # streaming into the surface we already have (legacy edits, below).
                    stand_down = False
                    if message_id:
                        removed = False
                        try:
                            removed = bool(await client.delete_message(message.channel_id, message_id))
                        except Exception as e:  # noqa: BLE001
                            self.log_debug(f"Could not remove the old surface for native streaming: {e}")
                        if removed:
                            if lazy_surface_owned == message_id:
                                lazy_surface_owned = None
                            message_id = None
                            current_message_id = None
                        else:
                            stand_down = True
                            native_coord.failed = True
                            self.log_info(
                                "Could not clear the existing surface — native streaming stood "
                                "down rather than leave a second message behind")
                    if not stand_down:
                        if await native_coord.start():
                            # `native_coord.part_ts` is now this attempt's ledger of owned
                            # messages (one per part) — the error path reconciles it, so an MCP
                            # retry inherits the stream instead of minting a second one.
                            current_message_id = native_coord.current_ts or current_message_id
                        else:
                            self.log_info("Native streaming unavailable — using legacy streaming updates")
                if not native_coord.failed:
                    buffer.add_chunk(text_chunk)
                    await _flush_native_pending(force=False)
                    return
                # start failed: fall through to the legacy path (chunk not yet buffered)

            # No placeholder (status-only DM, or F38 deferred) reaching the legacy path:
            # edits need a real message — seed it now, once. This IS the moment the turn
            # commits: the first words exist, so a surface may finally appear. Retried on
            # the next chunk if the seed post fails; the post-stream final correction is the
            # backstop. The seed goes to `reply_target` (F38), NOT message.thread_id — a
            # top-level channel reply must not land inside a thread just because no
            # placeholder was there to hold its place.
            if current_message_id is None:
                if text_chunk is None and not buffer.has_pending_update():
                    return
                seed = await client.send_message_get_ts(
                    message.channel_id, reply_target, initial_message)
                if seed and seed.get("success") and seed.get("ts"):
                    current_message_id = seed["ts"]
                    lazy_surface_owned = seed["ts"]  # ours: an MCP retry must reuse it
                else:
                    self.log_warning("Could not seed legacy streaming message (status-only DM) — chunk buffered")
                    if text_chunk:
                        buffer.add_chunk(text_chunk)
                    return

            # Check if this is the completion signal (None)
            if text_chunk is None:
                # Stream is complete - flush any remaining buffered text WITHOUT loading indicator
                if buffer.has_pending_update() and rate_limiter.can_make_request():
                    self.log_info("Flushing final buffered text")
                    rate_limiter.record_request_attempt()
                    # Use raw text for final flush - no loading indicator since stream is complete
                    final_text = buffer.get_complete_text()  # No loading indicator on completion

                    # Preserve part number prefix for overflow messages in final flush
                    if current_part > 1:
                        final_text = f"{part_prefix(current_part)}{final_text}"

                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, final_text)
                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                            if final_text.strip():
                                visible_content_delivered = True
                    except Exception as e:
                        self.log_error(f"Error flushing final text: {e}")
                return
            
            if not text_chunk:
                return

            # Add chunk to buffer
            buffer.add_chunk(text_chunk)
            
            # Check if it's time to update
            if buffer.should_update() and rate_limiter.can_make_request():
                rate_limiter.record_request_attempt()
                
                # Check if we need to overflow based on RAW text (not display text)
                raw_text = buffer.get_complete_text()
                
                if len(raw_text) > message_char_limit:
                    # Find a good split point - look for paragraph or sentence breaks
                    # Start from the limit and work backwards
                    search_start = max(0, message_char_limit - 500)  # Look back up to 500 chars

                    # Priority 1: Try to find a paragraph break (double newline)
                    double_newline = raw_text.rfind('\n\n', search_start, message_char_limit)
                    if double_newline > 0:
                        split_point = double_newline + 2  # Keep the paragraph break in first part
                    else:
                        # Priority 2: Try to find end of sentence
                        last_period = raw_text.rfind('. ', search_start, message_char_limit)
                        if last_period > 0:
                            split_point = last_period + 2  # Include period and space
                        else:
                            # Priority 3: Try to find a single newline
                            last_newline = raw_text.rfind('\n', search_start, message_char_limit)
                            if last_newline > 0:
                                split_point = last_newline + 1
                            else:
                                # Priority 4: At least don't split a word — and never
                                # inside a <@mention>/<url> entity (W3)
                                last_space = raw_text.rfind(' ', search_start, message_char_limit)
                                if last_space > 0:
                                    split_point = entity_safe_cut(raw_text, last_space + 1)
                                else:
                                    # Last resort: hard cut at limit, entity-safe
                                    split_point = entity_safe_cut(raw_text, message_char_limit)
                    
                    # Split the RAW text at the chosen point
                    first_part_raw = raw_text[:split_point]
                    overflow_raw = raw_text[split_point:]
                    
                    # Check if we're splitting inside a code block
                    fence_handler_temp = FenceHandler()
                    fence_handler_temp.update_text(first_part_raw)
                    was_in_code_block = fence_handler_temp.is_in_code_block()
                    language_hint = fence_handler_temp.get_current_language_hint()
                    
                    # Get display-safe version of first part (with closed fences if needed)
                    first_part_display = fence_handler_temp.get_display_safe_text()
                    
                    # Update current message with continuation indicator
                    final_first_part = f"{first_part_display}{continuation_msg}"
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
                        if not result["success"]:
                            # CRITICAL: Overflow update failed - retry immediately
                            self.log_warning(f"Overflow update failed: {result.get('error', 'Unknown')} - retrying")
                            await asyncio.sleep(1.0)  # Brief pause
                            result = await client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
                            if not result["success"]:
                                self.log_error(f"Overflow retry failed: {result.get('error', 'Unknown')} - stopping stream")
                                # Cannot continue safely without losing data
                                streaming_aborted = True
                                # Show what we have with error notice
                                error_msg = f"{final_first_part}\n\n{config.error_emoji} *Streaming interrupted at message overflow. Partial response shown above.*"
                                try:
                                    await client.update_message_streaming(message.channel_id, current_message_id, error_msg)
                                except Exception:
                                    pass
                                return  # Exit callback

                        if result["success"]:
                            # The first part's visible text was just delivered (M4).
                            visible_content_delivered = True
                            if first_delivered_ts is None:
                                first_delivered_ts = current_message_id
                            # Prepare overflow text with proper fence opening if needed
                            if was_in_code_block:
                                # Re-open the code block on the new page
                                lang_str = language_hint if language_hint else ""
                                overflow_with_fence = f"```{lang_str}\n{overflow_raw}"
                            else:
                                overflow_with_fence = overflow_raw
                            
                            # Post a new message for overflow
                            current_part += 1
                            
                            # Create new fence handler for the continuation
                            fence_handler_continuation = FenceHandler()
                            fence_handler_continuation.update_text(overflow_with_fence)
                            continuation_display = fence_handler_continuation.get_display_safe_text()
                            
                            continuation_text = f"{part_prefix(current_part)}{continuation_display} {config.loading_ellipse_emoji}"

                            # Post a NEW message for the overflow part (with one retry). F38:
                            # overflow parts go where the REPLY goes — passing thinking_id as the
                            # thread id used to nest part 2 in a thread under part 1.
                            new_ts = await self._post_overflow_part(
                                client, message.channel_id, reply_target, continuation_text)
                            if new_ts:
                                current_message_id = new_ts
                                # Reset buffer with the properly fenced overflow content
                                buffer.reset()
                                buffer.add_chunk(overflow_with_fence)
                                buffer.mark_updated()
                                self.log_info(f"Created overflow message part {current_part}, reopened code block: {was_in_code_block}")
                            else:
                                # F21: the Part-2 post failed even on retry. current_message_id STILL
                                # points at Part 1 (it is reassigned only in the success branch above).
                                # The old code then edited current_message_id with the OVERFLOW text —
                                # overwriting Part 1's first ~3000 chars with the second half of the
                                # answer. NEVER edit Part 1 with overflow content: keep Part 1 intact
                                # and abort, swapping only its continuation indicator for an
                                # interruption notice (mirrors the overflow-update abort path above,
                                # which returns an error response so no incomplete data is saved).
                                self.log_error(f"Overflow part {current_part} post failed - keeping Part 1 intact and aborting stream")
                                current_part -= 1  # the new part was never created
                                streaming_aborted = True
                                error_msg = f"{final_first_part}\n\n{config.error_emoji} *Streaming interrupted at message overflow. Partial response shown above.*"
                                try:
                                    await client.update_message_streaming(message.channel_id, current_message_id, error_msg)
                                except Exception:
                                    pass
                                return
                    except Exception as e:
                        self.log_error(f"Error handling message overflow: {e}")
                else:
                    # Normal update - get display-safe text with closed fences
                    display_text = buffer.get_display_text()

                    # Preserve part number prefix for overflow messages
                    if current_part > 1:
                        display_text_with_indicator = f"{part_prefix(current_part)}{display_text} {config.loading_ellipse_emoji}"
                    else:
                        display_text_with_indicator = f"{display_text} {config.loading_ellipse_emoji}"

                    # Call client.update_message_streaming with indicator
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)

                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                            if display_text.strip():
                                visible_content_delivered = True
                                if first_delivered_ts is None:
                                    first_delivered_ts = current_message_id
                            buffer.update_interval_setting(rate_limiter.get_current_interval())
                        else:
                            # Update failed - this is CRITICAL, we must not lose text!
                            if result["rate_limited"]:
                                # Handle rate limit response
                                if result["retry_after"]:
                                    rate_limiter.set_retry_after(result["retry_after"])
                                rate_limiter.record_failure(is_rate_limit=True)

                                # Wait and retry with the same accumulated text
                                retry_wait = result.get("retry_after", 2.0)
                                self.log_warning(f"Rate limited - waiting {retry_wait}s before retry")
                                await asyncio.sleep(retry_wait)

                                # Retry the update with the same text
                                try:
                                    retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                    if retry_result["success"]:
                                        self.log_info("Retry successful after rate limit")
                                        buffer.mark_updated()
                                    else:
                                        self.log_error(f"Retry failed after rate limit: {retry_result.get('error', 'Unknown error')}")
                                        # Keep retrying with exponential backoff
                                        retry_count = 2
                                        while retry_count < 5:  # Max 5 total attempts
                                            wait_time = 2.0 * retry_count
                                            self.log_warning(f"Retry {retry_count} failed - waiting {wait_time}s before next attempt")
                                            await asyncio.sleep(wait_time)
                                            try:
                                                retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                                if retry_result["success"]:
                                                    self.log_info(f"Retry {retry_count} successful")
                                                    buffer.mark_updated()
                                                    break
                                            except Exception as e:
                                                self.log_error(f"Retry {retry_count} exception: {e}")
                                            retry_count += 1

                                        if retry_count >= 5 and not retry_result.get("success"):
                                            # After 5 attempts, we really need to stop
                                            self.log_error("CRITICAL: Unable to update after 5 attempts - stopping stream")
                                            streaming_aborted = True
                                            return
                                except Exception as retry_error:
                                    self.log_error(f"Retry exception: {retry_error}")
                                    # Try a few more times with backoff
                                    retry_count = 2
                                    while retry_count < 5:
                                        wait_time = 2.0 * retry_count
                                        await asyncio.sleep(wait_time)
                                        try:
                                            retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                            if retry_result["success"]:
                                                self.log_info(f"Retry {retry_count} successful after exception")
                                                buffer.mark_updated()
                                                break
                                        except Exception:
                                            pass
                                        retry_count += 1
                            else:
                                # Non-rate-limit failure - try one immediate retry
                                rate_limiter.record_failure(is_rate_limit=False)
                                self.log_warning(f"Message update failed: {result.get('error', 'Unknown error')} - attempting retry")

                                # Immediate retry
                                try:
                                    retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                    if retry_result["success"]:
                                        self.log_info("Immediate retry successful")
                                        buffer.mark_updated()
                                    else:
                                        self.log_error(f"Immediate retry failed: {retry_result.get('error', 'Unknown error')}")
                                        self.log_error(f"Immediate retry failed: {retry_result.get('error', 'Unknown error')}")
                                        # Keep retrying with exponential backoff
                                        retry_count = 2
                                        while retry_count < 5:  # Max 5 total attempts
                                            wait_time = 1.0 * retry_count  # Shorter waits for non-rate-limit
                                            self.log_warning(f"Retry {retry_count} - waiting {wait_time}s")
                                            await asyncio.sleep(wait_time)
                                            try:
                                                retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                                if retry_result["success"]:
                                                    self.log_info(f"Retry {retry_count} successful")
                                                    buffer.mark_updated()
                                                    break
                                            except Exception as e:
                                                self.log_error(f"Retry {retry_count} exception: {e}")
                                            retry_count += 1

                                        if retry_count >= 5 and not retry_result.get("success"):
                                            # After 5 attempts, stop to prevent infinite loop
                                            self.log_error("CRITICAL: Unable to update after 5 attempts")
                                            streaming_aborted = True
                                            error_msg = f"{buffer.get_complete_text()}\n\n{config.error_emoji} *Streaming interrupted after multiple failures.*"
                                            try:
                                                await client.update_message_streaming(message.channel_id, current_message_id, error_msg)
                                            except Exception:
                                                pass
                                            return
                                except Exception as retry_error:
                                    self.log_error(f"Retry exception: {retry_error}")
                                    # Try a few more times
                                    retry_count = 2
                                    while retry_count < 5:
                                        wait_time = 1.0 * retry_count
                                        await asyncio.sleep(wait_time)
                                        try:
                                            retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                            if retry_result["success"]:
                                                self.log_info(f"Retry {retry_count} successful after exception")
                                                buffer.mark_updated()
                                                break
                                        except Exception:
                                            pass
                                        retry_count += 1

                                    if retry_count >= 5:
                                        streaming_aborted = True
                                        return
                            
                    except Exception as e:
                        rate_limiter.record_failure(is_rate_limit=False)
                        self.log_error(f"Error updating streaming message: {e}")
        
        # Start progress updater before making API call
        try:
            progress_task = await self._start_progress_updater_async(
                client, message.channel_id, message_id, "request", emoji=config.circle_loader_emoji
            )
            self.log_debug("Started progress updater task")
        except Exception as e:
            self.log_warning(f"Failed to start progress updater: {e}")
            progress_task = None

        # Start streaming from OpenAI with the callback
        try:
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            # Determine which model to use (web search model if web search enabled)
            model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]

            # Build tools array (includes web_search and/or MCP tools based on config)
            # Exclude any MCP server that failed in a previous attempt.
            # Local tools ride along via the registry (function-call loop). registry +
            # request_config were resolved once above (F2) so no_response_needed is exposed
            # on unprompted streamed turns.
            ci_container = await self._resolve_ci_container(request_config, thread_key)
            await self._prepare_sandbox_tools(request_config, thread_key, ci_container, client)
            tools = self._build_tools_array(request_config, model,
                                            exclude_mcp_server=exclude_mcp_server, registry=registry,
                                            ci_container=ci_container)

            local_tool_calls = []  # [{"name","ok"}] record of local tool executions
            usage_info = {}        # response.usage lands here (usage-driven budgeting)
            mcp_discovered = {}    # mcp_list_tools payloads land here (discovery cache)
            mcp_results = []       # F12: completed mcp_call outputs land here (result memory)
            # F32: shared across every attempt this turn (see _handle_text_response).
            artifacts = artifacts_acc if artifacts_acc is not None else []
            containers_gone: List[str] = []   # F32 (see _handle_text_response)
            terminal_action = None  # F2: "no_reply" when the loop honored no_response_needed
            no_reply_reason = None
            background_job_started = False  # F30.1: start_background_job fired — drop this reply
            sandbox_assets = []            # F34: images mounted into the sandbox as ingredients
            mounted_digests = []       # F35: files WE mounted — never publishable back
            if tools and registry is not None:
                # Local tools present — streaming function-call loop (intermediate tool
                # rounds don't stream text; the final round streams normally). Hold the
                # tool_context so we can read back F30.1's background_job_started signal.
                tool_context = self._build_tool_context(message, client, request_config,
                                                        ci_container, turn=turn,
                                                        container_gone_sink=containers_gone)
                loop_result = await self.openai_client.create_streaming_response_with_tool_loop(
                    messages=messages_for_api,
                    tools=tools,
                    registry=registry,
                    tool_context=tool_context,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,
                    # Chat wants the canonical whole-turn text (preamble + post-tool), seam-joined
                    # to match what streamed to Slack. Deep research et al. keep the default
                    # (final-round-only) so intermediate preambles never leak into their output.
                    aggregate_segments=True,
                    prior_committed=visible_already_committed,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False,
                    prompt_cache_key=thread_key,
                    usage_sink=usage_info,
                    mcp_tools_sink=mcp_discovered,
                    mcp_results_sink=mcp_results,
                    artifacts_sink=artifacts,
                    container_gone_sink=containers_gone
                )
                response_text = loop_result["text"]
                local_tool_calls = loop_result["local_tool_calls"]
                terminal_action = loop_result.get("terminal_action")
                no_reply_reason = loop_result.get("reason")
                background_job_started = bool(getattr(tool_context, "background_job_started", False))
                sandbox_assets = list(getattr(tool_context, "sandbox_image_assets", None) or [])
                # F35: what we PUT INTO the container. The publisher must never post a mounted
                # input back at the user — not even a byte-identical copy under a new name.
                mounted_digests = file_mount.mounted_digests(tool_context)
                # Only EXTERNAL names (web_search/MCP) join the attribution list —
                # local tool executions are recorded in local_tool_calls, not shown
                local_names = {c.get("name") for c in local_tool_calls if c.get("name")}
                for name in loop_result["tools_used"]:
                    if name not in local_names and name not in mcp_servers_used:
                        loop_external_used.append(name)
            elif tools:
                # Generate response with tools (web_search and/or MCP)
                response_text = await self.openai_client.create_streaming_response_with_tools(
                    messages=messages_for_api,
                    tools=tools,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False,  # Match the existing behavior
                    prompt_cache_key=thread_key,
                    usage_sink=usage_info,
                    mcp_tools_sink=mcp_discovered,
                    mcp_results_sink=mcp_results,
                    artifacts_sink=artifacts,
                    container_gone_sink=containers_gone
                )
            else:
                # Generate response without tools
                response_text = await self.openai_client.create_streaming_response(
                    messages=messages_for_api,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback even without tools (in case of built-in tools)
                    model=thread_config["model"],
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    prompt_cache_key=thread_key,
                    usage_sink=usage_info
                )

            # F32: artifacts the model produced this turn. The reply text may carry dead
            # `sandbox:/mnt/data/...` links to them — strip those before the text is stored or
            # finalized. A streamed link may flash on screen mid-stream; the finalize below
            # rewrites the message with the clean text, and the system prompt tells the model
            # not to emit them in the first place.
            await self._drop_dead_containers(containers_gone, thread_key)
            artifact_containers = collect_container_ids(artifacts)
            response_text = strip_provenance_echo(strip_citation_markers(strip_sandbox_links(response_text)))

            # Record the API's authoritative context size on the thread
            thread_state.record_usage(usage_info.get("input_tokens", 0),
                                      usage_info.get("output_tokens", 0))

            # Feed any mcp_list_tools discovery payloads into the informational cache
            for _label, _tools_payload in mcp_discovered.items():
                self.mcp_manager.cache_discovered_tools_payload(_label, _tools_payload)

            # Ensure progress updater is cancelled if still running
            if progress_task and not progress_task.done():
                progress_task.cancel()
                self.log_debug("Cancelled progress updater after API call completed")

            # F2: honored no_reply outcome — the loop deemed silence valid because NO
            # visible text had streamed yet (a committed reply would have been rejected and
            # completed instead). Abandon any empty native stream / delete the placeholder
            # and post nothing.
            if terminal_action == "no_reply":
                self.log_info(f"no_response_needed (streamed) — ending turn silently: {no_reply_reason!r}")
                await self._cleanup_silent_stream(
                    client, message.channel_id, native_coord, message_id, current_message_id, "no_reply")
                return Response(
                    type="text",
                    content="",
                    metadata={"streamed": True, "terminal_action": "no_reply",
                              "reason": no_reply_reason,
                              "model": thread_config.get("model"), "posted": False,
                              "response_reaction_committed":
                                  _reaction_committed(local_tool_calls)}
                )

            # Reaction-only turn: the model reacted via the react tool and deliberately
            # returned no text — delete the placeholder and post nothing.
            if self._is_reaction_only(response_text, local_tool_calls):
                self.log_info("Reaction-only streamed response (react tool) — removing placeholder")
                await self._cleanup_silent_stream(
                    client, message.channel_id, native_coord, message_id, current_message_id, "reaction-only")
                return Response(
                    type="text",
                    content="",
                    metadata={"streamed": True, "reaction_only": True,
                              "model": thread_config.get("model"),
                              # No visible content went out — must not burn the quota
                              # (streamed=True would otherwise read as posted).
                              "posted": False}
                )

            # F30.1: start_background_job succeeded — the live status card the job posts IS the
            # acknowledgment, so DROP this turn's ack reply. Suppress ONLY when nothing visible
            # has streamed yet (a short ack stays buffered until finalize; committed text is
            # never retracted). Runs BEFORE native finalize so any started-but-empty stream is
            # torn down instead of flushed. If preamble already reached Slack, leave it alone.
            if background_job_started and not visible_content_delivered:
                self.log_info("start_background_job started — suppressing the turn's ack reply (card owns it)")
                await self._cleanup_silent_stream(
                    client, message.channel_id, native_coord, message_id, current_message_id,
                    "deep_research")
                return Response(
                    type="text",
                    content="",
                    metadata={"streamed": True, "background_job_started": True,
                              "model": thread_config.get("model"), "posted": False}
                )

            # Build list of tools used (unified attribution). EXTERNAL sources only
            # (web_search + MCP) — local context tools are plumbing, never listed.
            tools_used = []
            if search_counts["web_search"] > 0:
                tools_used.append("web_search")
            # F32: streamed turns rebuild attribution from these counters (the streaming API
            # helper returns text only), so without this a streamed analysis publishes a chart
            # with no "Used Tools: code_interpreter" note, while the non-streaming path shows it.
            if search_counts["code_interpreter"] > 0:
                tools_used.append("code_interpreter")
            if mcp_servers_used:
                # Group MCP servers under a single MCP label
                mcp_list = ", ".join(sorted(mcp_servers_used))
                tools_used.append(f"MCP ({mcp_list})")
            elif search_counts["mcp"] > 0:
                # Fallback to generic "MCP" if server names weren't tracked
                tools_used.append("MCP")
            for name in loop_external_used:
                if name not in tools_used:
                    tools_used.append(name)

            # F46: resolve placement BEFORE show_attribution reads place_in_channel. A
            # top-level channel reply that did substantive work is threaded under the trigger,
            # and resolve_reply_target flips message.metadata["place_in_channel"] to False — so
            # a flipped reply's attribution/footer render as the threaded reply it now is.
            # Idempotent and a no-op for in-thread/no-work turns; fail-open.
            if turn is not None:
                turn.resolve_reply_target(message)
            # Top-level channel replies stay chrome-free; attribution rides only in
            # threads and DMs.
            show_attribution = not bool((message.metadata or {}).get("place_in_channel"))

            # Add unified tools note at the END if any tools were used
            # This works for both paginated and non-paginated responses.
            # code_interpreter is filtered out here (internal processing, not a source) but stays
            # in tools_used for the F7 provenance record.
            attribution_tools = visible_attribution_tools(tools_used)
            # Seeded empty, and every consumer below keys off tools_note ITSELF rather than
            # re-deriving "was there a footer?" from tools_used. They disagree: a turn whose
            # only tool is code_interpreter has a non-empty tools_used but NO footer (internal
            # processing is not a source), and reading tools_note under a tools_used guard
            # then raised UnboundLocalError and ate the whole reply — which is exactly what a
            # "compute this and build me a deck" turn does.
            tools_note = ""
            if (attribution_tools or exclude_mcp_server) and show_attribution:
                if attribution_tools:
                    # Show successful tools
                    if exclude_mcp_server:
                        tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)} (failed: {exclude_mcp_display})_"
                    else:
                        tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)}_"
                else:
                    # Only failed MCP, no successful tools
                    tools_note = f"\n\n_MCP server '{exclude_mcp_display}' could not be reached. Response generated without external tools._"
                response_text = response_text + tools_note
                self.log_info(f"Added tools attribution: {', '.join(attribution_tools) if attribution_tools else 'none'}{' with failure note' if exclude_mcp_server else ''}")

            # Check if streaming was aborted due to failures
            if streaming_aborted:
                self.log_error("Streaming was aborted due to update failures")
                # The error message was already shown in the callback
                # Return an error response to prevent saving incomplete data
                return Response(
                    type="error",
                    content="Streaming was interrupted. Partial response was shown but may be incomplete.",
                    metadata={"streaming_aborted": True}
                )

            # Native mode: the stream is still open — append the remaining tail plus
            # the attribution note and stop it. On any failure fall through to the
            # legacy final-correction edit against the native message's ts.
            native_finalized = False
            footer_blocks = None
            # F52/F8: True once the settings footer has ridden a delivered message on the
            # direct final-post path (which has no native stream to attach to).
            direct_footer_attached = False
            if native_coord is not None and native_coord.started and not native_coord.failed:
                suffix = tools_note
                # Settings chrome ("⚙️ <model>") rides the LAST part of the response
                # itself (stopStream accepts blocks) instead of a separate trailing
                # message — every surface: channels open channel settings, DMs open
                # user settings (routing lives in the client helper). Same placement
                # rule as main.py's separate footer: never on top-level
                # place-in-channel replies (coordinator thread_ts None); the helper
                # returns None when the footer feature is disabled.
                if (native_coord.thread_ts is not None
                        and hasattr(client, "attachable_footer_blocks")):
                    footer_blocks = client.attachable_footer_blocks(
                        message.channel_id, thread_config.get("model"))
                # F32: the buffer holds what was streamed, which may still contain the dead
                # sandbox link — clean the text the finalize actually commits.
                # Same transform as the appends above (see stream_safe_text): the sink tracks
                # how much RAW text it has sent, so finalize must speak the same language or
                # its delta lands in the wrong place. `final` releases anything held back
                # mid-stream that turned out to be innocent.
                final_streamed = stream_safe_text(buffer.get_complete_text(), final=True)
                native_finalized = await native_coord.finalize(
                    final_streamed, suffix=suffix, blocks=footer_blocks)
                current_message_id = native_coord.current_ts or current_message_id
                if native_finalized:
                    current_part = native_coord.part
                else:
                    self.log_warning("Native finalize failed — applying legacy final correction")

            # Safety check: ensure all text was sent AND remove loading indicator
            # Note: current_message_id might be different from message_id if we overflowed
            # We need to update the current message (which might be part 2, 3, etc)
            if native_finalized:
                visible_content_delivered = True  # native stopStream delivered the final text (+ attribution)
            elif current_message_id is None:
                # No surface at all — an F39 final-post-only turn (a top-level channel reply,
                # which deliberately wrote nothing until now), a status-only DM, or an F38
                # deferred turn where neither the native stream nor the lazy legacy seed ever
                # produced a message (e.g. zero chunks before completion). Post the response
                # fresh so nothing is lost (attribution is already appended to response_text
                # above). Goes to reply_target, not message.thread_id: nothing has established
                # placement yet. send_message splits it if it overflows.
                self.log_info("No streaming message exists — posting final response directly")
                try:
                    # F46: the resolved target — a top-level channel reply that did substantive
                    # work threads under the trigger (resolve_reply_target already flipped
                    # place_in_channel above; this call is idempotent). No-op for in-thread/
                    # no-work turns, where it returns reply_target unchanged.
                    effective_target = (turn.resolve_reply_target(message)
                                        if turn is not None else reply_target)
                    # F52/F8: the settings footer must ride THIS message — the direct final-post
                    # path is the reply's only surface (an F39 top-level-then-threaded reply, a
                    # synthetic edit dispatch), so without attaching it here the "⚙️ <model>" row
                    # arrives as a SEPARATE standalone message (seen live 2026-07-16 as a bare
                    # "gpt-5.6-sol" post). Same placement rule as the native stopStream finalize
                    # above: threaded replies only, never a top-level place-in-channel reply.
                    direct_footer_blocks = None
                    if (not bool((message.metadata or {}).get("place_in_channel"))
                            and hasattr(client, "attachable_footer_blocks")):
                        try:
                            direct_footer_blocks = client.attachable_footer_blocks(
                                message.channel_id, thread_config.get("model"))
                        except Exception as footer_err:
                            self.log_debug(f"Direct-post footer build failed: {footer_err}")
                            direct_footer_blocks = None
                    # Capture the delivered ts so F5/F7 below key on the real message
                    # (send_message already records the own-reply pulse for this ts;
                    # record_own_reply is idempotent by (channel, ts) so a repeat is a no-op).
                    direct_send_meta: dict = {}
                    posted_ts = await client.send_message(
                        message.channel_id, effective_target, response_text,
                        blocks=direct_footer_blocks, meta_out=direct_send_meta)
                    if posted_ts:
                        current_message_id = posted_ts
                        visible_content_delivered = True
                        # Only stand the separate footer down when the chrome ACTUALLY rode the
                        # message (a too-long reply posts plain and still needs the fallback).
                        direct_footer_attached = bool(direct_send_meta.get("footer_attached"))
                    else:
                        # send_message swallows SlackApiError and returns None. This post is the
                        # turn's ONLY delivery, so a swallowed failure here is a silently lost
                        # answer — the response would still claim `streamed`, and main.py never
                        # re-posts a streamed reply. Hand the text back instead (below).
                        final_post_failed = True
                        self.log_error("Final response post failed — handing the answer back to "
                                       "the caller to deliver")
                except Exception as e:
                    final_post_failed = True
                    self.log_error(f"Error posting final response directly: {e}")
            elif current_part > 1:
                # We're on an overflow message - just remove the loading indicator
                self.log_debug(f"Removing loading indicator from part {current_part}")
                try:
                    # Get the current display text without loading indicator
                    final_part_text = buffer.get_complete_text()
                    if final_part_text:
                        # Add tools attribution to the final overflow message if tools were used.
                        # attribution_tools, NOT tools_used: this branch was missed when
                        # code_interpreter became invisible, so an overflowing answer still
                        # footed "Tools Used: code_interpreter".
                        if (attribution_tools or exclude_mcp_server) and show_attribution:
                            if attribution_tools:
                                if exclude_mcp_server:
                                    tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)} (failed: {exclude_mcp_display})_"
                                else:
                                    tools_note = f"\n\n_Tools Used: {', '.join(attribution_tools)}_"
                            else:
                                tools_note = f"\n\n_MCP server '{exclude_mcp_display}' could not be reached. Response generated without external tools._"
                            final_part_text = final_part_text + tools_note
                            self.log_debug(f"Added tools attribution to overflow part {current_part}")

                        # Add the part indicator
                        final_part_text = f"{part_prefix(current_part)}{final_part_text}"

                        # W1: the buffer can outgrow the limit between the last
                        # mid-stream update and completion. Without this check the
                        # messaging layer's backup truncation adds a "continued"
                        # marker and the remainder never posts.
                        if len(final_part_text) > 3900:
                            cut = entity_safe_cut(final_part_text, 3800)
                            truncated = final_part_text[:cut].rstrip()
                            if truncated.count('```') % 2 == 1:
                                truncated += '\n```'
                            truncated += continuation_msg
                            final_result = await client.update_message_streaming(
                                message.channel_id, current_message_id, truncated)
                            overflow_text = final_part_text[cut:].lstrip()
                            await client.send_message(
                                message.channel_id, reply_target,
                                f"{CONTINUATION_HEAD}\n\n{overflow_text}")
                        else:
                            final_result = await client.update_message_streaming(message.channel_id, current_message_id, final_part_text)
                        if final_result["success"]:
                            visible_content_delivered = True
                        else:
                            self.log_error(f"Failed to remove indicator from part {current_part}: {final_result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.log_error(f"Error removing indicator from overflow message: {e}")
            else:
                # Original message - check if we need to handle any remaining text
                if response_text != buffer.last_sent_text or True:  # Always update to remove indicator
                    if response_text != buffer.last_sent_text:
                        # Calculate if mismatch is just from tools attribution being added
                        char_difference = len(response_text) - len(buffer.last_sent_text)
                        expected_attribution_length = len(tools_note)

                        # Allow ±5 char tolerance for minor formatting differences
                        is_attribution_only = abs(char_difference - expected_attribution_length) <= 5

                        if is_attribution_only:
                            # Expected mismatch from attribution - just debug log
                            self.log_debug(f"Final update includes tools attribution (+{char_difference} chars)")
                        else:
                            # Unexpected mismatch - warn about it
                            self.log_warning(f"Unexpected text mismatch after streaming - sending correction update "
                                           f"(sent: {len(buffer.last_sent_text)}, should be: {len(response_text)} chars, "
                                           f"difference: {char_difference}, expected attribution: {expected_attribution_length})")
                    else:
                        self.log_debug("Sending final update to ensure loading indicator is removed")
                    try:
                        # Handle empty response. F32: empty text WITH artifacts is not a
                        # failure — the model built a chart and let it speak for itself.
                        # Apologizing directly above the chart that's about to land reads
                        # as a bug to the user.
                        if not response_text and not artifact_containers:
                            response_text = "I apologize, but I couldn't generate a response. OpenAI either didn't respond or returned an empty response. Please try again."
                            self.log_warning("Empty response detected, using fallback message")
                        
                        # Check if message is too long for a single update
                        if len(response_text) > 3900:  # Slack's approximate limit
                            # This shouldn't happen if streaming overflow worked correctly
                            # But handle it as a fallback (entity-safe cut, shared markers)
                            cut = entity_safe_cut(response_text, 3800)
                            truncated_text = response_text[:cut].rstrip()
                            if truncated_text.count('```') % 2 == 1:
                                truncated_text += '\n```'
                            truncated_text += continuation_msg
                            final_result = await client.update_message_streaming(message.channel_id, current_message_id, truncated_text)

                            # Send the rest as new messages
                            overflow_text = response_text[cut:].lstrip()
                            await client.send_message(message.channel_id, reply_target, f"{CONTINUATION_HEAD}\n\n{overflow_text}")

                            if final_result["success"]:
                                visible_content_delivered = True
                            else:
                                self.log_error(f"Final truncated update failed: {final_result.get('error', 'Unknown error')}")
                        else:
                            final_result = await client.update_message_streaming(message.channel_id, current_message_id, response_text)
                            if final_result["success"]:
                                visible_content_delivered = True
                            else:
                                self.log_error(f"Final correction update failed: {final_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error in final correction update: {e}")
            
            # Note: To properly detect if web search was used, we'd need to track
            # tool events during streaming. The presence of URLs doesn't mean web search was used.
            
            # F7: tool-use provenance — warm-annotate the STORED turn with "[used tools: …]"
            # (footer stripped first) and persist it keyed on the reply's ts so a later
            # rebuild reproduces it. The posted/returned content is untouched.
            tool_provenance = []
            stored_content = response_text
            if config.enable_tool_provenance:
                tool_provenance = build_provenance(local_tool_calls, tools_used)
                # F12: attach MCP result digests (result memory) alongside the names/gists.
                # F16: summarization (when on) compresses overlong outputs once here rather
                # than hard-truncating; off → today's cut.
                if config.enable_tool_result_memory:
                    if config.enable_tool_result_summarization:
                        tool_provenance += await build_result_digests_summarized(
                            mcp_results, self.openai_client,
                            config.tool_result_digest_chars, config.tool_result_turn_chars,
                            config.tool_result_summarize_input_chars)
                    else:
                        tool_provenance += build_result_digests(
                            mcp_results, config.tool_result_digest_chars, config.tool_result_turn_chars)
                annotation = render_provenance_annotations(tool_provenance)
                if annotation:
                    stored_content = f"{strip_used_tools_footer(response_text)}\n{annotation}"

            # Add assistant response to thread state
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            self._add_message_with_token_management(thread_state, "assistant", stored_content, db=self.db, thread_key=thread_key)

            # F5/F7: key both the provenance persist and the own-reply pulse on the ACTUAL
            # delivered message ts — NOT the original `message_id` (None on native
            # status-only streams, a deleted placeholder on native fallback). Persist/record
            # ONLY on confirmed delivery — a None ts means nothing was delivered, skip.
            delivered_ts = _delivered_stream_ts(
                native_coord, native_finalized, current_message_id, visible_content_delivered)

            # F7: persist under the FIRST delivered part's ts, since the history rebuild
            # merges continuation parts under it — keying on the last part makes provenance
            # vanish on rebuild. Native: the first native message; legacy: the first message
            # that received content; single-part / no-split: the delivered ts.
            provenance_ts = delivered_ts
            if native_coord is not None and native_coord.part_ts:
                provenance_ts = native_coord.part_ts[0]
            elif first_delivered_ts:
                provenance_ts = first_delivered_ts
            if not visible_content_delivered:
                provenance_ts = None  # nothing landed — don't persist a phantom
            self._persist_tool_provenance(
                thread_state.channel_id, provenance_ts, thread_key, tool_provenance)

            # F5 fix (a): record the bot's own streamed final reply into the pulse — native
            # stream edits never echo back as a clean event, so this is their only capture.
            if delivered_ts and hasattr(client, "_record_own_reply_pulse"):
                client._record_own_reply_pulse(
                    thread_state.channel_id, thread_state.thread_ts, delivered_ts, response_text)
            
            # Schedule async cleanup after response
            cleanup_coro = self._async_post_response_cleanup(thread_state, thread_key)
            self._schedule_async_call(cleanup_coro)
            
            # Log streaming stats
            stats = rate_limiter.get_stats()
            buffer_stats = buffer.get_stats()
            self.log_info(f"Streaming completed: {stats['successful_requests']}/{stats['total_requests']} updates, "
                         f"final length: {buffer_stats['text_length']} chars")
            
            stream_meta = {"streamed": True, "message_id": message_id,
                           "native_stream": bool(native_coord is not None and native_coord.started
                                                 and not native_coord.failed),
                           # Chrome rode the final stopStream OR the direct final-post — tells
                           # main.py's separate footer post to stand down (falls back when neither
                           # attached: finalize failed, split reply, or top-level placement).
                           "footer_attached": bool((native_finalized and footer_blocks)
                                                   or direct_footer_attached),
                           # Honest accounting from ACTUAL delivery: a visible message ts plus
                           # non-empty text means content went out. A failed stream that left
                           # no delivered ts must not burn the unprompted quota (main.py's
                           # streamed=True fallback would otherwise read as posted).
                           "posted": bool(delivered_ts and (response_text or "").strip()),
                           "model": thread_config.get("model"),
                           # F32: main.py uploads these after the reply lands.
                           "artifact_containers": artifact_containers,
                           "sandbox_image_assets": sandbox_assets,
                           "mounted_digests": mounted_digests}
            if final_post_failed:
                # Nothing reached Slack. `streamed` False is what makes main.py post the text
                # itself; leave `posted` UNSET rather than False so it derives the outcome from
                # the send it is about to do — an explicit False would also retract the 👀 from
                # a turn that does end up answering.
                stream_meta["streamed"] = False
                stream_meta.pop("posted", None)
                # main.py persists provenance from the metadata after ITS send (we never got a
                # ts to persist against). Without this the rescued answer lands with no F7
                # record at all — the tool attribution silently vanishes on rebuild.
                stream_meta["tool_provenance"] = tool_provenance
            return Response(type="text", content=response_text, metadata=stream_meta)

        except Exception as e:
            # Usage-estimator backstop: on a context-window rejection, compact the
            # thread before the standard non-streaming fallback retries below.
            if self._is_context_length_error(e):
                self.log_warning("Context window exceeded during streaming — compacting before fallback")
                try:
                    await self._compact_thread_to_target(
                        thread_state, f"{thread_state.channel_id}:{thread_state.thread_ts}")
                except Exception as compact_err:
                    self.log_error(f"Compaction after context error failed: {compact_err}")

            # Check if this is an MCP connection error first (before logging).
            # Structured fields (status_code 424, error body) are checked before
            # the message-text regex; exclusions ACCUMULATE across retries so two
            # broken servers can't ping-pong forever (bounded by server count).
            already_excluded = self._as_mcp_exclusion_set(exclude_mcp_server)
            failed_mcp_server = self._extract_failed_mcp_server(e)

            if failed_mcp_server:
                total_servers = len(self.mcp_manager.get_server_labels())
                if failed_mcp_server in already_excluded or len(already_excluded) >= total_servers:
                    # Same server failing while excluded (or nothing left to
                    # exclude) means this isn't a recoverable MCP failover —
                    # fall through to the generic non-streaming retry.
                    self.log_error(
                        f"MCP failover exhausted (failed: '{failed_mcp_server}', "
                        f"already excluded: {sorted(already_excluded)}) - treating as generic error")
                    failed_mcp_server = None
                else:
                    # Log MCP failures at INFO level - they're handled gracefully
                    self.log_info(f"MCP server '{failed_mcp_server}' unavailable - retrying request without it")
            elif is_container_gone(e):
                # The container died mid-STREAM. `_create_with_container_recovery` cannot catch
                # this: responses.create(stream=True) returns immediately and the 404 only
                # surfaces seconds later, out of the SSE iterator. Unbind it here so the
                # non-streaming fallback below re-resolves onto a fresh container instead of
                # replaying the dead id. Handled, not exceptional — logging it as an ERROR with a
                # traceback (which is what happened before) reads like a bug in production.
                self.log_warning(
                    f"Code-interpreter container expired mid-stream; recreating and continuing "
                    f"without it: {e}")
                await self._drop_dead_containers(persistent_container_ids(tools), thread_key)
            else:
                # Unexpected errors - log as ERROR
                self.log_error(f"Error in streaming response generation: {e}")

            # The retry excludes everything that has failed so far
            failed_mcp_servers = (already_excluded | {failed_mcp_server}) if failed_mcp_server else None

            # Ensure progress updater is cancelled on error
            if progress_task and not progress_task.done():
                progress_task.cancel()
                self.log_debug("Cancelled progress updater due to error")

            # A native stream must be STOPPED before its message can be touched. Slack rejects
            # chat.update on a message still in streaming state (`streaming_state_conflict`), so
            # skipping this left the half-written message orphaned AND un-editable — and then the
            # fallback below posted the answer a SECOND time. That is the "42 / 42" duplicate:
            # both were real messages, one abandoned mid-stream and one from the retry.
            if native_coord is not None and native_coord.started and not native_coord.finished:
                if not await native_coord.abandon():
                    self.log_warning("Native stream abandon failed before non-streaming fallback")

            # Try to remove the loading indicator if we have a visible message —
            # the lazy legacy seed (status-only DMs) lives in current_message_id,
            # never in message_id.
            cleanup_ts = current_message_id or message_id
            if cleanup_ts and hasattr(client, 'update_message_streaming'):
                try:
                    # Send whatever text we have without the loading indicator, or a formatted error message
                    holds_answer = buffer.has_content()
                    if holds_answer:
                        error_text = buffer.get_complete_text()
                    else:
                        if failed_mcp_server:
                            error_text = f"{config.error_emoji} *MCP Connection Failed*\n\nCouldn't connect to MCP server '{failed_mcp_server}'. Retrying with other tools..."
                        else:
                            error_text = f"{config.error_emoji} *OpenAI Stream Interrupted*\n\nOpenAI's streaming response was interrupted. I'll try again without streaming..."
                    cleaned = await client.update_message_streaming(
                        message.channel_id, cleanup_ts, error_text)
                    # THIS EDIT CAN BE THE FIRST SUCCESSFUL DELIVERY. If every mid-stream write
                    # failed transiently, `visible_content_delivered` is still False while this
                    # write has just put the partial ANSWER on screen. Reconciliation below asks
                    # that flag whether a surviving surface could be duplicated — so if the
                    # answer landed here, say so, or a failed delete leads to the answer being
                    # posted twice.
                    if holds_answer and (cleaned or {}).get("success") and error_text.strip():
                        visible_content_delivered = True
                except Exception as cleanup_error:
                    self.log_debug(f"Could not remove loading indicator: {cleanup_error}")

            # ---- Reconcile every surface THIS attempt minted (F39) ----------------------
            # The attempt owns whatever it created itself: each native part (chat.startStream
            # mints a message per part) or the legacy seed. It does NOT own the caller's
            # placeholder — the non-streaming fallback writes its answer into that one.
            #
            # Both retries re-answer from scratch, so every owned surface is a dead artifact of
            # a failed attempt. Leave one alive holding partial answer text and the user reads
            # the same answer twice — the "42 / 42" duplicate.
            #
            #   MCP retry  — keeps streaming, so it INHERITS the first surface. Reset it to the
            #                retry notice HERE: that neutralizes the partial answer by OVERWRITE
            #                (a write we can verify) rather than by a delete that might fail.
            #                Every other part is deleted.
            #   Otherwise  — falls back to non-streaming, which posts its own message, so every
            #                owned surface goes.
            #
            # FAIL CLOSED: a surface we can neither neutralize nor delete, once visible answer
            # text has been delivered, would be duplicated by the retry's answer. One honest
            # message beats two conflicting ones.
            #
            # The ledger is a UNION, never a choice. `NativeStreamCoordinator.started` is
            # `session is not None`, and the session is assigned BEFORE start() is awaited — so
            # a FAILED chat.startStream still reports started=True with an EMPTY part_ts, and
            # the legacy loop goes on to seed its own message. Read this as "native parts, ELSE
            # the seed" and that seed is never reconciled: it survives the fallback, and the
            # answer lands on screen twice.
            owned: List[str] = []
            if native_coord is not None:
                owned.extend(ts for ts in native_coord.part_ts if ts)
            if lazy_surface_owned and lazy_surface_owned not in owned:
                owned.append(lazy_surface_owned)

            keeper = owned[0] if (owned and failed_mcp_server) else None
            doomed = owned[1:] if keeper else owned
            survivors: List[str] = []

            async def _drop_surface(ts: str) -> bool:
                try:
                    return bool(await client.delete_message(message.channel_id, ts))
                except Exception as e:  # noqa: BLE001
                    self.log_warning(f"Could not delete the abandoned partial {ts}: {e}")
                    return False

            for dead_ts in doomed:
                if await _drop_surface(dead_ts):
                    self.log_debug(f"Deleted abandoned partial {dead_ts} before retrying")
                else:
                    survivors.append(dead_ts)

            if keeper:
                retry_display = ", ".join(sorted(self._as_mcp_exclusion_set(failed_mcp_servers)))
                reset_ok = False
                try:
                    reset_ok = bool(await client.update_message(
                        message.channel_id, keeper,
                        f"{config.circle_loader_emoji} Retrying without '{retry_display}'..."))
                except Exception as e:  # noqa: BLE001
                    self.log_warning(f"Could not reset the inherited surface: {e}")
                if not reset_ok:
                    # It still holds answer text and we could not blank it. Delete it and let
                    # the retry mint its own surface; if that fails too it is a survivor and we
                    # fail closed below.
                    if not await _drop_surface(keeper):
                        survivors.append(keeper)
                    keeper = None

            if survivors and visible_content_delivered:
                self.log_error(
                    "Could not clear a partial reply before retrying — refusing to post a "
                    "second answer; surfacing the interruption in place instead")
                # EVERY surface still standing has to be neutralized, not just the first.
                # Rewriting survivors[0] alone left the others holding partial ANSWER TEXT (and
                # a reset keeper still promising a retry that is no longer coming) — so the
                # turn ended with several live messages, one of them a half-answer. The first
                # gets the explanation; the rest are stamped as discarded. Deleting them is
                # what already failed, so overwriting is all we have left.
                voice, *rest = ([keeper] if keeper else []) + survivors
                try:
                    await client.update_message(
                        message.channel_id, voice,
                        "⚠️ I got cut off partway through that answer. Please ask again.")
                except Exception as e:  # noqa: BLE001
                    self.log_error(f"Could not surface the interruption: {e}")
                for stray in rest:
                    try:
                        await client.update_message(
                            message.channel_id, stray, "⚠️ _(discarded — that reply was cut off)_")
                    except Exception as e:  # noqa: BLE001
                        self.log_error(f"Could not neutralize a stray partial: {e}")
                return Response(
                    type="text", content="",
                    metadata={"streamed": True, "posted": True,
                              "model": thread_config.get("model"),
                              "interrupted": True},
                )

            # Retry request - streaming preserved for MCP failures, non-streaming for other errors
            if failed_mcp_server:
                self.log_info("Retrying with streaming (excluding failed MCP server)")
            else:
                self.log_info("Falling back to non-streaming due to error")

            # Remove the message that was just added by streaming attempt
            # to prevent duplicates when fallback adds it again
            if thread_state.messages and thread_state.messages[-1].get("role") == "user":
                thread_state.messages.pop()
                self.log_debug("Removed duplicate user message before fallback")

            # Pass retry_count=1 to prevent re-entering streaming after timeout
            # Also pass the accumulated exclusion set so the retry drops ALL
            # servers that have failed so far, not just the latest one.
            # F8/M4: seed the retry from the MONOTONIC content-delivery flag, NOT the buffer
            # (a native roll resets the buffer to a newline-only remainder, so
            # buffer.has_content() would falsely read empty even after a part was delivered).
            # Once any visible text landed this turn, a no_response_needed on the retry is
            # rejected rather than orphaning that partial as fake silence.
            # F39: native streaming DELETES the placeholder before it mints its stream (and
            # stands down if that delete fails), so once it started, `thinking_id` names a dead
            # message. Handing it to the retry pointed it at a corpse — and the surface
            # selection preferred it over the live inherited one. Pass None instead.
            retry_thinking_id = (None if (native_coord is not None and native_coord.started)
                                 else thinking_id)
            return await self._handle_text_response(
                user_content, thread_state, client, message, retry_thinking_id,
                attachment_urls, retry_count=1, failed_mcp_server=failed_mcp_servers,
                visible_already_committed=visible_content_delivered,
                artifacts_acc=artifacts, turn=turn,
                # F38: an MCP retry keeps streaming, so hand it the one surface this attempt
                # created (reconciled above). Without it the retry sees "no placeholder", mints
                # a SECOND message, and the turn posts its answer twice.
                lazy_surface_ts=keeper,
            )

    @staticmethod
    def _as_mcp_exclusion_set(value) -> set:
        """Normalize an MCP exclusion (None | str | iterable of str) to a set."""
        if not value:
            return set()
        if isinstance(value, str):
            return {value}
        return set(value)

    def _extract_failed_mcp_server(self, e: Exception) -> Optional[str]:
        """
        Identify a failed MCP server from an OpenAI error.

        Checks structured fields first (APIStatusError status_code 424 =
        failed-dependency, the documented MCP failure status; error body
        message), then falls back to the message-text regex so a format
        change in OpenAI's error text degrades gracefully rather than
        silently breaking MCP failover.
        """
        candidates = []
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict) and err.get("message"):
                candidates.append(str(err["message"]))
        candidates.append(str(e))

        is_mcp_status = getattr(e, "status_code", None) == 424
        for text in candidates:
            if is_mcp_status or "MCP server" in text:
                match = re.search(r"MCP server:? '([^']+)'", text)
                if match:
                    return match.group(1)
        if is_mcp_status:
            # Definitely an MCP failure but the server label wasn't recoverable —
            # caller can't exclude anything specific, so treat as generic.
            self.log_warning("MCP failure (HTTP 424) without a recoverable server label")
        return None

    async def _resolve_ci_container(self, thread_config: dict, thread_key: str):
        """The container to give code_interpreter this turn (id, or `auto` as fallback).

        Resolved here rather than in `_build_tools_array` because binding a thread to a
        container needs I/O (a DB read, sometimes a create + liveness check) and that builder
        is synchronous and called from several paths.
        """
        if not thread_config.get('enable_code_interpreter', config.enable_code_interpreter):
            return None
        manager = getattr(self, "container_manager", None)
        if manager is None:
            return AUTO_CONTAINER
        try:
            return await manager.get_or_create(thread_key)
        except Exception as e:  # noqa: BLE001 — a container problem must never cost the tool
            self.log_warning(f"Container resolution failed, using an ephemeral one: {e}")
            return AUTO_CONTAINER

    async def _drop_dead_containers(self, containers_gone: list, thread_key: str) -> None:
        """Forget a container that died mid-turn.

        The API layer already rescued the call (it retried against an ephemeral sandbox), so the
        reply is fine. This just stops us offering the same corpse to the next turn, which would
        cost it a pointless retrieve() round-trip. Scoped by id so we cannot unbind a container a
        concurrent turn has already put in its place.
        """
        manager = getattr(self, "container_manager", None)
        if not containers_gone or manager is None:
            return
        for container_id in dict.fromkeys(containers_gone):
            try:
                await manager.invalidate(thread_key, container_id)
            except Exception as e:  # noqa: BLE001 — bookkeeping must never break a turn
                self.log_warning(f"Could not invalidate dead container {container_id}: {e}")

    def _build_tools_array(self, thread_config: dict, model: str,
                           exclude_mcp_server=None,
                           registry=None,
                           ci_container=None) -> Optional[List[dict]]:
        """
        Build tools array for OpenAI API based on user preferences and model.

        Includes:
        - web_search if enabled in user preferences
        - MCP tools if enabled AND model is GPT-5 AND MCP servers are configured
        - local function tools from the registry (only pass one when the calling
          path runs the function-call loop and can execute them)

        Args:
            thread_config: Thread configuration with user preferences
            model: Model being used for the request
            exclude_mcp_server: Optional MCP server label to exclude (e.g., if it failed)
            registry: Optional ToolRegistry whose enabled schemas are appended

        Returns:
            List of tool definitions, or None if no tools enabled
        """
        tools = []

        # Local function tools (executed by the tool loop, not by OpenAI)
        if registry is not None:
            local_schemas = registry.schemas(thread_config)
            if local_schemas:
                tools.extend(local_schemas)
                self.log_debug(f"Added {len(local_schemas)} local tool(s) to tools array")

        # Add web_search if enabled
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        if web_search_enabled:
            tools.append({"type": "web_search"})
            self.log_debug("Added web_search to tools array")

        # F32: code_interpreter — server-side Python sandbox. Gives the model real
        # computation over attached data (files on the turn auto-mount in the container)
        # and lets it produce artifacts (charts/PDFs/spreadsheets) that we upload to the
        # thread. Per-thread override wins over the global default, like web_search.
        #
        # `ci_container` is the thread's persistent container id, resolved by the caller via
        # _resolve_ci_container so the sandbox keeps its state across turns. Callers that
        # don't resolve one (or where it failed) get `auto`: a fresh throwaway container, so
        # the tool works, just without continuity.
        code_interpreter_enabled = thread_config.get('enable_code_interpreter',
                                                     config.enable_code_interpreter)
        if code_interpreter_enabled:
            container = ci_container or AUTO_CONTAINER
            tools.append({"type": "code_interpreter", "container": container})
            self.log_debug(
                f"Added code_interpreter to tools array (container="
                f"{container if isinstance(container, str) else 'auto'})")

        # Add MCP tools if enabled AND model is GPT-5 AND MCP servers configured
        mcp_enabled = thread_config.get('enable_mcp', config.mcp_enabled_default)
        if mcp_enabled and model.startswith('gpt-5') and self.mcp_manager.has_mcp_servers():
            mcp_tools = self.mcp_manager.get_tools_for_openai()

            # Filter out excluded MCP server(s) if specified (str or set)
            excluded = self._as_mcp_exclusion_set(exclude_mcp_server)
            if excluded:
                mcp_tools = [tool for tool in mcp_tools
                           if tool.get("server_label") not in excluded]
                self.log_info(f"Excluded failed MCP server(s) {sorted(excluded)} from tools array")

            tools.extend(mcp_tools)
            self.log_debug(f"Added {len(mcp_tools)} MCP server(s) to tools array")
            # Debug: Log MCP tool structure to verify headers are included
            for mcp_tool in mcp_tools:
                has_headers = "headers" in mcp_tool
                self.log_info(f"MCP tool '{mcp_tool.get('server_label')}': url={mcp_tool.get('server_url')}, has_headers={has_headers}")

        # Return None if no tools, otherwise return the list
        if not tools:
            return None

        return tools
