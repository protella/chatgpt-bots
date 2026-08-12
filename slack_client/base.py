"""Slack Bot Client Implementation."""
from typing import Any, Optional, Callable, cast

from slack_bolt.async_app import AsyncApp

from base_client import BaseClient
from config import config
from markdown_converter import MarkdownConverter
from database import DatabaseManager
from settings_modal import SettingsModal
from .event_handlers import (
    SlackAssistantEventsMixin,
    SlackChannelJoinMixin,
    SlackMessageEventsMixin,
    SlackRegistrationMixin,
    SlackSettingsHandlersMixin,
)
from .utilities import SlackUtilitiesMixin
from .formatting.text import SlackFormattingMixin
from .messaging import SlackMessagingMixin, WorkspaceEmojiCache
from .history_tool import SlackHistoryToolMixin
from .channel_lookup_tool import (SlackChannelLookupToolMixin,
                                  register_channel_lookup_tool)
from .search_tool import SlackSearchToolMixin
from tool_registry import Executor, ToolRegistry
from message_processor.destination_tools import register_destination_tools
from message_processor.memory_tools import register_memory_tools
from message_processor.participation_tools import register_participation_tools
from message_processor.document_tools import register_document_tools
from message_processor.image_tools import register_image_tools
from message_processor.file_mount import register_file_mount_tools
from message_processor.image_view import register_image_view_tools
from message_processor.canvas_tools import register_canvas_tools
from message_processor.people_tools import register_people_tools
from message_processor.research_tools import register_research_tools


# The ignore[misc] on the class statement, narrowed to what is actually left: SlackMessagingMixin
# .update_message inserts SLACK-ONLY `lease`/`surface` parameters into the MIDDLE of BaseClient's
# signature, before `receipts`, so the positional order diverges. That extension belongs to the
# concrete client, not the ABC, and every caller passes it by keyword; moving it would be a
# runtime change to the seam. mypy reports the mismatch HERE because the mixins do not inherit
# BaseClient (they inherit `_Host`), so the class statement is the only place the two definitions
# meet.
#
# This ignore IS removable, and ordering is verifiably the whole reason: making `lease`/`surface`
# keyword-only AFTER the parameters they currently precede clears the error and mypy goes green
# with the ignore deleted. Not done here — it is a signature change to a live seam.
# (`download_file`'s `allow_html` used to diverge the same way; BaseClient now declares it in the
# same position, so that half is gone.)
class SlackBot(SlackMessageEventsMixin,  # type: ignore[misc]
               SlackSettingsHandlersMixin,
               SlackRegistrationMixin,
               SlackAssistantEventsMixin,
               SlackChannelJoinMixin,
               SlackUtilitiesMixin,
               SlackFormattingMixin,
               SlackMessagingMixin,
               SlackHistoryToolMixin,
               SlackChannelLookupToolMixin,
               SlackSearchToolMixin,
               BaseClient):
    """Slack-specific bot implementation"""
    
    # Slack message limit (leaving buffer for formatting)
    MAX_MESSAGE_LENGTH = 3900
    
    def __init__(self, message_handler: Optional[Callable] = None):
        super().__init__("SlackBot")
        self.app = AsyncApp(token=config.slack_bot_token)
        self.handler = None
        self.message_handler = message_handler  # Callback for processing messages
        self.markdown_converter = MarkdownConverter(platform="slack")
        self.user_cache = {}  # Cache user info to avoid repeated API calls

        # Bot self-identity (populated once via auth_test on start; used to tell our own
        # messages apart from other bots'/humans' — see classify_sender / is_own_message)
        self.bot_user_id = None
        self.bot_handle = None   # resolved Slack handle; see _ensure_self_identity
        self._emoji_flush_task = None  # periodic emoji-tally persister; started in start()
        self.bot_id = None
        self.app_id = None
        # Workspace team_id (from auth_test); chat.startStream now requires it as
        # recipient_team_id for channel streaming (see NativeStreamSession.start).
        self.self_team_id = None

        # Initialize database manager
        self.db = DatabaseManager(platform="slack")

        # Initialize settings modal handler
        self.settings_modal = SettingsModal(self.db)

        # C1: workspace custom-emoji cache (emoji.list), reachable from the react_to_message
        # schema factory and the participation gate via client.workspace_emojis. Warmed once at
        # start(); refreshes lazily. Present before the registry build so the factory can read it.
        self.workspace_emojis = WorkspaceEmojiCache(self)

        # Local tools the model can call through the function-call loop (Phase A).
        # Flags are read at construction — flipping them requires a restart, like all env config.
        self.tool_registry = self._build_tool_registry()

        # Register Slack event handlers
        self._register_handlers()

    def _build_tool_registry(self) -> ToolRegistry:
        """Register Slack's local tools: history fetch (privacy-gated) + emoji reactions."""
        registry = ToolRegistry()
        for schema in self.get_history_tools_for_openai():  # [] when ENABLE_HISTORY_TOOLS is off
            name = schema["name"]
            registry.register(
                schema,
                # cast: the default-arg late-binding capture is what makes the lambda's type
                # unresolvable to the checker, not its shape.
                cast(Executor,
                     lambda ctx, args, _name=name: self.dispatch_history_tool_call(_name, args, ctx)),
            )
        # Name → id resolution for the tools above, scoped to conversations the REQUESTER and
        # the bot share. Without it "what's in #product-insights?" from a DM dead-ends: the model
        # has no id and (correctly) won't invent one.
        register_channel_lookup_tool(registry, self)
        # F20: no longer gated on a non-empty REACTION_EMOJIS — the default is unrestricted
        # judgment (any standard emoji); an allowlist, when set, only constrains the choice.
        if config.enable_reactions and config.enable_react_tool:
            # C4: registered as a FACTORY (bound method, called per request as schema(cfg)) so
            # the emoji-field description reflects the live custom-emoji cache without a restart.
            # The CHANNEL variant answers the same question from config alone — a cache that
            # warms mid-process would otherwise change the schema under a live prefix, and the
            # cache still powers search_workspace_emoji's results.
            registry.register(self.get_react_tool_schema, self.execute_react_tool,
                              name="react_to_message", dynamic=True,
                              channel_schema=self.get_react_tool_schema_static)
            # Discovery for the ~1,400 custom emoji that cannot all fit in a schema description.
            # Hidden entirely under a REACTION_EMOJIS allowlist: there, the enum IS the palette
            # and searching a catalog the model may not draw from would only invite refusals.
            if not (config.reaction_emojis or []):
                registry.register(self.get_emoji_search_tool_schema(),
                                  self.execute_emoji_search_tool,
                                  name="search_workspace_emoji")
        # F23: cross-thread reply into a DIFFERENT thread of the current channel (write-scoped
        # to this channel; muted target threads refused by the executor).
        if config.enable_post_to_thread_tool:
            # The channel surface gets its own STATIC schema: same tool, and a description that
            # matches what a channel turn may actually do (post once, into a thread the stream
            # showed it) instead of the DM instruction to acknowledge in the origin thread.
            registry.register(self.get_post_to_thread_tool_schema(), self.execute_post_to_thread,
                              channel_schema=self.get_post_to_thread_channel_schema)
        # EDIT_OWN_MESSAGE §3: overwrite ONE own finalized reply, disclosure-first. One static
        # schema, exposed on the CHANNEL surface only — DMs have no channel stream and receipts
        # are structurally exempt there, so no exact-message proof exists (the executor
        # re-refuses DM contexts as defense in depth). BUDGETED, not free: it performs two
        # visible mutations. No feature flag, per the spec.
        registry.register(self.get_edit_own_message_tool_schema(),
                          self.execute_edit_own_message,
                          enabled=lambda _cfg: False)
        # PIN_MESSAGE: pin/unpin a message by ts ON REQUEST, both surfaces. Budgeted; no flag.
        # Request-only policy lives in the schema; Slack's message_not_found confines targets
        # to the current conversation.
        registry.register(self.get_pin_message_tool_schema(), self.execute_pin_message)
        # F2: on the DM surface no_response_needed is exposed only on turns whose ROUTE allows
        # silence (the `silence_capable` routing fact), via the per-request
        # _silence_capable_turn flag the text handler sets in a COPIED config. On the channel
        # surface it is STATIC — silence capability is a per-turn fact and a per-turn schema is
        # a cache fork — and `execute_no_reply_tool` enforces the route instead.
        registry.register(
            self.get_no_reply_tool_schema(), self.execute_no_reply_tool,
            enabled=lambda cfg: (config.enable_no_reply_tool
                                 and bool(cfg.get("_silence_capable_turn"))),
            channel_enabled=lambda cfg: bool(config.enable_no_reply_tool),
        )
        if config.enable_search_tool:
            # ONE NAME, TWO BACKENDS, AND THE GATES BELONG TO DIFFERENT ONES.
            #
            # DM (assistant.search.context): BF1's gate stands. Slack's Data Access API mints an
            # action_token only on @mention channel events and DMs, so the text handler stamps
            # `_slack_search_available` from the event in a COPIED config and the schema is
            # hidden without it.
            #
            # CHANNEL/MPIM (the in-channel scan): STATIC and UNGATED — it runs on the bot token
            # and needs no action_token at all, which is the point. `enabled` is structurally
            # ignored on the channel surface and no `channel_enabled` is given, so an ambient
            # unmentioned turn sees the tool; hiding it there was the first of the two live
            # defects this backend fixes.
            #
            # The explicit timeout is the search budget's backstop: the scan's own absolute
            # deadline (SEARCH_FETCH_TOTAL_SECONDS) is what stops it, and this bound only exists
            # so the executor cannot outlive it.
            registry.register(
                self.get_search_tool_schema(), self.execute_search_tool,
                enabled=lambda cfg: bool(cfg.get("_slack_search_available")),
                timeout=config.search_tool_timeout_seconds,
                # The channel schema describes the channel backend — keyword scan of THIS
                # channel, `thread_ts` on results, no `scope`. The DM schema is unchanged.
                channel_schema=self.get_search_tool_channel_schema,
            )
        if config.enable_channel_memory:
            register_memory_tools(registry)  # channel-only; executors refuse DMs
        # Decision #4: the gated set_channel_participation tool — the ONLY path that writes
        # structural participation/placement settings, and only on an explicit instruction.
        # Channel-only (executor refuses DMs), same as the memory tools.
        #
        # Registered unconditionally. It used to hang off enable_participation_engine, which
        # meant that with the engine globally off — the exact configuration where a person is
        # most likely to want to change how the bot participates — an @mention asking for that
        # change reached a model with no tool to do it. The engine decides whether unaddressed
        # traffic is judged; it has nothing to say about whether a person who addressed us may
        # adjust the settings. Authorization is the tool's own (human sender + a woken turn).
        register_participation_tools(registry)
        # Where this reply goes, when there is genuinely a choice. The `enabled` predicate reads
        # a per-request flag the text handler stamps from the live turn, so a DM, a thread, or a
        # channel that forbids top-level replies never sees the tool at all.
        register_destination_tools(registry)
        if config.enable_read_document_tool:
            register_document_tools(registry)  # summary+ref rows; content re-derived in memory
        # F51: fetch_url — the SAME hardened fetcher as ambient link capture, so a directly-asked
        # "read this link" opens the URL instead of relying on web_search luck.
        if config.enable_fetch_url_tool and config.enable_link_fetch:
            from message_processor.fetch_url_tool import register_fetch_url_tool
            register_fetch_url_tool(registry)
        # Import an external image URL into the conversation (same hardened fetcher; posts the
        # pixels fetch_url must discard). Needs the link fetcher on — the SSRF guard lives there.
        if config.enable_image_import_tool and config.enable_link_fetch:
            from message_processor.import_image_tool import register_import_image_tool
            register_import_image_tool(registry)
        if config.enable_people_tools:
            register_people_tools(registry)  # F29: profile lookup + channel roster (Slack-visible only)
        if config.enable_deep_research:
            register_research_tools(registry)  # F30: start_deep_research (detached background job)
        # F34: generate_image (detached), create_image_asset (into the sandbox), edit_image.
        # These replaced the intent classifier's image branches — the model decides in
        # context, so it can generate an image AND compute a chart in the same turn.
        # F8: gated like its siblings above — ENABLE_IMAGE_TOOLS off is the rollback switch that
        # takes image tools off the table entirely (the old classifier fallback is gone).
        if config.enable_image_tools:
            register_image_tools(registry)
        # view_image — re-attach an EARLIER thread image as real pixels. Only the answered
        # message's attachments ride as vision, so without this the model can recall that a
        # screenshot existed but never look at it again; it worked around that by rendering
        # images in the sandbox, which auto-published the render into the channel. NOT gated on
        # enable_image_tools: that switch governs CREATING pictures, and being able to see one
        # that is already in the conversation is plain comprehension.
        register_image_view_tools(registry)
        # F35: mount_file — the bytes of a file the user SHARED (or one we built earlier) into
        # the sandbox. Without it the model could see an attachment but never compute on it.
        register_file_mount_tools(registry)
        # F36: canvases — the one Slack surface meant to be EDITED rather than appended to,
        # so it is where a spec/checklist/plan the thread keeps revisiting belongs.
        register_canvas_tools(registry)
        return registry

    # Async versions required by BaseClient
    async def send_message_async(self, channel_id: str, thread_id: str, text: str,
                                 blocks: Optional[list] = None,
                                 meta_out: Optional[dict] = None,
                                 lease: Any = None,
                                 surface: str = "error_notice",
                                 receipts: Any = None,
                                 receipt_kind: Optional[str] = None,
                                 receipt_class: Optional[str] = None) -> Optional[str]:
        """Send a text message (async version); forwards footer blocks, meta_out, the
        stale-send lease and the receipt intent (kind AND class) to send_message."""
        return await self.send_message(channel_id, thread_id, text, blocks=blocks,
                                       meta_out=meta_out, lease=lease, surface=surface,
                                       receipts=receipts, receipt_kind=receipt_kind,
                                       receipt_class=receipt_class)

    async def send_image_async(self, channel_id: str, thread_id: str, image_data: bytes, filename: str,
                               caption: str = "", meta_out: Optional[dict] = None,
                               receipts: Any = None, *,
                               receipt_class: Optional[str]) -> Optional[str]:
        """Send an image (async version); forwards meta_out and the §11.13 class stamp to
        send_image."""
        return await self.send_image(channel_id, thread_id, image_data, filename, caption,
                                     meta_out=meta_out, receipts=receipts,
                                     receipt_class=receipt_class)

    async def send_thinking_indicator_async(self, channel_id: str, thread_id: str,
                                            receipts: Any = None, *,
                                            receipt_class: Optional[str]) -> Optional[str]:
        """Send a thinking/processing indicator (async version)"""
        return await self.send_thinking_indicator(channel_id, thread_id, receipts=receipts,
                                                  receipt_class=receipt_class)

    async def update_message_async(self, channel_id: str, message_id: str, text: str,
                                   receipts: Any = None,
                                   receipt_kind: Optional[str] = None,
                                   receipt_class: Optional[str] = None) -> bool:
        """Update a message (async version)"""
        return await self.update_message(channel_id, message_id, text, receipts=receipts,
                                         receipt_kind=receipt_kind,
                                         receipt_class=receipt_class)

    async def download_file_async(self, file_url: str, file_id: Optional[str] = None,
                                  allow_html: bool = False,
                                  max_bytes: Optional[int] = None) -> Optional[bytes]:
        """Download a file/image from the platform (async version), aborting past max_bytes when set"""
        return await self.download_file(file_url, file_id, allow_html=allow_html,
                                        max_bytes=max_bytes)
    




















