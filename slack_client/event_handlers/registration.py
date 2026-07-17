from __future__ import annotations

from config import config
from slack_client.event_handlers import feedback as feedback_handlers


class SlackRegistrationMixin:
    def _register_handlers(self):
        """Register Slack-specific event handlers."""

        @self.app.event("app_mention")
        async def handle_app_mention(event, say, client):
            self.log_debug(f"App mention event: channel={event.get('channel')}, ts={event.get('ts')}")
            # F52: record this genuine Slack app_mention so the edit-reply path can tell that a
            # mention-added edit is already covered by Slack's own event (editing to add a mention
            # makes Slack deliver app_mention for the same ts) and skip a duplicate synthetic turn.
            if hasattr(self, "_note_app_mention_seen"):
                self._note_app_mention_seen(event.get("channel"), event.get("ts"))
            # F51: an @mention carries ambient content (images/links/files) too. Capture BEFORE
            # dispatch so it is kept even if the reply path drops it. Best-effort infra: guarded so
            # a registration host without the message-events mixin (test harnesses) still works.
            if hasattr(self, "_ambient_ingest"):
                await self._ambient_ingest(event, client)
            await self._handle_slack_message(event, client, wake_source="app_mention")

        @self.app.event("message")
        async def handle_message(event, say, client):
            # F51: ambient capture + lifecycle (edits/deletions) runs FIRST, independent of
            # channel_type and ENABLE_CHANNEL_LISTENING — memory is a distinct setting from
            # whether the bot replies. Never blocks the wake path (offer_event only enqueues).
            # Guarded for registration hosts without the message-events mixin (test harnesses).
            if hasattr(self, "_ambient_ingest"):
                await self._ambient_ingest(event, client)
            channel_type = event.get("channel_type")
            if channel_type == "im":
                # DMs from anyone except ourselves (other bots allowed so bot<->bot works).
                if not self.is_own_message(event):
                    self.log_debug(f"DM message event: channel={event.get('channel')}, ts={event.get('ts')}")
                    await self._handle_slack_message(event, client, wake_source="dm")
            elif channel_type in ("channel", "group", "mpim"):
                # Phase 5 channel listening — gated by the master switch (DEFAULT OFF). When off,
                # non-mention channel messages are ignored entirely (mentions still arrive via
                # the app_mention event above).
                if config.enable_channel_listening:
                    await self._handle_channel_message(event, client)

        @self.app.event("file_deleted")
        async def handle_file_deleted(event):
            # F51: a Slack file removed → purge summaries derived from it (best-effort).
            await self._ambient_file_deleted(event)

        # --- agent_view lifecycle (June 2026 surface, Phase G) ---
        @self.app.event("app_home_opened")
        async def handle_app_home_opened(event, client):
            # Messages tab opened: greet once per channel (tab filter + dedup inside).
            await self._handle_app_home_opened(event, client)

        @self.app.event("app_context_changed")
        async def handle_app_context_changed(event):
            await self._handle_app_context_changed(event)

        # --- LEGACY agent surface (deprecated by agent_view) ---
        # Keep during the transition (whichever fires, the greeting dedup makes it
        # fire once); remove one release after the manifest fully flips to agent_view.
        @self.app.event("assistant_thread_started")
        async def handle_assistant_thread_started(event, client):
            # Agent split-view opened: greet + set suggested prompts (best-effort, flag-gated).
            await self._handle_assistant_thread_started(event, client)

        @self.app.event("assistant_thread_context_changed")
        async def handle_assistant_thread_context_changed(event):
            await self._handle_assistant_thread_context_changed(event)

        @self.app.event("reaction_added")
        async def handle_reaction_added(event):
            # Phase H: passive feedback ingestion — thumbs reactions on OUR OWN
            # messages land in the response_feedback sink. Strictly recording:
            # no LLM call, no reply, never raises. Everything else is ignored
            # (acked so Bolt doesn't log every reaction as unhandled).
            await feedback_handlers.ingest_reaction(self, event)
            # F20 social proof: mirror the reaction onto the pulse ring (in-memory, additive).
            feedback_handlers.note_reaction_pulse(self, event, added=True)

        @self.app.event("reaction_removed")
        async def handle_reaction_removed(event):
            # F20: keep the pulse ring's social-proof counts in sync when reactions are
            # removed (only fires if the app is subscribed to reaction_removed).
            feedback_handlers.note_reaction_pulse(self, event, added=False)

        # Phase H: native feedback buttons (context_actions block on DM/assistant
        # responses) arrive as ordinary block_actions.
        @self.app.action(feedback_handlers.FEEDBACK_ACTION_ID)
        async def handle_response_feedback(ack, body):
            await feedback_handlers.handle_feedback_action(self, ack, body)

        # Register settings-related handlers
        self._register_settings_handlers()
