from __future__ import annotations

from config import config
from slack_client.event_handlers import feedback as feedback_handlers


class SlackRegistrationMixin:
    def _register_handlers(self):
        """Register Slack-specific event handlers."""

        @self.app.event("app_mention")
        async def handle_app_mention(event, say, client):
            self.log_debug(f"App mention event: channel={event.get('channel')}, ts={event.get('ts')}")
            await self._handle_slack_message(event, client)

        @self.app.event("message")
        async def handle_message(event, say, client):
            channel_type = event.get("channel_type")
            if channel_type == "im":
                # DMs from anyone except ourselves (other bots allowed so bot<->bot works).
                if not self.is_own_message(event):
                    self.log_debug(f"DM message event: channel={event.get('channel')}, ts={event.get('ts')}")
                    await self._handle_slack_message(event, client)
            elif channel_type in ("channel", "group", "mpim"):
                # Phase 5 channel listening — gated by the master switch (DEFAULT OFF). When off,
                # non-mention channel messages are ignored entirely (mentions still arrive via
                # the app_mention event above).
                if config.enable_channel_listening:
                    await self._handle_channel_message(event, client)

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

        # Phase H: native feedback buttons (context_actions block on DM/assistant
        # responses) arrive as ordinary block_actions.
        @self.app.action(feedback_handlers.FEEDBACK_ACTION_ID)
        async def handle_response_feedback(ack, body):
            await feedback_handlers.handle_feedback_action(self, ack, body)

        # Register settings-related handlers
        self._register_settings_handlers()
