from __future__ import annotations

from config import config


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

        @self.app.event("assistant_thread_started")
        async def handle_assistant_thread_started(event, client):
            # Agent split-view opened: greet + set suggested prompts (best-effort, flag-gated).
            await self._handle_assistant_thread_started(event, client)

        @self.app.event("assistant_thread_context_changed")
        async def handle_assistant_thread_context_changed(event):
            await self._handle_assistant_thread_context_changed(event)

        # Register settings-related handlers
        self._register_settings_handlers()
