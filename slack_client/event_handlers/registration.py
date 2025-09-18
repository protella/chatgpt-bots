from __future__ import annotations

class SlackRegistrationMixin:
    def _register_handlers(self):
        """Register Slack-specific event handlers."""

        @self.app.event("app_mention")
        async def handle_app_mention(event, say, client):
            self.log_debug(f"App mention event: channel={event.get('channel')}, ts={event.get('ts')}")
            await self._handle_slack_message(event, client)

        @self.app.event("message")
        async def handle_message(event, say, client):
            # Only process DMs and non-bot messages
            if event.get("channel_type") == "im" and not event.get("bot_id"):
                self.log_debug(f"DM message event: channel={event.get('channel')}, ts={event.get('ts')}")
                await self._handle_slack_message(event, client)

        # Register settings-related handlers
        self._register_settings_handlers()
