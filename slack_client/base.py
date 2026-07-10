"""Slack Bot Client Implementation."""
from typing import Optional, Callable, List

import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from base_client import BaseClient, Message
from config import config
from markdown_converter import MarkdownConverter
from database import DatabaseManager
from settings_modal import SettingsModal
from .event_handlers import (
    SlackAssistantEventsMixin,
    SlackMessageEventsMixin,
    SlackRegistrationMixin,
    SlackSettingsHandlersMixin,
)
from .utilities import SlackUtilitiesMixin
from .formatting.text import SlackFormattingMixin
from .messaging import SlackMessagingMixin
from .history_tool import SlackHistoryToolMixin
from .search_tool import SlackSearchToolMixin
from .channel_pulse import ChannelPulse
from tool_registry import ToolRegistry
from message_processor.memory_tools import register_memory_tools


class SlackBot(SlackMessageEventsMixin,
               SlackSettingsHandlersMixin,
               SlackRegistrationMixin,
               SlackAssistantEventsMixin,
               SlackUtilitiesMixin,
               SlackFormattingMixin,
               SlackMessagingMixin,
               SlackHistoryToolMixin,
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
        self.bot_id = None
        self.app_id = None

        # Initialize database manager
        self.db = DatabaseManager(platform="slack")

        # Initialize settings modal handler
        self.settings_modal = SettingsModal(self.db)

        # Local tools the model can call through the function-call loop (Phase A).
        # Flags are read at construction — flipping them requires a restart, like all env config.
        self.tool_registry = self._build_tool_registry()

        # Phase E: per-channel ambient-awareness ring buffer (process-lifetime, no DB).
        # None when disabled so consumers can simply `getattr(client, "channel_pulse", None)`.
        self.channel_pulse = ChannelPulse(size=config.channel_pulse_size) if config.enable_channel_pulse else None

        # Register Slack event handlers
        self._register_handlers()

    def _build_tool_registry(self) -> ToolRegistry:
        """Register Slack's local tools: history fetch (privacy-gated) + emoji reactions."""
        registry = ToolRegistry()
        for schema in self.get_history_tools_for_openai():  # [] when ENABLE_HISTORY_TOOLS is off
            name = schema["name"]
            registry.register(
                schema,
                lambda ctx, args, _name=name: self.dispatch_history_tool_call(_name, args),
            )
        if config.enable_reactions and config.enable_react_tool and config.reaction_emojis:
            registry.register(self.get_react_tool_schema(), self.execute_react_tool)
        if config.enable_search_tool:
            registry.register(self.get_search_tool_schema(), self.execute_search_tool)
        if config.enable_channel_memory:
            register_memory_tools(registry)  # channel-only; executors refuse DMs
        return registry

    # Async versions required by BaseClient
    async def send_message_async(self, channel_id: str, thread_id: str, text: str) -> bool:
        """Send a text message (async version)"""
        return await self.send_message(channel_id, thread_id, text)

    async def send_image_async(self, channel_id: str, thread_id: str, image_data: bytes, filename: str, caption: str = "") -> Optional[str]:
        """Send an image (async version)"""
        return await self.send_image(channel_id, thread_id, image_data, filename, caption)

    async def send_thinking_indicator_async(self, channel_id: str, thread_id: str) -> Optional[str]:
        """Send a thinking/processing indicator (async version)"""
        return await self.send_thinking_indicator(channel_id, thread_id)

    async def delete_message_async(self, channel_id: str, message_id: str) -> bool:
        """Delete a message (async version)"""
        return await self.delete_message(channel_id, message_id)

    async def update_message_async(self, channel_id: str, message_id: str, text: str) -> bool:
        """Update a message (async version)"""
        return await self.update_message(channel_id, message_id, text)

    async def get_thread_history_async(self, channel_id: str, thread_id: str, limit: int = None,
                                       oldest: str = None) -> List[Message]:
        """Get message history for a thread (async version)"""
        return await self.get_thread_history(channel_id, thread_id, limit, oldest=oldest)

    async def download_file_async(self, file_url: str, file_id: str = None) -> Optional[bytes]:
        """Download a file/image from the platform (async version)"""
        return await self.download_file(file_url, file_id)
    




















