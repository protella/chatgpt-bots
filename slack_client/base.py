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
    SlackMessageEventsMixin,
    SlackRegistrationMixin,
    SlackSettingsHandlersMixin,
)
from .utilities import SlackUtilitiesMixin
from .formatting.text import SlackFormattingMixin
from .messaging import SlackMessagingMixin


class SlackBot(SlackMessageEventsMixin,
               SlackSettingsHandlersMixin,
               SlackRegistrationMixin,
               SlackUtilitiesMixin,
               SlackFormattingMixin,
               SlackMessagingMixin,
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

        # Initialize database manager
        self.db = DatabaseManager(platform="slack")

        # Initialize settings modal handler
        self.settings_modal = SettingsModal(self.db)

        # Register Slack event handlers
        self._register_handlers()

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

    async def get_thread_history_async(self, channel_id: str, thread_id: str, limit: int = None) -> List[Message]:
        """Get message history for a thread (async version)"""
        return await self.get_thread_history(channel_id, thread_id, limit)

    async def download_file_async(self, file_url: str, file_id: str = None) -> Optional[bytes]:
        """Download a file/image from the platform (async version)"""
        return await self.download_file(file_url, file_id)
    




















