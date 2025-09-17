"""Slack Bot Client Implementation."""
from typing import Optional, Callable

from slack_bolt import App

from base_client import BaseClient
from config import config
from markdown_converter import MarkdownConverter
from database import DatabaseManager
from settings_modal import SettingsModal
from .event_handlers.core import SlackEventHandlersMixin
from .utilities import SlackUtilitiesMixin
from .formatting.text import SlackFormattingMixin
from .messaging import SlackMessagingMixin


class SlackBot(SlackEventHandlersMixin,
               SlackUtilitiesMixin,
               SlackFormattingMixin,
               SlackMessagingMixin,
               BaseClient):
    """Slack-specific bot implementation"""
    
    # Slack message limit (leaving buffer for formatting)
    MAX_MESSAGE_LENGTH = 3900
    
    def __init__(self, message_handler: Optional[Callable] = None):
        super().__init__("SlackBot")
        self.app = App(token=config.slack_bot_token)
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
    





















