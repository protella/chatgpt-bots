#!/usr/bin/env python3
"""
Slack Bot Wrapper Script
Simple launcher for running the Slack bot directly
Usage: python slackbot.py
"""

import sys
from main import ChatBotV2


def main():
    """Launch the Slack bot"""
    bot = ChatBotV2(platform="slack")
    bot.run()


if __name__ == "__main__":
    main()