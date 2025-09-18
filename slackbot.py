#!/usr/bin/env python3
"""
Slack Bot Wrapper Script
Simple launcher for running the Slack bot directly
Usage: python slackbot.py
"""

import asyncio
from main import ChatBotV2


async def main():
    """Launch the Slack bot"""
    bot = ChatBotV2(platform="slack")
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())