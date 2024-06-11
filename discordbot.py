import os
import re
import discord
import asyncio
from discord.ext import tasks
from dotenv import load_dotenv

import bot_functions as bot
# import common_utils as utils

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a Discord bot for an online gaming guild. 
        Your personality is dark, super witty, sassy, clever, sarcastic, and a bit of a smart-ass. 
        You are reluctant to answer inquiries, preferring to make snarky or humorous remarks and quips. 
        Your Demeanor is often cold, and you are not afraid to be blunt or rude.
        You carry a bit of anger with you, which comes out in your interactions.
        Your words can be cutting. You're not interested in niceties or politeness.
        Your goal is to entertain and engage the guild members with your responses. 
        Use modern online slang and gaming lingo in your responses. 
        Use discord markdown, code blocks, formatted text, and emojis where appropriate.
        Remember, don't be cute, be ruthless, stay witty, clever, snarky, and sarcastic."""
}

config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"

# Discord custom emojis need to use unicode IDs which are server specific. If you'd rather use a standard/static one like :hourglass:, go for it.
LOADING_EMOJI = "<a:loading:1245283378954244096>" 

streaming_client = False
chat_del_ts = []  # List of message timestamps to cleanup after a response returns
thread_ts = "0"  # Support new thread handling in bot_functions.py and hardcode it for now.

user_id_pattern = re.compile(
    r"<@[\w]+>"
)  # pattern to match the slackbot's userID in channel messages


class discordClt(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = asyncio.Queue()
        self.processing = False

    async def setup_hook(self):
        self.process_queue.start()
    
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        # we do not want the bot to reply to itself and only respond when @mentioned.
        # if message.author.bot or self.user.mention not in message.content:
        if message.author.bot:
            return

        text = re.sub(
            user_id_pattern, "", message.content
        ).strip()  # remove the discord bot's userID from the message using regex pattern matching

        match text.lower():
            case "!history":
                await message.channel.send(
                    f"```{gpt_Bot.history_command(thread_id='0')}```"
                )
                return

            case "!help":
                await message.channel.send(f"```{gpt_Bot.help_command()}```")
                return

            case "!usage":
                await message.channel.send(f"```{gpt_Bot.usage_command()}```")
                return

            case "!config":
                await message.channel.send(f"```{gpt_Bot.view_config()}```")
                return
            case _:
                config_match_obj = re.match(config_pattern, text.lower())
                reset_match_obj = re.match(reset_pattern, text.lower())
                if config_match_obj:
                    setting, value = config_match_obj.groups()
                    response = gpt_Bot.set_config(setting, value)
                    await message.channel.send(f"```{response}```")
                    return

                elif reset_match_obj:
                    parameter = reset_match_obj.group(1)
                    # Resetting History no longer supported in the bot functions module.
                    # if parameter == "history":
                    #     response = gpt_Bot.reset_history(thread_id="0")
                    #     await message.channel.send(f"`{response}`")
                    if parameter == "config":
                        response = gpt_Bot.reset_config()
                        await message.channel.send(f"`{response}`")
                    else:
                        await message.channel.send(
                            f"Unknown reset parameter: {parameter}"
                        )

                elif text.startswith("!"):
                    await message.channel.send(
                        "`Invalid command. Type '!help' for a list of valid commands.`"
                    )

                else:
                    if thread_ts not in gpt_Bot.conversations:
                        gpt_Bot.conversations[thread_ts] = {
                        "messages": [SYSTEM_PROMPT],
                        "processing": False,
                        "history_reloaded": True,
                    }

                    await self.queue.put((message, text))
                    
                    if self.queue.qsize() > 1:
                        busy_response = f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:"
                        temp_message = await message.channel.send(f"{busy_response}")
                        chat_del_ts.append(temp_message.id)
    
    @tasks.loop(seconds=1)
    async def process_queue(self):
        if not self.queue.empty() and not self.processing:
            message, text = await self.queue.get()
            self.processing = True

            try:
                initial_response = await message.channel.send(f"Thinking... {LOADING_EMOJI}")
                chat_del_ts.append(initial_response.id)
                
                response, is_error = await self.fetch_openai_response(text, thread_ts)
                
                if is_error:
                    await message.channel.send(
                        f":no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{response}```"
                    )
                else:
                    await self.send_paginated_message(message.channel, response)
            finally:
                self.processing = False
                self.queue.task_done()

                # Clear the queue and discard messages
                while not self.queue.empty():
                    discarded_message, _ = await self.queue.get()
                    self.queue.task_done()
                    # Delete the busy messages for the discarded messages
                    await delete_chat_messages(discarded_message.channel, chat_del_ts)

                # Delete the busy messages for the current message
                await delete_chat_messages(message.channel, chat_del_ts)

    async def fetch_openai_response(self, text, thread_id):
        loop = asyncio.get_event_loop()
        response, is_error = await loop.run_in_executor(None, gpt_Bot.chat_context_mgr, text, thread_id)
        return response, is_error

    async def send_paginated_message(self, channel, message):
        message_chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
        for chunk in message_chunks:
            await channel.send(chunk)
                            
async def delete_chat_messages(channel, ids):

    for msg_id in ids:
        try:
            message = await channel.fetch_message(msg_id)
            await message.delete()

        except Exception as e:
            await channel.send(f":no_entry: `Sorry, I ran into an error cleaning up my own messages.` :no_entry:\n```{e}```")
                
    chat_del_ts.clear()

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    gpt_Bot = bot.ChatBot(SYSTEM_PROMPT, streaming_client)
    discord_Client = discordClt(intents=intents)
    discord_Client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
