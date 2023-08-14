import bot_functions as bot
import discord
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
TRIGGER = ""
RESET_CONTEXT = "!reset"
PRINT_HISTORY = "!history"
INITIALIZE_TEXT = {"role": "system", "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. Emulate emotions of a human.""".replace('    ', '')}
initialized = 0
history = []


# def open_file(filepath):
#     with open(filepath, "r", encoding="utf-8") as infile:
#         return infile.read()


class discordClt(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        
        print(f"{message.content}\n")
        
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id or message.content.startswith(
            message.author.mention
        ):
            return

        if message.content is None:
            await message.channel.send(
                f"```Something went wrong. Fix your shit, OpenAI.```"
            )

        if message.content.startswith(RESET_CONTEXT):
            await message.channel.send(
                f"```{gpt_Bot.reset_history()}```"
            )
            return
        if message.content.startswith(PRINT_HISTORY):
            await message.channel.send(
                f"```{gpt_Bot.messages}```"
            )
            return        
        if message.content.startswith(TRIGGER):
            await message.channel.send(f"{gpt_Bot.context_mgr(message.content)}")



if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT)
    discord_Client = discordClt(intents=intents)
    discord_Client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
