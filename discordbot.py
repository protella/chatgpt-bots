import bot_functions as bot
import discord
import os
from dotenv import load_dotenv
import re

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
INITIALIZE_TEXT = {
    "role": "system",
    "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. Emulate emotions of a human.""".replace(
        "    ", ""
    ),
}
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"


# def open_file(filepath):
#     with open(filepath, "r", encoding="utf-8") as infile:
#         return infile.read()


class discordClt(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        print(f"{message.content}\n")

        text = message.content

        # we do not want the bot to reply to itself
        if message.author.id == self.user.id or message.content.startswith(
            message.author.mention
        ):
            return

        match text:
            case "!history":
                await message.channel.send(f"```{gpt_Bot.history_command()}```")
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
                config_match_obj = re.match(config_pattern, text)
                reset_match_obj = re.match(reset_pattern, text)
                if config_match_obj:
                    setting, value = config_match_obj.groups()
                    response = gpt_Bot.set_config(setting, value)
                    await message.channel.send(f"```{response}```")
                    return

                elif reset_match_obj:
                    parameter = reset_match_obj.group(1)
                    if parameter == "history":
                        response = gpt_Bot.reset_history()
                        await message.channel.send(f"`{response}`")
                    elif parameter == "config":
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
                    await message.channel.send(
                        f"{gpt_Bot.context_mgr(message.content)}"
                    )


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT)
    discord_Client = discordClt(intents=intents)
    discord_Client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
