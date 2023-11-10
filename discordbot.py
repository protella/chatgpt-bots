import bot_functions as bot
import discord
import os
from dotenv import load_dotenv
import re
from textwrap import dedent

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
INITIALIZE_TEXT = {
    "role": "system",
    "content": dedent(
        """\
        Act like the Jarvis AI assistant from the Ironman movies.
        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
        Keep your responses short, succinct, and to the point. Emulate emotions of a human."""
    ).replace("\n", " "),
}

config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"
streaming_client = False

user_id_pattern = re.compile(
    r"<@[\w]+>"
)  # pattern to match the slackbot's userID in channel messages


class discordClt(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        # we do not want the bot to reply to itself and only respond when @mentioned.
        if message.author.bot or self.user.mention not in message.content:
            return

        text = re.sub(
            user_id_pattern, "", message.content
        ).strip()  # remove the discord bot's userID from the message using regex pattern matching

        match text.lower():
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
                config_match_obj = re.match(config_pattern, text.lower())
                reset_match_obj = re.match(reset_pattern, text.lower())
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
                    content_type = "text"
                    await message.channel.send(
                        f"{gpt_Bot.context_mgr(text, content_type)}"
                    )


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT, streaming_client)
    discord_Client = discordClt(intents=intents)
    discord_Client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
