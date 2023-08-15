import bot_functions as bot
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
# BOT_ID = "U01AF99F3JR"
# CHANNEL_ID = "C04PQ5BK946"
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"
# INITIALIZE_TEXT = {'role': 'system', 'content': '''You are a chatbot that answers questions with accurate,
#                    informative, witty, and humorous responses.'''.replace('    ', '')}
INITIALIZE_TEXT = {
    "role": "system",
    "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. Emulate emotions of a human.""".replace(
        "    ", ""
    ),
}


app = App(token=SLACK_BOT_TOKEN)


# @app.event('message')
# def handle_message(event, say):
#     print(event['user'])


@app.event("app_mention")
def handle_mention(event, say):
    text = event["text"][14:].lower().strip()
    # print(text)
    # if event['user'] == BOT_ID:
    #     return

    match text:
        case "!history":
            say(f"```{gpt_Bot.history_command()}```")
            return

        case "!help":
            say(f"```{gpt_Bot.help_command()}```")
            return

        case "!usage":
            say(f"```{gpt_Bot.usage_command()}```")
            return

        case "!config":
            say(f"```Current Configuration:\n{gpt_Bot.view_config()}```")
            return

        case _:
            config_match_obj = re.match(config_pattern, text)
            reset_match_obj = re.match(reset_pattern, text)
            if config_match_obj:
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value)
                say(f"```{response}```")
                return

            elif reset_match_obj:
                parameter = reset_match_obj.group(1)
                if parameter == "history":
                    response = gpt_Bot.reset_history()
                    say(f"`{response}`")
                elif parameter == "config":
                    response = gpt_Bot.reset_config()
                    say(f"`{response}`")
                else:
                    say(f"Unknown reset parameter: {parameter}")

            elif text.startswith("!"):
                say("`Invalid command. Type '!help' for a list of valid commands.`")

            else:
                say(f"{gpt_Bot.context_mgr(text)}")


if __name__ == "__main__":
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT)
    handler = SocketModeHandler(
        app,
        SLACK_APP_TOKEN,
    )

    handler.start()
