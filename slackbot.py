import bot_functions as bot
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from textwrap import dedent

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
INITIALIZE_TEXT = {
    "role": "system",
    "content": dedent(
        """\
        You are a helpful chatbot running in a corporate Slack workspace.
        Respond with accurate, informative, and concise answers that are formatted appropriately for Slack, 
        including markdown and special characters for bullet points, bold, italics, and code blocks as necessary. 
        Always consider Slack formatting conventions in all messages within a conversation.
        If you don't have an answer, you will inform the user that you don't know."""
    ).replace("\n", " "),
}
# INITIALIZE_TEXT = {
#     "role": "system",
#     "content": dedent(
#         """\
#         Act like the Jarvis AI assistant from the Ironman movies.
#         Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality.
#         Keep your responses short, succinct, and to the point. Emulate emotions of a human."""
#     ).replace("\n", " "),
# }

config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"
streaming_client = False
app = App(token=SLACK_BOT_TOKEN)


# @app.event('message')
# def handle_message(event, say):
#     print(event['user'])


def parse_text(text, say):
    match text.lower():
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
                content_type = "text"
                say(f"{gpt_Bot.context_mgr(text, content_type)}")


user_id_pattern = re.compile(
    r"<@[\w]+>"
)  # pattern to match the slackbot's userID in channel messages


@app.event("app_mention")
def handle_mention(event, say):
    text = re.sub(
        user_id_pattern, "", event["text"]
    ).strip()  # remove the slackbot's userID from the message using regex pattern matching
    parse_text(text, say)
    # print(event)


@app.event("message")
def handle_message_events(event, say):
    channel_type = event["channel_type"]
    if channel_type == "im":
        text = event["text"]
        parse_text(text, say)


if __name__ == "__main__":
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT, streaming_client)
    handler = SocketModeHandler(
        app,
        SLACK_APP_TOKEN,
    )

    handler.start()
