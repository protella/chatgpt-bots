import bot_functions as bot
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from textwrap import dedent

load_dotenv()

LOADING_EMOJI = ":loading:"
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

# pattern to match commands
config_pattern = re.compile(r"!config\s+(\S+)\s+(.+)")
reset_pattern = re.compile(r"^!reset\s+(\S+)$")

# pattern to match the slackbot's userID in channel messages
user_id_pattern = re.compile(r"<@[\w]+>")

streaming_client = False
app = App(token=SLACK_BOT_TOKEN)


# @app.event('message')
# def handle_message(event, say):
#     print(event['user'])


def parse_text(text):
    match text.lower():
        case "!history":
            return f"```{gpt_Bot.history_command()}```"

        case "!help":
            return f"```{gpt_Bot.help_command()}```"

        case "!usage":
            return f"```{gpt_Bot.usage_command()}```"

        case "!config":
            return f"```Current Configuration:\n{gpt_Bot.view_config()}```"

        case _:
            if config_match_obj := config_pattern.match(text):
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value)
                return f"```{response}```"

            elif reset_match_obj := reset_pattern.match(text):
                parameter = reset_match_obj.group(1)
                if parameter == "history":
                    response = gpt_Bot.reset_history()
                    return f"`{response}`"
                elif parameter == "config":
                    response = gpt_Bot.reset_config()
                    return f"`{response}`"
                else:
                    return f"Unknown reset parameter: {parameter}"

            elif text.startswith("!"):
                return "`Invalid command. Type '!help' for a list of valid commands.`"

            else:
                content_type = "text"
                return f"{gpt_Bot.context_mgr(text, content_type)}"


def process_and_respond(event, say):
    channel_id = event['channel']
    initial_response = say(f"Thinking... {LOADING_EMOJI}")
    initial_response_ts = initial_response['message']['ts']

    # remove the slackbot's userID from the message using regex pattern matching
    text = re.sub(user_id_pattern, "", event["text"]).strip()
    response = parse_text(text)

    try:
        app.client.chat_delete(channel=channel_id, ts=initial_response_ts)

    except Exception as e:
        say(":no_entry: `Sorry, I ran into an error deleting my own message.` :no_entry:")

    say(response)


@app.event("app_mention")
def handle_mention(event, say):
    process_and_respond(event, say)


@app.event("message")
def handle_message_events(event, say):
    if event["channel_type"] == "im":
        process_and_respond(event, say)


if __name__ == "__main__":
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT, streaming_client)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    handler.start()
