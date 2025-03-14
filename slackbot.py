import re
from os import environ
from prompts import SLACK_SYSTEM_PROMPT
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from markdown_to_mrkdwn import SlackMarkdownConverter

import bot_functions as bot
import common_utils as utils

# For performance profiling.
# import cProfile
# import pstats
# import io

load_dotenv()  # load auth tokens from .env file
mrkdown_converter = SlackMarkdownConverter()

### Modify these values as needed. Note the tokens should be put in the .env file. See README. ###
LOADING_EMOJI = ":loading:"
SLACK_BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = environ.get("SLACK_APP_TOKEN")
DALLE3_CMD = environ.get("DALLE3_CMD", "/dalle-3")

show_dalle3_revised_prompt = False

#
### You shouldn't need to modify anything below this line ###
#

# patterns to match commands
CONFIG_PATTERN = re.compile(r"!config\s+(\S+)\s+(.+)")
RESET_PATTERN = re.compile(r"^!reset\s+(\S+)$")
# pattern to match the slackbot's userID in channel messages
USER_ID_PATTERN = re.compile(r"<@[\w]+>")
STREAMING_CLIENT = False  # not implemented for Slack...yet.
# GPT4 vision supported image types
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

chat_del_ts = []  # list of message timestamps to cleanup after a response returns

app = App(token=SLACK_BOT_TOKEN)

# Call the auth.test method to capture bot info
response = app.client.auth_test()
# Extract the Slackbot's user ID
bot_user_id = response.get("user_id")


# Check the message text to see if a bot command was sent. Respond accordingly.
def parse_text(text, say, thread_ts, is_thread=False):
    if not is_thread:
        thread_ts = None

    match text.lower():
        case "!history":
            say(f"```{gpt_Bot.history_command(thread_ts)}```", thread_ts=thread_ts)

        case "!help":
            say(f"```{gpt_Bot.help_command()}```", thread_ts=thread_ts)

        case "!usage":
            say(f"```{gpt_Bot.usage_command()}```", thread_ts=thread_ts)

        case "!config":
            say(
                f"```Current Configuration:\n{gpt_Bot.view_config(thread_ts)}```",
                thread_ts=thread_ts,
            )

        case _:
            if config_match_obj := CONFIG_PATTERN.match(text.lower()):
                setting, value = config_match_obj.groups()
                print(f"CONFIG CHANGE: {thread_ts}\n")
                response = gpt_Bot.set_config(setting, value, thread_ts)
                say(f"`{response}`", thread_ts=thread_ts)

            elif reset_match_obj := RESET_PATTERN.match(text.lower()):
                parameter = reset_match_obj.group(1)
                if parameter == "config":
                    response = gpt_Bot.reset_config(thread_ts)
                    say(f"`{response}`", thread_ts=thread_ts)
                else:
                    say(f"Unknown reset parameter: {parameter}", thread_ts=thread_ts)

            elif text.startswith("!"):
                say(
                    "`Invalid command. Type '!help' for a list of valid commands.`",
                    thread_ts=thread_ts,
                )

            else:
                return text


def rebuild_thread_history(say, channel_id, thread_id, bot_user_id):
    response = app.client.conversations_replies(channel=channel_id, ts=thread_id)
    messages = response.get("messages", [])
    gpt_Bot.conversations[thread_id] = {
        "messages": [SLACK_SYSTEM_PROMPT], # Assume default system prompt for now.
        "processing": False,
        "history_reloaded": True,
    }
    
    # Bot commands and responses to ignore
    bot_commands = ["!history", "!help", "!usage", "!config", "!reset"]
    response_patterns = [
        "Cumulative Token stats since last reset:",
        "Current Configuration:",
        "Configuration Defaults Reset!",
        "Updated config setting",
        "Unknown setting:",
        "Invalid command.",
        "[HISTORY]"
        ]

    for msg in messages[:-1]:
        text = msg.get("text", "").strip()

        # Skip bot command messages
        if any(text.lower().startswith(command) for command in bot_commands):
            # print(f"Skipped bot command: {text}")
            continue
        
        # Skip bot response messages
        if any(response_pattern in text for response_pattern in response_patterns):
            # print(f"Skipped bot response: {text}")
            continue        
        
        role = "assistant" if msg.get("user") == bot_user_id else "user"
        content = []

        content.append({"type": "text", "text": remove_userid(msg.get("text"))})

        # Rebuild image history in b64 encoded format
        files = msg.get("files", [])
        for file in files:
            if file.get("mimetype") in ALLOWED_MIMETYPES:
                image_url = file.get("url_private")
                if image_url:
                    encoded_image = utils.download_and_encode_file(
                        say, image_url, SLACK_BOT_TOKEN
                    )
                    if encoded_image:
                        if role == "assistant":
                            role = "user" # OpenAI API restriction doesn't allow image urls for the assistant role. Force them to user.
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}",
                                    "detail": gpt_Bot.current_config_options["detail"],
                                },
                            }
                        )

        gpt_Bot.conversations[thread_id]["messages"].append(
            {"role": role, "content": content}
        )
    # print(utils.format_message_for_debug(gpt_Bot.conversations[thread_id]))

def process_and_respond(event, say):
    channel_id = event["channel"]
    is_thread = "thread_ts" in event
    thread_ts = event["thread_ts"] if is_thread else event["ts"]

    # Get the message from the Slack event
    message_text = event.get("text") or event.get("message", {}).get("text", "")

    # Handle new or existing threads since last restart
    if thread_ts not in gpt_Bot.conversations:
        if is_thread:
            rebuild_thread_history(say, channel_id, thread_ts, bot_user_id)

        else:
            gpt_Bot.conversations[thread_ts] = {
                "messages": [gpt_Bot.SYSTEM_PROMPT],
                "processing": False,
                "history_reloaded": False,
            }
        # print(f"Initialized threads: {list(gpt_Bot.conversations.keys())}\n")  # Debug
        # print(f"Initialized conversation: {gpt_Bot.conversations}\n")  # Debug
        
            
    # Remove the userID from the message using regex pattern matching
    # Clean up the message text and then pass it to the parse_text function
    message_text = parse_text(
        remove_userid(message_text), say, thread_ts, is_thread
    )


    if message_text or ("files" in event and event["files"]):
        # If bot is still processing a previous request, inform user it's busy and track busy messages
        if gpt_Bot.is_processing(thread_ts):
            response = app.client.chat_postMessage(
                channel=channel_id,
                text=f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:",
                thread_ts=thread_ts,
            )
            chat_del_ts.append(response["message"]["ts"])
            return

        initial_response = say(f"Thinking... {LOADING_EMOJI}", thread_ts=thread_ts)
        chat_del_ts.append(initial_response["message"]["ts"])

        #  Check if user is requesting Dalle3 image gen via LLM response.
        trigger_check = utils.check_for_image_generation(
            message_text, gpt_Bot, thread_ts)

        # If intent was likely an dalle3 image gen request...
        if trigger_check:
            if "files" in event and event["files"]:
                say(
                    ":warning:Ignoring included file with Dalle-3 request. Image gen based on provided images is not yet supported with Dalle-3.:warning:",
                    thread_ts=thread_ts,
                )
            
            # create dalle3 prompt from history
            
            dalle3_prompt = utils.create_dalle3_prompt(message_text, gpt_Bot, thread_ts)
            
            
            # Manually construct event msg since the Slack Slash command repsonses are different
            message_event = {
                "user_id": event["user"],
                "text": dalle3_prompt.content,
                "channel_id": channel_id,
                "command": "dalle-3 via conversational chat",
            }
            process_image_and_respond(say, message_event, thread_ts)

        # If there are files in the message (GPT Vision request or other file types)
        elif "files" in event and event["files"]:

            files_data = event.get("files", [])
            vision_files = []
            # Future non-vision files. Requires preprocessing/extracting text.
            other_files = []

            # Iterate through files, check file type. If supported image type, b64 encode it, else not supported type.
            for file in files_data:
                file_url = file.get("url_private")
                file_mimetype = file.get("mimetype")

                if file_url and file_mimetype in ALLOWED_MIMETYPES:
                    encoded_file = utils.download_and_encode_file(
                        say, file_url, SLACK_BOT_TOKEN
                    )
                    if encoded_file:
                        vision_files.append(encoded_file)
                else:
                    encoded_file = utils.download_and_encode_file(
                        say, file_url, SLACK_BOT_TOKEN
                    )
                    if encoded_file:
                        other_files.append(encoded_file)

            if vision_files:
                response, is_error = gpt_Bot.vision_context_mgr(
                    message_text, vision_files, thread_ts
                )
                if is_error:
                    utils.handle_error(say, response, thread_ts=thread_ts)

                else:
                    converted_text = mrkdown_converter.convert(response)
                    response = re.sub(r'\s+,', ',', converted_text) # Remove extra spaces before commas
                    say(response, thread_ts=thread_ts)

            elif other_files:
                say(
                    ":no_entry: `Sorry, GPT4 Vision only supports jpeg, png, webp, and non-animated gif file types at this time.` :no_entry:",
                    thread_ts=thread_ts,
                )

            # Cleanup busy/loading chat msgs
            delete_chat_messages(channel_id, chat_del_ts, say)

        # If just a normal text message, process with default chat context manager
        else:
            response, is_error = gpt_Bot.chat_context_mgr(message_text, thread_ts)
            if is_error:
                utils.handle_error(say, response)

            else:
                converted_text = mrkdown_converter.convert(response)
                response = re.sub(r'\s+,', ',', converted_text) # Remove extra spaces before commas
                say(text=response, thread_ts=thread_ts)

            # Cleanup busy/loading chat msgs
            delete_chat_messages(channel_id, chat_del_ts, say)

# Dalle-3 image gen via /dalle-3 command or LLM verification
def process_image_and_respond(say, command, thread_ts=None):
    user_id = command["user_id"]
    text = command["text"]
    cmd = command["command"]
    channel = command["channel_id"]

    if gpt_Bot.is_processing(thread_ts):
        response = app.client.chat_postMessage(
            channel=channel,
            text=f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:",
            thread_ts=thread_ts,
        )
        chat_del_ts.append(response["message"]["ts"])

    else:
        if not text:
            app.client.chat_postEphemeral(
                channel=channel,
                user=user_id,
                text=":no_entry: You must provide a prompt when using `/dalle-3` :no_entry:",
                thread_ts=thread_ts,
            )
            return

        if cmd == DALLE3_CMD:
            response = app.client.chat_postMessage(
                channel=channel,
                text=f"<@{user_id}> used `{cmd}`.\n*Original Prompt:*\n_{text}_",
                thread_ts=thread_ts,
            )
            if not thread_ts:
                thread_ts = response["ts"]

        # Handle new threads
        if thread_ts not in gpt_Bot.conversations:
            gpt_Bot.conversations[thread_ts] = {
                "messages": [SLACK_SYSTEM_PROMPT],
                "processing": False,
                "history_reloaded": False,
            }

        # Image gen takes a while. Give the user some indication things are processing.
        delete_chat_messages(channel, chat_del_ts, say)

        temp_response = app.client.chat_postMessage(
            channel=channel,
            text=f"Generating image, please wait... {LOADING_EMOJI}",
            thread_ts=thread_ts,
        )
        chat_del_ts.append(temp_response["ts"])

        # Dalle-3 always responds with a more detailed revised prompt.
        image, revised_prompt, is_error = gpt_Bot.image_context_mgr(text, thread_ts)

        # revised_prompt holds any error values in this case
        if is_error:
            utils.handle_error(say, revised_prompt, thread_ts=thread_ts)

        # Build the response message and upload the generated image to Slack
        else:
            if gpt_Bot.current_config_options["d3_revised_prompt"]:
                file_description = f"*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_"
            else:
                file_description = None
            try:
                response = app.client.files_upload_v2(
                    channel=channel,
                    initial_comment=file_description,
                    file=image,
                    filename="Dalle3_image.png",
                    thread_ts=thread_ts,
                )

            except Exception:
                utils.handle_error(say, revised_prompt, thread_ts=thread_ts)
            
            # print(utils.format_message_for_debug(gpt_Bot.conversations[thread_ts]))
        delete_chat_messages(channel, chat_del_ts, say)


# Process timestamps of any temporary status or progress messages the bot sends to Slack. Called to clean them up once a response completes.
def delete_chat_messages(channel, timestamps, say, thread_ts=None):
    try:
        for ts in timestamps:
            app.client.chat_delete(channel=channel, ts=ts)

    except Exception as e:
        say(
            f":no_entry: `Sorry, I ran into an error cleaning up my own messages.` :no_entry:\n```{e}```",
            thread_ts=thread_ts,
        )
    finally:
        chat_del_ts.clear()

def remove_userid(message_text):
    message_text = re.sub(USER_ID_PATTERN, "", message_text).strip()
    return message_text
    

# Slack event handlers
@app.command(DALLE3_CMD)
def handle_dalle3(ack, say, command):
    ack()
    process_image_and_respond(say, command)


@app.event("app_mention")
def handle_mention(event, say):
    process_and_respond(event, say)


@app.event("message")
def handle_message_events(event, say):
    # Ignore 'message_changed' and other subtypes for now.
    # Deleting the "Thinking..." message after a response returns triggers an additional Slack event
    # which causes dupe responses by the bot in DMs w/ Threads.
    if "subtype" in event and event["subtype"] == "message_changed":
        return

    elif event["channel_type"] == "im":
        process_and_respond(event, say)


if __name__ == "__main__":
    gpt_Bot = bot.ChatBot(SLACK_SYSTEM_PROMPT, STREAMING_CLIENT, show_dalle3_revised_prompt)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    handler.start()


# pr = cProfile.Profile()
# pr.enable()
# myFunction()
# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(10)
# print(s.getvalue())