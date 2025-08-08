import re
import threading
from os import environ
from prompts import SLACK_SYSTEM_PROMPT
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from markdown_to_mrkdwn import SlackMarkdownConverter
from queue_manager import QueueManager
from logger import log_session_marker, setup_logger, get_log_level, get_logger

import bot_functions as bot
import common_utils as utils

## For performance profiling.
# import cProfile
# import pstats
# import io

# Unset any existing log level environment variables to ensure .env values are used
if "SLACK_LOG_LEVEL" in environ:
    del environ["SLACK_LOG_LEVEL"]

# Load environment variables and initialize converter
load_dotenv()
mrkdown_converter = SlackMarkdownConverter()

# Configuration variables
LOADING_EMOJI = ":loading:"
SLACK_BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = environ.get("SLACK_APP_TOKEN")
DALLE3_CMD = environ.get("DALLE3_CMD", "/dalle-3")

# Configure logging level from environment variable with fallback to INFO
LOG_LEVEL_NAME = environ.get("SLACK_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = get_log_level(LOG_LEVEL_NAME)
# Initialize logger with the configured log level
logger = get_logger('slack_bot', LOG_LEVEL)

show_dalle3_revised_prompt = False

# Patterns to match commands
CONFIG_PATTERN = re.compile(r"!config\s+(\S+)\s+(.+)")
RESET_PATTERN = re.compile(r"^!reset\s+(\S+)$")
# Pattern to match the slackbot's userID in channel messages
USER_ID_PATTERN = re.compile(r"<@[\w]+>")
STREAMING_CLIENT = False  # not implemented for Slack...yet.
# GPT4 vision supported image types
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Dictionary of message timestamps to cleanup after a response returns (per thread)
chat_del_ts = {}
chat_del_ts_lock = threading.Lock()  

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

# Call the auth.test method to capture bot info
response = app.client.auth_test()
# Extract the Slackbot's user ID
bot_user_id = response.get("user_id")

# Initialize the queue manager
queue_manager = QueueManager()


def parse_text(text, say, thread_ts, is_thread=False):
    """
    Parse the message text to check if a bot command was sent and respond accordingly.
    
    Args:
        text (str): The message text to parse.
        say (callable): A function to send messages to Slack.
        thread_ts (str): The timestamp of the thread.
        is_thread (bool, optional): Whether the message is in a thread. Defaults to False.
        
    Returns:
        str or None: The message text if it's not a command, None otherwise.
    """
    if not is_thread:
        thread_ts = None

    match text.lower():
        case "!history":
            logger.info(f"History command received in thread {thread_ts}")
            say(f"```{gpt_Bot.history_command(thread_ts)}```", thread_ts=thread_ts)

        case "!help":
            logger.info(f"Help command received in thread {thread_ts}")
            say(f"```{gpt_Bot.help_command()}```", thread_ts=thread_ts)

        case "!usage":
            logger.info(f"Usage command received in thread {thread_ts}")
            say(f"```{gpt_Bot.usage_command()}```", thread_ts=thread_ts)

        case "!config":
            logger.info(f"Config command received in thread {thread_ts}")
            say(
                f"```Current Configuration:\n{gpt_Bot.view_config(thread_ts)}```",
                thread_ts=thread_ts,
            )

        case _:
            if config_match_obj := CONFIG_PATTERN.match(text.lower()):
                setting, value = config_match_obj.groups()
                logger.info(f"Config change: {setting}={value} in thread {thread_ts}")
                response = gpt_Bot.set_config(setting, value, thread_ts)
                say(f"`{response}`", thread_ts=thread_ts)

            elif reset_match_obj := RESET_PATTERN.match(text.lower()):
                parameter = reset_match_obj.group(1)
                if parameter == "config":
                    logger.info(f"Config reset in thread {thread_ts}")
                    response = gpt_Bot.reset_config(thread_ts)
                    say(f"`{response}`", thread_ts=thread_ts)
                else:
                    logger.warning(f"Unknown reset parameter: {parameter} in thread {thread_ts}")
                    say(f"Unknown reset parameter: {parameter}", thread_ts=thread_ts)

            elif text.startswith("!"):
                logger.warning(f"Invalid command received: {text} in thread {thread_ts}")
                say(
                    "`Invalid command. Type '!help' for a list of valid commands.`",
                    thread_ts=thread_ts,
                )

            else:
                return text


def rebuild_thread_history(say, channel_id, thread_id, bot_user_id):
    """
    Rebuild the conversation history for a thread from Slack's API.
    
    This function fetches the conversation history from Slack's API and reconstructs
    the conversation history in the ChatBot's format, including handling images.
    
    Args:
        say (callable): A function to send messages to Slack.
        channel_id (str): The ID of the channel.
        thread_id (str): The ID of the thread.
        bot_user_id (str): The ID of the bot user.
    """
    logger.info(f"Rebuilding conversation history for thread {thread_id}")
    
    # Fetch conversation replies from Slack API
    try:
        response = app.client.conversations_replies(channel=channel_id, ts=thread_id)
        messages = response.get("messages", [])
        
        # Initialize conversation with default system prompt
        gpt_Bot.conversations[thread_id] = {
            "messages": [SLACK_SYSTEM_PROMPT], # Assume default system prompt for now.
            "history_reloaded": True,
        }
        
        # Bot commands and responses to ignore when rebuilding history
        bot_commands = ["!history", "!help", "!usage", "!config", "!reset"]
        response_patterns = [
            "Cumulative Token stats since last reset:",
            "Current Configuration:",
            "Configuration Defaults Reset!",
            "Updated config setting",
            "Unknown setting:",
            "Invalid command.",
            "[HISTORY]",
            "Thinking...",
            "Generating image",
            "I'm busy processing"
            ]

        # Process each message in the thread
        for msg in messages[:-1]:  # Skip the most recent message (current one)
            text = msg.get("text", "").strip()
            
            # Skip empty messages
            if not text:
                continue

            # Skip bot command messages
            if any(text.lower().startswith(command) for command in bot_commands):
                continue
            
            # Skip bot response messages
            if any(response_pattern in text for response_pattern in response_patterns):
                continue        
            
            # Determine message role (assistant or user)
            role = "assistant" if msg.get("user") == bot_user_id else "user"
            content = []

            # Add text content
            content.append({"type": "text", "text": remove_userid(msg.get("text"))})

            # Rebuild image history in b64 encoded format
            files = msg.get("files", [])
            for file in files:
                if file.get("mimetype") in ALLOWED_MIMETYPES:
                    image_url = file.get("url_private")
                    if image_url:
                        try:
                            encoded_image = utils.download_and_encode_file(
                                say, image_url, SLACK_BOT_TOKEN
                            )
                            if encoded_image:
                                if role == "assistant":
                                    # OpenAI API restriction doesn't allow image urls for the assistant role
                                    role = "user"  # Force them to user
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{encoded_image}",
                                            "detail": gpt_Bot.current_config_options["detail"],
                                        },
                                    }
                                )
                        except Exception as e:
                            logger.error(f"Error encoding image: {e}", exc_info=True)

            # Add message to conversation history
            gpt_Bot.conversations[thread_id]["messages"].append(
                {"role": role, "content": content}
            )
        
        logger.info(f"Rebuilt conversation history with {len(gpt_Bot.conversations[thread_id]['messages']) - 1} messages")
    except Exception as e:
        logger.error(f"Error rebuilding thread history: {e}", exc_info=True)
        raise


def process_and_respond(event, say):
    """
    Process a message event and respond accordingly.
    
    This function handles new or existing threads, processes messages with or without files,
    and manages the bot's response.
    
    Args:
        event (dict): The Slack event to process.
        say (callable): A function to send messages to Slack.
    """
    channel_id = event["channel"]
    is_thread = "thread_ts" in event
    thread_ts = event["thread_ts"] if is_thread else event["ts"]

    logger.info(f"Processing message in thread {thread_ts}")

    # Check if this thread is already processing a message
    if queue_manager.is_processing_sync(thread_ts):
        logger.info(f"Thread {thread_ts} is already processing, sending busy message")
        response = app.client.chat_postMessage(
            channel=channel_id,
            text=f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:",
            thread_ts=thread_ts,
        )
        with chat_del_ts_lock:
            if thread_ts not in chat_del_ts:
                chat_del_ts[thread_ts] = []
            chat_del_ts[thread_ts].append(response["message"]["ts"])
        return

    # Try to start processing this thread
    if not queue_manager.start_processing_sync(thread_ts):
        logger.info(f"Failed to start processing thread {thread_ts}")
        return  # Another concurrent call got here first

    try:
        # Get the message from the Slack event
        message_text = event.get("text") or event.get("message", {}).get("text", "")
        logger.debug(f"Message text: {message_text}")

        # Handle new or existing threads since last restart
        if thread_ts not in gpt_Bot.conversations:
            logger.info(f"Initializing new conversation for thread {thread_ts}")
            if is_thread:
                # Rebuild history for existing thread
                rebuild_thread_history(say, channel_id, thread_ts, bot_user_id)
            else:
                # Initialize new conversation
                gpt_Bot.conversations[thread_ts] = {
                    "messages": [gpt_Bot.SYSTEM_PROMPT],
                    "history_reloaded": False,
                }
            
        # Remove the userID from the message and parse it
        message_text = parse_text(
            remove_userid(message_text), say, thread_ts, is_thread
        )

        # If parse_text returned None, it means a command was processed
        # Don't continue with normal message processing
        if message_text is None:
            logger.debug(f"Command was processed for thread {thread_ts}, skipping normal message processing")
            return

        # Process the message if there's text or files
        if message_text or ("files" in event and event["files"]):
            # Send initial "thinking" message
            initial_response = say(f"Thinking... {LOADING_EMOJI}", thread_ts=thread_ts)
            with chat_del_ts_lock:
                if thread_ts not in chat_del_ts:
                    chat_del_ts[thread_ts] = []
                chat_del_ts[thread_ts].append(initial_response["message"]["ts"])

            # Check if user is requesting DALL-E 3 image generation
            logger.info(f"Checking if message is requesting image generation: {message_text}")
            trigger_check = utils.check_for_image_generation(
                message_text, gpt_Bot, thread_ts)
            logger.info(f"Image generation check result: {trigger_check}")

            # If intent was likely a DALL-E 3 image gen request
            if trigger_check:
                logger.info(f"Processing image generation request: {message_text}")
                if "files" in event and event["files"]:
                    logger.warning("Ignoring included file with Dalle-3 request")
                    say(
                        ":warning:Ignoring included file with Dalle-3 request. Image gen based on provided images is not yet supported with Dalle-3.:warning:",
                        thread_ts=thread_ts,
                    )
                
                # Create DALL-E 3 prompt from history
                logger.info("Creating DALL-E 3 prompt from history")
                dalle3_prompt = utils.create_dalle3_prompt(message_text, gpt_Bot, thread_ts)
                logger.debug(f"DALL-E 3 prompt: {dalle3_prompt.content}")
                
                # Manually construct event msg since the Slack Slash command responses are different
                message_event = {
                    "user_id": event["user"],
                    "text": dalle3_prompt.content,
                    "channel_id": channel_id,
                    "command": "dalle-3 via conversational chat",
                }
                
                # Release thread lock before image processing
                logger.info("Finishing processing before calling process_image_and_respond")
                queue_manager.finish_processing_sync(thread_ts)
                
                logger.info("Calling process_image_and_respond")
                process_image_and_respond(say, message_event, thread_ts)
                logger.info("Returned from process_image_and_respond")
                
                return  # Prevent duplicate finish_processing_sync calls

            # If there are files in the message (GPT Vision request or other file types)
            elif "files" in event and event["files"]:
                logger.info("Processing message with files")
                files_data = event.get("files", [])
                vision_files = []
                # Future non-vision files. Requires preprocessing/extracting text.
                other_files = []

                # Process each file
                for file in files_data:
                    file_url = file.get("url_private")
                    file_mimetype = file.get("mimetype")
                    logger.info(f"Processing file: {file.get('name')} ({file_mimetype})")

                    if file_url and file_mimetype in ALLOWED_MIMETYPES:
                        try:
                            encoded_file = utils.download_and_encode_file(
                                say, file_url, SLACK_BOT_TOKEN
                            )
                            if encoded_file:
                                vision_files.append(encoded_file)
                                logger.info(f"Added file to vision files: {file.get('name')}")
                        except Exception as e:
                            logger.error(f"Error encoding vision file: {e}", exc_info=True)
                    else:
                        try:
                            encoded_file = utils.download_and_encode_file(
                                say, file_url, SLACK_BOT_TOKEN
                            )
                            if encoded_file:
                                other_files.append(encoded_file)
                                logger.info(f"Added file to other files: {file.get('name')}")
                        except Exception as e:
                            logger.error(f"Error encoding other file: {e}", exc_info=True)

                # Handle vision files
                if vision_files:
                    logger.info(f"Processing {len(vision_files)} vision files")
                    try:
                        response, is_error = gpt_Bot.vision_context_mgr(
                            message_text, vision_files, thread_ts
                        )
                        if is_error:
                            logger.error(f"Error in vision context manager: {response}")
                            utils.handle_error(say, response, thread_ts=thread_ts)
                        else:
                            logger.info("Vision processing successful")
                            converted_text = mrkdown_converter.convert(response)
                            response = re.sub(r'\s+,', ',', converted_text)  # Remove extra spaces before commas
                            say(response, thread_ts=thread_ts)
                    except Exception as e:
                        logger.error(f"Error processing vision files: {e}", exc_info=True)
                        utils.handle_error(say, str(e), thread_ts=thread_ts)

                # Handle unsupported file types
                elif other_files:
                    logger.warning("Unsupported file types received")
                    say(
                        ":no_entry: `Sorry, GPT4 Vision only supports jpeg, png, webp, and non-animated gif file types at this time.` :no_entry:",
                        thread_ts=thread_ts,
                    )

                # Cleanup busy/loading chat msgs
                delete_chat_messages_sync(channel_id, thread_ts, say)

            # If just a normal text message, process with default chat context manager
            else:
                logger.info("Processing normal text message")
                try:
                    response, is_error = gpt_Bot.chat_context_mgr(message_text, thread_ts)
                    if is_error:
                        logger.error(f"Error in chat context manager: {response}")
                        utils.handle_error(say, response)
                    else:
                        logger.info("Chat processing successful")
                        converted_text = mrkdown_converter.convert(response)
                        response = re.sub(r'\s+,', ',', converted_text)  # Remove extra spaces before commas
                        say(text=response, thread_ts=thread_ts)
                except Exception as e:
                    logger.error(f"Error processing text message: {e}", exc_info=True)
                    utils.handle_error(say, str(e), thread_ts=thread_ts)

                # Cleanup busy/loading chat msgs
                delete_chat_messages_sync(channel_id, thread_ts, say)
    except Exception as e:
        logger.error(f"Unexpected error in process_and_respond: {e}", exc_info=True)
        try:
            say(
                f":no_entry: `Sorry, I ran into an unexpected error.` :no_entry:\n```{str(e)}```",
                thread_ts=thread_ts,
            )
        except:
            pass
    finally:
        # Always cleanup, even if there was an error
        logger.info(f"Finishing processing for thread {thread_ts}")
        queue_manager.finish_processing_sync(thread_ts)


def process_image_and_respond(say, command, thread_ts=None):
    """
    Process an image generation request and respond with the generated image.
    
    This function handles DALL-E 3 image generation requests from the /dalle-3 command
    or from the LLM verification process.
    
    Args:
        say (callable): A function to send messages to Slack.
        command (dict): The command data containing the prompt and user info.
        thread_ts (str, optional): The timestamp of the thread. Defaults to None.
    """
    user_id = command["user_id"]
    text = command["text"]
    cmd = command["command"]
    channel = command["channel_id"]

    logger.info(f"Processing image request: thread_ts={thread_ts}, text={text}")

    # Check if this thread is already processing
    if queue_manager.is_processing_sync(thread_ts):
        logger.info(f"Thread {thread_ts} is already processing, sending busy message")
        response = app.client.chat_postMessage(
            channel=channel,
            text=f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:",
            thread_ts=thread_ts,
        )
        with chat_del_ts_lock:
            if thread_ts not in chat_del_ts:
                chat_del_ts[thread_ts] = []
            chat_del_ts[thread_ts].append(response["message"]["ts"])
        return

    # Try to start processing this thread
    if not queue_manager.start_processing_sync(thread_ts):
        logger.info(f"Failed to start processing thread {thread_ts}")
        return  # Another concurrent call got here first

    try:
        # Validate prompt
        if not text:
            logger.warning("Empty prompt received")
            app.client.chat_postEphemeral(
                channel=channel,
                user=user_id,
                text=":no_entry: You must provide a prompt when using `/dalle-3` :no_entry:",
                thread_ts=thread_ts,
            )
            return

        # Handle slash command
        if cmd == DALLE3_CMD:
            logger.info(f"Processing slash command {cmd}")
            response = app.client.chat_postMessage(
                channel=channel,
                text=f"<@{user_id}> used `{cmd}`.\n*Original Prompt:*\n_{text}_",
                thread_ts=thread_ts,
            )
            if not thread_ts:
                thread_ts = response["ts"]
                logger.info(f"Created new thread with ts={thread_ts}")

        # Initialize new thread if needed
        if thread_ts not in gpt_Bot.conversations:
            logger.info(f"Initializing new conversation for thread {thread_ts}")
            gpt_Bot.conversations[thread_ts] = {
                "messages": [SLACK_SYSTEM_PROMPT],
                "history_reloaded": False,
            }

        # Cleanup any previous status messages
        delete_chat_messages_sync(channel, thread_ts, say)

        # Send "generating" message
        logger.info("Sending 'generating image' message")
        temp_response = app.client.chat_postMessage(
            channel=channel,
            text=f"Generating image, please wait... {LOADING_EMOJI}",
            thread_ts=thread_ts,
        )
        with chat_del_ts_lock:
            if thread_ts not in chat_del_ts:
                chat_del_ts[thread_ts] = []
            chat_del_ts[thread_ts].append(temp_response["ts"])

        # Generate image with DALL-E 3
        logger.info(f"Calling image_context_mgr with text={text}")
        try:
            image, revised_prompt, is_error = gpt_Bot.image_context_mgr(text, thread_ts)
            logger.info(f"image_context_mgr returned: is_error={is_error}")

            # Handle error case
            if is_error:
                logger.error(f"Error generating image: {revised_prompt}")
                utils.handle_error(say, revised_prompt, thread_ts=thread_ts)
            # Handle successful image generation
            else:
                logger.info("Image generated successfully")
                if gpt_Bot.current_config_options["d3_revised_prompt"]:
                    file_description = f"*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_"
                    logger.info(f"Revised prompt: {revised_prompt}")
                else:
                    file_description = None
                    
                try:
                    # Upload the generated image to Slack
                    logger.info("Uploading image to Slack")
                    response = app.client.files_upload_v2(
                        channel=channel,
                        initial_comment=file_description,
                        file=image,
                        filename="Dalle3_image.png",
                        thread_ts=thread_ts,
                    )
                    logger.info("Image uploaded successfully")
                except Exception as e:
                    logger.error(f"Error uploading image: {e}", exc_info=True)
                    utils.handle_error(say, str(e), thread_ts=thread_ts)
        except Exception as e:
            logger.error(f"Error in image generation process: {e}", exc_info=True)
            utils.handle_error(say, str(e), thread_ts=thread_ts)
            
        # Cleanup status messages
        delete_chat_messages_sync(channel, thread_ts, say)
    except Exception as e:
        logger.error(f"Unexpected error in process_image_and_respond: {e}", exc_info=True)
        try:
            say(
                f":no_entry: `Sorry, I ran into an unexpected error generating the image.` :no_entry:\n```{str(e)}```",
                thread_ts=thread_ts,
            )
        except:
            pass
    finally:
        # Always cleanup, even if there was an error
        logger.info(f"Finishing processing for thread {thread_ts}")
        queue_manager.finish_processing_sync(thread_ts)


def delete_chat_messages_sync(channel, thread_ts, say):
    """
    Delete temporary status or progress messages the bot sends to Slack.
    
    Args:
        channel (str): The channel ID.
        thread_ts (str): The timestamp of the thread.
        say (callable): A function to send messages to Slack.
    """
    # Get timestamps for this thread and remove from dictionary
    with chat_del_ts_lock:
        timestamps = chat_del_ts.pop(thread_ts, [])
    
    if not timestamps:
        return
        
    logger.debug(f"Deleting {len(timestamps)} messages for thread {thread_ts}")
    try:
        for ts in timestamps:
            try:
                app.client.chat_delete(channel=channel, ts=ts)
            except Exception as e:
                # Log errors for individual message deletions at debug level
                # These are often expected (e.g., message already deleted)
                logger.debug(f"Failed to delete message {ts}: {e}")
    except Exception as e:
        logger.error(f"Error deleting messages: {e}", exc_info=True)
        say(
            f":no_entry: `Sorry, I ran into an error cleaning up my own messages.` :no_entry:\n```{e}```",
            thread_ts=thread_ts,
        )


def remove_userid(message_text):
    """
    Remove user IDs from a message text.
    
    Args:
        message_text (str): The message text to process.
        
    Returns:
        str: The message text with user IDs removed.
    """
    message_text = re.sub(USER_ID_PATTERN, "", message_text).strip()
    return message_text
    

# Slack event handlers
@app.command(DALLE3_CMD)
def handle_dalle3(ack, say, command):
    """
    Handle the /dalle-3 command.
    
    Args:
        ack (callable): A function to acknowledge the command.
        say (callable): A function to send messages to Slack.
        command (dict): The command data.
    """
    logger.info(f"Received /dalle-3 command from user {command['user_id']}")
    ack()
    # Process the command synchronously
    process_image_and_respond(say, command)


@app.event("app_mention")
def handle_mention(event, say):
    """
    Handle app mention events.
    
    Args:
        event (dict): The event data.
        say (callable): A function to send messages to Slack.
    """
    logger.info(f"Received app mention from user {event.get('user')}")
    # Process the event synchronously
    process_and_respond(event, say)


@app.event("message")
def handle_message_events(event, say):
    """
    Handle message events.
    
    Args:
        event (dict): The event data.
        say (callable): A function to send messages to Slack.
    """
    # Ignore specific subtypes that cause issues
    # 'message_changed' is triggered when we delete the "Thinking..." message
    # which can cause duplicate responses in DMs with Threads
    if "subtype" in event and event["subtype"] == "message_changed":
        logger.debug(f"Ignoring message_changed event to prevent duplicate responses")
        return

    # Only process direct messages to the bot
    elif event["channel_type"] == "im":
        # Check if this is a bot message
        if event.get("bot_id") or event.get("user") == bot_user_id:
            logger.debug(f"Ignoring message from bot: {event.get('text', '')}")
            return
            
        logger.info(f"Received direct message from user {event.get('user')}")
        # Process the event synchronously
        process_and_respond(event, say)


if __name__ == "__main__":
    # Log session start marker
    log_session_marker(logger, "START")
    
    # Log the configured log level after the session marker
    logger.info(f"Slack logger initialized with log level: {LOG_LEVEL_NAME}")
    
    logger.info("Starting Slackbot")
    gpt_Bot = bot.ChatBot(SLACK_SYSTEM_PROMPT, STREAMING_CLIENT, show_dalle3_revised_prompt)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    try:
        logger.info("Starting SocketModeHandler")
        handler.start()
    except Exception as e:
        logger.critical(f"Failed to start Slack bot: {e}", exc_info=True)
        raise
    finally:
        # Log session end marker
        log_session_marker(logger, "END")


## Performance profiling code
# pr = cProfile.Profile()
# pr.enable()
# myFunction()
# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(10)
# print(s.getvalue())
