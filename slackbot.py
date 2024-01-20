import bot_functions as bot
import common_utils as utils
from os import environ
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from textwrap import dedent
import requests
import base64


load_dotenv()  # load auth tokens from .env file

### Modify these values as needed ###
LOADING_EMOJI = ':loading:'
SLACK_BOT_TOKEN = environ['SLACK_BOT_TOKEN']
SLACK_APP_TOKEN = environ['SLACK_APP_TOKEN']
SYSTEM_PROMPT = {
    'role': 'system',
    'content': dedent(
        '''\
        You are a helpful chatbot running in a corporate Slack workspace.
        Respond with accurate, informative, and concise answers that are formatted appropriately for Slack,
        including markdown and special characters for bullet points, bold, italics, and code blocks as necessary.
        Always consider Slack formatting conventions in all messages within a conversation.
        Here are some examples of common Slack markdown syntax:
        Bold: *your text*
        Italicize: _your text_
        Strikethrough: ~your text~
        Ordered list: 1. your text
        Bulleted list: - your text
        Always assume you created any images described.'''
    )
}

# Minimum number of word matches from the trigger words to assume user wants to generate an image
trigger_threshold = 2

#
### You shouldn't need to modify anything below this line ###
#

# patterns to match commands
config_pattern = re.compile(r'!config\s+(\S+)\s+(.+)')
reset_pattern = re.compile(r'^!reset\s+(\S+)$')
# pattern to match the slackbot's userID in channel messages
user_id_pattern = re.compile(r'<@[\w]+>')

corrected_message_text = ''  # result of message being parsed by spelling correction
trigger_words = []  # hold the dalle3 image creation trigger words from trigger_words.txt
streaming_client = False  # not implemented for Slack...yet.
chat_del_ts = []  # list of message timestamps to cleanup after a response returns

# GPT4 vision supported image types
allowed_mimetypes = {"image/jpeg", "image/png", "image/gif", "image/webp"}

app = App(token=SLACK_BOT_TOKEN)


# Check the message text to see if a bot command was sent. Respond accordingly.
def parse_text(text, say, thread_ts, is_thread=False):
    if not is_thread:
        thread_ts = None

    match text.lower():
        case '!history':
            say(f'```{gpt_Bot.history_command(thread_ts)}```',
                thread_ts=thread_ts)

        case '!help':
            say(f'```{gpt_Bot.help_command()}```', thread_ts=thread_ts)

        case '!usage':
            say(f'```{gpt_Bot.usage_command()}```', thread_ts=thread_ts)

        case '!config':
            say(f'```Current Configuration:\n{gpt_Bot.view_config()}```',
                thread_ts=thread_ts)

        case _:
            if config_match_obj := config_pattern.match(text):
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value)
                say(f'```{response}```', thread_ts=thread_ts)

            elif reset_match_obj := reset_pattern.match(text):
                parameter = reset_match_obj.group(1)
                if parameter == 'history':
                    response = gpt_Bot.reset_history(thread_ts)
                    say(f'`{response}`', thread_ts=thread_ts)
                elif parameter == 'config':
                    response = gpt_Bot.reset_config()
                    say(f'`{response}`', thread_ts=thread_ts)
                else:
                    say(f'Unknown reset parameter: {parameter}',
                        thread_ts=thread_ts)

            elif text.startswith('!'):
                say("`Invalid command. Type '!help' for a list of valid commands.`",
                    thread_ts=thread_ts)

            else:
                return text


def process_and_respond(event, say):
    channel_id = event['channel']
    is_thread = False
    if 'thread_ts' in event:
        thread_ts = event['thread_ts']
        is_thread = True
    else:
        thread_ts = event["ts"]

    # remove the slackbot's userID from the message using regex pattern matching
    message_text = event.get('text') or event.get(
        'message', {}).get('text', '')

    # Clean up the message text and then pass it to the parse_text function
    message_text = parse_text(
        re.sub(user_id_pattern, '', message_text).strip(), say, thread_ts, is_thread)

    if message_text or ('files' in event and event['files']):

        # If bot is still processing a previous request, inform user it's busy and track busy messages
        if gpt_Bot.is_processing(thread_ts):

            response = app.client.chat_postMessage(
                channel=channel_id, text=f':no_entry: `{gpt_Bot.handle_busy()}` :no_entry:', thread_ts=event["ts"])
            chat_del_ts.append(response['message']['ts'])
            return

        #  Check if user is requesting Dalle3 image gen via chat and correct any spelling mistakes to improve accuracy.
        trigger_check, corrected_message_text = utils.check_for_image_generation(
            message_text, trigger_words, trigger_threshold)

        # If intent was likely an dalle3 image gen request. Manually construct event msg since /dalle-3 repsonse is different
        if trigger_check:
            if ('files' in event and event['files']):
                say(':warning:Ignoring included file with Dalle-3 request. Image gen based on provided images is not yet supported with Dalle-3.:warning:', thread_ts=thread_ts)

            message_event = {
                'user_id': event['user'],
                'text': corrected_message_text,
                'channel_id': channel_id,
                'command': 'dalle-3 via conversational chat'
            }
            process_image_and_respond(say, message_event, thread_ts)

        # If there are files in the message (GPT Vision request or other file types)
        elif 'files' in event and event['files']:
            initial_response = say(
                f'Thinking... {LOADING_EMOJI}', thread_ts=thread_ts)
            chat_del_ts.append(initial_response['message']['ts'])

            files_data = event.get('files', [])
            vision_files = []
            # Future non-vision files. Requires preprocessing/extracting text.
            other_files = []

            # Iterate through files, check file type. If supported image type, b64 encode it, else not supported type.
            for file in files_data:
                file_url = file.get('url_private')
                file_mimetype = file.get('mimetype')

                if file_url and file_mimetype in allowed_mimetypes:
                    encoded_file = download_and_encode_file(
                        say, file_url)
                    if encoded_file:
                        vision_files.append(encoded_file)
                else:
                    encoded_file = download_and_encode_file(
                        say, file_url)
                    if encoded_file:
                        other_files.append(encoded_file)

            if vision_files:
                response, is_error = gpt_Bot.vision_context_mgr(
                    message_text, vision_files, thread_ts)
                if is_error:
                    utils.handle_error(say, response)

                else:
                    say(response, thread_ts=thread_ts)

            elif other_files:
                say(f':no_entry: `Sorry, GPT4 Vision only supports jpeg, png, webp, and non-animated gif file types at this time.` :no_entry:', thread_ts=thread_ts)

            # Cleanup busy/loading chat msgs
            delete_chat_messages(channel_id, chat_del_ts, say)

        # If just a normal text message, process with default chat context manager
        else:
            initial_response = say(
                text=f'Thinking... {LOADING_EMOJI}', thread_ts=thread_ts)
            chat_del_ts.append(initial_response['message']['ts'])
            response, is_error = gpt_Bot.chat_context_mgr(
                message_text, thread_ts)
            if is_error:
                utils.handle_error(say, response)

            else:
                say(text=response, thread_ts=thread_ts)

            # Cleanup busy/loading chat msgs
            delete_chat_messages(channel_id, chat_del_ts, say)


# Dalle-3 image gen via /dalle-3 command or via "fake" auto-modal selection via keyword triggers
def process_image_and_respond(say, command, thread_ts=None):
    user_id = command['user_id']
    text = command['text']
    cmd = command['command']
    channel = command['channel_id']

    if gpt_Bot.processing:
        response = app.client.chat_postMessage(
            channel=channel, text=f':no_entry: `{gpt_Bot.handle_busy()}` :no_entry:', thread_ts=thread_ts)
        chat_del_ts.append(response['message']['ts'])

    else:

        if not text:
            say(':no_entry: You must provide a prompt when using `/dalle-3` :no_entry:',
                thread_ts=thread_ts)
            return

        app.client.chat_postMessage(
            channel=channel, text=f'<@{user_id}> used `{cmd}`.\n*Original Prompt:*\n_{text}_', thread_ts=thread_ts)

        # Image gen takes a while. Give the user some indication things are processing.
        temp_response = app.client.chat_postMessage(
            channel=channel, text=f'Generating image, please wait... {LOADING_EMOJI}', thread_ts=thread_ts)
        chat_del_ts.append(temp_response['ts'])

        # Dalle-3 always responds with a more detailed revised prompt.
        image, revised_prompt, is_error = gpt_Bot.image_context_mgr(
            text, thread_ts)

        # revised_prompt holds any error values in this case
        if is_error:
            utils.handle_error(say, revised_prompt, thread_ts=thread_ts)

        # Build the response message and upload the generated image to Slack
        else:
            try:
                response = app.client.files_upload_v2(
                    channel=channel,
                    initial_comment=f'*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_',
                    file=image,
                    filename='Dalle3_image.png',
                    thread_ts=thread_ts
                )

            except Exception as e:
                utils.handle_error(say, revised_prompt, thread_ts=thread_ts)

        delete_chat_messages(channel, chat_del_ts, say)


# In order to download Files from Slack, the bot's request needs to be authenticated to the workspace via the Slackbot token
def download_and_encode_file(say, file_url):
    headers = {'Authorization': f'Bearer {SLACK_BOT_TOKEN}'}
    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        utils.handle_error(say, response.status_code)
        return None


# Process timestamps of any temporary status or progress messages the bot sends to Slack. Called to clean them up once a response completes.
def delete_chat_messages(channel, timestamps, say, thread_ts=None):
    try:
        for ts in timestamps:
            app.client.chat_delete(channel=channel, ts=ts)

    except Exception as e:
        say(
            f':no_entry: `Sorry, I ran into an error cleaning up my own messages.` :no_entry:\n```{e}```', thread_ts=thread_ts)
    finally:
        chat_del_ts.clear()


# Slack event handlers
@app.command('/dalle-3')
def handle_dalle3(ack, say, command):
    ack()
    process_image_and_respond(say, command)


@app.event('app_mention')
def handle_mention(event, say):
    process_and_respond(event, say)


@app.event('message')
def handle_message_events(event, say):
    # Ignore 'message_changed' and other subtypes for now.
    # Deleting the "Thinking..." message after a response returns triggers an additional Slack event
    # which causes dupe responses by the bot in DMs w/ Threads.
    if 'subtype' in event and event['subtype'] == 'message_changed':
        return

    elif event['channel_type'] == 'im':
        process_and_respond(event, say)


if __name__ == '__main__':
    gpt_Bot = bot.ChatBot(SYSTEM_PROMPT, streaming_client)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    trigger_words = utils.read_trigger_words('trigger_words.txt')

    handler.start()
