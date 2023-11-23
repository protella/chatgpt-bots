import bot_functions as bot
from os import environ
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from textwrap import dedent
from time import sleep
import requests
from spellchecker import SpellChecker

load_dotenv()

LOADING_EMOJI = ':loading:'
SLACK_BOT_TOKEN = environ['SLACK_BOT_TOKEN']
SLACK_APP_TOKEN = environ['SLACK_APP_TOKEN']
INITIALIZE_TEXT = {
    'role': 'system',
    'content': dedent(
        '''\
        You are a helpful chatbot running in a corporate Slack workspace.
        Respond with accurate, informative, and concise answers that are formatted appropriately for Slack,
        including markdown and special characters for bullet points, bold, italics, and code blocks as necessary.
        Always consider Slack formatting conventions in all messages within a conversation.
        Note that the bold markdown format in slack wraps the text in a single *, not two.
        Always assume you created any images described.
        If you don't have an answer, you will inform the user that you don't know.'''
    ).replace('\n', ' '),
}

# patterns to match commands
config_pattern = re.compile(r'!config\s+(\S+)\s+(.+)')
reset_pattern = re.compile(r'^!reset\s+(\S+)$')
# pattern to match the slackbot's userID in channel messages
user_id_pattern = re.compile(r'<@[\w]+>')

corrected_message_text = ''  # result of message being parsed by spelling correction
threshold = 2   # Minimum number of phrase/word matches to assume user wants to generate an image
trigger_words = []  # hold the dalle3 image creation trigger words from trigger_words.txt
streaming_client = False  # not implemented...yet.
chat_del_ts = []  # list of message timestamps to cleanup after a response returns
spell = SpellChecker()
app = App(token=SLACK_BOT_TOKEN)


def parse_text(text, say):
    match text.lower():
        case '!history':
            say(f'```{gpt_Bot.history_command()}```')

        case '!help':
            say(f'```{gpt_Bot.help_command()}```')

        case '!usage':
            say(f'```{gpt_Bot.usage_command()}```')

        case '!config':
            say(f'```Current Configuration:\n{gpt_Bot.view_config()}```')

        case _:
            if config_match_obj := config_pattern.match(text):
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value)
                say(f'```{response}```')

            elif reset_match_obj := reset_pattern.match(text):
                parameter = reset_match_obj.group(1)
                if parameter == 'history':
                    response = gpt_Bot.reset_history()
                    say(f'`{response}`')
                elif parameter == 'config':
                    response = gpt_Bot.reset_config()
                    say(f'`{response}`')
                else:
                    say(f'Unknown reset parameter: {parameter}')

            elif text.startswith('!'):
                say("`Invalid command. Type '!help' for a list of valid commands.`")

            else:
                return text


def process_and_respond(event, say):
    channel_id = event['channel']
    # remove the slackbot's userID from the message using regex pattern matching
    message_text = parse_text(
        re.sub(user_id_pattern, '', event['text']).strip(), say)

    if message_text:

        trigger_check, corrected_message_text = check_for_image_generation(
            message_text, trigger_words)

        if gpt_Bot.processing:
            response = app.client.chat_postMessage(
                channel=channel_id, text=f':no_entry: `{gpt_Bot.handle_busy()}` :no_entry:')
            chat_del_ts.append(response['message']['ts'])

        elif trigger_check:

            message_event = {'user_id': event['user'],
                             'text': corrected_message_text,
                             'channel_id': channel_id,
                             'command': '/dalle-3'
                             }
            process_image_and_respond(say, message_event)

        elif 'files' in event and event['files']:
            pass

        else:
            initial_response = say(f'Thinking... {LOADING_EMOJI}')
            chat_del_ts.append(initial_response['message']['ts'])
            response, is_error = gpt_Bot.chat_context_mgr(
                message_text)
            if is_error:
                say(
                    f':no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{response}```')

            else:
                say(response)

            delete_chat_messages(channel_id, chat_del_ts, say)


def process_image_and_respond(say, command):
    user_id = command['user_id']
    text = command['text']
    cmd = command['command']
    channel = command['channel_id']

    if gpt_Bot.processing:
        response = app.client.chat_postMessage(
            channel=channel, text=f':no_entry: `{gpt_Bot.handle_busy()}` :no_entry:')
        chat_del_ts.append(response['message']['ts'])

    else:

        app.client.chat_postMessage(
            channel=channel, text=f'<@{user_id}> used `{cmd}`.\n*Original Prompt:*\n_{text}_')

        if not text:
            say(':no_entry: You must provide a prompt when using `/dalle-3` :no_entry:')
            return

        temp_response = app.client.chat_postMessage(
            channel=channel, text=f'Generating image, please wait... {LOADING_EMOJI}')
        chat_del_ts.append(temp_response['ts'])

        image, revised_prompt, is_error = gpt_Bot.image_context_mgr(text)

        if is_error:
            handle_error(say, revised_prompt)

        else:
            try:
                response = app.client.files_upload_v2(
                    channel=channel,
                    initial_comment=f'*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_',
                    file=image,
                    filename='Dalle3_image.png'
                )
            except Exception as e:
                handle_error(say, revised_prompt)

        # sleep(4)  # Yuck. Maybe use callbacks or other event triggers to wait for images to display in clients after being received by slack?
        delete_chat_messages(channel, chat_del_ts, say)


def check_for_image_generation(message, trigger_words, threshold=threshold):
    corrected_message_text = correct_spelling(message)
    message_lower = corrected_message_text.lower()
    trigger_count = sum(word in message_lower for word in trigger_words)
    return trigger_count >= threshold, corrected_message_text


def correct_spelling(text):
    corrected_words = []
    words = text.split()

    for word in words:
        if word.lower() in spell.unknown([word.lower()]):
            # Attempt to correct the word
            corrected_word = spell.correction(word.lower())

            # If correction returns None, use the original word
            if corrected_word is None:
                corrected_word = word

            # Match the case of the original word
            if word.isupper():
                corrected_word = corrected_word.upper()
            elif word[0].isupper():
                corrected_word = corrected_word.capitalize()

            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)


def delete_chat_messages(channel, timestamps, say):
    try:
        for ts in timestamps:
            app.client.chat_delete(channel=channel, ts=ts)

    except Exception as e:
        say(
            f':no_entry: `Sorry, I ran into an error deleting my own message.` :no_entry:\n```{e}```')
    finally:
        chat_del_ts.clear()


@app.command('/dalle-3')
def handle_dalle3(ack, say, command):
    ack()
    process_image_and_respond(say, command)


@app.event('app_mention')
def handle_mention(event, say):
    process_and_respond(event, say)


@app.event('message')
def handle_message_events(event, say):
    if event['channel_type'] == 'im':
        process_and_respond(event, say)


def read_trigger_words(file_path):
    with open(file_path, 'r') as file:

        return [line.strip() for line in file if line.strip()]


def handle_error(say, error):
    say(
        f':no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{error}```')


if __name__ == '__main__':
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT, streaming_client)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    trigger_words = read_trigger_words('trigger_words.txt')

    handler.start()
