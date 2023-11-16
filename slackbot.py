import bot_functions as bot
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from textwrap import dedent

load_dotenv()

LOADING_EMOJI = ':loading:'
SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
SLACK_APP_TOKEN = os.environ['SLACK_APP_TOKEN']
INITIALIZE_TEXT = {
    'role': 'system',
    'content': dedent(
        '''\
        You are a helpful chatbot running in a corporate Slack workspace.
        Respond with accurate, informative, and concise answers that are formatted appropriately for Slack,
        including markdown and special characters for bullet points, bold, italics, and code blocks as necessary.
        Always consider Slack formatting conventions in all messages within a conversation.
        If you don't have an answer, you will inform the user that you don't know.'''
    ).replace('\n', ' '),
}
# INITIALIZE_TEXT = {
#     'role': 'system',
#     'content': dedent(
#         '''\
#         Act like the Jarvis AI assistant from the Ironman movies.
#         Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality.
#         Keep your responses short, succinct, and to the point. Emulate emotions of a human.'''
#     ).replace('\n', ' '),
# }

# patterns to match commands
config_pattern = re.compile(r'!config\s+(\S+)\s+(.+)')
reset_pattern = re.compile(r'^!reset\s+(\S+)$')
# pattern to match the slackbot's userID in channel messages
user_id_pattern = re.compile(r'<@[\w]+>')

content_type = 'text'
streaming_client = False
chat_del_ts = []
app = App(token=SLACK_BOT_TOKEN)


# @app.event('message')
# def handle_message(event, say):
#     print(event['user'])


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
        if gpt_Bot.processing:
            response = say(
                f':no_entry: `{gpt_Bot.handle_content_type(message_text, content_type)}` :no_entry:')
            chat_del_ts.append(response['message']['ts'])

        else:
            initial_response = say(f'Thinking... {LOADING_EMOJI}')
            chat_del_ts.append(initial_response['message']['ts'])
            response, is_error = gpt_Bot.handle_content_type(
                message_text, content_type)
            if is_error:
                say(
                    f':no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{response}```')

            else:
                say(response)

            delete_chat_messages(channel_id, chat_del_ts, say)


def delete_chat_messages(channel, timestamps, say):
    try:
        for ts in timestamps:
            app.client.chat_delete(channel=channel, ts=ts)
        chat_del_ts.clear()

    except Exception as e:
        say(
            f':no_entry: `Sorry, I ran into an error deleting my own message.` :no_entry:\n```{e}```')


@app.command('/dalle3')
def handle_dalle3(ack, say, command):
    ack()
    if gpt_Bot.processing:
        response = say(
            f':no_entry: `{gpt_Bot.handle_busy()}` :no_entry:')
        chat_del_ts.append(response['message']['ts'])
    else:
        content_type = "image"
        user_id = command['user_id']
        text = command['text']
        cmd = command['command']
        channel = command['channel_id']

        app.client.chat_postMessage(
            channel=channel, text=f'<@{user_id}> used `{cmd}`.\n*Prompt:*\n_{text}_')

        temp_response = app.client.chat_postMessage(
            channel=channel, text=f'Generating image, please wait... {LOADING_EMOJI}')

        ts = [temp_response['ts']]
        response, is_error = gpt_Bot.handle_content_type(text, content_type)

        if is_error:
            say(
                f':no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{response}```')

        else:
            say(f"{response.data[0].url}\n*Revised Prompt:*\n_{response.data[0].revised_prompt}_")

        delete_chat_messages(channel, ts, say)

        # print(response)


@app.event('app_mention')
def handle_mention(event, say):
    process_and_respond(event, say)


@app.event('message')
def handle_message_events(event, say):
    if event['channel_type'] == 'im':
        process_and_respond(event, say)


if __name__ == '__main__':
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT, streaming_client)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)

    handler.start()
