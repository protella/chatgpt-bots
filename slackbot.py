import openai
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_KEY = os.environ["OPENAI_KEY"]
RESET_CONTEXT = "!reset"
INITIALIZE_TEXT = "You are a chatbot that answers questions with accurate, informative, and sometimes humorous responses."
# INITIALIZE_TEXT = (
#     "You are a chatbot that responds kindly and accurately to all questions."
# )
BOT_ID = "U01AF99F3JR"
CHANNEL_ID = "C04PQ5BK946"
initialized = 0
history = []


def reset_history():
    global history
    global INITIALIZE_TEXT

    history = [INITIALIZE_TEXT]

    return "`Rebooting. Beep Beep Boop. My memory has been wiped!`"


app = App(token=SLACK_BOT_TOKEN)


# @app.event("message")
# def handle_message(event, say):
#     print(event["user"])


@app.event("app_mention")
def handle_mention(event, say):
    global history
    global INITIALIZE_TEXT

    # if event["user"] == BOT_ID:
    #     return
    if RESET_CONTEXT in event["text"].lower():
        say(reset_history())
        return

    # print(event["text"][15:])
    say(f"{context_mgr(event['text'][15:].strip())}")


def context_mgr(ai_prompt):
    global history
    global initialized
    global INITIALIZE_TEXT

    if initialized == 0:
        history.append(INITIALIZE_TEXT)
        initialized = 1

    chat_input = "Context: " + "\n".join(history) + "\n" + ai_prompt
    output = get_ai_response(chat_input)
    history += [ai_prompt, output.strip()]
    # print(output)
    return output.strip()


def get_ai_response(ai_prompt):
    global history
    global INITIALIZE_TEXT
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=ai_prompt,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            # stop=['']
        )
        return response.choices[0].text
    except openai.error.InvalidRequestError as e:
        reset_history()
        return "Sorry, I ran out of token memory. Rebooting. Beep Beep Boop."

    except openai.error.RateLimitError as r:
        return "My servers are too busy! Try your request again."


if __name__ == "__main__":
    openai.api_key = OPENAI_KEY

    handler = SocketModeHandler(
        app,
        SLACK_APP_TOKEN,
    )

    handler.start()
