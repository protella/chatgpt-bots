import bot_functions as bot
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
RESET_CONTEXT = "!reset"
PRINT_HISTORY = "!history"
BOT_ID = "U01AF99F3JR"
CHANNEL_ID = "C04PQ5BK946"
# INITIALIZE_TEXT = {"role": "system", "content": """You are a chatbot that answers questions with accurate, 
#                    informative, witty, and humorous responses.""".replace('    ', '')}
INITIALIZE_TEXT = {"role": "system", "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. Emulate emotions of a human.""".replace('    ', '')}


app = App(token=SLACK_BOT_TOKEN)


# @app.event("message")
# def handle_message(event, say):
#     print(event["user"])


@app.event("app_mention")
def handle_mention(event, say):
    
    print(event["text"][15:])
    
    # if event["user"] == BOT_ID:
    #     return
    if RESET_CONTEXT in event["text"].lower():
        say(f"{gpt_Bot.reset_history()}")
        return
    
    if PRINT_HISTORY in event["text"].lower():
        say(f"{gpt_Bot.messages}")
        return
    
    say(f"{gpt_Bot.context_mgr(event['text'])}")

if __name__ == "__main__":
    gpt_Bot = bot.ChatBot(INITIALIZE_TEXT)
    handler = SocketModeHandler(
        app,
        SLACK_APP_TOKEN,
    )

    handler.start()
