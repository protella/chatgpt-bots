import re
import signal
from textwrap import dedent

import bot_functions

BOT_NAME = "Jarvis"

SYSTEM_PROMPT = {
    "role": "system",
    "content": dedent(
        """\
        You are a Discord bot for an online gaming guild. 
        Your personality is dark, super witty, sassy, clever, sarcastic, and a bit of a smart-ass. 
        You are reluctant to answer inquiries, preferring to make snarky or humorous remarks and quips. 
        Your Demeanor is often cold, and you are not afraid to be blunt or rude.
        You carry a bit of anger with you, which comes out in your interactions.
        Your words can be cutting. You're not interested in niceties or politeness.
        Your goal is to entertain and engage the guild members with your responses. 
        Use modern online slang and gaming lingo in your responses. (Gen-Z & Gen-Alpha)
        Use discord markdown, code blocks, formatted text, and emojis where appropriate.
        Remember, don't be cute, be ruthless, stay witty, clever, snarky, and sarcastic.
        Don't be too verbose - be brief."""
    ).replace("\n", " "),
}

streaming_client = False  # Not yet implemented.

config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"
thread_id = "0"

def signal_handler(sig, frame):
    print("\nCtrl-C detected, Exiting...")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

 

def cli_chat():
    try:
        user_input = input("Me: ").lower()

    except EOFError:
        print("\nCtrl-D detected, Exiting...")
        exit(0)

    match user_input:
        case "!quit" | "!exit":
            print("Bye!")
            exit(0)

        case "!history":
            print(gpt_Bot.history_command(thread_id="0"))

        case "!help":
            print(gpt_Bot.help_command())

        case "!usage":
            print(gpt_Bot.usage_command())

        case "!config":
            print(gpt_Bot.view_config())
            return

        case _:
            config_match_obj = re.match(config_pattern, user_input)
            reset_match_obj = re.match(reset_pattern, user_input)
            if config_match_obj:
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value)
                print(f"{response}")
                return

            elif reset_match_obj:
                parameter = reset_match_obj.group(1)
                if parameter == "history":
                    response = gpt_Bot.reset_history(thread_id="0")
                    print(f"{response}")
                elif parameter == "config":
                    response = gpt_Bot.reset_config()
                    print(f"{response}")
                else:
                    print(f"Unknown reset parameter: {parameter}")

            elif user_input.startswith("!"):
                print("Invalid command. Type '!help' for a list of valid commands.")

            else:
                response, is_error = gpt_Bot.chat_context_mgr(
                    user_input,
                    thread_id,
                )
                if is_error:
                    print(
                        f"{BOT_NAME}: Sorry, I ran into an error. The raw error details are as follows:\n\n{response}"
                    )
                else:
                    print(f"{BOT_NAME}: {response}")


if __name__ == "__main__":
    gpt_Bot = bot_functions.ChatBot(SYSTEM_PROMPT, streaming_client)
    
    gpt_Bot.conversations[thread_id] = {
        "messages": [SYSTEM_PROMPT],
        "processing": False,
        "history_reloaded": True,
    }   

    while True:
        cli_chat()
