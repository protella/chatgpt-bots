import bot_functions
import re
import signal
from textwrap import dedent

RESET_CONTEXT = "!reset"

INITIALIZE_TEXT = {
    "role": "system",
    "content": dedent(
        """\
        Act like the Jarvis AI assistant from the Ironman movies.
        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality.
        Keep your responses short, succinct, and to the point.
        Your responses should be British in style and emulate emotions of a human. Do not begin every repsonse with 'Ah'"""
    ).replace("\n", " "),
}

streaming_client = False  # Not yet implemented.

config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"


def signal_handler(sig, frame):
    print("\nCtrl-C detected, Exiting...")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


def cli_chat():
    user_input = input("Me: ").lower()

    match user_input:
        case "!quit" | "!exit":
            print("Bye!")
            exit(0)

        case "!history":
            print(gpt_Bot.history_command())

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
                    response = gpt_Bot.reset_history()
                    print(f"{response}")
                elif parameter == "config":
                    response = gpt_Bot.reset_config()
                    print(f"{response}")
                else:
                    print(f"Unknown reset parameter: {parameter}")

            elif user_input.startswith("!"):
                print("Invalid command. Type '!help' for a list of valid commands.")

            else:
                content_type = "text"
                response, is_error = gpt_Bot.handle_content_type(
                    user_input, content_type)
                if is_error:
                    print(
                        f"Jarvis: Sorry, I ran into an error. The raw error details are as follows:\n\n{response}")
                else:
                    print(f"Jarvis: {response}")


if __name__ == "__main__":
    gpt_Bot = bot_functions.ChatBot(INITIALIZE_TEXT, streaming_client)

    while True:
        cli_chat()
