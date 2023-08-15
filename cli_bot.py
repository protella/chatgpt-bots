import bot_functions
import re

RESET_CONTEXT = "!reset"
INITIALIZE_TEXT = {
    "role": "system",
    "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. 
                        Your responses should be British in style and emulate emotions of a human.""".replace(
        "    ", ""
    ),
}
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"


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
                print(f"Jarvis: {gpt_Bot.context_mgr(user_input)}")


if __name__ == "__main__":
    gpt_Bot = bot_functions.ChatBot(INITIALIZE_TEXT)

    while True:
        cli_chat()
