import bot_functions

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


def cli_chat():
    user_input = input("Me: ")

    match user_input.lower():
        case "!quit" | "!exit":
            print("Bye!")
            exit(0)

        case "!reset":
            print(MyBot.reset_history())

        case "!history":
            print(MyBot.history_command())

        case "!help":
            print(MyBot.help_command())

        case "!usage":
            print(MyBot.usage_command())

        case _:
            print(f"Jarvis: {MyBot.context_mgr(user_input)}")


if __name__ == "__main__":
    MyBot = bot_functions.ChatBot(INITIALIZE_TEXT)

    while True:
        cli_chat()
