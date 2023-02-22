import bot_functions

RESET_CONTEXT = "!reset"
INITIALIZE_TEXT = 'Marv is a chatbot that reluctantly answers questions with witty and sarcastic responses. Preface your responses with "Marv: "'


def cli_chat():
    user_input = input("Me: ")

    if user_input == "!reset":
        print(MyBot.reset_history())

    elif user_input.lower() == "quit" or user_input.lower() == "exit":
        print("Bye!")
        exit(0)

    else:
        user_input = "Me: " + user_input
        print(MyBot.context_mgr(user_input))

    return


if __name__ == "__main__":
    MyBot = bot_functions.ChatBot(INITIALIZE_TEXT)

    while True:
        cli_chat()
