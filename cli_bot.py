import re
import signal
import os
from prompts import CLI_SYSTEM_PROMPT
import bot_functions

# Configuration variables
BOT_NAME = "Jarvis"
streaming_client = False  # Not yet implemented.
thread_id = "0"  # Single thread for CLI

# Patterns to match commands
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"


def signal_handler(sig, frame):
    """
    Handle Ctrl-C signal to gracefully exit the program.
    
    Args:
        sig: The signal number.
        frame: The current stack frame.
    """
    print("\nCtrl-C detected, Exiting...")
    exit(0)


# Register signal handler for Ctrl-C
signal.signal(signal.SIGINT, signal_handler)


def cli_chat():
    """
    Process a single chat interaction in the CLI.
    
    This function gets input from the user, processes commands,
    and displays the bot's response.
    """
    try:
        user_input = input("Me: ").lower()
    except EOFError:
        print("\nCtrl-D detected, Exiting...")
        exit(0)

    # Process commands
    match user_input:
        case "!quit" | "!exit":
            print("Bye!")
            exit(0)

        case "!history":
            print(gpt_Bot.history_command(thread_id=thread_id))

        case "!help":
            print(gpt_Bot.help_command())

        case "!usage":
            print(gpt_Bot.usage_command())

        case "!config":
            print(gpt_Bot.view_config(thread_id))
            return

        case _:
            # Check for config and reset commands
            config_match_obj = re.match(config_pattern, user_input)
            reset_match_obj = re.match(reset_pattern, user_input)
            
            if config_match_obj:
                setting, value = config_match_obj.groups()
                response = gpt_Bot.set_config(setting, value, thread_id)
                print(f"{response}")
                return

            elif reset_match_obj:
                parameter = reset_match_obj.group(1)
                if parameter == "history":
                    # Reset the conversation history
                    gpt_Bot.conversations[thread_id] = {
                        "messages": [CLI_SYSTEM_PROMPT],
                        "processing": False,
                        "history_reloaded": True,
                    }
                    print("Chat History cleared.")
                elif parameter == "config":
                    response = gpt_Bot.reset_config(thread_id)
                    print(f"{response}")
                else:
                    print(f"Unknown reset parameter: {parameter}")

            elif user_input.startswith("!"):
                print("Invalid command. Type '!help' for a list of valid commands.")

            else:
                # Process normal chat message
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
    # Initialize the ChatBot
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        print("\nError: OPENAI_KEY environment variable is not set.")
        print("Please set the OPENAI_KEY environment variable with your OpenAI API key.")
        print("Check the README.md for how to setup the .env file.")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)
    
    gpt_Bot = bot_functions.ChatBot(CLI_SYSTEM_PROMPT, streaming_client)
    
    # Initialize conversation
    gpt_Bot.conversations[thread_id] = {
        "messages": [CLI_SYSTEM_PROMPT],
        "processing": False,
        "history_reloaded": True,
    }   

    # Main chat loop
    print(f"CLI Chat Bot initialized. Type '!help' for commands or '!exit' to quit.")
    print(f"Chat with {BOT_NAME} below:")
    print("-" * 50)
    
    while True:
        cli_chat()
