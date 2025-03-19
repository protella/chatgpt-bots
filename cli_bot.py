import re
import signal
import os
from prompts import CLI_SYSTEM_PROMPT
import bot_functions

from dotenv import load_dotenv
from openai import OpenAI
from logger import get_log_level, get_logger, log_session_marker

GPT_MODEL = "chatgpt-4o-latest"
DALLE_MODEL = "dall-e-3"
conversations = {}
if "BOT_LOG_LEVEL" in os.environ:
    del os.environ["BOT_LOG_LEVEL"]

load_dotenv()  # load auth tokens from .env file

# Configure logging level from environment variable with fallback to INFO
LOG_LEVEL_NAME = os.environ.get("BOT_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = get_log_level(LOG_LEVEL_NAME)
# Initialize logger with the configured log level
logger = get_logger('cli_bot', LOG_LEVEL)
log_session_marker(logger, "START")  # Add session start marker


# Configuration variables
BOT_NAME = "Jarvis"
streaming_client = False  # Not yet implemented.
thread_id = "0"  # Single thread for CLI

# Patterns to match commands
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"

client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
def signal_handler(sig, frame):
    """
    Handle Ctrl-C signal to gracefully exit the program.
    
    Args:
        sig: The signal number.
        frame: The current stack frame.
    """
    logger.info("Ctrl-C detected, exiting...")
    log_session_marker(logger, "END")  # Add session end marker
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
        user_input = input("Me: ")
        logger.debug(f"Received user input: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
    except EOFError:
        logger.info("Ctrl-D detected, exiting...")
        log_session_marker(logger, "END")  # Add session end marker
        print("\nCtrl-D detected, Exiting...")
        exit(0)

    # Process commands
    match user_input.lower():
        case "!quit" | "!exit":
            logger.info("User requested exit")
            log_session_marker(logger, "END")  # Add session end marker
            print("Bye!")
            exit(0)

        case "!history":
            logger.debug("User requested history")
            print(gpt_Bot.history_command(thread_id=thread_id))

        case "!help":
            logger.debug("User requested help")
            print(gpt_Bot.help_command())

        case "!usage":
            logger.debug("User requested usage information")
            print(gpt_Bot.usage_command())

        case "!config":
            logger.debug("User requested config view")
            print(gpt_Bot.view_config(thread_id))
            return

        case _:
            # Check for config and reset commands
            config_match_obj = re.match(config_pattern, user_input.lower())
            reset_match_obj = re.match(reset_pattern, user_input.lower())
            
            if config_match_obj:
                setting, value = config_match_obj.groups()
                logger.info(f"Changing config: {setting} = {value}")
                response = gpt_Bot.set_config(setting, value, thread_id)
                print(f"{response}")
                return

            elif reset_match_obj:
                parameter = reset_match_obj.group(1)
                if parameter == "history":
                    # Reset the conversation history
                    logger.info("Resetting chat history")
                    gpt_Bot.conversations[thread_id] = {
                        "messages": [CLI_SYSTEM_PROMPT],
                        "history_reloaded": True,
                    }
                    print("Chat History cleared.")
                elif parameter == "config":
                    logger.info("Resetting configuration")
                    response = gpt_Bot.reset_config(thread_id)
                    print(f"{response}")
                else:
                    logger.warning(f"Unknown reset parameter: {parameter}")
                    print(f"Unknown reset parameter: {parameter}")

            elif user_input.startswith("!"):
                logger.warning(f"Invalid command: {user_input}")
                print("Invalid command. Type '!help' for a list of valid commands.")

            else:
                # Process normal chat message
                logger.debug("Processing chat message")
                
                if conversations[thread_id]["previous_response_id"] is None:
                    logger.debug("First message in conversation - including system prompt")
                    conversations[thread_id]["messages"] = [
                        CLI_SYSTEM_PROMPT, 
                        {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
                    ]
                else:
                    logger.debug(f"Using previous_response_id: {conversations[thread_id]['previous_response_id']}")
                    conversations[thread_id]["messages"] = [
                        {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
                    ]
                                    
                logger.info("Creating new GPT response")
                response, is_error = create_gpt_response(
                    conversations[thread_id]["messages"],
                    GPT_MODEL,
                    conversations[thread_id]["previous_response_id"]
                )
                conversations[thread_id]["previous_response_id"] = response.id

                if is_error:
                    logger.error(f"Error creating GPT response: {response}")
                    print(
                        f"{BOT_NAME}: Sorry, I ran into an error. The raw error details are as follows:\n\n{response}"
                    )
                else:
                    logger.debug(f"Response output length: {len(response.output_text)} chars")
                    print(f"{BOT_NAME}: {response.output_text}")

def create_gpt_response(messages_history, model, previous_response_id=None, temperature=None, max_output_tokens=2048):
    logger.debug(f"Creating GPT response with model: {model}, max tokens: {max_output_tokens}")
    
    if previous_response_id:
        logger.debug(f"Using previous_response_id: {previous_response_id}")
    
    try:
        # Call the OpenAI API for chat response
        logger.debug("Sending request to OpenAI API")
        response = client.responses.create(
            model=model,
            input=messages_history,
            max_output_tokens=max_output_tokens,
            store=True,
            previous_response_id=previous_response_id
        )
        is_error = False
        logger.debug(f"Successfully created response with ID: {response.id}")
        return response, is_error

    except Exception as e:
        is_error = True
        logger.error(f"Error creating GPT response: {e}", exc_info=True)
        print(f"##################\n{e}\n##################")
        return e, is_error

def main():
    """
    Main function to run the CLI chat bot.
    """
    logger.info("Starting CLI chat bot")
    # Initialize the ChatBot
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        logger.critical("OPENAI_KEY environment variable not set")
        print("\nError: OPENAI_KEY environment variable is not set.")
        print("Please set the OPENAI_KEY environment variable with your OpenAI API key.")
        print("Check the README.md for how to setup the .env file.")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)
    
    global gpt_Bot
    logger.info("Initializing ChatBot")
    gpt_Bot = bot_functions.ChatBot(CLI_SYSTEM_PROMPT, streaming_client)
    
    # Initialize conversation
    logger.debug("Initializing conversation")
  
    conversations[thread_id] = {
        "previous_response_id": None,
        "history_reloaded": True,
    }

    # Main chat loop
    logger.info("CLI Bot initialized and ready")
    print(f"CLI Chat Bot initialized. Type '!help' for commands or '!exit' to quit.")
    print(f"Chat with {BOT_NAME} below:")
    print("-" * 50)
    
    while True:
        cli_chat()


if __name__ == "__main__":
    # Run the main function
    logger.debug("Script started")
    main()
