import base64
from copy import deepcopy
from bot_functions import GPT_MODEL
from prompts import IMAGE_CHECK_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT
import requests


def create_dalle3_prompt(message, gpt_Bot, thread_id):
    """
    Use ChatGPT to generate a DALL-E 3 prompt based on the message and chat history.
    
    This function creates a copy of the conversation history, adds the user's message,
    changes the system prompt to the image generation prompt, and gets a response from GPT
    that can be used as a DALL-E 3 prompt.
    
    Args:
        message (str): The user's message.
        gpt_Bot (ChatBot): The ChatBot instance.
        thread_id (str): The ID of the thread/conversation.
        
    Returns:
        object: The GPT response containing the DALL-E 3 prompt.
    """
    gpt_Bot.conversations[thread_id]["processing"] = True
    
    # Create a deep copy of the conversation history to avoid modifying the original
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"])
    
    # Add the user's message to the chat history
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    
    # Change the system prompt to the image generation prompt
    chat_history[0]['content'] = IMAGE_GEN_SYSTEM_PROMPT
    
    # Get a response from GPT to use as a DALL-E 3 prompt
    dalle3_prompt = gpt_Bot.get_gpt_response(chat_history, GPT_MODEL)
    
    # For debugging
    # print(f'\nDalle-3 Prompt: {dalle3_prompt.content}\n')

    gpt_Bot.conversations[thread_id]["processing"] = False
    return dalle3_prompt


def check_for_image_generation(message, gpt_Bot, thread_id):
    """
    Use GPT-4 to check if the user is requesting an image generation.
    
    This function creates a copy of the conversation history, adds the user's message,
    changes the system prompt to the image check prompt, and gets a response from GPT
    that indicates whether the user is requesting an image generation.
    
    Args:
        message (str): The user's message.
        gpt_Bot (ChatBot): The ChatBot instance.
        thread_id (str): The ID of the thread/conversation.
        
    Returns:
        bool: True if the user is requesting an image generation, False otherwise.
    """
    gpt_Bot.conversations[thread_id]["processing"] = True
    
    # Create a deep copy of the conversation history to avoid modifying the original
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"])
    
    # Add the user's message to the chat history
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    
    # Change the system prompt to the image check prompt
    chat_history[0]['content'] = IMAGE_CHECK_SYSTEM_PROMPT

    # Set temperature to 0.0 to be fully deterministic and reduce randomness
    # Low max tokens helps force True/False response
    is_image_request = gpt_Bot.get_gpt_response(
        chat_history, 
        GPT_MODEL, 
        temperature=0.0, 
        max_completion_tokens=5
    )
    
    gpt_Bot.conversations[thread_id]["processing"] = False
    
    # For debugging
    # print(f'\nImage Request Check: {is_image_request.content}\n')
    
    # Return True if the response is 'true', False otherwise
    return is_image_request.content.strip().lower() == 'true'


def download_and_encode_file(say, file_url, bot_token):
    """
    Download a file from Slack and encode it as base64.
    
    In order to download files from Slack, the bot's request needs to be authenticated
    to the workspace via the Slackbot token.
    
    Args:
        say (callable): A function to send messages to Slack.
        file_url (str): The URL of the file to download.
        bot_token (str): The Slackbot token for authentication.
        
    Returns:
        str or None: The base64-encoded file content, or None if an error occurred.
    """
    headers = {"Authorization": f"Bearer {bot_token}"}
    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        return base64.b64encode(response.content).decode("utf-8")
    else:
        handle_error(say, response.status_code)
        return None


def handle_error(say, error, thread_ts=None):
    """
    Handle errors by sending an error message to Slack.
    
    Args:
        say (callable): A function to send messages to Slack.
        error (any): The error to handle.
        thread_ts (str, optional): The timestamp of the thread to reply to. Defaults to None.
    """
    say(
        f":no_entry: `An error occurred. Error details:` :no_entry:\n```{error}```",
        thread_ts=thread_ts,
    )


def format_message_for_debug(conversation_history):
    """
    Format a conversation history for debugging purposes.
    
    This function takes a conversation history and formats it as a string for debugging,
    replacing image data with placeholders.
    
    Args:
        conversation_history (dict): The conversation history to format.
        
    Returns:
        str: A formatted string representation of the conversation history.
    """
    formatted_output = []
    for message in conversation_history['messages']:
        role = message['role']
        content = message['content']
        
        message_texts = []  # To collect text and placeholders for each message

        # Check if content is a list (typically for 'user' or 'assistant' with mixed content)
        if isinstance(content, list):
            # Process each content item in the list
            for item in content:
                if item['type'] == 'text':
                    message_texts.append(item['text'])
                elif item['type'] == 'image_url':
                    # Add a placeholder for images
                    message_texts.append("[Image Data]")
        
        elif isinstance(content, str):
            # Directly append the content if it's a string
            message_texts.append(content)
                    
        # Join all parts of the message into a single string and append to the output
        formatted_message = ' '.join(message_texts)
        formatted_output.append(f"-- {role.capitalize()}: {formatted_message}")
    
    return "\n".join(formatted_output)


###############################