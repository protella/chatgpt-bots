import base64
from copy import deepcopy
from bot_functions import GPT_MODEL
import requests

IMAGE_CHECK_SYSTEM_PROMPT = """You will be provided with a user's chat message for a chatgpt chatbot integration. 
Determine if the user's intent is to request an image generation or if the message is just part of the ongoing chat conversation. 
Also consider if the message is in the form of a question when making your determination.
Respond with 'True' for image requests and 'False' otherwise. No other text should be provided except 'True' or 'False'.
For example:
User message: "Can you create an image of a sunset over the mountains?"
Response: True
User message: "What do you think about the new policy?"
Response: False"""

IMAGE_GEN_SYSTEM_PROMPT = """You will be provided with a user's chat message and context history for a chatgpt chatbot integration.
The message has been predetermined to be a request for a Dalle-3 generative art image. 
Based solely on the chat history and user message provided, format your response as a straightforward 
generative art prompt without any introductory text or explanation. 
Ensure the prompt is descriptive and detailed, but not too long."""


# Use ChatGPT to generate a Dalle-3 prompt based on the message and chat history
def create_dalle3_prompt(message, gpt_Bot, thread_id):
    gpt_Bot.conversations[thread_id]["processing"] = True
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"])
    
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
            )
    chat_history[0]['content'] = IMAGE_GEN_SYSTEM_PROMPT
    
    dalle3_prompt = gpt_Bot.get_gpt_response(chat_history, GPT_MODEL)
    
    # print(f'\nDalle-3 Prompt: {dalle3_prompt.content}\n')

    gpt_Bot.conversations[thread_id]["processing"] = False
    return dalle3_prompt

# Use GPT4 to check if the user is requesting an image
def check_for_image_generation(message, gpt_Bot, thread_id):
    gpt_Bot.conversations[thread_id]["processing"] = True
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"])
    
    chat_history.append(
                {"role": "user", "content": [{"type": "text", "text": message}]}
            )
    chat_history[0]['content'] = IMAGE_CHECK_SYSTEM_PROMPT

    # set temperature to 0.0 to be fully deterministic and reduce randomness for chance of non True/False response. Low Max tokens helps force T/F Response
    is_image_request = gpt_Bot.get_gpt_response(chat_history, GPT_MODEL, temperature = 0.0, max_tokens=5)
    
    gpt_Bot.conversations[thread_id]["processing"] = False
    # print(f'\nImage Request Check: {is_image_request.content}\n')
    return is_image_request.content.strip().lower() == 'true'


# In order to download Files from Slack, the bot's request needs to be authenticated to the workspace via the Slackbot token
def download_and_encode_file(say, file_url, bot_token):
    headers = {"Authorization": f"Bearer {bot_token}"}
    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        return base64.b64encode(response.content).decode("utf-8")
    else:
        handle_error(say, response.status_code)
        return None


def handle_error(say, error, thread_ts=None):
    say(
        f":no_entry: `An error occurred. Error details:` :no_entry:\n```{error}```",
        thread_ts=thread_ts,
    )

############## DEBUG ##########
def format_message_for_debug(conversation_history):
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