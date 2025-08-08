import base64
import os
from copy import deepcopy
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from logger import setup_logger, get_log_level, get_logger

# Unset any existing log level environment variables to ensure .env values are used
if "BOT_LOG_LEVEL" in os.environ:
    del os.environ["BOT_LOG_LEVEL"]

load_dotenv()  # load auth tokens from .env file

# Configure logging level from environment variable with fallback to INFO
LOG_LEVEL_NAME = os.environ.get("BOT_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = get_log_level(LOG_LEVEL_NAME)
# Initialize logger with the configured log level
logger = get_logger('bot_functions', LOG_LEVEL)

# Default models: https://platform.openai.com/docs/models
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-5-chat-latest")
DALLE_MODEL = os.environ.get("DALLE_MODEL", "dall-e-3")


class ChatBot:
    """
    A class that handles interactions with OpenAI's API for chat completions and image generation.
    
    This class manages conversations with OpenAI's GPT models and DALL-E models,
    handling message history, configuration options, and API responses.
    """
    
    def __init__(self, SYSTEM_PROMPT, streaming_client=False, show_dalle3_revised_prompt=False):
        """
        Initialize a new ChatBot instance.
        
        Args:
            SYSTEM_PROMPT (dict): The system prompt to use for conversations.
            streaming_client (bool, optional): Whether to use streaming for responses. Defaults to False.
            show_dalle3_revised_prompt (bool, optional): Whether to show DALL-E 3's revised prompts. Defaults to False.
        """
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.conversations = {}
        self.show_dalle3_revised_prompt = show_dalle3_revised_prompt
        self.streaming_client = streaming_client  # ToDo: Implement streaming support
        self.usage = {}
        
        # Default configuration options
        self.config_option_defaults = {
            "temperature": .8,  # 0.0 - 2.0 (GPT-5 reasoning models only support 1.0)
            "top_p": 1, # 0.0 - 1.0 (not supported in GPT-5 reasoning models)
            "max_completion_tokens": 2048,  # max 4096
            "reasoning_effort": "medium",  # GPT-5 only: minimal, low, medium, high
            "verbosity": "medium",  # GPT-5 only: low, medium, high
            "custom_init": "",
            "gpt_model": GPT_MODEL,
            "dalle_model": DALLE_MODEL,
            "size": "1024x1024",  # Dalle3 parameter: 1024x1024, 1024x1792, or 1792x1024
            "quality": "hd",  # Dalle3 parameter: standard or hd
            "style": "natural",  # Dalle3 parameter: natural or vivid
            "number": 1,  # number of images. Only 1 supported for Dalle3
            "detail": "auto",  # vision parameter: auto, low, high
            "d3_revised_prompt": self.show_dalle3_revised_prompt,
            "system_prompt": self.SYSTEM_PROMPT["content"] # content of system prompt
        }
        self.current_config_options = self.config_option_defaults.copy()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
        logger.info("ChatBot initialized with system prompt")
        logger.debug(f"Initial config: {self.current_config_options}")
        
    def chat_context_mgr(self, message_text, thread_id, files=""):
        """
        Manage the context for a chat conversation.
        
        Args:
            message_text (str): The message text from the user.
            thread_id (str): The ID of the thread/conversation.
            files (str, optional): Any files attached to the message. Defaults to "".
            
        Returns:
            tuple: (response_content, is_error) where response_content is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        logger.info(f"Processing chat message for thread {thread_id}")
        
        try:
            # Add user message to conversation history
            self.conversations[thread_id]["messages"].append(
                {"role": "user", "content": [{"type": "text", "text": message_text}]}
            )

            # Get response from GPT
            gpt_output = self.get_gpt_response(
                self.conversations[thread_id]["messages"],
                self.current_config_options["gpt_model"],
            )

            # Process the response
            if hasattr(gpt_output, "role"):
                is_error = False
                if gpt_output.role == "assistant":
                    self.conversations[thread_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": gpt_output.content}],
                        }
                    )
                return gpt_output.content, is_error
            else:
                # Handle error case
                is_error = True
                self.conversations[thread_id]["messages"].pop()  # Remove the user message on error
                return gpt_output, is_error
        except Exception as e:
            logger.error(f"Error in chat_context_mgr: {e}", exc_info=True)
            return str(e), True

    def image_context_mgr(self, message_text, thread_id):
        """
        Manage the context for an image generation conversation.
        
        Args:
            message_text (str): The prompt for image generation.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            tuple: (image, revised_prompt, is_error) where image is the generated image,
                  revised_prompt is DALL-E's revised prompt, and is_error indicates if an error occurred.
        """
        logger.info(f"Generating image for thread {thread_id}")
        logger.debug(f"Image prompt: {message_text}")
        
        try:
            # Add user message to conversation history
            self.conversations[thread_id]["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Dalle-3 User Prompt: {message_text}",
                        }
                    ],
                }
            )

            # Generate image with DALL-E
            image, revised_prompt = self.get_dalle_response(
                message_text, self.current_config_options["dalle_model"]
            )

            # Process the response
            if revised_prompt:
                is_error = False
                self.conversations[thread_id]["messages"].append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Based on the Dalle-3 user prompt, I created an image like this: {revised_prompt}
                                I can describe what this images looks like and the ideas it might convey.
                                I will act as if I created the this image in the context of this chat."""
                            }
                        ],
                    }
                )
            else:
                is_error = True
                self.conversations[thread_id]["messages"].pop()  # Remove the user message on error
                image = None
                revised_prompt = None

            return image, revised_prompt, is_error
        except Exception as e:
            logger.error(f"Error in image_context_mgr: {e}", exc_info=True)
            return None, str(e), True

    def vision_context_mgr(self, message_text, images, thread_id):
        """
        Manage the context for a vision-based conversation.
        
        Args:
            message_text (str): The message text from the user.
            images (list): List of base64-encoded images.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            tuple: (response_content, is_error) where response_content is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        logger.info(f"Processing vision message for thread {thread_id} with {len(images)} files")
        
        try:
            # Prepare the multipart message with text and images
            if not message_text:
                message_text = ""
            multi_part_msg = {
                "role": "user",
                "content": [{"type": "text", "text": f"{message_text}"}],
            }

            # Add each image to the message
            for image in images:
                new_image_element = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                        "detail": self.current_config_options["detail"],
                    },
                }
                multi_part_msg["content"].append(new_image_element)

            # Add the multipart message to conversation history
            self.conversations[thread_id]["messages"].append(multi_part_msg)

            # Get response from GPT
            gpt_output = self.get_gpt_response(
                self.conversations[thread_id]["messages"],
                self.current_config_options["gpt_model"],
            )

            # Process the response
            if hasattr(gpt_output, "role"):
                is_error = False
                if gpt_output.role == "assistant":
                    self.conversations[thread_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": gpt_output.content}],
                        }
                    )
                return gpt_output.content, is_error
            else:
                # Handle error case
                is_error = True
                self.conversations[thread_id]["messages"].pop()  # Remove the user message on error
                return gpt_output, is_error
        except Exception as e:
            logger.error(f"Error in vision_context_mgr: {e}", exc_info=True)
            return str(e), True

    def get_dalle_response(self, image_prompt, model):
        """
        Generate an image using DALL-E.
        
        Args:
            image_prompt (str): The prompt for image generation.
            model (str): The DALL-E model to use.
            
        Returns:
            tuple: (image, revised_prompt) where image is the generated image
                  and revised_prompt is DALL-E's revised prompt.
        """
        try:
            # Call the OpenAI API to generate an image
            response = self.client.images.generate(
                model=model,
                prompt=image_prompt,
                size=self.current_config_options["size"],
                quality=self.current_config_options["quality"],
                style=self.current_config_options["style"],
                n=1,  # Value of 1 is the only value supported in DALLE-3 as of now
                response_format="b64_json",
            )

            # Process the response
            image_binary = base64.b64decode(response.data[0].b64_json)
            image_object = BytesIO(image_binary)
            revised_prompt = response.data[0].revised_prompt
            
            # Convert from webP format to PNG
            with Image.open(image_object) as webp_image:
                png_image = BytesIO()
                webp_image.save(png_image, "PNG")
                png_image.seek(0)
                return png_image, revised_prompt

        except Exception as e:
            logger.error(f"Error generating DALL-E image: {e}", exc_info=True)
            return None, e

    def get_gpt_response(self, messages_history, model, temperature=None, max_completion_tokens=None, reasoning_effort=None, verbosity=None):
        """
        Get a response from GPT.
        
        Args:
            messages_history (list): The conversation history.
            model (str): The GPT model to use.
            temperature (float, optional): The temperature for response generation. 
                                          Defaults to the current config value.
            max_completion_tokens (int, optional): The maximum number of tokens for completion.
                                                 Defaults to the current config value.
            reasoning_effort (str, optional): GPT-5 only - controls reasoning depth (minimal/low/medium/high).
            verbosity (str, optional): GPT-5 only - controls response length (low/medium/high).
            
        Returns:
            object: The GPT response or an error.
        """
        # Use default values from config if not specified
        if temperature is None:
            temperature = self.current_config_options["temperature"]
        if max_completion_tokens is None:
            max_completion_tokens = self.current_config_options["max_completion_tokens"]
            
        try:
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": messages_history,
                "stream": self.streaming_client,
            }
            
            # Determine if this is a GPT-5 reasoning model
            # Reasoning models: gpt-5, gpt-5-mini, gpt-5-nano (with dates)
            # Non-reasoning: gpt-5-chat-latest, gpt-4 models, etc.
            model_lower = model.lower()
            is_gpt5_reasoning = (
                model_lower.startswith("gpt-5") and 
                not "chat" in model_lower and
                any(x in model_lower for x in ["gpt-5-", "gpt-5-mini", "gpt-5-nano"])
            )
            
            # Handle temperature based on model type
            if is_gpt5_reasoning:
                # GPT-5 reasoning models only support temperature=1
                if temperature != 1:
                    logger.debug(f"GPT-5 reasoning model detected, forcing temperature to 1 (was {temperature})")
                api_params["temperature"] = 1.0
                # GPT-5 reasoning models don't support top_p variations either
            else:
                # GPT-4, GPT-5-chat, and earlier support temperature and top_p
                api_params["temperature"] = float(temperature)
                api_params["top_p"] = float(self.current_config_options["top_p"])
            
            # Add max_completion_tokens only if specified (None means let model decide)
            if max_completion_tokens is not None:
                api_params["max_completion_tokens"] = int(max_completion_tokens)
            
            # Add GPT-5 reasoning-specific parameters only for reasoning models
            if is_gpt5_reasoning:
                # Use provided values or fall back to config defaults
                if reasoning_effort is not None:
                    api_params["reasoning_effort"] = reasoning_effort
                else:
                    api_params["reasoning_effort"] = self.current_config_options.get("reasoning_effort", "medium")
                logger.debug(f"Using reasoning_effort: {api_params.get('reasoning_effort')}")
                
                if verbosity is not None:
                    api_params["verbosity"] = verbosity
                else:
                    api_params["verbosity"] = self.current_config_options.get("verbosity", "medium")
                logger.debug(f"Using verbosity: {api_params.get('verbosity')}")
            
            # Call the OpenAI API for chat completion
            response = self.client.chat.completions.create(**api_params)
            self.usage = response.usage
            return response.choices[0].message

        except Exception as e:
            logger.error(f"Error getting GPT response: {e}", exc_info=True)
            return e

    def is_processing(self, thread_id):
        """
        Check if a specific thread is currently processing.
        
        This method is deprecated and will be removed in a future version.
        Use QueueManager.is_processing() instead.
        
        Args:
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            bool: True if the thread is processing, False otherwise.
        """
        # This method is kept for backward compatibility
        # It always returns False since processing state is now managed by QueueManager
        return False

    def usage_command(self):
        """
        Get the token usage statistics.
        
        Returns:
            str: A string representation of the token usage statistics.
        """
        logger.info("Viewing token usage")
        
        if self.usage:
            usage_str = f"""
            Cumulative Token stats since last reset:
            Prompt Tokens: {self.usage.prompt_tokens}
            Completion Tokens: {self.usage.completion_tokens}
            Total Tokens: {self.usage.total_tokens}"""
        else:
            usage_str = "No usage info yet. Ask the bot something and check again."
        return usage_str

    def history_command(self, thread_id):
        """
        Get the conversation history for a thread.
        
        Args:
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            str: A string representation of the conversation history.
        """
        logger.info(f"Viewing history for thread {thread_id}")
        
        if not thread_id:
            return "!history can only be run inside of a thread."

        # Deep copy of the messages to avoid modifying the original history
        display_history = deepcopy(self.conversations[thread_id]["messages"])

        # Replace b64 encoded images with placeholder text
        for message in display_history:
            if "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        # Replace image data with placeholder text
                        content_item["image_url"] = {
                            "url": "Image content not displayed"
                        }

        # Convert display_history to a string representation for the Slack message
        history_str = f"[HISTORY] thread_id = {thread_id}\n"
        for message in display_history:
            if "content" in message:
                if isinstance(message["content"], list):
                    content_str = "".join(
                        item["text"] if item["type"] == "text" else "[Image content not displayed]"
                        for item in message["content"]
                    )
                elif isinstance(message["content"], str):
                    content_str = message["content"]
                history_str += f"{message['role'].capitalize()}: {content_str}\n"

        return history_str.strip().replace("`", "")
        
    # To-Do, move all config options into threads. 
    def set_config(self, setting, value, thread_id=None):
        """
        Set a configuration option.
        
        Args:
            setting (str): The configuration option to set.
            value (any): The value to set the configuration option to.
            thread_id (str, optional): The ID of the thread/conversation. Defaults to None.
            
        Returns:
            str: A message indicating the result of the operation.
        """
        logger.info(f"Setting config {setting}={value} for thread {thread_id}")
        
        if thread_id is None:
            return "Adjust configuration options inside threads."
            
        if setting in self.current_config_options:
            # Convert string "true"/"false" to boolean
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            
            # Special handling for system_prompt
            if setting.lower() == "system_prompt":
                if thread_id in self.conversations and "messages" in self.conversations[thread_id] and self.conversations[thread_id]["messages"]:
                    self.conversations[thread_id]["messages"][0]["content"] = value

                    return f"Updated config setting \"{setting}\" to \"{value}\" for this channel/thread."
                else:
                    return f"Thread {thread_id} is not properly initialized."
            else:
                # Update the configuration option
                self.current_config_options[setting] = value
                return f"Updated config setting \"{setting}\" to \"{value}\""
                
        return f"Unknown setting: {setting}"

    def view_config(self, thread_id=None):
        """
        View the current configuration options.
        
        Args:
            thread_id (str, optional): The ID of the thread/conversation. Defaults to None.
            
        Returns:
            str: A string representation of the current configuration options.
        """
        logger.info(f"Viewing config for thread {thread_id}")
        
        if thread_id is not None and thread_id in self.conversations:
            if "messages" in self.conversations[thread_id] and self.conversations[thread_id]["messages"]:
                system_prompt_from_thread = self.conversations[thread_id]["messages"][0]["content"]
                if system_prompt_from_thread != self.current_config_options["system_prompt"]:
                    # Get all items except the last one
                    config_items = list(self.current_config_options.items())
                    config_except_last = config_items[:-1]

                    # Assume the last item's key is 'system_prompt' and get its value from another variable
                    last_item_key = config_items[-1][0]
                    last_item_value = system_prompt_from_thread

                    # Combine the items into a string
                    config = "\n".join(f"{setting}: {value}" for setting, value in config_except_last)
                    config += f"\n{last_item_key}: {last_item_value}"
                    
                    return config

        # Default to showing current configuration options
        return "\n".join(f"{setting}: {value}" for setting, value in self.current_config_options.items())

    def reset_config(self, thread_id):
        """
        Reset the configuration options to defaults.
        
        Args:
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            str: A message indicating the result of the operation.
        """
        logger.info(f"Resetting config for thread {thread_id}")
        
        self.current_config_options = self.config_option_defaults.copy()
        self.conversations[thread_id]["messages"][0]["content"] = self.current_config_options["system_prompt"]

        return "Configuration Defaults Reset!"

    @staticmethod
    def help_command():
        """
        Get help information about available commands.
        
        Returns:
            str: A string containing help information.
        """
        logger.info("Viewing help")
        
        help_str = """
    !help - This help.
    /dalle-3 {prompt} generate an image via text with Dalle-3. (Slack Only)
    !config - Displays the current configuration values. For now, these are global settings for everyone.
    !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
    --see https://platform.openai.com/docs/api-reference for more info.
    !history - Prints a json dump of the chat history since last reset.
    !reset config - Sets the config options back to defaults, e.g., temperature, max_completion_tokens, etc.
    !reset history - Resets conversation history. Command deprecated in Slack. Start a new thread in Slack.
    !usage - Prints token usage stats since the last reset."""
        
        return help_str

    @staticmethod
    def handle_busy():
        """
        Get a message indicating the bot is busy.
        
        Returns:
            str: A message indicating the bot is busy.
        """
        return "I'm busy processing a previous request, please wait a moment and try again."
