import base64
import os
import logging
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from logger import setup_logger, get_log_level, get_logger
from common_utils import get_model_capabilities

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
                {"role": "user", "content": [{"type": "input_text", "text": message_text}]}
            )
            
            # Debug: Log what we're about to send to the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Messages being sent to API (count: {len(self.conversations[thread_id]['messages'])})")
                # Log first and last few messages to avoid huge logs
                msgs = self.conversations[thread_id]["messages"]
                if len(msgs) <= 5:
                    for i, msg in enumerate(msgs):
                        logger.debug(f"  Message {i}: role={msg.get('role', 'system')}, content_type={type(msg.get('content'))}")
                else:
                    # Log first 2 and last 3 messages
                    for i in range(2):
                        msg = msgs[i]
                        logger.debug(f"  Message {i}: role={msg.get('role', 'system')}, content_type={type(msg.get('content'))}")
                    logger.debug(f"  ... {len(msgs) - 5} messages omitted ...")
                    for i in range(len(msgs) - 3, len(msgs)):
                        msg = msgs[i]
                        logger.debug(f"  Message {i}: role={msg.get('role', 'system')}, content_type={type(msg.get('content'))}")

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
                            "content": [{"type": "output_text", "text": gpt_output.content}],
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
                            "type": "input_text",
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
                
                # Convert the image to base64 for storage in conversation history
                # The image is returned as a BytesIO object from get_dalle_response
                if image:
                    image.seek(0)  # Reset to beginning of BytesIO
                    image_bytes = image.read()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    image.seek(0)  # Reset again so it can still be uploaded to Slack
                    
                    # Add both the text description and the image to conversation history
                    # Note: We use "user" role for the image due to API limitations
                    # but the text makes it clear this is an assistant-generated image
                    self.conversations[thread_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"""Based on the Dalle-3 user prompt, I created an image like this: {revised_prompt}
                                    I can describe what this images looks like and the ideas it might convey.
                                    I will act as if I created the this image in the context of this chat."""
                                }
                            ],
                        }
                    )
                    
                    # Add the actual image as a separate "user" message for API compatibility
                    # The system prompt tells the model these are its own creations
                    self.conversations[thread_id]["messages"].append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{image_b64}"
                                }
                            ],
                        }
                    )
                else:
                    # If no image (shouldn't happen), just add the text
                    self.conversations[thread_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
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
                "content": [{"type": "input_text", "text": f"{message_text}"}],
            }

            # Add each image to the message
            for image in images:
                new_image_element = {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image}"
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
                            "content": [{"type": "output_text", "text": gpt_output.content}],
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
            # Extract system prompt and user messages from history
            system_prompt = None
            user_messages = []
            
            for msg in messages_history:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                else:
                    # Messages should already be in Responses API format
                    user_messages.append(msg)
            
            # Build API call parameters for Responses API
            api_params = {
                "model": model,
                "input": user_messages,  # Responses API uses 'input' not 'messages'
                "store": False,  # Don't store since we're managing history locally
            }
            
            # Debug: Log the number and types of messages being sent
            logger.debug(f"Sending to Responses API: {len(user_messages)} messages (excluding system prompt)")
            
            # Add system prompt as instructions if present
            if system_prompt:
                api_params["instructions"] = system_prompt
            
            # Determine model capabilities
            capabilities = get_model_capabilities(model)
            
            # Handle temperature based on model capabilities
            if capabilities["is_reasoning"]:
                # GPT-5 reasoning models only support temperature=1
                if temperature != capabilities["fixed_temperature"]:
                    logger.debug(f"GPT-5 reasoning model detected, forcing temperature to {capabilities['fixed_temperature']} (was {temperature})")
                api_params["temperature"] = capabilities["fixed_temperature"]
                # GPT-5 reasoning models don't support top_p variations either
            else:
                # GPT-4, GPT-5-chat, and earlier support temperature and top_p
                api_params["temperature"] = float(temperature)
                api_params["top_p"] = float(self.current_config_options["top_p"])
            
            # Add max_output_tokens only if specified (None means let model decide)
            if max_completion_tokens is not None:
                api_params["max_output_tokens"] = int(max_completion_tokens)
            
            # Add GPT-5 reasoning-specific parameters only for reasoning models
            if capabilities["supports_reasoning_effort"]:
                # Use provided values or fall back to config defaults
                effort_value = reasoning_effort
                if effort_value is None:
                    effort_value = self.current_config_options.get("reasoning_effort", "medium")
                
                # Reasoning effort goes under 'reasoning' parameter
                api_params["reasoning"] = {"effort": effort_value}
                logger.debug(f"Using reasoning effort: {effort_value}")
            
            if capabilities["supports_verbosity"]:
                verbosity_value = verbosity
                if verbosity_value is None:
                    verbosity_value = self.current_config_options.get("verbosity", "medium")
                
                # Verbosity goes under 'text' parameter for GPT-5 models
                api_params["text"] = {"verbosity": verbosity_value}
                logger.debug(f"Using verbosity: {verbosity_value}")
            
            # Call the OpenAI Responses API
            response = self.client.responses.create(**api_params)
            
            # Convert response to match expected format
            # Responses API returns response.output_text
            class MessageWrapper:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            
            return MessageWrapper(response.output_text)

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
    !reset config - Sets the config options back to defaults, e.g., temperature, max_completion_tokens, etc.
    !reset history - Resets conversation history. Command deprecated in Slack. Start a new thread in Slack."""
        
        return help_str

    @staticmethod
    def handle_busy():
        """
        Get a message indicating the bot is busy.
        
        Returns:
            str: A message indicating the bot is busy.
        """
        return "I'm busy processing a previous request, please wait a moment and try again."
