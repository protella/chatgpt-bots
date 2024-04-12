import base64
import os
from copy import deepcopy
from io import BytesIO
from textwrap import dedent

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


load_dotenv()  # load auth tokens from .env file

# Default models: https://platform.openai.com/docs/models
GPT_MODEL = "gpt-4-turbo"
DALLE_MODEL = "dall-e-3"


class ChatBot:
    def __init__(self, SYSTEM_PROMPT, streaming_client=False):
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.conversations = {}
        self.streaming_client = streaming_client  # ToDo: Implement streaming support
        self.usage = {}
        self.config_option_defaults = {
            "temperature": 0.5,  # 0.0 - 2.0
            "top_p": 1,
            "max_tokens": 2048,  # max 4096
            "custom_init": "",
            "gpt_model": GPT_MODEL,
            "dalle_model": DALLE_MODEL,
            "size": "1024x1024",  # 1024x1024, 1024x1792 or 1792x1024
            "quality": "hd",  # standard or hd
            "style": "vivid",  # natural or vivid
            "number": 1,  # number of images. Only 1 supported for Dalle3
            "detail": "auto",  # vision parameter: auto, low, high
        }
        self.current_config_options = self.config_option_defaults.copy()
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def chat_context_mgr(self, message_text, thread_id, files=""):
        self.conversations[thread_id]["processing"] = True

        if not self.conversations[thread_id]["history_reloaded"]:
            # Review the need for the pop(): Delete trailing message to avoid dupe message.
            self.conversations[thread_id]["messages"].pop()
            self.conversations[thread_id]["history_reloaded"] = False

        self.conversations[thread_id]["messages"].append(
                {"role": "user", "content": [{"type": "text", "text": message_text}]}
            )
        
        # Using vision model for all chat prompts since images passed to non-vision model throws an error.
        # GPT4v is an extension of GPT4 with all the same functions and features.

        gpt_output = self.get_gpt_response(
            self.conversations[thread_id]["messages"],
            self.current_config_options["gpt_model"],
        )
        self.conversations[thread_id]["processing"] = False

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
            is_error = True
            self.conversations[thread_id]["messages"].pop()

            return gpt_output, is_error

    def image_context_mgr(self, message_text, thread_id):
        self.conversations[thread_id]["processing"] = True

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

        image, revised_prompt = self.get_dalle_response(
            message_text, self.current_config_options["dalle_model"]
        )

        self.conversations[thread_id]["processing"] = False

        if revised_prompt:
            is_error = False
            self.conversations[thread_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": dedent(
                                f"""\
                        Based on the Dalle-3 user prompt, I can imagine I created an image like this: {revised_prompt}
                        I can describe what such an image might look like or the ideas it might convey.
                        I will act as if I created the described image for the purposes of this chat."""
                            ),
                        }
                    ],
                }
            )

        else:
            is_error = True
            self.conversations[thread_id]["messages"].pop()
            image = None
            revised_prompt = None

        self.conversations[thread_id]["processing"] = False

        return image, revised_prompt, is_error

    def vision_context_mgr(self, message_text, images, thread_id):
        # self.rebuild_thread_history(thread_id)

        self.conversations[thread_id]["processing"] = True

        if not message_text:
            message_text = ""
        multi_part_msg = {
            "role": "user",
            "content": [{"type": "text", "text": f"{message_text}"}],
        }

        for image in images:
            new_image_element = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                    "detail": self.current_config_options["detail"],
                },
            }
            multi_part_msg["content"].append(new_image_element)

        self.conversations[thread_id]["messages"].append(multi_part_msg)

        gpt_output = self.get_gpt_response(
            self.conversations[thread_id]["messages"],
            self.current_config_options["gpt_model"],
        )

        self.conversations[thread_id]["processing"] = False

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
            is_error = True
            self.conversations[thread_id]["messages"].pop()
            return gpt_output, is_error

    def get_dalle_response(self, image_prompt, model):
        try:
            response = self.client.images.generate(
                model=model,
                prompt=image_prompt,
                size=self.current_config_options["size"],
                quality=self.current_config_options["quality"],
                style=self.current_config_options["style"],
                n=1,  # Value of 1 is the only value supported in DALLE-3 as of now
                response_format="b64_json",
            )

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
            print(f"##################\n{e}\n##################")
            return None, e

    def get_gpt_response(self, messages_history, model):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages_history,
                stream=self.streaming_client,
                temperature=float(self.current_config_options["temperature"]),
                max_tokens=int(self.current_config_options["max_tokens"]),
                top_p=float(self.current_config_options["top_p"]),
            )
            self.usage = response.usage
            return response.choices[0].message

        except Exception as e:
            print(f"##################\n{e}\n##################")
            return e

    def is_processing(self, thread_id):
        # Check if a specific thread is currently processing.
        return self.conversations.get(thread_id, {}).get("processing", False)

    def usage_command(self):  # Fix this later to aggregate all thread usage
        if self.usage:
            return dedent(
                f"""\
                Cumulative Token stats since last reset:
                Prompt Tokens: {self.usage.prompt_tokens}
                Completion Tokens: {self.usage.completion_tokens}
                Total Tokens: {self.usage.total_tokens}"""
            )

        else:
            return "No usage info yet. Ask the bot something and check again."

    def history_command(self, thread_id):
        if not thread_id:
            return "!history can only be run inside of a thread."
        # We don't want to display the b64 encoded images that may be present in the history, so replace them with placeholder text.
        # This is done in a separate instance of the message history so that the images remain for the bot to analyze in future conversations.
        display_history = deepcopy(self.conversations[thread_id]["messages"])
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

        return display_history

    def set_config(self, setting, value):
        if setting in self.current_config_options:
            self.current_config_options[setting] = value
            return f"Updated {setting} to {value}"
        return f"Unknown setting: {setting}"

    def view_config(self):
        return "\n".join(
            f"{setting}: {value}"
            for setting, value in self.current_config_options.items()
        )

    def reset_config(self):
        self.current_config_options = self.config_option_defaults

        return "Configuration Defaults Reset!"

    @staticmethod
    def help_command():
        return dedent(
            """\
            !help - This help.
            /dalle-3 {prompt} generate an image via text with Dalle-3.
            !config - Displays the current configuration values. For now, these are global settings for everyone.
            !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
              --see https://platform.openai.com/docs/api-reference for more info.
            !history - Prints a json dump of the chat history since last reset.
            !reset config - Sets the config options back to defaults, e.g., temperature, max_tokens, etc.
            !reset history - Command deprecated. Start a new thread with the bot for a fresh conversation.
            !usage - Prints token usage stats since the last reset."""
        )

    @staticmethod
    def handle_busy():
        return "I'm busy processing a previous request, please wait a moment and try again."
