from openai import OpenAI
import os
from dotenv import load_dotenv
from textwrap import dedent
from io import BytesIO
from PIL import Image


load_dotenv()
GPT4 = 'gpt-4-1106-preview'
DALLE3 = 'dall-e-3'
GPT4_VISION = 'gpt-4-vision-preview'


class ChatBot:
    def __init__(self, INITIALIZE_TEXT, streaming_client=False):
        self.messages = [INITIALIZE_TEXT]
        self.INITIALIZE_TEXT = INITIALIZE_TEXT
        self.streaming_client = streaming_client  # ToDo: Implement streaming support
        self.gpt_output = {}
        self.usage = {}
        self.processing = False
        self.config_option_defaults = {
            'temperature': 0.5,
            'top_p': 1,
            'max_tokens': 2048,
            'custom_init': '',
            'model': GPT4,
            'size': '1024x1024',
            'quality': 'hd',  # standard or hd
            'style': 'vivid',  # vivid or natural
            'number': 1
        }
        self.current_config_options = self.config_option_defaults.copy()
        self.client = OpenAI(api_key=os.environ['OPENAI_KEY'])

    def handle_content_type(self, message_content, content_type='text'):
        is_error = False

        if self.processing:
            return self.handle_busy()

        else:
            if content_type == 'text':
                return self.chat_context_mgr(message_content)

            elif content_type == 'image':
                return self.image_context_mgr(message_content)

            elif content_type == 'vision':
                return self.vision_context_mgr(message_content)

    def chat_context_mgr(self, message_content):
        self.processing = True
        self.current_config_options['model'] = GPT4
        self.messages.append({'role': 'user', 'content': message_content})
        self.gpt_output = self.get_chatgpt_response(self.messages)
        self.processing = False

        if hasattr(self.gpt_output, 'role'):
            is_error = False
            if self.gpt_output.role == 'assistant':
                self.messages.append(
                    {'role': 'assistant', 'content': self.gpt_output.content})

            return self.gpt_output.content, is_error
        else:
            is_error = True
            self.messages.pop()
            return self.gpt_output, is_error

    def image_context_mgr(self, message_content):
        self.processing = True
        self.current_config_options['model'] = DALLE3
        is_error = False
        self.gpt_output = self.get_dalle_response(message_content)
        self.processing = False
        return self.gpt_output, is_error

    def vision_context_mgr(self, message_content):
        self.processing = True
        self.current_config_options['model'] = GPT4_VISION
        is_error = False
        # process vision
        self.processing = False
        return "Vision Response", is_error

    def get_dalle_response(self, image_prompt):
        try:
            response = self.client.images.generate(
                model=self.current_config_options['model'],
                prompt=image_prompt,
                size=self.current_config_options['size'],
                quality=self.current_config_options['quality'],
                style=self.current_config_options['style'],
                n=1,
                response_format='url'
            )
            return response
        except Exception as e:
            print(f'##################\n{e}\n##################')
            return e

    def get_vision_response(self, messages_history):
        pass

    def get_chatgpt_response(self, messages_history):
        try:
            response = self.client.chat.completions.create(
                model=self.current_config_options['model'],
                messages=messages_history,
                stream=self.streaming_client,
                temperature=float(self.current_config_options['temperature']),
                max_tokens=int(self.current_config_options['max_tokens']),
                top_p=float(self.current_config_options['top_p']),

            )
            self.usage = response.usage
            return response.choices[0].message

        except Exception as e:
            print(f'##################\n{e}\n##################')
            return e

    def usage_command(self):
        if self.usage != {}:
            return dedent(
                f'''\
                Cumulative Token stats since last reset:
                Prompt Tokens: {self.usage.prompt_tokens}
                Completion Tokens: {self.usage.completion_tokens}
                Total Tokens: {self.usage.total_tokens}'''
            )

        else:
            return 'No usage info yet. Ask the bot something and check again.'

    def history_command(self):
        return self.messages

    def set_config(self, setting, value):
        if setting in self.current_config_options:
            self.current_config_options[setting] = value
            return f'Updated {setting} to {value}'
        return f'Unknown setting: {setting}'

    def view_config(self):
        return '\n'.join(
            f'{setting}: {value}'
            for setting, value in self.current_config_options.items()
        )

    def reset_history(self):
        self.messages = [self.INITIALIZE_TEXT]
        self.usage = {}
        self.processing = False

        return 'Rebooting. Beep Beep Boop. My memory has been wiped!'

    def reset_config(self):
        self.current_config_options = self.config_option_defaults

        return 'Configuration Defaults Reset!'

    @staticmethod
    def help_command():
        return dedent(
            """\
            !help - This help.
            !config - Displays the current configuration values.
            !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
              --see https://platform.openai.com/docs/api-reference for more info.
            !history - Prints a json dump of the chat history since last reset.
            !reset config - Sets the config options back to defaults, e.g., temperature, max_tokens, etc.
            !reset history - Clears the bots memory and resets the context to the default as configured in this script. (Always the first line of the '!history' output.)
            !usage - Prints token usage stats since the last reset."""
        )

    @staticmethod
    def handle_busy():
        return "I'm busy processing a previous request, please wait a moment and try again."
