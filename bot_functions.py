import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from textwrap import dedent


load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_KEY"])


class ChatBot:
    def __init__(self, INITIALIZE_TEXT, streaming_client=False):
        self.messages = [INITIALIZE_TEXT]
        self.INITIALIZE_TEXT = INITIALIZE_TEXT
        self.streaming_client = streaming_client  # ToDo: Implement streaming support
        self.gpt_output = {}
        self.usage = {}
        self.config_option_defaults = {
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 2048,
            "custom_init": "",
        }
        self.current_config_options = self.config_option_defaults.copy()

    def context_mgr(self, message_content, content_type="text"):
        if content_type == "text":
            self.messages.append({"role": "user", "content": message_content})
            self.gpt_output = self.get_ai_response(self.messages)

            if self.gpt_output.role != "error":
                self.messages.append(self.gpt_output.model_copy())

            return self.gpt_output.content

        elif content_type == "image":
            # handle images
            pass

    def get_ai_response(self, messages_history):
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages_history,
                stream=self.streaming_client,
                temperature=float(self.current_config_options["temperature"]),
                max_tokens=int(self.current_config_options["max_tokens"]),
                top_p=float(self.current_config_options["top_p"]),
            )
            self.usage = response.usage
            return response.choices[0].message

        except openai.RateLimitError as r:
            print(f"##################\n{r}\n##################")
            return {
                "role": "error",
                "content": "Rate Limit Error: My servers are too busy or you're spamming me. Try your request again in a moment.",
            }

        except openai.APIError as i:
            print(f"##################\n{i}\n##################")
            return {
                "role": "error",
                "content": "API Error: Sorry, I ran into an error with your request. Please try again.",
            }

    @staticmethod
    def help_command():
        return dedent(
            """\
            !help - This help.
            !config - Displays the current configuration values.
            !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
              --see https://platform.openai.com/docs/api-reference/completions/create for more info.
            !history - Prints a json dump of the chat history since last reset.
            !reset config - Sets the config options back to defaults, e.g., temperature, max_tokens, etc.
            !reset history - Clears the bots memory and resets the context to the default as configured in this script. (Always the first line of the '!history' output.)
            !usage - Prints token usage stats since the last reset."""
        )

    def usage_command(self):
        if self.usage != {}:
            return dedent(
                f"""\
                Cumulative Token stats since last reset:
                Prompt Tokens: {self.usage.prompt_tokens}
                Completion Tokens: {self.usage.completion_tokens}
                Total Tokens: {self.usage.total_tokens}"""
            )

        else:
            return "No usage info yet. Ask the bot something and check again."

    def history_command(self):
        return self.messages

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

    def reset_history(self):
        self.messages = [self.INITIALIZE_TEXT]
        self.usage = {}

        return "Rebooting. Beep Beep Boop. My memory has been wiped!"

    def reset_config(self):
        self.current_config_options = self.config_option_defaults

        return "Configuration Defaults Reset!"
