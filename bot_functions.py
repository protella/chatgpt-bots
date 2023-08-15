import openai
import os
from dotenv import load_dotenv


class ChatBot:
    def __init__(self, INITIALIZE_TEXT, filepath=""):
        self.messages = [INITIALIZE_TEXT]
        self.INITIALIZE_TEXT = INITIALIZE_TEXT
        self.initialized = 0
        self.filepath = filepath
        self.gpt_output = {}
        self.usage = {}
        self.config_option_defaults = {
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 512,
            "custom_init": "",
        }
        self.current_config_options = self.config_option_defaults.copy()

    load_dotenv()

    openai.api_key = os.environ["OPENAI_KEY"]

    # def open_file(self):
    #     with open(self.filepath, "r", encoding="utf-8") as infile:
    #         return infile.read()

    def context_mgr(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        self.gpt_output = self.get_ai_response(self.messages)

        if self.gpt_output["role"] != "error":
            self.messages.append(self.gpt_output.copy())

        return self.gpt_output["content"]

    def get_ai_response(self, messages_history):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages_history,
                temperature=float(self.current_config_options["temperature"]),
                max_tokens=int(self.current_config_options["max_tokens"]),
                top_p=float(self.current_config_options["top_p"]),
            )
            self.usage = response.usage
            return response.choices[0].message

        except openai.error.InvalidRequestError as i:
            print(f"##################\n{i}\n##################")
            return {
                "role": "error",
                "content": "Sorry, I ran into an error with your request. Please try again.",
            }

        except openai.error.RateLimitError as r:
            print(f"##################\n{r}\n##################")
            return {
                "role": "error",
                "content": "My servers are too busy or you're spamming me. Try your request again in a moment.",
            }

    @staticmethod
    def help_command():
        return f"""!help - This help.
            !config - Displays the current configuration values.
            !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
              --see https://platform.openai.com/docs/api-reference/completions/create for more info.
            !history - Prints a json dump of the chat history since last reset.
            !reset config - Sets the config options back to defaults, e.g., temperature, max_tokens, etc.
            !reset history - Clears the bots memory and resets the context to the default as configured in this script. (Always the first line of the '!history' output.)
            !usage - Prints token usage stats since the last reset.
                              
            """.replace(
            "    ", ""
        )

    def usage_command(self):
        if self.usage != {}:
            return f"""Cumulative Token stats since last reset:
                Prompt Tokens: {self.usage.prompt_tokens}
                Completion Tokens: {self.usage.completion_tokens}
                Total Tokens: {self.usage.total_tokens}
                """.replace(
                "    ", ""
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
