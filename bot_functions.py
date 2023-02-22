import openai
import os
from dotenv import load_dotenv


class ChatBot:
    def __init__(self, INITIALIZE_TEXT, filepath=""):
        self.history = []
        self.INITIALIZE_TEXT = INITIALIZE_TEXT
        self.initialized = 0
        self.filepath = filepath

    load_dotenv()
    openai.api_key = os.environ["OPENAI_KEY"]

    def open_file(self):
        with open(self.filepath, "r", encoding="utf-8") as infile:
            return infile.read()

    def reset_history(self):
        self.history = [self.INITIALIZE_TEXT]

        return "Rebooting. Beep Beep Boop. My memory has been wiped!"

    def context_mgr(self, ai_prompt):
        if self.initialized == 0:
            self.history.append(self.INITIALIZE_TEXT)
            self.initialized = 1

        chat_input = "context: " + "\n".join(self.history) + "\n" + ai_prompt
        output = self.get_ai_response(chat_input)
        self.history += [ai_prompt, output.strip()]
        return output

    def get_ai_response(self, ai_prompt):
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=ai_prompt,
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                # stop=['']
            )
            return response.choices[0].text

        except openai.error.InvalidRequestError as e:
            error_output = f"Sorry, I ran out of token memory.\n{self.reset_history()}"
            return error_output

        except openai.error.RateLimitError as r:
            return "My servers are too busy! Try your request again."
