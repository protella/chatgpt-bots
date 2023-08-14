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

    load_dotenv()
    
    openai.api_key = os.environ["OPENAI_KEY"]

    # def open_file(self):
    #     with open(self.filepath, "r", encoding="utf-8") as infile:
    #         return infile.read()

    def reset_history(self):
        self.messages = [self.INITIALIZE_TEXT]

        return "Rebooting. Beep Beep Boop. My memory has been wiped!"

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
                temperature=.5,
                max_tokens=512
            )
            return response.choices[0].message

        except openai.error.InvalidRequestError as i:
            print(f"##################\n{i}\n##################")
            return {"role": "error", "content": "Sorry, I ran into an error with your request. Please try again."}
            

        except openai.error.RateLimitError as r:
            print(f"##################\n{r}\n##################")
            return {"role": "error", "content": "My servers are too busy or you're spamming me. Try your request again in a moment."}
            
