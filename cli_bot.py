import openai
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.environ["OPENAI_KEY"]
RESET_CONTEXT = "!reset"
initialize_text = 'Marv is a chatbot that reluctantly answers questions with witty and sarcastic responses. Preface your responses with "Marv: "'
initialized = 0
history = []


# def open_file(filepath):
#     with open(filepath, "r", encoding="utf-8") as infile:
#         return infile.read()


def reset_history():
    global history
    global initialize_text

    history = [initialize_text]
    print("Marv: Rebooting. Beep Beep Boop. My memory has been wiped!")
    return


def chat():
    global history
    user_input = input("Me: ")

    if user_input == "!reset":
        reset_history()

    elif user_input.lower() == "quit" or user_input.lower() == "exit":
        print("Bye!")
        exit(0)

    else:
        user_input = "Me: " + user_input
        print(context_mgr(user_input).strip())

    return


def context_mgr(ai_prompt):
    global history
    global initialized
    global initialize_text

    if initialized == 0:
        history.append(initialize_text)
        initialized = 1

    chat_input = "context: " + "\n".join(history) + "\n" + ai_prompt
    output = get_ai_response(chat_input)
    history += [ai_prompt, output.strip()]
    return output


def get_ai_response(ai_prompt):
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
        reset_history()
        return "Marv: Sorry, I ran out of token memory. Rebooting. Beep Beep Boop."

    except openai.error.RateLimitError as r:
        return "Marv: My servers are too busy! Try your request again."


if __name__ == "__main__":
    openai.api_key = OPENAI_KEY

    while True:
        chat()
