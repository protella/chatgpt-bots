import discord
import openai
import os
from dotenv import load_dotenv
import bot_functions

load_dotenv()

OPENAI_KEY = os.environ["OPENAI_KEY"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
TRIGGER = ""
RESET_CONTEXT = "!reset"
initialize_text = "You are the Jarvis virtual assistant from the Ironman movies. Act and respond as such including the witty, snarky, and sarcastic responses typical of the Jarvis personality."
initialized = 0
history = []


# def open_file(filepath):
#     with open(filepath, "r", encoding="utf-8") as infile:
#         return infile.read()


openai.api_key = OPENAI_KEY


def reset_history():
    global history
    global initialize_text

    history = [initialize_text]
    print("Marv: Rebooting. Beep Beep Boop. My memory has been wiped!")
    return


class MyClient(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        global history
        global initialize_text

        if message.author.id == self.user.id or message.content.startswith(
            message.author.mention
        ):
            return

        if message.content is None:
            await message.channel.send(
                f"```Something went wrong. Fix your shit, OpenAI.```"
            )

        if message.content.startswith(RESET_CONTEXT):
            history = []
            history.append(initialize_text)
            await message.channel.send(
                f"```My Memory has been wiped. I'm dumb again.```"
            )
            return
        print(f"{message.content}\n")
        if message.content.startswith(TRIGGER):
            await message.channel.send(f"{context_mgr(message.content)}")


def context_mgr(ai_prompt):
    global history
    global initialized
    global initialize_text

    if initialized == 0:
        history.append(initialize_text)
        initialized = 1

    chat_input = "Context: " + "\n".join(history) + "\n" + ai_prompt
    debug_input = "Context: " + "\n".join(history) + "\nMe: " + ai_prompt
    output = get_ai_response(chat_input)
    history += [ai_prompt, output.strip()]
    # print(f"CHAT INPUT: {debug_input}\n")
    # print(f'HISTORY: {history}\n')
    return output


def get_ai_response(ai_prompt):
    global history
    global initialize_text
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
        # print(f'\n\nPROMPT TOKENS: {len(tokenizer(ai_prompt))}')
        # print(f'\n\nCOMPLETION TOKENS: {len(tokenizer(response.choices[0].text))}\n')
        return response.choices[0].text
    except openai.error.InvalidRequestError as e:
        reset_history()
        return "Sorry, I ran out of token memory. Rebooting. Beep Beep Boop."
    except openai.error.RateLimitError as r:
        return "My servers are too busy! Try your request again."


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    client = MyClient(intents=intents)
    client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
