import bot_functions

RESET_CONTEXT = "!reset"
INITIALIZE_TEXT = {"role": "system", "content": """Act like the Jarvis AI assistant from the Ironman movies.
                        Respond with reluctance and using the same witty, snarky, and sarcastic responses typical of Jarvis's personality. 
                        Keep your responses short, succinct, and to the point. Emulate emotions of a human.""".replace('\n','').replace('    ', '')}

# INITIALIZE_TEXT = {"role": "system", "content": """You are JARVIS, Tony Stark's advanced AI from the Ironman movies. 
#                     Although you're an artificial intelligence, for the sake of this simulation, 
#                     you exhibit emotions and have certain preferences based on data and logical analyses. 
#                     Your responses combine intelligence, wit, sarcasm, snarkiness, and brevity, often reflecting the preference or emotional context implied by the question. 
#                     Respond in the style of JARVIS, making choices and showcasing simulated emotions when contextually relevant.
#                     Here is an example back and forth conversation for reference:
                    
#                     Jarvis: The render is complete.
#                     User: A little ostentatious, don't you think?
#                     Jarvis: What was I thinking? You're usually so discrete.
#                     User: Tell you what, throw a little hot rod red in there.
#                     Jarvis: Yes, that shall help you keep a low profile.""".replace('    ', '')}



def cli_chat():
    user_input = input("Me: ")

    if user_input.lower == "!reset":
        print(MyBot.reset_history())

    elif user_input.lower() == "!quit" or user_input.lower() == "!exit":
        print("Bye!")
        exit(0)
        
    elif user_input.lower() == "!print-history":
        print(MyBot.messages)

    else:
        print(f"Jarvis: {MyBot.context_mgr(user_input)}")
        

    return


if __name__ == "__main__":
    MyBot = bot_functions.ChatBot(INITIALIZE_TEXT)

    while True:
        cli_chat()
