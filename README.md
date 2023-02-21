# ChatGPT Bots
Python based ChatGPT bot integrations

## Description
Various ChatBot Integrations like Slack and Discord as well as a CLI based version using Python and OpenAPI's ChatGPT platform.

## Getting Started

Requires `Python 3.6+`

### Install venv module
`python3 -m pip install --user venv`

### Create virtual environment
`python3 -m venv chatgpt-bots`

### Activate venv
```
cd chatgpt-bots
source bin/activate
```

### Installing Dependencies:
```pip3 install -r requirements.txt```

### Setup `.env` file
- Aquire the necessary keys or tokens from the integration you're using. 
I.e., OpenAI, Slack and Discord tokens.
The only required token is the OPENAI_KEY. The others depend on which integration you're using.

- Create a `.env` file in this folder and populate it with your keys or tokens as follows:

```
SLACK_BOT_TOKEN = 'YOURTOKENHERE'
SLACK_APP_TOKEN = 'YOURTOKENHERE'
OPENAI_KEY = 'YOURTOKENHERE'
DISCORD_TOKEN = 'YOURTOKENHERE'
```

### Configuration - Bot Tuning
The `initialize_text` variable at the top of each script will set the tone for how the bot will respond. Telling it that it is a chatbot and with any specific style of responses, along with how to preface it's responses (in the case of the cli, it can name itself in the chat history this way.) will help with more appropriate responses.

### Running the bot
Run the py file for your chosen interface

`python3 discordbot.py`


## ToDo:
- Add Tokenizer to count tokens in combined context + prompt.
- Use Tokenizer counts to manage a rolling history of chat to avoid the 4096 token limit on the current GPT3.5 language model.
- Clean up code, turn OpenAI functions into modules to call from the various integration py files.



