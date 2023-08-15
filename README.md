# ChatGPT Bots
Python based ChatGPT bot integrations

## Description
Various ChatBot Integrations like Slack and Discord as well as a CLI based version using Python and OpenAPI's ChatGPT platform.

## Getting Started

Requires `Python 3.10+` as the script takes advantage of the new structural pattern matching (match/case/switch) in this version.

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
The `INITIALIZE_TEXT` variable at the top of each script will set the tone for how the bot will respond. Telling it that it is a chatbot and with any specific style of responses will help with more appropriate responses.

### Running the bot
Run the py file for your chosen interface, e.g.:

- `python3 discordbot.py`
- `python3 slackbot.py`
- `python3 cli_bot.py`


## ToDo:
- Add command functionality to allow for changing the initial chatbot init phrase
- Clean up code, standardize style, move repeated client code to functions module



