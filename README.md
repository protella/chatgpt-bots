# ChatGPT Bots
Python based ChatGPT bot integrations

## `Notice - Discord client updated with some Slack client features and general roadmap plans...`
I finally got around to updating the Discord client to have a few of the basic features of the Slack client. It should support Vision, Dalle3 via NLP, and the ability to reply to individual messages. Discord threads/conversation support may or may not come in the future. Due to the Asyncio architecture of the Discord Python framework, things work a bit differently and I had to come up with some creative ways to deal with the event loop and blocking issues from the OpenAI requests. I'm holding off on major new features for the Slack client until the big new GPT-4o features come online. 

## Description
ChatBot Integrations for Slack, Discord, and the CLI using Python and OpenAPI's ChatGPT platform. This bot is designed around GPT4 and supports GPT4 Vision and Dalle-3. The bots allow iteration on Dalle-3 images and will also determine if image creation is the appropriate action by using NLP. Talk to it just like you would with the ChatGPT website. Upload (multiple) images and have discussions or conduct analysis on them all in a single conversation (Slack Thread or Discord Channel). The Discord client is still a bit behind in development. The CLI client is for basic testing only.

***Dev Note:*** I've had very good success with using NLP to determine whether an image request is intended without explicitly asking for one. I was hesitant to merge this branch into master as it added an entirely new api request for the image gen check as well as a third for generating a behind-the-scenes image prompt that includes the previous conversation history as part of its context (this is how you can iterate on an image). These additional api requests added a bit more latency than felt acceptable, but this has been remedied by OpenAI's launch of the new `GPT-4o` model which is twice as fast (and 50% cheaper!).

## Getting Started

Requires `Python 3.10+` as the script takes advantage of the new structural pattern matching (match/case) in this version.  

The setup of a Slack or Discord App is out of scope of this README. There's plenty of documentation online detailing these processes.
  
### Slack quickstart guide: https://api.slack.com/start/quickstart
#### The Slack event subscriptions and scope are as follows:

| Event Name  	| Description                                                       	| Required Scope    	|
|-------------	|-------------------------------------------------------------------	|-------------------	|
| app_mention 	| Subscribe to only the message events that mention your app or bot 	| app_mentions:read 	|
| message.im  	| A message was posted in a direct message channel                  	| im:history        	|    
    
#### Slack OAuth & Permissions (Scopes):
| Scope                	|
|----------------------	|
| app_mentions:read    	|
| channels:history     	|
| channels:join        	|
| chat:write           	|
| chat:write.customize 	|
| commands             	|
| files:read           	|
| files:write          	|
| im:history           	|
| im:read              	|
| im:write             	|

---

### Discord OAuth2 
<img src="Docs/Discord_OAuth2.png" alt="image" width="40%" height="auto">

---

### Install `venv` module if you don't already have it
`python3 -m pip install --user venv`

### Clone the repository
`git clone https://github.com/protella/chatgpt-bots`

### Create and Activate the Virtual Environment
```
cd chatgpt-bots
python3 -m venv chatbots
source chatbots/bin/activate
```

### Installing Dependencies:
```python3 -m pip install -U -r requirements.txt```

- _Note: The included `requirements.txt` file includes all of the dependencies for all 3 clients in this repo._

### Setup `.env` file
- Aquire the necessary keys or tokens from the integration you're using. 
I.e., OpenAI, Slack and Discord tokens.
The only required token is the OPENAI_KEY. The others depend on which integration you're using.

- Create a `.env` file in the root folder of the repo and populate it with your keys and tokens as follows:

```
SLACK_BOT_TOKEN = 'YOURTOKENHERE'
SLACK_APP_TOKEN = 'YOURTOKENHERE'
OPENAI_KEY = 'YOURTOKENHERE'
DISCORD_TOKEN = 'YOURTOKENHERE'
DISCORD_ALLOWED_CHANNEL_IDS = '1234567890, 1234567890' # Discord channel IDs that the bot is permitted to talk in.
```

### Configuration - Bot Tuning
The `SYSTEM_PROMPT` variable at the top of each script will set the tone for how the bot will respond. Telling it that it is a chatbot and with any specific style of responses will help with more appropriate responses.

### Running the bot
Run the py file for your chosen interface, e.g.:

- `python3 discordbot.py`
- `python3 slackbot.py`
- `python3 cli_bot.py`


## ToDo:
- Implement some basic text extraction for PDFs and other file types for analysis of non-image types.
- Fix bug w/ thread history rebuilds and Image gen check. Need to compare pre-post restart histories.
- Discord is still uses a shared history. Not sure how to handle threads/conversations w/ Discord. 
- Add command functionality to allow for changing the initial chatbot init phrase
- Update bot commands to use Slack/Discord's `/command` functionality rather than old school `!commands`
- Track context/history size using the usage stats and pop old items from the history to avoid going over the model's max context size (4k w/ 3.5-turbo but not as much of an issue with GPT4 Turbo) Adjust for different models if necessary. Lower Priority
- Add support for the bot to recognize individual users within a mult-user conversation.
- Fix usage stats function. Decide how/what to track. Global stats or conversation stats, or both?
- Clean up code, standardize style, move repeated client code to functions and utility modules.
- Create Slack app manifest file
- Setup Github workflows and unit tests 



