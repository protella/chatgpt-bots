We are going to refactor and rearchitect this project into a V2 version. 

If you review the code, you'll see the V1 project here has 3 separate clients supported. CLI, Slack, and Discord. We're going to focus only on the Slack client for now. I have gone ahead and prepared the new branch in git and the workspace is clean with an existing .venv, requirements.txt (which we will probably modify), and a .env file that is populated with the appropriate keys and tokens needed to integrate with the various service platforms. We have both a Dev and Prod set of keys and tokens to work with. We'll start with Dev only. You have permission to use the live third party services for testing, make sure they use the Dev tokens.

A little background on this project. This was my first real Python project that got deployed to actual use by others. I was a beginner learning Python and had a lot of help from you over the past two years, along with my own Googling. It is not well designed and is getting to the point where it's a bit difficult to manage. As you develop the new PRD, consider how to better architect the code, use OOP techniques, build for scale and modularity. The core code files are the slackbot.py (Slack client specific code), bot_functions.py (Shared core functions, like openai calls, etc.), common_utils.py (things like image and prompt mgmt.). There are a couple other files, like logger.py, queue_manager.py that were "afterthoughts" and could probably be implemented better. You should consider how to refactor all of this in the PRD. Consider if new code files should be made, how existing files, functions and features are broken up, and come up with more efficient methods for achieving the same outcome. 

Key differences that we'll be building into this version:
- We'll continue to use the Slackbolt SDK for integrating with Slack.
- OpenAI moves fast with development of their services. They have since come out with their new Responses API which will replace the Chat Completions API (Which the current version uses). I want to move to the responses API for V2. There are differences in the API shape between these two APIs that need to be considered. Note to use Context7 or web searches within Cursor to understand the new API.
- One big change is that in the Chat Completions API, it was up to the developer to maintain the chat history, sending the full chat history as context with each new request. The Responses API takes advantage of OpenAI side conversation logging. Each request has a "previous_response_id" which we can use to track the conversation history by pointing to the last message in that conversation.
- Slack will continue to be the source of truth for all existing conversations. E.g., if the bot is restarted, it has no conversations its aware of until a user either starts a new conversation (thread) in either a DM with the bot, or a chat channel it's been invited to, or replies to an existing thread (which is what we have been using as conversation IDs to this point). With the responses api, now, once an existing thread is revived by a user, the bot will need to rebuild the conversation from the slack API and and then that entire conversation will get sent to OpenAI as a single request. From there forward, we can use the previous_response_ID to maintain the conversation context and not rely on maintaining the conversation history locally in memory. 
- Setup a SQLite docker container (up to you to decide if that product is appropriate) to maintain per-user and per-thread/conversation configuration options. E.g., if the user is requesting image generations, the number of generations should be stored. The Image model to use could also be stored. We'll also need to keep track of threads (thread_ts) and previous_response_id associations. This is the way to link a Slack thread to an openai saved conversation log.
- The viewing and setting of configuration options was parsed per request previously, but I wanted to implement a better way of doing it. We will leverage NLP. If the user asks for 5 images, that would be translated into a config option and saved if it's detected as part of their prompt. We'll also support a slash command "/" to set them directly as well. The new slash command is: SLACK_SLASH_COMMAND="/chatgpt-config-dev" in the .env file. Make it parameterized.
- Image generation on the current version used Dalle-3. OpenAI just released their new "gpt-image-1" model for the API. It has been added along side the dalle3 as a .env variable. Note in your PRD to lookup the api reference for this new model. We should provide the option to use either one as a config option. 
- The new image gen model takes much longer (up to 2min) per request than dalle-3 (like 20sec). So we'd need to figure out a way to manage the UX around this as the conversations with LLMs are still sequential, so we couldn't allow the conversation to continue until after the image comes back. Our existing queue and locking mechanisms should work, but I'm open to suggestions on how to better handle the UX of waiting for a minute before being able to do anything in a thread.
- The new image model also supports generating a new image based of a provided image. The V1 already supports image uploads, so we can leverage that, but the ability to provide txt _and_ image as input for image gen didn't exist in dalle3. I want to allow this for the new model.
- I want the V2 bots to be able to keep track of which user is sending a message in the case of a multi-user conversation (2+ users and the bot). Then it knows who said what, and when appropriate may refer to them by user's First Name (i'd rather not use @mentions as those are annoying to get over and over).
- When we remove the user's SlackID from the message, we can replace it with a tag like [user=Name]. The system prompt will tell the bot what to do with these so it can opt to use the First name in the response. We should filter these tags out if they happen to come back in a response.
- We do not need to support the manual /dalle-3 slack slash command anymore. Remove this.
- Image requests should be determined via a separate LLM call with a smaller and faster model. They should return a low temperature deterministic response that we can convert to boolean T/F for whether the request should be sent to the image model or standard text model. The new UTILITY_MODEL = "gpt-4o-mini" var has been added to the .env for this purpose. All images will be kept as part of the conversations on the OpenAI side so they can be referenced as part of the conversation history by the model. 
- The responses API has many new features in it, like calling functions or tools. We can leverage some of those functions, such as conducting a websearch, or reading a file.
- Slack supports Block Kits which allow additional visual features and interactivity. I didn't explore these for V1, but perhaps they will be useful in V2. Consider this and what we could do with them or how we'd use them in our use case here.
- The V1 was run inside a venv and managed by PM2 on the host system. I would like the V2 version to be setup inside a docker container. It can run with the latest python (at least 3.12) that supports the latest Slackbolt and OpenAI SDKs. We can use Docker Compose, which is already installed. You will probably want to have Cursor setup the docker files and docker compose first. If you decide a small DB is required, make sure to mention that in the docs.

IMPORTANT: Be sure to include a test task for each feature as its completed. 

Remember, you're not generating code here, you're building out a PRD in raw text MD format for Cursor to consume. Write it in a way that Cursor can use and break down into features, and structure the flow from basic to complex features. While we're only focusing on the Slack client now, do architect it in a way that Discord and other clients can interface with the shared code modules. Keep the Client code in the client related files. Do not add any discord or cli features now.

For now, unless stated here, ignore the ToDo list in the README. We'll tackle those later.


The environment setup requirements are as follows:
- Development tools: Cursor IDE with agent mode.
- Tasks file (we will work on this later) for the AI to follow and update.
- Python 3.12 with a venv already activated and ready to go.
- Initial basic folders of Docs/ and tests/ (Empty)
- There is a pre-populated .env file with all necessary variables and auth tokens for the 3rd party services. This contains both Dev and Prod keys. Use only Dev keys for now.
- A prompts.py file already exists that has various system prompts ready to go when the time comes to implement the features that rely on them. 
- I will manage all git functions. A new branch is setup and ready to go.
- The app will run inside a docker container. You will need to include related instructions to prepare the dockerfiles and docker compose setup.
- Make sure to mention the need to properly build out a requirements.txt as necessary for all dependencies.

Coding practices:
- Write professional well documented code using modern techniques and OOP. 
- When writing tests, make sure that the tests are run inside of the docker container, not in the host environment. 
- Don't consider a task complete until the associated tests pass.
- You are permitted to write tests that interface with the live 3rd party services as needed. You don't need to mock api calls. Just use the Dev access tokens.
- There is an MCP server installed called "context7" which can pull up-to-date documentation and code reference examples. Include "use context7" with every prompt.

Requirements:
- Python 3.12
- OpenAI Python SDK v1.76.0
- We'll use the Slackbolt SDK for integrating with Slack. Ensure to check online for the latest version or package.
- Postgres 17.4 (in its own container)
- OpenAI moves fast with development of their services. They have come out with their new Responses API which will replace the Chat Completions API. I only want to use the responses API. Do not code any api calls with the Chat completions API. 
- The Responses API takes advantage of OpenAI side conversation logging. Each request has a "previous_response_id" which we can use to track the conversation history. Do not manage conversation history locally.
- Slack will continue to be the source of truth for all existing conversations (threads). E.g., if the bot is restarted, it has no conversations its aware of until a user either starts a new conversation (thread) in either a DM with the bot, or a chat channel it's been invited to, or replies to an existing thread. With the responses api, now, once an existing thread is revived by a user, the bot will need to rebuild the conversation from the slack API and and then that entire conversation will get sent to OpenAI as a single request with the system prompt prepended. From there forward, we can use the previous_response_id to maintain the conversation state. 
-  We will use Postgresql to maintain per-user and per-thread/conversation configuration options. E.g., if the user is requesting image generations, the number of generations should be stored for future requests. The chosen Image model to use could also be stored. (e.g., dall-e-3 (old) or gpt-image-1 (new/default)).
- Use NLP to view and manage configuration options. e.g., If the user asks for 5 images, that would be translated into a config option and saved if it's detected as part of their prompt. 
- The default image generation model is "gpt-image-1". Look up the api reference and shape to know how this differs from Dalle3. Their config options are different. We will provide the option to use either one. The new image gen model takes much longer (up to 2min) per request than dalle-3 (~20sec). So we'd need to figure out a way to manage the UX around this as the conversations with LLMs are still sequential, so we couldn't allow the conversation to continue until after the image comes back. 
- The new image model also supports generating a new image based of a user provided image. 
- All image models require images to be encoded/decoded using Base64. We are not using URLs to reference images in OpenAI calls.
- Slack stores images which we can reference using the "url_private" event element and passing a bearer token (slackbot token in .env). 
- In Slack chat channels (not DMs), the bot needs to keep track of which user is sending a message in the case of a multi-user conversation (2+ users and the bot). Then it knows who said what, and when appropriate may refer to them by user's First Name. 
- Inject a flag like [user=Name] into the beginning of the message so the bot is aware of who sent the message. The system prompt will instruct the the LLM how to use this. We should make sure to filter out any [user=name] tags that might get sent back in the LLM response.
- Remove any Slack userids from the inbound messages. (e.g., in regex, "<@[\w]+>"
- Image requests should be determined via a separate LLM call with a smaller and faster "utility" model (like gpt-4o-mini), return a low temperature, deterministic response that we can convert to boolean T/F for whether the request should be sent to the image model or standard text model. All images will be kept as part of the conversations on the OpenAI side so they can be referenced as part of the conversation by the model. 
- The responses API has many new features in it, like calling functions or tools. We can leverage some of those functions, such as conducting a websearch, or reading a file.
- Slack supports Block Kits which allow additional visual features and interactivity. Consider this and what we could do with them or how we'd use them in our use case here.
- We will use Docker Compose, which is already installed. You will need to instruct Cursor to setup the docker files and docker compose first. 
- All requests should provide visual user feedback by displaying a temporary "Thinking..." message followed by an emoji (defined as the THINKING_EMOJI in the .env). As soon as the response is displayed, this message should be deleted. You will need to keep track of all temporary status message ids for later cleanup.
- If the user's request is determined to be a request for an image generation, the "Thinking" message should be deleted and a new "Generating image, please wait..." message should be displayed, also with the thinking emoji.
- A logger should be implemented. Log levels are defined in the .env file. Implement logging messages as you write out the code. Make sure to include appropriate messages at every critical point. 
- All major functions should be wrapped in try/except blocks for errors. Friendly errors should be returned to the user. The logs and console can capture the raw error.
- There are 4 supported image mime types (Slack passes these with file downloads). OpenAI image requests or downloads should include them. They are:{"image/jpeg", "image/png", "image/gif", "image/webp"}
- A queue manager should be implemented to only allow one action per conversation/thread at a time. The user should receive a friendly "Busy or related "I'm still processing your previous request..." type message (and also cleaned up later). If they try to send another message prior to the previous message being displayed. Chat conversations are sequential in nature. Make sure there's no race conditions that allow any type of follow up request to be sent until the previous request is complete.
- Keep track of the bot's userid for easy reference later in chat operations.
- Ensure the bot doesn't respond to its own messages or event types that we're not interested in. We should only be responding to "message" and "app_mention" events for now.
- Implement a Slack slash (/) command for directly calling commands, like "help", token usage", and managing config. E.g., "/chatgpt-config-dev". This will be defined in the .env as "SLACK_SLASH_COMMAND" to make it configurable by the admin.
- The database should have a table to track conversation IDs Slack "thread_ts" and OpenAI "previous_response_id" These should be associated with each other and thread_ts should be the primary key.
- History rebuild for existing thread vs new conversation. Here's some example logic on how to implement this. Do not use a conversations dictionary, use the database.
    is_thread = "thread_ts" in event
    thread_ts = event["thread_ts"] if is_thread else event["ts"]

        # Handle new or existing threads since last restart
        if thread_ts not in gpt_Bot.conversations:
            logger.info(f"Initializing new conversation for thread {thread_ts}")
            if is_thread:
                # Rebuild history for existing thread
                rebuild_thread_history(say, channel_id, thread_ts, bot_user_id)
            else:
                # Initialize new conversation
                gpt_Bot.conversations[thread_ts] = {
                    "messages": [gpt_Bot.SYSTEM_PROMPT],
                    "history_reloaded": False,
                }      
The message event flow should resemble something like this:
Incoming Slack message > Check DB for existing thread_ts > If new thread, continue, else rebuild from slack history and continue. > Message cleanup (remove IDs, etc.) > Check if latest message contains text, files or both > Check if the message is requesting an image generation > If image gen request and/or files, call image gen (w/ or w/o files if NOT Dalle3 as model) > If not image request, and files exist, this is a GPT Vision request. Include images in normal response > Else just normal chat message request.
- When response from OpenAI returns, handle message with or without files (upload files to slack and then reference and send to Slack in the proper thread_ts. 
Example Slack Upload api call.
                    response = app.client.files_upload_v2(
                        channel=channel,
                        initial_comment=file_description,
                        file=image,
                        filename="Dalle3_image.png",
                        thread_ts=thread_ts,
                    )

- Make sure to validate inputs. Empty chat prompts with Images are valid. Both empty is not. 

- Make sure the bot can handle indepenent requests from multiple users or chat channels simultaeously and that the queue manager tracks them each separately.

Example configuration options that should be saved to the DB per user.
        # Default configuration options
        self.config_option_defaults = {
            "temperature": .8,  # 0.0 - 2.0
            "top_p": 1,
            "max_completion_tokens": 2048,  # max 4096
            "custom_init": "",
            "gpt_model": GPT_MODEL,
            "gpt_image_model": GPT_IMAGE_MODEL,
            "dalle_model": DALLE_MODEL,
            "size": "1024x1024",  # Dalle3 parameter: 1024x1024, 1024x1792, or 1792x1024
            "quality": "hd",  # Dalle3 parameter: standard or hd
            "style": "natural",  # Dalle3 parameter: natural or vivid
            "number": 1,  # number of images. Only 1 supported for Dalle3
            "detail": "auto",  # vision parameter: auto, low, high
            "d3_revised_prompt": self.show_dalle3_revised_prompt,
            "system_prompt": self.SYSTEM_PROMPT["content"] # content of system prompt
        }

Example B64 decode for images and files. Keep files and images in memory, no saving to disk. BytesIO might be optional. Revised prompt is for when Dalle3 is the selected image model.
            image_binary = base64.b64decode(response.data[0].b64_json)
            image_object = BytesIO(image_binary)
            revised_prompt = response.data[0].revised_prompt

- The GPT4.1 model is multimodal and can take text and images in API calls. 

IMPORTANT - Be sure to include a test for each feature as its implemented. 

Remember, you're not generating code here, you're building out a PRD in RAW TEXT MARKDOWN format for Cursor to consume. Write it in a way that Cursor can use and break down into features, and structure the flow from basic to complex features. Start with environment setup. While we're only focusing on the Slack client now, do architect it in a way that Discord and other clients can interface with the shared code modules. Keep the Client code in the client related files. 

If you have additional questions or need clarification on anything, please ask.