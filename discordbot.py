import base64
import os
import re
import discord
import asyncio
from discord.ext import tasks
from dotenv import load_dotenv
from prompts import DISCORD_SYSTEM_PROMPT
import bot_functions as bot
import common_utils as utils

# Load environment variables
load_dotenv()

# Configuration variables
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
show_dalle3_revised_prompt = True

# List of channel IDs the bot is allowed to talk in. Set these in your .env file, comma delimited.
discord_channel_ids = [int(id_str.strip()) for id_str in os.getenv('DISCORD_ALLOWED_CHANNEL_IDS').split(',')]

# Patterns to match commands
config_pattern = r"!config\s+(\S+)\s+(.+)"
reset_pattern = r"^!reset\s+(\S+)$"

# Discord custom emojis need to use unicode IDs which are server specific. 
# If you'd rather use a standard/static one like :hourglass:, go for it.
LOADING_EMOJI = "<a:loading:1245283378954244096>" 

streaming_client = False
chat_del_ts = []  # List of message IDs to cleanup after a response returns

# Pattern to match the Discord bot's userID in channel messages
user_id_pattern = re.compile(r"<@[\w]+>")


class discordClt(discord.Client):
    """
    Discord client for handling interactions with the Discord API.
    
    This class manages the Discord bot's behavior, including message processing,
    command handling, and interaction with the ChatBot instance.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a new Discord client instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.thread_id = ""
        self.queue = asyncio.Queue()
        self.processing = False

    async def setup_hook(self):
        """
        Set up the Discord client hooks.
        
        This method is called when the client is starting up.
        """
        self.process_queue.start()
    
    async def on_ready(self):
        """
        Handle the event when the Discord client is ready.
        
        This method is called when the client has successfully connected to Discord.
        """
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        """
        Handle incoming messages from Discord.
        
        This method processes commands and messages directed at the bot.
        
        Args:
            message (discord.Message): The message received from Discord.
        """
        # Initialize conversation if needed
        if self.thread_id not in gpt_Bot.conversations:
            await self.reset_history(self.thread_id)
                    
        # Check if the message is a reply to one of the bot's messages
        is_reply_to_bot = False
        if message.reference and message.reference.message_id:
            try:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                if original_message.author == self.user:
                    is_reply_to_bot = True
            except discord.NotFound:
                pass

        # We do not want the bot to reply to itself and to only respond when @mentioned, 
        # replied to, and only in the allowed channels.
        if (message.author.bot or 
            (self.user.mention not in message.content and not is_reply_to_bot) or 
            message.channel.id not in discord_channel_ids):
            return
              
        self.thread_id = str(message.channel.id)
        
        # Remove the discord bot's userID from the message using regex pattern matching
        text = re.sub(user_id_pattern, "", message.content).strip()

        # Combine original message and reply if it is a reply
        if is_reply_to_bot:               
            text = f"[Reply to bot's message]\nOriginal Bot Message: {original_message.content}\nUser Reply: {text}"

        # Process commands
        match text.lower():
            case "!history":
                if self.thread_id not in gpt_Bot.conversations:
                    await message.channel.send("`No history yet.`")
                    return
                await self.send_paginated_message(message.channel, f"{gpt_Bot.history_command(self.thread_id)}")
                return

            case "!help":
                await message.channel.send(f"```{gpt_Bot.help_command()}```")
                return

            case "!usage":
                await message.channel.send(f"```{gpt_Bot.usage_command()}```")
                return

            case "!config":
                await message.channel.send(f"```{gpt_Bot.view_config(self.thread_id)}```")
                return
                
            case _:
                # Check for config and reset commands
                config_match_obj = re.match(config_pattern, text.lower())
                reset_match_obj = re.match(reset_pattern, text.lower())
                
                if config_match_obj:
                    setting, value = config_match_obj.groups()
                    response = gpt_Bot.set_config(setting, value, self.thread_id)
                    await message.channel.send(f"```{response}```")
                    return

                elif reset_match_obj:
                    parameter = reset_match_obj.group(1)
                    # Reset history no longer supported in bot_functions.py. Add functionality here for now.
                    if parameter == "history":
                        response = await self.reset_history(self.thread_id)
                        await message.channel.send("`Chat History cleared.`")
                    elif parameter == "config":
                        response = gpt_Bot.reset_config(self.thread_id)
                        await message.channel.send(f"`{response}`")
                    else:
                        await message.channel.send(
                            f"Unknown reset parameter: {parameter}"
                        )

                elif text.startswith("!"):
                    await message.channel.send(
                        "`Invalid command. Type '!help' for a list of valid commands.`"
                    )

                else:
                    # Add message to processing queue
                    await self.queue.put((message, text))
                    
                    # If queue has more than one item, inform user the bot is busy
                    if self.queue.qsize() > 1:
                        busy_response = f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:"
                        temp_message = await message.channel.send(f"{busy_response}")
                        chat_del_ts.append(temp_message.id)
    
    @tasks.loop(seconds=.5)
    async def process_queue(self):
        """
        Process the message queue.
        
        This method runs as a background task, processing messages in the queue
        one at a time.
        """
        if not self.queue.empty() and not self.processing:
            message, text = await self.queue.get()
            self.processing = True

            try:
                # Send initial "thinking" message
                initial_response = await message.channel.send(f"Thinking... {LOADING_EMOJI}")
                chat_del_ts.append(initial_response.id)
                
                # Check if user is requesting DALL-E 3 image generation
                img_check = await self.image_check(text, gpt_Bot, self.thread_id)
                
                # Handle DALL-E 3 image generation request
                if img_check:
                    if message.attachments:
                        await message.channel.send(":warning: `Ignoring included file with Dalle-3 request. Image gen based on provided images is not yet supported with Dalle-3.` :warning:")
                        
                    # Create DALL-E 3 prompt from history
                    dalle3_prompt = await self.create_dalle3_prompt(text, gpt_Bot, self.thread_id)
                    
                    # Cleanup previous messages and show generating message
                    await delete_chat_messages(message.channel, chat_del_ts)
                    initial_response = await message.channel.send(f"Generating image, please wait... {LOADING_EMOJI}")
                    chat_del_ts.append(initial_response.id)
                    
                    # Generate image with DALL-E 3
                    image, revised_prompt, is_error = await self.create_dalle3_image(dalle3_prompt.content, self.thread_id)
                    
                    # Handle response
                    if is_error:
                        await message.channel.send(handle_error(revised_prompt))
                    else:
                        if gpt_Bot.current_config_options["d3_revised_prompt"]:
                            image_description = f"*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_"
                        else:
                            image_description = None
                            
                        discord_image = discord.File(image, filename="dalle3_image.png")
                        await message.channel.send(content=image_description, file=discord_image)
                
                # Handle files in the message (GPT Vision request or other file types)
                elif message.attachments:
                    vision_files = []
                    other_files = []
                    
                    # Process each attachment
                    for attachment in message.attachments:
                        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            img_bytes = await attachment.read()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            vision_files.append(img_b64)
                        else:
                            img_bytes = await attachment.read()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            other_files.append(img_b64)
                    
                    # Handle vision files        
                    if vision_files:
                        response, is_error = await self.vision_request(text, vision_files, self.thread_id)
                            
                        if is_error:
                            await message.channel.send(handle_error(response))
                        else:
                            await self.send_paginated_message(message.channel, response)
                
                    # Handle unsupported file types
                    elif other_files:
                        await message.channel.send(":no_entry: `Sorry, GPT4 Vision only supports jpeg, png, webp, and gif file types at this time.` :no_entry:")
                
                # Handle normal text message
                else:
                    response, is_error = await self.fetch_openai_response(text, self.thread_id)
                    
                    if is_error:
                        await message.channel.send(handle_error(response))
                    else:
                        await self.send_paginated_message(message.channel, response)
                        
            finally:
                self.processing = False
                self.queue.task_done()

                # Clear the queue and discard messages
                while not self.queue.empty():
                    discarded_message, _ = await self.queue.get()
                    self.queue.task_done()
                    # Delete the busy messages for the discarded messages
                    await delete_chat_messages(discarded_message.channel, chat_del_ts)

                # Delete the busy messages for the current message
                await delete_chat_messages(message.channel, chat_del_ts)

    async def vision_request(self, text, vision_files, thread_id):
        """
        Process a vision request.
        
        This method runs the vision context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            vision_files (list): List of base64-encoded image files.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            tuple: (response, is_error) where response is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        loop = asyncio.get_event_loop()
        response, is_error = await loop.run_in_executor(None, gpt_Bot.vision_context_mgr, text, vision_files, thread_id)
        return response, is_error    
    
    async def create_dalle3_image(self, text, thread_id):
        """
        Create a DALL-E 3 image.
        
        This method runs the image context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The prompt for image generation.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            tuple: (image, revised_prompt, is_error) where image is the generated image,
                  revised_prompt is DALL-E's revised prompt, and is_error indicates if an error occurred.
        """
        loop = asyncio.get_event_loop()
        image, revised_prompt, is_error = await loop.run_in_executor(None, gpt_Bot.image_context_mgr, text, thread_id)
        return image, revised_prompt, is_error
    
    async def image_check(self, text, gpt_Bot, thread_id):
        """
        Check if a message is requesting image generation.
        
        This method runs the image check in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            gpt_Bot (ChatBot): The ChatBot instance.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            bool: True if the message is requesting image generation, False otherwise.
        """
        loop = asyncio.get_event_loop()
        is_img_request = await loop.run_in_executor(None, utils.check_for_image_generation, text, gpt_Bot, thread_id)
        return is_img_request

    async def create_dalle3_prompt(self, text, gpt_Bot, thread_id):
        """
        Create a DALL-E 3 prompt.
        
        This method runs the DALL-E 3 prompt creation in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            gpt_Bot (ChatBot): The ChatBot instance.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            object: The GPT response containing the DALL-E 3 prompt.
        """
        loop = asyncio.get_event_loop()
        dalle3_prompt = await loop.run_in_executor(None, utils.create_dalle3_prompt, text, gpt_Bot, thread_id)
        return dalle3_prompt
        
    async def fetch_openai_response(self, text, thread_id):
        """
        Fetch a response from OpenAI.
        
        This method runs the chat context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            tuple: (response, is_error) where response is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        loop = asyncio.get_event_loop()
        response, is_error = await loop.run_in_executor(None, gpt_Bot.chat_context_mgr, text, thread_id)
        return response, is_error

    async def send_paginated_message(self, channel, message):
        """
        Send a message in chunks if it's too long for Discord.
        
        Discord has a 2000 character limit for messages, so this method
        splits the message into chunks and sends them separately.
        
        Args:
            channel (discord.TextChannel): The channel to send the message to.
            message (str): The message to send.
        """
        message_chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
        for chunk in message_chunks:
            await channel.send(chunk)
            
    async def reset_history(self, thread_id):
        """
        Reset the conversation history for a thread.
        
        Args:
            thread_id (str): The ID of the thread/conversation.
            
        Returns:
            str: A message indicating the result of the operation.
        """
        gpt_Bot.conversations[thread_id] = {
            "messages": [DISCORD_SYSTEM_PROMPT],
            "processing": False,
            "history_reloaded": True,
        }
        return "Chat History cleared."


async def delete_chat_messages(channel, ids):
    """
    Delete messages from a Discord channel.
    
    Args:
        channel (discord.TextChannel): The channel to delete messages from.
        ids (list): List of message IDs to delete.
    """
    try:
        for id in ids:
            try:
                message = await channel.fetch_message(id)
                await message.delete()
            except discord.NotFound:
                pass  # Message already deleted
            except discord.Forbidden:
                print("Bot doesn't have permission to delete messages")
            except Exception as e:
                print(f"Error deleting message: {e}")
    finally:
        chat_del_ts.clear()


async def handle_error(error):
    """
    Format an error message for Discord.
    
    Args:
        error (any): The error to format.
        
    Returns:
        str: A formatted error message.
    """
    return f":no_entry: `An error occurred. Error details:` :no_entry:\n```{error}```"


if __name__ == "__main__":
    # Initialize the ChatBot
    gpt_Bot = bot.ChatBot(DISCORD_SYSTEM_PROMPT, streaming_client, show_dalle3_revised_prompt)
    
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    
    # Create and run the Discord client
    client = discordClt(intents=intents)
    client.run(DISCORD_TOKEN)

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
