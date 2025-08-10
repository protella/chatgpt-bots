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
from queue_manager import QueueManager
from logger import log_session_marker, setup_logger, get_log_level, get_logger

# Unset any existing log level environment variables to ensure .env values are used
if "DISCORD_LOG_LEVEL" in os.environ:
    del os.environ["DISCORD_LOG_LEVEL"]

# Load environment variables
load_dotenv()

# Configuration variables
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
show_dalle3_revised_prompt = True

# Configure logging level from environment variable with fallback to INFO
LOG_LEVEL_NAME = os.environ.get("DISCORD_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = get_log_level(LOG_LEVEL_NAME)

# Initialize logger
logger = get_logger('discord_bot', LOG_LEVEL)

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
        self.queue = asyncio.Queue()
        # Initialize the queue manager
        self.queue_manager = QueueManager()

    async def setup_hook(self):
        """Set up Discord client hooks."""
        self.process_queue.start()
    
    async def on_ready(self):
        """Handle the event when the Discord client is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("Discord bot is ready")
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("---------------------------------------------")

    async def on_message(self, message):
        """Handle incoming messages from Discord."""
        # Get channel ID as thread_id
        channel_id = str(message.channel.id)
        
        # Initialize conversation if needed
        if channel_id not in gpt_Bot.conversations:
            logger.info(f"Initializing new conversation for channel {channel_id}")
            await self.reset_history(channel_id)
                    
        # Check if the message is a reply to one of the bot's messages
        is_reply_to_bot = False
        if message.reference and message.reference.message_id:
            try:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                if original_message.author == self.user:
                    is_reply_to_bot = True
                    logger.debug(f"Message is a reply to bot's message in channel {channel_id}")
            except discord.NotFound:
                logger.warning(f"Referenced message not found in channel {channel_id}")
                pass

        # We do not want the bot to reply to itself and to only respond when @mentioned, 
        # replied to, and only in the allowed channels.
        if (message.author.bot or 
            (self.user.mention not in message.content and not is_reply_to_bot) or 
            message.channel.id not in discord_channel_ids):
            return
              
        # Remove the discord bot's userID from the message using regex pattern matching
        text = re.sub(user_id_pattern, "", message.content).strip()
        logger.debug(f"Processed message text: {text}")

        # Combine original message and reply if it is a reply
        if is_reply_to_bot:               
            text = f"[Reply to bot's message]\nOriginal Bot Message: {original_message.content}\nUser Reply: {text}"
            logger.debug(f"Combined reply text: {text}")

        # Process commands
        match text.lower():
            case "!history":
                logger.info(f"History command received in channel {channel_id}")
                if channel_id not in gpt_Bot.conversations:
                    await message.channel.send("`No history yet.`")
                    return
                await self.send_paginated_message(message.channel, f"{gpt_Bot.history_command(channel_id)}")
                return

            case "!help":
                logger.info(f"Help command received in channel {channel_id}")
                await message.channel.send(f"```{gpt_Bot.help_command()}```")
                return

            case "!usage":
                logger.info(f"Usage command received in channel {channel_id}")
                await message.channel.send(f"```{gpt_Bot.usage_command()}```")
                return

            case "!config":
                logger.info(f"Config command received in channel {channel_id}")
                await message.channel.send(f"```{gpt_Bot.view_config(channel_id)}```")
                return
                
            case _:
                # Check for config and reset commands
                config_match_obj = re.match(config_pattern, text.lower())
                reset_match_obj = re.match(reset_pattern, text.lower())
                
                if config_match_obj:
                    setting, value = config_match_obj.groups()
                    logger.info(f"Config change: {setting}={value} in channel {channel_id}")
                    response = gpt_Bot.set_config(setting, value, channel_id)
                    await message.channel.send(f"```{response}```")
                    return

                elif reset_match_obj:
                    parameter = reset_match_obj.group(1)
                    # Reset history no longer supported in bot_functions.py. Add functionality here for now.
                    if parameter == "history":
                        logger.info(f"History reset in channel {channel_id}")
                        response = await self.reset_history(channel_id)
                        await message.channel.send("`Chat History cleared.`")
                    elif parameter == "config":
                        logger.info(f"Config reset in channel {channel_id}")
                        response = gpt_Bot.reset_config(channel_id)
                        await message.channel.send(f"`{response}`")
                    else:
                        logger.warning(f"Unknown reset parameter: {parameter} in channel {channel_id}")
                        await message.channel.send(
                            f"Unknown reset parameter: {parameter}"
                        )

                elif text.startswith("!"):
                    logger.warning(f"Invalid command received: {text} in channel {channel_id}")
                    await message.channel.send(
                        "`Invalid command. Type '!help' for a list of valid commands.`"
                    )

                else:
                    # Check if this channel is already processing
                    if await self.queue_manager.is_processing(channel_id):
                        logger.info(f"Channel {channel_id} is already processing a request")
                        busy_response = f":no_entry: `{gpt_Bot.handle_busy()}` :no_entry:"
                        temp_message = await message.channel.send(f"{busy_response}")
                        chat_del_ts.append(temp_message.id)
                        return
                        
                    # Add message to processing queue
                    logger.info(f"Adding message to queue for channel {channel_id}")
                    await self.queue.put((message, text, channel_id))
    
    @tasks.loop(seconds=.5)
    async def process_queue(self):
        """Process the message queue."""
        if not self.queue.empty():
            message, text, channel_id = await self.queue.get()
            logger.info(f"Processing message from queue for channel {channel_id}")
            
            # Try to start processing this channel
            if not await self.queue_manager.start_processing(channel_id):
                # Another concurrent call got here first, put the message back in the queue
                logger.debug(f"Channel {channel_id} is already being processed, returning to queue")
                await self.queue.put((message, text, channel_id))
                self.queue.task_done()
                return

            try:
                # Send initial "thinking" message
                logger.debug(f"Sending thinking message for channel {channel_id}")
                initial_response = await message.channel.send(f"Thinking... {LOADING_EMOJI}")
                chat_del_ts.append(initial_response.id)
                
                # Check if user is requesting DALL-E 3 image generation
                logger.debug(f"Checking if message is an image generation request")
                img_check = await self.image_check(text, gpt_Bot, channel_id)
                
                # Handle DALL-E 3 image generation request
                if img_check:
                    logger.info(f"Processing DALL-E 3 image generation request in channel {channel_id}")
                    if message.attachments:
                        logger.warning(f"Ignoring attachments with DALL-E 3 request in channel {channel_id}")
                        await message.channel.send(":warning: `Ignoring included file with Dalle-3 request. Image gen based on provided images is not yet supported with Dalle-3.` :warning:")
                        
                    # Create DALL-E 3 prompt from history
                    logger.debug(f"Creating DALL-E 3 prompt for channel {channel_id}")
                    dalle3_prompt = await self.create_dalle3_prompt(text, gpt_Bot, channel_id)
                    
                    # Cleanup previous messages and show generating message
                    await delete_chat_messages(message.channel, chat_del_ts)
                    initial_response = await message.channel.send(f"Generating image, please wait... {LOADING_EMOJI}")
                    chat_del_ts.append(initial_response.id)
                    
                    # Generate image with DALL-E 3
                    logger.info(f"Generating DALL-E 3 image for channel {channel_id}")
                    image, revised_prompt, is_error = await self.create_dalle3_image(dalle3_prompt.content, channel_id)
                    
                    # Handle response
                    if is_error:
                        logger.error(f"Error generating DALL-E 3 image: {revised_prompt}")
                        await message.channel.send(handle_error(revised_prompt))
                    else:
                        logger.info(f"Successfully generated DALL-E 3 image for channel {channel_id}")
                        if gpt_Bot.current_config_options["d3_revised_prompt"]:
                            image_description = f"*DALL·E-3 generated revised Prompt:*\n_{revised_prompt}_"
                        else:
                            image_description = None
                            
                        discord_image = discord.File(image, filename="dalle3_image.png")
                        await message.channel.send(content=image_description, file=discord_image)
                
                # Handle files in the message (GPT Vision request or other file types)
                elif message.attachments:
                    logger.info(f"Processing message with attachments in channel {channel_id}")
                    vision_files = []
                    other_files = []
                    
                    # Process each attachment
                    for attachment in message.attachments:
                        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            logger.debug(f"Processing image attachment: {attachment.filename}")
                            img_bytes = await attachment.read()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            vision_files.append(img_b64)
                        else:
                            logger.debug(f"Processing non-image attachment: {attachment.filename}")
                            img_bytes = await attachment.read()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            other_files.append(img_b64)
                    
                    # Handle vision files        
                    if vision_files:
                        logger.info(f"Processing GPT Vision request with {len(vision_files)} images in channel {channel_id}")
                        response, is_error = await self.vision_request(text, vision_files, channel_id)
                            
                        if is_error:
                            logger.error(f"Error processing GPT Vision request: {response}")
                            await message.channel.send(handle_error(response))
                        else:
                            logger.info(f"Successfully processed GPT Vision request in channel {channel_id}")
                            await self.send_paginated_message(message.channel, response)
                
                    # Handle unsupported file types
                    elif other_files:
                        logger.warning(f"Unsupported file types received in channel {channel_id}")
                        await message.channel.send(":no_entry: `Sorry, GPT4 Vision only supports jpeg, png, webp, and gif file types at this time.` :no_entry:")
                
                # Handle normal text message
                else:
                    logger.info(f"Processing normal text message in channel {channel_id}")
                    response, is_error = await self.fetch_openai_response(text, channel_id)
                    
                    if is_error:
                        logger.error(f"Error processing text message: {response}")
                        await message.channel.send(handle_error(response))
                    else:
                        logger.info(f"Successfully processed text message in channel {channel_id}")
                        await self.send_paginated_message(message.channel, response)
                        
            except Exception as e:
                logger.error(f"Unexpected error in process_queue: {e}", exc_info=True)
                await message.channel.send(handle_error(str(e)))
            finally:
                # Always cleanup, even if there was an error
                logger.info(f"Finishing processing for channel {channel_id}")
                await self.queue_manager.finish_processing(channel_id)
                self.queue.task_done()

                # Delete the busy messages for the current message
                await delete_chat_messages(message.channel, chat_del_ts)

    async def vision_request(self, text, vision_files, channel_id):
        """
        Process a vision request.
        
        This method runs the vision context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            vision_files (list): List of base64-encoded image files.
            channel_id (str): The ID of the channel.
            
        Returns:
            tuple: (response, is_error) where response is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        logger.info(f"Processing vision request in channel {channel_id}")
        loop = asyncio.get_event_loop()
        response, is_error = await loop.run_in_executor(None, gpt_Bot.vision_context_mgr, text, vision_files, channel_id)
        if is_error:
            logger.error(f"Error in vision context manager: {response}")
        else:
            logger.info(f"Vision request processed successfully in channel {channel_id}")
        return response, is_error    
    
    async def create_dalle3_image(self, text, channel_id):
        """
        Create a DALL-E 3 image.
        
        This method runs the image context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The prompt for image generation.
            channel_id (str): The ID of the channel.
            
        Returns:
            tuple: (image, revised_prompt, is_error) where image is the generated image,
                  revised_prompt is DALL-E's revised prompt, and is_error indicates if an error occurred.
        """
        logger.info(f"Creating DALL-E 3 image in channel {channel_id}")
        loop = asyncio.get_event_loop()
        image, revised_prompt, is_error = await loop.run_in_executor(None, gpt_Bot.image_context_mgr, text, channel_id)
        if is_error:
            logger.error(f"Error in image context manager: {revised_prompt}")
        else:
            logger.info(f"DALL-E 3 image created successfully in channel {channel_id}")
        return image, revised_prompt, is_error
    
    async def image_check(self, text, gpt_Bot, channel_id):
        """
        Check if a message is requesting image generation.
        
        This method runs the image check in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            gpt_Bot (ChatBot): The ChatBot instance.
            channel_id (str): The ID of the channel.
            
        Returns:
            bool: True if the message is requesting image generation, False otherwise.
        """
        logger.debug(f"Checking if message is an image generation request in channel {channel_id}")
        loop = asyncio.get_event_loop()
        is_img_request = await loop.run_in_executor(None, utils.check_for_image_generation, text, gpt_Bot, channel_id)
        if is_img_request:
            logger.info(f"Message identified as image generation request in channel {channel_id}")
        return is_img_request

    async def create_dalle3_prompt(self, text, gpt_Bot, channel_id):
        """
        Create a DALL-E 3 prompt.
        
        This method runs the DALL-E 3 prompt creation in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            gpt_Bot (ChatBot): The ChatBot instance.
            channel_id (str): The ID of the channel.
            
        Returns:
            object: The GPT response containing the DALL-E 3 prompt.
        """
        logger.info(f"Creating DALL-E 3 prompt in channel {channel_id}")
        loop = asyncio.get_event_loop()
        dalle3_prompt = await loop.run_in_executor(None, utils.create_dalle3_prompt, text, gpt_Bot, channel_id)
        logger.debug(f"DALL-E 3 prompt created: {dalle3_prompt.content[:100]}...")
        return dalle3_prompt
        
    async def fetch_openai_response(self, text, channel_id):
        """
        Fetch a response from OpenAI.
        
        This method runs the chat context manager in a separate thread to avoid
        blocking the event loop.
        
        Args:
            text (str): The message text.
            channel_id (str): The ID of the channel.
            
        Returns:
            tuple: (response, is_error) where response is the GPT response
                  and is_error is a boolean indicating if an error occurred.
        """
        logger.info(f"Fetching OpenAI response for channel {channel_id}")
        loop = asyncio.get_event_loop()
        response, is_error = await loop.run_in_executor(None, gpt_Bot.chat_context_mgr, text, channel_id)
        if is_error:
            logger.error(f"Error in chat context manager: {response}")
        else:
            logger.info(f"OpenAI response fetched successfully for channel {channel_id}")
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
        logger.debug(f"Sending paginated message with {len(message_chunks)} chunks")
        for chunk in message_chunks:
            await channel.send(chunk)
            
    async def reset_history(self, channel_id):
        """
        Reset the conversation history for a channel.
        
        Args:
            channel_id (str): The ID of the channel.
            
        Returns:
            str: A message indicating the result of the operation.
        """
        logger.info(f"Resetting conversation history for channel {channel_id}")
        gpt_Bot.conversations[channel_id] = {
            "messages": [DISCORD_SYSTEM_PROMPT],
            "history_reloaded": True,
        }
        return "Chat History cleared."


async def delete_chat_messages(channel_obj, message_ids):
    """
    Delete messages from a Discord channel.
    
    Args:
        channel_obj (discord.TextChannel): The Discord channel object to delete messages from.
        message_ids (list): List of message IDs to delete.
    """
    if not message_ids:
        return
        
    logger.debug(f"Deleting {len(message_ids)} messages from channel {channel_obj.id}")
    try:
        for message_id in message_ids:
            try:
                message = await channel_obj.fetch_message(message_id)
                await message.delete()
            except discord.NotFound:
                logger.debug(f"Message {message_id} already deleted")
                pass  # Message already deleted
            except discord.Forbidden:
                logger.warning(f"Bot doesn't have permission to delete messages in channel {channel_obj.id}")
            except Exception as e:
                logger.error(f"Error deleting message {message_id}: {e}")
    finally:
        chat_del_ts.clear()


def handle_error(error):
    """
    Format an error message for Discord.
    
    Args:
        error (any): The error to format.
        
    Returns:
        str: A formatted error message.
    """
    logger.error(f"Handling error: {error}")
    return f":no_entry: `An error occurred. Error details:` :no_entry:\n```{error}```"


if __name__ == "__main__":
    # Log session start marker
    log_session_marker(logger, "START")
    
    # Log the configured log level after the session marker
    logger.info(f"Discord bot logger initialized with log level: {LOG_LEVEL_NAME}")
    
    logger.info("Starting Discord bot")
    # Initialize the ChatBot
    gpt_Bot = bot.ChatBot(DISCORD_SYSTEM_PROMPT, streaming_client, show_dalle3_revised_prompt)
    
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    
    # Create the Discord client
    logger.info("Initializing Discord client")
    client = discordClt(intents=intents)
    
    try:
        logger.info("Running Discord client")
        client.run(DISCORD_TOKEN, log_handler=None)  # Disable Discord.py's built-in logging
    except Exception as e:
        logger.critical(f"Failed to start Discord bot: {e}", exc_info=True)
        raise
    finally:
        # Log session end marker (Disregard duplicate END marker log messages, buggy shutdown signal code)
        log_session_marker(logger, "END")

# Bot Invite / Auth URL: https://discord.com/api/oauth2/authorize?client_id=1067321050171457607&permissions=534723950656&scope=bot
