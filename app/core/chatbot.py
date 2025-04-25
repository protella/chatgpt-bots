import os
from typing import Dict, List, Optional, Any
import openai
import sys
import re

# Add the root directory to sys.path to allow importing prompts
sys.path.insert(0, '/app')

# Import system prompt
from prompts import SLACK_SYSTEM_PROMPT

# Import logging
from app.core.logging import setup_logger
from app.core.history import remove_personalization_tags

logger = setup_logger(__name__)

class ChatBot:
    """
    ChatBot class for interfacing with OpenAI's Responses API.
    Handles both text and vision (multimodal) requests using GPT-4.1.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ChatBot with an OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Dictionary to track conversations: thread_id -> conversation data
        # Each conversation contains:
        # - messages: List of all messages in the conversation
        # - response_id: The OpenAI response ID from the last API call
        self.conversations: Dict[str, Dict[str, Any]] = {}
        
        # Cache to store token usage metrics
        self.token_usage: Dict[str, Dict[str, int]] = {}
        
        # Default model to use - get from environment or use fallback
        self.model = os.environ.get("GPT_MODEL", "gpt-4.1-2025-04-14")
        logger.info(f"Using model: {self.model} for chat")
    
    def get_response(self, 
                     input_text: str, 
                     thread_id: str, 
                     images: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a response from the OpenAI Responses API.
        
        Args:
            input_text: The user's message
            thread_id: The Slack thread ID to associate with this conversation
            images: Optional list of base64-encoded images to include
            config: Optional configuration overrides
        
        Returns:
            Dict containing response content and metadata
        """
        try:
            config = config or {}
            
            # Check if this is a new thread or continuing conversation
            is_new_thread = thread_id not in self.conversations
            
            # Prepare the messages list for the API call
            messages = []
            
            # Get system prompt (from config or default)
            system_content = config.get("system_prompt", SLACK_SYSTEM_PROMPT)
            system_message = {
                "role": "system",
                "content": system_content
            }
            
            # Initialize new conversation if needed
            if is_new_thread:
                logger.info(f"Starting new conversation for thread {thread_id}")
                # For new threads, we just need the system prompt to start
                messages.append(system_message)
                self.conversations[thread_id] = {
                    "messages": [system_message],
                    "response_id": None
                }
            else:
                # For existing threads, include all previous messages
                messages = self.conversations[thread_id]["messages"].copy()
                logger.info(f"Continuing conversation for thread {thread_id} with {len(messages)} existing messages")
            
            # Add user message with the text content
            user_message: Dict[str, Any] = {
                "role": "user",
                "content": []
            }
            
            # Add text content
            user_message["content"].append({
                "type": "text",
                "text": input_text
            })
            
            # Add images if provided
            if images:
                # Get vision detail level from config (default to "auto")
                detail = config.get("detail", "auto")
                    
                for image_base64 in images:
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": detail
                        }
                    })
            
            # Add the user message to the list for the API call
            messages.append(user_message)
            
            # Add the user message to our stored conversation
            self.conversations[thread_id]["messages"].append(user_message)
            
            # If this is a request to repeat conversation history, clean personalization tags
            # from all messages to ensure they don't appear in the response
            if "repeat" in input_text.lower() and "conversation" in input_text.lower():
                logger.info("Detected conversation repeat request, cleaning personalization tags")
                # Clean up all messages before sending to API
                for msg in messages:
                    if msg["role"] == "user" and isinstance(msg["content"], list):
                        for content_item in msg["content"]:
                            if content_item["type"] == "text":
                                content_item["text"] = remove_personalization_tags(content_item["text"])
            
            logger.info(f"Sending request to OpenAI for thread {thread_id} with {len(messages)} messages")
            
            # Get configuration options from provided config or use defaults
            max_tokens = config.get("max_output_tokens", 4096)
            temperature = config.get("temperature", 0.8)
            top_p = config.get("top_p", 1.0)
            model = config.get("gpt_model", self.model)
                
            # Prepare API parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(**params)
            
            # Store the response ID for future continuation
            self.conversations[thread_id]["response_id"] = response.id
            
            # Add the assistant's response to our stored conversation
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content
            }
            self.conversations[thread_id]["messages"].append(assistant_message)
            
            # Extract and store token usage
            self.token_usage[thread_id] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Return the response
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in ChatBot.get_response: {str(e)}")
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }
    
    def get_token_usage(self, thread_id: str) -> Optional[Dict[str, int]]:
        """
        Get token usage statistics for a specific thread.
        
        Args:
            thread_id: The thread ID to get usage for
            
        Returns:
            dict: Token usage statistics or None if not available
        """
        return self.token_usage.get(thread_id)
    
    def initialize_from_history(self, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Initialize a conversation from existing message history (like from a Slack thread).
        
        Args:
            thread_id: The Slack thread ID to associate with this conversation
            messages: List of messages in the conversation in OpenAI format
        """
        self.conversations[thread_id] = {
            "messages": messages,
            "response_id": None  # Will be set on first API call
        }
        logger.info(f"Initialized conversation for thread {thread_id} with {len(messages)} messages from history") 