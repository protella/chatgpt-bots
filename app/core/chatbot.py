import os
import logging
from typing import Dict, List, Optional, Any
import openai
from openai.types.chat import ChatCompletion

# Import system prompt
import prompts

# Import logging
from app.core.logging import setup_logger

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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Cache to store previous_response_id for each thread
        self.thread_responses: Dict[str, str] = {}
        
        # Cache to store token usage metrics
        self.token_usage: Dict[str, Dict[str, int]] = {}
        
        # Default model to use
        self.model = "gpt-4.1-2025-04-14"
    
    def get_response(self, 
                     input_text: str, 
                     thread_id: str, 
                     images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get a response from OpenAI's API for the given input.
        Handles both text-only and multimodal (text + images) requests.
        
        Args:
            input_text: The text input from the user
            thread_id: The thread ID to maintain conversation context
            images: Optional list of base64-encoded image strings
            
        Returns:
            dict: Response containing content, success flag, and optional error
        """
        try:
            # Check if this is a new thread or continuing conversation
            is_new_thread = thread_id not in self.thread_responses
            
            # Prepare the messages list for the API call
            messages = []
            
            # Only add system prompt for new threads
            if is_new_thread:
                messages.append(prompts.SLACK_SYSTEM_PROMPT)
            
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
                for image_base64 in images:
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "auto"
                        }
                    })
            
            messages.append(user_message)
            
            logger.info(f"Sending request to OpenAI for thread {thread_id}")
            
            # Get configuration options (placeholder - will be implemented in Phase 4)
            # In a real implementation, this would come from core/config.py
            max_tokens = 4096
            temperature = 0.7
            
            # Prepare API call parameters
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "store": True,  # Store conversation state in OpenAI
            }
            
            # Add previous_response_id for continuing conversations
            if not is_new_thread and thread_id in self.thread_responses:
                params["previous_response_id"] = self.thread_responses[thread_id]
                logger.debug(f"Using previous_response_id: {self.thread_responses[thread_id]}")
            
            # Make the API call
            response = self.client.chat.completions.create(**params)
            
            # Store the response ID for future messages in this thread
            self.thread_responses[thread_id] = response.id
            logger.debug(f"Stored new response ID: {response.id} for thread {thread_id}")
            
            # Track token usage for this thread
            if hasattr(response, 'usage') and response.usage:
                self.token_usage[thread_id] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                logger.info(f"Thread {thread_id} token usage: {self.token_usage[thread_id]}")
            
            # Extract and return the response content
            return {
                "content": response.choices[0].message.content,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {str(e)}")
            return {
                "content": "",
                "success": False,
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