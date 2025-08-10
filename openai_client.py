"""
OpenAI Client wrapper for Responses API
Handles all interactions with OpenAI's GPT and image generation models
"""
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
from config import config
from logger import LoggerMixin
from prompts import IMAGE_CHECK_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT


@dataclass
class ImageData:
    """Container for image data"""
    base64_data: str
    format: str = "png"
    prompt: str = ""
    timestamp: float = 0
    slack_url: Optional[str] = None
    
    def to_bytes(self) -> BytesIO:
        """Convert base64 to BytesIO"""
        return BytesIO(base64.b64decode(self.base64_data))


class OpenAIClient(LoggerMixin):
    """Wrapper for OpenAI API using Responses API"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.log_info("OpenAI client initialized")
    
    def create_text_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,  # Don't store by default for stateless operation
    ) -> str:
        """
        Create a text response using the Responses API
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: System instructions
            reasoning_effort: For GPT-5 models (minimal, low, medium, high)
            verbosity: For GPT-5 models (low, medium, high)
            store: Whether to store the response (default False for stateless)
        
        Returns:
            Generated text response
        """
        model = model or config.gpt_model
        temperature = temperature if temperature is not None else config.default_temperature
        max_tokens = max_tokens or config.default_max_tokens
        
        # Build input for Responses API
        input_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            input_messages.append({
                "role": "developer",
                "content": system_prompt
            })
        
        # Add conversation messages
        input_messages.extend(messages)
        
        # Build request parameters
        request_params = {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "store": store,
        }
        
        # Handle model-specific parameters
        if model.startswith("gpt-5"):
            # Check if it's a reasoning model (not chat model)
            is_reasoning_model = "chat" not in model.lower()
            
            if is_reasoning_model:
                # GPT-5 reasoning models (nano, mini, full)
                # Fixed temperature, supports reasoning_effort and verbosity
                request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models
                reasoning_effort = reasoning_effort or config.default_reasoning_effort
                request_params["reasoning"] = {"effort": reasoning_effort}
                verbosity = verbosity or config.default_verbosity
                request_params["text"] = {"verbosity": verbosity}
            else:
                # GPT-5 chat models - standard parameters only
                # temperature and top_p work normally, no reasoning/verbosity
                pass
        
        self.log_debug(f"Creating text response with model {model}, temp {temperature}")
        
        try:
            response = self.client.responses.create(**request_params)
            
            # Extract text from response
            output_text = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                output_text += content.text
            
            self.log_info(f"Generated response: {len(output_text)} chars")
            return output_text
            
        except Exception as e:
            self.log_error(f"Error creating text response: {e}", exc_info=True)
            raise
    
    def classify_intent(
        self,
        messages: List[Dict[str, Any]],
        last_user_message: str
    ) -> str:
        """
        Classify user intent using a lightweight model
        
        Args:
            messages: Recent conversation context (last 6-8 exchanges)
            last_user_message: The latest user message to classify
        
        Returns:
            Intent classification: 'new_image', 'modify_image', or 'text_only'
        """
        # Create conversation history for context
        context = "Conversation History:\n"
        has_recent_image = False
        
        for msg in messages[-6:]:  # Last 6 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                content = " ".join(text_parts)
            
            # Check if assistant recently generated an image
            if role == "assistant" and "generated image" in content.lower():
                has_recent_image = True
            
            context += f"{role}: {content[:200]}...\n" if len(content) > 200 else f"{role}: {content}\n"
        
        context += f"\nCurrent User Message:\n{last_user_message}"
        
        try:
            # Use the IMAGE_CHECK_SYSTEM_PROMPT from prompts.py
            # Utility model is gpt-5-mini (reasoning model)
            response = self.client.responses.create(
                model=config.utility_model,
                input=[
                    {"role": "developer", "content": IMAGE_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                temperature=1.0,  # Fixed for reasoning models
                max_output_tokens=100,  # Reasonable minimum for reasoning tokens
                store=False,  # Never store classification calls
                reasoning={"effort": "minimal"},  # Minimal for speed
                text={"verbosity": "low"}  # Low verbosity for simple True/False
            )
            
            # Extract True/False response
            result = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                result += content.text
            
            result = result.strip().lower()
            
            # Debug logging
            self.log_debug(f"Image check raw result: '{result}' for message: '{last_user_message[:50]}...'")
            
            # Convert to our intent categories
            if result == "true":
                # If there was a recent image and user wants image, it's likely a modification
                if has_recent_image:
                    intent = "modify_image"
                else:
                    intent = "new_image"
            else:
                intent = "text_only"
            
            self.log_debug(f"Classified intent: {intent}")
            return intent
            
        except Exception as e:
            self.log_error(f"Error classifying intent: {e}")
            return 'text_only'  # Default to text on error
    
    def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[str] = None,
        format: Optional[str] = None,
        compression: Optional[int] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ImageData:
        """
        Generate an image using GPT-Image-1 model
        
        Args:
            prompt: Image generation prompt
            size: Image size (1024x1024, etc. - check model capabilities)
            quality: Quality setting (reserved for future DALL-E 3 support)
            background: Background type (reserved for future use)
            format: Output format (always returns png for now)
            compression: Compression level (reserved for future use)
            enhance_prompt: Whether to enhance the prompt first
        
        Returns:
            ImageData object with generated image
        """
        # Default size for gpt-image-1
        size = size or config.default_image_size
        
        # Quality parameter is not used by gpt-image-1
        # Reserved for future DALL-E 3 support
        
        # Enhance prompt if requested
        enhanced_prompt = prompt
        if enhance_prompt:
            enhanced_prompt = self._enhance_image_prompt(prompt, conversation_history)
        
        self.log_info(f"Generating image: {prompt[:100]}...")
        
        try:
            # Build parameters for images.generate
            # Default to gpt-image-1 parameters
            params = {
                "model": config.image_model,  # gpt-image-1
                "prompt": enhanced_prompt,  # Use the enhanced prompt
                "n": 1  # Number of images to generate
            }
            
            # Add size if specified (gpt-image-1 supports size)
            if size:
                params["size"] = size
            
            # Note: gpt-image-1 doesn't support response_format or quality parameters
            # It returns URLs that we'll download and convert to base64
            
            # Future: When adding DALL-E 3 support, check model and add:
            # - response_format="b64_json"
            # - quality parameter
            # - style parameter
            
            # Use the images.generate API for image generation
            response = self.client.images.generate(**params)
            
            # Extract image data from response
            if response.data and len(response.data) > 0:
                # Check if we have base64 data
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    image_data = response.data[0].b64_json
                # Otherwise, we might have a URL - need to download it
                elif hasattr(response.data[0], 'url') and response.data[0].url:
                    import requests
                    import base64
                    url = response.data[0].url
                    self.log_debug(f"Downloading image from URL: {url}")
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        image_data = base64.b64encode(img_response.content).decode('utf-8')
                    else:
                        raise ValueError(f"Failed to download image from URL: {url}")
                else:
                    raise ValueError("No image data or URL in response")
            else:
                raise ValueError("No image data in response")
            
            self.log_info("Image generated successfully")
            
            return ImageData(
                base64_data=image_data,
                format="png",  # API always returns PNG for now
                prompt=enhanced_prompt,  # Store the enhanced prompt that was actually used
            )
            
        except Exception as e:
            self.log_error(f"Error generating image: {e}", exc_info=True)
            raise
    
    def _enhance_image_prompt(self, prompt: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Enhance an image generation prompt for better results
        
        Args:
            prompt: Original user prompt
            conversation_history: Recent conversation messages for context
        
        Returns:
            Enhanced prompt
        """
        # Build conversation context
        context = "Conversation History:\n"
        
        if conversation_history:
            # Include recent messages for context (last 6-8 messages)
            for msg in conversation_history[-8:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle multi-part content
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                    content = " ".join(text_parts)
                
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                
                context += f"{role}: {content}\n"
        
        context += f"\nCurrent User Request: {prompt}"
        
        try:
            # Use the IMAGE_GEN_SYSTEM_PROMPT from prompts.py
            # Utility model is gpt-5-mini (reasoning model)
            response = self.client.responses.create(
                model=config.utility_model,
                input=[
                    {"role": "developer", "content": IMAGE_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                temperature=1.0,  # Fixed for reasoning models
                max_output_tokens=500,  # Increased for detailed image prompts
                store=False,
                reasoning={"effort": "minimal"},  # Minimal effort for faster processing
                text={"verbosity": "low"}  # Low verbosity for concise prompts
            )
            
            enhanced = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                enhanced += content.text
            
            enhanced = enhanced.strip()
            
            # Make sure we got a valid enhancement
            if enhanced and len(enhanced) > 10:
                self.log_debug(f"Enhanced prompt: {enhanced[:100]}...")
                return enhanced
            else:
                return prompt
            
        except Exception as e:
            self.log_warning(f"Failed to enhance prompt: {e}")
            return prompt  # Return original on error
    
    def analyze_image(
        self,
        image_data: str,
        question: str,
        detail: Optional[str] = None
    ) -> str:
        """
        Analyze an image with a question
        
        Args:
            image_data: Base64 encoded image data
            question: Question about the image
            detail: Analysis detail level (auto, low, high)
        
        Returns:
            Analysis response
        """
        detail = detail or config.default_detail_level
        
        try:
            response = self.client.responses.create(
                model=config.gpt_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": question},
                            {
                                "type": "input_image",
                                "image": {
                                    "base64": image_data,
                                    "detail": detail
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Extract response text
            output_text = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                output_text += content.text
            
            return output_text
            
        except Exception as e:
            self.log_error(f"Error analyzing image: {e}", exc_info=True)
            raise