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
from prompts import IMAGE_CHECK_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT, IMAGE_EDIT_SYSTEM_PROMPT


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
        top_p: Optional[float] = None,
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
            top_p: Nucleus sampling parameter (not supported by GPT-5 reasoning models)
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
        top_p = top_p if top_p is not None else config.default_top_p
        
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
                request_params["top_p"] = top_p
        else:
            # GPT-4 and other models - include top_p
            request_params["top_p"] = top_p
        
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
        
        for msg in messages:  # Include all messages for full context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                content = " ".join(text_parts)
            
            # Check if assistant recently generated an image
            if role == "assistant" and "generated image" in content.lower():
                has_recent_image = True
            
            # Truncate very long messages to keep context reasonable
            if len(content) > 500:
                context += f"{role}: {content[:500]}...\n"
            else:
                context += f"{role}: {content}\n"
        
        context += f"\nCurrent User Message:\n{last_user_message}"
        
        try:
            # Build request parameters
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": IMAGE_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                "max_output_tokens": 100,
                "store": False,  # Never store classification calls
            }
            
            # Check if we're using a GPT-5 reasoning model
            if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                # GPT-5 reasoning model - use fixed temperature and reasoning parameters
                request_params["temperature"] = 1.0  # Fixed for reasoning models
                request_params["reasoning"] = {"effort": "minimal"}  # Minimal for speed
                request_params["text"] = {"verbosity": "low"}  # Low verbosity for simple True/False
            else:
                # GPT-4 or other models - use standard parameters
                request_params["temperature"] = 0.3  # Low temperature for consistent classification
            
            response = self.client.responses.create(**request_params)
            
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
    
    def _enhance_image_edit_prompt(
        self,
        user_request: str,
        image_description: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Enhance an image editing prompt using the analyzed image description
        
        Args:
            user_request: User's edit request
            image_description: Description of the current image
            conversation_history: Recent conversation messages for additional context
        
        Returns:
            Enhanced edit prompt
        """
        # Build context with image description and user request
        context = f"Image Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"
        
        # Add conversation history if it exists and has messages
        if conversation_history and len(conversation_history) > 0:
            context = "Previous Conversation:\n"
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle multi-part content
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                    content = " ".join(text_parts)
                
                # Truncate long messages
                if len(content) > 150:
                    content = content[:150] + "..."
                
                context += f"{role}: {content}\n"
            
            context += f"\nImage Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"
        
        # Log the enhancement input
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 4: EDIT PROMPT ENHANCEMENT")
        print("="*80)
        print(f"User Request: {user_request}")
        print(f"Image Description: {image_description[:200]}..." if len(image_description) > 200 else f"Image Description: {image_description}")
        if conversation_history and len(conversation_history) > 0:
            print(f"Including {len(conversation_history)} conversation messages for context")
        print("="*80)
        
        try:
            # Build request parameters with edit-specific system prompt
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": IMAGE_EDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                "max_output_tokens": 500,
                "store": False,
            }
            
            # Check if we're using a GPT-5 reasoning model
            if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                request_params["temperature"] = 1.0
                request_params["reasoning"] = {"effort": "minimal"}
                request_params["text"] = {"verbosity": "low"}
            else:
                request_params["temperature"] = 0.7
            
            response = self.client.responses.create(**request_params)
            
            enhanced = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                enhanced += content.text
            
            enhanced = enhanced.strip()
            
            # Log the enhanced result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED EDIT PROMPT")
            print("="*80)
            print(f"Final Enhanced Edit Prompt:\n{enhanced}")
            print("="*80)
            
            if enhanced and len(enhanced) > 10:
                return enhanced
            else:
                # Fallback to simple combination
                return f"Edit the image: {image_description}. Change: {user_request}"
            
        except Exception as e:
            self.log_warning(f"Failed to enhance edit prompt: {e}")
            return f"Edit the image: {image_description}. Change: {user_request}"
    
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
        
        # Log the prompt enhancement input
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 4: PROMPT ENHANCEMENT INPUT")
        print("="*80)
        print(f"Original Prompt to Enhance:\n{prompt}")
        print(f"\nFull Context Sent to Enhancer:\n{context}")
        print("="*80)
        
        try:
            # Build request parameters
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": IMAGE_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                "max_output_tokens": 500,  # Increased for detailed image prompts
                "store": False,
            }
            
            # Check if we're using a GPT-5 reasoning model
            if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                # GPT-5 reasoning model - use fixed temperature and reasoning parameters
                request_params["temperature"] = 1.0  # Fixed for reasoning models
                request_params["reasoning"] = {"effort": "minimal"}  # Minimal effort for faster processing
                request_params["text"] = {"verbosity": "low"}  # Low verbosity for concise prompts
            else:
                # GPT-4 or other models - use standard parameters
                request_params["temperature"] = 0.7  # Moderate temperature for creative prompts
            
            response = self.client.responses.create(**request_params)
            
            enhanced = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                enhanced += content.text
            
            enhanced = enhanced.strip()
            
            # Log the enhanced prompt result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED PROMPT OUTPUT")
            print("="*80)
            print(f"Final Enhanced Prompt:\n{enhanced}")
            print("="*80)
            
            # Make sure we got a valid enhancement
            if enhanced and len(enhanced) > 10:
                self.log_debug(f"Enhanced prompt: {enhanced[:100]}...")
                return enhanced
            else:
                print("\n" + "="*80)
                print("DEBUG: Enhancement failed or too short, using original")
                print("="*80)
                return prompt
            
        except Exception as e:
            self.log_warning(f"Failed to enhance prompt: {e}")
            return prompt  # Return original on error
    
    def analyze_images(
        self,
        images: List[str],
        question: str,
        detail: Optional[str] = None
    ) -> str:
        """
        Analyze one or more images with a question
        
        Args:
            images: List of base64 encoded image data (max 10)
            question: Question about the image(s)
            detail: Analysis detail level (auto, low, high)
        
        Returns:
            Analysis response
        """
        detail = detail or config.default_detail_level
        
        # Limit to 10 images
        if len(images) > 10:
            self.log_warning(f"Limiting to 10 images (received {len(images)})")
            images = images[:10]
        
        # Build content array with text and images
        content = [{"type": "input_text", "text": question}]
        
        for image_data in images:
            # Use data URL format for base64 images
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_data}"
            })
        
        try:
            response = self.client.responses.create(
                model=config.gpt_model,
                input=[
                    {
                        "role": "user",
                        "content": content
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
            self.log_error(f"Error analyzing images: {e}", exc_info=True)
            raise
    
    def edit_image(
        self,
        input_images: List[str],
        prompt: str,
        input_mimetypes: Optional[List[str]] = None,
        image_description: Optional[str] = None,
        input_fidelity: str = "low",
        background: Optional[str] = None,
        mask: Optional[str] = None,
        output_format: str = "png",
        output_compression: int = 100,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ImageData:
        """
        Edit or combine images using GPT-Image-1 model
        
        Args:
            input_images: List of base64 encoded input images (max 16)
            prompt: Edit instructions (up to 32000 chars)
            input_fidelity: How closely to match input images (high/low)
            background: Background type (transparent/opaque/auto)
            mask: Optional base64 encoded PNG mask
            output_format: Format (png/jpeg/webp)
            output_compression: Compression level 0-100
            enhance_prompt: Whether to enhance the prompt first
            conversation_history: Recent conversation for context
        
        Returns:
            ImageData object with edited image
        """
        # Limit to 16 images
        if len(input_images) > 16:
            self.log_warning(f"Limiting to 16 images for editing (received {len(input_images)})")
            input_images = input_images[:16]
        
        # Default background
        background = background or config.default_image_background
        
        # Enhance prompt if requested
        enhanced_prompt = prompt
        if enhance_prompt:
            # Use the edit-specific enhancement for image editing
            if image_description:
                enhanced_prompt = self._enhance_image_edit_prompt(
                    user_request=prompt,
                    image_description=image_description,
                    conversation_history=conversation_history
                )
            else:
                # Fallback to regular enhancement if no description
                enhanced_prompt = self._enhance_image_prompt(prompt, conversation_history)
        
        self.log_info(f"Editing {len(input_images)} image(s): {prompt[:100]}...")
        
        try:
            # Convert base64 to BytesIO objects with proper file extension
            from io import BytesIO
            image_files = []
            
            # Default mimetypes if not provided
            if not input_mimetypes:
                input_mimetypes = ["image/png"] * len(input_images)
            
            for i, b64_data in enumerate(input_images):
                image_bytes = base64.b64decode(b64_data)
                bio = BytesIO(image_bytes)
                
                # Determine file extension from mimetype
                mimetype = input_mimetypes[i] if i < len(input_mimetypes) else "image/png"
                if mimetype == "image/jpeg":
                    bio.name = f"image_{i}.jpg"
                elif mimetype == "image/webp":
                    bio.name = f"image_{i}.webp"
                else:  # Default to PNG
                    bio.name = f"image_{i}.png"
                
                image_files.append(bio)
            
            # Build parameters for images.edit
            params = {
                "model": config.image_model,  # gpt-image-1
                "image": image_files if len(image_files) > 1 else image_files[0],
                "prompt": enhanced_prompt,
                "input_fidelity": input_fidelity,
                "background": background,
                "output_format": output_format,
                "n": 1
            }
            
            # Only add compression for JPEG/WebP (PNG must be 100)
            if output_format in ["jpeg", "webp"]:
                params["output_compression"] = output_compression
            elif output_format == "png" and output_compression != 100:
                self.log_debug(f"PNG format requires compression=100, ignoring {output_compression}")
            
            # Add mask if provided
            if mask:
                mask_bytes = base64.b64decode(mask)
                params["mask"] = BytesIO(mask_bytes)
            
            # Use the images.edit API
            response = self.client.images.edit(**params)
            
            # Extract image data from response
            if response.data and len(response.data) > 0:
                # Check if we have base64 data
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    image_data = response.data[0].b64_json
                # Otherwise, we might have a URL - need to download it
                elif hasattr(response.data[0], 'url') and response.data[0].url:
                    import requests
                    url = response.data[0].url
                    self.log_debug(f"Downloading edited image from URL: {url}")
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        image_data = base64.b64encode(img_response.content).decode('utf-8')
                    else:
                        raise ValueError(f"Failed to download edited image from URL: {url}")
                else:
                    raise ValueError("No image data or URL in response")
            else:
                raise ValueError("No image data in response")
            
            self.log_info("Image edited successfully")
            
            return ImageData(
                base64_data=image_data,
                format=output_format,
                prompt=enhanced_prompt,
            )
            
        except Exception as e:
            self.log_error(f"Error editing image: {e}", exc_info=True)
            raise
    
    def analyze_image(self, image_data: str, question: str, detail: Optional[str] = None) -> str:
        """
        Analyze a single image (backward compatibility wrapper)
        
        Args:
            image_data: Base64 encoded image data
            question: Question about the image
            detail: Analysis detail level (auto, low, high)
        
        Returns:
            Analysis response
        """
        return self.analyze_images([image_data], question, detail)