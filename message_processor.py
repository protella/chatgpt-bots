"""
Shared Message Processor
Client-agnostic message processing logic
"""
import base64
import time
from typing import Dict, Any, List, Optional
from base_client import BaseClient, Message, Response
from thread_manager import ThreadStateManager
from openai_client import OpenAIClient, ImageData
from config import config
from logger import LoggerMixin
from prompts import SLACK_SYSTEM_PROMPT, DISCORD_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT, IMAGE_ANALYSIS_PROMPT


class MessageProcessor(LoggerMixin):
    """Handles message processing logic independent of chat platform"""
    
    def __init__(self):
        self.thread_manager = ThreadStateManager()
        self.openai_client = OpenAIClient()
        self.log_info("MessageProcessor initialized")
    
    def process_message(self, message: Message, client: BaseClient, thinking_id: Optional[str] = None) -> Optional[Response]:
        """
        Process a message and return a response
        
        Args:
            message: Universal message object
            client: The client that received the message
            thinking_id: ID of the thinking indicator message to update
        
        Returns:
            Response object or None if unable to process
        """
        thread_key = f"{message.channel_id}:{message.thread_id}"
        
        # Check if thread is busy
        if not self.thread_manager.acquire_thread_lock(
            message.thread_id, 
            message.channel_id,
            timeout=0  # Don't wait, return immediately if busy
        ):
            return Response(
                type="busy",
                content="Thread is currently processing another request"
            )
        
        try:
            # Get or rebuild thread state
            thread_state = self._get_or_rebuild_thread_state(
                message,
                client
            )
            
            # Set platform-specific system prompt if not already set
            if not thread_state.system_prompt:
                thread_state.system_prompt = self._get_system_prompt(client)
            
            # Process any attachments (images)
            image_inputs = self._process_attachments(message, client)
            
            # Build user content
            user_content = self._build_user_content(message.text, image_inputs)
            
            # Determine intent based on context
            if image_inputs:
                # User uploaded images - determine if it's vision or edit request
                if not message.text or message.text.strip() == "":
                    # No text with images - default to vision (analyze)
                    intent = "vision"
                    self.log_debug("No text with images - defaulting to vision analysis")
                else:
                    # Has text with images - classify if it's edit or vision
                    intent = self.openai_client.classify_intent(
                        thread_state.get_recent_messages(),
                        message.text
                    )
                    # If classified as image generation with uploaded images, it's an edit
                    if intent in ["new_image", "modify_image"]:
                        intent = "edit_image"
                    else:
                        # Text request with images = vision analysis
                        intent = "vision"
            else:
                # No images uploaded - standard classification
                intent = self.openai_client.classify_intent(
                    thread_state.get_recent_messages(),
                    message.text if message.text else ""
                )
            
            self.log_debug(f"Classified intent: {intent}")
            
            # Update thinking indicator if generating/editing image
            if intent in ["new_image", "modify_image", "edit_image"] and thinking_id:
                self._update_thinking_for_image(client, message.channel_id, thinking_id)
            
            # Generate response based on intent
            if intent == "new_image":
                response = self._handle_image_generation(message.text, thread_state)
            elif intent == "modify_image":
                response = self._handle_image_modification(
                    message.text, 
                    thread_state, 
                    message.thread_id,
                    client
                )
            elif intent == "edit_image":
                # User uploaded images with edit request
                response = self._handle_image_edit(
                    message.text,
                    image_inputs,
                    thread_state
                )
            elif intent == "vision":
                # Vision analysis with uploaded images
                response = self._handle_text_response(user_content, thread_state, client)
            else:
                response = self._handle_text_response(user_content, thread_state, client)
            
            # DEBUG: Print conversation history after processing
            import json
            print("\n" + "="*80)
            print("DEBUG: CONVERSATION HISTORY (RAW JSON)")
            print("="*80)
            print(json.dumps(thread_state.messages, indent=2))
            print("="*80 + "\n")
            
            return response
            
        except Exception as e:
            self.log_error(f"Error processing message: {e}", exc_info=True)
            return Response(
                type="error",
                content=str(e)
            )
        finally:
            self.thread_manager.release_thread_lock(
                message.thread_id,
                message.channel_id
            )
    
    def _get_or_rebuild_thread_state(
        self,
        message: Message,
        client: BaseClient
    ) -> Any:
        """Get existing thread state or rebuild from platform history"""
        thread_state = self.thread_manager.get_or_create_thread(
            message.thread_id,
            message.channel_id
        )
        
        # If thread has no messages, rebuild from platform
        if not thread_state.messages:
            self.log_info(f"Rebuilding thread state for {message.thread_id}")
            
            # Get history from platform
            history = client.get_thread_history(
                message.channel_id,
                message.thread_id
            )
            
            # Get current message timestamp to exclude it
            current_ts = message.metadata.get("ts")
            
            # Convert to thread state messages
            for hist_msg in history:
                # Skip the current message being processed
                if hist_msg.metadata.get("ts") == current_ts:
                    continue
                    
                # Determine role based on metadata
                is_bot = hist_msg.metadata.get("is_bot", False)
                role = "assistant" if is_bot else "user"
                
                # Add breadcrumbs for attachments
                content = hist_msg.text
                if hist_msg.attachments:
                    att_count = len(hist_msg.attachments)
                    content += f" [Uploaded {att_count} file(s)]"
                
                thread_state.add_message(role, content)
            
            self.log_info(f"Rebuilt thread with {len(thread_state.messages)} messages")
        
        return thread_state
    
    def _process_attachments(
        self,
        message: Message,
        client: BaseClient
    ) -> List[Dict]:
        """Process message attachments (mainly images)"""
        image_inputs = []
        image_count = 0
        max_images = 10
        
        for attachment in message.attachments:
            # Stop if we've reached the limit
            if image_count >= max_images:
                self.log_warning(f"Limiting to {max_images} images (user uploaded more)")
                break
            if attachment.get("type") == "image":
                try:
                    # Download the image
                    image_data = client.download_file(
                        attachment.get("url"),
                        attachment.get("id")
                    )
                    
                    if image_data:
                        # Convert to base64
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        
                        # Format for Responses API with base64
                        mimetype = attachment.get("mimetype", "image/png")
                        image_inputs.append({
                            "type": "input_image",
                            "image_url": f"data:{mimetype};base64,{base64_data}"
                        })
                        
                        image_count += 1
                        self.log_debug(f"Processed image {image_count}/{max_images}: {attachment.get('name')}")
                
                except Exception as e:
                    self.log_error(f"Error processing attachment: {e}")
        
        return image_inputs
    
    def _build_user_content(self, text: str, image_inputs: List[Dict]) -> Any:
        """Build user message content"""
        if image_inputs:
            # Multi-part content with text and images
            content = [{"type": "input_text", "text": text}]
            content.extend(image_inputs)
            return content
        else:
            # Simple text content
            return text
    
    def _get_system_prompt(self, client: BaseClient) -> str:
        """Get the appropriate system prompt based on the client platform"""
        client_name = client.name.lower()
        
        if "slack" in client_name:
            return SLACK_SYSTEM_PROMPT["content"]
        elif "discord" in client_name:
            return DISCORD_SYSTEM_PROMPT["content"]
        else:
            # Default/CLI prompt
            return CLI_SYSTEM_PROMPT["content"]
    
    def _update_thinking_for_image(self, client: BaseClient, channel_id: str, thinking_id: str):
        """Update the thinking indicator to show image generation message"""
        if hasattr(client, 'update_message'):
            client.update_message(
                channel_id,
                thinking_id,
                f"{config.thinking_emoji} Generating image. This could take up to a minute, please wait..."
            )
        else:
            self.log_debug("Client doesn't support message updates")
    
    def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient) -> Response:
        """Handle text-only response generation"""
        # Add user message to thread state
        thread_state.add_message("user", user_content)
        
        # Get thread config
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Use thread's system prompt (which is now platform-specific)
        system_prompt = thread_state.system_prompt or self._get_system_prompt(client)
        
        # Generate response
        response_text = self.openai_client.create_text_response(
            messages=thread_state.messages,
            model=thread_config["model"],
            temperature=thread_config["temperature"],
            max_tokens=thread_config["max_tokens"],
            system_prompt=system_prompt,
            reasoning_effort=thread_config.get("reasoning_effort"),
            verbosity=thread_config.get("verbosity")
        )
        
        # Add assistant response to thread state
        thread_state.add_message("assistant", response_text)
        
        return Response(
            type="text",
            content=response_text
        )
    
    def _handle_image_generation(self, prompt: str, thread_state) -> Response:
        """Handle image generation request"""
        self.log_info(f"Generating image for prompt: {prompt[:100]}...")
        
        # Get thread config
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Generate image with conversation context for better prompt enhancement
        image_data = self.openai_client.generate_image(
            prompt=prompt,
            size=thread_config.get("image_size"),
            quality=thread_config.get("image_quality"),
            enhance_prompt=True,
            conversation_history=thread_state.get_recent_messages()  # Pass conversation context
        )
        
        # Store in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # Use the enhanced prompt
            time.time()
        )
        
        # Add breadcrumb to thread state with the enhanced prompt used
        thread_state.add_message("user", prompt)
        thread_state.add_message("assistant", f"Generated image: {image_data.prompt}")
        
        return Response(
            type="image",
            content=image_data
        )
    
    def _handle_image_modification(
        self,
        text: str,
        thread_state,
        thread_id: str,
        client: BaseClient
    ) -> Response:
        """Handle image modification request"""
        # Get asset ledger
        asset_ledger = self.thread_manager.get_asset_ledger(thread_id)
        
        if not asset_ledger or not asset_ledger.images:
            # No images in memory - check if there were images in conversation history
            has_previous_images = False
            previous_prompts = []
            
            # Look for image generation breadcrumbs in thread history
            for msg in thread_state.messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Check for either format (thread state vs Slack upload comment)
                    if "Here's the image I created of" in content:
                        has_previous_images = True
                        # Extract the prompt from the breadcrumb
                        prompt_start = content.find("Here's the image I created of") + len("Here's the image I created of")
                        previous_prompts.append(content[prompt_start:].strip())
                    elif "Generated image:" in content:
                        has_previous_images = True
                        # Extract the prompt from the upload comment
                        prompt_start = content.find("Generated image:") + len("Generated image:")
                        # Strip any formatting characters and emoji
                        prompt = content[prompt_start:].strip().strip("_")
                        previous_prompts.append(prompt)
            
            if has_previous_images:
                # User is asking to modify a previous image
                # Since we can't retrieve the actual image data after restart,
                # generate a new image based on the modification request and context
                self.log_info("Found previous image references, generating modified version")
                
                # Build context including previous prompts
                context_prompt = text
                if previous_prompts:
                    context_prompt = f"Previous image: {previous_prompts[-1]}\nModification request: {text}"
                
                return self._handle_image_generation(context_prompt, thread_state)
            else:
                # No images were ever generated, treat as text request
                return self._handle_text_response(text, thread_state, client)
        
        # Get recent images for context
        recent_images = asset_ledger.get_recent_images(1)  # Get the most recent image
        
        # User is asking to modify an existing image
        # Build context including the previous image prompt
        if recent_images:
            previous_prompt = recent_images[0].get("prompt", "")
            context_prompt = f"Previous image: {previous_prompt}\nModification request: {text}"
        else:
            # Shouldn't happen but fallback to just the request
            context_prompt = text
        
        self.log_info("Modifying existing image with new request")
        
        # Generate a new image based on the modification request
        return self._handle_image_generation(context_prompt, thread_state)
    
    def _handle_image_edit(
        self,
        text: str,
        image_inputs: List[Dict],
        thread_state
    ) -> Response:
        """Handle image editing with uploaded images"""
        # Extract base64 data and mime types from image inputs
        input_images = []
        input_mimetypes = []
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    # Parse data URL: data:image/png;base64,xxxxx
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        header, base64_data = parts
                        # Extract mimetype from header
                        mimetype_part = header.split(";")[0].replace("data:", "")
                        mimetype = mimetype_part if mimetype_part else "image/png"
                        
                        # OpenAI doesn't support GIF for editing, convert to PNG
                        if mimetype == "image/gif":
                            self.log_warning("Converting GIF to PNG for image edit (GIF not supported)")
                            mimetype = "image/png"
                        
                        input_images.append(base64_data)
                        input_mimetypes.append(mimetype)
        
        if not input_images:
            # Shouldn't happen but fallback to generation
            return self._handle_image_generation(text, thread_state)
        
        self.log_info(f"Editing {len(input_images)} uploaded image(s)")
        
        # First, analyze the uploaded images to get context
        self.log_debug("Analyzing uploaded images for context")
        
        # Log the analysis prompt
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 1: ANALYZE IMAGE")
        print("="*80)
        print(f"Analysis Question: {IMAGE_ANALYSIS_PROMPT}")
        print("="*80)
        
        try:
            # Analyze the images to understand what's in them
            image_description = self.openai_client.analyze_images(
                images=input_images,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high"
            )
            
            # Log the full analysis result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 2: ANALYSIS RESULT")
            print("="*80)
            print(f"Image Description (Full):\n{image_description}")
            print("="*80)
            
            # Log what we're sending to the enhancer
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 3: INPUTS FOR ENHANCEMENT")
            print("="*80)
            print(f"Image Description: {image_description[:200]}..." if len(image_description) > 200 else f"Image Description: {image_description}")
            print(f"\nUser's Edit Request: {text}")
            print("="*80)
            
            # Store the description and user request separately for clean enhancement
            image_analysis = image_description
            user_edit_request = text
            
        except Exception as e:
            self.log_warning(f"Failed to analyze images, continuing without context: {e}")
            image_analysis = None
            user_edit_request = text
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - ANALYSIS FAILED")
            print("="*80)
            print(f"Error: {e}")
            print(f"Falling back to user prompt only: {text}")
            print("="*80)
        
        # Get thread config for settings
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Use the edit_image API with separated inputs
        try:
            image_data = self.openai_client.edit_image(
                input_images=input_images,
                input_mimetypes=input_mimetypes,
                prompt=user_edit_request,  # Just the user's request
                image_description=image_analysis,  # The analyzed description
                input_fidelity=thread_config.get("input_fidelity", "high"),
                background=thread_config.get("image_background", "auto"),
                output_format=thread_config.get("image_format", "png"),
                output_compression=thread_config.get("image_compression", 100),
                enhance_prompt=True,
                conversation_history=thread_state.messages if thread_state.messages else None
            )
        except Exception as e:
            self.log_error(f"Error editing image: {e}")
            return Response(
                type="error",
                content=f"Failed to edit image: {str(e)}"
            )
        
        # Add breadcrumb to thread state
        thread_state.add_message("user", text or "Edit uploaded image")
        thread_state.add_message("assistant", f"Generated image: {image_data.prompt}")
        
        return Response(
            type="image",
            content=image_data
        )
    
    def update_thread_config(
        self,
        channel_id: str,
        thread_id: str,
        config_updates: Dict[str, Any]
    ):
        """Update configuration for a specific thread"""
        self.thread_manager.update_thread_config(
            thread_id,
            channel_id,
            config_updates
        )
        
    def get_stats(self) -> Dict[str, int]:
        """Get processor statistics"""
        return self.thread_manager.get_stats()