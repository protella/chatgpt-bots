# Slack App Requirements Documentation - V2 Planning

## Executive Summary

This document captures the functional requirements of the current Slack chatbot application, focusing on features and capabilities rather than technical implementation details. This serves as the foundation for planning V2 of the application.

## 1. Core Functionality

### 1.1 Chat Conversation Management
- **Threaded Conversations**: Support for Slack thread-based conversations with context preservation
- **Direct Messages**: Support for one-on-one conversations with the bot in DM channels
- **App Mentions**: Respond to @mentions in channels to engage in conversations
- **Session Persistence**: Users must be able to continue conversations seamlessly even if the bot restarts between messages
- **Context Continuity**: The bot must maintain full awareness of prior conversation context regardless of system restarts

#### Architectural Decision: Source of Truth
- **Slack as Primary Source**: Slack threads will remain the source of truth for conversation history
- **No Response Chains**: Do not use `previous_response_id` - each API call is stateless
- **Rationale**: Slack represents what users actually see and experience; handles message edits and deletions that would break Responses API chains
- **Stateless Requests**: Each request includes necessary context built from local memory
- **Trade-offs Accepted**: Slightly higher token usage per request is acceptable for simplicity and reliability

### 1.2 Message Processing
- **Text Processing**: Process and respond to text-based messages
- **User ID Removal**: Clean user mentions from messages before processing
- **Message Queue Management**: Handle concurrent message requests with thread-based queueing
- **Busy State Management**: Detect and handle when a thread is already processing a request
- **Duplicate Prevention**: Ignore message_changed events to prevent duplicate responses

### 1.3 Conversation State Management

#### Image Context Handling
- **Asset Ledger**: Maintain separate storage for generated images outside the main conversation chain
- **Breadcrumb Strategy**: Store lightweight text references in conversation (e.g., "Created image - 'enhanced prompt first 100 chars...'")
- **Selective Rehydration**: Only include image data in API calls when user is referencing them, not in every request
- **No Role Confusion**: Avoid storing assistant-generated images as user messages

#### Intent Classification System
- **LLM-Based Classification**: Use dedicated LLM calls (nano model) to classify user intent
- **Intent Types**: Classify messages as `new_image`, `modify_image`, or `text_only`
- **Context Window Approach**: Provide classifier with sliding window of recent conversation events (last 6-8 exchanges)
- **No State Tracking**: Avoid brittle state machines; instead track conversation flow naturally
- **Separate Classification Chain**: Maintain micro-chain or stateless calls for intent classification to avoid polluting main conversation
- **Minimal Context**: Classifier receives only essential context (~200-400 tokens) not full conversation history

#### Image Rehydration Strategy
- **Simple Approach**: When intent = `modify_image`, include ALL images from the thread's asset ledger
- **Let the Model Decide**: The main model is smart enough to determine which image the user is referencing
- **No Complex Matching**: Avoid fragile heuristics or complex parsing logic
- **User Experience First**: Better to send all images and guarantee correct behavior than risk confusion
- **Practical Limits**: Cap at last 5-10 images to avoid context window issues
- **Future Optimization**: Can add smarter selection later if needed, but start simple

#### Auxiliary API Calls
- **Isolation Principle**: Auxiliary calls (intent checks, prompt enhancement) must not affect main conversation chain
- **Ephemeral Processing**: Use `store=false` or separate chains for auxiliary operations
- **Clean Architecture**: Maintain clear separation between main conversation and meta-operations

## 2. AI Integration Features

### 2.1 Language Model Integration
- **Chat Completions**: Generate intelligent responses using OpenAI GPT models
- **Model Selection**: Configurable GPT model selection (GPT-4, GPT-5, etc.)
- **System Prompts**: Customizable system prompts to define bot personality and behavior
- **Reasoning Models Support**: Support for GPT-5 reasoning models with specific parameters
- **Temperature Control**: Adjustable response creativity/randomness (0.0-2.0 for most models; fixed at 1.0 for GPT-5 reasoning models)
- **Token Management**: Configurable maximum response length

### 2.2 Image Generation
- **GPT-Image-1 Model**: Generate images using the new gpt-image-1 model (replacing DALL-E 3)
- **Automatic Detection**: Intelligently detect when users want image generation vs. text responses
- **Prompt Enhancement**: Use AI to enhance user prompts for better image generation
- **Image Quality Options**: Support for high, medium, and low quality settings
- **Size Options**: 1024x1024 (square), 1536x1024 (landscape), 1024x1536 (portrait), or auto
- **Background Transparency**: Support for transparent, opaque, or auto background settings
- **Output Formats**: PNG, JPEG, or WebP with configurable compression (0-100%)
- **Streaming Support**: Ability to stream partial images during generation
- **Multiple Images**: Support for generating multiple images per request
- **Image Editing**: Support for editing existing images with up to 16 input images
- **Input Fidelity**: Control how closely edits match input images (high/low)
- **Content Moderation**: Configurable moderation levels (auto/low)
- **Base64 Only**: Images always returned as base64-encoded data (no URL option)
- **No Revised Prompts**: Unlike DALL-E 3, does not return revised/enhanced prompts

### 2.3 Vision Capabilities
- **Image Analysis**: Process and analyze uploaded images (JPEG, PNG, GIF, WebP)
- **Multi-Image Support**: Handle multiple images in a single message
- **Context-Aware Vision**: Combine text questions with image uploads
- **Image Detail Control**: Adjustable analysis detail level (auto, low, high)

## 3. Configuration Management

### 3.1 Configurable Parameters
- **Temperature**: Response creativity (0.0 - 2.0)
- **Top-p**: Nucleus sampling parameter (0.0 - 1.0)
- **Max Completion Tokens**: Maximum response length (up to 4096)
- **Reasoning Effort**: For GPT-5 models (minimal, low, medium, high)
- **Verbosity**: Response detail level (low, medium, high)
- **GPT Model**: Primary language model selection
- **DALL-E Model**: Image generation model selection
- **Image Size**: Generated image dimensions
- **Image Quality**: HD or standard quality
- **Image Style**: Natural or vivid style
- **Detail Level**: Vision analysis detail
- **System Prompt**: Bot personality and behavior definition

### 3.2 Configuration Persistence
- **Thread-Level Config**: Configuration changes apply to specific threads
- **Default Values**: Fallback to default configuration when not specified
- **Dynamic Updates**: Real-time configuration changes without restart

## 4. User Experience Features

### 4.1 Response Formatting
- **Markdown to Slack Mrkdwn**: Automatic conversion of standard Markdown to Slack formatting
- **Code Block Support**: Preserve code formatting in responses
- **List Formatting**: Proper bullet point hierarchy with Slack-specific characters
- **Text Styling**: Support for bold, italic, and strikethrough text
- **Link Formatting**: Convert Markdown links to Slack link format

### 4.2 Status Indicators
- **Processing Indicator**: "Thinking..." message with loading emoji
- **Image Generation Status**: "Generating image, please wait..." feedback
- **Busy Messages**: Clear indication when thread is processing another request
- **Error Messages**: Formatted error feedback with emoji indicators

### 4.3 Message Management
- **Temporary Message Cleanup**: Automatic deletion of status/loading messages
- **Error Recovery**: Graceful error handling with user-friendly messages
- **Thread Organization**: Maintain clean thread structure

## 5. Authentication & Authorization

### 5.1 Slack Authentication
- **Bot Token**: OAuth bot user token for API access
- **App Token**: Socket mode app-level token for real-time events
- **Workspace Authorization**: Proper workspace-level permissions
- **File Access**: Authorization for downloading user-uploaded files

### 5.2 OpenAI Authentication
- **API Key Management**: Secure storage and usage of OpenAI API key
- **Service Access**: Authentication for GPT and DALL-E services

## 6. Event Handling

### 6.1 Slack Events
- **App Mentions**: Respond to @bot mentions in channels
- **Direct Messages**: Process messages in DM channels
- **Message Events**: Handle new messages in threads
- **File Uploads**: Process file sharing events

### 6.2 Event Filtering
- **Bot Message Filtering**: Ignore bot's own messages
- **Subtype Filtering**: Skip message_changed events
- **Channel Type Detection**: Differentiate between DMs and channel messages

## 7. File Handling

### 7.1 File Upload Processing
- **Image Files**: Support for JPEG, PNG, GIF, WebP formats
- **File Download**: Secure download with authentication
- **Base64 Encoding**: Convert files for AI processing
- **File Type Validation**: Check MIME types before processing

### 7.2 File Response to Slack
- **Required Upload Method**: Use `files_upload_v2` SDK method (wraps the new async upload flow)
- **Deprecation Timeline**: Old `files.upload` sunset November 12, 2025
- **Under the Hood**: `files_upload_v2` automatically handles:
  - `files.getUploadURLExternal` for upload URL generation
  - Direct upload to storage URL
  - `files.completeUploadExternal` to finalize
- **Image Upload**: Upload generated images to Slack with metadata
- **File Naming**: Appropriate naming for generated content
- **Thread Association**: Attach files to correct conversation threads
- **Initial Comments**: Support for structured payloads with blocks and initial_comment
- **Async Processing**: File processing continues asynchronously on Slack's servers after upload

## 8. Conversation History Management

### 8.1 History Architecture
- **Source of Truth**: Slack threads are the authoritative source for conversation history
- **Local State Management**: Per-thread memory containing:
  - Text-only message history (user/assistant exchanges without images)
  - Sliding window of recent events for intent classification (last 6-8 exchanges)
  - Current system prompt
  - Thread metadata (thread_id, channel_id, processing state)
- **Asset Separation**: Store images in separate asset ledger with:
  - Image ID, base64 data (BytesIO in memory during runtime)
  - Slack file metadata (url_private, timestamp, caption)
  - Reference to which thread/message created it
  - **Persistence Strategy**: Memory-only, rebuilt from Slack on demand
  - **Recovery Approach**: When rebuilding thread from `conversations.replies`, extract file metadata but don't download actual bytes until needed
  - **Future Optimization**: Background prefetch - When rebuilding a thread, spawn background task to download images while processing intent classification. Since threads are accessed one at a time after restart, can repopulate asset ledger in parallel without blocking response
- **Breadcrumb Strategy**: Include text references to images (e.g., "Created image - 'prompt...'") instead of full image data
- **Role Clarity**: Maintain clear user/assistant role separation without confusion

### 8.2 History Reconstruction from Slack
- **Thread Recovery**: Rebuild history from Slack API when bot restarts or reconnects
- **Message Filtering**: Exclude temporary status messages, loading indicators, and error messages
- **Smart Image Handling**: 
  - Identify bot-generated images from file metadata
  - Convert to breadcrumb references in conversation
  - Store image data in asset ledger if needed for reference
- **Context Window Management**: Maintain sliding window of recent exchanges for efficient API calls
- **No State Assumptions**: Design for stateless recovery at any point

## 9. Concurrency & Performance

### 9.1 Thread Management
- **Concurrent Processing**: Support multiple threads processing simultaneously
- **Thread Isolation**: Prevent cross-thread interference
- **Busy State Blocking**: Reject rapid messages in same thread with busy feedback (no queuing)
- **Lock Management**: Thread-safe processing state management
- **Naming Clarity**: Rename QueueManager to ThreadLockManager for V2 (reflects actual behavior)

### 9.2 Rate Limiting
- **Busy State Detection**: Prevent duplicate processing in same thread
- **User Feedback**: Show clear busy message when thread is processing

## 10. Logging & Monitoring

### 10.1 Logging Capabilities
- **Configurable Log Levels**: Support for DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Module-Specific Logging**: Separate log configuration for different components
- **Session Markers**: Clear session start/end indicators
- **Error Tracking**: Detailed error logging with stack traces

### 10.2 Debug Features
- **Message History Logging**: Debug-level conversation tracking
- **API Call Logging**: Track interactions with external services
- **Configuration Logging**: Log configuration changes and values

## 11. Error Handling

### 11.1 Error Recovery
- **Graceful Degradation**: Continue operation despite individual errors
- **User Notification**: Clear error messages to users
- **Context Cleanup**: Remove failed messages from history
- **Retry Logic**: Handle transient failures appropriately

### 11.2 Error Types
- **API Errors**: Handle OpenAI API failures
- **Network Errors**: Manage connection issues
- **File Processing Errors**: Handle invalid or corrupted files
- **Configuration Errors**: Validate and handle invalid settings

## 12. Environment Configuration

### 12.1 Environment Variables
- **Service Tokens**: SLACK_BOT_TOKEN, SLACK_APP_TOKEN, OPENAI_KEY
- **Model Selection**: GPT_MODEL, DALLE_MODEL, UTILITY_MODEL
- **Logging Configuration**: Log level settings per module
- **Feature Flags**: Toggle specific capabilities

### 12.2 Deployment Flexibility
- **Socket Mode Support**: Real-time event handling
- **Configurable Endpoints**: Flexible service configuration
- **Environment-based Settings**: Different configs for dev/staging/prod

## 13. User Interaction Patterns

### 13.1 Conversation Flows
- **Question-Answer**: Simple Q&A interactions
- **Multi-turn Dialogue**: Extended conversations with context
- **Image Generation Workflow**: Prompt → Generate → Display → Discuss
- **Vision Analysis Flow**: Upload → Analyze → Respond
- **Configuration Management**: View → Modify → Confirm


## 14. Non-Functional Requirements (Observed)

### 14.1 Performance
- **Response Time**: Quick initial acknowledgment with loading indicator
- **Concurrent Users**: Support multiple simultaneous conversations
- **Memory Management**: Per-thread conversation isolation

### 14.2 Reliability
- **Error Recovery**: Continue operation after individual failures
- **Session Persistence**: Rebuild state after disconnection
- **Message Delivery**: Ensure responses reach correct threads

### 14.3 Usability
- **Natural Language**: No special syntax required for conversations
- **Visual Feedback**: Clear status indicators during processing
- **Error Clarity**: Understandable error messages for users

## 15. Integration Points

### 15.1 External Services
- **OpenAI API**: GPT models and image generation
- **Slack API**: Workspace integration and event handling
- **Memory-Based File Handling**: Use BytesIO for virtual files, no disk I/O required

### 15.2 Internal Components
- **Queue Manager**: Thread-safe message processing
- **Markdown Converter**: Format transformation
- **Logger System**: Centralized logging
- **Configuration Manager**: Settings and preferences

## 16. Future Features & Experimental Capabilities

### 16.1 Response Streaming (Experimental)
- **Streaming Updates**: Use `chat.update` to show response as it's generated
- **Rate Limit Awareness**: 
  - Tier 3 allows ~50 requests per minute per workspace per method
  - Effective rate: ~0.83 requests per second with burst allowance
  - Rate limits apply per method, per workspace, per app
- **Dynamic Fallback**: 
  - Start with streaming mode
  - Monitor for 429 responses and `X-RateLimit-Remaining` headers
  - Automatically fall back to single update mode when throttled
  - Re-enable streaming after cooldown period
- **Update Coalescing Strategy**:
  - Conservative: Update every 1-2 seconds (30-60 updates/min max)
  - Allows multiple concurrent streaming responses within limits
  - Adjust dynamically based on active streams in workspace
- **Smart Chunking**: Update on sentence/paragraph boundaries for better UX
- **Per-Workspace Tracking**: Each workspace has independent rate limit
- **Graceful Degradation**: Users still get full response even if streaming fails
- **2025 Rate Limit Changes**: Monitor for potential stricter limits for non-Marketplace apps

### 16.2 Extended File Support (Future)
- **Document Processing**: Support for PDF, DOCX, TXT files
- **Text Extraction**: Convert documents to text for LLM processing
- **File Type Detection**: Automatic handling based on MIME type

### 16.3 Tool Integration (Future)
- **Web Search**: Real-time web search capability
- **Code Execution**: Safe code interpretation
- **External APIs**: Extensible tool framework

## Conclusion

This requirements document captures the complete functional scope of the current Slack chatbot application. These requirements serve as the baseline for planning and developing V2, ensuring all existing capabilities are maintained while providing a foundation for enhancements and new features.

### Key Strengths to Preserve in V2
- Robust thread-based conversation management
- Intelligent context-aware responses
- Seamless image generation integration
- Flexible configuration system
- Strong error handling and recovery

### Areas for Potential Enhancement in V2
- Enhanced file type support beyond images
- Persistent storage of conversations
- User-specific preferences
- Advanced analytics and usage tracking
- Expanded command system
- Multi-workspace support
- Rate limiting and usage quotas
- Enhanced security features