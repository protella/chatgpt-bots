# MCP Tool Slash Command Implementation Guide

## Overview
This document describes the implementation of Slack slash commands that force specific MCP tool intents, bypassing non-deterministic LLM intent classification. This provides users a direct way to invoke MCP tools when automatic intent detection doesn't trigger reliably.

## Problem Statement
- MCP tool intent classification via LLM is non-deterministic and sometimes fails to trigger
- Users need a reliable way to directly access specific MCP tool functionality
- Solution: Implement slash commands that force specific MCP tool intents

## Implementation Steps

### 1. Environment Configuration

#### Add Environment Variables
**File: `.env`**
```bash
# MCP tool slash command (customize per tool and environment)
MCP_TOOL_SLASH_COMMAND = "/mcp-tool-dev"
```

#### Update Configuration Class
**File: `config.py`**

Add the new configuration field after the settings_slash_command (around line 99-102):
```python
# Slack settings configuration
settings_slash_command: str = field(default_factory=lambda: os.getenv("SETTINGS_SLASH_COMMAND", "/chatgpt-settings"))

# MCP tool slash command
mcp_tool_slash_command: str = field(default_factory=lambda: os.getenv("MCP_TOOL_SLASH_COMMAND", "/mcp-tool-dev"))

# MCP Integration settings
enable_mcp_tools: bool = True  # Feature flag for MCP tools
```

### 2. Slash Command Handler Implementation

**File: `slack_client/event_handlers/settings.py`**

Add the ReportPro slash command handler in the `_register_settings_handlers` method, BEFORE the existing settings command handler:

```python
def _register_settings_handlers(self):
    # Register ReportPro slash command handler
    @self.app.command(config.reportpro_slash_command)
    async def handle_reportpro_command(ack, body, client):
        """Handle the ReportPro slash command"""
        await ack()  # Acknowledge command receipt immediately

        user_id = body.get('user_id')
        channel_id = body.get('channel_id')
        command_text = body.get('text', '').strip()

        self.log_info(f"ReportPro command invoked by user {user_id} in channel {channel_id}: {command_text[:100]}")

        # For DM channels that start with 'D', check if it's a self-DM
        # Self-DMs will fail when we try to post, so redirect to bot's DM
        original_channel_id = channel_id
        if channel_id.startswith('D'):
            # Try to ensure we have the right DM channel with the bot
            try:
                open_result = await client.conversations_open(users=user_id)
                if open_result.get('ok'):
                    bot_dm_channel = open_result['channel']['id']
                    if bot_dm_channel != channel_id:
                        self.log_info(f"Redirecting from DM {channel_id} to bot DM {bot_dm_channel}")
                        channel_id = bot_dm_channel
            except Exception as e:
                self.log_debug(f"Could not verify DM channel: {e}")
                # Continue with original channel and let it fail naturally if needed

        # Check if command text is empty
        if not command_text:
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="❓ **Usage:** `/reportpro-dev [your question]`\n\nPlease provide a question about food industry data, trends, or statistics."
                )
            except Exception as e:
                self.log_error(f"Error sending usage message: {e}")
            return

        try:
            # First, send the user's query as the thread starter
            # This will show in the main channel/DM view
            initial_response = await client.chat_postMessage(
                channel=channel_id,
                text=f"<@{user_id}> asked ReportPro: _{command_text}_"
            )

            # Get the thread timestamp
            thread_ts = initial_response['ts']
            message_ts = initial_response['ts']

            # Now post the settings button as the first reply in the thread
            button_value = json.dumps({
                "channel_id": channel_id,
                "thread_id": thread_ts
            })

            # Use the same compact format as standard threads for existing users
            settings_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "⚙️ *Quick Settings Access*"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Settings"
                            },
                            "style": "primary",
                            "action_id": "open_welcome_settings",
                            "value": button_value
                        }
                    ]
                }
            ]

            # Post settings button as a reply to the initial message
            settings_response = await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,  # Reply to the initial message
                blocks=settings_blocks,
                text="Settings"  # Fallback text
            )

            if not settings_response.get('ok'):
                self.log_error(f"Failed to send settings button: {settings_response}")
                # Continue anyway - settings button is optional

            # Create a synthetic message event to process through normal flow
            synthetic_event = {
                'type': 'message',
                'text': command_text,
                'user': user_id,
                'channel': channel_id,
                'ts': message_ts,
                'thread_ts': thread_ts,  # Use the initial message as thread root
                # Add a special flag to force ReportPro intent
                'force_intent': 'reportpro',
                # Don't pass thinking_id - let the response be a new message in the thread
                # 'thinking_id': message_ts  # REMOVED - we want a new message, not update
            }

            # Process the message through the normal handler
            # This will use the forced intent and handle the ReportPro query
            await self._handle_slack_message(synthetic_event, client, is_slash_command=True)

        except Exception as e:
            self.log_error(f"Error handling ReportPro command: {e}", exc_info=True)

            # Check if it's a channel_not_found error
            error_msg = str(e)
            if 'channel_not_found' in error_msg or 'not_in_channel' in error_msg:
                # Bot is not in the channel
                try:
                    await client.chat_postEphemeral(
                        channel=original_channel_id,
                        user=user_id,
                        text="❌ I need to be added to this channel to use the `/reportpro-dev` command.\n\nPlease invite me to the channel or use the command in a direct message."
                    )
                except Exception:
                    pass
            else:
                # Other error
                try:
                    await client.chat_postEphemeral(
                        channel=original_channel_id,
                        user=user_id,
                        text="❌ Sorry, there was an error processing your ReportPro query. Please try again."
                    )
                except Exception:
                    pass

    # Register slash command handler (existing settings command)
    @self.app.command(config.settings_slash_command)
    async def handle_settings_command(ack, body, client):
        # ... existing settings command code ...
```

**Required imports at top of file:**
```python
import json
```

### 3. Message Handler Updates

**File: `slack_client/event_handlers/message_events.py`**

#### Update _handle_slack_message Signature
Add the `is_slash_command` parameter to the method signature (around line 12):
```python
async def _handle_slack_message(self, event: Dict[str, Any], client, is_slash_command: bool = False):
    """Convert Slack event to universal Message format"""
```

#### Pass Force Intent Through Metadata
Update the Message creation to include forced intent (around line 62-79):
```python
# Create universal message
message = Message(
    text=text,
    user_id=user_id,
    channel_id=event.get("channel"),
    thread_id=event.get("thread_ts") or event.get("ts"),
    attachments=attachments,
    metadata={
        "ts": event.get("ts"),
        "slack_client": client,
        "username": username,  # Add username to metadata
        "user_real_name": user_real_name,  # Add real name to metadata
        "force_intent": event.get("force_intent"),  # Pass through forced intent if present
        "thinking_id": event.get("thinking_id"),  # Pass through thinking_id for slash commands
        "user_email": user_email,  # Add email to metadata
        "user_timezone": user_timezone,  # Add timezone to metadata
        "user_tz_label": user_tz_label  # Add timezone label (EST, PST, etc.)
    }
)
```

#### Skip Settings Button for Slash Commands
Update the settings button posting logic (around line 261-265):
```python
else:
    # Existing user with preferences - check if this is a new thread that needs a settings button
    # Skip if this is a slash command (settings button already posted)
    if not is_slash_command:
        await self._post_settings_button_if_new_thread(message, client, user_prefs)

# Call the message handler if set
```

### 4. Message Processor Updates

**File: `message_processor/base.py`**

Add check for forced intent in process_message method (around line 376-382):
```python
    # Use the original request text for processing
    message.text = original_request
    self.log_debug(f"Clarified intent: {intent}")

# Check for forced intent from slash commands
elif message.metadata and message.metadata.get('force_intent'):
    intent = message.metadata.get('force_intent')
    self.log_info(f"Using forced intent from slash command: {intent}")

# Determine intent based on context
elif image_inputs:
```

### 5. Main Handler Updates

**File: `main.py`**

Update handle_message to check for existing thinking_id (around line 64-74):
```python
async def handle_message(self, message: Message, client: BaseClient):
    """Handle incoming message from any platform"""
    # Check if thinking_id was provided (e.g., from slash command)
    thinking_id = message.metadata.get('thinking_id') if message.metadata else None

    # Send initial thinking indicator if not already provided
    if not thinking_id:
        thinking_id = await client.send_thinking_indicator(
            message.channel_id,
            message.thread_id
        )
```

## Message Flow

When `/mcp-tool [query]` is used:

1. **Command Received**: Slash command handler acknowledges immediately
2. **Channel Validation**:
   - For DMs: Checks if it's a self-DM and redirects to bot's DM if needed
   - For channels: Attempts to post directly (will fail if bot not in channel)
3. **Thread Creation**:
   - Posts initial message: "<@user> invoked MCP tool: _[query]_"
   - Posts settings button as first reply
4. **Force Intent**: Creates synthetic message event with `force_intent: 'mcp_tool_name'`
5. **Process Message**:
   - Message processor detects forced intent and skips LLM classification
   - Routes directly to appropriate MCP tool handler
6. **Response**: MCP tool response posted as second reply in thread

## Error Handling

### Channel Access Issues
- **Self-DMs**: Automatically redirects to bot's DM channel
- **Channels bot not in**: Shows ephemeral error message
- **Private channels**: Works if bot is member (no special handling needed)

### Empty Commands
- Shows usage message if no query text provided

### API Failures
- Generic error message for unexpected failures
- Specific message for channel access issues

## Slack Configuration

When setting up the slash command in Slack admin:

- **Command**: `/mcp-tool-dev` (or `/mcp-tool` for production, customize per tool)
- **Request URL**: Your app's request URL endpoint
- **Short Description**: "Direct access to MCP tool functionality"
- **Usage Hint**: "Enter your query for the MCP tool"

## Testing Checklist

- [ ] Command works in public channels where bot is present
- [ ] Command works in private channels where bot is present
- [ ] Command shows error in channels where bot is not present
- [ ] Command works in DM with bot
- [ ] Command redirects from self-DM to bot's DM
- [ ] Settings button appears as first reply
- [ ] MCP tool response appears as second reply
- [ ] Original query is preserved in main message
- [ ] Empty command shows usage instructions

## Key Design Decisions

1. **Force Intent Pattern**: Used metadata to pass forced intent through the message pipeline
2. **Thread Structure**: Query as main message, settings button first, response second
3. **No Thinking ID Update**: Response posts as new message instead of updating initial message
4. **Channel Check Strategy**: Try posting directly instead of using conversations.info (avoids permission issues)
5. **Self-DM Handling**: Automatic redirect to bot's proper DM channel
6. **Settings Button Consistency**: Same format as regular threads

## Potential Issues & Solutions

1. **Issue**: `conversations.info` requires `groups:read` scope for private channels
   **Solution**: Removed pre-check, just try to post and handle errors

2. **Issue**: Self-DMs have different channel IDs that bots can't access
   **Solution**: Use `conversations.open` to get proper bot DM channel

3. **Issue**: Can't send ephemeral messages to self-DMs
   **Solution**: Silent redirect without notification

4. **Issue**: Settings button appearing after MCP tool response
   **Solution**: Post settings button before processing MCP tool query

## Future Improvements

- Add configuration for different MCP tool commands per environment
- Support for multiple MCP tool slash commands with dynamic registration
- Analytics tracking for MCP tool slash command usage
- Rate limiting per user
- Dynamic MCP tool discovery and slash command generation
- Support for MCP tool parameters in slash command text