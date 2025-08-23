# User Settings Modal Implementation Plan

## Overview
Implement a Slack Block Kit interactive modal that allows users to configure bot settings. The modal automatically appears for new users and can be accessed anytime via the `/settings` slash command.

## Key Features
- Auto-trigger on first user interaction
- Dynamic modal updates based on model selection
- Pre-populated with defaults from `.env`
- Persistent user preferences in database
- Settings hierarchy: Thread overrides > User preferences > System defaults

## User-Exposable Settings

### Basic Settings

**Model Selection**
- Display: Dropdown with `GPT-4o` | `GPT-5`
- Maps to: `gpt-4o` | `gpt-5`
- Default: From `GPT_MODEL` env variable

**Reasoning Level** (GPT-5 models only)
- Display: Radio buttons `Low` | `Medium` | `High`
- Maps to: `low` | `medium` | `high`
- Default: From `DEFAULT_REASONING_EFFORT`
- Note: Excludes `minimal` to maintain web search compatibility

**Response Detail** (GPT-5 models only)
- Display: Radio buttons `Concise` | `Standard` | `Detailed`
- Maps to: `low` | `medium` | `high` (verbosity)
- Default: From `DEFAULT_VERBOSITY`

**Temperature** (GPT-4o only)
- Display: Number input 0.0-2.0
- Default: From `DEFAULT_TEMPERATURE`

**Top P** (GPT-4o only)
- Display: Number input 0.0-1.0
- Default: From `DEFAULT_TOP_P`

### Feature Toggles

**Web Search**
- Display: Checkbox "Enable web search"
- Default: From `ENABLE_WEB_SEARCH`

**Streaming**
- Display: Checkbox "Enable streaming responses"
- Default: From `ENABLE_STREAMING`

### Image Generation Settings

**Default Image Size**
- Display: Dropdown
  - `Square (1024x1024)`
  - `Portrait (1024x1792)`
  - `Landscape (1792x1024)`
- Maps to: `1024x1024` | `1024x1792` | `1792x1024`
- Default: From `DEFAULT_IMAGE_SIZE`

**Input Fidelity** (for image edits)
- Display: Radio buttons
  - `Preserve Original Style` → `high`
  - `Allow Reinterpretation` → `low`
- Default: From `DEFAULT_INPUT_FIDELITY`

**Vision Analysis Detail**
- Display: Radio buttons `Auto` | `Low Detail` | `High Detail`
- Maps to: `auto` | `low` | `high`
- Default: From `DEFAULT_DETAIL_LEVEL`

## Database Schema

```sql
CREATE TABLE user_preferences (
    slack_user_id TEXT PRIMARY KEY,
    slack_email TEXT,
    
    -- Model settings
    model TEXT DEFAULT 'gpt-5',
    reasoning_effort TEXT DEFAULT 'low',
    verbosity TEXT DEFAULT 'low',
    temperature REAL DEFAULT 1.0,
    top_p REAL DEFAULT 1.0,
    
    -- Feature toggles
    enable_web_search BOOLEAN DEFAULT TRUE,
    enable_streaming BOOLEAN DEFAULT TRUE,
    
    -- Image settings
    image_size TEXT DEFAULT '1024x1024',
    input_fidelity TEXT DEFAULT 'high',
    vision_detail TEXT DEFAULT 'auto',
    
    -- Metadata
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now')),
    settings_completed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_user_prefs_email ON user_preferences(slack_email);
```

## Implementation Architecture

### 1. New User Detection

```python
async def handle_message(self, event):
    user_id = event['user']
    
    # Check if new user
    user_prefs = self.db.get_user_preferences(user_id)
    
    if not user_prefs:
        # New user detected
        default_prefs = self.create_default_preferences(user_id)
        self.db.insert_user_preferences(default_prefs)
        
        # Open welcome modal
        trigger_id = event.get('trigger_id')
        if trigger_id:
            self.show_welcome_modal(user_id, trigger_id, default_prefs)
            return "👋 Welcome! I've opened your settings. You can accept the defaults or customize them. I'll wait for you to save your preferences."
        
    # Continue with normal processing
    return await self.process_message(event, user_prefs)
```

### 2. Modal Construction

```python
def build_settings_blocks(self, user_id, current_settings, selected_model=None):
    """Build modal blocks dynamically based on model selection"""
    
    blocks = [
        # Welcome message for new users
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Welcome to the AI Assistant!*\nThese are your default settings. Feel free to customize them or click Save to accept."
            }
        },
        {"type": "divider"},
        
        # Model selection (always shown)
        {
            "type": "section",
            "block_id": "model_block",
            "text": {"type": "mrkdwn", "text": "*Model*"},
            "accessory": {
                "type": "static_select",
                "action_id": "model_select",
                "initial_option": {
                    "text": {"type": "plain_text", "text": "GPT-5" if current_settings['model'] == 'gpt-5' else "GPT-4o"},
                    "value": current_settings['model']
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "GPT-4o"}, "value": "gpt-4o"},
                    {"text": {"type": "plain_text", "text": "GPT-5"}, "value": "gpt-5"}
                ]
            }
        }
    ]
    
    # Conditionally add model-specific settings
    if selected_model == 'gpt-5':
        blocks.extend(self._add_gpt5_settings(current_settings))
    elif selected_model == 'gpt-4o':
        blocks.extend(self._add_gpt4o_settings(current_settings))
    
    # Add common settings
    blocks.extend(self._add_common_settings(current_settings))
    
    # Add footer with tip
    blocks.extend([
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"💡 *Tip:* You can change these settings anytime by typing `{SETTINGS_SLASH_COMMAND}`"
                }
            ]
        }
    ])
    
    return blocks
```

### 3. Dynamic Modal Updates

```python
@app.action("model_select")
async def handle_model_change(ack, body, client):
    """Update modal dynamically when model selection changes"""
    await ack()
    
    selected_model = body['actions'][0]['selected_option']['value']
    user_id = body['user']['id']
    
    # Get current form values to preserve them
    current_values = extract_current_values(body['view']['state']['values'])
    
    # Rebuild blocks with new model selection
    new_blocks = build_settings_blocks(
        user_id=user_id,
        current_settings=current_values,
        selected_model=selected_model
    )
    
    # Update the modal
    client.views_update(
        view_id=body['view']['id'],
        view={
            "type": "modal",
            "callback_id": body['view']['callback_id'],
            "title": {"type": "plain_text", "text": "Bot Settings"},
            "submit": {"type": "plain_text", "text": "Save Settings"},
            "blocks": new_blocks
        }
    )
```

### 4. Settings Submission Handler

```python
@app.view("welcome_settings_modal")
@app.view("settings_modal")
async def handle_settings_submission(ack, body, view):
    """Process and save user settings"""
    await ack()
    
    user_id = body['user']['id']
    values = view['state']['values']
    
    # Extract settings from form
    settings = {
        'slack_user_id': user_id,
        'model': extract_value(values, 'model_select'),
        'enable_web_search': 'web_search' in extract_checkboxes(values, 'features'),
        'enable_streaming': 'streaming' in extract_checkboxes(values, 'features'),
        'image_size': extract_value(values, 'image_size'),
        'input_fidelity': extract_value(values, 'input_fidelity'),
        'vision_detail': extract_value(values, 'vision_detail'),
        'settings_completed': True,
        'updated_at': int(time.time())
    }
    
    # Add model-specific settings
    if settings['model'] == 'gpt-5':
        settings['reasoning_effort'] = extract_value(values, 'reasoning_level')
        settings['verbosity'] = extract_value(values, 'verbosity')
    else:
        settings['temperature'] = extract_value(values, 'temperature')
        settings['top_p'] = extract_value(values, 'top_p')
    
    # Validate and save
    validated_settings = self.validate_settings(settings)
    self.db.update_user_preferences(user_id, validated_settings)
    
    # Send confirmation
    await self.send_confirmation(user_id, "✅ Settings saved successfully!")
```

### 5. Slash Command Handler

```python
@app.command("/settings")
async def handle_settings_command(ack, body, client):
    """Open settings modal via slash command"""
    await ack()
    
    user_id = body['user_id']
    trigger_id = body['trigger_id']
    
    # Get current settings
    current_settings = self.db.get_user_preferences(user_id)
    
    if not current_settings:
        # First time user using slash command
        current_settings = self.create_default_preferences(user_id)
        self.show_welcome_modal(user_id, trigger_id, current_settings)
    else:
        # Existing user
        self.show_settings_modal(user_id, trigger_id, current_settings)
```

## Settings Validation

```python
def validate_settings(self, settings):
    """Ensure settings compatibility and apply business rules"""
    
    # If web search enabled but reasoning too low, auto-upgrade
    if settings.get('enable_web_search') and settings.get('reasoning_effort') == 'minimal':
        settings['reasoning_effort'] = 'low'
        
    # Remove invalid parameters for model type
    if settings['model'] in ['gpt-5', 'gpt-5-mini']:
        settings.pop('temperature', None)
        settings.pop('top_p', None)
    else:
        settings.pop('reasoning_effort', None)
        settings.pop('verbosity', None)
    
    return settings
```

## Settings Priority System

```python
def get_effective_settings(self, user_id, thread_id):
    """Determine effective settings based on hierarchy"""
    
    # 1. Start with system defaults from .env
    settings = self.get_system_defaults()
    
    # 2. Override with user preferences
    user_prefs = self.db.get_user_preferences(user_id)
    if user_prefs:
        settings.update(user_prefs)
    
    # 3. Override with thread-specific settings
    thread_settings = self.thread_manager.get_thread_settings(thread_id)
    if thread_settings:
        settings.update(thread_settings)
    
    return settings
```

## Slack App Configuration Requirements

### Required Scopes
- `commands` - Register and handle slash commands
- `chat:write` - Send messages
- `im:write` - Send DMs for settings confirmation
- `views:write` - Open and update modals

### Event Subscriptions
- `app_mention` - Respond to @mentions
- `message.channels` - Listen to channel messages
- `message.im` - Listen to DMs

### Interactivity & Shortcuts
- Enable Interactivity
- Set Request URL to your app's endpoint
- Configure to handle:
  - `view_submission` - Modal form submissions
  - `block_actions` - Interactive component actions

### Slash Commands
Register settings command (configurable via SETTINGS_SLASH_COMMAND env variable):

**Production (`/chatgpt-settings`):**
- Command: `/chatgpt-settings`
- Request URL: `https://your-app.com/slack/commands`
- Short Description: "Configure your ChatGPT bot preferences"
- Usage Hint: "Opens settings for model, reasoning level, and features"

**Development (`/chatgpt-config-dev`):**
- Command: `/chatgpt-config-dev`
- Request URL: `https://your-app-dev.com/slack/commands`
- Short Description: "Configure ChatGPT bot preferences (dev)"
- Usage Hint: "Opens bot settings modal - DEV ENVIRONMENT"

**Environment Variable:**
```bash
# .env
SETTINGS_SLASH_COMMAND = "/chatgpt-settings"  # Or "/chatgpt-config-dev" for dev
```

## Error Handling

```python
class SettingsError(Exception):
    """Base exception for settings-related errors"""
    pass

def handle_settings_error(error, user_id):
    """Graceful error handling for settings operations"""
    
    if isinstance(error, ValidationError):
        return "⚠️ Invalid settings combination. Please check your selections."
    elif isinstance(error, DatabaseError):
        logger.error(f"Database error for user {user_id}: {error}")
        return "❌ Unable to save settings. Please try again."
    else:
        logger.error(f"Unexpected settings error: {error}")
        return "❌ An error occurred. Your settings were not saved."
```

## Testing Plan

### Unit Tests
- Test settings validation logic
- Test database operations
- Test default preference creation
- Test settings hierarchy resolution

### Integration Tests
- Test modal opening for new users
- Test slash command functionality
- Test dynamic modal updates
- Test settings persistence

### User Acceptance Tests
- New user sees modal on first interaction
- Settings persist across sessions
- Thread overrides work correctly
- Model-specific settings show/hide appropriately

## Migration Strategy

For existing deployments:
1. Add `user_preferences` table to existing databases
2. Run migration to populate with current defaults
3. Set `settings_completed = FALSE` for all users
4. Users will see welcome modal on next interaction

## Future Enhancements

1. **Settings Profiles**: Save multiple configuration presets
2. **Team Settings**: Admin-configured defaults for organization
3. **Export/Import**: Share settings configurations
4. **Settings History**: Track changes over time
5. **A/B Testing**: Test different default configurations