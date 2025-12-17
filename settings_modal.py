"""
User Settings Modal for Slack Bot
Handles the interactive settings configuration interface
"""
from typing import Dict, Optional, List
from config import config
from logger import LoggerMixin
import json
import uuid


class SettingsModal(LoggerMixin):
    """Manages the user settings modal interface"""
    
    def __init__(self, db):
        """Initialize with database connection"""
        self.db = db
        self.logger_name = "SettingsModal"
    
    async def build_settings_modal(self, user_id: str, trigger_id: str,
                            current_settings: Optional[Dict] = None,
                            is_new_user: bool = False,
                            thread_id: Optional[str] = None,
                            in_thread: bool = False,
                            scope: str = None,
                            pending_message: Optional[Dict] = None) -> Dict:
        """
        Build the complete settings modal.

        Args:
            user_id: Slack user ID
            trigger_id: Slack trigger ID for modal
            current_settings: Current user settings
            is_new_user: Whether this is a new user's first setup
            thread_id: Thread ID if opened from within a thread
            in_thread: Whether modal was opened from within a thread
            scope: Selected scope ('thread' or 'global')
            pending_message: Pending message to process after settings save (for new users)

        Returns:
            Modal view dictionary for Slack API
        """
        if not current_settings:
            current_settings = await self.db.get_user_preferences_async(user_id)
            if not current_settings:
                # Get user's email from users table
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = await self.db.create_default_user_preferences_async(user_id, email)
        
        # Determine which model is selected
        selected_model = current_settings.get('model', config.gpt_model)
        
        # Determine default scope if not provided
        if scope is None:
            # New users should always default to global settings
            if is_new_user:
                scope = 'global'
            else:
                scope = 'thread' if in_thread else 'global'
        
        # Build modal blocks
        blocks = self._build_modal_blocks(current_settings, selected_model, is_new_user, in_thread, scope)
        
        # Determine callback ID based on user status
        callback_id = "welcome_settings_modal" if is_new_user else "settings_modal"

        # Determine if we're in dev environment
        is_dev = config.settings_slash_command.endswith("-dev")
        # Slack modal titles have a 24 character limit
        modal_title = "ChatGPT Settings (Dev)" if is_dev else "ChatGPT Bot Settings"

        # Create session for modal state storage
        session_id = str(uuid.uuid4())

        # Build full state to store in DB
        session_state = {
            "settings": current_settings,
            "thread_id": thread_id,
            "in_thread": in_thread,
            "scope": scope
        }

        # Include pending message if provided (for new users)
        if pending_message:
            session_state["pending_message"] = pending_message

        # Store session in database
        await self.db.create_modal_session_async(session_id, user_id, session_state, modal_type='settings')

        # Only store session_id in metadata - much smaller!
        metadata = {
            "session_id": session_id
        }

        return {
            "type": "modal",
            "callback_id": callback_id,
            "title": {"type": "plain_text", "text": modal_title},
            "submit": {"type": "plain_text", "text": "Save Settings"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": blocks,
            "private_metadata": json.dumps(metadata)
        }
    
    def _build_modal_blocks(self, settings: Dict, selected_model: str, 
                           is_new_user: bool = False, in_thread: bool = False,
                           scope: str = None) -> List[Dict]:
        """Build the modal blocks based on current settings and model selection
        
        Args:
            settings: Current settings dictionary
            selected_model: Currently selected model
            is_new_user: Whether this is a new user
            in_thread: Whether modal was opened from within a thread
            scope: The selected scope ('thread' or 'global')
        """
        blocks = []
        
        # Determine default scope if not provided
        if scope is None:
            # New users should always default to global settings
            if is_new_user:
                scope = 'global'
            else:
                scope = 'thread' if in_thread else 'global'
        
        # Welcome message for new users
        if is_new_user:
            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Welcome to the AI Assistant!* 👋\nLet's configure your settings. You can accept the defaults or customize them."
                    }
                },
                {"type": "divider"}
            ])
        
        # Add scope selector for existing users only (new users must configure global first)
        scope_options = []
        
        # New users must configure global settings first
        if is_new_user:
            # For new users, don't show scope selector - they must configure global first
            if in_thread:
                blocks.append({
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "📌 _Setting up your global preferences (applies to all conversations). You can customize thread-specific settings later._"}]
                })
                blocks.append({"type": "divider"})
        else:
            # Add thread option if in a thread
            if in_thread:
                scope_options.append({
                    "text": {"type": "plain_text", "text": "💬 This Thread Only"},
                    "value": "thread",
                    "description": {"type": "plain_text", "text": "Settings apply only to this conversation"}
                })
            
            # Always add global option
            scope_options.append({
                "text": {"type": "plain_text", "text": "🌐 Global Settings"},
                "value": "global",
                "description": {"type": "plain_text", "text": "Settings apply to all conversations"}
            })
        
        # Only add scope selector if there are multiple options
        if len(scope_options) > 1:
            self.log_debug(f"Building scope selector - in_thread: {in_thread}, scope: {scope}, options: {[o['value'] for o in scope_options]}")
            
            # Find the matching option for initial selection
            initial_option = None
            for option in scope_options:
                if option['value'] == scope:
                    initial_option = option
                    break
            
            # If no match found, default to first option
            if not initial_option:
                self.log_warning(f"Scope '{scope}' not found in options, defaulting to first option")
                initial_option = scope_options[0]
            
            self.log_debug(f"Selected initial_option value: {initial_option['value']}")
            
            blocks.append({
                "type": "section",
                "block_id": "scope_selector",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Settings Scope*\nChoose where to save these settings:"
                },
                "accessory": {
                    "type": "radio_buttons",
                    "action_id": "settings_scope",
                    "options": scope_options,
                    "initial_option": initial_option
                }
            })
            
            blocks.append({"type": "divider"})
            
            # Add tip about accessing settings
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"💡 *Tip:* For global settings, type `{config.settings_slash_command}` in any channel/DM (not in a thread)"
                }]
            })
        elif not is_new_user:
            # Single scope available - show header indicating which one
            header_text = "Configure Your Global Settings"
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": header_text}
            })
            
            # Add tip about accessing settings when only global is available
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"💡 *Tip:* You can change these settings anytime by typing:\n`{config.settings_slash_command}` in any channel/DM (not in a thread)"
                }]
            })
        # Model selection (always shown)
        blocks.append({
            "type": "section",
            "block_id": "model_block",
            "text": {
                "type": "mrkdwn",
                "text": "*AI Model*\nChoose your preferred AI model"
            },
            "accessory": {
                "type": "static_select",
                "action_id": "model_select",
                "placeholder": {"type": "plain_text", "text": "Select model"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_model_display_name(selected_model)},
                    "value": selected_model
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "GPT-5.2"}, "value": "gpt-5.2"},
                    {"text": {"type": "plain_text", "text": "GPT-5.1"}, "value": "gpt-5.1"},
                    {"text": {"type": "plain_text", "text": "GPT-5"}, "value": "gpt-5"},
                    {"text": {"type": "plain_text", "text": "GPT-5 Mini"}, "value": "gpt-5-mini"},
                    {"text": {"type": "plain_text", "text": "GPT-4.1"}, "value": "gpt-4.1"},
                    {"text": {"type": "plain_text", "text": "GPT-4o"}, "value": "gpt-4o"}
                ]
            }
        })
        
        blocks.append({"type": "divider"})
        
        # Add model-specific settings
        if selected_model == 'gpt-5.2':
            blocks.extend(self._add_gpt52_settings(settings))
        elif selected_model in ['gpt-5', 'gpt-5-mini']:
            blocks.extend(self._add_gpt5_settings(settings))
        elif selected_model == 'gpt-5.1':
            blocks.extend(self._add_gpt51_settings(settings))
        else:  # gpt-4.1, gpt-4o
            blocks.extend(self._add_gpt4_settings(settings))
        
        # Add common settings (features and image settings)
        blocks.extend(self._add_common_settings(settings))
        
        return blocks
    
    def _add_gpt5_settings(self, settings: Dict) -> List[Dict]:
        """Add GPT-5 specific settings blocks"""
        blocks = []
        
        # Check if web search is enabled - explicitly check for False vs None/missing
        # Default to True only if the key is missing
        if 'enable_web_search' in settings:
            web_search_enabled = bool(settings['enable_web_search'])
        else:
            web_search_enabled = True
        self.log_debug(f"Settings passed to _add_gpt5_settings: enable_web_search={settings.get('enable_web_search')}, evaluated as {web_search_enabled}")
        
        # Reasoning Level
        # Ensure initial option is valid for the current options list
        # Use config defaults if not set in settings
        from config import config
        current_reasoning = settings.get('reasoning_effort', config.default_reasoning_effort)
        self.log_debug(f"Building reasoning options - web_search: {web_search_enabled}, current: {current_reasoning}")
        
        # Force-validate the reasoning level for web search compatibility
        if web_search_enabled and current_reasoning == 'minimal':
            # If web search is on but reasoning is minimal, use low
            current_reasoning = 'low'
            self.log_debug("Adjusted reasoning from minimal to low for display")
        elif not current_reasoning or current_reasoning not in ['minimal', 'low', 'medium', 'high']:
            # Fallback if no valid reasoning is set - use config defaults
            current_reasoning = 'low' if web_search_enabled else config.default_reasoning_effort
            self.log_debug(f"No valid reasoning, defaulting to {current_reasoning}")
        
        # Build options list - use the _get_reasoning_display function for consistency
        reasoning_options = []
        if not web_search_enabled:
            reasoning_options.append({"text": {"type": "plain_text", "text": self._get_reasoning_display('minimal')}, "value": "minimal"})
        reasoning_options.extend([
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('low')}, "value": "low"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('medium')}, "value": "medium"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('high')}, "value": "high"}
        ])
        
        # Log available options
        available_values = [opt['value'] for opt in reasoning_options]
        self.log_debug(f"Reasoning options available: {available_values}, initial: {current_reasoning}")
        
        # Final validation - ensure current_reasoning is in available options
        if current_reasoning not in available_values:
            # This shouldn't happen with our logic above, but let's be safe
            current_reasoning = 'low' if web_search_enabled else config.default_reasoning_effort
            self.log_warning(f"Current reasoning {current_reasoning} not in available options, using fallback")
        
        # Build the reasoning block
        # Use different block_id AND action_id to force Slack mobile to re-render
        # This works around a Slack bug where selections are lost when options change
        if web_search_enabled:
            block_id = "reasoning_block_web"
            action_id = "reasoning_level_no_minimal"
        else:
            block_id = "reasoning_block_no_web"
            action_id = "reasoning_level"

        reasoning_block = {
            "type": "section",
            "block_id": block_id,
            "text": {
                "type": "mrkdwn",
                "text": "*Reasoning Level*\nControls depth of analysis and problem-solving"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": action_id,
                "options": reasoning_options
            }
        }
        
        # Add initial_option if we have a valid selection
        # Always try to set an initial option to work around Slack mobile bug
        if current_reasoning and current_reasoning != 'None' and current_reasoning in available_values:
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
            self.log_debug(f"Set initial_option for reasoning: {current_reasoning}")
        else:
            # If no valid selection, try to provide a sensible default
            # This helps with the Slack mobile bug where selections are lost
            if available_values:
                # Use the first available option as default
                default_value = available_values[0]
                reasoning_block["accessory"]["initial_option"] = {
                    "text": {"type": "plain_text", "text": self._get_reasoning_display(default_value)},
                    "value": default_value
                }
                self.log_debug(f"No valid reasoning selection - set default initial_option: {default_value}")
            else:
                self.log_debug("No reasoning selection and no available values - cannot set initial_option")
        
        blocks.append(reasoning_block)

        # Add a warning if the initial option had to be defaulted (mobile bug workaround)
        if not (current_reasoning and current_reasoning != 'None' and current_reasoning in available_values):
            if available_values:
                blocks.append({
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"ℹ️ _Selection defaulted to {self._get_reasoning_display(available_values[0])} - please verify your preference_"
                    }]
                })
        
        # Add note about minimal restriction only if minimal is shown
        if not web_search_enabled:
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": "_Note: Minimal reasoning will be unavailable if Web Search is enabled_"
                }]
            })
        
        # Response Detail
        blocks.append({
            "type": "section",
            "block_id": "verbosity_block",
            "text": {
                "type": "mrkdwn",
                "text": "*Response Detail*\nControls how detailed responses are"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "verbosity",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_verbosity_display(settings.get('verbosity', config.default_verbosity))},
                    "value": settings.get('verbosity', config.default_verbosity)
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "📝 Concise"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "📄 Standard"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "📚 Detailed"}, "value": "high"}
                ]
            }
        })
        
        blocks.append({"type": "divider"})
        return blocks

    def _add_gpt51_settings(self, settings: Dict) -> List[Dict]:
        """Add GPT-5.1 specific settings blocks"""
        blocks = []

        # Check if web search is enabled - explicitly check for False vs None/missing
        # Default to True only if the key is missing
        if 'enable_web_search' in settings:
            web_search_enabled = bool(settings['enable_web_search'])
        else:
            web_search_enabled = True
        self.log_debug(f"Settings passed to _add_gpt51_settings: enable_web_search={settings.get('enable_web_search')}, evaluated as {web_search_enabled}")

        # Reasoning Level
        # Use config defaults if not set in settings
        from config import config
        current_reasoning = settings.get('reasoning_effort', 'none')  # Default to 'none' for GPT-5.1
        self.log_debug(f"Building reasoning options for GPT-5.1, current: {current_reasoning}")

        # Build options list - GPT-5.1 has 'none' instead of 'minimal' and no web search constraint
        reasoning_options = [
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('none')}, "value": "none"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('low')}, "value": "low"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('medium')}, "value": "medium"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('high')}, "value": "high"}
        ]

        # Log available options
        available_values = [opt['value'] for opt in reasoning_options]
        self.log_debug(f"Reasoning options available for GPT-5.1: {available_values}, initial: {current_reasoning}")

        # Final validation - ensure current_reasoning is in available options
        if current_reasoning not in available_values:
            old_reasoning = current_reasoning  # Save old value for logging
            current_reasoning = 'none'  # Default to 'none' for GPT-5.1
            self.log_warning(f"Current reasoning '{old_reasoning}' not in available options, using 'none'")

        # Build the reasoning block
        reasoning_block = {
            "type": "section",
            "block_id": "reasoning_block_gpt51",
            "text": {
                "type": "mrkdwn",
                "text": "*Reasoning Level*\nControls depth of analysis and problem-solving"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "reasoning_level_gpt51",
                "options": reasoning_options
            }
        }

        # Add initial_option if we have a valid selection
        if current_reasoning and current_reasoning != 'None' and current_reasoning in available_values:
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
            self.log_debug(f"Set initial_option for GPT-5.1 reasoning: {current_reasoning}")
        else:
            # Default to 'none' if no valid selection
            if available_values:
                default_value = 'none'
                reasoning_block["accessory"]["initial_option"] = {
                    "text": {"type": "plain_text", "text": self._get_reasoning_display(default_value)},
                    "value": default_value
                }
                self.log_debug(f"No valid reasoning selection - set default initial_option: {default_value}")

        blocks.append(reasoning_block)

        # Response Detail
        blocks.append({
            "type": "section",
            "block_id": "verbosity_block",
            "text": {
                "type": "mrkdwn",
                "text": "*Response Detail*\nControls how detailed responses are"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "verbosity",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_verbosity_display(settings.get('verbosity', config.default_verbosity))},
                    "value": settings.get('verbosity', config.default_verbosity)
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "📝 Concise"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "📄 Standard"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "📚 Detailed"}, "value": "high"}
                ]
            }
        })

        blocks.append({"type": "divider"})
        return blocks

    def _add_gpt52_settings(self, settings: Dict) -> List[Dict]:
        """Add GPT-5.2 specific settings blocks (includes xhigh reasoning)"""
        blocks = []

        # Check if web search is enabled
        if 'enable_web_search' in settings:
            web_search_enabled = bool(settings['enable_web_search'])
        else:
            web_search_enabled = True
        self.log_debug(f"Settings passed to _add_gpt52_settings: enable_web_search={settings.get('enable_web_search')}, evaluated as {web_search_enabled}")

        # Reasoning Level - GPT-5.2 supports 'xhigh' in addition to standard levels
        from config import config
        current_reasoning = settings.get('reasoning_effort', 'none')  # Default to 'none' for GPT-5.2
        self.log_debug(f"Building reasoning options for GPT-5.2, current: {current_reasoning}")

        # Build options list - GPT-5.2 has 'none' and adds 'xhigh' for maximum quality
        reasoning_options = [
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('none')}, "value": "none"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('low')}, "value": "low"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('medium')}, "value": "medium"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('high')}, "value": "high"},
            {"text": {"type": "plain_text", "text": self._get_reasoning_display('xhigh')}, "value": "xhigh"}
        ]

        # Log available options
        available_values = [opt['value'] for opt in reasoning_options]
        self.log_debug(f"Reasoning options available for GPT-5.2: {available_values}, initial: {current_reasoning}")

        # Final validation - ensure current_reasoning is in available options
        if current_reasoning not in available_values:
            old_reasoning = current_reasoning
            current_reasoning = 'none'  # Default to 'none' for GPT-5.2
            self.log_warning(f"Current reasoning '{old_reasoning}' not in available options, using 'none'")

        # Build the reasoning block
        reasoning_block = {
            "type": "section",
            "block_id": "reasoning_block_gpt52",
            "text": {
                "type": "mrkdwn",
                "text": "*Reasoning Level*\nControls depth of analysis and problem-solving"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "reasoning_level_gpt52",
                "options": reasoning_options
            }
        }

        # Add initial_option if we have a valid selection
        if current_reasoning and current_reasoning != 'None' and current_reasoning in available_values:
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
            self.log_debug(f"Set initial_option for GPT-5.2 reasoning: {current_reasoning}")
        else:
            # Default to 'none' if no valid selection
            if available_values:
                default_value = 'none'
                reasoning_block["accessory"]["initial_option"] = {
                    "text": {"type": "plain_text", "text": self._get_reasoning_display(default_value)},
                    "value": default_value
                }
                self.log_debug(f"No valid reasoning selection - set default initial_option: {default_value}")

        blocks.append(reasoning_block)

        # Add note about xhigh reasoning
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "_Note: Extra High reasoning provides maximum accuracy but is slower and more expensive_"
            }]
        })

        # Response Detail
        blocks.append({
            "type": "section",
            "block_id": "verbosity_block",
            "text": {
                "type": "mrkdwn",
                "text": "*Response Detail*\nControls how detailed responses are"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "verbosity",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_verbosity_display(settings.get('verbosity', config.default_verbosity))},
                    "value": settings.get('verbosity', config.default_verbosity)
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "📝 Concise"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "📄 Standard"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "📚 Detailed"}, "value": "high"}
                ]
            }
        })

        blocks.append({"type": "divider"})
        return blocks

    def _add_gpt4_settings(self, settings: Dict) -> List[Dict]:
        """Add GPT-4 specific settings blocks"""
        blocks = []
        
        # Important note about temperature vs top_p
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "⚠️ *Important:* Modify either Temperature OR Top P, not both. OpenAI recommends changing only one."
            }]
        })
        
        # Temperature
        blocks.append({
            "type": "input",
            "block_id": "temperature_block",
            "element": {
                "type": "number_input",
                "action_id": "temperature",
                "is_decimal_allowed": True,
                "min_value": "0.0",
                "max_value": "2.0",
                "initial_value": str(settings.get('temperature', 0.8))
            },
            "label": {"type": "plain_text", "text": "Temperature (0.0-2.0)"},
            "hint": {"type": "plain_text", "text": "Controls randomness. Use this OR Top P, not both. Default: 0.8"}
        })
        
        # Top P
        blocks.append({
            "type": "input",
            "block_id": "top_p_block",
            "element": {
                "type": "number_input",
                "action_id": "top_p",
                "is_decimal_allowed": True,
                "min_value": "0.0",
                "max_value": "1.0",
                "initial_value": str(settings.get('top_p', 1.0))
            },
            "label": {"type": "plain_text", "text": "Top P (0.0-1.0)"},
            "hint": {"type": "plain_text", "text": "Alternative to temperature. Keep at 1.0 if using temperature. Default: 1.0"}
        })
        
        blocks.append({"type": "divider"})
        return blocks
    
    def _add_common_settings(self, settings: Dict) -> List[Dict]:
        """Add settings common to all models"""
        blocks = []
        
        # Custom Instructions section (available for all models)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Custom Instructions*"}
        })
        
        # Ensure initial_value is always a string
        custom_instructions_value = settings.get('custom_instructions', '')
        if custom_instructions_value is None:
            custom_instructions_value = ''
        
        blocks.append({
            "type": "input",
            "block_id": "custom_instructions_block",
            "element": {
                "type": "plain_text_input",
                "action_id": "custom_instructions",
                "multiline": True,
                "initial_value": custom_instructions_value,
                "placeholder": {
                    "type": "plain_text",
                    "text": "- Be concise and use bullet points\n- Explain technical topics simply\n- Include code examples\n- Use professional tone"
                },
                "max_length": 3000
            },
            "label": {
                "type": "plain_text",
                "text": "How would you like the AI to respond? (Custom GPT Instructions)"
            },
            "hint": {
                "type": "plain_text",
                "text": "Tell the AI your preferences for tone, format, or style"
            },
            "optional": True
        })
        
        blocks.append({"type": "divider"})
        
        # Feature toggles
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Features*"}
        })

        # Check if model supports MCP (GPT-5 only)
        current_model = settings.get('model', config.gpt_model)
        is_gpt5 = current_model.startswith('gpt-5')

        # Adjust enable_mcp if model doesn't support it
        current_enable_mcp = settings.get('enable_mcp', True)
        if not is_gpt5 and current_enable_mcp:
            current_enable_mcp = False
            self.log_debug(f"Adjusted enable_mcp from True to False for display (model {current_model} doesn't support MCP)")

        # Build checkbox options for features
        feature_options = []
        initial_options = []

        # Web search
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🌐 *Web Search*\nAllow searching the web for current information\n_(Disables the \"Minimal\" reasoning option above when enabled)_"},
            "value": "web_search"
        })
        if settings.get('enable_web_search', True):
            initial_options.append(feature_options[-1])

        # Streaming
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🌊 *Streaming*\nShow responses as they're generated"},
            "value": "streaming"
        })
        if settings.get('enable_streaming', True):
            initial_options.append(feature_options[-1])

        # MCP Servers (only show if GPT-5 model)
        if is_gpt5:
            feature_options.append({
                "text": {"type": "mrkdwn", "text": "🔌 *MCP Servers*\nAccess specialized data sources"},
                "value": "mcp"
            })
            if current_enable_mcp:
                initial_options.append(feature_options[-1])

        # Build the features block
        # Use different block_id/action_id based on model to force Slack to re-render
        if is_gpt5:
            block_id = "features_block_gpt5"
            action_id = "features_with_mcp"
        else:
            block_id = "features_block_gpt4"
            action_id = "features_no_mcp"

        features_block = {
            "type": "section",
            "block_id": block_id,
            "text": {"type": "mrkdwn", "text": "Enable features:"},
            "accessory": {
                "type": "checkboxes",
                "action_id": action_id,
                "options": feature_options
            }
        }

        # Only add initial_options if we have some (Slack requires array or omitted entirely)
        if initial_options:
            features_block["accessory"]["initial_options"] = initial_options

        blocks.append(features_block)

        # Add note about MCP being unavailable if not GPT-5
        if not is_gpt5:
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": "_Note: MCP Servers feature will be available if you select a GPT-5 model_"
                }]
            })
        
        blocks.append({"type": "divider"})
        
        # Image Generation Settings
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Image Generation*"}
        })
        
        # Image size (orientation)
        blocks.append({
            "type": "section",
            "block_id": "image_size_block",
            "text": {"type": "mrkdwn", "text": "Image orientation:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_size",
                "placeholder": {"type": "plain_text", "text": "Select size"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_size_display(settings.get('image_size', '1024x1024'))},
                    "value": settings.get('image_size', '1024x1024')
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "Square 1:1"}, "value": "1024x1024"},
                    {"text": {"type": "plain_text", "text": "Portrait 2:3"}, "value": "1024x1536"},
                    {"text": {"type": "plain_text", "text": "Landscape 3:2"}, "value": "1536x1024"},
                    {"text": {"type": "plain_text", "text": "Auto"}, "value": "auto"}
                ]
            }
        })

        # Image quality
        blocks.append({
            "type": "section",
            "block_id": "image_quality_block",
            "text": {"type": "mrkdwn", "text": "Image quality:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_quality",
                "placeholder": {"type": "plain_text", "text": "Select quality"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_quality_display(settings.get('image_quality', 'auto'))},
                    "value": settings.get('image_quality', 'auto')
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "Auto"}, "value": "auto"},
                    {"text": {"type": "plain_text", "text": "Low (Faster, cheaper)"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "Medium (Balanced)"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "High (Best quality)"}, "value": "high"}
                ]
            }
        })

        # Image background
        blocks.append({
            "type": "section",
            "block_id": "image_background_block",
            "text": {"type": "mrkdwn", "text": "Image background:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_background",
                "placeholder": {"type": "plain_text", "text": "Select background"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_background_display(settings.get('image_background', 'auto'))},
                    "value": settings.get('image_background', 'auto')
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "Auto"}, "value": "auto"},
                    {"text": {"type": "plain_text", "text": "Transparent"}, "value": "transparent"},
                    {"text": {"type": "plain_text", "text": "Opaque"}, "value": "opaque"}
                ]
            }
        })

        # Input fidelity for edits
        blocks.append({
            "type": "section",
            "block_id": "input_fidelity_block",
            "text": {"type": "mrkdwn", "text": "Image edit style:"},
            "accessory": {
                "type": "radio_buttons",
                "action_id": "input_fidelity",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_fidelity_display(settings.get('input_fidelity', 'high'))},
                    "value": settings.get('input_fidelity', 'high')
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "🎨 Preserve Original Style"}, "value": "high"},
                    {"text": {"type": "plain_text", "text": "✨ Allow Reinterpretation"}, "value": "low"}
                ]
            }
        })
        
        # Vision detail level
        blocks.append({
            "type": "section",
            "block_id": "vision_detail_block",
            "text": {"type": "mrkdwn", "text": "Vision analysis detail:"},
            "accessory": {
                "type": "radio_buttons",
                "action_id": "vision_detail",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_vision_detail_display(settings.get('vision_detail', 'auto'))},
                    "value": settings.get('vision_detail', 'auto')
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "🤖 Auto"}, "value": "auto"},
                    {"text": {"type": "plain_text", "text": "🔍 Low Detail"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "🔬 High Detail"}, "value": "high"}
                ]
            }
        })
        
        return blocks
    
    def extract_form_values(self, view_state: Dict) -> Dict:
        """Extract form values from modal submission"""
        values = view_state.get('values', {})
        extracted = {}
        
        # Model selection
        model_block = values.get('model_block', {})
        if 'model_select' in model_block:
            selected = model_block['model_select'].get('selected_option')
            if selected:
                extracted['model'] = selected['value']
        
        # GPT-5/5.1/5.2 settings
        # Check all possible block_ids (different ones based on model and web search state)
        reasoning_block = (values.get('reasoning_block_no_web', {}) or
                          values.get('reasoning_block_web', {}) or
                          values.get('reasoning_block_gpt51', {}) or
                          values.get('reasoning_block_gpt52', {}))

        # Check all possible action_ids
        reasoning_found = False
        if 'reasoning_level' in reasoning_block:
            selected = reasoning_block['reasoning_level'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                # No selection - might happen during modal updates
                self.log_debug("No reasoning_level selected_option found")
        elif 'reasoning_level_no_minimal' in reasoning_block:
            selected = reasoning_block['reasoning_level_no_minimal'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                self.log_debug("No reasoning_level_no_minimal selected_option found")
        elif 'reasoning_level_gpt51' in reasoning_block:
            selected = reasoning_block['reasoning_level_gpt51'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                self.log_debug("No reasoning_level_gpt51 selected_option found")
        elif 'reasoning_level_gpt52' in reasoning_block:
            selected = reasoning_block['reasoning_level_gpt52'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                self.log_debug("No reasoning_level_gpt52 selected_option found")

        # Fallback if no reasoning selection due to Slack modal update bug
        if not reasoning_found:
            # Check if web search is enabled from the form
            features_block = values.get('features_block_gpt4', {}) or values.get('features_block_gpt5', {})
            web_search_enabled = False
            if 'features_no_mcp' in features_block:
                selected_options = features_block['features_no_mcp'].get('selected_options', [])
                selected_values = [opt['value'] for opt in selected_options]
                web_search_enabled = 'web_search' in selected_values
            elif 'features_with_mcp' in features_block:
                selected_options = features_block['features_with_mcp'].get('selected_options', [])
                selected_values = [opt['value'] for opt in selected_options]
                web_search_enabled = 'web_search' in selected_values

            # Use a safe default based on model and web search state
            model = extracted.get('model', 'gpt-5')
            if web_search_enabled:
                default_reasoning = 'low'
            else:
                # Use model-specific default when web search is off
                if model in ['gpt-5.1', 'gpt-5.2']:
                    default_reasoning = 'none'  # GPT-5.1/5.2 default
                else:
                    default_reasoning = 'minimal'  # GPT-5/GPT-5-mini default
            extracted['reasoning_effort'] = default_reasoning
            self.log_debug(f"No reasoning selection found - using default: {default_reasoning} for model {model} (web_search: {web_search_enabled})")
        
        verbosity_block = values.get('verbosity_block', {})
        if 'verbosity' in verbosity_block:
            selected = verbosity_block['verbosity'].get('selected_option')
            if selected:
                extracted['verbosity'] = selected['value']
        
        # GPT-4 settings
        temp_block = values.get('temperature_block', {})
        if 'temperature' in temp_block:
            extracted['temperature'] = float(temp_block['temperature'].get('value', 0.8))
        
        top_p_block = values.get('top_p_block', {})
        if 'top_p' in top_p_block:
            extracted['top_p'] = float(top_p_block['top_p'].get('value', 1.0))
        
        # Features
        # Check both possible block_ids (we use different ones based on model)
        features_block = values.get('features_block_gpt4', {}) or values.get('features_block_gpt5', {})

        # Check both possible action_ids
        if 'features_no_mcp' in features_block:
            selected_options = features_block['features_no_mcp'].get('selected_options', [])
            selected_values = [opt['value'] for opt in selected_options]
            extracted['enable_web_search'] = 'web_search' in selected_values
            extracted['enable_streaming'] = 'streaming' in selected_values
            # Don't set enable_mcp when it's not visible - preserve stored value
        elif 'features_with_mcp' in features_block:
            selected_options = features_block['features_with_mcp'].get('selected_options', [])
            selected_values = [opt['value'] for opt in selected_options]
            extracted['enable_web_search'] = 'web_search' in selected_values
            extracted['enable_streaming'] = 'streaming' in selected_values
            extracted['enable_mcp'] = 'mcp' in selected_values
        
        # Custom Instructions
        custom_instructions_block = values.get('custom_instructions_block', {})
        if 'custom_instructions' in custom_instructions_block:
            custom_value = custom_instructions_block['custom_instructions'].get('value')
            # Handle None (cleared field) or empty string
            if custom_value:
                custom_text = custom_value.strip()
                extracted['custom_instructions'] = custom_text if custom_text else None
            else:
                extracted['custom_instructions'] = None
        
        # Image settings
        image_size_block = values.get('image_size_block', {})
        if 'image_size' in image_size_block:
            selected = image_size_block['image_size'].get('selected_option')
            if selected:
                extracted['image_size'] = selected['value']

        image_quality_block = values.get('image_quality_block', {})
        if 'image_quality' in image_quality_block:
            selected = image_quality_block['image_quality'].get('selected_option')
            if selected:
                extracted['image_quality'] = selected['value']

        image_background_block = values.get('image_background_block', {})
        if 'image_background' in image_background_block:
            selected = image_background_block['image_background'].get('selected_option')
            if selected:
                extracted['image_background'] = selected['value']

        fidelity_block = values.get('input_fidelity_block', {})
        if 'input_fidelity' in fidelity_block:
            selected = fidelity_block['input_fidelity'].get('selected_option')
            if selected:
                extracted['input_fidelity'] = selected['value']
        
        vision_block = values.get('vision_detail_block', {})
        if 'vision_detail' in vision_block:
            selected = vision_block['vision_detail'].get('selected_option')
            if selected:
                extracted['vision_detail'] = selected['value']
        
        return extracted
    
    def validate_settings(self, settings: Dict) -> Dict:
        """Validate and adjust settings for compatibility"""
        validated = settings.copy()

        # If non-GPT-5 model selected, preserve existing MCP preference (don't force to False)
        # This allows users to switch between models without losing their MCP preference
        model = validated.get('model', config.gpt_model)
        if not model.startswith('gpt-5'):
            # For GPT-4 models, if MCP is somehow explicitly enabled, disable it
            if validated.get('enable_mcp'):
                validated['enable_mcp'] = False
                self.log_info(f"Auto-disabled MCP because {model} doesn't support it (requires GPT-5)")
            # If enable_mcp is not in validated (wasn't in the form), leave it as-is
            # Don't explicitly set to False - preserve user's preference for when they switch back to GPT-5

        # If web search enabled but reasoning too low (minimal), auto-upgrade
        if validated.get('enable_web_search') and validated.get('reasoning_effort') == 'minimal':
            validated['reasoning_effort'] = 'low'
            self.log_info("Auto-upgraded reasoning_effort from minimal to low for web search compatibility")
        
        # Check if both temperature and top_p are changed for chat models (GPT-4)
        model = validated.get('model', 'gpt-5')
        reasoning_models = ['gpt-5', 'gpt-5.1', 'gpt-5-mini', 'gpt-5.2']
        if model not in reasoning_models:
            # For chat models, warn if both temperature and top_p are changed from defaults
            default_temp = config.default_temperature
            default_top_p = config.default_top_p
            
            temp_changed = validated.get('temperature', default_temp) != default_temp
            top_p_changed = validated.get('top_p', default_top_p) != default_top_p
            
            if temp_changed and top_p_changed:
                self.log_warning(f"Both temperature ({validated.get('temperature')}) and top_p ({validated.get('top_p')}) "
                               f"were changed from defaults. OpenAI recommends using only one.")
                # Note: We don't reset either value, just warn - let user decide
        
        # Remove invalid parameters for model type
        reasoning_models = ['gpt-5', 'gpt-5.1', 'gpt-5-mini', 'gpt-5.2']
        if model in reasoning_models:
            # Remove GPT-4/chat model specific params
            validated.pop('temperature', None)
            validated.pop('top_p', None)
        else:
            # Remove GPT-5 reasoning model specific params (for GPT-4, GPT-5.2-chat-latest, etc.)
            validated.pop('reasoning_effort', None)
            validated.pop('verbosity', None)
        
        return validated
    
    # Helper methods for display names
    def _get_model_display_name(self, model: str) -> str:
        """Get user-friendly model name"""
        display_names = {
            'gpt-5.2': 'GPT-5.2',
            'gpt-5.2-pro': 'GPT-5.2 Pro',
            'gpt-5.2-chat-latest': 'GPT-5.2 Instant',
            'gpt-5': 'GPT-5',
            'gpt-5.1': 'GPT-5.1',
            'gpt-5-mini': 'GPT-5 Mini',
            'gpt-4.1': 'GPT-4.1',
            'gpt-4o': 'GPT-4o'
        }
        return display_names.get(model, model)
    
    def _get_reasoning_display(self, level: str) -> str:
        """Get display name for reasoning level"""
        displays = {
            'none': '🌟 None (Adaptive)',
            'minimal': '⚡ Minimal (Fastest, Chat-like)',
            'low': '🚀 Low (Fast)',
            'medium': '⚖️ Medium (Balanced)',
            'high': '🧠 High (Thorough)',
            'xhigh': '💎 Extra High (Maximum Quality)'
        }
        return displays.get(level, '⚖️ Medium (Balanced)')
    
    def _get_verbosity_display(self, level: str) -> str:
        """Get display name for verbosity"""
        displays = {
            'low': '📝 Concise',
            'medium': '📄 Standard',
            'high': '📚 Detailed'
        }
        return displays.get(level, '📄 Standard')
    
    def _get_image_size_display(self, size: str) -> str:
        """Get display name for image size"""
        displays = {
            '1024x1024': 'Square 1:1',
            '1024x1536': 'Portrait 2:3',
            '1536x1024': 'Landscape 3:2',
            'auto': 'Auto'
        }
        return displays.get(size, 'Square 1:1')
    
    def _get_fidelity_display(self, fidelity: str) -> str:
        """Get display name for input fidelity"""
        displays = {
            'high': '🎨 Preserve Original Style',
            'low': '✨ Allow Reinterpretation'
        }
        return displays.get(fidelity, '🎨 Preserve Original Style')
    
    def _get_vision_detail_display(self, detail: str) -> str:
        """Get display name for vision detail"""
        displays = {
            'auto': '🤖 Auto',
            'low': '🔍 Low Detail',
            'high': '🔬 High Detail'
        }
        return displays.get(detail, '🤖 Auto')

    def _get_image_quality_display(self, quality: str) -> str:
        """Get display name for image quality"""
        displays = {
            'auto': 'Auto',
            'low': 'Low (Faster, cheaper)',
            'medium': 'Medium (Balanced)',
            'high': 'High (Best quality)'
        }
        return displays.get(quality, 'Auto')

    def _get_image_background_display(self, background: str) -> str:
        """Get display name for image background"""
        displays = {
            'auto': 'Auto',
            'transparent': 'Transparent',
            'opaque': 'Opaque'
        }
        return displays.get(background, 'Auto')