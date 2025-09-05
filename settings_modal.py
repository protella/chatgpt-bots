"""
User Settings Modal for Slack Bot
Handles the interactive settings configuration interface
"""
from typing import Dict, Optional, Any, List
from config import config
from logger import LoggerMixin
import time
import json


class SettingsModal(LoggerMixin):
    """Manages the user settings modal interface"""
    
    def __init__(self, db):
        """Initialize with database connection"""
        self.db = db
        self.logger_name = "SettingsModal"
    
    def build_settings_modal(self, user_id: str, trigger_id: str, 
                            current_settings: Optional[Dict] = None,
                            is_new_user: bool = False,
                            thread_id: Optional[str] = None,
                            in_thread: bool = False) -> Dict:
        """
        Build the complete settings modal.
        
        Args:
            user_id: Slack user ID
            trigger_id: Slack trigger ID for modal
            current_settings: Current user settings
            is_new_user: Whether this is a new user's first setup
            thread_id: Thread ID if opened from within a thread
            in_thread: Whether modal was opened from within a thread
            
        Returns:
            Modal view dictionary for Slack API
        """
        if not current_settings:
            current_settings = self.db.get_user_preferences(user_id)
            if not current_settings:
                # Get user's email from users table
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = self.db.create_default_user_preferences(user_id, email)
        
        # Determine which model is selected
        selected_model = current_settings.get('model', config.gpt_model)
        
        # Build modal blocks
        blocks = self._build_modal_blocks(current_settings, selected_model, is_new_user, in_thread)
        
        # Determine callback ID based on user status
        callback_id = "welcome_settings_modal" if is_new_user else "settings_modal"
        
        # Determine if we're in dev environment
        is_dev = config.settings_slash_command.endswith("-dev")
        # Slack modal titles have a 24 character limit
        modal_title = "ChatGPT Settings (Dev)" if is_dev else "ChatGPT Bot Settings"
        
        return {
            "type": "modal",
            "callback_id": callback_id,
            "title": {"type": "plain_text", "text": modal_title},
            "submit": {"type": "plain_text", "text": "Save Settings"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": blocks,
            "private_metadata": json.dumps({
                "settings": current_settings,
                "thread_id": thread_id,
                "in_thread": in_thread
            })  # Store settings and context
        }
    
    def _build_modal_blocks(self, settings: Dict, selected_model: str, 
                           is_new_user: bool = False, in_thread: bool = False) -> List[Dict]:
        """Build the modal blocks based on current settings and model selection
        
        Args:
            settings: Current settings dictionary
            selected_model: Currently selected model
            is_new_user: Whether this is a new user
            in_thread: Whether modal was opened from within a thread
        """
        blocks = []
        
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
        else:
            # Determine header text based on scope
            if in_thread:
                header_text = "Configure Settings for This Thread"
            else:
                header_text = "Configure Your Global Settings"
            
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": header_text}
            })
            
            # Add tip right after header for better visibility
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"💡 *Tip:* For global settings, type:\n"
                            f"`{config.settings_slash_command}` in the main channel/DM (not in a thread)"
                            if in_thread else
                            f"💡 *Tip:* You can change these settings anytime by typing:\n`{config.settings_slash_command}` in the main channel/DM (not in a thread)\n\n"
                            f"For thread-specific settings:\n Hover over any bot message → Click ••• → Thread Settings"
                        )
                    }
                ]
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
                    {"text": {"type": "plain_text", "text": "GPT-5"}, "value": "gpt-5"},
                    {"text": {"type": "plain_text", "text": "GPT-5 Mini"}, "value": "gpt-5-mini"},
                    {"text": {"type": "plain_text", "text": "GPT-4.1"}, "value": "gpt-4.1"},
                    {"text": {"type": "plain_text", "text": "GPT-4o"}, "value": "gpt-4o"}
                ]
            }
        })
        
        blocks.append({"type": "divider"})
        
        # Add model-specific settings
        if selected_model in ['gpt-5', 'gpt-5-mini']:
            blocks.extend(self._add_gpt5_settings(settings))
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
        current_reasoning = settings.get('reasoning_effort', 'medium')
        self.log_debug(f"Building reasoning options - web_search: {web_search_enabled}, current: {current_reasoning}")
        
        # Force-validate the reasoning level for web search compatibility
        if web_search_enabled and current_reasoning == 'minimal':
            # If web search is on but reasoning is minimal, use low
            current_reasoning = 'low'
            self.log_debug(f"Adjusted reasoning from minimal to low for display")
        elif not current_reasoning or current_reasoning not in ['minimal', 'low', 'medium', 'high']:
            # Fallback if no valid reasoning is set
            current_reasoning = 'low' if web_search_enabled else 'medium'
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
            current_reasoning = 'low' if web_search_enabled else 'medium'
            self.log_warning(f"Current reasoning {current_reasoning} not in available options, using fallback")
        
        # Build the reasoning block
        # Use a different action_id based on whether minimal is available
        # This works around a Slack bug where selections are lost when options change
        action_id = "reasoning_level" if not web_search_enabled else "reasoning_level_no_minimal"
        
        reasoning_block = {
            "type": "section",
            "block_id": "reasoning_block",
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
        
        # Add initial_option only if we have a valid selection
        if current_reasoning and current_reasoning != 'None':
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
        else:
            # If no selection (can happen due to Slack bug when options change)
            # We'll add a note for the user
            self.log_debug("No reasoning selection - likely due to Slack option change bug")
        
        blocks.append(reasoning_block)
        
        # Add a warning if we detected the Slack bug (minimal was removed and no selection)
        if web_search_enabled and settings.get('reasoning_effort') == 'low' and not current_reasoning:
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn", 
                    "text": "⚠️ _Please select a reasoning level (Low is recommended)_"
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
                    "text": {"type": "plain_text", "text": self._get_verbosity_display(settings.get('verbosity', 'medium'))},
                    "value": settings.get('verbosity', 'medium')
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
        
        # Feature toggles
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Features*"}
        })
        
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
        
        # Build the features block
        features_block = {
            "type": "section",
            "block_id": "features_block",
            "text": {"type": "mrkdwn", "text": "Enable features:"},
            "accessory": {
                "type": "checkboxes",
                "action_id": "features",
                "options": feature_options
            }
        }
        
        # Only add initial_options if we have some (Slack requires array or omitted entirely)
        if initial_options:
            features_block["accessory"]["initial_options"] = initial_options
        
        blocks.append(features_block)
        
        blocks.append({"type": "divider"})
        
        # Image Generation Settings
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Image Generation*"}
        })
        
        # Image size
        blocks.append({
            "type": "section",
            "block_id": "image_size_block",
            "text": {"type": "mrkdwn", "text": "Default image size:"},
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
        
        # GPT-5 settings
        reasoning_block = values.get('reasoning_block', {})
        # Check both possible action_ids (we use different ones based on web search state)
        if 'reasoning_level' in reasoning_block:
            selected = reasoning_block['reasoning_level'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
            else:
                # No selection - might happen during modal updates
                self.log_debug("No reasoning_level selected_option found")
        elif 'reasoning_level_no_minimal' in reasoning_block:
            selected = reasoning_block['reasoning_level_no_minimal'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
            else:
                self.log_debug("No reasoning_level_no_minimal selected_option found")
        
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
        features_block = values.get('features_block', {})
        if 'features' in features_block:
            selected_options = features_block['features'].get('selected_options', [])
            selected_values = [opt['value'] for opt in selected_options]
            extracted['enable_web_search'] = 'web_search' in selected_values
            extracted['enable_streaming'] = 'streaming' in selected_values
        
        # Image settings
        image_size_block = values.get('image_size_block', {})
        if 'image_size' in image_size_block:
            selected = image_size_block['image_size'].get('selected_option')
            if selected:
                extracted['image_size'] = selected['value']
        
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
        
        # If web search enabled but reasoning too low (minimal), auto-upgrade
        if validated.get('enable_web_search') and validated.get('reasoning_effort') == 'minimal':
            validated['reasoning_effort'] = 'low'
            self.log_info("Auto-upgraded reasoning_effort from minimal to low for web search compatibility")
        
        # Check if both temperature and top_p are changed for GPT-4 models
        model = validated.get('model', 'gpt-5')
        if model not in ['gpt-5', 'gpt-5-mini']:
            # For GPT-4 models, warn if both temperature and top_p are changed from defaults
            default_temp = config.default_temperature
            default_top_p = config.default_top_p
            
            temp_changed = validated.get('temperature', default_temp) != default_temp
            top_p_changed = validated.get('top_p', default_top_p) != default_top_p
            
            if temp_changed and top_p_changed:
                self.log_warning(f"Both temperature ({validated.get('temperature')}) and top_p ({validated.get('top_p')}) "
                               f"were changed from defaults. OpenAI recommends using only one.")
                # Note: We don't reset either value, just warn - let user decide
        
        # Remove invalid parameters for model type
        if model in ['gpt-5', 'gpt-5-mini']:
            # Remove GPT-4 specific params
            validated.pop('temperature', None)
            validated.pop('top_p', None)
        else:
            # Remove GPT-5 specific params
            validated.pop('reasoning_effort', None) 
            validated.pop('verbosity', None)
        
        return validated
    
    # Helper methods for display names
    def _get_model_display_name(self, model: str) -> str:
        """Get user-friendly model name"""
        display_names = {
            'gpt-5': 'GPT-5',
            'gpt-5-mini': 'GPT-5 Mini',
            'gpt-4.1': 'GPT-4.1',
            'gpt-4o': 'GPT-4o'
        }
        return display_names.get(model, model)
    
    def _get_reasoning_display(self, level: str) -> str:
        """Get display name for reasoning level"""
        displays = {
            'minimal': '⚡ Minimal (Fastest, Chat-like)',
            'low': '🚀 Low (Fast)',
            'medium': '⚖️ Medium (Balanced)',
            'high': '🧠 High (Thorough, Slowest)'
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