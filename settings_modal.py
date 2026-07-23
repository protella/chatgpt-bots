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
        
        # Determine which model is selected. Coerce any stale/dropped model value
        # (e.g. an old thread override) to gpt-5.6-sol so the picker's initial_option
        # is always a valid option.
        from config import SUPPORTED_CHAT_MODELS
        selected_model = current_settings.get('model', config.gpt_model)
        if selected_model not in SUPPORTED_CHAT_MODELS:
            selected_model = 'gpt-5.6-sol'
        
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
    
    def build_channel_settings_modal(self, channel_id: str, current_settings: Optional[Dict],
                                     global_default_mode: str,
                                     channel_memories: Optional[List[Dict]] = None,
                                     memory_textarea_value: Optional[str] = None,
                                     mem_seed: Optional[List] = None) -> Dict:
        """Build the per-channel settings modal (Phase 7).

        `current_settings` is the DB row (or None). A NULL/absent response_mode means the channel
        inherits the global default; that is represented by the "inherit" option here, and the
        submission handler stores None (NULL) for it so the global default keeps applying.

        `channel_memories` (from ``get_channel_memory_async``) drives the memory sections at the
        bottom: channel-scope facts become ONE editable multiline textarea; workspace-scope facts
        render read-only below it. On a FRESH open (`memory_textarea_value`/`mem_seed` both None)
        the textarea value and the open-time seed ``[[id, hash], ...]`` are computed from the rows.
        On a RE-RENDER (model change / views_update) the caller passes both verbatim so an in-flight
        edit survives and the seed stays anchored to the rows the user first saw. `mem_seed` rides in
        `private_metadata` so the submit handler can reconcile exactly those rows. All optional so
        pure builder tests can render without a DB.
        """
        from message_processor.participation import MODE_TO_LEVEL, VALID_LEVELS

        from config import SUPPORTED_CHAT_MODELS, GPT56_EFFORTS, GPT55_EFFORTS

        cs = current_settings or {}
        directives_value = cs.get("directives") or ""

        def _select(action_id, label_map, current, inherit_label):
            """Static select with an 'inherit' first option; initial = current or inherit."""
            options = [{"text": {"type": "plain_text", "text": inherit_label}, "value": "inherit"}]
            options += [{"text": {"type": "plain_text", "text": text}, "value": value}
                        for value, text in label_map]
            selected = current if current in {v for v, _ in label_map} else "inherit"
            initial = next(o for o in options if o["value"] == selected)
            return {"type": "static_select", "action_id": action_id,
                    "options": options, "initial_option": initial}

        model_element = _select(
            "channel_model",
            [(m, m) for m in SUPPORTED_CHAT_MODELS],
            cs.get("model"),
            f"Use each person's own setting (default: {config.gpt_model})",
        )
        # Effort ladder follows the selected channel model (gpt-5.5 has no `max`).
        # 'Inherit' shows the full 5.6 ladder — the effective model varies per asker
        # and compose-time clamping adjusts anything a model doesn't support.
        selected_channel_model = cs.get("model") or ""
        effort_ladder = GPT55_EFFORTS if selected_channel_model.startswith("gpt-5.5") else GPT56_EFFORTS
        effort_element = _select(
            "channel_reasoning_effort",
            [(e, e) for e in effort_ladder],
            cs.get("reasoning_effort"),
            "Use each person's own setting",
        )
        verbosity_element = _select(
            "channel_verbosity",
            [(v, v) for v in ("low", "medium", "high")],
            cs.get("verbosity"),
            "Use each person's own setting",
        )

        # Phase F: one Participation select replaces the old response-mode select.
        # Legacy rows with only response_mode map cleanly (off≡off, tag_only≡mentions_only,
        # auto_respond≡judicious); submission writes BOTH columns in lockstep.
        global_default_level = MODE_TO_LEVEL.get((global_default_mode or "tag_only").lower(), "mentions_only")
        mode_options = [
            {"text": {"type": "plain_text", "text": f"Use default (inherit — currently: {global_default_level})"},
             "value": "inherit"},
            {"text": {"type": "plain_text", "text": "Mentions only — reply only when clearly addressed"},
             "value": "mentions_only"},
            {"text": {"type": "plain_text", "text": "Judicious — chime in when clearly valuable (recommended)"},
             "value": "judicious"},
            {"text": {"type": "plain_text", "text": "Active — participate more freely (higher reply cap)"},
             "value": "active"},
            {"text": {"type": "plain_text", "text": "Off — never respond here, even when @mentioned"},
             "value": "off"},
        ]
        current_level = cs.get("participation_level")
        if current_level not in VALID_LEVELS:
            # Fall back to the legacy column when only response_mode was ever set.
            current_level = MODE_TO_LEVEL.get(cs.get("response_mode") or "", None)
        selected_value = current_level if current_level in VALID_LEVELS else "inherit"
        initial_mode_option = next(o for o in mode_options if o["value"] == selected_value)

        # Reply placement is a TRI-STATE control (SHOULD-FIX #5): a stored None means "inherit the
        # workspace default", True means "reply at channel level", False means "threads only". The
        # old binary checkbox resolved NULL to today's global default, so merely opening + saving an
        # inheriting channel FROZE that default into an explicit row. These three options map
        # straight back to None / True / False on submit, so an untouched inheriting channel stays
        # NULL (still inheriting) and future global-config changes keep flowing through.
        ric_value = cs.get("reply_in_channel")  # None (inherit) | True | False
        default_placement_text = ("reply at channel level" if config.reply_in_channel_default
                                  else "threads only")
        placement_options = [
            {"text": {"type": "plain_text",
                      "text": f"Inherit workspace default (currently: {default_placement_text})"},
             "value": "inherit"},
            {"text": {"type": "plain_text", "text": "Reply at channel level"}, "value": "channel"},
            {"text": {"type": "plain_text", "text": "Threads only"}, "value": "threads"},
        ]
        if ric_value is None:
            placement_selected = "inherit"
        elif ric_value:
            placement_selected = "channel"
        else:
            placement_selected = "threads"
        reply_element = {
            "type": "static_select", "action_id": "reply_in_channel",
            "options": placement_options,
            "initial_option": next(o for o in placement_options if o["value"] == placement_selected),
        }

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn",
             "text": f"*Channel settings* for <#{channel_id}>\nHow I participate in this channel. "
                     f"Global defaults come from the bot's configuration; these override them here."},
             "accessory": {"type": "button", "action_id": "open_user_settings_push",
                           "text": {"type": "plain_text", "text": "👤 My personal settings"}}},
            {"type": "input", "block_id": "participation_block",
             "element": {"type": "static_select", "action_id": "participation_level",
                         "options": mode_options, "initial_option": initial_mode_option},
             "label": {"type": "plain_text", "text": "Participation"},
             "hint": {"type": "plain_text",
                      "text": f"How proactively I join conversations. 'Inherit' uses the global default ({global_default_level})."}},
            {"type": "input", "block_id": "directives_block", "optional": True,
             "element": {"type": "plain_text_input", "action_id": "directives", "multiline": True,
                         "initial_value": directives_value, "max_length": 1000,
                         "placeholder": {"type": "plain_text",
                                         "text": "e.g. Only jump in on deploy failures; otherwise stay quiet."}},
             "label": {"type": "plain_text", "text": "Channel ground rules"},
             "hint": {"type": "plain_text", "text": "Extra instructions for how I behave in this channel."}},
            {"type": "input", "block_id": "reply_in_channel_block", "optional": True,
             "element": reply_element,
             "label": {"type": "plain_text", "text": "Reply placement"},
             "hint": {"type": "plain_text",
                      "text": "'Inherit' follows the workspace default. 'Reply at channel level' lets me answer a top-level message in the channel — I still judge per message whether a thread fits better. 'Threads only' always routes replies into a thread."}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn",
             "text": "*Shared response settings* — apply to everyone in this channel. "
                     "Anything left on \"each person's own setting\" falls back to the "
                     "asker's personal preferences."}},
            {"type": "input", "block_id": "channel_model_block", "dispatch_action": True,
             "element": model_element,
             "label": {"type": "plain_text", "text": "Model"}},
            {"type": "input", "block_id": "channel_effort_block",
             "element": effort_element,
             "label": {"type": "plain_text", "text": "Reasoning effort"},
             "hint": {"type": "plain_text",
                      "text": "Efforts a model doesn't support are adjusted automatically (e.g. max → xhigh on gpt-5.5)."}},
            {"type": "input", "block_id": "channel_verbosity_block",
             "element": verbosity_element,
             "label": {"type": "plain_text", "text": "Verbosity"}},
        ]

        # Channel-memory editor + read-only workspace-shared list. On a fresh open we derive the
        # textarea value and the open-time seed from the DB rows; on a re-render the caller hands both
        # back verbatim so an in-flight edit survives. The seed lists EXACTLY the rows shown in the box
        # and rides in private_metadata so submit reconciles only what the user could actually see.
        memories = channel_memories or []
        channel_rows = [m for m in memories if m.get("scope") == "channel"]
        workspace_rows = [m for m in memories if m.get("scope") != "channel"]

        if memory_textarea_value is None and mem_seed is None:
            memory_textarea_value, mem_seed, hidden_count = self._compute_channel_memory_seed(channel_rows)
        else:
            mem_seed = mem_seed or []
            memory_textarea_value = memory_textarea_value or ""
            # Seed is carried verbatim on re-render; derive "+N more" from what it omits (normalize
            # returns "" for whitespace-only, so a bare strip test matches the fresh-open blank drop).
            non_blank = sum(1 for m in channel_rows if (m.get("content") or "").strip())
            hidden_count = max(0, non_blank - len(mem_seed))

        blocks.append({"type": "divider"})
        blocks.extend(self._build_channel_memory_blocks(
            memory_textarea_value, hidden_count, workspace_rows))

        return {
            "type": "modal",
            "callback_id": "channel_settings_modal",
            "title": {"type": "plain_text", "text": "Channel Settings"},
            "submit": {"type": "plain_text", "text": "Save"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "private_metadata": json.dumps({"channel_id": channel_id, "mem_seed": mem_seed}),
            "blocks": blocks,
        }

    # Read-only workspace-shared memories are one block per item; cap the list so a workspace with a
    # lot of shared facts can't blow Slack's 100-block modal limit. Channel-scope memory is a single
    # textarea now, so it needs no per-item cap — only the 2900-char textarea budget below.
    _MODAL_LIST_CAP = 10

    # Slack plain_text_input's max we build the channel-memory textarea against (value + budget guard).
    _MEMORY_TEXTAREA_MAX = 2900

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Trim to `limit` chars with an ellipsis so long facts/reasons stay on one line."""
        text = (text or "").strip()
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def _compute_channel_memory_seed(self, channel_rows: List[Dict]):
        """From channel-scope rows (oldest-first, as ``get_channel_memory_async`` returns them),
        build the textarea `initial_value` and the open-time seed ``[[id, hash], ...]``.

        Each row is normalized and blank rows are dropped. Rows are included oldest-first only while
        the joined value stays within the textarea budget; on the first row that would overflow we
        stop and the rest become the "+N more not shown" remainder. The seed lists EXACTLY the
        included rows — never seed a row that isn't in the box, or submit could "delete" a row the
        user never saw. Returns ``(initial_value, mem_seed, hidden_count)``.
        """
        from database import normalize_memory_line, memory_content_hash

        normed = [(m.get("id"), normalize_memory_line(m.get("content") or "")) for m in channel_rows]
        normed = [(mid, content) for mid, content in normed if content]  # drop blanks

        included: List[str] = []
        seed: List = []
        used = 0
        for mid, content in normed:
            addition = len(content) + (1 if included else 0)  # +1 for the joining newline
            if used + addition > self._MEMORY_TEXTAREA_MAX:
                break
            included.append(content)
            seed.append([mid, memory_content_hash(content)])
            used += addition

        return "\n".join(included), seed, len(normed) - len(included)

    def _build_channel_memory_blocks(self, textarea_value: str, hidden_count: int,
                                     workspace_rows: List[Dict]) -> List[Dict]:
        """Blocks for the channel-memory editor plus the read-only workspace-shared list.

        Channel-scope memory is ONE multiline textarea (`block_id="channel_memory_block"`,
        `action_id="channel_memory"`): edit or delete lines and Save to reconcile, blank it out to
        forget everything. Workspace-scope facts are visible context but READ-ONLY from a channel
        (see `message_processor/memory_tools.py` `_visible_row`), so they render without any control.
        """
        memory_input: Dict = {
            "type": "plain_text_input", "action_id": "channel_memory",
            "multiline": True, "max_length": self._MEMORY_TEXTAREA_MAX,
            "placeholder": {"type": "plain_text",
                            "text": "e.g. Deploys go out Thursdays — ping @oncall before merging."},
        }
        # Slack rejects an empty initial_value, so only set it when there's something to seed.
        if textarea_value:
            memory_input["initial_value"] = textarea_value

        blocks: List[Dict] = [
            {"type": "section",
             "text": {"type": "mrkdwn", "text": "*What I remember about this channel*"}},
            {"type": "input", "block_id": "channel_memory_block", "optional": True,
             "element": memory_input,
             "label": {"type": "plain_text", "text": "Channel memory"},
             "hint": {"type": "plain_text",
                      "text": "One note per line. Edit or delete lines and Save; "
                              "blank it out to forget everything here."}},
        ]
        if hidden_count > 0:
            blocks.append({"type": "context", "elements": [
                {"type": "mrkdwn", "text": f"_+{hidden_count} more not shown_"}]})

        # Workspace-scope memories: shown for context, but managed elsewhere — no edit control.
        if workspace_rows:
            cap = self._MODAL_LIST_CAP
            shown = workspace_rows[:cap]
            lines = "\n".join(f"• {self._truncate(m.get('content') or '', 200) or '(empty)'}"
                              for m in shown)
            if len(workspace_rows) > cap:
                lines += f"\n_+{len(workspace_rows) - cap} more_"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*Workspace-shared memories* (read-only here)\n{lines}"},
            })

        return blocks

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
            # Radio buttons render inline (no floating dropdown overlay, which Slack
            # clips at the modal edge) and scale fine for a short model list.
            "accessory": {
                "type": "radio_buttons",
                "action_id": "model_select",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_model_display_name(selected_model)},
                    "value": selected_model
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "GPT-5.6 Sol (Flagship)"}, "value": "gpt-5.6-sol"},
                    {"text": {"type": "plain_text", "text": "GPT-5.6 Terra (Balanced)"}, "value": "gpt-5.6-terra"},
                    {"text": {"type": "plain_text", "text": "GPT-5.6 Luna (Fast)"}, "value": "gpt-5.6-luna"},
                    {"text": {"type": "plain_text", "text": "GPT-5.5"}, "value": "gpt-5.5"}
                ]
            }
        })

        blocks.append({"type": "divider"})

        # Model-specific settings (reasoning ladder differs: 5.6 family adds `max`)
        blocks.extend(self._add_gpt55_settings(settings, selected_model))
        
        # Add common settings (features and image settings)
        blocks.extend(self._add_common_settings(settings))
        
        return blocks
    
    def _add_gpt55_settings(self, settings: Dict, selected_model: str = 'gpt-5.6-sol') -> List[Dict]:
        """Add model-specific settings blocks (reasoning ladder, temp/top_p when reasoning=none).

        The 5.6 family offers the full ladder incl. `max` (verified live on all three
        tiers); gpt-5.5 tops out at `xhigh`."""
        blocks = []

        # Check if web search is enabled
        if 'enable_web_search' in settings:
            web_search_enabled = bool(settings['enable_web_search'])
        else:
            web_search_enabled = True
        self.log_debug(f"Settings passed to _add_gpt55_settings: enable_web_search={settings.get('enable_web_search')}, evaluated as {web_search_enabled}")

        from config import config, clamp_effort, GPT56_EFFORTS, GPT55_EFFORTS
        current_reasoning = settings.get('reasoning_effort', 'none')
        self.log_debug(f"Building reasoning options for {selected_model}, current: {current_reasoning}")

        # Build options list per model family
        effort_values = GPT56_EFFORTS if selected_model.startswith('gpt-5.6') else GPT55_EFFORTS
        reasoning_options = [
            {"text": {"type": "plain_text", "text": self._get_reasoning_display(v)}, "value": v}
            for v in effort_values
        ]

        available_values = [opt['value'] for opt in reasoning_options]
        self.log_debug(f"Reasoning options available for {selected_model}: {available_values}, initial: {current_reasoning}")

        # Clamp stale stored values per model rules (e.g. legacy `minimal`, or `max`
        # carried over after switching 5.6 -> 5.5) instead of blindly resetting
        if current_reasoning not in available_values:
            old_reasoning = current_reasoning
            current_reasoning = clamp_effort(selected_model, current_reasoning)
            if current_reasoning not in available_values:
                current_reasoning = 'none'
            self.log_warning(f"Current reasoning '{old_reasoning}' not in available options, clamped to '{current_reasoning}'")

        # Build the reasoning block
        reasoning_block = {
            "type": "section",
            "block_id": "reasoning_block_gpt54",
            "text": {
                "type": "mrkdwn",
                "text": "*Reasoning Level*\nControls depth of analysis and problem-solving"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "reasoning_level_gpt54",
                "options": reasoning_options
            }
        }

        # Add initial_option if we have a valid selection
        if current_reasoning and current_reasoning != 'None' and current_reasoning in available_values:
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
            self.log_debug(f"Set initial_option for reasoning: {current_reasoning}")
        else:
            if available_values:
                default_value = 'none'
                reasoning_block["accessory"]["initial_option"] = {
                    "text": {"type": "plain_text", "text": self._get_reasoning_display(default_value)},
                    "value": default_value
                }
                self.log_debug(f"No valid reasoning selection - set default initial_option: {default_value}")

        blocks.append(reasoning_block)

        # Add note about xhigh reasoning and temperature availability
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "_Extra High reasoning provides maximum accuracy but is slower and more expensive. Temperature/Top P controls are available when reasoning is set to None._"
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

        # Temperature and Top P - only visible when reasoning=none
        if current_reasoning == 'none':
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": "⚠️ *Important:* Modify either Temperature OR Top P, not both. OpenAI recommends changing only one."
                }]
            })

            blocks.append({
                "type": "input",
                "block_id": "temperature_block",
                "element": {
                    "type": "number_input",
                    "action_id": "temperature",
                    "is_decimal_allowed": True,
                    "min_value": "0.0",
                    "max_value": "2.0",
                    "initial_value": str(settings.get('temperature', 1.0))
                },
                "label": {"type": "plain_text", "text": "Temperature (0.0-2.0)"},
                "hint": {"type": "plain_text", "text": "Controls randomness. Use this OR Top P, not both. Default: 1.0"}
            })

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

        # All supported models (gpt-5.5) support MCP; the old GPT-4-era
        # hide-MCP branch is gone with the pre-5.5 lineup.
        current_enable_mcp = settings.get('enable_mcp', True)

        # Build checkbox options for features
        feature_options = []
        initial_options = []

        # Web search
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🌐 *Web Search*\nAllow searching the web for current information"},
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

        # MCP Servers
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🔌 *MCP Servers*\nAccess specialized data sources"},
            "value": "mcp"
        })
        if current_enable_mcp:
            initial_options.append(feature_options[-1])

        # Build the features block (ids kept stable for the registered action handler)
        block_id = "features_block_gpt5"
        action_id = "features_with_mcp"

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

        blocks.append({"type": "divider"})

        # Image Generation Settings
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Image Generation*"}
        })

        # Image model — coerce a stale stored/config value (e.g. a retired gpt-image-1-mini)
        # to a live option so views.open doesn't reject the block. Also feeds is_image_v2 below.
        image_model_choices = {'gpt-image-2', 'gpt-image-1'}
        selected_image_model = self._coerce_choice(
            settings.get('image_model', config.image_model), image_model_choices,
            config.image_model if config.image_model in image_model_choices else 'gpt-image-2')
        blocks.append({
            "type": "section",
            "block_id": "image_model_block",
            "text": {"type": "mrkdwn", "text": "Image model:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_model",
                "placeholder": {"type": "plain_text", "text": "Select image model"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_model_display_name(selected_image_model)},
                    "value": selected_image_model
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "GPT Image 2"}, "value": "gpt-image-2"},
                    {"text": {"type": "plain_text", "text": "GPT Image 1"}, "value": "gpt-image-1"}
                ]
            }
        })

        # Image size (orientation)
        selected_image_size = self._coerce_choice(
            settings.get('image_size', '1024x1024'),
            {'1024x1024', '1024x1536', '1536x1024', 'auto'}, '1024x1024')
        blocks.append({
            "type": "section",
            "block_id": "image_size_block",
            "text": {"type": "mrkdwn", "text": "Image orientation:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_size",
                "placeholder": {"type": "plain_text", "text": "Select size"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_size_display(selected_image_size)},
                    "value": selected_image_size
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
        selected_image_quality = self._coerce_choice(
            settings.get('image_quality', 'auto'), {'auto', 'low', 'medium', 'high'}, 'auto')
        blocks.append({
            "type": "section",
            "block_id": "image_quality_block",
            "text": {"type": "mrkdwn", "text": "Image quality:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_quality",
                "placeholder": {"type": "plain_text", "text": "Select quality"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_quality_display(selected_image_quality)},
                    "value": selected_image_quality
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "Auto"}, "value": "auto"},
                    {"text": {"type": "plain_text", "text": "Low (Faster, cheaper)"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "Medium (Balanced)"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "High (Best quality)"}, "value": "high"}
                ]
            }
        })

        # Image background — filter unsupported options based on image model
        # gpt-image-2 does not support transparent backgrounds
        is_image_v2 = selected_image_model.startswith('gpt-image-2')
        background_options = [
            {"text": {"type": "plain_text", "text": "Auto"}, "value": "auto"},
            {"text": {"type": "plain_text", "text": "Opaque"}, "value": "opaque"},
        ]
        if not is_image_v2:
            background_options.insert(1, {"text": {"type": "plain_text", "text": "Transparent"}, "value": "transparent"})

        # Coerce the saved value against the visible options: 'transparent' vanishes under v2, and
        # any fully-stale value falls back to 'auto', so initial_option always matches an option.
        saved_background = self._coerce_choice(
            settings.get('image_background', 'auto'),
            {opt['value'] for opt in background_options}, 'auto')

        blocks.append({
            "type": "section",
            "block_id": "image_background_block",
            "text": {"type": "mrkdwn", "text": "Image background:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_background",
                "placeholder": {"type": "plain_text", "text": "Select background"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_background_display(saved_background)},
                    "value": saved_background
                },
                "options": background_options
            }
        })

        # Input fidelity for edits — hidden on gpt-image-2 (model auto-handles fidelity)
        if not is_image_v2:
            selected_input_fidelity = self._coerce_choice(
                settings.get('input_fidelity', 'high'), {'high', 'low'}, 'high')
            blocks.append({
                "type": "section",
                "block_id": "input_fidelity_block",
                "text": {"type": "mrkdwn", "text": "Image edit style:"},
                "accessory": {
                    "type": "radio_buttons",
                    "action_id": "input_fidelity",
                    "initial_option": {
                        "text": {"type": "plain_text", "text": self._get_fidelity_display(selected_input_fidelity)},
                        "value": selected_input_fidelity
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "🎨 Preserve Original Style"}, "value": "high"},
                        {"text": {"type": "plain_text", "text": "✨ Allow Reinterpretation"}, "value": "low"}
                    ]
                }
            })
        
        blocks.append({"type": "divider"})

        # Vision detail level
        selected_vision_detail = self._coerce_choice(
            settings.get('vision_detail', 'auto'), {'auto', 'low', 'high'}, 'auto')
        blocks.append({
            "type": "section",
            "block_id": "vision_detail_block",
            "text": {"type": "mrkdwn", "text": "Vision analysis detail:"},
            "accessory": {
                "type": "radio_buttons",
                "action_id": "vision_detail",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_vision_detail_display(selected_vision_detail)},
                    "value": selected_vision_detail
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
        
        # Reasoning effort (block/action ids kept stable across model families)
        reasoning_block = values.get('reasoning_block_gpt54', {})
        reasoning_found = False
        if 'reasoning_level_gpt54' in reasoning_block:
            selected = reasoning_block['reasoning_level_gpt54'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                # No selection - might happen during modal updates
                self.log_debug("No reasoning_level_gpt54 selected_option found")

        # Fallback if no reasoning selection due to Slack modal update bug
        if not reasoning_found:
            extracted['reasoning_effort'] = 'none'
            self.log_debug("No reasoning selection found - using default: none")
        
        verbosity_block = values.get('verbosity_block', {})
        if 'verbosity' in verbosity_block:
            selected = verbosity_block['verbosity'].get('selected_option')
            if selected:
                extracted['verbosity'] = selected['value']
        
        # Temperature / Top P (only present in the form when reasoning=none)
        temp_block = values.get('temperature_block', {})
        if 'temperature' in temp_block:
            extracted['temperature'] = float(temp_block['temperature'].get('value', 0.8))
        
        top_p_block = values.get('top_p_block', {})
        if 'top_p' in top_p_block:
            extracted['top_p'] = float(top_p_block['top_p'].get('value', 1.0))
        
        # Features
        features_block = values.get('features_block_gpt5', {})
        if 'features_with_mcp' in features_block:
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

        image_model_block = values.get('image_model_block', {})
        if 'image_model' in image_model_block:
            selected = image_model_block['image_model'].get('selected_option')
            if selected:
                extracted['image_model'] = selected['value']

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

        model = validated.get('model', config.gpt_model)

        # Clamp legacy/incompatible efforts per model (5.6 rejects `minimal`; 5.5 has no `max`)
        from config import clamp_effort
        if validated.get('reasoning_effort'):
            clamped = clamp_effort(model, validated['reasoning_effort'])
            if clamped != validated['reasoning_effort']:
                self.log_info(f"Clamped reasoning_effort {validated['reasoning_effort']} -> {clamped} for {model}")
                validated['reasoning_effort'] = clamped

        # Check if both temperature and top_p are changed for models that support them
        # (gpt-5.5 and the 5.6 family support temp/top_p only with reasoning=none)
        supports_temp = ((model.startswith('gpt-5.5') or model.startswith('gpt-5.6'))
                         and validated.get('reasoning_effort') == 'none')

        if supports_temp:
            default_temp = config.default_temperature
            default_top_p = config.default_top_p

            temp_changed = validated.get('temperature', default_temp) != default_temp
            top_p_changed = validated.get('top_p', default_top_p) != default_top_p

            if temp_changed and top_p_changed:
                self.log_warning(f"Both temperature ({validated.get('temperature')}) and top_p ({validated.get('top_p')}) "
                               f"were changed from defaults. OpenAI recommends using only one.")

        # Remove invalid parameters: reasoning models only take temp/top_p with reasoning=none
        if not supports_temp:
            validated.pop('temperature', None)
            validated.pop('top_p', None)

        return validated
    
    # Helper methods for display names
    def _get_model_display_name(self, model: str) -> str:
        """Get user-friendly model name"""
        display_names = {
            'gpt-5.6-sol': 'GPT-5.6 Sol (Flagship)',
            'gpt-5.6-terra': 'GPT-5.6 Terra (Balanced)',
            'gpt-5.6-luna': 'GPT-5.6 Luna (Fast)',
            'gpt-5.5': 'GPT-5.5',
        }
        return display_names.get(model, model)
    
    def _get_reasoning_display(self, level: str) -> str:
        """Get display name for reasoning level"""
        displays = {
            'none': '🌟 None (Adaptive)',
            'low': '🚀 Low (Fast)',
            'medium': '⚖️ Medium (Balanced)',
            'high': '🧠 High (Thorough)',
            'xhigh': '💎 Extra High (Maximum Quality)',
            'max': '🚀💎 Max (Deepest Reasoning, Slowest)'
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
    
    @staticmethod
    def _coerce_choice(value, valid, default):
        """Return `value` only if it is one of `valid`, else `default`.

        Slack rejects a whole static_select/radio block when its initial_option value is not
        also present in `options` (invalid_arguments on views.open). A stored value can drift
        out of range when an option is dropped (e.g. a retired gpt-image-1-mini default), so
        every stored select value is coerced against its live option list before it is rendered.
        """
        return value if value in valid else default

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

    def _get_image_model_display_name(self, model: str) -> str:
        """Get user-friendly image model name"""
        displays = {
            'gpt-image-2': 'GPT Image 2',
            'gpt-image-1': 'GPT Image 1',
            'gpt-image-1-mini': 'GPT Image 1 Mini',
        }
        return displays.get(model, model)