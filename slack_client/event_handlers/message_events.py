from __future__ import annotations

import json
from typing import Any, Dict

from slack_sdk.errors import SlackApiError

from base_client import Message
from config import config


class SlackMessageEventsMixin:
    async def _event_to_message(self, event: Dict[str, Any], client) -> Message:
        """Convert a Slack event into the universal Message format (no side effects).

        Shared by the mention/DM path (_handle_slack_message) and the channel-listening
        path (_handle_channel_message)."""
        # Extract text; note whether the bot itself was @-mentioned BEFORE we strip mentions
        # (used by channel-listening logic), then resolve mentions for the model.
        text = event.get("text", "")
        mentioned_self = False
        bot_user_id = getattr(self, "bot_user_id", None)
        if bot_user_id:
            from slack_client.formatting.text import text_mentions_user
            mentioned_self = text_mentions_user(text, bot_user_id)
        text = self._clean_mentions(text)

        # Process attachments (files)
        attachments = []
        files = event.get("files", [])
        for file in files:
            mimetype = file.get("mimetype", "")
            # Determine file type based on mimetype
            file_type = "image" if mimetype.startswith("image/") else "file"

            attachments.append({
                "type": file_type,
                "url": file.get("url_private"),
                "id": file.get("id"),
                "name": file.get("name"),
                "mimetype": mimetype
            })

        # Get username and timezone for logging
        user_id = event.get("user")
        username = await self.get_username(user_id, client) if user_id else "unknown"
        user_timezone = await self.get_user_timezone(user_id, client) if user_id else "UTC"

        # Get timezone label (EST, PST, etc.), real name, and email if available
        user_tz_label = None
        user_real_name = None
        user_email = None
        if user_id in self.user_cache:
            user_tz_label = self.user_cache[user_id].get('tz_label')
            user_real_name = self.user_cache[user_id].get('real_name')
            user_email = self.user_cache[user_id].get('email')
            self.log_debug(f"User cache for {user_id}: email={user_email}, real_name={user_real_name}")
        else:
            # Try to get from database if not in cache
            user_info = await self.db.get_user_info_async(user_id)
            if user_info:
                user_real_name = user_info.get('real_name')
                user_email = user_info.get('email')
                user_tz_label = user_info.get('tz_label')
                self.log_debug(f"User from DB for {user_id}: email={user_email}, real_name={user_real_name}")

        # Create universal message
        message = Message(
            text=text,
            user_id=user_id,
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=attachments,
            metadata={
                "ts": event.get("ts"),
                "mentioned_self": mentioned_self,  # was the bot @-mentioned in the raw text
                "slack_client": client,
                "username": username,  # Add username to metadata
                "user_real_name": user_real_name,  # Add real name to metadata
                "user_email": user_email,  # Add email to metadata
                "user_timezone": user_timezone,  # Add timezone to metadata
                "user_tz_label": user_tz_label  # Add timezone label (EST, PST, etc.)
            }
        )
        return message

    async def _get_channel_settings(self, channel_id: str):
        """Phase 7: fetch the per-channel settings row (or None). Best-effort; DMs have none."""
        if not channel_id or channel_id.startswith("D"):
            return None
        try:
            return await self.db.get_channel_settings_async(channel_id)
        except Exception as e:
            self.log_debug(f"_get_channel_settings failed: {e}")
            return None

    def _resolve_mode(self, cs) -> str:
        """Per-channel response_mode if set, else the global default."""
        mode = (cs or {}).get("response_mode") or getattr(config, "channel_response_mode", "tag_only")
        return (mode or "tag_only").strip().lower()

    async def _get_channel_response_mode(self, channel_id: str) -> str:
        """Resolve the response mode for a channel: per-channel DB override, else global default."""
        return self._resolve_mode(await self._get_channel_settings(channel_id))

    def _text_mentions_bot_name(self, text: str) -> bool:
        """True if one of the bot's name aliases appears as a whole word (case-insensitive)."""
        if not text:
            return False
        import re
        for alias in getattr(config, "bot_name_aliases", []) or []:
            if alias and re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
                return True
        return False

    async def _thread_participation(self, channel_id: str, thread_ts: str):
        """Best-effort (bot_present, distinct_human_count) for an existing thread.

        Lets an untagged reply count as 'for us' only in a 1:1 thread (bot + one human);
        in a multi-party thread it's for us only if explicitly addressed. On error → (False, 0)."""
        try:
            result = await self.app.client.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=50
            )
            msgs = result.get("messages", [])
        except Exception as e:
            self.log_debug(f"_thread_participation failed: {e}")
            return (False, 0)
        bot_present = False
        humans = set()
        for m in msgs:
            if self.is_own_message(m):
                bot_present = True
            elif self.classify_sender(m) == "human":
                uid = m.get("user")
                if uid:
                    humans.add(uid)
        return (bot_present, len(humans))

    async def _handle_channel_message(self, event: Dict[str, Any], client):
        """Phase 5: decide whether to respond to a NON-mention channel message, then dispatch.

        SAFE BY DEFAULT — the caller already gated on config.enable_channel_listening. Honors
        channel_response_mode (default 'tag_only'); short-circuits our own posts; de-dups against
        the app_mention event; and bypasses the welcome/settings onboarding flow entirely."""
        # Ignore non-real messages (edits, deletes, joins, message_changed, etc.)
        if event.get("subtype"):
            return
        # Loop guard FIRST: never act on our own posts.
        if self.is_own_message(event):
            return

        channel_id = event.get("channel")
        cs = await self._get_channel_settings(channel_id)
        mode = self._resolve_mode(cs)
        if mode == "off":
            return

        text = event.get("text", "") or ""

        # Dedup: an explicit @mention is already delivered via the app_mention event — skip here.
        bot_user_id = getattr(self, "bot_user_id", None)
        if bot_user_id:
            from slack_client.formatting.text import text_mentions_user
            if text_mentions_user(text, bot_user_id):
                return

        # Addressed = bot's name appears.
        addressed = self._text_mentions_bot_name(text)

        # Thread replies: an untagged reply is for us if it's a 1:1 thread, or our name appears.
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        if thread_ts and thread_ts != ts:
            bot_present, human_count = await self._thread_participation(channel_id, thread_ts)
            if bot_present and (human_count <= 1 or addressed):
                addressed = True

        # Decide.
        wake_classify = False
        if addressed:
            pass  # respond directly
        elif mode == "auto_respond":
            wake_classify = True  # let the wake classifier (in handle_message) make the call
        else:  # tag_only and not addressed
            return

        # Build the universal message (no onboarding side effects) and dispatch.
        message = await self._event_to_message(event, client)
        # Phase 6: reply in-thread by default (a top-level message keys as its own length-1 thread).
        message.thread_id = thread_ts or ts
        message.metadata["channel_listen"] = True
        message.metadata["channel_mode"] = mode
        if wake_classify:
            message.metadata["wake_classify"] = True
        # Phase 7: carry per-channel ground rules + placement into the response pipeline.
        if cs:
            if cs.get("directives"):
                message.metadata["channel_directives"] = cs["directives"]
            if cs.get("reply_in_channel"):
                message.metadata["reply_in_channel"] = True

        self.log_debug(
            f"Channel message dispatch: channel={channel_id}, ts={ts}, mode={mode}, "
            f"addressed={addressed}, wake_classify={wake_classify}"
        )
        if self.message_handler:
            await self.message_handler(message, self)

    async def _handle_slack_message(self, event: Dict[str, Any], client):
        """Handle a mention/DM event: build the message, run onboarding, dispatch (unchanged)."""

        # Skip message_changed events
        if event.get("subtype") == "message_changed":
            return

        message = await self._event_to_message(event, client)
        user_id = event.get("user")

        # Phase 7: surface per-channel ground rules (in-channel only) and skip the
        # settings-modal onboarding for BOT senders — a bot can't click the modal
        # (this is the bug where the bot told Claude "configure your settings").
        sender_type = self.classify_sender(event)
        if sender_type == "self":
            return  # loop guard (also guarded upstream for DMs)
        if message.channel_id and not message.channel_id.startswith("D"):
            cs = await self._get_channel_settings(message.channel_id)
            if cs and cs.get("directives"):
                message.metadata["channel_directives"] = cs["directives"]
        if sender_type == "other_bot":
            if self.message_handler:
                await self.message_handler(message, self)
            return

        # Assistant surface: title the split-view thread from the first user message
        # (best-effort; harmless no-op for classic DM threads and when the flag is off).
        if message.channel_id and message.channel_id.startswith("D"):
            await self._maybe_set_assistant_thread_title(
                message.channel_id, message.thread_id, message.text
            )

        # Check if this is a new user (for auto-modal trigger)
        user_prefs = await self.db.get_user_preferences_async(user_id)
        
        if not user_prefs:
            # Create default preferences for new user
            user_data = await self.db.get_or_create_user_async(user_id)
            email = user_data.get('email') if user_data else None
            user_prefs = await self.db.create_default_user_preferences_async(user_id, email)
            self.log_info(f"Created default preferences for new user {user_id}")
        
        # Check if user has completed settings
        if not user_prefs.get('settings_completed', False):
            # User hasn't completed settings - check if we've already sent welcome
            if not hasattr(self, '_welcomed_users'):
                self._welcomed_users = set()
            
            # Check if this is their first message this session
            is_first_message = user_id not in self._welcomed_users
            
            if is_first_message:
                # Mark as welcomed and send welcome button
                self._welcomed_users.add(user_id)
                
            # Check if we have a trigger_id for modal
            trigger_id = event.get('trigger_id')
            
            if trigger_id and is_first_message:
                # Create default preferences
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                default_prefs = await self.db.create_default_user_preferences_async(user_id, email)
                
                # Open welcome modal
                try:
                    modal = self.settings_modal.build_settings_modal(
                        user_id=user_id,
                        trigger_id=trigger_id,
                        current_settings=default_prefs,
                        is_new_user=True
                    )
                    
                    response = await client.views_open(
                        trigger_id=trigger_id,
                        view=modal
                    )
                    
                    if response.get('ok'):
                        self.log_info(f"Welcome modal opened for new user {user_id}")
                        
                        # Send welcome message
                        await client.chat_postMessage(
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've opened your settings panel. Please configure your preferences and I'll be ready to help!"
                        )
                        return  # Don't process the message until settings are saved
                    
                except SlackApiError as e:
                    self.log_error(f"Error opening welcome modal for new user: {e}")
                    # Continue with processing using defaults
            elif is_first_message:
                # No trigger_id available, first message - send interactive message with button
                try:
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments,  # Include file attachments
                        "ts": event.get("ts")  # Include timestamp for proper threading
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": event.get("ts"),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Welcome button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Check if we're in a channel/thread vs DM
                    is_dm = message.channel_id.startswith('D')
                    
                    if is_dm:
                        # For DMs, send the button in the same conversation
                        target_channel = message.channel_id
                        target_thread = message.thread_id
                    else:
                        # For channels/threads, send as a DM to the user
                        target_channel = user_id  # Send to user's DM
                        target_thread = None  # No thread in DM
                        
                        # Also send a brief message in the thread to acknowledge
                        await client.chat_postMessage(
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've sent you a direct message to configure your settings."
                        )
                    
                    # Send welcome button on first interaction
                    # On subsequent messages, the ephemeral will be sent from the outer check
                    
                    # Build blocks for welcome message
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "👋 *Welcome to the AI Assistant!*\n\nI need you to configure your preferences before we begin. Click the button below to open your settings:"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "⚙️ Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                    
                    response = await client.chat_postMessage(
                        channel=target_channel,
                        thread_ts=target_thread,
                        text="👋 Welcome! Please configure your settings to get started.",
                        blocks=blocks
                    )
                    
                    # Track welcome message for updating after settings saved
                    if response.get('ok'):
                        if not hasattr(self, '_welcome_messages'):
                            self._welcome_messages = {}
                        self._welcome_messages[user_id] = {
                            'channel': target_channel,
                            'ts': response.get('ts'),
                            'thread_ts': target_thread
                        }
                    
                    return  # Don't process until settings are configured
                except SlackApiError as e:
                    self.log_error(f"Error sending welcome message: {e}")
            else:
                # Not first message - send regular reminder that we can delete later
                try:
                    response = await client.chat_postMessage(
                        channel=message.channel_id,
                        thread_ts=message.thread_id,
                        text="⚠️ Please configure your settings before I can help you. Click the *Configure Settings* button above to get started."
                    )
                    # Track reminder message for cleanup
                    if response.get('ok'):
                        if not hasattr(self, '_reminder_messages'):
                            self._reminder_messages = {}
                        if user_id not in self._reminder_messages:
                            self._reminder_messages[user_id] = []
                        self._reminder_messages[user_id].append({
                            'channel': message.channel_id,
                            'ts': response.get('ts')
                        })
                except Exception as e:
                    self.log_debug(f"Could not send reminder: {e}")
                return  # Don't process until settings are configured
        else:
            # Existing user with preferences - check if this is a new thread that needs a settings button
            await self._post_settings_button_if_new_thread(message, client, user_prefs)
        
        # Call the message handler if set
        if self.message_handler:
            await self.message_handler(message, self)

    async def _post_settings_button_if_new_thread(self, message: Message, client, user_prefs: dict):
        """Post a settings button at the start of a new thread"""
        try:
            # Check if this is the start of a new thread
            # For channels: thread_id != ts means it's a reply in a thread
            # For DMs: we want to check if there's any history
            
            is_dm = message.channel_id.startswith('D')
            self.log_debug(f"Checking for new thread: is_dm={is_dm}, channel={message.channel_id}, thread={message.thread_id}")
            
            # Get thread history to check if this is a new conversation
            if is_dm:
                # In DMs, every message is technically a new "thread" (unique timestamp)
                # Check if this specific thread already has messages
                history = await client.conversations_replies(
                    channel=message.channel_id,
                    ts=message.thread_id
                )
                self.log_debug(f"DM thread history check: found {len(history.get('messages', []))} messages in thread {message.thread_id}")
                
                # If there's only 1 message (the current one), it's a new thread
                is_new_thread = len(history.get('messages', [])) <= 1
            else:
                # For channels, check if this is creating a new thread
                # When thread_id == ts, it's a new thread (first message)
                is_new_thread = (message.thread_id == message.metadata.get('ts'))
            
            self.log_info(f"New thread check result: is_new_thread={is_new_thread}")
            
            if is_new_thread:
                # Check if this is a new user who hasn't completed settings
                is_new_user = not user_prefs.get('settings_completed', False)
                
                if is_new_user:
                    # New user - need to store message for later processing
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": message.metadata.get('ts'),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Full welcome message for new users
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Welcome to the AI Assistant!* :wave:\n\nI need you to configure your preferences before we can start. You can accept the defaults or customize them."
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                else:
                    # Existing user - message is already being processed, just provide settings access
                    # Only store minimal context needed for settings modal
                    button_value = json.dumps({
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id
                    })
                    
                    # Compact settings button for existing users
                    blocks = [
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
                
                # Post the settings button as the first message in the thread
                await client.chat_postMessage(
                    channel=message.channel_id,
                    thread_ts=message.thread_id,  # Always use thread_ts to post in the thread
                    text="Settings available",
                    blocks=blocks
                )
                
        except Exception as e:
            self.log_debug(f"Could not post settings button: {e}")
            # Don't block message processing if button posting fails
