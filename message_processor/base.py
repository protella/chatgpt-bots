"""
Shared Message Processor
Client-agnostic message processing logic
"""
import asyncio
import logging
import time
from typing import Optional
from base_client import BaseClient, HistoryFetchError, Message, Response
from thread_manager import AsyncThreadStateManager
from openai_client import OpenAIClient
from config import config, pipeline_status
from logger import LoggerMixin
from . import image_catalog
from .containers import ContainerManager
from .message_timestamps import stamp_content
from .thread_management import ThreadManagementMixin
from .turn_runtime import TurnRuntime
from .handlers.text import TextHandlerMixin
from .handlers.image_gen import ImageJobMixin
from .utilities import MessageUtilitiesMixin
from image_url_handler import ImageURLHandler
from mcp_manager import MCPManager
try:
    from document_handler import DocumentHandler
    DOCUMENT_HANDLER_AVAILABLE = True
except ImportError:
    DocumentHandler = None
    DOCUMENT_HANDLER_AVAILABLE = False


# What a turn that ran out of time should actually say. The old copy ("Taking Too Long —
# OpenAI is being slow right now") asserted a cause we have no evidence for: from here a
# timeout looks identical whether the model was slow, the request was genuinely heavy (an
# image and a chart and a document is minutes of real work), or something wedged. So it
# says what we know, and — the part that actually matters — that detached work is NOT lost:
# a background image or research job keeps running and still posts on its own.
TIMEOUT_MESSAGE = (
    "⏱️ *I ran out of time waiting on that one.*\n\n"
    "I can't tell you exactly why — usually it's a heavy request (images, charts and "
    "documents are real work) or a slow spell on the model's end. Nothing is broken.\n\n"
    "Ask me again and I'll have another go. Anything already running in the background — an "
    "image, a research job — will still land here on its own."
)

# The one-line version for the status/thinking indicator, which has no room for the above.
TIMEOUT_STATUS = "That took too long — I stopped waiting."


class MessageProcessor(ThreadManagementMixin,
                       TextHandlerMixin,
                       ImageJobMixin,
                       MessageUtilitiesMixin,
                       LoggerMixin):
    """Handles message processing logic independent of chat platform"""
    
    def __init__(self, db = None):
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = OpenAIClient()
        self.image_url_handler = ImageURLHandler()
        self.document_handler = DocumentHandler() if DOCUMENT_HANDLER_AVAILABLE else None
        self.db = db  # Database manager

        # F32: thread-scoped code-interpreter containers (sandbox state survives the turn).
        self.container_manager = ContainerManager(self.openai_client, db=db)

        # Initialize MCP Manager
        self.mcp_manager = MCPManager(db=db)
        self.mcp_manager.initialize()

        # F51: ambient-memory ingestion service. Owned here so its lifecycle is drained in
        # cleanup() BEFORE the OpenAI client closes. channel_pulse is captured lazily from the
        # Slack client at first offer_event (the client isn't wired to the processor yet).
        from message_processor.ambient_memory import AmbientArtifactService
        self.ambient_service = AmbientArtifactService(
            db=db, openai_client=self.openai_client, channel_pulse=None)

        if not DOCUMENT_HANDLER_AVAILABLE:
            self.log_warning("DocumentHandler not available - document processing will be disabled")
        self.log_info(f"MessageProcessor initialized {'with' if db else 'without'} database")
    








    async def process_message(self, message: Message, client: BaseClient,
                              thinking_id: Optional[str] = None,
                              turn: Optional["TurnRuntime"] = None) -> Optional[Response]:
        """
        Process a message and return a response

        Args:
            message: Universal message object
            client: The client that received the message
            thinking_id: ID of the thinking indicator message to update
            turn: F38 per-turn presentation + work-claim state. Defaults (progress on,
                  never silent) keep non-main.py callers and older tests working.

        Returns:
            Response object or None if unable to process
        """
        if turn is None:
            turn = TurnRuntime(progress_enabled=True, reply_thread_id=message.thread_id)
        thread_key = f"{message.channel_id}:{message.thread_id}"
        
        # Log request start with clear markers
        username = message.metadata.get("username", message.user_id) if message.metadata else message.user_id
        self.log_info("")
        self.log_info("="*100)
        self.log_info(f"REQUEST START | Thread: {thread_key} | User: {username}")
        self.log_info(f"Message: {message.text[:100] if message.text else 'No text'}{'...' if message.text and len(message.text) > 100 else ''}")
        self.log_info("="*100)
        self.log_info("")
        
        request_start_time = time.time()

        # Check if thread is busy
        lock_acquired = False
        try:
            lock_acquired = await self.thread_manager.acquire_thread_lock(
                message.thread_id,
                message.channel_id,
                timeout=0  # Don't wait, return immediately if busy
            )
        except Exception as lock_error:
            self.log_error(f"Lock acquisition failed with error: {lock_error}", exc_info=True)
            raise

        if not lock_acquired:
            # Phase Q: conversational queueing — never reject. The message joins the
            # conversation's pending queue and the in-flight turn's drain hook answers
            # it (batched with any siblings) as one catch-up turn. Only messages that
            # were already going to be processed reach this point: the participation
            # gate (unprompted channel messages) runs BEFORE process_message, so
            # gate-ignored messages never queue. If the queue is full, enqueue_pending
            # drops the message and flags a transcript refetch (Slack still has it).
            elapsed = time.time() - request_start_time
            try:
                self.thread_manager.enqueue_pending(thread_key, message)
            except Exception as queue_error:
                self.log_error(f"Enqueue failed for {thread_key}: {queue_error}", exc_info=True)
                try:
                    self.thread_manager.mark_needs_refresh(thread_key)
                except Exception:
                    pass
            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: QUEUED | Time: {elapsed:.2f}s")
            self.log_info("="*100)
            self.log_info("")
            return Response(type="queued", content="")
        
        try:
            # Get or rebuild thread state
            thread_state = await self._get_or_rebuild_thread_state(
                message,
                client,
                thinking_id
            )

            # F3: if the root author is still unknown and THIS message is the thread root
            # (a new top-level message whose warm state skipped the rebuild), the sender is
            # the root author.
            if (config.enable_wake_envelope
                    and getattr(thread_state, "root_author", None) is None and message.metadata):
                if message.metadata.get("ts") == thread_state.thread_ts:
                    thread_state.root_author = (message.user_id, message.metadata.get("sender_type"))

            # Check if this thread had a previous timeout
            if hasattr(thread_state, 'had_timeout') and thread_state.had_timeout:
                # F38: this notice fires BEFORE the model has decided anything, so on a turn
                # that may end in silence it would break that silence all by itself — the bot
                # would announce a dead answer and then say nothing else. On an addressed turn
                # (where a reply always follows) it still earns its place. Either way the flag
                # is cleared: a stale one would make a LATER prompted turn describe an old
                # failure as "my last answer".
                #
                # F39: keyed on SILENCE, not on `progress_enabled`. A top-level channel reply
                # now sets progress_enabled False (it may not write anything before its finished
                # answer — see TurnRuntime.final_post_only), and reusing that flag here silently
                # swallowed a durable recovery notice on turns that were always going to answer.
                # This notice is not speculative chrome: it is a standalone post, never edited
                # into anything, so it carries no "(edited)" risk. The only reason to hold it
                # back is a turn that might say nothing at all.
                if not getattr(turn, "silence_capable", False):
                    timeout_msg = "⚠️ Heads up — my last answer in this thread never finished. Picking up from here."
                    await client.send_message(
                        channel_id=message.channel_id,
                        text=timeout_msg,
                        thread_id=message.thread_id
                    )
                    self.log_info(f"Notified user about previous timeout in thread {thread_key}")
                else:
                    self.log_debug(
                        f"Prior timeout in {thread_key} — clearing silently (turn may say nothing)")
                thread_state.had_timeout = False


            # Get thread config to determine model (user prefs + shared channel
            # settings; DMs simply have no channel_settings row → no-op there)
            thread_config = await config.get_thread_config_async(
                overrides=thread_state.config_overrides,
                user_id=message.user_id,
                db=self.db,
                channel_id=message.channel_id
            )
            
            # Update thread state with current model for token limit calculations
            thread_state.current_model = thread_config["model"]
            
            # Always regenerate system prompt to get current time
            user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
            user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
            user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
            user_email = message.metadata.get("user_email", None) if message.metadata else None
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            # Ensure the current requester is in the @mention roster, then build it
            if message.user_id:
                thread_state.participants.setdefault(
                    message.user_id,
                    user_real_name or (message.metadata.get("username") if message.metadata else None) or message.user_id,
                )
            participant_roster = self._build_participant_roster(thread_state, client)
            # Phase 7: per-channel ground rules ride on the message metadata into the prompt
            thread_state.channel_directives = message.metadata.get("channel_directives") if message.metadata else None
            # Phase 9: inject this channel's durable memory (None when disabled/empty → prompt unchanged)
            channel_memory_text = await self._build_channel_memory_text(message.channel_id)
            # Channel name/topic/purpose ride in the prompt by default (cached; None in DMs)
            channel_info = await self._build_channel_info(client, message.channel_id)
            thread_state.system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, thread_config["model"], web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'), participant_roster=participant_roster, channel_directives=thread_state.channel_directives, channel_memory=channel_memory_text, channel_info=channel_info)
            
            # F32: ONE artifact sink for the whole turn. The timeout/MCP retries below re-enter
            # the text handler; a per-attempt sink would drop the container id of an attempt that
            # ran code interpreter and then failed, stranding the file it wrote in the sandbox.
            turn_artifacts: list = []

            # Process any attachments (images, documents, and other files).
            # The CI setting must be the PER-THREAD one, resolved the same way the tools array
            # resolves it — a spreadsheet is only worth mounting when the sandbox that reads it
            # will actually be there.
            image_inputs, document_inputs, unsupported_files = await self._process_attachments(
                message, client, thinking_id,
                code_interpreter_enabled=thread_config.get(
                    'enable_code_interpreter', config.enable_code_interpreter))

            # T2-10: a catch-up trigger carries EARLIER batched messages' already-processed image
            # parts and attachment failures (staged in _dispatch_pending_batch — re-downloading
            # here would be wasteful). Merge failures into unsupported_files so the notice below
            # acknowledges them, and fold the image parts into THIS turn's image_inputs so the
            # model can actually see them. The trigger's OWN images win the per-turn slots;
            # earlier-batch images fill what's left; any overflow is noted in the text.
            batched_image_inputs = (message.metadata or {}).get("batched_image_inputs") or []
            batched_unsupported = (message.metadata or {}).get("batched_unsupported_files") or []
            if batched_unsupported:
                unsupported_files = list(unsupported_files) + list(batched_unsupported)
            batched_images_omitted = 0
            if batched_image_inputs:
                image_cap = 10  # matches _process_attachments' max_images (utilities.py)
                room = max(0, image_cap - len(image_inputs))
                if room:
                    image_inputs = list(image_inputs) + list(batched_image_inputs[:room])
                batched_images_omitted = max(0, len(batched_image_inputs) - room)

            # Files that were accepted but couldn't be fetched/processed create an
            # obligation: use them or tell the user they failed — never answer as
            # if they were never attached.
            if unsupported_files:
                files_str = ", ".join(f"*{f['name']}*" for f in unsupported_files)
                unsupported_msg = self._build_failed_files_notice(unsupported_files)
                
                # If there's also text, images, or documents, continue processing those
                if (message.text and message.text.strip()) or image_inputs or document_inputs:
                    unsupported_msg += "\n\nI'll process your text/image/document request now."
                    # The MIXED path continues on to generate the real reply, so — unlike the
                    # all-failed branch below, which RETURNS the notice for main.py to post — it
                    # must deliver this notice itself. Recording it only in thread state (as it
                    # used to) left the model believing it had acknowledged the failed files while
                    # the user saw nothing. Post it now, then record the same text as an assistant
                    # turn so the model's context matches what was actually delivered.
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    try:
                        await client.send_message(
                            channel_id=message.channel_id,
                            text=unsupported_msg,
                            thread_id=message.thread_id,
                        )
                    except Exception as notice_err:  # noqa: BLE001 — never fail the turn over the notice
                        self.log_warning(f"Failed to post mixed-path failed-files notice: {notice_err}")
                    # Add the unsupported files warning to conversation
                    formatted_content = self._format_user_content_with_username(f"[File(s) not processed: {files_str}]", message)
                    self._add_message_with_token_management(thread_state, "user", formatted_content, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", unsupported_msg, db=self.db, thread_key=thread_key)
                    # Continue processing if we have text or images
                else:
                    # Only unsupported files were uploaded, nothing else to process
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_content = self._format_user_content_with_username(f"[File(s) not processed: {files_str}]", message)
                    self._add_message_with_token_management(thread_state, "user", formatted_content, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", unsupported_msg, db=self.db, thread_key=thread_key)
                    elapsed = time.time() - request_start_time
                    self.log_info("")
                    self.log_info("="*100)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: UNSUPPORTED_FILE | Time: {elapsed:.2f}s")
                    self.log_info("="*100)
                    self.log_info("")
                    return Response(
                        type="text",
                        content=unsupported_msg
                    )
            
            # Build user content
            # First, format the base text with username
            username = message.metadata.get("username", "User") if message.metadata else "User"
            base_text_with_username = f"{username}: {message.text}" if message.text else f"{username}:"
            # F10: stamp this turn (aligns this bypass site with _format_user_content_with_username);
            # the stamp rides at the front, ahead of any document summaries appended below.
            if config.enable_message_timestamps and message.metadata:
                base_text_with_username = stamp_content(
                    base_text_with_username, message.metadata.get("ts"),
                    message.metadata.get("user_timezone") or "UTC")

            # If we have documents, enhance the text with their labeled SUMMARIES
            # (full content never enters context — read_document covers depth)
            enhanced_text = base_text_with_username
            file_inputs = []
            if document_inputs:
                enhanced_text = self._build_message_with_documents(base_text_with_username, document_inputs)
                # Native-eligible files additionally ride this turn as input_file parts:
                # PDFs so the model sees text + rendered pages (Phase D2), and F32
                # spreadsheets/CSVs so they auto-mount in the code-interpreter sandbox and
                # can actually be computed over. The mimetype MUST come from the document —
                # hard-coding application/pdf here would hand the API a CSV wearing a PDF
                # content type.
                for doc in document_inputs:
                    if doc.get("native") and doc.get("file_data_b64"):
                        mimetype = doc.get("mimetype") or "application/pdf"
                        file_inputs.append({
                            "type": "input_file",
                            "filename": doc.get("filename", "document.pdf"),
                            "file_data": f"data:{mimetype};base64,{doc['file_data_b64']}",
                        })

            # T2-10: if the per-turn image cap dropped some earlier-batch images, say so — the
            # model must not answer as if it saw every image in the catch-up.
            if batched_images_omitted:
                enhanced_text += (
                    f"\n\n[Note: {batched_images_omitted} image(s) from earlier messages in this "
                    f"catch-up couldn't be attached — the per-message image limit was reached.]")

            user_content = self._build_user_content(enhanced_text, image_inputs, file_inputs)

            # Check if adding this message would exceed limits and trim if needed
            # We temporarily add the message to check, then remove it
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Determine what content to use for checking
            content_to_check = enhanced_text if not image_inputs else (enhanced_text if enhanced_text else f"{username}: [uploaded image(s) for analysis]")
            
            # Check token count with the new message (WITHOUT adding it to thread yet)
            model = thread_state.current_model or config.gpt_model
            max_tokens = config.get_model_token_limit(model)

            # Calculate what the tokens would be with the new message
            temp_message = {"role": "user", "content": content_to_check}
            new_message_tokens = self.thread_manager._token_counter.count_message_tokens(temp_message)
            current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
            projected_tokens = current_tokens + new_message_tokens

            # Debug logging for token counting
            self.log_debug(f"Token calculation: current={current_tokens}, new_message={new_message_tokens}, projected={projected_tokens}")
            self.log_debug(f"New message length: {len(content_to_check)} chars = {new_message_tokens} tokens")

            # Apply smart trimming if needed - keep trimming until under limit
            if projected_tokens > max_tokens:
                self.log_info(f"Thread would exceed limit with new message ({projected_tokens}/{max_tokens} tokens), applying smart trim")

                # Update status to show we're optimizing (routes to the composer
                # status on status-only DMs where no indicator message exists)
                self._update_status(
                    client,
                    message.channel_id,
                    thinking_id,
                    pipeline_status("optimizing_history", f"Optimizing conversation history ({projected_tokens:,}/{max_tokens:,} tokens)…"),
                    emoji=config.circle_loader_emoji, thread_id=message.thread_id, turn=turn)

                total_trimmed = 0

                # Keep trimming until we're under the limit (accounting for the new message we'll add)
                while projected_tokens > max_tokens:
                    # Smart trim will work on existing messages only (not the temp one)
                    trimmed_count = await self._smart_trim_with_summarization(thread_state)
                    total_trimmed += trimmed_count
                    
                    if trimmed_count == 0:
                        # No more messages to trim, we've done all we can
                        self.log_warning(f"Cannot trim further - still at {projected_tokens} tokens")
                        break

                    # Recalculate tokens after trimming (including the message we'll add)
                    current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                    projected_tokens = current_tokens + new_message_tokens
                    self.log_debug(f"After trimming {trimmed_count} messages, now at {projected_tokens}/{max_tokens} tokens (current: {current_tokens} + new: {new_message_tokens})")
                
                if total_trimmed > 0:
                    self.log_info(f"Smart trim complete: {total_trimmed} total messages processed, final: {projected_tokens}/{max_tokens} tokens")

                # Check if we're still over the limit after trimming
                if projected_tokens > max_tokens:
                    self.log_warning(f"Smart trim insufficient. Need {projected_tokens - max_tokens} more tokens. Dropping oldest messages...")

                    # Keep dropping oldest messages until we fit
                    messages_dropped = 0
                    while projected_tokens > max_tokens and len(thread_state.messages) > 0:
                        # Drop the oldest non-preserved message
                        dropped = False
                        for i in range(len(thread_state.messages)):
                            if not self._should_preserve_message(thread_state.messages[i]):
                                dropped_msg = thread_state.messages.pop(i)
                                messages_dropped += 1
                                dropped = True

                                # Recalculate tokens
                                current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                                projected_tokens = current_tokens + new_message_tokens
                                self.log_debug(f"Dropped message {i}, now at {projected_tokens}/{max_tokens} tokens")
                                break

                        if not dropped:
                            # No more droppable messages
                            self.log_warning("No more messages can be dropped (all are preserved)")
                            break

                        # Safety check to prevent infinite loop
                        if messages_dropped > 50:
                            self.log_error("Dropped 50 messages but still over limit - something is wrong")
                            break

                    if messages_dropped > 0:
                        self.log_info(f"Dropped {messages_dropped} oldest messages to make room. Final: {projected_tokens}/{max_tokens} tokens")
                        # Mark that we've trimmed messages
                        thread_state.has_trimmed_messages = True

            # No need to remove temp message since we never added it to thread_state.messages
            
            # Check if this single message alone exceeds the model's context window
            model = thread_state.current_model or config.gpt_model
            max_model_tokens = config.get_model_token_limit(model)

            # Check if this single message exceeds the model's context window
            if new_message_tokens > max_model_tokens:
                error_msg = (
                    f"❌ Your message is too large for the model to process.\n\n"
                    f"• Message size: {new_message_tokens:,} tokens\n"
                    f"• Model limit: {max_model_tokens:,} tokens\n\n"
                    f"Please reduce the size of your documents or split them into smaller requests."
                )

                # Log the issue
                self.log_error(f"Message exceeds context window: {new_message_tokens} > {max_model_tokens}")
                
                # Add minimal breadcrumb to history
                thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                message_ts = message.metadata.get("ts") if message.metadata else None
                formatted_error_breadcrumb = self._format_user_content_with_username(
                    f"[Attempted to upload {len(document_inputs)} document(s) - exceeded context limit]", 
                    message
                )
                self._add_message_with_token_management(
                    thread_state, "user", 
                    formatted_error_breadcrumb,
                    db=self.db, thread_key=thread_key, message_ts=message_ts
                )
                self._add_message_with_token_management(
                    thread_state, "assistant", error_msg,
                    db=self.db, thread_key=thread_key
                )
                
                return Response(type="error", content=error_msg)
            
            # F34: image generation and editing are TOOLS, so there is nothing left for a
            # pre-flight router to decide. The model sees uploaded images directly (they ride
            # the turn as input_image parts) and calls generate_image / create_image_asset /
            # edit_image in context — so it can generate an image AND compute a chart from real
            # data in the SAME turn, which a single-choice classifier made impossible. That
            # router is also what drew a chart with invented numbers when someone said "chart
            # this CSV": it had to guess "image request" before the model ever saw the data.
            #
            # Uploaded images still earn a durable visual description, but as a background side
            # effect — not by routing the whole turn through a vision handler. The image tools
            # claim the upload latch themselves (image_tools.py), so there is no latch to set
            # here either.
            if image_inputs and message.attachments:
                self._schedule_async_call(image_catalog.catalog_uploads(
                    self, thread_key, message.attachments, image_inputs,
                    (message.metadata or {}).get("ts")))

            response = await self._handle_text_response(
                user_content, thread_state, client, message, thinking_id, retry_count=0,
                artifacts_acc=turn_artifacts, turn=turn)

            # DEBUG: log conversation history after processing (with truncated content).
            # log_debug, not print — conversation content must not leak to stdout
            # unconditionally, and the json.dumps is only worth building at debug level.
            if self.logger.isEnabledFor(logging.DEBUG):
                import json
                truncated_messages = []
                for msg in thread_state.messages:
                    truncated_msg = msg.copy()
                    content = str(truncated_msg.get("content", ""))
                    if len(content) > 100:
                        truncated_msg["content"] = content[:100] + f"... [truncated {len(content) - 100} chars]"
                    truncated_messages.append(truncated_msg)
                self.log_debug("CONVERSATION HISTORY (TRUNCATED):\n" + json.dumps(truncated_messages, indent=2))
            
            elapsed = time.time() - request_start_time
            response_type = response.type if response else "None"
            
            # Calculate final token count
            final_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)

            # F38: the "📊 CONTEXT USAGE NOTIFICATION" box is gone. Compaction is a
            # behind-the-scenes function and the bot has no business narrating it — this
            # posted a public ASCII box of token counts and "tips" into the thread, where
            # everyone could see it, over a thing the user never asked about and cannot act
            # on. The model is still TOLD its history was summarized (the has_trimmed_messages
            # note in the system prompt); that is where the fact belongs.

            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: {response_type.upper()} | Time: {elapsed:.2f}s | Tokens: {final_tokens}")
            self.log_info("="*100)
            self.log_info("")
            return response
            
        except TimeoutError as e:
            # Handle timeout errors gracefully without stack trace
            elapsed = time.time() - request_start_time
            # Try to get token count even on error
            try:
                error_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages) if 'thread_state' in locals() else 0
                token_info = f" | Tokens: {error_tokens}" if error_tokens > 0 else ""
            except Exception:
                token_info = ""

            # Get the operation type that timed out
            operation_type = getattr(e, 'operation_type', 'unknown')
            self.log_warning(f"Request timeout after {elapsed:.2f} seconds for thread {thread_key} (operation: {operation_type}): {e}")

            # Only a text turn is worth retrying. Image work is detached (the background job
            # is still running and posts on its own) and there is no classifier hop left to
            # re-run, so a single text retry with a shorter timeout is the whole recovery path.
            already_retried = getattr(e, 'already_retried', False)
            should_retry = operation_type == 'text_normal' and not already_retried

            if should_retry:
                # Mark as retry attempt to prevent infinite loops
                e.already_retried = True

                # Update status to show retry
                if thinking_id and hasattr(client, 'update_message'):
                    retry_msg = "OpenAI is slow to respond. Retrying with shorter timeout..."
                    try:
                        self._update_status(client, message.channel_id, thinking_id, retry_msg, emoji="⏳", thread_id=message.thread_id)
                        self.log_debug("Updated thinking message to show retry attempt")
                    except Exception as update_error:
                        self.log_error(f"Failed to update thinking message for retry: {update_error}")

                self.log_info(f"Retrying {operation_type} operation with 60s timeout...")

                # F7: retry with the ORIGINAL multipart user_content, not enhanced_text.
                # enhanced_text is always in locals (a plain string built at :298), so the old
                # guard never fell through to user_content — image/file parts (folded into
                # user_content at :317) were silently dropped on every timeout retry.
                retry_content = user_content if 'user_content' in locals() else enhanced_text
                # F7: the first attempt appended this turn's user message to thread state
                # (text.py:361/376) before the API call timed out. Pop it before retrying so the
                # retry doesn't append a second copy and duplicate the user turn — mirrors the
                # context-length cleanup in text.py:612-613.
                if thread_state.messages and thread_state.messages[-1].get("role") == "user":
                    thread_state.messages.pop()

                try:
                    response = await self._handle_text_response(
                        retry_content,
                        thread_state, client, message, thinking_id,
                        retry_count=1,
                        artifacts_acc=turn_artifacts, turn=turn
                    )
                    self.log_info(f"Retry successful for {operation_type}")
                    return response

                except TimeoutError as retry_error:
                    self.log_warning(f"Retry also failed for {operation_type}: {retry_error}")
                    # Continue to error handling below
                except Exception as retry_error:
                    self.log_error(f"Retry failed with unexpected error for {operation_type}: {retry_error}")
                    # Continue to error handling below

            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: TIMEOUT | Time: {elapsed:.2f}s{token_info}")
            self.log_info("="*100)
            self.log_info("")

            # Update thinking message to show final timeout
            if thinking_id and hasattr(client, 'update_message'):
                timeout_msg = TIMEOUT_STATUS
                try:
                    self._update_status(client, message.channel_id, thinking_id, timeout_msg, emoji=config.error_emoji, thread_id=message.thread_id)
                    self.log_debug("Updated thinking message to show timeout")
                except Exception as update_error:
                    self.log_error(f"Failed to update thinking message: {update_error}")

            # Mark thread as having a timeout for recovery
            if 'thread_state' in locals() and thread_state:
                thread_state.had_timeout = True

            error_message = TIMEOUT_MESSAGE

            return Response(
                type="error",
                content=error_message
            )
        except HistoryFetchError as e:
            # Slack wouldn't give us the thread transcript (rate-limited or hard API
            # error after retries). Since Phase S the platform IS the context — fail
            # the turn loudly rather than answering with amnesia (R1).
            self.log_error(f"History fetch failed for {thread_key}: {e}")
            elapsed = time.time() - request_start_time
            self.log_info("")
            self.log_info("=" * 100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: HISTORY_FETCH_FAILED | Time: {elapsed:.2f}s")
            self.log_info("=" * 100)
            self.log_info("")

            if thinking_id and hasattr(client, 'update_message'):
                try:
                    await client.update_message(
                        message.channel_id, thinking_id,
                        f"{config.error_emoji} Couldn't load this conversation's history from Slack."
                    )
                except Exception:
                    pass

            return Response(
                type="error",
                content=(
                    f"{config.error_emoji} **Couldn't Load Conversation History**\n\n"
                    "Slack didn't return this conversation's history (it may be busy or "
                    "rate-limiting). Your message wasn't processed — please try again in a moment."
                )
            )
        except Exception as e:
            # Log full error details for non-timeout exceptions
            self.log_error(f"Error processing message: {e}", exc_info=True)
            elapsed = time.time() - request_start_time
            # Try to get token count even on error
            try:
                error_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages) if 'thread_state' in locals() else 0
                token_info = f" | Tokens: {error_tokens}" if error_tokens > 0 else ""
            except Exception:
                token_info = ""

            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: ERROR | Time: {elapsed:.2f}s{token_info}")
            self.log_info("="*100)
            self.log_info("")

            # Check if this is a timeout error that wasn't caught as TimeoutError
            error_str = str(e)
            error_type = type(e).__name__

            # Check for various timeout error types that weren't caught as TimeoutError
            if any(timeout_indicator in error_str.lower() or timeout_indicator in error_type.lower()
                   for timeout_indicator in ['timeout', 'readtimeout', 'connecttimeout']):
                # Update thinking message to show timeout
                if thinking_id and hasattr(client, 'update_message'):
                    timeout_msg = f"{config.error_emoji} {TIMEOUT_STATUS}"
                    try:
                        await client.update_message(message.channel_id, thinking_id, timeout_msg)
                    except Exception:
                        pass  # Don't let update failure affect error handling

                # Mark as timeout for recovery
                if 'thread_state' in locals() and thread_state:
                    thread_state.had_timeout = True

                # Timeout-specific error message
                error_message = TIMEOUT_MESSAGE
                self.log_warning(f"Request timeout (via string match) after {elapsed:.2f} seconds for thread {thread_key}")
            else:
                # Update thinking message to show error
                if thinking_id and hasattr(client, 'update_message'):
                    error_msg = f"{config.error_emoji} Something went wrong. Try again."
                    try:
                        await client.update_message(message.channel_id, thinking_id, error_msg)
                    except Exception:
                        pass  # Don't let update failure affect error handling

                # Generic error message - keep it simple for users
                # Log the actual error for debugging, but don't show technical details to user
                error_details = str(e)

                # Check for common error types and provide user-friendly messages
                # IMPORTANT: Check MCP errors FIRST before generic "context" check (which would match "context7" server names)
                if "mcp server" in error_details.lower() and ("404" in error_details or "424" in error_details):
                    error_message = f"{config.error_emoji} **MCP Connection Failed**\n\nCouldn't connect to one or more MCP servers. Please check your MCP configuration or try again later."
                elif "rate" in error_details.lower() or "limit" in error_details.lower():
                    error_message = f"{config.error_emoji} **Too Many Requests**\n\nOpenAI is busy. Please wait a minute and try again."
                elif "context_length_exceeded" in error_details.lower() or "maximum context length" in error_details.lower():
                    # More specific context window check (avoid matching MCP server names like "context7")
                    error_message = f"{config.error_emoji} **Message Too Long**\n\nYour message is too long. Please try a shorter request."
                elif "api" in error_details.lower() or "openai" in error_details.lower():
                    error_message = f"{config.error_emoji} **Service Issue**\n\nOpenAI is having problems. Please try again shortly."
                else:
                    # Generic fallback
                    error_message = f"{config.error_emoji} **Something Went Wrong**\n\nPlease try again. If this keeps happening, try later."

            return Response(
                type="error",
                content=error_message
            )
        finally:
            # Phase Q drain hook — runs while we STILL HOLD the lock so that (a) no new
            # message can jump ahead of the queued backlog and (b) stragglers arriving
            # during the linger enqueue (lock held) and join the same batch. Must never
            # prevent the lock release below.
            try:
                await self._dispatch_pending_batch(message, client, thread_key)
            except Exception as drain_error:
                self.log_error(f"Pending-queue drain failed for {thread_key}: {drain_error}", exc_info=True)
                await self._notify_drain_failure(message, client, thread_key)
            # Always release the thread lock, even on timeout
            try:
                await self.thread_manager.release_thread_lock(
                    message.thread_id,
                    message.channel_id
                )
            except Exception as lock_error:
                # Even if release fails, log it but don't crash
                self.log_error(f"Error releasing thread lock for {thread_key}: {lock_error}", exc_info=True)

    async def _notify_drain_failure(self, message: Message, client: BaseClient, thread_key: str):
        """Queued messages were silently accepted — their senders must not get
        silence when the catch-up turn dies. Flag a transcript refetch so context
        recovers, and tell the thread to re-send (both best-effort)."""
        try:
            self.thread_manager.mark_needs_refresh(thread_key)
        except Exception:
            pass
        try:
            await client.send_message_async(
                message.channel_id, message.thread_id,
                "⚠️ I hit an error catching up on the last few messages — please re-send."
            )
        except Exception as notify_error:
            self.log_error(f"Failed to post drain-failure notice for {thread_key}: {notify_error}")

    @staticmethod
    def _build_failed_files_notice(unsupported_files: list) -> str:
        """User notice for files that were accepted but not processed.

        Four different failures, four different things worth saying. Oversized documents
        (fix-a1's `too_large` flag) get an honest size-vs-limit line — routing them through the
        download bucket read "Couldn't Download — try re-uploading", which is misleading advice
        for a file that arrived fine and was simply too big. Download failures get their own
        actionable line (re-upload). Images we FETCHED and then turned away (F50) carry a
        `reason` and get told exactly what was wrong with them — routing those through the
        generic explainer below would print "GIF is supported" underneath a rejected animated
        GIF, which is worse than saying nothing. Everything else keeps the supported-formats
        explainer.
        """
        too_large = [f for f in unsupported_files if f.get('too_large')]
        download_failures = [f for f in unsupported_files
                             if not f.get('too_large') and f.get('error') == 'download_failed']
        rejected_images = [f for f in unsupported_files
                           if not f.get('too_large') and f.get('error') != 'download_failed'
                           and f.get('reason')]
        truly_unsupported = [f for f in unsupported_files
                             if not f.get('too_large') and f.get('error') != 'download_failed'
                             and not f.get('reason')]

        def _mb(n):
            return f"{n / (1024 * 1024):.1f}MB" if isinstance(n, (int, float)) else "?"

        sections = []
        if too_large:
            lines = "\n".join(
                f"*{f['name']}* is too large ({_mb(f.get('size_bytes'))}, "
                f"max {_mb(f.get('limit_bytes'))})"
                for f in too_large)
            sections.append("⚠️ *File Too Large*\n\n" + lines)
        if download_failures:
            failed_str = ", ".join(f"*{f['name']}*" for f in download_failures)
            sections.append(
                "⚠️ *Couldn't Download File*\n\n"
                f"I couldn't download {failed_str} — try re-uploading."
            )
        if rejected_images:
            from image_validation import rejection_text
            lines = "\n".join(f"*{f['name']}* {rejection_text(f.get('reason'))}"
                              for f in rejected_images)
            sections.append("⚠️ *Couldn't Read Image*\n\n" + lines)
        if truly_unsupported:
            types_str = ", ".join(sorted({f['mimetype'] for f in truly_unsupported}))
            unsup_str = ", ".join(f"*{f['name']}*" for f in truly_unsupported)
            section = "⚠️ *Unsupported File Type*\n\n"
            section += f"I noticed you uploaded: {unsup_str}\n\n"
            section += f"*File type(s):* `{types_str}`\n\n"
            section += "───────────────\n"
            section += "*Currently supported:*\n"
            section += "• Images (JPEG, PNG, GIF, WebP)\n"
            # Generated from the handler's own table so this list can't lie. The set is
            # now large (dozens of code/config/text extensions), so we surface the common
            # ones and honestly summarize the tail as "and N more" rather than dumping all.
            from document_handler import DOCUMENT_EXTENSIONS
            common = ["PDF", "DOCX", "XLSX", "CSV", "TSV", "PPTX", "TXT", "MD", "JSON", "RTF"]
            shown = [t for t in common if f".{t.lower()}" in DOCUMENT_EXTENSIONS]
            remaining = len(DOCUMENT_EXTENSIONS) - len(shown)
            doc_types = ", ".join(shown)
            if remaining > 0:
                doc_types += f", and {remaining} more"
            section += f"• Documents ({doc_types})\n\n"
            section += "_Support for additional file types may be added in the future._"
            sections.append(section)
        return "\n\n".join(sections)

    async def _dispatch_pending_batch(self, finished_message: Message, client: BaseClient, thread_key: str):
        """Phase Q: after a turn finishes (lock still held), drain the conversation's
        pending queue into ONE batched catch-up turn and re-dispatch it through the
        normal message pipeline.

        Mechanics:
        - Linger QUEUE_DRAIN_LINGER_SECONDS while still holding the lock: stragglers
          arriving now enqueue (the lock is held) and are included in the pop below.
        - Pop up to QUEUE_MAX_BATCH messages atomically. All but the last are appended
          to thread state individually (attributed, ts-stamped) so history is correct
          and the model answers the combined content. The LAST message becomes the
          trigger for the re-dispatched turn (its ts/attachments drive ToolContext and
          reactions — a documented simplification: attachments on earlier batch messages
          are represented in text only until a rebuild).
        - The re-dispatch is a background task through client.message_handler (the
          same entry Slack events use), so the batch turn gets the full normal flow:
          thinking indicator, streaming, footer, participation stats. It starts after
          this turn releases the lock; if a brand-new message wins the lock race
          first, the batch trigger simply re-enqueues — nothing is ever lost.
        - Messages left beyond QUEUE_MAX_BATCH drain on the following turn via this
          same hook (loop-until-empty is emergent, no dedicated loop needed).
        """
        manager = self.thread_manager
        if manager.pending_count(thread_key) == 0:
            return

        linger = max(0.0, float(getattr(config, "queue_drain_linger_seconds", 1.0)))
        if linger:
            await asyncio.sleep(linger)

        batch = manager.pop_pending_batch(thread_key, int(getattr(config, "queue_max_batch", 10)))
        if not batch:
            return

        # F52 double-answer fix (queue-drop backstop): drop a queued PRE-EDIT participation
        # dispatch whose message was since edited and handled by the edit path. Such a dispatch
        # slipped into the busy queue before the engine supersession landed; carried forward it
        # RE-RUNS the gate on stale text and posts a duplicate (live 2026-07-16). It is identified
        # by carrying participation_check for a ts the edit path registered, WITHOUT the surviving
        # edit's marker (the edit's own engine re-dispatch carries it and is kept). Addressed
        # (app_mention/DM) turns and ordinary different messages carry no participation_check and
        # are never touched; a genuinely different queued message has a different ts.
        marker_getter = getattr(client, "edit_dispatch_marker", None)
        if callable(marker_getter):
            kept = []
            for queued_msg in batch:
                try:
                    meta = queued_msg.metadata or {}
                    ts = meta.get("ts")
                    if meta.get("participation_check") and ts is not None:
                        surviving = marker_getter(queued_msg.channel_id, ts)
                        if surviving is not None and meta.get("edit_reply_marker") != surviving:
                            self.log_info(
                                f"Dropping stale pre-edit participation dispatch (ts={ts}) "
                                f"superseded by an edit on {thread_key}")
                            continue
                except Exception as drop_err:  # noqa: BLE001 — never let the check lose a message
                    self.log_warning(f"Edit-stale drop check failed: {drop_err}")
                kept.append(queued_msg)
            batch = kept
            if not batch:
                return

        handler = getattr(client, "message_handler", None)
        if handler is None:
            # No re-dispatch path (exotic client) — the messages exist in Slack;
            # flag a transcript refetch so the next turn recovers them in context.
            manager.mark_needs_refresh(thread_key)
            self.log_warning(f"No message_handler to drain {len(batch)} queued message(s) on {thread_key}")
            return

        trigger = batch[-1]
        # T2-10: earlier messages' image parts + attachment failures are collected here and
        # carried to the trigger turn — images so the model can actually SEE them (not just
        # their catalogued description), failures so a dropped file is acknowledged.
        batched_image_inputs: list = []
        batched_unsupported_files: list = []
        if len(batch) > 1:
            # Append the earlier messages to warm state now (we hold the lock, the
            # state is current). The trigger message is NOT appended — its own turn
            # does that, exactly like any normal message.
            thread_state = await manager.get_thread_async(
                finished_message.thread_id, finished_message.channel_id
            )
            if thread_state is not None:
                # F10: earlier batch messages' attachments used to be dropped — only text was
                # appended, so their DOCUMENTS got no save_document row and were unreachable by
                # read_document/mount_file (and their images rode only ambient dual-write). Resolve
                # the per-thread code-interpreter setting once, and ONLY when some earlier message
                # actually carries attachments (the common no-attachment batch stays cheap), so the
                # attachment pipeline below makes the same native-vs-local call the trigger would.
                batch_ci_enabled = None
                if any(qm.attachments for qm in batch[:-1]):
                    batch_thread_config = await config.get_thread_config_async(
                        overrides=thread_state.config_overrides,
                        user_id=finished_message.user_id,
                        db=self.db,
                        channel_id=finished_message.channel_id,
                    )
                    batch_ci_enabled = batch_thread_config.get(
                        'enable_code_interpreter', config.enable_code_interpreter)
                for queued_msg in batch[:-1]:
                    try:
                        content = self._format_user_content_with_username(queued_msg.text or "", queued_msg)
                        # F10: run the SAME attachment pipeline the trigger turn runs, keyed on this
                        # message's own ts so documents persist under the right source and images are
                        # catalogued. Fold the document summaries into this message's appended content
                        # so the model sees them in context too (the trigger's enhanced_text pattern).
                        if queued_msg.attachments:
                            q_image_inputs, q_document_inputs, q_unsupported = await self._process_attachments(
                                queued_msg, client,
                                code_interpreter_enabled=batch_ci_enabled)
                            if q_document_inputs:
                                content = self._build_message_with_documents(content, q_document_inputs)
                            if q_image_inputs:
                                # Catalogue a durable description AND carry the raw parts to the
                                # trigger turn so the model actually sees the images (T2-10).
                                self._schedule_async_call(image_catalog.catalog_uploads(
                                    self, thread_key, queued_msg.attachments, q_image_inputs,
                                    (queued_msg.metadata or {}).get("ts")))
                                batched_image_inputs.extend(q_image_inputs)
                            if q_unsupported:
                                batched_unsupported_files.extend(q_unsupported)
                        self._add_message_with_token_management(
                            thread_state, "user", content,
                            db=self.db, thread_key=thread_key,
                            message_ts=(queued_msg.metadata or {}).get("ts"),
                        )
                    except Exception as append_error:
                        self.log_warning(f"Failed to append queued message to state: {append_error}")
        # Mark the trigger so the UI can show a catch-up status for multi-message batches.
        if trigger.metadata is None:
            trigger.metadata = {}
        trigger.metadata["queued_batch_size"] = len(batch)
        # T2-10: hand the trigger turn the earlier messages' image parts and attachment failures.
        # process_message folds the images into this turn's multipart content (respecting the
        # per-turn cap) and routes the failures through the failed-files notice.
        if batched_image_inputs:
            trigger.metadata["batched_image_inputs"] = batched_image_inputs
        if batched_unsupported_files:
            trigger.metadata["batched_unsupported_files"] = batched_unsupported_files

        self.log_info(f"Draining {len(batch)} queued message(s) on {thread_key} into one catch-up turn")
        self._schedule_async_call(handler(trigger, client))

    async def cleanup(self):
        """Clean up resources and close clients."""
        self.log_info("Cleaning up MessageProcessor resources...")
        # F51: drain the ambient service FIRST — its workers call the OpenAI client + DB, so it
        # must finish (or be cancelled) before the client is closed under it.
        if getattr(self, "ambient_service", None):
            try:
                await self.ambient_service.shutdown()
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"Ambient service shutdown error: {e}")
        if hasattr(self, 'openai_client') and self.openai_client:
            await self.openai_client.close()
        # Close thread manager resources if needed
        if hasattr(self.thread_manager, 'cleanup'):
            await self.thread_manager.cleanup()
        self.log_info("MessageProcessor cleanup completed")
    





















