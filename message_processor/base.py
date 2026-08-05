"""
Shared Message Processor
Client-agnostic message processing logic
"""
import asyncio
import logging
import time
from typing import Optional
from base_client import BaseClient, ChannelStreamError, HistoryFetchError, Message, Response
from thread_manager import AsyncThreadStateManager
from openai_client import OpenAIClient
from config import config, pipeline_status
from logger import LoggerMixin
from slack_client import admission_watermark
from . import channel_steering, image_catalog, participation_telemetry, routing_facts
from .containers import ContainerManager
from .message_timestamps import stamp_content
from .thread_management import ThreadManagementMixin
from .stale_send_guard import StaleSendSuppressed
from .turn_runtime import TurnRuntime
from .handlers.text import TextHandlerMixin, pinned_thread_config
from .handlers.image_gen import ImageJobMixin
from .utilities import MessageUtilitiesMixin, effective_request_model
from image_url_handler import ImageURLHandler
from mcp_manager import MCPManager
from tool_registry import SURFACE_CHANNEL
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
        # cleanup() BEFORE the OpenAI client closes.
        from message_processor.ambient_memory import AmbientArtifactService
        self.ambient_service = AmbientArtifactService(
            db=db, openai_client=self.openai_client)

        # Track 1: the per-channel narrative service. Only build_for_intro survives P2 (the
        # channel stream renders the room itself), and that one needs the db + openai client.
        from message_processor.channel_summary import ChannelSummaryService
        self.channel_summary_service = ChannelSummaryService(
            db=db, openai_client=self.openai_client)

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

        # THE POINT OF NO RETURN for queued-batch lineage. The lock is held, so this turn is
        # genuinely running and is what the messages it absorbed were folded into. An UNGATED
        # turn has no attempt id, so if this were written any earlier — back when the turn was
        # merely intended — a turn that reached the queue instead would have claimed messages it
        # never answered, with nothing downstream able to correct it. A gated turn wrote its
        # links at the gate (this is then a no-op); one whose attempt was never minted keeps its
        # sources staged for whichever turn does run. Reads its OWN routing fact rather than
        # trusting the caller: mislabeling the route here would misfile the lineage.
        gate_required = (message.metadata or {}).get("gate_required") is True
        participation_telemetry.emit_queue_links(message, gate_required=gate_required)

        channel_turn = TextHandlerMixin._turn_surface(message) == SURFACE_CHANNEL
        h_pin = None

        try:
            # H, pinned HERE and never refreshed (spec §1). This is the first instant at which the
            # turn genuinely exists — the lock is held, the queue is this turn's, and nothing has
            # been awaited since — so it is the honest answer to "what had this channel said by the
            # time we committed to answering". Pinned before the steering read and before any
            # stream or capability await, because every one of those is time in which the channel
            # moves, and a window whose ceiling drifts is a window two builds of one turn would
            # disagree about.
            #
            # The frontier travels WITH it, captured in the same synchronous step: the turn waits
            # only for index writes at or below the frontier, so an event that arrived after H can
            # neither delay this turn nor fail it.
            #
            # INSIDE the try, deliberately. A malformed timestamp anywhere on the turn path fails
            # closed [r3-24], and the pin is the first place that can happen — outside the try the
            # raise would escape past the `finally` that releases the conversation lock, and the
            # thread would be wedged for the life of the process by a single bad ts.
            if channel_turn:
                meta = message.metadata or {}
                h_pin = admission_watermark.pin(
                    message.channel_id,
                    meta.get("trigger_admission_ts") or meta.get("ts") or message.thread_id)
                turn.H = h_pin.h

            # An UNGATED turn (a DM, an @mention, a thread continuation) has no gate to have read
            # the channel's steering, so it reads it HERE — at the point of no return, not earlier.
            # A message that queues instead never reaches this line, so it costs nothing; when its
            # redispatch finally runs, it reads the steering as it is THEN, not as it was when the
            # message arrived. Gated turns arrive already stamped by the gate and must not read
            # again; see message_processor/channel_steering.py for why a second read is the bug.
            if not gate_required:
                channel_steering.stamp(message, await channel_steering.load_snapshot(
                    self.db, message.channel_id,
                    memory_enabled=bool(getattr(config, "enable_channel_memory", True))))

            # Step 2. A channel turn CREATES its state and never rebuilds one: the rebuild exists
            # to reconstruct a transcript from conversations.replies, and the room the turn needs
            # — a shallow recent window plus the complete origin thread — is about to be pinned
            # by the stream builder instead. The state is still needed for what is not a
            # transcript — the
            # config overrides, the participants, the root author, the timeout flag.
            #
            # The gate's coalesced cohort is no longer merged into that state either. Every member
            # Slack propagated is IN the stream with its real header; the rest are quoted
            # post-breakpoint as awaiting-stream evidence (message_processor/channel_request.py),
            # which is what they are.
            if channel_turn:
                thread_state = await self.get_or_create_channel_thread_state(message)
            else:
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

            # Check if this thread had a previous timeout.
            #
            # A CHANNEL turn only DECIDES here; the words wait. The notice is permanent prose, and
            # posting it before the stream is built and the request admitted meant a history,
            # origin, timestamp or budget failure could leave "Picking up from here" standing in
            # the thread with no answer behind it — the turn's only visible output being a promise
            # it then broke. It is posted below, once those have succeeded (still
            # leased, still receipted). A DM has neither step, so it posts here as it always has.
            prior_timeout_owed = False
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
                    if channel_turn:
                        prior_timeout_owed = True
                    else:
                        await self._post_prior_timeout_notice(message, client, turn, thread_key)
                else:
                    self.log_debug(
                        f"Prior timeout in {thread_key} — clearing silently (turn may say nothing)")
                thread_state.had_timeout = False


            # Get thread config to determine model (user prefs + shared channel
            # settings; DMs simply have no channel_settings row → no-op there).
            # Spec §3b: on a channel turn the capability keys are the CHANNEL's, resolved the
            # same way the handler resolves them. The values here are not advisory — they pick
            # the model this turn trims against and decide whether an attachment is mounted for
            # the sandbox — so reading them requester-first would let whoever spoke change the
            # room's machine before the handler ever corrected it.
            #
            # Resolved ONCE for the whole turn and pinned on the TurnRuntime: the handlers and
            # every retry path below read the pin rather than the table, so a settings change
            # landing mid-turn cannot split one request across two profiles.
            thread_config = await pinned_thread_config(
                self, thread_state, message, channel_turn, turn=turn)
            
            # Update thread state with current model for token limit calculations
            thread_state.current_model = thread_config["model"]
            
            # Ensure the current requester is in the @mention roster
            user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
            if message.user_id:
                thread_state.participants.setdefault(
                    message.user_id,
                    user_real_name or (message.metadata.get("username") if message.metadata else None) or message.user_id,
                )
            # THIS TURN's channel steering — the stamp, never a fresh read. The gate stamped it
            # (gated turns) or the point-of-no-return above did (ungated ones), and the handlers
            # get the string passed down because every retry path re-enters them (context retry,
            # streaming fallback, MCP retry, the timeout retry below, the non-streaming fallback)
            # and a per-attempt read would give two attempts of ONE turn different rules.
            #
            # No stamp on a gated turn is an invariant failure, not a cue to go and read: reading
            # here is precisely the divergence this commit exists to remove — the gate would have
            # judged the message against bytes the reply never saw. So it is logged loudly and
            # this turn proceeds with NO steering, which is the one state that cannot disagree
            # with anything.
            steering = channel_steering.stamped(message)
            if steering is None:
                if gate_required:
                    self.log_warning(
                        f"No channel-steering stamp on a gated turn in {message.channel_id} — "
                        "answering without steering rather than reading it a second time")
                steering = channel_steering.EMPTY_SNAPSHOT
            channel_steering_text = steering.text

            # F32: ONE artifact sink for the whole turn. The timeout/MCP retries below re-enter
            # the text handler; a per-attempt sink would drop the container id of an attempt that
            # ran code interpreter and then failed, stranding the file it wrote in the sandbox.
            turn_artifacts: list = []

            # Steps 3-10 — the channel stream. Everything the turn can see, pinned: the
            # capability profile's hash and the tool schema version go IN so a build is
            # attributable to the machine that made it, and the origin thread's files and people
            # come back OUT as side state the tools address.
            stream = None
            if channel_turn:
                stream = await self._build_channel_turn_stream(
                    message, client, turn, h_pin, thread_config, thread_state)

            # Process any attachments (images, documents, and other files).
            # The CI setting must be the PER-THREAD one, resolved the same way the tools array
            # resolves it — a spreadsheet is only worth mounting when the sandbox that reads it
            # will actually be there.
            #
            # SPLIT on a channel turn [r3-4]: download and extraction happen now, the utility
            # model's summary waits until the admission estimate below has passed. A DM keeps the
            # combined sequencing verbatim.
            image_inputs, document_inputs, unsupported_files = await self._process_attachments(
                message, client, thinking_id,
                code_interpreter_enabled=thread_config.get(
                    'enable_code_interpreter', config.enable_code_interpreter),
                defer_document_summaries=channel_turn)

            # T2-10: a catch-up trigger carries EARLIER batched messages' already-processed image
            # parts and attachment failures (staged in _dispatch_pending_batch — re-downloading
            # here would be wasteful). Merge failures into unsupported_files so the notice below
            # acknowledges them, and fold the image parts into THIS turn's image_inputs so the
            # model can actually see them. The trigger's OWN images win the per-turn slots;
            # earlier-batch images fill what's left; any overflow is noted in the text.
            #
            # A CHANNEL turn does neither fold: the carried images ride POST-BREAKPOINT as their
            # own labeled block, where the assembler applies the same cap and writes the same
            # omission note (channel_request.build_batched_images). Folding them into this turn's
            # own parts here would put earlier messages' pictures inside the block that says "the
            # message you are answering", and would spend the trigger's image slots on them.
            batched_image_inputs = (message.metadata or {}).get("batched_image_inputs") or []
            batched_unsupported = (message.metadata or {}).get("batched_unsupported_files") or []
            if batched_unsupported:
                unsupported_files = list(unsupported_files) + list(batched_unsupported)
            batched_images_omitted = 0
            if batched_image_inputs and not channel_turn:
                image_cap = 10  # matches _process_attachments' max_images (utilities.py)
                room = max(0, image_cap - len(image_inputs))
                if room:
                    image_inputs = list(image_inputs) + list(batched_image_inputs[:room])
                batched_images_omitted = max(0, len(batched_image_inputs) - room)

            # Files that were accepted but couldn't be fetched/processed create an
            # obligation: use them or tell the user they failed — never answer as
            # if they were never attached.
            failed_files_notice_owed = None
            if unsupported_files:
                files_str = ", ".join(f"*{f['name']}*" for f in unsupported_files)
                unsupported_msg = self._build_failed_files_notice(unsupported_files)

                # Is there anything left to answer? On a DM the trigger IS the turn, so its own text
                # and attachments settle it. On a CHANNEL turn they do not [r4-5]: a catch-up is
                # answering earlier messages too, and their images ride the request as their own
                # post-breakpoint block rather than being folded into `image_inputs` above. A trigger
                # whose only content was a file that failed can therefore still owe an answer to a
                # whole cohort, and shortcutting here would leave every one of those senders in
                # silence.
                anything_else = bool((message.text and message.text.strip())
                                     or image_inputs or document_inputs)
                if channel_turn and not anything_else:
                    from message_processor.channel_request import cohort_sources_from_message
                    anything_else = bool(batched_image_inputs
                                         or cohort_sources_from_message(message))

                # If there's also text, images, or documents, continue processing those
                if anything_else:
                    unsupported_msg += "\n\nI'll process your text/image/document request now."
                    # The MIXED path continues on to generate the real reply, so — unlike the
                    # all-failed branch below, which RETURNS the notice for main.py to post — it
                    # must deliver this notice itself. Recording it only in thread state (as it
                    # used to) left the model believing it had acknowledged the failed files while
                    # the user saw nothing.
                    #
                    # A CHANNEL turn holds the words until the request has been admitted [r3-4],
                    # for the same reason the prior-timeout notice does: this promises "I'll
                    # process your request now", and a coverage, history, timestamp or budget
                    # failure after it would leave that promise standing alone as the turn's only
                    # visible output. A DM has no admission step, so it posts here as it always has.
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    if channel_turn:
                        failed_files_notice_owed = unsupported_msg
                    else:
                        await self._post_failed_files_notice(message, client, turn,
                                                             unsupported_msg)
                    # Add the unsupported files warning to conversation. DM/legacy only: a
                    # channel turn's transcript is Slack, and this notice is a real message in it
                    # — next turn's stream carries it with a finalized receipt. Writing it into
                    # ThreadState.messages would put it in a list the channel request never sends.
                    # The channel responder learns about the failure from the trigger supplement
                    # instead (ChannelTurnContext.failed_attachment_names).
                    if not channel_turn:
                        formatted_content = self._format_user_content_with_username(f"[File(s) not processed: {files_str}]", message)
                        self._add_message_with_token_management(thread_state, "user", formatted_content, db=self.db, thread_key=thread_key, message_ts=message_ts)
                        self._add_message_with_token_management(thread_state, "assistant", unsupported_msg, db=self.db, thread_key=thread_key)
                    # Continue processing if we have text or images
                else:
                    # Only unsupported files were uploaded, nothing else to process
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    if not channel_turn:
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
                # A channel turn's summaries do not exist yet (they wait for the admission
                # estimate) and its trigger supplement is rendered by the assembler, from the
                # SAME helper, once they do.
                if not channel_turn:
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

            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            if channel_turn:
                # A notice this turn owes is prose in the thread, so the answer belongs with it —
                # which settles the destination. Settle it BEFORE the request is assembled, not
                # after [r3-3]: admission pins the tool tuple and the suffix, so a destination
                # settled afterwards would leave the admitted request advertising
                # `set_reply_destination` and saying nothing about where the reply goes, while the
                # request actually sent carries `reply_destination=thread` and refuses that tool at
                # runtime. Bytes admitted == bytes sent only if the lock precedes the estimate.
                #
                # Settled unconditionally rather than on a confirmed send, because there is no send
                # yet. The cost if the notice then fails to post is a reply in the thread rather
                # than at top level; the cost of the other order is an estimate that measured a
                # different request.
                if (prior_timeout_owed or failed_files_notice_owed) and turn is not None:
                    turn.settle_structural_thread()
                # Steps 3 + 11 pre-flight. A channel turn does not trim: the request is the
                # pinned window, and there is nothing in it a trim could drop without answering
                # a different question. It is ADMITTED instead — measured whole, before the first
                # API call, and refused outright if the worst case will not fit.
                await self._admit_channel_request(
                    message, client, turn, thread_state, thread_config, thinking_id,
                    stream=stream, steering=steering, image_inputs=image_inputs,
                    file_inputs=file_inputs, document_inputs=document_inputs,
                    batched_image_inputs=batched_image_inputs,
                    batched_images_omitted=batched_images_omitted,
                    failed_attachment_names=tuple(
                        str(f.get("name") or "") for f in (unsupported_files or [])))
                # The window exists and the request fits. NOW the owed prose can be said: every
                # fail-closed condition that would have contradicted it is behind us, so these are
                # promises the turn is in a position to keep. Chronological order — the failure
                # that already happened, then the work about to start.
                if prior_timeout_owed:
                    await self._post_prior_timeout_notice(message, client, turn, thread_key)
                if failed_files_notice_owed:
                    notice_ts = await self._post_failed_files_notice(
                        message, client, turn, failed_files_notice_owed)
                    # What the responder's evidence may CLAIM depends on this landing [r4-4]. The
                    # request the model actually gets is assembled after this line, so it says
                    # "the user has been told" only when Slack confirmed the notice, and otherwise
                    # asks the reply to carry the acknowledgement itself. Admission already charged
                    # the longer of the two wordings, so neither outcome exceeds what was paid.
                    ctx = getattr(turn, "channel_turn_context", None)
                    if ctx is not None:
                        ctx.notice_delivery["failed_attachments"] = bool(notice_ts)
            else:
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
            # NOT gated on message.attachments: an image pasted as a LINK produces an image_input
            # with no attachment behind it, and that guard is why link-borne images never earned
            # a description. catalog_uploads reads the urls off the parts themselves.
            if image_inputs:
                self._schedule_async_call(image_catalog.catalog_uploads(
                    self, thread_key, image_inputs,
                    (message.metadata or {}).get("ts")))
            # [r5-2] The earlier queued messages' images, on a CHANNEL catch-up. The drain staged
            # them instead of describing them, because a description is a Responses call and this
            # turn was not admitted yet — here it is, past admission, at the same point the
            # trigger's own images are catalogued. Grouped per source message so each description
            # is still stored against the ts that actually carried the image.
            for carried_ts, carried_images in (message.metadata
                                               or {}).get("batched_catalog_uploads") or ():
                self._schedule_async_call(image_catalog.catalog_uploads(
                    self, thread_key, carried_images, carried_ts))

            response = await self._handle_text_response(
                user_content, thread_state, client, message, thinking_id, retry_count=0,
                artifacts_acc=turn_artifacts, turn=turn,
                channel_steering_text=channel_steering_text)

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
                #
                # DM/legacy only. A channel turn never appends its input here (the stream is the
                # input), so this pop would take somebody's actual last message off a reused or
                # pre-P2 ThreadState — a mutation of demoted state that nothing would replace.
                if (not channel_turn and thread_state.messages
                        and thread_state.messages[-1].get("role") == "user"):
                    thread_state.messages.pop()

                try:
                    response = await self._handle_text_response(
                        retry_content,
                        thread_state, client, message, thinking_id,
                        retry_count=1,
                        artifacts_acc=turn_artifacts, turn=turn,
                        # Same snapshot as the attempt that timed out — one responder turn.
                        channel_steering_text=channel_steering_text
                    )
                    self.log_info(f"Retry successful for {operation_type}")
                    return response

                except TimeoutError as retry_error:
                    self.log_warning(f"Retry also failed for {operation_type}: {retry_error}")
                    # Continue to error handling below
                except StaleSendSuppressed:
                    # The retry declined to post because the conversation moved on. Falling
                    # through to the timeout notice below would answer with an apology for a
                    # turn that had nothing to apologize for.
                    raise
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
                try:
                    # Deliberately NOT routed through _update_status. That helper SCHEDULES the
                    # edit as a detached task, and a suppression raised inside one cannot reach
                    # this turn — it would be swallowed wherever stray task exceptions go. This
                    # notice is terminal ("stopped waiting"), so it is awaited directly with the
                    # lease, like the other terminal notices. The rendering matches what
                    # _update_status would have produced for these arguments.
                    await client.update_message(
                        message.channel_id, thinking_id,
                        f"{config.error_emoji} {TIMEOUT_STATUS}",
                        lease=getattr(turn, "send_lease", None),
                        surface="timeout_notice",
                        receipts=getattr(turn, "receipt_ledger", None),
                        receipt_kind="finalized",
                        receipt_class="system_notice")
                    self.log_debug("Updated thinking message to show timeout")
                except StaleSendSuppressed:
                    raise  # the conversation moved on; a "stopped waiting" notice is noise now
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

            if turn is not None:
                turn.turn_error = "history_fetch_failed"

            if thinking_id and hasattr(client, 'update_message'):
                try:
                    await client.update_message(
                        message.channel_id, thinking_id,
                        f"{config.error_emoji} Couldn't load this conversation's history from Slack.",
                        lease=getattr(turn, "send_lease", None),
                        surface="history_error_notice",
                        receipts=getattr(turn, "receipt_ledger", None),
                        receipt_kind="finalized",
                        receipt_class="system_notice",
                    )
                except StaleSendSuppressed:
                    raise      # no notice, and the suppression is the record — see update_message
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
        except ChannelStreamError as e:
            # The channel window could not be built, or could not be sent. Every branch here is
            # FAIL-CLOSED by design: the alternative is answering a room we cannot see, which
            # reads to everyone in it as a bot that has lost the thread of the conversation.
            #
            # Three distinct conditions, three distinct things worth saying. They differ in what
            # the user can do — wait, say less, or nothing at all — so a shared "something went
            # wrong" would withhold the only actionable part. `HistoryFetchError` is caught above,
            # before this, keeping the DM notice it has always had; its channel sibling
            # `StreamTimestampError` is NOT a Slack failure and must not borrow that notice.
            code, notice = self._channel_stream_failure(e)
            if turn is not None:
                turn.turn_error = code
            self.log_error(f"Channel stream unavailable for {thread_key} ({code}): {e}")
            elapsed = time.time() - request_start_time
            self.log_info("")
            self.log_info("=" * 100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: {code.upper()} | "
                          f"Time: {elapsed:.2f}s")
            self.log_info("=" * 100)
            self.log_info("")

            if thinking_id and hasattr(client, 'update_message'):
                try:
                    await client.update_message(
                        message.channel_id, thinking_id,
                        f"{config.error_emoji} {notice['status']}",
                        lease=getattr(turn, "send_lease", None),
                        surface="channel_stream_error_notice",
                        receipts=getattr(turn, "receipt_ledger", None),
                        receipt_kind="finalized",
                        receipt_class="system_notice",
                    )
                except StaleSendSuppressed:
                    raise      # the conversation moved on; the suppression is the record
                except Exception:
                    pass

            return Response(type="error", content=notice["message"])
        except StaleSendSuppressed:
            # A suppression that reached process_message's boundary is still control flow.
            # Converting it into an error Response here would put "something went wrong" in a
            # channel where nothing did — and would do it INSTEAD of the answer the guard
            # correctly withheld, which is the worst of both outcomes.
            raise
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
                        await client.update_message(
                            message.channel_id, thinking_id, timeout_msg,
                            lease=getattr(turn, "send_lease", None),
                            surface="timeout_notice",
                            receipts=getattr(turn, "receipt_ledger", None),
                            receipt_kind="finalized",
                            receipt_class="system_notice")
                    except StaleSendSuppressed:
                        raise  # the conversation moved on; a timeout notice would be noise
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
                        await client.update_message(
                            message.channel_id, thinking_id, error_msg,
                            lease=getattr(turn, "send_lease", None),
                            surface="error_notice",
                            receipts=getattr(turn, "receipt_ledger", None),
                            receipt_kind="finalized",
                            receipt_class="system_notice")
                    except StaleSendSuppressed:
                        raise  # "something went wrong" about a turn where nothing did
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

    @staticmethod
    def _channel_stream_failure(error: ChannelStreamError):
        """The error code and the honest user notice for one fail-closed channel condition.

        Honest means: say what we cannot do, say whether waiting helps, and do not invent a cause.
        None of these is "something went wrong" — each is a specific state the runtime is in.

        The INVENTORY is not among them any more (§2f): a channel whose sweep has not settled,
        or which has never been swept at all, answers from what it can reach and declares that
        in its horizon. Nothing about the thread index refuses a turn.
        """
        from message_processor.channel_stream import (OriginFetchError,
                                                     StreamOverBudgetError,
                                                     StreamTimestampError)
        if isinstance(error, StreamTimestampError):
            # NOT a Slack problem, so it must not wear the Slack notice: a timestamp we cannot
            # parse means a record — in a payload or in one of our own rows — is malformed, and
            # retrying reads the same bad value. Saying "try again in a moment" would send the
            # user back for a second helping of the same failure.
            return "stream_data_invalid", {
                "status": "A timestamp in this channel's records doesn't parse.",
                "message": (f"{config.error_emoji} **Can't Place This Channel's History**\n\n"
                            "A timestamp in this channel's records doesn't parse, so I can't tell "
                            "which messages belong in view — and I won't answer from a picture "
                            "with an unexplained hole in it. Retrying won't clear this one; it "
                            "needs a fix on my side."),
            }
        if isinstance(error, OriginFetchError):
            # BRANCHED ON THE TYPE, never on the message. The origin thread is the conversation
            # the turn is actually in, so a partial one is not a smaller answer — it is a
            # different conversation. The periphery's own failures stay `history_fetch_failed`;
            # distinguishing them by inspecting an error string is what this type exists to
            # replace.
            return "origin_fetch_failed", {
                "status": "Couldn't read this thread completely.",
                "message": (f"{config.error_emoji} **Couldn't Read This Thread**\n\n"
                            "Slack didn't give me all of this thread, and I won't answer from "
                            "part of a conversation — the half I'm missing is as likely to be "
                            "the half that matters. Please try again in a moment."),
            }
        if isinstance(error, StreamOverBudgetError):
            return "stream_over_budget", {
                "status": "That's more than I can fit in one request.",
                "message": (f"{config.error_emoji} **Too Much For One Request**\n\n"
                            "This channel's history plus what's attached here is larger than I "
                            "can send in one go, so I stopped rather than answer from part of it. "
                            "A smaller attachment, or asking in a thread with fewer files, will "
                            "get through."),
            }
        return "history_fetch_failed", {
            "status": "Couldn't load this channel from Slack.",
            "message": (f"{config.error_emoji} **Couldn't Load This Channel**\n\n"
                        "Slack didn't give me this channel's recent history, so I can't answer "
                        "from a complete picture. Please try again in a moment."),
        }

    # ------------------------------------------------------------------ channel turn (spec §3)

    async def _build_channel_turn_stream(self, message: Message, client: BaseClient, turn,
                                         h_pin, thread_config: dict, thread_state):
        """Steps 3-10: pin the capability profile, build the stream, ingest the origin's side state.

        Steps 4-9 belong to `build_channel_stream` (the index drain, the one sidecar
        transaction, the fetch, the serialization, the actor-tail reconcile). What is
        left here is what a turn knows and the stream builder does not: which capability profile
        and which tool schemas this build was made against, and where the origin thread's files
        and people should be registered afterwards.
        """
        from message_processor.channel_request import (capability_profile_hash,
                                                       origin_participants_from_slice,
                                                       origin_slice_messages, tool_schema_version)
        from message_processor.channel_stream import build_channel_stream
        from message_processor.utilities import reach_tools_for

        team_id = getattr(client, "self_team_id", None) or ""

        # NO DRAIN HERE. `prepare_channel_turn` owns it, alone: the split-phase build awaits that
        # phase to completion before any fetch exists, which makes "the drain precedes all Slack
        # I/O" structural rather than a convention two call sites have to keep. Draining here as
        # well was a second wait on the same watermark and a second place the turn could fail,
        # buying an ordering guarantee the builder already provides.
        registry = getattr(client, "tool_registry", None)
        # NO budgets are constructed here. The BUILDER owns the shared absolute deadline and
        # builds all three itself — constructing them out here is what produced three
        # independently started windows, and a turn that could spend three times its budget.
        result = await self._channel_stream_call(
            build_channel_stream, message, client, turn, h_pin, thread_config, registry,
            team_id=team_id, origin_root_ts=message.thread_id,
            reach_tools=reach_tools_for(),
            capability_profile_hash=capability_profile_hash,
            tool_schema_version=tool_schema_version)
        # THE CARRIER, not a bare stream: the two booleans and the three page counts went to the
        # telemetry emitter inside the builder, and everything downstream takes `.stream`.
        stream = result.stream
        turn.channel_stream = stream
        turn.stream_build_present = True
        self.log_info(
            f"Channel stream for {message.channel_id}: {stream.message_count} message(s), "
            f"{stream.root_count} root(s), {stream.byte_count} bytes, "
            f"{stream.origin_count} origin message(s), H={stream.pinned.H}, "
            f"floor={stream.periphery_floor_ts or '(none)'}")

        # Step 10 — the origin thread's side state, from the pinned stream rather than a refetch.
        slice_messages = origin_slice_messages(stream, message.thread_id)
        actor_names = stream.pinned.actor_names
        await self.ingest_channel_origin_slice(
            thread_state, slice_messages, pinned_sidecars=stream.pinned.sidecars, db=self.db,
            actor_names=actor_names)
        turn.channel_origin_participants = origin_participants_from_slice(
            slice_messages, actor_names)
        return stream

    async def _channel_stream_call(self, build_channel_stream, message: Message,
                                   client: BaseClient, turn, h_pin, thread_config: dict,
                                   registry, *, team_id: str, origin_root_ts, reach_tools,
                                   capability_profile_hash, tool_schema_version):
        """The builder call itself, kept whole so the caller above stays readable."""
        return await build_channel_stream(
            client=client, db=self.db,
            team_id=team_id,
            channel_id=message.channel_id,
            h=h_pin.h, frontier=h_pin.frontier,
            # A BUILD INPUT, not a telemetry label: the origin thread is FETCHED, not selected
            # out of an existing window.
            origin_root_ts=origin_root_ts,
            # Frozen onto the pin so `serialize_stream` stays a pure function of it and the
            # horizon's reach clause names only tools this attempt actually exposes.
            reach_tools=reach_tools,
            capability_profile_hash=capability_profile_hash(thread_config),
            # The MCP manager is part of the effective tool surface, so the digest that claims to
            # identify this build's tools has to see it.
            tool_schema_version=tool_schema_version(registry, thread_config,
                                                    mcp_manager=getattr(self, "mcp_manager", None)),
            drain_timeout=getattr(config, "index_drain_timeout_seconds", None),
            barrier_context={"turn_id": getattr(turn, "turn_id", None),
                             "trigger_ts": (message.metadata or {}).get("ts")},
            # CV8 stream_render joins to its turn by these; the builder knows the window and the
            # hashes, and nothing else about whose turn it is. The origin root reaches the
            # emitter as the build input above — it is no longer passed twice under two names.
            turn_id=getattr(turn, "turn_id", None),
            trigger_ts=(message.metadata or {}).get("ts"))

    async def _admit_channel_request(self, message: Message, client: BaseClient, turn,
                                     thread_state, thread_config: dict,
                                     thinking_id: Optional[str], *, stream, steering,
                                     image_inputs: list, file_inputs: list,
                                     document_inputs: list, batched_image_inputs: list,
                                     batched_images_omitted: int,
                                     failed_attachment_names: tuple = ()) -> None:
        """Step 11's pre-flight: pin the turn's context, ADMIT the request, then summarize.

        The ordering is the whole point [r3-4]. The estimate is the last thing that happens before
        the first Responses API call of the turn — and the document summarizer IS a Responses API
        call, so it has to come after. Otherwise a turn that could never have been sent still
        spends a utility-model call per attached document to find that out.

        Each summary is then capped to what the estimate reserved for that document's raw text: the
        request was admitted at a size, and a summary is not allowed to exceed it afterwards.

        A catch-up turn's carried documents are finalized here too [r5-2], for the same reason and
        under the same gate: the queue drain that staged them could not summarize them, because it
        runs before this turn exists.
        """
        from message_processor.channel_request import (ChannelTurnContext, RequesterFacts,
                                                       canonical_files_from_stream,
                                                       cohort_sources_from_message,
                                                       merge_absent_source_files,
                                                       raise_if_over_budget)

        meta = message.metadata or {}
        cohort = cohort_sources_from_message(message)
        canonical = merge_absent_source_files(
            canonical_files_from_stream(stream), cohort, message.channel_id)
        root_author = getattr(thread_state, "root_author", None)
        root_uid = root_author[0] if isinstance(root_author, (tuple, list)) and root_author else None
        num_members = None
        peek = getattr(client, "get_cached_channel_context", None)
        if callable(peek):
            try:
                num_members = (peek(message.channel_id) or {}).get("num_members")
            except Exception:  # noqa: BLE001 — a member count never costs a turn
                num_members = None

        ctx = ChannelTurnContext(
            stream=stream,
            steering=steering,
            thread_config=thread_config,
            channel_id=message.channel_id,
            team_id=getattr(client, "self_team_id", None) or "",
            trigger_ts=meta.get("ts"),
            origin_thread_ts=message.thread_id,
            trigger_text=message.text or "",
            trigger_attachment_names=tuple(
                str(a.get("name") or a.get("filename") or "")
                for a in (message.attachments or []) if a),
            failed_attachment_names=tuple(n for n in (failed_attachment_names or ()) if n),
            image_parts=tuple(image_inputs or ()),
            file_parts=tuple(file_inputs or ()),
            document_inputs=tuple(document_inputs or ()),
            batched_image_parts=tuple(batched_image_inputs or ()),
            batched_images_omitted=int(batched_images_omitted or 0),
            cohort_sources=cohort,
            canonical_files=canonical,
            origin_participants=dict(getattr(turn, "channel_origin_participants", None) or {}),
            requester=RequesterFacts(
                user_id=message.user_id,
                real_name=meta.get("user_real_name") or meta.get("username"),
                email=meta.get("user_email"),
                timezone=meta.get("user_timezone") or "UTC",
                tz_label=meta.get("user_tz_label"),
                sender_type=meta.get("sender_type"),
                is_root_author=(None if not root_uid or not message.user_id
                                else message.user_id == root_uid)),
            channel_info=await self._build_channel_info(client, message.channel_id),
            num_members=num_members,
            wake_source=meta.get("wake_source"),
            queued_batch_size=meta.get("queued_batch_size"),
        )
        turn.channel_turn_context = ctx

        model = effective_request_model(thread_config)
        request, *_ = await self._assemble_channel_attempt(
            client, message, thread_state, turn, thread_config, model,
            thread_key=f"{thread_state.channel_id}:{thread_state.thread_ts}",
            with_estimate=True)
        estimate = request.estimate
        self.log_info(
            f"Channel request admission for {message.channel_id}: ~{estimate.total_tokens:,} of "
            f"{estimate.limit_tokens:,} usable input tokens {estimate.breakdown}")
        raise_if_over_budget(estimate, channel_id=str(message.channel_id),
                             counted_text=("" if estimate.fits else request.countable_text))
        await self.finalize_deferred_documents(
            list(document_inputs or []), client, message, thinking_id,
            reserves=estimate.document_reserves,
            # CV8: attach-time summarization is a Responses call on this turn, so it needs the
            # turn's attempt sink or "one model_response per attempt" is false for it.
            turn=turn,
        )
        # [r5-2] And the documents an earlier queued message brought, which the drain staged rather
        # than summarizing. NO reserve: they are not in this request — the estimate never charged
        # them — so there is no room to cap their summaries against. What they are here for is the
        # ledger row that makes read_document/mount_file able to reach them.
        carried_documents = (message.metadata or {}).get("batched_deferred_documents") or []
        if carried_documents:
            await self.finalize_deferred_documents(
                list(carried_documents), client, message, thinking_id, turn=turn)

    async def _post_failed_files_notice(self, message: Message, client: BaseClient, turn,
                                        text: str) -> Optional[str]:
        """"These files failed; I'll do the rest now" — the mixed-attachment notice.

        GUARDED like any first surface: it promises work that takes time, and a newer message
        during that window must not leave the promise standing on its own.

        Returns the notice's ts, or None if it did not land — which the channel caller feeds to the
        responder's evidence, since what that evidence may claim depends on it [r4-4].
        """
        try:
            notice_ts = await client.send_message(
                channel_id=message.channel_id,
                text=text,
                thread_id=message.thread_id,
                lease=getattr(turn, "send_lease", None),
                surface="failed_files_notice",
                receipts=getattr(turn, "receipt_ledger", None),
                receipt_class="system_notice",
            )
            # A notice that LANDED is a visible surface, so the answer belongs with it. Only on a
            # confirmed send: send_message swallows SlackApiError and returns None. A channel turn
            # has already settled this before its request was measured [r3-3], and settling is
            # idempotent, so this is the DM/legacy rule surviving unchanged.
            if notice_ts and turn is not None:
                turn.settle_structural_thread()
            return notice_ts
        except StaleSendSuppressed:
            # Not a notice failure: the guard declined to post it. Swallowed here the turn carries
            # on, and if the responder then ends silently the suppression is never recorded
            # anywhere — the one outcome that leaves us unable to tell a working guard from a
            # broken one.
            raise
        except Exception as notice_err:  # noqa: BLE001 — never fail the turn over the notice
            self.log_warning(f"Failed to post mixed-path failed-files notice: {notice_err}")
            return None

    async def _post_prior_timeout_notice(self, message: Message, client: BaseClient, turn,
                                         thread_key: str) -> None:
        """"My last answer never finished" — the recovery notice, wherever the caller runs it.

        GUARDED like any first surface. This notice is the turn's first visible words and real work
        follows it, so without the lease a newer message could leave "picking up from here" sitting
        alone in the thread. Notices were never exempt: the exemption is for CHROME (a thinking
        bubble), and this is prose.
        """
        notice_ts = await client.send_message(
            channel_id=message.channel_id,
            text="⚠️ Heads up — my last answer in this thread never finished. Picking up from here.",
            thread_id=message.thread_id,
            lease=getattr(turn, "send_lease", None),
            surface="prior_timeout_notice",
            receipts=getattr(turn, "receipt_ledger", None),
            receipt_class="system_notice",
        )
        # A notice that LANDED is a visible surface in the thread, so the answer belongs with it —
        # settle the destination there. Only on a confirmed send: send_message swallows
        # SlackApiError and returns None, and settling on a notice nobody saw would silently
        # withhold the destination choice and force the reply into a thread for no reason the user
        # could observe.
        if notice_ts and turn is not None:
            turn.settle_structural_thread()
        if notice_ts:
            self.log_info(f"Notified user about previous timeout in thread {thread_key}")
        else:
            # send_message swallows SlackApiError and returns None. Logging the send as a
            # notification hid the one path where the user is told nothing at all [r4-7].
            self.log_warning(f"Prior-timeout notice for thread {thread_key} did not post; the turn "
                             "continues, but nobody was told the last answer never finished")

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
                "⚠️ I hit an error catching up on the last few messages — please re-send.",
                receipt_kind="finalized",
                receipt_class="system_notice",
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
        # by being gate-routed for a ts the edit path registered, WITHOUT the surviving edit's
        # marker (the edit's own engine re-dispatch carries it and is kept). Addressed
        # (app_mention/DM) turns and ordinary different messages are not gate-routed and are
        # never touched; a genuinely different queued message has a different ts.
        marker_getter = getattr(client, "edit_dispatch_marker", None)
        if callable(marker_getter):
            kept = []
            for queued_msg in batch:
                try:
                    meta = queued_msg.metadata or {}
                    ts = meta.get("ts")
                    if meta.get("gate_required") and ts is not None:
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
        # The batch members whose turns this one is about to absorb. Each closed its own gate
        # attempt with `queued`, and until now nothing said WHICH later turn covered them — the
        # trigger is linked by parent_attempt_id (it is re-gated as the same Message), the rest
        # were simply folded in and lost. Staged here, where the sources are known, and written
        # by the successor turn, which alone knows what it became.
        #
        # A member may ALSO be carrying sources it inherited from an earlier drain and never
        # answered (it was queued again before its turn ran). Those travel with it: an ungated
        # member mints no attempt, so if its inheritance stopped here nothing downstream could
        # ever say who covered those messages.
        absorbed = []
        for absorbed_msg in batch[:-1]:
            absorbed.extend(participation_telemetry.take_staged_links(absorbed_msg))
            attempt_id = participation_telemetry.attempt_id_for(absorbed_msg)
            if attempt_id:
                absorbed.append(attempt_id)
        participation_telemetry.stage_queue_links(trigger, absorbed)
        # THE BATCH ITSELF, as typed sources for the trigger's gate. Without this the redispatch
        # gate sees one message and decides for all of them, so a no-wake throws away everything
        # that queued behind it — messages that are in Slack, and now in this thread's state, but
        # that nobody ever answers. The gate should judge the batch it is actually standing in
        # front of.
        #
        # And if any of them had ALREADY earned a turn, there is nothing left to judge: the
        # requirement is cleared and the responder decides what to say. (Both matter — the first
        # covers ambient messages nobody has ruled on, the second covers an @mention that had the
        # bad luck to queue behind one.)
        from .participation import source_from_message
        if routing_facts.absorb_owed_answer(trigger, batch[:-1]):
            # This turn is an ungated route now, so the trigger must stop carrying the attempt
            # from its earlier gated pass — that attempt is CLOSED, and left in place it would
            # attribute this turn's reactions to it and swallow this turn's terminal event
            # entirely. The detached id joins the absorbed sources so it still says what became
            # of it.
            detached = participation_telemetry.detach_attempt(trigger)
            if detached:
                participation_telemetry.stage_queue_links(trigger, [detached])
        if isinstance(trigger.metadata, dict) and len(batch) > 1:
            trigger.metadata["carried_gate_sources"] = tuple(
                source_from_message(m) for m in batch[:-1])
        # T2-10: earlier messages' image parts + attachment failures are collected here and
        # carried to the trigger turn — images so the model can actually SEE them (not just
        # their catalogued description), failures so a dropped file is acknowledged.
        batched_image_inputs: list = []
        batched_unsupported_files: list = []
        # [r5-2] A CHANNEL catch-up may not spend a single Responses call before its turn is
        # admitted, and BOTH of the batch's attachment side effects are Responses calls: the
        # document summary and the image description. So on a channel batch they are staged here
        # and carried into the admitted turn, which runs them once the request has been measured
        # and accepted. A DM has no admission step, so it keeps running both inline, verbatim.
        channel_batch = TextHandlerMixin._turn_surface(finished_message) == SURFACE_CHANNEL
        batched_deferred_documents: list = []
        batched_catalog_groups: list = []
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
                        channel_turn=channel_batch,
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
                                code_interpreter_enabled=batch_ci_enabled,
                                defer_document_summaries=channel_batch)
                            if q_document_inputs:
                                if channel_batch:
                                    # Staged, not summarized [r5-2]. The fold is skipped with it:
                                    # what it would render now is an excerpt (there is no summary
                                    # yet), into ThreadState.messages — a list the channel request
                                    # never sends. The document's real destination is its ledger
                                    # row, and that is written when the turn finalizes it.
                                    batched_deferred_documents.extend(q_document_inputs)
                                else:
                                    content = self._build_message_with_documents(content, q_document_inputs)
                            if q_image_inputs:
                                # Catalogue a durable description AND carry the raw parts to the
                                # trigger turn so the model actually sees the images (T2-10).
                                if channel_batch:
                                    batched_catalog_groups.append(
                                        ((queued_msg.metadata or {}).get("ts"),
                                         list(q_image_inputs)))
                                else:
                                    self._schedule_async_call(image_catalog.catalog_uploads(
                                        self, thread_key, q_image_inputs,
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
        # [r5-2] The two Responses calls this drain refused to make. The catch-up turn runs them
        # once its request has been admitted; if admission refuses it, they never run at all —
        # which is the contract, not a gap: a refused turn must not have spent anything.
        if batched_deferred_documents:
            trigger.metadata["batched_deferred_documents"] = batched_deferred_documents
        if batched_catalog_groups:
            trigger.metadata["batched_catalog_uploads"] = batched_catalog_groups
        # [r6-3] The FILE payloads of the absorbed messages. Slack may not have propagated them
        # into the window this turn fetches, and a cohort member the fetch missed would then have
        # its question answered with its own attachment unreadable — so the ids are carried off the
        # live events we are already holding and merged into the turn's canonical files. Channel
        # only: nothing on a DM turn reads them.
        if channel_batch and len(batch) > 1:
            from message_processor.channel_request import stage_cohort_file_payloads
            stage_cohort_file_payloads(
                trigger.metadata,
                [((queued_msg.metadata or {}).get("ts"), queued_msg.attachments or [])
                 for queued_msg in batch[:-1]])

        self.log_info(f"Draining {len(batch)} queued message(s) on {thread_key} into one catch-up turn")
        self._schedule_async_call(handler(trigger, client))

    async def cleanup(self):
        """Clean up resources and close clients."""
        self.log_info("Cleaning up MessageProcessor resources...")
        # Spec §5: the generic background set holds receipt-producing work (image share
        # resolution), so it goes first — and main.py drains it again, earlier, so that work
        # cannot outlive the receipt service. Idempotent: a drained set is a no-op.
        try:
            await self.drain_background_tasks()
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"Background task drain error: {e}")
        # F51: drain the ambient service FIRST — its workers call the OpenAI client + DB, so it
        # must finish (or be cancelled) before the client is closed under it.
        if getattr(self, "ambient_service", None):
            try:
                await self.ambient_service.shutdown()
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"Ambient service shutdown error: {e}")
        # Track 1: drain background channel-summary builds too — they also call the OpenAI client
        # + DB, so they must finish (or be cancelled) before the client closes under them.
        if getattr(self, "channel_summary_service", None):
            try:
                await self.channel_summary_service.shutdown()
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"Channel summary service shutdown error: {e}")
        if hasattr(self, 'openai_client') and self.openai_client:
            await self.openai_client.close()
        # Close thread manager resources if needed
        if hasattr(self.thread_manager, 'cleanup'):
            await self.thread_manager.cleanup()
        self.log_info("MessageProcessor cleanup completed")
    





















