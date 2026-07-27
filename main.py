#!/usr/bin/env python3
"""
Multi-Platform Chat Bot V2 - Main Entry Point
Supports multiple chat platforms with shared AI capabilities
"""
import sys
import signal
import asyncio
import argparse
import time
from typing import Optional
from config import config
from logger import log_session_start, log_session_end, main_logger
from message_processor.base import MessageProcessor
from message_processor import channel_steering, participation_telemetry, routing_facts
from message_processor.participation import (ParticipationEngine,
                                             resolve_participation_level)
from message_processor.stale_send_guard import (ConversationWatermarks,
                                                StaleSendSuppressed)
from message_processor.turn_runtime import TurnRuntime
from message_processor import thread_files
from base_client import BaseClient, Message

# Terminals whose visible surface is the RESPONDER'S OWN reply, and which therefore have a
# destination worth recording. `destination` means where that reply actually went (on
# `delivery_failed`, where it was ATTEMPTED — the send is what failed, not the choice).
#
# `detached` is deliberately absent. A detached producer — generate_image, a background job's
# status card — posts its own surface, targeting the trigger's thread directly rather than the
# responder's destination, so reporting the turn's destination for it would describe a reply
# that was never sent. `kind=detached` already records what happened; the destination column
# stays honest by saying nothing. Silences, reactions, queued turns and contract failures went
# nowhere at all.
_DELIVERED_KINDS = frozenset({"reply", "delivery_failed", "interrupted"})


class ChatBotV2:
    """Main application class for multi-platform chat bot"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor = None  # Will be initialized after client
        self.participation_engine = None  # Phase F; set in initialize()
        self._watermarks = ConversationWatermarks()
        self.cleanup_task = None
        self.running = False
        self.sigint_count = 0  # Track number of SIGINT received
        self.last_sigint_time = 0  # Track time of last SIGINT
        
    async def initialize(self):
        """Initialize the bot components"""
        main_logger.info(f"Initializing Chat Bot V2 for {self.platform}...")

        # Open the participation ledger HERE, before any Slack traffic. Built lazily it would
        # have put a mkdir and a file open inside the first gate call — on the hot path of the
        # decision the whole turn is waiting for — and retried it after every failure.
        participation_telemetry.initialize()

        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            main_logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Initialize platform-specific client
        if self.platform == "slack":
            from slack_client import SlackBot
            self.client = SlackBot(message_handler=self.handle_message)
            # Initialize processor with database from client
            self.processor = MessageProcessor(db=self.client.db)
            # Give the client a reference to the processor for thread state updates
            self.client.processor = self.processor
            # Phase F: judgment layer for unprompted channel participation.
            self.participation_engine = ParticipationEngine(self.processor.openai_client)
            # F52: expose the engine to the Slack facade's edit-reply path (message_events) so a
            # mention-added / meaning edit can SUPERSEDE the original message's in-flight
            # participation evaluation — the double-answer fix.
            self.processor.participation_engine = self.participation_engine
            # Move any legacy channel_settings.directives into the reserved policy row, before
            # any Slack traffic: from this build on NOTHING reads that column, so a workspace
            # that upgraded with rules in it would otherwise start quietly ignoring them.
            # Idempotent — a second start finds nothing to move.
            #
            # FATAL on failure, deliberately. The tempting alternative is to log and carry on,
            # and that is the worst outcome available here: the bot comes up looking healthy in a
            # channel whose "only speak up about deploys" is now unread, and behaves as though
            # nobody ever set a rule. Nothing downstream can detect that, and the operator finds
            # out from the noise. A process that refuses to start is visible in the first place
            # anyone looks, and the fix (a reachable database) is the same either way.
            # Each migration moves state OUT of a store this build no longer reads, so each is
            # fatal on failure for the same reason: the bot would come up looking healthy while
            # quietly ignoring something an operator set. `why` says what would be silently lost,
            # because that is the sentence whoever reads the log needs.
            for migrate, what, why in (
                (self.client.db.migrate_channel_directives_to_policy_async,
                 "channel directives",
                 "channel policies set before this release live in a column nothing reads any "
                 "more, so those channels' operator rules would be silently ignored"),
                (self.client.db.migrate_participation_levels_to_binary_async,
                 "participation levels",
                 "a channel still set to judicious or active has a level this build does not "
                 "recognise, and would fall back to mentions_only — quieter than its operator "
                 "chose, with nothing to say so"),
                (self.client.db.migrate_participation_prefs_to_policy_async,
                 "participation preferences",
                 "the backoff preference rows are no longer written or read, so a channel's "
                 "recorded 'react less here' would stop being obeyed without anyone clearing it"),
            ):
                migrated, failed, fatal = 0, 0, None
                try:
                    migrated, failed = await migrate()
                except Exception as e:  # noqa: BLE001 — reported below, then fatal
                    fatal = f"could not run ({type(e).__name__}: {e})"
                if fatal is None and failed:
                    fatal = f"failed for {failed} channel(s), {migrated} migrated"
                if fatal:
                    main_logger.error(
                        f"Migration of {what} {fatal}. Refusing to start: {why}. Fix the database "
                        f"and restart.")
                    sys.exit(1)
                    return  # sys.exit is stubbed in some harnesses; never fall through to a live bot
        else:
            main_logger.error(f"Unknown platform: {self.platform}")
            sys.exit(1)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        main_logger.info("Initialization complete")
    
    @property
    def watermarks(self) -> ConversationWatermarks:
        """THE process-wide owner of "what is the newest message this conversation has seen".

        Lazy so that an instance built without __init__ still has a real one. Every entry point
        into handle_message opens a lease against it, and a None here would mean a turn that
        silently opted out of the guard — the failure mode being a duplicate answer, which is
        exactly what this exists to prevent."""
        existing = self.__dict__.get("_watermarks")
        if existing is None:
            existing = self.__dict__["_watermarks"] = ConversationWatermarks()
        return existing

    async def _run_participation_gate(self, message: Message, client: BaseClient):
        """The gate, plus the one thing that must happen whether or not we speak.

        Deciding not to REPLY to a message is not the same as deciding to FORGET it. Everything
        that records a shared file — the document row, the image row, and therefore the catalog
        that `mount_file` and `read_document` resolve against — lived inside the turn, so a
        message we stayed quiet about had its attachments dropped on the floor for good.

        That is not a rare corner. It happened on the very first live run of this feature: four
        files were dropped into a thread a couple of seconds apart, the CSV among them arrived
        while the gate was still debouncing, its message was superseded by the next one, and the
        CSV simply ceased to exist as far as the bot was concerned. The model then — correctly —
        refused to build the report, because it could not read the numbers and would not invent
        them. The file was sitting right there in the channel.

        So: run the gate, and if it does not wake (returns None), catalog the files anyway. On a
        wake the turn does the richer job (extraction, summaries, visual descriptions) and we leave
        it alone — `save_document` is a plain INSERT, so cataloguing here as well would just
        duplicate the row. A message whose own attempt was superseded is covered too: its files are
        catalogued by that attempt, and its text reaches the model in the survivor's cohort.
        """
        decision = await self._gate_verdict(message, client)
        # The ONE place that answers "did the gate hand this on" — for the routing fact and for
        # the ledger alike. Written on every run (not only on a wake) so a queued redispatch's
        # second gate cannot inherit the first gate's answer.
        routing_facts.set_gate_woke(message, decision is not None)
        if decision is not None:
            participation_telemetry.mark_gate_woke(message)
        if decision is None and (message.attachments or []):
            self.processor._schedule_async_call(
                thread_files.catalog_unattended(self.processor, client, message))
        return decision

    async def _gate_verdict(self, message: Message, client: BaseClient):
        """The binary wake gate for UNPROMPTED channel messages: hard rails → debounce cohort →
        ONE model call → one bit.

        Returns a WakeDecision(wake=True) for the caller to run through the responder, and None
        for every terminal outcome the gate owns — a decision not to wake, a cohort collapse, an
        edit cancellation, an image-only cohort, a classifier failure, or a crash. All of those
        are silence, and each closes its own attempt here."""
        # Mint this attempt's id FIRST — before the engine check below, so even an engine-off
        # decline is a countable attempt with a start AND a terminal event, and so a redispatch
        # of this same Message object is recorded as a linked second attempt rather than
        # overwriting the first.
        attempt_id = participation_telemetry.begin_attempt(message)
        # True once a decision has been recorded, so the except-clause below can tell a gate that
        # failed to DECIDE from a gate that decided and then failed to hand off.
        decision_recorded = False
        try:
            channel_id = message.channel_id
            ts = message.metadata.get("ts") or message.thread_id
            # The canonical level, for slicing the ledger — not a prompt input. `judicious` was
            # the old default and is no longer a level at all; an absent value means this message
            # took a path that did not stamp one, so fall back to what the channel would resolve to
            # rather than naming a level that cannot exist.
            level = (message.metadata.get("participation_level")
                     or resolve_participation_level(None))
            # The denominator. Logged BEFORE any of the context-building I/O below, so a message
            # that dies to an exception on the way to the model is still counted as gated.
            #
            # `is_dm` is gone: a DM never reaches the gate, so the field was constant-false and
            # invited exactly the wrong reading — that DMs are in this population and never
            # judged. What replaced it is the posture that actually varies.
            gate_started_at = time.monotonic()
            participation_telemetry.gate_start(
                channel_id, ts, attempt_id=attempt_id,
                level=level,
                thread_reply=bool(ts and message.thread_id and message.thread_id != ts),
                # WHY this message is in scope at all, stamped at dispatch (addressing +
                # topology only). `thread_reply` above is the raw topology; this is the
                # routing system's own account of the same message.
                routing_posture=message.metadata.get("routing_posture"),
                name_hit=message.metadata.get("participation_name_hit") is True,
                sender_is_bot=message.metadata.get("participation_sender_bot") is True,
                sender_type=message.metadata.get("sender_type"),
                wake_source=message.metadata.get("wake_source"),
                # Whether the message CARRIES images, read off the attachments themselves. It is
                # a slicing fact about the traffic, not a prompt input: the gate never sees the
                # pictures, so nothing is stamped on the message for it to see.
                has_images=any((a or {}).get("type") == "image"
                               for a in (message.attachments or [])),
                has_attachments=bool(message.attachments or []),
                # A re-gate of the SAME message, behind the Phase Q queue. Its verdict may differ
                # from the first attempt's, and averaging the two as independent decisions would
                # count one message's judgment twice.
                redispatch=bool((message.metadata or {}).get(
                    participation_telemetry.PARENT_KEY)),
                # F52: the edit path stamps its own marker on the dispatched message, so an
                # edit-driven judgment is separable from a fresh one. (The edit CONTEXT itself
                # lives on the Slack facade, not the metadata — this marker is the only edit
                # fact that rides the message.)
                edit=bool((message.metadata or {}).get("edit_reply_marker")))

            # AFTER gate_start, deliberately. An engine-off attempt is a minted attempt with a
            # terminal event, and leaving it without a start made it a decline and an outcome
            # with nothing to divide by — in a ledger whose gate_start IS the denominator.
            engine = self.participation_engine
            if engine is None or not getattr(config, "enable_participation_engine", True):
                participation_telemetry.gate_declined(
                    channel_id, ts, cause="engine_off", attempt_id=attempt_id)
                participation_telemetry.finish_attempt(
                    message, "none", ended_by="gate", cause="engine_off")
                # engine off → unaddressed messages stay unanswered (mentions_only behavior)
                return None

            # F5 fix (b): register this message's ts as its conversation's newest BEFORE
            # the memory/topic awaits below — an older event delayed by that I/O must not
            # overwrite a newer event's debounce marker and win the race. F21: the marker
            # is conversation-scoped (message.thread_id is the thread root). F27: sender_id
            # scopes the top-level stream per author so different people's unrelated
            # top-level questions never collide.
            engine.note_arrival(channel_id, ts, message.thread_id, message.user_id)

            # THE channel-steering read for this turn — the only one. It is rendered once and
            # STAMPED on the message, so if this gate wakes, the responder builds its prompt from
            # this exact string rather than reading the database a second time. Between two reads
            # a person can edit the channel's policy or a tool can write a fact, and the two
            # halves of one turn would then obey different rules while each looked correct. A
            # redispatch or an edit re-evaluation runs this again and overwrites the stamp: that
            # is a new gate attempt judging a new moment, and it must not inherit an older view.
            steering = channel_steering.stamp(message, await channel_steering.load_snapshot(
                self.processor.db, channel_id,
                memory_enabled=bool(getattr(config, "enable_channel_memory", True))))

            # NOTHING ELSE IS GATHERED HERE, and the absence is the commit. The rich gate built a
            # pulse envelope, a thread tail, an addressee tail, a channel topic, a member count, a
            # recent-speakers line, a canvas catalogue, a capability inventory, a custom-emoji
            # shortlist and a channel narrative — half a dozen API calls and cache reads on the hot
            # path of a decision the whole turn waits for — to support judgments about addressee,
            # answerability and which emoji to place. The gate makes none of those judgments now,
            # so every one of those inputs was work spent on a question nobody asks.
            evaluation = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_id=message.user_id,
                sender_name=(message.metadata.get("user_real_name")
                             or message.metadata.get("username")),
                sender_type=message.metadata.get("sender_type"),
                channel_steering_text=steering.text,
                # Names and types, already summarized at dispatch. No pixels: the binary gate
                # does not look at images, so nothing is downloaded for it and nothing waits on it.
                attachments=message.metadata.get("participation_attachments"),
                client=client,
                thread_root_ts=message.thread_id,
                # The edit's OWN marker, so only the edit's attempt can claim the stashed
                # before/after text — the superseded original carries no marker and gets none.
                edit_marker=(message.metadata or {}).get("edit_reply_marker"),
                # A Phase-Q drain folds queued messages into this turn; they never had a debounce
                # window of their own, so they join the cohort here rather than being decided for
                # in absentia by whatever happened to arrive last.
                carried_sources=(message.metadata or {}).get("carried_gate_sources"),
                attempt_id=attempt_id,
            )
            gate_latency_ms = int((time.monotonic() - gate_started_at) * 1000)
            decision = evaluation.decision
            # A decline is TERMINAL, and its detail was already recorded by the engine — which
            # alone knows the survivor ts or the exception type. `classifier_error` is a decline
            # like any other now: the rich gate manufactured a fail-safe `ignore` and emitted it as
            # a decision, which scored a provider outage as the model choosing restraint.
            if evaluation.decline_cause or decision is None:
                main_logger.debug(
                    f"Wake gate: nothing to act on "
                    f"({evaluation.decline_cause or 'no_decision'}) — silent")
                participation_telemetry.finish_attempt(
                    message, "none", ended_by="gate", cause=evaluation.decline_cause,
                    gate_ms=gate_latency_ms, classifier_ms=evaluation.classifier_ms)
                return None

            participation_telemetry.gate_decision(
                channel_id, ts, wake=decision.wake, attempt_id=attempt_id,
                gate_ms=gate_latency_ms, classifier_ms=evaluation.classifier_ms,
                source_count=len(evaluation.sources),
                newest_source_ts=(evaluation.sources[-1].ts if evaluation.sources else None))
            decision_recorded = True

            if not decision.wake:
                # A DECISION to stay out, and the only gate-terminal outcome that earns the
                # `silence` label. It carries no reason — not to the log, not to the ledger, not to
                # anywhere: a gate reason was free prose about someone's message that ended up
                # forwarded into the responder's prompt, where it pre-argued the turn.
                #
                # And no `silence_reason` either. That eight-value enum belongs to the RESPONDER,
                # which can say why it chose to stay quiet after seeing everything; the gate knows
                # only that it did not open.
                participation_telemetry.finish_attempt(message, "silence", ended_by="gate")
                return None

            # Wake. Everything about what to SAY — words, a reaction instead, where it lands,
            # whether to change a setting, whether to say nothing after all — belongs to the
            # responder from here, which is the model that can actually see the conversation.
            #
            # The cohort rides along so the turn answers the whole burst rather than only its
            # newest fragment. Typed records, merged into the responder's real input — not prose
            # metadata describing messages it cannot see.
            if isinstance(message.metadata, dict) and evaluation.sources:
                message.metadata["gate_sources"] = evaluation.sources
            return decision

        except Exception as e:
            # Fail-safe stays silence: worst failure mode is a missed reply, never spam.
            main_logger.warning(f"Participation gate error: {e}; staying silent")
            # A crash is silence the user cannot distinguish from judgment, which is exactly why
            # it has to be countable: a rising error rate would otherwise look like a bot that
            # has simply decided to talk less.
            #
            # WHICH failure, though, is not the same question. Before the verdict exists the gate
            # failed to decide — a decline. After it exists the gate decided and the ACTION blew
            # up (a reaction, a backoff write, the handoff), and filing that as a classifier
            # decline would report the model as unable to judge when its judgment is on record
            # two lines above.
            cause = "action_error" if decision_recorded else "error"
            participation_telemetry.gate_declined(
                message.channel_id, (message.metadata or {}).get("ts") or message.thread_id,
                cause=cause, attempt_id=attempt_id, detail=type(e).__name__)
            participation_telemetry.finish_attempt(
                message, "none", ended_by="gate", cause=cause,
                detail=type(e).__name__)
            return None

    @staticmethod
    def _produced_visible_output(response, turn) -> bool:
        """F38: did this turn actually do the thing the 👀 claimed?

        The claim is honored by anything the user can SEE: text that went out, a deliberate
        response reaction, or a tool that owns its own surface (a background job's status
        card, a detached image). It is NOT honored by silence, by an error notice, or by a
        turn that got queued behind another — in all three the bot claimed work and then
        produced none of it, so the eye comes back off."""
        # Both facts are read FIRST, before any outcome-shaped exit below. A turn that placed a
        # reaction and then errored, was interrupted, or failed to deliver its text still left an
        # emoji in the room — retracting the 👀 there takes back a mark the reader can still see,
        # and the failure is not the reaction's fault. (A queued turn cannot reach either: it ran
        # no tools, so nothing could have set them.)
        if turn is not None and (turn.visible_action_committed
                                 or getattr(turn, "reaction_committed", False)):
            return True   # a detached producer owns a surface, or our emoji is on the message
        if response is None or response.type in ("error", "queued"):
            return False
        meta = response.metadata or {}
        if meta.get("interrupted"):
            # The turn died partway through and all that reached the thread was an apology
            # for dying. It claimed work and delivered none of it — `posted` is True only
            # because a Slack surface exists to carry the notice.
            return False
        if meta.get("terminal_action") == "no_reply":
            # A silent turn is no longer an empty one: its siblings ran. A reaction that IS the
            # answer honors the claim, and so does a background job whose card is already in the
            # room (a detached surface is caught by the turn check above, but the job's fact
            # lives on the response).
            return bool(meta.get("response_reaction_committed")
                        or meta.get("background_job_started")
                        or (turn is not None and getattr(turn, "reaction_committed", False)))
        if meta.get("reaction_only") or meta.get("background_job_started"):
            return True
        posted = meta.get("posted")
        if posted is None:  # non-streaming handlers can't know; derive from the outcome
            posted = bool(response.type == "text"
                          and (meta.get("streamed") or (response.content or "").strip()))
        return bool(posted)

    @staticmethod
    def _stale_terminal_kind(response, turn, message) -> str:
        """What the room saw on a turn the stale guard refused.

        `stale_suppressed` only when the refusal is the WHOLE story — the model produced or
        intended words, the guard declined to create their first surface, and nothing else of
        this turn is visible. It is deliberately not `silence` (nobody chose it, and there is no
        silence reason to give) and never `delivery_failed` (Slack was intentionally not called,
        so nothing failed).

        When something else IS visible the louder fact wins the label and the suppression rides
        beside it as `reply_stale_suppressed`: a detached producer owns a surface, or a reaction
        is on the message — from the gate or from this turn."""
        meta = (response.metadata or {}) if response is not None else {}
        if (turn is not None and getattr(turn, "visible_action_committed", False)) \
                or meta.get("background_job_started"):
            return "detached"
        # The gate no longer places reactions, so there is no gate stamp to consult here: every
        # reaction on the message is this turn's own.
        if (meta.get("response_reaction_committed") is True
                or (turn is not None and getattr(turn, "reaction_committed", False))):
            return "reaction_only"
        return "stale_suppressed"

    @staticmethod
    def _classify_visible_action(response, turn) -> str:
        """The ONE outcome the room saw, as a single label for the telemetry ledger.

        Deliberately a pure function rather than an expression inside handle_message: what
        counts as "the bot said something" is genuinely subtle — a detached image posts itself
        and returns an empty Response, an interrupted turn posts nothing but an apology, a queued
        turn was never run — and every one of those looks like "posted nothing" if read literally.
        Filing them that way would put the bot's most visible turns in the same bucket as its
        broken ones, which is the reverse of the truth.

        Shares its judgment with _produced_visible_output (which answers the narrower F38
        question: was the 👀 honored?) but does not collapse to a boolean: `silence` and `empty`
        are identical in the room and opposite in meaning — one is the model choosing, the other
        is the contract failing — so the ledger has to keep them apart.

        There is no gate reaction to account for any more. The rich gate could place an emoji
        before the responder ran, so a responder that then vetoed with no_response_needed had not
        actually left the room silent, and this function needed to be told. The binary gate puts
        nothing in the room, so every reaction on the message belongs to this turn and the function
        can see all of its own inputs — which is why every one of these labels is testable as a
        table.
        """
        if response is None:
            # The responder handed back nothing at all. Not the gate's `none` (which means a
            # decision path ended with nothing to show) — this is the responder contract
            # breaking, and the caller adds detail=no_response_object to say which way.
            return "empty"
        meta = response.metadata or {}
        if response.type == "queued":
            return "queued"      # never ran; another turn owns this conversation
        if response.type == "error":
            return "error"
        if meta.get("interrupted"):
            return "interrupted"  # died partway; the thread got an apology, not an answer
        if meta.get("terminal_action") == "no_reply":
            # Silence ends the WORDS, not the turn's effects — the terminal tool runs alongside
            # its siblings now, so a turn can end without words and still have posted a picture,
            # a status card or a reply in another thread. Those are checked FIRST: they are the
            # loudest thing in the room, and a turn that visibly produced something is not a
            # silence no matter what the model called at the end.
            if (turn is not None and getattr(turn, "visible_action_committed", False)) \
                    or meta.get("background_job_started"):
                return "detached"
            # A no-reply turn that committed a reaction did not stay silent — the emoji WAS the
            # answer, and the room saw one. Filing it as silence understated how often the bot
            # participates without words, which is the very rate this ledger exists to measure.
            # The model's stated reason still rides the event, separately.
            #
            if (meta.get("response_reaction_committed") is True
                    or (turn is not None and getattr(turn, "reaction_committed", False))):
                return "reaction_only"
            return "silence"     # the model chose it, via the terminal tool
        if meta.get("reaction_only"):
            return "reaction_only"
        posted = meta.get("posted")
        if posted is None:  # non-streaming handlers can't know; derive from the outcome
            posted = bool(response.type == "text"
                          and (meta.get("streamed") or (response.content or "").strip()))
        elif posted is False and (response.content or "").strip():
            # The model wrote an answer and Slack did not take it. That is the OPPOSITE of
            # silence and it used to be filed as `reply`, because the content was non-empty —
            # so every delivery outage read as the bot talking normally.
            return "delivery_failed"
        if posted:
            return "reply"
        if (turn is not None and getattr(turn, "visible_action_committed", False)) \
                or meta.get("background_job_started"):
            # A producer that owns its own surface — generate_image posts the picture, a
            # background job posts its status card — so the Response is empty BY DESIGN.
            return "detached"
        # Posted nothing and never called the terminal tool: a contract violation, and the
        # single most important thing in this ledger to keep apart from a chosen silence.
        return "empty"

    async def _rescue_sandbox_images(self, response, client: BaseClient, message: Message,
                                     post_thread_id: str) -> int:
        """Post images the model made as sandbox ingredients but never turned into anything.

        create_image_asset deliberately does not publish: its image is a component of some
        larger artifact (a slide in a deck, a layer in a composite), and posting the raw
        ingredient alongside the finished thing would be noise. But if the turn published
        NOTHING, the model generated images and then failed to use them — and the container
        they live in is gone within 20 minutes. Handing them over beats losing them silently.

        Returns the number of images that actually reached the thread (F38: a rescued image
        IS visible output, so a turn that delivered one has honored its 👀).
        """
        assets = (response.metadata or {}).get("sandbox_image_assets") or []
        if not assets:
            return 0
        from message_processor.image_delivery import publish_image
        main_logger.warning(
            f"Turn published no artifacts but created {len(assets)} sandbox image(s) — "
            "posting them directly rather than letting them die with the container")
        thread_key = f"{message.channel_id}:{message.thread_id}"
        posted = 0
        for asset in assets:
            image_data = asset.get("image_data")
            if image_data is None:
                continue
            try:
                await publish_image(
                    processor=self.processor, client=client, channel_id=message.channel_id,
                    thread_id=post_thread_id, thread_key=thread_key, image_data=image_data,
                    checklist=None, generation_id=None,
                    prompt=asset.get("enhanced_prompt") or asset.get("prompt") or "",
                    db=getattr(self.processor, "db", None),
                    thread_manager=self.processor.thread_manager,
                    message_ts=(message.metadata or {}).get("ts"),
                    provenance_tool="create_image_asset",
                )
                posted += 1
            except Exception as e:
                main_logger.error(f"Sandbox image rescue failed: {e}", exc_info=True)
        return posted

    async def handle_message(self, message: Message, client: BaseClient):
        """Handle incoming message from any platform.

        The whole flow sits in one `try/finally` that starts BEFORE the participation gate.
        The gate itself can raise, the thinking indicator can raise, and the turn can be
        cancelled outright — and each of those used to leave a gate attempt with a
        `gate_start` and no ending, which in the ledger is indistinguishable from a decision
        that has not been written yet. `abort_attempt` closes any attempt that got this far
        without a terminal event; it is a no-op for the overwhelming majority that did.

        The stale-send lease opens on the FIRST executable line — before the gate, before any
        await. That placement is the definition of an "admitted" message: taking it later would
        leave a window where this turn is already running and the conversation's watermark does
        not know it. It is released in the same finally.
        """
        # Also records this message as its conversation's newest inbound (message_processor.
        # stale_send_guard). Nothing is dropped after this point, so the watermark and the set
        # of turns that exist stay in step.
        lease = self.watermarks.begin_turn(message)
        try:
            # The binary wake gate: for UNPROMPTED channel messages (every ordinary message at
            # level `on`, a bare-name message at `mentions_only`) it decides whether the responder
            # runs at all. Nothing is posted before it, and nothing is decided by it beyond that bit.
            gate_required = message.metadata.get("gate_required") is True
            if gate_required:
                try:
                    decision = await self._run_participation_gate(message, client)
                finally:
                    # AFTER the gate, and in a `finally`: the attempt is minted INSIDE that
                    # await, so a cancellation or a raise on the way back out would otherwise
                    # leave an attempt that absorbed queued messages and never said so. Emitting
                    # here is safe in both directions — it writes nothing while the attempt does
                    # not exist yet, and the ungated turn's links are written later still, at the
                    # conversation lock (see MessageProcessor.process_message).
                    participation_telemetry.emit_queue_links(message, gate_required=True)
                if decision is None:
                    return
                # Nothing is copied off the decision. It carries one bit, and the cohort it judged
                # was already stamped on the message by the gate — as typed records that become
                # real responder input, not prose metadata about messages the model cannot see.

            # WHERE the reply goes is the turn's own state now, opened here and settled by the
            # model on the one route where there is a choice (set_reply_destination). The gate's
            # `placement` verdict is deliberately not consulted: it is formed before the answer
            # exists, and only the writer knows whether what it just wrote belongs in the room or
            # in a thread. See message_processor/destination_tools.py.
            turn = TurnRuntime.for_message(
                message,
                channel_post_allowed=bool(message.metadata.get("channel_post_allowed")))
            # The turn carries the lease from here on: it is already threaded to every path
            # that can post, including ToolContext.turn.
            turn.send_lease = lease
            post_thread_id = turn.resolve_reply_target(message)

            # Phase Q: if this conversation is mid-processing, the message is about to be
            # queued (not answered now) — skip the thinking indicator so nothing flashes.
            # Advisory peek only: losing the race just means a briefly-posted indicator
            # that the queued short-circuit below deletes.
            # `is True` (not truthiness): same hardening as the wake gate — mocked or
            # malformed managers must never silently suppress the indicator.
            thread_manager = getattr(self.processor, "thread_manager", None)
            already_processing = (
                thread_manager is not None
                and hasattr(thread_manager, "is_thread_processing")
                and thread_manager.is_thread_processing(message.thread_id, message.channel_id) is True
            )

            # Send initial thinking indicator. A turn that may say nothing, one headed for the
            # top level of a channel, and one still waiting for the model to choose where its
            # reply goes all show NOTHING here — `turn.progress_enabled` is the single answer to
            # "may this turn render speculative chrome", and each of those three would otherwise
            # put a placeholder somewhere the answer might not arrive.
            thinking_id = None
            if not already_processing and turn.progress_enabled:
                thinking_id = await client.send_thinking_indicator(
                    message.channel_id,
                    post_thread_id
                )
                # Batched catch-up turn (drained queue): make the status say so.
                batch_size = message.metadata.get("queued_batch_size", 0)
                if isinstance(batch_size, int) and batch_size > 1:
                    catch_up = f"Catching up on {batch_size} messages..."
                    try:
                        if thinking_id and hasattr(client, "update_message"):
                            await client.update_message(  # unleased-ok: chrome — a placeholder/status write, never answer text
                                message.channel_id, thinking_id,
                                f"{config.circle_loader_emoji} {catch_up}"
                            )
                        elif thinking_id is None and hasattr(client, "set_assistant_status"):
                            # Status-only DM indicator: the composer status carries it.
                            await client.set_assistant_status(
                                message.channel_id, post_thread_id, status=catch_up
                            )
                    except Exception as e:
                        main_logger.debug(f"Catch-up status update failed: {e}")

            response = None
            try:
                # Immediately before the call, never earlier: a turn that died on the way here
                # (indicator, queue peek) has to stay distinguishable from one the model ran.
                participation_telemetry.mark_responder_started(message)
                response = await self.processor.process_message(message, client, thinking_id,
                                                                turn=turn)

                # The model may have chosen the destination DURING the turn (an eligible
                # top-level trigger starts unselected). Re-read it from the turn — the one place
                # that knows — so the fallback send, the footer guard and the pulse below all
                # agree with where the handler actually posted.
                post_thread_id = turn.resolve_reply_target(message)

                # Delete thinking indicator (but not if streaming was used — it's already the
                # response — and not when a ProgressChecklist owns the thinking message, F4).
                if (thinking_id and response
                        and not response.metadata.get("streamed")
                        and response.metadata.get("checklist") is None):
                    await client.delete_message(message.channel_id, thinking_id)  # unleased-ok: teardown — removing a surface can never be a stale answer
                elif thinking_id and not response:
                    await client.delete_message(message.channel_id, thinking_id)  # unleased-ok: teardown — removing a surface can never be a stale answer

                # Handle the response
                if response:
                    if response.type == "queued":
                        # Phase Q: the message joined its conversation's pending queue and
                        # will be answered by the in-flight turn's batched catch-up. Nothing
                        # to post (the indicator, if any, was already deleted above).
                        main_logger.debug(f"Message queued behind in-flight turn for {message.channel_id}:{message.thread_id}")
                    elif response.type == "text":
                        # Reaction-only turns (react tool, empty text) post no message at all
                        if not (response.content or "").strip():
                            main_logger.debug("Empty text response (reaction-only) — nothing to post")
                        # If streaming was used, the message is already displayed
                        elif not response.metadata.get("streamed"):
                            # Send raw content: send_message formats for the platform itself
                            # (messaging.py). Pre-formatting here double-ran the converter, and
                            # format_text is NOT idempotent (italic runs before bold, so a second
                            # pass turns **bold** → *bold* → _bold_ and renders as italic).
                            # F8: attach the settings-footer chrome to the message itself (same
                            # as the native-streaming path's stopStream blocks) instead of a
                            # separate trailing post. Suppressed for top-level channel placement
                            # (same rule as the separate footer below) and when block-building is
                            # unavailable — those fall back to maybe_post_response_footer.
                            footer_blocks = None
                            if (post_thread_id is not None
                                    and hasattr(client, "attachable_footer_blocks")):
                                try:
                                    footer_blocks = client.attachable_footer_blocks(
                                        message.channel_id, response.metadata.get("model"))
                                except Exception as e:
                                    main_logger.debug(f"Footer block build failed: {e}")
                                    footer_blocks = None
                            send_meta = {}
                            # The reply is going out NOW, in this place: the destination stops
                            # being a preference and becomes a fact about Slack. The streaming
                            # paths lock when they bind their surface; this is the same moment
                            # for a non-streamed reply, which main.py posts itself.
                            turn.lock_destination()
                            # The stale guard's last chance on this path: the lease refuses
                            # rather than posting if the conversation moved on while the model
                            # was writing. Raises StaleSendSuppressed, caught below.
                            sent_ts = await client.send_message(
                                message.channel_id,
                                post_thread_id,
                                response.content,
                                blocks=footer_blocks,
                                meta_out=send_meta,
                                lease=lease,
                            )
                            # Honest accounting: the ACTUAL send result decides `posted` (a
                            # failed send must not burn the hourly unprompted quota).
                            if isinstance(response.metadata, dict):
                                response.metadata["posted"] = bool(sent_ts)
                                # Only stand the separate footer down when the chrome ACTUALLY
                                # rode the message (a split/too-long reply doesn't attach it, so
                                # the separate footer post must still happen).
                                if sent_ts and send_meta.get("footer_attached"):
                                    response.metadata["footer_attached"] = True
                            # F7: persist tool-use provenance keyed on the reply's real ts.
                            if sent_ts:
                                self.processor._persist_tool_provenance(
                                    message.channel_id, sent_ts,
                                    f"{message.channel_id}:{message.thread_id}",
                                    (response.metadata or {}).get("tool_provenance"))
                        # Phase 7: Configure footer under the response (channels only, any
                        # member can open settings). Native-streamed responses attach the
                        # chrome to the message itself on stopStream (footer_attached
                        # metadata makes this call a no-op); everything else falls back to
                        # this separate trailing message.
                        # Best-effort: a cosmetic footer must never break message handling.
                        # Skipped for top-level placement — it would land as ANOTHER top-level
                        # message and read as spam.
                        # No footer under an empty turn (F2 no_reply / reaction-only) — there is
                        # no message for it to sit under.
                        # Also skip when the reply didn't actually post (posted is explicitly
                        # False) — a footer under a message that never landed reads as orphaned.
                        if (hasattr(client, "maybe_post_response_footer")
                                and post_thread_id is not None
                                and (response.content or "").strip()
                                and (response.metadata or {}).get("posted") is not False):
                            try:
                                await client.maybe_post_response_footer(message, response)
                            except Exception as e:
                                main_logger.debug(f"Response footer skipped: {e}")

                        # F32: upload any code-interpreter artifacts AFTER the answer lands, so the
                        # thread reads "explanation, then the chart" rather than the reverse. Runs
                        # even for an empty-text turn (a chart that speaks for itself). Strictly
                        # best-effort: the reply is already posted and an upload failure must never
                        # turn a delivered answer into an error.
                        artifact_containers = (response.metadata or {}).get("artifact_containers") or []
                        # Only hang files under an answer that actually landed. If a non-empty reply
                        # failed to post, a chart arriving alone with no explanation is worse than
                        # no chart. (A files-only turn has empty content by design — still publish.)
                        reply_landed = (response.metadata or {}).get("posted") is not False
                        files_only = not (response.content or "").strip()
                        published = []
                        if artifact_containers and (reply_landed or files_only):
                            try:
                                from message_processor.artifacts import publish_artifacts
                                # Whole-phase bound: the answer is already visible, but this still
                                # holds the turn open, and a wedged upload must not stall the next
                                # message in the thread.
                                published = await asyncio.wait_for(
                                    publish_artifacts(
                                        openai_client=self.processor.openai_client,
                                        client=client,
                                        channel_id=message.channel_id,
                                        # B2: artifacts always thread. post_thread_id is None on a
                                        # top-level channel reply, so thread off message.thread_id
                                        # instead — the chart hangs under the answer, never top-level.
                                        thread_id=message.thread_id,
                                        thread_key=f"{message.channel_id}:{message.thread_id}",
                                        container_ids=artifact_containers,
                                        db=getattr(self.processor, "db", None),
                                        message_ts=(message.metadata or {}).get("ts"),
                                        container_manager=getattr(
                                            self.processor, "container_manager", None),
                                        # F35: files the model MOUNTED are ingredients the user
                                        # already owns — never publish them back, even byte-copied.
                                        suppress_digests=(response.metadata or {}).get(
                                            "mounted_digests") or [],
                                    ),
                                    timeout=config.artifact_publish_timeout,
                                )
                                if published:
                                    main_logger.info(
                                        f"Published {len(published)} artifact(s) to the thread")
                                    # F38: a chart or a deck visibly landed. On an empty-text turn
                                    # (code interpreter answering with the file itself) the Response
                                    # says posted=False, and without this the end-of-turn settle
                                    # would read that as silence and retract the 👀 from a turn that
                                    # plainly delivered.
                                    turn.visible_action_committed = True
                            except asyncio.TimeoutError:
                                main_logger.error("Artifact publishing timed out — reply already posted")
                            except Exception as e:
                                main_logger.error(f"Artifact publishing failed: {e}", exc_info=True)
                        elif artifact_containers:
                            main_logger.warning(
                                "Reply did not post — suppressing its artifacts (a file with no "
                                "answer above it reads as a bug)")

                        # F34: create_image_asset mounts an image into the sandbox as an
                        # INGREDIENT, so it is deliberately not published — the deck or composite
                        # built from it is. But if the turn ended having published nothing at all,
                        # the model made images and then failed to use them, and they would die
                        # with the container. A silent no-output turn is the worst failure mode
                        # here, so hand them over rather than lose them.
                        if not published:
                            # B2: rescued sandbox images always thread — pass message.thread_id, not
                            # post_thread_id (None on a top-level channel reply).
                            rescued = await self._rescue_sandbox_images(response, client, message,
                                                                        message.thread_id)
                            if rescued:
                                turn.visible_action_committed = True  # F38: an image did land
                    elif response.type == "error":
                        # Send error message
                        # An error notice is terminal, and on a turn with no thinking surface
                        # it is the room's only word from us — so it is guarded like any first
                        # surface. Refusal raises, and the handler below records the
                        # suppression instead of the error.
                        await client.handle_error(
                            message.channel_id,
                            message.thread_id,
                            response.content,
                            lease=lease
                        )

                # Close the attempt with what the room actually SAW. Deliberately not folded into
                # the contract check below, which is additionally gated on a live pulse — a
                # delivery-side condition that has nothing to do with whether the outcome is worth
                # counting. `finish_attempt` is a no-op unless this turn came from the gate and is
                # still open, which is what keeps mentions, DMs and direct continuations out of a
                # ledger documented as gate attempts — and what keeps a turn the gate already
                # closed (a react verdict that fell through to nothing) from closing twice.
                meta = (response.metadata or {}) if response is not None else {}
                # No gate reaction to account for any more: the gate places nothing in the room, so
                # every reaction on this message came from this turn.
                kind = self._classify_visible_action(response, turn)
                # A destination is reported only for a delivered reply on a turn that actually
                # settled one. Computed once — three fields read it, and they must agree.
                _records_destination = bool(
                    turn is not None and kind in _DELIVERED_KINDS
                    and getattr(turn, "destination_selected", False))
                participation_telemetry.finish_attempt(
                    message, kind,
                    ended_by="responder",
                    # The reason rides even when the reaction made this a reaction_only: WHY the
                    # model declined to use words is the same question either way. Recorded
                    # VERBATIM — it is one of eight declared values, and the ledger's job here is
                    # to report the model's own account, never to correct or second-guess it.
                    silence_reason=(meta.get("silence_reason")
                                    if meta.get("terminal_action") == "no_reply" else None),
                    # A detached producer that owns a surface AND an error afterwards is one turn
                    # with two outcomes. `kind` keeps the error (it is the more actionable half),
                    # so the surface has to be recorded beside it or it vanishes from the analysis.
                    detached_started=(True if kind == "error" and turn is not None
                                      and getattr(turn, "visible_action_committed", False)
                                      else None),
                    # Was there an emoji in the room at the end of this turn? Explicit on EVERY
                    # responder terminal, true or false, not only on replies: paired with
                    # `silence_reason` it is what makes a model that said `reacted_instead` and
                    # then placed no reaction VISIBLE as a mismatch. Neither fact is corrected
                    # against the other — the disagreement is the measurement.
                    # Read from the turn, which the react tool stamps as the emoji lands, so
                    # this is true on a reply that also reacted and on a reaction-only turn —
                    # not just on the one branch that happened to build the metadata field.
                    reaction_visible=bool(
                        meta.get("response_reaction_committed") is True
                        or (turn is not None and getattr(turn, "reaction_committed", False))),
                    # WHICH model wrote it. Per-user and per-thread overrides mean two rows in
                    # this ledger can come from different models, and a reply-quality comparison
                    # that pools them describes neither. The footer path reads the same key.
                    model=meta.get("model"),
                    # An `empty` with no Response object at all is a different contract failure
                    # from one that returned an empty Response, and only this tells them apart.
                    detail="no_response_object" if response is None else None,
                    # WHERE the reply went and WHO decided. The old `placement` field said
                    # "thread" for every turn that produced no words at all — a silence has no
                    # placement — so half the column described a decision nobody made. Written
                    # only on terminals that delivered the responder's own reply (see
                    # _DELIVERED_KINDS) AND only when a destination was actually settled: an
                    # unsettled turn has no answer to this question, and `default` is a real
                    # answer meaning "the model was asked and did not choose".
                    destination=(turn.reply_destination if _records_destination else None),
                    destination_source=(turn.destination_source if _records_destination
                                        else None),
                    # Only when true: a column that is false on nearly every row costs bytes on
                    # every line to say nothing. The miss is the event worth finding. It rides
                    # with `source=default`, always — they are two halves of one fact.
                    destination_contract_miss=(
                        True if _records_destination
                        and getattr(turn, "destination_contract_miss", False) else None),
                    chars=len(response.content or "") if kind == "reply" else None,
                )

                # Post-delivery bookkeeping (F2: accounted AFTER delivery, honest posted). The
                # reply TALLY that used to live here is gone with the unprompted counter; what
                # remains is the contract check — a turn that posted nothing and did not call the
                # terminal no-reply tool is a violation worth catching.
                if (response and message.channel_id and not message.channel_id.startswith("D")
                        and getattr(client, "channel_pulse", None) is not None):
                    terminal = (response.metadata or {}).get("terminal_action")
                    if terminal == "no_reply":
                        main_logger.info(
                            f"no_response_needed — no reply posted "
                            f"(reason: {response.metadata.get('silence_reason')})")
                    else:
                        posted = response.metadata.get("posted")
                        if posted is None:
                            # Non-streaming handlers can't know; derive from the outcome.
                            posted = bool(
                                response.type == "text"
                                and (response.metadata.get("streamed")
                                     or (response.content or "").strip()))
                        if (not posted and response.type == "text"
                                and not (response.content or "").strip()
                                and not response.metadata.get("reaction_only")):
                            # Bare empty text without the terminal tool: contract violation.
                            # Fail-safe silence, no re-prompt this phase.
                            main_logger.warning(
                                "Empty text response without a terminal action — posting nothing")

            except StaleSendSuppressed as stale:
                # NOT an error, and it must never reach the handler below — that one logs an
                # exception, files `error_unhandled`, and posts an apology. Nothing went wrong
                # here: a newer message arrived while this turn was writing, so its answer was
                # deliberately not created. The room sees nothing, which is the point.
                main_logger.info(
                    f"Stale send suppressed on {message.channel_id}: {stale}")
                participation_telemetry.stale_send(
                    message.channel_id, (message.metadata or {}).get("ts"),
                    attempt_id=participation_telemetry.attempt_id_for(message),
                    last_seen_ts=stale.last_seen_ts,
                    observed_latest_ts=stale.observed_latest_ts,
                    scope=stale.scope[0] if stale.scope else None,
                    surface=stale.surface,
                    guard_mode=getattr(turn, "guard_mode", None) if turn is not None else None)
                # Any speculative chrome this turn put up has to come down with it — a
                # "Thinking…" bubble left over a reply that will never arrive is worse than
                # either outcome on its own.
                if thinking_id:
                    try:
                        await client.delete_message(message.channel_id, thinking_id)  # unleased-ok: teardown — removing a surface can never be a stale answer
                    except Exception as cleanup_error:  # noqa: BLE001
                        main_logger.debug(f"Stale-suppression cleanup failed: {cleanup_error}")
                if hasattr(client, "clear_assistant_status"):
                    try:
                        await client.clear_assistant_status(message.channel_id, post_thread_id)
                    except Exception as cleanup_error:  # noqa: BLE001
                        main_logger.debug(f"Stale-suppression status clear failed: {cleanup_error}")
                participation_telemetry.finish_attempt(
                    message,
                    # Something else may still be visible: a reaction the gate or the responder
                    # placed, or a detached producer's own surface. Those turns are not silent,
                    # and the suppression rides as a separate fact rather than overwriting the
                    # louder one.
                    self._stale_terminal_kind(response, turn, message),
                    ended_by="stale_guard",
                    last_seen_ts=stale.last_seen_ts,
                    observed_latest_ts=stale.observed_latest_ts,
                    scope=stale.scope[0] if stale.scope else None,
                    surface=stale.surface,
                    guard_mode=getattr(turn, "guard_mode", None) if turn is not None else None,
                    reply_stale_suppressed=True,
                    model=(response.metadata or {}).get("model") if response is not None else None,
                )
            except Exception as e:
                main_logger.error(f"Error handling message: {e}", exc_info=True)
                # Closed FIRST, before the two best-effort awaits below: either of them can fail
                # too, and the terminal event must not depend on our apology getting out.
                #
                # DELIVERY WINS. Everything between the send and the close above is bookkeeping —
                # provenance persistence, artifact publishing, the footer — and any of it can
                # raise long after Slack has the reply. Filing that as `error_unhandled` deletes
                # a delivered answer from the talk rate and files it under failures, so a bad
                # afternoon in the artifact path would read as a bot that stopped answering. The
                # room saw the reply; the ledger says so, and says the turn was not clean.
                delivered = response is not None and (response.metadata or {}).get("posted")
                if delivered:
                    participation_telemetry.finish_attempt(
                        message,
                        self._classify_visible_action(response, turn),
                        ended_by="responder", post_delivery_error=type(e).__name__)
                else:
                    participation_telemetry.finish_attempt(
                        message, "error_unhandled", ended_by="responder",
                        detail=type(e).__name__)

                # Delete thinking indicator on error — best-effort; a failed delete
                # must never swallow the user-facing notice below.
                if thinking_id:
                    try:
                        await client.delete_message(message.channel_id, thinking_id)  # unleased-ok: teardown — removing a surface can never be a stale answer
                    except Exception as delete_error:
                        main_logger.error(f"Failed to delete thinking indicator: {delete_error}")

                # Fixed, friendly notice — the raw exception stays in the logs only.
                try:
                    await client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        "⚠️ **Something Went Wrong**\n\n"
                        "I hit a snag finishing that response. Please try again in a moment.",
                        lease=lease
                    )
                except StaleSendSuppressed as stale:
                    # Nothing goes out. The turn already failed; posting an apology into a
                    # conversation that has moved on would be a second wrong thing. This is the
                    # one place a suppression is neither re-raised nor terminal: the turn's
                    # outcome is already recorded as an error just above, so the refusal of the
                    # NOTICE is a separate fact and is written down as one.
                    main_logger.info("Error notice suppressed — the conversation moved on")
                    participation_telemetry.stale_send(
                        message.channel_id, (message.metadata or {}).get("ts"),
                        attempt_id=participation_telemetry.attempt_id_for(message),
                        last_seen_ts=stale.last_seen_ts,
                        observed_latest_ts=stale.observed_latest_ts,
                        scope=stale.scope[0] if stale.scope else None,
                        surface=stale.surface,
                        guard_mode=getattr(turn, "guard_mode", None) if turn is not None else None)
                except Exception as notify_error:
                    main_logger.error(f"Failed to send error notice: {notify_error}")
            finally:
                # F38: settle the work claim. Runs in `finally` so an exception, a cancellation,
                # or an early return can't strand a 👀 on a message the bot then ignored.
                try:
                    await turn.settle_ack(
                        client, self._produced_visible_output(response, turn))
                except Exception as ack_error:  # noqa: BLE001
                    main_logger.debug(f"Ack settle failed: {ack_error}")

                # Native-streamed replies don't trip Slack's "auto-clear status on reply"
                # (it keys on chat.postMessage, not chat.stopStream), so a status-only turn
                # left the working bubble spinning forever (user report 2026-07-10).
                # Explicit best-effort clear. Skipped for queued turns — their status
                # belongs to the in-flight request that will answer them.
                # Also skipped for background image gen (background_owns_status): the job owns
                # the status-only progress surface and clears it on completion — clearing here
                # would blank it the instant the turn returns (Codex finding 8).
                # F38: and skipped entirely when progress was deferred — there is no status to
                # clear, and clearing one we never set would auto-open the thread to say so.
                if (thinking_id is None
                        and turn.progress_enabled
                        and not (response is not None and response.type == "queued")
                        and not (response is not None and response.metadata.get("background_owns_status"))
                        and hasattr(client, "clear_assistant_status")):
                    try:
                        await client.clear_assistant_status(message.channel_id, post_thread_id)
                    except Exception as clear_error:
                        main_logger.debug(f"Assistant status clear failed: {clear_error}")
        finally:
            participation_telemetry.abort_attempt(message)
            # Release the scope hold LAST. An entry survives while any lease in its scope is
            # open, so a newer turn that finishes early cannot erase the watermark an older,
            # still-running turn is about to read.
            lease.close()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals - double Ctrl-C for force exit"""
        import os
        import time

        # Handle SIGINT (Ctrl-C) with double-press for force exit
        if signum == signal.SIGINT:
            current_time = time.time()
            
            # If second Ctrl-C within 2 seconds, force exit
            if self.sigint_count > 0 and (current_time - self.last_sigint_time) < 2.0:
                main_logger.warning("Force exit requested (double Ctrl-C) - terminating immediately!")
                
                # Show active threads for debugging
                import threading
                active_threads = threading.enumerate()
                if len(active_threads) > 1:
                    main_logger.warning(f"Active threads at force exit: {len(active_threads)}")
                    for thread in active_threads:
                        if thread.name != "MainThread":
                            main_logger.warning(f"  - {thread.name} (daemon={thread.daemon})")
                
                # Force exit without cleanup
                os._exit(1)
            
            self.sigint_count += 1
            self.last_sigint_time = current_time
            
            if self.sigint_count == 1:
                main_logger.info(f"Received signal {signum}, attempting graceful shutdown...")
                main_logger.info("Press Ctrl-C again within 2 seconds to force exit")
                # Schedule shutdown on the event loop
                asyncio.create_task(self.shutdown())
            else:
                main_logger.warning("Shutdown already in progress... Press Ctrl-C again to force exit")
        else:
            # Handle other signals normally
            main_logger.info(f"Received signal {signum}, shutting down...")
            # Schedule shutdown on the event loop
            asyncio.create_task(self.shutdown())
    
    async def start_cleanup_task(self):
        """Start background task for periodic cleanup"""
        async def cleanup_worker():
            from croniter import croniter
            import datetime

            try:
                # Validate cron expression
                cron = croniter(config.cleanup_schedule, datetime.datetime.now())
                main_logger.info(f"Cleanup schedule configured: {config.cleanup_schedule} (cron format)")
                main_logger.info(f"Cleanup will remove threads older than {config.cleanup_max_age_hours} hours")
            except Exception as e:
                main_logger.error(f"Invalid cron expression '{config.cleanup_schedule}': {e}")
                main_logger.info("Falling back to daily at midnight (0 0 * * *)")
                cron = croniter("0 0 * * *", datetime.datetime.now())

            while self.running:
                try:
                    # Calculate next run time
                    next_run = cron.get_next(datetime.datetime)
                    now = datetime.datetime.now()
                    seconds_until_next = (next_run - now).total_seconds()

                    # Log when next cleanup will occur
                    if seconds_until_next > 3600:
                        main_logger.info(f"Next cleanup scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({seconds_until_next/3600:.1f} hours from now)")
                    else:
                        main_logger.info(f"Next cleanup scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({seconds_until_next/60:.1f} minutes from now)")

                    # Sleep until next scheduled time
                    await asyncio.sleep(seconds_until_next)

                    if self.running:
                        main_logger.info(f"Running scheduled cleanup (removing threads older than {config.cleanup_max_age_hours} hours)...")
                        # Convert hours to seconds for the cleanup function
                        max_age_seconds = config.cleanup_max_age_hours * 3600
                        await self.processor.thread_manager.cleanup_old_threads(max_age=max_age_seconds)

                        # Also clean up old modal sessions (24 hours old)
                        if hasattr(self.processor, 'db') and self.processor.db:
                            await self.processor.db.cleanup_old_modal_sessions_async(hours=24)
                            main_logger.info("Cleaned up old modal sessions")

                            # F7: sweep aged tool-use provenance rows (no FK cascade —
                            # PRAGMA foreign_keys is never enabled, so these need their own
                            # age sweep, same as documents).
                            try:
                                self.processor.db.delete_old_tool_usage(
                                    days=config.tool_usage_retention_days)
                            except Exception as e:
                                main_logger.debug(f"Tool-usage sweep skipped: {e}")

                            # Sweep aged document-extraction rows: SLIM (not delete) — the derived
                            # bulk (summary/page_structure/metadata) is nulled while the Slack ref
                            # row is kept, so read_document and rebuilds re-extract on demand and a
                            # file behind a compaction boundary stays resolvable indefinitely.
                            try:
                                self.processor.db.delete_old_documents(
                                    days=config.document_retention_days)
                            except Exception as e:
                                main_logger.debug(f"Document sweep skipped: {e}")

                            # F51: sweep expired ambient artifacts (retention). The sweep also
                            # deletes their late-artifact addenda and returns the affected thread
                            # keys; an ACTIVE warm thread still holds an in-memory summary head
                            # carrying the expired note, so mark each for refresh (fail-soft per
                            # thread — a marking failure must not break the sweep loop).
                            try:
                                # Off the event loop: the sweep deletes rows and can block.
                                # SQLite is threadsafety-3 (serialized), so cross-thread use
                                # is safe.
                                swept_keys = await asyncio.to_thread(
                                    self.processor.db.delete_expired_ambient_artifacts,
                                    days=config.ambient_artifact_retention_days)
                                tm = getattr(self.processor, "thread_manager", None)
                                if tm is not None and hasattr(tm, "mark_needs_refresh"):
                                    for thread_key in (swept_keys or []):
                                        try:
                                            tm.mark_needs_refresh(thread_key)
                                        except Exception as mark_err:
                                            main_logger.debug(
                                                f"mark_needs_refresh failed for {thread_key}: {mark_err}")
                            except Exception as e:
                                main_logger.debug(f"Ambient-artifact sweep skipped: {e}")

                            # F32: reap code-interpreter containers for threads that have gone
                            # quiet. The containers themselves idle-expired long ago (20-minute
                            # API ceiling), so this is mostly dropping their rows — a revived
                            # thread just gets a fresh container on its next turn.
                            try:
                                cm = getattr(self.processor, "container_manager", None)
                                if cm is not None:
                                    await cm.reap()
                            except Exception as e:
                                main_logger.debug(f"Container reap skipped: {e}")

                            # Scheduled database backup. Until now backup_database()
                            # was only ever called by the one-time migrations, so a
                            # steady-state bot took no backups at all despite the
                            # documented "automatic backups with 7-day retention".
                            # Untagged on purpose: cleanup_old_backups() (a tail-call
                            # of backup_database) prunes untagged dailies at 7 days.
                            # Isolated — a failed backup must never kill the cleanup
                            # worker or the bot.
                            try:
                                # Off the event loop: conn.backup() can take hundreds of ms
                                # and would otherwise freeze the bot for its duration.
                                await asyncio.to_thread(self.processor.db.backup_database)
                                main_logger.info("Scheduled database backup complete (7-day retention)")
                            except Exception as e:
                                main_logger.error(f"Scheduled database backup FAILED: {e}")

                        stats = self.processor.get_stats()
                        main_logger.info(f"Cleanup complete. Stats: {stats}")
                except asyncio.CancelledError:
                    main_logger.info("Cleanup task cancelled")
                    break
                except Exception as e:
                    main_logger.error(f"Error in cleanup task: {e}")
                    # Wait 5 minutes before retrying on error
                    await asyncio.sleep(300)

        self.cleanup_task = asyncio.create_task(cleanup_worker())
        main_logger.info("Started cleanup task")
    
    async def run(self):
        """Run the bot"""
        log_session_start()

        fatal = False
        try:
            await self.initialize()
            self.running = True

            # Start cleanup task
            await self.start_cleanup_task()

            # F51: start the ambient-memory service on the running loop and resume any
            # interrupted work (durable pending rows). Best-effort — a failure here must not
            # stop the bot from serving.
            svc = getattr(self.processor, "ambient_service", None)
            if svc is not None:
                try:
                    svc.start()
                    self._ambient_recover_task = asyncio.create_task(svc.recover_pending())
                    self._ambient_recover_task.add_done_callback(
                        lambda t: t.exception() and main_logger.warning(
                            f"Ambient recover error: {t.exception()}"))
                except Exception as e:
                    main_logger.warning(f"Ambient service start skipped: {e}")

            # MCP startup health probe (informational; runs in the background so
            # a slow server can't delay boot). Strong ref so it can't be GC'd.
            if getattr(self.processor, "mcp_manager", None) and self.processor.mcp_manager.has_mcp_servers():
                self._mcp_probe_task = asyncio.create_task(self.processor.mcp_manager.health_probe())
                self._mcp_probe_task.add_done_callback(
                    lambda t: t.exception() and main_logger.warning(f"MCP health probe error: {t.exception()}"))

            # Start the client (blocks)
            main_logger.info(f"Starting {self.platform} bot...")
            if self.client:
                try:
                    await self.client.start()
                except asyncio.CancelledError:
                    main_logger.info("Bot client cancelled during shutdown")
                    pass

        except KeyboardInterrupt:
            # Graceful: Ctrl-C / signal-driven shutdown is a clean stop, exit 0.
            main_logger.info("Received keyboard interrupt")
        except Exception as e:
            # An UNEXPECTED fatal error — NOT a signal-driven shutdown (those schedule
            # shutdown() via the handler and return cleanly, or raise CancelledError which
            # the client.start() block above absorbs). Still run the graceful shutdown in
            # `finally`, then exit non-zero so a supervisor (systemd/docker) sees a failure
            # instead of a clean exit 0 that reads as "stopped on purpose".
            main_logger.critical(f"Fatal error — bot is exiting: {e}", exc_info=True)
            fatal = True
        finally:
            await self.shutdown()

        if fatal:
            sys.exit(1)

    async def shutdown(self):
        """Shutdown the bot gracefully"""
        if not self.running:
            return

        self.running = False
        main_logger.info(f"Shutting down {self.platform} bot...")

        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # F1: cancel/await in-flight background image generations BEFORE the Slack client
        # stops — otherwise the client tears down mid-upload and the jobs fail noisily.
        tm = getattr(self.processor, "thread_manager", None) if self.processor else None
        if tm is not None and hasattr(tm, "cancel_generations"):
            try:
                await tm.cancel_generations(timeout=5.0)
            except Exception as e:
                main_logger.warning(f"Error cancelling background generations: {e}")
        # F30: same for in-flight background deep-research jobs.
        if tm is not None and hasattr(tm, "cancel_research_jobs"):
            try:
                await tm.cancel_research_jobs(timeout=5.0)
            except Exception as e:
                main_logger.warning(f"Error cancelling background research jobs: {e}")

        # F51: drain the ambient artifact workers BEFORE the Slack client stops — they use the
        # client's reusable download session for image/file capture, so tearing the client down
        # first would fail in-flight ambient downloads. Idempotent: processor.cleanup() calls
        # shutdown() again below (a no-op once drained).
        svc = getattr(self.processor, "ambient_service", None) if self.processor else None
        if svc is not None and hasattr(svc, "shutdown"):
            try:
                await svc.shutdown()
            except Exception as e:
                main_logger.warning(f"Error draining ambient workers: {e}")

        # Stop the client (this should interrupt any stuck operations)
        if self.client:
            try:
                await self.client.stop()
            except Exception as e:
                main_logger.warning(f"Error stopping client: {e}")

        # Clean up resources
        try:
            if self.processor:
                stats = self.processor.get_stats()
                main_logger.info(f"Final stats: {stats}")
                # Clean up processor resources
                await self.processor.cleanup()
        except Exception as e:
            main_logger.warning(f"Error during processor cleanup: {e}")

        # Give aiohttp sessions and pending coroutines a moment to clean up
        await asyncio.sleep(0.5)

        # Cancel any remaining tasks that might be lingering
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        if tasks:
            main_logger.warning(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            # Wait briefly for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)

        log_session_end()
        main_logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ChatGPT Slack Bot")
    parser.add_argument(
        "--platform",
        choices=["slack"],
        default="slack",
        help="Chat platform to use (default: slack)"
    )

    args = parser.parse_args()

    # Create and run bot
    bot = ChatBotV2(platform=args.platform)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())