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
from typing import Any, Dict, Optional
from config import GUIDANCE_TRUNCATION_CHARS, config
from logger import log_session_start, log_session_end, main_logger
from message_processor.base import MessageProcessor
from message_processor import participation_telemetry, routing_facts
from message_processor.participation import (ParticipationEngine,
                                             render_capabilities_line)
from message_processor.people_tools import format_people_summary
from message_processor.turn_runtime import TurnRuntime
from message_processor import thread_files
from base_client import BaseClient, Message


class ChatBotV2:
    """Main application class for multi-platform chat bot"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor = None  # Will be initialized after client
        self.participation_engine = None  # Phase F; set in initialize()
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
        else:
            main_logger.error(f"Unknown platform: {self.platform}")
            sys.exit(1)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        main_logger.info("Initialization complete")
    
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

        So: run the gate, and if it stays silent (returns None), catalog the files anyway. When
        the gate falls through to a reply (respond or react_and_respond), the turn does the richer
        job (extraction, summaries, visual descriptions) and we leave it alone — `save_document` is
        a plain INSERT, so cataloguing here as well would just duplicate the row.
        """
        verdict = await self._gate_verdict(message, client)
        # The ONE place that answers "did the gate hand this on" — for the routing fact and for
        # the ledger alike. It used to be three separate marks inside the gate, one per
        # fall-through branch, which is three chances for a new branch to forget. Written on
        # every run (not only on a wake) so a queued redispatch's second gate cannot inherit the
        # first gate's answer.
        routing_facts.set_gate_woke(message, verdict is not None)
        if verdict is not None:
            participation_telemetry.mark_gate_woke(message)
        if verdict is None and (message.attachments or []):
            self.processor._schedule_async_call(
                thread_files.catalog_unattended(self.processor, client, message))
        return verdict

    async def _gate_verdict(self, message: Message, client: BaseClient):
        """Phase F gate for UNPROMPTED channel messages: hard rails → debounce →
        ONE engine call → act. Returns a verdict for the fall-through outcomes —
        'respond', 'react_and_respond', and a backoff that requests an explicit
        settings change — which the caller runs through the response loop (a
        react_and_respond has ALREADY placed its gate reaction here). Every
        terminal outcome (ignore / react / a fully-handled backoff / superseded /
        any failure) is handled here and returns None so the caller stays silent."""
        # Mint this attempt's id FIRST — before the engine check below, so even an engine-off
        # decline is a countable attempt with a start AND a terminal event, and so a redispatch
        # of this same Message object is recorded as a linked second attempt rather than
        # overwriting the first.
        attempt_id = participation_telemetry.begin_attempt(message)
        # True once a verdict has been recorded, so the except-clause below can tell a gate that
        # failed to DECIDE from a gate that decided and then failed to ACT.
        verdict_recorded = False
        try:
            channel_id = message.channel_id
            ts = message.metadata.get("ts") or message.thread_id
            level = message.metadata.get("participation_level") or "judicious"
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
                has_images=bool(message.metadata.get("participation_images")),
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

            pulse = getattr(client, "channel_pulse", None)

            # F5 fix (b): register this message's ts as its conversation's newest BEFORE
            # the memory/topic awaits below — an older event delayed by that I/O must not
            # overwrite a newer event's debounce marker and win the race. F21: the marker
            # is conversation-scoped (message.thread_id is the thread root). F27: sender_id
            # scopes the top-level stream per author so different people's unrelated
            # top-level questions never collide.
            engine.note_arrival(channel_id, ts, message.thread_id, message.user_id)

            # Pacing is the classifier's judgment, and only its judgment. F17 removed the
            # hourly-cap hard rail; the count that fed it survived as a signal line until it
            # was removed too — it never prevented a misfire, and a rate number in a prompt
            # that is otherwise about WHO a message is for only competed with that judgment.
            name_hit = message.metadata.get("participation_name_hit") is True

            channel_activity = None
            if pulse is not None:
                channel_activity = pulse.render_envelope(
                    channel_id, exclude_thread_ts=None,
                    max_lines=config.channel_pulse_envelope_max,
                ) or None

            memory_facts = []
            try:
                if getattr(config, "enable_channel_memory", True) and self.processor.db:
                    memory_facts = await self.processor.db.get_channel_memory_async(channel_id)
            except Exception:
                memory_facts = []

            channel_topic = None
            num_members = None
            fetch_ctx = getattr(client, "get_channel_context", None)
            if fetch_ctx:
                try:
                    ctx = await fetch_ctx(channel_id)
                    channel_topic = (ctx or {}).get("topic") or None
                    num_members = (ctx or {}).get("num_members")
                except Exception:
                    channel_topic = None
                    num_members = None

            # F29: people signal — member count + recently active names (from the pulse ring)
            # — so the classifier can resolve WHO a message (and its "you") is aimed at.
            recent_names = []
            if pulse is not None:
                try:
                    recent_names = pulse.recent_speakers(channel_id)
                except Exception:
                    recent_names = []
            channel_people = format_people_summary(num_members, recent_names)

            is_thread_reply = bool(ts and message.thread_id and message.thread_id != ts)
            # F11: inventory of the assistant's own tools/data sources so the classifier
            # can weigh whether it is well-suited to answer an open question to the room.
            capabilities = render_capabilities_line(getattr(self.processor, "mcp_manager", None))
            # F36: canvases are channel furniture, like the topic. Cached per channel.
            channel_canvases = []
            try:
                from message_processor import canvas_tools
                channel_canvases = [c["title"] for c in
                                    await canvas_tools.build_catalog(client, channel_id)]
            except Exception:  # noqa: BLE001 — never cost the gate a verdict
                channel_canvases = []

            # C3: workspace custom emoji as EXTRA classifier choices — ONLY when there is no
            # REACTION_EMOJIS allowlist (a set allowlist is the exact hard constraint; customs
            # are never injected over it).
            #
            # These are the emoji THIS WORKSPACE actually reacts with, most-used first. The gate
            # is a single tool-free call whose emoji is placed directly (see _place_gate_reaction),
            # so unlike the responder it cannot search the catalog — its whole palette has to
            # arrive in this one prompt, which makes the choice of WHICH names to send the
            # entire game. It used to send the first N alphabetically out of ~1,400, i.e.
            # "000, 1password_icon, 2605732e-82a0-46b9-b1e0-ecc4f250eb35, 4cats_q, alabama" —
            # unusable, and worse than nothing because it is prompt noise the model must read.
            # Observed usage is the only real ranking signal Slack exposes (there is no
            # popularity endpoint, and emoji.list carries no tags or descriptions).
            #
            # When nothing has been observed yet, send NOTHING. Standard emoji are a clean
            # fallback the model already knows; an alphabetical slice is not a fallback at all.
            workspace_custom_emojis = []
            if not (config.reaction_emojis or []):
                emoji_cache = getattr(client, "workspace_emojis", None)
                pulse = getattr(client, "channel_pulse", None)
                if emoji_cache is not None and pulse is not None:
                    try:
                        cap = max(0, int(getattr(config, "participation_custom_emoji_cap", 32)))
                        workspace_custom_emojis = pulse.top_custom_reactions(
                            allowed=emoji_cache.get_custom_emoji_names(), limit=cap)
                    except Exception:  # noqa: BLE001 — never cost the gate a verdict
                        workspace_custom_emojis = []

            # Track 1: load the PRIOR channel narrative (never blocks — uses whatever is cached)
            # and kick a DETACHED refresh decision so the cache stays warm. The classifier reads
            # the narrative as background for relevance/value only; the framing forbids using it
            # for addressee resolution. Load via channel_id only (strict per-channel scope).
            channel_summary = None
            summary_svc = getattr(self.processor, "channel_summary_service", None)
            if summary_svc is not None:
                try:
                    channel_summary = await summary_svc.render_for_channel(channel_id)
                except Exception:
                    channel_summary = None
                try:
                    await summary_svc.maybe_refresh(channel_id, client=client, pulse=pulse)
                except Exception:
                    pass

            evaluation = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_id=message.user_id,
                sender_name=message.metadata.get("user_real_name") or message.metadata.get("username"),
                is_thread_reply=is_thread_reply, level=level,
                directives=message.metadata.get("channel_directives"),
                memory_facts=memory_facts, channel_activity=channel_activity,
                name_hit=name_hit,
                self_display_name=getattr(client, "bot_handle", None),
                sender_is_bot=message.metadata.get("participation_sender_bot") is True,
                channel_topic=channel_topic,
                channel_canvases=channel_canvases,
                channel_people=channel_people,
                channel_summary=channel_summary,
                capabilities=capabilities,
                workspace_custom_emojis=workspace_custom_emojis,
                attachments=message.metadata.get("participation_attachments"),
                # F40: descriptors only — the engine downloads the pixels itself, and only once
                # the message has survived the debounce.
                images=message.metadata.get("participation_images"),
                client=client,
                pulse=pulse, thread_root_ts=message.thread_id,
                attempt_id=attempt_id,
            )
            gate_latency_ms = int((time.monotonic() - gate_started_at) * 1000)
            verdict = evaluation.verdict
            # A decline is TERMINAL and its detail was already recorded by the engine (which
            # alone knows the survivor ts / the exception type). The outer backstop that used to
            # log a second `no_verdict` decline here is gone: it double-counted every
            # supersession, and a ledger whose declines outnumber its attempts measures nothing.
            # `classifier_error` arrives WITH a manufactured fail-safe `ignore` verdict — the
            # silence is byte-identical to before, but it is not scored as the model's judgment
            # and it produces no gate_decision.
            if evaluation.decline_cause or verdict is None:
                main_logger.debug(
                    f"Participation gate: no verdict to act on "
                    f"({evaluation.decline_cause or 'superseded'}) — silent")
                participation_telemetry.finish_attempt(
                    message, "none", ended_by="gate", cause=evaluation.decline_cause,
                    gate_ms=gate_latency_ms, classifier_ms=evaluation.classifier_ms)
                return None
            participation_telemetry.gate_decision(
                channel_id, ts, verdict, attempt_id=attempt_id,
                gate_ms=gate_latency_ms, classifier_ms=evaluation.classifier_ms)
            verdict_recorded = True
            main_logger.debug(f"Participation verdict: {verdict.action} ({verdict.reason})")
            # An overrule means the model's chosen action contradicted its OWN staged findings.
            # Logged at INFO because a rising rate is a signal about the prompt, not the code: the
            # invariants are a backstop, and if they start carrying the decision it is the prompt
            # that needs fixing. No message text here — only the declared classifications.
            # getattr, not attribute access: verdicts do not all come from validate_verdict (the
            # edit-reply path and tests construct their own), and a missing field must never turn
            # into an AttributeError that the gate's except-clause converts into silence.
            if getattr(verdict, "overruled_by", None):
                main_logger.info(
                    f"Participation invariant: "
                    f"{','.join(getattr(verdict, 'overruled_by', None) or [])} → "
                    f"{verdict.action} | channel={channel_id} ts={ts} "
                    f"relation={getattr(verdict, 'relation', None)} "
                    f"exchange={getattr(verdict, 'exchange_state', None)} "
                    f"answerability={getattr(verdict, 'answerability', None)}")

            if verdict.action == "react":
                react_ts = message.metadata.get("ts") or message.thread_id
                # Route through the shared gate-reaction helper (reservation guard + timeout +
                # once-per-message stamp). A reaction is this verdict's whole response — stay silent.
                placed = await self._place_gate_reaction(
                    message, client, channel_id, react_ts, verdict.emoji)
                # The emoji IS the turn, so whether it landed decides what the room saw. A
                # react verdict whose reaction was refused or timed out showed nothing at all,
                # and filing it as reaction_only would report a reaction that is not there.
                participation_telemetry.finish_attempt(
                    message, "reaction_only" if placed else "none",
                    ended_by="gate", action=verdict.action)
                return None
            if verdict.action == "react_and_respond":
                # BOTH react and reply in one turn: place the gate reaction (same path as `react`),
                # then fall through to the response loop like `respond` so the words go out too. The
                # response turn's developer suffix tells the model it already reacted (so it doesn't
                # add a second reaction). The stamp inside _place_gate_reaction means a queued
                # redispatch that re-picks a react/react_and_respond verdict can't stack a second one.
                react_ts = message.metadata.get("ts") or message.thread_id
                await self._place_gate_reaction(
                    message, client, channel_id, react_ts, verdict.emoji)
                # The responder owns the end of this turn, so it owns the terminal event too.
                # (The wake itself is recorded once by the caller, for every fall-through.)
                return verdict
            if verdict.action == "backoff":
                # The taxonomy decides what "backoff" means: a durable per-channel preference,
                # a real thread mute/unmute, a momentary aside (nothing persisted), or an
                # explicit channel-settings change — the last one falls through to the response
                # loop so the MAIN model applies it (with judgment) via set_channel_participation.
                fall_through, ack_placed = await self._apply_backoff(message, client, verdict)
                if fall_through:
                    return verdict
                # Fully handled here. An ack reaction that LANDED is what the room saw, so this
                # is not a silent turn — the feedback was visibly acknowledged. Without a
                # reaction (feedback about reactions never gets one, and a failed add is not
                # one) nothing was shown and it is a silence.
                participation_telemetry.finish_attempt(
                    message, "reaction_only" if ack_placed else "silence",
                    ended_by="gate", action=verdict.action, detail="backoff")
                return None
            if verdict.action == "respond":
                # F38: the gate no longer acks. It used to drop a 👀 here on a respond+ack
                # verdict, but that reaction was a PREDICTION that work was coming — made
                # before the model had done anything, and demonstrably overeager (it acked
                # "Never tried this. Not sure how it will turn out", a passing comment). A
                # teammate who drops eyes and then does nothing is misleading. The 👀 is now
                # a CLAIM ON WORK, staked by TurnRuntime.claim_work when a tool actually
                # starts doing something slow, and taken back if that work produces nothing.
                return verdict
            # ignore — the model was asked and chose to say nothing. A DECISION, and the only
            # gate-terminal outcome that deserves the `silence` label.
            participation_telemetry.finish_attempt(
                message, "silence", ended_by="gate", action=verdict.action)
            return None
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
            cause = "action_error" if verdict_recorded else "error"
            participation_telemetry.gate_declined(
                message.channel_id, (message.metadata or {}).get("ts") or message.thread_id,
                cause=cause, attempt_id=attempt_id, detail=type(e).__name__)
            participation_telemetry.finish_attempt(
                message, "none", ended_by="gate", cause=cause,
                detail=type(e).__name__)
            return None

    async def _place_gate_reaction(self, message: Message, client: BaseClient,
                                   channel_id: str, react_ts: str, emoji: Optional[str],
                                   origin: str = "gate") -> bool:
        """Place a participation-gate reaction ONCE per message, via the reservation guard so a
        later turn sees the slot consumed. Idempotent across queued redispatch: the SAME Message
        object is re-run through the gate (Phase Q) and a fresh pass may pick a different emoji or a
        different reaction-bearing verdict — so every gate-reaction branch (react, react_and_respond,
        backoff ack) checks the stamp first and no-ops if a gate reaction was already placed. On a
        genuine placement, stamps message.metadata['participation_reaction_emoji'] so the response
        turn's developer suffix can tell the model it already reacted.

        Returns True when the gate's emoji IS on the message when this returns — a genuine
        placement now, or one this same message already made on an earlier pass. The caller uses
        it to say what the room saw, and a refused or failed add showed nothing. `origin`
        separates a social gate reaction from a backoff acknowledgment: the second is a fixed
        protocol move, not taste, and pooling them would make the gate's emoji diversity look
        better than it is."""
        attempt_id = participation_telemetry.attempt_id_for(message)

        def _record(result_name: str, *, detail=None) -> None:
            if not attempt_id:
                return
            participation_telemetry.reaction(
                channel_id, react_ts, operation="add", result=result_name, origin=origin,
                emoji=emoji, target_ts=react_ts, attempt_id=attempt_id, detail=detail)

        # Each early return below is an intent that never reached Slack. They were invisible
        # before, which made "the gate stopped reacting" impossible to distinguish from "the
        # gate stopped choosing emoji".
        if not emoji or not react_ts or not channel_id:
            _record("refused", detail="invalid_target")
            return False
        md = message.metadata if isinstance(message.metadata, dict) else None
        # Dedup: a redispatch re-runs this same object through the gate, and a fresh pass can flip
        # react_and_respond→react (or pick a new emoji) — honor the first placement, never stack.
        #
        # TRUE, not False: the stamp is only ever written on a genuine placement, so reaching it
        # means OUR emoji is on that message right now. The caller asks "what did the room see",
        # and the room sees an emoji — returning False here filed a redispatched `react` as
        # `none` and a stamped backoff ack as `silence`, both describing an empty message that
        # visibly has a reaction on it. (The reaction row stays `already_present`: we did not
        # place one on THIS pass, and counting it as a placement would double the gate's
        # reaction rate on every redispatch.)
        if md is not None and md.get("participation_reaction_emoji"):
            _record("already_present", detail="already_stamped")
            return True
        result = None
        try:
            # Bound the gate's own react by the configured tool-call timeout so a wedged Slack call
            # can't stall the turn. F6: route through the reservation guard so a later main-model
            # turn honestly sees the slot consumed (and won't double-add). Falls back to raw react.
            if hasattr(client, "_reserve_and_react"):
                result = await asyncio.wait_for(
                    client._reserve_and_react(channel_id, react_ts, emoji),
                    timeout=config.tool_call_timeout)
            elif hasattr(client, "react"):
                result = await asyncio.wait_for(
                    client.react(channel_id, react_ts, emoji),
                    timeout=config.tool_call_timeout)
        except asyncio.TimeoutError:
            main_logger.debug("Participation gate react timed out")
            _record("failed", detail="timeout")
            return False
        except Exception as e:
            main_logger.debug(f"Participation gate react failed: {e}")
            _record("failed", detail=type(e).__name__)
            return False
        # Stamp ONLY on a genuine placement (_reserve_and_react → {"ok": True}; react → True). On a
        # cap/busy/timeout failure the slot wasn't taken, so leave the stamp unset: a later attempt
        # can still try, and the model must never be told it reacted when it didn't.
        placed = result is True or (isinstance(result, dict) and result.get("ok") is True)
        # Recorded either way: a verdict that CHOSE an emoji and lost it to the per-message cap is
        # a different story from one that never wanted to react, and only this line tells them
        # apart afterwards. `idempotent` from the reservation layer means somebody else's emoji
        # was already there — present, but not placed by us.
        if placed and isinstance(result, dict) and result.get("idempotent"):
            _record("already_present")
        elif placed:
            _record("added")
        else:
            _record("failed",
                    detail=(result.get("error") if isinstance(result, dict) else None))
        if placed and md is not None:
            md["participation_reaction_emoji"] = emoji
        return placed

    # ---- participation-feedback backoff taxonomy (redesign Layer 2) ----
    # Default guidance text per dimension, used to write a readable preference memory when the
    # classifier gives no `guidance` of its own.
    _PREF_DEFAULT_GUIDANCE = {
        "reactions": "react less often in this channel",
        "replies": "reply more sparingly in this channel",
        "verbosity": "keep replies short in this channel",
        "thread_participation": "participate more sparingly in this channel",
    }

    async def _apply_backoff(self, message: Message, client: BaseClient,
                             verdict) -> tuple[bool, bool]:
        """Route a participation-feedback ('backoff') verdict through the redesign taxonomy.

        Returns (fall_through, ack_placed). `fall_through` is True when the message should go
        to the MAIN response loop — an explicit channel-settings change the model applies with
        judgment via the gated set_channel_participation tool — and False when the feedback was
        fully handled here: a durable per-channel preference (memory), or a momentary/
        thread-scoped aside that persists nothing. `ack_placed` says whether the optional
        acknowledgment reaction actually landed, which is the only thing that makes a handled
        backoff visible in the room at all.

        Structural settings (participation level / placement) are NEVER written here. This
        routine only ever touches per-channel preference MEMORY, so a "react less" can no
        longer clobber a channel's response mode — the incident this redesign fixes. A
        thread-scoped "stop replying here" is guidance for the current message only; it writes
        nothing durable (there is no per-thread mute — that mechanism was removed)."""
        channel_id = message.channel_id
        react_ts = message.metadata.get("ts") or message.thread_id

        # 1. Explicit structural request → the model owns it. Nothing durable is written here;
        #    the taxonomy deliberately keeps settings changes in the response loop.
        if verdict.structural_request and verdict.structural_request != "none":
            main_logger.info(
                f"Participation backoff: explicit structural request "
                f"({verdict.structural_request}) in {channel_id} — routing to the response loop")
            return True, False

        standing = verdict.durability == "standing"
        db = getattr(self.processor, "db", None)

        # 2. The ONLY durable effect is a per-channel preference marker, and only for a
        #    standing, CHANNEL-scoped verdict. A momentary "not now" persists nothing; a
        #    thread-scoped "stop replying here" is guidance for this message alone — there is
        #    no per-thread mute to write (the mute mechanism was removed, so a thread aside can
        #    neither clobber channel settings nor leave a durable record). Any other/missing
        #    scope also writes nothing.
        if (standing and db is not None and verdict.scope == "channel"
                and getattr(config, "enable_channel_memory", True)):
            try:
                await self._apply_pref_memory(channel_id, verdict)
            except Exception as e:
                main_logger.warning(f"Backoff durable write failed: {e}")

        # 4. Conditional ack. Routed through the reservation/timeout path (like the gate's own
        #    react) so a later main-model turn honestly sees the slot consumed. NEVER react when
        #    the feedback is ABOUT reactions — acking "stop reacting" with a reaction is absurd.
        ack_placed = False
        if verdict.emoji and verdict.dimension != "reactions":
            ack_placed = await self._backoff_ack(
                message, client, channel_id, react_ts, verdict.emoji)

        return False, ack_placed

    # Reserved author prefix for the engine's own per-dimension preference markers. The backoff
    # memory CRUD may ONLY ever touch rows under this prefix — never a human's fact and never a
    # workspace fact (both of which get_channel_memory_async also returns).
    _PREF_MARKER_PREFIX = "participation_engine:pref:"

    def _own_pref_row(self, fact: Dict[str, Any]) -> bool:
        """True only for one of the engine's OWN channel-scope preference markers. Guards the
        backoff CRUD so an `update:<id>`/`delete:<id>` verdict can never rewrite or delete a
        workspace or human memory fact (redesign BLOCKER #4)."""
        return (((fact.get("scope") or "channel") == "channel")
                and str(fact.get("author") or "").startswith(self._PREF_MARKER_PREFIX))

    def _is_own_dimension_pref(self, fact: Dict[str, Any], marker: str) -> bool:
        """Stronger than `_own_pref_row`: True only for THIS dimension's own channel-scope marker
        row (`author == marker`). SHOULD-FIX 1: an `update:<id>`/`delete:<id>` verdict names a
        raw fact id, and `_own_pref_row` alone would accept ANY of the engine's markers — so a
        `reactions` verdict could rewrite or delete the `verbosity` marker. Requiring the author
        to equal the current dimension's marker refuses a cross-dimension id (it then falls back
        to this dimension's own marker row)."""
        return self._own_pref_row(fact) and str(fact.get("author") or "") == marker

    async def _apply_pref_memory(self, channel_id: str, verdict) -> None:
        """Record / refine / remove ONE per-channel, per-dimension participation preference.

        Keyed by a stable marker author `participation_engine:pref:<dimension>` so a repeat
        "react less" UPDATES the single marker row instead of accumulating duplicate facts —
        the false "REPEATED = observe-only" escalation the redesign removes.

        Scope discipline (BLOCKER #4): every write here is confined to the engine's OWN marker
        rows. An `update:<id>`/`delete:<id>` that names a workspace or human fact is REFUSED and
        falls back to the per-dimension marker path; it never rewrites or deletes someone else's
        memory. The add/refresh path goes through the atomic upsert_channel_pref_memory helper
        (SHOULD-FIX #8), which enforces one marker row per dimension, the MEMORY_MAX_ROWS cap,
        and the marker author — with no read-all-then-insert race."""
        db = self.processor.db
        dimension = verdict.dimension or "replies"
        marker = f"{self._PREF_MARKER_PREFIX}{dimension}"
        op = verdict.memory_op
        existing = await db.get_channel_memory_async(channel_id) or []

        # Reversal: delete the recorded preference. An explicit [#id] is honored ONLY when it
        # names one of our OWN markers; otherwise fall back to this dimension's marker row. A
        # workspace/human id is never deleted.
        if op.startswith("delete"):
            target = None
            if op.startswith("delete:"):
                wanted = int(op.split(":", 1)[1])
                cand = next((f for f in existing if f.get("id") == wanted), None)
                # SHOULD-FIX 1: only THIS dimension's own marker — a cross-dimension id is refused.
                if cand is not None and self._is_own_dimension_pref(cand, marker):
                    target = cand
            if target is None:
                target = next(
                    (f for f in existing if self._is_own_dimension_pref(f, marker)), None)
            if target is not None:
                await db.delete_channel_memory_async(target["id"])
                main_logger.info(
                    f"Participation reversal: removed preference [#{target['id']}] in {channel_id}")
            return

        content = self._pref_memory_content(verdict, dimension)

        # Explicit update of a specific numbered fact — honored ONLY for our own marker rows.
        # Updating in place keeps that row's marker author. A non-owned or stale id is refused and
        # falls through to the atomic marker upsert (which (re)writes the per-dimension marker).
        if op.startswith("update:"):
            wanted = int(op.split(":", 1)[1])
            row = next((f for f in existing if f.get("id") == wanted), None)
            # SHOULD-FIX 1: only THIS dimension's own marker may be updated in place. A row owned
            # by a DIFFERENT dimension (or a workspace/human fact) is refused and falls through to
            # the marker upsert, so a verdict never corrupts another dimension's preference.
            if row is not None and self._is_own_dimension_pref(row, marker):
                await db.update_channel_memory_async(row["id"], content)
                main_logger.info(f"Participation preference updated [#{row['id']}] in {channel_id}")
                return
            # non-owned / cross-dimension / stale id — fall through to the marker upsert

        # add / refresh: exactly one preference row per dimension, written atomically with the
        # marker author and the MEMORY_MAX_ROWS cap enforced inside the helper.
        cap = max(1, getattr(config, "memory_max_rows", 25))
        new_id = await db.upsert_channel_pref_memory(channel_id, marker, content, max_rows=cap)
        if new_id is None:
            main_logger.debug(
                f"Participation preference at memory cap and no marker row in {channel_id} — "
                "not adding (won't evict a human's memory)")
        else:
            main_logger.info(
                f"Participation preference recorded/refreshed [#{new_id}] ({dimension}) in {channel_id}")

    def _pref_memory_content(self, verdict, dimension: str) -> str:
        """The stored preference sentence: the classifier's normalized guidance when present,
        else a sensible per-dimension default, tagged with the dimension for readability."""
        guidance = " ".join((verdict.guidance or "").split())
        if not guidance:
            guidance = self._PREF_DEFAULT_GUIDANCE.get(
                dimension, "participate more sparingly in this channel")
        elif len(guidance) > GUIDANCE_TRUNCATION_CHARS:
            guidance = guidance[:GUIDANCE_TRUNCATION_CHARS] + "…"
        return f"Channel participation preference ({dimension}): {guidance}"

    async def _backoff_ack(self, message: Message, client: BaseClient, channel_id: str,
                           react_ts: str, emoji: str) -> bool:
        """Drop the optional acknowledgment reaction, routed through the shared gate-reaction
        helper so it goes through the same reservation guard the gate's own react uses (a later
        turn sees the slot consumed and never double-adds) AND honors the once-per-message stamp —
        a react_and_respond→backoff redispatch can't stack a second reaction onto this message.

        Recorded under its own origin. This emoji is a protocol acknowledgment of feedback, not
        a social reaction the classifier chose for its own sake, and counting it as one would
        quietly improve every "is its emoji use varied?" number with a fixed move."""
        return await self._place_gate_reaction(message, client, channel_id, react_ts, emoji,
                                               origin="backoff_ack")

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
    def _classify_visible_action(response, turn, gate_reaction_visible: bool = False) -> str:
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

        `gate_reaction_visible` is the one fact this function cannot see for itself: on a
        `react_and_respond` the GATE already put an emoji on the message before the responder
        ran, so a responder that then vetoes with no_response_needed did not leave the room
        silent. The caller reads it off the message stamp and passes it in — the function stays
        pure, which is why every one of these labels is testable as a table.
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
            # The gate's own emoji counts for exactly the same reason: on react_and_respond the
            # reaction is already up when the responder decides to use no words, and calling
            # that turn a silence describes an empty message that is visibly reacted to.
            if (meta.get("response_reaction_committed") is True or gate_reaction_visible
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
        """
        try:
            # Phase F participation gate: for UNPROMPTED channel messages (judicious/active
            # levels) the engine decides respond/react/react_and_respond/ignore/backoff BEFORE
            # anything is posted. The reply outcomes fall through (respond, react_and_respond, and
            # a backoff requesting a settings change); a react_and_respond has already placed its
            # reaction in the gate, so only the words remain to send.
            placement_verdict = None
            gate_required = message.metadata.get("gate_required") is True
            if gate_required:
                try:
                    verdict = await self._run_participation_gate(message, client)
                finally:
                    # AFTER the gate, and in a `finally`: the attempt is minted INSIDE that
                    # await, so a cancellation or a raise on the way back out would otherwise
                    # leave an attempt that absorbed queued messages and never said so. Emitting
                    # here is safe in both directions — it writes nothing while the attempt does
                    # not exist yet, and the ungated turn's links are written later still, at the
                    # conversation lock (see MessageProcessor.process_message).
                    participation_telemetry.emit_queue_links(message, gate_required=True)
                if verdict is None:
                    return
                placement_verdict = verdict.placement
                # Kept for logs and debugging ONLY — deliberately NOT rendered into the wake
                # envelope any more (see utilities._wake_trigger_line): forwarding the gate's own
                # justification pre-argued the turn and neutered the no_response_needed veto.
                if isinstance(message.metadata, dict) and getattr(verdict, "reason", None):
                    message.metadata["participation_reason"] = verdict.reason
                # F27: earlier same-author burst messages ride the wake envelope too, so the
                # reply is told to cover the whole burst, not just the triggering fragment.
                if isinstance(message.metadata, dict) and getattr(verdict, "burst_earlier", None):
                    message.metadata["participation_burst_earlier"] = verdict.burst_earlier

            # F46: judgment-call placement for MENTIONS/name-wakes. These run NO participation gate
            # (so placement_verdict is still None) and default to a top-level reply — but a
            # deliberately-requested long-form deliverable ("write me a 3-paragraph story") reads
            # better in a thread, and no tool fires for it so the did_substantive_work override can't
            # catch it. One lean utility-model call decides thread vs channel, feeding the UNCHANGED
            # place_in_channel logic below. Gated behind enable_mention_placement_model (DEFAULT OFF):
            # flag off ⇒ skipped entirely, zero added latency/cost, zero behavior change. Only for a
            # top-level PUBLIC-channel trigger where top-level replies are allowed and no gate verdict
            # exists (never override the engine's verdict). Fail-open: classify_placement returns
            # "channel" on any error, and a raised call must not break the reply.
            if (getattr(config, "enable_mention_placement_model", False)
                    and placement_verdict is None
                    and message.metadata.get("ts") == message.thread_id
                    and bool(message.metadata.get("reply_in_channel"))
                    and message.channel_id and not message.channel_id.startswith("D")):
                try:
                    placement_verdict = await self.processor.openai_client.classify_placement(
                        message.text)
                    main_logger.debug(
                        f"Mention placement: verdict={placement_verdict} for a top-level "
                        f"public-channel mention")
                except Exception as e:
                    main_logger.debug(f"Mention placement call failed ({e}); staying top-level")
                    placement_verdict = None

            # Phase F placement (plan §4a, revised 2026-07-10): the channel's
            # reply_in_channel setting is an ALLOWANCE, not a mandate — when it's ON and
            # the trigger was top-level, the engine's per-message placement verdict
            # decides ("channel" = quick top-level answer, "thread" = worth a thread).
            # Mentions/name-wakes carry no verdict (no engine call) and reply top-level:
            # the user summoned the bot at channel level. Setting OFF = everything
            # threads. Images always thread (enforced in the image branch, which keys
            # off message.thread_id regardless).
            is_top_level_trigger = message.metadata.get("ts") == message.thread_id
            place_in_channel = (
                bool(message.metadata.get("reply_in_channel")) and is_top_level_trigger
                and bool(message.channel_id) and not message.channel_id.startswith("D")
                and placement_verdict != "thread"
            )
            if placement_verdict:
                main_logger.debug(
                    f"Placement: verdict={placement_verdict}, reply_in_channel_setting="
                    f"{bool(message.metadata.get('reply_in_channel'))} → "
                    f"{'channel' if place_in_channel else 'thread'}"
                )
            post_thread_id = None if place_in_channel else message.thread_id
            # Handlers key presentation chrome off this (e.g. the Used Tools attribution
            # line is suppressed on top-level channel replies).
            if isinstance(message.metadata, dict):
                message.metadata["place_in_channel"] = place_in_channel

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

            # F38: what this turn is allowed to SHOW. A turn the model may end in silence gets no
            # speculative chrome at all — no placeholder, no composer status (which would also
            # auto-open the thread), no phase updates. The reply, if there is one, creates its own
            # surface when the first words arrive; if there is none, nothing was ever posted.
            turn = TurnRuntime.for_message(message, post_thread_id)

            # Send initial thinking indicator (streamed replies grow inside this message,
            # so placement is decided here).
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
                            await client.update_message(
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

                # F46: the handler may have flipped a top-level channel reply into a thread (a turn
                # that did substantive work — resolve_reply_target mutates message.metadata but NOT
                # these locals). Rebind from the metadata so the fallback send, the footer guard, and
                # channel_pulse below all agree with the placement text.py actually used. Fail-open:
                # only rebind when metadata is a dict; a missing key leaves the original value.
                if isinstance(message.metadata, dict):
                    place_in_channel = bool(message.metadata.get("place_in_channel", place_in_channel))
                    post_thread_id = None if place_in_channel else message.thread_id

                # Delete thinking indicator (but not if streaming was used — it's already the
                # response — and not when a ProgressChecklist owns the thinking message, F4).
                if (thinking_id and response
                        and not response.metadata.get("streamed")
                        and response.metadata.get("checklist") is None):
                    await client.delete_message(message.channel_id, thinking_id)
                elif thinking_id and not response:
                    await client.delete_message(message.channel_id, thinking_id)

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
                            if not place_in_channel and hasattr(client, "attachable_footer_blocks"):
                                try:
                                    footer_blocks = client.attachable_footer_blocks(
                                        message.channel_id, response.metadata.get("model"))
                                except Exception as e:
                                    main_logger.debug(f"Footer block build failed: {e}")
                                    footer_blocks = None
                            send_meta = {}
                            sent_ts = await client.send_message(
                                message.channel_id,
                                post_thread_id,
                                response.content,
                                blocks=footer_blocks,
                                meta_out=send_meta,
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
                        if (hasattr(client, "maybe_post_response_footer") and not place_in_channel
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
                        await client.handle_error(
                            message.channel_id,
                            message.thread_id,
                            response.content
                        )

                # Close the attempt with what the room actually SAW. Deliberately not folded into
                # the contract check below, which is additionally gated on a live pulse — a
                # delivery-side condition that has nothing to do with whether the outcome is worth
                # counting. `finish_attempt` is a no-op unless this turn came from the gate and is
                # still open, which is what keeps mentions, DMs and direct continuations out of a
                # ledger documented as gate attempts — and what keeps a turn the gate already
                # closed (a react verdict that fell through to nothing) from closing twice.
                meta = (response.metadata or {}) if response is not None else {}
                # What the GATE put in the room before the responder ran. The stamp is written
                # only on a genuine placement, so it is a fact about Slack, not an intention.
                gate_reaction = bool((message.metadata or {}).get(
                    "participation_reaction_emoji"))
                kind = self._classify_visible_action(response, turn, gate_reaction)
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
                        gate_reaction or meta.get("response_reaction_committed") is True
                        or (turn is not None and getattr(turn, "reaction_committed", False))),
                    # WHICH model wrote it. Per-user and per-thread overrides mean two rows in
                    # this ledger can come from different models, and a reply-quality comparison
                    # that pools them describes neither. The footer path reads the same key.
                    model=meta.get("model"),
                    # An `empty` with no Response object at all is a different contract failure
                    # from one that returned an empty Response, and only this tells them apart.
                    detail="no_response_object" if response is None else None,
                    placement="channel" if place_in_channel else "thread",
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
                        self._classify_visible_action(
                            response, turn,
                            bool((message.metadata or {}).get(
                                "participation_reaction_emoji"))),
                        ended_by="responder", post_delivery_error=type(e).__name__)
                else:
                    participation_telemetry.finish_attempt(
                        message, "error_unhandled", ended_by="responder",
                        detail=type(e).__name__)

                # Delete thinking indicator on error — best-effort; a failed delete
                # must never swallow the user-facing notice below.
                if thinking_id:
                    try:
                        await client.delete_message(message.channel_id, thinking_id)
                    except Exception as delete_error:
                        main_logger.error(f"Failed to delete thinking indicator: {delete_error}")

                # Fixed, friendly notice — the raw exception stays in the logs only.
                try:
                    await client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        "⚠️ **Something Went Wrong**\n\n"
                        "I hit a snag finishing that response. Please try again in a moment."
                    )
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