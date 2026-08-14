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
from typing import Any, Dict, Optional, cast
from config import config, dev_epoch_fence_requested
from logger import log_session_start, log_session_end, main_logger
from message_processor.base import MessageProcessor
from message_processor import (channel_steering, outbound_receipts,
                               participation_telemetry, routing_facts)
from message_processor.destination_tools import consume_destination_marker
from message_processor.participation import (ParticipationEngine,
                                             resolve_participation_level)
from message_processor.reconsideration import intercept_stale_send
from message_processor.stale_send_guard import (ConversationWatermarks,
                                                StaleSendSuppressed)
from message_processor import turn_runtime
from message_processor.turn_runtime import (DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                            DEST_KIND_POST_TO_THREAD, DEST_KIND_RECONCILED,
                                            DEST_KIND_REPLY, DEST_KIND_SPLIT, TurnRuntime)
from message_processor import thread_files
import message_processor.token_counter as token_counter
from message_processor.client_contract import BaseClient, Message
from slack_client import admission_watermark
from slack_client.event_handlers import registration
from slack_client.utilities import is_dm_conversation

# The dev-only epoch fence. THE IMPORT ITSELF IS GATED ON THE FLAG, so "without
# DEV_EPOCH_FENCE_ENABLE nothing happens" is literally true rather than nearly true: with the flag
# empty this module is never imported, so no code in it runs and no defect in it can reach a
# production boot.
#
# The two cases are NOT symmetric, and conflating them is what an earlier revision got wrong:
#   * flag EMPTY  — nothing is imported; `epoch_fence` is None and the bot is simply a bot.
#   * flag SET    — an import failure is FATAL, reported by `initialize()`. Someone asked for a
#     fence; booting without one would run a battery against an unfenced channel and report the
#     results as if they were isolated, which is worse than not booting.
#
# The predicate lives in `config` — a leaf this module already imports — because it must DECIDE
# whether the fence module is imported, and a predicate cannot gate its own import.
# `epoch_fence.fence_enabled()` delegates to the same function, so there is exactly one definition.
epoch_fence: Any = None   # rebound to the module below when the fence is requested
_epoch_import_error = None
if dev_epoch_fence_requested():
    try:
        from message_processor import epoch_fence
    except Exception as _e:  # noqa: BLE001 — recorded here, fatal in initialize()
        _epoch_import_error = _e

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

# How long shutdown lets an already-admitted turn finish before cancelling it. Generous, because
# the cost of cutting one short is a reply the room never sees; bounded, because a wedged model
# call must not hold the process open.
TURN_QUIESCE_TIMEOUT_SECONDS = 30.0


async def _finalize_turn_effects(turn: Any) -> None:
    """Settle what this turn CAUSED, then what it POSTED — as one indivisible sequence.

    Order is the invariant, not a preference. `finish_tool_flights` drains the flights that
    outran their bound, then CANCELS the stragglers, waits for that cancellation to land, and
    REVOKES the permission of anything that survived it. Only then may the receipts settle: a
    task that can still post must be stopped (or forbidden) before the ledger says the turn's
    words are all accounted for. `settle_ledger` closes the other half, waiting out any effect
    lease still held — an accepted post whose receipt is mid-write.

    Runs as its own task so a cancellation aimed at the turn cannot land BETWEEN these two.

    If the drain itself FAILS — not a tool failing, which it absorbs, but the bookkeeping that
    proves quiescence — the sequence does not simply carry on to the settle. Effect permission is
    withdrawn FIRST. Settling receipts while unknown tasks may still be alive is the receiptless
    post this whole mechanism exists to prevent, and revocation is what makes "unknown" safe: it
    cannot be made known any more, but it can be made unable to act.

    And if the REVOCATION fails too, nothing settles. That is the one state where finalizing is
    strictly worse than not: the turn cannot say what is running and cannot stop it, so a settle
    would record as accounted-for words a live task may still be adding to. Left in flight, those
    rows are exactly what boot's dead-session reconcile picks up — the sole reason it exists.
    """
    revoked = True
    if turn is not None:
        finish_flights = getattr(turn, "finish_tool_flights", None)
        if finish_flights is not None:
            try:
                pending = finish_flights()
                if hasattr(pending, "__await__"):
                    await pending
            except Exception as e:  # noqa: BLE001 — revoked below, then settled
                main_logger.error(
                    f"Tool flights did not settle: {e!r} — revoking this turn's effects before "
                    "its receipts settle")
                revoked = turn_runtime.revoke_turn_effects(turn, "tool flight settlement failed")
    if not revoked:
        main_logger.critical(
            "Turn %s: the flight drain FAILED and its effects could not be revoked — leaving "
            "this turn's receipts deliberately UNSETTLED for the next boot's dead-session "
            "reconcile. Something this turn started may still be running and may still post.",
            getattr(turn, "turn_id", None))
        return
    # Spec §5: the turn's own words stop being in-flight HERE, outside every inner handler, so a
    # cancellation or an unexpected raise cannot strand rows this session will never touch again
    # — dead-session reconcile only runs at boot, and a live process is not a dead session.
    await outbound_receipts.settle_ledger(
        getattr(turn, "receipt_ledger", None) if turn is not None else None, turn=turn)


async def _await_finalizer(finalizer: "asyncio.Task") -> None:
    """Wait for that unit to FINISH, through any number of cancellations aimed at us.

    A CancelledError here is OUR await being cancelled, never the unit's: the unit owns itself.
    It is not re-raised — everything the finalizer does is what keeps a delivered message from
    being stranded, and the turn is ending either way — and it is not a reason to walk away
    before the unit finishes either.

    COMPLETION-BOUND, deliberately without an attempt cap. A cap is a promise to abandon the unit
    on the Nth cancellation, and "the caller returned while the finalizer was still revoking and
    settling" is the exact state the shielded unit exists to make impossible — the turn goes on to
    release its state and remove itself, and whatever the finalizer had not reached yet belongs to
    nobody. The wait terminates on its own: the finalizer's own steps are each bounded (the flight
    grace, the lease bodies' transports), so a repeat-canceller can only make this loop spin as
    long as it keeps cancelling, and it stops the moment the unit is done.
    """
    while not finalizer.done():
        try:
            await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            main_logger.warning(
                "Turn finalizer outlived a cancellation of its caller — waiting for it to "
                "finish revoking and settling")
        except Exception as e:  # noqa: BLE001 — it reports its own failures
            main_logger.warning(f"Turn finalizer ended with: {e!r}")
    if not finalizer.cancelled():
        finalizer.exception()  # consumed


async def channel_identity_ready(client: BaseClient, message: Message) -> bool:
    """Whether a CHANNEL turn may run at all (spec §5).

    Every durable word a channel turn says needs a receipt, and a receipt needs the bot's team
    id. Without one the ledger builds itself INACTIVE and writes nothing — the turn looks
    perfectly healthy, and its replies are simply missing from every rebuilt stream afterwards.
    Silence on that is the one failure nobody would ever notice, so a channel turn that cannot
    establish identity is refused loudly instead of served quietly.

    DMs keep no receipts and are never blocked. Clients that are not the receipts transport
    (test doubles, other platforms) are not held to a contract they do not implement.
    """
    if not outbound_receipts.receipts_apply(getattr(message, "channel_id", None)):
        return True
    if getattr(client, "self_team_id", None) and getattr(client, "bot_user_id", None):
        return True
    ensure = getattr(client, "ensure_receipt_identity", None)
    if ensure is None:
        return True
    try:
        if await ensure():
            return True
    except Exception as e:  # noqa: BLE001 — reported by the refusal below
        main_logger.warning(f"Identity re-resolution raised: {e}")
    main_logger.error(
        f"Refusing a channel turn in {getattr(message, 'channel_id', None)}: the bot's own "
        "team/user identity is unresolved (auth.test), so nothing this turn said could be "
        "recorded as ours and it would vanish from the channel stream. Check the bot token "
        "and scopes.")
    return False


class ChatBotV2:
    """Main application class for multi-platform chat bot"""

    # Admission (spec §5): False once shutdown starts, so no NEW turn opens a lease or a receipt
    # ledger behind a queue that is closing. Turns already admitted are tracked in `active_turns`
    # and are finished — or cancelled and awaited — before receipts close.
    #
    # Class-level, and the task set is lazy, because handle_message is also driven against
    # instances built with `ChatBotV2.__new__` (tests that skip the heavy __init__). An admission
    # gate that only exists after __init__ would raise on the first message those send.
    _admitting: bool = True

    @property
    def active_turns(self) -> set:
        turns = self.__dict__.get("_active_turns")
        if turns is None:
            turns = self.__dict__["_active_turns"] = set()
        return turns

    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor: Any = None  # Will be initialized after client
        self.participation_engine = None  # Phase F; set in initialize()
        self._watermarks = ConversationWatermarks()
        self.cleanup_task = None
        self.coverage_bootstrap = None  # spec §4 background coverage sweep
        self.epoch_fence_watcher: Any = None  # dev-only; stays None unless DEV_EPOCH_FENCE_ENABLE
        self.receipt_service = None  # spec §5 outbound receipts
        self.pending_share_recovery = False
        self._pending_share_task = None
        self._scheduled_rehydrate_task: Optional[asyncio.Task] = None  # T1 scheduled deliveries
        self.running = False
        self._admitting = True
        self.sigint_count = 0  # Track number of SIGINT received
        self.last_sigint_time = 0  # Track time of last SIGINT
        # THE shutdown, as one task both paths await. A signal used to start its own, while `run()`
        # went on to finish and let the loop close — which cancelled that task partway, so the
        # telemetry ledger's `session_end` was never written and every restart read as a crash in
        # the one file that exists to tell those apart. Now the signal starts it and run()'s finally
        # awaits the same object, so the drain always completes before the process leaves.
        self._shutdown_task: Optional[asyncio.Task] = None
        # A shutdown was ASKED FOR. Durable, and deliberately separate from the task above: a
        # signal can land after the handlers are installed (inside `initialize`) but before the
        # run loop exists, and `shutdown()` on a bot that is not yet `running` correctly does
        # nothing — so the task it cached completed having stopped nothing, and every later
        # request was handed that finished task. A SIGTERM during startup left the bot running
        # and unstoppable. The REQUEST outlives the attempt to serve it.
        self._shutdown_requested = False
        # ...and a shutdown actually RAN to completion. Tells the finished-because-there-was-
        # nothing-to-do task apart from the finished-because-it-shut-the-bot-down one.
        self._shutdown_completed = False
        self._run_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the bot components"""
        main_logger.info(f"Initializing Chat Bot V2 for {self.platform}...")

        # Open the participation ledger HERE, before any Slack traffic. Built lazily it would
        # have put a mkdir and a file open inside the first gate call — on the hot path of the
        # decision the whole turn is waiting for — and retried it after every failure.
        participation_telemetry.initialize()

        # Same reasoning for the o200k tokenizer, though what rides on it is smaller than it looks:
        # admission is decided by the utf-8 byte bound, which needs no vocabulary at all, and this
        # counter's one production job is the REFUSAL diagnostic — the real token count logged
        # beside the charged bound so an operator can tell "this window needs compacting" from "the
        # bound refused something that would have fit". On a cold tiktoken cache `get_encoding`
        # fetches the vocabulary over the network, and a refusal is the worst place to discover
        # that. Started here so the fetch overlaps the rest of boot.
        #
        # timeout=0 is the point: this STARTS the loader (it runs in its own daemon thread) and
        # does not wait for it. Waiting would put a network round trip in front of the socket
        # connecting, and a blackholed egress would hold the process down for the TCP timeout — so
        # the head start is taken and nothing depends on it having finished. A load that fails logs
        # its own warning from that thread.
        token_counter.wait_for_admission_encoder(timeout=0)

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

            await self._start_outbound_receipts()

            if not await self._start_epoch_fence():
                return  # sys.exit is stubbed in some harnesses; never fall through to a live bot
        else:
            main_logger.error(f"Unknown platform: {self.platform}")
            sys.exit(1)
        
        if self.pending_share_recovery:
            # After the client is constructed (it owns resolve_file_share_ts) but off the boot
            # path: a Slack poll per leftover file must not hold the bot back from serving.
            # TRACKED, and cancelled+awaited in shutdown() before the receipt service and the
            # client go away — it writes receipts through both, and a leftover row it never
            # reaches is simply retried at the next boot, which is what those rows are for.
            self.pending_share_recovery = False
            self._pending_share_task = asyncio.create_task(
                outbound_receipts.recover_pending_shares(self.client.db, self.client))
            self._pending_share_task.add_done_callback(
                lambda t: t.cancelled() or (t.exception() and main_logger.warning(
                    f"Pending share recovery error: {t.exception()}")))

        # T1: re-learn the scheduled messages Slack is still holding for us. Same seam and same
        # reason as the share recovery above — it needs the client, and one listing must not sit
        # on the boot path — but its OWN condition, because it is not that recovery.
        #
        # Starting it at boot rather than lazily is what closes the first-event race: a delivery
        # can easily be the first own post a restarted process sees, and by then Slack no longer
        # lists that message as pending, so a listing beginning at that moment can never find it.
        # The ingress hook JOINS this task (rehydrate_scheduled_deliveries is single-flight), so an
        # event arriving mid-listing waits for it instead of matching an empty registry.
        if self.receipt_service is not None and self.client is not None:
            self._scheduled_rehydrate_task = outbound_receipts.start_scheduled_rehydrate(
                getattr(getattr(self.client, "app", None), "client", None),
                team_id=getattr(self.client, "self_team_id", None))
            if self._scheduled_rehydrate_task is not None:
                self._scheduled_rehydrate_task.add_done_callback(
                    lambda t: t.cancelled() or (t.exception() and main_logger.warning(
                        f"Scheduled-delivery rehydrate error: {t.exception()}")))

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        main_logger.info("Initialization complete")
    
    async def _start_epoch_fence(self) -> bool:
        """Start the dev-only epoch fence watcher. False means REFUSE TO START.

        Placed AFTER the receipts service and BEFORE the coverage bootstrap, so no channel work
        runs against an unstamped fence.

        WITHOUT the flag this does nothing at all — and the module was never even imported, so the
        control file is never opened, the fence table is never created and no watcher task exists.
        That is the whole of the production no-op contract, and it is why this returns True on the
        very first line for every production process.

        WITH the flag, EVERY failure here is FATAL, and that asymmetry is the point. A fence
        exists to make a battery's results trustworthy. A bot that boots anyway — unfenced,
        against the channel it was told to fence — produces results that LOOK isolated and are
        not, and nobody reading them afterwards would have any way to tell. Refusing to start is
        the only honest failure: the operator sees it immediately, and nothing has run yet.
        """
        if not dev_epoch_fence_requested():
            return True

        if epoch_fence is None:
            main_logger.error(
                f"DEV_EPOCH_FENCE_ENABLE is set but the epoch fence module could not be imported "
                f"({type(_epoch_import_error).__name__}: {_epoch_import_error}). Refusing to "
                f"start: a battery would run against an UNFENCED channel and report contaminated "
                f"results as isolated. Fix the import, or clear the flag to run as an ordinary "
                f"bot.")
            sys.exit(1)
            return False

        from message_processor.epoch_fence_control import EpochFenceWatcher
        main_logger.warning(
            "DEV_EPOCH_FENCE_ENABLE is set — the epoch fence watcher is starting. This is test "
            "infrastructure and must never be set in production.")
        self.epoch_fence_watcher = EpochFenceWatcher(self.client)
        self.epoch_fence_watcher.start()

        # AWAITED, not fire-and-forget. Startup INVALIDATES any battery a restart interrupted and
        # installs the deny-only fence that keeps that scope shut; if service began before that
        # landed, the interrupted channel would answer normally — unfenced, mid-battery — in
        # exactly the window the invalidation exists to close. One edge covers every way startup
        # can fail: unresolved identity, a refused control directory, a raise inside `_boot`, and
        # the timeout itself.
        if not await self.epoch_fence_watcher.wait_ready():
            main_logger.error(
                "The epoch fence watcher did not finish initialising (the error above says which "
                "step failed). Refusing to start: a battery a restart interrupted may not be "
                "fenced shut, and a new one would run unfenced.")
            await self.epoch_fence_watcher.stop()
            self.epoch_fence_watcher = None
            sys.exit(1)
            return False
        return True

    async def _start_outbound_receipts(self):
        """Spec §5 boot order: epoch, then dead-session reconciliation, then pending shares.

        The epoch is FATAL on failure. Without it the rebuild has no way to tell an own-message
        that predates this feature from one this build simply failed to register — it would
        either replay old chrome forever or drop real replies, silently, and the operator would
        see a bot that misremembers its own words rather than a bot that refused to start.
        """
        db = self.client.db
        try:
            epoch = await outbound_receipts.establish_epoch(db)
        except Exception as e:  # noqa: BLE001 — reported, then fatal
            main_logger.error(
                f"Outbound receipts epoch could not be established ({e}). Refusing to start: "
                "without it the channel stream cannot tell legacy own-messages from unregistered "
                "ones. Fix the database and restart.")
            sys.exit(1)
            return
        main_logger.info(f"Outbound receipts epoch: {epoch}")

        self.receipt_service = outbound_receipts.install_service(db)

        # Bounded retry, then FATAL. A reconcile that never ran leaves the previous session's
        # in_flight rows in_flight for this whole process, and every message behind them stays
        # out of the rebuilt stream — the bot would answer all day about a conversation it
        # cannot see its own half of, with one warning line at boot to explain it.
        last_error = None
        for attempt in range(3):
            try:
                # Through the wrapper, not the accessor: the reconcile writes one CV8
                # outbound_receipt row per receipt it finalizes, and boot is the only place those
                # transitions are visible at all.
                await outbound_receipts.reconcile_dead_sessions(db, outbound_receipts.SESSION_ID)
                last_error = None
                break
            except Exception as e:  # noqa: BLE001 — retried, then fatal
                last_error = e
                main_logger.warning(
                    f"Dead-session receipt reconciliation failed (attempt {attempt + 1}/3): {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            main_logger.error(
                f"Dead-session receipt reconciliation failed ({last_error}). Refusing to start: "
                "the previous session's replies would stay excluded from the channel stream for "
                "this entire run. Fix the database and restart.")
            sys.exit(1)
            return

        # Deferred to initialize()'s tail — it needs the Slack client to resolve share ts.
        # A FLAG, not a coroutine: anything can raise between here and there, and a coroutine
        # that is built and never scheduled is a warning from the GC with no context attached.
        self.pending_share_recovery = True

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
            #
            # W5c: the answer is KEPT and handed to evaluate() below. It says what this stream
            # looked like at the instant this message arrived — which is the only moment that
            # describes it. The steering load between here and there is real I/O, and during it
            # another message in this stream can arrive or the arrival map can evict the stream;
            # asking again down there would judge this message on somebody else's moment.
            arrival = engine.note_arrival(channel_id, ts, message.thread_id, message.user_id)

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
            # recent-activity envelope, a thread tail, an addressee tail, a channel topic, a member
            # count, a recent-speakers line, a canvas catalogue, a capability inventory, a
            # custom-emoji shortlist and a channel narrative — half a dozen API calls and cache
            # reads on the hot path of a decision the whole turn waits for — to support judgments
            # about addressee, answerability and which emoji to place. The gate makes none of those
            # judgments now, so every one of those inputs was work spent on a question nobody
            # asks.
            evaluation = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_id=message.user_id,
                sender_name=(message.metadata.get("user_real_name")
                             or message.metadata.get("username")),
                sender_type=message.metadata.get("sender_type"),
                channel_steering_text=steering.gate_text,
                # Names and types, already summarized at dispatch. No pixels: the binary gate
                # does not look at images, so nothing is downloaded for it and nothing waits on it.
                attachments=message.metadata.get("participation_attachments"),
                # The raw file payload behind those names, carried UNREAD. The gate never opens it;
                # it is retained because a cohort member's own dispatch ends here, and the turn that
                # survives needs the ids to stay authorizable when Slack's fetch has not yet caught
                # up to the message that carried them.
                file_payloads=message.attachments or (),
                client=client,
                thread_root_ts=message.thread_id,
                # The edit's OWN marker, so only the edit's attempt can claim the stashed
                # before/after text — the superseded original carries no marker and gets none.
                edit_marker=(message.metadata or {}).get("edit_reply_marker"),
                # A Phase-Q drain folds queued messages into this turn; they never had a debounce
                # window of their own, so they join the cohort here rather than being decided for
                # in absentia by whatever happened to arrive last.
                carried_sources=(message.metadata or {}).get("carried_gate_sources"),
                # …and that same drain already spent a coalescing window holding the lock, so the
                # debounce below it would be a second one. The turn is not a fresh arrival.
                queue_drained=bool((message.metadata or {}).get("queue_drained")),
                # This message's own arrival record, from the note above — not re-derived.
                arrival=arrival,
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
                # And their files, so a burst that dropped a CSV and then asked about it can READ
                # it on this turn even if the fetch window has not seen that message yet. Merges
                # with whatever a queue drain already staged; neither writer erases the other's.
                if evaluation.source_files:
                    from message_processor.channel_request import stage_cohort_file_payloads
                    stage_cohort_file_payloads(message.metadata, evaluation.source_files)
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
    def _committed_correction_announcement(turn) -> bool:
        """EDIT §7: whether this turn committed a correction disclosure — its OWN words in the
        room, so wherever it and `detached` could both apply, `reply` wins (§11.5)."""
        return turn is not None and any(
            getattr(r, "kind", None) == DEST_KIND_CORRECTION_ANNOUNCEMENT
            for r in (getattr(turn, "committed_destinations", None) or ()))

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
            #
            # EDIT §11.5: the committed-correction-announcement ⇒ `reply` override runs BEFORE
            # this terminal `detached` return. The disclosure sets visible_action_committed
            # too, but it is this turn's OWN words in the room — a reply, not a detached
            # producer's surface.
            if ChatBotV2._committed_correction_announcement(turn):
                return "reply"
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
        # EDIT §7: a committed correction announcement is this turn's OWN words in the room —
        # the disclosure edit_own_message posted before overwriting anything. An empty-response
        # turn that committed one spoke, so it is a `reply`, not a detached producer.
        if ChatBotV2._committed_correction_announcement(turn):
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
                                     post_thread_id: str, receipts=None) -> int:
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
                    receipts=receipts,
                )
                posted += 1
            except Exception as e:
                main_logger.error(f"Sandbox image rescue failed: {e}", exc_info=True)
        return posted

    @staticmethod
    async def _drop_chrome(client, turn, channel_id: str, ts: str) -> bool:
        """Delete a chrome surface and drop its receipt — but only once Slack CONFIRMS the
        delete. A row removed for a message still sitting in the channel would let that
        message back into the stream as an own-message nobody claims."""
        gone = await client.delete_message(channel_id, ts)  # unleased-ok: teardown — removing a surface can never be a stale answer
        ledger = getattr(turn, "receipt_ledger", None) if turn is not None else None
        if gone and ledger is not None:
            await ledger.abort(ts)
        return bool(gone)

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
        # Shutdown closes admission BEFORE it quiesces (see `shutdown`), and this is the same
        # boundary the lease defines: refused here, the message never becomes a turn, never
        # moves the watermark, and never opens a receipt ledger the closing queue would refuse.
        #
        # Read off `self` rather than assumed: this method is also driven unbound against plain
        # stand-in hosts, which have no lifecycle to close and are always admitting.
        if not getattr(self, "_admitting", True):
            main_logger.info(
                "Shutting down — message in %s refused before admission",
                getattr(message, "channel_id", None))
            return
        # Also records this message as its conversation's newest inbound (message_processor.
        # stale_send_guard). Nothing is dropped after this point, so the watermark and the set
        # of turns that exist stay in step.
        lease = self.watermarks.begin_turn(message)
        turn = None
        # Bound here, not at the inner try: the outer finally reports this turn's outcome, and it
        # is reachable from the early returns above that point. `outcome_kind` is whatever label
        # the path that ENDED the turn chose, so the turn row and the gate terminal can never
        # disagree about what the room saw; None means "derive it from the Response".
        response = None
        outcome_kind = None
        # A STRONG ref on the running turn, so shutdown can wait for what it already admitted
        # rather than closing the receipt queue underneath a reply that is still being written.
        # None for a stand-in host — nothing there is ever going to quiesce it.
        active_turns = getattr(self, "active_turns", None)
        turn_task = asyncio.current_task()
        if turn_task is not None and active_turns is not None:
            active_turns.add(turn_task)
        # Dev-only epoch fence: THE one stamping site. A turn is its own task, so this marks that
        # task's private context and everything it spawns inherits the mark. A no-op returning
        # None on every unfenced channel, which in production is all of them.
        if epoch_fence is not None:
            epoch_fence.stamp_current_task(
                getattr(client, "self_team_id", None), getattr(message, "channel_id", None))
        try:
            # Before the gate, because a turn that cannot be accounted for should not run at
            # all — not even to decide whether to speak.
            if not await channel_identity_ready(client, message):
                return

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
            # ...and its receipt ledger, opened here so the FIRST thing this turn can post is
            # already covered. Settled in the outer finally below, under a shield.
            turn.bind_receipts(client, message)
            # getattr, like the two calls in the outer finally: this method is also driven
            # unbound against plain stand-in hosts, and a ledger line must never be the reason a
            # turn fails to run.
            emit_start = getattr(self, "_emit_turn_start", None)
            if emit_start is not None:
                emit_start(message, turn, gate_required=gate_required)
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
                # cast: typing-only, a runtime no-op — the value is passed through unchanged.
                # `progress_enabled` is True only when the destination is not
                # DESTINATION_CHANNEL and is already selected (turn_runtime.py), and
                # resolve_reply_target returns None only on the CHANNEL branch; so in here it
                # returns `message.thread_id`, which Message declares as `str`. Nothing between
                # the turn's construction above and this line changes either field.
                thinking_id = await client.send_thinking_indicator(
                    message.channel_id,
                    cast(str, post_thread_id),
                    receipts=turn.receipt_ledger,
                    receipt_class="chrome",
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
                # that knows — so the fallback send, the footer guard and the artifact upload
                # below all agree with where the handler actually posted.
                post_thread_id = turn.resolve_reply_target(message)

                # Delete thinking indicator (but not if streaming was used — it's already the
                # response — and not when a ProgressChecklist owns the thinking message, F4).
                if (thinking_id and response
                        and not response.metadata.get("streamed")
                        and response.metadata.get("checklist") is None):
                    await self._drop_chrome(client, turn, message.channel_id, thinking_id)
                elif thinking_id and not response:
                    await self._drop_chrome(client, turn, message.channel_id, thinking_id)

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
                            send_meta: Dict[str, Any] = {}
                            # The reply is going out NOW, in this place: the destination stops
                            # being a preference and becomes a fact about Slack. The streaming
                            # paths lock when they bind their surface; this is the same moment
                            # for a non-streamed reply, which main.py posts itself.
                            turn.lock_destination()

                            def _observe_reply(ts: str, _turn=turn, _thread=post_thread_id
                                               ) -> None:
                                """Slack took the first part. Recorded at THAT instant rather
                                than after the send returns: a cancellation between the first
                                accepted chunk and the last would otherwise leave visible text
                                and receipts with nothing in the ledger claiming them."""
                                _turn.note_destination_observed(
                                    channel_id=message.channel_id, first_ts=ts,
                                    kind=DEST_KIND_REPLY, thread_root_ts=_thread)

                            async def _reconsidered_reply_send(text: str,
                                                               _response=response,
                                                               _meta=send_meta,
                                                               _blocks=footer_blocks,
                                                               _thread=post_thread_id
                                                               ) -> Optional[str]:
                                """The §4b delivery closure for this site. Replaces the site's
                                canonical text — `response.content` — BEFORE the send, so the
                                footer guard, the destination commit and the F7 persistence
                                below all read the chosen text with no second code path; then
                                re-runs the site's OWN send (same target, F39 top-level None
                                stays None). Returns send_message's native Optional ts;
                                StaleSendSuppressed propagates (a re-race is the next pass)."""
                                # W4: the revised text is FRESH model output — the runner asked
                                # the model to rewrite its own draft — so it can mint a marker
                                # the original never had. It goes through the same choke point
                                # as every other piece of model text before it becomes
                                # `response.content`, which is what this send posts, what F7
                                # persists, and what the thread state and compaction inherit.
                                # Selection is refused by here (the destination locked when this
                                # site claimed it); the strip is what matters.
                                _response.content = consume_destination_marker(
                                    text, turn=turn, message=message)
                                _meta.clear()
                                return await cast(Any, client).send_message(
                                    message.channel_id, _thread, _response.content,
                                    blocks=_blocks, meta_out=_meta, lease=lease,
                                    receipts=turn.receipt_ledger,
                                    receipt_class="assistant_reply",
                                    on_first_accept=_observe_reply)

                            # The stale guard's last chance on this path: the lease refuses
                            # rather than posting if the conversation moved on while the model
                            # was writing. Raises StaleSendSuppressed — and a COMPLETE drafted
                            # reply refused at its first surface is exactly the reconsideration
                            # case (STALE_RECONSIDERATION §3), so the suppression is handed to
                            # the runner instead of straight to the terminal catch. DMs go too
                            # now, over their own surface snapshot
                            # (STALE_SUPPRESSION_RECONSIDERATION ruling 6); `channel_turn` says
                            # WHICH surface, not whether. The runner either
                            # returns a delivered ts (bookkeeping below proceeds normally),
                            # returns None on its accounted delivery_failed and
                            # delivery_exception endings, or rethrows the suppression to the
                            # terminal catch below.
                            try:
                                sent_ts = await cast(Any, client).send_message(
                                    message.channel_id,
                                    post_thread_id,
                                    response.content,
                                    blocks=footer_blocks,
                                    meta_out=send_meta,
                                    lease=lease,
                                    receipts=turn.receipt_ledger,
                                    receipt_class="assistant_reply",
                                    on_first_accept=_observe_reply,
                                )
                            except StaleSendSuppressed as stale:
                                sent_ts = await intercept_stale_send(
                                    processor=self.processor, client=client,
                                    message=message, turn=turn, lease=lease,
                                    suppressed=stale, draft=response.content or "",
                                    deliver=_reconsidered_reply_send,
                                    # THE discriminator the handlers use (`_turn_surface`), not
                                    # a second copy of it: a "D…" test misses the DM whose
                                    # channel id is the recipient's "U…" id, and the two halves
                                    # of one turn must not disagree about which surface it is on.
                                    channel_turn=not is_dm_conversation(message.channel_id))
                            # Honest accounting: the ACTUAL send result decides `posted` (a
                            # failed send must not burn the hourly unprompted quota).
                            if isinstance(response.metadata, dict):
                                response.metadata["posted"] = bool(sent_ts)
                                # Only stand the separate footer down when the chrome ACTUALLY
                                # rode the message (a split/too-long reply doesn't attach it, so
                                # the separate footer post must still happen).
                                if sent_ts and send_meta.get("footer_attached"):
                                    response.metadata["footer_attached"] = True
                            # Destination record, COMMITTED from the transport's own account of
                            # what Slack accepted — never from the text we handed it. A reply
                            # that split and then failed partway commits its delivered prefix
                            # and carries the flag saying so, which is what stops channel
                            # memory remembering an answer the room never fully saw.
                            # A ts recovered by the uncertain-post reconcile is labelled as such:
                            # the words are in the room, but nobody watched them land.
                            delivery = send_meta.get("delivery")
                            if sent_ts and turn is not None:
                                if send_meta.get("reconciled"):
                                    kind = DEST_KIND_RECONCILED
                                elif delivery is not None and delivery.split:
                                    kind = DEST_KIND_SPLIT
                                else:
                                    kind = DEST_KIND_REPLY
                                turn.mark_destination_committed(
                                    first_ts=sent_ts, kind=kind,
                                    text=(delivery.text if delivery is not None
                                          else response.content or ""),
                                    complete=(delivery.complete if delivery is not None
                                              else True),
                                    channel_id=message.channel_id,
                                    thread_root_ts=post_thread_id)
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
                                await client.maybe_post_response_footer(
                                    message, response, receipts=turn.receipt_ledger)
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
                        # STALE_RECONSIDERATION §4b (r6-1): a reconsideration that ended in ANY
                        # non-posted outcome suppresses BOTH rescue paths — this publish and the
                        # sandbox-image rescue below — so a failed reconsidered reply can never
                        # be followed by a published artifact or image. Rethrow endings get this
                        # for free (the exception propagates past these blocks); the
                        # delivery_failed and delivery_exception RETURN paths are the ones that
                        # need the explicit gate.
                        reconsider_facts = getattr(turn, "reconsider", None)
                        reconsider_nonposted = bool(
                            reconsider_facts is not None
                            and reconsider_facts.outcome not in ("posted_asis",
                                                                 "posted_revised"))
                        published = []
                        if artifact_containers and reconsider_nonposted:
                            main_logger.warning(
                                "Reconsideration ended without a post — suppressing its "
                                "artifacts (a file with no answer above it reads as a bug)")
                        elif artifact_containers and (reply_landed or files_only):
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
                                        receipts=turn.receipt_ledger,
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
                        if not published and not reconsider_nonposted:
                            # B2: rescued sandbox images always thread — pass message.thread_id, not
                            # post_thread_id (None on a top-level channel reply).
                            # `send_image` carries no lease, so without the reconsideration gate
                            # above this rescue would post behind a reply the guard refused.
                            rescued = await self._rescue_sandbox_images(
                                response, client, message, message.thread_id,
                                receipts=turn.receipt_ledger)
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
                            lease=lease,
                            receipts=turn.receipt_ledger,
                        )

                # Close the attempt with what the room actually SAW. Deliberately not folded into
                # the contract check below, which asks a narrower question — is this channel one we
                # speak in at all — that has nothing to do with whether the outcome is worth
                # counting. `finish_attempt` is a no-op unless this turn came from the gate and is
                # still open, which is what keeps mentions, DMs and direct continuations out of a
                # ledger documented as gate attempts — and what keeps a turn the gate already
                # closed (a react verdict that fell through to nothing) from closing twice.
                meta = (response.metadata or {}) if response is not None else {}
                # No gate reaction to account for any more: the gate places nothing in the room, so
                # every reaction on this message came from this turn.
                kind = outcome_kind = self._classify_visible_action(response, turn)
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
                # The old form of this check also required a live ChannelPulse — a delivery-side
                # object with nothing to do with whether an outcome is worth counting. With the
                # pulse retired the condition would have been permanently false, silently
                # switching off the one check that catches a turn which posted nothing and never
                # called the terminal tool.
                if (response and message.channel_id
                        and not message.channel_id.startswith("D")):
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
                                and not response.metadata.get("reaction_only")
                                and not getattr(turn, "visible_action_committed", False)):
                            # Bare empty text without the terminal tool: contract violation.
                            # Fail-safe silence, no re-prompt this phase. A turn whose words
                            # landed elsewhere (committed cross-thread post) is NOT this case.
                            main_logger.warning(
                                "Empty text response without a terminal action — posting nothing")

            except StaleSendSuppressed as stale:
                # NOT an error, and it must never reach the handler below — that one logs an
                # exception, files `error_unhandled`, and posts an apology. Nothing went wrong
                # here: a newer message arrived while this turn was writing, so its answer was
                # deliberately not created. The room sees nothing, which is the point.
                main_logger.info(
                    f"Stale send suppressed on {message.channel_id}: {stale}")
                # Single-owner rule (STALE_RECONSIDERATION §5): the reconsideration runner marks
                # every suppression it already wrote a `stale_send` row for. The marker skips
                # EMISSION and only emission — retraction and cleanup below run either way.
                if not getattr(stale, "telemetry_recorded", False):
                    participation_telemetry.stale_send(
                        message.channel_id, (message.metadata or {}).get("ts"),
                        attempt_id=participation_telemetry.attempt_id_for(message),
                        turn_id=getattr(turn, "turn_id", None) if turn is not None else None,
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
                        await self._drop_chrome(client, turn, message.channel_id, thinking_id)
                    except Exception as cleanup_error:  # noqa: BLE001
                        main_logger.debug(f"Stale-suppression cleanup failed: {cleanup_error}")
                if hasattr(client, "clear_assistant_status"):
                    try:
                        await client.clear_assistant_status(message.channel_id, post_thread_id)
                    except Exception as cleanup_error:  # noqa: BLE001
                        main_logger.debug(f"Stale-suppression status clear failed: {cleanup_error}")
                outcome_kind = self._stale_terminal_kind(response, turn, message)
                participation_telemetry.finish_attempt(
                    message,
                    # Something else may still be visible: a reaction the gate or the responder
                    # placed, or a detached producer's own surface. Those turns are not silent,
                    # and the suppression rides as a separate fact rather than overwriting the
                    # louder one.
                    outcome_kind,
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
                    outcome_kind = self._classify_visible_action(response, turn)
                    participation_telemetry.finish_attempt(
                        message, outcome_kind,
                        ended_by="responder", post_delivery_error=type(e).__name__)
                else:
                    outcome_kind = "error_unhandled"
                    participation_telemetry.finish_attempt(
                        message, "error_unhandled", ended_by="responder",
                        detail=type(e).__name__)

                # Delete thinking indicator on error — best-effort; a failed delete
                # must never swallow the user-facing notice below.
                if thinking_id:
                    try:
                        await self._drop_chrome(client, turn, message.channel_id, thinking_id)
                    except Exception as delete_error:
                        main_logger.error(f"Failed to delete thinking indicator: {delete_error}")

                # Fixed, friendly notice — the raw exception stays in the logs only.
                try:
                    await client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        "⚠️ **Something Went Wrong**\n\n"
                        "I hit a snag finishing that response. Please try again in a moment.",
                        lease=lease,
                        receipts=turn.receipt_ledger if turn is not None else None,
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
                        turn_id=getattr(turn, "turn_id", None) if turn is not None else None,
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
            # Everything this turn CAUSED is settled before anything it POSTED is, and the whole
            # sequence — drain, cancel, revoke, wait out the live effects, settle the receipts —
            # is ONE unit that owns itself. Awaited through a shield for exactly that reason: a
            # cancellation landing on this await used to be caught and stepped over, which
            # skipped revocation and then settled anyway, and a shielded straggler could take a
            # lease and post AFTER settlement. The unit is not something a cancellation may
            # interleave with; it is the thing that makes the cancellation safe.
            await _await_finalizer(asyncio.ensure_future(_finalize_turn_effects(turn)))
            # Channel memory reads what the room actually SAW — the COMMITTED destination records
            # — and is therefore scheduled from HERE, after every commit point by construction.
            # A silent turn, a suppressed one, and one that died mid-stream all commit nothing and
            # write nothing. The in-handler scheduling (handlers/text.py) stays DM-only, where the
            # exchange is in ThreadState.messages and nothing else knows it.
            # The turn's own row, exactly once, from the turn's accumulated state. It sits after
            # every commit point by construction — this finally cannot run before the handlers
            # returned — and BEFORE the memory scheduling below, which reads the committed subset
            # of the same destination records.
            emit_outcome = getattr(self, "_emit_turn_outcome", None)
            if emit_outcome is not None:
                emit_outcome(message, turn, response, kind=outcome_kind)
            schedule_memory = getattr(self, "_schedule_channel_memory", None)
            if schedule_memory is not None:
                schedule_memory(message, turn)
            participation_telemetry.abort_attempt(message)
            if turn_task is not None and active_turns is not None:
                active_turns.discard(turn_task)
            # Release the scope hold LAST. An entry survives while any lease in its scope is
            # open, so a newer turn that finishes early cannot erase the watermark an older,
            # still-running turn is about to read.
            lease.close()

    @staticmethod
    def _turn_population_surface(message: Message) -> str:
        """`channel` or `dm` — the same discriminator receipts use, never a prefix test."""
        return ("channel" if outbound_receipts.receipts_apply(
            getattr(message, "channel_id", None)) else "dm")

    @staticmethod
    def _turn_telemetry_scope(message: Message, turn) -> bool:
        """Do this turn's rows belong in the turn population? EVERY turn with a runtime does.

        This used to be channel-only, on the reasoning that a DM has neither a stream nor a
        receipt to describe. That stopped being tenable when DM drafts started being
        reconsidered (STALE_SUPPRESSION_RECONSIDERATION ruling 8): a DM turn now writes
        `stale_send`, `reconsider_start`, `reconsider_outcome` and `model_response` rows, all
        keyed by `turn_id` — turn-population rows by construction — and a population row with
        NO terminal breaks the one property that makes any of this readable, exactly-one
        terminal per turn. So DM turns carry their own `turn_start`/`turn_outcome` pair, under
        `surface="dm"` (a value the grammar has always declared), and the two surfaces are
        counted SEPARATELY: a rate that pools them describes neither.
        """
        return turn is not None

    def _emit_turn_start(self, message: Message, turn, *, gate_required: bool) -> None:
        """The turn population's denominator. Written before anything can be posted."""
        if not self._turn_telemetry_scope(message, turn):
            return
        meta = message.metadata or {}
        participation_telemetry.turn_start(
            message.channel_id, meta.get("ts"),
            turn_id=getattr(turn, "turn_id", None),
            origin_thread_ts=message.thread_id,
            surface=self._turn_population_surface(message),
            gated=bool(gate_required),
            attempt_id=participation_telemetry.attempt_id_for(message),
            wake_source=meta.get("wake_source"))

    def _emit_turn_outcome(self, message: Message, turn, response, *,
                           kind: Optional[str] = None) -> None:
        """Close the turn's row. `kind` is the label the path that ended the turn already chose —
        an unhandled raise is `error_unhandled`, not the `empty` a missing Response would imply —
        and it falls back to the same classifier the gate terminal uses.

        `detached_started` means a producer owned its own surface this turn — a picture, a status
        card — which is a delivery the destination records cannot see."""
        if not self._turn_telemetry_scope(message, turn):
            return
        meta = (response.metadata or {}) if response is not None else {}
        participation_telemetry.emit_turn_outcome(
            turn, channel_id=message.channel_id,
            trigger_ts=(message.metadata or {}).get("ts"),
            kind=kind or self._classify_visible_action(response, turn),
            detached_started=bool(meta.get("background_job_started")
                                  or getattr(turn, "visible_action_committed", False)),
            attempt_id=participation_telemetry.attempt_id_for(message))

    def _schedule_channel_memory(self, message: Message, turn) -> None:
        """Extract a durable channel fact from what this turn COMMITTED, if it committed anything.

        Reads the committed destination records rather than the intended reply, because those are
        the two facts a memory has to be about: somebody said something, and we answered. A turn
        that produced words and then failed to deliver them has no exchange to remember, and
        recording one would have the bot remember a conversation the room never had.
        """
        if turn is None or not getattr(turn, "stream_build_present", False):
            return
        if not getattr(config, "enable_memory_extraction_fallback", False):
            return
        # A FOREIGN post is not this exchange. post_to_thread lands in another thread of the
        # channel (its executor refuses the current one outright), so the pairing "what was asked
        # HERE" + "what we said THERE" is not an exchange that happened anywhere — and storing it
        # as one would have the bot remember a conversation in the wrong room. The post stays
        # observable in turn_outcome's destinations; grouping it with its own thread's evidence is
        # P4's, where the target's side of the exchange is available to group it with.
        # A CORRECTION ANNOUNCEMENT is excluded for the same reason: it is disclosure chrome
        # about an edit, not this exchange's answer — remembering "what was asked HERE" paired
        # with "Correction to my earlier message…" would store a conversation nobody had.
        committed = [r for r in turn.committed_destinations
                     if (r.text or "").strip()
                     and r.kind not in (DEST_KIND_POST_TO_THREAD,
                                        DEST_KIND_CORRECTION_ANNOUNCEMENT)]
        if not committed:
            return
        processor = self.processor
        if processor is None or not hasattr(processor, "extract_channel_memory_from_exchange"):
            return
        try:
            processor._schedule_async_call(processor.extract_channel_memory_from_exchange(
                message.channel_id, getattr(message, "text", "") or "",
                "\n\n".join(r.text or "" for r in committed)))
        except Exception as e:  # noqa: BLE001 — memory is never worth a turn
            main_logger.debug(f"Channel memory extraction not scheduled: {e}")

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
                # Schedule THE shutdown on the event loop (run()'s finally awaits the same one)
                self.begin_shutdown()
            else:
                main_logger.warning("Shutdown already in progress... Press Ctrl-C again to force exit")
        else:
            # Handle other signals normally
            main_logger.info(f"Received signal {signum}, shutting down...")
            # Schedule THE shutdown on the event loop (run()'s finally awaits the same one), so a
            # SIGTERM's ledger drain is never cut short by the loop closing under it.
            self.begin_shutdown()
    
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
                            # file older than the window stays resolvable indefinitely.
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

                            # Receipts for messages Slack's own retention policy has already
                            # deleted. Slack announces none of that and exposes no retention API
                            # off Grid, so the boundary is inferred: one conversations.history
                            # probe per channel holding receipts, and in the normal case (nothing
                            # has aged out yet) that probe is the whole cost and nothing is
                            # deleted. Best-effort — a skipped channel is retried tomorrow.
                            try:
                                from message_processor.outbound_receipts import (
                                    sweep_receipts_past_retention)
                                await sweep_receipts_past_retention(
                                    self.processor.db,
                                    getattr(getattr(self.client, "app", None), "client", None))
                            except Exception as e:
                                main_logger.debug(f"Receipt retention sweep skipped: {e}")

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
    
    def begin_shutdown(self) -> asyncio.Task:
        """Start the one shutdown, or hand back the one already running. Idempotent.

        Never called from the raw signal handler's own frame for anything but this: the handler
        runs between bytecodes with no loop of its own, so all it may do is schedule.

        The REQUEST is recorded first and durably, because a request can arrive before there is
        anything to stop — a SIGTERM between installing the handlers and the run loop starting.
        The task that request created returns immediately (nothing is running yet) and is NOT
        allowed to stand as the shutdown: a later caller gets a fresh one, and `run()` checks the
        flag the moment it comes up so the signal is honored rather than lost.
        """
        self._shutdown_requested = True
        task = self._shutdown_task
        if task is not None and task.done() and not self._shutdown_completed:
            task = self._shutdown_task = None
        if task is None:
            task = self._shutdown_task = asyncio.create_task(self.shutdown())
        return task

    async def run(self):
        """Run the bot"""
        log_session_start()
        self._run_task = asyncio.current_task()

        fatal = False
        try:
            await self.initialize()
            self.running = True

            # A signal that arrived DURING initialization is honored here, at the first moment
            # there is something to stop. The handlers are installed inside `initialize`, so this
            # window is real: without this check the bot would come up and serve, having been
            # told to stop, and the finally below would await the no-op task that request left
            # behind. `begin_shutdown` refuses to hand that task back, so this starts a real one.
            if self._shutdown_requested:
                main_logger.info(
                    "A shutdown was requested during initialization — stopping without starting")
                await self.begin_shutdown()
                return

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

            # Spec §4: extend per-channel coverage backward in the background. Waits for the
            # bot's identity itself, so it starts before auth.test has landed.
            if self.client is not None and getattr(self.client, "db", None) is not None:
                try:
                    from slack_client.event_handlers.activity_index import ChannelCoverageBootstrap
                    self.coverage_bootstrap = ChannelCoverageBootstrap(self.client)
                    self.coverage_bootstrap.start()
                except Exception as e:
                    main_logger.warning(f"Coverage bootstrap start skipped: {e}")

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
            # The SAME task a signal would have started — awaited to completion, so the telemetry
            # drain and its `session_end` finish before this coroutine returns and the loop closes.
            try:
                await self.begin_shutdown()
            except asyncio.CancelledError:
                main_logger.warning("Shutdown was cancelled before it finished draining")
                raise
            except Exception as e:  # noqa: BLE001 — shutdown reports its own failures
                main_logger.warning(f"Shutdown ended with: {e}")

        if fatal:
            sys.exit(1)

    async def _quiesce_turns(self, timeout: float = TURN_QUIESCE_TIMEOUT_SECONDS) -> None:
        """Close admission, then let the turns already running finish.

        Spec §5: a receipt row can only be written while the queue is open, so a turn that is
        still posting when the queue closes has its registration AND its settle refused — the
        message sits in Slack with nothing claiming it, and the rebuilt stream will never
        contain it. Bounded, because a wedged model call must not hold shutdown open forever: a
        turn that overruns is cancelled, and its `finally` still settles the ledger (shielded)
        while the queue is open.
        """
        self._admitting = False
        pending = [t for t in list(self.active_turns)
                   if t is not asyncio.current_task() and not t.done()]
        if not pending:
            return
        main_logger.info(f"Waiting for {len(pending)} in-flight turn(s) to finish...")
        _done, still_running = await asyncio.wait(pending, timeout=timeout)
        if still_running:
            main_logger.warning(
                f"{len(still_running)} turn(s) did not finish in {timeout:.0f}s — cancelling")
            for task in still_running:
                task.cancel()
            await asyncio.gather(*still_running, return_exceptions=True)

    async def shutdown(self):
        """Shutdown the bot gracefully"""
        if not self.running:
            return

        self.running = False
        main_logger.info(f"Shutting down {self.platform} bot...")

        # Dev-only epoch fence: the watcher goes FIRST, ahead of everything below, so no fence
        # transition lands against a process that is already tearing itself down. None in every
        # production process — the flag is empty there and the task was never created.
        if self.epoch_fence_watcher is not None:
            try:
                await self.epoch_fence_watcher.stop()
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Error stopping epoch fence watcher: {e}")
            self.epoch_fence_watcher = None

        # First, before anything is torn down: stop admitting new turns and quiesce the ones
        # already running. Everything below — background drains, the receipt queue, the client,
        # the database — is machinery a live turn is still using.
        #
        # Ticket issuance stays OPEN through all of it, deliberately, and closes far below only
        # once Slack ingress is provably quiet. Closing it any earlier — while Socket Mode can
        # still dispatch — leaves an interval in which an event enters `_admit`, gets no ticket,
        # and if its index write then fails there is nothing to retain it and nothing to fail: a
        # loss that can only be logged (codex r7, r3-8). Nothing is gained by closing early
        # either, because the retry worker and the database both outlive this phase.
        try:
            await self._quiesce_turns()
        except Exception as e:  # noqa: BLE001
            main_logger.warning(f"Error quiescing in-flight turns: {e}")

        # Spec §5: a turn is not the only thing that puts our own prose in a channel. The channel
        # intro is detached off the Slack ingress side — which stops far below, AFTER the receipt
        # queue closes — so an intro landing in that window would have its registration refused
        # and the bot's own introduction would sit in the room, permanently outside the stream.
        if self.client is not None and hasattr(self.client, "drain_channel_intros"):
            try:
                await self.client.drain_channel_intros()
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Error draining channel intros: {e}")

        # …and the same for the callbacks Bolt is still dispatching. Socket Mode stays connected
        # until `client.stop()` at the very bottom of this method, so a settings confirmation or
        # an onboarding notice can land long after the receipt queue has closed. Admission shuts
        # here and the callbacks already inside are waited out; anything arriving later is
        # refused BEFORE it posts, because an unsent notice costs less than an unaccounted one.
        try:
            await outbound_receipts.drain_channel_post_callbacks()
        except Exception as e:  # noqa: BLE001
            main_logger.warning(f"Error draining channel-posting callbacks: {e}")

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

        # Spec §5: the boot recovery polls Slack and writes receipts, so it stops before both.
        # A row it never got to is retained by contract and retried at the next boot.
        task = getattr(self, "_pending_share_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Pending share recovery stopped with: {e}")

        # T1: the same, for the scheduled-message listing. It writes no rows — the expectations it
        # rebuilds are in memory and die with the process — so it is simply stopped, not drained.
        task = getattr(self, "_scheduled_rehydrate_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Scheduled-delivery rehydrate stopped with: {e}")

        # Spec §5: the generic background set produces receipts too (image share resolution),
        # and it is the ONLY producer nothing else here waits on. It must be off the field
        # before the queue below is declared final, or its rows land in a closed queue.
        if self.processor is not None and hasattr(self.processor, "drain_background_tasks"):
            try:
                await self.processor.drain_background_tasks()
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Error draining background tasks: {e}")

        # Spec §5: producers are done by now, so drain what the receipt queue still holds and
        # stop its worker BEFORE the database goes away. Anything left after this is logged as
        # permanently omitted rather than lost quietly.
        if self.receipt_service is not None:
            try:
                await self.receipt_service.shutdown()
            except Exception as e:
                main_logger.warning(f"Error draining outbound receipts: {e}")

        # Spec §4: the coverage sweep has Slack calls in flight — stop it before the client.
        if self.coverage_bootstrap is not None:
            try:
                await self.coverage_bootstrap.stop()
            except Exception as e:
                main_logger.warning(f"Error stopping coverage bootstrap: {e}")

        # Spec §1 shutdown contract, and the ORDER below is the whole of it. Issuance stays open
        # until ingress is quiet, so the ticketless interval is closed: every event a callback
        # admits while Bolt is still winding down gets a ticket, and a failed index write is
        # therefore retained and repaired rather than logged as a loss nobody can undo. The one
        # residual is a callback that resists cancellation at the barrier's deadline, which is
        # CRITICAL-logged by name below rather than papered over.
        #
        #   stop the client → prove ingress quiet → close issuance → drain the retry worker →
        #   drain late receipts → (below) tear down the database.
        if self.client:
            try:
                await self.client.stop()
            except Exception as e:
                main_logger.warning(f"Error stopping client: {e}")

        # `client.stop()` returning is NOT quiescence. Bolt dispatches each event as its own task,
        # production stop force-marks sessions closed and abandons the handler's own close, and the
        # close it walks away from can still deliver events. The barrier waits out everything that
        # could dispatch, then drains the callbacks themselves and re-proves the zero — see
        # `_IngressTracker`. Bounded, and the three outcomes are told apart here rather than
        # collapsed into one number: quiet was granted, quiet was taken by cancelling stragglers at
        # the deadline (nothing can dispatch now, but a callback cancelled mid-write may not have
        # persisted what it saw), or something refused to be cancelled at all.
        try:
            drain = await registration.drain_ingress_callbacks(
                timeout=float(getattr(config, "ingress_drain_timeout_seconds", 5.0)))
            if drain.survived:
                main_logger.critical(
                    f"Slack ingress teardown gave up and {', '.join(drain.survived)} survived "
                    "cancellation; an observation made from here on has no worker to persist it")
            elif drain.gave_up:
                main_logger.critical(
                    "Slack ingress did not go quiet within the drain deadline; every straggler was "
                    "cancelled, so nothing can dispatch from here — but a callback cancelled "
                    "mid-write may have left its observation unpersisted")
            else:
                main_logger.info("Slack ingress is quiet: no callback running, nothing left that "
                                 "could start one")
        except Exception as e:  # noqa: BLE001
            main_logger.warning(f"Error draining Slack ingress callbacks: {e}")

        # Ingress is quiet, so nothing can take a ticket any more. NOW issuance closes — and it
        # closes before the worker is drained, so no repair can be enqueued behind the drain.
        try:
            admission_watermark.close_issuance()
        except Exception as e:  # noqa: BLE001
            main_logger.warning(f"Error closing admission-ticket issuance: {e}")

        # Drain and stop the index retry worker HERE, while the database it repairs through is
        # still open — after this the pending set is empty or each residual is logged CRITICAL with
        # its channel and ts, which is the only honest end state for an in-memory retry queue.
        try:
            await admission_watermark.shutdown()
        except Exception as e:  # noqa: BLE001
            main_logger.warning(f"Error draining the admission-index retry worker: {e}")

        # The `file_deleted` listener is a receipt producer that stays registered long after the
        # queue closes, and ingress being quiet means it has definitely returned. Anything it
        # retained on a transient DB failure is sitting in a closed queue with no worker; this is
        # the one moment when it can still be written (the database is open) and nothing else can
        # add to it.
        if self.receipt_service is not None:
            try:
                await self.receipt_service.drain_late_arrivals()
            except Exception as e:  # noqa: BLE001
                main_logger.warning(f"Error draining late receipt rows: {e}")

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
        # …but never the two tasks that ARE this shutdown: the one running it and the one
        # awaiting it. When a signal starts the shutdown, `run()` is a plain pending task from
        # here — cancelling it would cancel the await that is keeping the loop open for this very
        # drain, and the last thing to be written (session_end) would be the thing lost.
        protected = {asyncio.current_task(), self._shutdown_task, self._run_task}
        tasks = [t for t in asyncio.all_tasks() if t not in protected]
        if tasks:
            main_logger.warning(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            # BOUNDED, and the bound is not about slow cleanup. Cancelling a task that is itself
            # WAITING on this shutdown walks CPython's Task.cancel() down the await chain and back
            # into the gather below — the two cancel each other until the recursion limit, the
            # gather never completes, and the shutdown hangs with `session_end` unwritten, which is
            # the exact failure this whole ordering exists to prevent. The set above keeps the two
            # tasks we KNOW are in that chain out of it; this makes any other one survivable.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                names = ", ".join(t.get_name() for t in tasks if not t.done())
                main_logger.warning(
                    f"Leftover task(s) did not finish cancelling: {names or 'unnamed'}")
            except RecursionError:
                main_logger.error(
                    "A leftover task was waiting on this shutdown; its cancellation was abandoned")

        log_session_end()
        # This shutdown STOPPED something, so the task that ran it is the shutdown and every
        # later request may be handed it. A run that returned early because there was nothing
        # running never sets this, and cannot stand in for the real one.
        self._shutdown_completed = True
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