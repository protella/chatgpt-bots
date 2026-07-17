#!/usr/bin/env python3
"""
Multi-Platform Chat Bot V2 - Main Entry Point
Supports multiple chat platforms with shared AI capabilities
"""
import sys
import signal
import asyncio
import argparse
from typing import Any, Dict, Optional
from config import config
from logger import log_session_start, log_session_end, main_logger
from message_processor.base import MessageProcessor
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
    
    @staticmethod
    def _is_unprompted_turn(message: Message) -> bool:
        """F14: whether a posted channel reply counts as UNPROMPTED for pulse pacing.

        A participation-gated turn is unprompted UNLESS it was woken by a name-hit — being
        called by name is prompted in spirit (like an @-mention), so its reply must not
        burn the runaway-brake budget."""
        md = message.metadata or {}
        return (md.get("participation_check") is True
                and md.get("participation_name_hit") is not True)

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

        So: run the gate, and if the answer is anything other than "respond", catalog the files
        anyway. When the answer IS respond, the turn does the richer job (extraction, summaries,
        visual descriptions) and we leave it alone — `save_document` is a plain INSERT, so
        cataloguing here as well would just duplicate the row.
        """
        verdict = await self._gate_verdict(message, client)
        if verdict is None and (message.attachments or []):
            self.processor._schedule_async_call(
                thread_files.catalog_unattended(self.processor, client, message))
        return verdict

    async def _gate_verdict(self, message: Message, client: BaseClient):
        """Phase F gate for UNPROMPTED channel messages: hard rails → debounce →
        ONE engine call → act. Returns a verdict only for action='respond';
        every other outcome (ignore / react / backoff / superseded / any failure)
        is handled here and returns None so the caller stays silent."""
        engine = self.participation_engine
        if engine is None or not getattr(config, "enable_participation_engine", True):
            return None  # engine off → unaddressed messages stay unanswered (mentions_only behavior)
        try:
            channel_id = message.channel_id
            ts = message.metadata.get("ts") or message.thread_id
            level = message.metadata.get("participation_level") or "judicious"
            pulse = getattr(client, "channel_pulse", None)

            # F5 fix (b): register this message's ts as its conversation's newest BEFORE
            # the memory/topic awaits below — an older event delayed by that I/O must not
            # overwrite a newer event's debounce marker and win the race. F21: the marker
            # is conversation-scoped (message.thread_id is the thread root). F27: sender_id
            # scopes the top-level stream per author so different people's unrelated
            # top-level questions never collide.
            engine.note_arrival(channel_id, ts, message.thread_id, message.user_id)

            # F17: no hourly-cap hard rail — pacing is the classifier's judgment. The
            # unprompted-reply count is still tallied and fed to the engine as a signal
            # (below), but a high count never silences a turn before the model sees it.
            name_hit = message.metadata.get("participation_name_hit") is True

            channel_activity = None
            unprompted = 0
            if pulse is not None:
                channel_activity = pulse.render_envelope(
                    channel_id, exclude_thread_ts=None,
                    max_lines=config.channel_pulse_envelope_max,
                ) or None
                unprompted = pulse.unprompted_count_last_hour(channel_id)

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
            # are never injected over it). Deterministic sorted cap; stale-ok cache getter.
            workspace_custom_emojis = []
            if not (config.reaction_emojis or []):
                emoji_cache = getattr(client, "workspace_emojis", None)
                if emoji_cache is not None:
                    try:
                        names = emoji_cache.get_custom_emoji_names()
                        cap = max(0, int(getattr(config, "participation_custom_emoji_cap", 32)))
                        # cap is a hard maximum: 0 → none (never "unlimited"); names[:0] == [].
                        workspace_custom_emojis = list(names[:cap])
                    except Exception:  # noqa: BLE001 — never cost the gate a verdict
                        workspace_custom_emojis = []

            verdict = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_id=message.user_id,
                sender_name=message.metadata.get("user_real_name") or message.metadata.get("username"),
                is_thread_reply=is_thread_reply, level=level,
                directives=message.metadata.get("channel_directives"),
                memory_facts=memory_facts, channel_activity=channel_activity,
                unprompted_last_hour=unprompted,
                name_hit=name_hit,
                sender_is_bot=message.metadata.get("participation_sender_bot") is True,
                channel_topic=channel_topic,
                channel_canvases=channel_canvases,
                channel_people=channel_people,
                capabilities=capabilities,
                workspace_custom_emojis=workspace_custom_emojis,
                attachments=message.metadata.get("participation_attachments"),
                # F40: descriptors only — the engine downloads the pixels itself, and only once
                # the message has survived the debounce.
                images=message.metadata.get("participation_images"),
                client=client,
                pulse=pulse, thread_root_ts=message.thread_id,
            )
            if verdict is None:  # superseded by a newer message during debounce
                main_logger.debug("Participation gate: superseded during debounce — silent")
                return None
            main_logger.debug(f"Participation verdict: {verdict.action} ({verdict.reason})")

            if verdict.action == "react":
                react_ts = message.metadata.get("ts") or message.thread_id
                # F6: route the gate's own reaction through the reservation guard so a
                # later main-model turn on this message honestly sees the slot consumed
                # (and won't double-add the same emoji). Falls back to the raw react.
                try:
                    # Bound the gate's own react by the configured tool-call timeout so a
                    # wedged Slack call can't stall the turn (the model-invoked react tool
                    # is already timeout-guarded by the tool loop; this direct path wasn't).
                    if hasattr(client, "_reserve_and_react"):
                        await asyncio.wait_for(
                            client._reserve_and_react(channel_id, react_ts, verdict.emoji),
                            timeout=config.tool_call_timeout)
                    elif hasattr(client, "react"):
                        await asyncio.wait_for(
                            client.react(channel_id, react_ts, verdict.emoji),
                            timeout=config.tool_call_timeout)
                except asyncio.TimeoutError:
                    main_logger.debug("Participation react timed out")
                except Exception as e:
                    main_logger.debug(f"Participation react failed: {e}")
                return None
            if verdict.action == "backoff":
                # The taxonomy decides what "backoff" means: a durable per-channel preference,
                # a real thread mute/unmute, a momentary aside (nothing persisted), or an
                # explicit channel-settings change — the last one falls through to the response
                # loop so the MAIN model applies it (with judgment) via set_channel_participation.
                fall_through = await self._apply_backoff(message, client, verdict)
                if fall_through:
                    return verdict
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
            return None  # ignore
        except Exception as e:
            # Fail-safe stays silence: worst failure mode is a missed reply, never spam.
            main_logger.warning(f"Participation gate error: {e}; staying silent")
            return None

    # ---- participation-feedback backoff taxonomy (redesign Layer 2) ----
    # Default guidance text per dimension, used to write a readable preference memory when the
    # classifier gives no `guidance` of its own.
    _PREF_DEFAULT_GUIDANCE = {
        "reactions": "react less often in this channel",
        "replies": "reply more sparingly in this channel",
        "verbosity": "keep replies short in this channel",
        "thread_participation": "participate more sparingly in this channel",
    }

    async def _apply_backoff(self, message: Message, client: BaseClient, verdict) -> bool:
        """Route a participation-feedback ('backoff') verdict through the redesign taxonomy.

        Returns True when the message should fall through to the MAIN response loop — an
        explicit channel-settings change the model applies with judgment via the gated
        set_channel_participation tool. Returns False when the feedback was fully handled
        here: a durable per-channel preference (memory), or a momentary/thread-scoped aside
        that persists nothing.

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
            # BLOCKER #3: this is the classifier's SEMANTIC judgment that the current human
            # message is an explicit structural request (it distinguishes addressed-to from
            # talked-about, which the raw name regex cannot). Stamp the turn so the gated
            # set_channel_participation tool is authorized in the response loop even without a
            # literal <@bot> mention ("only reply when I tag you" carries no mention). The flag
            # is one half of the authorization; the other half (human sender) is enforced in
            # handlers.text where the tool context is built.
            if isinstance(message.metadata, dict):
                message.metadata["gate_authorized_structural"] = True
            return True

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
        if verdict.emoji and verdict.dimension != "reactions":
            await self._backoff_ack(client, channel_id, react_ts, verdict.emoji)

        return False

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
        elif len(guidance) > 200:
            guidance = guidance[:200] + "…"
        return f"Channel participation preference ({dimension}): {guidance}"

    async def _backoff_ack(self, client: BaseClient, channel_id: str,
                           react_ts: str, emoji: str) -> None:
        """Drop the optional acknowledgment reaction, bounded and routed through the same
        reservation guard the gate's own react uses (main.py gate react) so a later turn sees
        the slot consumed and never double-adds."""
        try:
            if hasattr(client, "_reserve_and_react"):
                await asyncio.wait_for(
                    client._reserve_and_react(channel_id, react_ts, emoji),
                    timeout=config.tool_call_timeout)
            elif hasattr(client, "react"):
                await asyncio.wait_for(
                    client.react(channel_id, react_ts, emoji),
                    timeout=config.tool_call_timeout)
        except asyncio.TimeoutError:
            main_logger.debug("Backoff ack react timed out")
        except Exception as e:
            main_logger.debug(f"Backoff ack react failed: {e}")

    @staticmethod
    def _produced_visible_output(response, turn) -> bool:
        """F38: did this turn actually do the thing the 👀 claimed?

        The claim is honored by anything the user can SEE: text that went out, a deliberate
        response reaction, or a tool that owns its own surface (a background job's status
        card, a detached image). It is NOT honored by silence, by an error notice, or by a
        turn that got queued behind another — in all three the bot claimed work and then
        produced none of it, so the eye comes back off."""
        if turn is not None and turn.visible_action_committed:
            return True   # a detached producer (image gen / background job) owns a surface
        if response is None or response.type in ("error", "queued"):
            return False
        meta = response.metadata or {}
        if meta.get("interrupted"):
            # The turn died partway through and all that reached the thread was an apology
            # for dying. It claimed work and delivered none of it — `posted` is True only
            # because a Slack surface exists to carry the notice.
            return False
        if meta.get("terminal_action") == "no_reply":
            # The one sibling a no-reply turn may have: a reaction that IS the answer.
            return bool(meta.get("response_reaction_committed"))
        if meta.get("reaction_only") or meta.get("background_job_started"):
            return True
        posted = meta.get("posted")
        if posted is None:  # non-streaming handlers can't know; derive from the outcome
            posted = bool(response.type == "text"
                          and (meta.get("streamed") or (response.content or "").strip()))
        return bool(posted)

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
                    thread_manager=self.processor.thread_manager, unprompted=False,
                    message_ts=(message.metadata or {}).get("ts"),
                )
                posted += 1
            except Exception as e:
                main_logger.error(f"Sandbox image rescue failed: {e}", exc_info=True)
        return posted

    async def handle_message(self, message: Message, client: BaseClient):
        """Handle incoming message from any platform"""
        # Phase F participation gate: for UNPROMPTED channel messages (judicious/active
        # levels) the engine decides respond/react/ignore/backoff BEFORE anything is
        # posted. Only action='respond' falls through.
        placement_verdict = None
        if message.metadata.get("participation_check") is True:
            verdict = await self._run_participation_gate(message, client)
            if verdict is None:
                return
            placement_verdict = verdict.placement
            # F3: the engine's reason rides the wake envelope for ambient wakes.
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
                        # Format and send text (top-level when placement chose channel)
                        formatted_text = client.format_text(response.content)
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
                            formatted_text,
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

            # Phase E/F participation stats (F2: accounted AFTER delivery, honest posted).
            # Count a reply only when visible content actually went out on an unprompted
            # (wake-gate) channel turn. Image/background turns account in publish_image.
            if (response and message.channel_id and not message.channel_id.startswith("D")
                    and getattr(client, "channel_pulse", None) is not None):
                terminal = (response.metadata or {}).get("terminal_action")
                if terminal == "no_reply":
                    main_logger.info(
                        f"no_response_needed — no reply posted "
                        f"(reason: {response.metadata.get('reason')!r})")
                else:
                    posted = response.metadata.get("posted")
                    if posted is None:
                        # Non-streaming handlers can't know; derive from the outcome.
                        posted = bool(
                            response.type == "text"
                            and (response.metadata.get("streamed")
                                 or (response.content or "").strip()))
                    if posted:
                        try:
                            client.channel_pulse.record_bot_reply(
                                message.channel_id, message.metadata.get("ts"),
                                unprompted=self._is_unprompted_turn(message),
                            )
                        except Exception as e:
                            main_logger.debug(f"participation stat record failed: {e}")
                    elif (response.type == "text"
                          and not (response.content or "").strip()
                          and not response.metadata.get("reaction_only")):
                        # Bare empty text without the terminal tool: contract violation.
                        # Fail-safe silence, no quota burn, no re-prompt this phase.
                        main_logger.warning(
                            "Empty text response without a terminal action — posting nothing")

        except Exception as e:
            main_logger.error(f"Error handling message: {e}", exc_info=True)

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
                                swept_keys = self.processor.db.delete_expired_ambient_artifacts(
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
                                self.processor.db.backup_database()
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
            main_logger.info("Received keyboard interrupt")
        except Exception as e:
            main_logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
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