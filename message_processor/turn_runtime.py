"""F38 — what a turn is allowed to SHOW, and what it has CLAIMED.

Two questions used to be answered by the same overloaded value, `thinking_id`:

    thinking_id is not None  -> we have a placeholder message to edit
    thinking_id is None      -> ...one of three completely different things

`None` meant "setStatus worked, the composer status is the indicator" (DMs and channel
threads on the agent surface), and it ALSO meant "the indicator failed outright", and under
the deferral below it would have meant "we deliberately showed nothing". Downstream code
read `None` as the first of those and cheerfully pushed phase updates to setStatus — which
renders a thinking status AND auto-opens the thread. Deferring the placeholder without
disentangling this would have moved the flash, not removed it.

So the turn carries its own state:

* ``progress_enabled`` — may this turn show speculative "working on it" chrome at all?
  False on a turn that may decide to say nothing. Nothing renders until the turn commits.
* ``silence_capable`` — the same predicate that decides whether ``no_response_needed`` is
  exposed to the model. One value drives both, so the tool and the UI policy can never drift
  apart: if the model can stay quiet, we don't pre-announce that it won't.
* ``reply_destination`` + ``destination_source`` + ``destination_selected`` +
  ``destination_locked`` — WHERE the reply goes, WHO decided, whether the decision has been
  made yet, and whether it can still change. Destination used to be a boolean
  (``place_in_channel``) recomputed in four places from a channel setting, a gate verdict and
  a "did the turn do real work" heuristic — so the streaming coordinator, the footer, the
  wake envelope and the ledger could each believe something different about one reply. Now
  one turn states it once, and the MODEL makes the call on the only route where there is a
  call to make (`set_reply_destination`).
* ``ack_lease`` — the receipt for a 👀 this turn placed, and the only thing that lets it be
  taken back.

The 👀 rule (the user's, verbatim): "Other human teammates don't drop eyes and then do
nothing, that's misleading. If it adds it, it needs to do something, or go back and remove
it." So 👀 is not "seen" and not "thinking about it" — it is a CLAIM ON WORK. It goes on when
real work starts and comes off if that work evaporates.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from logger import setup_logger
from message_processor import participation_telemetry

logger = setup_logger(name="slack_bot.TurnRuntime")

# Where a reply can go. `dm` is its own value rather than a flavour of `thread`: a DM has no
# other option, and collapsing it into `thread` loses the distinction between "there was no
# choice" and "the choice was made and came out this way".
DESTINATION_DM = "dm"
DESTINATION_THREAD = "thread"
DESTINATION_CHANNEL = "channel"
DESTINATIONS = frozenset({DESTINATION_DM, DESTINATION_THREAD, DESTINATION_CHANNEL})
# The values the MODEL may choose between. A DM is not on the menu — nothing to decide.
SELECTABLE_DESTINATIONS = (DESTINATION_THREAD, DESTINATION_CHANNEL)

SOURCE_STRUCTURAL = "structural"
SOURCE_MODEL = "model"
SOURCE_DEFAULT = "default"


@dataclass
class TurnRuntime:
    """Per-turn presentation + work-claim state. Created in main.py, threaded to the handlers."""

    silence_capable: bool = False
    progress_enabled: bool = True
    reply_thread_id: Optional[str] = None
    # --- where this turn's reply goes, stated rather than inferred ---
    # WHERE: "dm" | "thread" | "channel". Always populated, from the first moment of the turn.
    reply_destination: str = DESTINATION_THREAD
    # WHO decided: "structural" (the route left no choice), "model" (it called the tool), or
    # "default" (it was offered the choice and did not take it).
    destination_source: str = SOURCE_STRUCTURAL
    # False ONLY while an eligible top-level turn is still waiting for the model to choose. It
    # is the flag that keeps a surface from being minted in a place the answer may not go.
    destination_selected: bool = True
    # True once a reply surface exists. After that the destination is a fact about Slack, not a
    # preference — a late change would leave a message stranded in the other place.
    destination_locked: bool = False
    # The model was offered the choice, produced words, and never called the tool. Recorded (not
    # corrected): the answer is delivered in the default thread, and the miss is a prompt problem
    # worth counting rather than a delivery problem worth guessing about.
    destination_contract_miss: bool = False
    # Did THIS turn put one of our emoji on a message? Set by the react tool at the moment the
    # reaction lands, because that is the only place every reacting path passes through. It used
    # to be derived per-Response in the handlers, and only the no-reply branch actually built
    # the field — so a reaction-only turn and a reply that also reacted both told the ledger
    # "no emoji" while the reaction event sat two lines above it in the same file.
    reaction_committed: bool = False
    ack_lease: Optional[dict] = field(default=None, repr=False)
    ack_target_ts: Optional[str] = None
    # Where that claim was staked. Kept beside the ts purely so settle_ack can report a RETRACTED
    # 👀 against the right conversation — by then the Message that carried it is out of scope.
    ack_channel_id: Optional[str] = None
    # ...and which gate attempt it belonged to, read off the message in claim_work for the same
    # reason. None when the turn was not a gate attempt at all, which is also the signal that
    # neither the claim nor its retraction belongs in the participation ledger.
    ack_attempt_id: Optional[str] = None
    visible_action_committed: bool = False
    _claiming: bool = field(default=False, repr=False)

    @property
    def final_post_only(self) -> bool:
        """F39 — the "(edited)" rule. Slack can only STREAM into a thread: chat.startStream
        REQUIRES thread_ts. So a reply headed for the top level of a channel has no native path,
        and the legacy fallback fakes streaming by posting a stub and chat.update-ing it — which
        stamps the message "(edited)" forever. A human teammate doesn't post a stub and revise it
        in public; they post once, finished.

        So a channel-destined turn writes NOTHING until the answer is whole. DMs and threads are
        unaffected: they stream exactly as before. This is now derived rather than stored,
        because it was only ever a restatement of where the reply is going, and two fields for
        one fact is how they came to disagree."""
        return self.reply_destination == DESTINATION_CHANNEL

    @classmethod
    def for_message(cls, message: Any, *, channel_post_allowed: bool = False) -> "TurnRuntime":
        """Open a turn with its destination already stated.

        Three of the four routes have no decision to make, and say so with
        `destination_source="structural"`: a DM has nowhere else to go, a reply inside an
        existing thread belongs to that thread, and a channel that forbids top-level replies has
        settled the question in its settings. Only a top-level trigger in a channel that allows
        both is genuinely open — and that turn starts UNSELECTED, defaulting to the thread, and
        shows nothing anywhere until the model chooses.

        Silence-capable == the routing fact of the same name, and nothing else (routing_facts.py).
        The config switch stays here: the route says whether silence is allowed, the flag says
        whether the tool that performs it exists. Mirrors text.py::_materialize_request_tools."""
        meta = getattr(message, "metadata", None) or {}
        silence_capable = (meta.get("silence_capable") is True
                           and bool(getattr(config, "enable_no_reply_tool", True)))
        channel_id = str(getattr(message, "channel_id", "") or "")
        thread_id = getattr(message, "thread_id", None)

        if channel_id.startswith("D"):
            destination, selected = DESTINATION_DM, True
        elif meta.get("ts") != thread_id or not channel_post_allowed:
            # An existing thread, or a channel whose settings forbid a top-level reply.
            destination, selected = DESTINATION_THREAD, True
        else:
            # Both destinations legal: the model decides, and until it does the answer is
            # provisionally headed for the thread.
            destination, selected = DESTINATION_THREAD, False

        return cls(
            silence_capable=silence_capable,
            # A turn that may say nothing shows no chrome; neither does one that cannot show
            # chrome without editing it into the answer afterwards; neither does one that does
            # not yet know WHERE its chrome would go.
            progress_enabled=not (silence_capable
                                  or destination == DESTINATION_CHANNEL
                                  or not selected),
            reply_thread_id=None if destination == DESTINATION_CHANNEL else thread_id,
            reply_destination=destination,
            destination_source=SOURCE_STRUCTURAL if selected else SOURCE_DEFAULT,
            destination_selected=selected,
        )

    def select_destination(self, destination: Any, message: Any = None) -> dict:
        """The model's choice, from `set_reply_destination`. Returns the tool result.

        Refuses rather than guesses, in every direction: an unrecognized value, a second call
        that contradicts the first, and any call once a surface exists. The last one matters
        most — after a message is up, moving the destination would leave that message stranded
        in a place the rest of the answer is not going. An identical repeat is fine and changes
        nothing (models re-state decisions; that is not a conflict)."""
        if destination not in SELECTABLE_DESTINATIONS:
            return {"ok": False, "error": "invalid_destination",
                    "message": ("`destination` must be exactly one of "
                                f"{', '.join(SELECTABLE_DESTINATIONS)}.")}
        if self.destination_locked:
            return {"ok": False, "error": "destination_locked",
                    "message": ("The reply has already started going out — its destination "
                                "cannot change now.")}
        if self.destination_source == SOURCE_STRUCTURAL:
            # A DM, a thread, a channel that forbids top-level replies, or a turn that already
            # posted a notice. The route decided, so there is nothing to choose — and the tool
            # is not offered on these turns at all. This is the backstop for the gap between
            # those two facts: the registry checks `enabled` when it builds the SCHEMA set, not
            # again at dispatch, so a call that arrives anyway must be refused by the state
            # rather than silently overwrite it.
            return {"ok": False, "error": "destination_not_open",
                    "message": ("This reply's destination is already settled by where the "
                                "conversation is — there is nothing to choose here.")}
        if self.destination_selected and self.destination_source == SOURCE_MODEL:
            if destination == self.reply_destination:
                return {"ok": True, "destination": destination, "idempotent": True}
            return {"ok": False, "error": "destination_conflict",
                    "message": (f"You already chose `{self.reply_destination}` for this reply. "
                                "One destination per turn.")}
        self.reply_destination = destination
        self.destination_source = SOURCE_MODEL
        self.destination_selected = True
        self.reply_thread_id = (None if destination == DESTINATION_CHANNEL
                                else getattr(message, "thread_id", self.reply_thread_id))
        return {"ok": True, "destination": destination}

    def settle_structural_thread(self) -> None:
        """A visible surface already exists in the thread — a prior-timeout notice, a
        failed-files notice — so the question is settled by fact rather than by preference.

        Both notices post BEFORE the model runs. Leaving the turn open after one would let the
        model send the answer to the channel top level, splitting a single turn across two
        surfaces: a warning in the thread and the answer somewhere else. So the destination
        becomes the thread, STRUCTURALLY (the route left no choice — this is not a model that
        declined to choose, so it is not a contract miss), and it locks. The tool is not exposed
        on a settled turn, so the model is never offered a choice that no longer exists."""
        if self.destination_locked:
            return
        self.reply_destination = DESTINATION_THREAD
        self.destination_source = SOURCE_STRUCTURAL
        self.destination_selected = True
        self.destination_locked = True

    def settle_default_destination(self) -> None:
        """Words are arriving and the model never chose. The answer is NOT dropped and NOT
        guessed at from its text: it goes to the default (the thread), and the miss is recorded
        so a prompt that keeps producing it is visible in the ledger rather than only in the
        room."""
        if self.destination_selected:
            return
        self.destination_selected = True
        self.destination_source = SOURCE_DEFAULT
        self.destination_contract_miss = True

    def lock_destination(self) -> None:
        """A reply surface now exists. Idempotent; called from every path that mints one."""
        self.destination_locked = True

    def resolve_reply_target(self, message: Any) -> Optional[str]:
        """The thread_ts a reply should be posted with — None means top-level in the channel.

        A pure mapping from the stated destination, nothing more. It used to also carry a
        substantive-work override that re-threaded a channel reply after the fact, which is a
        heuristic second-guessing a decision the model is now asked to make outright."""
        if self.reply_destination == DESTINATION_CHANNEL:
            return None
        return getattr(message, "thread_id", None) if message is not None else self.reply_thread_id

    async def claim_work(self, client: Any, message: Any) -> None:
        """Real work is starting: stake the 👀 claim on the triggering message.

        Idempotent — many tools may call this in one turn, and exactly one reaction lands.
        Call it AFTER a tool's arguments and capacity checks pass and immediately BEFORE the
        slow part begins, never from a generic 'a tool was mentioned' hook: a rejected call
        (an invalid argument, a duplicate background job) must not flash an eye it is about
        to retract.

        Purely additive and fails silent — an emoji is never worth failing a turn over."""
        if self.ack_lease is not None or self._claiming:
            return
        if not getattr(config, "enable_ack_reaction", True):
            return
        meta = getattr(message, "metadata", None) or {}
        react_ts = meta.get("ts") or getattr(message, "thread_id", None)
        channel_id = getattr(message, "channel_id", None)
        if not react_ts or not channel_id:
            return
        if not hasattr(client, "_reserve_and_react_owned"):
            return
        # Read the attempt id HERE, while the Message is still in scope — settle_ack runs at the
        # end of the turn with nothing but this object, the same reason ack_channel_id is stashed.
        attempt_id = participation_telemetry.attempt_id_for(message)
        self.ack_attempt_id = attempt_id
        self._claiming = True  # before the await: concurrent tool calls must not double-add
        try:
            # BOUNDED. This runs inside the tool callback, so for a hosted tool the Responses
            # event loop is waiting on us — a wedged Slack call would stall the web search or
            # the code run it is announcing. The emoji must never hold up the work.
            _result, lease = await asyncio.wait_for(
                client._reserve_and_react_owned(
                    channel_id, react_ts, config.ack_reaction_emoji),
                timeout=config.tool_call_timeout)
            self.ack_lease = lease
            self.ack_target_ts = react_ts
            self.ack_channel_id = channel_id
            # The work-claim 👀 never goes through execute_react_tool, so without this it is the
            # one reaction the room can see that leaves no record. It also has to be TELLABLE
            # from a chosen emoji: it is a fixed operational marker, and counting it as taste
            # would flatten the diversity number every analysis is trying to read.
            #
            # A LEASE is the only proof we placed it. Without one the emoji was already up
            # there — a previous turn's, or the model's own react tool — and `ok` says only
            # that it is present, not that we put it there. Recording that as `added` would
            # inflate the claim rate with reactions the bot never made.
            if attempt_id:
                if lease is not None:
                    outcome, detail = "added", None
                elif isinstance(_result, dict) and _result.get("ok") is True:
                    outcome, detail = "already_present", None
                else:
                    outcome = "failed"
                    detail = _result.get("error") if isinstance(_result, dict) else None
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result=outcome,
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail=detail)
        except asyncio.TimeoutError:
            logger.debug("Work-claim reaction timed out")
            if attempt_id:
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result="failed",
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail="timeout")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Work-claim reaction failed: {e}")
            if attempt_id:
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result="failed",
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail=type(e).__name__)
        finally:
            self._claiming = False

    async def settle_ack(self, client: Any, produced_output: bool) -> None:
        """End of turn. Did we actually do the thing we claimed?

        `produced_output` False — the model chose silence, the turn errored, it got queued,
        or it started work and then backed off (the other bot answered first) — so the claim
        was not honored and the 👀 comes back off. True: it stays."""
        lease = self.ack_lease
        if lease is None:
            return
        self.ack_lease = None
        try:
            if produced_output:
                if hasattr(client, "settle_reaction_lease"):
                    client.settle_reaction_lease(lease)
            elif hasattr(client, "remove_owned_reaction"):
                removed = await client.remove_owned_reaction(lease)
                # A retracted claim was still visible in the room for the length of the turn.
                # Recorded so "we promised work and delivered nothing" is countable — it is the
                # single most annoying failure this system can produce, and it is invisible
                # afterwards because the evidence deletes itself.
                #
                # The REAL return value, not an assumption: remove_owned_reaction refuses a
                # stale lease and returns False, leaving the 👀 up. A row saying we took it
                # back when it is still sitting there would describe the opposite of the room.
                if self.ack_attempt_id:
                    participation_telemetry.reaction(
                        self.ack_channel_id, self.ack_target_ts, operation="remove",
                        result="removed" if removed else "remove_failed",
                        origin="work_claim", emoji=config.ack_reaction_emoji,
                        target_ts=self.ack_target_ts, attempt_id=self.ack_attempt_id,
                        detail="retracted")
        except Exception as e:  # noqa: BLE001
            # The 👀 is still up there and we no longer hold the lease, so the claim is stranded.
            # Recorded for the same reason the honest False above is: a lifecycle that ends with
            # an add and no removal outcome reads as a claim that was HONORED, which is the exact
            # opposite of a turn that promised work, produced none, and then failed to clean up.
            # Only on the RETRACTION path: honoring a claim removes nothing, so a failure there
            # is not a failed removal and must not be written as one.
            if self.ack_attempt_id and not produced_output:
                participation_telemetry.reaction(
                    self.ack_channel_id, self.ack_target_ts, operation="remove",
                    result="remove_failed", origin="work_claim",
                    emoji=config.ack_reaction_emoji, target_ts=self.ack_target_ts,
                    attempt_id=self.ack_attempt_id, detail=type(e).__name__)
            logger.debug(f"Ack settle failed: {e}")
