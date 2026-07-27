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
* ``reply_thread_id`` — where a reply actually goes (None = top-level in the channel). The
  streaming paths used to infer this from ``message.thread_id``, which is only ever right
  because the placeholder already existed in the correct place. Take the placeholder away
  and a top-level ambient reply lands in a thread instead.
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


@dataclass
class TurnRuntime:
    """Per-turn presentation + work-claim state. Created in main.py, threaded to the handlers."""

    silence_capable: bool = False
    progress_enabled: bool = True
    reply_thread_id: Optional[str] = None
    final_post_only: bool = False
    # F46: did this turn do real, thread-worthy work? Set by mark_substantive_work() at every
    # site that stakes a work claim (a hosted tool ran, an MCP call was made, or a slow local
    # deliverable tool ran). Drives resolve_reply_target's top-level→thread override. Tracked
    # SEPARATELY from the 👀/claim_work, which early-returns when enable_ack_reaction is off.
    did_substantive_work: bool = False
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

    @classmethod
    def for_message(cls, message: Any, reply_thread_id: Optional[str]) -> "TurnRuntime":
        """Silence-capable == the routing fact of the same name, and nothing else. Which routes
        may end without words is decided once at dispatch (routing_facts.py); this used to
        re-derive it from a gate flag plus a `wake_source` string, which is a second copy of
        that table living where nobody would think to update it. The config switch stays
        here — the route says whether silence is allowed, the flag says whether the tool that
        performs it exists. Mirrors text.py::_materialize_request_tools; keep them in step."""
        meta = getattr(message, "metadata", None) or {}
        silence_capable = (meta.get("silence_capable") is True
                           and bool(getattr(config, "enable_no_reply_tool", True)))

        # F39 — the "(edited)" rule. Slack can only STREAM into a thread: chat.startStream
        # REQUIRES thread_ts. So a reply we're about to post at the top level of a channel has
        # no native path, and the legacy fallback fakes streaming by posting a stub and
        # chat.update-ing it — which stamps the message "(edited)" forever. A human teammate
        # doesn't post a stub and revise it in public; they post once, finished. Neither does
        # Claude, which is why its top-level replies carry no edit marker and ours did.
        #
        # So these turns write NOTHING until the answer is whole: no placeholder, no composer
        # status (it would auto-open a thread anyway), no edit loop. Just the finished message.
        # DMs are excluded — they also can't stream natively, but a DM is a conversation, not a
        # public channel, and losing the live reveal there is a real cost with no edit-marker
        # complaint attached. Threads keep streaming exactly as before.
        channel_id = str(getattr(message, "channel_id", "") or "")
        final_post_only = (reply_thread_id is None and bool(channel_id)
                           and not channel_id.startswith("D"))
        return cls(
            silence_capable=silence_capable,
            # A turn that may say nothing shows no chrome; neither does one that can't show
            # chrome without editing it into the answer afterwards.
            progress_enabled=not (silence_capable or final_post_only),
            reply_thread_id=reply_thread_id,
            final_post_only=final_post_only,
        )

    def mark_substantive_work(self) -> None:
        """F46: record that this turn did real, thread-worthy work (a hosted tool ran, an MCP
        call was made, or a deliverable local tool ran). Drives the top-level→thread override at
        final-post time. Separate from claim_work()/the 👀, which is gated on enable_ack_reaction."""
        self.did_substantive_work = True

    def resolve_reply_target(self, message: Any) -> Optional[str]:
        """F46: the thread_ts a final reply should go to. A top-level channel reply (reply_thread_id
        is None, final_post_only) that did substantive work is threaded under the trigger; otherwise
        the original target stands. Mutates message.metadata['place_in_channel']=False when it flips,
        so attribution/footer render as a threaded reply. Idempotent; fail-open."""
        try:
            if (self.final_post_only and self.reply_thread_id is None
                    and self.did_substantive_work):
                meta = getattr(message, "metadata", None)
                if isinstance(meta, dict):
                    meta["place_in_channel"] = False
                return getattr(message, "thread_id", None)
        except Exception:
            pass
        return self.reply_thread_id

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
