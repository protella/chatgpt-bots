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
    visible_action_committed: bool = False
    _claiming: bool = field(default=False, repr=False)

    @classmethod
    def for_message(cls, message: Any, reply_thread_id: Optional[str]) -> "TurnRuntime":
        """Silence-capable == exactly the turns where `no_response_needed` is on the table:
        an unprompted channel message the wake gate let through, or a 1:1 thread continuation
        (which skips the gate entirely — there the model is the ONLY decider). Mirrors
        text.py::_materialize_request_tools; keep them in step."""
        meta = getattr(message, "metadata", None) or {}
        unprompted = meta.get("participation_check") is True
        continuation = (not unprompted
                        and meta.get("wake_source") == "thread_continuation")
        silence_capable = ((unprompted or continuation)
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
        except asyncio.TimeoutError:
            logger.debug("Work-claim reaction timed out")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Work-claim reaction failed: {e}")
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
                await client.remove_owned_reaction(lease)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Ack settle failed: {e}")
