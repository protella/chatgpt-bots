"""WHY a message is being handled, stated once at dispatch and never re-derived.

There used to be a single overloaded boolean on the metadata, and it answered four different
questions at once: does this message have to pass the participation model, did that model wake
the responder, what kind of thing is this message socially, and may the turn end without words.
Every consumer re-derived the parts it needed from that one flag plus whatever else was lying
around on the metadata — `wake_source` in one place, the channel-id prefix in another — so a
change to any of them silently changed the other three.

So each producer now stamps the four facts explicitly, on EVERY dispatched Message, and every
consumer reads them. A fact that is False is still PRESENT: "this DM does not need the gate" and
"nobody said" are different states, and only the first one is true of a DM.

THE FOUR FACTS

``gate_required`` (bool)
    This message must pass the participation model before it may reach the responder. True for
    ambient/name/thread traffic the engine judges (including the edit path's re-dispatch); False
    for a DM, a real @mention, a thread continuation, and the engine-off legacy name wake — those
    go straight to the responder. A thread continuation is either of two rules: a strict 1:1
    thread (level-independent), or membership in an `on` channel — participation in a thread is
    itself the wake signal, a thread we have posted in is one we are already part of, and the
    responder, which can see the thread, decides what the turn owes, including nothing.

``gate_woke`` (bool)
    A required gate handed THIS attempt to the full responder — the gate said wake, and nothing
    else, because that is the whole of what it says. Initialized False at stamp time and written
    centrally when the
    gate returns a verdict, so it is never a guess about what the gate did. It is meaningless
    without `gate_required`: ``gate_required=False`` with ``gate_woke=True`` is an illegal state
    and `set_gate_woke` refuses to create it.

``routing_posture`` (str)
    Why the message is conversationally in scope — derived ONLY from explicit Slack addressing
    and message topology. NOT from the participation level, not from placement, and not from a
    name match: a message that merely says our name is discussing us, not addressing us, and it
    keeps the ambient posture it would have had anyway. (`wake_source` and
    `participation_name_hit` still carry that narrower provenance; they answer a different
    question.) It is NOT an authorization signal.

``silence_capable`` (bool)
    The responder may end this turn without words — the same predicate that decides whether
    `no_response_needed` is on the table. True on the gate-routed routes and on the thread
    continuation (which skips the gate, so the model is the only decider); False on a DM, a real
    @mention and the engine-off legacy name wake, where somebody asked us directly and an answer
    is owed. Route truth, not config truth: the `enable_no_reply_tool` switch is applied by the
    consumers that expose the tool, exactly where it lived before.
"""
from typing import Any, Optional

from logger import setup_logger

logger = setup_logger(name="slack_bot.RoutingFacts")

# --- metadata keys (public: these ARE the contract other modules read) ---
GATE_REQUIRED = "gate_required"
GATE_WOKE = "gate_woke"
ROUTING_POSTURE = "routing_posture"
SILENCE_CAPABLE = "silence_capable"

# --- routing_posture values ---
# The bot was addressed: a DM (every message in one is for us) or a genuine @mention, including
# the edit path's synthetic addressed wake, which routes through the same handler.
POSTURE_ADDRESSED = "addressed_to_assistant"
# A reply inside a thread that did not address us explicitly — either ungated thread
# continuation (strict 1:1, or membership in an `on` channel) as well as any gate-routed
# thread message.
POSTURE_THREAD = "thread_activity"
# Top-level channel traffic that did not address us explicitly.
POSTURE_CHANNEL = "channel_activity"

POSTURES = frozenset({POSTURE_ADDRESSED, POSTURE_THREAD, POSTURE_CHANNEL})


def derive_posture(*, addressed: bool, ts: Optional[str],
                   thread_ts: Optional[str]) -> str:
    """The posture for one dispatch. `addressed` is explicit Slack addressing (DM or a real
    @mention) and wins outright; otherwise topology decides — a reply under a thread root is
    thread activity, anything else is channel activity."""
    if addressed:
        return POSTURE_ADDRESSED
    if thread_ts and ts and str(thread_ts) != str(ts):
        return POSTURE_THREAD
    return POSTURE_CHANNEL


def stamp_routing_facts(message: Any, *, gate_required: bool, silence_capable: bool,
                        addressed: bool, ts: Optional[str] = None,
                        thread_ts: Optional[str] = None) -> Optional[dict]:
    """Write all four facts onto a Message about to be dispatched. Returns the stamped facts.

    Every dispatch site calls this, including the ones where three of the four are False — a
    consumer must never have to tell "not required" apart from "never stamped"."""
    facts = {
        GATE_REQUIRED: bool(gate_required),
        GATE_WOKE: False,
        ROUTING_POSTURE: derive_posture(addressed=addressed, ts=ts, thread_ts=thread_ts),
        SILENCE_CAPABLE: bool(silence_capable),
    }
    meta = getattr(message, "metadata", None)
    if not isinstance(meta, dict):
        logger.debug("Routing facts not stamped: message carries no metadata dict")
        return None
    meta.update(facts)
    return facts


def owes_answer(message: Any) -> bool:
    """True when this message has ALREADY earned a turn and is still waiting for one.

    Two ways to earn it: nobody gated it (a DM, a real @mention, the engine-off name wake — we
    were addressed and an answer is owed), or a gate ran and said wake. Either way the decision
    is made and the only thing outstanding is the turn itself."""
    meta = getattr(message, "metadata", None)
    if not isinstance(meta, dict):
        return False
    return meta.get(GATE_REQUIRED) is not True or meta.get(GATE_WOKE) is True


def owes_words(message: Any) -> bool:
    """True when this message was ADDRESSED — a DM, a real @mention, the engine-off name wake.

    Stronger than `owes_answer`, and the difference is the whole of `silence_capable`: somebody
    spoke to us directly, so a turn that ends in silence is a turn that ignored them. An ambient
    message a gate woke on is owed a TURN but not necessarily words — the responder can see the
    conversation and may legitimately find nothing worth adding.

    BOTH facts, not just `gate_required`. "Ungated" is not a synonym for "addressed": a thread
    continuation — strict 1:1, or membership in an `on` channel — skips the gate too, and is
    stamped silence_capable ON PURPOSE. No gate ran, so the responder is the only decider, and it
    is allowed to decide there is nothing to say; on the membership route that is the entire
    point, since the reply may not have been meant for us at all. Reading the absence of a gate as
    an obligation to speak would absorb one of
    those into a batch and take that decision away, which is the same class of mistake in the
    opposite direction: manufacturing words instead of losing them.

    The route that owes words is the one stamped `gate_required=False` AND `silence_capable=False`
    — which is exactly how `stamp_routing_facts` records "somebody asked us directly"."""
    meta = getattr(message, "metadata", None)
    if not isinstance(meta, dict):
        return False
    return meta.get(GATE_REQUIRED) is False and meta.get(SILENCE_CAPABLE) is False


def absorb_owed_answer(trigger: Any, absorbed: Any) -> bool:
    """A batch that already contains an answer we owe is NOT re-judged. Returns whether the
    trigger's gate requirement was cleared.

    The Phase-Q drain folds several queued messages into one catch-up turn whose TRIGGER is the
    newest of them. If that trigger happens to be gate-routed, the whole batch inherits its
    verdict — so a no-wake there silently discards messages that had already been addressed to us,
    or that a gate had already decided to answer, purely because something ambient landed after
    them. The messages are still in Slack and still in the thread state; what is lost is the reply
    somebody was waiting for, with no trace of the loss anywhere.

    So an owed answer survives absorption: the successor turn runs, and the responder — which can
    see all of it — decides what to say. The gate is not consulted a second time about a question
    that was already settled."""
    meta = getattr(trigger, "metadata", None)
    if not isinstance(meta, dict) or meta.get(GATE_REQUIRED) is not True:
        return False
    if not any(owes_answer(m) for m in (absorbed or [])):
        return False
    meta[GATE_REQUIRED] = False
    meta[GATE_WOKE] = False
    # And the OBLIGATION travels with the answer, not just the permission to run. An absorbed
    # @mention owes WORDS: the trigger was ambient, so it arrived here silence-capable, and
    # leaving it that way lets the responder end this turn with `no_response_needed` on a batch
    # containing a message somebody addressed to us directly. That is the original bug wearing a
    # different hat — the reply is still lost, just one layer further down.
    #
    # An absorbed message that a GATE woke on is different and keeps silence_capable True: it was
    # never addressed to us, and the responder — which can see the whole conversation, where the
    # gate saw one moment — is entitled to conclude there is nothing worth adding.
    if any(owes_words(m) for m in absorbed):
        meta[SILENCE_CAPABLE] = False
        logger.info(
            "Queued batch carries a message addressed to us — the catch-up turn owes words")
    else:
        logger.info(
            "Queued batch carries an answer already owed — the catch-up turn runs without "
            "re-gating")
    return True


def set_gate_woke(message: Any, woke: bool) -> None:
    """Record whether the gate handed this attempt on. Written once per gate run (so a queued
    redispatch's second gate cannot inherit the first one's answer), and only where a gate was
    required — the illegal state is created nowhere rather than checked everywhere."""
    meta = getattr(message, "metadata", None)
    if not isinstance(meta, dict):
        return
    meta[GATE_WOKE] = bool(woke) and meta.get(GATE_REQUIRED) is True
