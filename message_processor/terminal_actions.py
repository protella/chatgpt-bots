"""How a turn may end without words, and the eight reasons it may say so.

The terminal tool used to take a free-prose `reason`: one short sentence, model's choice of
words. That is unanalysable. "not really for me", "this one's for Dana", "I think Claude has
this" and "addressed to another assistant" are one fact written four ways, so the only question
worth asking of the silence ledger — WHY does this bot stay quiet, and is it the right why —
could be answered only by reading every line and judging it by hand.

So the reason is a closed vocabulary. Eight values, chosen to be mutually distinguishable by the
model at the moment of deciding, with `other` present so a genuine ninth case is recorded as
itself rather than crammed into the nearest wrong bucket.

THE VALUE IS NEVER INFERRED. Not from the message text, not from whether a reaction landed, not
from the routing posture, not from which tools ran. A missing or unrecognized reason is an
INVALID tool call that the model is told to fix — never quietly rewritten to `other`, which
would put our guess in the ledger under the model's name and make the whole column untrustworthy
at exactly the points where it is most interesting.

This module is the single definition. The tool schema, the tool loop, the handlers, the
telemetry and the tests all import from here, so the enum cannot drift between the prompt the
model is given and the column the analysis reads.
"""
from typing import Any, Literal

# The canonical vocabulary, in the order the schema presents it.
SILENCE_REASONS = (
    "addressed_to_other",
    "reacted_instead",
    "nothing_to_add",
    "not_relevant",
    "duplicate",
    "user_requested_silence",
    "awaiting_context",
    "other",
)

# Spelled out rather than derived from the tuple: a Literal needs its members at type-check
# time. `test_terminal_actions` asserts the two stay in lockstep.
SilenceReason = Literal[
    "addressed_to_other",
    "reacted_instead",
    "nothing_to_add",
    "not_relevant",
    "duplicate",
    "user_requested_silence",
    "awaiting_context",
    "other",
]

# What each value means, in the words the model reads in the schema. These are definitions, not
# hints: the difference between `nothing_to_add` and `not_relevant` decides whether a quiet
# channel reads as a bot with nothing to say or a bot in the wrong room.
SILENCE_REASON_DESCRIPTIONS: dict = {
    "addressed_to_other": (
        "the message or exchange belongs to another person or another assistant — including "
        "another AI assistant in this conversation, since two assistants answering the same "
        "message is worse than none"),
    "reacted_instead": (
        "an emoji reaction you successfully placed is the complete response"),
    "nothing_to_add": (
        "relevant to you, but words would add no useful information"),
    "not_relevant": (
        "you are not relevant to this exchange"),
    "duplicate": (
        "you would repeat something already supplied"),
    "user_requested_silence": (
        "a person explicitly asked you not to respond"),
    "awaiting_context": (
        "any useful action needs context that is still expected from a human or an external "
        "source. NEVER for work you dispatched yourself — that is yours to finish"),
    "other": (
        "a legitimate reason none of the above describes. Choose this deliberately, not as a "
        "fallback for a value you did not read"),
}


def render_reason_guide() -> str:
    """The per-value definitions as one schema description string."""
    return "Why no reply is right. " + "; ".join(
        f"{value}: {SILENCE_REASON_DESCRIPTIONS[value]}" for value in SILENCE_REASONS)


def is_valid_silence_reason(value: Any) -> bool:
    """Exactly one of the eight, as a string. Anything else — absent, misspelled, prose, a
    number, a list — is invalid, and invalid is a contract error the model gets to correct.
    Deliberately strict about type: `True` and `1` are not reasons, and Python would happily
    compare them against a tuple of strings without complaint."""
    return isinstance(value, str) and value in SILENCE_REASONS
