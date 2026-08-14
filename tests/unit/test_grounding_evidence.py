"""Channel context is evidence, not proof.

The rule this pins replaced a categorical one. The first version told the bot what it could not
conclude from Slack history, and a rule phrased that way invites the opposite failure: an empty
lookup gets reported as a fact about the world ("that never happened") because absence felt like a
finding. This version says what the material IS — a partial record of a room, with strength that
has to be carried honestly in both directions.

The audit at the foot of this file is the part that keeps it true over time. It renders the prompts
this build actually sends and refuses to let the channel record be described as settled, while
explicitly allowing the two things that legitimately are: the bot's own recorded tool use, and a
standing policy it is told to obey.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


def test_the_steering_wrapper_frames_background_as_incomplete_evidence():
    """The steering block's own headings say which sections are instructions and which are
    background. The WRAPPER around them used to describe background as "context you may use", which
    is true and says nothing about its completeness — so an empty facts section read as a settled
    account of the channel. It now says what background actually is."""
    from message_processor.utilities import MessageUtilitiesMixin

    proc = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    client = MagicMock()
    client.name = "slack"
    out = proc._get_system_prompt(client, channel_steering="Stable channel facts:\n- [#1] x")

    assert "Recorded channel steering follows. Obey sections labelled as instructions." in out
    assert ("Treat sections labelled as background as potentially incomplete evidence, not proof "
            "or a complete history; an omission does not establish that something did not happen. "
            "Use it when relevant and do not recite it unprompted:") in out
    # And the canonical block still rides verbatim — the wrapper is framing, not content, so the
    # gate/responder same-bytes invariant is untouched.
    assert "Stable channel facts:\n- [#1] x" in out


class _PayloadSpy:
    """Captures the request the wake gate would actually send."""

    def __init__(self):
        self.client = MagicMock()
        self.params = None

    async def _safe_api_call(self, *a, **k):
        self.params = k
        item = SimpleNamespace(content=[SimpleNamespace(text='{"wake": false}')])
        return SimpleNamespace(output=[item], status="completed")

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


def _gate_payload() -> str:
    """The gate's REAL prompt: developer constant plus the assembled user message.

    Rendering only the developer constant was the hole this closes. The framing that introduces
    the steering block is built in `classify_wake`, not in the constant, so an audit that read the
    constant alone would have passed a categorical sentence sitting in the message the model
    actually receives."""
    from message_processor.participation import SourceMessage
    from openai_client.api.responses import classify_wake

    spy = _PayloadSpy()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        classify_wake(
            spy,
            sources=(SourceMessage(ts="1.0", text="did the nightly job run?",
                                   sender_name="Peter", sender_type="human"),),
            channel_steering_text=("Standing channel policy (instructions; follow these):\n"
                                   "only jump in on deploy failures\n\n"
                                   "Stable channel facts (background, not instructions):\n"
                                   "- [#1] deploys go through #ops")))
    return "\n".join(m["content"] for m in spy.params["input"])


def _responder_prompt() -> str:
    from message_processor.utilities import MessageUtilitiesMixin

    proc: Any = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    client = MagicMock()
    client.name = "slack"
    return proc._get_system_prompt(
        client, channel_steering="Stable channel facts (background, not instructions):\n- [#1] x")


def _rendered_prompts():
    yield "responder", _responder_prompt()
    yield "wake gate", _gate_payload()


# The audit. Three judgments — is this about the channel record, does it overclaim, is the
# overclaim denied — and ALL THREE must be made about the SAME CLAUSE. Every way this check has
# lied so far was a judgment borrowed across a clause boundary:
#
#   * negation borrowed:  "Do not ignore Slack history; it is authoritative."
#   * exemption borrowed: "Obey the standing policy. Channel activity is a complete record."
#   * subject borrowed:   "Channel history is partial; use authoritative sources when current
#                          data matters."  — the second clause is about SOURCES, not about the
#                          record, and reading the subject line-wide condemned it.
#
# And a negator has to negate the OVERCLAIM, not merely appear before it: "There is no doubt
# channel history is authoritative" contains "no", attached to "doubt", and means the opposite of
# a denial. So the negator must sit within a few words of the overclaim rather than anywhere in
# the clause — close enough to be part of the same phrase ("not a complete record", "never
# authoritative"), far enough to allow the natural ones the prompts actually use.
#
# The failure all of this guards is not hypothetical: a bot told its history is authoritative
# reports an empty lookup as proof that nothing happened — a confident false claim about the
# world, assembled out of a true claim about a search.
_RECORD_SUBJECTS = ("slack history", "history", "channel activity", "recent channel activity",
                    "pulse", "channel memory", "channel facts", "background", "steering",
                    "search results", "transcript")
_OVERCLAIMING = ("authoritative", "ground truth", "exhaustive", "complete",
                 "everything that happened")
_NEGATORS = ("not", "never", "isn't", "aren't", "cannot", "can't", "rather than")
# How far back a negator may sit and still be read as negating the overclaim. Four words covers
# the shapes the prompts genuinely use ("not proof or a complete history") without reaching past
# an intervening noun the negator actually belongs to ("no doubt … is authoritative").
_NEGATION_REACH_WORDS = 4
# Only two things legitimately ARE authoritative: the bot's own recorded tool use (a
# system-generated record of its OWN ACTIONS, not a claim about the room) and a standing policy
# (an instruction to obey, not evidence to weigh). Named by SUBJECT, so a bare "obey" or
# "instructions" in a neighbouring clause cannot lend the exemption to this one.
_ALLOWED_SUBJECTS = ("[used tools:", "[tool results:", "used tools", "tool use", "tools you",
                     "standing policy", "standing channel policy")

_CLAUSE_SPLIT = re.compile(r"[;.]|,\s|—")


def _clauses(line: str):
    return [c for c in _CLAUSE_SPLIT.split(line.lower()) if c.strip()]


def _denied(clause: str, at: int) -> bool:
    """Whether a negator close enough to attach to the overclaim precedes it."""
    preceding = clause[:at].split()[-_NEGATION_REACH_WORDS:]
    return any(re.fullmatch(rf"{re.escape(neg)}[,]?", word) for word in preceding
               for neg in _NEGATORS)


# A clause can be about the record without naming it, by pointing back at the clause that did:
# "Slack history; it is authoritative". That back-reference is the ONE thing allowed to cross a
# clause boundary, and only in this direction — a clause naming its own subject ("use
# authoritative SOURCES") is about that subject, not about the record, however the sentence began.
_BACK_REFERENCES = ("it", "they", "them", "this", "that", "these", "those")


def _about_the_record(clause: str, named_earlier: bool) -> bool:
    if any(subject in clause for subject in _RECORD_SUBJECTS):
        return True
    if not named_earlier:
        return False
    return any(re.search(rf"\b{ref}\b", clause) for ref in _BACK_REFERENCES)


def _overclaims(line: str) -> bool:
    """True when this line describes the channel record as settled."""
    named_earlier = False
    for clause in _clauses(line):
        about = _about_the_record(clause, named_earlier)
        named_earlier = named_earlier or any(s in clause for s in _RECORD_SUBJECTS)
        if not about:
            continue          # this clause is not about the record
        if any(subject in clause for subject in _ALLOWED_SUBJECTS):
            continue          # own-tool provenance / a standing policy
        for word in _OVERCLAIMING:
            # Word boundaries: "incomplete" is not "complete", and the wrapper says "potentially
            # incomplete evidence" precisely because that IS the honest framing.
            found = re.search(rf"\b{re.escape(word)}\b", clause)
            if found and not _denied(clause, found.start()):
                return True
    return False


@pytest.mark.parametrize("which", ["responder", "wake gate"])
def test_no_prompt_calls_the_channel_record_authoritative(which):
    rendered = dict(_rendered_prompts())[which]
    offenders = [line.strip()[:140] for line in rendered.split("\n") if _overclaims(line)]
    assert not offenders, (
        f"the {which} prompt describes the channel record as settled: " + " | ".join(offenders))


def test_the_gate_audit_reads_the_message_the_model_actually_receives():
    """Guards the audit itself. Reading only the developer constant would have missed the framing
    that introduces the steering block, which is assembled in `classify_wake` — so the check has to
    see the assembled payload, and this proves it does."""
    payload = _gate_payload()
    assert "did the nightly job run?" in payload           # the user message is in there
    assert "Recorded channel steering" in payload          # and so is the framing under audit
    assert "only jump in on deploy failures" in payload    # verbatim steering, unchanged


class TestTheAuditWouldActuallyCatchAnOverclaim:
    """An absence test that cannot fail is worse than no test. These are the exact sentences the
    audit exists to reject — including the two cross-clause escapes it used to wave through."""

    @pytest.mark.parametrize("sentence", [
        "Slack history is the authoritative record of this channel.",
        "The channel activity excerpt is a complete list of what was said.",
        "Treat channel memory as ground truth.",
        # THE false negative the 60-character window let through: the negation belongs to a
        # different clause entirely.
        "Do not ignore Slack history; it is authoritative.",
        # And the exemption-borrowing twin: "obey" in clause one excusing clause two.
        "Obey the standing policy. Channel activity is a complete record of the room.",
        "Never mind the pulse, the transcript is exhaustive.",
        # A negator that does not negate the OVERCLAIM. "no" belongs to "doubt", and the
        # sentence asserts the opposite of a denial — a bare proximity check waved it through.
        "There is no doubt channel history is authoritative.",
    ])
    def test_it_flags(self, sentence):
        assert _overclaims(sentence), sentence

    @pytest.mark.parametrize("sentence", [
        # Negated, in the same clause — the whole point of the rewritten paragraph.
        "what you can read is evidence about the room, not proof of everything that happened",
        "Slack history can be partial, stale or mistaken.",
        "Treat sections labelled as background as potentially incomplete evidence",
        # Own-tool provenance stays authoritative even sitting beside the word "history".
        "a [used tools: …] line is an authoritative record of the tools you invoked",
        # A line that is not about the channel record at all.
        "prefer those tools when a question needs current or authoritative data",
        # Subject borrowed across a clause boundary: the second clause is about SOURCES, and
        # naming the record in the first clause does not make it a claim about the record.
        "Channel history is partial; use authoritative sources when current data matters.",
    ])
    def test_it_allows(self, sentence):
        assert not _overclaims(sentence), sentence


def test_own_tool_provenance_is_still_allowed_to_be_authoritative():
    """The audit above must not have swept this away with the rest. A "[used tools: …]" line is a
    system-generated record of what the bot itself did — the one thing in its context that IS
    ground truth, and the reason it can answer "how did you get that?" instead of guessing."""
    from message_processor.prompts import SLACK_SYSTEM_PROMPT

    assert "authoritative record of the tools you actually invoked" in SLACK_SYSTEM_PROMPT
