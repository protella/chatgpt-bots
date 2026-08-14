"""Where the reply goes, chosen by the model that is writing it.

The destination used to be inferred, and the inference had three separate authors. The
channel's `reply_in_channel` setting said top-level replies were allowed; the participation
gate attached a `placement` verdict formed before the answer existed; and a
`did_substantive_work` flag re-threaded a top-level reply after the fact if any tool had run.
None of the three could see the thing that actually decides it — the answer. "What's the deploy
status?" belongs in the channel where everyone reading the room gets it for free. A
three-paragraph write-up belongs in a thread, whether or not a tool happened to run while
producing it. Only the writer knows which one it just wrote.

So the model states it, once, on the only route where there is anything to state: a top-level
message in a channel that allows both. Everywhere else the route decides and the tool is not
offered at all — a DM has nowhere else to go, a reply in a thread belongs to that thread, and a
channel that forbids top-level replies has already answered the question in its settings.

The tool is FREE (the F37 bookkeeping allowance): it produces nothing, costs no round budget,
and must never compete with a real tool call for a slot.

W4 — THE FIRST-TOKEN MARKER. Stating the choice with a TOOL costs a whole extra round before a
single word can be written: the model calls `set_reply_destination`, we answer it, and only then
does the request that produces the answer go out. That pre-round was ~3s on every selectable
channel turn, spent on a decision the model had already made. So the choice now rides the answer
itself: the model BEGINS its reply with `[[reply:thread]]` or `[[reply:channel]]`, the parser
below consumes it out of the very first tokens, and the destination is settled inside the same
call that writes the answer.

The tool is retired from the CHANNEL surface (see `register_destination_tools`) and survives
everywhere else, unchanged. The marker needs no schema at all — it is text — so the contract
paragraph that teaches it does not key off tool exposure.

ONE PARSER, ONE RULE, ONE CHOKE POINT. `parse_destination_marker` is the only thing in the
codebase that knows the grammar, and every path routes through
`consume_destination_marker` / `DestinationMarkerReader`: the streaming callback, the canonical
`response_text` of both handlers, the timeout retry, the non-streaming fallback and the
reconsideration/final-correction deliveries. The rule is the same in all of them — the FIRST
complete marker ANYWHERE in the text selects, the moment it parses. "Leading" is what the prompt
demands of the model; "anywhere" is what the parser accepts, so a preamble the model wrote before
its marker can never be read as a missing one. MISSING is a verdict only terminal time may reach
(no marker in the canonical text), and it costs the turn nothing but a `destination_contract_miss`
— the answer still lands in the default thread.

And the marker NEVER survives the turn: stripping happens here, at the parser, so the Slack post,
the thread state, reconsideration and compaction all see text that never had one.
"""
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from logger import setup_logger
from message_processor.message_markers import join_segments
from message_processor.turn_runtime import SELECTABLE_DESTINATIONS
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.DestinationTools")

SET_REPLY_DESTINATION = "set_reply_destination"


# --------------------------------------------------------------------------- the marker grammar

def destination_marker(destination: str) -> str:
    """The literal a model writes to choose `destination`. The prompt quotes these; nothing else
    may spell them, so the grammar has exactly one author."""
    return f"[[reply:{destination}]]"


# The trailing `\s*` belongs to the marker, not to the answer: the model is told to begin with
# the marker and then write, so whatever separates the two — a space, a newline, a blank line —
# is punctuation for a thing that is about to disappear. Leaving it behind would open every such
# reply with stray whitespace.
_MARKER_RE = re.compile(
    r"\[\[reply:(" + "|".join(re.escape(d) for d in SELECTABLE_DESTINATIONS) + r")\]\]\s*")

# Every proper prefix of a legal marker, for the streaming reader: a chunk boundary can fall
# anywhere, and text ending in one of these may still be growing into a marker.
_MARKER_PREFIXES = frozenset(
    destination_marker(d)[:n]
    for d in SELECTABLE_DESTINATIONS
    for n in range(1, len(destination_marker(d))))
_MAX_MARKER_LEN = max(len(destination_marker(d)) for d in SELECTABLE_DESTINATIONS)


def parse_destination_marker(text: Optional[str]) -> Tuple[Optional[str], str]:
    """THE parser. Returns (destination or None, text with every marker removed).

    One rule, used by every caller: the FIRST complete marker anywhere in `text` names the
    destination; ALL of them are stripped. Text with no marker comes back byte-identical — that
    identity is what keeps non-selectable turns unchanged by W4.
    """
    if not text:
        return None, text or ""
    match = _MARKER_RE.search(text)
    if match is None:
        return None, text
    return match.group(1), _MARKER_RE.sub("", text)


def _consume_one(text: Optional[str], turn: Any, message: Any) -> str:
    destination, cleaned = parse_destination_marker(text)
    if destination is not None and turn is not None:
        turn.select_destination(destination, message=message)
    return cleaned


def consume_destination_marker(text: Optional[str], *, turn: Any = None, message: Any = None,
                               segments: Optional[Sequence[str]] = None) -> str:
    """Parse, select, strip — the single acceptance rule, wherever text is handled whole.

    Selection goes through `TurnRuntime.select_destination`, which stamps
    `destination_source="model"` itself and refuses on a turn that has nothing to choose (a DM, a
    thread, a locked reply). A refusal is not an error here: the strip is unconditional, because a
    marker must never reach a reader whether or not it was allowed to mean anything.

    `segments` — THE ROUND BOUNDARY. A multi-round turn's canonical text is its rounds with seams
    put BETWEEN them, and those seams belong to no round. Parsing the joined string lets a marker
    that ended a round eat the separator that was inserted after it, which the streaming reader
    (flushed at every round boundary, so it only ever sees one round at a time) does not do — and
    on a native turn the streamed buffer is what Slack keeps, so the two would disagree about the
    finished answer. Given the rounds, each is parsed ALONE and the results re-joined by the same
    `join_segments` the loop used, so the parser never meets a seam it did not come with. The
    order is unchanged, so "the first marker anywhere" still means the same marker.
    """
    if segments is not None:
        return join_segments([_consume_one(s, turn, message) for s in segments])
    return _consume_one(text, turn, message)


def _unresolved_tail(text: str) -> int:
    """How many trailing characters of `text` are not yet safe to release.

    Two ways a tail can still change meaning: it is a partial marker the next chunk completes, or
    it IS a marker whose trailing whitespace run has not finished arriving. Holding both is what
    makes the streaming reader produce, chunk by chunk, exactly the text one whole-string parse
    would have produced.
    """
    for n in range(min(len(text), _MAX_MARKER_LEN - 1), 0, -1):
        if text[-n:] in _MARKER_PREFIXES:
            return n
    last = None
    for last in _MARKER_RE.finditer(text):  # noqa: B007 — the LAST match is the one in question
        pass
    if last is not None and last.end() == len(text):
        return len(text) - last.start()
    return 0


class DestinationMarkerReader:
    """The streaming half of the same rule: deltas in, marker-free text out.

    Selection fires the moment the accumulated text carries a complete marker — mid-stream, not
    at the end — because that is when the surface can finally be bound and the answer can start
    going out live. Text ahead of the marker is held rather than released, so a marker split
    across chunk boundaries is still one marker, and nothing partial is ever handed downstream.
    """

    def __init__(self, turn: Any, message: Any = None) -> None:
        self._turn = turn
        self._message = message
        self._pending = ""
        self.destination: Optional[str] = None

    def _accept(self, text: str) -> None:
        destination, _ = parse_destination_marker(text)
        if destination is None or self.destination is not None:
            return
        self.destination = destination
        if self._turn is not None:
            self._turn.select_destination(destination, message=self._message)

    def feed(self, chunk: Optional[str]) -> str:
        """One delta. Returns the text (possibly empty) that may go downstream now."""
        self._pending += chunk or ""
        # Acceptance reads the WHOLE accumulation, including the held tail: a marker that arrived
        # complete must select on the chunk that carried it, not on the one after.
        self._accept(self._pending)
        held = _unresolved_tail(self._pending)
        cut = len(self._pending) - held
        source, self._pending = self._pending[:cut], self._pending[cut:]
        return parse_destination_marker(source)[1]

    def flush(self) -> str:
        """The stream ended: release whatever was being held back."""
        self._accept(self._pending)
        source, self._pending = self._pending, ""
        return parse_destination_marker(source)[1]


def get_set_reply_destination_schema() -> dict:
    """The tool the model calls to place its own reply. Closed enum, required argument: an
    unrecognized value is a rejected call, never a guess at what was meant."""
    return {
        "type": "function",
        "name": SET_REPLY_DESTINATION,
        "description": (
            "Choose where this reply goes, before you write it. Call this exactly once, and "
            "always before your answer: `thread` starts a thread under the message you are "
            "answering, `channel` posts at the top level of the channel where everyone reading "
            "along sees it without opening anything. This posts nothing itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": list(SELECTABLE_DESTINATIONS),
                    "description": (
                        "thread: the default, and right for anything long, detailed, "
                        "specialized, or of interest mainly to the person who asked. "
                        "channel: a short answer the whole room benefits from seeing inline."
                    ),
                },
            },
            "required": ["destination"],
            "additionalProperties": False,
        },
    }


async def execute_set_reply_destination(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Record the choice on the turn. Posts nothing, and never raises.

    All the judgment lives in `TurnRuntime.select_destination` — the validity rules are about
    turn state (already chosen? already posted?), so they belong with the state rather than
    here, where they would be a second copy for the next caller to diverge from."""
    turn = getattr(ctx, "turn", None)
    if turn is None or not hasattr(turn, "select_destination"):
        # No turn to write to: the caller is a path that does not own a reply surface. Refusing
        # is honest — claiming success would tell the model its choice took effect.
        logger.debug("set_reply_destination called with no turn in context")
        return {"ok": False, "error": "no_turn",
                "message": "This turn cannot choose a reply destination."}
    return turn.select_destination(args.get("destination"),
                                   message=getattr(ctx, "message", None))


def destination_tool_available(cfg: dict) -> bool:
    """Exposed ONLY where both destinations are genuinely legal. `_destination_choice_open` is
    stamped per request by the text handler from the turn's own state; absent → not offered,
    which is the safe default (the reply lands in the thread, as it always did)."""
    return bool(cfg.get("_destination_choice_open"))


def register_destination_tools(registry: ToolRegistry) -> None:
    """W4: RETIRED ON THE CHANNEL SURFACE, alive everywhere else.

    The channel surface is the only place the tool was ever offered, and the marker replaced it
    there. Retirement is STATIC — a constant `channel_enabled` decided at registry construction,
    not a per-turn predicate — because per-request gates are structurally ignored on that surface
    anyway, and a schema set that varies within a channel is the cache fork the channel layout
    exists to remove. The tool and its executor stay registered so any other surface keeps the
    behavior it has today, and `select_destination` still refuses a call that has nothing to
    choose.
    """
    registry.register(get_set_reply_destination_schema(), execute_set_reply_destination,
                      name=SET_REPLY_DESTINATION, enabled=destination_tool_available,
                      channel_enabled=lambda _cfg: False)
