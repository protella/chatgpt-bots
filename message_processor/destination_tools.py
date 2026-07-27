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
"""
from typing import Any, Dict

from logger import setup_logger
from message_processor.turn_runtime import SELECTABLE_DESTINATIONS
from tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.DestinationTools")

SET_REPLY_DESTINATION = "set_reply_destination"


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
    registry.register(get_set_reply_destination_schema(), execute_set_reply_destination,
                      name=SET_REPLY_DESTINATION, enabled=destination_tool_available)
