from __future__ import annotations

import asyncio
import random
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping,
                    Optional, Tuple, cast)
from uuid import uuid4

import aiohttp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

import prompts
from base_client import HistoryFetchError, Message
from config import (SUPPORTED_CHAT_MODELS, config, dev_epoch_fence_requested,
                    pipeline_status_markers, valid_emoji_name)
from message_processor import participation_telemetry
from message_processor.stale_send_guard import StaleSendSuppressed
from message_processor.turn_runtime import (DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                            EDIT_STATE_COMMITTED, EditRecord, EffectRevoked,
                                            LaunchNotRecorded, mark_tool_launched, run_effect)
from message_markers import (
    CONTINUATION_HEAD,
    continuation_trailer,
    extract_continuation_markers,
    fence_safe_chunks,
    is_checklist_status_text,
)
from slack_client.event_handlers.feedback import (
    FEEDBACK_ACTION_ID,
    USER_SETTINGS_ACTION_ID,
    build_feedback_blocks,
    feedback_enabled,
    should_offer_feedback,
)
from slack_client._host import _Host
from slack_client.formatting.blocks import extract_supplementary_text
from slack_client.normalizer import TimestampError, parse_ts
from slack_client.utilities import is_user_shaped_id, strip_citations

import re as _re

# The dev-only epoch fence. THE IMPORT IS GATED ON THE FLAG, exactly as in `main.py`: with
# DEV_EPOCH_FENCE_ENABLE empty this module is never imported, so flag-off is import-inert and not
# merely failure-inert, and no defect in the fence can reach a production send path.
#
# The predicate is `config.dev_epoch_fence_requested`, the single definition every import gate in
# this codebase shares — it decides whether the fence module is imported, so it cannot live in it.
#
# With the flag SET, an import failure here is logged at ERROR and leaves the fence disabled, but
# it is `main.initialize` that REFUSES TO START on it. This module is also imported by tools and
# tests that have no boot sequence to abort, so the abort belongs where the boot is.
_epoch_fence: Any = None   # the module itself once imported; typed loosely so the rebind is legal
if dev_epoch_fence_requested():
    try:
        from message_processor import epoch_fence as _epoch_fence
    except Exception as _e:  # noqa: BLE001 — reported here, fatal in main.initialize()
        import logging as _logging
        _logging.getLogger("slack_bot.Messaging").error(
            f"DEV_EPOCH_FENCE_ENABLE is set but the epoch fence module could not be imported "
            f"({type(_e).__name__}: {_e}); every send on this path is UNFENCED.")


class _NeverRaised(Exception):
    """Stands in for `EpochEffectRefused` when the fence module is unavailable, so the `except`
    clauses below stay well-formed instead of branching on whether an import succeeded."""


_EPOCH_REFUSED = (_epoch_fence.EpochEffectRefused if _epoch_fence is not None else _NeverRaised)


# MODULE-LEVEL, not methods on the mixin, and that is deliberate. Several of the methods below are
# driven UNBOUND against `MagicMock` stand-ins, where every attribute access invents a coroutine or
# a truthy value — so an instance-method form would fabricate a refusal and silently skip a write
# nothing was fencing. Whether an effect is allowed is not a question the receiver gets to answer.


def _epoch_authorize(client: Any, channel_id: Optional[str], site: str) -> None:
    """RAISE when a fence refuses this write. For the answer and upload paths, where a swallowed
    refusal would look to the caller exactly like a successful post.

    `team_id` comes off the client's authenticated identity, never an argument: the object making
    the call already holds the real workspace, so it cannot be handed a guessed one. A stand-in
    whose `self_team_id` is a mock simply matches no fence, which is the right answer.
    """
    if _epoch_fence is None:
        return
    _epoch_fence.authorize_effect(getattr(client, "self_team_id", None), channel_id, site=site)


def _epoch_refused(client: Any, channel_id: Optional[str], site: str) -> bool:
    """True when a fence refuses this write, logged at WARNING. For the best-effort helpers whose
    documented contract is that they never raise — chrome, reactions, status, footers."""
    try:
        _epoch_authorize(client, channel_id, site)
        return False
    except _EPOCH_REFUSED as e:
        log = getattr(client, "log_warning", None)
        if callable(log):
            log(f"Epoch fence refused {site}: {e}")
        return True

# EDIT §5: one process-wide keyed async lock per (team, channel, message_ts) — the WHOLE
# edit_own_message transaction runs under it, preflight through accounting, so two turns can
# never both announce and overwrite one message. Module-level on purpose (the keying is
# process-wide, not per client instance). §11.11: an entry is PRUNED when it is uncontended at
# transaction end, so the map never grows monotonically with the messages ever edited.
_EDIT_TRANSACTION_LOCKS: Dict[Any, "asyncio.Lock"] = {}


def _edit_transaction_lock(team_id: Any, channel_id: Any, message_ts: Any) -> "asyncio.Lock":
    key = (team_id, channel_id, message_ts)
    lock = _EDIT_TRANSACTION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _EDIT_TRANSACTION_LOCKS[key] = lock
    return lock


def _prune_edit_transaction_lock(key: Any, lock: "asyncio.Lock") -> None:
    """§11.11: drop the map entry when its transaction ends uncontended.

    Sound because the fetch (`_edit_transaction_lock`) and the start of acquisition happen
    without an intervening suspension point: a contender is either already IN `_waiters` (or
    holding the lock) — in which case the entry survives — or has not fetched yet, and will
    fetch a fresh entry with nothing left to race."""
    if (_EDIT_TRANSACTION_LOCKS.get(key) is lock and not lock.locked()
            and not getattr(lock, "_waiters", None)):
        del _EDIT_TRANSACTION_LOCKS[key]


@asynccontextmanager
async def _edit_transaction_locked(team_id: Any, channel_id: Any, message_ts: Any):
    """The §5 keyed critical section, with the §11.11 pruning on the way out. The fetch and
    the start of acquisition run without an intervening suspension point (the fast-path
    acquire never yields), which is what makes the prune sound."""
    key = (team_id, channel_id, message_ts)
    lock = _edit_transaction_lock(team_id, channel_id, message_ts)
    try:
        async with lock:
            yield
    finally:
        _prune_edit_transaction_lock(key, lock)


def _strip_block_ids(value: Any) -> Any:
    """Blocks as WE wrote them: Slack stamps server-side `block_id`s on read-back, and the §5
    ambiguous-update equality is about the content we sent, not identifiers we never chose."""
    if isinstance(value, dict):
        return {k: _strip_block_ids(v) for k, v in value.items() if k != "block_id"}
    if isinstance(value, (list, tuple)):
        return [_strip_block_ids(v) for v in value]
    return value


def _blocks_match(fetched: Any, rebuilt: Any) -> bool:
    """§5's footer-reply equality half: the exact rebuilt blocks, modulo the `block_id`s Slack
    mints on storage. Anything unreadable compares unequal — no blind retry either way."""
    try:
        return _strip_block_ids(fetched or []) == _strip_block_ids(rebuilt or [])
    except Exception:  # noqa: BLE001
        return False


# Block action_ids that mark a message as one of our UI helpers (channel footer's
# Configure button, Phase H feedback strip + its user-settings button). PURE-chrome
# messages are not conversation — the history rebuild must never feed them back into
# context. But a real response can now CARRY the Configure chrome (native streaming
# attaches it on stopStream), so helper action_ids alone no longer condemn a message:
# only the known chrome fallback texts do. Those fallbacks are exactly what the
# separate helper posts set as `text`: the model label (footer), "Rate this response"
# (feedback strip), and the legacy "Settings available".
_UI_HELPER_ACTION_IDS = frozenset({
    "open_channel_settings", FEEDBACK_ACTION_ID, USER_SETTINGS_ACTION_ID,
})
# Notification/accessibility fallback text for a standalone Configure-footer post. NOT a bare
# model name (that reads as a spurious message when Slack surfaces the fallback — the live
# "gpt-5.6-sol" post 2026-07-16); still recognized here so the history rebuild skips it as chrome.
RESPONSE_FOOTER_FALLBACK_TEXT = "Channel settings"
_UI_HELPER_FALLBACK_TEXTS = frozenset(
    {"Rate this response", "Settings available", RESPONSE_FOOTER_FALLBACK_TEXT}
    | set(SUPPORTED_CHAT_MODELS)
)


@dataclass(frozen=True)
class Delivery:
    """What Slack ACTUALLY accepted for one send, as against what we handed it.

    `text` is the POSTED text: the formatted string that went to Slack, not the caller's source
    markdown. Those two differ — `format_text` rewrites bold, links and lists into Slack's own
    syntax — and reporting the source made the claim "what the room saw" false in every send that
    formatting touched. A split reports the chunks that landed, joined the way Slack holds them
    (the continuation heads are chrome the rebuild strips, so they are not part of what was said).
    `complete` is the flag a COMMITTED destination record carries, so a half-delivered answer can
    never be remembered as a whole one.
    """

    first_ts: Optional[str]
    text: str
    complete: bool
    parts_delivered: int
    parts_total: int
    split: bool = False
    # 1-based index of the part that failed twice and aborted the remainder. None when whole.
    truncated_at: Optional[int] = None


def _report_delivery(meta_out: Optional[dict], delivery: Delivery) -> Delivery:
    if meta_out is not None:
        meta_out["delivery"] = delivery
    return delivery


def _note_first_accept(callback: Optional[Callable[[str], None]], ts: Optional[str]) -> None:
    """Tell the caller the room can see words NOW, before the rest of the send runs.

    A record appended only after the whole coroutine returns is lost to a cancellation that
    lands between the first accepted chunk and the last — and that is exactly the case where
    text is visible in Slack with nothing in the ledger claiming it. Bookkeeping never breaks
    a send, so the callback's own failure is swallowed.
    """
    if callback is None or not ts:
        return
    try:
        callback(ts)
    except Exception:  # noqa: BLE001
        pass


# A mention of a bot OBJECT id. Slack has no such mention: `<@B07ABC>` reaches the channel as
# those literal characters, so the room reads an id instead of a name. The model only ever sees
# one when an upstream surface offered it one (a peer app posting in agent mode names itself by
# its B id and nothing else), which the roster guard and the resolver now prevent — this is the
# last line, at the transport, where "never let it reach Slack" can actually be guaranteed.
_BOT_OBJECT_MENTION_RE = re.compile(r"<@(B[A-Z0-9]+)(?:\|[^>]*)?>")

# CODE IS QUOTED, NOT ADDRESSED. A fenced block or an inline span is content the model is showing
# the room — a log line, a payload, the very id we are discussing — and rewriting inside it
# changes what the answer says it saw. Slack's own link form `<http…|label>` needs no exemption:
# the pattern above anchors on `<@`, which a link never has.
_FENCE = "```"

# The longest trailing fragment the native-stream guard will hold waiting for a `>`. A real
# mention (`<@B` + id + optional `|label`) is far shorter; past this the text is prose that
# happens to contain a `<`, and holding it back would stall the stream.
_BOT_MENTION_HOLD_MAX = 128

# An unfinished `<@B…` candidate: the id, and an optional label that has not closed yet.
_OPEN_BOT_MENTION_RE = re.compile(r"<@B[A-Z0-9]*(?:\|[^>]*)?")


def _code_spans(text: str, *, in_fence: bool = False, in_inline: bool = False,
                terminal: bool = True) -> Tuple[List[Tuple[str, bool]], bool, bool]:
    """Split `text` into `(chunk, is_code)` runs, carrying fence/inline state IN and OUT.

    One definition of "what is quoted", used twice. A whole message answers the question by
    itself (`terminal=True`: an unclosed fence or backtick is prose, exactly as Slack renders it).
    A STREAM cannot: the fence opens in one delta and the mention arrives in the next, so the
    state has to survive between calls, and an unclosed span is treated as open because the delta
    that closes it has not arrived yet.
    """
    segments: List[Tuple[str, bool]] = []
    end = len(text)
    i = 0

    def emit(chunk: str, is_code: bool) -> None:
        if chunk:
            segments.append((chunk, is_code))

    while i < end:
        if in_fence:
            close = text.find(_FENCE, i)
            if close < 0:
                emit(text[i:], True)
                i = end
            else:
                emit(text[i:close + len(_FENCE)], True)
                i = close + len(_FENCE)
                in_fence = False
        elif in_inline:
            close = i
            while close < end and text[close] not in ("`", "\n"):
                close += 1
            if close >= end:
                emit(text[i:], True)
                i = end
            elif text[close] == "`":
                emit(text[i:close + 1], True)
                i = close + 1
                in_inline = False
            else:
                # A newline ends an inline span — Slack has no multi-line inline code.
                emit(text[i:close], True)
                emit("\n", False)
                i = close + 1
                in_inline = False
        else:
            tick = text.find("`", i)
            if tick < 0:
                emit(text[i:], False)
                break
            emit(text[i:tick], False)
            if text.startswith(_FENCE, tick):
                if terminal and text.find(_FENCE, tick + len(_FENCE)) < 0:
                    emit(_FENCE, False)       # never closed: prose, not a block
                else:
                    emit(_FENCE, True)
                    in_fence = True
                i = tick + len(_FENCE)
            else:
                line_end = text.find("\n", tick + 1)
                closes_here = text.find("`", tick + 1, end if line_end < 0 else line_end) >= 0
                if terminal and not closes_here:
                    emit("`", False)          # a lone backtick is punctuation, not a span
                else:
                    emit("`", True)
                    in_inline = True
                i = tick + 1
    return segments, in_fence, in_inline


def _mention_rewriter(client: Any) -> Callable[[str], str]:
    """The substitution itself: `<@B…>` -> the cached USER id, or the plain word `@bot`, which
    reads as a reference to some bot rather than as a machine id nobody can parse."""
    lookup = getattr(client, "bot_user_id_for", None)

    def _sub(match: "re.Match[str]") -> str:
        resolved = None
        if callable(lookup):
            try:
                resolved = lookup(match.group(1))
            except Exception:  # noqa: BLE001 — a cache peek never breaks a send
                resolved = None
        # Only a genuine USER id may be written back: whatever the cache hands us, the one thing
        # this function promises is that what reaches Slack is mentionable.
        if isinstance(resolved, str) and is_user_shaped_id(resolved):
            return f"<@{resolved}>"
        return "@bot"

    return lambda chunk: _BOT_OBJECT_MENTION_RE.sub(_sub, chunk)


def rewrite_bot_object_mentions(client: Any, text: str) -> str:
    """Outbound text with every `<@B…>` made postable: the bot's USER id when the cache knows it,
    otherwise the plain word `@bot`. Code spans are left exactly as written, and text with no such
    mention is returned untouched."""
    if not text or "<@B" not in text:
        return text
    rewrite = _mention_rewriter(client)
    segments, _fence, _inline = _code_spans(text)
    return "".join(chunk if is_code else rewrite(chunk) for chunk, is_code in segments)


def split_partial_code_delimiter(text: str) -> Tuple[str, str]:
    """`(scannable, hold_back)` — hold a trailing run of ONE or TWO backticks.

    A delimiter arrives in deltas too: a fence sent as "`" then "``\n" would open an inline span
    in the first chunk and close nothing in the second, and every mention after it would be
    classified against a code state that never existed. Three or more backticks are already a
    complete delimiter and go out. The hold is two characters, so nothing can stall on it."""
    run = 0
    while run < len(text) and text[len(text) - 1 - run] == "`":
        run += 1
    if run in (1, 2):
        return text[:len(text) - run], text[len(text) - run:]
    return text, ""


def split_streamable_bot_mention(text: str) -> Tuple[str, str]:
    """`(release_now, hold_back)` for one streamed accumulation.

    A native stream sends DELTAS, so `<@B07ABC>` can arrive as `<@B0` + `7ABC>` and a per-delta
    rewrite would see neither half. The fragment that could still become a mention is held for the
    delta that finishes it (the DestinationMarkerReader rule, applied to a different marker).

    PAST THE BOUND the fragment is not held — but it is not released as written either. A
    `<@B…` that goes out intact becomes a real mention the moment a later delta brings the `>`,
    which is the whole failure this guard exists to prevent, so the opening bracket is dropped:
    what reaches the room is literal text Slack cannot render as anything.
    """
    start = text.rfind("<")
    if start < 0:
        return text, ""
    fragment = text[start:]
    if ">" in fragment:
        return text, ""
    if len(fragment) < len("<@B"):
        return (text[:start], fragment) if "<@B".startswith(fragment) else (text, "")
    if not _OPEN_BOT_MENTION_RE.fullmatch(fragment):
        return text, ""
    if len(fragment) <= _BOT_MENTION_HOLD_MAX:
        return text[:start], fragment
    return text[:start] + fragment[1:], ""    # "<@B07…" -> "@B07…": nothing left to close


# EDIT §11.21: announcement-send exceptions whose PHYSICAL outcome is ambiguous — the request
# may already have reached Slack when they were raised. Timeouts and connection failures are
# raised at/after dispatch (and a reconciliation miss re-raises exactly those, flagged via
# meta_out["transport_ambiguous"] for the types this tuple cannot anticipate); SlackApiError
# rides along defensively for the paths where it escapes rather than becoming a None return.
# Every exception OUTSIDE this classification is deterministic-local: raised before anything
# could reach Slack.
_AMBIGUOUS_SEND_EXC = (SlackApiError, TimeoutError, ConnectionError, OSError,
                       asyncio.TimeoutError)


def _is_ui_helper_message(msg: dict) -> bool:
    """True when a message is PURE UI chrome: a helper action_id in its blocks AND no
    text beyond the helper's own notification fallback. A content-bearing response
    with the Configure chrome attached keeps its real text and is preserved."""
    has_helper = any(
        el.get("action_id") in _UI_HELPER_ACTION_IDS
        for b in (msg.get("blocks") or []) if b.get("type") in ("actions", "context_actions")
        for el in (b.get("elements") or [])
    )
    if not has_helper:
        return False
    text = (msg.get("text") or "").strip()
    return not text or text in _UI_HELPER_FALLBACK_TEXTS

# Own-message placeholder/status filters for history rebuild (R3): only messages
# shaped like ":emoji: Status text" AND carrying a known transient marker are
# skipped — a human message merely containing the word "Thinking" is kept.
_SELF_STATUS_RE = _re.compile(r"^:[a-z0-9_+\-]+:\s")

# assistant.threads.setStatus renders PLAIN TEXT — :shortcodes: appear literally
# (user screenshot 2026-07-09). Posted messages render them fine, so status strings
# are sanitized only at the setStatus boundary: known shortcodes become Unicode,
# unknown ones (incl. workspace custom emoji, which have no Unicode form) are
# stripped. One configured string thus renders correctly on both surfaces.
_SHORTCODE_TO_UNICODE = {
    "hourglass_flowing_sand": "⏳", "hourglass": "⌛", "mag": "🔍",
    "bar_chart": "📊", "brain": "🧠", "bulb": "💡", "gear": "⚙️",
    "robot_face": "🤖", "sparkles": "✨", "thinking_face": "🤔",
    "memo": "📝", "art": "🎨", "camera": "📷", "globe_with_meridians": "🌐",
}
_SHORTCODE_RE = _re.compile(r":([a-z0-9_+\-]+):")


def _status_plain_text(text: str) -> str:
    """Render a status string for the plain-text setStatus surface."""
    def sub(m):
        return _SHORTCODE_TO_UNICODE.get(m.group(1), "")
    return _SHORTCODE_RE.sub(sub, text or "").strip() or "working on it…"
_SELF_STATUS_MARKERS = (
    "Thinking...",
    "Rebuilding thread history",
    "Catching up on",
)


def is_self_chrome_message(text: str, msg: dict, *, markers=None) -> bool:
    """True when a message is our OWN transient UI chrome — a status/placeholder line
    (":emoji: Thinking…"), a progress checklist ("✓ …"), the legacy processing notice, the
    "Settings available" button, or a pure UI-helper block (Configure button / feedback strip)
    with no real text. Such messages must never be replayed as an assistant turn (history
    rebuild) NOR recorded as authoritative `[self]` addressee evidence (F47 cold-start backfill).

    Content-bearing replies — even ones carrying the Configure chrome attached on stopStream —
    are NOT chrome and return False. The caller decides ownership (only pass our own messages for
    the self-status checks to be meaningful); this only classifies the shape. Fail-open: any
    error classifies as NOT chrome, so a real reply is never silently dropped.

    `markers` PINS the pipeline status-marker list for one caller. Omitted, it reads the LIVE
    list exactly as it always has, so every existing call site is byte-identical. The channel
    turn passes the list frozen into its pin instead: a marker-list change landing mid-turn would
    otherwise silently alter the bytes an admitted turn was already committed to."""
    try:
        text = text or ""
        # F1 progress-checklist ("✓ …") — carries an invisible marker, not the ":emoji:" shape.
        if is_checklist_status_text(text):
            return True
        # Transient placeholders/status lines: ":emoji: Thinking..." and same-shaped updates.
        if _SELF_STATUS_RE.match(text) and (
            any(marker in text for marker in _SELF_STATUS_MARKERS)
            or any(marker in text
                   for marker in (pipeline_status_markers() if markers is None else markers))
        ):
            return True
        # Legacy busy/processing notice.
        if ":warning:" in text and "currently processing" in text:
            return True
        # Settings button message.
        if text == "Settings available":
            return True
        # Pure UI-helper block (Configure button / Phase H feedback strip) with no real text.
        if _is_ui_helper_message(msg):
            return True
    except Exception:
        return False
    return False


class NativeStreamSession:
    """Adapter over Slack's native streaming API (chat.startStream/appendStream/stopStream).

    The existing streaming handler thinks in CUMULATIVE text ("everything so far"), while the
    native API wants DELTAS. This session bridges the two: feed it the cumulative text each tick
    via ``update()`` and it appends only the new tail. ``finish()`` closes the stream (optionally
    with blocks, which Slack only allows on stop).

    Fully best-effort: if startStream fails the session is inert (``active`` False) and the caller
    falls back to the legacy ``update_message_streaming`` path. Any later failure also flips it
    inert so the caller can recover.
    """

    def __init__(self, client, channel_id: str, thread_id: str, logger=None,
                 team_id: Optional[str] = None, user_id: Optional[str] = None,
                 owner: Any = None):
        self._client = client
        # The BOT (not the web client) — the only object that holds the bot_id -> user_id cache
        # the outbound mention guard reads. Optional so a caller with no bot still works; the
        # guard then falls back to "@bot", which is still never a raw `<@B…>`.
        self._owner = owner
        self._channel = channel_id
        self._thread = thread_id
        self._log = logger
        # chat.startStream requires BOTH recipient_team_id (workspace) and
        # recipient_user_id (the triggering user) for channel streaming — missing either
        # 400s (missing_recipient_team_id / missing_recipient_user_id). appendStream/
        # stopStream key off the returned ts and need neither.
        self._team_id = team_id
        self._user_id = user_id
        self.ts: Optional[str] = None
        self.active: bool = False
        self._sent: str = ""
        # Raw text withheld from Slack because it could still become a `<@B…>` mention. Released
        # by the next delta that resolves it, or by finish().
        self._held: str = ""
        # Where the code-span scanner stands, CUMULATIVELY. A fence opens in one delta and the
        # mention lands in the next, so a per-chunk scan would rewrite inside a block it never
        # saw open.
        self._in_fence: bool = False
        self._in_inline: bool = False

    async def start(self, initial_text: str = "", lease: Any = None) -> bool:
        """`lease` (stale guard): chat.startStream MINTS the reply message, so this is a first
        answer surface. Checked before the call, never after — once a stream is up, the rest of
        the answer follows it."""
        if lease is not None:
            lease.authorize("native_start")
        # EPOCH FENCE: chat.startStream MINTS the reply message, so refusing here refuses the
        # whole stream — appendStream and stopStream key off a ts that will never exist. The
        # session goes inert exactly as it does on any other start failure.
        try:
            if _epoch_fence is not None:
                _epoch_fence.authorize_effect(self._team_id, self._channel,
                                              site="chat_startStream")
        except _EPOCH_REFUSED as e:
            if self._log:
                self._log(f"epoch fence refused the native stream: {e}")
            self.active = False
            return False
        if not self._thread:
            # chat.startStream REQUIRES thread_ts (Slack: "missing required field"),
            # so top-level channel replies can never stream natively. Skip the
            # guaranteed-to-fail call; the caller falls back to legacy streaming.
            if self._log:
                self._log("native streaming requires a thread — top-level reply falls back to legacy")
            self.active = False
            return False
        if not self._team_id or not self._user_id:
            # chat.startStream now REQUIRES recipient_team_id AND recipient_user_id for
            # channel streaming (Slack: "missing_recipient_team_id" /
            # "missing_recipient_user_id"). Without both the call is guaranteed to fail —
            # skip it and let the caller fall back to legacy streaming (never crash).
            if self._log:
                missing = "team_id" if not self._team_id else "user_id"
                self._log(f"native streaming requires a {missing} — falling back to legacy")
            self.active = False
            return False
        try:
            opening = self._releasable(initial_text or "")
            resp = await self._client.chat_startStream(  # unleased-ok: inside NativeStreamSession.start, which authorized at its entry
                channel=self._channel,
                thread_ts=self._thread,
                markdown_text=opening or None,
                recipient_team_id=self._team_id,
                recipient_user_id=self._user_id,
            )
            self.ts = resp.get("ts")
            self._sent = initial_text or ""
            self.active = bool(self.ts)
            if self.active and lease is not None:
                lease.commit()      # the reply message exists; the rest belongs with it
            return self.active
        except Exception as e:  # noqa: BLE001 - best-effort, never fatal
            if self._log:
                self._log(f"native stream start failed, will fall back: {e}")
            self.active = False
            return False

    def _releasable(self, raw: str) -> str:
        """The part of `self._held + raw` that may go out NOW, rewritten; the rest is held.

        THE ORDER IS THE POINT. Classification runs first, on text whose partial delimiters have
        been held back, and only the trailing PROSE run is then split for an unfinished mention:
        a `<@B…` the model is quoting inside a fence is code, and neutralizing its bracket would
        edit what the answer says it saw. A held prose fragment can never contain a backtick —
        any backtick would have started a code run — so holding it cannot desynchronize the
        code state either.
        """
        scannable, ticks = split_partial_code_delimiter(self._held + (raw or ""))
        segments, self._in_fence, self._in_inline = _code_spans(
            scannable, in_fence=self._in_fence, in_inline=self._in_inline, terminal=False)
        rewrite = _mention_rewriter(self._owner)
        held = ""
        out: List[str] = []
        last = len(segments) - 1
        for index, (chunk, is_code) in enumerate(segments):
            if is_code:
                out.append(chunk)               # quoted verbatim, mentions and all
                continue
            if index == last:
                # Only the FINAL prose run can still grow: anything before it is already
                # terminated by the code run that follows.
                chunk, held = split_streamable_bot_mention(chunk)
            out.append(rewrite(chunk))
        self._held = held + ticks
        return "".join(out)

    def _flush(self) -> str:
        """The stream is over: everything held goes out, since no delta can complete it now."""
        pending, self._held = self._held, ""
        if not pending:
            return ""
        segments, self._in_fence, self._in_inline = _code_spans(
            pending, in_fence=self._in_fence, in_inline=self._in_inline, terminal=False)
        rewrite = _mention_rewriter(self._owner)
        return "".join(chunk if is_code else rewrite(chunk) for chunk, is_code in segments)

    async def update(self, cumulative_text: str) -> bool:
        """Append the new tail of ``cumulative_text`` since the last update."""
        if not self.active or self.ts is None:
            return False
        delta = cumulative_text[len(self._sent):] if cumulative_text.startswith(self._sent) else cumulative_text
        if not delta:
            return True
        outgoing = self._releasable(delta)
        # The whole delta is a possible mention prefix: nothing may be appended yet, but the
        # cumulative mark still advances or the next tick would resend it.
        if not outgoing:
            self._sent = cumulative_text
            return True
        try:
            await self._client.chat_appendStream(channel=self._channel, ts=self.ts, markdown_text=outgoing)
            self._sent = cumulative_text
            return True
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log(f"native stream append failed: {e}")
            self.active = False
            return False

    async def finish(self, final_text: Optional[str] = None, blocks=None) -> bool:
        if self.ts is None:
            return False
        try:
            kwargs: Dict[str, Any] = {"channel": self._channel, "ts": self.ts}
            if final_text is not None and final_text.startswith(self._sent):
                tail = final_text[len(self._sent):]
            elif final_text is not None:
                tail = final_text
            else:
                tail = ""
            # THE FLUSH. The stream is over, so a held fragment will never be completed by
            # another delta: it goes out now, rewritten, rather than being lost.
            released = self._releasable(tail) + self._flush()
            if released:
                kwargs["markdown_text"] = released
            if blocks is not None:
                kwargs["blocks"] = blocks
            await self._client.chat_stopStream(**kwargs)
            self.active = False
            return True
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log(f"native stream stop failed: {e}")
            self.active = False
            return False


class WorkspaceEmojiCache:
    """Process-lifetime cache of the workspace's CUSTOM emoji shorthand names (emoji.list).

    Reachable from BOTH the react_to_message tool-schema factory and the participation gate
    via ``client.workspace_emojis``. Holds a sorted/deduped tuple of names plus a monotonic
    expiry; ``refresh()`` is the only thing that hits Slack, and ``get_custom_emoji_names()``
    is a sync, stale-tolerant getter that schedules a background refresh when expired but
    never blocks a request.

    Fail-soft everywhere: any error (including the emoji:read scope being absent — which is
    also the de-facto off switch) RETAINS the last good tuple; an empty tuple only ever means
    we have never had a successful fetch. The model then simply sees no customs, and no turn
    fails on account of emoji.list.
    """

    def __init__(self, client):
        self._client = client
        self._names: tuple = ()
        self._expiry: float = 0.0        # monotonic deadline; 0 = never fetched
        self._lock = asyncio.Lock()
        self._refreshing: bool = False   # guards against scheduling overlapping refreshes
        self._refresh_task = None        # ref to the scheduled task (GC + lifecycle guard)

    def _log_debug(self, msg: str) -> None:
        log = getattr(self._client, "log_debug", None)
        if log:
            log(msg)

    async def refresh(self) -> tuple:
        """Fetch emoji.list and rebuild the name tuple.

        Parses resp["emoji"] KEYS (both real customs and ``alias:*`` entries — the KEY is the
        alias NAME reactions.add accepts, so aliases are kept), filters each through
        ``valid_emoji_name``, then sorts + dedupes. On ANY error the last good tuple is kept
        (empty only if never fetched). The TTL is reset either way, so a persistent failure
        (e.g. missing emoji:read) backs off instead of hammering the API on every getter call.
        """
        async with self._lock:
            try:
                resp = await self._client.app.client.emoji_list()
                emoji = (resp or {}).get("emoji") or {}
                names = {
                    name for name in ((k or "").strip().strip(":") for k in emoji.keys())
                    if valid_emoji_name(name)
                }
                self._names = tuple(sorted(names))
            except Exception as e:  # noqa: BLE001 — never fatal; keep the last good tuple
                self._log_debug(f"workspace emoji refresh failed, keeping last good: {e}")
            finally:
                self._expiry = time.monotonic() + max(
                    1.0, float(getattr(config, "workspace_emoji_ttl_seconds", 3600)))
            return self._names

    def get_custom_emoji_names(self) -> tuple:
        """Sync, stale-ok. Returns the current tuple immediately; if it has expired AND no
        refresh is already running, schedules a background refresh (fire-and-forget). Never
        awaits, never raises — a request path can call this freely."""
        if time.monotonic() >= self._expiry and not self._refreshing:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return self._names  # no running loop — just return what we have
            self._refreshing = True  # set synchronously so a burst schedules exactly one refresh
            try:
                task = loop.create_task(self.refresh())
                self._refresh_task = task  # keep a ref so the task isn't GC'd mid-flight
                # Own the guard's lifecycle HERE, not in refresh()'s finally: the callback fires on
                # completion, error, AND cancellation, so a task cancelled before refresh() runs can't
                # wedge _refreshing=True and block every future refresh. And since only this getter
                # sets the flag True (once, under the guard), there is no premature-clear race.
                task.add_done_callback(self._on_refresh_done)
            except Exception as e:  # noqa: BLE001
                self._refreshing = False
                self._log_debug(f"workspace emoji background refresh not scheduled: {e}")
        return self._names

    def _on_refresh_done(self, task) -> None:
        """Clear the refresh guard once the scheduled task settles (success, error, or cancel)."""
        self._refreshing = False
        self._refresh_task = None

    @staticmethod
    def _tokens(name: str) -> set:
        """Word-ish pieces of an emoji name, so `party-parrot` and `party_parrot` match alike.

        Single characters are dropped: the stray "a" in a query like "celebrate a win" is a
        real token of `alphabet-white-a` too, and it floated junk to the top of every result."""
        return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) > 1}

    def search(self, query: str, limit: int = 16) -> list:
        """Rank the workspace's CUSTOM emoji names against a free-text query.

        Purely lexical and in-memory: no API call, no model call, no keyword→emoji table. The
        model supplies the query and the model picks from what comes back — this only decides
        which CANDIDATES are worth showing, which is the same job the (alphabetical, useless)
        prefix slice used to do badly.

        Ranked: exact name > a query token that IS a name token > prefix > substring. Names
        that match more of the query win within a tier, shorter names break ties (`shipit`
        should outrank `shipit-parrot-gif` for "ship"). Never raises; returns [] for an empty
        query or an empty catalog."""
        q = (query or "").strip().lower()
        if not q:
            return []
        try:
            names = self.get_custom_emoji_names()
        except Exception:  # noqa: BLE001 — discovery must never fail a turn
            return []
        q_tokens = self._tokens(q)
        q_flat = re.sub(r"[^a-z0-9]+", "", q)
        scored = []
        for name in names:
            low = name.lower()
            flat = re.sub(r"[^a-z0-9]+", "", low)
            n_tokens = self._tokens(name)
            hits = len(q_tokens & n_tokens)
            if low == q or flat == q_flat:
                tier = 0
            elif hits:
                tier = 1
            elif any(flat.startswith(t) or low.startswith(t) for t in q_tokens):
                tier = 2
            elif any(t in flat for t in q_tokens if len(t) >= 3):
                tier = 3
            else:
                continue
            scored.append((tier, -hits, len(name), name))
        scored.sort()
        return [s[3] for s in scored[:max(0, int(limit or 0))]]


class SlackMessagingMixin(_Host):
    # How long stop() will wait out a socket-mode close that hung past its first 0.1s try. Long
    # enough for a real close to finish, short enough that a wedged one cannot hold shutdown.
    _HANDLER_CLOSE_TIMEOUT = 5.0

    if TYPE_CHECKING:
        # Created lazily on first use (see reserve_reaction_slot), so they are declared rather
        # than assigned — the shapes are (channel, ts) -> {emoji: Future|True} and
        # (channel, ts) -> monotonic touch time.
        _reaction_guard: "OrderedDict[Any, Dict[str, Any]]"
        _reaction_guard_ts: Dict[Any, float]

    async def start(self):
        """Start the Slack bot"""
        self.handler = AsyncSocketModeHandler(self.app, config.slack_app_token)
        self.log_info("Starting Slack bot in socket mode...")

        # F9: detection-only socket-liveness monitor. Hook every inbound envelope on the
        # async socket client and start the 60s watchdog (never touches the socket).
        try:
            from slack_client.socket_liveness import SocketLivenessMonitor
            self._socket_liveness = SocketLivenessMonitor(
                getattr(self.handler, "client", None),
                timeout=config.socket_liveness_timeout,
                log_info=self.log_info,
                log_warning=self.log_warning,
                log_error=self.log_error,
            )
            self._socket_liveness.attach()
            self._socket_liveness.start()
        except Exception as e:
            # Partial start (e.g. attach() installed the listener but start() failed): stop
            # the monitor so we never leave a dangling listener behind.
            self.log_warning(f"Could not start socket-liveness monitor: {e}")
            await self._stop_socket_liveness_quietly()

        # Resolve our own identity up front so we can tell our messages apart from other bots'
        await self._ensure_self_identity()

        # C1: warm the workspace custom-emoji cache once, now that identity is set. Fail-soft —
        # a missing emoji:read scope (or any API error) just leaves the cache empty and the
        # model sees no customs; the getter refreshes it lazily thereafter.
        cache = getattr(self, "workspace_emojis", None)
        if cache is not None:
            try:
                await cache.refresh()
            except Exception as e:  # noqa: BLE001 — startup must never break on emoji.list
                self.log_debug(f"initial workspace emoji refresh failed: {e}")

        # Create a task for start_async that can be cancelled
        self._start_task = asyncio.create_task(self.handler.start_async())

        try:
            await self._start_task
        except asyncio.CancelledError:
            self.log_info("Slack bot start task cancelled")
            await self._stop_socket_liveness_quietly()  # detach before propagating
            raise
        except Exception as e:
            self.log_error(f"Error in Slack bot start: {e}")
            await self._stop_socket_liveness_quietly()  # detach before propagating
            raise

    async def _stop_socket_liveness_quietly(self) -> None:
        """Stop + detach the socket-liveness monitor, swallowing errors and clearing the ref."""
        monitor = getattr(self, "_socket_liveness", None)
        self._socket_liveness = None
        if monitor is not None:
            try:
                await monitor.stop()
            except Exception as e:
                self.log_debug(f"Error stopping socket-liveness monitor: {e}")

    async def stop(self):
        """Stop the Slack bot"""
        # F9: stop the liveness monitor first (independent of the handler teardown below).
        monitor = getattr(self, "_socket_liveness", None)
        if monitor is not None:
            try:
                await monitor.stop()
            except Exception as e:
                self.log_debug(f"Error stopping socket-liveness monitor: {e}")
            self._socket_liveness = None

        if self.handler:
            self.log_info("Stopping Slack bot...")

            # Cancel the start task to break out of the blocking start_async call
            if hasattr(self, '_start_task') and not self._start_task.done():
                self.log_info("Cancelling start task...")
                self._start_task.cancel()
                try:
                    await self._start_task
                except asyncio.CancelledError:
                    self.log_info("Slack bot start task cancelled")

            # Try to close handler sessions first before calling handler.close_async()
            # Also try to close the socket client's session if it exists
            if hasattr(self.handler, 'client') and self.handler.client:
                if hasattr(self.handler.client, 'session') and self.handler.client.session:
                    if not self.handler.client.session.closed:
                        self.log_debug("Closing handler client session")
                        try:
                            await asyncio.wait_for(self.handler.client.session.close(), timeout=0.5)
                            self.log_debug("Handler client session closed")
                        except asyncio.TimeoutError:
                            self.log_warning("Timeout closing handler client session")
                        except Exception as e:
                            self.log_warning(f"Error closing handler client session: {e}")

                if hasattr(self.handler.client, 'aiohttp_client_session') and self.handler.client.aiohttp_client_session:
                    session = self.handler.client.aiohttp_client_session
                    if not session.closed:
                        # Don't call session.close() or connector.close() as they hang
                        # Just forcibly mark everything as closed
                        try:
                            # Mark the connector as closed without actually closing it
                            if hasattr(session, '_connector') and session._connector:
                                if hasattr(session._connector, '_closed'):
                                    session._connector._closed = True
                                # Clear any transports
                                if hasattr(session._connector, '_transports'):
                                    session._connector._transports = []
                                # Clear conns if it exists
                                if hasattr(session._connector, '_conns'):
                                    session._connector._conns = {}

                            # Also try the public connector attribute
                            if hasattr(session, 'connector') and session.connector:
                                if hasattr(session.connector, '_closed'):
                                    session.connector._closed = True

                            # Mark session as closed
                            if hasattr(session, '_closed'):
                                session._closed = True

                            # Try to detach from the event loop
                            if hasattr(session, '_loop'):
                                session._loop = None

                        except Exception as e:
                            self.log_warning(f"Error during force-close of aiohttp_client_session: {e}")

            # Now try to close the socket mode handler itself - but skip if it might hang
            # Check if we should even try - if we manually closed sessions, maybe skip handler close
            skip_handler_close = False
            if hasattr(self.handler, 'client') and self.handler.client:
                if hasattr(self.handler.client, 'aiohttp_client_session'):
                    # If we have the session and it's closed, we probably don't need handler.close_async
                    if self.handler.client.aiohttp_client_session.closed:
                        skip_handler_close = True

            if not skip_handler_close:
                try:
                    # Create a task for handler close so it doesn't block
                    close_task = asyncio.create_task(self.handler.close_async())

                    # Wait for it with a very short timeout since it tends to hang
                    try:
                        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.1)
                        self.log_debug("Socket mode handler closed")
                    except asyncio.TimeoutError:
                        self.log_warning("Socket mode handler close timed out after 0.1 seconds, "
                                         "waiting it out below")
                        # NOT abandoned. The close that is still running is the thing that stops
                        # Bolt dispatching, so walking away from it here made `stop()` returning
                        # mean nothing: callbacks kept arriving while the caller believed ingress
                        # was down. It is awaited at the end of stop() instead, on a real timeout.
                        self._handler_close_task = close_task
                        # And registered with the ingress barrier, because the end-of-stop() wait
                        # is bounded: if it times out, quiescence must not be declared on a close
                        # that can still dispatch. The barrier waits this task out before it looks
                        # at the callback count at all, and names it if it outlives that too.
                        # Local import: registration imports the event-handler stack, and this
                        # module sits underneath it.
                        from slack_client.event_handlers.registration import ingress
                        ingress.track_dispatcher(close_task, name="the socket-mode handler close")
                except Exception as e:
                    self.log_warning(f"Error closing socket mode handler: {e}")

        # Close the web client's aiohttp session if it exists
        if self.app:
            # Try the main client
            if self.app.client:
                try:
                    # The AsyncWebClient has a _session attribute that needs closing
                    if hasattr(self.app.client, '_session') and self.app.client._session:
                        if not self.app.client._session.closed:
                            await self.app.client._session.close()
                            self.log_info("Closed Slack web client session")
                except Exception as e:
                    self.log_warning(f"Error closing web client session: {e}")

            # Check for _async_client as well (some versions use this)
            if hasattr(self.app, '_async_client') and self.app._async_client:
                try:
                    if hasattr(self.app._async_client, '_session') and self.app._async_client._session:
                        if not self.app._async_client._session.closed:
                            await self.app._async_client._session.close()
                            self.log_info("Closed app._async_client session")
                except Exception as e:
                    self.log_warning(f"Error closing _async_client session: {e}")

        # Clean up utilities session if it exists
        if hasattr(self, '_cleanup_session'):
            try:
                await self._cleanup_session()
            except Exception as e:
                self.log_warning(f"Error cleaning up utilities session: {e}")

        # The socket-mode close that outran its 0.1s wait above. Everything after this call treats
        # ingress as stopped, so the one thing stop() must not do is return while that close is
        # still in progress. Bounded, and loud if the bound is hit: the caller's own callback drain
        # is what then catches whatever Bolt is still dispatching.
        close_task = getattr(self, "_handler_close_task", None)
        if close_task is not None:
            self._handler_close_task = None
            try:
                await asyncio.wait_for(asyncio.shield(close_task),
                                       timeout=self._HANDLER_CLOSE_TIMEOUT)
                self.log_debug("Socket mode handler closed")
            except asyncio.TimeoutError:
                # The task stays registered with the ingress barrier, deliberately: this bound
                # exists so a wedged close cannot hold stop() open, not so the caller can treat
                # ingress as down. The barrier is what refuses to call it quiet.
                self.log_error(
                    f"Socket mode handler close did not finish within "
                    f"{self._HANDLER_CLOSE_TIMEOUT:g}s — Slack may still be dispatching events")
            except Exception as e:  # noqa: BLE001
                self.log_warning(f"Socket mode handler close ended with: {e}")

    # Slack section-block text hard limit is 3000 chars; keep a margin so the reply text
    # fits one section when we attach the footer as blocks.
    _SECTION_TEXT_LIMIT = 2900

    # UX cap for the INLINE footer. Wrapping the reply in a section block to carry the
    # footer makes Slack collapse it behind a "Show more" link once it runs past ~4
    # wrapped lines in the narrow thread pane. Measured 2026-07-17 in #chatgpt-bot-test:
    # a section-block reply is fine at 220 chars and collapses at 300, while the SAME text
    # posted as plain text never collapses at any length. So the tidy inline footer only
    # rides short, few-line replies; anything longer posts as plain text and takes the
    # separate footer post, which renders in full. Set well below the 220/300 boundary so
    # it also survives the narrower right-hand thread flexpane and multi-line (list) replies.
    _FOOTER_INLINE_MAX = 180

    def _compose_reply_with_footer(self, formatted_text: str, footer_blocks: list):
        """Blocks that render the REPLY TEXT plus the footer actions row.

        Attaching action blocks alone makes Slack render blocks INSTEAD of the top-level
        `text`, hiding the answer (only the ⚙️ button shows). So the reply rides a leading
        section block, then the footer actions.

        The catch: that section block collapses behind Slack's "Show more" link once it's
        more than a few wrapped lines (see _FOOTER_INLINE_MAX); plain text never does. So
        we inline the footer ONLY for a short, few-line reply that won't collapse. Longer
        or taller replies return None — the caller then posts plain text and lets the
        separate footer post happen instead, which renders in full."""
        if not footer_blocks:
            return None
        if len(formatted_text) > self._FOOTER_INLINE_MAX or formatted_text.count("\n") > 2:
            return None
        return [{"type": "section", "text": {"type": "mrkdwn", "text": formatted_text}}] + list(footer_blocks)

    async def _record_receipt(self, channel_id: Optional[str], message_ts: Optional[str], *,
                              receipts: Any = None, receipt_kind: Optional[str] = None,
                              receipt_class: Optional[str],
                              thread_root_ts: Optional[str] = None, site: str = "") -> None:
        """Spec §5 intent contract for one durable post. Never raises into the send path.

        `receipt_class` (EDIT_OWN_MESSAGE §4) is a required keyword with no default: every
        posting site says what kind of surface it minted."""
        from message_processor.outbound_receipts import record_transport_post
        try:
            await record_transport_post(
                team_id=getattr(self, "self_team_id", None), channel_id=channel_id,
                message_ts=message_ts, receipts=receipts, receipt_kind=receipt_kind,
                receipt_class=receipt_class,
                thread_root_ts=thread_root_ts, site=site)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"receipt record failed at {site}: {e}")

    # --- the dev-only epoch fence's effect check ------------------------------------------
    #
    # SLIM SCOPE, stated once so the omissions are deliberate rather than forgotten. The check
    # guards the sites that MINT something in the room — a post, an upload, a reaction, a status
    # card, a footer, a native stream — and not the sites that update or delete what one of those
    # already created. Refusing the create is enough to keep a fenced turn's work out of the
    # channel, and fencing the follow-ups would strand a mid-flight turn's own cleanup. The spec's
    # full design fences all seventeen methods here and has an AST tripwire proving the set is
    # complete; slim has neither, so a posting site added later does NOT automatically inherit the
    # fence. The helpers are `_epoch_authorize` / `_epoch_refused` at module scope above.

    async def send_message(self, channel_id: str, thread_id: str, text: str,
                           blocks: Optional[list] = None,
                           meta_out: Optional[dict] = None,
                           username: Optional[str] = None,
                           lease: Any = None,
                           surface: str = "final_post",
                           receipts: Any = None,
                           receipt_kind: Optional[str] = None,
                           receipt_class: Optional[str] = None,
                           on_first_accept: Optional[Callable[[str], None]] = None
                           ) -> Optional[str]:
        """Send a text message to Slack, splitting if needed.

        `receipt_class` (EDIT_OWN_MESSAGE §4): the caller-stamped class for the receipt this
        post earns — a reply site stamps `assistant_reply`, a notice site `system_notice`, a
        job delivery `background_job`. Every channel producer passes it explicitly; the
        split-abort truncation notice below stamps its own `system_notice`.

        Returns the posted message ts (the FIRST chunk's ts when split), or None on
        failure. Truthy-on-success, so legacy `if await send_message(...)` callers keep
        working while F7 can key tool-use provenance on the returned ts.

        `on_first_accept(ts)` fires the instant Slack accepts the FIRST message or chunk, before
        the rest of a split runs, so a caller can record a surface the room can already see even
        if the send is then cancelled.

        WHAT WAS ACTUALLY DELIVERED rides `meta_out["delivery"]` (a `Delivery`): the text Slack
        accepted and whether all of it did. A split that aborts partway reports the delivered
        prefix with `complete=False` — the caller must commit that, never the text it asked for,
        or the room and the record disagree about what the bot said.

        `blocks` (F8): the settings-footer ACTIONS row. When provided AND the reply fits a
        single section block, the reply text + footer ride the message itself (composed via
        _compose_reply_with_footer) instead of a separate trailing post. When the text is
        too long for a section, or the message must split, the footer is NOT attached and
        the plain text posts — the caller's separate footer post covers it.

        `meta_out` (F8): optional dict the caller can read back — `meta_out["footer_attached"]`
        reports whether the footer ACTUALLY rode the message, so the caller sets its
        footer_attached flag from reality (a split/too-long reply must still get the
        separate footer)."""
        # §11.9: a receipted post says its class in the SAME call. A ledger with no class is a
        # programming error, refused loudly here — before Slack — rather than laundered into a
        # NULL (class-ineligible) row after the message is already in the room.
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "send_message: receipts passed without receipt_class (EDIT §4/§11.9)")

        def _set_attached(v: bool) -> None:
            if meta_out is not None:
                meta_out["footer_attached"] = v
        # STALE GUARD: the last thing before Slack. Raises StaleSendSuppressed rather than
        # posting, so a superseded answer is never created — and deliberately OUTSIDE the try
        # below, which turns exceptions into a None return the caller reads as "the send
        # failed". Nothing failed here; the turn was overtaken. Unguarded callers (notices,
        # welcome cards, a job's own status card) pass no lease and are unaffected.
        if lease is not None:
            lease.authorize(surface)
        # EPOCH FENCE: beside the stale guard and for the same reason — the last check before
        # Slack, and outside the try below, so a refusal is never laundered into "the send failed".
        _epoch_authorize(self, channel_id, f"send_message:{surface}")
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            # A bot OBJECT id can never be mentioned; never let one reach the channel.
            text = rewrite_bot_object_mentions(self, text)
            # Format text for Slack
            formatted_text = self.format_text(text)

            # Check if we need to split the message
            if len(formatted_text) <= self.MAX_MESSAGE_LENGTH:
                # Single message. Link previews follow ENABLE_LINK_PREVIEWS (default off:
                # links stay inline; no unfurl cards, and no Slack-unfurler "(edited)" marks).
                unfurl = bool(getattr(config, "enable_link_previews", False))
                post_kwargs = dict(channel=channel_id, thread_ts=thread_id, text=formatted_text,
                                   unfurl_links=unfurl, unfurl_media=unfurl)
                composed = self._compose_reply_with_footer(formatted_text, blocks) if blocks else None
                if composed is not None:
                    # text stays as the notification fallback; blocks carry the visible reply.
                    post_kwargs["blocks"] = composed
                # F30: optional username override (labelled research findings). Needs the
                # chat:write.customize scope; without it Slack raises missing_scope and this
                # whole send returns None, so the caller retries the plain path.
                if username:
                    post_kwargs["username"] = username
                # Wall-clock instant just before the post. Reconcile anchors its lower bound
                # here: a message that landed can only carry a Slack ts at or after we tried,
                # so an older own-message (an identical earlier reply) can never be mistaken
                # for this post. Captured on the happy path too — one cheap time.time() — but
                # only ever READ in the ambiguous branch below.
                attempt_start = time.time()
                try:
                    result = await self.app.client.chat_postMessage(**post_kwargs)  # unleased-ok: inside send_message, which authorized before this try
                except SlackApiError:
                    # Definitive API rejection — nothing landed. Let the outer handler return
                    # None so the caller's single retry is correct; no reconcile (there is no
                    # ambiguity to resolve, and this path must add zero extra work).
                    raise
                except Exception as transport_error:
                    # AMBIGUOUS transport failure (timeout / connection reset raised AFTER the
                    # request may have already reached Slack). chat.postMessage has no server-side
                    # idempotency key, so before letting the caller re-post — which would double
                    # the reply — reconcile against recent history: if our own message with this
                    # text is already there, the post landed and we return its ts as success. If
                    # it is NOT found (or the reconcile query itself fails) we re-raise unchanged
                    # so the caller's existing single retry still runs (a missing answer is worse
                    # than a rare duplicate).
                    reconciled_ts = await self._reconcile_uncertain_post(
                        channel_id, thread_id, formatted_text, attempt_start)
                    if not reconciled_ts:
                        # §11.21: mark the re-raise as a RECONCILIATION MISS — the request may
                        # have reached Slack, whatever the exception's type — so a caller that
                        # must classify the outcome (edit_own_message's announcement) can call
                        # it unknown rather than guessing from the type.
                        if meta_out is not None:
                            meta_out["transport_ambiguous"] = True
                        raise
                    self.log_warning(
                        f"Final post response timed out but the message landed "
                        f"(reconciled ts={reconciled_ts}); not re-posting: {transport_error}")
                    result = {"ts": reconciled_ts}
                    # The caller records this as its destination, and it is not an ordinary
                    # reply: nobody watched it land, the ts came back out of history. Reported
                    # so the ledger can tell a delivery we saw from one we reconstructed.
                    if meta_out is not None:
                        meta_out["reconciled"] = True
                posted_ts = result.get("ts")
                # First (and only) accepted surface — before the receipt write, which is the
                # first thing after this that can raise or be cancelled.
                _note_first_accept(on_first_accept, posted_ts)
                if posted_ts:
                    _report_delivery(meta_out, Delivery(
                        first_ts=posted_ts, text=formatted_text, complete=True,
                        parts_delivered=1, parts_total=1))
                await self._record_receipt(
                    channel_id, posted_ts, receipts=receipts, receipt_kind=receipt_kind,
                    receipt_class=receipt_class,
                    thread_root_ts=thread_id, site="send_message")
                # Report footer attachment only AFTER Slack returns a ts — a post that never
                # landed hasn't attached anything, and the separate footer must still fire.
                _set_attached(composed is not None and bool(posted_ts))
                # The reply is IN the room. Committing here (not at some later bookkeeping
                # step) is what stops a second visible piece of this same turn — a footer, an
                # artifact, a post_to_thread — being refused after the answer is already up.
                if lease is not None and posted_ts:
                    lease.commit()
                return posted_ts
            else:
                _set_attached(False)  # split replies never attach the footer
                # Split into multiple messages, "Continued..." style (shared markers so
                # the rebuild-side stripper always recognizes them). A failed chunk is
                # retried once (honoring a rate-limit Retry-After); if it STILL fails the
                # remainder is ABORTED — later chunks around a hole read worse than an
                # honest cut — and a loud truncation note posts in its place. Silent
                # partial delivery is never acceptable (Codex review find).
                chunks = self._split_message(formatted_text)
                last = len(chunks) - 1
                first_ts = None
                delivered_parts = 0
                truncated_at: Optional[int] = None
                for i, chunk in enumerate(chunks):
                    body = chunk
                    if i > 0:
                        body = f"{CONTINUATION_HEAD}\n\n{body}"
                    # No "Continued in next message..." trailer (user directive 2026-07-11):
                    # the "...continued" HEAD on the next chunk alone marks the seam, and the
                    # rebuild merger fires on EITHER marker (thread_management merge is OR).
                    unfurl = bool(getattr(config, "enable_link_previews", False))
                    chunk_kwargs = dict(channel=channel_id, thread_ts=thread_id, text=body,
                                        unfurl_links=unfurl, unfurl_media=unfurl)
                    if username:
                        chunk_kwargs["username"] = username  # F30: labelled findings
                    posted = False
                    for attempt in (1, 2):
                        try:
                            # REAUTHORIZE before every mutation while the lease is still
                            # pending. The retry below waits a second or more, which is ample
                            # time for a newer message to arrive — and the check that ran
                            # before the FIRST attempt says nothing about the second. Once a
                            # chunk lands the lease is committed and this short-circuits, so
                            # the rest of a split reply is never re-examined.
                            if lease is not None:
                                lease.authorize(surface)
                            result = await self.app.client.chat_postMessage(**chunk_kwargs)  # unleased-ok: reauthorized on the line above, every attempt
                            if first_ts is None:
                                first_ts = result.get("ts")
                                # The FIRST chunk landing is the moment the turn became
                                # visible. Waiting until the loop finished meant a suppression
                                # could still be raised against a reply already half in the
                                # room — and a half answer is worse than a late one. The
                                # caller's record is appended here for the same reason: from
                                # this instant the room can read us, whatever happens next.
                                if lease is not None and first_ts:
                                    lease.commit()
                                _note_first_accept(on_first_accept, first_ts)
                            delivered_parts = i + 1
                            # EVERY part earns its own receipt — the split is a delivery
                            # detail, and parts 2..N are as much of the answer as part 1.
                            # They finalize together when the turn settles.
                            await self._record_receipt(
                                channel_id, result.get("ts"), receipts=receipts,
                                receipt_kind=receipt_kind, receipt_class=receipt_class,
                                thread_root_ts=thread_id,
                                site="send_message_split")
                            posted = True
                            break
                        except SlackApiError as chunk_error:
                            self.log_error(
                                f"Error sending message chunk {i + 1}/{last + 1} "
                                f"(attempt {attempt}/2): {chunk_error}")
                            if attempt == 2:
                                break
                            # Honor Slack's Retry-After on 429s; brief pause otherwise.
                            delay = 1.0
                            try:
                                delay = float(cast(Any, getattr(chunk_error, "response", None))
                                              .headers.get("Retry-After", 1))
                            except Exception:
                                pass
                            await asyncio.sleep(min(max(delay, 0.5), 30.0))
                    if not posted:
                        missing = last + 1 - i
                        truncated_at = i + 1
                        self.log_error(
                            f"Aborting split post after chunk {i + 1}/{last + 1} failed twice — "
                            f"{missing} part(s) not delivered")
                        if meta_out is not None:
                            meta_out["split_truncated"] = True
                        # With ZERO chunks landed this notice would be the turn's first and
                        # only visible words, so it is guarded like any first surface. Once a
                        # chunk HAS landed the lease is committed and this permits — the reader
                        # is owed an explanation for the half message they can already see.
                        try:
                            if lease is not None:
                                lease.authorize(surface)
                            notice = await self.app.client.chat_postMessage(  # unleased-ok: reauthorized on the line above
                                channel=channel_id, thread_ts=thread_id,
                                text=(f"⚠️ This message was cut off — the remaining {missing} "
                                      f"part(s) failed to post to Slack."))
                            # A terminal explanation the room can read: conversation, not chrome.
                            await self._record_receipt(
                                channel_id, notice.get("ts") if notice else None,
                                receipts=receipts, receipt_kind="finalized",
                                receipt_class="system_notice",
                                thread_root_ts=thread_id, site="send_message_truncation")
                        except SlackApiError:
                            pass  # posting is broken; the ERROR log above stays loud
                        break
                if first_ts:
                    # The delivered text is the chunks that landed, joined as Slack holds them
                    # (continuation heads are chrome the rebuild strips, so they are not part of
                    # what was said). Whole or truncated, it is the same construction: reporting
                    # the caller's own text for a complete split described a delivery in markdown
                    # the room never received.
                    complete = truncated_at is None
                    delivered = "\n\n".join(chunks[:delivered_parts])
                    _report_delivery(meta_out, Delivery(
                        first_ts=first_ts, text=delivered, complete=complete,
                        parts_delivered=delivered_parts, parts_total=last + 1,
                        split=True, truncated_at=truncated_at))
                return first_ts
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return None

    # Ambiguous-commit reconcile window: an own message this recent that matches what we tried
    # to send is treated as the post that just landed. Slack timestamps are epoch seconds. The
    # primary lower bound is the attempt-start instant (below); this 120s value is a secondary
    # ceiling only.
    _RECONCILE_WINDOW_SECS = 120
    # Slack `ts` is server-stamped while attempt_start is local; allow a little drift so a post
    # that truly landed isn't rejected for a ts a hair before our local clock read.
    _RECONCILE_CLOCK_SKEW_SECS = 5
    # Prefix matching (Slack may append/trim chrome around the fallback text) is only safe for
    # long messages: below this length a short reply can be a prefix of an unrelated longer one.
    _RECONCILE_PREFIX_MIN_LEN = 200
    # conversations.replies returns the EARLIEST in-window messages first, so the freshly-posted
    # tail can sit past the first page. Follow the cursor up to this many pages before giving up.
    _RECONCILE_MAX_PAGES = 3

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Whitespace/entity-normalized text for comparing what we SENT against what Slack
        STORED. Slack collapses runs of whitespace and HTML-escapes &/</>, so undo both before
        comparing so a benign normalization can't defeat the match."""
        text = (text or "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return " ".join(text.split())

    async def _reconcile_uncertain_post(self, channel_id: str, thread_id: Optional[str],
                                        formatted_text: str, attempt_start: float) -> Optional[str]:
        """After an AMBIGUOUS transport failure on a single-message post, look for the message we
        just tried to send in recent history. Returns its ts when our OWN bot posted matching
        text at or after `attempt_start` (the instant just before this post), else None (the
        caller then reports failure so its single retry runs).

        `formatted_text` is the exact post-conversion payload text handed to chat.postMessage.
        `attempt_start` is a local wall-clock timestamp captured immediately before the post; a
        message that actually landed can only carry a Slack ts at/after it (minus a small skew
        for clock drift), so an identical EARLIER reply can never be mistaken for this one.

        Best-effort: any error querying history returns None — a missing answer is worse than a
        rare duplicate. Only the ambiguous branch calls this, so the happy path pays nothing."""
        target = self._normalize_for_match(formatted_text)
        if not target:
            return None
        # Lower bound anchored to the attempt (minus drift skew). The 120s window is a secondary
        # ceiling: whichever bound is more recent wins, and the attempt-anchored floor normally
        # does, so a stale identical reply is excluded outright.
        floor_ts = attempt_start - self._RECONCILE_CLOCK_SKEW_SECS
        cutoff = max(floor_ts, time.time() - self._RECONCILE_WINDOW_SECS)
        oldest = f"{floor_ts:.6f}"
        min_len = self._RECONCILE_PREFIX_MIN_LEN
        # `oldest` + `inclusive` scope the query to the attempt window. conversations.replies
        # returns the EARLIEST in-window messages first, so the freshly-posted tail can sit past
        # the first page when >100 replies fall in-window — page the cursor (bounded to
        # _RECONCILE_MAX_PAGES) and scan EVERY page. conversations.history returns newest-first so
        # it matches on page 1 in practice, but the loop shape is shared. Any query error anywhere
        # returns None — a missing answer is worse than a rare duplicate.
        cursor: Optional[str] = None
        for _page in range(self._RECONCILE_MAX_PAGES):
            try:
                if thread_id:
                    resp = await self.app.client.conversations_replies(
                        channel=channel_id, ts=thread_id, oldest=oldest, inclusive=True,
                        limit=100, cursor=cursor)
                else:
                    resp = await self.app.client.conversations_history(
                        channel=channel_id, oldest=oldest, inclusive=True,
                        limit=100, cursor=cursor)
                messages = resp.get("messages", []) if resp else []
            except Exception as e:
                self.log_warning(f"Reconcile query failed after uncertain post: {e}")
                return None
            for msg in messages:
                if not self.is_own_message(msg):
                    continue
                try:
                    if float(msg.get("ts", 0)) < cutoff:
                        continue
                except (TypeError, ValueError):
                    continue
                candidate = self._normalize_for_match(msg.get("text", ""))
                if not candidate:
                    continue
                if candidate == target:
                    return msg.get("ts")
                # Full-prefix match (Slack may append/trim chrome around the fallback text) only
                # when BOTH normalized strings are long — otherwise a short reply is a prefix of an
                # unrelated longer one ("OK" vs "OK, done") and a genuinely lost new post gets
                # swallowed. Compare the WHOLE shorter string, never a 200-char head: two long
                # replies sharing a 200-char boilerplate prefix then diverging must NOT match.
                if (len(candidate) >= min_len and len(target) >= min_len
                        and (candidate.startswith(target) or target.startswith(candidate))):
                    return msg.get("ts")
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") if resp else None
            if not cursor:
                break
        return None

    def _split_message(self, text: str) -> List[str]:
        """Split a long message into chunks that fit within Slack's limit.

        Fence-aware (code blocks are closed and reopened across the seam) and
        entity-safe; oversized single fragments are hard-wrapped instead of
        being sent as-is (which used to msg_too_long and abort the remainder).
        Margin covers the continuation markers + fence reopen prefix.
        """
        return fence_safe_chunks(text, self.MAX_MESSAGE_LENGTH - 150)

    async def send_message_get_ts(self, channel_id: str, thread_id: str, text: str,
                                  lease: Any = None,
                                  surface: str = "legacy_seed",
                                  receipts: Any = None,
                                  receipt_kind: Optional[str] = None,
                                  receipt_class: Optional[str] = None) -> Dict:
        """Send a message and return the response including timestamp.

        `lease` (stale guard): the legacy streaming path seeds its reply with this call, so it
        is a first answer surface and is checked like one."""
        # §11.9: receipts-without-class is a programming error — see send_message.
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "send_message_get_ts: receipts passed without receipt_class (EDIT §4/§11.9)")
        if lease is not None:
            lease.authorize(surface)
        _epoch_authorize(self, channel_id, f"send_message_get_ts:{surface}")
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            text = rewrite_bot_object_mentions(self, text)
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            # Safety check - this should never happen for continuation messages
            # but if somehow the text is too long, truncate it
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 80] + "\n\n*[Message exceeded Slack limit]*"
            
            result = await self.app.client.chat_postMessage(  # unleased-ok: inside send_message_get_ts, which authorized at its entry
                channel=channel_id,
                thread_ts=thread_id,
                text=formatted_text
            )
            
            await self._record_receipt(
                channel_id, result.get("ts"), receipts=receipts, receipt_kind=receipt_kind,
                receipt_class=receipt_class,
                thread_root_ts=thread_id, site="send_message_get_ts")
            if lease is not None:
                lease.commit()
            return {"success": True, "ts": result["ts"]}
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}

    async def _record_pending_share(self, channel_id: Optional[str], file_id: Optional[str],
                                    receipts: Any, thread_id: Optional[str],
                                    site: str, resolve_share: bool = False, *,
                                    receipt_class: Optional[str]) -> None:
        """Spec §5: an upload has no message ts, so the file id is the only handle we can
        record. Written the instant Slack returns it, before resolution starts — the crash
        window in between is accepted (Slack supplies no id before the upload).

        `resolve_share` starts the share-ts poll that finalizes the receipt. Image delivery
        passes False and runs its own (one poll there feeds the upload indicator and provenance
        as well); every other upload has nobody else asking the question, and without the poll
        the file stays pending — outside the stream — until a restart recovers it.
        """
        if receipts is None or not file_id:
            return
        from message_processor.outbound_receipts import (record_pending_share,
                                                         schedule_share_resolution)
        try:
            # Spec §4/§11.13: every image/file share is class `artifact` — stamped by the
            # PRODUCER on the transport call, carried end-to-end into the resolved receipt.
            await record_pending_share(
                getattr(self, "db", None), team_id=getattr(self, "self_team_id", None),
                channel_id=channel_id, file_id=file_id,
                owner_turn_id=getattr(receipts, "owner_id", ""), thread_root_ts=thread_id,
                receipt_class=receipt_class)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"pending share record failed at {site}: {e}")
        if not resolve_share:
            return
        try:
            schedule_share_resolution(
                self, getattr(self, "db", None),
                team_id=getattr(self, "self_team_id", None), channel_id=channel_id,
                file_id=file_id, site=site)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"share resolution not scheduled at {site}: {e}")

    async def ensure_receipt_identity(self) -> bool:
        """Whether bot + team identity are known — the two facts every channel receipt needs.

        Re-runs auth.test when either is missing. `_ensure_self_identity` returns early once a
        user id is cached, so a boot that resolved the user but not the team could never repair
        itself through it. A channel turn without a team id would write no receipts at all and
        say nothing about it, which is why the caller refuses to run one.
        """
        if getattr(self, "self_team_id", None) and getattr(self, "bot_user_id", None):
            return True
        try:
            resp = await self.app.client.auth_test()
            if resp.get("ok"):
                self.bot_user_id = resp.get("user_id") or getattr(self, "bot_user_id", None)
                self.bot_id = resp.get("bot_id") or getattr(self, "bot_id", None)
                self.bot_handle = resp.get("user") or getattr(self, "bot_handle", None)
                self.self_team_id = resp.get("team_id") or getattr(self, "self_team_id", None)
            else:
                self.log_warning(f"auth_test returned not ok: {resp.get('error')}")
        except Exception as e:  # noqa: BLE001 — reported by the caller's refusal
            self.log_warning(f"Could not re-resolve bot identity: {e}")
        return bool(getattr(self, "self_team_id", None) and getattr(self, "bot_user_id", None))

    async def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str,
                         caption: str = "", meta_out: Optional[dict] = None,
                         receipts: Any = None, *,
                         receipt_class: Optional[str]) -> Optional[str]:
        """Send an image to Slack and return the file URL.

        `meta_out` (F7): optional dict the caller reads back — `meta_out["file_id"]` is the
        uploaded file's id, set only once Slack accepts the upload. The RETURN stays the bare
        URL (base_client / slack_client.base declare that contract), so the file id rides a
        side channel rather than breaking every existing caller. It exists because
        files_upload_v2 hands back no share ts: the file id is the only handle from which the
        image message's ts can later be resolved (see resolve_file_share_ts).

        `receipt_class` (§11.9/§11.13) is REQUIRED — producers pass `artifact` (the §4
        inventory: an image share is job output, never hardcoded in the transport), and
        receipts-without-class raises BEFORE Slack.
        """
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "send_image: receipts passed without receipt_class (EDIT §4/§11.9)")
        # BEFORE the upload, never after: an upload is irreversible and visible, so authorizing
        # it afterwards fences nothing.
        _epoch_authorize(self, channel_id, "send_image")
        try:
            # Use files_upload_v2 for image upload
            result = await self.app.client.files_upload_v2(
                channel=channel_id,  # Changed from channels to channel (singular)
                thread_ts=thread_id,
                file=image_data,
                filename=filename,
                initial_comment=caption
            )

            # Extract the file URL from the response
            if result and "files" in result and len(result["files"]) > 0:
                file_info = result["files"][0]
                file_url = file_info.get("url_private", file_info.get("permalink"))
                if meta_out is not None:
                    meta_out["file_id"] = file_info.get("id")
                await self._record_pending_share(
                    channel_id, file_info.get("id"), receipts, thread_id, "send_image",
                    receipt_class=receipt_class)
                self.log_info(f"Image uploaded: {filename} - URL: {file_url}")
                return file_url
            else:
                self.log_warning("Image uploaded but no URL found in response")
                return None

        except SlackApiError as e:
            self.log_error(f"Error uploading image: {e}")
            return None

    # Poll schedule for resolve_file_share_ts, in seconds. Mild backoff rather than a fixed
    # tight interval: the share lands ~1.8s (channel) to ~3.8s (DM) after upload, so a 0.4s
    # loop would burn ~10 calls on a DM to learn nothing the 5th poll wouldn't have.
    _SHARE_TS_BACKOFF = (0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0)

    # Slack errors that will not come right inside the budget, so polling on until the
    # deadline is pure waste. Everything ELSE is worth another poll while budget remains —
    # including `file_not_found`, which here is the upload's own eventual consistency (the
    # very race this poll exists to paper over), not a verdict that the file isn't real.
    _SHARE_TS_PERMANENT_ERRORS = frozenset({
        "invalid_auth", "not_authed", "missing_scope", "access_denied"})

    @staticmethod
    def _share_ts_error_code(error: SlackApiError) -> str:
        """Slack's machine-readable `error` string, or "" when the shape isn't what we expect
        (an unrecognized code is treated as transient, which is the safe default here)."""
        response = getattr(error, "response", None)
        if response is None:
            return ""
        try:
            return str(response.get("error") or "")
        except (AttributeError, TypeError):
            return ""

    @staticmethod
    def _share_ts_retry_after(error: SlackApiError) -> Optional[float]:
        """Seconds Slack asked us to wait on a 429, if it said. Header lookup is
        case-insensitive because the casing varies with the transport underneath the SDK."""
        headers = getattr(getattr(error, "response", None), "headers", None)
        if not headers:
            return None
        try:
            for key, value in headers.items():
                if str(key).lower() == "retry-after":
                    return max(0.0, float(value))
        except (AttributeError, TypeError, ValueError):
            return None
        return None

    async def resolve_file_share_ts(self, channel_id: str, file_id: str) -> Optional[str]:
        """The ts of the message that shares an uploaded file, or None.

        Why this exists: files_upload_v2's response DOES carry a `shares` key, and it is
        always `{}` at upload time — Slack populates it asynchronously, so the share ts is
        only readable from a later files.info call. Measured live 2026-07-16: the entry
        appeared ~1.76s after upload in a private channel and ~3.81s in a DM (DMs are
        markedly slower).

        The entry sits at `shares["private"][channel_id][0]` for private channels AND DMs;
        public channels use `shares["public"][channel_id][0]`. Both scopes are checked — the
        caller has no way to know which applies. That entry's `ts` IS the file-share
        message's ts (cross-checked against conversations.replies / conversations.history).

        The retry contract is a TIME BUDGET, not an attempt count: polling continues on the
        backoff below until `IMAGE_SHARE_TS_TIMEOUT_SECONDS` (config.image_share_ts_timeout_
        seconds) is spent. A transient failure is retried inside it rather than surrendering the
        row — a 429 or a blip is not an answer, and giving up on the first one throws away
        provenance a second poll would have had, while a fixed "3 attempts" would turn a slow
        day into lost provenance. Only clearly permanent errors bail early.

        Best-effort chrome: a timeout, a SlackApiError, or any other failure returns None and
        is logged, never raised. The image is already posted by the time anyone calls this,
        and provenance must never be able to touch it.
        """
        if not channel_id or not file_id:
            return None
        deadline = time.monotonic() + max(0.0, float(config.image_share_ts_timeout_seconds))
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            # Budget is checked BEFORE the request, not after: waking exactly AT the deadline
            # and polling once more is how a "15s bound" quietly becomes 15s plus a request.
            # Attempt 0 is the deliberate exception — the share is often already there, so the
            # first poll is always worth making even on an exhausted budget.
            if attempt and remaining <= 0:
                self.log_debug(f"share ts for {file_id} did not appear before the timeout")
                return None

            retry_after: Optional[float] = None
            try:
                # The budget bounds the CALL too, not just the gaps between calls, or one hung
                # request sails past the deadline on its own (the SDK's default timeout being
                # the only other ceiling). The guaranteed first poll has no budget left to be
                # bounded by, so it keeps that SDK default.
                result = await asyncio.wait_for(
                    self.app.client.files_info(file=file_id),
                    timeout=remaining if remaining > 0 else None)
                shares = ((result or {}).get("file") or {}).get("shares") or {}
                for scope in ("public", "private"):
                    entries = (shares.get(scope) or {}).get(channel_id) or []
                    if entries and entries[0].get("ts"):
                        return entries[0]["ts"]
            except SlackApiError as e:
                if self._share_ts_error_code(e) in self._SHARE_TS_PERMANENT_ERRORS:
                    self.log_debug(f"files.info share-ts lookup gave up for {file_id}: {e}")
                    return None
                retry_after = self._share_ts_retry_after(e)
                self.log_debug(f"files.info share-ts lookup will retry for {file_id}: {e}")
            except Exception as e:  # noqa: BLE001 — never load-bearing; see docstring
                # Transport blips and our own call timeout above: transient like a 429, so
                # they buy another poll rather than costing the row.
                self.log_debug(f"share-ts resolve error for {file_id}: {e}")

            delay = self._SHARE_TS_BACKOFF[min(attempt, len(self._SHARE_TS_BACKOFF) - 1)]
            if retry_after is not None:
                # Slack said when to come back; polling sooner just earns another 429.
                delay = max(delay, retry_after)
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(delay, remaining))

    async def send_file(self, channel_id: str, thread_id: str, file_data,
                        filename: str, title: Optional[str] = None,
                        initial_comment: str = "",
                        receipts: Any = None, *,
                        receipt_class: Optional[str]) -> Optional[Dict[str, Any]]:
        """F32: upload an arbitrary file (BytesIO) and return its full Slack identity.

        Distinct from send_image, which returns a bare URL: an artifact has to be findable
        again, so callers get {"file_id", "url_private", "permalink"} to persist. The
        file_id is what `read_document` looks up, so without it the model could never
        re-read its own artifact.

        Returns None on any failure — the caller decides whether that's fatal (for an
        artifact it never is: the text answer already landed). `receipt_class`
        (§11.9/§11.13) is REQUIRED — producers pass `artifact` (the §4 inventory, never
        hardcoded in the transport), and receipts-without-class raises BEFORE Slack.
        """
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "send_file: receipts passed without receipt_class (EDIT §4/§11.9)")
        _epoch_authorize(self, channel_id, "send_file")
        try:
            result = await self.app.client.files_upload_v2(
                channel=channel_id,
                thread_ts=thread_id,
                file=file_data,
                filename=filename,
                title=title or filename,
                initial_comment=initial_comment or "",
            )
            files = (result or {}).get("files") or []
            if not files:
                self.log_warning(f"File uploaded but no file info returned: {filename}")
                return None
            info = files[0]
            file_id = info.get("id")
            url = info.get("url_private") or info.get("permalink")
            if not file_id or not url:
                # Without an id we could never find this file again, and the caller would
                # persist a ref that points at nothing. A response we can't use is not success.
                self.log_warning(
                    f"File upload returned no usable identity (id={file_id!r}): {filename}")
                return None
            identity = {"file_id": file_id, "url_private": url,
                        "permalink": info.get("permalink")}
            await self._record_pending_share(
                channel_id, file_id, receipts, thread_id, "send_file", resolve_share=True,
                receipt_class=receipt_class)
            self.log_info(f"File uploaded: {filename} (id={file_id})")
            return identity
        except SlackApiError as e:
            self.log_error(f"Error uploading file '{filename}': {e}")
            return None
        except Exception as e:  # noqa: BLE001 — an upload problem must never break the turn
            self.log_error(f"Unexpected error uploading file '{filename}': {e}")
            return None

    async def send_thinking_indicator(self, channel_id: str, thread_id: str,
                                      receipts: Any = None, *,
                                      receipt_class: Optional[str]) -> Optional[str]:
        """Show a progress indicator; returns the placeholder message ts, or None.

        `receipt_class` (§11.9/§11.13) is REQUIRED — producers pass `chrome` (the §4
        inventory: a placeholder is excluded scaffolding until promotion, never hardcoded in
        the transport), and receipts-without-class raises BEFORE Slack.

        Contract: assistant.threads.setStatus is the SOLE indicator wherever Slack
        accepts it — the June-2026 agent surface renders the composer status in DMs
        AND channel threads (verified live 2026-07-09: a channel thread showed both
        the status line and our redundant placeholder). Native status means no
        message, no "(edited)" churn, auto-clears on reply. Returns None; downstream
        consumers treat a None ts as "status-only" (streaming seeds its own message
        lazily, phase updates route to setStatus, deletes no-op).

        Only where setStatus FAILS (non-agent contexts, older surfaces) do we post
        the classic "Thinking..." placeholder and return its ts.
        """
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "send_thinking_indicator: receipts passed without receipt_class "
                "(EDIT §4/§11.9)")
        if _epoch_refused(self, channel_id, "send_thinking_indicator"):
            return None
        status_set = await self.set_assistant_status(channel_id, thread_id)
        if status_set:
            return None
        try:
            result = await self.app.client.chat_postMessage(  # unleased-ok: send_thinking_indicator — a placeholder is chrome, never answer text
                channel=channel_id,
                thread_ts=thread_id,
                text=f"{config.circle_loader_emoji} {config.random_loading_message()}"
            )
            # Chrome until the turn writes its answer INTO it — the streaming handlers promote
            # it (same owner) right before the first answer-bearing edit, and the promotion
            # atomically maps the class chrome → assistant_reply (spec §4). The class is the
            # caller's stamp (§11.13 — producers pass chrome).
            await self._record_receipt(
                channel_id, result.get("ts"), receipts=receipts, receipt_kind="chrome",
                receipt_class=receipt_class,
                thread_root_ts=thread_id, site="send_thinking_indicator")
            return result.get("ts")  # Return message timestamp for deletion
        except SlackApiError as e:
            self.log_error(f"Error sending thinking indicator: {e}")
            return None

    async def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message from Slack"""
        try:
            await self.app.client.chat_delete(  # unleased-ok: teardown — removing a surface can never be a stale answer
                channel=channel_id,
                ts=message_id
            )
            return True
        except SlackApiError as e:
            self.log_debug(f"Could not delete message: {e}")
            return False

    async def update_message(self, channel_id: str, message_id: str, text: str,
                             lease: Any = None,
                             surface: str = "error_notice",
                             receipts: Any = None,
                             receipt_kind: Optional[str] = None,
                             receipt_class: Optional[str] = None) -> bool:
        """Update a message in Slack.

        `lease` (stale guard): passed by the callers whose edit publishes TERMINAL text — an
        error or timeout notice written into the thinking placeholder. That notice is the turn's
        answer as far as the room is concerned, so when the conversation has already moved on it
        must not be written: the suppression terminal is the honest record, and "something went
        wrong" would be a claim about a turn where nothing did. Chrome callers pass nothing and
        are unaffected."""
        # §11.9/§11.13: a receipted edit says its class in the SAME call — refused loudly
        # BEFORE Slack, never laundered into a class-less (NULL) row after the write landed.
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "update_message: receipts passed without receipt_class (EDIT §4/§11.9)")
        if lease is not None:
            lease.authorize(surface)
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            text = rewrite_bot_object_mentions(self, text)
            await self.app.client.chat_update(  # unleased-ok: inside update_message, which authorizes at its entry when a caller supplies a lease
                channel=channel_id,
                ts=message_id,
                text=text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            # An EDIT mints no ts, so there is nothing to register — only a state change, and
            # only when the caller names one (a terminal notice written into a chrome surface
            # is conversation from that moment on).
            if receipts is not None and receipt_kind:
                await self._record_receipt(
                    channel_id, message_id, receipts=receipts, receipt_kind=receipt_kind,
                    receipt_class=receipt_class, site="update_message")
            # A terminal notice that LANDED is what the room saw, so this turn has spoken —
            # exactly like every other transport. Without it a leased error notice stayed
            # `pending`, and any later visible piece of the same turn could still be refused
            # after the room had already read something.
            if lease is not None:
                lease.commit()
            return True
        except SlackApiError as e:
            self.log_error(f"Could not update message: {e}")
            return False

    async def post_status_card(self, channel_id: str, thread_id: str, text: str,
                               blocks: list, username: Optional[str] = None,
                               receipts: Any = None, *,
                               receipt_class: Optional[str]) -> Optional[str]:
        """F30.1: post a blocks status card (e.g. the deep-research todo card). Returns the
        posted ts, or None on failure. `text` is the CONSTANT notification fallback; `blocks`
        carry the visible card. `username` optionally labels the poster (needs the
        chat:write.customize scope; without it Slack raises missing_scope → None so the caller
        can retry unlabeled). Best-effort — never raises, EXCEPT the §11.9/§11.13 contract:
        `receipt_class` is REQUIRED (its producers pass `background_job` — the §4 inventory,
        never hardcoded in the transport), and receipts-without-class is a programming error
        raised BEFORE Slack."""
        if receipts is not None and receipt_class is None:
            raise ValueError(
                "post_status_card: receipts passed without receipt_class (EDIT §4/§11.9)")
        if _epoch_refused(self, channel_id, "post_status_card"):
            return None
        try:
            kwargs = dict(channel=channel_id, thread_ts=thread_id, text=text, blocks=blocks,
                          unfurl_links=False, unfurl_media=False)
            if username:
                kwargs["username"] = username
            result = await self.app.client.chat_postMessage(**kwargs)  # unleased-ok: post_ephemeral-style helper for chrome/notices, not the turn's answer
            # A background job's own status surface: chrome STATE (never in the stream),
            # class per the caller's §4 stamp (research status/checklist ⇒ background_job).
            await self._record_receipt(
                channel_id, result.get("ts"), receipts=receipts, receipt_kind="chrome",
                receipt_class=receipt_class,
                thread_root_ts=thread_id, site="post_status_card")
            return result.get("ts")
        except SlackApiError as e:
            self.log_warning(f"Status card post failed: {e}")
            return None

    async def update_status_card(self, channel_id: str, ts: str, text: str,
                                 blocks: list, receipts: Any = None) -> bool:
        """F30.1: update a blocks status card in place. `text` MUST stay CONSTANT across
        updates (Slack badges '(edited)' only when the top-level text changes; blocks-only
        edits don't badge). Best-effort — returns False on failure, never raises.

        `receipts` is accepted for symmetry with post_status_card and deliberately unused: an
        edit mints no ts, and the card's chrome row was written when it was posted."""
        try:
            await self.app.client.chat_update(  # unleased-ok: a background job's own status card — a detached surface the guard exempts
                channel=channel_id, ts=ts, text=text, blocks=blocks)
            return True
        except SlackApiError as e:
            self.log_debug(f"Status card update failed: {e}")
            return False

    async def _replies_page_with_retry(self, kwargs: Dict, attempts: int = 3):
        """One conversations.replies page, honoring Retry-After on 429s (R1).

        Retries only rate-limit errors; anything else propagates immediately.
        """
        for attempt in range(attempts):
            try:
                return await self.app.client.conversations_replies(**kwargs)
            except SlackApiError as e:
                err = e.response.get("error") if getattr(e, "response", None) else None
                status = getattr(getattr(e, "response", None), "status_code", None)
                if (err == "ratelimited" or status == 429) and attempt < attempts - 1:
                    headers = getattr(getattr(e, "response", None), "headers", None) or {}
                    try:
                        delay = float(headers.get("Retry-After") or 1)
                    except (TypeError, ValueError):
                        delay = 1.0
                    delay = min(max(delay, 0.5), 30.0)
                    self.log_warning(
                        f"conversations.replies rate-limited (attempt {attempt + 1}/{attempts}), "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    async def get_thread_history(self, channel_id: str, thread_id: str,
                                 limit: Optional[int] = None,
                                 oldest: Optional[str] = None) -> List[Message]:
        """Get COMPLETE thread history from Slack - fetches ALL messages by default.

        `oldest` (Slack ts) fetches only messages strictly after it (Slack's default
        inclusive=false) — used by the Phase S summary-tail rebuild so a compacted
        1500-message thread doesn't refetch the summarized head.

        Raises HistoryFetchError on terminal fetch failure. Since Phase S, Slack is
        the ONLY transcript — returning [] here would make the bot answer with
        amnesia, silently. An actually-empty thread still returns [] normally.
        """
        messages = []

        try:
            from slack_client.formatting.text import extract_mention_ids
            # Fetch page by page; each page is turned into PROVISIONAL Messages immediately and
            # the raw page dropped, so raw Slack payloads (blocks/files/supplementary) never
            # accumulate for the whole thread (Blocker 3). Text is left UNCLEANED here — mentions
            # are cleaned in a final pass, after ONE batched read-only username resolve.
            provisional = []
            cursor = None
            total_fetched = 0

            while True:
                # Slack's max per request is 1000
                per_request_limit = 1000
                if limit and limit - total_fetched < 1000:
                    per_request_limit = limit - total_fetched

                kwargs = {
                    "channel": channel_id,
                    "ts": thread_id,
                    "limit": per_request_limit
                }
                if oldest:
                    kwargs["oldest"] = oldest
                if cursor:
                    kwargs["cursor"] = cursor

                result = await self._replies_page_with_retry(kwargs)
                slack_messages = result.get("messages", [])

                if not slack_messages:
                    break

                for msg in slack_messages:
                    text = msg.get("text", "")
                    sender_type = self.classify_sender(msg)
                    # Skip our OWN transient chrome and pure UI-helper blocks (same rule as before).
                    if sender_type == "self" and is_self_chrome_message(text, msg):
                        continue
                    if sender_type != "self" and _is_ui_helper_message(msg):
                        continue
                    is_bot = bool(msg.get("bot_id"))
                    # Display name for bot authors (used to name-prefix other bots like humans)
                    bot_name = msg.get("username") or (msg.get("bot_profile") or {}).get("name")
                    # Combine supplementary BEFORE cleaning, matching the live path, so a mention
                    # living only in a quoted/table block resolves too. Cleaning is DEFERRED.
                    if sender_type != "self":
                        supplementary = extract_supplementary_text(msg, primary_text=text)
                        if supplementary:
                            text = f"{text}\n\n{supplementary}" if text.strip() else supplementary
                    attachments = []
                    for file in msg.get("files", []):
                        mimetype = file.get("mimetype", "")
                        file_type = "image" if mimetype.startswith("image/") else "file"
                        attachments.append({
                            "type": file_type,
                            "name": file.get("name"),
                            "mimetype": mimetype,
                            "url": file.get("url_private", file.get("permalink")),
                            # Match the live path (message_events) so a rebuilt Message carries the
                            # same provenance: the file id and declared size enable later
                            # re-download / size-gate decisions.
                            "id": file.get("id"),
                            "size": file.get("size"),
                        })
                    author_id = msg.get("user", "bot" if is_bot else "unknown")
                    provisional.append(Message(
                        text=text,  # UNCLEANED; mentions cleaned in the final pass below
                        user_id=author_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        attachments=attachments,
                        metadata={
                            "ts": msg.get("ts"),
                            "is_bot": is_bot,
                            "sender_type": sender_type,
                            "bot_name": bot_name,
                            # "username" is filled after the batch resolve; the KEY is ALWAYS set.
                            # Raw reactions from conversations.replies (name/users/count) —
                            # rendered into rebuilt context as a deterministic annotation
                            "reactions": msg.get("reactions") or None
                        }
                    ))

                total_fetched += len(slack_messages)

                # Check if we've hit our limit
                if limit and total_fetched >= limit:
                    break

                # Check for pagination
                next_cursor = (result.get("response_metadata") or {}).get("next_cursor")
                if not next_cursor:
                    break
                cursor = next_cursor

            # Blocker 2: build an ORDERED, deduped id list so the remote budget resolves the SAME
            # users across cold starts (set iteration is hash-seed dependent, which would change
            # rebuilt serialization). Priority when the budget bites on a long thread: the root
            # author, then authors NEWEST→OLDEST, then mention ids in the same newest→oldest scan
            # (recent speakers matter most). resolve_usernames preserves this input order.
            api_client = getattr(getattr(self, "app", None), "client", None)
            self_id = getattr(self, "bot_user_id", None)
            ordered_ids = []
            seen = set()

            def _add(uid):
                if uid and uid not in ("bot", "unknown") and uid != self_id and uid not in seen:
                    seen.add(uid)
                    ordered_ids.append(uid)

            if provisional and provisional[0].metadata.get("sender_type") not in ("self", "other_bot"):
                _add(provisional[0].user_id)                       # root author first
            for m in reversed(provisional):                        # then authors newest→oldest
                if m.metadata.get("sender_type") not in ("self", "other_bot"):
                    _add(m.user_id)
            for m in reversed(provisional):                        # then mentions newest→oldest
                if not m.metadata.get("is_bot"):
                    for uid in extract_mention_ids(m.text):
                        _add(uid)

            name_map = {}
            if ordered_ids:
                try:
                    name_map = await self.resolve_usernames(ordered_ids, api_client)
                except Exception as e:
                    self.log_debug(f"batch username resolution failed: {e}")

            # Final pass: clean mentions against the now-warmed cache and stamp the author name.
            # The "username" KEY is ALWAYS set (None when unresolved or not a human author) so the
            # rebuild consumer treats its presence as proof this batch already ran (Blocker 1).
            for m in provisional:
                if not m.metadata.get("is_bot"):
                    m.text = self._clean_mentions(m.text)
                if (m.metadata.get("sender_type") not in ("self", "other_bot")
                        and m.user_id not in ("bot", "unknown")):
                    m.metadata["username"] = name_map.get(m.user_id)
                else:
                    m.metadata["username"] = None
                messages.append(m)

            self.log_info(f"Fetched {len(messages)} messages from thread {thread_id}")
            return messages
            
        except SlackApiError as e:
            # Terminal failure (rate-limit retries exhausted or a hard API error).
            # Do NOT return [] — Slack is the only transcript, and an empty result
            # here would silently rebuild the thread with no context (R1).
            self.log_error(f"Error getting thread history: {e}")
            raise HistoryFetchError(
                f"Could not fetch thread history for {channel_id}:{thread_id}: {e}"
            ) from e

    def supports_streaming(self) -> bool:
        """Returns True if streaming is enabled for Slack"""
        return config.enable_streaming and config.slack_streaming

    def get_streaming_config(self) -> Dict:
        """Returns platform-specific streaming configuration"""
        return {
            "update_interval": config.streaming_update_interval,
            "min_interval": config.streaming_min_interval,
            "max_interval": config.streaming_max_interval,
            "buffer_size": config.streaming_buffer_size,
            "circuit_breaker_threshold": config.streaming_circuit_breaker_threshold,
            "circuit_breaker_cooldown": config.streaming_circuit_breaker_cooldown,
            "platform": "slack"
        }

    def supports_native_streaming(self) -> bool:
        """True if native Slack streaming (chat.startStream/appendStream/stopStream) is enabled
        and available on the SDK. Default OFF via config pending live dev-bot verification."""
        return (
            config.slack_native_streaming
            and self.supports_streaming()
            and hasattr(self.app.client, "chat_startStream")
        )

    def begin_native_stream(self, channel_id: str, thread_id: str,
                            user_id: Optional[str] = None) -> "NativeStreamSession":
        """Create a (not-yet-started) NativeStreamSession bound to this channel/thread.

        chat.startStream requires recipient_team_id (workspace, resolved once via auth.test
        in _ensure_self_identity) AND recipient_user_id (the triggering user, plumbed in by
        the handler) for channel streaming — both are threaded onto the session here."""
        return NativeStreamSession(
            self.app.client, channel_id, thread_id, logger=self.log_debug,
            team_id=getattr(self, "self_team_id", None), user_id=user_id, owner=self)

    async def set_assistant_status(self, channel_id: str, thread_id: str,
                                   status: Optional[str] = None,
                                   loading_messages: Optional[List[str]] = None) -> bool:
        """Best-effort assistant.threads.setStatus (Phase 3.2).

        Shows a transient 'thinking/working' status on the assistant-thread surface with a
        rotating branded loading_messages set; auto-clears when the app replies. Degrades to a
        silent no-op in plain channels / non-assistant contexts — must never raise.

        GUARD (Phase G / agent_view): on the June-2026 surface setStatus AUTO-OPENS the
        thread for the user. Never call this speculatively for a channel message the
        participation engine might still ignore — the only caller is
        send_thinking_indicator, which main.py invokes strictly AFTER the engine's
        'respond' verdict. Keep it that way (regression-tested in test_phase_g.py).
        """
        if not config.enable_assistant_status:
            return False
        if not hasattr(self.app.client, "assistant_threads_setStatus"):
            return False
        # Slack API contract (verified live 2026-07-10): a NON-EMPTY `status`
        # string is what renders — status:"" is the CLEAR signal and hides the
        # indicator entirely, loading_messages never render without a status,
        # and an empty loading_messages array is rejected ("must provide at
        # least 1 items"). So every visible update sends ONE text in BOTH
        # fields, keeping the in-thread transient and the composer line
        # identical (mismatched texts read as two indicators; user screenshots
        # 2026-07-09/10). Variety comes from the pools: the initial call picks
        # a random loading message, phase updates pick a random stage variant.
        # An explicit status="" (clear_assistant_status) goes out bare.
        if status == "":
            msgs = []  # clear: bare empty status, never loading_messages
            status_text = ""
        elif status is not None:
            msgs = loading_messages if loading_messages is not None else [status]
            status_text = status
        elif loading_messages:
            msgs = loading_messages
            status_text = loading_messages[0]
        else:
            pool = config.get_loading_messages() or [config.status_loading_fallback]
            pick = random.choice(pool)
            msgs = [pick]
            status_text = pick
        try:
            kwargs: Dict[str, Any] = {
                "channel_id": channel_id, "thread_ts": thread_id,
                "status": _status_plain_text(status_text) if status_text else ""}
            texts = [t for t in (_status_plain_text(m) for m in msgs) if t]
            if texts:
                kwargs["loading_messages"] = texts
            await self.app.client.assistant_threads_setStatus(**kwargs)
            return True
        except SlackApiError as e:
            # Most common in a plain channel: not an assistant thread -> just skip it.
            err = e.response.get("error") if getattr(e, "response", None) else e
            self.log_debug(f"assistant setStatus unavailable here ({err}); continuing without it")
            return False
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"assistant setStatus error: {e}")
            return False

    async def clear_assistant_status(self, channel_id: str, thread_id: str) -> bool:
        """Clear the assistant status: bare status="" with NO loading_messages (the API
        rejects an empty array and treats "" as the clear signal). Needed explicitly for
        native-streamed replies — Slack's auto-clear keys on chat.postMessage only."""
        return await self.set_assistant_status(channel_id, thread_id, status="")

    async def _react_add(self, channel_id: str, message_ts: str, emoji: str) -> tuple:
        """The raw add, returning (ok, added).

        `added` is the bit `react()` throws away and F38 needs: True only when THIS call
        actually put the reaction there. Slack's `already_reacted` is still ok=True — the
        emoji is present, the caller's intent is satisfied — but added=False, because a
        reaction we did not place is not a reaction we may take away. (Slack scopes
        reactions per user, so `already_reacted` can only mean the BOT reacted before, never
        that a human did.)"""
        if not config.enable_reactions:
            return False, False
        name = (emoji or "").strip().strip(":")
        if not name:
            return False, False
        if _epoch_refused(self, channel_id, "reactions_add"):
            return False, False
        try:
            await self.app.client.reactions_add(channel=channel_id, name=name, timestamp=message_ts)
            return True, True
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            if err == "already_reacted":
                return True, False  # present, but not ours to remove
            self.log_warning(f"Could not add reaction :{name}: ({err})")
            return False, False
        except Exception as e:  # noqa: BLE001
            self.log_error(f"Unexpected error adding reaction :{name}: {e}")
            return False, False

    async def react(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Add an emoji reaction to a message (Phase 4). ``emoji`` may include or omit colons.

        Best-effort: treats already_reacted as success, never raises.
        """
        ok, _added = await self._react_add(channel_id, message_ts, emoji)
        return ok

    async def unreact(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Remove one of the BOT'S OWN reactions (F38). Slack scopes reactions.remove to the
        authenticated user, so this can never strip a human's emoji off a message.

        `no_reaction` counts as success — the goal state is "the emoji is not there", and
        something else having removed it already satisfies that. Never raises."""
        if not config.enable_reactions:
            return False
        name = (emoji or "").strip().strip(":")
        if not name:
            return False
        if _epoch_refused(self, channel_id, "reactions_remove"):
            return False
        try:
            await self.app.client.reactions_remove(
                channel=channel_id, name=name, timestamp=message_ts)
            return True
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            if err == "no_reaction":
                return True  # already gone — the intended end state
            self.log_warning(f"Could not remove reaction :{name}: ({err})")
            return False
        except Exception as e:  # noqa: BLE001
            self.log_error(f"Unexpected error removing reaction :{name}: {e}")
            return False

    # --- react_to_message local tool (redesign Phase D) ---

    def _custom_emoji_available(self) -> bool:
        """Whether the workspace has ANY custom emoji — the cheap check behind the react
        schema's pointer to search_workspace_emoji. Never raises; False when the cache is
        absent or has never fetched (emoji:read missing is the de-facto off switch)."""
        cache = getattr(self, "workspace_emojis", None)
        if cache is None:
            return False
        try:
            return bool(cache.get_custom_emoji_names())
        except Exception:  # noqa: BLE001 — a schema build must never fail the turn
            return False

    def get_react_tool_schema(self, cfg: Optional[dict] = None) -> dict:
        """Registry FACTORY (called per request as ``schema(cfg)``) for the react_to_message
        tool. By default the model may pick ANY standard Slack emoji shorthand name — choosing
        the right one IS the judgment. If REACTION_EMOJIS is configured, it constrains the
        choice to that allowlist via an enum (brand control), and customs are NOT injected. When
        no allowlist is set, the workspace's custom emoji are surfaced as EXTRA named choices in
        the field DESCRIPTION (never an enum — an enum would forbid every standard emoji)."""
        allowed = [e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")]
        emoji_schema: Dict[str, Any] = {
            "type": "string",
            "description": "Any standard Slack emoji shorthand name (no colons), e.g. joy, tada, fire."}
        if allowed:
            emoji_schema["enum"] = allowed
        else:
            # No name list here any more. This used to inject the first ~64 custom names
            # ALPHABETICALLY out of ~1,400 ("000, 1password_icon, 4cats_q, alabama…"), which
            # spent ~600 chars on every single request to show the model an unusable slice of
            # its own workspace. search_workspace_emoji reaches all of them on demand instead,
            # so the pointer costs a fraction of the list and is actually useful.
            if self._custom_emoji_available():
                emoji_schema["description"] += (
                    " This workspace also has custom emoji; call search_workspace_emoji to find "
                    "one by meaning when a workspace-specific reaction would fit better."
                )
        return self._react_tool_schema(emoji_schema)

    def _react_tool_schema(self, emoji_schema: dict) -> dict:
        return {
            "type": "function",
            "name": "react_to_message",
            "description": (
                "Add an emoji reaction to a Slack message, the way a teammate would — when "
                "something lands, when you agree, when the room is already reacting, or to "
                "acknowledge a completed request (a ✅, a celebration). If a reaction alone "
                "fully answers the message, react and reply with empty text. Defaults to the "
                "message you are answering. Call once per emoji when asked for multiple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": emoji_schema,
                    "ts": {"type": "string",
                           "description": "Optional ts of another recent message in this channel to react to."},
                },
                "required": ["emoji"],
            },
        }

    def get_react_tool_schema_static(self, cfg: Optional[dict] = None) -> dict:
        """Channel-surface react_to_message: identical shape, but the custom-emoji pointer is
        decided by CONFIG rather than by the live cache.

        `_custom_emoji_available()` reads a cache that warms after start and can empty on an
        API failure, so the DM factory's sentence appears and disappears under a running
        process — a schema fork with no channel-level cause. The pointer instead rides exactly
        when `search_workspace_emoji` is registered: no REACTION_EMOJIS allowlist. The cache
        still decides what that tool RETURNS; it just no longer decides what the schema says.

        `cfg` is accepted and ignored so the registry can call it like a factory."""
        allowed = [e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")]
        emoji_schema: Dict[str, Any] = {
            "type": "string",
            "description": "Any standard Slack emoji shorthand name (no colons), e.g. joy, tada, fire."}
        if allowed:
            emoji_schema["enum"] = allowed
        else:
            emoji_schema["description"] += (
                " This workspace may also have custom emoji; call search_workspace_emoji to "
                "find one by meaning when a workspace-specific reaction would fit better."
            )
        return self._react_tool_schema(emoji_schema)

    def get_emoji_search_tool_schema(self) -> dict:
        """Schema for search_workspace_emoji — lookup over the workspace's CUSTOM emoji.

        Only registered when no REACTION_EMOJIS allowlist is set (see _build_tool_registry).
        The description has to earn its tokens on every request, so it says three things: what
        this reaches (customs only), when NOT to bother (standard emoji need no lookup), and
        that the call is silent. That last one matters — a reaction-only turn streams, so a
        narrated "let me look for an emoji…" would commit visible text and wreck the very
        thing a wordless reaction is for."""
        return {
            "type": "function",
            "name": "search_workspace_emoji",
            "description": (
                "Look up this workspace's CUSTOM emoji by meaning, e.g. 'ship it', 'celebrate "
                "a win', 'on fire'. Use it when a workspace-specific reaction would land "
                "better than a generic one; you do NOT need it for standard Slack emoji, "
                "which you can already name yourself. Returns candidate names to pass to "
                "react_to_message. Call it silently and say nothing about having searched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What the reaction should convey, in a few words."},
                    "limit": {"type": "integer",
                              "description": "Max names to return (default 16, max 40)."},
                },
                "required": ["query"],
            },
        }

    async def execute_emoji_search_tool(self, ctx, args: dict) -> dict:
        """Executor for search_workspace_emoji. In-memory, read-only, never raises.

        Re-checks the allowlist as defence in depth even though the tool is not registered
        when one is set: discovery must never become authorization. `execute_react_tool` and
        the schema enum stay the authority on what may actually be placed."""
        if not (config.enable_reactions and config.enable_react_tool):
            return {"ok": False, "error": "disabled", "message": "Reactions are disabled."}
        if config.reaction_emojis or []:
            return {"ok": False, "error": "allowlist_active",
                    "message": "A fixed reaction allowlist is configured; choose from it."}
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "no_query", "message": "Give a short description."}
        try:
            limit = int(args.get("limit") or 16)
        except (TypeError, ValueError):
            limit = 16
        cache = getattr(self, "workspace_emojis", None)
        if cache is None:
            return {"ok": True, "query": query, "matches": []}
        try:
            matches = cache.search(query, limit=max(1, min(40, limit)))
        except Exception as e:  # noqa: BLE001 — discovery must never fail a turn
            self.log_debug(f"emoji search failed: {e}")
            return {"ok": True, "query": query, "matches": []}
        return {"ok": True, "query": query, "matches": matches,
                "note": ("No custom emoji matched; use a standard Slack emoji."
                         if not matches else
                         "Custom names — pass one to react_to_message, or use a standard emoji.")}

    async def execute_react_tool(self, ctx, args: dict) -> dict:
        """Executor for react_to_message. Syntactic emoji validation (Slack's invalid_name
        is the semantic backstop) + optional REACTION_EMOJIS allowlist + per-message cap
        (REACTION_MAX_PER_MESSAGE distinct emoji); never raises (returns
        {"ok": False, ...} on any refusal/failure)."""
        channel_id = getattr(ctx, "channel_id", None)
        # The message that CAUSED this turn, which is not necessarily the one being reacted to —
        # the model may target an older message by ts. Both are recorded, separately.
        trigger_ts = getattr(ctx, "trigger_ts", None)
        # Scope guard: only a gate attempt has an id, and this ledger's population is gate
        # attempts. A mention or a DM reacting is a real reaction and simply not ours to count.
        attempt_id = getattr(ctx, "attempt_id", None)

        def _record(result_name: str, *, target_ts=None, detail=None) -> None:
            if not attempt_id:
                return
            # Recorded here rather than in _reserve_and_react, which is the shared choke point
            # for every reaction path and so cannot say WHICH decision chose this emoji. Origin
            # is the whole point: the gate picks from one blind prompt, the responder can search
            # the catalog, and a diversity number that averages the two describes neither.
            participation_telemetry.reaction(
                channel_id, trigger_ts, operation="add", result=result_name,
                origin="responder", emoji=emoji or None, target_ts=target_ts,
                attempt_id=attempt_id, detail=detail)

        emoji = (args.get("emoji") or "").strip().strip(":")
        # Resolved BEFORE the gauntlet, so a refusal can say what was aimed at. It used to be
        # resolved after, which recorded every refusal as "nothing was aimed at" — including the
        # ones that named an older message explicitly. A model repeatedly asking for a banned
        # emoji ON ONE PARTICULAR MESSAGE is a different prompt problem from one flailing at the
        # trigger, and the target is the only thing that tells them apart.
        ts = (args.get("ts") or "").strip() or trigger_ts or getattr(ctx, "thread_ts", None)
        # Every refusal below is an INTENT the room never saw. A model that keeps asking for a
        # disallowed emoji, or keeps aiming at nothing, is a prompt problem — and before these
        # were recorded the only refusal with any trace at all was the one that reached Slack.
        if not (config.enable_reactions and config.enable_react_tool):
            _record("refused", target_ts=ts, detail="disabled")
            return {"ok": False, "error": "disabled", "message": "Reactions are disabled."}
        if not valid_emoji_name(emoji):
            _record("refused", target_ts=ts, detail="invalid_emoji")
            return {"ok": False, "error": "invalid_emoji", "message": "Not a valid emoji shorthand name."}
        allowed = {e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")}
        if allowed and emoji not in allowed:
            _record("refused", target_ts=ts, detail="emoji_not_allowed")
            return {"ok": False, "error": "emoji_not_allowed", "allowed": sorted(allowed)}
        if not channel_id or not ts:
            _record("refused", detail="no_target")
            return {"ok": False, "error": "no_target", "message": "No message to react to."}
        result = await self._reserve_and_react(channel_id, ts, emoji)
        ok = bool(isinstance(result, dict) and result.get("ok") is True)
        # The turn now carries the fact, recorded where the emoji actually landed. Every path
        # that ends a turn reads it from here — a reaction-only turn, a reply that also reacted,
        # and a turn that ended in silence are all the same question ("is one of our emoji on
        # this message?"), and answering it per-branch is how three of the four branches came to
        # answer it wrong.
        turn = getattr(ctx, "turn", None)
        if ok and turn is not None:
            try:
                turn.reaction_committed = True
            except Exception as e:  # noqa: BLE001 — bookkeeping never fails a placed reaction
                self.log_debug(f"react tool: could not mark the turn's reaction: {e}")
        # `idempotent` means the emoji was already on the message — the reservation layer reports
        # ok either way, and counting that as a placement would credit us with somebody else's
        # reaction (see the ownership note above _reserve_and_react_owned).
        if ok and isinstance(result, dict) and result.get("idempotent"):
            _record("already_present", target_ts=ts)
        elif ok:
            _record("added", target_ts=ts)
        else:
            _record("failed", target_ts=ts,
                    detail=(result.get("error") if isinstance(result, dict) else None))
        return result

    # --- pin_message local tool (PIN §2/§3/§4) ---

    def get_pin_message_tool_schema(self) -> dict:
        """PIN §3: the one static schema, both surfaces, no channel_id property — the
        conversation comes from the ToolContext, and Slack's own `message_not_found` is what
        confines a target to it. The request-only policy lives in this description; enforcement
        is social, exactly as it is for the other participant-facing writes."""
        return {
            "type": "function",
            "name": "pin_message",
            "description": (
                "Pin a message to this conversation's pinned items, or unpin one — only when "
                "someone here asks you to. The target must be a message in THIS conversation. "
                "Slack shows who pinned what, and a pin is a shared surface: never pin on your "
                "own initiative, and never pin to make a point. Never call this just to "
                "check or refresh a pin — if nothing needs to change, make no call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pin", "unpin"],
                               "description": "pin adds the message to the pinned items; unpin removes it."},
                    "ts": {"type": "string",
                           "description": ("Timestamp (ts) of the target message in THIS "
                                           "conversation, exactly as an id you can see this "
                                           "turn: from the turn coordinates, a message header "
                                           "in the stream, or a tool result about this "
                                           "conversation. Never guess or derive one. A thread "
                                           "reply's ts pins that reply itself.")},
                },
                "required": ["action", "ts"],
            },
        }

    async def execute_pin_message(self, ctx: Any, args: Any) -> dict:
        """Shim: result observability + never-raise. One log line per unsuccessful call —
        except workspace_unavailable, whose single line is the epoch helper's own warning."""
        try:
            result = await self._execute_pin_message(ctx, args)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the tool contract is a dict, never a raise
            self.log_error(f"pin_message failed: {e}")
            return {"ok": False, "error": "pin_failed"}
        if isinstance(result, dict) and not result.get("ok") \
                and result.get("error") != "workspace_unavailable":
            self.log_info(f"pin_message refused: {result.get('error')}")
        return result

    async def _execute_pin_message(self, ctx: Any, args: Any) -> dict:
        """PIN §4: pin or unpin ONE message of this conversation, on request.

        §4.8 — NO `run_effect` lease, an explicit architectural exception: `run_effect` guards
        IRREVERSIBLE effects, and pin/unpin is reversible and idempotent. The §4.4 keyed lock
        covers ordering, and the worst case of a duplicate dispatch is `already_pinned`.

        Every model-supplied value is TYPE-CHECKED before any coercion and no Slack call may
        precede a refusal, so a malformed argument can never reach the workspace."""
        channel_id = getattr(ctx, "channel_id", None)
        if not channel_id:
            return {"ok": False, "error": "no_channel_context"}
        action = args.get("action")
        if not isinstance(action, str):
            return {"ok": False, "error": "invalid_action"}
        action = action.strip().lower()
        if action not in ("pin", "unpin"):
            return {"ok": False, "error": "invalid_action"}
        raw_ts = args.get("ts")
        if not isinstance(raw_ts, str):
            return {"ok": False, "error": "invalid_ts"}
        ts = raw_ts.strip()
        try:
            parse_ts(ts)
        except TimestampError:
            return {"ok": False, "error": "invalid_ts"}
        # §4.4: the module-level edit lock, reused AS-IS and keyed the same way, so two turns
        # aiming at one message serialize. Last-writer-wins across turns is accepted.
        lock_key = (getattr(self, "self_team_id", None), channel_id, ts)
        lock = _edit_transaction_lock(*lock_key)
        try:
            async with lock:
                # Epoch check INSIDE the lock, at the mutation point: a turn parked behind
                # this lock must not spend an authorization that went stale while it waited.
                if _epoch_refused(self, channel_id,
                                  "pins_add" if action == "pin" else "pins_remove"):
                    # The helper already warned; the wrapper stays quiet about this one.
                    return {"ok": False, "error": "workspace_unavailable"}
                try:
                    if action == "pin":
                        await self.app.client.pins_add(channel=channel_id, timestamp=ts)
                    else:
                        await self.app.client.pins_remove(channel=channel_id, timestamp=ts)
                    return {"ok": True, "action": action, "ts": ts}
                except SlackApiError as e:
                    # Never str(e): the API error name is the whole classification.
                    resp = getattr(e, "response", None)
                    err = (resp.get("error")
                           if resp is not None and callable(getattr(resp, "get", None))
                           else None)
                    if err == "already_pinned":
                        return {"ok": True, "action": "pin", "ts": ts, "note": "already pinned"}
                    if err in ("no_pin", "not_pinned"):
                        return {"ok": True, "action": "unpin", "ts": ts,
                                "note": "was not pinned"}
                    if isinstance(err, str) and err and err not in ("fatal_error",
                                                                    "internal_error"):
                        return {"ok": False, "error": err}
                    # AMBIGUOUS (§4.6): the mutation may have landed → reconcile below.
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    pass  # AMBIGUOUS for the same reason: no answer is not a failed write.
                # §4.7: EXACTLY ONE read, no retry and no second mutation. Whatever this cannot
                # settle is `outcome_unknown` — a guess here is how a pin gets double-applied.
                try:
                    resp = await self.app.client.pins_list(channel=channel_id)
                except Exception:  # noqa: BLE001 — an unreadable state is outcome_unknown
                    return {"ok": False, "error": "outcome_unknown"}
                # The SDK hands back an AsyncSlackResponse, not a dict.
                body = resp.data if isinstance(getattr(resp, "data", None), dict) else (
                    resp if isinstance(resp, dict) else None)
                if (body is None or body.get("ok") is not True
                        or not isinstance(body.get("items"), list)):
                    return {"ok": False, "error": "outcome_unknown"}
                # A malformed ITEM is skipped, not fatal — one junk entry must not turn a
                # readable answer into an unknown one.
                present = any(
                    isinstance(item, dict)
                    and isinstance(item.get("message"), dict)
                    and item["message"].get("ts") == ts
                    for item in body["items"])
                if action == "pin":
                    if present:
                        return {"ok": True, "action": "pin", "ts": ts,
                                "note": "confirmed pinned after an ambiguous response"}
                    return {"ok": False, "error": "pin_failed"}
                if present:
                    return {"ok": False, "error": "unpin_failed"}
                return {"ok": True, "action": "unpin", "ts": ts,
                        "note": "confirmed removed after an ambiguous response"}
        finally:
            _prune_edit_transaction_lock(lock_key, lock)

    # Reaction-guard eviction tuning. Entries touched within the recency window are PINNED
    # (never evicted) — this covers both the committed slots of an ACTIVE turn (so a burst
    # of 2000+ reactions on other messages can't resurrect a message's consumed slots) and
    # fresh pending reservations. A pending future untouched for the whole window is treated
    # as abandoned and becomes evictable (bounded expiry for a never-resolving Future).
    _REACTION_GUARD_MAX = 2000
    _REACTION_GUARD_RECENCY_S = 120.0

    def _trim_reaction_guard(self, guard, ts_map, now, keep=None) -> None:
        """Evict oldest guard entries beyond the cap, pinning anything touched within the
        recency window (and always ``keep``).

        F38: an entry holding a LIVE OWNED slot is pinned unconditionally. Ownership is what
        lets a turn take its 👀 back, and a long turn (a research job runs for minutes) would
        otherwise age out of the recency window and lose the right to clean up after itself."""
        if len(guard) <= self._REACTION_GUARD_MAX:
            return
        cutoff = now - self._REACTION_GUARD_RECENCY_S
        for k in list(guard.keys()):
            if len(guard) <= self._REACTION_GUARD_MAX:
                break
            entry = guard.get(k)
            if entry is keep:
                continue
            if any(isinstance(v, dict) for v in (entry or {}).values()):
                continue  # a live claim lives here — evicting it would strand the 👀
            if ts_map.get(k, 0.0) >= cutoff:
                continue  # recently touched → pinned (active-turn committed or fresh pending)
            del guard[k]
            ts_map.pop(k, None)

    # --- F38: reaction leases (a 👀 the turn can take back) ---
    #
    # Ownership is part of the GUARD, not a map beside it. That matters, and it took a review
    # round to see why: a parallel map cannot be kept honest, because Slack's `already_reacted`
    # is silent about WHO reacted. Sequence that breaks the parallel design —
    #
    #   1. turn A adds 👀 and records itself as owner
    #   2. A's guard entry is evicted (2000-entry LRU)
    #   3. turn B reserves the same emoji: no slot, so it calls Slack
    #   4. Slack: `already_reacted` (A's 👀 is still up there) → B gets no lease...
    #   5. ...and B never overwrites A's ownership record, because it never had one to write
    #   6. A ends silently, its token still "matches", and it rips the 👀 out from under B
    #
    # Holding ownership in the slot makes step 2 self-correcting: eviction destroys the claim,
    # so A can no longer prove the emoji is its own and declines to touch it. Losing the right
    # to clean up is the safe failure; removing someone else's reaction is not.
    #
    # A slot is therefore one of:
    #   Future              an add is in flight (a concurrent sibling reserved it first)
    #   True                committed, unowned — nobody may remove it
    #   {"token": ...}      committed and OWNED by the turn holding the matching lease
    #   {"token", "removing"}  that owner is mid-removal; no one else may touch it
    _REMOVING = "removing"

    @staticmethod
    def _is_committed(slot) -> bool:
        """True/owned/removing all mean 'the emoji is on the message'. A Future does not."""
        return slot is True or isinstance(slot, dict)

    def settle_reaction_lease(self, lease: Optional[dict]) -> None:
        """The turn produced something: the reaction has earned its place. Drop the claim —
        the emoji and the guard slot stay exactly as they are.

        Releasing matters even for reactions nobody intends to remove (a gate verdict, the
        model's own react tool): an owned slot is pinned against eviction, so never settling
        one would slowly fill the guard with unevictable entries."""
        if not lease:
            return
        slots = (getattr(self, "_reaction_guard", None) or {}).get(
            (lease.get("channel_id"), lease.get("ts")))
        if slots is None:
            return
        slot = slots.get(lease.get("emoji"))
        if isinstance(slot, dict) and slot.get("token") == lease.get("token"):
            slots[lease["emoji"]] = True  # committed, unowned, evictable again

    def _settle_removal_slot(self, channel_id: str, ts: str, emoji: str, token: str,
                             ok: bool) -> None:
        """Transition a `removing` slot to its final state. Runs from the removal TASK's
        `finally`, so it happens even if the turn that asked for the removal is cancelled —
        otherwise the slot would stay `removing` forever, and since owned slots are pinned
        against eviction, a run of cancelled turns would grow the guard without bound."""
        guard = getattr(self, "_reaction_guard", None)
        slots: Any = (guard or {}).get((channel_id, ts))
        slot = slots.get(emoji) if slots is not None else None
        if not (isinstance(slot, dict) and slot.get("token") == token
                and slot.get(self._REMOVING) is not None):
            return  # someone else already resolved it; don't stomp their state
        if not ok:
            # The emoji may well still be up there. Demote to committed-unowned rather than
            # dropping the slot: a stale 👀 is survivable, a guard that thinks a live reaction
            # is gone is not (it would let the cap be exceeded and re-add over the top).
            slots[emoji] = True
            self.log_debug(f"Could not take back :{emoji}: — leaving the bookkeeping intact")
            return
        slots.pop(emoji, None)
        if not slots and guard is not None and guard.get((channel_id, ts)) is slots:
            guard.pop((channel_id, ts), None)
            ts_map = getattr(self, "_reaction_guard_ts", None)
            if ts_map is not None:
                ts_map.pop((channel_id, ts), None)
        self.log_debug(f"Took back :{emoji}: — the turn produced nothing")

    async def _run_reaction_removal(self, channel_id: str, ts: str, emoji: str,
                                    token: str) -> bool:
        """The removal itself, as its own task. Bounded, and it ALWAYS settles the slot."""
        ok = False
        try:
            ok = await asyncio.wait_for(
                self.unreact(channel_id, ts, emoji),
                timeout=max(1.0, float(getattr(config, "tool_call_timeout", 20))))
        except asyncio.TimeoutError:
            self.log_debug(f"Reaction removal timed out for :{emoji}:")
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"Reaction removal failed for :{emoji}: {e}")
        finally:
            # `finally`, not `except Exception` — a CancelledError is a BaseException and
            # would otherwise sail straight past, stranding the slot in `removing`.
            self._settle_removal_slot(channel_id, ts, emoji, token, ok)
        return ok

    async def remove_owned_reaction(self, lease: Optional[dict]) -> bool:
        """The turn produced nothing: take the reaction back off.

        Refuses unless the slot still carries OUR token — if the entry was evicted and
        re-committed, or another turn claimed the emoji, it is no longer ours and we leave it.

        The removal runs as its own task and settles the guard from a `finally`, so a
        cancelled turn cannot strand the slot mid-removal. The caller merely waits for it."""
        if not lease:
            return False
        # A lease is only ever minted by reserve_reaction_slot, which fills all four keys.
        channel_id, ts = cast(str, lease.get("channel_id")), cast(str, lease.get("ts"))
        emoji, token = cast(str, lease.get("emoji")), cast(str, lease.get("token"))
        guard = getattr(self, "_reaction_guard", None)
        slots: Any = (guard or {}).get((channel_id, ts))
        slot = slots.get(emoji) if slots is not None else None
        if not (isinstance(slot, dict) and slot.get("token") == token
                and slot.get(self._REMOVING) is None):
            self.log_debug(f"Reaction lease for :{emoji}: is stale — leaving it alone")
            return False
        # Publish the removal SYNCHRONOUSLY, before any await, so a concurrent remover bails
        # and a concurrent reserver can WAIT on the outcome rather than being told the emoji
        # is safely present when it is moments from disappearing.
        task = asyncio.ensure_future(
            self._run_reaction_removal(channel_id, ts, emoji, token))
        slots[emoji] = {"token": token, self._REMOVING: task}
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise      # the task lives on and settles the slot itself
        except Exception:  # noqa: BLE001
            return False

    async def _reserve_and_react(self, channel_id: str, ts: str, emoji: str) -> dict:
        """F6 reservation for a PERMANENT reaction — one the caller never takes back.

        Settles the lease immediately: this reaction is staying, so it must not sit in the
        guard as an unevictable owned slot."""
        result, lease = await self._reserve_and_react_owned(channel_id, ts, emoji)
        self.settle_reaction_lease(lease)
        return result

    async def _reserve_and_react_owned(self, channel_id: str, ts: str, emoji: str) -> tuple:
        """F6 reservation + F38 lease. Returns (result, lease).

        Retries until the slot reaches a STABLE state, under one absolute deadline. Every
        await in `_reserve_once` — waiting on someone else's in-flight add, waiting on a
        removal — can be overtaken while we sleep: the owner of the add we waited for may
        start removing it, a removal we waited for may be followed by a fresh add and a second
        removal. So a pass that had to wait never trusts what it learned; it re-reads the slot
        and decides again.

        A fixed number of passes cannot express that (I tried two, and codex pointed out that
        two removal generations back-to-back fall straight through it and return nothing at
        all). The deadline can: we either converge on a real answer or say so honestly."""
        deadline = time.monotonic() + max(1.0, float(getattr(config, "tool_call_timeout", 20)))
        while True:
            result, lease, retry = await self._reserve_once(
                channel_id, ts, emoji, deadline)
            if not retry:
                return result, lease
            if time.monotonic() >= deadline:
                # Never fall through with a None result — the react tool subscripts it.
                return ({"ok": False, "error": "reaction_busy",
                         "message": f"Could not settle :{emoji}: — it is being changed "
                                    f"concurrently. Try again."}, None)

    async def _reserve_once(self, channel_id: str, ts: str, emoji: str,
                            deadline: Optional[float] = None) -> tuple:
        """One reservation pass. Returns (result, lease, retry).

        Guard: bounded LRU map (channel, ts) -> {emoji: Future(pending) | True(committed)
        | {"token"}(owned) | {"token","removing"}(being taken back)}, plus a parallel
        (channel, ts) -> monotonic touch time. Distinct emoji up to REACTION_MAX_PER_MESSAGE
        land; a duplicate emoji is idempotent success WITHOUT consuming a slot. Because
        dispatch_all runs sibling calls concurrently, the slot is reserved SYNCHRONOUSLY
        (before the first await) so N+1 distinct reactions can't all pass the cap; a
        failed/cancelled Slack call rolls the reservation back in `finally`. A duplicate whose
        in-flight owner FAILS must not report success (round-2 fix a). Eviction pins
        recently-touched entries and any entry holding a live claim; a duplicate's wait on an
        in-flight owner is time-bounded.

        F38 — the LEASE. Non-None only when THIS call genuinely added the reaction (not a
        duplicate, not `already_reacted`, not a wait on someone else's in-flight add). It is
        the receipt that lets `remove_owned_reaction` prove the emoji on screen is the one we
        put there, so a work-claim 👀 we take back can never strip a reaction that a
        concurrent turn — or the model's own react tool — has since made its own."""
        now = time.monotonic()
        cap = max(1, int(getattr(config, "reaction_max_per_message", 4)))
        guard = getattr(self, "_reaction_guard", None)
        if guard is None:
            guard = self._reaction_guard = OrderedDict()  # (channel, ts) -> {emoji: Future|True}
        ts_map = getattr(self, "_reaction_guard_ts", None)
        if ts_map is None:
            ts_map = self._reaction_guard_ts = {}       # (channel, ts) -> monotonic touch time
        key = (channel_id, ts)
        slots = guard.get(key)
        if slots is None:
            slots = guard[key] = {}
        guard.move_to_end(key)  # LRU recency refresh
        ts_map[key] = now
        self._trim_reaction_guard(guard, ts_map, now, keep=slots)

        # How long we may wait on someone else's in-flight operation: whatever is left of the
        # caller's overall deadline, so a slot that keeps churning can't outlast it.
        wait_bound = max(1.0, float(getattr(config, "tool_call_timeout", 20)))
        if deadline is not None:
            wait_bound = min(wait_bound, max(0.0, deadline - now))

        busy = ({"ok": False, "error": "reaction_busy",
                 "message": f"Could not settle :{emoji}: — it is being changed concurrently. "
                            f"Try again."}, None, False)

        existing = slots.get(emoji)
        if existing is not None:
            removal = existing.get(self._REMOVING) if isinstance(existing, dict) else None
            if removal is not None:
                # The emoji is being TAKEN BACK right now. Reporting "it's there" would be a
                # lie the moment the removal lands — and for the model's react tool that lie
                # becomes a reaction-only reply whose reaction does not exist. Wait for the
                # real outcome instead.
                try:
                    await asyncio.wait_for(asyncio.shield(removal), timeout=wait_bound)
                except asyncio.TimeoutError:
                    # Still running. We do NOT know how it ends, and guessing "still present"
                    # would be the same lie one step later. Say so honestly.
                    return busy
                except Exception:
                    pass
                # Resolved, one way or the other — and the task has already settled the slot
                # (popped on success, demoted to True on failure). Anything we remember about
                # it is stale, so decide again from what the guard says NOW.
                return None, None, True
            # Duplicate emoji — idempotent, no new slot consumed. No lease: we did not put
            # this one there, so it is not ours to take back.
            if self._is_committed(existing):
                return {"ok": True, "emoji": emoji, "ts": ts, "idempotent": True}, None, False
            # In-flight ADD: await the owner's real outcome (shield so our cancellation
            # doesn't cancel theirs), time-bounded so a never-resolving owner can't hang us.
            try:
                ok = await asyncio.wait_for(asyncio.shield(existing), timeout=wait_bound)
            except asyncio.TimeoutError:
                return busy
            except Exception:
                ok = False
            if not ok:
                return ({"ok": False, "error": "reaction_failed",
                         "message": f"Could not add :{emoji}:."}, None, False)
            # The add succeeded — but that was then. While we slept, its owner may already
            # have started taking it back (a work-claim turn that produced nothing). Reporting
            # success on the strength of a stale future would promise a reaction that is on
            # its way out. Re-read the slot and decide again.
            return None, None, True

        # New emoji — enforce the cap over committed + pending distinct emoji.
        if len(slots) >= cap:
            return ({"ok": False, "error": "reaction_cap",
                     "message": f"Already at the max of {cap} reactions on that message."},
                    None, False)

        # Reserve synchronously (before any await) so concurrent siblings see the slot.
        fut = asyncio.get_event_loop().create_future()
        slots[emoji] = fut
        committed = False
        try:
            ok, added = await self._react_add(channel_id, ts, emoji)
            if ok:
                committed = True
                if not fut.done():
                    fut.set_result(True)
                if not added:
                    # Slack said already_reacted: the emoji is present, but WE did not put it
                    # there this time — a previous turn did, and the guard entry proving it was
                    # evicted. Commit the slot UNOWNED and mint no lease: removing it would
                    # take back a reaction that is not ours.
                    slots[emoji] = True
                    return {"ok": True, "emoji": emoji, "ts": ts, "idempotent": True}, None, False
                token = uuid4().hex
                slots[emoji] = {"token": token}   # committed AND owned by this caller
                return ({"ok": True, "emoji": emoji, "ts": ts},
                        {"token": token, "channel_id": channel_id, "ts": ts,
                         "emoji": emoji},
                        False)
            if not fut.done():
                fut.set_result(False)
            return ({"ok": False, "error": "reaction_failed",
                     "message": f"Could not add :{emoji}:."}, None, False)
        finally:
            if not committed:
                # Roll back the reservation — covers failure, timeout, and cancellation.
                if slots.get(emoji) is fut:
                    del slots[emoji]
                if not fut.done():
                    fut.set_result(False)
                # Identity-conditional cleanup: only drop the key when it STILL maps to
                # our own slots object (a concurrent recreate after eviction installs a
                # different dict, which we must not delete).
                if not slots and guard.get(key) is slots:
                    guard.pop(key, None)
                    ts_map.pop(key, None)
            # Retrim after settlement: a burst may have blown past the cap with everything
            # pending (nothing evictable then); now that this call resolved, sweep again.
            self._trim_reaction_guard(guard, ts_map, time.monotonic())

    # The DM surface's wording, kept as the single source of the legacy bytes: the channel schema
    # below reads its own constants and falls back to these, so a wave that lands the READ without
    # the words changes nothing anywhere.
    _POST_TO_THREAD_DESCRIPTION = (
        "Post a reply into a DIFFERENT thread in THIS channel. Use when a reply "
        "belongs somewhere other than the current conversation — because that thread "
        "holds something this turn settles: a question left open there, an answer you "
        "owed there, an earlier answer of yours that is now wrong. "
        "Acknowledge briefly in the current thread rather than "
        "duplicating the whole answer in both places. Only targets threads in the "
        "current channel; there is no way to post to another channel."
    )
    _POST_TO_THREAD_TARGET_DESCRIPTION = (
        "Root ts of the target conversation (a top-level message's ts "
        "targets its thread). Must be a ts you have actually seen in "
        "context or from a tool — never guess one."
    )

    @staticmethod
    def _post_to_thread_schema(description: str, target_description: str) -> dict:
        return {
            "type": "function",
            "name": "post_to_thread",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_ts": {
                        "type": "string",
                        "description": target_description,
                    },
                    "text": {
                        "type": "string",
                        "description": "The reply to post, in normal markdown (converted to Slack "
                                       "formatting automatically).",
                    },
                },
                "required": ["thread_ts", "text"],
            },
        }

    def get_post_to_thread_tool_schema(self) -> dict:
        """F23: schema for the cross-thread reply tool. CURRENT CHANNEL ONLY — there is no
        channel_id param; cross-channel posting is out of scope (a write boundary, unlike the
        read tools that can reach other channels)."""
        return self._post_to_thread_schema(self._POST_TO_THREAD_DESCRIPTION,
                                           self._POST_TO_THREAD_TARGET_DESCRIPTION)

    def get_post_to_thread_channel_schema(self, thread_config: Optional[dict] = None) -> dict:
        """The CHANNEL surface's variant: static (``thread_config`` is accepted and ignored, so
        the registry can call it like any channel schema) and read from prompts constants.

        Two things the channel wording has to say differently, which is why it is its own schema
        rather than a shared string. The origin-acknowledgment instruction cannot stand on a
        surface where the model may post once and stay quiet here — and the promise about WHERE a
        target id may come from has to match what the executor actually allows, which on a channel
        turn is the stream's own thread labels and nothing else. An empty constant means the words
        have not landed yet, and the legacy description is used verbatim."""
        return self._post_to_thread_schema(
            getattr(prompts, "CHANNEL_POST_TO_THREAD_DESCRIPTION", "")
            or self._POST_TO_THREAD_DESCRIPTION,
            getattr(prompts, "CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION", "")
            or self._POST_TO_THREAD_TARGET_DESCRIPTION)

    async def execute_post_to_thread(self, ctx, args: dict) -> dict:
        """Executor for post_to_thread (F23). Posts a markdown-converted reply into another
        thread of the CURRENT channel via the standard messaging layer. Every refusal and every
        failure is an {"ok": False, ...} result, with ONE deliberate exception:
        ``StaleSendSuppressed`` is re-raised (the conversation moved on and nothing was
        attempted, which is control flow the turn's own handler files — reporting it as a tool
        error would have the model retry the post the guard just refused). Runs inside an
        addressed/judged turn, so no unprompted accounting is added."""
        if not config.enable_post_to_thread_tool:
            return {"ok": False, "error": "disabled", "message": "Cross-thread posting is disabled."}
        channel_id = getattr(ctx, "channel_id", None)
        if not channel_id:
            return {"ok": False, "error": "no_channel", "message": "No channel to post into."}
        target = (args.get("thread_ts") or "").strip()
        text = (args.get("text") or "").strip()
        if not target:
            return {"ok": False, "error": "missing_thread_ts", "message": "A target thread_ts is required."}
        if not text:
            return {"ok": False, "error": "empty_text", "message": "Nothing to post — text was empty."}
        # Posting into the CURRENT conversation would double-post alongside the normal reply.
        current = getattr(ctx, "thread_ts", None)
        trigger = getattr(ctx, "trigger_ts", None)
        if target == current or target == trigger:
            return {"ok": False, "error": "same_thread",
                    "message": "That's the current thread — just reply normally instead."}
        # AUTHORIZATION, before anything is sent. The allowlist is the set of thread roots this
        # turn's stream actually SHOWED the model, frozen when the stream was pinned — so the
        # answer to "may I post there?" is a fact about what was rendered, not a live Slack lookup
        # that could say yes to a thread the model never saw and is only guessing at.
        #
        # None means there is no stream to authorize against (a DM, a background agent, a
        # hand-built context) and the legacy behavior is kept exactly — which is why the handler
        # never hands None for a stream that exists but is unreadable: that case arrives as the
        # EMPTY set, and an empty set still enforces, the same as a turn whose stream genuinely
        # rendered no thread. Either way there is no thread to post into.
        #
        # W3 widened WHERE the set comes from and changed nothing here. A root a search or
        # history result returned this turn joins it at the moment that tool result was produced
        # — upstream of this check, from the result's own contents, before the model had said
        # anything about it — and is carried in on `trusted_thread_roots` like any other member.
        # It is still never widened HERE, at the moment of posting, because then "authorized"
        # would mean "the model named it".
        trusted = getattr(ctx, "trusted_thread_roots", None)
        if trusted is not None and target not in trusted:
            return {"ok": False, "error": "unknown_thread",
                    "message": ("That thread isn't one of the threads in front of you. Use a "
                                "thread's root ts exactly as it appears in this channel's stream, "
                                "or open it first with fetch_thread_messages, or answer here "
                                "instead.")}
        turn = getattr(ctx, "turn", None)
        lease = getattr(turn, "send_lease", None) if turn is not None else None
        from message_processor.turn_runtime import (DEST_KIND_POST_TO_THREAD, EffectRevoked,
                                                   LaunchNotRecorded, mark_tool_launched,
                                                   run_effect)

        def _observe(ts: str) -> None:
            """Slack accepted the first part. Recorded HERE, not after the send returns: a
            cancellation in between would otherwise leave words in another thread that this
            turn's ledger never mentions."""
            if turn is None:
                return
            try:
                turn.note_destination_observed(channel_id=channel_id, first_ts=ts,
                                              kind=DEST_KIND_POST_TO_THREAD,
                                              thread_root_ts=target)
            except Exception as e:  # noqa: BLE001 — bookkeeping never fails a post
                self.log_debug(f"post_to_thread: could not observe the destination: {e}")

        send_meta: dict = {}

        async def _post_and_account():
            """THE critical section: launch, delivery, receipt, and every record of both.

            All of it inside the lease because a cancellation between any two of these steps
            leaves our own words in somebody else's thread described as something they are not.
            `mark_launched` is first and has no await after it before the send — a cancelled
            flight whose shielded body is still posting must keep its key, or a replay of the
            same call id would post the message a second time.
            """
            mark_tool_launched(ctx)
            posted = await self.send_message(
                channel_id, target, text, lease=lease, surface="post_to_thread",
                meta_out=send_meta, on_first_accept=_observe,
                receipts=getattr(turn, "receipt_ledger", None) if turn is not None else None,
                receipt_class="assistant_reply")
            if not posted:
                return None, None
            landed = send_meta.get("delivery")
            # Words went into the workspace, just not into this thread. The turn may still end
            # with no_response_needed — that is now a legitimate pairing ("I answered over
            # there") — and without this the ledger would file a turn that visibly posted as a
            # silence. It is written HERE, beside the delivery it describes, because a
            # continuation that ran only if the caller was still waiting would leave an accepted
            # foreign post recorded as an observation with no commitment and no visible action.
            if turn is not None:
                try:
                    turn.visible_action_committed = True
                    # WHERE they landed: a different thread of this channel, so the record is
                    # keyed on the TARGET root rather than the turn's own conversation. A
                    # cross-thread post is written once and never edited afterwards, so
                    # acceptance IS finalization — but only of what Slack took: a split that
                    # aborted partway commits its delivered prefix and says so.
                    turn.mark_destination_committed(
                        first_ts=posted, kind=DEST_KIND_POST_TO_THREAD,
                        text=landed.text if landed is not None else text,
                        complete=landed.complete if landed is not None else True,
                        channel_id=channel_id, thread_root_ts=target)
                except Exception as e:  # noqa: BLE001 — never fail a delivered post
                    self.log_debug(f"post_to_thread: could not record the destination: {e}")
            return posted, landed

        try:
            # The target root rides the receipt: a cross-thread post belongs to the thread it
            # landed in, not to the turn's own conversation (spec §5 / P2 root discovery).
            #
            # LEASED: Slack accepting the post, `note_post` claiming it as ours, and this turn's
            # record of both are one critical section. Split them and a cancellation in between
            # leaves our own words in somebody else's thread with nothing claiming them —
            # permanently outside the stream we rebuild the room from, and unattributable
            # forever after. The lease also refuses outright once this turn has revoked its
            # effects, so a straggler that outlived its own cancellation cannot post at all.
            posted_ts, delivery = await run_effect(turn, "post_to_thread", _post_and_account)
        except LaunchNotRecorded as e:
            # The one mechanism that stops this call id posting twice is broken. Nothing was
            # sent, and it is said plainly rather than sent anyway.
            self.log_error(f"post_to_thread: launch not recorded for {channel_id}/{target}: {e}")
            return {"ok": False, "error": "launch_not_recorded",
                    "message": ("Something went wrong before the post was sent, so nothing was "
                                "posted in that thread.")}
        except EffectRevoked:
            # The turn withdrew permission before this post started, so NOTHING was attempted.
            # Reported as its own outcome rather than as a failed post: "could not post" would
            # invite a retry of something we deliberately did not do.
            self.log_warning(
                f"post_to_thread: refused for {channel_id}/{target} — the turn was cut short")
            return {"ok": False, "error": "turn_cancelled",
                    "message": ("This turn was cut short before the post went out, so nothing "
                                "was posted in that thread.")}
        except StaleSendSuppressed:
            # The conversation moved on before this landed. NOT a post failure: nothing was
            # attempted, so telling the model the send failed would invite a retry of something
            # we deliberately did not do. Re-raised so the turn's own handler files it as the
            # suppression it is.
            raise
        except Exception as e:
            self.log_warning(f"post_to_thread: send failed for {channel_id}/{target}: {e}")
            return {"ok": False, "error": "post_failed", "message": "Could not post to that thread."}
        if not posted_ts:
            return {"ok": False, "error": "post_failed", "message": "Could not post to that thread."}
        if delivery is not None and not delivery.complete:
            # The model is told the truth so it can decide what to do about it, and the post is
            # still reported as landed — part of it did.
            return {"ok": True, "thread_ts": target, "posted_ts": posted_ts,
                    "truncated": True,
                    "message": (f"Only {delivery.parts_delivered} of {delivery.parts_total} "
                                "parts reached that thread; the rest failed to post.")}
        return {"ok": True, "thread_ts": target, "posted_ts": posted_ts}

    # ------------------------------------------------------------------ edit_own_message (EDIT)

    # §2c's one refusal for a target that is not on this turn's mapping — identical whether the
    # ts exists, is somebody else's, is chrome, or was invented, so a probe learns nothing.
    _EDIT_UNAUTHORIZED = {
        "ok": False, "error": "unauthorized_target",
        "message": "That message was not an editable reply shown to you this turn.",
    }

    def get_edit_own_message_tool_schema(self) -> dict:
        """EDIT §3: the one static schema, both surfaces (the DM surface never exposes it).
        No channel_id, no thread_ts, no announcement text, no old-text echo — the channel comes
        from the ToolContext exactly as post_to_thread's does, and the disclosure is SYNTHESIZED
        by the executor so it can never be omitted or weakened."""
        return {
            "type": "function",
            "name": "edit_own_message",
            "description": (
                "Replace the text of ONE of your own earlier messages in this channel. This is "
                "the exception, not the default: correct yourself with a NEW message unless the "
                "standing wrong text itself would keep misleading anyone who reads it where it "
                "is. Every edit posts a public correction notice into the edited message's "
                "thread — there is no silent edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_ts": {
                        "type": "string",
                        "description": ("Exact ts of one editable assistant message shown in "
                                        "this channel stream or returned by a read tool this "
                                        "turn."),
                    },
                    "new_text": {
                        "type": "string",
                        "description": ("Complete corrected replacement body in normal "
                                        "markdown. Omit continuation markers and footer "
                                        "chrome; the tool preserves those."),
                    },
                    "correction_note": {
                        "type": "string",
                        "description": ("A concise public description of the specific fact or "
                                        "detail being corrected."),
                    },
                },
                "required": ["message_ts", "new_text", "correction_note"],
            },
        }

    async def execute_edit_own_message(self, ctx, args: dict) -> dict:
        """Thin observability shim: every refusal names itself in the log — the quiet
        validation returns are invisible otherwise (the tool loop logs only ok/error), and a
        live refusal with no name cost a debugging round."""
        result = await self._execute_edit_own_message(ctx, args)
        if isinstance(result, dict) and not result.get("ok"):
            self.log_info(f"edit_own_message refused: {result.get('error')}")
        return result

    async def _execute_edit_own_message(self, ctx, args: dict) -> dict:
        """EDIT §5/§7: the two-effect transaction — disclosure FIRST, overwrite second.

        The ENTIRE transaction (preflight re-reads → announcement → post-announcement second
        read → epoch authorization + chat.update → ambiguous-result reconciliation →
        EditRecord/accounting) runs under ONE process-wide keyed lock `(team, channel, ts)`,
        INSIDE one `run_effect` lease, so a cancellation after launch cannot split the pair and
        two turns can never both announce and overwrite. `StaleSendSuppressed` from the
        announcement's leased send re-raises UNCHANGED, exactly as post_to_thread's does — the
        conversation moved on and nothing was attempted.

        §11.14/§11.22: the `edit_failed` backstop guards from the executor's LITERAL first
        line — the imports live at module scope and even the ctx reads run inside the try, so
        a raising context property answers `edit_failed`, never a traceback. §11.15/§11.19:
        acceptance accounting happens at send_message's `on_first_accept` seam over a
        PRE-CREATED EditRecord, and is verified-and-repaired in-lock after the send.
        §11.16/§11.21: announcement-send exceptions are classified exhaustively —
        ambiguous-outcome ones return `announcement_outcome_unknown`, deterministic-local ones
        the §5 `announcement_failed` row; either way no update and no EditRecord."""
        # §11.22: nothing but bare-literal assignments before the try — exactly the names the
        # except handlers read, none of which can raise.
        channel_id: Optional[str] = None
        message_ts = ""
        turn = None
        # Mutable cell rather than a nonlocal, so `_observe` (called from inside send_message)
        # can read the landing thread the transaction resolved after the closure was built.
        announcement_thread: Dict[str, Optional[str]] = {"ts": None}
        # §11.8: the PHASE marker for the backstops. Filled the instant Slack accepts the
        # disclosure, so an unexpected exception after that point can never be reported as a
        # bare failure that invites a duplicate correction.
        accepted: Dict[str, Any] = {"record": None, "ts": None}
        try:
            channel_id = getattr(ctx, "channel_id", None)
            # §3: defense-in-depth for the surface the registration already hides this on. DMs
            # have no receipts, so no exact-message proof can exist there.
            if getattr(ctx, "is_dm", False) or str(channel_id or "").startswith("D"):
                return {"ok": False, "error": "channel_only",
                        "message": "Editing your own messages is only available in channels."}
            if not channel_id:
                return {"ok": False, "error": "no_channel", "message": "No channel to edit in."}
            message_ts = str(args.get("message_ts") or "").strip()
            if not message_ts:
                return {"ok": False, "error": "missing_message_ts",
                        "message": "A target message_ts is required."}
            # §11.14/§11.22: model-supplied values are TYPE-CHECKED before any coercion — a
            # numeric/list value must be `edit_failed`, never coerced into a postable string
            # (str() on a list would PUBLISH the repr) and never laundered into an `empty_*`
            # error by a falsey `or ""`.
            raw_text = args.get("new_text")
            raw_note = args.get("correction_note")
            if (raw_text is not None and not isinstance(raw_text, str)) or \
                    (raw_note is not None and not isinstance(raw_note, str)):
                return {"ok": False, "error": "edit_failed",
                        "message": "new_text and correction_note must be strings."}
            # §11.17: citations are stripped BEFORE the emptiness checks — citation-only input
            # names nothing, and a disclosure must never post about it.
            new_text = rewrite_bot_object_mentions(self, strip_citations(raw_text or ""))
            if not new_text.strip():
                return {"ok": False, "error": "empty_new_text",
                        "message": "The complete corrected replacement text is required."}
            note = strip_citations(raw_note or "").strip()
            if not note:
                return {"ok": False, "error": "empty_correction_note",
                        "message": "A specific correction note is required."}
            # §2: authorized iff the ts is a KEY of this round's mapping. Missing/malformed
            # mapping is the EMPTY mapping — fail closed, never open.
            targets = getattr(ctx, "authorized_edit_targets", None)
            if not isinstance(targets, Mapping):
                targets = {}
            target = targets.get(message_ts)
            if target is None or getattr(target, "channel_id", None) != channel_id:
                return dict(self._EDIT_UNAUTHORIZED)
            turn = getattr(ctx, "turn", None)
            # §7: ONE edit attempt per turn, reserved atomically BEFORE any await — a second
            # distinct call id is refused here (a duplicate of the SAME id never reaches this
            # method twice; the tool flight serves it the first call's outcome).
            if turn is not None:
                if getattr(turn, "_edit_attempt_reserved", False):
                    return {"ok": False, "error": "edit_already_attempted",
                            "message": ("This turn already attempted an edit — one edit per "
                                        "turn. Post a normal correction message instead if "
                                        "more is needed.")}
                turn._edit_attempt_reserved = True
            team_id = getattr(self, "self_team_id", None)
            lease = getattr(turn, "send_lease", None) if turn is not None else None
            receipts = getattr(turn, "receipt_ledger", None) if turn is not None else None

            # §11.19: the EditRecord is PRE-CREATED before the announcement send, so the
            # acceptance observer below performs ASSIGNMENTS ONLY — and the post-send in-lock
            # verification can repair whatever a swallowed callback crash cut short.
            record = EditRecord(channel_id=channel_id, target_ts=message_ts,
                                announcement_ts="")

            def _observe(ts: str) -> None:
                """§11.15/§11.19: the acceptance seam. Slack has taken the disclosure THIS
                instant — inside send_message, BEFORE its receipt bookkeeping can raise — so
                the durable facts land here, not after the send returns (§11.10 crash order).
                Assignments only, in this order: the `turn.edits` append, the acceptance
                flags, and `accepted["record"]` LAST — `_note_first_accept` swallows callback
                exceptions, and with the phase marker set last a swallowed crash can never
                leave the transaction believing the accounting succeeded."""
                if turn is not None:
                    turn.edits.append(record)
                record.announcement_ts = ts
                if turn is not None:
                    turn.visible_action_committed = True
                accepted["ts"] = ts
                accepted["record"] = record

            def _partial(record: Any, code: str, announcement_ts: str) -> dict:
                """§5: the announcement landed and the overwrite did not. The model must NOT
                post a second correction — the disclosure already carries it. Every partial
                carries the CV10 join fact too: a landed disclosure ALWAYS has a committed
                correction_announcement destination (review r5), committed here best-effort
                when the normal path's commit never ran."""
                record.error = code
                if turn is not None and announcement_ts:
                    try:
                        already = any(
                            getattr(d, "first_ts", None) == announcement_ts
                            for d in getattr(turn, "destinations", ()) or ())
                        if not already:
                            # The synthesized disclosure text lives in the transaction body's
                            # scope; on these exception paths the best-effort record carries
                            # the note-derived reconstruction (the join key is the ts).
                            turn.mark_destination_committed(
                                first_ts=announcement_ts,
                                kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                                text=f"Correction to my earlier message: {note}",
                                complete=True, channel_id=channel_id,
                                thread_root_ts=announcement_thread.get("ts"))
                    except Exception as e:  # noqa: BLE001 — never fail a delivered disclosure
                        self.log_debug(
                            f"edit_own_message: could not record the announcement "
                            f"destination on the partial path: {e}")
                return {"ok": False, "error": code, "announcement_posted": True,
                        "edited": False,
                        "message_ts": message_ts, "announcement_ts": announcement_ts,
                        "message": ("The correction notice is already posted in that thread; "
                                    "the message itself was not changed. Do NOT post another "
                                    "correction.")}

            async def _transaction() -> dict:
                async with _edit_transaction_locked(team_id, channel_id, message_ts):
                    # §11.14: the post-acceptance backstop lives INSIDE the lock — its
                    # accounting is part of the transaction, so no second contender can
                    # interleave with a partial that is still being written down.
                    try:
                        return await self._edit_transaction_body(
                            ctx, team_id, channel_id, message_ts, target, turn, lease,
                            receipts, new_text, note, announcement_thread, accepted, record,
                            _observe, _partial)
                    except StaleSendSuppressed:
                        raise
                    except Exception as e:  # noqa: BLE001 — phase-aware, in-lock (§11.14)
                        if accepted["record"] is None:
                            # Pre-announcement: Slack is untouched — let the outer §11.14
                            # backstop answer `edit_failed` (or its named refusals).
                            raise
                        self.log_error(
                            f"edit_own_message: failed after the announcement for "
                            f"{channel_id}/{message_ts}: {e}")
                        return _partial(record, "edit_failed_after_announcement",
                                        accepted["ts"])

            # LEASED, and the lock is INSIDE the leased body: a caller that stops waiting
            # does not stop the effect, and the lock is only released once the shielded
            # transaction has actually finished its accounting — the in-lock backstop
            # included.
            return await run_effect(turn, "edit_own_message", _transaction)
        except LaunchNotRecorded as e:
            self.log_error(
                f"edit_own_message: launch not recorded for {channel_id}/{message_ts}: {e}")
            return {"ok": False, "error": "launch_not_recorded",
                    "message": ("Something went wrong before the correction notice was posted, "
                                "so nothing was changed.")}
        except EffectRevoked:
            self.log_warning(
                f"edit_own_message: refused for {channel_id}/{message_ts} — the turn was cut "
                "short")
            return {"ok": False, "error": "turn_cancelled",
                    "message": ("This turn was cut short before the correction went out, so "
                                "nothing was posted and nothing was edited.")}
        except StaleSendSuppressed:
            # The conversation moved on before the disclosure landed. Nothing was attempted —
            # re-raised UNCHANGED so the turn's own handler files it (matches post_to_thread).
            raise
        except Exception as e:  # noqa: BLE001 — a tool bug must not kill the response
            # §11.8/§11.14: PHASE-AWARE, and it guards from the FIRST line of the executor —
            # argument coercion included, so model-supplied junk never surfaces as a raw
            # traceback. After the disclosure was accepted a bare failure would invite a
            # duplicate correction, so the partial contract holds (normally already answered
            # by the in-lock backstop; kept here as defense in depth).
            record = accepted["record"]
            if record is not None:
                self.log_error(
                    f"edit_own_message: failed after the announcement for "
                    f"{channel_id}/{message_ts}: {e}")
                return _partial(record, "edit_failed_after_announcement", accepted["ts"])
            # Before the announcement, Slack is untouched: `edit_failed` is the §3
            # pre-announcement backstop, and the model may fall back to a normal correction.
            self.log_error(f"edit_own_message: failed for {channel_id}/{message_ts}: {e}")
            return {"ok": False, "error": "edit_failed",
                    "message": "Could not edit that message."}

    async def _edit_transaction_body(self, ctx, team_id, channel_id: str, message_ts: str,
                                     target, turn, lease, receipts, new_text: str, note: str,
                                     announcement_thread: Dict[str, Optional[str]],
                                     accepted: Dict[str, Any], record, _observe,
                                     _partial) -> dict:
        """The §5 two-effect transaction body, run under the keyed lock by
        `execute_edit_own_message` (which owns the §11.14 backstops around it). `record` is
        the §11.19 pre-created EditRecord the `_observe` acceptance seam assigns into."""
        # -- preflight (§5 steps 1-4): everything read and validated before either
        # -- mutation, under the same lock that covers the mutations.
        row = await self._edit_preflight_receipt(ctx, team_id, channel_id, message_ts)
        if row is None:
            return dict(self._EDIT_UNAUTHORIZED)
        thread_root = row.get("thread_root_ts") or None
        live = await self._read_exact_message(channel_id, message_ts, thread_root)
        if live is None:
            return {"ok": False, "error": "stale_target",
                    "message": ("That message could not be re-read as it stood — it "
                                "may have changed or been deleted. Post a normal "
                                "correction message instead.")}
        if not self.is_own_message(live):
            return dict(self._EDIT_UNAUTHORIZED)
        if self._live_edited_ts(live) != getattr(target, "edited_ts", None):
            return {"ok": False, "error": "stale_target",
                    "message": ("That message changed after it was shown to you. Post "
                                "a normal correction message instead.")}
        shape = self._classify_edit_shape(live)
        if shape is None:
            return {"ok": False, "error": "unsupported_message_shape",
                    "message": ("That message carries files, attachments or blocks an "
                                "edit cannot faithfully preserve. Post a normal "
                                "correction message instead.")}
        shape_kind, footer_actions = shape
        permalink = await self._edit_permalink(ctx, channel_id, message_ts)
        if not permalink:
            return {"ok": False, "error": "permalink_failed",
                    "message": ("Could not link the target message for the public "
                                "correction notice, so nothing was changed. Post a "
                                "normal correction message instead.")}
        live_text = live.get("text") or ""
        prefix, _body, suffix = extract_continuation_markers(live_text)
        # §11.17: `new_text` and `note` arrive citation-stripped — the validation stripped
        # them BEFORE the emptiness checks, so there is nothing left to strip here.
        formatted_body = self.format_text(new_text)
        replacement = f"{prefix}{formatted_body}{suffix}"
        if replacement == live_text:
            return {"ok": False, "error": "no_change",
                    "message": "The replacement is identical to the standing text."}
        if shape_kind == "footer":
            # §6: the inline-footer UX caps are the SEND path's — a replacement that no
            # longer fits the section cannot ride this shape.
            if (len(formatted_body) > self._SECTION_TEXT_LIMIT
                    or len(formatted_body) > self._FOOTER_INLINE_MAX
                    or formatted_body.count("\n") > 2):
                return {"ok": False, "error": "edit_too_long_for_inline_footer",
                        "message": ("The replacement no longer fits the inline-footer "
                                    "message shape. Post a normal correction message "
                                    "instead.")}
        if len(replacement) > self.MAX_MESSAGE_LENGTH:
            # NEVER split and never truncate an edit — the message is one message.
            return {"ok": False, "error": "replacement_too_long",
                    "message": (f"The replacement exceeds one Slack message "
                                f"({self.MAX_MESSAGE_LENGTH} chars). Post a normal "
                                "correction message instead.")}
        # §5: the executor SYNTHESIZES the disclosure, so it can never be omitted or
        # weakened — and it must fit ONE message; it never enters the split path.
        announcement = f"Correction to [my earlier message]({permalink}): {note}"
        formatted_announcement = self.format_text(announcement)
        if len(formatted_announcement) > self.MAX_MESSAGE_LENGTH:
            return {"ok": False, "error": "announcement_too_long",
                    "message": ("The correction note is too long for one message — "
                                "shorten it, or post a normal correction message "
                                "instead.")}
        announcement_thread["ts"] = thread_root or message_ts
        # -- announcement FIRST (§5): the only order where a silent edit is
        # -- unreachable. Launch is recorded immediately before the disclosure post.
        mark_tool_launched(ctx)
        send_meta: dict = {}
        try:
            posted = await self.send_message(
                channel_id, cast(str, announcement_thread["ts"]), announcement, lease=lease,
                surface="edit_own_message", meta_out=send_meta, on_first_accept=_observe,
                receipts=receipts, receipt_kind="finalized",
                receipt_class="correction_announcement")
        except StaleSendSuppressed:
            raise
        except Exception as send_error:  # noqa: BLE001 — §11.16/§11.21 exhaustive
            if accepted["record"] is not None:
                # Slack ACCEPTED the disclosure (the seam fired), then the send's own
                # bookkeeping raised — the record stands (§11.15/§11.19) and the in-lock
                # backstop answers with the partial contract, never a bare failure.
                raise
            delivery = send_meta.get("delivery")
            accepted_ts = getattr(delivery, "first_ts", None) if delivery is not None else None
            if accepted_ts:
                # The COMBINED failure (§11.19 review r4): Slack accepted the disclosure but
                # the acceptance callback's own accounting crashed (swallowed by
                # _note_first_accept, so accepted["record"] never set) AND the send then
                # raised. A visible announcement must never escape accounting or read as
                # announcement_failed — repair in-lock and answer with the partial contract.
                record.announcement_ts = str(accepted_ts)
                if turn is not None and record not in turn.edits:
                    turn.edits.append(record)
                if turn is not None:
                    turn.visible_action_committed = True
                accepted["ts"] = str(accepted_ts)
                accepted["record"] = record
                self.log_error(
                    f"edit_own_message: acceptance accounting repaired after combined "
                    f"callback+send failure for {channel_id}/{message_ts}: {send_error}")
                return _partial(record, "edit_failed_after_announcement", str(accepted_ts))
            if (isinstance(send_error, (_EPOCH_REFUSED,) + _AMBIGUOUS_SEND_EXC)
                    or send_meta.get("transport_ambiguous")):
                # §11.21: the notice's physical outcome is UNKNOWN — an epoch refusal at
                # the announcement, a transport error after dispatch (SlackApiError, a
                # timeout) or a reconciliation miss. The transaction STOPS: no update
                # (silent-edit safety), no EditRecord, and the model is told the notice
                # may already be visible.
                self.log_error(
                    f"edit_own_message: announcement outcome unknown for "
                    f"{channel_id}/{message_ts}: {send_error}")
                return {"ok": False, "error": "announcement_outcome_unknown",
                        "message": ("The correction notice may or may not have posted — its "
                                    "outcome is unknown — and the message was NOT edited. Do "
                                    "not retry the edit; if a correction is still needed, post "
                                    "at most ONE normal follow-up message.")}
            # §11.21: every OTHER exception is deterministic-local — raised before anything
            # could reach Slack — and maps to the §5 announcement_failed row: Slack
            # untouched, and the model may fall back to a normal correction message.
            self.log_error(
                f"edit_own_message: announcement failed for "
                f"{channel_id}/{message_ts}: {send_error}")
            return {"ok": False, "error": "announcement_failed",
                    "message": ("The public correction notice could not be posted, so "
                                "the message was NOT edited. Post a normal correction "
                                "message instead.")}
        if not posted:
            return {"ok": False, "error": "announcement_failed",
                    "message": ("The public correction notice could not be posted, so "
                                "the message was NOT edited. Post a normal correction "
                                "message instead.")}
        # §11.19: in-lock VERIFICATION-AND-REPAIR, before any update. `_note_first_accept`
        # swallows callback exceptions, so the `_observe` accounting is PROVED here rather
        # than trusted: whatever assignment a swallowed crash cut short is re-made (each is
        # idempotent), and a transport that reported success without ever firing the seam
        # is accounted the same way.
        if not record.announcement_ts:
            record.announcement_ts = posted
        if turn is not None:
            if record not in turn.edits:
                self.log_error(
                    f"edit_own_message: acceptance accounting for {channel_id}/"
                    f"{message_ts} was cut short by a swallowed callback error — "
                    f"repaired in-lock before the update")
                turn.edits.append(record)
            turn.visible_action_committed = True
        if accepted["ts"] is None:
            accepted["ts"] = record.announcement_ts
        accepted["record"] = record
        if turn is not None:
            try:
                # Best-effort destination OBSERVATION — moved out of the observer (§11.19
                # allows it assignments only); the committed destination below is the
                # durable record.
                turn.note_destination_observed(
                    channel_id=channel_id, first_ts=posted,
                    kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                    thread_root_ts=announcement_thread.get("ts"))
            except Exception as e:  # noqa: BLE001 — bookkeeping never fails a post
                self.log_debug(f"edit_own_message: could not observe the announcement: {e}")
        try:
            # Only the destination/telemetry bookkeeping stays best-effort: the
            # committed destination (kind correction_announcement, first_ts = its own
            # ts — the CV10 join key) must never fail a delivered disclosure.
            landed = send_meta.get("delivery")
            turn.mark_destination_committed(
                first_ts=posted, kind=DEST_KIND_CORRECTION_ANNOUNCEMENT,
                text=landed.text if landed is not None else formatted_announcement,
                complete=landed.complete if landed is not None else True,
                channel_id=channel_id, thread_root_ts=announcement_thread["ts"])
        except AttributeError:
            pass  # no turn (a hand-built context): nothing to account on
        except Exception as e:  # noqa: BLE001 — never fail a delivered disclosure
            self.log_debug(f"edit_own_message: could not record the announcement: {e}")
        # -- post-announcement SECOND exact read (§5): the target must still be the
        # -- message that was proved, or nothing is overwritten.
        second = await self._read_exact_message(channel_id, message_ts, thread_root)
        if (second is None
                or self._live_edited_ts(second) != getattr(target, "edited_ts", None)):
            return _partial(record, "stale_target_after_announcement", posted)
        # -- epoch authorization + the DEDICATED update ------------------------------
        rebuilt_blocks = None
        if shape_kind == "footer":
            rebuilt_blocks = [{"type": "section",
                              "text": {"type": "mrkdwn", "text": formatted_body}}
                              ] + [footer_actions]
        try:
            await self._update_edited_message(channel_id, message_ts, replacement,
                                              blocks=rebuilt_blocks)
        except _EPOCH_REFUSED:
            return _partial(record, "epoch_refused_after_announcement", posted)
        except SlackApiError as e:
            # Definitive API rejection: the correction notice stays visible — its
            # wording is true either way — and there is no compensating deletion.
            self.log_warning(
                f"edit_own_message: update failed for {channel_id}/{message_ts}: {e}")
            return _partial(record, "update_failed_after_announcement", posted)
        except Exception as e:  # noqa: BLE001 — transport-ambiguous: reconcile, once
            reconciled = await self._reconcile_uncertain_update(
                channel_id, message_ts, thread_root, shape_kind, replacement,
                rebuilt_blocks)
            if not reconciled:
                self.log_warning(
                    f"edit_own_message: update outcome unknown for "
                    f"{channel_id}/{message_ts}: {e}")
                return _partial(record, "update_outcome_unknown_after_announcement",
                                posted)
        record.state = EDIT_STATE_COMMITTED
        record.error = None
        return {"ok": True, "message_ts": message_ts, "announcement_ts": posted,
                "announcement_posted": True, "edited": True}

    async def _edit_preflight_receipt(self, ctx, team_id, channel_id: str,
                                      message_ts: str) -> Optional[dict]:
        """§5 preflight step 1, receipt half: re-read the durable row and require finalized
        `assistant_reply` — a legacy NULL-class row, chrome, and an unreadable DB all answer
        None (fail closed, reveal nothing)."""
        db = getattr(ctx, "db", None)
        if db is None or not team_id:
            return None
        try:
            payload = await db.read_channel_sidecars_for_async(team_id, channel_id,
                                                               [message_ts])
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"edit_own_message: receipt re-read failed for {message_ts}: {e}")
            return None
        for row in (payload or {}).get("receipts") or ():
            if isinstance(row, dict) and str(row.get("message_ts")) == message_ts:
                if (str(row.get("state")) == "finalized"
                        and row.get("receipt_class") == "assistant_reply"):
                    return row
                return None
        return None

    async def _read_exact_message(self, channel_id: str, message_ts: str,
                                  thread_root_ts: Optional[str]) -> Optional[dict]:
        """§5: one EXACT Slack read of one message — the preflight read and the
        post-announcement second read are both this. None on any failure or a missing ts."""
        try:
            if thread_root_ts and thread_root_ts != message_ts:
                resp = await self.app.client.conversations_replies(
                    channel=channel_id, ts=thread_root_ts, latest=message_ts,
                    oldest=message_ts, inclusive=True, limit=1)
            else:
                resp = await self.app.client.conversations_history(
                    channel=channel_id, latest=message_ts, oldest=message_ts, inclusive=True,
                    limit=1)
        except Exception as e:  # noqa: BLE001 — an unreadable target authorizes nothing
            self.log_warning(f"edit_own_message: exact read failed for {message_ts}: {e}")
            return None
        for msg in (resp.get("messages") or ()) if resp else ():
            if isinstance(msg, dict) and msg.get("ts") == message_ts:
                return msg
        return None

    @staticmethod
    def _live_edited_ts(message: dict) -> Optional[str]:
        edited = message.get("edited")
        ts = edited.get("ts") if isinstance(edited, dict) else None
        return str(ts) if ts else None

    @staticmethod
    def _classify_edit_shape(live: dict) -> Optional[tuple]:
        """§6: exactly two supported shapes, everything else refused.

        ("plain", None) — no files/attachments, and either NO blocks or exactly ONE
        Slack-minted `rich_text` mirror block (§11.25: Slack materializes one on every plain
        chat.postMessage, so live read-back never shows a block-less plain message): the
        update replaces formatted `text` and Slack remints the mirror.
        ("footer", actions_block) — optional leading mrkdwn section + trailing actions block:
        rebuilt as [new section] + the EXISTING actions block, byte-for-byte. STRICT (§11.4):
        the section may carry ONLY `type`/`text` (+ the Slack-minted `block_id`) with mrkdwn
        text — an `accessory`, `fields` or any other decoration cannot be faithfully rebuilt;
        the actions block must carry EXACTLY ONE element, a BUTTON with action_id
        `open_channel_settings` — duplicates and other element types are unsupported. None —
        unsupported (files, attachments, rich-text or research blocks, unknown actions): never
        silently dropped, never retained around a contradictory edit."""
        if live.get("files") or live.get("attachments"):
            return None
        blocks = live.get("blocks") or []
        if not blocks:
            return "plain", None
        # §11 live amendment (2026-08-05): Slack MATERIALIZES a `rich_text` block on every
        # plain chat.postMessage, so "no blocks" never exists in read-back — the first live
        # edit refused a one-line plain note as unsupported_message_shape. ONE rich_text
        # block IS the plain shape: a text-only chat.update makes Slack regenerate it, which
        # is exactly the replacement semantics the plain shape wants. Anything beyond that
        # single Slack-minted mirror stays unsupported.
        if len(blocks) == 1 and isinstance(blocks[0], dict) \
                and blocks[0].get("type") == "rich_text":
            return "plain", None
        if len(blocks) == 1:
            section, actions = None, blocks[0]
        elif len(blocks) == 2:
            section, actions = blocks[0], blocks[1]
        else:
            return None
        if section is not None:
            if not isinstance(section, dict) or section.get("type") != "section":
                return None
            if not set(section) <= {"type", "text", "block_id"}:
                return None
            section_text = section.get("text")
            if (not isinstance(section_text, dict)
                    or section_text.get("type") != "mrkdwn"):
                return None
        if not isinstance(actions, dict) or actions.get("type") != "actions":
            return None
        elements = actions.get("elements") or []
        if len(elements) != 1:
            return None
        element = elements[0]
        if (not isinstance(element, dict) or element.get("type") != "button"
                or element.get("action_id") != "open_channel_settings"):
            return None
        return "footer", actions

    async def _edit_permalink(self, ctx, channel_id: str, message_ts: str) -> Optional[str]:
        """§5 preflight step 3: the permalink for the synthesized disclosure, via the existing
        history-tool chat.getPermalink path. Any failure is None → `permalink_failed`, before
        anything posts."""
        try:
            resp = await self.get_message_permalink_tool(channel_id, message_ts, ctx=ctx)
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"edit_own_message: permalink failed for {message_ts}: {e}")
            return None
        if isinstance(resp, dict) and resp.get("ok") is True and resp.get("permalink"):
            return str(resp["permalink"])
        return None

    async def _update_edited_message(self, channel_id: str, message_ts: str, text: str,
                                     blocks: Optional[list] = None) -> None:
        """§6/§7: the DEDICATED updater — NOT update_message and NOT the streaming updater
        (wrong formatting, shape checks, reconciliation and epoch behavior). The caller has
        already formatted and validated `text`; epoch authorization runs HERE, immediately
        before the write, because generic updates don't check it and this one must. Raises —
        the transaction owns the outcome classification."""
        _epoch_authorize(self, channel_id, "edit_own_message:update")
        kwargs: Dict[str, Any] = dict(channel=channel_id, ts=message_ts, text=text,
                                      mrkdwn=True)
        if blocks is not None:
            kwargs["blocks"] = blocks
        await self.app.client.chat_update(**kwargs)  # unleased-ok: inside the edit_own_message transaction's own run_effect lease

    async def _reconcile_uncertain_update(self, channel_id: str, message_ts: str,
                                          thread_root_ts: Optional[str], shape_kind: str,
                                          replacement: str,
                                          rebuilt_blocks: Optional[list]) -> bool:
        """§5's ambiguous-update row: re-fetch the target and compare CONTENT — the exact
        formatted text for a plain reply, the exact fallback text plus the exact rebuilt blocks
        for a footer reply — NEVER Slack-generated `edited.ts`, and no blind retry. True means
        the update landed (reconciled success). Block equality is "exact modulo the Slack-minted
        `block_id`s" — §11.7 rules that stripping as the meaning of "exact rebuilt blocks"."""
        fetched = await self._read_exact_message(channel_id, message_ts, thread_root_ts)
        if fetched is None:
            return False
        if (fetched.get("text") or "") != replacement:
            return False
        if shape_kind == "footer":
            return _blocks_match(fetched.get("blocks"), rebuilt_blocks)
        return True

    def get_no_reply_tool_schema(self) -> dict:
        """Function-tool schema for the F2 terminal no-reply action (silence-capable turns only).

        The reason is a CLOSED enum (message_processor.terminal_actions), not prose: it is the
        one column that says why this bot stays quiet, and free text cannot be counted. An
        unrecognized value is a rejected call, never a silent rewrite to `other`."""
        from message_processor.terminal_actions import (SILENCE_REASONS,
                                                        render_reason_guide)
        return {
            "type": "function",
            "name": "no_response_needed",
            "description": (
                "End this turn without posting a normal text reply in the current "
                "conversation. This is TERMINAL: it ends the turn, so call it instead of "
                "replying, never after writing one. It does not cancel other tools you call "
                "in the same round — a reaction, a memory write or another surface still "
                "happens; it only means you add no words here. Never use it to wait for work "
                "you dispatched yourself: finish that work and report it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": list(SILENCE_REASONS),
                        "description": render_reason_guide(),
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        }

    async def execute_no_reply_tool(self, ctx, args: dict) -> dict:
        """Executor for no_response_needed. Terminal signal only — the tool loop stops the
        turn and the handler surfaces the outcome; nothing is posted here.

        The route's authorization now lives HERE rather than in the schema gate. On the channel
        surface the tool is statically exposed (a per-turn schema is a cache fork), so the only
        remaining place that can tell an owed-words turn from a silence-capable one is the turn
        itself. Fails CLOSED: an absent turn, or one that owes words, is refused, and the loop
        keeps going rather than swallowing a reply somebody is waiting for."""
        if not config.enable_no_reply_tool:
            return {"ok": False, "error": "disabled",
                    "message": "Ending a turn without a reply is disabled here."}
        turn = getattr(ctx, "turn", None)
        if turn is None or not getattr(turn, "silence_capable", False):
            return {
                "ok": False, "error": "reply_owed",
                "message": ("Not run: this turn owes a reply — you were addressed directly, so "
                            "silence is not available. Answer normally."),
            }
        return {"ok": True}

    async def update_message_streaming(self, channel_id: str, message_id: str, text: str,
                                       lease: Any = None,
                                       surface: str = "legacy_update",
                                       receipts: Any = None) -> Dict:
        """Updates a message with rate limit awareness.

        `lease` (stale guard): the caller passes it only for the FIRST edit that turns a
        thinking placeholder into answer text — that edit is when the room first reads an
        answer. Subsequent edits grow a surface that already exists and are never guarded.

        `receipts`: passed by the ANSWER-bearing callers only. The edited surface stops being
        chrome at that moment, so it is promoted (same owner) and settles with the turn.
        Status/phase callers pass nothing and the surface stays excluded."""
        if lease is not None:
            lease.authorize(surface)
        try:
            # Strip MCP citations from text before sending to Slack
            # This is the single point of control for all streaming updates
            text = strip_citations(text)
            text = rewrite_bot_object_mentions(self, text)

            # For messages that already contain Slack mrkdwn (like enhanced prompts with _italics_),
            # skip the markdown conversion to avoid double-processing
            if text.startswith("✨") or text.startswith("*Enhanced Prompt:*") or text.startswith("Enhancing your prompt:"):
                # This is an enhanced prompt - it already has proper Slack formatting
                formatted_text = text
            else:
                # Format text for Slack using markdown conversion
                formatted_text = self.format_text(text)
            
            # More aggressive truncation for streaming to avoid msg_too_long errors
            # Account for Slack's markdown expansion and special characters
            safe_length = self.MAX_MESSAGE_LENGTH - 200  # More buffer for safety
            if len(formatted_text) > safe_length:
                # Try to truncate at a reasonable boundary (code block or paragraph)
                truncated = formatted_text[:safe_length]
                
                # If we're in the middle of a code block, close it
                if truncated.count('```') % 2 == 1:
                    truncated += '\n```'
                
                formatted_text = truncated + continuation_trailer()
            
            # Call Slack API's chat_update method
            result = await self.app.client.chat_update(  # unleased-ok: inside update_message_streaming, which already authorized at its entry
                channel=channel_id,
                ts=message_id,
                text=formatted_text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            
            if receipts is not None:
                try:
                    await receipts.promote(message_id)
                except Exception as e:  # noqa: BLE001
                    self.log_debug(f"receipt promote failed for {message_id}: {e}")
            # The placeholder now reads as an answer — a first surface for the guard's
            # purposes, so the rest of this turn's edits proceed without rechecking.
            if lease is not None:
                lease.commit()
            # Return success status
            return {
                "success": True,
                "rate_limited": False,
                "retry_after": None,
                "result": result
            }
            
        except SlackApiError as e:
            # Handle msg_too_long error specifically
            if e.response.get('error') == 'msg_too_long':
                # Backstop only: the streaming handler (handlers/text.py) triggers its own
                # overflow well before this (message_char_limit << 3700) and owns Part-N
                # creation. If we still land here, this single chunk is over the API limit
                # and there is no safe way to post the remainder from the messaging layer —
                # we have no thread_ts, and an out-of-band post would race the handler's
                # own chunk bookkeeping. So truncate HONESTLY: never promise a "next
                # message" that nothing will post, and log the dropped tail loudly.
                dropped = max(0, len(formatted_text) - 2000)
                self.log_warning(
                    f"update_message_streaming: msg_too_long backstop hit — truncating, "
                    f"~{dropped} chars dropped (no continuation is posted from here)")
                very_short = formatted_text[:2000].rstrip() + "\n\n*[truncated — too long for Slack]*"
                if very_short.count('```') % 2 == 1:
                    very_short += '\n```'
                
                try:
                    result = await self.app.client.chat_update(  # unleased-ok: the retry of that same already-authorized edit
                        channel=channel_id,
                        ts=message_id,
                        text=very_short,
                        mrkdwn=True
                    )
                    return {
                        "success": True,
                        "rate_limited": False,
                        "retry_after": None,
                        "result": result
                    }
                except Exception:
                    # If even the short version fails, just acknowledge the error
                    self.log_error("Even truncated message failed to send")
                    raise
            
            # Handle 429 rate limit responses
            elif e.response.status_code == 429:
                # Extract retry-after header
                retry_after = None
                if hasattr(e.response, 'headers') and 'Retry-After' in e.response.headers:
                    try:
                        retry_after = int(e.response.headers['Retry-After'])
                    except (ValueError, KeyError):
                        retry_after = None
                
                self.log_warning("🚨🚨🚨 HIT RATE LIMIT 429 🚨🚨🚨")
                
                return {
                    "success": False,
                    "rate_limited": True,
                    "retry_after": retry_after,
                    "error": str(e)
                }
            else:
                # Handle other API errors
                self.log_error(f"Error updating message in streaming: {e}")
                return {
                    "success": False,
                    "rate_limited": False,
                    "retry_after": None,
                    "error": str(e)
                }
        except Exception as e:
            # Handle unexpected errors
            self.log_error(f"Unexpected error updating message in streaming: {e}")
            return {
                "success": False,
                "rate_limited": False,
                "retry_after": None,
                "error": str(e)
            }

    def _build_response_footer_blocks(self, model: Optional[str]) -> list:
        """Footer: a single compact row — one small button carrying the model name that opens
        the per-channel settings modal (handled by the ``open_channel_settings`` action)."""
        model_label = model or config.gpt_model
        return [
            {"type": "actions", "elements": [
                {"type": "button",
                 "text": {"type": "plain_text", "text": f"⚙️ {model_label}"},
                 "action_id": "open_channel_settings"}
            ]},
        ]

    def attachable_footer_blocks(self, channel_id: Optional[str], model: Optional[str] = None):
        """Settings chrome to ATTACH to the final part of a native-streamed response
        (chat.stopStream accepts blocks), so the "⚙️ <model>" row rides the response
        message itself instead of a separate trailing post — on EVERY surface
        (user request 2026-07-10; matches Claude's per-message footer row).

        Surface routing: channels/channel threads get the per-channel settings button
        (open_channel_settings); DMs/assistant threads get the personal settings
        button (open_user_settings) since there are no channel settings there. This
        is independent of the feedback strip (ENABLE_FEEDBACK_BUTTONS), which stays
        off by the operator's choice.

        Returns None when ENABLE_RESPONSE_FOOTER is off. Fallback for paths that
        can't attach: channels may still post the separate footer message
        (maybe_post_response_footer); DMs simply get no gear — /chatgpt-settings
        always works."""
        if not channel_id:
            return None
        if not getattr(config, "enable_response_footer", True):
            return None
        if channel_id.startswith("D"):
            model_label = model or config.gpt_model
            return [
                {"type": "actions", "elements": [
                    {"type": "button",
                     "text": {"type": "plain_text", "text": f"⚙️ {model_label}"},
                     "action_id": USER_SETTINGS_ACTION_ID}
                ]},
            ]
        return self._build_response_footer_blocks(model)

    async def maybe_post_response_footer(self, message, response, receipts: Any = None) -> None:
        """Trailing chrome under a final text response — surface-dependent:

        - Channels: the Phase 7 footer (model + Configure button), when
          ENABLE_RESPONSE_FOOTER is on.
        - DMs/assistant threads: the Phase H native feedback buttons strip, when
          ENABLE_FEEDBACK_BUTTONS is on (channels deliberately get no feedback strip —
          pixels matter there; reactions cover feedback in channels).

        Posted as a SEPARATE trailing message, so it never touches the text/split/streaming
        path and is inherently attached only after the final part (streamed included).
        Fires once, only for text responses. Never raises.
        """
        try:
            if not response or getattr(response, "type", None) != "text":
                return
            # Reaction-only turns post no message — nothing to hang chrome under
            if not (getattr(response, "content", None) or "").strip():
                return
            # The chrome already rode the response message itself (native streaming
            # attaches it on stopStream) — don't double up with a separate post.
            if (getattr(response, "metadata", None) or {}).get("footer_attached"):
                return
            channel_id = getattr(message, "channel_id", None)
            if not channel_id:
                return
            if channel_id.startswith("D"):
                # Phase H: feedback buttons + "⚙️ <model>" (user settings) on the
                # assistant/DM surface.
                if not feedback_enabled():
                    return
                # The whole strip (feedback thumbs + "⚙️ <model>" settings button)
                # posts ONCE, under the first reply of a thread — later replies get
                # no trailing chrome at all (user feedback 2026-07-09: per-message
                # buttons are bulky; a hyperlink can't open a modal — no trigger_id).
                thread_ts = getattr(message, "thread_id", None)
                if not should_offer_feedback(channel_id, thread_ts):
                    return
                model = (getattr(response, "metadata", None) or {}).get("model")
                # A refusal must return, not raise: this whole method sits inside a broad
                # `except Exception` that logs at DEBUG, so a raise here would be swallowed
                # where nobody looks.
                if _epoch_refused(self, channel_id, "feedback_footer"):
                    return
                posted = await self.app.client.chat_postMessage(  # unleased-ok: the settings footer, which only ever follows an answer already posted
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="Rate this response",  # fallback text for notifications
                    blocks=build_feedback_blocks(model),
                )
                await self._record_receipt(
                    channel_id, posted.get("ts") if posted else None, receipts=receipts,
                    receipt_kind="chrome", receipt_class="chrome",
                    thread_root_ts=thread_ts, site="feedback_footer")
                return
            # Channels: per-channel settings footer.
            if not getattr(config, "enable_response_footer", True):
                return
            model = (getattr(response, "metadata", None) or {}).get("model")
            blocks = self._build_response_footer_blocks(model)
            thread_ts = getattr(message, "thread_id", None)
            if _epoch_refused(self, channel_id, "response_footer"):
                return
            posted = await self.app.client.chat_postMessage(  # unleased-ok: the same footer on the fallback path
                channel=channel_id,
                thread_ts=thread_ts,
                # Describes the footer's purpose instead of showing a bare model name (which reads
                # as a spurious standalone message — the "gpt-5.6-sol" post seen live 2026-07-16).
                text=RESPONSE_FOOTER_FALLBACK_TEXT,
                blocks=blocks,
            )
            await self._record_receipt(
                channel_id, posted.get("ts") if posted else None, receipts=receipts,
                receipt_kind="chrome", receipt_class="chrome",
                thread_root_ts=thread_ts, site="response_footer")
        except Exception as e:
            self.log_debug(f"Could not post response footer: {e}")
