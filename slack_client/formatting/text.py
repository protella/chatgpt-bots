from __future__ import annotations

import re
from typing import List

from config import config
from slack_client._host import _Host

# A real Slack mention id is a user/workspace/bot id: U/W/B + uppercase alnum.
_SLACK_ID_RE = re.compile(r'^[UWB][A-Z0-9]{6,}$')
# Inner of a <@...> user mention (may carry an inline "|label").
_MENTION_RE = re.compile(r'<@([^>]+)>')
# Standalone "@Name" token (letters first), not already inside a <@...> and not an email local part.
_AT_TOKEN_RE = re.compile(r"(?<![<\w])@([A-Za-z][\w.'-]*)")
# A trailing "@" or "@Name…" a streamed delta may still extend, and the characters that keep an
# "@" from being a mention when they precede it (the lookbehind above).
_AT_CANDIDATE_RE = re.compile(r"@(?:[A-Za-z][\w.'-]*)?")
_NO_MENTION_BEFORE_RE = re.compile(r"[<\w]")
# Past this a trailing "@token" is prose, not a name being typed; holding it would stall a stream.
_AT_HOLD_MAX = 80


def extract_mention_ids(text):
    """All user/bot ids <@…>-mentioned in raw inbound text (labels stripped, order kept).

    Used by the event handlers to warm the user cache BEFORE resolve_inbound_mentions,
    so a first-ever mention of any user or bot (e.g. a co-resident assistant) renders
    as "@Name" instead of hitting the unknown-id fallback."""
    if not text:
        return []
    return [m.group(1).split('|', 1)[0].strip() for m in _MENTION_RE.finditer(text)]


def resolve_inbound_mentions(text, user_cache=None, bot_user_id=None, bot_handle=None):
    """Resolve raw Slack <@ID> mentions in inbound text into a readable form.

    - the bot's own mention renders as "@<bot_handle>" (the handle auth.test returned), so a
      draft the user wants proofread keeps the tag it carries; routing facts (mentioned_self,
      name hits) are computed from the RAW text and never from this rendering
    - other users resolve to "@DisplayName" when known in user_cache (so the model sees who
      was addressed) or to the inline "|label" if present
    - unknown ids render as "@<id>" — NEVER silently stripped. Deleting an unresolved
      mention destroys the addressee signal: "<@Claude's-id> can you…" became "can you…",
      and the participation classifier read the question as aimed at THIS bot (live
      misfire 2026-07-11). A plain "@U…" token is ugly but keeps "someone else was
      addressed" visible; raw <@ID> bracket syntax still never reaches the model.

    Resilient: with no cache/identity every mention still renders readable.
    """
    if not text:
        return text
    cache = user_cache or {}

    def repl(m):
        inner = m.group(1).strip()
        uid = inner.split('|', 1)[0].strip()
        if bot_user_id and bot_handle and uid == bot_user_id:
            return f'@{bot_handle}'
        info = cache.get(uid)
        name = info.get('username') if isinstance(info, dict) else None
        if name:
            return f'@{name}'
        if '|' in inner:
            label = inner.split('|', 1)[1].strip()
            if label:
                return f'@{label}'
        return f'@{uid}'  # unresolved -> visible addressee marker, never a silent strip

    return _MENTION_RE.sub(repl, text).strip()


def text_mentions_user(text, user_id):
    """True if text contains a Slack <@ID> mention of the given user id."""
    if not text or not user_id:
        return False
    for m in _MENTION_RE.finditer(text):
        if m.group(1).split('|', 1)[0].strip() == user_id:
            return True
    return False


def encode_outbound_mentions(text, name_to_id=None):
    """Turn model-produced pseudo-mentions into valid Slack <@ID> syntax (outbound safety net).

    - "<@Display Name>" (brackets around a non-ID): resolve to <@ID> when the name is known,
      else strip the brackets to plain text (never leave Slack-breaking syntax)
    - "@Name" standalone token: encode to <@ID> only when Name exactly matches a known name
    - a valid "<@ID>" is left untouched; bare names without "@" are left untouched
    """
    if not text:
        return text
    lut = {k.lower(): v for k, v in (name_to_id or {}).items()}

    def repl_bracket(m):
        inner = m.group(1).strip()
        uid = inner.split('|', 1)[0].strip()
        if _SLACK_ID_RE.match(uid):
            return m.group(0)  # already a valid id -> untouched
        hit = lut.get(inner.lower())
        return f'<@{hit}>' if hit else inner  # resolve, else strip broken brackets

    out = _MENTION_RE.sub(repl_bracket, text)

    if lut:
        def repl_at(m):
            token = m.group(1)
            hit, rest = _lookup_at_token(token, lut)
            return f'<@{hit}>{rest}' if hit else m.group(0)
        out = _AT_TOKEN_RE.sub(repl_at, out)

    return out


def _lookup_at_token(token, lut):
    """`(id, trailing)` for an "@token" body: an exact name, else the name with sentence
    punctuation or a possessive peeled off ("Claude." / "Claude's"), which stays as `trailing`.
    A hyphen is a legal name character, never trailing punctuation."""
    stem = token.rstrip(".'")
    bare = stem[:-2] if stem.lower().endswith("'s") else stem
    for cand in (token, stem, bare):
        if cand:
            hit = lut.get(cand.lower())
            if hit:
                return hit, token[len(cand):]
    return None, ''


def split_streamable_name_mention(text, prev=''):
    """`(release_now, hold_back)` — hold a trailing "@Partial" that a later delta may finish.

    A native stream sends DELTAS, so "@Claude" can arrive as "@Cl" + "aude"; encoding each half
    would match neither. The candidate is the text from the last "@" when it is a bare "@" or
    "@" + a name-shaped token running to the end, and the character before it (`prev` when the
    "@" opens the text) is not a word character or "<" — an email local part is never held.
    """
    if not text:
        return text, ''
    at = text.rfind('@')
    if at < 0:
        return text, ''
    fragment = text[at:]
    if len(fragment) > _AT_HOLD_MAX or not _AT_CANDIDATE_RE.fullmatch(fragment):
        return text, ''
    before = text[at - 1] if at > 0 else (prev or '')
    if before and _NO_MENTION_BEFORE_RE.match(before):
        return text, ''
    return text[:at], fragment


def encode_outbound_chunk(chunk, name_to_id, prev=''):
    """encode_outbound_mentions for one piece of a longer text whose preceding character is
    `prev`: an "@" right after a word character (an email) stays as written."""
    if not chunk or '@' not in chunk:
        return chunk
    context = 'a' if prev and _NO_MENTION_BEFORE_RE.match(prev) else ''
    return encode_outbound_mentions(context + chunk, name_to_id)[len(context):]


def strip_leading_self_prefix(text, names=None):
    """Strip a single leading 'Name:' self-attribution prefix the model may echo (Phase 3.4).

    Other bots now appear in history as 'Name: …' user turns, so the model can be tempted to
    reply as 'ChatGPT: …'. Only strips when the token before the first ':' EXACTLY matches one of
    ``names`` (case-insensitive), so legitimate content like 'Note: …' / 'Step 1: …' is untouched.
    """
    if not text or not names:
        return text
    stripped = text.lstrip()
    idx = stripped.find(':')
    if idx <= 0 or idx > 40:  # no colon near the start -> nothing to strip
        return text
    head = stripped[:idx].strip()
    if head.lower() in {n.lower() for n in names}:
        return stripped[idx + 1:].lstrip()
    return text


class SlackFormattingMixin(_Host):
    def _clean_mentions(self, text: str) -> str:
        """Resolve inbound Slack mentions: our own renders as "@<handle>", others as @name,
        unknown ids as a visible "@<id>". (Method name kept for existing call sites.)"""
        return resolve_inbound_mentions(
            text,
            user_cache=getattr(self, 'user_cache', None),
            bot_user_id=getattr(self, 'bot_user_id', None),
            bot_handle=getattr(self, 'bot_handle', None),
        )

    def _build_name_to_id_map(self) -> dict:
        """Reverse map of known display/real names (lowercased, as the encoder looks them up) ->
        user id: our own handle (from auth.test, no cache row needed) plus every name in the user
        cache. Our handle is authoritative: a cached name that differs from it only in case never
        takes it over."""
        mapping: dict = {}
        handle = getattr(self, 'bot_handle', None)
        self_id = getattr(self, 'bot_user_id', None)
        if handle and self_id:
            mapping[str(handle).lower()] = self_id
        cache = getattr(self, 'user_cache', None) or {}
        for uid, info in cache.items():
            if not isinstance(info, dict):
                continue
            for key in ('username', 'real_name'):
                name = info.get(key)
                if name:
                    mapping.setdefault(str(name).lower(), uid)
        return mapping

    def _encode_mentions(self, text: str) -> str:
        """Outbound safety net: convert pseudo-mentions to valid <@ID> / strip broken brackets.
        Code spans are quoted content and pass through exactly as written."""
        if not text or '@' not in text:
            return text  # cheap fast-path: nothing to encode
        from slack_client.messaging import _code_spans  # lazy: messaging imports the mixins
        lut = self._build_name_to_id_map()
        segments, _fence, _inline = _code_spans(text)
        out: List[str] = []
        prev = ''
        for chunk, is_code in segments:
            out.append(chunk if is_code else encode_outbound_chunk(chunk, lut, prev))
            prev = chunk[-1:]
        return ''.join(out)

    def format_text(self, text: str) -> str:
        """Format text for Slack using mrkdwn.

        Outbound hygiene: strip a leading self-name prefix the model may echo, then encode any
        pseudo-mentions into valid <@ID> syntax, before mrkdwn conversion.
        """
        text = strip_leading_self_prefix(text, getattr(config, 'self_prefix_names', None))
        text = self._encode_mentions(text)
        return self.markdown_converter.convert(text)

    # Error copy authored by the message processor: optional emoji (unicode or
    # :name:) followed by a **Bold Title** on the first line. Raw exception
    # strings never look like this.
    # The prefix group is atomic ((?>…)): ':' and '+' match both alternation branches, so a
    # non-matching ':+:+:…' prefix would otherwise backtrack exponentially (CodeQL py/redos).
    _AUTHORED_ERROR_RE = re.compile(r"^(?>(?::[\w+-]+:|[^\w*`\n])*)\*{1,2}[^\n*]+\*{1,2}")

    # The only error text a raw/technical failure may show the user. The actual
    # exception belongs in the logs (callers log it before reaching here).
    GENERIC_ERROR_MESSAGE = (
        "⚠️ **Something Went Wrong**\n\n"
        "Please try again in a moment. If it keeps happening, let an admin know."
    )

    def format_error_message(self, error: str) -> str:
        """Single gate for user-facing error text.

        Error copy authored upstream (emoji + **Bold Title** + one actionable
        sentence) passes through untouched. Anything else is technical detail —
        the user gets one fixed, friendly line instead of a code dump; the raw
        text lives only in the logs.
        """
        if error and self._AUTHORED_ERROR_RE.match(error):
            return error
        return self.GENERIC_ERROR_MESSAGE
