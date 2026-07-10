from __future__ import annotations

import re

from config import config

# A real Slack mention id is a user/workspace/bot id: U/W/B + uppercase alnum.
_SLACK_ID_RE = re.compile(r'^[UWB][A-Z0-9]{6,}$')
# Inner of a <@...> user mention (may carry an inline "|label").
_MENTION_RE = re.compile(r'<@([^>]+)>')
# Standalone "@Name" token (letters first), not already inside a <@...> and not an email local part.
_AT_TOKEN_RE = re.compile(r"(?<![<\w])@([A-Za-z][\w.'-]*)")


def resolve_inbound_mentions(text, user_cache=None, bot_user_id=None):
    """Resolve raw Slack <@ID> mentions in inbound text into a readable form.

    - the bot's own mention is removed (it's just the trigger)
    - other users resolve to "@DisplayName" when known in user_cache (so the model sees who
      was addressed) or to the inline "|label" if present
    - unknown ids are stripped (legacy behavior) so no raw <@ID> ever reaches the model

    Resilient: with no cache/identity it degrades to the original strip behavior.
    """
    if not text:
        return text
    cache = user_cache or {}

    def repl(m):
        inner = m.group(1).strip()
        uid = inner.split('|', 1)[0].strip()
        if bot_user_id and uid == bot_user_id:
            return ''
        info = cache.get(uid)
        name = info.get('username') if isinstance(info, dict) else None
        if name:
            return f'@{name}'
        if '|' in inner:
            label = inner.split('|', 1)[1].strip()
            if label:
                return f'@{label}'
        return ''  # unknown id -> strip (legacy fallback)

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
            hit = lut.get(m.group(1).lower())
            return f'<@{hit}>' if hit else m.group(0)
        out = _AT_TOKEN_RE.sub(repl_at, out)

    return out


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


class SlackFormattingMixin:
    def _clean_mentions(self, text: str) -> str:
        """Resolve inbound Slack mentions: drop our own mention, render others as @name, strip
        unknown ids. (Method name kept for backward compatibility with existing call sites.)"""
        return resolve_inbound_mentions(
            text,
            user_cache=getattr(self, 'user_cache', None),
            bot_user_id=getattr(self, 'bot_user_id', None),
        )

    def _build_name_to_id_map(self) -> dict:
        """Reverse map of known display/real names -> user id, derived from the user cache."""
        mapping = {}
        cache = getattr(self, 'user_cache', None) or {}
        for uid, info in cache.items():
            if not isinstance(info, dict):
                continue
            for key in ('username', 'real_name'):
                name = info.get(key)
                if name:
                    mapping.setdefault(name, uid)
        return mapping

    def _encode_mentions(self, text: str) -> str:
        """Outbound safety net: convert pseudo-mentions to valid <@ID> / strip broken brackets."""
        if not text or '@' not in text:
            return text  # cheap fast-path: nothing to encode
        return encode_outbound_mentions(text, self._build_name_to_id_map())

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
    _AUTHORED_ERROR_RE = re.compile(r"^(?::[\w+-]+:|[^\w*`\n])*\*{1,2}[^\n*]+\*{1,2}")

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
