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

    def format_error_message(self, error: str) -> str:
        """Format error messages for Slack with emojis and code blocks"""
        import re

        # Check for specific error types first
        if "taking too long" in error.lower() or "timeout" in error.lower():
            error_code = "TIMEOUT"
            error_type = "timeout_error"
            error_message = error
        elif "rate limit" in error.lower():
            error_code = "RATE_LIMIT"
            error_type = "rate_limit_error"
            error_message = error
        else:
            # Extract error code if present
            error_code_match = re.search(r'Error code: (\d+)', error)
            error_code = error_code_match.group(1) if error_code_match else "Unknown"

        # Try to extract the actual error message (if not already set)
        if "error_message" not in locals():
            if "{'error':" in error:
                # Parse OpenAI API error format
                try:
                    import json
                    error_dict_str = error[error.find("{'error':"):].replace("'", '"')
                    error_dict = json.loads(error_dict_str)
                    error_message = error_dict.get('error', {}).get('message', error)
                    if "error_type" not in locals():
                        error_type = error_dict.get('error', {}).get('type', 'unknown_error')
                except Exception:
                    # Fallback to simpler extraction
                    if "'message':" in error:
                        msg_start = error.find("'message': '") + len("'message': '")
                        msg_end = error.find("',", msg_start)
                        if msg_end > msg_start:
                            error_message = error[msg_start:msg_end]
                        else:
                            error_message = error
                    else:
                        error_message = error
                    if "error_type" not in locals():
                        error_type = "api_error"
            else:
                error_message = error
                if "error_type" not in locals():
                    error_type = "general_error"
        
        # Format the error message for Slack
        formatted = ":warning: *Oops! Something went wrong*\n\n"
        formatted += f"*Error Code:* `{error_code}`\n"
        formatted += f"*Type:* `{error_type}`\n\n"
        formatted += f"*Details:*\n```{error_message}```\n\n"
        formatted += ":bulb: *What you can do:*\n"
        
        # Add helpful suggestions based on error type
        if "rate_limit" in error_type.lower():
            formatted += "• Wait a moment and try again\n"
            formatted += "• The API rate limit has been reached"
        elif "invalid_request" in error_type.lower():
            formatted += "• Try rephrasing your request\n"
            formatted += "• The request format may be invalid"
        elif "context_length" in error_message.lower():
            formatted += "• Start a new thread\n"
            formatted += "• The conversation has become too long"
        else:
            formatted += "• Try again in a moment\n"
            formatted += "• If the problem persists, contact support"
        
        return formatted
