"""Unit tests for Phase 2 — @mention tagging.

Covers inbound mention resolution, self-mention detection, the participant roster, and the
outbound mention-encoding safety net. Pure functions are tested directly; the SlackFormattingMixin
methods are exercised via a tiny stub.
"""

from slack_client.formatting.text import (
    resolve_inbound_mentions,
    text_mentions_user,
    encode_outbound_mentions,
    SlackFormattingMixin,
)
from message_processor.utilities import build_roster_text

# Realistic Slack ids (U + >=6 alnum) so the "already valid" detector recognizes them.
PETER = "U07PETER01"
CLAUDE = "U07CLAUDE9"
BOT = "U07SELFBOT"

CACHE = {
    PETER: {"username": "peter", "real_name": "Erin Evans"},
    CLAUDE: {"username": "Claude"},
}


# --- inbound resolution -------------------------------------------------------------------

def test_inbound_other_user_resolves_to_at_name():
    assert resolve_inbound_mentions(f"hey <@{PETER}> look", CACHE) == "hey @peter look"


def test_inbound_self_mention_removed():
    out = resolve_inbound_mentions(f"<@{BOT}> what's up", CACHE, bot_user_id=BOT)
    assert out == "what's up"


def test_inbound_self_and_other_mixed():
    out = resolve_inbound_mentions(f"<@{BOT}> ask <@{PETER}> please", CACHE, bot_user_id=BOT)
    assert "@peter" in out and BOT not in out


def test_inbound_unknown_id_stripped():
    assert resolve_inbound_mentions("<@U07UNKNOWN1> hi", {}) == "hi"


def test_inbound_inline_label_used_when_uncached():
    assert resolve_inbound_mentions("<@U07UNKNOWN1|bob> hi", {}) == "@bob hi"


def test_inbound_channel_mention_untouched():
    # <#...> and <!...> are not user mentions and must be left alone
    assert resolve_inbound_mentions("see <#C123|general> and <!here>", CACHE) == "see <#C123|general> and <!here>"


def test_inbound_empty_safe():
    assert resolve_inbound_mentions("", CACHE) == ""
    assert resolve_inbound_mentions(None, CACHE) is None


# --- self-mention detection ---------------------------------------------------------------

def test_text_mentions_user_true():
    assert text_mentions_user(f"yo <@{BOT}>", BOT) is True


def test_text_mentions_user_with_label():
    assert text_mentions_user(f"yo <@{BOT}|chatgpt>", BOT) is True


def test_text_mentions_user_false():
    assert text_mentions_user(f"yo <@{PETER}>", BOT) is False
    assert text_mentions_user("", BOT) is False


# --- outbound encoding safety net ---------------------------------------------------------

NAME_TO_ID = {"peter": PETER, "Erin Evans": PETER, "Claude": CLAUDE}


def test_outbound_bracketed_name_resolves():
    assert encode_outbound_mentions("ping <@Erin Evans> now", NAME_TO_ID) == f"ping <@{PETER}> now"


def test_outbound_at_token_resolves():
    assert encode_outbound_mentions("hey @peter", NAME_TO_ID) == f"hey <@{PETER}>"


def test_outbound_unresolvable_bracket_stripped():
    # Never leave Slack-breaking <@nonid> syntax: strip to plain text
    assert encode_outbound_mentions("who is <@Ghost User>?", NAME_TO_ID) == "who is Ghost User?"


def test_outbound_valid_id_untouched():
    assert encode_outbound_mentions(f"thanks <@{CLAUDE}>", NAME_TO_ID) == f"thanks <@{CLAUDE}>"


def test_outbound_bare_name_untouched():
    assert encode_outbound_mentions("Peter said hello", NAME_TO_ID) == "Peter said hello"


def test_outbound_email_local_part_not_encoded():
    # "@peter" preceded by a word char (email) must not be treated as a mention
    assert encode_outbound_mentions("mail bob@peter for info", NAME_TO_ID) == "mail bob@peter for info"


def test_outbound_empty_map_strips_broken_brackets():
    assert encode_outbound_mentions("hi <@Some Name>", {}) == "hi Some Name"
    assert encode_outbound_mentions("hi @peter", {}) == "hi @peter"  # no map -> @tokens untouched


# --- participant roster -------------------------------------------------------------------

def test_roster_builds_block_with_ids():
    out = build_roster_text({PETER: "Erin Evans", CLAUDE: "Claude"})
    assert f"<@{PETER}>" in out and f"<@{CLAUDE}>" in out
    assert "Erin Evans" in out


def test_roster_empty_returns_blank():
    assert build_roster_text({}) == ""
    assert build_roster_text({"bot": "x", "unknown": "y"}) == ""


def test_roster_skips_self():
    out = build_roster_text({PETER: "Erin Evans", BOT: "ChatGPT"}, bot_user_id=BOT)
    assert f"<@{PETER}>" in out and BOT not in out


def test_roster_prefers_cache_name():
    out = build_roster_text({PETER: "stale"}, user_cache=CACHE)
    assert "peter" in out and "stale" not in out


# --- mixin integration --------------------------------------------------------------------

class _FakeMD:
    def convert(self, text):
        return text  # identity, so we can assert on mention encoding alone


class _FmtStub(SlackFormattingMixin):
    def __init__(self, user_cache=None, bot_user_id=None):
        self.user_cache = user_cache or {}
        self.bot_user_id = bot_user_id
        self.markdown_converter = _FakeMD()


def test_mixin_clean_mentions_resolves_and_drops_self():
    stub = _FmtStub(user_cache=CACHE, bot_user_id=BOT)
    assert stub._clean_mentions(f"<@{BOT}> tell <@{PETER}> hi") == "tell @peter hi"


def test_mixin_build_name_to_id_map():
    stub = _FmtStub(user_cache=CACHE)
    m = stub._build_name_to_id_map()
    assert m["peter"] == PETER and m["Erin Evans"] == PETER and m["Claude"] == CLAUDE


def test_mixin_encode_fast_path_no_at():
    stub = _FmtStub(user_cache=CACHE)
    assert stub._encode_mentions("no mentions here") == "no mentions here"


def test_mixin_format_text_encodes_then_converts():
    stub = _FmtStub(user_cache=CACHE)
    assert stub.format_text("ping <@Erin Evans>") == f"ping <@{PETER}>"


def test_mixin_resilient_without_identity_or_cache():
    stub = _FmtStub(user_cache={}, bot_user_id=None)
    # falls back to legacy strip behavior, never errors
    assert stub._clean_mentions("<@U07UNKNOWN1> hello") == "hello"
