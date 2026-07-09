"""Phase 1 — sender classification + multi-bot history role mapping.

Covers the new logic added for human/self/other_bot detection and the metadata that drives
the thread-history role fix (own bot -> assistant; humans AND other bots -> user). These are
standalone targeted tests; the legacy suite is not exercised here.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from slack_client.utilities import SlackUtilitiesMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin


# --- Lightweight carriers binding the real mixin methods (no full SlackBot needed) ---

class _Ident:
    """Minimal object exposing the real classify_sender / is_own_message."""
    is_own_message = SlackUtilitiesMixin.is_own_message
    classify_sender = SlackUtilitiesMixin.classify_sender

    def __init__(self, bot_id=None, bot_user_id=None, app_id=None):
        self.bot_id = bot_id
        self.bot_user_id = bot_user_id
        self.app_id = app_id


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Minimal harness to exercise the real get_thread_history against a mocked client."""
    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_warning(self, *a, **k): pass


SELF_BOT_ID = "B07SELF"
SELF_USER_ID = "U07SELF"


# --- classify_sender / is_own_message ---

def test_human_message_is_human():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"user": "U07HUMAN", "text": "hi"}
    assert b.classify_sender(msg) == "human"
    assert b.is_own_message(msg) is False


def test_own_message_by_bot_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "mine"}
    assert b.is_own_message(msg) is True
    assert b.classify_sender(msg) == "self"


def test_own_message_by_user_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    assert b.classify_sender({"user": SELF_USER_ID, "text": "mine"}) == "self"


def test_own_message_by_app_id():
    b = _Ident(app_id="A07SELF")
    assert b.is_own_message({"app_id": "A07SELF"}) is True
    assert b.is_own_message({"api_app_id": "A07SELF"}) is True


def test_other_bot_by_bot_id():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    msg = {"bot_id": "B07OTHER", "username": "Claude", "user": "U07X"}
    assert b.classify_sender(msg) == "other_bot"
    assert b.is_own_message(msg) is False


def test_other_bot_by_app_id_only():
    b = _Ident(bot_id=SELF_BOT_ID, bot_user_id=SELF_USER_ID)
    # app-posted message without subtype=="bot_message" must still be detected
    assert b.classify_sender({"app_id": "A07OTHER", "text": "x"}) == "other_bot"


def test_non_dict_defaults_human():
    b = _Ident(bot_id=SELF_BOT_ID)
    assert b.classify_sender(None) == "human"
    assert b.is_own_message(None) is False


# --- get_thread_history: metadata that feeds the role mapping ---

@pytest.mark.asyncio
async def test_get_thread_history_sets_sender_metadata():
    b = _Bot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "<@U07SELF> hello"},          # human
        {"ts": "2", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "my reply"},  # self
        {"ts": "3", "bot_id": "B07OTHER", "username": "Claude", "text": "from claude"},  # other bot
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )

    result = await b.get_thread_history("C1", "1")
    by_ts = {m.metadata["ts"]: m for m in result}

    # human
    assert by_ts["1"].metadata["sender_type"] == "human"
    assert by_ts["1"].metadata["is_bot"] is False
    assert by_ts["1"].metadata["bot_name"] is None
    assert by_ts["1"].text == "hello"  # mention stripped for humans

    # self
    assert by_ts["2"].metadata["sender_type"] == "self"
    assert by_ts["2"].metadata["is_bot"] is True

    # other bot — carries its display name for user-role prefixing
    assert by_ts["3"].metadata["sender_type"] == "other_bot"
    assert by_ts["3"].metadata["is_bot"] is True
    assert by_ts["3"].metadata["bot_name"] == "Claude"


@pytest.mark.asyncio
@pytest.mark.parametrize("footer_blocks", [
    # Current compact footer: single actions row, model name inside the button
    [
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "⚙️ gpt-5.5"},
             "action_id": "open_channel_settings"},
        ]},
    ],
    # Legacy two-row footer (context line + Configure button) — still present in old
    # channel history, must stay skipped
    [
        {"type": "context", "elements": [{"type": "mrkdwn", "text": ":robot_face: gpt-5.5"}]},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "⚙️ Configure"},
             "action_id": "open_channel_settings"},
        ]},
    ],
])
async def test_get_thread_history_skips_response_footer(footer_blocks):
    """The Configure footer (a separate own-bot message) must not enter rebuilt history,
    or every channel exchange would gain a bogus assistant turn saying just the model name."""
    b = _Bot()
    messages = [
        {"ts": "1", "user": "U07HUMAN", "text": "what model are you?"},
        {"ts": "2", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "I'm running gpt-5.5."},
        {"ts": "3", "bot_id": SELF_BOT_ID, "user": SELF_USER_ID, "text": "gpt-5.5",
         "blocks": footer_blocks},  # the footer message
    ]
    b.app.client.conversations_replies = AsyncMock(
        return_value={"messages": messages, "response_metadata": {}}
    )

    result = await b.get_thread_history("C1", "1")
    assert [m.metadata["ts"] for m in result] == ["1", "2"]  # footer (ts=3) skipped


# --- role mapping contract (as implemented in thread_management rebuild) ---

def _role_for(sender_type):
    """Mirror of the rule in ThreadManagementMixin._get_or_rebuild_thread_state:
    only our own messages are assistant turns; everyone else is a user turn."""
    return "assistant" if sender_type == "self" else "user"


@pytest.mark.parametrize("sender_type,expected", [
    ("self", "assistant"),
    ("other_bot", "user"),
    ("human", "user"),
])
def test_role_mapping_contract(sender_type, expected):
    assert _role_for(sender_type) == expected
