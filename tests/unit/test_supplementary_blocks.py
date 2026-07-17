"""F48 — content Slack delivers OUTSIDE `text` (blocks + attachments).

The incident: a user uploaded a TSV with the comment "what about this?". Slack delivered
it as a native `table` block inside `attachments[]` with NO `files` entry, our ingest read
only `text` and `files`, and the bot answered "I only see 'what about this?' — nothing
attached or quoted." `event["attachments"]` had zero readers on every ingest path.

tests/fixtures/slack_table_event.json is that exact payload, fetched live from Slack
(ts 1784222395.492299, C04QDHE8W8M). It is the ground truth here because several of its
properties break a naive implementation: header cells are rich_text while data cells are
raw_text, one cell is literal JSON null, and `fallback` is the noise string
"[no preview available]" rather than the table's content.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from slack_client.channel_pulse import ChannelPulse, pulse_supplementary_budget
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.formatting.blocks import (SUPPLEMENTARY_CHAR_BUDGET,
                                            extract_supplementary_text)
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin

FIXTURE = Path(__file__).parent.parent / "fixtures" / "slack_table_event.json"


def _fixture():
    with open(FIXTURE) as fh:
        return json.load(fh)


def _extract(msg, primary_text="", **kw):
    return extract_supplementary_text(msg, primary_text=primary_text, **kw)


# --------------------------------------------------------------- the real payload


def test_real_incident_payload_yields_the_table():
    ev = _fixture()
    # Guard the fixture's own shape: if these stop holding, the payload was edited and
    # the rest of this file is testing something other than the incident.
    assert "files" not in ev and ev.get("subtype") is None
    assert ev["attachments"][0]["blocks"][0]["type"] == "table"

    out = _extract(ev, primary_text=ev["text"])

    assert "[Slack table]" in out
    # Header row: every cell is rich_text (styled bold), so a naive cell["text"] read
    # returns None for all six and silently drops the column names.
    assert "Name | Email | User type | Groups | Last Active | ID" in out.splitlines()[1]
    # Data rows (raw_text cells) survive alongside them.
    assert "Erin Evans | erin.evans@example.com" in out
    assert "Quinn Quill | quinn.quill@example.com" in out
    # All 18 rows render — nothing silently lost, no truncation marker.
    assert len([ln for ln in out.splitlines() if " | " in ln]) == 18
    assert "omitted" not in out


def test_real_payload_null_cell_does_not_erase_the_table():
    # Row 10 col 4 ("Last Active" for Jack Jones) is literal JSON null. cell.get()
    # on it raises AttributeError; a "return '' on any error" rule would erase all 18 rows.
    ev = _fixture()
    rows = ev["attachments"][0]["blocks"][0]["rows"]
    assert rows[10][4] is None, "fixture no longer carries the null cell"

    out = _extract(ev, primary_text=ev["text"])

    assert "Jack Jones | jack.jones@example.com" in out
    # The null renders as an empty cell in place — the row keeps its shape, and the
    # rows after it still render.
    assert "Expresso,User |  | 1000000010" in out
    assert "Karen Kim" in out


def test_real_payload_fallback_placeholder_is_not_injected():
    # `fallback` here is the literal "[no preview available]" — NOT the table content and
    # NOT equal to `text`, so a "skip fallback when it equals text" rule never fires and
    # the noise string reaches the model.
    ev = _fixture()
    assert ev["attachments"][0]["fallback"] == "[no preview available]"

    out = _extract(ev, primary_text=ev["text"])

    assert "no preview available" not in out


def test_real_payload_does_not_duplicate_primary_text():
    # msg["blocks"] is just the rich_text rendering of "what about this?" — already in
    # `text`. Rendering top-level rich_text would double the prompt.
    ev = _fixture()
    assert [b["type"] for b in ev["blocks"]] == ["rich_text"]

    out = _extract(ev, primary_text=ev["text"])

    assert "what about this?" not in out


def test_extraction_is_deterministic():
    ev = _fixture()
    assert _extract(ev, primary_text=ev["text"]) == _extract(
        ev, primary_text=ev["text"]
    )


# ------------------------------------------------------------------- cell shapes


def test_header_rich_text_and_raw_text_cells_both_render():
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {
                                                    "type": "text",
                                                    "text": "Region",
                                                    "style": {"bold": True},
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                            [{"type": "raw_text", "text": "EMEA"}],
                        ],
                    }
                ]
            }
        ]
    }

    out = _extract(msg)

    # Bold styling is dropped: the model needs the word, and emphasis markers inside a
    # table cell are noise.
    assert "Region" in out and "*Region*" not in out
    assert "EMEA" in out


def test_raw_number_cells_render():
    # The fixture has no raw_number cells but Slack's schema permits them; read the
    # documented display fields rather than guessing one key.
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {"type": "raw_number", "text": "42"},
                                {"type": "raw_number", "value": 7},
                                {"type": "raw_number", "number": 3.5},
                            ],
                        ],
                    }
                ]
            }
        ]
    }

    assert "42 | 7 | 3.5" in _extract(msg)


def test_malformed_cells_fail_open_per_node():
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {"type": "raw_text", "text": "keep me"},
                                ["not-a-cell"],
                                None,
                                {"junk": True},
                            ],
                            [{"type": "raw_text", "text": "and me"}],
                        ],
                    }
                ]
            }
        ]
    }

    out = _extract(msg)

    assert "keep me" in out and "and me" in out


def test_multiline_cell_stays_on_one_row():
    # A cell containing newlines would otherwise break the " | " row format apart.
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {"type": "raw_text", "text": "line one\nline two"},
                                {"type": "raw_text", "text": "b"},
                            ],
                        ],
                    }
                ]
            }
        ]
    }

    assert "line one line two | b" in _extract(msg)


# ---------------------------------------------------------------------- unfurls


def _unfurl(**kw):
    att = {
        "service_name": "GitHub",
        "title": "Fix the table bug",
        "title_link": "https://github.com/x/pull/7",
        "text": "Adds an extractor.",
    }
    att.update(kw)
    return {"text": kw.pop("_text", ""), "attachments": [att]}


def test_unfurl_renders_with_link_preview_provenance():
    out = _extract(_unfurl(), primary_text="look at this")

    assert "[Link preview]" in out
    assert "Fix the table bug" in out
    assert "https://github.com/x/pull/7" in out
    assert "Adds an extractor." in out
    assert "Via: GitHub" in out


def test_unfurl_title_suppressed_only_on_exact_match():
    # Primary text IS the title -> exact canonical match -> not sent twice.
    out = _extract(_unfurl(), primary_text="Fix the table bug")
    assert "Fix the table bug" not in out
    assert "Adds an extractor." in out  # the rest survives


def test_unfurl_title_kept_inside_a_longer_sentence():
    # Substring, NOT an exact match. Keeping the small duplicate is safer than deleting
    # content that may be distinct.
    out = _extract(
        _unfurl(), primary_text="hey team, Fix the table bug is ready to review"
    )
    assert "Fix the table bug" in out


def test_unfurl_title_suppressed_when_primary_is_only_that_slack_link():
    # Canonicalization resolves <url|label> to its visible label, so a message that is
    # solely the link doesn't get its own label echoed back.
    out = _extract(
        _unfurl(), primary_text="<https://github.com/x/pull/7|Fix the table bug>"
    )
    assert "Fix the table bug" not in out


def test_bare_url_primary_dedupes_against_the_unfurl_link():
    out = _extract(_unfurl(), primary_text="https://github.com/x/pull/7")
    assert out.count("https://github.com/x/pull/7") == 0


# --------------------------------------------------------- quoted / forwarded


def test_quoted_message_carries_quoted_provenance():
    # A quoted third party must not flatten into the speaker's own words.
    msg = {
        "text": "thoughts?",
        "attachments": [
            {
                "author_name": "Dana",
                "ts": "1700000000.1",
                "message_blocks": [
                    {
                        "message": {
                            "blocks": [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {
                                                    "type": "text",
                                                    "text": "We should ship it Friday.",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ],
            }
        ],
    }

    out = _extract(msg, primary_text="thoughts?")

    assert "[Quoted Slack message]" in out
    assert "We should ship it Friday." in out
    assert "Author: Dana" in out


def test_quoted_nested_rich_text_is_rendered_not_skipped():
    # Top-level rich_text duplicates `text` and is skipped; NESTED rich_text inside a
    # quote has no such duplicate, so skipping it there would lose the quote entirely.
    msg = {
        "text": "",
        "attachments": [
            {
                "message_blocks": [
                    {
                        "message": {
                            "blocks": [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_quote",
                                            "elements": [
                                                {"type": "text", "text": "quoted line"}
                                            ],
                                        },
                                        {
                                            "type": "rich_text_list",
                                            "style": "bullet",
                                            "elements": [
                                                {
                                                    "type": "rich_text_section",
                                                    "elements": [
                                                        {
                                                            "type": "text",
                                                            "text": "item one",
                                                        }
                                                    ],
                                                }
                                            ],
                                        },
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    }

    out = _extract(msg)

    assert "> quoted line" in out
    assert "- item one" in out


def test_quoted_text_and_message_blocks_do_not_double():
    # Slack sends the same quoted content twice: mrkdwn in `text`, rich_text in
    # `message_blocks`. Canonicalization normalizes the emphasis markers so the two
    # compare equal and only one copy reaches the model.
    msg = {
        "text": "see below",
        "attachments": [
            {
                "text": "We should *ship* it Friday.",
                "message_blocks": [
                    {
                        "message": {
                            "blocks": [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {"type": "text", "text": "We should "},
                                                {
                                                    "type": "text",
                                                    "text": "ship",
                                                    "style": {"bold": True},
                                                },
                                                {"type": "text", "text": " it Friday."},
                                            ],
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ],
            }
        ],
    }

    out = _extract(msg, primary_text="see below")

    assert out.count("it Friday.") == 1


# ------------------------------------------------------- sections / fields / webhooks


def test_section_text_and_fields_render():
    msg = {
        "text": "",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "Deploy report"}},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Build *123* finished."},
                "fields": [
                    {"type": "mrkdwn", "text": "Duration: 4m"},
                    {"type": "mrkdwn", "text": "Status: green"},
                ],
            },
        ],
    }

    out = _extract(msg)

    assert "Deploy report" in out
    assert "Build *123* finished." in out
    assert "Duration: 4m" in out and "Status: green" in out


def test_legacy_webhook_fields_are_the_whole_payload():
    # Jira/GitHub/Drive posts put everything in attachments[].fields[] with text empty.
    msg = {
        "text": "",
        "attachments": [
            {
                "fallback": "PROJ-42 assigned to Dana",
                "fields": [
                    {"title": "Issue", "value": "PROJ-42"},
                    {"title": "Assignee", "value": "Dana"},
                ],
            }
        ],
    }

    out = _extract(msg)

    assert "Issue: PROJ-42" in out
    assert "Assignee: Dana" in out


def test_sibling_fields_sharing_a_value_both_survive():
    # Dedupe must key on the rendered field (title included) — keying on the bare value
    # would drop "Target: main" as a duplicate of "Branch: main".
    msg = {
        "text": "",
        "attachments": [
            {
                "fields": [
                    {"title": "Branch", "value": "main"},
                    {"title": "Target", "value": "main"},
                ]
            }
        ],
    }

    out = _extract(msg)

    assert "Branch: main" in out and "Target: main" in out


def test_fallback_kept_when_it_carries_distinct_news():
    # The counter-example to "prefer structured blocks over fallback": both matter here.
    msg = {
        "text": "",
        "attachments": [
            {
                "fallback": "Build #123 failed — 4 errors",
                "text": "See logs",
                "fields": [{"title": "Branch", "value": "main"}],
            }
        ],
    }

    out = _extract(msg)

    assert "Build #123 failed — 4 errors" in out
    assert "See logs" in out and "Branch: main" in out


def test_fallback_dropped_when_it_merely_repeats():
    msg = {"text": "", "attachments": [{"fallback": "See logs", "text": "See logs"}]}

    assert _extract(msg).count("See logs") == 1


def test_fallback_dropped_when_it_repeats_primary_text():
    msg = {"text": "deploy done", "attachments": [{"fallback": "deploy done"}]}

    assert _extract(msg, primary_text="deploy done") == ""


# --------------------------------------------------------------- mention policy


def test_mentions_render_raw_for_the_callers_mention_pass():
    # The extractor emits the same raw syntax Slack puts in `text`, so the caller's
    # existing mention policy resolves cell mentions too. Rendering them pre-cleaned
    # here would leave "<@U…>" raw in the model's context.
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {"type": "user", "user_id": "U123"},
                                                {"type": "text", "text": " owns it"},
                                            ],
                                        }
                                    ],
                                }
                            ],
                        ],
                    }
                ]
            }
        ]
    }

    assert "<@U123> owns it" in _extract(msg)


# ---------------------------------------------------------------------- bounds


def _big_table(rows):
    return {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {"type": "raw_text", "text": f"r{i}c0"},
                                {"type": "raw_text", "text": f"r{i}c1"},
                            ]
                            for i in range(rows)
                        ],
                    }
                ]
            }
        ]
    }


def test_row_cap_announces_itself():
    out = _extract(_big_table(5000))

    assert "r0c0" in out  # header/first rows survive
    assert "[… 4,900 more table rows omitted]" in out
    assert len([ln for ln in out.splitlines() if " | " in ln]) == 100


def test_column_cap_announces_itself():
    msg = {
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [{"type": "raw_text", "text": f"c{i}"} for i in range(30)]
                        ],
                    }
                ]
            }
        ]
    }

    out = _extract(msg)

    assert "c0 | c1" in out
    assert "[… 10 more columns omitted]" in out


def test_char_budget_truncation_is_honest_and_bounded():
    msg = {
        "text": "",
        "attachments": [
            {"fields": [{"title": f"F{i}", "value": "v" * 400} for i in range(10)]}
            for _ in range(20)
        ],
    }

    out = _extract(msg, budget=2000)

    assert len(out) <= 2000
    assert "Supplementary Slack content truncated" in out
    assert "characters omitted" in out


def test_tight_budget_still_yields_header_first_rows_and_a_marker():
    # The ChannelPulse case: cut at ROW boundaries so a table-only message still says
    # something true, rather than being dropped whole.
    ev = _fixture()

    out = _extract(ev, primary_text=ev["text"], budget=482)

    assert len(out) <= 482
    assert "Name | Email | User type" in out  # header survives
    assert "Alice Anderson" in out  # and at least one data row
    assert "more table rows omitted]" in out


def test_budget_below_the_floor_returns_nothing_rather_than_a_lie():
    # Too small for a label + content + marker to coexist honestly.
    ev = _fixture()

    assert _extract(ev, primary_text=ev["text"], budget=40) == ""


def test_default_budget_is_respected():
    out = _extract(_big_table(5000), budget=SUPPLEMENTARY_CHAR_BUDGET)

    assert len(out) <= SUPPLEMENTARY_CHAR_BUDGET


def test_deep_nesting_is_bounded_and_admits_it():
    # A malformed/adversarial payload must not recurse without end.
    att = {
        "message_blocks": [
            {
                "message": {
                    "blocks": [
                        {"type": "section", "text": {"type": "mrkdwn", "text": "deep"}}
                    ]
                }
            }
        ]
    }
    for _ in range(12):
        att = {
            "attachments": [att],
            "message_blocks": [
                {
                    "message": {
                        "blocks": [
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": "deep"},
                            }
                        ]
                    }
                }
            ],
        }

    out = _extract({"attachments": [att] * 60})

    assert isinstance(out, str)
    assert len(out) <= SUPPLEMENTARY_CHAR_BUDGET


def test_node_budget_exhaustion_admits_itself():
    msg = {
        "text": "",
        "attachments": [
            {"fields": [{"title": f"F{i}", "value": f"v{i}"} for i in range(50)]}
            for _ in range(20)
        ],
    }

    out = _extract(msg)

    assert "more Slack content items omitted]" in out


# --------------------------------------------------------- nothing to extract


@pytest.mark.parametrize(
    "msg",
    [
        {},
        {"text": "plain message"},
        {"text": "hi", "blocks": [{"type": "rich_text", "elements": []}]},
        {"text": "hi", "attachments": []},
        {"text": "hi", "blocks": [{"type": "divider"}]},
        None,
        "not-a-dict",
    ],
)
def test_nothing_to_extract_returns_empty(msg):
    assert _extract(msg, primary_text="hi") == ""


def test_interactive_chrome_is_never_scraped():
    # Button values carry serialized context; generically recursing into action values,
    # action_ids or private_metadata would duplicate content and leak opaque state.
    msg = {
        "text": "hi",
        "blocks": [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "open_welcome_settings",
                        "value": '{"secret_ctx":"leak-me"}',
                        "text": {"type": "plain_text", "text": "Configure"},
                    }
                ],
            }
        ],
    }

    out = _extract(msg, primary_text="hi")

    assert "leak-me" not in out and "Configure" not in out


# ==================================================================== call sites


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """Minimal harness binding the real get_thread_history to a mocked Slack client."""

    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.user_cache = {}
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k):
        pass

    def log_debug(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


def _replies(*messages):
    bot = _Bot()
    bot.app.client.conversations_replies = AsyncMock(
        return_value={"messages": list(messages), "has_more": False}
    )
    return bot


@pytest.mark.asyncio
async def test_history_rebuild_keeps_the_table_the_tuesday_amnesia_case():
    # Slack is the ONLY transcript. Fixing the live path alone buys exactly one turn:
    # the table ingests Monday, Tuesday's rebuild re-drops it, and the bot contradicts
    # itself about a file it already discussed.
    ev = _fixture()
    bot = _replies({**ev, "user": "U07HUMAN"})

    history = await bot.get_thread_history("C04QDHE8W8M", "1784222395.492299")

    assert len(history) == 1
    assert "what about this?" in history[0].text
    assert "[Slack table]" in history[0].text
    assert "Erin Evans | erin.evans@example.com" in history[0].text


@pytest.mark.asyncio
async def test_history_rebuild_and_live_path_agree():
    # A message must serialize IDENTICALLY whether it arrived live or was rebuilt, or
    # the model sees the thread change under it across a restart.
    ev = _fixture()
    bot = _replies({**ev, "user": "U07HUMAN"})
    history = await bot.get_thread_history("C04QDHE8W8M", "1784222395.492299")

    live_supplementary = _extract(ev, primary_text=ev["text"])

    assert live_supplementary and live_supplementary in history[0].text


@pytest.mark.asyncio
async def test_history_rebuild_does_not_extract_our_own_chrome():
    # The deep-research card slips past the existing chrome predicates: its fallback is
    # "Deep research in progress…" and its blocks look semantic. Extracting supplementary
    # content from our OWN messages turns todos and progress counters into assistant
    # "evidence" — the F47 attribution bug. Real answers always have canonical top-level
    # text, so a self message loses nothing by skipping extraction.
    card = {
        "bot_id": "B07SELF",
        "user": "U07SELF",
        "text": "Here is what I found: ship it.",
        "attachments": [
            {
                "fallback": "Deep research in progress…",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "✓ Searched 14 sources"},
                    },
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": "3 tools used"}],
                    },
                ],
            }
        ],
    }
    bot = _replies(card)

    history = await bot.get_thread_history("C1", "1.0")

    assert history[0].text == "Here is what I found: ship it."
    assert "Searched 14 sources" not in history[0].text
    assert "Deep research in progress" not in history[0].text


@pytest.mark.asyncio
async def test_history_rebuild_extracts_for_other_bots():
    # A co-resident assistant's webhook-shaped post is real context, not our chrome.
    bot = _replies(
        {
            "bot_id": "B07OTHER",
            "username": "Jira",
            "text": "",
            "attachments": [{"fields": [{"title": "Issue", "value": "PROJ-42"}]}],
        }
    )

    history = await bot.get_thread_history("C1", "1.0")

    assert "Issue: PROJ-42" in history[0].text


@pytest.mark.asyncio
async def test_history_rebuild_cleans_mentions_inside_table_cells():
    # Ordering proof: supplementary must be combined RAW and cleaned WITH the primary
    # text. Appending after the mention pass leaves "<@U…>" raw inside cells.
    msg = {
        "user": "U07HUMAN",
        "text": "who owns this?",
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {"type": "user", "user_id": "U0DANA"}
                                            ],
                                        }
                                    ],
                                }
                            ]
                        ],
                    }
                ]
            }
        ],
    }
    bot = _replies(msg)
    bot.user_cache = {"U0DANA": {"username": "dana"}}

    history = await bot.get_thread_history("C1", "1.0")

    assert "@dana" in history[0].text
    assert "<@U0DANA>" not in history[0].text


# ------------------------------------------------------------------ live path


class _Host(SlackMessageEventsMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.user_cache = {}
        self.db = MagicMock()
        self.db.get_user_info_async = AsyncMock(return_value=None)
        self.channel_pulse = None

    async def get_username(self, uid, client=None):
        return "tester"

    async def get_user_timezone(self, uid, client=None):
        return "UTC"

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_live_event_to_message_carries_the_table():
    ev = _fixture()

    msg = await _Host()._event_to_message({**ev, "channel": "C04QDHE8W8M"}, MagicMock())

    assert "what about this?" in msg.text
    assert "[Slack table]" in msg.text
    assert "Erin Evans | erin.evans@example.com" in msg.text


@pytest.mark.asyncio
async def test_live_path_cleans_mentions_inside_table_cells():
    host = _Host()
    host.user_cache = {"U0DANA": {"username": "dana"}}
    ev = {
        "channel": "C1",
        "ts": "1.0",
        "user": "U07HUMAN",
        "text": "who owns this?",
        "attachments": [
            {
                "blocks": [
                    {
                        "type": "table",
                        "rows": [
                            [
                                {
                                    "type": "rich_text",
                                    "elements": [
                                        {
                                            "type": "rich_text_section",
                                            "elements": [
                                                {"type": "user", "user_id": "U0DANA"}
                                            ],
                                        }
                                    ],
                                }
                            ]
                        ],
                    }
                ]
            }
        ],
    }

    msg = await host._event_to_message(ev, MagicMock())

    assert "@dana" in msg.text and "<@U0DANA>" not in msg.text


@pytest.mark.asyncio
async def test_live_path_skips_our_own_chrome():
    host = _Host()
    ev = {
        "channel": "C1",
        "ts": "1.0",
        "user": "U07SELF",
        "bot_id": "B07SELF",
        "text": "Here is what I found.",
        "attachments": [
            {
                "fallback": "Deep research in progress…",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "✓ Searched 14 sources"},
                    }
                ],
            }
        ],
    }

    msg = await host._event_to_message(ev, MagicMock())

    assert msg.text == "Here is what I found."


# ---------------------------------------------------------------- channel pulse


class _PulseHost(SlackMessageEventsMixin, SlackUtilitiesMixin):
    def __init__(self, pulse):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.user_cache = {}
        self.channel_pulse = pulse

    def log_debug(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_pulse_feed_records_a_webhook_post_with_empty_text():
    # The empty-text guard returned early, so a webhook post whose content lives only in
    # attachments[] never reached awareness at all.
    pulse = ChannelPulse()
    host = _PulseHost(pulse)

    await host._feed_channel_pulse(
        {
            "channel": "C1",
            "ts": "100.0",
            "subtype": "bot_message",
            "bot_id": "BJIRA",
            "username": "Jira",
            "text": "",
            "attachments": [{"fields": [{"title": "Issue", "value": "PROJ-42"}]}],
        }
    )

    assert "PROJ-42" in pulse.render_envelope("C1")


@pytest.mark.asyncio
async def test_pulse_feed_keeps_its_marker_inside_the_entry_cap():
    # record() head-slices to pulse_text_truncate (500). Extracting against the default
    # 12,000-char budget would put the extractor's own end marker past the slice, leaving
    # a partial table that looks complete.
    pulse = ChannelPulse()
    host = _PulseHost(pulse)
    ev = _fixture()

    await host._feed_channel_pulse({**ev, "channel": "C1", "ts": "100.0"})

    entry = pulse._buffers["C1"][0]
    assert len(entry["text"]) <= int(config.pulse_text_truncate)
    assert "Name | Email" in entry["text"]  # header survives
    assert "more table rows omitted]" in entry["text"]  # and says so


@pytest.mark.asyncio
async def test_pulse_backfill_and_live_feed_agree():
    # Fixing only the live feed gives a cold start and a live session DIFFERENT evidence
    # for the same message.
    ev = _fixture()
    live, cold = ChannelPulse(), ChannelPulse()
    await _PulseHost(live)._feed_channel_pulse({**ev, "channel": "C1", "ts": "100.0"})

    bot = _Bot()
    client = MagicMock()
    client.conversations_history = AsyncMock(
        return_value={"messages": [{**ev, "channel": "C1", "ts": "100.0"}]}
    )
    await cold.ensure_backfill("C1", client, bot)

    assert live._buffers["C1"][0]["text"] == cold._buffers["C1"][0]["text"]
    assert "Name | Email" in cold._buffers["C1"][0]["text"]


def test_pulse_supplementary_budget_leaves_room_for_the_marker():
    assert (
        pulse_supplementary_budget("what about this?")
        == int(config.pulse_text_truncate) - 18
    )
    # A long primary text can't drive the budget to a point where nothing honest fits.
    assert pulse_supplementary_budget("x" * 5000) == 160


def test_pulse_record_truncation_admits_what_it_dropped():
    pulse = ChannelPulse()
    pulse.record(
        "C1",
        ts="1.0",
        thread_ts=None,
        user_id="U1",
        display_name="Dana",
        sender_type="human",
        text="A" * 900,
        is_bot=False,
    )

    entry = pulse._buffers["C1"][0]
    assert len(entry["text"]) <= int(config.pulse_text_truncate)
    assert "chars truncated]" in entry["text"]
