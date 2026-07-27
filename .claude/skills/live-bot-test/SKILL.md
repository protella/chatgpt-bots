---
name: live-bot-test
description: Drive the dev Slack bot end-to-end against real Slack. Use when verifying bot behavior live — participation/wake decisions, reactions, streaming, image and file paths — or when a unit test passes but the real path is suspect. Covers the user-token harness, the classifier probe, and the contamination traps.
---

# Live dev-bot testing

Authorized scope: **C0BKX77NU66** (`#chatgpt-bot-test`) and DMs with the dev bot. Starting,
stopping and restarting the dev bot process is fine. **Prod is hands-off.**

## Post as the user, never as the bot

Test messages go through `SLACK_TEST_USER_TOKEN` (`.env`), not `SLACK_BOT_TOKEN`.

```python
AsyncWebClient(token=os.getenv("SLACK_TEST_USER_TOKEN"))   # post / upload
AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))         # read the bot's reply back
```

The bot ignores its own messages, so a bot-token "test" exercises nothing. Only a real user
message through Socket Mode drives the full path — which is where the bugs actually live. One
example: "chart it" routed to the image generator and drew a fake chart with invented numbers.
Every unit test passed; calling the text handler directly sails straight past the router.

## The bot_id carve-out

Messages posted with the user token still carry a `bot_id`/`app_id` in Slack's record, which used
to make `classify_sender` label the test user `[bot]` and invalidate any sender-sensitive test.
`DEV_TREAT_BOT_IDS_AS_HUMAN` in `.env` (empty in prod) fixes that — harness posts now classify as
`[human]`, and verdict reasons come back naming the human.

Residual, dev-only: `get_thread_history` gates mention cleaning on raw `is_bot = bool(msg.get("bot_id"))`
and does not consult the carve-out, so a harness post keeps `<@U…>` raw instead of rendering the
display name in rebuilt history. Author-name prefixes are unaffected. Prod can't hit it.

## Contamination — the standing trap

Repeated live runs poison the channel ring: the bot's own prior answers become "evidence" that the
next similar question is for it, and a restart backfills that residue right back in. This retired
the previous test channel. Space runs out, vary the wording, and read the verdict *reasons*, not
just the outcomes.

## Prefer a clean probe for classifier work

For participation/attribution behavior, skip live entirely — call the real utility model with a
hand-built input:

```python
OpenAIClient().classify_participation(text, signals=...)   # signals["channel_addressee_tail"]
                                                           # built from a freshly-seeded ChannelPulse
```

That isolates the exact scenario, free of `bot_id` artifacts and ring contamination.
