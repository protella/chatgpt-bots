"""Live thread-activity index feed (single-stream P1, spec §4).

The feed sits on the raw Slack listeners, ahead of every participation/listening filter and
behind the own-message check. Its job is to notice that a root has replies we have not seen —
including replies whose root scrolled out of the fetch window years ago — so the bias is
always toward recording MORE: mutations mark dirty rather than guess, hints move forward only,
and the same event arriving twice (message + app_mention) must land as one row.
"""
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from config import config
from database import DatabaseManager
from slack_client.admission_watermark import FAILED, REPAIRED
from slack_client.event_handlers import activity_index
from slack_client.event_handlers.activity_index import (feed_thread_activity_index,
                                                        normalize_activity_event)
from slack_client.event_handlers.registration import SlackRegistrationMixin
from slack_client.utilities import SlackUtilitiesMixin

TEAM = "T1"
CH = "C1"
ROOT = "100.000100"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


class _Client(SlackUtilitiesMixin):
    def __init__(self, db, team=TEAM):
        super().__init__()
        self.db = db
        self.self_team_id = team
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = "A123"


@pytest.fixture
def client(temp_db):
    return _Client(temp_db)


@pytest.fixture
def wm():
    """The ticket singleton, reset afterwards: a channel left degraded is process-wide state."""
    instance = activity_index.admission_watermark.watermark
    instance.reset()
    yield instance
    instance.reset()


def _reply(ts="101.000100", root=ROOT, channel=CH, **extra):
    event = {"type": "message", "channel": channel, "channel_type": "channel",
             "user": "U1", "ts": ts, "thread_ts": root, "text": "a reply",
             "event_ts": ts}
    event.update(extra)
    return event


def _top_level(ts=ROOT, channel=CH, **extra):
    event = {"type": "message", "channel": channel, "channel_type": "channel",
             "user": "U1", "ts": ts, "text": "top level", "event_ts": ts}
    event.update(extra)
    return event


def _changed(message, event_ts="900.000100", channel=CH, channel_type="channel"):
    return {"type": "message", "subtype": "message_changed", "channel": channel,
            "channel_type": channel_type, "message": message, "event_ts": event_ts}


def _deleted(previous, deleted_ts, event_ts="900.000100", channel=CH):
    return {"type": "message", "subtype": "message_deleted", "channel": channel,
            "channel_type": "channel", "previous_message": previous,
            "deleted_ts": deleted_ts, "event_ts": event_ts}


async def _rows(db, channel=CH, team=TEAM):
    return await db.get_thread_activity_async(team, channel)


async def _row(db, root=ROOT, channel=CH, team=TEAM):
    return next((r for r in await _rows(db, channel, team) if r["root_ts"] == root), None)


# ------------------------------------------------------------------ payload shapes

async def test_reply_records_root_reply_and_event_ts(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    row = await _row(temp_db)
    assert row["last_observed_reply_ts"] == "101.000100"
    assert row["last_index_event_ts"] == "101.000100"
    assert row["dirty"] == 0


async def test_thread_broadcast_counts_as_reply_activity(client, temp_db):
    await feed_thread_activity_index(
        client, _reply(ts="102.000100", subtype="thread_broadcast"))
    row = await _row(temp_db)
    assert row["last_observed_reply_ts"] == "102.000100"
    assert row["dirty"] == 0


async def test_reply_edit_marks_dirty_with_the_outer_event_ts(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _changed(
        {"ts": "101.000100", "thread_ts": ROOT, "user": "U1", "text": "edited",
         "edited": {"user": "U1", "ts": "888.000100"}}))
    row = await _row(temp_db)
    assert row["dirty"] == 1
    # The envelope's event_ts is when the mutation happened; edited.ts is only the fallback.
    assert row["last_index_event_ts"] == "900.000100"
    assert row["last_observed_reply_ts"] == "101.000100"


async def test_reply_edit_falls_back_to_nested_edited_ts(client, temp_db):
    event = _changed({"ts": "101.000100", "thread_ts": ROOT, "user": "U1", "text": "x",
                      "edited": {"user": "U1", "ts": "888.000100"}})
    event.pop("event_ts")
    await feed_thread_activity_index(client, event)
    assert (await _row(temp_db))["last_index_event_ts"] == "888.000100"


async def test_reply_delete_marks_dirty_and_ignores_deleted_ts(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _deleted(
        {"ts": "101.000100", "thread_ts": ROOT, "user": "U1", "text": "gone"},
        deleted_ts="101.000100"))
    row = await _row(temp_db)
    assert row["dirty"] == 1
    # deleted_ts identifies WHICH message went; it is never the activity time.
    assert row["last_index_event_ts"] == "900.000100"


async def test_tombstoned_root_keeps_the_row_and_marks_it_dirty(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _changed(
        {"ts": ROOT, "thread_ts": ROOT, "subtype": "tombstone",
         "text": "This message was deleted."}))
    row = await _row(temp_db)
    assert row is not None
    assert row["dirty"] == 1
    assert row["last_observed_reply_ts"] == "101.000100"


async def test_tombstone_recognized_by_sentinel_text_alone(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": ROOT, "thread_ts": ROOT, "user": "U1",
         "text": "This message was deleted."}))
    assert (await _row(temp_db))["dirty"] == 1


async def test_top_level_non_threaded_message_writes_nothing(client, temp_db):
    await feed_thread_activity_index(client, _top_level())
    assert await _rows(temp_db) == []


async def test_non_threaded_deletion_writes_nothing(client, temp_db):
    await feed_thread_activity_index(client, _deleted(
        {"ts": "500.000100", "user": "U1", "text": "gone"}, deleted_ts="500.000100"))
    assert await _rows(temp_db) == []


# ---------------------------------------------------------- roots without a thread_ts
# A Slack root normally carries no thread_ts of its own: it advertises its thread through
# reply_count/latest_reply. Recognizing a root only by thread_ts meant edits and deletions of
# real threaded roots fell through and never marked their thread dirty.

async def test_root_edit_is_recognized_through_reply_count(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": ROOT, "user": "U1", "text": "edited root", "reply_count": 3}))
    row = await _row(temp_db)
    assert row is not None and row["dirty"] == 1
    assert row["last_index_event_ts"] == "900.000100"


async def test_root_edit_is_recognized_through_latest_reply_alone(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": ROOT, "user": "U1", "text": "edited root", "latest_reply": "101.000100"}))
    assert (await _row(temp_db))["dirty"] == 1


async def test_root_edit_is_recognized_because_the_index_already_knows_it(client, temp_db):
    """No hints at all in the payload — the index's own row is the evidence that this
    plain-looking top-level ts is a thread parent."""
    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _changed(
        {"ts": ROOT, "user": "U1", "text": "edited root"}))
    row = await _row(temp_db)
    assert row["dirty"] == 1
    assert row["last_observed_reply_ts"] == "101.000100"


async def test_raw_root_deletion_marks_the_thread_dirty(client, temp_db):
    """Deleting a root usually arrives as a tombstone, but a raw message_deleted must not fall
    through — the replies underneath it are still live."""
    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _deleted(
        {"ts": ROOT, "user": "U1", "text": "root", "reply_count": 1}, deleted_ts=ROOT))
    row = await _row(temp_db)
    assert row is not None
    assert row["dirty"] == 1
    assert row["last_index_event_ts"] == "900.000100"


async def test_a_plain_top_level_edit_still_writes_nothing(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": "500.000100", "user": "U1", "text": "typo fixed"}))
    assert await _rows(temp_db) == []


async def test_a_zero_reply_count_is_not_a_threading_hint(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": "500.000100", "user": "U1", "text": "typo fixed", "reply_count": 0}))
    assert await _rows(temp_db) == []


async def test_channel_type_absent_falls_back_to_the_id_prefix(client, temp_db):
    mention = _reply(ts="103.000100")
    mention.pop("channel_type")
    mention["type"] = "app_mention"
    await feed_thread_activity_index(client, mention)
    assert (await _row(temp_db))["last_observed_reply_ts"] == "103.000100"

    dm = _reply(ts="104.000100", channel="D999")
    dm.pop("channel_type")
    await feed_thread_activity_index(client, dm)
    assert await _rows(temp_db, channel="D999") == []


async def test_harmless_kinds_record_nothing_and_report_success(client, temp_db, wm):
    """Events with nothing to index. Their tickets complete OK — a shape we have no business
    recording must not take a channel out of service."""
    tickets = []
    for event in ({"type": "message", "channel": CH, "channel_type": "channel",
                   "thread_ts": ROOT},                       # no ts of its own
                  "not a dict",
                  {"type": "message", "subtype": "channel_join", "channel": CH,
                   "channel_type": "channel", "user": "U1", "ts": "500.000100"}):
        ticket = wm.issue(CH)
        tickets.append(ticket)
        await feed_thread_activity_index(client, event, ticket=ticket)

    assert await _rows(temp_db) == []
    assert [t.state for t in tickets] == ["ok", "ok", "ok"]
    assert wm.is_degraded(CH) is False


async def test_a_malformed_known_mutation_fails_its_observation(client, temp_db, wm, caplog):
    """r2-2: these are NOT unknown kinds, which is how they used to be filed. An edit or a
    deletion with no subject is a mutation that HAPPENED — its outer event_ts already advanced H —
    so reporting the index caught up would let a turn answer without it."""
    for event in (_changed(None), _deleted(None, deleted_ts=None),
                  _changed({"type": "message", "user": "U1", "text": "no ts here"})):
        ticket = wm.issue(CH)
        with caplog.at_level(logging.CRITICAL):
            await feed_thread_activity_index(client, event, ticket=ticket)
        assert ticket.state == "failed", f"{event} completed as if it were indexed"

    assert await _rows(temp_db) == []
    assert wm.is_degraded(CH) is True


async def test_an_UNEXPECTED_normalizer_failure_fails_the_observation_too(client, temp_db, wm,
                                                                          monkeypatch, caplog):
    """r3-6. Only ValueError failed closed; every other exception was logged at WARNING and
    execution fell through to `complete_ok` — so a normalizer bug or an unforeseen payload shape
    certified that the index had caught up on an event nobody indexed, and every turn in that
    window answered from a stream quietly missing a thread. The declared-harmless outcomes are the
    ones that may pass; a surprise is not one of them."""
    def _boom(*a, **k):
        raise KeyError("a field the normalizer assumed was always there")

    monkeypatch.setattr(activity_index, "normalize_slack_event", _boom)
    ticket = wm.issue(CH)
    with caplog.at_level(logging.CRITICAL):
        await feed_thread_activity_index(client, _reply(), ticket=ticket)

    assert ticket.state == "failed", "an unexpected exception certified the index caught up"
    assert wm.is_degraded(CH) is True
    assert await _rows(temp_db) == []
    assert "KeyError" in "\n".join(r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ sender semantics

async def test_own_reply_is_never_indexed(client, temp_db):
    await feed_thread_activity_index(client, _reply(user="UBOT"))
    await feed_thread_activity_index(client, _reply(ts="102.000100", user=None,
                                                    bot_id="BBOT"))
    assert await _rows(temp_db) == []


async def test_nested_own_message_is_skipped_on_edits(client, temp_db):
    """Our own streaming edits arrive as high-volume message_changed; the own-message check
    has to read the NESTED message, since the envelope carries no author at all."""
    await feed_thread_activity_index(client, _changed(
        {"ts": "101.000100", "thread_ts": ROOT, "user": "UBOT", "text": "streaming"}))
    assert await _rows(temp_db) == []


async def test_dev_allowlisted_bot_is_indexed_as_a_human(client, temp_db, monkeypatch):
    monkeypatch.setattr(config, "dev_treat_bot_ids_as_human", ["B999"])
    event = _reply(ts="105.000100", user=None, bot_id="B999")
    assert normalize_activity_event(client, event).sender_type == "human"
    await feed_thread_activity_index(client, event)
    assert (await _row(temp_db))["last_observed_reply_ts"] == "105.000100"


async def test_other_bot_replies_are_indexed(client, temp_db):
    event = _reply(ts="106.000100", user=None, bot_id="BOTHER")
    assert normalize_activity_event(client, event).sender_type == "other_bot"
    await feed_thread_activity_index(client, event)
    assert (await _row(temp_db))["last_observed_reply_ts"] == "106.000100"


# ------------------------------------------------------------------ surfaces

async def test_mpim_is_indexed_and_im_is_skipped(client, temp_db):
    await feed_thread_activity_index(
        client, _reply(ts="107.000100", channel="G777", channel_type="mpim"))
    assert (await _row(temp_db, channel="G777"))["last_observed_reply_ts"] == "107.000100"

    await feed_thread_activity_index(
        client, _reply(ts="108.000100", channel="D777", channel_type="im"))
    assert await _rows(temp_db, channel="D777") == []


async def test_no_team_id_means_no_row(temp_db):
    client = _Client(temp_db)
    client.self_team_id = None
    await feed_thread_activity_index(client, _reply())
    assert await _rows(temp_db) == []


# ------------------------------------------------------------------ idempotence & order

async def test_message_and_app_mention_double_ingestion_is_idempotent(client, temp_db):
    event = _reply(ts="109.000100")
    mention = dict(event)
    mention["type"] = "app_mention"
    mention.pop("channel_type")
    await feed_thread_activity_index(client, event)
    await feed_thread_activity_index(client, mention)
    rows = await _rows(temp_db)
    assert len(rows) == 1
    assert rows[0]["last_observed_reply_ts"] == "109.000100"
    assert rows[0]["dirty"] == 0


async def test_out_of_order_delivery_never_walks_a_hint_backward(client, temp_db):
    await feed_thread_activity_index(client, _reply(ts="1000.000200"))
    await feed_thread_activity_index(client, _reply(ts="999.999900"))
    row = await _row(temp_db)
    assert row["last_observed_reply_ts"] == "1000.000200"
    assert row["last_index_event_ts"] == "1000.000200"


async def test_dirty_survives_later_replies_until_compare_and_cleared(client, temp_db):
    await feed_thread_activity_index(client, _changed(
        {"ts": "101.000100", "thread_ts": ROOT, "user": "U1", "text": "edited"},
        event_ts="900.000100"))
    await feed_thread_activity_index(client, _reply(ts="901.000100"))
    assert (await _row(temp_db))["dirty"] == 1

    # A reader that saw an older event must not clear what it never looked at.
    assert await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, "900.000100") is False
    assert (await _row(temp_db))["dirty"] == 1
    assert await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, "901.000100") is True
    assert (await _row(temp_db))["dirty"] == 0


# ------------------------------------------------------------------ seeding & safety

async def test_first_feed_seeds_channel_coverage_once(client, temp_db, monkeypatch):
    seeds = []
    original = temp_db.seed_channel_coverage_async

    async def _spy(team, channel, start_ts):
        seeds.append((team, channel, start_ts))
        return await original(team, channel, start_ts)

    monkeypatch.setattr(temp_db, "seed_channel_coverage_async", _spy)
    # Even a message the index has nothing to record for proves the channel is live.
    await feed_thread_activity_index(client, _top_level())
    await feed_thread_activity_index(client, _reply())
    assert len(seeds) == 1
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert row["bootstrap_status"] == "pending"
    assert float(row["inventory_start_ts"]) > 0


async def test_feed_never_raises_when_the_database_fails(client, monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(client.db, "record_thread_activity_async", _boom)
    monkeypatch.setattr(client.db, "seed_channel_coverage_async", _boom)
    await feed_thread_activity_index(client, _reply())


async def test_feed_never_raises_on_a_hostile_client(temp_db):
    class _Hostile:
        self_team_id = TEAM
        db = None

        def is_own_message(self, msg):
            raise RuntimeError("classification exploded")

    await feed_thread_activity_index(_Hostile(), _reply())


# ------------------------------------------------------------------ listener wiring

class _RegBot(SlackRegistrationMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass

    def __init__(self, db):
        self.db = db
        self.self_team_id = TEAM
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = None
        self._handle_slack_message = AsyncMock()
        self._handle_channel_message = AsyncMock()
        self._register_settings_handlers = MagicMock()
        self._handlers = {}
        self.app = MagicMock()

        def _event(name):
            def _decorator(fn):
                self._handlers[name] = fn
                return fn
            return _decorator

        self.app.event = _event
        self._register_handlers()


async def test_both_listeners_feed_the_index_and_still_dispatch(temp_db, monkeypatch):
    monkeypatch.setattr(config, "enable_channel_listening", True)
    bot = _RegBot(temp_db)
    event = _reply(ts="120.000100")
    mention = dict(event)
    mention.pop("channel_type")

    await bot._handlers["message"](event=event, say=None, client=None)
    await bot._handlers["app_mention"](event=mention, say=None, client=None)

    rows = await _rows(temp_db)
    assert len(rows) == 1
    assert rows[0]["last_observed_reply_ts"] == "120.000100"
    bot._handle_channel_message.assert_awaited_once()
    bot._handle_slack_message.assert_awaited_once()


async def test_the_index_is_fed_even_when_channel_listening_is_off(temp_db, monkeypatch):
    """The index is not participation: a channel we never speak in still has threads whose
    replies the next turn has to know about."""
    monkeypatch.setattr(config, "enable_channel_listening", False)
    bot = _RegBot(temp_db)

    await bot._handlers["message"](event=_reply(ts="121.000100"), say=None, client=None)

    assert (await _row(temp_db))["last_observed_reply_ts"] == "121.000100"
    bot._handle_channel_message.assert_not_awaited()


def test_a_broken_index_import_is_logged_as_an_error():
    """The index is the only route to pre-boundary roots, so it must never disappear quietly
    because of a packaging or import-cycle defect. caplog cannot witness that: the app's
    handlers hang off `slack_bot.*` with propagate=False, so this listens on the configured
    logger itself — which is also the assertion that the module uses it."""
    import builtins
    import importlib

    import slack_client.event_handlers.registration as registration
    from logger import setup_logger

    records = []

    class _Capture(logging.Filter):
        # A FILTER, not a handler: reloading registration re-enters setup_logger, and an
        # uncached name there runs `handlers.clear()` — which would drop a test handler before
        # the error is ever emitted. Filters are left alone, so this survives suite order.
        def filter(self, record):
            records.append(record)
            return True

    configured = setup_logger(name="slack_bot.Registration")
    capture = _Capture()
    configured.addFilter(capture)
    level, disabled = configured.level, logging.Logger.manager.disable
    configured.setLevel(logging.ERROR)
    logging.Logger.manager.disable = logging.NOTSET
    real_import = builtins.__import__

    def _broken(name, *args, **kwargs):
        if name.endswith("activity_index"):
            raise ImportError("simulated import cycle")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = _broken
        importlib.reload(registration)
        assert registration.feed_thread_activity_index is None
        assert registration.logger is configured
        assert any(r.levelno >= logging.ERROR and "ImportError" in r.getMessage()
                   for r in records)
    finally:
        builtins.__import__ = real_import
        configured.removeFilter(capture)
        configured.setLevel(level)
        logging.Logger.manager.disable = disabled
        importlib.reload(registration)

    assert registration.feed_thread_activity_index is not None


async def test_a_failed_seed_is_retried_on_the_next_event(client, temp_db, monkeypatch):
    """A failed seed leaves the channel unseeded, so a later event tries again.

    Each event also retries in place before giving up (a contended WAL writer is the common
    cause and it clears in milliseconds), so the call count is attempts × events — what matters
    is that the channel never gets marked seeded on a failure.
    """
    calls = []

    async def _flaky(team, channel, start_ts):
        calls.append(channel)
        raise RuntimeError("transient")

    monkeypatch.setattr(temp_db, "seed_channel_coverage_async", _flaky)
    monkeypatch.setattr(activity_index, "_FEED_RETRY_BASE_SECONDS", 0)
    await feed_thread_activity_index(client, _reply())
    first = len(calls)
    await feed_thread_activity_index(client, _reply(ts="110.000100"))
    assert first == activity_index._FEED_ATTEMPTS
    assert len(calls) == 2 * activity_index._FEED_ATTEMPTS
    assert calls == [CH] * len(calls)
    assert CH not in activity_index._seeded_channels(client)


# ------------------------------------------------- anonymous deletions (the receipt oracle)
# Slack's message_deleted and tombstone payloads routinely omit user AND bot_id, so the
# own-message check in front of the feed fails OPEN. Live, that meant the bot's own
# housekeeping deletes read as channel activity: every image post dirtied its own thread when
# the "Uploading…" indicator came off, and deleting a bot part recreated a row we had cleared.

def _anon_deleted(ts, root=ROOT, **extra):
    previous = {"ts": ts, "thread_ts": root, "text": "gone"}
    previous.update(extra)
    return _deleted(previous, deleted_ts=ts)


def _anon_tombstone(ts=ROOT, root=ROOT):
    return _changed({"ts": ts, "thread_ts": root, "subtype": "tombstone",
                     "text": "This message was deleted."})


async def _own_post(db, ts, root=ROOT, receipt_class="assistant_reply"):
    await db.register_receipt_async(TEAM, CH, ts, "sess:turn-1", "finalized",
                                    thread_root_ts=root, receipt_class=receipt_class)


def test_an_anonymous_deletion_asks_for_the_deleted_ts(client):
    obs = normalize_activity_event(client, _anon_deleted("101.000100"))
    assert obs.owner_probe_ts == "101.000100"


def test_an_anonymous_tombstone_asks_for_the_nested_message_ts(client):
    obs = normalize_activity_event(client, _anon_tombstone())
    assert obs.owner_probe_ts == ROOT


@pytest.mark.parametrize("identity", [{"user": "U1"}, {"bot_id": "BOTHER"},
                                      {"app_id": "AOTHER"}])
def test_a_deletion_that_names_its_sender_never_probes(client, identity):
    obs = normalize_activity_event(client, _anon_deleted("101.000100", **identity))
    assert obs.owner_probe_ts is None


def test_an_anonymous_edit_never_probes(client):
    """Edits carry their author, and a DB read here would sit in front of every reply."""
    obs = normalize_activity_event(client, _changed(
        {"ts": "101.000100", "thread_ts": ROOT, "text": "edited"}))
    assert obs.owner_probe_ts is None


async def test_our_own_anonymous_deletion_is_not_channel_activity(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    await _own_post(temp_db, "102.000100")

    await feed_thread_activity_index(client, _anon_deleted("102.000100"))

    row = await _row(temp_db)
    assert row["dirty"] == 0
    assert row["last_observed_reply_ts"] == "101.000100"


async def test_somebody_elses_anonymous_deletion_still_marks_dirty(client, temp_db):
    """No receipt means we never posted it, so the deletion is real channel activity."""
    await feed_thread_activity_index(client, _reply())

    await feed_thread_activity_index(client, _anon_deleted("102.000100"))

    assert (await _row(temp_db))["dirty"] == 1


async def test_our_own_anonymous_tombstone_is_skipped(client, temp_db):
    await _own_post(temp_db, ROOT, root=ROOT)

    await feed_thread_activity_index(client, _anon_tombstone())

    assert await _rows(temp_db) == []


async def test_a_receipt_in_any_state_counts_as_ours(client, temp_db):
    await feed_thread_activity_index(client, _reply())
    await temp_db.register_receipt_async(TEAM, CH, "102.000100", "sess:turn-1", "chrome",
                                         thread_root_ts=ROOT, receipt_class="chrome")

    await feed_thread_activity_index(client, _anon_deleted("102.000100"))

    assert (await _row(temp_db))["dirty"] == 0


async def test_the_image_indicator_delete_no_longer_dirties_its_thread(client, temp_db):
    """The live regression: an image post's "Uploading…" placeholder is our own threaded
    message, and removing it arrives as an identity-less deletion."""
    await feed_thread_activity_index(client, _reply())
    await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT,
                                           if_event_ts_equals="101.000100")
    indicator_ts = "103.000100"
    await _own_post(temp_db, indicator_ts, receipt_class="chrome")

    await feed_thread_activity_index(client, _anon_deleted(indicator_ts))

    assert (await _row(temp_db))["dirty"] == 0


async def test_the_oracle_failing_open_still_records_the_deletion(client, temp_db):
    """A missed root costs a thread; a spurious dirty costs one fetch. Fail toward the fetch."""
    await feed_thread_activity_index(client, _reply())
    temp_db.get_receipt_async = AsyncMock(side_effect=RuntimeError("db down"))

    await feed_thread_activity_index(client, _anon_deleted("102.000100"))

    assert (await _row(temp_db))["dirty"] == 1


async def test_the_reply_path_never_reads_a_receipt(client, temp_db):
    probe = AsyncMock(return_value=None)
    temp_db.get_receipt_async = probe

    await feed_thread_activity_index(client, _reply())
    await feed_thread_activity_index(client, _changed(
        {"ts": "101.000100", "thread_ts": ROOT, "user": "U1", "text": "edited"}))

    probe.assert_not_called()


def test_the_index_logs_into_the_configured_slack_bot_tree():
    """Handlers hang off `slack_bot.*` with propagate=False, so a bare getLogger(__name__)
    wrote every coverage-sweep and index line to nowhere."""
    assert activity_index.logger.name.startswith("slack_bot.")
    assert activity_index.logger.handlers
    assert activity_index.logger.propagate is False


# ------------------------------- W1: the write path after the mutation half was removed

async def test_the_activity_write_survives_the_mutation_removal(client, temp_db, wm):
    """T13. `_apply_observation` used to commit an index row and a mutation observation as ONE
    ticketed transaction. W1 removes the mutation half, so it calls the ALREADY-EXISTING
    `record_thread_activity_async` — the same monotonic merge, through the same upsert.

    The ticket contract is what must not have moved with it: the write still RAISES on failure
    so the admission ticket fails, a cancellation leaves no half-state, and a replay is still
    idempotent. All four cases, because the interesting failure is the one where a swallowed
    error lets a turn believe the index caught up on a write that never landed.
    """
    # 1. A normal upsert lands, through the surviving accessor.
    calls = []
    original = temp_db.record_thread_activity_async

    async def _spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    temp_db.record_thread_activity_async = _spy
    ticket = wm.issue(CH)
    await feed_thread_activity_index(client, _reply(), ticket=ticket)
    assert calls and calls[0]["root_ts"] == ROOT and calls[0]["reply_ts"] == "101.000100"
    assert ticket.state != FAILED
    rows = await _rows(temp_db)
    assert rows and rows[0]["last_observed_reply_ts"] == "101.000100"

    # 2. A DB error PROPAGATES, so the ticket fails. The feed retries in place first, so this
    #    raises on every attempt rather than once.
    async def _boom(**kwargs):
        raise RuntimeError("WAL writer gone")

    temp_db.record_thread_activity_async = _boom
    failing = wm.issue(CH)
    await feed_thread_activity_index(client, _reply(ts="102.000100"), ticket=failing)
    assert failing.state == FAILED, "a lost index write must fail its ticket, not pass quietly"

    # 3. A cancellation AT THE WRITE SEAM — not before it. The accessor is autocommit, so the
    #    interesting instant is after the statement has been handed to SQLite, where a naive
    #    reading would expect a half-committed row. Cancelling before the call would only prove
    #    "we never wrote", which is not the claim.
    import asyncio

    seam_reached = {"hit": False}

    async def _cancelled_at_seam(**kwargs):
        # Let the real write run to completion, THEN cancel where the awaiting feed resumes.
        await original(**kwargs)
        seam_reached["hit"] = True
        raise asyncio.CancelledError

    temp_db.record_thread_activity_async = _cancelled_at_seam
    cancelled_ticket = wm.issue(CH)
    with pytest.raises(asyncio.CancelledError):
        await feed_thread_activity_index(client, _reply(ts="103.000100"),
                                         ticket=cancelled_ticket)
    assert seam_reached["hit"], "the cancellation must land after the write, not before it"
    # The ticket fails rather than staying pending: every later turn whose frontier includes it
    # would otherwise wait out the drain timeout.
    assert cancelled_ticket.state == FAILED
    # And the row that DID land is a complete row, not a torn one — autocommit means the write
    # either happened or did not, and recovery is the replay below rather than a repair.
    landed = {r["root_ts"]: dict(r) for r in await _rows(temp_db)}
    assert landed[ROOT]["last_observed_reply_ts"] == "103.000100"

    #    THE RETAINED RETRY LIFECYCLE, which is the point of failing the ticket rather than
    #    swallowing: the feed hands `complete_failed` a `retry` closure, and the repair worker is
    #    what turns a FAILED ticket into a REPAIRED one and brings the channel back into service.
    #    A fresh-ticket resubmit (below) proves idempotency; it does NOT prove that the ORIGINAL
    #    failure is ever cleared, and a channel left degraded fails every later turn's drain.
    assert wm.is_degraded(CH), "a failed observation must degrade the channel until repaired"
    assert cancelled_ticket.retry is not None, "nothing retained to replay"

    temp_db.record_thread_activity_async = original
    repaired_any, remaining = await wm._repair_pass()
    assert repaired_any and remaining == 0
    assert cancelled_ticket.state == REPAIRED
    assert not wm.is_degraded(CH), "the channel must leave degraded state once repaired"

    #    Idempotent RECOVERY after a write that may already have landed: replaying the same
    #    event over the committed row changes nothing.
    before_replay = [dict(r) for r in await _rows(temp_db)]
    await feed_thread_activity_index(client, _reply(ts="103.000100"), ticket=wm.issue(CH))
    assert [dict(r) for r in await _rows(temp_db)] == before_replay

    # 4. A replay is idempotent — the same event twice is one row, at the same values.
    temp_db.record_thread_activity_async = original
    await feed_thread_activity_index(client, _reply(ts="104.000100"), ticket=wm.issue(CH))
    once = await _rows(temp_db)
    await feed_thread_activity_index(client, _reply(ts="104.000100"), ticket=wm.issue(CH))
    twice = await _rows(temp_db)
    assert len(once) == len(twice)
    assert [dict(r) for r in once] == [dict(r) for r in twice]


async def test_our_own_housekeeping_delete_is_not_human_activity(client, temp_db, wm):
    """T14. `_deletion_was_ours` is NOT dead code, and this pins the decision to keep it.

    It has a live caller in `_index_row`, where the receipt ledger is the oracle for an
    ANONYMOUS deletion: a row in any state means we posted that ts, so removing it is our own
    housekeeping — the "Uploading…" indicator behind every image post is the loudest case — and
    not activity under a thread. Dropping it during a later cleanup would silently start
    recording our own tidying as somebody talking.
    """
    ours, theirs = "500.000100", "600.000100"
    await temp_db.register_receipt_async(TEAM, CH, ours, turn_id="t1", state="in_flight",
                                         thread_root_ts=ROOT,
                                         receipt_class="assistant_reply")

    # A deletion of a message WE posted records nothing.
    await feed_thread_activity_index(
        client, _deleted({"ts": ours, "thread_ts": ROOT}, ours), ticket=wm.issue(CH))
    assert await _rows(temp_db) == []

    # A third party's deletion does — the root comes back dirty so the next turn refetches it.
    await feed_thread_activity_index(
        client, _deleted({"ts": theirs, "thread_ts": ROOT}, theirs), ticket=wm.issue(CH))
    rows = await _rows(temp_db)
    assert rows and rows[0]["root_ts"] == ROOT and rows[0]["dirty"] == 1
