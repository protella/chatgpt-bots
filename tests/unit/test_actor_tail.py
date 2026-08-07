"""The per-thread actor tail — who spoke in a thread, and the one decision that depends on it.

ChannelPulse is retired: the channel stream is how a turn learns what was said. What survives is
structural. A thread continuation answers with no gate involved at all, and the replies fast path
that finds those threads scans only Slack's oldest page — so a second agent further down a long
thread is invisible to it. `thread_has_other_bot` is what sees that agent.

What it does with that sight narrowed when thread membership became the wake signal. It is no
longer a veto on WAKING: in an `on` channel a thread we have posted in wakes us whoever else is
in it. It decides STRICT status — the level-independent rule of us, one human, no other agents,
which is what still reaches a `mentions_only` channel and what still carries structural
authority. Everything in this file exists to keep that answer correct:

- the ring's shape and bounds (a bound that silently narrowed would push a second agent out of
  view and call a crowded thread strictly 1:1);
- the LIVE feed, which runs at the raw Slack listener rather than inside ambient ingest, because
  the strict test must not depend on whether ambient memory happens to be wired;
- the FETCH feed (reconcile_window), which hydrates the same rings from a turn's own Slack pages,
  and the generation counter that makes live writes win over a stale fetch;
- the mutation path: a deleted or tombstoned message stops counting as a speaker.
"""
from __future__ import annotations

import pytest

from config import config
from slack_client import actor_tail as tail_mod
from slack_client.actor_tail import ActorTail, TailRecord, tail_record
from slack_client.normalizer import normalize_slack_message


@pytest.fixture(autouse=True)
def _clean_singleton():
    """The module singleton is process-wide; every test starts from empty."""
    tail_mod.actor_tail.reset()
    yield
    tail_mod.actor_tail.reset()


def _human(t, ts, root, channel="C1"):
    t.record(channel, ts=ts, root_ts=root, is_bot=False, sender_type="human")


def _other_bot(t, ts, root, channel="C1"):
    t.record(channel, ts=ts, root_ts=root, is_bot=True, sender_type="other_bot")


def _self(t, ts, root, channel="C1"):
    t.record(channel, ts=ts, root_ts=root, is_bot=True, sender_type="self")


# --------------------------------------------------------------------------- structure

def test_a_root_and_its_replies_share_one_bucket():
    """The root asymmetry, deliberately preserved: Slack sends a root with no thread_ts, but the
    root files under its OWN ts so it lands with its replies. Filing it under None is how
    thread_has_other_bot silently stopped matching."""
    t = ActorTail()
    _human(t, "100.0", None)          # the root, as the raw payload presents it
    _human(t, "101.0", "100.0")
    _other_bot(t, "102.0", "100.0")
    assert [e.ts for e in t.entries("C1", "100.0")] == ["100.0", "101.0", "102.0"]
    assert [e.is_bot for e in t.entries("C1", "100.0")] == [False, False, True]


def test_the_ring_stores_actors_not_messages():
    """No text, no display name. Anything more would be storage nothing reads — and, for an
    untrusted display name, storage kept for no reason at all."""
    t = ActorTail()
    _human(t, "100.0", None)
    entry = t.entries("C1", "100.0")[0]
    assert set(vars(entry)) == {"ts", "is_bot", "sender_type"}


def test_record_is_idempotent_by_ts():
    t = ActorTail()
    _human(t, "100.0", None)
    _human(t, "101.0", "100.0")
    _human(t, "101.0", "100.0")       # a retry, or the same mention delivered twice
    assert [e.ts for e in t.entries("C1", "100.0")] == ["100.0", "101.0"]


def test_an_unparseable_ts_is_refused():
    t = ActorTail()
    t.record("C1", ts="not-a-ts", root_ts="100.0", is_bot=True, sender_type="other_bot")
    assert t.thread_has_other_bot("C1", "100.0") is False


def test_tail_record_derives_bot_and_root_from_a_normalized_message():
    client = _feed_host()
    msg = normalize_slack_message(
        client,
        {"ts": "101.0", "thread_ts": "100.0", "bot_id": "BOTHER", "username": "Other",
         "text": "hi"},
        team_id="T1", channel_id="C1", origin="history")
    rec = tail_record(msg)
    assert (rec.ts, rec.root_ts, rec.is_bot, rec.sender_type) \
        == ("101.0", "100.0", True, "other_bot")


# --------------------------------------------------------------------------- the reader

def test_only_a_bot_that_is_not_us_counts():
    t = ActorTail()
    _human(t, "100.0", None)
    _human(t, "101.0", "100.0")
    assert t.thread_has_other_bot("C1", "100.0") is False
    _self(t, "102.0", "100.0")        # our own reply is not a second agent
    assert t.thread_has_other_bot("C1", "100.0") is False
    _other_bot(t, "103.0", "100.0")
    assert t.thread_has_other_bot("C1", "100.0") is True


def test_a_cold_ring_degrades_to_false():
    """Nothing recorded is not evidence of absence, but it is the only safe answer: the fast path
    stays available and the replies scan remains the check."""
    assert ActorTail().thread_has_other_bot("C1", "999.0") is False


def test_threads_are_scoped_per_channel():
    t = ActorTail()
    _other_bot(t, "101.0", "100.0", channel="C1")
    assert t.thread_has_other_bot("C2", "100.0") is False


# --------------------------------------------------------------------------- bounds

def test_depth_bound_can_push_a_second_agent_out_of_view(monkeypatch):
    """Where the edge is. Narrowing participation_thread_tail drops an agent out of the window and
    re-opens the fast path, so the bound is pinned rather than assumed."""
    monkeypatch.setattr(config, "participation_thread_tail", 2)
    t = ActorTail()
    _human(t, "100.0", None)
    _other_bot(t, "101.0", "100.0")
    assert t.thread_has_other_bot("C1", "100.0") is True
    for i in range(2, 9):
        _human(t, f"10{i}.0", "100.0")
    assert t.thread_has_other_bot("C1", "100.0") is False


def test_zero_depth_disables_recording(monkeypatch):
    monkeypatch.setattr(config, "participation_thread_tail", 0)
    t = ActorTail()
    _human(t, "100.0", None)
    _other_bot(t, "101.0", "100.0")
    assert t.entries("C1", "100.0") == ()
    assert t.thread_has_other_bot("C1", "100.0") is False


def test_thread_lru_evicts_the_least_recently_touched(monkeypatch):
    monkeypatch.setattr(config, "actor_tail_threads_max", 2)
    t = ActorTail()
    for root in ("100.0", "200.0"):
        _human(t, root, None)
        _other_bot(t, root.replace("00", "01"), root)
    _human(t, "102.0", "100.0")       # touching thread 100 makes it the most recent
    _human(t, "300.0", None)
    _other_bot(t, "301.0", "300.0")
    assert t.thread_has_other_bot("C1", "200.0") is False   # evicted
    assert t.thread_has_other_bot("C1", "100.0") is True
    assert t.thread_has_other_bot("C1", "300.0") is True


def test_channel_lru_bound(monkeypatch):
    monkeypatch.setattr(config, "actor_tail_channels_max", 2)
    t = ActorTail()
    for ch in ("C1", "C2", "C3"):
        _human(t, "100.0", None, channel=ch)
        _other_bot(t, "101.0", "100.0", channel=ch)
    assert t.thread_has_other_bot("C1", "100.0") is False
    assert t.thread_has_other_bot("C3", "100.0") is True


def test_config_reads_the_renamed_keys():
    """The pulse-era names are gone; a stale fallback would keep reading a key nobody sets."""
    import inspect

    src = inspect.getsource(tail_mod)
    assert "pulse_thread_tails_max" not in src
    assert "pulse_thread_tail_channels_max" not in src
    assert "actor_tail_threads_max" in src and "actor_tail_channels_max" in src


# --------------------------------------------------------------------------- removal

def test_remove_forgets_one_ts_and_can_clear_the_verdict():
    t = ActorTail()
    _human(t, "100.0", None)
    _other_bot(t, "101.0", "100.0")
    assert t.remove("C1", "101.0") is True
    assert t.thread_has_other_bot("C1", "100.0") is False
    assert t.remove("C1", "101.0") is False     # already gone


# --------------------------------------------------------------------------- generation

def test_every_write_bumps_the_channel_generation():
    t = ActorTail()
    start = t.generation("C1")
    _human(t, "100.0", None)
    assert t.generation("C1") == start + 1
    t.remove("C1", "100.0")
    assert t.generation("C1") == start + 2
    assert t.generation("C2") == 0              # per channel


def test_reconcile_replaces_the_covered_interval():
    t = ActorTail()
    _other_bot(t, "150.0", "100.0")             # a live write inside the window
    gen = t.generation("C1")
    ok = t.reconcile_window(
        "C1",
        [TailRecord(root_ts="100.0", ts="110.0", is_bot=False, sender_type="human"),
         TailRecord(root_ts="100.0", ts="120.0", is_bot=False, sender_type="human")],
        window=("100.0", True, "200.0"), expected_generation=gen)
    assert ok is True
    assert [e.ts for e in t.entries("C1", "100.0")] == ["110.0", "120.0"]
    assert t.thread_has_other_bot("C1", "100.0") is False   # the fetch is authoritative here


def test_reconcile_leaves_entries_outside_the_window_alone():
    t = ActorTail()
    _other_bot(t, "500.0", "100.0")             # after H — not this fetch's business
    gen = t.generation("C1")
    t.reconcile_window(
        "C1", [TailRecord(root_ts="100.0", ts="110.0", is_bot=False, sender_type="human")],
        window=("100.0", True, "200.0"), expected_generation=gen)
    assert [e.ts for e in t.entries("C1", "100.0")] == ["110.0", "500.0"]
    assert t.thread_has_other_bot("C1", "100.0") is True


def test_reconcile_skips_when_a_live_event_landed_mid_fetch():
    """Live wins. What we hold is newer than what we fetched, so replacing the interval would
    forget a speaker Slack has already delivered; the next turn re-hydrates."""
    t = ActorTail()
    gen = t.generation("C1")
    _other_bot(t, "150.0", "100.0")             # arrives while the fetch is in flight
    ok = t.reconcile_window(
        "C1", [TailRecord(root_ts="100.0", ts="110.0", is_bot=False, sender_type="human")],
        window=("100.0", True, "200.0"), expected_generation=gen)
    assert ok is False
    assert t.thread_has_other_bot("C1", "100.0") is True


# --------------------------------------------------------------------- the live feed

def _feed_host():
    """The listener-level feed on a bare host: no ambient service, no DB, no Slack client — which
    is the point. The strict test must work in a process where none of those are wired."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class Host(SlackMessageEventsMixin):
        def __init__(self):
            self.bot_user_id = "UBOT"

        def is_own_message(self, e):
            return e.get("user") == "UBOT" or e.get("bot_id") == "BSELF"

        def classify_sender(self, e):
            if self.is_own_message(e):
                return "self"
            return "other_bot" if (e.get("bot_id") or e.get("app_id")) else "human"

        def log_debug(self, *a, **k):
            pass

    return Host()


def test_the_live_feed_records_a_second_agents_reply():
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "ts": "100.0", "user": "U1", "text": "root"})
    host._feed_actor_tail({"channel": "C1", "ts": "101.0", "thread_ts": "100.0", "user": "U1",
                           "text": "mine"})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is False
    host._feed_actor_tail({"channel": "C1", "ts": "102.0", "thread_ts": "100.0",
                           "subtype": "bot_message", "bot_id": "BCLAUDE", "text": "theirs"})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is True


def test_the_live_feed_dedupes_dual_delivery():
    """A mention arrives on both listeners. The second delivery is not a second speaker, and it
    must not bump the generation either — a turn's reconcile watches that number."""
    host = _feed_host()
    ev = {"channel": "C1", "ts": "100.0", "thread_ts": "99.0", "user": "U7", "text": "hey"}
    host._feed_actor_tail(ev)
    gen = tail_mod.generation("C1")
    host._feed_actor_tail(ev)
    assert tail_mod.generation("C1") == gen
    assert [e.ts for e in tail_mod.actor_tail.entries("C1", "99.0")] == ["100.0"]


def test_the_live_feed_skips_our_own_posts_and_churn():
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "ts": "100.0", "user": "UBOT", "text": "own echo"})
    host._feed_actor_tail({"channel": "C1", "ts": "101.0", "subtype": "channel_join",
                           "user": "U1", "text": "joined"})
    host._feed_actor_tail({"channel": "C1", "ts": "102.0", "subtype": "message_replied",
                           "user": "U1"})
    assert tail_mod.actor_tail.entries("C1", "100.0") == ()
    assert tail_mod.actor_tail.entries("C1", "101.0") == ()
    assert tail_mod.actor_tail.entries("C1", "102.0") == ()


def test_the_live_feed_ignores_dms():
    host = _feed_host()
    host._feed_actor_tail({"channel": "D123", "ts": "100.0", "subtype": "bot_message",
                           "bot_id": "BOTHER", "text": "hi"})
    assert tail_mod.thread_has_other_bot("D123", "100.0") is False


def test_the_live_feed_never_raises_on_a_malformed_event():
    host = _feed_host()
    host._feed_actor_tail(None)
    host._feed_actor_tail({})
    host._feed_actor_tail({"channel": "C1"})               # no ts
    host._feed_actor_tail({"ts": "100.0"})                 # no channel


# ------------------------------------------------------- the listener-level mutation path

def test_a_deletion_forgets_the_speaker():
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "ts": "100.0", "user": "U1", "text": "root"})
    host._feed_actor_tail({"channel": "C1", "ts": "101.0", "thread_ts": "100.0",
                           "subtype": "bot_message", "bot_id": "BCLAUDE", "text": "theirs"})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is True
    host._feed_actor_tail({"channel": "C1", "subtype": "message_deleted", "deleted_ts": "101.0",
                           "previous_message": {"ts": "101.0", "thread_ts": "100.0"}})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is False


def test_a_tombstoned_root_stops_counting():
    """Deleting a root that has replies arrives as message_changed carrying a tombstone, never as
    message_deleted. Treating it as an ordinary edit would keep a deleted speaker in the ring."""
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "ts": "100.0", "subtype": "bot_message",
                           "bot_id": "BCLAUDE", "text": "theirs"})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is True
    host._feed_actor_tail({"channel": "C1", "subtype": "message_changed",
                           "message": {"ts": "100.0", "subtype": "tombstone",
                                       "text": "This message was deleted."}})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is False


def test_a_tombstone_without_the_subtype_is_still_a_deletion():
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "ts": "100.0", "subtype": "bot_message",
                           "bot_id": "BCLAUDE", "text": "theirs"})
    host._feed_actor_tail({"channel": "C1", "subtype": "message_changed",
                           "message": {"ts": "100.0", "bot_id": "BCLAUDE",
                                       "text": "This message was deleted."}})
    assert tail_mod.thread_has_other_bot("C1", "100.0") is False


def test_an_edit_teaches_the_tail_about_a_message_it_never_saw():
    """An ordinary edit changes the words, not the speaker — but it is also the one chance to learn
    about a message posted before this process started."""
    host = _feed_host()
    host._feed_actor_tail({"channel": "C1", "subtype": "message_changed",
                           "message": {"ts": "100.0", "thread_ts": "99.0", "bot_id": "BCLAUDE",
                                       "text": "edited", "edited": {"ts": "150.0"}}})
    assert tail_mod.thread_has_other_bot("C1", "99.0") is True


def test_a_re_post_after_a_deletion_is_recorded_again():
    """The dedup set must not outlive the entry it deduped, or a ts that came back would be
    ignored forever."""
    host = _feed_host()
    ev = {"channel": "C1", "ts": "101.0", "thread_ts": "100.0", "subtype": "bot_message",
          "bot_id": "BCLAUDE", "text": "theirs"}
    host._feed_actor_tail(ev)
    host._feed_actor_tail({"channel": "C1", "subtype": "message_deleted", "deleted_ts": "101.0",
                           "previous_message": {"ts": "101.0"}})
    host._feed_actor_tail(ev)
    assert tail_mod.thread_has_other_bot("C1", "100.0") is True


# --------------------------------------------------------- the listeners + the strict test

def test_both_raw_listeners_feed_the_tail_before_any_await():
    """The feed sits with the watermark admission at the top of each listener — synchronous, ahead
    of the first await, and NOT inside ambient ingest. Inside `_ambient_ingest` it would be gated
    on a service that has nothing to do with whether a second agent is in the thread."""
    import inspect

    from slack_client.event_handlers.registration import SlackRegistrationMixin

    src = inspect.getsource(SlackRegistrationMixin._register_handlers)
    assert src.count("self._feed_actor_tail(event)") == 2
    for listener in ('@self.app.event("app_mention")', '@self.app.event("message")'):
        body = src[src.index(listener):]
        before = body[:body.index("self._feed_actor_tail(event)")]
        code = [ln for ln in before.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("await" in ln for ln in code), \
            f"{listener} awaits before feeding the tail"

    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    ambient = inspect.getsource(SlackMessageEventsMixin._ambient_ingest)
    assert "_feed_actor_tail" not in ambient


def test_the_actor_tail_is_read_at_the_continuation_decision():
    """Asserted on the source of the decision site: the surrounding handler needs a whole Slack
    client to run, and what matters is that the tail is consulted at exactly the point that
    decides STRICT status.

    It is no longer a veto on WAKING. In an `on` channel a thread we have posted in wakes us
    whoever else is in it, so a second agent no longer cancels the dispatch — it only
    disqualifies the thread from the level-independent strict 1:1 rule. So what the call must
    still gate is `strict_1to1`, and `strict_1to1` is what gates the direct continuation."""
    import inspect

    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    src = inspect.getsource(SlackMessageEventsMixin)
    idx = src.index("actor_tail.thread_has_other_bot(channel_id, thread_ts)")
    window = src[max(0, idx - 200):idx + 200]
    assert "strict_1to1 = (" in window
    assert "not actor_tail.thread_has_other_bot(channel_id, thread_ts)" in window
    assert "if strict_1to1:" in window
