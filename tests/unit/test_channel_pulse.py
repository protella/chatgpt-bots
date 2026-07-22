"""Phase E — ChannelPulse ambient awareness + response envelope.

Covers the ring buffer, once-only backfill, DM exclusion, deterministic envelope
rendering with current-thread exclusion, suffix (not system prompt) placement,
participation stats, feed-before-gates wiring, and flag gating.
"""

import time

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from slack_client.channel_pulse import ChannelPulse
from config import config


def _entry(ts, text="hello", thread_ts=None, name="Alice", sender="human"):
    return dict(ts=ts, thread_ts=thread_ts, user_id="U1", display_name=name,
                sender_type=sender, text=text, is_bot=sender != "human")


# ------------------------------------------------------------------ ring core

def test_ring_eviction_keeps_newest():
    p = ChannelPulse(size=3)
    for i in range(5):
        p.record("C1", **_entry(f"{i}.0", text=f"m{i}"))
    env = p.render_envelope("C1")
    assert "m2" in env and "m3" in env and "m4" in env
    assert "m0" not in env and "m1" not in env


def test_dm_and_malformed_excluded():
    p = ChannelPulse(size=5)
    p.record("D123", **_entry("1.0"))          # DM
    p.record(None, **_entry("2.0"))            # no channel
    p.record("C1", **_entry(None))             # no ts
    assert p.render_envelope("D123") == ""
    assert p.render_envelope("C1") == ""


def test_own_and_ignored_messages_still_recorded():
    # The buffer takes everything it's fed — gating happens in the event handler,
    # which feeds BEFORE the own-message/mode gates (see wiring test below).
    p = ChannelPulse(size=5)
    p.record("C1", **_entry("1.0", text="bot said", name="ChatGPT", sender="self"))
    p.record("C1", **_entry("2.0", text="claude said", name="Claude", sender="other_bot"))
    env = p.render_envelope("C1")
    assert "bot said" in env and "claude said" in env


# ---------------------------------------------------------- recent_speakers (F29)

def test_recent_speakers_distinct_newest_first_excludes_self():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("1.0", name="Alice"))
    p.record("C1", **_entry("2.0", name="Bob"))
    p.record("C1", **_entry("3.0", name="Alice"))                 # dupe (older instance drops)
    p.record("C1", **_entry("4.0", name="ChatGPT", sender="self"))  # bot's own reply excluded
    p.record("C1", **_entry("5.0", name="Claude", sender="other_bot"))  # other bot KEPT
    assert p.recent_speakers("C1") == ["Claude", "Alice", "Bob"]


def test_recent_speakers_respects_limit():
    p = ChannelPulse(size=10)
    for i in range(5):
        p.record("C1", **_entry(f"{i}.0", name=f"U{i}"))
    assert p.recent_speakers("C1", limit=2) == ["U4", "U3"]


def test_recent_speakers_unknown_channel_and_dm():
    p = ChannelPulse(size=10)
    assert p.recent_speakers("C-nope") == []
    p.record("D1", **_entry("1.0", name="Alice"))  # DM never recorded
    assert p.recent_speakers("D1") == []


def test_recent_speakers_neutralizes_bracket_names():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("1.0", name="Claude [bot]"))
    # Brackets are folded so the name can't forge/close a [Channel people: …] frame.
    assert p.recent_speakers("C1") == ["Claude (bot)"]


# ------------------------------------------------------------------- backfill

@pytest.mark.asyncio
async def test_backfill_once_per_channel():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock(return_value={"messages": [
        {"ts": "2.0", "text": "second", "user": "U2"},
        {"ts": "1.0", "text": "first", "user": "U1"},  # Slack returns newest first
    ]})
    bot = MagicMock()
    bot.classify_sender = MagicMock(return_value="human")
    bot.user_cache = {}

    await p.ensure_backfill("C1", client, bot)
    await p.ensure_backfill("C1", client, bot)  # second call must not re-fetch
    assert client.conversations_history.await_count == 1

    env = p.render_envelope("C1")
    # oldest -> newest ordering after backfill
    assert env.index("first") < env.index("second")


@pytest.mark.asyncio
async def test_backfill_failure_is_silent_and_nonfatal():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock(side_effect=RuntimeError("boom"))
    await p.ensure_backfill("C1", client, MagicMock())
    assert p.render_envelope("C1") == ""  # empty, no raise


@pytest.mark.asyncio
async def test_backfill_skips_dms():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock()
    await p.ensure_backfill("D999", client, MagicMock())
    client.conversations_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_keeps_peers_and_clean_self_but_drops_own_chrome():
    # F47: cold-start backfill must record other assistants + our clean final replies, but NOT our
    # own UI chrome (status/checklist/"Settings available"/UI-helper block) — recorded as [self]
    # addressee evidence, chrome would push a real Claude exchange out of the ring and masquerade
    # as a continuation with US (the exact attribution bug). And it must retain the content-bearing
    # subtypes the live feed keeps (thread_broadcast, file_share), dropping only churn.
    from message_markers import CHECKLIST_STATUS_MARKER
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    def _classify(m):
        if m.get("bot_id") == "BSELF":
            return "self"
        if m.get("bot_id"):
            return "other_bot"
        return "human"

    bot = MagicMock()
    bot.classify_sender = _classify
    bot.user_cache = {}
    bot._PULSE_FEED_SKIP_SUBTYPES = SlackMessageEventsMixin._PULSE_FEED_SKIP_SUBTYPES

    messages = [
        {"ts": "1.0", "bot_id": "BCLAUDE", "username": "Claude", "text": "Once upon a time a fox"},
        {"ts": "2.0", "bot_id": "BSELF", "username": "ChatGPT",
         "text": ":hourglass_flowing_sand: Thinking..."},                  # our status chrome
        {"ts": "3.0", "bot_id": "BSELF", "username": "ChatGPT",
         "text": "✓ mounted the file" + CHECKLIST_STATUS_MARKER},          # our checklist chrome
        {"ts": "4.0", "bot_id": "BSELF", "username": "ChatGPT", "text": "Settings available"},  # chrome
        {"ts": "5.0", "bot_id": "BSELF", "username": "ChatGPT", "text": "Rate this response",
         "blocks": [{"type": "actions", "elements": [{"action_id": "open_channel_settings"}]}]},  # UI-helper
        {"ts": "6.0", "bot_id": "BSELF", "username": "ChatGPT", "text": "Sure, here is the answer."},  # clean self
        {"ts": "7.0", "subtype": "thread_broadcast", "bot_id": "BCLAUDE", "username": "Claude",
         "text": "broadcasting my reply", "thread_ts": "1.0"},             # content subtype: kept
        {"ts": "8.0", "subtype": "file_share", "user": "U1", "text": "here is the file",
         "files": [{"name": "a.pdf", "mimetype": "application/pdf"}]},     # content subtype: kept
        {"ts": "9.0", "subtype": "channel_join", "user": "U2", "text": "has joined"},  # churn: dropped
    ]
    client = MagicMock()
    client.conversations_history = AsyncMock(return_value={"messages": list(reversed(messages))})
    p = ChannelPulse(size=30)
    await p.ensure_backfill("C1", client, bot)
    env = p.render_envelope("C1")
    # Kept: the other assistant, our clean reply, and the content-bearing subtypes.
    assert "Once upon a time a fox" in env
    assert "Sure, here is the answer." in env
    assert "broadcasting my reply" in env
    assert "here is the file" in env
    # Dropped: every flavor of our own chrome, and churn.
    assert "Thinking" not in env
    assert "mounted the file" not in env
    assert "Settings available" not in env
    assert "Rate this response" not in env
    assert "has joined" not in env


# ------------------------------------------------------------------- envelope

def _seeded_pulse():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="deploy question here"))
    p.record("C1", **_entry("101.0", text="reply inside deploy thread",
                            thread_ts="100.0", name="Bob"))
    p.record("C1", **_entry("102.0", text="unrelated top level", name="Cara"))
    return p


def test_envelope_thread_vs_top_level_lines(monkeypatch):
    # About the thread-vs-top-level line format, not F10 stamps — disable the stamp so the
    # exact-line assertions stay readable (stamped-envelope coverage lives in F10 tests).
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    env = _seeded_pulse().render_envelope("C1")
    # Alice's root carries a reply, so it renders with the has-thread hint; Cara's doesn't.
    assert '- Alice (top-level, has thread): deploy question here' in env
    assert '- Bob (in thread "deploy question here…"): reply inside deploy thread' in env
    assert '- Cara (top-level): unrelated top level' in env


# ------------------------------------------------------- has-thread marker (thread discovery)

def test_live_reply_marks_its_parent_as_threaded():
    """A live message event carries no reply_count — the reply itself is the only signal
    that its parent has a discussion under it."""
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="root msg"))
    assert p._buffers["C1"][0]["has_thread"] is False
    p.record("C1", **_entry("101.0", text="a reply", thread_ts="100.0", name="Bob"))
    assert p._buffers["C1"][0]["has_thread"] is True


def test_reply_whose_parent_aged_out_is_harmless():
    p = ChannelPulse(size=2)
    p.record("C1", **_entry("100.0", text="root msg"))
    p.record("C1", **_entry("200.0", text="filler"))
    p.record("C1", **_entry("201.0", text="filler2"))       # evicts the root
    p.record("C1", **_entry("101.0", text="late reply", thread_ts="100.0"))
    assert all(not e["has_thread"] for e in p._buffers["C1"])


def test_backfill_reply_count_marks_cold_ring(monkeypatch):
    """The cold-start case: conversations.history returns parents only, so without
    reply_count a message with 40 replies would look identical to a dead one-liner."""
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="busy root"), reply_count=40)
    p.record("C1", **_entry("101.0", text="quiet root"), reply_count=0)
    env = p.render_envelope("C1")
    assert "- Alice (top-level, has thread): busy root" in env
    assert "- Alice (top-level): quiet root" in env


def test_reply_count_never_marks_a_reply_itself():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("101.0", text="a reply", thread_ts="100.0"), reply_count=5)
    assert p._buffers["C1"][0]["has_thread"] is False


def test_root_normalized_when_slack_sets_thread_ts_equal_to_ts():
    """Slack stamps thread_ts == ts on a thread ROOT; record() normalizes that to None, so
    the root must still take the marker from reply_count."""
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="root", thread_ts="100.0"), reply_count=3)
    e = p._buffers["C1"][0]
    assert e["thread_ts"] is None and e["has_thread"] is True


def test_marker_survives_an_edit_re_record():
    """An EDIT removes the entry and re-records it from a message_changed payload that
    carries no reply_count, and the earlier replies never run again — the marker has to
    come from remembered root state or a typo fix would erase the thread hint."""
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="root with a thread"))
    p.record("C1", **_entry("101.0", text="a reply", thread_ts="100.0", name="Bob"))
    assert p._buffers["C1"][0]["has_thread"] is True

    p.remove_message("C1", "100.0")                      # edit: drop the stale entry…
    p.record("C1", **_entry("100.0", text="root with a thread (typo fixed)"))  # …re-feed
    reroot = [e for e in p._buffers["C1"] if e["ts"] == "100.0"][0]
    assert reroot["has_thread"] is True


def test_reply_recorded_before_its_parent_still_marks_it():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("101.0", text="a reply", thread_ts="100.0", name="Bob"))
    p.record("C1", **_entry("100.0", text="parent arrives late"))
    parent = [e for e in p._buffers["C1"] if e["ts"] == "100.0"][0]
    assert parent["has_thread"] is True


def test_known_thread_roots_are_bounded_by_ring_size():
    p = ChannelPulse(size=3)
    for i in range(10):
        p.record("C1", **_entry(f"{i}.1", text="r", thread_ts=f"{i}.0"))
    assert len(p._thread_roots["C1"]) <= 3


def test_envelope_excludes_current_thread():
    env = _seeded_pulse().render_envelope("C1", exclude_thread_ts="100.0")
    # the thread root AND its replies are the model's full context already
    assert "deploy question" not in env and "reply inside" not in env
    assert "unrelated top level" in env


def test_envelope_cap_and_zero():
    p = ChannelPulse(size=10)
    for i in range(8):
        p.record("C1", **_entry(f"{i}.0", text=f"m{i}"))
    capped = p.render_envelope("C1", max_lines=3)
    assert len([ln for ln in capped.splitlines() if ln.startswith("- ")]) == 3
    assert "m7" in capped and "m0" not in capped  # newest kept
    assert p.render_envelope("C1", max_lines=0) == ""


def test_envelope_deterministic_given_same_state():
    a, b = _seeded_pulse(), _seeded_pulse()
    assert a.render_envelope("C1") == b.render_envelope("C1")
    assert a.render_envelope("C1") == a.render_envelope("C1")


def test_envelope_header_is_modest_peripheral():
    # F47: the general channel-activity envelope is peripheral again. The authoritative
    # "who addresses whom" record is the SEPARATE addressee tail (see
    # test_addressee_tail.py); overstating this capped, process-local ring as that record
    # both duplicated its job and read as an invite to continue other conversations. So the
    # header must read as peripheral and must NOT claim to be the addressee record.
    env = _seeded_pulse().render_envelope("C1")
    header = env.splitlines()[0]
    assert "peripheral" in header.lower()                 # framed as reference context, not the record
    assert "other conversations" in header.lower()        # explicitly someone else's exchanges
    assert "who has been talking to whom" not in header.lower()  # NOT the addressee record


# --------------------------------------------------------- participation stats

def test_participation_stat_math():
    p = ChannelPulse()
    now = 10_000.0
    p.record_bot_reply("C1", "1.0", unprompted=True, now=now - 3599)   # inside window
    p.record_bot_reply("C1", "2.0", unprompted=True, now=now - 3601)   # outside
    p.record_bot_reply("C1", "3.0", unprompted=False, now=now)         # prompted: not counted
    p.record_bot_reply("D1", "4.0", unprompted=True, now=now)          # DM: not counted
    assert p.unprompted_count_last_hour("C1", now=now) == 1
    assert p.unprompted_count_last_hour("C2", now=now) == 0


# ------------------------------------------------- idempotence / dedup (F5)

def test_record_idempotent_by_ts():
    # Dual delivery (app_mention + message) / retries: same ts recorded once.
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("1.0", text="hi"))
    p.record("C1", **_entry("1.0", text="hi"))
    assert len(p._buffers["C1"]) == 1


def test_seen_ts_outer_map_bounded(monkeypatch):
    # The OUTER channel map must be a bounded whole-channel LRU (no unbounded growth).
    monkeypatch.setattr(config, "pulse_thread_tail_channels_max", 3)
    p = ChannelPulse(size=5)
    for i in range(10):
        p.record(f"C{i}", **_entry("1.0", text="hi"))
    assert len(p._seen_ts) <= 3
    assert "C9" in p._seen_ts          # newest channels retained
    assert "C0" not in p._seen_ts      # oldest evicted


def test_seen_ts_resurrection_guard_uses_live_ring():
    # A ts that aged out of the dedup window but is STILL in the live ring must be
    # treated as already-recorded — a delayed retry can't resurrect it (F5).
    p = ChannelPulse(size=30)
    p.record("C1", **_entry("100.1", text="original"))
    assert len(p._buffers["C1"]) == 1
    # Simulate the ts falling out of the bounded _seen_ts window while it lives on in
    # the buffer (and the thread-tail ring).
    p._seen_ts["C1"].clear()
    p.record("C1", **_entry("100.1", text="original"))   # delayed retry
    assert len(p._buffers["C1"]) == 1                     # NOT re-appended
    # A genuinely new ts (not in any ring) still records normally.
    p.record("C1", **_entry("200.2", text="new"))
    assert len(p._buffers["C1"]) == 2


# ------------------------------------------------------------------ wiring

def _mixin_host(pulse):
    """Minimal object binding the real _feed_channel_pulse."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class Host(SlackMessageEventsMixin):
        def __init__(self):
            self.channel_pulse = pulse
            self.user_cache = {"U7": {"real_name": "Peter"}}
            self.bot_user_id = "UBOT"

        def is_own_message(self, e):
            return e.get("user") == "UBOT"

        def classify_sender(self, e):
            return "self" if e.get("user") == "UBOT" else "human"

        def log_debug(self, *a, **k): pass

    return Host()


@pytest.mark.asyncio
async def test_feed_skips_own_records_others():
    # F5 fix (a): the event feed skips OUR OWN posts (echoed placeholders/footers/edits
    # are chrome) — the bot's own final reply is recorded cleanly at the messaging layer.
    # Other senders (incl. bot_message subtype) are recorded.
    p = ChannelPulse(size=5)
    host = _mixin_host(p)
    await host._feed_channel_pulse({"channel": "C1", "ts": "1.0", "user": "UBOT",
                                    "text": "my own post"})
    await host._feed_channel_pulse({"channel": "C1", "ts": "2.0", "user": "U7",
                                    "text": "human post"})
    env = p.render_envelope("C1")
    assert "human post" in env and "Peter" in env
    assert "my own post" not in env
    # ...but the bot's own reply DOES enter the ring via the messaging-layer recorder.
    p.record_own_reply("C1", thread_ts=None, ts="3.0", text="my own post")
    assert "my own post" in p.render_envelope("C1")


@pytest.mark.asyncio
async def test_feed_noop_when_pulse_disabled():
    host = _mixin_host(None)
    # must not raise
    await host._feed_channel_pulse({"channel": "C1", "ts": "1.0", "text": "x"})


# -------------------------------------------------- suffix placement (not prefix)

def _bind_utils():
    from message_processor.utilities import MessageUtilitiesMixin

    class P(MessageUtilitiesMixin):
        def log_debug(self, *a, **k): pass

    return P.__new__(P)


def test_envelope_rides_suffix_not_system_prompt():
    proc = _bind_utils()
    client = MagicMock()
    client.channel_pulse = _seeded_pulse()
    # F51 role authority: the envelope no longer rides INSIDE the developer suffix (it carries
    # ambient, attacker-influenceable content). It is built separately and injected as a USER
    # message by the text handler. The developer suffix keeps the trusted time context only.
    suffix = proc._build_suffix_context(client, "C1", None)
    assert "[Recent channel activity" not in suffix
    assert "[Current date and time:" in suffix
    # The envelope IS produced for a channel (still volatile suffix content, just user-scoped)…
    envelope = proc._build_pulse_envelope(client, "C1", None)
    assert envelope and "[Recent channel activity" in envelope
    # …and absent for a DM.
    assert proc._build_pulse_envelope(client, "D1", None) is None


# ----------------------------------------------- F29 channel-people suffix line

def _people_client(pulse, num_members=12):
    client = SimpleNamespace(
        channel_pulse=pulse,
        get_cached_channel_context=lambda cid: {"num_members": num_members},
    )
    return client


def test_channel_people_line_in_suffix():
    proc = _bind_utils()
    client = _people_client(_seeded_pulse(), num_members=12)
    line = proc._build_channel_people_line(client, "C1")
    assert line is not None
    assert "Channel people: ~12 members; recently active:" in line
    assert "Cara" in line and "Bob" in line   # active speakers surfaced
    # And it rides the assembled suffix.
    suffix = proc._build_suffix_context(client, "C1", None)
    assert "[Channel people:" in suffix


def test_channel_people_line_absent_pieces_skip_cleanly():
    proc = _bind_utils()
    # No count available (peek returns None) but speakers present → speakers-only line.
    client = SimpleNamespace(channel_pulse=_seeded_pulse(),
                             get_cached_channel_context=lambda cid: None)
    line = proc._build_channel_people_line(client, "C1")
    assert line is not None and "recently active:" in line and "members" not in line


def test_channel_people_line_none_when_nothing_known():
    proc = _bind_utils()
    empty = ChannelPulse(size=5)  # no messages → no speakers
    client = SimpleNamespace(channel_pulse=empty,
                             get_cached_channel_context=lambda cid: None)
    assert proc._build_channel_people_line(client, "C1") is None


def test_channel_people_line_none_for_dm():
    proc = _bind_utils()
    client = _people_client(_seeded_pulse())
    assert proc._build_channel_people_line(client, "D1") is None
    # And the envelope never appears in the system prompt builder's output — the
    # system prompt has no pulse/client access at all; assert the source contract.
    import inspect
    from message_processor import utilities as u
    src = inspect.getsource(u.MessageUtilitiesMixin._get_system_prompt)
    assert "pulse" not in src and "Recent channel activity" not in src


def test_envelope_disabled_when_pulse_none():
    proc = _bind_utils()
    client = MagicMock(spec=[])  # no channel_pulse attribute at all
    assert proc._build_pulse_envelope(client, "C1", None) is None


def test_envelope_respects_env_cap(monkeypatch):
    proc = _bind_utils()
    client = MagicMock()
    client.channel_pulse = _seeded_pulse()
    monkeypatch.setattr(config, "channel_pulse_envelope_max", 0)
    assert proc._build_pulse_envelope(client, "C1", None) is None


# ------------------------------------------------- F14b/F25 attachment note

def test_attachment_note_helper():
    from slack_client.channel_pulse import _attachment_note
    assert _attachment_note(None) == ""
    assert _attachment_note([]) == ""
    assert _attachment_note([{"mimetype": "image/png"}]) == "[+1 image]"
    assert _attachment_note(
        [{"mimetype": "application/pdf"}, {"mimetype": "text/csv"}]) == "[+2 files]"
    assert _attachment_note(
        [{"mimetype": "image/png"}, {"mimetype": "application/pdf"}]
    ) == "[+1 image, +1 file]"


def test_attachment_note_includes_filenames():
    # F25: filenames are what read_document needs to reach a cross-thread file —
    # a count-only note left the model unable to name a document from another thread.
    from slack_client.channel_pulse import _attachment_note
    assert _attachment_note(
        [{"mimetype": "image/png", "name": "food.png"}]) == "[+1 image: food.png]"
    assert _attachment_note(
        [{"mimetype": "application/pdf", "name": "a.pdf"},
         {"mimetype": "text/csv", "name": "b.csv"}]) == "[+2 files: a.pdf, b.csv]"


def test_attachment_note_sanitizes_and_caps_filenames():
    from slack_client.channel_pulse import _attachment_note
    # Hostile name: brackets/backticks/quotes/newlines stripped so the note can't
    # break the bracketed grammar or spoof another context line.
    assert _attachment_note(
        [{"mimetype": "application/pdf", "name": "x[y]`z\"w.pdf\n"}]) == "[+1 file: xyzw.pdf]"
    # Display truncation.
    long = _attachment_note([{"mimetype": "text/csv", "name": "n" * 200 + ".csv"}])
    assert len(long) < 80
    # At most 3 names, then +N more.
    note = _attachment_note(
        [{"mimetype": "text/csv", "name": f"f{i}.csv"} for i in range(5)])
    assert note == "[+5 files: f0.csv, f1.csv, f2.csv, +2 more]"


def test_record_appends_attachment_note_to_envelope_and_tail():
    # F14b/F25: a message carrying files surfaces a bracketed note (with filename) that
    # both the envelope line and the thread-tail entry inherit (folded in at record level).
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="what do we think?"),
             files=[{"mimetype": "image/png", "name": "food.png"}])
    env = p.render_envelope("C1")
    assert "what do we think? [+1 image: food.png]" in env
    tail = p.render_thread_tail("C1", "100.0", before_ts="200.0")
    assert "[+1 image: food.png]" in tail


def test_record_without_files_leaves_text_untouched():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="just talking"))
    assert "[+" not in p.render_envelope("C1")


# ------------------------------------------- recent_taggable_speakers (A1)
# Unlike _entry (user_id pinned to "U1"), these tests need distinct ids AND fresh ts:
# recent_taggable_speakers returns {user_id, name} and applies a 24h age horizon
# (now - float(ts)), so ancient ts like "1.0" would all age out.

def _trec(p, ts, uid, *, name=None, sender="human", is_bot=False, channel="C1"):
    p.record(channel, ts=ts, thread_ts=None, user_id=uid, display_name=name,
             sender_type=sender, text="hi", is_bot=is_bot)


def _fresh(delta=10.0):
    """A ts `delta` seconds in the past — inside the default 24h horizon."""
    return f"{time.time() - delta:.6f}"


def test_taggable_newest_first_dedup_id_and_name():
    p = ChannelPulse(size=20)
    _trec(p, _fresh(50), "UA", name="Alice")
    _trec(p, _fresh(40), "UB", name="Bob")
    _trec(p, _fresh(10), "UA", name="Alice New")   # newest instance per uid wins
    out = p.recent_taggable_speakers("C1", bot_user_id="UBOT")
    assert out == [{"user_id": "UA", "name": "Alice New"},
                   {"user_id": "UB", "name": "Bob"}]


def test_taggable_excludes_bot_self_sentinels_keeps_other_bots():
    p = ChannelPulse(size=20)
    _trec(p, _fresh(60), "UBOT", name="ChatGPT")             # the bot itself (by id) — excluded
    _trec(p, _fresh(55), "UME", name="Me", sender="self")    # bot's own turn (self) — excluded
    _trec(p, _fresh(50), None, name=None)                    # no real user_id — excluded
    _trec(p, _fresh(45), "bot", name="Botish")               # sentinel id — excluded
    _trec(p, _fresh(40), "unknown", name="Ghost")            # sentinel id — excluded
    _trec(p, _fresh(35), "UCLAUDE", name="Claude", sender="other_bot", is_bot=True)  # peer KEPT
    _trec(p, _fresh(30), "UH", name="Human")                 # ordinary human KEPT
    out = p.recent_taggable_speakers("C1", bot_user_id="UBOT")
    assert out == [{"user_id": "UH", "name": "Human"},          # newest-first
                   {"user_id": "UCLAUDE", "name": "Claude"}]


def test_taggable_dm_and_unknown_channel_empty():
    p = ChannelPulse(size=20)
    assert p.recent_taggable_speakers("D123") == []    # DM rejected
    assert p.recent_taggable_speakers("C-none") == []  # unknown channel → empty ring
    _trec(p, _fresh(10), "UA", name="Alice", channel="D123")  # DMs are never recorded anyway
    assert p.recent_taggable_speakers("D123") == []


def test_taggable_age_horizon_drops_stale_and_unparseable():
    p = ChannelPulse(size=20)
    _trec(p, _fresh(10), "UFRESH", name="Fresh")
    _trec(p, f"{time.time() - 200000:.6f}", "USTALE", name="Stale")  # > 24h old
    _trec(p, "not-a-ts", "UBAD", name="Bad")                          # unparseable → skipped
    assert p.recent_taggable_speakers("C1") == [{"user_id": "UFRESH", "name": "Fresh"}]
    # A wider horizon lets the aged one back in; the unparseable ts still can't prove freshness.
    wide = p.recent_taggable_speakers("C1", max_age_seconds=300000)
    assert {d["user_id"] for d in wide} == {"UFRESH", "USTALE"}


def test_taggable_respects_limit():
    p = ChannelPulse(size=30)
    for i in range(6):
        _trec(p, _fresh(60 - i), f"U{i}", name=f"User{i}")   # U0 oldest … U5 newest
    out = p.recent_taggable_speakers("C1", limit=2)
    assert [d["user_id"] for d in out] == ["U5", "U4"]


def test_taggable_name_sanitized_and_capped():
    p = ChannelPulse(size=20)
    _trec(p, _fresh(20), "UA", name="Claude [bot]")   # brackets folded (no frame spoofing)
    _trec(p, _fresh(10), "UB", name="Z" * 200)        # length-capped
    by_id = {d["user_id"]: d["name"] for d in p.recent_taggable_speakers("C1")}
    assert by_id["UA"] == "Claude (bot)"
    assert len(by_id["UB"]) == 80


def test_taggable_never_raises_on_bad_state():
    # A malformed buffer must degrade to [] rather than propagate (never raises contract).
    p = ChannelPulse(size=5)
    p._buffers["C1"] = "not-a-deque"
    assert p.recent_taggable_speakers("C1") == []
