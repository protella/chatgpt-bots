"""Unit tests for the canonical Slack normalizer (spec §2)."""
import pytest

from slack_client.normalizer import (KIND_DELETE, KIND_EDIT, KIND_MESSAGE, KIND_TOMBSTONE,
                                     MUTATION_SUBTYPES, ORIGIN_LIVE, SKIP_SUBTYPES,
                                     MalformedEventError, TimestampError, in_window, is_own_event,
                                     normalize_slack_event, normalize_slack_message, parse_ts,
                                     render_mentions, sanitize_field, sanitize_name, ts_key,
                                     ts_le, ts_lt, ts_max)

TEAM = "T1"
CH = "C1"
FLOOR = "1752600100.000100"
HIGH = "1752600200.000200"


class _Client:
    """Mirrors SlackUtilitiesMixin identity semantics without the Slack plumbing."""

    def __init__(self, team=TEAM):
        self.self_team_id = team
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = "A123"

    def is_own_message(self, msg):
        if not isinstance(msg, dict):
            return False
        return bool(msg.get("bot_id") == self.bot_id
                    or msg.get("user") == self.bot_user_id
                    or msg.get("app_id") == self.app_id
                    or msg.get("api_app_id") == self.app_id)

    def classify_sender(self, msg):
        if not isinstance(msg, dict):
            return "human"
        if self.is_own_message(msg):
            return "self"
        if msg.get("bot_id") or msg.get("app_id") or msg.get("api_app_id"):
            return "other_bot"
        return "human"


class _OwnOnlyClient:
    """No classify_sender attribute at all."""

    self_team_id = TEAM
    bot_user_id = "UBOT"
    bot_id = "BBOT"

    def is_own_message(self, msg):
        return isinstance(msg, dict) and msg.get("bot_id") == self.bot_id


@pytest.fixture
def client():
    return _Client()


def _payload(ts="1752600150.000100", **extra):
    p = {"type": "message", "channel": CH, "user": "U1", "ts": ts, "text": "hello"}
    p.update(extra)
    return p


def _event(ts="1752600150.000100", **extra):
    e = {"type": "message", "channel": CH, "channel_type": "channel", "user": "U1",
         "ts": ts, "text": "hello", "event_ts": ts}
    e.update(extra)
    return e


def _changed(message, event_ts="1752600900.000900", **extra):
    e = {"type": "message", "subtype": "message_changed", "channel": CH,
         "channel_type": "channel", "message": message, "event_ts": event_ts}
    e.update(extra)
    return e


def _deleted(previous, deleted_ts, event_ts="1752600900.000900", **extra):
    e = {"type": "message", "subtype": "message_deleted", "channel": CH,
         "channel_type": "channel", "previous_message": previous,
         "deleted_ts": deleted_ts, "event_ts": event_ts}
    e.update(extra)
    return e


# ------------------------------------------------------------------ parse_ts


def test_parse_ts_returns_integer_pair():
    assert parse_ts("1752600000.000100") == (1752600000, 100)
    assert all(isinstance(part, int) for part in parse_ts("1752600000.000100"))


def test_parse_ts_pads_short_fraction():
    assert parse_ts("1.5") == (1, 500000)
    assert parse_ts("1.05") == (1, 50000)
    assert parse_ts("1.0000001") == (1, 0)


def test_parse_ts_without_fraction():
    assert parse_ts("1752600000") == (1752600000, 0)
    assert parse_ts(1752600000) == (1752600000, 0)
    assert parse_ts("  1752600000.000100  ") == (1752600000, 100)


@pytest.mark.parametrize("raw", ["", None, "abc", "1.2.3", "-1", "1.2a", " ", "1e9"])
def test_parse_ts_rejects_garbage(raw):
    with pytest.raises(TimestampError):
        parse_ts(raw)


def test_ts_key_is_parse_ts():
    assert ts_key("1752600000.5") == parse_ts("1752600000.5")


def test_unequal_width_fractions_compare_numerically():
    low, high = "1752600000.10", "1752600000.5"
    assert parse_ts(low) == (1752600000, 100000)
    assert parse_ts(high) == (1752600000, 500000)
    assert ts_lt(low, high)
    assert not ts_lt(high, low)
    assert ts_max(low, high) == high
    # An unpadded fraction-integer compare inverts this pair.
    assert int(low.partition(".")[2]) > int(high.partition(".")[2])


def test_unequal_width_seconds_beat_lexicographic_order():
    small, big = "9999999999.000100", "10000000000.000100"
    assert ts_lt(small, big)
    assert ts_max(small, big) == big
    assert small > big


def test_adjacent_microseconds_stay_distinct():
    a, b = "1752600000.000001", "1752600000.000002"
    assert parse_ts(a) != parse_ts(b)
    assert parse_ts(a) == (1752600000, 1)
    assert parse_ts(b) == (1752600000, 2)
    assert ts_lt(a, b)
    assert not ts_le(b, a)
    assert ts_max(a, b) == b


# ------------------------------------------------------- comparators / window


def test_ts_lt_and_ts_le():
    assert ts_lt("1.000001", "1.000002")
    assert not ts_lt("1.000002", "1.000001")
    assert not ts_lt("1.000001", "1.000001")
    assert ts_le("1.000001", "1.000001")
    assert ts_le("1.000001", "1.000002")
    assert not ts_le("1.000002", "1.000001")


def test_ts_max_handles_none():
    assert ts_max(None, None) is None
    assert ts_max(None, "1.5") == "1.5"
    assert ts_max("1.5", None) == "1.5"
    assert ts_max("1.5", "1.5") == "1.5"
    assert isinstance(ts_max(None, 1752600000), str)


def test_in_window_floor_inclusive_at_the_exact_floor():
    assert in_window(FLOOR, FLOOR, True, HIGH)
    assert not in_window(FLOOR, FLOOR, False, HIGH)


def test_in_window_high_is_inside():
    assert in_window(HIGH, FLOOR, True, HIGH)
    assert in_window(HIGH, FLOOR, False, HIGH)


def test_in_window_rejects_outside():
    assert not in_window("1752600099.999999", FLOOR, True, HIGH)
    assert not in_window("1752600200.000201", FLOOR, True, HIGH)
    assert in_window("1752600150.000000", FLOOR, False, HIGH)


# ---------------------------------------------------------------- sanitizers


def test_sanitize_field_folds_line_endings():
    assert sanitize_field("a\r\nb\rc") == "a\nb\nc"


def test_sanitize_field_strips_control_chars_and_keeps_newlines():
    assert sanitize_field("a\x00b\x07c\x1fd\x7fe") == "abcde"
    assert sanitize_field("a\nb") == "a\nb"
    assert sanitize_field("a\tb") == "a\tb"
    assert sanitize_field(None) == ""
    assert sanitize_field(0) == ""


def test_sanitize_name_replaces_newlines_with_space():
    assert sanitize_name("a\nb") == "a b"
    assert sanitize_name("a\r\nb") == "a b"


def test_sanitize_name_drops_brackets_and_quotes():
    assert sanitize_name('[Header] "quoted" name') == "Header quoted name"
    assert sanitize_name('  x\x00[y]"z"  ') == "xyz"


# --------------------------------------------------- normalize_slack_message


@pytest.mark.parametrize("subtype", sorted(SKIP_SUBTYPES))
def test_skip_subtypes_yield_none(client, subtype):
    assert normalize_slack_message(client, _payload(subtype=subtype)) is None


@pytest.mark.parametrize("subtype", sorted(MUTATION_SUBTYPES))
def test_mutation_subtypes_yield_none(client, subtype):
    assert normalize_slack_message(client, _payload(subtype=subtype)) is None


def test_tombstone_subtype_is_not_skipped(client):
    rec = normalize_slack_message(
        client, _payload(subtype="tombstone", text="This message was deleted."))
    assert rec is not None
    assert rec.subtype == "tombstone"
    assert rec.is_tombstone is True


def test_tombstone_detected_by_text_alone(client):
    rec = normalize_slack_message(client, _payload(text="This message was deleted."))
    assert rec.is_tombstone is True
    assert normalize_slack_message(client, _payload(text="hello")).is_tombstone is False


def test_thread_broadcast_sets_is_broadcast(client):
    rec = normalize_slack_message(
        client, _payload(subtype="thread_broadcast", thread_ts="1752600100.000100"))
    assert rec.is_broadcast is True
    assert rec.subtype == "thread_broadcast"
    assert rec.is_reply is True
    assert rec.root_ts == "1752600100.000100"


def test_non_dict_and_missing_fields_yield_none(client):
    assert normalize_slack_message(client, None) is None
    assert normalize_slack_message(client, "text") is None
    assert normalize_slack_message(client, {"ts": "1.000001", "text": "x"}) is None


@pytest.mark.parametrize("payload", [
    {"channel": CH, "text": "x"},                 # no ts at all
    {"channel": CH, "text": "x", "ts": ""},
    {"channel": CH, "text": "x", "ts": None},
    {"channel": CH, "text": "x", "ts": "yesterday"},
])
def test_a_message_with_no_usable_ts_raises_rather_than_vanishing(client, payload):
    """An absent ts is not a lesser problem than a malformed one. Either way the message cannot
    be placed in the window, and returning None would drop it from a stream whose whole claim is
    that it shows the room — so the turn path fails closed on it instead."""
    with pytest.raises(TimestampError):
        normalize_slack_message(client, payload)


def test_channel_and_team_defaults(client):
    rec = normalize_slack_message(client, {"ts": "1.000001", "text": "x"}, channel_id="C9")
    assert (rec.channel_id, rec.team_id, rec.origin) == ("C9", TEAM, "history")
    rec = normalize_slack_message(client, _payload(), origin=ORIGIN_LIVE, team_id="T9")
    assert (rec.team_id, rec.origin) == ("T9", ORIGIN_LIVE)


def test_files_map_to_file_refs(client):
    payload = _payload(files=[
        {"id": "F1", "name": "shot.png", "mimetype": "image/png", "size": 12,
         "url_private": "https://x/1"},
        {"id": "F2", "title": "report", "mimetype": "application/pdf", "size": "big"},
        {"mimetype": "image/gif"},
        "not-a-dict",
    ])
    rec = normalize_slack_message(client, payload)
    assert len(rec.files) == 3
    first, second, third = rec.files
    assert (first.id, first.name, first.kind, first.size) == ("F1", "shot.png", "image", 12)
    assert first.url_private == "https://x/1"
    assert (second.id, second.name, second.kind) == ("F2", "report", "file")
    assert second.size is None and second.url_private is None
    assert (third.id, third.name, third.kind) == (None, "file", "image")


def test_reactions_count_falls_back_to_users(client):
    rec = normalize_slack_message(client, _payload(reactions=[
        {"name": "eyes", "users": ["U1", "U2"]},
        {"name": "tada", "count": 7, "users": ["U1"]},
        {"name": "shrug", "count": "many"},
        {"users": ["U1"]},
        "not-a-dict",
    ]))
    assert [(r.name, r.count) for r in rec.reactions] == [
        ("eyes", 2), ("tada", 7), ("shrug", 0)]


def test_reaction_mine_is_membership_of_a_list_of_users_only(client):
    """Slack puts the authenticated user in `users` whenever that user reacted, even when other
    reactor ids are truncated — so a bigger `count` than `users` still reads as ours, and our
    absence is a real non-reaction. Nothing but a list/tuple is tested for membership."""
    rec = normalize_slack_message(client, _payload(reactions=[
        {"name": "this", "count": 2, "users": ["U1", "UBOT"]},
        {"name": "+1", "count": 2, "users": ["U1", "U2"]},
        {"name": "eyes", "count": 40, "users": ["UBOT"]},
        {"name": "wave", "count": 1},
        {"name": "tada", "count": 1, "users": None},
        {"name": "boom", "count": 1, "users": "UBOTHERED"},
        {"name": "zap", "count": 1, "users": {"UBOT": 1}},
    ]))
    assert [(r.name, r.mine) for r in rec.reactions] == [
        ("this", True), ("+1", False), ("eyes", True), ("wave", False),
        ("tada", False), ("boom", False), ("zap", False)]


def test_reaction_mine_fails_closed_without_an_identity():
    """A caller that never wired `bot_user_id` gets `mine=False` everywhere rather than an
    exception: the normalizer must not turn a missing identity into a failed read."""
    anonymous = _Client()
    anonymous.bot_user_id = None
    rec = normalize_slack_message(anonymous, _payload(reactions=[
        {"name": "this", "count": 1, "users": ["UBOT"]},
        {"name": "+1", "count": 1, "users": ["U1"]},
    ]))
    assert [r.mine for r in rec.reactions] == [False, False]


def test_edited_ts_reply_count_and_latest_reply_carried(client):
    rec = normalize_slack_message(client, _payload(
        edited={"user": "U1", "ts": "1752600180.000180"},
        reply_count=3, latest_reply="1752600190.000190"))
    assert rec.edited_ts == "1752600180.000180"
    assert rec.reply_count == 3
    assert rec.latest_reply == "1752600190.000190"

    bare = normalize_slack_message(client, _payload(edited="nope", reply_count="3"))
    assert bare.edited_ts is None and bare.reply_count is None and bare.latest_reply is None


def test_mention_ids_extracted_raw_and_text_left_uncleaned(client):
    rec = normalize_slack_message(
        client, _payload(text="hey <@U1|ann> and <@UBOT> ping"))
    assert rec.mention_ids == ("U1", "UBOT")
    assert rec.text == "hey <@U1|ann> and <@UBOT> ping"


def test_bot_name_and_sender_id(client):
    rec = normalize_slack_message(client, _payload(
        user=None, bot_id="B999", username='[Jira]\nbot"'))
    assert rec.sender_type == "other_bot"
    assert rec.sender_id == "B999"
    assert rec.raw_bot_name == "Jira bot"

    profiled = normalize_slack_message(client, _payload(
        user=None, bot_id="B999", bot_profile={"name": "Deploybot"}))
    assert profiled.raw_bot_name == "Deploybot"
    assert normalize_slack_message(client, _payload()).raw_bot_name is None


@pytest.mark.parametrize("bad", ["abc", "1.2.3", "-1"])
def test_malformed_ts_raises(client, bad):
    with pytest.raises(TimestampError):
        normalize_slack_message(client, _payload(ts=bad))


def test_malformed_nested_timestamps_raise(client):
    with pytest.raises(TimestampError):
        normalize_slack_message(client, _payload(edited={"ts": "nope"}))
    with pytest.raises(TimestampError):
        normalize_slack_message(client, _payload(latest_reply="nope"))


def test_own_message_is_represented_never_none(client):
    for own in ({"bot_id": "BBOT"}, {"user": "UBOT"}, {"app_id": "A123"},
                {"api_app_id": "A123"}):
        rec = normalize_slack_message(client, _payload(**{"user": None, **own}))
        assert rec is not None
        assert rec.sender_type == "self"


def test_is_own_message_alone_yields_self():
    rec = normalize_slack_message(_OwnOnlyClient(), _payload(user=None, bot_id="BBOT"))
    assert rec.sender_type == "self"
    assert normalize_slack_message(_OwnOnlyClient(), _payload()).sender_type == "human"


# ------------------------------------------------- the user-less peer bot (agent mode)


class _CachingClient(_Client):
    """A client whose bots.info cache has already answered for some bot object ids."""

    def __init__(self, cache=None):
        super().__init__()
        self._cache = cache or {}

    def bot_user_id_for(self, bot_id):
        return self._cache.get(bot_id)


def _agent_mode(**extra):
    """A peer app posting in agent mode: a username override, bot_id + app_id, NO `user`."""
    fields = {"user": None, "subtype": "bot_message", "username": "Bot",
              "bot_id": "B999", "app_id": "A999"}
    fields.update(extra)
    return _payload(**fields)


def test_agent_mode_bot_is_named_by_its_user_id_when_the_cache_knows_it():
    rec = normalize_slack_message(_CachingClient({"B999": "UPEER"}), _agent_mode())
    assert rec.sender_id == "UPEER"
    assert rec.sender_type == "other_bot"


def test_agent_mode_bot_falls_back_to_the_bot_id_when_the_cache_does_not(client):
    assert normalize_slack_message(client, _agent_mode()).sender_id == "B999"
    unknown = _CachingClient({"BOTHER": "UOTHER"})
    assert normalize_slack_message(unknown, _agent_mode()).sender_id == "B999"


def test_a_named_user_always_wins_over_the_cache():
    rec = normalize_slack_message(_CachingClient({"B999": "UPEER"}),
                                  _agent_mode(user="U1"))
    assert rec.sender_id == "U1"


# -------------------------------------------------------- supplementary fold


def _fielded(text="deploy update", **extra):
    return _payload(text=text, attachments=[
        {"fields": [{"title": "Branch", "value": "main"},
                    {"title": "Status", "value": "failed"}]}], **extra)


def test_supplementary_fields_folded_for_other_sender(client):
    rec = normalize_slack_message(client, _fielded(user=None, bot_id="B999"))
    assert rec.sender_type == "other_bot"
    assert rec.text.startswith("deploy update")
    assert "Branch: main" in rec.text
    assert "Status: failed" in rec.text


def test_supplementary_fields_not_folded_for_self(client):
    rec = normalize_slack_message(client, _fielded(user=None, bot_id="BBOT"))
    assert rec.sender_type == "self"
    assert rec.text == "deploy update"


def test_supplementary_becomes_the_text_when_primary_is_empty(client):
    rec = normalize_slack_message(client, _fielded(user=None, bot_id="B999", text=""))
    assert "Branch: main" in rec.text


# ----------------------------------------------------- normalize_slack_event


def test_plain_message_event(client):
    ev = normalize_slack_event(client, _event())
    assert ev.kind == KIND_MESSAGE
    assert (ev.team_id, ev.channel_id) == (TEAM, CH)
    assert ev.subject_ts == "1752600150.000100"
    assert ev.activity_ts is None
    assert ev.deleted_ts is None
    assert ev.owner_probe_ts is None
    assert ev.root_if_indexed is False
    assert ev.message is not None
    assert ev.message.origin == ORIGIN_LIVE


def test_message_changed_is_an_edit_with_outer_event_ts(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "user": "U1", "ts": "1752600150.000100", "text": "new",
        "edited": {"user": "U1", "ts": "1752600160.000160"}}))
    assert ev.kind == KIND_EDIT
    assert ev.activity_ts == "1752600900.000900"
    assert ev.subject_ts == "1752600150.000100"
    assert ev.message.text == "new"
    assert ev.owner_probe_ts is None


def test_edit_activity_ts_falls_back_to_nested_edited_ts(client):
    event = _changed({"type": "message", "user": "U1", "ts": "1752600150.000100",
                      "text": "new", "edited": {"ts": "1752600160.000160"}})
    event.pop("event_ts")
    assert normalize_slack_event(client, event).activity_ts == "1752600160.000160"


def test_the_activity_ts_has_one_definition_shared_with_admission(client):
    """r3-5. `registration._admit` used to compute the watermark's ts itself, with no nested
    fallback, so an edit carrying only `edited.ts` was admitted as unobservable — a permanent,
    unrepairable failure that took the channel out of service — while the normalizer placed the
    same edit without trouble. One function now answers the question for both."""
    from slack_client.normalizer import mutation_activity_ts

    for event in (
        _changed({"type": "message", "user": "U1", "ts": "1752600150.000100", "text": "new",
                  "edited": {"ts": "1752600160.000160"}}),
        _deleted({"type": "message", "user": "U1", "ts": "1752600150.000100", "text": "gone"},
                 deleted_ts="1752600150.000100"),
    ):
        assert mutation_activity_ts(event) == normalize_slack_event(client, event).activity_ts

    edit = _changed({"type": "message", "user": "U1", "ts": "1752600150.000100", "text": "new",
                     "edited": {"ts": "1752600160.000160"}})
    edit.pop("event_ts")
    assert mutation_activity_ts(edit) == "1752600160.000160"
    # A plain message is not a mutation: its activity IS its own ts, recorded elsewhere.
    assert mutation_activity_ts(_event()) is None


def test_message_changed_carrying_a_tombstone(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "tombstone", "ts": "1752600150.000100",
        "text": "This message was deleted.", "hidden": True}))
    assert ev.kind == KIND_TOMBSTONE
    assert ev.activity_ts == "1752600900.000900"
    assert ev.root_if_indexed is False
    assert ev.message.is_tombstone is True


def test_message_deleted(client):
    ev = normalize_slack_event(client, _deleted(
        {"type": "message", "user": "U1", "ts": "1752600150.000100", "text": "gone"},
        deleted_ts="1752600150.000100"))
    assert ev.kind == KIND_DELETE
    assert ev.deleted_ts == "1752600150.000100"
    assert ev.activity_ts == "1752600900.000900"
    assert ev.subject_ts == "1752600150.000100"
    assert ev.message.text == "gone"


def test_mutation_without_a_nested_subject_fails_rather_than_skipping(client):
    """r2-2: this used to return None, which the index feed read as "nothing to record" and
    reported as a SUCCESSFUL observation. The outer event_ts has already advanced H by then, so a
    turn could answer without proving the mutation was ever indexed."""
    with pytest.raises(MalformedEventError):
        normalize_slack_event(client, _changed(None))
    with pytest.raises(MalformedEventError):
        normalize_slack_event(client, _deleted(None, "1.000001"))


def test_a_mutation_that_names_no_subject_ts_fails(client):
    """A mutation identifies its subject by ts and by nothing else."""
    with pytest.raises(MalformedEventError):
        normalize_slack_event(client, _changed({"type": "message", "user": "U1", "text": "hi"}))
    with pytest.raises(MalformedEventError):
        normalize_slack_event(client, _deleted({"type": "message", "user": "U1"}, "1.000001"))


def test_an_unknown_kind_with_no_ts_is_still_only_skipped(client):
    """The other half of the contract: a non-mutation envelope records nothing and must NOT take a
    channel out of service. Its own admission step already failed the observation."""
    assert normalize_slack_event(client, {
        "type": "reaction_added", "user": "U1", "channel": CH, "channel_type": "channel",
        "item": {"type": "message", "channel": CH, "ts": "1752600150.000100"}}) is None


def test_item_only_shape_resolves_the_channel(client):
    ev = normalize_slack_event(client, {
        "type": "reaction_added", "user": "U1", "ts": "1752600150.000100",
        "item": {"type": "message", "channel": CH, "ts": "1752600150.000100"},
        "event_ts": "1752600900.000900"})
    assert ev is not None
    assert ev.channel_id == CH
    assert ev.kind == KIND_MESSAGE
    assert ev.subject_ts == "1752600150.000100"


def test_item_only_shape_without_a_top_level_ts_is_skipped(client):
    assert normalize_slack_event(client, {
        "type": "reaction_added", "user": "U1",
        "item": {"type": "message", "channel": CH, "ts": "1752600150.000100"},
        "event_ts": "1752600900.000900"}) is None


def test_dm_events_are_skipped(client):
    assert normalize_slack_event(client, _event(channel_type="im")) is None
    assert normalize_slack_event(
        client, _event(channel="D123", channel_type=None)) is None


def test_no_channel_or_bad_shape_is_skipped(client):
    assert normalize_slack_event(client, None) is None
    assert normalize_slack_event(client, "message") is None
    assert normalize_slack_event(client, {"type": "message", "ts": "1.000001"}) is None
    assert normalize_slack_event(
        client, _event(channel=123, channel_type="channel")) is None


def test_no_self_team_id_is_skipped():
    assert normalize_slack_event(_Client(team=None), _event()) is None
    assert normalize_slack_event(_Client(team=""), _event()) is None


def test_root_if_indexed_only_for_a_hintless_mutation(client):
    previous = {"type": "message", "user": "U1", "ts": "1752600150.000100", "text": "x"}
    assert normalize_slack_event(
        client, _deleted(dict(previous), "1752600150.000100")).root_if_indexed is True

    with_thread = dict(previous, thread_ts="1752600100.000100")
    assert normalize_slack_event(
        client, _deleted(with_thread, "1752600150.000100")).root_if_indexed is False

    with_latest = dict(previous, latest_reply="1752600160.000160")
    assert normalize_slack_event(
        client, _deleted(with_latest, "1752600150.000100")).root_if_indexed is False

    with_count = dict(previous, reply_count=2)
    assert normalize_slack_event(
        client, _deleted(with_count, "1752600150.000100")).root_if_indexed is False

    assert normalize_slack_event(client, _deleted(
        dict(previous, reply_count=0), "1752600150.000100")).root_if_indexed is True


def test_root_if_indexed_false_for_a_plain_message(client):
    assert normalize_slack_event(client, _event()).root_if_indexed is False


def test_root_if_indexed_false_for_a_tombstone(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "tombstone", "ts": "1752600150.000100",
        "text": "This message was deleted."}))
    assert ev.kind == KIND_TOMBSTONE
    assert ev.root_if_indexed is False


def test_owner_probe_ts_set_for_an_anonymous_delete(client):
    ev = normalize_slack_event(client, _deleted(
        {"type": "message", "ts": "1752600150.000100", "text": "gone"},
        deleted_ts="1752600150.000100"))
    assert ev.owner_probe_ts == "1752600150.000100"


@pytest.mark.parametrize("key", ["user", "bot_id", "app_id", "api_app_id"])
def test_owner_probe_ts_absent_when_the_delete_names_a_sender(client, key):
    ev = normalize_slack_event(client, _deleted(
        {"type": "message", "ts": "1752600150.000100", "text": "gone", key: "X1"},
        deleted_ts="1752600150.000100"))
    assert ev.owner_probe_ts is None


def test_owner_probe_ts_set_for_an_anonymous_tombstone(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "tombstone", "ts": "1752600150.000100",
        "text": "This message was deleted."}))
    assert ev.owner_probe_ts == "1752600150.000100"


def test_owner_probe_ts_never_set_for_an_edit(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "ts": "1752600150.000100", "text": "new",
        "edited": {"ts": "1752600160.000160"}}))
    assert ev.kind == KIND_EDIT
    assert ev.owner_probe_ts is None


def test_envelope_subtype_does_not_filter_the_subject(client):
    """The skip-set is WAIVED for a mutation's subject, not applied to a subtype-less copy of
    it: the subtype is where `tombstone` and `thread_broadcast` live, so removing it made the
    record's flags disagree with the event's own kind."""
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "channel_join", "user": "U1",
        "ts": "1752600150.000100", "text": "joined"}))
    assert ev.kind == KIND_EDIT
    assert ev.message is not None
    assert ev.message.subtype == "channel_join"

    gone = normalize_slack_event(client, _deleted({
        "type": "message", "subtype": "thread_broadcast", "user": "U1",
        "ts": "1752600150.000100", "text": "shout"}, "1752600150.000100"))
    assert gone.message is not None
    assert gone.message.is_broadcast is True


def test_a_subtype_only_tombstone_agrees_with_its_event_kind(client):
    """No sentinel text — the ONLY tombstone signal is the subtype, which is exactly what a
    subtype-stripping subject copy threw away."""
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "tombstone", "ts": "1752600150.000100",
        "hidden": True}))
    assert ev.kind == KIND_TOMBSTONE
    assert ev.message is not None
    assert ev.message.is_tombstone is True
    assert ev.message.subtype == "tombstone"


def test_an_edited_thread_broadcast_keeps_its_broadcast_flag(client):
    ev = normalize_slack_event(client, _changed({
        "type": "message", "subtype": "thread_broadcast", "user": "U1",
        "thread_ts": "1752600100.000100", "ts": "1752600150.000100", "text": "shout",
        "edited": {"ts": "1752600900.000900"}}))
    assert ev.kind == KIND_EDIT
    assert ev.message.is_broadcast is True
    assert ev.message.thread_root_ts == "1752600100.000100"


def test_a_mutation_subject_that_is_a_skip_subtype_still_yields_a_record(client):
    """The waiver is what keeps the index able to place a deleted join notice: it needs the
    record's thread hints even though the stream would never render one."""
    ev = normalize_slack_event(client, _deleted({
        "type": "message", "subtype": "channel_join", "user": "U1",
        "ts": "1752600150.000100", "text": "joined"}, "1752600150.000100"))
    assert ev.message is not None
    assert ev.message.subtype == "channel_join"


def test_plain_message_with_a_skip_subtype_yields_an_event_without_a_record(client):
    ev = normalize_slack_event(client, _event(subtype="channel_join"))
    assert ev is not None
    assert ev.kind == KIND_MESSAGE
    assert ev.message is None
    assert ev.channel_id == CH


def test_event_malformed_timestamps_raise(client):
    with pytest.raises(TimestampError):
        normalize_slack_event(client, _event(ts="nope"))
    with pytest.raises(TimestampError):
        normalize_slack_event(client, _changed(
            {"type": "message", "ts": "1752600150.000100", "text": "new",
             "edited": {"ts": "1752600160.000160"}}, event_ts="nope"))


# ------------------------------------------------------ self admission is the
# consumer's call, not the normalizer's


def test_self_events_are_normalized_and_filtered_by_is_own_event(client):
    ev = normalize_slack_event(client, _event(user="UBOT"))
    assert ev is not None
    assert ev.message is not None
    assert ev.message.sender_type == "self"
    assert is_own_event(client, ev) is True


def test_is_own_event_false_for_others_and_for_recordless_events(client):
    assert is_own_event(client, normalize_slack_event(client, _event())) is False
    assert is_own_event(
        client, normalize_slack_event(client, _event(subtype="channel_join"))) is False


# ------------------------------------------------- r3-7: falsey-PRESENT secondary timestamps


@pytest.mark.parametrize("key", ["thread_ts", "latest_reply"])
@pytest.mark.parametrize("value", ["", 0, False])
def test_a_secondary_ts_that_is_present_and_falsey_is_malformed(client, key, value):
    """Only truthy values were validated, so a field Slack HAD sent read as absent. An empty
    thread_ts made a reply look top-level — rendered outside its thread by a stream that claims to
    be the whole room — and nothing anywhere said so."""
    with pytest.raises(TimestampError):
        normalize_slack_message(client, _payload(**{key: value}))


@pytest.mark.parametrize("key", ["thread_ts", "latest_reply"])
def test_an_absent_or_null_secondary_ts_is_absent(client, key):
    """Absent is a normal shape and must stay one; `null` says the same thing as omitted."""
    assert normalize_slack_message(client, _payload()) is not None
    rec = normalize_slack_message(client, _payload(**{key: None}))
    assert rec is not None
    assert rec.thread_root_ts is None and rec.latest_reply is None


def test_a_falsey_thread_ts_fails_the_whole_event_closed(client):
    """The event path is where it matters: this is a live listener payload, and the ticket-holding
    caller fails the observation on ValueError rather than certifying the index caught up."""
    with pytest.raises(TimestampError):
        normalize_slack_event(client, _event(thread_ts=""))


# --------------------------------------------------------------- mention render


def test_render_mentions_resolves_both_forms():
    names = {"U1": "Ann", "U2": "Bob"}
    assert render_mentions("hi <@U1>", names) == "hi @Ann"
    assert render_mentions("hi <@U1|ann>", names) == "hi @Ann"
    assert render_mentions("<@U1> and <@U2|bob>", names) == "@Ann and @Bob"


def test_render_mentions_falls_back_to_the_raw_id():
    assert render_mentions("hi <@U9>", {"U1": "Ann"}) == "hi @U9"
    assert render_mentions("hi <@U9|nine>", {}) == "hi @U9"
    assert render_mentions("hi <@U1>", {"U1": ""}) == "hi @U1"


def test_render_mentions_leaves_plain_text_alone():
    assert render_mentions("no mentions here", {"U1": "Ann"}) == "no mentions here"
    assert render_mentions("", {"U1": "Ann"}) == ""
    assert render_mentions("email a@b.com", {}) == "email a@b.com"
