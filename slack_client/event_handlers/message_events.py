from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from slack_sdk.errors import SlackApiError

from base_client import Message
from config import config
from message_processor.routing_facts import stamp_routing_facts
from slack_client import actor_tail
from slack_client._host import _Host
from slack_client.formatting.blocks import extract_supplementary_text


# Slack's own text for a message deleted with replies (it tombstones the root rather than
# sending message_deleted). Matched as a fallback where the `tombstone` subtype is absent.
_TOMBSTONE_TEXT = "This message was deleted."

# Bound on the live actor-tail dedup set (per process, all channels).
_ACTOR_TAIL_SEEN_MAX = 512


def _channel_post_allowed(cs: Any) -> bool:
    """Does this channel allow a reply at the TOP LEVEL, rather than only inside a thread?

    A row's EXPLICIT True/False wins; None (inherit) or no row at all falls back to the global
    default. Stamped as a real boolean on every dispatch: the old convention wrote the key only
    when it was true, so "threads only" and "nobody resolved this yet" were the same absence —
    and every consumer had to re-derive the setting to tell them apart."""
    value = (cs or {}).get("reply_in_channel")
    if value is None:
        value = config.reply_in_channel_default
    return bool(value)


def _attachment_descriptors(files: Any) -> tuple:
    """Names and TYPES of a message's files, one descriptor each — e.g. ("food.png (image)",
    "report.pdf (file)").

    Names and types only, never content: the gate this feeds does not look inside anything, and it
    is deciding whether the responder runs rather than what the file says. It used to be a single
    prose sentence ("1 image, 1 file (chart.png, notes.pdf)") assembled here and pasted into the
    prompt — which meant this function was quietly writing part of a prompt.

    The per-descriptor format belongs to the gate (``participation.describe_attachment``), because
    the gate CLASSIFIES on it: a captionless cohort is only structurally uninteresting when every
    attachment is an image, and a format invented independently at each end is how that check
    quietly starts matching a spreadsheet. Empty tuple when there are no files.
    """
    from message_processor.participation import describe_attachment

    return tuple(describe_attachment((f or {}).get("name"), (f or {}).get("mimetype"))
                 for f in (files or []))


def attest_message_origin(message: Message, event: Dict[str, Any],
                          sender_type: Optional[str]) -> None:
    """Record that `message` came from a Slack-DELIVERED human event in the conversation it
    names — the one case where the requester's membership needs no lookup, because Slack just
    proved it by delivering their message from there.

    Read by handlers/text.py, which turns these markers into
    ``ToolContext.origin_membership_attested`` for the channel-read authorization gate.

    Called ONLY from the two genuine entry points (a Bolt-delivered app_mention/DM, and a
    Bolt-delivered channel message). Deliberately NOT called for:
      * the settings/welcome replay (event_handlers/settings.py), which re-dispatches a
        message the user sent minutes earlier — they may have left the channel since;
      * the edit-triggered re-dispatch, whose event we assemble ourselves;
      * anything without a `user` (bot posts, system subtypes).
    The markers name the channel and user they were minted for, so a context that later
    carries a different channel id can't reuse them — id equality alone is forgeable.
    """
    if sender_type != "human":
        return
    user_id = event.get("user")
    channel_id = event.get("channel")
    if not isinstance(user_id, str) or not user_id:
        return
    if not isinstance(channel_id, str) or not channel_id:
        return
    if user_id != message.user_id or channel_id != message.channel_id:
        return
    message.metadata["origin_event_verified"] = True
    message.metadata["origin_user_id"] = user_id
    message.metadata["origin_channel_id"] = channel_id



async def _register_raw_receipt(client_self, channel_id, message_ts, kind,
                                thread_root_ts=None, site="onboarding"):
    """Spec §5 for the onboarding posts, which go out through raw chat.postMessage.

    Every one of them is replaceable setup UI — a welcome, a "check your DMs", a reminder to
    configure — so they register as CHROME and stay out of the stream. DM targets no-op by
    themselves (is_dm_conversation)."""
    if not message_ts:
        return
    from message_processor.outbound_receipts import record_transport_post
    try:
        await record_transport_post(
            team_id=getattr(client_self, "self_team_id", None), channel_id=channel_id,
            message_ts=message_ts, receipts=None, receipt_kind=kind,
            # Spec §4: onboarding posts (welcome / reminder / settings-button) are class chrome.
            receipt_class="chrome",
            thread_root_ts=thread_root_ts, site=site)
    except Exception:  # noqa: BLE001 — onboarding chrome never fails a turn
        pass


async def _post_onboarding_notice(client_self, client, *, site, receipt_channel,
                                  thread_root_ts=None, **post_kwargs):
    """Post one piece of onboarding chrome and register it, under the shutdown gate.

    These run inside Bolt callbacks, and Socket Mode stays connected until the very end of
    shutdown — long after the receipt queue closes. Returns None when admission refused it, so
    the message is never sent at all; callers already handle a post that did not happen.

    The post and its registration go through `post_then_register`, so a drain cancelling this
    callback cannot land between Slack accepting the message and the row that accounts for it.
    """
    from message_processor.outbound_receipts import (channel_post_admission,
                                                     post_then_register)

    async with channel_post_admission(site) as admitted:
        if not admitted:
            return None

        async def _post_and_register():
            resp = await client.chat_postMessage(**post_kwargs)
            await _register_raw_receipt(client_self, receipt_channel,
                                        resp.get("ts") if resp else None, "chrome",
                                        thread_root_ts=thread_root_ts, site=site)
            return resp

        return await post_then_register(_post_and_register())


class SlackMessageEventsMixin(_Host):
    if TYPE_CHECKING:
        # Both created lazily on first onboarding turn (hasattr-guarded below), so they are
        # declared rather than assigned.
        _welcomed_users: set
        _reminder_messages: Dict[Any, list]

    async def _event_to_message(self, event: Dict[str, Any], client) -> Message:
        """Convert a Slack event into the universal Message format (no side effects).

        Shared by the mention/DM path (_handle_slack_message) and the channel-listening
        path (_handle_channel_message)."""
        # Extract text; note whether the bot itself was @-mentioned BEFORE we strip mentions
        # (used by channel-listening logic), then resolve mentions for the model.
        text = event.get("text", "")
        mentioned_self = False
        bot_user_id = getattr(self, "bot_user_id", None)
        if bot_user_id:
            from slack_client.formatting.text import text_mentions_user
            mentioned_self = text_mentions_user(text, bot_user_id)
        # Warm the user cache for every mentioned id BEFORE cleaning, so a first-ever
        # mention of any user or bot (e.g. a co-resident assistant) resolves to "@Name".
        # An unresolved mention must never vanish — stripping "<@other-bot> can you…"
        # down to "can you…" made the participation classifier read the question as
        # aimed at THIS bot (live misfire 2026-07-11). Best-effort: on lookup failure
        # the resolver now renders "@<id>", still a visible addressee marker.
        # F3 sender classification (human | self | other_bot) for the wake envelope.
        # Guarded: _event_to_message can run before bot identity is fully wired. Computed
        # HERE (not after cleaning) because the F48 supplementary extraction below needs it.
        try:
            event_sender_type = self.classify_sender(event)
        except Exception:
            event_sender_type = None

        # F48: content Slack delivers OUTSIDE `text` — a pasted TSV arrives as a `table`
        # block in `attachments[]` with no `files` entry at all, and webhook posts carry
        # their whole payload in `attachments[].fields[]`. Rendered RAW and combined with
        # RAW text BEFORE the mention pass below, or `<@U…>` stays raw inside table cells.
        # Never extracted for our OWN messages: our status/welcome/deep-research cards live
        # in exactly these fields and would replay as "evidence" (the F47 attribution bug).
        supplementary = ""
        if event_sender_type != "self":
            supplementary = extract_supplementary_text(event, primary_text=text)
        if supplementary:
            text = f"{text}\n\n{supplementary}" if text.strip() else supplementary

        from slack_client.formatting.text import extract_mention_ids
        user_cache = getattr(self, "user_cache", {}) or {}
        for uid in extract_mention_ids(text):
            if uid and uid != bot_user_id and uid not in user_cache:
                try:
                    await self.get_username(uid, client)
                except Exception as e:
                    self.log_debug(f"Mention warm-up lookup failed for {uid}: {e}")
        text = self._clean_mentions(text)

        # Process attachments (files)
        attachments = []
        files = event.get("files", [])
        for file in files:
            mimetype = file.get("mimetype", "")
            # Determine file type based on mimetype
            file_type = "image" if mimetype.startswith("image/") else "file"

            attachments.append({
                "type": file_type,
                "url": file.get("url_private"),
                "id": file.get("id"),
                "name": file.get("name"),
                "mimetype": mimetype,
                # F40: the wake gate checks the DECLARED size before it downloads anything, so
                # an oversized image is skipped rather than pulled into memory and then thrown
                # away. Slack always sends this on file_share.
                "size": file.get("size"),
            })

        # Get username and timezone for logging
        user_id = event.get("user")
        username = await self.get_username(user_id, client) if user_id else "unknown"
        user_timezone = await self.get_user_timezone(user_id, client) if user_id else "UTC"

        # Get timezone label (EST, PST, etc.), real name, and email if available
        user_tz_label = None
        user_real_name = None
        user_email = None
        if user_id in self.user_cache:
            user_tz_label = self.user_cache[user_id].get('tz_label')
            user_real_name = self.user_cache[user_id].get('real_name')
            user_email = self.user_cache[user_id].get('email')
            self.log_debug(f"User cache for {user_id}: email={user_email}, real_name={user_real_name}")
        else:
            # Try to get from database if not in cache
            user_info = await self.db.get_user_info_async(user_id)
            if user_info:
                user_real_name = user_info.get('real_name')
                user_email = user_info.get('email')
                user_tz_label = user_info.get('tz_label')
                self.log_debug(f"User from DB for {user_id}: email={user_email}, real_name={user_real_name}")

        # Create universal message
        # Every event that reaches here carries a channel and a ts; the casts only tell the
        # checker that, they neither test nor change anything.
        message = Message(
            text=text,
            user_id=cast(str, user_id),
            channel_id=cast(str, event.get("channel")),
            thread_id=cast(str, event.get("thread_ts") or event.get("ts")),
            attachments=attachments,
            metadata={
                "ts": event.get("ts"),
                "mentioned_self": mentioned_self,  # was the bot @-mentioned in the raw text
                "slack_client": client,
                "username": username,  # Add username to metadata
                "user_real_name": user_real_name,  # Add real name to metadata
                "user_email": user_email,  # Add email to metadata
                "user_timezone": user_timezone,  # Add timezone to metadata
                "user_tz_label": user_tz_label,  # Add timezone label (EST, PST, etc.)
                # Minted by Slack on message/app_mention events for AI apps; authorizes
                # assistant.search.context for this interaction. Absent on older/replayed
                # events — the search tool degrades gracefully (Phase B).
                "action_token": event.get("action_token"),
                # F3: human | self | other_bot — lets the wake envelope render "— bot".
                "sender_type": event_sender_type,
                # IMMUTABLE Slack identity, for the stale-send guard's per-sender top-level
                # scope. A display name would be wrong twice over: people rename themselves,
                # and two accounts can share one. Absent (an unattributed post) → the guard
                # omits the top scope entirely rather than bucketing strangers together.
                "sender_id": (event.get("user") or event.get("bot_id")
                              or event.get("app_id")),
            }
        )
        return message

    async def _get_channel_settings(self, channel_id: str):
        """Phase 7: fetch the per-channel settings row (or None). Best-effort; DMs have none."""
        if not channel_id or channel_id.startswith("D"):
            return None
        try:
            return await self.db.get_channel_settings_async(channel_id)
        except Exception as e:
            self.log_debug(f"_get_channel_settings failed: {e}")
            return None

    def _resolve_mode(self, cs) -> str:
        """Per-channel response_mode if set, else the global default."""
        mode = (cs or {}).get("response_mode") or getattr(config, "channel_response_mode", "tag_only")
        return (mode or "tag_only").strip().lower()

    async def _get_channel_response_mode(self, channel_id: str) -> str:
        """Resolve the response mode for a channel: per-channel DB override, else global default."""
        return self._resolve_mode(await self._get_channel_settings(channel_id))

    def _text_mentions_bot_name(self, text: str) -> bool:
        """Deterministic addressing prefilter used only to decide whether configured
        name-addressing makes a message eligible for dispatch. It never decides relevance, wake,
        reply, silence, placement, or settings.

        True if one of the bot's name aliases appears as a whole word (case-insensitive). Whether
        the name is an ADDRESS ("chatgpt, help") or merely a mention of the subject ("chatgpt was
        wrong earlier") is not decided here and cannot be: this only says a `mentions_only` channel
        may let the message through to the gate, which is what judges it."""
        if not text:
            return False
        import re
        for alias in getattr(config, "bot_name_aliases", []) or []:
            if alias and re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
                return True
        return False

    async def _thread_participation(self, channel_id: str, thread_ts: str):
        """Best-effort (bot_present, distinct_human_count, other_bot_count) for a thread.

        `bot_present` is what decides whether an untagged reply may skip the gate: participation
        in a thread is itself the wake signal — a thread we have posted in is one we are already
        part of, and the responder, which can see the thread, decides what the turn owes,
        including nothing. `human_count` and `other_bots` now decide only STRICT status (the bot,
        at most one human, no other agents), which is the level-independent rule; membership on
        its own wakes us only in an `on` channel.

        Three things this probe is not:
        - It is `conversations_replies(limit=50)`, ascending from the root, so `bot_present` really
          means "we posted within the thread's first 50 messages" and the two counts are FIRST-PAGE
          counts. Accepted: it fails safe — a miss sends the reply to the gate, which is today's
          behaviour.
        - `bot_present` is set by `is_own_message` alone, so ANY post of ours establishes
          membership, including chrome (a thinking placeholder, a settings footer, an error
          notice). Accepted: chrome only appears in a thread because a turn ran for us there, so it
          is weak evidence of the same thing rather than false evidence.
        - On error → (False, 0, 0), which is indistinguishable from an empty first page. The
          exception line logged below is where the two are told apart."""
        try:
            result = await self.app.client.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=50
            )
            msgs = result.get("messages", [])
        except Exception as e:
            self.log_debug(f"_thread_participation failed: {e}")
            return (False, 0, 0)
        bot_present = False
        humans = set()
        other_bots = set()
        for m in msgs:
            if self.is_own_message(m):
                bot_present = True
            elif self.classify_sender(m) == "human":
                uid = m.get("user")
                if uid:
                    humans.add(uid)
            else:
                other_bots.add(m.get("bot_id") or m.get("user") or "bot")
        return (bot_present, len(humans), len(other_bots))

    # Slack subtypes that are NOT semantic messages (edits/deletes/membership/topic
    # churn) — no actor spoke, so they never reach the actor tail. Everything else,
    # INCLUDING bot_message and ordinary content subtypes (file_share, thread_broadcast),
    # is a real speaker in a real thread.
    _TAIL_FEED_SKIP_SUBTYPES = frozenset({
        "message_changed", "message_deleted", "message_replied",
        "channel_join", "channel_leave", "channel_topic", "channel_purpose",
        "channel_name", "channel_archive", "channel_unarchive",
        "group_join", "group_leave", "bot_add", "bot_remove",
        "tombstone", "reminder_add", "pinned_item", "unpinned_item",
    })

    # Content-bearing subtypes that DO drive a RESPONSE (F14): file/image/doc uploads
    # arrive as `file_share` and thread→channel broadcasts as `thread_broadcast`. Both
    # carry real content (and, for file_share, a `files` array) and must reach the
    # response gate so intent classification can route vision/document flows. Every
    # OTHER subtype (edits/deletes/joins/topic churn) stays excluded from the gate.
    _RESPONSE_GATE_CONTENT_SUBTYPES = frozenset({"file_share", "thread_broadcast"})

    @staticmethod
    def _bot_message_has_content(event: Dict[str, Any]) -> bool:
        """True when a `bot_message` actually carries something to respond to — real text, files,
        or supplementary block/attachment content (a webhook's fields) — versus bare chrome. Used
        so a Jira/GitHub webhook with empty `text` still reaches the response gate (F48)."""
        if (event.get("text") or "").strip():
            return True
        if event.get("files"):
            return True
        try:
            return bool(extract_supplementary_text(event, primary_text=event.get("text") or ""))
        except Exception:  # noqa: BLE001 — never let content-sniffing break dispatch
            return False

    # ------------------------------------------------------------- actor tail (live feed)

    def _actor_tail_seen(self):
        """Bounded (channel, ts) set of actors already recorded this process (lazy-init)."""
        seen = getattr(self, "_actor_tail_seen_map", None)
        if seen is None:
            from collections import OrderedDict
            seen = OrderedDict()
            self._actor_tail_seen_map = seen
        return seen

    def _feed_actor_tail(self, event: Dict[str, Any]) -> None:
        """Record WHO spoke in a thread — never what they said (spec §8). SYNCHRONOUS.

        Called straight from the raw Slack listeners, deliberately NOT from ambient ingest: the
        tail's one reader is the thread-continuation fast path, where it decides STRICT status —
        a second agent past the replies probe's first page means the thread is not 1:1, though in
        an `on` channel membership still wakes us — and that route must not depend on whether
        ambient memory happens to be wired. Both listeners call it, so a
        mention arrives twice; the (channel, ts) dedup keeps the second delivery from counting as
        a second speaker — and from bumping the generation a turn's stream reconcile is watching.

        Never raises: a missed actor costs one too-eager continuation, an exception here costs
        the event.
        """
        try:
            if not isinstance(event, dict):
                return
            channel_id = event.get("channel")
            if not channel_id or str(channel_id).startswith("D"):
                return  # a DM has no second agent to find and no continuation to cancel
            subtype = event.get("subtype")
            if subtype == "message_deleted":
                prev = event.get("previous_message") or {}
                self._forget_actor(channel_id, event.get("deleted_ts") or prev.get("ts"))
                return
            if subtype == "message_changed":
                edited = event.get("message") or {}
                if edited.get("subtype") == "tombstone" or (
                        (edited.get("text") or "").strip() == _TOMBSTONE_TEXT):
                    self._forget_actor(channel_id, edited.get("ts"))
                    return
                # An ordinary edit changes the words, not the speaker — recording is idempotent,
                # and it is the one chance to learn about a message posted before we started.
                self._record_actor(channel_id, edited)
                return
            if subtype in self._TAIL_FEED_SKIP_SUBTYPES:
                return
            self._record_actor(channel_id, event)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"actor tail feed failed: {e}")

    def _record_actor(self, channel_id: str, msg: Dict[str, Any]) -> None:
        ts = msg.get("ts")
        if not ts:
            return
        seen = self._actor_tail_seen()
        key = f"{channel_id}|{ts}"
        if key in seen:
            return
        try:
            if self.is_own_message(msg):
                return  # the tail exists to spot OTHER agents; our own posts are noise in it
            sender_type = self.classify_sender(msg)
        except Exception:  # noqa: BLE001
            return
        if sender_type == "self":
            return
        # Root asymmetry (see actor_tail): a root files under its OWN ts, so it shares a bucket
        # with its replies.
        actor_tail.record(channel_id, ts=ts, root_ts=msg.get("thread_ts") or ts,
                          is_bot=sender_type != "human", sender_type=sender_type)
        seen[key] = True
        seen.move_to_end(key)
        while len(seen) > _ACTOR_TAIL_SEEN_MAX:
            seen.popitem(last=False)

    def _forget_actor(self, channel_id: str, ts: Optional[str]) -> None:
        if not ts:
            return
        actor_tail.remove(channel_id, ts)
        self._actor_tail_seen().pop(f"{channel_id}|{ts}", None)

    def _ambient_service(self):
        """The AmbientArtifactService (owned by the processor), or None if not wired/available."""
        proc = getattr(self, "processor", None)
        return getattr(proc, "ambient_service", None) if proc is not None else None

    def _channel_summary_service(self):
        """The ChannelSummaryService (owned by the processor), or None if not wired/available."""
        proc = getattr(self, "processor", None)
        return getattr(proc, "channel_summary_service", None) if proc is not None else None

    async def _invalidate_channel_summary(self, channel_id, ts) -> None:
        """Track 1: an edit/delete of message `ts` may have touched the summarized window — tell
        the ChannelSummaryService so it invalidates + stops injecting until a rebuild. Best-effort;
        never raises into the ambient path."""
        svc = self._channel_summary_service()
        if svc is None:
            return
        try:
            await svc.note_message_mutation(channel_id, ts)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"channel summary invalidate hook failed: {e}")

    def _mark_thread_refresh(self, channel_id: str, thread_root: str) -> None:
        """Flag a thread's warm ThreadState for rebuild-from-Slack. On an edit/delete the pulse is
        corrected, but a live in-memory ThreadState can still hold the deleted/pre-edit message —
        marking it needs_refresh makes the next turn refetch from Slack (the source of truth)."""
        if not channel_id or not thread_root:
            return
        proc = getattr(self, "processor", None)
        tm = getattr(proc, "thread_manager", None) if proc is not None else None
        if tm is not None and hasattr(tm, "mark_needs_refresh"):
            try:
                tm.mark_needs_refresh(f"{channel_id}:{thread_root}")
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"mark_needs_refresh failed: {e}")

    async def _ambient_ingest(self, event: Dict[str, Any], client) -> None:
        """F51 capture + lifecycle seam, invoked at the registered Slack message event BEFORE the
        channel_type / channel-listening branch — so ambient content is captured even when
        listening or participation is off. Handles new content (enqueue), edits (reconcile +
        re-enqueue), and deletions (purge artifacts). Best-effort; never raises, never blocks the
        wake path (offer_event only enqueues)."""
        svc = self._ambient_service()
        if svc is None:
            return
        # The service needs the SlackBot FACADE — it owns download_file() (image/file capture).
        # The Bolt `client` is a raw AsyncWebClient without it, so passing that makes every
        # image/file job AttributeError into download_failed. `self` IS that facade.
        facade = self
        try:
            subtype = event.get("subtype")
            channel_id = event.get("channel")
            if subtype == "message_deleted":
                prev = event.get("previous_message") or {}
                deleted_ts = event.get("deleted_ts") or prev.get("ts")
                if channel_id and deleted_ts:
                    db = getattr(self, "db", None)
                    if db is not None:
                        try:
                            await db.delete_ambient_artifacts_by_source(channel_id, deleted_ts)
                        except Exception as e:
                            self.log_debug(f"ambient delete-by-source failed: {e}")
                    # A warm ThreadState may still hold the deleted message — force a rebuild.
                    self._mark_thread_refresh(channel_id, prev.get("thread_ts") or deleted_ts)
                    # Track 1: a deleted message inside the narrative's window invalidates the cache.
                    await self._invalidate_channel_summary(channel_id, deleted_ts)
                    self.log_debug(f"message_deleted: purged artifacts for "
                                   f"{channel_id}:{deleted_ts}")
                return
            if subtype == "message_changed":
                edited = event.get("message") or {}
                new_ts = edited.get("ts")
                # Deleting a root that has (or had) replies does NOT arrive as
                # message_deleted — Slack tombstones it: message_changed whose nested
                # message carries subtype "tombstone" / the text "This message was
                # deleted." Treating that as an ordinary edit runs the edit-triggered engine
                # on the tombstone text (seen live 2026-07-18: six tombstones dispatched, one
                # classified — the model then "remembered" threads that no longer existed).
                # It is a deletion: purge the root's ambient artifacts and force a thread
                # rebuild — never offer, never classify.
                if edited.get("subtype") == "tombstone" or (
                        (edited.get("text") or "").strip() == _TOMBSTONE_TEXT):
                    if channel_id and new_ts:
                        db = getattr(self, "db", None)
                        if db is not None:
                            try:
                                await db.delete_ambient_artifacts_by_source(channel_id, new_ts)
                            except Exception as e:
                                self.log_debug(f"ambient tombstone delete failed: {e}")
                        self._mark_thread_refresh(
                            channel_id, edited.get("thread_ts") or new_ts)
                        # Track 1: a deleted-with-replies root inside the window invalidates too.
                        await self._invalidate_channel_summary(channel_id, new_ts)
                        self.log_debug(f"tombstoned root: purged {channel_id}:{new_ts} "
                                       f"(deleted-with-replies)")
                    return
                if channel_id and new_ts:
                    db = getattr(self, "db", None)
                    if db is not None:
                        try:
                            await db.delete_ambient_artifacts_by_source(channel_id, new_ts)
                        except Exception as e:
                            self.log_debug(f"ambient reconcile delete failed: {e}")
                    # Track 1: an edit to a message inside the narrative's window makes the cache
                    # stale — invalidate and stop injecting until a background rebuild succeeds.
                    await self._invalidate_channel_summary(channel_id, new_ts)
                    # Re-offer the edited content as a synthetic message event.
                    synthetic = dict(edited)
                    synthetic["channel"] = channel_id
                    synthetic.setdefault("ts", new_ts)
                    if not synthetic.get("thread_ts") and event.get("message", {}).get("thread_ts"):
                        synthetic["thread_ts"] = event["message"]["thread_ts"]
                    if not self.is_own_message(synthetic):
                        # A warm ThreadState may still hold the pre-edit text — force a rebuild.
                        self._mark_thread_refresh(
                            channel_id, synthetic.get("thread_ts") or new_ts)
                        svc.offer_event(synthetic, facade)
                # F52: after the reconcile above, an edit may also DRIVE a reply (feature-flagged).
                # Zero-cost pre-gates run synchronously inside; nothing is scheduled unless they
                # all pass, so an unfurl/attachment-only or identical-text edit still costs nothing.
                self._maybe_edit_triggered_reply(event, client)
                return
            # Ordinary content: enqueue. Own messages are excluded (recursion guard).
            if self.is_own_message(event):
                return
            # Images are admitted IMMEDIATELY, exactly like links and files. The rich gate used to
            # download a message's pictures for its verdict, so an image's ambient vision job was
            # held back to let one look serve both; the binary gate never looks at a picture, so a
            # hold would wait on a resolver that is never coming.
            svc.offer_event(event, facade)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"ambient ingest failed: {e}")

    # ------------------------------------------------------------- F52: edit-triggered replies

    @staticmethod
    def _edit_normalize(text: Any) -> str:
        """Whitespace-normalized text for the 'did the content actually change?' pre-gate.
        Slack fires message_changed for link unfurls and attachment changes with byte-identical
        text; collapsing whitespace makes those compare equal so they cost — and trigger — nothing."""
        return " ".join(str(text or "").split())

    def _edit_reply_seqs(self) -> Dict[str, str]:
        """Per-(channel, message) marker of the NEWEST edit seen, for burst collapse (lazy-init)."""
        seqs = getattr(self, "_edit_reply_seq_map", None)
        if seqs is None:
            seqs = {}
            self._edit_reply_seq_map = seqs
        return seqs

    def _supersede_original_participation(self, channel_id: str, msg_ts: str,
                                          edited: Dict[str, Any]) -> None:
        """F52: tell the participation engine to CANCEL the original (pre-edit) message's
        in-flight evaluation. An edit keeps the message's ts, so the engine's ordinary
        newer-arrival supersession can't fire; this marks the exact (conversation, ts) so a
        stale respond verdict never posts a duplicate. Best-effort — the engine is only wired
        in the live app (main.py sets processor.participation_engine); absent in unit harnesses."""
        engine = getattr(getattr(self, "processor", None), "participation_engine", None)
        if engine is None or not hasattr(engine, "supersede"):
            return
        try:
            engine.supersede(channel_id, msg_ts,
                             thread_root=(edited or {}).get("thread_ts"),
                             sender_id=(edited or {}).get("user"))
        except Exception as e:  # noqa: BLE001 — never let supersession break ingest
            self.log_debug(f"edit participation supersede failed: {e}")

    def _register_edit_dispatch(self, channel_id: str, msg_ts: str, marker: str) -> None:
        """F52 queue-drop backstop: record that (channel, ts) was edited and is being handled by
        the edit path, tagged with the surviving edit's `marker`. The drain (base.py) drops a
        queued PRE-EDIT participation dispatch for this ts — one whose marker doesn't match — that
        slipped into the busy queue before supersession landed. The edit's OWN engine re-dispatch
        carries the matching marker and is kept. Bounded."""
        from collections import OrderedDict
        reg = getattr(self, "_edit_dispatch_reg", None)
        if reg is None:
            reg = OrderedDict()
            self._edit_dispatch_reg = reg
        key = f"{channel_id}|{msg_ts}"
        reg[key] = str(marker)
        reg.move_to_end(key)
        while len(reg) > 256:
            reg.popitem(last=False)

    def edit_dispatch_marker(self, channel_id: str, ts: str):
        """The surviving edit's marker for (channel, ts), or None. Read by the queue drain to tell
        the edit's own re-dispatch (marker matches) from a stale pre-edit dispatch (it doesn't)."""
        reg = getattr(self, "_edit_dispatch_reg", None)
        if not reg or not channel_id or ts is None:
            return None
        return reg.get(f"{channel_id}|{ts}")

    def _note_app_mention_seen(self, channel_id: str, ts: str) -> None:
        """F52: record a GENUINE Slack app_mention delivery, keyed (channel, ts). Editing a
        message to ADD the bot's @mention makes Slack deliver a real app_mention for the same ts
        (observed live 2026-07-16); the edit-reply path checks this to avoid dispatching a
        duplicate synthetic addressed turn. Bounded."""
        if not channel_id or not ts:
            return
        from collections import OrderedDict
        seen = getattr(self, "_app_mention_seen", None)
        if seen is None:
            seen = OrderedDict()
            self._app_mention_seen = seen
        key = f"{channel_id}|{ts}"
        seen[key] = time.time()
        seen.move_to_end(key)
        while len(seen) > 512:
            seen.popitem(last=False)

    def _app_mention_recently_seen(self, channel_id: str, ts: str) -> bool:
        """F52: True iff Slack already delivered a genuine app_mention for (channel, ts)."""
        seen = getattr(self, "_app_mention_seen", None)
        if not seen or not channel_id or not ts:
            return False
        return f"{channel_id}|{ts}" in seen

    def _stash_edit_context(self, channel_id: str, msg_ts: str, *, old_text: str,
                            new_text: str, already_replied: bool, marker: str) -> None:
        """Stash edit context on THIS facade (which the engine's evaluate is handed as `client`),
        keyed by (channel, ts, MARKER).

        The marker is ownership. An edit keeps its original Slack timestamp, so (channel, ts) alone
        cannot distinguish the edit's own evaluation from the stale original attempt it superseded —
        and whichever ran first popped the context. The original could therefore arrive holding the
        edit's before/after text, conclude it WAS the edit, and in doing so skip the supersession
        check meant to silence it. Only the attempt carrying this marker may consume it. Bounded so
        a long-lived process can't accumulate stale contexts."""
        from collections import OrderedDict
        store = getattr(self, "_edit_reply_ctx_map", None)
        if store is None:
            store = OrderedDict()
            self._edit_reply_ctx_map = store
        key = f"{channel_id}|{msg_ts}|{marker}"
        store[key] = {"old_text": old_text or "", "new_text": new_text or "",
                      "already_replied": bool(already_replied)}
        store.move_to_end(key)
        while len(store) > 256:
            store.popitem(last=False)

    def _schedule_edit_reply(self, coro) -> None:
        """Fire-and-forget the debounce+routing so ambient ingest never blocks. Prefer the
        processor's tracked scheduler; fall back to a tracked create_task (and, with no running
        loop, close the coroutine cleanly rather than leak an un-awaited warning)."""
        proc = getattr(self, "processor", None)
        if proc is not None and hasattr(proc, "_schedule_async_call"):
            proc._schedule_async_call(coro)
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return
        task = asyncio.create_task(coro)
        tasks = getattr(self, "_edit_reply_tasks", None)
        if tasks is None:
            tasks = set()
            self._edit_reply_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def _maybe_edit_triggered_reply(self, event: Dict[str, Any], client) -> None:
        """F52: decide (zero-cost) whether an edit should drive a reply, and if so hand the
        debounce + routing to a background task. The pre-gates run in order and each costs
        NOTHING (no model call, no I/O); only when they all pass is a task scheduled. Best-effort;
        never raises (an edit-reply failure must never break ambient ingest)."""
        try:
            # 1. Flag off → exactly today's behavior.
            if not getattr(config, "enable_edit_triggered_replies", False):
                return
            edited = event.get("message") or {}
            previous = event.get("previous_message") or {}
            channel_id = event.get("channel")
            msg_ts = edited.get("ts")
            if not channel_id or not msg_ts:
                return
            # 2. Human author only — never the bot itself (its streamed chat.update edits arrive
            #    here as subtype bot_message / own) and never another bot/app.
            if self.classify_sender(edited) != "human":
                return
            # 4. Normalized text must ACTUALLY change (unfurl / attachment-only edits carry
            #    identical text → cost and trigger nothing).
            old_text = previous.get("text") or ""
            new_text = edited.get("text") or ""
            if self._edit_normalize(old_text) == self._edit_normalize(new_text):
                return
            # 5. Only edits of messages younger than the window — age from the ORIGINAL ts.
            window_min = int(getattr(config, "edit_reply_window_minutes", 60) or 0)
            if window_min > 0:
                try:
                    age = time.time() - float(msg_ts)
                except (TypeError, ValueError):
                    return
                if age > window_min * 60:
                    return
            # 3. Channel type + routing branch. A DM is inherently addressed (the ordinary DM path
            #    answers every message); a channel edit that ADDS the bot's @mention is an
            #    addressed wake app_mention never fires for. Both take the addressed path and do
            #    NOT require channel listening (an @mention/DM is answered regardless). Every other
            #    channel edit goes to the engine's typo-vs-meaning judgment, and only where a NEW
            #    non-mention channel message would be seen at all — i.e. channel listening on.
            bot_uid = getattr(self, "bot_user_id", None)
            from slack_client.formatting.text import text_mentions_user
            mention_new = bool(bot_uid and text_mentions_user(new_text, bot_uid))
            mention_old = bool(bot_uid and text_mentions_user(old_text, bot_uid))
            mention_added = mention_new and not mention_old
            is_dm = str(channel_id).startswith("D")
            addressed = is_dm or mention_added
            if not addressed and not config.enable_channel_listening:
                return
            # F52 double-answer fix: as EARLY as possible (synchronously, before the edit's own
            # debounce), cancel the original message's in-flight participation evaluation. The
            # original kept this ts, so the engine's newer-arrival supersession can't fire on its
            # own — without this, an already-answerable pre-edit message posts a stale second
            # answer while the edit is handled on the addressed / fresh-eval path.
            self._supersede_original_participation(channel_id, msg_ts, edited)
            self._schedule_edit_reply(self._run_edit_triggered_reply(
                event, client, channel_id, msg_ts, old_text, new_text, is_dm, mention_added))
        except Exception as e:  # noqa: BLE001 — never let edit-reply gating break ingest
            self.log_debug(f"edit-triggered reply gating failed: {e}")

    async def _run_edit_triggered_reply(self, event: Dict[str, Any], client, channel_id: str,
                                        msg_ts: str, old_text: str, new_text: str,
                                        is_dm: bool, mention_added: bool) -> None:
        """F52 (background): collapse an edit BURST, then route. Rapid successive edits of one
        message keep the SAME message ts, so the engine's ts-keyed debounce can't separate them —
        we collapse here on the edit's own unique marker, keyed per (channel, message) so only the
        NEWEST edit in a burst survives and unrelated traffic never interferes. Best-effort."""
        try:
            edited = event.get("message") or {}
            # A unique-per-edit marker: the edited-at ts, falling back to the message_changed
            # event ts. Two edits of the same message get two different markers.
            marker = str((edited.get("edited") or {}).get("ts")
                         or event.get("ts") or event.get("event_ts") or msg_ts)
            # The OUTER message_changed event_ts — when the edit ARRIVED, not when the message was
            # first posted. It is what the listeners admitted into the watermark for this event, so
            # it is also what this turn must pin H against: the subject ts may be hours old, and a
            # turn pinned there would build a stream that stops before the edit that woke it.
            admission_ts = event.get("event_ts") or event.get("ts") or msg_ts
            seq_key = f"{channel_id}|{msg_ts}"
            seqs = self._edit_reply_seqs()
            seqs[seq_key] = marker
            wait = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
            if wait:
                await asyncio.sleep(wait)
            if seqs.get(seq_key) != marker:
                return  # a newer edit of the SAME message arrived → this one is collapsed away
            seqs.pop(seq_key, None)

            # Build a synthetic FRESH message event (no message_changed subtype) carrying the
            # edited content at its ORIGINAL ts, so threading / reply-placement behave as if the
            # message were posted fresh.
            synthetic = dict(edited)
            synthetic["channel"] = channel_id
            synthetic.setdefault("ts", msg_ts)
            thread_ts = edited.get("thread_ts")
            if thread_ts:
                synthetic["thread_ts"] = thread_ts
            synthetic.pop("subtype", None)
            synthetic.pop("edited", None)

            # F52 queue-drop backstop: tag this ts as edit-handled with the surviving marker, so a
            # stale PRE-EDIT participation dispatch that already slipped into the busy queue is
            # dropped at drain (the edit's own engine re-dispatch below carries the same marker
            # and is kept).
            self._register_edit_dispatch(channel_id, msg_ts, marker)

            if is_dm or mention_added:
                # F52 double-answer fix: a mention ADDED by an edit makes Slack deliver a GENUINE
                # app_mention for the same ts (observed live 2026-07-16). When that already
                # arrived, this synthetic addressed dispatch is a pure duplicate — skip it and let
                # Slack's app_mention answer. Kept as a fallback for surfaces where Slack fires
                # none (the original F52 assumption). DMs never fire app_mention → always dispatch.
                if (mention_added and not is_dm
                        and self._app_mention_recently_seen(channel_id, msg_ts)):
                    self.log_debug(
                        f"Edit added a mention but Slack already delivered app_mention for "
                        f"{channel_id}:{msg_ts} — skipping duplicate synthetic dispatch")
                    return
                # Addressed wake — route into the very path an ordinary new mention/DM takes.
                await self._handle_slack_message(
                    synthetic, client, wake_source="dm" if is_dm else "app_mention",
                    admission_ts=admission_ts)
                return
            # Otherwise: the participation engine's full judgment, carrying the edit context. The
            # marker rides the dispatched message so the queue drain keeps THIS (edit) dispatch.
            await self._dispatch_edit_to_engine(
                client, synthetic, channel_id, msg_ts, old_text, new_text, marker=marker,
                admission_ts=admission_ts)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"edit-triggered reply run failed: {e}")

    async def _dispatch_edit_to_engine(self, client, synthetic: Dict[str, Any], channel_id: str,
                                       msg_ts: str, old_text: str, new_text: str,
                                       marker: Optional[str] = None,
                                       admission_ts: Optional[str] = None) -> None:
        """F52: send a non-mention channel edit through the participation engine, respecting the
        SAME gating a new message gets, and stashing the edit context so the classifier can make
        the typo-vs-meaning call. Mirrors _handle_channel_message's participation-check condition:
        the gate only judges a message a new post would also reach (level `on` always; a
        name/mention hit under any level). An edit that a new message wouldn't respond to stays
        silent."""
        from message_processor.participation import resolve_participation_level
        cs = await self._get_channel_settings(channel_id)
        level = resolve_participation_level(cs)
        if level == "off":
            return  # participation off means off — an edit must never respond where a new msg can't
        if not getattr(config, "enable_participation_engine", True):
            return  # no engine → no typo-vs-meaning judgment → silent (like a new ambient message)

        from slack_client.formatting.text import text_mentions_user
        bot_uid = getattr(self, "bot_user_id", None)
        mention_present = bool(bot_uid and text_mentions_user(new_text, bot_uid))
        name_hit = self._text_mentions_bot_name(new_text)
        if not (mention_present or name_hit or level == "on"):
            return  # mentions_only + no mention/name → silent, exactly as a new ambient message is

        # Already-replied signal: _thread_participation runs one conversations.replies and reports
        # bot_present — the bot already appears in this message's thread (a top-level answer lands
        # in-thread under the original ts). This is the cheapest reliable "did we answer it" signal.
        thread_root = synthetic.get("thread_ts") or msg_ts
        already_replied = False
        try:
            bot_present, _, _ = await self._thread_participation(channel_id, thread_root)
            already_replied = bool(bot_present)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"edit already-replied probe failed: {e}")

        # Stash the edit context where evaluate (handed this same facade as `client`) reads it,
        # under THIS edit's marker so only this attempt can claim it.
        self._stash_edit_context(channel_id, msg_ts, old_text=old_text,
                                 new_text=new_text, already_replied=already_replied,
                                 marker=str(marker or ""))

        message = await self._event_to_message(synthetic, client)
        message.thread_id = synthetic.get("thread_ts") or msg_ts
        message.metadata["channel_listen"] = True
        message.metadata["participation_level"] = level
        # An edit re-dispatch is ambient traffic that must pass the gate — and, like any
        # gate-routed turn, the responder may decide the edit deserves no words at all.
        stamp_routing_facts(message, gate_required=True, silence_capable=True,
                            addressed=False, ts=msg_ts,
                            thread_ts=synthetic.get("thread_ts"))
        # The edit's OUTER event_ts, not the message ts (see _run_edit_triggered_reply).
        message.metadata["trigger_admission_ts"] = str(admission_ts or msg_ts)
        # F52: mark this as the EDIT's own dispatch so the queue drain keeps it (a queued PRE-EDIT
        # dispatch for the same ts carries no marker and is dropped as stale).
        if marker is not None:
            message.metadata["edit_reply_marker"] = str(marker)
        # A mention/name hit reads as prompted (like the wake path) so its reply doesn't burn the
        # unprompted-pacing budget; the engine still judges whether it's genuinely addressed.
        if mention_present or name_hit:
            message.metadata["participation_name_hit"] = True
            message.metadata["wake_source"] = "name_mention"
        else:
            message.metadata["wake_source"] = "ambient"
        descriptors = _attachment_descriptors(synthetic.get("files"))
        if descriptors:
            message.metadata["participation_attachments"] = descriptors
        message.metadata["channel_post_allowed"] = _channel_post_allowed(cs)

        self.log_debug(
            f"Edit-triggered engine dispatch: channel={channel_id}, ts={msg_ts}, level={level}, "
            f"name_hit={name_hit}, mention_present={mention_present}, "
            f"already_replied={already_replied}")
        if self.message_handler:
            await self.message_handler(message, self)

    async def _ambient_file_deleted(self, event: Dict[str, Any]) -> None:
        """F51: a Slack `file_deleted` event — purge summaries derived from that file id across
        the workspace. Best-effort; never raises.

        Spec §5: it is also the ONLY Slack-confirmed deletion a pending share will ever get. A
        file that was deleted before its share ts resolved would otherwise be re-polled and
        logged critically at every boot for the life of the database.
        """
        db = getattr(self, "db", None)
        file_id = event.get("file_id") or (event.get("file") or {}).get("id")
        if db is None or not file_id:
            return
        try:
            await db.delete_ambient_artifacts_by_file_id(file_id)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"ambient file_deleted purge failed for {file_id}: {e}")
        try:
            from message_processor.outbound_receipts import delete_pending_shares_for_file
            await delete_pending_shares_for_file(db, file_id)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"pending share cleanup failed for {file_id}: {e}")

    async def _handle_channel_message(self, event: Dict[str, Any], client):
        """Phase 5: decide whether to respond to a NON-mention channel message, then dispatch.

        SAFE BY DEFAULT — the caller already gated on config.enable_channel_listening. Honors
        channel_response_mode (default 'tag_only'); short-circuits our own posts; de-dups against
        the app_mention event; and bypasses the welcome/settings onboarding flow entirely."""
        # Ignore non-real messages (edits, deletes, joins, message_changed, etc.) for the
        # RESPONSE gate — they never drive a reply (the stream and the actor tail have already
        # taken account of them at the listener).
        # EXCEPTION (F14): content-bearing subtypes (file_share uploads, thread_broadcast)
        # ARE real content and proceed through the gate; _event_to_message plumbs any
        # `files` onto the Message exactly as the @-mention path does, so downstream intent
        # classification can route vision/document flows.
        subtype = event.get("subtype")
        if subtype and subtype not in self._RESPONSE_GATE_CONTENT_SUBTYPES:
            # F48: a `bot_message` webhook (Jira/GitHub/Drive) often has EMPTY text and carries
            # its whole payload in attachments[].fields[] / blocks. Such a supplementary-bearing
            # bot post IS real content and must reach the gate (the engine then judges it); bare
            # bot chrome with nothing to say still drops here.
            if subtype == "bot_message" and self._bot_message_has_content(event):
                pass
            else:
                return
        # Loop guard FIRST: never act on our own posts.
        if self.is_own_message(event):
            return

        channel_id = cast(str, event.get("channel"))
        cs = await self._get_channel_settings(channel_id)
        # Participation levels (off / mentions_only / on). participation_level wins over the
        # legacy response_mode; both map cleanly (off≡off, tag_only≡mentions_only, auto_respond≡on).
        from message_processor.participation import resolve_participation_level
        level = resolve_participation_level(cs)
        if level == "off":
            return

        text = event.get("text", "") or ""

        # Dedup: an explicit @mention is already delivered via the app_mention event — skip here.
        bot_user_id = getattr(self, "bot_user_id", None)
        if bot_user_id:
            from slack_client.formatting.text import text_mentions_user
            if text_mentions_user(text, bot_user_id):
                return

        # A name-in-text hit is a SIGNAL, not a verdict: "chatgpt, help" (addressed),
        # "chatgpt was wrong earlier" (discussed), and "I asked ChatGPT on my phone"
        # (OpenAI's product) all match the regex — only the engine can tell them apart.
        # True @mentions stay deterministic via the app_mention event (deduped above).
        name_hit = self._text_mentions_bot_name(text)

        # Thread replies: an untagged reply in a thread we are already part of skips the gate.
        # Two rules produce that, and they are not the same rule:
        #   strict 1:1 — a HUMAN sender, the bot, at most one human, no other bots/agents.
        #     Level-independent, and unchanged: cheap, and practically always right. It is also
        #     the only one of the two that carries structural authority.
        #   membership, `on` channels only — we have posted in this thread, whoever else is in it
        #     and whoever wrote the reply, another bot included. Participation in a thread is
        #     itself the wake signal: a thread we have posted in is one we are already part of,
        #     and the responder, which can SEE the thread (the gate sees only the trigger text),
        #     decides what the turn owes — including nothing.
        #
        # OTHER BOTS IN A THREAD ARE NOT GATED (owner decision, uncapped). Two assistants
        # discussing something is a thing people here deliberately set up, and the gate — which
        # sees one message and no thread — is the wrong judge of whether the exchange is worth
        # continuing. Nothing bounds the exchange in code: the ONLY brake is each side deciding
        # it has nothing to add, which for us is the responder's silence rule ("two assistants
        # answering is worse than none") on a silence_capable turn. That is the accepted trade.
        # A bot reply at TOP LEVEL is unchanged and still gated — this is about threads.
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")

        sender_is_bot = self.classify_sender(event) != "human"
        direct_continuation = False
        # True only when the WIDENED rule is the sole reason we skipped the gate. It withholds
        # structural authority downstream (handlers/text.py) and nothing else. Always true for a
        # bot sender, which can never reach the strict rule.
        membership_wake = False
        # What the thread probe said, or None when it never ran (not a thread reply). A bare 0
        # would be ambiguous — a failed probe and a genuinely empty first page both produce it —
        # so the log carries this string instead. Debug only.
        thread_probe: Optional[str] = None
        if thread_ts and thread_ts != ts:
            bot_present, human_count, other_bots = await self._thread_participation(channel_id, thread_ts)
            # NOTE: `_thread_participation` returns (False, 0, 0) on API failure, so
            # "bot_present=False,humans=0,other_bots=0" reads the same for an empty probe and a
            # failed one; it logs its own exception line, which is where the two are told apart.
            thread_probe = f"bot_present={bot_present},humans={human_count},other_bots={other_bots}"
            if bot_present:
                # F5 fix (b): the replies fast path only scans the oldest page (limit=50) and can
                # miss a SECOND bot later in a long thread. The actor tail knows who has spoken in
                # this thread — if it shows another agent, this thread is not strictly 1:1. It is
                # no longer a veto on waking, only on strict status.
                #
                # `not sender_is_bot` is part of strict on purpose: a judgment-free answer to a
                # bot is a loop seed, and strict is the route that answers with no gate AND full
                # structural authority. A bot reply takes the membership route instead, which is
                # `on`-only and authority-free.
                strict_1to1 = (not sender_is_bot and human_count <= 1 and other_bots == 0
                               and not actor_tail.thread_has_other_bot(channel_id, thread_ts))
                if strict_1to1:
                    direct_continuation = True
                elif level == "on":
                    # Ruling 1A: the widening respects `mentions_only`, where we tell the user
                    # verbatim that nothing but a mention or a bare name wakes us.
                    direct_continuation = True
                    membership_wake = True

        # Decide: a thread continuation (strict 1:1, or membership in an `on` channel) → respond
        # directly; `on` → the gate judges every message; `mentions_only` → the gate judges ONLY
        # name-bearing messages (zero model cost otherwise, and a real @mention never arrives here
        # — it comes via app_mention); engine disabled → legacy deterministic name wake (humans
        # only — a bot naming us at top level must never trigger a judgment-free reply, that's a
        # loop seed; in a thread we are part of, the membership rule above has already decided).
        engine_on = getattr(config, "enable_participation_engine", True)
        gate_required = False
        if direct_continuation:
            pass  # respond directly
        elif engine_on and (level == "on" or name_hit):
            gate_required = True
        elif not engine_on and name_hit and not sender_is_bot:
            pass  # legacy deterministic name wake (engine disabled)
        else:
            return

        # Build the universal message (no onboarding side effects) and dispatch.
        message = await self._event_to_message(event, client)
        # Slack delivered this human message from this channel, so the sender is provably in
        # it — the channel-read gate may skip the membership lookup for THIS conversation.
        attest_message_origin(message, event, message.metadata.get("sender_type"))
        # Phase 6: reply in-thread by default (a top-level message keys as its own length-1 thread).
        message.thread_id = cast(str, thread_ts or ts)
        message.metadata["channel_listen"] = True
        message.metadata["participation_level"] = level
        # F3 wake source: a name-in-text hit reads as name_mention (engine-gated or the
        # legacy deterministic wake); a thread reply we skipped the gate for — strict 1:1 or
        # membership in an `on` channel alike — as thread_continuation; anything else the engine
        # woke on is ambient. The value does NOT split by which of the two rules fired: it is
        # provenance for the wake envelope, and both routes are the same provenance. What
        # distinguishes them is `membership_wake`, stamped below for exactly that reason.
        if direct_continuation:
            message.metadata["wake_source"] = "thread_continuation"
        elif name_hit:
            message.metadata["wake_source"] = "name_mention"
        else:
            message.metadata["wake_source"] = "ambient"
        # Which rule woke us, on EVERY dispatch out of this path, true or false — a consumer must
        # never have to tell "not a membership wake" from "never stamped". Read only by the
        # structural-authorization predicate (ruling 2A).
        message.metadata["membership_wake"] = membership_wake
        # The routing facts, stamped on EVERY dispatch out of this path — including the two
        # routes that need no gate. A turn may end in silence when the gate judged it (the
        # model that woke it can also decide there is nothing to add) or when it is a thread
        # continuation (no gate ran, so the responder is the only decider); the engine-off
        # legacy name wake is a deterministic answer and owes words.
        stamp_routing_facts(message, gate_required=gate_required,
                            silence_capable=gate_required or direct_continuation,
                            addressed=False, ts=ts, thread_ts=thread_ts)
        # What this turn pins H against: for an ordinary post, the message's own ts (the listener
        # admitted exactly that).
        message.metadata["trigger_admission_ts"] = str(ts or "")
        if gate_required:
            if name_hit:
                message.metadata["participation_name_hit"] = True
            if sender_is_bot:
                message.metadata["participation_sender_bot"] = True
            # Names and types of any files, so the gate knows an artifact is attached. The
            # PIXELS no longer ride along: the binary gate does not look at images (it decides
            # whether the responder runs, and the responder is what actually reads the picture),
            # so nothing is downloaded for it and no ambient work waits on it.
            descriptors = _attachment_descriptors(event.get("files"))
            if descriptors:
                message.metadata["participation_attachments"] = descriptors
        # Whether a top-level reply is ALLOWED here (redesign Layer 1). An allowance, not a
        # mandate: where both destinations are legal the model chooses per reply
        # (set_reply_destination). Always stamped, true or false.
        message.metadata["channel_post_allowed"] = _channel_post_allowed(cs)

        # A DEBUG LOG, and nothing more. A direct continuation mints no gate attempt — the ledger
        # writes gate_start/gate_decision only when a gate runs — so these facts are not joinable
        # to any gate row and no participation.jsonl gate row will carry them. The turn's outcome
        # still lands in a `turn_outcome` row, joined to `turn_start` by turn_id; reading a live
        # pass on this route means these lines plus that row.
        self.log_debug(
            f"Channel message dispatch: channel={channel_id}, ts={ts}, level={level}, "
            f"name_hit={name_hit}, direct_continuation={direct_continuation}, "
            f"gate_required={gate_required}, membership_wake={membership_wake}, "
            f"thread_probe={thread_probe}"
        )
        if self.message_handler:
            await self.message_handler(message, self)

    async def _handle_slack_message(self, event: Dict[str, Any], client,
                                    wake_source: Optional[str] = None,
                                    origin_verified: bool = False,
                                    admission_ts: Optional[str] = None):
        """Handle a mention/DM event: build the message, run onboarding, dispatch (unchanged).

        wake_source (F3): "app_mention" or "dm" — this path is shared by both, so the
        caller (registration) tags which one so the wake envelope can tell them apart.

        admission_ts: the ts the LISTENER admitted for this event, when that is not the message's
        own ts — an edit-triggered dispatch carries the outer message_changed event_ts, so the turn
        pins H at the moment the edit arrived rather than when the message was first posted.

        origin_verified: True only when `event` came straight from Slack (the Bolt handlers in
        registration.py). It authorizes the membership attestation below. It defaults False
        because this method is ALSO the entry point for replayed/synthetic events — the
        post-onboarding welcome replay and the edit-triggered re-dispatch — where the event is
        reconstructed, possibly minutes later, and must not be treated as live proof."""

        # Skip message_changed events
        if event.get("subtype") == "message_changed":
            return

        # A message event carrying neither a `user` nor a bot identity is a Slack subtype we do
        # not act on — a deletion (message_deleted / tombstone) or an unattributed system post,
        # NOT a human turn. It must never fall through to onboarding below: with user_id=None the
        # new-user branch has no saved prefs, so it creates "default preferences for new user
        # None" and fires the Configure-Settings welcome card into whoever's DM the event landed
        # in (observed live: a deletion echo in an active DM greeted an established user).
        # classify_sender can't catch it — with no bot_id/app_id it reads as 'human'. Bot senders
        # keep their path (they carry bot_id, so classify_sender still routes them to other_bot).
        if not event.get("user") and not (event.get("bot_id") or event.get("app_id")):
            self.log_debug(
                f"Dropping unattributed message event (subtype={event.get('subtype')}, "
                f"ts={event.get('ts')}) — not a human turn")
            return

        message = await self._event_to_message(event, client)
        if wake_source:
            message.metadata["wake_source"] = wake_source
        # This path is only ever reached by a message that ADDRESSED us — a DM, a real
        # @mention, or the edit path's synthetic addressed wake — so it runs no gate and owes
        # an answer. Stamped here rather than at each of the three dispatch sites below, and
        # stamped even though three of the four facts are False: an absent fact and a False
        # one must never look the same downstream.
        stamp_routing_facts(message, gate_required=False, silence_capable=False,
                            addressed=True, ts=event.get("ts"),
                            thread_ts=event.get("thread_ts"))
        # Channel turns only: it is what a channel stream pins H against, and a DM has no stream.
        if message.channel_id and not str(message.channel_id).startswith("D"):
            message.metadata["trigger_admission_ts"] = str(admission_ts or event.get("ts") or "")
        # Explicit default. A DM has no channel settings and no top level to post at, and the
        # non-DM block below overwrites this with the resolved allowance — but every dispatched
        # message carries the key either way, so nothing downstream has to guess.
        message.metadata["channel_post_allowed"] = False
        user_id = event.get("user")

        # Phase 7: surface per-channel ground rules (in-channel only) and skip the
        # settings-modal onboarding for BOT senders — a bot can't click the modal
        # (this is the bug where the bot told Claude "configure your settings").
        sender_type = self.classify_sender(event)
        if sender_type == "self":
            return  # loop guard (also guarded upstream for DMs)
        if origin_verified:
            # Live Slack delivery of this person's message from this conversation.
            attest_message_origin(message, event, sender_type)
        if message.channel_id and not message.channel_id.startswith("D"):
            cs = await self._get_channel_settings(message.channel_id)
            # Participation "off" means OFF — the modal promises "never respond in this
            # channel", and that must include explicit @mentions (otherwise off collapses
            # into mentions_only). This path only fires for app_mention wakes: DMs have no
            # channel settings, and the channel-listening path gates itself upstream.
            if wake_source == "app_mention":
                from message_processor.participation import resolve_participation_level
                if resolve_participation_level(cs) == "off":
                    self.log_info(
                        f"Participation OFF for {message.channel_id} — dropping @mention "
                        f"(ts={event.get('ts')})")
                    return
            # B1: the mention path resolves the allowance too — without it an @mention could
            # never reply at channel level, whatever the channel's settings say. Same rule as
            # the channel-dispatch path: a row's EXPLICIT True/False wins, None/absent falls
            # back to the global default. It permits the choice; the model makes it.
            message.metadata["channel_post_allowed"] = _channel_post_allowed(cs)
        if sender_type == "other_bot":
            if self.message_handler:
                await self.message_handler(message, self)
            return

        # Assistant surface: title the split-view thread from the first user message
        # (best-effort; harmless no-op for classic DM threads and when the flag is off).
        if message.channel_id and message.channel_id.startswith("D"):
            await self._maybe_set_assistant_thread_title(
                message.channel_id, message.thread_id, message.text
            )

        # Check if this is a new user (for auto-modal trigger)
        user_prefs = await self.db.get_user_preferences_async(user_id)
        
        if not user_prefs:
            # Create default preferences for new user
            user_data = await self.db.get_or_create_user_async(user_id)
            email = user_data.get('email') if user_data else None
            user_prefs = await self.db.create_default_user_preferences_async(user_id, email)
            self.log_info(f"Created default preferences for new user {user_id}")
        
        # Check if user has completed settings
        if not user_prefs.get('settings_completed', False):
            # V3 channel-teammate: the DM-first onboarding below (public "I've DM'd you" notice +
            # withholding the answer until the settings modal is completed) must NEVER run in a
            # channel — it made sense only when a DM was the only way to reach the bot. In any
            # channel/group/MPIM we answer the mention immediately with channel + default settings,
            # and — silently, exactly once — DM the settings button so the newcomer can tune it if
            # they want. No public onboarding chrome, no blocking. DMs keep the full flow below.
            if message.channel_id and not message.channel_id.startswith('D'):
                await self._welcome_new_channel_user_via_dm(cast(str, user_id), client)
                if self.message_handler:
                    await self.message_handler(message, self)
                return

            # User hasn't completed settings - check if we've already sent welcome
            if not hasattr(self, '_welcomed_users'):
                self._welcomed_users = set()
            
            # Check if this is their first message this session
            is_first_message = user_id not in self._welcomed_users
            
            if is_first_message:
                # Mark as welcomed and send welcome button
                self._welcomed_users.add(user_id)
                
            # Check if we have a trigger_id for modal
            trigger_id = event.get('trigger_id')
            
            if trigger_id and is_first_message:
                # Create default preferences
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                default_prefs = await self.db.create_default_user_preferences_async(user_id, email)
                
                # Open welcome modal
                try:
                    modal = self.settings_modal.build_settings_modal(
                        user_id=user_id,
                        trigger_id=trigger_id,
                        current_settings=default_prefs,
                        is_new_user=True
                    )
                    
                    response = await client.views_open(
                        trigger_id=trigger_id,
                        view=modal
                    )
                    
                    if response.get('ok'):
                        self.log_info(f"Welcome modal opened for new user {user_id}")
                        
                        # Send welcome message
                        await _post_onboarding_notice(
                            self, client, site="welcome_modal_notice",
                            receipt_channel=message.channel_id,
                            thread_root_ts=message.thread_id,
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've opened your settings panel. Please configure your preferences and I'll be ready to help!"
                        )
                        return  # Don't process the message until settings are saved
                    
                except SlackApiError as e:
                    self.log_error(f"Error opening welcome modal for new user: {e}")
                    # Continue with processing using defaults
            elif is_first_message:
                # No trigger_id available, first message - send interactive message with button
                try:
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments,  # Include file attachments
                        "ts": event.get("ts")  # Include timestamp for proper threading
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": event.get("ts"),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Welcome button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Check if we're in a channel/thread vs DM
                    is_dm = message.channel_id.startswith('D')
                    
                    if is_dm:
                        # For DMs, send the button in the same conversation
                        target_channel = message.channel_id
                        target_thread = message.thread_id
                    else:
                        # For channels/threads, send as a DM to the user
                        target_channel = cast(str, user_id)  # Send to user's DM
                        target_thread = None  # No thread in DM
                        
                        # Also send a brief message in the thread to acknowledge
                        await _post_onboarding_notice(
                            self, client, site="welcome_dm_notice",
                            receipt_channel=message.channel_id,
                            thread_root_ts=message.thread_id,
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've sent you a direct message to configure your settings."
                        )
                    
                    # Send welcome button on first interaction
                    # On subsequent messages, the ephemeral will be sent from the outer check
                    
                    # Build blocks for welcome message
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "👋 *Welcome to the AI Assistant!*\n\nI need you to configure your preferences before we begin. Click the button below to open your settings:"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "⚙️ Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                    
                    response = await _post_onboarding_notice(
                        self, client, site="welcome_button",
                        receipt_channel=target_channel, thread_root_ts=target_thread,
                        channel=target_channel,
                        thread_ts=target_thread,
                        text="👋 Welcome! Please configure your settings to get started.",
                        blocks=blocks
                    )
                    
                    # Track welcome message for updating after settings saved
                    if response and response.get('ok'):
                        if not hasattr(self, '_welcome_messages'):
                            self._welcome_messages = {}
                        self._welcome_messages[user_id] = {
                            'channel': target_channel,
                            'ts': response.get('ts'),
                            'thread_ts': target_thread
                        }
                    
                    return  # Don't process until settings are configured
                except SlackApiError as e:
                    self.log_error(f"Error sending welcome message: {e}")
            else:
                # Not first message - send regular reminder that we can delete later
                try:
                    response = await _post_onboarding_notice(
                        self, client, site="settings_reminder",
                        receipt_channel=message.channel_id,
                        thread_root_ts=message.thread_id,
                        channel=message.channel_id,
                        thread_ts=message.thread_id,
                        text="⚠️ Please configure your settings before I can help you. Click the *Configure Settings* button above to get started."
                    )
                    # Track reminder message for cleanup
                    if response and response.get('ok'):
                        if not hasattr(self, '_reminder_messages'):
                            self._reminder_messages = {}
                        if user_id not in self._reminder_messages:
                            self._reminder_messages[user_id] = []
                        self._reminder_messages[user_id].append({
                            'channel': message.channel_id,
                            'ts': response.get('ts')
                        })
                except Exception as e:
                    self.log_debug(f"Could not send reminder: {e}")
                return  # Don't process until settings are configured
        else:
            # Existing user with preferences - check if this is a new thread that needs a settings button
            await self._post_settings_button_if_new_thread(message, client, user_prefs)
        
        # Call the message handler if set
        if self.message_handler:
            await self.message_handler(message, self)

    async def _welcome_new_channel_user_via_dm(self, user_id: str, client) -> None:
        """Silently DM a first-time channel user the Configure Settings button — once, ever.

        The caller has already answered the mention in-channel with default settings; this is a
        no-pressure, DM-only nudge so they CAN set personal prefs, never a channel post and never a
        gate. The "once" is a DURABLE DB claim, not an in-memory set: this bot rebuilds from scratch
        on every restart, so a session guard would re-DM the same newcomer after each deploy. The
        button value is empty (`{}`) on purpose — it carries NO original-message context, so clicking
        it opens the settings modal WITHOUT replaying the message we already answered (a replay would
        double-answer). If the send fails, we release the claim so a later interaction can retry."""
        try:
            claimed = await self.db.claim_channel_onboarding_nudge_async(user_id)
        except Exception as e:
            self.log_debug(f"Could not claim onboarding nudge for {user_id}: {e}")
            return
        if not claimed:
            return  # already nudged, or a concurrent mention won the claim

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ("👋 Thanks for the mention — I answered you back in the channel. "
                             "If you'd ever like to set your personal preferences (model, "
                             "response length, and more), you can do that here. Totally optional.")
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "⚙️ Configure Settings"},
                        "style": "primary",
                        "action_id": "open_welcome_settings",
                        "value": "{}"
                    }
                ]
            }
        ]
        try:
            # Posting to a user id routes to that user's DM with the bot.
            await client.chat_postMessage(  # DM target: structurally exempt from receipts
                channel=user_id,
                text="Set your personal preferences anytime (optional).",
                blocks=blocks
            )
        except SlackApiError as e:
            self.log_debug(f"Could not send silent settings DM to {user_id}: {e}")
            try:
                await self.db.clear_channel_onboarding_nudge_async(user_id)
            except Exception:
                pass

    async def _post_settings_button_if_new_thread(self, message: Message, client, user_prefs: dict):
        """Post a settings button at the start of a new thread"""
        try:
            # Check if this is the start of a new thread
            # For channels: thread_id != ts means it's a reply in a thread
            # For DMs: we want to check if there's any history
            
            is_dm = message.channel_id.startswith('D')
            self.log_debug(f"Checking for new thread: is_dm={is_dm}, channel={message.channel_id}, thread={message.thread_id}")
            
            # Get thread history to check if this is a new conversation
            if is_dm:
                # In DMs, every message is technically a new "thread" (unique timestamp)
                # Check if this specific thread already has messages
                history = await client.conversations_replies(
                    channel=message.channel_id,
                    ts=message.thread_id
                )
                self.log_debug(f"DM thread history check: found {len(history.get('messages', []))} messages in thread {message.thread_id}")
                
                # If there's only 1 message (the current one), it's a new thread
                is_new_thread = len(history.get('messages', [])) <= 1
            else:
                # For channels, check if this is creating a new thread
                # When thread_id == ts, it's a new thread (first message)
                is_new_thread = (message.thread_id == message.metadata.get('ts'))
            
            self.log_info(f"New thread check result: is_new_thread={is_new_thread}")
            
            if is_new_thread:
                # Check if this is a new user who hasn't completed settings
                is_new_user = not user_prefs.get('settings_completed', False)
                
                if is_new_user:
                    # New user - need to store message for later processing
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": message.metadata.get('ts'),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Full welcome message for new users
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Welcome to the AI Assistant!* :wave:\n\nI need you to configure your preferences before we can start. You can accept the defaults or customize them."
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                else:
                    # Existing user: no chrome. The old "Quick Settings Access"
                    # button per new DM thread is retired — settings are reachable
                    # via the slash command, the channel ⚙️ footer, and the
                    # Configure icon-button on DM responses.
                    return

                # Post the onboarding settings button as the first message in the thread
                await _post_onboarding_notice(
                    self, client, site="settings_button",
                    receipt_channel=message.channel_id,
                    thread_root_ts=message.thread_id,
                    channel=message.channel_id,
                    thread_ts=message.thread_id,  # Always use thread_ts to post in the thread
                    text="Settings available",
                    blocks=blocks
                )
                
        except Exception as e:
            self.log_debug(f"Could not post settings button: {e}")
            # Don't block message processing if button posting fails
