from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

from slack_sdk.errors import SlackApiError

from base_client import Message
from config import config
from slack_client.channel_pulse import pulse_supplementary_budget as _pulse_supplementary_budget
from slack_client.formatting.blocks import extract_supplementary_text


def _summarize_attachments(files: Any) -> Any:
    """F14b: compact summary of a message's files for the participation classifier —
    count + kind breakdown + filenames only, never content. Images (mimetype image/*)
    and other files are counted separately; filenames render images-first.

    Examples: "1 image (food.png)", "2 files (report.pdf, data.csv)",
    "1 image, 1 file (chart.png, notes.pdf)". Returns None when there are no files."""
    if not files:
        return None
    image_names, file_names = [], []
    for f in files:
        f = f or {}
        name = f.get("name") or "file"
        if str(f.get("mimetype", "") or "").startswith("image/"):
            image_names.append(name)
        else:
            file_names.append(name)
    parts = []
    n_img, n_file = len(image_names), len(file_names)
    if n_img:
        parts.append(f"{n_img} image" + ("s" if n_img != 1 else ""))
    if n_file:
        parts.append(f"{n_file} file" + ("s" if n_file != 1 else ""))
    if not parts:
        return None
    names = image_names + file_names
    return f"{', '.join(parts)} ({', '.join(names)})"


class SlackMessageEventsMixin:
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
        message = Message(
            text=text,
            user_id=user_id,
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
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
        """True if one of the bot's name aliases appears as a whole word (case-insensitive)."""
        if not text:
            return False
        import re
        for alias in getattr(config, "bot_name_aliases", []) or []:
            if alias and re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
                return True
        return False

    async def _thread_participation(self, channel_id: str, thread_ts: str):
        """Best-effort (bot_present, distinct_human_count, other_bot_count) for a thread.

        Lets an untagged reply count as 'for us' only in a genuinely 1:1 thread: the bot,
        at most one human, and NO other bots/agents. Another agent in the thread (e.g. a
        second assistant) means messages may be for it — only the engine can tell, so no
        deterministic continuation there. On error → (False, 0, 0)."""
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
    # churn) — excluded from the pulse. Everything else, INCLUDING bot_message and
    # ordinary content subtypes (file_share, thread_broadcast), is real awareness.
    _PULSE_FEED_SKIP_SUBTYPES = frozenset({
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

    def _ambient_service(self):
        """The AmbientArtifactService (owned by the processor), or None if not wired/available."""
        proc = getattr(self, "processor", None)
        return getattr(proc, "ambient_service", None) if proc is not None else None

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
        re-enqueue), and deletions (purge artifacts + pulse entry). Best-effort; never raises,
        never blocks the wake path (offer_event only enqueues)."""
        svc = self._ambient_service()
        if svc is None:
            return
        # The service needs the SlackBot FACADE — it owns download_file() (image/file capture)
        # and channel_pulse (late-summary patching). The Bolt `client` is a raw AsyncWebClient
        # with neither, so passing it makes every image/file job AttributeError into
        # download_failed and link summaries never patch the pulse. `self` IS that facade.
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
                    pulse = getattr(self, "channel_pulse", None)
                    if pulse is not None:
                        pulse.remove_message(channel_id, deleted_ts)
                    # A warm ThreadState may still hold the deleted message — force a rebuild.
                    self._mark_thread_refresh(channel_id, prev.get("thread_ts") or deleted_ts)
                    self.log_debug(f"message_deleted: purged {channel_id}:{deleted_ts} "
                                   f"from pulse/artifacts")
                return
            if subtype == "message_changed":
                edited = event.get("message") or {}
                new_ts = edited.get("ts")
                # Deleting a root that has (or had) replies does NOT arrive as
                # message_deleted — Slack tombstones it: message_changed whose nested
                # message carries subtype "tombstone" / the text "This message was
                # deleted." Treating that as an ordinary edit re-feeds the tombstone text
                # into the pulse as content and runs the edit-triggered engine on it (seen
                # live 2026-07-18: six tombstones dispatched, one classified — the model
                # then "remembered" threads that no longer existed). It is a deletion:
                # purge the root's pulse entry + ambient artifacts and force a thread
                # rebuild — never re-feed, never offer, never classify. The thread's
                # surviving replies keep their own pulse entries, which stays accurate:
                # Slack keeps them visible under a tombstoned root.
                if edited.get("subtype") == "tombstone" or (
                        (edited.get("text") or "").strip() == "This message was deleted."):
                    if channel_id and new_ts:
                        db = getattr(self, "db", None)
                        if db is not None:
                            try:
                                await db.delete_ambient_artifacts_by_source(channel_id, new_ts)
                            except Exception as e:
                                self.log_debug(f"ambient tombstone delete failed: {e}")
                        pulse = getattr(self, "channel_pulse", None)
                        if pulse is not None:
                            pulse.remove_message(channel_id, new_ts)
                        self._mark_thread_refresh(
                            channel_id, edited.get("thread_ts") or new_ts)
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
                    # Re-offer the edited content as a synthetic message event.
                    synthetic = dict(edited)
                    synthetic["channel"] = channel_id
                    synthetic.setdefault("ts", new_ts)
                    if not synthetic.get("thread_ts") and event.get("message", {}).get("thread_ts"):
                        synthetic["thread_ts"] = event["message"]["thread_ts"]
                    if not self.is_own_message(synthetic):
                        # Replace the stale pulse entry: drop the old text + its now-deleted
                        # artifact notes, then re-record the edited content so a warm thread
                        # doesn't keep showing the pre-edit message.
                        pulse = getattr(self, "channel_pulse", None)
                        if pulse is not None:
                            pulse.remove_message(channel_id, new_ts)
                            try:
                                await self._feed_channel_pulse(synthetic)
                            except Exception as e:  # noqa: BLE001
                                self.log_debug(f"ambient edit pulse re-feed failed: {e}")
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
            # F51b: if this message is headed into the participation wake gate, HOLD its ambient
            # image jobs so the gate's single vision look serves the stored observations too,
            # instead of the worker downloading + analyzing the same picture a second time.
            svc.offer_event(event, facade, defer_images=self._gate_will_see_images(event))
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"ambient ingest failed: {e}")

    def _gate_will_see_images(self, event: Dict[str, Any]) -> bool:
        """F51b: whether this message's ambient images should be HELD for the participation wake
        gate (which already downloads and shows them) rather than analyzed a second time by the
        vision worker.

        True only when the gate could plausibly run on this message: ambient image memory is on
        (else nothing is stored either way), channel listening + the participation engine + the
        multimodal gate are all on, it's a real channel (not a DM), it carries files, and it does
        NOT @-mention the bot — a mention is answered directly and skips the gate. This is a cheap,
        conservative predicate: a held image the gate never resolves is admitted after a bounded
        timeout, so a false positive only DELAYS analysis, and a false negative just keeps today's
        behavior (worker analyzes immediately). Never raises."""
        try:
            if not (config.enable_ambient_memory and config.enable_ambient_image_memory
                    and config.enable_channel_listening
                    and getattr(config, "enable_participation_engine", True)
                    and getattr(config, "enable_multimodal_gate", True)):
                return False
            channel_id = event.get("channel")
            if not channel_id or channel_id.startswith("D") or not event.get("files"):
                return False
            bot_user_id = getattr(self, "bot_user_id", None)
            if bot_user_id:
                from slack_client.formatting.text import text_mentions_user
                if text_mentions_user(event.get("text") or "", bot_user_id):
                    return False  # answered via app_mention; the gate won't run on this message
            return True
        except Exception:  # noqa: BLE001 — never let the hold predicate break ingest
            return False

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
                            new_text: str, already_replied: bool) -> None:
        """Stash edit context on THIS facade (which the engine's evaluate is handed as `client`),
        keyed by (channel, ts). evaluate pops it and folds old-text/already-replied into the
        classifier prompt. Bounded so a long-lived process can't accumulate stale contexts."""
        from collections import OrderedDict
        store = getattr(self, "_edit_reply_ctx_map", None)
        if store is None:
            store = OrderedDict()
            self._edit_reply_ctx_map = store
        key = f"{channel_id}|{msg_ts}"
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
                    synthetic, client, wake_source="dm" if is_dm else "app_mention")
                return
            # Otherwise: the participation engine's full judgment, carrying the edit context. The
            # marker rides the dispatched message so the queue drain keeps THIS (edit) dispatch.
            await self._dispatch_edit_to_engine(
                client, synthetic, channel_id, msg_ts, old_text, new_text, marker=marker)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"edit-triggered reply run failed: {e}")

    async def _dispatch_edit_to_engine(self, client, synthetic: Dict[str, Any], channel_id: str,
                                       msg_ts: str, old_text: str, new_text: str,
                                       marker: Optional[str] = None) -> None:
        """F52: send a non-mention channel edit through the participation engine, respecting the
        SAME gating a new message gets, and stashing the edit context so the classifier can make
        the typo-vs-meaning call. Mirrors _handle_channel_message's participation-check condition:
        the engine only judges a message a new post would also reach (judicious/active always; a
        name/mention hit under any mode). An edit that a new message wouldn't respond to stays
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
        if not (mention_present or name_hit or level in ("judicious", "active")):
            return  # mentions_only + no mention/name → silent, exactly as a new ambient message is

        # Seed the pulse (idempotent) so the classifier envelope has content, matching the wake path.
        pulse = getattr(self, "channel_pulse", None)
        if pulse is not None:
            try:
                await pulse.ensure_backfill(channel_id, self.app.client, self)
            except Exception as e:  # noqa: BLE001
                self.log_debug(f"edit engine backfill failed: {e}")

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

        # Stash the edit context where evaluate (handed this same facade as `client`) reads it.
        self._stash_edit_context(channel_id, msg_ts, old_text=old_text,
                                 new_text=new_text, already_replied=already_replied)

        message = await self._event_to_message(synthetic, client)
        message.thread_id = synthetic.get("thread_ts") or msg_ts
        message.metadata["channel_listen"] = True
        message.metadata["participation_level"] = level
        message.metadata["participation_check"] = True
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
        attach_summary = _summarize_attachments(synthetic.get("files"))
        if attach_summary:
            message.metadata["participation_attachments"] = attach_summary
        gate_images = [a for a in (message.attachments or [])
                       if (a or {}).get("type") == "image" and (a or {}).get("url")]
        if gate_images:
            message.metadata["participation_images"] = gate_images
        if cs and cs.get("directives"):
            message.metadata["channel_directives"] = cs["directives"]
        reply_in_channel = (cs or {}).get("reply_in_channel")
        if reply_in_channel is None:
            reply_in_channel = config.reply_in_channel_default
        if reply_in_channel:
            message.metadata["reply_in_channel"] = True

        self.log_debug(
            f"Edit-triggered engine dispatch: channel={channel_id}, ts={msg_ts}, level={level}, "
            f"name_hit={name_hit}, mention_present={mention_present}, "
            f"already_replied={already_replied}")
        if self.message_handler:
            await self.message_handler(message, self)

    async def _ambient_file_deleted(self, event: Dict[str, Any]) -> None:
        """F51: a Slack `file_deleted` event — purge summaries derived from that file id across
        the workspace. Best-effort; never raises."""
        db = getattr(self, "db", None)
        file_id = event.get("file_id") or (event.get("file") or {}).get("id")
        if db is None or not file_id:
            return
        try:
            await db.delete_ambient_artifacts_by_file_id(file_id)
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"ambient file_deleted purge failed for {file_id}: {e}")

    async def _feed_channel_pulse(self, event: Dict[str, Any]) -> None:
        """F5 fix (a): the single reliable semantic feed into the ambient ring buffer.
        Covers channel message events (including other apps' `bot_message` posts) and
        app_mention events; excludes edits/deletes/membership churn and our OWN posts —
        the bot's own final replies are recorded cleanly at the messaging layer
        (record_own_reply), so the echoed placeholder/footer/streamed-edit chrome must
        not leak in here. Best-effort; never blocks dispatch."""
        pulse = getattr(self, "channel_pulse", None)
        if pulse is None:
            return
        try:
            if event.get("subtype") in self._PULSE_FEED_SKIP_SUBTYPES:
                return
            if self.is_own_message(event):
                return  # recorded at the messaging layer with clean final text
            sender_type = self.classify_sender(event)
            # F48: a legacy webhook post (Jira/GitHub/Drive) has EMPTY text — its whole
            # payload is in attachments[].fields[] — so the empty-text guard below used to
            # drop it from awareness entirely. Extract first, then decide. Budgeted to the
            # pulse's own cap so the extractor's honest end marker lands INSIDE the entry
            # rather than being sliced off by record()'s head truncation.
            text = event.get("text", "") or ""
            supplementary = ""
            if sender_type != "self":
                supplementary = extract_supplementary_text(
                    event, primary_text=text, budget=_pulse_supplementary_budget(text))
            # F51: a bare file share with no comment (empty text, no supplementary) STILL carries
            # awareness — its files feed the attachment note + ambient artifacts. Cold backfill
            # records it, so dropping it live created a live/restart divergence. Record it when
            # files are present even with empty text.
            if not text.strip() and not supplementary and not event.get("files"):
                return  # nothing to add
            if supplementary:
                text = f"{text}\n\n{supplementary}" if text.strip() else supplementary
            user_id = event.get("user")
            display_name = event.get("username")
            if not display_name and user_id and user_id in self.user_cache:
                display_name = self.user_cache[user_id].get("real_name")
            pulse.record(
                event.get("channel"),
                ts=event.get("ts"),
                thread_ts=event.get("thread_ts"),
                user_id=user_id,
                display_name=display_name,
                sender_type=sender_type,
                text=text,
                is_bot=sender_type != "human",
                files=event.get("files"),
            )
        except Exception as e:
            self.log_debug(f"channel_pulse feed failed: {e}")

    async def _handle_channel_message(self, event: Dict[str, Any], client):
        """Phase 5: decide whether to respond to a NON-mention channel message, then dispatch.

        SAFE BY DEFAULT — the caller already gated on config.enable_channel_listening. Honors
        channel_response_mode (default 'tag_only'); short-circuits our own posts; de-dups against
        the app_mention event; and bypasses the welcome/settings onboarding flow entirely."""
        # Phase E / F5: awareness BEFORE any gate — feed EVERY semantic message (incl.
        # other apps' bot_message posts and ones we go on to ignore) into the pulse. The
        # feed does its own subtype/own-message filtering. This path only runs when channel
        # listening is on, so the pulse is inert otherwise by construction.
        await self._feed_channel_pulse(event)
        # Ignore non-real messages (edits, deletes, joins, message_changed, etc.) for the
        # RESPONSE gate — they never drive a reply (awareness already captured above).
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

        channel_id = event.get("channel")
        cs = await self._get_channel_settings(channel_id)
        # Phase F: participation levels (off / mentions_only / judicious / active).
        # participation_level wins over the legacy response_mode; both map cleanly
        # (off≡off, tag_only≡mentions_only, auto_respond≡judicious).
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

        # Thread replies: an untagged HUMAN reply in a genuinely 1:1 thread with the
        # bot (one human, no other bots/agents) continues that conversation
        # deterministically (cheap, and practically always right). A message from
        # another bot is never a continuation — it goes to the engine or nowhere.
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")

        sender_is_bot = self.classify_sender(event) != "human"
        direct_continuation = False
        if not sender_is_bot and thread_ts and thread_ts != ts:
            bot_present, human_count, other_bots = await self._thread_participation(channel_id, thread_ts)
            if bot_present and human_count <= 1 and other_bots == 0:
                direct_continuation = True
                # F5 fix (b): the replies fast path only scans the oldest page (limit=50)
                # and can miss a SECOND bot later in a long thread. The pulse tail holds
                # recent senders — if it shows another agent, drop the deterministic
                # continuation and let the engine judge (a bot may be the real addressee).
                pulse = getattr(self, "channel_pulse", None)
                if pulse is not None and pulse.thread_has_other_bot(channel_id, thread_ts):
                    direct_continuation = False

        # Decide (Phase F, revised): 1:1 thread continuation → respond directly;
        # judicious/active → engine judges every message; mentions_only → engine
        # judges ONLY name-bearing messages (zero model cost otherwise); engine
        # disabled → legacy deterministic name wake (humans only — a bot naming us
        # must never trigger a judgment-free reply, that's a loop seed).
        engine_on = getattr(config, "enable_participation_engine", True)
        participation_check = False
        if direct_continuation:
            pass  # respond directly
        elif engine_on and (level in ("judicious", "active") or name_hit):
            participation_check = True
        elif not engine_on and name_hit and not sender_is_bot:
            pass  # legacy deterministic name wake (engine disabled)
        else:
            return

        # Phase E: first wake in this channel since startup seeds the pulse ring
        # (one conversations.history call, once per channel per process).
        pulse = getattr(self, "channel_pulse", None)
        if pulse is not None:
            await pulse.ensure_backfill(channel_id, self.app.client, self)

        # Build the universal message (no onboarding side effects) and dispatch.
        message = await self._event_to_message(event, client)
        # Phase 6: reply in-thread by default (a top-level message keys as its own length-1 thread).
        message.thread_id = thread_ts or ts
        message.metadata["channel_listen"] = True
        message.metadata["participation_level"] = level
        # F3 wake source: a name-in-text hit reads as name_mention (engine-gated or the
        # legacy deterministic wake); a 1:1 thread reply as thread_continuation; anything
        # else the engine woke on is ambient.
        if direct_continuation:
            message.metadata["wake_source"] = "thread_continuation"
        elif name_hit:
            message.metadata["wake_source"] = "name_mention"
        else:
            message.metadata["wake_source"] = "ambient"
        if participation_check:
            message.metadata["participation_check"] = True
            if name_hit:
                message.metadata["participation_name_hit"] = True
            if sender_is_bot:
                message.metadata["participation_sender_bot"] = True
            # F14b: summarize any files so the classifier knows an artifact is attached
            # (the gate got file_share through, but text-only signals hid the image).
            attach_summary = _summarize_attachments(event.get("files"))
            if attach_summary:
                message.metadata["participation_attachments"] = attach_summary
            # F40: and hand it the IMAGES themselves — descriptors only here. Nothing is
            # downloaded until the message survives the debounce, so a superseded burst never
            # spends bandwidth on pictures whose verdict is thrown away.
            gate_images = [a for a in (message.attachments or [])
                           if (a or {}).get("type") == "image" and (a or {}).get("url")]
            if gate_images:
                message.metadata["participation_images"] = gate_images
        # Phase 7: carry per-channel ground rules + placement into the response pipeline.
        if cs and cs.get("directives"):
            message.metadata["channel_directives"] = cs["directives"]
        # reply_in_channel resolution (redesign Layer 1): a row's EXPLICIT True/False wins;
        # None (inherit) OR no row at all falls back to the global default. The old `elif` hung
        # off `if cs:`, so a channel WITH a row but a NULL reply_in_channel never reached the
        # default and was silently forced to threads-only. The engine still judges placement
        # per message when top-level replies are allowed.
        reply_in_channel = (cs or {}).get("reply_in_channel")
        if reply_in_channel is None:
            reply_in_channel = config.reply_in_channel_default
        if reply_in_channel:
            message.metadata["reply_in_channel"] = True

        self.log_debug(
            f"Channel message dispatch: channel={channel_id}, ts={ts}, level={level}, "
            f"name_hit={name_hit}, direct_continuation={direct_continuation}, "
            f"participation_check={participation_check}"
        )
        if self.message_handler:
            await self.message_handler(message, self)

    async def _handle_slack_message(self, event: Dict[str, Any], client, wake_source: str = None):
        """Handle a mention/DM event: build the message, run onboarding, dispatch (unchanged).

        wake_source (F3): "app_mention" or "dm" — this path is shared by both, so the
        caller (registration) tags which one so the wake envelope can tell them apart."""

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
        user_id = event.get("user")

        # Phase E: an @mention in a channel is also a wake — seed the pulse ring so the
        # response envelope has content. Also enabled when ambient memory is on even if channel
        # listening is off: F51 summarizes images/links/files posted in OTHER threads, and the
        # ONLY way a cross-thread ambient artifact reaches this @mention turn is the pulse
        # envelope (backfill batch-loads the artifacts). Some staleness is acceptable — a mention
        # is an explicit interaction where peripheral context helps.
        pulse = getattr(self, "channel_pulse", None)
        if (pulse is not None and message.channel_id
                and not message.channel_id.startswith("D")):
            # Channel listening populates the pulse regardless. The AMBIENT-driven widening
            # (listening off, ambient memory on) must honor the per-channel opt-out — load the
            # settings BEFORE the backfill, not after, or an opted-out channel still gets a full
            # backfill + cross-thread raw content in the mention response.
            do_backfill = config.enable_channel_listening
            if not do_backfill and config.enable_ambient_memory:
                cs_early = await self._get_channel_settings(message.channel_id)
                do_backfill = not (cs_early and cs_early.get("ambient_memory") is False)
            if do_backfill:
                # F5 fix (a): an @mention is a semantic message too — feed it (idempotent with
                # the parallel `message` event via record()'s (channel, ts) dedup).
                await self._feed_channel_pulse(event)
                await pulse.ensure_backfill(message.channel_id, self.app.client, self)

        # Phase 7: surface per-channel ground rules (in-channel only) and skip the
        # settings-modal onboarding for BOT senders — a bot can't click the modal
        # (this is the bug where the bot told Claude "configure your settings").
        sender_type = self.classify_sender(event)
        if sender_type == "self":
            return  # loop guard (also guarded upstream for DMs)
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
            if cs and cs.get("directives"):
                message.metadata["channel_directives"] = cs["directives"]
            # B1: the mention path must resolve placement too, or main.py place_in_channel
            # is always False and every @mention reply threads. Mirror the channel-dispatch
            # path (~392): a row's EXPLICIT True/False wins; None/absent falls back to the
            # global default. A mention carries no engine verdict, so a truthy setting on a
            # top-level trigger yields a top-level reply (the user summoned us at channel level).
            reply_in_channel = (cs or {}).get("reply_in_channel")
            if reply_in_channel is None:
                reply_in_channel = config.reply_in_channel_default
            if reply_in_channel:
                message.metadata["reply_in_channel"] = True
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
                        await client.chat_postMessage(
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
                        target_channel = user_id  # Send to user's DM
                        target_thread = None  # No thread in DM
                        
                        # Also send a brief message in the thread to acknowledge
                        await client.chat_postMessage(
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
                    
                    response = await client.chat_postMessage(
                        channel=target_channel,
                        thread_ts=target_thread,
                        text="👋 Welcome! Please configure your settings to get started.",
                        blocks=blocks
                    )
                    
                    # Track welcome message for updating after settings saved
                    if response.get('ok'):
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
                    response = await client.chat_postMessage(
                        channel=message.channel_id,
                        thread_ts=message.thread_id,
                        text="⚠️ Please configure your settings before I can help you. Click the *Configure Settings* button above to get started."
                    )
                    # Track reminder message for cleanup
                    if response.get('ok'):
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
                await client.chat_postMessage(
                    channel=message.channel_id,
                    thread_ts=message.thread_id,  # Always use thread_ts to post in the thread
                    text="Settings available",
                    blocks=blocks
                )
                
        except Exception as e:
            self.log_debug(f"Could not post settings button: {e}")
            # Don't block message processing if button posting fails
