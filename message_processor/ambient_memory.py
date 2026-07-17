"""F51 — ambient memory service.

"There should be nothing that is just ignored and forgotten in the running channel or thread
history... even if the bot decides not to act on it."

Images, links, and files posted in a channel/thread are looked at, summarized, and kept as
DERIVED ARTIFACTS in the running context — even when the bot stays silent. Slack remains the only
transcript: this persists summaries + refs only, never message-text mirrors, never image bytes.

Architecture (per the codex design review):
- A single `AmbientArtifactService` owned by MessageProcessor, offered every Slack `message`
  event BEFORE the channel-listening branch, so ambient content is captured even when listening
  or participation is off (those are reply settings, not memory settings).
- A non-blocking `offer_event()` that only ENQUEUES onto bounded per-kind queues. The wake path
  never awaits DNS/HTTP/vision/extraction. Overflow persists an honest `omitted/queue_overload`
  row — never a silent drop.
- Fixed worker pools per kind (fetch / vision / document). Durable pending/ready/failed/blocked
  status rows so interrupted work is visible and (for links) resumable on restart.
- Per-ref singleflight in-process + same-channel reuse of a ready summary in the DB.
- Explicit `shutdown()` drained BEFORE the OpenAI client closes.

The one hardened fetcher (ambient_fetch) also backs the model-callable `fetch_url` tool.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import ambient_fetch
from config import clamp_effort, config
from logger import setup_logger

logger = setup_logger(name="slack_bot.AmbientMemory")

KIND_IMAGE = "image"
KIND_LINK = "link"
KIND_FILE = "file"
_KINDS = (KIND_LINK, KIND_IMAGE, KIND_FILE)

# F51b — gate/ambient vision piggyback. When an ambient image rides a message that is about to go
# through the participation WAKE GATE, the gate already downloads it and shows it to the utility
# model. Rather than spend a SECOND vision call, that image's ambient vision job is HELD here and
# resolved by the gate outcome: an observation stores it (gate provenance) and drops the held job;
# no observation (blind gate, message not gated, superseded) admits the held job to the vision
# worker as normal. The hold is BOUNDED — if the gate never reports back within this window, the
# worker runs anyway, so a picture is never stranded unanalyzed.
_GATE_HOLD_SECONDS = 45.0

# Slack wraps URLs in mrkdwn as <url> or <url|label>. Also catch bare http(s) URLs.
_SLACK_LINK_RE = re.compile(r"<(https?://[^>|]+)(?:\|[^>]*)?>")
_BARE_URL_RE = re.compile(r"(?<![<\"'])\bhttps?://[^\s<>|)\]]+")

# Tracking params dropped during normalization so ?utm_source=… variants don't multiply fetches.
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref_src"}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- helpers

def extract_urls(text: str, *, limit: int) -> List[str]:
    """URLs from a message's text (mrkdwn angle form + bare), de-duped by normalized form,
    capped at `limit`. Order preserved."""
    if not text:
        return []
    found: List[str] = []
    for m in _SLACK_LINK_RE.finditer(text):
        found.append(m.group(1))
    # Bare URLs only outside the angle forms (strip angle-wrapped spans first).
    stripped = _SLACK_LINK_RE.sub(" ", text)
    for m in _BARE_URL_RE.finditer(stripped):
        found.append(m.group(0).rstrip(".,;:!?"))
    out: List[str] = []
    seen: set = set()
    for u in found:
        norm = normalize_url(u)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max(0, limit):
            break
    return out


def normalize_url(url: str) -> Optional[str]:
    """Canonical form for dedupe/reuse: lowercase scheme+host, drop fragment + tracking params,
    keep the rest of the query. Returns None for a non-http(s) or malformed URL."""
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    # `parts.hostname` strips the brackets from an IPv6 literal; re-wrap it or the reassembled
    # netloc becomes the malformed `https://2606:.../`. A ":" in the host means IPv6.
    host_render = f"[{host}]" if ":" in host else host
    netloc = host_render
    if parts.port:
        netloc = f"{host_render}:{parts.port}"
    query_pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if k.lower() not in _TRACKING_PARAMS]
    query = urlencode(query_pairs)
    path = parts.path or ""
    return urlunsplit((scheme, netloc, path, query, ""))


def sanitize_summary(text: Optional[str], *, max_chars: int) -> str:
    """Neutralize a derived summary/title/ref before it is rendered into a context line: strip
    control chars/newlines and brackets so untrusted fetched content can't forge a speaker line
    or break the bracketed note (round-2 spoof resistance)."""
    if not text:
        return ""
    s = _CONTROL_RE.sub(" ", str(text))
    s = s.replace("\n", " ").replace("\r", " ").replace("[", "(").replace("]", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def _coerce_size(size: Any) -> Optional[int]:
    """Slack's declared file size as a non-negative int, or None when it is missing or not a
    clean number. A string-valued or dishonest size must not slip past the pre-download gate."""
    if isinstance(size, bool):
        return None
    if isinstance(size, int):
        return size if size >= 0 else None
    try:
        v = int(str(size).strip())
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def _expiry(days: int) -> Optional[str]:
    if days <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _fresh_after(days: int) -> Optional[str]:
    if days <= 0:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def render_artifact_note(art: Dict[str, Any], *, max_chars: int = 400) -> str:
    """Deterministic, sanitized inline note for a READY artifact — the same string used in the
    pulse and in thread-history context. Contains NO volatile fetched_at text (cache stability)."""
    kind = art.get("kind")
    title = sanitize_summary(art.get("title"), max_chars=120)
    summary = sanitize_summary(art.get("summary"), max_chars=max_chars)
    src = art.get("derivation_source")
    if kind == KIND_IMAGE:
        return f"[image (analyzed): {summary}]" if summary else "[image: not analyzed]"
    if kind == KIND_LINK:
        tag = "link content" if src != "unfurl" else "link preview"
        head = f"{title} — " if title else ""
        return f"[{tag}: {head}{summary}]" if (summary or title) else "[link: not readable]"
    if kind == KIND_FILE:
        head = f"{title}: " if title else ""
        return f"[file (summarized): {head}{summary}]" if summary else "[file: not summarized]"
    return ""


# --------------------------------------------------------------------------- jobs

@dataclass
class _Job:
    kind: str
    channel_id: str
    source_ts: str
    conversation_ts: str
    ref: str                       # normalized url (link) or Slack file id (image/file)
    url: Optional[str] = None      # download url for image/file; == ref for links
    filename: Optional[str] = None
    mimetype: Optional[str] = None
    size: Optional[int] = None
    unfurls: List[Dict[str, str]] = field(default_factory=list)

    def key(self) -> Tuple[str, str, str, str]:
        return (self.channel_id, self.source_ts, self.kind, self.ref)


# --------------------------------------------------------------------------- service

class AmbientArtifactService:
    """Bounded-queue ingestion service for ambient artifacts. Loop-affine (single asyncio loop);
    all shared state is plain dict/set touched only on that loop."""

    def __init__(self, *, db, openai_client, channel_pulse=None, cfg=config):
        self.db = db
        self.openai_client = openai_client
        self.channel_pulse = channel_pulse
        self.config = cfg
        self._client = None                      # Slack client, captured at first offer
        self._queues: Dict[str, asyncio.Queue] = {}
        self._workers: List[asyncio.Task] = []
        self._inflight: set = set()
        # F51b: image jobs HELD for the participation gate to resolve. Keyed by _Job.key()
        # (channel_id, source_ts, kind, ref) -> {"job": _Job, "timer": Task}. A held key also sits
        # in _inflight (singleflight), so a duplicate offer of the same upload — the parallel
        # app_mention + message events — never double-holds or double-admits it.
        self._deferred: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        # Fire-and-forget persistence tasks (durable claims, overflow rows). TRACKED so shutdown
        # can drain them — an untracked create_task can be GC'd or lost on shutdown, dropping the
        # very honest omitted/queue_overload row it was supposed to persist.
        self._bg_tasks: set = set()
        self._started = False
        self._closing = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Create per-kind bounded queues + fixed worker pools on the running loop. Idempotent."""
        if self._started:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; offer_event will start lazily
        cap = max(1, int(self.config.ambient_queue_capacity))
        counts = {
            KIND_LINK: max(1, int(self.config.ambient_fetch_workers)),
            KIND_IMAGE: max(1, int(self.config.ambient_vision_workers)),
            KIND_FILE: max(1, int(self.config.ambient_document_workers)),
        }
        for kind in _KINDS:
            self._queues[kind] = asyncio.Queue(maxsize=cap)
            for _ in range(counts[kind]):
                self._workers.append(asyncio.create_task(self._worker(kind)))
        self._started = True
        logger.info(f"AmbientArtifactService started (cap={cap}, workers={counts})")

    async def recover_pending(self) -> None:
        """Restart recovery: re-enqueue interrupted link fetches (fully recoverable from ref=url);
        non-recoverable image/file pending rows (their download url wasn't persisted) are marked
        `failed/interrupted` — honest, never a silent zombie."""
        if not self.db:
            return
        try:
            rows = await self.db.get_pending_ambient_artifacts(limit=500)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ambient recover_pending failed: {e}")
            return
        for r in rows:
            kind = r.get("kind")
            # A channel that opted out AFTER these rows were claimed must not be re-processed —
            # and the row must be RETIRED (omitted) so it isn't recovered on every restart forever.
            if await self._channel_opted_out(r["channel_id"]):
                try:
                    await self.db.set_ambient_artifact_status(
                        channel_id=r["channel_id"], source_ts=r["source_ts"], kind=kind,
                        ref=r["ref"], status="omitted", error_code="opted_out")
                except Exception:  # noqa: BLE001
                    pass
                continue
            if kind == KIND_LINK:
                job = _Job(kind=KIND_LINK, channel_id=r["channel_id"], source_ts=r["source_ts"],
                           conversation_ts=r["conversation_ts"], ref=r["ref"], url=r["ref"])
                self._enqueue_recovered(job)
            else:
                try:
                    await self.db.set_ambient_artifact_status(
                        channel_id=r["channel_id"], source_ts=r["source_ts"], kind=kind,
                        ref=r["ref"], status="failed", error_code="interrupted")
                except Exception:  # noqa: BLE001
                    pass

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Drain then cancel workers. MUST run before the OpenAI client closes (workers call it)."""
        self._closing = True
        # F51b: a job HELD for the gate at shutdown has no durable row yet (nothing is persisted
        # while held). Left as-is it vanishes with no record — recover_pending has nothing to find,
        # so a picture held during the 45s window is permanently absent after restart. Persist a
        # durable pending CLAIM for each held job BEFORE draining so recover_pending finds it after
        # restart (links resume; image claims become honest failed/interrupted rows — a visible,
        # recoverable state, never a silent drop). Pop the entry FIRST so a hold timer that fires
        # during the claim's await sees an empty _deferred and returns without admitting; cancel the
        # timer only AFTER the claim commits so the two paths can never both act on one key.
        for key in list(self._deferred):
            entry = self._deferred.pop(key, None)
            if entry is None:
                continue
            job = entry.get("job")
            if job is not None:
                # F51e: a locked/slow DB (or many held jobs) must not blow past `timeout` — bound
                # each persist to 2s; a timeout counts as a persist-failure. And `_persist_claim`
                # returns False on any persistence error: honoring that bool is the whole point.
                persisted = False
                try:
                    persisted = await asyncio.wait_for(self._persist_claim(job), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"ambient shutdown: claim persist timed out for "
                        f"{job.channel_id}:{job.source_ts} ({job.kind} {job.ref})")
                except Exception:  # noqa: BLE001 — shutdown must drain regardless
                    pass
                if not persisted:
                    # No durable row committed. Hand the held job to its worker queue so the drain
                    # below may still process it before workers are cancelled; log either way so a
                    # loss is never silent.
                    q = self._queues.get(job.kind)
                    try:
                        if q is not None:
                            q.put_nowait(job)
                            logger.warning(
                                f"ambient shutdown: claim not durable for "
                                f"{job.channel_id}:{job.source_ts} ({job.kind} {job.ref}); "
                                f"enqueued for drain")
                        else:
                            logger.warning(
                                f"ambient shutdown: claim not durable and no queue for "
                                f"{job.channel_id}:{job.source_ts} ({job.kind} {job.ref}); LOST")
                    except asyncio.QueueFull:
                        logger.warning(
                            f"ambient shutdown: claim not durable and queue full for "
                            f"{job.channel_id}:{job.source_ts} ({job.kind} {job.ref}); LOST")
            timer = entry.get("timer")
            if timer is not None and not timer.done():
                timer.cancel()
            self._inflight.discard(key)
        if not self._workers and not self._bg_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(q.join() for q in self._queues.values())), timeout=timeout)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — cancel regardless
            pass
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        # Let the tracked persistence tasks (durable claims / overflow rows) finish so an
        # honest omitted/failed row is never lost to shutdown; bounded by the same timeout.
        if self._bg_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(self._bg_tasks), return_exceptions=True), timeout=timeout)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        self._started = False
        logger.info("AmbientArtifactService shut down")

    # -- ingest --------------------------------------------------------------

    def offer_event(self, event: Dict[str, Any], client: Any, *,
                    defer_images: bool = False) -> None:
        """Non-blocking: parse the event into jobs and admit them. Never awaits, never raises.

        Admission (per-channel opt-out check + durable claim + enqueue) runs OFF the wake path in
        a scheduled task — the wake path only reserves the singleflight slot and returns.

        F51b: when `defer_images` is set (the ingest seam judged this message headed into the
        participation gate), IMAGE jobs are HELD for the gate to resolve instead of admitted
        straight to the vision worker — see `_defer_image` / `resolve_gate`. Links and files are
        unaffected; a held image is never dropped (a bounded timer admits it if the gate is
        silent)."""
        try:
            if not self.config.enable_ambient_memory:
                return
            self._client = client or self._client
            if self.channel_pulse is None and client is not None:
                self.channel_pulse = getattr(client, "channel_pulse", None)
            if not self._started:
                self.start()
                if not self._started:
                    return
            jobs = self._jobs_from_event(event)
            for job in jobs:
                if defer_images and job.kind == KIND_IMAGE:
                    self._defer_image(job)
                else:
                    self._admit(job)
        except Exception as e:  # noqa: BLE001 — ingestion must never break the wake path
            logger.debug(f"ambient offer_event failed: {e}")

    def _jobs_from_event(self, event: Dict[str, Any]) -> List[_Job]:
        channel_id = event.get("channel")
        source_ts = event.get("ts")
        if not channel_id or not source_ts:
            return []
        conversation_ts = event.get("thread_ts") or source_ts
        jobs: List[_Job] = []
        # Links from message text.
        if self.config.enable_link_fetch:
            from slack_client.formatting.blocks import extract_unfurls
            unfurls = extract_unfurls(event)
            for url in extract_urls(event.get("text") or "",
                                    limit=int(self.config.ambient_max_links_per_message)):
                jobs.append(_Job(kind=KIND_LINK, channel_id=channel_id, source_ts=source_ts,
                                 conversation_ts=conversation_ts, ref=url, url=url,
                                 unfurls=unfurls))
        # Files → images vs documents.
        files = event.get("files") or []
        img_cap = int(self.config.ambient_max_images_per_message)
        file_cap = int(self.config.ambient_max_files_per_message)
        n_img = n_file = 0
        for f in files:
            f = f or {}
            fid = f.get("id")
            file_url = f.get("url_private")
            mimetype = str(f.get("mimetype") or "").lower()
            if not fid or not file_url:
                continue
            if mimetype.startswith("image/"):
                if not self.config.enable_ambient_image_memory or n_img >= img_cap:
                    if n_img >= img_cap:
                        self._offer_overflow(channel_id, source_ts, conversation_ts, KIND_IMAGE, fid)
                    continue
                n_img += 1
                jobs.append(_Job(kind=KIND_IMAGE, channel_id=channel_id, source_ts=source_ts,
                                 conversation_ts=conversation_ts, ref=fid, url=file_url,
                                 filename=f.get("name"), mimetype=mimetype, size=f.get("size")))
            else:
                if not self.config.enable_ambient_file_memory or n_file >= file_cap:
                    if n_file >= file_cap:
                        self._offer_overflow(channel_id, source_ts, conversation_ts, KIND_FILE, fid)
                    continue
                n_file += 1
                jobs.append(_Job(kind=KIND_FILE, channel_id=channel_id, source_ts=source_ts,
                                 conversation_ts=conversation_ts, ref=fid, url=file_url,
                                 filename=f.get("name"), mimetype=mimetype, size=f.get("size")))
        return jobs

    def _admit(self, job: _Job) -> None:
        """Reserve the singleflight slot SYNCHRONOUSLY (wake path stays non-blocking) then admit
        asynchronously off the wake path. Synchronous reservation closes the race where two rapid
        offers of the same ref both pass the in-flight check across an await."""
        key = job.key()
        if key in self._inflight:
            return  # in-process singleflight
        self._inflight.add(key)
        self._schedule(self._admit_async(job))

    async def _admit_async(self, job: _Job) -> None:
        """Off the wake path: honor the per-channel opt-out FIRST (an opted-out channel persists
        NOTHING and enqueues nothing — no pending rows to recover forever), then commit the durable
        pending claim BEFORE the job is handed to a worker. So every job accepted onto a queue has
        a committed row; a crash while queued is recoverable. The pre-claim window (offered but not
        yet admitted) loses the job, but it was never accepted — an honest, documented boundary."""
        key = job.key()
        enqueued = False
        try:
            if await self._channel_opted_out(job.channel_id):
                return
            if not await self._persist_claim(job):  # no committed claim -> not accepted
                return
            q = self._queues.get(job.kind)
            if q is None:
                return
            try:
                q.put_nowait(job)
                enqueued = True                     # worker's finally will release the slot
            except asyncio.QueueFull:
                await self._persist_overflow(job)
        except Exception as e:  # noqa: BLE001 — admission must never break anything
            logger.debug(f"ambient admit failed: {e}")
        finally:
            if not enqueued:
                self._inflight.discard(key)         # no worker will run to release it

    def _enqueue_recovered(self, job: _Job) -> None:
        """Re-enqueue a row recovered from the DB. Its pending claim already exists and opt-out was
        already checked by recover_pending, so this only reserves + puts (no re-persist)."""
        key = job.key()
        if key in self._inflight:
            return
        q = self._queues.get(job.kind)
        if q is None:
            return
        try:
            q.put_nowait(job)
            self._inflight.add(key)
        except asyncio.QueueFull:
            pass  # stays pending in the DB; a later recovery retries

    def _offer_overflow(self, channel_id, source_ts, conversation_ts, kind, ref) -> None:
        job = _Job(kind=kind, channel_id=channel_id, source_ts=source_ts,
                   conversation_ts=conversation_ts, ref=ref)
        self._schedule(self._persist_overflow(job))

    async def _persist_overflow(self, job: _Job) -> None:
        if not self.db:
            return
        # Opted-out channels record nothing — not even an overflow row (this covers the
        # per-message image/file cap path, which does not go through _admit_async).
        if await self._channel_opted_out(job.channel_id):
            return
        try:
            await self.db.insert_pending_ambient_artifact(
                channel_id=job.channel_id, source_ts=job.source_ts,
                conversation_ts=job.conversation_ts, kind=job.kind, ref=job.ref)
            await self.db.set_ambient_artifact_status(
                channel_id=job.channel_id, source_ts=job.source_ts, kind=job.kind,
                ref=job.ref, status="omitted", error_code="queue_overload")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient overflow persist failed: {e}")

    def _schedule(self, coro) -> None:
        self._spawn(coro)

    def _spawn(self, coro) -> Optional["asyncio.Task"]:
        """Schedule a tracked background task, or None outside a running loop (sync/test context —
        the coroutine is closed so it doesn't leak un-awaited). Returns the task so a caller that
        must later cancel it (the gate-hold timer) can hold the handle."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return None
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # -- gate/ambient piggyback (F51b) ---------------------------------------

    def _defer_image(self, job: _Job) -> None:
        """Hold an ambient IMAGE job while its message goes through the participation gate.

        Reserves the singleflight slot SYNCHRONOUSLY (like _admit) so a duplicate offer of the same
        upload doesn't double-hold or double-admit, and starts a bounded timer that admits the job
        if the gate never reports back. Nothing durable is persisted while held — a held-but-
        unresolved job that a crash/shutdown loses was never accepted (the same honest boundary as
        the pre-claim window); the gate resolves the common case in a few seconds."""
        key = job.key()
        if key in self._inflight or key in self._deferred:
            return
        self._inflight.add(key)
        timer = self._spawn(self._gate_hold_timeout(key))
        if timer is None:
            # No running loop to hold or time out the job (sync/test context) — admit normally
            # rather than strand it held forever.
            self._inflight.discard(key)
            self._admit(job)
            return
        self._deferred[key] = {"job": job, "timer": timer}

    async def _gate_hold_timeout(self, key: Tuple[str, str, str, str]) -> None:
        """The gate never reported back within the hold window → admit the held job so the vision
        worker analyzes it as normal. Cancelled cleanly when resolve_gate gets there first."""
        try:
            await asyncio.sleep(_GATE_HOLD_SECONDS)
        except asyncio.CancelledError:
            return  # resolved/released before the window closed
        entry = self._deferred.pop(key, None)
        if entry is None:
            return
        self._inflight.discard(key)
        self._admit(entry["job"])

    def resolve_gate(self, channel_id: str, source_ts: str,
                     observations: Dict[str, str]) -> None:
        """The participation gate has classified `source_ts` — resolve its held image jobs.

        For each IMAGE held for this message: a matching observation STORES it as the ambient
        artifact (gate provenance, no second vision call) and drops the held job; no observation
        ADMITS the held job to the vision worker so the image is still analyzed exactly once.
        Non-blocking and never raises — the caller returns the verdict without waiting on this."""
        try:
            keys = [k for k in list(self._deferred)
                    if k[0] == channel_id and k[1] == source_ts and k[2] == KIND_IMAGE]
            for key in keys:
                entry = self._deferred.pop(key, None)
                if entry is None:
                    continue
                timer = entry.get("timer")
                if timer is not None and not timer.done():
                    timer.cancel()
                job = entry["job"]
                text = (observations or {}).get(job.ref)
                if text and text.strip():
                    # Keep the singleflight slot reserved across the async store (released in its
                    # finally) so a late duplicate offer can't admit a second copy meanwhile.
                    if self._spawn(self._store_gate_observation(job, text.strip())) is None:
                        self._inflight.discard(key)  # no loop to run the store's finally
                else:
                    self._inflight.discard(key)
                    self._admit(job)
        except Exception as e:  # noqa: BLE001 — piggyback must never break the gate
            logger.debug(f"resolve_gate failed: {e}")

    async def _store_gate_observation(self, job: _Job, observation: str) -> None:
        """Persist a gate-produced image observation as the ambient artifact — the SAME shape the
        vision worker writes (ready row + image-ledger dual-write + pulse note), differing only in
        `derivation_source='gate_vision'` and the model that produced it, and in that no second
        vision call was spent. Obeys the per-channel opt-out exactly like the worker; a row another
        writer already made `ready` (the worker won a race) is left untouched — the gate result is
        simply discarded.

        On ANY storage failure (e.g. a transient SQLite lock in the insert or in `_ready`) the held
        image would otherwise be dropped entirely: resolve_gate already cancelled its hold timer, so
        no worker job is queued. Instead the job is RELEASED to the ordinary vision-worker admission
        path — the same release resolve_gate uses for a no-observation message — so the picture is
        still analyzed exactly once. Re-admission is idempotent: if `_ready` had already committed
        the ready row before failing, the worker sees `status == 'ready'` and skips a second call."""
        key = job.key()
        released = False
        try:
            if await self._channel_opted_out(job.channel_id):
                return  # opted-out channel persists nothing (parity with the worker)
            row = await self.db.insert_pending_ambient_artifact(
                channel_id=job.channel_id, source_ts=job.source_ts,
                conversation_ts=job.conversation_ts, kind=KIND_IMAGE, ref=job.ref,
                content_type=job.mimetype, derivation_source="gate_vision",
                expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
            if row and row.get("status") == "ready":
                return  # worker (or a prior store) already won — discard the gate result
            await self._ready(job, KIND_IMAGE, title=job.filename, summary=observation,
                              model=self.config.utility_model, derivation_source="gate_vision",
                              content_type=job.mimetype)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"gate observation store failed for {job.ref}: {e}")
            # Release the held job to the normal worker path instead of dropping it. Discard the
            # singleflight slot FIRST so _admit re-reserves cleanly (release once, no double-drop).
            self._inflight.discard(key)
            released = True
            self._admit(job)
        finally:
            if not released:
                self._inflight.discard(key)

    async def _persist_claim(self, job: _Job) -> bool:
        """Persist a pending row the moment a job is durably accepted onto a queue, so a crash
        with jobs still queued leaves recoverable rows (links resume; image/file claims become
        honest interrupted rows) rather than nothing. Idempotent with the worker's own insert.

        Returns True only when the claim is durably committed (or there is no DB, i.e. an
        in-memory-only context that promises no durability). A False return means the job must
        NOT be enqueued — accepting work whose claim failed to commit would silently void the
        crash-recovery guarantee the claim exists to provide."""
        if not self.db:
            return True
        try:
            await self.db.insert_pending_ambient_artifact(
                channel_id=job.channel_id, source_ts=job.source_ts,
                conversation_ts=job.conversation_ts, kind=job.kind, ref=job.ref,
                content_type=job.mimetype,
                expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ambient claim persist failed; job not accepted: {e}")
            return False

    # -- workers -------------------------------------------------------------

    async def _worker(self, kind: str) -> None:
        q = self._queues[kind]
        while True:
            job = await q.get()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — one bad job never kills the pool
                logger.warning(f"ambient {kind} job failed: {e}")
            finally:
                self._inflight.discard(job.key())
                q.task_done()

    async def _process(self, job: _Job) -> None:
        # Per-channel opt-out (participation `off` is NOT memory-off).
        if await self._channel_opted_out(job.channel_id):
            return
        if job.kind == KIND_LINK:
            await self._process_link(job)
        elif job.kind == KIND_IMAGE:
            await self._process_image(job)
        elif job.kind == KIND_FILE:
            await self._process_file(job)

    async def _channel_opted_out(self, channel_id: str) -> bool:
        if not self.db or not channel_id or channel_id.startswith("D"):
            return False
        try:
            cs = await self.db.get_channel_settings_async(channel_id)
        except Exception:  # noqa: BLE001
            return False
        if cs and cs.get("ambient_memory") is False:
            return True
        return False

    # -- link ----------------------------------------------------------------

    async def _process_link(self, job: _Job) -> None:
        # Same-channel reuse of a fresh ready summary (no re-fetch).
        reuse = await self.db.find_reusable_ambient_summary(
            job.channel_id, KIND_LINK, job.ref,
            fresh_after=_fresh_after(int(self.config.ambient_link_stale_days)))
        row = await self.db.insert_pending_ambient_artifact(
            channel_id=job.channel_id, source_ts=job.source_ts,
            conversation_ts=job.conversation_ts, kind=KIND_LINK, ref=job.ref,
            derivation_source="fetch", expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
        if row and row.get("status") == "ready":
            return  # this occurrence already summarized
        if reuse and reuse.get("summary"):
            await self._ready(job, KIND_LINK, title=reuse.get("title"), summary=reuse["summary"],
                              model=reuse.get("model"), derivation_source="fetch",
                              content_type=reuse.get("content_type"))
            return
        result = await ambient_fetch.fetch_url(
            job.url or job.ref,
            max_bytes=int(self.config.link_fetch_max_bytes),
            connect_timeout=float(self.config.link_fetch_connect_timeout_s),
            read_timeout=float(self.config.link_fetch_read_timeout_s),
            total_timeout=float(self.config.link_fetch_total_timeout_s),
            max_redirects=int(self.config.link_fetch_max_redirects),
            dns_timeout=float(self.config.link_fetch_connect_timeout_s),
            max_chars=int(self.config.ambient_extract_max_chars))
        if result.kind == "image":
            # A direct image URL: summarize it through the vision path, still kind=link ref=url.
            await self._summarize_image_bytes(job, result.raw_bytes, result.content_type,
                                              kind=KIND_LINK, derivation_source="vision_worker")
            return
        if result.kind == "text" and result.text:
            summary = await self._summarize_text(result.text, link=True)
            if summary:
                await self._ready(job, KIND_LINK, title=result.title, summary=summary,
                                  model=self.config.utility_model, derivation_source="fetch",
                                  content_type=result.content_type)
                return
            await self._fail(job, KIND_LINK, ambient_fetch.ERR_EXTRACT_FAILED,
                             derivation_source="fetch")
            return
        # Fetch failed → unfurl fallback ONLY when the preview URL matches this link.
        fallback = self._matching_unfurl(job)
        if fallback:
            summary = sanitize_summary(fallback.get("text"),
                                       max_chars=int(self.config.ambient_summary_max_chars))
            title = fallback.get("title")
            if summary or title:
                await self._ready(job, KIND_LINK, title=title, summary=summary or (title or ""),
                                  model=None, derivation_source="unfurl")
                return
        status = "blocked" if result.error_code == ambient_fetch.ERR_BLOCKED_SSRF else "failed"
        await self._fail(job, KIND_LINK, result.error_code or "failed",
                         status=status, derivation_source="fetch")

    def _matching_unfurl(self, job: _Job) -> Optional[Dict[str, str]]:
        for u in job.unfurls or []:
            if normalize_url(u.get("url") or "") == job.ref:
                return u
        return None

    # -- image ---------------------------------------------------------------

    async def _process_image(self, job: _Job) -> None:
        reuse = await self.db.find_reusable_ambient_summary(job.channel_id, KIND_IMAGE, job.ref)
        # Addressed-turn reuse: if the upload was already analyzed into the image catalog for this
        # thread (catalog_uploads), reuse that description and suppress the second vision call.
        if not reuse:
            reuse = await self._reuse_from_image_catalog(job)
        row = await self.db.insert_pending_ambient_artifact(
            channel_id=job.channel_id, source_ts=job.source_ts,
            conversation_ts=job.conversation_ts, kind=KIND_IMAGE, ref=job.ref,
            content_type=job.mimetype, derivation_source="vision_worker",
            expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
        if row and row.get("status") == "ready":
            return
        if reuse and reuse.get("summary"):
            await self._ready(job, KIND_IMAGE, title=job.filename, summary=reuse["summary"],
                              model=reuse.get("model"), derivation_source="vision_worker",
                              content_type=job.mimetype)
            return
        # Declared-size pre-gate (parity with the file worker): reject an honestly-oversized image
        # BEFORE downloading. Missing/dishonest sizes fall through to the streamed download cap.
        declared = _coerce_size(job.size)
        if declared is not None and declared > int(self.config.ambient_file_max_bytes):
            await self._fail(job, KIND_IMAGE, ambient_fetch.ERR_TOO_LARGE, status="omitted",
                             derivation_source="vision_worker")
            return
        raw = await self._download(job)
        if raw is None:
            # None ← download failure OR the streamed byte cap tripped (dishonest/missing size).
            await self._fail(job, KIND_IMAGE, "download_failed", derivation_source="vision_worker")
            return
        # Post-download ceiling: redundant with the streamed cap, kept as a belt-and-suspenders
        # guard for any download path that doesn't honor max_bytes.
        if len(raw) > int(self.config.ambient_file_max_bytes):
            await self._fail(job, KIND_IMAGE, ambient_fetch.ERR_TOO_LARGE, status="omitted",
                             derivation_source="vision_worker")
            return
        await self._summarize_image_bytes(job, raw, job.mimetype, kind=KIND_IMAGE,
                                          derivation_source="vision_worker")

    async def _reuse_from_image_catalog(self, job: _Job) -> Optional[Dict[str, Any]]:
        """If an addressed turn already analyzed this exact upload (same url in this thread), reuse
        that analysis instead of re-downloading and re-calling vision."""
        if not job.url:
            return None
        try:
            thread_key = f"{job.channel_id}:{job.conversation_ts}"
            rows = await self.db.find_thread_images_async(thread_key)
        except Exception:  # noqa: BLE001
            return None
        for r in rows or []:
            if r.get("url") == job.url and (r.get("analysis") or "").strip():
                return {"summary": r["analysis"], "model": None}
        return None

    async def _summarize_image_bytes(self, job: _Job, raw: Optional[bytes],
                                     mimetype: Optional[str], *, kind: str,
                                     derivation_source: str) -> None:
        if not raw:
            await self._fail(job, kind, "download_failed", derivation_source=derivation_source)
            return
        # PARSE the bytes, don't just sniff a magic prefix: a payload of "PNG signature + junk"
        # sails past a prefix match and then 400s the vision call. ensure_api_compatible runs the
        # Pillow verify/load path, returns the canonical mimetype the API will accept, and
        # transcodes a decodable-but-unsupported format (BMP, TIFF, ...) to PNG in memory (F50b).
        from image_validation import ensure_api_compatible
        raw, mime = ensure_api_compatible(raw)
        if not raw:
            await self._fail(job, kind, ambient_fetch.ERR_UNSUPPORTED_TYPE,
                             derivation_source=derivation_source)
            return
        # Ambient vision runs on the UTILITY model at utility effort (an image the bot never
        # even answered isn't worth primary-model spend); record the model that ACTUALLY ran.
        model = self.config.utility_model
        try:
            import base64
            data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            from prompts import IMAGE_ANALYSIS_PROMPT
            description = await self.openai_client.analyze_images(
                images=[{"type": "input_image", "image_url": data_url, "detail": "low"}],
                question=IMAGE_ANALYSIS_PROMPT, enhance_prompt=False,
                model=model,
                reasoning_effort=clamp_effort(model, self.config.utility_reasoning_effort),
                verbosity=self.config.utility_verbosity)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient vision failed for {job.ref}: {e}")
            await self._fail(job, kind, "vision_failed", derivation_source=derivation_source)
            return
        if not description:
            await self._fail(job, kind, "vision_failed", derivation_source=derivation_source)
            return
        summary = sanitize_summary(description, max_chars=int(self.config.ambient_summary_max_chars))
        await self._ready(job, kind, title=job.filename, summary=summary,
                          model=model, derivation_source=derivation_source,
                          content_type=mime)

    # -- file ----------------------------------------------------------------

    async def _process_file(self, job: _Job) -> None:
        row = await self.db.insert_pending_ambient_artifact(
            channel_id=job.channel_id, source_ts=job.source_ts,
            conversation_ts=job.conversation_ts, kind=KIND_FILE, ref=job.ref,
            content_type=job.mimetype, derivation_source="document",
            expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
        if row and row.get("status") == "ready":
            return
        # Hard pre-download size gate from Slack's declared size (existing Slack downloads buffer
        # the whole body, so "reuse document caps" is NOT a memory cap). Coerce the declared size
        # to int first — Slack may send it as a string, and `isinstance(_, int)` would then let a
        # dishonestly-labeled large file through the full 50MB download path.
        declared = _coerce_size(job.size)
        if declared is not None and declared > int(self.config.ambient_file_max_bytes):
            await self._fail(job, KIND_FILE, ambient_fetch.ERR_TOO_LARGE, status="omitted",
                             derivation_source="document")
            return
        raw = await self._download(job)
        if raw is None:
            await self._fail(job, KIND_FILE, "download_failed", derivation_source="document")
            return
        if len(raw) > int(self.config.ambient_file_max_bytes):
            await self._fail(job, KIND_FILE, ambient_fetch.ERR_TOO_LARGE, status="omitted",
                             derivation_source="document")
            return
        text = await self._extract_document(job, raw)
        if not text:
            await self._fail(job, KIND_FILE, ambient_fetch.ERR_EXTRACT_FAILED,
                             derivation_source="document")
            return
        summary = await self._summarize_text(text, link=False)
        if not summary:
            await self._fail(job, KIND_FILE, "summarize_failed", derivation_source="document")
            return
        await self._ready(job, KIND_FILE, title=job.filename, summary=summary,
                          model=self.config.utility_model, derivation_source="document",
                          content_type=job.mimetype)

    async def _extract_document(self, job: _Job, raw: bytes) -> Optional[str]:
        """Bounded in-memory extraction via the shared DocumentHandler. No ambient OCR by default
        (ocr_images/ocr_text off) — OCR is subprocess+CPU heavy and not worth it for a message the
        bot didn't even answer."""
        handler = getattr(self, "_document_handler", None)
        if handler is None:
            try:
                from document_handler import DocumentHandler
                handler = self._document_handler = DocumentHandler()
            except Exception:  # noqa: BLE001
                return None
        if not handler.is_document_file(job.filename or "", job.mimetype):
            return None
        try:
            result = await handler.safe_extract_content_async(
                raw, job.mimetype or "application/octet-stream", job.filename or "file",
                ocr_images=False, ocr_text=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient document extract failed for {job.ref}: {e}")
            return None
        text = ""
        if isinstance(result, dict):
            text = result.get("content") or result.get("text") or ""
        elif isinstance(result, str):
            text = result
        text = (text or "").strip()
        return text[:int(self.config.ambient_extract_max_chars)] or None

    # -- shared --------------------------------------------------------------

    async def _download(self, job: _Job) -> Optional[bytes]:
        client = self._client
        if client is None or not job.url:
            return None
        try:
            # Streamed cap: the ambient ceiling is enforced DURING download (stop at limit+1),
            # not after buffering a whole body — a missing/dishonest declared size can't blow
            # memory. Returns None when the cap is hit (treated as too_large by the caller).
            return await client.download_file(
                job.url, job.ref, max_bytes=int(self.config.ambient_file_max_bytes))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient download failed for {job.ref}: {e}")
            return None

    async def _summarize_text(self, text: str, *, link: bool) -> Optional[str]:
        """Utility-model summary. Passes utility reasoning/verbosity EXPLICITLY — create_text_response
        falls back to DEFAULT (not utility) settings when omitted."""
        text = (text or "").strip()
        if not text:
            return None
        from prompts import AMBIENT_FILE_SUMMARY_PROMPT, AMBIENT_LINK_SUMMARY_PROMPT
        prompt = AMBIENT_LINK_SUMMARY_PROMPT if link else AMBIENT_FILE_SUMMARY_PROMPT
        capped = text[:int(self.config.ambient_extract_max_chars)]
        try:
            out = await self.openai_client.create_text_response(
                messages=[{"role": "user",
                           "content": f"{prompt}\n\n<<<UNTRUSTED EXTERNAL CONTENT>>>\n{capped}"}],
                model=self.config.utility_model,
                reasoning_effort=clamp_effort(self.config.utility_model,
                                              self.config.utility_reasoning_effort),
                verbosity=self.config.utility_verbosity,
                max_tokens=max(256, int(self.config.utility_max_tokens)),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient summarize failed: {e}")
            return None
        return sanitize_summary(out, max_chars=int(self.config.ambient_summary_max_chars)) or None

    async def _ready(self, job: _Job, kind: str, *, title, summary, model,
                     derivation_source: str, content_type=None) -> None:
        summary = sanitize_summary(summary, max_chars=int(self.config.ambient_summary_max_chars))
        title = sanitize_summary(title, max_chars=200) or None
        await self.db.set_ambient_artifact_ready(
            channel_id=job.channel_id, source_ts=job.source_ts, kind=kind, ref=job.ref,
            title=title, summary=summary, model=model, derivation_source=derivation_source,
            content_type=content_type,
            expires_at=_expiry(int(self.config.ambient_artifact_retention_days)))
        # Dual-write the image catalog so read_document/edit paths still see ambient images.
        # Store the Slack file id + channel id in metadata so deletion/retention can match this
        # row EXACTLY (json_extract), never by a fragile substring LIKE.
        if kind == KIND_IMAGE and job.url:
            try:
                thread_key = f"{job.channel_id}:{job.conversation_ts}"
                await self.db.save_image_metadata_async(
                    thread_id=thread_key, url=job.url, image_type="uploaded",
                    prompt="", analysis=summary,
                    metadata={"ambient": True, "file_id": job.ref, "channel_id": job.channel_id},
                    message_ts=job.source_ts)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"ambient image catalog dual-write failed: {e}")
        art = {"kind": kind, "title": title, "summary": summary,
               "derivation_source": derivation_source}
        self._patch_pulse(job, art)
        # F51c: if this artifact's source message was ALREADY folded into the thread's
        # compaction summary, the pulse patch above is not enough — the thread's rebuilt
        # context can never see it (the message is gone from the tail, the summary was
        # written without this note). Persist a late addendum the rebuild folds onto the head.
        await self._record_late_addendum(job, art)

    async def _fail(self, job: _Job, kind: str, error_code: str, *, status: str = "failed",
                    derivation_source: Optional[str] = None) -> None:
        try:
            await self.db.set_ambient_artifact_status(
                channel_id=job.channel_id, source_ts=job.source_ts, kind=kind, ref=job.ref,
                status=status, error_code=error_code, increment_attempt=True,
                derivation_source=derivation_source)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient fail-persist error: {e}")

    async def _record_late_addendum(self, job: _Job, art: Dict[str, Any]) -> None:
        """Fold a just-completed artifact onto the compaction summary head IFF its source
        message has already been compacted away (source_ts <= boundary_ts for this thread).

        No-op when the message is still in the live tail (source_ts > boundary): the normal
        batch-load renders the note there, so adding an addendum too would double-describe it.
        Parity with the pulse/batch-load dedupe: an unfurl-sourced note is F48's Slack preview,
        already owned by that path — never re-describe it here. Idempotent + bounded in the DB
        layer, so a race with the compaction-time capture can't double-record or bloat."""
        if not self.db or not hasattr(self.db, "get_thread_summary_async"):
            return
        if art.get("derivation_source") == "unfurl":
            return
        try:
            note = render_artifact_note(art)
        except Exception:  # noqa: BLE001
            return
        if not note:
            return
        thread_key = f"{job.channel_id}:{job.conversation_ts}"
        try:
            summary = await self.db.get_thread_summary_async(thread_key)
            if not summary:
                return  # thread never compacted — nothing for this message to be behind
            boundary = float(summary.get("boundary_ts"))
            source = float(job.source_ts)
        except Exception:  # noqa: BLE001 — missing/non-numeric boundary or ts: skip safely
            return
        if source > boundary:
            return  # still in the live tail — the batch-load renders it (no double-inject)
        try:
            inserted = await self.db.add_thread_summary_addendum_async(
                thread_id=thread_key, channel_id=job.channel_id, source_ts=job.source_ts,
                kind=job.kind, ref=job.ref, note=note)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"late addendum record failed for {job.ref}: {e}")
            return
        # F51c: the addendum is in SQLite, but an ACTIVE warm thread still holds a summary head
        # written WITHOUT this note — it would answer from the stale head until a cold rebuild or
        # the next compaction. Flag it for refresh so its next turn rebuilds from Slack and folds
        # the addendum in. Fail-soft: a marking failure must not affect the addendum or artifact.
        if inserted:
            self._mark_thread_needs_refresh(thread_key)

    def _mark_thread_needs_refresh(self, thread_key: str) -> None:
        """Flag a warm thread whose summary head is now stale (a late addendum landed behind its
        compaction boundary), reusing the same thread-manager handle message edit/delete uses
        (`_mark_thread_refresh` in message_events). Reached via the Slack client's processor —
        the cheapest correct handle the service holds. Never raises."""
        try:
            client = self._client
            proc = getattr(client, "processor", None) if client is not None else None
            tm = getattr(proc, "thread_manager", None) if proc is not None else None
            if tm is not None and hasattr(tm, "mark_needs_refresh"):
                tm.mark_needs_refresh(thread_key)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient thread refresh mark failed for {thread_key}: {e}")

    def _patch_pulse(self, job: _Job, art: Dict[str, Any]) -> None:
        pulse = self.channel_pulse
        if pulse is None or not hasattr(pulse, "upsert_artifacts"):
            return
        # Dedupe against F48: an unfurl fallback is the SAME Slack preview F48 already rendered
        # into this message's pulse text (extract_supplementary_text). Rendering it again as an
        # artifact double-describes the link, so an unfurl-sourced note is not patched in.
        if art.get("derivation_source") == "unfurl":
            return
        try:
            note = render_artifact_note(art)
            if note:
                pulse.upsert_artifacts(job.channel_id, job.source_ts, [note])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ambient pulse upsert failed: {e}")
