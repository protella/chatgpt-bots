"""
Native Slack streaming coordinator (Phase G).

Drives chat.startStream/appendStream/stopStream for the streaming handlers via the
client's NativeStreamSession. The legacy sink EDITS one message per tick (display-safe
text, temporary fence closing, loader emoji); the native sink is APPEND-ONLY, so this
coordinator owns the differences:

- ticks append raw cumulative markdown (Slack renders progressively; no loader emoji,
  no temporary fence closing);
- when a part outgrows the per-message limit it "rolls": closes any open code fence,
  appends the continuation trailer, stops the stream, and starts a new native message
  whose base is the part prefix (+ reopened fence). Markers come from
  message_processor.message_markers in their markdown-flavored forms, which Slack stores as
  the exact canonical mrkdwn shapes the rebuild-side merger (_merge_continuation_history)
  strips — do NOT inline
  marker strings here (R2 context-pollution bug);
- any failure flips ``failed`` and the caller falls back to the legacy
  update_message_streaming edit loop on ``current_ts``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from message_processor.stale_send_guard import StaleSendSuppressed

from message_processor.message_markers import (
    _fence_state,
    entity_safe_cut,
    part_prefix_markdown,
)
from slack_client.formatting.text import split_streamable_name_mention

# A character that continues an "@Name" token (the outbound encoder's token class).
_NAME_CHARS = frozenset(".'-_")


# Agent surfaces §C.5: at most this many step cards per message part; further tool calls show no
# new card (updates to cards already shown still go). Slack's own ceiling is 50.
STEP_CARDS_PER_PART = 40
# §C.4: sources ride a completed web search's card, deduped, at most this many.
STEP_CARD_MAX_SOURCES = 10
_STEP_OPEN = "in_progress"


# §C (live review): a card's detail line — the query, the page — clipped to this many chars.
STEP_CARD_DETAILS_MAX = 200


def _source_chunk(url: str) -> Optional[Dict[str, str]]:
    """A task source for one URL, or None for anything that is not a well-formed http(s) URL."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return {"type": "url", "text": host, "url": url}


# Owner decision (live review): a back-to-back run of the same step is ONE card. The noun its
# count reads in; anything not listed counts "times". The merged detail block's ceiling.
_RUN_NOUNS = {
    "Searching the web": "searches",
    "Searching Slack": "searches",
    "Opening a page": "pages",
    "Reading a thread": "threads",
    "Reading channel history": "reads",
    "Running code": "runs",
    "Searching files": "searches",
    "Checking saved notes": "searches",
    "Opening a file": "files",
    "Downloading a file": "files",
    "Opening a link": "links",
    "Reading a document": "documents",
    "Reading a canvas": "canvases",
    "Looking at an image": "images",
    "Making an image": "images",
    "Starting an image": "images",
    "Importing an image": "images",
    "Looking up a channel": "channels",
    "Looking up a person": "people",
    "Posting to a thread": "posts",
    "Reacting": "reactions",
    "Remembering a fact": "facts",
    "Pinning a message": "messages",
}
STEP_CARD_DETAILS_BLOCK_MAX = 1000
# §F: how a turn ended, for the final plan (R16), and the model plan's step cap (R9).
ENDING_DELIVERED = "delivered"
ENDING_HANDOFF = "handoff"
ENDING_STOPPED = "stopped"
ENDING_FAILED = "failed"
HANDOFF_RESULT = "Continuing in the background"
_MODEL_MAX_STEPS = 6
_TERMINAL = ("complete", "error")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# A step's one-line result (`output`), plain, at most this long.
STEP_CARD_OUTPUT_MAX = 120
_SINGULAR = {"results": "result", "matches": "match", "messages": "message"}


def _count_phrase(k: int, unit: str) -> str:
    return f"{k} {_SINGULAR.get(unit, unit) if k == 1 else unit}"


# Owner feedback (sources panel): a task's sources go out ONCE, at turn end, at most this many.
# Preferred: what the answer CITED, labelled by the citation title; else a few distinct pages of
# the search's own hits, labelled "domain › path".
FINAL_SOURCES_MAX = 5
_CITED_LABEL_MAX = 80
_PATH_LABEL_MAX = 60


def _norm_url(url: str) -> str:
    try:
        p = urlparse(url)
    except ValueError:
        return url
    host = (p.netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{(p.path or '').rstrip('/')}{'?' + p.query if p.query else ''}"


def _path_label(url: str) -> str:
    """"abs.gov.au › statistics › population" — the domain and the first path segments."""
    try:
        p = urlparse(url)
    except ValueError:
        return _clip(url, _PATH_LABEL_MAX)
    host = (p.netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    segments = [seg for seg in (p.path or "").split("/") if seg][:2]
    return _clip(" › ".join([host] + segments), _PATH_LABEL_MAX)


def _domain(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


class StepCards:
    """One turn's tool steps as Slack's PLAN display: a `plan_update` title plus one
    `task_update` per card.

    RECORDING IS SYNCHRONOUS (§C.2): the hosted progress observer is called between two tokens
    of a live stream and must never schedule I/O, so it only writes here. The stream writer
    DRAINS at a safe boundary, and only then. Nothing here talks to Slack, and a turn that never
    opens a visible stream simply drops this object (§C.3).

    RUNS MERGE (owner decision): a new step whose label matches the card the PREVIOUS new step
    landed on joins that card; any different step ends the run. The merged card keeps its first
    step's id, so Slack updates it in place.

    APPEND-ONLY FIELDS (probed 2026-11-19): a task's `title` and `status` REPLACE, but its
    `details`, `output` and `sources` ACCUMULATE across updates. So details go out as deltas,
    sources once each, and the one-line output exactly once — when the card's run is over.

    The model-written plan (§F) swaps content, not structure: once `apply_plan` has run, the
    plan title and the counted steps are the model's (namespaced `m:<id>`, never merged with
    the automatic cards), NEW automatic cards stop, automatic cards already shown still get
    their terminal updates, and automatic events never shown are dropped (R13).
    `plan_title` overrides the computed title and `record(output=…)` a card's result line."""

    def __init__(self) -> None:
        self._cards: Dict[str, Dict[str, Any]] = {}     # card id -> card
        self._card_of: Dict[str, str] = {}              # step key -> card id
        self._last_card: Optional[str] = None           # the card the latest NEW step joined
        self._pending: List[str] = []                   # card ids changed since the last drain
        self._dropped: set = set()
        self._part_cards: set = set()
        self._seq = 0
        self._finished = False
        self._sent_title: Optional[str] = None
        # §F: the model-written plan. Steps by model id, in first-emit order, capped (R9).
        self._model_title: Optional[str] = None
        self._model: Dict[str, Dict[str, Any]] = {}
        self._model_order: List[str] = []
        self._model_pending: List[str] = []
        self._model_sent_output: set = set()            # per message part (R10/R15)
        self._title_dirty = False
        # Sources (owner feedback): every web search's own hits, by the task they belong to, and
        # what the answer cited. Nothing goes out until the turn ends.
        self._searches: List[Tuple[Optional[str], List[str]]] = []  # (target, urls)
        self._citations: List[Dict[str, str]] = []
        self._final_sources: Dict[str, List[Dict[str, str]]] = {}  # target -> source chunks
        # A caller-written plan title (the model-written plan); None = computed.
        self.plan_title: Optional[str] = None
        # Has any card ever been handed to the stream? (A silent ending keeps such a stream.)
        self.shown = False

    @property
    def has_pending(self) -> bool:
        return bool(self._pending or self._model_pending or self._title_dirty)

    @property
    def has_cards(self) -> bool:
        """The plan holds at least one task."""
        return bool(self._cards or self._model)

    @property
    def model_active(self) -> bool:
        """The model has written the plan this turn (§F takeover)."""
        return self._model_title is not None

    @property
    def auto_cards_shown(self) -> bool:
        """An automatic tool card reached the stream."""
        return any(c.get("ever_shown") for c in self._cards.values())

    def apply_plan(self, title: str, steps: List[Dict[str, Any]]) -> None:
        """§F: one model snapshot, already validated and sanitized. Whole-list semantics with
        Slack's limits (R9): a step keeps its id-assigned slot and first-emit order, a step
        missing from a later list keeps its last state, ids past the cap are ignored. Applied
        synchronously, so same-round calls land in call order and the last one wins (R11)."""
        if self._model_title is None:
            self._takeover()
        self._model_title = title
        for st in steps:
            mid = st["id"]
            step = self._model.get(mid)
            if step is None:
                if len(self._model_order) >= _MODEL_MAX_STEPS:
                    continue
                step = {"id": mid, "wire": f"m:{mid}", "title": st["title"],
                        "status": st["status"], "result": None, "locked_result": None}
                self._model[mid] = step
                self._model_order.append(mid)
            step["title"] = st["title"]
            step["status"] = st["status"]
            if st.get("result"):
                step["result"] = st["result"]
            if mid not in self._model_pending:
                self._model_pending.append(mid)
        self._title_dirty = True

    def _takeover(self) -> None:
        """R13: automatic cards never shown are dropped; shown ones keep their slot."""
        for cid in [c for c, card in self._cards.items() if not card.get("ever_shown")]:
            for m in self._cards[cid]["members"]:
                self._dropped.add(m["key"])
                self._card_of.pop(m["key"], None)
            self._cards.pop(cid)
            self._part_cards.discard(cid)
            if cid in self._pending:
                self._pending.remove(cid)
        self._last_card = None

    def add_citations(self, items: List[Dict[str, Any]]) -> None:
        """The answer's own `url_citation`s ({"url", "title"}), deduped by normalized URL."""
        seen = {_norm_url(c["url"]) for c in self._citations}
        for item in items or []:
            url = item.get("url") if isinstance(item, dict) else None
            if not isinstance(url, str) or _source_chunk(url) is None:
                continue
            key = _norm_url(url)
            if key in seen:
                continue
            seen.add(key)
            title = item.get("title") if isinstance(item.get("title"), str) else ""
            self._citations.append({"url": url, "title": title or ""})

    def _note_search(self, key: str, sources: Optional[List[str]]) -> None:
        """Remember which task a web search's hits belong to: its automatic card, or — once the
        model writes the plan — the model step in progress when the search ran (None = the last
        step, decided at the end)."""
        urls = [u for u in (sources or []) if isinstance(u, str) and _source_chunk(u) is not None]
        if not urls:
            return
        if self._model_title is not None:
            running = [mid for mid in self._model_order
                       if self._model[mid]["status"] == _STEP_OPEN]
            target: Optional[str] = f"m:{running[-1]}" if running else None
        else:
            target = self._card_of.get(key)
        self._searches.append((target, urls))

    def record(self, key: str, title: str, status: str,
               sources: Optional[List[str]] = None, details: Optional[str] = None,
               outcome: Optional[Dict[str, Any]] = None,
               output: Optional[str] = None, kind: Optional[str] = None) -> None:
        """Note one step's current state. A step that would open a card past this part's cap
        is ignored for good. Sources are validated HERE (http/https only), so a bad URL can
        never reach a drain. `outcome` is the step's result in counting form ({"count", "unit",
        "domains"} or {"text"}); `output` overrides the card's whole result line."""
        if not key or not title or key in self._dropped:
            return
        card_id = self._card_of.get(key)
        if card_id is None and self._model_title is not None:
            if kind == "web_search":
                self._note_search(key, sources)     # its hits still belong to a model step
            self._dropped.add(key)          # R13: the model's plan owns the display now
            return
        if card_id is None:
            last = self._cards.get(self._last_card or "")
            if last is not None and last["label"] == title:
                card_id = last["id"]
            else:
                if len(self._part_cards) >= STEP_CARDS_PER_PART:
                    self._dropped.add(key)
                    return
                self._seq += 1
                card_id = f"s{self._seq}"
                self._cards[card_id] = {"id": card_id, "label": title, "members": [],
                                        "output": None, "sent_details": "",
                                        "sent_sources": set(), "sent_output": False}
                self._part_cards.add(card_id)
                # The run before this one is over: its result line may now go out.
                if last is not None and last["id"] not in self._pending:
                    self._pending.append(last["id"])
            self._cards[card_id]["members"].append(
                {"key": key, "title": title, "status": status, "details": "", "sources": [],
                 "outcome": None})
            self._card_of[key] = card_id
            self._last_card = card_id
        card = self._cards[card_id]
        member = next(m for m in card["members"] if m["key"] == key)
        member["title"] = title
        member["status"] = status
        if sources:
            valid: List[Dict[str, str]] = []
            for url in sources:
                if not isinstance(url, str) or any(v["url"] == url for v in valid):
                    continue
                chunk = _source_chunk(url)
                if chunk is not None:
                    valid.append(chunk)
            member["sources"] = valid
        if isinstance(details, str) and details.strip():
            member["details"] = _clip(" ".join(details.split()), STEP_CARD_DETAILS_MAX)
        if isinstance(outcome, dict):
            member["outcome"] = outcome
        if isinstance(output, str) and output.strip():
            card["output"] = output
        if kind == "web_search" and sources:
            self._note_search(key, sources)
        if card_id not in self._pending:
            self._pending.append(card_id)

    # ---- rendering ------------------------------------------------------------------------

    @staticmethod
    def _status(card: Dict[str, Any]) -> str:
        statuses = [m["status"] for m in card["members"]]
        if any(st == _STEP_OPEN for st in statuses):
            return _STEP_OPEN
        if any(st == "complete" for st in statuses):
            return "complete"
        if all(st == "error" for st in statuses):
            return "error"
        return statuses[-1]

    @staticmethod
    def _title(card: Dict[str, Any]) -> str:
        members = card["members"]
        n = len(members)
        # A lone step keeps its own (possibly refined) title; a run counts itself.
        return (members[0]["title"] if n == 1
                else f"{card['label']} ({n} {_RUN_NOUNS.get(card['label'], 'times')})")

    @staticmethod
    def _output(card: Dict[str, Any]) -> Optional[str]:
        """The card's one-line result: counts summed across a merged run, else a short status."""
        if card["output"]:
            return _clip(" ".join(str(card["output"]).split()), STEP_CARD_OUTPUT_MAX)
        members = card["members"]
        ok = [m for m in members if m["status"] == "complete"]
        if not ok:
            failed = [m for m in members if m["status"] == "error"]
            if not failed:
                return None
            text = (failed[-1]["outcome"] or {}).get("text")
            return text if isinstance(text, str) and text else "Failed"
        outcomes = [m["outcome"] for m in ok if isinstance(m["outcome"], dict)]
        counted = [o for o in outcomes if isinstance(o.get("count"), int)]
        if counted and len({o.get("unit") for o in counted}) == 1:
            total = sum(o["count"] for o in counted)
            line = _count_phrase(total, str(counted[0].get("unit") or "results"))
            domains: List[str] = []
            for o in counted:
                for d in o.get("domains") or []:
                    if d and d not in domains:
                        domains.append(d)
            if domains:
                line += " from " + ", ".join(domains[:3])
            return _clip(line, STEP_CARD_OUTPUT_MAX)
        texts = [o["text"] for o in outcomes if isinstance(o.get("text"), str) and o["text"]]
        return _clip(texts[-1], STEP_CARD_OUTPUT_MAX) if texts else None

    def _closed(self, card: Dict[str, Any]) -> bool:
        """The run is over: nothing in it runs, and either a new card followed it or the turn
        has ended. Only then is its result line final — `output` can never be rewritten."""
        return (self._status(card) != _STEP_OPEN
                and (self._finished or card["id"] != self._last_card))

    def _model_chunk(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """A model step's task_update. Its output goes out ONCE per part, when the step is
        terminal AND has a result; after that the line is locked (R10)."""
        chunk: Dict[str, Any] = {"type": "task_update", "id": step["wire"],
                                 "title": step["title"], "status": step["status"]}
        final = self._final_sources.get(step["wire"])
        if final and not step.get("sources_sent"):
            chunk["sources"] = final
            step["sources_sent"] = True
        if step["status"] in _TERMINAL and step["id"] not in self._model_sent_output:
            line = step["locked_result"] or step["result"]
            if line:
                chunk["output"] = line
                step["locked_result"] = line
                self._model_sent_output.add(step["id"])
        return chunk

    def _task_chunk(self, card: Dict[str, Any]) -> Dict[str, Any]:
        card["ever_shown"] = True
        chunk: Dict[str, Any] = {"type": "task_update", "id": card["id"],
                                 "title": self._title(card), "status": self._status(card)}
        lines = [m["details"] for m in card["members"] if m["details"]]
        block = _clip("\n".join(lines), STEP_CARD_DETAILS_BLOCK_MAX) if lines else ""
        sent = card["sent_details"]
        if block and block.startswith(sent) and len(block) > len(sent):
            chunk["details"] = block[len(sent):]
            card["sent_details"] = block
        final = self._final_sources.get(card["id"])
        if final and not card["sent_sources"]:
            chunk["sources"] = final
            card["sent_sources"].update(f["url"] for f in final)
        if not card["sent_output"] and self._closed(card):
            line = self._output(card)
            if line:
                chunk["output"] = line
            card["sent_output"] = True
        return chunk

    def _plan_title(self) -> str:
        if self._model_title is not None:
            # R8: counted over the MODEL's steps only; done = complete or error.
            steps = [self._model[mid] for mid in self._model_order]
            done = sum(1 for st in steps if st["status"] in _TERMINAL)
            return f"{self._model_title} ({done} of {len(steps)} done)"
        cards = list(self._cards.values())
        total = len(cards)
        # R8: pending is NOT done — only finished (complete or error) cards count.
        done = sum(1 for c in cards if self._status(c) in _TERMINAL)
        if self.plan_title:
            base = self.plan_title
        elif self._finished:
            labels = {c["label"] for c in cards}
            base = (next(iter(labels)) if len(labels) == 1
                    else f"Worked through {total} steps")
        else:
            running = [c for c in cards if self._status(c) == _STEP_OPEN]
            current = running[-1] if running else cards[-1]
            base = current["label"]
        return f"{base} ({done} of {total} done)"

    def drain(self) -> List[Dict[str, Any]]:
        """The plan title (when it changed) and the task_update chunks for every card that
        changed since the last drain."""
        if not self.has_pending:
            return []
        chunks = [self._task_chunk(self._cards[cid]) for cid in self._pending
                  if cid in self._cards]
        chunks += [self._model_chunk(self._model[mid]) for mid in self._model_pending]
        self._pending = []
        self._model_pending = []
        self._title_dirty = False
        if not self._cards and not self._model:
            return []
        title = self._plan_title()
        if title != self._sent_title:
            chunks.insert(0, {"type": "plan_update", "title": title})
            self._sent_title = title
        if chunks:
            self.shown = True
        return chunks

    def progress_snapshot(self) -> Optional[Tuple[str, List[Dict[str, str]]]]:
        """§G GR4: (title, tasks) for a held turn's temporary card — the model plan when there
        is one, else the automatic steps. READ-ONLY: nothing pending, sent or shown changes, so
        the turn's final delivery still carries the complete plan. None when nothing shows."""
        if self._model_title is not None:
            # Review 12: ids by first-emit POSITION — unique however the model spelled its ids
            # ("a:b" and "a-b" would otherwise serialize alike). Results ride too (review 13).
            tasks = [{"id": f"m{n}", "title": self._model[mid]["title"],
                      "status": self._model[mid]["status"],
                      "result": self._model[mid]["locked_result"] or self._model[mid]["result"]
                      or ""}
                     for n, mid in enumerate(self._model_order, start=1)]
        else:
            tasks = [{"id": card["id"], "title": self._title(card), "status": self._status(card)}
                     for card in self._cards.values()]
        if not tasks:
            return None
        return self._plan_title(), tasks

    def _open(self, card_id: str) -> bool:
        return self._status(self._cards[card_id]) == _STEP_OPEN

    def reopen_for_new_part(self) -> List[Dict[str, Any]]:
        """A part rolled (§C.5): its plan stays on the old message, so the new part gets the
        title again plus every still-open card and every undrained change, sent whole (merged
        state and all), and starts its own cap."""
        carry = [cid for cid in self._cards if cid in self._pending or self._open(cid)]
        self._part_cards = set(carry[:STEP_CARDS_PER_PART])
        self._pending = [cid for cid in carry if cid in self._part_cards]
        for cid in self._pending:
            card = self._cards[cid]
            card["sent_details"], card["sent_sources"], card["sent_output"] = "", set(), False
        # R15: the new part gets the FULL model plan — every step, its state, and the output
        # of every finished step, re-emitted on this part.
        self._model_sent_output = set()
        self._model_pending = list(self._model_order)
        self._sent_title = None
        self._title_dirty = True
        return self.drain()

    def close_open(self, success: bool) -> List[Dict[str, Any]]:
        """The final drain for a turn that succeeded or failed (see `close`)."""
        return self.close(ENDING_DELIVERED if success else ENDING_FAILED)

    def close(self, ending: str) -> List[Dict[str, Any]]:
        """The final drain, by how the turn ended (R16).

        Automatic cards: an open step ends `complete` on delivered/handoff, `error` otherwise.
        Model steps: delivered ⇒ non-terminal → complete, no result invented; handoff ⇒
        non-terminal → complete, "Continuing in the background"; stopped ⇒ in_progress →
        error, pending stays pending; failed ⇒ everything not complete → error. Every result
        line still owed goes out, with the final title."""
        self._finished = True
        ok = ending in (ENDING_DELIVERED, ENDING_HANDOFF)
        for cid, card in self._cards.items():
            changed = False
            for m in card["members"]:
                if m["status"] == _STEP_OPEN:
                    m["status"] = "complete" if ok else "error"
                    changed = True
            if (changed or not card["sent_output"]) and cid not in self._pending:
                self._pending.append(cid)
        for mid in self._model_order:
            st = self._model[mid]
            before = (st["status"], st["result"])
            if ending == ENDING_DELIVERED and st["status"] not in _TERMINAL:
                st["status"] = "complete"
            elif ending == ENDING_HANDOFF and st["status"] not in _TERMINAL:
                st["status"] = "complete"
                st["result"] = st["result"] or HANDOFF_RESULT
            elif ending == ENDING_STOPPED and st["status"] == _STEP_OPEN:
                st["status"] = "error"
            elif ending == ENDING_FAILED and st["status"] != "complete":
                st["status"] = "error"
            if ((st["status"], st["result"]) != before
                    or (st["status"] in _TERMINAL and st["result"]
                        and mid not in self._model_sent_output)) \
                    and mid not in self._model_pending:
                self._model_pending.append(mid)
        self._assign_sources()
        self._title_dirty = True
        return self.drain()

    def _assign_sources(self) -> None:
        """The turn is over: pick each task's sources, once (append-only — they can never be
        corrected). The answer's citations when it has any, each on the search task whose own
        hits contained it (else the last one), labelled by title; otherwise a few distinct
        pages of each search's own hits, distinct domains first, labelled "domain › path"."""
        last_model = f"m:{self._model_order[-1]}" if self._model_order else None
        resolved: List[Tuple[str, List[str]]] = []
        for target, urls in self._searches:
            if target is None or (target.startswith("m:") and target[2:] not in self._model):
                target = last_model
            if target is not None and (target.startswith("m:") or target in self._cards):
                resolved.append((target, urls))
        final: Dict[str, List[Dict[str, str]]] = {}
        if self._citations:
            fallback = resolved[-1][0] if resolved else last_model
            for cite in self._citations:
                key = _norm_url(cite["url"])
                target = next((t for t, urls in resolved
                               if key in {_norm_url(u) for u in urls}), fallback)
                if target is None:
                    continue
                bucket = final.setdefault(target, [])
                if len(bucket) < FINAL_SOURCES_MAX:
                    label = (_clip(" ".join(cite["title"].split()), _CITED_LABEL_MAX)
                             if cite["title"].strip() else _path_label(cite["url"]))
                    bucket.append({"type": "url", "text": label, "url": cite["url"]})
        else:
            pooled: Dict[str, List[str]] = {}
            for target, urls in resolved:
                pool = pooled.setdefault(target, [])
                for url in urls:
                    if all(_norm_url(url) != _norm_url(u) for u in pool):
                        pool.append(url)
            for target, pool in pooled.items():
                picked: List[str] = []
                domains: set = set()
                for url in pool:                    # distinct domains first …
                    if _domain(url) not in domains:
                        picked.append(url)
                        domains.add(_domain(url))
                for url in pool:                    # … then other pages to fill the cap
                    if url not in picked:
                        picked.append(url)
                final[target] = [{"type": "url", "text": _path_label(u), "url": u}
                                 for u in picked[:FINAL_SOURCES_MAX]]
        self._final_sources = final
        for target in final:
            if target.startswith("m:"):
                mid = target[2:]
                if mid not in self._model_pending:
                    self._model_pending.append(mid)
            elif target not in self._pending:
                self._pending.append(target)


def find_stream_split(text: str, limit: int, floor: int = 0) -> int:
    """Best split index in (floor, limit] — paragraph, then sentence, then newline,
    then word boundary, then an entity-safe hard cut. ``floor`` is the number of
    characters an append-only sink has already sent (can't be unsent)."""
    floor = max(0, min(floor, len(text)))
    if limit <= floor:
        return floor
    search_start = max(floor, limit - 500)
    for probe, offset in (("\n\n", 2), (". ", 2), ("\n", 1), (" ", 1)):
        idx = text.rfind(probe, search_start, limit)
        if idx > floor:
            return idx + offset
    return max(entity_safe_cut(text, limit), floor) or limit


class NativeStreamCoordinator:
    """Multi-part native streaming with shared continuation markers.

    Usage: ``start()`` once on first content; ``update(raw)`` per tick with the
    CURRENT part's cumulative raw text — returns ``(ok, overflow)`` where a non-None
    overflow means the part rolled and the caller must reset its buffer to exactly
    that text; ``finalize(raw, suffix)`` at the end appends the tail (+ attribution
    suffix) and stops the stream.
    """

    def __init__(self, client, channel_id: str, thread_ts: Optional[str],
                 char_limit: int, logger=None, user_id: Optional[str] = None,
                 receipts=None, chunks_mode: bool = False,
                 steps: Optional[StepCards] = None):
        self._client = client
        # Agent surfaces §C: every part (the first AND each roll) streams in chunks mode, and
        # `steps` is the turn's step-card record drained into those parts. Off ⇒ unchanged.
        self.chunks_mode = bool(chunks_mode)
        self.steps = steps if self.chunks_mode else None
        # Raw text of every part already closed by a roll (base excluded), for `accepted_text`.
        self._closed_raw: List[str] = []
        # chat.startStream MINTS a message, so every part is a durable post this turn owns.
        # Registered as it is minted, never at lease-commit time: the lease says the turn may
        # speak, the receipt says the room has words it will keep (spec §5).
        self._receipts = receipts
        self.channel = channel_id
        self.thread_ts = thread_ts
        # Triggering user's id — chat.startStream needs it as recipient_user_id for
        # channel streaming; forwarded to begin_native_stream on start() and roll.
        self.user_id = user_id
        self.char_limit = max(200, char_limit)
        self._log = logger or (lambda msg: None)
        # The client's NativeStreamSession, minted by begin_native_stream; `_client` is duck-typed
        # (real bot or a test double), so the session is too.
        self.session: Any = None
        self.base = ""       # non-buffer prefix of the current part (part prefix + fence reopen)
        self.part = 1
        self.failed = False
        self.finished = False
        self.part_ts: list = []  # ts of every native message created (finished + current)
        # HIDDEN-BUFFER MODE. The stale-send guard refused `start()`, so chat.startStream was
        # never called and this coordinator owns no message at all. It is deliberately NOT
        # `failed`: `failed` means "Slack let us down, fall back to the legacy edit loop", and
        # that fallback would post the very answer the guard refused. Every write path below
        # checks `hidden` first and does nothing, so the generation runs to completion with no
        # visible surface and the finished draft goes to reconsideration instead.
        self.suppression: Optional[Exception] = None

    @property
    def hidden(self) -> bool:
        """Was the first surface refused before Slack was ever called?"""
        return self.suppression is not None

    @property
    def active(self) -> bool:
        return (not self.failed and not self.finished and not self.hidden
                and self.session is not None and self.session.active)

    @property
    def started(self) -> bool:
        return self.session is not None

    @property
    def user_stopped(self) -> bool:
        """Slack stopped the current part itself (the native Stop button, §A.3.3)."""
        return bool(self.session is not None and getattr(self.session, "stopped_by_user", False))

    def _begin(self) -> Any:
        if self.chunks_mode:
            return self._client.begin_native_stream(self.channel, self.thread_ts,
                                                    user_id=self.user_id, chunks_mode=True)
        return self._client.begin_native_stream(self.channel, self.thread_ts, user_id=self.user_id)

    def _boundary_steps(self) -> Optional[List[Dict[str, Any]]]:
        """Pending step cards, but only at a safe boundary: nothing withheld in the session.
        Cards are auxiliary: anything going wrong here is logged and the cards dropped."""
        if self.steps is None or not self.steps.has_pending:
            return None
        if self.session is None or getattr(self.session, "holding", False):
            return None
        try:
            return self.steps.drain()
        except Exception as e:  # noqa: BLE001
            self._log(f"step cards dropped: {e}")
            return None

    def _final_steps(self, ending: str) -> Optional[List[Dict[str, Any]]]:
        if self.steps is None:
            return None
        try:
            return self.steps.close(ending)
        except Exception as e:  # noqa: BLE001
            self._log(f"final step cards dropped: {e}")
            return None

    def accepted_text(self) -> str:
        """§A.3: the raw answer text Slack actually accepted across every part (base, part
        prefixes and withheld text excluded) — what a stopped turn records as said."""
        current = ""
        if self.session is not None:
            accepted = getattr(self.session, "accepted_text", None)
            sent = accepted() if callable(accepted) else getattr(self.session, "_sent", "")
            current = sent[len(self.base):] if sent.startswith(self.base) else sent
        return "".join(self._closed_raw) + current

    async def finish_stopped(self, blocks=None) -> bool:
        """End a stopped turn's stream with nothing more said: no tail, withheld text dropped,
        open cards closed as errors. True when the stream is closed (or Slack closed it)."""
        if self.hidden or self.session is None or self.finished:
            return False
        if self.user_stopped or not self.session.active:
            self.finished = True
            return True
        try:
            ok = await self.session.finish(blocks=blocks,
                                           task_chunks=self._final_steps(ENDING_STOPPED),
                                           release_held=False)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator stopped-finish error: {e}")
            ok = False
        self.finished = True
        return bool(ok)

    @property
    def current_ts(self) -> Optional[str]:
        return self.session.ts if self.session is not None else None

    def _sent_raw_len(self) -> int:
        """Chars of the current part's RAW text already appended (base excluded)."""
        if self.session is None:
            return 0
        return max(0, len(self.session._sent) - len(self.base))

    async def start(self, lease: Any = None) -> bool:
        """`lease` (stale guard): chat.startStream MINTS the reply message, so it is a first
        answer surface and the session checks the lease before making the call."""
        try:
            self.session = self._begin()
            # §C.3: steps buffered before the stream existed open it, with their current state.
            early = self.steps.drain() if (self.steps is not None and self.steps.has_pending) else None
            if early:
                ok = await self.session.start(lease=lease, task_chunks=early)
            else:
                ok = await self.session.start(lease=lease)
        except StaleSendSuppressed as suppressed:
            # Not a sink failure: the conversation moved on and the stream was deliberately
            # never opened. Swallowed here it would look like Slack refusing us, and the caller
            # would quietly fall back to the legacy edit loop and post the stale answer anyway.
            # Remembered rather than merely re-raised: this exact object is the one the
            # reconsideration machinery is identity-bound to, and `hidden` is what keeps every
            # later write path (tool flush, finalize, abandon) from touching Slack.
            self.suppression = suppressed
            raise
        except Exception as e:  # noqa: BLE001 - best-effort sink, never fatal
            self._log(f"native coordinator start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
            await self._note_part(self.session.ts)
        else:
            self.failed = True
        return not self.failed

    async def _note_part(self, ts: str) -> None:
        """Claim one native part. Never raises — the message is already in the room."""
        if self._receipts is None:
            return
        try:
            # Spec §4: every native stream part is the answer itself — assistant_reply.
            await self._receipts.note_post(ts, thread_root_ts=self.thread_ts,
                                           receipt_class="assistant_reply")
        except Exception as e:  # noqa: BLE001
            self._log(f"native part receipt failed: {e}")

    async def update(self, raw_text: str, with_steps: bool = True) -> Tuple[bool, Optional[str]]:
        """Append the tail of the current part's cumulative raw text; roll on overflow.

        `with_steps=False` sends text only: the caller is flushing text that PREDATES the step
        cards now pending, so they must not go ahead of it."""
        if self.hidden:
            # Nothing to append to and nothing to fail: report "not written" WITHOUT flipping
            # `failed`, which is the flag that hands the turn to the legacy edit loop.
            return False, None
        if self.user_stopped:
            # Slack closed the stream for the user's Stop: the prefix stands, nothing more goes
            # in, and this is not a failure (no legacy fallback, no republish — §A.3.3).
            return True, None
        if not self.active:
            self.failed = True
            return False, None
        try:
            if len(raw_text) <= self.char_limit:
                steps = self._boundary_steps() if with_steps else None
                if steps:
                    ok = await self.session.update(self.base + raw_text, task_chunks=steps)
                else:
                    ok = await self.session.update(self.base + raw_text)
                if self.user_stopped:
                    return True, None
                if not ok:
                    self.failed = True
                return ok, None
            return await self._roll(raw_text)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator update error: {e}")
            self.failed = True
            return False, None

    async def push_steps(self) -> bool:
        """A step-card-only append at a tool event, so a card is live while the tool runs.
        Only at a safe boundary (nothing withheld); otherwise the cards wait for the next
        append. True when nothing needed sending or the append landed."""
        if self.hidden or self.user_stopped or not self.active:
            return True
        steps = self._boundary_steps()
        if not steps:
            return True
        try:
            ok = await self.session.update(self.session._sent, task_chunks=steps)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator step push error: {e}")
            ok = False
        if self.user_stopped:
            return True
        if not ok:
            self.failed = True
        return ok

    async def _roll(self, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Close the current part and open the next one."""
        # A held "@Na" may still become "@Name": finishing the part would resolve the half it
        # has. It goes back to the buffer, and the split never lands inside a name, so the whole
        # token opens the next part instead.
        withdraw = getattr(self.session, "withdraw_held_name", None)
        if callable(withdraw):
            withdraw()
        floor = self._sent_raw_len()
        split = find_stream_split(raw_text, self.char_limit, floor=floor)
        if 0 < split < len(raw_text) and (raw_text[split].isalnum() or raw_text[split] in _NAME_CHARS):
            release, candidate = split_streamable_name_mention(raw_text[:split])
            if candidate and release and len(release) >= floor:
                split = len(release)
        first = raw_text[:split]
        if not await self.session.update(self.base + first):
            self.failed = True
            return False, None
        if self.user_stopped:
            return True, None       # stopped from Slack: no next part is opened
        # Fence continuity: append-only means we close the fence by APPENDING "```",
        # then reopen it (with the language hint) in the next part's base.
        in_block, lang = _fence_state(self.base + first)
        closing = "\n```" if in_block else ""
        # No trailing "Continued in next message..." (user directive 2026-07-11): the next
        # part's "Part N (continued)" header marks the seam by itself, and the rebuild
        # merger fires on EITHER side's marker (thread_management merge is OR).
        if self.chunks_mode:
            # The turn is still working: the intermediate part leaves the session processing.
            finished = await self.session.finish(final_text=self.base + first + closing,
                                                 session_status="processing")
        else:
            finished = await self.session.finish(final_text=self.base + first + closing)
        if self.user_stopped:
            return True, None       # stopped from Slack while closing: no next part is opened
        overflow = raw_text[split:].lstrip("\n")
        if not finished:
            self.failed = True
            return False, overflow
        self._closed_raw.append(first)
        self.part += 1
        self.base = part_prefix_markdown(self.part) + (f"```{lang}\n" if in_block else "")
        try:
            self.session = self._begin()
            reopened = self.steps.reopen_for_new_part() if self.steps is not None else None
            if reopened:
                ok = await self.session.start(self.base + overflow, task_chunks=reopened)
            else:
                ok = await self.session.start(self.base + overflow)
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator roll-start error: {e}")
            ok = False
        if ok and self.session.ts:
            self.part_ts.append(self.session.ts)
            await self._note_part(self.session.ts)
            self._log(f"native stream rolled to part {self.part} (fence reopened: {in_block})")
            return True, overflow
        if self.chunks_mode and self.session is not None:
            # The closed part left the session `processing` for a part that never opened.
            clear = getattr(self.session, "clear_session_status", None)
            if callable(clear):
                await clear()
        self.failed = True
        return False, overflow

    async def finalize(self, final_raw: str, suffix: str = "", blocks=None,
                       success: bool = True, ending: Optional[str] = None) -> bool:
        """Append any remaining tail (+ suffix, e.g. tools attribution) and stop.

        ``blocks`` (e.g. the channel Configure chrome) ride the LAST part's stopStream
        only — intermediate rolled parts close without them, so a multi-part response
        carries the chrome exactly once, at the very end.

        Returns False if anything failed — the caller should fall back to the legacy
        final-correction edit on ``current_ts`` (and post any chrome separately)."""
        if self.hidden:
            # The refused stream has no message to finalize. Returning False here does NOT mean
            # "fall back and post it anyway": the caller's hidden branch owns this turn's
            # delivery, through reconsideration.
            return False
        if self.session is None or self.finished:
            return False
        if self.user_stopped:
            # The stream was closed by the user's Stop: its delivered prefix IS the reply.
            self.finished = True
            return True
        try:
            text = final_raw
            while self.active and len(text) + len(suffix) > self.char_limit:
                ok, overflow = await self._roll(text)
                if self.user_stopped:
                    self.finished = True
                    return True
                if overflow is None or not ok:
                    return False
                text = overflow
            if self.user_stopped:
                self.finished = True
                return True
            if not self.active:
                return False
            if self.steps is not None:
                ok = await self.session.finish(final_text=self.base + text + suffix, blocks=blocks,
                                               task_chunks=self._final_steps(
                                                   ending or (ENDING_DELIVERED if success
                                                              else ENDING_FAILED)))
            else:
                ok = await self.session.finish(final_text=self.base + text + suffix, blocks=blocks)
            if ok:
                self.finished = True
            else:
                self.failed = True
            return ok
        except Exception as e:  # noqa: BLE001
            self._log(f"native coordinator finalize error: {e}")
            self.failed = True
            return False

    async def abandon(self) -> bool:
        """Stop the stream without appending anything (e.g. reaction-only turns).

        Returns False if the stop call failed (the caller should treat the stream message
        as possibly-lingering and fall back to an explicit delete)."""
        if self.hidden:
            # No stream was ever opened, so there is nothing to stop and nothing lingering.
            self.finished = True
            return True
        ok = True
        if self.session is not None and self.session.active:
            try:
                if self.steps is not None:
                    # A failed or abandoned turn: any step still in progress ends in error.
                    ok = await self.session.finish(task_chunks=self._final_steps(ENDING_FAILED))
                else:
                    ok = await self.session.finish()
            except Exception as e:  # noqa: BLE001
                self._log(f"native coordinator abandon error: {e}")
                ok = False
        self.finished = True
        return ok
