# Tool Subsystems — hard-won API facts

Read this before touching image tools, the code interpreter/container path, file mounting,
artifact publishing, deep research, or Slack canvases. Every rule below is here because it
shipped a bug or cost a live debugging session; almost none of it is in the vendor docs.

## Image generation as tools (F34)

**There is no intent classifier on the image path any more.** `ENABLE_IMAGE_TOOLS` (default on)
makes image work a set of TOOLS the model calls in context (`message_processor/image_tools.py`),
so a turn can generate an image AND compute a chart from real data — which the old single-choice
router made impossible. The classifier + the `vision`/`new_image`/`edit_image`/`ambiguous_image`
routing in `message_processor/base.py` only run when the flag is OFF (an escape hatch, not the
intended path).

Three tools, named so their execution contracts can't be confused:
- **`generate_image`** — DETACHED. Schedules the existing background job; returns immediately;
  the image posts itself. The everyday case.
- **`create_image_asset`** — SYNCHRONOUS. Pushes the bytes into the thread's persistent
  container at `/mnt/data` and posts NOTHING. The image is an *ingredient*; whatever the code
  builds from it (a `.pptx`, a composite) is the deliverable. Only possible because container
  ids are persisted (F32) — under `{"type":"auto"}` there is nothing to push into. Gated off
  when there is no addressable container.
- **`edit_image`** — SYNCHRONOUS. Sources are named by opaque id (`img_<row id>`) from a catalog
  we inject as a literal `enum` (`message_processor/image_catalog.py`). There is **no "most
  recent" fallback** and no utility-model "which one did they mean?" guess — editing the wrong
  image is expensive and irreversible, so an unresolvable id is an error the model recovers
  from. A syntactically valid id is not authorization: ids are re-validated against the turn's
  snapshot.

**Settings (`message_processor/image_service.py`)**: the image MODEL is a HARD constraint from
the user's saved prefs and is deliberately absent from every schema — the model cannot express a
different one. Everything else (size/quality/background/format/compression) is a *preference*:
the user's value is the default, exposed in the tool description, and the model may override it
via an `overrides` object when the task warrants. Schemas are FACTORIES, because the legal option
space depends on the selected model (gpt-image-2 has no transparent background, auto-handles
input fidelity, and takes arbitrary WxH).

Image model defaults to **`gpt-image-2`** (`GPT_IMAGE_MODEL`); `gpt-image-1` and
`gpt-image-1-mini` remain selectable per-user in `/settings`.

**Custom sizes are on a 16px grid.** Verified live: the API rejects any side not divisible by 16
("Width and height must both be divisible by 16"), so `1920x1080` is a hard 400. `normalize_size()`
SNAPS a near-miss (`1920x1080` → `1920x1088`, still 16:9) rather than falling back to the user's
default, which would silently hand back a *square*. A true 16:9 that is legal: `1536x864`.

**Vision is not a route.** Uploaded images ride the turn as `input_image` parts, so the model just
sees them. What survives from the old vision handler is the durable part —
`image_catalog.catalog_uploads()` stores a canonical *visual description* in the background (what
the image IS, not the model's answer about it), so a later "edit that screenshot" has something to
work from.

**Charts/graphs/plots are NOT image generation.** They are computed in the code-interpreter sandbox
from real data. An image model draws a plausible-looking chart with invented numbers *and invented
categories* — this shipped once. The `generate_image` tool description says so explicitly.

## Code interpreter + artifacts (F32)

The model writes and runs Python in an **OpenAI-hosted container** — this is the "scratch space",
so the no-local-disk rule holds with **zero new local dependencies**. Files it writes are
downloaded and uploaded into the Slack thread (`message_processor/artifacts.py`).

**Containers are THREAD-SCOPED and persisted** (`message_processor/containers.py`,
`thread_containers` table). One container per `channel:thread_ts`, reused across turns so sandbox
state survives the turn boundary. **The API caps idle life at 20 minutes** (`expires_after.minutes`
must be ≤ 20 — 60 returns HTTP 400), so a revived thread necessarily gets a fresh, empty one.
Reaped in the daily cleanup worker. Any failure degrades to `{"type":"auto"}` (ephemeral
container) — never to a missing tool.

Two traps that already shipped bugs once each:
- **File citations are useless here.** They only appear if the model writes a `sandbox:` link,
  which our prompt forbids. The container **LISTING** (`source == "assistant"`, fails closed) is
  the only artifact source. A citation-driven v1 passed every unit test and published zero files.
- **A reused container's listing is CUMULATIVE** — turn 2 sees turn 1's files. Published file ids
  are persisted (`published_files_json`) because the in-memory dedupe dies with the process.

**Sandbox capabilities** (probed live 2026-07-12, Python 3.13):
- *Data*: pandas, numpy, scipy, sklearn, statsmodels, sympy, numba, networkx, h5py
- *Charts*: matplotlib, seaborn, plotly, wordcloud, graphviz + `dot`, pydot
- *Office*: **python-docx, openpyxl, xlsxwriter, python-pptx** — plus **LibreOffice/`soffice`** and
  **pandoc** for format conversion. (`xlrd` and `pyarrow`/`fastparquet` are absent.)
- *PDF*: pypdf, reportlab, fpdf, weasyprint, `pdftoppm`. (`PyPDF2` is absent — use `pypdf`.)
- *Images/media*: PIL, cv2, imageio, cairosvg, svglib, **ffmpeg**, moviepy, **tesseract**/pytesseract
- *Archives*: `zipfile`, `tarfile`, `zip`/`unzip` binaries
- *Geo*: geopandas, shapely, folium
- **NO network egress.** `pip install` fails at DNS; exfiltration is impossible. Bytes must be
  *pushed in* via `containers.files.create(container_id, file=…)` — which needs a known container
  id, i.e. only works now that containers are persisted.

## mount_file — user files INTO the sandbox (F35)

The sandbox starts EMPTY. For a long time the ONLY bytes that could reach `/mnt/data` were images
the bot generated itself, so "build a PDF from these screenshots" and "analyse this 50k-row CSV"
were structurally impossible — the model could SEE an attachment (images ride as `input_image`,
documents are text-extracted) but never compute on it. `message_processor/file_mount.py` is the
bridge; `thread_files.py` is the unified catalog (images + documents behind one opaque id space,
`file_img_<id>` / `file_doc_<id>`, advertised as a literal enum and re-validated at execution).

- **The old prompt LIED.** `CODE_INTERPRETER_GUIDANCE` used to say "Files attached to this
  conversation are already mounted in the sandbox's working directory". Nothing ever mounted them.
- Mounting is **lazy** (the model calls the tool) and **idempotent**, cached by
  `(container_id, file_id)` — which is exactly what makes "come back after lunch" work: a live
  container skips the upload, an expired one (new id) misses and re-mounts from Slack.
- Pushed bytes land as `source="user"`, so the publisher (assistant-only) never posts them back.
  A digest guard (`suppress_digests`) also catches a model that copies an input to a new name.
- **Cataloguing must not depend on replying.** Document/image rows used to be written only inside
  a turn, so a file shared in a message the bot stayed quiet about — or one superseded during the
  participation gate's debounce — vanished for good. `thread_files.catalog_unattended()` (called
  from `main.py`'s gate on any non-respond verdict) fixes that. Found live: a CSV dropped 2s
  before another file was lost, and the model then correctly refused to build the report.

## Publishing: only the deliverable, never its ingredients

`artifacts.py` decides what the user actually sees. Four rules, each of which shipped a bug once:
1. **Zip-member hashes** — an image embedded in a `.pptx/.docx/.xlsx` is a byte-identical zip
   entry, so the deck itself names its own ingredients. Exact, not a guess.
2. **A PDF is NOT a zip.** It re-encodes what it embeds, so rule 1 is blind to it. Hence: *if any
   DOCUMENT (pdf/pptx/docx/xlsx) is being published, loose images are its ingredients.* Scoped to
   turns that produce a document — "draw me a chart" still publishes the chart.
3. **Superseded drafts** — a model that revises leaves `Board_Ready.pdf` behind when it writes
   `Board_Brief.pdf`. Within one extension, the LAST document written is the finished one.
4. **`expect_filenames`** — a background build declares its deliverables up front, and a declared
   manifest beats every heuristic above.

## Background jobs: the model decides delivery (F37)

`start_background_job` (was `start_deep_research`) takes a `mode` — `research`, `build`, or
`research_and_build` — plus an optional `deliverables` array. `build` skips research entirely and
works from what already exists (thread files via `mount_file`); it is how "chart the CSV I posted"
is expressible at all.

**The job PRODUCES; it does not deliver.** This is the whole design, and it exists because of a
real bug: a job asked for a PDF posted the entire 21k-char findings report into the thread as
seven chunked messages — raw markdown tables and all — and *then* uploaded the PDF containing the
same report. The code had no way to know the PDF **was** the report. The model that asked for the
PDF did. Worse, the research prompt deliberately asks for markdown data tables (they are the only
input the build phase gets), so the text we posted was feedstock, not prose.

So the job now ends by calling the model back — `_plan_delivery` — with the report, a manifest of
staged files, and one tool. The model returns `{reply, publish: [artifact_id], post_report}` and
`_transact_delivery` executes it in reading order: message → report (if asked) → files.

- **Artifacts are STAGED, not published, when the build ends** (`artifacts.stage_artifacts`). The
  bytes come out of the container and wait in memory; the container is released immediately. A
  container's idle life is capped at 20 minutes — putting a model call on the critical path of an
  expiring resource loses a deliverable eventually. Staging kills that race outright.
- **Selection is by opaque `artifact_id`, never filename.** Filenames are model-authored and
  therefore hallucinable, and `publish_staged` DROPS an unknown id rather than resolving it to a
  "close enough" file. (Contrast `expect_filenames` below, which may fall back — that fallback is
  correct for a manifest heuristic and would be catastrophic as a selection contract.)
- **The findings can never silently vanish.** The report lives only in the job's memory, and Slack
  is the only transcript we keep. If the model ships no file *and* declines to post the report,
  `_transact_delivery` overrides it and posts the report anyway. This is not the model's call.
- **The delivery call gets exactly ONE tool** (`deliver`). No recursion guard is needed because
  there is no tool to recurse with — it cannot start another job, write memory, or draw an image.
- **The report reaches that call as USER-role data**, never developer-role. It is scraped off the
  open web; a developer block outranks the user, so a page saying "ignore your instructions" must
  arrive as something the job FOUND, not something the system SAID. Our instruction goes last.
- **No plan → post everything.** A model that never calls `deliver` falls back to the old
  behaviour. Noisy, but it has never lost anyone's work.
- **The build phase gets its OWN container** (`{thread_key}#job:{job_id}`), never the thread's.
  Sharing is unsafe: `containers._snapshot_baseline` marks everything already in a reused container
  as "already published", so a chat turn sent while the job was building would baseline the
  half-written deck as published and the job's own publisher would then SKIP it. The deck would
  vanish, silently, and the more the user chatted the likelier it got.
- `ledger_key` vs `thread_key`: `ledger_key` scopes the container concerns (lock + published-file
  record), `thread_key` stays the thread the DB rows belong to — so tomorrow's "revise that deck"
  can still find the file.
- `generate_image` is **excluded** from the build registry: it is detached and posts straight to
  Slack, so inside a build it would land a loose image in the thread and might arrive after the job
  ended. Only `create_image_asset` (into the sandbox) is offered.
- The card finalizes **last**, from Slack receipts — what actually posted — never from the plan's
  intent. It used to go green before the report was even posted.

### The status card is a live TODO LIST

The **dispatching** model writes the `plan` (a required arg on `start_background_job`), so the card
posts already populated — the user reads what the job intends to do at t=0, not a bare
"Researching…" until the job model's first round lands. The job then **revises** it with
`update_todos`, a rewrite-the-whole-list tool: tick an item to `in_progress`/`done`, add a step it
didn't foresee, drop one that turned out to be irrelevant.

- **`update_todos` is a FREE tool** — `free_tools` in `tool_loop.py` exempts it from the round and
  call budget. This is not a micro-optimisation: a live list fires on every transition, and on the
  meter those calls eat the budget the build phase needs for `mount_file` / `create_image_asset`.
  The card would starve the deck it is reporting on, and the loop would force a final answer before
  the file was ever built. The caps are a runaway guard; a status update is not what they guard
  against, and the wall-clock timeout is what actually bounds a detached job. Free is still
  *bounded* — free rounds AND free calls have their own ceilings, because a round's calls dispatch
  in parallel, so one "free" round could otherwise carry fifty updates.
- **`_TodoState` is the source of truth; the card only renders it.** The render is lossy on purpose
  (trims to four lines, truncates text, decorates with glyphs), so the build phase — a fresh model
  loop — is handed `as_prompt_block()`, never the card's text. Otherwise it would inherit a
  mangled, possibly front-trimmed copy of its own plan.
- **The executor validates; the schema can't.** JSON Schema cannot express "exactly one
  `in_progress`", and nothing between the model and us checks the arguments. `_TodoState.set()`
  rejects (whole, never half-applied) an empty list, >4 items, a bad status, duplicates, and two
  spinners — and hands the model back a message saying *why*, so it self-corrects next call.
- **Four lines, always.** The plan caps at **three** (`_MAX_PLAN`) because at t=0 nothing is
  `in_progress`, so the card renders the plan *plus* a "⏳ Researching…" tail — four pending items
  plus that tail is five lines, and the trim would drop the first step, the one about to start. The
  job may grow the list to four once something is spinning (the spinner replaces the tail).
- **On failure the in-flight step is pinned.** It is the one line that says where the job stopped;
  a plain "keep the last four" would evict it whenever later items had already completed.

**Known gap:** the delivery call sees the conversation as it was at *dispatch* (the snapshot), not
as it is now. A user who says "actually, don't post that" while the job runs is not heard.

## Slack canvases (F36) — `message_processor/canvas_tools.py`

Probed live, because the docs omit all of this:

**We create the CHANNEL canvas (`conversations.canvases.create`), never a standalone one.** Only
the channel canvas gets a **TAB** at the top of the channel, and the tab is the whole point: Slack
posts no message when a canvas is shared, so an untabbed canvas appears NOWHERE — not in history,
not in the transcript we rebuild from it. A standalone canvas cannot be pinned by any route:
`pins.add(timestamp=<share ts>)` → `message_not_found` (a canvas share is not a message);
`pins.add(file=…)` → `no_item_specified` (Slack dropped file pinning); `bookmarks.add(link=…)`
succeeds but comes back `type="file"`, is **not returned by `bookmarks.list`**, and
`bookmarks.remove` refuses it (`invalid_bookmark_type`) — a write-only bookmark, and no tab either.

Consequences of that choice, each of which bites:
- **`title` is UNDOCUMENTED, is absent from the slack_sdk signature, and works anyway** — the
  signature takes `**kwargs`, so it reaches the API. It labels the canvas TAB. Reading the
  signature and concluding "no title is possible" shipped a canvas called `Untitled` — which is a
  document no ask can ever match — so pass it, always. It is also the ONLY chance: **there is no
  rename, by any route.** `files.rename`, `canvases.setTitle` and `conversations.canvases.setTitle`
  are each `unknown_method`; `files.edit` is `not_allowed_token_type` for a bot. Get it right at
  creation or live with it. (`_first_heading` salvages a legacy `Untitled` canvas for the catalog
  by reading its top heading, but that is a fallback, not the mechanism.)
- **It is NOT idempotent**: a second call = a second canvas AND a second permanent tab. So
  `create_channel_canvas` is a schema FACTORY that disappears once a canvas exists (and re-checks
  live before creating), making "create if not exists" unmakeable rather than remembered.
- **`properties.canvas` is NULL even when a channel canvas exists.** The real record is
  `properties.tabs` → `{"type":"canvas","data":{"file_id":…}}`. And **a tab OUTLIVES its canvas**,
  so a tab alone is not proof one exists. But **`files.list` is eventually consistent in BOTH
  directions** — it keeps a deleted canvas for a while AND does not yet know about one created
  seconds ago — so absence from it is not proof of death either. Taking it as proof was a live bug:
  right after the bot created the agenda, the catalog dropped it, re-offered `create_channel_canvas`,
  and left `edit_canvas` with no id to aim at. `files.list` is the fast path; anything missing from
  it is settled by `files.info`, which answers `file_deleted` precisely. A fresh canvas is also
  INSERTED into the catalog, or the turn that just made it cannot then edit it.
  `_channel_canvas_id(strict=)` exists because "I couldn't tell" must not read as "there isn't one"
  on the delete path — swallowing that error turns a Slack outage into a licence to delete (a test
  caught exactly this).
- The channel canvas is **not deletable** — excluded from `delete_canvas`'s enum *and* re-checked at
  execution. Clearing it out is an edit.

**The tools were dead without a system prompt.** With all 23 tools on the table, "start a running
agenda for our devops call" produced a *chat message*. A tool description is only read once the
model has decided to reach for a tool; the reply-or-document choice happens before that. Hence
`CANVAS_GUIDANCE` (`prompts.py`). On the retry it then read an **unrelated** canvas ("Q4 Launch —
Hot Sauce") and tried to append the agenda to it — only Slack's ACL stopped a silent rewrite of
someone else's document — so the guidance also says *never write what was asked for into an
unrelated canvas just because it is the one that exists*.

Everything below applies to any canvas, the channel's or a human's:
- A canvas IS a file (`F…`). **There is no `canvases.read`** — you download `url_private` and get
  **HTML**, not the markdown you put in. `_html_to_markdown` does the round trip.
- `download_file` REJECTS an html body by default (for an image, html means auth failed and Slack
  served a login page). Canvases must pass `allow_html=True` — and then check for themselves that
  what came back is a canvas (`quip-canvas-content`) and not a login screen.
- **A "section" is ONE block** — a heading, a paragraph, a single list item. Not a region. You
  cannot replace "the Steps section" in one call. And `canvases.edit` takes exactly **one change per
  call** (`no more than 1 items allowed [json-pointer:/changes]`), so a batch is not an option.
- A replacement for a list item must **not carry its own bullet**: `- [x] beta` makes Slack parse a
  NEW list, deleting the item and appending a fresh list at the bottom. Bare text replaces in place.
  (`_replacement_for_section` strips it; a heading's `##` is kept, or it demotes.)
- `read_canvas` returns markdown, so the model quotes `- Launch lead — Dana` as `find_text` — but
  `contains_text` searches the plain TEXT, where the bullet does not exist. `_searchable` strips the
  scaffolding, or every first edit misses.

**Checkboxes work — and a box cannot be ticked in place.** An earlier probe here concluded canvases
had no checkboxes at all, and the model was TOLD so; in fact `- [ ]` / `- [x]` render as a real
checklist (`data-section-style='7'`, ticked items `<li class='checked'>`), which is what a meeting
agenda wants. But every route to toggling one item fails:

    replace(item, "- [x] beta")       item LEAVES the list; a new one-item list is appended → it
                                      silently jumps out of order
    insert_after(item, "- [x] beta")  same — a new list block, not a sibling item
    replace(item, "beta")             lands in place, but the tick state is UNTOUCHED → "mark it
                                      done" reports success and does nothing

Any markdown carrying list syntax becomes a NEW list block; only bare text lands in place. So the
unit of a tick is the **list**: `_rewrite_list` replaces the FIRST item's section with the whole new
list (which plants it right after the old container) and then deletes each leftover, so the old
container empties and vanishes and the rebuilt list ends up exactly where it was. That is the
`replace_list` operation, and `replace_section` REFUSES a checkbox rather than silently no-op.

**Markdown that renders** (probed line by line, `CANVAS_MARKDOWN_HELP`): headings, bold/italic/
strike/code, links, bullet + numbered + check lists, tables, blockquotes, code fences, `---`, and
nested lists. **Mixing list kinds when nesting is a hard failure** — a `- [ ]` under a plain `-`
returns `canvas_creation_failed: Unsupported list type (checklist) within bullet list`, so the whole
write is lost, not degraded. **Images are silently dropped.**

**ADD, don't replace.** A canvas is a record; its old entries are the point of keeping one. New
material is an `append`/`prepend`, and a recurring meeting is a rolling log — the new date's section
is PREPENDED so the newest sits on top and every past meeting survives below it. `replace_section` /
`replace_list` are only for changing something specific that already exists (tick a box, correct a
figure). This is in `CANVAS_GUIDANCE` and in the `edit_canvas` description, because a model asked to
"add next week's agenda" will otherwise happily rewrite the document.

**The date chip is client-only.** `document_content` accepts markdown and NOTHING else (`html` and
`rich_text` both fail schema validation), and every date syntax — Slack's `<!date^…>`, `<time>`,
`[[date:]]` — comes back escaped as literal text. Only a person in the app can insert a real chip,
so the bot writes the date as a plain heading. It must still READ them: a chip comes back as
`<control data-remapped="true">`, and users do chip the headings of the canvases the bot edits — drop
the element and the bot reads a dated heading as blank, then "helpfully" rewrites it.

**The HTML you read back is not the HTML you'd guess**, and reading it wrong matters because the
model edits what it reads: a list's KIND lives on the container (`data-section-style` 5/6/7), not
the item — an unticked item carries no marker at all, so keying off the item made a checklist read
back as plain bullets. Links come back as `<lnk>`, not `<a>`. A code block is `<p class="prettyprint">`,
not `<pre>`. Table cells hold `<p>`, so walking every `<p>` shreds a table into loose paragraphs.
- Private channels report in `groups`, not `channels` (and `shares.private`). Checking only
  `channels` refuses every edit in a private channel.
- `delete_canvas` exists but is withheld on turns nobody ADDRESSED the bot on (`_addressed_turn`,
  not `_unprompted_turn` — a name-hit is prompted in spirit), and never offers the channel canvas.
  A canvas the bot did not create refuses deletion anyway (`restricted_action`).

## Scanned / image-only PDF handling

- On the attach turn, PDFs within the native limits ride the message as `input_file` parts, so the
  model sees rendered pages directly (scans are readable).
- On LATER turns content is re-derived by local extraction. For image-only/scanned PDFs
  (`_is_image_based_pdf`) local text extraction yields nothing, so `document_handler.ocr_pdf_pages`
  renders pages with poppler and OCRs them with tesseract into page-structured text (`[Page N]`
  blocks). The `read_document` tool requests `ocr_text=True`, so scans readable on the attach turn
  stay readable later.
- Gated by `ENABLE_PDF_OCR` (default on), bounded by `OCR_MAX_PAGES` (loud truncation note beyond
  it) at `OCR_DPI`. OCR runs in the extraction executor thread (it is subprocess+CPU heavy) and
  degrades gracefully: missing tesseract/poppler or any OCR error falls back to the honest "scanned
  document" note, never raising.

## The mock-stream OOM (why the vision guards exist)

A mock stream MUST terminate and MUST yield real strings. `output_text += chunk` where `chunk` is a
MagicMock does not raise: `str.__add__` returns `NotImplemented`, Python falls back to
`MagicMock.__radd__`, and `output_text` silently BECOMES a mock that retains the previous one. Every
`+=` then builds another. A stale two-item `side_effect` left `analyze_images` async-iterating a bare
MagicMock — which never ends — the suite grew to **30GB** and OOM-killed the dev box. An unconsumed
`side_effect` entry is not inert: it is the next call's input. `openai_client/api/vision.py` now
guards both (non-str delta → stop; `_MAX_ANALYSIS_CHARS` ceiling).
