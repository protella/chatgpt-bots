# Slackbot V2 — Phased Task List

> ✱ Cursor does *not* commit to Git. Tests are run locally. Docker Compose is used as the runtime shell.
> The environment file `.env` is the only external config aside from the prompts and must be mounted into the container.

---

## Phase 0 – Bootstrap Environment

- [x] **Create Folder Structure**
  - `app/` as main package.
  - `core/` for shared logic.
  - `clients/slack/` for Slack-only code.
  - `tests/unit/` and `tests/integration/`.

- [x] **Dockerfile Setup**
  - Base image: `python:3.12-slim`
  - Non-root user.
  - Install deps via pip from `requirements.txt`.
  - Default entrypoint is a script like `start.sh`.

- [x] **Docker Compose Setup**
  - Mount `.env` from project root.
  - Named volume: `sqlite_data:/data`.
  - Service: `bot`, image build from local Dockerfile.
  - Container command: `python app/clients/slack/slack_bot.py`

- [x] **Initial Tools**
  - Add `requirements.txt`:
    - `slack-bolt==1.23.0`
    - `openai==1.76.0`
    - `aiosqlite`
    - `pytest`, `ruff`, `mypy`
  - Add `Makefile`:
    - `make build` – docker build
    - `make run` – docker compose up
    - `make test` – docker run with `pytest`

- [x] **Test:** run `make test` inside container, ensure 0 tests and exit code 0.

---

## Phase 1 – Core Utilities

- [x] **Logging**
  - Implement `core/logging.py`
  - Rotating file logs (10MB, 5 backups)
  - Console toggle via `CONSOLE_LOGGING_ENABLED`

- [x] **Thread Lock Manager**
  - Refactor `queue_manager.py` into `core/queue.py`
  - Thread-safe `start_processing_sync()`, `finish_processing_sync()` for Slack

- [x] **Test:** race condition where thread is locked — only one call processes

---

## Phase 2 – OpenAI API Integration

- [x] **Responses API Adapter**
  - `chatbot.py`: `get_response` handles text, images, and vision in one unified method
  - Handles `previous_response_id` to maintain conversation context
  - Tracks usage tokens

- [x] **Test:** mock OpenAI and test valid/invalid usage

---

## Phase 3 – Slack Event Handling

- [x] **SlackBot Entrypoint**
  - Create `clients/slack/slack_bot.py`
  - Connect using `App(token=SLACK_BOT_TOKEN)`
  - Respond to `app_mention`, `message.im`, `/chatgpt-config-dev`

- [x] **History Rebuild Logic**
  - Add `core/history.py`
  - On new event with unknown thread_ts:
    - Fetch all messages in thread
    - Extract text, images
    - Rebuild initial OpenAI message payload

- [x] **Test:** trigger a mock thread rebuild and check token format

---

## Phase 4 – Intent & Config

- [ ] **Intent Detection Service**
  - `intent_service.py`: classify intent as text, image, or vision
  - Uses `gpt-4.1-mini-2025-04-14`
  - Always returns True/False string (no extra text)

- [ ] **Config Layer**
  - `config.py`: load default from `.env`, override per `(user_id, thread_id)`
  - Store in `sqlite3` at `/data/config.db`
  - NLP stub: extract `number of images`, `style`, etc. from message

- [ ] **Slash Command Modal**
  - Triggered via `/chatgpt-config-dev`
  - Display: number of images, model type, image size

- [ ] **Test:** NLP correctly updates config, persists through restart

---

## Phase 5 – Vision and Image Gen

- [ ] **Image Handling Pipeline**
  - `image_service.py`
  - Detects whether to use GPT-Image-Gen or if a message is just text chat and conversational (model name: "gpt-image-1")
  - Respects configuration as whether to use new GPT img gen or Dallee. Defaults to new GPT Image gen.
  - Handles revised prompts (For Dalle-3 only)

- [ ] **Vision Input Handling**
  - Accept up to 4 images in allowed formats
  - Convert Slack URLs to B64 on first load only

- [ ] **B64 Constraints**
  - Images **never** passed unless required by OpenAI
  - All Slack uploads referenced by URL after initial use

- [ ] **Test:** simulate 2min latency + Slack file preview behavior

---

## Phase 6 – Final Touches

- [ ] **Busy Messaging**
  - Single inflight task per thread
  - Send BlockKit busy card or text fallback

- [ ] **User Personalization**
  - Insert `[username=user]` inline token into GPT payload
  - Let GPT decide if name should be used

- [ ] **Test:** multiple users in one thread, ensure role attribution

---

## Phase 7 – Prompt File Management

- [ ] **Prompt Definitions File**
  - Keep `prompts.py` at root or move to `core/prompts.py`
  - Ensure it defines:
    - `SLACK_SYSTEM_PROMPT`
    - `IMAGE_CHECK_SYSTEM_PROMPT`
    - `IMAGE_GEN_SYSTEM_PROMPT`
  - Replace deprecated or outdated text (e.g., DALL-E 3 only, revise to cover GPT-Image-Gen usage)
  - Ensure prompts are passed into `ChatBot` as config

- [ ] **Admin Configuration Support**
  - Prompts should be editable in a way that allows config override (or file reload)
  - Consider adding prompt override path via `.env`

- [ ] **Test:** prompt substitutions propagate to active bot conversations

---

## Phase 8 – Test Coverage & Lint

- [ ] Run all tests with `pytest --cov=app`
- [ ] Achieve > 80% coverage
- [ ] Run `ruff check --fix .`
- [ ] Run `mypy app/`
- [ ] Output coverage report to `dist/`