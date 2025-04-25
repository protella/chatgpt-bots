# Technical Design Document — Slackbot V2

## 1. Overview
This document outlines the architecture and design decisions for refactoring the ChatGPT Slackbot project into a modular, scalable, and testable version 2 (V2). This version drops CLI and Discord support initially and focuses exclusively on Slack with improved architecture, modern API usage, and production-readiness.

---

## 2. Project Goals
- Simplify and modularize the codebase.
- Migrate from OpenAI Chat Completions API to the new Responses API.
- Use OpenAI conversation memory (`previous_response_id`) instead of manual history storage.
- Fully containerize using Docker and Docker Compose.
- Support Slack message threads, image uploads, GPT Vision, and the new GPT-Image-Gen.
- Add support for per-thread config via SQLite.
- Minimize B64 image handling — only use when uploading new images to OpenAI.
- Use Slack file URLs and metadata for persisted images when rebuilding conversations.
- Include rich UX via Slack Block Kit.

---

## 3. Primary Models
- Text, Vision, and Image Gen: `gpt-4.1-2025-04-14`
- Intent Classification: `gpt-4.1-mini-2025-04-14`

---

## 4. Directory Structure
app/
├── core/
│   ├── chatbot.py             # OpenAI API interface
│   ├── config.py              # Per-thread user config via SQLite
│   ├── events.py              # Domain events / message types
│   ├── image_service.py       # GPT-Image-Gen + DALL-E 3 support
│   ├── intent_service.py      # Classifier for image requests
│   ├── history.py             # Slack history reconstruction helpers
│   ├── queue.py               # Thread lock manager
│   └── logging.py             # Standardized logging setup
├── clients/
│   └── slack/
│       └── slack_bot.py       # SlackBolt event handlers and routing
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── docker-compose.yml
└── .env                       # Single source of truth (mounted in container)

---

## 5. Core System Behaviors

### 5.1 Conversation Management
- Slack thread_ts is used as conversation ID.
- On new message:
  - If thread exists: load previous_response_id.
  - If new thread: rebuild entire thread using Slack API, then initiate OpenAI Responses conversation.

### 5.2 Image Uploads
- Users and bot-uploaded images live on Slack (90-day retention).
- Slack requires bearer token from .env as a header value
- On history rebuild: download + encode only once per image.
- B64 image passing to OpenAI only done when vision/image gen occurs.

### 5.3 Config Storage
- Lightweight SQLite DB stores per-thread config.
- Auto-update config via NLP parsing of incoming messages (e.g., "generate 5 images").

### 5.4 Intent Classification
- Every message is passed through a deterministic classifier using GPT-4.1 Mini to determine if the request is:
  - Text
  - Image generation
  - Vision analysis

### 5.5 Personalization
- Multi-user threads supported.
- Inject `[username=user]` token into GPT input — model decides if it should personalize response.

---

## 6. Slack Features
- `/chatgpt-config-dev`: opens Block Kit modal to view and update config.
- Block Kit loaders for long-running tasks.
- Rich errors and busy messages when thread is locked.
- Command-only: `!help`

---

## 7. Deployment & Environment
- Containerized with Docker (python:3.12-slim)
- Single .env file in repo root for both dev and prod values.
- Local testing via `docker compose run bot bash`

---

## 8. Security & Compliance
- No logs or files written outside container.
- Slack tokens and OpenAI keys in `.env`; never stored long-term.
- OpenAI logs expire after 30 days; Slack retains content for 90 days.
- All persistent config via SQLite volume (`/data`).

---

## 9. Test Strategy
- Unit: intent detection, config ops, image routing.
- Integration: Slack event mocks, SQLite persistence, OpenAI mocks.
- All run locally via `pytest --cov`


## Reference Doc links:
- Chat completions vs Responses API: https://platform.openai.com/docs/guides/responses-vs-chat-completions
- Text and Prompting: https://platform.openai.com/docs/guides/text?api-mode=responses
- Image gen: https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1
- Vision: https://platform.openai.com/docs/guides/images-vision?api-mode=responses
