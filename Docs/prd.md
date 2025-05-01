# Slackbot V2 PRD

Additional context found in the "additional_context.md" file. Refer to this for some code examples and requirements.
# VERY IMPORTANT: The current code base is a fully working "V1" version of the production application. The instructions in the refactor overview are meant as the steps to make changes that will result in a V2. Work accordingly and be mindful of what you're changing. At all times, the app should start and be functional. Make sure to review the code base before assuming you need to do something or generate tasks for redundant work. E.g., .env file already exists. No need to recreate it.

### Note: We're not supporting any streaming features or clients even though OpenAI supports it.
For all tests, test against the real third party APIs when possible using the Dev env tokens.

## Project Overview
We are building V2 of the Slackbot project focused exclusively on the Slack client, with full architectural refactor for modularity, scale, and maintainability.  
This version will:
- Use continue to use the Slackbolt SDK
- Transition from OpenAI Chat Completions API to the Responses API
- Implement SQLite (Dockerized) for persistent config management
- Move the environment to a clean Docker Compose setup

---

# 🧩 Architecture Overview

## 0. Core Principles
0.1 Modular design: Core utilities shared across clients, with client-specific code isolated.  
0.2 Object-Oriented Programming (OOP) design patterns.  
0.3 Extensible structure to easily add future clients (Discord, CLI).  
0.4 Use of latest APIs (Slackbolt, OpenAI Responses API).  
0.5 Docker-first deployment.  
0.6 Persistent configuration storage via SQLite.  
0.7 Dev and Prod environments isolated through `.env` file variables. (.env exists already pre-populated)
0.8 Test suite tasks included for each feature.

## 0.9 Proposed Modules
- `/clients/`
  - `slack_client.py`
- `/core/`
  - `bot_functions.py`
  - `conversation_manager.py`
  - `config_manager.py`
  - `image_manager.py`
  - `nlp_parser.py`
  - `common_utils.py`
- `/database/`
  - `db_models.py`
  - `db_manager.py`
- `/infra/`
  - `docker-compose.yml`
  - `Dockerfile`
- `/logging/`
  - `logger.py`
- `/queue/`
  - `queue_manager.py`
- `/tests/`
- `/config/`
  - `.env` (already exists)

---

# 🛠️ Phase 1: Base System Setup

## 1. Environment and Deployment
1.1 Dockerize application using `Dockerfile` and `docker-compose.yml`.
- Python 3.12+
- Docker container for the app
- Docker container for SQLite (unless determined better to embed SQLite directly without a service)

1.2 Use existing `.env` for Dev/Prod variables.

1.3 Run inside a virtual environment (`.venv`) inside the Docker container.

1.4 PM2 no longer used; Docker manages lifecycle.

**Test Task:**  
- Confirm containers start correctly and Slack bot connects to workspace (Dev tokens).

---

# 🧠 Phase 2: Core Refactors

## 2. Slack Client (Slackbolt SDK)

**Refactor Goals:**
2.1 Slack-specific code isolated to `slack_client.py`.  
2.2 All message parsing, thread management, and event handling reside here.

**New Behavior:**
2.3 Track thread_ts ↔ previous_response_id in SQLite rather than local python dict

**Test Task:**  
- Confirm new conversation starts successfully.
- Confirm revived thread correctly rebuilds and sends prior conversation.
- Confirm thread_ts and previous_response_id is properly correlated in DB

---

## 3. Conversation Manager (OpenAI Responses API)

**Refactor Goals:**
3.1 Replace Chat Completions API with Responses API.  
3.2 Maintain conversation flow using `previous_response_id`.  
3.3 Keep Slack as the Source of Truth for conversations.
3.4 Remove local conext managers in favor of remote conversation tracking with previous_response_id.

**Implementation Details:**
3.5 On revived conversation, send full rebuilt context to OpenAI.  
3.6 Store previous_response_id with thread_ts in DB.  
3.7 Maintain lightweight local cache only if needed.
3.8 Refactor OpenAI API calls based on responses API shape. use context7

**Test Task:**  
- Validate previous_response_id chaining works.
- Confirm lost conversations are restored when threads revive.

---

## 4. Persistent User/Thread Configuration (SQLite)

**Database Tables:**
4.1 `users` (id, slack_id, first_name, config_options)  
4.2 `threads` (id, thread_ts, previous_response_id, channel_id, user_id, config_options)

**Config Options Stored:**
4.3 Number of images to generate  
4.4 Image model choice (dalle-3 or gpt-image-1)  
4.5 Any other per-thread settings (expandable), e.g., model parameters.

**Test Task:**  
- Set and retrieve per-user and per-thread config options successfully.

---

# 🎨 Phase 3: New Feature Implementations

## 5. Configuration Management

**New Behavior:**
5.1 NLP parsing of prompts to detect configuration changes.  
5.2 If a user asks for "5 images", auto-update config.  
5.3 Manual config setting through Slack Slash Command:
  - Command: `/chatgpt-config-dev`
  - Command parameterized via `.env` (already exists in .env)

**Test Task:**  
- Confirm NLP-detected settings are saved.
- Confirm slash command manually updates config.

---

## 6. Image Generation Management

**Changes:**
6.1 Support both dalle-3 and gpt-image-1.  
6.2 Determine whether to trigger image generation via fast LLM check (gpt-4o-mini).  
6.3 Manage UX around 20 sec (dalle-3) vs 2 min (gpt-image-1) latency.

**New Features:**
6.4 Support image+text input for gpt-image-1 model.  
6.5 If file is uploaded along with a prompt, detect and pair.  
6.6 Slack thread is locked (queued) while generation is in progress.

**Test Task:**  
- Confirm text-only, image-only, and text+image generations work.
- Confirm user blocked from further messages while image generation pending.

---

# 🧹 Phase 4: Utility Improvements

## 7. Multi-User Conversations

**Features:**
7.1 Track which user said what inside threads.  
7.2 Replace Slack IDs in messages with `[user=First Name]` tags.  
7.3 Strip `[user=...]` tags before sending back responses.  
7.4 Allow OpenAI to refer to users by their first names contextually.

**Test Task:**  
- Confirm SlackIDs are replaced with tags.
- Confirm system prompt guides bot to use first names when appropriate.
- Confirm no tags leak back into output.

---

## 8. Utility Model (Fast Prompt Classifier)

**Details:**
8.1 Use UTILITY_MODEL (`gpt-4o-mini`) to quickly determine:
- Is the prompt an image request? (Y/N)

**Test Task:**  
- Confirm classifier model correctly returns deterministic Boolean.
- Confirm routes prompt to either text or image flow accordingly.

---

# 🧰 Phase 5: API Enhancements

## 9. OpenAI Responses API Tool Support

**Optional Enhancements:**
9.1 Plan ahead to allow calling OpenAI tools:
- Web search
- File reading

9.2 Not implemented immediately but scaffold API interaction design for future.

**Test Task:**  
- (Future) Validate if OpenAI tool calls are triggered properly.

---

# 🖼️ Phase 6: UX Enhancements

## 10. Slack Block Kit Integration

**Potential Uses:**
10.1 Loading indicators during image generation  
10.2 Buttons for user-config updates  
10.3 Richer formatted outputs (images, metadata)

**Test Task:**  
- Design basic loading block during long generations.
- Confirm interaction does not interrupt conversation flow.

---

# 🧪 Phase 7: Testing and Validation

## 11. Testing Strategy
11.1 Every Feature Must Include:
- Unit tests (if applicable)
- Functional tests through Slack Dev workspace
- Confirm Dev tokens used, not Prod
- Validate conversation restore, config changes, image generation



Updates to tasks:
Task 03. - Bot_functions.py already exists. Don't create a new one, rather read the file to understand what is there and how to refactor it. Adjust subtasks accordingly.
Task 04 - Slackbot.py client already exists, same as above, with bot_functions.py. Don't create a new one, refactor the old one as necessary. Don't delete any features or code unless you replace the functionality. Just modify according to the refactor plan. Move the files around as needed to fit the new project structure (goes for task 03 as well).
Task 05 - Note that currently these functions exist in the bot_functions.py, so work may be tied to that task unless you're refactoring to new files/modules.
Task 06 - Reminder that there is no history management locally. See the refactor overview for these details. History management is maintained on the openai side with the responses api and the "previous_response_id". The configuration in the new DB should be for per user and/or per-thread settings. Also it should contain associations between thread_ts (slack conversation IDs and OpenAI "previous_response_id"
Task 07 - Queue manager already exists and is working and is not just for images, but for all activies/events.  Only work regarding queue may be to move it or refactor slightly. For now, we will just deal with the long image gen request time with the queue locks.
Task 08 - Just a note here that the existing model logic is fairly sound. It just needs to be broken up to suport both models individually using the new image creation API. Use web search to find the new api doc reference.
Task 10 - The current Slack client handles image attachments (for vision). We just need to allow images and text to be sent to the new image gen model (gpt-image-1). If the model selected is dalle3 and the initial request is determined to be a request for an image,and both text and images exist in an event, the user should be notified that dalle3 doesn't support image gen based on other images. Recommend switching to the newer model instead. If the initial request is determined to not be an image request and the request has both images and text, it means its a vision request. 



