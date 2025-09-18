# Async/Await Refactor Plan - Fix for Bot Hanging Issues

## ✅ REFACTOR COMPLETED - 2025-09-17

All components have been successfully converted to async/await architecture. The bot now uses an event loop instead of worker threads, eliminating the thread exhaustion issues that caused hanging.

## Problem Summary

### The Issue
The Slack bot hangs and becomes unresponsive after handling multiple OpenAI timeout errors. When users send rapid messages or when OpenAI API is slow/unresponsive, the bot stops processing new messages entirely, requiring manual restart via PM2.

### Root Cause Analysis

1. **Worker Thread Exhaustion**
   - Slack Bolt's SocketModeHandler uses a thread pool with 10 worker threads (default concurrency)
   - Each incoming Slack event is assigned to a worker thread
   - The worker thread calls `handle_message()` which **synchronously** calls OpenAI API
   - With 305-second timeout configured, each stuck request blocks a worker for 5+ minutes
   - After 10 stuck requests, all worker threads are occupied = bot completely hangs

2. **Force-Release Lock Corruption**
   - Current watchdog tries to force-release locks after timeout
   - This creates corruption when:
     - Thread A holds lock and is waiting for OpenAI
     - Watchdog force-releases the lock
     - Thread A returns and tries to release already-released lock
     - Results in `RuntimeError: release unlocked lock` or inconsistent state

3. **Sequential Processing Within Threads**
   - Thread locks with `timeout=0` (non-blocking) cause busy responses
   - This is correct behavior (one message per Slack thread at a time)
   - But combined with worker exhaustion, it appears fully sequential

## Solution: Convert to Async/Await Architecture

### Benefits
1. **No Worker Thread Exhaustion**: Event loop can handle thousands of concurrent operations
2. **Clean Timeout Handling**: `asyncio.wait_for()` provides cooperative timeouts
3. **No Force-Release Needed**: Async locks timeout naturally without corruption
4. **Better Resource Utilization**: One event loop vs 10 blocked threads
5. **Maintains All Current Functionality**: Thread locking, busy states, etc. remain the same

### Architecture Change

**Current (Synchronous)**:
```
Slack Event → Worker Thread → Blocking OpenAI Call (305s) → Response
              (1 of 10)        (Thread blocked)
```

**New (Asynchronous)**:
```
Slack Event → Event Loop → Async OpenAI Call → Response
              (∞ capacity)  (Yields control)
```

## Implementation Plan

### Pre-Work: Inventory Synchronous Touchpoints

#### Current Sync Touchpoints (Updated 2025-09-17)

- `main.py`: message handler and cleanup loop are synchronous; uses `threading.Thread`, blocking `time.sleep`, and invokes sync client/database layers.
- `slack_client/` mixins: built on `slack_bolt.App` with sync handlers; all Slack WebClient calls (`views_open`, `chat_postMessage`, file download) block the worker threads.
- `message_processor/`: `process_message` uses blocking `thread_manager`, synchronous OpenAI calls, direct sqlite access, background `threading.Thread` progress updaters, and `time.sleep` paths in image handling.
- `openai_client/`: wraps `OpenAI` sync client and exposes only blocking create/stream methods with manual timeout wrappers.
- `thread_manager.py`: relies on `threading.Lock`, watchdog thread, and synchronous cleanup; exposes blocking `acquire_*` APIs.
- `database.py`: uses `sqlite3` with autocommit connection accessed synchronously from multiple threads.
- `settings_modal.py`: synchronous DB lookups and Slack interactions triggered inside request threads.
- `document_handler.py`: runs CPU-heavy parsers synchronously on request threads; safe move to `asyncio.to_thread` when wiring async pipeline.
- `image_url_handler.py`: performs network I/O via `requests` in blocking fashion for validation and download.
- `base_client.py` & derivatives: contract is synchronous (`send_message`, `send_image`, etc.), which propagates blocking semantics throughout the app.
- Tests (`tests/unit`, `tests/integration`): assume synchronous interfaces and patch threading constructs; will need async fixtures and `pytest-asyncio` coverage once refactor lands.


- Catalog every Slack handler, mixin, and shared utility that currently performs blocking work (Slack WebClient calls, database/file operations, OpenAI requests).
- Map call graphs to understand which functions and helpers will require signature changes before converting to async.
- Flag thread-based helpers, watchdog behaviors, or other concurrency constructs slated for removal in the async migration.

### Phase 1: Core Infrastructure Changes

#### 1. Update Slack Client (`slack_client/` package)
- [x] Import `AsyncApp` / `AsyncSocketModeHandler` in `slack_client/base.py`
- [x] Update `SlackBot` to inherit async mixins and store `AsyncApp`
- [x] Convert mixins to async:
  - [x] `slack_client/event_handlers/registration.py` → async Slack registration (delegate settings hook to async)
  - [x] `slack_client/event_handlers/settings.py` → async slash-command, modal, action handlers
  - [x] `slack_client/event_handlers/message_events.py` → async message ingestion and welcome flow
- [x] Update supporting mixins:
  - [x] `slack_client/messaging.py` → async send/update/delete/history methods using `await self.app.client.*`
  - [x] `slack_client/utilities.py` → async user/file helpers (use `await client.users_info` etc.)
  - [x] `slack_client/formatting/text.py` remains sync (pure string ops)
- [x] Update `SlackBot.start/stop` to use async socket mode handler
- [x] Ensure all Slack API calls across mixins await the async WebClient methods
- [x] Update any remaining direct Slack client usage in other modules (e.g., settings modal) to async equivalents

#### 2. Update Main Entry Point (`main.py`)
- [x] Convert `handle_message()` to `async def handle_message()`
- [x] Update all internal calls to use await
- [x] Modify `run()` method to use `asyncio.run()` for the main loop
- [x] Update cleanup thread to use asyncio tasks instead of threading
- [x] Convert signal handlers to async-safe operations

#### 3. Update Message Processor (`message_processor/` package)
- [x] Convert `message_processor/base.py::process_message` to async and update caller contract
- [x] Update mixins to async:
  - [x] `thread_management.py` (locks, cleanup)
  - [x] `utilities.py` (attachment processing, prompt building, Slack status updates)
  - [x] `handlers/text.py`, `handlers/vision.py`, `handlers/image_gen.py`, `handlers/image_edit.py`
- [x] Ensure all OpenAI / DB calls inside mixins use awaitables
- [x] Replace threading-based progress/updater logic with asyncio tasks
- [x] Propagate async signatures to any helper methods invoked externally

#### 4. Update OpenAI Client (`openai_client/`)
- [x] Change to use async OpenAI client:
  ```python
  from openai import AsyncOpenAI
  self.client = AsyncOpenAI(...)
  ```
- [x] Convert all API methods to async:
  - [x] `client.chat.completions.create()` → `client.chat.completions.acreate()`
  - [x] `client.images.generate()` → `client.images.agenerate()`
  - [x] `client.models.list()` → `client.models.alist()`
- [x] Update streaming to use async generators:
  ```python
  async for chunk in response:
      yield chunk
  ```
- [x] Implement proper timeout with `asyncio.wait_for()`:
  ```python
  try:
      response = await asyncio.wait_for(
          self.client.chat.completions.acreate(...),
          timeout=305
      )
  except asyncio.TimeoutError:
      # Clean timeout handling
  ```
- [x] Remove any timeout_wrapper decorators (not needed with async)

#### 5. Update Thread Manager (`thread_manager.py`)
- [x] Convert `ThreadLockManager` to use `asyncio.Lock()` instead of `threading.Lock()`
- [x] Convert all lock operations to async:
  - [x] `acquire_thread_lock()` → `async def acquire_thread_lock()`
  - [x] `release_thread_lock()` → `async def release_thread_lock()`
  - [x] `get_lock()` → `async def get_lock()`
- [x] **Remove force-release mechanism entirely** (not needed with async timeouts)
- [x] Update watchdog to be async task (only for monitoring/logging):
  ```python
  async def _watchdog_task(self):
      while True:
          await asyncio.sleep(10)
          # Check for stuck operations (logging only, no force-release)
  ```
- [x] Convert cleanup operations to async

#### 6. Update Database Manager (`database.py`)
- [x] Add aiosqlite as dependency
- [x] Create async versions of all database methods:
  - [x] Use `aiosqlite.connect()` instead of `sqlite3.connect()`
  - [x] All queries become `await cursor.execute()`
  - [x] All fetches become `await cursor.fetchall()`
- [x] Implement async context managers:
  ```python
  async with aiosqlite.connect(self.db_path) as db:
      async with db.cursor() as cursor:
          await cursor.execute(...)
  ```
- [x] Update connection pool to be async-safe
- [x] Maintain transaction integrity with async commits

### Phase 2: Supporting Components

#### 7. Update Base Client (`base_client.py`)
- [x] Make abstract methods async-compatible
- [x] Update Response and Message classes if needed
- [x] Convert send/update/delete methods to async

#### 8. Update Settings Modal (`settings_modal.py`)
- [x] Convert view submission handlers to async
- [x] Update database calls to use async versions
- [x] Convert modal building methods if they do I/O

#### 9. Update Document Handler (`document_handler.py`)
- [x] Convert file reading operations to async
- [x] Use aiofiles for async file I/O
- [x] Update PDF/document processing to async where possible

#### 10. Update Utilities
- [x] `token_counter.py` - Keep synchronous (pure computation)
- [x] `prompts.py` - Convert if any OpenAI calls exist
- [x] `markdown_converter.py` - Keep synchronous (no I/O)
- [x] `image_url_handler.py` - Convert download methods to async using aiohttp

### Phase 3: Testing & Dependencies

#### 10. Update Requirements (`requirements.txt`)
- [x] Add async dependencies:
  ```
  aiosqlite>=0.19.0
  aiofiles>=23.0.0
  aiohttp>=3.9.0
  pytest-asyncio>=0.21.0
  ```
- [ ] Verify all libraries support async:
  - Slack Bolt: ✅ (has AsyncApp)
  - OpenAI SDK: ✅ (has AsyncOpenAI)
  - SQLite: ✅ (via aiosqlite)

#### 11. Update Tests
- [ ] Convert test fixtures to async where needed
- [ ] Update mock objects for async methods:
  ```python
  @pytest.mark.asyncio
  async def test_process_message():
      async with mock.patch('openai.AsyncOpenAI'):
          result = await processor.process_message(...)
  ```
- [ ] Add timeout scenario tests:
  ```python
  async def test_timeout_handling():
      with mock.patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
          # Verify graceful handling
  ```
- [ ] Create load tests for concurrent requests

### Phase 3: Configuration & Deployment

#### 11. Configuration Updates
- [ ] No changes needed to `.env` file
- [ ] Keep 305-second timeout (it will work properly with async)
- [ ] Consider adding:
  ```env
  ASYNC_MAX_CONCURRENT_REQUESTS=100
  ASYNC_TIMEOUT_SECONDS=305
  ```

#### 12. Migration Strategy
- [ ] Create feature flag for async mode:
  ```python
  USE_ASYNC_MODE = config.get('USE_ASYNC_MODE', 'false').lower() == 'true'
  ```
- [ ] Run both implementations in parallel initially
- [ ] Gradual rollout:
  1. Dev environment testing
  2. Enable for specific test channels
  3. Monitor metrics
  4. Full production rollout
- [ ] Keep sync version branch for emergency rollback

## Key Code Patterns

### Event Handler Pattern
**Before (Synchronous)**:
```python
@app.event("message")
def handle_message(event, say, client):
    self._handle_slack_message(event, client)
```

**After (Asynchronous)**:
```python
@app.event("message")
async def handle_message(event, say, client):
    await self._handle_slack_message(event, client)
```

### OpenAI API Call Pattern
**Before (Synchronous)**:
```python
def generate_response(self, messages):
    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        timeout=305
    )
    return response
```

**After (Asynchronous)**:
```python
async def generate_response(self, messages):
    try:
        response = await asyncio.wait_for(
            self.client.chat.completions.acreate(
                model=self.model,
                messages=messages
            ),
            timeout=305
        )
        return response
    except asyncio.TimeoutError:
        await self.handle_timeout()
        raise
```

### Database Operation Pattern
**Before (Synchronous)**:
```python
def get_thread_history(self, thread_id):
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE thread_id = ?", (thread_id,))
    results = cursor.fetchall()
    conn.close()
    return results
```

**After (Asynchronous)**:
```python
async def get_thread_history(self, thread_id):
    async with aiosqlite.connect(self.db_path) as db:
        async with db.cursor() as cursor:
            await cursor.execute("SELECT * FROM messages WHERE thread_id = ?", (thread_id,))
            results = await cursor.fetchall()
            return results
```

## Important Considerations

### 1. Mixing Sync and Async
- **Never** call sync functions from async context without `run_in_executor()`
- **Never** call async functions from sync context without `asyncio.run()`
- Use `asyncio.to_thread()` for CPU-bound sync operations

### 2. Database Transactions
- Ensure proper async transaction handling
- Use async context managers for automatic cleanup
- Handle connection pool limits

### 3. Error Propagation
- Async exceptions propagate differently
- Use try/except at appropriate levels
- Ensure errors reach user feedback

### 4. Testing
- All tests involving async code need `@pytest.mark.asyncio`
- Mock async operations properly
- Test concurrent scenarios

### 5. Backwards Compatibility
- External behavior must remain identical
- API endpoints stay the same
- Database schema unchanged

## Success Criteria

1. **Bot handles 50+ rapid-fire messages without hanging**
   - Test: Send 50 messages in quick succession
   - Success: All messages processed or explicitly failed

2. **Timeouts don't cause worker thread exhaustion**
   - Test: Trigger 15 OpenAI timeouts
   - Success: Bot remains responsive to new requests

3. **No force-release corruption issues**
   - Test: Monitor logs for lock errors
   - Success: Zero lock-related exceptions

4. **Maintains all current functionality**
   - Test: Full regression test suite
   - Success: All existing tests pass

5. **Performance improvement in high-load scenarios**
   - Test: Benchmark with concurrent requests
   - Success: 2x throughput improvement

## Monitoring & Metrics

Track these metrics before/after:
- Response time percentiles (p50, p95, p99)
- Concurrent request handling capacity
- Memory usage under load
- Timeout recovery success rate
- Error rates by type

## Rollback Plan

If issues arise:
1. Set `USE_ASYNC_MODE=false` in environment
2. Restart bot with PM2: `pm2 restart slackbot`
3. Git checkout to sync version branch if needed
4. All async changes are internal - no database or external API changes

## Timeline

- **Day 1-2**: Complete modularization (prerequisite)
- **Day 3-4**: Core async infrastructure (Phases 1-2)
- **Day 5**: Testing and validation (Phase 3)
- **Day 6**: Deployment preparation (Phase 4)
- **Day 7**: Production rollout and monitoring

## Next Steps

1. Complete modularization first (see `modularization-plan.md`)
2. Start with Phase 1, component by component
3. Test each component individually
4. Integration test after each phase
5. Full system test before deployment

---

*This plan created on 2025-09-16 to fix bot hanging issues discovered during rapid-fire message testing.*
*Root cause: Synchronous OpenAI calls blocking Slack Bolt worker threads for 305+ seconds.*
