# Streaming Implementation Plan for Slack Bot

## Overview
This document outlines the implementation plan for adding streaming support to the Slack bot, allowing real-time display of GPT responses as they're generated.

## Architecture Goals
1. **Platform-agnostic core**: Streaming logic should work with future platforms (Discord, Teams, etc.)
2. **Rate limit aware**: Respect Slack's Tier 3 rate limits (50 requests/minute for chat.update)
3. **Graceful degradation**: Fall back to non-streaming if rate limited
4. **Workspace-friendly**: Minimize impact on other workspace users and integrations

## Key Components

### 1. StreamingBuffer Class
**Location**: `/streaming/buffer.py`

Responsibilities:
- Accumulate text chunks from OpenAI SSE stream
- Determine when to trigger updates (time-based)
- Handle markdown fence closing for incomplete code blocks
- Provide display-safe text for updates

Key Methods:
```python
- add_chunk(text: str)
- should_update() -> bool
- get_display_text() -> str  # Returns text with closed fences
- get_complete_text() -> str  # Returns accumulated text as-is
- reset()
```

### 2. Rate Limit Manager
**Location**: `/streaming/rate_limiter.py`

Responsibilities:
- Track update intervals and enforce rate limits
- Handle 429 responses with exponential backoff
- Honor Slack's `retry-after` header
- Implement circuit breaker pattern for workspace protection

Key Features:
- Base interval from config (default 2.0 seconds)
- Exponential backoff on rate limits
- Circuit breaker opens after 5 consecutive failures
- 5-minute cooldown when circuit opens

### 3. Fence Handler
**Location**: `/streaming/fence_handler.py`

Responsibilities:
- Count unclosed backticks in text
- Temporarily close open code blocks
- Handle both single and triple backticks
- Preserve language hints (```python, ```javascript, etc.)

### 4. OpenAI Client Modifications
**Location**: `/openai_client.py`

New Methods:
```python
create_streaming_response(
    messages: List[Dict],
    model: str,
    ...,
    stream_callback: Callable[[str], None]
) -> Generator[str, None, None]
```

Changes:
- Add `stream=True` parameter to responses.create()
- Parse SSE events from OpenAI
- Yield text chunks as they arrive
- Handle stream completion and errors

### 5. Slack Client Updates
**Location**: `/slack_client.py`

New Methods:
```python
supports_streaming() -> bool
get_streaming_config() -> Dict
update_message_streaming(channel_id, message_id, text) -> Dict
```

Features:
- Return streaming capability (True for Slack)
- Handle 429 responses with retry-after
- Log rate limit events
- Return success/failure status

### 6. Message Processor Integration
**Location**: `/message_processor.py`

Flow:
1. Check if client supports streaming and it's enabled
2. Post initial message to get message ID
3. Start streaming response from OpenAI
4. Update message every 2 seconds (configurable)
5. Apply fence closing for display safety
6. Final update with complete response
7. Fall back to non-streaming on failures

### 7. Configuration
**Location**: `/.env` and `/config.py`

New Environment Variables:
```env
# Streaming Configuration
ENABLE_STREAMING = "true"              # Global streaming toggle
SLACK_STREAMING = "true"               # Platform-specific toggle
STREAMING_UPDATE_INTERVAL = "2.0"      # Seconds between updates
STREAMING_MIN_INTERVAL = "1.0"         # Minimum interval (rate limit floor)
STREAMING_MAX_INTERVAL = "30.0"        # Maximum backoff interval
STREAMING_BUFFER_SIZE = "500"          # Chars before forced update
STREAMING_CIRCUIT_BREAKER_THRESHOLD = "5"  # Failures before circuit opens
STREAMING_CIRCUIT_BREAKER_COOLDOWN = "300" # Seconds to wait before reset
```

## Implementation Phases

### Phase 1: Core Utilities
1. Create `/streaming/` directory structure
2. Implement StreamingBuffer class
3. Implement FenceHandler for markdown safety
4. Implement RateLimitManager
5. Add unit tests for utilities

### Phase 2: OpenAI Integration
1. Add streaming support to OpenAI client
2. Handle SSE event parsing
3. Add error handling for stream disconnections
4. Test with various response types

### Phase 3: Slack Integration
1. Update Slack client with streaming methods
2. Implement rate limit handling
3. Add streaming configuration checks
4. Test rate limit behavior

### Phase 4: Message Processor Integration
1. Add streaming flow to message processor
2. Implement fallback logic
3. Add status updates during streaming
4. Test end-to-end streaming

### Phase 5: Testing & Refinement
1. Test with various content types (code, lists, tables)
2. Test rate limit scenarios
3. Test circuit breaker behavior
4. Performance testing and optimization

## Rate Limiting Strategy

### Slack's Rate Limits
- **Tier 3**: ~50 requests per minute for chat.update
- **Scope**: Per workspace, per method, per app
- **Headers**: Uses `retry-after` header (in seconds) on 429 responses

### Our Approach
1. **Conservative baseline**: 2-second intervals (30 updates/minute)
2. **Reactive throttling**: Increase interval on 429 responses
3. **Exponential backoff**: Double interval on consecutive failures
4. **Circuit breaker**: Disable streaming workspace-wide after repeated failures
5. **Gradual recovery**: Slowly reduce interval after successful updates

## Error Handling

### Connection Errors
- Retry with exponential backoff
- Fall back to non-streaming after 3 failures
- Log errors for debugging

### Rate Limit Errors (429)
- Honor retry-after header
- Increase update interval
- Open circuit breaker if persistent

### Partial Responses
- Store accumulated text
- Provide partial response on disconnection
- Log incomplete responses

## Markdown Handling

### Fence Closing Strategy
1. Count unclosed triple backticks
2. Count unclosed single backticks
3. Add temporary closing fences
4. Preserve language hints
5. Remove temporary fences in final update

### Example:
```
Input: "Here's code:\n```python\nprint('hello"
Output: "Here's code:\n```python\nprint('hello\n```"
```

## Monitoring & Metrics

### Metrics to Track
- Total streaming sessions
- Average update interval
- Rate limit hits
- Circuit breaker activations
- Fallback count
- Average response time

### Logging
- INFO: Streaming session start/end
- WARNING: Rate limits approached/hit
- ERROR: Circuit breaker activated
- DEBUG: Individual update timings

## Future Enhancements

### Discord Support
- Different markdown flavor
- 5 edits per 5 seconds rate limit
- Shows "(edited)" indicator
- May need different fence handling

### Progressive Rendering
- Send summary first
- Fill in details progressively
- Prioritize important information

### Smart Chunking
- Break at sentence boundaries
- Avoid splitting code blocks
- Respect markdown structure

### User Controls
- Allow cancellation mid-stream
- Configurable update frequency per user
- Option to disable streaming per thread

## Testing Plan

### Unit Tests
- StreamingBuffer accumulation
- Fence handler edge cases
- Rate limit manager backoff logic
- Circuit breaker state transitions

### Integration Tests
- OpenAI streaming with mock SSE
- Slack update with rate limiting
- End-to-end streaming flow
- Fallback scenarios

### Load Tests
- Multiple concurrent streams
- Rate limit behavior under load
- Circuit breaker effectiveness
- Memory usage with long streams

## Success Criteria
1. Responses stream smoothly at 2-second intervals
2. No more than 5% of streams hit rate limits
3. Circuit breaker prevents workspace-wide issues
4. Graceful fallback maintains functionality
5. Code blocks display correctly during streaming
6. Final response matches non-streaming version

## Rollback Plan
If streaming causes issues:
1. Set ENABLE_STREAMING=false in .env
2. Circuit breaker automatically disables streaming
3. All responses fall back to non-streaming
4. No code changes required for rollback

---
*Last Updated: 2025-08-11*
*Author: Claude with Human Guidance*