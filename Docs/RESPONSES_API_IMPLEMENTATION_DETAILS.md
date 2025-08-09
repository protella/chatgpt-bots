# OpenAI Responses API Implementation Details
## Technical Reference for Future Re-Implementation

**Document Version:** 1.0  
**Created:** 2025-08-09  
**Purpose:** Capture all technical details needed to re-implement Responses API after reverting to pre-migration state

---

## Executive Summary

This document preserves all critical implementation details discovered during the Responses API migration. While we're reverting the main codebase to manage history locally (removing dependency on `previous_response_id`), these details will enable clean re-implementation when desired.

---

## 1. API Endpoint & Client Changes

### 1.1 Client Initialization
```python
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
```

### 1.2 API Call Structure

**OLD (Chat Completions API):**
```python
response = client.chat.completions.create(
    model=model,
    messages=messages_history,  # Full conversation array
    temperature=temperature,
    max_tokens=max_tokens
)
# Access: response.choices[0].message.content
```

**NEW (Responses API):**
```python
response = client.responses.create(
    model=model,
    instructions=system_prompt,  # System prompt as instructions
    messages=[{"role": "user", "content": message_text}],  # Only current message
    previous_response_id=previous_response_id,  # Links to conversation chain
    store=True,  # CRITICAL: Must be True for conversation storage
    temperature=temperature,
    max_completion_tokens=max_completion_tokens,  # Note: different parameter name
    reasoning_effort=reasoning_effort,  # GPT-5 reasoning models only
    verbosity=verbosity  # GPT-5 reasoning models only
)
# Access: response.message (direct property, not nested)
# Response ID: response.id (needed for chaining)
```

---

## 2. Key API Differences

### 2.1 Parameter Changes
| Chat Completions | Responses API | Notes |
|-----------------|---------------|-------|
| `messages` (array) | `messages` (single msg) + `previous_response_id` | Fundamental change |
| System in messages[0] | `instructions` parameter | Cleaner separation |
| `max_tokens` | `max_completion_tokens` | Renamed parameter |
| N/A | `store` | Must be True for persistence |
| N/A | `previous_response_id` | Links conversation chain |
| N/A | `reasoning_effort` | GPT-5 reasoning only |
| N/A | `verbosity` | GPT-5 reasoning only |

### 2.2 Response Object Structure
```python
# Chat Completions Response
response.choices[0].message.content  # Text content
response.choices[0].message          # Message object
response.usage                       # Token usage

# Responses API Response  
response.message                     # Direct message content
response.id                          # Response ID for chaining
response.output_text                 # Alternative text access
# Note: No usage/token tracking in Responses API
```

### 2.3 Message Format (CLI Bot Example)
```python
# Input format for Responses API
messages = [
    {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
]

# With system prompt (first message only)
messages = [
    CLI_SYSTEM_PROMPT,  # System message object
    {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
]
```

---

## 3. Conversation State Management

### 3.1 Required State Per Thread
```python
# Minimal in-memory state needed
conversations[thread_id] = {
    "previous_response_id": "resp_abc123",  # From last API response
    "system_prompt": "...",                 # For instructions parameter
    "config": {...}                         # Model settings
}
```

### 3.2 Persistence Requirements
- **Thread → Response ID mapping**: Must persist across restarts
- **Thread → Config mapping**: Optional but recommended
- **Expiration**: OpenAI stores responses for 30 days, plan accordingly

### 3.3 SQLite Schema (If Using Persistence)
```sql
CREATE TABLE thread_mappings (
    thread_id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE thread_configs (
    thread_id TEXT PRIMARY KEY,
    config_json TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Critical Implementation Patterns

### 4.1 Store Parameter Usage
```python
# Main conversations - ALWAYS store=True
response = client.responses.create(
    ...,
    store=True  # Preserves in conversation chain
)

# Utility functions - Use store=False
response = client.responses.create(
    ...,
    store=False,  # Temporary, doesn't pollute chain
    previous_response_id=main_thread_response_id  # Still has context
)
```

### 4.2 Thread Initialization
```python
def initialize_thread(thread_id):
    # First message: No previous_response_id
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        messages=[{"role": "user", "content": message}],
        store=True
        # NO previous_response_id parameter
    )
    # Save response.id for next message
```

### 4.3 Continuing Conversations
```python
def continue_conversation(thread_id, message):
    previous_id = get_last_response_id(thread_id)
    
    response = client.responses.create(
        model=model,
        instructions=get_system_prompt(thread_id),
        messages=[{"role": "user", "content": message}],
        previous_response_id=previous_id,  # Links to chain
        store=True
    )
    # Save new response.id
```

---

## 5. Model-Specific Parameters

### 5.1 GPT-5 Reasoning Models
Models: `gpt-5-nano-*`, `gpt-5-mini-*`, `gpt-5-*` (without "chat")
```python
# Constraints
temperature = 1.0  # FIXED, cannot change
top_p = None      # NOT supported

# Exclusive parameters
reasoning_effort = "medium"  # minimal/low/medium/high
verbosity = "medium"         # low/medium/high
```

### 5.2 GPT-5 Chat Models
Models: `gpt-5-chat-*`
```python
# Standard parameters work
temperature = 0.0 to 2.0
top_p = 0.0 to 1.0
# No reasoning_effort or verbosity
```

### 5.3 Model Capability Detection
```python
def get_model_capabilities(model_name):
    is_reasoning = (
        model_name.startswith("gpt-5") and 
        "chat" not in model_name
    )
    return {
        "supports_temperature": not is_reasoning,
        "fixed_temperature": 1.0 if is_reasoning else None,
        "supports_reasoning_effort": is_reasoning,
        "supports_verbosity": is_reasoning
    }
```

---

## 6. Context-Aware Utility Functions

### 6.1 Maintaining Context Without Polluting Chain
```python
def utility_check_with_context(thread_id, check_prompt):
    # Get main conversation's last response ID
    previous_response_id = get_last_response_id(thread_id)
    
    response = client.responses.create(
        model=UTILITY_MODEL,  # e.g., gpt-5-nano for speed
        instructions=UTILITY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": check_prompt}],
        previous_response_id=previous_response_id,  # Has full context
        store=False,  # IMPORTANT: Doesn't save to chain
        # Model-specific params...
    )
    return response.message
```

### 6.2 Image Generation Check Example
```python
def check_for_image_generation(message, thread_id):
    previous_response_id = get_last_response_id(thread_id)
    
    check_prompt = f"Is this requesting image generation: {message}?"
    
    response = client.responses.create(
        model="gpt-5-nano-2025-08-07",  # Fast utility model
        instructions=IMAGE_CHECK_PROMPT,
        messages=[{"role": "user", "content": check_prompt}],
        previous_response_id=previous_response_id,
        store=False,  # Temporary check
        reasoning_effort="minimal",
        verbosity="low"
    )
    
    return "true" in response.message.lower()
```

---

## 7. Migration Checklist

When re-implementing Responses API:

### 7.1 Code Changes Required
- [ ] Replace `client.chat.completions.create()` with `client.responses.create()`
- [ ] Move system prompt from messages[0] to `instructions` parameter
- [ ] Change `max_tokens` to `max_completion_tokens`
- [ ] Add `store=True` to all main conversation calls
- [ ] Add `store=False` to utility function calls
- [ ] Implement response ID tracking per thread
- [ ] Change response parsing from `response.choices[0].message` to `response.message`
- [ ] Save `response.id` after each API call

### 7.2 Features to Remove
- [ ] Message array management
- [ ] History reconstruction functions
- [ ] Token/usage tracking (not available in Responses API)
- [ ] Manual conversation pruning (server handles via 30-day expiry)

### 7.3 Infrastructure to Add
- [ ] Persistence layer for thread → response_id mapping
- [ ] Thread initialization logic (first message without previous_response_id)
- [ ] Response ID retrieval for continuing conversations
- [ ] Cleanup mechanism for expired threads (>30 days)

---

## 8. Error Handling Considerations

### 8.1 Common Errors
```python
# Invalid previous_response_id
# Solution: Treat as new conversation, omit previous_response_id

# Missing store parameter
# Solution: Always explicitly set store=True or store=False

# Temperature not 1.0 for reasoning models
# Solution: Detect model type, force temperature=1.0

# Using deprecated parameters
# Solution: Don't pass top_p to reasoning models
```

### 8.2 Fallback Strategy
```python
try:
    response = client.responses.create(...)
except Exception as e:
    if "previous_response_id" in str(e):
        # Retry without previous_response_id (new conversation)
        response = client.responses.create(
            # Same params but without previous_response_id
        )
```

---

## 9. Testing Considerations

### 9.1 Critical Tests
1. **New conversation**: Verify works without previous_response_id
2. **Continued conversation**: Verify previous_response_id chains correctly
3. **Utility functions**: Verify store=False doesn't affect main chain
4. **Thread persistence**: Verify response IDs survive restart
5. **Model detection**: Verify correct params for reasoning vs chat models
6. **Error recovery**: Verify graceful handling of invalid response IDs

### 9.2 Integration Points
- Database persistence (if used)
- Response ID management
- System prompt handling
- Model capability detection
- Parameter validation

---

## 10. Performance Optimizations

### 10.1 Utility Model Selection
```python
# Use fastest model for utility checks
UTILITY_MODEL = "gpt-5-nano-2025-08-07"  # Fastest
# vs
GPT_MODEL = "gpt-5-chat-latest"  # Main conversations
```

### 10.2 Caching Considerations
- Cache response IDs in memory
- Batch database writes if using persistence
- Use store=False for all temporary operations

---

## 11. Example Implementation

### 11.1 Minimal Working Example
```python
class ResponsesAPIChat:
    def __init__(self):
        self.client = OpenAI()
        self.threads = {}  # thread_id -> last_response_id
    
    def send_message(self, thread_id, message, system_prompt):
        params = {
            "model": "gpt-5-chat-latest",
            "instructions": system_prompt,
            "messages": [{"role": "user", "content": message}],
            "store": True,
            "max_completion_tokens": 2048
        }
        
        # Add previous_response_id if continuing conversation
        if thread_id in self.threads:
            params["previous_response_id"] = self.threads[thread_id]
        
        response = self.client.responses.create(**params)
        
        # Save response ID for next message
        self.threads[thread_id] = response.id
        
        return response.message
```

---

## Notes

1. **Why We're Reverting**: Managing previous_response_id proved complex for our use case
2. **What We're Keeping**: All the API knowledge and patterns documented here
3. **Future Path**: This document enables clean re-implementation when needed
4. **Key Learning**: Responses API works well but requires different architecture than message arrays

This document contains everything needed to re-implement Responses API support without repeating the discovery process.