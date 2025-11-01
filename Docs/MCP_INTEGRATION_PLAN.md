# MCP Integration Implementation Plan

## Overview

This document outlines the implementation plan for integrating OpenAI's native Model Context Protocol (MCP) support into the Slackbot. The integration leverages OpenAI's Responses API to connect to external MCP servers, enabling the bot to access specialized data sources and tools without manual function calling implementation.

### Key Features
- **Plug-and-play MCP support** - Add servers via configuration file
- **OpenAI-managed connections** - No manual MCP client implementation needed
- **Graceful model handling** - GPT-5 supports MCP, GPT-4 falls back gracefully
- **Tool discovery on startup** - Cache available tools from all configured servers
- **User preference control** - Enable/disable MCP per user alongside web_search

---

## Architecture Overview

### Current Flow
```
Message → Intent Classifier → Handler
                               ├→ Image Generation
                               ├→ Image Edit
                               ├→ Vision Analysis
                               └→ Text Response (with optional web_search)
```

### New Flow (with MCP)
```
Message → Intent Classifier → Handler
                               ├→ Image Generation (unchanged)
                               ├→ Image Edit (unchanged)
                               ├→ Vision Analysis (unchanged)
                               └→ Text Response
                                   ↓
                              Build tools array:
                                - web_search (if enabled)
                                - MCP servers (if enabled + GPT-5)
                                   ↓
                              Send to OpenAI Responses API
                                   ↓
                              OpenAI decides & executes tools
                                   ↓
                              Return response to Slack
```

### Key Principles
1. **Existing intent classifier unchanged** - Still classifies structural intents (image vs text)
2. **MCP only applies to text operations** - Image/vision handlers unmodified
3. **OpenAI handles all tool decisions** - No manual tool routing logic
4. **Graceful degradation** - GPT-4 requests exclude MCP, include web_search only

---

## How OpenAI Native MCP Works

### The Flow
1. You pass MCP server configs in the `tools` array when calling OpenAI
2. OpenAI connects to your MCP servers and discovers available tools
3. OpenAI includes discovered tools in the LLM's context
4. LLM decides if any tools are needed for the user's query
5. If tools are needed, OpenAI calls them and includes results in response
6. You receive the final response with tool results already incorporated

### Tool Definition Format
```json
{
  "type": "mcp",
  "server_label": "server-name",
  "server_url": "https://mcp-server.com/mcp",
  "server_description": "What this server provides",
  "authorization": "Bearer TOKEN",
  "require_approval": "never",
  "allowed_tools": ["tool1", "tool2"]
}
```

### Key Points
- OpenAI manages the MCP protocol - you don't need an MCP client library
- Tools are cached in conversation context via `mcp_list_tools` output items
- You can mix `web_search` and `mcp` tools in the same request
- Only GPT-5 models support MCP tools

---

## Database Changes

### New Table: `mcp_tools`

**Purpose:** Cache discovered tools from MCP servers for faster lookup and UI display.

**Scope:** Global cache (not per-thread). Populated on bot startup.

**Schema Requirements:**
- Store server label, tool name, description, input schema
- Track when tools were discovered and last verified
- Ensure uniqueness per (server_label, tool_name) combination
- Index on server_label for fast queries

**Usage:**
- Populate during startup tool discovery
- Query when showing available capabilities in logs/UI
- Update when servers are added/removed from config

---

## Configuration

### 1. MCP Configuration File

**File:** `mcp_config.json` (user creates from example, not in git)

**Format:** JSON object with `mcpServers` key containing server configurations.

**Required Fields per Server:**
- `server_url` - HTTPS endpoint for MCP server

**Optional Fields per Server:**
- `server_description` - Helps OpenAI understand when to use this server
- `authorization` - OAuth token or API key
- `require_approval` - **IGNORED** (always set to "never" internally). Config value preserved for future approval UI implementation
- `allowed_tools` - Whitelist specific tools (empty = all tools)

**Example Template:** See `mcp_config.example.json`

### 2. Environment Variables

**Add to `.env`:**
- `MCP_ENABLED_DEFAULT` - Default on/off for new users (default: true)
- `MCP_CONFIG_PATH` - Path to MCP config file (default: mcp_config.json)

### 3. Bot Configuration Class

**Update:** `config.py` → `BotConfig` dataclass

**Add Fields:**
- `mcp_enabled_default: bool` - From env var
- `mcp_config_path: str` - From env var

### 4. Git Ignore

**Already present:** `mcp_config.json` is in `.gitignore` (line 174)

**Commit to git:** `mcp_config.example.json` as template

---

## Code Components

### 1. MCP Manager (New File: `mcp_manager.py`)

**Responsibilities:**
- Load MCP server configurations from `mcp_config.json`
- Provide MCP tool definitions formatted for OpenAI API
- Perform tool discovery on startup (optional, for caching)
- Maintain in-memory cache of available tools

**Key Methods:**
- `initialize()` - Load config, populate cache from DB, start discovery
- `get_tools_for_openai()` - Return list of tool definitions for API calls
- `has_mcp_servers()` - Check if any servers configured
- `get_server_labels()` - List configured servers

**Implementation Notes:**
- Should gracefully handle missing config file (log warning, continue)
- Should not block bot startup on initialization failures
- Tool discovery can run asynchronously after startup
- Cache should be both in-memory (fast) and DB (persistent)

### 2. Database Methods

**Update:** `database.py` → `DatabaseManager` class

**New Methods Needed:**
- `save_mcp_tool()` - Insert or update tool in cache
- `get_mcp_tools()` - Retrieve cached tools (optionally by server)
- `clear_mcp_tools()` - Remove cached tools (for refresh)

**Migration:**
- Add new migration method to create `mcp_tools` table
- Run migration in `run_migrations()` during bot startup

### 3. Message Processor

**Update:** `message_processor/base.py` → `MessageProcessor.__init__()`

**Changes:**
- Import and instantiate `MCPManager`
- Call `mcp_manager.initialize()` asynchronously on startup
- Pass MCP manager to handlers (it's already accessible via `self`)

**No Changes to:**
- Intent classification logic
- Image/vision handlers
- Thread management

### 4. Text Handler

**Update:** `message_processor/handlers/text.py` → `TextHandlerMixin`

**Changes in `_handle_text_response()`:**
- Build tools array based on user preferences and model
- Pass tools to OpenAI client

**New Helper Method:** `_build_tools_array()`
- Check user preferences for web_search and MCP
- Include web_search if enabled
- Include MCP servers if enabled AND model is GPT-5
- Return None if no tools enabled

**Logic:**
```
If web_search enabled:
    Add {"type": "web_search"}

If MCP enabled AND model starts with "gpt-5" AND MCP servers configured:
    For each server in MCP config:
        Add MCP tool definition with server details

Return tools list (or None if empty)
```

### 5. OpenAI Client

**Update:** `openai_client/api/responses.py` → `create_text_response()`

**Changes:**
- Add `tools` parameter (optional, defaults to None)
- Include tools in request params if provided
- Log when tools are included

**Behavior:**
- If tools provided, add to request as `"tools": tools`
- OpenAI SDK handles the rest (tool discovery, execution, etc.)
- Response parsing unchanged (tool results already in `content`)

### 6. User Settings

**Update:** `settings_modal.py`

**Changes in Settings UI:**
- Add "Enable MCP Servers" checkbox (similar to web_search)
- Add hint text: "Requires GPT-5. Access specialized data sources."
- Modify model dropdown to exclude GPT-4 options when MCP enabled
- Update hint for model dropdown when MCP enabled

**Changes in Settings Handler:**
- Save MCP preference to user preferences
- Validate model selection is compatible with MCP preference

**Conditional Logic:**
- If MCP enabled: Only show GPT-5 models in dropdown
- If MCP disabled: Show all models (GPT-4 + GPT-5)
- If user enables MCP while GPT-4 selected: Override to GPT-5-mini

### 7. System Prompt

**Update:** `prompts.py` → `SLACK_SYSTEM_PROMPT`

**Addition to Capabilities List:**
Add bullet point under existing capabilities:
```
- External data tools: You have access to specialized data sources and tools (via MCP).
  When users ask questions requiring current data, specialized knowledge, or
  external information, use these tools to provide accurate, authoritative answers.
```

**Keep Generic:** Don't mention specific MCP servers or tools - this applies to any MCP server.

---

## Tool Discovery Process

### On Bot Startup

**Objective:** Populate cache with available tools from all configured MCP servers.

**Approach:**
1. Load MCP server configs from file
2. For each server, make a minimal request to OpenAI with that MCP tool
3. OpenAI connects to server and discovers tools (returns `mcp_list_tools` output)
4. Parse tool definitions from response
5. Save to database and in-memory cache

**Implementation Notes:**
- Run asynchronously (don't block bot startup)
- Handle failures gracefully (log error, continue with other servers)
- Use smallest/cheapest model for discovery (GPT-5-nano or GPT-5-mini)
- Keep request minimal (short system prompt, no conversation history)

**Alternative:** Skip explicit discovery, let tools be discovered on first actual use per thread. Cache is then built organically.

---

## User Preference Logic

### Storage

MCP preference stored in user preferences (same as web_search):
- Database: `user_preferences` table
- Key: `enable_mcp`
- Value: boolean (true/false)
- Default: From `config.mcp_enabled_default`

### Retrieval

When processing message:
1. Get thread config (includes user preferences)
2. Check `enable_mcp` preference
3. Check model compatibility (GPT-5 only)
4. Decide whether to include MCP tools

### Model Compatibility

**Rule:** MCP tools only included if:
- User has `enable_mcp: true` in preferences
- AND current model starts with "gpt-5"

**If GPT-4 selected:**
- Skip MCP tools entirely
- Still include web_search if enabled
- No error, graceful degradation

---

## Error Handling

### MCP Config Missing
- Log warning: "MCP config file not found"
- Continue without MCP support
- Don't crash bot

### MCP Server Unreachable
- OpenAI handles connection errors
- Tool call will fail gracefully
- Response may mention tools unavailable

### Tool Discovery Fails
- Log error with server name
- Continue with other servers
- Don't block bot operation

### Invalid Configuration
- Validate config file is valid JSON
- Log specific parsing errors
- Skip malformed server entries

---

## Testing Strategy

### Unit Tests

**New Test File:** `tests/unit/test_mcp_manager.py`

**Test Cases:**
- Config loading from valid JSON
- Handling missing config file
- Building tool definitions for OpenAI
- Caching mechanism (DB + memory)

### Integration Tests

**New Test File:** `tests/integration/test_mcp_integration.py`

**Use Real Public MCP Server:** Context7 or similar (no auth required)

**Test Cases:**
- Tool discovery with real server
- Sending request with MCP tools to OpenAI
- Receiving valid response (content non-deterministic, just verify structure)
- Using web_search + MCP together

**Note:** LLM responses are non-deterministic. Test for structural correctness, not exact content.

### Manual Testing Checklist

- [ ] Bot starts with MCP config present
- [ ] Bot starts without MCP config (logs warning)
- [ ] Settings modal shows MCP checkbox
- [ ] Enabling MCP hides GPT-4 models
- [ ] Disabling MCP shows GPT-4 models
- [ ] MCP request with GPT-5 includes tools
- [ ] MCP request with GPT-4 excludes MCP tools
- [ ] Tool discovery runs on startup (check logs)
- [ ] Tools cached in database
- [ ] Web search + MCP work together

---

## Documentation Updates

### README.md

**New Section:** "MCP (Model Context Protocol) Integration"

**Content:**
- Brief explanation of what MCP is
- How to set up `mcp_config.json`
- Where to find MCP servers (links to directories)
- How users enable/disable in settings
- GPT-5 requirement
- Troubleshooting common issues

**Tone:** User-friendly, assumes no prior MCP knowledge.

### CLAUDE.md

**Updates:**
- Add MCP to architecture overview
- Document MCP manager component
- Note GPT-5 requirement for MCP
- Link to implementation plan

---

## Implementation Phases

### Phase 1: Configuration & Database (2-3 hours)
- Create `mcp_config.example.json`
- Add env vars and config fields
- Write database migration
- Add DB methods for tool caching

### Phase 2: Core MCP Manager (4-5 hours)
- Create `mcp_manager.py`
- Implement config loading
- Implement tool definition building
- Add startup discovery logic

### Phase 3: Integration Points (3-4 hours)
- Update message processor to initialize manager
- Update text handler to build tools array
- Update OpenAI client to accept tools param
- Update system prompt

### Phase 4: User Interface (2-3 hours)
- Add MCP checkbox to settings modal
- Add conditional model dropdown logic
- Add hint text about GPT-5 requirement
- Handle settings save/load

### Phase 5: Testing (4-5 hours)
- Write unit tests
- Write integration tests with real MCP
- Manual testing against checklist
- Fix any issues found

### Phase 6: Documentation (2 hours)
- Update README with setup guide
- Update CLAUDE.md
- Add code comments
- Document troubleshooting

**Total Estimated Effort:** 20-25 hours

---

## Key Design Decisions

### Why OpenAI Native MCP vs. Manual Client?
- **OpenAI Native:** They handle protocol, discovery, execution
- **Manual:** Would require implementing full MCP protocol client
- **Decision:** Use native support (simpler, maintained by OpenAI)

### Why Tool Discovery on Startup?
- **Pros:** Tools cached, faster responses, can show in UI
- **Cons:** Adds startup time, costs tokens
- **Decision:** Optional async discovery, not blocking startup

### Why Global Tool Cache vs. Per-Thread?
- **Behavior:** Tools don't change per-thread, they're server capabilities
- **Decision:** Global cache, shared across all threads

### Why Auto-Approve vs. Approval UI?
- **Current:** Only "never" supported (auto-approve all)
- **Future:** Approval UI for sensitive tools
- **Decision:** Start simple, iterate later

### Why GPT-5 Only?
- **Limitation:** GPT-4 doesn't support MCP tools in OpenAI API
- **Decision:** Gracefully exclude MCP for GPT-4 users

---

## Security Considerations

### API Key Protection
- `mcp_config.json` in `.gitignore`
- Never log authorization tokens
- Provide example template without real keys

### Server Trust
- Only connect to HTTPS endpoints
- User responsible for vetting MCP servers
- OpenAI validates server responses

### Data Privacy
- MCP servers receive conversation context
- User should only enable trusted servers
- Consider data residency requirements

### Tool Approval
- Currently auto-approve (trusted servers)
- Future: Per-tool approval policies
- Log all tool executions for audit

---

## Future Enhancements

### Approval Flow UI
- Slack blocks for approving/denying tool calls
- Per-tool approval policies
- Approval audit log

### Enhanced Error Messages
- Show which specific tool failed
- Surface error details from MCP server
- Suggest remediation steps

### Tool Usage Analytics
- Track which tools used most
- Measure response quality with/without tools
- Cost analysis per tool

### Dynamic Tool Management
- Let users enable/disable specific servers
- Per-thread tool preferences
- Smart tool suggestion based on topic

### Health Monitoring
- Periodic health checks for servers
- Auto-disable unhealthy servers
- Admin notifications on failures

---

## OpenAI MCP Reference

### Tool Definition Structure
```
{
  "type": "mcp",
  "server_label": "unique-name",
  "server_url": "https://...",
  "server_description": "optional-description",
  "authorization": "optional-token",
  "require_approval": "never|always|object",
  "allowed_tools": ["optional", "whitelist"]
}
```

### Response Structure
OpenAI returns output items in response:
- `mcp_list_tools` - Discovered tools from server
- `mcp_call` - Tool execution results
- `text` - Final generated response

### Transport Support
OpenAI native MCP supports:
- Streamable HTTP
- HTTP with SSE (Server-Sent Events)

Does NOT support:
- stdio (local process communication)

---

## Questions for Implementation

None currently - design decisions are finalized based on user clarifications.

---

## References

- [OpenAI MCP Documentation](https://platform.openai.com/docs/guides/mcp)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [MCP Server Directory](https://modelcontextprotocol.io/servers)
