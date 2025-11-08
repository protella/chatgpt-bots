# Generic OAuth Architecture for MCP Servers

## Document Status
**Status:** Design Proposal
**Created:** 2025-01-01
**Purpose:** Define a modular, provider-agnostic OAuth framework for MCP server authentication

---

## Overview

This document describes a generic architecture for supporting OAuth-based MCP servers in the bot, designed to be completely provider-agnostic and configuration-driven. The goal is to enable adding new MCP servers (Atlassian, GitHub, Linear, Notion, etc.) through configuration changes only, with zero code modifications.

### Goals

1. **Provider Neutrality** - No hardcoded provider-specific logic
2. **Configuration-Driven** - All provider details in config files
3. **Standards-Based** - Use OAuth 2.0 RFC standards and metadata discovery
4. **Plug-and-Play** - Add new servers by editing config + env vars
5. **User-Scoped** - Support per-user OAuth tokens for personalized data access

### Non-Goals

- OAuth 1.0 support (use OAuth 2.0 only)
- Custom/proprietary auth protocols (standardize on OAuth 2.0)
- Built-in OAuth UI (use provider-hosted consent screens)

---

## Architecture Principles

### Separation of Concerns

| Layer | Responsibility | Provider-Specific? |
|-------|---------------|-------------------|
| **Configuration** | Define provider OAuth endpoints, scopes | ✅ Yes (in config) |
| **OAuth Manager** | Generic OAuth flow execution | ❌ No |
| **MCP Manager** | Token injection into tool definitions | ❌ No |
| **Database** | Token storage/retrieval | ❌ No |
| **Web Endpoints** | OAuth redirect handling | ❌ No |

**Key Principle:** Provider-specific information lives ONLY in configuration files and environment variables. Python code knows nothing about specific providers.

### Authentication Types

The architecture supports multiple authentication patterns:

| Type | Description | Use Case | Example |
|------|-------------|----------|---------|
| `none` | No authentication | Public APIs, local servers | Stdio filesystem |
| `static_bearer` | Static Bearer token | Service API keys | Context7, OpenAI-style APIs |
| `static_api_key` | Custom header format | Proprietary APIs | Legacy systems |
| `user_oauth2` | Per-user OAuth 2.0 | User-scoped data | Atlassian, GitHub, Linear |
| `service_oauth2` | Service account OAuth | Bot-level access | Background services |

---

## Component Design

### 1. Enhanced MCP Configuration Schema

**Location:** `mcp_config.json`

**Structure:**
```json
{
  "mcpServers": {
    "server-label": {
      "server_url": "https://...",
      "server_description": "Human-readable description",
      "auth": {
        "type": "user_oauth2|static_bearer|none|...",
        "flow": "authorization_code|client_credentials|device_code",
        "provider_id": "unique-provider-identifier",

        // Option 1: Explicit configuration
        "authorize_url": "https://...",
        "token_url": "https://...",

        // Option 2: Metadata discovery (RFC 8414)
        "metadata_url": "https://.../.well-known/oauth-authorization-server",

        "scopes": ["scope1", "scope2"],
        "extra_params": {
          "audience": "...",
          "prompt": "consent"
        }
      }
    }
  }
}
```

**Key Fields:**

- `auth.type` - Determines authentication strategy
- `auth.provider_id` - Unique identifier for token storage (defaults to server label)
- `auth.metadata_url` - Optional RFC 8414 metadata endpoint for auto-discovery
- `auth.authorize_url` / `token_url` - Explicit OAuth endpoints
- `auth.scopes` - Requested OAuth scopes
- `auth.extra_params` - Provider-specific OAuth parameters (e.g., Atlassian's `audience`)

**Examples:**

**User OAuth (Atlassian):**
```json
{
  "atlassian": {
    "server_url": "https://mcp.atlassian.com/v1/sse",
    "server_description": "Access your Atlassian Confluence and Jira",
    "auth": {
      "type": "user_oauth2",
      "flow": "authorization_code",
      "provider_id": "atlassian",
      "authorize_url": "https://auth.atlassian.com/authorize",
      "token_url": "https://auth.atlassian.com/oauth/token",
      "scopes": ["read:confluence-content.all", "read:jira-work", "offline_access"],
      "extra_params": {
        "audience": "api.atlassian.com",
        "prompt": "consent"
      }
    }
  }
}
```

**Metadata Discovery (Linear):**
```json
{
  "linear": {
    "server_url": "https://mcp.linear.app/v1/sse",
    "server_description": "Access your Linear issues and projects",
    "auth": {
      "type": "user_oauth2",
      "provider_id": "linear",
      "metadata_url": "https://api.linear.app/.well-known/oauth-authorization-server",
      "scopes": ["read", "issues:read"]
    }
  }
}
```

**Static Bearer Token:**
```json
{
  "context7": {
    "server_url": "https://api.context7.com/mcp",
    "server_description": "Documentation search",
    "auth": {
      "type": "static_bearer",
      "token": "${CONTEXT7_API_KEY}"
    }
  }
}
```

**No Authentication:**
```json
{
  "local-filesystem": {
    "server_url": "http://localhost:8080/stdio/filesystem",
    "server_description": "Local filesystem access",
    "auth": {
      "type": "none"
    }
  }
}
```

### 2. OAuth Provider Manager

**Purpose:** Generic OAuth flow orchestration without provider-specific logic

**Responsibilities:**
- Load provider configurations from MCP config
- Generate authorization URLs
- Exchange authorization codes for tokens
- Refresh access tokens
- Discover endpoints via RFC 8414 metadata

**Key Methods:**

```python
class OAuthProviderManager:
    def get_authorization_url(provider_id, state, redirect_uri) -> str
    def exchange_code_for_token(provider_id, code, redirect_uri) -> dict
    def refresh_token(provider_id, refresh_token) -> dict
```

**Convention-Based Secrets:**
- Client ID from env: `{PROVIDER_ID}_CLIENT_ID`
- Client secret from env: `{PROVIDER_ID}_CLIENT_SECRET`
- Example: `ATLASSIAN_CLIENT_ID`, `GITHUB_CLIENT_SECRET`

### 3. Database Schema

**New Table:** `oauth_tokens`

```sql
CREATE TABLE oauth_tokens (
    user_id TEXT NOT NULL,              -- Slack/Discord user ID
    provider TEXT NOT NULL,             -- Provider ID from config
    access_token TEXT NOT NULL,         -- Encrypted access token
    refresh_token TEXT,                 -- Encrypted refresh token
    token_type TEXT DEFAULT 'Bearer',   -- Token type
    expires_at INTEGER,                 -- Unix timestamp
    scope TEXT,                         -- Granted scopes
    created_at INTEGER,                 -- Unix timestamp
    updated_at INTEGER,                 -- Unix timestamp
    PRIMARY KEY (user_id, provider)
);
```

**Security:** Tokens MUST be encrypted at rest using Fernet or similar symmetric encryption.

**Methods:**

```python
class DatabaseManager:
    def save_oauth_token(user_id, provider, access_token, refresh_token, expires_in, scope)
    def get_oauth_token(user_id, provider) -> dict  # Returns decrypted token
    def delete_oauth_token(user_id, provider)
    def get_expired_tokens() -> list  # For background refresh
```

### 4. Generic OAuth Web Endpoints

**Required Endpoints:**

1. **`GET /oauth/start`** - Initiate OAuth flow
   - Query params: `provider`, `user_id`, `channel_id`
   - Generates state token (CSRF protection)
   - Redirects to provider's authorization URL

2. **`GET /oauth/callback`** - Handle OAuth redirect
   - Query params: `code`, `state`, `error`
   - Verifies state token
   - Exchanges code for tokens
   - Stores encrypted tokens in database
   - Notifies user in Slack/Discord

**State Management:**
- Temporary storage of OAuth state (Redis recommended, in-memory for MVP)
- State payload: `{provider_id, user_id, channel_id}`
- TTL: 10 minutes

**Framework:** Flask or FastAPI (preference: FastAPI for async support)

### 5. MCP Manager Integration

**Updated Method Signature:**

```python
def get_tools_for_openai(user_id: str = None) -> List[Dict[str, Any]]
```

**Authorization Injection Logic:**

For each MCP server:

1. Read `auth.type` from config
2. Based on type:
   - `none`: No authorization header
   - `static_bearer`: Resolve token from env/config, add `authorization` field
   - `user_oauth2`: Look up user token in DB, inject if valid
   - `service_oauth2`: Use service token (from background refresher)
3. If token missing/expired for `user_oauth2`, skip that tool (don't include in array)
4. Return tool definitions for OpenAI

**Key Behavior:** Tools array is dynamically built per user, per message. Different users get different tools based on their authentication status.

### 6. Message Handler Integration

**Pre-Flight Auth Check:**

Before processing any message that could use MCP tools:

1. Query `mcp_manager.get_missing_auth_providers(user_id)`
2. If providers missing, send authentication prompt:
   ```
   🔐 To use the following integrations, please authenticate:

   • [Provider Name] - Description
   • [Provider Name] - Description

   After authenticating, please retry your query.
   ```
3. Stop processing (don't send to OpenAI)

**Note:** This check is optional/configurable. Bot could also proceed with available tools only.

---

## User Experience Flows

### Flow 1: First-Time Authentication

1. **User:** "What's in my Linear backlog?"
2. **Bot:** Checks DB, no token for Linear
3. **Bot responds:**
   ```
   🔐 To use the following integrations, please authenticate:

   • Linear - Access your Linear issues and projects

   After authenticating, please retry your query.
   ```
4. **User:** Clicks authentication link
5. **Browser:** Redirects to Linear OAuth consent screen
6. **User:** Approves access
7. **Browser:** Redirects to bot's callback endpoint
8. **Bot:** Stores encrypted tokens, sends Slack message:
   ```
   ✅ @user Successfully connected to Linear!
   ```
9. **User:** "What's in my Linear backlog?" (retry)
10. **Bot:** Builds tools array with user's token, sends to OpenAI
11. **OpenAI:** Calls Linear MCP with user's token, returns response
12. **Bot:** Sends personalized Linear data to user

### Flow 2: Token Refresh

1. **User:** Makes request requiring OAuth MCP
2. **Bot:** Retrieves token from DB, detects expiration
3. **Bot:** Calls `oauth_manager.refresh_token(provider_id, refresh_token)`
4. **OAuth Manager:** Exchanges refresh token for new access token
5. **Bot:** Updates DB with new token
6. **Bot:** Continues with request using fresh token

### Flow 3: Multiple Providers

1. **User:** "Show me my GitHub PRs and Linear tasks"
2. **Bot:** Checks auth status:
   - GitHub: ✅ Authenticated
   - Linear: ❌ Not authenticated
3. **Bot:** Prompts for Linear authentication only
4. **User:** Authenticates to Linear
5. **User:** Retries query
6. **Bot:** Builds tools with both GitHub and Linear tokens
7. **OpenAI:** Uses both MCPs in single response

---

## Security Considerations

### Token Encryption

**Requirement:** All OAuth tokens MUST be encrypted at rest.

**Implementation:**
- Use Fernet symmetric encryption (from `cryptography` library)
- Master key stored in environment variable: `TOKEN_ENCRYPTION_KEY`
- Key rotation strategy: Generate new key, re-encrypt all tokens, update env

**Example:**
```python
from cryptography.fernet import Fernet

cipher = Fernet(os.getenv('TOKEN_ENCRYPTION_KEY').encode())
encrypted = cipher.encrypt(access_token.encode())
decrypted = cipher.decrypt(encrypted).decode()
```

### CSRF Protection

**Requirement:** OAuth flows MUST use state parameter for CSRF protection.

**Implementation:**
- Generate cryptographically secure random state token: `secrets.token_urlsafe(32)`
- Store state with user context (in-memory or Redis)
- Verify state on callback
- TTL: 10 minutes

### Environment Variables

**Required:**
```bash
# OAuth configuration
OAUTH_BASE_URL=https://your-bot.com
TOKEN_ENCRYPTION_KEY=<fernet-key>

# Per-provider OAuth clients (convention-based)
{PROVIDER_ID}_CLIENT_ID=...
{PROVIDER_ID}_CLIENT_SECRET=...
```

**Example:**
```bash
ATLASSIAN_CLIENT_ID=abc123
ATLASSIAN_CLIENT_SECRET=secret123
GITHUB_CLIENT_ID=xyz789
GITHUB_CLIENT_SECRET=secret789
```

### Token Scope Validation

**Best Practice:** Validate that granted scopes match requested scopes.

**Implementation:**
- Check `scope` field in token response
- Log warning if scopes don't match
- Store actual granted scopes in database

### SSL/TLS Requirements

**Requirement:** OAuth callback URL MUST use HTTPS in production.

**Exceptions:** `localhost` for development only.

---

## Standards and References

### OAuth 2.0 Specifications

- **RFC 6749** - The OAuth 2.0 Authorization Framework
- **RFC 6750** - The OAuth 2.0 Bearer Token Usage
- **RFC 7636** - Proof Key for Code Exchange (PKCE) - Consider for enhanced security
- **RFC 8414** - OAuth 2.0 Authorization Server Metadata (for endpoint discovery)

### Metadata Discovery Example

If provider supports RFC 8414, bot can auto-discover endpoints:

**Request:**
```
GET https://auth.example.com/.well-known/oauth-authorization-server
```

**Response:**
```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "scopes_supported": ["read", "write"],
  "response_types_supported": ["code"]
}
```

**Benefit:** Provider configs only need `metadata_url`, everything else is discovered.

---

## Implementation Phases

### Phase 1: Foundation (MVP)
**Goal:** Support single OAuth provider (Atlassian) with manual configuration

- [ ] Database schema for `oauth_tokens` table
- [ ] Token encryption/decryption utilities
- [ ] Flask/FastAPI OAuth endpoints (`/oauth/start`, `/oauth/callback`)
- [ ] Basic `OAuthProviderManager` (no metadata discovery)
- [ ] MCP Manager integration for token injection
- [ ] Manual testing with Atlassian MCP

**Deliverables:**
- Working OAuth flow for one provider
- Encrypted token storage
- User authentication prompt in Slack

### Phase 2: Generalization
**Goal:** Support multiple providers through configuration

- [ ] Enhanced MCP config schema with `auth` section
- [ ] Convention-based environment variable lookup
- [ ] Generic web endpoints (no provider-specific logic)
- [ ] Pre-flight auth check in message handler
- [ ] Add 2-3 additional providers (GitHub, Linear, Notion)

**Deliverables:**
- Add new provider by editing config + env only
- Multi-provider authentication flow
- Documentation for adding providers

### Phase 3: Production Hardening
**Goal:** Enterprise-ready deployment

- [ ] RFC 8414 metadata discovery
- [ ] Token refresh background job
- [ ] Redis-backed state storage (replace in-memory)
- [ ] PKCE support for enhanced security
- [ ] Token rotation and revocation
- [ ] Monitoring and alerting
- [ ] User token management UI (list/revoke tokens)

**Deliverables:**
- Production-ready OAuth system
- Monitoring dashboard
- User documentation

### Phase 4: Advanced Features (Optional)
**Goal:** Enhanced capabilities

- [ ] Service OAuth (client credentials flow)
- [ ] Device code flow for CLI-based auth
- [ ] Multi-workspace support (same provider, different workspaces)
- [ ] Token sharing across bot instances (distributed deployment)
- [ ] OAuth scope upgrading (request additional scopes)

---

## Configuration Examples

### Adding New Provider: Notion

**Step 1:** Update `mcp_config.json`:
```json
{
  "mcpServers": {
    "notion": {
      "server_url": "https://mcp.notion.so/v1/sse",
      "server_description": "Access your Notion workspace",
      "auth": {
        "type": "user_oauth2",
        "flow": "authorization_code",
        "provider_id": "notion",
        "authorize_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "scopes": ["read"]
      }
    }
  }
}
```

**Step 2:** Add to `.env`:
```bash
NOTION_CLIENT_ID=your_notion_client_id
NOTION_CLIENT_SECRET=your_notion_client_secret
```

**Step 3:** Register OAuth app with Notion:
- Redirect URI: `https://your-bot.com/oauth/callback`
- Requested scopes: `read`
- Obtain client ID and secret

**Step 4:** Restart bot

**Result:** Users can now authenticate to Notion via bot. No code changes required.

---

## Testing Strategy

### Unit Tests

**Test Coverage:**
- `OAuthProviderManager` methods (authorization URL generation, token exchange, refresh)
- Token encryption/decryption
- MCP Manager authorization injection logic
- Database token CRUD operations

**Mocking:**
- Mock HTTP requests to OAuth endpoints
- Mock database for token storage
- Mock provider metadata responses

### Integration Tests

**Test Scenarios:**
- Full OAuth flow (start → callback → token storage)
- Token refresh flow
- Multi-provider authentication
- Expired token handling
- Missing provider configuration

**Requirements:**
- Test OAuth provider (e.g., GitHub test app)
- Test database instance
- Mock Slack client for notifications

### Manual Testing Checklist

- [ ] Authenticate to new provider (first time)
- [ ] Token refresh after expiration
- [ ] Revoke token externally, verify bot handles gracefully
- [ ] Multiple users with different providers
- [ ] Provider returns error during OAuth (user denies)
- [ ] Invalid state token (CSRF attack simulation)
- [ ] Provider metadata discovery fallback

---

## Deployment Considerations

### SSL Certificate

**Requirement:** OAuth callback URL must use HTTPS.

**Options:**
- Let's Encrypt (free, automated)
- Cloud provider certificate (AWS ACM, GCP managed certs)
- Reverse proxy (Nginx/Caddy) handling SSL termination

### OAuth Callback URL

**Format:** `https://your-bot.com/oauth/callback`

**DNS Configuration:**
- Point subdomain to bot server IP
- Configure reverse proxy to forward `/oauth/*` to Flask/FastAPI app

### Provider Registration

**For each OAuth provider, register OAuth application:**
- Application name: "Your Bot Name"
- Redirect URI: `https://your-bot.com/oauth/callback`
- Requested scopes: Per provider requirements
- Obtain: Client ID, Client Secret

**Provider-Specific Notes:**
- **Atlassian:** Requires `audience: api.atlassian.com` in auth params
- **GitHub:** Scopes are space-separated
- **Linear:** Supports metadata discovery
- **Notion:** Internal integrations vs public OAuth app

### Environment Management

**Development:**
```bash
OAUTH_BASE_URL=http://localhost:5000  # Allow HTTP for local testing
```

**Production:**
```bash
OAUTH_BASE_URL=https://your-bot.com   # Require HTTPS
```

**Secret Management:**
- Use environment variables (12-factor app methodology)
- Consider secrets manager (AWS Secrets Manager, HashiCorp Vault) for production
- Rotate `TOKEN_ENCRYPTION_KEY` periodically

---

## Monitoring and Observability

### Metrics to Track

- OAuth flow success/failure rates (per provider)
- Token refresh success/failure rates
- Active authenticated users (per provider)
- Token expiration distribution
- Average token lifetime

### Logging

**Log Events:**
- OAuth flow initiated (user_id, provider_id)
- OAuth callback received (state, provider_id)
- Token stored successfully
- Token refresh initiated
- Token refresh failed
- Provider error responses

**Log Levels:**
- INFO: Normal OAuth flows
- WARNING: Token refresh failures, expired tokens
- ERROR: OAuth errors, provider unavailable

### Alerts

**Critical Alerts:**
- OAuth provider unreachable (multiple failures)
- Token encryption key invalid
- Database connection failures for token storage

**Warning Alerts:**
- High token refresh failure rate (>10% over 1 hour)
- Unusual number of failed OAuth flows (possible attack)

---

## Future Enhancements

### 1. Token Sharing Across Bot Instances

**Problem:** Multiple bot instances (scaled horizontally) don't share token state.

**Solution:** Centralized token storage (Redis, PostgreSQL with connection pooling)

### 2. Multi-Workspace Support

**Use Case:** User has multiple Atlassian sites, wants to query both.

**Design:** Extend `oauth_tokens` table with `workspace_id` column:
```sql
PRIMARY KEY (user_id, provider, workspace_id)
```

### 3. Granular Scope Management

**Use Case:** Request minimal scopes initially, upgrade when needed.

**Design:**
- Track granted scopes in database
- Detect when API call needs additional scope
- Prompt user to re-authorize with expanded scopes

### 4. User Token Management UI

**Features:**
- List all connected providers
- View granted scopes
- Revoke access to specific provider
- Re-authenticate to refresh permissions

**Implementation:** Slack slash command (e.g., `/bot-auth list`)

---

## Appendix

### A. Provider-Specific Notes

**Atlassian:**
- Requires `audience: api.atlassian.com` in authorization request
- OAuth flow is site-independent (user selects site after auth)
- Access token format: JWT
- Refresh token expiration: 90 days (varies by plan)

**GitHub:**
- Scopes: `repo`, `read:org`, `user`, etc.
- Token format: `ghp_...` prefix
- No refresh tokens (tokens don't expire)
- Supports fine-grained personal access tokens (alternative to OAuth)

**Linear:**
- Supports RFC 8414 metadata discovery
- Scopes: `read`, `write`, `issues:read`, etc.
- Access token expiration: Varies
- Webhook support for real-time updates

**Notion:**
- Two OAuth types: Public integrations vs Internal integrations
- Public integrations require approval process
- Scopes are granular (page-level permissions)
- Token format: `secret_...` prefix

### B. Error Handling Matrix

| Error | Cause | Bot Response | User Action |
|-------|-------|-------------|-------------|
| Missing token | User not authenticated | Prompt with auth link | Click link, authorize |
| Expired token | Token past TTL | Attempt refresh, prompt if fails | Re-authenticate |
| Invalid token | Revoked externally | Prompt to re-auth | Click link, authorize |
| Insufficient scopes | Scope changed/insufficient | Log warning, skip tool | Re-authenticate with new scopes |
| Provider unavailable | Network/provider down | Use other tools, notify user | Retry later |
| OAuth error (user denied) | User declined consent | Notify user, skip tool | Retry or ignore |

### C. Security Checklist

- [ ] Tokens encrypted at rest with Fernet
- [ ] Master encryption key in environment variable (not in code)
- [ ] CSRF protection via state parameter
- [ ] State tokens expire after 10 minutes
- [ ] HTTPS required for OAuth callback (production)
- [ ] Client secrets in environment variables (not in config files)
- [ ] Token scopes validated against requested scopes
- [ ] Failed OAuth attempts logged for audit
- [ ] Token refresh uses refresh token (not re-authentication)
- [ ] Revoked tokens removed from database

---

## Conclusion

This architecture provides a truly modular, provider-agnostic OAuth system for MCP servers. By isolating provider-specific configuration from generic OAuth logic, the bot can support unlimited OAuth providers through simple configuration changes.

The key insight is that OAuth 2.0 is standardized enough that we can build generic flows, with provider differences handled through configuration (scopes, extra params) and environment variables (client credentials).

**For implementers:** Start with Phase 1 (single provider MVP), validate the architecture, then generalize in Phase 2. The investment in generic design pays off immediately when adding the second and third providers.
