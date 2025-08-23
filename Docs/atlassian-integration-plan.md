# Atlassian Integration Implementation Plan

## Overview
Integrate JIRA and Confluence search capabilities into the chatbot, allowing users to query company information with results automatically scoped to their permissions using their Slack email address.

## Architecture

### Core Principle
- Single service account with broad read access
- Pre-fetch user's accessible projects/spaces based on their email
- Scope all searches to user's accessible resources via JQL/CQL
- No user credentials stored

### Authentication Flow
```
User Query → Extract Slack Email → Check Access Cache
→ If expired/missing: Fetch user's accessible resources
→ Build scoped JQL/CQL queries → Search APIs
→ Return only permitted results → Model interprets → Response with citations
```

## Implementation Components

### 1. New Environment Variables (.env)
```bash
# Atlassian API Configuration
ATLASSIAN_DOMAIN=company.atlassian.net
ATLASSIAN_SERVICE_EMAIL=service-account@company.com
ATLASSIAN_SERVICE_TOKEN=<api-token>
ATLASSIAN_ACCESS_CACHE_TTL=1800  # 30 minutes default
```

### 2. New Module: atlassian_client.py
```python
class AtlassianClient:
    def __init__(self):
        self.domain = os.getenv('ATLASSIAN_DOMAIN')
        self.auth = (
            os.getenv('ATLASSIAN_SERVICE_EMAIL'),
            os.getenv('ATLASSIAN_SERVICE_TOKEN')
        )
        self.cache_ttl = int(os.getenv('ATLASSIAN_ACCESS_CACHE_TTL', 1800))
        self.user_access_cache = {}
        
    def search_for_user(self, user_email: str, query: str) -> dict:
        """Main entry point for user-scoped searches"""
        access = self.get_user_access(user_email)
        
        # Build scoped queries
        jql = self._build_jql(access['projects'], query)
        cql = self._build_cql(access['spaces'], query)
        
        # Parallel search both systems
        jira_results = self.search_jira(jql)
        confluence_results = self.search_confluence(cql)
        
        return self.format_results(jira_results, confluence_results)
```

### 3. Permission Caching System
```python
def get_user_access(self, user_email: str) -> dict:
    """Get or refresh user's accessible resources"""
    # Check cache
    if self._is_cache_valid(user_email):
        return self.user_access_cache[user_email]
    
    # Fetch fresh permissions
    user_id = self._get_atlassian_user_id(user_email)
    
    access_data = {
        'projects': self._get_user_projects(user_id),
        'spaces': self._get_user_spaces(user_email),
        'expires': time.time() + self.cache_ttl,
        'user_id': user_id
    }
    
    self.user_access_cache[user_email] = access_data
    return access_data
```

### 4. Scoped Query Builders
```python
def _build_jql(self, projects: list, query: str) -> str:
    """Build JIRA Query Language with project scope"""
    if not projects:
        return None  # User has no JIRA access
        
    project_filter = f"project IN ({','.join(projects)})"
    text_search = f'text ~ "{query}"'
    
    return f"{project_filter} AND {text_search} ORDER BY updated DESC"

def _build_cql(self, spaces: list, query: str) -> str:
    """Build Confluence Query Language with space scope"""
    if not spaces:
        return None  # User has no Confluence access
        
    space_filter = f"space IN ({','.join(spaces)})"
    text_search = f'text ~ "{query}"'
    
    return f"{space_filter} AND {text_search} AND type IN (page, blogpost)"
```

## API Endpoints

### JIRA APIs
```python
# Get user ID from email
GET /rest/api/3/user/search?query={email}

# Get user's accessible projects
GET /rest/api/3/project/search?expand=permissions

# Search with JQL
GET /rest/api/3/search?jql={jql}&fields=summary,description,status,assignee,updated

# Get issue details
GET /rest/api/3/issue/{issueKey}
```

### Confluence APIs
```python
# Get user's accessible spaces
GET /wiki/rest/api/space?type=global&status=current

# Check space permissions
GET /wiki/rest/api/space/{spaceKey}/permission/check

# Search with CQL
GET /wiki/rest/api/search?cql={cql}&limit=10

# Get page content
GET /wiki/rest/api/content/{pageId}?expand=body.view
```

## Intent Classification Update

### prompts.py Additions
```python
ATLASSIAN_INTENT_KEYWORDS = [
    'jira', 'ticket', 'issue', 'confluence', 'documentation',
    'wiki', 'project', 'epic', 'story', 'bug', 'task',
    'assigned to', 'status of', 'find ticket', 'search for'
]

def classify_intent(message: str) -> str:
    message_lower = message.lower()
    
    # Check for Atlassian search intent
    if any(keyword in message_lower for keyword in ATLASSIAN_INTENT_KEYWORDS):
        return 'atlassian_search'
    
    # Existing intent classification...
```

## Message Processing Flow

### message_processor.py Updates
```python
async def process_message(self, message: str, thread_state: dict) -> str:
    intent = classify_intent(message)
    
    if intent == 'atlassian_search':
        return await self._handle_atlassian_search(message, thread_state)
    
    # Existing intent handling...

async def _handle_atlassian_search(self, message: str, thread_state: dict):
    user_email = thread_state.get('user_email')  # From Slack/Discord
    
    # Search Atlassian systems
    atlassian_client = AtlassianClient()
    results = atlassian_client.search_for_user(user_email, message)
    
    if not results['jira'] and not results['confluence']:
        return "No results found in JIRA or Confluence for your query."
    
    # Format results for model interpretation
    context = self._format_atlassian_context(results)
    
    # Use OpenAI to interpret and respond
    response = await self.openai_client.create_response(
        messages=[
            {"role": "system", "content": "Interpret these Atlassian search results and provide a helpful response with citations."},
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Search results:\n{context}"}
        ]
    )
    
    return self._add_citations(response, results)
```

## Response Formatting

### Citation Format
```python
def _add_citations(self, response: str, results: dict) -> str:
    """Add clickable citations to response"""
    citations = []
    
    # JIRA citations
    for issue in results['jira']:
        key = issue['key']
        url = f"https://{self.domain}/browse/{key}"
        response = response.replace(key, f"[{key}]({url})")
        citations.append(f"[{key}]({url}): {issue['summary']}")
    
    # Confluence citations
    for page in results['confluence']:
        title = page['title']
        url = page['_links']['webui']
        response = response.replace(title, f"[📄 {title}]({url})")
        citations.append(f"[📄 {title}]({url})")
    
    # Add sources footer
    if citations:
        response += "\n\n**Sources:**\n" + "\n".join(citations[:5])
    
    return response
```

## Database Schema Update

### New Table: atlassian_access_cache
```sql
CREATE TABLE atlassian_access_cache (
    user_email TEXT PRIMARY KEY,
    atlassian_user_id TEXT,
    projects TEXT,  -- JSON array of project keys
    spaces TEXT,    -- JSON array of space keys
    expires_at INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX idx_atlassian_cache_expires ON atlassian_access_cache(expires_at);
```

## Error Handling

```python
class AtlassianError(Exception):
    """Base exception for Atlassian integration"""
    pass

class AtlassianAuthError(AtlassianError):
    """Authentication or permission error"""
    pass

class AtlassianSearchError(AtlassianError):
    """Search query error"""
    pass

def handle_atlassian_error(error: Exception) -> str:
    if isinstance(error, AtlassianAuthError):
        return "❌ Unable to verify your Atlassian permissions. Please contact your administrator."
    elif isinstance(error, AtlassianSearchError):
        return "❌ Search failed. Please try rephrasing your query."
    else:
        logger.error(f"Atlassian integration error: {error}")
        return "❌ An error occurred while searching Atlassian. Please try again."
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_atlassian_client.py
class TestAtlassianClient:
    def test_user_access_caching(self):
        """Test that user access is cached properly"""
        
    def test_jql_builder(self):
        """Test JQL query construction with project scope"""
        
    def test_cql_builder(self):
        """Test CQL query construction with space scope"""
        
    def test_permission_refresh(self):
        """Test cache expiration and refresh"""
```

### Integration Tests
```python
# tests/integration/test_atlassian_integration.py
@pytest.mark.integration
class TestAtlassianIntegration:
    def test_real_jira_search(self):
        """Test actual JIRA API search"""
        
    def test_real_confluence_search(self):
        """Test actual Confluence API search"""
        
    def test_user_permission_filtering(self):
        """Test that results respect user permissions"""
```

## Security Considerations

1. **Service Account Security**
   - Use read-only API token
   - Rotate token regularly
   - Monitor usage for anomalies

2. **Permission Validation**
   - Always verify user email matches Slack/Discord email
   - Never expose content from personal spaces
   - Log all search queries for audit

3. **Data Protection**
   - Don't cache sensitive content
   - Clear cache on errors
   - Implement rate limiting per user


## Configuration Example

```bash
# .env additions
ATLASSIAN_DOMAIN=acme.atlassian.net
ATLASSIAN_SERVICE_EMAIL=bot-service@acme.com
ATLASSIAN_SERVICE_TOKEN=ATATT3xFfGF0...
ATLASSIAN_ACCESS_CACHE_TTL=1800
ATLASSIAN_MAX_RESULTS=10
ATLASSIAN_SEARCH_TIMEOUT=30
```

## Usage Examples

### User Queries
```
User: "What's the status of PROJ-1234?"
Bot: PROJ-1234 is currently **In Progress** assigned to John Doe. 
     Last updated: 2 hours ago
     [View in JIRA](https://acme.atlassian.net/browse/PROJ-1234)

User: "Find documentation about deployment process"
Bot: Found 3 relevant pages in Confluence:
     1. [📄 Deployment Guide](link) - Updated last week
     2. [📄 CI/CD Pipeline Overview](link) - Core documentation
     3. [📄 Production Deployment Checklist](link) - Step-by-step guide

User: "Show me all critical bugs assigned to me"
Bot: You have 2 critical bugs:
     - [BUG-456](link): Login fails on mobile devices
     - [BUG-789](link): Data export timeout for large datasets
```

## Monitoring & Metrics

- Track search query volume per user
- Monitor cache hit/miss rates
- Log API response times
- Alert on authentication failures
- Track most searched terms for optimization

## Future Enhancements

1. **Smart Query Expansion**
   - Synonym matching
   - Fuzzy search
   - Natural language to JQL/CQL conversion

2. **Proactive Notifications**
   - Alert users about ticket updates
   - Notify about new documentation

3. **Rich Responses**
   - Include issue attachments
   - Show Confluence page previews
   - Display JIRA field values

4. **Advanced Caching**
   - Pre-fetch popular content
   - Differential cache updates
   - Distributed cache for scale