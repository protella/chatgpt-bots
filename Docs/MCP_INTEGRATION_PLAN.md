# MCP (Model Context Protocol) Integration Plan

## Overview

This document outlines the plan to integrate MCP server support into the chatbot using FastMCP, enabling modular integration with external services like ReportPro while maintaining the core bot's neutrality and modularity.

## Current State

### Branch: `feature/reportpro-ai`
- Direct API integration with ReportPro service
- Hardcoded "reportpro" intent in classification system
- Mixed bug fixes that should be in master
- Proprietary code that cannot be pushed to GitHub

### Issues with Current Approach
1. **Coupling**: Bot is directly coupled to ReportPro implementation
2. **Maintenance**: Merge conflicts between GitHub (open source) and Bitbucket (proprietary)
3. **Scalability**: Adding new services requires modifying core bot code
4. **Intent System**: Hardcoded intents limit extensibility

## MCP Architecture Solution

### What is MCP?
Model Context Protocol (MCP) is an open standard by Anthropic for connecting AI assistants to external systems. It provides:
- Standard discovery mechanism for tools and resources
- JSON-RPC 2.0 transport protocol
- Dynamic capability discovery
- Tool metadata and descriptions

### Benefits of MCP Integration
1. **Modularity**: Services are completely decoupled from bot
2. **Dynamic Discovery**: Bot discovers capabilities at runtime
3. **Standards Compliant**: Industry standard protocol
4. **Zero Coupling**: Bot remains neutral and service-agnostic
5. **Easy Integration**: New services just need MCP server implementation

## Implementation Plan

### Phase 1: Extract Bug Fixes to Master

#### Changes to Cherry-Pick
These bug fixes from `feature/reportpro-ai` should go to master:

1. **Document Handling Improvements** (commit: caee566)
   - `base_client.py`: Change error logging to warning for handled cases
   - `message_processor/thread_management.py`: Fix document summarization for oversized documents
   - `message_processor/base.py`: Add fallback to drop oldest messages when smart trim insufficient
   - `config.py`: Add `utility_max_tokens` parameter
   - `message_processor/handlers/image_edit.py`: Use configurable utility_max_tokens

2. **API Parameter Fixes** (commit: f369627)
   - `openai_client/api/responses.py`: Fix "max_output_tokens" parameter usage
   - `openai_client/base.py`: Add safety wrapper for API calls

#### Commands
```bash
# Create bugfix branch from master
git checkout master
git checkout -b bugfix/document-handling

# Cherry-pick specific fixes (need to extract from commits)
git cherry-pick -n caee566  # -n to not auto-commit, allows selective staging
# Stage only the bug fix files, not ReportPro-specific changes
git add base_client.py config.py message_processor/thread_management.py
git commit -m "Fix document handling and improve error logging"

# Create PR to master
```

### Phase 2: Implement MCP Client Support

#### Create Clean MCP Branch
```bash
git checkout master
git checkout -b feature/mcp-integration
```

#### Core Components to Build

##### 1. MCP Client Manager (`mcp_client_manager.py`)
```python
import json
from pathlib import Path
from fastmcp import Client
from logger import LoggerMixin

class MCPClientManager(LoggerMixin):
    def __init__(self, config_path="mcp_config.json"):
        super().__init__()
        self.config = self._load_config(config_path)
        self.client = None
        self.tools = []
        self.tool_descriptions = {}

    async def initialize(self):
        """Connect to MCP servers and discover tools"""
        if not self.config.get("mcpServers"):
            self.log_info("No MCP servers configured")
            return

        self.client = Client(self.config)
        await self.client.__aenter__()

        # Discover available tools
        self.tools = await self.client.list_tools()

        # Build tool context for intent classification
        for tool in self.tools:
            self.tool_descriptions[tool.name] = {
                "title": getattr(tool, 'title', tool.name),
                "description": getattr(tool, 'description', ''),
                "server": self._get_server_for_tool(tool.name)
            }

        self.log_info(f"Discovered {len(self.tools)} MCP tools")
```

##### 2. Configuration Format (`mcp_config.json`)
```json
{
  "mcpServers": {
    "datassential-ai": {
      "transport": "http",
      "url": "http://localhost:8001/mcp",
      "description": "Datassential F&B Market Intelligence"
    }
  }
}
```

##### 3. Dynamic Intent Classification

Modify `openai_client/api/responses.py`:
```python
async def classify_intent(self, messages, mcp_tools=None):
    """Enhanced intent classification with MCP tool awareness"""

    # Build dynamic prompt including MCP tools
    if mcp_tools:
        # Add tool descriptions to classification prompt
        tool_context = self._build_mcp_tool_context(mcp_tools)
        # Enhance the classification prompt with available tools
```

##### 4. Message Processing Integration

Update `message_processor/base.py`:
```python
class MessageProcessor:
    def __init__(self):
        # ... existing init ...
        self.mcp_manager = None

    async def initialize_mcp(self, config_path="mcp_config.json"):
        """Initialize MCP connections"""
        from mcp_client_manager import MCPClientManager
        self.mcp_manager = MCPClientManager(config_path)
        await self.mcp_manager.initialize()
```

### Phase 3: MCP Tool Discovery Protocol

#### How MCP Tools Self-Describe

MCP servers provide tool metadata in this format:
```json
{
  "name": "search_reports",
  "title": "Search F&B Market Reports",
  "description": "Search Datassential's database for food & beverage industry reports. Use when users ask about restaurant data, QSR trends, menu analysis, or flavor profiles.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query"
      }
    },
    "required": ["query"]
  }
}
```

#### Intent Classification Strategy

1. **Primary Classification**: Use existing 5 base intents (new_image, edit_image, vision, ambiguous, none)
2. **Secondary MCP Check**: If intent is "none", check if message matches any MCP tool descriptions
3. **Tool Selection**: Use LLM to select best matching MCP tool based on descriptions
4. **Tool Invocation**: Call selected tool through MCP client

### Phase 4: Migration Path

#### What to Keep from ReportPro Branch

1. **Refined Intent Language**: The improved food/beverage detection patterns from `prompts.py`
2. **Answer Deduplication**: The `check_answer_exists_in_conversation()` logic (generalized)
3. **Ambiguous Intent Handling**: Improved clarification flow

#### What to Replace

- Remove `message_processor/handlers/reportpro.py`
- Remove hardcoded "reportpro" intent
- Remove ReportPro configuration from `config.py`
- Replace with dynamic MCP tool discovery

### Phase 5: Testing Strategy

1. **Unit Tests**: Mock MCP client and tool discovery
2. **Integration Tests**: Test with actual ReportPro MCP server
3. **Compatibility Tests**: Ensure bot works without any MCP servers configured
4. **Dynamic Tests**: Add/remove MCP servers at runtime

## Configuration Examples

### Development Setup
```json
{
  "mcpServers": {
    "datassential-ai": {
      "transport": "http",
      "url": "http://localhost:8001/mcp",
      "description": "Local ReportPro MCP server"
    }
  }
}
```

### Production Setup
```json
{
  "mcpServers": {
    "datassential-ai": {
      "transport": "http",
      "url": "https://mcp.datassential.com/api",
      "headers": {
        "Authorization": "Bearer ${REPORTPRO_API_KEY}"
      },
      "description": "Production ReportPro MCP server"
    },
    "other-service": {
      "transport": "stdio",
      "command": "python",
      "args": ["./mcp_servers/other_server.py"],
      "description": "Another MCP service"
    }
  }
}
```

## Workflow Comparison

### Current Workflow (Problematic)
```
GitHub (main) <---> Local Dev <---> Bitbucket (proprietary)
                        |
                  Merge Conflicts
                   Manual Syncing
```

### New MCP Workflow (Clean)
```
GitHub (main) --> Bot with MCP Client
                        |
                    Discovers
                        |
                  MCP Servers
                 /            \
        ReportPro MCP    Other MCP Services
        (Bitbucket)      (Anywhere)
```

## Key Advantages

1. **Clean Separation**: Bot code stays on GitHub, proprietary services stay private
2. **No Merge Conflicts**: Services are completely decoupled
3. **Dynamic Integration**: Add/remove services without code changes
4. **Standards Based**: Uses industry standard MCP protocol
5. **Extensible**: Any team can add their own MCP server

## Implementation Timeline

1. **Week 1**: Extract bug fixes, create MCP branch
2. **Week 2**: Implement FastMCP client integration
3. **Week 3**: Test with ReportPro MCP server
4. **Week 4**: Documentation and deployment

## Technical Details

### FastMCP Client
- Python library for MCP client implementation
- Supports HTTP, SSE, and stdio transports
- Handles tool discovery and invocation
- Documentation: https://gofastmcp.com/clients/client

### MCP Protocol
- JSON-RPC 2.0 based
- Supports tools, resources, and prompts
- Dynamic capability discovery
- Specification: https://modelcontextprotocol.io/specification

## Success Criteria

1. Bot can discover and use MCP tools without hardcoded knowledge
2. ReportPro functionality works through MCP server
3. Bot remains fully functional without any MCP servers
4. New MCP servers can be added via configuration only
5. No proprietary code in GitHub repository

## Notes

- MCP servers self-describe their capabilities - no separate "intents file" needed
- Tool descriptions guide when they should be used
- Bot uses LLM to match user messages to appropriate tools
- Complete decoupling between bot and services