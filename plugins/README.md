# MCP Response Formatter Plugins

This directory holds custom response formatters for MCP servers. Each formatter is maintained as a separate git repository.

## Why Plugins?

The main bot repository (on GitHub) stays neutral and doesn't contain service-specific formatting logic. Custom formatters are added as separate repos in this directory, allowing you to:

- Keep proprietary formatting logic private
- Version control formatters independently
- Share formatters with others who use the same MCP server
- Maintain clean separation between framework and implementation

## Directory Structure

```
plugins/
├── .gitignore              # Ignores all subdirectories (separate repos)
├── README.md               # This file
└── your-mcp-formatter/     # Your formatter repo (gitignored)
    ├── formatter.py        # Must have this file
    └── README.md
```

## Installing a Formatter Plugin

Clone your formatter repository into this directory:

```bash
cd plugins/
git clone https://your-repo.com/your-mcp-formatter.git
```

The bot will automatically discover and load any formatters it finds.

## Creating a Formatter Plugin

Your formatter plugin must contain a `formatter.py` file with this structure:

```python
from typing import Any

def format_response(response_content: str, tool_info: dict, client: Any) -> str:
    """
    Format MCP server response for display.

    Args:
        response_content: Raw HTML/text content from MCP server
        tool_info: Tool metadata (name, description, server, etc.)
        client: Platform client for platform-specific formatting

    Returns:
        Formatted response string ready for display
    """
    # Your formatting logic here
    return formatted_response

# Server name this formatter applies to
SERVER_NAME = "your-mcp-server-name"
```

The bot will automatically register your formatter for the specified `SERVER_NAME`.

## Example

See the Datassential AI formatter as a reference implementation (if available in your deployment).

## Notes

- This directory is gitignored by the main repository
- Each plugin manages its own git history
- Plugins are loaded at bot startup
- If a formatter isn't found, responses are displayed with default formatting