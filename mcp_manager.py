"""
MCP (Model Context Protocol) Manager
Handles loading, caching, and formatting of MCP server configurations
"""
import json
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from logger import LoggerMixin
from config import config


class MCPManager(LoggerMixin):
    """Manages MCP server configurations and tool discovery"""

    def __init__(self, db=None):
        """
        Initialize MCP Manager

        Args:
            db: Database manager instance for caching tools
        """
        self.db = db
        self.servers = {}  # server_label -> server config
        self.tools_cache = {}  # server_label -> list of tools
        self.log_info("MCPManager initialized")

    def initialize(self):
        """
        Load MCP server configurations and populate cache from database.

        This is a synchronous initialization that should be called during bot startup.
        Tool discovery happens later asynchronously.
        """
        self.log_info("Initializing MCP Manager...")

        # Load server configurations from file
        config_loaded = self._load_config()

        if not config_loaded:
            self.log_warning("No MCP configuration loaded - MCP features will be disabled")
            return

        # Load cached tools from database
        if self.db:
            self._load_cache_from_db()

        self.log_info(f"MCP Manager initialized with {len(self.servers)} server(s)")

    def _load_config(self) -> bool:
        """
        Load MCP server configurations from JSON file.

        Returns:
            True if config was loaded successfully, False otherwise
        """
        config_path = config.mcp_config_path

        if not os.path.exists(config_path):
            self.log_info(f"MCP config file not found: {config_path}")
            return False

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Expected format: {"mcpServers": {"server_label": {...}, ...}}
            if "mcpServers" not in config_data:
                self.log_error(f"Invalid MCP config format: missing 'mcpServers' key")
                return False

            # Validate mcpServers is a dictionary
            if not isinstance(config_data["mcpServers"], dict):
                self.log_error(f"Invalid MCP config: 'mcpServers' must be an object/dict, got {type(config_data['mcpServers']).__name__}")
                return False

            # Validate and filter server configurations
            valid_servers = {}
            for label, server_config in config_data["mcpServers"].items():
                # Check if server config is a dict
                if not isinstance(server_config, dict):
                    self.log_warning(f"Skipping invalid server config for '{label}': not an object/dict")
                    continue

                # Warn if server_url is missing (may fail at runtime)
                if "server_url" not in server_config:
                    self.log_warning(f"Server '{label}' missing 'server_url' - may fail when OpenAI attempts to connect")
                elif server_config["server_url"]:
                    # Validate URL format
                    try:
                        parsed = urlparse(server_config["server_url"])
                        if parsed.scheme not in ['http', 'https']:
                            self.log_warning(f"Server '{label}' has invalid URL scheme '{parsed.scheme}' (expected http/https) - may fail at runtime")
                        if not parsed.netloc:
                            self.log_warning(f"Server '{label}' has invalid URL format (missing domain) - may fail at runtime")
                    except Exception as e:
                        self.log_warning(f"Server '{label}' has malformed server_url: {e}")

                # Server config passed basic validation
                valid_servers[label] = server_config

            self.servers = valid_servers

            # Warn if no valid servers after validation
            if len(self.servers) == 0:
                self.log_warning(f"MCP config loaded but no valid servers found")
                return False

            self.log_info(f"Loaded {len(self.servers)} valid MCP server(s) from {config_path}")

            # Log server labels
            for label in self.servers.keys():
                self.log_debug(f"  - {label}")

            return True

        except json.JSONDecodeError as e:
            self.log_error(f"Failed to parse MCP config file (malformed JSON): {e}")
            return False
        except Exception as e:
            self.log_error(f"Error loading MCP config: {e}", exc_info=True)
            return False

    def _load_cache_from_db(self):
        """Load cached tool definitions from database."""
        if not self.db:
            return

        try:
            cached_tools = self.db.get_mcp_tools()

            # Organize by server label
            for tool in cached_tools:
                server_label = tool['server_label']
                if server_label not in self.tools_cache:
                    self.tools_cache[server_label] = []
                self.tools_cache[server_label].append(tool)

            if cached_tools:
                self.log_info(f"Loaded {len(cached_tools)} cached MCP tool(s) from database")
        except Exception as e:
            self.log_error(f"Error loading MCP tools from database: {e}", exc_info=True)

    def has_mcp_servers(self) -> bool:
        """
        Check if any MCP servers are configured.

        Returns:
            True if at least one server is configured
        """
        return len(self.servers) > 0

    def get_server_labels(self) -> List[str]:
        """
        Get list of configured MCP server labels.

        Returns:
            List of server labels
        """
        return list(self.servers.keys())

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """
        Build MCP tool definitions formatted for OpenAI Responses API.

        Returns:
            List of MCP tool definitions in OpenAI format
        """
        if not self.has_mcp_servers():
            return []

        tools = []

        for server_label, server_config in self.servers.items():
            # Build tool definition in OpenAI's MCP format
            tool_def = {
                "type": "mcp",
                "server_label": server_label
            }

            # Add optional fields if present
            if "server_url" in server_config:
                tool_def["server_url"] = server_config["server_url"]

            if "server_description" in server_config:
                tool_def["server_description"] = server_config["server_description"]

            if "headers" in server_config:
                tool_def["headers"] = server_config["headers"]

            # FUTURE FEATURE: require_approval support
            # Currently hardcoded to "never" because we don't have an approval UI implemented.
            # Other values like "untrusted" or "always" would cause the bot to hang waiting
            # for approval that can never be provided in our stateless Slack architecture.
            # TODO: Implement approval UI flow to support user confirmation before MCP tool execution
            # Config value is ignored for now but preserved for future implementation
            tool_def["require_approval"] = "never"

            if "allowed_tools" in server_config and server_config["allowed_tools"]:
                tool_def["allowed_tools"] = server_config["allowed_tools"]

            tools.append(tool_def)

        return tools

    def cache_discovered_tool(self, server_label: str, tool_name: str,
                             description: Optional[str] = None,
                             input_schema: Optional[str] = None):
        """
        Cache a discovered tool to the database and in-memory cache.

        Args:
            server_label: MCP server label
            tool_name: Tool name
            description: Tool description (optional)
            input_schema: Tool input schema as JSON string (optional)
        """
        if not self.db:
            return

        try:
            # Save to database
            self.db.save_mcp_tool(server_label, tool_name, description, input_schema)

            # Update in-memory cache
            if server_label not in self.tools_cache:
                self.tools_cache[server_label] = []

            # Check if tool already in cache
            for cached_tool in self.tools_cache[server_label]:
                if cached_tool['tool_name'] == tool_name:
                    # Update existing
                    cached_tool['description'] = description
                    cached_tool['input_schema'] = input_schema
                    return

            # Add new tool to cache
            self.tools_cache[server_label].append({
                'tool_name': tool_name,
                'description': description,
                'input_schema': input_schema
            })

            self.log_debug(f"Cached MCP tool: {server_label}:{tool_name}")

        except Exception as e:
            self.log_error(f"Error caching MCP tool: {e}", exc_info=True)

    def get_cached_tools(self, server_label: Optional[str] = None) -> List[Dict]:
        """
        Get cached tools from in-memory cache.

        Args:
            server_label: Optional server label to filter by

        Returns:
            List of cached tool dictionaries
        """
        if server_label:
            return self.tools_cache.get(server_label, [])

        # Return all cached tools
        all_tools = []
        for tools in self.tools_cache.values():
            all_tools.extend(tools)
        return all_tools

    def clear_cache(self, server_label: Optional[str] = None):
        """
        Clear tool cache from both database and memory.

        Args:
            server_label: Optional server label to clear (clears all if not provided)
        """
        if self.db:
            self.db.clear_mcp_tools(server_label)

        if server_label:
            if server_label in self.tools_cache:
                del self.tools_cache[server_label]
                self.log_info(f"Cleared cache for MCP server: {server_label}")
        else:
            self.tools_cache.clear()
            self.log_info("Cleared all MCP tool caches")
