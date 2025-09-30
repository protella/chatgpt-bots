"""
MCP Client Manager for connecting to external services via Model Context Protocol
Handles discovery and invocation of tools from MCP servers
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastmcp import Client
from config import config
from logger import LoggerMixin
import asyncio


class MCPClientManager(LoggerMixin):
    """Manages connections to MCP servers and tool discovery/invocation"""

    def __init__(self, config_path: str = "mcp_config.json"):
        """
        Initialize MCP Client Manager

        Args:
            config_path: Path to MCP configuration file
        """
        super().__init__()
        self.config_path = config_path
        self.config = {}
        self.client = None
        self.tools = []
        self.tool_descriptions = {}
        self.servers = {}
        self.initialized = False

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from file"""
        # Check if config file exists
        if not os.path.exists(self.config_path):
            self.log_info(f"MCP config file not found at {self.config_path}, MCP features disabled")
            return {}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.log_info(f"Loaded MCP configuration with {len(config.get('mcpServers', {}))} servers")
                return config
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON in MCP config file: {e}")
            return {}
        except Exception as e:
            self.log_error(f"Error loading MCP config: {e}")
            return {}

    async def initialize(self) -> bool:
        """
        Connect to MCP servers and discover available tools

        Returns:
            True if at least one server connected successfully
        """
        if self.initialized:
            self.log_debug("MCP Client Manager already initialized")
            return True

        self.config = self._load_config()

        # Check if any servers are configured
        if not self.config.get("mcpServers"):
            self.log_info("No MCP servers configured, MCP features disabled")
            return False

        connected_servers = 0

        # Connect to each configured server
        for server_name, server_config in self.config["mcpServers"].items():
            try:
                self.log_info(f"Connecting to MCP server: {server_name}")

                # Create client with server configuration
                # FastMCP Client accepts config dict with url, transport, headers
                if "url" in server_config:
                    # HTTP/SSE server with optional headers
                    from fastmcp.client.transports import StreamableHttpTransport

                    url = server_config["url"]
                    headers = server_config.get("headers", {})

                    self.log_debug(f"Creating transport for {url} with headers: {list(headers.keys())}")

                    # Create transport with headers for authentication
                    transport = StreamableHttpTransport(url, headers=headers)
                    client = Client(transport=transport)
                elif "command" in server_config:
                    # Stdio server
                    client = Client(server_config["command"], server_config.get("args", []))
                else:
                    self.log_error(f"Invalid server config for {server_name}: missing url or command")
                    continue

                # Connect to the server
                await client.__aenter__()

                # Store the client for this server
                self.servers[server_name] = {
                    "client": client,
                    "config": server_config,
                    "tools": []
                }

                # Discover tools from this server
                server_tools = await client.list_tools()

                # Process and store tool information
                for tool in server_tools:
                    tool_info = {
                        "name": tool.name,
                        "server": server_name,
                        "title": getattr(tool, 'title', tool.name),
                        "description": getattr(tool, 'description', ''),
                        "inputSchema": getattr(tool, 'inputSchema', {}),
                    }

                    # Add to our tool registry
                    self.tools.append(tool_info)
                    self.servers[server_name]["tools"].append(tool_info)

                    # Build searchable descriptions for intent matching
                    self.tool_descriptions[f"{server_name}.{tool.name}"] = tool_info

                self.log_info(f"Connected to {server_name}: discovered {len(server_tools)} tools")
                connected_servers += 1

            except Exception as e:
                self.log_error(f"Failed to connect to MCP server {server_name}: {e}")
                # Continue trying other servers
                continue

        if connected_servers > 0:
            self.initialized = True
            self.log_info(f"MCP initialization complete: {connected_servers} servers, {len(self.tools)} total tools")
            return True
        else:
            self.log_warning("No MCP servers connected successfully")
            return False

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools from all connected servers

        Returns:
            List of tool descriptions with metadata
        """
        if not self.initialized:
            await self.initialize()

        return self.tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
        progress_handler: Optional[Any] = None
    ) -> Any:
        """
        Call a specific tool with given arguments

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            server_name: Optional server name if tool exists on multiple servers
            progress_handler: Optional callback for streaming progress updates

        Returns:
            Tool execution result
        """
        if not self.initialized:
            await self.initialize()

        # Find the tool and its server
        tool_key = None

        if server_name:
            # Direct server specified
            tool_key = f"{server_name}.{tool_name}"
        else:
            # Search for tool across all servers
            for key in self.tool_descriptions:
                if key.endswith(f".{tool_name}"):
                    tool_key = key
                    break

        if not tool_key or tool_key not in self.tool_descriptions:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Extract server name from tool key
        server_name = tool_key.split('.')[0]

        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not connected")

        try:
            # Get the client for this server
            client = self.servers[server_name]["client"]

            # Call the tool
            self.log_debug(f"Calling tool {tool_name} on server {server_name} with args: {arguments}")

            # Debug: Check if transport has headers
            if hasattr(client, 'transport') and hasattr(client.transport, 'headers'):
                self.log_debug(f"Transport headers present: {list(client.transport.headers.keys())}")

            # Call tool with progress handler if provided
            result = await client.call_tool(
                tool_name,
                arguments,
                progress_handler=progress_handler
            )

            self.log_info(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            self.log_error(f"Error calling tool {tool_name}: {e}")
            raise

    def get_tools_for_prompt(self) -> str:
        """
        Get a formatted string describing all available tools for LLM context

        Returns:
            Formatted string with tool descriptions
        """
        if not self.tools:
            return "No MCP tools available."

        tool_descriptions = []
        for tool in self.tools:
            desc = f"- **{tool['title']}** (`{tool['name']}`): {tool['description']}"
            tool_descriptions.append(desc)

        return "Available MCP Tools:\n" + "\n".join(tool_descriptions)

    async def select_tool_for_message(
        self,
        message: str,
        openai_client: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to select the best tool for a given message

        Args:
            message: User message to match against tools
            openai_client: OpenAI client for LLM-based selection

        Returns:
            Selected tool info or None if no tool matches
        """
        if not self.tools:
            return None

        # Build prompt for tool selection using the comprehensive prompt from prompts.py
        from prompts import MCP_TOOL_SELECTION_PROMPT

        tools_context = self.get_tools_for_prompt()
        selection_prompt = MCP_TOOL_SELECTION_PROMPT.format(
            message=message,
            tools_context=tools_context
        )

        try:
            # Use the utility model for quick tool selection
            # Pass utility config params and let create_text_response handle model-specific details
            response = await openai_client.create_text_response(
                messages=[{"role": "user", "content": selection_prompt}],
                model=config.utility_model,
                max_tokens=config.utility_max_tokens,
                reasoning_effort=config.utility_reasoning_effort,
                verbosity=config.utility_verbosity
            )

            # Parse the response
            result = json.loads(response)

            if result.get("tool") == "NONE":
                return None

            # Find the full tool info
            tool_name = result.get("tool")
            for tool in self.tools:
                if tool["name"] == tool_name:
                    return {
                        "tool": tool
                    }

            return None

        except json.JSONDecodeError:
            self.log_warning("Failed to parse tool selection response as JSON")
            return None
        except Exception as e:
            self.log_error(f"Error in tool selection: {e}")
            return None

    async def cleanup(self):
        """Clean up MCP connections"""
        if not self.initialized:
            return

        self.log_info("Cleaning up MCP connections...")

        for server_name, server_info in self.servers.items():
            try:
                client = server_info["client"]
                await client.__aexit__(None, None, None)
                self.log_debug(f"Closed connection to {server_name}")
            except Exception as e:
                self.log_error(f"Error closing connection to {server_name}: {e}")

        self.servers.clear()
        self.tools.clear()
        self.tool_descriptions.clear()
        self.initialized = False

        self.log_info("MCP cleanup complete")