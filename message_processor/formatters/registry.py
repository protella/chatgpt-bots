"""
Formatter Registry for MCP Response Formatting

Auto-discovers formatter plugins from the plugins/ directory and provides
a centralized registry for formatting MCP server responses.
"""
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Callable, Any, Optional
from logger import LoggerMixin


class FormatterRegistry(LoggerMixin):
    """Registry for MCP response formatters"""

    _formatters: Dict[str, Callable] = {}
    _initialized = False

    @classmethod
    def initialize(cls):
        """
        Auto-discover and load formatter plugins from plugins/ directory.

        Looks for formatter.py files in subdirectories and registers them
        based on their SERVER_NAME constant.
        """
        if cls._initialized:
            return

        logger = cls()
        plugins_dir = Path(__file__).parent.parent.parent / "plugins"

        if not plugins_dir.exists():
            logger.log_debug("Plugins directory not found, no formatters loaded")
            cls._initialized = True
            return

        logger.log_info(f"Scanning for formatter plugins in {plugins_dir}")

        # Scan for formatter.py files in plugin directories
        formatter_count = 0
        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith('.'):
                continue

            # Look for client-plugins/chatgpt-bot/formatter.py pattern
            formatter_paths = [
                plugin_dir / "client-plugins" / "chatgpt-bot" / "formatter.py",
                plugin_dir / "formatter.py",  # Also support root-level formatters
            ]

            for formatter_path in formatter_paths:
                if formatter_path.exists():
                    try:
                        cls._load_formatter(formatter_path, logger)
                        formatter_count += 1
                        break  # Found a formatter, stop looking in this plugin
                    except Exception as e:
                        logger.log_error(f"Error loading formatter from {formatter_path}: {e}")

        if formatter_count > 0:
            logger.log_info(f"Loaded {formatter_count} formatter plugin(s): {list(cls._formatters.keys())}")
        else:
            logger.log_debug("No formatter plugins found")

        cls._initialized = True

    @classmethod
    def _load_formatter(cls, formatter_path: Path, logger):
        """
        Load a formatter module and register it.

        Args:
            formatter_path: Path to formatter.py file
            logger: Logger instance for logging
        """
        # Import the formatter module
        spec = importlib.util.spec_from_file_location(
            f"formatter_{formatter_path.parent.name}",
            formatter_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Check for required components
            if not hasattr(module, 'SERVER_NAME'):
                logger.log_warning(f"Formatter at {formatter_path} missing SERVER_NAME constant, skipping")
                return

            if not hasattr(module, 'format_response'):
                logger.log_warning(f"Formatter at {formatter_path} missing format_response function, skipping")
                return

            server_name = module.SERVER_NAME
            format_func = module.format_response

            # Register the formatter
            cls._formatters[server_name] = format_func
            logger.log_info(f"Registered formatter for MCP server: {server_name}")

    @classmethod
    def get_formatter(cls, server_name: str) -> Optional[Callable]:
        """
        Get formatter function for a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Formatter function or None if no formatter registered
        """
        if not cls._initialized:
            cls.initialize()

        return cls._formatters.get(server_name)

    @classmethod
    def format_response(
        cls,
        server_name: str,
        response_content: str,
        tool_info: dict,
        client: Any
    ) -> str:
        """
        Format MCP response using registered formatter or return as-is.

        Args:
            server_name: Name of the MCP server
            response_content: Raw response content
            tool_info: Tool metadata
            client: Platform client

        Returns:
            Formatted response (or original if no formatter found)
        """
        formatter = cls.get_formatter(server_name)

        if formatter:
            try:
                return formatter(response_content, tool_info, client)
            except Exception as e:
                logger = cls()
                logger.log_error(f"Error formatting response with {server_name} formatter: {e}")
                # Fall through to return original content

        # No formatter or error - return original content
        return response_content