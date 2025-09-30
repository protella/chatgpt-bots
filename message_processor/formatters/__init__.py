"""
MCP Response Formatter Registry

Auto-discovers and loads response formatters from plugins directory.
"""
from .registry import FormatterRegistry

__all__ = ["FormatterRegistry"]