"""
MCP (Model Context Protocol) Client for FourierSDK.

This module provides MCP client support with multiple connection methods:
- Remote MCP servers via URL
- Configuration-based MCP connections
- Directory-based MCP tool loading
"""

from .client import MCPClient
from .connectors import URLConnector, ConfigConnector, LocalConnector
from .loader import MCPDirectoryLoader, MCPToolLoader, MCPRegistry
from .config import MCPConfig, MCPServerConfig
from .protocol import MCPTool, MCPConnector, MCPMessage, MCPResponse

__all__ = [
    "MCPClient",
    "URLConnector",
    "ConfigConnector",
    "LocalConnector",
    "MCPDirectoryLoader",
    "MCPToolLoader",
    "MCPRegistry",
    "MCPConfig",
    "MCPServerConfig",
    "MCPTool",
    "MCPConnector",
    "MCPMessage",
    "MCPResponse",
]
