"""
MCP Client - Main interface for Model Context Protocol.

Provides a unified client for connecting to MCP servers and managing tools
across multiple connection types (remote, config-based, local).
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from pathlib import Path

from .protocol import MCPTool, MCPConnector
from .connectors import URLConnector, ConfigConnector, LocalConnector
from .config import MCPConfig, MCPServerConfig
from .loader import MCPDirectoryLoader, MCPToolLoader, MCPRegistry

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Main MCP client for managing connections and tools.

    Provides a unified interface for:
    - Connecting to remote MCP servers
    - Loading MCP servers from configuration files
    - Loading tools from directories
    - Managing multiple MCP sources
    """

    def __init__(self):
        """Initialize MCP client."""
        self.registry = MCPRegistry()
        self.connectors: Dict[str, MCPConnector] = {}
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get or create event loop for async operations.

        Returns:
            Event loop instance
        """
        if self._event_loop is None or self._event_loop.is_closed():
            try:
                self._event_loop = asyncio.get_event_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)

        return self._event_loop

    def _run_async(self, coro):
        """
        Run async coroutine in sync context.

        Args:
            coro: Coroutine to run

        Returns:
            Coroutine result
        """
        loop = self._get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task
            return asyncio.create_task(coro)
        else:
            # Run the coroutine
            return loop.run_until_complete(coro)

    def add_url(
        self,
        url: str,
        name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add a remote MCP server via URL.

        Args:
            url: MCP server URL
            name: Optional connector name (uses URL if not provided)
            headers: Optional HTTP headers

        Returns:
            True if connected successfully

        Example:
            >>> client = MCPClient()
            >>> client.add_url("https://mcp.example.com/api")
            True
        """
        connector_name = name or url

        try:
            connector = URLConnector(url, headers)
            result = self._run_async(connector.connect())

            if result:
                self.connectors[connector_name] = connector
                self.registry.add_connector(connector_name, connector)
                logger.info(f"Connected to remote MCP: {url}")
                return True
            else:
                logger.error(f"Failed to connect to remote MCP: {url}")
                return False

        except Exception as e:
            logger.error(f"Error connecting to MCP URL {url}: {e}", exc_info=True)
            return False

    def add_config(
        self,
        config: Union[str, Dict[str, Any], MCPServerConfig],
        name: Optional[str] = None
    ) -> bool:
        """
        Add MCP server from configuration.

        Args:
            config: Config file path, config dict, or MCPServerConfig instance
            name: Optional connector name

        Returns:
            True if connected successfully

        Example:
            >>> client = MCPClient()
            >>> client.add_config({
            ...     "command": "python",
            ...     "args": ["-m", "mcp_server"],
            ...     "env": {"API_KEY": "123"}
            ... })
            True
        """
        try:
            # Handle different config input types
            if isinstance(config, str):
                # Load from file
                mcp_config = MCPConfig.from_file(config)
                success = True

                for server in mcp_config.get_all_servers():
                    if server.is_local():
                        connector = ConfigConnector(server.to_dict())
                        result = self._run_async(connector.connect())

                        if result:
                            connector_name = name or server.name
                            self.connectors[connector_name] = connector
                            self.registry.add_connector(connector_name, connector)
                            logger.info(f"Connected to MCP server: {server.name}")
                        else:
                            logger.error(f"Failed to connect to MCP server: {server.name}")
                            success = False

                    elif server.is_remote():
                        self.add_url(server.url, server.name, server.headers)

                return success

            elif isinstance(config, dict):
                # Direct config dict
                connector = ConfigConnector(config)
                result = self._run_async(connector.connect())

                if result:
                    connector_name = name or "config_mcp"
                    self.connectors[connector_name] = connector
                    self.registry.add_connector(connector_name, connector)
                    logger.info(f"Connected to MCP server from config")
                    return True
                else:
                    logger.error("Failed to connect to MCP server from config")
                    return False

            elif isinstance(config, MCPServerConfig):
                # MCPServerConfig instance
                if config.is_remote():
                    return self.add_url(config.url, config.name, config.headers)
                else:
                    connector = ConfigConnector(config.to_dict())
                    result = self._run_async(connector.connect())

                    if result:
                        connector_name = name or config.name
                        self.connectors[connector_name] = connector
                        self.registry.add_connector(connector_name, connector)
                        logger.info(f"Connected to MCP server: {config.name}")
                        return True
                    else:
                        logger.error(f"Failed to connect to MCP server: {config.name}")
                        return False

        except Exception as e:
            logger.error(f"Error adding MCP config: {e}", exc_info=True)
            return False

    def add_directory(
        self,
        directory: str,
        name: Optional[str] = None
    ) -> bool:
        """
        Add tools from a directory.

        Args:
            directory: Path to tool directory
            name: Optional connector name (uses directory name if not provided)

        Returns:
            True if loaded successfully

        Example:
            >>> client = MCPClient()
            >>> client.add_directory("./mcp_tools")
            True
        """
        try:
            self.registry.add_directory(directory, name)
            logger.info(f"Loaded tools from directory: {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading directory {directory}: {e}", exc_info=True)
            return False

    def add_tool_file(
        self,
        file_path: str,
        connector_name: Optional[str] = None
    ) -> bool:
        """
        Add tools from a Python file.

        Args:
            file_path: Path to Python file containing tools
            connector_name: Optional connector name

        Returns:
            True if loaded successfully

        Example:
            >>> client = MCPClient()
            >>> client.add_tool_file("./my_tool.py")
            True
        """
        try:
            tools = MCPToolLoader.load_from_file(file_path)

            if tools:
                connector = LocalConnector(tools, name=connector_name or Path(file_path).stem)
                name = connector_name or Path(file_path).stem
                self.connectors[name] = connector
                self.registry.add_connector(name, connector)
                logger.info(f"Loaded {len(tools)} tools from {file_path}")
                return True
            else:
                logger.warning(f"No tools found in {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error loading tool file {file_path}: {e}", exc_info=True)
            return False

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            MCPTool object or None if not found
        """
        return self.registry.get_tool(name)

    def get_all_tools(self) -> List[MCPTool]:
        """
        Get all registered tools.

        Returns:
            List of all MCPTool objects
        """
        return self.registry.get_all_tools()

    def get_tool_names(self) -> List[str]:
        """
        Get list of all tool names.

        Returns:
            List of tool names
        """
        return self.registry.get_tool_names()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        # Find the connector that has this tool
        for connector in self.connectors.values():
            if hasattr(connector, 'tools') and name in connector.tools:
                return self._run_async(connector.call_tool(name, arguments))

        # If tool exists in registry but not in any connector, try to execute directly
        if tool.function:
            try:
                return tool.function(**arguments)
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}", exc_info=True)
                raise

        raise ValueError(f"Tool {name} has no execution method")

    def remove_connector(self, name: str) -> bool:
        """
        Remove a connector and its tools.

        Args:
            name: Connector name

        Returns:
            True if removed, False if not found
        """
        if name in self.connectors:
            del self.connectors[name]
            self.registry.remove_connector(name)
            logger.info(f"Removed connector: {name}")
            return True
        return False

    def get_connectors(self) -> List[str]:
        """
        Get list of all connector names.

        Returns:
            List of connector names
        """
        return list(self.connectors.keys())

    def close(self) -> None:
        """
        Close all connections and cleanup resources.
        """
        for name, connector in self.connectors.items():
            try:
                if hasattr(connector, 'disconnect'):
                    self._run_async(connector.disconnect())
                logger.debug(f"Closed connector: {name}")
            except Exception as e:
                logger.error(f"Error closing connector {name}: {e}", exc_info=True)

        self.connectors.clear()

        if self._event_loop and not self._event_loop.is_closed():
            self._event_loop.close()

        logger.info("MCP client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def to_fourier_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all MCP tools to Fourier SDK tool format.

        Returns:
            List of tool dictionaries compatible with Fourier SDK
        """
        return [tool.to_fourier_tool() for tool in self.get_all_tools()]
