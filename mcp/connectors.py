"""
MCP Connectors for different connection types.

Implements connectors for:
- Remote MCP servers (URL/HTTP)
- Configuration-based connections
- Local MCP servers (stdio)
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
import logging
import requests
import subprocess
import uuid
from .protocol import MCPConnector, MCPTool, MCPMessage, MCPResponse

logger = logging.getLogger(__name__)


class URLConnector(MCPConnector):
    """
    Connector for remote MCP servers via HTTP/HTTPS.

    Connects to MCP servers exposed over HTTP using JSON-RPC 2.0.
    """

    def __init__(
        self,
        url: str,
        name: str = "url_mcp",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        """
        Initialize URL connector.

        Args:
            url: MCP server URL (e.g., "https://example.com/mcp")
            name: Connector identifier
            headers: Optional HTTP headers (e.g., for authentication)
            timeout: Request timeout in seconds
        """
        super().__init__(name)
        self.url = url.rstrip("/")
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.session_id: Optional[str] = None

    async def connect(self) -> bool:
        """
        Connect to remote MCP server.

        Returns:
            True if connection successful
        """
        try:
            # Initialize session
            message = MCPMessage(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "fourier-sdk",
                        "version": "0.1.0"
                    }
                },
                id=str(uuid.uuid4())
            )

            response = requests.post(
                self.url,
                json=message.to_dict(),
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = MCPResponse.from_dict(response.json())

            if result.success:
                self.connected = True
                self.session_id = result.result.get("sessionId")
                logger.info(f"Connected to MCP server: {self.url}")

                # List available tools
                await self.list_tools()

                return True
            else:
                logger.error(f"Connection failed: {result.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.url}: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self.connected = False
        self.session_id = None
        logger.info(f"Disconnected from MCP server: {self.url}")

    async def list_tools(self) -> List[MCPTool]:
        """
        Get list of available tools.

        Returns:
            List of MCPTool objects
        """
        if not self.connected:
            logger.warning("Not connected to MCP server")
            return []

        try:
            message = MCPMessage(
                method="tools/list",
                params={},
                id=str(uuid.uuid4())
            )

            response = requests.post(
                self.url,
                json=message.to_dict(),
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = MCPResponse.from_dict(response.json())

            if result.success:
                tools_data = result.result.get("tools", [])
                self.tools = {}

                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        metadata=tool_data
                    )
                    self.tools[tool.name] = tool

                logger.info(f"Loaded {len(self.tools)} tools from MCP server")
                return list(self.tools.values())
            else:
                logger.error(f"Failed to list tools: {result.error}")
                return []

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")

        try:
            message = MCPMessage(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                },
                id=str(uuid.uuid4())
            )

            response = requests.post(
                self.url,
                json=message.to_dict(),
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = MCPResponse.from_dict(response.json())

            if result.success:
                return result.result.get("content", [])
            else:
                error_msg = result.error.get("message", "Unknown error")
                raise Exception(f"Tool execution failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise


class ConfigConnector(MCPConnector):
    """
    Connector for configuration-based MCP servers.

    Loads MCP server configuration from a config file (similar to claude_desktop_config.json)
    and establishes connection based on the configuration.
    """

    def __init__(self, config: Dict[str, Any], name: str = "config_mcp"):
        """
        Initialize configuration-based connector.

        Args:
            config: MCP server configuration dictionary
            name: Connector identifier

        Config format:
            {
                "command": "path/to/mcp-server",
                "args": ["--option", "value"],
                "env": {"VAR": "value"}
            }
        """
        super().__init__(name)
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.command = config.get("command")
        self.args = config.get("args", [])
        self.env = config.get("env", {})

    async def connect(self) -> bool:
        """
        Start MCP server process based on configuration.

        Returns:
            True if connection successful
        """
        if not self.command:
            logger.error("No command specified in configuration")
            return False

        try:
            import os
            env = os.environ.copy()
            env.update(self.env)

            # Start MCP server process
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1
            )

            # Send initialize message
            init_message = MCPMessage(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "fourier-sdk", "version": "0.1.0"}
                },
                id=str(uuid.uuid4())
            )

            self.process.stdin.write(json.dumps(init_message.to_dict()) + "\n")
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            response = MCPResponse.from_dict(json.loads(response_line))

            if response.success:
                self.connected = True
                logger.info(f"Connected to MCP server: {self.command}")

                # List tools
                await self.list_tools()

                return True
            else:
                logger.error(f"Initialization failed: {response.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    async def disconnect(self) -> None:
        """Stop MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.process = None

        self.connected = False
        logger.info("MCP server process stopped")

    async def list_tools(self) -> List[MCPTool]:
        """Get list of available tools from the MCP server."""
        if not self.connected or not self.process:
            return []

        try:
            message = MCPMessage(
                method="tools/list",
                params={},
                id=str(uuid.uuid4())
            )

            self.process.stdin.write(json.dumps(message.to_dict()) + "\n")
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            result = MCPResponse.from_dict(json.loads(response_line))

            if result.success:
                tools_data = result.result.get("tools", [])
                self.tools = {}

                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        metadata=tool_data
                    )
                    self.tools[tool.name] = tool

                logger.info(f"Loaded {len(self.tools)} tools")
                return list(self.tools.values())
            else:
                return []

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the MCP server."""
        if not self.connected or not self.process:
            raise ConnectionError("Not connected to MCP server")

        try:
            message = MCPMessage(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                },
                id=str(uuid.uuid4())
            )

            self.process.stdin.write(json.dumps(message.to_dict()) + "\n")
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            result = MCPResponse.from_dict(json.loads(response_line))

            if result.success:
                return result.result.get("content", [])
            else:
                raise Exception(f"Tool execution failed: {result.error}")

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise


class LocalConnector(MCPConnector):
    """
    Connector for local Python-based MCP tools.

    Loads tools directly from Python modules without external processes.
    """

    def __init__(self, tools: List[MCPTool], name: str = "local_mcp"):
        """
        Initialize local connector with tools.

        Args:
            tools: List of MCPTool objects with function implementations
            name: Connector identifier
        """
        super().__init__(name)
        for tool in tools:
            self.tools[tool.name] = tool

    async def connect(self) -> bool:
        """Mark connector as connected (no actual connection needed)."""
        self.connected = True
        logger.info(f"Local MCP connector initialized with {len(self.tools)} tools")
        return True

    async def disconnect(self) -> None:
        """Mark connector as disconnected."""
        self.connected = False

    async def list_tools(self) -> List[MCPTool]:
        """Return list of loaded tools."""
        return list(self.tools.values())

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a local tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        if not tool.function:
            raise ValueError(f"Tool '{tool_name}' has no function implementation")

        try:
            # Execute the tool function
            result = tool.function(**arguments)
            return result

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
