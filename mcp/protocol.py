"""
MCP Protocol definitions and base classes.

Implements the Model Context Protocol specification for tool/resource access.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """
    Represents an MCP tool/function.

    MCP tools follow the JSON-RPC format with JSON Schema parameter definitions.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_fourier_tool(self) -> Dict[str, Any]:
        """Convert MCP tool to Fourier SDK tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "required": self.input_schema.get("required", [])
        }


@dataclass
class MCPResource:
    """Represents an MCP resource (file, data, etc)."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template."""
    name: str
    description: Optional[str] = None
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""


class MCPConnector(ABC):
    """
    Abstract base class for MCP connectors.

    Connectors handle the actual communication with MCP servers,
    whether remote (URL), local (stdio), or configuration-based.
    """

    def __init__(self, name: str = "mcp_connector"):
        """
        Initialize the connector.

        Args:
            name: Identifier for this connector
        """
        self.name = name
        self.connected = False
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """
        Get list of available tools from MCP server.

        Returns:
            List of MCPTool objects
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        pass

    async def list_resources(self) -> List[MCPResource]:
        """
        Get list of available resources from MCP server.

        Returns:
            List of MCPResource objects
        """
        return list(self.resources.values())

    async def read_resource(self, uri: str) -> Optional[str]:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content or None
        """
        return None

    async def list_prompts(self) -> List[MCPPrompt]:
        """
        Get list of available prompts from MCP server.

        Returns:
            List of MCPPrompt objects
        """
        return list(self.prompts.values())

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get a prompt from the MCP server.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Rendered prompt or None
        """
        return None


class MCPMessage:
    """Represents an MCP protocol message (JSON-RPC 2.0)."""

    def __init__(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ):
        """
        Create an MCP message.

        Args:
            method: RPC method name
            params: Method parameters
            id: Message ID for request/response matching
        """
        self.jsonrpc = "2.0"
        self.method = method
        self.params = params or {}
        self.id = id

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        msg = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params
        }
        if self.id is not None:
            msg["id"] = self.id
        return msg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create message from dictionary."""
        return cls(
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id")
        )


class MCPResponse:
    """Represents an MCP protocol response."""

    def __init__(
        self,
        result: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ):
        """
        Create an MCP response.

        Args:
            result: Response result (if successful)
            error: Error information (if failed)
            id: Message ID matching the request
        """
        self.jsonrpc = "2.0"
        self.result = result
        self.error = error
        self.id = id

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        msg = {
            "jsonrpc": self.jsonrpc,
            "id": self.id
        }
        if self.error:
            msg["error"] = self.error
        else:
            msg["result"] = self.result
        return msg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        """Create response from dictionary."""
        return cls(
            result=data.get("result"),
            error=data.get("error"),
            id=data.get("id")
        )

    @property
    def success(self) -> bool:
        """Check if response indicates success."""
        return self.error is None
