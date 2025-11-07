"""
MCP Directory and Tool Loaders.

Provides functionality to load MCP tools from:
- Directory structures
- Python modules
- Configuration files
"""

from typing import Dict, Any, List, Optional, Callable
import os
import json
import importlib.util
import sys
from pathlib import Path
import logging
from .protocol import MCPTool
from .connectors import LocalConnector

logger = logging.getLogger(__name__)


class MCPToolLoader:
    """
    Loader for individual MCP tools from Python files.

    Loads tools defined in Python modules following MCP conventions.
    """

    @staticmethod
    def load_from_file(file_path: str) -> List[MCPTool]:
        """
        Load MCP tools from a Python file.

        The Python file should define:
        - Functions with type hints
        - MCP_TOOLS list or dict containing tool metadata

        Args:
            file_path: Path to Python file

        Returns:
            List of MCPTool objects

        Example file format:
            ```python
            # my_tool.py

            def calculate(operation: str, a: float, b: float) -> float:
                '''Perform arithmetic operations.'''
                if operation == "add":
                    return a + b
                return 0

            MCP_TOOLS = [{
                "name": "calculator",
                "description": "Perform arithmetic",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                },
                "function": calculate
            }]
            ```
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                Path(file_path).stem,
                file_path
            )
            if not spec or not spec.loader:
                logger.error(f"Failed to load spec from {file_path}")
                return []

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            tools = []

            # Look for MCP_TOOLS in module
            if hasattr(module, "MCP_TOOLS"):
                mcp_tools_data = getattr(module, "MCP_TOOLS")

                if isinstance(mcp_tools_data, list):
                    for tool_data in mcp_tools_data:
                        tool = MCPTool(
                            name=tool_data["name"],
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("input_schema", {}),
                            function=tool_data.get("function"),
                            metadata=tool_data.get("metadata", {})
                        )
                        tools.append(tool)
                        logger.info(f"Loaded tool: {tool.name}")

                elif isinstance(mcp_tools_data, dict):
                    tool = MCPTool(
                        name=mcp_tools_data["name"],
                        description=mcp_tools_data.get("description", ""),
                        input_schema=mcp_tools_data.get("input_schema", {}),
                        function=mcp_tools_data.get("function"),
                        metadata=mcp_tools_data.get("metadata", {})
                    )
                    tools.append(tool)
                    logger.info(f"Loaded tool: {tool.name}")

            return tools

        except Exception as e:
            logger.error(f"Error loading tools from {file_path}: {e}")
            return []

    @staticmethod
    def load_from_json(file_path: str) -> List[Dict[str, Any]]:
        """
        Load tool metadata from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of tool metadata dictionaries

        Example JSON format:
            ```json
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
            ```
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            tools_data = data.get("tools", [])
            logger.info(f"Loaded {len(tools_data)} tool definitions from {file_path}")
            return tools_data

        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return []


class MCPDirectoryLoader:
    """
    Loader for MCP tool directories.

    Scans directories for MCP tools and creates a connector with all found tools.

    Directory structure:
        mcp_tools/
            ├── calculator/
            │   ├── tool.py          # Tool implementation
            │   └── tool.json        # Tool metadata (optional)
            ├── search/
            │   ├── tool.py
            │   └── tool.json
            └── config.json          # Directory-level config (optional)
    """

    def __init__(self, base_dir: str):
        """
        Initialize directory loader.

        Args:
            base_dir: Base directory containing MCP tools
        """
        self.base_dir = Path(base_dir)
        self.tools: Dict[str, MCPTool] = {}

    def load_all(self) -> LocalConnector:
        """
        Load all tools from the directory.

        Returns:
            LocalConnector with all loaded tools

        Raises:
            FileNotFoundError: If base directory doesn't exist
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.base_dir}")

        logger.info(f"Loading MCP tools from: {self.base_dir}")

        # Load individual tool files in root
        for file_path in self.base_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            tools = MCPToolLoader.load_from_file(str(file_path))
            for tool in tools:
                self.tools[tool.name] = tool

        # Load tool subdirectories
        for subdir in self.base_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue

            self._load_tool_directory(subdir)

        logger.info(f"Loaded {len(self.tools)} tools from {self.base_dir}")

        # Create connector with loaded tools
        connector = LocalConnector(
            tools=list(self.tools.values()),
            name=f"dir_{self.base_dir.name}"
        )

        return connector

    def _load_tool_directory(self, tool_dir: Path) -> None:
        """
        Load tools from a tool-specific subdirectory.

        Args:
            tool_dir: Path to tool directory
        """
        # Look for tool.py
        tool_file = tool_dir / "tool.py"
        if tool_file.exists():
            tools = MCPToolLoader.load_from_file(str(tool_file))
            for tool in tools:
                self.tools[tool.name] = tool

        # Look for tool.json (metadata only)
        json_file = tool_dir / "tool.json"
        if json_file.exists():
            tools_data = MCPToolLoader.load_from_json(str(json_file))
            # Note: These won't have function implementations
            # They can be used for documentation or remote tools
            for tool_data in tools_data:
                if tool_data["name"] not in self.tools:
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("input_schema", {}),
                        metadata=tool_data.get("metadata", {})
                    )
                    self.tools[tool.name] = tool

    def load_tool(self, tool_name: str) -> Optional[MCPTool]:
        """
        Load a specific tool by name.

        Args:
            tool_name: Name of the tool to load

        Returns:
            MCPTool object or None if not found
        """
        if tool_name in self.tools:
            return self.tools[tool_name]

        # Try to find and load the tool
        tool_dir = self.base_dir / tool_name
        if tool_dir.exists() and tool_dir.is_dir():
            self._load_tool_directory(tool_dir)
            return self.tools.get(tool_name)

        return None

    def get_tool_names(self) -> List[str]:
        """
        Get list of all loaded tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())


class MCPRegistry:
    """
    Registry for managing multiple MCP connectors and tools.

    Allows aggregating tools from multiple sources (directories, URLs, configs).
    """

    def __init__(self):
        """Initialize the registry."""
        self.connectors: Dict[str, Any] = {}
        self.tools: Dict[str, MCPTool] = {}

    def add_connector(self, name: str, connector: Any) -> None:
        """
        Add a connector to the registry.

        Args:
            name: Connector identifier
            connector: MCP connector instance
        """
        self.connectors[name] = connector

        # Import tools from connector
        if hasattr(connector, 'tools'):
            for tool_name, tool in connector.tools.items():
                self.tools[tool_name] = tool

        logger.info(f"Added connector '{name}' with {len(connector.tools)} tools")

    def add_directory(self, directory: str, name: Optional[str] = None) -> None:
        """
        Add a directory of MCP tools.

        Args:
            directory: Path to tool directory
            name: Optional connector name (uses directory name if not provided)
        """
        loader = MCPDirectoryLoader(directory)
        connector = loader.load_all()

        connector_name = name or Path(directory).name
        self.add_connector(connector_name, connector)

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            MCPTool object or None
        """
        return self.tools.get(name)

    def get_all_tools(self) -> List[MCPTool]:
        """
        Get all registered tools.

        Returns:
            List of all MCPTool objects
        """
        return list(self.tools.values())

    def get_tool_names(self) -> List[str]:
        """
        Get list of all tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def remove_connector(self, name: str) -> None:
        """
        Remove a connector and its tools.

        Args:
            name: Connector identifier
        """
        if name in self.connectors:
            connector = self.connectors[name]

            # Remove tools from this connector
            if hasattr(connector, 'tools'):
                for tool_name in list(connector.tools.keys()):
                    if tool_name in self.tools:
                        del self.tools[tool_name]

            del self.connectors[name]
            logger.info(f"Removed connector '{name}'")
