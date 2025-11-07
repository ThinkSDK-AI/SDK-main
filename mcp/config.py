"""
MCP Configuration Management.

Provides configuration schema and loading for MCP servers,
compatible with Claude Desktop config format.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server.

    Compatible with Claude Desktop config format:
    {
        "mcpServers": {
            "server-name": {
                "command": "python",
                "args": ["-m", "my_mcp_server"],
                "env": {"API_KEY": "..."}
            }
        }
    }
    """

    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None  # For remote MCP servers
    headers: Optional[Dict[str, str]] = None  # For remote MCP servers

    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> 'MCPServerConfig':
        """
        Create MCPServerConfig from dictionary.

        Args:
            name: Server name
            config: Configuration dictionary

        Returns:
            MCPServerConfig instance
        """
        return cls(
            name=name,
            command=config.get("command", ""),
            args=config.get("args", []),
            env=config.get("env"),
            url=config.get("url"),
            headers=config.get("headers")
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        result = {
            "command": self.command,
            "args": self.args,
        }

        if self.env:
            result["env"] = self.env
        if self.url:
            result["url"] = self.url
        if self.headers:
            result["headers"] = self.headers

        return result

    def is_remote(self) -> bool:
        """Check if this is a remote MCP server."""
        return bool(self.url)

    def is_local(self) -> bool:
        """Check if this is a local stdio MCP server."""
        return bool(self.command) and not self.url


class MCPConfig:
    """
    MCP configuration manager.

    Handles loading and managing MCP server configurations from files
    compatible with Claude Desktop config format.
    """

    def __init__(self):
        """Initialize empty configuration."""
        self.servers: Dict[str, MCPServerConfig] = {}

    @classmethod
    def from_file(cls, config_path: str) -> 'MCPConfig':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            MCPConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid

        Example file format:
            ```json
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "python",
                        "args": ["-m", "mcp_server_filesystem"],
                        "env": {
                            "ROOT_PATH": "/home/user"
                        }
                    },
                    "web-search": {
                        "url": "https://mcp.example.com/search",
                        "headers": {
                            "Authorization": "Bearer token123"
                        }
                    }
                }
            }
            ```
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            config = cls()

            # Handle both Claude Desktop format and simplified format
            servers_data = data.get("mcpServers", data)

            for server_name, server_config in servers_data.items():
                config.add_server(
                    MCPServerConfig.from_dict(server_name, server_config)
                )

            logger.info(f"Loaded {len(config.servers)} MCP servers from {config_path}")
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e
        except Exception as e:
            logger.error(f"Error loading MCP config: {e}", exc_info=True)
            raise

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MCPConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            MCPConfig instance
        """
        config = cls()

        servers_data = config_dict.get("mcpServers", config_dict)

        for server_name, server_config in servers_data.items():
            config.add_server(
                MCPServerConfig.from_dict(server_name, server_config)
            )

        return config

    def add_server(self, server: MCPServerConfig) -> None:
        """
        Add a server configuration.

        Args:
            server: MCPServerConfig instance
        """
        self.servers[server.name] = server
        logger.debug(f"Added MCP server config: {server.name}")

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """
        Get server configuration by name.

        Args:
            name: Server name

        Returns:
            MCPServerConfig or None if not found
        """
        return self.servers.get(name)

    def get_all_servers(self) -> List[MCPServerConfig]:
        """
        Get all server configurations.

        Returns:
            List of MCPServerConfig instances
        """
        return list(self.servers.values())

    def get_remote_servers(self) -> List[MCPServerConfig]:
        """
        Get all remote server configurations.

        Returns:
            List of remote MCPServerConfig instances
        """
        return [s for s in self.servers.values() if s.is_remote()]

    def get_local_servers(self) -> List[MCPServerConfig]:
        """
        Get all local stdio server configurations.

        Returns:
            List of local MCPServerConfig instances
        """
        return [s for s in self.servers.values() if s.is_local()]

    def remove_server(self, name: str) -> bool:
        """
        Remove a server configuration.

        Args:
            name: Server name

        Returns:
            True if removed, False if not found
        """
        if name in self.servers:
            del self.servers[name]
            logger.debug(f"Removed MCP server config: {name}")
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "mcpServers": {
                name: server.to_dict()
                for name, server in self.servers.items()
            }
        }

    def save(self, config_path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            config_path: Path to save configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved MCP config to {config_path}")
