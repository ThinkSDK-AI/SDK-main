"""
Fourier SDK - Central Configuration and Registry System

This module provides a centralized configuration system that auto-discovers
agents, workflows, and tools from your project structure. It allows you to:

1. Configure the SDK once, use everywhere
2. Auto-discover resources from standard folder structure
3. Invoke agents/workflows by name without imports
4. Smart path detection and configuration

Example:
    # In your main script or config
    from fourier.config import FourierConfig

    # Initialize once
    config = FourierConfig()
    config.discover_resources()

    # Use anywhere in your codebase
    from fourier.config import get_config

    config = get_config()
    response = config.invoke_agent("my_agent", query="Hello")
"""

import os
import json
import logging
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class FourierPaths:
    """Configuration paths for Fourier resources"""
    base_dir: str = "."
    agents_dir: Optional[str] = None
    agents_file: Optional[str] = None
    workflows_dir: Optional[str] = None
    workflows_file: Optional[str] = None
    tools_dir: Optional[str] = None
    tools_file: Optional[str] = None
    config_file: str = ".fourier_config.json"

    def __post_init__(self):
        """Auto-detect standard paths if not specified"""
        base = Path(self.base_dir)

        # Auto-detect agents
        if self.agents_dir is None and (base / "agents").exists():
            self.agents_dir = str(base / "agents")
        if self.agents_file is None and (base / "agents.py").exists():
            self.agents_file = str(base / "agents.py")

        # Auto-detect workflows
        if self.workflows_dir is None and (base / "workflows").exists():
            self.workflows_dir = str(base / "workflows")
        if self.workflows_file is None and (base / "workflows.py").exists():
            self.workflows_file = str(base / "workflows.py")

        # Auto-detect tools
        if self.tools_dir is None and (base / "tools").exists():
            self.tools_dir = str(base / "tools")
        if self.tools_file is None and (base / "tools.py").exists():
            self.tools_file = str(base / "tools.py")


class ResourceRegistry:
    """Registry for agents, workflows, and tools"""

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.workflows: Dict[str, Any] = {}
        self.tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict] = {
            "agents": {},
            "workflows": {},
            "tools": {}
        }

    def register_agent(self, name: str, agent: Any, metadata: Optional[Dict] = None):
        """Register an agent"""
        self.agents[name] = agent
        self._metadata["agents"][name] = metadata or {}
        logger.info(f"Registered agent: {name}")

    def register_workflow(self, name: str, workflow: Any, metadata: Optional[Dict] = None):
        """Register a workflow"""
        self.workflows[name] = workflow
        self._metadata["workflows"][name] = metadata or {}
        logger.info(f"Registered workflow: {name}")

    def register_tool(self, name: str, tool: Callable, metadata: Optional[Dict] = None):
        """Register a tool"""
        self.tools[name] = tool
        self._metadata["tools"][name] = metadata or {}
        logger.info(f"Registered tool: {name}")

    def get_agent(self, name: str) -> Any:
        """Get agent by name"""
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self.agents.keys())}")
        return self.agents[name]

    def get_workflow(self, name: str) -> Any:
        """Get workflow by name"""
        if name not in self.workflows:
            raise KeyError(f"Workflow '{name}' not found. Available: {list(self.workflows.keys())}")
        return self.workflows[name]

    def get_tool(self, name: str) -> Callable:
        """Get tool by name"""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self.tools.keys())}")
        return self.tools[name]

    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self.agents.keys())

    def list_workflows(self) -> List[str]:
        """List all registered workflow names"""
        return list(self.workflows.keys())

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())

    def get_metadata(self, resource_type: str, name: str) -> Dict:
        """Get metadata for a resource"""
        return self._metadata.get(resource_type, {}).get(name, {})


class FourierConfig:
    """
    Central configuration system for Fourier SDK.

    This class manages global configuration, auto-discovers resources,
    and provides a unified interface for invoking agents, workflows, and tools.

    Usage:
        # Initialize once (typically in main or __init__.py)
        config = FourierConfig(base_dir=".")
        config.discover_resources()

        # Use anywhere
        from fourier.config import get_config
        config = get_config()
        result = config.invoke_agent("my_agent", query="Hello")
    """

    _instance: Optional['FourierConfig'] = None

    def __init__(
        self,
        base_dir: str = ".",
        auto_discover: bool = True,
        paths: Optional[FourierPaths] = None,
        provider_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Fourier configuration.

        Args:
            base_dir: Base directory for resource discovery
            auto_discover: Automatically discover resources on init
            paths: Custom paths configuration
            provider_config: Provider-specific configuration
        """
        self.base_dir = Path(base_dir).resolve()
        self.paths = paths or FourierPaths(base_dir=str(self.base_dir))
        self.registry = ResourceRegistry()
        self.provider_config = provider_config or {}

        # Set as global instance
        FourierConfig._instance = self

        # Load saved configuration if exists
        self._load_config()

        if auto_discover:
            self.discover_resources()

    @classmethod
    def get_instance(cls) -> 'FourierConfig':
        """Get the global FourierConfig instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_config(self):
        """Load saved configuration from file"""
        config_path = self.base_dir / self.paths.config_file

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)

                # Load provider config
                self.provider_config = data.get("provider_config", {})

                # Update paths if specified
                if "paths" in data:
                    for key, value in data["paths"].items():
                        if hasattr(self.paths, key) and value:
                            setattr(self.paths, key, value)

                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")

    def save_config(self):
        """Save current configuration to file"""
        config_path = self.base_dir / self.paths.config_file

        data = {
            "provider_config": self.provider_config,
            "paths": {
                "agents_dir": self.paths.agents_dir,
                "agents_file": self.paths.agents_file,
                "workflows_dir": self.paths.workflows_dir,
                "workflows_file": self.paths.workflows_file,
                "tools_dir": self.paths.tools_dir,
                "tools_file": self.paths.tools_file,
            },
            "registry": {
                "agents": self.registry.list_agents(),
                "workflows": self.registry.list_workflows(),
                "tools": self.registry.list_tools(),
            }
        }

        try:
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def discover_resources(self, interactive: bool = False):
        """
        Auto-discover agents, workflows, and tools from configured paths.

        Args:
            interactive: If True, prompt user for paths if not found
        """
        logger.info("Starting resource discovery...")

        # Discover agents
        self._discover_agents(interactive)

        # Discover workflows
        self._discover_workflows(interactive)

        # Discover tools
        self._discover_tools(interactive)

        # Save discovered configuration
        self.save_config()

        logger.info(f"Discovery complete. Found: "
                   f"{len(self.registry.agents)} agents, "
                   f"{len(self.registry.workflows)} workflows, "
                   f"{len(self.registry.tools)} tools")

    def _discover_agents(self, interactive: bool = False):
        """Discover agents from directory or file"""
        # Try directory first
        if self.paths.agents_dir and Path(self.paths.agents_dir).exists():
            self._load_from_directory(self.paths.agents_dir, "agents")

        # Try file
        elif self.paths.agents_file and Path(self.paths.agents_file).exists():
            self._load_from_file(self.paths.agents_file, "agents")

        # Interactive mode
        elif interactive:
            self._prompt_for_path("agents")

    def _discover_workflows(self, interactive: bool = False):
        """Discover workflows from directory or file"""
        if self.paths.workflows_dir and Path(self.paths.workflows_dir).exists():
            self._load_from_directory(self.paths.workflows_dir, "workflows")

        elif self.paths.workflows_file and Path(self.paths.workflows_file).exists():
            self._load_from_file(self.paths.workflows_file, "workflows")

        elif interactive:
            self._prompt_for_path("workflows")

    def _discover_tools(self, interactive: bool = False):
        """Discover tools from directory or file"""
        if self.paths.tools_dir and Path(self.paths.tools_dir).exists():
            self._load_from_directory(self.paths.tools_dir, "tools")

        elif self.paths.tools_file and Path(self.paths.tools_file).exists():
            self._load_from_file(self.paths.tools_file, "tools")

        elif interactive:
            self._prompt_for_path("tools")

    def _load_from_directory(self, directory: str, resource_type: str):
        """Load resources from a directory"""
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"{resource_type} directory not found: {directory}")
            return

        # Load all .py files in directory
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                self._load_module_resources(str(py_file), resource_type)
            except Exception as e:
                logger.error(f"Error loading {py_file}: {e}")

    def _load_from_file(self, file_path: str, resource_type: str):
        """Load resources from a single file"""
        if not Path(file_path).exists():
            logger.warning(f"{resource_type} file not found: {file_path}")
            return

        try:
            self._load_module_resources(file_path, resource_type)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    def _load_module_resources(self, file_path: str, resource_type: str):
        """Load resources from a Python module"""
        spec = importlib.util.spec_from_file_location(f"fourier_user_{resource_type}", file_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Look for exported resources
        if resource_type == "agents":
            self._extract_agents(module, file_path)
        elif resource_type == "workflows":
            self._extract_workflows(module, file_path)
        elif resource_type == "tools":
            self._extract_tools(module, file_path)

    def _extract_agents(self, module, source_file: str):
        """Extract agents from module"""
        # Look for Agent instances or __agents__ export
        if hasattr(module, '__agents__'):
            agents = module.__agents__
            if isinstance(agents, dict):
                for name, agent in agents.items():
                    self.registry.register_agent(
                        name,
                        agent,
                        metadata={"source": source_file}
                    )
        else:
            # Auto-detect Agent instances
            from agent import Agent
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, Agent):
                    self.registry.register_agent(
                        name,
                        obj,
                        metadata={"source": source_file}
                    )

    def _extract_workflows(self, module, source_file: str):
        """Extract workflows from module"""
        if hasattr(module, '__workflows__'):
            workflows = module.__workflows__
            if isinstance(workflows, dict):
                for name, workflow in workflows.items():
                    self.registry.register_workflow(
                        name,
                        workflow,
                        metadata={"source": source_file}
                    )
        else:
            from workflow import Workflow
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, Workflow):
                    self.registry.register_workflow(
                        name,
                        obj,
                        metadata={"source": source_file}
                    )

    def _extract_tools(self, module, source_file: str):
        """Extract tools from module"""
        if hasattr(module, '__tools__'):
            tools = module.__tools__
            if isinstance(tools, dict):
                for name, tool in tools.items():
                    self.registry.register_tool(
                        name,
                        tool,
                        metadata={"source": source_file}
                    )
        else:
            # Look for functions with @tool decorator or tool_ prefix
            for name in dir(module):
                if name.startswith("tool_") or (hasattr(getattr(module, name), '__is_tool__')):
                    obj = getattr(module, name)
                    if callable(obj):
                        self.registry.register_tool(
                            name,
                            obj,
                            metadata={"source": source_file}
                        )

    def _prompt_for_path(self, resource_type: str):
        """Interactively prompt user for resource path"""
        print(f"\n{resource_type.title()} not found in standard locations.")
        print(f"Expected: {resource_type}/ directory or {resource_type}.py file")

        choice = input(f"Specify custom path for {resource_type}? (y/n): ").strip().lower()

        if choice == 'y':
            path = input(f"Enter path to {resource_type} directory or file: ").strip()

            if Path(path).is_dir():
                setattr(self.paths, f"{resource_type}_dir", path)
                self._load_from_directory(path, resource_type)
            elif Path(path).is_file():
                setattr(self.paths, f"{resource_type}_file", path)
                self._load_from_file(path, resource_type)
            else:
                logger.warning(f"Invalid path: {path}")

    # High-level invocation methods

    def invoke_agent(
        self,
        name: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke an agent by name.

        Args:
            name: Agent name
            query: Query to process
            **kwargs: Additional arguments for agent.run()

        Returns:
            Agent response dictionary
        """
        agent = self.registry.get_agent(name)
        return agent.run(query, **kwargs)

    def invoke_workflow(
        self,
        name: str,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke a workflow by name.

        Args:
            name: Workflow name
            input_data: Input data for workflow
            **kwargs: Additional arguments for workflow.execute()

        Returns:
            Workflow execution result
        """
        workflow = self.registry.get_workflow(name)
        return workflow.execute(input_data, **kwargs)

    def invoke_tool(
        self,
        name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Invoke a tool by name.

        Args:
            name: Tool name
            *args: Positional arguments for tool
            **kwargs: Keyword arguments for tool

        Returns:
            Tool execution result
        """
        tool = self.registry.get_tool(name)
        return tool(*args, **kwargs)

    # Utility methods

    def list_resources(self) -> Dict[str, List[str]]:
        """List all registered resources"""
        return {
            "agents": self.registry.list_agents(),
            "workflows": self.registry.list_workflows(),
            "tools": self.registry.list_tools(),
        }

    def get_resource_info(self, resource_type: str, name: str) -> Dict:
        """Get detailed information about a resource"""
        if resource_type == "agents":
            agent = self.registry.get_agent(name)
            metadata = self.registry.get_metadata("agents", name)
            return {
                "name": name,
                "type": "agent",
                "class": type(agent).__name__,
                "model": getattr(agent, "model", None),
                "config": getattr(agent, "config", None),
                "metadata": metadata
            }
        elif resource_type == "workflows":
            workflow = self.registry.get_workflow(name)
            metadata = self.registry.get_metadata("workflows", name)
            return {
                "name": name,
                "type": "workflow",
                "class": type(workflow).__name__,
                "metadata": metadata
            }
        elif resource_type == "tools":
            tool = self.registry.get_tool(name)
            metadata = self.registry.get_metadata("tools", name)
            return {
                "name": name,
                "type": "tool",
                "callable": tool.__name__,
                "metadata": metadata
            }


# Global instance access
def get_config() -> FourierConfig:
    """
    Get the global FourierConfig instance.

    Returns:
        FourierConfig: The global configuration instance

    Example:
        from fourier.config import get_config

        config = get_config()
        response = config.invoke_agent("my_agent", "Hello")
    """
    return FourierConfig.get_instance()


def tool(func: Callable) -> Callable:
    """
    Decorator to mark a function as a Fourier tool.

    Usage:
        @tool
        def my_custom_tool(arg1: str, arg2: int) -> str:
            return f"Result: {arg1} - {arg2}"
    """
    func.__is_tool__ = True
    return func
