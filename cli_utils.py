"""
CLI utilities for FourierSDK.

Provides helper classes and functions for CLI operations including:
- Agent management
- MCP tool management
- Configuration management
- Interactive shell
- Output formatting
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import cmd

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print header text."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}{text}{Colors.ENDC}")


class ConfigManager:
    """Manage CLI configuration."""

    DEFAULT_CONFIG = {
        'default_provider': 'groq',
        'default_model': {
            'groq': 'mixtral-8x7b-32768',
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-sonnet-20240229',
            'together': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        },
        'agents': {},
        'mcp_sources': {},
        'verbose': False,
        'thinking_mode_default': False,
        'thinking_depth_default': 2
    }

    def __init__(self, config_path: str):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    return {**self.DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()

    def _save_config(self, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses self.config if None)
        """
        config = config or self.config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self._save_config()

    def set_value(self, key: str, value: str):
        """Set configuration value from string."""
        # Try to parse as JSON for complex types
        try:
            parsed_value = json.loads(value)
            self.set(key, parsed_value)
        except:
            # Use as string
            self.set(key, value)

    def show_config(self):
        """Display current configuration."""
        print_header("Current Configuration")
        print(json.dumps(self.config, indent=2))

    def reset(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self._save_config()


class AgentManager:
    """Manage agents."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize agent manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def create_agent(self, agent_config: Dict[str, Any], save: bool = True) -> bool:
        """
        Create a new agent.

        Args:
            agent_config: Agent configuration
            save: Whether to save agent configuration

        Returns:
            True if successful
        """
        try:
            from fourier import Fourier
            from agent import Agent, AgentConfig

            name = agent_config['name']

            # Get API key
            provider = agent_config.get('provider', 'groq')
            api_key = os.getenv(f"{provider.upper()}_API_KEY")

            if not api_key:
                print_error(f"API key not found. Set {provider.upper()}_API_KEY environment variable")
                return False

            # Create Fourier client
            client = Fourier(api_key=api_key, provider=provider)

            # Determine model
            model = agent_config.get('model')
            if not model:
                default_models = self.config_manager.get('default_model', {})
                model = default_models.get(provider, 'mixtral-8x7b-32768')

            # Create agent config
            config = AgentConfig(
                thinking_mode=agent_config.get('thinking_mode', False),
                thinking_depth=agent_config.get('thinking_depth', 2),
                verbose=agent_config.get('verbose', False)
            )

            # Create agent
            agent = Agent(
                client=client,
                name=name,
                model=model,
                system_prompt=agent_config.get('system_prompt'),
                config=config
            )

            # Save configuration if requested
            if save:
                agents = self.config_manager.get('agents', {})
                agents[name] = {
                    'provider': provider,
                    'model': model,
                    'system_prompt': agent_config.get('system_prompt'),
                    'thinking_mode': agent_config.get('thinking_mode', False),
                    'thinking_depth': agent_config.get('thinking_depth', 2),
                    'mcp_sources': []
                }
                self.config_manager.set('agents', agents)

            return True

        except Exception as e:
            logger.error(f"Failed to create agent: {e}", exc_info=True)
            return False

    def load_agent(self, name: str):
        """
        Load saved agent.

        Args:
            name: Agent name

        Returns:
            Agent instance or None
        """
        try:
            agents = self.config_manager.get('agents', {})
            if name not in agents:
                return None

            agent_config = agents[name]

            from fourier import Fourier
            from agent import Agent, AgentConfig

            # Get API key
            provider = agent_config.get('provider', 'groq')
            api_key = os.getenv(f"{provider.upper()}_API_KEY")

            if not api_key:
                print_error(f"API key not found. Set {provider.upper()}_API_KEY environment variable")
                return None

            # Create client
            client = Fourier(api_key=api_key, provider=provider)

            # Create agent config
            config = AgentConfig(
                thinking_mode=agent_config.get('thinking_mode', False),
                thinking_depth=agent_config.get('thinking_depth', 2),
                verbose=agent_config.get('verbose', False)
            )

            # Create agent
            agent = Agent(
                client=client,
                name=name,
                model=agent_config.get('model'),
                system_prompt=agent_config.get('system_prompt'),
                config=config
            )

            # Load MCP tools
            mcp_sources = agent_config.get('mcp_sources', [])
            for source in mcp_sources:
                self._load_mcp_for_agent(agent, source)

            return agent

        except Exception as e:
            logger.error(f"Failed to load agent: {e}", exc_info=True)
            return None

    def _load_mcp_for_agent(self, agent, mcp_source: Dict[str, Any]):
        """Load MCP tools for agent."""
        try:
            source_type = mcp_source.get('type')
            source_value = mcp_source.get('value')

            if source_type == 'url':
                agent.register_mcp_url(source_value)
            elif source_type == 'config':
                agent.register_mcp_config(source_value)
            elif source_type == 'directory':
                agent.register_mcp_directory(source_value)

        except Exception as e:
            logger.warning(f"Failed to load MCP source: {e}")

    def list_agents(self, detailed: bool = False):
        """List all saved agents."""
        agents = self.config_manager.get('agents', {})

        if not agents:
            print_info("No saved agents")
            return

        print_header("Saved Agents")

        for name, config in agents.items():
            print(f"\n{Colors.BOLD}{name}{Colors.ENDC}")
            if detailed:
                print(f"  Provider: {config.get('provider')}")
                print(f"  Model: {config.get('model')}")
                print(f"  Thinking Mode: {config.get('thinking_mode', False)}")
                if config.get('system_prompt'):
                    prompt = config['system_prompt']
                    print(f"  System Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  System Prompt: {prompt}")
                mcp_count = len(config.get('mcp_sources', []))
                if mcp_count > 0:
                    print(f"  MCP Sources: {mcp_count}")

    def run_agent(self, name: str, query: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Run saved agent with query."""
        agent = self.load_agent(name)
        if not agent:
            return None

        try:
            response = agent.run(query)
            return response
        except Exception as e:
            logger.error(f"Failed to run agent: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def delete_agent(self, name: str) -> bool:
        """Delete saved agent."""
        agents = self.config_manager.get('agents', {})
        if name in agents:
            del agents[name]
            self.config_manager.set('agents', agents)
            return True
        return False

    def add_mcp_to_agent(self, agent_name: str, source: str, source_type: str) -> bool:
        """Add MCP source to agent."""
        agents = self.config_manager.get('agents', {})
        if agent_name not in agents:
            return False

        mcp_sources = agents[agent_name].get('mcp_sources', [])
        mcp_sources.append({'type': source_type, 'value': source})
        agents[agent_name]['mcp_sources'] = mcp_sources

        self.config_manager.set('agents', agents)
        return True


class MCPManager:
    """Manage MCP tools."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize MCP manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def add_url(self, url: str, name: Optional[str] = None) -> bool:
        """Add MCP URL source."""
        try:
            from mcp import MCPClient

            client = MCPClient()
            success = client.add_url(url, name)

            if success:
                # Save to config
                sources = self.config_manager.get('mcp_sources', {})
                source_name = name or url
                sources[source_name] = {'type': 'url', 'value': url}
                self.config_manager.set('mcp_sources', sources)

            return success

        except Exception as e:
            logger.error(f"Failed to add MCP URL: {e}", exc_info=True)
            return False

    def add_config(self, config_path: str, name: Optional[str] = None) -> bool:
        """Add MCP config source."""
        try:
            from mcp import MCPClient

            client = MCPClient()
            success = client.add_config(config_path, name)

            if success:
                # Save to config
                sources = self.config_manager.get('mcp_sources', {})
                source_name = name or config_path
                sources[source_name] = {'type': 'config', 'value': config_path}
                self.config_manager.set('mcp_sources', sources)

            return success

        except Exception as e:
            logger.error(f"Failed to add MCP config: {e}", exc_info=True)
            return False

    def add_directory(self, directory: str, name: Optional[str] = None) -> bool:
        """Add MCP directory source."""
        try:
            from mcp import MCPClient

            client = MCPClient()
            success = client.add_directory(directory, name)

            if success:
                # Save to config
                sources = self.config_manager.get('mcp_sources', {})
                source_name = name or directory
                sources[source_name] = {'type': 'directory', 'value': directory}
                self.config_manager.set('mcp_sources', sources)

            return success

        except Exception as e:
            logger.error(f"Failed to add MCP directory: {e}", exc_info=True)
            return False

    def list_tools(self, agent_name: Optional[str] = None):
        """List MCP tools."""
        if agent_name:
            # List tools for specific agent
            agents = self.config_manager.get('agents', {})
            if agent_name not in agents:
                print_error(f"Agent '{agent_name}' not found")
                return

            mcp_sources = agents[agent_name].get('mcp_sources', [])
            print_header(f"MCP Tools for Agent '{agent_name}'")

            if not mcp_sources:
                print_info("No MCP sources configured")
                return

            for source in mcp_sources:
                print(f"\n{Colors.BOLD}{source['type'].upper()}: {source['value']}{Colors.ENDC}")

        else:
            # List all MCP sources
            sources = self.config_manager.get('mcp_sources', {})

            if not sources:
                print_info("No MCP sources configured")
                return

            print_header("MCP Sources")

            for name, source in sources.items():
                print(f"\n{Colors.BOLD}{name}{Colors.ENDC}")
                print(f"  Type: {source['type']}")
                print(f"  Value: {source['value']}")


class InteractiveShell(cmd.Cmd):
    """Interactive shell for FourierSDK CLI."""

    intro = f"""
{Colors.HEADER}{Colors.BOLD}╔═══════════════════════════════════════════════════════════╗
║              FourierSDK Interactive Shell                 ║
║                                                           ║
║  Type 'help' for commands, 'exit' to quit               ║
╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
    """

    prompt = f"{Colors.OKGREEN}fourier> {Colors.ENDC}"

    def __init__(self, config_manager: ConfigManager, verbose: bool = False):
        """
        Initialize interactive shell.

        Args:
            config_manager: Configuration manager instance
            verbose: Enable verbose output
        """
        super().__init__()
        self.config_manager = config_manager
        self.agent_manager = AgentManager(config_manager)
        self.mcp_manager = MCPManager(config_manager)
        self.current_agent = None
        self.verbose = verbose

    def do_exit(self, arg):
        """Exit the interactive shell."""
        print_info("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the interactive shell."""
        return self.do_exit(arg)

    def do_agent(self, arg):
        """Create or switch agent. Usage: agent [name]"""
        if not arg:
            if self.current_agent:
                print_info(f"Current agent: {self.current_agent.name}")
            else:
                print_info("No agent loaded")
            return

        # Try to load agent
        agent = self.agent_manager.load_agent(arg)
        if agent:
            self.current_agent = agent
            print_success(f"Loaded agent: {arg}")
            self.prompt = f"{Colors.OKGREEN}fourier({arg})> {Colors.ENDC}"
        else:
            print_warning(f"Agent '{arg}' not found. Create it? (y/N): ")
            response = input()
            if response.lower() == 'y':
                self.do_create_agent(arg)

    def do_create_agent(self, arg):
        """Create new agent interactively. Usage: create-agent [name]"""
        name = arg if arg else input("Agent name: ")

        print_info("\nAgent Configuration:")
        provider = input(f"Provider (groq/openai/anthropic/together) [groq]: ") or 'groq'
        model = input(f"Model [auto]: ") or None
        system_prompt = input(f"System prompt [auto]: ") or None

        thinking_mode_input = input(f"Enable thinking mode? (y/N): ")
        thinking_mode = thinking_mode_input.lower() == 'y'

        thinking_depth = 2
        if thinking_mode:
            depth_input = input(f"Thinking depth (1-5) [2]: ")
            if depth_input:
                thinking_depth = int(depth_input)

        save_input = input(f"Save agent? (Y/n): ")
        save = save_input.lower() != 'n'

        agent_config = {
            'name': name,
            'provider': provider,
            'model': model,
            'system_prompt': system_prompt,
            'thinking_mode': thinking_mode,
            'thinking_depth': thinking_depth,
            'verbose': self.verbose
        }

        if self.agent_manager.create_agent(agent_config, save=save):
            print_success(f"Agent '{name}' created")
            # Load the agent
            self.current_agent = self.agent_manager.load_agent(name)
            self.prompt = f"{Colors.OKGREEN}fourier({name})> {Colors.ENDC}"
        else:
            print_error("Failed to create agent")

    def do_chat(self, arg):
        """Chat with current agent. Usage: chat <message>"""
        if not self.current_agent:
            print_error("No agent loaded. Use 'agent <name>' to load an agent")
            return

        if not arg:
            print_error("Usage: chat <message>")
            return

        try:
            print_info("Processing...\n")
            response = self.current_agent.run(arg)

            if response['success']:
                print(f"\n{response['output']}\n")
                if self.verbose:
                    print_info(f"Iterations: {response['iterations']}, Tool calls: {response['tool_calls']}")
            else:
                print_error(f"Error: {response.get('error', 'Unknown error')}")

        except Exception as e:
            print_error(f"Chat failed: {e}")

    def do_add_mcp(self, arg):
        """Add MCP tools. Usage: add-mcp <url|config|directory> <value>"""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            print_error("Usage: add-mcp <url|config|directory> <value>")
            return

        source_type, value = parts

        if source_type == 'url':
            success = self.mcp_manager.add_url(value)
        elif source_type == 'config':
            success = self.mcp_manager.add_config(value)
        elif source_type == 'directory':
            success = self.mcp_manager.add_directory(value)
        else:
            print_error(f"Unknown source type: {source_type}")
            return

        if success:
            print_success(f"MCP {source_type} added")

            # Add to current agent if loaded
            if self.current_agent:
                add_to_agent = input(f"Add to current agent '{self.current_agent.name}'? (Y/n): ")
                if add_to_agent.lower() != 'n':
                    self.agent_manager.add_mcp_to_agent(self.current_agent.name, value, source_type)
                    print_info(f"MCP added to agent '{self.current_agent.name}'")
        else:
            print_error(f"Failed to add MCP {source_type}")

    def do_list_agents(self, arg):
        """List all agents. Usage: list-agents [-d]"""
        detailed = '-d' in arg or '--detailed' in arg
        self.agent_manager.list_agents(detailed=detailed)

    def do_list_mcp(self, arg):
        """List MCP tools. Usage: list-mcp [agent_name]"""
        self.mcp_manager.list_tools(agent_name=arg if arg else None)

    def do_verbose(self, arg):
        """Toggle verbose mode. Usage: verbose [on|off]"""
        if arg:
            self.verbose = arg.lower() in ['on', 'true', '1', 'yes']
        else:
            self.verbose = not self.verbose

        print_info(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")

        if self.current_agent:
            self.current_agent.config.verbose = self.verbose

    def do_config(self, arg):
        """Show configuration. Usage: config"""
        self.config_manager.show_config()

    def load_agent(self, name: str) -> bool:
        """Load agent by name."""
        agent = self.agent_manager.load_agent(name)
        if agent:
            self.current_agent = agent
            self.prompt = f"{Colors.OKGREEN}fourier({name})> {Colors.ENDC}"
            return True
        return False

    def run(self):
        """Run the interactive shell."""
        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print_info("\nExiting...")

    def emptyline(self):
        """Handle empty line (do nothing)."""
        pass

    def default(self, line):
        """Handle unknown commands."""
        if self.current_agent and not line.startswith(('exit', 'quit', 'help')):
            # Treat as chat message
            self.do_chat(line)
        else:
            print_error(f"Unknown command: {line}")
            print_info("Type 'help' for available commands")
