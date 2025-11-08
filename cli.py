#!/usr/bin/env python3
"""
FourierSDK Command Line Interface

Production-grade CLI for managing agents, MCP tools, and running queries.

Usage:
    python cli.py interactive              # Start interactive shell
    python cli.py chat "Your question"     # Quick chat
    python cli.py create-agent             # Create new agent
    python cli.py add-mcp                  # Add MCP tools
    python cli.py list-agents              # List saved agents
    python cli.py --help                   # Show help
"""

import argparse
import sys
import os
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import CLI modules
try:
    from cli_utils import (
        InteractiveShell,
        AgentManager,
        MCPManager,
        ConfigManager,
        print_header,
        print_success,
        print_error,
        print_info,
        print_warning
    )
except ImportError:
    logger.error("CLI utilities not found. Make sure cli_utils.py is in the same directory.")
    sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """
    Setup command line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='FourierSDK CLI - Manage agents, MCP tools, and run queries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py interactive

  # Quick chat
  python cli.py chat "What is quantum computing?"

  # Initialize new project
  python cli.py init
  python cli.py init --name my_project --dir ./projects

  # Create agent with thinking mode
  python cli.py create-agent --name ResearchBot --thinking-mode

  # Add MCP tools
  python cli.py add-mcp --url https://mcp.example.com/api
  python cli.py add-mcp --directory ./mcp_tools
  python cli.py add-mcp --config ./mcp_config.json

  # List agents
  python cli.py list-agents

  # Run saved agent
  python cli.py run --agent MyAgent --query "Your question"

For more information, visit: https://github.com/Fourier-AI/SDK-main
        """
    )

    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=os.path.expanduser('~/.fourier/config.json'),
        help='Path to configuration file (default: ~/.fourier/config.json)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Start interactive shell'
    )
    interactive_parser.add_argument(
        '--agent',
        type=str,
        help='Load specific agent in interactive mode'
    )

    # Init command
    init_parser = subparsers.add_parser(
        'init',
        help='Initialize new Fourier project'
    )
    init_parser.add_argument(
        '--name',
        type=str,
        help='Project name'
    )
    init_parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Base directory for project creation (default: current directory)'
    )
    init_parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Skip interactive prompts and use defaults'
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        'chat',
        help='Quick chat query'
    )
    chat_parser.add_argument(
        'query',
        type=str,
        help='Your question or query'
    )
    chat_parser.add_argument(
        '--provider',
        type=str,
        default='groq',
        choices=['groq', 'openai', 'anthropic', 'together', 'perplexity', 'nebius'],
        help='LLM provider (default: groq)'
    )
    chat_parser.add_argument(
        '--model',
        type=str,
        help='Model name (provider-specific)'
    )
    chat_parser.add_argument(
        '--thinking-mode',
        action='store_true',
        help='Enable thinking mode for research'
    )

    # Create agent
    create_parser = subparsers.add_parser(
        'create-agent',
        help='Create new agent'
    )
    create_parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Agent name'
    )
    create_parser.add_argument(
        '--provider',
        type=str,
        default='groq',
        choices=['groq', 'openai', 'anthropic', 'together', 'perplexity', 'nebius'],
        help='LLM provider (default: groq)'
    )
    create_parser.add_argument(
        '--model',
        type=str,
        help='Model name'
    )
    create_parser.add_argument(
        '--system-prompt',
        type=str,
        help='System prompt for agent'
    )
    create_parser.add_argument(
        '--thinking-mode',
        action='store_true',
        help='Enable thinking mode'
    )
    create_parser.add_argument(
        '--thinking-depth',
        type=int,
        default=2,
        help='Thinking mode depth (1-5, default: 2)'
    )
    create_parser.add_argument(
        '--save',
        action='store_true',
        help='Save agent configuration'
    )

    # Add MCP tools
    mcp_parser = subparsers.add_parser(
        'add-mcp',
        help='Add MCP tools'
    )
    mcp_group = mcp_parser.add_mutually_exclusive_group(required=True)
    mcp_group.add_argument(
        '--url',
        type=str,
        help='Remote MCP server URL'
    )
    mcp_group.add_argument(
        '--config',
        type=str,
        help='Path to MCP configuration file'
    )
    mcp_group.add_argument(
        '--directory',
        type=str,
        help='Path to MCP tools directory'
    )
    mcp_parser.add_argument(
        '--name',
        type=str,
        help='Name for this MCP source'
    )
    mcp_parser.add_argument(
        '--agent',
        type=str,
        help='Add MCP tools to specific agent'
    )

    # List agents
    list_parser = subparsers.add_parser(
        'list-agents',
        help='List saved agents'
    )
    list_parser.add_argument(
        '--details',
        action='store_true',
        help='Show detailed information'
    )

    # List MCP tools
    list_mcp_parser = subparsers.add_parser(
        'list-mcp',
        help='List MCP tools'
    )
    list_mcp_parser.add_argument(
        '--agent',
        type=str,
        help='List MCP tools for specific agent'
    )

    # Run agent
    run_parser = subparsers.add_parser(
        'run',
        help='Run saved agent'
    )
    run_parser.add_argument(
        '--agent',
        type=str,
        required=True,
        help='Agent name to run'
    )
    run_parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query to send to agent'
    )
    run_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed execution'
    )

    # Delete agent
    delete_parser = subparsers.add_parser(
        'delete-agent',
        help='Delete saved agent'
    )
    delete_parser.add_argument(
        'name',
        type=str,
        help='Agent name to delete'
    )
    delete_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation'
    )

    # Show config
    config_parser = subparsers.add_parser(
        'config',
        help='Show or edit configuration'
    )
    config_parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    config_parser.add_argument(
        '--set',
        type=str,
        metavar='KEY=VALUE',
        help='Set configuration value'
    )
    config_parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset configuration to defaults'
    )

    return parser


def handle_interactive(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle interactive mode.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        shell = InteractiveShell(config_manager, verbose=args.verbose)

        if args.agent:
            # Load specific agent
            if not shell.load_agent(args.agent):
                print_error(f"Agent '{args.agent}' not found")
                return 1

        shell.run()
        return 0

    except KeyboardInterrupt:
        print_info("\nExiting...")
        return 0
    except Exception as e:
        print_error(f"Interactive mode failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_chat(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle quick chat command.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        from fourier import Fourier
        from agent import Agent, AgentConfig

        # Get API key from environment
        api_key = os.getenv(f"{args.provider.upper()}_API_KEY")
        if not api_key:
            print_error(f"API key not found. Set {args.provider.upper()}_API_KEY environment variable")
            return 1

        # Create client
        client = Fourier(api_key=api_key, provider=args.provider)

        # Determine model
        model = args.model
        if not model:
            default_models = {
                'groq': 'mixtral-8x7b-32768',
                'openai': 'gpt-3.5-turbo',
                'anthropic': 'claude-3-sonnet-20240229',
                'together': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            }
            model = default_models.get(args.provider, 'mixtral-8x7b-32768')

        # Create agent
        config = AgentConfig(
            thinking_mode=args.thinking_mode,
            verbose=args.verbose
        )

        agent = Agent(
            client=client,
            name="ChatAgent",
            model=model,
            config=config
        )

        # Run query
        print_info(f"Processing: {args.query}\n")
        response = agent.run(args.query)

        if response['success']:
            print_success(response['output'])
            if args.verbose:
                print_info(f"\nIterations: {response['iterations']}, Tool calls: {response['tool_calls']}")
        else:
            print_error(f"Query failed: {response.get('error', 'Unknown error')}")
            return 1

        return 0

    except Exception as e:
        print_error(f"Chat failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_create_agent(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle agent creation.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        agent_manager = AgentManager(config_manager)

        # Create agent configuration
        agent_config = {
            'name': args.name,
            'provider': args.provider,
            'model': args.model,
            'system_prompt': args.system_prompt,
            'thinking_mode': args.thinking_mode,
            'thinking_depth': args.thinking_depth,
        }

        # Create agent
        if agent_manager.create_agent(agent_config, save=args.save):
            print_success(f"Agent '{args.name}' created successfully")
            if args.save:
                print_info(f"Agent saved to configuration")
            return 0
        else:
            print_error(f"Failed to create agent '{args.name}'")
            return 1

    except Exception as e:
        print_error(f"Agent creation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_add_mcp(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle MCP tool addition.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        mcp_manager = MCPManager(config_manager)

        if args.url:
            success = mcp_manager.add_url(args.url, args.name)
            source_type = "URL"
            source = args.url
        elif args.config:
            success = mcp_manager.add_config(args.config, args.name)
            source_type = "config"
            source = args.config
        elif args.directory:
            success = mcp_manager.add_directory(args.directory, args.name)
            source_type = "directory"
            source = args.directory

        if success:
            print_success(f"MCP {source_type} '{source}' added successfully")

            # Add to agent if specified
            if args.agent:
                agent_manager = AgentManager(config_manager)
                if agent_manager.add_mcp_to_agent(args.agent, source, source_type):
                    print_info(f"MCP tools added to agent '{args.agent}'")

            return 0
        else:
            print_error(f"Failed to add MCP {source_type}")
            return 1

    except Exception as e:
        print_error(f"MCP addition failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_list_agents(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle listing agents.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        agent_manager = AgentManager(config_manager)
        agent_manager.list_agents(detailed=args.details)
        return 0

    except Exception as e:
        print_error(f"Failed to list agents: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_list_mcp(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle listing MCP tools.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        mcp_manager = MCPManager(config_manager)
        mcp_manager.list_tools(agent_name=args.agent)
        return 0

    except Exception as e:
        print_error(f"Failed to list MCP tools: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_run(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle running saved agent.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        agent_manager = AgentManager(config_manager)
        response = agent_manager.run_agent(args.agent, args.query, verbose=args.verbose)

        if response and response.get('success'):
            print_success(response['output'])
            if args.verbose:
                print_info(f"\nIterations: {response['iterations']}, Tool calls: {response['tool_calls']}")
            return 0
        else:
            print_error(f"Agent execution failed: {response.get('error') if response else 'Unknown error'}")
            return 1

    except Exception as e:
        print_error(f"Failed to run agent: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_delete_agent(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle deleting agent.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        agent_manager = AgentManager(config_manager)

        # Confirm deletion unless --force
        if not args.force:
            response = input(f"Delete agent '{args.name}'? (y/N): ")
            if response.lower() != 'y':
                print_info("Deletion cancelled")
                return 0

        if agent_manager.delete_agent(args.name):
            print_success(f"Agent '{args.name}' deleted")
            return 0
        else:
            print_error(f"Agent '{args.name}' not found")
            return 1

    except Exception as e:
        print_error(f"Failed to delete agent: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_config(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle configuration management.

    Args:
        args: Parsed arguments
        config_manager: Configuration manager

    Returns:
        Exit code
    """
    try:
        if args.show:
            config_manager.show_config()
        elif args.set:
            if '=' not in args.set:
                print_error("Invalid format. Use: KEY=VALUE")
                return 1
            key, value = args.set.split('=', 1)
            config_manager.set_value(key.strip(), value.strip())
            print_success(f"Configuration updated: {key} = {value}")
        elif args.reset:
            response = input("Reset configuration to defaults? (y/N): ")
            if response.lower() == 'y':
                config_manager.reset()
                print_success("Configuration reset to defaults")
            else:
                print_info("Reset cancelled")
        else:
            config_manager.show_config()

        return 0

    except Exception as e:
        print_error(f"Configuration management failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def handle_init(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """
    Handle project initialization.

    Args:
        args: Command line arguments
        config_manager: Configuration manager instance

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        from init_project import initialize_project

        print_header("Fourier Project Initialization")

        success = initialize_project(
            base_dir=args.dir,
            interactive=not args.non_interactive
        )

        if success:
            print_success("Project initialized successfully!")
            return 0
        else:
            print_error("Project initialization cancelled or failed")
            return 1

    except ImportError:
        print_error("init_project module not found. Make sure init_project.py is available.")
        return 1
    except Exception as e:
        print_error(f"Error initializing project: {e}")
        logger.exception("Init error")
        return 1


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Initialize configuration manager
    try:
        config_manager = ConfigManager(args.config)
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        return 1

    # Handle commands
    if not args.command:
        parser.print_help()
        return 0

    command_handlers = {
        'interactive': handle_interactive,
        'chat': handle_chat,
        'create-agent': handle_create_agent,
        'add-mcp': handle_add_mcp,
        'list-agents': handle_list_agents,
        'list-mcp': handle_list_mcp,
        'run': handle_run,
        'delete-agent': handle_delete_agent,
        'config': handle_config,
        'init': handle_init,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args, config_manager)
    else:
        print_error(f"Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
