# FourierSDK Command Line Interface (CLI)

Comprehensive command-line interface for managing agents, MCP tools, and running queries with FourierSDK.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Interactive Mode](#interactive-mode)
  - [Chat](#chat)
  - [Agent Management](#agent-management)
  - [MCP Management](#mcp-management)
  - [Configuration](#configuration)
- [Examples](#examples)
- [Configuration File](#configuration-file)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

## Installation

The CLI is included with FourierSDK:

```bash
# Clone repository
git clone https://github.com/Fourier-AI/SDK-main.git
cd SDK-main

# Install dependencies
pip install -r requirements.txt

# Make CLI executable (optional)
chmod +x cli.py
```

## Quick Start

### 1. Set up API keys

```bash
export GROQ_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
```

### 2. Quick chat

```bash
python cli.py chat "What is quantum computing?"
```

### 3. Interactive mode

```bash
python cli.py interactive
```

### 4. Create an agent

```bash
python cli.py create-agent --name ResearchBot --thinking-mode --save
```

### 5. Add MCP tools

```bash
python cli.py add-mcp --directory ./mcp_tools
```

## Commands

### Interactive Mode

Start an interactive shell for continuous interaction:

```bash
# Basic interactive mode
python cli.py interactive

# Load specific agent
python cli.py interactive --agent MyAgent
```

**Interactive Commands:**

```
fourier> help                    # Show help
fourier> agent MyAgent           # Load/switch agent
fourier> create-agent            # Create new agent interactively
fourier> chat Hello!             # Chat with current agent
fourier> Hello!                  # Direct chat (shortcut)
fourier> add-mcp url https://... # Add MCP tools
fourier> list-agents             # List all agents
fourier> list-mcp                # List MCP tools
fourier> verbose on              # Enable verbose mode
fourier> config                  # Show configuration
fourier> exit                    # Exit shell
```

### Chat

Quick one-off queries:

```bash
# Basic chat
python cli.py chat "Explain neural networks"

# With specific provider
python cli.py chat "Latest AI news" --provider openai

# With thinking mode
python cli.py chat "Research quantum computing advances" --thinking-mode

# With specific model
python cli.py chat "Hello" --provider openai --model gpt-4
```

**Options:**
- `--provider`: LLM provider (groq, openai, anthropic, together, perplexity, nebius)
- `--model`: Specific model name
- `--thinking-mode`: Enable deep research mode
- `--verbose`: Show detailed execution

### Agent Management

#### Create Agent

```bash
# Basic agent
python cli.py create-agent --name MyAgent

# Agent with thinking mode
python cli.py create-agent \
  --name ResearchBot \
  --provider groq \
  --model mixtral-8x7b-32768 \
  --system-prompt "You are a research assistant" \
  --thinking-mode \
  --thinking-depth 3 \
  --save

# Simple agent without saving
python cli.py create-agent --name TempAgent
```

**Options:**
- `--name`: Agent name (required)
- `--provider`: LLM provider (default: groq)
- `--model`: Model name (auto-detected if not specified)
- `--system-prompt`: Custom system prompt
- `--thinking-mode`: Enable thinking mode
- `--thinking-depth`: Research depth 1-5 (default: 2)
- `--save`: Save agent configuration

#### List Agents

```bash
# List all agents
python cli.py list-agents

# Detailed listing
python cli.py list-agents --details
```

#### Run Agent

```bash
# Run saved agent with query
python cli.py run --agent ResearchBot --query "Latest AI developments"

# With verbose output
python cli.py run --agent MyAgent --query "Hello" --verbose
```

#### Delete Agent

```bash
# Delete agent (with confirmation)
python cli.py delete-agent MyAgent

# Force delete (no confirmation)
python cli.py delete-agent MyAgent --force
```

### MCP Management

#### Add MCP Tools

**Remote MCP Server:**
```bash
python cli.py add-mcp --url https://mcp.example.com/api

# With custom name
python cli.py add-mcp --url https://mcp.example.com/api --name my-mcp

# Add to specific agent
python cli.py add-mcp --url https://api.mcp.com --agent ResearchBot
```

**Configuration File:**
```bash
python cli.py add-mcp --config ./mcp_config.json

# Add to agent
python cli.py add-mcp --config ./mcp_config.json --agent MyAgent
```

**Local Directory:**
```bash
python cli.py add-mcp --directory ./mcp_tools

# With name
python cli.py add-mcp --directory ./mcp_tools --name local-tools
```

#### List MCP Tools

```bash
# List all MCP sources
python cli.py list-mcp

# List MCP tools for specific agent
python cli.py list-mcp --agent ResearchBot
```

### Configuration

#### View Configuration

```bash
# Show current configuration
python cli.py config --show
```

#### Set Configuration Value

```bash
# Set default provider
python cli.py config --set default_provider=openai

# Set complex value (JSON)
python cli.py config --set 'default_model={"groq":"llama-3-70b"}'
```

#### Reset Configuration

```bash
# Reset to defaults
python cli.py config --reset
```

### Global Options

Available with all commands:

```bash
# Verbose output
python cli.py --verbose chat "Hello"

# Debug mode
python cli.py --debug create-agent --name Test

# Custom config file
python cli.py --config ~/my-config.json list-agents
```

## Examples

### Example 1: Quick Research Query

```bash
# Enable thinking mode for deep research
python cli.py chat "What are the latest breakthroughs in fusion energy?" \
  --thinking-mode \
  --verbose
```

### Example 2: Create Research Agent

```bash
# Create agent with thinking mode and MCP tools
python cli.py create-agent \
  --name DeepResearch \
  --provider groq \
  --system-prompt "You are an expert research assistant" \
  --thinking-mode \
  --thinking-depth 4 \
  --save

# Add MCP tools
python cli.py add-mcp --directory ./mcp_tools --agent DeepResearch

# Use the agent
python cli.py run \
  --agent DeepResearch \
  --query "Analyze current trends in quantum computing" \
  --verbose
```

### Example 3: Interactive Workflow

```bash
# Start interactive mode
python cli.py interactive

# Inside shell:
fourier> create-agent
Agent name: ChatBot
Provider (groq/openai/anthropic/together) [groq]: groq
Model [auto]: mixtral-8x7b-32768
System prompt [auto]: You are a helpful assistant
Enable thinking mode? (y/N): n
Save agent? (Y/n): y
✓ Agent 'ChatBot' created

fourier> add-mcp directory ./mcp_tools
✓ MCP directory added
Add to current agent 'ChatBot'? (Y/n): y

fourier> Hello, how are you?
Processing...

I'm doing well, thank you for asking! How can I assist you today?

fourier> exit
```

### Example 4: Multi-Agent Workflow

```bash
# Create specialized agents
python cli.py create-agent \
  --name Researcher \
  --thinking-mode \
  --thinking-depth 3 \
  --save

python cli.py create-agent \
  --name Coder \
  --system-prompt "You are an expert programmer" \
  --save

python cli.py create-agent \
  --name Writer \
  --system-prompt "You are a professional writer" \
  --save

# Add MCP tools to researcher
python cli.py add-mcp --directory ./research_tools --agent Researcher

# Use agents
python cli.py run --agent Researcher --query "Research Python async programming"
python cli.py run --agent Coder --query "Write async code example"
python cli.py run --agent Writer --query "Write documentation for the code"
```

### Example 5: Advanced Configuration

```bash
# Set up custom configuration
python cli.py config --set default_provider=anthropic
python cli.py config --set thinking_mode_default=true
python cli.py config --set thinking_depth_default=3

# Show configuration
python cli.py config --show

# Create agent using defaults
python cli.py create-agent --name AutoAgent --save

# Reset when done
python cli.py config --reset
```

## Configuration File

The CLI uses a JSON configuration file stored at `~/.fourier/config.json`:

```json
{
  "default_provider": "groq",
  "default_model": {
    "groq": "mixtral-8x7b-32768",
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-sonnet-20240229",
    "together": "mistralai/Mixtral-8x7B-Instruct-v0.1"
  },
  "agents": {
    "ResearchBot": {
      "provider": "groq",
      "model": "mixtral-8x7b-32768",
      "system_prompt": "You are a research assistant",
      "thinking_mode": true,
      "thinking_depth": 3,
      "mcp_sources": [
        {
          "type": "directory",
          "value": "./mcp_tools"
        }
      ]
    }
  },
  "mcp_sources": {
    "local-tools": {
      "type": "directory",
      "value": "./mcp_tools"
    }
  },
  "verbose": false,
  "thinking_mode_default": false,
  "thinking_depth_default": 2
}
```

### Configuration Keys

- `default_provider`: Default LLM provider
- `default_model`: Default model per provider
- `agents`: Saved agent configurations
- `mcp_sources`: Global MCP tool sources
- `verbose`: Default verbose mode
- `thinking_mode_default`: Default thinking mode setting
- `thinking_depth_default`: Default thinking depth

### Custom Configuration Location

```bash
# Use custom config file
export FOURIER_CONFIG=~/my-config.json
python cli.py --config ~/my-config.json <command>
```

## Environment Variables

Required environment variables for API keys:

```bash
# Groq
export GROQ_API_KEY="your-groq-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Together AI
export TOGETHER_API_KEY="your-together-api-key"

# Perplexity
export PERPLEXITY_API_KEY="your-perplexity-api-key"

# Nebius
export NEBIUS_API_KEY="your-nebius-api-key"
```

### .env File Support

Create a `.env` file in the project directory:

```bash
# .env
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

The CLI will automatically load these variables if `python-dotenv` is installed.

## Best Practices

### 1. Organize Agents by Purpose

```bash
# Create specialized agents
python cli.py create-agent --name Researcher --thinking-mode --save
python cli.py create-agent --name Coder --save
python cli.py create-agent --name Writer --save
```

### 2. Use Thinking Mode for Research

```bash
# Enable thinking mode for research queries
python cli.py chat "Latest quantum computing advances" --thinking-mode
```

### 3. Save Frequently Used Agents

```bash
# Save agents you use often
python cli.py create-agent --name DailyHelper --save
```

### 4. Manage MCP Tools Centrally

```bash
# Add MCP tools to multiple agents
python cli.py add-mcp --directory ./common_tools
python cli.py add-mcp --directory ./common_tools --agent Agent1
python cli.py add-mcp --directory ./common_tools --agent Agent2
```

### 5. Use Interactive Mode for Experimentation

```bash
# Interactive mode is great for trying things out
python cli.py interactive
```

### 6. Version Control Configuration

```bash
# Export configuration
cp ~/.fourier/config.json ./fourier-config-backup.json

# Import configuration
cp ./fourier-config-backup.json ~/.fourier/config.json
```

### 7. Use Verbose Mode for Debugging

```bash
# Enable verbose mode to see what's happening
python cli.py --verbose run --agent MyAgent --query "Test"
```

### 8. Organize MCP Tools

```
mcp_tools/
├── research/
│   └── tool.py
├── coding/
│   └── tool.py
└── utilities/
    └── tool.py
```

```bash
python cli.py add-mcp --directory ./mcp_tools/research --agent Researcher
python cli.py add-mcp --directory ./mcp_tools/coding --agent Coder
```

## Troubleshooting

### Common Issues

**Issue**: "API key not found"
```bash
# Solution: Set environment variable
export GROQ_API_KEY="your-key"
```

**Issue**: "Agent not found"
```bash
# Solution: List agents to see available names
python cli.py list-agents
```

**Issue**: "MCP directory not found"
```bash
# Solution: Use absolute path
python cli.py add-mcp --directory /full/path/to/mcp_tools
```

**Issue**: "Configuration file error"
```bash
# Solution: Reset configuration
python cli.py config --reset
```

### Debug Mode

```bash
# Enable debug mode for detailed error information
python cli.py --debug <command>
```

### Logs

Check logs for detailed information:

```bash
# CLI logs to stdout
python cli.py --verbose <command> 2>&1 | tee cli.log
```

## Advanced Usage

### Scripting with CLI

```bash
#!/bin/bash
# research.sh - Automated research script

# Create agent
python cli.py create-agent \
  --name ResearchAgent \
  --thinking-mode \
  --thinking-depth 4 \
  --save

# Add MCP tools
python cli.py add-mcp --directory ./research_tools --agent ResearchAgent

# Run queries
python cli.py run --agent ResearchAgent --query "AI trends 2024" > result1.txt
python cli.py run --agent ResearchAgent --query "Quantum computing" > result2.txt
python cli.py run --agent ResearchAgent --query "Fusion energy" > result3.txt

echo "Research complete!"
```

### Batch Processing

```bash
# queries.txt
What is quantum computing?
Explain neural networks
Latest AI developments

# process.sh
while IFS= read -r query; do
  echo "Processing: $query"
  python cli.py chat "$query" --thinking-mode >> results.txt
  echo "---" >> results.txt
done < queries.txt
```

## Integration

### CI/CD Pipeline

```yaml
# .github/workflows/research.yml
name: Automated Research

on:
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  research:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run research
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          python cli.py chat "Latest AI news" --thinking-mode > daily-report.txt
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: research-results
          path: daily-report.txt
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/Fourier-AI/SDK-main/issues
- Documentation: See README.md, AGENT.md, MCP.md
- Examples: See examples/ directory

## License

MIT License - See LICENSE file for details
