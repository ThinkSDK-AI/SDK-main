# Model Context Protocol (MCP) Support

FourierSDK includes comprehensive support for the Model Context Protocol (MCP), a standardized protocol for connecting AI systems with external tools and data sources.

## Table of Contents

- [What is MCP?](#what-is-mcp)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Connection Methods](#connection-methods)
  - [Remote MCP (URL)](#remote-mcp-url)
  - [Configuration-Based](#configuration-based)
  - [Directory-Based](#directory-based)
- [Using MCP with Agents](#using-mcp-with-agents)
- [Creating MCP Tools](#creating-mcp-tools)
- [API Reference](#api-reference)
- [Examples](#examples)

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that standardizes how AI applications connect with external tools and data sources. It uses JSON-RPC 2.0 for communication and supports:

- **Remote MCP Servers**: HTTP/HTTPS endpoints providing tools
- **Local MCP Servers**: Subprocess-based servers using stdio
- **Tool Discovery**: Automatic enumeration of available tools
- **Standardized Execution**: Consistent tool calling interface

FourierSDK's MCP implementation allows you to seamlessly integrate MCP tools into your agents and applications.

## Installation

MCP support is included with FourierSDK. No additional packages are required:

```bash
pip install fourier-sdk
```

For development:

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main
pip install -r requirements.txt
```

## Quick Start

### Standalone MCP Client

```python
from mcp import MCPClient

# Create MCP client
client = MCPClient()

# Add a remote MCP server
client.add_url("https://mcp.example.com/api")

# Add a local MCP server from config
client.add_config({
    "command": "python",
    "args": ["-m", "mcp_server_filesystem"],
    "env": {"ROOT_PATH": "/home/user"}
})

# Add tools from a directory
client.add_directory("./mcp_tools")

# List available tools
print(client.get_tool_names())

# Call a tool
result = client.call_tool("search", {"query": "AI research"})
print(result)
```

### MCP with Agents

```python
from fourier import Fourier
from agent import Agent

# Create Fourier client
fourier_client = Fourier(api_key="...", provider="groq")

# Create agent
agent = Agent(
    client=fourier_client,
    name="MCPAgent",
    system_prompt="You are a helpful assistant with access to MCP tools.",
    model="mixtral-8x7b-32768"
)

# Register MCP tools from different sources
agent.register_mcp_url("https://mcp.example.com/api")
agent.register_mcp_config("./mcp_config.json")
agent.register_mcp_directory("./mcp_tools")

# Agent can now use all MCP tools automatically
response = agent.run("Search for recent AI papers and summarize the top result")
print(response["output"])
```

## Connection Methods

FourierSDK supports three methods for connecting to MCP servers:

### Remote MCP (URL)

Connect to remote MCP servers via HTTP/HTTPS:

```python
from mcp import MCPClient

client = MCPClient()

# Basic connection
client.add_url("https://mcp.example.com/api")

# With custom headers (authentication)
client.add_url(
    "https://mcp.example.com/api",
    headers={"Authorization": "Bearer token123"}
)

# With custom name
client.add_url(
    "https://mcp.example.com/api",
    name="my_mcp_server"
)
```

**Use Case**: Connect to hosted MCP services, enterprise MCP servers, or cloud-based tool providers.

### Configuration-Based

Load MCP servers from configuration files (compatible with Claude Desktop format):

```python
from mcp import MCPClient

client = MCPClient()

# From configuration file
client.add_config("./mcp_config.json")

# From configuration dict
client.add_config({
    "command": "python",
    "args": ["-m", "mcp_server"],
    "env": {"API_KEY": "secret123"}
})
```

**Configuration File Format** (`mcp_config.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "env": {
        "ROOT_PATH": "/home/user/documents"
      }
    },
    "database": {
      "command": "node",
      "args": ["./mcp_db_server.js"],
      "env": {
        "DB_CONNECTION": "postgresql://localhost/mydb"
      }
    },
    "remote_api": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer token123"
      }
    }
  }
}
```

**Use Case**: Manage multiple MCP servers, share configurations across team, subprocess-based MCP servers.

### Directory-Based

Load tools from directories containing Python-based MCP tools:

```python
from mcp import MCPClient

client = MCPClient()

# Load from single directory
client.add_directory("./mcp_tools")

# Load from multiple directories
client.add_directory("./mcp_tools")
client.add_directory("./custom_tools")
client.add_directory("./external_tools")
```

**Directory Structure**:

```
mcp_tools/
├── calculator/
│   ├── tool.py          # Tool implementation
│   └── tool.json        # Tool metadata (optional)
├── search/
│   ├── tool.py
│   └── tool.json
└── weather/
    └── tool.py
```

**Tool File Example** (`calculator/tool.py`):

```python
def calculate(operation: str, a: float, b: float) -> float:
    """Perform arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else 0
    return 0

MCP_TOOLS = [{
    "name": "calculator",
    "description": "Perform basic arithmetic operations",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    },
    "function": calculate
}]
```

**Use Case**: Organize tools in filesystem, version control tool libraries, share tools across projects.

## Using MCP with Agents

Agents provide the most powerful way to use MCP tools with automatic execution and conversation management:

```python
from fourier import Fourier
from agent import Agent, AgentConfig

# Create Fourier client
client = Fourier(api_key="...", provider="groq")

# Create agent with configuration
agent = Agent(
    client=client,
    name="ResearchAgent",
    system_prompt="You are a research assistant with access to various tools.",
    model="mixtral-8x7b-32768",
    config=AgentConfig(
        verbose=True,
        max_iterations=10,
        auto_execute_tools=True
    )
)

# Method 1: Register MCP URL
agent.register_mcp_url("https://mcp.example.com/api")

# Method 2: Register from config file
agent.register_mcp_config("./mcp_config.json")

# Method 3: Register single directory
agent.register_mcp_directory("./mcp_tools")

# Method 4: Register multiple directories
agent.register_mcp_directories([
    "./mcp_tools",
    "./custom_tools",
    "./external_tools"
])

# Check registered tools
print(f"Available tools: {agent.get_tool_names()}")
print(f"MCP tools: {agent.get_mcp_tool_names()}")

# Use the agent - it will automatically use MCP tools as needed
response = agent.run(
    "Search for recent research on quantum computing and summarize the findings"
)

print(f"Response: {response['output']}")
print(f"Tools used: {response['tool_calls']}")
print(f"Iterations: {response['iterations']}")
```

### Mixing Regular and MCP Tools

You can use both regular tools and MCP tools together:

```python
from fourier import Fourier
from agent import Agent

client = Fourier(api_key="...", provider="groq")
agent = Agent(client=client, model="mixtral-8x7b-32768")

# Register regular Python function as tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent.register_tool(
    name="get_time",
    description="Get current date and time",
    parameters={"type": "object", "properties": {}},
    function=get_current_time
)

# Register MCP tools
agent.register_mcp_directory("./mcp_tools")

# Agent can use both types of tools seamlessly
response = agent.run("What time is it? Also search for today's tech news.")
```

## Creating MCP Tools

### Python-Based Tools

Create a tool file with the `MCP_TOOLS` variable:

```python
# my_tool.py

def search_web(query: str, num_results: int = 5) -> dict:
    """Search the web for information."""
    # Your implementation here
    results = perform_search(query, num_results)
    return {
        "query": query,
        "results": results
    }

MCP_TOOLS = [{
    "name": "web_search",
    "description": "Search the web for information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "number",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    },
    "function": search_web,
    "metadata": {
        "version": "1.0.0",
        "author": "Your Name"
    }
}]
```

### Multiple Tools in One File

```python
# multi_tools.py

def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

MCP_TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "function": add
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "function": multiply
    }
]
```

### Tool Metadata (Optional)

Add a `tool.json` file alongside your `tool.py`:

```json
{
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web for information",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"]
      },
      "metadata": {
        "version": "1.0.0",
        "category": "search",
        "tags": ["web", "search", "information"]
      }
    }
  ]
}
```

## API Reference

### MCPClient

Main client for managing MCP connections.

#### Methods

**`add_url(url, name=None, headers=None)`**
- Connect to remote MCP server
- Returns: `bool` (success status)

**`add_config(config, name=None)`**
- Load MCP server from configuration
- `config`: File path, dict, or `MCPServerConfig`
- Returns: `bool` (success status)

**`add_directory(directory, name=None)`**
- Load tools from directory
- Returns: `bool` (success status)

**`get_tool(name)`**
- Get tool by name
- Returns: `MCPTool` or `None`

**`get_all_tools()`**
- Get all registered tools
- Returns: `List[MCPTool]`

**`get_tool_names()`**
- Get list of tool names
- Returns: `List[str]`

**`call_tool(name, arguments)`**
- Execute a tool
- Returns: Tool result

**`close()`**
- Close all connections

### Agent MCP Methods

**`register_mcp_url(url, name=None, headers=None)`**
- Register tools from remote MCP server
- Returns: `bool`

**`register_mcp_config(config, name=None)`**
- Register tools from configuration
- Returns: `bool`

**`register_mcp_directory(directory, name=None)`**
- Register tools from directory
- Returns: `bool`

**`register_mcp_directories(directories)`**
- Register tools from multiple directories
- Returns: `Dict[str, bool]` (directory -> success)

**`get_mcp_tool_names()`**
- Get list of MCP tool names
- Returns: `List[str]`

### MCPConfig

Configuration management for MCP servers.

**`MCPConfig.from_file(config_path)`**
- Load from JSON file
- Returns: `MCPConfig`

**`MCPConfig.from_dict(config_dict)`**
- Create from dictionary
- Returns: `MCPConfig`

**`add_server(server)`**
- Add server configuration

**`get_server(name)`**
- Get server by name
- Returns: `MCPServerConfig` or `None`

## Examples

### Example 1: Basic MCP Client Usage

```python
from mcp import MCPClient

# Initialize client
client = MCPClient()

# Add different types of MCP sources
client.add_url("https://mcp-tools.example.com")
client.add_config("./mcp_config.json")
client.add_directory("./local_tools")

# Explore available tools
tools = client.get_all_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Parameters: {tool.input_schema}")

# Execute a tool
result = client.call_tool("calculator", {
    "operation": "multiply",
    "a": 25,
    "b": 4
})
print(f"Result: {result}")

# Cleanup
client.close()
```

### Example 2: Agent with Multiple MCP Sources

```python
from fourier import Fourier
from agent import Agent, AgentConfig

# Setup
client = Fourier(api_key="...", provider="together")
agent = Agent(
    client=client,
    name="MultiToolAgent",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    config=AgentConfig(verbose=True, max_iterations=15)
)

# Register MCP tools from multiple sources
agent.register_mcp_config({
    "command": "python",
    "args": ["-m", "mcp_server_filesystem"],
    "env": {"ROOT_PATH": "./data"}
})

agent.register_mcp_directories([
    "./mcp_tools/search",
    "./mcp_tools/analysis",
    "./mcp_tools/visualization"
])

# Execute complex task
response = agent.run("""
    1. Search for recent papers on machine learning
    2. Read the abstracts from ./data/papers/
    3. Analyze common themes
    4. Create a summary visualization
""")

print(response["output"])
```

### Example 3: Creating and Using Custom MCP Directory

```python
# 1. Create directory structure
# mcp_tools/
#   weather/
#     tool.py

# 2. Create tool (mcp_tools/weather/tool.py)
import requests

def get_weather(city: str) -> dict:
    """Get weather for a city."""
    # Mock implementation
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny"
    }

MCP_TOOLS = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    },
    "function": get_weather
}]

# 3. Use in agent
from fourier import Fourier
from agent import Agent

client = Fourier(api_key="...", provider="groq")
agent = Agent(client=client, model="mixtral-8x7b-32768")

agent.register_mcp_directory("./mcp_tools/weather")

response = agent.run("What's the weather like in San Francisco?")
print(response["output"])
```

### Example 4: Loading MCP Config Compatible with Claude Desktop

```python
from mcp import MCPConfig, MCPClient

# Load Claude Desktop compatible config
config = MCPConfig.from_file("~/.config/claude/mcp_config.json")

# Create MCP client and add servers
client = MCPClient()
for server in config.get_all_servers():
    if server.is_remote():
        client.add_url(server.url, server.name, server.headers)
    else:
        client.add_config(server, server.name)

# Now use the tools
print(client.get_tool_names())
```

## Best Practices

1. **Error Handling**: Always handle potential connection failures
2. **Resource Cleanup**: Use context managers or explicitly call `close()`
3. **Configuration Management**: Store MCP configs in version control
4. **Tool Organization**: Group related tools in directories
5. **Naming Conventions**: Use clear, descriptive tool names
6. **Documentation**: Document tool parameters and return types
7. **Testing**: Test tools independently before using with agents
8. **Security**: Never hardcode credentials; use environment variables

## Troubleshooting

### Connection Issues

```python
# Check if connection succeeded
if not client.add_url("https://mcp.example.com"):
    print("Failed to connect - check URL and network")

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Tool Not Found

```python
# List all available tools
print(client.get_tool_names())

# Check if specific tool exists
tool = client.get_tool("my_tool")
if tool is None:
    print("Tool not registered")
```

### Execution Errors

```python
# Wrap tool calls in try-except
try:
    result = client.call_tool("calculator", {"operation": "add", "a": 1, "b": 2})
except Exception as e:
    print(f"Tool execution failed: {e}")
```

## License

FourierSDK and its MCP implementation are released under the MIT License.
