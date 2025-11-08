# FourierSDK

A Python SDK for accessing Large Language Models (LLMs) from various inference providers like Groq, Together AI, OpenAI, Anthropic, Perplexity, Nebius, and AWS Bedrock. FourierSDK provides a unified interface similar to the OpenAI SDK while adding support for function calling, internet search, autonomous agents, and multiple providers with standardized response formats.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
  - [API Keys](#api-keys)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Function Calling](#function-calling)
  - [Internet Search](#internet-search)
  - [Autonomous Agents](#autonomous-agents)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  - [Provider-Specific Examples](#provider-specific-examples)
- [Features](#features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### Quick Start (All Features)

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main
pip install -e .[all]
python setup_providers.py  # Interactive configuration
```

### Modular Installation (Recommended)

Install only the providers you need:

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main

# Base installation (Groq, OpenAI, Anthropic, Together, Perplexity, Nebius)
pip install -e .

# Add AWS Bedrock support
pip install -e .[bedrock]

# Add web search feature
pip install -e .[search]

# Run interactive setup
python setup_providers.py
```

### Installation Options

| Command | Providers Included |
|---------|-------------------|
| `pip install -e .` | Base providers (no extra dependencies) |
| `pip install -e .[bedrock]` | Base + AWS Bedrock |
| `pip install -e .[search]` | Base + Web Search |
| `pip install -e .[all]` | All features |

**See [INSTALLATION.md](INSTALLATION.md) for detailed installation guide.**

## Setup

### API Keys

FourierSDK supports multiple LLM providers, each requiring its own API key:

- **Groq**: Get an API key from [Groq](https://console.groq.com/)
- **Together AI**: Get an API key from [Together AI](https://www.together.ai/)
- **OpenAI**: Get an API key from [OpenAI](https://platform.openai.com/)
- **Anthropic**: Get an API key from [Anthropic](https://console.anthropic.com/)
- **Perplexity**: Get an API key from [Perplexity](https://www.perplexity.ai/)
- **Nebius**: Get an API key from [Nebius](https://nebius.ai/)
- **AWS Bedrock**: Configure AWS credentials (see [Bedrock Documentation](BEDROCK.md))

### Environment Variables

Create a `.env` file in the root directory of the project with your API keys:

```
# API Keys for different providers
GROQ_API_KEY=your_groq_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
NEBIUS_API_KEY=your_nebius_api_key_here

# AWS Bedrock Configuration (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
```

**Important**: Never commit your `.env` file to version control. It's already added to `.gitignore`.

## Usage

### Quick Start

```python
from fourier import Fourier
import os

# Initialize the SDK client
api_key = os.getenv("GROQ_API_KEY")
client = Fourier(api_key=api_key, provider="groq")

# Create a chat completion
response = client.chat(
    model="mixtral-8x7b-32768",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Access the response
print(response["response"]["output"])

# Access token usage information
usage = response.get("usage", {})
print(f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out")
```

### Command Line Interface (CLI)

FourierSDK includes a comprehensive command-line interface for managing agents, MCP tools, and running queries without writing code.

**Quick Start:**

```bash
# Quick chat
python cli.py chat "What is quantum computing?"

# Interactive mode
python cli.py interactive

# Create an agent
python cli.py create-agent --name ResearchBot --thinking-mode --save

# Add MCP tools
python cli.py add-mcp --directory ./mcp_tools --agent ResearchBot

# Run saved agent
python cli.py run --agent ResearchBot --query "Latest AI developments"
```

**Interactive Shell:**

```bash
$ python cli.py interactive
╔═══════════════════════════════════════════════════════════╗
║              FourierSDK Interactive Shell                 ║
║                                                           ║
║  Type 'help' for commands, 'exit' to quit               ║
╚═══════════════════════════════════════════════════════════╝

fourier> create-agent
Agent name: MyAgent
Provider (groq/openai/anthropic/together) [groq]: groq
Enable thinking mode? (y/N): y
✓ Agent 'MyAgent' created

fourier> chat Hello, how can you help?
Processing...

Hello! I'm your AI assistant. I can help with...

fourier> add-mcp directory ./mcp_tools
✓ MCP directory added

fourier> list-agents
Saved Agents:
  MyAgent
  ResearchBot

fourier> exit
```

**Key CLI Commands:**

- `interactive`: Start interactive shell
- `chat`: Quick one-off queries
- `create-agent`: Create and save agents
- `add-mcp`: Add MCP tools (URL, config, directory)
- `list-agents`: List all saved agents
- `list-mcp`: List MCP tools
- `run`: Run saved agent with query
- `delete-agent`: Delete saved agent
- `config`: Manage configuration

**Example Workflow:**

```bash
# 1. Create research agent
python cli.py create-agent \
  --name DeepResearch \
  --thinking-mode \
  --thinking-depth 3 \
  --save

# 2. Add MCP tools
python cli.py add-mcp --directory ./research_tools --agent DeepResearch
python cli.py add-mcp --url https://mcp.example.com/api --agent DeepResearch

# 3. Run research queries
python cli.py run \
  --agent DeepResearch \
  --query "Analyze quantum computing trends" \
  --verbose

# 4. List everything
python cli.py list-agents --details
python cli.py list-mcp --agent DeepResearch
```

**Benefits:**

- ✅ **No coding required**: Use FourierSDK from command line
- ✅ **Agent persistence**: Save and reuse agent configurations
- ✅ **MCP integration**: Easily manage MCP tools
- ✅ **Interactive mode**: REPL-style interface
- ✅ **Scriptable**: Use in bash scripts and automation
- ✅ **Configuration management**: JSON-based config storage

See [CLI.md](CLI.md) for complete CLI documentation.

### Function Calling

FourierSDK supports function/tool calling across multiple providers:

```python
from fourier import Fourier
import os

# Initialize the SDK client
api_key = os.getenv("TOGETHER_API_KEY")
client = Fourier(api_key=api_key, provider="together")

# Define a function/tool
calculator = client.create_tool(
    name="calculator",
    description="A simple calculator that can perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        }
    },
    required=["operation", "a", "b"]
)

# Use the tool in a chat completion
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "What is 25 multiplied by 4?"}
    ],
    tools=[calculator]
)

print(response)
```

### Internet Search

FourierSDK supports internet search capabilities, allowing the LLM to access up-to-date information from the web:

```python
from fourier import Fourier
import os

# Initialize the SDK client
api_key = os.getenv("TOGETHER_API_KEY")
client = Fourier(api_key=api_key, provider="together")

# Use internet search in a chat completion
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "What are the latest developments in AI in 2025?"}
    ],
    internet_search=True,  # Enable internet search
    search_results=3  # Number of search results to use
)

print(response["response"]["output"])

# Access search metadata
if "search_metadata" in response.get("response", {}):
    print(f"Search query: {response['response']['search_metadata']['query']}")
    print(f"Sources: {response['response'].get('citations', [])}")
```

You can also specify a custom search query:

```python
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "Tell me about the weather"}
    ],
    internet_search=True,
    search_query="current weather forecast New York City"  # Custom search query
)
```

### Autonomous Agents

FourierSDK includes a powerful Agent framework for creating autonomous agents that can use tools, manage conversations, and execute complex workflows automatically.

```python
from fourier import Fourier
from agent import Agent, AgentConfig
import os

# Create Fourier client
client = Fourier(api_key=os.getenv("GROQ_API_KEY"), provider="groq")

# Create an agent
agent = Agent(
    client=client,
    name="MathAssistant",
    system_prompt="You are a helpful math assistant. Use tools when needed.",
    model="mixtral-8x7b-32768",
    config=AgentConfig(verbose=True, return_intermediate_steps=True)
)

# Define and register a tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0

agent.register_tool(
    name="calculator",
    description="Perform arithmetic operations: add, subtract, multiply, divide",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "multiply"]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    },
    required=["operation", "a", "b"],
    function=calculator
)

# Run the agent - it will automatically use tools as needed
response = agent.run("What is 25 times 4, then add 10?")

print(f"Answer: {response['output']}")
print(f"Tool calls made: {response['tool_calls']}")
print(f"Iterations: {response['iterations']}")

# View intermediate steps
for step in response['intermediate_steps']:
    print(f"Step: {step['tool']}({step['parameters']}) = {step['result']}")
```

**Key Agent Features:**
- **Automatic Tool Execution**: Agents automatically execute tools when the LLM requests them
- **Conversation Memory**: Maintains context across multiple interactions
- **Configurable Behavior**: Control iterations, error handling, verbosity, and more
- **Intermediate Steps**: Track what tools were used and when
- **Error Resilience**: Continue execution even when tools fail
- **Thinking Mode**: Deep research capability with automatic web searches

**Thinking Mode - Deep Research:**

Agents can enable Thinking Mode to perform deep research before answering queries:

```python
from agent import Agent, AgentConfig

# Create agent with thinking mode enabled
agent = Agent(
    client=client,
    name="ResearchAgent",
    model="mixtral-8x7b-32768",
    config=AgentConfig(
        thinking_mode=True,              # Enable deep research
        thinking_depth=2,                # Number of research queries
        thinking_web_search_results=5,   # Results per query
        verbose=True
    )
)

# Ask a research question - agent will automatically search the web
response = agent.run("What are the latest developments in quantum computing?")
print(response["output"])
```

Thinking Mode automatically:
1. Generates diverse search queries based on the question
2. Performs web searches for each query
3. Gathers and compiles research context
4. Synthesizes information from multiple sources
5. Provides comprehensive, well-researched answers

See [AGENT.md](AGENT.md) for complete documentation and advanced examples.

**Production-Grade Features:**

Thinking Mode includes enterprise-ready features:
- ✅ **Input sanitization**: Automatic query validation and cleaning
- ✅ **Rate limiting**: Built-in delays prevent API abuse
- ✅ **Context management**: Automatic truncation prevents token limits
- ✅ **Error resilience**: Graceful degradation with partial results
- ✅ **Performance monitoring**: Detailed timing and success metrics
- ✅ **Configuration validation**: Auto-correction of invalid parameters

See [PRODUCTION_FEATURES.md](PRODUCTION_FEATURES.md) for comprehensive production features documentation.

### Model Context Protocol (MCP)

FourierSDK includes comprehensive support for the Model Context Protocol (MCP), allowing you to connect your agents and applications to external tools and data sources through a standardized protocol.

**What is MCP?**

The Model Context Protocol is an open protocol that standardizes how AI applications connect with external tools and data sources. It enables seamless integration with:

- Remote MCP servers via HTTP/HTTPS
- Local MCP servers via subprocess (stdio)
- Directory-based tool loading
- Configuration-based tool management

**Quick MCP Example:**

```python
from fourier import Fourier
from agent import Agent

# Create agent
client = Fourier(api_key="...", provider="groq")
agent = Agent(client=client, model="mixtral-8x7b-32768")

# Register MCP tools from different sources
agent.register_mcp_url("https://mcp.example.com/api")  # Remote server
agent.register_mcp_config("./mcp_config.json")         # Config file
agent.register_mcp_directory("./mcp_tools")            # Local directory

# Agent now has access to all MCP tools
response = agent.run("Use the available tools to complete this task")
```

**Three Ways to Connect:**

1. **Remote MCP Server (URL)**:
   ```python
   agent.register_mcp_url("https://mcp.example.com/api")
   ```

2. **Configuration File** (Compatible with Claude Desktop format):
   ```python
   agent.register_mcp_config("./mcp_config.json")
   ```

3. **Local Tool Directories**:
   ```python
   agent.register_mcp_directory("./mcp_tools")
   # Or multiple directories
   agent.register_mcp_directories(["./tools1", "./tools2", "./tools3"])
   ```

**Creating MCP Tools:**

Create a tool file in your MCP directory:

```python
# mcp_tools/calculator/tool.py

def calculate(operation: str, a: float, b: float) -> float:
    """Perform arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0

MCP_TOOLS = [{
    "name": "calculator",
    "description": "Perform arithmetic operations",
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

See [MCP.md](MCP.md) for complete MCP documentation, configuration formats, and advanced examples.

### Provider-Specific Examples

#### OpenAI

```python
from fourier import Fourier
import os

api_key = os.getenv("OPENAI_API_KEY")
client = Fourier(api_key=api_key, provider="openai")

response = client.chat(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)

print(response["response"]["output"])
```

#### Anthropic

```python
from fourier import Fourier
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Fourier(api_key=api_key, provider="anthropic")

response = client.chat(
    model="claude-3-opus-20240229",
    messages=[
        {"role": "user", "content": "Write a short story about a robot learning to paint"}
    ]
)

print(response["response"]["output"])
```

#### Perplexity

```python
from fourier import Fourier
import os

api_key = os.getenv("PERPLEXITY_API_KEY")
client = Fourier(api_key=api_key, provider="perplexity")

response = client.chat(
    model="sonar-medium-online",
    messages=[
        {"role": "user", "content": "What are the latest breakthroughs in fusion energy?"}
    ]
)

print(response["response"]["output"])
```

#### AWS Bedrock

```python
from fourier import Fourier
import os

# Option 1: Using IAM credentials
client = Fourier(
    api_key=None,  # Not needed for IAM
    provider="bedrock",
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region="us-east-1"
)

# Option 2: Using AWS profile
client = Fourier(
    api_key=None,
    provider="bedrock",
    profile_name="default",
    region="us-east-1"
)

# Option 3: Default credential chain
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)

# Basic usage
response = client.chat(
    model="claude-3-5-sonnet",  # or "anthropic.claude-3-5-sonnet-20241022-v2:0"
    messages=[
        {"role": "user", "content": "Explain serverless computing"}
    ],
    max_tokens=500
)

print(response["response"]["content"])

# Advanced: Cross-region inference
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_cross_region=True  # High availability
)

# Advanced: Global inference profiles
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_global_inference=True
)
```

**See [BEDROCK.md](BEDROCK.md) for complete Bedrock documentation, including:**
- All authentication methods
- 40+ supported models (Claude, Llama, Mistral, Titan, Cohere, AI21)
- Cross-region and global inference
- Tool calling with Bedrock
- Agent framework integration
- Complete examples

## Features

- **Multi-Provider Support**: Easily switch between Groq, Together AI, OpenAI, Anthropic, Perplexity, Nebius, and AWS Bedrock
- **Standardized Response Format**: Consistent response structure regardless of the provider
- **Command Line Interface (CLI)**: Comprehensive CLI for managing agents and MCP tools without writing code
- **Function Calling**: Define and use functions/tools with JSON schema validation
- **Internet Search**: Augment LLM responses with up-to-date information from the web
- **Autonomous Agents**: Create agents that automatically use tools and manage conversations
- **Thinking Mode**: Deep research capability with automatic multi-query web searches and synthesis
- **Model Context Protocol (MCP)**: Connect to remote MCP servers, load tools from directories, and use Claude Desktop-compatible configurations
- **Interactive Shell**: REPL-style interface for agent interaction and management
- **Agent Persistence**: Save and load agent configurations with JSON storage
- **Conversation Management**: Built-in conversation history and context management
- **Customizable Base URLs**: For enterprise deployments or custom endpoints
- **Token Usage Tracking**: Monitor token consumption across providers
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive exception hierarchy for precise error handling
- **Logging**: Production-ready logging framework
- **Configurable Behavior**: Fine-tune agent iterations, timeouts, and error handling
- **Scriptable**: Use CLI in automation and CI/CD pipelines

## API Reference

### Fourier Class

The main SDK client for interacting with LLM providers.

#### Constructor

```python
Fourier(
    api_key: str,
    provider: str = "groq",
    base_url: Optional[str] = None,
    **provider_kwargs
)
```

**Parameters:**
- `api_key`: API key for the LLM provider (not required for Bedrock with IAM)
- `provider`: Provider name (default: "groq"). Supported: groq, together, nebius, openai, anthropic, perplexity, bedrock
- `base_url`: Custom base URL for the API (optional)
- `**provider_kwargs`: Additional provider-specific arguments

#### Methods

##### chat()

Create a chat completion request.

```python
client.chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Tool]] = None,
    internet_search: bool = False,
    search_query: Optional[str] = None,
    search_results: int = 3,
    **kwargs
) -> Dict[str, Any]
```

**Returns:** Standardized response dictionary with structure:
```python
{
    "status": "success",
    "timestamp": "2024-01-01T12:00:00Z",
    "metadata": {
        "request_id": "...",
        "model": "...",
        "provider": "...",
        "response_type": "text" | "tool_call",
        "latency_ms": 123
    },
    "response": {
        "type": "text",
        "output": "...",
        "citations": [...] # if internet_search=True
    },
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30
    },
    "error": None
}
```

##### create_tool()

Create a new function/tool definition.

```python
client.create_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
) -> Tool
```

### Tool Class

A class for defining functions/tools that can be used by the LLM.

#### Attributes

- `name: str`: The name of the tool
- `description: str`: A description of what the tool does
- `parameters: Dict[str, Any]`: The parameters the tool accepts in JSON Schema format
- `required: List[str]`: A list of required parameter names

## Troubleshooting

### Common Issues

- **API Key Issues**: Ensure your API keys are correctly set in the `.env` file and loaded properly
- **Model Availability**: Different providers support different models; check provider documentation
- **Rate Limiting**: If you encounter rate limit errors, reduce request frequency or upgrade your API plan
- **Internet Search Failures**: Check your internet connection and ensure DuckDuckGo is accessible

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Custom Exceptions

FourierSDK provides custom exceptions for precise error handling:

- `FourierSDKError`: Base exception class
- `InvalidAPIKeyError`: Invalid or missing API key
- `UnsupportedProviderError`: Unsupported provider specified
- `ProviderAPIError`: Provider API returned an error
- `ToolExecutionError`: Tool/function execution failed
- `WebSearchError`: Web search operation failed

## License

MIT License - see LICENSE file for details
