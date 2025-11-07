# FourierSDK

A Python SDK for accessing Large Language Models (LLMs) from various inference providers like Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius. FourierSDK provides a unified interface similar to the OpenAI SDK while adding support for function calling, internet search, autonomous agents, and multiple providers with standardized response formats.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
  - [API Keys](#api-keys)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Function Calling](#function-calling)
  - [Internet Search](#internet-search)
  - [Autonomous Agents](#autonomous-agents)
  - [Provider-Specific Examples](#provider-specific-examples)
- [Features](#features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install via pip (once published):

```bash
pip install fourier-sdk
```

## Setup

### API Keys

FourierSDK supports multiple LLM providers, each requiring its own API key:

- **Groq**: Get an API key from [Groq](https://console.groq.com/)
- **Together AI**: Get an API key from [Together AI](https://www.together.ai/)
- **OpenAI**: Get an API key from [OpenAI](https://platform.openai.com/)
- **Anthropic**: Get an API key from [Anthropic](https://console.anthropic.com/)
- **Perplexity**: Get an API key from [Perplexity](https://www.perplexity.ai/)
- **Nebius**: Get an API key from [Nebius](https://nebius.ai/)

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

See [AGENT.md](AGENT.md) for complete documentation and advanced examples.

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

## Features

- **Multi-Provider Support**: Easily switch between Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius
- **Standardized Response Format**: Consistent response structure regardless of the provider
- **Function Calling**: Define and use functions/tools with JSON schema validation
- **Internet Search**: Augment LLM responses with up-to-date information from the web
- **Autonomous Agents**: Create agents that automatically use tools and manage conversations
- **Conversation Management**: Built-in conversation history and context management
- **Customizable Base URLs**: For enterprise deployments or custom endpoints
- **Token Usage Tracking**: Monitor token consumption across providers
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive exception hierarchy for precise error handling
- **Logging**: Production-ready logging framework
- **Configurable Behavior**: Fine-tune agent iterations, timeouts, and error handling

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
- `api_key`: API key for the LLM provider
- `provider`: Provider name (default: "groq"). Supported: groq, together, nebius, openai, anthropic, perplexity
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
