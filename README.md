# ThinkSDK

A Python SDK for accessing Large Language Models (LLMs) from various inference providers like Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius. ThinkSDK follows a similar pattern to the OpenAI SDK while adding support for tools, internet search, and multiple providers with a standardized response format.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
  - [API Keys](#api-keys)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Using Tools](#using-tools)
  - [Internet Search](#internet-search)
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

## Setup

### API Keys

ThinkSDK supports multiple LLM providers, each requiring its own API key:

- **Groq**: Get an API key from [Groq](https://console.groq.com/)
- **Together AI**: Get an API key from [Together AI](https://www.together.ai/)
- **OpenAI**: Get an API key from [OpenAI](https://platform.openai.com/)
- **Anthropic**: Get an API key from [Anthropic](https://console.anthropic.com/)
- **Perplexity**: Get an API key from [Perplexity](https://www.perplexity.ai/)
- **Nebius**: Get an API key from [Nebius](https://nebius.ai/)
- **OpenRouter**: Get an API key from [OpenRouter](https://openrouter.ai/)

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
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Important**: Never commit your `.env` file to version control. It's already added to `.gitignore`.

## Usage

### Basic Usage

```python
from think import Think
import os

# Initialize the SDK with environment variable
api_key = os.getenv("GROQ_API_KEY")
client = Think(api_key=api_key, provider="groq")

# Create a chat completion
response = client.chat(
    model="mixtral-8x7b-32768",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response["choices"][0]["message"]["content"])

# Access token usage information
usage = response.get("usage", {})
print(f"Token Usage: {usage.get('input_tokens', 0)} input / {usage.get('output_tokens', 0)} output / {usage.get('total_tokens', 0)} total")
```

### Using Tools

```python
from think import Think
import os

# Initialize the SDK
api_key = os.getenv("TOGETHER_API_KEY")
client = Think(api_key=api_key, provider="together")

# Create a tool
calculator = client.create_tool(
    name="calculator",
    description="A simple calculator that can perform basic arithmetic operations",
    parameters={
        "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"]
        },
        "a": {"type": "number"},
        "b": {"type": "number"}
    },
    required=["operation", "a", "b"]
)

# Use the tool in a chat completion
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "What is 5 + 3?"}
    ],
    tools=[calculator]
)

print(response["choices"][0]["message"]["content"])
```

### Internet Search

ThinkSDK supports internet search capabilities, allowing the LLM to access up-to-date information from the web:

```python
from think import Think
import os

# Initialize the SDK
api_key = os.getenv("TOGETHER_API_KEY")
client = Think(api_key=api_key, provider="together")

# Use internet search in a chat completion
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "What are the latest developments in AI in 2025?"}
    ],
    internet_search=True,  # Enable internet search
    search_results_count=3  # Number of search results to use
)

print(response["choices"][0]["message"]["content"])
```

You can also specify a custom search query:

```python
response = client.chat(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "Tell me about the weather in New York"}
    ],
    internet_search=True,
    search_query="current weather forecast New York City"  # Custom search query
)
```

### Provider-Specific Examples

#### OpenAI

```python
from think import Think
import os

api_key = os.getenv("OPENAI_API_KEY")
client = Think(api_key=api_key, provider="openai")

response = client.chat(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)

print(response["choices"][0]["message"]["content"])
```

#### Anthropic

```python
from think import Think
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Think(api_key=api_key, provider="anthropic")

response = client.chat(
    model="claude-3-opus-20240229",
    messages=[
        {"role": "user", "content": "Write a short story about a robot learning to paint"}
    ]
)

print(response["choices"][0]["message"]["content"])
```

#### Perplexity

```python
from think import Think
import os

api_key = os.getenv("PERPLEXITY_API_KEY")
client = Think(api_key=api_key, provider="perplexity")

response = client.chat(
    model="sonar-medium-online",
    messages=[
        {"role": "user", "content": "What are the latest breakthroughs in fusion energy?"}
    ]
)

print(response["choices"][0]["message"]["content"])
```

## Features

- **Multi-Provider Support**: Easily switch between Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius
- **Standardized Response Format**: Consistent response structure regardless of the provider
- **Tool Support**: Define and use tools with schema validation
- **Internet Search**: Augment LLM responses with up-to-date information from the web
- **Customizable Base URLs**: For enterprise deployments or custom endpoints
- **Token Usage Tracking**: Monitor token consumption across providers
- **Type Hints**: Full type annotations for better IDE support

## API Reference

### Think Class

The main class for interacting with LLM providers.

#### Methods

- `__init__(api_key: str, provider: str = "groq", base_url: Optional[str] = None, **provider_kwargs)`
- `chat(model: str, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: Optional[int] = None, tools: Optional[List[Tool]] = None, internet_search: bool = False, search_query: Optional[str] = None, search_results_count: int = 3, **kwargs) -> Dict[str, Any]`
- `create_tool(name: str, description: str, parameters: Dict[str, Any], required: List[str] = None) -> Tool`

### Tool Class

A class for defining tools that can be used by the LLM.

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
- **Internet Search Failures**: Check your internet connection and ensure you're using supported providers

### Debugging

Set up logging to debug issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT