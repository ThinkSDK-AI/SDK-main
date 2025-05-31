# ThinkSDK

A Python SDK for accessing LLMs from various inference providers like Groq and Together AI. The SDK follows a similar pattern to the OpenAI SDK while adding support for tools and custom providers.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from think import Think

# Initialize the SDK
client = Think(api_key="your-api-key", provider="groq")

# Create a chat completion
response = client.chat(
    model="mixtral-8x7b-32768",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response["choices"][0]["message"]["content"])
```

### Using Tools

```python
from think import Think

# Initialize the SDK
client = Think(api_key="your-api-key")

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
    model="mixtral-8x7b-32768",
    messages=[
        {"role": "user", "content": "What is 5 + 3?"}
    ],
    tools=[calculator]
)

print(response["choices"][0]["message"]["content"])
```

## Features

- Support for multiple LLM providers (Groq, Together AI, etc.)
- Similar interface to OpenAI's SDK
- Built-in tool support with schema validation
- Customizable base URLs for different providers
- Type hints and documentation

## API Reference

### Think Class

The main class for interacting with LLM providers.

#### Methods

- `__init__(api_key: str, provider: str = "groq", base_url: Optional[str] = None)`
- `chat(**kwargs) -> Dict[str, Any]`
- `create_tool(name: str, description: str, parameters: Dict[str, Any], required: List[str] = None) -> Tool`

### Tool Class

A class for defining tools that can be used by the LLM.

#### Attributes

- `name: str`
- `description: str`
- `parameters: Dict[str, Any]`
- `required: List[str]`

## License

MIT 