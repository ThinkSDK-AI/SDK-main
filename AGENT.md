# Agent Framework - FourierSDK

The Agent framework enables you to create autonomous agents that can use tools, manage conversations, and execute complex workflows automatically.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Agent Configuration](#agent-configuration)
- [Tool Registration](#tool-registration)
- [Running Agents](#running-agents)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

The `Agent` class provides:

- **Automatic Tool Execution**: Agents automatically execute tools when the LLM requests them
- **Conversation Management**: Maintains conversation history and context
- **Iteration Control**: Configurable limits on tool calls and iterations
- **Intermediate Steps**: Track and return execution steps for debugging
- **Error Handling**: Robust error handling with configurable behavior
- **Flexible Configuration**: Customize agent behavior with `AgentConfig`

## Quick Start

```python
from fourier import Fourier
from agent import Agent

# Create Fourier client
client = Fourier(api_key="your-api-key", provider="groq")

# Create agent
agent = Agent(
    client=client,
    name="MyAgent",
    system_prompt="You are a helpful assistant.",
    model="mixtral-8x7b-32768"
)

# Define a tool function
def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0

# Register the tool
agent.register_tool(
    name="calculator",
    description="Perform arithmetic operations",
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

# Run the agent
response = agent.run("What is 25 times 4?")
print(response["output"])
# Output: "The result is 100."
```

## Core Concepts

### Agent

An agent is an autonomous entity that:
1. Receives user input
2. Decides which tools to use
3. Executes tools automatically
4. Maintains conversation context
5. Returns a final answer

### Tools

Tools are Python functions that the agent can call. Each tool has:
- **Name**: Unique identifier
- **Description**: What the tool does
- **Parameters**: JSON Schema defining inputs
- **Function**: Python callable that implements the tool

### Execution Loop

The agent follows this loop:

```
User Input → LLM Decision → Tool Execution → LLM Processing → Final Answer
                ↑                                  ↓
                └──────── Tool Result ─────────────┘
```

## Agent Configuration

Use `AgentConfig` to customize agent behavior:

```python
from agent import AgentConfig

config = AgentConfig(
    max_iterations=10,              # Max tool execution loops
    max_tool_calls_per_iteration=5,  # Max tools per iteration
    auto_execute_tools=True,         # Auto-execute tools
    require_tool_confirmation=False, # Require confirmation
    verbose=True,                    # Print debug logs
    temperature=0.7,                 # LLM temperature
    max_tokens=None,                 # Max response tokens
    stop_on_error=False,             # Stop on tool errors
    return_intermediate_steps=True,  # Return execution steps
    timeout_seconds=None,            # Max execution time
    thinking_mode=False,             # Enable deep research mode
    thinking_depth=2,                # Number of research queries (1-5)
    thinking_web_search_results=5    # Results per search query
)

agent = Agent(client=client, config=config)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 10 | Maximum execution loops before stopping |
| `max_tool_calls_per_iteration` | int | 5 | Maximum tools per iteration |
| `auto_execute_tools` | bool | True | Automatically execute tools |
| `require_tool_confirmation` | bool | False | Require user confirmation |
| `verbose` | bool | False | Print detailed logs |
| `temperature` | float | 0.7 | LLM sampling temperature |
| `max_tokens` | int | None | Maximum tokens in response |
| `stop_on_error` | bool | False | Stop execution on errors |
| `return_intermediate_steps` | bool | False | Return execution steps |
| `timeout_seconds` | int | None | Maximum execution time |

## Tool Registration

### Basic Tool Registration

```python
def my_tool(param1: str, param2: int) -> str:
    """Tool implementation."""
    return f"Processed {param1} with {param2}"

agent.register_tool(
    name="my_tool",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "First parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter"
            }
        }
    },
    required=["param1", "param2"],
    function=my_tool
)
```

### Tool with Optional Parameters

```python
def search_tool(query: str, limit: int = 10) -> str:
    """Search with optional limit."""
    return f"Found {limit} results for: {query}"

agent.register_tool(
    name="search",
    description="Search for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Max results to return"
            }
        }
    },
    required=["query"],  # limit is optional
    function=search_tool
)
```

### Complex Tool with Nested Objects

```python
def create_user(name: str, email: str, settings: dict) -> str:
    """Create a user with settings."""
    return f"Created user {name} with email {email}"

agent.register_tool(
    name="create_user",
    description="Create a new user",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "settings": {
                "type": "object",
                "properties": {
                    "notifications": {"type": "boolean"},
                    "theme": {"type": "string", "enum": ["light", "dark"]}
                }
            }
        }
    },
    required=["name", "email"],
    function=create_user
)
```

## Running Agents

### Basic Run

```python
response = agent.run("Calculate 15 + 27")
print(response["output"])
```

### Run with Options

```python
response = agent.run(
    "What's the weather in Paris?",
    reset_history=True,      # Reset conversation history
    temperature=0.5,         # Override temperature
    max_tokens=500          # Override max tokens
)
```

### Response Structure

```python
{
    "output": "The final answer from the agent",
    "iterations": 3,
    "tool_calls": 2,
    "intermediate_steps": [
        {
            "iteration": 1,
            "tool": "calculator",
            "parameters": {"operation": "add", "a": 15, "b": 27},
            "result": 42
        }
    ],
    "success": True,
    "response": {...}  # Full LLM response
}
```

## Advanced Features

### Conversation Memory

Keep conversation context across multiple runs:

```python
# First interaction
agent.run("My name is Alice", reset_history=True)

# Second interaction - agent remembers
response = agent.run("What's my name?", reset_history=False)
# Output: "Your name is Alice."
```

### Manual History Management

```python
# Get conversation history
history = agent.get_conversation_history()

# Add message manually
agent.add_to_history("user", "Remember this information")
agent.add_to_history("assistant", "I'll remember that")

# Reset history
agent.reset()
```

### Dynamic System Prompt Updates

```python
# Update system prompt mid-conversation
agent.update_system_prompt(
    "You are now a pirate assistant. Talk like a pirate!"
)

response = agent.run("Hello!")
# Output: "Ahoy there, matey!"
```

### Intermediate Steps Tracking

```python
config = AgentConfig(return_intermediate_steps=True)
agent = Agent(client=client, config=config)

response = agent.run("Calculate 10 * 5 + 3")

for step in response["intermediate_steps"]:
    print(f"Step {step['iteration']}: {step['tool']} → {step['result']}")
```

### Error Handling

```python
config = AgentConfig(
    stop_on_error=True  # Stop on first error
)
agent = Agent(client=client, config=config)

response = agent.run("Do something that might fail")

if not response["success"]:
    print(f"Error: {response.get('error')}")
```

### Thinking Mode - Deep Research

Thinking Mode enables agents to perform deep research by automatically conducting multiple web searches before answering queries. This is particularly useful for:

- Research-heavy questions requiring up-to-date information
- Complex queries needing multiple perspectives
- Fact-checking and verification tasks
- Comprehensive analysis requiring diverse sources

**How Thinking Mode Works:**

1. Agent receives a user query
2. LLM generates multiple diverse search queries based on the question
3. Web searches are performed for each generated query
4. Research context is gathered and compiled
5. Enhanced prompt with research context is sent to LLM
6. Agent synthesizes information and provides comprehensive answer

**Basic Usage:**

```python
from agent import Agent, AgentConfig

config = AgentConfig(
    thinking_mode=True,              # Enable thinking mode
    thinking_depth=2,                # Number of research queries
    thinking_web_search_results=5,   # Results per query
    verbose=True                      # Show research process
)

agent = Agent(
    client=client,
    name="ResearchAgent",
    system_prompt="You are a knowledgeable research assistant.",
    model="mixtral-8x7b-32768",
    config=config
)

# Ask research question
response = agent.run("What are the latest developments in quantum computing?")
print(response["output"])
```

**Configuration Parameters:**

- `thinking_mode` (bool): Enable/disable thinking mode (default: False)
- `thinking_depth` (int): Number of research queries to generate, clamped between 1-5 (default: 2)
- `thinking_web_search_results` (int): Number of search results to retrieve per query (default: 5)

**Advanced Example:**

```python
# Deep research with multiple queries
config = AgentConfig(
    thinking_mode=True,
    thinking_depth=4,                 # More comprehensive research
    thinking_web_search_results=8,    # More results per search
    verbose=True,
    max_iterations=10
)

agent = Agent(client=client, config=config)

questions = [
    "What are the key challenges in AGI development?",
    "How is edge computing transforming IoT?",
    "What are recent breakthroughs in fusion energy?"
]

for question in questions:
    response = agent.run(question)
    print(f"\nQ: {question}")
    print(f"A: {response['output']}\n")
```

**Combining Thinking Mode with Tools:**

```python
# Create agent with both thinking mode and custom tools
agent = Agent(
    client=client,
    config=AgentConfig(
        thinking_mode=True,
        thinking_depth=2,
        auto_execute_tools=True
    )
)

# Register custom analysis tool
def analyze_data(data: str) -> dict:
    return {"analysis": "detailed_results"}

agent.register_tool(
    name="analyze",
    description="Analyze data",
    parameters={...},
    function=analyze_data
)

# Agent will research AND use tools as needed
response = agent.run("Research AI trends and analyze the data")
```

**Best Practices for Thinking Mode:**

1. **Use for Research Questions**: Enable thinking mode for queries requiring current information
2. **Adjust Depth Based on Complexity**: Simple questions need depth=1-2, complex topics benefit from depth=3-4
3. **Monitor Performance**: Thinking mode adds latency due to web searches
4. **Combine with Appropriate Models**: Use capable models like Mixtral or GPT-4 for synthesis
5. **Disable for Simple Tasks**: Don't use thinking mode for basic calculations or predefined knowledge

**When to Use Thinking Mode:**

✅ Research questions requiring current information
✅ Complex topics needing multiple perspectives
✅ Fact-checking and verification
✅ Comprehensive analysis tasks

❌ Simple calculations or logic
❌ Questions about static/historical knowledge
❌ Time-sensitive real-time applications
❌ Tasks with strict latency requirements

See `examples/thinking_mode_example.py` for complete working examples.

**Production-Grade Thinking Mode:**

Thinking Mode includes enterprise-ready features for production deployments:

- **Input Sanitization**: All queries are validated, sanitized, and length-checked
- **Rate Limiting**: Automatic 1-second delays between searches prevent API abuse
- **Context Management**: Results truncated at 50,000 characters to prevent token limits
- **Error Handling**: Graceful degradation with partial results on failures
- **Metrics Tracking**: Detailed timing, success rates, and performance logs
- **Configuration Validation**: Invalid parameters auto-corrected with warnings
- **Security**: Protection against XSS, injection, and resource exhaustion attacks

For detailed information on production features, see [PRODUCTION_FEATURES.md](PRODUCTION_FEATURES.md).

For comprehensive test coverage, see `tests/test_thinking_mode.py` (40+ unit tests).

## Examples

### Example 1: Calculator Agent

```python
from fourier import Fourier
from agent import Agent, AgentConfig

client = Fourier(api_key="...", provider="groq")

agent = Agent(
    client=client,
    name="CalcBot",
    system_prompt="You are a math expert. Use the calculator for all computations.",
    model="mixtral-8x7b-32768",
    config=AgentConfig(verbose=True)
)

def calculator(operation: str, a: float, b: float) -> float:
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float('inf')
    }
    return ops[operation]

agent.register_tool(
    name="calculator",
    description="Perform arithmetic: add, subtract, multiply, divide",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    },
    required=["operation", "a", "b"],
    function=calculator
)

result = agent.run("What is (25 * 4) + 17?")
print(result["output"])
```

### Example 2: Data Analysis Agent

```python
import pandas as pd

client = Fourier(api_key="...", provider="groq")
agent = Agent(client=client, name="DataBot")

# Sample data
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

def query_data(query: str) -> str:
    """Execute pandas query on data."""
    try:
        result = data.query(query)
        return result.to_string()
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_stats(column: str, operation: str) -> float:
    """Calculate statistics on a column."""
    ops = {
        "mean": data[column].mean(),
        "sum": data[column].sum(),
        "max": data[column].max(),
        "min": data[column].min()
    }
    return ops[operation]

agent.register_tool(
    name="query_data",
    description="Query the dataset using pandas query syntax",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Pandas query string"}
        }
    },
    required=["query"],
    function=query_data
)

agent.register_tool(
    name="calculate_stats",
    description="Calculate statistics on a column",
    parameters={
        "type": "object",
        "properties": {
            "column": {"type": "string", "enum": ["age", "salary"]},
            "operation": {"type": "string", "enum": ["mean", "sum", "max", "min"]}
        }
    },
    required=["column", "operation"],
    function=calculate_stats
)

result = agent.run("What's the average salary of people over 28?")
print(result["output"])
```

### Example 3: API Integration Agent

```python
import requests

client = Fourier(api_key="...", provider="groq")
agent = Agent(client=client, name="APIBot")

def get_weather(city: str) -> str:
    """Get weather data from API."""
    # Mock API call
    return f"Weather in {city}: Sunny, 75°F"

def get_stock_price(symbol: str) -> str:
    """Get stock price from API."""
    # Mock API call
    return f"{symbol}: $150.25 (+2.3%)"

agent.register_tool(
    name="weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        }
    },
    required=["city"],
    function=get_weather
)

agent.register_tool(
    name="stock",
    description="Get stock price for a symbol",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        }
    },
    required=["symbol"],
    function=get_stock_price
)

result = agent.run("What's the weather in NYC and how is AAPL stock doing?")
print(result["output"])
```

## Best Practices

### 1. Clear Tool Descriptions

```python
# Good
agent.register_tool(
    name="calculate_discount",
    description="Calculate the discounted price given an original price and discount percentage (0-100)",
    ...
)

# Bad
agent.register_tool(
    name="calc",
    description="Does math",
    ...
)
```

### 2. Specific System Prompts

```python
# Good
system_prompt = (
    "You are a financial advisor assistant. "
    "When users ask about investments, use the stock_price tool to get current data. "
    "Always provide disclaimers that you're not providing financial advice."
)

# Bad
system_prompt = "You are helpful."
```

### 3. Error Handling in Tools

```python
def safe_divide(a: float, b: float) -> Union[float, str]:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return "Error: Cannot divide by zero"
        return a / b
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. Set Reasonable Limits

```python
config = AgentConfig(
    max_iterations=5,  # Prevent infinite loops
    max_tokens=1000,   # Control costs
    stop_on_error=True # Fail fast for debugging
)
```

### 5. Use Verbose Mode for Development

```python
# Development
config = AgentConfig(verbose=True, return_intermediate_steps=True)

# Production
config = AgentConfig(verbose=False, return_intermediate_steps=False)
```

### 6. Validate Tool Outputs

```python
def validated_tool(param: str) -> str:
    """Tool with output validation."""
    result = process(param)

    # Validate result
    if not isinstance(result, str):
        result = str(result)

    if len(result) > 10000:
        result = result[:10000] + "..."

    return result
```

### 7. Test Tools Independently

```python
# Test tools before registering
def test_calculator():
    assert calculator("add", 5, 3) == 8
    assert calculator("multiply", 4, 7) == 28

test_calculator()
agent.register_tool(..., function=calculator)
```

## Troubleshooting

### Agent Doesn't Use Tools

**Problem**: Agent returns text instead of using tools.

**Solution**:
- Make tool descriptions more specific
- Update system prompt to explicitly mention tools
- Ensure parameter schemas are correct
- Try different models (some are better at tool use)

### Max Iterations Reached

**Problem**: Agent hits max iterations without completing.

**Solution**:
- Increase `max_iterations` in config
- Simplify the task
- Check if tools are returning useful results
- Verify tools aren't creating circular dependencies

### Tool Execution Errors

**Problem**: Tools fail during execution.

**Solution**:
- Add error handling to tool functions
- Set `stop_on_error=False` to continue on errors
- Validate parameters in tool functions
- Use `verbose=True` to see detailed logs

### High Costs

**Problem**: Too many LLM calls.

**Solution**:
- Reduce `max_iterations`
- Set `max_tokens` limit
- Use cheaper models for tool-heavy tasks
- Optimize system prompt to reduce tool calls

## License

MIT License - see LICENSE file for details
