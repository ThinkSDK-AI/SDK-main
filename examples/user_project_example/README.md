# Fourier Config System - User Project Example

This example demonstrates how to structure your project to use Fourier's centralized configuration and auto-discovery system.

## Project Structure

```
user_project_example/
├── main.py                    # Application entry point
├── api_example.py             # API integration example
├── agents/                    # Agent definitions
│   ├── customer_support.py    # Customer support agent
│   └── research_agent.py      # Research agent
├── workflows.py               # Workflow definitions (single file)
├── tools/                     # Tool definitions
│   └── data_tools.py          # Data manipulation tools
└── .fourier_config.json       # Auto-generated config (gitignore this)
```

## Alternative Structures

You can organize resources in different ways:

### Option 1: Directory-based (current example)
```
agents/
├── agent1.py
├── agent2.py
└── agent3.py

tools/
├── tool1.py
└── tool2.py

workflows/
├── workflow1.py
└── workflow2.py
```

### Option 2: File-based
```
agents.py      # All agents in one file
tools.py       # All tools in one file
workflows.py   # All workflows in one file
```

### Option 3: Mixed
```
agents/        # Complex agents in directory
├── support.py
└── research.py

tools.py       # Simple tools in file
workflows.py   # Workflows in file
```

## Quick Start

### 1. Initialize Config (Once)

In your main application file:

```python
from fourier.config import FourierConfig
from dotenv import load_dotenv

load_dotenv()

# Initialize config - discovers all resources automatically
config = FourierConfig(base_dir=".", auto_discover=True)

# That's it! Your agents, workflows, and tools are now registered
```

### 2. Use Anywhere in Your Codebase

```python
from fourier.config import get_config

# Get the global config instance
config = get_config()

# Invoke agent by name - no imports needed!
response = config.invoke_agent("customer_support", query="Hello")

# Use tools
user_data = config.invoke_tool("fetch_user_data", user_id="123")

# Run workflows
result = config.invoke_workflow("customer_onboarding", input_data={...})
```

## Creating Agents

### Method 1: Using `__agents__` export (Recommended)

```python
# agents/my_agent.py
from fourier import Fourier
from agent import Agent, AgentConfig

client = Fourier(api_key="...", provider="groq")
config = AgentConfig(name="my_agent")

my_agent = Agent(client=client, model="llama-3.1-8b", config=config)

# Export for auto-discovery
__agents__ = {
    "my_agent": my_agent,
    "support": my_agent  # Can have multiple names
}
```

### Method 2: Auto-detection

```python
# agents/my_agent.py
from agent import Agent, AgentConfig

# Just create an Agent instance at module level
my_agent = Agent(...)

# The config system will auto-detect it
```

## Creating Tools

### Method 1: Using `@tool` decorator (Recommended)

```python
# tools/my_tools.py
from fourier.config import tool

@tool
def my_function(arg1: str, arg2: int) -> str:
    """Tool description"""
    return f"Result: {arg1} - {arg2}"
```

### Method 2: Using `tool_` prefix

```python
def tool_my_function(arg1: str) -> str:
    """Tool description"""
    return arg1.upper()
```

### Method 3: Using `__tools__` export

```python
def custom_func(x):
    return x * 2

__tools__ = {
    "double": custom_func
}
```

## Creating Workflows

```python
# workflows.py or workflows/my_workflow.py
from workflow import Workflow

my_workflow = Workflow(name="my_workflow")
# ... configure workflow nodes ...

__workflows__ = {
    "my_workflow": my_workflow
}
```

## API Integration Example

Perfect for Flask/FastAPI applications:

```python
from fourier.config import get_config

class MyAPI:
    def __init__(self):
        self.config = get_config()

    def chat_endpoint(self, agent_name: str, message: str):
        """POST /api/chat"""
        response = self.config.invoke_agent(agent_name, query=message)
        return {"response": response['response']}
```

## Running the Examples

```bash
# Install dependencies
pip install -e .[all]

# Set up environment
cp .env.template .env
# Edit .env with your API keys

# Run main example
cd examples/user_project_example
python main.py

# Run API example
python api_example.py
```

## Benefits

### 1. No Import Hell
Instead of:
```python
from agents.customer_support import customer_support_agent
from agents.research import research_agent
from tools.data_tools import fetch_user_data
# ... 20 more imports
```

Just:
```python
from fourier.config import get_config
config = get_config()
```

### 2. Dynamic Invocation
```python
# Perfect for API endpoints
agent_name = request.json.get('agent')
response = config.invoke_agent(agent_name, query=...)
```

### 3. Easy to Extend
Add new agents/tools/workflows - they're automatically discovered!

### 4. Testable
```python
def test_agent():
    config = get_config()
    response = config.invoke_agent("test_agent", "test query")
    assert response['response'] is not None
```

## Configuration File

`.fourier_config.json` is auto-generated and contains:

```json
{
  "provider_config": {},
  "paths": {
    "agents_dir": "./agents",
    "workflows_file": "./workflows.py",
    "tools_dir": "./tools"
  },
  "registry": {
    "agents": ["customer_support", "research_agent"],
    "workflows": ["customer_onboarding"],
    "tools": ["fetch_user_data", "send_notification"]
  }
}
```

**Tip:** Add `.fourier_config.json` to `.gitignore` if it contains sensitive paths.

## Troubleshooting

### Resources not found

```python
# Use interactive mode
config = FourierConfig(base_dir=".", auto_discover=False)
config.discover_resources(interactive=True)
# You'll be prompted for custom paths
```

### List available resources

```python
config = get_config()
print(config.list_resources())
# Output: {'agents': [...], 'workflows': [...], 'tools': [...]}
```

### Get resource information

```python
info = config.get_resource_info("agents", "customer_support")
print(info)
# Shows: name, type, model, config, metadata
```

## Best Practices

1. **Initialize once** - Call `FourierConfig()` in your main entry point
2. **Use `get_config()`** - Access config from anywhere without re-initialization
3. **Export explicitly** - Use `__agents__`, `__workflows__`, `__tools__` for clarity
4. **Name consistently** - Use descriptive names for your resources
5. **Document tools** - Add docstrings to your tool functions
6. **Structure logically** - Group related agents/tools in subdirectories

## Next Steps

- Check out `main.py` for complete examples
- See `api_example.py` for API integration
- Read `CONFIG_SYSTEM.md` for architecture details
