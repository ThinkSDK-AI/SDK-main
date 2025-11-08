# Fourier Config System - Architecture Documentation

Complete guide to the centralized configuration and execution system in Fourier SDK.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Usage Patterns](#usage-patterns)
- [Folder Structures](#folder-structures)
- [Auto-Discovery](#auto-discovery)
- [API Integration](#api-integration)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

---

## Overview

The Fourier Config System provides a **centralized configuration and execution engine** that eliminates the need to import agents, workflows, and tools directly. Instead, you:

1. **Initialize once** - Configure the system in your main entry point
2. **Auto-discover** - The system finds all your resources automatically
3. **Invoke by name** - Call agents/workflows/tools without imports
4. **Use anywhere** - Access the global config from any part of your codebase

### Benefits

✅ **No import hell** - Stop importing dozens of modules
✅ **Dynamic invocation** - Perfect for APIs and event handlers
✅ **Auto-discovery** - New resources are found automatically
✅ **Clean architecture** - Separation of config and business logic
✅ **Easy testing** - Mock and test resources easily
✅ **Flexible structure** - Support for files or directories

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────┐
│         Application Entry Point             │
│  (main.py, __init__.py, app.py)            │
│                                             │
│  config = FourierConfig(auto_discover=True) │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Auto-Discovery System               │
│  • Scans agents/, tools/, workflows/        │
│  • Loads agents.py, tools.py, workflows.py  │
│  • Extracts resources from modules          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Resource Registry                 │
│  • agents: {"name": Agent(...)}            │
│  • workflows: {"name": Workflow(...)}      │
│  • tools: {"name": function}               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Invocation API                      │
│  • invoke_agent(name, query)               │
│  • invoke_workflow(name, data)             │
│  • invoke_tool(name, *args)                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      Application Code Anywhere              │
│  (API handlers, message queues, etc.)      │
│                                             │
│  config = get_config()                     │
│  response = config.invoke_agent("name", ...)│
└─────────────────────────────────────────────┘
```

---

## Core Components

### 1. FourierConfig

Central configuration class that manages everything.

```python
from fourier.config import FourierConfig

config = FourierConfig(
    base_dir=".",           # Base directory for discovery
    auto_discover=True,     # Auto-discover resources
    paths=None,             # Custom paths (optional)
    provider_config={}      # Provider-specific config
)
```

**Key Methods:**
- `discover_resources()` - Find and register resources
- `invoke_agent(name, query, **kwargs)` - Run an agent
- `invoke_workflow(name, input_data, **kwargs)` - Execute workflow
- `invoke_tool(name, *args, **kwargs)` - Call a tool
- `list_resources()` - List all registered resources
- `save_config()` / `load_config()` - Persist configuration

### 2. ResourceRegistry

Internal registry that stores all discovered resources.

```python
# Accessed via config.registry
agents = config.registry.list_agents()
workflows = config.registry.list_workflows()
tools = config.registry.list_tools()

# Get specific resource
agent = config.registry.get_agent("my_agent")
```

### 3. FourierPaths

Configuration for resource locations.

```python
from fourier.config import FourierPaths

paths = FourierPaths(
    base_dir=".",
    agents_dir="./agents",          # Directory of agents
    agents_file="./agents.py",      # Single agents file
    workflows_dir="./workflows",
    workflows_file="./workflows.py",
    tools_dir="./tools",
    tools_file="./tools.py",
)
```

**Auto-detection:** If paths are not specified, the system automatically looks for:
- `agents/` directory or `agents.py` file
- `workflows/` directory or `workflows.py` file
- `tools/` directory or `tools.py` file

### 4. get_config()

Global accessor function.

```python
from fourier.config import get_config

# Get the global config instance (anywhere in your code)
config = get_config()
response = config.invoke_agent("my_agent", "Hello")
```

### 5. @tool Decorator

Mark functions as tools for auto-discovery.

```python
from fourier.config import tool

@tool
def my_function(arg1: str, arg2: int) -> str:
    """Tool description"""
    return f"Result: {arg1} - {arg2}"
```

---

## Usage Patterns

### Pattern 1: Initialize Once, Use Everywhere

**In main.py or __init__.py:**
```python
from fourier.config import FourierConfig
from dotenv import load_dotenv

load_dotenv()

# Initialize config once at startup
config = FourierConfig(base_dir=".", auto_discover=True)
```

**In any other module:**
```python
from fourier.config import get_config

def handle_user_request(agent_name: str, message: str):
    config = get_config()
    response = config.invoke_agent(agent_name, query=message)
    return response
```

### Pattern 2: API Integration

**Flask/FastAPI Example:**
```python
from flask import Flask, request, jsonify
from fourier.config import get_config

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    agent_name = data.get('agent')
    message = data.get('message')

    config = get_config()
    response = config.invoke_agent(agent_name, query=message)

    return jsonify({
        'success': True,
        'response': response['response']
    })

@app.route('/api/agents', methods=['GET'])
def list_agents():
    config = get_config()
    return jsonify({
        'agents': config.registry.list_agents()
    })
```

### Pattern 3: Event-Driven

**Message Queue Handler:**
```python
from fourier.config import get_config

def on_message_received(message):
    """Kafka/RabbitMQ message handler"""
    config = get_config()

    # Determine which agent to use based on message type
    agent_name = message['routing_key']

    response = config.invoke_agent(
        agent_name,
        query=message['content']
    )

    return response
```

### Pattern 4: Background Jobs

**Celery Task:**
```python
from celery import Celery
from fourier.config import get_config

app = Celery('tasks')

@app.task
def process_research_query(query: str):
    config = get_config()
    result = config.invoke_agent("research_agent", query=query)
    return result
```

---

## Folder Structures

### Structure 1: Directory-Based (Recommended for large projects)

```
my_project/
├── main.py
├── agents/
│   ├── __init__.py
│   ├── customer_support.py
│   ├── research.py
│   └── sales.py
├── workflows/
│   ├── __init__.py
│   ├── onboarding.py
│   └── processing.py
└── tools/
    ├── __init__.py
    ├── data_tools.py
    └── api_tools.py
```

**Pros:**
- Organized and scalable
- Easy to find specific resources
- Clean separation

### Structure 2: File-Based (Good for smaller projects)

```
my_project/
├── main.py
├── agents.py      # All agents in one file
├── workflows.py   # All workflows in one file
└── tools.py       # All tools in one file
```

**Pros:**
- Simpler for small projects
- Everything in one place
- Less file navigation

### Structure 3: Mixed (Flexible approach)

```
my_project/
├── main.py
├── agents/        # Complex agents in directory
│   ├── support.py
│   └── research.py
├── workflows.py   # Simple workflows in file
└── tools.py       # Simple tools in file
```

**Pros:**
- Best of both worlds
- Use directories for complex resources
- Use files for simple resources

### Structure 4: Custom Paths

```
my_project/
├── app/
│   ├── main.py
│   └── config/
│       ├── my_agents/
│       ├── my_workflows/
│       └── my_tools/
```

**Configuration:**
```python
from fourier.config import FourierConfig, FourierPaths

paths = FourierPaths(
    base_dir="./app/config",
    agents_dir="./app/config/my_agents",
    workflows_dir="./app/config/my_workflows",
    tools_dir="./app/config/my_tools"
)

config = FourierConfig(paths=paths, auto_discover=True)
```

---

## Auto-Discovery

### How It Works

1. **Scan** - System scans configured paths
2. **Load** - Imports Python modules dynamically
3. **Extract** - Finds resources using multiple methods
4. **Register** - Adds resources to registry

### Discovery Methods

#### Method 1: Explicit Export (Recommended)

**For Agents:**
```python
# agents/my_agent.py
from agent import Agent

my_agent = Agent(...)

__agents__ = {
    "my_agent": my_agent,
    "support": my_agent  # Can have multiple names
}
```

**For Workflows:**
```python
# workflows/my_workflow.py
from workflow import Workflow

workflow = Workflow(...)

__workflows__ = {
    "my_workflow": workflow
}
```

**For Tools:**
```python
# tools/my_tools.py
def my_function(x):
    return x * 2

__tools__ = {
    "double": my_function
}
```

#### Method 2: Decorator (Tools only)

```python
from fourier.config import tool

@tool
def my_function(arg: str) -> str:
    """This function will be auto-discovered"""
    return arg.upper()
```

#### Method 3: Naming Convention (Tools only)

```python
# Functions starting with tool_ are auto-discovered
def tool_process_data(data: dict) -> dict:
    """Process data"""
    return {"processed": True, "data": data}
```

#### Method 4: Auto-Detection (Agents/Workflows)

```python
# Just create instances at module level
my_agent = Agent(...)  # Auto-detected
my_workflow = Workflow(...)  # Auto-detected
```

### Interactive Discovery

If resources aren't found automatically:

```python
config = FourierConfig(auto_discover=False)
config.discover_resources(interactive=True)
```

This will prompt:
```
Agents not found in standard locations.
Expected: agents/ directory or agents.py file
Specify custom path for agents? (y/n): y
Enter path to agents directory or file: ./my_custom_agents
```

---

## API Integration

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from fourier.config import FourierConfig, get_config
from pydantic import BaseModel

# Initialize config at startup
app = FastAPI()

@app.on_event("startup")
async def startup():
    config = FourierConfig(base_dir=".", auto_discover=True)
    print(f"Discovered: {config.list_resources()}")

# Request models
class ChatRequest(BaseModel):
    agent: str
    message: str

class ToolRequest(BaseModel):
    tool: str
    params: dict

# Endpoints
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = get_config()
        response = config.invoke_agent(request.agent, query=request.message)
        return {"success": True, "response": response['response']}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/tool")
async def execute_tool(request: ToolRequest):
    try:
        config = get_config()
        result = config.invoke_tool(request.tool, **request.params)
        return {"success": True, "result": result}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/resources")
async def list_resources():
    config = get_config()
    return config.list_resources()
```

### Flask Example

```python
from flask import Flask, request, jsonify
from fourier.config import FourierConfig, get_config

app = Flask(__name__)

# Initialize at startup
with app.app_context():
    config = FourierConfig(base_dir=".", auto_discover=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    config = get_config()

    try:
        response = config.invoke_agent(
            data['agent'],
            query=data['message']
        )
        return jsonify({'success': True, 'response': response['response']})
    except KeyError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
```

---

## Advanced Usage

### Custom Resource Metadata

```python
# agents/my_agent.py
my_agent = Agent(...)

__agents__ = {
    "my_agent": my_agent
}

# Optional: Add metadata
__metadata__ = {
    "my_agent": {
        "version": "1.0.0",
        "author": "Your Name",
        "description": "Customer support agent",
        "tags": ["support", "customer"],
        "rate_limit": 100
    }
}
```

Access metadata:
```python
config = get_config()
metadata = config.registry.get_metadata("agents", "my_agent")
```

### Dynamic Agent Selection

```python
def handle_request(user_tier: str, message: str):
    config = get_config()

    # Choose agent based on user tier
    agent_map = {
        "free": "basic_agent",
        "pro": "advanced_agent",
        "enterprise": "premium_agent"
    }

    agent_name = agent_map.get(user_tier, "basic_agent")
    response = config.invoke_agent(agent_name, query=message)

    return response
```

### Chaining Tools and Agents

```python
config = get_config()

# Use tool to get data
user_data = config.invoke_tool("fetch_user_data", user_id="123")

# Use agent to process it
analysis = config.invoke_agent(
    "analysis_agent",
    query=f"Analyze this user data: {user_data}"
)

# Use another tool to send results
config.invoke_tool("send_email", to=user_data['email'], content=analysis['response'])
```

### Conditional Resource Loading

```python
import os

def should_load_agent(filename: str) -> bool:
    """Custom logic for which agents to load"""
    # Only load agents for current environment
    env = os.getenv("ENVIRONMENT", "dev")
    return env in filename or "common" in filename

# Custom discovery with filtering
config = FourierConfig(auto_discover=False)
# Implement custom loading logic here
```

---

## Best Practices

### 1. Initialize Early

```python
# In main.py or __init__.py
if __name__ == "__main__":
    config = FourierConfig(auto_discover=True)
    # Rest of your application
```

### 2. Use Explicit Exports

```python
# Prefer this
__agents__ = {"name": agent}

# Over implicit detection
# (which might pick up test objects)
```

### 3. Document Your Resources

```python
@tool
def process_data(data: dict) -> dict:
    """
    Process user data and return results.

    Args:
        data: Input data dictionary

    Returns:
        Processed data with additional fields
    """
    return {...}
```

### 4. Handle Errors Gracefully

```python
try:
    response = config.invoke_agent("my_agent", query="...")
except KeyError:
    # Agent not found - handle gracefully
    available = config.registry.list_agents()
    print(f"Agent not found. Available: {available}")
```

### 5. Save Configuration

```python
# After discovery, save for future use
config.discover_resources()
config.save_config()  # Saves to .fourier_config.json
```

### 6. Version Your Resources

```python
__agents__ = {
    "support_v1": support_agent_v1,
    "support_v2": support_agent_v2,
    "support": support_agent_v2  # Alias to latest
}
```

---

## Configuration File

`.fourier_config.json` stores discovered configuration:

```json
{
  "provider_config": {
    "default_provider": "groq",
    "timeout": 30
  },
  "paths": {
    "agents_dir": "./agents",
    "workflows_file": "./workflows.py",
    "tools_dir": "./tools"
  },
  "registry": {
    "agents": ["customer_support", "research_agent"],
    "workflows": ["onboarding"],
    "tools": ["fetch_data", "send_notification"]
  }
}
```

**Tip:** Add to `.gitignore` if it contains sensitive paths.

---

## Migration Guide

### From Direct Imports

**Before:**
```python
from agents.support import customer_support
from agents.research import research_agent
from tools.data import fetch_user_data

response = customer_support.run("Hello")
data = fetch_user_data(user_id="123")
```

**After:**
```python
from fourier.config import get_config

config = get_config()
response = config.invoke_agent("customer_support", "Hello")
data = config.invoke_tool("fetch_user_data", user_id="123")
```

### Benefits of Migration

- ✅ Fewer imports
- ✅ Dynamic agent selection
- ✅ Easier to test
- ✅ Auto-discovery of new resources
- ✅ Centralized configuration

---

## Examples

See `examples/user_project_example/` for complete working examples:

- `main.py` - Basic usage
- `api_example.py` - API integration
- `agents/` - Example agents
- `tools/` - Example tools
- `workflows.py` - Example workflows

---

## Troubleshooting

### Resources not discovered

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

config = FourierConfig(auto_discover=True)
```

### List what was found

```python
config = get_config()
print(config.list_resources())
```

### Use interactive mode

```python
config = FourierConfig(auto_discover=False)
config.discover_resources(interactive=True)
```

---

## Summary

The Fourier Config System provides:

- **Centralized configuration** - Initialize once, use everywhere
- **Auto-discovery** - Finds resources automatically
- **Dynamic invocation** - Call by name without imports
- **Flexible structure** - Supports files or directories
- **Perfect for APIs** - Ideal for web services and event handlers
- **Easy testing** - Mock and test resources easily

**Get started:**

```bash
cd examples/user_project_example
python main.py
```
