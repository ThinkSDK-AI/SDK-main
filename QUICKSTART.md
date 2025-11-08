# Fourier SDK - Quick Start Guide

Get started with Fourier SDK in under 5 minutes!

## Installation

### Option 1: Quick Install (Everything)

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main
pip install -e .[all]
```

### Option 2: Minimal Install

```bash
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main
pip install -e .
```

## Initialize Your Project

The easiest way to get started is using the `init` command:

```bash
python cli.py init
```

This will guide you through an interactive wizard:

```
====================================================================
  Fourier SDK - Project Initialization Wizard
====================================================================

Project name [my_fourier_project]: my_app

📁 Choose project structure:
  1. Directory-based (agents/, tools/, workflows/)
  2. File-based (agents.py, tools.py, workflows.py)
Choice [1]: 1

🔧 What do you want to include?
Agents? [Y/n]: y
Workflows? [y/N]: n
Tools? [Y/n]: y

🤖 Select LLM providers (comma-separated):
  1. Groq
  2. Anthropic
  3. OpenAI
  4. Together AI
  5. Bedrock
  6. All
Providers [1]: 1,2

✨ Additional features:
Generate API server template? [y/N]: y
Include web search? [y/N]: y

📋 Summary:
  Project: my_app
  Structure: directory
  Agents: Yes
  Workflows: No
  Tools: Yes
  Providers: groq, anthropic
  Features: api, search

Create project? [Y/n]: y

🚀 Creating Fourier Project
============================================================
  📁 Created: my_app
  📁 Created: my_app/agents
  📄 Created: my_app/agents/__init__.py
  📄 Created: my_app/agents/example_agent.py
  📁 Created: my_app/tools
  📄 Created: my_app/tools/__init__.py
  📄 Created: my_app/tools/example_tools.py
  📄 Created: my_app/main.py
  📄 Created: my_app/api.py
  📄 Created: my_app/.env
  📄 Created: my_app/.gitignore
  📄 Created: my_app/requirements.txt
  📄 Created: my_app/README.md
  📄 Created: my_app/.fourier_config.json

✅ Project 'my_app' created successfully!
📁 Location: ./my_app
```

## Next Steps

### 1. Navigate to Your Project

```bash
cd my_app
```

### 2. Configure API Keys

Edit the `.env` file:

```bash
nano .env
```

Add your API keys:

```bash
# Groq
GROQ_API_KEY=your_groq_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Run Your Application

```bash
python main.py
```

Output:

```
============================================================
my_app - Initializing
============================================================

Discovered Resources:
  Agents: 1 - ['example_agent']
  Workflows: 0 - []
  Tools: 2 - ['example_tool', 'fetch_data']

Application started!
Use get_config() to access resources from anywhere.

Example: Invoking agent 'example_agent'...
Response: Hello! I'm your example agent...
```

## Using the Config System

The auto-generated project uses the Fourier Config System for clean architecture:

### In Your Main File

```python
from fourier.config import FourierConfig

# Initialize once
config = FourierConfig(base_dir=".", auto_discover=True)
```

### Anywhere Else in Your Code

```python
from fourier.config import get_config

# Get the global config
config = get_config()

# Invoke agent by name
response = config.invoke_agent("example_agent", query="Hello!")

# Use tool by name
result = config.invoke_tool("fetch_data", query="search term")
```

## Running the API Server

If you generated an API template:

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Run the server
python api.py
```

Server runs at: http://localhost:8000

### API Endpoints

```bash
# Health check
curl http://localhost:8000/

# List resources
curl http://localhost:8000/resources

# Chat with agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent": "example_agent", "message": "Hello"}'

# Execute tool
curl -X POST http://localhost:8000/tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "fetch_data", "params": {"query": "test"}}'
```

## Adding New Agents

Create a file in `agents/` directory:

**agents/support_agent.py**

```python
from fourier import Fourier
from agent import Agent, AgentConfig
import os

client = Fourier(
    api_key=os.getenv("GROQ_API_KEY"),
    provider="groq"
)

config = AgentConfig(
    name="support_agent",
    description="Customer support agent",
    max_iterations=5
)

support_agent = Agent(
    client=client,
    model="llama-3.1-8b-instant",
    config=config
)

# Export for auto-discovery
__agents__ = {
    "support": support_agent
}
```

The agent is automatically discovered! Use it:

```python
from fourier.config import get_config

config = get_config()
response = config.invoke_agent("support", query="Help me!")
```

## Adding New Tools

Create a file in `tools/` directory:

**tools/api_tools.py**

```python
from fourier.config import tool

@tool
def call_external_api(endpoint: str) -> dict:
    """Call an external API"""
    # Your implementation
    return {"status": "success", "data": "..."}
```

Use it:

```python
config = get_config()
result = config.invoke_tool("call_external_api", endpoint="/users")
```

## CLI Commands

The Fourier CLI provides many useful commands:

```bash
# Initialize new project
python cli.py init

# Interactive shell
python cli.py interactive

# Quick chat
python cli.py chat "What is AI?"

# Create agent
python cli.py create-agent --name MyAgent

# List agents
python cli.py list-agents

# Run agent
python cli.py run --agent MyAgent --query "Hello"
```

## Project Structure Overview

After initialization, your project looks like:

```
my_app/
├── agents/                  # Agent definitions
│   ├── __init__.py
│   └── example_agent.py     # Auto-generated example
├── tools/                   # Tool definitions
│   ├── __init__.py
│   └── example_tools.py     # Auto-generated examples
├── main.py                  # Application entry point
├── api.py                   # FastAPI server (optional)
├── .env                     # API keys (gitignored)
├── .fourier_config.json     # Auto-generated config
├── .gitignore               # Git ignore rules
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Common Patterns

### Pattern 1: API Integration

```python
from fastapi import FastAPI
from fourier.config import get_config

app = FastAPI()

@app.post("/chat")
async def chat(agent: str, message: str):
    config = get_config()
    response = config.invoke_agent(agent, query=message)
    return {"response": response['response']}
```

### Pattern 2: Message Queue Handler

```python
from fourier.config import get_config

def on_message(message):
    config = get_config()
    agent = message['routing_key']
    return config.invoke_agent(agent, query=message['content'])
```

### Pattern 3: Background Jobs

```python
from celery import Celery
from fourier.config import get_config

@app.task
def process_query(agent_name, query):
    config = get_config()
    return config.invoke_agent(agent_name, query=query)
```

## Learning More

### Documentation

- **[CONFIG_SYSTEM.md](CONFIG_SYSTEM.md)** - Complete config system guide
- **[INSTALLATION.md](INSTALLATION.md)** - Installation options
- **[BEDROCK.md](BEDROCK.md)** - AWS Bedrock guide
- **[README.md](README.md)** - Main documentation

### Examples

- **examples/user_project_example/** - Complete working example
- **examples/bedrock_*.py** - Bedrock examples
- **examples/agent_example.py** - Agent examples
- **examples/workflow_example.py** - Workflow examples

### CLI Help

```bash
python cli.py --help
python cli.py init --help
python cli.py create-agent --help
```

## Troubleshooting

### Issue: "No module named 'fourier'"

```bash
# Make sure you installed the SDK
pip install -e .
```

### Issue: "API key cannot be empty"

```bash
# Edit .env and add your API keys
nano .env
```

### Issue: "Agent not found"

```bash
# Check if agent is registered
python -c "from fourier.config import get_config; print(get_config().list_resources())"
```

### Issue: "Provider requires additional dependencies"

```bash
# Install provider-specific dependencies
pip install -e .[bedrock]  # For Bedrock
pip install -e .[all]       # For all providers
```

## Get Help

- **GitHub Issues**: https://github.com/ThinkSDK-AI/SDK-main/issues
- **Documentation**: Complete guides in the repository
- **Examples**: Working examples in `examples/` directory

## Next Steps

1. ✅ Initialize project: `python cli.py init`
2. ✅ Configure API keys: Edit `.env`
3. ✅ Run application: `python main.py`
4. ✅ Add your agents: Create in `agents/`
5. ✅ Add your tools: Create in `tools/`
6. ✅ Build your app: Use `get_config()` everywhere!

Happy coding with Fourier SDK! 🚀
