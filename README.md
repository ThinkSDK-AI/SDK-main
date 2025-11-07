# FourierSDK

A Python SDK for accessing Large Language Models (LLMs) from various inference providers like Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius. FourierSDK provides a unified interface similar to the OpenAI SDK while adding support for function calling, internet search, autonomous agents, and multiple providers with standardized response formats.

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
  - [Assistants](#assistants)
  - [Workflows](#workflows)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
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

### Assistants

FourierSDK includes an Assistant framework for simple, context-aware conversational interfaces. Unlike Agents, Assistants do not automatically execute tools - they focus on maintaining conversation history and can optionally integrate with RAG (Retrieval Augmented Generation) for document-based Q&A.

**When to Use Assistants:**
- Simple conversational interactions
- Document Q&A with RAG support
- Maintaining conversation history
- Building chatbots without automatic tool execution
- Lower overhead compared to Agents

**Basic Assistant Example:**

```python
from fourier import Fourier
from assistant import Assistant, AssistantConfig

# Create client
client = Fourier(api_key=os.getenv("GROQ_API_KEY"), provider="groq")

# Create assistant
assistant = Assistant(
    client=client,
    name="ChatBot",
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        temperature=0.7,
        max_history=50,
        system_prompt="You are a friendly and helpful assistant."
    )
)

# Have a conversation
response1 = assistant.chat("My name is Alice")
print(response1["output"])

response2 = assistant.chat("What's my name?")
print(response2["output"])  # Will remember "Alice"

# Get conversation stats
stats = assistant.get_stats()
print(f"Messages: {stats['total_messages']}")
```

**RAG-Enabled Assistant:**

```python
# Create assistant with RAG
assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        enable_rag=True,
        rag_top_k=3,
        system_prompt="Answer questions based on the provided documents."
    )
)

# Add documents
documents = [
    {
        "content": "FourierSDK is a Python SDK for accessing multiple LLM providers.",
        "metadata": {"source": "docs.txt"}
    },
    {
        "content": "The Agent class allows autonomous tool execution with configurable behavior.",
        "metadata": {"source": "agent_docs.txt"}
    }
]

assistant.add_documents(documents)

# Ask questions about documents
response = assistant.chat("What is FourierSDK?")
print(response["output"])  # Uses document context to answer
```

**Conversation Persistence:**

```python
# Save conversation
assistant.save_conversation("conversation.json")

# Load in new assistant
new_assistant = Assistant(client=client, model="mixtral-8x7b-32768")
new_assistant.load_conversation("conversation.json")

# Continue conversation
response = new_assistant.chat("What were we discussing?")
```

**Key Assistant Features:**
- ✅ **Conversation History**: Automatic history management with configurable limits
- ✅ **RAG Support**: Document-based context with keyword retrieval
- ✅ **Persistence**: Save and load conversations
- ✅ **Multiple Personalities**: Create specialized assistants with different prompts
- ✅ **Simple API**: Easier to use than Agents when tools aren't needed
- ✅ **Lower Overhead**: No tool execution loop, faster responses

**Assistant vs Agent:**

| Feature | Assistant | Agent |
|---------|-----------|-------|
| Conversation History | ✅ Yes | ✅ Yes |
| RAG Support | ✅ Yes | ❌ No |
| Auto Tool Execution | ❌ No | ✅ Yes |
| Thinking Mode | ❌ No | ✅ Yes |
| Complexity | Low | High |
| Best For | Chat, Q&A | Task Completion |

See [ASSISTANT.md](ASSISTANT.md) for complete documentation, RAG examples, and best practices.

### Workflows

FourierSDK includes a powerful Workflow system for orchestrating complex AI pipelines. Similar to n8n, workflows let you compose agents, assistants, and transformations into sophisticated multi-step processes with conditional branching.

**Workflow Features:**
- 7 node types: INPUT, AGENT, ASSISTANT, TRANSFORM, CONDITION, OUTPUT, TOOL, RAG
- Sequential execution engine
- Conditional branching
- Visual workflow representation
- Execution tracking and debugging
- JSON persistence

**Simple Linear Workflow:**

```python
from fourier import Fourier
from assistant import Assistant
from workflow import Workflow

# Create client and assistant
client = Fourier(api_key=os.getenv("GROQ_API_KEY"), provider="groq")
assistant = Assistant(client=client, model="mixtral-8x7b-32768")

# Create workflow
workflow = Workflow(name="SimpleWorkflow")

# Add nodes
input_node = workflow.add_input_node("Start")
assistant_node = workflow.add_assistant_node(assistant, "Assistant")
output_node = workflow.add_output_node("Result")

# Connect nodes
workflow.connect(input_node.node_id, assistant_node.node_id)
workflow.connect(assistant_node.node_id, output_node.node_id)

# Visualize
print(workflow.visualize())

# Execute
result = workflow.execute("What is the capital of France?")
print(f"Output: {result['output']}")
print(f"Execution time: {result['execution_time']:.2f}s")
```

**Conditional Branching Workflow:**

```python
# Create assistants with different prompts
brief_assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="Give very brief, one-sentence answers."
    )
)

detailed_assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="Give detailed, comprehensive answers."
    )
)

# Create workflow with branching
workflow = Workflow(name="ConditionalWorkflow")

input_node = workflow.add_input_node()

# Condition: Check if query is short
condition_node = workflow.add_condition_node(
    lambda x: len(str(x)) < 20,
    "IsShortQuery"
)

brief_node = workflow.add_assistant_node(brief_assistant, "BriefResponse")
detailed_node = workflow.add_assistant_node(detailed_assistant, "DetailedResponse")
output_node = workflow.add_output_node()

# Connect and set branches
workflow.connect(input_node.node_id, condition_node.node_id)
condition_node.true_branch = brief_node.node_id
condition_node.false_branch = detailed_node.node_id
workflow.connect(brief_node.node_id, output_node.node_id)
workflow.connect(detailed_node.node_id, output_node.node_id)

# Execute with different inputs
result1 = workflow.execute("What is AI?")  # Short → brief response
result2 = workflow.execute("Can you explain quantum computing?")  # Long → detailed response
```

**Multi-Agent Research Pipeline:**

```python
from agent import Agent, AgentConfig

# Create specialized agents
researcher = Agent(
    client=client,
    name="Researcher",
    config=AgentConfig(
        system_prompt="You are a researcher. Provide factual information.",
        thinking_mode=True
    )
)

analyzer = Agent(
    client=client,
    name="Analyzer",
    config=AgentConfig(
        system_prompt="You are an analyzer. Break down and structure information."
    )
)

summarizer = Assistant(
    client=client,
    config=AssistantConfig(
        system_prompt="You are a summarizer. Create concise summaries."
    )
)

# Create pipeline
workflow = Workflow(name="ResearchPipeline")

input_node = workflow.add_input_node("Topic")
researcher_node = workflow.add_agent_node(researcher, "Research")
analyzer_node = workflow.add_agent_node(analyzer, "Analyze")
summarizer_node = workflow.add_assistant_node(summarizer, "Summarize")
output_node = workflow.add_output_node("Report")

# Connect stages
workflow.connect(input_node.node_id, researcher_node.node_id)
workflow.connect(researcher_node.node_id, analyzer_node.node_id)
workflow.connect(analyzer_node.node_id, summarizer_node.node_id)
workflow.connect(summarizer_node.node_id, output_node.node_id)

# Execute research pipeline
result = workflow.execute("Quantum computing applications", verbose=True)
print(result["output"])
```

**Transform Pipeline:**

```python
# Add data transformations between processing steps
workflow = Workflow(name="TransformPipeline")

input_node = workflow.add_input_node()

# Preprocessing
clean_node = workflow.add_transform_node(
    lambda x: x.strip().lower(),
    "Clean"
)

prefix_node = workflow.add_transform_node(
    lambda x: f"Question: {x}",
    "AddPrefix"
)

# Processing
assistant_node = workflow.add_assistant_node(assistant, "Process")

# Postprocessing
extract_node = workflow.add_transform_node(
    lambda x: x.split('.')[0] + '.',
    "ExtractFirstSentence"
)

output_node = workflow.add_output_node()

# Connect all stages
workflow.connect(input_node.node_id, clean_node.node_id)
workflow.connect(clean_node.node_id, prefix_node.node_id)
workflow.connect(prefix_node.node_id, assistant_node.node_id)
workflow.connect(assistant_node.node_id, extract_node.node_id)
workflow.connect(extract_node.node_id, output_node.node_id)

result = workflow.execute("what is machine learning?", verbose=True)
```

**Workflow Visualization:**

```python
print(workflow.visualize())

# Output:
# Workflow: ResearchPipeline
# Nodes: 5
# ─────────────────────────────
# INPUT → input_123 (Topic)
# AGENT → agent_456 (Research)
# AGENT → agent_789 (Analyze)
# ASSISTANT → asst_012 (Summarize)
# OUTPUT → output_345 (Report)
#
# Connections:
# input_123 → agent_456
# agent_456 → agent_789
# agent_789 → asst_012
# asst_012 → output_345
```

**Workflow Persistence:**

```python
# Save workflow structure
workflow.save("workflow.json")

# View saved structure
import json
with open("workflow.json", 'r') as f:
    data = json.load(f)

print(f"Workflow: {data['name']}")
print(f"Nodes: {len(data['nodes'])}")
print(f"Executions: {data['execution_count']}")
```

**Key Workflow Benefits:**
- ✅ **Composable**: Mix agents, assistants, and transformations
- ✅ **Visual**: Text-based workflow visualization
- ✅ **Conditional**: Branching logic based on data
- ✅ **Traceable**: Full execution tracking and debugging
- ✅ **Reusable**: Save and version workflow structures
- ✅ **Flexible**: 7 node types for different operations

See [WORKFLOW.md](WORKFLOW.md) for complete documentation, advanced patterns, and best practices.

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

## Features

### Core Features
- **Multi-Provider Support**: Easily switch between Groq, Together AI, OpenAI, Anthropic, Perplexity, and Nebius
- **Standardized Response Format**: Consistent response structure regardless of the provider
- **Function Calling**: Define and use functions/tools with JSON schema validation
- **Internet Search**: Augment LLM responses with up-to-date information from the web
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive exception hierarchy for precise error handling
- **Logging**: Production-ready logging framework
- **Token Usage Tracking**: Monitor token consumption across providers
- **Customizable Base URLs**: For enterprise deployments or custom endpoints

### CLI Features
- **Command Line Interface (CLI)**: Comprehensive CLI for managing agents and MCP tools without writing code
- **Interactive Shell**: REPL-style interface for agent interaction and management
- **Agent Persistence**: Save and load agent configurations with JSON storage
- **Configuration Management**: JSON-based config storage at ~/.fourier/config.json
- **Scriptable**: Use CLI in automation and CI/CD pipelines
- **ANSI Colored Output**: Enhanced readability in terminal

### Agent Features
- **Autonomous Agents**: Create agents that automatically use tools and manage conversations
- **Thinking Mode**: Deep research capability with automatic multi-query web searches and synthesis
- **Tool Registration**: Register custom tools with JSON schema validation
- **Conversation Memory**: Maintains context across multiple interactions
- **Configurable Behavior**: Fine-tune iterations, timeouts, and error handling
- **Intermediate Steps**: Track tool usage and execution flow
- **Error Resilience**: Continue execution even when tools fail

### Assistant Features
- **Simple Assistants**: Context-aware LLM wrappers without automatic tool execution
- **RAG Support**: Document-based Q&A with keyword-based retrieval
- **Conversation History**: Automatic history management with configurable limits
- **Conversation Persistence**: Save and load conversation state
- **Multiple Personalities**: Create specialized assistants with different prompts
- **Low Overhead**: Simpler API and faster responses than agents

### Workflow Features
- **Node-Based Orchestration**: Compose complex AI pipelines like n8n
- **7 Node Types**: INPUT, AGENT, ASSISTANT, TRANSFORM, CONDITION, OUTPUT, TOOL, RAG
- **Sequential Execution**: Ordered execution with cycle detection
- **Conditional Branching**: Dynamic routing based on data
- **Visual Representation**: Text-based workflow visualization
- **Execution Tracking**: Full debugging and performance metrics
- **Workflow Persistence**: Save and version workflow structures

### MCP Features
- **Model Context Protocol (MCP)**: Connect to remote MCP servers, load tools from directories
- **MCP URL Support**: Connect to remote MCP servers via HTTP/HTTPS
- **MCP Config Files**: Use Claude Desktop-compatible configuration files
- **MCP Directories**: Load tools from local directories
- **Dynamic Tool Loading**: Add and manage MCP tools at runtime

### Production Features
- **Input Sanitization**: Automatic query validation and cleaning (XSS/injection prevention)
- **Rate Limiting**: Built-in delays prevent API abuse
- **Context Management**: Automatic truncation prevents token limit issues
- **Configuration Validation**: Auto-correction of invalid parameters
- **Graceful Degradation**: Fallback mechanisms for partial failures
- **Performance Monitoring**: Detailed timing and success metrics
- **Comprehensive Testing**: 40+ unit tests for thinking mode and core features

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
