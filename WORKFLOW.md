# Workflow System

The Workflow system provides node-based orchestration for composing complex AI pipelines. Similar to tools like n8n, workflows allow you to connect different processing nodes (agents, assistants, transformations, conditions) into sophisticated multi-step processes.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Node Types](#node-types)
- [Building Workflows](#building-workflows)
- [Execution Model](#execution-model)
- [Conditional Logic](#conditional-logic)
- [Workflow Visualization](#workflow-visualization)
- [Persistence](#persistence)
- [Best Practices](#best-practices)
- [Advanced Patterns](#advanced-patterns)
- [API Reference](#api-reference)

## Overview

Workflows enable you to:
- Chain multiple agents and assistants
- Transform data between steps
- Implement conditional branching
- Create reusable processing pipelines
- Visualize and debug complex flows

**Key Features:**
- 7 node types (INPUT, AGENT, ASSISTANT, TRANSFORM, CONDITION, OUTPUT, TOOL, RAG)
- Sequential execution engine
- Conditional branching support
- Cycle detection
- Execution tracking and debugging
- JSON serialization for persistence
- Text-based visualization

## Core Concepts

### Workflow
A workflow is a directed graph of nodes connected by edges. Data flows from node to node, with each node performing a specific operation.

### Node
A node is a single processing unit in the workflow. Each node:
- Has a unique ID
- Has a name for identification
- Receives input data
- Produces output data
- Can be connected to other nodes

### Edge
An edge connects two nodes, defining the flow of data from one node (source) to another (target).

### Execution
Execution starts at INPUT nodes and follows edges sequentially until OUTPUT nodes are reached. The workflow tracks execution status and results for each node.

### Context
A shared dictionary that persists across node executions, allowing nodes to share state and metadata.

## Quick Start

### Simple Linear Workflow

```python
from fourier import Fourier
from assistant import Assistant
from workflow import Workflow

# Create client and assistant
client = Fourier(api_key="your-api-key", provider="groq")
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

# Execute
result = workflow.execute("What is the capital of France?")
print(result["output"])  # "Paris" or similar response
```

### Flow Diagram
```
[INPUT: Start] → [ASSISTANT: Assistant] → [OUTPUT: Result]
```

## Node Types

### 1. INPUT Node
Entry point for data into the workflow.

```python
input_node = workflow.add_input_node("UserInput")
```

**Behavior:**
- Receives initial data
- Passes data unchanged to next node
- Every workflow must start with an INPUT node

### 2. AGENT Node
Executes an autonomous agent with tool access.

```python
from agent import Agent, AgentConfig

agent = Agent(
    client=client,
    name="MathAgent",
    model="mixtral-8x7b-32768"
)

# Register tools
agent.register_tool(
    name="calculator",
    description="Perform arithmetic",
    parameters={...},
    function=calculator_fn
)

agent_node = workflow.add_agent_node(agent, "Calculator")
```

**Behavior:**
- Receives text input
- Executes agent with potential tool calls
- Returns agent's final response

**Use Cases:**
- Complex problem solving
- Multi-step reasoning
- Tool-based operations
- Research tasks

### 3. ASSISTANT Node
Executes a simple assistant (no automatic tools).

```python
assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="You are a helpful assistant."
    )
)

assistant_node = workflow.add_assistant_node(assistant, "Helper")
```

**Behavior:**
- Receives text input
- Generates conversational response
- Maintains conversation history
- Optional RAG support

**Use Cases:**
- Simple Q&A
- Text generation
- Document-based responses
- Conversational interfaces

### 4. TRANSFORM Node
Applies a custom transformation function to data.

```python
# Simple transformation
uppercase_node = workflow.add_transform_node(
    lambda x: x.upper(),
    "ToUppercase"
)

# Complex transformation
def extract_keywords(text: str) -> str:
    words = text.split()
    keywords = [w for w in words if len(w) > 5]
    return ", ".join(keywords)

keyword_node = workflow.add_transform_node(
    extract_keywords,
    "ExtractKeywords"
)
```

**Behavior:**
- Receives any input type
- Applies transformation function
- Returns transformed output

**Use Cases:**
- Data preprocessing
- Format conversion
- Text manipulation
- Filtering and extraction

### 5. CONDITION Node
Implements conditional branching based on a predicate.

```python
# Simple condition
condition_node = workflow.add_condition_node(
    lambda x: len(x) < 20,
    "IsShort"
)

# Set branches
condition_node.true_branch = node_a.node_id
condition_node.false_branch = node_b.node_id
```

**Behavior:**
- Evaluates condition function
- Routes to `true_branch` if condition is True
- Routes to `false_branch` if condition is False
- Condition function must return boolean

**Use Cases:**
- Dynamic routing
- Input validation
- Content filtering
- Business logic

### 6. OUTPUT Node
Terminal node that captures final workflow result.

```python
output_node = workflow.add_output_node("FinalResult")
```

**Behavior:**
- Receives input
- Returns input as final workflow output
- Marks workflow execution as complete

### 7. TOOL Node (Advanced)
Executes a custom function with arguments.

```python
def custom_tool(text: str, suffix: str) -> str:
    return f"{text} {suffix}"

tool_node = workflow.add_tool_node(
    custom_tool,
    {"suffix": "!!!"},  # Arguments
    "AddExclamation"
)
```

**Behavior:**
- Calls function with provided arguments
- First argument is the input data
- Returns function result

## Building Workflows

### Linear Workflows

Simplest pattern: A → B → C → D

```python
workflow = Workflow(name="Linear")

n1 = workflow.add_input_node()
n2 = workflow.add_transform_node(lambda x: x.upper())
n3 = workflow.add_assistant_node(assistant)
n4 = workflow.add_output_node()

workflow.connect(n1.node_id, n2.node_id)
workflow.connect(n2.node_id, n3.node_id)
workflow.connect(n3.node_id, n4.node_id)
```

### Branching Workflows

Conditional paths based on input:

```python
workflow = Workflow(name="Branching")

input_node = workflow.add_input_node()

# Condition: is input long?
condition = workflow.add_condition_node(
    lambda x: len(x) > 50,
    "IsLongInput"
)

# Path A: Detailed processing
detailed_assistant = workflow.add_assistant_node(detailed_bot, "Detailed")

# Path B: Quick processing
quick_assistant = workflow.add_assistant_node(quick_bot, "Quick")

output_node = workflow.add_output_node()

# Connect
workflow.connect(input_node.node_id, condition.node_id)
condition.true_branch = detailed_assistant.node_id
condition.false_branch = quick_assistant.node_id
workflow.connect(detailed_assistant.node_id, output_node.node_id)
workflow.connect(quick_assistant.node_id, output_node.node_id)
```

**Flow Diagram:**
```
                  ┌─→ [Detailed Assistant] ─┐
[INPUT] → [CONDITION]                        → [OUTPUT]
                  └─→ [Quick Assistant] ────┘
```

### Multi-Stage Pipelines

Research → Analysis → Summarization:

```python
workflow = Workflow(name="ResearchPipeline")

# Nodes
input_node = workflow.add_input_node("Topic")
researcher = workflow.add_agent_node(research_agent, "Research")
analyzer = workflow.add_agent_node(analysis_agent, "Analyze")
summarizer = workflow.add_assistant_node(summary_assistant, "Summarize")
output_node = workflow.add_output_node("Report")

# Connect
workflow.connect(input_node.node_id, researcher.node_id)
workflow.connect(researcher.node_id, analyzer.node_id)
workflow.connect(analyzer.node_id, summarizer.node_id)
workflow.connect(summarizer.node_id, output_node.node_id)
```

**Flow Diagram:**
```
[INPUT] → [Research Agent] → [Analysis Agent] → [Summarizer] → [OUTPUT]
```

### Transform Pipelines

Data preprocessing and postprocessing:

```python
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

# Connect
workflow.connect(input_node.node_id, clean_node.node_id)
workflow.connect(clean_node.node_id, prefix_node.node_id)
workflow.connect(prefix_node.node_id, assistant_node.node_id)
workflow.connect(assistant_node.node_id, extract_node.node_id)
workflow.connect(extract_node.node_id, output_node.node_id)
```

## Execution Model

### Sequential Execution

Workflows execute sequentially, one node at a time:

1. Start at INPUT node
2. Execute node
3. Get next node from connections or branches
4. Repeat until OUTPUT node is reached

### Execution Result

```python
result = workflow.execute("input data", verbose=True)

# Result structure
{
    "success": True,
    "output": "final output from OUTPUT node",
    "execution_time": 5.23,  # seconds
    "nodes_executed": ["input_123", "agent_456", "output_789"],
    "node_results": {
        "input_123": NodeResult(...),
        "agent_456": NodeResult(...),
        "output_789": NodeResult(...)
    }
}
```

### Node Results

Each node execution creates a `NodeResult`:

```python
@dataclass
class NodeResult:
    success: bool
    output: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Verbose Mode

Enable verbose mode to see execution details:

```python
result = workflow.execute("input", verbose=True)

# Output:
# → Executing INPUT node: UserInput
#   Completed in 0.001s
# → Executing AGENT node: Researcher
#   Completed in 3.245s
# → Executing OUTPUT node: Result
#   Completed in 0.001s
```

### Error Handling

If a node fails:

```python
result = workflow.execute("input")

if not result["success"]:
    print(f"Workflow failed")
    print(f"Error: {result.get('error')}")

    # Check individual node results
    for node_id, node_result in result["node_results"].items():
        if not node_result.success:
            print(f"Failed at {node_id}: {node_result.error}")
```

### Cycle Detection

Workflows automatically detect cycles:

```python
# This will raise an error
workflow.connect(n1.node_id, n2.node_id)
workflow.connect(n2.node_id, n3.node_id)
workflow.connect(n3.node_id, n1.node_id)  # Cycle!

result = workflow.execute("input")
# ValueError: Cycle detected at node: n1
```

## Conditional Logic

### Basic Conditions

```python
# Numeric threshold
condition = workflow.add_condition_node(
    lambda x: float(x) > 100,
    "IsLarge"
)

# String matching
condition = workflow.add_condition_node(
    lambda x: "urgent" in x.lower(),
    "IsUrgent"
)

# Length check
condition = workflow.add_condition_node(
    lambda x: len(str(x)) < 50,
    "IsShort"
)
```

### Setting Branches

```python
# Create condition node
condition_node = workflow.add_condition_node(predicate, "Check")

# Create branch nodes
true_node = workflow.add_assistant_node(assistant_a, "PathA")
false_node = workflow.add_assistant_node(assistant_b, "PathB")

# Set branches
condition_node.true_branch = true_node.node_id
condition_node.false_branch = false_node.node_id

# Both branches can connect to same output
output_node = workflow.add_output_node()
workflow.connect(true_node.node_id, output_node.node_id)
workflow.connect(false_node.node_id, output_node.node_id)
```

### Complex Conditions

```python
def complex_condition(text: str) -> bool:
    # Multiple checks
    if len(text) < 10:
        return False
    if any(word in text.lower() for word in ["help", "urgent", "asap"]):
        return True
    if text.count("?") > 2:
        return True
    return False

condition_node = workflow.add_condition_node(
    complex_condition,
    "PriorityCheck"
)
```

### Nested Conditions

```python
# First condition
condition1 = workflow.add_condition_node(
    lambda x: len(x) > 50,
    "IsLong"
)

# Nested conditions for each branch
condition2a = workflow.add_condition_node(
    lambda x: "?" in x,
    "IsQuestion"
)

condition2b = workflow.add_condition_node(
    lambda x: "!" in x,
    "IsExclamation"
)

# Set up nested structure
condition1.true_branch = condition2a.node_id
condition1.false_branch = condition2b.node_id

# ... continue building branches
```

## Workflow Visualization

### Text Visualization

```python
print(workflow.visualize())

# Output:
# Workflow: ResearchPipeline
# Nodes: 5
# ─────────────────────────────
# INPUT → input_abc123 (Topic)
# AGENT → agent_def456 (Research)
# AGENT → agent_ghi789 (Analyze)
# ASSISTANT → asst_jkl012 (Summarize)
# OUTPUT → output_mno345 (Report)
#
# Connections:
# input_abc123 → agent_def456
# agent_def456 → agent_ghi789
# agent_ghi789 → asst_jkl012
# asst_jkl012 → output_mno345
```

### Conditional Visualization

```python
# Workflow with conditions
print(workflow.visualize())

# Output shows branches:
# CONDITION → cond_123 (IsLarge)
#   TRUE → agent_456
#   FALSE → agent_789
```

### Debugging with Visualization

```python
# Before execution
print("Workflow Structure:")
print(workflow.visualize())

# After execution
result = workflow.execute("input", verbose=True)

print("\nExecution Path:")
print(f"Nodes executed: {result['nodes_executed']}")

print("\nNode Results:")
for node_id in result['nodes_executed']:
    node_result = result['node_results'][node_id]
    print(f"{node_id}: {node_result.success} ({node_result.execution_time:.3f}s)")
```

## Persistence

### Save Workflow

```python
# Save to file
workflow.save("workflow.json")
```

**Saved Format:**
```json
{
    "name": "ResearchPipeline",
    "nodes": [
        {
            "node_id": "input_123",
            "type": "input",
            "name": "Topic"
        },
        {
            "node_id": "agent_456",
            "type": "agent",
            "name": "Research"
        }
    ],
    "edges": [
        {
            "from": "input_123",
            "to": "agent_456"
        }
    ],
    "execution_count": 5,
    "created_at": "2024-01-15T10:30:00",
    "last_executed": "2024-01-15T11:45:00"
}
```

**Note:** Agents and assistants are **not serialized**. You must recreate them when loading.

### Load Workflow

```python
# Workflows cannot be fully loaded from JSON
# because agents/assistants cannot be serialized

# However, you can use the saved structure as a template
import json

with open("workflow.json", 'r') as f:
    data = json.load(f)

print(f"Workflow: {data['name']}")
print(f"Nodes: {len(data['nodes'])}")
print(f"Executed {data['execution_count']} times")

# Recreate workflow manually using the structure
workflow = Workflow(name=data['name'])
# ... add nodes and connections based on saved data
```

### Versioning Workflows

```python
# Save with version info
workflow.metadata["version"] = "1.0"
workflow.metadata["description"] = "Research pipeline v1"
workflow.save("workflow_v1.json")

# Update and save new version
workflow.metadata["version"] = "1.1"
workflow.save("workflow_v1.1.json")
```

## Best Practices

### 1. Name Nodes Clearly

```python
# Good: Descriptive names
input_node = workflow.add_input_node("UserQuery")
research_node = workflow.add_agent_node(agent, "WebResearch")
summary_node = workflow.add_assistant_node(assistant, "Summarizer")

# Bad: Generic names
n1 = workflow.add_input_node("Node1")
n2 = workflow.add_agent_node(agent, "Node2")
```

### 2. Use Transforms for Data Preprocessing

```python
# Clean input before processing
clean_node = workflow.add_transform_node(
    lambda x: x.strip().lower(),
    "CleanInput"
)

# Format output
format_node = workflow.add_transform_node(
    lambda x: f"Result: {x}",
    "FormatOutput"
)
```

### 3. Add Error Handling in Transforms

```python
def safe_transform(data):
    try:
        return data.upper()
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        return data  # Return original on error

transform_node = workflow.add_transform_node(safe_transform, "SafeUpper")
```

### 4. Use Verbose Mode for Debugging

```python
# During development
result = workflow.execute("input", verbose=True)

# In production
result = workflow.execute("input", verbose=False)
```

### 5. Validate Workflow Structure

```python
# Check connectivity
workflow_viz = workflow.visualize()
print(workflow_viz)

# Ensure all nodes are reachable
if "AGENT → agent_123" not in workflow_viz:
    print("Warning: Agent node not connected!")
```

### 6. Keep Workflows Simple

```python
# Good: 3-5 nodes
INPUT → TRANSFORM → AGENT → OUTPUT

# Be cautious: 10+ nodes
# Consider breaking into sub-workflows
```

### 7. Document Complex Workflows

```python
workflow = Workflow(name="ComplexPipeline")
workflow.metadata["description"] = """
This workflow performs:
1. Input validation
2. Web research
3. Analysis
4. Summary generation
"""

workflow.metadata["author"] = "Team Name"
workflow.metadata["created"] = "2024-01-15"
```

### 8. Test Workflows Incrementally

```python
# Test first part
workflow_part1 = Workflow(name="Part1")
n1 = workflow_part1.add_input_node()
n2 = workflow_part1.add_agent_node(agent)
n3 = workflow_part1.add_output_node()
workflow_part1.connect(n1.node_id, n2.node_id)
workflow_part1.connect(n2.node_id, n3.node_id)

result1 = workflow_part1.execute("test")
assert result1["success"], "Part 1 failed"

# Add more nodes
n4 = workflow_part1.add_transform_node(lambda x: x.upper())
workflow_part1.connect(n2.node_id, n4.node_id)
# ... continue testing
```

## Advanced Patterns

### Pattern 1: Fan-out/Fan-in

Process input through multiple parallel paths (simulated with conditions):

```python
workflow = Workflow(name="FanOut")

input_node = workflow.add_input_node()

# Process through multiple agents
agent1_node = workflow.add_agent_node(agent1, "Perspective1")
agent2_node = workflow.add_agent_node(agent2, "Perspective2")
agent3_node = workflow.add_agent_node(agent3, "Perspective3")

# Combine results (simplified - in practice you'd need custom logic)
combine_node = workflow.add_transform_node(
    lambda x: x,  # Last result wins - customize as needed
    "Combine"
)

output_node = workflow.add_output_node()

# Connect (last agent's output used as final)
workflow.connect(input_node.node_id, agent1_node.node_id)
workflow.connect(input_node.node_id, agent2_node.node_id)  # Note: Current implementation is sequential
workflow.connect(input_node.node_id, agent3_node.node_id)
# ... would need custom merge logic
```

### Pattern 2: Retry Logic

Implement retry with conditions:

```python
def check_success(result: str) -> bool:
    return "success" in result.lower()

# Main processing
agent_node = workflow.add_agent_node(agent, "Processor")

# Check result
condition_node = workflow.add_condition_node(check_success, "CheckSuccess")

# Retry node
retry_node = workflow.add_transform_node(
    lambda x: x + " [RETRY]",
    "AddRetryTag"
)

# Set branches
condition_node.true_branch = output_node.node_id  # Success
condition_node.false_branch = retry_node.node_id  # Retry

# Note: This is simplified - production retry logic would need
# iteration counting and max retries
```

### Pattern 3: Dynamic Routing

Route based on content type:

```python
def detect_type(text: str) -> bool:
    return "code" in text.lower()

type_detector = workflow.add_condition_node(detect_type, "DetectType")

# Specialized assistants
code_assistant = workflow.add_assistant_node(code_bot, "CodeHelper")
text_assistant = workflow.add_assistant_node(text_bot, "TextHelper")

type_detector.true_branch = code_assistant.node_id
type_detector.false_branch = text_assistant.node_id
```

### Pattern 4: Pipeline Stages

ETL-style pipeline:

```python
# Extract
extract_node = workflow.add_transform_node(
    extract_data,
    "Extract"
)

# Transform
transform_node = workflow.add_transform_node(
    clean_data,
    "Transform"
)

# Load/Process
load_node = workflow.add_agent_node(processing_agent, "Load")

# Connect stages
workflow.connect(input_node.node_id, extract_node.node_id)
workflow.connect(extract_node.node_id, transform_node.node_id)
workflow.connect(transform_node.node_id, load_node.node_id)
workflow.connect(load_node.node_id, output_node.node_id)
```

### Pattern 5: Context Accumulation

Use workflow context to accumulate information:

```python
def accumulate_context(text: str, context: Dict) -> str:
    # Access shared context
    previous_results = context.get("results", [])
    previous_results.append(text)
    context["results"] = previous_results
    return text

accumulator = workflow.add_transform_node(
    accumulate_context,
    "Accumulator"
)
```

## API Reference

### Workflow Class

```python
class Workflow:
    def __init__(self, name: str = "Workflow")
```

### Node Creation Methods

#### add_input_node()
```python
def add_input_node(self, name: str = "Input") -> InputNode
```

#### add_output_node()
```python
def add_output_node(self, name: str = "Output") -> OutputNode
```

#### add_agent_node()
```python
def add_agent_node(self, agent: Agent, name: str = "Agent") -> AgentNode
```

#### add_assistant_node()
```python
def add_assistant_node(self, assistant: Assistant, name: str = "Assistant") -> AssistantNode
```

#### add_transform_node()
```python
def add_transform_node(
    self,
    transform_fn: Callable[[Any], Any],
    name: str = "Transform"
) -> TransformNode
```

#### add_condition_node()
```python
def add_condition_node(
    self,
    condition_fn: Callable[[Any], bool],
    name: str = "Condition"
) -> ConditionNode
```

#### add_tool_node()
```python
def add_tool_node(
    self,
    tool_fn: Callable,
    tool_args: Dict[str, Any],
    name: str = "Tool"
) -> ToolNode
```

### Connection Methods

#### connect()
```python
def connect(self, from_node_id: str, to_node_id: str)
```
Connect two nodes.

### Execution Methods

#### execute()
```python
def execute(
    self,
    input_data: Any,
    verbose: bool = False
) -> Dict[str, Any]
```

**Returns:**
```python
{
    "success": bool,
    "output": Any,
    "execution_time": float,
    "nodes_executed": List[str],
    "node_results": Dict[str, NodeResult],
    "error": Optional[str]
}
```

### Utility Methods

#### visualize()
```python
def visualize(self) -> str
```
Generate text-based visualization.

#### save()
```python
def save(self, filepath: str)
```
Save workflow structure to JSON.

#### get_stats()
```python
def get_stats(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "name": str,
    "node_count": int,
    "edge_count": int,
    "execution_count": int,
    "created_at": str,
    "last_executed": Optional[str]
}
```

### NodeResult Class

```python
@dataclass
class NodeResult:
    success: bool
    output: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### NodeType Enum

```python
class NodeType(Enum):
    INPUT = "input"
    AGENT = "agent"
    ASSISTANT = "assistant"
    TRANSFORM = "transform"
    CONDITION = "condition"
    OUTPUT = "output"
    TOOL = "tool"
    RAG = "rag"
```

### ExecutionStatus Enum

```python
class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

## Examples

See `examples/workflow_example.py` for complete examples:
- Simple linear workflow
- Agent workflow with tools
- Transform pipeline
- Conditional branching
- Multi-agent collaboration
- Workflow persistence

## Limitations

Current limitations:
1. **Sequential Only**: No true parallel execution
2. **No Loops**: Cycles are prevented
3. **No Sub-workflows**: Cannot nest workflows
4. **No Serialization**: Agents/assistants cannot be saved to JSON
5. **Simple Merge**: No built-in fan-in/merge logic

Future enhancements may address these limitations.

## Related Documentation

- [Agent Framework](AGENT.md) - For autonomous agents
- [Assistant Framework](ASSISTANT.md) - For simple assistants
- [Main README](README.md) - Getting started guide
