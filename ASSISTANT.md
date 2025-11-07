# Assistant Framework

The Assistant framework provides a simple, context-aware LLM wrapper for conversational use cases. Unlike Agents, Assistants do not automatically execute tools - they focus on maintaining conversation history and can optionally integrate with RAG (Retrieval Augmented Generation) for document-based context.

## Table of Contents

- [Overview](#overview)
- [When to Use Assistants vs Agents](#when-to-use-assistants-vs-agents)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [RAG Support](#rag-support)
- [Conversation Management](#conversation-management)
- [Persistence](#persistence)
- [Multiple Assistants](#multiple-assistants)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

Assistants are designed for:
- Simple conversational interactions
- Document Q&A with RAG
- Context-aware responses
- Multi-turn conversations
- Stateful chat applications

**Key Features:**
- Automatic conversation history management
- Optional RAG for document-based context
- Conversation persistence (save/load)
- Configurable history limits
- Multiple personalities
- No automatic tool execution (unlike Agents)

## When to Use Assistants vs Agents

### Use Assistants when:
- You need simple conversational interactions
- You want to maintain conversation history
- You need document Q&A with RAG
- You don't need automatic tool execution
- You want lower overhead and simpler API
- You're building a chatbot or Q&A system

### Use Agents when:
- You need autonomous tool execution
- Complex task completion is required
- You want the system to decide which tools to use
- Multi-step problem solving is needed
- You need thinking mode for deep research

**Quick Comparison:**

| Feature | Assistant | Agent |
|---------|-----------|-------|
| Conversation History | ✅ Yes | ✅ Yes |
| RAG Support | ✅ Yes | ❌ No |
| Auto Tool Execution | ❌ No | ✅ Yes |
| Thinking Mode | ❌ No | ✅ Yes |
| Complexity | Low | High |
| Use Case | Chat, Q&A | Task Completion |

## Basic Usage

### Simple Chat

```python
from fourier import Fourier
from assistant import Assistant, AssistantConfig

# Create client
client = Fourier(api_key="your-api-key", provider="groq")

# Create assistant
assistant = Assistant(
    client=client,
    name="ChatBot",
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        temperature=0.7,
        max_history=10,
        system_prompt="You are a friendly and helpful assistant."
    )
)

# Have a conversation
response1 = assistant.chat("My name is Alice")
print(response1["output"])

response2 = assistant.chat("What's my name?")
print(response2["output"])  # Will remember "Alice"
```

### Response Format

The `chat()` method returns a dictionary:

```python
{
    "output": "Assistant's response text",
    "message_count": 4,  # Messages in history
    "success": True,
    "tokens": {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200
    },
    "response": {...}  # Full API response
}
```

## Configuration

### AssistantConfig

```python
@dataclass
class AssistantConfig:
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    max_history: int = 50
    enable_rag: bool = False
    rag_top_k: int = 3
```

**Parameters:**

- **temperature** (float, default: 0.7): Controls randomness (0.0-2.0)
  - Lower = more focused and deterministic
  - Higher = more creative and random

- **max_tokens** (int, optional): Maximum tokens in response
  - None = provider default
  - Set to control response length

- **system_prompt** (str, optional): System prompt for the assistant
  - None = uses default prompt
  - Customize to change personality/behavior

- **max_history** (int, default: 50): Maximum messages to keep
  - Automatically trims older messages
  - Includes both user and assistant messages

- **enable_rag** (bool, default: False): Enable RAG support
  - False = simple conversation
  - True = document-based context

- **rag_top_k** (int, default: 3): Number of documents to retrieve
  - Only used when enable_rag=True
  - Higher = more context, but more tokens

### Configuration Examples

**Conservative (Precise Answers):**
```python
config = AssistantConfig(
    temperature=0.3,
    max_tokens=500,
    max_history=10
)
```

**Creative (Open-ended Conversations):**
```python
config = AssistantConfig(
    temperature=0.9,
    max_tokens=1000,
    max_history=50
)
```

**Document Q&A:**
```python
config = AssistantConfig(
    temperature=0.5,
    enable_rag=True,
    rag_top_k=5,
    system_prompt="Answer questions based on the provided documents."
)
```

## RAG Support

RAG (Retrieval Augmented Generation) allows assistants to answer questions based on custom documents.

### Basic RAG Usage

```python
# Create assistant with RAG enabled
assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        enable_rag=True,
        rag_top_k=3
    )
)

# Add documents
documents = [
    {
        "content": "FourierSDK is a Python SDK for LLM providers.",
        "metadata": {"source": "docs.txt", "section": "overview"}
    },
    {
        "content": "The Agent class allows autonomous tool execution.",
        "metadata": {"source": "agent_docs.txt"}
    }
]

assistant.add_documents(documents)

# Ask questions about documents
response = assistant.chat("What is FourierSDK?")
print(response["output"])
# Uses document content to answer
```

### Document Format

Each document should have:
- **content** (required): The document text
- **metadata** (optional): Dictionary with additional info
  - Common fields: source, section, page, date, etc.

### How RAG Works

1. **Query Analysis**: Extracts last user message as query
2. **Document Retrieval**: Finds relevant documents using keyword matching
3. **Scoring**:
   - Counts matching words between query and documents
   - Bonus points for exact phrase matches
4. **Context Building**: Formats top-k documents into context
5. **LLM Request**: Appends context to system prompt
6. **Response**: LLM uses document context to answer

### RAG Methods

```python
# Add documents
assistant.add_documents([
    {"content": "...", "metadata": {...}},
    {"content": "...", "metadata": {...}}
])

# Clear all documents
assistant.clear_documents()

# Check document count
stats = assistant.get_stats()
print(f"Documents: {stats['rag_documents']}")
```

### RAG Best Practices

1. **Document Quality**: Keep documents focused and concise
2. **Metadata**: Include useful metadata for debugging
3. **Top-K Tuning**: Adjust `rag_top_k` based on:
   - More documents = more context, but more tokens
   - Fewer documents = faster, cheaper, but less context
4. **Query Clarity**: Encourage users to ask specific questions
5. **Document Count**: Works well with 10-1000 documents
   - For larger collections, consider external vector DB

### Advanced RAG Example

```python
# Load documents from files
import os

def load_documents_from_directory(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        "path": filepath
                    }
                })
    return documents

# Create RAG assistant
assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        enable_rag=True,
        rag_top_k=5,
        system_prompt="You are a helpful assistant. Use the provided context to answer questions accurately. If you're not sure based on the context, say so."
    )
)

# Load and add documents
docs = load_documents_from_directory("./knowledge_base")
assistant.add_documents(docs)

print(f"Loaded {len(docs)} documents")

# Interactive Q&A
while True:
    question = input("Question: ")
    if question.lower() in ['quit', 'exit']:
        break

    response = assistant.chat(question)
    print(f"Answer: {response['output']}\n")
```

## Conversation Management

### History Management

```python
# Get conversation history
history = assistant.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")

# Clear history
assistant.clear_history()

# Reset history during chat
response = assistant.chat("New topic", reset_history=True)
```

### History Format

```python
[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me about Python"},
    {"role": "assistant", "content": "Python is..."}
]
```

### Automatic History Trimming

When history exceeds `max_history`:
- Oldest messages are removed
- Most recent messages are kept
- Trimming is automatic

```python
config = AssistantConfig(max_history=10)  # Keep only 10 messages
assistant = Assistant(client=client, config=config)

# After 12 messages, oldest 2 are automatically removed
```

### System Prompt Updates

```python
# Update system prompt at any time
assistant.update_system_prompt(
    "You are now a Python expert. Provide code examples."
)

# Next responses will use new prompt
response = assistant.chat("How do I read a file?")
```

## Persistence

Save and load conversations for continuity across sessions.

### Save Conversation

```python
# Save to file
assistant.save_conversation("conversation.json")
```

**Saved Format:**
```json
{
    "assistant_name": "ChatBot",
    "model": "mixtral-8x7b-32768",
    "created_at": "2024-01-15T10:30:00",
    "total_messages": 8,
    "conversation": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
}
```

### Load Conversation

```python
# Create new assistant
new_assistant = Assistant(
    client=client,
    model="mixtral-8x7b-32768"
)

# Load previous conversation
new_assistant.load_conversation("conversation.json")

# Continue conversation
response = new_assistant.chat("What were we talking about?")
```

### Full Persistence Example

```python
def chat_session(session_file: str):
    """Resume or start a chat session."""
    client = Fourier(api_key=os.getenv("GROQ_API_KEY"), provider="groq")

    assistant = Assistant(
        client=client,
        name="PersistentBot",
        model="mixtral-8x7b-32768"
    )

    # Load previous session if exists
    if os.path.exists(session_file):
        assistant.load_conversation(session_file)
        print("Resumed previous session")
    else:
        print("Started new session")

    # Chat loop
    try:
        while True:
            message = input("You: ")
            if message.lower() in ['quit', 'exit']:
                break

            response = assistant.chat(message)
            print(f"Bot: {response['output']}")
    finally:
        # Always save on exit
        assistant.save_conversation(session_file)
        print(f"Session saved to {session_file}")

# Usage
chat_session("my_session.json")
```

## Multiple Assistants

Create multiple assistants with different personalities or purposes.

### Different Personalities

```python
# Formal assistant
formal = Assistant(
    client=client,
    name="FormalBot",
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="You are a formal, professional assistant. Use formal language.",
        temperature=0.5
    )
)

# Casual assistant
casual = Assistant(
    client=client,
    name="CasualBot",
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="You are a casual, friendly assistant. Be conversational.",
        temperature=0.8
    )
)

# Creative assistant
creative = Assistant(
    client=client,
    name="CreativeBot",
    model="mixtral-8x7b-32768",
    config=AssistantConfig(
        system_prompt="You are creative and imaginative. Use metaphors.",
        temperature=0.9
    )
)

# Ask same question to all
question = "Explain what an API is"
print("Formal:", formal.chat(question)["output"])
print("Casual:", casual.chat(question)["output"])
print("Creative:", creative.chat(question)["output"])
```

### Specialized Assistants

```python
# Code helper
code_assistant = Assistant(
    client=client,
    name="CodeHelper",
    config=AssistantConfig(
        system_prompt="You are a Python expert. Provide code examples and explanations.",
        temperature=0.3
    )
)

# Writing assistant
writing_assistant = Assistant(
    client=client,
    name="Writer",
    config=AssistantConfig(
        system_prompt="You are a writing assistant. Help with grammar, style, and clarity.",
        temperature=0.7
    )
)

# Research assistant with RAG
research_assistant = Assistant(
    client=client,
    name="Researcher",
    config=AssistantConfig(
        enable_rag=True,
        rag_top_k=5,
        system_prompt="You are a research assistant. Cite sources when answering."
    )
)
```

## Best Practices

### 1. Choose Appropriate Temperature

```python
# Factual tasks (low temperature)
config = AssistantConfig(temperature=0.3)

# Creative tasks (high temperature)
config = AssistantConfig(temperature=0.9)

# Balanced (default)
config = AssistantConfig(temperature=0.7)
```

### 2. Manage History Effectively

```python
# Short conversations
config = AssistantConfig(max_history=10)

# Long conversations
config = AssistantConfig(max_history=50)

# Reset when changing topics
response = assistant.chat("New topic", reset_history=True)
```

### 3. Use Clear System Prompts

```python
# Good: Specific and clear
system_prompt = "You are a Python tutor. Explain concepts simply and provide code examples. Focus on best practices."

# Bad: Vague
system_prompt = "Help with programming"
```

### 4. Handle Errors Gracefully

```python
response = assistant.chat(message)
if response["success"]:
    print(response["output"])
else:
    print(f"Error: {response.get('error', 'Unknown error')}")
    # Fallback logic here
```

### 5. Monitor Token Usage

```python
response = assistant.chat(message)
tokens = response.get("tokens", {})
print(f"Tokens used: {tokens.get('total_tokens', 0)}")

# Adjust max_history if using too many tokens
if tokens.get('total_tokens', 0) > 5000:
    assistant.config.max_history = 20
```

### 6. Save Important Conversations

```python
# Save after important exchanges
if important_conversation:
    assistant.save_conversation(f"important_{datetime.now().isoformat()}.json")
```

### 7. Use RAG for Factual Accuracy

```python
# Enable RAG for domain-specific knowledge
config = AssistantConfig(
    enable_rag=True,
    temperature=0.5,  # Lower temperature with RAG
    system_prompt="Answer based on provided documents. Cite sources."
)
```

### 8. Test Different Configurations

```python
# A/B test different prompts
configs = [
    AssistantConfig(system_prompt="Prompt A"),
    AssistantConfig(system_prompt="Prompt B")
]

for i, config in enumerate(configs):
    assistant = Assistant(client=client, config=config)
    response = assistant.chat(test_query)
    print(f"Config {i}: {response['output']}")
```

## API Reference

### Assistant Class

```python
class Assistant:
    def __init__(
        self,
        client: Fourier,
        name: str = "Assistant",
        model: str = "mixtral-8x7b-32768",
        config: Optional[AssistantConfig] = None,
        **kwargs
    )
```

### Methods

#### chat()
```python
def chat(
    self,
    message: str,
    reset_history: bool = False,
    **override_kwargs
) -> Dict[str, Any]
```
Send a message to the assistant.

**Parameters:**
- `message` (str): User message
- `reset_history` (bool): Clear history before this message
- `**override_kwargs`: Override default client.chat() parameters

**Returns:**
- Dictionary with `output`, `success`, `message_count`, `tokens`

#### add_documents()
```python
def add_documents(self, documents: List[Dict[str, Any]])
```
Add documents for RAG context.

**Parameters:**
- `documents`: List of dicts with `content` and optional `metadata`

#### clear_documents()
```python
def clear_documents()
```
Remove all RAG documents.

#### get_conversation_history()
```python
def get_conversation_history() -> List[Dict[str, str]]
```
Get conversation history.

**Returns:**
- List of message dictionaries

#### clear_history()
```python
def clear_history()
```
Clear conversation history.

#### save_conversation()
```python
def save_conversation(self, filepath: str)
```
Save conversation to JSON file.

#### load_conversation()
```python
def load_conversation(self, filepath: str)
```
Load conversation from JSON file.

#### get_stats()
```python
def get_stats() -> Dict[str, Any]
```
Get assistant statistics.

**Returns:**
```python
{
    "name": "AssistantName",
    "model": "mixtral-8x7b-32768",
    "created_at": "2024-01-15T10:30:00",
    "total_messages": 24,
    "current_history_length": 10,
    "rag_enabled": True,
    "rag_documents": 15,
    "max_history": 50
}
```

#### update_system_prompt()
```python
def update_system_prompt(self, system_prompt: str)
```
Update system prompt.

### AssistantConfig Class

```python
@dataclass
class AssistantConfig:
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    max_history: int = 50
    enable_rag: bool = False
    rag_top_k: int = 3
```

Automatic validation in `__post_init__`:
- Temperature clamped to 0.0-2.0
- max_history >= 1

## Examples

See `examples/assistant_example.py` for complete examples:
- Basic conversation
- RAG with document Q&A
- Conversation persistence
- Multiple personalities

## Related Documentation

- [Agent Framework](AGENT.md) - For autonomous tool execution
- [Workflow System](WORKFLOW.md) - For orchestrating assistants and agents
- [Main README](README.md) - Getting started guide
