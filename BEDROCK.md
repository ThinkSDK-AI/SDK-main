# AWS Bedrock Provider Documentation

Complete guide to using AWS Bedrock with the Fourier SDK.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Authentication Methods](#authentication-methods)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Supported Models](#supported-models)
7. [Cross-Region and Global Inference](#cross-region-and-global-inference)
8. [Tool Calling / Function Calling](#tool-calling--function-calling)
9. [Using with Agents](#using-with-agents)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

AWS Bedrock is Amazon's fully managed service for foundation models from leading AI companies. The Fourier SDK provides comprehensive support for Bedrock with:

- **Multiple authentication methods** (IAM credentials, AWS profiles, instance roles)
- **All Bedrock foundation models** (Claude, Llama, Mistral, Titan, Cohere, AI21)
- **Cross-region inference** for high availability
- **Global inference profiles** for worldwide optimization
- **Tool calling** with Bedrock Converse API
- **Direct model invocation** with InvokeModel API
- **Seamless integration** with Agent framework

---

## Setup and Installation

### 1. Install Dependencies

```bash
pip install fourier-sdk boto3>=1.28.0
```

### 2. Configure AWS Credentials

You have multiple options for authentication:

#### Option 1: Environment Variables

Create a `.env` file:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

#### Option 2: AWS Profile

Configure in `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
region = us-east-1
```

#### Option 3: IAM Role

If running on EC2, ECS, or Lambda, use IAM roles (no credentials needed).

### 3. Enable Bedrock Model Access

1. Go to AWS Console → Bedrock → Model Access
2. Request access to the models you want to use
3. Wait for approval (usually instant for most models)

---

## Authentication Methods

### Method 1: IAM Credentials (Access Key + Secret Key)

```python
from fourier import Fourier

client = Fourier(
    api_key=None,  # Not needed for IAM
    provider="bedrock",
    access_key="your_access_key",
    secret_key="your_secret_key",
    region="us-east-1"
)
```

### Method 2: AWS Profile

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    profile_name="default",  # or your profile name
    region="us-east-1"
)
```

### Method 3: Default Credential Chain

```python
# Uses environment variables, credential files, or IAM role
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)
```

### Method 4: Temporary Credentials (STS)

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    access_key="temporary_access_key",
    secret_key="temporary_secret_key",
    session_token="session_token",
    region="us-east-1"
)
```

### Method 5: API Key (Custom Endpoints)

For custom Bedrock-compatible endpoints:

```python
client = Fourier(
    api_key="your_api_key",
    provider="bedrock",
    endpoint_url="https://custom-bedrock-endpoint.com",
    region="us-east-1"
)
```

---

## Basic Usage

### Simple Chat Completion

```python
from fourier import Fourier

client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)

response = client.chat(
    model="claude-3-5-sonnet",
    messages=[
        {"role": "user", "content": "What is AWS Bedrock?"}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response['response']['content'])
```

### Using Full Model IDs

```python
response = client.chat(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Full model ID
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)
```

### Multi-Turn Conversation

```python
messages = [
    {"role": "user", "content": "What is machine learning?"}
]

# First turn
response = client.chat(
    model="claude-3-5-sonnet",
    messages=messages,
    max_tokens=200
)

# Add response to conversation
messages.append({
    "role": "assistant",
    "content": response['response']['content']
})

# Second turn
messages.append({
    "role": "user",
    "content": "Can you give me an example?"
})

response = client.chat(
    model="claude-3-5-sonnet",
    messages=messages,
    max_tokens=300
)
```

### System Prompts

```python
response = client.chat(
    model="claude-3-5-sonnet",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful Python programming assistant."
        },
        {
            "role": "user",
            "content": "Write a function to reverse a string."
        }
    ],
    max_tokens=500
)
```

---

## Advanced Features

### Cross-Region Inference

Enable automatic failover across multiple AWS regions:

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_cross_region=True  # Enable cross-region inference
)

response = client.chat(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

**Benefits:**
- High availability
- Automatic failover
- Load balancing across regions

### Global Inference Profiles

Use inference profiles for optimal global routing:

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_global_inference=True
)
```

### Specific Inference Profile

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    inference_profile_id="your-profile-id"
)
```

### Multi-Region Setup

```python
# Primary region
primary_client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)

# Backup region
backup_client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-west-2"
)

# Try primary, fall back to backup
try:
    response = primary_client.chat(...)
except Exception:
    response = backup_client.chat(...)
```

---

## Supported Models

### Anthropic Claude Models

| Friendly Name | Model ID | Description |
|--------------|----------|-------------|
| `claude-3-5-sonnet` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Most capable, latest |
| `claude-3-5-haiku` | `anthropic.claude-3-5-haiku-20241022-v1:0` | Fast and efficient |
| `claude-3-opus` | `anthropic.claude-3-opus-20240229-v1:0` | Highly capable |
| `claude-3-sonnet` | `anthropic.claude-3-sonnet-20240229-v1:0` | Balanced |
| `claude-3-haiku` | `anthropic.claude-3-haiku-20240307-v1:0` | Fast |
| `claude-instant` | `anthropic.claude-instant-v1` | Instant responses |
| `claude-2.1` | `anthropic.claude-v2:1` | Previous generation |

### Meta Llama Models

| Friendly Name | Model ID |
|--------------|----------|
| `llama3-2-1b` | `us.meta.llama3-2-1b-instruct-v1:0` |
| `llama3-2-3b` | `us.meta.llama3-2-3b-instruct-v1:0` |
| `llama3-2-11b` | `us.meta.llama3-2-11b-instruct-v1:0` |
| `llama3-2-90b` | `us.meta.llama3-2-90b-instruct-v1:0` |
| `llama3-1-8b` | `meta.llama3-1-8b-instruct-v1:0` |
| `llama3-1-70b` | `meta.llama3-1-70b-instruct-v1:0` |
| `llama3-1-405b` | `meta.llama3-1-405b-instruct-v1:0` |

### Amazon Titan Models

| Friendly Name | Model ID | Type |
|--------------|----------|------|
| `titan-text-express` | `amazon.titan-text-express-v1` | Text generation |
| `titan-text-lite` | `amazon.titan-text-lite-v1` | Lightweight text |
| `titan-text-premier` | `amazon.titan-text-premier-v1:0` | Premier text |
| `titan-embed-text` | `amazon.titan-embed-text-v1` | Embeddings |
| `titan-image-generator` | `amazon.titan-image-generator-v1` | Image generation |

### Mistral AI Models

| Friendly Name | Model ID |
|--------------|----------|
| `mistral-7b` | `mistral.mistral-7b-instruct-v0:2` |
| `mixtral-8x7b` | `mistral.mixtral-8x7b-instruct-v0:1` |
| `mistral-large` | `mistral.mistral-large-2402-v1:0` |
| `mistral-large-2407` | `mistral.mistral-large-2407-v1:0` |

### Cohere Models

| Friendly Name | Model ID | Type |
|--------------|----------|------|
| `cohere-command-r` | `cohere.command-r-v1:0` | Chat |
| `cohere-command-r-plus` | `cohere.command-r-plus-v1:0` | Chat (enhanced) |
| `cohere-embed-english` | `cohere.embed-english-v3` | Embeddings |
| `cohere-embed-multilingual` | `cohere.embed-multilingual-v3` | Embeddings |

### AI21 Labs Models

| Friendly Name | Model ID |
|--------------|----------|
| `jamba-instruct` | `ai21.jamba-instruct-v1:0` |
| `j2-ultra` | `ai21.j2-ultra-v1` |
| `j2-mid` | `ai21.j2-mid-v1` |

### List Available Models Programmatically

```python
from providers.bedrock import BedrockProvider

provider = BedrockProvider(region="us-east-1")

# List all models
all_models = provider.list_foundation_models()
for model in all_models:
    print(f"{model['modelId']} - {model['providerName']}")

# Filter by provider
claude_models = provider.list_foundation_models(by_provider="Anthropic")

# Filter by modality
text_models = provider.list_foundation_models(by_output_modality="TEXT")
```

---

## Cross-Region and Global Inference

### What is Cross-Region Inference?

Cross-region inference automatically routes requests across multiple AWS regions for:
- **High availability**: Automatic failover if one region is unavailable
- **Load balancing**: Distribute traffic across regions
- **Improved latency**: Route to nearest available region

### Enabling Cross-Region Inference

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_cross_region=True
)
```

### Global Inference Profiles

Inference profiles optimize routing globally:

```python
# Enable global inference
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_global_inference=True
)

# Or use a specific profile
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    inference_profile_id="arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile"
)
```

### List Available Inference Profiles

```python
from providers.bedrock import BedrockProvider

provider = BedrockProvider(region="us-east-1")
profiles = provider.get_inference_profiles()

for profile in profiles:
    print(f"Profile: {profile['inferenceProfileId']}")
    print(f"Status: {profile['status']}")
    print(f"Type: {profile['type']}")
```

---

## Tool Calling / Function Calling

Bedrock supports tool calling via the Converse API (for compatible models like Claude).

### Basic Tool Calling

```python
from fourier import Fourier

client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)

# Define a tool
calculator_tool = client.create_tool(
    name="calculator",
    description="Performs arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }
)

# Use the tool
response = client.chat(
    model="claude-3-5-sonnet",
    messages=[
        {"role": "user", "content": "What is 25 * 4?"}
    ],
    tools=[calculator_tool],
    max_tokens=500
)

# Check if tool was called
if response['metadata'].get('response_type') == 'tool_call':
    print(f"Tool used: {response['response']['tool_used']}")
    print(f"Parameters: {response['response']['tool_parameters']}")
```

### Tool Execution with Agents

See the [Using with Agents](#using-with-agents) section for automatic tool execution.

---

## Using with Agents

The Agent framework provides autonomous tool execution with Bedrock.

### Basic Agent

```python
from fourier import Fourier
from agent import Agent, AgentConfig

client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"
)

config = AgentConfig(
    name="BedrockAgent",
    max_iterations=5,
    verbose=True
)

agent = Agent(
    client=client,
    model="claude-3-5-sonnet",
    config=config
)

# Define and register a tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: 72°F, Sunny"

agent.register_tool(
    name="get_weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    },
    function=get_weather
)

# Run the agent
response = agent.run("What's the weather in Seattle?")
print(response['response'])
```

### Multi-Tool Agent

```python
# Register multiple tools
agent.register_tool(name="calculator", ...)
agent.register_tool(name="search_database", ...)
agent.register_tool(name="send_email", ...)

# Agent will autonomously choose and execute tools
response = agent.run(
    "Calculate 150 * 3, search for 'laptop' in the database, "
    "and email the results to user@example.com"
)
```

### Agent with Cross-Region Bedrock

```python
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_cross_region=True  # High availability
)

agent = Agent(client=client, model="claude-3-5-sonnet", config=config)
```

For complete agent examples, see `examples/bedrock_agent_example.py`.

---

## Best Practices

### 1. Use Appropriate Models

- **Claude 3.5 Sonnet**: Complex reasoning, coding, analysis
- **Claude 3.5 Haiku**: Fast responses, simple tasks
- **Llama 3.1**: Open source, cost-effective
- **Mistral**: Multilingual, specialized tasks

### 2. Implement Error Handling

```python
try:
    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )
except Exception as e:
    print(f"Error: {e}")
    # Implement retry logic or fallback
```

### 3. Use Cross-Region for Production

```python
# High availability setup
client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1",
    use_cross_region=True
)
```

### 4. Optimize Token Usage

```python
# Set appropriate max_tokens
response = client.chat(
    model="claude-3-haiku",  # More cost-effective
    messages=messages,
    max_tokens=500,  # Only what you need
    temperature=0.3  # Lower for factual responses
)
```

### 5. Use System Prompts Effectively

```python
messages = [
    {
        "role": "system",
        "content": "You are a concise assistant. Respond in 2-3 sentences."
    },
    {"role": "user", "content": query}
]
```

### 6. Secure Credentials

```python
# Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

client = Fourier(
    api_key=None,
    provider="bedrock",
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION")
)
```

### 7. Monitor Usage

```python
response = client.chat(...)

# Track token usage
print(f"Prompt tokens: {response['usage']['prompt_tokens']}")
print(f"Completion tokens: {response['usage']['completion_tokens']}")
print(f"Total tokens: {response['usage']['total_tokens']}")
```

---

## Troubleshooting

### Issue: "No AWS credentials found"

**Solution:**
```python
# Explicitly provide credentials
client = Fourier(
    api_key=None,
    provider="bedrock",
    access_key="your_access_key",
    secret_key="your_secret_key",
    region="us-east-1"
)
```

### Issue: "Model not found" or "Access denied"

**Solution:**
1. Go to AWS Console → Bedrock → Model Access
2. Request access to the model
3. Wait for approval
4. Verify the model ID is correct

### Issue: boto3 not installed

**Solution:**
```bash
pip install boto3>=1.28.0
```

### Issue: "Region not supported"

**Solution:**
```python
# Use a supported region
supported_regions = [
    "us-east-1", "us-west-2", "eu-west-1",
    "eu-central-1", "ap-northeast-1", "ap-southeast-1"
]

client = Fourier(
    api_key=None,
    provider="bedrock",
    region="us-east-1"  # Use supported region
)
```

### Issue: Token limit exceeded

**Solution:**
```python
# Reduce max_tokens or truncate conversation history
response = client.chat(
    model="claude-3-5-sonnet",
    messages=messages[-10:],  # Keep only last 10 messages
    max_tokens=1000  # Reduce token limit
)
```

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('providers.bedrock')
logger.setLevel(logging.DEBUG)

# Now you'll see detailed Bedrock API calls
```

---

## Examples

All examples are in the `examples/` directory:

- **bedrock_basic_example.py** - Authentication methods, basic chat, models
- **bedrock_advanced_example.py** - Cross-region, global inference, tool calling
- **bedrock_agent_example.py** - Agent framework with Bedrock

Run examples:

```bash
python examples/bedrock_basic_example.py
python examples/bedrock_advanced_example.py
python examples/bedrock_agent_example.py
```

---

## API Reference

### BedrockProvider Class

```python
from providers.bedrock import BedrockProvider

provider = BedrockProvider(
    api_key=None,
    access_key="...",
    secret_key="...",
    region="us-east-1",
    use_cross_region=False,
    use_global_inference=False,
    inference_profile_id=None,
    profile_name=None,
    session_token=None,
    endpoint_url=None
)
```

### Methods

- `chat_completion(request)` - Create chat completion using Converse API
- `invoke_model(model_id, body)` - Direct model invocation
- `list_foundation_models()` - List available models
- `get_inference_profiles()` - List inference profiles
- `prepare_request(request)` - Prepare request payload
- `process_response(response)` - Process API response

---

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
- [IAM Permissions for Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html)

---

## Support

For issues or questions:

- GitHub Issues: [SDK Issues](https://github.com/ThinkSDK-AI/SDK-main/issues)
- Documentation: [Main README](README.md)
- Examples: See `examples/` directory

---

**Last Updated:** 2025-01-08
