"""
AWS Bedrock with Agent Framework

This example demonstrates how to use AWS Bedrock with the autonomous
Agent framework for complex tasks with tool execution.
"""

import os
from dotenv import load_dotenv
from fourier import Fourier
from agent import Agent, AgentConfig
import json

# Load environment variables
load_dotenv()


def get_weather(location: str) -> str:
    """
    Simulated weather API - returns mock weather data.

    Args:
        location: City name

    Returns:
        Weather information as JSON string
    """
    # Mock weather data
    weather_data = {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 45,
        "wind_speed": 10
    }
    return json.dumps(weather_data)


def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }

    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"

    return operations[operation](a, b)


def search_database(query: str) -> str:
    """
    Simulated database search.

    Args:
        query: Search query

    Returns:
        Search results as JSON string
    """
    # Mock database results
    mock_db = {
        "products": [
            {"id": 1, "name": "Laptop", "price": 1200},
            {"id": 2, "name": "Mouse", "price": 25},
            {"id": 3, "name": "Keyboard", "price": 80},
        ]
    }

    # Simple search logic
    results = [
        item for item in mock_db["products"]
        if query.lower() in item["name"].lower()
    ]

    return json.dumps({"results": results, "count": len(results)})


def example_1_basic_agent():
    """Example 1: Basic agent with Bedrock"""
    print("=" * 60)
    print("Example 1: Basic Agent with Bedrock")
    print("=" * 60)

    # Initialize Fourier client with Bedrock
    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    # Create an agent
    config = AgentConfig(
        name="BedrockAssistant",
        description="A helpful assistant powered by AWS Bedrock",
        max_iterations=5,
        verbose=True
    )

    agent = Agent(
        client=client,
        model="claude-3-5-sonnet",
        config=config
    )

    # Register a tool
    agent.register_tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        },
        function=get_weather
    )

    # Run the agent
    query = "What's the weather like in San Francisco?"
    print(f"\nQuery: {query}")

    response = agent.run(query)

    print(f"\nFinal Response: {response['response']}")
    print(f"Iterations: {response['iterations']}")
    print(f"Tools used: {len(response.get('tool_calls', []))}")
    print()


def example_2_multi_tool_agent():
    """Example 2: Agent with multiple tools"""
    print("=" * 60)
    print("Example 2: Multi-Tool Agent")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    config = AgentConfig(
        name="MultiToolAgent",
        description="Agent with multiple specialized tools",
        max_iterations=10,
        verbose=True
    )

    agent = Agent(
        client=client,
        model="claude-3-5-sonnet",
        config=config
    )

    # Register multiple tools
    agent.register_tool(
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
        },
        function=calculate
    )

    agent.register_tool(
        name="get_weather",
        description="Get weather information",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        },
        function=get_weather
    )

    agent.register_tool(
        name="search_database",
        description="Search product database",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        },
        function=search_database
    )

    # Complex query requiring multiple tools
    query = (
        "What's the weather in New York? Also, calculate 150 * 3.5. "
        "And search for laptops in the database."
    )
    print(f"\nQuery: {query}")

    response = agent.run(query)

    print(f"\nFinal Response: {response['response']}")
    print(f"\nTools executed:")
    for i, call in enumerate(response.get('tool_calls', []), 1):
        print(f"{i}. {call.get('tool_name', 'Unknown')} - {call.get('status', 'Unknown')}")
    print()


def example_3_thinking_mode():
    """Example 3: Agent with thinking mode (deep research)"""
    print("=" * 60)
    print("Example 3: Thinking Mode with Bedrock")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    config = AgentConfig(
        name="ResearchAgent",
        description="Research agent with deep thinking capabilities",
        max_iterations=5,
        verbose=True,
        thinking_enabled=True  # Enable thinking mode
    )

    agent = Agent(
        client=client,
        model="claude-3-5-sonnet",
        config=config
    )

    # Research query
    query = "What are the latest developments in quantum computing as of 2024?"
    print(f"\nQuery: {query}")

    response = agent.run(query)

    print(f"\nResponse with research: {response['response'][:500]}...")
    print(f"\nThinking mode performed web research and synthesis.")
    print()


def example_4_bedrock_with_cross_region():
    """Example 4: Agent with cross-region Bedrock"""
    print("=" * 60)
    print("Example 4: Cross-Region Agent")
    print("=" * 60)

    # Initialize with cross-region inference
    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1",
        use_cross_region=True
    )

    config = AgentConfig(
        name="CrossRegionAgent",
        description="High-availability agent with cross-region support",
        max_iterations=3,
        verbose=True
    )

    agent = Agent(
        client=client,
        model="claude-3-5-sonnet",
        config=config
    )

    query = "Explain the benefits of cross-region deployment."
    print(f"\nQuery: {query}")

    response = agent.run(query)

    print(f"\nResponse: {response['response']}")
    print("\nThis agent uses cross-region inference for high availability.")
    print()


def example_5_agent_memory():
    """Example 5: Agent with conversation memory"""
    print("=" * 60)
    print("Example 5: Agent with Memory")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    config = AgentConfig(
        name="MemoryAgent",
        description="Agent that maintains conversation context",
        max_iterations=5,
        verbose=False  # Less verbose for cleaner output
    )

    agent = Agent(
        client=client,
        model="claude-3-5-sonnet",
        config=config
    )

    # First interaction
    query1 = "My name is Alice and I like machine learning."
    print(f"\nQuery 1: {query1}")
    response1 = agent.run(query1)
    print(f"Response 1: {response1['response']}")

    # Second interaction - agent should remember
    query2 = "What's my name and what do I like?"
    print(f"\nQuery 2: {query2}")
    response2 = agent.run(query2)
    print(f"Response 2: {response2['response']}")

    print("\nThe agent maintains conversation history across interactions.")
    print()


def example_6_different_bedrock_models():
    """Example 6: Testing agents with different Bedrock models"""
    print("=" * 60)
    print("Example 6: Different Bedrock Models")
    print("=" * 60)

    models = [
        ("claude-3-5-sonnet", "Most capable Claude model"),
        ("claude-3-haiku", "Fast and efficient Claude model"),
        ("llama3-1-8b", "Meta's Llama 3.1 8B model"),
    ]

    query = "What is artificial intelligence in one sentence?"

    for model_id, description in models:
        print(f"\n{description} ({model_id}):")
        print("-" * 60)

        try:
            client = Fourier(
                api_key=None,
                provider="bedrock",
                region="us-east-1"
            )

            config = AgentConfig(
                name=f"Agent-{model_id}",
                max_iterations=3,
                verbose=False
            )

            agent = Agent(
                client=client,
                model=model_id,
                config=config
            )

            response = agent.run(query)
            print(f"Response: {response['response']}")

        except Exception as e:
            print(f"Error: {str(e)}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AWS Bedrock Agent Examples")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_1_basic_agent()
        example_2_multi_tool_agent()
        # example_3_thinking_mode()  # Requires internet search
        example_4_bedrock_with_cross_region()
        example_5_agent_memory()
        example_6_different_bedrock_models()

        print("=" * 60)
        print("All agent examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. AWS credentials configured")
        print("2. Access to Bedrock models")
        print("3. Proper IAM permissions")
