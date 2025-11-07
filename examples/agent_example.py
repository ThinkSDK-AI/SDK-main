"""
Example usage of the FourierSDK Agent class.

This example demonstrates how to create an autonomous agent with tools
that can execute complex tasks automatically.
"""

import os
from dotenv import load_dotenv
from fourier import Fourier
from agent import Agent, AgentConfig

# Load environment variables
load_dotenv()


# Example 1: Math Assistant Agent
def example_math_agent():
    """Create an agent that can perform calculations."""
    print("\n=== Example 1: Math Assistant Agent ===\n")

    # Initialize Fourier client
    client = Fourier(
        api_key=os.getenv("GROQ_API_KEY"),
        provider="groq"
    )

    # Create agent with custom configuration
    config = AgentConfig(
        max_iterations=5,
        verbose=True,
        return_intermediate_steps=True
    )

    agent = Agent(
        client=client,
        name="MathWizard",
        system_prompt=(
            "You are MathWizard, an expert mathematics assistant. "
            "When users ask math questions, use the calculator tool to compute answers accurately. "
            "Always show your work and explain the steps."
        ),
        model="mixtral-8x7b-32768",
        config=config
    )

    # Define calculator tool
    def calculator(operation: str, a: float, b: float) -> float:
        """Perform basic arithmetic operations."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }
        return operations.get(operation, lambda x, y: "Invalid operation")(a, b)

    # Register the calculator tool
    agent.register_tool(
        name="calculator",
        description="Perform arithmetic operations: add, subtract, multiply, or divide two numbers",
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
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            }
        },
        required=["operation", "a", "b"],
        function=calculator
    )

    # Run the agent
    result = agent.run("What is 156 multiplied by 23? Then add 47 to the result.")

    print(f"\n--- Agent Response ---")
    print(f"Output: {result['output']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tool Calls: {result['tool_calls']}")
    print(f"Success: {result['success']}")

    if result['intermediate_steps']:
        print("\n--- Intermediate Steps ---")
        for step in result['intermediate_steps']:
            print(f"Step {step['iteration']}: {step['tool']}({step['parameters']}) = {step['result']}")


# Example 2: Research Assistant with Web Search
def example_research_agent():
    """Create an agent that can search the web and summarize information."""
    print("\n=== Example 2: Research Assistant Agent ===\n")

    client = Fourier(
        api_key=os.getenv("TOGETHER_API_KEY"),
        provider="together"
    )

    config = AgentConfig(
        max_iterations=3,
        verbose=True
    )

    agent = Agent(
        client=client,
        name="ResearchBot",
        system_prompt=(
            "You are ResearchBot, a research assistant. "
            "When users ask for information, use the search tool to find relevant data. "
            "Provide comprehensive answers with citations."
        ),
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        config=config
    )

    # Mock search function (replace with actual API)
    def web_search(query: str, num_results: int = 3) -> str:
        """Search the web for information."""
        # In a real implementation, this would call a search API
        return f"Search results for '{query}': Found {num_results} relevant articles about {query}."

    agent.register_tool(
        name="web_search",
        description="Search the internet for information on a given topic",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            }
        },
        required=["query"],
        function=web_search
    )

    result = agent.run("What are the latest trends in artificial intelligence?")

    print(f"\n--- Agent Response ---")
    print(f"Output: {result['output']}")
    print(f"Tool Calls: {result['tool_calls']}")


# Example 3: Multi-Tool Agent
def example_multi_tool_agent():
    """Create an agent with multiple tools."""
    print("\n=== Example 3: Multi-Tool Agent ===\n")

    client = Fourier(
        api_key=os.getenv("GROQ_API_KEY"),
        provider="groq"
    )

    config = AgentConfig(
        max_iterations=10,
        verbose=True,
        return_intermediate_steps=True
    )

    agent = Agent(
        client=client,
        name="MultiBot",
        system_prompt=(
            "You are a versatile assistant with access to multiple tools. "
            "Use the appropriate tool for each task. Be efficient and accurate."
        ),
        model="mixtral-8x7b-32768",
        config=config
    )

    # Define multiple tools
    def calculator(operation: str, a: float, b: float) -> float:
        """Calculate arithmetic operations."""
        ops = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero"
        }
        return ops.get(operation, 0)

    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 72°F"

    def translate_text(text: str, target_language: str) -> str:
        """Translate text to target language."""
        return f"[Translated to {target_language}]: {text}"

    # Register all tools
    agent.register_tool(
        name="calculator",
        description="Perform arithmetic calculations",
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

    agent.register_tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or location"}
            }
        },
        required=["location"],
        function=get_weather
    )

    agent.register_tool(
        name="translate",
        description="Translate text to another language",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language (e.g., 'Spanish', 'French')"}
            }
        },
        required=["text", "target_language"],
        function=translate_text
    )

    # Run complex query
    result = agent.run(
        "What's 50 times 3, and what's the weather like in Paris?"
    )

    print(f"\n--- Agent Response ---")
    print(f"Output: {result['output']}")
    print(f"Tool Calls: {result['tool_calls']}")

    if result['intermediate_steps']:
        print("\n--- Tools Used ---")
        for step in result['intermediate_steps']:
            print(f"- {step['tool']}")


# Example 4: Conversation Agent with Memory
def example_conversation_agent():
    """Create an agent that maintains conversation context."""
    print("\n=== Example 4: Conversation Agent with Memory ===\n")

    client = Fourier(
        api_key=os.getenv("GROQ_API_KEY"),
        provider="groq"
    )

    config = AgentConfig(
        max_iterations=5,
        verbose=False
    )

    agent = Agent(
        client=client,
        name="MemoryBot",
        system_prompt=(
            "You are a helpful assistant with a memory. "
            "Remember previous interactions and build on them. "
            "Use tools when needed to provide accurate information."
        ),
        model="mixtral-8x7b-32768",
        config=config
    )

    # Add a note-taking tool
    notes = {}

    def save_note(key: str, value: str) -> str:
        """Save a note for later retrieval."""
        notes[key] = value
        return f"Saved note '{key}': {value}"

    def get_note(key: str) -> str:
        """Retrieve a previously saved note."""
        return notes.get(key, f"No note found for '{key}'")

    agent.register_tool(
        name="save_note",
        description="Save information for later use",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Name/key for the note"},
                "value": {"type": "string", "description": "Content to save"}
            }
        },
        required=["key", "value"],
        function=save_note
    )

    agent.register_tool(
        name="get_note",
        description="Retrieve a previously saved note",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Name/key of the note to retrieve"}
            }
        },
        required=["key"],
        function=get_note
    )

    # Multi-turn conversation
    print("Turn 1:")
    result1 = agent.run("Remember that my favorite color is blue.", reset_history=True)
    print(f"Agent: {result1['output']}\n")

    print("Turn 2:")
    result2 = agent.run("What's my favorite color?", reset_history=False)
    print(f"Agent: {result2['output']}")


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("FourierSDK Agent Examples")
    print("=" * 60)

    # Uncomment the examples you want to run:

    example_math_agent()
    # example_research_agent()
    # example_multi_tool_agent()
    # example_conversation_agent()
