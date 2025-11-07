"""
Thinking Mode Example

This example demonstrates the Thinking Mode feature in FourierSDK's Agent class.
Thinking mode enables deep research and analysis by automatically performing
web searches to gather comprehensive context before answering queries.
"""

import os
from fourier import Fourier
from agent import Agent, AgentConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def basic_thinking_mode_example():
    """
    Basic example of using thinking mode for research-heavy queries.
    """
    print("=== Basic Thinking Mode Example ===\n")

    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return

    # Create Fourier client
    client = Fourier(api_key=api_key, provider="groq")

    # Create agent with thinking mode enabled
    agent = Agent(
        client=client,
        name="ResearchAgent",
        system_prompt="You are a knowledgeable research assistant. Provide comprehensive, well-researched answers.",
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            thinking_mode=True,           # Enable thinking mode
            thinking_depth=2,             # Perform 2 research queries
            thinking_web_search_results=5, # Get 5 results per query
            verbose=True,                 # Show detailed logs
            max_iterations=5
        )
    )

    # Ask a research question
    question = "What are the latest developments in quantum computing in 2024?"

    print(f"Question: {question}\n")
    print("Agent is performing deep research...\n")

    response = agent.run(question)

    print("\n=== Agent Response ===")
    print(response["output"])
    print(f"\nIterations: {response['iterations']}")
    print(f"Tool calls: {response['tool_calls']}")
    print(f"Success: {response['success']}")


def comparison_example():
    """
    Compare responses with and without thinking mode.
    """
    print("\n\n=== Comparison Example: With vs Without Thinking Mode ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return

    client = Fourier(api_key=api_key, provider="groq")

    question = "What are the main challenges in artificial general intelligence (AGI) research?"

    # Agent WITHOUT thinking mode
    print("1. WITHOUT Thinking Mode:")
    print("-" * 50)

    agent_normal = Agent(
        client=client,
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            thinking_mode=False,
            verbose=False
        )
    )

    response_normal = agent_normal.run(question)
    print(f"\n{response_normal['output']}\n")

    # Agent WITH thinking mode
    print("\n2. WITH Thinking Mode:")
    print("-" * 50)

    agent_thinking = Agent(
        client=client,
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            thinking_mode=True,
            thinking_depth=3,
            thinking_web_search_results=5,
            verbose=False  # Disable verbose for cleaner output
        )
    )

    response_thinking = agent_thinking.run(question)
    print(f"\n{response_thinking['output']}\n")

    print("\n=== Comparison Results ===")
    print(f"Normal mode iterations: {response_normal['iterations']}")
    print(f"Thinking mode iterations: {response_thinking['iterations']}")


def advanced_thinking_mode_example():
    """
    Advanced example with custom thinking depth and multiple questions.
    """
    print("\n\n=== Advanced Thinking Mode Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create agent with deep research capability
    agent = Agent(
        client=client,
        name="DeepResearchAgent",
        system_prompt=(
            "You are an expert research analyst. "
            "Synthesize information from multiple sources to provide "
            "comprehensive, well-structured answers with clear insights."
        ),
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            thinking_mode=True,
            thinking_depth=4,              # Deep research with 4 queries
            thinking_web_search_results=8, # More results per query
            verbose=True,
            max_iterations=10,
            temperature=0.7
        )
    )

    questions = [
        "What are the key trends in AI safety research in 2024?",
        "How is edge computing transforming IoT applications?",
        "What are the latest breakthroughs in fusion energy?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)

        response = agent.run(question)

        print(f"\nAnswer:\n{response['output']}")
        print(f"\nMetrics:")
        print(f"  - Iterations: {response['iterations']}")
        print(f"  - Success: {response['success']}")


def thinking_mode_with_tools_example():
    """
    Example combining thinking mode with custom tools.
    """
    print("\n\n=== Thinking Mode with Tools Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create agent with thinking mode
    agent = Agent(
        client=client,
        name="AnalystAgent",
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            thinking_mode=True,
            thinking_depth=2,
            thinking_web_search_results=5,
            verbose=True,
            auto_execute_tools=True
        )
    )

    # Register a custom tool
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment of text."""
        # Simple sentiment analysis
        positive_words = ["good", "great", "excellent", "positive", "breakthrough", "success"]
        negative_words = ["bad", "poor", "negative", "challenge", "problem", "issue"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"

    agent.register_tool(
        name="analyze_sentiment",
        description="Analyze the sentiment of text",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            }
        },
        function=analyze_sentiment,
        required=["text"]
    )

    # Ask a question that requires both research and tool use
    question = (
        "Research the current state of nuclear fusion energy research, "
        "then analyze the overall sentiment of recent developments."
    )

    print(f"Question: {question}\n")

    response = agent.run(question)

    print("\n=== Response ===")
    print(response["output"])
    print(f"\nTools used: {response['tool_calls']}")
    print(f"Iterations: {response['iterations']}")


def main():
    """
    Run all thinking mode examples.
    """
    print("="*60)
    print("FourierSDK Thinking Mode Examples")
    print("="*60)

    try:
        # Run examples
        basic_thinking_mode_example()

        # Uncomment to run additional examples
        # comparison_example()
        # advanced_thinking_mode_example()
        # thinking_mode_with_tools_example()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Examples Complete")
    print("="*60)


if __name__ == "__main__":
    main()
