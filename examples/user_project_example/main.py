"""
Main application using Fourier config system

This demonstrates how to use the centralized config to invoke agents,
workflows, and tools without importing them directly.
"""

from fourier.config import FourierConfig, get_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup():
    """Initialize Fourier config (call this once at startup)"""
    config = FourierConfig(
        base_dir=".",
        auto_discover=True
    )

    print("\n=== Fourier Config Initialized ===")
    print(f"Discovered resources:")
    print(f"  Agents: {config.registry.list_agents()}")
    print(f"  Workflows: {config.registry.list_workflows()}")
    print(f"  Tools: {config.registry.list_tools()}")
    print()

    return config


def use_agent_from_api():
    """
    Example: Invoking an agent from an API handler

    This shows how you can invoke agents by name without importing them.
    Perfect for API endpoints, message handlers, etc.
    """
    # Get the global config (no imports needed!)
    config = get_config()

    # Invoke agent by name
    print("=== Invoking Customer Support Agent ===")
    response = config.invoke_agent(
        "customer_support",
        query="What is the status of order #12345?"
    )

    print(f"Agent response: {response['response']}")
    print()


def use_multiple_agents():
    """Example: Using multiple agents"""
    config = get_config()

    # Customer support query
    print("=== Customer Support Query ===")
    cs_response = config.invoke_agent(
        "customer_support",
        "I need help with my order"
    )
    print(f"CS Agent: {cs_response['response'][:100]}...\n")

    # Research query
    print("=== Research Query ===")
    research_response = config.invoke_agent(
        "research_agent",
        "Explain quantum computing"
    )
    print(f"Research Agent: {research_response['response'][:100]}...\n")


def use_tools():
    """Example: Using registered tools"""
    config = get_config()

    print("=== Using Tools ===")

    # Invoke tool by name
    user_data = config.invoke_tool("fetch_user_data", user_id="123")
    print(f"User data: {user_data}")

    # Use another tool
    success = config.invoke_tool(
        "send_notification",
        user_id="123",
        message="Welcome to Fourier!",
        channel="email"
    )
    print(f"Notification sent: {success}\n")


def use_workflows():
    """Example: Using workflows"""
    config = get_config()

    print("=== Using Workflows ===")

    if "customer_onboarding" in config.registry.list_workflows():
        result = config.invoke_workflow(
            "customer_onboarding",
            input_data={"user_id": "123", "plan": "premium"}
        )
        print(f"Workflow result: {result}\n")


def get_resource_info():
    """Example: Getting information about resources"""
    config = get_config()

    print("=== Resource Information ===")

    # Get agent info
    if "customer_support" in config.registry.list_agents():
        info = config.get_resource_info("agents", "customer_support")
        print(f"Agent info: {info}")
    print()


def main():
    """Main application"""
    print("\n" + "=" * 60)
    print("Fourier Config System - Example Application")
    print("=" * 60 + "\n")

    # Initialize config once
    config = setup()

    # Now you can use agents/workflows/tools anywhere in your codebase
    # Just call get_config() and invoke by name!

    try:
        use_agent_from_api()
        # use_multiple_agents()
        use_tools()
        # use_workflows()
        get_resource_info()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
