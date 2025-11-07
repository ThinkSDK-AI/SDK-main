"""
MCP with Agent Example

This example demonstrates how to use MCP tools with the Agent class
for autonomous tool execution and conversation management.
"""

import os
from fourier import Fourier
from agent import Agent, AgentConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    print("=== FourierSDK MCP Agent Example ===\n")

    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        print("Please set your API key in .env file")
        return

    # Create Fourier client
    print("1. Initializing Fourier client...")
    client = Fourier(api_key=api_key, provider="groq")
    print("   ✓ Client initialized\n")

    # Create agent with configuration
    print("2. Creating agent...")
    agent = Agent(
        client=client,
        name="MCPAgent",
        system_prompt="You are a helpful assistant with access to various tools via MCP.",
        model="mixtral-8x7b-32768",
        config=AgentConfig(
            verbose=True,
            max_iterations=10,
            auto_execute_tools=True,
            return_intermediate_steps=True
        )
    )
    print("   ✓ Agent created\n")

    # Example 1: Register MCP directory
    print("3. Registering MCP tools from directory...")
    if agent.register_mcp_directory("./mcp_tools"):
        print("   ✓ Loaded tools from ./mcp_tools")
    else:
        print("   ✗ Could not load directory (may not exist)")
        print("   Creating sample tools is recommended - see MCP.md")

    # Example 2: Register multiple directories
    print("\n4. Registering multiple MCP directories...")
    directories = [
        "./mcp_tools/search",
        "./mcp_tools/analysis",
        "./mcp_tools/utils"
    ]

    results = agent.register_mcp_directories(directories)
    for directory, success in results.items():
        status = "✓" if success else "✗"
        print(f"   {status} {directory}")

    # Example 3: Register from configuration file
    print("\n5. Registering MCP tools from config...")
    if os.path.exists("./mcp_config.json"):
        if agent.register_mcp_config("./mcp_config.json"):
            print("   ✓ Loaded tools from config")
        else:
            print("   ✗ Could not load config")
    else:
        print("   ⓘ Config file not found (./mcp_config.json)")

    # Example 4: Register remote MCP server
    print("\n6. Registering remote MCP server...")
    remote_url = os.getenv("MCP_SERVER_URL")
    if remote_url:
        headers = {"Authorization": f"Bearer {os.getenv('MCP_TOKEN')}"}
        if agent.register_mcp_url(remote_url, headers=headers):
            print(f"   ✓ Connected to {remote_url}")
        else:
            print(f"   ✗ Could not connect to {remote_url}")
    else:
        print("   ⓘ MCP_SERVER_URL not set in environment")

    # Show registered tools
    print("\n7. Registered Tools:")
    all_tools = agent.get_tool_names()
    mcp_tools = agent.get_mcp_tool_names()

    print(f"   Total tools: {len(all_tools)}")
    print(f"   MCP tools: {len(mcp_tools)}")

    if all_tools:
        print("\n   Available tools:")
        for i, tool_name in enumerate(all_tools, 1):
            is_mcp = " (MCP)" if tool_name in mcp_tools else ""
            print(f"   {i}. {tool_name}{is_mcp}")
    else:
        print("\n   No tools registered")
        print("\n   To see the agent in action:")
        print("   1. Create MCP tools in ./mcp_tools/ directory")
        print("   2. See MCP.md for tool creation guide")
        print("   3. Run this example again")
        return

    # Example 5: Use the agent
    print("\n8. Running agent with task...")
    print("   Task: 'Tell me about the tools you have available'\n")

    try:
        response = agent.run("Tell me about the tools you have available")

        print("\n   === Agent Response ===")
        print(f"   {response['output']}")

        print(f"\n   Iterations: {response['iterations']}")
        print(f"   Tool calls: {response['tool_calls']}")

        if response.get('intermediate_steps'):
            print("\n   Intermediate steps:")
            for step in response['intermediate_steps']:
                print(f"   - Tool: {step['tool']}")
                print(f"     Parameters: {step['parameters']}")
                print(f"     Result: {step['result']}\n")

    except Exception as e:
        print(f"\n   Error running agent: {e}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
