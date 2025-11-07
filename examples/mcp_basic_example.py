"""
Basic MCP Client Example

This example demonstrates how to use the MCPClient to connect to
different types of MCP servers and execute tools.
"""

import os
from mcp import MCPClient

def main():
    print("=== FourierSDK MCP Basic Example ===\n")

    # Create MCP client
    client = MCPClient()

    # Example 1: Add a directory of tools
    print("1. Loading tools from directory...")
    if client.add_directory("./mcp_tools"):
        print(f"   ✓ Loaded tools from ./mcp_tools")
    else:
        print("   ✗ Failed to load directory (may not exist)")

    # Example 2: Add MCP server from configuration
    print("\n2. Loading MCP server from configuration...")

    # Configuration for a local MCP server
    config = {
        "command": "python",
        "args": ["-m", "mcp_server_example"],
        "env": {
            "API_KEY": os.getenv("MCP_API_KEY", "demo_key")
        }
    }

    # Note: This will fail if the MCP server isn't installed
    # For demo purposes, we'll catch the error
    try:
        if client.add_config(config, name="example_server"):
            print("   ✓ Connected to MCP server")
        else:
            print("   ✗ Could not connect (server may not be installed)")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 3: Add remote MCP server
    print("\n3. Connecting to remote MCP server...")

    # Note: This is a demo URL and won't actually work
    # Replace with your actual MCP server URL
    remote_url = os.getenv("MCP_SERVER_URL", "https://mcp.example.com/api")
    headers = {
        "Authorization": f"Bearer {os.getenv('MCP_TOKEN', 'demo_token')}"
    }

    try:
        if client.add_url(remote_url, headers=headers):
            print(f"   ✓ Connected to {remote_url}")
        else:
            print(f"   ✗ Could not connect to {remote_url}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # List all available tools
    print("\n4. Available Tools:")
    tool_names = client.get_tool_names()

    if tool_names:
        for i, tool_name in enumerate(tool_names, 1):
            tool = client.get_tool(tool_name)
            print(f"   {i}. {tool_name}")
            print(f"      Description: {tool.description}")
    else:
        print("   No tools loaded")
        print("\n   To use this example, create MCP tools in ./mcp_tools/")
        print("   See MCP.md for documentation on creating tools")

    # Example tool call (if tools are available)
    if "calculator" in tool_names:
        print("\n5. Executing 'calculator' tool:")
        try:
            result = client.call_tool("calculator", {
                "operation": "multiply",
                "a": 25,
                "b": 4
            })
            print(f"   Result: 25 × 4 = {result}")
        except Exception as e:
            print(f"   Error: {e}")

    # Cleanup
    print("\n6. Closing connections...")
    client.close()
    print("   ✓ All connections closed")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
