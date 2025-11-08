"""
Example: Using Fourier config in an API

This shows how to use the config system in a FastAPI/Flask application
where you want to invoke agents based on API calls.
"""

from fourier.config import get_config
from dotenv import load_dotenv

load_dotenv()

# Simulated API framework (like FastAPI/Flask)
class MockAPI:
    """Mock API class to demonstrate the pattern"""

    def __init__(self):
        # Config is already initialized in main.py
        self.config = get_config()

    def handle_chat_request(self, agent_name: str, message: str):
        """
        API endpoint handler for chat

        POST /api/chat
        {
            "agent": "customer_support",
            "message": "What is my order status?"
        }
        """
        try:
            # Invoke agent by name - no imports needed!
            response = self.config.invoke_agent(
                agent_name,
                query=message
            )

            return {
                "success": True,
                "agent": agent_name,
                "response": response['response'],
                "iterations": response.get('iterations', 1)
            }

        except KeyError as e:
            # Agent not found
            return {
                "success": False,
                "error": str(e),
                "available_agents": self.config.registry.list_agents()
            }

    def handle_tool_request(self, tool_name: str, **kwargs):
        """
        API endpoint handler for tools

        POST /api/tool
        {
            "tool": "fetch_user_data",
            "params": {"user_id": "123"}
        }
        """
        try:
            result = self.config.invoke_tool(tool_name, **kwargs)

            return {
                "success": True,
                "tool": tool_name,
                "result": result
            }

        except KeyError as e:
            return {
                "success": False,
                "error": str(e),
                "available_tools": self.config.registry.list_tools()
            }

    def handle_workflow_request(self, workflow_name: str, input_data: dict):
        """
        API endpoint handler for workflows

        POST /api/workflow
        {
            "workflow": "customer_onboarding",
            "data": {"user_id": "123"}
        }
        """
        try:
            result = self.config.invoke_workflow(workflow_name, input_data)

            return {
                "success": True,
                "workflow": workflow_name,
                "result": result
            }

        except KeyError as e:
            return {
                "success": False,
                "error": str(e),
                "available_workflows": self.config.registry.list_workflows()
            }

    def list_resources(self):
        """
        API endpoint to list all available resources

        GET /api/resources
        """
        return self.config.list_resources()


# Example usage
if __name__ == "__main__":
    # First, initialize config (typically in main.py or __init__.py)
    from main import setup
    setup()

    # Create API instance
    api = MockAPI()

    # Simulate API requests
    print("\n=== API Example ===\n")

    # 1. Chat with agent
    print("1. Chat Request:")
    chat_response = api.handle_chat_request(
        agent_name="customer_support",
        message="What is my order status?"
    )
    print(f"Response: {chat_response}\n")

    # 2. Use tool
    print("2. Tool Request:")
    tool_response = api.handle_tool_request(
        tool_name="fetch_user_data",
        user_id="456"
    )
    print(f"Response: {tool_response}\n")

    # 3. List resources
    print("3. List Resources:")
    resources = api.list_resources()
    print(f"Available resources: {resources}\n")

    # 4. Error handling - agent not found
    print("4. Error Handling:")
    error_response = api.handle_chat_request(
        agent_name="nonexistent_agent",
        message="Hello"
    )
    print(f"Response: {error_response}\n")
