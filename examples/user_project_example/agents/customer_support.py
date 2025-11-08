"""
Example agent: Customer Support Agent

This demonstrates how to create an agent that will be auto-discovered
by the Fourier config system.
"""

from fourier import Fourier
from agent import Agent, AgentConfig
import os

# Create the agent
fourier_client = Fourier(
    api_key=os.getenv("GROQ_API_KEY"),
    provider="groq"
)

config = AgentConfig(
    name="customer_support",
    description="Customer support agent for handling queries",
    max_iterations=5,
    verbose=True
)

customer_support_agent = Agent(
    client=fourier_client,
    model="llama-3.1-8b-instant",
    config=config
)

# Define a tool for the agent
def get_order_status(order_id: str) -> str:
    """Get order status"""
    # Mock implementation
    return f"Order {order_id}: Shipped, arriving in 2 days"

customer_support_agent.register_tool(
    name="get_order_status",
    description="Get the status of an order by order ID",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID"}
        },
        "required": ["order_id"]
    },
    function=get_order_status
)

# Export for auto-discovery
# Option 1: Use __agents__ dict (recommended)
__agents__ = {
    "customer_support": customer_support_agent
}

# Option 2: Just define Agent instances at module level
# The config system will auto-detect them
