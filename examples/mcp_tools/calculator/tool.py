"""
Example MCP Tool: Calculator

This is a simple calculator tool that demonstrates how to create
MCP-compatible tools for FourierSDK.
"""


def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform arithmetic operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


# MCP Tools definition
# This is what FourierSDK will look for when loading tools
MCP_TOOLS = [{
    "name": "calculator",
    "description": "Perform basic arithmetic operations: add, subtract, multiply, divide",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    },
    "function": calculate,
    "metadata": {
        "version": "1.0.0",
        "author": "FourierSDK",
        "category": "math"
    }
}]
