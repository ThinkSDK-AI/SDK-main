from fourier import Fourier
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
from models import Tool
import re
import json

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

def calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator function."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

def get_weather(location: str, unit: str = "celsius") -> str:
    """A mock weather function."""
    return f"The weather in {location} is 25 degrees {unit}"

def test_groq():
    """Test Groq provider"""
    print("\n=== Testing Groq Provider ===")
    client = Fourier(
        api_key=os.getenv("GROQ_API_KEY"),
        provider="groq"
    )
    
    # Create a simple calculator tool
    calculator = client.create_tool(
        name="calculator",
        description="A simple calculator that can perform basic arithmetic operations",
        parameters={
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        required=["operation", "a", "b"]
    )
    
    response = client.chat(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "What is 5 + 3? Use the calculator tool if needed."}
        ],
        tools=[calculator]
    )
    
    print("Response:", response["choices"][0]["message"]["content"])

def test_together_with_tools():
    """Test Together AI provider with tool calling"""
    print("\n=== Testing Together AI Provider with Tools ===")
    client = Fourier(
        api_key=os.getenv("TOGETHER_API_KEY"),
        provider="together"
    )
    
    # Create tools
    calculator_tool = Tool(
        name="calculator",
        description="A simple calculator that can perform basic arithmetic operations",
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
    
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    )
    
    # Register tool implementations
    client.provider_instance.register_tool(calculator_tool, calculator)
    client.provider_instance.register_tool(weather_tool, get_weather)
    
    # Test with a calculation
    print("\nTesting calculator tool...")
    response = client.chat(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "Calculate 5 + 3 using the calculator tool. Respond with ONLY a JSON object containing the tool call."}
        ],
        tools=[calculator_tool.model_dump()]
    )
    
    print("Raw Response:", json.dumps(response, indent=2))
    if "choices" in response:
        print("Calculation Response:", response["choices"][0]["message"]["content"])
    elif "message" in response:
        print("Calculation Response:", response["message"]["content"])
    else:
        print("Calculation Response:", response.get("content", "No content found"))
    
    # Test with weather
    print("\nTesting weather tool...")
    response = client.chat(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "Check the weather in New York using the weather tool. Respond with ONLY a JSON object containing the tool call."}
        ],
        tools=[weather_tool.model_dump()]
    )
    
    print("Raw Response:", json.dumps(response, indent=2))
    if "choices" in response:
        print("Weather Response:", response["choices"][0]["message"]["content"])
    elif "message" in response:
        print("Weather Response:", response["message"]["content"])
    else:
        print("Weather Response:", response.get("content", "No content found"))

def test_nebius_with_tools():
    """Test Nebius Cloud provider with tool calling"""
    print("\n=== Testing Nebius Cloud Provider with Tools ===")
    client = Fourier(
        api_key=os.getenv("NEBIUS_API_KEY"),
        provider="nebius"
    )
    
    # Create tools
    calculator_tool = Tool(
        name="calculator",
        description="A simple calculator that can perform basic arithmetic operations",
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
    
    # Register tool implementation
    client.provider_instance.register_tool(calculator_tool, calculator)
    
    print("\nTesting calculator tool...")
    response = client.chat(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "user", "content": "Calculate 10 * 5 using the calculator tool. Respond with ONLY a JSON object containing the tool call."}
        ],
        tools=[calculator_tool.model_dump()],
        temperature=0.6
    )
    
    print("Raw Response:", json.dumps(response, indent=2))
    if "choices" in response:
        print("Response:", response["choices"][0]["message"]["content"])
    elif "message" in response:
        print("Response:", response["message"]["content"])
    else:
        print("Response:", response.get("content", "No content found"))


def test_anthropic_with_tools():
    """Test Anthropic provider with tool calling"""
    print("\n=== Testing Anthropic Provider with Tools ===")
    client = Fourier(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        provider="anthropic"
    )
    
    # Create tools
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    )
    
    # Register tool implementation
    client.provider_instance.register_tool(weather_tool, get_weather)
    
    print("\nTesting weather tool...")
    response = client.chat(
        model="claude-3-opus-20240229",
        messages=[
            {"role": "user", "content": "Check the weather in Paris using the weather tool. Respond with ONLY a JSON object containing the tool call."}
        ],
        tools=[weather_tool.model_dump()]
    )
    
    print("Raw Response:", json.dumps(response, indent=2))
    if "choices" in response:
        print("Response:", response["choices"][0]["message"]["content"])
    elif "message" in response:
        print("Response:", response["message"]["content"])
    else:
        print("Response:", response.get("content", "No content found"))

def test_perplexity_with_tools():
    """Test Perplexity provider with tool calling"""
    print("\n=== Testing Perplexity Provider with Tools ===")
    client = Fourier(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        provider="perplexity"
    )
    
    # Create tools using the Think class
    weather_tool = client.create_tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    )
    
    # Register tool implementation
    client.provider_instance.register_tool(weather_tool, get_weather)
    
    print("\nTesting weather tool...")
    response = client.chat(
        model="sonar",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can use tools to get information."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        tools=[weather_tool],  # Pass as Tool object
        temperature=0.2,
        max_tokens=123,
        top_p=0.9
    )
    
    
    print("Raw Response:", json.dumps(response, indent=2))
    if "choices" in response:
        text = response["choices"][0]["message"]["content"]
        # Unescape the string if it's escaped (e.g., \\n or \\" etc.)
        text = text.encode().decode('unicode_escape')

# Regex pattern to find content inside ```json ... ```
        pattern = r'```json\s*(\{.*?\})\s*```'

        # Use DOTALL to match across newlines
        match = re.search(pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                parsed_json = json.loads(json_str)
                print("Extracted JSON:")
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", e)
        else:
            print("No JSON found")
        # print("Response:", response["choices"][0]["message"]["content"])
    elif "message" in response:
        print("Response:", response["message"]["content"])
    else:
        print("Response:", response.get("content", "No content found"))

if __name__ == "__main__":
    # Test each provider
    # test_groq()
    test_together_with_tools()
    test_nebius_with_tools()
    # test_anthropic_with_tools()
    test_perplexity_with_tools()
    # test_openai() 