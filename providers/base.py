from typing import Dict, Any, Optional, List, Callable
from models import ChatCompletionRequest, Tool
import json

class BaseProvider:
    """Base class for all providers with tool calling functionality."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(self, tool: Tool, function: Callable):
        """Register a tool with its implementation."""
        self.tools[tool.name] = function
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload with tool schemas."""
        payload = request.model_dump(exclude_none=True)
        
        if request.tools:
            # Format tools for the system message
            tools_schema = [tool.model_dump() for tool in request.tools]
            system_message = {
                "role": "system",
                "content": f"""You are a helpful AI assistant with access to the following tools. Use them when appropriate to help answer the user's questions.

Tools:
{json.dumps(tools_schema, indent=2)}

IMPORTANT: When you need to use a tool, you MUST respond with a JSON object in this exact format:
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}

For example, if you need to calculate 5 + 3, you MUST respond with:
{{
    "tool": "calculator",
    "parameters": {{
        "operation": "add",
        "a": 5,
        "b": 3
    }}
}}

DO NOT include any additional text or explanation. ONLY respond with the JSON object when using a tool.
If you don't need to use any tools, respond normally with a regular text message."""
            }
            payload["messages"].insert(0, system_message)
        
        return payload
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
        return self.tools[tool_name](**parameters)
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the response and handle tool calls."""
        try:
            # Handle different response formats
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
            elif "message" in response:
                content = response["message"]["content"]
            else:
                content = response.get("content", "")
            
            # Check if the response contains a tool call
            if content.startswith("{") and content.endswith("}"):
                try:
                    tool_call = json.loads(content)
                    if "tool" in tool_call and "parameters" in tool_call:
                        # Execute the tool
                        result = self.execute_tool(tool_call["tool"], tool_call["parameters"])
                        
                        # Add the tool result to the conversation
                        if "choices" in response:
                            response["choices"][0]["message"]["content"] = str(result)
                        elif "message" in response:
                            response["message"]["content"] = str(result)
                        else:
                            response["content"] = str(result)
                except json.JSONDecodeError:
                    pass
            
            return response
        except Exception as e:
            print(f"Error processing response: {e}")
            return response 