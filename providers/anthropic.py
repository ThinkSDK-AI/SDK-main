from typing import Dict, Any, Optional
from models import ChatCompletionRequest, Tool
from .base import BaseProvider
import json

class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic Claude API."""
    
    BASE_URL = "https://api.anthropic.com/v1"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload for Anthropic API."""
        # Get base payload from parent class
        payload = super().prepare_request(request)
        
        # Convert messages to Anthropic format
        messages = []
        for msg in payload["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Handle tools if present
        if "tools" in payload:
            tools = payload["tools"]
            formatted_tools = []
            
            for tool in tools:
                # If tool is already in Anthropic format, use it directly
                if isinstance(tool, dict) and "name" in tool and "description" in tool:
                    formatted_tools.append(tool)
                else:
                    # Convert our Tool model to Anthropic format
                    formatted_tool = {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                    formatted_tools.append(formatted_tool)
            
            payload["tools"] = formatted_tools
        
        # Update payload with Anthropic-specific fields
        payload["messages"] = messages
        payload["model"] = payload.get("model", "claude-3-opus-20240229")
        payload["max_tokens"] = payload.get("max_tokens", 4096)
        
        return payload
    
    def get_chat_completion_endpoint(self) -> str:
        """Get the chat completion endpoint."""
        return f"{self.BASE_URL}/messages"
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the response from Anthropic API."""
        try:
            if "error" in response:
                print(f"Anthropic Error: {response['error']}")
                return response
            
            if "content" in response:
                content = response["content"][0]["text"]
                
                # Check for tool calls in the content
                try:
                    tool_call = json.loads(content)
                    if isinstance(tool_call, dict) and "tool" in tool_call:
                        response["choices"] = [{
                            "message": {
                                "content": content,
                                "role": "assistant"
                            }
                        }]
                except json.JSONDecodeError:
                    # If not a tool call, format as regular response
                    response["choices"] = [{
                        "message": {
                            "content": content,
                            "role": "assistant"
                        }
                    }]
            
            return super().process_response(response)
        except Exception as e:
            print(f"Error processing Anthropic response: {e}")
            return response 