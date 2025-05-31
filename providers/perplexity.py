from typing import Dict, Any, Optional
from models import ChatCompletionRequest, Tool
from .base import BaseProvider
import json

class PerplexityProvider(BaseProvider):
    """Provider implementation for Perplexity API."""
    
    BASE_URL = "https://api.perplexity.ai"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload for Perplexity API."""
        # Get base payload from parent class
        payload = super().prepare_request(request)
        
        # Convert messages to Perplexity format
        messages = []
        for msg in payload["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Create Perplexity-specific payload
        perplexity_payload = {
            "model": payload.get("model", "sonar"),
            "messages": messages,
            "max_tokens": payload.get("max_tokens", 123),
            "temperature": payload.get("temperature", 0.2),
            "top_p": payload.get("top_p", 0.9),
            "search_domain_filter": ["wikipedia.org"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "response_format": {
                "type": "text"
            },
            "web_search_options": {"search_context_size": "high"}
        }
        
        # Handle tools if present
        if "tools" in payload:
            tools = payload["tools"]
            formatted_tools = []
            
            for tool in tools:
                # If tool is already in Perplexity format, use it directly
                if isinstance(tool, dict) and "name" in tool and "description" in tool:
                    formatted_tools.append(tool)
                else:
                    # Convert our Tool model to Perplexity format
                    formatted_tool = {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"]
                        }
                    }
                    formatted_tools.append(formatted_tool)
            
            perplexity_payload["tools"] = formatted_tools
            
            # Add tool instructions to the system message
            tool_instructions = """You are a helpful AI assistant with access to the following tools. Use them when appropriate to help answer the user's questions.

IMPORTANT: When you need to use a tool, you MUST respond with a JSON object in this exact format:
{
    "tool": "tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}

For example, if you need to get the weather for Tokyo, you MUST respond with:
{
    "tool": "get_weather",
    "parameters": {
        "location": "Tokyo"
    }
}

DO NOT include any additional text or explanation. ONLY respond with the JSON object when using a tool.
If you don't need to use any tools, respond normally with a regular text message."""
            
            # Add or update system message
            system_message = {
                "role": "system",
                "content": tool_instructions
            }
            
            # Insert system message at the beginning if not already present
            if not any(msg["role"] == "system" for msg in perplexity_payload["messages"]):
                perplexity_payload["messages"].insert(0, system_message)
            else:
                # Update existing system message
                for msg in perplexity_payload["messages"]:
                    if msg["role"] == "system":
                        msg["content"] = tool_instructions
                        break
        
        return perplexity_payload
    
    def get_chat_completion_endpoint(self) -> str:
        """Get the chat completion endpoint."""
        return f"{self.BASE_URL}/chat/completions"
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the response from Perplexity API."""
        try:
            if "error" in response:
                print(f"Perplexity Error: {response['error']}")
                return response
            
            if "choices" in response:
                message = response["choices"][0]["message"]
                content = message["content"]
                
                # Try to find JSON in the content
                try:
                    # Look for JSON between ```json and ```
                    if "```json" in content and "```" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        tool_call = json.loads(json_str)
                    else:
                        # Try to parse the entire content as JSON
                        tool_call = json.loads(content)
                    
                    # If we found a valid tool call, execute it
                    if isinstance(tool_call, dict) and "tool" in tool_call and "parameters" in tool_call:
                        # Execute the tool
                        result = self.execute_tool(tool_call["tool"], tool_call["parameters"])
                        
                        # Update the response with the tool result
                        message["content"] = str(result)
                        
                except (json.JSONDecodeError, IndexError):
                    # Content is not JSON or doesn't contain a tool call, leave as is
                    pass
            
            return super().process_response(response)
        except Exception as e:
            print(f"Error processing Perplexity response: {e}")
            return response 