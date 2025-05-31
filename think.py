from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import requests
import json
import re
from models import ChatCompletionRequest, Tool
from providers import (
    GroqProvider,
    TogetherProvider,
    NebiusProvider,
    OpenAIProvider,
    AnthropicProvider,
    PerplexityProvider
)

class ResponseNormalizer:
    """Normalizes responses from different providers into a standard format."""
    
    @staticmethod
    def normalize(response: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Normalize a response from a specific provider into a standard format.
        
        Args:
            response (Dict[str, Any]): The raw response from the provider
            provider (str): The provider name
            
        Returns:
            Dict[str, Any]: Normalized response with standardized tool data
        """
        normalizer_map = {
            "meta": ResponseNormalizer._normalize_meta_response,
            "llama": ResponseNormalizer._normalize_meta_response,  # Same format as Meta
            "anthropic": ResponseNormalizer._normalize_anthropic_response,
            "perplexity": ResponseNormalizer._normalize_perplexity_response,
            "openai": ResponseNormalizer._normalize_openai_response,
            "groq": ResponseNormalizer._normalize_groq_response,
            "together": ResponseNormalizer._normalize_together_response,
            "nebius": ResponseNormalizer._normalize_nebius_response,
        }
        
        # Get the appropriate normalizer function for this provider
        normalizer = normalizer_map.get(provider.lower(), ResponseNormalizer._normalize_generic_response)
        return normalizer(response)
    
    @staticmethod
    def _normalize_meta_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Meta/Llama response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
                    # Try to extract JSON from the content
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        # Return standardized tool response
                        return {
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"]
                        }
        except Exception as e:
            print(f"Error normalizing Meta response: {e}")
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_perplexity_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Perplexity response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]
                content = message["content"]
                
                # Try to find JSON in the content
                # First check for markdown code blocks
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if json_match:
                    json_str = json_match.group(1).strip()
                    tool_data = json.loads(json_str)
                else:
                    # Try to parse the entire content as JSON
                    tool_data = ResponseNormalizer._extract_json(content)
                
                if tool_data and "tool" in tool_data and "parameters" in tool_data:
                    # Return standardized tool response
                    return {
                        "tool": tool_data["tool"],
                        "parameters": tool_data["parameters"]
                    }
        except Exception as e:
            print(f"Error normalizing Perplexity response: {e}")
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_openai_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAI response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                
                # Check if there are tool calls in the response
                if "message" in choice and "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
                    tool_call = choice["message"]["tool_calls"][0]
                    if "function" in tool_call:
                        # Return standardized tool response
                        return {
                            "tool": tool_call["function"]["name"],
                            "parameters": json.loads(tool_call["function"]["arguments"])
                        }
                
                # If no tool_calls, try to extract from content
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        return {
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"]
                        }
        except Exception as e:
            print(f"Error normalizing OpenAI response: {e}")
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_anthropic_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Anthropic response"""
        try:
            if "content" in response and len(response["content"]) > 0:
                for content_item in response["content"]:
                    if content_item["type"] == "text":
                        tool_data = ResponseNormalizer._extract_json(content_item["text"])
                        if tool_data and "tool" in tool_data and "parameters" in tool_data:
                            return {
                                "tool": tool_data["tool"],
                                "parameters": tool_data["parameters"]
                            }
                    elif content_item["type"] == "tool_use":
                        return {
                            "tool": content_item["name"],
                            "parameters": content_item["input"]
                        }
        except Exception as e:
            print(f"Error normalizing Anthropic response: {e}")
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_groq_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Groq response"""
        # Similar to OpenAI format
        return ResponseNormalizer._normalize_openai_response(response)
    
    @staticmethod
    def _normalize_together_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Together response"""
        # Handle Together-specific format, fallback to generic
        return ResponseNormalizer._normalize_generic_response(response)
    
    @staticmethod
    def _normalize_nebius_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Nebius response"""
        # Handle Nebius-specific format, fallback to generic
        return ResponseNormalizer._normalize_generic_response(response)
    
    @staticmethod
    def _normalize_generic_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Generic response normalizer for providers without specific handling"""
        try:
            # Try to extract tool data from common response patterns
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        return {
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"]
                        }
            
            # Try direct content
            if "content" in response:
                tool_data = ResponseNormalizer._extract_json(response["content"])
                if tool_data and "tool" in tool_data and "parameters" in tool_data:
                    return {
                        "tool": tool_data["tool"],
                        "parameters": tool_data["parameters"]
                    }
        except Exception as e:
            print(f"Error in generic normalizer: {e}")
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """Helper method to extract JSON from a string content"""
        try:
            # First check if the entire content is valid JSON
            if content.strip().startswith("{") and content.strip().endswith("}"):
                return json.loads(content)
            
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try to find JSON with regex for more complex content
            json_match = re.search(r"({[\s\S]*})", content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
        
        return None

class Tool(BaseModel):
    """Base class for defining tools that can be used by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

class Think:
    """Main class for interacting with LLM providers."""
    
    def __init__(self, api_key: str, provider: str = "groq", base_url: Optional[str] = None, **provider_kwargs):
        """
        Initialize the Think SDK.
        
        Args:
            api_key (str): API key for the LLM provider
            provider (str): Name of the LLM provider (default: "groq")
            base_url (Optional[str]): Custom base URL for the API
            **provider_kwargs: Additional provider-specific arguments
        """
        self.api_key = api_key
        self.provider = provider
        self.provider_kwargs = provider_kwargs
        
        # Initialize the appropriate provider
        provider_map = {
            "groq": GroqProvider,
            "together": TogetherProvider,
            "nebius": NebiusProvider,
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "perplexity": PerplexityProvider
        }
        
        provider_class = provider_map.get(provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.provider_instance = provider_class(api_key, **provider_kwargs)
        if base_url:
            self.provider_instance.BASE_URL = base_url
    
    def chat(self, **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            model (str): The model to use
            messages (List[Dict[str, str]]): List of message objects
            temperature (float, optional): Sampling temperature
            max_tokens (int, optional): Maximum number of tokens to generate
            top_p (float, optional): Nucleus sampling parameter
            frequency_penalty (float, optional): Frequency penalty
            presence_penalty (float, optional): Presence penalty
            tools (List[Tool], optional): List of tools available to the model
            tool_choice (str, optional): How the model should use the tools
            
        Returns:
            Dict[str, Any]: The chat completion response
        """
        # Convert Tool objects to dictionaries if they are not already
        if 'tools' in kwargs and kwargs['tools'] is not None:
            kwargs['tools'] = [tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in kwargs['tools']]
        
        request = ChatCompletionRequest(**kwargs)
        
        # Prepare the request payload using the provider
        payload = self.provider_instance.prepare_request(request)
        
        # If the provider doesn't support tool calling, append tool details to the system prompt
        if not hasattr(self.provider_instance, 'supports_tool_calling') or not self.provider_instance.supports_tool_calling:
            tool_descriptions = "\n".join([
                f"Tool: {tool['name']}\nDescription: {tool['description']}\nParameters: {json.dumps(tool['parameters'])}"
                for tool in kwargs.get('tools', [])
            ])
            if tool_descriptions:
                system_message = {
                    "role": "system",
                    "content": f"Available tools:\n{tool_descriptions}"
                }
                payload["messages"].insert(0, system_message)
        
        # Make the API request using the provider's endpoint and headers
        response = requests.post(
            self.provider_instance.get_chat_completion_endpoint(),
            json=payload,
            headers=self.provider_instance.headers
        )
        response.raise_for_status()
        
        # Get the raw response
        raw_response = response.json()
        
        # Normalize the response to extract tool data consistently
        normalized_response = ResponseNormalizer.normalize(raw_response, self.provider)
        
        # Check if the normalized response contains tool data
        if isinstance(normalized_response, dict) and "tool" in normalized_response and "parameters" in normalized_response:
            # If a tool is called, it's already in the standardized format
            return normalized_response
        
        # Otherwise return the provider-processed response
        return self.provider_instance.process_response(raw_response)
    
    def create_tool(self, name: str, description: str, parameters: Dict[str, Any], required: List[str] = None) -> Tool:
        """
        Create a new tool definition.
        
        Args:
            name (str): Name of the tool
            description (str): Description of what the tool does
            parameters (Dict[str, Any]): Tool parameters schema
            required (List[str], optional): List of required parameters
            
        Returns:
            Tool: The created tool definition
        """
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            required=required or []
        )