from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import requests
import json
from models import ChatCompletionRequest, Tool
from providers import (
    GroqProvider,
    TogetherProvider,
    NebiusProvider,
    OpenAIProvider,
    AnthropicProvider,
    PerplexityProvider
)
from response_normalizer import ResponseNormalizer

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
        
        # Return the normalized response (works for both tool and standard responses)
        return normalized_response
    
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
