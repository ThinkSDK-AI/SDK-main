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
from web_search import WebSearch

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
            internet_search (bool, optional): Whether to search the internet for context (default False)
            search_query (str, optional): Custom search query, if None will use the last user message
            search_results (int, optional): Number of search results to include (default 3)
            
        Returns:
            Dict[str, Any]: The chat completion response in standardized JSON format
        """
        # Handle internet search if enabled
        internet_search = kwargs.pop("internet_search", False)
        search_query = kwargs.pop("search_query", None)
        search_results_count = kwargs.pop("search_results", 3)
        
        if internet_search:
            # If no search query is provided, use the last user message
            if not search_query and "messages" in kwargs:
                for message in reversed(kwargs["messages"]):
                    if message.get("role") == "user":
                        search_query = message.get("content", "")
                        break
            
            if search_query:
                # Perform internet search
                search_data = WebSearch.search_and_extract(
                    query=search_query, 
                    num_results=search_results_count,
                    max_chars_per_result=2000
                )
                
                # Format search results as context
                search_context = WebSearch.format_context_for_llm(search_data)
                
                # Add system message with search context if not already present
                system_message_exists = False
                
                for message in kwargs.get("messages", []):
                    if message.get("role") == "system":
                        # Append to existing system message
                        message["content"] += f"\n\nINTERNET SEARCH RESULTS:\n{search_context}\n\nUse the above information to provide a comprehensive answer. Cite sources when using specific information."
                        system_message_exists = True
                        break
                
                if not system_message_exists:
                    # Create new system message with search context
                    if "messages" not in kwargs:
                        kwargs["messages"] = []
                        
                    kwargs["messages"].insert(0, {
                        "role": "system",
                        "content": f"INTERNET SEARCH RESULTS:\n{search_context}\n\nUse the above information to provide a comprehensive answer. Cite sources when using specific information."
                    })
        
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
        
        # Use the enhanced ResponseNormalizer to create standardized JSON response
        # This handles both tool and non-tool responses with the same format
        standardized_response = ResponseNormalizer.normalize(raw_response, self.provider)
        
        # If the standardized response contains tool data and we have a registered tool,
        # we can optionally execute the tool
        if (standardized_response.get("metadata", {}).get("response_type") == "tool_call" and 
            hasattr(self.provider_instance, "tools") and 
            standardized_response.get("response", {}).get("tool_used") in self.provider_instance.tools):
            
            tool_name = standardized_response["response"]["tool_used"]
            tool_params = standardized_response["response"]["tool_parameters"]
            
            try:
                # Execute the tool
                tool_result = self.provider_instance.execute_tool(tool_name, tool_params)
                
                # Update the tool execution output
                standardized_response["response"]["tool_execution_output"] = tool_result
            except Exception as e:
                # If tool execution fails, add error info
                standardized_response["response"]["tool_execution_error"] = str(e)
        
        # If internet search was performed, add search metadata to the response
        if internet_search and search_query:
            if "response" not in standardized_response:
                standardized_response["response"] = {}
                
            standardized_response["response"]["search_metadata"] = {
                "query": search_query,
                "num_results": search_results_count,
                "engine": "duckduckgo"
            }
            
            # Add citations directly to standardized response
            if ("results" in search_data and len(search_data["results"]) > 0 and 
                "type" in standardized_response.get("response", {}) and 
                standardized_response["response"]["type"] == "text"):
                
                citations = []
                for result in search_data["results"]:
                    citations.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", "")
                    })
                
                standardized_response["response"]["citations"] = citations
        
        return standardized_response
    
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
