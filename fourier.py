"""
FourierSDK - Unified LLM Provider Interface

This module provides the main Fourier class for interacting with multiple LLM providers
through a standardized interface.
"""

from typing import List, Optional, Dict, Any
import requests
import json
import logging
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
from exceptions import (
    InvalidAPIKeyError,
    UnsupportedProviderError,
    ProviderAPIError,
    InvalidRequestError
)

logger = logging.getLogger(__name__)


class Fourier:
    """
    Main SDK client for interacting with LLM providers.

    Provides a unified interface for chat completions across multiple
    LLM providers with support for function calling and internet search.
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "groq",
        base_url: Optional[str] = None,
        **provider_kwargs
    ):
        """
        Initialize the Fourier SDK client.

        Args:
            api_key: API key for the LLM provider
            provider: Name of the LLM provider (default: "groq")
                     Supported: groq, together, nebius, openai, anthropic, perplexity
            base_url: Custom base URL for the API (optional)
            **provider_kwargs: Additional provider-specific arguments

        Raises:
            InvalidAPIKeyError: If API key is empty or invalid
            UnsupportedProviderError: If provider is not supported
        """
        if not api_key or not api_key.strip():
            raise InvalidAPIKeyError("API key cannot be empty")

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
            supported = ", ".join(provider_map.keys())
            raise UnsupportedProviderError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {supported}"
            )

        self.provider_instance = provider_class(api_key, **provider_kwargs)
        if base_url:
            self.provider_instance.BASE_URL = base_url

    def chat(self, **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion request.

        Args:
            model: The model to use
            messages: List of message objects with 'role' and 'content' keys
            temperature: Sampling temperature (optional)
            max_tokens: Maximum number of tokens to generate (optional)
            top_p: Nucleus sampling parameter (optional)
            frequency_penalty: Frequency penalty (optional)
            presence_penalty: Presence penalty (optional)
            tools: List of Tool objects available to the model (optional)
            tool_choice: How the model should use the tools (optional)
            internet_search: Whether to search the internet for context (optional, default False)
            search_query: Custom search query (optional, uses last user message if None)
            search_results: Number of search results to include (optional, default 3)

        Returns:
            Standardized response dictionary with structure:
            {
                "status": "success",
                "timestamp": "2024-01-01T12:00:00Z",
                "metadata": {...},
                "response": {...},
                "usage": {...},
                "error": None
            }

        Raises:
            ProviderAPIError: If the API request fails
            InvalidRequestError: If required parameters are missing
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
                        message["content"] += (
                            f"\n\nINTERNET SEARCH RESULTS:\n{search_context}\n\n"
                            "Use the above information to provide a comprehensive answer. "
                            "Cite sources when using specific information."
                        )
                        system_message_exists = True
                        break

                if not system_message_exists:
                    # Create new system message with search context
                    if "messages" not in kwargs:
                        kwargs["messages"] = []

                    kwargs["messages"].insert(0, {
                        "role": "system",
                        "content": (
                            f"INTERNET SEARCH RESULTS:\n{search_context}\n\n"
                            "Use the above information to provide a comprehensive answer. "
                            "Cite sources when using specific information."
                        )
                    })

        # Convert Tool objects to dictionaries if they are not already
        if 'tools' in kwargs and kwargs['tools'] is not None:
            kwargs['tools'] = [
                tool.model_dump() if hasattr(tool, 'model_dump') else tool
                for tool in kwargs['tools']
            ]

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
        try:
            response = requests.post(
                self.provider_instance.get_chat_completion_endpoint(),
                json=payload,
                headers=self.provider_instance.headers
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            error_msg = str(e)
            if e.response:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", str(e))
                except:
                    pass
            raise ProviderAPIError(error_msg, self.provider, status_code) from e
        except requests.RequestException as e:
            logger.error(f"Network error when calling {self.provider} API: {e}", exc_info=True)
            raise ProviderAPIError(f"Network error: {str(e)}", self.provider) from e

        # Get the raw response
        raw_response = response.json()

        # Use the ResponseNormalizer to create standardized JSON response
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

    def create_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None
    ) -> Tool:
        """
        Create a new function/tool definition for the model.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: Tool parameters schema (JSON Schema format)
            required: List of required parameter names (optional)

        Returns:
            Tool object that can be passed to the chat() method

        Example:
            >>> client = Fourier(api_key="...", provider="groq")
            >>> calculator = client.create_tool(
            ...     name="calculator",
            ...     description="Performs basic arithmetic",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "operation": {"type": "string"},
            ...             "a": {"type": "number"},
            ...             "b": {"type": "number"}
            ...         }
            ...     },
            ...     required=["operation", "a", "b"]
            ... )
        """
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            required=required or []
        )
