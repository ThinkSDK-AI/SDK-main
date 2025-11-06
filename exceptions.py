"""
Custom exceptions for FourierSDK.

This module defines all custom exception classes used throughout the SDK
for more precise error handling and reporting.
"""


class FourierSDKError(Exception):
    """Base exception class for all FourierSDK errors."""
    pass


class InvalidAPIKeyError(FourierSDKError):
    """Raised when an API key is invalid or missing."""
    pass


class UnsupportedProviderError(FourierSDKError):
    """Raised when an unsupported provider is specified."""
    pass


class ProviderAPIError(FourierSDKError):
    """Raised when a provider API returns an error."""

    def __init__(self, message: str, provider: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message}")


class ToolExecutionError(FourierSDKError):
    """Raised when a tool execution fails."""

    def __init__(self, message: str, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class ToolNotFoundError(FourierSDKError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found or not registered")


class ResponseNormalizationError(FourierSDKError):
    """Raised when response normalization fails."""

    def __init__(self, message: str, provider: str):
        self.provider = provider
        super().__init__(f"Failed to normalize {provider} response: {message}")


class WebSearchError(FourierSDKError):
    """Raised when web search operations fail."""
    pass


class InvalidRequestError(FourierSDKError):
    """Raised when a request contains invalid parameters."""
    pass
