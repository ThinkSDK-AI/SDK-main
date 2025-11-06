"""
FourierSDK - Unified LLM Provider Interface

A Python SDK for accessing multiple Large Language Model providers
with a standardized interface, function calling support, and internet search capabilities.
"""

from fourier import Fourier
from models import Tool, ChatCompletionRequest
from response_normalizer import ResponseNormalizer
from web_search import WebSearch

__version__ = "0.1.0"

__all__ = [
    "Fourier",
    "Tool",
    "ChatCompletionRequest",
    "ResponseNormalizer",
    "WebSearch",
]
