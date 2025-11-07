"""
FourierSDK - Unified LLM Provider Interface

A Python SDK for accessing multiple Large Language Model providers
with a standardized interface, function calling support, internet search capabilities,
and autonomous agent framework.
"""

from fourier import Fourier
from models import Tool, ChatCompletionRequest
from response_normalizer import ResponseNormalizer
from web_search import WebSearch
from agent import Agent, AgentConfig

__version__ = "0.1.0"

__all__ = [
    "Fourier",
    "Tool",
    "ChatCompletionRequest",
    "ResponseNormalizer",
    "WebSearch",
    "Agent",
    "AgentConfig",
]
