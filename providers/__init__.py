"""
Provider module with conditional imports.

Providers are imported conditionally based on whether their dependencies are installed.
This allows users to only install dependencies for the providers they actually use.
"""

import logging

logger = logging.getLogger(__name__)

# Always available providers (no extra dependencies)
from providers.groq import GroqProvider
from providers.together import TogetherProvider
from providers.nebius import NebiusProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider
from providers.perplexity import PerplexityProvider

# Conditional imports for providers with extra dependencies
BedrockProvider = None
try:
    from providers.bedrock import BedrockProvider
except ImportError as e:
    logger.debug(f"BedrockProvider not available: {e}")
    logger.debug("Install with: pip install fourier-sdk[bedrock]")


__all__ = [
    'GroqProvider',
    'TogetherProvider',
    'NebiusProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'PerplexityProvider',
]

# Add BedrockProvider to exports if available
if BedrockProvider is not None:
    __all__.append('BedrockProvider')


def get_available_providers():
    """
    Get a dictionary of available providers.

    Returns:
        Dict[str, type]: Mapping of provider names to provider classes
    """
    providers = {
        "groq": GroqProvider,
        "together": TogetherProvider,
        "nebius": NebiusProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "perplexity": PerplexityProvider,
    }

    if BedrockProvider is not None:
        providers["bedrock"] = BedrockProvider

    return providers


def is_provider_available(provider_name: str) -> bool:
    """
    Check if a provider is available (dependencies installed).

    Args:
        provider_name: Name of the provider (e.g., "bedrock", "groq")

    Returns:
        bool: True if provider is available, False otherwise
    """
    return provider_name.lower() in get_available_providers()
