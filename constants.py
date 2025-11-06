"""
Constants and configuration values for ThinkSDK.

This module centralizes all configuration constants used throughout the SDK.
"""

# Default model parameters
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Anthropic-specific defaults
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_DEFAULT_MODEL = "claude-3-opus-20240229"

# Perplexity-specific defaults
PERPLEXITY_DEFAULT_MODEL = "sonar"
PERPLEXITY_DEFAULT_TEMPERATURE = 0.2
PERPLEXITY_DEFAULT_TOP_K = 0
PERPLEXITY_DEFAULT_PRESENCE_PENALTY = 0
PERPLEXITY_DEFAULT_FREQUENCY_PENALTY = 1
PERPLEXITY_SEARCH_RECENCY_FILTER = "month"

# Web search configuration
WEB_SEARCH_DEFAULT_NUM_RESULTS = 5
WEB_SEARCH_DEFAULT_MAX_CHARS = 8000
WEB_SEARCH_MAX_CHARS_PER_RESULT = 4000
WEB_SEARCH_DELAY_SECONDS = 0.5
WEB_SEARCH_RATE_LIMIT_DELAY = 0.2

# HTTP timeout settings
HTTP_TIMEOUT_SECONDS = 10

# User agents for web scraping
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]
