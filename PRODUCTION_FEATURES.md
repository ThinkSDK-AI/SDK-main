# Production-Grade Features - FourierSDK

This document outlines the production-grade features and safety mechanisms implemented in FourierSDK, particularly for the Thinking Mode functionality.

## Table of Contents

- [Overview](#overview)
- [Configuration Validation](#configuration-validation)
- [Input Sanitization](#input-sanitization)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Context Management](#context-management)
- [Performance Monitoring](#performance-monitoring)
- [Security Features](#security-features)
- [Testing](#testing)
- [Best Practices](#best-practices)

## Overview

FourierSDK implements comprehensive production-grade features to ensure:

- **Reliability**: Robust error handling and graceful degradation
- **Security**: Input validation and sanitization
- **Performance**: Rate limiting and context management
- **Observability**: Detailed logging and metrics
- **Maintainability**: Comprehensive test coverage

## Configuration Validation

### Automatic Parameter Validation

All `AgentConfig` parameters are validated and clamped to safe ranges:

```python
from agent import AgentConfig

config = AgentConfig(
    thinking_mode=True,
    thinking_depth=10,              # Automatically clamped to 5
    thinking_web_search_results=20  # Automatically clamped to 10
)

# Config values are safe
assert config.thinking_depth == 5
assert config.thinking_web_search_results == 10
```

### Validation Rules

| Parameter | Min | Max | Default | Auto-Fix |
|-----------|-----|-----|---------|----------|
| `thinking_depth` | 1 | 5 | 2 | Yes (clamp) |
| `thinking_web_search_results` | 1 | 10 | 5 | Yes (clamp) |
| `max_iterations` | 1 | 100 | 10 | Yes (clamp) |
| `temperature` | 0.0 | 2.0 | 0.7 | Yes (clamp) |
| `timeout_seconds` | 1 | ∞ | None | Yes (set None if < 1) |

### Validation Benefits

- **Prevents misconfiguration**: Invalid values are automatically corrected
- **Clear warnings**: All auto-fixes are logged with warnings
- **Fail-safe defaults**: System continues with safe values

## Input Sanitization

### Query Sanitization

All user inputs and generated search queries are sanitized:

```python
from agent import Agent

# Handles malicious input
query = "<script>alert('xss')</script> What is AI?"
sanitized = Agent._sanitize_query(query)
# Result: "scriptalertxssscript What is AI?"

# Handles excessive whitespace
query = "What  are   the    latest     AI     developments?"
sanitized = Agent._sanitize_query(query)
# Result: "What are the latest AI developments?"

# Truncates overly long queries
query = "a" * 10000
sanitized = Agent._sanitize_query(query)
# Result: First 500 characters only
```

### Sanitization Rules

1. **Length validation**: Queries must be 2-500 characters
2. **Whitespace normalization**: Multiple spaces reduced to single space
3. **Character filtering**: Removes suspicious/harmful characters
4. **Unicode support**: Handles international characters safely
5. **Type checking**: Ensures input is a string

### Protection Against

- **XSS attacks**: Removes script tags and HTML
- **Injection attacks**: Filters special characters
- **Resource exhaustion**: Enforces length limits
- **Type errors**: Validates input types

## Error Handling

### Hierarchical Error Handling

Errors are handled at multiple levels with appropriate fallbacks:

```python
from agent import Agent, AgentConfig
from fourier import Fourier

client = Fourier(api_key="...", provider="groq")

# Configure error handling behavior
agent = Agent(
    client=client,
    config=AgentConfig(
        stop_on_error=False,  # Continue on errors
        verbose=True           # Log all errors
    )
)

# Errors are logged but don't crash the application
response = agent.run("Research query that might fail")
# System gracefully degrades and returns available results
```

### Error Handling Levels

1. **Query Generation**: Falls back to original user input
2. **Individual Searches**: Continues with remaining searches
3. **Research Execution**: Returns partial results or empty string
4. **Main Agent Loop**: Handles all exceptions with proper logging

### Error Categories

- **Input Validation Errors**: Logged, input sanitized or rejected
- **Network Errors**: Logged, search skipped, continues
- **LLM Errors**: Logged, falls back to original query
- **Timeout Errors**: Logged, operation continues
- **Critical Errors**: Raised only if `stop_on_error=True`

## Rate Limiting

### Automatic Rate Limiting

Built-in rate limiting prevents overwhelming external services:

```python
# Configurable delay between searches
RATE_LIMIT_DELAY = 1.0  # seconds

# Automatically applied between consecutive searches
agent = Agent(client=client, config=AgentConfig(thinking_depth=5))
# 5 searches with 1-second delays = ~4 seconds of rate limiting
```

### Rate Limiting Strategy

1. **Inter-search delays**: 1 second between consecutive searches
2. **No delay for first search**: Immediate start
3. **Linear scaling**: Delay proportional to number of searches
4. **Configurable**: Modify `RATE_LIMIT_DELAY` constant

### Benefits

- **API protection**: Prevents rate limit violations
- **Resource management**: Avoids overwhelming services
- **Cost control**: Reduces unnecessary API calls
- **Reliability**: Improves success rate of searches

## Context Management

### Automatic Context Truncation

Research context is automatically managed to prevent token limit issues:

```python
from agent import Agent

# Context automatically truncated if too long
MAX_CONTEXT_LENGTH = 50000  # characters

# Very long research results are truncated
agent = Agent(client=client, config=AgentConfig(thinking_mode=True))
response = agent.run("Complex query with lots of results")
# Context is capped at 50,000 characters
```

### Context Management Features

1. **Size limiting**: Maximum 50,000 characters
2. **Truncation notice**: Clear indication when truncated
3. **Smart truncation**: Preserves structure where possible
4. **Logging**: Warnings for truncated content

### Example Output

```
Research results...
[lots of content]
...
[Context truncated due to length...]
```

### Configuration

```python
# Global constant
MAX_CONTEXT_LENGTH = 50000  # characters

# Modify for specific use cases
Agent._truncate_context(context, max_length=100000)
```

## Performance Monitoring

### Detailed Metrics Tracking

All operations are timed and logged:

```python
# Automatic timing and metrics
agent = Agent(
    client=client,
    config=AgentConfig(thinking_mode=True, verbose=True)
)

response = agent.run("Research query")

# Logs include:
# - Query generation time
# - Individual search times
# - Total research time
# - Success/failure counts
# - Context size metrics
```

### Logged Metrics

| Metric | Description | Threshold | Action if Exceeded |
|--------|-------------|-----------|-------------------|
| Query Generation Time | Time to generate search queries | 15s | Warning logged |
| Search Time | Time per web search | 30s | Warning logged |
| Total Research Time | Total thinking mode time | - | Logged for analysis |
| Context Length | Size of research context | 50,000 chars | Truncated |
| Success Rate | Searches successful/total | - | Logged for analysis |

### Example Verbose Output

```
[Thinking Mode] Starting deep research for: What are the latest...
[Thinking Mode] Generating research queries...
[Thinking Mode] Generated 3 valid queries
[Thinking Mode] Search 1/3: Latest AI developments
[Thinking Mode] Search completed: 5432 characters in 2.3s
[Thinking Mode] Search 2/3: AI breakthroughs 2024
[Thinking Mode] Search completed: 6123 characters in 2.7s
[Thinking Mode] Search 3/3: Recent AI innovations
[Thinking Mode] Search completed: 4876 characters in 2.1s
[Thinking Mode] Research complete in 9.2s: 16431 chars, 3 successful, 0 failed
```

## Security Features

### Multi-Layer Security

1. **Input Validation**
   - Type checking
   - Length validation
   - Character filtering

2. **Query Sanitization**
   - XSS prevention
   - Injection protection
   - Unicode handling

3. **Resource Limits**
   - Maximum query length (500 chars)
   - Maximum context size (50,000 chars)
   - Maximum search results (10)
   - Maximum research depth (5)

4. **Safe Defaults**
   - Thinking mode disabled by default
   - Conservative rate limits
   - Graceful error handling

### Security Best Practices

```python
# 1. Never disable input sanitization
# Sanitization is always enabled

# 2. Use stop_on_error for critical systems
config = AgentConfig(stop_on_error=True)

# 3. Monitor verbose logs in production
config = AgentConfig(verbose=True)

# 4. Set appropriate timeouts
config = AgentConfig(timeout_seconds=300)

# 5. Use conservative research settings
config = AgentConfig(
    thinking_depth=2,  # Not 5
    thinking_web_search_results=3  # Not 10
)
```

## Testing

### Comprehensive Test Suite

FourierSDK includes extensive unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run thinking mode tests
python -m pytest tests/test_thinking_mode.py

# Run with coverage
python -m pytest tests/ --cov=agent --cov-report=html
```

### Test Coverage

- **Configuration Validation**: 12 tests
- **Input Sanitization**: 10 tests
- **Context Management**: 3 tests
- **Research Functionality**: 5 tests
- **Query Generation**: 5 tests
- **Search Execution**: 5 tests

**Total: 40+ unit tests**

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: Component interaction testing
3. **Error Tests**: Failure scenario testing
4. **Security Tests**: Input validation testing
5. **Performance Tests**: Timing and rate limit testing

### Running Tests

```python
# From Python
from tests.test_thinking_mode import run_tests
run_tests()

# From command line
python tests/test_thinking_mode.py

# With pytest
pytest tests/test_thinking_mode.py -v
```

## Best Practices

### 1. Configuration

```python
# Production configuration
production_config = AgentConfig(
    thinking_mode=True,
    thinking_depth=2,               # Conservative
    thinking_web_search_results=5,  # Balanced
    verbose=True,                    # Monitor performance
    stop_on_error=False,            # Graceful degradation
    max_iterations=10,              # Prevent infinite loops
    timeout_seconds=300             # 5-minute timeout
)
```

### 2. Error Handling

```python
# Always check response success
response = agent.run("Query")

if response["success"]:
    print(response["output"])
else:
    print(f"Error: {response.get('error')}")
    # Handle error appropriately
```

### 3. Logging

```python
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
```

### 4. Resource Management

```python
# Use context managers where possible
with Agent(client=client, config=config) as agent:
    response = agent.run("Query")
    # Resources cleaned up automatically
```

### 5. Monitoring

```python
# Track metrics in production
import time

start = time.time()
response = agent.run("Query")
elapsed = time.time() - start

# Log metrics
logger.info(f"Query completed in {elapsed:.2f}s")
logger.info(f"Tool calls: {response['tool_calls']}")
logger.info(f"Iterations: {response['iterations']}")
```

### 6. Validation

```python
# Validate inputs before processing
def validate_query(query: str) -> bool:
    """Validate query before passing to agent."""
    if not query or not isinstance(query, str):
        return False
    if len(query) < 5 or len(query) > 1000:
        return False
    return True

if validate_query(user_query):
    response = agent.run(user_query)
else:
    print("Invalid query")
```

## Constants Reference

### Global Safety Constants

```python
MAX_THINKING_DEPTH = 5              # Maximum research depth
MIN_THINKING_DEPTH = 1              # Minimum research depth
MAX_SEARCH_RESULTS = 10             # Maximum results per search
MIN_SEARCH_RESULTS = 1              # Minimum results per search
MAX_CONTEXT_LENGTH = 50000          # Max context chars
SEARCH_TIMEOUT = 30                 # Search timeout seconds
QUERY_GENERATION_TIMEOUT = 15       # Query gen timeout seconds
MAX_QUERY_LENGTH = 500              # Max query length chars
RATE_LIMIT_DELAY = 1.0              # Delay between searches
```

### Modifying Constants

```python
# For special use cases, modify constants carefully
import agent

# Increase context limit for specialized applications
agent.MAX_CONTEXT_LENGTH = 100000

# Adjust rate limiting
agent.RATE_LIMIT_DELAY = 0.5  # Faster (use cautiously)

# Modify timeouts
agent.SEARCH_TIMEOUT = 60  # Longer timeout
```

## Performance Characteristics

### Typical Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Query Generation | 2-5s | Depends on LLM speed |
| Single Search | 1-3s | Depends on network |
| Full Research (depth=2) | 5-10s | Includes rate limiting |
| Full Research (depth=5) | 15-25s | Includes rate limiting |

### Optimization Tips

1. **Reduce thinking_depth**: Lower depth = faster
2. **Reduce search_results**: Fewer results = faster
3. **Disable verbose**: Less logging overhead
4. **Adjust rate_limit**: Lower delay = faster (use cautiously)
5. **Use caching**: Cache repeated queries

## Troubleshooting

### Common Issues

**Issue**: Queries being rejected
- **Cause**: Invalid characters or length
- **Solution**: Check input sanitization logs

**Issue**: Empty research results
- **Cause**: All searches failed
- **Solution**: Check network, API keys, verbose logs

**Issue**: Context truncation
- **Cause**: Results exceed 50,000 chars
- **Solution**: Reduce search_results or depth

**Issue**: Slow performance
- **Cause**: High depth, network latency
- **Solution**: Reduce depth, check network

### Debug Checklist

1. ✅ Enable verbose logging
2. ✅ Check configuration validation warnings
3. ✅ Review sanitization logs
4. ✅ Monitor timing metrics
5. ✅ Check error logs
6. ✅ Validate API keys
7. ✅ Test network connectivity

## Conclusion

FourierSDK's production-grade features ensure:

- ✅ **Reliability**: Robust error handling
- ✅ **Security**: Input validation and sanitization
- ✅ **Performance**: Rate limiting and optimization
- ✅ **Observability**: Comprehensive logging
- ✅ **Maintainability**: Extensive test coverage

These features make FourierSDK suitable for production deployments in enterprise environments.

## Support

For issues or questions:
- GitHub Issues: https://github.com/Fourier-AI/SDK-main/issues
- Documentation: See README.md, AGENT.md, MCP.md
- Tests: See tests/test_thinking_mode.py
