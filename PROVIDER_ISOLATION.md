# Provider Isolation Architecture

This document explains the provider isolation system in Fourier SDK, which allows users to install only the dependencies they need for their specific use case.

## Overview

The Fourier SDK now uses a **modular dependency system** where:

1. **Base providers** require no extra dependencies (Groq, OpenAI, Anthropic, Together, Perplexity, Nebius)
2. **Optional providers** require specific packages (AWS Bedrock requires boto3)
3. **Optional features** can be enabled independently (Web search requires beautifulsoup4)

## Architecture

### 1. Dependency Structure

```
fourier-sdk/
├── requirements-base.txt          # Core dependencies (always required)
├── requirements-providers.txt     # Optional provider dependencies
├── requirements.txt               # References base + instructions
└── setup.py                       # Defines extras_require for pip install
```

### 2. Conditional Imports

**File: `providers/__init__.py`**

```python
# Always available (no extra deps)
from providers.groq import GroqProvider
from providers.together import TogetherProvider
# ... other base providers

# Conditional import
BedrockProvider = None
try:
    from providers.bedrock import BedrockProvider
except ImportError:
    logger.debug("BedrockProvider not available")
```

### 3. Dynamic Provider Registration

**File: `providers/__init__.py`**

```python
def get_available_providers():
    """Returns only providers with satisfied dependencies"""
    providers = {
        "groq": GroqProvider,
        "together": TogetherProvider,
        # ... base providers
    }

    if BedrockProvider is not None:
        providers["bedrock"] = BedrockProvider

    return providers
```

### 4. Runtime Provider Detection

**File: `fourier.py`**

```python
# Get available providers dynamically
provider_map = get_available_providers()

if provider not in provider_map:
    # Helpful error message with install instructions
    raise UnsupportedProviderError(
        f"Provider '{provider}' requires additional dependencies. "
        f"Install with: pip install fourier-sdk[{provider}]"
    )
```

## Installation Methods

### Method 1: Base Installation

```bash
pip install -e .
```

**Installs:**
- requests
- pydantic
- typing-extensions
- python-dotenv

**Available Providers:**
- Groq, Together AI, OpenAI, Anthropic, Perplexity, Nebius

### Method 2: With Specific Providers

```bash
# AWS Bedrock
pip install -e .[bedrock]

# Web search
pip install -e .[search]

# Multiple extras
pip install -e .[bedrock,search]
```

### Method 3: All Features

```bash
pip install -e .[all]
```

### Method 4: Interactive Setup

```bash
python setup_providers.py
```

## Interactive Setup Script

**File: `setup_providers.py`**

The interactive setup wizard:

1. **Presents all available providers** with descriptions
2. **Allows user selection** (single or multiple)
3. **Installs required dependencies** automatically
4. **Generates .env file** with selected providers
5. **Saves configuration** to `.fourier_config`

### Example Usage

```bash
$ python setup_providers.py

======================================================================
  Fourier SDK - Provider Selection
======================================================================

Available providers:

  1. Groq
     Fast inference with Llama, Mixtral, and Gemma models

  2. Together AI
     Access to 50+ open-source models

  ...

  7. AWS Bedrock
     40+ models: Claude, Llama, Titan, Mistral, Cohere, AI21
     Dependencies: boto3>=1.28.0

Which providers would you like to use?
Enter numbers separated by commas (e.g., 1,3,7) or 'all':
> 1,7

Selected providers: groq, bedrock
Confirm? (y/n): y

Installing dependencies...
✓ Installed boto3>=1.28.0

✓ Environment configuration written to .env
```

## Benefits

### 1. Reduced Installation Size

| Installation | Size | Dependencies |
|-------------|------|--------------|
| Base only | ~15 MB | 4 packages |
| + Bedrock | ~80 MB | +boto3 (AWS SDK) |
| + Search | ~20 MB | +beautifulsoup4 |
| All | ~85 MB | All packages |

### 2. Faster Installation

Users who don't need AWS Bedrock avoid installing the 65MB boto3 package and its transitive dependencies.

### 3. Cleaner Environments

- No unused dependencies
- Smaller Docker images
- Faster CI/CD pipelines
- Lower security surface area

### 4. Better Error Messages

```python
# Old behavior
ImportError: No module named 'boto3'

# New behavior
UnsupportedProviderError: Provider 'bedrock' requires additional dependencies.
Install with: pip install fourier-sdk[bedrock]
```

## Implementation Details

### setup.py Extras

```python
extras_require = {
    "bedrock": ["boto3>=1.28.0"],
    "search": ["beautifulsoup4>=4.12.0"],
    "all": ["boto3>=1.28.0", "beautifulsoup4>=4.12.0"],
    "dev": ["pytest>=7.0.0", "black>=23.0.0", ...],
}
```

### Provider Detection

```python
from providers import is_provider_available

if is_provider_available("bedrock"):
    # Use Bedrock
    client = Fourier(provider="bedrock", ...)
else:
    # Show install instructions
    print("Install Bedrock: pip install fourier-sdk[bedrock]")
```

### Configuration Storage

**File: `.fourier_config`**

```json
{
  "providers": ["groq", "bedrock"],
  "features": ["search"]
}
```

This file is created by `setup_providers.py` and can be used by the SDK to:
- Remember user preferences
- Show relevant documentation
- Optimize imports

## Migration Guide

### Upgrading from Previous Version

If you previously used:

```bash
pip install -r requirements.txt
```

Now use one of:

```bash
# Option 1: Install everything (same as before)
pip install -e .[all]

# Option 2: Interactive setup (recommended)
python setup_providers.py

# Option 3: Install only what you need
pip install -e .                 # Base
pip install -e .[bedrock]        # + Bedrock
```

## Adding New Providers

To add a new provider with extra dependencies:

### 1. Add to setup.py

```python
extras_require = {
    # ...
    "newprovider": ["special-package>=1.0.0"],
}
```

### 2. Update providers/__init__.py

```python
NewProvider = None
try:
    from providers.newprovider import NewProvider
except ImportError:
    logger.debug("NewProvider not available")

def get_available_providers():
    providers = { ... }

    if NewProvider is not None:
        providers["newprovider"] = NewProvider

    return providers
```

### 3. Update setup_providers.py

```python
PROVIDERS = {
    # ...
    "newprovider": {
        "name": "New Provider",
        "description": "Description here",
        "env_vars": ["NEWPROVIDER_API_KEY"],
        "dependencies": ["special-package>=1.0.0"],
        "signup_url": "https://newprovider.com/",
    },
}
```

## Testing

### Test Base Installation

```bash
# Create clean virtualenv
python -m venv test_env
source test_env/bin/activate

# Install base only
pip install -e .

# Test imports
python -c "from fourier import Fourier; print('OK')"

# Verify Bedrock is NOT available
python -c "from providers import is_provider_available; assert not is_provider_available('bedrock')"
```

### Test Bedrock Installation

```bash
# Install with Bedrock
pip install -e .[bedrock]

# Verify Bedrock is available
python -c "from providers import is_provider_available; assert is_provider_available('bedrock')"

# Test usage
python -c "from fourier import Fourier; client = Fourier(provider='bedrock', region='us-east-1')"
```

## Troubleshooting

### Issue: "Provider requires additional dependencies"

**Error:**
```
UnsupportedProviderError: Provider 'bedrock' requires additional dependencies.
Install with: pip install fourier-sdk[bedrock]
```

**Solution:**
```bash
pip install fourier-sdk[bedrock]
```

### Issue: "No module named 'boto3'"

**Error:**
```python
from providers.bedrock import BedrockProvider
ImportError: No module named 'boto3'
```

**Solution:**

This shouldn't happen with the new architecture (imports are conditional), but if it does:

```bash
pip install boto3>=1.28.0
```

### Issue: Interactive setup fails

**Solution:**

Run manual installation:

```bash
# Install provider dependencies manually
pip install boto3>=1.28.0  # For Bedrock

# Copy environment template
cp .env.template .env

# Edit .env with your API keys
nano .env
```

## Best Practices

### 1. For Development

```bash
# Install everything for development
pip install -e .[all,dev]
```

### 2. For Production

```bash
# Install only what you need
pip install -e .[bedrock,search]
```

### 3. For Docker

```dockerfile
# Multi-stage build for minimal image

# Stage 1: Base
FROM python:3.11-slim AS base
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

# Stage 2: Add Bedrock (if needed)
FROM base AS bedrock
RUN pip install boto3>=1.28.0

# Stage 3: Final
FROM bedrock
COPY . /app
WORKDIR /app
```

### 4. For CI/CD

```yaml
# GitHub Actions example
- name: Install dependencies
  run: |
    pip install -e .
    if [ "${{ matrix.provider }}" == "bedrock" ]; then
      pip install -e .[bedrock]
    fi
```

## Future Enhancements

1. **Auto-detection**: Detect which providers are configured in .env and suggest installations
2. **Provider packages**: Separate each provider into its own installable package
3. **Lazy loading**: Load provider modules only when first used
4. **Plugin system**: Allow third-party provider plugins
5. **Dependency resolution**: Smart dependency management for conflicting versions

## Summary

The provider isolation system makes Fourier SDK:

- ✅ **Flexible** - Install only what you need
- ✅ **Lightweight** - Minimal base installation
- ✅ **User-friendly** - Interactive setup wizard
- ✅ **Production-ready** - Optimized Docker images
- ✅ **Developer-friendly** - Clear error messages
- ✅ **Maintainable** - Easy to add new providers

Users can choose their installation method based on their needs:
- **Quick start**: `pip install -e .[all]`
- **Interactive**: `python setup_providers.py`
- **Minimal**: `pip install -e .`
- **Custom**: `pip install -e .[bedrock,search]`
