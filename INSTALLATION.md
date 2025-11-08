# Fourier SDK Installation Guide

This guide covers different installation methods for the Fourier SDK, including how to install only the providers you need.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Provider-Specific Installation](#provider-specific-installation)
- [Interactive Setup](#interactive-setup)
- [Manual Configuration](#manual-configuration)
- [Verification](#verification)

---

## Quick Start

### Option 1: Install Everything (Recommended for First-Time Users)

```bash
# Clone the repository
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main

# Install with all providers
pip install -e .[all]

# Run interactive setup
python setup_providers.py
```

### Option 2: Install Only What You Need

```bash
# Clone the repository
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main

# Install base requirements only
pip install -e .

# Run interactive setup to select providers
python setup_providers.py
```

---

## Installation Methods

### Method 1: Install from Source (Development)

```bash
# Clone repository
git clone https://github.com/ThinkSDK-AI/SDK-main.git
cd SDK-main

# Install in editable mode
pip install -e .
```

### Method 2: Install from PyPI (When Published)

```bash
# Install base package
pip install fourier-sdk

# Or install with all providers
pip install fourier-sdk[all]
```

### Method 3: Install from requirements.txt

```bash
# Install only base requirements
pip install -r requirements-base.txt

# Or install base + all provider dependencies
pip install -r requirements.txt
```

---

## Provider-Specific Installation

The SDK supports modular installation - only install dependencies for the providers you'll actually use:

### Available Providers (No Extra Dependencies)

These providers work with just the base installation:

- **Groq** - Fast inference with Llama and Mixtral
- **Together AI** - 50+ open-source models
- **OpenAI** - GPT-4, GPT-3.5 Turbo
- **Anthropic** - Claude models
- **Perplexity** - LLMs with web search
- **Nebius** - European AI infrastructure

```bash
# Base installation includes all these providers
pip install -e .
```

### AWS Bedrock Provider

Requires `boto3` for AWS SDK:

```bash
# Install with Bedrock support
pip install -e .[bedrock]

# Or install boto3 manually
pip install boto3>=1.28.0
```

### Web Search Feature

Requires `beautifulsoup4` for web scraping:

```bash
# Install with web search support
pip install -e .[search]

# Or install beautifulsoup4 manually
pip install beautifulsoup4>=4.12.0
```

### Install Multiple Features

```bash
# Bedrock + Web Search
pip install -e .[bedrock,search]

# All features
pip install -e .[all]
```

### Development Installation

```bash
# Install with development tools
pip install -e .[dev]

# This includes: pytest, pytest-cov, black, flake8, mypy
```

---

## Interactive Setup

The SDK includes an interactive setup wizard that guides you through provider selection and configuration:

```bash
python setup_providers.py
```

### What the Setup Wizard Does:

1. **Provider Selection** - Choose which LLM providers you want to use
2. **Dependency Installation** - Automatically installs required packages
3. **Environment Configuration** - Creates .env file with placeholders
4. **Feature Selection** - Enable optional features like web search
5. **Configuration Saving** - Saves your preferences to `.fourier_config`

### Example Setup Session:

```
======================================================================
  Fourier SDK - Provider Setup
======================================================================

Available providers:

  1. Groq
     Fast inference with Llama, Mixtral, and Gemma models

  2. Together AI
     Access to 50+ open-source models

  3. OpenAI
     GPT-4, GPT-3.5, and other OpenAI models

  4. Anthropic
     Claude models (Opus, Sonnet, Haiku)

  5. Perplexity
     LLMs with built-in web search

  6. Nebius
     European AI infrastructure provider

  7. AWS Bedrock
     40+ models: Claude, Llama, Titan, Mistral, Cohere, AI21
     Dependencies: boto3>=1.28.0

Which providers would you like to use?
Enter numbers separated by commas (e.g., 1,3,7) or 'all' for all providers:
> 1,4,7

Selected providers: groq, anthropic, bedrock
Confirm? (y/n): y

======================================================================
  Installing Dependencies
======================================================================

The following packages will be installed:
  - boto3>=1.28.0

Proceed with installation? (y/n): y

Installing dependencies...
✓ Installed boto3>=1.28.0

======================================================================
  Environment Configuration
======================================================================

Creating .env...

✓ Environment configuration written to .env

⚠ IMPORTANT: Edit the .env file and add your actual API keys!
```

---

## Manual Configuration

### 1. Create .env File

Create a `.env` file in the project root:

```bash
# For Groq
GROQ_API_KEY=your_groq_api_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For AWS Bedrock (Option 1: IAM Credentials)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# For AWS Bedrock (Option 2: Profile - alternative to keys)
# AWS_PROFILE=your_aws_profile_name
```

### 2. Get API Keys

- **Groq**: https://console.groq.com/
- **Together AI**: https://www.together.ai/
- **OpenAI**: https://platform.openai.com/
- **Anthropic**: https://console.anthropic.com/
- **Perplexity**: https://www.perplexity.ai/
- **Nebius**: https://nebius.ai/
- **AWS Bedrock**: https://aws.amazon.com/bedrock/

### 3. Install Provider Dependencies

```bash
# If using Bedrock
pip install boto3>=1.28.0

# If using web search
pip install beautifulsoup4>=4.12.0
```

---

## Verification

### Verify Installation

```bash
# Test import
python -c "from fourier import Fourier; print('✓ SDK imported successfully')"

# Check available providers
python -c "from providers import get_available_providers; print('Available:', list(get_available_providers().keys()))"
```

### Test a Provider

```python
from fourier import Fourier
import os
from dotenv import load_dotenv

load_dotenv()

# Test with Groq
client = Fourier(
    api_key=os.getenv("GROQ_API_KEY"),
    provider="groq"
)

response = client.chat(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response['response']['output'])
```

### Check Installed Features

```python
from providers import is_provider_available

# Check if Bedrock is available
if is_provider_available("bedrock"):
    print("✓ Bedrock provider is available")
else:
    print("✗ Bedrock provider not available")
    print("  Install with: pip install fourier-sdk[bedrock]")
```

---

## Troubleshooting

### Issue: "Provider requires additional dependencies"

```
UnsupportedProviderError: Provider 'bedrock' requires additional dependencies.
Install with: pip install fourier-sdk[bedrock]
```

**Solution:**
```bash
pip install fourier-sdk[bedrock]
# or
pip install boto3>=1.28.0
```

### Issue: "No module named 'boto3'"

**Solution:**
```bash
pip install boto3>=1.28.0
```

### Issue: ".env file not found"

**Solution:**
```bash
# Run interactive setup
python setup_providers.py

# Or copy template
cp .env.template .env
```

### Issue: "API key cannot be empty"

**Solution:**
1. Make sure `.env` file exists
2. Add your API key to `.env`
3. Load it with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Installation Options Summary

| Installation Method | Command | Use Case |
|-------------------|---------|----------|
| **Base Only** | `pip install -e .` | Core providers (no extra deps) |
| **With Bedrock** | `pip install -e .[bedrock]` | AWS Bedrock support |
| **With Search** | `pip install -e .[search]` | Web search feature |
| **All Features** | `pip install -e .[all]` | Everything included |
| **Development** | `pip install -e .[dev]` | Development tools |
| **Custom** | `python setup_providers.py` | Interactive selection |

---

## Quick Reference

```bash
# Interactive setup (recommended)
python setup_providers.py

# Install specific provider
pip install -e .[bedrock]

# Install all features
pip install -e .[all]

# Verify installation
python -c "from fourier import Fourier; print('OK')"

# Edit configuration
nano .env

# Test CLI
python cli.py interactive
```

---

## Next Steps

After installation:

1. **Configure Environment** - Add API keys to `.env`
2. **Read Documentation** - Check README.md for usage examples
3. **Try Examples** - Run examples from `examples/` directory
4. **Use CLI** - Try the interactive CLI: `python cli.py interactive`
5. **Provider Docs** - Read provider-specific docs (e.g., BEDROCK.md)

---

## Support

- **GitHub Issues**: https://github.com/ThinkSDK-AI/SDK-main/issues
- **Documentation**: README.md
- **Examples**: `examples/` directory
