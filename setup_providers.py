#!/usr/bin/env python3
"""
Interactive provider configuration script for Fourier SDK.

This script helps users configure only the providers they need,
install required dependencies, and set up environment variables.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Set


class ProviderConfig:
    """Configuration for each provider"""

    PROVIDERS = {
        "groq": {
            "name": "Groq",
            "description": "Fast inference with Llama, Mixtral, and Gemma models",
            "env_vars": ["GROQ_API_KEY"],
            "dependencies": [],
            "signup_url": "https://console.groq.com/",
        },
        "together": {
            "name": "Together AI",
            "description": "Access to 50+ open-source models",
            "env_vars": ["TOGETHER_API_KEY"],
            "dependencies": [],
            "signup_url": "https://www.together.ai/",
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT-4, GPT-3.5, and other OpenAI models",
            "env_vars": ["OPENAI_API_KEY"],
            "dependencies": [],
            "signup_url": "https://platform.openai.com/",
        },
        "anthropic": {
            "name": "Anthropic",
            "description": "Claude models (Opus, Sonnet, Haiku)",
            "env_vars": ["ANTHROPIC_API_KEY"],
            "dependencies": [],
            "signup_url": "https://console.anthropic.com/",
        },
        "perplexity": {
            "name": "Perplexity",
            "description": "LLMs with built-in web search",
            "env_vars": ["PERPLEXITY_API_KEY"],
            "dependencies": [],
            "signup_url": "https://www.perplexity.ai/",
        },
        "nebius": {
            "name": "Nebius",
            "description": "European AI infrastructure provider",
            "env_vars": ["NEBIUS_API_KEY"],
            "dependencies": [],
            "signup_url": "https://nebius.ai/",
        },
        "bedrock": {
            "name": "AWS Bedrock",
            "description": "40+ models: Claude, Llama, Titan, Mistral, Cohere, AI21",
            "env_vars": [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION",
            ],
            "optional_vars": [
                "AWS_PROFILE",
                "BEDROCK_USE_CROSS_REGION",
                "BEDROCK_USE_GLOBAL_INFERENCE",
            ],
            "dependencies": ["boto3>=1.28.0"],
            "signup_url": "https://aws.amazon.com/bedrock/",
        },
    }

    FEATURES = {
        "search": {
            "name": "Web Search",
            "description": "Internet search integration for all providers",
            "dependencies": ["beautifulsoup4>=4.12.0"],
        },
    }


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print a formatted subheader"""
    print(f"\n{text}")
    print("-" * 70)


def get_user_choice(prompt: str, options: List[str], allow_multiple: bool = False) -> List[str]:
    """Get user selection from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    if allow_multiple:
        print("\nEnter numbers separated by commas (e.g., 1,3,5) or 'all' for all options:")
    else:
        print("\nEnter the number of your choice:")

    while True:
        try:
            choice = input("> ").strip().lower()

            if allow_multiple and choice == "all":
                return options

            if allow_multiple:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [options[i] for i in indices if 0 <= i < len(options)]
                if selected:
                    return selected
            else:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return [options[index]]

            print("Invalid choice. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers.")


def select_providers() -> Set[str]:
    """Interactive provider selection"""
    print_header("Provider Selection")

    print("\nAvailable providers:")
    provider_list = []
    for key, config in ProviderConfig.PROVIDERS.items():
        provider_list.append(key)
        print(f"\n  {len(provider_list)}. {config['name']}")
        print(f"     {config['description']}")
        if config.get("dependencies"):
            print(f"     Dependencies: {', '.join(config['dependencies'])}")

    print("\n\nWhich providers would you like to use?")
    print("Enter numbers separated by commas (e.g., 1,3,7) or 'all' for all providers:")

    while True:
        try:
            choice = input("> ").strip().lower()

            if choice == "all":
                return set(provider_list)

            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = {provider_list[i] for i in indices if 0 <= i < len(provider_list)}

            if selected:
                print(f"\nSelected providers: {', '.join(selected)}")
                confirm = input("Confirm? (y/n): ").strip().lower()
                if confirm == "y":
                    return selected

            print("Invalid choice. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers separated by commas.")


def select_features() -> Set[str]:
    """Interactive feature selection"""
    print_header("Optional Features")

    print("\nAvailable features:")
    feature_list = []
    for key, config in ProviderConfig.FEATURES.items():
        feature_list.append(key)
        print(f"\n  {len(feature_list)}. {config['name']}")
        print(f"     {config['description']}")
        if config.get("dependencies"):
            print(f"     Dependencies: {', '.join(config['dependencies'])}")

    print("\n\nWould you like to enable optional features?")
    print("Enter numbers separated by commas, 'all' for all features, or 'none' to skip:")

    while True:
        choice = input("> ").strip().lower()

        if choice == "none":
            return set()

        if choice == "all":
            return set(feature_list)

        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = {feature_list[i] for i in indices if 0 <= i < len(feature_list)}

            if selected or choice == "":
                print(f"\nSelected features: {', '.join(selected) if selected else 'None'}")
                confirm = input("Confirm? (y/n): ").strip().lower()
                if confirm == "y":
                    return selected

            print("Invalid choice. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid numbers separated by commas.")


def install_dependencies(providers: Set[str], features: Set[str]):
    """Install required dependencies for selected providers and features"""
    print_header("Installing Dependencies")

    # Collect all dependencies
    dependencies = set()

    for provider in providers:
        config = ProviderConfig.PROVIDERS.get(provider, {})
        dependencies.update(config.get("dependencies", []))

    for feature in features:
        config = ProviderConfig.FEATURES.get(feature, {})
        dependencies.update(config.get("dependencies", []))

    if not dependencies:
        print("\n✓ No additional dependencies needed for selected providers.")
        return

    print(f"\nThe following packages will be installed:")
    for dep in dependencies:
        print(f"  - {dep}")

    confirm = input("\nProceed with installation? (y/n): ").strip().lower()
    if confirm != "y":
        print("Skipping dependency installation.")
        return

    print("\nInstalling dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            print(f"  You can install it manually with: pip install {dep}")


def configure_env_vars(providers: Set[str]):
    """Generate .env file with selected providers"""
    print_header("Environment Configuration")

    env_path = Path(".env")

    if env_path.exists():
        print(f"\n⚠ {env_path} already exists.")
        choice = input("Overwrite (o), Append (a), or Skip (s)? ").strip().lower()

        if choice == "s":
            print("Skipping .env configuration.")
            return
        elif choice == "a":
            mode = "a"
            print("\nAppending configuration to existing .env file...")
        else:
            mode = "w"
            print("\nOverwriting .env file...")
    else:
        mode = "w"
        print(f"\nCreating {env_path}...")

    with open(env_path, mode) as f:
        if mode == "w":
            f.write("# Fourier SDK Configuration\n")
            f.write("# Generated by setup_providers.py\n\n")

        for provider in sorted(providers):
            config = ProviderConfig.PROVIDERS.get(provider, {})
            f.write(f"\n# {config['name']}\n")
            f.write(f"# Sign up: {config.get('signup_url', 'N/A')}\n")

            for env_var in config.get("env_vars", []):
                # Generate appropriate placeholder
                if "KEY" in env_var:
                    placeholder = f"your_{provider}_api_key_here"
                elif "REGION" in env_var:
                    placeholder = "us-east-1"
                else:
                    placeholder = f"your_{env_var.lower()}_here"

                f.write(f"{env_var}={placeholder}\n")

            # Add optional vars as comments
            for env_var in config.get("optional_vars", []):
                f.write(f"# {env_var}=\n")

    print(f"\n✓ Environment configuration written to {env_path}")
    print("\n⚠ IMPORTANT: Edit the .env file and add your actual API keys!")
    print("  Never commit the .env file to version control.")


def create_provider_config_file(providers: Set[str], features: Set[str]):
    """Create a config file to track enabled providers"""
    config_path = Path(".fourier_config")

    config_data = {
        "providers": sorted(list(providers)),
        "features": sorted(list(features)),
    }

    import json
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n✓ Provider configuration saved to {config_path}")


def print_next_steps(providers: Set[str]):
    """Print instructions for next steps"""
    print_header("Setup Complete!")

    print("\n✅ Next Steps:\n")
    print("1. Edit the .env file and add your API keys:")
    print("   - Open .env in your text editor")
    print("   - Replace placeholder values with your actual API keys")

    print("\n2. Get API keys from these URLs:")
    for provider in sorted(providers):
        config = ProviderConfig.PROVIDERS.get(provider, {})
        print(f"   - {config['name']}: {config.get('signup_url', 'N/A')}")

    print("\n3. Test your configuration:")
    print("   python -c \"from fourier import Fourier; print('✓ SDK imported successfully')\"")

    print("\n4. Try the interactive CLI:")
    print("   python cli.py interactive")

    print("\n5. See examples:")
    print("   ls examples/")

    print("\n📚 Documentation:")
    print("   - README.md - Main documentation")
    print("   - BEDROCK.md - AWS Bedrock guide (if selected)")
    print("   - CLI.md - CLI reference")


def main():
    """Main setup process"""
    print_header("Fourier SDK - Provider Setup")

    print("\nThis script will help you:")
    print("  ✓ Select the LLM providers you want to use")
    print("  ✓ Install required dependencies")
    print("  ✓ Configure environment variables")

    input("\nPress Enter to continue...")

    # Step 1: Select providers
    providers = select_providers()

    # Step 2: Select features
    features = select_features()

    # Step 3: Install dependencies
    install_dependencies(providers, features)

    # Step 4: Configure environment
    configure_env_vars(providers)

    # Step 5: Save configuration
    create_provider_config_file(providers, features)

    # Step 6: Show next steps
    print_next_steps(providers)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
