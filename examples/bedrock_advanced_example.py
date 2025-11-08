"""
Advanced AWS Bedrock Features

This example demonstrates advanced Bedrock features:
1. Cross-region inference
2. Global inference (inference profiles)
3. Tool calling / function calling
4. Streaming responses (if supported)
5. Model listing and discovery
"""

import os
from dotenv import load_dotenv
from fourier import Fourier
from providers.bedrock import BedrockProvider

# Load environment variables
load_dotenv()


def example_1_cross_region_inference():
    """Example 1: Cross-region inference for high availability"""
    print("=" * 60)
    print("Example 1: Cross-Region Inference")
    print("=" * 60)

    # Initialize with cross-region inference enabled
    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1",
        use_cross_region=True  # Enable cross-region inference
    )

    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": "What is cross-region inference in AWS Bedrock?"}
        ],
        max_tokens=300
    )

    print(f"\nResponse: {response['response']['content']}")
    print("\nNote: Cross-region inference provides high availability by")
    print("automatically routing requests to multiple regions.")
    print()


def example_2_global_inference():
    """Example 2: Using global inference profiles"""
    print("=" * 60)
    print("Example 2: Global Inference Profiles")
    print("=" * 60)

    # Initialize with global inference
    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1",
        use_global_inference=True  # Use global inference profiles
    )

    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": "Explain global inference in 2 sentences."}
        ],
        max_tokens=200
    )

    print(f"\nResponse: {response['response']['content']}")
    print("\nGlobal inference uses inference profiles to route requests")
    print("optimally across AWS regions worldwide.")
    print()


def example_3_specific_inference_profile():
    """Example 3: Using a specific inference profile ID"""
    print("=" * 60)
    print("Example 3: Specific Inference Profile")
    print("=" * 60)

    # Get the profile ID from environment or use a default
    profile_id = os.getenv("BEDROCK_INFERENCE_PROFILE_ID")

    if not profile_id:
        print("\nSkipping: No BEDROCK_INFERENCE_PROFILE_ID set in environment")
        print("Set BEDROCK_INFERENCE_PROFILE_ID to test this feature")
        return

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1",
        inference_profile_id=profile_id
    )

    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": "Hello from a specific inference profile!"}
        ],
        max_tokens=100
    )

    print(f"\nUsing profile: {profile_id}")
    print(f"Response: {response['response']['content']}")
    print()


def example_4_tool_calling():
    """Example 4: Function calling / tool use with Bedrock"""
    print("=" * 60)
    print("Example 4: Tool Calling with Bedrock")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    # Define a tool
    calculator_tool = client.create_tool(
        name="calculator",
        description="Performs basic arithmetic operations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    )

    # Make a request that should trigger tool use
    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": "What is 25 multiplied by 4?"}
        ],
        tools=[calculator_tool],
        max_tokens=500
    )

    print(f"\nResponse: {response['response']}")
    print("\nNote: The model should recognize when to use the calculator tool.")
    print()


def example_5_list_available_models():
    """Example 5: List all available Bedrock models"""
    print("=" * 60)
    print("Example 5: List Available Models")
    print("=" * 60)

    # Create a BedrockProvider instance directly to access utility methods
    provider = BedrockProvider(
        region="us-east-1"
    )

    # List all foundation models
    all_models = provider.list_foundation_models()

    print(f"\nTotal models available: {len(all_models)}")
    print("\nSample models:")
    for i, model in enumerate(all_models[:10]):  # Show first 10
        print(f"{i+1}. {model.get('modelId', 'Unknown')}")
        print(f"   Provider: {model.get('providerName', 'Unknown')}")
        print(f"   Type: {model.get('outputModalities', 'Unknown')}")
        print()

    # Filter by provider
    print("\nClaude models only:")
    claude_models = provider.list_foundation_models(by_provider="Anthropic")
    for model in claude_models[:5]:  # Show first 5 Claude models
        print(f"  - {model.get('modelId', 'Unknown')}")

    print()


def example_6_list_inference_profiles():
    """Example 6: List available inference profiles"""
    print("=" * 60)
    print("Example 6: List Inference Profiles")
    print("=" * 60)

    provider = BedrockProvider(
        region="us-east-1"
    )

    # Get inference profiles
    profiles = provider.get_inference_profiles()

    print(f"\nTotal inference profiles: {len(profiles)}")

    if profiles:
        print("\nAvailable profiles:")
        for i, profile in enumerate(profiles[:5]):  # Show first 5
            print(f"{i+1}. {profile.get('inferenceProfileId', 'Unknown')}")
            print(f"   Status: {profile.get('status', 'Unknown')}")
            print(f"   Type: {profile.get('type', 'Unknown')}")
            print()
    else:
        print("\nNo inference profiles found or feature not available in this region.")

    print()


def example_7_low_level_invoke_model():
    """Example 7: Using low-level InvokeModel API for custom requests"""
    print("=" * 60)
    print("Example 7: Low-Level InvokeModel API")
    print("=" * 60)

    provider = BedrockProvider(
        region="us-east-1"
    )

    # Use InvokeModel for direct access (Claude model format)
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": "Write a haiku about AWS."
            }
        ]
    }

    try:
        response = provider.invoke_model(
            model_id=model_id,
            body=request_body
        )

        print(f"\nRaw response from InvokeModel API:")
        print(f"Content: {response}")
        print("\nThis demonstrates direct model invocation for")
        print("advanced use cases or models not yet in Converse API.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Some models may require different request formats.")

    print()


def example_8_multiregion_setup():
    """Example 8: Multi-region setup for redundancy"""
    print("=" * 60)
    print("Example 8: Multi-Region Setup")
    print("=" * 60)

    regions = ["us-east-1", "us-west-2", "eu-west-1"]

    print("\nTesting availability across multiple regions:")

    for region in regions:
        try:
            client = Fourier(
                api_key=None,
                provider="bedrock",
                region=region
            )

            response = client.chat(
                model="claude-3-haiku",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=50
            )

            print(f"\n✓ {region}: Available")
            print(f"  Sample response: {response['response']['content'][:50]}...")

        except Exception as e:
            print(f"\n✗ {region}: Error - {str(e)[:50]}...")

    print("\nMulti-region setup allows failover between regions.")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AWS Bedrock Advanced Features")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_1_cross_region_inference()
        example_2_global_inference()
        example_3_specific_inference_profile()
        example_4_tool_calling()
        example_5_list_available_models()
        example_6_list_inference_profiles()
        example_7_low_level_invoke_model()
        example_8_multiregion_setup()

        print("=" * 60)
        print("All advanced examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
