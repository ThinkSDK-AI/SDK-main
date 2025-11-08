"""
Basic AWS Bedrock Usage Examples

This example demonstrates how to use AWS Bedrock with the Fourier SDK using
different authentication methods:
1. IAM credentials (access key + secret key)
2. AWS profile
3. Default AWS credential chain
"""

import os
from dotenv import load_dotenv
from fourier import Fourier

# Load environment variables
load_dotenv()


def example_1_iam_credentials():
    """Example 1: Using IAM credentials (access key + secret key)"""
    print("=" * 60)
    print("Example 1: Bedrock with IAM Credentials")
    print("=" * 60)

    # Initialize Fourier with Bedrock provider using IAM credentials
    client = Fourier(
        api_key=None,  # Not needed for IAM auth
        provider="bedrock",
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )

    # Make a simple chat completion request
    response = client.chat(
        model="claude-3-5-sonnet",  # Friendly name
        messages=[
            {"role": "user", "content": "What is AWS Bedrock?"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    print(f"\nModel: {response['metadata']['model']}")
    print(f"Response: {response['response']['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    print()


def example_2_aws_profile():
    """Example 2: Using AWS profile from ~/.aws/credentials"""
    print("=" * 60)
    print("Example 2: Bedrock with AWS Profile")
    print("=" * 60)

    # Initialize Fourier with AWS profile
    client = Fourier(
        api_key=None,
        provider="bedrock",
        profile_name=os.getenv("AWS_PROFILE", "default"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )

    # Make a chat completion request
    response = client.chat(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Full model ID
        messages=[
            {"role": "user", "content": "Explain the benefits of serverless computing in 3 sentences."}
        ],
        max_tokens=300
    )

    print(f"\nResponse: {response['response']['content']}")
    print()


def example_3_default_credentials():
    """Example 3: Using default AWS credential chain"""
    print("=" * 60)
    print("Example 3: Bedrock with Default Credential Chain")
    print("=" * 60)

    # Initialize Fourier without explicit credentials
    # This will use:
    # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # 2. AWS credentials file (~/.aws/credentials)
    # 3. IAM role (if running on EC2/Lambda)
    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    # Make a chat completion request
    response = client.chat(
        model="claude-3-haiku",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What are the key features of Amazon Bedrock?"}
        ],
        temperature=0.5,
        max_tokens=400
    )

    print(f"\nResponse: {response['response']['content']}")
    print()


def example_4_different_models():
    """Example 4: Using different Bedrock models"""
    print("=" * 60)
    print("Example 4: Testing Different Bedrock Models")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    models_to_test = [
        "claude-3-5-sonnet",
        "claude-3-haiku",
        "llama3-1-8b",
        "mistral-7b",
    ]

    prompt = "Say hello in a unique way."

    for model in models_to_test:
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            print(f"\n{model}:")
            print(f"  Response: {response['response']['content'][:100]}...")
        except Exception as e:
            print(f"\n{model}:")
            print(f"  Error: {str(e)}")

    print()


def example_5_conversation():
    """Example 5: Multi-turn conversation"""
    print("=" * 60)
    print("Example 5: Multi-turn Conversation")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    # Build a conversation
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]

    # First turn
    response = client.chat(
        model="claude-3-5-sonnet",
        messages=messages,
        max_tokens=200
    )

    first_response = response['response']['content']
    print(f"\nUser: {messages[0]['content']}")
    print(f"Assistant: {first_response}")

    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": first_response})

    # Second turn
    messages.append({"role": "user", "content": "Can you give me a simple example?"})

    response = client.chat(
        model="claude-3-5-sonnet",
        messages=messages,
        max_tokens=300
    )

    second_response = response['response']['content']
    print(f"\nUser: {messages[-1]['content']}")
    print(f"Assistant: {second_response}")
    print()


def example_6_with_system_prompt():
    """Example 6: Using system prompts effectively"""
    print("=" * 60)
    print("Example 6: System Prompts")
    print("=" * 60)

    client = Fourier(
        api_key=None,
        provider="bedrock",
        region="us-east-1"
    )

    # Create a specialized assistant with system prompt
    response = client.chat(
        model="claude-3-5-sonnet",
        messages=[
            {
                "role": "system",
                "content": "You are a Python programming expert. Provide concise, "
                          "well-commented code examples. Always explain your code."
            },
            {
                "role": "user",
                "content": "Write a function to calculate fibonacci numbers."
            }
        ],
        max_tokens=500,
        temperature=0.3
    )

    print(f"\nResponse:\n{response['response']['content']}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AWS Bedrock Basic Examples")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_1_iam_credentials()
        example_2_aws_profile()
        example_3_default_credentials()
        example_4_different_models()
        example_5_conversation()
        example_6_with_system_prompt()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have:")
        print("1. boto3 installed: pip install boto3")
        print("2. AWS credentials configured")
        print("3. Access to AWS Bedrock models")
        print("4. Proper IAM permissions for Bedrock")
