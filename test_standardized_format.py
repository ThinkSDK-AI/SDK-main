from think_standardized import Think
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import json

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

def test_standard_response(provider, output_file=None):
    """Test the standard response format for a provider"""
    print(f"\n=== Testing {provider.capitalize()} Provider (Standardized Format) ===")
    
    # Get the API key environment variable name
    api_key_var = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(api_key_var)
    
    if not api_key:
        print(f"[ERROR] No API key found for {provider}. Set the {api_key_var} environment variable.")
        return
    
    # Initialize the client
    client = Think(
        api_key=api_key,
        provider=provider
    )
    
    # Configure model and prompt based on provider
    config = {
        "groq": {
            "model": "llama-3.1-70b-instruct",
            "prompt": "Write a short poem about artificial intelligence."
        },
        "together": {
            "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "prompt": "Explain quantum computing in simple terms."
        },
        "nebius": {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "prompt": "What are three ways to improve productivity?"
        },
        "openai": {
            "model": "gpt-4o",
            "prompt": "Recommend three books on machine learning for beginners."
        },
        "anthropic": {
            "model": "claude-3-opus-20240229",
            "prompt": "Explain how blockchain technology works."
        },
        "perplexity": {
            "model": "sonar",
            "prompt": "What were the key technological advancements in 2023?"
        }
    }.get(provider.lower(), {"model": "", "prompt": "Hello, how are you?"})
    
    # Make the API request
    response = client.chat(
        model=config["model"],
        messages=[
            {"role": "user", "content": config["prompt"]}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    # Print formatted response
    print("\n=== STANDARDIZED JSON RESPONSE ===")
    formatted_json = json.dumps(response, indent=2)
    print(formatted_json)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(formatted_json)
        print(f"\nResponse saved to {output_file}")
    
    # Print a summary of key fields
    print("\n=== RESPONSE SUMMARY ===")
    print(f"Status: {response.get('status', 'unknown')}")
    print(f"Provider: {response.get('metadata', {}).get('provider', 'unknown')}")
    print(f"Model: {response.get('metadata', {}).get('model', 'unknown')}")
    print(f"Response Type: {response.get('metadata', {}).get('response_type', 'unknown')}")
    print(f"Request ID: {response.get('metadata', {}).get('request_id', 'unknown')}")
    print(f"Latency: {response.get('metadata', {}).get('latency_ms', 0)}ms")
    
    # Print token usage
    usage = response.get('usage', {})
    print(f"Token Usage: {usage.get('input_tokens', 0)} input / {usage.get('output_tokens', 0)} output / {usage.get('total_tokens', 0)} total")
    
    # Print output content (truncated if too long)
    output = response.get('response', {}).get('output', '')
    if len(output) > 150:
        print(f"\nOutput (truncated): {output[:150]}...")
    else:
        print(f"\nOutput: {output}")
    
    # Print citations if available
    citations = response.get('response', {}).get('citations', [])
    if citations:
        print("\nCitations:")
        for i, citation in enumerate(citations[:3], 1):
            print(f"  {i}. {citation.get('url', '')}")
        if len(citations) > 3:
            print(f"  ... and {len(citations) - 3} more")
    
    print("=" * 50)
    
    return response

if __name__ == "__main__":
    # By default, test Together and Perplexity as they have different response formats
    # but can add others as needed
    test_standard_response("together", "together_response.json")
    test_standard_response("nebius", "nebius_response.json")
    test_standard_response("perplexity", "perplexity_response.json")
    
    # Uncomment to test other providers
    # test_standard_response("groq", "groq_response.json")
    # test_standard_response("openai", "openai_response.json")
    # test_standard_response("anthropic", "anthropic_response.json")
