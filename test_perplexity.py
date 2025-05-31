import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def test_perplexity():
    """Test Perplexity API directly"""
    print("\n=== Testing Perplexity API ===")
    
    url = "https://api.perplexity.ai/chat/completions"
    
    # Define a weather tool
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use tools to get information."
            },
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ],
        "max_tokens": 123,
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": ["wikipedia.org"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": {
            "type": "text"
        },
        "web_search_options": {"search_context_size": "high"},
        "tools": [weather_tool]
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to Perplexity API...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print("\nResponse Status:", response.status_code)
        print("Response Headers:", response.headers)
        print("\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\nError occurred: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("Response Status:", e.response.status_code)
            print("Response Body:", e.response.text)

if __name__ == "__main__":
    test_perplexity() 