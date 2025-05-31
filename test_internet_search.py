from think_standardized import Think
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("TOGETHER_API_KEY")

def test_internet_search():
    """
    Test the internet search feature in the Think SDK.
    """
    # Initialize the Think SDK with Together AI provider
    think = Think(api_key=API_KEY, provider="together")
    
    # Example user query that would benefit from internet search
    query = "What are the latest developments in quantum computing?"
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    
    # Make a chat completion request with internet search enabled
    response = think.chat(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",  # Use the model that works with test_providers.py
        messages=messages,
        temperature=0.7,
        internet_search=True,  # Enable internet search
        search_results=5  # Get top 5 search results
    )
    
    # Print the response
    print("\n===== RESPONSE WITH INTERNET SEARCH =====")
    print(f"Query: {query}")
    print("\nResponse output:")
    print(response["response"]["output"])
    
    # Print citations if available
    if "citations" in response["response"]:
        print("\nSources:")
        for i, citation in enumerate(response["response"]["citations"], 1):
            print(f"{i}. {citation['title']} - {citation['url']}")
    
    # Print search metadata
    if "search_metadata" in response["response"]:
        print("\nSearch metadata:")
        print(json.dumps(response["response"]["search_metadata"], indent=2))
    
    # Compare with a response without internet search
    print("\n\n===== RESPONSE WITHOUT INTERNET SEARCH =====")
    standard_response = think.chat(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        messages=messages,
        temperature=0.7
    )
    print("\nResponse without search:")
    print(standard_response["response"]["output"])

def test_custom_search_query():
    """
    Test using a custom search query that differs from the user message.
    """
    # Initialize the Think SDK
    think = Think(api_key=API_KEY, provider="together")
    
    # Example where the search query is different from the user query
    user_query = "Tell me about advances in renewable energy in 2023"
    search_query = "latest renewable energy technology breakthroughs 2023"
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_query}
    ]
    
    # Make a chat completion request with custom search query
    response = think.chat(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        messages=messages,
        temperature=0.7,
        internet_search=True,
        search_query=search_query,  # Use custom search query
        search_results=3
    )
    
    # Print the response
    print("\n===== RESPONSE WITH CUSTOM SEARCH QUERY =====")
    print(f"User query: {user_query}")
    print(f"Search query: {search_query}")
    print("\nResponse:")
    print(response["response"]["output"])
    
    if "citations" in response["response"]:
        print("\nSources:")
        for i, citation in enumerate(response["response"]["citations"], 1):
            print(f"{i}. {citation['title']} - {citation['url']}")

if __name__ == "__main__":
    print("Testing internet search feature in Think SDK...")
    test_internet_search()
    print("\n" + "-" * 50 + "\n")
    test_custom_search_query()
