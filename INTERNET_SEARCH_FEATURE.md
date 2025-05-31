# Internet Search Feature for ThinkSDK

## Overview

The Internet Search feature enhances the Think SDK with the ability to automatically search the web to provide context-enriched responses. When enabled, the SDK performs a search using DuckDuckGo, extracts content from the top results, and passes this information to the language model along with the user's prompt, allowing for more up-to-date and factually grounded responses.

## How It Works

1. **Search Engine Integration**: The SDK uses DuckDuckGo for web searches, which doesn't require an API key.
2. **Content Extraction**: For each search result, the SDK extracts and cleans the page content.
3. **Context Enrichment**: The extracted information is formatted and added to the system message sent to the LLM.
4. **Citation Support**: The SDK also adds citation information to the response, allowing the source of information to be tracked.

## Usage

### Basic Usage

```python
from think_standardized import Think

# Initialize the SDK
think = Think(api_key="your_api_key", provider="openai")  # or any supported provider

# Create a chat completion with internet search enabled
response = think.chat(
    model="gpt-3.5-turbo",  # or any model supported by your provider
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the latest developments in quantum computing?"}
    ],
    temperature=0.7,
    internet_search=True  # Enable internet search
)

# Print the response
print(response["response"]["output"])

# Print citations
if "citations" in response["response"]:
    for citation in response["response"]["citations"]:
        print(f"Source: {citation['title']} - {citation['url']}")
```

### Advanced Options

The internet search feature includes several customizable parameters:

- `internet_search` (bool): Enable/disable the feature (default: False)
- `search_query` (str, optional): Specify a custom search query instead of using the last user message
- `search_results` (int, optional): Number of search results to include (default: 3)

Example with custom search query:

```python
response = think.chat(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about renewable energy advancements."}
    ],
    internet_search=True,
    search_query="latest breakthroughs in solar and wind energy technology 2023",
    search_results=5  # Get top 5 search results
)
```

## Response Format

When internet search is enabled, the standardized response will include additional fields:

```json
{
  "response": {
    "type": "text",
    "output": "The response text from the LLM...",
    "citations": [
      {
        "url": "https://example.com/article1",
        "title": "Example Article Title",
        "snippet": "Brief description of the article..."
      },
      // Additional citations...
    ],
    "search_metadata": {
      "query": "The search query used",
      "num_results": 3,
      "engine": "duckduckgo"
    }
  },
  // Other standard response fields...
}
```

## Limitations

- The search functionality uses HTML parsing and may break if DuckDuckGo changes their page structure
- Content extraction may not work perfectly on all websites
- Some websites may block automated requests
- The feature adds processing time to requests due to web scraping

## Dependencies

The feature requires these additional Python packages:
- `requests`
- `beautifulsoup4`
