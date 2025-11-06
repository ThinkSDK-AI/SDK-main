from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import random
import logging
from constants import (
    USER_AGENTS,
    HTTP_TIMEOUT_SECONDS,
    WEB_SEARCH_DEFAULT_NUM_RESULTS,
    WEB_SEARCH_DEFAULT_MAX_CHARS,
    WEB_SEARCH_DELAY_SECONDS,
    WEB_SEARCH_RATE_LIMIT_DELAY,
    WEB_SEARCH_MAX_CHARS_PER_RESULT
)

logger = logging.getLogger(__name__)

class WebSearch:
    """Handles web search functionality and content extraction from webpages."""
    
    @staticmethod
    def search_duckduckgo(query: str, num_results: int = WEB_SEARCH_DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query and return top results.
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return (default 5)
            
        Returns:
            List[Dict[str, str]]: List of search results with title, url, and description
        """
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        
        # DuckDuckGo HTML search URL
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://duckduckgo.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        try:
            response = requests.get(search_url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = []
            
            # Find all result elements
            results = soup.select(".result")
            
            for result in results[:num_results]:
                # Extract title and URL
                title_elem = result.select_one(".result__a")
                if not title_elem:
                    continue
                    
                title = title_elem.get_text().strip()
                # DuckDuckGo uses redirects, extract the actual URL from the href
                url_path = title_elem.get("href", "")
                if url_path:
                    # Parse URL parameters to get the actual URL
                    parsed_url = urllib.parse.urlparse(url_path)
                    url_params = urllib.parse.parse_qs(parsed_url.query)
                    actual_url = url_params.get("uddg", [""])[0]
                    if not actual_url:
                        continue
                else:
                    continue
                
                # Extract snippet/description
                description_elem = result.select_one(".result__snippet")
                description = description_elem.get_text().strip() if description_elem else ""
                
                search_results.append({
                    "title": title,
                    "url": actual_url,
                    "snippet": description
                })
                
                # Avoid rate limiting
                if len(search_results) < num_results:
                    time.sleep(WEB_SEARCH_RATE_LIMIT_DELAY)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def extract_content(url: str, max_chars: int = WEB_SEARCH_DEFAULT_MAX_CHARS) -> str:
        """
        Extract and clean text content from a URL.
        
        Args:
            url (str): URL to extract content from
            max_chars (int): Maximum number of characters to extract
            
        Returns:
            str: Extracted content
        """
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.extract()
                
            # Get text and clean it
            text = soup.get_text()
            
            # Clean up text: replace multiple newlines, remove extra whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if necessary
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                
            return text
        
        except Exception as e:
            # In case of any error, return a message
            return f"Failed to extract content: {str(e)}"
    
    @staticmethod
    def search_and_extract(query: str, num_results: int = WEB_SEARCH_DEFAULT_NUM_RESULTS, max_chars_per_result: int = WEB_SEARCH_MAX_CHARS_PER_RESULT) -> Dict[str, Any]:
        """
        Search for the query and extract content from found URLs.
        
        Args:
            query (str): Search query
            num_results (int): Number of results to process
            max_chars_per_result (int): Maximum characters to extract per URL
            
        Returns:
            Dict[str, Any]: Search results and extracted content
        """
        # Get search results
        search_results = WebSearch.search_duckduckgo(query, num_results)
        
        # Extract content from each URL
        for result in search_results:
            # Extract content
            extracted_content = WebSearch.extract_content(result["url"], max_chars_per_result)
            result["content"] = extracted_content
            
            # Add small delay between requests
            time.sleep(WEB_SEARCH_DELAY_SECONDS)
        
        return {
            "query": query,
            "results": search_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    
    @staticmethod
    def format_context_for_llm(search_data: Dict[str, Any]) -> str:
        """
        Format the search results into a context string for the LLM.
        
        Args:
            search_data (Dict[str, Any]): Search data from search_and_extract
            
        Returns:
            str: Formatted context string
        """
        context = f"Search results for: {search_data['query']}\n\n"
        
        for i, result in enumerate(search_data.get("results", []), 1):
            context += f"SOURCE {i}: {result.get('title', 'Untitled')}\n"
            context += f"URL: {result.get('url', '')}\n"
            context += f"SNIPPET: {result.get('snippet', '')}\n\n"
            context += f"CONTENT:\n{result.get('content', 'No content extracted')[:500]}...\n\n"
            
            # Add separator between results
            if i < len(search_data.get("results", [])):
                context += "---\n\n"
        
        return context
