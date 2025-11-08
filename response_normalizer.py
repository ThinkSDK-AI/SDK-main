from typing import List, Optional, Dict, Any, Union
import json
import re
import time
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseNormalizer:
    """Normalizes responses from different providers into a standard format."""
    
    @staticmethod
    def normalize(response: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Normalize a response from a specific provider into a standard format.
        
        Args:
            response (Dict[str, Any]): The raw response from the provider
            provider (str): The provider name
            
        Returns:
            Dict[str, Any]: Normalized response with standardized data
        """
        start_time = time.time()
        
        normalizer_map = {
            "meta": ResponseNormalizer._normalize_meta_response,
            "llama": ResponseNormalizer._normalize_meta_response,  # Same format as Meta
            "anthropic": ResponseNormalizer._normalize_anthropic_response,
            "perplexity": ResponseNormalizer._normalize_perplexity_response,
            "openai": ResponseNormalizer._normalize_openai_response,
            "groq": ResponseNormalizer._normalize_groq_response,
            "together": ResponseNormalizer._normalize_together_response,
            "nebius": ResponseNormalizer._normalize_nebius_response,
            "bedrock": ResponseNormalizer._normalize_bedrock_response,
        }
        
        # Get the appropriate normalizer function for this provider
        normalizer = normalizer_map.get(provider.lower(), ResponseNormalizer._normalize_generic_response)
        processed_response = normalizer(response)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Create the standardized response
        if isinstance(processed_response, dict) and "is_tool_response" in processed_response:
            return ResponseNormalizer._create_standard_response_format(
                processed_response, 
                provider,
                latency_ms
            )
        else:
            # If the normalizer didn't return our expected structure, use the original response
            return ResponseNormalizer._create_standard_response_format(
                {
                    "content": ResponseNormalizer._extract_content(response, provider),
                    "is_tool_response": False,
                    "raw_response": response,
                    "model": ResponseNormalizer._extract_model(response, provider),
                    "request_id": ResponseNormalizer._extract_request_id(response, provider),
                    "usage": ResponseNormalizer._extract_usage(response, provider),
                    "references": ResponseNormalizer._extract_references(response, provider),
                },
                provider,
                latency_ms
            )
    
    @staticmethod
    def _normalize_meta_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Meta/Llama response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
                    # Try to extract JSON from the content
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        # Return standardized tool response
                        return {
                            "content": content,
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"],
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                    
                    # If not a tool response, normalize as standard response
                    return {
                        "content": content,
                        "is_tool_response": False,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
        except Exception as e:
            logger.error(f"Error normalizing Meta response: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_perplexity_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Perplexity response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]
                content = message["content"]
                
                # Try to find JSON in the content
                # First check for markdown code blocks
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if json_match:
                    json_str = json_match.group(1).strip()
                    tool_data = json.loads(json_str)
                else:
                    # Try to parse the entire content as JSON
                    tool_data = ResponseNormalizer._extract_json(content)
                
                if tool_data and "tool" in tool_data and "parameters" in tool_data:
                    # Return standardized tool response
                    return {
                        "content": content,
                        "tool": tool_data["tool"],
                        "parameters": tool_data["parameters"],
                        "is_tool_response": True,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        },
                        "references": response.get("citations", [])
                    }
                
                # Extract citations/references if present
                references = []
                if "citations" in response:
                    references = response["citations"]
                
                # If not a tool response, normalize as standard response
                return {
                    "content": content,
                    "is_tool_response": False,
                    "raw_response": response,
                    "model": response.get("model", ""),
                    "request_id": response.get("id", ""),
                    "usage": {
                        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                    },
                    "references": references
                }
        except Exception as e:
            logger.error(f"Error normalizing Perplexity response: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_openai_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAI response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                
                # Check if there are tool calls in the response
                if "message" in choice and "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
                    tool_call = choice["message"]["tool_calls"][0]
                    if "function" in tool_call:
                        # Return standardized tool response
                        return {
                            "content": choice["message"].get("content", ""),
                            "tool": tool_call["function"]["name"],
                            "parameters": json.loads(tool_call["function"]["arguments"]),
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                
                # If no tool_calls, try to extract from content
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        return {
                            "content": content,
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"],
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                    
                    # If not a tool response, normalize as standard response
                    return {
                        "content": content,
                        "is_tool_response": False,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
        except Exception as e:
            logger.error(f"Error normalizing OpenAI response: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_anthropic_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Anthropic response"""
        try:
            content_text = ""
            if "content" in response and len(response["content"]) > 0:
                for content_item in response["content"]:
                    if content_item["type"] == "text":
                        content_text += content_item["text"]
                        tool_data = ResponseNormalizer._extract_json(content_item["text"])
                        if tool_data and "tool" in tool_data and "parameters" in tool_data:
                            return {
                                "content": content_text,
                                "tool": tool_data["tool"],
                                "parameters": tool_data["parameters"],
                                "is_tool_response": True,
                                "raw_response": response,
                                "model": response.get("model", ""),
                                "request_id": response.get("id", ""),
                                "usage": {
                                    "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                                    "output_tokens": response.get("usage", {}).get("output_tokens", 0),
                                    "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
                                }
                            }
                    elif content_item["type"] == "tool_use":
                        return {
                            "content": content_text,
                            "tool": content_item["name"],
                            "parameters": content_item["input"],
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("output_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
                            }
                        }
                
                # If we have content text but no tool calls, return standard response
                if content_text:
                    return {
                        "content": content_text,
                        "is_tool_response": False,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("output_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
                        }
                    }
        except Exception as e:
            logger.error(f"Error normalizing Anthropic response: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_groq_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Groq response"""
        # Similar to OpenAI format
        return ResponseNormalizer._normalize_openai_response(response)
    
    @staticmethod
    def _normalize_together_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Together response"""
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
                    # Try to extract JSON from the content
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        # Return standardized tool response
                        return {
                            "content": content,
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"],
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                    
                    # If not a tool response, normalize as standard response
                    return {
                        "content": content,
                        "is_tool_response": False,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
        except Exception as e:
            logger.error(f"Error normalizing Together response: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _normalize_nebius_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Nebius response"""
        # Based on actual response format, similar to other providers
        return ResponseNormalizer._normalize_together_response(response)

    @staticmethod
    def _normalize_bedrock_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize AWS Bedrock response"""
        try:
            # Bedrock responses are already processed by the provider
            # They come in a standardized format from BedrockProvider.process_response()
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # Check for tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]
                    function = tool_call.get("function", {})

                    try:
                        parameters = json.loads(function.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        parameters = {}

                    return {
                        "content": content,
                        "tool": function.get("name", ""),
                        "parameters": parameters,
                        "is_tool_response": True,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }

                # Standard text response
                return {
                    "content": content,
                    "is_tool_response": False,
                    "raw_response": response,
                    "model": response.get("model", ""),
                    "request_id": response.get("id", ""),
                    "usage": {
                        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                    }
                }
        except Exception as e:
            logger.error(f"Error normalizing Bedrock response: {e}", exc_info=True)

        # Return original response if normalization fails
        return response

    @staticmethod
    def _normalize_generic_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Generic response normalizer for providers without specific handling"""
        try:
            # Try to extract tool data from common response patterns
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    tool_data = ResponseNormalizer._extract_json(content)
                    if tool_data and "tool" in tool_data and "parameters" in tool_data:
                        return {
                            "content": content,
                            "tool": tool_data["tool"],
                            "parameters": tool_data["parameters"],
                            "is_tool_response": True,
                            "raw_response": response,
                            "model": response.get("model", ""),
                            "request_id": response.get("id", ""),
                            "usage": {
                                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                    
                    # If not a tool response, normalize as standard response
                    return {
                        "content": content,
                        "is_tool_response": False,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
            
            # Try direct content
            if "content" in response:
                content = response["content"]
                tool_data = ResponseNormalizer._extract_json(content)
                if tool_data and "tool" in tool_data and "parameters" in tool_data:
                    return {
                        "content": content,
                        "tool": tool_data["tool"],
                        "parameters": tool_data["parameters"],
                        "is_tool_response": True,
                        "raw_response": response,
                        "model": response.get("model", ""),
                        "request_id": response.get("id", ""),
                        "usage": {
                            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
                
                # If not a tool response, normalize as standard response
                return {
                    "content": content,
                    "is_tool_response": False,
                    "raw_response": response,
                    "model": response.get("model", ""),
                    "request_id": response.get("id", ""),
                    "usage": {
                        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                    }
                }
        except Exception as e:
            logger.error(f"Error in generic normalizer: {e}", exc_info=True)
        
        # Return original response if no tool data found
        return response
    
    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """Helper method to extract JSON from a string content"""
        try:
            # First check if the entire content is valid JSON
            if content.strip().startswith("{") and content.strip().endswith("}"):
                return json.loads(content)
            
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # Try to find JSON with regex for more complex content
            json_match = re.search(r"({[\s\S]*})", content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
        
        return None
    
    @staticmethod
    def _extract_content(response: Dict[str, Any], provider: str) -> str:
        """Extract content from any provider response"""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        if "content" in response:
            if isinstance(response["content"], str):
                return response["content"]
            elif isinstance(response["content"], list):
                # Handle Anthropic style responses
                content_text = ""
                for item in response["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content_text += item.get("text", "")
                return content_text
        
        return ""
    
    @staticmethod
    def _extract_model(response: Dict[str, Any], provider: str) -> str:
        """Extract model from any provider response"""
        return response.get("model", "")
    
    @staticmethod
    def _extract_request_id(response: Dict[str, Any], provider: str) -> str:
        """Extract request ID from any provider response"""
        return response.get("id", str(uuid.uuid4()))
    
    @staticmethod
    def _extract_usage(response: Dict[str, Any], provider: str) -> Dict[str, int]:
        """Extract token usage from any provider response"""
        usage = response.get("usage", {})
        
        # Different providers use different keys for token counts
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
    
    @staticmethod
    def _extract_references(response: Dict[str, Any], provider: str) -> List[Dict[str, str]]:
        """Extract references/citations from provider responses"""
        if provider.lower() == "perplexity" and "citations" in response:
            return [{"url": url} for url in response["citations"]]
        return []
    
    @staticmethod
    def _create_standard_response_format(
        processed_response: Dict[str, Any],
        provider: str,
        latency_ms: int
    ) -> Dict[str, Any]:
        """Create the final standardized response format"""
        # Get current timestamp in ISO format
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Initialize standard format
        standard_format = {
            "status": "success",
            "timestamp": timestamp,
            "metadata": {
                "request_id": processed_response.get("request_id", str(uuid.uuid4())),
                "model": processed_response.get("model", ""),
                "provider": provider,
                "response_type": "tool_call" if processed_response.get("is_tool_response", False) else "text",
                "latency_ms": latency_ms
            },
            "response": {},
            "usage": {
                "input_tokens": processed_response.get("usage", {}).get("input_tokens", 0),
                "output_tokens": processed_response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": processed_response.get("usage", {}).get("total_tokens", 0)
            },
            "error": None
        }
        
        # Handle tool call responses
        if processed_response.get("is_tool_response", False):
            standard_format["response"] = {
                "type": "tool_call",
                "output": processed_response.get("content", ""),
                "tool_used": processed_response.get("tool", ""),
                "tool_parameters": processed_response.get("parameters", {}),
                "tool_execution_output": {}  # This would be filled by the tool execution logic
            }
        else:
            # Handle standard text responses
            standard_format["response"] = {
                "type": "text",
                "output": processed_response.get("content", "")
            }
            
            # Add references if available
            if "references" in processed_response and processed_response["references"]:
                # Format references
                citations = []
                for ref in processed_response["references"]:
                    if isinstance(ref, str):
                        citations.append({
                            "url": ref,
                            "title": "",
                            "snippet": ""
                        })
                    elif isinstance(ref, dict):
                        citations.append({
                            "url": ref.get("url", ""),
                            "title": ref.get("title", ""),
                            "snippet": ref.get("snippet", "")
                        })
                
                standard_format["response"]["citations"] = citations
            
            # Add search metadata for Perplexity
            if provider.lower() == "perplexity":
                standard_format["response"]["search_metadata"] = {
                    "engine": "perplexity",
                    "recency_filter": processed_response.get("raw_response", {}).get("search_recency_filter", "day"),
                    "num_results": len(processed_response.get("references", []))
                }
        
        return standard_format
