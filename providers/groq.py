from typing import Dict, Any, Optional
from models import ChatCompletionRequest

class GroqProvider:
    """Provider implementation for Groq API."""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload for Groq API."""
        payload = request.model_dump(exclude_none=True)
        return payload
    
    def get_chat_completion_endpoint(self) -> str:
        """Get the chat completion endpoint."""
        return f"{self.BASE_URL}/chat/completions" 