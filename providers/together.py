from typing import Dict, Any, Optional
from models import ChatCompletionRequest
from .base import BaseProvider

class TogetherProvider(BaseProvider):
    """Provider implementation for Together AI API."""
    
    BASE_URL = "https://api.together.xyz/v1"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload for Together AI API."""
        payload = super().prepare_request(request)
        # Add Together AI specific parameters
        payload["context_length_exceeded_behavior"] = "error"
        return payload
    
    def get_chat_completion_endpoint(self) -> str:
        """Get the chat completion endpoint."""
        return f"{self.BASE_URL}/chat/completions" 