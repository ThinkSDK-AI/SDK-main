from typing import Dict, Any, Optional
from models import ChatCompletionRequest
from .base import BaseProvider

class NebiusProvider(BaseProvider):
    """Provider implementation for Nebius Cloud API."""
    
    BASE_URL = "https://api.studio.nebius.com/v1"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare the request payload for Nebius Cloud API."""
        payload = super().prepare_request(request)
        
        # Nebius API expects tools in a specific format
        if "tools" in payload:
            tools = payload.pop("tools")
            payload["functions"] = tools
            payload["function_call"] = "auto"
        
        return payload
    
    def get_chat_completion_endpoint(self) -> str:
        """Get the chat completion endpoint."""
        return f"{self.BASE_URL}/chat/completions" 