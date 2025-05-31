from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Tool(BaseModel):
    """Base class for defining tools that can be used by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

class ChatCompletionRequest(BaseModel):
    """Parameters for chat completion request."""
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None 