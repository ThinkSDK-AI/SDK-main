"""
FourierSDK - Unified LLM Provider Interface

A Python SDK for accessing multiple Large Language Model providers
with a standardized interface, function calling support, internet search capabilities,
autonomous agent framework, assistants, and workflow orchestration.
"""

from fourier import Fourier
from models import Tool, ChatCompletionRequest
from response_normalizer import ResponseNormalizer
from web_search import WebSearch
from agent import Agent, AgentConfig
from assistant import Assistant, AssistantConfig
from workflow import Workflow, WorkflowNode, NodeType, ExecutionStatus
from config import FourierConfig, get_config, tool, ResourceRegistry, FourierPaths

__version__ = "0.2.0"

__all__ = [
    "Fourier",
    "Tool",
    "ChatCompletionRequest",
    "ResponseNormalizer",
    "WebSearch",
    "Agent",
    "AgentConfig",
    "Assistant",
    "AssistantConfig",
    "Workflow",
    "WorkflowNode",
    "NodeType",
    "ExecutionStatus",
    "FourierConfig",
    "get_config",
    "tool",
    "ResourceRegistry",
    "FourierPaths",
]
