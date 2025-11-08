"""
Example agent: Research Agent with thinking mode
"""

from fourier import Fourier
from agent import Agent, AgentConfig
import os

fourier_client = Fourier(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    provider="anthropic"
)

config = AgentConfig(
    name="research_agent",
    description="Deep research agent with thinking mode enabled",
    max_iterations=10,
    verbose=True,
    thinking_enabled=True
)

research_agent = Agent(
    client=fourier_client,
    model="claude-3-5-sonnet-20241022",
    config=config
)

# Export
__agents__ = {
    "research_agent": research_agent,
    "deep_research": research_agent  # Can export with multiple names
}
