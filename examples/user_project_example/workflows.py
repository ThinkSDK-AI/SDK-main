"""
Example workflows defined in a single file

This shows the alternative to a workflows/ directory.
You can define all your workflows in workflows.py
"""

from workflow import Workflow, WorkflowNode, NodeType, ExecutionStatus
from fourier import Fourier
from agent import Agent, AgentConfig
import os


# Create a simple workflow
def create_customer_onboarding_workflow():
    """Create a customer onboarding workflow"""

    workflow = Workflow(name="customer_onboarding")

    # You would define nodes here
    # This is a simplified example

    return workflow


# Export workflows
onboarding_workflow = create_customer_onboarding_workflow()

__workflows__ = {
    "customer_onboarding": onboarding_workflow
}
