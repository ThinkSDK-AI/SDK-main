"""
Workflow framework for FourierSDK.

Provides node-based workflow system similar to n8n, allowing orchestration
of agents, assistants, and other processing nodes in a directed graph.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of workflow nodes."""
    INPUT = "input"
    AGENT = "agent"
    ASSISTANT = "assistant"
    TRANSFORM = "transform"
    CONDITION = "condition"
    OUTPUT = "output"
    TOOL = "tool"
    RAG = "rag"


class ExecutionStatus(Enum):
    """Execution status for nodes."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result from node execution."""
    node_id: str
    node_type: NodeType
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowNode(ABC):
    """Base class for workflow nodes."""

    def __init__(self, node_id: str, name: str, node_type: NodeType):
        """
        Initialize workflow node.

        Args:
            node_id: Unique node identifier
            name: Node name
            node_type: Type of node
        """
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.next_nodes: List[str] = []
        self.config: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """
        Execute the node.

        Args:
            input_data: Input data from previous node
            context: Workflow execution context

        Returns:
            NodeResult with execution outcome
        """
        pass

    def add_next(self, node_id: str):
        """Add next node in workflow."""
        if node_id not in self.next_nodes:
            self.next_nodes.append(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "type": self.node_type.value,
            "next_nodes": self.next_nodes,
            "config": self.config
        }


class InputNode(WorkflowNode):
    """Input node - entry point for workflow."""

    def __init__(self, node_id: str, name: str = "Input"):
        super().__init__(node_id, name, NodeType.INPUT)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Pass through input data."""
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            status=ExecutionStatus.SUCCESS,
            output=input_data
        )


class AgentNode(WorkflowNode):
    """Agent node - executes an agent."""

    def __init__(self, node_id: str, name: str = "Agent", agent=None):
        super().__init__(node_id, name, NodeType.AGENT)
        self.agent = agent

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Execute agent with input."""
        import time
        start_time = time.time()

        try:
            if not self.agent:
                raise ValueError("Agent not configured")

            # Convert input to string query
            query = str(input_data) if not isinstance(input_data, str) else input_data

            # Run agent
            response = self.agent.run(query)

            execution_time = time.time() - start_time

            if response.get("success"):
                return NodeResult(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    status=ExecutionStatus.SUCCESS,
                    output=response["output"],
                    execution_time=execution_time,
                    metadata={
                        "iterations": response.get("iterations", 0),
                        "tool_calls": response.get("tool_calls", 0)
                    }
                )
            else:
                return NodeResult(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    status=ExecutionStatus.FAILED,
                    output=None,
                    error=response.get("error", "Unknown error"),
                    execution_time=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent node execution failed: {e}", exc_info=True)
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=execution_time
            )


class AssistantNode(WorkflowNode):
    """Assistant node - executes an assistant."""

    def __init__(self, node_id: str, name: str = "Assistant", assistant=None):
        super().__init__(node_id, name, NodeType.ASSISTANT)
        self.assistant = assistant

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Execute assistant with input."""
        import time
        start_time = time.time()

        try:
            if not self.assistant:
                raise ValueError("Assistant not configured")

            # Convert input to string message
            message = str(input_data) if not isinstance(input_data, str) else input_data

            # Chat with assistant
            response = self.assistant.chat(message)

            execution_time = time.time() - start_time

            if response.get("success"):
                return NodeResult(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    status=ExecutionStatus.SUCCESS,
                    output=response["output"],
                    execution_time=execution_time,
                    metadata={
                        "message_count": response.get("message_count", 0)
                    }
                )
            else:
                return NodeResult(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    status=ExecutionStatus.FAILED,
                    output=None,
                    error=response.get("error", "Unknown error"),
                    execution_time=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Assistant node execution failed: {e}", exc_info=True)
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=execution_time
            )


class TransformNode(WorkflowNode):
    """Transform node - applies transformation function to data."""

    def __init__(self, node_id: str, name: str = "Transform", transform_fn: Optional[Callable] = None):
        super().__init__(node_id, name, NodeType.TRANSFORM)
        self.transform_fn = transform_fn or (lambda x: x)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Apply transformation to input."""
        import time
        start_time = time.time()

        try:
            output = self.transform_fn(input_data)
            execution_time = time.time() - start_time

            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Transform node execution failed: {e}", exc_info=True)
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=execution_time
            )


class ConditionNode(WorkflowNode):
    """Condition node - branches based on condition."""

    def __init__(self, node_id: str, name: str = "Condition", condition_fn: Optional[Callable] = None):
        super().__init__(node_id, name, NodeType.CONDITION)
        self.condition_fn = condition_fn or (lambda x: True)
        self.true_branch: Optional[str] = None
        self.false_branch: Optional[str] = None

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Evaluate condition and determine next node."""
        import time
        start_time = time.time()

        try:
            condition_result = self.condition_fn(input_data)
            execution_time = time.time() - start_time

            # Set next node based on condition
            next_node = self.true_branch if condition_result else self.false_branch

            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.SUCCESS,
                output=input_data,  # Pass through input
                execution_time=execution_time,
                metadata={
                    "condition_result": condition_result,
                    "next_node": next_node
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Condition node execution failed: {e}", exc_info=True)
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=execution_time
            )


class OutputNode(WorkflowNode):
    """Output node - final output of workflow."""

    def __init__(self, node_id: str, name: str = "Output"):
        super().__init__(node_id, name, NodeType.OUTPUT)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> NodeResult:
        """Return final output."""
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            status=ExecutionStatus.SUCCESS,
            output=input_data
        )


class Workflow:
    """
    Workflow orchestration system.

    Manages execution of node-based workflows with agents, assistants,
    and other processing nodes.

    Example:
        >>> workflow = Workflow(name="MyWorkflow")
        >>> input_node = workflow.add_input_node()
        >>> agent_node = workflow.add_agent_node(agent=my_agent)
        >>> output_node = workflow.add_output_node()
        >>>
        >>> workflow.connect(input_node.node_id, agent_node.node_id)
        >>> workflow.connect(agent_node.node_id, output_node.node_id)
        >>>
        >>> result = workflow.execute("User query")
    """

    def __init__(self, name: str = "Workflow", workflow_id: Optional[str] = None):
        """
        Initialize workflow.

        Args:
            name: Workflow name
            workflow_id: Unique workflow ID (auto-generated if None)
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.start_node_id: Optional[str] = None
        self.created_at = datetime.now()
        self.execution_count = 0

    def add_node(self, node: WorkflowNode) -> WorkflowNode:
        """
        Add node to workflow.

        Args:
            node: Workflow node to add

        Returns:
            The added node
        """
        self.nodes[node.node_id] = node
        logger.debug(f"Added node: {node.node_id} ({node.node_type.value})")
        return node

    def add_input_node(self, name: str = "Input") -> InputNode:
        """Add input node (workflow entry point)."""
        node = InputNode(node_id=self._generate_node_id(), name=name)
        self.add_node(node)
        if self.start_node_id is None:
            self.start_node_id = node.node_id
        return node

    def add_agent_node(self, agent, name: str = "Agent") -> AgentNode:
        """Add agent node."""
        node = AgentNode(node_id=self._generate_node_id(), name=name, agent=agent)
        self.add_node(node)
        return node

    def add_assistant_node(self, assistant, name: str = "Assistant") -> AssistantNode:
        """Add assistant node."""
        node = AssistantNode(node_id=self._generate_node_id(), name=name, assistant=assistant)
        self.add_node(node)
        return node

    def add_transform_node(self, transform_fn: Callable, name: str = "Transform") -> TransformNode:
        """Add transform node."""
        node = TransformNode(node_id=self._generate_node_id(), name=name, transform_fn=transform_fn)
        self.add_node(node)
        return node

    def add_condition_node(self, condition_fn: Callable, name: str = "Condition") -> ConditionNode:
        """Add condition node."""
        node = ConditionNode(node_id=self._generate_node_id(), name=name, condition_fn=condition_fn)
        self.add_node(node)
        return node

    def add_output_node(self, name: str = "Output") -> OutputNode:
        """Add output node."""
        node = OutputNode(node_id=self._generate_node_id(), name=name)
        self.add_node(node)
        return node

    def connect(self, from_node_id: str, to_node_id: str):
        """
        Connect two nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Destination node ID
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node not found: {from_node_id}")
        if to_node_id not in self.nodes:
            raise ValueError(f"Destination node not found: {to_node_id}")

        self.nodes[from_node_id].add_next(to_node_id)
        logger.debug(f"Connected: {from_node_id} -> {to_node_id}")

    def execute(self, input_data: Any, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute workflow.

        Args:
            input_data: Input data for workflow
            verbose: Enable verbose logging

        Returns:
            Dictionary with execution results:
                - output: Final output
                - success: Whether execution succeeded
                - node_results: List of NodeResult objects
                - execution_time: Total execution time
        """
        import time

        if not self.start_node_id:
            raise ValueError("Workflow has no start node")

        start_time = time.time()
        self.execution_count += 1

        context = {
            "workflow_id": self.workflow_id,
            "workflow_name": self.name,
            "execution_number": self.execution_count,
            "verbose": verbose
        }

        node_results: List[NodeResult] = []
        current_node_id = self.start_node_id
        current_data = input_data

        if verbose:
            logger.info(f"[{self.name}] Starting workflow execution #{self.execution_count}")

        try:
            visited = set()

            while current_node_id:
                if current_node_id in visited:
                    raise ValueError(f"Cycle detected at node: {current_node_id}")

                visited.add(current_node_id)
                node = self.nodes.get(current_node_id)

                if not node:
                    raise ValueError(f"Node not found: {current_node_id}")

                if verbose:
                    logger.info(f"[{self.name}] Executing node: {node.name} ({node.node_type.value})")

                # Execute node
                result = node.execute(current_data, context)
                node_results.append(result)

                if verbose:
                    logger.info(f"[{self.name}] Node {node.name} completed: {result.status.value}")

                # Handle failure
                if result.status == ExecutionStatus.FAILED:
                    execution_time = time.time() - start_time
                    return {
                        "output": None,
                        "success": False,
                        "error": result.error,
                        "node_results": node_results,
                        "execution_time": execution_time,
                        "failed_node": current_node_id
                    }

                # Update current data
                current_data = result.output

                # Determine next node
                if node.node_type == NodeType.CONDITION and result.metadata.get("next_node"):
                    current_node_id = result.metadata["next_node"]
                elif node.node_type == NodeType.OUTPUT:
                    break  # End of workflow
                elif node.next_nodes:
                    current_node_id = node.next_nodes[0]  # Take first next node
                else:
                    break  # No more nodes

            execution_time = time.time() - start_time

            if verbose:
                logger.info(f"[{self.name}] Workflow completed in {execution_time:.2f}s")

            return {
                "output": current_data,
                "success": True,
                "node_results": node_results,
                "execution_time": execution_time,
                "nodes_executed": len(node_results)
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[{self.name}] Workflow execution failed: {e}", exc_info=True)
            return {
                "output": None,
                "success": False,
                "error": str(e),
                "node_results": node_results,
                "execution_time": execution_time
            }

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return f"node_{len(self.nodes)}_{uuid.uuid4().hex[:8]}"

    def visualize(self) -> str:
        """
        Generate text-based visualization of workflow.

        Returns:
            String representation of workflow
        """
        lines = [f"Workflow: {self.name} (ID: {self.workflow_id})"]
        lines.append("=" * 60)
        lines.append("")

        if not self.nodes:
            lines.append("(empty workflow)")
            return "\n".join(lines)

        # Build graph representation
        for node_id, node in self.nodes.items():
            start_marker = "→ " if node_id == self.start_node_id else "  "
            lines.append(f"{start_marker}[{node.node_type.value}] {node.name} (ID: {node_id})")

            if node.next_nodes:
                for next_id in node.next_nodes:
                    next_node = self.nodes.get(next_id)
                    if next_node:
                        lines.append(f"    └─> {next_node.name}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "execution_count": self.execution_count,
            "start_node_id": self.start_node_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }

    def save(self, filepath: str):
        """
        Save workflow to file.

        Args:
            filepath: Path to save workflow
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"[{self.name}] Saved workflow to {filepath}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save workflow: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"Workflow(name='{self.name}', nodes={len(self.nodes)}, executions={self.execution_count})"
