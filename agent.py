"""
Agent framework for FourierSDK.

This module provides an Agent class that enables autonomous agent behavior
with automatic tool execution, conversation management, and response handling.
"""

from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
import json
from fourier import Fourier
from models import Tool
from exceptions import ToolExecutionError, InvalidRequestError
from mcp import MCPClient
from web_search import web_search

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for Agent behavior."""

    max_iterations: int = 10
    """Maximum number of tool execution iterations before stopping."""

    max_tool_calls_per_iteration: int = 5
    """Maximum number of tool calls in a single iteration."""

    auto_execute_tools: bool = True
    """Whether to automatically execute tools when requested by the LLM."""

    require_tool_confirmation: bool = False
    """Whether to require confirmation before executing tools."""

    verbose: bool = False
    """Whether to print detailed execution logs."""

    temperature: float = 0.7
    """Default temperature for LLM requests."""

    max_tokens: Optional[int] = None
    """Default max tokens for LLM requests."""

    stop_on_error: bool = False
    """Whether to stop execution on tool errors or continue."""

    return_intermediate_steps: bool = False
    """Whether to return intermediate steps in the response."""

    timeout_seconds: Optional[int] = None
    """Maximum execution time in seconds (None for no limit)."""

    thinking_mode: bool = False
    """Enable thinking mode for deep research and analysis."""

    thinking_depth: int = 2
    """Number of research queries to perform in thinking mode (1-5)."""

    thinking_web_search_results: int = 5
    """Number of web search results to retrieve per query in thinking mode."""


class Agent:
    """
    Autonomous agent that manages tools, executes them, and maintains conversation state.

    The Agent class provides a high-level interface for creating autonomous agents
    that can use tools, manage conversations, and execute complex workflows.

    Example:
        >>> from fourier import Fourier
        >>> from agent import Agent
        >>>
        >>> # Create client
        >>> client = Fourier(api_key="...", provider="groq")
        >>>
        >>> # Create agent
        >>> agent = Agent(
        ...     client=client,
        ...     name="MathAssistant",
        ...     system_prompt="You are a helpful math assistant.",
        ...     model="mixtral-8x7b-32768"
        ... )
        >>>
        >>> # Register a tool
        >>> def calculator(operation: str, a: float, b: float) -> float:
        ...     if operation == "add": return a + b
        ...     elif operation == "multiply": return a * b
        ...     return 0
        >>>
        >>> agent.register_tool(
        ...     name="calculator",
        ...     description="Perform arithmetic operations",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "operation": {"type": "string", "enum": ["add", "multiply"]},
        ...             "a": {"type": "number"},
        ...             "b": {"type": "number"}
        ...         }
        ...     },
        ...     required=["operation", "a", "b"],
        ...     function=calculator
        ... )
        >>>
        >>> # Execute agent
        >>> response = agent.run("What is 25 times 4?")
        >>> print(response["output"])
    """

    def __init__(
        self,
        client: Fourier,
        name: str = "Agent",
        system_prompt: Optional[str] = None,
        model: str = "mixtral-8x7b-32768",
        config: Optional[AgentConfig] = None,
        **kwargs
    ):
        """
        Initialize the Agent.

        Args:
            client: FourierSDK client instance
            name: Name of the agent (for logging/debugging)
            system_prompt: System prompt that defines the agent's behavior
            model: Default model to use for chat completions
            config: Agent configuration (uses defaults if not provided)
            **kwargs: Additional arguments passed to chat completions
        """
        self.client = client
        self.name = name
        self.model = model
        self.config = config or AgentConfig()
        self.kwargs = kwargs

        # Initialize system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Tool management
        self.tools: Dict[str, Tool] = {}
        self.tool_functions: Dict[str, Callable] = {}

        # MCP client for managing MCP tools
        self.mcp_client: Optional[MCPClient] = None

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.intermediate_steps: List[Dict[str, Any]] = []

        logger.info(f"Initialized agent: {self.name}")

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return (
            f"You are {self.name}, a helpful AI assistant. "
            "When you need to use a tool, respond with the exact format requested. "
            "Be precise and follow instructions carefully."
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable,
        required: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool with the agent.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: JSON Schema describing the tool's parameters
            function: Python function that implements the tool
            required: List of required parameter names

        Example:
            >>> def search_web(query: str) -> str:
            ...     return f"Results for: {query}"
            >>>
            >>> agent.register_tool(
            ...     name="search",
            ...     description="Search the web",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "query": {"type": "string"}
            ...         }
            ...     },
            ...     required=["query"],
            ...     function=search_web
            ... )
        """
        tool = self.client.create_tool(
            name=name,
            description=description,
            parameters=parameters,
            required=required or []
        )

        self.tools[name] = tool
        self.tool_functions[name] = function

        if self.config.verbose:
            logger.info(f"Registered tool: {name}")

    def _ensure_mcp_client(self) -> MCPClient:
        """
        Ensure MCP client is initialized.

        Returns:
            MCPClient instance
        """
        if self.mcp_client is None:
            self.mcp_client = MCPClient()
            if self.config.verbose:
                logger.info("Initialized MCP client")
        return self.mcp_client

    def register_mcp_url(
        self,
        url: str,
        name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Register tools from a remote MCP server via URL.

        Args:
            url: MCP server URL
            name: Optional connector name
            headers: Optional HTTP headers

        Returns:
            True if successful

        Example:
            >>> agent.register_mcp_url("https://mcp.example.com/api")
            True
        """
        mcp = self._ensure_mcp_client()
        success = mcp.add_url(url, name, headers)

        if success:
            self._sync_mcp_tools()
            if self.config.verbose:
                logger.info(f"Registered MCP tools from URL: {url}")

        return success

    def register_mcp_config(
        self,
        config: Union[str, Dict[str, Any]],
        name: Optional[str] = None
    ) -> bool:
        """
        Register tools from MCP server configuration.

        Args:
            config: Config file path or config dictionary
            name: Optional connector name

        Returns:
            True if successful

        Example:
            >>> agent.register_mcp_config("./mcp_config.json")
            True
            >>> agent.register_mcp_config({
            ...     "command": "python",
            ...     "args": ["-m", "mcp_server"],
            ...     "env": {"API_KEY": "123"}
            ... })
            True
        """
        mcp = self._ensure_mcp_client()
        success = mcp.add_config(config, name)

        if success:
            self._sync_mcp_tools()
            if self.config.verbose:
                logger.info(f"Registered MCP tools from config: {config}")

        return success

    def register_mcp_directory(self, directory: str, name: Optional[str] = None) -> bool:
        """
        Register tools from a directory containing MCP tools.

        Args:
            directory: Path to tool directory
            name: Optional connector name

        Returns:
            True if successful

        Example:
            >>> agent.register_mcp_directory("./mcp_tools")
            True
        """
        mcp = self._ensure_mcp_client()
        success = mcp.add_directory(directory, name)

        if success:
            self._sync_mcp_tools()
            if self.config.verbose:
                logger.info(f"Registered MCP tools from directory: {directory}")

        return success

    def register_mcp_directories(self, directories: List[str]) -> Dict[str, bool]:
        """
        Register tools from multiple directories.

        Args:
            directories: List of directory paths

        Returns:
            Dictionary mapping directory paths to success status

        Example:
            >>> agent.register_mcp_directories([
            ...     "./mcp_tools",
            ...     "./custom_tools",
            ...     "./external_tools"
            ... ])
            {'./mcp_tools': True, './custom_tools': True, './external_tools': True}
        """
        results = {}

        for directory in directories:
            results[directory] = self.register_mcp_directory(directory)

        return results

    def _sync_mcp_tools(self) -> None:
        """
        Synchronize MCP tools into the agent's tool registry.

        This converts MCP tools to Fourier tools and registers their
        execution functions.
        """
        if self.mcp_client is None:
            return

        mcp_tools = self.mcp_client.get_all_tools()

        for mcp_tool in mcp_tools:
            # Skip if already registered
            if mcp_tool.name in self.tools:
                continue

            # Convert MCP tool to Fourier tool format
            fourier_tool_dict = mcp_tool.to_fourier_tool()

            # Create Tool instance
            tool = self.client.create_tool(
                name=fourier_tool_dict["name"],
                description=fourier_tool_dict["description"],
                parameters=fourier_tool_dict["parameters"],
                required=fourier_tool_dict.get("required", [])
            )

            self.tools[mcp_tool.name] = tool

            # Register execution function
            if mcp_tool.function:
                # Direct Python function
                self.tool_functions[mcp_tool.name] = mcp_tool.function
            else:
                # Use MCP client to call the tool
                def create_mcp_caller(tool_name):
                    def mcp_caller(**kwargs):
                        return self.mcp_client.call_tool(tool_name, kwargs)
                    return mcp_caller

                self.tool_functions[mcp_tool.name] = create_mcp_caller(mcp_tool.name)

            if self.config.verbose:
                logger.debug(f"Synced MCP tool: {mcp_tool.name}")

    def get_mcp_tool_names(self) -> List[str]:
        """
        Get list of all MCP tool names.

        Returns:
            List of MCP tool names
        """
        if self.mcp_client is None:
            return []

        return self.mcp_client.get_tool_names()

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Result of the tool execution

        Raises:
            ToolExecutionError: If tool execution fails
        """
        if tool_name not in self.tool_functions:
            raise ToolExecutionError(
                f"Tool '{tool_name}' not registered",
                tool_name
            )

        try:
            if self.config.verbose:
                logger.info(f"Executing tool: {tool_name} with params: {parameters}")

            result = self.tool_functions[tool_name](**parameters)

            if self.config.verbose:
                logger.info(f"Tool result: {result}")

            return result

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)

            if self.config.stop_on_error:
                raise ToolExecutionError(error_msg, tool_name) from e

            return f"Error: {str(e)}"

    def _format_tool_result_message(
        self,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        result: Any
    ) -> str:
        """
        Format tool execution result as a message for the LLM.

        Args:
            tool_name: Name of the tool that was executed
            tool_parameters: Parameters used for the tool
            result: Result from the tool execution

        Returns:
            Formatted message string
        """
        return (
            f"Tool '{tool_name}' was executed with parameters {json.dumps(tool_parameters)}. "
            f"Result: {result}"
        )

    def _perform_thinking_research(self, user_input: str) -> str:
        """
        Perform deep research using web search for thinking mode.

        This method conducts multiple web searches to gather comprehensive
        context and information related to the user's query.

        Args:
            user_input: The user's input/query

        Returns:
            Formatted research context string

        Raises:
            Exception: If research fails and stop_on_error is True
        """
        if self.config.verbose:
            logger.info(f"[Thinking Mode] Starting deep research for: {user_input[:100]}...")

        research_context = []

        # Clamp thinking depth between 1 and 5
        depth = max(1, min(5, self.config.thinking_depth))

        try:
            # First, generate research queries based on the user input
            # Use LLM to generate diverse search queries
            query_generation_prompt = f"""Given this user question: "{user_input}"

Generate {depth} diverse search queries that would help gather comprehensive information to answer this question.
Each query should focus on a different aspect or angle of the question.

Return ONLY the queries, one per line, without numbering or explanations."""

            if self.config.verbose:
                logger.info("[Thinking Mode] Generating research queries...")

            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": query_generation_prompt}],
                temperature=0.7,
                max_tokens=500
            )

            # Extract queries from response
            queries_text = response.get("response", {}).get("output", "")
            search_queries = [q.strip() for q in queries_text.split("\n") if q.strip()]

            # Fallback to user input if no queries generated
            if not search_queries:
                search_queries = [user_input]

            # Limit to configured depth
            search_queries = search_queries[:depth]

            if self.config.verbose:
                logger.info(f"[Thinking Mode] Generated {len(search_queries)} research queries")

            # Perform web searches for each query
            for i, query in enumerate(search_queries, 1):
                if self.config.verbose:
                    logger.info(f"[Thinking Mode] Research query {i}/{len(search_queries)}: {query}")

                try:
                    # Perform web search
                    search_results = web_search(
                        query,
                        num_results=self.config.thinking_web_search_results
                    )

                    if search_results:
                        research_context.append(f"\n=== Research Query {i}: {query} ===\n")
                        research_context.append(search_results)

                        if self.config.verbose:
                            logger.info(f"[Thinking Mode] Gathered {len(search_results)} chars of context")

                except Exception as e:
                    logger.warning(f"[Thinking Mode] Search failed for query '{query}': {e}")
                    if self.config.stop_on_error:
                        raise

            # Compile research context
            if research_context:
                full_context = "\n".join(research_context)

                if self.config.verbose:
                    logger.info(f"[Thinking Mode] Research complete: {len(full_context)} total characters")

                return full_context
            else:
                logger.warning("[Thinking Mode] No research context gathered")
                return ""

        except Exception as e:
            logger.error(f"[Thinking Mode] Research failed: {e}", exc_info=True)
            if self.config.stop_on_error:
                raise
            return ""

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        Build the messages array for the LLM request.

        Args:
            user_input: The user's input/query

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user input if not already in history
        if not self.conversation_history or self.conversation_history[-1].get("content") != user_input:
            messages.append({
                "role": "user",
                "content": user_input
            })

        return messages

    def run(
        self,
        user_input: str,
        reset_history: bool = True,
        **override_kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent with the given input.

        This method executes the agent's main loop:
        1. Send user input to LLM
        2. If LLM requests tool use, execute the tool
        3. Send tool result back to LLM
        4. Repeat until final answer or max iterations

        Args:
            user_input: The user's input/query
            reset_history: Whether to reset conversation history before running
            **override_kwargs: Override default kwargs for this run

        Returns:
            Dictionary containing:
                - output: Final response from the agent
                - iterations: Number of iterations taken
                - tool_calls: Number of tools called
                - intermediate_steps: List of intermediate steps (if enabled)
                - success: Whether execution completed successfully

        Example:
            >>> response = agent.run("Calculate 15 + 27")
            >>> print(response["output"])
            >>> print(f"Used {response['tool_calls']} tool calls")
        """
        if reset_history:
            self.conversation_history = []
            self.intermediate_steps = []

        # Merge kwargs
        request_kwargs = {**self.kwargs, **override_kwargs}

        # Perform thinking mode research if enabled
        enhanced_input = user_input
        if self.config.thinking_mode:
            if self.config.verbose:
                logger.info("[Thinking Mode] Enabled - performing deep research")

            research_context = self._perform_thinking_research(user_input)

            if research_context:
                # Enhance the user input with research context
                enhanced_input = f"""I have gathered the following research context to help answer your question:

{research_context}

---

Based on the above research context, please answer this question:
{user_input}

Synthesize the information from multiple sources and provide a comprehensive, well-reasoned answer."""

                if self.config.verbose:
                    logger.info(f"[Thinking Mode] Enhanced input with {len(research_context)} chars of context")

        # Build initial messages
        messages = self._build_messages(enhanced_input)

        # Track execution
        iterations = 0
        total_tool_calls = 0

        if self.config.verbose:
            logger.info(f"Starting agent run for: {user_input[:100]}...")

        while iterations < self.config.max_iterations:
            iterations += 1

            if self.config.verbose:
                logger.info(f"Iteration {iterations}/{self.config.max_iterations}")

            try:
                # Make request to LLM
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=list(self.tools.values()) if self.tools else None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **request_kwargs
                )

                # Check response type
                response_type = response.get("metadata", {}).get("response_type")

                if response_type == "tool_call" and self.config.auto_execute_tools:
                    # Extract tool information
                    tool_name = response.get("response", {}).get("tool_used")
                    tool_params = response.get("response", {}).get("tool_parameters", {})

                    if not tool_name:
                        # No tool in response, treat as final answer
                        break

                    total_tool_calls += 1

                    if self.config.verbose:
                        logger.info(f"LLM requested tool: {tool_name}")

                    # Execute the tool
                    tool_result = self.execute_tool(tool_name, tool_params)

                    # Record intermediate step
                    if self.config.return_intermediate_steps:
                        self.intermediate_steps.append({
                            "iteration": iterations,
                            "tool": tool_name,
                            "parameters": tool_params,
                            "result": tool_result
                        })

                    # Format tool result message
                    tool_message = self._format_tool_result_message(
                        tool_name,
                        tool_params,
                        tool_result
                    )

                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"I need to use the {tool_name} tool."
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_message
                    })

                    # Rebuild messages for next iteration
                    messages = self._build_messages(user_input)

                else:
                    # Got final answer
                    output = response.get("response", {}).get("output", "")

                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": output
                    })

                    if self.config.verbose:
                        logger.info("Agent completed successfully")

                    return {
                        "output": output,
                        "iterations": iterations,
                        "tool_calls": total_tool_calls,
                        "intermediate_steps": self.intermediate_steps if self.config.return_intermediate_steps else [],
                        "success": True,
                        "response": response
                    }

            except Exception as e:
                logger.error(f"Error in agent iteration {iterations}: {e}", exc_info=True)

                if self.config.stop_on_error:
                    return {
                        "output": f"Agent error: {str(e)}",
                        "iterations": iterations,
                        "tool_calls": total_tool_calls,
                        "intermediate_steps": self.intermediate_steps if self.config.return_intermediate_steps else [],
                        "success": False,
                        "error": str(e)
                    }

                # Continue on error
                continue

        # Max iterations reached
        logger.warning(f"Agent reached max iterations ({self.config.max_iterations})")

        return {
            "output": "Maximum iterations reached without final answer.",
            "iterations": iterations,
            "tool_calls": total_tool_calls,
            "intermediate_steps": self.intermediate_steps if self.config.return_intermediate_steps else [],
            "success": False,
            "error": "max_iterations_reached"
        }

    def reset(self) -> None:
        """Reset the agent's conversation history and intermediate steps."""
        self.conversation_history = []
        self.intermediate_steps = []

        if self.config.verbose:
            logger.info("Agent state reset")

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the agent's system prompt.

        Args:
            new_prompt: New system prompt to use
        """
        self.system_prompt = new_prompt

        if self.config.verbose:
            logger.info("System prompt updated")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()

    def add_to_history(self, role: str, content: str) -> None:
        """
        Manually add a message to conversation history.

        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
        """
        if role not in ["user", "assistant", "system"]:
            raise InvalidRequestError(f"Invalid role: {role}")

        self.conversation_history.append({
            "role": role,
            "content": content
        })
