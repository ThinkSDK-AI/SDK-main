"""
Assistant framework for FourierSDK.

Assistants are simple, context-aware LLM wrappers that maintain conversation
history and can optionally integrate with RAG (Retrieval Augmented Generation).

Unlike Agents, Assistants do not automatically execute tools. They are simpler
and more suitable for conversational use cases.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime
from fourier import Fourier

logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """Configuration for Assistant behavior."""

    temperature: float = 0.7
    """Temperature for LLM responses."""

    max_tokens: Optional[int] = None
    """Maximum tokens in response."""

    system_prompt: Optional[str] = None
    """System prompt for the assistant."""

    max_history: int = 50
    """Maximum number of messages to keep in history."""

    enable_rag: bool = False
    """Enable RAG (Retrieval Augmented Generation)."""

    rag_top_k: int = 3
    """Number of top documents to retrieve for RAG."""

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            logger.warning(f"Temperature should be 0.0-2.0, got {self.temperature}. Clamping.")
            self.temperature = max(0.0, min(2.0, self.temperature))

        if self.max_history < 1:
            logger.warning(f"max_history must be >= 1, got {self.max_history}. Setting to 1.")
            self.max_history = 1


class Assistant:
    """
    Simple context-aware LLM assistant.

    Unlike Agents, Assistants do not automatically execute tools. They maintain
    conversation history and can integrate with RAG for document-based context.

    Example:
        >>> from fourier import Fourier
        >>> from assistant import Assistant, AssistantConfig
        >>>
        >>> client = Fourier(api_key="...", provider="groq")
        >>> assistant = Assistant(
        ...     client=client,
        ...     name="MyAssistant",
        ...     model="mixtral-8x7b-32768",
        ...     config=AssistantConfig(temperature=0.7)
        ... )
        >>>
        >>> response = assistant.chat("Hello!")
        >>> print(response["output"])
    """

    def __init__(
        self,
        client: Fourier,
        name: str = "Assistant",
        model: str = "mixtral-8x7b-32768",
        config: Optional[AssistantConfig] = None,
        **kwargs
    ):
        """
        Initialize Assistant.

        Args:
            client: Fourier client instance
            name: Assistant name
            model: Model to use
            config: Assistant configuration
            **kwargs: Additional arguments for client.chat()
        """
        self.client = client
        self.name = name
        self.model = model
        self.config = config or AssistantConfig()
        self.kwargs = kwargs

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # RAG context (if enabled)
        self.rag_documents: List[Dict[str, Any]] = []

        # Metadata
        self.created_at = datetime.now()
        self.total_messages = 0

        logger.info(f"Initialized assistant: {self.name}")

    def chat(
        self,
        message: str,
        reset_history: bool = False,
        **override_kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to the assistant.

        Args:
            message: User message
            reset_history: Whether to reset conversation history
            **override_kwargs: Override default kwargs

        Returns:
            Dictionary with response and metadata:
                - output: Assistant's response
                - message_count: Number of messages in history
                - success: Whether the chat was successful
                - tokens: Token usage information
        """
        if reset_history:
            self.conversation_history = []
            logger.debug(f"[{self.name}] Conversation history reset")

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Trim history if needed
        self._trim_history()

        # Build messages
        messages = self._build_messages()

        # Merge kwargs
        request_kwargs = {**self.kwargs, **override_kwargs}

        try:
            # Make request
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **request_kwargs
            )

            # Extract output
            output = response.get("response", {}).get("output", "")

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": output
            })

            # Update metadata
            self.total_messages += 2  # User + assistant

            return {
                "output": output,
                "message_count": len(self.conversation_history),
                "success": True,
                "tokens": response.get("usage", {}),
                "response": response
            }

        except Exception as e:
            logger.error(f"[{self.name}] Chat failed: {e}", exc_info=True)
            return {
                "output": "",
                "message_count": len(self.conversation_history),
                "success": False,
                "error": str(e)
            }

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        Build messages array for LLM request.

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt
        system_prompt = self.config.system_prompt or self._default_system_prompt()

        # Add RAG context if enabled
        if self.config.enable_rag and self.rag_documents:
            rag_context = self._get_rag_context()
            if rag_context:
                system_prompt += f"\n\nRelevant Context:\n{rag_context}"

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Add conversation history
        messages.extend(self.conversation_history)

        return messages

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return f"You are {self.name}, a helpful AI assistant. Provide clear, concise, and accurate responses."

    def _trim_history(self):
        """Trim conversation history to max_history length."""
        if len(self.conversation_history) > self.config.max_history:
            # Keep most recent messages
            removed = len(self.conversation_history) - self.config.max_history
            self.conversation_history = self.conversation_history[-self.config.max_history:]
            logger.debug(f"[{self.name}] Trimmed {removed} messages from history")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents for RAG context.

        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata'

        Example:
            >>> assistant.add_documents([
            ...     {"content": "Document text here", "metadata": {"source": "file.txt"}},
            ...     {"content": "More text", "metadata": {"source": "file2.txt"}}
            ... ])
        """
        if not self.config.enable_rag:
            logger.warning(f"[{self.name}] RAG not enabled. Enable with config.enable_rag=True")
            return

        for doc in documents:
            if "content" not in doc:
                logger.warning(f"[{self.name}] Document missing 'content' field, skipping")
                continue

            self.rag_documents.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "added_at": datetime.now().isoformat()
            })

        logger.info(f"[{self.name}] Added {len(documents)} documents. Total: {len(self.rag_documents)}")

    def clear_documents(self):
        """Clear all RAG documents."""
        count = len(self.rag_documents)
        self.rag_documents = []
        logger.info(f"[{self.name}] Cleared {count} documents")

    def _get_rag_context(self, query: Optional[str] = None) -> str:
        """
        Get relevant context from RAG documents.

        Args:
            query: Optional query for retrieval (uses last user message if None)

        Returns:
            Formatted context string
        """
        if not self.rag_documents:
            return ""

        # Get query from last user message if not provided
        if query is None:
            for msg in reversed(self.conversation_history):
                if msg["role"] == "user":
                    query = msg["content"]
                    break

        if not query:
            return ""

        # Simple keyword-based retrieval (can be enhanced with embeddings)
        scored_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in self.rag_documents:
            content_lower = doc["content"].lower()
            content_words = set(content_lower.split())

            # Simple scoring: count of matching words
            score = len(query_words.intersection(content_words))

            # Bonus for exact phrase match
            if query_lower in content_lower:
                score += 10

            scored_docs.append((score, doc))

        # Sort by score and take top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = scored_docs[:self.config.rag_top_k]

        # Format context
        context_parts = []
        for i, (score, doc) in enumerate(top_docs, 1):
            if score > 0:  # Only include relevant documents
                source = doc["metadata"].get("source", "Unknown")
                context_parts.append(f"[Document {i} - {source}]\n{doc['content']}")

        return "\n\n".join(context_parts)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.

        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        count = len(self.conversation_history)
        self.conversation_history = []
        logger.info(f"[{self.name}] Cleared {count} messages from history")

    def save_conversation(self, filepath: str):
        """
        Save conversation to file.

        Args:
            filepath: Path to save conversation
        """
        data = {
            "assistant_name": self.name,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "total_messages": self.total_messages,
            "conversation": self.conversation_history
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[{self.name}] Saved conversation to {filepath}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save conversation: {e}")
            raise

    def load_conversation(self, filepath: str):
        """
        Load conversation from file.

        Args:
            filepath: Path to load conversation from
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.conversation_history = data.get("conversation", [])
            self.total_messages = data.get("total_messages", len(self.conversation_history))

            logger.info(f"[{self.name}] Loaded conversation from {filepath}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load conversation: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get assistant statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "name": self.name,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "total_messages": self.total_messages,
            "current_history_length": len(self.conversation_history),
            "rag_enabled": self.config.enable_rag,
            "rag_documents": len(self.rag_documents),
            "max_history": self.config.max_history
        }

    def update_system_prompt(self, system_prompt: str):
        """
        Update system prompt.

        Args:
            system_prompt: New system prompt
        """
        self.config.system_prompt = system_prompt
        logger.info(f"[{self.name}] Updated system prompt")

    def __repr__(self) -> str:
        """String representation."""
        return f"Assistant(name='{self.name}', model='{self.model}', messages={len(self.conversation_history)})"
