"""
Assistant Examples

Demonstrates how to use the Assistant class for simple, context-aware
conversations with optional RAG support.
"""

import os
from fourier import Fourier
from assistant import Assistant, AssistantConfig
from dotenv import load_dotenv

load_dotenv()


def basic_assistant_example():
    """Basic assistant usage."""
    print("=== Basic Assistant Example ===\n")

    # Create Fourier client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create assistant
    assistant = Assistant(
        client=client,
        name="ChatBot",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            temperature=0.7,
            max_history=10,
            system_prompt="You are a friendly and helpful assistant."
        )
    )

    # Have conversation
    print("Assistant: Hello! I'm your assistant. How can I help you today?")

    queries = [
        "My name is Alice",
        "What's the capital of France?",
        "What's my name?"  # Tests conversation memory
    ]

    for query in queries:
        print(f"\nYou: {query}")
        response = assistant.chat(query)

        if response["success"]:
            print(f"Assistant: {response['output']}")
        else:
            print(f"Error: {response.get('error')}")

    # Show stats
    print(f"\n{assistant.get_stats()}")


def rag_assistant_example():
    """Assistant with RAG (document context)."""
    print("\n\n=== RAG Assistant Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create assistant with RAG enabled
    assistant = Assistant(
        client=client,
        name="DocAssistant",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            temperature=0.7,
            enable_rag=True,
            rag_top_k=2,
            system_prompt="You are a helpful assistant. Use the provided context to answer questions accurately."
        )
    )

    # Add documents
    documents = [
        {
            "content": "FourierSDK is a Python SDK for accessing Large Language Models from various providers like Groq, OpenAI, and Anthropic.",
            "metadata": {"source": "fourier_docs.txt"}
        },
        {
            "content": "The Agent class in FourierSDK allows autonomous execution of tools. Agents can automatically use registered tools to complete tasks.",
            "metadata": {"source": "agent_docs.txt"}
        },
        {
            "content": "Thinking Mode is a feature that performs deep research by conducting multiple web searches before answering questions.",
            "metadata": {"source": "thinking_mode_docs.txt"}
        }
    ]

    assistant.add_documents(documents)
    print(f"Added {len(documents)} documents to assistant\n")

    # Ask questions about the documents
    queries = [
        "What is FourierSDK?",
        "Tell me about the Agent class",
        "What is Thinking Mode?"
    ]

    for query in queries:
        print(f"You: {query}")
        response = assistant.chat(query)

        if response["success"]:
            print(f"Assistant: {response['output']}\n")
        else:
            print(f"Error: {response.get('error')}\n")


def conversation_persistence_example():
    """Save and load conversations."""
    print("\n\n=== Conversation Persistence Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create assistant
    assistant = Assistant(
        client=client,
        name="PersistentBot",
        model="mixtral-8x7b-32768"
    )

    # Have conversation
    print("=== First Conversation ===")
    response1 = assistant.chat("My favorite color is blue")
    print(f"You: My favorite color is blue")
    print(f"Assistant: {response1['output']}\n")

    response2 = assistant.chat("What's my favorite color?")
    print(f"You: What's my favorite color?")
    print(f"Assistant: {response2['output']}\n")

    # Save conversation
    assistant.save_conversation("conversation.json")
    print("Conversation saved to conversation.json\n")

    # Create new assistant and load conversation
    print("=== Loading Conversation in New Assistant ===")
    new_assistant = Assistant(
        client=client,
        name="PersistentBot2",
        model="mixtral-8x7b-32768"
    )

    new_assistant.load_conversation("conversation.json")
    print("Conversation loaded from conversation.json\n")

    # Continue conversation
    response3 = new_assistant.chat("And what's my favorite color again?")
    print(f"You: And what's my favorite color again?")
    print(f"Assistant: {response3['output']}\n")


def multi_assistant_example():
    """Multiple assistants with different personalities."""
    print("\n\n=== Multi-Assistant Example ===\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set")
        return

    client = Fourier(api_key=api_key, provider="groq")

    # Create different assistants
    formal_assistant = Assistant(
        client=client,
        name="FormalBot",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="You are a formal, professional assistant. Use formal language and be concise."
        )
    )

    casual_assistant = Assistant(
        client=client,
        name="CasualBot",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="You are a casual, friendly assistant. Use informal language and be conversational."
        )
    )

    creative_assistant = Assistant(
        client=client,
        name="CreativeBot",
        model="mixtral-8x7b-32768",
        config=AssistantConfig(
            system_prompt="You are a creative, imaginative assistant. Use metaphors and creative language.",
            temperature=0.9
        )
    )

    # Ask same question to all assistants
    query = "Explain what an API is"
    print(f"Question: {query}\n")

    print("=== Formal Assistant ===")
    response = formal_assistant.chat(query)
    print(f"{response['output']}\n")

    print("=== Casual Assistant ===")
    response = casual_assistant.chat(query)
    print(f"{response['output']}\n")

    print("=== Creative Assistant ===")
    response = creative_assistant.chat(query)
    print(f"{response['output']}\n")


def main():
    """Run all examples."""
    print("="*60)
    print("FourierSDK Assistant Examples")
    print("="*60)

    try:
        basic_assistant_example()
        rag_assistant_example()
        conversation_persistence_example()
        multi_assistant_example()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Examples Complete")
    print("="*60)


if __name__ == "__main__":
    main()
