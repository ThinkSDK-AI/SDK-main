"""
Example MCP Tools: Text Utilities

This file demonstrates multiple tools in a single file.
"""


def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def reverse_text(text: str) -> str:
    """
    Reverse the input text.

    Args:
        text: Input text

    Returns:
        Reversed text
    """
    return text[::-1]


def to_uppercase(text: str) -> str:
    """
    Convert text to uppercase.

    Args:
        text: Input text

    Returns:
        Uppercase text
    """
    return text.upper()


def to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.

    Args:
        text: Input text

    Returns:
        Lowercase text
    """
    return text.lower()


# Multiple tools can be defined in a single MCP_TOOLS list
MCP_TOOLS = [
    {
        "name": "count_words",
        "description": "Count the number of words in a text",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in"
                }
            },
            "required": ["text"]
        },
        "function": count_words,
        "metadata": {
            "category": "text",
            "version": "1.0.0"
        }
    },
    {
        "name": "reverse_text",
        "description": "Reverse the order of characters in a text",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to reverse"
                }
            },
            "required": ["text"]
        },
        "function": reverse_text,
        "metadata": {
            "category": "text",
            "version": "1.0.0"
        }
    },
    {
        "name": "to_uppercase",
        "description": "Convert text to uppercase letters",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert"
                }
            },
            "required": ["text"]
        },
        "function": to_uppercase,
        "metadata": {
            "category": "text",
            "version": "1.0.0"
        }
    },
    {
        "name": "to_lowercase",
        "description": "Convert text to lowercase letters",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert"
                }
            },
            "required": ["text"]
        },
        "function": to_lowercase,
        "metadata": {
            "category": "text",
            "version": "1.0.0"
        }
    }
]
