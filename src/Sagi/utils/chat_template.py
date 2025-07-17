from typing import List

from autogen_core.memory import MemoryContent


def format_memory_to_string(messages: List[MemoryContent], model_name: str):
    """
    Format messages to a string for the model.

    Args:
        messages: List of messages to format
        model_name: Name of the model to format messages for

    Returns:
        Formatted string representation of messages
    """
    # TODO(kaili): Change it to a proper format
    memory_strings = [
        f"{i}. {str(memory.content)}" for i, memory in enumerate(messages, 1)
    ]
    memory_context = "\nRelevant memories:\n" + "\n".join(memory_strings)
    return memory_context
