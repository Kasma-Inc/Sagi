from typing import List

from autogen_core.memory import MemoryContent


def format_memory_to_string(messages: List[MemoryContent]) -> str:
    """
    Format messages to a string for the model.

    Args:
        messages: List of messages to format

    Returns:
        Formatted string representation of messages
    """
    # TODO(kaili): Change it to a proper format
    memory_strings = [
        f"{memory.metadata.get('source', 'unknown')}: {str(memory.content)}"
        for memory in messages
    ]
    memory_context = "\nRelevant memories:\n" + "\n".join(memory_strings)
    return memory_context
