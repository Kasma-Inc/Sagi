from typing import Any, Dict, Type

from autogen_agentchat.messages import (
    CodeExecutionEvent,
    CodeGenerationEvent,
    HandoffMessage,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    SelectSpeakerEvent,
    StopMessage,
    StructuredMessage,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_core.memory import MemoryMimeType

# Mapping from autogen message types to memory MIME types
MESSAGE_TO_MEMORY_TYPE_MAP: Dict[Type[Any], MemoryMimeType] = {
    # Text-based messages
    TextMessage: MemoryMimeType.TEXT,
    ThoughtEvent: MemoryMimeType.TEXT,
    CodeGenerationEvent: MemoryMimeType.TEXT,
    # Structured/JSON messages
    StructuredMessage: MemoryMimeType.JSON,
    HandoffMessage: MemoryMimeType.JSON,
    StopMessage: MemoryMimeType.JSON,
    ToolCallSummaryMessage: MemoryMimeType.JSON,
    ToolCallExecutionEvent: MemoryMimeType.JSON,
    ToolCallRequestEvent: MemoryMimeType.JSON,
    MemoryQueryEvent: MemoryMimeType.JSON,
    UserInputRequestedEvent: MemoryMimeType.JSON,
    SelectSpeakerEvent: MemoryMimeType.JSON,
    CodeExecutionEvent: MemoryMimeType.JSON,
    # Multimodal messages (may contain images)
    MultiModalMessage: MemoryMimeType.IMAGE,
    # Streaming chunks (binary data)
    ModelClientStreamingChunkEvent: MemoryMimeType.BINARY,
}


def get_memory_type_for_message(message: Any) -> str:
    """
    Get the appropriate memory MIME type for a given autogen message.

    Args:
        message: An autogen message object

    Returns:
        str: The string value of the appropriate MIME type for storing this message in memory

    Raises:
        ValueError: If the message type is not recognized
    """
    message_type = type(message)

    if message_type in MESSAGE_TO_MEMORY_TYPE_MAP:
        return str(MESSAGE_TO_MEMORY_TYPE_MAP[message_type])

    # Fallback to TEXT for unknown message types
    return str(MemoryMimeType.TEXT)
