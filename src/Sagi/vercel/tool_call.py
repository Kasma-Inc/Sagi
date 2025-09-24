from typing import Any, Dict

from utils.camel_model import CamelModel


class ToolCall(CamelModel):
    pass


class ToolInputStart(ToolCall):
    tool_name: str


class ToolInputDelta(ToolCall):
    input_text_delta: str


class ToolInputAvailable(ToolCall):
    input: Dict[str, Any]


class ToolOutputAvailable(ToolCall):
    output: Dict[str, Any]
