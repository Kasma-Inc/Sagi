from typing import List

from utils.camel_model import CamelModel


class RagSearchToolCallOutputItem(CamelModel):
    fileName: str
    fileUrl: str
    type: str


class RagSearchToolCallOutput(CamelModel):
    data: List[RagSearchToolCallOutputItem]


class LoadFileToolCallOutput(CamelModel):
    success: bool
