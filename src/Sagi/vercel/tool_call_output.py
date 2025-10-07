from typing import List

from api.schema.chats.request import ReferenceChunkType
from utils.camel_model import CamelModel


class RagSearchToolCallOutputItem(CamelModel):
    fileName: str
    fileUrl: str
    type: str


class RagSearchToolCallOutput(CamelModel):
    data: List[RagSearchToolCallOutputItem]


class FilterChunkData(CamelModel):
    included: List[ReferenceChunkType]
    excluded: List[ReferenceChunkType]


class RagFilterToolCallOutput(CamelModel):
    type: str = "RagFilterToolCallOutput"
    data: FilterChunkData


class LoadFileToolCallOutput(CamelModel):
    success: bool
