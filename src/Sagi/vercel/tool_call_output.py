from typing import List, Literal

from api.schema.chats.request import ReferenceChunkType
from utils.camel_model import CamelModel


class RagSearchToolCallOutputItem(CamelModel):
    fileName: str
    fileUrl: str
    type: str


class RagSearchToolCallOutput(CamelModel):
    type: Literal["ragSearch-output"] = "ragSearch-output"
    data: List[RagSearchToolCallOutputItem]


class FilterChunkData(CamelModel):
    included: List[ReferenceChunkType]
    excluded: List[ReferenceChunkType]


class RagFilterToolCallOutput(CamelModel):
    type: Literal["ragFilter-output"] = "ragFilter-output"
    data: FilterChunkData


class LoadFileToolCallOutput(CamelModel):
    type: Literal["loadFile-output"] = "loadFile-output"
    success: bool
