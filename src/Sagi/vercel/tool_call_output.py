from typing import List, Literal

from api.schema.chats.request import ReferenceChunkType
from utils.camel_model import CamelModel


class FileToolCallItem(CamelModel):
    fileName: str
    fileUrl: str
    type: str


class RagSearchToolCallOutput(CamelModel):
    type: Literal["ragSearch-output"] = "ragSearch-output"
    data: List[FileToolCallItem]


class FilterChunkData(CamelModel):
    included: List[ReferenceChunkType]
    excluded: List[ReferenceChunkType]


class RagFilterToolCallOutput(CamelModel):
    type: Literal["ragFilter-output"] = "ragFilter-output"
    data: FilterChunkData


class RagFileListToolCallOutput(CamelModel):
    type: Literal["ragFileList-output"] = "ragFileList-output"
    data: List[FileToolCallItem]


class RagFileSelectToolCallOutput(CamelModel):
    type: Literal["ragFileSelect-output"] = "ragFileSelect-output"
    response_text: str
    data: List[FileToolCallItem]


class RagHeaderSelectToolCallOutput(CamelModel):
    type: Literal["ragHeaderSelect-output"] = "ragHeaderSelect-output"
    response_text: str
    headers: List[dict]


class RagRetrievalToolCallOutput(CamelModel):
    type: Literal["ragRetrieval-output"] = "ragRetrieval-output"
    data: List[ReferenceChunkType]


class LoadFileToolCallOutput(CamelModel):
    type: Literal["loadFile-output"] = "loadFile-output"
    success: bool
