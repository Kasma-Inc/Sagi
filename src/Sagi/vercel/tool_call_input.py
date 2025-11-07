from typing import Literal

from utils.camel_model import CamelModel


class RagSearchToolCallInput(CamelModel):
    type: Literal["ragSearch-input"] = "ragSearch-input"
    query: str


class RagFilterToolCallInput(CamelModel):
    type: Literal["ragFilter-input"] = "ragFilter-input"
    num_chunks: int


class RagFileListToolCallInput(CamelModel):
    type: Literal["ragFileList-input"] = "ragFileList-input"


class RagFileSelectToolCallInput(CamelModel):
    type: Literal["ragFileSelect-input"] = "ragFileSelect-input"
    query: str
    data: list[str]


class RagHeaderSelectToolCallInput(CamelModel):
    type: Literal["ragHeaderSelect-input"] = "ragHeaderSelect-input"
    table_of_contents: list[dict]


class RagRetrievalToolCallInput(CamelModel):
    type: Literal["ragRetrieval-input"] = "ragRetrieval-input"
    data: list[dict]


class LoadFileToolCallInput(CamelModel):
    type: Literal["loadFile-input"] = "loadFile-input"
    file_name: str
    file_url: str
    media_type: str
