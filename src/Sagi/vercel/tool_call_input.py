from typing import Literal

from utils.camel_model import CamelModel


class RagSearchToolCallInput(CamelModel):
    type: Literal["ragSearch-input"] = "ragSearch-input"
    query: str


class RagFilterToolCallInput(CamelModel):
    type: Literal["ragFilter-input"] = "ragFilter-input"
    num_chunks: int


class LoadFileToolCallInput(CamelModel):
    type: Literal["loadFile-input"] = "loadFile-input"
    file_name: str
    file_url: str
    media_type: str
