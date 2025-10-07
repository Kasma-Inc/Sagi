from utils.camel_model import CamelModel


class RagSearchToolCallInput(CamelModel):
    query: str


class RagFilterToolCallInput(CamelModel):
    num_chunks: int


class LoadFileToolCallInput(CamelModel):
    file_name: str
    file_url: str
    media_type: str
