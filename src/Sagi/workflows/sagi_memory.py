from logging import getLogger
from typing import Any, List, Optional, Union

from autogen_core import CancellationToken, Component, Image
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.models import SystemMessage
from autogen_core.model_context import ChatCompletionContext
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker
from typing_extensions import Self

from Sagi.utils.queries import (
    MultiRoundMemory,
    getMultiRoundMemory,
    saveMultiRoundMemories,
)

logger = getLogger("Sagi.workflows.multi_rounds.SagiMemory")

DB_CONNECTION_MAX_RETRIES = 3


class SagiMemoryConfig(BaseModel):
    chat_id: str
    max_tokens: int


class SagiMemory(Memory, Component[SagiMemoryConfig]):
    component_type = "memory"
    component_provider_override = "Sagi.workflows.multi_rounds.SagiMemory"
    component_config_schema = SagiMemoryConfig
    session_maker: Optional[async_sessionmaker] = None
    chat_id: str
    max_tokens: int

    def __init__(self, chat_id: str, max_tokens: int):
        self.chat_id = chat_id
        self.max_tokens = max_tokens

    def set_session_maker(self, session_maker: async_sessionmaker):
        self.session_maker = session_maker

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        elif isinstance(content, Image):
            raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages or len(messages) == 0:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        memory_query_result = await self.query(messages[-1].content)
        if memory_query_result.results:
            # Format memories as numbered list
            # TODO(klma): FIFO strategy
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(memory_query_result.results, 1)]
            memory_context = "\nRelevant memories:\n" + "\n".join(memory_strings)

            # Add as system message
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=MemoryQueryResult(results=[]))

    async def add(self, contents: Union[MemoryContent, List[MemoryContent]]):
        assert (
            self.session_maker is not None
        ), "Session maker is not set, please call the set_session_maker method"
        async with self.session_maker() as session:
            if session is None:
                raise ValueError("Failed to get database session")

            if isinstance(contents, MemoryContent):
                contents = [contents]

            memory_list = []
            for content in contents:
                source = (
                    content.metadata.get("source", "unknown")
                    if content.metadata
                    else "unknown"
                )
                memory_list.append(
                    {
                        "content": content.content,
                        "mime_type": content.mime_type,
                        "source": source,
                    }
                )
            await saveMultiRoundMemories(session, self.chat_id, memory_list)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        # TODO(kaili): We support to extract all the messages with the same chat_id for now.
        # TODO(kaili): We need to support to query the memory with the query text.

        try:
            assert (
                self.session_maker is not None
            ), "Session maker is not set, please call the set_session_maker method"
            async with self.session_maker() as session:
                # Extract text for query, and we don't use it for now
                # query_text = self._extract_text(query)

                # Get all memories from the database
                results = await getMultiRoundMemory(session, self.chat_id)

                # Convert results to MemoryContent list
                memory_results: List[MemoryContent] = (
                    await self.data_model_to_memory_content(results)
                )

                return MemoryQueryResult(results=memory_results)

        except Exception as e:
            logger.error(f"Failed to query SagiMemory: {e}")
            raise

    async def data_model_to_memory_content(
        self, memories: List[MultiRoundMemory]
    ) -> List[MemoryContent]:
        memory_contents: List[MemoryContent] = []
        for data_model in memories:
            metadata = {
                "source": data_model.source,
                "created_at": data_model.createdAt,
            }
            content = self._extract_text(data_model.content)
            memory_contents.append(
                MemoryContent(
                    content=content,
                    mime_type=data_model.mimeType,
                    metadata=metadata,
                )
            )
        return memory_contents

    @classmethod
    def _from_config(cls, config: SagiMemoryConfig) -> Self:
        return cls(config.chat_id, config.max_tokens)

    async def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass
