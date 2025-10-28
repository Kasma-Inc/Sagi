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
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from configs.functions import get_llm_context_window
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker
from typing_extensions import Self

from Sagi.utils.chat_template import format_memory_to_string
from Sagi.utils.queries import (
    MultiRoundMemory,
    dropMultiRoundMemory,
    getMultiRoundMemory,
    saveMultiRoundMemories,
)
from Sagi.utils.token_usage import count_tokens_messages

logger = getLogger("Sagi.workflows.multi_rounds.SagiMemory")

DB_CONNECTION_MAX_RETRIES = 3
CONTEXT_WINDOW_BUFFER = 100


class SagiMemoryConfig(BaseModel):
    chat_id: str
    model_name: str


class SagiMemory(Memory, Component[SagiMemoryConfig]):
    component_type = "memory"
    component_provider_override = "Sagi.workflows.multi_rounds.SagiMemory"
    component_config_schema = SagiMemoryConfig
    session_maker: Optional[async_sessionmaker] = None
    chat_id: str
    model_name: str

    def __init__(self, chat_id: str, model_name: str):
        self.chat_id = chat_id
        self.model_name = model_name

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

    async def _drop_old_messages(
        self, model_context: ChatCompletionContext, memories: List[MemoryContent]
    ):
        """
        Drop old messages from the context to keep the context window size.
        Args:
            model_context: The model context to drop old messages from
            memories: The memories to add to the context

        Returns:
            The new context with the old messages dropped
        """
        # FIFO strategy:
        # 1. Count the number of tokens in the memory
        context = await model_context.get_messages()
        total_tokens = count_tokens_messages(context, self.model_name)
        # 2. If the number of tokens is less than the max tokens, add the memory to the context
        memory_to_add = []
        # memories = memory_query_result.results
        context_window = get_llm_context_window(self.model_name)
        assert context_window is not None, "Context window is not set"

        if total_tokens < context_window - CONTEXT_WINDOW_BUFFER:
            for memory in reversed(memories):
                num_tokens = count_tokens_messages([memory], self.model_name)
                total_tokens += num_tokens
                if total_tokens > context_window - CONTEXT_WINDOW_BUFFER:
                    break
                memory_to_add.append(memory)

        # Reverse the order to get the original order (since we reversed when iterating)
        memory_to_add.reverse()
        memory_context = format_memory_to_string(memory_to_add, self.model_name)
        return memory_context

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """
        Update the model context with the latest memories.
        Args:
            model_context: The model context to update
        Returns:
            UpdateContextResult with the updated memories
        """
        messages = await model_context.get_messages()
        if not messages or len(messages) == 0:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        memory_query_result = await self.query(messages[-1].content)
        if memory_query_result.results:
            memories = memory_query_result.results
            memory_context = await self._drop_old_messages(model_context, memories)
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
                metadata = content.metadata or {}
                source = metadata.get("source", "unknown")
                message_id = metadata.get("message_id", None)

                memory_list.append(
                    {
                        "content": content.content,
                        "mime_type": content.mime_type,
                        "source": source,
                        "message_id": message_id,
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
                query_text = self._extract_text(query)

                context_window = get_llm_context_window(self.model_name)
                assert context_window is not None, "Context window is not set"

                # Get memories from the database via similarity search with the query and not exceed the context window
                results = await getMultiRoundMemory(
                    session,
                    self.chat_id,
                    model_name=self.model_name,
                    query_text=query_text,
                    context_window=context_window,
                )

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
        return cls(config.chat_id, config.model_name)

    async def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def drop_messages(self, message_ids: List[str]) -> None:
        async with self.session_maker() as session:
            await dropMultiRoundMemory(session, message_ids)
