import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Embedding service from HiRAG for generating embeddings
from hirag_prod._llm import EmbeddingService, LocalEmbeddingService
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from Sagi.utils.token_usage import count_tokens_messages

EMBEDDING_SERVICE: Optional[Union[LocalEmbeddingService, EmbeddingService]] = None


def get_memory_embedding_service():
    global EMBEDDING_SERVICE
    if not EMBEDDING_SERVICE:
        if os.getenv("EMBEDDING_SERVICE_TYPE") == "local":
            EMBEDDING_SERVICE = LocalEmbeddingService()
        else:
            EMBEDDING_SERVICE = EmbeddingService()
    return EMBEDDING_SERVICE


class MultiRoundMemory(SQLModel, table=True):
    __tablename__ = "MultiRoundMemory"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    chatId: str
    messageId: str = Field(default=None, nullable=True)
    content: str
    source: str
    mimeType: str
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_type=Vector(int(os.getenv("EMBEDDING_DIM", "4096"))),
        nullable=True,
    )
    createdAt: str = Field(default=datetime.now().isoformat())


async def saveMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
    content: str,
    source: str,
    mime_type: str,
):
    timestamp = datetime.now().isoformat()

    # Generate embedding for the content using HiRAG's embedding service
    try:
        content_embedding = await get_memory_embedding_service().create_embeddings(
            [content]
        )
        # Extract the embedding vector from the response
        embedding = content_embedding[0] if content_embedding else None
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        embedding = None

    memory = MultiRoundMemory(
        chatId=chat_id,
        content=content,
        source=source,
        mimeType=mime_type,
        embedding=embedding,
        createdAt=timestamp,
    )
    session.add(memory)
    await session.commit()


async def saveMultiRoundMemories(
    session: AsyncSession,
    chat_id: str,
    contents: List[Dict[str, Any]],
):
    # Initialize embedding service once for batch processing
    try:
        memory_embedding_service = LocalEmbeddingService()
        # Extract all content strings for batch embedding generation
        content_texts = [content_data["content"] for content_data in contents]
        # Generate embeddings in batch for efficiency
        embeddings = await memory_embedding_service.create_embeddings(content_texts)
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        embeddings = [None] * len(contents)

    for i, content_data in enumerate(contents):
        # Use the corresponding embedding from the batch
        timestamp = datetime.now().isoformat()
        embedding = embeddings[i] if i < len(embeddings) else None

        memory = MultiRoundMemory(
            chatId=chat_id,
            content=content_data["content"],
            source=content_data["source"],
            mimeType=content_data["mime_type"],
            embedding=embedding,
            createdAt=timestamp,
        )
        session.add(memory)
        await session.commit()
        await session.refresh(memory)


async def getMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    context_window: Optional[int] = None,
) -> List[MultiRoundMemory]:
    # first get everything back and test if exceeds context window
    memories = await session.execute(
        select(MultiRoundMemory)
        .where(MultiRoundMemory.chatId == chat_id)
        .order_by(MultiRoundMemory.createdAt)
    )
    all_memories = memories.scalars().all()
    total_tokens = sum(
        count_tokens_messages([{"content": mem.content}], model=model_name)
        for mem in all_memories
    )

    if total_tokens <= context_window:
        return all_memories

    print(f"Total tokens {total_tokens} exceed context window {context_window}")

    # only if memory exceeds limit && query_text, context_window and model_name are provided, we can rank and filter memories
    if query_text and context_window and model_name:
        try:
            # Generate embedding for the query text
            query_embedding = await get_memory_embedding_service().create_embeddings(
                [query_text]
            )
            if query_embedding is None or len(query_embedding) == 0:
                return []

            # Use the embedding to filter memories by similarity
            query_vector = query_embedding[0]
            memory = await session.execute(
                select(MultiRoundMemory)
                .where(MultiRoundMemory.chatId == chat_id)
                .where(MultiRoundMemory.embedding.is_not(None))
                .order_by(MultiRoundMemory.embedding.op("<=>")(query_vector))
            )

            # Show Debug information
            # print(f"Query vector: {query_vector}")
            # print(f"Context window: {context_window}")

            # Calculate the context length and remove memories that exceed the context window
            memories = memory.scalars().all()
            print(f"Number of memories found: {len(memories)}")

            # Filter memories based on token count and context window
            total_tokens = 0
            filtered_memories = []
            for mem in memories:
                print(
                    f"Processing memory ID: {mem.id}, Created At: {mem.createdAt}, Content: {mem.content}",
                    end=", ",
                )
                content_tokens = count_tokens_messages(
                    [{"content": mem.content}], model=model_name
                )
                # Debug information
                print(f"Tokens: {content_tokens}")

                if total_tokens + content_tokens <= context_window:
                    filtered_memories.append(mem)
                    total_tokens += content_tokens
                else:
                    print(
                        f"Skipping memories after {mem.id} due to exceeding context window. "
                        f"Total tokens: {total_tokens}, Content tokens: {content_tokens}"
                    )
                    break

            # Sort memories by creation time
            filtered_memories.sort(key=lambda x: x.createdAt)

            # List out the filtered memories for debugging
            # print(f"Number of memories after filtered: {len(memories)}")
            # for mem in filtered_memories:
            #     print(
            #         f"Memory ID: {mem.id}, Created At: {mem.createdAt}, Content: {mem.content[:50]}"
            #     )

            return filtered_memories

        except Exception as e:
            print(f"Failed to query memories with embedding: {e}")
            return []

    return all_memories
