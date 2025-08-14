import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from asyncpg import DuplicateTableError
from autogen_core.memory import (
    MemoryContent,
    MemoryMimeType,
)

# Embedding service from HiRAG for generating embeddings
from hirag_prod._llm import EmbeddingService, LocalEmbeddingService
from pgvector.sqlalchemy import Vector
from sqlalchemy import inspect, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import JSON, Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from Sagi.utils.token_usage import count_tokens_messages

if os.getenv("EMBEDDING_SERVICE_TYPE") == "local":
    memory_embedding_service = LocalEmbeddingService()
else:
    memory_embedding_service = EmbeddingService()

TEST_TIMING = True


# Timing utilities
@contextmanager
def time_operation(operation_name: str):
    """Context manager for timing synchronous operations."""
    if not TEST_TIMING:
        yield
        return
    
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[TIMING] {operation_name}: {execution_time:.4f} seconds")


@asynccontextmanager
async def async_time_operation(operation_name: str):
    """Context manager for timing asynchronous operations."""
    if not TEST_TIMING:
        yield
        return
    
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[TIMING] {operation_name}: {execution_time:.4f} seconds")


def time_function(func_name: str = None):
    """Decorator for timing functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not TEST_TIMING:
                return await func(*args, **kwargs)
            
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"[TIMING] {name}: {execution_time:.4f} seconds")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not TEST_TIMING:
                return func(*args, **kwargs)
            
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"[TIMING] {name}: {execution_time:.4f} seconds")
        
        # Return appropriate wrapper based on whether function is async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

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


class Chats(SQLModel, table=True):
    __tablename__ = "Chats"
    id: str = Field(default=None, primary_key=True)
    createdAt: str
    title: str
    userId: str
    modelName: str
    modelConfig: Dict[str, Any] = Field(sa_type=JSON)
    modelClientStream: bool = True
    systemPrompt: str = Field(default=None)
    visibility: str = "private"


@time_function("create_db_engine")
async def create_db_engine(postgres_url: str) -> AsyncEngine:
    """
    Create database engine and ensure vector extension exists.
    """
    # connect to postgres db
    # Replace postgres:// with postgresql:// for SQLAlchemy
    if postgres_url.startswith("postgres://"):
        postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif postgres_url.startswith("postgresql://"):
        postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif postgres_url.startswith("postgresql+asyncpg://"):
        pass
    else:
        raise ValueError(
            "Invalid PostgreSQL URL format. Must start with 'postgresql://' or 'postgresql+asyncpg://'."
        )

    db = create_async_engine(
        postgres_url,
        pool_pre_ping=True,  # tests connections before use
    )

    # Create the vector extension if it doesn't exist
    async with db.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    return db


async def _ensure_table(session: AsyncSession, table) -> None:
    """
    Create table on first access.
    SQLModel / SQLAlchemy's DDL calls can only be executed in a synchronous context,
    so we need to proxy to the synchronous engine via `run_sync()`.
    """

    def _sync_create(sync_session: AsyncSession):
        # Use the inspector from sqlalchemy to check if table exists
        engine = sync_session.get_bind()
        if not inspect(engine).has_table(table.__tablename__):
            try:
                SQLModel.metadata.create_all(engine, tables=[table.__table__])
            except ProgrammingError as e:
                if isinstance(e.__cause__.__cause__, DuplicateTableError):
                    pass
                else:
                    raise

    await session.run_sync(_sync_create)


@time_function("saveMultiRoundMemory")
async def saveMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
    content: str,
    source: str,
    mime_type: str,
):
    await _ensure_table(session, MultiRoundMemory)
    timestamp = datetime.now().isoformat()

    # Generate embedding for the content using HiRAG's embedding service
    try:
        global memory_embedding_service
        content_embedding = await memory_embedding_service.create_embeddings([content])
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


@time_function("saveMultiRoundMemories")
async def saveMultiRoundMemories(
    session: AsyncSession,
    chat_id: str,
    contents: List[Dict[str, Any]],
):
    await _ensure_table(session, MultiRoundMemory)

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


@time_function("getMultiRoundMemory")
async def getMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    context_window: Optional[int] = None,
) -> List[MultiRoundMemory]:
    await _ensure_table(session, MultiRoundMemory)
    
    # first get everything back and test if exceeds context window
    memories = await session.execute(
        select(MultiRoundMemory)
        .where(MultiRoundMemory.chatId == chat_id)
        .order_by(MultiRoundMemory.createdAt)
    )
    all_memories = memories.scalars().all()
    total_tokens = sum(count_tokens_messages([{"content": mem.content}], model=model_name) for mem in all_memories)

    if total_tokens <= context_window:
        return all_memories

    print(f"Total tokens {total_tokens} exceed context window {context_window}")

    # only if memory exceeds limit && query_text, context_window and model_name are provided, we can rank and filter memories
    if query_text and context_window and model_name:
        global memory_embedding_service
        try:
            # Generate embedding for the query text
            query_embedding = await memory_embedding_service.create_embeddings(
                [query_text]
            )
            if query_embedding is None or len(query_embedding) == 0:
                return []

            # Use the embedding to filter memories by similarity
            async with async_time_operation("Database similarity search"):
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
            async with async_time_operation("Memory filtering by context window"):
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


@time_function("saveChats")
async def saveChats(
    session: AsyncSession,
    chat_id: str,
    title: str,
    user_id: str,
    model_name: str,
    model_config: Dict[str, Any],
    model_client_stream: bool,
    system_prompt: str,
    visibility: str = "private",
):
    await _ensure_table(session, Chats)
    chat = Chats(
        id=chat_id,
        createdAt=datetime.now().isoformat(),
        title=title,
        userId=user_id,
        modelName=model_name,
        modelConfig=model_config,
        modelClientStream=model_client_stream,
        systemPrompt=system_prompt,
        visibility=visibility,
    )
    session.add(chat)
    await session.commit()


@time_function("getChats")
async def getChats(
    session: AsyncSession,
    chat_id: str,
) -> Chats:
    await _ensure_table(session, Chats)
    chat = await session.execute(select(Chats).where(Chats.id == chat_id))
    return chat.scalars().first()
