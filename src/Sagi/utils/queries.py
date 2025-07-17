import uuid
from datetime import datetime
from typing import Any, Dict, List

from asyncpg import DuplicateTableError
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import JSON, Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


class MultiRoundMemory(SQLModel, table=True):
    __tablename__ = "MultiRoundMemory"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    chatId: str
    messageId: str = Field(default=None, nullable=True)
    content: str
    source: str
    mimeType: str
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


def create_db_engine(postgres_url: str) -> AsyncEngine:
    # connect to postgres db
    # Replace postgres:// with postgresql:// for SQLAlchemy
    postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
    db = create_async_engine(
        postgres_url,
        pool_pre_ping=True,  # tests connections before use
    )
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


async def saveMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
    content: str,
    source: str,
    mime_type: str,
):
    await _ensure_table(session, MultiRoundMemory)
    timestamp = datetime.now().isoformat()
    memory = MultiRoundMemory(
        chatId=chat_id,
        content=content,
        source=source,
        mimeType=mime_type,
        createdAt=timestamp,
    )
    session.add(memory)
    await session.commit()


async def saveMultiRoundMemories(
    session: AsyncSession,
    chat_id: str,
    contents: List[Dict[str, Any]],
):
    await _ensure_table(session, MultiRoundMemory)
    timestamp = datetime.now().isoformat()

    for content_data in contents:
        memory = MultiRoundMemory(
            chatId=chat_id,
            content=content_data["content"],
            source=content_data["source"],
            mimeType=content_data["mime_type"],
            createdAt=timestamp,
        )
        session.add(memory)
        await session.commit()
        await session.refresh(memory)


async def getMultiRoundMemory(
    session: AsyncSession,
    chat_id: str,
) -> List[MultiRoundMemory]:
    await _ensure_table(session, MultiRoundMemory)
    memory = await session.execute(
        select(MultiRoundMemory)
        .where(MultiRoundMemory.chatId == chat_id)
        .order_by(MultiRoundMemory.createdAt)
    )
    return memory.scalars().all()


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


async def getChats(
    session: AsyncSession,
    chat_id: str,
) -> Chats:
    await _ensure_table(session, Chats)
    chat = await session.execute(select(Chats).where(Chats.id == chat_id))
    return chat.scalars().first()
