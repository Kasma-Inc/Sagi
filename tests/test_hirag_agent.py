import os

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from dotenv import load_dotenv
from hirag_prod._llm import EmbeddingService
from hirag_prod.storage.lancedb import LanceDB
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

from Sagi.workflows.planning import MCPSessionManager

session_manager = MCPSessionManager()


@pytest.mark.asyncio
async def test_hirag_agent():
    load_dotenv("/chatbot/.env")
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )
    hirag_server_params = StdioServerParams(
        command="mcp-hirag-tool",
        args=[],
        read_timeout_seconds=100,
        env={
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
        },
    )

    hirag_retrival = await session_manager.create_session(
        "hirag_retrival", create_mcp_server_session(hirag_server_params)
    )
    await hirag_retrival.initialize()
    hirag_retrival_tools = await mcp_server_tools(
        hirag_server_params, session=hirag_retrival
    )

    rag_agent = AssistantAgent(
        name="retrieval_agent",
        description="a retrieval agent that provides relevant information from the internal database.",
        model_client=model_client,
        tools=hirag_retrival_tools,  # type: ignore
        system_message="You are a information retrieval agent that provides relevant information from the internal database.",
    )
    result = await rag_agent.run(
        task="Please search the information about Insurance in the US",
        cancellation_token=CancellationToken(),
    )
    tool_call_execution_result = result.messages[-2].content[0].content
    tool_call_name = result.messages[-2].content[0].name
    assert tool_call_name in ["hi_search", "naive_search"]
    assert tool_call_execution_result is not None


@pytest.mark.asyncio
async def test_lancedb():
    load_dotenv("/chatbot/.env")
    strategy_provider = RetrievalStrategyProvider()
    lance_db = await LanceDB.create(
        embedding_func=EmbeddingService().create_embeddings,
        db_url="/kb/hirag.db",
        strategy_provider=strategy_provider,
    )
    table_chunks = await lance_db.get_table("chunks")
    table_entities = await lance_db.get_table("entities")
    assert table_chunks.num_rows > 0
    assert table_entities.num_rows > 0
    async_table = await lance_db.db.open_table("chunks")
    result = await lance_db.query(
        query="main stakeholders in U.S. healthcare system 2023",
        table=async_table,
        topk=5,
    )
    assert len(result) > 0
