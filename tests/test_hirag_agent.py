import json
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
from hirag_prod import HiRAG
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
    hirag_retrival_tools = [
        tool for tool in hirag_retrival_tools if tool.name == "hi_search"
    ]

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


@pytest.mark.asyncio
async def test_query_all():
    hirag = await HiRAG.create()

    document_path = f"tests/Guide-to-U.S.-Healthcare-System.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    # await hirag.insert_to_kb(
    #     document_path=document_path,
    #     content_type=content_type,
    #     document_meta=document_meta,
    # )
    result = await hirag.query_all(
        query="main stakeholders in U.S. healthcare system 2023"
    )
    assert len(result) > 0


def extract_texts(content):
    texts = []
    for line in content.strip().split("\n"):
        if line:
            array = json.loads(line)
            for item in array:
                if item.get("type") == "text" and "text" in item:
                    texts.append(item["text"])
    return texts


def parse_rag_result(message_content):
    texts = extract_texts(message_content)
    chunks, entities, relations, neighbors = [], [], [], []

    for text in texts:
        query_result_json = json.loads(text)
        chunks.extend(query_result_json.get("chunks", []))
        entities.extend(query_result_json.get("entities", []))
        relations.extend(query_result_json.get("relations", []))
        neighbors.extend(query_result_json.get("neighbors", []))

    chunks = unique_by_key(chunks, key="document_key")
    chunks_str = "\n".join([chunk["text"] for chunk in chunks])

    entities = [
        (
            entity["document_key"],
            entity["text"],
            entity["entity_type"],
            entity["description"],
        )
        for entity in entities
    ]
    neighbors = [
        (
            neighbor["id"],
            neighbor["page_content"],
            neighbor["metadata"]["entity_type"],
            neighbor["metadata"]["description"],
        )
        for neighbor in neighbors
    ]
    entities_with_neighbors = unique_by_first_element(entities + neighbors)
    entities_with_neighbors_str = "\n".join(
        [
            f"{entity[1]} with type {entity[2]} and description {entity[3]}"
            for entity in entities_with_neighbors
        ]
    )

    relations_str = "\n".join(
        [relation["properties"]["description"] for relation in relations]
    )

    # Prepare the information for the LLM to answer the question
    return (
        f"The following is the information you can use to answer the question:\n\n"
        f"Chunks:\n{chunks_str}\n\n"
        f"Entities:\n{entities_with_neighbors_str}\n\n"
        f"Relations:\n{relations_str}\n\n"
    )


def unique_by_key(items, key):
    seen = set()
    result = []
    for item in items:
        identifier = item[key]
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result


def unique_by_first_element(tuples):
    seen = set()
    result = []
    for tup in tuples:
        identifier = tup[0]
        if identifier not in seen:
            seen.add(identifier)
            result.append(tup)
    return result


@pytest.mark.asyncio
async def test_hirag_agent_parse():
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
    hirag_retrival_tools = [
        tool for tool in hirag_retrival_tools if tool.name == "hi_search"
    ]

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
    hirag_message = parse_rag_result(result.messages[-1].content)
    assert hirag_message != ""
