import os
from contextlib import AsyncExitStack

import pytest
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from dotenv import load_dotenv

from Sagi.tools.web_search_agent import WebSearchAgent

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_web_search_agent():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )

    # Add the following code if you are using NVM
    # nvm_node_path = os.path.expanduser("~/.nvm/versions/node")
    # node_versions = sorted(os.listdir(nvm_node_path), reverse=True)
    # if node_versions:
    #     latest_node = os.path.join(nvm_node_path, node_versions[0], "bin")
    #     os.environ["PATH"] = f"{latest_node}:{os.environ['PATH']}"
    web_search_server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        read_timeout_seconds=60,
    )
    context_manager = create_mcp_server_session(web_search_server_params)
    exit_stack = AsyncExitStack()
    web_search = await exit_stack.enter_async_context(context_manager)
    await web_search.initialize()
    tools = await mcp_server_tools(web_search_server_params, session=web_search)

    agent = WebSearchAgent(
        name="web_search_agent",
        model_client=model_client,
        reflect_on_tool_use=True,
        tools=tools,  # type: ignore
        max_retries=2,
    )

    result = await agent.run(
        task="Please search the latest news about CUHK",
        cancellation_token=CancellationToken(),
    )
    print(result)

    await exit_stack.aclose()
