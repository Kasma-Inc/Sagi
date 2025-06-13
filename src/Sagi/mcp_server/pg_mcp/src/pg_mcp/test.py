"""
PG MCP Tool Server Test Script

This script tests the PostgreSQL MCP Tool server (pg_query). It:
1. Starts the MCP server with pg_query tool
2. Creates an agent with access to that tool
3. Sends a test query via natural language
4. Logs the tool response (SQL query result)
"""

import asyncio
import logging
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from dotenv import load_dotenv

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_pg_query_tool():
    load_dotenv()

    # Step 1: Launch pg_mcp tool server
    prompt_server_params = StdioServerParams(
        command="uv",
        args=[
            "--directory",
            "src/Sagi/mcp_server/pg_mcp/src/pg_mcp",
            "run",
            "/chatbot/Sagi/.venv/bin/python",
            "server.py",
        ],
    )

    logger.info("Starting pg_mcp tool server...")
    pg_tools = await mcp_server_tools(prompt_server_params)
    logger.info("pg_mcp tool server started successfully!")

    # Step 2: Set up model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),  # Optional
    )

    # Step 3: Create agent with pg_query tool
    pg_agent = AssistantAgent(
        name="pg_query_agent",
        model_client=model_client,
        tools=pg_tools,
        system_message="You are a data analyst assistant. Use the tools to query the database and answer questions.",
    )

    # Step 4: Send query to agent
    logger.info("Sending query to agent...")
    user_query = "What are the data in the first six rows of the table transaction_data?"  # You can change this
    response = await pg_agent.on_messages(
        [TextMessage(content=user_query, source="test_user")], CancellationToken()
    )

    logger.info("Agent response:")
    logger.info(response.chat_message.content)

    # Optional: extract tool response if needed
    try:
        content = response.chat_message.content
        if "Query Result:" in content:
            logger.info("Query executed successfully.")
        else:
            logger.warning("No query result found in response.")
    except Exception as e:
        logger.error(f"Error parsing response: {e}")


if __name__ == "__main__":
    asyncio.run(test_pg_query_tool())
