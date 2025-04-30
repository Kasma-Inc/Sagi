"""
Domain-Specific MCP Server Test Module

This module tests the domain-specific MCP server that provides structured prompt templates for various domains. The script:
1. Initializes and starts the MCP server
2. Creates an AI assistant agent equipped with domain-specific templates
3. Tests the agent with a sample query
4. Extracts and logs the response templates

The test demonstrates how domain-specific templates can be integrated with an agent to generate structured responses for specific use cases.
"""

import asyncio
import json
import logging
import os
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "test.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_domain_specific_prompts():
    """Test the MCP server that provides domain-specific prompt templates."""

    load_dotenv()

    prompt_server_params = StdioServerParams(
        command="uv",
        args=[
            "--directory",
            "mcp_server/domain_specific_mcp",
            "run",
            "python",
            "src/domain_specific_mcp/server.py",
        ],
    )

    logger.info("Starting domain-specific MCP server...")
    domain_specific_templates = await mcp_server_tools(prompt_server_params)
    logger.info("MCP server started successfully!")
    logger.info(
        f"Domain-specific templates initialized:\n{domain_specific_templates}\n"
    )

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
    )

    domain_specific_agent = AssistantAgent(
        name="prompt_template_expert",
        model_client=model_client,
        tools=domain_specific_templates,
        system_message="You are a prompt expert that provides structured templates for different domains.",
    )

    logger.info("Testing domain-specific agent equipped with above MCP server")
    agent_response = await domain_specific_agent.on_messages(
        [TextMessage(content="hello, world", source="tester")],
        CancellationToken(),
    )

    def extract_tool_response(response_content):
        match = re.search(r"text='(.*?)'", response_content)
        if match:
            tool_response = match.group(1)
            return json.loads(tool_response)
        return None

    try:
        templates = extract_tool_response(agent_response.chat_message.content)
        if templates:
            logger.info("Extracted templates:")
            logger.info(json.dumps(templates, indent=2))
    except Exception as e:
        logger.error(f"Error extracting templates: {e}")
    logger.info(
        "Test completed successfully!\nRemember to check whether the selected templates are as expected.\n"
    )


if __name__ == "__main__":
    asyncio.run(test_domain_specific_prompts())
