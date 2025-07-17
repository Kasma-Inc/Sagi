import os
from contextlib import AsyncExitStack

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)

from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.utils.prompt import get_general_agent_prompt, get_web_search_agent_prompt
from Sagi.services.global_resource_manager import GlobalResourceManager

DEFAULT_WEB_SEARCH_MAX_RETRIES = 3


class MCPSessionManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}

    async def create_session(self, name: str, context_manager):
        """create and store a session"""
        session = await self.exit_stack.enter_async_context(context_manager)
        self.sessions[name] = session
        return session

    async def close_all(self):
        """close all sessions"""
        await self.exit_stack.aclose()
        self.sessions.clear()


class GeneralChatWorkflow:
    @classmethod
    async def create(
        cls,
        config_path: str,
        web_search: bool = True,
    ):
        self = cls()
        self.session_manager = MCPSessionManager()

        config = load_toml_with_env_vars(config_path)

        # Initialize orchestrator model client using ModelClientService
        model_client_service = GlobalResourceManager.get_model_client_service()
        self.general_model_client = await model_client_service.get_client(
            "general_client", config_path
        )
        web_search_server_params = StdioServerParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        )

        self.web_search = await self.session_manager.create_session(
            "web_search", create_mcp_server_session(web_search_server_params)
        )
        await self.web_search.initialize()
        web_search_tools = await mcp_server_tools(
            web_search_server_params, session=self.web_search
        )

        # Create general agent
        general_agent = AssistantAgent(
            name="general_agent",
            model_client=self.general_model_client,
            description="a general agent that provides answer for simple questions.",
            system_message=get_general_agent_prompt(),
        )

        surfer = WebSearchAgent(
            name="web_search",
            description="a web search agent that collect data and relevant information from the web.",
            system_message=get_web_search_agent_prompt(),
            model_client=self.general_model_client,
            # reflect_on_tool_use=True,  # enable llm summary for contents web search returns
            tools=web_search_tools,
            max_retries=DEFAULT_WEB_SEARCH_MAX_RETRIES,
        )

        if web_search:
            self.team = RoundRobinGroupChat(
                participants=[surfer, general_agent],
                termination_condition=TextMessageTermination("general_agent"),
            )
        else:
            self.team = RoundRobinGroupChat(
                participants=[general_agent],
                termination_condition=TextMessageTermination("general_agent"),
            )
        return self

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
