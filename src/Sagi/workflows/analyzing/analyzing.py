import os
from contextlib import AsyncExitStack
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    mcp_server_tools,
)
from pydantic import BaseModel

from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.workflows.analyzing.analyzing_group_chat import AnalyzingGroupChat


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


class AnalyzingWorkflow:
    def __init__(self, config_path: str):

        config = load_toml_with_env_vars(config_path)

        config_analyze_client = config["model_clients"]["analyze_client"]
        if "model_info" in config_analyze_client:
            model_info = config_analyze_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.analyze_model_client = OpenAIChatCompletionClient(
                model=config_analyze_client["model"],
                base_url=config_analyze_client["base_url"],
                api_key=config_analyze_client["api_key"],
                max_tokens=config_analyze_client["max_tokens"],
                model_info=model_info,
            )
        else:
            self.analyze_model_client = OpenAIChatCompletionClient(
                model=config_analyze_client["model"],
                base_url=config_analyze_client["base_url"],
                api_key=config_analyze_client["api_key"],
                max_tokens=config_analyze_client["max_tokens"],
            )

        class StepTriageNextSpeakerResponse(BaseModel):
            instruction: str
            answer: Literal["pg_agent", "general_agent"]  # type: ignore

        class StepTriageResponse(BaseModel):
            next_speaker: StepTriageNextSpeakerResponse

        self.step_triage_model_client = OpenAIChatCompletionClient(
            model=config["model_clients"]["step_triage_client"]["model"],
            base_url=config["model_clients"]["step_triage_client"]["base_url"],
            api_key=config["model_clients"]["step_triage_client"]["api_key"],
            max_tokens=config["model_clients"]["step_triage_client"]["max_tokens"],
            response_format=StepTriageResponse,
        )

        config_pg_client = config["model_clients"]["pg_client"]
        if "model_info" in config_pg_client:
            model_info = config_pg_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.pg_model_client = OpenAIChatCompletionClient(
                model=config_pg_client["model"],
                base_url=config_pg_client["base_url"],
                api_key=config_pg_client["api_key"],
                max_tokens=config_pg_client["max_tokens"],
                model_info=model_info,
            )
        else:
            self.pg_model_client = OpenAIChatCompletionClient(
                model=config_pg_client["model"],
                base_url=config_pg_client["base_url"],
                api_key=config_pg_client["api_key"],
                max_tokens=config_pg_client["max_tokens"],
            )

    @classmethod
    async def create(
        cls,
        config_path: str,
    ):
        self = cls(config_path)

        self.session_manager = MCPSessionManager()

        # set env MCP_SERVER_PATH, default is "src/Sagi/mcp_server/"
        mcp_server_path = os.getenv("MCP_SERVER_PATH", "src/Sagi/mcp_server/")
        prompt_server_params = StdioServerParams(
            command="uv",
            args=[
                "--directory",
                os.path.join(mcp_server_path, "pg_mcp/src/pg_mcp"),
                "run",
                "/chatbot/Sagi/.venv/bin/python",
                "server.py",
            ],
            env={"DATABASE_URL": os.getenv("DATABASE_URL")},
        )
        pg_tools = await mcp_server_tools(prompt_server_params)

        pg_agent = AssistantAgent(
            name="pg_agent",
            description="An agent that answers questions by querying a PostgreSQL database.",
            model_client=self.pg_model_client,
            tools=pg_tools,
            system_message=(
                "You are a database expert. Use the available tools to query a PostgreSQL "
                "database and return concise, correct results. Format SQL properly. "
                "Only use the provided tools to answer questions about the database."
            ),
        )

        general_agent = AssistantAgent(
            name="general_agent",
            model_client=self.analyze_model_client,
            description="a general agent that provides answer for simple questions.",
            system_message="You are a general AI assistant that provides answer for simple questions.",
        )

        self.team = AnalyzingGroupChat(
            participants=[
                pg_agent,
                general_agent,
            ],
            analyzing_model_client=self.analyze_model_client,
            pg_model_client=self.pg_model_client,
            step_triage_model_client=self.step_triage_model_client,
        )
        return self

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)

    async def cleanup(self):
        # """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
