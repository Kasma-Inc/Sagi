import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import List, Literal, Optional

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from pydantic import BaseModel

from utils.load_config import load_toml_with_env_vars
from workflows.planning_group_chat import PlanningGroupChat


class Step(BaseModel):
    name: str
    description: str
    data_collection_task: Optional[str] = None
    code_executor_task: Optional[str] = None


class PlanningResponse(BaseModel):
    steps: List[Step]


class ReflectionResponse(BaseModel):
    is_complete: Literal["true", "false"]
    reason: str


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


class StepTriageInnerResponse(BaseModel):
    reason: str
    answer: str


class StepTriageResponse(BaseModel):
    instruction_or_question: StepTriageInnerResponse
    next_speaker: StepTriageInnerResponse


class PlanningWorkflow:
    def __init__(self, config_path: str):
        config = load_toml_with_env_vars(config_path)

        config_orchestrator_client = config["model_clients"]["orchestrator_client"]
        if "model_info" in config_orchestrator_client:
            model_info = config_orchestrator_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.model_client = OpenAIChatCompletionClient(
                model=config_orchestrator_client["model"],
                base_url=config_orchestrator_client["base_url"],
                api_key=config_orchestrator_client["api_key"],
                max_tokens=config_orchestrator_client["max_tokens"],
                model_info=model_info,
            )
        else:
            self.model_client = OpenAIChatCompletionClient(
                model=config_orchestrator_client["model"],
                base_url=config_orchestrator_client["base_url"],
                api_key=config_orchestrator_client["api_key"],
                max_tokens=config_orchestrator_client["max_tokens"],
            )

        config_planning_client = config["model_clients"]["planning_client"]
        if "model_info" in config_planning_client:
            model_info = config_planning_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.planning_model_client = OpenAIChatCompletionClient(
                model=config_planning_client["model"],
                base_url=config_planning_client["base_url"],
                api_key=config_planning_client["api_key"],
                response_format=PlanningResponse,
                model_info=model_info,
                max_tokens=config_planning_client["max_tokens"],
            )
        else:
            self.planning_model_client = OpenAIChatCompletionClient(
                model=config_planning_client["model"],
                base_url=config_planning_client["base_url"],
                api_key=config_planning_client["api_key"],
                response_format=PlanningResponse,
                max_tokens=config_planning_client["max_tokens"],
            )

        config_reflection_client = config["model_clients"]["reflection_client"]
        if "model_info" in config_reflection_client:
            model_info = config_reflection_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.reflection_model_client = OpenAIChatCompletionClient(
                model=config_reflection_client["model"],
                base_url=config_reflection_client["base_url"],
                api_key=config_reflection_client["api_key"],
                response_format=ReflectionResponse,
                model_info=model_info,
                max_tokens=config_reflection_client["max_tokens"],
            )
        else:
            self.reflection_model_client = OpenAIChatCompletionClient(
                model=config_reflection_client["model"],
                base_url=config_reflection_client["base_url"],
                api_key=config_reflection_client["api_key"],
                response_format=ReflectionResponse,
                max_tokens=config_reflection_client["max_tokens"],
            )

        config_step_triage_client = config["model_clients"]["step_triage_client"]
        if "model_info" in config_step_triage_client:
            model_info = config_step_triage_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.step_triage_model_client = OpenAIChatCompletionClient(
                model=config_step_triage_client["model"],
                base_url=config_step_triage_client["base_url"],
                api_key=config_step_triage_client["api_key"],
                response_format=StepTriageResponse,
                model_info=model_info,
            )
        else:
            self.step_triage_model_client = OpenAIChatCompletionClient(
                model=config_step_triage_client["model"],
                base_url=config_step_triage_client["base_url"],
                api_key=config_step_triage_client["api_key"],
                response_format=StepTriageResponse,
            )

        config_code_client = config["model_clients"]["code_client"]
        if "model_info" in config_code_client:
            model_info = config_code_client["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            model_info = ModelInfo(**model_info)
        else:
            model_info = None

        if model_info is not None:
            self.code_model_client = OpenAIChatCompletionClient(
                model=config_code_client["model"],
                base_url=config_code_client["base_url"],
                api_key=config_code_client["api_key"],
                model_info=model_info,
                max_tokens=config_code_client["max_tokens"],
            )
        else:
            self.code_model_client = OpenAIChatCompletionClient(
                model=config_code_client["model"],
                base_url=config_code_client["base_url"],
                api_key=config_code_client["api_key"],
                max_tokens=config_code_client["max_tokens"],
            )

    @classmethod
    async def create(
        cls,
        config_path: str,
    ):
        self = cls(config_path)

        self.session_manager = MCPSessionManager()

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
        domain_specific_tools = await mcp_server_tools(prompt_server_params)

        hirag_server_params = StdioServerParams(
            command="uv",
            args=[
                "--directory",
                "mcp_server/hirag_mcp",
                "run",
                "python",
                "server.py",
            ],
            read_timeout_seconds=100,
        )

        self.hirag_retrival = await self.session_manager.create_session(
            "hirag_retrival", create_mcp_server_session(hirag_server_params)
        )
        await self.hirag_retrival.initialize()
        hirag_retrival_tools = await mcp_server_tools(
            hirag_server_params, session=self.hirag_retrival
        )

        rag_agent = AssistantAgent(
            name="retrieval_agent",
            description="a retrieval agent that provides relevant information from the internal database.",
            model_client=self.model_client,
            tools=hirag_retrival_tools,  # type: ignore
            system_message="You are a information retrieval agent that provides relevant information from the internal database.",
        )

        # for new feat: domain specific prompt
        domain_specific_agent = AssistantAgent(
            name="prompt_template_expert",
            model_client=self.model_client,
            tools=domain_specific_tools,  # type: ignore
            system_message="You are a prompt expert that provides structured templates for different domains.",
        )

        general_agent = AssistantAgent(
            name="general_agent",
            model_client=self.model_client,
            description="a general agent that provides answer for simple questions.",
            system_message="You are a general AI assistant that provides answer for simple questions.",
        )

        surfer = AssistantAgent(
            name="web_search",
            description="a web search agent that collect data and relevant information from the web.",
            model_client=self.model_client,
            reflect_on_tool_use=True,  # enable llm summary for contents web search returns
            tools=web_search_tools,  # type: ignore
        )
        work_dir = Path(
            "coding_files"
        )  # the output directory for code generation execution
        code_executor = CodeExecutorAgent(
            name="CodeExecutor",
            description="a code executor agent that handles code related tasks.",
            code_executor=LocalCommandLineCodeExecutor(work_dir=work_dir),
            model_client=self.code_model_client,
            max_retries_on_error=3,
        )

        # Pass prompt_template_agent as a separate parameter
        self.team = PlanningGroupChat(
            participants=[
                surfer,
                code_executor,
                general_agent,
            ],  # can utilize rag_agent
            model_client=self.model_client,
            planning_model_client=self.planning_model_client,
            reflection_model_client=self.reflection_model_client,
            domain_specific_agent=domain_specific_agent,  # Add this parameter
            step_triage_model_client=self.step_triage_model_client,
        )
        return self

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
