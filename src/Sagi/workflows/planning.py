import os
from contextlib import AsyncExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from pydantic import BaseModel

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.json_handler import get_template_num
from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.workflows.planning_group_chat import PlanningGroupChat

DEFAULT_WORK_DIR = "coding_files"
DEFAULT_MCP_SERVER_PATH = "src/Sagi/mcp_server/"
DEFAULT_WEB_SEARCH_MAX_RETRIES = 3
DEFAULT_CODE_MAX_RETRIES = 3


class Slide(BaseModel):
    category: str
    description: str


class HighLevelPlanPPT(BaseModel):
    slides: List[Slide]


class Group(BaseModel):
    name: str
    description: str
    data_collection_task: Optional[str] = None
    code_executor_task: Optional[str] = None


class PlanningResponse(BaseModel):
    groups: List[Group]


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


T = TypeVar("T", bound=BaseModel)


class ModelClientFactory:
    @staticmethod
    def _init_model_info(client_config: Dict[str, Any]) -> Optional[ModelInfo]:
        if "model_info" in client_config:
            model_info = client_config["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            return ModelInfo(**model_info)
        return None

    @classmethod
    def create_model_client(
        cls,
        client_config: Dict[str, Any],
        response_format: Optional[Type[T]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> OpenAIChatCompletionClient:
        model_info = cls._init_model_info(client_config)
        client_kwargs = {
            "model": client_config["model"],
            "base_url": client_config["base_url"],
            "api_key": client_config["api_key"],
            "model_info": model_info,
            "max_tokens": client_config["max_tokens"],
        }

        if response_format:
            client_kwargs["response_format"] = response_format

        if parallel_tool_calls is not None:
            client_kwargs["parallel_tool_calls"] = parallel_tool_calls

        return OpenAIChatCompletionClient(**client_kwargs)


class PlanningWorkflow:
    def __init__(self):
        pass

    @classmethod
    async def create(
        cls,
        config_path: str,
        team_config_path: str,
        template_work_dir: str | None = None,
    ):
        self = cls()

        config = load_toml_with_env_vars(config_path)
        team_config = load_toml_with_env_vars(team_config_path)

        # TeamMember enum dynamically from team.toml
        team_members = list(team_config["team"].values())
        TeamMembers = Enum("TeamMembers", team_members)

        class StepTriageNextSpeakerResponse(BaseModel):
            reason: str
            answer: TeamMembers

        class StepTriageResponse(BaseModel):
            instruction_or_question: StepTriageInnerResponse
            next_speaker: StepTriageNextSpeakerResponse

        # Initialize all model clients using ModelClientFactory
        config_orchestrator_client = config["model_clients"]["orchestrator_client"]
        self.orchestrator_model_client = ModelClientFactory.create_model_client(
            config_orchestrator_client
        )

        config_reflection_client = config["model_clients"]["reflection_client"]
        self.reflection_model_client = ModelClientFactory.create_model_client(
            config_reflection_client, response_format=ReflectionResponse
        )

        config_step_triage_client = config["model_clients"]["step_triage_client"]
        self.step_triage_model_client = ModelClientFactory.create_model_client(
            config_step_triage_client, response_format=StepTriageResponse
        )

        config_code_client = config["model_clients"]["code_client"]
        self.code_model_client = ModelClientFactory.create_model_client(
            config_code_client
        )

        config_single_tool_use_client = config["model_clients"][
            "single_tool_use_client"
        ]
        self.single_tool_use_model_client = ModelClientFactory.create_model_client(
            config_single_tool_use_client,
            parallel_tool_calls=config_single_tool_use_client.get(
                "parallel_tool_calls"
            ),
        )

        config_planning_client = config["model_clients"]["planning_client"]
        self.planning_model_client = ModelClientFactory.create_model_client(
            config_planning_client, response_format=PlanningResponse
        )

        # Initialize template based planning client using the same config as planning client
        self.template_based_planning_model_client = (
            ModelClientFactory.create_model_client(
                config_planning_client, response_format=HighLevelPlanPPT
            )
        )

        # Initialize single group planning client using the same config as planning client
        self.single_group_planning_model_client = (
            ModelClientFactory.create_model_client(
                config_planning_client, response_format=Group
            )
        )

        # Initialize template selection client if template_work_dir is provided
        if template_work_dir is not None:
            template_num = get_template_num(
                os.path.join(template_work_dir, "slide_induction.json")
            )

            template_choices = [f"template_{i}" for i in range(1, template_num + 1)]
            TemplateList = Enum("TemplateList", template_choices)

            class TemplateSelection(BaseModel):
                template_id: TemplateList

            self.template_selection_model_client = (
                ModelClientFactory.create_model_client(
                    config_planning_client, response_format=TemplateSelection
                )
            )

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

        # set env MCP_SERVER_PATH, default is "src/Sagi/mcp_server/"
        mcp_server_path = os.getenv("MCP_SERVER_PATH", DEFAULT_MCP_SERVER_PATH)
        prompt_server_params = StdioServerParams(
            command="uv",
            args=[
                "--directory",
                os.path.join(
                    mcp_server_path, "domain_specific_mcp/src/domain_specific_mcp"
                ),
                "run",
                "python",
                "server.py",
            ],
        )
        domain_specific_tools = await mcp_server_tools(prompt_server_params)

        # ========== Uncomment below code block if you want to use rag_agent ==========
        # hirag_server_params = StdioServerParams(
        #     command="uv",
        #     args=[
        #         "--directory",
        #         os.path.join(mcp_server_path, "hirag_mcp"),
        #         "run",
        #         "python",
        #         "server.py",
        #     ],
        #     read_timeout_seconds=100,
        # )

        # self.hirag_retrival = await self.session_manager.create_session(
        #     "hirag_retrival", create_mcp_server_session(hirag_server_params)
        # )
        # await self.hirag_retrival.initialize()
        # hirag_retrival_tools = await mcp_server_tools(
        #     hirag_server_params, session=self.hirag_retrival
        # )

        # rag_agent = AssistantAgent(
        #     name="retrieval_agent",
        #     description="a retrieval agent that provides relevant information from the internal database.",
        #     model_client=self.orchestrator_model_client,
        #     tools=hirag_retrival_tools,  # type: ignore
        #     system_message="You are a information retrieval agent that provides relevant information from the internal database.",
        # )
        # ========== Uncomment above code block if you want to use rag_agent ==========

        # for new feat: domain specific prompt
        domain_specific_agent = AssistantAgent(
            name="prompt_template_expert",
            model_client=self.single_tool_use_model_client,
            tools=domain_specific_tools,  # type: ignore
            system_message="You are a prompt expert that provides structured templates for different domains.",
        )

        general_agent = AssistantAgent(
            name="general_agent",
            model_client=self.orchestrator_model_client,
            description="a general agent that provides answer for simple questions.",
            system_message="You are a general AI assistant that provides answer for simple questions.",
        )

        surfer = WebSearchAgent(
            name="web_search",
            description="a web search agent that collect data and relevant information from the web.",
            model_client=self.orchestrator_model_client,
            # reflect_on_tool_use=True,  # enable llm summary for contents web search returns
            tools=web_search_tools,  # type: ignore
            max_retries=DEFAULT_WEB_SEARCH_MAX_RETRIES,
        )
        work_dir = Path(
            DEFAULT_WORK_DIR
        )  # the output directory for code generation execution
        code_executor = StreamCodeExecutorAgent(
            name="CodeExecutor",
            description="a code executor agent that handles code related tasks.",
            system_message="You are a Code Execution Agent. Your role is to generate and execute Python code and shell scripts based on user instructions, ensuring correctness, efficiency, and minimal errors. Handle edge cases gracefully. Python code should be provided in ```python code blocks, and sh shell scripts should be provided in ```sh code blocks for execution.",
            # stream_code_executor=StreamLocalCommandLineCodeExecutor(work_dir=work_dir),
            stream_code_executor=StreamDockerCommandLineCodeExecutor(
                work_dir=work_dir,
                bind_dir=(
                    os.getenv("HOST_PATH") + "/" + str(work_dir)
                    if os.getenv("ENVIRONMENT") == "docker"
                    else work_dir
                ),
            ),
            model_client=self.code_model_client,
            max_retries_on_error=DEFAULT_CODE_MAX_RETRIES,
        )

        # mapping of team member names to their agent instances
        agent_mapping: Dict[str, Any] = {
            "web_search": surfer,
            "CodeExecutor": code_executor,
            "general_agent": general_agent,
        }

        # participants list dynamically based on team.toml
        participants = []
        for member in team_members:
            if member in agent_mapping:
                participants.append(agent_mapping[member])

        # Pass prompt_template_agent as a separate parameter
        self.team = PlanningGroupChat(
            participants=participants,
            orchestrator_model_client=self.orchestrator_model_client,
            planning_model_client=self.planning_model_client,
            reflection_model_client=self.reflection_model_client,
            domain_specific_agent=domain_specific_agent,  # Add this parameter
            step_triage_model_client=self.step_triage_model_client,
            template_based_planning_model_client=self.template_based_planning_model_client,
            template_selection_model_client=(
                self.template_selection_model_client
                if template_work_dir is not None
                else None
            ),
            single_group_planning_model_client=self.single_group_planning_model_client,
            template_work_dir=template_work_dir,  # Add template work directory parameter
        )
        return self

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
