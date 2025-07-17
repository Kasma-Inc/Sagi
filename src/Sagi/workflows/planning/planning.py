import logging
import os
from contextlib import AsyncExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar

from autogen_agentchat.agents import AssistantAgent

# ModelFamily, ModelInfo removed as they're now handled in ModelClientFactory
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from mcp import ClientSession
from pydantic import BaseModel

from Sagi.services.global_resource_manager import GlobalResourceManager
from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.json_handler import get_template_num
from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.utils.prompt import (
    get_code_executor_prompt,
    get_code_executor_prompt_cn,
    get_domain_specific_agent_prompt,
    get_domain_specific_agent_prompt_cn,
    get_general_agent_prompt,
    get_general_agent_prompt_cn,
    get_rag_agent_prompt,
    get_rag_agent_prompt_cn,
)
from Sagi.workflows.planning.planning_group_chat import PlanningGroupChat

DEFAULT_WORK_DIR = "coding_files"
DEFAULT_MCP_SERVER_PATH = "src/Sagi/mcp_server/"
DEFAULT_WEB_SEARCH_MAX_RETRIES = 3
DEFAULT_CODE_MAX_RETRIES = 3
DEFAULT_MAX_RUNS_PER_STEP = os.getenv("DEFAULT_MAX_RUNS_PER_STEP", 5)


class Slide(BaseModel):
    category: str
    description: str


class HighLevelPlanPPT(BaseModel):
    slides: List[Slide]


class Task(BaseModel):
    name: str
    description: str
    data_collection_task: Optional[str] = None
    code_executor_task: Optional[str] = None


class PlanningResponse(BaseModel):
    tasks: List[Task]


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


T = TypeVar("T", bound=BaseModel)


class PlanningWorkflow:
    orchestrator_model_client: OpenAIChatCompletionClient
    reflection_model_client: OpenAIChatCompletionClient
    step_triage_model_client: OpenAIChatCompletionClient
    code_model_client: OpenAIChatCompletionClient
    single_tool_use_model_client: OpenAIChatCompletionClient
    planning_model_client: OpenAIChatCompletionClient
    template_based_planning_model_client: OpenAIChatCompletionClient
    single_group_planning_model_client: OpenAIChatCompletionClient
    template_selection_model_client: OpenAIChatCompletionClient
    web_search: ClientSession
    session_manager: MCPSessionManager
    team: PlanningGroupChat

    @classmethod
    async def create(
        cls,
        config_path: str,
        team_config_path: str,
        template_work_dir: str | None = None,
        language: str = "en",
        countdown_timer: int = 60,  # time before the docker container is stopped
        external_mcp_tools: Optional[
            Dict[str, List[Any]]
        ] = None,  # Shared MCP tools from GlobalResourceManager
    ):
        self = cls()

        config = load_toml_with_env_vars(config_path)
        team_config = load_toml_with_env_vars(team_config_path)

        # TeamMember enum dynamically from team.toml
        team_members = list(team_config["team"].values())
        # TeamMembers = Enum("TeamMembers", team_members)

        class StepTriageNextSpeakerResponse(BaseModel):
            instruction: str
            answer: Literal[tuple(team_members)]  # type: ignore

        class StepTriageResponse(BaseModel):
            next_speaker: StepTriageNextSpeakerResponse

        # Initialize all model clients using ModelClientService for caching and reuse
        model_client_service = GlobalResourceManager.get_model_client_service()

        self.orchestrator_model_client = await model_client_service.get_client(
            "orchestrator_client", config_path
        )

        self.reflection_model_client = await model_client_service.get_client(
            "reflection_client", config_path, response_format=ReflectionResponse
        )

        self.step_triage_model_client = await model_client_service.get_client(
            "step_triage_client", config_path, response_format=StepTriageResponse
        )

        self.code_model_client = await model_client_service.get_client(
            "code_client", config_path
        )

        # Handle single_tool_use_client with special parallel_tool_calls logic
        self.single_tool_use_model_client = await model_client_service.get_client(
            "single_tool_use_client", config_path
        )

        self.planning_model_client = await model_client_service.get_client(
            "planning_client", config_path, response_format=PlanningResponse
        )

        # Initialize template based planning client using the same config as planning client
        self.template_based_planning_model_client = (
            await model_client_service.get_client(
                "planning_client", config_path, response_format=HighLevelPlanPPT
            )
        )

        # Initialize single group planning client using the same config as planning client
        self.single_group_planning_model_client = await model_client_service.get_client(
            "planning_client", config_path, response_format=Task
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
                await model_client_service.get_client(
                    "planning_client", config_path, response_format=TemplateSelection
                )
            )

        # MCP Tools initialization - use external tools if provided, otherwise create own sessions
        if external_mcp_tools:
            # Use shared MCP tools from GlobalResourceManager
            web_search_tools = external_mcp_tools.get("web_search", [])
            domain_specific_tools = external_mcp_tools.get("domain_specific", [])
            hirag_retrieval_tools = external_mcp_tools.get("hirag_retrieval", [])

            # Filter HiRAG tools to only include hi_search
            hirag_retrieval_tools = [
                tool for tool in hirag_retrieval_tools if tool.name == "hi_search"
            ]

            # Initialize session manager for compatibility but don't create new sessions
            self.session_manager = MCPSessionManager()
            self.web_search = None  # Not needed when using external tools
            self.hirag_retrieval = None  # Not needed when using external tools

            # Debug: Log tool availability
            logging.info(
                f"🔧 [WORKFLOW] Using external MCP tools - web_search: {len(web_search_tools)}, domain_specific: {len(domain_specific_tools)}, hirag: {len(hirag_retrieval_tools)}"
            )

        else:
            # Fallback to creating own MCP sessions (backward compatibility)
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

            hirag_server_params = StdioServerParams(
                command="mcp-hirag-tool",
                args=[],
                read_timeout_seconds=100,
                env={
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                    "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
                    "DOC2X_API_KEY": os.getenv("DOC2X_API_KEY"),
                },
            )

            self.hirag_retrieval = await self.session_manager.create_session(
                "hirag_retrieval", create_mcp_server_session(hirag_server_params)
            )
            await self.hirag_retrieval.initialize()
            hirag_retrieval_tools = await mcp_server_tools(
                hirag_server_params, session=self.hirag_retrieval
            )
            hirag_retrieval_tools = [
                tool for tool in hirag_retrieval_tools if tool.name == "hi_search"
            ]

            # Debug: Log tool availability
            logging.info(
                f"🔧 [WORKFLOW] Using own MCP sessions - web_search: {len(web_search_tools)}, domain_specific: {len(domain_specific_tools)}, hirag: {len(hirag_retrieval_tools)}"
            )

        rag_agent = AssistantAgent(
            name="retrieval_agent",
            description="a retrieval agent that provides relevant information from the internal database.",
            model_client=self.single_tool_use_model_client,
            tools=hirag_retrieval_tools,
            system_message=(
                get_rag_agent_prompt()
                if language == "en"
                else get_rag_agent_prompt_cn()
            ),
        )

        # for new feat: domain specific prompt
        domain_specific_agent = AssistantAgent(
            name="prompt_template_expert",
            model_client=self.single_tool_use_model_client,
            tools=(
                domain_specific_tools if domain_specific_tools else []
            ),  # Ensure tools is never None
            system_message=(
                get_domain_specific_agent_prompt()
                if language == "en"
                else get_domain_specific_agent_prompt_cn()
            ),
        )

        # Debug: Log domain specific agent configuration
        logging.info(
            f"🔧 [WORKFLOW] Domain specific agent created with {len(domain_specific_tools) if domain_specific_tools else 0} tools"
        )

        general_agent = AssistantAgent(
            name="general_agent",
            model_client=self.orchestrator_model_client,
            description="a general agent that provides answer for simple questions.",
            system_message=(
                get_general_agent_prompt()
                if language == "en"
                else get_general_agent_prompt_cn()
            ),
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

        # stream_code_executor=StreamLocalCommandLineCodeExecutor(work_dir=work_dir)
        stream_code_executor = StreamDockerCommandLineCodeExecutor(
            work_dir=work_dir,
            bind_dir=(
                os.getenv("HOST_PATH") + "/" + str(work_dir)
                if os.getenv("ENVIRONMENT") == "docker"
                else work_dir
            ),
        )

        code_executor = StreamCodeExecutorAgent(
            name="CodeExecutor",
            description="a code executor agent that can generate and execute Python and shell scripts to assist in code based tasks such as generating files, appending files, calculating data, etc.",
            system_message=(
                get_code_executor_prompt()
                if language == "en"
                else get_code_executor_prompt_cn()
            ),
            stream_code_executor=stream_code_executor,
            model_client=self.code_model_client,
            max_retries_on_error=DEFAULT_CODE_MAX_RETRIES,
            countdown_timer=countdown_timer,  # time before the docker container is stopped
        )

        # mapping of team member names to their agent instances
        agent_mapping: Dict[str, Any] = {
            "web_search": surfer,
            "CodeExecutor": code_executor,
            "general_agent": general_agent,
            "retrieval_agent": rag_agent,
        }

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
            language=language,
            max_runs_per_step=DEFAULT_MAX_RUNS_PER_STEP,
        )
        return self

    def set_language(self, language: str) -> None:
        if hasattr(self.team, "set_language"):
            self.team.set_language(language)

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
