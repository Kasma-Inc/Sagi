import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, TypeVar

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from mcp import ClientSession
from pydantic import BaseModel

from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.utils.prompt import (
    get_domain_specific_agent_prompt,
    get_domain_specific_agent_prompt_cn,
    get_general_agent_prompt,
    get_general_agent_prompt_cn,
)
from Sagi.workflows.planning_html.planning_html_group_chat import PlanningHtmlGroupChat

DEFAULT_WORK_DIR = "coding_files"
DEFAULT_MCP_SERVER_PATH = "src/Sagi/mcp_server/"
DEFAULT_WEB_SEARCH_MAX_RETRIES = 3
DEFAULT_CODE_MAX_RETRIES = 3
LANGUAGE_MAP = {
    "en": "English",
    "cn": "Chinese",
}
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


class PlanningHtmlResponse(BaseModel):
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
    ) -> OpenAIChatCompletionClient:
        model_info = cls._init_model_info(client_config)
        client_kwargs = {
            "model": client_config["model"],
            "base_url": client_config["base_url"],
            "api_key": client_config["api_key"],
            "model_info": model_info,
            "max_tokens": client_config["max_tokens"],
        }

        # Handle optional parameters from client_config
        if "response_format" in client_config:
            client_kwargs["response_format"] = client_config["response_format"]

        if "parallel_tool_calls" in client_config:
            client_kwargs["parallel_tool_calls"] = client_config["parallel_tool_calls"]

        return OpenAIChatCompletionClient(**client_kwargs)


class PlanningHtmlWorkflow:
    orchestrator_model_client: OpenAIChatCompletionClient
    reflection_model_client: OpenAIChatCompletionClient
    step_triage_model_client: OpenAIChatCompletionClient
    code_model_client: OpenAIChatCompletionClient
    single_tool_use_model_client: OpenAIChatCompletionClient
    planning_model_client: OpenAIChatCompletionClient
    single_group_planning_model_client: OpenAIChatCompletionClient
    html_generator_model_client: AnthropicChatCompletionClient
    web_search: ClientSession
    session_manager: MCPSessionManager
    team: PlanningHtmlGroupChat

    @classmethod
    async def create(
        cls,
        config_path: str,
        team_config_path: str,
        language: str = "en",
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

        # Initialize all model clients using ModelClientFactory
        config_orchestrator_client = config["model_clients"]["orchestrator_client"]
        self.orchestrator_model_client = ModelClientFactory.create_model_client(
            config_orchestrator_client
        )

        config_reflection_client = config["model_clients"]["reflection_client"].copy()
        config_reflection_client["response_format"] = ReflectionResponse
        self.reflection_model_client = ModelClientFactory.create_model_client(
            config_reflection_client
        )

        config_step_triage_client = config["model_clients"]["step_triage_client"].copy()
        config_step_triage_client["response_format"] = StepTriageResponse
        self.step_triage_model_client = ModelClientFactory.create_model_client(
            config_step_triage_client
        )

        config_code_client = config["model_clients"]["code_client"]
        self.code_model_client = ModelClientFactory.create_model_client(
            config_code_client
        )

        config_single_tool_use_client = config["model_clients"][
            "single_tool_use_client"
        ].copy()
        parallel_tool_calls_setting = config_single_tool_use_client.get(
            "parallel_tool_calls"
        )
        if parallel_tool_calls_setting is not None:
            config_single_tool_use_client["parallel_tool_calls"] = (
                parallel_tool_calls_setting
            )
        self.single_tool_use_model_client = ModelClientFactory.create_model_client(
            config_single_tool_use_client
        )

        config_planning_client = config["model_clients"]["planning_client"]
        config_planning_client_with_format = config_planning_client.copy()
        config_planning_client_with_format["response_format"] = PlanningHtmlResponse
        self.planning_model_client = ModelClientFactory.create_model_client(
            config_planning_client_with_format
        )

        # Initialize single group planning client using the same config as planning client
        config_single_group_planning_client = config_planning_client.copy()
        config_single_group_planning_client["response_format"] = Task
        self.single_group_planning_model_client = (
            ModelClientFactory.create_model_client(config_single_group_planning_client)
        )

        config_html_generator_client = config["model_clients"]["html_generator_client"]
        self.html_generator_model_client = AnthropicChatCompletionClient(
            model=config_html_generator_client["model"],
            auth_token=config_html_generator_client["auth_token"],
            base_url=config_html_generator_client["base_url"],
            model_info=ModelInfo(
                vision=True,
                function_calling=True,
                json_output=False,
                family="unknown",
                structured_output=True,
            ),
            max_tokens=config_html_generator_client["max_tokens"],
        )

        self.session_manager = MCPSessionManager()

        web_search_server_params = StdioServerParams(
            command="npx",
            args=["-y", "brave-search-mcp"],
            env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        )

        self.web_search = await self.session_manager.create_session(
            "web_search", create_mcp_server_session(web_search_server_params)
        )
        await self.web_search.initialize()
        web_search_tools = await mcp_server_tools(
            web_search_server_params, session=self.web_search
        )
        web_search_tools = [
            tool
            for tool in web_search_tools
            if tool.name in ["brave_web_search", "brave_news_search"]
        ]

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

        # for new feat: domain specific prompt
        domain_specific_agent = AssistantAgent(
            name="prompt_template_expert",
            model_client=self.single_tool_use_model_client,
            tools=domain_specific_tools,
            system_message=(
                get_domain_specific_agent_prompt()
                if language == "en"
                else get_domain_specific_agent_prompt_cn()
            ),
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

        html_generator = AssistantAgent(
            name="html_generator",
            model_client=self.html_generator_model_client,
            description="a html generator agent that can generate html code.",
            system_message=f"""You are a html magazine generator agent that can generate html/css code. 
            You can use Tailwind CSS to style the html page. You should use chart.js to create the charts.
            Use {language} as the language of the content in the html page.

            MANDATORY RULES (prevents infinite stretching):
            1. Canvas elements must NEVER have width/height attributes
            2. Charts must be wrapped in divs with fixed height (e.g., height: 300px)
            3. Chart.js responsive: true requires maintainAspectRatio: false
            4. Chart containers need: position: relative; height: [specific value]; width: 100%;

            BAD: <canvas width="400" height="200"></canvas>
            GOOD: <div style="position:relative;height:300px;width:100%;"><canvas></canvas></div>

            Always test that your HTML won't cause infinite vertical stretching.
                
            """.format(
                language=LANGUAGE_MAP[language]
            ),
        )

        # mapping of team member names to their agent instances
        agent_mapping: Dict[str, Any] = {
            "web_search": surfer,
            "general_agent": general_agent,
            "html_generator": html_generator,
        }

        participants = []
        for member in team_members:
            if member in agent_mapping:
                participants.append(agent_mapping[member])

        # Pass prompt_template_agent as a separate parameter
        self.team = PlanningHtmlGroupChat(
            participants=participants,
            orchestrator_model_client=self.orchestrator_model_client,
            planning_model_client=self.planning_model_client,
            reflection_model_client=self.reflection_model_client,
            domain_specific_agent=domain_specific_agent,  # Add this parameter
            step_triage_model_client=self.step_triage_model_client,
            single_group_planning_model_client=self.single_group_planning_model_client,
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
