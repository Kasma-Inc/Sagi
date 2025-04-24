import os
from pathlib import Path
from typing import List, Literal, Optional

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from pydantic import BaseModel

from utils.load_config import load_toml_with_env_vars

from .planning_group_chat import PlanningGroupChat


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
                max_tokens=config_reflection_client["max_tokens"],
            )
        else:
            self.code_model_client = OpenAIChatCompletionClient(
                model=config_code_client["model"],
                base_url=config_code_client["base_url"],
                api_key=config_code_client["api_key"],
                max_tokens=config_reflection_client["max_tokens"],
            )

    @classmethod
    async def create(
        cls,
        config_path: str,
    ):
        self = cls(config_path)
        web_search_server_params = StdioServerParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        )
        web_search_tools = await mcp_server_tools(web_search_server_params)

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

        # for new feat: domain specific prompt
        domain_specific_agent = AssistantAgent(
            name="prompt_template_expert",
            model_client=self.model_client,
            tools=domain_specific_tools,  # type: ignore
            system_message="You are a prompt expert that provides structured templates for different domains.",
        )

        surfer = AssistantAgent(
            name="web_search",
            model_client=self.model_client,
            reflect_on_tool_use=True,  # enable llm summary for contents web search returns
            tools=web_search_tools,  # type: ignore
        )
        work_dir = Path(
            "coding_files"
        )  # the output directory for code generation execution
        code_executor = CodeExecutorAgent(
            name="CodeExecutor",
            code_executor=LocalCommandLineCodeExecutor(work_dir=work_dir),
            model_client=self.code_model_client,
            max_retries_on_error=3,
        )

        # Pass prompt_template_agent as a separate parameter
        self.team = PlanningGroupChat(
            [surfer, code_executor],
            model_client=self.model_client,
            planning_model_client=self.planning_model_client,
            reflection_model_client=self.reflection_model_client,
            domain_specific_agent=domain_specific_agent,  # Add this parameter
        )
        return self

    def run_workflow(self, user_input: str):
        return self.team.run_stream(task=user_input)
