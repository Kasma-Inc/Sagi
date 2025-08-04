import os
from contextlib import AsyncExitStack
from typing import Dict, List, Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)

from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.utils.prompt import (
    get_rag_agent_prompt,
    get_user_intent_recognition_agent_prompt,
)
from Sagi.workflows.planning.planning import ModelClientFactory
from Sagi.workflows.question_prediction.question_prediction_agent import (
    QuestionPredictionAgent,
)
from Sagi.workflows.question_prediction.question_prediction_web_search_agent import (
    QuestionPredictionWebSearchAgent,
)

DEFAULT_WEB_SEARCH_MAX_RETRIES = 3
PARTICIPANT_LIST: List[BaseChatAgent] = []
PARTICIPANT_DICT: Dict[str, int] = {}


# def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
#     last_agent_name: str = messages[-1].source
#     if last_agent_name == "question_validation_agent":
#         return "question_prediction_agent"
#     elif last_agent_name not in PARTICIPANT_DICT:
#         return PARTICIPANT_LIST[0].name
#     else:
#         return PARTICIPANT_LIST[PARTICIPANT_DICT[last_agent_name] + 1].name


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


class QuestionPredictionWorkflow:

    async def _init_question_prediction_web_search_agent(self):
        web_search_server_params = StdioServerParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        )
        web_search = await self.session_manager.create_session(
            "web_search", create_mcp_server_session(web_search_server_params)
        )
        await web_search.initialize()
        web_search_tools = await mcp_server_tools(
            web_search_server_params, session=web_search
        )

        return QuestionPredictionWebSearchAgent(
            name="question_prediction_web_search_agent",
            model_client=self.question_prediction_model_client,
            tools=web_search_tools,
        )

    async def _init_hirag_agent(self):
        hirag_server_params = StdioServerParams(
            command="mcp-hirag-tool",
            args=[],
            read_timeout_seconds=100,
            env={
                "LLM_API_KEY": os.getenv("OPENAI_API_KEY"),
                "LLM_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
            },
        )

        hirag_retrieval = await self.session_manager.create_session(
            "hirag_retrieval", create_mcp_server_session(hirag_server_params)
        )
        await hirag_retrieval.initialize()
        hirag_retrieval_tools = await mcp_server_tools(
            hirag_server_params, session=hirag_retrieval
        )

        hirag_set_language_tool = [
            tool for tool in hirag_retrieval_tools if tool.name == "hi_set_language"
        ][0]

        if hirag_set_language_tool:
            try:
                result = await hirag_set_language_tool.run_json(
                    {"language": self.language}, CancellationToken()
                )

                return f"Language successfully set to {self.language}: {result}"

            except Exception as e:
                return f"Failed to set language: {e}"

        hirag_retrieval_tools = [
            tool for tool in hirag_retrieval_tools if tool.name == "hi_search"
        ]

        return AssistantAgent(
            name="hirag_agent",
            model_client=self.question_prediction_model_client,
            model_client_stream=True,
            system_message=get_rag_agent_prompt(self.language),
            tools=hirag_retrieval_tools,
        )

    @classmethod
    async def create(
        cls,
        config_path: str,
        language: str,
        web_search: bool = False,
        hirag: bool = False,
    ):
        self = cls()
        self.language = language
        self.session_manager = MCPSessionManager()

        config = load_toml_with_env_vars(config_path)
        config_question_prediction_client = config["model_clients"][
            "question_prediction_client"
        ]
        self.question_prediction_model_client = ModelClientFactory.create_model_client(
            config_question_prediction_client
        )

        if hirag:
            hirag_agent: AssistantAgent = await self._init_hirag_agent()
            PARTICIPANT_LIST.append(hirag_agent)
            PARTICIPANT_DICT[hirag_agent.name] = len(PARTICIPANT_LIST) - 1

        # Create user_intent_recognition_agent agent
        user_intent_recognition_agent = AssistantAgent(
            name="user_intent_recognition_agent",
            model_client=self.question_prediction_model_client,
            model_client_stream=True,
            system_message=get_user_intent_recognition_agent_prompt(self.language),
        )
        PARTICIPANT_LIST.append(user_intent_recognition_agent)
        PARTICIPANT_DICT[user_intent_recognition_agent.name] = len(PARTICIPANT_LIST) - 1

        if web_search:
            question_prediction_web_search_agent: QuestionPredictionWebSearchAgent = (
                await self._init_question_prediction_web_search_agent()
            )
            PARTICIPANT_LIST.append(question_prediction_web_search_agent)
            PARTICIPANT_DICT[question_prediction_web_search_agent.name] = (
                len(PARTICIPANT_LIST) - 1
            )

        # Create question_prediction agent
        question_prediction_agent = QuestionPredictionAgent(
            name="question_prediction_agent",
            model_client=self.question_prediction_model_client,
            model_client_stream=True,
        )
        PARTICIPANT_LIST.append(question_prediction_agent)
        PARTICIPANT_DICT[question_prediction_agent.name] = len(PARTICIPANT_LIST) - 1

        self.team = RoundRobinGroupChat(
            participants=PARTICIPANT_LIST,
            termination_condition=TextMessageTermination(
                source="question_prediction_agent"
            ),
        )
        return self

    def run_workflow(self, user_input: Sequence[BaseChatMessage]):
        return self.team.run_stream(task=user_input)

    def reset(self):
        return self.team.reset()

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
