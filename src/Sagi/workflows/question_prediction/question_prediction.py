from typing import Dict, List, Sequence, Any, Callable, Awaitable

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools._base import BaseTool
from pydantic import BaseModel

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


class QuestionsResponse(BaseModel):
    questions: List[str]


class QuestionPredictionWorkflow:
    participant_list: List[BaseChatAgent] = []

    @classmethod
    async def create(
        cls,
        config_path: str,
        mcp_tools: Dict[str, List[
            BaseTool[Any, Any] | Callable[..., Any] | Callable[
                ..., Awaitable[Any]]]],
        language: str,
        web_search: bool = False,
        hirag: bool = False,
    ):
        self = cls()
        self.participant_list = []
        self.participant_dict = {}
        self.language = language

        config = load_toml_with_env_vars(config_path)
        config_question_prediction_client = config["model_clients"][
            "question_prediction_client"
        ]
        self.model_client = ModelClientFactory.create_model_client(
            config_question_prediction_client,
        )

        self.question_prediction_model_client = ModelClientFactory.create_model_client(
            config_question_prediction_client,
            response_format=QuestionsResponse,
        )

        user_intent_recognition_agent = AssistantAgent(
            name="user_intent_recognition_agent",
            model_client=self.model_client,
            model_client_stream=True,
            system_message=get_user_intent_recognition_agent_prompt(self.language),
        )
        self.participant_list.append(user_intent_recognition_agent)

        if web_search:
            question_prediction_web_search_agent: QuestionPredictionWebSearchAgent = QuestionPredictionWebSearchAgent(
                name="question_prediction_web_search_agent",
                model_client=self.model_client,
                tools=mcp_tools["web_search"],
            )
            self.participant_list.append(question_prediction_web_search_agent)
        if hirag:
            hirag_agent: AssistantAgent = AssistantAgent(
                name="hirag_agent",
                model_client=self.model_client,
                model_client_stream=True,
                system_message=get_rag_agent_prompt(self.language),
                tools=mcp_tools["hirag_retrieval"],
            )
            self.participant_list.append(hirag_agent)

        # Create question_prediction agent
        question_prediction_agent = QuestionPredictionAgent(
            name="question_prediction_agent",
            model_client=self.question_prediction_model_client,
            model_client_stream=True,
        )
        self.participant_list.append(question_prediction_agent)

        self.team = RoundRobinGroupChat(
            participants=self.participant_list,
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
