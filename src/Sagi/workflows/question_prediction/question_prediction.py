from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from hirag_prod.tracing import traced
from pydantic import BaseModel
from resources.model_client_wrapper import ModelClientWrapper

from Sagi.utils.prompt import (
    get_rag_agent_prompt,
    get_user_intent_recognition_agent_prompt,
)
from Sagi.workflows.question_prediction.question_prediction_agent import (
    QuestionPredictionAgent,
)
from Sagi.workflows.question_prediction.question_prediction_web_search_agent import (
    QuestionPredictionWebSearchAgent,
)


class QuestionsResponse(BaseModel):
    questions: List[str]


T = TypeVar("T", bound=BaseModel)


class QuestionPredictionWorkflow:
    participant_list: List[BaseChatAgent] = []

    @classmethod
    async def create(
        cls,
        model_name: str,
        mcp_tools: Dict[
            str,
            List[
                BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
            ],
        ],
        language: str,
        web_search: bool = False,
        hirag: bool = False,
    ):
        self = cls()
        self.model_client_dict = {
            "General": ModelClientWrapper(model_name, None),
            "QuestionsResponse": ModelClientWrapper(model_name, QuestionsResponse),
        }
        self.language = language
        self.participant_list = []

        user_intent_recognition_agent = AssistantAgent(
            name="user_intent_recognition_agent",
            model_client=self.model_client_dict["General"],
            model_client_stream=True,
            system_message=get_user_intent_recognition_agent_prompt(self.language),
        )
        self.participant_list.append(user_intent_recognition_agent)

        if web_search:
            question_prediction_web_search_agent: QuestionPredictionWebSearchAgent = (
                QuestionPredictionWebSearchAgent(
                    name="question_prediction_web_search_agent",
                    model_client=self.model_client_dict["General"],
                    tools=mcp_tools["web_search"],
                )
            )
            self.participant_list.append(question_prediction_web_search_agent)
        if hirag:
            hirag_agent: AssistantAgent = AssistantAgent(
                name="hirag_agent",
                model_client=self.model_client_dict["General"],
                model_client_stream=True,
                system_message=get_rag_agent_prompt(self.language),
                tools=mcp_tools["hirag_retrieval"],
            )
            self.participant_list.append(hirag_agent)

        # Create question_prediction agent
        question_prediction_agent = QuestionPredictionAgent(
            name="question_prediction_agent",
            model_client=self.model_client_dict["QuestionsResponse"],
            model_client_stream=True,
            language=self.language,
        )
        self.participant_list.append(question_prediction_agent)

        self.team = RoundRobinGroupChat(
            participants=self.participant_list,
            termination_condition=TextMessageTermination(
                source="question_prediction_agent"
            ),
        )
        return self

    @traced()
    def run_workflow(
        self,
        user_input: Sequence[BaseChatMessage],
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ):
        token = kwargs.pop("cancellation_token", None) or cancellation_token
        extra_kwargs = {}
        if token is not None:
            extra_kwargs["cancellation_token"] = token
        return self.team.run_stream(task=user_input, **extra_kwargs)

    def set_model(self, model_name: str) -> None:
        for model_client in self.model_client_dict.values():
            model_client.set_model_client(model_name=model_name)

    def reset(self):
        return self.team.reset()

    async def cleanup(self):
        """close activated MCP servers"""
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()
