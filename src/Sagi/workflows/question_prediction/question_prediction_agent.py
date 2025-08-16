from typing import AsyncGenerator, Optional, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    TextMessage,
)
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    UserMessage,
)
from pydantic import BaseModel
from typing_extensions import Self

from Sagi.utils.prompt import get_question_prediction_agent_prompt


class QuestionPredictionAgentConfig(BaseModel):
    name: str
    model_client: ComponentModel
    model_client_stream: bool = (False,)


class QuestionPredictionAgent(BaseChatAgent, Component[QuestionPredictionAgentConfig]):
    component_config_schema = QuestionPredictionAgentConfig

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        model_client_stream: bool = False,
        language: str = "en",
    ):
        super().__init__(
            name=name,
            description="An agent that predicts the next user question based on user intent and chat history.",
        )
        self._model_client = model_client
        self._model_client_stream = model_client_stream
        self._language = language

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")

        return final_response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:

        # Get user intent, web search results and conversation history
        user_intent: str = "Unknown"
        if messages[-2].source == "user_intent_recognition_agent":
            user_intent = messages[-2].content

        web_search_results: str = "Not conducted"
        if messages[-1].source == "question_prediction_web_search_agent":
            web_search_results = messages[-1].content

        chat_history = "\n".join(
            [
                (msg.source if hasattr(msg, "source") else "system")
                + ": "
                + (msg.content if isinstance(msg.content, str) else "")
                + "\n"
                for msg in messages[:-1]
            ]
        )

        query: UserMessage = UserMessage(
            content=get_question_prediction_agent_prompt(
                user_intent=user_intent,
                web_search_results=web_search_results,
                chat_history=chat_history,
                language=self._language,
            ),
            source="user",
        )

        model_result: Optional[CreateResult] = None
        if self._model_client_stream:
            async for chunk in self._model_client.create_stream(
                [query],
                cancellation_token=cancellation_token,
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source=self.name
                    )
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
        else:
            model_result = await self._model_client.create(
                [query],
                cancellation_token=cancellation_token,
            )

        yield Response(
            chat_message=TextMessage(
                content=model_result.content,
                source=self.name,
                models_usage=model_result.usage,
            ),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @classmethod
    def _from_config(cls, config: QuestionPredictionAgentConfig) -> Self:
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            model_client_stream=config.model_client_stream,
        )

    def _to_config(self) -> QuestionPredictionAgentConfig:
        return QuestionPredictionAgentConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            model_client_stream=self._model_client_stream,
        )
