from typing import AsyncGenerator, List, Optional, Sequence

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
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel
from typing_extensions import Self

from Sagi.utils.prompt import get_search_result_analysis_prompt


class SearchResultAnalysisAgentConfig(BaseModel):
    name: str
    model_client: ComponentModel
    model_client_stream: bool = False
    language: str = "en"


class SearchResultAnalysisAgent(
    BaseChatAgent, Component[SearchResultAnalysisAgentConfig]
):
    """
    An agent specialized in analyzing web search results and providing structured insights.
    
    This agent takes raw search results as input and produces professional analysis
    including key findings, source evaluation, actionable insights, and summary assessment.
    """
    component_config_schema = SearchResultAnalysisAgentConfig

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        language: str = "en",
        model_client_stream: bool = False,
    ):
        super().__init__(
            name=name,
            description="A professional search result analyst that extracts key insights from web search results and provides structured analysis for decision-making.",
        )
        self._model_client = model_client
        self._language = language
        self._model_client_stream = model_client_stream
        self._system_message = get_search_result_analysis_prompt(language)

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
        """
        Analyze the provided search results and generate structured analysis.
        
        Expected input: A message containing raw search results from web search tools.
        Output: Structured analysis of the search results.
        """
        
        search_results = ""
        if messages and isinstance(messages[-1].content, str):
            search_results = messages[-1].content
        else:
            search_results = "No search results provided for analysis."

        analysis_prompt = f"""Please analyze the following web search results and provide a comprehensive analysis:

## Search Results to Analyze:
{search_results}

## Analysis Request:
Please provide a structured analysis following the framework in your system instructions. Focus on extracting actionable insights and evaluating the reliability of sources.
"""

        query = UserMessage(
            content=analysis_prompt,
            source="user",
        )

        model_result: Optional[CreateResult] = None
        
        if self._model_client_stream:
            async for chunk in self._model_client.create_stream(
                [SystemMessage(content=self._system_message), query],
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
                [SystemMessage(content=self._system_message), query],
                cancellation_token=cancellation_token,
            )

        if isinstance(model_result.content, str):
            yield Response(
                chat_message=TextMessage(
                    content=model_result.content,
                    source=self.name,
                    models_usage=model_result.usage,
                ),
                inner_messages=[],
            )
        else:
            yield Response(
                chat_message=TextMessage(
                    content="Analysis could not be completed due to unexpected response format.",
                    source=self.name,
                    models_usage=model_result.usage,
                ),
                inner_messages=[],
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @classmethod
    def _from_config(cls, config: SearchResultAnalysisAgentConfig) -> Self:
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            language=config.language,
            model_client_stream=config.model_client_stream,
        )

    def _to_config(self) -> SearchResultAnalysisAgentConfig:
        return SearchResultAnalysisAgentConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            language=self._language,
            model_client_stream=self._model_client_stream,
        )

