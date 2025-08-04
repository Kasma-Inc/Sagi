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
    ):
        super().__init__(
            name=name,
            description="An agent that predicts the next user question based on user intent and chat history.",
        )
        self._model_client = model_client
        self._model_client_stream = model_client_stream
        self._prompt_template = """You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your goal is to generate realistic, natural responses that a user might give in this scenario.

## Input Information:
You will be provided with:
- Your Intent: The goal you want to achieve.
- Web search results: The web search results you obtained.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Your Intent (Not visible to the AI)|>
{user_intent}
<|The End of Your Intent|>

<|The Start of Web Search Results (Not visible to the AI)|>
{web_search_results}
<|The End of Web Search Results|>

<|The Start of Chat History|>
{chat_history}
<|The End of Chat History|>


## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
- Minimize Effort: IMPORTANT! As a user, avoid being too detailed in your responses. Provide vague or incomplete demands in the early stages of the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. Ask questions that demonstrate your current understanding and areas of confusion.
- Mention Personal Preferences: Include preferences or constraints that might influence your requests or responses. For example, "I prefer short answers," "I need this done quickly," or "I like detailed comments in code."
- Goal-Oriented: Keep the chat focused on your intent. Avoid small talk or digressions. Redirect the chat back to the main objective if it starts to stray.

## Output Format:
You should output a JSON object with three entries:
- "current_answer" (str): Briefly summerize the AI's current solution to the task.
- "thought" (str): Output your thought process as a user deciding what to say next. Consider:
1. What specific part of the problem or solution are you struggling with?
2. Has the AI asked you to perform a task or answer a question? If so, how should you approach it?
3. Are you noticing any patterns or potential misunderstandings that need clarification?
4. If you're stuck, how can you phrase your question to get the most helpful response while demonstrating your current understanding?
- "response" (list of str): Based on your thought process, respond to the AI as the user you are role-playing. Please provide 3 possible responses and output them as a JSON list. Stop immediately when the 3 responses are completed.

## Important Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Conversation Flow: If "Current Chat History" is empty, start the conversation from scratch with an initial request. Otherwise, continue based on the existing conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries or any provided information directly in your responses.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully."""

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
            content=self._prompt_template.format(
                user_intent=user_intent,
                web_search_results=web_search_results,
                chat_history=chat_history,
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
