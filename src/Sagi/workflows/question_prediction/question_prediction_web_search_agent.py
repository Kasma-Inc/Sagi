import asyncio
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.base._handoff import Handoff as HandoffBase
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken, Component, ComponentModel, FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    UserMessage,
)
from autogen_core.tools import BaseTool, FunctionTool, StaticWorkbench, Workbench
from pydantic import BaseModel
from typing_extensions import Self

event_logger = logging.getLogger(EVENT_LOGGER_NAME)


class QuestionPredictionWebSearchAgentConfig(BaseModel):
    name: str
    model_client: ComponentModel
    model_client_stream: bool = (False,)


class QuestionPredictionWebSearchAgent(
    BaseChatAgent, Component[QuestionPredictionWebSearchAgentConfig]
):
    component_config_schema = QuestionPredictionWebSearchAgentConfig

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        tools: List[
            BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
        ],
        model_client_stream: bool = False,
    ):
        super().__init__(
            name=name,
            description="An agent that role-playing as a human user who collect data and relevant information from the web based on user intent and chat history.",
        )
        self._model_client = model_client
        self._tools: List[BaseTool[Any, Any]] = []
        if tools is not None:
            if not model_client.model_info["function_calling"]:
                raise ValueError("The model does not support function calling.")
            for tool in tools:
                if isinstance(tool, BaseTool):
                    self._tools.append(tool)
                elif callable(tool):
                    if hasattr(tool, "__doc__") and tool.__doc__ is not None:
                        description = tool.__doc__
                    else:
                        description = ""
                    self._tools.append(FunctionTool(tool, description=description))
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")
        # Check if tool names are unique.
        tool_names = [tool.name for tool in self._tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Tool names must be unique: {tool_names}")
        self._workbench = StaticWorkbench(self._tools)
        self._model_client_stream = model_client_stream
        self._prompt_template = """You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your have asked some questions to the AI collaborator and got some responses. Based on the chat history and your intent, you may want to use tools to perform some web search yourself to enrich your knowledge and achieve your goal.

## Input Information:
You will be provided with:
- Your Intent: The goal you want to achieve.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Your Intent (Not visible to the AI)|>
{user_intent}
<|The End of Your Intent|>

<|The Start of Chat History|>
{chat_history}
<|The End of Chat History|>


## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. If the user is less knowledgeable about a task, they might perform a web search to enrich their knowledge. Perform the web search based on your current understanding and areas of confusion.
- Goal-Oriented: Keep the web search focused on your intent.

## Output Format:
1. If you think it is necessary to perform a web search, use you tools to perform a web search.
2. If you think you do not need to perform a web search, respond with 'No need to search' directly.

## Important Notes:
- Base on Previous Messages: Your should base on the context of the current chat history. Carefully read the previous messages.
- Double check if the output format is correctly.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully."""
        self._tool_call_summary_format = "{result}"

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return TextMessage, ToolCallSummaryMessage

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

        # Get user intent and conversation history
        user_intent: str = "Unknown"
        if messages[-1].source == "user_intent_recognition_agent":
            user_intent = messages[-1].content

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
                user_intent=user_intent, chat_history=chat_history
            ),
            source="user",
        )

        model_result: Optional[CreateResult] = None
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        if self._model_client_stream:
            async for chunk in self._model_client.create_stream(
                [query],
                tools=self._tools,
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
                tools=self._tools,
                cancellation_token=cancellation_token,
            )

        # If direct text response (string)
        if isinstance(model_result.content, str):
            yield Response(
                chat_message=TextMessage(
                    content=model_result.content,
                    source=self.name,
                    models_usage=model_result.usage,
                ),
                inner_messages=inner_messages,
            )
            return

        # Otherwise, we have function calls
        assert isinstance(model_result.content, list) and all(
            isinstance(item, FunctionCall) for item in model_result.content
        )

        # STEP 4A: Yield ToolCallRequestEvent
        tool_call_msg = ToolCallRequestEvent(
            content=model_result.content,
            source=self.name,
            models_usage=model_result.usage,
        )
        event_logger.debug(tool_call_msg)
        inner_messages.append(tool_call_msg)
        yield tool_call_msg

        # STEP 4B: Execute tool calls
        executed_calls_and_results = await asyncio.gather(
            *[
                self._execute_tool_call(
                    tool_call=call,
                    workbench=self._workbench,
                    handoff_tools=[],
                    agent_name=self.name,
                    cancellation_token=cancellation_token,
                )
                for call in model_result.content
            ]
        )
        exec_results = [result for _, result in executed_calls_and_results]

        # Yield ToolCallExecutionEvent
        tool_call_result_msg = ToolCallExecutionEvent(
            content=exec_results,
            source=self.name,
        )
        event_logger.debug(tool_call_result_msg)
        inner_messages.append(tool_call_result_msg)
        yield tool_call_result_msg

        # STEP 4D: Summarize tool results
        yield self._summarize_tool_use(
            executed_calls_and_results=executed_calls_and_results,
            inner_messages=inner_messages,
            handoffs={},
            tool_call_summary_format=self._tool_call_summary_format,
            agent_name=self.name,
        )

    @staticmethod
    def _summarize_tool_use(
        executed_calls_and_results: List[Tuple[FunctionCall, FunctionExecutionResult]],
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        handoffs: Dict[str, HandoffBase],
        tool_call_summary_format: str,
        agent_name: str,
    ) -> Response:
        """
        If reflect_on_tool_use=False, create a summary message of all tool calls.
        """
        # Filter out calls which were actually handoffs
        normal_tool_calls = [
            (call, result)
            for call, result in executed_calls_and_results
            if call.name not in handoffs
        ]
        tool_call_summaries: List[str] = []
        for tool_call, tool_call_result in normal_tool_calls:
            tool_call_summaries.append(
                tool_call_summary_format.format(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=tool_call_result.content,
                )
            )
        tool_call_summary = "\n".join(tool_call_summaries)
        return Response(
            chat_message=ToolCallSummaryMessage(
                content=tool_call_summary,
                source=agent_name,
            ),
            inner_messages=inner_messages,
        )

    @staticmethod
    async def _execute_tool_call(
        tool_call: FunctionCall,
        workbench: Workbench,
        handoff_tools: List[BaseTool[Any, Any]],
        agent_name: str,
        cancellation_token: CancellationToken,
    ) -> Tuple[FunctionCall, FunctionExecutionResult]:
        """Execute a single tool call and return the result."""
        # Load the arguments from the tool call.
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            return (
                tool_call,
                FunctionExecutionResult(
                    content=f"Error: {e}",
                    call_id=tool_call.id,
                    is_error=True,
                    name=tool_call.name,
                ),
            )

        # Check if the tool call is a handoff.
        # TODO: consider creating a combined workbench to handle both handoff and normal tools.
        for handoff_tool in handoff_tools:
            if tool_call.name == handoff_tool.name:
                # Run handoff tool call.
                result = await handoff_tool.run_json(arguments, cancellation_token)
                result_as_str = handoff_tool.return_value_as_string(result)
                return (
                    tool_call,
                    FunctionExecutionResult(
                        content=result_as_str,
                        call_id=tool_call.id,
                        is_error=False,
                        name=tool_call.name,
                    ),
                )

        # Handle normal tool call using workbench.
        result = await workbench.call_tool(
            name=tool_call.name,
            arguments=arguments,
            cancellation_token=cancellation_token,
        )
        return (
            tool_call,
            FunctionExecutionResult(
                content=result.to_text(),
                call_id=tool_call.id,
                is_error=result.is_error,
                name=tool_call.name,
            ),
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @classmethod
    def _from_config(cls, config: QuestionPredictionWebSearchAgentConfig) -> Self:
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            tools=[BaseTool.load_component(tool) for tool in config.tools],
            model_client_stream=config.model_client_stream,
        )

    def _to_config(self) -> QuestionPredictionWebSearchAgentConfig:
        return QuestionPredictionWebSearchAgentConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            model_client_stream=self._model_client_stream,
        )