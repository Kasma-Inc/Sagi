import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Mapping

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    BaseAgentEvent,
    BaseChatMessage,
    ChatMessage,
    HandoffMessage,
    MessageFactory,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams._group_chat._base_group_chat_manager import (
    BaseGroupChatManager,
)
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatRequestPublish,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_agentchat.utils import content_to_str, remove_images
from autogen_core import CancellationToken, DefaultTopicId, MessageContext, event, rpc
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, Field

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
)
from Sagi.utils.prompt import (
    get_appended_plan_prompt,
    get_final_answer_prompt,
    get_reflection_step_completion_prompt,
    get_step_triage_prompt,
)
from Sagi.workflows.analyzing.analyze_manager import AnalyzeManager

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


class UserInputMessage(BaseModel):
    messages: List[ChatMessage]


class AnalyzingOrchestratorState(BaseModel):
    type: str = Field(default="AnalyzingOrchestratorState")
    analyze_manager_state: Dict = Field(default_factory=dict)


class AnalyzingOrchestrator(BaseGroupChatManager):
    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        group_chat_manager_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        max_turns: int | None,
        message_factory: MessageFactory,
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        emit_team_events: bool,
        analyzing_model_client: ChatCompletionClient,
        pg_model_client: ChatCompletionClient,
        user_proxy: Any | None = None,
    ):
        super().__init__(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            max_turns=max_turns,
            message_factory=message_factory,
            emit_team_events=emit_team_events,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
        )
        self._analyzing_model_client = analyzing_model_client
        self._pg_model_client = pg_model_client
        self._user_proxy = user_proxy
        self._group_chat_manager_topic_type = group_chat_manager_topic_type
        self._prompt_templates = {}  # to store prompts
        self._analyze_manager = AnalyzeManager()

        # Produce a team description. Each agent sould appear on a single line.
        self._team_description = ""
        for topic_type, description in zip(
            self._participant_names, self._participant_descriptions, strict=True
        ):
            self._team_description += (
                re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
            )
        self._team_description = self._team_description.strip()
        # TODO: register the new message type in a systematic way
        self._message_factory.register(CodeFileMessage)

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._analyze_manager.reset()
        if self._termination_condition is not None:
            await self._termination_condition.reset()

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore
        """Handle the start of a task."""
        # Check if the conversation has already terminated.
        if (
            self._termination_condition is not None
            and self._termination_condition.terminated
        ):
            early_stop_message = StopMessage(
                content="The group chat has already terminated.", source=self._name
            )
            # Signal termination.
            await self._signal_termination(early_stop_message)
            # Stop the group chat.
            return
        assert message is not None and message.messages is not None

        # Validate the group state given all the messages.
        await self.validate_group_state(message.messages)

        user_input_message = UserInputMessage(messages=message.messages)
        # Publish the message to orchestrator, which will trigger the router
        await self._runtime.publish_message(
            message=user_input_message,
            topic_id=DefaultTopicId(type=self._group_chat_manager_topic_type),
        )

    @event
    async def router(self, message: UserInputMessage, ctx: MessageContext) -> None:
        if not self._analyze_manager.is_plan_awaiting_confirmation():
            if self._analyze_manager.get_plan_count() > 1:
                assert 0
                # await self._get_appended_plan(message, ctx)
            else:
                await self._get_initial_plan(message, ctx)

    async def _stream_from_model(
        self,
        model_client,
        messages,
        cancellation_token,
    ):
        """Stream response from a model and collect the full content.
        Args:
            model_client: The model client to use for streaming
            messages: The messages to send to the model
            cancellation_token: Cancellation token for the request
        Returns:
            The complete streamed content
        """
        content = ""
        stream = model_client.create_stream(
            messages=self._get_compatible_context(
                model_client=model_client, messages=messages
            ),
            cancellation_token=cancellation_token,
        )

        async for response in stream:
            if isinstance(response, str):
                chunk_event = ModelClientStreamingChunkEvent(
                    content=response, source=self._name
                )
                # Add to message queue
                content += response
            else:
                content = response.content

        assert isinstance(content, str)
        return content

    @event
    async def handle_agent_response(self, message: GroupChatAgentResponse, ctx: MessageContext) -> None:  # type: ignore
        delta: List[BaseAgentEvent | BaseChatMessage] = []
        if message.agent_response.inner_messages is not None:
            for inner_message in message.agent_response.inner_messages:
                delta.append(inner_message)
                inner_message.metadata["step_id"] = (
                    self._analyze_manager.get_current_step()[0] #!!
                )
                await self._output_message_queue.put(inner_message)

        # # For web app
        # plan_state = {
        #     k: v.state for k, v in self._analyze_manager._current_plan.steps.items()
        # }
        # plan_state_message = TextMessage(
        #     content=json.dumps(plan_state, indent=4),
        #     source="PlanState",
        #     metadata={"step_id": self._analyze_manager.get_current_step()[0]},
        # )
        # await self._output_message_queue.put(plan_state_message)

        self._analyze_manager.add_message_to_step(
            step_id=self._analyze_manager.get_current_step()[0],
            message=message.agent_response.chat_message,
        )
        delta.append(message.agent_response.chat_message)

        if self._termination_condition is not None:
            stop_message = await self._termination_condition(delta)
            if stop_message is not None:
                # Reset the termination conditions.
                await self._termination_condition.reset()
                # Signal termination.
                await self._signal_termination(stop_message)
                return
        await self._orchestrate_step(ctx.cancellation_token)

    async def _compose_task(self, message: UserInputMessage) -> str:
        """Combine all message contents for the task."""
        return " ".join([content_to_str(msg.content) for msg in message.messages])

    async def _get_initial_plan(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        """Create the initial plan based on user input."""

        # Compose the task from message contents
        task = await self._compose_task(message)
        logging.info(f"Task: {task}")

        # Get prompt templates for the given task
        self._prompt_templates = await self._get_prompt_templates(
            task, ctx.cancellation_token
        )
        logging.info(self._prompt_templates)

        # Create the initial plan
        plan_prompt = self._prompt_templates["plan_prompt"].format(
            team=self._team_description,
            task=task,
        )
        model_response = {
            "steps": [
                {
                    "name": "Query Database",
                    "description": f"Here is the task: {{task}}. If the task involves obtaining data from the database 'transaction_data', please generate a valid SQL SELECT statement to retrieve the data from the table. Use the pg_query tool via the MCP server to run the query and return the results. If the task does not require querying the transaction_data, please report a warning. No analysis is required at this stage."
                },
                {
                    "name": "Analyze Retrieved Entries",
                    "description": "Analyze the table result from the previous step, which contains transaction data. Based on the content of the rows, provide a brief summary or insight. Additionally, offer regulatory recommendations or suggestions based on the patterns, anomalies, or trends observed in the data.",
                },

            ]
        }
        model_response_string = json.dumps(model_response)

        self._analyze_manager.new_plan(task=task, model_response=model_response_string)
        await self._orchestrate_step(cancellation_token=ctx.cancellation_token)

    async def _get_appended_plan(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        """Create a new plan for multi-round conversations based on context history."""

        task = await self._compose_task(message)
        formatted_history = self._analyze_manager.get_analyze_history_str()

        analyze_prompt = get_appended_plan_prompt(
            current_task=task,
            contexts_history=formatted_history,
            team_composition=self._team_description,
        )
        self._analyze_manager.new_plan(task=task, model_response=plan_response) #!!

    
    def messages_to_context(
        self, messages: List[BaseAgentEvent | BaseChatMessage]
    ) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in messages:
            if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                # Ignore tool call messages.
                continue
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == self._name:
                assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                context.append(AssistantMessage(content=m.content, source=m.source))
            else:
                assert isinstance(
                    m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage)
                )
                context.append(UserMessage(content=m.content, source=m.source))
        return context

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        current_step = self._analyze_manager.get_current_step()
        if current_step is None:
            return
        else:
            current_step_id, current_step_content = current_step
        

        context = self.messages_to_context(
            self._analyze_manager.get_messages_of_current_step()
        )
        filtered_context = [
            msg
            for msg in context
            if isinstance(msg, (UserMessage, FunctionExecutionResultMessage))
        ]

        logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        logging.info(current_step)
        logging.info(context)
        logging.info(filtered_context)
        logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        is_complete, reason = await self._check_step_completion(
            current_step_content, filtered_context, cancellation_token
        )

        if is_complete:
            self._analyze_manager.set_step_state(current_step_id, "completed")
            step_completion_message = TextMessage(
                content=json.dumps(
                    {
                        "step": current_step_content,
                        "reason": f"completed: {reason}",
                    },
                    indent=4,
                ),
                source="StepCompletionNotifier",
            )
            self._analyze_manager.add_reflection_to_step(current_step_id, reason)
            await self._output_message_queue.put(step_completion_message)

            # Find the next pending step after completing the current one
            current_step = self._analyze_manager.get_current_step()

            if current_step is None:
                return
            else:
                current_step_id, current_step_content = current_step

        # Initialize or increment the counter
        if self._analyze_manager.get_step_progress_counter(current_step_id) == 0:
            self._analyze_manager.set_step_state(current_step_id, "in_progress")
            step_start_json = {
                "stepId": current_step_id,
                "content": current_step_content,
                "totalSteps": self._analyze_manager.get_total_steps_current_plan(),
            }
            step_start_message = TextMessage(
                content=json.dumps(step_start_json, indent=4),
                source="NewStepNotifier",
            )
            await self._output_message_queue.put(step_start_message)

        # Check if the plan has been in progress for too long
        # If the plan has been in progress for more than 5 iterations, mark it as completed
        self._analyze_manager.increment_step_counter(current_step_id)
        if self._analyze_manager.get_step_progress_counter(current_step_id) >= 5:
            await self._log_message(
                f"Plan '{current_step}' has been in progress for 5 iterations. Marking as completed."
            )
            self._analyze_manager.set_step_state(current_step_id, "failed")
            # Log the forced completion
            step_failed_message = TextMessage(
                content=json.dumps(
                    {
                        "step": current_step_content,
                        "reason": f"failed: {reason}",
                    },
                    indent=4,
                ),
                source="StepCompletionNotifier",
            )
            self._analyze_manager.add_reflection_to_step(current_step_id, reason)
            await self._output_message_queue.put(step_failed_message)

            # Find the next pending step
            current_step = self._analyze_manager.get_current_step()

            # If there's no next pending step, we're done
            if current_step is None:
                return
            else:
                current_step_id, current_step_content = current_step

            self._analyze_manager.set_step_state(current_step_id, "in_progress")

        step_triage_prompt = get_step_triage_prompt( #!!
            task=self._analyze_manager.get_task(),
            current_plan=current_step_content,
            names=self._participant_names,
            team_description=self._team_description,
        )
        
        context.append(UserMessage(content=step_triage_prompt, source=self._name))



        step_triage_response = await self._llm_create(
            self._analyzing_model_client, context, cancellation_token
        )
        logging.info(step_triage_response)
        logging.info("aaaaaaaaaaaaaaaaaaaaaaa")
        step_triage = json.loads(step_triage_response)

        # Broadcast the next step
        instruction_or_question = step_triage["instruction_or_question"]["answer"]
        message = TextMessage(
            content=instruction_or_question,
            source=self._name,
        )
        self._analyze_manager.add_message_to_step(
            step_id=current_step_id,
            message=message,
        )

        next_speaker = step_triage["next_speaker"]["answer"]
        logging.info(f"Next Speaker: {next_speaker}")

        step_running_message = TextMessage(
            content=json.dumps(
                {
                    "tool": next_speaker,
                    "instruction": instruction_or_question,
                    "stepId": current_step_id,
                },
                indent=4,
            ),
            source="ToolCaller",
        )
        # Log it to the output queue.
        await self._output_message_queue.put(step_running_message)

        # Broadcast it
        await self.publish_message(  # Broadcast
            GroupChatAgentResponse(agent_response=Response(chat_message=message)),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Check if the next speaker is valid
        # TODO: handle the case where the next speaker is not in the team
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the step triage, participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _check_step_completion(
        self,
        current_plan_content: str,
        filtered_context: List[LLMMessage],
        cancellation_token: CancellationToken,
    ) -> bool:
        """Check if the current plan step has been completed based on conversation context."""
        # Format the context for better LLM understanding
        formatted_context = (
            "\n".join(
                [
                    f"Message {i+1} from {msg.source}:\n{msg.content}"
                    for i, msg in enumerate(filtered_context)
                ]
            )
            if filtered_context
            else "No relevant messages found in the conversation context."
        )

        # # Create a reflection prompt
        # reflection_prompt = get_reflection_step_completion_prompt(
        #     current_plan=current_plan_content, conversation_context=formatted_context
        # )

        # reflection_context = [UserMessage(content=reflection_prompt, source=self._name)]

        # reflection_response = await self._llm_create(
        #     self._reflection_model_client, reflection_context, cancellation_token
        # )
        # reflection = json.loads(reflection_response)

        # reason = reflection.get("reason", "No reason provided")
        # return reflection.get("is_complete", "false") == "true", reason
        return "true", "No reason provided"

    async def _get_prompt_templates(
        self,
        task_description: str,
        cancellation_token: CancellationToken,
    ) -> dict:

        # # Create a message asking for appropriate templates
        # message = TextMessage(
        #     content=f"Based on this task, please determine the most appropriate prompt template type and provide it:\n\n{task_description}",
        #     source=self._name,
        # )

        # # Get response from the domain specific agent
        # response = await self._domain_specific_agent.on_messages(
        #     [message], cancellation_token=cancellation_token
        # )
        # tool_response = json.loads(response.chat_message.content)[0].get("text")
        # prompt_dict = json.loads(tool_response)

        prompt_dict = {
            "plan_prompt": "You are a database expert. Use the available tools to query a PostgreSQL database and return concise, correct results. {task}",
            "analyze_prompt": "The data has now been queried. Please generate risk monitoring suggestions based on the indicators. ",
        }
        return prompt_dict

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = AnalyzingOrchestratorState.model_validate(state)
        self._analyze_manager = AnalyzeManager.load(orchestrator_state.analyze_manager_state)

    async def save_state(self) -> Mapping[str, Any]:
        state = AnalyzingOrchestratorState(
            analyze_manager_state=self._analyze_manager.dump(),
        )
        return state.model_dump()

    async def _log_message(self, log_message: str) -> None:
        trace_logger.debug(log_message)

    async def validate_group_state(
        self, messages: List[BaseChatMessage] | None
    ) -> None:
        pass

    async def select_speaker(
        self, thread: List[BaseAgentEvent | BaseChatMessage]
    ) -> str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return ""

    def _get_compatible_context(
        self, model_client: ChatCompletionClient, messages: List[LLMMessage]
    ) -> List[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)

    async def _llm_create(self, client, conversation: list, cancellation_token) -> str:
        """Send conversation to LLM client and return the string content."""
        response = await self._stream_from_model(
            model_client=client,
            messages=conversation,
            cancellation_token=cancellation_token,
        )
        return response
