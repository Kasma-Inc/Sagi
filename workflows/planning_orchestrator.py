import asyncio
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    MessageFactory,
    StopMessage,
    TextMessage,
)
from autogen_agentchat.state import MagenticOneOrchestratorState
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import (
    MagenticOneOrchestrator,
)
from autogen_agentchat.utils import content_to_str
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    event,
    rpc,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, Field

from utils.prompt import (
    appended_plan_prompt,
    orchestrator_progress_ledger_prompt,
    reflection_step_completion_prompt,
)


@dataclass
class StepContext:
    """Stores the context of a completed step for multi-round conversations."""

    step: str
    result: List[LLMMessage]
    reason: str


class UserInputMessage(BaseModel):
    messages: List[ChatMessage]


class PLAN_STATE(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

    def __json__(self) -> str:
        return self.value

    def dict(self) -> Dict[str, Any]:
        return {"value": self.value}


class PlanManager:
    """Manages plan creation, tracking, and execution for planning workflows."""

    def __init__(self):
        self._plan = None
        self._plan_state = None
        self._step_progress_counter = {}

    def is_tbd(self) -> bool:
        """Check if the plan is still TBD."""
        return self._plan is None

    def set_plan(self, model_response: str) -> None:
        """Set the plan."""
        self._plan = json.loads(model_response)["steps"]

    def set_plan_state(self) -> None:
        self._plan_state = OrderedDict()

        # Handle the new Step structure with data_collection_task and code_executor_task
        for step in self._plan:
            step_name = step["name"]
            step_description = step["description"]
            # If both tasks are present, create two separate steps
            if step.get("data_collection_task") and step.get("code_executor_task"):
                self._plan_state[
                    f"data collection for {step_name}: {step['data_collection_task']}"
                ] = PLAN_STATE.PENDING
                self._plan_state[
                    f"code executor task for {step_name}: {step['code_executor_task']}"
                ] = PLAN_STATE.PENDING
            # If only data_collection_task is present
            elif step.get("data_collection_task"):
                self._plan_state[
                    f"data collection for {step_name}: {step['data_collection_task']}"
                ] = PLAN_STATE.PENDING
            # If only code_executor_task is present
            elif step.get("code_executor_task"):
                self._plan_state[
                    f"code executor task for {step_name}: {step['code_executor_task']}"
                ] = PLAN_STATE.PENDING
            else:
                self._plan_state[f"{step_name}: {step_description}"] = (
                    PLAN_STATE.PENDING
                )

    def set_status(self, step: str, status: PLAN_STATE) -> None:
        """Set the plan state to the specified status."""
        self._plan_state[step] = status

    def get_plan(self) -> List:
        """Get the plan."""
        return self._plan

    def get_plan_str(self) -> str:
        """Get the plan as a string."""
        return json.dumps(self._plan, indent=4)

    def get_plan_state(self) -> OrderedDict:
        """Get the plan state."""
        return self._plan_state

    def get_current_step(self) -> str | None:
        """Get the current step."""
        current_step = next(
            (
                step
                for step, state in self._plan_state.items()
                if state in [PLAN_STATE.PENDING, PLAN_STATE.IN_PROGRESS]
            ),
            None,
        )
        return current_step

    def get_step_progress_counter(self, step: str) -> int:
        """Get the progress counter for a specific step."""
        return self._step_progress_counter.get(step, 0)

    def increment_step_counter(self, step: str) -> None:
        """Increment the counter for a step."""
        if step not in self._step_progress_counter:
            self._step_progress_counter[step] = 0
        self._step_progress_counter[step] += 1

    def clear(self) -> None:
        """Clear the plan."""
        self._plan = None
        self._plan_state = None
        self._step_progress_counter = {}


class PlanningOrchestratorState(MagenticOneOrchestratorState):
    type: str = Field(default="PlanningOrchestratorState")
    contexts_history: List[StepContext] = Field(default=[])


class PlanningOrchestrator(MagenticOneOrchestrator):
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
        model_client: ChatCompletionClient,
        max_stalls: int,
        final_answer_prompt: str,
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        emit_team_events: bool,
        planning_model_client: ChatCompletionClient,
        reflection_model_client: ChatCompletionClient,
        # TODO: Use model_client_dict to assign separate model clients for different agents.
        user_proxy: Any | None = None,
        domain_specific_agent: (
            Any | None
        ) = None,  # for new feat: domain specific prompt
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
            model_client=model_client,
            max_stalls=max_stalls,
            final_answer_prompt=final_answer_prompt,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
        )
        self._planning_model_client = planning_model_client
        self._reflection_model_client = reflection_model_client
        self._user_proxy = user_proxy
        self._domain_specific_agent = (
            domain_specific_agent  # for new feat: domain specific prompt
        )
        self._group_chat_manager_topic_type = group_chat_manager_topic_type
        self._prompt_templates = {}  # to store domain specific prompts
        self._multi_round = False  # Initialize multi_round flag
        self._contexts_history = (
            []
        )  # Store contexts_history for multi-round conversations
        self._plan_manager = PlanManager()  # Initialize plan manager

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
        if self._plan_manager.is_tbd():
            if self._multi_round:
                await self._get_appended_plan(message, ctx)
            else:
                await self._get_initial_plan(message, ctx)
        else:
            await self._human_in_the_loop(message, ctx)

    async def _get_initial_plan(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        # Create the initial task ledger
        #################################
        # Combine all message contents for task
        self._task = " ".join([content_to_str(msg.content) for msg in message.messages])

        # Get appropriate prompt templates based on the task description
        self._prompt_templates = await self._get_prompt_templates(
            self._task, ctx.cancellation_token
        )

        planning_conversation: List[LLMMessage] = []

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append(
            UserMessage(
                content=self._prompt_templates["facts_prompt"].format(task=self._task),
                source=self._name,
            )
        )
        response = await self._model_client.create(
            self._get_compatible_context(planning_conversation),
            cancellation_token=ctx.cancellation_token,
        )

        planning_conversation.clear()
        assert isinstance(response.content, str)
        self._facts = response.content
        planning_conversation.append(
            AssistantMessage(content=self._facts, source=self._name)
        )

        # 2. CREATE A PLAN
        ## plan based on available information
        planning_conversation.append(
            UserMessage(
                content=self._prompt_templates["plan_prompt"].format(
                    team=self._team_description,
                    task=self._task,
                ),
                source=self._name,
            )
        )

        planning_conversation.append(
            SystemMessage(
                content="CODE GENERATION and CODE EXECUTION MUST be combined into ONE STEP for each sub-task, DO NOT separated CODE GENERATION and CODE EXECUTION into two steps.",
            )
        )
        response = await self._planning_model_client.create(
            self._get_compatible_context(planning_conversation),
            cancellation_token=ctx.cancellation_token,
        )

        assert isinstance(response.content, str)
        self._plan_manager.set_plan(response.content)

        # Request human's feedback
        await self._get_user_feedback_on_plan(self._user_proxy, ctx.cancellation_token)

    async def _get_appended_plan(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        """Create a new plan for multi-round conversations based on context history."""
        # Combine all message contents for task
        self._task = " ".join([content_to_str(msg.content) for msg in message.messages])
        # Include context history
        formatted_contexts_history = self._format_contexts_history()
        _appended_plan_prompt = appended_plan_prompt(
            current_task=self._task,
            contexts_history=formatted_contexts_history,
            team_composition=self._team_description,
        )

        planning_conversation: List[LLMMessage] = []

        # CREATE A PLAN that builds on context history
        planning_conversation.append(
            UserMessage(
                content=_appended_plan_prompt,
                source=self._name,
            )
        )

        planning_conversation.append(
            SystemMessage(
                content="CODE GENERATION and CODE EXECUTION MUST be combined into ONE STEP for each sub-task. DO NOT separate CODE GENERATION and CODE EXECUTION into two steps. ",
            )
        )

        response = await self._planning_model_client.create(
            self._get_compatible_context(planning_conversation),
            cancellation_token=ctx.cancellation_token,
        )

        assert isinstance(response.content, str)
        self._plan_manager.set_plan(response.content)

        # Request human's feedback
        await self._get_user_feedback_on_plan(self._user_proxy, ctx.cancellation_token)

    def _format_contexts_history(self) -> str:
        """Format the context history for inclusion in prompts."""
        if not self._contexts_history:
            return "No context history available."

        formatted_contexts = []
        for i, context in enumerate(self._contexts_history):
            formatted_contexts.append(f"Round {i+1}:")
            formatted_contexts.append(f"Step: {context.step}")
            formatted_contexts.append(f"Result: {context.result}")
            formatted_contexts.append(f"Complete Reason: {context.reason}")
            formatted_contexts.append("")

        return "\n".join(formatted_contexts)

    async def _human_in_the_loop(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        feedback = message.messages[-1].content
        if feedback.strip().lower() in {"ok", "yes", "y"}:
            self._plan_manager.set_plan_state()

            # Log the plan to the output topic.
            planning_message = TextMessage(
                content=json.dumps(self._plan_manager.get_plan_state(), indent=4),
                source="Planner",
            )
            # Log the message to the output topic.
            await self.publish_message(
                GroupChatMessage(message=planning_message),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            # Log the message to the output queue.
            await self._output_message_queue.put(planning_message)
            await self._orchestrate_step(cancellation_token=ctx.cancellation_token)
            # await self._reenter_outer_loop(ctx.cancellation_token)
        else:
            await self._update_plan_with_feedback(feedback, ctx.cancellation_token)
            await self._get_user_feedback_on_plan(
                self._user_proxy, ctx.cancellation_token
            )

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        """Implements the inner loop of the orchestrator and selects next speaker."""
        # Check if we reached the maximum number of rounds
        if self._max_turns is not None and self._n_rounds > self._max_turns:
            await self._prepare_final_answer("Max rounds reached.", cancellation_token)
            return
        self._n_rounds += 1

        # Update the progress ledger
        context = self._thread_to_context()
        current_step = self._plan_manager.get_current_step()
        if current_step is None:
            await self._prepare_final_answer("All steps completed.", cancellation_token)
            return

        filtered_context = [
            msg
            for msg in context
            if isinstance(msg, (UserMessage, FunctionExecutionResultMessage))
        ]

        is_complete, reason = await self._check_step_completion(
            current_step, filtered_context, cancellation_token
        )

        if is_complete:
            self._plan_manager.set_status(current_step, PLAN_STATE.COMPLETED)
            stop_message = TextMessage(
                content=json.dumps(
                    {
                        "step": current_step,
                        "reason": f"{PLAN_STATE.COMPLETED}: {reason}",
                    },
                    indent=4,
                ),
                source="StepCompletionNotifier",
            )
            # Add this step to _contexts_history
            self._contexts_history.append(
                StepContext(step=current_step, result=filtered_context, reason=reason)
            )
            await self.publish_message(
                GroupChatMessage(message=stop_message),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(stop_message)

            # Find the next pending step after completing the current one
            current_step = self._plan_manager.get_current_step()

            if current_step is None:
                await self._prepare_final_answer(
                    "All plans completed.", cancellation_token
                )
                return

        self._plan_manager.set_status(current_step, PLAN_STATE.IN_PROGRESS)
        # Initialize or increment the counter
        if self._plan_manager.get_step_progress_counter(current_step) == 0:
            step_start_message = TextMessage(
                content=current_step,
                source="NewStepNotifier",
            )
            await self.publish_message(
                GroupChatMessage(message=step_start_message),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(step_start_message)

        # Check if the plan has been in progress for too long
        self._plan_manager.increment_step_counter(current_step)

        # If the plan has been in progress for more than 5 iterations, mark it as completed
        if self._plan_manager.get_step_progress_counter(current_step) >= 5:
            await self._log_message(
                f"Plan '{current_step}' has been in progress for 5 iterations. Marking as completed."
            )
            self._plan_manager.set_status(current_step, PLAN_STATE.FAILED)
            # Log the forced completion
            stop_message = TextMessage(
                content=json.dumps(
                    {
                        "step": current_step,
                        "reason": f"{PLAN_STATE.FAILED}: {reason}",
                    },
                    indent=4,
                ),
                source="StepCompletionNotifier",
            )
            # Add this step to _contexts_history
            self._contexts_history.append(
                StepContext(step=current_step, result=filtered_context, reason=reason)
            )
            await self.publish_message(
                GroupChatMessage(message=stop_message),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(stop_message)

            # Find the next pending step
            current_step = self._plan_manager.get_current_step()
            self._plan_manager.set_status(current_step, PLAN_STATE.IN_PROGRESS)

            # If there's no next pending step, we're done
            if current_step is None:
                await self._prepare_final_answer(
                    "All plans completed.", cancellation_token
                )
                return

        progress_ledger_prompt = orchestrator_progress_ledger_prompt(
            task=self._task, current_plan=current_step, names=self._participant_names
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))

        progress_ledger: Dict[str, Any] = {}
        assert self._max_json_retries > 0
        key_error: bool = False
        for _ in range(self._max_json_retries):
            response = await self._model_client.create(
                self._get_compatible_context(context), json_output=True
            )
            ledger_str = response.content
            try:
                assert isinstance(ledger_str, str)
                progress_ledger = json.loads(ledger_str)

                # Validate the structure
                required_keys = [
                    "instruction_or_question",
                    "next_speaker",
                ]

                key_error = False
                for key in required_keys:
                    if (
                        key not in progress_ledger
                        or not isinstance(progress_ledger[key], dict)
                        or "answer" not in progress_ledger[key]
                        or "reason" not in progress_ledger[key]
                    ):
                        key_error = True
                        break

                    if (
                        progress_ledger["next_speaker"]["answer"]
                        not in self._participant_names
                    ):
                        key_error = True
                        break

                if not key_error:
                    break
                await self._log_message(
                    f"Failed to parse ledger information, retrying: {ledger_str}"
                )
            except (json.JSONDecodeError, TypeError):
                key_error = True
                await self._log_message(
                    "Invalid ledger format encountered, retrying..."
                )
                continue
        if key_error:
            raise ValueError(
                "Failed to parse ledger information after multiple retries."
            )
        await self._log_message(f"Progress Ledger: {progress_ledger}")

        # Broadcast the next step
        message = TextMessage(
            content=progress_ledger["instruction_or_question"]["answer"],
            source=self._name,
        )
        self._message_thread.append(message)  # My copy

        # await self._log_message(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")
        logging.info(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")

        step_message = TextMessage(
            content=json.dumps(
                {
                    "tool": progress_ledger["next_speaker"]["answer"],
                    "instruction": progress_ledger["instruction_or_question"]["answer"],
                },
                indent=4,
            ),
            source="ToolCaller",
        )
        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=step_message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(step_message)

        # Broadcast it
        await self.publish_message(  # Broadcast
            GroupChatAgentResponse(agent_response=Response(chat_message=message)),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Request that the step be completed
        next_speaker = progress_ledger["next_speaker"]["answer"]
        # Check if the next speaker is valid
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the ledger, participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

    # for new feat: human in the loop
    async def _get_user_feedback_on_plan(
        self, user_proxy: UserProxyAgent, cancellation_token: CancellationToken
    ) -> None:
        # TODO(kaili): We hack the user_proxy here.
        request_user_feedback_prompt = f"Please review the plan above and provide modification suggestions. Type 'y' if the plan is acceptable."

        await self.publish_message(
            GroupChatMessage(
                message=TextMessage(
                    content=request_user_feedback_prompt, source="UserProxyAgent"
                )
            ),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        plan_to_user_dict = {}
        plan_to_user_dict["steps"] = self._plan_manager.get_plan().copy()
        plan_to_user_dict["request_user_feedback_prompt"] = request_user_feedback_prompt
        await self._output_message_queue.put(
            TextMessage(
                content=json.dumps(plan_to_user_dict, indent=4), source="UserProxyAgent"
            )
        )

    # for new feat: human in the loop
    async def _update_plan_with_feedback(
        self, feedback: str, cancellation_token: CancellationToken
    ) -> None:
        planning_conversation = []
        planning_conversation.append(
            UserMessage(
                content=f"Current Plan:\n{self._plan_manager.get_plan_str()}\n\n Update the current plan based on the following feedback:\n\nUser Feedback: {feedback}\n\n",
                source=self._name,
            )
        )
        response = await self._planning_model_client.create(
            self._get_compatible_context(planning_conversation),
            cancellation_token=cancellation_token,
        )
        assert isinstance(response.content, str)
        self._plan_manager.set_plan(response.content)

    def _format_context_for_reflection(self, messages: List[LLMMessage]) -> str:
        """Format the context messages to be more readable for the LLM."""
        formatted_lines = []

        for i, msg in enumerate(messages):
            # Extract just the content
            content = msg.content
            # Format each message with source and content
            formatted_lines.append(f"Message {i+1} from {msg.source}:\n{content}\n")

        # If there are no messages, indicate that
        if not formatted_lines:
            return "No relevant messages found in the conversation context."

        return "\n".join(formatted_lines)

    # for new feat: reflection
    async def _check_step_completion(
        self,
        current_plan: str,
        filtered_context: List[LLMMessage],
        cancellation_token: CancellationToken,
    ) -> bool:
        """Check if the current plan step has been completed based on conversation context."""
        # Format the context for better LLM understanding
        formatted_context = self._format_context_for_reflection(filtered_context)

        # Create a reflection prompt
        reflection_prompt = reflection_step_completion_prompt(
            current_plan=current_plan, conversation_context=formatted_context
        )

        reflection_context = [UserMessage(content=reflection_prompt, source=self._name)]

        response = await self._reflection_model_client.create(
            self._get_compatible_context(reflection_context),
            cancellation_token=cancellation_token,
        )

        assert isinstance(response.content, str)
        reflection = json.loads(response.content)

        reason = reflection.get("reason", "No reason provided")
        return reflection.get("is_complete", "false") == "true", reason

    async def _get_prompt_templates(
        self,
        task_description: str,
        cancellation_token: CancellationToken,
    ) -> dict:

        # Create a message asking for appropriate templates
        message = TextMessage(
            content=f"Based on this task, please determine the most appropriate prompt template type and provide it:\n\n{task_description}",
            source=self._name,
        )

        # Get response from the domain specific agent
        response = await self._domain_specific_agent.on_messages(
            [message], cancellation_token=cancellation_token
        )
        match = re.search(r"text='(.*?)'", response.chat_message.content)
        tool_response = match.group(1)
        prompt_dict = json.loads(tool_response)
        return prompt_dict

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await super().load_state(state=state)
        orchestrator_state = PlanningOrchestratorState.model_validate(state)
        self._contexts_history = orchestrator_state.contexts_history

    async def save_state(self) -> Mapping[str, Any]:
        state = await super().save_state()
        state = PlanningOrchestratorState(**state)
        state.type = "PlanningOrchestratorState"
        state.contexts_history = self._contexts_history
        return state.model_dump()

    async def _prepare_final_answer(
        self, reason: str, cancellation_token: CancellationToken
    ) -> None:
        """Prepare the final answer for the task."""
        context = self._thread_to_context()

        # Get the final answer
        final_answer_prompt = self._get_final_answer_prompt(self._task)
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )
        assert isinstance(response.content, str)
        message = TextMessage(content=response.content, source=self._name)

        self._message_thread.append(message)  # My copy

        # Set multi-round flag to true for subsequent conversations
        self._multi_round = True

        # Clear the current plan to prepare for the next round
        self._plan_manager.clear()

        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(message)

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(agent_response=Response(chat_message=message)),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        if self._termination_condition is not None:
            await self._termination_condition.reset()
        # Signal termination
        await self._signal_termination(StopMessage(content=reason, source=self._name))
