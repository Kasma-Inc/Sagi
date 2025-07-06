import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Mapping, Optional, Tuple

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    BaseAgentEvent,
    BaseChatMessage,
    BaseMessage,
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
    GroupChatReset,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_agentchat.utils import content_to_str, remove_images
from autogen_core import (
    AgentId,
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

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
)
from Sagi.utils.hirag_message import hirag_message_to_llm_message
from Sagi.utils.json_handler import (
    format_file_content,
    format_slide_info,
    format_templates,
)
from Sagi.utils.prompt import (
    get_appended_plan_prompt,
    get_appended_plan_prompt_cn,
    get_expand_plan_prompt,
    get_final_answer_prompt,
    get_final_answer_prompt_cn,
    get_high_level_ppt_plan_prompt,
    get_new_task_description_prompt,
    get_reflection_step_completion_prompt,
    get_reflection_step_completion_prompt_cn,
    get_step_triage_prompt,
    get_step_triage_prompt_cn,
    get_template_selection_prompt,
)
from Sagi.workflows.planning_html.plan_manager import PlanManager

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


class UserInputMessage(BaseModel):
    messages: List[ChatMessage]


class PlanningHtmlOrchestratorState(BaseModel):
    type: str = Field(default="PlanningHtmlOrchestratorState")
    plan_manager_state: Dict = Field(default_factory=dict)


class PlanningHtmlOrchestrator(BaseGroupChatManager):
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
        orchestrator_model_client: ChatCompletionClient,
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        emit_team_events: bool,
        planning_model_client: ChatCompletionClient,
        reflection_model_client: ChatCompletionClient,
        step_triage_model_client: ChatCompletionClient,
        template_selection_model_client: ChatCompletionClient,
        template_based_planning_model_client: ChatCompletionClient,
        single_group_planning_model_client: ChatCompletionClient,
        user_proxy: Any | None = None,
        domain_specific_agent: Any | None = None,
        template_work_dir: str | None = None,
        language: str = "en",
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
        self._orchestrator_model_client = orchestrator_model_client
        self._planning_model_client = planning_model_client
        self._reflection_model_client = reflection_model_client
        self._step_triage_model_client = step_triage_model_client
        self._domain_specific_agent = (
            domain_specific_agent  # for new feat: domain specific prompt
        )
        self._template_work_dir = template_work_dir
        self._template_selection_model_client = template_selection_model_client
        self._template_based_planning_model_client = (
            template_based_planning_model_client
        )
        self._single_group_planning_model_client = single_group_planning_model_client
        self._group_chat_manager_topic_type = group_chat_manager_topic_type
        self._prompt_templates = {}  # to store domain specific prompts
        self._language = language
        self._plan_manager = PlanManager()  # Initialize plan manager

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
        self._plan_manager.reset()
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
        if not self._plan_manager.is_plan_awaiting_confirmation():
            if self._plan_manager.get_plan_count() > 1:
                await self._get_appended_plan(message, ctx)
            else:
                if self._template_work_dir is not None:
                    await self._get_initial_plan_based_templates(message, ctx)
                else:
                    await self._get_initial_plan(message, ctx)
        else:
            await self._human_in_the_loop(message, ctx)

    async def _stream_from_model(
        self,
        model_client,
        messages,
        cancellation_token,
        source_name: Optional[str] = None,
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

        if stream is None:
            return ""
        cur_stream_id = str(uuid.uuid4())
        async for response in stream:
            if isinstance(response, str):
                chunk_event = ModelClientStreamingChunkEvent(
                    content=response,
                    source=self._name if source_name is None else source_name,
                    metadata={
                        "stream_id": cur_stream_id,
                    },
                )
                await self._output_message_queue.put(chunk_event)
            else:
                content = response.content

        assert isinstance(content, str)
        return content

    @event
    async def handle_agent_response(self, message: GroupChatAgentResponse, ctx: MessageContext) -> None:  # type: ignore
        delta: List[BaseAgentEvent | BaseChatMessage] = []
        current_step = self._plan_manager.get_current_step()
        step_id = current_step[0] if current_step is not None else ""
        if message.agent_response.inner_messages is not None:
            for inner_message in message.agent_response.inner_messages:
                delta.append(inner_message)
                inner_message.metadata["step_id"] = step_id
                await self._output_message_queue.put(inner_message)

        # For web app
        plan_state_message = TextMessage(
            content=json.dumps(self._plan_manager.get_all_step_states(), indent=4),
            source="PlanState",
            metadata={"step_id": step_id},
        )
        await self._output_message_queue.put(plan_state_message)

        self._plan_manager.add_message_to_step(
            step_id=step_id,
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

    async def _get_facts_message(
        self, task: str, ctx: MessageContext
    ) -> AssistantMessage:
        """Collect facts for a given task and return as an AssistantMessage."""
        facts_prompt = self._prompt_templates["facts_prompt"].format(task=task)
        facts_conversation = [UserMessage(content=facts_prompt, source=self._name)]
        facts_response = await self._llm_create(
            self._orchestrator_model_client,
            facts_conversation,
            ctx.cancellation_token,
            source_name="PlanningStage",
        )
        return AssistantMessage(content=facts_response, source=self._name)

    async def _get_plan_and_feedback(
        self,
        task: str,
        plan_prompt: str,
        facts_message: AssistantMessage,
        client,
        ctx: MessageContext,
    ) -> None:
        """Helper to send plan prompt to LLM, update plan, and request feedback."""
        conversation = [
            facts_message,
            UserMessage(content=plan_prompt, source=self._name),
        ]
        plan_response = await self._llm_create(
            client, conversation, ctx.cancellation_token, source_name="PlanningStage"
        )
        # TODO: Add the step to generate the html layout
        self._plan_manager.new_plan(plan_description=task, model_response=plan_response)
        await self._get_user_feedback_on_plan()

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

        # Collect facts using the extracted method
        facts_message = await self._get_facts_message(task, ctx)

        # Create the initial plan
        plan_prompt = self._prompt_templates["plan_prompt"].format(
            team=self._team_description,
            task=task,
        )
        await self._get_plan_and_feedback(
            task, plan_prompt, facts_message, self._planning_model_client, ctx
        )

    async def _get_initial_plan_based_templates(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        """Create the initial plan based on user input and templates"""

        # Compose the task from message contents
        task = await self._compose_task(message)
        logging.info(f"Task: {task}")

        if self._template_work_dir is None:
            raise ValueError("Template work directory is not set")

        file_content_path = os.path.join(self._template_work_dir, "file_content.json")
        high_level_ppt_plan_prompt = get_high_level_ppt_plan_prompt(
            task=task, file_content=format_file_content(file_content_path)
        )

        template_based_high_level_ppt_plan = await self._llm_create(
            self._template_based_planning_model_client,
            [SystemMessage(content=high_level_ppt_plan_prompt)],
            ctx.cancellation_token,
            source_name="PlanningStage",
        )

        template_based_high_level_ppt_plan_enum = json.loads(
            template_based_high_level_ppt_plan
        )

        slide_induction_path = os.path.join(
            self._template_work_dir, "slide_induction.json"
        )
        template_options = format_templates(slide_induction_path)

        task_groups: List[Dict[str, str]] = []
        for _, slide in enumerate(template_based_high_level_ppt_plan_enum["slides"]):
            expand_plan_prompt = get_expand_plan_prompt(
                plan_description=task,
                slide_content=format_slide_info(slide),
            )
            expand_single_group_response = await self._llm_create(
                self._single_group_planning_model_client,
                [SystemMessage(content=expand_plan_prompt)],
                ctx.cancellation_token,
                source_name="PlanningStage",
            )
            expand_single_group = json.loads(expand_single_group_response)

            template_selection_prompt = get_template_selection_prompt(
                slide_content=format_slide_info(slide),
                template_options=template_options,
            )
            template_selection_response = await self._llm_create(
                self._template_selection_model_client,
                [SystemMessage(content=template_selection_prompt)],
                ctx.cancellation_token,
            )
            template_selection = json.loads(template_selection_response)
            expand_single_group["template_id"] = str(template_selection["template_id"])
            task_groups.append(expand_single_group)
        self._plan_manager.new_plan(
            plan_description=task, model_response=json.dumps({"tasks": task_groups})
        )
        # TODO: add fine-grained human in the loop for plan adjustment
        self._plan_manager.confirm_plan()
        await self._orchestrate_step(cancellation_token=ctx.cancellation_token)

    async def _get_appended_plan(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        """Create a new plan for multi-round conversations based on context history."""

        task = await self._compose_task(message)
        formatted_history = self._plan_manager.get_plan_history_summary()

        # Collect facts using the extracted method
        facts_message = await self._get_facts_message(task, ctx)

        if self._language == "en":
            plan_prompt = get_appended_plan_prompt(
                current_task=task,
                contexts_history=formatted_history,
                team_composition=self._team_description,
            )
        else:
            plan_prompt = get_appended_plan_prompt_cn(
                current_task=task,
                contexts_history=formatted_history,
                team_composition=self._team_description,
            )
        await self._get_plan_and_feedback(
            task, plan_prompt, facts_message, self._planning_model_client, ctx
        )

    async def _human_in_the_loop(
        self, message: UserInputMessage, ctx: MessageContext
    ) -> None:
        feedback = message.messages[-1].content
        assert isinstance(feedback, str)
        if feedback.strip().lower() in {"ok", "yes", "y"}:
            self._plan_manager.confirm_plan()
            await self._orchestrate_step(cancellation_token=ctx.cancellation_token)
        else:
            await self._update_plan_with_feedback(message, ctx.cancellation_token)
            await self._get_user_feedback_on_plan()

    def messages_to_context(self, messages: List[BaseMessage]) -> List[LLMMessage]:
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
            elif m.source == "retrieval_agent":
                try:
                    context.append(UserMessage(content=m.content, source=m.source))
                except Exception as e:
                    logging.error(f"Error in hirag_message_to_llm_message: {e}")
                    context.append(m)
            else:
                assert isinstance(
                    m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage)
                )
                context.append(UserMessage(content=m.content, source=m.source))
        return context

    async def _get_task_summary(
        self,
        filtered_context: List[LLMMessage],
        current_step_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        if len(filtered_context) > 0:
            # Update the task summary
            if self._language == "en":
                summary_prompt = f"""Please summary the results of the current task, please list the key points and the results. Meanwhile, please list the points that are not completed.
                The goal of the task is {self._plan_manager.get_current_task_description()}"""
            else:
                summary_prompt = f"""请总结当前任务的结果，请列出关键点和结果。同时，请列出未完成的关键点。
                当前任务的目标是 {self._plan_manager.get_current_task_description()}"""

            filtered_context = [
                msg for msg in filtered_context if msg.source != "StepReflection"
            ]
            summary_prompt += f"\n\nThe current execution result is: {filtered_context}"

            existing_task_summary = self._plan_manager.get_task_summary_by_step_id(
                current_step_id
            )
            if existing_task_summary != "":
                summary_prompt += (
                    f"\n\nThe existing task summary is: {existing_task_summary}"
                )

            context = [UserMessage(content=summary_prompt, source=self._name)]
            summary = await self._llm_create(
                self._orchestrator_model_client, context, cancellation_token
            )
            return summary
        else:
            return ""

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        current_step = self._plan_manager.get_current_step()
        if current_step is None:
            await self._prepare_final_answer("All steps completed.", cancellation_token)
            return
        else:
            current_step_id, current_step_content = current_step

        # Check if the step is complete
        context = self.messages_to_context(
            self._plan_manager.get_step_messages(current_step_id)
        )
        filtered_context = [
            msg
            for msg in context
            if isinstance(msg, (UserMessage, FunctionExecutionResultMessage))
        ]

        is_complete, reason = await self._check_step_completion(
            current_step_content, filtered_context, cancellation_token
        )

        if is_complete:
            self._plan_manager.update_step_state(current_step_id, "completed")
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
            self._plan_manager.add_step_reflection(current_step_id, reason)
            await self._output_message_queue.put(step_completion_message)
            task_summary = await self._get_task_summary(
                filtered_context, current_step_id, cancellation_token
            )
            self._plan_manager.add_task_summary(
                current_step_id, task_summary, overwrite=True
            )

            # Find the next pending step after completing the current one
            current_step = self._plan_manager.get_current_step()

            if current_step is None:
                await self._prepare_final_answer(
                    "All plans completed.", cancellation_token
                )
                return
            else:
                current_step_id, current_step_content = current_step
        else:
            if self._plan_manager.get_step_progress_counter(current_step_id) > 0:
                self._plan_manager.add_message_to_step(
                    step_id=current_step_id,
                    message=TextMessage(
                        content=reason,
                        source="StepReflection",
                    ),
                )

        # Initialize or increment the counter
        if self._plan_manager.get_step_progress_counter(current_step_id) == 0:
            # Start a new step
            self._plan_manager.update_step_state(current_step_id, "in_progress")
            # Notify the frontend that a new step has started
            step_start_json = {
                "stepId": current_step_id,
                "content": current_step_content,
                "totalSteps": self._plan_manager.get_total_steps(),
            }
            step_start_message = TextMessage(
                content=json.dumps(step_start_json, indent=4),
                source="NewStepNotifier",
            )
            await self._output_message_queue.put(step_start_message)
        else:
            # Add the reflection to the model_context for the non-first attempt
            await self.publish_message(
                GroupChatAgentResponse(
                    agent_response=Response(
                        chat_message=TextMessage(
                            content=reason, source="StepReflection"
                        )
                    )
                ),
                topic_id=DefaultTopicId(type=self._group_topic_type),
                cancellation_token=cancellation_token,
            )

        # Check if the plan has been in progress for too long
        self._plan_manager.increment_step_progress_counter(current_step_id)

        # If the plan has been in progress for more than 5 iterations, mark it as completed
        if self._plan_manager.get_step_progress_counter(current_step_id) >= 2:
            await self._log_message(
                f"Plan '{current_step}' has been in progress for 5 iterations. Marking as completed."
            )
            self._plan_manager.update_step_state(current_step_id, "failed")
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
            self._plan_manager.add_step_reflection(current_step_id, reason)
            await self._output_message_queue.put(step_failed_message)
            task_summary = await self._get_task_summary(
                filtered_context, current_step_id, cancellation_token
            )
            self._plan_manager.add_task_summary(
                current_step_id, task_summary, overwrite=True
            )

            # Find the next pending step
            current_step = self._plan_manager.get_current_step()

            # If there's no next pending step, we're done
            if current_step is None:
                await self._prepare_final_answer(
                    "All plans completed.", cancellation_token
                )
                return
            else:
                current_step_id, current_step_content = current_step

            self._plan_manager.update_step_state(current_step_id, "in_progress")

        if self._language == "en":
            step_triage_prompt = get_step_triage_prompt(
                task=self._plan_manager.get_current_task_description(),
                current_plan=current_step_content,
                names=self._participant_names,
                team_description=self._team_description,
            )
        else:
            step_triage_prompt = get_step_triage_prompt_cn(
                task=self._plan_manager.get_current_task_description(),
                current_plan=current_step_content,
                names=self._participant_names,
                team_description=self._team_description,
            )
        context.append(UserMessage(content=step_triage_prompt, source=self._name))

        step_triage_response = await self._llm_create(
            self._step_triage_model_client, context, cancellation_token
        )
        step_triage = json.loads(step_triage_response)

        next_speaker = step_triage["next_speaker"]["answer"]
        logging.info(f"Next Speaker: {next_speaker}")

        step_running_message = TextMessage(
            content=json.dumps(
                {
                    "tool": next_speaker,
                    "instruction": step_triage["next_speaker"]["instruction"],
                    "stepId": current_step_id,
                },
                indent=4,
            ),
            source="ToolCaller",
        )
        # Log it to the output queue.
        await self._output_message_queue.put(step_running_message)

        messages_for_current_step: List[BaseMessage] = []
        messages_for_current_step = [
            TextMessage(
                content=f"You are now focusing on solving the following step: {current_step_content}",
                source=self._name,
            ),
            TextMessage(
                content=get_new_task_description_prompt(
                    plan_description=self._plan_manager.get_plan_description(),
                    tasks_in_plan=self._plan_manager.get_all_task_descriptions(),
                    previous_task_summary=self._plan_manager.get_task_summaries_text(),
                    task_description=self._plan_manager.get_current_task_description(),
                ),
                source=self._name,
            ),
        ]
        if len(self._plan_manager.get_step_messages(current_step_id)) > 0:
            messages_for_current_step.append(
                TextMessage(
                    content="Recall that so far, you have tried the following attempts:\n",
                    source=self._name,
                )
            )
            messages_for_current_step.extend(
                self._plan_manager.get_step_messages(current_step_id)
            )

        # TODO: handle the case where the next speaker is not in the team
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the step triage, participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]

        # Clear the message buffer for agent since we record the messages via self._plan_manager
        await self.send_message(
            GroupChatReset(),
            recipient=AgentId(type=participant_topic_type, key=next_speaker),
        )

        # content = self.messages_to_context(messages_for_current_step)
        try:
            messages_for_current_step = [
                hirag_message_to_llm_message(m) if m.source == "retrieval_agent" else m
                for m in messages_for_current_step
            ]
        except Exception as e:
            logging.error(f"Error in hirag_message_to_llm_message: {e}")

        # Set the _buffer_message of the participant to the messages_for_current_step
        await self.publish_message(  # Broadcast
            GroupChatStart(messages=messages_for_current_step),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _get_user_feedback_on_plan(self) -> None:
        request_user_feedback_prompt = f"Please review the plan above and provide modification suggestions. Type 'y' if the plan is acceptable."

        plan_to_user_dict = {}
        plan_to_user_dict["steps"] = self._plan_manager.get_all_step_contents().copy()
        # plan_to_user_dict["request_user_feedback_prompt"] = request_user_feedback_prompt
        # Generate the feedback
        plan_to_user_dict["posFeedback"] = "Start to research"
        plan_to_user_dict["negFeedback"] = "Modify the plan"
        plan_to_user_dict["planId"] = self._plan_manager.get_current_plan_id()
        await self._output_message_queue.put(
            TextMessage(
                content=json.dumps(plan_to_user_dict, indent=4), source="UserProxyAgent"
            )
        )

    # for new feat: human in the loop
    async def _update_plan_with_feedback(
        self, message: UserInputMessage, cancellation_token: CancellationToken
    ) -> None:
        human_feedback = await self._compose_task(message)
        planning_conversation = []
        current_plan_contents = json.dumps(
            self._plan_manager.get_all_step_contents(), indent=4
        )

        if self._language == "en":
            planning_conversation.append(
                UserMessage(
                    content=f"Current Plan:\n{current_plan_contents}\n\n Update the current plan based on the following feedback:\n\nUser Feedback: {human_feedback}\n\n",
                    source=self._name,
                )
            )
        else:
            planning_conversation.append(
                UserMessage(
                    content=f"当前计划：\n{current_plan_contents}\n\n 根据以下的用户反馈更新当前计划：\n\n用户反馈：{human_feedback}\n\n**计划内容使用中文**",
                    source=self._name,
                )
            )
        plan_response = await self._llm_create(
            self._planning_model_client,
            planning_conversation,
            cancellation_token,
            source_name="PlanningStage",
        )
        self._plan_manager.new_plan(
            model_response=plan_response,
            human_feedback=human_feedback,
        )

    async def _check_step_completion(
        self,
        current_plan_content: str,
        filtered_context: List[LLMMessage],
        cancellation_token: CancellationToken,
    ) -> Tuple[bool, str]:
        """Check if the current plan step has been completed based on conversation context."""
        # Format the context for better LLM understanding
        formatted_context = (
            "\n".join(
                [
                    f"Reponses from tool call({msg.source}):\n{msg.content}"
                    for i, msg in enumerate(filtered_context)
                ]
            )
            if filtered_context
            else "No relevant messages found in the conversation context."
        )

        # Create a reflection prompt
        if self._language == "en":
            reflection_prompt = get_reflection_step_completion_prompt(
                current_plan=current_plan_content,
                conversation_context=formatted_context,
            )
        else:
            reflection_prompt = get_reflection_step_completion_prompt_cn(
                current_plan=current_plan_content,
                conversation_context=formatted_context,
            )

        reflection_context = [UserMessage(content=reflection_prompt, source=self._name)]

        reflection_response = await self._llm_create(
            self._reflection_model_client, reflection_context, cancellation_token
        )
        reflection = json.loads(reflection_response)

        reason = reflection.get("reason", "No reason provided")
        return reflection.get("is_complete", "false") == "true", reason

    async def _get_prompt_templates(
        self,
        task_description: str,
        cancellation_token: CancellationToken,
    ) -> dict:

        # Create a message asking for appropriate templates (English Version)
        if self._language == "en":
            message = TextMessage(
                content=f"Based on this task, please determine the most appropriate prompt template type (English Version) and provide it:\n\n{task_description}",
                source=self._name,
            )
        else:
            message = TextMessage(
                content=f"请根据本任务确定最合适的提示模板类型（中文版）并提供：\n\n{task_description}",
                source=self._name,
            )

        assert (
            self._domain_specific_agent is not None
        ), "Domain specific agent is not set"
        # Get response from the domain specific agent
        response = await self._domain_specific_agent.on_messages(
            [message], cancellation_token=cancellation_token
        )
        tool_response = json.loads(response.chat_message.content)[0].get("text")
        prompt_dict = json.loads(tool_response)
        return prompt_dict

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = PlanningHtmlOrchestratorState.model_validate(state)
        self._plan_manager = PlanManager.load(orchestrator_state.plan_manager_state)

    async def save_state(self) -> Mapping[str, Any]:
        state = PlanningHtmlOrchestratorState(
            plan_manager_state=self._plan_manager.dump(),
        )
        return state.model_dump()

    async def _prepare_final_answer(
        self, reason: str, cancellation_token: CancellationToken
    ) -> None:
        """Prepare the final answer for the task."""
        context = self.messages_to_context(self._plan_manager.get_all_plan_messages())

        # Get the final answer
        if self._language == "en":
            final_answer_prompt = get_final_answer_prompt(
                task=self._plan_manager.get_plan_description()
            )
        else:
            final_answer_prompt = get_final_answer_prompt_cn(
                task=self._plan_manager.get_plan_description()
            )
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        final_answer_response = await self._llm_create(
            self._orchestrator_model_client, context, cancellation_token
        )

        message = TextMessage(
            content=json.dumps(
                {
                    "content": final_answer_response,
                    "planId": self._plan_manager.get_current_plan_id(),
                },
                indent=4,
            ),
            source="final_answer",
        )

        self._plan_manager.add_plan_summary(
            summary=message.content,
        )

        # Clear the current plan to prepare for the next round
        self._plan_manager.commit_plan()

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

    async def _llm_create(
        self,
        client,
        conversation: list,
        cancellation_token,
        source_name: Optional[str] = None,
    ) -> str:
        """Send conversation to LLM client and return the string content."""
        response = await self._stream_from_model(
            model_client=client,
            messages=conversation,
            cancellation_token=cancellation_token,
            source_name=source_name,
        )
        return response
