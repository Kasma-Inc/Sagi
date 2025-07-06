import asyncio
from typing import Any, Callable, List, Mapping

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.state import TeamState
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams._group_chat._chat_agent_container import ChatAgentContainer
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_core import AgentRuntime, AgentType, TypeSubscription
from autogen_core.models import ChatCompletionClient
from pydantic import Field

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from Sagi.workflows.planning_html.planning_html_orchestrator import (
    PlanningHtmlOrchestrator,
)


class PlanningHtmlChatState(TeamState):
    """State for a team of agents."""

    agent_states: Mapping[str, Any] = Field(default_factory=dict)
    type: str = Field(default="PlanningHtmlChatState")


class PlanningHtmlGroupChat(BaseGroupChat):
    def __init__(
        self,
        participants: List[ChatAgent],
        orchestrator_model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        planning_model_client: ChatCompletionClient,  # for json plan output
        reflection_model_client: ChatCompletionClient,  # for reflection
        step_triage_model_client: ChatCompletionClient,
        domain_specific_agent: (
            AssistantAgent | None
        ) = None,  # for new feat: domain specific prompt
        template_work_dir: str | None = None,  # for template working directory
        template_selection_model_client: ChatCompletionClient,
        template_based_planning_model_client: ChatCompletionClient,
        single_group_planning_model_client: ChatCompletionClient,
        language: str = "en",
    ):
        super().__init__(
            participants,
            group_chat_manager_name="PlanningHtmlOrchestrator",
            group_chat_manager_class=PlanningHtmlOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError(
                "At least one participant is required for PlanningHtmlGroupChat."
            )
        # Initialize the model clients.
        self._orchestrator_model_client = orchestrator_model_client
        self._planning_model_client = planning_model_client  # for json plan output
        self._reflection_model_client = (
            reflection_model_client  # for new feat: reflection
        )
        self._step_triage_model_client = step_triage_model_client
        self._domain_specific_agent = (
            domain_specific_agent  # for new feat: domain specific prompt
        )
        self._template_work_dir = template_work_dir  # for template working directory
        self._template_selection_model_client = template_selection_model_client
        self._template_based_planning_model_client = (
            template_based_planning_model_client
        )
        self._single_group_planning_model_client = single_group_planning_model_client
        self._language = language

    async def _init(self, runtime: AgentRuntime) -> None:
        # Constants for the group chat manager.
        group_chat_manager_agent_type = AgentType(self._group_chat_manager_topic_type)

        # Register participants.
        # Use the participant topic type as the agent type.
        for participant, agent_type in zip(
            self._participants, self._participant_topic_types, strict=True
        ):
            # Register the participant factory.
            await ChatAgentContainer.register(
                runtime,
                type=agent_type,
                factory=self._create_participant_factory(
                    self._group_topic_type,
                    self._output_topic_type,
                    participant,
                    self._message_factory,
                ),
            )
            # Add subscriptions for the participant.
            # The participant should be able to receive messages from its own topic.
            await runtime.add_subscription(
                TypeSubscription(topic_type=agent_type, agent_type=agent_type)
            )
            # The participant should be able to receive messages from the group topic.
            await runtime.add_subscription(
                TypeSubscription(
                    topic_type=self._group_topic_type, agent_type=agent_type
                )
            )

        # Register the group chat manager.
        await self._base_group_chat_manager_class.register(
            runtime,
            type=group_chat_manager_agent_type.type,
            factory=self._create_group_chat_manager_factory(
                name=self._group_chat_manager_name,
                group_topic_type=self._group_topic_type,
                output_topic_type=self._output_topic_type,
                group_chat_manager_topic_type=self._group_chat_manager_topic_type,
                participant_names=self._participant_names,
                participant_topic_types=self._participant_topic_types,
                participant_descriptions=self._participant_descriptions,
                output_message_queue=self._output_message_queue,
                termination_condition=self._termination_condition,
                max_turns=self._max_turns,
            ),
        )
        # Add subscriptions for the group chat manager.
        # The group chat manager should be able to receive messages from the its own topic.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._group_chat_manager_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )
        # The group chat manager should be able to receive messages from the group topic.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._group_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )
        # The group chat manager will relay the messages from output topic to the output message queue.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._output_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )

        self._initialized = True

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        group_chat_manager_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
    ) -> Callable[[], PlanningHtmlOrchestrator]:
        return lambda: PlanningHtmlOrchestrator(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            group_chat_manager_topic_type=group_chat_manager_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            max_turns=max_turns,
            message_factory=self._message_factory,
            orchestrator_model_client=self._orchestrator_model_client,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            emit_team_events=self._emit_team_events,
            planning_model_client=self._planning_model_client,
            reflection_model_client=self._reflection_model_client,  # for new feat: reflection
            step_triage_model_client=self._step_triage_model_client,
            domain_specific_agent=self._domain_specific_agent,  # for new feat: domain specific prompt
            template_work_dir=self._template_work_dir,  # for template working directory
            template_selection_model_client=self._template_selection_model_client,
            template_based_planning_model_client=self._template_based_planning_model_client,
            single_group_planning_model_client=self._single_group_planning_model_client,
            language=self._language,
        )

    def set_language(self, language: str) -> None:
        self._language = language

    async def load_chat_id(self, chat_id: str) -> None:
        for participant in self._participants:
            if isinstance(participant, StreamCodeExecutorAgent):
                participant.chat_id = chat_id

    async def save_state(self) -> Mapping[str, Any]:
        base_state = await super().save_state()
        state = PlanningHtmlChatState(
            agent_states=base_state["agent_states"],
        )
        return state.model_dump()

    async def set_id_info(self, user_id: str, chat_id: str) -> None:
        pass
