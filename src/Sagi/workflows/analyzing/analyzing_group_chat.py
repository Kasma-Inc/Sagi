import asyncio
from typing import Any, Callable, List, Mapping

from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.state import TeamState
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams._group_chat._chat_agent_container import ChatAgentContainer
from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_core import AgentRuntime, AgentType, TypeSubscription
from autogen_core.models import ChatCompletionClient
from pydantic import Field

from Sagi.workflows.analyzing.analyzing_orchestrator import AnalyzingOrchestrator


class AnalyzingChatState(TeamState):
    """State for a team of agents."""

    agent_states: Mapping[str, Any] = Field(default_factory=dict)
    type: str = Field(default="AnalyzingChatState")


class AnalyzingGroupChat(BaseGroupChat):
    def __init__(
        self,
        participants: List[ChatAgent],
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        analyzing_model_client: ChatCompletionClient,
        pg_model_client: ChatCompletionClient,
        step_triage_model_client: ChatCompletionClient,
        language: str = "en",
    ):
        super().__init__(
            participants,
            group_chat_manager_name="AnalyzingOrchestrator",
            group_chat_manager_class=AnalyzingOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError(
                "At least one participant is required for MagenticOneGroupChat."
            )
        self._analyzing_model_client = analyzing_model_client
        self._pg_model_client = pg_model_client
        self._step_triage_model_client = step_triage_model_client
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
    ) -> Callable[[], AnalyzingOrchestrator]:
        return lambda: AnalyzingOrchestrator(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            group_chat_manager_topic_type=group_chat_manager_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            max_turns=max_turns,
            message_factory=self._message_factory,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            emit_team_events=self._emit_team_events,
            analyzing_model_client=self._analyzing_model_client,
            pg_model_client=self._pg_model_client,
            step_triage_model_client=self._step_triage_model_client,
            language=self._language,
        )

    def set_language(self, language: str) -> None:
        self._language = language

    async def save_state(self) -> Mapping[str, Any]:
        base_state = await super().save_state()
        state = AnalyzingChatState(
            agent_states=base_state["agent_states"],
        )
        return state.model_dump()
