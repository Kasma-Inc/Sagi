import json
import uuid
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, BaseMessage
from pydantic import BaseModel, Field


class Step(BaseModel):
    """
    Represents a single step in a plan.

    Attributes:
        step_id (str): The unique identifier inside the plan
        group_id (str): A group contains multiple steps (currently are data_collection, code_executor, and general steps)
        content (str): The content describing what this step should accomplish
        step_progress_counter (int): The number of times the step has been executed, incremented by 1 each time the step is executed
        state (Literal["pending", "completed", "failed", "in_progress"]): The current state of the step
        messages (List[BaseAgentEvent | BaseChatMessage]): The conversation messages associated with executing this step
        reflection (Optional[str]): The reflection after the step is completed
        template_id (Optional[str]): The template ID of the step
    """

    step_id: str
    group_id: str
    content: str
    step_progress_counter: int
    state: Literal["pending", "completed", "failed", "in_progress"]
    reflection: Optional[str]
    messages: List[BaseAgentEvent | BaseChatMessage]
    template_id: Optional[str]

    def dump(self) -> Dict:
        """
        Serialize the Step object into a dictionary.

        Returns:
            Dict: A dictionary containing the step's attributes, with messages serialized as a list.
        """
        return {
            "step_id": self.step_id,
            "group_id": self.group_id,
            "content": self.content,
            "step_progress_counter": self.step_progress_counter,
            "state": self.state,
            "reflection": self.reflection,
            "messages": [msg.dump() for msg in self.messages] if self.messages else [],
            # Messages sent by the Orchestrator agent and tool agents
            "template_id": self.template_id,
        }

    @classmethod
    def load(cls, data: Dict) -> "Step":
        """
        Deserialize a dictionary into a Step object.

        Args:
            data (Dict): The dictionary containing step data.

        Returns:
            Step: A new Step instance populated with the provided data.
        """
        return cls(
            step_id=data["step_id"],
            group_id=data["group_id"],
            content=data["content"],
            step_progress_counter=data.get("step_progress_counter", 0),
            state=data.get("state", "pending"),
            reflection=data.get("reflection"),
            messages=[BaseMessage.load(msg) for msg in data.get("messages", [])],
            template_id=data.get("template_id", None),
        )


class Plan(BaseModel):
    """
    Represents a plan consisting of multiple steps, identified by a unique ID.

    Attributes:
        plan_id (str): Unique identifier for the plan.
        task (str): Description of the task this plan addresses.
        steps (OrderedDict[str, Step]): Dictionary mapping step IDs to Step objects.
        awaiting_confirmation (bool): Whether the plan awaits user confirmation.
        summary (Optional[str]): Summary of the plan.
        shared_context (OrderedDict[str, str]): A dictionary to dynamically store and update the concise result summary of each completed task group.
        group_template_ids (Dict[str, Optional[str]]): A dictionary to store template IDs for each group.
    """

    plan_id: str
    task: str
    steps: OrderedDict[str, Step]
    awaiting_confirmation: bool
    summary: Optional[str]
    shared_context: OrderedDict[str, str] = Field(
        default_factory=OrderedDict
    )  # Initialize as empty OrderedDict
    group_template_ids: Dict[int, Optional[str]] = Field(default_factory=dict)

    def get_current_step(self) -> Optional[Tuple[str, str]]:
        """
        Retrieve the first step that is either pending or in progress.
        Return the step_id and step content.

        Returns:
            Optional[Tuple[str, str]]: A tuple of (step_id, content) if found, else None.
        """
        for step_id, step in self.steps.items():
            if step.state in ["pending", "in_progress"]:
                return (step_id, step.content)
        return None

    def get_all_contents(self) -> List[str]:
        """
        Retrieve the content of all steps in the plan.

        Returns:
            List[str]: A list containing the content of each step in the plan.
        """
        return [step.content for step in self.steps.values()]

    def get_all_states(self) -> OrderedDict[str, str]:
        """
        Retrieve all step contents and their corresponding states.

        Returns:
            OrderedDict[str, str]: An ordered dictionary mapping step content to its state.
        """
        return OrderedDict((step.content, step.state) for step in self.steps.values())

    def add_message_to_step(
        self, step_id: str, message: BaseAgentEvent | BaseChatMessage
    ) -> None:
        """
        Add a message to the message list of a specific step identified by step_id.

        Args:
            step_id (str): The unique identifier of the step to add the message to.
            message (BaseAgentEvent | BaseChatMessage): The message to add to the step.

        Raises:
            ValueError: If the step with the given step_id is not found.
        """
        if step_id in self.steps:
            self.steps[step_id].messages.append(message)
        else:
            raise ValueError(f"Step with id {step_id} not found")

    def add_reflection_to_step(self, step_id: str, reflection: str) -> None:
        """
        Add a reflection to a specific step identified by step_id.
        The reflection is the result of the step execution.

        Args:
            step_id (str): The unique identifier of the step to add the reflection to.
            reflection (str): The reflection text to add to the step.

        Raises:
            ValueError: If the step with the given step_id is not found.
        """
        if step_id in self.steps:
            self.steps[step_id].reflection = reflection
        else:
            raise ValueError(f"Step with step_id {step_id} not found")

    def get_messages_by_step_id(
        self, step_id: str
    ) -> List[BaseAgentEvent | BaseChatMessage]:
        """
        Retrieve the messages of a specific step identified by step_id.

        Args:
            step_id (str): The unique identifier of the step to retrieve messages from.

        Returns:
            List[BaseAgentEvent | BaseChatMessage]: A list of messages associated with the step.

        Raises:
            ValueError: If the step with the given step_id is not found.
        """
        if step_id in self.steps:
            return self.steps[step_id].messages
        else:
            raise ValueError(f"Step with step_id {step_id} not found")

    def add_summary_to_plan(self, summary: str) -> None:
        """
        Add a summary to the plan.

        Args:
            summary (str): The summary text to add to the plan.
        """
        self.summary = summary

    def get_summary(self) -> Optional[str]:
        """
        Retrieve the summary of the plan.

        Returns:
            Optional[str]: The summary of the plan.
        """
        return self.summary

    def update_shared_context(self, step_id: str, summary: str) -> None:
        """
        Update the shared_context dictionary with a new group task summary.

        Args:
            group_id (str): The unique identifier of the group.
            summary (str): The concise summary of the group's result.
        """
        if step_id in self.steps:
            group_id = self.steps[step_id].group_id
            self.shared_context[group_id] = summary
        else:
            raise ValueError(f"Step with step_id {step_id} not found")

    def get_shared_context(self) -> OrderedDict[str, str]:
        """
        Get the current shared_context dictionary.

        Returns:
            OrderedDict[str, str]: A dictionary containing the concise result summaries of completed task groups.
        """
        return self.shared_context

    def dump(self) -> Dict:
        """
        Serialize the plan to a dictionary format.

        This method converts the Plan object into a dictionary representation
        that can be easily serialized to JSON or other formats for storage
        or transmission.

        Returns:
            Dict: A dictionary containing the serialized plan data with the following keys:
                - plan_id: The unique identifier of the plan
                - steps: A dictionary mapping step IDs to their serialized representations
                - awaiting_confirmation: Boolean indicating if the plan is awaiting user confirmation
                - summary: The plan summary text
                - shared_context: Dictionary mapping task group IDs to their summaries
                - group_template_ids: Dictionary mapping task group IDs to their template IDs
        """
        return {
            "plan_id": self.plan_id,
            "steps": {step_id: step.dump() for step_id, step in self.steps.items()},
            "awaiting_confirmation": self.awaiting_confirmation,
            "summary": self.summary,
            "task": self.task,
            "shared_context": dict(
                self.shared_context
            ),  # Convert OrderedDict to regular dict for serialization
            "group_template_ids": self.group_template_ids,
        }

    @classmethod
    def load(cls, data: Dict) -> "Plan":
        """
        Deserialize the plan from a dictionary format.

        This class method creates a new Plan object from a dictionary representation,
        typically one that was previously created by the dump() method.

        Args:
            data (Dict): A dictionary containing the serialized plan data with the following keys:
                - plan_id: The unique identifier of the plan
                - steps: A dictionary mapping step IDs to their serialized step data
                - awaiting_confirmation: Boolean indicating if the plan is awaiting user confirmation
                - summary: The plan summary text
                - shared_context: Dictionary mapping task group IDs to their summaries
                - group_template_ids: Dictionary mapping task group IDs to their template IDs

        Returns:
            Plan: A new Plan object populated with the deserialized data
        """
        return cls(
            plan_id=data["plan_id"],
            steps=OrderedDict(
                (step_id, Step.load(step)) for step_id, step in data["steps"].items()
            ),
            awaiting_confirmation=data["awaiting_confirmation"],
            summary=data["summary"],
            task=data["task"],
            shared_context=OrderedDict(
                data.get("shared_context", {})
            ),  # Convert dict back to OrderedDict
            group_template_ids=data.get("group_template_ids", {}),
        )

    def get_group_template_id(self, group_id: int) -> Optional[str]:
        return self.group_template_ids.get(group_id, None)


class PlanHistory(BaseModel):
    """
    Represents a history of plans, allowing for the storage and retrieval of multiple plans.

    Attributes:
        plan_history (List[Plan]): A list of Plan objects representing the history of plans.
    """

    plan_history: List[Plan]

    def append_plan(self, plan: Plan) -> None:
        """
        Append a new plan to the history.

        Args:
            plan (Plan): The plan to append to the history.
        """
        self.plan_history.append(plan)

    def dump(self) -> Dict:
        """
        Serialize the plan history to a dictionary format.

        This method converts the PlanHistory object into a dictionary representation
        that can be easily serialized to JSON or other formats. It iterates through
        all plans in the history and calls their dump() method to serialize each one.

        Returns:
            Dict: A dictionary containing the serialized plan history with the following structure:
                - plan_history: A list of serialized Plan objects
        """
        return {"plan_history": [plan.dump() for plan in self.plan_history]}

    @classmethod
    def load(cls, data: Dict) -> "PlanHistory":
        """
        Deserialize a dictionary into a PlanHistory object.

        This class method takes a dictionary representation of a plan history and
        converts it back into a PlanHistory object. It extracts the plan_history list
        from the input dictionary, deserializes each plan using the Plan.load method,
        and constructs a new PlanHistory instance with the deserialized plans.

        Args:
            data (Dict): A dictionary containing the serialized plan history with the following structure:
                - plan_history: A list of serialized Plan objects

        Returns:
            PlanHistory: A new PlanHistory object populated with the deserialized plans
        """
        plans_data = data.get("plan_history", [])
        return cls(plan_history=[Plan.load(plan) for plan in plans_data])


class PlanManager:
    """
    PlanManager maintains the state and history of all plans within a chat session, and retrieval the plan history for the multi-round conversation:

    """

    def __init__(self):
        """
        Initialize a new PlanManager instance.

        The PlanManager is responsible for maintaining the state and history of all plans
        within a chat session. It tracks the current active plan and stores a history
        of all plans for multi-round conversations.

        Attributes:
            _plan_history (PlanHistory): A container for all plans created in the session.
            _current_plan (Optional[Plan]): The currently active plan, or None if no plan is active.
            _human_feedback (OrderedDict[Tuple[str, ...], str]): A dictionary mapping previous plan content to corresponding human feedback.
        """
        self._plan_history = PlanHistory(plan_history=[])
        self._current_plan: Optional[Plan] = None
        self._human_feedback: OrderedDict[Tuple[str, ...], str] = OrderedDict()

    def is_plan_awaiting_confirmation(self) -> bool:
        """
        Check if the current plan is still awaiting confirmation.

        Returns:
            bool: True if the current plan is awaiting confirmation, False otherwise.
        """
        return (
            self._current_plan is not None and self._current_plan.awaiting_confirmation
        )

    def get_plan_count(self) -> int:
        """
        Get the total number of plans in the history.

        This method returns the count of all plans that have been created and stored
        in the plan history during the current chat session.

        Returns:
            int: The total number of plans in the plan history.
        """
        return len(self._plan_history.plan_history)

    def new_plan(
        self, task: str = "", model_response: str = "", human_feedback: str = ""
    ) -> None:
        """
        Set a new plan based on the task and model response.

        This method creates a new Plan object with groups extracted from the model response.
        The plan is set as the current plan and marked as awaiting confirmation.
        Adds previous plan with human feedback to the feedback dictionary.

        Args:
            task (str): The task description for the plan.
            model_response (str): JSON string containing the groups for the plan.
                                 Expected format: {"groups": [{"name": "...", "description": "...",
                                 "data_collection_task": "...", "code_executor_task": "..."}]}
            human_feedback (str): Feedback provided by the user about the plan.

        Returns:
            None
        """

        def append_step(
            steps: Dict[str, Step],
            step_id: int,
            content: str,
            group_id: int,
            template_id: Optional[str],
        ):
            """
            Utility function to append a step to the plan.

            Args:
                steps (Dict[str, Step]): A dictionary mapping step IDs to Step objects.
                step_id (int): The unique identifier for the step.
                content (str): The content of the step.
                group_id (int): The group identifier.
            """
            steps[f"step_{step_id}"] = Step(
                step_id=f"step_{step_id}",
                group_id=f"group_{group_id}",
                content=content,
                step_progress_counter=0,
                state="pending",
                reflection=None,
                messages=[],
                template_id=template_id,
            )

        def validate_model_response(model_response: str) -> None:
            """
            Validate the model response format.

            Args:
                model_response (str): JSON string containing the groups for the plan.
            """
            try:
                response_data = json.loads(model_response)
                if not isinstance(response_data, dict):
                    raise ValueError("Model response must be a JSON object")

                if "groups" not in response_data:
                    raise ValueError("Model response must contain a 'groups' key")

                if not isinstance(response_data["groups"], list):
                    raise ValueError("The 'groups' value must be a list")

                for i, group in enumerate(response_data["groups"]):
                    if not isinstance(group, dict):
                        raise ValueError(f"Group {i} must be a dictionary")

                    if "name" not in group:
                        raise ValueError(f"Group {i} must contain a 'name' key")

                    if "description" not in group:
                        raise ValueError(f"Group {i} must contain a 'description' key")

                    # data_collection_task and code_executor_task are optional
            except json.JSONDecodeError:
                raise ValueError("Model response must be a valid JSON string")

        if not task:
            task = self._current_plan.task if self._current_plan else ""
        if human_feedback and self._current_plan:
            # Save previous plan's feedback
            self._human_feedback[tuple(self._current_plan.get_all_contents())] = (
                human_feedback
            )
        validate_model_response(model_response)
        current_plan_groups = json.loads(model_response)["groups"]

        steps, group_template_ids, step_id, group_id = {}, {}, 0, 0
        for group in current_plan_groups:
            group_name = group["name"]
            group_description = group["description"]
            tasks_added = False
            template_id = group.get("template_id", None)
            if group.get("data_collection_task"):
                content = group["data_collection_task"]
                append_step(steps, step_id, content, group_id, template_id)
                step_id += 1
                tasks_added = True
            if group.get("code_executor_task") and group.get(
                "code_executor_task"
            ) not in ["N/A"]:
                content = group["code_executor_task"]
                append_step(steps, step_id, content, group_id, template_id)
                step_id += 1
                tasks_added = True
            if not tasks_added:
                content = f"{group_name}: {group_description}"
                append_step(steps, step_id, content, group_id, template_id)
                step_id += 1
            group_template_ids[group_id] = template_id
            group_id += 1
        # Create new plan
        self._current_plan = Plan(
            plan_id=f"plan_{uuid.uuid4()}",
            task=task,
            steps=steps,
            awaiting_confirmation=True,
            summary=None,
            group_template_ids=group_template_ids,
        )

    def confirm_plan(self) -> None:
        """
        Confirm the current plan, changing its state from awaiting confirmation to active.

        This method ensures the plan is ready for execution after user review.

        Raises:
            ValueError: If there is no current plan to confirm.
        """
        if not isinstance(self._current_plan, Plan):
            raise TypeError("Current plan must be a Plan object")
        self._current_plan.awaiting_confirmation = False

    def set_step_state(
        self,
        step_id: str,
        state: Literal["pending", "completed", "failed", "in_progress"],
    ) -> None:
        """
        Set the state of a specific step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to update.
            state (Literal["pending", "completed", "failed", "in_progress"]): The new state for the step.
        """
        if self._current_plan:
            self._current_plan.steps[step_id].state = state
        else:
            raise ValueError("No running plan to set step status")

    def add_summary_to_plan(self, summary: str) -> None:
        """
        Add a summary to the current plan.

        Args:
            summary (str): The summary text to add to the plan.
        """
        if self._current_plan:
            self._current_plan.add_summary_to_plan(summary)
        else:
            raise ValueError("No running plan to add summary")

    def get_current_plan_contents(self) -> List[str]:
        """
        Get the contents of all steps in the current plan.

        Returns:
            List[str]: A list of all step contents in the current plan.
        """
        return self._current_plan.get_all_contents() if self._current_plan else []

    def get_current_plan_state(self) -> OrderedDict:
        """
        Get the state of all steps in the current plan.

        Returns:
            OrderedDict[str, str]: An ordered dictionary mapping step content to its state.
        """
        return (
            self._current_plan.get_all_states() if self._current_plan else OrderedDict()
        )

    def get_current_step(self) -> Optional[Tuple[str, str]]:
        """
        Get the current step (step_id, content).

        Returns:
            Optional[Tuple[str, str]]: A tuple of (step_id, content) if found, else None.
        """
        if self._current_plan:
            return self._current_plan.get_current_step()
        return None

    def add_message_to_step(
        self, step_id: str, message: BaseAgentEvent | BaseChatMessage
    ) -> None:
        """
        Add a message to a specific step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to add the message to.
            message (BaseAgentEvent | BaseChatMessage): The message to add to the step.
        """
        if self._current_plan:
            self._current_plan.add_message_to_step(step_id, message)
        else:
            raise ValueError("No running plan")

    def add_reflection_to_step(self, step_id: str, reflection: str) -> None:
        """
        Add a reflection to a specific step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to add the reflection to.
            reflection (str): The reflection text to add to the step.
        """
        if self._current_plan:
            self._current_plan.add_reflection_to_step(step_id, reflection)
        else:
            raise ValueError("No running plan")

    def get_messages_by_step_id(
        self, step_id: str
    ) -> List[BaseAgentEvent | BaseChatMessage]:
        """
        Get the messages for a specific step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to retrieve messages from.
        """
        if self._current_plan:
            return self._current_plan.get_messages_by_step_id(step_id)
        else:
            raise ValueError("No running plan")

    def get_messages_of_current_step(self) -> List[BaseAgentEvent | BaseChatMessage]:
        """
        Get the messages of the current step.

        Returns:
            List[BaseAgentEvent | BaseChatMessage]: A list of messages associated with the current step.
        """
        current_step = self.get_current_step()
        if current_step is None:
            return []
        return self.get_messages_by_step_id(current_step[0])

    def get_messages_of_current_plan(self) -> List[BaseAgentEvent | BaseChatMessage]:
        """
        Get the messages of all steps in the current plan.

        Returns:
            List[BaseAgentEvent | BaseChatMessage]: A list of messages associated with all steps in the current plan.
        """
        if self._current_plan:
            return [
                msg
                for step in self._current_plan.steps.values()
                for msg in step.messages
            ]
        else:
            raise ValueError("No running plan")

    def get_step_progress_counter(self, step_id: str) -> int:
        """
        Get the progress counter for a specific step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to get the progress counter for.

        Returns:
            int: The progress counter for the step.

        Raises:
            ValueError: If there is no running plan or the step with the given step_id is not found.
        """
        if not self._current_plan:
            raise ValueError("No running plan")
        if step_id in self._current_plan.steps:
            return self._current_plan.steps[step_id].step_progress_counter
        raise ValueError(f"Step with step_id {step_id} not found")

    def get_current_plan_id(self) -> str:
        """
        Get the ID of the current plan.

        Returns:
            str: The ID of the current plan.
        """
        if self._current_plan:
            return self._current_plan.plan_id
        else:
            raise ValueError("No running plan")

    def get_total_steps_current_plan(self) -> int:
        """
        Get the total number of steps in the current plan.
        """
        if self._current_plan:
            return len(self._current_plan.steps)
        return 0

    def increment_step_counter(self, step_id: str) -> None:
        """
        Increment the counter for a step in the current plan by step_id.

        Args:
            step_id (str): The unique identifier of the step to increment the counter for.

        Raises:
            ValueError: If there is no running plan or the step with the given step_id is not found.
        """
        if not self._current_plan:
            raise ValueError("No running plan")
        if step_id in self._current_plan.steps:
            self._current_plan.steps[step_id].step_progress_counter += 1
        else:
            raise ValueError(f"Step with step_id {step_id} not found")

    def get_task(self) -> str:
        """
        Get the task of the current plan.

        Returns:
            str: The task of the current plan.

        Raises:
            ValueError: If there is no running plan.
        """
        if self._current_plan:
            return self._current_plan.task
        else:
            raise ValueError("No running plan")

    def set_task(self, task: str) -> None:
        """
        Set the task of the current plan.

        Args:
            task (str): The task to set for the current plan.

        Raises:
            ValueError: If there is no running plan.
        """
        if self._current_plan:
            self._current_plan.task = task
        else:
            raise ValueError("No running plan")

    def commit_plan(self) -> None:
        """
        Commit the current plan to the plan history and reset the current plan.

        Raises:
            ValueError: If there is no running plan.
        """
        if self._current_plan:
            self._plan_history.append_plan(self._current_plan)
            self._current_plan = None
        else:
            raise ValueError("No running plan")

    def get_plan_history(self) -> PlanHistory:
        """
        Get the plan history.

        Returns:
            PlanHistory: The plan history.
        """
        return self._plan_history

    def get_plan_history_str(self) -> str:
        """
        Get the plan history as a string.

        Returns:
            str: The plan history as a string.
        """
        if self.get_plan_count() == 0:
            return "No context history available."

        return "\n".join(
            [
                f"Content: {step.content}\nResult: {step.messages[-1] if step.messages else 'No messages'}\nReason: {step.reflection if hasattr(step, 'reflection') else 'No reflection'}"
                for plan in self._plan_history.plan_history
                for step in plan.steps.values()
                if hasattr(plan, "steps") and isinstance(plan.steps, dict)
            ]
        )

    def dump(self) -> Dict:
        """
        Serialize the plan manager.

        Returns:
            Dict: The serialized plan manager.
        """
        return {
            "plan_history": self._plan_history.dump(),
            "current_plan": (self._current_plan.dump() if self._current_plan else None),
            "human_feedback": {",".join(k): v for k, v in self._human_feedback.items()},
        }

    @classmethod
    def load(cls, data: Dict) -> "PlanManager":
        """
        Deserialize the plan manager.

        Args:
            data (Dict): The serialized plan manager data.

        Returns:
            PlanManager: The deserialized plan manager.
        """
        obj = cls()
        obj._plan_history = PlanHistory.load(data["plan_history"])
        obj._current_plan = (
            Plan.load(data["current_plan"]) if data.get("current_plan") else None
        )
        obj._human_feedback = OrderedDict(
            (tuple(k.split(",")), v) for k, v in data.get("human_feedback", {}).items()
        )
        return obj

    def reset(self) -> None:
        """
        Reset the plan manager.

        This method clears the current plan and resets the plan history to an empty state.
        All previously stored plans and their associated data will be removed.
        """
        self._current_plan = None
        self._plan_history = PlanHistory(plan_history=[])

    def get_group_template_id(self, group_id: int) -> Optional[str]:
        if self._current_plan:
            return self._current_plan.group_template_ids.get(group_id, None)
        else:
            raise ValueError("No running plan")
