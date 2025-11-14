import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple

from autogen_agentchat.messages import (
    BaseMessage,
    CodeExecutionEvent,
    CodeGenerationEvent,
    HandoffMessage,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    SelectSpeakerEvent,
    StopMessage,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from hirag_prod.json_utils import ModelJSONDecodeError, safe_model_json_loads
from pydantic import BaseModel, Field

MessageClsDict = {
    "TextMessage": TextMessage,
    "MultiModalMessage": MultiModalMessage,
    "StopMessage": StopMessage,
    "ToolCallSummaryMessage": ToolCallSummaryMessage,
    "HandoffMessage": HandoffMessage,
    "ToolCallRequestEvent": ToolCallRequestEvent,
    "ToolCallExecutionEvent": ToolCallExecutionEvent,
    "MemoryQueryEvent": MemoryQueryEvent,
    "UserInputRequestedEvent": UserInputRequestedEvent,
    "ModelClientStreamingChunkEvent": ModelClientStreamingChunkEvent,
    "ThoughtEvent": ThoughtEvent,
    "SelectSpeakerEvent": SelectSpeakerEvent,
    "CodeGenerationEvent": CodeGenerationEvent,
    "CodeExecutionEvent": CodeExecutionEvent,
}

# ===== Data Models =====


class Step(BaseModel):
    """
    Represents a single step in a plan.

    Attributes:
        step_id (str): The unique identifier inside the plan
        task_id (str): A task contains multiple steps (currently are data_collection, code_executor, and general steps)
        content (str): The content describing what this step should accomplish
        step_progress_counter (int): The number of times the step has been executed, incremented by 1 each time the step is executed
        state (Literal["pending", "completed", "failed", "in_progress"]): The current state of the step
        reflection (Optional[str]): The reflection after the step is completed
        messages (List[BaseMessage]): The conversation messages associated with executing this step
        template_id (Optional[str]): The template ID of the step
    """

    step_id: str
    task_id: str
    content: str
    step_progress_counter: int
    state: Literal["pending", "completed", "failed", "in_progress"] = "pending"
    reflection: Optional[str] = None
    messages: List[BaseMessage] = Field(default_factory=list)
    template_id: Optional[str] = None

    def dump(self) -> Dict:
        """Serialize the Step object into a dictionary.

        Returns:
            Dict: The serialized Step object
        """
        return {
            "step_id": self.step_id,
            "task_id": self.task_id,
            "content": self.content,
            "step_progress_counter": self.step_progress_counter,
            "state": self.state,
            "reflection": self.reflection,
            "messages": [msg.dump() for msg in self.messages],
            "template_id": self.template_id,
        }

    @classmethod
    def load(cls, data: Dict) -> "Step":
        """Deserialize a dictionary into a Step object.

        Args:
            data (Dict): The dictionary to deserialize

        Raises:
            ValueError: If task_id is not found
        """
        task_id = data.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for loading a step")

        return cls(
            step_id=data["step_id"],
            task_id=task_id,
            content=data["content"],
            step_progress_counter=data.get("step_progress_counter", 0),
            state=data.get("state", "pending"),
            reflection=data.get("reflection"),
            messages=[
                MessageClsDict[msg.get("type", "TextMessage")].load(msg)
                for msg in data.get("messages", [])
            ],
            template_id=data.get("template_id"),
        )


class Task(BaseModel):
    """
    Represents a task containing multiple steps.

    Attributes:
        task_id (str): The unique identifier inside the plan
        task_description (str): The description of the task
        task_summary (str): The summary of the task
    """

    task_id: str
    task_description: str
    task_summary: str = ""

    def dump(self) -> Dict:
        """Serialize the Task object into a dictionary.

        Returns:
            Dict: The serialized Task object
        """
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "task_summary": self.task_summary,
        }

    @classmethod
    def load(cls, data: Dict) -> "Task":
        """Deserialize a dictionary into a Task object.

        Args:
            data (Dict): The dictionary to deserialize

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            Task: _description_
        """
        task_id = data.get("task_id")
        if not task_id:
            raise ValueError("task_id is required for loading a task")

        task_description = data.get("task_description")
        if not task_description:
            raise ValueError("task_description is required for loading a task")

        task_summary = data.get("task_summary", "")

        return cls(
            task_id=task_id,
            task_description=task_description,
            task_summary=task_summary,
        )

    def update_summary(self, summary: str, overwrite: bool = False) -> None:
        """Update the task summary.

        Args:
            summary (str): The summary to update
            overwrite (bool): Whether to overwrite the existing summary
        """
        existing_summary = self.task_summary
        if self.task_summary:
            if overwrite:
                self.task_summary = summary
            else:
                self.task_summary = f"{existing_summary}\n{summary}"
        else:
            self.task_summary = summary


class Plan(BaseModel):
    """
    Represents a plan consisting of multiple tasks and steps.

    Attributes:
        plan_id (str): The unique identifier inside the plan
        plan_description (str): The description of the plan
        steps (OrderedDict[str, Step]): The steps in the plan
        tasks (OrderedDict[str, Task]): The tasks in the plan
        awaiting_confirmation (bool): Whether the plan is awaiting confirmation
        summary (Optional[str]): The summary of the plan
        shared_context (OrderedDict[str, str]): The shared context of the plan
        task_template_ids (Dict[int, Optional[str]]): The template IDs of the tasks
    """

    plan_id: str
    plan_description: str
    steps: OrderedDict[str, Step] = Field(default_factory=OrderedDict)
    tasks: OrderedDict[str, Task] = Field(default_factory=OrderedDict)
    awaiting_confirmation: bool = True
    summary: Optional[str] = None
    shared_context: OrderedDict[str, str] = Field(default_factory=OrderedDict)
    task_template_ids: Dict[int, Optional[str]] = Field(default_factory=dict)

    # ===== Step Management =====

    def get_current_step(self) -> Optional[Tuple[str, str]]:
        """Get the first pending or in-progress step."""
        for step_id, step in self.steps.items():
            if step.state in ["pending", "in_progress"]:
                return (step_id, step.content)
        return None

    def get_step_by_id(self, step_id: str) -> Step:
        """Get a step by its ID."""
        if step_id not in self.steps:
            raise ValueError(f"Step with id {step_id} not found")
        return self.steps[step_id]

    def add_message_to_step(self, step_id: str, message: BaseMessage) -> None:
        """Add a message to a specific step."""
        step = self.get_step_by_id(step_id)
        step.messages.append(message)

    def add_reflection_to_step(self, step_id: str, reflection: str) -> None:
        """Add a reflection to a specific step."""
        step = self.get_step_by_id(step_id)
        step.reflection = reflection

    def get_messages_by_step_id(self, step_id: str) -> List[BaseMessage]:
        """Get messages for a specific step."""
        step = self.get_step_by_id(step_id)
        return step.messages

    # ===== Plan Information =====

    def get_all_contents(self) -> List[str]:
        """Get content of all steps."""
        return [step.content for step in self.steps.values()]

    def get_all_states(self) -> OrderedDict[str, str]:
        """Get all step contents and their states."""
        return OrderedDict((step.content, step.state) for step in self.steps.values())

    def get_current_task_description(
        self,
    ) -> str:  # Renamed from get_current_group_description
        """Get the description of the current task."""
        current = self.get_current_step()
        if not current:
            return ""
        current_step_id, _ = current
        current_task_id = self.steps[current_step_id].task_id
        return self.tasks.get(
            current_task_id, Task(task_id="", task_description="")
        ).task_description

    # ===== Context and Summary Management =====

    def add_summary_to_plan(self, summary: str) -> None:
        """Add a summary to the plan."""
        self.summary = summary

    def get_summary(self) -> Optional[str]:
        """Get the plan summary."""
        return self.summary

    def update_shared_context(self, step_id: str, summary: str) -> None:
        """Update shared context with a task summary."""
        step = self.get_step_by_id(step_id)
        self.shared_context[step.task_id] = summary

    def get_shared_context(self) -> OrderedDict[str, str]:
        """Get the shared context."""
        return self.shared_context

    # ===== Serialization =====

    def dump(self) -> Dict:
        """Serialize the plan to a dictionary."""
        return {
            "plan_id": self.plan_id,
            "plan_description": self.plan_description,  # Renamed from task
            "steps": {step_id: step.dump() for step_id, step in self.steps.items()},
            "tasks": {
                task_id: task.dump() for task_id, task in self.tasks.items()
            },  # Renamed from groups
            "awaiting_confirmation": self.awaiting_confirmation,
            "summary": self.summary,
            "shared_context": dict(self.shared_context),
            "task_template_ids": self.task_template_ids,  # Renamed from group_template_ids
        }

    @classmethod
    def load(cls, data: Dict) -> "Plan":
        """Deserialize a plan from a dictionary."""
        # Handle backward compatibility
        plan_description = data.get("plan_description", data.get("task", ""))
        tasks_data = data.get("tasks", data.get("groups", {}))
        task_template_ids = data.get(
            "task_template_ids", data.get("group_template_ids", {})
        )

        return cls(
            plan_id=data["plan_id"],
            plan_description=plan_description,
            steps=OrderedDict(
                (step_id, Step.load(step)) for step_id, step in data["steps"].items()
            ),
            tasks=OrderedDict(
                (task_id, Task.load(task)) for task_id, task in tasks_data.items()
            ),
            awaiting_confirmation=data["awaiting_confirmation"],
            summary=data.get("summary"),
            shared_context=OrderedDict(data.get("shared_context", {})),
            task_template_ids=task_template_ids,
        )


class PlanHistory(BaseModel):
    """Container for plan history."""

    plan_history: List[Plan] = Field(default_factory=list)

    def append_plan(self, plan: Plan) -> None:
        """Add a plan to history."""
        self.plan_history.append(plan)

    def dump(self) -> Dict:
        """Serialize plan history."""
        return {"plan_history": [plan.dump() for plan in self.plan_history]}

    @classmethod
    def load(cls, data: Dict) -> "PlanHistory":
        """Deserialize plan history."""
        return cls(
            plan_history=[Plan.load(plan) for plan in data.get("plan_history", [])]
        )


class BasePlanManager(ABC):
    """Base class for plan management with common functionality.

    This abstract base class provides core functionality for managing plans,
    including creation, execution tracking, history management, and serialization.
    Subclasses must implement the _create_steps_from_tasks method to define
    workflow-specific step creation logic.

    Attributes:
        _plan_history: Container for all completed plans
        _current_plan: The currently active plan (if any)
        _human_feedback: Mapping of plan contents to human feedback
    """

    def __init__(self):
        """Initialize a new PlanManager instance."""
        self._plan_history = PlanHistory()
        self._current_plan: Optional[Plan] = None
        self._human_feedback: OrderedDict[Tuple[str, ...], str] = OrderedDict()

    # ===== Abstract Methods =====

    @abstractmethod
    def _create_steps_from_tasks(
        self, tasks: List[Dict]
    ) -> Tuple[Dict[str, Step], Dict[int, Optional[str]], Dict[str, Task]]:
        """Create steps from task data.

        This method must be implemented by subclasses to define how tasks
        are converted into executable steps based on the specific workflow.

        Args:
            tasks: List of task dictionaries containing at minimum:
                - name: Task name
                - description: Task description
                - Additional workflow-specific fields

        Returns:
            A tuple containing:
                - Dictionary mapping step IDs to Step objects
                - Dictionary mapping task indices to template IDs
                - Dictionary mapping task IDs to Task objects
        """

    # ===== Core Plan Lifecycle =====

    def new_plan(
        self,
        plan_description: str = "",
        model_response: str = "",
        human_feedback: str = "",
    ) -> None:
        """Create a new plan from model response.

        Creates a new plan by parsing the model response and converting tasks
        into executable steps. If a current plan exists, it can inherit the
        description and store feedback about the previous plan.

        Args:
            plan_description: Description of what the plan should accomplish.
                If empty and a current plan exists, uses current plan's description.
            model_response: JSON string containing task definitions. Expected format:
                {"tasks": [{"name": "...", "description": "...", ...}]}
            human_feedback: Feedback about the previous plan (if any).
                Will be stored for future reference.

        Raises:
            ValueError: If model_response is not valid JSON or has incorrect format
        """
        # Handle plan description and feedback
        if not plan_description and self._current_plan:
            plan_description = self._current_plan.plan_description

        if human_feedback and self._current_plan:
            self._human_feedback[tuple(self._current_plan.get_all_contents())] = (
                human_feedback
            )

        # Validate and parse response
        self._validate_model_response(model_response)
        tasks = safe_model_json_loads(model_response).get("tasks", [])

        # Create steps using subclass implementation
        steps, task_template_ids, tasks_dict = self._create_steps_from_tasks(tasks)

        # Create new plan
        self._current_plan = Plan(
            plan_id=f"plan_{uuid.uuid4()}",
            plan_description=plan_description,
            steps=OrderedDict(steps),
            tasks=OrderedDict(tasks_dict),
            awaiting_confirmation=True,
            task_template_ids=task_template_ids,
        )

    def confirm_plan(self) -> None:
        """Confirm the current plan, making it ready for execution.

        Changes the plan state from awaiting_confirmation to confirmed,
        allowing execution to begin.

        Raises:
            ValueError: If no current plan exists
        """
        self.current_plan = self._ensure_current_plan()
        self.current_plan.awaiting_confirmation = False

    def commit_plan(self) -> None:
        """Commit current plan to history and clear current plan.

        Moves the current plan to the history, typically done after
        plan execution is complete.

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        self._plan_history.append_plan(self._current_plan)
        self._current_plan = None

    def reset(self) -> None:
        """Reset plan manager to initial state.

        Clears all data including current plan, history, and feedback.
        """
        self._current_plan = None
        self._plan_history = PlanHistory()
        self._human_feedback.clear()

    # ===== Plan Queries =====

    def has_current_plan(self) -> bool:
        """Check if there is a current plan.

        Returns:
            True if a current plan exists, False otherwise
        """
        return self._current_plan is not None

    def is_plan_awaiting_confirmation(self) -> bool:
        """Check if current plan awaits confirmation.

        Returns:
            True if current plan exists and is awaiting confirmation, False otherwise
        """
        return (
            (self.has_current_plan() and self._current_plan.awaiting_confirmation)
            if self._current_plan
            else False
        )

    def get_current_plan_id(self) -> str:
        """Get current plan ID.

        Returns:
            The unique identifier of the current plan

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        return self._current_plan.plan_id

    def get_plan_description(self) -> str:
        """Get current plan description.

        Returns:
            The description of what the current plan accomplishes

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        return self._current_plan.plan_description

    def set_plan_description(self, description: str) -> None:
        """Set current plan description.

        Args:
            description: New description for the plan

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.plan_description = description

    def get_plan_summary(self) -> Optional[str]:
        """Get current plan summary.

        Returns:
            The plan summary if it exists, None otherwise
        """
        return self._current_plan.summary if self._current_plan else None

    def add_plan_summary(self, summary: str) -> None:
        """Add summary for current plan.

        Args:
            summary: Summary text describing plan outcomes

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.add_summary_to_plan(summary)

    # ===== Step Queries =====

    def get_total_steps(self) -> int:
        """Get total number of steps in current plan.

        Returns:
            Number of steps in current plan, or 0 if no plan exists
        """
        return len(self._current_plan.steps) if self._current_plan else 0

    def get_current_step(self) -> Optional[Tuple[str, str]]:
        """Get current step information.

        Finds the first step that is either pending or in_progress.

        Returns:
            Tuple of (step_id, step_content) if found, None otherwise
        """
        return self._current_plan.get_current_step() if self._current_plan else None

    def get_step_by_id(self, step_id: str) -> Step:
        """Get a specific step by ID.

        Args:
            step_id: The unique identifier of the step

        Returns:
            The Step object

        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        return self._current_plan.get_step_by_id(step_id)

    def get_all_step_contents(self) -> List[str]:
        """Get contents of all steps in current plan.

        Returns:
            List of step content strings, empty list if no plan exists
        """
        return self._current_plan.get_all_contents() if self._current_plan else []

    def get_all_step_states(self) -> OrderedDict[str, str]:
        """Get mapping of step content to state for all steps.

        Returns:
            OrderedDict mapping step content to state ('pending', 'completed', etc.),
            empty OrderedDict if no plan exists
        """
        return (
            self._current_plan.get_all_states() if self._current_plan else OrderedDict()
        )

    # ===== Step State Management =====

    def update_step_state(
        self,
        step_id: str,
        state: Literal["pending", "completed", "failed", "in_progress"],
    ) -> None:
        """Update the state of a specific step.

        Args:
            step_id: The unique identifier of the step
            state: The new state for the step

        Raises:
            ValueError: If no current plan exists
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.steps[step_id].state = state

    def get_step_progress_counter(self, step_id: str) -> int:
        """Get how many times a step has been executed.

        Args:
            step_id: The unique identifier of the step

        Returns:
            The number of times the step has been executed

        Raises:
            ValueError: If no current plan exists or step not found
        """
        return self.get_step_by_id(step_id).step_progress_counter

    def increment_step_progress_counter(self, step_id: str) -> None:
        """Increment the execution counter for a step.

        Args:
            step_id: The unique identifier of the step

        Raises:
            ValueError: If no current plan exists or step not found
        """
        step = self.get_step_by_id(step_id)
        step.step_progress_counter += 1

    def add_step_reflection(self, step_id: str, reflection: str) -> None:
        """Add reflection/result to a step.

        Reflections typically contain the outcome or learnings from
        executing a step.

        Args:
            step_id: The unique identifier of the step
            reflection: Text describing the step outcome

        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.add_reflection_to_step(step_id, reflection)

    # ===== Message Management =====

    def add_message_to_step(self, step_id: str, message: BaseMessage) -> None:
        """Add a message to a specific step.

        Messages track the conversation history during step execution.

        Args:
            step_id: The unique identifier of the step
            message: The message to add

        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.add_message_to_step(step_id, message)

    def get_step_messages(self, step_id: str) -> List[BaseMessage]:
        """Get all messages for a specific step.

        Args:
            step_id: The unique identifier of the step

        Returns:
            List of messages associated with the step

        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        return self._current_plan.get_messages_by_step_id(step_id)

    def get_current_step_messages(self) -> List[BaseMessage]:
        """Get messages for the current step.

        Returns:
            List of messages for current step, empty list if no current step
        """
        current = self.get_current_step()
        return self.get_step_messages(current[0]) if current else []

    def get_all_plan_messages(self) -> List[BaseMessage]:
        """Get all messages from all steps in current plan.

        Returns:
            List of all messages across all steps, empty list if no plan exists
        """
        if not self._current_plan:
            return []
        return [
            msg for step in self._current_plan.steps.values() for msg in step.messages
        ]

    # ===== Task Management =====

    def get_current_task_description(self) -> str:
        """Get description of the task containing current step.

        Returns:
            Task description for current step, empty string if no current step
        """
        return (
            self._current_plan.get_current_task_description()
            if self._current_plan
            else ""
        )

    def get_task_by_id(self, task_id: str) -> Task:
        """Get a specific task by ID.

        Args:
            task_id: The unique identifier of the task

        Returns:
            The Task object

        Raises:
            ValueError: If no current plan exists or task not found
        """
        self._current_plan = self._ensure_current_plan()
        if task_id not in self._current_plan.tasks:
            raise ValueError(f"Task with id {task_id} not found")
        return self._current_plan.tasks[task_id]

    def get_all_task_descriptions(self) -> List[str]:
        """Get descriptions of all tasks in current plan.

        Returns:
            List of task descriptions, empty list if no plan exists
        """
        if not self._current_plan:
            return []
        return [task.task_description for task in self._current_plan.tasks.values()]

    def add_task_summary(
        self, step_id: str, summary: str, overwrite: bool = False
    ) -> None:
        """Add summary to the task containing the given step.

        Task summaries accumulate - new summaries are appended to existing ones.

        Args:
            step_id: ID of a step within the task
            summary: Summary text to add
            overwrite (bool): Whether to overwrite the existing summary
        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        task_id = self.get_step_by_id(step_id).task_id
        self._current_plan.tasks[task_id].update_summary(summary, overwrite)

    def get_task_summaries(self) -> Dict[str, str]:
        """Get all task summaries (only non-empty ones).

        Returns:
            Dictionary mapping task descriptions to their summaries
        """
        if not self._current_plan:
            return {}
        return {
            task.task_description: task.task_summary
            for task in self._current_plan.tasks.values()
            if task.task_summary
        }

    def get_task_summaries_text(self) -> str:
        """Get formatted text of all task summaries.

        Returns:
            Multi-line string with format "task_description: summary",
            empty string if no summaries exist
        """
        summaries = self.get_task_summaries()
        return "\n".join(f"{desc}: {summary}" for desc, summary in summaries.items())

    def get_task_summary_by_step_id(self, step_id: str) -> str:
        """Get the summary of the task containing the given step.

        Returns:
            The summary of the task containing the given step
        """
        return self.get_task_by_id(self.get_step_by_id(step_id).task_id).task_summary

    # ===== Shared Context Management =====

    def update_shared_context(self, step_id: str, context: str) -> None:
        """Update shared context for the task containing the given step.

        Shared context allows tasks to share information with subsequent tasks.

        Args:
            step_id: ID of a step within the task
            context: Context information to store

        Raises:
            ValueError: If no current plan exists or step not found
        """
        self._current_plan = self._ensure_current_plan()
        self._current_plan.update_shared_context(step_id, context)

    def get_shared_context(self) -> OrderedDict[str, str]:
        """Get all shared context.

        Returns:
            OrderedDict mapping task IDs to their shared context,
            empty OrderedDict if no plan exists
        """
        return (
            self._current_plan.get_shared_context()
            if self._current_plan
            else OrderedDict()
        )

    # ===== History Management =====

    def get_plan_count(self) -> int:
        """Get total number of plans in history.

        Returns:
            Number of completed plans in history
        """
        return len(self._plan_history.plan_history)

    def get_plan_history(self) -> PlanHistory:
        """Get the plan history object.

        Returns:
            The PlanHistory object containing all completed plans
        """
        return self._plan_history

    def get_plan_history_summary(self) -> str:
        """Get formatted summary of plan history.

        Provides a concise overview of all completed plans with their
        descriptions and summaries.

        Returns:
            Formatted string with plan summaries, or message if no history exists
        """
        if not self.get_plan_count():
            return "No plan history available."

        summaries = []
        for i, plan in enumerate(self._plan_history.plan_history, 1):
            summary = f"Plan {i}: {plan.plan_description}"
            if plan.summary:
                summary += f"\nSummary: {plan.summary}"
            summaries.append(summary)

        return "\n\n".join(summaries)

    def get_detailed_plan_history(self) -> str:
        """Get detailed history with all steps and results.

        Provides complete information about all plans including individual
        steps, their states, messages, and reflections.

        Returns:
            Detailed formatted string of plan history, or message if no history exists
        """
        if not self.get_plan_count():
            return "No plan history available."

        lines = []
        for plan in self._plan_history.plan_history:
            lines.append(f"Plan: {plan.plan_description}")
            for step in plan.steps.values():
                lines.append(f"  Step: {step.content}")
                lines.append(f"  State: {step.state}")
                if step.messages:
                    lines.append(f"  Last Message: {step.messages[-1]}")
                if step.reflection:
                    lines.append(f"  Result: {step.reflection}")
                lines.append("")  # Empty line between steps

        return "\n".join(lines)

    # ===== Human Feedback Management =====

    def get_human_feedback_for_plan(
        self, plan_contents: Tuple[str, ...]
    ) -> Optional[str]:
        """Get human feedback for a specific plan configuration.

        Args:
            plan_contents: Tuple of step contents identifying the plan

        Returns:
            The feedback string if found, None otherwise
        """
        return self._human_feedback.get(plan_contents)

    def get_all_human_feedback(self) -> Dict[Tuple[str, ...], str]:
        """Get all stored human feedback.

        Returns:
            Dictionary mapping plan content tuples to feedback strings
        """
        return dict(self._human_feedback)

    # ===== Serialization =====

    def dump(self) -> Dict:
        """Serialize plan manager state.

        Converts the entire plan manager state to a dictionary that can
        be saved and later restored.

        Returns:
            Dictionary containing all plan manager state
        """
        return {
            "plan_history": self._plan_history.dump(),
            "current_plan": self._current_plan.dump() if self._current_plan else None,
            "human_feedback": {",".join(k): v for k, v in self._human_feedback.items()},
        }

    @classmethod
    def load(cls, data: Dict) -> "BasePlanManager":
        """Deserialize plan manager from saved state.

        Restores a plan manager instance from previously saved state.

        Args:
            data: Dictionary containing serialized plan manager state

        Returns:
            New plan manager instance with restored state
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

    # ===== Private Helper Methods =====

    def _ensure_current_plan(self) -> Plan:
        """Ensure current plan exists, raise error if not.

        Raises:
            ValueError: If no current plan exists
        """
        if not self._current_plan:
            raise ValueError("No current plan exists")
        else:
            return self._current_plan

    def _validate_model_response(self, model_response: str) -> None:
        """Validate the format of model response.

        Checks that the response is valid JSON with the expected structure.
        Supports both "tasks" and "groups" keys for backward compatibility.

        Args:
            model_response: JSON string to validate

        Raises:
            ValueError: If response is not valid JSON or has incorrect structure
        """
        try:
            data = safe_model_json_loads(model_response)
            if not isinstance(data, dict):
                raise ValueError("Model response must be a JSON object")

            # Support both "tasks" and "groups" for backward compatibility
            if "tasks" not in data and "groups" not in data:
                raise ValueError(
                    "Model response must contain a 'tasks' or 'groups' key"
                )

            tasks = data.get("tasks", data.get("groups", []))
            if not isinstance(tasks, list):
                raise ValueError("The 'tasks' value must be a list")

            for i, task in enumerate(tasks):
                if not isinstance(task, dict):
                    raise ValueError(f"Task {i} must be a dictionary")
                if "name" not in task or "description" not in task:
                    raise ValueError(
                        f"Task {i} must contain 'name' and 'description' keys"
                    )
        except ModelJSONDecodeError as e:
            raise ModelJSONDecodeError(
                f"Model response is not valid JSON: {str(e)}", model_response, e.pos
            )
