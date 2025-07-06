from typing import Dict, List, Optional, Tuple

from autogen_agentchat.messages import (
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

from Sagi.workflows.base_plan_manager import BasePlanManager, Step, Task

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


class PlanManager(BasePlanManager):
    def _create_steps_from_tasks(
        self, tasks: List[Dict]
    ) -> Tuple[Dict[str, Step], Dict[int, Optional[str]], Dict[str, Task]]:
        """Create steps for data collection and code execution."""
        steps = {}
        task_template_ids = {}
        tasks_dict = {}
        step_id = 0

        for task_id, task in enumerate(tasks):
            # Create task
            tasks_dict[f"task_{task_id}"] = Task(
                task_id=f"task_{task_id}", task_description=task["description"]
            )

            tasks_added = False
            template_id = task.get("template_id")

            # Add data collection task
            if task.get("data_collection_task"):
                steps[f"step_{step_id}"] = Step(
                    step_id=f"step_{step_id}",
                    task_id=f"task_{task_id}",
                    content=task["data_collection_task"],
                    template_id=template_id,
                    step_progress_counter=0,
                )
                step_id += 1
                tasks_added = True

            # Add code executor task
            if task.get("code_executor_task") and task["code_executor_task"] != "N/A":
                steps[f"step_{step_id}"] = Step(
                    step_id=f"step_{step_id}",
                    task_id=f"task_{task_id}",
                    content=task["code_executor_task"],
                    template_id=template_id,
                    step_progress_counter=0,
                )
                step_id += 1
                tasks_added = True

            # Add general task if no specific tasks
            if not tasks_added:
                steps[f"step_{step_id}"] = Step(
                    step_id=f"step_{step_id}",
                    task_id=f"task_{task_id}",
                    content=f"{task['name']}: {task['description']}",
                    template_id=template_id,
                    step_progress_counter=0,
                )
                step_id += 1

            task_template_ids[task_id] = template_id

        return steps, task_template_ids, tasks_dict
