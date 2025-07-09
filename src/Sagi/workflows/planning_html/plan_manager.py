from typing import Dict, List, Optional, Tuple

from Sagi.workflows.base_plan_manager import BasePlanManager, Step, Task


class PlanManager(BasePlanManager):
    def _create_steps_from_tasks(
        self, tasks: List[Dict]
    ) -> Tuple[Dict[str, Step], Dict[int, Optional[str]], Dict[str, Task]]:
        """Create steps focused on data collection and HTML generation."""
        steps = {}
        task_template_ids = {}
        tasks_dict = {}
        step_id = 0

        for task_id, task in enumerate(tasks):
            # Create task
            tasks_dict[f"task_{task_id}"] = Task(
                task_id=f"task_{task_id}", task_description=task["description"]
            )

            # Add data collection step if exists
            if task.get("data_collection_task"):
                steps[f"step_{step_id}"] = Step(
                    step_id=f"step_{step_id}",
                    task_id=f"task_{task_id}",
                    content=task["data_collection_task"],
                    template_id=task.get("template_id"),
                    step_progress_counter=0,
                )
                step_id += 1

            task_template_ids[task_id] = task.get("template_id")

        # Add final HTML generation task
        final_task_id = len(tasks)
        tasks_dict[f"task_{final_task_id}"] = Task(
            task_id=f"task_{final_task_id}",
            task_description="Generate a html page based on collected information",
        )

        steps[f"step_{step_id}"] = Step(
            step_id=f"step_{step_id}",
            task_id=f"task_{final_task_id}",
            content="Generate a html page based on collected information",
            step_progress_counter=0,
        )

        return steps, task_template_ids, tasks_dict
