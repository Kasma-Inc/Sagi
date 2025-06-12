from .general.general_chat import GeneralChatWorkflow
from .planning.planning import PlanningWorkflow

workflowName = PlanningWorkflow | GeneralChatWorkflow

__all__ = [
    "PlanningWorkflow",
    "GeneralChatWorkflow",
    "workflowName",
]
