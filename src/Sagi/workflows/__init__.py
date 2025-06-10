from .analyzing.analyzing import AnalyzingWorkflow
from .planning.planning import PlanningWorkflow

workflowName = PlanningWorkflow | AnalyzingWorkflow

__all__ = [
    "PlanningWorkflow",
    "AnalyzingWorkflow",
    "workflowName",
]
