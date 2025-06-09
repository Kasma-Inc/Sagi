from .planning.planning import PlanningWorkflow
from .analyzing.analyzing import AnalyzingWorkflow
workflowName = PlanningWorkflow | AnalyzingWorkflow

__all__ = [
    "PlanningWorkflow",
    "AnalyzingWorkflow",
    "workflowName",
]