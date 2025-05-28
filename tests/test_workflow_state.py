import pytest

from Sagi.workflows.planning import PlanningWorkflow


@pytest.mark.asyncio
async def test_workflow_reset():
    workflow = await PlanningWorkflow.create("src/Sagi/workflows/planning.toml")
    async for _ in workflow.run_workflow("Hello, how are you?"):
        pass
    state0 = await workflow.team.save_state()
    for k, v in state0["agent_states"].items():
        if v["type"] == "PlanningOrchestratorState":
            plan_manager_state = v["plan_manager_state"]
            assert plan_manager_state["current_plan"] != None
    await workflow.team.reset()
    state = await workflow.team.save_state()
    for k, v in state["agent_states"].items():
        if v["type"] == "ChatAgentContainer":
            assert v["agent_state"]["message_buffer"] == []
            assert v["agent_state"]["agent_state"]["llm_context"]["messages"] == []
        elif v["type"] == "PlanningOrchestratorState":
            plan_manager_state = v["plan_manager_state"]
            assert plan_manager_state["plan_history"]["plan_history"] == []
            assert plan_manager_state["current_plan"] == None
            assert plan_manager_state["human_feedback"] == {}


@pytest.mark.asyncio
async def test_workflow_load_state():
    workflow = await PlanningWorkflow.create("src/Sagi/workflows/planning.toml")
    async for _ in workflow.run_workflow("Hello, how are you?"):
        pass
    state = await workflow.team.save_state()
    await workflow.team.reset()
    await workflow.team.load_state(state)
    state1 = await workflow.team.save_state()
    assert state1 == state
