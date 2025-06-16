import pytest

from Sagi.workflows.general.general_chat import GeneralChatWorkflow
from Sagi.workflows.planning.planning import PlanningWorkflow

DEFAULT_TEAM_CONFIG_PATH = "src/Sagi/workflows/team.toml"
DEFAULT_PLANNING_CONFIG_PATH = "src/Sagi/workflows/planning/planning.toml"
DEFAULT_GENERAL_CONFIG_PATH = "src/Sagi/workflows/general/general.toml"


@pytest.mark.asyncio
async def test_workflow_reset():
    workflow = await PlanningWorkflow.create(
        DEFAULT_PLANNING_CONFIG_PATH,
        DEFAULT_TEAM_CONFIG_PATH,
    )
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
    workflow = await PlanningWorkflow.create(
        DEFAULT_PLANNING_CONFIG_PATH,
        DEFAULT_TEAM_CONFIG_PATH,
    )
    async for _ in workflow.run_workflow("Hello, how are you?"):
        pass
    state = await workflow.team.save_state()
    await workflow.team.reset()
    await workflow.team.load_state(state)
    state1 = await workflow.team.save_state()
    assert state1 == state


@pytest.mark.asyncio
async def test_general_chat_workflow_reset():
    workflow = await GeneralChatWorkflow.create(
        DEFAULT_GENERAL_CONFIG_PATH,
        web_search=True,
    )
    async for _ in workflow.run_workflow("Hello, what's up?"):
        pass
    await workflow.team.reset()
    state = await workflow.team.save_state()
    # Check that message buffers are empty after reset
    for k, v in state["agent_states"].items():
        if v["type"] == "ChatAgentContainer":
            assert v["agent_state"]["message_buffer"] == []
            assert v["agent_state"]["agent_state"]["llm_context"]["messages"] == []
    await workflow.cleanup()


@pytest.mark.asyncio
async def test_general_chat_workflow_load_state():
    workflow = await GeneralChatWorkflow.create(
        DEFAULT_GENERAL_CONFIG_PATH,
        web_search=True,
    )
    async for _ in workflow.run_workflow("Hello, what's up?"):
        pass
    state = await workflow.team.save_state()
    await workflow.team.reset()
    await workflow.team.load_state(state)
    state1 = await workflow.team.save_state()
    assert state1 == state
    await workflow.cleanup()
