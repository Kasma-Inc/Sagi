import json
from collections import OrderedDict

import pytest
from autogen_agentchat.messages import TextMessage

from Sagi.workflows.plan_manager import Plan, PlanHistory, PlanManager, Step


def test_step_creation_and_serialization():
    # Test Step creation
    step = Step(
        step_id="step1",
        group_id="group_1",
        content="Test step content",
        step_progress_counter=0,
        state="pending",
        reflection=None,
        messages=[],
    )

    assert step.step_id == "step1"
    assert step.content == "Test step content"
    assert step.step_progress_counter == 0
    assert step.state == "pending"
    assert step.reflection is None
    assert step.messages == []

    # Test Step serialization
    step_dict = step.dump()
    assert step_dict["step_id"] == "step1"
    assert step_dict["content"] == "Test step content"
    assert step_dict["step_progress_counter"] == 0
    assert step_dict["state"] == "pending"
    assert step_dict["reflection"] is None
    assert step_dict["messages"] == []

    # Test Step deserialization
    loaded_step = Step.load(step_dict)
    assert loaded_step.step_id == step.step_id
    assert loaded_step.content == step.content
    assert loaded_step.step_progress_counter == step.step_progress_counter
    assert loaded_step.state == step.state
    assert loaded_step.reflection == step.reflection
    assert loaded_step.messages == step.messages


def test_plan_creation_and_operations():
    # Create test steps
    step1 = Step(
        step_id="step1",
        group_id="group_1",
        content="First step",
        step_progress_counter=0,
        state="pending",
        reflection=None,
        messages=[],
    )

    step2 = Step(
        step_id="step2",
        group_id="group_2",
        content="Second step",
        step_progress_counter=0,
        state="pending",
        reflection=None,
        messages=[],
    )

    # Create plan with steps
    steps = OrderedDict([("step1", step1), ("step2", step2)])
    plan = Plan(
        plan_id="plan1",
        task="Test task",
        steps=steps,
        awaiting_confirmation=False,
        summary=None,
    )

    # Test basic attributes
    assert plan.plan_id == "plan1"
    assert plan.task == "Test task"
    assert len(plan.steps) == 2
    assert not plan.awaiting_confirmation
    assert plan.summary is None

    # Test get_current_step
    current_step = plan.get_current_step()
    assert current_step == ("step1", "First step")

    # Test get_all_contents
    contents = plan.get_all_contents()
    assert contents == ["First step", "Second step"]

    # Test get_all_states
    states = plan.get_all_states()
    assert states == OrderedDict(
        [("First step", "pending"), ("Second step", "pending")]
    )

    # Test add_message_to_step
    message = TextMessage(content="Test message", source="Test source")
    plan.add_message_to_step("step1", message)
    assert len(plan.steps["step1"].messages) == 1
    assert plan.steps["step1"].messages[0].content == "Test message"

    # Test add_reflection_to_step
    plan.add_reflection_to_step("step1", "Test reflection")
    assert plan.steps["step1"].reflection == "Test reflection"

    # Test add_summary_to_plan
    plan.add_summary_to_plan("Test summary")
    assert plan.summary == "Test summary"


def test_plan_history():
    # Create test plan
    step = Step(
        step_id="step1",
        group_id="group_1",
        content="Test step",
        step_progress_counter=0,
        state="pending",
        reflection=None,
        messages=[],
    )

    plan = Plan(
        plan_id="plan1",
        task="Test task",
        steps=OrderedDict([("step1", step)]),
        awaiting_confirmation=False,
        summary=None,
    )

    # Create plan history
    history = PlanHistory(plan_history=[])

    # Test append_plan
    history.append_plan(plan)
    assert len(history.plan_history) == 1
    assert history.plan_history[0].plan_id == "plan1"

    # Test serialization
    history_dict = history.dump()
    assert "plan_history" in history_dict
    assert len(history_dict["plan_history"]) == 1

    # Test deserialization
    loaded_history = PlanHistory.load(history_dict)
    assert len(loaded_history.plan_history) == 1
    assert loaded_history.plan_history[0].plan_id == "plan1"


def test_plan_manager():
    # Initialize PlanManager
    manager = PlanManager()

    # Test initial state
    assert manager.get_plan_count() == 0
    assert not manager.is_plan_awaiting_confirmation()
    assert manager.get_current_step() is None

    # Test new_plan
    task = "Test task"
    # when model_response is not a valid json string, manager.new_plan should raise a ValueError
    model_response = "1. First step\n2. Second step"
    with pytest.raises(ValueError):
        manager.new_plan(task, model_response)

    model_response = json.dumps(
        {
            "steps": [
                {
                    "name": "Step 1",
                    "description": "First step",
                    "data_collection_task": "Collect data",
                    "code_executor_task": "Execute code",
                },
                {
                    "name": "Step 2",
                    "description": "Second step",
                    "data_collection_task": "Collect more data",
                    "code_executor_task": "Execute more code",
                },
            ]
        }
    )
    manager.new_plan(task, model_response)

    assert manager.get_task() == task
    assert len(manager.get_current_plan_contents()) == 4
    assert manager.get_current_step() == (
        "step_0",
        "data collection task for Step 1: Collect data",
    )

    # Test set_step_state
    manager.set_step_state("step_0", "in_progress")
    states = manager.get_current_plan_state()
    assert list(states.values())[0] == "in_progress"

    # Test add_message_to_step
    message = TextMessage(content="Test message", source="Test source")
    manager.add_message_to_step("step_0", message)
    messages = manager.get_messages_by_step_id("step_0")
    assert len(messages) == 1
    assert messages[0].content == "Test message"

    # Test add_reflection_to_step
    manager.add_reflection_to_step("step_1", "Test reflection")
    step = manager._current_plan.steps["step_1"]
    assert step.reflection == "Test reflection"

    # Test increment_step_counter
    manager.increment_step_counter("step_1")
    assert manager.get_step_progress_counter("step_1") == 1

    # Test commit_plan
    manager.commit_plan()
    assert manager.get_plan_count() == 1
    assert manager._current_plan is None

    # Test reset
    manager.reset()
    assert manager.get_plan_count() == 0
    assert manager._current_plan is None
    assert manager._plan_history.plan_history == []


def test_plan_manager_serialization():
    # Initialize PlanManager and create a plan
    manager = PlanManager()
    task = "Test task"
    model_response = json.dumps(
        {
            "steps": [
                {
                    "name": "Step 1",
                    "description": "First step",
                    "data_collection_task": "Collect data",
                    "code_executor_task": "Execute code",
                },
                {
                    "name": "Step 2",
                    "description": "Second step",
                    "data_collection_task": "Collect more data",
                    "code_executor_task": "Execute more code",
                },
            ]
        }
    )
    manager.new_plan(task, model_response)

    # Test serialization
    manager_dict = manager.dump()
    assert "current_plan" in manager_dict
    assert "plan_history" in manager_dict

    # Test deserialization
    loaded_manager = PlanManager.load(manager_dict)
    assert loaded_manager.get_plan_count() == 0
    assert loaded_manager.get_task() == task
    assert len(loaded_manager.get_current_plan_contents()) == 4

def test_group_message_retrieval():
    manager = PlanManager()
    task = "Test task"
    model_response = json.dumps(
        {
            "steps": [
                {
                    "name": "Step 1",
                    "description": "First step",
                    "data_collection_task": "Collect data",
                    "code_executor_task": "N/A",
                },
                {
                    "name": "Step 2",
                    "description": "Second step",
                    "data_collection_task": "Collect more data",
                    "code_executor_task": "N/A",
                },
            ]
        }
    )
    manager.new_plan(task, model_response)

    # Add messages to both steps (group_0 and group_1)
    manager.add_message_to_step("step_0", TextMessage(content="msg1", source="tester"))
    manager.add_message_to_step("step_1", TextMessage(content="msg2", source="tester"))

    # Retrieve messages of the current group (initially group_0)
    msgs = manager.get_messages_of_current_group()
    assert len(msgs) == 1
    assert msgs[0].content == "msg1"
