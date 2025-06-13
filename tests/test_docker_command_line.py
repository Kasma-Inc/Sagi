import asyncio
import json
import os
import time
from pathlib import Path

import pytest
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    TextMessage,
)
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from dotenv import load_dotenv

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from Sagi.workflows.planning import PlanningWorkflow


@pytest.mark.asyncio
async def test_add_install_dependencies():

    install_dependencies_scripts = [
        """```sh
pip install fpdf
pip install pdf2image
```""",
        """```sh
pip install reportlab
pip install pdfkit
pip install pikepdf
pip install img2pdf
```""",
    ]

    python_script = """```py
import fpdf
import pdf2image
import reportlab
```"""

    work_dir = Path("coding_files")
    code_executor = StreamDockerCommandLineCodeExecutor(
        work_dir=work_dir,
        bind_dir=(
            os.getenv("HOST_PATH") + "/" + str(work_dir)
            if os.getenv("ENVIRONMENT") == "docker"
            else work_dir
        ),
    )

    docker_executor_agent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=code_executor,
    )

    await code_executor.start()
    assert (
        await code_executor.is_running() is True
    ), "The code executor should be running after start()"

    # Try to add dependencies
    for script in install_dependencies_scripts:
        async for result in docker_executor_agent.on_messages_stream(
            messages=[TextMessage(content=script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(result, Response):
                assert result.chat_message.source == "stream_code_executor_agent"

    assert len(code_executor.docker_installed_dependencies) == 2
    assert (
        await code_executor.is_running() is True
    ), "The code executor should still be running after adding dependencies"

    # in the on_stream_messages_stream, we have assert(result.exit_code == 0) to ensure these libraries are installed
    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"

    await code_executor.countdown(3)
    await code_executor.stop_countdown()

    await asyncio.sleep(15)  # it takes times to stop the Docker container (10.xx secs)
    assert (
        await code_executor.is_running() is True
    ), "The code executor should still be running after cancelling countdown"

    await code_executor.countdown(3)
    await asyncio.sleep(15)  # it takes times to stop the Docker container (10.xx secs)
    assert (
        await code_executor.is_running() is False
    ), "The code executor should not be running after countdown finishes"

    await code_executor.resume_docker_container()
    assert (
        await code_executor.is_running() is True
    ), "The code executor should be running after resuming the container"

    # in the on_stream_messages_stream, we have assert(result.exit_code == 0) to ensure these libraries are installed
    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"

    start_time = time.time()
    await code_executor.stop()
    stop_duration = time.time() - start_time
    print(f"Stop duration: {stop_duration} seconds")  # around 10.xx seconds

    assert (
        await code_executor.is_running() is False
    ), "The code executor should not be running after stop()"


@pytest.mark.asyncio
async def test_save_state_dependencies():
    config_path = "src/Sagi/workflows/planning.toml"
    team_config_path = "src/Sagi/workflows/team.toml"
    template_work_dir = None
    mode = "deep_research"
    language = "en"

    install_dependencies_scripts = [
        """```sh
pip install fpdf
pip install pdf2image
```""",
        """```sh
pip install reportlab
pip install pdfkit
pip install pikepdf
pip install img2pdf
```""",
    ]

    python_script = """```py
import fpdf
import pdf2image
import reportlab
```"""

    dotenv_path = Path("/chatbot/.env")
    assert dotenv_path.exists(), "The .env file should exist at the specified path"
    load_dotenv(dotenv_path=dotenv_path, override=True)

    workflow, stream_code_executor = await PlanningWorkflow.create(
        config_path,
        team_config_path,
        template_work_dir=template_work_dir,
        mode=mode,
        language=language,
    )

    await stream_code_executor.start()

    docker_executor_agent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_code_executor,
    )

    # Try to add dependencies
    for script in install_dependencies_scripts:
        async for result in docker_executor_agent.on_messages_stream(
            messages=[TextMessage(content=script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(result, Response):
                assert result.chat_message.source == "stream_code_executor_agent"

    assert (
        len(stream_code_executor.docker_installed_dependencies) == 2
    ), "There should be 2 installed dependencies"

    saved_state = await workflow.team.save_state()

    state_file_path = Path("state_backup.json")
    assert (
        state_file_path.exists()
    ), "The state file should exist after saving the state"

    with open(state_file_path, "w") as f:
        json.dump(saved_state, f)

    await stream_code_executor.stop()
    await asyncio.sleep(13)  # wait for the container to stop
    stream_code_executor.docker_installed_dependencies = []

    with open(state_file_path, "r") as f:
        loaded_state_json = json.load(f)

    loaded_state = [
        CodeBlock(code=dep.get("code", ""), language=dep.get("language", ""))
        for dep in loaded_state_json["dependencies"]
    ]

    stream_code_executor.docker_installed_dependencies = loaded_state
    print(f"Loaded dependencies: {stream_code_executor.docker_installed_dependencies}")
    assert (
        len(stream_code_executor.docker_installed_dependencies) == 2
    ), "The loaded state should contain the installed dependencies"

    await stream_code_executor.resume_docker_container()

    # in the on_stream_messages_stream, we have assert(result.exit_code == 0) to ensure these libraries are installed
    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"

    await stream_code_executor.stop()


"""
Testing manually 1
-> input prompt1: give me the pdf file with the text "sad"

    case 1: the next instruction is immediately after the first one (the container is still running)
        expectation: the container must still have the installed dependencies
    case 2: the next instruction is after a countdown (the container is stopped)
        expectation: the container can start again, and install all the previous dependencies again

-> input prompt2: give me the pdf file with the text "happy"
    expectation: the program can run the code without any dependencies issues (installation of the previous dependencies shouldn't happen in this stage, should be at the starting of the container)


Testing manually 2
-> input prompt: any prompt

-> try terminate the program with Ctrl+C at some point
    expectation: the program should be able to stop gracefully, and the container should be stopped -> "Starting graceful shutdown..." should be printed
"""
