import asyncio
import json
import os
from pathlib import Path

import pytest
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    TextMessage,
)
from autogen_core import CancellationToken

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)


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
import pdf2image
import fpdf
import reportlab
import pdfkit
import pikepdf
import img2pdf
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
        countdown_timer=4,
    )

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

    await code_executor.stop_countdown()
    await asyncio.sleep(16)  # it takes times to stop the Docker container (10.xx secs)
    assert (
        await code_executor.is_running() is True
    ), "The code executor should still be running after cancelling countdown"

    # in the on_stream_messages_stream, we have assert(result.exit_code == 0) to ensure these libraries are installed
    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"

    await asyncio.sleep(16)
    assert (
        await code_executor.is_running() is False
    ), "The code executor should not be running after stop()"


@pytest.mark.asyncio
async def test_save_state_dependencies():

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
import pdf2image
import fpdf
import reportlab
import pdfkit
import pikepdf
import img2pdf
```"""

    work_dir = Path("coding_files")
    stream_code_executor = StreamDockerCommandLineCodeExecutor(
        work_dir=work_dir,
        bind_dir=(
            os.getenv("HOST_PATH") + "/" + str(work_dir)
            if os.getenv("ENVIRONMENT") == "docker"
            else work_dir
        ),
    )

    docker_executor_agent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_code_executor,
        countdown_timer=2,
    )

    assert (
        len(stream_code_executor.docker_installed_dependencies) == 0
    ), "There should be no installed dependencies at the start"

    # Try to add dependencies
    for script in install_dependencies_scripts:
        async for result in docker_executor_agent.on_messages_stream(
            messages=[TextMessage(content=script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(result, Response):
                assert result.chat_message.source == "stream_code_executor_agent"

    assert (
        await stream_code_executor.is_running() is True
    ), "The code executor should still be running after adding dependencies"

    assert (
        len(stream_code_executor.docker_installed_dependencies) == 2
    ), "There should be 2 installed dependencies"

    saved_state = await docker_executor_agent.save_state()

    state_file_path = Path("state_backup.json")
    assert (
        state_file_path.exists()
    ), "The state file should exist after saving the state"

    with open(state_file_path, "w") as f:
        json.dump(saved_state, f)

    await asyncio.sleep(20)  # wait for the container to stop
    assert (
        await stream_code_executor.is_running() is False
    ), "The code executor should not be running after the countdown"
    stream_code_executor.docker_installed_dependencies = []

    with open(state_file_path, "r") as f:
        loaded_state = json.load(f)

    await docker_executor_agent.load_state(loaded_state)
    assert (
        len(stream_code_executor.docker_installed_dependencies) == 2
    ), "load_state() should update the installed dependencies"

    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"


"""
Testing manually 1
-> input prompt1: give me the pdf file with the text "sad"

    case 1: the next instruction is immediately after the first one (the container is still running)
        expectation: the container must still have the installed dependencies
    case 2: the next instruction is after a countdown (the container is stopped)
        expectation: the container can start again, and install all the previous dependencies again

-> input prompt2: give me the pdf file with the text "happy"
    expectation: the program can run the code without any dependencies issues (installation of the previous dependencies shouldn't happen in this stage, should be at the starting of the container)
"""
