import pytest
import time
import os
import asyncio

from pathlib import Path

from autogen_core import CancellationToken
from autogen_agentchat.base import Response

from autogen_agentchat.messages import (
    TextMessage,
)

from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)

@pytest.mark.asyncio
async def test_add_install_dependencies():

    install_dependencies_scripts = ["""```sh
pip install fpdf
pip install pdf2image
```""",
"""```sh
pip install reportlab
pip install pdfkit
pip install pikepdf
pip install img2pdf
```"""]

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
    assert await code_executor.is_running() is True, "The code executor should be running after start()"

    # Try to add dependencies
    for script in install_dependencies_scripts:
        async for result in docker_executor_agent.on_messages_stream(
            messages=[TextMessage(content=script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if isinstance(result, Response):
                assert result.chat_message.source == "stream_code_executor_agent"

    assert (len(code_executor.docker_installed_dependencies)==2)
    assert await code_executor.is_running() is True, "The code executor should still be running after adding dependencies"

    # in the on_stream_messages_stream, we have assert(result.exit_code == 0) to ensure these libraries are installed
    async for result in docker_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"

    await code_executor.countdown(3)
    await code_executor.stop_countdown()

    await asyncio.sleep(15) # it takes times to stop the Docker container (10.xx secs)
    assert await code_executor.is_running() is True, "The code executor should still be running after cancelling countdown"

    await code_executor.countdown(3)
    await asyncio.sleep(15) # it takes times to stop the Docker container (10.xx secs)
    assert await code_executor.is_running() is False, "The code executor should not be running after countdown finishes"

    await code_executor.resume_docker_container()
    assert await code_executor.is_running() is True, "The code executor should be running after resuming the container"

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
    print(f"Stop duration: {stop_duration} seconds") # around 10.xx seconds


    assert await code_executor.is_running() is False, "The code executor should not be running after stop()"



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