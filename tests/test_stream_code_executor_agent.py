import os

import pytest
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    CodeGenerationEvent,
    TextMessage,
)
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
    CustomCommandLineCodeResult,
)
from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    CodeExecutionEvent,
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from Sagi.tools.stream_code_executor.stream_local_command_line_code_executor import (
    StreamLocalCommandLineCodeExecutor,
)

load_dotenv(override=True)


@pytest.mark.asyncio
async def test_local_command_line_code_executor_sh():
    sh_script = """```sh
echo "Hello World"
sleep 1
echo "Hello World"
sleep 1
echo "Hello World"
```"""

    stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor(
        work_dir="coding_files",
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_local_command_line_code_executor,
    )
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=sh_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )

        elif isinstance(result, CodeFileMessage):
            assert result.code_file.endswith(".sh")
        elif isinstance(result, CustomCommandLineCodeResult):
            assert result.command.startswith("sh")


@pytest.mark.asyncio
async def test_local_command_line_code_executor_python():
    python_script = """```py
import time
print("Hello World")
time.sleep(1)
print("Hello World")
time.sleep(1)
print("Hello World")
```"""
    stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor(
        work_dir="coding_files",
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_local_command_line_code_executor,
    )
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )

        elif isinstance(result, CodeFileMessage):
            assert result.code_file.endswith(".py")
        elif isinstance(result, CustomCommandLineCodeResult):
            assert result.command.split(" ")[0].endswith(("python", "python3"))


@pytest.mark.asyncio
async def test_local_command_line_code_executor_python_with_model():
    stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor(
        work_dir="coding_files",
    )
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_local_command_line_code_executor,
        model_client=model_client,
        max_retries_on_error=3,
    )

    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content="write a numpy demo script'", source="user")],
        cancellation_token=CancellationToken(),
    ):
        print(result)
        assert isinstance(
            result,
            CodeFileMessage | CodeExecutionEvent | CodeGenerationEvent | Response,
        )


@pytest.mark.asyncio
async def test_docker_command_line_code_executor_sh():
    sh_script = """```sh
echo "Hello World"
sleep 1
echo "Hello World"
sleep 1
echo "Hello World"
```"""

    stream_docker_command_line_code_executor = StreamDockerCommandLineCodeExecutor(
        work_dir="coding_files",
        bind_dir=(
            os.getenv("CODING_FILES_PATH")
            if os.getenv("ENVIRONMENT") == "docker"
            else "coding_files"
        ),
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_docker_command_line_code_executor,
    )
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=sh_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )

        elif isinstance(result, CodeFileMessage):
            assert result.code_file.endswith(".sh")
            # assert result.command.split(" ")[2].startswith("sh")


@pytest.mark.asyncio
async def test_docker_command_line_code_executor_python():
    python_script = """```py
import time
print("Hello World")
time.sleep(1)
print("Hello World")
time.sleep(1)
print("Hello World")
```"""
    stream_docker_command_line_code_executor = StreamDockerCommandLineCodeExecutor(
        work_dir="coding_files",
        bind_dir=(
            os.getenv("CODING_FILES_PATH")
            if os.getenv("ENVIRONMENT") == "docker"
            else "coding_files"
        ),
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_docker_command_line_code_executor,
    )
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(result, Response):
            assert result.chat_message.source == "stream_code_executor_agent"
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )

        elif isinstance(result, CodeFileMessage):
            assert result.code_file.endswith(".py")
            # assert result.command.split(" ")[2].endswith(("python", "python3"))


@pytest.mark.asyncio
async def test_docker_command_line_code_executor_python_with_model():
    stream_docker_command_line_code_executor = StreamDockerCommandLineCodeExecutor(
        work_dir="coding_files",
        bind_dir=(
            os.getenv("CODING_FILES_PATH")
            if os.getenv("ENVIRONMENT") == "docker"
            else "coding_files"
        ),
    )
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_docker_command_line_code_executor,
        model_client=model_client,
        max_retries_on_error=3,
    )

    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content="write a numpy demo script'", source="user")],
        cancellation_token=CancellationToken(),
    ):
        print(result)
        assert isinstance(
            result,
            CodeFileMessage | CodeExecutionEvent | CodeGenerationEvent | Response,
        )
