import json
import os
from typing import Dict

import pytest
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    CodeGenerationEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
)
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from Sagi.tools.stream_code_executor.stream_code_executor_agent import (
    CodeExecutionEvent,
    StreamCodeExecutorAgent,
)
from Sagi.tools.stream_code_executor.stream_local_command_line_code_executor import (
    StreamLocalCommandLineCodeExecutor,
)

load_dotenv(".env")


@pytest.mark.asyncio
async def test_local_command_line_code_executor_sh():
    sh_script = """```sh
echo "Hello World"
sleep 1
echo "Hello World"
sleep 1
echo "Hello World"
```"""

    stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor()
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_local_command_line_code_executor,
    )
    index: int = 0
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=sh_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        print(result)
        if index in range(0, 4):
            assert isinstance(result, ModelClientStreamingChunkEvent)
            if index == 0:
                content: Dict = json.loads(result.content)
                assert content["type"] == "filename"
                assert content["result"] is not None
            elif index in range(1, 4):
                content: Dict = json.loads(result.content)
                assert content["type"] == "stdout"
                assert content["result"] == "Hello World\n"
        elif index == 4:
            assert isinstance(result, Response)
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )
        index += 1


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
        work_dir="temp",
    )
    stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
        name="stream_code_executor_agent",
        stream_code_executor=stream_local_command_line_code_executor,
    )
    index: int = 0
    async for result in stream_code_executor_agent.on_messages_stream(
        messages=[TextMessage(content=python_script, source="")],
        cancellation_token=CancellationToken(),
    ):
        print(result)
        if index in range(0, 4):
            assert isinstance(result, ModelClientStreamingChunkEvent)
            if index == 0:
                content: Dict = json.loads(result.content)
                assert content["type"] == "filename"
                assert content["result"] is not None
            elif index in range(1, 4):
                content: Dict = json.loads(result.content)
                assert content["type"] == "stdout"
                assert content["result"] == "Hello World\n"

        elif index == 4:
            assert isinstance(result, Response)
            assert (
                result.chat_message.content == "Hello World\nHello World\nHello World\n"
            )
        index += 1


@pytest.mark.asyncio
async def test_local_command_line_code_executor_python_with_model():
    stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor(
        work_dir="temp",
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
            ModelClientStreamingChunkEvent
            | CodeExecutionEvent
            | CodeGenerationEvent
            | Response,
        )
