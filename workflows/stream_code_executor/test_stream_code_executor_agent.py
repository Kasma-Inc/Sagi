import json
import time
import unittest
from typing import Dict

from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken

from workflows.stream_code_executor.stream_code_executor_agent import (
    StreamCodeExecutorAgent,
)
from workflows.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from workflows.stream_code_executor.stream_local_command_line_code_executor import (
    StreamLocalCommandLineCodeExecutor,
)


class TestStreamCodeExecutorAgent(unittest.IsolatedAsyncioTestCase):
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)
        self.sh_script: str = """```sh
echo "Hello World"
sleep 1
echo "Hello World"
sleep 1
echo "Hello World"
```"""

    async def test_local_command_line_code_executor(self) -> None:
        stream_local_command_line_code_executor = StreamLocalCommandLineCodeExecutor(
            work_dir="./temp"
        )
        stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
            name="stream_code_executor_agent",
            stream_code_executor=stream_local_command_line_code_executor,
        )
        current_time: float = 0.0
        index: int = 0
        async for result in stream_code_executor_agent.on_messages_stream(
            messages=[TextMessage(content=self.sh_script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if index in range(0, 4):
                self.assertIsInstance(result, ModelClientStreamingChunkEvent)
                if index == 0:
                    content: Dict = json.loads(result.content)
                    self.assertEqual(content["type"], "filename")
                    self.assertEqual(content["result"], None)
                elif index in range(1, 4):
                    content: Dict = json.loads(result.content)
                    self.assertEqual(content["type"], "stdout")
                    self.assertEqual(content["result"], "Hello World")
                    if current_time == 0.0:
                        current_time = time.perf_counter()
                    else:
                        new_time: float = time.perf_counter()
                        self.assertGreater(new_time - current_time, 0.9)
                        current_time = new_time
            elif index == 4:
                self.assertIsInstance(result, Response)
                content: Dict = json.loads(result.chat_message.content)
                self.assertEqual(content["type"], "full_result")
                self.assertEqual(content["result"], "Hello WorldHello WorldHello World")
            index += 1

    async def test_docker_command_line_code_executor(self) -> None:
        stream_docker_command_line_code_executor = StreamDockerCommandLineCodeExecutor(
            work_dir="./temp"
        )
        await stream_docker_command_line_code_executor.start()
        stream_code_executor_agent: StreamCodeExecutorAgent = StreamCodeExecutorAgent(
            name="stream_code_executor_agent",
            stream_code_executor=stream_docker_command_line_code_executor,
        )
        current_time: float = 0.0
        index: int = 0
        async for result in stream_code_executor_agent.on_messages_stream(
            messages=[TextMessage(content=self.sh_script, source="")],
            cancellation_token=CancellationToken(),
        ):
            if index in range(0, 4):
                self.assertIsInstance(result, ModelClientStreamingChunkEvent)
                if index == 0:
                    content: Dict = json.loads(result.content)
                    self.assertEqual(content["type"], "filename")
                    self.assertEqual(content["result"], None)
                elif index in range(1, 4):
                    content: Dict = json.loads(result.content)
                    self.assertEqual(content["type"], "stdout")
                    self.assertEqual(content["result"], "Hello World\n")
                    if current_time == 0.0:
                        current_time = time.perf_counter()
                    else:
                        new_time: float = time.perf_counter()
                        self.assertGreater(new_time - current_time, 0.9)
                        current_time = new_time
            elif index == 4:
                self.assertIsInstance(result, Response)
                content: Dict = json.loads(result.chat_message.content)
                self.assertEqual(content["type"], "full_result")
                self.assertEqual(
                    content["result"], "Hello World\nHello World\nHello World\n"
                )
            index += 1


if __name__ == "__main__":
    unittest.main()
