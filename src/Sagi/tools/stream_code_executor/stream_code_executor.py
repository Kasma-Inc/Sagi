from abc import abstractmethod
from typing import AsyncGenerator, List

from autogen_agentchat.messages import BaseTextChatMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult


class CodeFileMessage(BaseTextChatMessage):
    code_file: str
    command: str
    content: str


class StreamCodeExecutor(CodeExecutor):
    @abstractmethod
    async def execute_code_blocks_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeFileMessage | CodeResult, None]:
        pass
