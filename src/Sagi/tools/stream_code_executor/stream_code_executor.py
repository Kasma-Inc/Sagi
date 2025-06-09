from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List, Literal, Optional

from autogen_agentchat.messages import BaseTextChatMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult
from autogen_ext.code_executors._common import CommandLineCodeResult


class CodeFileMessage(BaseTextChatMessage):
    code_file: str
    # command: str
    content: str
    type: str = "CodeFileMessage"


@dataclass
class CustomCommandLineCodeResult(CommandLineCodeResult):
    command: str
    hostname: str
    user: str
    pwd: str


class CodeResultBlockMessage(BaseTextChatMessage):
    type: Literal["stdout", "stderr"]


class StreamCodeExecutor(CodeExecutor):
    @abstractmethod
    async def execute_code_blocks_stream(
        self,
        chat_id: Optional[str],
        code_blocks: List[CodeBlock],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[CodeFileMessage | CodeResult, None]:
        pass
