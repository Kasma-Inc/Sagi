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


@dataclass
class CodeBlockErrorHistory:
    code_blocks: List[CodeBlock]
    shell_commands: List[CodeBlock]
    error: str
    previous_state: int

    # this is the node in the tree-liked structure, where the error occurred.
    # this contains the error children nodes (and the children nodes also contains their error children, and so on)
    # represented as the tree structure
    children_error_nodes: List["CodeBlockErrorHistory"] = None

    def __post_init__(self):
        if self.children_error_nodes is None:
            self.children_error_nodes = []


@dataclass
class CodeStepHistory:
    instruction: str
    code_blocks: List[CodeBlock]
    result: str
