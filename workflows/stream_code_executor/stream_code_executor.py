from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List, Literal

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor, CodeResult


@dataclass
class CodeResultBlock:
    type: Literal["filename", "stderr", "stdout"]
    output: str


class StreamCodeExecutor(CodeExecutor):
    @abstractmethod
    async def execute_code_blocks_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeResultBlock | CodeResult, None]:
        pass
