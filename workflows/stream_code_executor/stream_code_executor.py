from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeExecutor, CodeResult, CodeBlock


@dataclass
class CodeResultBlock:
    type: str
    result: str


class StreamCodeExecutor(CodeExecutor):
    @abstractmethod
    async def execute_code_blocks_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeResultBlock | CodeResult, None]:
        pass
