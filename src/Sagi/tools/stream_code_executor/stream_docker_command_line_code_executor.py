import asyncio
import json
from asyncio import Event
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Union

from autogen_core import CancellationToken
from autogen_core.code_executor import (
    CodeBlock,
    CodeResult,
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
)
from autogen_ext.code_executors._common import (
    CommandLineCodeResult,
    get_file_name_from_content,
    lang_to_cmd,
    silence_pip,
)
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import A
from docker.types import CancellableStream, DeviceRequest

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
    CodeResultBlockMessage,
    StreamCodeExecutor,
)


class StreamDockerCommandLineCodeExecutor(
    DockerCommandLineCodeExecutor, StreamCodeExecutor
):
    def __init__(
        self,
        image: str = "python:3-slim",
        container_name: Optional[str] = None,
        *,
        timeout: int = 60,
        work_dir: Union[Path, str, None] = None,
        bind_dir: Optional[Union[Path, str]] = None,
        auto_remove: bool = True,
        stop_container: bool = True,
        device_requests: Optional[List[DeviceRequest]] = None,
        functions: Sequence[
            Union[
                FunctionWithRequirements[Any, A],
                Callable[..., Any],
                FunctionWithRequirementsStr,
            ]
        ] = [],
        functions_module: str = "functions",
        extra_volumes: Optional[Dict[str, Dict[str, str]]] = None,
        extra_hosts: Optional[Dict[str, str]] = None,
        init_command: Optional[str] = None,
        delete_tmp_files: bool = False,
    ):
        super().__init__(
            image=image,
            container_name=container_name,
            timeout=timeout,
            work_dir=work_dir,
            bind_dir=bind_dir,
            auto_remove=auto_remove,
            stop_container=stop_container,
            device_requests=device_requests,
            functions=functions,
            functions_module=functions_module,
            extra_volumes=extra_volumes,
            extra_hosts=extra_hosts,
            init_command=init_command,
            delete_tmp_files=delete_tmp_files,
        )

    async def execute_code_blocks_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[
        CodeFileMessage | CodeResultBlockMessage | CommandLineCodeResult, None
    ]:
        if not self._setup_functions_complete:
            await self._setup_functions(cancellation_token)

        async for result in self._execute_code_dont_check_setup_stream(
            code_blocks, cancellation_token
        ):
            yield result

    async def _execute_code_dont_check_setup_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[
        CodeFileMessage | CodeResultBlockMessage | CommandLineCodeResult, None
    ]:
        if self._container is None or not self._running:
            raise ValueError(
                "Container is not running. Must first be started with either start or a context manager."
            )

        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs: List[str] = []
        files: List[Path] = []
        last_exit_code = 0
        try:
            for code_block in code_blocks:
                lang = code_block.language.lower()
                code = silence_pip(code_block.code, lang)

                # Check if there is a filename comment
                try:
                    filename = get_file_name_from_content(code, self.work_dir)
                except ValueError:
                    outputs.append("Filename is not in the workspace")
                    last_exit_code = 1
                    break

                if not filename:
                    filename = f"tmp_code_{sha256(code.encode()).hexdigest()}.{lang}"

                code_path = self.work_dir / filename
                with code_path.open("w", encoding="utf-8") as fout:
                    fout.write(code)
                files.append(code_path)

                lang_cmd: str = lang_to_cmd(lang)
                if lang_cmd == "python":
                    command = ["timeout", str(self._timeout), "python", "-u", filename]
                else:
                    command = ["timeout", str(self._timeout), lang_cmd, filename]

                content_json = {
                    "code_file": str(code_path),
                    "code_block": code_block.code,
                    "code_block_language": code_block.language,
                }
                yield CodeFileMessage(
                    content=json.dumps(content_json),
                    code_file=str(code_path),
                    source=self.__class__.__name__,
                )

                async for result in self._execute_command_stream(
                    command, cancellation_token
                ):
                    if isinstance(result, CodeResult):
                        last_exit_code = int(result.exit_code)
                        outputs.append(result.output)
                    else:
                        yield result

                if last_exit_code != 0:
                    break
        finally:
            if self._delete_tmp_files:
                for file in files:
                    try:
                        file.unlink()
                    except (OSError, FileNotFoundError):
                        pass

        code_file = str(files[0]) if files else None
        yield CommandLineCodeResult(
            exit_code=last_exit_code, output="".join(outputs), code_file=code_file
        )

    async def handle_execute_command_stream_cancel(
        self, event: Event, command: List[str]
    ):
        try:
            await event.wait()
        except asyncio.CancelledError:
            self._cancellation_tasks.append(
                asyncio.create_task(self._kill_running_command(command))
            )

    async def _execute_command_stream(
        self, command: List[str], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeResultBlockMessage | CodeResult, None]:
        if self._container is None or not self._running:
            raise ValueError(
                "Container is not running. Must first be started with either start or a context manager."
            )
        exec_id: str = (
            await asyncio.to_thread(
                self._container.client.api.exec_create, self._container.id, command
            )
        )["Id"]
        result: CancellableStream = await asyncio.to_thread(
            self._container.client.api.exec_start, exec_id, stream=True, demux=True
        )

        event: Event = asyncio.Event()
        handle_execute_command_stream_cancel_task = asyncio.create_task(
            self.handle_execute_command_stream_cancel(event, command)
        )
        cancellation_token.link_future(handle_execute_command_stream_cancel_task)

        output: str = ""

        while True:
            stdout, stderr = await asyncio.to_thread(next, result, (None, None))
            if stdout is None and stderr is None:
                break
            else:
                if stdout is not None:
                    stdout_decode: str = stdout.decode("utf-8")
                    output += stdout_decode
                    yield CodeResultBlockMessage(
                        type="stdout",
                        content=stdout_decode,
                        source=self.__class__.__name__,
                    )
                if stderr is not None:
                    stderr_decode: str = stderr.decode("utf-8")
                    output += stderr_decode
                    yield CodeResultBlockMessage(
                        type="stderr",
                        content=stderr_decode,
                        source=self.__class__.__name__,
                    )

        event.set()
        exit_code: int = self._container.client.api.exec_inspect(exec_id)["ExitCode"]
        if exit_code == 124:
            output += "\n Timeout"
        if cancellation_token.is_cancelled():
            output = "Code execution was cancelled."
            exit_code = 1

        yield CodeResult(exit_code=exit_code, output=output)
