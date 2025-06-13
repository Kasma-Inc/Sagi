import asyncio
import json
import os.path
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
    PYTHON_VARIANTS,
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
    CustomCommandLineCodeResult,
    StreamCodeExecutor,
)


class StreamDockerCommandLineCodeExecutor(
    DockerCommandLineCodeExecutor, StreamCodeExecutor
):
    def __init__(
        self,
        image: str = "sagicuhk/docker_executor_image",
        container_name: Optional[str] = None,
        *,
        timeout: int = 60,
        work_dir: Union[Path, str, None] = None,
        bind_dir: Optional[Union[Path, str]] = None,
        auto_remove: bool = True,  # Keep the auto_remove option to be true to make sure the container is deleted even if the program terminates
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

        # List of installed dependencies in the Docker container
        self.docker_installed_dependencies: List[CodeBlock] = []

        # for countdown functionality
        self._countdown_task = None
        self._countdown_running = False

    async def execute_code_blocks_stream(
        self,
        chat_id: Optional[str],
        code_blocks: List[CodeBlock],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[
        CodeFileMessage | CodeResultBlockMessage | CommandLineCodeResult, None
    ]:
        if not self._setup_functions_complete:
            await self._setup_functions(cancellation_token)

        async for result in self._execute_code_dont_check_setup_stream(
            chat_id, code_blocks, cancellation_token
        ):
            yield result

    async def _execute_code_dont_check_setup_stream(
        self,
        chat_id: Optional[str],
        code_blocks: List[CodeBlock],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[CodeFileMessage | CustomCommandLineCodeResult, None]:
        if self._container is None or not self._running:
            raise ValueError(
                "Container is not running. Must first be started with either start or a context manager."
            )

        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs: List[str] = []
        file_names: List[str] = []
        last_exit_code = 0
        try:
            for code_block in code_blocks:
                lang, code = code_block.language, code_block.code
                lang = lang.lower()

                # Remove pip output where possible
                code = silence_pip(code, lang)

                # Normalize python variants to "python"
                if lang in PYTHON_VARIANTS:
                    lang = "python"

                # Abort if not supported
                if lang not in self.SUPPORTED_LANGUAGES:
                    exitcode = 1
                    outputs += "\n" + f"unknown language {lang}"
                    break

                # Check if there is a filename comment
                try:
                    filename = get_file_name_from_content(code, self.work_dir)
                except ValueError:
                    yield CustomCommandLineCodeResult(
                        exit_code=1,
                        output="Filename is not in the workspace",
                        code_file=None,
                        command="",
                        hostname="",
                        user="",
                        pwd="",
                    )
                    return

                if filename is None:
                    code_hash = sha256(code.encode()).hexdigest()
                    if lang.startswith("python"):
                        ext = "py"
                    elif lang in ["pwsh", "powershell", "ps1"]:
                        ext = "ps1"
                    else:
                        ext = lang

                    filename = f"tmp_code_{code_hash}.{ext}"

                if chat_id:
                    written_file = (self.work_dir / chat_id / filename).resolve()
                else:
                    written_file = (self.work_dir / filename).resolve()

                # Ensure parent directory exists
                written_file.parent.mkdir(parents=True, exist_ok=True)

                with written_file.open("w", encoding="utf-8") as f:
                    f.write(code)
                file_names.append(written_file)

                # Execute the command
                lang_cmd: str = lang_to_cmd(lang)
                if lang_cmd == "python":
                    command = ["python", "-u", filename]
                else:
                    command = [lang_cmd, filename]

                content_json = {
                    "code_file": str(filename),
                    "code_block": code_block.code,
                    "code_block_language": code_block.language,
                }
                yield CodeFileMessage(
                    content=json.dumps(content_json),
                    code_file=str(filename),
                    source=self.__class__.__name__,
                )

                async for result in self._execute_command_stream(
                    chat_id, command, cancellation_token
                ):
                    if isinstance(result, CodeResult):
                        last_exit_code = int(result.exit_code)
                        outputs.append(result.output)
                    # else:
                    #     yield result
                    # TODO: Handle streaming output

                if last_exit_code != 0:
                    break
        finally:
            if self._delete_tmp_files:
                for file in file_names:
                    try:
                        file.unlink()
                    except (OSError, FileNotFoundError):
                        pass

        hostname = self._container.id[:12]  # 12 chars of container id as the hostname
        user = "root"
        pwd = "workspace"
        yield CustomCommandLineCodeResult(
            exit_code=last_exit_code,
            output="".join(outputs),
            code_file=filename,
            command=" ".join(command),
            hostname=hostname,
            user=user,
            pwd=pwd,
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
        self,
        chat_id: Optional[str],
        command: List[str],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[CodeResultBlockMessage | CodeResult, None]:
        if self._container is None or not self._running:
            raise ValueError(
                "Container is not running. Must first be started with either start or a context manager."
            )
        workdir_in_container: str = "/workspace"
        if chat_id:
            workdir_in_container = os.path.join(workdir_in_container, chat_id)
        exec_id: str = (
            await asyncio.to_thread(
                self._container.client.api.exec_create,
                self._container.id,
                command,
                workdir=workdir_in_container,
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

    # Add a dependency to the list of installed dependencies in the Docker container.
    # Make sure it is a CodeBlock object, NOT list[CodeBlock].
    def add_dependency(self, code_block: CodeBlock):
        self.docker_installed_dependencies.append(code_block)

    # Install dependencies in the Docker container.
    async def _install_dependencies(self, cancellation_token: CancellationToken):
        if not self._running:
            raise ValueError("Container is not running. Must first be started.")

        await self.execute_code_blocks(
            self.docker_installed_dependencies, cancellation_token
        )

    async def countdown(self, second: int):
        await self.stop_countdown()  # Ensure any existing countdown is stopped

        print(f"** Container will be stopped in {second} second if inactive...")
        self._countdown_running = True
        self._countdown_task = asyncio.create_task(self._countdown_worker(second))

    async def stop_countdown(self):
        # stop the countdown if it is running
        if (
            self._countdown_running
            and self._countdown_task
            and not self._countdown_task.done()
        ):
            self._countdown_task.cancel()
            try:
                # Wait briefly for cancellation to complete
                await asyncio.wait_for(self._countdown_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _countdown_worker(self, second: int):
        # worker coroutine for countdown if the user does not respond
        try:
            await asyncio.sleep(second)

            print(
                "Countdown completed. Stopping the container due to inactivity. You can continue by entering the response.\nUser: ",
                end="",
            )
            await self.stop()

        except asyncio.CancelledError:
            print("Countdown cancelled. Container will remain active")

        finally:
            self._countdown_running = False
            self._countdown_task = None

    async def resume_docker_container(
        self, cancellation_token: CancellationToken = CancellationToken()
    ):  # this will be called when user's response is received
        await self.stop_countdown()
        if self._running:
            print("Docker container is already running. Resuming...")
        else:
            print("Start the docker constainer again...")
            await self.start()
            assert (
                await self.is_running()
            ), "Docker container should be running after start()"
            print("Docker container started.")

            if len(self.docker_installed_dependencies) > 0:
                # Install dependencies if there are any
                print("Installing dependencies...")
                await self._install_dependencies(cancellation_token)
                print("Dependencies installed.")
            else:
                # No dependencies to install
                print("No dependencies have to be installed.")

    # check if the container is running
    async def is_running(self) -> bool:
        return self._running
