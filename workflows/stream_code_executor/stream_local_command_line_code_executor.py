import asyncio
import os
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from autogen_core import CancellationToken
from autogen_core.code_executor import (
    CodeBlock,
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
from autogen_ext.code_executors.local import A, LocalCommandLineCodeExecutor

from workflows.stream_code_executor.stream_code_executor import (
    CodeResultBlock,
    StreamCodeExecutor,
)


class StreamLocalCommandLineCodeExecutor(
    LocalCommandLineCodeExecutor, StreamCodeExecutor
):
    def __init__(
        self,
        timeout: int = 60,
        work_dir: Optional[Union[Path, str]] = None,
        functions: Sequence[
            Union[
                FunctionWithRequirements[Any, A],
                Callable[..., Any],
                FunctionWithRequirementsStr,
            ]
        ] = [],
        functions_module: str = "functions",
        virtual_env_context: Optional[SimpleNamespace] = None,
    ):
        super().__init__(
            timeout=timeout,
            work_dir=work_dir,
            functions=functions,
            functions_module=functions_module,
            virtual_env_context=virtual_env_context,
        )

    async def execute_code_blocks_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeResultBlock | CommandLineCodeResult, None]:
        if not self._setup_functions_complete:
            await self._setup_functions(cancellation_token)

        async for result in self._execute_code_dont_check_setup_stream(
            code_blocks, cancellation_token
        ):
            yield result

    async def _execute_code_dont_check_setup_stream(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeResultBlock | CommandLineCodeResult, None]:
        logs_all: str = ""
        file_names: List[Path] = []
        exitcode = 0

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
                logs_all += "\n" + f"unknown language {lang}"
                break

            # Try extracting a filename (if present)
            try:
                filename = get_file_name_from_content(code, self.work_dir)
                yield CodeResultBlock(type="filename", output=filename)
            except ValueError:
                yield CommandLineCodeResult(
                    exit_code=1,
                    output="Filename is not in the workspace",
                    code_file=None,
                )
                return

            # If no filename is found, create one
            if filename is None:
                code_hash = sha256(code.encode()).hexdigest()
                if lang.startswith("python"):
                    ext = "py"
                elif lang in ["pwsh", "powershell", "ps1"]:
                    ext = "ps1"
                else:
                    ext = lang

                filename = f"tmp_code_{code_hash}.{ext}"

            written_file = (self.work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as f:
                f.write(code)
            file_names.append(written_file)

            # Build environment
            env = os.environ.copy()
            if self._virtual_env_context:
                virtual_env_bin_abs_path = os.path.abspath(
                    self._virtual_env_context.bin_path
                )
                env["PATH"] = f"{virtual_env_bin_abs_path}{os.pathsep}{env['PATH']}"

            # Decide how to invoke the script
            if lang == "python":
                program = (
                    os.path.abspath(self._virtual_env_context.env_exe)
                    if self._virtual_env_context
                    else sys.executable
                )
                extra_args = [str(written_file.absolute())]
            else:
                # Get the appropriate command for the language
                program = lang_to_cmd(lang)

                # Special handling for PowerShell
                if program == "pwsh":
                    extra_args = [
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(written_file.absolute()),
                    ]
                else:
                    # Shell commands (bash, sh, etc.)
                    extra_args = [str(written_file.absolute())]

            # Create a subprocess and run
            task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    program,
                    *extra_args,
                    cwd=self.work_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            )
            cancellation_token.link_future(task)

            proc = None  # Track the process
            try:
                proc = await task
                stderr: str = ""
                stdout: str = ""

                async def read_stream(stream, stream_type: Literal["stderr", "stdout"]):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        yield CodeResultBlock(type=stream_type, output=line.decode())

                async for result in read_stream(proc.stdout, "stdout"):
                    stdout += result.output
                    yield result
                async for result in read_stream(proc.stderr, "stderr"):
                    stderr += result.output
                    yield result

                await proc.wait()
                exitcode: int = proc.returncode or 0
            except asyncio.TimeoutError:
                logs_all += "\nTimeout"
                exitcode = 124
                if proc:
                    proc.terminate()
                    await proc.wait()  # Ensure process is fully dead
                break
            except asyncio.CancelledError:
                logs_all += "\nCancelled"
                exitcode = 125
                if proc:
                    proc.terminate()
                    await proc.wait()
                break

            logs_all += stderr
            logs_all += stdout

            if exitcode != 0:
                break

        code_file = str(file_names[0]) if file_names else None
        yield CommandLineCodeResult(
            exit_code=exitcode, output=logs_all, code_file=code_file
        )
