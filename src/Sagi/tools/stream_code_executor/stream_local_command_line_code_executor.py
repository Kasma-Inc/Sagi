import asyncio
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable, List, Optional, Sequence, Union

from autogen_core import CancellationToken
from autogen_core.code_executor import (
    CodeBlock,
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
)
from autogen_ext.code_executors._common import (
    PYTHON_VARIANTS,
    get_file_name_from_content,
    lang_to_cmd,
    silence_pip,
)
from autogen_ext.code_executors.local import A, LocalCommandLineCodeExecutor

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
    CustomCommandLineCodeResult,
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
        self,
        chat_id: Optional[str],
        code_blocks: List[CodeBlock],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[CodeFileMessage | CustomCommandLineCodeResult, None]:
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

            if chat_id:
                written_file = (self.work_dir / chat_id / filename).resolve()
            else:
                written_file = (self.work_dir / filename).resolve()

            # Ensure parent directory exists
            written_file.parent.mkdir(parents=True, exist_ok=True)

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

            command = " ".join([program] + extra_args)
            content_json = {
                "code_file": str(written_file),
                # "command": command,
                "code_block": code_block.code,
                "code_block_language": code_block.language,
            }
            yield CodeFileMessage(
                content=json.dumps(content_json),
                code_file=str(self.work_dir / filename),
                # command=command,
                source=self.__class__.__name__,
            )
            # Create a subprocess and run
            cwd: Path = self.work_dir
            if chat_id:
                cwd = (cwd / chat_id).resolve()
            task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    program,
                    *extra_args,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            )
            cancellation_token.link_future(task)

            proc = None  # Track the process
            try:
                proc = await task
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), self._timeout
                )
                exitcode = proc.returncode or 0

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

            logs_all += stderr.decode()
            logs_all += stdout.decode()

            if exitcode != 0:
                break

        hostname = subprocess.check_output("hostname").decode().strip()
        user = subprocess.check_output("whoami").decode().strip()
        pwd = (
            subprocess.check_output("pwd")
            .decode()
            .strip()
            .replace(os.path.expanduser("~"), "~")
        )

        code_file = str(file_names[0]) if file_names else None
        yield CustomCommandLineCodeResult(
            exit_code=exitcode,
            output=logs_all,
            code_file=code_file,
            command=command,
            hostname=hostname,
            user=user,
            pwd=pwd,
        )
