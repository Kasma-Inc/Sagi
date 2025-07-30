import re
from typing import Any, AsyncGenerator, List, Mapping, Optional, Sequence, Union, Literal
import json

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    CodeExecutionEvent,
    CodeGenerationEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
    ThoughtEvent,
)
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    SystemMessage,
    UserMessage,
)

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeBlockErrorHistory,
    CodeFileMessage,
    CodeStepHistory,
    StreamCodeExecutor,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)

from pydantic import BaseModel

class AgentResponse(BaseModel):
    content: str
    env_issue: bool
    previous_state: Optional[int] = None
    
    def to_string(self) -> str:
        return (
            f"Content: {self.content}\n"
            f"Environment Issue: {self.env_issue}\n"
            f"Previous State: {self.previous_state}"
        )

def response_output(error_history_length: int): # create a structured response type for the agent response based on the length of the error history
    allowed_values = tuple(range(-1, error_history_length))
    StateType = Union[Literal[allowed_values], None]
    
    class RangeAgentResponse(AgentResponse):
        previous_state: StateType = None
    
    return RangeAgentResponse

class StreamCodeExecutorAgent(CodeExecutorAgent):
    def __init__(
        self,
        name: str,
        stream_code_executor: StreamCodeExecutor,
        *,
        model_client: ChatCompletionClient | None = None,
        model_context: ChatCompletionContext | None = None,
        model_client_stream: bool = False,
        max_retries_on_error: int = 0,
        description: str | None = None,
        system_message: str | None = CodeExecutorAgent.DEFAULT_SYSTEM_MESSAGE,
        sources: Sequence[str] | None = None,
        countdown_timer: int = 60,  # time before the docker container is stopped
    ) -> None:
        super().__init__(
            name,
            stream_code_executor,
            model_client=model_client,
            model_context=model_context,
            model_client_stream=model_client_stream,
            max_retries_on_error=max_retries_on_error,
            description=description,
            system_message=system_message,
            sources=sources,
        )
        self._code_executor: StreamCodeExecutor = stream_code_executor
        self.user_id: Optional[str] = None
        self.chat_id: Optional[str] = None
        self._step_history: List[CodeStepHistory] = (
            []
        )  # To keep track of summary of each code execution steps
        self._countdown_timer = countdown_timer

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            await self._code_executor.resume_docker_container()

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        system_messages = self._system_messages
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        max_retries_on_error = self._max_retries_on_error

        if model_client is None:  # default behaviour for backward compatibility
            # execute generated code if present
            code_blocks: List[CodeBlock] = await self.extract_code_blocks_from_messages(
                messages
            )
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=self.NO_CODE_BLOCKS_FOUND_MESSAGE,
                        source=agent_name,
                    )
                )
                await self.EXIT()
                return

            for code_block in code_blocks:
                async for result in self.execute_code_block(
                    [code_block], cancellation_token
                ):
                    if isinstance(result, CodeFileMessage):
                        yield result
                    elif isinstance(result, CodeResult):
                        yield Response(
                            chat_message=TextMessage(
                                content=result.output,
                                source=self.name,
                            )
                        )

                assert result.exit_code == 0, "Expected code execution to succeed."
                if code_block.language == "sh" and isinstance(
                    self._code_executor, StreamDockerCommandLineCodeExecutor
                ):
                    self._code_executor.add_dependency(code_block)

            await self.EXIT()
            return

        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []

        await model_context.clear()  # Clear the model context to start fresh

        # Add the summary of the previous steps to the model context
        for history in self._step_history:
            await model_context.add_message(
                SystemMessage(
                    content=self.history_to_model_text(history),
                    source=agent_name,
                )
            )

        current_history_path: List[CodeBlockErrorHistory] = (
            []
        )  # current error history path
        message_texts = [
            f" - {msg.to_model_text()}" for msg in messages
        ]  # messages instruction
        current_history_path.append(  # Append the first stage of error history (root node)
            CodeBlockErrorHistory(
                code_blocks=None,  # No code blocks at the first stage
                shell_commands=None,  # No command blocks at the first stage
                error="This is the first stage. The code has not been generated yet. Please follow the instructions above to generate the code.",
                previous_state=-1,
            )
        )

        # Add the instruction messages to the model context
        await model_context.add_message(
            UserMessage(
                content="Current Instructions with the previous messages history:\n"
                + "\n".join(message_texts),
                source=agent_name,
            )
        )

        # Initialize code_blocks here, so that the final code_blocks after the loop can be used by refering to this variable
        code_blocks: CodeBlock = None

        # Initialize the result of the code execution, so that the final result can be used by refering to this variable
        result_output: str = None

        for nth_try in range(
            max_retries_on_error + 1
        ):  # Do one default generation, execution and inference loop
            # Step 1: Add new user/handoff messages to the model context

            original_messages = await model_context.save_state()

            assert (
                len(current_history_path) > 0
            ), "Expected current_history_path to be initialized."

            await model_context.add_message(
                UserMessage(
                    content=self.error_history_prompt(current_history_path),
                    source=agent_name,
                )
            )

            # Step 2: Run inference with the model context
            model_result = None
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
                output_content_type=response_output(len(current_history_path)),  # Specify the output content type for structured response
            ):
                if isinstance(inference_output, CreateResult):
                    model_result = inference_output
                else:
                    # Streaming chunk event
                    yield inference_output

            assert model_result is not None, "No model result was produced."

            model_response = AgentResponse.model_validate_json(model_result.content)

            assert isinstance(model_response, AgentResponse), (
                "Expected model_response to be of type AgentResponse, "
                f"but got {type(model_response)}."
            )

            # Step 3: [NEW] If the model produced a hidden "thought," yield it as an event
            if model_result.thought:
                thought_event = ThoughtEvent(
                    content=model_result.thought, source=agent_name
                )
                yield thought_event
                inner_messages.append(thought_event)

            # Step 4: Add the assistant message to the model context (including thought if present)
            await model_context.add_message(
                AssistantMessage(
                    content=model_response.to_string(),
                    source=agent_name,
                    thought=getattr(model_result, "thought", None),
                )
            )

            # Step 5: Extract the code blocks and previous state from inferred text

            previous_state = model_response.previous_state
            env_issue = model_response.env_issue

            if previous_state == -1:
                # No solution can be found, so we exit the loop
                yield Response(
                    chat_message=TextMessage(
                        content=f"No solution can be found. The model's response was: {model_response.to_string()}",
                        source=agent_name,
                    )
                )
                await self.EXIT()
                return

            assert (
                0 <= previous_state < len(current_history_path)
            ), "Expected 'PREVIOUS_STATE' to be within the range of error history."

            if previous_state < 0 or previous_state >= len(current_history_path):
                previous_state = len(current_history_path) - 1

            shell_commands: List[CodeBlock] = None

            if env_issue:
                # The issue arises from the environment itself, so the code_blocks can be refer from the ancestor error history state(node)
                # We don't store the code blocks in the error history to save the model_context_size and token used,
                # because the code blocks are not changed to fix the environment issue.
                now = previous_state
                while now != -1 and current_history_path[now].code_blocks is None:
                    now = current_history_path[now].previous_state

                assert now != -1, "Expected to find a ancestor state with code blocks."
                code_blocks = current_history_path[now].code_blocks

                # Shell to fix the environment issue
                command_blocks = self._extract_markdown_code_blocks(
                    str(model_response.content)
                )
                shell_commands = [
                    block for block in command_blocks if block.language == "sh"
                ]

            else:
                # The issue arises from the code itself, so we need to extract the code blocks from the model's response
                code_blocks = self._extract_markdown_code_blocks(
                    str(model_response.content)
                )

            # Step 6: Exit the loop if no code blocks found
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=f"No code blocks found. The model's response was: {model_response.to_string()}",
                        source=agent_name,
                    )
                )
                await self.EXIT()
                return

            # combine the code blocks with the shell commands if they are present
            combined_blocks = []
            if env_issue and shell_commands:
                combined_blocks.extend(shell_commands)
            if code_blocks:
                combined_blocks.extend(code_blocks)

            # Step 7: Yield a CodeGenerationEvent
            inferred_text_message: CodeGenerationEvent = CodeGenerationEvent(
                retry_attempt=nth_try,
                content=model_response.content,
                code_blocks=combined_blocks,
                source=agent_name,
            )

            yield inferred_text_message
            inner_messages.append(inferred_text_message)

            # Step 8: Execute the extracted code blocks
            for codeblock in combined_blocks:
                async for result in self.execute_code_block(
                    [codeblock], cancellation_token
                ):
                    # Return CodeFileMessage or CodeResult
                    # CodeFileMessage stores the command and the code file content
                    # CodeResult stores the exit code, stdout and stderr
                    # First yield the CodeFileMessage and then the CodeResult
                    if isinstance(result, CodeFileMessage):
                        result.source = self.name
                        yield result
                        inner_messages.append(result)
                    elif isinstance(result, CodeResult):
                        # Step 9: Update model context with the code execution result
                        result_output = result.output

                        await model_context.add_message(
                            SystemMessage(
                                content=f"The command {result.code_file} was executed with the following output: {result.description}",
                                source=agent_name,
                            )
                        )

                        # Step 10: Yield a CodeExecutionEvent
                        code_execution_event = CodeExecutionEvent(
                            retry_attempt=nth_try, result=result, source=self.name
                        )
                        yield code_execution_event
                        inner_messages.append(code_execution_event)

                if result.exit_code != 0:
                    # If the code execution failed, we break the loop and store the error -> no need to continue the execution
                    break

                # check if the code block is a shell command (without error) and we run in docker container command line
                if (
                    isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor)
                    and codeblock.language == "sh"
                ):
                    self._code_executor.add_dependency(codeblock)

            # If execution was successful or last retry, then exit
            if result.exit_code == 0 or nth_try == max_retries_on_error:
                if nth_try == max_retries_on_error and result.exit_code != 0:
                    # If we reached the maximum number of retries and the exit code is still non-zero, we break the loop
                    yield Response(
                        chat_message=TextMessage(
                            content=f"Reached maximum retries ({max_retries_on_error}) with exit code {result.exit_code}. The model's response was: {model_response.to_string()}",
                            source=agent_name,
                        )
                    )

                    await model_context.add_message(
                        SystemMessage(
                            content=f"Reached maximum retries ({max_retries_on_error}) with exit code {result.exit_code}. The model's response was: {model_response.to_string()}",
                            source=agent_name,
                        )
                    )

                break

            # Reset context only when we know that the code will be executed again (in order to add the new history tree)
            # If the code can be run, we have to keep the model_context because this context will be used for summarization and reflection
            await model_context.clear()
            await model_context.load_state(original_messages)

            # Step 11: If exit code is non-zero, then we add the error to the error history
            #          and pop the current_history_path until the previous state

            # Pop the current history path until the previous state
            while len(current_history_path) > previous_state + 1:
                current_history_path.pop()

            # Create a new error node with the current code blocks/command blocks and error message
            new_error_node = CodeBlockErrorHistory(
                shell_commands=(
                    shell_commands if env_issue else None
                ),  # If the error is due to syntax/logical issue, we don't need to store the command blocks
                code_blocks=(
                    code_blocks if not env_issue else None
                ),  # If the error is due to environment issue, we don't need to store the code blocks
                error=f"The command {result.code_file} was executed with the following output: {result.description}",
                previous_state=previous_state,
            )

            current_history_path[previous_state].children_error_nodes.append(
                new_error_node
            )  # connect the new error node to the previous state
            current_history_path.append(
                new_error_node
            )  # Add the new error node to the current history path

        self._step_history.append(
            CodeStepHistory(
                instruction="\n".join(message_texts),
                code_blocks=code_blocks,
                result=(
                    result_output
                    if result_output is not None
                    else "No code execution in this step."
                ),
            )
        )

        # Always reflect on the execution result
        async for (
            reflection_response
        ) in StreamCodeExecutorAgent._reflect_on_code_block_results_flow(
            system_messages=system_messages,
            model_client=model_client,
            model_client_stream=model_client_stream,
            model_context=model_context,
            agent_name=agent_name,
            inner_messages=inner_messages,
        ):
            yield reflection_response  # Last reflection_response is of type Response so it will finish the routine

        await self.EXIT()

    async def execute_code_block(  # TO DO: have to fix this function, sometimes it executes the code in the near time (conflict)
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeFileMessage | CodeResult, None]:
        # Execute the code blocks.

        # Ensure the code executor is running
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            assert (
                await self._code_executor.is_running()
            ), "Expected StreamDockerCommandLineCodeExecutor to be running."

        async for result in self._code_executor.execute_code_blocks_stream(
            self.chat_id, code_blocks, cancellation_token=cancellation_token
        ):
            if isinstance(result, CodeResult):
                if result.output.strip() == "":
                    # No output
                    result.description = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed."
                elif result.exit_code != 0:
                    # Error
                    result.description = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"
                else:
                    result.description = result.output
            yield result

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()

    def history_to_model_text(self, history: CodeStepHistory) -> str:
        return (
            f"Previous Instruction: {history.instruction}\n"
            f"Result Code Blocks: {history.code_blocks}\n"
            f"Result Execution: {history.result}\n"
        )

    def error_history_prompt(
        self, current_history_path: List[CodeBlockErrorHistory]
    ) -> str:
        return (
            "Review this error history tree showing previous code execution attempts\n"
            "Each node(history path item) is a CodeBlockErrorHistory objects, formatted as :\n"
            " - code_blocks: code at that stage (None at first stage)\n"
            " - error: error message\n"
            " - children_error_nodes: recursive list of CodeBlockErrorHistory objects (tree structure)\n"
            "   representing previously failed solution attempts at that stage\n"
            " - previous_state: index of parent state (0-based)\n\n"
            "First try fixing the last stage (leaf node in the current path). If too difficult, go back to earlier stages but avoid methods already in children_error_nodes.\n\n"
            "Current error path:\n"
            + "\n".join(
                f" Stage {i}:\n"
                f" Code: {current_history_path[i].code_blocks}\n"
                f" Error: {current_history_path[i].error}\n"
                f" children_error_nodes: {current_history_path[i].children_error_nodes}\n"
                f" shell_commands: {current_history_path[i].shell_commands}"
                for i in range(len(current_history_path))
            )

            + "\n\nGenerate a response to fix the issue using the following structure:\n\n"
            
            "content: Provide the solution as either:\n"
            " - For environment issues: A shell command in a code block like this\n"
            "```sh\n"
            "pip install package_name\n"
            "```\n"
            " - For non-environment issues: The code to fix the issue in a code block\n"
            " - If no solution exists, explain why no solution is available\n\n"

            "env_issue: Determine if this is an environment-related issue (missing packages, installation problems, etc.) True if the issue is related to the environment, otherwise False\n\n"

            "previous_state: Specify which stage to continue from:\n"
            "- For environment issues: Must NOT be 0 (since there will be no code blocks at stage 0)\n"
            "- For non-environment issues: Can be any appropriate stage number\n"
            "- Use -1 if no solution exists\n"
            "- Leave as null only for reflection processes\n\n"

            "Notes:\n"
            "- Unless you are reflecting on the process, you should specify a previous state\n"
        )

    @classmethod
    async def _reflect_on_code_block_results_flow(
        cls,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        model_context: ChatCompletionContext,
        agent_name: str,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
    ) -> AsyncGenerator[Response | ModelClientStreamingChunkEvent | ThoughtEvent, None]:
        """
        If reflect_on_code_block_results=True, we do another inference based on tool results
        and yield the final text response (or streaming chunks).
        """
        reflection_prompt = (
            "Reflect briefly ( <= 4 sentences ) on the code execution results. "
            "If the process was error, please state clearly that the code was errored. "
            "If the process was successful, please state clearly that the code was successful.\n"
            "IMPORTANT: do NOT repeat the code blocks!!!\n"
            "Remember: you are reflection agent, you do NOT have to provide PREVIOUS_STATE or ENVIRONMENT_ISSUE, you can just reflect on the process.\n"
            "Please ensure to return the response ONLY in string format.\n"
        )

        all_messages = (
            system_messages
            + await model_context.get_messages()
            + [UserMessage(content=reflection_prompt, source=agent_name)]
        )
        llm_messages = cls._get_compatible_context(
            model_client=model_client, messages=all_messages
        )

        reflection_result: Optional[CreateResult] = None

        if model_client_stream:
            async for chunk in model_client.create_stream(llm_messages):
                if isinstance(chunk, CreateResult):
                    reflection_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source=agent_name
                    )
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        else:
            reflection_result = await model_client.create(llm_messages)

        if not reflection_result or not isinstance(reflection_result.content, str):
            raise RuntimeError("Reflect on tool use produced no valid text response.")

        # --- NEW: If the reflection produced a thought, yield it ---
        if reflection_result.thought:
            thought_event = ThoughtEvent(
                content=reflection_result.thought, source=agent_name
            )
            yield thought_event
            inner_messages.append(thought_event)

        # Add to context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=reflection_result.content,
                source=agent_name,
                thought=getattr(reflection_result, "thought", None),
            )
        )

        yield Response(
            chat_message=TextMessage(
                content=reflection_result.content,
                source=agent_name,
                models_usage=reflection_result.usage,
            ),
            inner_messages=inner_messages,
        )

    async def save_state(self) -> Mapping[str, Any]:
        state = await super().save_state()
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            state["dependencies"] = [
                {"code": dep.code, "language": dep.language}
                for dep in self._code_executor.docker_installed_dependencies
            ]
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await super().load_state(state)
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            dependencies = [
                CodeBlock(code=dep["code"], language=dep["language"])
                for dep in state.get("dependencies", [])
            ]
            self._code_executor.docker_installed_dependencies = dependencies

    async def EXIT(self):
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            # upload results files to s3 when exit the execution
            self._code_executor.upload_results_files_to_s3(self.user_id, self.chat_id)
            # exit
            await self._code_executor.countdown(self._countdown_timer)

    @classmethod
    async def _call_llm(
        cls,
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        agent_name: str,
        cancellation_token: CancellationToken,
        output_content_type: type[BaseModel] | None,
    ) -> AsyncGenerator[Union[CreateResult, ModelClientStreamingChunkEvent], None]:
        """
        Perform a model inference and yield either streaming chunk events or the final CreateResult.
        """
        all_messages = await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=system_messages + all_messages)

        if model_client_stream:
            model_result: Optional[CreateResult] = None
            async for chunk in model_client.create_stream(
                llm_messages, tools=[], cancellation_token=cancellation_token, json_output=output_content_type
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
            yield model_result
        else:
            model_result = await model_client.create(llm_messages, tools=[], cancellation_token=cancellation_token, json_output=output_content_type)
            yield model_result